"""Prepaid, once-only forward input preparation; never serving authorization.

The issuer must first authorize the actual lifecycle role. This helper binds the
planned complete recipe to one shared reservation BEFORE extracting its snapshot.
Two immutable intent anchors detect missing local state; the wallet never refunds.
A loaded intent is read-only. A partially prepared job cannot be retried with a new
lease. Final model publication still requires its separate publication witness.
"""
from __future__ import annotations

import copy
import json
import os
import platform
import sys
from importlib import metadata
from pathlib import Path

from filelock import FileLock

from factorlab.protocol import fingerprint, project_path

from .evidence import file_hash
from .monthly_work_authority import DEFAULT_LIMITS, MonthlyWorkAuthority
from .neural_forward_ticket import _bind_authority, _LifecycleReader, _read
from .v7_lifecycle import stamp
from .v7_storage import immutable_record
from .work_lease import FrozenWorkLease

VERSION = 'v7-forward-preparation-v1'


def _execution():
    return {'python': platform.python_version(), 'executable': str(Path(sys.executable).resolve()),
            'packages': {name: metadata.version(name) for name in
                         ('numpy', 'pandas', 'scipy', 'scikit-learn', 'pyarrow', 'filelock', 'psutil')}}


def _recipe(recipe):
    # Imports are lazy: producers can import this helper without a cycle.
    from .known_linear_reference import FrozenLinearRecipe
    from .pool_forward import PoolForwardRecipe, pipeline_for
    if type(recipe) is FrozenLinearRecipe:
        from .known_linear_forward import pipeline_for as known_pipeline
        canonical = FrozenLinearRecipe.from_dict(recipe.to_dict())
        return canonical, known_pipeline(canonical), 1, 'known_reference', 'known_forward_tickets'
    if type(recipe) is PoolForwardRecipe:
        canonical = PoolForwardRecipe.from_dict(recipe.to_dict())
        return canonical, pipeline_for(canonical), canonical.solve_limit, 'pool_forward', 'pool_forward_tickets'
    raise TypeError('Preparation requires a complete frozen known or pool recipe')


def _write(path, value):
    return immutable_record(path, {'value': value, 'sha256': fingerprint(value)})


def _record(path):
    raw = _read(path)
    if set(raw) != {'value', 'sha256'} or fingerprint(raw['value']) != raw['sha256']:
        raise ValueError('Preparation record checksum changed')
    return raw['value']


class ForwardPreparation:
    """A preparation is a spend binding, not role, training or serving permission."""

    @classmethod
    def begin(cls, runtime, authority, job, recipe, *, lease=None):
        from .v7_forward import ForwardRuntime
        if type(runtime) is not ForwardRuntime or type(authority) is not MonthlyWorkAuthority:
            raise TypeError('Actual forward runtime and shared authority required')
        # The original clock includes identity validation and all preparation I/O.
        original = FrozenWorkLease.issue(300) if lease is None else lease
        if type(original) is not FrozenWorkLease or os.getpid() != original.parent_pid:
            raise ValueError('Preparation must retain its original issuer lease')
        original.assert_parent_alive()
        recipe, pipeline, solves, purpose, ticket_directory = _recipe(recipe)
        from .nominal_actual_trial import validate_preparation
        validate_preparation(runtime, job, pipeline)
        runtime.lifecycle.verify()
        row = runtime.lifecycle.db.execute(
            'SELECT cutoff,stream,pipeline,status FROM jobs WHERE id=?', (job,)).fetchone()
        if row is None or row[3] != 'planned' or json.loads(row[2]) != pipeline:
            raise ValueError('Preparation requires an untouched planned full-recipe job')
        cutoff = runtime.lifecycle.fit_cutoff(row[0])
        if (cutoff > runtime.lifecycle.now()
                or recipe.rolling_spec != runtime.spec):
            raise ValueError('Preparation cutoff or rolling specification changed')
        _bind_authority(runtime, authority)
        work = fingerprint({'plan': runtime.lifecycle.plan.digest, 'job': job})
        primary = authority.directory / 'forward_preparations' / work
        secondary = runtime.directory / 'forward_preparations' / work
        primary.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(primary) + '.lock', timeout=0):
            if primary.exists() or secondary.exists():
                raise ValueError('Preparation already attempted; no replacement lease or retry')
            header = {'version': VERSION, 'work': work, 'runtime_directory': str(runtime.directory),
                      'forward_hash': file_hash(runtime.directory / 'forward.json'),
                      'runtime_identity': copy.deepcopy(runtime.identity), 'execution': _execution(),
                      'authority_directory': str(authority.directory),
                      'authority_manifest_hash': authority.manifest_hash,
                      'plan_id': runtime.lifecycle.plan.digest, 'job': job,
                      'cutoff': cutoff.isoformat(), 'stream': row[1], 'pipeline': pipeline,
                      'recipe': recipe.to_dict(), 'purpose': purpose, 'ticket_directory': ticket_directory,
                      'feed': runtime.feed.identity, 'feed_directory': str(runtime.feed.directory),
                      'counts': {k: solves if k == 'solve' else 0 for k in DEFAULT_LIMITS},
                      'lease': original.to_dict(), 'lease_hash': original.digest,
                      'source_hash': fingerprint(authority.manifest['source_files'])}
            obj = cls.__new__(cls)
            obj.authority, obj._runtime, obj._fresh = authority, runtime, True
            header = json.loads(json.dumps(header, allow_nan=False))
            obj._header, obj._digest = header, fingerprint(header)
            obj._primary, obj._secondary = primary, secondary
            obj._verify_environment()
            obj._dual_write('intent.json', header)
            record = authority._reserve_with_lease('forward_preparation:' + work,
                counts=header['counts'], binding=obj._binding(), lease=original)
            try:
                obj._dual_write('reservation.json', {'reservation_id': record['reservation_id'],
                                                    'terms_hash': record['terms_hash']})
                obj._live()
            except BaseException as exc:
                obj._fail(exc, reservation_id=record['reservation_id'])
                raise
            return obj

    @classmethod
    def load(cls, authority, reference):
        """Historical read-only open; never rebuild absent anchors or issue time."""
        if type(authority) is not MonthlyWorkAuthority or set(reference) != {'path', 'digest'}:
            raise ValueError('Canonical authority and preparation reference required')
        path = project_path(reference['path'])
        header = _record(path)
        work = fingerprint({'plan': header['plan_id'], 'job': header['job']})
        expected = authority.directory / 'forward_preparations' / work / 'intent.json'
        if (path != expected or header['version'] != VERSION or header['work'] != work
                or fingerprint(header) != reference['digest']):
            raise ValueError('Preparation reference changed or moved')
        obj = cls.__new__(cls)
        obj.authority, obj._runtime, obj._fresh = authority, None, False
        obj._header, obj._digest = header, reference['digest']
        obj._primary = path.parent
        obj._secondary = project_path(header['runtime_directory']) / 'forward_preparations' / work
        obj._verify_environment()
        obj._read_dual('intent.json')
        _ = obj.record
        return obj

    @property
    def reference(self):
        return {'path': str(self._primary / 'intent.json'), 'digest': self._digest}

    def _binding(self):
        h = self._header
        return {'prepare_terms': self._digest, 'job': h['job'],
                'pipeline_id': h['pipeline']['pipeline_id'], 'purpose': h['purpose']}

    def _dual_write(self, name, value):
        _write(self._primary / name, value)
        _write(self._secondary / name, value)

    def _read_dual(self, name):
        left, right = _record(self._primary / name), _record(self._secondary / name)
        if left != right or (name == 'intent.json' and fingerprint(left) != self._digest):
            raise ValueError('Preparation dual anchors differ')
        return left

    def _verify_environment(self):
        h = self._header
        if fingerprint(h) != self._digest:
            raise ValueError('Preparation header was mutated')
        self.authority.verify()
        root = project_path(h['runtime_directory'])
        manifest = _read(root / 'forward.json')
        if (h['authority_directory'] != str(self.authority.directory)
                or h['authority_manifest_hash'] != self.authority.manifest_hash
                or h['execution'] != _execution()
                or file_hash(root / 'forward.json') != h['forward_hash']
                or manifest != {'identity': h['runtime_identity'], 'sha256': fingerprint(h['runtime_identity'])}
                or h['runtime_identity']['source_files'] != self.authority.manifest['source_files']
                or h['source_hash'] != fingerprint(self.authority.manifest['source_files'])
                or h['feed_directory'] != h['runtime_identity']['feed_directory']):
            raise ValueError('Preparation runtime, source or shared authority changed')
        for relative, digest in self.authority.manifest['source_files'].items():
            if file_hash(root / 'source' / relative) != digest:
                raise ValueError('Preparation runtime source archive changed')
        reader = _LifecycleReader(root)
        try:
            verified = reader.verify()
            from .neural_forward_ticket import _verify_authority_binding
            _verify_authority_binding(reader, self.authority, verified['operations'])
            row = reader.db.execute('SELECT cutoff,stream,pipeline FROM jobs WHERE id=?', (h['job'],)).fetchone()
            if (reader.plan.digest != h['plan_id'] or row is None
                    or stamp(row[0]).isoformat() != h['cutoff'] or row[1] != h['stream']
                    or json.loads(row[2]) != h['pipeline']
                    or reader.plan.source_id != h['feed']):
                raise ValueError('Preparation actual lifecycle job or feed changed')
        finally:
            reader.close()

    @property
    def record(self):
        self._read_dual('intent.json')
        link = self._read_dual('reservation.json')
        record = self.authority.get(link['reservation_id'])
        h = self._header
        if (record['terms_hash'] != link['terms_hash'] or record['binding'] != self._binding()
                or record['work_key'] != 'forward_preparation:' + h['work']
                or record['counts'] != h['counts'] or record['lease'] != h['lease']
                or record['lease_hash'] != h['lease_hash']
                or record['status'] in {'failed_consumed', 'interrupted_consumed'}
                or any((p / 'failure.json').exists() for p in (self._primary, self._secondary))):
            raise ValueError('Preparation reservation changed, failed or consumed')
        return record

    def _live(self):
        if not self._fresh or self._runtime is None:
            raise ValueError('Loaded preparation is read-only; no retry')
        self._verify_environment()
        record = self.record
        if record['status'] != 'reserved':
            raise ValueError('Preparation must retain its reserved original work')
        lease = FrozenWorkLease.from_dict(record['lease'])
        if os.getpid() != lease.parent_pid:
            raise ValueError('Only the original preparation issuer can execute')
        lease.assert_parent_alive()
        return record

    def _fail(self, exc, *, reservation_id=None):
        # Preserve the initiating exception even when storage also fails. Missing
        # anchors remain closed; neither load nor begin repairs execution state.
        try:
            rid = reservation_id or self._read_dual('reservation.json')['reservation_id']
            self.authority.fail_consumed(rid, 'forward_preparation:' + type(exc).__name__)
        except BaseException as secondary:  # noqa: BLE001 -- preserve the initiating failure
            exc.add_note('Preparation accounting failure: ' + type(secondary).__name__)
        try:
            self._dual_write('failure.json', {'error_type': type(exc).__name__, 'prepare_terms': self._digest})
        except BaseException as secondary:  # noqa: BLE001 -- preserve the initiating failure
            exc.add_note('Preparation failure anchor unavailable: ' + type(secondary).__name__)

    def snapshot(self):
        """Fixed actual-feed producer; no injected callback or replacement input."""
        if not self._fresh:
            raise ValueError('Loaded preparation is read-only; no retry')
        with FileLock(str(self._primary) + '.lock', timeout=0):
            if any((p / 'snapshot_started.json').exists() for p in (self._primary, self._secondary)):
                raise ValueError('Snapshot preparation already attempted; no retry')
            try:
                self._live()
                from .nominal_prefix_session import preparation_guard
                with preparation_guard(self):
                    self._dual_write('snapshot_started.json', {'prepare_terms': self._digest})
                    prepared = self._runtime.prepare_fit_snapshot(self._header['job'])
                    from .forward_fit_snapshot import PreparedForwardSnapshot
                    if type(prepared) is not PreparedForwardSnapshot:
                        raise TypeError('Snapshot producer did not return its actual sealed input')
                    prepared.verify()
                    path = self._runtime.directory / 'snapshots' / (self._header['job'] + '.json')
                    if _read(path) != prepared.record:
                        raise ValueError('Snapshot bytes differ from actual prepared input')
                    self._check_snapshot(prepared.record)
                    ref = {'job': self._header['job'], 'path': str(path), 'sha256': file_hash(path),
                           'snapshot_id': prepared.record['snapshot_id']}
                    self._dual_write('snapshot.json', {'prepare_terms': self._digest, 'snapshot_ref': ref})
                    self._live()
                    return prepared
            except BaseException as exc:
                self._fail(exc)
                raise

    def _check_snapshot(self, value):
        h = self._header
        if any(value.get(k) != h[k] for k in ('job', 'plan_id', 'cutoff', 'feed', 'feed_directory', 'source_hash')):
            raise ValueError('Prepared snapshot changed its frozen input scope')
        from .neural_forward_ticket import _LifecycleReader
        from .quarantined_observation import job_binding
        reader = _LifecycleReader(h['runtime_directory'])
        try:
            prefix = job_binding(reader, h['job'])
        finally:
            reader.close()
        availability = 'actual_receipts' if prefix is None else 'quarantined_finite_observation'
        if (value.get('observation_prefix') != prefix
                or prefix is None and 'observation_prefix' in value
                or value.get('pipeline_id') != h['pipeline']['pipeline_id']
                or value.get('availability') != availability
                or value.get('snapshot_id') != fingerprint({k: v for k, v in value.items() if k != 'snapshot_id'})):
            raise ValueError('Prepared snapshot changed its complete recipe or identity')

    @property
    def snapshot_ref(self):
        _ = self.record
        claim = self._read_dual('snapshot_started.json')
        value = self._read_dual('snapshot.json')
        if claim != {'prepare_terms': self._digest} or value['prepare_terms'] != self._digest:
            raise ValueError('Snapshot seal lost original preparation')
        ref = value['snapshot_ref']
        expected = project_path(self._header['runtime_directory']) / 'snapshots' / (self._header['job'] + '.json')
        if (set(ref) != {'job', 'path', 'sha256', 'snapshot_id'} or project_path(ref['path']) != expected
                or ref['job'] != self._header['job'] or file_hash(expected) != ref['sha256']):
            raise ValueError('Preparation snapshot bytes or path changed')
        actual = _read(expected)
        self._check_snapshot(actual)
        if actual['snapshot_id'] != ref['snapshot_id']:
            raise ValueError('Preparation snapshot seal changed')
        return copy.deepcopy(ref)

    def _ticket(self, path, value):
        path = project_path(path)
        h, record = self._header, self.record
        expected = self.authority.directory / h['ticket_directory'] / h['work'] / 'ticket.json'
        if path != expected or _record(path) != value:
            raise ValueError('Final preparation ticket changed or moved')
        if (value.get('preparation_ref') != self.reference or value.get('snapshot_ref') != self.snapshot_ref
                or value.get('recipe') != h['recipe'] or value.get('source_hash') != h['source_hash']
                or value.get('authority_directory') != h['authority_directory']
                or any(value.get(k) != record[k] for k in
                       ('reservation_id', 'authority_manifest_hash', 'terms_hash', 'lease', 'lease_hash'))
                or any(value.get('context', {}).get(k) != h[k] for k in
                       ('job', 'plan_id', 'pipeline', 'cutoff', 'stream'))
                or value.get('context', {}).get('lifecycle_directory') != h['runtime_directory']):
            raise ValueError('Final ticket changed the original preparation binding')
        return {'path': str(path), 'sha256': file_hash(path), 'digest': fingerprint(value),
                'prepare_terms': self._digest}

    def seal_ticket(self, path, value):
        """Seal already-published producer ticket bytes once, retaining its lease."""
        if not self._fresh:
            raise ValueError('Loaded preparation is read-only; no retry')
        with FileLock(str(self._primary) + '.lock', timeout=0):
            if any((p / 'ticket.json').exists() for p in (self._primary, self._secondary)):
                raise ValueError('Final ticket already sealed; no new input seal')
            try:
                self._live()
                ref = self._ticket(path, value)
                self._dual_write('ticket.json', ref)
                self._live()
                return copy.deepcopy(ref)
            except BaseException as exc:
                self._fail(exc)
                raise

    @classmethod
    def verify_ticket_reference(cls, path, value):
        """Historical producer verifier: no runtime constructor or new stores."""
        from .neural_forward_ticket import open_existing_authority
        authority = open_existing_authority(value['authority_directory'])
        obj = cls.load(authority, value['preparation_ref'])
        return obj.verify_ticket(path, value)

    def verify_ticket(self, path, value):
        """Read-only chain check, not fitting or model publication permission."""
        self._verify_environment()
        actual = self._ticket(path, value)
        if self._read_dual('ticket.json') != actual:
            raise ValueError('Final ticket seal changed')
        return self.record




