"""Original finite-law observation prefixes, separate from physical work time.

This component does not change ReceiptArchive, a work lease, a monthly wallet,
or their physical deadlines. Logical time is an original DGP release, never an
input argument. Physical observations are local integrity evidence, not an
administrator-resistant signature or Linux resource certification.
"""
from __future__ import annotations

import math
import copy
from contextlib import contextmanager
from contextvars import ContextVar
import os
import time
from threading import get_ident
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import psutil
from filelock import FileLock

from factorlab.discovery.calibration_runtime_contract import _original_operations
from factorlab.discovery.calibration_trial_context import _no_redirect
from factorlab.discovery.evidence import file_hash
from factorlab.discovery.finite_market_oracle import (
    FiniteGenerator, _read as generator_read, _read_draw, _read_draw_sources,
    _verified_publication_sources, bind_draw, verify_publications,
)
from factorlab.discovery.full_calibration_registry import QuarantinedTrialPermit, _read
from factorlab.discovery.v7_forward import ForwardRuntime
from factorlab.discovery.v7_lifecycle import canonical
from factorlab.discovery.v7_storage import immutable_record
from factorlab.discovery.work_lease import _boot_id
from factorlab.protocol import fingerprint

VERSION = 'quarantined-observation-prefix-v2'
BOUND = 'quarantined_observation_bound'
ADVANCED = 'quarantined_observation_advanced'
_READ_SCOPE = ContextVar('original_observation_read_scope', default=None)


def _active_read_scope():
    scope = _READ_SCOPE.get()
    if scope is not None and (not scope.get('active', False) or scope.get('thread') != get_ident()
                              or scope.get('pid') != os.getpid()):
        raise ValueError('Original observation read scope has expired or belongs to another thread or process')
    return scope


@contextmanager
def _original_observation_read_scope(lifecycle):
    """Internal producer invocation only; original full checks at both ends.

    Holds the canonical journal lock, so observation cannot advance during
    the invocation. The cached reconstruction never survives this context;
    final original bytes/chain/source checks happen before the producer's
    final physical lease assertion and before it returns a usable result.
    """
    parent = _active_read_scope()
    if parent is not None:
        if parent['directory'] != str(lifecycle.directory):
            raise ValueError('Nested read cannot borrow another original lifecycle prefix')
        # Historical model admission can verify another ticket in this same
        # issuer call. It remains inside the outer journal lock and full
        # closing audit; no nested result escapes the outer producer early.
        yield
        return
    from .nominal_trial_controller import from_lifecycle, observation_read_scope
    if from_lifecycle(lifecycle) is not None:
        with observation_read_scope(lifecycle):
            yield
        return
    obj = QuarantinedObservation._open_original(lifecycle)
    if obj is None:
        yield
        return
    with FileLock(str(obj.directory)+'.lock', timeout=0):
        scope = {'object': obj, 'directory': str(lifecycle.directory), 'verified': None,
                 'active': True, 'thread': get_ident(), 'pid': os.getpid()}
        if obj.permit.trial.get('estimand_slot') is not None:
            # Verify the separate validation runtime before installing this
            # replica's scope. Its own paid model readbacks must not borrow or
            # collide with a different runtime's cached observation authority.
            from .fixed_recipe_oracle import _bound_identity
            scope['fixed_oracle_identity'] = _bound_identity(lifecycle)
        token = _READ_SCOPE.set(scope)
        try:
            yield
        finally:
            # Remove permission before the mandatory complete closing audit.
            scope['active'] = False
            _READ_SCOPE.reset(token)
            if scope['verified'] is not None:
                final = obj.inspect(lifecycle)
                if final != scope['verified']:
                    raise ValueError('Original source or observation prefix changed inside read invocation')
            if 'fixed_oracle_identity' in scope:
                from .fixed_recipe_oracle import _bound_identity
                if _bound_identity(lifecycle) != scope['fixed_oracle_identity']:
                    raise ValueError('Original fixed oracle selection/source changed inside read invocation')


def _physical():
    return {'wall': datetime.now(UTC).isoformat(), 'monotonic': time.monotonic(),
            'boot': _boot_id(), 'pid': os.getpid(),
            'process_created': psutil.Process().create_time()}


def _check_physical(value, previous=None):
    if (type(value) is not dict or set(value) != {'wall', 'monotonic', 'boot', 'pid', 'process_created'}
            or type(value['wall']) is not str or type(value['boot']) is not str
            or not value['boot'].startswith(('windows:', 'linux:'))
            or type(value['pid']) is not int or value['pid'] <= 0
            or any(type(value[k]) not in (int, float) or not math.isfinite(value[k]) or value[k] <= 0
                   for k in ('monotonic', 'process_created'))):
        raise ValueError('Original physical observation schema changed')
    wall = pd.Timestamp(value['wall'])
    if (wall.tzinfo is None or wall.utcoffset().total_seconds() != 0
            or wall > pd.Timestamp.now(tz='UTC') or value['process_created'] > wall.timestamp()):
        raise ValueError('Original physical UTC observation invalid')
    if previous and (wall < pd.Timestamp(previous['wall']) or
            value['boot'] == previous['boot'] and value['monotonic'] < previous['monotonic']):
        raise ValueError('Original physical clock moved backward')
    return wall


def _stat(path):
    value = path.stat()
    return {'path': str(path), 'sha256': file_hash(path), 'size': value.st_size,
            'mtime_ns': value.st_mtime_ns, 'ctime_ns': value.st_ctime_ns}


def _event(lifecycle, kind, detail, physical):
    # Do not call Lifecycle._event: it samples the caller-configurable LC clock.
    previous = lifecycle.db.execute('SELECT hash FROM operations ORDER BY seq DESC LIMIT 1').fetchone()
    seq = lifecycle.db.execute('SELECT COALESCE(MAX(seq),0)+1 FROM operations').fetchone()[0]
    record = {'seq': seq, 'previous': previous[0] if previous else '0'*64,
              'kind': kind, 'at': physical['wall'], 'detail': detail}
    lifecycle.db.execute('INSERT INTO operations VALUES (?,?,?)', (seq, canonical(record), fingerprint(record)))
    return record


class QuarantinedObservation:
    """Fail-closed local journal; positive completion can reopen without replay."""

    def __init__(self, generator, permit):
        if type(generator) is not FiniteGenerator or type(permit) is not QuarantinedTrialPermit:
            raise TypeError('Actual finite generator and original enrolled permit required')
        self.generator, self.permit = generator, permit
        self.directory = permit.paths['root']/'observation'
        self.mirror = generator.directory/'observation_clocks'/permit.trial['trial_id']
        for path in (self.directory, self.mirror):
            _no_redirect(path, path.parents[1])

    def _runtime(self, runtime):
        if type(runtime) is not ForwardRuntime or runtime.directory != self.permit.paths['runtime']:
            raise TypeError('Only the original actual runtime can bind observation')
        self.permit.verify(runtime.lifecycle)

    def _identity(self):
        self.permit.verify()
        draw, atom = _read_draw_sources(self.generator, self.permit)
        return self._identity_for_sources(draw, atom)

    def _identity_for_sources(self, draw, atom):
        """Pure descriptor for freshly checked sources, never an admission grant."""
        return {'version': VERSION, 'registry_digest': self.permit.registry.digest,
                'trial': self.permit.trial, 'generator_digest': self.generator.digest,
                'generator_directory': str(self.generator.directory),
                'draw_hash': fingerprint(draw), 'atom_hash': fingerprint(atom),
                'paths': {k: str(v) for k, v in self.permit.paths.items()},
                'mapping': 'original_atom_schedule_release_not_before_in_hour_order',
                'physical_policy': 'completed_actual_publication_required_no_clock_override',
                'source_files': self.permit.registry.identity['source_files'],
                'implementation_hash': file_hash(Path(__file__)),
                'execution_scope': 'quarantined_simulation_only',
                'formal_gate_eligible': False, 'work_budget_granted': False}

    @classmethod
    def bind(cls, runtime, generator, permit):
        obj = cls(generator, permit)
        obj._runtime(runtime)
        with runtime.lifecycle.controller_lock(), FileLock(str(obj.directory)+'.lock', timeout=0):
            bind_draw(generator, permit)
            operations = _original_operations(runtime.lifecycle)
            if any(r['kind'] == BOUND for r in operations):
                obj.inspect(runtime.lifecycle)
                return obj
            if obj.directory.exists() or obj.mirror.exists():
                raise ValueError('Uncommitted observation binding cannot be reset')
            allowed = {'forward_runtime_bound', 'monthly_work_authority_bound', 'full_calibration_quarantine_enrolled'}
            if permit.trial.get('estimand_slot') is not None:
                from .fixed_recipe_oracle import _bound_identity
                _bound_identity(runtime.lifecycle, permit)
                allowed.add('fixed_oracle_controller_bound')
            if any(r['kind'] not in allowed for r in operations):
                raise ValueError('Observation must bind before research, jobs, claims or outcomes')
            if runtime.feed.db.execute('SELECT COUNT(*) FROM batches').fetchone()[0]:
                raise ValueError('Observation cannot retrospectively bind an exposed feed')
            from factorlab.discovery.neural_forward_ticket import open_existing_authority
            if open_existing_authority(permit.paths['authority']).verify()['reservations']:
                raise ValueError('Observation cannot bind after charged work')
            identity = obj._identity()
            for root in (obj.directory, obj.mirror):
                immutable_record(root/'identity.json', identity)
                # JSON string intentionally; immutable_record serializes JSON.
                immutable_record(root/'implementation.json', Path(__file__).read_text('utf-8'))
            physical = _physical()
            _check_physical(physical)
            detail = {'identity_hash': fingerprint(identity), 'directory': str(obj.directory),
                      'physical_observation': physical}
            with runtime.lifecycle.db:
                runtime.lifecycle.db.execute('BEGIN IMMEDIATE')
                operation = _event(runtime.lifecycle, BOUND, detail, physical)
            for root in (obj.directory, obj.mirror):
                immutable_record(root/'bound.json', operation)
            obj.inspect(runtime.lifecycle)
            return obj

    def _source_event(self, hour, publications, *, draw=None, atom=None, original_hashes=None):
        if draw is None:
            draw = _read_draw(self.generator, self.permit)
            atom = self.generator.read_atom(draw['atom'])
        if type(hour) is not int or not 0 <= hour < len(atom['schedule']):
            raise ValueError('Original finite observation schedule exhausted')
        if original_hashes is None:
            original_hashes = (fingerprint(draw), fingerprint(atom))
        scheduled = atom['schedule'][hour]
        receipt, files = None, []
        if not scheduled['gap']:
            if hour not in publications['completed_hours']:
                raise ValueError('Next observation has no completed actual receipt')
            roots = (self.permit.paths['root']/'generator_publications',
                     self.generator.directory/'publications'/self.permit.trial['trial_id'])
            paths = [root/f'{hour:05d}-complete.json' for root in roots]
            receipt = generator_read(paths[0])
            files = [_stat(p) for p in paths]
        return {'hour': hour, 'draw_hash': original_hashes[0], 'atom_hash': original_hashes[1],
                'logical_at': scheduled['release_not_before'], 'scheduled_gap': scheduled['gap'],
                'receipt_hash': None if receipt is None else fingerprint(receipt),
                'physical_received_at': None if receipt is None else receipt['received_at'],
                'source_publications': files}

    def inspect(self, lifecycle=None):
        if lifecycle is None:
            with self.permit.open_lifecycle() as reader:
                return self.inspect(reader)
        scope = _active_read_scope()
        scoped = (scope is not None and scope['object'] is self
                  and scope['directory'] == str(lifecycle.directory))
        if scoped and scope['verified'] is not None:
            return copy.deepcopy(scope['verified'])
        result, _, _, _ = self._inspect_original(lifecycle)
        if scoped:
            scope['verified'] = copy.deepcopy(result)
        return result

    def _inspect_original(self, lifecycle):
        """Full original audit plus invocation-local sources, with no cache hit."""
        _active_read_scope()
        self.permit.verify(lifecycle)
        # Full source/publication audit first; reuse its checked values only
        # inside this invocation. The original closing audit runs afresh.
        publications, draw, atom = _verified_publication_sources(self.generator, self.permit)
        identity = self._identity_for_sources(draw, atom)
        for root in (self.directory, self.mirror):
            _no_redirect(root, root.parents[1])
            if (_read(root/'identity.json') != identity
                    or _read(root/'implementation.json') != Path(__file__).read_text('utf-8')):
                raise ValueError('Original observation identity or component changed')
        operations = _original_operations(lifecycle)
        bound = [r for r in operations if r['kind'] == BOUND]
        if len(bound) != 1 or any(_read(p/'bound.json') != bound[0] for p in (self.directory, self.mirror)):
            raise ValueError('Original observation lifecycle binding missing or changed')
        expected = {'identity_hash': fingerprint(identity), 'directory': str(self.directory),
                    'physical_observation': bound[0]['detail'].get('physical_observation')}
        if bound[0]['detail'] != expected or bound[0]['at'] != expected['physical_observation']['wall']:
            raise ValueError('Original observation lifecycle binding changed')
        previous = expected['physical_observation']
        _check_physical(previous)
        rows = [r for r in operations if r['kind'] == ADVANCED]
        names = {f'{i:05d}.json' for i in range(len(rows))}
        for root in (self.directory, self.mirror):
            for folder in ('events', 'commits'):
                if {p.name for p in (root/folder).glob('*.json')} != names:
                    raise ValueError('Orphan or missing observation event cannot be replayed')
        # Same original immutable bytes, hashed once per complete verification.
        # This local tuple is never retained as a permission across calls.
        original_hashes = (fingerprint(draw), fingerprint(atom))
        logical_at = None
        prior_head = fingerprint(bound[0])
        for hour, operation in enumerate(rows):
            name = f'{hour:05d}.json'
            value = _read(self.directory/'events'/name)
            source = self._source_event(hour, publications, draw=draw, atom=atom,
                                        original_hashes=original_hashes)
            if (value != {'source': source, 'previous_prefix': prior_head}
                    or any(_read(p/'events'/name) != value for p in (self.directory, self.mirror))
                    or any(_read(p/'commits'/name) != operation for p in (self.directory, self.mirror))
                    or operation['seq'] <= bound[0]['seq']):
                raise ValueError('Original observation source, prefix or physical witness changed')
            detail = operation['detail']
            physical = detail.get('physical_observation')
            wall = _check_physical(physical, previous)
            files = [_stat(p/'events'/name) for p in (self.directory, self.mirror)]
            expected_detail = {'identity_hash': fingerprint(identity), 'event_hash': fingerprint(value),
                               'event_files': files, 'physical_observation': physical}
            if (detail != expected_detail or operation['at'] != physical['wall']
                    or any(max(p['mtime_ns'], p['ctime_ns']) > wall.value for p in files+source['source_publications'])
                    or source['physical_received_at'] is not None and wall < pd.Timestamp(source['physical_received_at'])
                    or wall < pd.Timestamp(source['logical_at'])):
                raise ValueError('Observation lacks original physical publication order')
            previous, logical_at, prior_head = physical, source['logical_at'], fingerprint(operation)
        latest = None if not rows else self._binding_value(
            fingerprint(identity), len(rows)-1, rows[-1], logical_at)
        result = {'identity_hash': fingerprint(identity), 'events': len(rows), 'logical_at': logical_at,
                  'head': prior_head, 'physical_observation': previous,
                  'latest_binding': latest,
                  'formal_gate_eligible': False, 'work_budget_granted': False}
        return result, publications, draw, atom

    def advance(self, runtime):
        """Consume exactly the next original release; accepts no time argument."""
        self._runtime(runtime)
        with runtime.lifecycle.controller_lock(), FileLock(str(self.directory)+'.lock', timeout=0):
            before = self.inspect(runtime.lifecycle)
            source = self._source_event(before['events'], verify_publications(self.generator, self.permit))
            physical = _physical()
            wall = _check_physical(physical, before['physical_observation'])
            if (pd.Timestamp(source['logical_at']) > wall
                    or source['physical_received_at'] is not None and pd.Timestamp(source['physical_received_at']) > wall
                    or any(max(p['mtime_ns'], p['ctime_ns']) > wall.value for p in source['source_publications'])):
                raise ValueError('Original release or physical publication is not yet available')
            name = f'{before["events"]:05d}.json'
            value = {'source': source, 'previous_prefix': before['head']}
            for root in (self.directory, self.mirror):
                immutable_record(root/'events'/name, value)
            files = [_stat(p/'events'/name) for p in (self.directory, self.mirror)]
            physical = _physical()
            wall = _check_physical(physical, before['physical_observation'])
            if any(max(p['mtime_ns'], p['ctime_ns']) > wall.value for p in files):
                raise ValueError('Physical observation precedes durable event write')
            with runtime.lifecycle.db:
                runtime.lifecycle.db.execute('BEGIN IMMEDIATE')
                operation = _event(runtime.lifecycle, ADVANCED,
                    {'identity_hash': before['identity_hash'], 'event_hash': fingerprint(value),
                     'event_files': files, 'physical_observation': physical}, physical)
            # An interrupted SQL or mirror commit is retained and fails closed.
            for root in (self.directory, self.mirror):
                immutable_record(root/'commits'/name, operation)
            return self.binding(before['events'], runtime.lifecycle)

    def binding(self, hour, lifecycle=None):
        if lifecycle is None:
            with self.permit.open_lifecycle() as reader:
                return self.binding(hour, reader)
        state = self.inspect(lifecycle)
        if type(hour) is not int or not 0 <= hour < state['events']:
            raise ValueError('Only an original committed observation prefix can be bound')
        op = [r for r in _original_operations(lifecycle) if r['kind'] == ADVANCED][hour]
        return self._binding_value(state['identity_hash'], hour, op,
            _read(self.directory/'events'/f'{hour:05d}.json')['source']['logical_at'])

    def _binding_value(self, identity_hash, hour, op, logical_at):
        return {'version': VERSION, 'registry_digest': self.permit.registry.digest,
                'trial_id': self.permit.trial['trial_id'], 'identity_hash': identity_hash,
                'hour': hour, 'logical_at': logical_at,
                'event_hash': op['detail']['event_hash'], 'operation_hash': fingerprint(op),
                'physical_observation': op['detail']['physical_observation'],
                'availability': 'quarantined_observation_prefix_not_actual_receipts',
                'formal_gate_eligible': False, 'work_budget_granted': False}

    def advance_batch(self, runtime, *, max_events=96):
        """Bounded contiguous original releases; no clock input or reused grant.

        The original SQL, source, draw and publications are fully checked at
        both ends. Each event still has its own durable files, physical sample,
        SQL commit and dual commit anchors. A partial failure is retained.
        """
        if type(max_events) is not int or not 1 <= max_events <= 96:
            raise ValueError('At most 96 original releases per bounded observation batch')
        self._runtime(runtime)
        with runtime.lifecycle.controller_lock(), FileLock(str(self.directory)+'.lock', timeout=0):
            # Reuse only values captured by this fresh complete opening audit.
            # Closing inspection below independently verifies all original bytes.
            before, publications, draw, atom = self._inspect_original(runtime.lifecycle)
            original_hashes = (fingerprint(draw), fingerprint(atom))
            limit = min(max_events, len(atom['schedule'])-before['events'])
            if not limit:
                raise ValueError('Original finite observation schedule exhausted')
            previous, head = before['physical_observation'], before['head']
            for hour in range(before['events'], before['events']+limit):
                source = self._source_event(hour, publications, draw=draw, atom=atom,
                                        original_hashes=original_hashes)
                physical = _physical()
                wall = _check_physical(physical, previous)
                if (pd.Timestamp(source['logical_at']) > wall
                        or source['physical_received_at'] is not None and pd.Timestamp(source['physical_received_at']) > wall
                        or any(max(p['mtime_ns'], p['ctime_ns']) > wall.value for p in source['source_publications'])):
                    raise ValueError('Original release or physical publication is not yet available')
                name = f'{hour:05d}.json'
                value = {'source': source, 'previous_prefix': head}
                for root in (self.directory, self.mirror):
                    immutable_record(root/'events'/name, value)
                files = [_stat(p/'events'/name) for p in (self.directory, self.mirror)]
                physical = _physical()
                wall = _check_physical(physical, previous)
                if any(max(p['mtime_ns'], p['ctime_ns']) > wall.value for p in files):
                    raise ValueError('Physical observation precedes durable event write')
                with runtime.lifecycle.db:
                    runtime.lifecycle.db.execute('BEGIN IMMEDIATE')
                    operation = _event(runtime.lifecycle, ADVANCED,
                        {'identity_hash': before['identity_hash'], 'event_hash': fingerprint(value),
                         'event_files': files, 'physical_observation': physical}, physical)
                for root in (self.directory, self.mirror):
                    immutable_record(root/'commits'/name, operation)
                previous, head = physical, fingerprint(operation)
            after = self.inspect(runtime.lifecycle)
            if after['events'] != before['events']+limit or after['head'] != head:
                raise ValueError('Original observation batch changed at publication boundary')
            return {'first_hour': before['events'], 'last_hour': after['events']-1,
                    'events_committed': limit, 'event_files_written': 2*limit,
                    'commit_files_written': 2*limit, 'lifecycle_commits': limit,
                    'formal_gate_eligible': False, 'work_budget_granted': False}

    def verify_binding(self, value):
        if type(value) is not dict or value != self.binding(value.get('hour')):
            raise ValueError('Observation prefix belongs to another trial or original event')
        return value

    @classmethod
    def from_lifecycle(cls, lifecycle):
        """Read original LC enrollment and original dual bound identity only."""
        obj = cls._open_original(lifecycle)
        if obj is not None:
            obj.inspect(lifecycle)
        return obj

    @classmethod
    def _open_original(cls, lifecycle):
        """Private reconstruction; trusted read invocation may reuse its object."""
        from .nominal_trial_controller import from_lifecycle
        nominal = from_lifecycle(lifecycle)
        if nominal is not None:
            return nominal
        scope = _active_read_scope()
        if scope is not None and scope['directory'] == str(lifecycle.directory):
            return scope['object']
        rows = [o for o in _original_operations(lifecycle) if o['kind'] == BOUND]
        if not rows:
            return None
        if len(rows) != 1:
            raise ValueError('Ambiguous original observation binding')
        from factorlab.discovery.quarantine_execution import permit_for
        from factorlab.discovery.finite_market_oracle import FiniteMarketLaw
        permit = permit_for(lifecycle)
        if permit is None:
            raise ValueError('Observation cannot authorize a non-enrolled runtime')
        identity = _read(permit.paths['root']/'observation/identity.json')
        if fingerprint(identity) != rows[0]['detail']['identity_hash']:
            raise ValueError('Observation identity differs from original lifecycle')
        gen = FiniteGenerator.__new__(FiniteGenerator)
        gen.directory = Path(identity['generator_directory'])
        _no_redirect(gen.directory, permit.registry.directory.parent)
        gen.identity = generator_read(gen.directory/'identity.json')
        law = dict(gen.identity['law']); law['symbols'] = tuple(law['symbols'])
        gen.law, gen.acceptance = FiniteMarketLaw(**law), permit.registry.plan.acceptance
        gen.verify_registry(permit.registry)
        obj = cls(gen, permit)
        return obj


def current_binding(lifecycle):
    from .nominal_trial_controller import from_lifecycle, current_binding as nominal_binding
    if from_lifecycle(lifecycle) is not None:
        return nominal_binding(lifecycle)
    obj = QuarantinedObservation._open_original(lifecycle)
    if obj is None:
        return None
    state = obj.inspect(lifecycle)
    if not state['events']:
        raise ValueError('Original observation has no committed logical release')
    return state['latest_binding']


def logical_now(lifecycle):
    value = current_binding(lifecycle)
    return None if value is None else pd.Timestamp(value['logical_at'])


def job_binding(lifecycle, job):
    from .nominal_trial_controller import from_lifecycle, job_binding as nominal_binding
    if from_lifecycle(lifecycle) is not None:
        return nominal_binding(lifecycle, job)
    obj = QuarantinedObservation._open_original(lifecycle)
    matches = [o for o in _original_operations(lifecycle)
               if o['kind'] == 'fit_planned' and o['detail']['job'] == job]
    if len(matches) != 1:
        raise ValueError('Job lacks its original planning event')
    prefix = matches[0]['detail'].get('observation_prefix')
    if obj is None:
        if prefix is not None:
            raise ValueError('Non-enrolled job cannot borrow simulation provenance')
        return None
    obj.verify_binding(prefix)
    if (pd.Timestamp(matches[0]['detail']['cutoff']) > pd.Timestamp(prefix['logical_at'])
            or pd.Timestamp(matches[0]['at']) != pd.Timestamp(prefix['logical_at'])):
        raise ValueError('Job observation differs from its original logical planning time')
    return prefix


def snapshot_for_prefix(lifecycle, prefix, cutoff, *, months=3, context_hours=1234):
    """Reconstruct numeric inputs from original released law, retaining UTC IO."""
    from .nominal_prefix_session import active_for_lifecycle
    session = active_for_lifecycle(lifecycle)
    if session is not None and session.has_read_epoch():
        return session.snapshot_in_epoch(lifecycle, prefix, cutoff, months=months, context_hours=context_hours)
    from factorlab.time_utils import HOUR, month_start
    from factorlab.discovery.v7_feed import SnapshotUnavailable
    obj = QuarantinedObservation._open_original(lifecycle)
    if obj is None:
        raise ValueError('Quarantined snapshot requires original enrolled observation')
    obj.verify_binding(prefix)
    end = pd.Timestamp(cutoff)
    if end.tzinfo is None or end > pd.Timestamp(prefix['logical_at']):
        raise ValueError('Snapshot cutoff exceeds its captured original prefix')
    start = ((end-pd.Timedelta(days=30*months)) if lifecycle.plan.training_window_days is not None
             else month_start(end)-pd.DateOffset(months=months))-context_hours*HOUR
    logical_releases = {}
    for hour in range(prefix['hour']+1):
        from .nominal_trial_controller import NominalObservation
        event = (obj.source_event(hour) if type(obj) is NominalObservation
                 else _read(obj.directory/'events'/f'{hour:05d}.json')['source'])
        if (event['scheduled_gap'] or pd.Timestamp(event['logical_at']) > end
                or pd.Timestamp(obj.generator.law.start)+hour*HOUR < start):
            continue
        receipt = generator_read(obj.permit.paths['root']/'generator_publications'/f'{hour:05d}-complete.json')
        logical_releases[receipt['receipt']['batch']] = pd.Timestamp(event['logical_at'])
    if not logical_releases:
        raise SnapshotUnavailable('No original observed law rows before logical cutoff')
    # Read actual feed bytes at the captured physical boundary, then restrict
    # strictly to this logical prefix. Preserve original imported_at, receipts,
    # revision hashes and batch IDs; only available_at has simulation meaning.
    import sqlite3
    from factorlab.discovery.v7_feed import ReceiptArchive
    feed = ReceiptArchive.__new__(ReceiptArchive)
    feed.directory, feed.symbols = obj.permit.paths['feed'], obj.generator.law.symbols
    feed.identity = lifecycle.plan.source_id
    feed.clock = lambda: pd.Timestamp.now(tz='UTC')
    feed.db = sqlite3.connect((feed.directory/'receipts.sqlite').as_uri()+'?mode=ro', uri=True)
    try:
        feed.verify()
        physical = feed.snapshot(prefix['physical_observation']['wall'], start=start)
    finally:
        feed.db.close()
    frame = physical[physical.batch_id.isin(logical_releases)].copy().reset_index(drop=True)
    if frame.empty or set(frame.batch_id) != set(logical_releases):
        raise ValueError('Original prefix lacks its physically published rows')
    frame['available_at'] = pd.to_datetime(frame.batch_id.map(logical_releases), utc=True)
    frame['availability_basis'] = 'quarantined_finite_observation'
    frame.attrs = {'availability_mode': 'quarantined_finite_observation',
                   'feed_identity': lifecycle.plan.source_id, 'snapshot_asof': end.isoformat(),
                   'lifecycle_directory': str(lifecycle.directory), 'observation_prefix': prefix,
                   'months': months, 'context_hours': context_hours}
    return frame


def verify_frame(frame):
    """A provenance string or a caller-created DataFrame cannot confer scope."""
    from factorlab.discovery.frame_identity import frame_identity
    from factorlab.discovery.neural_forward_ticket import _LifecycleReader
    attrs = frame.attrs
    from .nominal_prefix_session import active_for_frame
    session = active_for_frame(frame)
    if session is not None and session.has_read_epoch():
        return session.validate_in_epoch(frame)
    if set(attrs) != {'availability_mode', 'feed_identity', 'snapshot_asof', 'lifecycle_directory',
                      'observation_prefix', 'months', 'context_hours'}:
        raise ValueError('Quarantined frame lacks exact original reconstruction identity')
    reader = _LifecycleReader(attrs['lifecycle_directory'])
    try:
        original = snapshot_for_prefix(reader, attrs['observation_prefix'], attrs['snapshot_asof'],
                                       months=attrs['months'], context_hours=attrs['context_hours'])
        if frame_identity(original) != frame_identity(frame) or original.attrs != attrs:
            raise ValueError('Quarantined frame differs from its original observed law')
    finally:
        reader.close()
    return attrs['observation_prefix']


def _publication_paths(lifecycle, kind, key):
    if kind == 'confirmation':
        return [lifecycle.directory/'claim_publications'/f'{fingerprint(key)}.json']
    if kind == 'model':
        row = lifecycle.db.execute('SELECT model FROM jobs WHERE id=?', (key,)).fetchone()
        if row is None or not row[0]:
            raise ValueError('Model publication lacks its verified original job')
        folder = lifecycle.directory/'models'/row[0]
        return [folder/name for name in ('model.json', 'manifest.json', 'registration_completed.json')]
    raise ValueError('Unknown quarantined publication kind')


def seal_publication(lifecycle, kind, key):
    """Only after original admission and durable LC completion; never repair."""
    obj = QuarantinedObservation._open_original(lifecycle)
    if obj is None:
        return None
    name = fingerprint({'kind': kind, 'key': key})+'.json'
    roots = (obj.directory/'publications', obj.mirror/'publications')
    found = [o for o in _original_operations(lifecycle) if o['kind'] == 'quarantined_publication'
             and o['detail']['kind'] == kind and o['detail']['key'] == key]
    if found or any((root/name).exists() for root in roots):
        return read_publication(lifecycle, kind, key)
    prefix = current_binding(lifecycle)
    files = [_stat(p) for p in _publication_paths(lifecycle, kind, key)]
    physical = _physical()
    wall = _check_physical(physical, prefix['physical_observation'])
    if any(max(p['mtime_ns'], p['ctime_ns']) > wall.value for p in files):
        raise ValueError('Publication observer precedes original durable files')
    detail = {'kind': kind, 'key': key, 'observation_prefix': prefix,
              'files': files, 'physical_observation': physical}
    with lifecycle.db:
        lifecycle.db.execute('BEGIN IMMEDIATE')
        operation = _event(lifecycle, 'quarantined_publication', detail, physical)
    for root in roots:
        immutable_record(root/name, operation)
    return read_publication(lifecycle, kind, key)


def read_publication(lifecycle, kind, key):
    return _read_publication(lifecycle, kind, key)


def _read_publication(lifecycle, kind, key, *, isolated_copy=None):
    obj = QuarantinedObservation._open_original(lifecycle)
    if obj is None:
        return None
    name = fingerprint({'kind': kind, 'key': key})+'.json'
    found = [o for o in _original_operations(lifecycle) if o['kind'] == 'quarantined_publication'
             and o['detail']['kind'] == kind and o['detail']['key'] == key]
    if len(found) != 1 or any(_read(root/'publications'/name) != found[0]
                            for root in (obj.directory, obj.mirror)):
        raise ValueError('Original quarantined publication missing or changed')
    value = found[0]['detail']
    obj.verify_binding(value['observation_prefix'])
    wall = _check_physical(value['physical_observation'], value['observation_prefix']['physical_observation'])
    paths = _publication_paths(lifecycle, kind, key)
    files = []
    for path in paths:
        if isolated_copy is not None and str(path) == isolated_copy['damaged_copy']['path']:
            if kind != 'model' or key != isolated_copy['job'] or path.name != 'model.json':
                raise ValueError('Only the original isolated registry model copy can differ')
            originals = [entry for entry in value['files'] if entry['path'] == str(path)]
            if len(originals) != 1 or originals[0]['sha256'] != isolated_copy['expected_artifact_sha256']:
                raise ValueError('Original physical publication differs from isolated copy evidence')
            from .model_copy_isolation import _artifact
            if _artifact(path) != isolated_copy['damaged_copy']:
                raise ValueError('Isolated registry fault observation changed')
            files.append(originals[0])
        else:
            files.append(_stat(path))
    if (value['files'] != files or found[0]['at'] != value['physical_observation']['wall']
            or any(max(p['mtime_ns'], p['ctime_ns']) > wall.value for p in files)):
        raise ValueError('Original physical publication files changed or arrived late')
    return value


def publication_visible(lifecycle, kind, key, logical_cutoff):
    """Physical publication must predate the ORIGINAL release crossing cutoff."""
    from .nominal_trial_controller import from_lifecycle, publication_visible as nominal_visible
    if from_lifecycle(lifecycle) is not None:
        return nominal_visible(lifecycle, kind, key, logical_cutoff)
    value = read_publication(lifecycle, kind, key)
    if value is None:
        return None
    obj = QuarantinedObservation._open_original(lifecycle)
    state = obj.inspect(lifecycle)
    operations = [o for o in _original_operations(lifecycle) if o['kind'] == ADVANCED]
    for hour, op in enumerate(operations):
        source = _read(obj.directory/'events'/f'{hour:05d}.json')['source']
        prefix = obj._binding_value(state['identity_hash'], hour, op, source['logical_at'])
        if pd.Timestamp(prefix['logical_at']) >= pd.Timestamp(logical_cutoff):
            return (pd.Timestamp(value['observation_prefix']['logical_at']) <= pd.Timestamp(logical_cutoff)
                    and pd.Timestamp(value['physical_observation']['wall'])
                        <= pd.Timestamp(prefix['physical_observation']['wall']))
    return False
