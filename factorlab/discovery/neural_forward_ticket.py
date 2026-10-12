"""Canonical prepaid tickets for the fixed complete-neural forward worker.

Tickets are data, not commands or applicant assertions. Every verification
opens the actual wallet, lifecycle operation chain and cutoff receipt archive.
Historical completion verification never reloads registered models, avoiding
an admission/registry recursion, and never requires a dead parent to revive.
"""
from __future__ import annotations

import copy
import importlib.metadata
import json
import os
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import psutil
from filelock import FileLock

from factorlab.protocol import fingerprint, project_path

from .evidence import file_hash
from .exposure import ExposureLedger
from .forward_fit_snapshot import PreparedForwardSnapshot
from .frame_identity import frame_identity
from .model_registry import ModelRegistry
from .monthly_work_authority import MonthlyWorkAuthority, _nested_identity_read
from .neural_forward import FrozenNeuralRecipe
from .neural_forward_contract import (
    _confirmed_adoption_binding,
    assert_frozen_forward_fit,
    neural_pipeline,
)
from .nonlinear_launcher import _load_data, save_worker_data
from .nonlinear_orchestration import causal_worker_data, input_data_identity
from .operation_inputs import build_rolling_data
from .rolling import RollingSpec
from .v7_feed import ReceiptArchive
from .v7_forward import ForwardConfig, ForwardRuntime
from .v7_lifecycle import Lifecycle, LifecyclePlan, canonical, stamp
from .v7_storage import immutable_record
from .work_lease import FrozenWorkLease

VERSION = 'v7-neural-forward-fit-ticket-v2'
PACKAGES = ('numpy', 'pandas', 'scipy', 'scikit-learn', 'pyarrow', 'filelock', 'threadpoolctl', 'torch')


def _now():
    return pd.Timestamp.now(tz='UTC')


def runtime_identity():
    return {'executable': str(Path(sys.executable).resolve()), 'python': list(sys.version_info[:3]),
            'packages': {name: importlib.metadata.version(name) for name in PACKAGES}}


def _read(path):
    path = project_path(path)
    if not path.is_file() or path.stat().st_size > 16 * 2**20:
        raise ValueError('Missing or oversized forward ticket evidence')

    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('Duplicate forward evidence JSON key')
            value[key] = item
        return value

    def invalid(_):
        raise ValueError('Nonfinite forward evidence JSON')

    return json.loads(path.read_text('utf-8'), object_pairs_hook=unique, parse_constant=invalid)


def open_existing_authority(directory):
    """Open an existing authority without initializing or supplying new limits."""
    root = project_path(directory)
    wrapper = _read(root / 'manifest.json')
    if (not isinstance(wrapper, dict) or set(wrapper) != {'record', 'hash'}
            or not isinstance(wrapper['record'], dict)
            or wrapper['hash'] != fingerprint(wrapper['record'])):
        raise ValueError('Authority manifest changed')
    manifest = wrapper['record']
    result = MonthlyWorkAuthority.__new__(MonthlyWorkAuthority)
    result.directory, result.database = root, root / 'authority.sqlite'
    result.manifest, result.manifest_hash = manifest, wrapper['hash']
    result.limits = manifest['limits'].copy()
    result.index_path = project_path('work/monthly_work_authority_index') / (fingerprint(manifest['scope']) + '.json')
    if not _nested_identity_read(result):
        result.verify()
    return result


def _authority_reference(lifecycle, authority):
    scope = authority.manifest['scope']
    if scope['task'] != 'volatility' or scope['universe'] != sorted(lifecycle.plan.symbols):
        raise ValueError('Monthly work authority belongs to another task or universe')
    return {'version': 'v7-forward-work-authority-v1', 'plan_id': lifecycle.plan.digest,
            'authority_directory': str(authority.directory), 'authority_manifest_hash': authority.manifest_hash,
            'scope': scope}


def _verify_authority_binding(lifecycle, authority, operations):
    expected = _authority_reference(lifecycle, authority)
    registered = [o['detail'] for o in operations if o['kind'] == 'monthly_work_authority_bound']
    if registered != [expected] or _read(lifecycle.directory / 'work_authority.json') != expected:
        raise ValueError('Forward runtime changed or lost its unique monthly work authority')


def _bind_authority(runtime, authority):
    """Freeze one wallet for this actual runtime before any ticket preparation."""
    expected = _authority_reference(runtime.lifecycle, authority)
    path = runtime.directory / 'work_authority.json'
    with FileLock(str(path) + '.binding.lock', timeout=0), runtime.lifecycle.db:
        runtime.lifecycle.db.execute('BEGIN IMMEDIATE')
        operations = [json.loads(row[0]) for row in runtime.lifecycle.db.execute(
            'SELECT record FROM operations ORDER BY seq')]
        registered = [o['detail'] for o in operations if o['kind'] == 'monthly_work_authority_bound']
        if registered:
            _verify_authority_binding(runtime.lifecycle, authority, operations)
            if authority.manifest.get('partition_policy') is not None:
                immutable_record(authority.directory/'runtime_bindings'/fingerprint(str(runtime.directory))/'identity.json',
                    {'runtime_directory': str(runtime.directory), 'authority_manifest_hash': authority.manifest_hash,
                     'reference': expected})
            return
        immutable_record(path, expected)
        runtime.lifecycle._event('monthly_work_authority_bound', expected)
        if authority.manifest.get('partition_policy') is not None:
            immutable_record(authority.directory/'runtime_bindings'/fingerprint(str(runtime.directory))/'identity.json',
                {'runtime_directory': str(runtime.directory), 'authority_manifest_hash': authority.manifest_hash,
                 'reference': expected})


class _LifecycleReader(Lifecycle):
    """Read operation/index bindings without recursing into model admission."""
    def __init__(self, directory):
        self.directory = project_path(directory)
        value = _read(self.directory / 'lifecycle.json')
        plan = dict(value['plan']); plan['symbols'] = tuple(plan['symbols'])
        self.plan = LifecyclePlan(**plan)
        if value != {'plan': json.loads(canonical(asdict(self.plan))), 'sha256': self.plan.digest}:
            raise ValueError('Frozen lifecycle plan changed')
        self.clock = _now
        self.db = sqlite3.connect((self.directory / 'lifecycle.sqlite').as_uri() + '?mode=ro', uri=True)
        self.exposure = ExposureLedger.__new__(ExposureLedger)
        self.exposure.path = self.directory / 'exposure.sqlite'
        self.exposure.clock = self.now
        self.exposure.db = sqlite3.connect(self.exposure.path.as_uri() + '?mode=ro', uri=True)
        # A verifier must not initialize missing stores while loading history.
        self.registry = ModelRegistry.__new__(ModelRegistry)
        self.registry.directory = self.directory / 'models'
        if not self.registry.directory.is_dir():
            self.close()
            raise ValueError('Missing authoritative model registry')

    def verify(self):
        prior, jobs, active, operations = '0' * 64, {}, {}, []
        check_time = self._time_event_verifier()
        for sequence, text, digest in self.db.execute('SELECT * FROM operations ORDER BY seq'):
            operation = json.loads(text)
            if (sequence != len(operations) + 1 or operation['seq'] != sequence
                    or operation['previous'] != prior or fingerprint(operation) != digest):
                raise ValueError('Lifecycle operations chain mismatch')
            prior = digest; operations.append(operation)
            detail, kind = operation['detail'], operation['kind']
            check_time(operation)
            if kind == 'fit_planned':
                jobs[detail['job']] = [detail['cutoff'], detail['stream'], canonical(detail['pipeline']),
                                      'planned', None, None, None, None]
            elif kind in {'snapshot_sealed', 'features_ready', 'fitted'}:
                jobs[detail['job']][3] = kind
            elif kind == 'model_verified':
                jobs[detail['job']][3:6] = ['verified', detail['model'], detail['completed']]
                jobs[detail['job']][7] = None
            elif kind == 'fit_failed_old_model_retained':
                jobs[detail['job']][7] = detail['code']
            elif kind == 'activated':
                jobs[detail['job']][3], jobs[detail['job']][6] = 'activated', detail['origin']
                active[jobs[detail['job']][1]] = (detail['job'], detail['generation'])
        if {row[0]: list(row[1:]) for row in self.db.execute('SELECT * FROM jobs')} != jobs:
            raise ValueError('Monthly job index differs from operations chain')
        if {row[0]: tuple(row[1:]) for row in self.db.execute('SELECT * FROM active')} != active:
            raise ValueError('Active model index differs from operations chain')
        self._verify_batches(operations)
        return {'event_head': prior, 'operations': operations}

    def close(self):
        self.exposure.db.close()
        self.db.close()


def _snapshot_from_actual_source(reader, snapshot_ref, recipe, *, expected_pipeline=None):
    job = snapshot_ref['job']
    path = reader.directory / 'snapshots' / f'{job}.json'
    if str(path) != snapshot_ref['path'] or file_hash(path) != snapshot_ref['sha256']:
        raise ValueError('Ticket snapshot bytes or path changed')
    value = _read(path)
    from .quarantined_observation import job_binding, snapshot_for_prefix
    prefix = job_binding(reader, job)
    if (value.get('version') != 'v7-forward-fit-snapshot-v3'
            or value.get('raw_hash_rule') != 'typed-frame-columns-v1'
            or value['snapshot_id'] != fingerprint({k: v for k, v in value.items() if k != 'snapshot_id'})
            or value['job'] != job or value['pipeline_id'] != recipe.pipeline_id
            or value['plan_id'] != reader.plan.digest
            or value['availability'] != ('actual_receipts' if prefix is None else 'quarantined_finite_observation')
            or value.get('observation_prefix') != prefix):
        raise ValueError('Forward snapshot identity changed')
    row = reader.db.execute('SELECT cutoff,pipeline FROM jobs WHERE id=?', (job,)).fetchone()
    pipeline = neural_pipeline(recipe, recipe.rolling_spec) if expected_pipeline is None else expected_pipeline
    if not row or row[0] != value['cutoff'] or json.loads(row[1]) != pipeline:
        raise ValueError('Snapshot is not its authoritative neural fit job')
    operations = reader.verify()['operations']
    for kind, output in [('snapshot_sealed', value['snapshot_id']), ('features_ready', value['training'])]:
        found = [o['detail']['output'] for o in operations if o['kind'] == kind and o['detail']['job'] == job]
        if found != [output]:
            raise ValueError('Snapshot lacks its original lifecycle seal')
    forward = _read(reader.directory / 'forward.json')
    identity = forward['identity']
    if forward['sha256'] != fingerprint(identity) or identity['lifecycle'] != json.loads(canonical(asdict(reader.plan))):
        raise ValueError('Forward runtime identity changed')
    for relative, expected in identity['source_files'].items():
        if file_hash(reader.directory / 'source' / relative) != expected:
            raise ValueError('Frozen forward source archive changed')
    config = dict(identity['config']); config['symbols'] = tuple(config['symbols'])
    rolling = dict(identity['rolling']); rolling['event_bins'] = tuple(rolling['event_bins'])
    runtime = ForwardRuntime.__new__(ForwardRuntime)
    runtime.config, runtime.spec = ForwardConfig(**config), RollingSpec(**rolling)
    if runtime.spec.digest != recipe.rolling_spec.digest or runtime.config.context_hours != value['context_hours']:
        raise ValueError('Forward snapshot changed rolling or context policy')
    feed_root = project_path(identity['feed_directory'])
    if (str(feed_root) != value['feed_directory'] or value['feed'] != reader.plan.source_id
            or tuple(value['symbols']) != reader.plan.symbols):
        raise ValueError('Forward snapshot changed receipt archive')
    feed = ReceiptArchive.__new__(ReceiptArchive)
    feed.directory, feed.symbols, feed.identity = feed_root, runtime.config.symbols, value['feed']
    feed.clock = _now
    if _read(feed_root / 'identity.json') != {'source_id': feed.identity, 'symbols': list(feed.symbols)}:
        raise ValueError('Receipt archive identity changed')
    feed.db = sqlite3.connect((feed_root / 'receipts.sqlite').as_uri() + '?mode=ro', uri=True)
    runtime.feed = feed
    try:
        feed.verify()
        runtime.lifecycle = reader
        bars = (runtime._snapshot(value['cutoff']) if prefix is None else snapshot_for_prefix(
            reader, prefix, value['cutoff'], context_hours=runtime.config.context_hours))
        lineage = []
        for batch in sorted(set(bars.batch_id)):
            r = feed.db.execute('SELECT b.hash,p.hash,p.published_at FROM batches b '
                                'JOIN publications p ON p.batch=b.id WHERE b.id=?', (batch,)).fetchone()
            lineage.append({'batch': batch, 'batch_hash': r[0], 'publication_hash': r[1], 'published_at': r[2]})
        if (frame_identity(bars) != value['raw_asof_hash']
                or lineage != value['publication_lineage'] or list(bars.columns) != value['raw_schema']
                or list(runtime.config.symbols) != value['symbols']):
            raise ValueError('Actual cutoff source or publication lineage changed')
        data = causal_worker_data(build_rolling_data(bars, runtime.spec), value['cutoff'])
        PreparedForwardSnapshot(data, value, fingerprint(value)).verify()
        return value, data, identity['source_files']
    finally:
        feed.db.close()


def _verify_gates(gates, data, recipe, context, *, historical=False):
    from .nonlinear_runtime import verify_pre_resource_artifacts
    if set(gates) == {'shared_S', 'resource_certificate'}:
        from .s_nonlinear_gates import verify_shared_gates
        from .monthly_work_authority import source_files
        source_hash = fingerprint(source_files())
        report = verify_shared_gates(gates['shared_S'], data, recipe, context,
                                     source_hash=source_hash, require_resource=False)
    elif set(gates) == {'directory', 'artifacts', 'resource_certificate'}:
        report = verify_pre_resource_artifacts(gates['directory'], data, context['cutoff'], recipe.pool,
            gates['artifacts'], source_id=context['source_id'], config=recipe.config,
            forward_job=({'lifecycle_directory': context['lifecycle_directory'], 'job': context['job']}
                         if context.get('observation_prefix') is not None else None))
        source_hash = fingerprint(report['context']['source']['files'])
    else:
        raise ValueError('Forward gates must be actual registered producer references')
    if report['test_only']:
        raise ValueError('Artificial gates cannot permit a production worker')
    if context['purpose'] == 'resource_trial':
        if gates['resource_certificate'] is not None:
            raise ValueError('Resource trial cannot borrow a serving certificate')
    else:
        from .neural_forward_admission import verify_completed_artifact
        resource = verify_completed_artifact(gates['resource_certificate'], allow_resource_trial=True)
        trial = resource['context']
        resource_store = (project_path(gates['resource_certificate']['ticket']['path']).parent
                          / 'launch/publication_witness/witness.sqlite')
        # Logical simulation time grants no resource exemption. The original
        # resource must have physically completed before this job's prefix.
        physical_cutoff = stamp(context['cutoff'])
        if context.get('observation_prefix') is not None:
            from .quarantined_observation import job_binding
            reader = _LifecycleReader(context['lifecycle_directory'])
            try:
                prefix = job_binding(reader, context['job'])
                if prefix != context['observation_prefix']:
                    raise ValueError('Resource comparison lost original job prefix')
                physical_cutoff = stamp(prefix['physical_observation']['wall'])
            finally:
                reader.close()
        if (trial['purpose'] != 'resource_trial' or resource['recipe'].pipeline_id != recipe.pipeline_id
                or trial['source_id'] != context['source_id']
                or stamp(resource['reservation']['completed_at']) > physical_cutoff
                or stamp(resource['receipt']['final_io_observation']['wall']) > physical_cutoff
                or stamp(resource['publication']['observation']['wall']) > physical_cutoff
                or _publication_mtime_ns(resource_store) > physical_cutoff.value
                or resource['source_hash'] != source_hash):
            raise ValueError('Full forward workload resource proof belongs to another learner or cutoff')
        required = data.frame.values
        # Capacity comes from the verified original workload input, never a
        # caller-declared field in a receipt or a different S-only trial.
        capacity = resource['data'].frame.values
        if (capacity.shape[0] < required.shape[0] or capacity.shape[1] < required.shape[1]
                or capacity.nbytes < required.nbytes):
            raise ValueError('Full forward workload resource proof is too small')
    return report


def _publication_mtime_ns(path):
    # The final observer precedes COMMIT. Also require the actual immutable
    # per-ticket DELETE-journal database write to predate the next cutoff.
    # Filesystem timestamp precision is a local engineering anchor, not a
    # cryptographic or nanosecond physical-completion guarantee.
    return project_path(path).stat().st_mtime_ns


def _verify_historical_context(reader, context, recipe):
    """Bind completed work to original jobs/windows without reopening them."""
    fields = {'version', 'plan_id', 'lifecycle_directory', 'job', 'stream', 'cutoff',
              'pipeline', 'purpose', 'source_id', 'symbols', 'exposure_path', 'batch',
              'batch_hash', 'confirmation', 'adoption', 'serving_eligible'}
    protected = context.get('purpose') in {'V', 'C'}
    if protected:
        fields.add('authorized_at')
    from .quarantined_observation import job_binding
    prefix = job_binding(reader, context.get('job'))
    if prefix is not None:
        fields.add('observation_prefix')
    if context.get('observation_prefix') != prefix:
        raise ValueError('Completed context lost original job observation prefix')
    row = reader.db.execute('SELECT cutoff,stream,pipeline FROM jobs WHERE id=?',
                            (context.get('job'),)).fetchone()
    if (set(context) != fields or row is None
            or context['version'] != ('v7-neural-forward-context-v3' if protected
                                      else 'v7-neural-forward-context-v2')
            or context['plan_id'] != reader.plan.digest
            or project_path(context['lifecycle_directory']) != reader.directory
            or context['source_id'] != reader.plan.source_id
            or context['symbols'] != list(reader.plan.symbols)
            or project_path(context['exposure_path']) != reader.exposure.path
            or context['cutoff'] != row[0] or context['stream'] != row[1]
            or context['pipeline'] != json.loads(row[2])
            or context['pipeline'] != neural_pipeline(recipe, recipe.rolling_spec)):
        raise ValueError('Completed forward context differs from its original job')
    if row[1] == 'A0_policy_trial':
        from .typed_policy_confirmation import policy_context
        if (context['purpose'] != 'policy_trial' or context['serving_eligible'] is not True
                or context['adoption'] != policy_context(reader, context['job'], recipe)
                or any(context[k] is not None for k in ('batch', 'batch_hash', 'confirmation'))):
            raise ValueError('Policy trial changed original stream or whole-policy claim permission')
        return
    if row[1].startswith('neural-resource:'):
        if (context['purpose'] != 'resource_trial' or context['serving_eligible'] is not False
                or any(context[k] is not None for k in ('batch', 'batch_hash', 'confirmation', 'adoption'))):
            raise ValueError('Resource stream borrowed a serving authority')
        from .forward_resource_work import verify_resource_plan
        verify_resource_plan(reader, context['job'], recipe)
        return
    if row[1] == 'A0':
        if (context['purpose'] not in {'A0', 'resource_trial'}
                or any(context[k] is not None for k in ('batch', 'batch_hash', 'confirmation'))
                or context['serving_eligible'] is not (context['purpose'] == 'A0')):
            raise ValueError('Completed A0/resource context borrowed a protected role')
        if context['purpose'] == 'A0':
            _confirmed_adoption_binding(reader, context['cutoff'], context['pipeline'],
                                        expected=context['adoption'], historical=True)
        elif context['adoption'] is not None:
            raise ValueError('Completed resource trial cannot borrow an adoption authority')
        return
    batch, separator, role = row[1].rpartition(':')
    found = reader.db.execute('SELECT identity,hash FROM batches WHERE id=?', (batch,)).fetchone()
    if (not separator or not found or context['batch'] != batch
            or context['batch_hash'] != found[1] or context['serving_eligible'] is not True
            or context['adoption'] is not None):
        raise ValueError('Completed protected context lacks its frozen batch')
    identity = json.loads(found[0])
    expected = identity['reference'] if role == 'reference' else next(
        (c for c in identity['candidates'] if c['pipeline_id'] == role), None)
    if expected is None or {k: v for k, v in expected.items() if k != 'complexity'} != context['pipeline']:
        raise ValueError('Completed learner differs from its frozen batch')
    cutoff = stamp(row[0])
    if context['purpose'] == 'V':
        if (context['confirmation'] is not None
                or not reader.fit_cycle_start(identity['start']) <= cutoff < stamp(identity['v_end'])
                or not stamp(identity['start']) <= stamp(context['authorized_at']) < stamp(identity['v_end'])):
            raise ValueError('Completed V context is outside its original window')
        window_start, window_end = identity['start'], identity['v_end']
    elif context['purpose'] == 'C':
        from factorlab.time_utils import month_start
        found = reader.exposure.db.execute('SELECT identity FROM confirmations WHERE batch=?', (batch,)).fetchone()
        if not found or json.loads(found[0]) != context['confirmation']:
            raise ValueError('Completed C context lacks its original confirmation')
        window = context['confirmation']
        if role not in {'reference', window['candidate_id']}:
            raise ValueError('Completed C context is not its selected candidate')
        if (not reader.fit_cycle_start(window['actual_start']) <= cutoff < stamp(window['end'])
                or not stamp(window['actual_start']) <= stamp(context['authorized_at']) < stamp(window['end'])):
            raise ValueError('Completed C context is outside its original window')
        window_start, window_end = window['actual_start'], window['end']
    else:
        raise ValueError('Completed protected context borrowed another role')
    # Reconstruct only; a missing original authorization cannot be repaired by
    # reading a completed ticket, including after its V/C period has ended.
    from .calibration_runtime_contract import _original_operations
    events = [o for o in _original_operations(reader)
              if o['kind'] == 'neural_forward_authorized'
              and o['detail'].get('context', {}).get('job') == context['job']
              and o['detail'].get('context', {}).get('purpose') == context['purpose']]
    base = {k: v for k, v in context.items() if k != 'authorized_at'}
    if (len(events) != 1 or events[0]['detail'] != {'context': base}
            or events[0]['at'] != context['authorized_at']
            or not stamp(window_start) <= stamp(events[0]['at']) < stamp(window_end)):
        raise ValueError('Completed protected fit lost its original authorization')


@dataclass(frozen=True)
class ForwardFitTicket:
    path: Path
    value: dict
    _digest: str = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, 'path', project_path(self.path))
        object.__setattr__(self, 'value', json.loads(canonical(copy.deepcopy(self.value))))
        expected = {'version', 'reservation_id', 'authority_directory', 'authority_manifest_hash', 'terms_hash',
                    'lease', 'lease_hash', 'context', 'snapshot_ref', 'preparation_ref', 'data', 'recipe', 'runtime', 'source_hash', 'gates'}
        if set(self.value) != expected or self.value['version'] != VERSION:
            raise ValueError('Unknown complete forward ticket schema')
        lease = FrozenWorkLease.from_dict(self.value['lease'])
        if lease.digest != self.value['lease_hash']:
            raise ValueError('Forward ticket changed original lease')
        FrozenNeuralRecipe.from_dict(self.value['recipe'])
        root = project_path(self.value['authority_directory'])
        key = fingerprint({'plan': self.value['context']['plan_id'], 'job': self.value['context']['job']})
        if self.path != root / 'forward_tickets' / key / 'ticket.json':
            raise ValueError('Forward ticket is outside its canonical work directory')
        object.__setattr__(self, '_digest', fingerprint(self.value))

    @property
    def digest(self):
        if fingerprint(self.value) != self._digest:
            raise ValueError('Frozen forward ticket was modified')
        return self._digest

    @classmethod
    def load(cls, path):
        record = _read(path)
        if set(record) != {'value', 'sha256'} or record['sha256'] != fingerprint(record['value']):
            raise ValueError('Forward ticket checksum changed')
        return cls(project_path(path), record['value'])

    def _verify(self, *, launching, as_worker=False, running_parent=False):
        value = self.value
        if ForwardFitTicket.load(self.path).digest != self.digest:
            raise ValueError('Forward ticket differs from its persisted terms')
        authority = open_existing_authority(value['authority_directory'])
        reservation = authority.get(value['reservation_id'])
        verify_prepaid_ticket(self, authority, reservation)
        if (reservation['authority_manifest_hash'] != value['authority_manifest_hash']
                or reservation['terms_hash'] != value['terms_hash'] or reservation['lease'] != value['lease']
                or reservation['lease_hash'] != value['lease_hash']
                or runtime_identity() != value['runtime']
                or fingerprint(authority.manifest['source_files']) != value['source_hash']):
            raise ValueError('Forward ticket differs from its prepaid authority or frozen runtime')
        recipe = FrozenNeuralRecipe.from_dict(value['recipe'])
        limits = {k: n for k, n in recipe.budget_limits.items() if k != 'seconds'}
        if any(reservation['counts'][k] != limits.get(k, 0) for k in reservation['counts']):
            raise ValueError('Forward ticket changed its complete workload cost')
        if launching:
            if reservation['status'] not in ({'running'} if as_worker or running_parent else {'reserved'}):
                raise ValueError('Forward work is already consumed or not launchable')
            FrozenWorkLease.from_dict(value['lease']).assert_parent_alive(as_worker=as_worker)
            if as_worker:
                actual = {'pid': os.getpid(), 'created': psutil.Process().create_time()}
                if reservation['process_identity'] != actual:
                    raise ValueError('Worker is not the canonical started process')
        elif reservation['status'] != 'completed':
            raise ValueError('Forward artifact lacks completed canonical work')
        reader = _LifecycleReader(value['context']['lifecycle_directory'])
        try:
            from .quarantined_observation import _original_observation_read_scope
            with _original_observation_read_scope(reader):
                history = reader.verify()
                _verify_authority_binding(reader, authority, history['operations'])
                if launching:
                    context = assert_frozen_forward_fit(reader, value['context']['job'], recipe,
                                                       purpose=value['context']['purpose'])['binding']
                    if context != value['context']:
                        raise ValueError('Forward job/batch/stream permission changed')
                else:
                    _verify_historical_context(reader, value['context'], recipe)
                snapshot, actual, source = _snapshot_from_actual_source(reader, value['snapshot_ref'], recipe)
                if source != authority.manifest['source_files']:
                    raise ValueError('Forward source is not the authority source')
                if snapshot['snapshot_id'] != value['snapshot_ref']['snapshot_id']:
                    raise ValueError('Forward ticket snapshot seal changed')
                if any(project_path(value['data'][key]).parent != self.path.parent / 'payload'
                       for key in ('arrays', 'metadata')):
                    raise ValueError('Worker payload paths are not canonical')
                for key in ('arrays', 'metadata'):
                    if project_path(value['data'][key]).stat().st_size > 16 * 2**20:
                        raise ValueError('Worker payload exceeds its bounded size')
                payload = _load_data({'data': value['data'], 'rolling': asdict(recipe.rolling_spec)})
                PreparedForwardSnapshot(payload, snapshot, fingerprint(snapshot)).verify()
                if input_data_identity(payload, value['context']['cutoff']) != input_data_identity(actual, value['context']['cutoff']):
                    raise ValueError('Worker payload is not the actual sealed cutoff input')
                _verify_gates(value['gates'], payload, recipe, value['context'], historical=not launching)
        finally:
            reader.close()
        if launching:
            FrozenWorkLease.from_dict(value['lease']).assert_parent_alive(as_worker=as_worker)
        return {'authority': authority, 'reservation': reservation, 'recipe': recipe, 'context': value['context'],
                'snapshot': snapshot, 'data_contract': value['data'], 'data': payload,
                'runtime': value['runtime'], 'ticket_digest': self.digest, 'source_hash': value['source_hash']}

    def verify_for_launch(self, *, as_worker=False):
        result = self._verify(launching=True, as_worker=as_worker)
        return result['authority'], result['reservation'], result['data'], result['recipe']

    def verify_completed_binding(self):
        return self._verify(launching=False)

    def verify_running_parent(self):
        result = self._verify(launching=True, running_parent=True)
        return result['authority'], result['reservation'], result['data'], result['recipe']


def _allocation_pair(authority, work):
    primary = authority.directory/'forward_tickets'/work
    intent = _read(primary/'allocation.json')
    if not isinstance(intent, dict) or set(intent) != {
            'version', 'work_key', 'authority_directory', 'authority_manifest_hash', 'context',
            'recipe', 'runtime', 'source_hash', 'gates', 'feed_directory', 'feed_identity',
            'forward_hash', 'counts', 'lease'}:
        raise ValueError('Unknown canonical neural prework intent')
    runtime_directory = project_path(intent['context']['lifecycle_directory'])
    secondary = runtime_directory/'forward_allocations'/work
    if _read(secondary/'allocation.json') != intent:
        raise ValueError('Neural prework intent anchors differ')
    return intent, primary, secondary


def read_prepaid_intent(authority, work_key, binding, lease, counts):
    """Readonly canonical debit proof; callable inside the wallet transaction.

    No snapshot/final ticket is needed to authorize its previously frozen cost.
    This is local trusted-producer provenance, not administrator authentication.
    """
    prefix = 'neural_forward:'
    work = work_key.removeprefix(prefix)
    if not work_key.startswith(prefix) or len(work) != 64 or any(c not in '0123456789abcdef' for c in work):
        raise ValueError('Invalid neural prework identity')
    intent, _, _ = _allocation_pair(authority, work)
    recipe = FrozenNeuralRecipe.from_dict(intent['recipe'])
    context = intent['context']
    directory = project_path(context['lifecycle_directory'])
    frozen_runtime = _read(directory/'forward.json')
    limits = {k: recipe.budget_limits.get(k, 0) for k in counts}
    expected = {'prework_terms': fingerprint(intent), 'job': context['job'],
        'pipeline_id': recipe.pipeline_id, 'purpose': context['purpose']}
    if (intent['version'] != 'neural-forward-prework-v2' or intent['work_key'] != work_key
            or work != fingerprint({'plan': context['plan_id'], 'job': context['job']})
            or binding != expected or intent['lease'] != lease.to_dict() or lease.lease_seconds != 300
            or counts != intent['counts'] or counts != limits
            or intent['authority_directory'] != str(authority.directory)
            or intent['authority_manifest_hash'] != authority.manifest_hash
            or intent['source_hash'] != fingerprint(authority.manifest['source_files'])
            or file_hash(directory/'forward.json') != intent['forward_hash']
            or frozen_runtime != {'identity': frozen_runtime.get('identity'), 'sha256': fingerprint(frozen_runtime.get('identity'))}
            or frozen_runtime['identity']['source_files'] != authority.manifest['source_files']
            or context['source_id'] != intent['feed_identity']
            or frozen_runtime['identity']['feed_directory'] != intent['feed_directory']
            or _read(project_path(intent['feed_directory'])/'identity.json') != {
                'source_id': intent['feed_identity'], 'symbols': context['symbols']}):
        raise ValueError('Neural prework differs from complete original cost/source/lease')
    return intent


def verify_prepaid_ticket(ticket, authority, record):
    """Readback binding for ticket/admission/publication, without old-parent liveness."""
    value = ticket.value
    intent = read_prepaid_intent(authority, record['work_key'], record['binding'],
        FrozenWorkLease.from_dict(record['lease']), record['counts'])
    work = record['work_key'].removeprefix('neural_forward:')
    _, primary, secondary = _allocation_pair(authority, work)
    ref = {'path': str(primary/'allocation.json'), 'digest': fingerprint(intent)}
    expected_seal = {'prework_hash': fingerprint(intent), 'ticket_digest': ticket.digest,
        'ticket_path': str(ticket.path), 'snapshot_ref': value['snapshot_ref']}
    if (ticket.path != primary/'ticket.json' or value['preparation_ref'] != ref
            or any(value[k] != intent[k] for k in ('context', 'recipe', 'runtime', 'source_hash', 'gates', 'authority_directory'))
            or any(value[k] != record[k] for k in ('reservation_id', 'authority_manifest_hash', 'terms_hash', 'lease', 'lease_hash'))
            or _read(primary/'seal.json') != expected_seal or _read(secondary/'seal.json') != expected_seal):
        raise ValueError('Final neural ticket lost its original prepaid authority preparation seal')
    if _read(primary/'preparation.json') != {'reservation_id': record['reservation_id'], 'lease': record['lease']}:
        raise ValueError('Neural preparation lost its canonical reservation')
    return record


def _canonical_forward_gates(gates):
    gates = json.loads(canonical(gates))
    if isinstance(gates.get('artifacts', {}).get('probes'), list):
        gates['artifacts']['probes'] = sorted(gates['artifacts']['probes'])
    return gates


def issue_forward_ticket(runtime, authority, job, recipe, *, purpose, gates):
    """Prepay the complete job BEFORE snapshot, gate computation and payload I/O.

    Retry is historical readback only. One stable job cannot obtain another
    lease by changing its recipe, input, purpose, gates or final snapshot.
    """
    if type(runtime) is not ForwardRuntime or type(authority) is not MonthlyWorkAuthority or type(recipe) is not FrozenNeuralRecipe:
        raise TypeError('Actual runtime, shared wallet and full frozen recipe required')
    work = fingerprint({'plan': runtime.lifecycle.plan.digest, 'job': job})
    folder = authority.directory/'forward_tickets'/work
    path = folder/'ticket.json'
    secondary = runtime.directory/'forward_allocations'/work
    with FileLock(str(folder)+'.lock', timeout=0):
        if path.exists():
            gates = _canonical_forward_gates(gates)
            existing = ForwardFitTicket.load(path)
            if (existing.value['recipe'] != json.loads(canonical(recipe.to_dict()))
                    or existing.value['gates'] != gates or existing.value['context']['purpose'] != purpose):
                raise ValueError('Forward job cannot change its gates, purpose or recipe on retry')
            verify_prepaid_ticket(existing, authority, authority.get(existing.value['reservation_id']))
            return existing
        if folder.exists() or secondary.exists():
            raise ValueError('Incomplete forward authority preparation cannot reset or retry its original work')
        # Locating existing immutable work is readonly. New work starts its one
        # original clock before gate serialization, authorization or preparation.
        from .forward_service_scope import check_service_work
        check_service_work(runtime, new_work=True)
        lease = FrozenWorkLease.issue(300)
        check_service_work(runtime, lease=lease)
        from .quarantined_observation import _original_observation_read_scope
        record = None
        try:
            with _original_observation_read_scope(runtime.lifecycle):
                gates = _canonical_forward_gates(gates)
                context = assert_frozen_forward_fit(runtime.lifecycle, job, recipe, purpose=purpose)['binding']
                if runtime.identity['source_files'] != authority.manifest['source_files']:
                    raise ValueError('Forward runtime and work authority have different frozen sources')
                _bind_authority(runtime, authority)
                from .monthly_work_authority import DEFAULT_LIMITS
                counts = {k: recipe.budget_limits.get(k, 0) for k in DEFAULT_LIMITS}
                intent = {'version': 'neural-forward-prework-v2', 'work_key': 'neural_forward:'+work,
                    'authority_directory': str(authority.directory), 'authority_manifest_hash': authority.manifest_hash,
                    'context': context, 'recipe': json.loads(canonical(recipe.to_dict())), 'runtime': runtime_identity(),
                    'source_hash': fingerprint(authority.manifest['source_files']), 'gates': gates,
                    'feed_directory': str(runtime.feed.directory), 'feed_identity': runtime.feed.identity,
                    'forward_hash': file_hash(runtime.directory/'forward.json'), 'counts': counts, 'lease': lease.to_dict()}
                for directory in (folder, secondary):
                    immutable_record(directory/'allocation.json', intent)
                record = authority._reserve_with_lease(intent['work_key'], counts=counts, lease=lease,
                    binding={'prework_terms': fingerprint(intent), 'job': job,
                        'pipeline_id': recipe.pipeline_id, 'purpose': purpose})
                immutable_record(folder/'preparation.json', {'reservation_id': record['reservation_id'], 'lease': record['lease']})
                lease.assert_parent_alive()
                prepared = runtime.prepare_fit_snapshot(job)
                prepared.verify()
                lease.assert_parent_alive()
                _verify_gates(gates, prepared.data, recipe, context)
                lease.assert_parent_alive()
                payload = save_worker_data(folder/'payload', prepared.data, cutoff=context['cutoff'])
                lease.assert_parent_alive()
                snapshot_path = runtime.directory/'snapshots'/(job+'.json')
                base = {'version': VERSION, 'authority_directory': str(authority.directory),
                    'context': context, 'snapshot_ref': {'job': job, 'path': str(snapshot_path),
                        'sha256': file_hash(snapshot_path), 'snapshot_id': prepared.record['snapshot_id']},
                    'preparation_ref': {'path': str(folder/'allocation.json'), 'digest': fingerprint(intent)},
                    'data': payload, 'recipe': intent['recipe'], 'runtime': intent['runtime'],
                    'source_hash': intent['source_hash'], 'gates': gates}
                ticket = ForwardFitTicket(path, {**base, **{k: record[k] for k in (
                    'reservation_id', 'authority_manifest_hash', 'terms_hash', 'lease', 'lease_hash')}})
                immutable_record(path, {'value': ticket.value, 'sha256': ticket.digest})
                seal = {'prework_hash': fingerprint(intent), 'ticket_digest': ticket.digest,
                    'ticket_path': str(path), 'snapshot_ref': base['snapshot_ref']}
                for directory in (folder, secondary):
                    immutable_record(directory/'seal.json', seal)
                lease.assert_parent_alive()
            lease.assert_parent_alive()
            return ticket
        except BaseException as exc:
            try:
                if record is not None:
                    authority.fail_consumed(record['reservation_id'], 'neural_preparation:'+type(exc).__name__)
            except BaseException as secondary_error:  # noqa: BLE001 -- retain initiating error; no retry path
                exc.add_note('Prepaid failure accounting unavailable: '+type(secondary_error).__name__)
            raise
