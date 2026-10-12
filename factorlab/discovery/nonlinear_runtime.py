"""Receipt-bound S-only neural research, with a durable independent ledger.

This module is the runtime admission boundary. The lower-level neural function
is an engineering primitive, not an admission API. Linux trials require the
fixed local launcher and its canonical RunStore reservation; a JSON claim or
hash alone cannot install a verifier. This is not remote identity attestation.
"""
from __future__ import annotations

import copy
import json
import math
import sqlite3
import time
from datetime import UTC, datetime

import pandas as pd
from filelock import FileLock

from factorlab.protocol import fingerprint, project_path
from factorlab.time_utils import calendar_window

from .evidence import file_hash, source_paths
from .frame_identity import frame_identity
from .nonlinear_channel import (
    GATE_FIELDS,
    ChannelBudget,
    NeuralGate,
    ProbeConfig,
    ResidualConfig,
    research_residual,
    run_probe,
)
from .rolling import development_folds
from .v7_storage import immutable_record


def _now():
    return pd.Timestamp(datetime.now(UTC))


def _stamp(value):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None or pd.isna(stamp):
        raise ValueError('Evidence times must be timezone-aware')
    return stamp.tz_convert('UTC')


def _source():
    root = project_path('.')
    files = source_paths(root)
    manifest = {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(files)}
    return {'files': manifest, 'digest': fingerprint(manifest)}


def _context(data, cutoff, pool, source_id):
    from .joint_pool import FrozenPool
    _, end = data.spec.window(cutoff)
    if data.spec.task != 'volatility' or data.spec.horizon_hours != 12 or data.spec.months != 3:
        raise ValueError('Receipt runtime isolates volatility12h; event/state need a separate protocol')
    if not isinstance(pool, FrozenPool) or not isinstance(source_id, str) or not source_id.strip():
        raise ValueError('A frozen pool and upstream source identity are required')
    visible = (data.frame.metadata.available_at < end) & (data.maturity <= end)
    visible = visible & data.observed & data.frame.valid_rows
    data_id = fingerprint({'rule': 'typed-frame-columns-v1',
                           'metadata': frame_identity(data.frame.metadata.loc[visible]),
                           'values': data.frame.values[visible].tolist(),
                           'target': data.target[visible].tolist(),
                           'maturity': frame_identity(pd.DataFrame({'maturity': data.maturity[visible]}))})
    return {'task': 'volatility12h', 'source_id': source_id, 'source': _source(),
            'protocol_id': data.spec.digest, 'pool_id': pool.digest,
            'input_schema': data.frame.schema_hash, 'data_id': data_id, 'cutoff': end.isoformat()}


def _common(context):
    return {k: v for k, v in context.items() if k not in {'cutoff', 'data_id'}}


class _Store:
    def __init__(self, directory):
        self.directory = project_path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.directory / 'nonlinear.sqlite')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS records (kind TEXT, id TEXT, value TEXT, '
                        'digest TEXT, PRIMARY KEY(kind,id))')

    def put(self, kind, key, value):
        payload = json.dumps(value, sort_keys=True, allow_nan=False)
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO records VALUES (?,?,?,?)',
                            (kind, key, payload, fingerprint(value)))

    def get(self, kind, key):
        row = self.db.execute('SELECT value,digest FROM records WHERE kind=? AND id=?',
                              (kind, key)).fetchone()
        if row is None:
            return None
        value = json.loads(row[0])
        if fingerprint(value) != row[1]:
            raise ValueError('Nonlinear registry integrity failure')
        return value

    def close(self):
        self.db.close()


def _register(store, kind, context, result, *, test_only=False, started_at=None):
    """Internal producer-only registration, deliberately no public JSON importer."""
    value = {'version': 'v7-nonlinear-receipt-v1', 'kind': kind, 'role': 'S',
             'context': context, 'available_at': _now().isoformat(),
             'started_at': (_now() if started_at is None else _stamp(started_at)).isoformat(),
             'test_only': test_only, 'result': result, 'result_hash': fingerprint(result)}
    receipt_id = fingerprint(value)
    path = store.directory / 'receipts' / f'{receipt_id}.json'
    immutable_record(path, value)
    store.put('receipt', receipt_id, {'file_hash': file_hash(path), 'receipt_id': receipt_id})
    return receipt_id


def physical_bound_for_prefix(lifecycle, prefix):
    """Original observed physical boundary, never a caller-provided clock."""
    from .quarantined_observation import QuarantinedObservation
    observed = QuarantinedObservation.from_lifecycle(lifecycle)
    if observed is None:
        raise ValueError('Physical simulation bound requires original enrolled observation')
    observed.verify_binding(prefix)
    return _stamp(prefix['physical_observation']['wall'])


def _gate_physical_cutoff(context, forward_job):
    if forward_job is None:
        return _stamp(context['cutoff'])
    if type(forward_job) is not dict or set(forward_job) != {'lifecycle_directory', 'job'}:
        raise ValueError('Original lifecycle and fit job required, not a caller clock')
    from .neural_forward_ticket import _LifecycleReader
    from .neural_forward import FrozenNeuralRecipe
    from .quarantined_observation import job_binding
    reader = _LifecycleReader(forward_job['lifecycle_directory'])
    try:
        prefix = job_binding(reader, forward_job['job'])
        row = reader.db.execute('SELECT cutoff,pipeline FROM jobs WHERE id=?', (forward_job['job'],)).fetchone()
        pipeline = json.loads(row[1])
        if pipeline.get('kind') != 'v7_neural_forward':
            raise ValueError('Neural gate cannot borrow another original learner')
        recipe = FrozenNeuralRecipe.from_dict(pipeline['recipe'])
        source = json.loads((reader.directory/'forward.json').read_text('utf-8'))['identity']['source_files']
        if (prefix is None or _stamp(context['cutoff']) != _stamp(row[0])
                or context['source_id'] != reader.plan.source_id
                or context['protocol_id'] != recipe.rolling_spec.digest
                or context['pool_id'] != recipe.pool.digest
                or context['source'] != {'files': source, 'digest': fingerprint(source)}):
            raise ValueError('Neural gate differs from its original job/source observation')
        return physical_bound_for_prefix(reader, prefix)
    finally:
        reader.close()


def _read(store, receipt_id, context, cutoff, *, forward_job=None):
    if not isinstance(receipt_id, str) or len(receipt_id) != 64:
        raise ValueError('Gate requires registered artifact IDs, not booleans')
    index = store.get('receipt', receipt_id)
    if index is None:
        raise ValueError('Unregistered neural gate artifact')
    path = store.directory / 'receipts' / f'{receipt_id}.json'
    if file_hash(path) != index['file_hash']:
        raise ValueError('Neural receipt bytes changed')
    value = json.loads(path.read_text('utf-8'))
    physical_cutoff = _stamp(cutoff) if forward_job is None else _gate_physical_cutoff(context, forward_job)
    physical_late = False
    if forward_job is not None:
        observed = path.stat()
        physical_late = max(observed.st_mtime_ns, observed.st_ctime_ns) > physical_cutoff.value
    if (fingerprint(value) != receipt_id or value['result_hash'] != fingerprint(value['result'])
            or value['role'] != 'S' or _common(value['context']) != _common(context)
            or _stamp(value['available_at']) > physical_cutoff or physical_late
            or _stamp(value['context']['cutoff']) > cutoff
            or _stamp(value['started_at']) > _stamp(value['available_at'])
            or (value['kind'] != 'protocol'
                and _stamp(value['available_at']) < _stamp(value['context']['cutoff']))):
        raise ValueError('Neural receipt identity, role or availability mismatch')
    return value


def freeze_protocol(directory, data, cutoff, pool, *, source_id, config=None):
    """Freeze the bounded candidate/configuration before collecting gate evidence."""
    config = config or ResidualConfig(enabled=True)
    if not config.enabled or any(s.experts != 1 for s in config.structures):
        raise ValueError('Only enabled single-expert protocols are supported')
    context = _context(data, cutoff, pool, source_id)
    store = _Store(directory)
    try:
        return _register(store, 'protocol', context,
                         {'config_id': config.digest, 'probe_id': ProbeConfig().digest,
                          'criterion': 'two_distinct_quarters_all_three_paired_gains_positive',
                          'positive_control': 'known_component_recovery_probe:rv_24',
                          'claim': 'development_permission_only_not_efficacy_or_confirmation'})
    finally:
        store.close()


def record_probe(directory, data, cutoff, pool, *, source_id, budget=None):
    """Execute the actual fixed probe; registering user-authored results is unsupported."""
    return _produce(directory, data, cutoff, pool, source_id, 'probe',
                    lambda ledger: run_probe(data, cutoff, pool, ledger), budget)


def record_positive_control(directory, data, cutoff, pool, *, source_id, budget=None):
    from .v7_controls import known_component_recovery_probe

    def produce(ledger):
        ledger.consume('solve', 6)
        return known_component_recovery_probe(data, cutoff, component='rv_24')

    return _produce(directory, data, cutoff, pool, source_id, 'positive_control', produce, budget)


def _external_resource_verifier(artifact, expected):
    """Accept only the trusted launcher's actual run-bound resource proof.

    No JSON `verified` field is trusted. Private monkeypatches are restricted to
    synthetic test fixtures and are never forwarded to a production worker.
    """
    if not isinstance(artifact, dict) or not artifact.get('launcher_receipt'):
        raise ValueError('Independent Linux supervisor receipt verifier is not installed')
    from .nonlinear_launcher import verify_launcher_artifact
    return verify_launcher_artifact(artifact, expected)


def _resource(value, expected):
    attestation = _external_resource_verifier(value, expected)
    if not isinstance(attestation, dict) or attestation.get('verified') is not True:
        raise ValueError('Independent resource attestation rejected')
    required = {'platform': 'linux', 'scope': 'whole_cgroup', 'oom': 0, 'oom_kill': 0,
                'memory_max': 768 * 2**20, 'memory_high': 640 * 2**20,
                'swap_max': 0, 'cpu_quota_fraction': 1.0}
    if any(type(value.get(k)) is not type(v) or value.get(k) != v for k, v in required.items()):
        raise ValueError('Linux resource contract mismatch')
    peak = value.get('memory_peak')
    if (type(peak) is not int or not 0 < peak < 600 * 2**20
            or value.get('trial_context') != expected or value.get('status') != 'completed'
            or not value.get('raw_receipt_hash') or not value.get('workload_result_hash')):
        raise ValueError('Whole-cgroup peak, workload or scope is not verified')
    if (not value.get('trial_started_at') or not value.get('trial_finished_at')
            or not _stamp(value['trial_started_at']) <= _stamp(value['trial_finished_at']) <= _now()):
        raise ValueError('Resource trial requires observed start and completion times')
    return attestation


def workload_capacity(data, cutoff):
    return {'train_row_capacity': max(int(data.train_mask(f['fit_cutoff']).sum())
                                     for f in development_folds(cutoff, spec=data.spec)),
            'score_row_capacity': max(int(data.score_mask(f['fit_cutoff'], f['score_end'], f['score_end']).sum())
                                     for f in development_folds(cutoff, spec=data.spec)),
            'frame_row_capacity': len(data.frame.values), 'feature_capacity': data.frame.values.shape[1],
            'frame_byte_capacity': data.frame.values.nbytes}


def _capacity(value, data, cutoff):
    """Cover the full materialized frame, not only rows entering fold losses."""
    expected = workload_capacity(data, cutoff)
    if any(type(value.get(k)) is not int or value[k] < amount for k, amount in expected.items()):
        raise ValueError('Linux workload receipt does not cover current row capacity')


def record_resource_receipt(directory, data, cutoff, pool, artifact, *, source_id, config=None):
    config = config or ResidualConfig(enabled=True)
    context = _context(data, cutoff, pool, source_id)
    if _stamp(context['cutoff']) > _now():
        raise ValueError('Resource trial cannot certify a future cutoff')
    expected = {'context': _common(context), 'config_id': config.digest}
    attestation = _resource(artifact, expected)
    _capacity(artifact, data, cutoff)
    store = _Store(directory)
    try:
        return _register(store, 'resource', context, artifact,
                         test_only=attestation.get('test_only') is True)
    finally:
        store.close()


def _verify_research_evidence(directory, data, cutoff, pool, gate_artifacts, *, source_id,
                              config=None, require_resource=True, forward_job=None):
    config = config or ResidualConfig(enabled=True)
    context = _context(data, cutoff, pool, source_id)
    end = _stamp(context['cutoff'])
    if end > _now():
        raise ValueError('Cannot open a future neural research cutoff')
    required = {'protocol', 'probes', 'positive_control'} | ({'resource'} if require_resource else set())
    if (not isinstance(gate_artifacts, dict)
            or set(gate_artifacts) != required
            or not isinstance(gate_artifacts['probes'], (tuple, list))
            or not 2 <= len(gate_artifacts['probes']) <= 8):
        raise ValueError('Missing independent neural gate artifacts')
    store = _Store(directory)
    try:
        records = {name: _read(store, key, context, end, forward_job=forward_job) for name, key in gate_artifacts.items()
                   if name != 'probes'}
        probes = [_read(store, key, context, end, forward_job=forward_job) for key in gate_artifacts['probes']]
    finally:
        store.close()
    if any(row['kind'] != name for name, row in records.items()):
        raise ValueError('Wrong gate artifact producer')
    protocol = records['protocol']
    if protocol['result']['config_id'] != config.digest:
        raise ValueError('Neural configuration differs from frozen protocol')
    quarters, schemas = set(), set()
    for row in probes:
        result = row['result']
        point = _stamp(row['context']['cutoff'])
        if (row['kind'] != 'probe' or point.month not in (1, 4, 7, 10)
                or result.get('status') != 'completed' or result.get('role') != 'S'
                or result.get('protocol_id') != ProbeConfig().digest
                or len(result.get('folds', [])) != 3
                or _stamp(row['started_at']) < _stamp(protocol['available_at'])):
            raise ValueError('Probe is not a protocol-frozen quarterly result')
        quarters.add((point.year, (point.month - 1) // 3))
        for fold, frozen in zip(result['folds'], development_folds(point, spec=data.spec), strict=True):
            if (any(fold.get(k) != v for k, v in frozen.items())
                    or fold.get('pool_id') != pool.digest or fold.get('status') != 'scored'
                    or not fold.get('model_id') or not fold.get('reference_model_id')
                    or not fold.get('sample_ids') or fold.get('support') != len(fold['sample_ids'])
                    or not math.isfinite(fold.get('loss', math.nan))
                    or not math.isfinite(fold.get('reference_loss', math.nan))
                    or not 0 <= fold['loss'] < fold['reference_loss']):
                raise ValueError('Probe lacks repeatable same-input paired improvement')
            schemas.add(fold['feature_schema'])
    if len(quarters) < 2 or len(schemas) != 1:
        raise ValueError('Need two distinct quarters under one input schema')
    control = records['positive_control']
    result = control['result']
    if (result.get('scope') != 'fixed_S_known_component_recovery_diagnostic'
            or result.get('component') != 'rv_24' or len(result.get('folds', [])) != 3
            or _stamp(control['started_at']) < _stamp(protocol['available_at'])):
        raise ValueError('Missing actual known-component recovery probe')
    losses = []
    for fold, frozen in zip(result['folds'], development_folds(control['context']['cutoff'], spec=data.spec), strict=True):
        if (any(fold.get(k) != v for k, v in frozen.items())
                or not fold.get('support_hash') or fold.get('score_rows', 0) < 1):
            raise ValueError('Known-component support mismatch')
        pair = fold['losses']
        if any(not math.isfinite(pair[k]) or pair[k] < 0 for k in ('removed', 'restored')):
            raise ValueError('Nonfinite known-component control')
        losses.append(pair['removed'] - pair['restored'])
    if sum(losses) <= 0:
        raise ValueError('Known-component recovery is not positive')
    return context, records, probes, protocol, config


def verify_pre_resource_artifacts(directory, data, cutoff, pool, gate_artifacts, *, source_id, config=None,
                                  forward_job=None):
    """Only permits a separately bounded resource trial, never research enablement."""
    context, records, probes, _, _ = _verify_research_evidence(
        directory, data, cutoff, pool, gate_artifacts, source_id=source_id,
        config=config, require_resource=False, forward_job=forward_job)
    return {'context': context, 'gate_id': fingerprint(gate_artifacts),
            'test_only': any(row['test_only'] for row in [*records.values(), *probes]),
            'permission': 'controlled_resource_trial_only', 'research_enabled': False}


def verify_gate_artifacts(directory, data, cutoff, pool, gate_artifacts, *, source_id, config=None,
                          forward_job=None):
    context, records, probes, protocol, config = _verify_research_evidence(
        directory, data, cutoff, pool, gate_artifacts, source_id=source_id, config=config, forward_job=forward_job)
    expected = {'context': _common(context), 'config_id': config.digest}
    attestation = _resource(records['resource']['result'], expected)
    _capacity(records['resource']['result'], data, cutoff)
    if (_stamp(records['resource']['result']['trial_started_at']) < _stamp(protocol['available_at'])
            or _stamp(records['resource']['result']['trial_finished_at'])
            > _stamp(records['resource']['available_at'])):
        raise ValueError('Resource trial predates the frozen protocol')
    test_only = attestation.get('test_only') is True or any(
        row['test_only'] for row in [*records.values(), *probes])
    return {'context': context, 'artifacts': copy.deepcopy(gate_artifacts), 'test_only': test_only,
            'formal_claim': False, 'gate_id': fingerprint(gate_artifacts)}


def _produce(directory, data, cutoff, pool, source_id, kind, producer, budget):
    context = _context(data, cutoff, pool, source_id)
    if _stamp(context['cutoff']) > _now():
        raise ValueError('Cannot run a future development cutoff')
    key = fingerprint({'kind': kind, 'context': context})
    return _durable(directory, key, context, producer, budget, receipt_kind=kind)


def _restore_shared(current, saved):
    if current.limits != saved.limits:
        raise ValueError('Shared budget limits changed')

    def dominates(left, right):
        return (all(left.counts[k] >= v for k, v in right.counts.items())
                and all(k in left.families and all(left.families[k][c] >= n for c, n in value.items())
                        for k, value in right.families.items()))

    if dominates(current, saved):
        return
    if not dominates(saved, current):
        raise ValueError('Divergent shared ledgers require the authoritative controller; cannot merge spending')
    current.counts, current.families = copy.deepcopy(saved.counts), copy.deepcopy(saved.families)


def _budget_state(store, month):
    """A missing checkpoint is not evidence that this month never spent anything."""
    state = store.get('budget', month)
    marker = store.directory / 'budgets' / f'{month}.started.json'
    expected = {'version': 'v7-nonlinear-budget-start-v1', 'month': month}
    if marker.exists() and json.loads(marker.read_text('utf-8')) != expected:
        raise ValueError('Nonlinear budget start marker changed')
    if state is not None:
        if not marker.exists():
            raise ValueError('Missing nonlinear budget start checkpoint')
        return state
    if marker.exists():
        raise ValueError('Missing nonlinear budget checkpoint for a consumed month')
    # Also reject old/orphan producer artifacts when the SQL row and start marker
    # are missing. Protocol/resource registration itself does not spend fit budget.
    for path in (store.directory / 'receipts').glob('*.json'):
        receipt = json.loads(path.read_text('utf-8'))
        if (receipt.get('kind') in {'probe', 'positive_control', 'residual'}
                and _stamp(receipt['context']['cutoff']).strftime('%Y-%m') == month):
            raise ValueError('Orphan nonlinear result has no budget checkpoint')
    return None


def restore_shared_budget(directory, cutoff, budget):
    """Controller must reconcile this durable ledger before resuming other channels.

    The external monthly owner still provides its own exclusive controller lock.
    This restores all previously persisted spending, including failed NN work.
    """
    from .local_mutation import MonthlyBudget
    month = _stamp(cutoff).strftime('%Y-%m')
    if not isinstance(budget, MonthlyBudget) or budget.month_id != month:
        raise ValueError('Shared budget must be the matching monthly ledger')
    store = _Store(directory)
    try:
        state = _budget_state(store, month)
        if state:
            _restore_shared(budget, MonthlyBudget.from_dict(state['shared']))
        return budget
    finally:
        store.close()


def _durable(directory, key, context, producer, shared, *, receipt_kind=None):
    from .local_mutation import MonthlyBudget
    directory = project_path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    month = _stamp(context['cutoff']).strftime('%Y-%m')
    if shared is not None and (not isinstance(shared, MonthlyBudget) or shared.month_id != month):
        raise ValueError('Shared budget must be the matching monthly ledger')
    with FileLock(str(directory / 'nonlinear.lock'), timeout=0):
        store = _Store(directory)
        try:
            state = _budget_state(store, month)
            if state is None:
                immutable_record(directory / 'budgets' / f'{month}.started.json',
                                 {'version': 'v7-nonlinear-budget-start-v1', 'month': month})
                shared = shared or MonthlyBudget(month)
                state = {'context': context, 'channel': ChannelBudget().to_dict(),
                         'shared': shared.to_dict(), 'attempts': {}}
            else:
                if state['context'] != context:
                    raise ValueError('Nonlinear month source/protocol/pool identity changed')
                saved = MonthlyBudget.from_dict(state['shared'])
                if shared is None:
                    shared = saved
                else:
                    _restore_shared(shared, saved)
            ledger = ChannelBudget.from_dict(state['channel'], shared=shared)

            def persist(value):
                state['channel'], state['shared'] = value, shared.to_dict()
                store.put('budget', month, state)

            ledger.persist = persist
            previous = state['attempts'].get(key)
            if previous:
                if previous['status'] == 'completed':
                    result = previous['result']
                    if receipt_kind:
                        _read(store, result, context, _now())
                    else:
                        receipt = _read(store, result['result_artifact'], context, _now())
                        if receipt['result'] != {k: v for k, v in result.items() if k != 'result_artifact'}:
                            raise ValueError('Residual result differs from its registered artifact')
                    return copy.deepcopy(result)
                if previous['status'] == 'active':
                    elapsed = max(0., (_now() - _stamp(previous['started_at'])).total_seconds())
                    previous['status'] = 'interrupted_consumed'
                    ledger.spent_seconds += elapsed
                    persist(ledger.to_dict())
                raise ValueError('Nonlinear attempt already consumed; no silent retry')
            state['attempts'][key] = {'status': 'active', 'started_at': _now().isoformat()}
            persist(ledger.to_dict())
            started = time.monotonic()
            initial_seconds = ledger.spent_seconds
            original_error = None
            try:
                ledger.begin()
                result = producer(ledger)
                # The final fit may do no further consume/check call. Its elapsed
                # work still counts, and it must not publish a completed artifact.
                ledger.check_time()
                if (ledger.spent_seconds >= ledger.max_seconds
                        or time.monotonic() - started >= ledger.max_seconds - initial_seconds):
                    raise ValueError('Nonlinear wall-time budget exhausted before completion')
                if receipt_kind:
                    result = _register(store, receipt_kind, context, result,
                                       started_at=state['attempts'][key]['started_at'])
                else:
                    receipt_id = _register(store, 'residual', context, result,
                                           test_only=result.get('test_only', False),
                                           started_at=state['attempts'][key]['started_at'])
                    result = {**result, 'result_artifact': receipt_id}
                state['attempts'][key].update(status='completed', result=result)
                return copy.deepcopy(result)
            except BaseException as exc:
                original_error = exc
                state['attempts'][key].update(status='failed_consumed', error=type(exc).__name__)
                raise
            finally:
                # Primitive fit timers already charged part of this interval.
                ledger.spent_seconds = max(ledger.spent_seconds,
                                          initial_seconds + time.monotonic() - started)
                try:
                    persist(ledger.to_dict())
                except BaseException as persistence_error:
                    if original_error is None:
                        raise
                    original_error.add_note('Nonlinear final checkpoint failed; prior active reservation remains '
                                            f'fail-closed: {type(persistence_error).__name__}')
        finally:
            store.close()


def run_gated_residual(directory, data, cutoff, pool, gate_artifacts, *, source_id,
                       config=None, budget=None):
    """Only runtime entry for enabled NN; never accepts a caller-supplied gate."""
    config = config or ResidualConfig()
    if not config.enabled:
        return {'status': 'disabled', 'formal_claim': False}
    if any(s.experts != 1 for s in config.structures):
        raise ValueError('Multiple experts are unsupported: no independently verified MoE gate')
    verified = verify_gate_artifacts(directory, data, cutoff, pool, gate_artifacts,
                                     source_id=source_id, config=config)
    context = verified['context']
    key = fingerprint({'kind': 'residual', 'context': context, 'config': config.digest,
                       'gate_id': verified['gate_id'],
                       'data': [data.training_identity(f['fit_cutoff'])
                                for f in development_folds(cutoff, spec=data.spec)]})

    def produce(ledger):
        gate = NeuralGate(**{k: True for k in GATE_FIELDS},
                          evidence_ids={k: verified['gate_id'] for k in GATE_FIELDS})
        result = research_residual(data, cutoff, pool, gate, config, ledger)
        return {**result, 'receipt_gate': verified, 'test_only': verified['test_only'],
                'formal_claim': False, 'runtime_admitted': not verified['test_only']}

    return _durable(directory, key, context, produce, budget)
