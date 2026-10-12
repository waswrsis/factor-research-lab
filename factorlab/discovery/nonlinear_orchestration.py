"""Controller-owned nonlinear S research; no automatic pool/forecast adoption.

The monthly controller retains its exclusive RunStore lock. Its existing
DurableBudget callback commits BOTH channel reservations and the global solve
budget in one checkpoint. No second monthly wallet or caller-supplied gate is
created. K0 epoch recipes stay fixed across quarterly evidence collection.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from factorlab.protocol import fingerprint, project_path

from .frame_identity import frame_identity
from .joint_pool import FrozenPool
from .nonlinear_channel import ChannelBudget, ProbeConfig, ResidualConfig, StructureSpec, run_probe
from .nonlinear_runtime import (
    _common,
    _context,
    _read,
    _register,
    _Store,
    freeze_protocol,
    record_resource_receipt,
    verify_gate_artifacts,
    verify_pre_resource_artifacts,
)
from .v7_storage import immutable_record


def _now():
    return pd.Timestamp.now(tz='UTC')


def _config(value):
    value = dict(value)
    value['structures'] = tuple(StructureSpec(**s) for s in value['structures'])
    value['seeds'] = tuple(value['seeds'])
    return ResidualConfig(**value)


def epoch_pool():
    return FrozenPool(version='nonlinear_K0_epoch_v1', role='reference')


def causal_worker_data(data, cutoff):
    """Transfer only past inputs and labels actually mature by this cutoff."""
    from .known_library import FactorFrame
    from .rolling import RollingData
    end = pd.Timestamp(cutoff)
    if end.tzinfo is None or pd.isna(end):
        raise ValueError('Neural input cutoff requires an actual UTC boundary')
    selected = (data.frame.metadata.available_at < end).to_numpy()
    frame = data.frame.take(selected)
    observed = data.observed[selected] & np.asarray(data.maturity[selected] <= end)
    target = np.where(observed, data.target[selected], 0.)
    maturity = pd.DatetimeIndex(data.maturity[selected]).where(observed, pd.NaT)
    metadata = frame.metadata.copy()
    if 'label_status' in metadata:
        metadata.loc[~observed, 'label_status'] = 'unavailable_asof_cutoff'
    frame = FactorFrame(frame.values, metadata, frame.feature_names, frame.bypass_mask, frame.schema_hash)
    return RollingData(frame, target, maturity, observed, data.spec, data.availability_mode)


def input_data_identity(data, cutoff):
    """Component identity frozen by the controller, never supplied by a model."""
    data = causal_worker_data(data, cutoff)
    return fingerprint({'cutoff': pd.Timestamp(cutoff).isoformat(), 'rolling': data.spec.digest,
        'availability_mode': data.availability_mode, 'identity_rule': 'typed-frame-columns-v1',
        'metadata': frame_identity(data.frame.metadata),
        'feature_names': list(data.frame.feature_names), 'bypass_mask': list(data.frame.bypass_mask),
        'schema': data.frame.schema_hash, 'finite': np.isfinite(data.frame.values).tolist(),
        'values': np.nan_to_num(data.frame.values, nan=0.).tolist(),
        'target': data.target.tolist(), 'observed': data.observed.tolist(),
        'maturity': frame_identity(pd.DataFrame({'maturity': data.maturity}))})


@dataclass(frozen=True)
class NonlinearEpoch:
    value: dict

    def __post_init__(self):
        value = self.value
        if (value.get('version') != 'v7-nonlinear-epoch-v1'
                or value.get('pool') != epoch_pool().to_dict()
                or value.get('task') != 'volatility12h'
                or value.get('probe_config') != asdict(ProbeConfig())
                or value.get('adoption') != 'unregistered_neural_recipe_never_auto_adopt'):
            raise ValueError('Invalid frozen K0 nonlinear epoch')
        config = _config(value['residual_config'])
        if not config.enabled or any(s.experts != 1 for s in config.structures):
            raise ValueError('Epoch permits only bounded single-expert residual structures')

    @property
    def digest(self):
        return fingerprint(self.value)

    @property
    def config(self):
        return _config(self.value['residual_config'])


def freeze_epoch(directory, bridge, data, cutoff, *, config=None):
    bridge.verify()
    _verify_input(bridge, data, cutoff)
    config = config or ResidualConfig(enabled=True)
    directory = project_path(directory)
    path = directory / 'epoch.json'
    if path.exists():
        record = json.loads(path.read_text('utf-8'))
        epoch = NonlinearEpoch(record['value'])
        if record['sha256'] != epoch.digest or epoch.config.digest != config.digest:
            raise ValueError('Nonlinear epoch changed')
        _verify_epoch(epoch, bridge, data)
        return epoch
    protocol = freeze_protocol(directory, data, cutoff, epoch_pool(),
                               source_id=bridge.source_id, config=config)
    epoch = NonlinearEpoch({'version': 'v7-nonlinear-epoch-v1', 'task': 'volatility12h',
        'source_id': bridge.source_id, 'source_hash': fingerprint(bridge.store.identity['source_files']),
        'rolling_id': data.spec.digest, 'input_schema': data.frame.schema_hash,
        'pool': epoch_pool().to_dict(), 'protocol_receipt': protocol,
        'probe_config': asdict(ProbeConfig()), 'residual_config': asdict(config),
        'created_at': _now().isoformat(), 'adoption': 'unregistered_neural_recipe_never_auto_adopt'})
    immutable_record(path, {'value': epoch.value, 'sha256': epoch.digest})
    return epoch


def _verify_epoch(epoch, bridge, data):
    _verify_input(bridge, data, bridge.state['cutoff'])
    if (epoch.value['source_id'] != bridge.source_id
            or epoch.value['source_hash'] != fingerprint(bridge.store.identity['source_files'])
            or epoch.value['rolling_id'] != data.spec.digest
            or epoch.value['input_schema'] != data.frame.schema_hash):
        raise ValueError('Epoch source, protocol or input differs from the monthly S owner')


def _verify_input(bridge, data, cutoff):
    if (pd.Timestamp(cutoff).isoformat() != bridge.state['cutoff']
            or bridge.state.get('nonlinear_input_data_id') != input_data_identity(data, cutoff)):
        raise ValueError('Nonlinear data snapshot differs from the controller input identity')


class RunStoreBudgetBridge:
    """Use under the monthly controller's existing exclusive run lock.

    `state` must be the SAME live dictionary captured by DurableBudget.persist.
    Every commit verifies it reached that RunStore. A detached or no-op callback
    fails before any fit. Completed/finalizing months cannot be reopened here.
    """
    def __init__(self, store, month, state, budget):
        from .v7_run import DurableBudget
        if not isinstance(budget, DurableBudget):
            raise TypeError('Nonlinear production requires the RunStore DurableBudget owner')
        self.store, self.month, self.state, self.budget = store, month, state, budget
        self.verify()

    @property
    def source_id(self):
        return self.state['search_exposure']['source']

    def verify(self):
        from .v7_run import source_files
        self.store.verify()
        if (source_files() != self.store.identity['source_files']
                or self.state.get('status') != 'running' or self.budget.month_id != self.month
                or pd.Timestamp(self.state['cutoff']).strftime('%Y-%m') != self.month
                or self.state.get('budget') != self.budget.to_dict()
                or self.store.get(self.month) != self.state):
            raise ValueError('Nonlinear bridge requires the exact active monthly checkpoint')
        exposed = self.state.get('search_exposure', {})
        if exposed.get('data_id') != self.state.get('data_id'):
            raise ValueError('Nonlinear research data differs from its S exposure binding')
        self.store.exposure.assert_prompt_search(**{k: v for k, v in exposed.items() if k != 'data_id'})

    def commit(self):
        self.budget.persist(self.budget.to_dict())
        if self.store.get(self.month) != self.state:
            raise ValueError('Nonlinear reservation did not reach the authoritative RunStore')

    def section(self, epoch):
        self.verify()
        if 'nonlinear' not in self.state:
            self.state['nonlinear'] = {'epoch_id': epoch.digest, 'channel': ChannelBudget().to_dict(),
                                       'attempts': {}, 'role': 'S', 'automatic_adoption': False}
            self.commit()
        section = self.state['nonlinear']
        if section['epoch_id'] != epoch.digest:
            raise ValueError('A monthly nonlinear epoch cannot be replaced')
        return section

    def channel(self, epoch):
        bridge = self
        section = self.section(epoch)

        class AtomicChannelBudget(ChannelBudget):
            def consume(self, kind, amount=1):
                self.check_time()
                if (kind not in self.limits or type(amount) is not int or amount < 1
                        or self.counts[kind] + amount > self.limits[kind]
                        or self.spent_seconds >= self.max_seconds):
                    raise ValueError('Independent nonlinear budget exhausted')
                before = self.counts[kind]
                global_before = bridge.budget.counts['solve']
                self.counts[kind] += amount
                section['channel'] = self.to_dict()
                try:
                    if kind == 'solve':
                        # The global callback saves the already-updated channel
                        # field and its global increment in the SAME transaction.
                        bridge.budget.consume('solve', amount)
                        if bridge.store.get(bridge.month) != bridge.state:
                            raise ValueError('Global reservation was not atomically persisted')
                    else:
                        bridge.commit()
                except BaseException:
                    if kind == 'solve' and bridge.budget.counts['solve'] == global_before:
                        self.counts[kind] = before
                        section['channel'] = self.to_dict()
                    raise

        def persist(value):
            section['channel'] = value
            bridge.commit()

        return AtomicChannelBudget.from_dict(section['channel'], persist=persist)


def _produce(bridge, directory, data, cutoff, epoch, kind, producer):
    bridge.verify()
    _verify_epoch(epoch, bridge, data)
    if pd.Timestamp(cutoff).isoformat() != bridge.state['cutoff'] or pd.Timestamp(cutoff) > _now():
        raise ValueError('Producer must use the current completed as-of S cutoff')
    section = bridge.section(epoch)
    context = _context(data, cutoff, epoch_pool(), bridge.source_id)
    key = fingerprint({'epoch': epoch.digest, 'kind': kind, 'context': context,
                       'controller_data_id': bridge.state['data_id']})
    previous = section['attempts'].get(key)
    store = _Store(directory)
    try:
        if previous is not None:
            if previous['status'] == 'completed':
                _read(store, previous['receipt_id'], context, _now())
                return {'status': 'completed', 'receipt_id': previous['receipt_id'], 'cached': True}
            if previous['status'] == 'active':
                previous['status'] = 'interrupted_consumed'
                # Reserve the remaining channel time after unobserved process
                # death. No deadline refund or automatic retry in this month.
                section['channel']['spent_seconds'] = max(section['channel']['spent_seconds'],
                                                           section['channel']['max_seconds'])
                bridge.commit()
            return {'status': previous['status'], 'receipt_id': None, 'cached': True}
        attempt = {'status': 'active', 'kind': kind, 'started_at': _now().isoformat()}
        section['attempts'][key] = attempt
        bridge.commit()
        ledger, started = bridge.channel(epoch), time.monotonic()
        initial_seconds, original_error = ledger.spent_seconds, None
        try:
            ledger.begin()
            result = producer(ledger)
            ledger.check_time()
            if time.monotonic() - started >= ledger.max_seconds - initial_seconds:
                raise ValueError('Nonlinear channel deadline consumed')
            result = {**result, 'producer_binding': {'epoch_id': epoch.digest,
                'run_directory': str(bridge.store.directory), 'month': bridge.month,
                'controller_data_id': bridge.state['data_id'], 'attempt_id': key}}
            identifier = _register(store, kind, context, result, started_at=attempt['started_at'])
            attempt.update(status='completed', receipt_id=identifier)
            return {'status': 'completed', 'receipt_id': identifier, 'cached': False}
        except BaseException as exc:
            original_error = exc
            attempt.update(status='failed_consumed', error_type=type(exc).__name__, error=str(exc)[:400])
            raise
        finally:
            ledger.spent_seconds = max(ledger.spent_seconds, initial_seconds + time.monotonic() - started)
            section['channel'] = ledger.to_dict()
            try:
                bridge.commit()
            except BaseException as persistence_error:
                if original_error is None:
                    raise
                original_error.add_note('Nonlinear final checkpoint failed: ' + type(persistence_error).__name__)
    finally:
        store.close()


def produce_quarterly_artifacts(directory, bridge, data, cutoff, epoch):
    """Actual fixed tree and fixed positive control, using global S reservations."""
    if (pd.Timestamp(cutoff).month - ProbeConfig().anchor_month) % 3:
        return {'status': 'not_due', 'formal_claim': False}
    from .v7_controls import known_component_recovery_probe

    probe = _produce(bridge, directory, data, cutoff, epoch, 'probe',
                     lambda ledger: run_probe(data, cutoff, epoch_pool(), ledger))

    def control(ledger):
        ledger.consume('solve', 6)
        return known_component_recovery_probe(data, cutoff, component='rv_24')

    positive = _produce(bridge, directory, data, cutoff, epoch, 'positive_control', control)
    return {'status': 'completed' if probe['status'] == positive['status'] == 'completed' else 'incomplete',
            'probe': probe, 'positive_control': positive, 'usable_from':
            (pd.Timestamp(cutoff) + pd.DateOffset(months=1)).isoformat(),
            'automatic_adoption': False, 'formal_claim': False}


def reserve_neural_ticket(bridge, epoch, gate_id, *, mode='residual', seconds=300):
    """Escrow exact maximum solves before isolated work; unused funds stay spent."""
    section = bridge.section(epoch)
    if mode not in {'resource_trial', 'residual'}:
        raise ValueError('Unknown isolated neural work mode')
    # An unsuccessful experiment cannot become new work by changing its wall
    # allowance or choosing a different equivalent set of gate receipt IDs.
    for key, previous in section['attempts'].items():
        saved = previous.get('ticket', {})
        if saved.get('mode') == mode and saved.get('epoch_id') == epoch.digest:
            return {'status': 'already_consumed', 'ticket_id': key}
    if type(seconds) is not int or not 1 <= seconds <= 300:
        raise ValueError('Neural tickets reserve at most 300 whole seconds')
    config = epoch.config
    fits = 3 * len(config.structures) * len(config.seeds)
    ticket = {'version': 'v7-neural-ticket-v1', 'epoch_id': epoch.digest, 'mode': mode,
              'gate_id': gate_id, 'month': bridge.month, 'cutoff': bridge.state['cutoff'],
              'controller_data_id': bridge.state['data_id'],
              'input_data_id': bridge.state['nonlinear_input_data_id'],
              'run_directory': str(bridge.store.directory),
              'limits': {'solve': 3 + 2 * fits, 'neural_fit': fits, 'epoch': fits * config.epochs},
              'config_id': config.digest, 'seconds': seconds, 'refundable': False}
    key = fingerprint(ticket)
    if key in section['attempts']:
        return {'status': 'already_consumed', 'ticket_id': key}
    ledger = bridge.channel(epoch)
    if ledger.spent_seconds + seconds > ledger.max_seconds:
        raise ValueError('Insufficient independent wall budget for complete neural ticket')
    for kind, amount in ticket['limits'].items():
        if ledger.counts[kind] + amount > ledger.limits[kind]:
            raise ValueError('Insufficient independent budget for complete neural ticket')
    if bridge.budget.counts['solve'] + ticket['limits']['solve'] > bridge.budget.limits['solve']:
        raise ValueError('Insufficient global budget for complete neural ticket')
    previous_channel = copy.deepcopy(section['channel'])
    global_before = bridge.budget.counts['solve']
    section['attempts'][key] = {'status': 'reserved', 'kind': 'neural_ticket', 'ticket': ticket}
    for kind, amount in ticket['limits'].items():
        ledger.counts[kind] += amount
    ledger.spent_seconds += seconds
    section['channel'] = ledger.to_dict()
    try:
        bridge.budget.consume('solve', ticket['limits']['solve'])
    except BaseException:
        if bridge.budget.counts['solve'] == global_before:
            section['channel'] = previous_channel
            del section['attempts'][key]
        raise
    if bridge.store.get(bridge.month) != bridge.state:
        raise ValueError('Neural ticket was not atomically committed')
    return {'status': 'reserved', 'ticket_id': key, 'ticket': copy.deepcopy(ticket)}


def structure_proposals(epoch):
    """Only the epoch's actually executable structures, never a pool replacement."""
    return [{'structure_id': spec.digest, 'structure': asdict(spec), 'config_id': epoch.config.digest,
             'evaluation': 'three_cold_folds_common_seeds_epoch0_and_same_convex_readout',
             'recipe_status': 'neural_forward_registry_not_registered', 'automatic_adoption': False}
            for spec in epoch.config.structures]


def prior_gate_artifacts(directory, bridge, data, cutoff, epoch):
    """Last two scheduled quarters only; no cherry-picking successful quarters."""
    _verify_epoch(epoch, bridge, data)
    bridge.verify()
    end = pd.Timestamp(cutoff)
    dates = pd.date_range(end - pd.DateOffset(months=12), end, freq='MS')
    expected = [point.isoformat() for point in dates if point < end and point.month in (1, 4, 7, 10)][-2:]
    context = _context(data, cutoff, epoch_pool(), bridge.source_id)
    matches = {'probe': {}, 'positive_control': {}}
    store = _Store(directory)
    try:
        for path in sorted((store.directory / 'receipts').glob('*.json')):
            value = json.loads(path.read_text('utf-8'))
            if value.get('kind') not in matches or value['context']['cutoff'] not in expected:
                continue
            if pd.Timestamp(value['available_at']) > end:
                continue
            value = _read(store, path.stem, context, end)
            producer = value['result'].get('producer_binding')
            if not producer or producer['epoch_id'] != epoch.digest:
                raise ValueError('Quarterly evidence lacks its actual controller producer binding')
            if project_path(producer['run_directory']) != bridge.store.directory:
                raise ValueError('Quarterly evidence belongs to another monthly research run')
            origin = bridge.store.get(producer['month'])
            attempt = (origin or {}).get('nonlinear', {}).get('attempts', {}).get(producer['attempt_id'])
            if (not attempt or attempt.get('status') != 'completed' or attempt.get('receipt_id') != path.stem
                    or origin['data_id'] != producer['controller_data_id']):
                raise ValueError('Quarterly evidence has no completed nonrefundable S work unit')
            exposed = origin['search_exposure']
            bridge.store.exposure.assert_prompt_search(**{k: v for k, v in exposed.items() if k != 'data_id'})
            point = value['context']['cutoff']
            if point in matches[value['kind']]:
                raise ValueError('Multiple results for one frozen quarterly producer')
            matches[value['kind']][point] = path.stem
    finally:
        store.close()
    if len(expected) != 2 or any(point not in matches['probe'] for point in expected):
        return {'status': 'pending_quarterly_evidence', 'required_quarters': expected}
    latest = expected[-1]
    if latest not in matches['positive_control']:
        return {'status': 'pending_positive_control', 'required_quarter': latest}
    artifacts = {'protocol': epoch.value['protocol_receipt'],
                 'probes': [matches['probe'][point] for point in expected],
                 'positive_control': matches['positive_control'][latest]}
    try:
        verified = verify_pre_resource_artifacts(directory, data, cutoff, epoch_pool(), artifacts,
                                                 source_id=bridge.source_id, config=epoch.config)
    except ValueError as exc:
        return {'status': 'closed_research_gate', 'reason': str(exc)[:400]}
    if verified['test_only']:
        return {'status': 'closed_test_only_evidence'}
    return {'status': 'research_evidence_verified', 'artifacts': artifacts, 'verified': verified}


def run_isolated_research(directory, bridge, data, cutoff, epoch, *, resource_receipt_id=None,
                          resource_trial=False):
    """Gate-first isolated execution. Returned NN recipes are never A0 pools."""
    from . import nonlinear_launcher as launcher
    gate = prior_gate_artifacts(directory, bridge, data, cutoff, epoch)
    if gate['status'] != 'research_evidence_verified':
        return {**gate, 'automatic_adoption': False}
    artifacts = gate['artifacts']
    data = causal_worker_data(data, cutoff)
    mode = 'resource_trial' if resource_trial else 'residual'
    if not resource_trial:
        if resource_receipt_id is None:
            return {'status': 'pending_real_linux_resource_receipt', 'automatic_adoption': False}
        artifacts = {**artifacts, 'resource': resource_receipt_id}
        try:
            verified = verify_gate_artifacts(directory, data, cutoff, epoch_pool(), artifacts,
                                             source_id=bridge.source_id, config=epoch.config)
        except (ValueError, OSError) as exc:
            return {'status': 'closed_resource_gate', 'reason': str(exc)[:400], 'automatic_adoption': False}
        if verified['test_only']:
            return {'status': 'closed_test_only_evidence', 'automatic_adoption': False}
    available = launcher.availability()
    if available['status'] != 'available':
        return {**available, 'automatic_adoption': False}
    remaining = int(bridge.store.identity['config']['research_seconds'] - bridge.state.get('spent_seconds', 0.))
    if remaining < 1:
        return {'status': 'monthly_deadline_consumed', 'automatic_adoption': False}
    seconds = min(300, remaining)
    ticket = reserve_neural_ticket(bridge, epoch, fingerprint(artifacts), mode=mode, seconds=seconds)
    if ticket['status'] != 'reserved':
        return {**ticket, 'automatic_adoption': False}
    launch_directory = bridge.store.directory / 'nonlinear_launches' / ticket['ticket_id']
    attempt = bridge.state['nonlinear']['attempts'][ticket['ticket_id']]
    try:
        from .nonlinear_runtime import workload_capacity
        context = _context(data, cutoff, epoch_pool(), bridge.source_id)
        data_record = launcher.save_worker_data(launch_directory, data, cutoff=cutoff)
        capacity = workload_capacity(data, cutoff)
        job = {'version': 'v7-fixed-neural-worker-v1', 'mode': mode,
               'launch_directory': str(launch_directory), 'ticket': ticket['ticket'],
               'ticket_id': ticket['ticket_id'], 'context': context, 'data': data_record,
               'config': asdict(epoch.config), 'rolling': asdict(data.spec), 'pool': epoch_pool().to_dict(),
               'artifact_directory': str(project_path(directory)), 'gate_artifacts': artifacts,
               'cutoff': pd.Timestamp(cutoff).isoformat(), 'capacity': capacity, 'seconds': seconds,
               'resource_context': {'context': _common(context), 'config_id': epoch.config.digest}}
        outcome = launcher.launch(launch_directory, job)
        if outcome['status'] == 'completed':
            attempt.update(status='completed', launcher_receipt=outcome['receipt'])
            bridge.commit()
            if resource_trial:
                identifier = record_resource_receipt(directory, data, cutoff, epoch_pool(),
                    launcher.artifact_from_receipt(outcome['receipt']), source_id=bridge.source_id,
                    config=epoch.config)
                attempt['resource_receipt_id'] = identifier
                bridge.commit()
            return {**outcome, 'resource_receipt_id': attempt.get('resource_receipt_id'),
                    'recipe_status': 'neural_forward_registry_not_registered',
                    'automatic_adoption': False, 'formal_claim': False}
        attempt.update(status='failed_consumed', outcome=outcome)
        bridge.commit()
        return {**outcome, 'automatic_adoption': False}
    except BaseException as exc:
        attempt.update(status='failed_consumed', error_type=type(exc).__name__, error=str(exc)[:400])
        try:
            bridge.commit()
        except BaseException as persistence_error:  # noqa: BLE001 - preserve the original failure, never refund
            exc.add_note('Neural failure checkpoint failed: ' + type(persistence_error).__name__)
        raise
