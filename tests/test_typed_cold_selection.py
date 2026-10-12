"""Staged typed preparation tests, artificial clocks/kernel; no V performance proof."""
import multiprocessing
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_v7_neural_research_ticket import _actual_child, _synthetic_kernel, launcher
from test_v7_nonlinear_channel import artificial_data
from test_v7_rolling_registry import candles

from factorlab.discovery import algorithm_validation as module
from factorlab.discovery import work_lease
from factorlab.discovery.algorithm_shortlist import AlgorithmCatalogue, AlgorithmShortlistRule
from factorlab.discovery.joint_pool import FrozenPool
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
from factorlab.discovery.neural_forward import FrozenNeuralRecipe
from factorlab.discovery.neural_research_ticket import issue_research_ticket
from factorlab.discovery.nonlinear_channel import ResidualConfig
from factorlab.discovery.nonlinear_orchestration import input_data_identity
from factorlab.discovery.rolling import RollingSpec
from factorlab.discovery.v7_forward import ForwardConfig, ForwardRuntime
from factorlab.discovery.v7_memory import search_exposure_context
from factorlab.discovery.v7_run import RunStore, V7Config, monthly_budget
from factorlab.discovery.work_authority_bridge import WorkAuthorityBridge
from factorlab.protocol import fingerprint


@pytest.fixture
def original_case(tmp_path, monkeypatch):
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    (tmp_path / 'factorlab').mkdir()
    (tmp_path / 'factorlab/frozen.py').write_text('source=1')
    (tmp_path / 'pyproject.toml').write_text('[project]')
    (tmp_path / 'uv.lock').write_text('version=1')
    data = artificial_data()
    # Genuine frozen production minimum=240, rather than silently changing the
    # future runtime to the small legacy S fixture's minimum_train_rows=12.
    metadata = data.frame.metadata.loc[data.frame.metadata.index.repeat(2)].reset_index(drop=True)
    times = pd.DatetimeIndex(metadata.available_at) - pd.to_timedelta(np.tile([6, 0], len(data.target)), unit='h')
    metadata['available_at'] = times
    metadata['forecast_window_start'] = times
    metadata['feature_open_time'] = times-pd.Timedelta(hours=1)
    metadata['sample_id'] = [f'synthetic-shared-{i}' for i in range(len(metadata))]
    values = np.zeros((len(metadata), data.frame.values.shape[1]))
    values[:, :8] = 1.
    values[:, 8] = np.repeat(data.frame.values[:, 8], 2)
    # Predeclared simple nonlinearity: this artificial positive signal supplies
    # a completed candidate, not evidence of performance on market data.
    target = np.exp(.8*np.tanh(2*values[:, 8]))
    data = replace(data, frame=replace(data.frame, values=values, metadata=metadata),
        target=target, observed=np.repeat(data.observed, 2),
        maturity=times+pd.Timedelta(hours=12), spec=RollingSpec())
    runtime = ForwardRuntime(tmp_path / 'forward', tmp_path / 'live-feed',
        config=ForwardConfig(symbols=('X',), archive_delay_seconds=data.spec.availability_delay_seconds))
    config = V7Config(evaluation_mode='joint', symbols=('X',))
    store = RunStore(tmp_path / 'research', config, data.spec, exposure_path=runtime.lifecycle.exposure.path)
    authority = MonthlyWorkAuthority(tmp_path / 'wallet', run_id='campaign', task='volatility', universe=['X'], partition_policy='standard')
    month, cutoff = '2025-04', '2025-04-01T00:00:00+00:00'
    exposed = search_exposure_context('controller-data', cutoff, ['X'])
    store.exposure.record(**{k: v for k, v in exposed.items() if k != 'data_id'}, role='S')
    state = {'status': 'running', 'cutoff': cutoff, 'data_id': 'controller-data',
        'search_exposure': exposed, 'budget': monthly_budget(config, month).to_dict(),
        'frozen_reference_pool': FrozenPool().to_dict(),
        'nonlinear_input_data_id': input_data_identity(data, cutoff)}
    store.save(month, state, 'frozen_controller_input')

    def persist(value):
        state['budget'] = value
        store.save(month, state, 'budget_projection')

    bridge = WorkAuthorityBridge(store, month, state, authority, persist)
    recipe = FrozenNeuralRecipe(rolling_spec=data.spec, config=ResidualConfig(enabled=True, epochs=1), output='encoder_ridge')
    gates = {'directory': str(tmp_path / 'artificial-gate-placeholder'),
             'artifacts': {'protocol': 'p', 'probes': ['q1', 'q2'], 'positive_control': 'c'}}
    monkeypatch.setattr('factorlab.discovery.neural_research_ticket._gates', lambda *args: {'private_test_seam': True})
    yield bridge, data, recipe, gates, runtime
    store.close()
    runtime.close()



import time
from factorlab.discovery import v7_service as service_module
from factorlab.discovery.neural_forward_ticket import ForwardFitTicket
from test_v7_algorithm_validation import completed

@pytest.fixture
def case(original_case):
    from factorlab.discovery.monthly_schedule import default_schedule
    from factorlab.discovery.scheduled_search import ScheduledSearch
    from factorlab.discovery.s_nonlinear_ticket import freeze_nonlinear_plan
    from factorlab.discovery.v7_run import V7Config
    bridge, data, recipe, gates, runtime = original_case
    config = V7Config(**{**bridge.store.identity['config'], 'symbols': tuple(bridge.store.identity['config']['symbols'])})
    ScheduledSearch(bridge, default_schedule(config, month=bridge.month))
    freeze_nonlinear_plan(bridge, recipe, kinds=('complete_neural_S',))
    bridge.begin_work('baseline', counts={'solve': 7}, lease_seconds=300, binding={'stage': 'baseline'}).complete({'artificial_owner_setup': True, 'training_performed': False})
    return bridge, data, recipe, gates, runtime


def _child(root, path, shift_seconds):
    import factorlab.protocol
    from factorlab.discovery import neural_forward_ticket
    factorlab.protocol.ROOT = Path(root)
    neural_forward_ticket._now = lambda: pd.Timestamp.now(tz='UTC')+pd.Timedelta(seconds=shift_seconds)
    producer = module._producer()
    original = work_lease._observe
    def observe(lease):
        value = original(lease)
        value['wall'] = (datetime.now(UTC)+pd.Timedelta(seconds=shift_seconds).to_pytimedelta()).timestamp()
        return value
    work_lease._observe = observe
    from factorlab.discovery import forward_service_scope
    forward_service_scope._observe = observe
    producer.verify_gates = lambda *a, **kw: None
    producer._kernel_snapshot = lambda: _synthetic_kernel([os.getpid(), os.getppid()])
    producer.worker(path)


def _future_clock(monkeypatch, runtime, start):
    # Import before monkeypatching, so teardown restores the physical observer.
    from factorlab.discovery import forward_service_scope
    clock = [start+pd.Timedelta(seconds=10)]
    original_issue, original_observe = work_lease.FrozenWorkLease.issue, work_lease._observe
    shifts = [pd.Timedelta(seconds=int((clock[0]-pd.Timestamp.now(tz='UTC')).total_seconds()))]
    def issue(seconds):
        lease = original_issue(seconds)
        next_shift = pd.Timedelta(seconds=int((clock[0]-pd.Timestamp.now(tz='UTC')).total_seconds()))
        # Calendar replay may advance, but issuing work cannot roll time back.
        shifts[0] = max(shifts[0], next_shift)
        return replace(lease, wall_started_at=(pd.Timestamp(lease.wall_started_at)+shifts[0]).isoformat(),
            wall_deadline=(pd.Timestamp(lease.wall_deadline)+shifts[0]).isoformat())
    def observe(lease):
        value = original_observe(lease)
        value['wall'] = (datetime.now(UTC)+shifts[0].to_pytimedelta()).timestamp()
        return value
    monkeypatch.setattr(work_lease.FrozenWorkLease, 'issue', issue)
    monkeypatch.setattr(work_lease, '_observe', observe)
    monkeypatch.setattr(forward_service_scope, '_observe', observe)
    class FutureDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.now(tz)+shifts[0].to_pytimedelta()
    monkeypatch.setattr('factorlab.discovery.monthly_work_authority.datetime', FutureDatetime)
    monkeypatch.setattr(module, '_now', lambda: clock[0])
    monkeypatch.setattr('factorlab.discovery.neural_forward_ticket._now', lambda: clock[0])
    runtime.clock = runtime.lifecycle.clock = runtime.feed.clock = lambda: clock[0]
    producer = module._producer()
    original_observation = producer._observation
    def observation(lease):
        value = original_observation(lease)
        value['wall'] = (pd.Timestamp(value['wall'])+shifts[0]).isoformat()
        return value
    monkeypatch.setattr(producer, '_observation', observation)
    return clock, shifts


def _seed(runtime, clock, start):
    clock[0] = start-pd.Timedelta(minutes=30)
    bars = candles(start-pd.DateOffset(months=5), start-pd.Timedelta(hours=1))
    bars['symbol'] = 'X'
    path = runtime.directory.parent/'actual-artificial-seed.parquet'
    bars.to_parquet(path)
    runtime.feed.import_seed(path)
    clock[0] = start+pd.Timedelta(seconds=10)


@pytest.fixture
def tmp_path(tmp_path_factory):
    path = tmp_path_factory.mktemp('ts')
    return path


def test_actual_typed_V_winner_feeds_complete_cold_recipe(completed, monkeypatch):  # noqa: F811
    bridge, catalogue, recipe, runtime, _ = completed
    start = module._now().normalize().replace(day=1)+pd.DateOffset(months=1)
    protocol = module.AlgorithmValidation.freeze(runtime, bridge.authority, catalogue, start=start,
        rule=module.AlgorithmValidationRule(minimum_coverage=.0001, minimum_gain=0.))
    producer = module._producer()
    clock, shift = _future_clock(monkeypatch, runtime, start)
    _seed(runtime, clock, start)
    monkeypatch.setattr(producer, 'verify_gates', lambda *a, **kw: None)
    monkeypatch.setattr(producer, '_strict_kernel', lambda *a, **kw: None)
    children = []
    class ActualChild:
        def __init__(self, command, **kwargs):
            assert Path(command[2]).name == 'algorithm_validation_producer.py'
            assert kwargs['env']['OMP_NUM_THREADS'] == '1'
            self.process = multiprocessing.get_context('spawn').Process(target=_child,
                args=(str(runtime.directory.parent), command[-1], shift[0].total_seconds()))
            self.process.start()
            self.pid = self.process.pid
            children.append(self)
        @property
        def returncode(self):
            return self.process.exitcode
        def poll(self):
            return self.process.exitcode
    monkeypatch.setattr(producer.subprocess, 'Popen', ActualChild)
    monkeypatch.setattr(producer, '_kernel_snapshot', lambda: _synthetic_kernel(
        [os.getpid()]+[c.pid for c in children if c.poll() is None]))
    protocol.register_fit_gates(recipe.pipeline_id, start,
        gates={'directory': str(runtime.directory/'private-gates'), 'artifacts': {}})
    service = service_module.ForwardService(runtime, work_authority=bridge.authority,
        config=service_module.ServiceConfig(poll=False, wall_seconds=7200), guard=lambda: {'allowed': True})
    try:
        response = service.step()
        controller = next(a for a in response['actions'] if a['kind'] == 'algorithm_controller')
        result = next(a for a in controller['actions'] if a['kind'] == 'algorithm_fit' and a.get('status') == 'completed')
        assert len([k for k in service.state['fit_attempts'] if k.startswith('algorithm:')]) == 1
        prepared = protocol.read_prepared(recipe.pipeline_id, start, statuses=('completed',))
        before = bridge.authority.status(start.strftime('%Y-%m'))
        actions = []
        service._fit_pending(clock[0], actions, time.monotonic()+300, fit_allowance=0)
        assert any(a['kind'] == 'algorithm_owned_job' and a['job'] == prepared['job'] for a in actions)
        assert bridge.authority.status(start.strftime('%Y-%m')) == before
        assert runtime.lifecycle.db.execute('SELECT failure FROM jobs WHERE id=?', (prepared['job'],)).fetchone() == (None,)
    finally:
        # Keep fixture-owned runtime open for readonly artifact/endpoint checks.
        service.db.close()
        service._owner_lock.release()
    assert result['shadow_only'] and not result['serving_eligible']
    with pytest.raises(ValueError):
        ForwardFitTicket.load(result['ticket'])
    admitted = producer.admit(protocol, recipe.pipeline_id, start)
    counts = bridge.authority.get(prepared['reservation_id'])['actual_counts']
    assert counts['neural_fit'] == 3 and counts['epoch'] == 3 and counts['solve'] == 8
    with pytest.raises(ValueError):
        protocol.launch_fit(prepared)
    # Actual new receipt; prior assumed seed produces no old origins.
    clock[0] = start+pd.Timedelta(hours=1, seconds=1)
    bars = candles(start-pd.DateOffset(months=5), start+pd.Timedelta(hours=1)).tail(2).copy()
    bars['symbol'] = 'X'
    runtime.feed.receive(bars, received_at=clock[0], response_hash=fingerprint('actual-source-one'))
    assert protocol.tick()['predictions'] == 1
    assert protocol.tick()['predictions'] == 0
    forecasts = [e['payload'] for e in producer._events(protocol) if e['kind'] == 'forecast']
    assert len(forecasts) == 1 and set(forecasts[0]['predictions']) == {recipe.pipeline_id, 'K0', 'Pv'}
    assert forecasts[0]['tickets'] == {recipe.pipeline_id: admitted['ticket_digest']}
    # Synthetic positive-control future path; no score/certificate is supplied.
    # The actual producer must still derive the target from 13 received OHLCs.
    target = forecasts[0]['predictions'][recipe.pipeline_id]
    clock[0] = start+pd.Timedelta(hours=14, seconds=1)
    future = candles(start+pd.Timedelta(hours=1), start+pd.Timedelta(hours=14))
    prices = np.exp(4+np.arange(len(future))*target/np.sqrt(12))
    future['symbol'], future['open'], future['close'] = 'X', prices, prices
    future['high'], future['low'] = prices*1.003, prices/1.003
    runtime.feed.receive(future, received_at=clock[0], response_hash=fingerprint('actual-source-label'))
    assert protocol.tick()['settlements'] == 1
    with pytest.raises(ValueError, match='not matured'):
        protocol.select_validation()
    clock[0] = pd.Timestamp(protocol.identity['V_scoring_cutoff'])
    endpoint = protocol.select_validation()
    assert endpoint['score']['common_pairs'] == 1 and endpoint['score']['nominal_hours'] >= 672
    assert endpoint['formal_claim'] is False and endpoint['serving_eligible'] is False
    assert endpoint['score']['selected'] == recipe.pipeline_id, forecasts[0]['predictions']
    assert protocol.select_validation() == endpoint
    from factorlab.discovery.typed_cold_selection import _selected_protocol
    selected = _selected_protocol(protocol)
    assert selected['candidate'] == runtime.recipe_for(recipe)
    assert selected['reference'] == protocol.identity['references']['Pv']
    assert selected['start'] == protocol.identity['qualifier_start']
    assert selected['original_estimand']['winner'] == recipe.pipeline_id
    assert not selected['original_estimand']['full_adaptive_A0_truth']
    anchor = bridge.authority.directory/'algorithm_validation_events'/protocol.digest
    target = sorted(anchor.glob('*.json'), key=lambda p:int(p.stem))[-1]
    raw = target.read_bytes(); target.unlink()
    with pytest.raises(ValueError, match='anchors lost'):
        _selected_protocol(protocol)
    target.write_bytes(raw)
    assert _selected_protocol(protocol) == selected
