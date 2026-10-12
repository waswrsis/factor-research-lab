"""Actual new S completion can join the same bounded shadow service session."""
import copy

import pandas as pd
import pytest
from test_v7_algorithm_validation import case, completed, module  # noqa: F401
from test_v7_algorithm_validation_producer import service_module
from test_v7_rolling_registry import candles

from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority


def _service(runtime, authority):
    return service_module.ForwardService(runtime, work_authority=authority,
        config=service_module.ServiceConfig(session_id='continuous-shadow', poll=False, wall_seconds=7200),
        guard=lambda: {'allowed': True})


def _seed_current(runtime):
    now = pd.Timestamp.now(tz='UTC')
    bars = candles(now.normalize()-pd.DateOffset(months=5), now.floor('h')-pd.Timedelta(hours=1))
    bars['symbol'] = 'X'
    path = runtime.directory.parent/'continuous-initial-seed.parquet'
    bars.to_parquet(path)
    runtime.feed.import_seed(path)


def test_service_max_four_accepts_actual_prospective_protocol_without_new_work(completed):
    bridge, catalogue, _, runtime, _ = completed
    start = module._now().normalize().replace(day=1)+pd.DateOffset(months=1)
    module.AlgorithmValidation.freeze(runtime, bridge.authority, catalogue, start=start)
    runner = service_module.ForwardService(runtime, work_authority=bridge.authority,
        config=service_module.ServiceConfig(session_id='max-four-shadow', poll=False,
            wall_seconds=7200, max_fits_per_step=4), guard=lambda: {'allowed': True})
    try:
        before = bridge.authority.verify()['reservations']
        actions = []
        assert runner._algorithm_step(actions, runner.monotonic()+600) == 0
        assert len(actions) == 1
        assert not runner.state['fit_attempts']
        assert bridge.authority.verify()['reservations'] == before
    finally:
        runner.close(close_runtime=False)


def test_three_protocol_queues_and_ordinary_work_share_four_attempts(case, monkeypatch):  # noqa: F811
    """Private queues isolate dispatch arithmetic; they are NOT evidence issuers."""
    from types import SimpleNamespace
    from factorlab.discovery import prospective_resource_work
    # This test supplies deliberately thin private queues to isolate the shared
    # attempt arithmetic. Resource production has its own genuine-protocol
    # tests; do not pass these fakes into its strict canonical authority reader.
    monkeypatch.setattr(prospective_resource_work, 'pending_prospective_resource', lambda protocol: [])
    bridge, _, _, _, runtime = case
    _seed_current(runtime)
    runner = service_module.ForwardService(runtime, work_authority=bridge.authority,
        config=service_module.ServiceConfig(session_id='four-shared-attempts', poll=False,
            wall_seconds=7200, max_fits_per_step=4), guard=lambda: {'allowed': True})
    offered, ordinary = [], []
    def queue(name):
        def dispatch(*, deadline_monotonic, max_fits):
            offered.append((name, max_fits))
            return {'actions': [], 'formal_claim': False}
        return SimpleNamespace(digest=name, resolve_fit_gates=lambda: None,
            pending_work=lambda: [{'work_key': f'{name}-{n}'} for n in range(3)], step=dispatch)
    monkeypatch.setattr(runner, '_sync_algorithm_protocols', lambda: [queue(n) for n in ('a', 'b', 'c')])
    monkeypatch.setattr(runner, '_fit_pending', lambda *a, fit_allowance: ordinary.append(fit_allowance))
    try:
        runner.step()
        assert offered == [('a', 3), ('b', 1), ('c', 0)]
        assert ordinary == [0]
        assert set(runner.state['fit_attempts']) == {'algorithm:a-0', 'algorithm:a-1', 'algorithm:a-2', 'algorithm:b-0'}
    finally:
        runner.close(close_runtime=False)


def test_empty_inspected_queue_cannot_gain_budget_during_controller_reinspection(case, monkeypatch):  # noqa: F811
    """Clock/gate changes after the queue snapshot cannot create new admission."""
    from types import SimpleNamespace
    from factorlab.discovery import prospective_resource_work
    # This test supplies deliberately thin private queues to isolate the shared
    # attempt arithmetic. Resource production has its own genuine-protocol
    # tests; do not pass these fakes into its strict canonical authority reader.
    monkeypatch.setattr(prospective_resource_work, 'pending_prospective_resource', lambda protocol: [])
    bridge, _, _, _, runtime = case
    runner = _service(runtime, bridge.authority)
    offered = []
    protocol = SimpleNamespace(digest='private-empty-queue', resolve_fit_gates=lambda: None, pending_work=list,
        step=lambda **kw: offered.append(kw['max_fits']) or {'actions': []})
    monkeypatch.setattr(runner, '_sync_algorithm_protocols', lambda: [protocol])
    try:
        assert runner._algorithm_step([], runner.monotonic()+600) == 0
        assert offered == [0]
        assert not runner.state['fit_attempts']
    finally:
        runner.close(close_runtime=False)


def test_new_real_S_completion_future_V_same_session_then_restart(case, monkeypatch):  # noqa: F811
    bridge, _, _, _, runtime = case
    _seed_current(runtime)
    runner = _service(runtime, bridge.authority)
    assert runner.state['algorithm_protocols'] == {}
    # S actually runs after this service session already exists.
    completion = completed.__wrapped__(case, monkeypatch)
    values = next(completion)
    try:
        catalogue = values[1]
        start = module._now().normalize().replace(day=1)+pd.DateOffset(months=1)
        protocol = module.AlgorithmValidation.freeze(runtime, bridge.authority, catalogue, start=start)
        before = bridge.authority.verify()['reservations']
        result = runner.step(defer_fits=True)
        assert result['status'] == 'stepped'
        assert set(runner.state['algorithm_protocols']) == {protocol.digest}
        original = copy.deepcopy(runner.state['algorithm_protocols'])
        assert original[protocol.digest]['start'] == start.isoformat()
        assert any(a['kind'] == 'algorithm_controller' for a in result['actions'])
        steps = runner.state['steps']
        runner.close(close_runtime=False)
        runner = _service(runtime, bridge.authority)
        assert runner.state['steps'] == steps and runner.state['algorithm_protocols'] == original
        runner.step(defer_fits=True)
        assert runner.state['algorithm_protocols'] == original
        assert bridge.authority.verify()['reservations'] == before
        assert runtime.lifecycle.db.execute('SELECT COUNT(*) FROM batches').fetchone()[0] == 0
        assert runtime.lifecycle.exposure.db.execute('SELECT COUNT(*) FROM confirmations').fetchone()[0] == 0
    finally:
        runner.close(close_runtime=False)
        completion.close()


@pytest.mark.parametrize('damage', ['delete', 'replace'])
def test_seen_id_or_window_cannot_be_forgotten_on_restart(completed, damage):
    bridge, catalogue, _, runtime, _ = completed
    start = module._now().normalize().replace(day=1)+pd.DateOffset(months=1)
    protocol = module.AlgorithmValidation.freeze(runtime, bridge.authority, catalogue, start=start)
    runner = _service(runtime, bridge.authority)
    if damage == 'delete':
        del runner.state['algorithm_protocols'][protocol.digest]
    else:
        runner.state['algorithm_protocols'][protocol.digest]['qualifier_end'] = start.isoformat()
    # Even an erroneously re-journalled local projection cannot erase the
    # independent immutable seen anchor / original full contract.
    runner._save('private_projection_corruption')
    runner.close(close_runtime=False)
    with pytest.raises(ValueError, match='anchors/checkpoint|removed or replaced'):
        _service(runtime, bridge.authority)


def test_future_dated_contract_and_foreign_wallet_cannot_join(completed, monkeypatch):
    bridge, catalogue, _, runtime, _ = completed
    runner = _service(runtime, bridge.authority)
    _seed_current(runtime)
    now = module._now()
    start = now.normalize().replace(day=1)+pd.DateOffset(months=1)
    other = MonthlyWorkAuthority(runtime.directory.parent/'foreign-wallet',
        run_id='foreign-campaign', task='volatility', universe=['X'])
    with pytest.raises(ValueError, match='second wallet'):
        module.AlgorithmValidation.freeze(runtime, other, catalogue, start=start)
    # Private incompatible producer clock makes an otherwise genuine two-anchor
    # contract future-dated relative to the service's actual lifecycle clock.
    monkeypatch.setattr(module, '_now', lambda: now+pd.Timedelta(days=1))
    monkeypatch.setattr(runtime.lifecycle.exposure, 'clock', module._now)
    protocol = module.AlgorithmValidation.freeze(runtime, bridge.authority, catalogue, start=start)
    assert protocol.digest not in runner.state['algorithm_protocols']
    before = bridge.authority.verify()['reservations']
    try:
        with pytest.raises(ValueError, match='future-dated'):
            runner.step(defer_fits=True)
        assert runner.state['algorithm_protocols'] == {}
        assert bridge.authority.verify()['reservations'] == before
    finally:
        runner.close(close_runtime=False)
