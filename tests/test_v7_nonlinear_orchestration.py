"""Real local ledger + artificial data. No market, GPT, Linux or adoption claim."""
import copy
import json
from dataclasses import replace

import pandas as pd
import pytest
from test_v7_nonlinear_channel import artificial_data
from test_v7_run_factor_pool import source_root

from factorlab.discovery import nonlinear_launcher as launcher
from factorlab.discovery import nonlinear_orchestration as orchestration
from factorlab.discovery import nonlinear_runtime
from factorlab.discovery.local_mutation import MonthlyBudget
from factorlab.discovery.nonlinear_channel import ResidualConfig, StructureSpec
from factorlab.discovery.v7_memory import search_exposure_context
from factorlab.discovery.v7_run import DurableBudget, RunStore, V7Config


@pytest.fixture
def owner(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    clock = pd.Timestamp('2025-04-02T00:00:00Z')
    monkeypatch.setattr(orchestration, '_now', lambda: clock)
    monkeypatch.setattr(nonlinear_runtime, '_now', lambda: clock)
    data = artificial_data()
    store = RunStore(tmp_path / 'run', V7Config(), data.spec)
    month, cutoff = '2025-04', '2025-04-01T00:00:00+00:00'
    state = {'status': 'running', 'cutoff': cutoff, 'data_id': 'artificial-controller-snapshot',
             'budget': MonthlyBudget(month).to_dict(),
             'nonlinear_input_data_id': orchestration.input_data_identity(data, cutoff)}
    state['search_exposure'] = search_exposure_context(state['data_id'], cutoff, ['X'], 'fixture-feed')
    store.exposure.record(**{k: v for k, v in state['search_exposure'].items() if k != 'data_id'}, role='S')
    store.save(month, state, 'synthetic_owner')
    committed = []

    def persist(value):
        state['budget'] = value
        store.save(month, state, 'atomic_global_and_channel_reservation')
        committed.append(copy.deepcopy(state))

    budget = DurableBudget(MonthlyBudget.from_dict(state['budget']), persist)
    bridge = orchestration.RunStoreBudgetBridge(store, month, state, budget)
    folder = tmp_path / 'epoch'
    epoch = orchestration.freeze_epoch(folder, bridge, data, cutoff)
    yield bridge, data, cutoff, folder, epoch, committed
    store.close()


def test_real_quarterly_tree_and_control_have_one_atomic_global_ledger(owner):
    bridge, data, cutoff, folder, epoch, committed = owner
    result = orchestration.produce_quarterly_artifacts(folder, bridge, data, cutoff, epoch)
    assert result['status'] == 'completed' and not result['automatic_adoption']
    assert result['usable_from'] == '2025-05-01T00:00:00+00:00'
    assert bridge.budget.counts['solve'] == 9
    assert bridge.state['nonlinear']['channel']['counts']['solve'] == 9
    assert bridge.state['nonlinear']['channel']['counts']['tree_fit'] == 3
    for state in committed:
        assert state['budget']['counts']['solve'] == state['nonlinear']['channel']['counts']['solve']
    for kind in ('probe', 'positive_control'):
        value = json.loads((folder / 'receipts' / f'{result[kind]["receipt_id"]}.json').read_text())
        assert value['role'] == 'S'
        assert value['result']['producer_binding']['controller_data_id'] == bridge.state['data_id']
    again = orchestration.produce_quarterly_artifacts(folder, bridge, data, cutoff, epoch)
    assert again['probe']['cached'] and again['positive_control']['cached']
    assert bridge.budget.counts['solve'] == 9


def test_raw_monthly_wallet_or_detached_callback_is_not_a_bridge(owner):
    bridge, _data, _cutoff, _folder, epoch, _ = owner
    with pytest.raises(TypeError, match='DurableBudget'):
        orchestration.RunStoreBudgetBridge(bridge.store, bridge.month, bridge.state, MonthlyBudget(bridge.month))
    detached = DurableBudget(MonthlyBudget.from_dict(bridge.state['budget']), lambda _: None)
    wrong = orchestration.RunStoreBudgetBridge(bridge.store, bridge.month, bridge.state, detached)
    with pytest.raises(ValueError, match='authoritative'):
        wrong.section(epoch)


@pytest.mark.parametrize('damage', ['missing', 'V', 'C', 'complete'])
def test_global_exposure_and_active_month_are_checked_before_fit(owner, monkeypatch, damage):
    bridge, data, cutoff, folder, epoch, _ = owner
    if damage == 'complete':
        bridge.state['status'] = 'complete'
        bridge.store.save(bridge.month, bridge.state, 'closed')
    elif damage == 'missing':
        bridge.store.exposure.db.execute('DELETE FROM evidence')
        bridge.store.exposure.db.commit()
    else:
        bridge.store.exposure.record(**{k: v for k, v in bridge.state['search_exposure'].items()
                                        if k != 'data_id'}, role=damage)
    monkeypatch.setattr(orchestration, 'run_probe', lambda *args: pytest.fail('Protected data reached fit'))
    with pytest.raises(ValueError):
        orchestration.produce_quarterly_artifacts(folder, bridge, data, cutoff, epoch)
    assert bridge.budget.counts['solve'] == 0


def test_failed_probe_is_consumed_and_never_rerun(owner, monkeypatch):
    bridge, data, cutoff, folder, epoch, _ = owner
    calls = []

    def fail(ledger):
        ledger.consume('tree_fit')
        ledger.consume('solve')
        calls.append(1)
        raise RuntimeError('synthetic fit failure')

    with pytest.raises(RuntimeError):
        orchestration._produce(bridge, folder, data, cutoff, epoch, 'probe', fail)
    result = orchestration._produce(bridge, folder, data, cutoff, epoch, 'probe', fail)
    assert result['status'] == 'failed_consumed' and len(calls) == 1
    assert bridge.budget.counts['solve'] == 1
    assert bridge.store.get(bridge.month)['nonlinear']['channel']['counts']['solve'] == 1


def test_neural_ticket_reserves_complete_work_once_atomically(owner):
    bridge, _, _, _, epoch, committed = owner
    ticket = orchestration.reserve_neural_ticket(bridge, epoch, 'frozen-gate')
    assert ticket['status'] == 'reserved'
    assert ticket['ticket']['limits'] == {'solve': 21, 'neural_fit': 9, 'epoch': 36}
    assert bridge.budget.counts['solve'] == 21
    last = committed[-1]
    assert last['budget']['counts']['solve'] == last['nonlinear']['channel']['counts']['solve'] == 21
    assert last['nonlinear']['attempts'][ticket['ticket_id']]['status'] == 'reserved'
    assert orchestration.reserve_neural_ticket(bridge, epoch, 'frozen-gate')['status'] == 'already_consumed'
    assert bridge.budget.counts['solve'] == 21


def test_new_epoch_cannot_mix_a_different_structure_protocol(owner):
    bridge, data, cutoff, folder, _, _ = owner
    with pytest.raises(ValueError, match='changed'):
        orchestration.freeze_epoch(folder, bridge, data, cutoff,
            config=ResidualConfig(enabled=True, structures=(StructureSpec(16, 4, 1),)))


def test_current_quarter_result_does_not_open_current_cutoff_gate(owner):
    bridge, data, cutoff, folder, epoch, _ = owner
    result = orchestration.prior_gate_artifacts(folder, bridge, data, cutoff, epoch)
    assert result['status'] == 'pending_quarterly_evidence'
    assert pd.Timestamp(cutoff).isoformat() not in result['required_quarters']
    assert orchestration.run_isolated_research(folder, bridge, data, cutoff, epoch)['status'] == result['status']
    assert bridge.budget.counts['solve'] == 0


def test_structure_proposals_are_actual_epoch_candidates_without_pool_adoption(owner):
    *_, epoch, _ = owner
    records = orchestration.structure_proposals(epoch)
    assert len(records) == len(epoch.config.structures)
    assert all(r['structure']['experts'] == 1 and not r['automatic_adoption'] for r in records)
    assert all(r['recipe_status'] == 'neural_forward_registry_not_registered' for r in records)


def test_windows_and_missing_torch_have_typed_closed_status(monkeypatch):
    monkeypatch.setattr(launcher.sys, 'platform', 'win32')
    assert launcher.availability()['status'] == 'linux_resource_proof_unavailable'
    monkeypatch.setattr(launcher.sys, 'platform', 'linux')
    monkeypatch.setattr(launcher.importlib.util, 'find_spec', lambda _: None)
    assert launcher.availability()['status'] == 'dependency_unavailable'


def test_worker_data_roundtrip_has_no_pickle_and_preserves_causal_masks(owner):
    _bridge, data, cutoff, folder, _, _ = owner
    record = launcher.save_worker_data(folder / 'test-data', data, cutoff=cutoff)
    from dataclasses import asdict
    restored = launcher._load_data({'data': record, 'rolling': asdict(data.spec)})
    assert restored.training_identity(cutoff) == data.training_identity(cutoff)
    assert restored.frame.metadata.equals(orchestration.causal_worker_data(data, cutoff).frame.metadata)
    changed = replace(data, target=data.target.copy())
    assert restored.spec.digest == changed.spec.digest


def test_controller_must_explicitly_freeze_actual_input_component(owner):
    bridge, data, cutoff, folder, _epoch, _ = owner
    bridge.state.pop('nonlinear_input_data_id')
    bridge.store.save(bridge.month, bridge.state, 'fixture_missing_input_anchor')
    with pytest.raises(ValueError, match='controller input identity'):
        orchestration.freeze_epoch(folder, bridge, data, cutoff)
    assert bridge.budget.counts['solve'] == 0


def test_transfer_clears_unmatured_targets_even_for_past_origins(owner):
    _bridge, data, cutoff, folder, _, _ = owner
    maturity = data.maturity.to_series(index=range(len(data.maturity)))
    past = data.frame.metadata.available_at < pd.Timestamp(cutoff)
    maturity.loc[past[past].index[-1]] = pd.Timestamp(cutoff) + pd.Timedelta(hours=1)
    data = replace(data, maturity=pd.DatetimeIndex(maturity))
    record = launcher.save_worker_data(folder / 'causal-only', data, cutoff=cutoff)
    from dataclasses import asdict
    restored = launcher._load_data({'data': record, 'rolling': asdict(data.spec)})
    assert (restored.frame.metadata.available_at < pd.Timestamp(cutoff)).all()
    mask = ~restored.observed
    assert mask.any() and (restored.target[mask] == 0).all()
    assert restored.maturity[mask].isna().all()
    assert orchestration.input_data_identity(restored, cutoff) == orchestration.input_data_identity(data, cutoff)


def test_trial_retry_cannot_obtain_new_budget_by_changing_gate_receipts(owner):
    bridge, _, _, _, epoch, _ = owner
    first = orchestration.reserve_neural_ticket(bridge, epoch, 'first-evidence', mode='resource_trial')
    repeated = orchestration.reserve_neural_ticket(bridge, epoch, 'changed-evidence', mode='resource_trial')
    assert repeated == {'status': 'already_consumed', 'ticket_id': first['ticket_id']}
    assert bridge.budget.counts['solve'] == 21
