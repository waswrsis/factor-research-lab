"""Independent controller-boundary audit; artificial data, no external calls."""
import copy
import json
import sqlite3

import pandas as pd
import pytest
from test_v7_nonlinear_orchestration import owner as owner_fixture
from test_v7_rolling_registry import candles
from test_v7_run_factor_pool import source_root

from factorlab.discovery import nonlinear_orchestration as orchestration
from factorlab.discovery import nonlinear_runtime as runtime
from factorlab.discovery import v7_run


@pytest.fixture
def owner(tmp_path, monkeypatch):
    yield from owner_fixture.__wrapped__(tmp_path, monkeypatch)


def hook(owner, *, seconds=3600, mode='probes'):
    bridge, data, cutoff, *_ = owner
    events = []

    def checkpoint(event):
        bridge.store.save(bridge.month, bridge.state, event)
        events.append(event)

    result = v7_run._nonlinear_month(bridge.store, bridge.month, bridge.state,
        bridge.budget, data, pd.Timestamp(cutoff), mode, seconds, checkpoint)
    return result, events


def test_short_remaining_wall_does_not_reserve_or_prepare_channel(owner, monkeypatch):
    bridge, *_ = owner
    before = copy.deepcopy(bridge.state)
    monkeypatch.setattr(orchestration, 'freeze_epoch', lambda *a: pytest.fail('Started short-wall work'))
    result, events = hook(owner, seconds=3599.999)
    assert result['status'] == 'deferred_monthly_wall_budget'
    assert not events and bridge.state == before


def test_committed_channel_report_is_reused_without_new_work(owner, monkeypatch):
    bridge, *_ = owner
    bridge.state['nonlinear_report'] = {'status': 'previous_result', 'automatic_adoption': False}
    bridge.commit()
    monkeypatch.setattr(orchestration, 'freeze_epoch', lambda *a: pytest.fail('Repeated completed work'))
    result, events = hook(owner)
    assert result == bridge.state['nonlinear_report'] and not events


@pytest.mark.parametrize('role', ['V', 'C'])
def test_hook_checks_actual_shared_exposure_before_producer(owner, monkeypatch, role):
    bridge, *_ = owner
    exposed = bridge.state['search_exposure']
    bridge.store.exposure.record(**{k: v for k, v in exposed.items() if k != 'data_id'}, role=role)
    monkeypatch.setattr(orchestration, 'produce_quarterly_artifacts',
        lambda *a: pytest.fail('Protected S input reached producer'))
    with pytest.raises(ValueError):
        hook(owner)
    assert bridge.budget.counts['solve'] == 0


def test_insufficient_capacity_cannot_hide_forged_kernel_proof(owner, monkeypatch):
    bridge, data, cutoff, *_ = owner
    folder = bridge.store.directory / 'nonlinear_epoch'
    context = runtime._context(data, cutoff, orchestration.epoch_pool(), bridge.source_id)
    # A structurally registered historic receipt still needs its independent
    # launcher/kernel evidence. Small size must not turn corruption into a skip.
    monkeypatch.setattr(runtime, '_now', lambda: pd.Timestamp(cutoff))
    store = runtime._Store(folder)
    try:
        runtime._register(store, 'resource', context,
            {**{key: 0 for key in runtime.workload_capacity(data, cutoff)},
             'verified': True, 'scope': 'whole_cgroup'})
    finally:
        store.close()
    monkeypatch.setattr(orchestration, 'produce_quarterly_artifacts', lambda *a: {'status': 'cached'})
    monkeypatch.setattr(orchestration, 'prior_gate_artifacts',
        lambda *a: {'status': 'research_evidence_verified', 'artifacts': {}})
    launches = []
    monkeypatch.setattr(orchestration, 'run_isolated_research',
        lambda *a, **kw: launches.append(kw) or {'status': 'unexpected_resource_replacement'})
    with pytest.raises(ValueError):
        hook(owner, mode='isolated')
    assert not launches


@pytest.mark.parametrize('sufficient', [False, True])
def test_resource_proof_is_verified_before_capacity_skip_or_reuse(owner, monkeypatch, sufficient):
    bridge, data, cutoff, *_ = owner
    folder = bridge.store.directory / 'nonlinear_epoch'
    work_data = orchestration.causal_worker_data(data, cutoff)
    context = runtime._context(work_data, cutoff, orchestration.epoch_pool(), bridge.source_id)
    capacity = runtime.workload_capacity(work_data, cutoff)
    runtime._capacity(capacity, work_data, cutoff)
    artifact = {**capacity, 'synthetic_proof_id': 'verified-by-test-only-kernel-verifier'}
    if not sufficient:
        artifact['frame_byte_capacity'] -= 1
        with pytest.raises(ValueError, match='capacity'):
            runtime._capacity(artifact, work_data, cutoff)
    monkeypatch.setattr(runtime, '_now', lambda: pd.Timestamp(cutoff))
    store = runtime._Store(folder)
    try:
        identifier = runtime._register(store, 'resource', context, artifact)
    finally:
        store.close()
    verified = []

    def verify_resource(value, expected):
        assert value == artifact
        assert expected['context'] == runtime._common(context)
        verified.append(value)
        return {'verified': True}

    monkeypatch.setattr(runtime, '_resource', verify_resource)
    monkeypatch.setattr(runtime, 'verify_gate_artifacts', lambda *a, **kw: {'test_only': False})
    monkeypatch.setattr(orchestration, 'produce_quarterly_artifacts', lambda *a: {'status': 'cached'})
    monkeypatch.setattr(orchestration, 'prior_gate_artifacts',
        lambda *a: {'status': 'research_evidence_verified', 'artifacts': {}})
    launches = []
    monkeypatch.setattr(orchestration, 'run_isolated_research',
        lambda *a, **kw: launches.append(kw) or {'status': 'synthetic_launch'})
    result, _ = hook(owner, mode='isolated')
    assert len(verified) == 1
    assert launches == [{'resource_receipt_id': identifier if sufficient else None,
                         'resource_trial': not sufficient}]
    assert bool(result['resource_capacity_rejections']) is not sufficient


def test_disabled_month_does_not_touch_channel(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    monkeypatch.setattr(v7_run, '_nonlinear_month', lambda *a: pytest.fail('Disabled hook called'))
    result = v7_run.run_month('research', candles('2024-08-01', '2025-04-02'),
        '2025-04-01T00:00:00Z',
        config=v7_run.V7Config(evaluation_mode='joint', formulas_per_family=1))
    assert result['nonlinear'] is None
    assert not (tmp_path / 'research/nonlinear_epoch').exists()


def test_finalizing_resume_republishes_without_hook_and_keeps_post_channel_budget(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    calls = []

    def measured_hook(store, month, state, budget, data, cutoff, mode, remaining, checkpoint):
        assert store.get(month) == state
        assert cutoff == pd.Timestamp(state['cutoff'])
        budget.consume('solve', 7)
        checkpoint('independent_synthetic_channel_charge')
        calls.append(budget.counts['solve'])
        result = {'status': 'synthetic_channel_complete', 'automatic_adoption': False,
                  'structure_proposals': [{'structure_id': 'unregistered-neural-sentinel'}],
                  'recipe_status': 'neural_forward_registry_not_registered'}
        state['nonlinear_report'] = result
        checkpoint('independent_synthetic_channel_result')
        return result

    monkeypatch.setattr(v7_run, '_nonlinear_month', measured_hook)
    original_publish = v7_run.immutable_json

    def fail_publication(path, value):
        if path.parent.name == 'months':
            raise OSError('synthetic publication interruption')
        return original_publish(path, value)

    monkeypatch.setattr(v7_run, 'immutable_json', fail_publication)
    config = v7_run.V7Config(evaluation_mode='joint', formulas_per_family=1, nonlinear_mode='probes')
    bars = candles('2024-08-01', '2025-04-02')
    with pytest.raises(OSError, match='publication interruption'):
        v7_run.run_month('research', bars, '2025-04-01T00:00:00Z', config=config)
    with sqlite3.connect(tmp_path / 'research/state.sqlite') as db:
        state = json.loads(db.execute('SELECT value FROM months').fetchone()[0])
    assert state['status'] == 'finalizing'
    assert state['summary']['budget'] == state['budget']
    assert state['budget']['counts']['solve'] == calls[0]
    monkeypatch.setattr(v7_run, 'immutable_json', original_publish)
    monkeypatch.setattr(v7_run, '_nonlinear_month', lambda *a: pytest.fail('Finalizing resumed training'))
    restored = v7_run.run_month('research', bars, '2025-04-01T00:00:00Z', config=config)
    assert restored == state['summary'] and len(calls) == 1
    assert restored['A0'] == 'K0' and not restored['formal_confirmation_enabled']
    assert 'unregistered-neural-sentinel' not in json.dumps(restored['shortlist'])
