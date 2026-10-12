"""Small actual service/wallet rejection checks; no provider or trained model.

Only the shared-schema dispatcher test replaces a verifier, explicitly testing
dispatch shape, not evidence authenticity or completed Linux resource work.
"""
import time
from pathlib import Path

import pandas as pd
import pytest

from factorlab.discovery.forward_gate_resolver import ForwardGatesPending, resolve_forward_gates
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
from factorlab.discovery.neural_forward import FrozenNeuralRecipe
from factorlab.discovery.neural_forward_ticket import _bind_authority, _verify_gates
from factorlab.discovery.v7_forward import ForwardRuntime
from factorlab.discovery.v7_service import ForwardService, ServiceConfig


def fit_pending_in_original_service_scope(service, now, actions):
    """Use the real service's original absolute budget, not a fresh child window.

    The dispatcher needs a complete unchanged 300s lease plus its fixed 2s
    publication margin. This fixture enters the actual owned service scope and
    derives its deadline from the durable first clock; repeated calls never
    issue a new service clock or widen the original service budget.
    """
    from factorlab.discovery.forward_service_scope import _service_step_scope
    with _service_step_scope(service):
        original = service.state['native_service_clock']
        deadline = (original['issuer']['monotonic_started'] + original['service_seconds']
                    - original['spent_before'])
        service._fit_pending(now, actions, deadline)


@pytest.fixture
def gate_case(tmp_path, monkeypatch):
    from factorlab import protocol
    monkeypatch.setattr(protocol, 'ROOT', tmp_path)
    (tmp_path/'factorlab').mkdir()
    (tmp_path/'factorlab'/'fixture.py').write_text('SYNTHETIC = True\n')
    (tmp_path/'pyproject.toml').write_text('[project]')
    (tmp_path/'uv.lock').write_text('version=1')
    clock = [pd.Timestamp('2026-10-31T00:00:00+00:00')]
    runtime = ForwardRuntime(tmp_path/'forward', tmp_path/'feed', clock=lambda: clock[0])
    wallet = MonthlyWorkAuthority(tmp_path/'wallet', run_id='fixture', task='volatility',
                                 universe=['BTCUSDT'], partition_policy='standard')
    _bind_authority(runtime, wallet)
    recipe = FrozenNeuralRecipe(rolling_spec=runtime.spec)
    lc = runtime.lifecycle
    lc.exposure.record(source=lc.plan.source_id, symbols=lc.plan.symbols,
        start='2026-07-01T00:00:00+00:00', end='2026-10-01T00:00:00+00:00', role='S', batch='B')
    lc.freeze_batch(batch='B', ordinal=1, start='2026-11-01T00:00:00+00:00',
        candidates=[{**runtime.recipe_for(recipe), 'complexity': 1}], reference=runtime.recipe_for(),
        evidence_sources=[{'start': '2026-07-01T00:00:00+00:00', 'end': '2026-10-01T00:00:00+00:00'}])
    clock[0] = pd.Timestamp('2026-11-02T00:00:00+00:00')
    job = lc.plan_fit('2026-11-01T00:00:00+00:00', 'B:'+recipe.pipeline_id, runtime.recipe_for(recipe))
    try:
        yield runtime, wallet, recipe, job
    finally:
        runtime.close()


def test_missing_original_dynamic_gates_wait_without_work_or_manual_map(gate_case):
    runtime, wallet, recipe, job = gate_case
    before = wallet.verify()
    with pytest.raises(ForwardGatesPending, match='completed_same_recipe_shared_S'):
        resolve_forward_gates(runtime, wallet, job, recipe)
    service = ForwardService(runtime, config=ServiceConfig(poll=False), work_authority=wallet)
    try:
        for _ in range(2):
            actions = []
            fit_pending_in_original_service_scope(service, runtime.lifecycle.now(), actions)
            assert actions[-1]['status'] == 'waiting_original_completed_same_recipe_shared_S_gates'
        assert service.state['fit_attempts'] == {}
        assert wallet.verify() == before
    finally:
        service.close(close_runtime=False)


def test_changed_static_map_is_still_rejected(gate_case):
    runtime, wallet, recipe, _ = gate_case
    service = ForwardService(runtime, config=ServiceConfig(poll=False), work_authority=wallet,
        neural_authority=wallet, neural_gates={recipe.pipeline_id: {'unverified_fixture': True}})
    try:
        service.neural_gates[recipe.pipeline_id]['changed'] = True
        with pytest.raises(ValueError, match='static authority/gate identity changed'):
            fit_pending_in_original_service_scope(service, runtime.lifecycle.now(), [])
        assert wallet.verify()['reservations'] == 0
    finally:
        service.close(close_runtime=False)


def test_resolver_rejects_foreign_recipe_and_caller_objects(gate_case):
    runtime, wallet, recipe, job = gate_case
    with pytest.raises(TypeError):
        resolve_forward_gates({'verified': True}, wallet, job, recipe)
    with pytest.raises(ValueError, match='frozen recipe job'):
        resolve_forward_gates(runtime, wallet, job, FrozenNeuralRecipe(output='encoder_ridge', rolling_spec=runtime.spec))


def test_shared_S_resource_cannot_be_passed_as_full_forward_gate(gate_case):
    runtime, wallet, recipe, _ = gate_case
    gates = {'shared_S': {'epoch': {}, 'probes': [], 'resource': 'S-only'}, 'resource_certificate': None}
    with pytest.raises(ValueError, match='shared producer tickets'):
        _verify_gates(gates, None, recipe, {'purpose': 'resource_trial'})
    with pytest.raises(ValueError, match='registered producer references'):
        _verify_gates({'epoch': {}, 'probes': [], 'resource': 'S-only'}, None, recipe, {})


def test_shared_schema_dispatch_only_calls_canonical_verifier(gate_case, monkeypatch):
    _, _, recipe, _ = gate_case
    calls = []
    def schema_fixture(gates, data, actual_recipe, context, **kwargs):
        calls.append((gates, actual_recipe.pipeline_id, kwargs))
        return {'test_only': False, 'permission': 'controlled_resource_trial_only'}
    monkeypatch.setattr('factorlab.discovery.s_nonlinear_gates.verify_shared_gates', schema_fixture)
    gates = {'shared_S': {'epoch': {}, 'probes': []}, 'resource_certificate': None}
    assert _verify_gates(gates, None, recipe, {'purpose': 'resource_trial'})['permission'] == 'controlled_resource_trial_only'
    assert calls[0][2]['require_resource'] is False
