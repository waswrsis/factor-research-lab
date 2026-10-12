"""Synthetic missing-data paid route; never claims real gate/resource success.

Only discovery is stubbed with empty shared references to reach the actual paid
allocator. Preparation must fail on the empty feed; no model or provider runs.
"""
import json

import pytest

from factorlab.discovery import forward_resource_work as resource
from factorlab.discovery.neural_forward_contract import assert_frozen_forward_fit
from factorlab.discovery.neural_forward_ticket import _verify_historical_context
import pandas as pd
from factorlab.discovery.v7_forward import ForwardRuntime
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
from factorlab.discovery.neural_forward import FrozenNeuralRecipe
from factorlab.discovery.neural_forward_ticket import _bind_authority

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



def synthetic_resolution():
    return {'purpose': 'V', 'research_ticket_digest': 'synthetic-no-proof',
            'gates': {'shared_S': {'epoch': {}, 'probes': []}, 'resource_certificate': None}}


def test_dedicated_original_resource_plan_cannot_serve_or_replace_a0(gate_case):
    runtime, wallet, recipe, target = gate_case
    a0 = runtime.lifecycle.plan_fit('2026-11-01T00:00:00+00:00', 'A0', runtime.recipe_for())
    plan = resource._plan(runtime, wallet, target, recipe, synthetic_resolution())
    assert plan['job'] not in {a0, target}
    context = assert_frozen_forward_fit(runtime.lifecycle, plan['job'], recipe, purpose='resource_trial')['binding']
    assert context['serving_eligible'] is False
    assert context['stream'] == resource.PREFIX+recipe.pipeline_id
    _verify_historical_context(runtime.lifecycle, context, recipe)
    for role in ('A0', 'V', 'C'):
        with pytest.raises(ValueError, match='Non-serving'):
            assert_frozen_forward_fit(runtime.lifecycle, plan['job'], recipe, purpose=role)
    assert runtime.lifecycle.db.execute('SELECT pipeline FROM jobs WHERE id=?', (a0,)).fetchone()[0] == json.dumps(runtime.recipe_for(), sort_keys=True, separators=(',', ':'))


def test_resource_producer_real_paid_missing_data_work_is_not_repeated(gate_case, monkeypatch):
    runtime, wallet, recipe, target = gate_case
    from factorlab.discovery import forward_gate_resolver
    monkeypatch.setattr(forward_gate_resolver, 'resolve_forward_gates', lambda *a, **kw: synthetic_resolution())
    first = resource.produce_resource_work(runtime, wallet, target, recipe)
    assert first['status'] == 'failed_consumed' and first['executed'] is True
    paid = wallet.verify()
    assert paid['reservations'] == 1
    second = resource.produce_resource_work(runtime, wallet, target, recipe)
    assert second['status'] == 'failed_consumed' and second['executed'] is False
    assert wallet.verify() == paid
    assert runtime.lifecycle.db.execute('SELECT COUNT(*) FROM active').fetchone()[0] == 0


def test_resource_context_rejects_changed_original_registration(gate_case):
    runtime, wallet, recipe, target = gate_case
    plan = resource._plan(runtime, wallet, target, recipe, synthetic_resolution())
    path = runtime.directory/'resource_work'/plan['job']/'plan.json'
    value = json.loads(path.read_text())
    value['gates']['resource_certificate'] = {'caller_success': True}
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='original registration'):
        resource.verify_resource_plan(runtime.lifecycle, plan['job'], recipe)
