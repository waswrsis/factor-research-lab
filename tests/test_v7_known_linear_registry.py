"""Real artificial ridge/registry checks; scheduling parsing is not forward fitting.

No market data, provider call, resource certificate, or formal claim is used.
"""
import copy
import json
from dataclasses import replace

import numpy as np
import pytest
from test_v7_nonlinear_channel import artificial_data

from factorlab.discovery.claim_protocol import FrozenClaimEndpoint
from factorlab.discovery.convex_eval import fit_ridge
from factorlab.discovery.joint_pool import FrozenPool
from factorlab.discovery.known_linear_reference import FrozenLinearRecipe, fit_known_linear
from factorlab.discovery.model_registry import ModelRegistry
from factorlab.discovery.nonlinear_channel import ChannelBudget
from factorlab.discovery.v7_forward import ForwardRuntime
from factorlab.discovery.v7_service import ForwardService
from factorlab.protocol import fingerprint


@pytest.fixture
def control(tmp_path):
    data = artificial_data()
    runtime = ForwardRuntime.__new__(ForwardRuntime)
    runtime.spec = data.spec
    service = ForwardService.__new__(ForwardService)
    service.runtime = runtime
    return data, runtime, service, ModelRegistry(tmp_path / 'registry')


@pytest.mark.parametrize('name', ['HAR_cal', 'K0'])
def test_real_fixed_model_registry_roundtrip_and_recipe_parse(control, name):
    data, runtime, service, registry = control
    recipe = FrozenLinearRecipe(name, data.spec)
    budget = ChannelBudget()
    model = fit_known_linear(data, '2025-04-01T00:00Z', recipe, budget=budget)
    pipeline = runtime.recipe_for(recipe)
    identifier = registry.save(model, pipeline)
    restored, identity = ModelRegistry(registry.directory).load(identifier, expected_pipeline=pipeline)
    assert budget.counts['solve'] == 1
    assert restored.digest == model.digest and identity['pipeline'] == pipeline
    np.testing.assert_array_equal(restored.predict(data.frame), model.predict(data.frame))
    assert registry.save(restored, pipeline) == identifier
    assert service._pool(pipeline) == recipe
    assert service._pool({**pipeline, 'complexity': 7}) == recipe
    assert identity['task_support'] == ['volatility']
    assert 'admission' not in identity  # Convex identity is not an NN admission certificate.


@pytest.mark.parametrize('mutation', ['id', 'rolling', 'extra', 'columns', 'decode'])
def test_scheduler_rejects_changed_full_control_identity(control, mutation):
    data, runtime, service, registry = control
    recipe = FrozenLinearRecipe('HAR_cal', data.spec)
    model = fit_known_linear(data, '2025-04-01T00:00Z', recipe, budget=ChannelBudget())
    pipeline = copy.deepcopy(runtime.recipe_for(recipe))
    if mutation == 'id':
        pipeline['pipeline_id'] = '0' * 64
    elif mutation == 'rolling':
        pipeline['rolling_hash'] = '0' * 64
    elif mutation == 'extra':
        pipeline['unknown_policy'] = 'unfrozen'
    elif mutation == 'columns':
        pipeline['recipe']['columns'].reverse()
    else:
        pipeline['recipe']['decode'] = 'exp(q)'
    with pytest.raises(ValueError):
        service._pool(pipeline)
    with pytest.raises(ValueError):
        registry.save(model, pipeline)


def test_runtime_rejects_control_with_another_rolling_protocol(control):
    data, runtime, _, _ = control
    recipe = FrozenLinearRecipe('HAR_cal', replace(data.spec, ridge=data.spec.ridge * 2))
    with pytest.raises(ValueError, match='rolling rule'):
        runtime.recipe_for(recipe)


def test_bare_ridge_cannot_borrow_known_control_pipeline(control):
    data, runtime, _, registry = control
    bare = fit_ridge(np.arange(20).reshape(-1, 1), np.linspace(.01, .1, 20))
    with pytest.raises(ValueError, match='complete matching'):
        registry.save(bare, runtime.recipe_for(FrozenLinearRecipe('HAR_cal', data.spec)))


@pytest.mark.parametrize('mutation', ['pipeline', 'schema', 'task_support'])
def test_recomputed_registry_hash_cannot_hide_wrong_model_identity(control, mutation):
    data, runtime, _, registry = control
    model = fit_known_linear(data, '2025-04-01T00:00Z',
        FrozenLinearRecipe('HAR_cal', data.spec), budget=ChannelBudget())
    identifier = registry.save(model, runtime.recipe_for(model.recipe))
    old = registry.directory / identifier
    record = json.loads((old / 'manifest.json').read_text())
    if mutation == 'pipeline':
        record['identity']['pipeline'] = runtime.recipe_for(FrozenLinearRecipe('K0', data.spec))
    elif mutation == 'schema':
        record['identity']['schema']['recipe']['name'] = 'K0'
    else:
        record['identity']['task_support'] = ['first_event']
    record['id'] = fingerprint(record['identity'])
    new = registry.directory / record['id']
    new.mkdir()
    (new / 'model.json').write_bytes((old / 'model.json').read_bytes())
    (new / 'manifest.json').write_text(json.dumps(record))
    with pytest.raises(ValueError):
        registry.load(record['id'])
    assert registry.load(identifier)[0].digest == model.digest


def test_claim_can_freeze_actual_har_reference_with_original_pool_k0(control):
    data, runtime, service, _ = control
    k0 = runtime.recipe_for(FrozenPool(role='reference', version='K0_v1'))
    har = runtime.recipe_for(FrozenLinearRecipe('HAR_cal', data.spec))
    endpoint = FrozenClaimEndpoint.library('2026-01-01T00:00Z', k0, har)
    assert service._pool(endpoint.reference).name == 'HAR_cal'
    assert service._pool(endpoint.candidate).members == ()


def test_typed_linear_k0_not_silently_equated_to_original_pool_pipeline(control):
    data, runtime, _, _ = control
    typed = runtime.recipe_for(FrozenLinearRecipe('K0', data.spec))
    pool = runtime.recipe_for(FrozenPool(role='reference', version='K0_v1'))
    assert typed['pipeline_id'] != pool['pipeline_id']
    # Current claim policy explicitly retains the old empty-pool K0 identity.
    # A future typed-K0 migration needs an explicit policy, not a renamed hash.
    with pytest.raises(ValueError, match='empty-pool'):
        FrozenClaimEndpoint.library('2026-01-01T00:00Z', typed, pool)
