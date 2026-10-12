"""Protected fit permissions; synthetic lifecycle data are plumbing only."""
import json
from dataclasses import replace

import pandas as pd
import pytest
from test_v7_lifecycle_audit import _loss_rows
from test_v7_lifecycle_audit import audit_store as audit_store  # noqa: PLC0414

from factorlab.discovery.convex_eval import fit_ridge
from factorlab.discovery.neural_forward import FrozenNeuralRecipe
from factorlab.discovery.neural_forward_contract import assert_frozen_forward_fit, neural_pipeline
from factorlab.discovery.v7_forward import ForwardRuntime


def freeze(store, recipe):
    store.exposure.record(source='audit-feed', symbols=['BTCUSDT'],
        start='2026-07-01T00:00:00Z', end='2026-10-01T00:00:00Z', role='S', batch='A')
    pipeline = neural_pipeline(recipe, recipe.rolling_spec)
    store.freeze_batch(batch='A', ordinal=1, start='2026-11-01T00:00:00Z',
        candidates=[{**pipeline, 'complexity': 2}], reference=pipeline,
        evidence_sources=[{'start': '2026-07-01T00:00:00Z', 'end': '2026-10-01T00:00:00Z'}])
    return pipeline


def plan(store, recipe, cutoff, stream):
    return store.plan_fit(cutoff, stream, neural_pipeline(recipe, recipe.rolling_spec))


def test_full_recipe_roundtrips_without_silently_using_its_pool():
    recipe = FrozenNeuralRecipe()
    runtime = ForwardRuntime.__new__(ForwardRuntime)
    runtime.spec = recipe.rolling_spec
    pipeline = runtime.recipe_for(recipe)
    assert pipeline['kind'] == 'v7_neural_forward'
    assert pipeline['pipeline_id'] == recipe.pipeline_id
    assert FrozenNeuralRecipe.from_dict(pipeline['recipe']) == recipe
    assert runtime.recipe_for(recipe.pool)['pipeline_id'] != recipe.pipeline_id
    with pytest.raises(ValueError, match='rolling'):
        runtime.recipe_for(replace(recipe, rolling_spec=replace(recipe.rolling_spec, ridge=.5)))


def test_real_frozen_v_fit_permission_and_reference(audit_store):
    store, clock = audit_store
    recipe = FrozenNeuralRecipe()
    pipeline = freeze(store, recipe)
    clock[0] = pd.Timestamp('2026-11-01T00:00:05Z')
    for stream in ['A:' + recipe.pipeline_id, 'A:reference']:
        job = plan(store, recipe, '2026-11-01T00:00:00Z', stream)
        context = assert_frozen_forward_fit(store, job, recipe, purpose='V')
        assert context['binding']['pipeline'] == pipeline
        assert context['binding']['batch'] == 'A'
        assert context['binding']['serving_eligible']
        assert context['observed_event_head'] == store.verify()['event_head']


@pytest.mark.parametrize('purpose', ['A0', 'C', 'resource_trial', 'S', 'anything'])
def test_v_job_cannot_borrow_another_role(audit_store, purpose):
    store, clock = audit_store
    recipe = FrozenNeuralRecipe()
    freeze(store, recipe)
    clock[0] = pd.Timestamp('2026-11-01T00:00:05Z')
    job = plan(store, recipe, '2026-11-01T00:00:00Z', 'A:' + recipe.pipeline_id)
    with pytest.raises(ValueError):
        assert_frozen_forward_fit(store, job, recipe, purpose=purpose)


def test_changed_output_cannot_fit_the_registered_full_recipe(audit_store):
    store, clock = audit_store
    recipe = FrozenNeuralRecipe()
    freeze(store, recipe)
    clock[0] = pd.Timestamp('2026-11-01T00:00:05Z')
    job = plan(store, recipe, '2026-11-01T00:00:00Z', 'A:' + recipe.pipeline_id)
    with pytest.raises(ValueError, match='authoritative'):
        assert_frozen_forward_fit(store, job, replace(recipe, output='encoder_ridge'), purpose='V')


@pytest.mark.parametrize('moment', ['2026-12-01T00:00:00Z', '2027-01-01T00:00:00Z'])
def test_late_v_training_does_not_extend_the_frozen_period(audit_store, moment):
    store, clock = audit_store
    recipe = FrozenNeuralRecipe()
    freeze(store, recipe)
    clock[0] = pd.Timestamp('2026-11-01T00:00:05Z')
    job = plan(store, recipe, '2026-11-01T00:00:00Z', 'A:' + recipe.pipeline_id)
    clock[0] = pd.Timestamp(moment)
    with pytest.raises(ValueError, match='window'):
        assert_frozen_forward_fit(store, job, recipe, purpose='V')


def test_trial_is_distinct_from_new_a0_adoption(audit_store):
    store, _ = audit_store
    recipe = FrozenNeuralRecipe()
    job = plan(store, recipe, '2026-10-01T00:00:00Z', 'A0')
    context = assert_frozen_forward_fit(store, job, recipe, purpose='resource_trial')
    assert not context['binding']['serving_eligible']
    with pytest.raises(ValueError, match='adoption authority'):
        assert_frozen_forward_fit(store, job, recipe, purpose='A0')


def test_old_convex_model_with_neural_label_cannot_authorize_a0_refit(audit_store):
    import numpy as np
    store, clock = audit_store
    recipe = FrozenNeuralRecipe()
    previous = plan(store, recipe, '2026-10-01T00:00:00Z', 'A0')
    store.advance_fit(previous, 'snapshot_sealed')
    store.advance_fit(previous, 'features_ready')
    # A legacy learner cannot enter the neural namespace through metadata.
    ridge = fit_ridge(np.arange(40).reshape(-1, 1), np.linspace(.01, .03, 40))
    with pytest.raises(ValueError, match='complete neural model'):
        store.seal_model(previous, ridge)
    clock[0] = pd.Timestamp('2026-11-01T00:00:05Z')
    job = plan(store, recipe, '2026-11-01T00:00:00Z', 'A0')
    with pytest.raises(ValueError, match='adoption authority'):
        assert_frozen_forward_fit(store, job, recipe, purpose='A0')


def test_c_requires_real_selected_reservation_and_exact_actual_window(audit_store, monkeypatch):
    store, clock = audit_store
    recipe = FrozenNeuralRecipe()
    freeze(store, recipe)
    clock[0] = pd.Timestamp('2026-12-01T00:00:05Z')
    monkeypatch.setattr(store, '_losses', lambda stream, start, end, as_of, **kwargs:
        _loss_rows(start, end, 1. if stream.endswith(':reference') else .9))
    store.select_validation('A')
    job = plan(store, recipe, '2026-12-01T00:00:00Z', 'A:' + recipe.pipeline_id)
    with pytest.raises(ValueError, match='confirmation'):
        assert_frozen_forward_fit(store, job, recipe, purpose='C')
    monkeypatch.setattr(store, 'calibration_ready', lambda: True)  # Plumbing, not a certificate.
    store.reserve_selected_confirmation('A')
    reservation = json.loads(store.exposure.db.execute('SELECT identity FROM confirmations').fetchone()[0])
    with pytest.raises(ValueError, match='window'):
        assert_frozen_forward_fit(store, job, recipe, purpose='C')
    clock[0] = pd.Timestamp(reservation['actual_start'])
    context = assert_frozen_forward_fit(store, job, recipe, purpose='C')
    assert context['binding']['confirmation']['candidate_id'] == recipe.pipeline_id
    clock[0] = pd.Timestamp(context['binding']['confirmation']['end'])
    with pytest.raises(ValueError, match='window'):
        assert_frozen_forward_fit(store, job, recipe, purpose='C')


def test_forged_mutable_job_index_cannot_grant_permission(audit_store):
    store, clock = audit_store
    recipe = FrozenNeuralRecipe()
    freeze(store, recipe)
    clock[0] = pd.Timestamp('2026-11-01T00:00:05Z')
    job = plan(store, recipe, '2026-11-01T00:00:00Z', 'A:' + recipe.pipeline_id)
    store.db.execute("UPDATE jobs SET stream='A0' WHERE id=?", (job,))
    store.db.commit()
    with pytest.raises(ValueError, match='operations chain'):
        assert_frozen_forward_fit(store, job, recipe, purpose='resource_trial')
