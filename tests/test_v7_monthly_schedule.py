"""No fitting: explicit fixed ceilings and pool-dependent full-work bounds."""
from dataclasses import replace

import pytest
from factorlab.discovery.monthly_schedule import (
    default_control_schedules,
    default_schedule,
    nonlinear_capacity_plan,
)

from factorlab.discovery.neural_forward import FrozenNeuralRecipe
from factorlab.discovery.v7_run import V7Config


def test_standard_default_leaves_explicit_real_forward_and_nonlinear_capacity():
    schedule = default_schedule(V7Config(evaluation_mode='joint', profile_candidates=6), month='2026-10').to_dict()
    assert schedule['allocated_counts']['solve'] == 943
    assert schedule['reserved_seconds'] == 3480
    value = schedule['allocation']
    assert 943 + value['forward_solve_holdback'] + value['nonlinear_solve_holdback'] == 1600
    assert 3480 + value['forward_seconds_holdback'] + value['nonlinear_seconds_holdback'] == 7200
    assert value['nonlinear_max_counts']['neural_fit'] + value['forward_max_counts']['neural_fit'] == 18
    assert value['nonlinear_max_counts']['epoch'] + value['forward_max_counts']['epoch'] == 180
    assert schedule['config']['profile_candidates'] == 6
    assert sum(s['counts'].get('solve') == 198 for s in schedule['slots']) == 3


def test_controls_three_identical_scientific_allocations_share_one_total():
    schedules = default_control_schedules(symbols=['BTCUSDT'], month='2026-10')
    bodies = [s.to_dict() for s in schedules.values()]
    assert sum(b['allocated_counts']['solve'] for b in bodies) == 1011
    assert sum(b['reserved_seconds'] for b in bodies) == 4320
    assert all(b['config']['profile_candidates'] == 2 for b in bodies)
    assert all(b['allocation']['forward_solve_holdback'] == 349 for b in bodies)
    assert all(b['slots'] == bodies[0]['slots'] for b in bodies)


def test_resource_trial_cost_included_without_authorizing_from_boolean():
    recipe = FrozenNeuralRecipe()
    plan = nonlinear_capacity_plan(recipe)
    assert [w['key'] for w in plan['work']] == ['quarterly_probe', 'resource_trial']
    assert sum(w['counts'].get('solve', 0) for w in plan['work']) == 16
    assert sum(w['counts'].get('neural_fit', 0) for w in plan['work']) == 3
    assert sum(w['counts'].get('epoch', 0) for w in plan['work']) == 12
    assert plan['execution_authorized'] is False
    assert nonlinear_capacity_plan(recipe, prior_resource_verified=True)['execution_authorized'] is False


def test_large_recipe_does_not_shorten_epochs_or_gain_a_fresh_pool_of_funds():
    recipe = FrozenNeuralRecipe(config=replace(FrozenNeuralRecipe().config, epochs=20))
    plan = nonlinear_capacity_plan(recipe)
    assert recipe.config.epochs == 20
    assert [w['key'] for w in plan['work']] == ['quarterly_probe']
    assert 'exceeds' in plan['deferred_reason']


def test_unknown_short_original_scope_is_not_filled_with_default_7200_seconds():
    with pytest.raises(ValueError, match='7200s'):
        default_schedule(V7Config(evaluation_mode='joint', research_seconds=3600), month='2026-10')


def test_current_trial_cannot_authorize_same_cutoff_and_nondue_month_skips_probe():
    recipe = FrozenNeuralRecipe()
    plan = nonlinear_capacity_plan(recipe, quarterly_due=False)
    assert [w['key'] for w in plan['work']] == ['resource_trial']
    assert plan['deferred_reason'] == 'complete_S_waits_for_prior_month_resource_publication'
    prior = nonlinear_capacity_plan(recipe, prior_resource_verified=True, quarterly_due=False)
    assert [w['key'] for w in prior['work']] == ['complete_neural_S']
    assert prior['execution_authorized'] is False
