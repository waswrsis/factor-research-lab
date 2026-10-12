"""Explicit new allocation identity; fixed work caps never shorten a fit rule.

An allocation is a ceiling and priority contract, not a guarantee that every
registered learner and all forward streams fit. No unused work credit transfers.
"""
import json
from dataclasses import asdict

from factorlab.protocol import fingerprint, project_path

from .scheduled_search import FrozenSWorkSchedule
from .v7_run import V7Config, monthly_budget

VERSION = 'v7-shared-monthly-allocation-20261005-v1'
FORWARD_PRIORITY = ('incumbent_A0', 'K0_reference', 'HAR_cal_reference',
                    'committed_C_reference', 'committed_C_candidate',
                    'ordinary_V_reference', 'ordinary_V_candidate',
                    'algorithm_V_reference', 'algorithm_V_candidate',
                    'algorithm_qualifier_reference', 'algorithm_qualifier_candidate')


def _slots(families):
    slots = [{'key': 'baseline', 'stage': 'baseline', 'counts': {'solve': 7}, 'seconds': 300}]
    for family in range(families):
        slots.append({'key': f'family:{family}', 'stage': 'family',
                      'counts': {'family': 1, 'gpt_attempt': 2, 'ast': 32}, 'seconds': 300})
        for position in range(8):
            slots.append({'key': f'candidate:{family}:{position}', 'stage': 'candidate',
                          'counts': {'f': 1, 'solve': 198 if position == 0 else 15}, 'seconds': 90})
    slots.append({'key': 'pruning', 'stage': 'pruning', 'counts': {'solve': 27}, 'seconds': 120})
    return slots


def default_schedule(config, *, month):
    """Config remains frozen: 3 folds,60 profiling calls, original P max6/max2."""
    if type(config) is not V7Config or config.evaluation_mode != 'joint' or config.formulas_per_family != 8:
        raise ValueError('Default allocation requires the complete joint eight-program family contract')
    controls = config.budget_policy == 'controls_third'
    if (not controls and config.research_seconds != 7200):
        raise ValueError('Default standard allocation requires its original 7200s outer scope')
    slots = _slots(1 if controls else 3)
    allocation = {'version': VERSION, 'S_solve': sum(s['counts'].get('solve', 0) for s in slots),
        'forward_solve_holdback': 349 if controls else 411, 'nonlinear_solve_holdback': 240,
        'forward_seconds_holdback': 1980 if controls else 1020, 'nonlinear_seconds_holdback': 900,
        'unused_credit_transfer': False, 'shared_ceiling': 1600 if controls else 1133,
        'baseline_solve': 7, 'per_work_seconds_max': 300,
        'S_seconds_sum': sum(s['seconds'] for s in slots),
        'global_S_solve': 1011 if controls else 482,
        'global_S_seconds': 4320 if controls else 1020,
        'nonlinear_max_counts': {'tree_fit': 3, 'neural_fit': 12, 'epoch': 48, 'solve': 240},
        'forward_max_counts': {'tree_fit': 0, 'neural_fit': 6, 'epoch': 132,
                               'solve': 349 if controls else 411},
        'forward_priority': list(FORWARD_PRIORITY),
        'capacity_rule': 'freeze_actual_recipe_cost_before_work;defer_if_any_original_scope_insufficient',
        'P_slot_rule': 'only_position0_per_family_funds_K0_worstcase;others_fail_closed_if_parameters_need_more',
        'unfunded_reason_required': True,
        'algorithm_change': 'explicit_shared_forward_and_nonlinear_holdbacks;no_guarantee_all_opportunities_fit'}
    if not controls and config.search_mode == 'gpt':
        from .prompt_request_kinds import POLICY
        allocation['request_kind_policy'] = json.loads(json.dumps(POLICY))
    return FrozenSWorkSchedule.freeze(config=asdict(config), limits=monthly_budget(config, month).limits,
                                      slots=slots, allocation=allocation)


def default_control_schedules(*, symbols, month):
    return {arm: default_schedule(V7Config(symbols=tuple(symbols), search_mode=arm,
        evaluation_mode='joint', profile_candidates=2, research_seconds=2400,
        budget_policy='controls_third'), month=month) for arm in ('local', 'random', 'gpt')}


def load_schedule(reference, *, config, month):
    """The default is chosen explicitly and expanded to full immutable bytes."""
    if reference == VERSION:
        return default_schedule(config, month=month)
    return FrozenSWorkSchedule.from_dict(json.loads(project_path(reference).read_text('utf-8')))


def load_control_schedules(reference, *, symbols, month):
    if reference == VERSION:
        return default_control_schedules(symbols=symbols, month=month)
    return {key: FrozenSWorkSchedule.from_dict(value) for key, value in
            json.loads(project_path(reference).read_text('utf-8')).items()}


def nonlinear_capacity_plan(recipe, *, prior_resource_verified=False, quarterly_due=True):
    """Planning only. A bool cannot enable work; producer independently verifies receipts.

    Pool-dependent cost is calculated before a reservation. Full3fold S cannot
    borrow a resource trial slot or shorten its epochs/profile/folds to fit.
    """
    from .neural_forward import FrozenNeuralRecipe
    from .neural_research import research_capacity
    if type(recipe) is not FrozenNeuralRecipe or type(prior_resource_verified) is not bool or type(quarterly_due) is not bool:
        raise TypeError('Exact complete frozen recipe and planning state required')
    trial = {k: recipe.budget_limits[k] for k in ('neural_fit', 'epoch', 'solve')}
    research = research_capacity(recipe)
    baseline = recipe.budget_limits['solve'] - 2 * len(recipe.config.seeds)
    probe = {'tree_fit': 3, 'solve': 3 * baseline + 6}
    selected = [('quarterly_probe', probe)] if quarterly_due else []
    if not prior_resource_verified:
        selected.append(('resource_trial', trial))
    # New evidence is published after this cutoff; it can authorize only a later month.
    proposed = [*selected, ('complete_neural_S', research)] if prior_resource_verified else selected
    limits = {'tree_fit': 3, 'neural_fit': 12, 'epoch': 48, 'solve': 240}
    def fits(works):
        return all(sum(counts.get(k, 0) for _, counts in works) <= n for k, n in limits.items())
    reason = None if prior_resource_verified else 'complete_S_waits_for_prior_month_resource_publication'
    if fits(proposed):
        selected = proposed
    else:
        reason = 'pool_dependent_complete_learner_cost_exceeds_original_nonlinear_holdback'
    if not fits(selected):
        selected = [('quarterly_probe', probe)] if quarterly_due and fits([('quarterly_probe', probe)]) else []
        reason = 'pool_dependent_resource_trial_cost_exceeds_original_nonlinear_holdback'
    body = {'version': VERSION, 'recipe_id': recipe.pipeline_id,
            'work': [{'key': key, 'counts': counts, 'seconds': 300} for key, counts in selected],
            'deferred_reason': reason, 'limits': limits, 'seconds_limit': 900,
            'prior_resource_reference_required': prior_resource_verified,
            'resource_available_before_cutoff_required': True, 'quarterly_due': quarterly_due,
            'planning_only': True, 'execution_authorized': False,
            'full_recipe_unchanged': True, 'unused_credit_transfer': False}
    return {**body, 'plan_hash': fingerprint(body)}
