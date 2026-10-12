"""Numeric accounting examples are artificial; they authenticate no DGP or fit."""
from dataclasses import replace

import pytest
from scipy.stats import beta

from factorlab.discovery.calibration_acceptance import (
    CLAIMS, AcceptanceCell, AcceptancePlan, ClaimObservation, OracleRiskEnvelope,
    TrialObservation, acceptance_report,
)
from factorlab.protocol import fingerprint


@pytest.fixture(autouse=True)
def refuse_external_work(monkeypatch):
    """Reversible entry-point guards; no process-wide audit hook or module rewrite."""
    import socket
    import subprocess
    from factorlab.discovery import convex_eval, joint_pool, v7_run
    from factorlab.discovery.v7_agent import CodexBatchBackend

    def refuse(*args, **kwargs):
        raise AssertionError('Numeric accounting tests cannot fit or call a provider')

    monkeypatch.setattr(convex_eval, 'fit_ridge', refuse)
    monkeypatch.setattr(joint_pool, 'evaluate_candidate', refuse)
    monkeypatch.setattr(v7_run, '_baseline_readout', refuse)
    monkeypatch.setattr(CodexBatchBackend, '__call__', refuse)
    monkeypatch.setattr(subprocess, 'Popen', refuse)
    monkeypatch.setattr(socket.socket, 'connect', refuse)
    monkeypatch.setattr(socket, 'create_connection', refuse)


def registered_plan(power=('factor1',), names=('mixed',)):
    return AcceptancePlan(tuple(AcceptanceCell(name, fingerprint(('DGP', name)),
        1000, power) for name in names))


def envelope(plan, truth):
    if truth == 'absent':
        return None
    # Two paired risk coordinates. These fixtures test real interval arithmetic,
    # not a supplied Boolean truth label or an authenticated population law.
    reference, candidate = {
        'null': ((1., 1.), (1., 1.)),
        'alternative': ((1., 1.), (.98, .98)),
        'boundary': ((1., 1.), (.995, .995)),
        'uncertain': ((.99, 1.01), (.98, 1.02)),
        'zero': ((0., 0.), (0., 0.)),
    }[truth]
    return OracleRiskEnvelope(reference, candidate, 'finite_exhaustive_risk_enclosure',
        fingerprint(('numeric-fixture', truth)), plan.oracle_risk_alpha)


def trial(plan, index, *, cell='mixed', state='complete', accepted=('factor1',), truths=None):
    truths = truths or {'factor1': 'alternative'}
    return TrialObservation(cell, index, state, fingerprint(('execution', cell, index)),
        tuple(ClaimObservation(slot, slot in accepted,
            envelope(plan, truths.get(slot, 'null'))) for slot in CLAIMS))


def rows(plan, *, successes=900, cell='mixed'):
    return [trial(plan, i, cell=cell, accepted=('factor1',) if i < successes else ())
        for i in range(1000)]


def assert_nonformal(report):
    assert report['formal_gate_eligible'] is False
    assert report['full_V7'] is False
    assert report['execution_scope_authenticated'] is False
    assert report['oracle_scope_authenticated'] is False
    assert report['power_is_end_to_end_discovery_probability'] is True


@pytest.mark.parametrize('state', ['failed_consumed', 'interrupted_consumed', 'missing'])
def test_one_consumed_or_missing_trial_can_meet_conservative_numeric_targets(state):
    plan = registered_plan()
    observations = rows(plan)
    clean = acceptance_report(plan, observations)
    if state == 'missing':
        observations.pop()
    else:
        observations[-1] = trial(plan, 999, state=state, accepted=())
    report = acceptance_report(plan, observations)
    cell = report['cells'][0]
    assert report['numeric_targets_met'] is True
    assert report['all_trials_complete'] is False
    assert cell['complete'] == 999
    assert cell['failed'] == int(state != 'missing')
    assert cell['missing'] == int(state == 'missing')
    assert cell['registered_repetitions'] == 1000
    assert cell['possible_family_errors'] == 1
    assert cell['certain_family_errors'] == 0
    assert cell['family_error_upper'] == pytest.approx(beta.ppf(1-.025/2, 2, 999))
    assert clean['cells'][0]['family_error_upper'] < cell['family_error_upper'] < .05
    power = cell['discovery_power']['factor1']
    assert power['discoveries'] == 900 and power['denominator'] == 1000
    assert power['lower'] == pytest.approx(beta.ppf(.025/2, 900, 101))
    assert power['lower'] > .8
    assert report['plan_digest'] == clean['plan_digest'] == plan.digest
    assert report['confidence'] == clean['confidence']
    assert_nonformal(report)


def test_lost_success_is_zero_power_and_one_error_not_removed_from_denominator():
    plan = registered_plan()
    observations = rows(plan)
    clean = acceptance_report(plan, observations)
    observations[0] = trial(plan, 0, state='failed_consumed', accepted=())
    report = acceptance_report(plan, observations)
    power = report['cells'][0]['discovery_power']['factor1']
    assert power['discoveries'] == 899 and power['denominator'] == 1000
    assert power['lower'] < clean['cells'][0]['discovery_power']['factor1']['lower']
    assert report['cells'][0]['possible_family_errors'] == 1
    assert report['numeric_targets_met'] is True
    assert_nonformal(report)


@pytest.mark.parametrize('failed_count', [50, 201, 1000])
def test_many_failures_cannot_pass_by_conditioning_on_success(failed_count):
    plan = registered_plan()
    observations = [trial(plan, i, state='failed_consumed', accepted=())
        if i < failed_count else trial(plan, i) for i in range(1000)]
    report = acceptance_report(plan, observations)
    cell = report['cells'][0]
    assert cell['failed'] == cell['possible_family_errors'] == failed_count
    assert cell['family_error_upper'] > .05
    assert cell['discovery_power']['factor1']['discoveries'] == 1000-failed_count
    assert cell['discovery_power']['factor1']['denominator'] == 1000
    if failed_count >= 201:
        assert cell['discovery_power']['factor1']['lower'] < .8
    assert report['numeric_targets_met'] is False
    assert_nonformal(report)


@pytest.mark.parametrize('truth', ['absent', 'uncertain', 'boundary', 'zero'])
def test_unknown_or_boundary_truth_never_manufactures_power(truth):
    plan = registered_plan()
    report = acceptance_report(plan, [trial(plan, i, truths={'factor1': truth})
        for i in range(1000)])
    cell = report['cells'][0]
    assert report['all_trials_complete'] is True
    assert cell['unknown_truth_trials'] == cell['possible_family_errors'] == 1000
    assert cell['certain_family_errors'] == 0
    assert cell['family_error_upper'] == 1.
    assert cell['discovery_power']['factor1']['discoveries'] == 0
    assert cell['discovery_power']['factor1']['lower'] == 0.
    assert report['numeric_targets_met'] is False
    assert_nonformal(report)


def test_five_claim_mixed_truth_family_union_counts_each_trial_once():
    plan = registered_plan(power=('factor1', 'algorithm'))
    truths = dict(zip(CLAIMS, ('null', 'alternative', 'absent', 'null', 'alternative')))
    observations = [trial(plan, i, truths=truths, accepted=('factor1', 'algorithm'))
        for i in range(1000)]
    observations[0] = trial(plan, 0, truths=truths, accepted=CLAIMS)
    observations[1] = trial(plan, 1, truths=truths, accepted=('factor1', 'factor2', 'algorithm'))
    observations[999] = trial(plan, 999, state='failed_consumed', accepted=())
    report = acceptance_report(plan, observations)
    cell = report['cells'][0]
    assert cell['certain_family_errors'] == 1
    assert cell['possible_family_errors'] == 3  # union, unknown accepted, failure
    assert cell['unknown_truth_trials'] == cell['mixed_truth_trials'] == 999
    assert cell['alternative_trials'] == 999
    assert cell['family_error_upper'] == pytest.approx(beta.ppf(1-.025/3, 4, 997))
    for power in cell['discovery_power'].values():
        assert power['discoveries'] == 999 and power['denominator'] == 1000
    assert report['numeric_targets_met'] is True and report['all_trials_complete'] is False
    assert_nonformal(report)


def test_all_five_power_slots_and_multiple_cells_keep_simultaneous_budget():
    plan = registered_plan(power=CLAIMS, names=('mixed', 'second'))
    truth = dict.fromkeys(CLAIMS, 'alternative')
    observations = [trial(plan, i, cell=name, truths=truth,
        accepted=CLAIMS if i < 900 else ()) for name in ('mixed', 'second') for i in range(1000)]
    observations[-1] = trial(plan, 999, cell='second', state='interrupted_consumed', accepted=())
    report = acceptance_report(plan, observations)
    assert report['confidence']['directional_bounds'] == 12
    assert report['confidence']['per_bound_tail_alpha'] == .025/12
    assert report['confidence']['per_oracle_risk_tail_alpha'] == .025/(4*5*2000)
    assert report['confidence']['joint_failure_budget'] == .05
    assert report['plan']['delta0'] == .005
    for cell in report['cells']:
        assert tuple(cell['discovery_power']) == CLAIMS
        for power in cell['discovery_power'].values():
            assert power['lower'] == pytest.approx(beta.ppf(.025/12, 900, 101))
            assert power['target_met'] is True
    assert report['numeric_targets_met'] is True and report['all_trials_complete'] is False
    assert_nonformal(report)


def test_one_failed_cell_blocks_whole_grid_without_averaging():
    plan = registered_plan(names=('mixed', 'second'))
    report = acceptance_report(plan, rows(plan) + rows(plan, successes=750, cell='second'))
    assert report['all_trials_complete'] is True
    assert report['cells'][0]['discovery_power']['factor1']['target_met'] is True
    assert report['cells'][1]['discovery_power']['factor1']['target_met'] is False
    assert report['numeric_targets_met'] is False
    assert_nonformal(report)


def test_absence_retains_all_registered_denominators_and_no_success_fields():
    plan = registered_plan(power=CLAIMS)
    report = acceptance_report(plan, [])
    cell = report['cells'][0]
    for name in ('complete', 'failed', 'certain_family_errors', 'unknown_truth_trials',
                 'alternative_trials', 'mixed_truth_trials'):
        assert cell[name] == 0
    assert cell['registered_repetitions'] == cell['missing'] == cell['possible_family_errors'] == 1000
    assert cell['family_error_upper'] == 1. and cell['error_target_met'] is False
    for power in cell['discovery_power'].values():
        assert power == {'discoveries': 0, 'denominator': 1000, 'lower': 0.,
                         'target': .8, 'target_met': False}
    assert report['numeric_targets_met'] is False and report['all_trials_complete'] is False
    assert_nonformal(report)


@pytest.mark.parametrize('violation', ['duplicate', 'execution', 'outside_index', 'outside_cell'])
def test_invalid_grid_evidence_is_refused_before_any_report(violation):
    plan = registered_plan()
    original = trial(plan, 0)
    extra = {'duplicate': original, 'execution': replace(original, index=1),
        'outside_index': replace(original, index=1000),
        'outside_cell': replace(original, cell='unregistered')}[violation]
    with pytest.raises(ValueError):
        acceptance_report(plan, [original, extra])


def test_partial_reordered_and_successful_failed_claims_are_refused():
    plan = registered_plan()
    original = trial(plan, 0)
    for claims in (original.claims[:-1], original.claims[::-1]):
        with pytest.raises(ValueError):
            replace(original, claims=claims)
    with pytest.raises(ValueError):
        replace(original, state='failed_consumed')


@pytest.mark.parametrize('field,value', [('delta0', .006), ('monte_carlo_alpha', .05),
    ('oracle_alpha', 0.), ('family_error_target', .051), ('discovery_power_target', .79)])
def test_original_thresholds_and_budget_cannot_be_relaxed(field, value):
    with pytest.raises(ValueError):
        replace(registered_plan(), **{field: value})
