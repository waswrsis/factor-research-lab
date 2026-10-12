"""Error-budget inversion and unconditional full-repetition accounting."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import binom, binomtest

from factorlab.discovery.calibration_acceptance import (
    CLAIMS,
    AcceptanceCell,
    AcceptancePlan,
    ClaimObservation,
    OracleRiskEnvelope,
    TrialObservation,
    acceptance_report,
    bounded_path_envelope,
    exact_binomial_bounds,
)
from factorlab.protocol import fingerprint


def plan(*, power=('factor1',)):
    return AcceptancePlan((AcceptanceCell('mixed', fingerprint('mixed-dgp'), 1000, power),))


def oracle(p, kind):
    candidate = {'null': 1., 'alternative': .98, 'uncertain': .995}[kind]
    return OracleRiskEnvelope((1., 1.), (candidate, candidate),
        'finite_exhaustive_risk_enclosure', fingerprint(('oracle', kind)), p.oracle_risk_alpha)


def row(p, index, *, state='complete', accepted=('factor1',), kind='alternative'):
    return TrialObservation('mixed', index, state, fingerprint(('execution', index)), tuple(
        ClaimObservation(s, s in accepted, oracle(p, kind if s == 'factor1' else 'null'))
        for s in CLAIMS))


def test_one_sided_exact_inversion_not_two_sided_or_zero_width():
    for k, n in [(0, 1000), (12, 1000), (900, 1000), (1000, 1000)]:
        low, high = exact_binomial_bounds(k, n, tail_alpha=.025)
        interval = binomtest(k, n).proportion_ci(confidence_level=.95, method='exact')
        assert low == pytest.approx(interval.low, abs=2e-12)
        assert high == pytest.approx(interval.high, abs=2e-12)
        if k < n:
            assert binom.cdf(k, n, high) == pytest.approx(.025, abs=1e-12)
        if k:
            assert binom.sf(k-1, n, low) == pytest.approx(.025, abs=1e-12)
    assert exact_binomial_bounds(0, 1000, tail_alpha=.025)[1] > 0


def test_confidence_coverage_at_low_medium_and_high_error_probabilities():
    # Sum exact binomial rejection probabilities instead of a flaky sampled
    # coverage test. Separate directional coverage must hold at every p here.
    n, a = 60, .025
    bounds = np.asarray([exact_binomial_bounds(k, n, tail_alpha=a) for k in range(n+1)])
    for p in [0., .0001, .005, .05, .5, .8, .9999, 1.]:
        probability = binom.pmf(np.arange(n+1), n, p)
        assert probability[bounds[:, 1] < p].sum() <= a + 1e-12
        assert probability[bounds[:, 0] > p].sum() <= a + 1e-12


@pytest.mark.parametrize('k,n,a', [(True, 10, .025), (0, 0, .025), (11, 10, .025),
    (1, 10, float('nan')), (0, 10, 0), (1, 10, .5)])
def test_invalid_mc_counts_and_confidence_refused(k, n, a):
    with pytest.raises(ValueError):
        exact_binomial_bounds(k, n, tail_alpha=a)


def test_every_registered_claim_bound_and_oracle_consumes_confidence():
    a = plan()
    b = AcceptancePlan(a.cells + (AcceptanceCell('drift', fingerprint('drift'), 2000, CLAIMS),))
    assert b.tail_alpha == .025 / 8
    assert b.oracle_risk_alpha == .025 / (3000 * 20)
    assert b.tail_alpha < a.tail_alpha and b.oracle_risk_alpha < a.oracle_risk_alpha
    with pytest.raises(ValueError, match='full registered-grid'):
        oracle(a, 'alternative').truth(b)


def test_relative_risk_truth_has_zero_boundary_and_uncertain_cases():
    p = plan()
    assert oracle(p, 'null').truth(p) == 'null'
    assert oracle(p, 'alternative').truth(p) == 'alternative'
    assert oracle(p, 'uncertain').truth(p) != 'alternative'
    zero = replace(oracle(p, 'null'), reference=(0., 0.), candidate=(0., 0.))
    assert zero.effect_interval() is None and zero.truth(p) == 'uncertain'
    wide = replace(oracle(p, 'alternative'), candidate=(.98, 1.02))
    assert wide.truth(p) == 'uncertain'
    # Extreme endpoints fail closed rather than overflowing Fraction conversion.
    extreme = replace(oracle(p, 'null'), reference=(1e308, float(np.finfo(float).max)))
    assert extreme.truth(p) == 'uncertain'
    tiny = replace(oracle(p, 'null'), reference=(1e-308, 1e-308), candidate=(1e308, 1e308))
    assert tiny.truth(p) == 'uncertain'


def test_path_oracle_cannot_flatten_dependent_hours_or_use_truncated_paths():
    p = plan()
    args = {'loss_upper_bound': 2., 'expected_paths': 1000,
            'oracle_digest': fingerprint('bounded-DGP'), 'plan': p}
    result = bounded_path_envelope(np.ones(1000), np.full(1000, .98), **args)
    assert result.truth(p) == 'uncertain'  # tiny effect is not a known injected truth
    with pytest.raises(ValueError, match='path losses'):
        bounded_path_envelope(np.ones((1000, 24)), np.ones((1000, 24)), **args)
    with pytest.raises(ValueError, match='path losses'):
        bounded_path_envelope(np.ones(999), np.ones(999), **args)
    with pytest.raises(ValueError, match='path losses'):
        bounded_path_envelope(np.full(1000, 3.), np.ones(1000), **args)
    with pytest.raises(ValueError, match='numeric'):
        bounded_path_envelope(np.ones(1000), np.ones(1000), **{**args, 'loss_upper_bound': float('inf')})


def test_mixed_truth_full_repetitions_can_pass_numbers_without_minting_certificate():
    p = plan()
    report = acceptance_report(p, [row(p, i) for i in range(1000)])
    c = report['cells'][0]
    assert c['mixed_truth_trials'] == 1000
    assert c['possible_family_errors'] == 0
    assert c['family_error_upper'] < .05
    assert c['discovery_power']['factor1']['lower'] > .8
    assert report['numeric_targets_met'] is True
    assert report['formal_gate_eligible'] is report['full_V7'] is False
    assert not report['execution_scope_authenticated'] and not report['oracle_scope_authenticated']


def test_fewer_successes_unknown_truth_and_failures_cannot_improve_report():
    p = plan()
    records = [row(p, i) for i in range(900)]
    records += [row(p, i, state='failed_consumed', accepted=()) for i in range(900, 950)]
    records += [row(p, i, accepted=(), kind='uncertain') for i in range(950, 1000)]
    r = acceptance_report(p, records)['cells'][0]
    assert r['registered_repetitions'] == 1000
    assert r['failed'] == 50 and r['possible_family_errors'] == 50
    assert r['discovery_power']['factor1']['discoveries'] == 900
    assert r['discovery_power']['factor1']['denominator'] == 1000
    smaller = acceptance_report(p, records[:900])
    assert smaller['cells'][0]['possible_family_errors'] == 100
    assert smaller['cells'][0]['family_error_upper'] > r['family_error_upper']
    assert smaller['numeric_targets_met'] is False


def test_accepted_null_or_unknown_claims_count_jointly_not_per_claim_average():
    p = plan()
    records = [row(p, i, accepted=('factor1', 'factor2') if i < 50 else ('factor1',))
               for i in range(1000)]
    r = acceptance_report(p, records)['cells'][0]
    assert r['certain_family_errors'] == r['possible_family_errors'] == 50
    assert r['family_error_upper'] > .05
    uncertain = [row(p, i, kind='uncertain') for i in range(1000)]
    r = acceptance_report(p, uncertain)['cells'][0]
    assert r['certain_family_errors'] == 0 and r['possible_family_errors'] == 1000
    assert r['discovery_power']['factor1']['discoveries'] == 0


def test_nondiscovery_keeps_end_to_end_power_denominator():
    p = plan()
    r = acceptance_report(p, [row(p, i, accepted=() if i >= 750 else ('factor1',))
                             for i in range(1000)])
    power = r['cells'][0]['discovery_power']['factor1']
    assert power['denominator'] == 1000 and power['discoveries'] == 750
    assert power['lower'] < .8 and not r['numeric_targets_met']


def test_no_registered_executions_is_worst_case_not_a_zero_error_pass():
    report = acceptance_report(plan(), [])
    assert report['cells'][0]['missing'] == 1000
    assert report['cells'][0]['family_error_upper'] == 1.
    assert report['cells'][0]['discovery_power']['factor1']['lower'] == 0.
    assert not report['numeric_targets_met']


def test_duplicate_seed_execution_grid_and_claim_scope_refused():
    p = plan()
    first = row(p, 0)
    for second in [first, replace(first, index=1), replace(first, index=1000), replace(first, cell='new')]:
        with pytest.raises(ValueError):
            acceptance_report(p, [first, second])
    with pytest.raises(ValueError, match='five ordered'):
        replace(first, claims=tuple(reversed(first.claims)))
    with pytest.raises(ValueError, match='Failed trial'):
        replace(first, state='failed_consumed')
