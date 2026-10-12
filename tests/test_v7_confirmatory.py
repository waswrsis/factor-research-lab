import numpy as np
import pandas as pd
import pytest

from factorlab.discovery.confirmatory import (
    ConfirmationRule,
    fixed_candidate_selection_diagnostics,
    paired_effect_interval,
    wilson_interval,
)


def rule(**kwargs):
    return ConfirmationRule(block_hours=kwargs.pop('block_hours', 1),
                            bootstrap_samples=1000, **kwargs)


def test_hour_equal_aggregation_and_explicit_symbol_hour_coverage():
    time = pd.date_range('2026-01-01', periods=24, freq='h', tz='UTC')
    origins = time.repeat(2).delete(1)
    symbols = np.tile(['A', 'B'], 24)[1:]
    # First hour has only A; all other hours have A and B. Pair coverage 47/48
    # passes .95, yet that first hour must retain the same time weight.
    symbols[0] = 'A'
    base = np.ones(47)
    candidate = np.r_[0., np.full(46, 2.)]
    result = paired_effect_interval(origins, base, candidate, symbols=symbols,
                                    expected_symbols=['A', 'B'], eligible_count=24, rule=rule())
    expected = np.mean(np.r_[.995, np.full(23, -1.005)])
    assert result['d_mean'] == pytest.approx(expected)
    assert result['coverage'] == 1 and result['pair_coverage'] == pytest.approx(47/48)
    assert result['support'] == 24 and result['pair_support'] == 47
    assert result['relative_gain'] == pytest.approx(1 - 46/24)


def test_duplicate_hours_and_symbol_pairs_are_rejected():
    time = pd.date_range('2026-01-01', periods=12, freq='h', tz='UTC').repeat(2)
    with pytest.raises(ValueError, match='unique UTC hours'):
        paired_effect_interval(time, np.ones(24), np.ones(24), rule=rule())
    with pytest.raises(ValueError, match='Duplicate'):
        paired_effect_interval(time, np.ones(24), np.ones(24), rule=rule(),
                               symbols=['A'] * 24, expected_symbols=['A'])
    with pytest.raises(ValueError, match='universe'):
        paired_effect_interval(time, np.ones(24), np.ones(24), rule=rule(),
                               symbols=['A', 'B'] * 12)


def test_missing_whole_symbol_cannot_shrink_expected_universe():
    time = pd.date_range('2026-01-01', periods=24, freq='h', tz='UTC')
    result = paired_effect_interval(time, np.ones(24), np.zeros(24), rule=rule(),
                                    symbols=['A'] * 24, expected_symbols=['A', 'B'])
    assert result['coverage'] == 1 and result['pair_coverage'] == .5
    assert result['status'] == 'insufficient_evidence' and result['supported'] is False
    assert 'insufficient_symbol_hour_coverage' in result['reasons']


def test_gap_splits_calendar_block_and_short_fragments_do_not_create_support():
    # Epoch anchor aligns these blocks. One omitted middle hour splits a block
    # into 2h and 1h fragments, not one fictitious 3h continuous block.
    time = pd.date_range('1970-01-01', periods=52, freq='h', tz='UTC').delete(2)
    result = paired_effect_interval(time, np.ones(51), np.full(51, .98),
                                    eligible_count=52, rule=rule(block_hours=4))
    assert result['gap_count'] == 1
    assert result['block_hour_counts'][:2] == [2, 1]
    assert result['full_blocks'] == 12 and result['partial_blocks'] == 2
    fragmented = time[::2]
    short = paired_effect_interval(fragmented, np.ones(len(fragmented)),
                                   np.zeros(len(fragmented)), rule=rule(block_hours=4))
    assert short['blocks'] >= 12 and short['full_blocks'] == 0
    assert short['status'] == 'insufficient_evidence'


def test_uncalibrated_positive_diagnostic_never_promotes_and_units_are_invariant():
    time = pd.date_range('2026-01-01', periods=24, freq='h', tz='UTC')
    baseline, candidate = np.ones(24), np.full(24, .97)
    result = paired_effect_interval(time, baseline, candidate, rule=rule())
    assert result['diagnostic_status'] == 'supported_target'
    assert result['status'] == 'uncalibrated_diagnostic' and result['supported'] is False
    converted = paired_effect_interval(time.as_unit('ms'), baseline, candidate, rule=rule())
    assert converted == result
    with pytest.raises(ValueError, match='boolean'):
        paired_effect_interval(time, baseline, candidate, rule=rule(), calibration_passed='true')


def test_zero_reference_and_empty_support_are_not_ratio_claims():
    time = pd.date_range('2026-01-01', periods=24, freq='h', tz='UTC')
    result = paired_effect_interval(time, np.zeros(24), np.zeros(24), rule=rule())
    assert result['relative_gain'] is None and result['absolute_delta'] == 0
    assert result['diagnostic_status'] == 'target_not_reached'
    empty = paired_effect_interval(time[:0], [], [], rule=rule(), eligible_count=24)
    assert empty['status'] == 'insufficient_evidence'
    assert 'empty_support' in empty['reasons']


def test_wilson_bounds_and_invalid_confidence():
    lower, upper = wilson_interval(0, 1000)
    assert lower == pytest.approx(0) and 0 < upper < .005
    with pytest.raises(ValueError, match='Confidence'):
        wilson_interval(0, 1000, confidence=1)


def test_mixed_fixed_candidate_selection_control_cannot_issue_calibration_certificate():
    result = fixed_candidate_selection_diagnostics(repetitions=20, hours=24*100, seed=19)
    assert result['full_pipeline_calibrated'] is False
    assert result['formal_gate_eligible'] is False
    assert result['includes_gpt_or_fitted_features'] is False
    assert result['fwer_bound_is_calibration_certificate'] is False
    assert len(result['claims']) == 5
    assert any(row['has_true_candidate'] for row in result['claims'])
    assert any(not row['has_true_candidate'] for row in result['claims'])
    for claim in result['claims']:
        assert sum(claim['selected']) + claim['no_selection'] == 20
    assert 0 <= result['empirical_fwer'] <= result['fwer_wilson_one_sided_95_upper'] <= 1
