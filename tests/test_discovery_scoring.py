"""Independent scoring checks on explicit outcomes, not market evidence."""
import json

import numpy as np
import pandas as pd
import pytest

from factorlab.discovery.contracts import ExperimentSpec, TaskDataset, TaskSpec
from factorlab.discovery.scoring import (
    ProbabilityCalibrator,
    adaptive_horizons,
    evaluate,
    paired_comparison,
    per_sample_loss,
)

EVENT = TaskSpec('event', 'first_event', 4, (1, 4))
RETURN = TaskSpec('return_task', 'return', 4)
VOLATILITY = TaskSpec()
STATE = TaskSpec('state', 'state')


def dataset(hours=24, symbols=('A',), tasks=(RETURN,)):
    times = pd.date_range('2025-01-01', periods=hours, freq='h', tz='UTC')
    n = hours * len(symbols)
    meta = pd.DataFrame({'sample_id': [f's{i}' for i in range(n)],
                         'symbol': np.tile(symbols, hours),
                         'available_at': times.repeat(len(symbols))})
    available = meta.available_at.to_numpy(dtype='datetime64[ns]')
    return TaskDataset(
        np.zeros((n, 2, 1), dtype=np.float32),
        {t.name: np.ones(n) if t.kind == 'volatility' else np.zeros(n) for t in tasks},
        {t.name: np.ones(n, dtype=bool) for t in tasks},
        {t.name: available + np.timedelta64(t.horizon_hours if t.predictive else 0, 'h')
         for t in tasks}, meta, ('x',), tasks,
    )


def experiment(tasks=(RETURN,), **kwargs):
    return ExperimentSpec(tasks=tasks, primary_task=next(t.name for t in tasks if t.predictive),
                          bootstrap_block_hours=4, bootstrap_samples=500, min_blocks=3,
                          **kwargs)


def test_losses_match_independently_calculated_examples():
    np.testing.assert_allclose(per_sample_loss(VOLATILITY, np.array([2., 4.]),
                                             np.array([4., 2.])), np.log(2) ** 2)
    np.testing.assert_allclose(per_sample_loss(RETURN, np.array([1., -1.]),
                                             np.array([2., 1.])), [1, 4])
    np.testing.assert_allclose(per_sample_loss(STATE, np.array([1., 0.]),
                                             np.array([[0.], [2.]])), [1, 4])
    p = np.array([[.4, .1, .1, .2, .2], [.4, .1, .1, .2, .2]])
    np.testing.assert_allclose(per_sample_loss(EVENT, np.array([0, 4]), p),
                               [-np.log(.4), -np.log(.2)])


def test_zero_volatility_and_zero_event_probability_are_finite():
    loss = per_sample_loss(VOLATILITY, np.array([0., 0.]), np.array([0., 0.]))
    np.testing.assert_array_equal(loss, [0., 0.])
    loss = per_sample_loss(EVENT, np.array([0]), np.array([[0., 1., 0., 0., 0.]]))
    assert np.isfinite(loss).all() and loss[0] > 20


def test_volatility_near_floor_matches_training_log_plus_floor_target():
    y = np.array([0., 1e-8, 2e-8])
    p = np.array([1e-8, 2e-8, 0.])
    expected = np.square(np.log(np.array([2., 3., 1.])) - np.log(np.array([1., 2., 3.])))
    np.testing.assert_allclose(per_sample_loss(VOLATILITY, y, p), expected)
    for target, prediction in (([-1.], [0.]), ([0.], [-1.])):
        with pytest.raises(ValueError, match='nonnegative'):
            per_sample_loss(VOLATILITY, np.array(target), np.array(prediction))


@pytest.mark.parametrize('p', [
    [[.1, .1, .1, .1, .1]],
    [[-.1, .3, .3, .3, .2]],
    [[1.1, -.1, 0, 0, 0]],
    [[np.nan, .25, .25, .25, .25]],
    [[np.inf, 0, 0, 0, 0]],
])
def test_illegal_joint_probabilities_rejected(p):
    with pytest.raises(ValueError):
        per_sample_loss(EVENT, np.array([0]), np.array(p))


def test_targets_and_prediction_shapes_are_not_silently_broadcast():
    with pytest.raises(ValueError, match='align'):
        per_sample_loss(RETURN, np.zeros(3), np.zeros(2))
    with pytest.raises(ValueError, match='finite'):
        per_sample_loss(RETURN, np.zeros(3), np.array([0., np.nan, 0.]))
    with pytest.raises(ValueError, match='support'):
        per_sample_loss(EVENT, np.array([.5]), np.full((1, 5), .2))


def test_evaluate_reports_missing_tasks_and_description_separately():
    data = dataset(tasks=(RETURN, EVENT, STATE))
    results = evaluate(data, {RETURN.name: np.zeros(len(data.meta)),
                              STATE.name: np.zeros(len(data.meta))})
    assert results[RETURN.name]['loss'] == 0
    assert results[EVENT.name]['status'] == 'unavailable'
    assert results[EVENT.name]['coverage'] == 0
    assert results[EVENT.name]['support'] == 0
    assert results[EVENT.name]['eligible_support'] == len(data.meta)
    assert not results[STATE.name]['predictive']
    assert results[STATE.name]['interpretation'] == 'descriptive_only'
    json.dumps(results, allow_nan=False)


def test_missing_prediction_inside_eligible_rows_is_an_error():
    data = dataset()
    p = np.zeros(len(data.meta))
    p[4] = np.nan
    with pytest.raises(ValueError, match='finite'):
        evaluate(data, {RETURN.name: p})
    with pytest.raises(ValueError, match='all dataset rows'):
        evaluate(data, {RETURN.name: p[:-1]})
    with pytest.raises(ValueError, match='finite'):
        paired_comparison(data, {RETURN.name: p}, {RETURN.name: np.zeros(len(p))},
                          RETURN.name, experiment())
    with pytest.raises(ValueError, match='Missing task'):
        paired_comparison(data, {}, {RETURN.name: np.zeros(len(p))}, RETURN.name, experiment())


def test_unobserved_labels_are_not_scored_but_coverage_is_preserved():
    data = dataset(hours=8)
    data.observed[RETURN.name][0] = False
    data.targets[RETURN.name][0] = np.nan
    p = np.zeros(8)
    result = evaluate(data, {RETURN.name: p})[RETURN.name]
    assert result['support'] == 7 and result['coverage'] == 7 / 8
    assert result['loss'] == 0
    data.observed[RETURN.name][:] = False
    result = evaluate(data, {RETURN.name: p})[RETURN.name]
    assert result['status'] == 'insufficient_evidence' and result['loss'] is None


def test_numerical_prediction_failure_is_rejected_even_with_unobserved_label():
    data = dataset(hours=8, tasks=(RETURN, EVENT, VOLATILITY))
    for task in data.tasks:
        data.observed[task.name][0] = False
    valid = {RETURN.name: np.zeros(8), EVENT.name: np.full((8, 5), .2),
             VOLATILITY.name: np.ones(8)}
    for task in data.tasks:
        failed = {k: v.copy() for k, v in valid.items()}
        failed[task.name][0] = np.nan
        with pytest.raises(ValueError, match='all dataset rows'):
            evaluate(data, failed)
        with pytest.raises(ValueError, match='all dataset rows'):
            paired_comparison(data, failed, valid, task.name,
                              experiment(tasks=data.tasks))
    for bad in ([-.1, .3, .3, .3, .2], [0., 0., 0., 0., 0.]):
        failed = {k: v.copy() for k, v in valid.items()}
        failed[EVENT.name][0] = bad
        with pytest.raises(ValueError, match='probabilit'):
            evaluate(data, failed)
    failed = {k: v.copy() for k, v in valid.items()}
    failed[VOLATILITY.name][0] = -1.
    with pytest.raises(ValueError, match='nonnegative'):
        evaluate(data, failed)


def test_full_joint_brier_and_calibration_diagnostics_have_support():
    data = dataset(hours=10, tasks=(EVENT,))
    data.targets[EVENT.name][:] = np.arange(10) % 5
    p = np.full((10, 5), .2)
    result = evaluate(data, {EVENT.name: p})[EVENT.name]
    assert result['loss'] == pytest.approx(np.log(5))
    assert result['brier'] == pytest.approx(.8)
    assert sum(b['support'] for b in result['calibration_bins']) == 10
    assert result['expected_calibration_error'] == pytest.approx(0)
    assert result['calibration_is_diagnostic_only']


def test_identical_and_constant_improvement_produce_exact_intervals():
    data = dataset()
    zero = {RETURN.name: np.zeros(24)}
    reference = {RETURN.name: np.full(24, 2.)}
    result = paired_comparison(data, zero, reference, RETURN.name, experiment())
    assert result['status'] == 'supported'
    assert result['estimate'] == result['ci_lower'] == result['ci_upper'] == 4
    assert result['n_blocks'] == 6 and result['n_hours'] == 24
    assert result['predictive_claim_supported']
    result = paired_comparison(data, zero, zero, RETURN.name, experiment())
    assert result['estimate'] == result['ci_lower'] == result['ci_upper'] == 0
    assert result['status'] == 'target_not_met'
    result = paired_comparison(data, zero, reference, RETURN.name,
                               experiment(minimum_increment=5))
    assert result['status'] == 'target_not_met'


def test_simultaneous_symbol_duplication_does_not_create_more_evidence():
    single = dataset(hours=24)
    many = dataset(hours=24, symbols=('A', 'B', 'C', 'D'))
    losses = np.repeat(np.array([1., 2., 3., 1., 4., 2.]), 4)
    one = paired_comparison(single, {RETURN.name: np.zeros(24)},
                            {RETURN.name: np.sqrt(losses)}, RETURN.name, experiment())
    four = paired_comparison(many, {RETURN.name: np.zeros(96)},
                             {RETURN.name: np.repeat(np.sqrt(losses), 4)},
                             RETURN.name, experiment())
    for key in ('estimate', 'ci_lower', 'ci_upper', 'n_blocks', 'n_hours'):
        assert one[key] == four[key]
    assert four['support'] == 4 * one['support']


def test_equal_hour_estimate_is_not_distorted_by_asset_coverage():
    data = dataset(hours=2, symbols=('A', 'B')).subset(np.array([True, True, True, False]))
    spec = ExperimentSpec(tasks=(RETURN,), primary_task=RETURN.name,
                          bootstrap_block_hours=4, bootstrap_samples=30, min_blocks=1)
    result = paired_comparison(data, {RETURN.name: np.zeros(3)},
                               {RETURN.name: np.array([1., 3., 1.])}, RETURN.name, spec)
    assert result['estimate'] == 3  # Hour means: (1+9)/2=5, then 1; mean=3.
    assert result['conditional_scope_only']


def test_insufficient_time_blocks_and_calendar_gaps_are_explicit():
    data = dataset(hours=8, symbols=('A', 'B', 'C'))
    result = paired_comparison(data, {RETURN.name: np.zeros(24)},
                               {RETURN.name: np.ones(24)}, RETURN.name, experiment())
    assert result['status'] == 'insufficient_evidence'
    assert result['n_blocks'] == 2 and result['ci_lower'] is None
    assert result['reason'] == 'minimum_time_blocks_not_met'
    sparse = dataset(hours=24).subset(np.array([0, 8, 16]))
    result = paired_comparison(sparse, {RETURN.name: np.zeros(3)},
                               {RETURN.name: np.ones(3)}, RETURN.name, experiment())
    assert result['n_hours'] == 3 and result['n_blocks'] == 3


def test_family_adjustment_widens_same_paired_bootstrap_interval():
    data = dataset(hours=48)
    losses = np.repeat(np.arange(12, dtype=float) + 1, 4)
    candidate, reference = {RETURN.name: np.zeros(48)}, {RETURN.name: np.sqrt(losses)}
    one = paired_comparison(data, candidate, reference, RETURN.name, experiment())
    ten = paired_comparison(data, candidate, reference, RETURN.name, experiment(), family_size=10)
    assert ten['adjusted_confidence'] == pytest.approx(.995)
    assert ten['ci_lower'] <= one['ci_lower'] and ten['ci_upper'] >= one['ci_upper']
    with pytest.raises(ValueError, match='family'):
        paired_comparison(data, candidate, reference, RETURN.name, experiment(), family_size=0)


def test_uncertain_increment_is_not_promoted():
    data = dataset(hours=48)
    loss = np.tile(np.repeat([0., 2.], 4), 6)
    result = paired_comparison(data, {RETURN.name: np.sqrt(loss)},
                               {RETURN.name: np.ones(48)}, RETURN.name, experiment())
    assert result['ci_lower'] < 0 < result['ci_upper']
    assert result['status'] == 'insufficient_evidence'


def test_descriptive_improvement_cannot_support_a_predictive_claim():
    data = dataset(tasks=(RETURN, STATE))
    result = paired_comparison(data, {STATE.name: np.zeros(24)},
                               {STATE.name: np.ones(24)}, STATE.name,
                               experiment(tasks=(RETURN, STATE)))
    assert result['status'] == 'supported'
    assert not result['predictive'] and not result['predictive_claim_supported']


def test_temperature_improves_joint_nll_without_changing_regression(tmp_path):
    data = dataset(hours=100, tasks=(EVENT, RETURN))
    data.targets[EVENT.name][:] = np.arange(100) % 5
    probabilities = np.tile([.8, .05, .05, .05, .05], (100, 1))
    regression = np.linspace(-1, 1, 100)
    predictions = {EVENT.name: probabilities, RETURN.name: regression}
    calibrator = ProbabilityCalibrator().fit(data, predictions)
    result = calibrator.transform(predictions)
    assert calibrator.diagnostics[EVENT.name]['status'] == 'fitted'
    before = per_sample_loss(EVENT, data.targets[EVENT.name], probabilities).mean()
    after = per_sample_loss(EVENT, data.targets[EVENT.name], result[EVENT.name]).mean()
    assert after < before - .5
    np.testing.assert_allclose(result[EVENT.name].sum(axis=1), 1.)
    np.testing.assert_array_equal(result[RETURN.name], regression)
    np.testing.assert_array_equal(predictions[EVENT.name], probabilities)
    restored = ProbabilityCalibrator.from_dict(json.loads(json.dumps(calibrator.to_dict())))
    np.testing.assert_array_equal(restored.transform(predictions)[EVENT.name], result[EVENT.name])
    path = tmp_path / 'calibrator.json'
    calibrator.save(path)
    np.testing.assert_array_equal(ProbabilityCalibrator.load(path).transform(predictions)[EVENT.name],
                                  result[EVENT.name])


def test_calibrator_missing_or_small_support_is_identity_with_status():
    data = dataset(hours=3, tasks=(EVENT, RETURN))
    probabilities = np.full((3, 5), .2)
    calibrator = ProbabilityCalibrator().fit(data, {EVENT.name: probabilities})
    assert calibrator.diagnostics[EVENT.name]['status'] == 'insufficient_support'
    np.testing.assert_array_equal(calibrator.transform({EVENT.name: probabilities})[EVENT.name],
                                  probabilities)
    data.observed[EVENT.name][:] = False
    calibrator.fit(data, {EVENT.name: probabilities})
    assert calibrator.diagnostics[EVENT.name]['status'] == 'no_support'
    calibrator.fit(data, {})
    assert calibrator.diagnostics[EVENT.name]['status'] == 'unavailable'
    assert calibrator.transform({}) == {}


def test_calibration_uses_only_declared_observed_support_but_rejects_bad_eligible_p():
    data = dataset(hours=30, tasks=(EVENT,))
    p = np.full((30, 5), .2)
    data.observed[EVENT.name][0] = False
    data.targets[EVENT.name][0] = np.nan
    calibrator = ProbabilityCalibrator().fit(data, {EVENT.name: p})
    assert calibrator.diagnostics[EVENT.name]['support'] == 29
    p[0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        ProbabilityCalibrator().fit(data, {EVENT.name: p})


def test_calibrator_requires_fit_and_rejects_invalid_serialized_temperature():
    with pytest.raises(ValueError, match='fitted'):
        ProbabilityCalibrator().transform({})
    data = dataset(hours=2, tasks=(EVENT,))
    value = ProbabilityCalibrator().fit(data, {}).to_dict()
    value['temperatures'][EVENT.name] = -1
    with pytest.raises(ValueError, match='temperature'):
        ProbabilityCalibrator.from_dict(value)


def test_adaptive_horizon_preserves_complete_joint_mass_and_noevent_semantics():
    p = np.array([[.1, .4, .1, .2, .2], [.9, .05, 0., .05, 0.]])
    result = adaptive_horizons(p, EVENT, coverage_threshold=.5)
    np.testing.assert_array_equal(result['horizon_hours'], [1, 4])
    np.testing.assert_allclose(result['cumulative_up'], [[.4, .6], [.05, .1]])
    np.testing.assert_allclose(result['no_event_by_horizon'], [[.5, .1], [.95, .9]])
    np.testing.assert_array_equal(result['coverage_achieved'], [True, False])
    total = (result['selected_up_probability'] + result['selected_down_probability']
             + result['selected_no_event_probability'])
    np.testing.assert_allclose(total, 1.)
    assert not result['concentration_is_confidence']
    assert result['evaluation_target'] == 'complete_joint_first_event_distribution'
    # Horizon selection does not replace full-joint event-time scoring.
    np.testing.assert_allclose(per_sample_loss(EVENT, np.array([3, 0]), p),
                               [-np.log(.2), -np.log(.9)])


def test_adaptive_horizon_complete_noevent_and_invalid_mass():
    result = adaptive_horizons(np.array([[1., 0., 0., 0., 0.]]), EVENT)
    assert result['horizon_hours'][0] == 4 and not result['coverage_achieved'][0]
    assert result['event_concentration'][0] == 0
    with pytest.raises(ValueError, match='mass'):
        adaptive_horizons(np.zeros((1, 5)), EVENT)
    with pytest.raises(ValueError, match='threshold'):
        adaptive_horizons(np.full((1, 5), .2), EVENT, coverage_threshold=0)
