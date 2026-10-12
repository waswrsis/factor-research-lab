"""Hand-computed target, receipt and loss boundary controls; no market data."""
import numpy as np
import pandas as pd
import pytest

from factorlab.discovery.target_experiments import (
    TargetProtocol,
    build_variance_targets,
    compare_variance_forecasts,
    variance_losses,
)


def fixture():
    times = pd.date_range('2026-01-01T23:55Z', periods=160, freq='5min')
    bars = pd.DataFrame({'symbol': 'X', 'open_time': times,
                         'close': np.exp(np.arange(160) * .001 + 4),
                         'received_at': times + pd.Timedelta(minutes=5, seconds=2)})
    origin = pd.DataFrame({'symbol': ['X'], 'available_at': [pd.Timestamp('2026-01-01T23:59Z')],
                           'forecast_window_start': [pd.Timestamp('2026-01-02T00:00Z')],
                           'sample_id': ['x0']})
    return bars, origin


def test_exact_144_returns_variance_not_volatility_and_distinct_task_identity():
    bars, origins = fixture()
    target = build_variance_targets(bars, origins, as_of='2026-01-02T13:00Z')
    row = target.iloc[0]
    assert row.observed and row.return_count == 144
    assert row.target == pytest.approx(144 * .001**2, rel=1e-11)
    assert row.label_available_at == pd.Timestamp('2026-01-02T12:00:02Z')
    assert row.quantity == 'variance_sum_squared_log_returns'
    assert target.attrs['target_id'] == TargetProtocol().digest
    assert 'no_hourly_task_replacement' in target.attrs['scope']
    with pytest.raises(ValueError):
        TargetProtocol(horizon_hours=20)


def test_full_horizon_and_delayed_dependency_receipt_control_publication():
    bars, origins = fixture()
    bars.loc[20, 'received_at'] = pd.Timestamp('2026-01-02T14:00Z')
    pending = build_variance_targets(bars, origins, as_of='2026-01-02T13:00Z').iloc[0]
    assert not pending.observed and np.isnan(pending.target)
    assert pending.label_available_at == pd.Timestamp('2026-01-02T14:00Z')
    observed = build_variance_targets(bars, origins, as_of='2026-01-02T14:00Z').iloc[0]
    assert observed.observed


@pytest.mark.parametrize('missing', [0, 30, 144])
def test_missing_baseline_interior_or_final_bar_is_not_filled(missing):
    bars, origins = fixture()
    row = build_variance_targets(bars.drop(index=missing), origins, as_of='2026-01-02T14:00Z').iloc[0]
    assert not row.observed and np.isnan(row.target)
    assert row.label_status == 'gap_or_missing_baseline'


def test_future_changes_outside_dependency_do_not_change_label_and_input_order_is_irrelevant():
    bars, origins = fixture()
    original = build_variance_targets(bars, origins, as_of='2026-01-02T14:00Z')
    bars.loc[145:, 'close'] *= 100
    changed = build_variance_targets(bars.sample(frac=1, random_state=17), origins,
                                     as_of='2026-01-02T14:00Z')
    pd.testing.assert_frame_equal(original, changed)


def test_backdated_receipts_naive_times_duplicates_and_nonpositive_prices_rejected():
    bars, origins = fixture()
    for operation in ('receipt', 'duplicate', 'price', 'naive'):
        bad = bars.copy()
        if operation == 'receipt':
            bad.loc[0, 'received_at'] = bad.open_time.iloc[0]
        elif operation == 'duplicate':
            bad = pd.concat([bad, bad.iloc[[0]]], ignore_index=True)
        elif operation == 'price':
            bad.loc[0, 'close'] = 0.
        else:
            bad['open_time'] = bad.open_time.dt.tz_localize(None)
        with pytest.raises(ValueError):
            build_variance_targets(bad, origins, as_of='2026-01-02T14:00Z')


def test_qlike_paired_difference_is_scale_invariant_but_percentage_is_undefined():
    base = compare_variance_forecasts([1.], [2.], [1.])
    scaled = compare_variance_forecasts([.01], [.02], [.01])
    assert base['mean_qlike_difference'] == pytest.approx(scaled['mean_qlike_difference'])
    assert scaled['reference']['qlike'] < 0
    assert base['relative_qlike_improvement'] is None
    assert base['mean_qlike_difference'] > 0


def test_zero_variance_allowed_but_nonpositive_prediction_and_bad_rows_fail_whole_score():
    result = variance_losses([0., .1], [.01, .2])
    assert all(np.isfinite(values).all() for values in result.values())
    for prediction in ([0., .2], [-.01, .2], [np.nan, .2]):
        with pytest.raises(ValueError):
            variance_losses([0., .1], prediction)
    assert TargetProtocol(log_floor=1e-10).digest != TargetProtocol().digest
