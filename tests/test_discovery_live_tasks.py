"""Actual REST receipt time, backlog origins and delayed label maturity."""
import numpy as np
import pandas as pd
import pytest
from test_discovery_tasks import bars, index_at, spec

from factorlab.discovery.evolution import _visible_hash
from factorlab.discovery.tasks import build_task_dataset


def observed_feed(*, second_batch=False, second_end=80, second_seconds=2700):
    """Sixty historical seed rows, then observed batches with one newest origin."""
    source = bars(hours=second_end + 1 if second_batch else 71)
    source['received_at'] = pd.Series(pd.NaT, index=source.index, dtype='datetime64[ns, UTC]')
    source['availability_basis'] = 'historical_assumed_seed'
    source['prediction_origin'] = True
    batches = [(60, 70, 1800)]
    if second_batch:
        batches.append((71, second_end, second_seconds))
    for first, last, seconds in batches:
        receipt = source.close_time.iloc[last] + pd.Timedelta(seconds=seconds)
        source.loc[first:last, 'available_at'] = receipt
        source.loc[first:last, 'received_at'] = receipt
        source.loc[first:last, 'availability_basis'] = 'actual_rest_receipt'
        source.loc[first:last, 'prediction_origin'] = False
        source.loc[last, 'prediction_origin'] = True
    source.attrs['availability_mode'] = 'observed_rest_feed'
    source.attrs['feed_identity'] = {'format': 'synthetic-observed-feed-v1',
                                     'seed_rows': 60, 'symbol': 'BTCUSDT'}
    return source


def test_backlog_rows_only_supply_features_never_backfilled_origins():
    source = observed_feed()
    result = build_task_dataset(source, spec())
    receipt = source.received_at.iloc[70]
    i = index_at(result, 70)
    assert result.meta.available_at.iloc[i] == receipt
    assert result.meta.forecast_window_start.iloc[i] == receipt.ceil('h')
    assert result.meta.attrs['availability_mode'] == 'observed_rest_feed'
    assert result.meta.attrs['feed_identity'] == source.attrs['feed_identity']
    assert not result.meta.feature_open_time.isin(source.open_time.iloc[60:70]).any()
    assert len(result.meta[result.meta.available_at == receipt]) == 1
    assert result.meta.window_start_open_time.iloc[i] == source.open_time.iloc[67]
    assert np.isfinite(result.x).all()
    for task in result.tasks:
        if task.predictive:
            assert np.isnat(result.maturity[task.name][i])
            assert not result.observed[task.name][i]
        else:
            assert result.observed[task.name][i]


def test_future_append_preserves_existing_inputs_and_waits_for_actual_receipt():
    prefix, full = observed_feed(), observed_feed(second_batch=True)
    past = build_task_dataset(prefix, spec())
    future = build_task_dataset(full, spec())
    positions = future.meta.set_index('sample_id').index.get_indexer(past.meta.sample_id)
    np.testing.assert_array_equal(past.x, future.x[positions])
    old = index_at(past, 70)
    new = index_at(future, 70)
    actual_receipt = full.received_at.iloc[80]
    for task in future.tasks:
        if task.predictive:
            planned = future.meta.forecast_window_start.iloc[new] + pd.Timedelta(
                hours=task.horizon_hours, seconds=spec().availability_delay_seconds)
            assert np.isnat(past.maturity[task.name][old])
            assert pd.Timestamp(future.maturity[task.name][new], tz='UTC') == max(planned, actual_receipt)
            assert not future.task_mask(task.name, actual_receipt - pd.Timedelta(nanoseconds=1))[new]
            assert future.task_mask(task.name, actual_receipt)[new]
    assert future.meta.available_at.iloc[new] == prefix.received_at.iloc[70]
    changed = full.copy()
    changed.loc[71:, ['open', 'high', 'low', 'close']] *= 1.2
    changed.loc[71:, 'volume'] *= 2
    altered = build_task_dataset(changed, spec())
    np.testing.assert_array_equal(past.x, altered.x[positions])


def test_fast_receipt_does_not_bypass_frozen_planned_maturity():
    source = observed_feed(second_batch=True, second_end=75, second_seconds=1)
    result = build_task_dataset(source, spec())
    i = index_at(result, 70)
    receipt = source.received_at.iloc[75]
    for task in result.tasks:
        if task.predictive:
            planned = result.meta.forecast_window_start.iloc[i] + pd.Timedelta(
                hours=task.horizon_hours, seconds=spec().availability_delay_seconds)
            assert receipt < planned
            assert pd.Timestamp(result.maturity[task.name][i], tz='UTC') == planned
            assert not result.task_mask(task.name, receipt)[i]


def test_unknown_future_maturity_is_hash_stable_until_observations_arrive():
    prefix, full = observed_feed(), observed_feed(second_batch=True)
    past = build_task_dataset(prefix, spec())
    future = build_task_dataset(full, spec())
    cutoff = prefix.received_at.iloc[70]
    assert _visible_hash(past, cutoff) == _visible_hash(future, cutoff)
    assert _visible_hash(past, cutoff) != _visible_hash(future, full.received_at.iloc[80])


def test_duplicate_symbol_and_receipt_prediction_origins_are_rejected():
    source = observed_feed()
    source.loc[69, 'prediction_origin'] = True
    with pytest.raises(ValueError, match='duplicate prediction origins'):
        build_task_dataset(source, spec())


def test_origin_must_be_latest_closed_bar_in_observed_batch():
    source = observed_feed()
    source.loc[70, 'prediction_origin'] = False
    source.loc[60, 'prediction_origin'] = True
    with pytest.raises(ValueError, match='latest|newest'):
        build_task_dataset(source, spec())


@pytest.mark.parametrize('change', ['seed_delay', 'basis', 'missing_receipt',
                                   'before_close', 'after_availability', 'origin_dtype', 'receipt_naive'])
def test_invalid_live_provenance_is_rejected(change):
    source = observed_feed()
    if change == 'seed_delay':
        source.loc[0, 'available_at'] += pd.Timedelta(seconds=1)
    elif change == 'basis':
        source.loc[60, 'availability_basis'] = 'fabricated_historical_publish_time'
    elif change == 'missing_receipt':
        source.loc[60, 'received_at'] = pd.NaT
    elif change == 'before_close':
        source.loc[60, 'received_at'] = source.close_time.iloc[60] - pd.Timedelta(seconds=1)
    elif change == 'after_availability':
        source.loc[60, 'received_at'] = source.available_at.iloc[60] + pd.Timedelta(seconds=1)
    elif change == 'origin_dtype':
        source['prediction_origin'] = source.prediction_origin.astype(int)
    else:
        source['received_at'] = source.received_at.dt.tz_localize(None)
    with pytest.raises(ValueError):
        build_task_dataset(source, spec())


def test_actual_receipts_require_explicit_observed_feed_mode():
    source = observed_feed()
    source.attrs.pop('availability_mode')
    with pytest.raises(ValueError, match='availability'):
        build_task_dataset(source, spec())
