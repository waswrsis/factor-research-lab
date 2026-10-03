"""Independent semantic checks for root-authored sample/execution-time logic."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from factorlab.data import parse_klines_csv
from factorlab.dataset import make_dataset
from factorlab.protocol import Protocol
from factorlab.splits import chronological_oof, outer_folds, segment_masks


def bars(hours=350, skip=(), symbol='BTCUSDT'):
    rows = []
    for i in range(hours):
        if i in skip:
            continue
        t = 1735689600000 + i * 3600000
        p = 100 * np.exp(i * .0002 + np.sin(i) * .01)
        rows.append(f'{t},{p},{p+3},{p-3},{p+1},{i%17+1},{t+3599999},0,1,0,0,0')
    return parse_klines_csv(('\n'.join(rows) + '\n').encode(), symbol)


def test_delayed_bar_entry_12h_label_and_maturity():
    source = bars()
    dataset = make_dataset(source, Protocol())
    row = dataset.meta.iloc[0]
    assert row['available_at'] == row['feature_open_time'] + pd.Timedelta(hours=1, seconds=5)
    assert row['entry_time'] == row['feature_open_time'] + pd.Timedelta(hours=2)
    assert row['entry_time'] > row['available_at']
    assert row['label_end'] == row['entry_time'] + pd.Timedelta(hours=12)
    assert row['label_available_at'] == row['label_end'] + pd.Timedelta(seconds=5)
    prices = source.set_index('open_time')['open']
    assert dataset.y[0] == pytest.approx(prices[row['label_end']] / prices[row['entry_time']] - 1)
    assert dataset.x.shape[1] == Protocol().lookback
    assert np.isfinite(dataset.x).all()


def test_sample_feature_prefix_invariance_and_future_target_separation():
    source = bars()
    full = make_dataset(source, Protocol())
    prefix_source = source.iloc[:200]
    prefix = make_dataset(prefix_source, Protocol())
    selected = full.meta['label_end'] <= prefix_source['open_time'].iloc[-1]
    np.testing.assert_array_equal(prefix.x, full.x[selected])
    np.testing.assert_array_equal(prefix.y, full.y[selected])
    pd.testing.assert_frame_equal(prefix.meta, full.meta.loc[selected].reset_index(drop=True))
    changed = source.copy()
    changed.loc[200:, ['open', 'high', 'low', 'close']] *= 10
    altered = make_dataset(changed, Protocol())
    past = full.meta['feature_open_time'] < source['open_time'].iloc[200]
    np.testing.assert_array_equal(full.x[past], altered.x[past])
    # A future label is permitted to change; its corresponding past feature cannot.
    across = (full.meta['entry_time'] < source['open_time'].iloc[200]) & (
        full.meta['label_end'] >= source['open_time'].iloc[200])
    assert across.any()
    assert not np.allclose(full.y[across], altered.y[across])


def test_missing_hour_invalidates_outcome_and_restarts_feature_history():
    source = bars(skip=(160,))
    dataset = make_dataset(source, Protocol())
    gap = pd.Timestamp('2025-01-01T00:00Z') + pd.Timedelta(hours=160)
    assert not ((dataset.meta.entry_time <= gap) & (dataset.meta.label_end >= gap)).any()
    after = dataset.meta[dataset.meta.feature_open_time > gap]
    # A complete 48h-return feature followed by the 48-row model window is required.
    assert after.feature_open_time.min() >= gap + pd.Timedelta(hours=96)


def test_empty_history_rejected():
    with pytest.raises(ValueError, match='No complete'):
        make_dataset(bars(30), Protocol())


def test_oof_training_labels_never_reach_validation_start():
    dataset = make_dataset(bars(1000), Protocol())
    cutoff = dataset.meta.label_available_at.max() + pd.Timedelta(seconds=1)
    seen = np.zeros(len(dataset.meta), dtype=int)
    for train, valid in chronological_oof(dataset.meta, cutoff, min_train_hours=168):
        assert train.any() and valid.any()
        assert dataset.meta.label_available_at[train].max() < dataset.meta.available_at[valid].min()
        assert not (train & valid).any()
        seen += valid.astype(int)
    assert seen.max() == 1


def test_all_segment_boundary_maturity_is_strict():
    fold = outer_folds(Protocol())[0]
    boundaries = [fold.validation_start, fold.confirmation_start, fold.test_start, fold.test_end]
    for segment, boundary in zip(('train', 'search', 'confirmation', 'test'), boundaries, strict=True):
        decision = boundary - pd.Timedelta(hours=1)
        meta = pd.DataFrame({'available_at': [decision, decision],
                             'label_available_at': [boundary - pd.Timedelta(seconds=1), boundary]})
        assert segment_masks(meta, fold)[segment].tolist() == [True, False]


def test_protocol_rejects_invalid_economic_and_resource_parameters():
    for change in ({'fee_bps': -5}, {'slippage_bps': -1}, {'risk_aversion': float('nan')},
                   {'max_parameters': -1}, {'bootstrap_samples': 0}, {'volatility_floor': 0}):
        with pytest.raises(ValueError):
            replace(Protocol(), **change)


def test_dataset_obeys_frozen_symbols_and_calendar():
    with pytest.raises(ValueError, match='symbol|universe'):
        make_dataset(bars(symbol='XRPUSDT'), Protocol())
    protocol = replace(Protocol(), start='2025-01-05')
    dataset = make_dataset(bars(), protocol)
    assert dataset.meta.available_at.min() >= pd.Timestamp(protocol.start, tz='UTC')


def test_declared_availability_assumption_must_match_protocol():
    source = bars()
    source['available_at'] += pd.Timedelta(seconds=5)
    with pytest.raises(ValueError, match='avail|delay'):
        make_dataset(source, Protocol())
