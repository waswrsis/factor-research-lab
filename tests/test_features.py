import numpy as np
import pandas as pd
import pytest

from factorlab.data import parse_klines_csv
from factorlab.features import FEATURE_COLUMNS, STATE_COLUMNS, build_features


def canonical(hours=140, skip=(), symbol='BTCUSDT'):
    rows = []
    for i in range(hours):
        if i in skip:
            continue
        t = 1735689600000 + i * 3600000
        p = 100 * np.exp(i * .002 + np.sin(i) * .01)
        rows.append(f'{t},{p},{p+3},{p-3},{p+1},{i%17+1},{t+3599999},0,1,0,0,0')
    return parse_klines_csv(('\n'.join(rows) + '\n').encode(), symbol)


def test_future_prefix_invariance():
    source = canonical()
    full = build_features(source)
    for stop in (1, 24, 25, 49, 96):
        prefix = build_features(source.iloc[:stop])
        pd.testing.assert_frame_equal(prefix, full.iloc[:stop])
    changed = source.copy()
    changed.loc[100:, ['open', 'high', 'low', 'close']] *= 100
    changed.loc[100:, 'volume'] *= 100
    pd.testing.assert_frame_equal(build_features(changed).iloc[:100], full.iloc[:100])


def test_hand_calculation_and_warmup_masks():
    source = canonical(55)
    frame = build_features(source)
    assert frame['return_12h'].iloc[12] == pytest.approx(source['close'].iloc[12] / source['close'].iloc[0] - 1)
    observed_returns = source['close'].iloc[:25].pct_change(fill_method=None).iloc[1:]
    assert frame['volatility_24h'].iloc[24] == pytest.approx(observed_returns.std(ddof=0))
    assert frame['return_48h'].iloc[:48].isna().all()
    assert not frame['mask_return_48h'].iloc[:48].any()
    assert frame['mask_return_48h'].iloc[48]
    for column in (*FEATURE_COLUMNS, *STATE_COLUMNS):
        assert frame[f'mask_{column}'].equals(np.isfinite(frame[column]).rename(f'mask_{column}'))


def test_gap_restarts_every_lookback():
    source = canonical(120, skip=(60,))
    frame = build_features(source)
    after_gap = frame[frame['open_time'] > pd.Timestamp('2025-01-03T12:00Z')]
    assert np.isnan(after_gap['return_1h'].iloc[0])
    assert after_gap['return_24h'].iloc[:24].isna().all()
    assert after_gap['volatility_24h'].iloc[:24].isna().all()
    assert after_gap['mask_range_1h'].iloc[0]
    isolated = build_features(source.loc[after_gap.index])
    pd.testing.assert_frame_equal(after_gap, isolated)


def test_symbol_isolation_interleaved_and_index_preserved():
    btc = canonical(70)
    eth = canonical(70, symbol='ETHUSDT')
    eth[['open', 'high', 'low', 'close']] *= 2
    source = pd.concat([btc, eth]).sort_values('open_time', kind='stable')
    source.index = np.zeros(len(source), dtype=int)
    frame = build_features(source)
    assert frame.index.equals(source.index)
    btc_output = frame[frame.symbol == 'BTCUSDT'].reset_index(drop=True)
    pd.testing.assert_frame_equal(btc_output, build_features(btc))


def test_zero_variance_is_missing_and_zero_volume_valid():
    source = canonical(55)
    source['volume'] = 0.
    source[['open', 'close']] = 100.
    source['high'] = 101.
    source['low'] = 99.
    frame = build_features(source)
    assert (frame['volume_log_1h'] == 0).all()
    assert frame['price_z_24h'].isna().all()
    assert frame['volume_z_24h'].isna().all()
    assert not frame['mask_volume_z_24h'].any()
    assert (frame['volatility_24h'].iloc[24:] == 0).all()
