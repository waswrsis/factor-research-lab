import numpy as np
import pandas as pd
import pytest

from factorlab.discovery.known_library import (
    FEATURE_NAMES,
    RV_WINDOWS,
    build_known_library,
    library_schema,
)


def candles(n=800):
    time = pd.date_range('2024-01-01', periods=n, freq='h', tz='UTC')
    close = 100 * np.exp(np.arange(n) * .001)
    return pd.DataFrame({'symbol': 'BTCUSDT', 'open_time': time,
                         'available_at': time + pd.Timedelta(hours=1, seconds=5),
                         'close': close, 'high': close * np.exp(.01),
                         'low': close * np.exp(-.01)})


def test_raw_window_units_warmup_and_calendar():
    bars = candles()
    frame = build_known_library(bars)
    assert frame.values.shape == (800, 12)
    assert frame.feature_names == FEATURE_NAMES
    assert not frame.valid_rows[:720].any()
    assert frame.valid_rows[720:].all()
    np.testing.assert_allclose(frame.values[720, :6], np.array(RV_WINDOWS) * 1e-6,
                               rtol=1e-10)
    np.testing.assert_allclose(frame.values[720, 6:8],
                               np.array([24, 168]) * .02**2 / (4 * np.log(2)))
    assert frame.bypass_mask == (False,) * 8 + (True,) * 4
    subset = frame.take(frame.valid_rows)
    assert len(subset.values) == 80 and subset.schema_hash == frame.schema_hash
    assert library_schema()['continuous_transform'] == 'identity_raw_units'


def test_calendar_uses_supplied_forecast_window_not_open_offset():
    bars = candles()
    meta = pd.DataFrame({'symbol': ['BTCUSDT'], 'feature_open_time': [bars.open_time[730]],
                         'available_at': [bars.available_at[730]],
                         'forecast_window_start': [pd.Timestamp('2024-02-02T17:00Z')]})
    row = build_known_library(bars, meta).values[0]
    np.testing.assert_allclose(row[8:10], [np.sin(2*np.pi*17/24), np.cos(2*np.pi*17/24)])
    np.testing.assert_allclose(row[10:], [np.sin(2*np.pi*4/7), np.cos(2*np.pi*4/7)])


def test_gap_requires_fresh_720_returns_and_bad_prices_restart():
    bars = candles(1500)
    broken = bars.drop(index=740)
    frame = build_known_library(broken)
    mask = (frame.metadata.feature_open_time >= bars.open_time[741]) & (
        frame.metadata.feature_open_time < bars.open_time[1461])
    assert not frame.valid_rows[mask].any()
    assert frame.valid_rows[-1]
    bars.loc[740, 'close'] = np.nan
    bad = build_known_library(bars)
    assert not bad.valid_rows[740:1461].any()
    assert bad.valid_rows[1461]


@pytest.mark.parametrize('unit', ['ms', 'us', 'ns'])
def test_datetime_units_and_future_perturbations(unit):
    bars = candles()
    before = build_known_library(bars)
    bars['open_time'] = bars.open_time.dt.as_unit(unit)
    bars['available_at'] = bars.available_at.dt.as_unit(unit)
    bars.loc[780:, ['close', 'high', 'low']] *= 1.5
    after = build_known_library(bars)
    np.testing.assert_equal(before.values[:780], after.values[:780])


def test_source_arrival_and_identity_guards():
    bars = candles()
    bars.loc[0, 'available_at'] = bars.available_at[750] + pd.Timedelta(seconds=1)
    frame = build_known_library(bars)
    assert not frame.valid_rows[720]
    assert frame.valid_rows[721]  # Baseline row 0 no longer needed.
    with pytest.raises(ValueError, match='unique'):
        build_known_library(pd.concat([bars, bars.iloc[:1]]))
    bars.loc[1, 'available_at'] = bars.open_time[1]
    with pytest.raises(ValueError, match='before it closes'):
        build_known_library(bars)


def test_two_symbols_do_not_share_return_baselines():
    a = candles(725)
    b = a.copy()
    b['symbol'] = 'ETHUSDT'
    b[['close', 'high', 'low']] *= 7
    frame = build_known_library(pd.concat([a, b]))
    assert frame.valid_rows.sum() == 10
    np.testing.assert_allclose(frame.values[720:725], frame.values[1445:1450])
