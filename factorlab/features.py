"""Trailing causal features with explicit missing masks and per-gap restarts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import HOUR, validate_klines

FEATURE_COLUMNS = ('return_1h', 'return_6h', 'return_12h', 'return_24h', 'return_48h',
                   'volatility_24h', 'volatility_48h', 'range_1h', 'body_1h',
                   'volume_log_1h', 'volume_z_24h', 'price_z_24h')
STATE_COLUMNS = ('state_trend_24h', 'state_volatility_24h', 'state_volume_24h')


def _segment(group: pd.DataFrame) -> pd.DataFrame:
    result = group.copy()
    close = group['close']
    returns = close.pct_change(fill_method=None)
    for hours in (1, 6, 12, 24, 48):
        result[f'return_{hours}h'] = close.pct_change(periods=hours, fill_method=None)
    for hours in (24, 48):
        result[f'volatility_{hours}h'] = returns.rolling(hours, min_periods=hours).std(ddof=0)
    result['range_1h'] = (group['high'] - group['low']) / close
    result['body_1h'] = (close - group['open']) / group['open']
    log_volume = np.log1p(group['volume'])
    result['volume_log_1h'] = log_volume
    for name, values in (('volume', log_volume), ('price', np.log(close))):
        mean = values.rolling(24, min_periods=24).mean()
        std = values.rolling(24, min_periods=24).std(ddof=0)
        result[f'{name}_z_24h'] = (values - mean) / std.where(std > 1e-12)
    result['state_trend_24h'] = result['return_24h']
    result['state_volatility_24h'] = result['volatility_24h']
    result['state_volume_24h'] = result['volume_z_24h']
    return result


def build_features(canonical: pd.DataFrame) -> pd.DataFrame:
    """Every feature uses only candles available by that row's available_at.

    All rolling windows restart after a missing hour. Missing/warm-up/undefined
    values remain NaN, with True masks indicating observed finite features.
    """
    validate_klines(canonical)
    # Unique positional index avoids duplicate user indexes corrupting group writes.
    source = canonical.reset_index(drop=True).copy()
    parts = []
    for _, group in source.groupby('symbol', sort=False):
        segments = group['open_time'].diff().ne(HOUR).cumsum()
        for _, segment in group.groupby(segments, sort=False):
            parts.append(_segment(segment))
    result = pd.concat(parts).sort_index()
    for name in (*FEATURE_COLUMNS, *STATE_COLUMNS):
        finite = np.isfinite(result[name])
        result[name] = result[name].where(finite, np.nan)
        result[f'mask_{name}'] = finite.astype(bool)
    result.index = canonical.index
    result.attrs = canonical.attrs.copy()
    result.attrs['feature_policy'] = 'trailing_current_observable_bar_reset_at_hour_gaps'
    return result
