"""Causal, raw-unit K0 features; deliberately independent of neural modules.

Rows are retained through warmup and data gaps. Consumers must explicitly select
``valid_rows``; missing support is never filled or shortened to fit a window.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from factorlab.protocol import fingerprint

RV_WINDOWS = (1, 6, 24, 48, 168, 720)
PARKINSON_WINDOWS = (24, 168)
HOUR = pd.Timedelta(hours=1)
FEATURE_NAMES = tuple(f'rv_{h}' for h in RV_WINDOWS) + tuple(
    f'parkinson_{h}' for h in PARKINSON_WINDOWS
) + ('hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos')


def _utc(series):
    values = pd.to_datetime(series, utc=True)
    if values.isna().any():
        raise ValueError('Feature timestamps cannot be missing')
    return values


def library_schema() -> dict:
    """A new transform/window requires a new library identity."""
    return {
        'kind': 'K0', 'version': 1, 'feature_names': list(FEATURE_NAMES),
        'rv': 'sum of squared close-to-close log returns; no annualization',
        'parkinson': 'sum(log(high/low)^2)/(4*log(2)); no annualization',
        'calendar': 'UTC forecast_window_start; hour and weekday sin/cos',
        'continuous_transform': 'identity_raw_units',
        'bypass_mask': [False] * 8 + [True] * 4,
        'history_hours': 720, 'extra_close_baseline_hours': 1,
        'gap_policy': 'restart; full window only', 'dtype': 'float64',
        'availability': 'every source row available no later than origin',
    }


@dataclass(frozen=True)
class FactorFrame:
    values: np.ndarray
    metadata: pd.DataFrame
    feature_names: tuple[str, ...] = FEATURE_NAMES
    bypass_mask: tuple[bool, ...] = (False,) * 8 + (True,) * 4
    schema_hash: str = ''

    def __post_init__(self):
        values = np.array(self.values, dtype=np.float64, copy=True)
        names = tuple(self.feature_names)
        bypass = tuple(self.bypass_mask)
        if values.ndim != 2 or values.shape != (len(self.metadata), len(names)):
            raise ValueError('FactorFrame must have aligned [sample, feature] rows')
        if len(set(names)) != len(names) or len(bypass) != len(names):
            raise ValueError('Feature names must be unique and bypass mask aligned')
        if any(type(v) is not bool for v in bypass) or any(not n for n in names):
            raise ValueError('Invalid feature schema')
        if np.isinf(values).any():
            raise ValueError('Infinite features are invalid; missing support uses NaN')
        values.setflags(write=False)
        object.__setattr__(self, 'values', values)
        object.__setattr__(self, 'metadata', self.metadata.reset_index(drop=True).copy())
        object.__setattr__(self, 'feature_names', names)
        object.__setattr__(self, 'bypass_mask', bypass)
        if not self.schema_hash:
            object.__setattr__(self, 'schema_hash', fingerprint({
                'feature_names': names, 'bypass_mask': bypass, 'dtype': 'float64'}))

    @property
    def valid_rows(self) -> np.ndarray:
        return np.isfinite(self.values).all(axis=1)

    def take(self, selection) -> FactorFrame:
        positions = np.arange(len(self.values))[selection]
        positions = np.atleast_1d(positions)
        return FactorFrame(self.values[positions], self.metadata.iloc[positions],
                           self.feature_names, self.bypass_mask, self.schema_hash)


def build_known_library(bars: pd.DataFrame, origins: pd.DataFrame | None = None, *, context_only=False) -> FactorFrame:
    """Build K0 at explicit origins, preserving their order and sample identity.

    ``bars`` needs symbol/open_time/available_at/close/high/low. With explicit
    origins, feature_open_time selects the last closed bar; calendar features
    use the supplied forecast_window_start (not an open_time offset). Source
    revisions must already be resolved by the caller's as-of snapshot.
    """
    required = {'symbol', 'open_time', 'available_at', 'close', 'high', 'low'}
    if not required <= set(bars):
        raise ValueError(f'Missing bar fields: {sorted(required - set(bars))}')
    frame = bars.copy()
    frame['open_time'] = _utc(frame.open_time)
    frame['available_at'] = _utc(frame.available_at)
    if frame.symbol.isna().any() or frame.duplicated(['symbol', 'open_time']).any():
        raise ValueError('Bar symbol/time identity must be unique and nonmissing')
    if not (frame.open_time == frame.open_time.dt.floor('h')).all():
        raise ValueError('K0 requires hourly UTC candle opens')
    if (frame.available_at < frame.open_time + HOUR).any():
        raise ValueError('An hourly candle cannot be available before it closes')
    for col in ('close', 'high', 'low'):
        frame[col] = pd.to_numeric(frame[col], errors='raise').astype(float)
    prices = frame[['close', 'high', 'low']].to_numpy()
    valid_price = (np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
                   & (frame.high >= frame.low).to_numpy()
                   & (frame.close <= frame.high).to_numpy()
                   & (frame.close >= frame.low).to_numpy())
    frame['_valid_price'] = valid_price
    frame = frame.sort_values(['symbol', 'open_time'], kind='stable').reset_index(drop=True)
    if origins is None:
        meta = frame[['symbol', 'open_time', 'available_at']].rename(
            columns={'open_time': 'feature_open_time'})
        meta['forecast_window_start'] = (
            _utc(frame.forecast_window_start) if 'forecast_window_start' in frame
            else meta.available_at.dt.ceil('h'))
        meta['sample_id'] = [f'{s}:{t.isoformat()}' for s, t in
                             zip(meta.symbol, meta.feature_open_time, strict=True)]
    else:
        needed = {'symbol', 'feature_open_time', 'available_at', 'forecast_window_start'}
        if not needed <= set(origins):
            raise ValueError(f'Missing origin fields: {sorted(needed - set(origins))}')
        meta = origins.copy().reset_index(drop=True)
    for col in ('feature_open_time', 'available_at', 'forecast_window_start'):
        meta[col] = _utc(meta[col])
    identity_columns = ['symbol', 'feature_open_time'] if context_only else ['symbol', 'available_at']
    if meta.symbol.isna().any() or meta.duplicated(identity_columns).any():
        raise ValueError('Origins require unique symbol/available_at identity')
    if not context_only and (meta.forecast_window_start < meta.available_at).any():
        raise ValueError('Forecast window cannot start before prediction availability')
    if not (meta.forecast_window_start == meta.forecast_window_start.dt.floor('h')).all():
        raise ValueError('Forecast windows must begin on a complete UTC hour')
    matrix = np.full((len(frame), len(FEATURE_NAMES)), np.nan, dtype=np.float64)
    ready = np.full(len(frame), pd.NaT, dtype=object)
    for _, group in frame.groupby('symbol', sort=False):
        # Bad candles split support as well as explicit missing hours.
        split = group.open_time.diff().ne(HOUR) | ~group._valid_price
        split |= ~group._valid_price.shift(fill_value=False)
        for _, segment in group.groupby(split.cumsum(), sort=False):
            if not segment._valid_price.all():
                continue
            idx = segment.index.to_numpy()
            returns = np.log(segment.close).diff()
            for j, h in enumerate(RV_WINDOWS):
                matrix[idx, j] = returns.pow(2).rolling(h, min_periods=h).sum()
            ranges = np.log(segment.high / segment.low).pow(2) / (4 * np.log(2))
            for j, h in enumerate(PARKINSON_WINDOWS, start=len(RV_WINDOWS)):
                matrix[idx, j] = ranges.rolling(h, min_periods=h).sum()
            # Full K0 requires 720 returns plus the previous close. Datetimes are
            # compared as timestamps; their internal ns/us/ms unit is irrelevant.
            queue = deque()
            for pos, timestamp in enumerate(segment.available_at):
                while queue and queue[0][0] < pos - 720:
                    queue.popleft()
                while queue and queue[-1][1] <= timestamp:
                    queue.pop()
                queue.append((pos, timestamp))
                if pos >= 720:
                    ready[idx[pos]] = queue[0][1]
    lookup = {(row.symbol, row.open_time): i for i, row in frame.iterrows()}
    values = np.full((len(meta), len(FEATURE_NAMES)), np.nan, dtype=np.float64)
    for i, row in meta.iterrows():
        source = lookup.get((row.symbol, row.feature_open_time))
        if source is None:
            continue
        # If the full dependency is late, the row is invalid rather than leaking
        # a value from a bar the origin could not yet know.
        if pd.notna(ready[source]) and ready[source] <= row.available_at:
            values[i, :8] = matrix[source, :8]
        hour = row.forecast_window_start.hour
        weekday = row.forecast_window_start.dayofweek
        values[i, 8:] = (np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                         np.sin(2 * np.pi * weekday / 7), np.cos(2 * np.pi * weekday / 7))
    return FactorFrame(values, meta, FEATURE_NAMES, (False,) * 8 + (True,) * 4,
                       fingerprint(library_schema()))


def build_feature_history(bars):
    """Causal feature contexts; equal receipt times do not imply forecasts.

    Historical context calendar coordinates are the nominal next full hour.
    Only a selected real origin uses its actual future-window calendar. Every
    numeric dependency still must have arrived by the row's recorded receipt.
    This never constructs labels or permission to backfill a prediction.
    """
    meta = bars[['symbol', 'open_time', 'available_at']].rename(columns={'open_time': 'feature_open_time'}).copy()
    meta['forecast_window_start'] = meta.feature_open_time + 2 * HOUR
    if 'prediction_origin' in bars:
        live = bars.prediction_origin.to_numpy(dtype=bool)
        meta.loc[live, 'forecast_window_start'] = pd.DatetimeIndex(meta.loc[live, 'available_at']).ceil('h')
    meta['sample_id'] = [f'context:{s}:{t.isoformat()}' for s, t in zip(meta.symbol, meta.feature_open_time, strict=True)]
    meta['context_only'] = True
    return build_known_library(bars, meta, context_only=True)
