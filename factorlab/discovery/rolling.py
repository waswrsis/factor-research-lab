"""V7 two-dimensional, cold calendar-month research datasets.

Archive replay has assumed historical availability. Only an observed receipt
feed can supply forward evidence; downloading old data today cannot create it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from factorlab.protocol import fingerprint
from factorlab.time_utils import HOUR, calendar_window, utc_index

from .frame_identity import frame_identity
from .rolling_cadence import elapsed_window
from .known_library import FactorFrame, build_known_library, library_schema


@dataclass(frozen=True)
class RollingSpec:
    months: int = 3
    horizon_hours: int = 12
    availability_delay_seconds: int = 1
    task: str = 'volatility'
    event_bins: tuple[int, ...] = (4, 12, 20)
    barrier_multiplier: float = 1.0
    ridge: float = 1e-3
    minimum_train_rows: int = 240
    window_days: int | None = None

    def __post_init__(self):
        if self.window_days is not None and (type(self.window_days) is not int or self.window_days != 90):
            raise ValueError('Elapsed rolling version requires exactly 90 days')
        if self.months != 3:
            raise ValueError('V7 freezes three calendar months; change protocol for another window')
        if type(self.horizon_hours) is not int or not 1 <= self.horizon_hours <= 168:
            raise ValueError('Invalid predictive horizon')
        if self.task not in {'volatility', 'first_event'}:
            raise ValueError('V7 isolates volatility and first_event tasks')
        if (type(self.availability_delay_seconds) is not int
                or not 0 <= self.availability_delay_seconds <= 3600):
            raise ValueError('Invalid assumed availability delay')
        if (type(self.minimum_train_rows) is not int or self.minimum_train_rows < 2
                or not np.isfinite(self.ridge) or self.ridge <= 0):
            raise ValueError('Invalid fitting bounds')
        if (not np.isfinite(self.barrier_multiplier) or self.barrier_multiplier <= 0
                or (self.task == 'first_event' and (
                    not self.event_bins or tuple(sorted(set(self.event_bins))) != self.event_bins
                    or self.event_bins[-1] != self.horizon_hours
                    or any(type(v) is not int or v < 1 for v in self.event_bins)))):
            raise ValueError('Invalid first-event protocol')

    def window(self, cutoff):
        return calendar_window(cutoff, self.months) if self.window_days is None else elapsed_window(cutoff, self.window_days)

    @property
    def digest(self):
        fields = asdict(self)
        if self.window_days is None:
            fields.pop('window_days')
        version = 'v7-rolling-v1' if self.window_days is None else 'v7-rolling-elapsed90-v2'
        return fingerprint({'version': version, **fields,
                            'library': library_schema(), 'fit': 'cold',
                            'label': 'full future UTC candles, receipt-aware maturity'})


@dataclass(frozen=True)
class RollingData:
    frame: FactorFrame
    target: np.ndarray
    maturity: pd.DatetimeIndex
    observed: np.ndarray
    spec: RollingSpec
    availability_mode: str

    def __post_init__(self):
        n = len(self.frame.values)
        target = np.array(self.target, dtype=float, copy=True)
        observed = np.asarray(self.observed, dtype=bool).copy()
        maturity = pd.DatetimeIndex(self.maturity)
        if target.shape != (n,) or observed.shape != (n,) or len(maturity) != n:
            raise ValueError('Unaligned rolling labels')
        if maturity.tz is None or not np.isfinite(target[observed]).all():
            raise ValueError('Observed labels require finite values and UTC maturity')
        if (maturity[observed].hasnans
                or (maturity[observed] <= utc_index(self.frame.metadata.available_at)[observed]).any()):
            raise ValueError('Predictive labels must mature after their origin')
        target.setflags(write=False)
        observed.setflags(write=False)
        object.__setattr__(self, 'target', target)
        object.__setattr__(self, 'observed', observed)
        object.__setattr__(self, 'maturity', maturity.tz_convert('UTC'))

    def train_mask(self, cutoff):
        start, end = self.spec.window(cutoff)
        origins = utc_index(self.frame.metadata.available_at)
        return np.asarray((origins >= start) & (origins < end)
                          & (self.maturity <= end) & self.observed & self.frame.valid_rows)

    def score_mask(self, start, end, as_of):
        start, end, as_of = (pd.Timestamp(v) for v in (start, end, as_of))
        if any(v.tzinfo is None for v in (start, end, as_of)) or not start < end <= as_of:
            raise ValueError('Invalid causal scoring interval')
        origins = utc_index(self.frame.metadata.available_at)
        return np.asarray((origins >= start) & (origins < end)
                          & (self.maturity <= as_of) & self.observed & self.frame.valid_rows)

    def training_identity(self, cutoff):
        mask = self.train_mask(cutoff)
        if int(mask.sum()) < self.spec.minimum_train_rows:
            raise ValueError('Insufficient mature, fully warmed training support')
        start, end = self.spec.window(cutoff)
        return {'rolling_hash': self.spec.digest, 'train_start': start.isoformat(),
                'cutoff': end.isoformat(), 'availability_mode': self.availability_mode,
                'rows': int(mask.sum()), 'fit': 'cold',
                'visible_data_hash': fingerprint({
                    'rule': 'typed-frame-columns-v1',
                    'meta': frame_identity(self.frame.metadata.loc[mask]),
                    'X': self.frame.values[mask].tolist(), 'y': self.target[mask].tolist(),
                    'maturity': frame_identity(pd.DataFrame({'maturity': self.maturity[mask]}))})}


def build_rolling_data(bars: pd.DataFrame, spec=None) -> RollingData:
    """Build features/labels without [sample,time,feature] allocation.

    Callers retain raw warmup candles outside the three-month fit window. Full
    720-hour RV needs a preceding close. Source revisions must be resolved as of
    the data snapshot, before calling this function.
    """
    spec = RollingSpec() if spec is None else spec
    source = bars.copy()
    for column in ('open_time', 'available_at'):
        if column not in source:
            raise ValueError(f'Missing {column}')
        source[column] = utc_index(source[column])
    if source.duplicated(['symbol', 'open_time']).any():
        raise ValueError('Resolve duplicate/revised bars before creating a snapshot')
    source = source.sort_values(['symbol', 'open_time'], kind='stable').reset_index(drop=True)
    numeric = ('open', 'high', 'low', 'close') if spec.task == 'first_event' else ('close', 'high', 'low')
    prices = source.loc[:, numeric].to_numpy(dtype=float)
    if (not np.isfinite(prices).all() or (prices <= 0).any()
            or (source.high < source.low).any() or (source.high < source.close).any()
            or (source.low > source.close).any()):
        raise ValueError('Invalid OHLC source data')
    if spec.task == 'first_event' and ((source.high < source.open).any() or (source.low > source.open).any()):
        raise ValueError('Invalid open within OHLC range')
    mode = bars.attrs.get('availability_mode', 'assumed_historical_delay')
    if mode not in {'assumed_historical_delay', 'observed_rest_feed', 'mixed_seed_and_observed_feed',
                    'quarantined_finite_observation'}:
        raise ValueError('Unknown availability provenance')
    if mode == 'quarantined_finite_observation':
        from .quarantined_observation import verify_frame
        verify_frame(bars)
    if mode in {'observed_rest_feed', 'mixed_seed_and_observed_feed'}:
        if not {'received_at', 'prediction_origin'} <= set(source):
            raise ValueError('Observed feed requires receipts and selected prediction origins')
        live = (source.availability_basis == 'actual_receipt' if mode == 'mixed_seed_and_observed_feed'
                else pd.Series(True, index=source.index))
        if mode == 'mixed_seed_and_observed_feed':
            if not set(source.availability_basis) <= {'actual_receipt', 'assumed_archive'}:
                raise ValueError('Mixed feed needs explicit seed/receipt provenance')
            seed = source.loc[~live]
            expected = seed.open_time + HOUR + pd.Timedelta(seconds=spec.availability_delay_seconds)
            if seed.received_at.notna().any() or not (seed.available_at == expected).all():
                raise ValueError('Imported seed must retain assumed availability, not real past receipt')
        received = utc_index(source.loc[live, 'received_at'])
        if ((received > utc_index(source.loc[live, 'available_at'])).any()
                or (received < utc_index(source.loc[live, 'open_time']) + HOUR).any()
                or source.prediction_origin.dtype != bool):
            raise ValueError('Invalid observed-feed receipts/origins')
        for _, batch in source.loc[live].groupby(['symbol', 'available_at']):
            origins = batch[batch.prediction_origin]
            if len(origins) > 1 or (len(origins) and origins.open_time.iloc[0] != batch.open_time.max()):
                raise ValueError('Only latest closed candle of a receipt batch is an origin')
        for _, group in source.loc[live].groupby('symbol', sort=False):
            by_receipt = group.sort_values(['available_at', 'open_time'], kind='stable')
            latest = by_receipt.open_time.cummax()
            stale = by_receipt.prediction_origin & (by_receipt.open_time < latest)
            if stale.any():
                raise ValueError('A delayed old candle cannot become a new latest prediction origin')
    else:
        expected = source.open_time + HOUR + pd.Timedelta(seconds=spec.availability_delay_seconds)
        if not (source.available_at == expected).all():
            raise ValueError('Archive availability differs from frozen close-plus-delay rule')
    if spec.task == 'first_event' and not {'open', 'high', 'low'} <= set(source):
        raise ValueError('Event labels require OHLC')
    if mode == 'mixed_seed_and_observed_feed':
        origins = source.loc[source.prediction_origin | ~live]
    else:
        origins = source.loc[source.prediction_origin] if mode == 'observed_rest_feed' else source
    origins = origins[['symbol', 'open_time', 'available_at']].rename(
        columns={'open_time': 'feature_open_time'}).copy()
    origins['forecast_window_start'] = origins.available_at.dt.ceil('h')
    origins['sample_id'] = [f'{s}:{t.isoformat()}' for s, t in
                            zip(origins.symbol, origins.feature_open_time, strict=True)]
    origins = origins.sort_values(['available_at', 'symbol'], kind='stable').reset_index(drop=True)
    frame = build_known_library(source, origins)
    labels = {}
    delay = pd.Timedelta(seconds=spec.availability_delay_seconds)
    h = spec.horizon_hours
    for symbol, group in source.groupby('symbol', sort=False):
        for _, segment in group.groupby(group.open_time.diff().ne(HOUR).cumsum(), sort=False):
            times = pd.DatetimeIndex(segment.open_time)
            avail = pd.DatetimeIndex(segment.available_at)
            close = np.log(segment.close.to_numpy(dtype=float))
            increments = np.diff(close)
            for i, stamp in enumerate(times):
                start = avail[i].ceil('h')
                j = int(times.searchsorted(start))
                ready = start + h * HOUR + delay
                value, valid, status = np.nan, False, 'incomplete_horizon'
                if j > 0 and j + h <= len(times) and times[j] == start:
                    ready = max(ready, avail[j - 1:j + h].max())
                    if spec.task == 'volatility':
                        value = float(np.sqrt(np.square(increments[j - 1:j + h - 1]).sum()))
                        valid, status = True, 'observed'
                    elif i >= 24:
                        width = spec.barrier_multiplier * float(increments[i - 24:i].std()) * np.sqrt(h)
                        if width > 1e-12 and np.isfinite(width):
                            upper, lower = close[i] + width, close[i] - width
                            value, valid, status = 0., True, 'no_event'
                            for step in range(1, h + 1):
                                row = segment.iloc[j + step - 1]
                                op = np.log(row.open)
                                up_open, down_open = op >= upper, op <= lower
                                up, down = up_open or np.log(row.high) >= upper, down_open or np.log(row.low) <= lower
                                if up_open or down_open or (up != down):
                                    down_hit = down_open or (not up_open and down)
                                    event_bin = int(np.searchsorted(spec.event_bins, step, side='left'))
                                    value = float(1 + 2 * event_bin + int(down_hit))
                                    status = 'down_event' if down_hit else 'up_event'
                                    break
                                if up and down:
                                    value, valid, status = np.nan, False, 'ambiguous_double_touch'
                                    break
                        else:
                            status = 'undefined_barrier_scale'
                elif mode == 'observed_rest_feed':
                    ready = pd.NaT
                labels[(symbol, stamp)] = (value, valid, ready, status)
    rows = [labels[(s, t)] for s, t in zip(origins.symbol, origins.feature_open_time, strict=True)]
    meta = frame.metadata.copy()
    meta['label_status'] = [r[3] for r in rows]
    frame = FactorFrame(frame.values, meta, frame.feature_names, frame.bypass_mask, frame.schema_hash)
    return RollingData(frame, np.asarray([r[0] for r in rows]),
                       pd.DatetimeIndex([r[2] for r in rows], tz='UTC'),
                       np.asarray([r[1] for r in rows]), spec, mode)


def development_folds(cutoff, count=3, *, spec=None):
    """Three historical one-month readouts; all are exposed development data."""
    spec = spec or RollingSpec()
    _, end = spec.window(cutoff)
    if count != 3:
        raise ValueError('V7 freezes three development folds')
    offset = (lambda i: pd.Timedelta(days=30*i)) if spec.window_days is not None else (
        lambda i: pd.DateOffset(months=i))
    return tuple({'fit_cutoff': (end - offset(i)).isoformat(),
                  'score_end': (end - offset(i - 1)).isoformat(),
                  'role': 'search'} for i in (3, 2, 1))
