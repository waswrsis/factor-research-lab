"""Separate, receipt-aware five-minute variance experiments; no task migration.

RV is a realized squared-return proxy, not a claim of unbiased latent variance.
QLIKE proxy-ranking robustness requires conditional unbiasedness relative to
the common forecast information, and integrable loss differences. Those
conditions are not assumed to hold for these data. This is neither universal
outlier robustness nor a guarantee against market drift.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from factorlab.protocol import fingerprint

BAR = pd.Timedelta(minutes=5)


def _utc(value, label):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None or pd.isna(stamp):
        raise ValueError(f'{label} must be timezone-aware and nonmissing')
    return stamp.tz_convert('UTC')


@dataclass(frozen=True)
class TargetProtocol:
    version: str = 'v7-separate-5min-variance-v1'
    interval_minutes: int = 5
    horizon_hours: int = 12
    primary_loss: str = 'qlike_absolute_paired_difference'
    log_floor: float = 1e-12
    quantity: str = 'variance_sum_squared_log_returns'
    availability: str = 'observed_receipts_only'

    def __post_init__(self):
        if (self.version != 'v7-separate-5min-variance-v1'
                or type(self.interval_minutes) is not int or self.interval_minutes != 5
                or type(self.horizon_hours) is not int or self.horizon_hours != 12
                or self.primary_loss != 'qlike_absolute_paired_difference'
                or self.quantity != 'variance_sum_squared_log_returns'
                or self.availability != 'observed_receipts_only'
                or isinstance(self.log_floor, bool)
                or not np.isfinite(self.log_floor) or self.log_floor <= 0):
            raise ValueError('Unsupported separate target protocol')

    @property
    def digest(self):
        return fingerprint({**asdict(self), 'gap_policy': 'require all 145 consecutive closes',
                            'zero_label': 'allowed; prediction strictly positive, no silent floor',
                            'comparison': 'mean(reference_qlike-candidate_qlike), never percentage'})


def build_variance_targets(bars: pd.DataFrame, origins: pd.DataFrame, *, as_of,
                           protocol: TargetProtocol | None = None) -> pd.DataFrame:
    """Label explicit future [start,start+12h) windows at five-minute resolution.

    Bars use open_time and close prices, with a genuine received_at >= bar close.
    The close immediately before the target start is the return baseline. No
    target is published until all dependency receipts and the horizon mature.
    Gaps are missing evidence; they are never filled or replaced by hourly bars.
    """
    protocol = protocol or TargetProtocol()
    protocol.__post_init__()
    now = _utc(as_of, 'as_of')
    if not {'symbol', 'open_time', 'close', 'received_at'} <= set(bars):
        raise ValueError('Five-minute targets require symbol/open_time/close/received_at')
    if not {'symbol', 'available_at', 'forecast_window_start'} <= set(origins):
        raise ValueError('Origins require symbol/available_at/forecast_window_start')
    source = bars.copy()
    for column in ('open_time', 'received_at'):
        source[column] = pd.to_datetime([_utc(v, column) for v in source[column]], utc=True)
    if (source.symbol.isna().any() or source.duplicated(['symbol', 'open_time']).any()
            or not (source.open_time == source.open_time.dt.floor('5min')).all()
            or (source.received_at < source.open_time + BAR).any()):
        raise ValueError('Invalid or duplicate five-minute bar/receipt identity')
    if 'available_at' in source:
        source['available_at'] = [_utc(v, 'bar available_at') for v in source.available_at]
        if (source.available_at < source.received_at).any():
            raise ValueError('Bar availability cannot predate receipt')
        source['_ready'] = source.available_at
    else:
        source['_ready'] = source.received_at
    closes = source.close.to_numpy(dtype=float)
    if not np.isfinite(closes).all() or (closes <= 0).any():
        raise ValueError('Closes must be positive finite observed prices')
    lookup = {(symbol, stamp): (float(price), ready) for symbol, stamp, price, ready in
              source[['symbol', 'open_time', 'close', '_ready']].itertuples(index=False, name=None)}
    # Explicit lookup construction preserves original row order independently of
    # the order in which receipts or input files arrived.
    rows, identities = [], set()
    for origin in origins.to_dict('records'):
        available = _utc(origin['available_at'], 'origin available_at')
        start = _utc(origin['forecast_window_start'], 'forecast_window_start')
        if (pd.isna(origin['symbol']) or available > now or start < available
                or start != start.floor('5min')):
            raise ValueError('Origin must be available as-of and precede an aligned future start')
        sample_id = origin.get('sample_id', f'{origin["symbol"]}:{available.isoformat()}')
        key = (origin['symbol'], available)
        if key in identities:
            raise ValueError('Duplicate target origin')
        identities.add(key)
        end = start + pd.Timedelta(hours=protocol.horizon_hours)
        required = pd.date_range(start - BAR, end - BAR, freq=BAR)
        dependency = [lookup.get((origin['symbol'], t)) for t in required]
        value, observed, maturity = np.nan, False, pd.NaT
        if any(item is None for item in dependency):
            present = source[source.symbol == origin['symbol']].open_time
            status = ('incomplete_horizon' if present.empty or present.max() < end - BAR
                      else 'gap_or_missing_baseline')
        else:
            maturity = max(end, *(item[1] for item in dependency))
            if maturity <= now:
                returns = np.diff(np.log([item[0] for item in dependency]))
                value = float(returns @ returns)
                if not np.isfinite(value):
                    raise ValueError('Nonfinite realized variance')
                observed, status = True, 'observed'
            else:
                status = 'pending_receipt_or_horizon'
        rows.append({'sample_id': sample_id, 'symbol': origin['symbol'],
                     'available_at': available, 'forecast_window_start': start,
                     'forecast_window_end': end, 'label_available_at': maturity,
                     'target': value, 'observed': observed, 'label_status': status,
                     'target_id': protocol.digest, 'quantity': protocol.quantity,
                     'return_count': 144 if observed else 0})
    result = pd.DataFrame(rows)
    if len(result):
        result['label_available_at'] = pd.to_datetime(result.label_available_at, utc=True)
    result.attrs['target_protocol'] = asdict(protocol)
    result.attrs['target_id'] = protocol.digest
    result.attrs['scope'] = 'separate_experiment_no_hourly_task_replacement'
    return result


def variance_losses(target, prediction, *, protocol: TargetProtocol | None = None):
    """Raw QLIKE may be negative. Zero labels are valid; zero forecasts are not."""
    protocol = protocol or TargetProtocol()
    y, f = np.asarray(target, dtype=float), np.asarray(prediction, dtype=float)
    if (y.ndim != 1 or not len(y) or f.shape != y.shape or not np.isfinite(y).all()
            or not np.isfinite(f).all() or (y < 0).any() or (f <= 0).any()):
        raise ValueError('Variance labels must be nonnegative and forecasts strictly positive')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        losses = {'qlike': y / f + np.log(f), 'variance_mse': np.square(y - f),
                  'log_variance_mse': np.square(np.log(y + protocol.log_floor)
                                                 - np.log(f + protocol.log_floor))}
    if any(not np.isfinite(values).all() for values in losses.values()):
        raise ValueError('Nonfinite variance score; no sample deletion or silent forecast clipping')
    return losses


def compare_variance_forecasts(target, reference, candidate, *, protocol=None):
    protocol = protocol or TargetProtocol()
    base = variance_losses(target, reference, protocol=protocol)
    new = variance_losses(target, candidate, protocol=protocol)
    difference = base['qlike'] - new['qlike']
    def mean(values):
        result = float(np.sum(values / len(values)))
        if not np.isfinite(result):
            raise ValueError('Nonfinite aggregate variance score')
        return result
    if not np.isfinite(difference).all():
        raise ValueError('Nonfinite paired variance difference')
    return {'target_id': protocol.digest, 'primary': 'paired_qlike_difference',
            'qlike_difference': difference.tolist(), 'mean_qlike_difference': mean(difference),
            'support': len(difference), 'relative_qlike_improvement': None,
            'reference': {k: mean(v) for k, v in base.items()},
            'candidate': {k: mean(v) for k, v in new.items()},
            'proxy_condition': 'E[proxy-latent variance | common forecast information]=0 '
                               'and integrable loss differences; not established for this data',
            'claim': 'descriptive comparison; no confirmation or proxy-unbiasedness claim'}
