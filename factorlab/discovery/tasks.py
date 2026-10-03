"""Causal task datasets and explicit, audited kline preparation.

Forecast observations are the latest known hourly candle. Every predictive
window starts at the first full UTC hour after availability. Volatility is
realized log-return energy, return is cumulative log return, and event
boundaries are frozen at the known observation. Availability is an assumed historical
close-plus-delay convention, not evidence of historical delivery latency.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock

from factorlab.data import (
    HOUR,
    audit_dataset,
    download_month,
    load_archive,
    validate_klines,
)
from factorlab.features import FEATURE_COLUMNS, build_features
from factorlab.protocol import fingerprint, project_path, save_json, utc

from .contracts import ExperimentSpec, TaskDataset

FEATURE_WARMUP_HOURS = 48
EVENT_SCALE_HOURS = 24


def build_task_dataset(bars: pd.DataFrame, spec: ExperimentSpec) -> TaskDataset:
    """Build finite causal windows without dropping rows for unavailable labels.

    Windows and label observation periods never cross an hourly gap. Predictive
    labels conservatively mature after the complete horizon, even when an event
    occurs earlier. Missing future candles and within-candle double touches are
    unobserved, never class zero. Tail maturity is scheduled but not observed.
    State is the existing trailing 24-hour trend, a descriptive auxiliary target.
    """
    validate_klines(bars)
    if set(bars.symbol) != set(spec.symbols):
        raise ValueError('Input universe differs from the frozen experiment universe')
    delay = pd.Timedelta(seconds=spec.availability_delay_seconds)
    live = bars.attrs.get('availability_mode') == 'observed_rest_feed'
    if live:
        required = {'received_at', 'availability_basis', 'prediction_origin'}
        if not required <= set(bars) or bars.prediction_origin.dtype != bool:
            raise ValueError('Observed feed requires explicit receipt/origin provenance')
        if not bars.availability_basis.isin(['historical_assumed_seed', 'actual_rest_receipt']).all():
            raise ValueError('Unknown observed feed availability basis')
        if (not isinstance(bars.received_at.dtype, pd.DatetimeTZDtype)
                or str(bars.received_at.dt.tz) != 'UTC'):
            raise ValueError('Observed feed receipt timestamps must be timezone-aware UTC')
        actual = bars.availability_basis == 'actual_rest_receipt'
        if (bars.loc[actual, 'received_at'].isna().any()
                or (bars.loc[actual, 'received_at'] > bars.loc[actual, 'available_at']).any()
                or (bars.loc[actual, 'received_at'] < bars.loc[actual, 'close_time']).any()):
            raise ValueError('Observed feed has invalid receipt times')
        if not (bars.loc[~actual, 'available_at'] == bars.loc[~actual, 'close_time'] + delay).all():
            raise ValueError('Historical feed seed differs from frozen availability convention')
        for _, group in bars.groupby('symbol'):
            if not group.available_at.is_monotonic_increasing:
                raise ValueError('Observed feed availability must not move backward')
        origins = bars[bars.prediction_origin]
        if origins.duplicated(['symbol', 'available_at']).any():
            raise ValueError('Observed feed has duplicate prediction origins')
        for _, batch in bars.loc[actual].groupby(['symbol', 'available_at']):
            selected = batch[batch.prediction_origin]
            if len(selected) and selected.open_time.iloc[0] != batch.open_time.max():
                raise ValueError('Only the latest closed candle in a receipt batch may be a prediction origin')
    elif not (bars.available_at == bars.close_time + delay).all():
        raise ValueError('Bar availability differs from the frozen close-plus-delay convention')
    # Input order is checked before sorting: out-of-order source rows are errors.
    features = build_features(bars.reset_index(drop=True))
    records, windows = [], []
    targets = {t.name: [] for t in spec.tasks}
    observed = {t.name: [] for t in spec.tasks}
    maturity = {t.name: [] for t in spec.tasks}
    start, end = utc(spec.start), utc(spec.end)
    for symbol, group in features.groupby('symbol', sort=False):
        segments = group.open_time.diff().ne(HOUR).cumsum()
        for _, segment in group.groupby(segments, sort=False):
            segment = segment.reset_index(drop=True)
            values = segment.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
            finite = np.isfinite(values).all(axis=1)
            # Float32 overflow is a feature-quality failure rather than implicit clipping.
            finite &= (np.abs(values) <= np.finfo(np.float32).max).all(axis=1)
            invalid = np.concatenate(([0], np.cumsum(~finite)))
            log_close = np.log(segment.close.to_numpy(dtype=float))
            log_returns = np.diff(log_close)
            highs = np.log(segment.high.to_numpy(dtype=float))
            lows = np.log(segment.low.to_numpy(dtype=float))
            for i in range(spec.lookback - 1, len(segment)):
                row = segment.iloc[i]
                if live and not row.prediction_origin:
                    continue
                if not start <= row.available_at < end:
                    continue
                left = i + 1 - spec.lookback
                if invalid[i + 1] != invalid[left]:
                    continue
                record = {
                    'sample_id': f'{symbol}:{row.open_time.isoformat()}',
                    'symbol': symbol, 'available_at': row.available_at,
                    'feature_open_time': row.open_time,
                    'window_start_open_time': segment.open_time.iloc[left],
                    'forecast_window_start': row.available_at.ceil('h'),
                }
                future_start = int((record['forecast_window_start'] - row.open_time) / HOUR) + i
                windows.append(values[left:i + 1].astype(np.float32))
                for task in spec.tasks:
                    h = task.horizon_hours
                    ready = (row.available_at if task.kind == 'state'
                             else record['forecast_window_start'] + h * HOUR + delay)
                    if live and task.predictive:
                        # A scheduled horizon is not proof the observations were received.
                        ready = (max(ready, segment.available_at.iloc[future_start:future_start + h].max())
                                 if future_start + h <= len(segment) else pd.NaT)
                    valid, target = False, np.nan
                    status = 'incomplete_horizon'
                    if task.kind == 'state':
                        target = float(row.state_trend_24h)
                        valid = bool(np.isfinite(target))
                        status = 'descriptive' if valid else 'undefined_state'
                    elif future_start + h <= len(segment):
                        # The pre-window close is a label baseline, never a model input.
                        future_returns = log_returns[future_start - 1:future_start + h - 1]
                        if task.kind == 'volatility':
                            target = float(np.sqrt(np.square(future_returns).sum()))
                            valid, status = True, 'observed'
                        elif task.kind == 'return':
                            target = float(log_close[future_start + h - 1] - log_close[future_start - 1])
                            valid, status = True, 'observed'
                        else:
                            # A trailing scale in the same units as the log-price barriers.
                            scale = float(np.std(log_returns[i - EVENT_SCALE_HOURS:i], ddof=0))
                            width = task.barrier_multiplier * scale * np.sqrt(h)
                            record[f'event_barrier_log_width_{task.name}'] = width
                            if i < EVENT_SCALE_HOURS or not np.isfinite(width) or width <= 1e-12:
                                status = 'undefined_barrier_scale'
                            else:
                                valid, target, status = True, 0., 'no_event'
                                upper, lower = log_close[i] + width, log_close[i] - width
                                for step in range(1, h + 1):
                                    future_index = future_start + step - 1
                                    log_open = float(np.log(segment.open.iloc[future_index]))
                                    # An open beyond a boundary has known first direction;
                                    # later intrabar extremes cannot make that hit ambiguous.
                                    open_up, open_down = log_open >= upper, log_open <= lower
                                    up = open_up or highs[future_index] >= upper
                                    down = open_down or lows[future_index] <= lower
                                    if open_up or open_down:
                                        event_bin = int(np.searchsorted(task.event_bins, step, side='left'))
                                        target = float(1 + 2 * event_bin + int(open_down))
                                        status = 'up_event' if open_up else 'down_event'
                                        break
                                    if up and down:
                                        valid, target, status = False, np.nan, 'ambiguous_double_touch'
                                        break
                                    if up or down:
                                        event_bin = int(np.searchsorted(task.event_bins, step, side='left'))
                                        target = float(1 + 2 * event_bin + int(down))
                                        status = 'up_event' if up else 'down_event'
                                        break
                    # Freeze diagnostic scale even for an incomplete horizon; it uses no future.
                    if task.kind == 'first_event' and f'event_barrier_log_width_{task.name}' not in record:
                        scale = float(np.std(log_returns[i - EVENT_SCALE_HOURS:i], ddof=0))
                        record[f'event_barrier_log_width_{task.name}'] = (
                            task.barrier_multiplier * scale * np.sqrt(h)
                        )
                    targets[task.name].append(target)
                    observed[task.name].append(valid)
                    maturity[task.name].append(ready.to_datetime64())
                    record[f'label_status_{task.name}'] = status
                records.append(record)
    columns = ['sample_id', 'symbol', 'available_at', 'feature_open_time',
               'window_start_open_time', 'forecast_window_start']
    meta = pd.DataFrame(records) if records else pd.DataFrame(columns=columns)
    for name in ('available_at', 'feature_open_time', 'window_start_open_time', 'forecast_window_start'):
        meta[name] = pd.to_datetime(meta[name], utc=True)
    order = meta.sort_values(['available_at', 'symbol'], kind='stable').index.to_numpy()
    x = (np.stack(windows) if windows else
         np.empty((0, spec.lookback, len(FEATURE_COLUMNS)), dtype=np.float32))
    meta = meta.iloc[order].reset_index(drop=True)
    meta.attrs['task_semantics'] = {
        'forecast_window_start': 'first full UTC hour at or after available_at',
        'volatility': 'sqrt(sum of h full future candle log increments squared), pre-window close baseline',
        'return': 'log(last future window close / pre-window close), label-only baseline',
        'state': 'descriptive trailing return_24h, no predictive claim',
        'first_event': 'frozen trailing 24h log std * sqrt(h) * multiplier barriers',
        'maturity': 'forecast_window_start + h hours + frozen assumed availability delay',
        'gap_policy': 'restart features and windows; incomplete labels unobserved',
    }
    meta.attrs['availability_mode'] = 'observed_rest_feed' if live else 'assumed_historical_delay'
    if live:
        meta.attrs['feed_identity'] = bars.attrs.get('feed_identity')
    return TaskDataset(
        x[order], {k: np.asarray(v, dtype=float)[order] for k, v in targets.items()},
        {k: np.asarray(v, dtype=bool)[order] for k, v in observed.items()},
        {k: np.asarray(v, dtype='datetime64[ns]')[order] for k, v in maturity.items()},
        meta, FEATURE_COLUMNS, spec.tasks,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_bars(spec: ExperimentSpec, output_dir: str | Path) -> pd.DataFrame:
    """Explicitly reuse/download SHA-verified Binance UM hourly klines only.

    Write bars.parquet and bars_manifest.json under the project. An existing
    manifest pins the exact data request and parquet hash; mismatches fail rather
    than overwriting old coverage. Archive acquisition itself preserves SHA
    versions and verifies provenance/checksum sidecars on every cache reuse.
    """
    directory = project_path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    warmup_start = utc(spec.start) - (FEATURE_WARMUP_HOURS + spec.lookback) * HOUR
    end = utc(spec.end)
    request = {
        'schema_version': 1, 'symbols': list(spec.symbols), 'interval': '1h',
        'kind': 'klines', 'start': warmup_start.isoformat(), 'end_exclusive': end.isoformat(),
        'availability_delay_seconds': spec.availability_delay_seconds,
        'gap_policy': 'preserve_no_fill',
    }
    digest = fingerprint(request)
    parquet, manifest_path = directory / 'bars.parquet', directory / 'bars_manifest.json'
    with FileLock(str(directory / '.bars.lock')):
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text('utf-8'))
            if manifest.get('request_sha256') != digest or manifest.get('request') != request:
                raise ValueError('Existing data manifest differs; choose a new output directory')
            if not parquet.is_file() or _sha256(parquet) != manifest.get('parquet_sha256'):
                raise ValueError('Prepared parquet missing or checksum mismatch')
            # A verified source history remains required even when parquet is cached.
            for item in manifest['archives']:
                path = project_path(item['path'])
                frame = load_archive(path, item['symbol'], 'klines', spec.availability_delay_seconds)
                if frame.attrs['provenance']['sha256'] != item['sha256']:
                    raise ValueError('Cached archive provenance differs from manifest')
            bars = pd.read_parquet(parquet)
            validate_klines(bars)
            bars.attrs['discovery_manifest'] = manifest
            return bars
        if parquet.exists():
            raise ValueError('Unmanifested parquet exists; choose a new output directory')
        months = pd.period_range(warmup_start.tz_localize(None).to_period('M'),
                                 (end - pd.Timedelta(nanoseconds=1)).tz_localize(None).to_period('M'),
                                 freq='M').astype(str)
        parts, archives = [], []
        for symbol in spec.symbols:
            for month in months:
                path = download_month(symbol, month, 'klines')
                frame = load_archive(path, symbol, 'klines', spec.availability_delay_seconds)
                provenance = frame.attrs['provenance']
                archives.append({'path': str(path.resolve()), **provenance})
                selected = frame[(frame.open_time >= warmup_start) & (frame.open_time < end)]
                if len(selected):
                    selected = selected.copy()
                    selected.attrs = {}
                    parts.append(selected)
        if not parts:
            raise ValueError('No klines in the frozen requested interval')
        bars = pd.concat(parts, ignore_index=True).sort_values(['symbol', 'open_time']).reset_index(drop=True)
        if set(bars.symbol) != set(spec.symbols):
            raise ValueError('Requested universe has missing archive coverage')
        audit = audit_dataset(bars)
        # Count grid-aligned opens in [start, end), including nonhour boundaries.
        expected_hours = int((end.ceil('h') - warmup_start.ceil('h')) / HOUR)
        audit['expected_hours_per_symbol'] = expected_hours
        audit['missing_hours_in_requested_calendar'] = {
            symbol: expected_hours - int((bars.symbol == symbol).sum()) for symbol in spec.symbols
        }
        staged = directory / 'bars.parquet.tmp'
        bars.to_parquet(staged, index=False)
        staged.replace(parquet)
        manifest = {
            'request': request, 'request_sha256': digest, 'parquet_sha256': _sha256(parquet),
            'archives': archives, 'audit': audit,
        }
        save_json(manifest_path, manifest)
        bars.attrs['discovery_manifest'] = manifest
        return bars
