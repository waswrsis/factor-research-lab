"""Optional point-in-time context; historical event time is not publication time.

Missing/stale observations are encoded as zero plus an explicit observed channel.
The old price/volume dataset and frozen protocols are unchanged unless opted in.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from factorlab.features import build_features
from factorlab.protocol import fingerprint, project_path, utc

from .contracts import ExperimentSpec, TaskDataset
from .evidence import file_hash, immutable_json
from .tasks import build_task_dataset

TABLE_COLUMNS = ('symbol', 'observed_at', 'available_at', 'value')
API_BASE = 'https://fapi.binance.com'
OFFICIAL_DOCS = ('https://developers.binance.com/en/docs/catalog/'
                 'core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data')


def _aware(values, name):
    parsed = []
    for value in values:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp) or stamp.tzinfo is None:
            raise ValueError(f'{name} requires explicit timezone-aware timestamps')
        parsed.append(stamp.tz_convert('UTC'))
    return pd.DatetimeIndex(parsed, tz='UTC')


def validate_context_table(table: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Validate explicit event and knowability times, preserving revisions.

    No API/archive funding_time + assumed latency is accepted as publication
    evidence. Imported available_at is a user/source assertion, not certified here.
    """
    if kind not in {'funding', 'open_interest'} or not set(TABLE_COLUMNS) <= set(table):
        raise ValueError('Context table requires symbol/observed_at/available_at/value')
    if table.attrs.get('availability_basis') == 'assumed_live_delay':
        raise ValueError('Assumed archive delay is not point-in-time publication evidence')
    result = table.copy()
    for name in ('observed_at', 'available_at'):
        result[name] = _aware(result[name], name)
    if result.symbol.isna().any() or not result.symbol.map(
            lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError('Context symbols must be nonempty strings')
    result['value'] = pd.to_numeric(result.value, errors='raise').astype(float)
    if not np.isfinite(result.value).all():
        raise ValueError('Context observations must be finite; omit missing source rows')
    if kind == 'open_interest' and (result.value < 0).any():
        raise ValueError('Open interest must be nonnegative')
    if (result.observed_at > result.available_at).any():
        raise ValueError('Observation cannot occur after availability')
    if result.duplicated(['symbol', 'observed_at', 'available_at']).any():
        raise ValueError('Duplicate point-in-time observation')
    # Multiple event rows received in one historical API response are valid.
    return result.sort_values(['symbol', 'available_at', 'observed_at'], kind='stable')


def read_context_table(path: str | Path, kind: str) -> pd.DataFrame:
    source = project_path(path)
    if source.suffix.lower() == '.csv':
        table = pd.read_csv(source)
    elif source.suffix.lower() == '.parquet':
        table = pd.read_parquet(source)
    else:
        raise ValueError('Context imports support CSV or parquet')
    result = validate_context_table(table, kind)
    result.attrs['provenance'] = {'path': str(source), 'sha256': file_hash(source),
                                'available_at_basis': 'explicit_source_assertion'}
    return result


def _asof_context(rows, table, kind, max_age_hours):
    values = np.zeros((len(rows), 3), dtype=np.float64)
    for symbol, group in rows.groupby('symbol', sort=False):
        source = table[table.symbol == symbol].copy()
        if source.empty:
            continue
        # Latest event known so far, with revisions only taking effect on receipt.
        # A late receipt of an older event must not replace a newer known event.
        source = source.sort_values(['available_at', 'observed_at'], kind='stable')
        newest = source.observed_at.cummax()
        source = source[source.observed_at == newest]
        source = source.drop_duplicates('available_at', keep='last')
        right = source[['available_at', 'observed_at', 'value']].rename(
            columns={'available_at': 'source_available_at'})
        left = group[['available_at']].sort_values('available_at').copy()
        left['position'] = left.index
        joined = pd.merge_asof(left, right.sort_values('source_available_at'),
                               left_on='available_at', right_on='source_available_at',
                               direction='backward', allow_exact_matches=True)
        age = (joined.available_at - joined.observed_at).dt.total_seconds() / 3600
        good = joined.value.notna() & age.between(0, max_age_hours)
        positions = joined.loc[good, 'position'].to_numpy(dtype=int)
        values[positions, 0] = joined.loc[good, 'value']
        values[positions, 1] = 1.0
        values[positions, 2] = age[good]
    return values, (kind, f'{kind}_observed', f'{kind}_age_hours')


def _cross_context(rows, symbols, max_age_hours):
    arrays, names = [], []
    for peer in symbols:
        source = rows[rows.symbol == peer][
            ['available_at', 'open_time', 'return_1h', 'volatility_24h']].rename(
                columns={'available_at': 'peer_available_at', 'open_time': 'peer_open_time',
                         'return_1h': 'peer_return', 'volatility_24h': 'peer_volatility'})
        block = np.zeros((len(rows), 8), dtype=np.float64)
        for symbol, group in rows.groupby('symbol', sort=False):
            if symbol == peer or source.empty:
                continue
            left = group.sort_values('available_at').copy()
            left['position'] = left.index
            joined = pd.merge_asof(left, source.sort_values('peer_available_at'),
                left_on='available_at', right_on='peer_available_at', direction='backward',
                tolerance=pd.Timedelta(hours=min(1, max_age_hours)))
            good_ret = np.isfinite(joined.peer_return)
            good_vol = np.isfinite(joined.peer_volatility)
            spread = joined.return_1h - joined.peer_return
            good_spread = good_ret & np.isfinite(spread)
            # Restart trailing coupling after gaps or a repeated/stale peer candle.
            restart = (joined.open_time.diff().ne(pd.Timedelta(hours=1))
                       | joined.peer_open_time.diff().ne(pd.Timedelta(hours=1)))
            corr = pd.Series(np.nan, index=joined.index)
            for _, part in joined.groupby(restart.cumsum()):
                corr.loc[part.index] = part.return_1h.rolling(24, min_periods=24).corr(part.peer_return)
            good_corr = np.isfinite(corr)
            for pair, value, good in ((0, joined.peer_return, good_ret),
                                      (2, joined.peer_volatility, good_vol),
                                      (4, spread, good_spread), (6, corr, good_corr)):
                positions = joined.loc[good, 'position'].to_numpy(dtype=int)
                block[positions, pair] = value[good]
                block[positions, pair + 1] = 1.0
        arrays.append(block)
        for base in ('return_1h', 'volatility_24h', 'return_spread_1h', 'correlation_24h'):
            names.extend((f'peer_{peer}_{base}', f'peer_{peer}_{base}_observed'))
    return (np.concatenate(arrays, axis=1) if arrays else np.empty((len(rows), 0))), tuple(names)


def augment_context(data: TaskDataset, bars: pd.DataFrame, *, funding=None, open_interest=None,
                    cross_assets=True, max_age_hours=24) -> TaskDataset:
    """Append context at every historical window timestamp, never only at origin."""
    if (isinstance(max_age_hours, bool) or not isinstance(max_age_hours, (int, float))
            or not np.isfinite(max_age_hours) or max_age_hours <= 0):
        raise ValueError('Context maximum age must be finite and positive')
    needed = {'feature_open_time', 'window_start_open_time'}
    if not needed <= set(data.meta):
        raise ValueError('Context augmentation needs canonical bar window metadata')
    rows = build_features(bars).reset_index(drop=True)
    if set(data.meta.symbol) - set(rows.symbol):
        raise ValueError('Dataset symbols absent from supplied bars')
    symbols = tuple(sorted(rows.symbol.unique()))
    blocks, additions, sources = [], [], {}
    for kind, table in (('funding', funding), ('open_interest', open_interest)):
        if table is None:
            continue
        table = validate_context_table(table, kind)
        if not set(table.symbol) <= set(rows.symbol):
            raise ValueError('Context table contains symbols outside supplied bars')
        values, names = _asof_context(rows, table, kind, max_age_hours)
        blocks.append(values)
        additions.extend(names)
        sources[kind] = {'rows': len(table), 'sha256': fingerprint([
            {'symbol': r.symbol, 'observed_at': r.observed_at.isoformat(),
             'available_at': r.available_at.isoformat(), 'value': r.value}
            for r in table.itertuples()]), 'provenance': table.attrs.get('provenance'),
            'availability_basis': table.attrs.get('availability_basis', 'explicit_source_assertion')}
    if cross_assets:
        values, names = _cross_context(rows, symbols, max_age_hours)
        blocks.append(values)
        additions.extend(names)
    if not additions:
        return data
    if set(additions) & set(data.features):
        raise ValueError('Context features already present')
    context = np.concatenate(blocks, axis=1).astype(np.float32)
    if not np.isfinite(context).all():
        raise ValueError('Context cannot be represented as finite float32')
    keys = pd.MultiIndex.from_frame(rows[['symbol', 'open_time']])
    if not keys.is_unique:
        raise ValueError('Duplicate canonical bar keys')
    windows = np.empty((len(data.meta), data.x.shape[1], len(additions)), dtype=np.float32)
    for i, record in enumerate(data.meta.itertuples()):
        times = pd.date_range(record.window_start_open_time, periods=data.x.shape[1], freq='h')
        positions = keys.get_indexer(pd.MultiIndex.from_arrays([[record.symbol] * len(times), times]))
        if (positions < 0).any() or times[-1] != record.feature_open_time:
            raise ValueError('Context windows do not match the supplied canonical bars')
        if (rows.iloc[positions].available_at > record.available_at).any():
            raise ValueError('Context input bar was not available by prediction origin')
        windows[i] = context[positions]
    meta = data.meta.copy()
    meta.attrs.update(data.meta.attrs)
    meta.attrs['context_policy'] = {'format': 'causal-context-v1', 'max_age_hours': max_age_hours,
        'cross_assets': bool(cross_assets), 'peer_universe': symbols, 'sources': sources,
        'missing': 'zero_value_with_zero_observed_channel',
        'knowability': 'backward_asof_explicit_available_at_each_window_step'}
    return TaskDataset(np.concatenate((data.x, windows), axis=2),
        {k: v.copy() for k, v in data.targets.items()}, {k: v.copy() for k, v in data.observed.items()},
        {k: v.copy() for k, v in data.maturity.items()}, meta, (*data.features, *additions), data.tasks)


def build_context_dataset(bars: pd.DataFrame, spec: ExperimentSpec, **kwargs) -> TaskDataset:
    return augment_context(build_task_dataset(bars, spec), bars, **kwargs)


def funding_archive_context(frame: pd.DataFrame, received_at) -> pd.DataFrame:
    """Reuse parsed archive values, replacing its assumed latency with actual receipt."""
    received = _aware([received_at], 'received_at')[0]
    table = pd.DataFrame({'symbol': frame.symbol, 'observed_at': frame.funding_time,
                          'available_at': received, 'value': frame.funding_rate})
    table.attrs['availability_basis'] = 'actual_archive_receipt_not_historical_publication'
    return validate_context_table(table, 'funding')


def fetch_context(kind: str, symbol: str, *, start=None, end=None, limit=100,
                  session=None) -> pd.DataFrame:
    """One bounded public request. Funding history or current OI snapshot.

    Funding event times do not establish historical publish/receive times.
    Consequently all downloaded historical events become available on receipt.
    No old OI is manufactured from the limited recent-history endpoint.
    """
    if kind not in {'funding', 'open_interest'} or not symbol.isalnum():
        raise ValueError('Unknown context kind or invalid symbol')
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError('Funding request limit must be 1..1000')
    params = {'symbol': symbol}
    if kind == 'funding':
        params['limit'] = limit
        for name, value in (('startTime', start), ('endTime', end)):
            if value is not None:
                params[name] = int(utc(value).timestamp() * 1000)
        if start is not None and end is not None and utc(start) > utc(end):
            raise ValueError('Funding request start is after end')
        endpoint = '/fapi/v1/fundingRate'
    else:
        if start is not None or end is not None:
            raise ValueError('Current OI snapshot cannot supply a historical requested interval')
        endpoint = '/fapi/v1/openInterest'
    client = session or requests
    response = client.get(API_BASE + endpoint, params=params, timeout=(10, 30))
    response.raise_for_status()
    payload = response.json()
    received = pd.Timestamp.now(tz='UTC')
    records = payload if kind == 'funding' else [payload]
    if not isinstance(records, list):
        raise TypeError('Context endpoint did not return expected records')
    rows = []
    for record in records:
        if record['symbol'] != symbol:
            raise ValueError('Endpoint returned another symbol')
        event = record['fundingTime'] if kind == 'funding' else record['time']
        rows.append({'symbol': symbol, 'observed_at': pd.Timestamp(event, unit='ms', tz='UTC'),
                     'available_at': received, 'value': float(record[
                         'fundingRate' if kind == 'funding' else 'openInterest'])})
    table = pd.DataFrame(rows, columns=TABLE_COLUMNS)
    table.attrs.update(availability_basis='actual_http_receipt',
        provenance={'endpoint': API_BASE + endpoint, 'request': params,
                    'retrieved_at': received.isoformat(), 'official_docs': OFFICIAL_DOCS,
                    'bounded_request': True, 'complete_history_claimed': False})
    return validate_context_table(table, kind)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Bounded optional funding/OI point-in-time acquisition')
    parser.add_argument('kind', choices=['funding', 'open_interest'])
    parser.add_argument('--symbol', required=True)
    parser.add_argument('--start')
    parser.add_argument('--end')
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    path = project_path(args.output)
    if path.suffix != '.parquet' or path.exists() or path.with_suffix('.json').exists():
        raise ValueError('Use a new parquet destination')
    table = fetch_context(args.kind, args.symbol, start=args.start, end=args.end, limit=args.limit)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    immutable_json(path.with_suffix('.json'), {'kind': args.kind, 'rows': len(table),
        'sha256': file_hash(path), 'provenance': table.attrs,
        'historical_publication_claimed': False})
    print(json.dumps({'output': str(path), 'rows': len(table),
                      'historical_publication_claimed': False}))


if __name__ == '__main__':
    main()
