"""Opt-in, append-only public hourly REST feed with actual receipt provenance.

No network call happens on import. Immutable snapshots plus a hash journal are
the commit boundary; bars.parquet is a recoverable, atomically replaced mirror.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import pandas as pd
import requests
from filelock import FileLock

from factorlab.data import CANONICAL_COLUMNS, HOUR, validate_klines
from factorlab.protocol import fingerprint, project_path, utc

from .evidence import ExperimentJournal, file_hash, immutable_json

API_URL = 'https://fapi.binance.com/fapi/v1/klines'
OFFICIAL_DOCS = ('https://developers.binance.com/en/docs/catalog/'
                 'core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data')
PAGE_SIZE = 96
MAX_RESPONSE_BYTES = 1024 * 1024
EXTRA_COLUMNS = ('received_at', 'availability_basis', 'prediction_origin')
SEED_BASIS = 'historical_assumed_seed'
REST_BASIS = 'actual_rest_receipt'


def _clock(now=None):
    stamp = utc(now() if callable(now) else now) if now is not None else pd.Timestamp(datetime.now(UTC))
    # Python's real clock has microsecond resolution; match canonical Parquet
    # timestamps so appending a batch does not degrade the UTC column to object.
    return stamp.as_unit('ns')


def _directory(directory):
    target = project_path(directory)
    protected = project_path('data/discovery/2025-hourly')
    if target == protected or target.is_relative_to(protected):
        raise ValueError('A live feed must use a new directory, not the historical 2025-hourly cache')
    return target


def _symbols(symbols):
    result = tuple(symbols)
    if (not 1 <= len(result) <= 3 or len(set(result)) != len(result)
        or any(not isinstance(s, str) or not re.fullmatch(r'[A-Z0-9]{3,30}', s) for s in result)):
        raise ValueError('Feed requires one to three unique uppercase public symbols')
    return result


def _validate(frame):
    audit = validate_klines(frame)
    if set(frame.columns) != {*CANONICAL_COLUMNS, *EXTRA_COLUMNS}:
        raise ValueError('Feed schema changed')
    if (frame.prediction_origin.dtype != np.bool_
        or not set(frame.availability_basis) <= {SEED_BASIS, REST_BASIS}):
        raise ValueError('Invalid feed origin or availability provenance')
    if not isinstance(frame.received_at.dtype, pd.DatetimeTZDtype) or str(frame.received_at.dt.tz) != 'UTC':
        raise ValueError('Feed receipt timestamps must be UTC')
    live = frame.availability_basis == REST_BASIS
    if (frame.received_at[live].isna().any()
        or (frame.received_at[live] < frame.close_time[live]).any()
        or (frame.available_at[live] < frame.received_at[live]).any()):
        raise ValueError('Live bars cannot predate actual complete response receipt')
    if frame.received_at[~live].notna().any():
        raise ValueError('Historical seed cannot assert an actual historical receipt')
    origins = frame[live & frame.prediction_origin]
    if origins.duplicated(['symbol', 'available_at']).any():
        raise ValueError('Only one live prediction origin per symbol/arrival batch')
    return audit


def _atomic_copy(source, destination):
    temporary = destination.with_name(destination.name + '.' + uuid4().hex + '.tmp')
    shutil.copyfile(source, temporary)
    with temporary.open('r+b') as stream:
        os.fsync(stream.fileno())
    temporary.replace(destination)


def _commit(directory, frame, event, details):
    frame.attrs = {'availability_mode': 'observed_rest_feed',
                   'feed_identity': details['feed_identity']}
    staged = directory / 'snapshots' / (uuid4().hex + '.tmp.parquet')
    staged.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(staged, index=False)
    with staged.open('r+b') as stream:
        os.fsync(stream.fileno())
    digest = file_hash(staged)
    snapshot = staged.with_name(digest + '.parquet')
    if snapshot.exists():
        if file_hash(snapshot) != digest:
            raise ValueError('Immutable feed snapshot changed')
        # Keep an unreferenced staging file after an equal commit; no risky cleanup.
    else:
        staged.replace(snapshot)
    record = {**details, 'snapshot': str(snapshot.relative_to(directory)).replace('\\', '/'),
              'sha256': digest, 'rows': len(frame)}
    ExperimentJournal(directory / 'journal.jsonl').append(event, record)
    _atomic_copy(snapshot, directory / 'bars.parquet')
    return record


def initialize_feed(directory, seed_path, symbols=None):
    """Copy a canonical historical seed into an explicitly new feed directory.

    Receipt of the copied file today does not establish historical publication.
    The original canonical availability assertions remain marked as assumptions.
    """
    target = _directory(directory)
    source = project_path(seed_path)
    if source.is_dir():
        source = source / 'bars.parquet'
    if not source.is_file() or source.suffix != '.parquet':
        raise ValueError('Feed initialization requires an existing seed bars.parquet')
    if source.is_relative_to(target):
        raise ValueError('Seed must be separate from the new feed directory')
    frame = pd.read_parquet(source)
    validate_klines(frame)
    if set(EXTRA_COLUMNS) & set(frame.columns):
        raise ValueError('Seed must be a canonical historical source, not a previously observed live feed')
    chosen = _symbols(symbols if symbols is not None else sorted(frame.symbol.unique()))
    if set(frame.symbol) != set(chosen):
        raise ValueError('Seed universe must match the declared feed symbols')
    if (frame.close_time > _clock()).any():
        raise ValueError('Seed contains unclosed future bars')
    target.mkdir(parents=True, exist_ok=True)
    with FileLock(str(target / '.feed.lock'), timeout=0):
        if any(p.name != '.feed.lock' for p in target.iterdir()):
            raise ValueError('Feed initialization requires a new empty directory')
        seed_folder = target / 'seed'
        seed_folder.mkdir()
        copied = seed_folder / 'bars.parquet'
        _atomic_copy(source, copied)
        manifest = {'format': 'discovery-feed-v1', 'symbols': list(chosen), 'endpoint': API_URL,
                    'official_docs': OFFICIAL_DOCS, 'interval': '1h', 'page_size': PAGE_SIZE,
                    'initialized_at': _clock().isoformat(),
                    'seed': {'source_path': str(source), 'sha256': file_hash(copied),
                             'rows': len(frame), 'availability_basis': SEED_BASIS,
                             'historical_publication_observed': False},
                    'revision_policy': 'reject changed existing canonical OHLCV; never overwrite',
                    'prediction_origin_policy': 'latest newly received closed bar per symbol per batch',
                    'gap_policy': 'preserve gaps; old interior gaps are not rewritten'}
        identity = fingerprint(manifest)
        immutable_json(target / 'feed_manifest.json', {'manifest': manifest, 'sha256': identity})
        immutable_json(target / 'feed.json', {'format': 'discovery-feed-v1',
                                             'manifest': 'feed_manifest.json', 'feed_identity': identity})
        frame = frame.loc[:, CANONICAL_COLUMNS].copy()
        frame['received_at'] = pd.Series(pd.NaT, index=frame.index, dtype='datetime64[ns, UTC]')
        frame['availability_basis'] = SEED_BASIS
        frame['prediction_origin'] = True
        _validate(frame)
        record = _commit(target, frame, 'seed_committed', {'feed_identity': identity,
                          'added_rows': len(frame), 'network_requests': 0})
    return {'status': 'initialized', 'directory': str(target), **record}


def _load(directory):
    manifest = json.loads((directory / 'feed_manifest.json').read_text('utf-8'))
    if manifest['sha256'] != fingerprint(manifest['manifest']):
        raise ValueError('Feed manifest integrity failure')
    marker = json.loads((directory / 'feed.json').read_text('utf-8'))
    if marker != {'format': 'discovery-feed-v1', 'manifest': 'feed_manifest.json',
                  'feed_identity': manifest['sha256']}:
        raise ValueError('Feed marker identity changed')
    copied_seed = directory / 'seed' / 'bars.parquet'
    if file_hash(copied_seed) != manifest['manifest']['seed']['sha256']:
        raise ValueError('Immutable feed seed changed')
    journal = ExperimentJournal(directory / 'journal.jsonl').read()
    commits = [row['value'] for row in journal if row['event'] in {'seed_committed', 'bars_committed'}]
    if not commits or commits[0]['feed_identity'] != manifest['sha256']:
        raise ValueError('Feed has no committed seed identity')
    for record in commits:
        snapshot = project_path(directory / record['snapshot'])
        if (not snapshot.is_relative_to(directory / 'snapshots')
            or record['feed_identity'] != manifest['sha256'] or file_hash(snapshot) != record['sha256']):
            raise ValueError('Immutable feed snapshot integrity failure')
    latest = commits[-1]
    source = directory / latest['snapshot']
    mirror = directory / 'bars.parquet'
    if not mirror.exists() or file_hash(mirror) != latest['sha256']:
        if mirror.exists() and file_hash(mirror) not in {row['sha256'] for row in commits}:
            raise ValueError('Feed bars mirror changed outside a committed refresh')
        # Recover a process interrupted after the journal commit but before mirror replacement.
        _atomic_copy(source, mirror)
    frame = pd.read_parquet(source)
    _validate(frame)
    if len(frame) != latest['rows']:
        raise ValueError('Feed committed row count differs')
    frame.attrs = {'availability_mode': 'observed_rest_feed', 'feed_identity': manifest['sha256']}
    return frame, manifest, latest


def load_feed(directory):
    target = _directory(directory)
    with FileLock(str(target / '.feed.lock'), timeout=0):
        return _load(target)[0]


def _request(session, params, now, attempts):
    for attempt in range(2):
        response = None
        try:
            attempts.append({'endpoint': API_URL, 'params': params, 'attempt': attempt + 1})
            response = session.get(API_URL, params=params, timeout=(5, 15), stream=True)
            response.raise_for_status()
            chunks, size = [], 0
            for chunk in response.iter_content(65536):
                size += len(chunk)
                if size > MAX_RESPONSE_BYTES:
                    raise ValueError('Feed REST response exceeds bounded size')
                chunks.append(chunk)
            received = _clock(now)
            payload = json.loads(b''.join(chunks))
            attempts[-1].update(received_at=received.isoformat(), response_hash=fingerprint(payload))
            return payload, received
        except requests.RequestException as exc:
            status = getattr(getattr(exc, 'response', None), 'status_code', None)
            attempts[-1]['error'] = f'{type(exc).__name__}: HTTP {status}'
            if attempt == 1 or (status is not None and status < 500):
                raise
        finally:
            if response is not None:
                response.close()
    raise RuntimeError('Unreachable feed retry state')


def _parse(payload, symbol, received, cutoff):
    if not isinstance(payload, list) or len(payload) > PAGE_SIZE:
        raise ValueError('Feed klines require a bounded array response')
    rows = []
    for item in payload:
        if not isinstance(item, list) or len(item) != 12:
            raise ValueError('Feed kline must have twelve REST fields')
        if type(item[0]) is not int or type(item[6]) is not int:
            raise ValueError('Feed REST timestamps must be integer milliseconds')
        opened = pd.Timestamp(item[0], unit='ms', tz='UTC')
        closed = opened + HOUR
        if pd.Timestamp(item[6], unit='ms', tz='UTC') != closed - pd.Timedelta(milliseconds=1):
            raise ValueError('Feed REST candle is not exactly one hour')
        if closed > cutoff or closed > received:
            continue  # Never accept a still-open hourly bar even if the server returns one.
        rows.append({'symbol': symbol, 'open_time': opened, 'close_time': closed,
                     'available_at': received, **{name: float(item[index]) for index, name in
                         enumerate(('open', 'high', 'low', 'close', 'volume'), 1)},
                     'received_at': received, 'availability_basis': REST_BASIS,
                     'prediction_origin': False})
    if not rows:
        return None
    frame = pd.DataFrame(rows, columns=(*CANONICAL_COLUMNS, *EXTRA_COLUMNS))
    validate_klines(frame)
    return frame


def refresh_feed(directory, symbols, *, now=None, max_pages=1, session=None):
    """One bounded poll, with no background process and no credentials.

    ``now`` is an injected clock for deterministic tests (timestamp or callable).
    Production callers omit it: receipt is recorded after the full HTTP body.
    Pages go backward from the latest closed hour; gaps outside this small
    catch-up window remain explicit rather than initiating a bulk download.
    """
    target, chosen = _directory(directory), _symbols(symbols)
    if type(max_pages) is not int or not 1 <= max_pages <= 4:
        raise ValueError('Feed max_pages must be an integer in 1..4')
    with FileLock(str(target / '.feed.lock'), timeout=0):
        frame, manifest, _latest = _load(target)
        if set(chosen) != set(manifest['manifest']['symbols']):
            raise ValueError('Feed symbol identity changed')
        cutoff = _clock(now)
        client = session or requests.Session()
        attempts, acquired = [], []
        try:
            for symbol in chosen:
                last_open = frame.loc[frame.symbol == symbol, 'open_time'].max()
                end_ms = int(cutoff.floor('h').timestamp() * 1000) - 1
                for _ in range(max_pages):
                    params = {'symbol': symbol, 'interval': '1h', 'limit': PAGE_SIZE, 'endTime': end_ms}
                    payload, received = _request(client, params, now, attempts)
                    page = _parse(payload, symbol, received, cutoff)
                    if page is None:
                        break
                    acquired.append(page)
                    oldest = page.open_time.min()
                    next_end = int(oldest.timestamp() * 1000) - 1
                    if next_end >= end_ms:
                        raise ValueError('Feed backward pagination did not advance')
                    if oldest <= last_open or len(payload) < PAGE_SIZE:
                        break
                    end_ms = next_end
            publish = _clock(now)
            if acquired:
                incoming = pd.concat(acquired, ignore_index=True).sort_values(['symbol', 'open_time'])
                if incoming.duplicated(['symbol', 'open_time']).any():
                    raise ValueError('Feed source repeated a bar across pages')
                existing = frame.set_index(['symbol', 'open_time'])
                additions = []
                for _, row in incoming.iterrows():
                    key = (row.symbol, row.open_time)
                    if key in existing.index:
                        previous = existing.loc[key]
                        fields = ('close_time', 'open', 'high', 'low', 'close', 'volume')
                        if any(previous[name] != row[name] for name in fields):
                            raise ValueError('Source revised an existing immutable hourly bar')
                    elif row.open_time > frame.loc[frame.symbol == row.symbol, 'open_time'].max():
                        additions.append(row.to_dict())
                if additions:
                    appended = pd.DataFrame(additions, columns=frame.columns)
                    appended['available_at'] = publish
                    for _, group in appended.groupby('symbol'):
                        appended.loc[group.open_time.idxmax(), 'prediction_origin'] = True
                    frame = pd.concat((frame, appended), ignore_index=True).sort_values(
                        ['symbol', 'open_time'], kind='stable').reset_index(drop=True)
                    audit = _validate(frame)
                    record = _commit(target, frame, 'bars_committed', {
                        'feed_identity': manifest['sha256'], 'added_rows': len(appended),
                        'live_prediction_origins': int(appended.prediction_origin.sum()),
                        'published_at': publish.isoformat(), 'requests': attempts,
                        'network_requests': len(attempts), 'audit': audit})
                    return {'status': 'updated', 'directory': str(target), **record}
            ExperimentJournal(target / 'journal.jsonl').append('poll_unchanged', {
                'feed_identity': manifest['sha256'], 'checked_at': publish.isoformat(),
                'network_requests': len(attempts), 'requests': attempts})
            return {'status': 'unchanged', 'directory': str(target), 'added_rows': 0,
                    'network_requests': len(attempts), 'live_prediction_origins': 0}
        except (ValueError, OSError, requests.RequestException) as exc:
            ExperimentJournal(target / 'journal.jsonl').append('poll_failed', {
                'feed_identity': manifest['sha256'], 'checked_at': _clock(now).isoformat(),
                'network_requests': len(attempts), 'requests': attempts,
                'error': f'{type(exc).__name__}: {exc}'})
            raise
        finally:
            if session is None:
                client.close()
