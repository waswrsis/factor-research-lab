"""Bounded public candle receipts, as-of revisions and incremental raw archive.

Downloads are training data. Only the latest newly received closed candle is a
potential forward origin, and the controller must issue before its next window.
Imported archive times remain explicitly assumed, never actual past receipts.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from filelock import FileLock

from factorlab.protocol import fingerprint, project_path
from factorlab.time_utils import HOUR

from .evidence import file_hash
from .process_policy import isolated_process_kwargs, restricted_environment
from .v7_lifecycle import stamp
from .v7_storage import immutable_record

ENDPOINT = 'https://fapi.binance.com/fapi/v1/klines'
POLL_SECONDS = 30.
POLL_CLEANUP_SECONDS = 2.
STORAGE_FORMAT = 'v7-receipt-cas-v3'


class SnapshotUnavailable(ValueError):
    """The verified archive has no published observation by this cutoff.

    This is an input availability failure, distinct from damaged provenance.
    It does not authorize a retry of an already prepaid forward work item.
    """


class TransientFeedError(TimeoutError):
    """Explicit retryable transport failure, never an integrity/schema failure."""
    def __init__(self, code):
        if code not in {'timeout', 'connection', 'http_429'} and not re.fullmatch(r'http_5\d\d', code):
            raise ValueError('Unknown transient feed error code')
        self.code = code
        detail = ' (deadline exhausted or late response)' if code == 'timeout' else ''
        super().__init__(f'Public feed transient transport: {code}{detail}')


def _transient_packet(value):
    if not isinstance(value, dict):
        raise ValueError('Malformed public transport error')  # noqa: TRY004 - untrusted wire schema
    if value in ({'kind': 'timeout'}, {'kind': 'connection'}):
        return TransientFeedError(value['kind'])
    status = value.get('status')
    if (set(value) == {'kind', 'status'} and value['kind'] == 'http' and type(status) is int
            and (status == 429 or 500 <= status <= 599)):
        return TransientFeedError(f'http_{status}')
    raise ValueError('Malformed or nonretryable public transport error')


class ReceiptArchive:
    def __init__(self, directory, symbols=('BTCUSDT',), *, clock=None):
        self.directory = project_path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.clock = clock or (lambda: pd.Timestamp.now(tz='UTC'))
        self.symbols = tuple(symbols)
        if (not 1 <= len(self.symbols) <= 3 or len(set(self.symbols)) != len(self.symbols)
                or any(not s.isalnum() or not s.isupper() for s in self.symbols)):
            raise ValueError('Bounded uppercase symbol universe required')
        self.identity = fingerprint({'version': 'v7-receipt-archive-v1', 'symbols': self.symbols,
            'endpoint': ENDPOINT, 'interval': '1h', 'revision_policy': 'append_asof_never_overwrite'})
        immutable_record(self.directory / 'identity.json', {'source_id': self.identity, 'symbols': list(self.symbols)})
        self.db = sqlite3.connect(self.directory / 'receipts.sqlite', timeout=30)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS batches (id TEXT PRIMARY KEY, record TEXT, hash TEXT);
            CREATE TABLE IF NOT EXISTS bars (symbol TEXT, opened TEXT, receipt TEXT, version TEXT,
                origin INTEGER, body TEXT, batch TEXT, PRIMARY KEY(symbol,opened,version));
            CREATE INDEX IF NOT EXISTS asof_bars ON bars(receipt,opened,symbol);
            CREATE TABLE IF NOT EXISTS publications (batch TEXT PRIMARY KEY, sequence INTEGER UNIQUE,
                published_at TEXT, record TEXT, hash TEXT);
        ''')

    def close(self):
        self.db.close()

    def _commit(self, frame, *, received_at, actual, source_ref, nominal_source=None):
        with FileLock(str(self.directory / '.receipt.lock'), timeout=30):
            return self._commit_locked(frame, received_at=received_at, actual=actual, source_ref=source_ref, nominal_source=nominal_source)

    def _commit_locked(self, frame, *, received_at, actual, source_ref, nominal_source=None):
        if nominal_source is not None:
            from .nominal_trial_controller import NominalPublication
            if type(nominal_source) is not NominalPublication:
                raise TypeError("Original typed nominal publication required")
            nominal_source.verify(self, frame, received_at, actual, source_ref)
        received = stamp(received_at)
        now = stamp(self.clock())
        if received > now or frame.empty or not set(frame.symbol) <= set(self.symbols):
            raise ValueError('Archive needs a nonempty actually received, bounded source')
        clean = frame.copy()
        clean['open_time'] = pd.DatetimeIndex(clean.open_time).tz_convert('UTC')
        if (clean.duplicated(['symbol', 'open_time']).any()
                or (clean.open_time != clean.open_time.dt.floor('h')).any()
                or (nominal_source is None and (clean.open_time + HOUR > received).any())):
            raise ValueError('Unique, fully closed hourly candles required')
        records = []
        for row in clean.itertuples():
            prices = [float(getattr(row, n)) for n in ('open', 'high', 'low', 'close')]
            volume = float(row.volume) if hasattr(row, 'volume') else 0.
            import math
            if (not all(math.isfinite(v) and v > 0 for v in prices) or not math.isfinite(volume)
                    or volume < 0 or max(prices[0], prices[3]) > prices[1]
                    or min(prices[0], prices[3]) < prices[2]):
                raise ValueError('Invalid OHLCV cannot enter receipt archive')
            records.append({'symbol': row.symbol, 'open_time': row.open_time.isoformat(),
                            **dict(zip(('open', 'high', 'low', 'close'), prices, strict=True)),
                            'volume': volume})
        # Serialization changes, not source/availability semantics: v1 identity
        # and old inline receipts remain valid. Never rewrite or remove them.
        digest = fingerprint(records)
        relative = f'objects/{digest}.json'
        immutable_record(self.directory / relative, records)
        # Serialize writers before choosing revision order. Data preparation is
        # committed first; unpublished preparations NEVER enter snapshots.
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            now = stamp(self.clock())
            if now < received:
                raise ValueError('Archive publication clock moved backward')
            sequence = self.db.execute('SELECT COALESCE(MAX(sequence),0)+1 FROM publications').fetchone()[0]
            batch = {'source': self.identity, 'received_at': received.isoformat(),
                     'prepared_at': now.isoformat(), 'actual': actual, 'source_ref': source_ref,
                     'storage_format': STORAGE_FORMAT, 'revision_sequence': sequence,
                     'bars_ref': {'path': relative, 'hash': digest, 'count': len(records)}}
            identifier = fingerprint(batch)
            immutable_record(self.directory / 'raw' / f'{identifier}.json', batch)
            new_rows = new_origins = 0
            prepared = []
            source_cache = {}
            for value in records:
                old = self.db.execute('SELECT b.body,b.batch,b.version FROM bars b LEFT JOIN publications p ON p.batch=b.batch '
                    'WHERE b.symbol=? AND b.opened=? AND (p.batch IS NULL OR p.published_at IS NOT NULL) '
                    'ORDER BY COALESCE(p.published_at,b.receipt) DESC, '
                    '(p.batch IS NOT NULL) DESC,p.sequence DESC,b.rowid DESC LIMIT 1',
                    (value['symbol'], value['open_time'])).fetchone()
                # Polling the same unchanged candle is not a new origin or revision.
                if old:
                    body = json.loads(old[0])
                    if fingerprint(body) != old[2]:
                        raise ValueError('Prior candle revision integrity mismatch')
                    self._verify_bar_source(old[1], body, cache=source_cache)
                    if body['ohlcv'] == value:
                        continue
                body = {'ohlcv': value, 'received_at': received.isoformat() if actual else None,
                        'prepared_at': now.isoformat(), 'revision_sequence': sequence,
                        'basis': 'actual_receipt' if actual else 'assumed_archive'}
                prepared.append((value, body))
            latest_by_symbol = {}
            for value, _ in prepared:
                latest_by_symbol[value['symbol']] = max(latest_by_symbol.get(value['symbol'], ''), value['open_time'])
            for value, body in prepared:
                latest_stored = self.db.execute('SELECT MAX(b.opened) FROM bars b '
                    'LEFT JOIN publications p ON p.batch=b.batch WHERE b.symbol=? '
                    'AND (p.batch IS NULL OR p.published_at IS NOT NULL)',
                                               (value['symbol'],)).fetchone()[0]
                latest_batch = latest_by_symbol[value['symbol']]
                origin = bool(actual and value['open_time'] == latest_batch
                              and (latest_stored is None or value['open_time'] > latest_stored))
                body['prediction_origin'] = origin
                version = fingerprint(body)
                self.db.execute('INSERT INTO bars VALUES (?,?,?,?,?,?,?)',
                    (value['symbol'], value['open_time'], now.isoformat(), version, int(origin),
                     json.dumps(body, sort_keys=True, allow_nan=False), identifier))
                new_rows += 1
                new_origins += origin
            self.db.execute('INSERT INTO batches VALUES (?,?,?)',
                (identifier, json.dumps(batch, sort_keys=True, allow_nan=False), identifier))
            self.db.execute('INSERT INTO publications VALUES (?,?,NULL,NULL,NULL)', (identifier, sequence))
        # All source bytes and the revision index are now durable. This clock
        # is a data-publication linearization observation, not a claim that a
        # later certificate/index write took zero time. If this publication
        # transaction fails, retain the preparation but never recover its old
        # timestamp automatically. A later actual poll gets a fresh receipt.
        self._publish(identifier, sequence, now)
        return {'batch': identifier, 'new_rows': new_rows, 'new_origins': new_origins}

    def _publish(self, identifier, sequence, prepared_at):
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            published = stamp(self.clock())
            if published < prepared_at:
                raise ValueError('Archive publication clock moved backward')
            value = {'source': self.identity, 'batch': identifier, 'revision_sequence': sequence,
                     'published_at': published.isoformat(),
                     'policy': 'after_durable_data_and_revision_commit'}
            immutable_record(self.directory / 'publications' / f'{identifier}.json', value)
            self.db.execute('UPDATE publications SET published_at=?,record=?,hash=? WHERE batch=?',
                (published.isoformat(), json.dumps(value, sort_keys=True), fingerprint(value), identifier))

    def import_seed(self, path):
        path = project_path(path)
        frame = pd.read_parquet(path)
        return self._commit(frame, received_at=self.clock(), actual=False,
                            source_ref={'path': str(path), 'sha256': file_hash(path)})

    def receive(self, frame, *, received_at, response_hash):
        return self._commit(frame, received_at=received_at, actual=True,
                            source_ref={'response_sha256': response_hash})

    def snapshot(self, as_of, *, start=None, archive_delay_seconds=5):
        if type(archive_delay_seconds) is not int or not 0 <= archive_delay_seconds <= 3600:
            raise ValueError('Assumed archive availability delay must be integer seconds in bounds 0..3600')
        as_of = stamp(as_of)
        if as_of > stamp(self.clock()):
            raise ValueError('Cannot snapshot a future receipt boundary')
        start = stamp(start) if start is not None else pd.Timestamp('1970-01-01', tz='UTC')
        rows = self.db.execute('SELECT b.body,b.origin,b.version,b.receipt,b.batch,b.symbol,b.opened FROM bars b '
            'LEFT JOIN publications p ON p.batch=b.batch '
            'WHERE (p.batch IS NULL OR p.published_at IS NOT NULL) '
            'AND COALESCE(p.published_at,b.receipt)<=? AND b.opened>=? '
            'ORDER BY b.symbol,b.opened,COALESCE(p.published_at,b.receipt),'
            '(p.batch IS NOT NULL),p.sequence,b.rowid',
            (as_of.isoformat(), start.isoformat()))
        chosen = {}
        source_cache = {}
        for text, origin, version, receipt, batch, symbol, opened in rows:
            body = json.loads(text)
            if (fingerprint(body) != version or body.get('prepared_at', body.get('imported_at')) != receipt
                    or body['prediction_origin'] != bool(origin)
                    or body['ohlcv']['symbol'] != symbol or body['ohlcv']['open_time'] != opened):
                raise ValueError('Raw candle revision integrity mismatch')
            published = self._verify_bar_source(batch, body, cache=source_cache)
            if published is None:
                continue
            item = body['ohlcv']
            chosen[item['symbol'], item['open_time']] = {**item, 'received_at': body['received_at'],
                'imported_at': published, 'availability_basis': body['basis'],
                'prediction_origin': bool(origin), 'revision': version, 'batch_id': batch}
        if not chosen:
            raise SnapshotUnavailable('No source received by snapshot cutoff')
        frame = pd.DataFrame(chosen.values()).sort_values(['symbol', 'open_time']).reset_index(drop=True)
        # Stored timestamps are canonical ISO values but may have different
        # subsecond precision. Inferring one format from the first row rejects
        # valid later polling receipts; preserve every observed instant exactly.
        frame['open_time'] = pd.to_datetime(frame.open_time, utc=True, format='ISO8601')
        frame['received_at'] = pd.to_datetime(frame.received_at, utc=True, format='ISO8601')
        frame['imported_at'] = pd.to_datetime(frame.imported_at, utc=True, format='ISO8601')
        frame['close_time'] = frame.open_time + HOUR
        frame['available_at'] = frame.imported_at.where(frame.received_at.notna(),
            frame.close_time + pd.Timedelta(seconds=archive_delay_seconds))
        frame.attrs = {'availability_mode': 'mixed_seed_and_observed_feed',
                       'feed_identity': self.identity, 'snapshot_asof': as_of.isoformat()}
        return frame

    def poll(self, *, session=None, pages=1):
        """One 30s transport budget across all symbols/pages, including startup.

        Default HTTP runs in an owned, lightweight child with its own watchdog.
        Injected sessions are a cooperative test seam, not a hard timeout claim.
        Earlier completed pages remain committed if a later page fails. Archive
        disk I/O / SQLite locking is outside the HTTP worker deadline guarantee.
        """
        if type(pages) is not int or not 1 <= pages <= 4:
            raise ValueError('Hourly catch-up has at most four pages')
        client = session
        if client is not None:
            _public_session(client)
        deadline = time.monotonic() + POLL_SECONDS
        results = []
        with FileLock(str(self.directory / '.poll.lock'), timeout=0):
            try:
                for symbol in self.symbols:
                    end = stamp(self.clock()).floor('h')
                    lower_bound = end - pages * 96 * HOUR
                    for _ in range(pages):
                        params = {'symbol': symbol, 'interval': '1h', 'limit': 96,
                                  'endTime': int(end.timestamp() * 1000) - 1}
                        payload, received = self._fetch_page(params, client, deadline)
                        if not isinstance(payload, list) or not 1 <= len(payload) <= 96:
                            raise ValueError('Public klines require a bounded nonempty list')
                        bars = []
                        for item in payload:
                            if (not isinstance(item, list) or len(item) != 12 or type(item[0]) is not int
                                    or type(item[6]) is not int):
                                raise ValueError('Public hourly candle schema changed')
                            opened = pd.Timestamp(item[0], unit='ms', tz='UTC')
                            if pd.Timestamp(item[6], unit='ms', tz='UTC') != opened + HOUR - pd.Timedelta(milliseconds=1):
                                raise ValueError('Public hourly candle endpoint changed')
                            if opened + HOUR <= received:
                                bars.append({'symbol': symbol, 'open_time': opened,
                                    **{n: float(item[i]) for i, n in enumerate(('open', 'high', 'low', 'close', 'volume'), 1)}})
                        if not bars:
                            break
                        if time.monotonic() >= deadline - POLL_CLEANUP_SECONDS:
                            raise TransientFeedError('timeout')
                        result = self.receive(pd.DataFrame(bars), received_at=received,
                                              response_hash=fingerprint(payload))
                        results.append(result)
                        oldest = min(b['open_time'] for b in bars)
                        if len(payload) < 96 or (not result['new_rows'] and
                                self._published_coverage(symbol, lower_bound, oldest)):
                            break
                        end = oldest
            finally:
                pass  # Caller owns injected clients; each child closes its own session.
        return results

    def _published_coverage(self, symbol, start, end):
        """Bounded request-skipping check, never a replacement for source verify.

        Only the originally requested <=384-hour interval is considered.
        Missing v3 publication indices and pending preparations cannot count.
        """
        if end <= start:
            return True
        hours = int((end - start) / HOUR)
        if hours > 384:
            return False
        rows = self.db.execute('SELECT DISTINCT b.opened FROM bars b '
            'JOIN batches r ON r.id=b.batch LEFT JOIN publications p ON p.batch=b.batch '
            'WHERE b.symbol=? AND b.opened>=? AND b.opened<? AND '
            "(p.published_at IS NOT NULL OR (p.batch IS NULL AND "
            "COALESCE(json_extract(r.record,'$.storage_format'),'inline') != ?))",
            (symbol, start.isoformat(), end.isoformat(), STORAGE_FORMAT))
        expected = {value.isoformat() for value in pd.date_range(start, end, freq='h', inclusive='left')}
        return {row[0] for row in rows} == expected

    def _fetch_page(self, params, client, deadline):
        from scripts.v7_public_fetch import transient_error
        try:
            return self._fetch_page_once(params, client, deadline)
        except TransientFeedError:
            raise
        except Exception as exc:
            packet = transient_error(exc)
            if packet is not None:
                raise _transient_packet(packet) from exc
            raise

    def _fetch_page_once(self, params, client, deadline):
        work_deadline = deadline - POLL_CLEANUP_SECONDS
        if time.monotonic() >= work_deadline:
            raise TimeoutError('Public poll total deadline exhausted')
        if client is None:
            payload, received = _owned_fetch(params, deadline)
        else:
            remaining = work_deadline - time.monotonic()
            response = client.get(ENDPOINT, params=params, timeout=(min(5, remaining), min(15, remaining)),
                                  stream=True, allow_redirects=False)
            try:
                response.raise_for_status()
                if 300 <= response.status_code < 400:
                    raise ValueError('Public receipt endpoint cannot redirect')
                chunks, size = [], 0
                for chunk in response.iter_content(65536):
                    if time.monotonic() >= work_deadline:
                        raise TimeoutError('Public poll total deadline exhausted')
                    size += len(chunk)
                    if size > 1_048_576:
                        raise ValueError('Public response exceeds one MiB')
                    chunks.append(chunk)
                payload, received = json.loads(b''.join(chunks)), stamp(self.clock())
            finally:
                response.close()
        if time.monotonic() >= work_deadline:
            raise TimeoutError('Public poll late response rejected')
        return payload, received

    def verify(self):
        if self.db.execute('SELECT COUNT(*) FROM publications p LEFT JOIN batches b ON b.id=p.batch '
                           'WHERE b.id IS NULL').fetchone()[0]:
            raise ValueError('Publication index refers to a missing raw batch')
        count = 0
        verified_objects = {}  # At most 128 count/content-hash tuples; scoped to this call.
        for identifier, text, digest in self.db.execute('SELECT * FROM batches'):
            value = json.loads(text)
            path = self.directory / 'raw' / f'{identifier}.json'
            if (fingerprint(value) != digest or digest != identifier
                    or json.loads(path.read_text('utf-8')) != value):
                raise ValueError('Receipt archive raw object mismatch')
            self._batch_bars(value, verified_objects=verified_objects)
            self._publication(identifier, value)
            count += 1
        source_cache = {}
        for symbol, opened, receipt, version, origin, text, batch in self.db.execute('SELECT * FROM bars'):
            body = json.loads(text)
            if (fingerprint(body) != version or body.get('prepared_at', body.get('imported_at')) != receipt
                    or body['prediction_origin'] != bool(origin)
                    or body['ohlcv']['symbol'] != symbol or body['ohlcv']['open_time'] != opened):
                raise ValueError('Candle index does not match immutable revision')
            self._verify_bar_source(batch, body, cache=source_cache)
        return {'batches': count, 'revisions': self.db.execute('SELECT COUNT(*) FROM bars').fetchone()[0],
                'source_id': self.identity, 'automatic_deletion': False,
                'storage_format': STORAGE_FORMAT, 'legacy_inline_supported': True,
                'unpublished_batches': self.db.execute(
                    'SELECT COUNT(*) FROM publications WHERE published_at IS NULL').fetchone()[0]}

    def _publication(self, identifier, batch):
        if batch.get('storage_format') != STORAGE_FORMAT:
            return batch['published_at']
        row = self.db.execute('SELECT sequence,published_at,record,hash FROM publications WHERE batch=?',
                              (identifier,)).fetchone()
        if row is None or row[0] != batch['revision_sequence']:
            raise ValueError('Missing or inconsistent publication index')
        if row[1] is None:
            if row[2] is not None or row[3] is not None:
                raise ValueError('Unpublished receipt has an inconsistent publication index')
            return None  # Never auto-adopt a stray certificate after a failed commit.
        record = json.loads(row[2])
        expected = {'source': self.identity, 'batch': identifier, 'revision_sequence': row[0],
                    'published_at': row[1], 'policy': 'after_durable_data_and_revision_commit'}
        path = self.directory / 'publications' / f'{identifier}.json'
        if (record != expected or fingerprint(record) != row[3]
                or json.loads(path.read_text('utf-8')) != record
                or stamp(row[1]) < stamp(batch['prepared_at'])):
            raise ValueError('Publication certificate/index mismatch')
        return row[1]

    def _batch_bars(self, batch, *, verified_objects=None):
        """Return content, or None for a verification-only cache hit.

        Every reference still checks source, format, path, hash syntax and count.
        The optional cache holds only bounded count/content-hash tuples, never candles.
        Snapshot and lineage callers always read content without this shortcut.
        """
        if batch.get('source') != self.identity:
            raise ValueError('Raw receipt source identity mismatch')
        if 'storage_format' not in batch:
            if 'bars_ref' in batch or not isinstance(batch.get('bars'), list):
                raise ValueError('Invalid legacy raw receipt')
            return batch['bars']
        reference = batch.get('bars_ref', {})
        digest = reference.get('hash', '') if isinstance(reference, dict) else None
        if (not isinstance(reference, dict) or batch['storage_format'] not in {STORAGE_FORMAT, 'v7-receipt-cas-v2'} or 'bars' in batch
                or not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None
                or reference.get('path') != f'objects/{digest}.json'
                or type(reference.get('count')) is not int or reference['count'] <= 0):
            raise ValueError('Invalid content-addressed receipt reference')
        path = self.directory / reference['path']
        if (path.resolve().parent != (self.directory / 'objects').resolve()
                or not path.resolve().is_relative_to(self.directory.resolve())):
            raise ValueError('Receipt object escapes archive')
        # Read once: metadata is not evidence of content identity. Cache hits
        # still hash the actual bytes; misses parse these same captured bytes.
        # This also prevents binding one read's hash to another read's content.
        raw = path.read_bytes()
        signature = (len(raw), hashlib.sha256(raw).hexdigest())
        if verified_objects is not None and digest in verified_objects:
            cached_count, cached_signature = verified_objects[digest]
            if cached_count != reference['count']:
                raise ValueError('Content-addressed candle object count mismatch')
            if cached_signature == signature:
                return None
        values = json.loads(raw.decode('utf-8'))
        if (not isinstance(values, list) or len(values) != reference['count']
                or fingerprint(values) != digest):
            raise ValueError('Content-addressed candle object mismatch')
        if verified_objects is not None:
            if len(verified_objects) >= 128:
                verified_objects.pop(next(iter(verified_objects)))
            verified_objects[digest] = (len(values), signature)
        return values

    def _verify_bar_source(self, identifier, body, *, cache=None):
        cache = {} if cache is None else cache
        if identifier not in cache:
            row = self.db.execute('SELECT record,hash FROM batches WHERE id=?', (identifier,)).fetchone()
            if not row:
                raise ValueError('Candle points to a missing committed raw batch')
            batch = json.loads(row[0])
            raw = self.directory / 'raw' / f'{identifier}.json'
            if (row[1] != identifier or fingerprint(batch) != identifier
                    or json.loads(raw.read_text('utf-8')) != batch):
                raise ValueError('Committed raw receipt mismatch')
            if len(cache) >= 128:
                cache.pop(next(iter(cache)))
            cache[identifier] = (batch, {fingerprint(value) for value in self._batch_bars(batch)},
                                 self._publication(identifier, batch))
        batch, hashes, published = cache[identifier]
        new = batch.get('storage_format') == STORAGE_FORMAT
        if (fingerprint(body['ohlcv']) not in hashes
                or body.get('prepared_at', body.get('imported_at')) != batch['prepared_at' if new else 'published_at']
                or (new and body.get('revision_sequence') != batch['revision_sequence'])
                or body['received_at'] != (batch['received_at'] if batch['actual'] else None)):
            raise ValueError('Candle/raw receipt lineage mismatch')
        return published


def _public_session(client):
    if hasattr(client, 'trust_env'):
        client.trust_env = False
    headers = {str(key).lower() for key in getattr(client, 'headers', {})}
    if (headers & {'authorization', 'proxy-authorization', 'cookie', 'x-mbx-apikey', 'x-api-key'}
            or any(getattr(client, key, None) for key in ('auth', 'cookies', 'params', 'proxies'))):
        raise ValueError('Public market session cannot carry credentials, cookies or request overrides')


def _owned_fetch(params, deadline):
    """Kill only this Popen child: the fixed worker cannot spawn descendants.

    Popen retains its OS process handle / unreaped child identity, avoiding a
    PID lookup or a broad tree/group signal. Child inherits service cgroups.
    """
    remaining = deadline - POLL_CLEANUP_SECONDS - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('Public poll total deadline exhausted before launch')
    script = Path(__file__).resolve().parents[2] / 'scripts' / 'v7_public_fetch.py'
    command = [sys.executable, '-B', str(script), str(time.time() + remaining)]
    environment = restricted_environment()
    worker = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, shell=False, env=environment,
                              **isolated_process_kwargs())
    try:
        remaining = deadline - POLL_CLEANUP_SECONDS - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Public poll startup exhausted total deadline')
        try:
            output, _ = worker.communicate(json.dumps(params).encode(), timeout=remaining)
        except subprocess.TimeoutExpired as error:
            raise TimeoutError('Public poll total deadline exhausted') from error
        if time.monotonic() >= deadline - POLL_CLEANUP_SECONDS:
            raise TimeoutError('Public poll late response rejected')
        if worker.returncode == 124:
            raise TransientFeedError('timeout')
        if worker.returncode != 0 or len(output) > 1_060_000:
            raise ValueError('Public receipt worker failed or exceeded output bounds')
        value = json.loads(output)
        if isinstance(value, dict) and set(value) == {'error'}:
            raise _transient_packet(value['error'])
        if set(value) != {'payload', 'received_at'}:
            raise ValueError('Invalid public receipt worker output')
        return value['payload'], stamp(value['received_at'])
    finally:
        try:
            if worker.poll() is None:
                worker.kill()
                try:
                    worker.wait(timeout=max(.001, deadline - time.monotonic()))
                except subprocess.TimeoutExpired as error:
                    raise RuntimeError('Owned public receipt worker cleanup failed') from error
                if worker.poll() is None:
                    raise RuntimeError('Owned public receipt worker remained alive after cleanup')
        finally:
            for pipe in (worker.stdin, worker.stdout):
                if pipe is not None:
                    pipe.close()
