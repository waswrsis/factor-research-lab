"""Bounded daily USD-M archive bridge, with actual later feed import.

Daily history is assumed training data, never a past prediction origin. Source,
UTC day interval and all attempts are frozen; import/download are explicit.
Default HTTP is an owned deadline-limited process. Session injection is only a
cooperative transport seam. No network runs merely by importing this module.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import sqlite3
import stat
import subprocess
import sys
import threading
import time
import zipfile

import pandas as pd
import requests
from filelock import FileLock

from factorlab.data import _checksum, parse_klines_csv
from factorlab.protocol import fingerprint, project_path

from .evidence import file_hash
from .process_policy import isolated_process_kwargs, restricted_environment
from .v7_lifecycle import stamp
from .v7_run import source_files
from .v7_seed import (
    ARCHIVE_SECONDS,
    MAX_ATTEMPTS,
    MAX_BYTES,
    _blob,
    _fetch,
    _public_session,
    _read,
    _record,
)

BASE = 'https://data.binance.vision/data/futures/um/daily/klines'
MAX_DAYS = 366


def _now():
    return pd.Timestamp.now(tz='UTC')


def _day(value):
    value = pd.Timestamp(value, tz='UTC') if isinstance(value, str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}', value) else stamp(value)
    if value != value.floor('D'):
        raise ValueError('Daily bridge boundaries must be exact UTC days')
    return value


def _archive(symbol, day):
    name = f'{symbol}-1h-{day}.zip'
    return {'symbol': symbol, 'day': day, 'filename': name,
            'url': f'{BASE}/{symbol}/1h/{name}'}


def _remaining(request, *, cleanup=2):
    elapsed = (stamp(_now()) - stamp(request['reserved_at'])).total_seconds()
    if elapsed < 0:
        raise ValueError('Daily reservation clock moved backward')
    return max(0., ARCHIVE_SECONDS - cleanup - elapsed)


def _validate(payload, archive):
    if len(payload) > MAX_BYTES:
        raise ValueError('Daily ZIP exceeds bounded bytes')
    with zipfile.ZipFile(io.BytesIO(payload)) as zipped:
        members = zipped.infolist()
        if len(members) != 1:
            raise ValueError('Daily ZIP requires exactly one CSV')
        member = members[0]
        if (member.filename != archive['filename'][:-4] + '.csv' or member.is_dir()
                or member.flag_bits & 1 or stat.S_IFMT(member.external_attr >> 16) not in (0, stat.S_IFREG)
                or member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                or member.file_size > MAX_BYTES):
            raise ValueError('Unsafe or oversized daily container')
        raw = zipped.read(member)
    if len(raw.splitlines()) > 25:
        raise ValueError('Daily CSV exceeds 24 hours plus header')
    frame = parse_klines_csv(raw, archive['symbol'])
    start = pd.Timestamp(archive['day'], tz='UTC')
    if not pd.DatetimeIndex(frame.open_time).equals(pd.date_range(start, periods=24, freq='h')):
        raise ValueError('Daily archive must contain its exact complete 24 UTC hours')
    return frame


def _network(request_path, session=None):
    path = project_path(request_path)
    request = _read(path)
    manifest = _read(request['manifest'])
    archive = request['archive']
    if (request['job_hash'] != fingerprint(manifest) or source_files() != manifest['identity']['source_files']
            or archive not in manifest['identity']['archives']
            or archive != _archive(archive['symbol'], archive['day'])):
        raise ValueError('Daily request differs from frozen source/endpoint')
    remaining = _remaining(request)
    client = requests.Session() if session is None else session
    began = time.monotonic()
    deadline = began + remaining
    result = {'job_hash': request['job_hash'], 'status': 'failed', 'started_at': _now().isoformat(),
              'transport': 'owned_worker' if session is None else 'injected_session'}
    try:
        checksum = _fetch(client, archive['url'] + '.CHECKSUM', deadline)
        _blob(path.parent / 'checksum.bin', checksum)
        expected = _checksum(checksum.decode(), archive['filename'])
        if request['pinned_sha256'] is not None and expected != request['pinned_sha256']:
            raise ValueError('Daily archive revision changed after first checksum')
        payload = _fetch(client, archive['url'], deadline)
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError('Daily checksum mismatch')
        _blob(path.parent / 'archive.bin', payload)
        result.update(status='downloaded', sha256=expected, bytes=len(payload))
    except (OSError, ValueError, UnicodeError, requests.RequestException, TimeoutError) as exc:
        result.update(error_type=type(exc).__name__)
    finally:
        if session is None:
            client.close()
        result.update(finished_at=_now().isoformat(), elapsed_seconds=time.monotonic() - began)
        _record(path.parent / 'transport.json', result)
    return result


def _download(request_path, session):
    if session is not None:
        return {**_network(request_path, session), 'hard_deadline_enforced': False}
    request = _read(request_path)
    command = [sys.executable, '-B', '-m', 'factorlab.discovery.v7_daily', 'fetch', '--request', str(request_path)]
    environment = restricted_environment()
    environment.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    remaining = _remaining(request, cleanup=0)
    deadline = time.monotonic() + remaining
    if remaining <= 2:
        return {'status': 'failed', 'error_type': 'ReservationDeadline', 'hard_deadline_enforced': True}
    with (request_path.parent / 'worker.log').open('ab') as log:
        worker = subprocess.Popen(command, cwd=project_path('.'), env=environment, shell=False,
            stdin=subprocess.DEVNULL, stdout=log, stderr=log, **isolated_process_kwargs())
        try:
            remaining = min(_remaining(request), deadline - 2 - time.monotonic())
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, 0)
            worker.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            return {'status': 'failed', 'error_type': 'WorkerDeadline', 'hard_deadline_enforced': True}
        finally:
            if worker.poll() is None:
                worker.kill()  # Only our retained Popen identity; fixed child spawns no descendants.
                try:
                    worker.wait(timeout=2)
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError('Daily owned worker cleanup unverified') from exc
                if worker.poll() is None:
                    raise RuntimeError('Daily owned worker remained alive')
    receipt = request_path.parent / 'transport.json'
    if worker.returncode != 0 or not receipt.exists():
        return {'status': 'failed', 'error_type': 'WorkerFailure', 'hard_deadline_enforced': True}
    return {**_read(receipt), 'hard_deadline_enforced': True}


def _pin(raw, archive, attempt, job_hash):
    checksum = attempt / 'checksum.bin'
    if checksum.exists():
        digest = _checksum(checksum.read_text('utf-8'), archive['filename'])
        _record(raw / 'selection.json', {'job_hash': job_hash, 'sha256': digest})


def _accept(raw, archive, attempt, job_hash):
    _pin(raw, archive, attempt, job_hash)
    if not (attempt / 'transport.json').exists():
        return None
    transport = _read(attempt / 'transport.json')
    if transport['job_hash'] != job_hash or transport['status'] != 'downloaded':
        return None
    supervision = attempt / 'supervision.json'
    if supervision.exists() and _read(supervision)['status'] != 'downloaded':
        return None
    request = _read(attempt / 'request.json')
    reserved = stamp(request['reserved_at'])
    elapsed = transport.get('elapsed_seconds')
    if (type(elapsed) not in (int, float) or not math.isfinite(elapsed) or not 0 <= elapsed < ARCHIVE_SECONDS - 2
            or not reserved <= stamp(transport['started_at']) <= stamp(transport['finished_at'])
            or stamp(transport['finished_at']) >= reserved + pd.Timedelta(seconds=ARCHIVE_SECONDS - 2)):
        raise ValueError('Daily transport completed outside original deadline')
    digest = _read(raw / 'selection.json')['sha256']
    payload = (attempt / 'archive.bin').read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError('Daily staged archive hash mismatch')
    frame = _validate(payload, archive)
    target = raw / f'{digest}.zip'
    _blob(target, payload)
    previous = _read(raw / 'accepted.json') if (raw / 'accepted.json').exists() else None
    accepted = {'job_hash': job_hash, 'archive': archive, 'path': str(target), 'sha256': digest,
                'rows': len(frame), 'request_hash': fingerprint(request), 'attempt': request['attempt'],
                'received_at': transport['finished_at'],
                'accepted_at': previous['accepted_at'] if previous else _now().isoformat(),
                'historical_published_at_known': False, 'actual_feed_imported_at': None}
    if not stamp(transport['finished_at']) <= stamp(accepted['accepted_at']) <= stamp(_now()):
        raise ValueError('Daily acceptance clock invalid')
    _record(raw / 'accepted.json', accepted)
    return accepted


def _accepted(raw, archive, job_hash, requests_for_archive):
    path = raw / 'accepted.json'
    if not path.exists():
        return None
    value = _read(path)
    target = project_path(value['path'])
    number = value['attempt']
    if (value['job_hash'] != job_hash or value['archive'] != archive or target.parent != raw
            or value['sha256'] != _read(raw / 'selection.json')['sha256']
            or type(number) is not int or not 0 <= number < len(requests_for_archive)
            or fingerprint(requests_for_archive[number]) != value['request_hash']
            or file_hash(target) != value['sha256'] or value['rows'] != 24):
        raise ValueError('Accepted daily archive differs from frozen receipt')
    if _accept(raw, archive, raw / f'attempt-{number}', job_hash) != value:
        raise ValueError('Accepted daily transport receipt changed')
    return value


def prepare_daily_bridge(directory, symbols=('BTCUSDT',), *, start, end=None, as_of=None,
                         max_new_archives=1, session=None):
    """Freeze <=366 complete days; <=3 attempts/day, <=3 new attempts/call.

    Default end leaves at most 16 days for bounded REST at planning time.
    Resume with as_of/end omitted retains the original window, never extends it.
    A completed bridge is still training history needing actual feed import.
    """
    if (not isinstance(symbols, (tuple, list)) or not 1 <= len(symbols) <= 3
            or len(set(symbols)) != len(symbols)
            or any(not isinstance(s, str) or not re.fullmatch('[A-Z0-9]{3,30}', s) for s in symbols)
            or type(max_new_archives) is not int or not 0 <= max_new_archives <= 3):
        raise ValueError('Daily bridge needs 1..3 symbols and 0..3 new attempts per call')
    if session is not None:
        _public_session(session)
    directory = project_path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / '.daily.lock'), timeout=0):
        manifest_path, database = directory / 'manifest.json', directory / 'budget.sqlite'
        previous = _read(manifest_path) if manifest_path.exists() else None
        if previous is None and any(p.name != '.daily.lock' for p in directory.iterdir()):
            raise ValueError('Daily manifest missing; cannot reset a nonempty job')
        if previous is not None and not database.exists():
            raise ValueError('Daily budget journal missing; never reset attempts')
        now = stamp(_now())
        observed = stamp(previous['identity']['as_of'] if previous and as_of is None else now if as_of is None else as_of)
        start = _day(start)
        end = _day(previous['identity']['end'] if previous and end is None else
                   max(start, observed.floor('D') - pd.Timedelta(days=15)) if end is None else end)
        if not start < end <= observed.floor('D') or observed > now or (end - start).days > MAX_DAYS:
            raise ValueError('Daily bridge requires 1..366 completed UTC days within observed cutoff')
        archives = [_archive(symbol, day.strftime('%Y-%m-%d')) for symbol in symbols
                    for day in pd.date_range(start, end, freq='D', inclusive='left')]
        identity = {'version': 'v7-daily-bridge-v1', 'directory': str(directory), 'symbols': list(symbols), 'start': start.isoformat(),
            'end': end.isoformat(), 'as_of': observed.isoformat(), 'source_files': source_files(),
            'archives': archives, 'attempts_per_archive': MAX_ATTEMPTS, 'archive_seconds': ARCHIVE_SECONDS,
            'http_reservation_ceiling': 2 * MAX_ATTEMPTS * len(archives), 'byte_ceiling': MAX_BYTES,
            'transport': 'injected_session' if session is not None else 'bounded_subprocess'}
        manifest = previous or {'identity': identity, 'planned_at': now.isoformat()}
        if manifest['identity'] != identity:
            raise ValueError('Frozen daily identity/source/window cannot change')
        _record(manifest_path, manifest)
        job_hash = fingerprint(manifest)
        db = sqlite3.connect(database)
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA synchronous=FULL')
        db.execute('CREATE TABLE IF NOT EXISTS attempts (seq INTEGER PRIMARY KEY,record TEXT,hash TEXT)')
        try:
            requests_all = []
            for seq, text, digest in db.execute('SELECT seq,record,hash FROM attempts ORDER BY seq'):
                request = json.loads(text)
                if (seq != len(requests_all) or fingerprint(request) != digest or request['job_hash'] != job_hash
                        or request['manifest'] != str(manifest_path) or request['archive'] not in archives):
                    raise ValueError('Daily attempt journal integrity failure')
                reservation = directory / 'reservations' / f'{seq:05d}.json'
                if not reservation.exists() or _read(reservation) != request:
                    raise ValueError('Daily independent reservation missing or changed')
                requests_all.append(request)
            if len(list((directory / 'reservations').glob('*.json'))) != len(requests_all):
                raise ValueError('Daily journal differs from immutable reservations')
            accepted, new_attempts, failure = [], 0, None
            for archive in archives:
                raw = directory / 'raw' / archive['symbol'] / archive['day']
                raw.mkdir(parents=True, exist_ok=True)
                prior = [r for r in requests_all if r['archive'] == archive]
                if len(prior) > MAX_ATTEMPTS:
                    raise ValueError('Daily attempts exceed frozen maximum')
                if not prior and any(raw.iterdir()):
                    raise ValueError('Daily evidence exists without charged attempts')
                pinned_checked = False
                for number, request in enumerate(prior):
                    attempt = raw / f'attempt-{number}'
                    if request['attempt'] != number or _read(attempt / 'request.json') != request:
                        raise ValueError('Daily archive request differs from reservation')
                    checksum = attempt / 'checksum.bin'
                    if checksum.exists() and not pinned_checked:
                        try:
                            digest = _checksum(checksum.read_text('utf-8'), archive['filename'])
                        except (ValueError, UnicodeError):
                            continue
                        _record(raw / 'selection.json', {'job_hash': job_hash, 'sha256': digest})
                        pinned_checked = True
                if len(list(raw.glob('attempt-*'))) != len(prior):
                    raise ValueError('Daily attempt directory differs from journal')
                value = _accepted(raw, archive, job_hash, prior)
                if value is None:
                    for number in range(len(prior)):
                        try:
                            value = _accept(raw, archive, raw / f'attempt-{number}', job_hash)
                        except (ValueError, OSError, zipfile.BadZipFile):
                            continue
                        if value is not None:
                            break
                if value is not None:
                    accepted.append(value)
                    continue
                if len(prior) >= MAX_ATTEMPTS:
                    failure = 'budget_exhausted'
                    continue
                if new_attempts >= max_new_archives:
                    continue
                selection = _read(raw / 'selection.json')['sha256'] if (raw / 'selection.json').exists() else None
                request = {'manifest': str(manifest_path), 'job_hash': job_hash, 'archive': archive,
                           'attempt': len(prior), 'reserved_at': _now().isoformat(), 'pinned_sha256': selection}
                seq = len(requests_all)
                # Independent reservation first: an interrupted split commit
                # fails closed rather than silently refunding an unobserved attempt.
                _record(directory / 'reservations' / f'{seq:05d}.json', request)
                with db:
                    db.execute('INSERT INTO attempts VALUES (?,?,?)', (seq, json.dumps(request), fingerprint(request)))
                requests_all.append(request)
                attempt = raw / f'attempt-{len(prior)}'
                _record(attempt / 'request.json', request)
                new_attempts += 1
                result = {}
                try:
                    result = _download(attempt / 'request.json', session)
                    _record(attempt / 'supervision.json', result)
                    value = _accept(raw, archive, attempt, job_hash)
                    if result['status'] != 'downloaded' or value is None:
                        raise ValueError('Daily transfer failed')
                    accepted.append(value)
                except (OSError, ValueError, requests.RequestException, zipfile.BadZipFile):
                    failure = 'attempt_failed'
                _record(attempt / 'outcome.json', {'job_hash': job_hash,
                    'status': 'accepted' if value is not None else 'attempt_failed', 'finished_at': _now().isoformat()})
            counts = {'archives_required': len(archives), 'archives_accepted': len(accepted),
                      'archive_attempts_consumed': len(requests_all), 'reserved_http_requests': 2 * len(requests_all),
                      'new_archive_attempts': new_attempts}
            base = {'job_hash': job_hash, 'manifest_path': str(manifest_path), 'data_ready': False,
                    'actual_feed_imported_at': None, 'counts': counts, 'bridge_end': end.isoformat(),
                    'remaining_rest_hours_at_this_call': (now - end).total_seconds() / 3600,
                    'requires_new_bridge': now - end > pd.Timedelta(days=16)}
            if len(accepted) != len(archives):
                return {**base, 'status': failure or 'pending', 'bridge_artifact_ready': False}
            if source_files() != identity['source_files']:
                raise ValueError('Daily source changed during preparation')
            complete_path, bars_path = directory / 'complete.json', directory / 'bars.parquet'
            if complete_path.exists():
                complete = _read(complete_path)
                if (complete['job_hash'] != job_hash or complete['archives'] != accepted
                        or file_hash(bars_path) != complete['bars_sha256']):
                    raise ValueError('Completed daily bridge integrity failure')
            else:
                frames = [_validate(project_path(a['path']).read_bytes(), a['archive']) for a in accepted]
                bars = pd.concat(frames, ignore_index=True).sort_values(['symbol', 'open_time']).reset_index(drop=True)
                bars.attrs = {'availability_basis': 'assumed_live_delay', 'daily_job_hash': job_hash,
                              'historical_published_at_known': False, 'actual_feed_import_required': True}
                buffer = io.BytesIO()
                bars.to_parquet(buffer, index=False)
                _blob(bars_path, buffer.getvalue())
                complete = {'job_hash': job_hash, 'archives': accepted, 'bars_sha256': file_hash(bars_path),
                            'rows': len(bars), 'prepared_at': _now().isoformat(), 'actual_feed_imported_at': None}
                _record(complete_path, complete)
            return {**base, 'status': 'complete', 'bridge_artifact_ready': True, 'bars_path': str(bars_path),
                    'completion_receipt': str(complete_path), 'prepared_at': complete['prepared_at']}
        finally:
            db.close()


def import_daily_bridge(directory, feed):
    """Verify the complete frozen job, then import at the actual feed clock."""
    directory = project_path(directory)
    manifest = _read(directory / 'manifest.json')
    identity = manifest['identity']
    if identity['source_files'] != source_files() or tuple(identity['symbols']) != tuple(feed.symbols):
        raise ValueError('Daily bridge source/universe differs from actual feed')
    # A transport marker preserves frozen mode without performing network.
    session = None
    if identity['transport'] == 'injected_session':
        class NoNetwork:
            def __init__(self):
                self.headers, self.cookies, self.proxies, self.params = {}, {}, {}, {}
                self.auth = None
        session = NoNetwork()
    result = prepare_daily_bridge(directory, identity['symbols'], start=identity['start'],
        end=identity['end'], as_of=identity['as_of'], max_new_archives=0, session=session)
    if not result['bridge_artifact_ready']:
        raise ValueError('Daily bridge must be complete before actual import')
    key = fingerprint({'directory': str(feed.directory), 'source_id': feed.identity})
    with FileLock(str(directory / f'.import-{key}.lock'), timeout=0):
        return _import_verified(directory, feed, result, key)


def _import_verified(directory, feed, result, key):
    if stamp(feed.clock()) < stamp(result['prepared_at']):
        raise ValueError('Actual feed import cannot predate completed daily preparation')
    receipt_path = directory / 'imports' / f'{key}.json'
    if receipt_path.exists():
        record = _read(receipt_path)
        feed.verify()
        row = feed.db.execute('SELECT record FROM batches WHERE id=?', (record['batch'],)).fetchone()
        batch = json.loads(row[0]) if row else {}
        if (record['job_hash'] != result['job_hash'] or record['feed_identity'] != feed.identity
                or record['feed_directory'] != str(feed.directory)
                or record['bars_sha256'] != file_hash(result['bars_path'])
                or batch.get('actual') is not False
                or batch.get('source_ref') != {'path': result['bars_path'], 'sha256': record['bars_sha256']}
                or feed._publication(record['batch'], batch) != record['actual_feed_imported_at']):
            raise ValueError('Daily import receipt differs from completed job')
        return record
    outcome = feed.import_seed(result['bars_path'])
    batch = json.loads(feed.db.execute('SELECT record FROM batches WHERE id=?', (outcome['batch'],)).fetchone()[0])
    published = feed._publication(outcome['batch'], batch)
    if published is None or outcome['new_origins'] != 0:
        raise ValueError('Daily history import was not safely published without origins')
    record = {'job_hash': result['job_hash'], 'feed_identity': feed.identity, 'feed_directory': str(feed.directory),
              'batch': outcome['batch'], 'bars_sha256': file_hash(result['bars_path']),
              'actual_feed_imported_at': published, 'recorded_at': stamp(feed.clock()).isoformat(),
              'new_rows': outcome['new_rows'], 'new_origins': 0, 'data_ready': False}
    _record(receipt_path, record)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['fetch'])
    parser.add_argument('--request', required=True)
    args = parser.parse_args()
    request = _read(args.request)
    remaining = _remaining(request)
    if remaining <= 0:
        raise TimeoutError('Daily worker original deadline consumed')
    watchdog = threading.Timer(remaining, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    result = _network(args.request)
    raise SystemExit(0 if result['status'] == 'downloaded' else 2)


if __name__ == '__main__':
    main()
