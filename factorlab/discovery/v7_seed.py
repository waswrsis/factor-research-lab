"""Explicit, bounded preparation of immutable public monthly USD-M hour bars.

Nothing downloads on import. A completed seed still needs actual ReceiptArchive
import and current-month polling; assumed historical close times are not receipt
or publication times. Default network work runs in an owned 30-second worker.
Injected sessions are a synchronous test/transport seam, not hard-deadline proof.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import stat
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from uuid import uuid4

import pandas as pd
import requests
from filelock import FileLock

from factorlab.data import _checksum, _source, load_archive, parse_klines_csv
from factorlab.protocol import fingerprint, project_path
from factorlab.time_utils import month_start

from .evidence import file_hash
from .process_policy import isolated_process_kwargs, restricted_environment, terminate_process_tree
from .v7_lifecycle import stamp
from .v7_run import source_files
from .v7_storage import immutable_record

MAX_BYTES = 2 * 1024 * 1024
MAX_ATTEMPTS = 3
ARCHIVE_SECONDS = 30


def _now():
    return pd.Timestamp.now(tz='UTC')


def _record(path, value):
    return immutable_record(path, {'value': value, 'sha256': fingerprint(value)})


def _read(path):
    value = json.loads(Path(path).read_text('utf-8'))
    if set(value) != {'value', 'sha256'} or fingerprint(value['value']) != value['sha256']:
        raise ValueError('Seed receipt integrity failure')
    return value['value']


def _blob(path, payload):
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError('Immutable seed bytes cannot change')
        return
    temporary = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _public_session(session):
    session.trust_env = False
    headers = {str(k).lower() for k in getattr(session, 'headers', {})}
    if (headers & {'authorization', 'proxy-authorization', 'cookie', 'x-api-key', 'x-mbx-apikey'}
            or getattr(session, 'auth', None) or getattr(session, 'cookies', None)
            or getattr(session, 'proxies', None) or getattr(session, 'params', None)):
        raise ValueError('Public seed transport cannot carry credentials, cookies, proxies or parameters')


def _fetch(session, url, deadline):
    _public_session(session)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('Monthly archive time budget consumed')
    with session.get(url, timeout=(min(5., remaining), min(10., remaining)),
                     stream=True, allow_redirects=False) as response:
        if (300 <= response.status_code < 400 or getattr(response, 'history', [])
                or response.url != url):
            raise ValueError('Public seed endpoint cannot redirect or change URL')
        response.raise_for_status()
        length = response.headers.get('Content-Length')
        if length is not None and (not re.fullmatch(r'\d+', str(length)) or int(length) > MAX_BYTES):
            raise ValueError('Invalid or oversized public response length')
        pieces, size = [], 0
        for chunk in response.iter_content(65536):
            if time.monotonic() >= deadline:
                raise TimeoutError('Monthly archive time budget consumed')
            size += len(chunk)
            if size > MAX_BYTES:
                raise ValueError('Public seed response exceeds 2 MiB')
            pieces.append(chunk)
        if length is not None and size != int(length):
            raise ValueError('Public response length does not match body')
        return b''.join(pieces)


def _network(request_path, session=None):
    request_path = project_path(request_path)
    request = _read(request_path)
    manifest = _read(request['manifest'])
    if (request['job_hash'] != fingerprint(manifest) or source_files() != manifest['identity']['source_files']
            or request['archive'] not in manifest['identity']['archives']):
        raise ValueError('Seed network request differs from frozen source job')
    archive = request['archive']
    url, filename = _source(archive['symbol'], archive['month'], 'klines')
    if archive['url'] != url or archive['filename'] != filename:
        raise ValueError('Untrusted seed archive URL')
    own_session = session is None
    client = requests.Session() if own_session else session
    start = time.monotonic()
    reserved_at = stamp(request['reserved_at'])
    wall_remaining = ARCHIVE_SECONDS - max(0., (_now() - reserved_at).total_seconds())
    deadline = start + wall_remaining
    result = {'job_hash': request['job_hash'], 'status': 'failed', 'started_at': _now().isoformat(),
              'hard_deadline_enforced': False,
              'transport_mode': 'parent_limited_worker' if own_session else 'injected_session'}
    try:
        checksum = _fetch(client, url + '.CHECKSUM', deadline)
        _blob(request_path.parent / 'checksum.bin', checksum)
        expected = _checksum(checksum.decode('utf-8'), filename)
        if request['pinned_sha256'] and expected != request['pinned_sha256']:
            raise ValueError('Public archive revision changed after its first frozen checksum')
        payload = _fetch(client, url, deadline)
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError('Public archive SHA-256 mismatch')
        _blob(request_path.parent / 'archive.bin', payload)
        result.update(status='downloaded', sha256=expected, bytes=len(payload), checksum_bytes=len(checksum))
    except (requests.RequestException, TimeoutError, OSError, ValueError, UnicodeError) as exc:
        result.update(error_code='network_timeout' if isinstance(exc, (TimeoutError, requests.Timeout)) else
                      'network_error' if isinstance(exc, requests.RequestException) else 'invalid_or_incomplete_archive',
                      error_type=type(exc).__name__)
    finally:
        if own_session:
            client.close()
        result.update(finished_at=_now().isoformat(), elapsed_seconds=time.monotonic() - start)
        _record(request_path.parent / 'transport.json', result)
    return result


def _download(request_path, session):
    if session is not None:
        return _network(request_path, session)
    environment = restricted_environment()
    temporary = request_path.parent / 'tmp'
    temporary.mkdir(exist_ok=True)
    environment.update(TMP=str(temporary), TEMP=str(temporary), TMPDIR=str(temporary),
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    command = [sys.executable, '-B', '-m', 'factorlab.discovery.v7_seed', 'fetch', '--request', str(request_path)]
    with (request_path.parent / 'worker.log').open('ab') as log:
        worker = subprocess.Popen(command, cwd=project_path('.'), env=environment, shell=False,
                                  stdin=subprocess.DEVNULL, stdout=log, stderr=log, **isolated_process_kwargs())
        try:
            worker.wait(timeout=ARCHIVE_SECONDS)
        except subprocess.TimeoutExpired:
            terminate_process_tree(worker)
            if worker.poll() is None:
                raise RuntimeError('Seed worker deadline cleanup did not terminate the owned process')
            return {'status': 'failed', 'error_code': 'network_timeout', 'error_type': 'WorkerDeadline',
                    'hard_deadline_enforced': True, 'finished_at': _now().isoformat()}
        finally:
            if worker.poll() is None:
                terminate_process_tree(worker)
                if worker.poll() is None:
                    raise RuntimeError('Seed worker cleanup did not terminate the owned process')
    path = request_path.parent / 'transport.json'
    return {**_read(path), 'hard_deadline_enforced': True} if path.exists() else {
        'status': 'failed', 'error_code': 'worker_interrupted', 'error_type': 'MissingTransportReceipt'}


def _validate_container(payload, archive):
    if len(payload) > MAX_BYTES:
        raise ValueError('Monthly ZIP exceeds 2 MiB')
    with zipfile.ZipFile(io.BytesIO(payload)) as zipped:
        members = zipped.infolist()
        if len(members) != 1:
            raise ValueError('Monthly ZIP must contain exactly one CSV')
        member = members[0]
        mode = stat.S_IFMT(member.external_attr >> 16)
        if (member.filename != archive['filename'][:-4] + '.csv' or member.is_dir()
                or member.flag_bits & 1 or mode not in (0, stat.S_IFREG)
                or member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                or member.file_size > MAX_BYTES):
            raise ValueError('Unsafe or oversized monthly CSV container')
        raw = zipped.read(member)  # ZIP CRC checked while decoding; no filesystem extraction.
    if len(raw.splitlines()) > 745:
        raise ValueError('Monthly CSV exceeds 744 hourly rows plus header')
    frame = parse_klines_csv(raw, archive['symbol'])
    start = pd.Timestamp(archive['month'] + '-01', tz='UTC')
    expected = pd.date_range(start, start + pd.DateOffset(months=1), freq='h', inclusive='left')
    if len(frame) > 744 or not pd.DatetimeIndex(frame.open_time).equals(expected):
        raise ValueError('Monthly archive must cover its complete exact UTC hour calendar')
    return frame


def _accept(directory, archive, attempt_dir, job_hash):
    selection_path = directory / 'selection.json'
    checksum_path = attempt_dir / 'checksum.bin'
    if not checksum_path.exists():
        return None
    checksum = checksum_path.read_bytes()
    digest = _checksum(checksum.decode('utf-8'), archive['filename'])
    selection = {'job_hash': job_hash, 'sha256': digest, 'source_filename': archive['filename']}
    _record(selection_path, selection)  # First checksum freezes the only acceptable version.
    transport_path = attempt_dir / 'transport.json'
    if not transport_path.exists():
        return None
    transport = _read(transport_path)
    if transport['job_hash'] != job_hash or transport['status'] != 'downloaded':
        return None
    elapsed = transport.get('elapsed_seconds')
    request = _read(attempt_dir / 'request.json')
    reserved_at = stamp(request['reserved_at'])
    started_at, finished_at = stamp(transport['started_at']), stamp(transport['finished_at'])
    if (type(elapsed) not in (int, float) or not math.isfinite(elapsed)
            or not 0 <= elapsed < ARCHIVE_SECONDS or not reserved_at <= started_at <= finished_at
            or finished_at >= reserved_at + pd.Timedelta(seconds=ARCHIVE_SECONDS)):
        raise ValueError('Seed transport completed outside its frozen deadline')
    supervision = attempt_dir / 'supervision.json'
    if supervision.exists() and _read(supervision)['status'] != 'downloaded':
        raise ValueError('Seed parent rejected this transport; late bytes cannot revive it')
    payload_path = attempt_dir / 'archive.bin'
    if not payload_path.exists():
        return None
    payload = payload_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError('Staged monthly archive hash mismatch')
    frame = _validate_container(payload, archive)
    target = directory / f'{archive["filename"][:-4]}.{digest}.zip'
    metadata = {'schema_version': 1, 'symbol': archive['symbol'], 'month': archive['month'], 'kind': 'klines',
        'source_url': archive['url'], 'source_filename': archive['filename'], 'sha256': digest,
        'bytes': len(payload), 'ingested_at': transport['finished_at'],
        'download_started_at': transport['started_at'],
        'observed_prior_versions': [], 'availability_basis': 'assumed_history_not_observed_receipt',
        'seed_job_hash': job_hash, 'actual_feed_imported_at': None}
    _blob(target.with_suffix('.zip.CHECKSUM'), checksum)
    _blob(target.with_suffix('.zip.json'), json.dumps(metadata, sort_keys=True).encode())
    _blob(target, payload)
    restored = load_archive(target, archive['symbol'])
    if len(restored) != len(frame):
        raise ValueError('Accepted monthly parser round trip changed row support')
    value = {'job_hash': job_hash, 'path': str(target), 'sha256': digest, 'rows': len(frame),
             'metadata_hash': file_hash(target.with_suffix('.zip.json')),
             'checksum_hash': file_hash(target.with_suffix('.zip.CHECKSUM'))}
    _record(directory / 'accepted.json', value)
    return value


def prepare_seed(directory, symbols=('BTCUSDT',), as_of=None, months=8, session=None, *,
                 max_new_archives=1, allow_daily_bridge=False):
    """Prepare one source-frozen seed job; every attempted archive stays charged.

    A supplied session is explicit trusted transport injection. The production
    CLI omits it and uses an owned, time-limited subprocess per archive attempt.
    """
    directory = project_path(directory)
    if (type(months) is not int or not 8 <= months <= 10 or not isinstance(symbols, (tuple, list))
            or not 1 <= len(symbols) <= 3 or len(set(symbols)) != len(symbols)
            or any(not isinstance(s, str) or not re.fullmatch(r'[A-Z0-9]{3,30}', s) for s in symbols)
            or type(max_new_archives) is not int or not 0 <= max_new_archives <= 30
            or type(allow_daily_bridge) is not bool):
        raise ValueError('Seed needs 8..10 completed months, 1..3 symbols and 0..30 new archive attempts')
    if session is not None:
        _public_session(session)
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / '.seed.lock'), timeout=0):
        manifest_path = directory / 'manifest.json'
        previous = _read(manifest_path) if manifest_path.exists() else None
        if previous is None and any(path.name != '.seed.lock' for path in directory.iterdir()):
            raise ValueError('Seed manifest missing from a nonempty job; never reset its budget')
        now = stamp(_now())
        as_of = stamp(previous['identity']['as_of'] if previous and as_of is None else now if as_of is None else as_of)
        if as_of > now:
            raise ValueError('Seed cutoff cannot be a future observation')
        cutoff = month_start(as_of)
        dates = [(cutoff - pd.DateOffset(months=i)).strftime('%Y-%m') for i in range(months, 0, -1)]
        archives = [{'symbol': symbol, 'month': month, 'url': _source(symbol, month, 'klines')[0],
                     'filename': _source(symbol, month, 'klines')[1]} for symbol in symbols for month in dates]
        identity = {'version': 'v7-monthly-seed-v1', 'as_of': as_of.isoformat(), 'symbols': list(symbols),
            'months': months, 'archives': archives, 'source_files': source_files(),
            'archive_byte_ceiling': MAX_BYTES, 'expanded_byte_ceiling': MAX_BYTES,
            'attempts_per_archive': MAX_ATTEMPTS, 'archive_seconds': ARCHIVE_SECONDS,
            'maximum_archives': len(archives), 'http_reservation_ceiling': len(archives) * MAX_ATTEMPTS * 2,
            'transport': 'injected_session' if session is not None else 'bounded_subprocess'}
        if allow_daily_bridge:
            identity['daily_bridge_policy'] = 'prepare_months_then_explicit_daily_bridge_v1'
        manifest = previous or {'identity': identity, 'planned_at': now.isoformat(),
            'historical_published_at_known': False, 'actual_feed_imported_at': None,
            'month_policy': 'complete_UTC_calendar_months_only_no_automatic_window_change'}
        if manifest['identity'] != identity:
            raise ValueError('Frozen seed identity/source/as_of cannot change on resume')
        _record(manifest_path, manifest)
        job_hash = fingerprint(manifest)
        bridge_hours = max(0., (now - cutoff).total_seconds() / 3600)
        base = {'manifest_path': str(manifest_path), 'job_hash': job_hash,
                'bars_path': str(directory / 'bars.parquet'), 'actual_feed_imported_at': None,
                'data_ready': False, 'poll_pages_required': 4, 'bridge_hours_at_this_call': bridge_hours,
                'earliest_receipt_based_fit_cutoff': (month_start(now) + pd.DateOffset(months=1)).isoformat()}
        if bridge_hours > 16 * 24 and not allow_daily_bridge:
            return {**base, 'status': 'requires_daily_bridge', 'reason': 'four_poll_pages_cover_at_most_16_days'}
        accepted, spent, new_attempts, failure = [], 0, 0, None
        for archive in archives:
            raw_dir = directory / 'raw' / archive['symbol'] / archive['month']
            raw_dir.mkdir(parents=True, exist_ok=True)
            attempt_paths = sorted(raw_dir.glob('attempt-*/request.json'))
            reservations = sorted((directory / 'reservations').glob(f'{archive["symbol"]}_{archive["month"]}_*.json'))
            if len(reservations) != len(attempt_paths):
                raise ValueError('Seed attempt reservation journal is missing; consumed budget cannot reset')
            if {p.parent for p in attempt_paths} != set(raw_dir.glob('attempt-*')):
                raise ValueError('Seed attempt reservation is missing; never reset its budget')
            if len(attempt_paths) > MAX_ATTEMPTS:
                raise ValueError('Seed attempt journal exceeds frozen budget')
            spent += len(attempt_paths)
            for number, path in enumerate(attempt_paths):
                request = _read(path)
                if (path.parent.name != f'attempt-{number}' or request['job_hash'] != job_hash
                        or request['archive'] != archive or request['attempt'] != number):
                    raise ValueError('Seed attempt journal does not match frozen archive')
                if _read(reservations[number]) != request:
                    raise ValueError('Seed attempt differs from its independent reservation journal')
            accepted_path = raw_dir / 'accepted.json'
            # Pin the earliest valid checksum even if the parent died before
            # acceptance or the corresponding ZIP transfer failed.
            for path in attempt_paths:
                checksum_path = path.parent / 'checksum.bin'
                if checksum_path.exists():
                    try:
                        digest = _checksum(checksum_path.read_text('utf-8'), archive['filename'])
                    except (ValueError, UnicodeError):
                        continue
                    _record(raw_dir / 'selection.json', {'job_hash': job_hash, 'sha256': digest,
                                                       'source_filename': archive['filename']})
                    break
            if not accepted_path.exists():
                # Recover bytes from a completed transport before reserving any
                # fresh work, including an interrupted parent publication.
                for path in attempt_paths:
                    receipt = path.parent / 'transport.json'
                    if receipt.exists() and _read(receipt)['status'] == 'downloaded':
                        try:
                            _accept(raw_dir, archive, path.parent, job_hash)
                        except (ValueError, OSError, zipfile.BadZipFile):
                            pass  # Already charged rejected content; later retry keeps its pinned hash.
                        if accepted_path.exists():
                            break
            if accepted_path.exists():
                record = _read(accepted_path)
                target = project_path(record['path'])
                if (record['job_hash'] != job_hash or target.parent != raw_dir
                        or file_hash(target) != record['sha256']
                        or file_hash(target.with_suffix('.zip.json')) != record['metadata_hash']
                        or file_hash(target.with_suffix('.zip.CHECKSUM')) != record['checksum_hash']):
                    raise ValueError('Frozen accepted seed archive integrity failure')
                if len(load_archive(target, archive['symbol'])) != record['rows']:
                    raise ValueError('Seed row receipt differs from parsed archive')
                accepted.append(record)
                continue
            if len(attempt_paths) == MAX_ATTEMPTS:
                failure = {'status': 'budget_exhausted', 'archive': archive}
                continue
            if new_attempts >= max_new_archives:
                continue
            attempt_dir = raw_dir / f'attempt-{len(attempt_paths)}'
            selection = _read(raw_dir / 'selection.json') if (raw_dir / 'selection.json').exists() else None
            request = {'manifest': str(manifest_path), 'job_hash': job_hash, 'archive': archive,
                'attempt': len(attempt_paths), 'reserved_at': _now().isoformat(),
                'pinned_sha256': selection['sha256'] if selection else None}
            reservation = directory / 'reservations' / f'{archive["symbol"]}_{archive["month"]}_{len(attempt_paths)}.json'
            _record(reservation, request)
            attempt_dir.mkdir(exist_ok=True)
            _record(attempt_dir / 'request.json', request)
            spent += 1
            new_attempts += 1
            result = {}
            try:
                result = _download(attempt_dir / 'request.json', session)
                _record(attempt_dir / 'supervision.json', result)
                record = _accept(raw_dir, archive, attempt_dir, job_hash)
                if result['status'] != 'downloaded' or record is None:
                    raise ValueError(result.get('error_code', 'incomplete_transport'))
                accepted.append(record)
                outcome = {'status': 'accepted', 'archive_hash': record['sha256']}
            except (OSError, ValueError, requests.RequestException, zipfile.BadZipFile) as exc:
                outcome = {'status': 'attempt_failed', 'error_type': type(exc).__name__,
                           'error_code': result.get('error_code', 'invalid_or_incomplete_archive')}
                failure = {**outcome, 'archive': archive}
            _record(attempt_dir / 'outcome.json', {**outcome, 'finished_at': _now().isoformat(), 'job_hash': job_hash})
        counts = {'archives_required': len(archives), 'archives_accepted': len(accepted),
                  'archive_attempts_consumed': spent, 'reserved_http_requests': 2 * spent,
                  'new_archive_attempts': new_attempts}
        if len(accepted) != len(archives):
            return {**base, 'status': failure['status'] if failure else 'pending', 'counts': counts, 'failure': failure}
        frames = [load_archive(record['path'], archive['symbol']) for archive, record in zip(archives, accepted, strict=True)]
        bars = pd.concat(frames, ignore_index=True).sort_values(['symbol', 'open_time']).reset_index(drop=True)
        bars.attrs = {'seed_job_hash': job_hash, 'availability_basis': 'assumed_live_delay',
                      'historical_published_at_known': False, 'actual_feed_import_required': True}
        complete_path, bars_path = directory / 'complete.json', directory / 'bars.parquet'
        if complete_path.exists():
            complete = _read(complete_path)
            if complete['job_hash'] != job_hash or file_hash(bars_path) != complete['bars_sha256']:
                raise ValueError('Completed seed artifact integrity failure')
        else:
            buffer = io.BytesIO()
            bars.to_parquet(buffer, index=False)
            _blob(bars_path, buffer.getvalue())
            complete = {'job_hash': job_hash, 'bars_sha256': file_hash(bars_path), 'rows': len(bars),
                        'prepared_at': _now().isoformat(), 'archives': accepted,
                        'actual_feed_imported_at': None, 'historical_published_at_known': False}
            _record(complete_path, complete)
        bridge = ({'status': 'requires_daily_bridge', 'daily_bridge_start': cutoff.isoformat(),
                   'daily_bridge_end': (now.floor('D') - pd.Timedelta(days=15)).isoformat(),
                   'reason': 'monthly_seed_complete_rest_tail_requires_explicit_daily_archive_import'}
                  if bridge_hours > 16 * 24 else {'status': 'complete'})
        return {**base, **bridge, 'counts': counts, 'seed_artifact_ready': True,
                'completion_receipt': str(complete_path), 'prepared_at': complete['prepared_at'],
                'next_action': 'actual_receipt_archive_import_then_poll_four_pages_no_historical_prediction'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['fetch'])
    parser.add_argument('--request', required=True)
    args = parser.parse_args()
    # The child retains its own deadline if its supervising parent disappears.
    # There are no grandchildren or publication side effects beyond this job.
    deadline = threading.Timer(ARCHIVE_SECONDS, lambda: os._exit(124))
    deadline.daemon = True
    deadline.start()
    try:
        result = _network(args.request)
    finally:
        deadline.cancel()
    raise SystemExit(0 if result['status'] == 'downloaded' else 2)


if __name__ == '__main__':
    main()
