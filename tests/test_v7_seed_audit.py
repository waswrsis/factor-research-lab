"""Independent seed ingestion audit with artificial ZIPs and fake transports only."""
import hashlib
import io
import subprocess
import zipfile

import pandas as pd
import pytest
import requests

from factorlab.discovery import v7_seed as seed
from factorlab.discovery.v7_feed import ReceiptArchive


@pytest.fixture
def fixed(monkeypatch):
    clock = [pd.Timestamp('2026-10-04T12:00:00Z')]
    monkeypatch.setattr(seed, '_now', lambda: clock[0])
    monkeypatch.setattr(seed, 'source_files', lambda: {'synthetic_seed_source.py': 'frozen'})
    return clock


def archive_bytes(month='2026-02', *, value=100., damage=None):
    start = pd.Timestamp(month + '-01', tz='UTC')
    dates = list(pd.date_range(start, start + pd.DateOffset(months=1), freq='h', inclusive='left'))
    if damage == 'missing':
        dates.pop(5)
    elif damage == 'duplicate':
        dates[5] = dates[4]
    elif damage == 'out_of_month':
        dates[-1] += pd.Timedelta(hours=1)
    lines = []
    for i, stamp in enumerate(dates):
        opened = int(stamp.timestamp() * 1000)
        if damage == 'microseconds':
            opened *= 1000
        low = value + 2 if damage == 'OHLC' and i == 0 else value - 1
        lines.append(f'{opened},{value},{value+1},{low},{value},1,{opened+3599999},1,1,1,1,0')
    buffer = io.BytesIO()
    filename = f'BTCUSDT-1h-{month}.csv'
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        info = zipfile.ZipInfo(filename, date_time=(2026, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, '\n'.join(lines))
    return buffer.getvalue()


class Response:
    status_code, history = 200, []

    def __init__(self, url, body):
        self.url, self.body = url, body
        self.headers = {'Content-Length': str(len(body))}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, size):
        yield self.body


class Session:
    def __init__(self, *, value=100., interrupt_archive=False, fail_checksum=False):
        self.headers, self.auth, self.cookies, self.proxies, self.params = {}, None, {}, {}, {}
        self.trust_env = True
        self.calls = []
        self.value, self.interrupt_archive, self.fail_checksum = value, interrupt_archive, fail_checksum

    def get(self, url, **kwargs):
        assert self.trust_env is False
        assert kwargs['stream'] and kwargs['allow_redirects'] is False
        self.calls.append(url)
        filename = url.rsplit('/', 1)[-1].removesuffix('.CHECKSUM')
        month = filename[-11:-4]
        payload = archive_bytes(month, value=self.value)
        if url.endswith('.CHECKSUM'):
            if self.fail_checksum:
                raise requests.ConnectionError('synthetic unavailable')
            body = f'{hashlib.sha256(payload).hexdigest()}  {filename}'.encode()
        else:
            if self.interrupt_archive:
                raise KeyboardInterrupt('synthetic death after checksum persistence')
            body = payload
        return Response(url, body)


def test_checksum_persisted_before_interruption_remains_pinned(tmp_path, fixed):
    first = Session(interrupt_archive=True)
    with pytest.raises(KeyboardInterrupt):
        seed.prepare_seed(tmp_path / 'seed', session=first)
    original = hashlib.sha256(archive_bytes()).hexdigest()
    revised = Session(value=101.)
    result = seed.prepare_seed(tmp_path / 'seed', session=revised)
    raw = tmp_path / 'seed' / 'raw' / 'BTCUSDT' / '2026-02'
    assert seed._read(raw / 'selection.json')['sha256'] == original
    assert not (raw / 'accepted.json').exists()
    assert result['counts']['archive_attempts_consumed'] == 2


def test_removing_failed_attempt_directory_cannot_refund_consumption(tmp_path, fixed):
    directory = tmp_path / 'seed'
    first = seed.prepare_seed(directory, session=Session(fail_checksum=True))
    assert first['counts']['archive_attempts_consumed'] == 1
    attempt = directory / 'raw' / 'BTCUSDT' / '2026-02' / 'attempt-0'
    assert attempt.resolve().is_relative_to(tmp_path.resolve())
    for path in attempt.iterdir():
        assert path.is_file()
        path.unlink()
    attempt.rmdir()
    with pytest.raises(ValueError, match='budget|journal|reservation|attempt|missing|consumed'):
        seed.prepare_seed(directory, session=Session(fail_checksum=True))


def test_parent_deadline_failure_cannot_publish_downloaded_late_bytes(tmp_path, fixed, monkeypatch):
    def late(request_path, session):
        request = seed._read(request_path)
        archive = request['archive']
        payload = archive_bytes(archive['month'])
        checksum = hashlib.sha256(payload).hexdigest()
        seed._blob(request_path.parent / 'checksum.bin', f'{checksum}  {archive["filename"]}'.encode())
        seed._blob(request_path.parent / 'archive.bin', payload)
        seed._record(request_path.parent / 'transport.json', {
            'job_hash': request['job_hash'], 'status': 'downloaded', 'sha256': checksum,
            'started_at': (fixed[0] - pd.Timedelta(seconds=31)).isoformat(),
            'finished_at': fixed[0].isoformat(), 'elapsed_seconds': 31.,
            'hard_deadline_enforced': False})
        return {'status': 'failed', 'error_code': 'network_timeout', 'hard_deadline_enforced': True}

    monkeypatch.setattr(seed, '_download', late)
    directory = tmp_path / 'seed'
    seed.prepare_seed(directory, session=Session())
    accepted = directory / 'raw' / 'BTCUSDT' / '2026-02' / 'accepted.json'
    assert not accepted.exists()
    replay = seed.prepare_seed(directory, session=Session(), max_new_archives=0)
    assert replay['counts']['archives_accepted'] == 0


def test_cleanup_exception_cannot_leave_late_worker_bytes_recoverable(tmp_path, fixed, monkeypatch):
    def orphaned(request_path, session):
        request = seed._read(request_path)
        archive = request['archive']
        payload = archive_bytes(archive['month'])
        digest = hashlib.sha256(payload).hexdigest()
        seed._blob(request_path.parent / 'checksum.bin', f'{digest}  {archive["filename"]}'.encode())
        seed._blob(request_path.parent / 'archive.bin', payload)
        # Child startup took 2 seconds. Its own elapsed29 is below30, but its
        # result completes after the parent's already expired whole-attempt cap.
        reserved = pd.Timestamp(request['reserved_at'])
        fixed[0] = reserved + pd.Timedelta(seconds=31)
        seed._record(request_path.parent / 'transport.json', {
            'job_hash': request['job_hash'], 'status': 'downloaded', 'sha256': digest,
            'started_at': (reserved + pd.Timedelta(seconds=2)).isoformat(),
            'finished_at': fixed[0].isoformat(), 'elapsed_seconds': 29.,
            'hard_deadline_enforced': False})
        raise RuntimeError('synthetic owned worker cleanup failed')

    directory = tmp_path / 'seed'
    monkeypatch.setattr(seed, '_download', orphaned)
    with pytest.raises(RuntimeError, match='cleanup'):
        seed.prepare_seed(directory, session=Session())
    result = seed.prepare_seed(directory, session=Session(), max_new_archives=0)
    assert result['counts']['archives_accepted'] == 0
    assert not (directory / 'raw' / 'BTCUSDT' / '2026-02' / 'accepted.json').exists()


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'out_of_month', 'OHLC', 'microseconds'])
def test_exact_calendar_prices_and_time_units_are_required(damage):
    url, filename = seed._source('BTCUSDT', '2026-02', 'klines')
    archive = {'symbol': 'BTCUSDT', 'month': '2026-02', 'url': url, 'filename': filename}
    with pytest.raises((ValueError, OverflowError)):
        seed._validate_container(archive_bytes(damage=damage), archive)


@pytest.mark.parametrize('field,value', [('headers', {'Authorization': 'synthetic'}),
    ('headers', {'x-MBX-ApiKey': 'synthetic'}), ('cookies', {'session': 'synthetic'}),
    ('auth', ('synthetic', 'synthetic')), ('proxies', {'https': 'synthetic'})])
def test_public_transport_rejects_credentials_before_request(tmp_path, fixed, field, value):
    session = Session()
    setattr(session, field, value)
    with pytest.raises(ValueError, match='credentials'):
        seed.prepare_seed(tmp_path / 'seed', session=session)
    assert not session.calls


def test_frozen_clock_source_and_completed_months_are_preserved(tmp_path, fixed, monkeypatch):
    directory = tmp_path / 'seed'
    session = Session()
    first = seed.prepare_seed(directory, session=session, max_new_archives=0)
    manifest = seed._read(directory / 'manifest.json')
    assert manifest['identity']['as_of'] == fixed[0].isoformat()
    assert [a['month'] for a in manifest['identity']['archives']] == [f'2026-{m:02}' for m in range(2, 10)]
    fixed[0] += pd.Timedelta(days=1)
    second = seed.prepare_seed(directory, session=session, max_new_archives=0)
    assert first['job_hash'] == second['job_hash']
    assert second['earliest_receipt_based_fit_cutoff'] == '2026-11-01T00:00:00+00:00'
    assert not second['data_ready'] and second['actual_feed_imported_at'] is None
    monkeypatch.setattr(seed, 'source_files', lambda: {'synthetic_seed_source.py': 'changed'})
    with pytest.raises(ValueError, match='source|identity'):
        seed.prepare_seed(directory, session=session, max_new_archives=0)
    assert not session.calls


def test_missing_manifest_cannot_rebind_old_seed_directory(tmp_path, fixed):
    directory = tmp_path / 'seed'
    seed.prepare_seed(directory, session=Session(), max_new_archives=0)
    (directory / 'manifest.json').unlink()
    with pytest.raises(ValueError, match='manifest missing|budget'):
        seed.prepare_seed(directory, session=Session(), max_new_archives=0)


def test_real_requests_preparation_never_loads_netrc_or_environment_auth(tmp_path, fixed, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Public request consulted environment credentials')

    monkeypatch.setattr(requests.sessions, 'get_netrc_auth', forbidden)
    monkeypatch.setattr(requests.sessions, 'get_environ_proxies', forbidden)
    sent = []
    fake = Session()
    with requests.Session() as session:
        def send(prepared, **kwargs):
            sent.append(prepared)
            assert not {'authorization', 'proxy-authorization', 'cookie', 'x-mbx-apikey'} & {
                str(k).lower() for k in prepared.headers}
            return fake.get(prepared.url, stream=True, allow_redirects=False)

        fake.trust_env = False
        monkeypatch.setattr(session, 'send', send)
        result = seed.prepare_seed(tmp_path / 'netrc', session=session)
    assert result['counts']['archives_accepted'] == 1 and len(sent) == 2


def test_current_partial_month_bridge_boundary_is_honest(tmp_path, fixed):
    directory = tmp_path / 'seed'
    fixed[0] = pd.Timestamp('2026-10-17T00:00:00Z')
    session = Session()
    first = seed.prepare_seed(directory, session=session, max_new_archives=0)
    assert first['status'] == 'pending' and first['bridge_hours_at_this_call'] == 384
    fixed[0] += pd.Timedelta(seconds=1)
    second = seed.prepare_seed(directory, session=session, max_new_archives=0)
    assert second['status'] == 'requires_daily_bridge' and not second['data_ready']
    assert first['job_hash'] == second['job_hash'] and not session.calls


def test_completed_seed_is_not_an_old_available_receipt_or_forecast_origin(tmp_path, fixed):
    directory = tmp_path / 'seed'
    result = seed.prepare_seed(directory, session=Session(), max_new_archives=8)
    assert result['status'] == 'complete' and result['seed_artifact_ready']
    assert not result['data_ready'] and result['actual_feed_imported_at'] is None
    assert result['counts']['archives_accepted'] == 8
    complete = seed._read(directory / 'complete.json')
    assert complete['prepared_at'] == fixed[0].isoformat()
    assert complete['historical_published_at_known'] is False
    feed = ReceiptArchive(tmp_path / 'actual_feed', clock=lambda: fixed[0])
    try:
        imported = feed.import_seed(result['bars_path'])
        assert imported['new_origins'] == 0
        with pytest.raises(ValueError, match='No source received'):
            feed.snapshot('2026-10-01T00:00Z')
        visible = feed.snapshot(fixed[0])
        assert len(visible) == complete['rows']
        assert not visible.prediction_origin.any()
    finally:
        feed.close()


def test_failed_owned_cleanup_cannot_claim_enforced_hard_deadline(tmp_path, monkeypatch):
    request = tmp_path / 'request.json'
    request.write_text('{}')

    class Worker:
        pid = 42

        def wait(self, timeout):
            raise subprocess.TimeoutExpired('synthetic-worker', timeout)

        def poll(self):
            return None

    monkeypatch.setattr(seed.subprocess, 'Popen', lambda *a, **kw: Worker())
    monkeypatch.setattr(seed, 'terminate_process_tree', lambda worker: {
        'status': 'failed', 'tree_command_completed': False, 'escaped_descendants_verified': False})
    with pytest.raises((RuntimeError, ValueError), match='terminat|cleanup|worker|deadline'):
        seed._download(request, None)
