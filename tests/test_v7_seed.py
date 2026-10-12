import hashlib
import io
import json
import subprocess
import time
import zipfile
from pathlib import Path

import pandas as pd
import pytest
import requests
from test_v7_run_factor_pool import source_root

from factorlab.discovery import v7_seed as seed


def monthly_zip(month, symbol='BTCUSDT', *, omit=0, member=None):
    start = pd.Timestamp(month + '-01', tz='UTC')
    hours = pd.date_range(start, start + pd.DateOffset(months=1), freq='h', inclusive='left')
    rows = []
    for hour in hours[omit:]:
        t = hour.value // 1_000_000
        rows.append(f'{t},100,101,99,100.2,1,{t + 3599999},100,1,1,100,0')
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        info = zipfile.ZipInfo(member or f'{symbol}-1h-{month}.csv', date_time=(2026, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, '\n'.join(rows))
    return buffer.getvalue()


class Response:
    def __init__(self, url, payload, status=200):
        self.url, self.payload, self.status_code = url, payload, status
        self.headers = {'Content-Length': str(len(payload))}
        self.history = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError('artificial')

    def iter_content(self, size):
        yield self.payload


class Session:
    def __init__(self):
        self.trust_env = True
        self.headers, self.cookies, self.proxies, self.params = {}, {}, {}, {}
        self.auth = None
        self.calls = []
        self.payloads = {}
        self.fail_zip = False

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        name = url.rsplit('/', 1)[1].removesuffix('.CHECKSUM')
        month = name[-11:-4]
        payload = self.payloads.setdefault(name, monthly_zip(month))
        if url.endswith('.CHECKSUM'):
            return Response(url, f'{hashlib.sha256(payload).hexdigest()}  {name}\n'.encode())
        if self.fail_zip:
            raise requests.Timeout('artificial')
        return Response(url, payload)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    monkeypatch.setattr(seed, '_now', lambda: pd.Timestamp('2026-10-04T00:00:00Z'))
    return tmp_path / 'seed', Session()


@pytest.mark.parametrize('kind', ['monthly', 'audit'])
def test_monthly_zip_fixtures_ignore_wall_clock(kind, monkeypatch):
    from test_v7_seed_audit import archive_bytes
    factory = monthly_zip if kind == 'monthly' else archive_bytes
    monkeypatch.setattr(zipfile.time, 'localtime', lambda *args: (2026, 1, 1, 0, 0, 0, 3, 1, 0))
    first = factory('2026-02')
    monkeypatch.setattr(zipfile.time, 'localtime', lambda *args: (2026, 1, 1, 0, 0, 2, 3, 1, 0))
    second = factory('2026-02')
    assert first == second
    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        info, = archive.infolist()
        assert info.date_time == (2026, 1, 1, 0, 0, 0)
        assert info.compress_type == zipfile.ZIP_DEFLATED


def test_bounded_incremental_resume_complete_receipts(setup):
    path, client = setup
    for i in range(8):
        result = seed.prepare_seed(path, session=client)
        assert result['counts']['archive_attempts_consumed'] == i + 1
        assert result['counts']['new_archive_attempts'] == 1
    assert result['status'] == 'complete'
    assert not result['data_ready'] and result['seed_artifact_ready']
    assert result['earliest_receipt_based_fit_cutoff'] == '2026-11-01T00:00:00+00:00'
    assert len(client.calls) == 16
    assert not client.trust_env
    assert all(call[1]['allow_redirects'] is False for call in client.calls)
    frame = pd.read_parquet(result['bars_path'])
    assert frame.open_time.min() == pd.Timestamp('2026-02-01T00:00:00Z')
    assert frame.open_time.max() == pd.Timestamp('2026-09-30T23:00:00Z')
    assert frame.attrs['actual_feed_import_required']
    meta = json.loads(next((path / 'raw').rglob('*.zip.json')).read_text())
    assert meta['ingested_at'] == '2026-10-04T00:00:00+00:00'
    assert meta['actual_feed_imported_at'] is None
    before = Path(result['bars_path']).read_bytes()
    assert seed.prepare_seed(path, session=client)['status'] == 'complete'
    assert len(client.calls) == 16 and Path(result['bars_path']).read_bytes() == before


def test_failure_retry_is_charged_and_revision_is_frozen(setup):
    path, client = setup
    client.fail_zip = True
    result = seed.prepare_seed(path, session=client)
    assert result['status'] == 'attempt_failed'
    selected = seed._read(path / 'raw/BTCUSDT/2026-02/selection.json')
    client.fail_zip = False
    client.payloads['BTCUSDT-1h-2026-02.zip'] = monthly_zip('2026-02', omit=1)
    result = seed.prepare_seed(path, session=client)
    assert result['status'] == 'attempt_failed'
    assert len(client.calls) == 3  # Changed checksum rejected before ZIP GET.
    assert seed._read(path / 'raw/BTCUSDT/2026-02/selection.json') == selected
    result = seed.prepare_seed(path, session=client)
    assert result['counts']['archive_attempts_consumed'] == 3
    result = seed.prepare_seed(path, session=client, max_new_archives=0)
    assert result['status'] == 'budget_exhausted'
    assert len(client.calls) == 4


def test_frozen_asof_source_and_missing_reservation_fail_closed(setup, monkeypatch):
    path, client = setup
    seed.prepare_seed(path, session=client)
    monkeypatch.setattr(seed, '_now', lambda: pd.Timestamp('2026-10-05T00:00:00Z'))
    assert seed.prepare_seed(path, session=client, max_new_archives=0)['status'] == 'pending'
    with pytest.raises(ValueError, match='identity'):
        seed.prepare_seed(path, session=client, as_of='2026-10-05T00:00:00Z')
    next(path.rglob('request.json')).unlink()
    with pytest.raises(ValueError, match='reservation'):
        seed.prepare_seed(path, session=client)


def test_bridge_refuses_data_ready_without_network(setup, monkeypatch):
    path, client = setup
    monkeypatch.setattr(seed, '_now', lambda: pd.Timestamp('2026-10-18T00:00:00Z'))
    result = seed.prepare_seed(path, session=client)
    assert result['status'] == 'requires_daily_bridge' and not result['data_ready']
    assert not client.calls


@pytest.mark.parametrize('kwargs', [{'months': 7}, {'months': 11}, {'months': True},
    {'symbols': ('BTCUSDT',) * 2}, {'symbols': ('../bad',)}, {'max_new_archives': 31}])
def test_configuration_bounds(setup, kwargs):
    path, client = setup
    with pytest.raises(ValueError):
        seed.prepare_seed(path, session=client, **kwargs)
    assert not client.calls


@pytest.mark.parametrize('setting,value', [('auth', ('u', 'p')), ('cookies', {'x': 'secret'}),
    ('proxies', {'https': 'proxy'}), ('headers', {'Authorization': 'secret'})])
def test_private_session_rejected_before_network(setup, setting, value):
    path, client = setup
    setattr(client, setting, value)
    with pytest.raises(ValueError, match='credentials'):
        seed.prepare_seed(path, session=client)
    assert not client.calls


@pytest.mark.parametrize('kwargs', [{'omit': 1}, {'member': '../evil.csv'}, {'member': '/evil.csv'}])
def test_calendar_and_container_reject(kwargs):
    archive = {'symbol': 'BTCUSDT', 'month': '2026-02', 'filename': 'BTCUSDT-1h-2026-02.zip'}
    with pytest.raises(ValueError):
        seed._validate_container(monthly_zip('2026-02', **kwargs), archive)


def test_redirect_and_oversize_response_rejected():
    client = Session()
    response = Response('https://unexpected.example', b'x', 302)
    client.get = lambda *args, **kwargs: response
    with pytest.raises(ValueError, match='redirect'):
        seed._fetch(client, 'https://data.binance.vision/public', time.monotonic() + 1)
    response.status_code = 200
    response.url = 'https://data.binance.vision/public'
    response.headers['Content-Length'] = str(seed.MAX_BYTES + 1)
    with pytest.raises(ValueError, match='oversized'):
        seed._fetch(client, response.url, time.monotonic() + 1)


def test_owned_worker_timeout_is_bounded(setup, monkeypatch):
    path, _ = setup
    request = path / 'attempt/request.json'
    request.parent.mkdir(parents=True)
    calls = []

    class Worker:
        alive = True

        def wait(self, timeout):
            assert timeout == 30
            raise subprocess.TimeoutExpired('artificial', timeout)

        def poll(self):
            return None if self.alive else 124

    worker = Worker()

    def launch(command, **kwargs):
        assert command[2:5] == ['-m', 'factorlab.discovery.v7_seed', 'fetch']
        assert kwargs['shell'] is False
        assert Path(kwargs['env']['TMP']).is_relative_to(path)
        return worker

    def stop(owned):
        assert owned is worker
        calls.append('stop')
        worker.alive = False

    monkeypatch.setattr(seed.subprocess, 'Popen', launch)
    monkeypatch.setattr(seed, 'terminate_process_tree', stop)
    result = seed._download(request, None)
    assert result['error_code'] == 'network_timeout'
    assert calls == ['stop'] and result['hard_deadline_enforced']
