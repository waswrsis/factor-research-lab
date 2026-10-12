import hashlib
import io
import json
import sqlite3
import subprocess
import zipfile
from pathlib import Path

import pandas as pd
import pytest
import requests
from test_v7_seed import Response

from factorlab.discovery import v7_daily as daily
from factorlab.discovery.v7_feed import ReceiptArchive


def daily_zip(day='2026-10-01', symbol='BTCUSDT', *, omit=0, member=None,
              wrong_prices=False, multiplier=1, zip_timestamp=(2026, 1, 1, 0, 0, 0)):
    rows = []
    for hour in pd.date_range(pd.Timestamp(day, tz='UTC'), periods=24, freq='h')[omit:]:
        opened = hour.value // 1_000_000 * multiplier
        high = 98 if wrong_prices else 101
        rows.append(f'{opened},100,{high},99,100.2,1,{opened + 3599999},100,1,1,100,0')
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        info = zipfile.ZipInfo(member or f'{symbol}-1h-{day}.csv', date_time=zip_timestamp)
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, '\n'.join(rows))
    return buffer.getvalue()


class Session:
    def __init__(self):
        self.headers, self.cookies, self.proxies, self.params = {}, {}, {}, {}
        self.auth = None
        self.trust_env = True
        self.calls, self.payloads = [], {}
        self.fail_zip = False

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        name = url.rsplit('/', 1)[1].removesuffix('.CHECKSUM')
        day, symbol = name[-14:-4], name.split('-')[0]
        payload = self.payloads.setdefault(name, daily_zip(day, symbol))
        if url.endswith('.CHECKSUM'):
            return Response(url, f'{hashlib.sha256(payload).hexdigest()}  {name}\n'.encode())
        if self.fail_zip:
            raise requests.Timeout('artificial')
        return Response(url, payload)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    clock = [pd.Timestamp('2026-10-20T12:00:00Z')]
    monkeypatch.setattr(daily, '_now', lambda: clock[0])
    monkeypatch.setattr(daily, 'source_files', lambda: {'synthetic.py': 'frozen'})
    return tmp_path / 'daily', Session(), clock


def prepare(path, session, **kwargs):
    return daily.prepare_daily_bridge(path, start='2026-10-01', end='2026-10-02', session=session, **kwargs)


def test_incremental_frozen_window_and_bounded_requests(setup):
    path, session, clock = setup
    for i in range(4):
        result = daily.prepare_daily_bridge(path, start='2026-10-01', session=session)
        assert result['counts']['archive_attempts_consumed'] == i + 1
        assert result['counts']['new_archive_attempts'] == 1
    assert result['status'] == 'complete' and result['bridge_artifact_ready']
    assert result['bridge_end'] == '2026-10-05T00:00:00+00:00'
    assert not result['data_ready'] and not result['requires_new_bridge']
    assert len(session.calls) == 8 and session.trust_env is False
    assert all(not kwargs['allow_redirects'] for _, kwargs in session.calls)
    bars = pd.read_parquet(result['bars_path'])
    assert len(bars) == 96 and bars.attrs['actual_feed_import_required']
    clock[0] += pd.Timedelta(days=2)
    resumed = daily.prepare_daily_bridge(path, start='2026-10-01', session=session)
    assert resumed['job_hash'] == result['job_hash'] and resumed['requires_new_bridge']
    assert len(session.calls) == 8 and resumed['counts']['new_archive_attempts'] == 0


def test_actual_import_is_later_training_only_and_idempotent(setup):
    path, session, clock = setup
    result = prepare(path, session)
    prepared = clock[0]
    clock[0] += pd.Timedelta(hours=1)
    feed = ReceiptArchive(path.parent / 'feed', clock=lambda: clock[0])
    try:
        receipt = daily.import_daily_bridge(path, feed)
        assert receipt['new_rows'] == 24 and receipt['new_origins'] == 0
        assert pd.Timestamp(receipt['actual_feed_imported_at']) == clock[0]
        with pytest.raises(ValueError, match='No source'):
            feed.snapshot(prepared)
        frame = feed.snapshot(clock[0])
        assert frame.received_at.isna().all() and not frame.prediction_origin.any()
        assert (frame.imported_at == clock[0]).all()
        assert set(frame.availability_basis) == {'assumed_archive'}
        clock[0] += pd.Timedelta(hours=1)
        assert daily.import_daily_bridge(path, feed) == receipt
        assert feed.db.execute('SELECT COUNT(*) FROM batches').fetchone()[0] == 1
        assert daily._read(Path(result['completion_receipt']))['actual_feed_imported_at'] is None
    finally:
        feed.close()


def test_checksum_first_version_survives_failed_zip_and_revised_retry(setup):
    path, session, _ = setup
    session.fail_zip = True
    assert prepare(path, session)['status'] == 'attempt_failed'
    pinned = daily._read(path / 'raw/BTCUSDT/2026-10-01/selection.json')
    session.fail_zip = False
    session.payloads['BTCUSDT-1h-2026-10-01.zip'] = daily_zip(omit=1)
    assert prepare(path, session)['status'] == 'attempt_failed'
    assert len(session.calls) == 3
    session.payloads['BTCUSDT-1h-2026-10-01.zip'] = daily_zip()
    result = prepare(path, session)
    assert result['status'] == 'complete' and result['counts']['archive_attempts_consumed'] == 3
    assert daily._read(path / 'raw/BTCUSDT/2026-10-01/selection.json') == pinned
    assert len(session.calls) == 5


def test_daily_zip_fixture_ignores_wall_clock(monkeypatch):
    monkeypatch.setattr(zipfile.time, 'localtime', lambda *args: (2026, 1, 1, 0, 0, 0, 3, 1, 0))
    first = daily_zip()
    monkeypatch.setattr(zipfile.time, 'localtime', lambda *args: (2026, 1, 1, 0, 0, 2, 3, 1, 0))
    second = daily_zip()
    assert first == second
    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        info, = archive.infolist()
        assert info.date_time == (2026, 1, 1, 0, 0, 0)
        assert info.compress_type == zipfile.ZIP_DEFLATED


def test_checksum_rejects_only_container_timestamp_revision(setup):
    path, session, _ = setup
    original = daily_zip()
    revised = daily_zip(zip_timestamp=(2026, 1, 1, 0, 0, 2))
    assert original != revised
    with zipfile.ZipFile(io.BytesIO(original)) as left, zipfile.ZipFile(io.BytesIO(revised)) as right:
        assert left.read(left.namelist()[0]) == right.read(right.namelist()[0])
    name = 'BTCUSDT-1h-2026-10-01.zip'
    session.payloads[name] = original
    session.fail_zip = True
    assert prepare(path, session)['status'] == 'attempt_failed'
    pinned = daily._read(path / 'raw/BTCUSDT/2026-10-01/selection.json')
    session.fail_zip = False
    session.payloads[name] = revised
    assert prepare(path, session)['status'] == 'attempt_failed'
    assert daily._read(path / 'raw/BTCUSDT/2026-10-01/selection.json') == pinned
    assert not (path / 'raw/BTCUSDT/2026-10-01/attempt-1/archive.bin').exists()
    session.payloads[name] = original
    result = prepare(path, session)
    assert result['status'] == 'complete'
    assert result['counts']['archive_attempts_consumed'] == 3 and len(session.calls) == 5


def test_failure_budget_never_refunded(setup):
    path, session, _ = setup
    session.fail_zip = True
    for i in range(3):
        assert prepare(path, session)['counts']['archive_attempts_consumed'] == i + 1
    result = prepare(path, session)
    assert result['status'] == 'budget_exhausted' and len(session.calls) == 6


@pytest.mark.parametrize('item', ['manifest.json', 'budget.sqlite', 'reservations/00000.json',
    'raw/BTCUSDT/2026-10-01/attempt-0/request.json'])
def test_missing_identity_or_budget_proof_is_not_recreated(setup, item):
    path, session, _ = setup
    prepare(path, session)
    (path / item).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare(path, session)
    assert len(session.calls) == 2


def test_deleting_journal_rows_cannot_refund_reserved_attempt(setup):
    path, session, _ = setup
    session.fail_zip = True
    prepare(path, session)
    with sqlite3.connect(path / 'budget.sqlite') as db:
        db.execute('DELETE FROM attempts')
    with pytest.raises(ValueError, match='reservations'):
        prepare(path, session)
    assert len(session.calls) == 2


@pytest.mark.parametrize('kwargs', [{'end': '2026-10-03'}, {'as_of': '2026-10-19T00:00:00Z'},
    {'symbols': ('ETHUSDT',)}])
def test_resume_identity_cannot_change(setup, kwargs):
    path, session, _ = setup
    prepare(path, session)
    options = {'start': '2026-10-01', 'end': '2026-10-02', 'session': session, **kwargs}
    with pytest.raises(ValueError, match='identity'):
        daily.prepare_daily_bridge(path, **options)


def test_changed_source_refuses_resume_and_import(setup, monkeypatch):
    path, session, clock = setup
    prepare(path, session)
    monkeypatch.setattr(daily, 'source_files', lambda: {'synthetic.py': 'changed'})
    with pytest.raises(ValueError, match='identity'):
        prepare(path, session)
    feed = ReceiptArchive(path.parent / 'feed', clock=lambda: clock[0])
    try:
        with pytest.raises(ValueError, match='source'):
            daily.import_daily_bridge(path, feed)
        assert feed.db.execute('SELECT COUNT(*) FROM batches').fetchone()[0] == 0
    finally:
        feed.close()


def test_import_cannot_use_an_earlier_feed_clock_than_completed_preparation(setup):
    path, session, clock = setup
    prepare(path, session)
    feed = ReceiptArchive(path.parent / 'feed', clock=lambda: clock[0] - pd.Timedelta(hours=1))
    try:
        with pytest.raises(ValueError, match='predate'):
            daily.import_daily_bridge(path, feed)
        assert feed.db.execute('SELECT COUNT(*) FROM bars').fetchone()[0] == 0
    finally:
        feed.close()


@pytest.mark.parametrize('kwargs', [{'omit': 1}, {'member': '../evil.csv'}, {'wrong_prices': True},
    {'multiplier': 1000}, {'day': '2026-10-02'}])
def test_invalid_daily_container_and_exact_calendar(kwargs):
    with pytest.raises((ValueError, pd.errors.OutOfBoundsDatetime)):
        daily._validate(daily_zip(**kwargs), daily._archive('BTCUSDT', '2026-10-01'))


@pytest.mark.parametrize('kwargs', [{'start': '2025-01-01'}, {'start': '2026-10-01T01:00:00Z'},
    {'end': '2026-10-21'}, {'max_new_archives': 4}, {'max_new_archives': True},
    {'symbols': ('BTCUSDT', 'BTCUSDT')}])
def test_request_bounds_before_network(setup, kwargs):
    path, session, _ = setup
    with pytest.raises(ValueError):
        daily.prepare_daily_bridge(path, **{'start': '2026-10-01', 'end': '2026-10-02',
                                           'session': session, **kwargs})
    assert not session.calls


@pytest.mark.parametrize('setting,value', [('auth', ('user', 'password')),
    ('headers', {'Authorization': 'secret'}), ('cookies', {'secret': 'x'})])
def test_credentials_rejected_before_network(setup, setting, value):
    path, session, _ = setup
    setattr(session, setting, value)
    with pytest.raises(ValueError, match='credentials'):
        prepare(path, session)
    assert not session.calls


@pytest.mark.parametrize('target', ['bars.parquet', 'raw/BTCUSDT/2026-10-01/attempt-0/archive.bin',
    'raw/BTCUSDT/2026-10-01/attempt-0/transport.json'])
def test_tampered_completed_job_cannot_import(setup, target):
    path, session, clock = setup
    prepare(path, session)
    (path / target).write_bytes(b'corruption')
    feed = ReceiptArchive(path.parent / 'feed', clock=lambda: clock[0])
    try:
        with pytest.raises((ValueError, OSError)):
            daily.import_daily_bridge(path, feed)
        assert feed.db.execute('SELECT COUNT(*) FROM bars').fetchone()[0] == 0
    finally:
        feed.close()


def test_downloaded_bytes_survive_parent_crash_without_new_network(setup, monkeypatch):
    path, session, clock = setup
    original = daily._download

    def crash(request, session):
        original(request, session)
        raise RuntimeError('artificial parent crash before acceptance')

    monkeypatch.setattr(daily, '_download', crash)
    with pytest.raises(RuntimeError, match='artificial'):
        prepare(path, session)
    clock[0] += pd.Timedelta(hours=1)
    result = prepare(path, session, max_new_archives=0)
    assert result['status'] == 'complete' and len(session.calls) == 2
    accepted = daily._read(path / 'raw/BTCUSDT/2026-10-01/accepted.json')
    assert pd.Timestamp(accepted['accepted_at']) == clock[0]
    assert pd.Timestamp(accepted['received_at']) < clock[0]


def test_known_failed_supervision_cannot_accept_late_bytes(setup, monkeypatch):
    path, session, _ = setup

    def failed(request, session):
        daily._network(request, session)
        return {'status': 'failed', 'hard_deadline_enforced': False}

    monkeypatch.setattr(daily, '_download', failed)
    assert prepare(path, session)['status'] == 'attempt_failed'
    assert prepare(path, session, max_new_archives=0)['status'] == 'pending'
    assert not (path / 'raw/BTCUSDT/2026-10-01/accepted.json').exists()


def test_late_transport_rejected_even_with_downloaded_status(setup):
    path, session, clock = setup
    original = session.get

    def slow(url, **kwargs):
        response = original(url, **kwargs)
        if not url.endswith('.CHECKSUM'):
            clock[0] += pd.Timedelta(seconds=29)
        return response

    session.get = slow
    assert prepare(path, session)['status'] == 'attempt_failed'
    assert not (path / 'raw/BTCUSDT/2026-10-01/accepted.json').exists()


def test_owned_timeout_kills_only_retained_child_and_does_not_accept(setup, monkeypatch):
    path, _, _ = setup
    instances = []

    class Child:
        def __init__(self, *args, **kwargs):
            self.returncode, self.killed = None, False
            instances.append(self)

        def wait(self, timeout):
            if not self.killed:
                raise subprocess.TimeoutExpired('owned synthetic child', timeout)
            self.returncode = -1

        def kill(self):
            self.killed = True

        def poll(self):
            return self.returncode

    monkeypatch.setattr(daily.subprocess, 'Popen', Child)
    result = prepare(path, None)
    assert result['status'] == 'attempt_failed' and instances[0].killed
    supervision = daily._read(path / 'raw/BTCUSDT/2026-10-01/attempt-0/supervision.json')
    assert supervision['status'] == 'failed' and supervision['hard_deadline_enforced']


def test_import_receipt_is_checked_against_real_feed_batch(setup):
    path, session, clock = setup
    prepare(path, session)
    feed = ReceiptArchive(path.parent / 'feed', clock=lambda: clock[0])
    try:
        record = daily.import_daily_bridge(path, feed)
        receipt = next((path / 'imports').glob('*.json'))
        changed = {**record, 'batch': 'different'}
        receipt.write_text(json.dumps({'value': changed, 'sha256': daily.fingerprint(changed)}))
        with pytest.raises(ValueError, match='receipt differs'):
            daily.import_daily_bridge(path, feed)
    finally:
        feed.close()


def test_monthly_daily_rest_chain_has_complete_hours_without_old_origins(setup, monkeypatch):
    from test_v7_feed_audit import Response as FeedResponse
    from test_v7_feed_audit import Session as FeedSession
    from test_v7_feed_audit import kline
    from test_v7_seed import Session as MonthlySession

    from factorlab.discovery import v7_seed as seed

    path, session, clock = setup
    monkeypatch.setattr(seed, '_now', lambda: clock[0])
    monkeypatch.setattr(seed, 'source_files', daily.source_files)
    monthly_session = MonthlySession()
    monthly = seed.prepare_seed(path.parent / 'monthly', session=monthly_session,
                               max_new_archives=8, allow_daily_bridge=True)
    assert monthly['status'] == 'requires_daily_bridge' and monthly['seed_artifact_ready']
    for _ in range(4):
        result = daily.prepare_daily_bridge(path, start=monthly['daily_bridge_start'],
            end=monthly['daily_bridge_end'], session=session)
    assert result['bridge_artifact_ready'] and len(session.calls) == 8
    prepared_at = clock[0]
    clock[0] += pd.Timedelta(seconds=1)
    feed = ReceiptArchive(path.parent / 'feed', clock=lambda: clock[0])
    try:
        assert feed.import_seed(monthly['bars_path'])['new_origins'] == 0
        assert daily.import_daily_bridge(path, feed)['new_origins'] == 0
        with pytest.raises(ValueError, match='No source'):
            feed.snapshot(prepared_at)
        end = clock[0].floor('h')
        pages = [[kline(hour) for hour in pd.date_range(end - pd.Timedelta(hours=96 * page),
                  periods=96, freq='h')] for page in range(1, 5)]
        replies = feed.poll(pages=4, session=FeedSession([FeedResponse(rows) for rows in pages]))
        assert len(replies) == 4 and sum(reply['new_origins'] for reply in replies) == 1
        snapshot = feed.snapshot(clock[0])
        expected = pd.date_range(pd.Timestamp('2026-02-01', tz='UTC'), end, freq='h', inclusive='left')
        assert pd.DatetimeIndex(snapshot.open_time).equals(expected)
        origins = snapshot[snapshot.prediction_origin]
        assert len(origins) == 1 and origins.iloc[0].open_time == end - pd.Timedelta(hours=1)
        assert (snapshot.imported_at > prepared_at).all()
        feed.verify()
    finally:
        feed.close()
