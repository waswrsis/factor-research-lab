"""Independent receipt provenance and bounded-public-network regression tests."""
import json
from pathlib import Path

import pandas as pd
import pytest
import requests

from factorlab.discovery.v7_feed import ENDPOINT, ReceiptArchive


class Clock:
    def __init__(self, now='2026-10-04T12:00:05Z'):
        self.now = pd.Timestamp(now)

    def __call__(self):
        return self.now


def bars(times, price=100.):
    return pd.DataFrame({'symbol': 'BTCUSDT', 'open_time': pd.to_datetime(times, utc=True),
                         'open': price, 'close': price, 'high': price + 1,
                         'low': price - 1, 'volume': 10.})


@pytest.fixture
def archive(tmp_path):
    clock = Clock()
    result = ReceiptArchive(tmp_path / 'archive', clock=clock)
    yield result, clock
    result.close()


@pytest.mark.parametrize('operation', ['verify', 'snapshot'])
def test_raw_object_damage_cannot_be_ignored(archive, operation):
    store, clock = archive
    receipt = store.receive(bars(['2026-10-04T11:00Z']), received_at=clock(), response_hash='a' * 64)
    (store.directory / 'raw' / f"{receipt['batch']}.json").write_text('{}', encoding='utf-8')
    with pytest.raises((ValueError, FileNotFoundError), match='.*'):
        store.verify() if operation == 'verify' else store.snapshot(clock())


@pytest.mark.parametrize('operation', ['verify', 'snapshot'])
def test_orphan_state_row_without_raw_batch_reference_is_rejected(archive, operation):
    store, clock = archive
    store.receive(bars(['2026-10-04T11:00Z']), received_at=clock(), response_hash='a' * 64)
    store.db.execute('DELETE FROM batches')
    store.db.commit()
    with pytest.raises(ValueError, match='.*'):
        store.verify() if operation == 'verify' else store.snapshot(clock())


@pytest.mark.parametrize('column,value', [('receipt', '2026-10-04T09:00:00+00:00'),
                                       ('opened', '2026-10-04T09:00:00+00:00'), ('origin', 0),
                                       ('batch', 'f' * 64)])
def test_state_index_fields_are_bound_to_content(archive, column, value):
    store, clock = archive
    store.receive(bars(['2026-10-04T11:00Z']), received_at=clock(), response_hash='a' * 64)
    store.db.execute(f'UPDATE bars SET {column}=?', (value,))
    store.db.commit()
    with pytest.raises(ValueError, match='.*'):
        store.snapshot(clock())


def test_received_timestamp_cannot_create_preimport_snapshot(archive):
    store, clock = archive
    old = clock() - pd.Timedelta(hours=1)
    try:
        store.receive(bars(['2026-10-04T09:00Z']), received_at=old, response_hash='b' * 64)
    except ValueError:
        return  # Rejecting a backdated public receipt is also a valid policy.
    with pytest.raises(ValueError, match='No source|cutoff|received|import'):
        store.snapshot(old + pd.Timedelta(minutes=1))


def test_seed_import_stays_assumed_and_has_no_historical_origin(archive, tmp_path):
    store, clock = archive
    path = tmp_path / 'seed.parquet'
    bars(['2026-10-01T09:00Z', '2026-10-01T10:00Z']).to_parquet(path)
    receipt = store.import_seed(path)
    assert receipt['new_origins'] == 0
    with pytest.raises(ValueError, match='No source'):
        store.snapshot(clock() - pd.Timedelta(seconds=1))
    frame = store.snapshot(clock())
    assert not frame.prediction_origin.any()
    assert frame.received_at.isna().all()
    assert (frame.imported_at == clock()).all()
    assert set(frame.availability_basis) == {'assumed_archive'}


@pytest.mark.parametrize('delay', [-1, 3601, True, float('nan')])
def test_seed_availability_delay_cannot_backdate_or_escape_frozen_bounds(archive, tmp_path, delay):
    store, clock = archive
    path = tmp_path / 'seed.parquet'
    bars(['2026-10-01T09:00Z']).to_parquet(path)
    store.import_seed(path)
    with pytest.raises(ValueError, match='delay|availability|bound|seconds'):
        store.snapshot(clock(), archive_delay_seconds=delay)


def test_duplicate_and_revision_preserve_asof_history_and_no_new_origin(archive):
    store, clock = archive
    first = store.receive(bars(['2026-10-04T11:00Z']), received_at=clock(), response_hash='a' * 64)
    old = clock()
    clock.now += pd.Timedelta(seconds=10)
    duplicate = store.receive(bars(['2026-10-04T11:00Z']), received_at=clock(), response_hash='a' * 64)
    assert first['new_origins'] == 1 and duplicate['new_rows'] == 0 and duplicate['new_origins'] == 0
    clock.now += pd.Timedelta(seconds=10)
    changed = store.receive(bars(['2026-10-04T11:00Z'], price=102.), received_at=clock(), response_hash='b' * 64)
    assert changed['new_rows'] == 1 and changed['new_origins'] == 0
    assert store.snapshot(old).close.iloc[0] == 100.
    assert store.snapshot(clock()).close.iloc[0] == 102.


def kline(opened):
    opened = pd.Timestamp(opened)
    begin = int(opened.timestamp() * 1000)
    return [begin, '100', '101', '99', '100', '10', begin + 3_600_000 - 1,
            '1000', 10, '1', '100', '0']


class Response:
    def __init__(self, payload, raw=None):
        self.raw = json.dumps(payload).encode() if raw is None else raw
        self.closed = False
        self.status_code = 200
        self.headers = {}
        self.url = ENDPOINT

    def raise_for_status(self):
        pass

    def iter_content(self, size):
        for i in range(0, len(self.raw), size):
            yield self.raw[i:i + size]

    def close(self):
        self.closed = True


class Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.auth = None
        self.headers = {}
        self.cookies = {}
        self.params = {}
        self.trust_env = False
        self.proxies = {}
        self.closed = False

    def get(self, endpoint, **kwargs):
        self.calls.append((endpoint, kwargs))
        return self.responses.pop(0)

    def close(self):
        self.closed = True


def test_poll_two_pages_has_one_latest_origin_and_exact_millisecond_query(archive):
    store, clock = archive
    end = clock().floor('h')
    latest = [kline(t) for t in pd.date_range(end - pd.Timedelta(hours=96), periods=96, freq='h')]
    older = [kline(t) for t in pd.date_range(end - pd.Timedelta(hours=192), periods=96, freq='h')]
    responses = [Response(latest), Response(older)]
    session = Session(responses)
    receipts = store.poll(session=session, pages=2)
    assert len(receipts) == 2 and sum(r['new_rows'] for r in receipts) == 192
    assert sum(r['new_origins'] for r in receipts) == 1
    frame = store.snapshot(clock())
    assert frame.loc[frame.prediction_origin, 'open_time'].tolist() == [end - pd.Timedelta(hours=1)]
    assert all(response.closed for response in responses)
    assert session.calls[0][1]['params']['endTime'] == int(end.timestamp() * 1000) - 1
    assert session.calls[1][1]['params']['endTime'] == latest[0][0] - 1
    assert all(c[1]['timeout'] == (5, 15) and c[1]['stream'] is True for c in session.calls)
    assert all(c[1].get('allow_redirects') is False for c in session.calls)


@pytest.mark.parametrize('credential', ['Authorization', 'X-MBX-APIKEY', 'Cookie', 'auth', 'cookies'])
def test_public_poll_does_not_use_credential_bearing_sessions(archive, credential):
    store, _ = archive
    session = Session([Response([kline('2026-10-04T11:00Z')])])
    if credential == 'auth':
        session.auth = ('username', 'test-secret')
    elif credential == 'cookies':
        session.cookies = {'session': 'test-secret'}
    else:
        session.headers[credential] = 'test-secret'
    with pytest.raises(ValueError, match='credential|auth|cookie|public|header'):
        store.poll(session=session)
    assert not session.calls


def test_public_response_size_is_bounded_and_always_closed(archive):
    store, _ = archive
    response = Response([], raw=b' ' * (1_048_576 + 1))
    with pytest.raises(ValueError, match='MiB'):
        store.poll(session=Session([response]))
    assert response.closed
    assert store.verify()['batches'] == 0


def test_default_public_session_disables_netrc_environment_auth(archive, monkeypatch):
    store, _ = archive
    session = Session([Response([kline('2026-10-04T11:00Z')])])
    session.trust_env = True  # requests.Session defaults to this.
    store.poll(session=session)
    assert session.trust_env is False
    assert not session.closed  # Explicit injected clients remain caller-owned.


def test_injected_public_session_cannot_implicitly_send_netrc_credentials(archive, monkeypatch):
    store, _ = archive
    session = requests.Session()
    sent_headers = []
    monkeypatch.setattr('requests.sessions.get_netrc_auth', lambda url: ('test-user', 'test-secret'))

    def sent(request, **kwargs):
        sent_headers.append(dict(request.headers))
        return Response([kline('2026-10-04T11:00Z')])

    monkeypatch.setattr(session, 'send', sent)
    try:
        store.poll(session=session)
    except ValueError:
        assert not sent_headers
    finally:
        session.close()
    assert all('Authorization' not in headers for headers in sent_headers)


def test_snapshot_validates_each_raw_batch_once_not_once_per_candle(archive, monkeypatch):
    store, clock = archive
    times = pd.date_range(clock().floor('h') - pd.Timedelta(hours=24), periods=24, freq='h')
    receipt = store.receive(bars(times), received_at=clock(), response_hash='a' * 64)
    expected = store.directory / 'raw' / f"{receipt['batch']}.json"
    original, reads = Path.read_text, []

    def counted(path, *args, **kwargs):
        if path == expected:
            reads.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', counted)
    frame = store.snapshot(clock())
    assert len(frame) == 24
    assert len(reads) == 1  # A seed file can have thousands of candles.
