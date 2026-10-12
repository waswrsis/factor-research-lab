"""Bounded synthetic receipts and owned-transport tests; no real HTTP."""
import importlib.util
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest
import requests
from test_v7_feed_audit import Clock, Response, Session, bars, kline

from factorlab.discovery import v7_feed as module
from factorlab.discovery.v7_feed import ReceiptArchive
from factorlab.protocol import fingerprint


def test_repeated_96_bar_receipts_share_one_object_and_preserve_original_times(tmp_path):
    clock = Clock()
    store = ReceiptArchive(tmp_path, clock=clock)
    frame = bars(pd.date_range(clock().floor('h') - pd.Timedelta(hours=96), periods=96, freq='h'))
    first = store.receive(frame, received_at=clock(), response_hash='a' * 64)
    original = store.snapshot(clock())
    payload = [kline(value) for value in frame.open_time]
    for _ in range(100):
        clock.now += pd.Timedelta(minutes=1)
        reply, = store.poll(session=Session([Response(payload)]))
        assert reply['new_rows'] == reply['new_origins'] == 0
    assert len(list((tmp_path / 'objects').glob('*.json'))) == 1
    receipts = list((tmp_path / 'raw').glob('*.json'))
    assert len(receipts) == 101
    assert max(path.stat().st_size for path in receipts) < 1000
    sql_texts = [text for text, in store.db.execute('SELECT record FROM batches')]
    assert all('"bars":' not in text and len(text) < 1000 for text in sql_texts)
    object_size = next((tmp_path / 'objects').glob('*.json')).stat().st_size
    assert sum(path.stat().st_size for path in receipts) + object_size < object_size * 101 / 10
    pd.testing.assert_frame_equal(store.snapshot(clock()), original)
    assert first['new_origins'] == 1 and store.verify()['revisions'] == 96
    store.close()


def test_seed_and_revision_keep_full_content_and_asof(tmp_path):
    clock = Clock()
    seed = tmp_path / 'seed.parquet'
    bars(['2026-10-04T11:00Z']).to_parquet(seed)
    store = ReceiptArchive(tmp_path / 'archive', clock=clock)
    store.import_seed(seed)
    first_time = clock()
    assert not store.snapshot(first_time).prediction_origin.any()
    with pytest.raises(ValueError, match='No source'):
        store.snapshot(first_time - pd.Timedelta(seconds=1))
    clock.now += pd.Timedelta(minutes=1)
    changed = store.receive(bars(['2026-10-04T11:00Z'], price=102), received_at=clock(), response_hash='b' * 64)
    assert changed['new_rows'] == 1 and not changed['new_origins']
    assert store.snapshot(first_time).close.iloc[0] == 100
    assert store.snapshot(clock()).close.iloc[0] == 102
    assert len(list((store.directory / 'objects').glob('*.json'))) == 2
    store.verify()
    store.close()


def test_legacy_inline_receipt_remains_unchanged_and_mixes_with_cas(tmp_path):
    clock = Clock()
    store = ReceiptArchive(tmp_path, clock=clock)
    frame = bars(['2026-10-04T11:00Z'])
    item = {'symbol': 'BTCUSDT', 'open_time': frame.open_time.iloc[0].isoformat(),
            'open': 100., 'high': 101., 'low': 99., 'close': 100., 'volume': 10.}
    legacy = {'source': store.identity, 'received_at': clock().isoformat(),
              'published_at': clock().isoformat(), 'actual': True,
              'source_ref': {'response_sha256': 'a' * 64}, 'bars': [item]}
    batch = fingerprint(legacy)
    raw = tmp_path / 'raw' / f'{batch}.json'
    raw.parent.mkdir()
    original = json.dumps(legacy, indent=2).encode()
    raw.write_bytes(original)
    body = {'ohlcv': item, 'received_at': clock().isoformat(), 'imported_at': clock().isoformat(),
            'basis': 'actual_receipt', 'prediction_origin': True}
    store.db.execute('INSERT INTO batches VALUES (?,?,?)', (batch, json.dumps(legacy), batch))
    store.db.execute('INSERT INTO bars VALUES (?,?,?,?,?,?,?)', ('BTCUSDT', item['open_time'],
        clock().isoformat(), fingerprint(body), 1, json.dumps(body), batch))
    store.db.commit()
    before = store.snapshot(clock())
    clock.now += pd.Timedelta(minutes=1)
    assert store.receive(frame, received_at=clock(), response_hash='a' * 64)['new_rows'] == 0
    pd.testing.assert_frame_equal(before, store.snapshot(clock()))
    assert raw.read_bytes() == original
    assert store.verify()['batches'] == 2
    store.close()


@pytest.mark.parametrize('damage', ['missing', 'bytes', 'path', 'count', 'hash'])
def test_object_references_verified_even_for_duplicate_only_receipts(tmp_path, damage):
    clock = Clock()
    store = ReceiptArchive(tmp_path, clock=clock)
    frame = bars(['2026-10-04T11:00Z'])
    store.receive(frame, received_at=clock(), response_hash='a' * 64)
    clock.now += pd.Timedelta(minutes=1)
    result = store.receive(frame, received_at=clock(), response_hash='a' * 64)
    old_id = result['batch']
    batch = json.loads(store.db.execute('SELECT record FROM batches WHERE id=?', (old_id,)).fetchone()[0])
    obj = tmp_path / batch['bars_ref']['path']
    if damage == 'missing':
        obj.unlink()
    elif damage == 'bytes':
        obj.write_text('[]')
    else:
        if damage == 'path':
            batch['bars_ref']['path'] = '../outside.json'
        elif damage == 'count':
            batch['bars_ref']['count'] = 2
        else:
            batch['bars_ref']['hash'] = 'f' * 64
        new_id = fingerprint(batch)
        (tmp_path / 'raw' / f'{new_id}.json').write_text(json.dumps(batch))
        store.db.execute('UPDATE batches SET id=?,record=?,hash=? WHERE id=?',
                         (new_id, json.dumps(batch), new_id, old_id))
        store.db.commit()
    with pytest.raises((ValueError, FileNotFoundError)):
        store.verify()
    store.close()


def worker_module():
    path = Path(module.__file__).resolve().parents[2] / 'scripts' / 'v7_public_fetch.py'
    spec = importlib.util.spec_from_file_location('feed_http_worker_test', path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker


def test_worker_real_request_preparation_cannot_read_netrc_or_env_proxy(monkeypatch):
    worker = worker_module()
    session = requests.Session()
    sent = []

    def forbidden(*args, **kwargs):
        pytest.fail('Ambient credentials/proxies were consulted')

    def send(request, **kwargs):
        sent.append(request)
        assert not kwargs.get('proxies')
        return Response([kline('2026-10-04T11:00Z')])

    monkeypatch.setattr('requests.sessions.get_netrc_auth', forbidden)
    monkeypatch.setattr('requests.sessions.get_environ_proxies', forbidden)
    monkeypatch.setattr(session, 'send', send)
    monkeypatch.setattr('requests.Session', lambda: session)
    params = {'symbol': 'BTCUSDT', 'interval': '1h', 'limit': 96, 'endTime': 1}
    assert worker.fetch(params, time.time() + 20)['payload']
    assert len(sent) == 1 and 'Authorization' not in sent[0].headers
    assert not session.trust_env


def test_injected_slow_stream_rejected_before_late_receipt(tmp_path, monkeypatch):
    clock = Clock()
    store = ReceiptArchive(tmp_path, clock=clock)
    monotonic = [0.]
    monkeypatch.setattr(module.time, 'monotonic', lambda: monotonic[0])

    class Slow(Response):
        def iter_content(self, size):
            monotonic[0] = 29.
            yield self.raw

    response = Slow([kline('2026-10-04T11:00Z')])
    with pytest.raises(TimeoutError):
        store.poll(session=Session([response]))
    assert response.closed and store.verify()['batches'] == 0
    store.close()


def test_default_transport_uses_owned_light_worker_and_common_deadline(tmp_path, monkeypatch):
    clock = Clock()
    store = ReceiptArchive(tmp_path, clock=clock)
    calls = []
    latest = [kline(t) for t in pd.date_range(clock().floor('h') - pd.Timedelta(hours=96), periods=96, freq='h')]

    def fetched(params, deadline):
        calls.append((params, deadline))
        if len(calls) == 2:
            raise TimeoutError('next page deadline')
        return latest, clock()

    monkeypatch.setattr(module, '_owned_fetch', fetched)
    with pytest.raises(TimeoutError):
        store.poll(pages=2)
    assert len(calls) == 2 and calls[0][1] == calls[1][1]
    assert store.verify()['revisions'] == 96
    store.close()


@pytest.mark.parametrize('survives', [False, True])
def test_owned_timeout_kills_only_its_retained_child_and_checks_exit(monkeypatch, survives):
    class Child:
        returncode = None
        stdin, stdout = io.BytesIO(), io.BytesIO()
        killed = False

        def communicate(self, data, timeout):
            raise subprocess.TimeoutExpired('fixed worker', timeout)

        def poll(self):
            return self.returncode

        def kill(self):
            self.killed = True
            if not survives:
                self.returncode = -9

        def wait(self, timeout):
            return self.returncode

    child = Child()
    calls = []

    def launch(command, **kwargs):
        calls.append((command, kwargs))
        return child

    monkeypatch.setattr(module.subprocess, 'Popen', launch)
    with pytest.raises(RuntimeError if survives else TimeoutError):
        module._owned_fetch({'symbol': 'BTCUSDT'}, time.monotonic() + 30)
    assert child.killed and len(calls) == 1
    command, kwargs = calls[0]
    assert command[2].endswith('v7_public_fetch.py') and '-m' not in command
    assert kwargs['shell'] is False


def test_light_worker_watchdog_stops_blocking_fake_stream_without_network(tmp_path):
    # The real script imports this artificial transport instead of requests.
    (tmp_path / 'requests.py').write_text('''
import time
class Response:
    status_code = 200
    def raise_for_status(self): pass
    def iter_content(self, size):
        time.sleep(10)
        yield b"[]"
    def close(self): pass
class Session:
    headers = {}
    auth = cookies = params = proxies = None
    def get(self, *args, **kwargs): return Response()
    def close(self): pass
''', encoding='utf-8')
    script = Path(module.__file__).resolve().parents[2] / 'scripts' / 'v7_public_fetch.py'
    env = {**os.environ, 'PYTHONPATH': str(tmp_path)}
    started = time.monotonic()
    reply = subprocess.run([sys.executable, '-B', str(script), str(time.time() + 1.5)],
        input=json.dumps({'symbol': 'BTCUSDT', 'interval': '1h', 'limit': 96, 'endTime': 1}).encode(),
        capture_output=True, env=env, timeout=5, check=False, **module.isolated_process_kwargs())
    assert reply.returncode == 124 and not reply.stdout
    assert time.monotonic() - started < 5


def test_default_worker_late_success_is_not_published(tmp_path, monkeypatch):
    store = ReceiptArchive(tmp_path, clock=Clock())
    monotonic = [0.]
    monkeypatch.setattr(module.time, 'monotonic', lambda: monotonic[0])

    def late(params, deadline):
        monotonic[0] = 28.1
        return [kline('2026-10-04T11:00Z')], store.clock()

    monkeypatch.setattr(module, '_owned_fetch', late)
    with pytest.raises(TimeoutError, match='late'):
        store.poll()
    assert store.verify()['batches'] == 0
    store.close()


@pytest.mark.parametrize('phase', ['raw_write', 'prepare_commit', 'prepare_lock'])
def test_publication_waits_for_all_durable_data_preparation(tmp_path, monkeypatch, phase):
    clock = Clock('2026-10-31T23:59:55Z')
    store = ReceiptArchive(tmp_path, clock=clock)
    received = clock()
    original_write = module.immutable_record

    def slow_write(path, value):
        result = original_write(path, value)
        if phase == 'raw_write' and path.parent.name == 'raw':
            clock.now += pd.Timedelta(seconds=20)
        return result

    monkeypatch.setattr(module, 'immutable_record', slow_write)
    database = store.db

    class DelayedDatabase:
        commits = locks = 0

        def __getattr__(self, key):
            return getattr(database, key)

        def execute(self, sql, *args):
            if sql == 'BEGIN IMMEDIATE':
                self.locks += 1
                if self.locks == 1 and phase == 'prepare_lock':
                    clock.now += pd.Timedelta(seconds=20)
            return database.execute(sql, *args)

        def __enter__(self):
            database.__enter__()
            return self

        def __exit__(self, *args):
            result = database.__exit__(*args)
            self.commits += 1
            if self.commits == 1 and phase == 'prepare_commit':
                clock.now += pd.Timedelta(seconds=20)
            return result

    store.db = DelayedDatabase()
    try:
        store.receive(bars(['2026-10-31T22:00Z']), received_at=received, response_hash='a' * 64)
        with pytest.raises(ValueError, match='No source'):
            store.snapshot(received + pd.Timedelta(seconds=1))
        result = store.snapshot(clock())
        assert result.imported_at.iloc[0] == clock()
        assert result.received_at.iloc[0] == received
    finally:
        store.close()


@pytest.mark.parametrize('after_certificate', [False, True])
def test_unpublished_preparation_never_backfills_and_real_repoll_can_publish(tmp_path, monkeypatch, after_certificate):
    clock = Clock()
    store = ReceiptArchive(tmp_path, clock=clock)
    frame = bars(['2026-10-04T11:00Z'])
    first = clock()
    original = module.immutable_record

    def fail_publication(path, value):
        if path.parent.name == 'publications':
            if after_certificate:
                original(path, value)
            raise OSError('Artificial publication commit failure')
        return original(path, value)

    monkeypatch.setattr(module, 'immutable_record', fail_publication)
    with pytest.raises(OSError, match='publication'):
        store.receive(frame, received_at=clock(), response_hash='a' * 64)
    assert store.verify()['unpublished_batches'] == 1
    store.close()
    monkeypatch.setattr(module, 'immutable_record', original)
    clock.now += pd.Timedelta(minutes=1)
    store = ReceiptArchive(tmp_path, clock=clock)
    try:
        with pytest.raises(ValueError, match='No source'):
            store.snapshot(first)
        with pytest.raises(ValueError, match='No source'):
            store.snapshot(clock())
        outcome = store.receive(frame, received_at=clock(), response_hash='a' * 64)
        assert outcome['new_rows'] == 1 and outcome['new_origins'] == 1
        published = store.snapshot(clock())
        assert published.imported_at.iloc[0] == clock()
        assert published.received_at.iloc[0] == clock()
        with pytest.raises(ValueError, match='No source'):
            store.snapshot(first)
        clock.now += pd.Timedelta(minutes=1)
        assert store.receive(frame, received_at=clock(), response_hash='b' * 64)['new_rows'] == 0
        pd.testing.assert_frame_equal(store.snapshot(clock()), published)
    finally:
        store.close()


def test_same_clock_price_reversion_is_a_distinct_ordered_revision(tmp_path):
    clock = Clock()
    store = ReceiptArchive(tmp_path, clock=clock)
    try:
        for price, digest in [(100, 'a'), (102, 'b'), (100, 'c')]:
            assert store.receive(bars(['2026-10-04T11:00Z'], price),
                received_at=clock(), response_hash=digest * 64)['new_rows'] == 1
        assert store.snapshot(clock()).close.iloc[0] == 100
        assert store.verify()['revisions'] == 3
    finally:
        store.close()


@pytest.mark.parametrize('damage', ['missing', 'time', 'sequence', 'certificate'])
def test_publication_index_is_bound_to_immutable_certificate(tmp_path, damage):
    clock = Clock()
    store = ReceiptArchive(tmp_path, clock=clock)
    try:
        result = store.receive(bars(['2026-10-04T11:00Z']), received_at=clock(), response_hash='a' * 64)
        if damage == 'missing':
            store.db.execute('DELETE FROM publications')
        elif damage == 'time':
            store.db.execute("UPDATE publications SET published_at='2000-01-01T00:00:00+00:00'")
        elif damage == 'sequence':
            store.db.execute('UPDATE publications SET sequence=900')
        else:
            (tmp_path / 'publications' / f'{result["batch"]}.json').write_text('{}')
        store.db.commit()
        with pytest.raises(ValueError, match='[Pp]ublication'):
            store.verify()
        with pytest.raises(ValueError, match='[Pp]ublication'):
            store.snapshot(clock())
    finally:
        store.close()
