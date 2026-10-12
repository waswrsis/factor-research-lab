"""Peer audit: publication causality, revision ordering and bounded transport.

All transport and time faults are artificial. No real HTTP or deployment.
"""
import io
import json
import subprocess

import pandas as pd
import pytest
from test_v7_feed_audit import Clock, Response, Session, bars, kline
from test_v7_feed_storage import worker_module

from factorlab.discovery import v7_feed as feed


def test_publication_clock_cannot_precede_completed_object_write(tmp_path, monkeypatch):
    clock = Clock('2026-10-04T12:59:59Z')
    archive = feed.ReceiptArchive(tmp_path, clock=clock)
    received = clock()
    original = feed.immutable_record

    def slow_object_write(path, value):
        original(path, value)
        if path.parent.name == 'objects':
            clock.now += pd.Timedelta(seconds=20)

    monkeypatch.setattr(feed, 'immutable_record', slow_object_write)
    try:
        archive.receive(bars(['2026-10-04T11:00Z']), received_at=received, response_hash='a' * 64)
        # At this cutoff, the content object was still being written. It must
        # not later appear retrospectively available in a training snapshot.
        with pytest.raises(ValueError, match='No source'):
            archive.snapshot(received + pd.Timedelta(seconds=1))
        assert archive.snapshot(clock()).available_at.iloc[0] >= clock()
    finally:
        archive.close()


def test_equal_publication_clock_revisions_have_deterministic_last_version(tmp_path):
    clock = Clock()
    archive = feed.ReceiptArchive(tmp_path, clock=clock)
    try:
        archive.receive(bars(['2026-10-04T11:00Z']), received_at=clock(), response_hash='a' * 64)
        revised = bars(['2026-10-04T11:00Z'], price=102.)
        archive.receive(revised, received_at=clock(), response_hash='b' * 64)
        duplicate = archive.receive(revised, received_at=clock(), response_hash='c' * 64)
        assert duplicate['new_rows'] == 0
        assert archive.snapshot(clock()).close.iloc[0] == 102.
        assert archive.verify()['revisions'] == 2
    finally:
        archive.close()


def test_damaged_later_revision_does_not_invalidate_an_earlier_asof(tmp_path):
    clock = Clock()
    archive = feed.ReceiptArchive(tmp_path, clock=clock)
    try:
        old = clock()
        archive.receive(bars(['2026-10-04T11:00Z']), received_at=clock(), response_hash='a' * 64)
        clock.now += pd.Timedelta(minutes=1)
        receipt = archive.receive(bars(['2026-10-04T11:00Z'], price=102.),
                                  received_at=clock(), response_hash='b' * 64)
        manifest = json.loads((tmp_path / 'raw' / f'{receipt["batch"]}.json').read_text())
        (tmp_path / manifest['bars_ref']['path']).write_text('[]')
        assert archive.snapshot(old).close.iloc[0] == 100.
        with pytest.raises(ValueError, match='object mismatch'):
            archive.snapshot(clock())
    finally:
        archive.close()


def test_one_deadline_covers_all_symbols_and_keeps_only_completed_pages(tmp_path, monkeypatch):
    clock = Clock()
    archive = feed.ReceiptArchive(tmp_path, symbols=('BTCUSDT', 'ETHUSDT', 'SOLUSDT'), clock=clock)
    elapsed, calls = [0.], []
    monkeypatch.setattr(feed.time, 'monotonic', lambda: elapsed[0])

    class SlowSecond(Response):
        def iter_content(self, size):
            elapsed[0] = 28.1
            yield self.raw

    responses = [Response([kline('2026-10-04T11:00Z')]), SlowSecond([kline('2026-10-04T11:00Z')])]
    client = Session(responses)
    original = client.get

    def get(endpoint, **kwargs):
        calls.append(kwargs['params']['symbol'])
        return original(endpoint, **kwargs)

    client.get = get
    try:
        with pytest.raises(TimeoutError):
            archive.poll(session=client)
        assert calls == ['BTCUSDT', 'ETHUSDT']
        assert all(response.closed for response in responses)
        assert archive.verify()['revisions'] == 1
        assert set(archive.snapshot(clock()).symbol) == {'BTCUSDT'}
    finally:
        archive.close()


def test_slow_process_startup_is_inside_total_deadline_and_owned_child_is_reaped(monkeypatch):
    elapsed = [0.]
    monkeypatch.setattr(feed.time, 'monotonic', lambda: elapsed[0])

    class Child:
        stdin, stdout = io.BytesIO(), io.BytesIO()
        returncode = None
        waited = False

        def poll(self):
            return self.returncode

        def communicate(self, *args, **kwargs):
            pytest.fail('Startup already consumed the work deadline')

        def kill(self):
            self.returncode = -9

        def wait(self, timeout):
            self.waited = True
            return self.returncode

    child = Child()

    def start(*args, **kwargs):
        elapsed[0] = 28.1
        return child

    monkeypatch.setattr(feed.subprocess, 'Popen', start)
    with pytest.raises(TimeoutError, match='startup'):
        feed._owned_fetch({'symbol': 'BTCUSDT'}, 30.)
    assert child.waited and child.returncode == -9
    assert child.stdin.closed and child.stdout.closed


def test_worker_cleanup_time_is_part_of_transport_deadline(monkeypatch):
    worker = worker_module()
    now = [0.]
    monkeypatch.setattr(worker.time, 'time', lambda: now[0])

    class DelayedClose(Response):
        def close(self):
            self.closed = True
            now[0] = 21.

    response = DelayedClose([kline('2026-10-04T11:00Z')])
    client = Session([response])
    with pytest.raises(TimeoutError, match='cleanup'):
        worker.fetch({'symbol': 'BTCUSDT', 'interval': '1h', 'limit': 96, 'endTime': 1},
                     20., session=client)
    assert response.closed


def test_owned_worker_cleanup_failure_is_explicit(monkeypatch):
    class Child:
        stdin, stdout = io.BytesIO(), io.BytesIO()
        returncode = None

        def poll(self):
            return None

        def communicate(self, data, timeout):
            raise subprocess.TimeoutExpired('owned-worker', timeout)

        def kill(self):
            pass

        def wait(self, timeout):
            raise subprocess.TimeoutExpired('owned-worker-cleanup', timeout)

    monkeypatch.setattr(feed.subprocess, 'Popen', lambda *a, **k: Child())
    with pytest.raises(RuntimeError, match='cleanup failed'):
        feed._owned_fetch({'symbol': 'BTCUSDT'}, feed.time.monotonic() + 30.)
