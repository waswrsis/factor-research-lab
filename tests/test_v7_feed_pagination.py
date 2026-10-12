"""Bounded pagination recovery regression tests; no actual HTTP."""
import pandas as pd
import pytest
from test_v7_feed_audit import Clock, Response, Session, kline

from factorlab.discovery.v7_feed import ReceiptArchive


def page(end, number):
    return [kline(value) for value in pd.date_range(end - pd.Timedelta(hours=96 * number), periods=96, freq='h')]


class FailedPage(Response):
    def iter_content(self, size):
        raise TimeoutError('Synthetic interrupted older page')
        yield b''


def test_interrupted_third_page_resumes_through_unchanged_middle_page(tmp_path):
    clock = Clock()
    store = ReceiptArchive(tmp_path, clock=clock)
    end = clock().floor('h')
    try:
        with pytest.raises(TimeoutError):
            store.poll(pages=3, session=Session([Response(page(end, 1)),
                Response(page(end, 2)), FailedPage(page(end, 3))]))
        clock.now += pd.Timedelta(hours=1)
        end = clock().floor('h')
        retry = Session([Response(page(end, number)) for number in range(1, 4)])
        replies = store.poll(pages=3, session=retry)
        assert len(retry.calls) == 3
        assert [value['new_rows'] for value in replies] == [1, 0, 95]
        assert len(store.snapshot(clock())) == 288
        assert sum(value['new_origins'] for value in replies) == 1
    finally:
        store.close()


def test_pending_older_page_does_not_satisfy_published_coverage(tmp_path, monkeypatch):
    clock = Clock()
    store = ReceiptArchive(tmp_path, clock=clock)
    end = clock().floor('h')
    original = store._publish

    def interrupted(*args):
        raise OSError('Synthetic publication interruption')

    try:
        monkeypatch.setattr(store, '_publish', interrupted)
        # The older page is durable but deliberately unpublished.
        with pytest.raises(OSError):
            store.poll(session=Session([Response(page(end, 2))]))
        monkeypatch.setattr(store, '_publish', original)
        clock.now += pd.Timedelta(minutes=1)
        store.poll(session=Session([Response(page(end, 1))]))
        retry = Session([Response(page(end, 1)), Response(page(end, 2))])
        replies = store.poll(pages=2, session=retry)
        assert len(retry.calls) == 2
        assert [value['new_rows'] for value in replies] == [0, 96]
        assert sum(value['new_origins'] for value in replies) == 0
        assert len(store.snapshot(clock())) == 192
    finally:
        store.close()


def test_complete_coverage_keeps_one_request_for_unchanged_poll(tmp_path):
    clock = Clock()
    store = ReceiptArchive(tmp_path, clock=clock)
    end = clock().floor('h')
    try:
        seed = Session([Response(page(end, number)) for number in range(1, 5)])
        assert len(store.poll(pages=4, session=seed)) == 4
        clock.now += pd.Timedelta(minutes=1)
        repeat = Session([Response(page(end, 1))])
        assert len(store.poll(pages=4, session=repeat)) == 1
        assert len(repeat.calls) == 1
        assert len(store.snapshot(clock())) == 384
    finally:
        store.close()


def test_empty_archive_never_requests_a_fifth_page(tmp_path):
    clock = Clock()
    store = ReceiptArchive(tmp_path, clock=clock)
    end = clock().floor('h')
    try:
        client = Session([Response(page(end, number)) for number in range(1, 6)])
        result = store.poll(pages=4, session=client)
        assert len(client.calls) == len(result) == 4
        assert len(client.responses) == 1
        assert sum(value['new_origins'] for value in result) == 1
    finally:
        store.close()
