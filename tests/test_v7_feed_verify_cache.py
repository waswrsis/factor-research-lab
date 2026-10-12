"""Bounded per-verify deduplication keeps receipt and later-tamper checks."""
import json
import os
from pathlib import Path

import pandas as pd
import pytest
from test_v7_feed_audit import Clock, bars

from factorlab.discovery.v7_feed import ReceiptArchive
from factorlab.protocol import fingerprint


def repeated_archive(tmp_path, count=20):
    clock = Clock()
    store = ReceiptArchive(tmp_path, clock=clock)
    for _ in range(count):
        store.receive(bars(['2026-10-04T11:00Z']), received_at=clock(), response_hash='a' * 64)
        clock.now += pd.Timedelta(minutes=1)
    return store, clock


def test_same_content_is_not_read_once_per_receipt_and_next_call_rereads(tmp_path, monkeypatch):
    store, _ = repeated_archive(tmp_path)
    reads = []
    parses = []
    original = Path.read_bytes
    original_loads = json.loads

    def read(path, *args, **kwargs):
        if path.parent.name == 'objects':
            reads.append(path)
        return original(path, *args, **kwargs)

    def loads(*args, **kwargs):
        result = original_loads(*args, **kwargs)
        if (isinstance(result, list) and result and isinstance(result[0], dict)
                and 'open_time' in result[0] and 'symbol' in result[0]):
            parses.append(1)
        return result

    monkeypatch.setattr(Path, 'read_bytes', read)
    monkeypatch.setattr(json, 'loads', loads)
    try:
        assert store.verify()['batches'] == 20
        # Every reference checks actual bytes; only parsing/semantic checks
        # are deduplicated. A stat-only hit would miss same-stat tampering.
        assert len(reads) >= 20
        assert 1 <= len(parses) <= 2
        first_count = len(reads)
        first_parses = len(parses)
        store.verify()
        assert len(reads) >= first_count + 20
        assert first_parses < len(parses) <= 2 * first_parses
        reads[0].write_text('[]')
        with pytest.raises(ValueError, match='object mismatch'):
            store.verify()
    finally:
        store.close()


def test_same_size_and_mtime_content_change_invalidates_helper_cache(tmp_path):
    store, _ = repeated_archive(tmp_path, count=1)
    try:
        batch = json.loads(store.db.execute('SELECT record FROM batches').fetchone()[0])
        path = store.directory / batch['bars_ref']['path']
        cache = {}
        assert store._batch_bars(batch, verified_objects=cache)
        assert store._batch_bars(batch, verified_objects=cache) is None
        raw = path.read_bytes()
        assert raw.count(b'"open": 100.0') == 1
        changed = raw.replace(b'"open": 100.0', b'"open": 900.0', 1)
        assert len(changed) == len(raw) and changed != raw
        stat = path.stat()
        path.write_bytes(changed)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        assert path.stat().st_size == stat.st_size
        assert path.stat().st_mtime_ns == stat.st_mtime_ns
        with pytest.raises(ValueError, match='Content-addressed candle object mismatch'):
            store._batch_bars(batch, verified_objects=cache)
    finally:
        store.close()


def test_helper_checks_captured_bytes_even_if_file_restored_after_read(tmp_path, monkeypatch):
    store, _ = repeated_archive(tmp_path, count=1)
    try:
        batch = json.loads(store.db.execute('SELECT record FROM batches').fetchone()[0])
        path = store.directory / batch['bars_ref']['path']
        good = path.read_bytes()
        values = json.loads(good)
        values[0]['open'] += 800.
        bad = json.dumps(values, sort_keys=True, separators=(',', ':')).encode()
        path.write_bytes(bad)
        original = Path.read_bytes
        reads = []
        cache = {}

        def restore_after_capture(current, *args, **kwargs):
            raw = original(current, *args, **kwargs)
            if current == path:
                reads.append(1)
                current.write_bytes(good)
            return raw

        with monkeypatch.context() as context:
            context.setattr(Path, 'read_bytes', restore_after_capture)
            with pytest.raises(ValueError, match='Content-addressed candle object mismatch'):
                store._batch_bars(batch, verified_objects=cache)
        assert reads == [1] and cache == {}
        assert path.read_bytes() == good
        assert store._batch_bars(batch, verified_objects=cache) == json.loads(good)
    finally:
        store.close()


@pytest.mark.parametrize('field,value', [('count', 2), ('path', '../outside.json'), ('hash', 'f' * 64)])
def test_each_later_receipt_still_checks_reference_despite_cached_content(tmp_path, field, value):
    store, _ = repeated_archive(tmp_path, count=2)
    try:
        identifier, text = store.db.execute('SELECT id,record FROM batches ORDER BY rowid DESC LIMIT 1').fetchone()
        batch = json.loads(text)
        batch['bars_ref'][field] = value
        replacement = fingerprint(batch)
        (tmp_path / 'raw' / f'{replacement}.json').write_text(json.dumps(batch))
        # Keep the publication row linked so the early orphan check doesn't
        # mask the intended cached-content reference validation.
        store.db.execute('UPDATE batches SET id=?,record=?,hash=? WHERE id=?',
                         (replacement, json.dumps(batch), replacement, identifier))
        store.db.execute('UPDATE publications SET batch=? WHERE batch=?', (replacement, identifier))
        store.db.commit()
        with pytest.raises(ValueError, match='reference|count|object'):
            store.verify()
    finally:
        store.close()


def test_verification_cache_has_bounded_metadata_not_candle_arrays(tmp_path, monkeypatch):
    clock = Clock()
    store = ReceiptArchive(tmp_path, clock=clock)
    original = store._batch_bars
    seen = []
    try:
        for index in range(132):
            store.receive(bars(['2026-10-04T11:00Z'], price=100. + index),
                          received_at=clock(), response_hash='a' * 64)
            clock.now += pd.Timedelta(seconds=1)

        def inspect(batch, **kwargs):
            result = original(batch, **kwargs)
            cache = kwargs.get('verified_objects')
            if cache is not None:
                seen.append(len(cache))
                assert all(isinstance(value[0], int) and len(value[1]) == 2 for value in cache.values())
            return result

        monkeypatch.setattr(store, '_batch_bars', inspect)
        assert store.verify()['revisions'] == 132
        assert len(seen) == 132 and max(seen) == 128
    finally:
        store.close()
