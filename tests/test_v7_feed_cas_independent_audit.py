"""Independent provenance checks for shared content and per-receipt clocks."""
import json

import pandas as pd
import pytest
from test_v7_feed_audit import Clock, bars

from factorlab.discovery.v7_feed import ReceiptArchive


def test_two_sources_can_share_content_without_sharing_receipt_time(tmp_path):
    first_clock, second_clock = Clock(), Clock('2026-10-04T13:00:05Z')
    first = ReceiptArchive(tmp_path / 'first', clock=first_clock)
    second = ReceiptArchive(tmp_path / 'second', clock=second_clock)
    try:
        frame = bars(['2026-10-04T11:00Z'])
        first.receive(frame, received_at=first_clock(), response_hash='a' * 64)
        second.receive(frame, received_at=second_clock(), response_hash='a' * 64)
        assert {p.name for p in (first.directory / 'objects').glob('*.json')} == {
            p.name for p in (second.directory / 'objects').glob('*.json')}
        assert first.snapshot(first_clock()).imported_at.iloc[0] != second.snapshot(second_clock()).imported_at.iloc[0]
        with pytest.raises(ValueError, match='No source'):
            second.snapshot(first_clock())
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize('operation', ['snapshot', 'verify'])
def test_referenced_content_damage_cannot_hide_behind_valid_receipt_hash(tmp_path, operation):
    clock = Clock()
    store = ReceiptArchive(tmp_path, clock=clock)
    try:
        store.receive(bars(['2026-10-04T11:00Z']), received_at=clock(), response_hash='a' * 64)
        raw = next((tmp_path / 'raw').glob('*.json'))
        before = raw.read_bytes()
        obj = next((tmp_path / 'objects').glob('*.json'))
        value = json.loads(obj.read_text())
        value[0]['close'] = 99.
        obj.write_text(json.dumps(value))
        assert raw.read_bytes() == before
        with pytest.raises(ValueError):
            store.snapshot(clock()) if operation == 'snapshot' else store.verify()
    finally:
        store.close()


def test_revision_recovery_keeps_original_asof_under_repeated_content(tmp_path):
    clock = Clock()
    store = ReceiptArchive(tmp_path, clock=clock)
    try:
        old = clock()
        frame = bars(['2026-10-04T11:00Z'])
        store.receive(frame, received_at=clock(), response_hash='a' * 64)
        clock.now += pd.Timedelta(minutes=1)
        corrected = bars(['2026-10-04T11:00Z'], price=101.)
        revised = clock()
        store.receive(corrected, received_at=clock(), response_hash='b' * 64)
        for _ in range(5):
            clock.now += pd.Timedelta(minutes=1)
            result = store.receive(corrected, received_at=clock(), response_hash='b' * 64)
            assert result['new_rows'] == result['new_origins'] == 0
        assert store.snapshot(old).close.iloc[0] == 100.
        current = store.snapshot(clock())
        assert current.close.iloc[0] == 101.
        assert current.imported_at.iloc[0] == revised
        assert store.verify()['revisions'] == 2
    finally:
        store.close()
