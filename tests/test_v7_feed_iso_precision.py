"""Actual seed + polled public-receipt chain; artificial HTTP/candles only."""
import pandas as pd
import pytest
from test_v7_feed_audit import Clock, Response, Session, bars, kline

from factorlab.discovery.v7_feed import ReceiptArchive


@pytest.mark.parametrize(('seed_time', 'poll_time'), [
    ('2026-10-04T12:00:05Z', '2026-10-04T12:00:15.961088Z'),
    ('2026-10-04T12:00:05.123456Z', '2026-10-04T12:00:15Z'),
    ('2026-10-04T12:00:05Z', '2026-10-04T12:00:15.123456789Z'),
])
def test_seed_and_observed_receipts_with_distinct_iso_precision(tmp_path, seed_time, poll_time):
    clock = Clock(seed_time)
    archive = ReceiptArchive(tmp_path/'archive', clock=clock)
    path = tmp_path/'seed.parquet'
    bars(['2026-10-04T10:00Z']).to_parquet(path)
    try:
        archive.import_seed(path)
        original = archive.snapshot(clock())
        clock.now = pd.Timestamp(poll_time)
        archive.poll(session=Session([Response([kline('2026-10-04T11:00Z')])]))
        frame = archive.snapshot(clock())
        assert frame.open_time.tolist() == [pd.Timestamp('2026-10-04T10:00Z'),
                                          pd.Timestamp('2026-10-04T11:00Z')]
        assert frame.imported_at.tolist() == [pd.Timestamp(seed_time), pd.Timestamp(poll_time)]
        assert pd.isna(frame.received_at.iloc[0])
        assert frame.received_at.iloc[1] == pd.Timestamp(poll_time)
        assert frame.prediction_origin.tolist() == [False, True]
        assert frame.available_at.iloc[0] == pd.Timestamp('2026-10-04T11:00:05Z')
        assert frame.available_at.iloc[1] == pd.Timestamp(poll_time)
        pd.testing.assert_frame_equal(archive.snapshot(seed_time), original)
        archive.verify()
    finally:
        archive.close()
