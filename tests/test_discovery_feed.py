"""Bounded HTTP fixtures test live arrival mechanics without market claims."""
import json

import numpy as np
import pandas as pd
import pytest
import requests

from factorlab.discovery import feed
from factorlab.discovery.contracts import ExperimentSpec, TaskSpec
from factorlab.discovery.evidence import ExperimentJournal, file_hash
from factorlab.discovery.evolution import _visible_hash
from factorlab.discovery.feed import initialize_feed, load_feed, refresh_feed
from factorlab.discovery.tasks import build_task_dataset

BASE = pd.Timestamp('2024-01-01', tz='UTC')


def seed(tmp_path, symbols=('BTCUSDT',), n=60, varied=False):
    times = pd.date_range(BASE, periods=n, freq='h')
    frames = [pd.DataFrame({'symbol': symbol, 'open_time': times,
        'close_time': times + pd.Timedelta(hours=1),
        'available_at': times + pd.Timedelta(hours=1, seconds=5),
        'open': 100., 'high': 102., 'low': 99., 'close': 101., 'volume': 3.}) for symbol in symbols]
    if varied:
        for frame in frames:
            prices = 100 + np.arange(n) * .1 + np.sin(np.arange(n)) * .05
            frame['open'], frame['close'] = prices, prices + .01
            frame['high'], frame['low'] = prices + 1, prices - 1
            frame['volume'] = 3 + np.arange(n) % 5
    source = tmp_path / 'seed.parquet'
    pd.concat(frames, ignore_index=True).to_parquet(source, index=False)
    directory = tmp_path / 'feed'
    original_hash = file_hash(source)
    initialize_feed(directory, source, symbols)
    return directory, source, original_hash


def kline(hour, close='101'):
    opened = BASE + pd.Timedelta(hours=hour)
    return [int(opened.timestamp() * 1000), '100', '102', '99', close, '3',
            int((opened + pd.Timedelta(hours=1)).timestamp() * 1000) - 1,
            '0', 1, '0', '0', '0']


class Response:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError('fixture HTTP error', response=self)

    def iter_content(self, _size):
        yield json.dumps(self.payload).encode()

    def close(self):
        self.closed = True


class Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append({'url': url, **kwargs})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_initialization_copies_seed_and_live_poll_never_backdates_or_replays_backfill(tmp_path):
    directory, source, original_hash = seed(tmp_path)
    before = load_feed(directory)
    assert before.attrs['availability_mode'] == 'observed_rest_feed'
    assert before.received_at.isna().all() and before.prediction_origin.all()
    assert set(before.availability_basis) == {'historical_assumed_seed'}
    now = BASE + pd.Timedelta(hours=64, seconds=5)
    client = Session([Response([kline(hour) for hour in range(58, 65)])])
    result = refresh_feed(directory, ('BTCUSDT',), now=now, session=client)
    assert result['added_rows'] == 4 and result['live_prediction_origins'] == 1
    assert len(client.calls) == 1 and client.calls[0]['params']['limit'] == 96
    assert client.calls[0]['params']['endTime'] == int(now.floor('h').timestamp() * 1000) - 1
    after = load_feed(directory)
    pd.testing.assert_frame_equal(before, after.iloc[:len(before)].reset_index(drop=True))
    added = after.iloc[len(before):]
    assert added.available_at.eq(now).all() and added.received_at.eq(now).all()
    assert set(added.availability_basis) == {'actual_rest_receipt'}
    assert added.prediction_origin.tolist() == [False, False, False, True]
    assert after.open_time.max() == BASE + pd.Timedelta(hours=63)
    assert file_hash(source) == original_hash == file_hash(directory / 'seed' / 'bars.parquet')
    unchanged = refresh_feed(directory, ('BTCUSDT',), now=now,
        session=Session([Response([kline(hour) for hour in range(62, 65)])]))
    assert unchanged['status'] == 'unchanged'
    pd.testing.assert_frame_equal(load_feed(directory), after)


def test_receipts_are_recorded_per_response_and_publication_waits_for_complete_batch(tmp_path):
    directory, _source, _digest = seed(tmp_path, ('BTCUSDT', 'ETHUSDT'))
    start = BASE + pd.Timedelta(hours=64, seconds=5)
    instants = iter(start + pd.Timedelta(seconds=i) for i in range(4))
    refresh_feed(directory, ('BTCUSDT', 'ETHUSDT'), now=lambda: next(instants),
        session=Session([Response([kline(63)]), Response([kline(63)])]))
    added = load_feed(directory).query("availability_basis == 'actual_rest_receipt'")
    assert added.received_at.tolist() == [start + pd.Timedelta(seconds=1), start + pd.Timedelta(seconds=2)]
    assert added.available_at.eq(start + pd.Timedelta(seconds=3)).all()
    assert added.prediction_origin.all()


def test_real_clock_microsecond_resolution_preserves_canonical_utc_on_append(tmp_path, monkeypatch):
    directory, _source, _digest = seed(tmp_path)
    native_time = (BASE + pd.Timedelta(hours=64, seconds=5, microseconds=123456)).to_pydatetime()

    class NativeClock:
        @staticmethod
        def now(_timezone):
            return native_time

    monkeypatch.setattr(feed, 'datetime', NativeClock)
    result = refresh_feed(directory, ('BTCUSDT',),
        session=Session([Response([kline(hour) for hour in range(60, 64)])]))
    assert result['added_rows'] == 4
    rows = load_feed(directory)
    assert str(rows.available_at.dtype) == 'datetime64[ns, UTC]'
    assert str(rows.received_at.dtype) == 'datetime64[ns, UTC]'
    added = rows.iloc[-4:]
    assert added.available_at.eq(pd.Timestamp(native_time)).all()
    assert added.received_at.eq(pd.Timestamp(native_time)).all()


def test_revision_rejects_entire_poll_and_keeps_original_snapshot(tmp_path):
    directory, _source, _digest = seed(tmp_path)
    before_hash = file_hash(directory / 'bars.parquet')
    with pytest.raises(ValueError, match='revised'):
        refresh_feed(directory, ('BTCUSDT',), now=BASE + pd.Timedelta(hours=64, seconds=5),
            session=Session([Response([kline(59, close='100.5'), kline(63)])]))
    assert file_hash(directory / 'bars.parquet') == before_hash
    assert len(load_feed(directory)) == 60
    assert ExperimentJournal(directory / 'journal.jsonl').read()[-1]['event'] == 'poll_failed'


def test_retries_and_rate_limit_failures_are_bounded_without_background_retries(tmp_path):
    directory, _source, _digest = seed(tmp_path)
    now = BASE + pd.Timedelta(hours=64, seconds=5)
    client = Session([requests.Timeout('fixture timeout'), Response([kline(63)])])
    assert refresh_feed(directory, ('BTCUSDT',), now=now, session=client)['network_requests'] == 2
    limited = Session([Response([], 429), Response([kline(64)])])
    with pytest.raises(requests.HTTPError):
        refresh_feed(directory, ('BTCUSDT',), now=now, session=limited)
    assert len(limited.calls) == 1
    failed = Session([requests.Timeout('one'), requests.Timeout('two')])
    with pytest.raises(requests.Timeout):
        refresh_feed(directory, ('BTCUSDT',), now=now, session=failed)
    assert len(failed.calls) == 2


def test_page_budget_catchup_and_only_latest_new_bar_is_origin(tmp_path):
    directory, _source, _digest = seed(tmp_path, n=1)
    now = BASE + pd.Timedelta(hours=300, seconds=5)
    client = Session([Response([kline(i) for i in range(204, 300)]),
                      Response([kline(i) for i in range(108, 204)]),
                      Response([kline(i) for i in range(12, 108)])])
    result = refresh_feed(directory, ('BTCUSDT',), now=now, max_pages=2, session=client)
    assert result['network_requests'] == 2 and result['added_rows'] == 192
    assert result['audit']['missing_hours'] == 107
    added = load_feed(directory).query("availability_basis == 'actual_rest_receipt'")
    assert added.prediction_origin.sum() == 1
    assert added.loc[added.prediction_origin, 'open_time'].iloc[0] == BASE + pd.Timedelta(hours=299)
    assert client.calls[1]['params']['endTime'] < client.calls[0]['params']['endTime']


def test_committed_snapshot_repairs_old_mirror_after_interruption_but_rejects_tamper(tmp_path, monkeypatch):
    directory, _source, _digest = seed(tmp_path)
    copy = feed._atomic_copy
    old_hash = file_hash(directory / 'bars.parquet')

    def interrupted_copy(source, destination):
        if destination.name == 'bars.parquet':
            raise KeyboardInterrupt('fixture after journal commit')
        return copy(source, destination)

    monkeypatch.setattr(feed, '_atomic_copy', interrupted_copy)
    with pytest.raises(KeyboardInterrupt):
        refresh_feed(directory, ('BTCUSDT',), now=BASE + pd.Timedelta(hours=64, seconds=5),
            session=Session([Response([kline(63)])]))
    assert file_hash(directory / 'bars.parquet') == old_hash
    monkeypatch.setattr(feed, '_atomic_copy', copy)
    assert len(load_feed(directory)) == 61
    assert file_hash(directory / 'bars.parquet') != old_hash
    (directory / 'bars.parquet').write_bytes(b'edited outside journal')
    with pytest.raises(ValueError, match='mirror changed'):
        load_feed(directory)


@pytest.mark.parametrize('max_pages', [0, 5, True])
def test_resource_bounds_and_original_historical_cache_are_rejected(tmp_path, max_pages):
    with pytest.raises(ValueError, match='max_pages'):
        refresh_feed(tmp_path / 'unused', ('BTCUSDT',), max_pages=max_pages, session=Session([]))
    with pytest.raises(ValueError, match='historical 2025-hourly'):
        refresh_feed('data/discovery/2025-hourly', ('BTCUSDT',), session=Session([]))


def test_immutable_snapshot_and_feed_identity_edit_are_detected(tmp_path):
    directory, _source, _digest = seed(tmp_path)
    snapshot = next((directory / 'snapshots').glob('*.parquet'))
    snapshot.write_bytes(b'edited immutable source')
    with pytest.raises(ValueError, match='snapshot integrity'):
        load_feed(directory)


def test_duplicate_unordered_and_non_hourly_source_rows_are_rejected(tmp_path):
    directory, _source, _digest = seed(tmp_path)
    now = BASE + pd.Timedelta(hours=64, seconds=5)
    malformed = kline(63)
    malformed[6] += 1
    for payload in ([kline(63), kline(63)], [kline(63), kline(62)], [malformed]):
        with pytest.raises(ValueError):
            refresh_feed(directory, ('BTCUSDT',), now=now, session=Session([Response(payload)]))
    assert len(load_feed(directory)) == 60


def test_file_feed_builds_actual_origins_and_future_append_keeps_prior_visible_identity(tmp_path):
    directory, _source, _digest = seed(tmp_path, varied=True)
    spec = ExperimentSpec(tasks=(TaskSpec('ret', 'return', 2),), primary_task='ret',
        symbols=('BTCUSDT',), lookback=2, start=BASE.isoformat(),
        train_end=(BASE + pd.Timedelta(hours=60)).isoformat(),
        dev_end=(BASE + pd.Timedelta(hours=90)).isoformat(),
        calibration_end=(BASE + pd.Timedelta(hours=95)).isoformat(),
        end=(BASE + pd.Timedelta(hours=100)).isoformat())
    first_receipt = BASE + pd.Timedelta(hours=64, seconds=5)
    refresh_feed(directory, spec.symbols, now=first_receipt,
        session=Session([Response([kline(i) for i in range(60, 64)])]))
    before = build_task_dataset(load_feed(directory), spec)
    first = before.meta.available_at.eq(first_receipt).to_numpy()
    assert first.sum() == 1 and np.isnat(before.maturity['ret'][first]).all()
    second_receipt = BASE + pd.Timedelta(hours=68, seconds=5)
    refresh_feed(directory, spec.symbols, now=second_receipt,
        session=Session([Response([kline(i) for i in range(64, 68)])]))
    after = build_task_dataset(load_feed(directory), spec)
    previous = after.meta.available_at.eq(first_receipt).to_numpy()
    assert after.observed['ret'][previous].all()
    assert pd.Timestamp(after.maturity['ret'][previous][0], tz='UTC') == second_receipt
    assert _visible_hash(before, first_receipt) == _visible_hash(after, first_receipt)
    locations = after.meta.set_index('sample_id').index.get_indexer(before.meta.sample_id)
    np.testing.assert_array_equal(before.x, after.x[locations])
