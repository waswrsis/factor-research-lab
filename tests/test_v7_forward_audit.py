"""Independent synthetic forward-clock integration tests; no network or trading."""
import json
import os
from datetime import UTC, datetime

import pandas as pd
import pytest
from test_v7_feed_audit import Clock
from test_v7_rolling_registry import candles
from test_v7_run_factor_pool import source_root

from factorlab.discovery import (
    model_registry,
    neural_forward_ticket,
    pool_forward,
    v7_forward,
    work_lease,
)
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    clock = Clock('2026-09-30T23:30:00Z')
    result = v7_forward.ForwardRuntime(tmp_path / 'run', tmp_path / 'feed', clock=clock)
    history = candles('2026-05-01', '2026-11-04')
    path = tmp_path / 'seed.parquet'
    history.loc[history.open_time < '2026-09-30T23:00Z'].to_parquet(path)
    result.feed.import_seed(path)
    _attach_work(result, tmp_path, monkeypatch, clock)
    yield result, clock, history, tmp_path
    result.close()



def _attach_work(store, root, monkeypatch, clock):
    """Real shared wallet; only the Linux observation is artificial."""
    store.test_authority = MonthlyWorkAuthority(root / 'authority', run_id='forward-audit',
        task='volatility', universe=list(store.config.symbols))
    boot = 'd43d6a03-a96b-47b0-8e7d-e8d1bb65eaba'
    monkeypatch.setattr(work_lease, '_boot_id', lambda: 'linux:' + boot)
    monkeypatch.setattr(neural_forward_ticket, '_now', clock)

    def kernel():
        return {'path': '/sys/fs/cgroup/artificial.service', 'inode': 321, 'boot_id': boot,
            'scope': 'whole_cgroup', 'platform': 'linux', 'peak': 100 * 2**20,
            'events': {'oom': 0, 'oom_kill': 0}, 'observed_at': datetime.now(UTC).isoformat(),
            'raw': {'memory.max': str(768 * 2**20), 'memory.high': str(640 * 2**20),
                'memory.swap.max': '0', 'pids.max': '64', 'cpu.max': '25000 100000',
                'memory.peak': str(100 * 2**20), 'memory.events': 'oom 0\noom_kill 0',
                'cgroup.procs': str(os.getpid())}}
    monkeypatch.setattr(pool_forward, '_kernel_snapshot', kernel)


def _fit(store, cutoff):
    return store.fit_month(cutoff, authority=store.test_authority)

def _receive(runtime, begin, end, received):
    store, clock, history, _ = runtime
    clock.now = pd.Timestamp(received)
    frame = history.loc[(history.open_time >= pd.Timestamp(begin)) & (history.open_time < pd.Timestamp(end))]
    return store.feed.receive(frame, received_at=clock(), response_hash='a' * 64)


def _ready_and_forecast(runtime):
    store, clock, _, _ = runtime
    clock.now = pd.Timestamp('2026-10-01T00:00:10Z')
    fit = _fit(store, '2026-10-01T00:00Z')
    _receive(runtime, '2026-09-30T23:00Z', '2026-10-01T01:00Z', '2026-10-01T01:00:05Z')
    tick = store.tick()
    assert tick['new_predictions'] == 1 and not tick['formal_gate_ready']
    return fit


def test_seed_import_before_cutoff_can_fit_but_late_import_cannot(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    clock = Clock('2026-10-01T00:00:10Z')
    store = v7_forward.ForwardRuntime(tmp_path / 'run', tmp_path / 'feed', clock=clock)
    try:
        path = tmp_path / 'late-seed.parquet'
        candles('2026-05-01', '2026-10-01').to_parquet(path)
        store.feed.import_seed(path)
        _attach_work(store, tmp_path, monkeypatch, clock)
        with pytest.raises(ValueError, match='No source|snapshot|received'):
            _fit(store, '2026-10-01T00:00Z')
        assert store.lifecycle.db.execute('SELECT failure FROM jobs').fetchone()[0] == 'data_incomplete'
    finally:
        store.close()


def test_fit_receipt_uses_three_month_window_and_cached_model_is_verified(runtime):
    store, clock, _, _ = runtime
    clock.now = pd.Timestamp('2026-10-01T00:00:10Z')
    first = _fit(store, '2026-10-01T00:00Z')
    repeated = _fit(store, '2026-10-01T00:00Z')
    assert first['train']['train_start'] == '2026-07-01T00:00:00+00:00'
    assert first['train']['cutoff'] == '2026-10-01T00:00:00+00:00'
    assert first['model_id'] == repeated['model_id'] and repeated['cached']
    artifact = store.directory / 'models' / first['model_id'] / 'model.json'
    artifact.write_text('{}', encoding='utf-8')
    with pytest.raises((ValueError, KeyError), match='.*'):
        _fit(store, '2026-10-01T00:00Z')


def test_archive_source_copy_is_verified_on_reopen(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    clock = Clock()
    store = v7_forward.ForwardRuntime(tmp_path / 'run', tmp_path / 'feed', clock=clock)
    store.close()
    (tmp_path / 'run' / 'source' / 'factorlab' / 'v7.py').write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='source|archive|integrity'):
        reopened = v7_forward.ForwardRuntime(tmp_path / 'run', tmp_path / 'feed', clock=clock)
        reopened.close()


def test_current_source_cannot_change_on_restart(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    clock = Clock()
    store = v7_forward.ForwardRuntime(tmp_path / 'run', tmp_path / 'feed', clock=clock)
    store.close()
    (tmp_path / 'factorlab' / 'v7.py').write_text('version=8', encoding='utf-8')
    with pytest.raises(ValueError, match='Immutable|source|identity'):
        v7_forward.ForwardRuntime(tmp_path / 'run', tmp_path / 'feed', clock=clock)


def test_interrupted_convex_fit_remains_consumed_without_new_fit(runtime, monkeypatch):
    store, clock, _, _ = runtime
    clock.now = pd.Timestamp('2026-10-01T00:00:10Z')
    original = pool_forward.fit_pool

    class Crash(BaseException):
        pass

    def interrupted(*args, **kwargs):
        original(*args, **kwargs)
        raise Crash

    monkeypatch.setattr(pool_forward, 'fit_pool', interrupted)
    with pytest.raises(Crash):
        _fit(store, '2026-10-01T00:00Z')
    assert store.lifecycle.db.execute('SELECT status FROM jobs').fetchone()[0] == 'features_ready'
    charged = store.test_authority.verify()
    monkeypatch.setattr(pool_forward, 'fit_pool', lambda *a, **k: pytest.fail('Consumed work refitted'))
    with pytest.raises(ValueError, match='already attempted|consumed'):
        _fit(store, '2026-10-01T00:00Z')
    assert store.test_authority.verify() == charged
    assert store.lifecycle.db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == 1
    assert store.lifecycle.db.execute('SELECT model FROM jobs').fetchone()[0] is None


def test_fit_completion_waits_for_next_actual_origin_and_duplicates_do_not_reissue(runtime, monkeypatch):
    store, clock, _, _ = runtime
    _receive(runtime, '2026-09-30T23:00Z', '2026-10-01T00:00Z', '2026-10-01T00:00:05Z')
    clock.now = pd.Timestamp('2026-10-01T00:00:10Z')
    original = pool_forward.fit_pool

    def slow_fit(*args, **kwargs):
        result = original(*args, **kwargs)
        clock.now = pd.Timestamp('2026-10-01T00:02:00Z')
        return result

    monkeypatch.setattr(pool_forward, 'fit_pool', slow_fit)
    _fit(store, '2026-10-01T00:00Z')
    assert store.tick()['new_predictions'] == 0
    _receive(runtime, '2026-10-01T00:00Z', '2026-10-01T01:00Z', '2026-10-01T01:00:05Z')
    assert store.tick()['new_predictions'] == 1
    row = next(store.lifecycle.forecasts.iter_records())
    payload = json.loads(row['prediction'])
    assert payload['forecast_window_start'] == '2026-10-01T02:00:00+00:00'
    clock.now += pd.Timedelta(seconds=1)
    assert store.tick()['new_predictions'] == 0
    assert list(store.lifecycle.forecasts.iter_records()) == [row]


def test_late_required_candle_delays_full_twelve_hour_label_without_old_forecast(runtime):
    store, clock, history, _ = runtime
    _ready_and_forecast(runtime)
    clock.now = pd.Timestamp('2026-10-01T14:00:05Z')
    required = history.loc[(history.open_time >= '2026-10-01T01:00Z') &
                           (history.open_time < '2026-10-01T14:00Z') &
                           (history.open_time != '2026-10-01T05:00Z')]
    store.feed.receive(required, received_at=clock(), response_hash='b' * 64)
    assert store.tick()['new_settlements'] == 0
    _receive(runtime, '2026-10-01T05:00Z', '2026-10-01T06:00Z', '2026-10-01T15:00:05Z')
    result = store.tick()
    assert result['new_settlements'] == 1 and result['new_predictions'] == 0
    settlement = json.loads(store.lifecycle.db.execute('SELECT record FROM settlement').fetchone()[0])
    assert settlement['maturity'] == clock().isoformat()
    assert store.tick()['new_settlements'] == 0


def test_assumed_seed_cannot_settle_actual_forward_labels(runtime):
    store, clock, history, root = runtime
    _ready_and_forecast(runtime)
    clock.now = pd.Timestamp('2026-10-01T14:00:05Z')
    path = root / 'label-seed.parquet'
    history.loc[(history.open_time >= '2026-10-01T01:00Z') &
                (history.open_time < '2026-10-01T14:00Z')].to_parquet(path)
    store.feed.import_seed(path)
    assert store.tick()['new_settlements'] == 0


def test_job_completed_index_cannot_override_recorded_fit_completion(runtime):
    store, clock, _, _ = runtime
    clock.now = pd.Timestamp('2026-10-01T00:02:00Z')
    _fit(store, '2026-10-01T00:00Z')
    store.lifecycle.db.execute('UPDATE jobs SET completed=?', ('2026-10-01T00:00:00+00:00',))
    store.lifecycle.db.commit()
    with pytest.raises(ValueError, match='integrity|index|state|job|completed'):
        store.status()


def test_verified_failed_job_is_reverified_on_controller_retry(runtime):
    store, clock, _, _ = runtime
    clock.now = pd.Timestamp('2026-10-01T00:00:10Z')
    fitted = _fit(store, '2026-10-01T00:00Z')
    store.lifecycle.fail_fit(fitted['job'], 'interrupted')
    clock.now += pd.Timedelta(seconds=1)
    recovered = _fit(store, '2026-10-01T00:00Z')
    assert recovered['model_id'] == fitted['model_id']
    assert store.lifecycle.db.execute('SELECT failure FROM jobs WHERE id=?', (fitted['job'],)).fetchone()[0] is None


def test_superseded_unactivated_month_does_not_interrupt_current_month(runtime):
    store, clock, history, root = runtime
    clock.now = pd.Timestamp('2026-10-01T00:00:10Z')
    old = _fit(store, '2026-10-01T00:00Z')
    # No live origin happened while this first model was ready.
    clock.now = pd.Timestamp('2026-10-31T23:30:00Z')
    path = root / 'october-seed.parquet'
    history.loc[(history.open_time >= '2026-09-30T23:00Z') &
                (history.open_time < '2026-10-31T23:00Z')].to_parquet(path)
    store.feed.import_seed(path)
    clock.now = pd.Timestamp('2026-11-01T00:00:10Z')
    new = _fit(store, '2026-11-01T00:00Z')
    assert old['model_id'] != new['model_id']
    _receive(runtime, '2026-10-31T23:00Z', '2026-11-01T01:00Z', '2026-11-01T01:00:05Z')
    assert store.tick()['new_predictions'] == 1
    assert store.lifecycle.active_model('A0')['model_id'] == new['model_id']
    _receive(runtime, '2026-11-01T01:00Z', '2026-11-01T02:00Z', '2026-11-01T02:00:05Z')
    assert store.tick()['new_predictions'] == 1
    assert store.lifecycle.active_model('A0')['model_id'] == new['model_id']


@pytest.mark.parametrize('field', ['prediction', 'origin'])
def test_original_forecast_content_and_logical_index_are_verified(runtime, field):
    store, _, _, _ = runtime
    _ready_and_forecast(runtime)
    db = store.lifecycle.forecasts.db
    if field == 'prediction':
        payload = json.loads(db.execute('SELECT prediction FROM forecasts').fetchone()[0])
        payload['value'] *= 2
        db.execute('UPDATE forecasts SET prediction=?', (json.dumps(payload, sort_keys=True),))
    else:
        db.execute('UPDATE forecasts SET origin=?', ('2026-10-01T00:00:05+00:00',))
    db.commit()
    with pytest.raises((ValueError, FileNotFoundError)):
        store.status()


def test_late_backfill_is_not_a_fresh_latest_closed_candle_forecast(runtime):
    store, clock, _, _ = runtime
    clock.now = pd.Timestamp('2026-10-01T00:00:10Z')
    _fit(store, '2026-10-01T00:00Z')
    _receive(runtime, '2026-09-30T23:00Z', '2026-10-01T00:00Z', '2026-10-01T04:00:05Z')
    tick = store.tick()
    assert tick['new_predictions'] == 0


def test_previous_month_continues_until_new_fit_can_use_next_receipt(runtime, monkeypatch):
    store, clock, history, root = runtime
    old = _ready_and_forecast(runtime)
    clock.now = pd.Timestamp('2026-10-31T23:30:00Z')
    path = root / 'next-month-seed.parquet'
    history.loc[(history.open_time >= '2026-10-01T01:00Z') &
                (history.open_time < '2026-10-31T23:00Z')].to_parquet(path)
    store.feed.import_seed(path)
    _receive(runtime, '2026-10-31T23:00Z', '2026-11-01T00:00Z', '2026-11-01T00:00:05Z')
    original = pool_forward.fit_pool

    def slow_fit(*args, **kwargs):
        result = original(*args, **kwargs)
        clock.now = pd.Timestamp('2026-11-01T00:02:00Z')
        return result

    clock.now = pd.Timestamp('2026-11-01T00:00:10Z')
    monkeypatch.setattr(pool_forward, 'fit_pool', slow_fit)
    new = _fit(store, '2026-11-01T00:00Z')
    assert store.tick()['new_predictions'] == 1
    assert store.lifecycle.active_model('A0')['model_id'] == old['model_id']
    _receive(runtime, '2026-11-01T00:00Z', '2026-11-01T01:00Z', '2026-11-01T01:00:05Z')
    assert store.tick()['new_predictions'] == 1
    assert store.lifecycle.active_model('A0')['model_id'] == new['model_id']


@pytest.mark.parametrize('restart', [False, True])
def test_forecast_receipt_written_before_sql_insert_recovers_without_reissue(runtime, monkeypatch, restart):
    store, clock, _, _ = runtime
    clock.now = pd.Timestamp('2026-10-01T00:00:10Z')
    _fit(store, '2026-10-01T00:00Z')
    _receive(runtime, '2026-09-30T23:00Z', '2026-10-01T01:00Z', '2026-10-01T01:00:05Z')
    original = model_registry.immutable_json

    class Crash(BaseException):
        pass

    def interrupted(path, value):
        result = original(path, value)
        if path.parent.name == 'forecasts.records':
            raise Crash
        return result

    monkeypatch.setattr(model_registry, 'immutable_json', interrupted)
    with pytest.raises(Crash):
        store.tick()
    receipts = list((store.directory / 'forecasts.records').glob('*.json'))
    assert len(receipts) == 1
    saved = json.loads(receipts[0].read_text('utf-8'))['receipt']
    monkeypatch.setattr(model_registry, 'immutable_json', original)
    clock.now += pd.Timedelta(hours=2) if restart else pd.Timedelta(seconds=1)
    if restart:
        directory, feed_directory = store.directory, store.feed.directory
        store.close()
        store = v7_forward.ForwardRuntime(directory, feed_directory, clock=clock)
    try:
        store.tick()
        rows = list(store.lifecycle.forecasts.iter_records())
        assert len(rows) == 1
        assert json.loads(rows[0]['prediction']) == saved['prediction']
        store.status()
    finally:
        if restart:
            store.close()
