"""Independent real-runtime service audits; synthetic bars and injected clocks only."""
import json

import pandas as pd
import pytest
from filelock import Timeout
from test_v7_forward_audit import _fit, _receive
from test_v7_forward_audit import runtime as forward_fixture
from test_v7_hypotheses import program

from factorlab.discovery import v7_lifecycle
from factorlab.discovery.joint_pool import FrozenPool, PoolMember
from factorlab.discovery.v7_service import ForwardService, HostGuard, ServiceConfig


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    yield from forward_fixture.__wrapped__(tmp_path, monkeypatch)


def service(runtime, **kwargs):
    store, _, _, _ = runtime
    return ForwardService(store, config=ServiceConfig(poll=False, **kwargs),
                          guard=lambda: {'allowed': True}, monotonic=lambda: 0.,
                          work_authority=store.test_authority)


def freeze_request(runtime):
    store, _, _, _ = runtime
    store.lifecycle.exposure.record(source=store.feed.identity, symbols=['BTCUSDT'],
        start='2026-07-01T00:00Z', end='2026-09-01T00:00Z', role='S', batch='audit')
    recipe = store.recipe_for()
    return {'batch': 'audit', 'ordinal': 1, 'start': '2026-10-01T00:00Z',
            'reference': recipe, 'candidates': [{**recipe, 'complexity': 0}],
            'evidence_sources': [{'start': '2026-07-01T00:00Z', 'end': '2026-09-01T00:00Z'}]}


def test_real_service_cold_fit_waits_for_next_receipt(runtime):
    store, clock, _, _ = runtime
    clock.now = pd.Timestamp('2026-10-01T00:00:10Z')
    runner = service(runtime)
    try:
        first = runner.step()
        assert first['tick']['new_predictions'] == 0
        assert len([a for a in first['actions'] if a['kind'] == 'fit']) == 1
        assert store.lifecycle.active_model('A0') is None
        _receive(runtime, '2026-09-30T23:00Z', '2026-10-01T01:00Z', '2026-10-01T01:00:05Z')
        assert runner.step()['tick']['new_predictions'] == 1
        assert store.lifecycle.active_model('A0') is not None
    finally:
        runner.close(close_runtime=False)


def test_verified_failure_must_be_reverified_by_service(runtime):
    store, clock, _, _ = runtime
    clock.now = pd.Timestamp('2026-10-01T00:00:10Z')
    fitted = _fit(store, '2026-10-01T00:00Z')
    store.lifecycle.fail_fit(fitted['job'], 'interrupted')
    runner = service(runtime)
    try:
        runner.step()
        assert store.lifecycle.db.execute('SELECT failure FROM jobs WHERE id=?', (fitted['job'],)).fetchone()[0] is None
    finally:
        runner.close(close_runtime=False)


def test_missing_checkpoint_with_existing_journal_cannot_reset_lifetime_budget(runtime):
    runner = service(runtime)
    runner.state['steps'] = 9
    runner.state['spent_seconds'] = 999.
    runner._save('synthetic_budget_used')
    runner.db.execute('DELETE FROM checkpoint')
    runner.db.commit()
    runner.close(close_runtime=False)
    with pytest.raises(ValueError, match='checkpoint|journal|integrity'):
        restarted = service(runtime)
        restarted.close(close_runtime=False)


def test_owner_lock_is_shared_across_distinct_sessions(runtime):
    runner = service(runtime, session_id='first')
    try:
        with pytest.raises(Timeout):
            other = service(runtime, session_id='second')
            other.close(close_runtime=False)
    finally:
        runner.close(close_runtime=False)


def test_interrupted_wall_time_is_charged_before_another_step(runtime):
    _, clock, _, _ = runtime
    runner = service(runtime, wall_seconds=60)
    runner.state['active_since'] = clock().isoformat()
    runner._save('simulated_process_loss')
    runner.close(close_runtime=False)
    clock.now += pd.Timedelta(seconds=61)
    restarted = service(runtime, wall_seconds=60)
    try:
        assert restarted.state['spent_seconds'] == 61
        assert restarted.step()['status'] == 'budget_exhausted'
    finally:
        restarted.close(close_runtime=False)


def test_missed_queued_freeze_is_not_backdated(runtime):
    store, clock, _, _ = runtime
    request = freeze_request(runtime)
    runner = service(runtime)
    try:
        runner.queue_batch(freeze_at='2026-09-30T23:59Z', **request)
        clock.now = pd.Timestamp('2026-10-01T00:00:05Z')
        actions = []
        runner._freeze_due(clock(), actions)
        assert actions[0]['status'] == 'missed_prospective_start'
        assert store.lifecycle.db.execute('SELECT COUNT(*) FROM batches').fetchone()[0] == 0
    finally:
        runner.close(close_runtime=False)


def test_one_look_published_result_recovers_without_reading_validation_again(runtime, monkeypatch):
    store, clock, _, _ = runtime
    store.lifecycle.freeze_batch(**freeze_request(runtime))
    clock.now = pd.Timestamp('2026-11-01T00:00:05Z')
    original = v7_lifecycle.immutable_record

    class Crash(BaseException):
        pass

    def fail_after_publish(path, value):
        result = original(path, value)
        if path.parent.name == 'validation':
            raise Crash
        return result

    monkeypatch.setattr(v7_lifecycle, 'immutable_record', fail_after_publish)
    with pytest.raises(Crash):
        store.lifecycle.select_validation('audit')
    artifacts = list((store.directory / 'validation').glob('*.json'))
    assert len(artifacts) == 1
    before = json.loads(artifacts[0].read_text('utf-8'))
    monkeypatch.setattr(v7_lifecycle, 'immutable_record', original)

    def no_second_look(*args, **kwargs):
        raise AssertionError('A published one-look result must not rescore V')

    monkeypatch.setattr(store.lifecycle, '_losses', no_second_look)
    runner = service(runtime)
    try:
        runner._endpoints(clock(), [])
        assert store.lifecycle.db.execute('SELECT status FROM batches WHERE id=?', ('audit',)).fetchone()[0] == 'concluded_no_candidate'
        assert json.loads(artifacts[0].read_text('utf-8')) == before
    finally:
        runner.close(close_runtime=False)


def test_host_guard_only_observes_and_blocks_before_any_runtime_work(runtime, monkeypatch):
    store, _, _, _ = runtime

    def forbidden(*args, **kwargs):
        raise AssertionError('Blocked service must not poll or fit')

    monkeypatch.setattr(store, 'tick', forbidden)
    monkeypatch.setattr(store, 'fit_month', forbidden)
    runner = ForwardService(store, config=ServiceConfig(poll=False), guard=HostGuard(inspect=lambda: {
        'rss_mib': 10., 'available_mib': 10000.,
        'processes': [{'pid': 123, 'command': 'python /research-long80/control.py'}]}))
    try:
        result = runner.step()
        assert result['status'] == 'blocked_by_host_guard'
        assert result['guard']['blocking_pids'] == [123]
        assert result['guard']['hard_limits_verified'] is False
        assert runner.state['steps'] == 0
    finally:
        runner.close(close_runtime=False)


def test_three_protected_candidates_keep_frozen_recipes_and_common_ready_support(runtime):
    store, clock, history, _ = runtime
    request = freeze_request(runtime)
    request['candidates'] = [{**store.recipe_for(FrozenPool(members=(PoolMember(
        program({'op': 'const', 'value': float(i)})),), role='confirmation')), 'complexity': 1}
        for i in (1, 2, 3)]
    store.lifecycle.freeze_batch(**request)
    runner = service(runtime, max_fits_per_step=1)
    try:
        clock.now = pd.Timestamp('2026-10-01T00:00:10Z')
        runner.step()
        for hour in range(1, 6):
            origin = pd.Timestamp('2026-10-01T00:00:05Z') + pd.Timedelta(hours=hour)
            start = origin.floor('h') - pd.Timedelta(hours=1)
            clock.now = origin
            begin = pd.Timestamp('2026-09-30T23:00Z') if hour == 1 else start
            frame = history.loc[(history.open_time >= begin) & (history.open_time <= start)]
            store.feed.receive(frame, received_at=origin, response_hash='c' * 64)
            runner.step()
        jobs = list(store.lifecycle.db.execute('SELECT id,stream,pipeline,completed,activation FROM jobs'))
        assert len(jobs) == 5  # A0 plus protected reference and three frozen candidates.
        assert all(runner.state['fit_attempts'][job]['count'] == 1 for job, *_ in jobs)
        protected = [row for row in jobs if row[1] != 'A0']
        expected = {f'audit:{c["pipeline_id"]}': c['pipeline_id'] for c in request['candidates']}
        expected['audit:reference'] = request['reference']['pipeline_id']
        by_stream = {}
        for _, stream, recipe, completed, activated in protected:
            assert json.loads(recipe)['pipeline_id'] == expected[stream]
            assert pd.Timestamp(activated) >= pd.Timestamp(completed)
            by_stream[stream] = {row['origin'] for row in store.lifecycle.forecasts.iter_records()
                                 if row['stream'] == stream}
        common = set.intersection(*by_stream.values())
        assert common
        assert min(map(pd.Timestamp, common)) >= max(pd.Timestamp(row[3]) for row in protected)
        assert not store.lifecycle.calibration_ready()
    finally:
        runner.close(close_runtime=False)
