"""Scheduler tests use genuine lifecycle persistence and tiny synthetic models."""
import json

import numpy as np
import pandas as pd
import pytest
from filelock import Timeout

from factorlab.discovery.convex_eval import fit_ridge
from factorlab.discovery.joint_pool import FrozenPool
from factorlab.discovery.rolling import RollingSpec
from factorlab.discovery.v7_lifecycle import Lifecycle, LifecyclePlan
from factorlab.discovery.v7_service import ForwardService, HostGuard, ServiceConfig
from factorlab.protocol import fingerprint


class Clock:
    def __init__(self):
        self.now = pd.Timestamp('2026-10-01T00:00:05Z')
        self.seconds = 0.
        self.sleeps = []

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.seconds += seconds
        self.now += pd.Timedelta(seconds=seconds)


class Runtime:
    """No market feed, network, or GPT. Activation mimics next-receipt ownership."""
    def __init__(self, directory, clock):
        self.directory, self.clock = directory, clock
        # Explicitly artificial standalone identity. Registered-controller
        # tests use the genuine ForwardRuntime; this fixture has no permit.
        self.identity = {'artificial_scheduler_fixture': True}
        self.spec = RollingSpec()
        self.lifecycle = Lifecycle(directory, LifecyclePlan('synthetic-feed'), clock=lambda: clock.now)
        self.calls = []
        self.missing = False

    def recipe_for(self, pool=None):
        pool = pool or FrozenPool(role='reference', version='K0_v1')
        return {'pipeline_id': fingerprint({'pool_template': pool.pipeline_id,
            'rolling_hash': self.spec.digest, 'ridge': self.spec.ridge, 'weighting': 'equal_symbol_then_hour'}),
            'pool': pool.to_dict(), 'task': 'volatility', 'rolling_hash': self.spec.digest}

    def tick(self, **kwargs):
        self.calls.append(('tick', kwargs))
        for job, stream in self.lifecycle.db.execute('SELECT id,stream FROM jobs WHERE status=? AND failure IS NULL', ('verified',)).fetchall():
            active = self.lifecycle.active_model(stream)
            self.lifecycle.activate(job, self.clock.now, expected_generation=active['generation'] if active else 0)
        return {'new_predictions': 0, 'actual_feed': False}

    def fit_month(self, cutoff, *, stream='A0', pool=None):
        self.calls.append(('fit', cutoff, stream))
        job = self.lifecycle.plan_fit(cutoff, stream, self.recipe_for(pool))
        if self.missing:
            self.lifecycle.fail_fit(job, 'data_incomplete')
            raise ValueError('Synthetic missing three-month receipt history')
        row = self.lifecycle.db.execute('SELECT status FROM jobs WHERE id=?', (job,)).fetchone()
        if row[0] == 'planned':
            self.lifecycle.advance_fit(job, 'snapshot_sealed')
            self.lifecycle.advance_fit(job, 'features_ready')
        model = fit_ridge(np.arange(30).reshape(-1, 1), np.linspace(.01, .03, 30))
        identifier = self.lifecycle.seal_model(job, model)
        return {'job': job, 'model_id': identifier}

    def close(self):
        self.lifecycle.close()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    clock = Clock()
    runtime = Runtime(tmp_path / 'forward', clock)
    yield runtime, clock
    runtime.close()


def service(runtime, clock, **kwargs):
    return ForwardService(runtime, guard=lambda: {'allowed': True},
                          monotonic=lambda: clock.seconds, sleeper=clock.sleep, **kwargs)


def test_tick_precedes_cold_fit_and_only_next_tick_activates_with_adoption_authority(setup, monkeypatch):
    runtime, clock = setup
    original = runtime.lifecycle.adoption_pipeline

    def choose(cutoff, k0):
        runtime.calls.append(('adoption', cutoff))
        return original(cutoff, k0)

    monkeypatch.setattr(runtime.lifecycle, 'adoption_pipeline', choose)
    runner = service(runtime, clock)
    try:
        first = runner.step()
        assert [r[0] for r in runtime.calls] == ['tick', 'adoption', 'fit']
        assert first['actions'][0]['status'] == 'verified_waiting_for_receipt'
        assert runtime.lifecycle.active_model('A0') is None
        clock.sleep(1)
        runner.step()
        assert runtime.lifecycle.active_model('A0') is not None
        assert sum(c[0] == 'fit' for c in runtime.calls) == 1
    finally:
        runner.close(close_runtime=False)


def test_missing_new_month_keeps_old_model_and_persists_failure_quota(setup):
    runtime, clock = setup
    config = ServiceConfig(max_fit_attempts=1)
    runner = service(runtime, clock, config=config)
    runner.step()
    clock.sleep(1)
    runner.step()
    previous = runtime.lifecycle.active_model('A0')['model_id']
    clock.now = pd.Timestamp('2026-11-01T00:00:05Z')
    runtime.missing = True
    result = runner.step()
    assert result['actions'][0]['status'] == 'data_incomplete'
    assert runtime.lifecycle.active_model('A0')['model_id'] == previous
    runner.close(close_runtime=False)
    restarted = service(runtime, clock, config=config)
    try:
        count = sum(c[0] == 'fit' for c in runtime.calls)
        clock.sleep(120)
        restarted.step()
        assert sum(c[0] == 'fit' for c in runtime.calls) == count
        assert runtime.lifecycle.active_model('A0')['model_id'] == previous
    finally:
        restarted.close(close_runtime=False)


def test_guard_and_stop_do_not_poll_fit_or_touch_legacy_processes(setup):
    runtime, _clock = setup
    inspection = lambda: {'rss_mib': 20, 'available_mib': 8000,
        'processes': [{'pid': 101, 'command': 'python /work/research-long80/control.py'}]}
    runner = ForwardService(runtime, guard=HostGuard(inspect=inspection))
    try:
        assert runner.step()['status'] == 'blocked_by_host_guard'
        assert runtime.calls == []
        (runner.directory / 'STOP').write_text('operator stop', encoding='utf-8')
        assert runner.step()['status'] == 'stopped'
        assert runtime.calls == []
    finally:
        runner.close(close_runtime=False)


def test_bounded_run_sleeps_and_lifetime_budget_survives_restart(setup):
    runtime, clock = setup
    config = ServiceConfig(wall_seconds=3, tick_seconds=2, max_steps=10)
    runner = service(runtime, clock, config=config)
    result = runner.run()
    # The immutable native origin also charges actual setup/step IO. The
    # lifetime budget therefore expires before the injected invocation timer;
    # the second sleep must be shortened, never refunding that real overhead.
    assert result['status'] == 'budget_exhausted'
    assert len(clock.sleeps) == 2 and clock.sleeps[0] == 2
    assert 0 < clock.sleeps[1] < 1
    assert runner.state['steps'] == 2 and runner.state['spent_seconds'] == pytest.approx(3)
    runner.close(close_runtime=False)
    restarted = service(runtime, clock, config=config)
    try:
        assert restarted.step()['status'] == 'budget_exhausted'
    finally:
        restarted.close(close_runtime=False)


def test_prospective_batch_freeze_does_not_execute_before_real_clock(setup):
    runtime, clock = setup
    runner = service(runtime, clock)
    recipe = runtime.recipe_for()
    runtime.lifecycle.exposure.record(source='synthetic-feed', symbols=['BTCUSDT'],
        start='2026-07-01T00:00Z', end='2026-10-01T00:00Z', role='S', batch='A')
    request = {'batch': 'A', 'ordinal': 1, 'start': '2026-11-01T00:00:00Z',
        'reference': recipe, 'candidates': [{**recipe, 'complexity': 1}],
        'evidence_sources': [{'start': '2026-07-01T00:00:00Z', 'end': '2026-10-01T00:00:00Z'}]}
    try:
        runner.queue_batch(freeze_at='2026-10-02T00:00Z', **request)
        runner.step()
        assert runtime.lifecycle.db.execute('SELECT COUNT(*) FROM batches').fetchone()[0] == 0
        clock.now = pd.Timestamp('2026-10-02T00:00Z')
        runner.step()
        assert runtime.lifecycle._batch('A')[1] == 'frozen_V'
        assert runtime.lifecycle.db.execute('SELECT COUNT(*) FROM jobs WHERE stream LIKE ?', ('A:%',)).fetchone()[0] == 0
        clock.now = pd.Timestamp('2026-11-01T00:00:05Z')
        runner.step()
        rows = runtime.lifecycle.db.execute('SELECT pipeline FROM jobs WHERE stream LIKE ?', ('A:%',)).fetchall()
        assert len(rows) == 2
        assert all(json.loads(row[0])['pipeline_id'] == recipe['pipeline_id'] for row in rows)
        clock.sleep(1)
        result = runner.step()
        assert any(a.get('stream', '').startswith('A:') and
                   a['status'] == 'verified_waiting_for_receipt' for a in result['actions'])
    finally:
        runner.close(close_runtime=False)


def test_uncalibrated_selected_candidate_waits_without_reserving_formal_C(setup, monkeypatch):
    runtime, clock = setup
    runner = service(runtime, clock)
    recipe = runtime.recipe_for()
    runtime.lifecycle.exposure.record(source='synthetic-feed', symbols=['BTCUSDT'],
        start='2026-07-01T00:00Z', end='2026-10-01T00:00Z', role='S', batch='A')
    runtime.lifecycle.freeze_batch(batch='A', ordinal=1, start='2026-11-01T00:00Z',
        candidates=[{**recipe, 'complexity': 1}], reference=recipe,
        evidence_sources=[{'start': '2026-07-01T00:00Z', 'end': '2026-10-01T00:00Z'}])
    def losses(stream, start, end, *args, **kwargs):
        loss = 1. if stream.endswith(':reference') else .9
        return {(t.isoformat(), 'BTCUSDT'): (loss, .03, 'same') for t in
                pd.date_range(start, end, freq='h', inclusive='left')}
    monkeypatch.setattr(runtime.lifecycle, '_losses', losses)
    clock.now = pd.Timestamp('2026-12-01T00:00:05Z')
    try:
        result = runner.step()
        assert any(a.get('status') == 'waiting_for_calibration' for a in result['actions'])
        assert runtime.lifecycle.exposure.db.execute('SELECT COUNT(*) FROM confirmations').fetchone()[0] == 0
        assert runtime.lifecycle._batch('A')[1] == 'selected_C'
    finally:
        runner.close(close_runtime=False)


@pytest.mark.parametrize('observation', [
    {'rss_mib': 900, 'available_mib': 9000, 'processes': []},
    {'rss_mib': float('nan'), 'available_mib': 9000, 'processes': []},
    {'rss_mib': 20, 'available_mib': 100, 'processes': []},
])
def test_resource_guard_rejects_exhausted_or_invalid_observations(observation):
    result = HostGuard(inspect=lambda: observation)()
    assert result['allowed'] is False and result['hard_limits_verified'] is False


@pytest.mark.parametrize('status', ['running','stopped','zombie'])
def test_actual_host_collector_never_treats_unreadable_stopped_python_as_dead(monkeypatch,status):
    import psutil
    observed_status=status
    class Process:
        info={'pid':404,'cmdline':None,'name':'python.exe','status':observed_status}
        def status(self):
            return status
    monkeypatch.setattr(psutil,'process_iter',lambda fields:iter([Process()]))
    if status=='zombie':
        # Only the collector's confirmed zombie path is harmless. This synthetic
        # status does not declare the actual Windows opaque process terminated.
        assert HostGuard._inspect()['processes']==[]
    else:
        result=HostGuard()()
        assert result['allowed'] is False
        assert result['reasons']==['host_inspection_failed']
        assert result['unreadable_process']=={'pid':404,'status':status}


def test_actual_host_collector_ignores_process_confirmed_gone_during_inspection(monkeypatch):
    import psutil
    class Process:
        info={'pid':404,'cmdline':None,'name':'python.exe','status':None}
        def status(self):
            raise psutil.NoSuchProcess(404)
    monkeypatch.setattr(psutil,'process_iter',lambda fields:iter([Process()]))
    assert HostGuard._inspect()['processes']==[]


def test_parallel_session_cannot_compete_for_same_runtime_and_crash_time_is_charged(setup):
    runtime, clock = setup
    runner = service(runtime, clock)
    with pytest.raises(Timeout):
        service(runtime, clock, config=ServiceConfig(session_id='other'))
    # A durable started marker with no finishing write is exactly what a hard
    # interruption leaves; restarting charges the unobserved interval.
    runner.state['active_since'] = clock.now.isoformat()
    runner._save('synthetic_interruption')
    runner.close(close_runtime=False)
    clock.sleep(35)
    restarted = service(runtime, clock)
    try:
        assert restarted.state['spent_seconds'] == 35
        assert restarted.state['active_since'] is None
    finally:
        restarted.close(close_runtime=False)


def test_service_only_uses_frozen_validation_recipe_and_rejects_self_reported_adoption(setup, monkeypatch):
    runtime, clock = setup
    runner = service(runtime, clock)
    invalid = {**runtime.recipe_for(), 'pipeline_id': 'self-reported-winner'}
    monkeypatch.setattr(runtime.lifecycle, 'adoption_pipeline', lambda *a: invalid)
    try:
        with pytest.raises(ValueError, match='Frozen recipe'):
            runner.step()
        assert all(call[0] != 'fit' for call in runtime.calls)
        assert runtime.lifecycle.active_model('A0') is None
    finally:
        runner.close(close_runtime=False)


def test_confirmation_scheduler_obeys_reserved_future_endpoint(setup, monkeypatch):
    runtime, clock = setup
    runner = service(runtime, clock)
    recipe = runtime.recipe_for()
    runtime.lifecycle.exposure.record(source='synthetic-feed', symbols=['BTCUSDT'],
        start='2026-07-01T00:00Z', end='2026-10-01T00:00Z', role='S', batch='A')
    runtime.lifecycle.freeze_batch(batch='A', ordinal=1, start='2026-11-01T00:00Z',
        candidates=[{**recipe, 'complexity': 1}], reference=recipe,
        evidence_sources=[{'start': '2026-07-01T00:00Z', 'end': '2026-10-01T00:00Z'}])

    def losses(stream, start, end, *args, **kwargs):
        loss = 1. if stream.endswith(':reference') else .9
        return {(t.isoformat(), 'BTCUSDT'): (loss, .03, 'same') for t in
                pd.date_range(start, end, freq='h', inclusive='left')}

    monkeypatch.setattr(runtime.lifecycle, '_losses', losses)
    # Fake certificate and score explicitly isolate scheduling, not inference.
    monkeypatch.setattr(runtime.lifecycle, 'calibration_ready', lambda: True)
    scored = []
    monkeypatch.setattr('factorlab.discovery.v7_lifecycle.paired_effect_interval',
        lambda *args, **kwargs: scored.append(kwargs) or {'supported': False, 'status': 'unresolved'})
    clock.now = pd.Timestamp('2026-12-01T00:00:05Z')
    try:
        runner.step()
        identity = json.loads(runtime.lifecycle.exposure.db.execute(
            'SELECT identity FROM confirmations WHERE batch=?', ('A',)).fetchone()[0])
        assert identity['actual_start'] == '2026-12-01T01:00:00+00:00'
        assert not scored
        clock.now = pd.Timestamp(identity['scoring_cutoff']) - pd.Timedelta(seconds=1)
        runner.step()
        assert not scored
        clock.now += pd.Timedelta(seconds=1)
        runner.step()
        assert len(scored) == 1 and scored[0]['eligible_count'] == 90 * 24
        assert runtime.lifecycle._batch('A')[1] == 'concluded'
        runner.step()
        assert len(scored) == 1
    finally:
        runner.close(close_runtime=False)


def test_immediate_queue_uses_one_clock_read_for_prospective_comparison(setup, monkeypatch):
    runtime, clock = setup
    runner = service(runtime, clock)
    recipe = runtime.recipe_for()

    def advancing_now():
        clock.now += pd.Timedelta(microseconds=1)
        return clock.now

    monkeypatch.setattr(runtime.lifecycle, 'now', advancing_now)
    try:
        assert runner.queue_batch(batch='A', ordinal=1, start='2026-11-01T00:00:00Z',
            candidates=[{**recipe, 'complexity': 1}], reference=recipe,
            evidence_sources=[{'start': '2026-07-01T00:00Z', 'end': '2026-10-01T00:00Z'}])
    finally:
        runner.close(close_runtime=False)
