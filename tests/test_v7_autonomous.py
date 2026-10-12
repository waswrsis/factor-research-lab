"""Artificial receipts and injected runners; no network or real provider calls."""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from test_v7_feed_audit import Clock
from test_v7_rolling_registry import candles
from test_v7_run_factor_pool import source_root
from test_v7_s_controller_integration import FakeBatch

from factorlab.discovery import v7_autonomous as autonomous
from factorlab.discovery.v7_forward import ForwardRuntime
from factorlab.discovery.v7_run import V7Config
from factorlab.discovery.v7_service import ForwardService, ServiceConfig


class Runner:
    def __init__(self):
        self.starts = []
        self.processes = {}
        self.stops = []

    def start(self, path, attempt):
        identity = {'pid': len(self.starts) + 1, 'created': 100., 'command': [str(path), attempt['token']]}
        self.starts.append((path, dict(attempt), identity))
        self.processes[identity['pid']] = 'running'
        return identity

    def find(self, path, attempt):
        return next((identity for prior, reserved, identity in self.starts
                     if prior == path and reserved['token'] == attempt['token']), None)

    def status(self, identity):
        return self.processes[identity['pid']]

    def stop(self, identity):
        self.stops.append(identity)
        self.processes[identity['pid']] = 'exited'
        return {'status': 'synthetic_owned_termination'}


def actual_wallet_schedule(runtime, config):
    from factorlab.discovery.monthly_schedule import default_schedule
    from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
    from factorlab.discovery.neural_forward_ticket import _bind_authority
    authority = MonthlyWorkAuthority(runtime.directory.parent / 'wallet', run_id='autonomous-artificial-audit',
                                     task='volatility', universe=list(runtime.config.symbols), partition_policy='standard')
    _bind_authority(runtime, authority)
    schedule = default_schedule(config.research, month=pd.Timestamp(runtime.clock()).strftime('%Y-%m'))
    return {'authority': authority, 'schedule': schedule}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    cutoff = pd.Timestamp.now(tz='UTC').normalize().replace(day=1)
    clock = Clock(cutoff - pd.Timedelta(seconds=1))
    runtime = ForwardRuntime(tmp_path / 'forward', tmp_path / 'feed', clock=clock)
    bars = candles((cutoff - pd.DateOffset(months=8)).isoformat(), (cutoff - pd.Timedelta(hours=2)).isoformat())
    seed = tmp_path / 'seed.parquet'
    bars.to_parquet(seed, index=False)
    runtime.feed.import_seed(seed)
    clock.now = pd.Timestamp.now(tz='UTC')
    config = autonomous.AutonomousConfig(backend='disabled', research=V7Config(
        evaluation_mode='joint', search_mode='gpt', formulas_per_family=8, lookback=2))
    runner = Runner()
    scheduler = autonomous.ResearchScheduler(runtime, config=config, runner=runner,
        **actual_wallet_schedule(runtime, config))
    yield runtime, clock, scheduler, runner, config
    scheduler.close()
    runtime.close()


def job(scheduler):
    month = pd.Timestamp(scheduler.runtime.clock()).strftime('%Y-%m')
    return scheduler._load(month)


def test_nonblocking_once_per_month_same_run_and_pid_recovery(setup):
    runtime, clock, scheduler, runner, config = setup
    first = scheduler.step()
    assert first['actions'][-1]['status'] == 'spawned'
    state = job(scheduler)
    original_manifest = autonomous._read(state['job_path'])
    assert original_manifest['snapshot_start'] == (
        pd.Timestamp(original_manifest['cutoff']) - pd.DateOffset(months=6) - pd.Timedelta(hours=1234)).isoformat()
    for _ in range(3):
        scheduler.step()
    assert len(runner.starts) == 1
    assert not (Path(state['job_path']).parent / 'snapshot.parquet').exists()
    # Lose the parent's post-spawn checkpoint; scan the reserved exact command.
    state['attempts'][0]['process'] = None
    scheduler._save(state)
    reopened = autonomous.ResearchScheduler(runtime, config=config, runner=runner,
        authority=scheduler.authority, schedule=scheduler.schedule)
    try:
        reopened.step()
        assert len(runner.starts) == 1 and job(reopened)['attempts'][0]['process']['pid'] == 1
        runner.processes[1] = 'exited'
        clock.now += pd.Timedelta(seconds=20)
        reopened.step()
        assert len(runner.starts) == 2
        assert autonomous._read(job(reopened)['job_path']) == original_manifest
    finally:
        reopened.close()


def test_deadline_and_stop_only_signal_verified_owned_process_no_refund(setup):
    _, clock, scheduler, runner, _ = setup
    scheduler.step()
    state = job(scheduler)
    manifest = autonomous._read(state['job_path'])
    clock.now = pd.Timestamp(manifest['deadline'])
    scheduler.step()
    assert job(scheduler)['status'] == 'deadline_consumed' and len(runner.stops) == 1
    scheduler.step()
    assert len(runner.starts) == 1 and len(job(scheduler)['attempts']) == 1


def test_pid_reuse_is_fail_closed_without_killing_other_process(setup):
    _, _, scheduler, runner, _ = setup
    scheduler.step()
    runner.processes[1] = 'identity_mismatch'
    scheduler.step()
    assert job(scheduler)['status'] == 'process_identity_lost'
    assert not runner.stops and len(runner.starts) == 1


def test_three_interrupted_launches_keep_one_manifest_and_no_new_run(setup):
    _, clock, scheduler, runner, _ = setup
    scheduler.step()
    original = job(scheduler)['job_hash']
    for index in range(3):
        runner.processes[index + 1] = 'exited'
        clock.now += pd.Timedelta(seconds=20)
        scheduler.step()
    assert len(runner.starts) == 3 and job(scheduler)['status'] == 'attempts_consumed'
    assert job(scheduler)['job_hash'] == original


def test_snapshot_keeps_receipt_attrs_and_rejects_corruption_or_future_revision(setup):
    runtime, clock, scheduler, _, _ = setup
    scheduler.step()
    path = job(scheduler)['job_path']
    before = autonomous.sealed_snapshot(path)
    assert before.attrs['feed_identity'] == runtime.feed.identity
    assert before.attrs['availability_mode'] == 'mixed_seed_and_observed_feed'
    assert {'received_at', 'imported_at', 'availability_basis', 'revision', 'prediction_origin'} <= set(before)
    revised = before.iloc[:1].copy()
    revised['volume'] += 1
    runtime.feed.receive(revised, received_at=clock(), response_hash='a' * 64)
    after = autonomous.sealed_snapshot(path)
    pd.testing.assert_frame_equal(before, after)
    snapshot = Path(path).parent / 'snapshot.parquet'
    snapshot.write_bytes(snapshot.read_bytes() + b'damaged')
    with pytest.raises(ValueError, match='snapshot'):
        autonomous.sealed_snapshot(path)


def test_actual_worker_stub_has_shared_feed_S_provenance_and_resumes_no_calls(setup, monkeypatch):
    runtime, clock, scheduler, runner, _ = setup
    scheduler.step()
    state = job(scheduler)
    attempt = state['attempts'][0]
    backend = FakeBatch()
    first = autonomous.run_worker(state['job_path'], 0, attempt['token'], backend=backend, enforce_resources=False)
    assert len(backend.calls) == 3 and first['budget']['counts']['ast'] == 3
    clock.now = pd.Timestamp.now(tz='UTC')
    runner.processes[1] = 'exited'
    scheduler.step()
    assert job(scheduler)['status'] == 'complete' and len(runner.starts) == 1
    manifest = autonomous._read(state['job_path'])
    with __import__('sqlite3').connect(manifest['run_directory'] + '/state.sqlite') as db:
        measured = json.loads(db.execute('SELECT value FROM months').fetchone()[0])
    assert measured['search_exposure']['source'] == runtime.feed.identity
    context = measured['search_exposure']
    runtime.lifecycle.exposure.assert_prompt_search(source=runtime.feed.identity, symbols=runtime.config.symbols,
        start=context['start'], end=context['end'], batch=context['batch'])
    # Completion queues recipes from the original Pv, never the updated P_dev.
    service = ForwardService(runtime, config=ServiceConfig(poll=False), guard=lambda: {'allowed': True})
    try:
        # This queue-only fixture still enters the real service step/clock/ledger.
        # The public defer_fits flag prevents unrelated forward training; tick,
        # planning and the original native deadline checks all remain active.
        entered = service.step(defer_fits=True)
        assert entered['status'] == 'stepped' and service.state['steps'] == 1
        assert service.state['native_service_clock'] == json.loads(
            (service.directory / 'native-clock.json').read_text('utf-8'))
        assert service.state['fit_attempts'] == {}
        original_queue = service.queue_batch

        def publish_then_crash(**kwargs):
            original_queue(**kwargs)
            raise RuntimeError('Synthetic post-queue publication crash')

        monkeypatch.setattr(service, 'queue_batch', publish_then_crash)
        with pytest.raises(RuntimeError, match='post-queue'):
            scheduler.step(service=service)
        assert job(scheduler)['validation']['status'] == 'queue_pending'
        monkeypatch.setattr(service, 'queue_batch', original_queue)
        clock.now += pd.Timedelta(seconds=5)
        scheduler.step(service=service)
        queued = job(scheduler)['validation']
        assert queued['A0_changed'] is False
        assert queued['status'] == 'queued_V_only'
        request = queued['request']
        assert request['reference']['pool'] == measured['frozen_reference_pool']
        assert len(request['candidates']) == 3
        for candidate in request['candidates']:
            assert candidate['pool']['reference_pool'] == measured['frozen_reference_pool']
            assert isinstance(candidate['complexity'], int) and candidate['complexity'] > 0
        assert pd.Timestamp(request['start']) > pd.Timestamp(manifest['cutoff'])
        service._freeze_due(clock(), [])
        frozen = runtime.lifecycle._batch(request['batch'])
        assert frozen[1] == 'frozen_V' and not runtime.lifecycle.calibration_ready()
        assert frozen[0]['candidates'] == request['candidates']
        assert runtime.lifecycle.active_model('A0') is None
    finally:
        service.close(close_runtime=False)


def test_research_service_followup_rejects_unentered_native_scope_without_work(setup):
    runtime, _, scheduler, runner, _ = setup
    service = ForwardService(runtime, config=ServiceConfig(poll=False), guard=lambda: {'allowed': True})
    try:
        before = scheduler.authority.verify()
        with pytest.raises(ValueError, match='original already-entered service clock'):
            scheduler.step(service=service)
        assert scheduler.authority.verify() == before
        assert not runner.starts and not runner.stops
        assert scheduler.db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == 0
        assert service.state['steps'] == 0 and service.state['fit_attempts'] == {}
        assert service.state.get('native_service_clock') is None
        assert not (service.directory / 'native-clock.json').exists()
    finally:
        service.close(close_runtime=False)


def test_disabled_provider_requires_explicit_stub_and_no_implicit_calls(setup):
    _, _, scheduler, _, _ = setup
    scheduler.step()
    state = job(scheduler)
    with pytest.raises(ValueError, match='explicitly configured'):
        autonomous.run_worker(state['job_path'], 0, state['attempts'][0]['token'], enforce_resources=False)


def test_wrapper_tick_before_research_and_guard_blocks_launch():
    calls = []
    runtime = object()
    service = SimpleNamespace(runtime=runtime, step=lambda **kw: calls.append('tick') or {'status': 'stepped'})
    scheduler = SimpleNamespace(runtime=runtime, worker_active=lambda: False,
                                step=lambda **kw: calls.append('research') or {'status': 'stepped'},
                                stop=lambda reason: {'status': 'stopped', 'reason': reason})
    wrapper = autonomous.AutonomousService(service, scheduler)
    wrapper.step()
    assert calls == ['tick', 'research']
    service.step = lambda **kw: {'status': 'blocked_by_host_guard'}
    assert wrapper.step()['research']['reason'] == 'host_guard'


def test_default_real_runner_requires_inherited_linux_limits(monkeypatch):
    monkeypatch.setattr(autonomous.sys, 'platform', 'win32')
    with pytest.raises(ValueError, match='Linux'):
        autonomous.inherited_limits()
    with pytest.raises(ValueError):
        replace(autonomous.AutonomousConfig(), deadline_seconds=7201)


def test_timely_physical_readback_survives_later_logical_clock(setup):
    _, clock, scheduler, runner, _ = setup
    scheduler.step()
    state = job(scheduler)
    manifest = autonomous._read(state['job_path'])
    attempt = state['attempts'][0]
    autonomous._write(Path(state['job_path']).parent / 'outcome-0.json',
        {'job_hash': state['job_hash'], 'token': attempt['token'], 'attempt': attempt['number'],
         'status': 'complete', 'finished_at': pd.Timestamp.now(tz='UTC').isoformat()})
    runner.processes[1] = 'exited'
    clock.now = pd.Timestamp(manifest['deadline']) + pd.Timedelta(seconds=20)
    scheduler.step()
    assert job(scheduler)['status'] == 'complete' and not runner.stops


def test_unreserved_worker_token_rejected_before_snapshot_or_provider(setup):
    _, _, scheduler, _, _ = setup
    scheduler.step()
    state = job(scheduler)
    with pytest.raises(ValueError, match='reservation'):
        autonomous.run_worker(state['job_path'], 0, 'wrong-token', enforce_resources=False)
    assert not (Path(state['job_path']).parent / 'snapshot.parquet').exists()


def test_run_charges_durable_sleeps_defers_fits_and_stops_at_budget():
    runtime, elapsed, events, stopped, deferred = object(), [0.], [], [], []
    state = {'spent_seconds': 0., 'active_since': None}
    service = SimpleNamespace(runtime=runtime, config=SimpleNamespace(wall_seconds=3, tick_seconds=1),
        state=state, monotonic=lambda: elapsed[0], sleeper=lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds),
        lifecycle=SimpleNamespace(now=lambda: pd.Timestamp('2026-10-01T00:00Z') + pd.Timedelta(seconds=elapsed[0])),
        _save=events.append, step=lambda **kw: deferred.append(kw['defer_fits']) or {'status': 'stepped'})
    scheduler = SimpleNamespace(runtime=runtime, worker_active=lambda: True,
        step=lambda **kw: {'status': 'stepped'}, stop=lambda reason: stopped.append(reason) or {'status': 'stopped'})
    result = autonomous.AutonomousService(service, scheduler).run()
    assert result['status'] == 'invocation_wall_limit'
    assert state['spent_seconds'] == 3 and state['active_since'] is None
    assert deferred == [True, True, True] and len(events) == 6
    assert stopped == ['supervisor_budget', 'supervisor_stopped']


def test_run_exception_stops_owned_research():
    runtime, stopped = object(), []

    def fail(**kwargs):
        raise RuntimeError('synthetic tick failure')

    service = SimpleNamespace(runtime=runtime, config=SimpleNamespace(wall_seconds=3),
                               monotonic=lambda: 0, step=fail)
    scheduler = SimpleNamespace(runtime=runtime, worker_active=lambda: False, stop=stopped.append)
    with pytest.raises(RuntimeError, match='tick failure'):
        autonomous.AutonomousService(service, scheduler).run()
    assert stopped == ['supervisor_stopped']


def test_hard_worker_loss_retains_measured_cost_and_denies_refitting(setup, monkeypatch):
    _, clock, scheduler, runner, _ = setup
    scheduler.step()
    state = job(scheduler)
    backend = FakeBatch()
    original_save, original_write = autonomous.RunStore.save, autonomous._write

    class HardLoss(BaseException):
        pass

    def crash_after_commit(self, month, record, event):
        original_save(self, month, record, event)
        if event == 'measured_result_persisted':
            raise HardLoss

    def simulate_no_finally_receipt(path, value):
        if Path(path).name.startswith('outcome-'):
            raise HardLoss
        return original_write(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(autonomous.RunStore, 'save', crash_after_commit)
        patch.setattr(autonomous, '_write', simulate_no_finally_receipt)
        with pytest.raises(HardLoss):
            autonomous.run_worker(state['job_path'], 0, state['attempts'][0]['token'],
                                  backend=backend, enforce_resources=False)
    runner.processes[1] = 'exited'
    clock.now += pd.Timedelta(seconds=20)
    scheduler.step()
    resumed = job(scheduler)
    assert resumed['job_hash'] == state['job_hash'] and len(resumed['attempts']) == 2
    counts_before = scheduler.authority.status(pd.Timestamp.now(tz='UTC').strftime('%Y-%m'))
    calls_before = len(backend.calls)
    with pytest.raises(ValueError, match='readback only|Original S preparation identity/source'):
        autonomous.run_worker(resumed['job_path'], 1, resumed['attempts'][1]['token'],
                              backend=backend, enforce_resources=False)
    assert len(backend.calls) == calls_before
    assert scheduler.authority.status(pd.Timestamp.now(tz='UTC').strftime('%Y-%m')) == counts_before
    with __import__('sqlite3').connect(autonomous._read(state['job_path'])['run_directory'] + '/state.sqlite') as db:
        measured = json.loads(db.execute('SELECT value FROM months').fetchone()[0])
    assert measured['pool']['records'] and measured['budget']['counts']['f'] >= 1
