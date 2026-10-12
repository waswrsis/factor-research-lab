"""Artificial lifecycle/controller checks; no subprocess, network, GPT or systemd."""
import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
from filelock import Timeout
from test_v7_run_factor_pool import source_root
from test_v7_service import Clock, Runtime as SyntheticRuntime

from factorlab.discovery.v7_service import ForwardService, ServiceConfig
from scripts import v7_long_running as long


class Runtime(long.ForwardRuntime):
    """Actual frozen runtime/journal; original test-only tick/fit behavior."""
    def __init__(self, directory, clock):
        super().__init__(directory, directory.parent / 'feed', clock=lambda: clock.now)
        class ClockView:
            def __call__(self):
                return clock.now
            def __getattr__(self, name):
                return getattr(clock, name)
        self.clock = ClockView()
        self.calls, self.missing = [], False

    tick = SyntheticRuntime.tick
    fit_month = SyntheticRuntime.fit_month


def bind_campaign_scope(path, clock, plan):
    """Freeze the real feed/runtime identity before the campaign/wallet is used."""
    runtime = long.ForwardRuntime(plan.forward_directory, plan.feed_directory, clock=lambda: clock.now)
    try:
        plan = replace(plan, lifecycle=long.asdict(runtime.lifecycle.plan))
    finally:
        runtime.close()
    from factorlab.discovery.monthly_schedule import default_schedule
    from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
    authority = MonthlyWorkAuthority(path / 'actual-wallet', run_id='rotation-test',
        task='volatility', universe=list(plan.symbols), partition_policy='standard')
    schedule = default_schedule(long.research_config(plan.research, plan.symbols).research,
        month=clock.now.strftime('%Y-%m'))
    return replace(plan, work_authority={'directory': str(authority.directory),
        'manifest_hash': authority.manifest_hash}, work_schedule=schedule.to_dict())


@pytest.fixture
def setup(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    # The synthetic workspace carries the actual five-source guard inventory
    # before the wallet/campaign freezes its source identity. No guard is run.
    repository = Path(long.__file__).resolve().parents[1]
    for name in ('lease_enforcer.py', 'multiwork_enforcer.py', 'tree_work_enforcer.py',
                 'guarded_runtime.py', 'campaign_guard_entry.py'):
        target = tmp_path / 'scripts/v7_guard' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((repository / 'scripts/v7_guard' / name).read_bytes())
    clock = Clock()
    # Metadata/recovery scenarios observe after real original lease issuance.
    # This artificial service clock does not authorize a live work lease.
    clock.now = long.stamp('2026-10-30T00:00:05Z')
    monkeypatch.setattr(long.time, 'monotonic', lambda: clock.seconds)
    plan = long.CampaignPlan(end_at='2026-10-31T00:00:00Z', max_sessions=4,
        max_reserved_seconds=480, max_reserved_steps=8, max_fit_attempts=5,
        forward_directory=str(tmp_path / 'forward'), feed_directory=str(tmp_path / 'feed'),
        research={'backend': 'disabled', 'research': {'evaluation_mode': 'joint', 'search_mode': 'local'}},
        lifecycle={'source_id': 'synthetic-feed'}, session_seconds=120, session_steps=2)
    from factorlab.discovery.monthly_schedule import default_schedule
    from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
    authority = MonthlyWorkAuthority(tmp_path / 'actual-wallet', run_id='rotation-test', task='volatility', universe=['BTCUSDT'], partition_policy='standard')
    schedule = default_schedule(long.research_config(plan.research, plan.symbols).research, month='2026-10')
    plan = replace(plan, work_authority={'directory': str(authority.directory), 'manifest_hash': authority.manifest_hash},
                   work_schedule=schedule.to_dict())
    return tmp_path, clock, bind_campaign_scope(tmp_path, clock, plan)


def open_store(setup):
    path, clock, plan = setup
    return long.CampaignStore(path / 'campaign', plan, clock=lambda: clock.now)


def test_sessions_prepay_allowance_without_restart_refund(setup):
    path, clock, plan = setup
    store = open_store(setup)
    first = store.reserve_session()
    assert first['session_id'].endswith('-0')
    store.close()
    store = long.CampaignStore(path / 'campaign', plan, clock=lambda: clock.now)
    try:
        for _ in range(3):
            assert store.reserve_session()
        assert store.reserve_session() is None
        state = store.load()
        assert sum(v['wall_seconds'] for v in state['sessions']) == 480
        assert sum(v['max_steps'] for v in state['sessions']) == 8
    finally:
        store.close()


def make_service(runtime, clock, store, name, *, wall_seconds=120):
    return long.CampaignService(runtime, store,
        config=ServiceConfig(session_id=name, wall_seconds=wall_seconds, max_steps=2),
        guard=lambda: {'allowed': True}, monotonic=lambda: clock.seconds, sleeper=clock.sleep)


def test_campaign_inherited_initialization_preserves_caps_on_resume_and_rotation(setup):
    path, clock, _ = setup
    store = open_store(setup)
    runtime = Runtime(path / 'forward', clock)
    service = None
    try:
        # Exercise the actual parent constructor's initialize=True call and the
        # campaign's dynamically dispatched _save('created') callback.
        assert long.CampaignService._open_state is ForwardService._open_state
        reserved = store.reserve_session()
        service = make_service(runtime, clock, store, reserved['session_id'])
        service.state['fit_attempts']['original-job'] = {
            'count': 1, 'last_at': clock.now.isoformat()}
        service._save('fit_attempt_reserved')
        service.state.update(steps=1, spent_seconds=7.)
        service._save('bounded_test_checkpoint')
        expected = copy.deepcopy(store.load())
        service.close(close_runtime=False)
        service = None

        service = make_service(runtime, clock, store, reserved['session_id'])
        assert service.state['steps'] == 1
        assert service.state['spent_seconds'] == 7.
        assert service.state['fit_attempts'] == expected['fit_attempts']
        assert store.load() == expected
        service.close(close_runtime=False)
        service = None

        next_reserved = store.reserve_session()
        service = make_service(runtime, clock, store, next_reserved['session_id'])
        assert service.state['fit_attempts'] == expected['fit_attempts']
        current = store.load()
        assert len(current['sessions']) == 2
        assert sum(v['wall_seconds'] for v in current['sessions']) == 240
        assert sum(v['max_steps'] for v in current['sessions']) == 4
        assert current['fit_attempts'] == expected['fit_attempts']
    finally:
        if service is not None:
            service.close(close_runtime=False)
        runtime.close()
        store.close()


@pytest.mark.parametrize('missing', [('state.sqlite',), ('identity.json',),
                                    ('state.sqlite', 'identity.json')])
def test_campaign_missing_service_state_does_not_reset_global_caps(setup, missing):
    path, clock, _ = setup
    store = open_store(setup)
    runtime = Runtime(path / 'forward', clock)
    service = None
    try:
        reserved = store.reserve_session()
        service = make_service(runtime, clock, store, reserved['session_id'])
        service.state['fit_attempts']['original-job'] = {
            'count': 1, 'last_at': clock.now.isoformat()}
        service._save('fit_attempt_reserved')
        directory = service.directory
        service.close(close_runtime=False)
        service = None
        expected = copy.deepcopy(store.load())
        for name in missing:
            (directory / name).rename(directory / (name + '.preserved'))
        with pytest.raises(ValueError, match='missing.*cannot reset'):
            make_service(runtime, clock, store, reserved['session_id'])
        assert store.load() == expected
        for name in missing:
            assert not (directory / name).exists()
            assert (directory / (name + '.preserved')).is_file()
    finally:
        if service is not None:
            service.close(close_runtime=False)
        runtime.close()
        store.close()


def test_failed_job_gets_three_total_attempts_across_four_sessions(setup):
    path, clock, plan = setup
    plan = replace(plan, session_seconds=600, max_reserved_seconds=2400)
    store = long.CampaignStore(path / 'campaign', plan, clock=lambda: clock.now)
    runtime = Runtime(path / 'forward', clock)
    runtime.missing = True
    try:
        for index in range(4):
            reserved = store.reserve_session()
            service = make_service(runtime, clock, store, reserved['session_id'], wall_seconds=600)
            service.step()
            service.close(close_runtime=False)
            clock.sleep(61)
        assert sum(call[0] == 'fit' for call in runtime.calls) == 3
        attempts = next(iter(store.load()['fit_attempts'].values()))
        assert attempts['count'] == 3
        assert len(store.load()['sessions']) == 4
        assert sum(s['wall_seconds'] for s in store.load()['sessions']) == 2400
    finally:
        runtime.close()
        store.close()


def test_global_commit_survives_local_save_failure(setup, monkeypatch):
    path, clock, _ = setup
    store = open_store(setup)
    runtime = Runtime(path / 'forward', clock)
    first = make_service(runtime, clock, store, 'first')
    original = ForwardService._save
    try:
        value = {'count': 1, 'last_at': clock.now.isoformat()}
        first.state['fit_attempts']['job'] = value

        def die(self, event):
            if event == 'fit_attempt_reserved':
                raise OSError('synthetic local commit death')
            return original(self, event)

        monkeypatch.setattr(ForwardService, '_save', die)
        with pytest.raises(OSError, match='death'):
            first._save('fit_attempt_reserved')
        first.close(close_runtime=False)
        second = make_service(runtime, clock, store, 'second')
        assert second.state['fit_attempts']['job'] == value
        second.close(close_runtime=False)
        resumed = make_service(runtime, clock, store, 'first')
        assert resumed.state['fit_attempts']['job'] == value
        resumed.close(close_runtime=False)
    finally:
        runtime.close()
        store.close()


def test_queue_identity_and_completed_status_carry_cannot_be_overwritten(setup):
    store = open_store(setup)
    value = {'batch': {'request': {'batch': '2026-10', 'candidate': 'frozen'},
                       'freeze_at': '2026-11-01T00:00:00Z', 'status': 'pending'}}
    try:
        store.save_queues(value)
        bad = copy.deepcopy(value)
        bad['batch']['request']['candidate'] = 'replacement'
        with pytest.raises(ValueError, match='identity'):
            store.save_queues(bad)
        value['batch']['status'] = 'frozen'
        store.save_queues(value)
        value['batch']['status'] = 'pending'
        with pytest.raises(ValueError, match='status'):
            store.save_queues(value)
        assert store.load()['freeze_requests']['batch']['status'] == 'frozen'
    finally:
        store.close()


def test_total_fit_cap_does_not_refund_on_new_session(setup):
    path, clock, plan = setup
    plan = replace(plan, max_fit_attempts=1)
    store = long.CampaignStore(path / 'campaign', plan, clock=lambda: clock.now)
    try:
        store.reserve_fit('one', {'count': 1, 'last_at': clock.now.isoformat()})
        with pytest.raises(RuntimeError, match='consumed'):
            store.reserve_fit('two', {'count': 1, 'last_at': clock.now.isoformat()})
        assert set(store.load()['fit_attempts']) == {'one'}
    finally:
        store.close()


class FakeRunner:
    def __init__(self):
        self.starts, self.stops, self.live = [], [], False

    def start(self, directory, launch):
        self.starts.append(copy.deepcopy(launch))
        self.live = True
        return {'pid': 123, 'created': 100., 'command': ['artificial', launch['token']]}

    def status(self, identity):
        return 'running' if self.live else 'exited'

    def find(self, *args):
        return None

    def stop(self, identity):
        self.stops.append(identity)
        self.live = False


def test_single_owner_stop_is_terminal_and_never_relaunches(setup):
    path, clock, plan = setup
    runner = FakeRunner()
    controller = long.LongController(path / 'campaign', plan, clock=lambda: clock.now,
        runner=runner, guard=lambda: {'allowed': True}, sleeper=lambda x: None)
    try:
        assert controller.step()['status'] == 'spawned'
        with pytest.raises(Timeout):
            long.LongController(path / 'campaign', plan, clock=lambda: clock.now)
        (path / 'campaign/STOP').write_text('stop')
        assert controller.step()['reason'] == 'operator_stop'
        (path / 'campaign/STOP').unlink()
        assert controller.step()['reason'] == 'operator_stop'
        assert len(runner.starts) == len(runner.stops) == 1
    finally:
        controller.close()


def test_host_guard_blocks_before_worker_launch(setup):
    path, clock, plan = setup
    runner = FakeRunner()
    controller = long.LongController(path / 'campaign', plan, clock=lambda: clock.now,
        runner=runner, guard=lambda: {'allowed': False, 'reason': 'long80_active'})
    try:
        assert controller.step()['status'] == 'blocked'
        assert controller.store.load()['terminal'] is None
        assert not runner.starts
        controller.guard = lambda: {'allowed': True}
        assert controller.step()['status'] == 'spawned'
        controller.guard = lambda: {'allowed': False}
        assert controller.step()['status'] == 'terminal'
    finally:
        controller.close()


def test_missing_or_corrupt_store_refuses_restart(setup):
    path, clock, plan = setup
    store = open_store(setup)
    store.reserve_session()
    store.db.execute("UPDATE state SET hash='broken' WHERE id=1")
    store.db.commit()
    store.close()
    with pytest.raises(ValueError, match='integrity'):
        long.CampaignStore(path / 'campaign', plan, clock=lambda: clock.now)


def test_new_campaign_cannot_rebind_same_forward_directory(setup):
    path, clock, plan = setup
    store = open_store(setup)
    store.close()
    with pytest.raises(ValueError, match='Immutable'):
        long.CampaignStore(path / 'other-campaign', plan, clock=lambda: clock.now)


def test_calendar_deadline_stops_owned_worker(setup):
    path, clock, plan = setup
    runner = FakeRunner()
    controller = long.LongController(path / 'campaign', plan, clock=lambda: clock.now,
        runner=runner, guard=lambda: {'allowed': True})
    try:
        controller.step()
        clock.now = long.stamp(plan.end_at)
        assert controller.step()['reason'] == 'calendar_deadline'
        assert len(runner.stops) == 1
    finally:
        controller.close()


def test_worker_rotates_internal_sessions_without_stopping_research_each_time(setup):
    path, clock, plan = setup
    from factorlab.discovery.monthly_schedule import default_schedule
    from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
    authority = MonthlyWorkAuthority(path / 'actual-wallet', run_id='rotation-test', task='volatility', universe=['BTCUSDT'], partition_policy='standard')
    schedule = default_schedule(long.research_config(plan.research, plan.symbols).research, month='2026-10')
    plan = replace(plan, session_seconds=600, max_sessions=2, max_reserved_seconds=1200, max_reserved_steps=4, step_limit_seconds=30,
        work_authority={'directory': str(authority.directory), 'manifest_hash': authority.manifest_hash},
        work_schedule=schedule.to_dict())
    store = long.CampaignStore(path / 'campaign', plan, clock=lambda: clock.now)
    store.update('fake_launch', lambda s: s['launches'].append({'token': 'fake', 'process': None}))
    store.close()
    class BoundRuntime(Runtime):
        def fit_month(self, cutoff, *, stream='A0', pool=None, authority=None):
            assert authority.directory == path / 'actual-wallet'
            return super().fit_month(cutoff, stream=stream, pool=pool)
    runtime = BoundRuntime(path / 'forward', clock)
    runtime.missing = True
    records = []

    class Scheduler:
        def __init__(self, rt, **kwargs):
            assert kwargs['authority'].directory == authority.directory
            assert kwargs['schedule'].digest == schedule.digest
            self.runtime = rt

        def worker_active(self):
            return False

        def step(self, **kwargs):
            records.append('research_tick')
            return {'status': 'fake'}

        def stop(self, reason):
            records.append('stop')

        def close(self):
            records.append('close')

    long.run_worker(path / 'campaign', 'fake', clock=lambda: clock.now,
        runtime_factory=lambda *args, **kwargs: runtime, scheduler_factory=Scheduler,
        guard=lambda: {'allowed': True}, sleeper=clock.sleep, enforce_watchdog=False)
    assert records == ['research_tick'] * 4 + ['stop', 'close']
    store = long.CampaignStore(path / 'campaign', plan, clock=lambda: clock.now)
    try:
        assert len(store.load()['sessions']) == 2
        assert sum(s['wall_seconds'] for s in store.load()['sessions']) == 1200
        assert sum(v['count'] for v in store.load()['fit_attempts'].values()) == 3
    finally:
        store.close()


@pytest.mark.parametrize('active,loaded', [('active', 'loaded'), ('inactive', 'not-found')])
def test_actual_legacy_unit_state_must_be_loaded_and_inactive(setup, monkeypatch, active, loaded):
    _, _, plan = setup
    monkeypatch.setattr(long, 'inherited_limits', lambda: {'synthetic': True})

    class Result:
        returncode = 0
        stdout = f'LoadState={loaded}\nActiveState={active}\nExecStart={plan.old_exec_start}'

    calls = []
    monkeypatch.setattr(long.subprocess, 'run', lambda argv, **kwargs: calls.append(argv) or Result())
    assert not long.production_guard(plan)['allowed']
    assert calls[0][:4] == ['systemctl', '--user', 'show', 'fr-regression-active.service']


def test_manifest_explicit_config_cannot_be_changed(setup):
    path, clock, plan = setup
    store = open_store(setup)
    store.close()
    with pytest.raises(ValueError, match='Immutable'):
        long.CampaignStore(path / 'campaign', replace(plan, max_sessions=5), clock=lambda: clock.now)
    manifest = json.loads((path / 'campaign/campaign.json').read_text())
    assert manifest['identity']['adapter_sha256'] == long.file_hash(long.Path(long.__file__))


def test_interrupted_local_queue_uses_committed_global_final_status(setup):
    path, clock, _ = setup
    store = open_store(setup)
    runtime = Runtime(path / 'forward', clock)
    service = make_service(runtime, clock, store, 'interrupted')
    value = {'batch': {'request': {'batch': '2026-10', 'candidate': 'unchanged'},
                       'freeze_at': '2026-11-01T00:00:00Z', 'status': 'pending'}}
    try:
        service.state['freeze_requests'] = copy.deepcopy(value)
        service.state['active_since'] = clock.now.isoformat()
        service._save('synthetic_before_interrupt')
        value['batch']['status'] = 'frozen'
        store.save_queues(value)  # Global commit succeeded; local publication did not.
        service.close(close_runtime=False)
        clock.sleep(61)
        resumed = make_service(runtime, clock, store, 'interrupted')
        assert resumed.state['freeze_requests'] == value
        assert resumed.state['spent_seconds'] == 61
        resumed.close(close_runtime=False)
    finally:
        runtime.close()
        store.close()


def test_actual_service_global_cap_precedes_new_job_fit(setup):
    path, clock, plan = setup
    plan = replace(plan, session_seconds=600, max_reserved_seconds=2400, max_fit_attempts=1)
    store = long.CampaignStore(path / 'campaign', plan, clock=lambda: clock.now)
    runtime = Runtime(path / 'forward', clock)
    runtime.missing = True
    reserved = store.reserve_session()
    service = make_service(runtime, clock, store, reserved['session_id'], wall_seconds=600)
    try:
        service.step()
        clock.sleep(61)
        with pytest.raises(RuntimeError, match='allowance consumed'):
            service.step()
        assert sum(call[0] == 'fit' for call in runtime.calls) == 1
        assert next(iter(store.load()['fit_attempts'].values()))['count'] == 1
        assert store.load()['terminal']['reason'] == 'global_fit_allowance_consumed'
        assert len(store.load()['sessions']) == 1
        assert sum(s['wall_seconds'] for s in store.load()['sessions']) == 600
    finally:
        service.close(close_runtime=False)
        runtime.close()
        store.close()


def test_frozen_total_research_cap_is_calendar_derived_not_new_session_budget(setup):
    store = open_store(setup)
    try:
        assert store.identity['eligible_research_months'] == ['2026-10']
        assert store.identity['research_cumulative_ceiling']['solve'] == 1600
        assert store.identity['research_cumulative_ceiling']['gpt_attempt'] == 6
        for _ in range(4):
            store.reserve_session()
        assert store.identity['research_cumulative_ceiling']['solve'] == 1600
    finally:
        store.close()


def test_pid_identity_loss_never_signals_reused_pid(setup):
    path, clock, plan = setup
    runner = FakeRunner()
    controller = long.LongController(path / 'campaign', plan, clock=lambda: clock.now,
        runner=runner, guard=lambda: {'allowed': True})
    try:
        controller.step()
        runner.status = lambda identity: 'identity_mismatch'
        assert controller.step()['reason'] == 'process_identity_lost'
        assert not runner.stops
    finally:
        controller.close()


def test_crash_after_launch_reservation_waits_without_immediate_new_spawn(setup):
    path, clock, plan = setup
    runner = FakeRunner()
    controller = long.LongController(path / 'campaign', plan, clock=lambda: clock.now,
        runner=runner, guard=lambda: {'allowed': True})
    try:
        controller.store.update('simulated_reserved_spawn', lambda s: s['launches'].append({
            'token': 'old', 'reserved_at': clock.now.isoformat(), 'process': None}))
        assert controller.step()['status'] == 'awaiting_reserved_spawn'
        assert not runner.starts
        clock.sleep(16)
        assert controller.step()['status'] == 'spawned'
        assert len(controller.store.load()['launches']) == 2
    finally:
        controller.close()


def test_ten_thousand_events_verify_only_new_rows_over_six_months(setup):
    path, clock, plan = setup
    plan = replace(plan, end_at='2027-04-02T00:00:00Z')
    store = long.CampaignStore(path / 'campaign', plan, clock=lambda: clock.now)
    reads = 0
    try:
        for index in range(10_000):
            clock.now += long.pd.Timedelta(seconds=1560)
            store.update('synthetic_26_minute_tick', lambda state: None)
            store.load()
            reads += store.last_verification_rows
            store.load()
            assert store.last_verification_rows == 0
        assert reads == 10_000
        events = store.db.execute('SELECT COUNT(*) FROM events').fetchone()[0]
        for _ in range(100):
            store.save_queues({})
        assert store.db.execute('SELECT COUNT(*) FROM events').fetchone()[0] == events
    finally:
        store.close()
    restored = long.CampaignStore(path / 'campaign', plan, clock=lambda: clock.now)
    try:
        assert restored.last_verification_rows == 10_001
    finally:
        restored.close()


def test_cursor_detects_rewind_missing_anchor_and_restart_detects_old_tamper(setup):
    path, clock, plan = setup
    store = open_store(setup)
    store.reserve_session()
    store.load()
    store.db.execute('DELETE FROM events WHERE seq=1')
    store.db.commit()
    with pytest.raises(ValueError, match='rewound'):
        store.load()
    store.close()
    with pytest.raises(ValueError, match='head'):
        long.CampaignStore(path / 'campaign', plan, clock=lambda: clock.now)


def test_past_historical_event_tamper_is_detected_by_full_restart_verify(setup):
    path, clock, plan = setup
    store = open_store(setup)
    store.reserve_session()
    store.load()
    store.db.execute("UPDATE events SET value='{}' WHERE seq=0")
    store.db.commit()
    store.close()
    with pytest.raises((ValueError, KeyError)):
        long.CampaignStore(path / 'campaign', plan, clock=lambda: clock.now)


def test_read_only_status_refuses_new_campaign_and_keeps_existing_files_unchanged(setup):
    path, _, _ = setup
    with pytest.raises(ValueError, match='existing'):
        long.inspect_campaign(path / 'new')
    assert not (path / 'new').exists()
    store = open_store(setup)
    store.reserve_session()
    store.close()
    before = {str(p): p.read_bytes() for p in (path / 'campaign').rglob('*') if p.is_file()}
    assert len(long.inspect_campaign(path / 'campaign')['sessions']) == 1
    after = {str(p): p.read_bytes() for p in (path / 'campaign').rglob('*') if p.is_file()}
    assert before == after


def test_fit_timestamp_is_actual_clock_and_rejects_future_or_rewind(setup):
    _, clock, _ = setup
    store = open_store(setup)
    try:
        with pytest.raises(ValueError, match='reservation'):
            store.reserve_fit('job', {'count': 1, 'last_at': (clock.now + long.pd.Timedelta(seconds=1)).isoformat()})
        original = clock.now
        clock.sleep(1)
        result = store.reserve_fit('job', {'count': 1, 'last_at': original.isoformat()})
        assert result['last_at'] == clock.now.isoformat()
        with pytest.raises(ValueError, match='reservation'):
            store.reserve_fit('job', {'count': 2, 'last_at': original.isoformat()})
    finally:
        store.close()


def test_watchdog_blocked_step_expires_but_idle_rollover_does_not(setup):
    _, clock, _ = setup
    store = open_store(setup)
    cache = {}
    try:
        first = store.reserve_session('owner')
        step = store.begin_step(first['session_id'])
        assert not long.active_step_expired(store.load(), 'owner', cache,
            clock=lambda: clock.now, monotonic=lambda: clock.seconds)
        clock.sleep(store.plan.effective_step_seconds - 1)
        store.end_step(step['id'])
        clock.sleep(2)  # Controller has not yet reserved the next session.
        assert not long.active_step_expired(store.load(), 'owner', cache,
            clock=lambda: clock.now, monotonic=lambda: clock.seconds)
        with pytest.raises(RuntimeError, match='deadline'):
            store.begin_step(first['session_id'])
        second = store.reserve_session('owner')
        active = store.begin_step(second['session_id'])
        assert not long.active_step_expired(store.load(), 'owner', cache,
            clock=lambda: clock.now, monotonic=lambda: clock.seconds)
        clock.sleep(120)
        assert long.active_step_expired(store.load(), 'owner', cache,
            clock=lambda: clock.now, monotonic=lambda: clock.seconds)
        with pytest.raises(TimeoutError, match='deadline'):
            store.end_step(active['id'])
        assert store.load()['terminal']['reason'] == 'session_step_deadline'
        assert len(store.load()['sessions']) == 2
    finally:
        store.close()


def test_restart_consumes_interrupted_step_and_reserves_new_allowance(setup):
    store = open_store(setup)
    try:
        first = store.reserve_session('old')
        active = store.begin_step(first['session_id'])
        second = store.reserve_session('new')
        state = store.load()
        assert state['last_interrupted_step'] == active and state['active_step'] is None
        assert second['session_id'] != first['session_id']
        assert sum(s['wall_seconds'] for s in state['sessions']) == 240
    finally:
        store.close()


def test_long_unit_template_is_output_only_and_uses_frozen_remaining_deadline(setup):
    path, clock, plan = setup
    store = open_store(setup)
    store.close()
    before = {str(p): p.read_bytes() for p in (path / 'campaign').rglob('*') if p.is_file()}
    template = long.unit_template(path / 'campaign', clock=lambda: clock.now)
    for line in ['CPUQuota=25%', 'MemoryHigh=640M', 'MemoryMax=768M', 'MemorySwapMax=0',
                 'TasksMax=64', 'NoNewPrivileges=yes', 'UMask=0077', 'Restart=no',
                 'KillMode=control-group', 'TimeoutStopSec=20']:
        assert line in template.splitlines()
    assert f'RuntimeMaxSec={int((long.stamp(plan.end_at) - clock.now).total_seconds())}' in template
    assert 'campaign_guard_entry.py" "guard"' in template
    assert 'campaign_guard_entry.py" "postmortem"' in template
    assert 'ExecStopPost=:' in template
    assert '"-m" "scripts.v7_long_running" "run"' not in template
    after = {str(p): p.read_bytes() for p in (path / 'campaign').rglob('*') if p.is_file()}
    assert before == after


def test_empty_exec_requires_genuine_existing_frozen_operator_receipt(setup, monkeypatch):
    path, _, plan = setup
    monkeypatch.setattr(long, 'inherited_limits', lambda: {'synthetic': True})
    monkeypatch.setattr(long, '_headroom_host_guard', lambda: {'allowed': True})

    class Result:
        returncode = 0
        stdout = 'LoadState=loaded\nActiveState=inactive\nExecStart='

    monkeypatch.setattr(long.subprocess, 'run', lambda *args, **kwargs: Result())
    assert not long.production_guard(plan)['allowed']
    receipt = {'unit': plan.old_unit, 'scope': 'user', 'exec_start': plan.old_exec_start}
    receipt['sha256'] = long.fingerprint(receipt)
    file = path / 'operator-inspection.json'
    file.write_text(json.dumps(receipt))
    frozen = replace(plan, old_service_receipt=str(file), old_service_receipt_sha256=long.file_hash(file))
    assert long.production_guard(frozen)['allowed']
    file.write_text(json.dumps({**receipt, 'scope': 'system'}))
    assert not long.production_guard(frozen)['allowed']


def test_unit_template_rejects_expired_campaign_without_resetting_state(setup):
    path, clock, plan = setup
    store = open_store(setup)
    store.close()
    clock.now = long.stamp(plan.end_at)
    with pytest.raises(ValueError, match='remaining runtime'):
        long.unit_template(path / 'campaign', clock=lambda: clock.now)


def actual_scheduler_kwargs(runtime, plan):
    from factorlab.discovery.neural_forward_ticket import _bind_authority
    authority, schedule = long.actual_work_contract(plan)
    _bind_authority(runtime, authority)
    return {'authority': authority, 'schedule': schedule}


def inherited_fixture(setup):
    path, clock, plan = setup
    runtime = long.ForwardRuntime(plan.forward_directory, plan.feed_directory, clock=lambda: clock.now)
    plan = replace(plan, lifecycle=long.asdict(runtime.lifecycle.plan))
    scheduler = long.ResearchScheduler(runtime, config=long.research_config(plan.research, plan.symbols),
        **actual_scheduler_kwargs(runtime, plan))
    scheduler._plan(long.stamp('2026-09-01T00:00:00Z'), long.stamp('2026-09-30T23:30:00Z'))
    return path, clock, plan, runtime, scheduler


def test_existing_month_union_frozen_once_without_regranting_budget(setup):
    path, clock, plan, runtime, scheduler = inherited_fixture(setup)
    try:
        store = long.CampaignStore(path / 'campaign', plan, clock=lambda: clock.now)
        assert store.identity['inherited_research_months'] == ['2026-09']
        assert store.identity['research_budget_months'] == ['2026-09', '2026-10']
        assert store.identity['research_cumulative_ceiling']['solve'] == 3200
        identity = copy.deepcopy(store.identity)
        store.close()
        scheduler._plan(long.stamp('2026-10-01T00:00:00Z'), clock.now)
        resumed = long.CampaignStore(path / 'campaign', plan, clock=lambda: clock.now)
        assert resumed.identity == identity
        assert scheduler.db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == 2
        resumed.close()
    finally:
        scheduler.close()
        runtime.close()


def test_inherited_job_hash_damage_and_missing_index_fail_closed(setup):
    path, clock, plan, runtime, scheduler = inherited_fixture(setup)
    try:
        scheduler.db.execute("UPDATE jobs SET hash='damaged'")
        scheduler.db.commit()
        with pytest.raises(ValueError, match='integrity'):
            long.CampaignStore(path / 'campaign', plan, clock=lambda: clock.now)
    finally:
        scheduler.close()
        runtime.close()


def test_inherited_scheduler_source_identity_mismatch_is_rejected(setup):
    path, clock, plan, runtime, scheduler = inherited_fixture(setup)
    try:
        file = path / 'forward/autonomous/identity.json'
        value = long._enveloped(file)
        value['source_files'] = {'unrelated.py': 'different'}
        file.write_text(json.dumps({'value': value, 'sha256': long.fingerprint(value)}))
        with pytest.raises(ValueError, match='source/config'):
            long.CampaignStore(path / 'campaign', plan, clock=lambda: clock.now)
    finally:
        scheduler.close()
        runtime.close()


def test_month_created_after_campaign_is_anchored_before_any_worker_launch(setup):
    path, clock, plan = setup
    runtime = long.ForwardRuntime(plan.forward_directory, plan.feed_directory, clock=lambda: clock.now)
    plan = replace(plan, lifecycle=long.asdict(runtime.lifecycle.plan))
    store = long.CampaignStore(path / 'campaign', plan, clock=lambda: clock.now)
    scheduler = long.CampaignScheduler(runtime, store,
        config=long.research_config(plan.research, plan.symbols), runner=FakeRunner(),
        **actual_scheduler_kwargs(runtime, plan))
    try:
        assert store.identity['inherited_research_months'] == []
        scheduler.step()
        local = scheduler._load('2026-10')
        assert len(store.load()['research_anchors']['2026-10']['attempts']) == 1
        # Coherent old local state cannot erase the independent global prefix.
        long.ResearchScheduler._save(scheduler, {**local, 'attempts': [], 'status': 'planned'})
        store.close()
        with pytest.raises(ValueError, match='rewound'):
            long.CampaignStore(path / 'campaign', plan, clock=lambda: clock.now)
    finally:
        scheduler.close()
        runtime.close()


def test_research_global_reservation_survives_local_scheduler_commit_failure(setup, monkeypatch):
    path, clock, plan = setup
    runtime = long.ForwardRuntime(plan.forward_directory, plan.feed_directory, clock=lambda: clock.now)
    plan = replace(plan, lifecycle=long.asdict(runtime.lifecycle.plan))
    store = long.CampaignStore(path / 'campaign', plan, clock=lambda: clock.now)
    runner = FakeRunner()
    scheduler = long.CampaignScheduler(runtime, store,
        config=long.research_config(plan.research, plan.symbols), runner=runner,
        **actual_scheduler_kwargs(runtime, plan))
    original = long.ResearchScheduler._save

    def fail_local(self, state):
        if state['attempts']:
            raise OSError('synthetic death after global research reservation')
        original(self, state)

    monkeypatch.setattr(long.ResearchScheduler, '_save', fail_local)
    try:
        with pytest.raises(OSError, match='global research reservation'):
            scheduler.step()
        assert not runner.starts
        assert len(store.load()['research_anchors']['2026-10']['attempts']) == 1
        assert scheduler._load('2026-10')['attempts'] == []
        store.close()
        with pytest.raises(ValueError, match='rewound'):
            long.CampaignStore(path / 'campaign', plan, clock=lambda: clock.now)
    finally:
        scheduler.close()
        runtime.close()
