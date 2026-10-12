"""Not-before month freezes without erasing existing evidence or ownership."""
from dataclasses import asdict, replace

import pandas as pd
import pytest
from test_v7_autonomous import Runner, actual_wallet_schedule
from test_v7_feed_audit import Clock
from test_v7_run_factor_pool import source_root

from factorlab.discovery import v7_autonomous as autonomous
from factorlab.discovery.v7_forward import ForwardRuntime
from factorlab.discovery.v7_run import V7Config
from scripts.v7_long_running import research_config


def bind_original_clock(monkeypatch, clock):
    # Explicit calendar fixture: issue and verify against the same UTC clock.
    # Original 300-second leases, 7200-second outer limit, monotonic time and
    # real process/boot identity checks are unchanged; no receipt is rewritten.
    from datetime import datetime
    from factorlab.discovery import s_outer_context, work_lease

    class CalendarDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = clock().tz_convert('UTC').to_pydatetime()
            return value.replace(tzinfo=None) if tz is None else value.astimezone(tz)

    monkeypatch.setattr(work_lease, 'datetime', CalendarDatetime)
    monkeypatch.setattr(s_outer_context, 'datetime', CalendarDatetime)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    clock = Clock('2026-10-31T23:59:00Z')
    bind_original_clock(monkeypatch, clock)
    runtime = ForwardRuntime(tmp_path / 'forward', tmp_path / 'feed', clock=clock)
    config = autonomous.AutonomousConfig(backend='disabled', not_before_month='2026-11',
        research=V7Config(evaluation_mode='joint', search_mode='local'))
    runner = Runner()
    scheduler = autonomous.ResearchScheduler(runtime, config=config, runner=runner,
        **actual_wallet_schedule(runtime, config))
    yield runtime, clock, scheduler, runner, config
    scheduler.close()
    runtime.close()


@pytest.mark.parametrize('value', ['2026-1', '2026-00', '2026-13', '0000-01',
    ' 2026-11', '2026-11 ', '2026-11-01', '2026-11Z', '2026-11-01T00:00:00Z',
    202611, True, '', '２０２６-１１'])
def test_only_canonical_month_strings_are_accepted(value):
    with pytest.raises(ValueError, match='canonical UTC YYYY-MM'):
        autonomous.AutonomousConfig(not_before_month=value)


def test_default_is_compatible_and_wire_field_roundtrips():
    assert autonomous.AutonomousConfig().not_before_month is None
    config = autonomous.AutonomousConfig(not_before_month='2026-11')
    restored = research_config(asdict(config), ('BTCUSDT',))
    assert restored == config and restored.not_before_month == '2026-11'


def test_waiting_month_does_not_create_or_charge_a_job_then_threshold_starts_once(setup):
    _, clock, scheduler, runner, _ = setup
    for _ in range(3):
        result = scheduler.step()
        assert result['status'] == 'waiting_not_before' and result['not_before_month'] == '2026-11'
        assert not result['actions'] and not result['A0_changed']
    assert scheduler.db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == 0
    assert not list((scheduler.directory / 'jobs').glob('*/job.json'))
    assert not runner.starts
    clock.now = pd.Timestamp('2026-11-01T00:00:00Z')
    assert scheduler.step()['actions'][-1]['status'] == 'spawned'
    scheduler.step()
    assert len(runner.starts) == 1 and scheduler._load('2026-10') is None
    assert len(scheduler._load('2026-11')['attempts']) == 1


def test_waiting_policy_is_frozen_in_scheduler_identity_and_cannot_be_moved(setup):
    runtime, _, scheduler, runner, config = setup
    identity = autonomous._read(scheduler.directory / 'identity.json')
    assert identity['config']['not_before_month'] == '2026-11'
    reopened = autonomous.ResearchScheduler(runtime, config=config, runner=runner,
        authority=scheduler.authority, schedule=scheduler.schedule)
    try:
        assert reopened.step()['status'] == 'waiting_not_before'
    finally:
        reopened.close()
    for value in (None, '2026-10', '2026-12'):
        with pytest.raises(ValueError, match='Immutable'):
            autonomous.ResearchScheduler(runtime, config=replace(config, not_before_month=value), runner=runner,
                authority=scheduler.authority, schedule=scheduler.schedule)


def test_not_before_uses_utc_month_when_clock_has_another_offset(setup):
    _, clock, scheduler, runner, _ = setup
    clock.now = pd.Timestamp('2026-10-31T20:00:00-05:00')  # November 1 in UTC.
    assert scheduler.step()['actions'][-1]['status'] == 'spawned'
    assert scheduler._load('2026-10') is None and len(runner.starts) == 1
    manifest = autonomous._read(scheduler._load('2026-11')['job_path'])
    assert manifest['cutoff'] == '2026-11-01T00:00:00+00:00'


def existing_running(scheduler, runner, clock):
    state = scheduler._plan(clock.now.normalize().replace(day=1), clock.now)
    attempt = {'number': 0, 'token': 'original', 'reserved_at': clock.now.isoformat(), 'process': None}
    state['attempts'].append(attempt)
    state['status'] = 'running'
    attempt['process'] = runner.start(state['job_path'], attempt)
    scheduler._save(state)
    return state


def test_waiting_still_checks_existing_process_without_replacing_attempt(setup):
    _, clock, scheduler, runner, _ = setup
    original = existing_running(scheduler, runner, clock)
    result = scheduler.step()
    assert result['status'] == 'waiting_not_before'
    assert result['actions'][0]['status'] == 'running'
    assert scheduler._load('2026-10')['attempts'] == original['attempts']
    runner.processes[1] = 'exited'
    clock.now += pd.Timedelta(seconds=20)
    result = scheduler.step()
    assert result['actions'][0]['status'] == 'interrupted'
    assert len(runner.starts) == 1 and len(scheduler._load('2026-10')['attempts']) == 1


def test_waiting_stop_still_stops_owned_existing_process_and_keeps_charge(setup):
    _, clock, scheduler, runner, _ = setup
    existing_running(scheduler, runner, clock)
    (scheduler.directory / 'STOP').write_text('operator stop', encoding='utf-8')
    assert scheduler.step()['status'] == 'stopped'
    assert len(runner.stops) == 1 and len(runner.starts) == 1
    assert scheduler._load('2026-10')['status'] == 'stopped'
    assert len(scheduler._load('2026-10')['attempts']) == 1


def test_waiting_still_enforces_an_existing_jobs_original_deadline(setup):
    _, clock, scheduler, runner, _ = setup
    clock.now = pd.Timestamp('2026-10-31T20:00:00Z')
    original = existing_running(scheduler, runner, clock)
    clock.now += pd.Timedelta(seconds=7200)
    result = scheduler.step()
    assert result['status'] == 'waiting_not_before'
    assert result['actions'][0]['status'] == 'deadline_consumed'
    assert len(runner.stops) == 1 and len(runner.starts) == 1
    assert scheduler._load('2026-10')['attempts'] == original['attempts']


def test_waiting_does_not_skip_integrity_or_completed_queue_processing(setup, monkeypatch):
    runtime, clock, scheduler, _, _ = setup
    state = scheduler._plan(clock.now.normalize().replace(day=1), clock.now)
    state['status'] = 'complete'
    scheduler._save(state)
    processed = []
    monkeypatch.setattr(scheduler, '_queue', lambda state, now, service: processed.append(('verify_queue', state['month'])))
    monkeypatch.setattr(scheduler, '_publish_queue', lambda state, service: processed.append(('publish_queue', state['month'])))
    monkeypatch.setattr(scheduler, '_queue_algorithm', lambda state, service: processed.append(('algorithm_queue', state['month'])))
    # Followup now requires an original owned service clock, not a bare object.
    # Enter the real scope without dispatching feed/model/research work.
    from factorlab.discovery.v7_service import ForwardService, ServiceConfig
    from factorlab.discovery.forward_service_scope import _service_step_scope
    service = ForwardService(runtime, config=ServiceConfig(poll=False),
                             work_authority=scheduler.authority)
    try:
        with _service_step_scope(service):
            pass
        assert scheduler.step(service=service)['status'] == 'waiting_not_before'
    finally:
        service.close(close_runtime=False)
    assert processed == [('verify_queue', '2026-10'), ('publish_queue', '2026-10'), ('algorithm_queue', '2026-10')]
    scheduler.db.execute("UPDATE jobs SET hash='tampered' WHERE month='2026-10'")
    scheduler.db.commit()
    with pytest.raises(ValueError, match='identity'):
        scheduler.step()


def test_default_none_still_plans_current_month(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    clock = Clock('2026-10-31T23:59:00Z')
    bind_original_clock(monkeypatch, clock)
    runtime = ForwardRuntime(tmp_path / 'forward', tmp_path / 'feed', clock=clock)
    config = autonomous.AutonomousConfig(backend='disabled', research=V7Config(evaluation_mode='joint', search_mode='local'))
    scheduler = autonomous.ResearchScheduler(runtime, runner=Runner(), config=config,
                                            **actual_wallet_schedule(runtime, config))
    try:
        assert scheduler.step()['actions'][-1]['status'] == 'spawned'
        assert scheduler._load('2026-10')['status'] == 'running'
    finally:
        scheduler.close()
        runtime.close()
