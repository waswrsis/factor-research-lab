"""Deadline races: artificial callbacks and intercepted exits, never real processes."""
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import pandas as pd
import pytest
from test_v7_autonomous import job
from test_v7_autonomous import setup as original_setup

from factorlab.discovery import v7_autonomous as autonomous


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    fixture = original_setup.__wrapped__(tmp_path, monkeypatch)
    context = next(fixture)
    scheduler = context[2]
    scheduler.step()
    state = job(scheduler)
    autonomous.sealed_snapshot(state['job_path'])
    manifest = autonomous._read(state['job_path'])
    try:
        yield context, state, manifest
    finally:
        with pytest.raises(StopIteration):
            next(fixture)


def _clock(monkeypatch, deadline):
    wall = [pd.Timestamp(deadline).timestamp() - 1]
    monkeypatch.setattr(autonomous, 'time', SimpleNamespace(time=lambda: wall[0]))

    class PandasClock:
        Timestamp = SimpleNamespace(now=lambda **kwargs: pd.Timestamp(wall[0], unit='s', tz='UTC'))

        def __getattr__(self, name):
            return getattr(pd, name)

    monkeypatch.setattr(autonomous, 'pd', PandasClock())
    from factorlab.discovery import s_outer_context
    def check_outer(value):
        if wall[0] >= pd.Timestamp(value['wall_deadline']).timestamp():
            raise TimeoutError('artificial original outer deadline consumed')
        return value
    monkeypatch.setattr(s_outer_context, 'check_clock', check_outer)
    monkeypatch.setattr(s_outer_context, '_check_clock_live', check_outer)
    return wall


def test_worker_cannot_publish_complete_when_callback_returns_after_deadline(prepared, monkeypatch):
    _, state, manifest = prepared
    wall = _clock(monkeypatch, manifest['deadline'])

    def finish_late(*args, **kwargs):
        wall[0] += 2
        return {'synthetic_summary': 'returned one second beyond deadline'}

    monkeypatch.setattr(autonomous, 'run_month', finish_late)
    try:
        autonomous.run_worker(state['job_path'], 0, state['attempts'][0]['token'],
                              backend=object(), enforce_resources=False)
    except TimeoutError:
        pass
    outcome = autonomous._read(Path(state['job_path']).parent / 'outcome-0.json')
    assert outcome['status'] == 'deadline_consumed', outcome
    assert 'summary_hash' not in outcome


class SimulatedExit(BaseException):
    pass


def _watchdog_harness(monkeypatch):
    callbacks, exits, errors = [], [], []

    class Event:
        def wait(self, timeout):
            return False  # Deterministic expiration, without sleeping.

        def set(self):
            pass

    class Thread:
        def __init__(self, *, target, daemon):
            callbacks.append(target)

        def start(self):
            pass  # Invoke the actual closure at the selected interleaving.

    def exit_process(code):
        exits.append(code)
        raise SimulatedExit

    monkeypatch.setattr(autonomous, 'threading', SimpleNamespace(Event=Event, Thread=Thread, Lock=Lock))
    monkeypatch.setattr(autonomous, 'inherited_limits', lambda: {'synthetic': True})
    monkeypatch.setattr(autonomous, 'psutil', SimpleNamespace(
        Process=lambda: SimpleNamespace(children=lambda **kwargs: []), NoSuchProcess=RuntimeError))
    monkeypatch.setattr(autonomous.os, '_exit', exit_process)
    return callbacks, exits, errors


def test_watchdog_receipt_failure_must_not_prevent_hard_exit(prepared, monkeypatch):
    _, state, manifest = prepared
    wall = _clock(monkeypatch, manifest['deadline'])
    callbacks, exits, errors = _watchdog_harness(monkeypatch)
    original = autonomous._write
    writes = []

    def disk_failure_once(path, value):
        writes.append(value['status'])
        if len(writes) == 1:
            raise OSError('synthetic full disk during deadline receipt')
        return original(path, value)

    def finish(*args, **kwargs):
        wall[0] += 2
        try:
            callbacks[0]()
        except (OSError, SimulatedExit) as exc:  # Intercept fake exit; never kill pytest.
            errors.append(type(exc).__name__)
        return {'synthetic_summary': 'worker concurrently finishing'}

    monkeypatch.setattr(autonomous, '_write', disk_failure_once)
    monkeypatch.setattr(autonomous, 'run_month', finish)
    try:
        autonomous.run_worker(state['job_path'], 0, state['attempts'][0]['token'], backend=object())
    except (TimeoutError, OSError):
        pass
    assert exits == [124], {'exits': exits, 'watchdog_errors': errors, 'writes': writes}


def test_timeout_winner_and_main_finally_do_not_publish_conflicting_records(prepared, monkeypatch):
    _, state, manifest = prepared
    wall = _clock(monkeypatch, manifest['deadline'])
    callbacks, _, _ = _watchdog_harness(monkeypatch)

    def finish(*args, **kwargs):
        wall[0] += 2
        try:
            callbacks[0]()
        except SimulatedExit:
            pass  # Model the main thread running after deadline publication, before hard exit.
        return {'synthetic_summary': 'concurrent completion'}

    monkeypatch.setattr(autonomous, 'run_month', finish)
    try:
        autonomous.run_worker(state['job_path'], 0, state['attempts'][0]['token'], backend=object())
    except TimeoutError:
        pass
    except ValueError as exc:
        pytest.fail(f'Conflicting immutable deadline/finally publications: {exc}')
    outcome = autonomous._read(Path(state['job_path']).parent / 'outcome-0.json')
    assert outcome['status'] == 'deadline_consumed' and 'summary_hash' not in outcome


@pytest.fixture
def service_prepared(tmp_path, monkeypatch):
    fixture = original_setup.__wrapped__(tmp_path, monkeypatch)
    context = next(fixture)
    runtime, _, scheduler, _, _ = context
    from factorlab.discovery.v7_service import ForwardService, ServiceConfig
    service = ForwardService(runtime, config=ServiceConfig(poll=False, wall_seconds=7200),
        guard=lambda: {'allowed': True})
    try:
        # The service restriction must exist before the original research job.
        entered = service.step(defer_fits=True)
        assert entered['status'] == 'stepped' and service.state['steps'] == 1
        assert service.state['native_service_clock'] is not None
        assert service.state['fit_attempts'] == {}
        scheduler.step(service=service)
        state = job(scheduler)
        manifest = autonomous._read(state['job_path'])
        assert manifest['version'] == 3
        assert manifest['outer_clock']['service_restriction']['directory'] == str(service.directory)
        yield context, state, manifest, service
    finally:
        service.close(close_runtime=False)
        with pytest.raises(StopIteration):
            next(fixture)


def test_scheduler_already_refuses_late_complete_before_any_validation_queue(service_prepared, monkeypatch):
    context, state, manifest, service = service_prepared
    runtime, clock, scheduler, runner, _ = context
    clock.now = pd.Timestamp(manifest['deadline']) + pd.Timedelta(seconds=1)
    runner.processes[1] = 'exited'
    autonomous._write(Path(state['job_path']).parent / 'outcome-0.json',
        {'job_hash': state['job_hash'], 'token': state['attempts'][0]['token'], 'status': 'complete',
         'attempt': state['attempts'][0]['number'],
         'finished_at': clock().isoformat(), 'summary_hash': 'synthetic-unused-hash'})

    def forbidden(**kwargs):
        raise AssertionError('Expired research must not queue V')

    monkeypatch.setattr(service, 'queue_batch', forbidden)
    scheduler.step(service=service)
    after = job(scheduler)
    assert after['status'] == 'deadline_consumed' and after['validation'] is None
    assert len(after['attempts']) == 1 and len(runner.starts) == 1
