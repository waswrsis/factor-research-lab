"""Frozen work leases and narrow transient retries; no network or real child."""
import io
import json
import time
from dataclasses import replace

import pytest
import requests
from test_v7_long_running import setup as base_setup
from test_v7_long_running import Runtime

from factorlab.discovery import v7_feed as feed
from factorlab.discovery.v7_autonomous import AutonomousService
from factorlab.discovery.v7_service import ServiceConfig
from scripts import v7_long_running as long
from scripts import v7_public_fetch as public


@pytest.fixture
def setup(tmp_path, monkeypatch):
    path, clock, plan = base_setup.__wrapped__(tmp_path, monkeypatch)
    return path, clock, replace(plan, step_limit_seconds=60)


class Scheduler:
    def __init__(self, runtime, **kwargs):
        self.runtime = runtime
        self.calls = []

    def worker_active(self):
        return False

    def step(self, **kwargs):
        self.calls.append('step')
        return {'status': 'synthetic'}

    def stop(self, reason):
        self.calls.append(('stop', reason))

    def close(self):
        self.calls.append('close')


def run(setup, runtime):
    path, clock, plan = setup
    store = long.CampaignStore(path / 'campaign', plan, clock=lambda: clock.now)
    store.update('fake_launch', lambda state: state['launches'].append({'token': 'owned', 'process': None}))
    store.close()
    original_fit = runtime.fit_month
    def fit_with_wallet(cutoff, *, stream='A0', pool=None, authority=None):
        assert authority.directory == path / 'actual-wallet'
        return original_fit(cutoff, stream=stream, pool=pool)
    runtime.fit_month = fit_with_wallet
    scheduler = Scheduler(runtime)
    long.run_worker(path / 'campaign', 'owned', clock=lambda: clock.now,
        runtime_factory=lambda *a, **kw: runtime, scheduler_factory=lambda *a, **kw: scheduler,
        guard=lambda: {'allowed': True}, sleeper=clock.sleep, enforce_watchdog=False)
    return long.CampaignStore(path / 'campaign', plan, clock=lambda: clock.now), scheduler


def test_slow_poll_plus_fit_rotates_before_tail_without_stopping_research(setup):
    path, clock, plan = setup
    plan = replace(plan, session_seconds=604, max_sessions=2, max_reserved_seconds=1208,
                   max_reserved_steps=4, step_limit_seconds=300)
    runtime = Runtime(path / 'forward', clock)
    runtime.missing = True
    original_tick, original_fit = runtime.tick, runtime.fit_month
    starts = []

    def tick(**kwargs):
        starts.append(clock.seconds)
        clock.sleep(5)
        return original_tick(**kwargs)

    def fit(*args, **kwargs):
        clock.sleep(260)
        return original_fit(*args, **kwargs)

    runtime.tick, runtime.fit_month = tick, fit
    store, scheduler = run((path, clock, plan), runtime)
    try:
        assert starts == [0, 325]  # Second work begins only in a newly prepaid session.
        state = store.load()
        assert len(state['sessions']) == 2 and sum(s['wall_seconds'] for s in state['sessions']) == 1208
        assert state['terminal']['reason'] == 'campaign_allowance_consumed'
        assert sum(v['count'] for v in state['fit_attempts'].values()) == 2
        assert scheduler.calls == ['step', 'step', ('stop', 'supervisor_stopped'), 'close']
    finally:
        store.close()


def test_clock_advancing_during_admission_rotates_without_killing_research(setup, monkeypatch):
    path, clock, plan = setup
    plan = replace(plan, max_sessions=2, max_reserved_seconds=240, max_reserved_steps=4)
    runtime = Runtime(path / 'forward', clock)
    runtime.missing = True
    original = long.CampaignStore.begin_step
    attempts = []

    def delayed(self, session_id):
        attempts.append(session_id)
        if len(attempts) == 1:
            clock.sleep(60)  # Clock changed after the outer loop's admission observation.
        return original(self, session_id)

    monkeypatch.setattr(long.CampaignStore, 'begin_step', delayed)
    store, scheduler = run((path, clock, plan), runtime)
    try:
        assert len(attempts) == 2 and attempts[0] != attempts[1]
        assert scheduler.calls == ['step', ('stop', 'supervisor_stopped'), 'close']
        assert len(store.load()['sessions']) == 2
        assert store.load()['terminal']['reason'] == 'campaign_allowance_consumed'
    finally:
        store.close()


def test_five_transient_failures_across_sessions_are_charged_and_terminal(setup):
    path, clock, plan = setup
    plan = replace(plan, step_limit_seconds=30)
    runtime = Runtime(path / 'forward', clock)
    calls = []

    def tick(**kwargs):
        calls.append(clock.seconds)
        clock.sleep(1)
        raise feed.TransientFeedError('http_503')

    runtime.tick = tick
    store, scheduler = run((path, clock, plan), runtime)
    try:
        state = store.load()
        assert len(calls) == 5 and state['consecutive_transport_failures'] == state['total_transport_failures'] == 5
        assert len(state['sessions']) == 3
        assert state['terminal']['reason'] == 'transient_transport_allowance_consumed'
        assert state['active_step'] is None
        assert scheduler.calls == [('stop', 'supervisor_stopped'), 'close']
        # Actual service journals retain each attempted tick, even without a prediction.
        steps = 0
        import sqlite3
        for database in (path / 'forward/service').glob('*/state.sqlite'):
            with sqlite3.connect(database) as db:
                steps += json.loads(db.execute('SELECT value FROM checkpoint').fetchone()[0])['steps']
        assert steps == 5
    finally:
        store.close()


def test_failure_counter_survives_reopen_and_only_success_resets(setup):
    path, clock, plan = setup
    store = long.CampaignStore(path / 'campaign', plan, clock=lambda: clock.now)
    for _ in range(3):
        store.record_transport_outcome('timeout')
    store.reserve_session('owned')
    store.close()
    store = long.CampaignStore(path / 'campaign', plan, clock=lambda: clock.now)
    try:
        assert store.load()['consecutive_transport_failures'] == 3
        store.reserve_session('owned')
        assert store.load()['consecutive_transport_failures'] == 3
        store.record_transport_outcome()
        assert store.load()['consecutive_transport_failures'] == 0
        assert store.load()['total_transport_failures'] == 3
    finally:
        store.close()


@pytest.mark.parametrize('failure', [ValueError('schema or integrity'), MemoryError('resource'),
                                  OSError('disk write'), RuntimeError('clock')])
def test_nontransport_failure_is_not_swallowed_and_keeps_step_charge(setup, failure):
    path, clock, plan = setup
    store = long.CampaignStore(path / 'campaign', plan, clock=lambda: clock.now)
    reserved = store.reserve_session('owner')
    runtime = Runtime(path / 'forward', clock)

    def tick(**kwargs):
        clock.sleep(1)
        raise failure

    runtime.tick = tick
    service = long.CampaignService(runtime, store, config=ServiceConfig(session_id=reserved['session_id'],
        wall_seconds=120, max_steps=2), guard=lambda: {'allowed': True}, monotonic=lambda: clock.seconds)
    try:
        with pytest.raises(type(failure)):
            long.campaign_step(store, AutonomousService(service, Scheduler(runtime)), reserved['session_id'])
        assert service.state['steps'] == 1 and service.state['spent_seconds'] == 1
        assert store.load()['total_transport_failures'] == 0
        assert store.load()['active_step'] is None
        assert store.load()['terminal']['reason'] == f'nontransient_step_failure:{type(failure).__name__}'
    finally:
        service.close()
        store.close()


def test_work_hard_deadline_and_tail_admission_are_frozen(setup):
    path, clock, plan = setup
    store = long.CampaignStore(path / 'campaign', plan, clock=lambda: clock.now)
    try:
        slot = store.reserve_session('owner')
        clock.sleep(60)  # 60s remains; frozen work59 + cleanup2 cannot fit.
        with pytest.raises(RuntimeError, match='deadline'):
            store.begin_step(slot['session_id'])
        assert store.load()['terminal'] is None
        slot = store.reserve_session('owner')
        active = store.begin_step(slot['session_id'])
        assert (long.stamp(active['deadline']) - clock.now).total_seconds() == plan.effective_step_seconds
        clock.sleep(plan.effective_step_seconds)
        assert long.active_step_expired(store.load(), 'owner', {}, clock=lambda: clock.now,
                                       monotonic=lambda: clock.seconds)
        with pytest.raises(TimeoutError):
            store.end_step(active['id'])
    finally:
        store.close()


def test_parent_cannot_clear_expired_work_by_launching_a_new_worker(setup):
    from test_v7_long_running import FakeRunner
    path, clock, plan = setup
    runner = FakeRunner()
    controller = long.LongController(path / 'campaign', plan, clock=lambda: clock.now,
        runner=runner, guard=lambda: {'allowed': True})
    try:
        controller.step()
        token = controller.store.load()['launches'][-1]['token']
        slot = controller.store.reserve_session(token)
        controller.store.begin_step(slot['session_id'])
        clock.sleep(plan.effective_step_seconds)
        runner.live = False  # Owned watchdog has exited, leaving the durable active lease.
        assert controller.step()['reason'] == 'step_work_deadline_consumed'
        assert len(runner.starts) == 1
        assert controller.store.load()['active_step'] is not None
    finally:
        controller.close()


def test_original_monotonic_lease_and_boot_cannot_be_reissued_on_first_check(setup, monkeypatch):
    path, clock, plan = setup
    store = long.CampaignStore(path / 'campaign', plan, clock=lambda: clock.now)
    try:
        slot = store.reserve_session('owner')
        active = store.begin_step(slot['session_id'])
        state = store.load()
        assert active['monotonic_deadline'] == clock.seconds + plan.effective_step_seconds
        # A wall correction that stays after began_at must still not add time.
        clock.sleep(5)
        clock.now -= long.pd.Timedelta(seconds=4)
        cache = {}
        assert not long.active_step_expired(state, 'owner', cache, clock=lambda: clock.now)
        assert cache['deadline'] == active['monotonic_deadline']
        clock.seconds = active['monotonic_deadline']
        assert long.active_step_expired(state, 'owner', {}, clock=lambda: clock.now)
        clock.seconds = active['monotonic_started']
        monkeypatch.setattr(long, 'boot_identity', lambda: 'different-boot')
        assert long.active_step_expired(state, 'owner', {}, clock=lambda: clock.now)
    finally:
        store.close()


class HTTPClient:
    headers, cookies, proxies, params = None, None, None, None
    auth = None

    def __init__(self, status):
        self.headers = {}
        self.status = status

    def get(self, *a, **kw):
        response = requests.Response()
        response.status_code = self.status
        response._content = b'secret response body must never be serialized'
        response.url = public.ENDPOINT
        response.raw = io.BytesIO(response._content)
        return response

    def close(self):
        pass


@pytest.mark.parametrize('status', [429, 500, 503, 599, 401, 403, 451])
def test_actual_http_status_controls_retry_classification(tmp_path, monkeypatch, status):
    client = HTTPClient(status)
    params = {'symbol': 'BTCUSDT', 'interval': '1h', 'limit': 96, 'endTime': 1000}
    with pytest.raises(requests.HTTPError) as raised:
        public.fetch(params, time.time() + 20, session=client)
    packet = public.transient_error(raised.value)
    retryable = status == 429 or status >= 500
    assert packet == ({'kind': 'http', 'status': status} if retryable else None)
    store = feed.ReceiptArchive(tmp_path)
    try:
        with pytest.raises(feed.TransientFeedError if retryable else requests.HTTPError):
            store._fetch_page(params, client, time.monotonic() + 30)
    finally:
        store.close()


@pytest.mark.parametrize('failure', [requests.exceptions.SSLError('certificate'),
    requests.exceptions.ProxyError('proxy'), ValueError('schema'), MemoryError('resource')])
def test_security_schema_resource_errors_are_not_typed_transient(failure):
    assert public.transient_error(failure) is None


def test_owned_worker_error_packet_is_finite_and_validated(monkeypatch):
    class Child:
        returncode = 0
        stdin, stdout = None, None

        def communicate(self, data, timeout):
            return json.dumps({'error': {'kind': 'http', 'status': 503}}).encode(), b''

        def poll(self):
            return 0

    monkeypatch.setattr(feed.subprocess, 'Popen', lambda *a, **kw: Child())
    with pytest.raises(feed.TransientFeedError, match='503'):
        feed._owned_fetch({}, time.monotonic() + 30)
    with pytest.raises(ValueError):
        feed._transient_packet({'kind': 'http', 'status': 451})
    with pytest.raises(ValueError):
        feed._transient_packet({'kind': 'http', 'status': 503, 'body': 'secret'})


@pytest.mark.parametrize('status', [429, 503, 451])
def test_default_helper_emits_only_actual_retryable_status_not_body(monkeypatch, status):
    client = HTTPClient(status)
    monkeypatch.setattr(requests, 'Session', lambda: client)
    monkeypatch.setattr(public.sys, 'argv', ['fixed-public-helper', str(time.time() + 20)])
    request = {'symbol': 'BTCUSDT', 'interval': '1h', 'limit': 96, 'endTime': 1000}
    monkeypatch.setattr(public.sys, 'stdin', io.TextIOWrapper(io.BytesIO(json.dumps(request).encode())))
    output = io.StringIO()
    monkeypatch.setattr(public.sys, 'stdout', output)

    class Timer:
        def __init__(self, *args):
            pass

        def start(self):
            pass

    monkeypatch.setattr(public.threading, 'Timer', Timer)
    if status == 451:
        with pytest.raises(requests.HTTPError):
            public.main()
        assert output.getvalue() == ''
    else:
        public.main()
        assert json.loads(output.getvalue()) == {'error': {'kind': 'http', 'status': status}}
        assert 'secret' not in output.getvalue()
