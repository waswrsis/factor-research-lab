"""Independent lease counterexamples; artificial clocks, no child/network/GPT."""
from dataclasses import replace

import pandas as pd
import pytest
from test_v7_long_running import FakeRunner
from test_v7_long_running import setup as base_setup
from test_v7_long_running_transport import run
from test_v7_long_running import Runtime

from scripts import v7_long_running as long


def test_short_session_can_admit_work_after_nonzero_service_initialization(tmp_path, monkeypatch):
    path, clock, plan = base_setup.__wrapped__(tmp_path, monkeypatch)
    assert plan.session_seconds == 120 and plan.step_limit_seconds == 300
    runtime = Runtime(path / 'forward', clock)
    runtime.missing = True
    initialize = long.CampaignService.__init__

    def nonzero_initialize(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        clock.sleep(.001)  # SQL/service setup cannot take exactly zero real time.

    monkeypatch.setattr(long.CampaignService, '__init__', nonzero_initialize)
    store, _ = run((path, clock, plan), runtime)
    try:
        assert any(call[0] == 'tick' for call in runtime.calls), (
            'A valid short session consumed every prepaid session without admitting one tick')
    finally:
        store.close()


def test_parent_uses_original_monotonic_deadline_even_if_wall_does_not_advance(tmp_path, monkeypatch):
    path, clock, plan = base_setup.__wrapped__(tmp_path, monkeypatch)
    mono = [100.]
    monkeypatch.setattr(long.time, 'monotonic', lambda: mono[0])
    runner = FakeRunner()
    controller = long.LongController(path / 'campaign', plan, clock=lambda: clock.now,
        runner=runner, guard=lambda: {'allowed': True})
    try:
        controller.step()
        token = controller.store.load()['launches'][-1]['token']
        slot = controller.store.reserve_session(token)
        controller.store.begin_step(slot['session_id'])
        mono[0] += plan.effective_step_seconds
        runner.live = False
        assert controller.step()['reason'] == 'step_work_deadline_consumed'
        assert len(runner.starts) == 1
        assert controller.store.load()['active_step'] is not None
    finally:
        controller.close()


def test_late_completion_is_consumed_by_original_monotonic_deadline(tmp_path, monkeypatch):
    path, clock, plan = base_setup.__wrapped__(tmp_path, monkeypatch)
    mono = [100.]
    monkeypatch.setattr(long.time, 'monotonic', lambda: mono[0])
    store = long.CampaignStore(path / 'campaign', plan, clock=lambda: clock.now)
    try:
        slot = store.reserve_session('owned')
        active = store.begin_step(slot['session_id'])
        mono[0] += plan.effective_step_seconds
        with pytest.raises(TimeoutError):
            store.end_step(active['id'])
        state = store.load()
        assert state['terminal']['reason'] == 'session_step_deadline'
        assert state['sessions'] == [slot] and state['active_step'] is None
        assert store.reserve_session('new') is None
    finally:
        store.close()


def test_rollback_before_first_watch_observation_cannot_extend_original_work_lease(tmp_path, monkeypatch):
    path, clock, plan = base_setup.__wrapped__(tmp_path, monkeypatch)
    plan = replace(plan, step_limit_seconds=60)
    mono = [100.]
    monkeypatch.setattr(long.time, 'monotonic', lambda: mono[0])
    store = long.CampaignStore(path / 'campaign', plan, clock=lambda: clock.now)
    try:
        slot = store.reserve_session('owned')
        store.begin_step(slot['session_id'])
        state = store.load()
        clock.now -= pd.Timedelta(hours=1)
        cache = {}
        # Immediate fail-closed on rollback is valid too; no renewed lease is.
        long.active_step_expired(state, 'owned', cache,
            clock=lambda: clock.now, monotonic=lambda: mono[0])
        mono[0] += 61.
        assert long.active_step_expired(state, 'owned', cache,
            clock=lambda: clock.now, monotonic=lambda: mono[0]), (
                'First watchdog wall sample must not replace the reservation monotonic origin')
    finally:
        store.close()
