"""Independent artificial campaign audit. No process launch, HTTP, GPT or deployment."""
import copy
import json
from dataclasses import replace

import pytest
from test_v7_long_running import FakeRunner, make_service, Runtime, bind_campaign_scope, actual_scheduler_kwargs
from test_v7_run_factor_pool import source_root
from test_v7_service import Clock

from scripts import v7_long_running as long


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    clock = Clock()
    # Observe after the real original issuer, matching the companion campaign
    # fixture; keep the same October month and one-day calendar envelope.
    clock.now = long.stamp('2026-10-30T00:00:05Z')
    plan = long.CampaignPlan(end_at='2026-10-31T00:00:00Z', max_sessions=4,
        max_reserved_seconds=480, max_reserved_steps=8, max_fit_attempts=5,
        forward_directory=str(tmp_path / 'forward'), feed_directory=str(tmp_path / 'feed'),
        research={'backend': 'disabled', 'research': {'evaluation_mode': 'joint', 'search_mode': 'local'}},
        lifecycle={'source_id': 'synthetic-feed'}, session_seconds=120, session_steps=2)
    return tmp_path, clock, bind_campaign_scope(tmp_path, clock, plan)


def test_inherited_current_month_reservation_cannot_disappear_on_campaign_resume(scenario):
    path, clock, plan = scenario
    runtime = long.ForwardRuntime(plan.forward_directory, plan.feed_directory, clock=lambda: clock.now)
    plan = replace(plan, lifecycle=long.asdict(runtime.lifecycle.plan))
    scheduler = long.ResearchScheduler(runtime, config=long.research_config(plan.research, plan.symbols),
        **actual_scheduler_kwargs(runtime, plan))
    store = None
    try:
        original = scheduler._plan(clock.now.normalize().replace(day=1), clock.now)
        store = long.CampaignStore(path / 'campaign', plan, clock=lambda: clock.now)
        assert store.identity['inherited_research_months'] == ['2026-10']
        store.close()
        store = None
        scheduler.db.execute('DELETE FROM jobs WHERE month=?', ('2026-10',))
        scheduler.db.commit()
        long.Path(original['job_path']).unlink()  # Artificial fixture only; simulate loss of both anchors.
        clock.sleep(60)
        with pytest.raises(ValueError, match='[Ii]nherit|reservation|missing|journal|anchor|rewind'):
            store = long.CampaignStore(path / 'campaign', plan, clock=lambda: clock.now)
    finally:
        if store is not None:
            store.close()
        scheduler.close()
        runtime.close()


def test_inherited_attempt_count_cannot_rewind_on_campaign_resume(scenario):
    path, clock, plan = scenario
    runtime = long.ForwardRuntime(plan.forward_directory, plan.feed_directory, clock=lambda: clock.now)
    plan = replace(plan, lifecycle=long.asdict(runtime.lifecycle.plan))
    scheduler = long.ResearchScheduler(runtime, config=long.research_config(plan.research, plan.symbols), runner=FakeRunner(),
        **actual_scheduler_kwargs(runtime, plan))
    store = None
    try:
        scheduler.step()
        original = scheduler._load('2026-10')
        assert len(original['attempts']) == 1
        store = long.CampaignStore(path / 'campaign', plan, clock=lambda: clock.now)
        store.close()
        store = None
        rollback = {**original, 'attempts': [], 'status': 'planned'}
        scheduler._save(rollback)  # Coherent older checkpoint, not a trivial hash corruption.
        with pytest.raises(ValueError, match='[Ii]nherit|attempt|anchor|rewind'):
            store = long.CampaignStore(path / 'campaign', plan, clock=lambda: clock.now)
    finally:
        if store is not None:
            store.close()
        scheduler.close()
        runtime.close()


def test_campaign_step_deadline_is_clamped_to_absolute_campaign_end(scenario):
    path, clock, plan = scenario
    plan = replace(plan, end_at=(clock.now + long.pd.Timedelta(seconds=30)).isoformat())
    store = long.CampaignStore(path / 'campaign', plan, clock=lambda: clock.now)
    try:
        # The new full-work-envelope admission rule is stricter than clamping:
        # 30s cannot admit this frozen 59s work lease plus cleanup.
        assert store.reserve_session('artificial') is None
        assert store.load()['active_step'] is None
    finally:
        store.close()


def test_blocked_without_launch_can_resume_with_exact_original_deadline(scenario):
    path, clock, plan = scenario
    runner = FakeRunner()
    controller = long.LongController(path / 'campaign', plan, clock=lambda: clock.now,
        runner=runner, guard=lambda: {'allowed': False})
    original = copy.deepcopy(controller.store.identity)
    try:
        assert controller.step()['status'] == 'blocked'
        assert not controller.store.load()['sessions']
    finally:
        controller.close()
    clock.sleep(60)
    resumed = long.LongController(path / 'campaign', plan, clock=lambda: clock.now,
        runner=runner, guard=lambda: {'allowed': True})
    try:
        assert resumed.store.identity == original
        assert resumed.step()['status'] == 'spawned'
        assert len(runner.starts) == 1
    finally:
        resumed.close()


def test_completed_queue_survives_global_commit_before_local_write_failure(scenario, monkeypatch):
    path, clock, plan = scenario
    store = long.CampaignStore(path / 'campaign', plan, clock=lambda: clock.now)
    runtime = Runtime(path / 'forward', clock)
    service = make_service(runtime, clock, store, 'old')
    requests = {'batch': {'request': {'batch': 'unchanged', 'candidate': 'frozen'},
                          'freeze_at': '2026-11-01T00:00:00Z', 'status': 'pending'}}
    try:
        service.state['freeze_requests'] = copy.deepcopy(requests)
        service._save('queue_seeded')
        original = long.ForwardService._save

        def failed_local(self, event):
            if event == 'batch_freeze_handled':
                raise OSError('local publication failed after global commit')
            return original(self, event)

        monkeypatch.setattr(long.ForwardService, '_save', failed_local)
        service.state['freeze_requests']['batch']['status'] = 'frozen'
        with pytest.raises(OSError):
            service._save('batch_freeze_handled')
        service.close(close_runtime=False)
        service = make_service(runtime, clock, store, 'new')
        assert service.state['freeze_requests']['batch']['status'] == 'frozen'
        local = json.loads(service.db.execute('SELECT value FROM checkpoint').fetchone()[0])
        assert local['freeze_requests'] == store.load()['freeze_requests']
    finally:
        service.close(close_runtime=False)
        runtime.close()
        store.close()
