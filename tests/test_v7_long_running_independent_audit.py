"""Cross-boundary audit of finite campaign ownership and waiting semantics."""
# ruff: noqa: F811 -- imported pytest fixture intentionally names each parameter
import pytest
from test_v7_long_running import FakeRunner, setup  # noqa: F401 - imported fixture

from scripts import v7_long_running as module


def test_waiting_for_legacy_service_cannot_extend_original_calendar_deadline(setup):
    path, clock, plan = setup
    runner = FakeRunner()
    controller = module.LongController(path / 'campaign', plan, clock=lambda: clock.now,
        runner=runner, guard=lambda: {'allowed': False})
    try:
        for _ in range(10):
            assert controller.step()['status'] == 'blocked'
            clock.sleep(60)
        assert not controller.store.load()['sessions']
        clock.now = module.stamp(plan.end_at)
        controller.guard = lambda: {'allowed': True}
        assert controller.step()['reason'] == 'calendar_deadline'
        assert not runner.starts
    finally:
        controller.close()


def test_unreserved_worker_token_cannot_open_runtime_or_charge_new_session(setup):
    path, clock, plan = setup
    controller = module.LongController(path / 'campaign', plan, clock=lambda: clock.now,
        runner=FakeRunner(), guard=lambda: {'allowed': True})
    try:
        controller.step()
        called = []

        def forbidden_runtime(*args, **kwargs):
            called.append(True)
            raise AssertionError('Unreserved worker reached runtime')

        with pytest.raises(ValueError, match='launch reservation'):
            module.run_worker(path / 'campaign', 'wrong-token', clock=lambda: clock.now,
                runtime_factory=forbidden_runtime, guard=lambda: {'allowed': True}, enforce_watchdog=False)
        assert not called
        assert not controller.store.load()['sessions']
    finally:
        controller.close()


def test_global_calendar_clock_rollback_never_adds_a_reservation(setup):
    path, clock, plan = setup
    store = module.CampaignStore(path / 'campaign', plan, clock=lambda: clock.now)
    try:
        store.reserve_session()
        count = len(store.load()['sessions'])
        clock.now -= module.pd.Timedelta(seconds=1)
        with pytest.raises(ValueError, match='backward'):
            store.reserve_session()
        assert len(store.load()['sessions']) == count
    finally:
        store.close()
