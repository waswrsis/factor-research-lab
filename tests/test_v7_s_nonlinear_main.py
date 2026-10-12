"""Main-loop integration on actual wallet/archive; dispatcher seam is explicit."""
from dataclasses import replace

import pytest
from test_v7_s_controller_integration import FakeBatch
from test_v7_s_controller_integration import case as case  # noqa: PLC0414
from test_v7_s_controller_integration import staged as staged  # noqa: PLC0414

from factorlab.discovery.monthly_schedule import default_schedule

pytestmark = pytest.mark.parametrize('case', ['standard'], indirect=True)


def test_main_shared_loop_passes_exact_original_bridge_input_and_config_to_producer(case, monkeypatch):
    from factorlab.discovery import s_nonlinear_controller
    config = replace(case.config, nonlinear_mode='probes')
    schedule = default_schedule(config, month='2026-10')
    calls = []
    def handoff(bridge, data, cutoff, *, mode, feed_directory):
        assert bridge.authority.directory == case.authority.directory
        assert bridge.state['status'] == 'running'
        assert bridge._active is None
        assert bridge.store.identity['config']['nonlinear_mode'] == mode == 'probes'
        assert feed_directory == str(case.runtime.feed.directory) or feed_directory == case.runtime.feed.directory
        assert data.spec.digest == case.runtime.spec.digest
        assert bridge.state['frozen_reference_pool']
        assert bridge.authority.verify()['reservations'] > 1
        calls.append(bridge.identity)
        return {'status': 'private_dispatch_fixture', 'serving_eligible': False}
    monkeypatch.setattr(s_nonlinear_controller, 'run_shared_nonlinear', handoff)
    result = case.module.run_month(case.directory, case.bars, case.cutoff,
        config=config, spec=case.runtime.spec, backend=FakeBatch(),
        exposure_path=case.runtime.lifecycle.exposure.path, feed_directory=case.runtime.feed.directory,
        authority=case.authority, schedule=schedule)
    assert len(calls) == 1
    assert result['nonlinear']['status'] == 'private_dispatch_fixture'
    assert result['serving_eligible'] is False
    assert result['schedule_hash'] == schedule.digest
    assert result['budget']['counts']['gpt_attempt'] == 3


def test_producer_IO_identity_failure_does_not_silently_finalize_S(case, monkeypatch):
    from factorlab.discovery import s_nonlinear_controller
    config = replace(case.config, nonlinear_mode='probes')
    schedule = default_schedule(config, month='2026-10')
    monkeypatch.setattr(s_nonlinear_controller, 'run_shared_nonlinear',
        lambda *a, **k: (_ for _ in ()).throw(OSError('producer evidence unavailable')))
    with pytest.raises(OSError, match='producer evidence'):
        case.module.run_month(case.directory, case.bars, case.cutoff,
            config=config, spec=case.runtime.spec, backend=FakeBatch(),
            exposure_path=case.runtime.lifecycle.exposure.path, feed_directory=case.runtime.feed.directory,
            authority=case.authority, schedule=schedule)
    from factorlab.discovery.neural_research_ticket import _reader
    store = _reader(case.directory)
    try:
        assert store.get('2026-10')['status'] != 'complete'
    finally:
        store.close()
