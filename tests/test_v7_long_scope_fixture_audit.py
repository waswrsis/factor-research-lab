"""Real local scope metadata and negative gates; no fitted model or child."""
import json
import pytest
from test_v7_long_running import Runtime, setup, open_store, actual_scheduler_kwargs
from scripts import v7_long_running as long
from factorlab.discovery.v7_service import ServiceConfig
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
from factorlab.discovery.calibration_runtime_contract import _original_operations
from factorlab.protocol import fingerprint


def test_short120_session_polls_but_cannot_charge_or_start_new300_work(setup):
    path,clock,plan=setup
    runtime=Runtime(path/'forward',clock);runtime.missing=True
    binding=actual_scheduler_kwargs(runtime,plan);authority=binding['authority']
    store=open_store(setup)
    service=long.CampaignService(runtime,store,config=ServiceConfig(session_id='short-negative',wall_seconds=120,max_steps=3),
        guard=lambda:{'allowed':True},monotonic=lambda:clock.seconds,work_authority=authority)
    try:
        manifest=json.loads((runtime.directory/'forward.json').read_text())
        assert manifest=={'identity':json.loads(json.dumps(runtime.identity)),
                          'sha256':fingerprint(runtime.identity)}
        assert any(row['kind']=='forward_runtime_bound' for row in _original_operations(runtime.lifecycle))
        before=authority.verify()['reservations']
        result=service.step()
        assert any(row.get('status')=='insufficient_service_headroom' for row in result['actions'])
        assert [row[0] for row in runtime.calls]==['tick']
        assert service.state['fit_attempts']==store.load()['fit_attempts']=={}
        assert authority.verify()['reservations']==before
        original=json.loads((service.directory/'native-clock.json').read_text())
        assert original['issuer']['lease_seconds']==300 and original['service_seconds']==120
        clock.sleep(1);service.step()
        assert json.loads((service.directory/'native-clock.json').read_text())==original
        assert [row[0] for row in runtime.calls]==['tick','tick']
        assert authority.verify()['reservations']==before
    finally:service.close();store.close()


def test_missing_original_clock_mirror_cannot_be_recreated_by_fixture(setup):
    path,clock,plan=setup;runtime=Runtime(path/'forward',clock);runtime.missing=True
    store=open_store(setup)
    service=long.CampaignService(runtime,store,config=ServiceConfig(session_id='mirror-negative',wall_seconds=120,max_steps=3),guard=lambda:{'allowed':True},monotonic=lambda:clock.seconds)
    try:
        service.step();calls=list(runtime.calls)
        mirror=service.directory/'native-clock.json';mirror.rename(mirror.with_suffix('.preserved'))
        with pytest.raises(ValueError,match='mirror missing'):
            service.step()
        assert runtime.calls==calls and not mirror.exists()
        assert mirror.with_suffix('.preserved').is_file()
    finally:service.close();store.close()


def test_scheduler_rejects_missing_and_foreign_actual_wallet(setup):
    path,clock,plan=setup;runtime=Runtime(path/'forward',clock)
    binding=actual_scheduler_kwargs(runtime,plan)
    config=long.research_config(plan.research,plan.symbols)
    try:
        with pytest.raises(ValueError,match='actual service wallet'):
            long.ResearchScheduler(runtime,config=config)
        foreign=MonthlyWorkAuthority(path/'foreign-wallet',run_id='foreign',task='volatility',universe=['BTCUSDT'],partition_policy='standard')
        with pytest.raises(ValueError):
            long.ResearchScheduler(runtime,config=config,authority=foreign,schedule=binding['schedule'])
        assert not (runtime.directory/'autonomous/scheduler.sqlite').exists()
    finally:runtime.close()
