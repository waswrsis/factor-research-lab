"""Actual registered controller clocks, tiny artificial source/feed; no provider.

No native clock, lease duration, wallet or controller verification is replaced.
The paid positive stops explicitly after the original reservation, before numerics.
Its host observation is an explicit healthy artificial fixture: it proves the
native controller/paid-ticket boundary, not actual host health. A separate
negative checks that an unreadable host prevents reservations. Production host
inspection and the real Linux acceptance are unchanged.
"""
import copy,json,os,subprocess,sys,time
from dataclasses import asdict
from pathlib import Path
import pandas as pd
import pytest
from test_controller_native_fixture import case
from factorlab.discovery import forward_service_scope as scope
from factorlab.discovery.v7_service import ForwardService,ServiceConfig,HostGuard
from factorlab.discovery.work_lease import FrozenWorkLease,_utc
from factorlab.protocol import fingerprint


def manual(case):
    _,runtime,wallet,_,driver=case
    lease=driver._start_execution()
    attempt={'index':0,'due_at':driver.schedule.due_at[0], 'controller_hash':fingerprint(driver.identity),
        'lease':lease.to_dict(),'clock_hash':fingerprint(driver._execution_clock()),
        'physical_start':lease.observe(),'observation_prefix':None,'previous':'0'*64}
    driver._write_pair(0,'attempt',attempt)
    service=ForwardService(runtime,config=ServiceConfig(**driver.schedule.service),work_authority=wallet)
    service.state['spent_seconds']=lease.elapsed(lease.observe())
    service.state['controller_research_active']={'controller_hash':fingerprint(driver.identity),
        'attempt_hash':fingerprint(attempt),'spent_before':service.state['spent_seconds']}
    service._save('controller_dispatch_time_observed')
    return service,lease


def test_original_registered_C_uses_one_origin_across_two_actual_dispatches(case,monkeypatch):
    _,runtime,wallet,_,driver=case;seen=[];finishes=[]
    original=scope._Scope.__init__;finish=scope._Scope.finish
    def enter(self,*a,**kw):
        time.sleep(.15)  # real preparation delay, never a supplied clock
        original(self,*a,**kw)
        c=driver._execution_clock()['lease']['issuer']
        assert self.clock['version']==scope.CONTROLLER_CLOCK_VERSION
        assert self.origin.to_dict()==c and self.clock['spent_before']==0
        assert self.ceiling.monotonic_deadline==c['monotonic_started']+300
        assert _utc(self.ceiling.wall_deadline).timestamp()==pytest.approx(_utc(c['wall_started_at']).timestamp()+300)
        seen.append(copy.deepcopy(self.clock))
    def closing(self):
        before=self.service.state['spent_seconds'];finish(self)
        actual=max(time.monotonic()-self.origin.monotonic_started,time.time()-_utc(self.origin.wall_started_at).timestamp())
        after=self.service.state['spent_seconds']
        assert before<=after<=max(before,actual)+.01
        finishes.append({'before':before,'after':after,'actual':actual})
    monkeypatch.setattr(scope._Scope,'__init__',enter);monkeypatch.setattr(scope._Scope,'finish',closing)
    now=pd.Timestamp.now(tz='UTC')
    runtime.feed.receive(pd.DataFrame([dict(symbol='BTCUSDT',open_time=now.floor('h')-pd.Timedelta(hours=2),
        open=100.,high=101.,low=99.,close=100.,volume=10.)]),received_at=now,response_hash=fingerprint('tiny artificial negative-headroom feed'))
    before=wallet.verify()
    assert driver.step(0)['state']=='awaiting_successor_witness'
    assert driver.pump()['state']=='completed_step'
    assert driver.step(1)['state']=='awaiting_successor_witness'
    assert driver.pump()['state']=='completed_step'
    assert len(seen)>=2 and all(s==seen[0] for s in seen)
    assert wallet.verify()==before
    (runtime.directory/'diagnostic-accounting.json').write_text(json.dumps({'scopes':seen,'finishes':finishes},indent=2))


@pytest.mark.parametrize('case',[7200],indirect=True)
def test_registered_paid_original_300_ticket_within_C_deadline(case,monkeypatch):
    _,runtime,wallet,_,driver=case;records=[]
    from factorlab.discovery.forward_preparation import ForwardPreparation
    def stop(preparation):
        records.append(copy.deepcopy(preparation.record))
        raise ValueError('disclosed stop after original paid reservation, no model')
    monkeypatch.setattr(ForwardPreparation,'snapshot',stop)
    # This artificial controller fixture must not depend on unrelated processes
    # on the developer's machine. Do not replace clocks, leases or accounting.
    host_observation={'rss_mib':0.,'available_mib':9000.,'processes':[]}
    monkeypatch.setattr(HostGuard,'_inspect',staticmethod(lambda: copy.deepcopy(host_observation)))
    now=pd.Timestamp.now(tz='UTC')
    runtime.feed.receive(pd.DataFrame([dict(symbol='BTCUSDT',open_time=now.floor('h')-pd.Timedelta(hours=2),
        open=100.,high=101.,low=99.,close=100.,volume=10.)]),received_at=now,response_hash=fingerprint('tiny artificial feed'))
    before=wallet.verify()['reservations'];driver.step(0)
    assert len(records)==1 and wallet.verify()['reservations']==before+1
    lease=records[0]['lease'];outer=driver._execution_clock()['lease']['issuer']
    assert lease['lease_seconds']==300
    assert lease['monotonic_deadline']<=outer['monotonic_started']+7200
    assert _utc(lease['wall_deadline']).timestamp()<=_utc(outer['wall_started_at']).timestamp()+7200
    (runtime.directory/'diagnostic-paid-ticket-host-fixture.json').write_text(json.dumps({
        'artificial_healthy_host_observation':host_observation,
        'actual_host_health_or_hard_limits_proved':False,
        'original_paid_lease_seconds':lease['lease_seconds'],
        'original_controller_deadline_checks_passed':True},indent=2))


@pytest.mark.parametrize('damage',['drop_active','wrong_attempt','wrong_controller','wrong_config','clock_mirror','clock_anchor','native_issuer'])
def test_registered_binding_damage_never_falls_back_to_native(case,damage):
    _,runtime,wallet,_,driver=case;service,lease=manual(case);before=wallet.verify()
    try:
        if damage=='drop_active':service.state['controller_research_active']=None;service._save('diagnostic_drop')
        elif damage=='wrong_attempt':service.state['controller_research_active']['attempt_hash']='0'*64;service._save('diagnostic_wrong')
        elif damage=='wrong_controller':service.state['controller_research_active']['controller_hash']='0'*64;service._save('diagnostic_wrong')
        elif damage=='wrong_config':object.__setattr__(service.config,'wall_seconds',301)
        elif damage=='clock_mirror':driver._paths(-1,'clock')[1].rename(driver._paths(-1,'clock')[1].with_suffix('.retained'))
        elif damage=='clock_anchor':
            # Both rehashed copies still cannot replace the original LC anchor.
            for p in driver._paths(-1,'clock'):
                value=json.loads(p.read_text());value['lease']['issuer']['monotonic_started']-=1;p.write_text(json.dumps(value))
        elif damage=='native_issuer':
            identity=json.loads((service.directory/'identity.json').read_text())
            identity['native_execution_origin']['parent_created']+=1
            (service.directory/'identity.json').write_text(json.dumps(identity))
        with pytest.raises((ValueError,FileNotFoundError)):
            scope._Scope(service)
        assert not (service.directory/'native-clock.json').exists()
        assert wallet.verify()==before
    finally:service.close()


@pytest.mark.parametrize('case',[7200],indirect=True)
def test_unreadable_host_blocks_registered_controller_without_paid_reservation(case,monkeypatch):
    _,runtime,wallet,_,driver=case
    from factorlab.discovery.forward_preparation import ForwardPreparation
    called=[]
    def snapshot(preparation):
        called.append(preparation)
        raise AssertionError('Blocked host must never reach paid preparation')
    def unreadable():
        error=OSError('Explicit artificial unreadable host observation')
        error.guard_process={'pid':404,'status':'stopped'}
        raise error
    monkeypatch.setattr(ForwardPreparation,'snapshot',snapshot)
    monkeypatch.setattr(HostGuard,'_inspect',staticmethod(unreadable))
    now=pd.Timestamp.now(tz='UTC')
    runtime.feed.receive(pd.DataFrame([dict(symbol='BTCUSDT',open_time=now.floor('h')-pd.Timedelta(hours=2),
        open=100.,high=101.,low=99.,close=100.,volume=10.)]),received_at=now,response_hash=fingerprint('tiny artificial host-block feed'))
    before=wallet.verify()
    result=driver.step(0)
    assert result['state']=='awaiting_successor_witness'
    output=result['output']
    assert output['forward']['status']=='blocked_by_host_guard'
    assert output['forward']['guard']['allowed'] is False
    assert output['forward']['guard']['unreadable_process']=={'pid':404,'status':'stopped'}
    assert output['research']['reason']=='host_guard'
    assert not called and wallet.verify()==before


def test_original_nested_record_cannot_be_rehashed_as_native_or_renewed(case):
    _,runtime,_,_,_=case;service,lease=manual(case)
    try:
        with scope._service_step_scope(service):pass
        original=json.loads((service.directory/'native-clock.json').read_text())
        service.state['native_service_clock']={k:v for k,v in original.items() if k!='controller_binding'}
        service.state['native_service_clock'].update(version=scope.VERSION,issuer=service._native_construction_origin.to_dict())
        (service.directory/'native-clock.json').write_text(json.dumps(service.state['native_service_clock']))
        service._save('diagnostic_changed_both_native_mirrors')
        with pytest.raises(ValueError,match='clock changed'):scope._Scope(service)
    finally:service.close()


def test_original_nested_reference_child_readonly_and_exact_deadline(case):
    _,runtime,wallet,_,driver=case;service,lease=manual(case)
    try:
        with scope._service_step_scope(service):reference=scope.current_service_reference()
        assert scope.verify_service_reference(reference).monotonic_deadline==lease.supervisor().monotonic_deadline
        code='''import json,sys
from factorlab import protocol
protocol.ROOT=__import__('pathlib').Path(sys.argv[1])
from factorlab.discovery.forward_service_scope import verify_service_reference
v=json.loads(sys.argv[2]);s=verify_service_reference(v)
print(json.dumps({'deadline':s.monotonic_deadline,'issuer':s.parent_pid,'reader':__import__('os').getpid()}))
'''
        from factorlab import protocol
        child=subprocess.run([sys.executable,'-B','-c',code,str(protocol.ROOT),json.dumps(reference)],
            text=True,capture_output=True,timeout=30)
        assert child.returncode==0,child.stderr
        result=json.loads(child.stdout)
        assert result['reader']!=os.getpid() and result['issuer']==os.getpid()
        assert result['deadline']==lease.supervisor().monotonic_deadline
        changed={**reference,'monotonic_deadline':reference['monotonic_deadline']+1}
        with pytest.raises(ValueError):scope.verify_service_reference(changed)
        driver._paths(-1,'clock')[1].rename(driver._paths(-1,'clock')[1].with_suffix('.retained'))
        with pytest.raises((ValueError,FileNotFoundError)):scope.verify_service_reference(reference,_verify_source=False)
    finally:service.close()
