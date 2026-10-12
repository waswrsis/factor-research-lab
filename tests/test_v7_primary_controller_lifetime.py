"""Controller lifetime engineering tests: real clocks, DB/ledger, and helper PIDs.

The worker helper only writes a synthetic unscored outcome; it never calls GPT,
fits a model, provides gates, or establishes registered provider/runtime proof.
The forward dispatch is a disclosed no-work seam to isolate supervisor lifetime.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import psutil
import pytest

from test_v7_full_trial_controller import case
from factorlab.discovery import full_trial_controller as ctl
from factorlab.discovery import v7_autonomous as auto
from factorlab.discovery import v7_service
from factorlab.discovery.s_outer_context import original_clock, check_clock
from factorlab.protocol import fingerprint

HELPER = r'''
import hashlib,json,sys,time
from datetime import datetime,timezone
from pathlib import Path
jobpath=Path(sys.argv[1]); attempt=json.loads(sys.argv[2]); delay=float(sys.argv[3])
time.sleep(delay)
job=json.loads(jobpath.read_text())['value']
def fp(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
out={'job_hash':fp(job),'token':attempt['token'],'attempt':attempt['number'],'status':'complete','finished_at':datetime.now(timezone.utc).isoformat()}
with (jobpath.parent/('outcome-'+str(attempt['number'])+'.json')).open('x') as f:json.dump({'value':out,'sha256':fp(out)},f)
'''


def spawn_seam(monkeypatch, processes, *, delay=.35, fail=False):
    monkeypatch.setattr(v7_service.ForwardService, 'step', lambda self, **kw: {'status':'stepped'})
    monkeypatch.setattr(v7_service.HostGuard, '__call__', lambda self: {'allowed':True})
    def step(self, **kw):
        now=self.runtime.lifecycle.now();month=now.normalize().replace(day=1)
        state=self._plan(month,now)
        attempt={'number':0,'token':'synthetic-supervisor-only','reserved_at':now.isoformat(),'process':None}
        state.update(status='running',attempts=[attempt]);self._save(state)
        command=[sys.executable,'-B','-c',HELPER,state['job_path'],json.dumps(attempt),str(delay)]
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        ident={'pid':child.pid,'created':psutil.Process(child.pid).create_time(),'command':command}
        self.runner.children[child.pid]=child;attempt['process']=ident;self._save(state)
        processes.append((child,state['job_path']))
        if fail:raise KeyboardInterrupt('synthetic failure after durable spawn')
        return {'status':'stepped','actions':[{'status':'spawned'}]}
    monkeypatch.setattr(auto.ResearchScheduler,'step',step)


def service_state(runtime):
    import sqlite3
    with sqlite3.connect(runtime.directory/'service/registered/state.sqlite') as db:
        state=json.loads(db.execute('SELECT value FROM checkpoint').fetchone()[0])
        events=[json.loads(r[0])['event'] for r in db.execute('SELECT value FROM events ORDER BY seq')]
    with sqlite3.connect(runtime.directory/'autonomous/scheduler.sqlite') as db:
        jobs=[json.loads(r[0]) for r in db.execute('SELECT value FROM jobs')]
    return state,events,jobs


def test_same_issuer_pump_waits_and_charges_original_terminal(case,monkeypatch):
    _,runtime,wallet,_,driver=case;children=[]
    spawn_seam(monkeypatch,children)
    before=wallet.verify();result=driver.step(0)
    assert result['output']['research_terminal']['actions'][0]['status']=='complete'
    state,events,jobs=service_state(runtime)
    assert state['spent_seconds']>=.35 and state['controller_research_active'] is None
    assert events[-1]=='controller_research_finished'
    assert jobs[0]['status']=='complete' and len(jobs[0]['attempts'])==1
    job=auto._read(Path(children[0][1]));assert job['outer_clock']['issuer']['parent_pid']==os.getpid()
    assert children[0][0].poll() is not None
    assert wallet.verify()==before
    assert driver.step(0)==result and len(children)==1


def test_original_service_timeout_stops_and_reaps_without_retry(case,monkeypatch):
    _,runtime,wallet,_,driver=case;children=[]
    spawn_seam(monkeypatch,children,delay=30)
    original=v7_service.ForwardService.__init__
    def near_limit(self,*a,**kw):
        original(self,*a,**kw);self.state['spent_seconds']=self.config.wall_seconds-.4;self._save('synthetic_existing_service_charge')
    monkeypatch.setattr(v7_service.ForwardService,'__init__',near_limit)
    before=wallet.verify();result=driver.step(0)
    state,events,jobs=service_state(runtime)
    assert state['spent_seconds']>=driver.schedule.service['wall_seconds']
    assert jobs[0]['status']=='stopped' and jobs[0]['stop_reason']=='supervisor_budget'
    assert len(jobs[0]['attempts'])==1 and children[0][0].poll() is not None
    assert result['output']['research_terminal']['status']=='stopped'
    assert wallet.verify()==before


def test_exception_after_spawn_consumes_and_reaps_original_child(case,monkeypatch):
    _,runtime,_,_,driver=case;children=[]
    spawn_seam(monkeypatch,children,delay=30,fail=True)
    with pytest.raises(KeyboardInterrupt,match='synthetic'):
        driver.step(0)
    state,events,jobs=service_state(runtime)
    assert state['spent_seconds']>0 and state['controller_research_active'] is None
    assert jobs[0]['status']=='stopped' and len(jobs[0]['attempts'])==1
    assert children[0][0].poll() is not None
    assert driver.step(0)['state']=='failed_consumed'


def test_dead_original_issuer_is_consumed_not_adopted_by_new_pid(case,monkeypatch):
    _,runtime,_,_,driver=case;children=[]
    result=subprocess.run([sys.executable,'-B','-c',
        'import json;from factorlab.discovery.s_outer_context import original_clock;print(json.dumps(original_clock(30)))'],
        capture_output=True,text=True,check=True)
    clock=json.loads(result.stdout)
    with pytest.raises(ValueError,match='process/clock lost'):check_clock(clock)
    monkeypatch.setattr('factorlab.discovery.s_outer_context.original_clock',lambda seconds:clock)
    spawn_seam(monkeypatch,children,delay=30)
    # The actual original-clock validator now rejects during _plan, before
    # any job manifest, spawn intent or helper can exist. That stronger early
    # rejection still consumes the original controller attempt and cannot be
    # retried under a replacement issuer.
    before=case[2].verify()
    with pytest.raises(ValueError,match='process/clock lost'):
        driver.step(0)
    _,_,jobs=service_state(runtime)
    assert not jobs and not children
    rows=driver.verify()
    assert len(rows)==1 and rows[0]['result']['state']=='failed_consumed'
    failed=rows[0]['result']
    assert driver.step(0)==failed and not children
    assert case[2].verify()==before


def test_orphan_drain_marker_consumes_remaining_service_on_reopen(case,monkeypatch):
    _,runtime,_,_,driver=case
    monkeypatch.setattr(v7_service.HostGuard,'__call__',lambda self:{'allowed':True})
    original=v7_service.ForwardService.__init__
    def interrupted(self,*a,**kw):
        original(self,*a,**kw)
        self.state['controller_research_active']={'synthetic_prior_unfinished_drain':True}
        self._save('synthetic_prior_interrupted_drain')
    monkeypatch.setattr(v7_service.ForwardService,'__init__',interrupted)
    result=driver.step(0)
    state,events,jobs=service_state(runtime)
    assert result['output']['forward']['status']=='budget_exhausted'
    assert state['spent_seconds']>=driver.schedule.service['wall_seconds']
    assert 'controller_interrupted_research_consumed' in events and not jobs


def test_run_count_is_bounded_by_original_schedule(case,monkeypatch):
    *_,driver=case;calls=[]
    monkeypatch.setattr(ctl.FullTrialController,'_dispatch',lambda self:calls.append(1) or {'forward':{'status':'stepped'}})
    result=driver.run(max_pumps=7200)
    assert len(result['progress'])==2*len(driver.schedule.due_at)
    assert len(driver.schedule.due_at)==len(calls)
    assert driver.run()['state']=='controller_schedule_exhausted' and len(calls)==2
    for value in (0,7201,True,1.5):
        with pytest.raises(ValueError):driver.run(max_pumps=value)


def test_run_stops_at_original_service_terminal(case,monkeypatch):
    *_,driver=case;calls=[]
    monkeypatch.setattr(ctl.FullTrialController,'_dispatch',lambda self:calls.append(1) or {'forward':{'status':'budget_exhausted'}})
    assert len(driver.run(max_pumps=7200)['progress'])==2
    assert len(calls)==1


def test_worker_exit_without_original_outcome_consumes_attempt(case,monkeypatch):
    _,runtime,_,_,driver=case;children=[]
    monkeypatch.setattr(sys.modules[__name__],'HELPER','import sys;sys.exit(7)')
    spawn_seam(monkeypatch,children,delay=0)
    driver.step(0)
    _,_,jobs=service_state(runtime)
    assert jobs[0]['status']=='interrupted_consumed'
    assert jobs[0]['stop_reason']=='original_worker_exited_without_outcome'
    assert len(jobs[0]['attempts'])==1 and children[0][0].returncode==7


def test_mismatched_process_identity_is_never_killed_or_waited():
    child=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(30)'])
    try:
        runner=auto.SubprocessRunner();runner.children[child.pid]=child
        wrong={'pid':child.pid,'created':psutil.Process(child.pid).create_time()+1,'command':['not-original']}
        assert runner.stop(wrong)['status']=='not_running_or_identity_changed'
        assert runner.reap_stopped(wrong)['status']=='identity_lost_no_signal'
        assert child.poll() is None
    finally:
        child.kill();child.wait(timeout=5)


def test_actual_original_outer_deadline_is_terminal_not_renewed():
    # Genuine minimum valid 30s original scope; no artificial clock or deadline.
    clock=original_clock(30);check_clock(clock)
    time.sleep(max(0.,clock['monotonic_deadline']-time.monotonic())+.02)
    with pytest.raises(TimeoutError,match='deadline consumed'):check_clock(clock)
    assert clock['issuer']['parent_pid']==os.getpid() and clock['seconds']==30


@pytest.fixture
def original_scheduler(case):
    from factorlab.discovery.v7_run import V7Config
    from factorlab.discovery.scheduled_search import FrozenSWorkSchedule
    _,runtime,wallet,_,driver=case
    fields=dict(driver.schedule.autonomous);research=dict(fields['research']);research['symbols']=tuple(research['symbols'])
    fields['research']=V7Config(**research)
    scheduler=auto.ResearchScheduler(runtime,config=auto.AutonomousConfig(**fields),authority=wallet,
        schedule=FrozenSWorkSchedule.from_dict(driver.schedule.search_schedule))
    try:yield scheduler
    finally:
        scheduler.stop_and_reap();scheduler.close()


def test_settle_rejects_original_job_index_loss(original_scheduler):
    s=original_scheduler
    extra=s.directory/'jobs/2098-01/job.json';extra.parent.mkdir(parents=True);extra.write_text('{}')
    with pytest.raises(ValueError,match='index differs'):s.settle_running()
    assert not s.db.execute('SELECT month FROM jobs').fetchall()


def test_settle_pending_is_read_only_when_state_is_unchanged(original_scheduler,monkeypatch):
    s=original_scheduler;children=[];spawn_seam(monkeypatch,children,delay=30)
    s.step();sql=[];s.db.set_trace_callback(sql.append)
    try:
        assert s.settle_running()['status']=='pending'
        assert s.settle_running()['status']=='pending'
        assert not any(q.lstrip().upper().startswith(('INSERT','UPDATE','DELETE','BEGIN','COMMIT')) for q in sql)
    finally:s.db.set_trace_callback(None)


def test_settle_never_spawns_original_unreserved_plan(original_scheduler):
    s=original_scheduler;now=s.runtime.lifecycle.now();state=s._plan(now.normalize().replace(day=1),now)
    result=s.settle_running()
    original=s._load(state['month'])
    assert original['status']=='interrupted_consumed' and original['attempts']==[]
    assert original['stop_reason']=='original_plan_has_no_reserved_attempt'
    assert result['status']=='settled' and not s.runner.children


def test_settle_rejects_missing_original_authority(original_scheduler):
    s=original_scheduler;path=s.authority.directory/'manifest.json';kept=path.with_name('manifest-preserved-test.json')
    path.rename(kept)
    try:
        with pytest.raises((ValueError,FileNotFoundError)):s.settle_running()
        assert not path.exists()
    finally:kept.rename(path)


def test_settle_rejects_source_change(original_scheduler,monkeypatch):
    s=original_scheduler
    monkeypatch.setattr(auto,'source_files',lambda:{'synthetic_changed_source':'not-original'})
    with pytest.raises(ValueError,match='source changed'):s.settle_running()


def test_failed_teardown_keeps_marker_and_recovery_consumes_remaining(case,monkeypatch):
    _,runtime,_,_,driver=case;children=[]
    original_step=v7_service.ForwardService.step
    spawn_seam(monkeypatch,children,delay=30,fail=True)
    original=auto.SubprocessRunner.reap_stopped
    monkeypatch.setattr(auto.SubprocessRunner,'reap_stopped',lambda self,identity:(_ for _ in ()).throw(OSError('synthetic unresolved teardown')))
    with pytest.raises(OSError,match='unresolved teardown'):driver.step(0)
    state,events,jobs=service_state(runtime)
    assert state['controller_research_active'] is not None
    assert events[-1]!='controller_research_finished'
    children[0][0].wait(timeout=5)
    monkeypatch.setattr(auto.SubprocessRunner,'reap_stopped',original)
    monkeypatch.setattr(v7_service.ForwardService,'step',original_step)
    # A failed teardown closes this original execution. No later opportunity
    # may reopen the service merely to clear its original unresolved marker.
    result=driver.step(1)
    state,events,jobs=service_state(runtime)
    assert result['state']=='predecessor_unconfirmed_or_consumed'
    assert state['controller_research_active'] is not None and len(children)==1
    assert not driver.step_completed(driver.verify()[0])
