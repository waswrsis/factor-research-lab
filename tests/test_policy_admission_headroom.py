"""Bounded engineering boundaries, never genuine provider/Linux acceptance.

Admission uses the existing explicit artificial kernel/ticket/publication seams
plus actual registry and authority. Service tests use original local service,
SQLite, runtime and wallet; only the downstream work/selection is disclosed.
"""
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from test_v7_neural_forward import fitted
from test_v7_neural_forward_admission import evidence, reseal
from test_v7_run_factor_pool import source_root
from factorlab.protocol import fingerprint
from factorlab.discovery import neural_forward_admission as admission
from factorlab.discovery.model_registry import ModelRegistry
from factorlab.discovery.v7_forward import ForwardRuntime
from factorlab.discovery.v7_service import ForwardService, ServiceConfig
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
from factorlab.discovery.known_linear_reference import FrozenLinearRecipe
from factorlab.discovery.work_lease import FrozenWorkLease
from factorlab.discovery.forward_service_scope import _service_step_scope, check_service_work, ServiceDeadlineClosed


def policy_evidence(e):
    e.context.update(purpose='policy_trial', stream='A0_policy_trial')
    e.receipt['context']=copy.deepcopy(e.context)
    reseal(e)


def test_policy_purpose_reaches_actual_admission_registry_and_readback(evidence, tmp_path, monkeypatch):
    policy_evidence(evidence)
    proof=admission.verify_forward_admission(evidence.model,evidence.cert,
        expected_job='job',expected_pipeline=evidence.context['pipeline'])
    assert proof['publication']==evidence.proof and proof['context']['purpose']=='policy_trial'
    registry=ModelRegistry(tmp_path/'models')
    identifier=registry.save(evidence.model,evidence.context['pipeline'],certificate=evidence.cert,expected_job='job')
    before=evidence.authority.verify()
    monkeypatch.setattr(FrozenWorkLease,'issue',lambda *a,**k:pytest.fail('Historical readback may not issue time'))
    restored,identity=registry.load(identifier)
    assert restored.digest==evidence.model.digest and identity['job']=='job'
    assert evidence.authority.verify()==before


@pytest.mark.parametrize('damage',['ordinary_A0','resource','serving_flag','wrong_stream','raw_context','missing_publication','payload'])
def test_policy_context_flags_or_receipt_hash_cannot_replace_original_evidence(evidence,damage):
    policy_evidence(evidence)
    if damage=='ordinary_A0': evidence.context['purpose']='A0'
    elif damage=='resource': evidence.context['purpose']='resource_trial'
    elif damage=='serving_flag': evidence.context['serving_eligible']=False
    elif damage=='wrong_stream': evidence.context['stream']='A0'
    elif damage=='raw_context': evidence.receipt['context']['purpose']='A0'
    elif damage=='missing_publication': evidence.proof['certificate_digest']='0'*64
    elif damage=='payload': evidence.cert['payload_digest']='0'*64
    if damage not in {'raw_context','missing_publication','payload'}:
        evidence.receipt['context']=copy.deepcopy(evidence.context); reseal(evidence)
    elif damage=='raw_context': reseal(evidence)
    with pytest.raises(ValueError):
        admission.verify_forward_admission(evidence.model,evidence.cert)


@pytest.fixture
def actual_service(tmp_path,monkeypatch,request):
    source_root(tmp_path,monkeypatch)
    runtime=ForwardRuntime(tmp_path/'r',tmp_path/'f')
    wallet=MonthlyWorkAuthority(tmp_path/'w',run_id='headroom',task='volatility',universe=['BTCUSDT'],
        partition_policy='standard' if getattr(request,'param','known')=='standard' else None)
    service=ForwardService(runtime,config=ServiceConfig(poll=False,wall_seconds=7200),
        guard=lambda:{'allowed':True},work_authority=wallet)
    if getattr(request,'param','known')=='neural':
        from factorlab.discovery.neural_forward import FrozenNeuralRecipe
        recipe=FrozenNeuralRecipe(rolling_spec=runtime.spec)
    else:
        recipe=FrozenLinearRecipe(name='K0',rolling_spec=runtime.spec)
    cutoff=runtime.lifecycle.now().normalize().replace(day=1)
    job=runtime.lifecycle.plan_fit(cutoff,'A0',runtime.recipe_for(recipe))
    try: yield SimpleNamespace(runtime=runtime,wallet=wallet,service=service,recipe=recipe,job=job,cutoff=cutoff)
    finally: service.close()


@pytest.mark.parametrize('remaining',[.2,400.])
def test_actual_wallet_new_dispatch_headroom_and_sufficient_original_300_lease(actual_service,monkeypatch,remaining):
    c=actual_service; calls=[]
    # The actual issuer runs until its original prepaid reservation exists;
    # stop before source snapshot/numerics. This is not a completed model.
    from factorlab.discovery.forward_preparation import ForwardPreparation
    def stop(preparation):
        record=preparation.record
        calls.append(record)
        raise ValueError('disclosed-stop-after-original-prepayment')
    monkeypatch.setattr(ForwardPreparation,'snapshot',stop)
    c.service.state['spent_seconds']=7200-remaining;c.service._save('existing_service_charge_fixture')
    before=c.wallet.verify()['reservations']
    closed=False
    try:
        with _service_step_scope(c.service):
            c.service._fit_pending(c.runtime.lifecycle.now(),[],time.monotonic()+remaining)
    except ServiceDeadlineClosed:
        # With 0.2 seconds of native time left, setup and durable IO can
        # exhaust the service. That negative terminal is required behavior,
        # not evidence that the unfunded 300-second dispatch was started.
        assert remaining <= 302
        closed=True
    assert len(calls)==int(remaining>302)
    assert c.wallet.verify()['reservations']==before+int(remaining>302)
    if calls:
        assert calls[0]['lease']['lease_seconds']==300
        assert c.service.state['fit_attempts'][c.job]['count']==1
    else: assert c.job not in c.service.state['fit_attempts']
    if closed:
        assert c.service.state['spent_seconds'] >= c.service.config.wall_seconds
        assert c.service.state['last_result']['status']=='budget_exhausted'


def test_exhausted_native_headroom_closes_without_prepaid_dispatch(actual_service,monkeypatch):
    c=actual_service
    c.service.state['spent_seconds']=7199.8;c.service._save('short_native_headroom_fixture')
    before=c.wallet.verify()
    with pytest.raises(ServiceDeadlineClosed):
        with _service_step_scope(c.service):
            time.sleep(.25)
            c.service._fit_pending(c.runtime.lifecycle.now(),[],time.monotonic()+.2)
    assert c.wallet.verify()==before
    assert c.job not in c.service.state['fit_attempts']
    assert c.service.state['spent_seconds']>=c.service.config.wall_seconds


def test_slow_durable_attempt_save_cannot_start_work_and_attempt_is_not_refunded(actual_service,monkeypatch):
    c=actual_service; offset=[0.]
    monkeypatch.setattr(c.service,'monotonic',lambda:100.+offset[0])
    saved=c.service._save
    def slow(event):
        saved(event)
        if event=='fit_attempt_reserved':offset[0]=399.9
    monkeypatch.setattr(c.service,'_save',slow)
    monkeypatch.setattr(c.runtime,'fit_month',lambda *a,**k:pytest.fail('Expired dispatch started'))
    actions=[];before=c.wallet.verify()
    c.service._fit_pending(c.runtime.lifecycle.now(),actions,500.)
    assert actions[-1]['status']=='insufficient_service_headroom'
    assert c.service.state['fit_attempts'][c.job]['count']==1
    assert c.wallet.verify()==before


def test_native_scope_cannot_be_extended_by_injected_service_clock(actual_service,monkeypatch):
    c=actual_service
    c.service.state['spent_seconds']=7199.8;c.service._save('existing_native_charge_fixture')
    monkeypatch.setattr(c.service,'monotonic',lambda:0.)
    before=c.wallet.verify()
    from factorlab.discovery.known_linear_forward import issue_known_ticket
    with _service_step_scope(c.service):
        with pytest.raises(ServiceDeadlineClosed):
            issue_known_ticket(c.runtime,c.wallet,c.job,c.recipe)
    assert c.wallet.verify()==before
    assert not list((c.wallet.directory/'known_forward_tickets').glob('*/ticket.json'))


@pytest.mark.parametrize('actual_service',['neural'],indirect=True)
@pytest.mark.parametrize('boundary',['gate','resource_save'])
def test_slow_gate_or_resource_intent_cannot_start_a_new_paid_dispatch(actual_service,monkeypatch,boundary):
    c=actual_service; offset=[0.]
    monkeypatch.setattr(c.service,'monotonic',lambda:100.+offset[0])
    from factorlab.discovery import forward_gate_resolver, forward_resource_work
    def resolve(*a,**k):
        if boundary=='resource_save':
            raise forward_gate_resolver.ForwardGatesPending('waiting_original_same_recipe_full_forward_resource_workload')
        offset[0]=399.9
        return {'version':'disclosed-routing-fixture','job':c.job,'pipeline_id':c.recipe.pipeline_id,
                'source_hash':'fixture','authority_manifest_hash':c.wallet.manifest_hash,'gates':{}}
    monkeypatch.setattr(forward_gate_resolver,'resolve_forward_gates',resolve)
    saved=c.service._save
    def slow_save(event):
        saved(event)
        if event=='full_forward_resource_attempt_reserved':offset[0]=399.9
    monkeypatch.setattr(c.service,'_save',slow_save)
    monkeypatch.setattr(c.runtime,'fit_month',lambda *a,**k:pytest.fail('Gate delay escaped deadline'))
    monkeypatch.setattr(forward_resource_work,'produce_resource_work',lambda *a,**k:pytest.fail('Slow resource save escaped deadline'))
    before=c.wallet.verify();actions=[]
    c.service._fit_pending(c.runtime.lifecycle.now(),actions,500.)
    assert any(a['status']=='insufficient_service_headroom' for a in actions)
    assert c.wallet.verify()==before
    if boundary=='resource_save': assert c.service.state['fit_attempts']['resource:'+c.job]['count']==1


def test_exact_post_issue_check_cannot_extend_300s_after_preissue_check(actual_service,monkeypatch):
    c=actual_service
    c.service.state['spent_seconds']=7200-302.2;c.service._save('original_charge_fixture')
    original=FrozenWorkLease.issue.__func__;count=[0]
    def delayed(cls,*a,**kw):
        count[0]+=1
        # The service constructor already sampled its native origin.
        if count[0]==1:time.sleep(2.4)
        return original(cls,*a,**kw)
    monkeypatch.setattr(FrozenWorkLease,'issue',classmethod(delayed))
    from factorlab.discovery.known_linear_forward import issue_known_ticket
    before=c.wallet.verify()
    with _service_step_scope(c.service):
        with pytest.raises(ServiceDeadlineClosed,match='300s lease'):
            issue_known_ticket(c.runtime,c.wallet,c.job,c.recipe)
    assert c.wallet.verify()==before


def test_expired_scope_cannot_reach_new_model_seal(actual_service,monkeypatch):
    c=actual_service
    from factorlab.discovery import known_linear_forward
    def late(*a,**kw):
        time.sleep(.25)
        return {'model':object(),'certificate':{},'cached':False}
    monkeypatch.setattr(known_linear_forward,'fit_known_forward',late)
    monkeypatch.setattr(c.runtime.lifecycle,'seal_model',lambda *a,**k:pytest.fail('Late model sealed'))
    c.service.state['spent_seconds']=7199.8;c.service._save('existing_native_charge_fixture')
    with pytest.raises(ServiceDeadlineClosed):
        with _service_step_scope(c.service):
            with pytest.raises(ServiceDeadlineClosed):
                c.runtime.fit_month(c.cutoff,pool=c.recipe,authority=c.wallet)
    assert c.runtime.lifecycle.db.execute('SELECT failure FROM jobs WHERE id=?',(c.job,)).fetchone()[0]=='integrity_failure'


def test_original_completed_registry_recovery_does_not_reserve_new_attempt(actual_service,monkeypatch):
    c=actual_service
    # Disclosed already-admitted registry seam: this tests scheduling only.
    # Actual historical admission/registry no-lease readback is tested above.
    with c.runtime.lifecycle.db:
        c.runtime.lifecycle.db.execute('UPDATE jobs SET status=?,failure=? WHERE id=?',('verified','data_incomplete',c.job))
    c.service.state['fit_attempts'][c.job]={'count':3,'last_at':c.runtime.lifecycle.now().isoformat()}
    c.service._save('original_attempts_fixture')
    calls=[]
    monkeypatch.setattr(c.runtime,'fit_month',lambda *a,**k:calls.append(k) or {'cached':True})
    before=c.wallet.verify()
    c.service._fit_pending(c.runtime.lifecycle.now(),[],time.monotonic()+.2)
    assert len(calls)==1 and c.service.state['fit_attempts'][c.job]['count']==3
    assert c.wallet.verify()==before


# Keep timing tests on real native clocks; downstream numerical work is omitted.
def quiet_step(c, monkeypatch):
    monkeypatch.setattr(c.runtime, 'tick', lambda **kw: {'fixture': 'no_market_data'})
    monkeypatch.setattr(c.service, '_freeze_due', lambda *a: None)
    monkeypatch.setattr(c.service, '_endpoints', lambda *a: None)
    monkeypatch.setattr(c.service, '_plan_month', lambda *a: None)
    monkeypatch.setattr(c.service, '_algorithm_step', lambda *a, **kw: 0)
    monkeypatch.setattr(c.service, '_fit_pending', lambda *a, **kw: None)


@pytest.mark.parametrize('boundary', ['setup', 'guard', 'body_save', 'closing_save'])
def test_native_continuous_ceiling_includes_slow_setup_and_final_IO_across_steps(actual_service, monkeypatch, boundary):
    c=actual_service; quiet_step(c,monkeypatch)
    c.service.state['spent_seconds']=7199.8; c.service._save('original_charge_fixture')
    original_save=c.service._save; delayed=[False]
    event={'setup':'native_service_clock_bound', 'body_save':'step_finished',
           'closing_save':'native_service_time_observed'}.get(boundary)
    def save(kind):
        original_save(kind)
        if kind==event and not delayed[0]:
            delayed[0]=True;time.sleep(.25)
    monkeypatch.setattr(c.service,'_save',save)
    if boundary=='guard':
        def guard():
            time.sleep(.25);return {'allowed':True}
        monkeypatch.setattr(c.service,'guard',guard)
    before=c.wallet.verify()
    if boundary=='closing_save':
        # This writer never claims its own IO tail is self-certified. The next
        # entry uses the SAME absolute origin and cannot borrow that tail back.
        with pytest.raises(ServiceDeadlineClosed):c.service.step(defer_fits=True)
        first=json.loads((c.service.directory/'native-clock.json').read_text())
        assert c.service.step(defer_fits=True)['status']=='budget_exhausted'
        assert json.loads((c.service.directory/'native-clock.json').read_text())==first
    else:
        with pytest.raises(ServiceDeadlineClosed):c.service.step(defer_fits=True)
    assert c.service.state['spent_seconds']>=7200
    assert c.service.step(defer_fits=True)['status']=='budget_exhausted'
    assert c.wallet.verify()==before


@pytest.mark.parametrize('damage',['clock','state','both'])
def test_original_native_clock_or_state_loss_never_reinitializes(actual_service, monkeypatch, damage):
    c=actual_service;quiet_step(c,monkeypatch)
    c.service.step(defer_fits=True)
    clock=c.service.directory/'native-clock.json'
    if damage in {'clock','both'}:clock.rename(clock.with_suffix('.retained'))
    if damage in {'state','both'}:
        del c.service.state['native_service_clock'];c.service._save('disclosed_lost_clock_projection')
    with pytest.raises(ValueError,match='never renew|cannot restart'):
        c.service.step(defer_fits=True)
    assert clock.exists()==(damage=='state')


def test_timely_steps_keep_first_native_origin_and_STOP_creates_no_clock(actual_service,monkeypatch):
    c=actual_service;quiet_step(c,monkeypatch)
    stop=c.service.directory/'STOP';stop.write_text('stop')
    assert c.service.step()['status']=='stopped'
    assert not (c.service.directory/'native-clock.json').exists()
    stop.rename(stop.with_suffix('.retained'))
    before=c.wallet.verify()
    assert c.service.step(defer_fits=True)['status']=='stepped'
    first=json.loads((c.service.directory/'native-clock.json').read_text())
    time.sleep(.02)
    assert c.service.step(defer_fits=True)['status']=='stepped'
    assert json.loads((c.service.directory/'native-clock.json').read_text())==first
    assert c.service.state['spent_seconds']>=.02 and c.wallet.verify()==before


def test_original_parent_cannot_be_adopted_by_live_direct_child(actual_service,monkeypatch):
    c=actual_service;quiet_step(c,monkeypatch)
    c.service.step(defer_fits=True)
    original=json.loads((c.service.directory/'native-clock.json').read_text())
    # Real subprocess reopens the original durable service and ledger after
    # parent relinquishes handles. Parent is alive: direct-child lease access
    # alone must not grant ownership of the parent's service session.
    import subprocess, sys
    root=c.runtime.directory.parent
    c.service.close(close_runtime=False)
    c.runtime.close()
    script="""
import sys
from pathlib import Path
import factorlab.protocol as p
p.ROOT=Path(sys.argv[1])
from factorlab.discovery.v7_forward import ForwardRuntime
from factorlab.discovery.v7_service import ForwardService,ServiceConfig
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
r=ForwardRuntime(p.ROOT/'r',p.ROOT/'f')
w=MonthlyWorkAuthority(p.ROOT/'w',run_id='headroom',task='volatility',universe=['BTCUSDT'])
s=ForwardService(r,config=ServiceConfig(poll=False,wall_seconds=7200),guard=lambda:{'allowed':True},work_authority=w)
try:
    try:s.step(defer_fits=True)
    except ValueError as e:
        assert type(e) is ValueError and str(e)=='Original service deadline scope identity changed'
        print(type(e).__name__)
    else:raise AssertionError('New issuer adopted old session')
    assert s.state['spent_seconds']>=7200
finally:s.close()
"""
    result=subprocess.run([sys.executable,'-c',script,str(root)],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    assert result.stdout.strip()=='ValueError'
    assert json.loads((c.service.directory/'native-clock.json').read_text())==original
    # Fixture teardown already closed; use explicit no-op only for teardown.
    monkeypatch.setattr(c.service,'close',lambda:None)


from test_v7_algorithm_validation import case, completed as original_completed, freeze
@pytest.fixture
def completed(case,monkeypatch):
    from factorlab.discovery import algorithm_shortlist
    from datetime import datetime,UTC
    class RegistrationTime(datetime):
        @classmethod
        def now(cls,tz=None):
            return (pd.Timestamp.now(tz='UTC').normalize().replace(day=1)-pd.Timedelta(days=1)).to_pydatetime()
    # Explicit historical registration premise, before the actual S work.
    # Every lease, physical witness and wallet allocation remains native.
    monkeypatch.setattr(algorithm_shortlist,'datetime',RegistrationTime)
    yield from original_completed.__wrapped__(case,monkeypatch)

@pytest.mark.parametrize('remaining',[.2,400.])
def test_typed_original_preparation_uses_same_native_scope_and_300_lease(completed,monkeypatch,remaining):
    from factorlab.discovery import algorithm_validation as module
    bridge,_,recipe,runtime,_=completed
    start=pd.Timestamp.now(tz='UTC').normalize().replace(day=1)
    monkeypatch.setattr(module,'observation_now',lambda runtime:start)
    monkeypatch.setattr(runtime.lifecycle.exposure,'clock',lambda:start)
    protocol=module.AlgorithmValidation.freeze(runtime,bridge.authority,completed[1],start=start)
    # Prospective logical registration is an explicit fixture within this real
    # wall month; the native wallet allocator still checks its ORIGINAL window.
    # Future logical validation window is an explicit engineering fixture;
    # neither actual native issuer nor native scope clock is changed.
    monkeypatch.setattr(module,'observation_now',lambda runtime:start+pd.Timedelta(seconds=10))
    service=ForwardService(runtime,config=ServiceConfig(poll=False,wall_seconds=7200),
        guard=lambda:{'allowed':True},work_authority=bridge.authority)
    service.state['spent_seconds']=7200-remaining;service._save('existing_charge_fixture')
    before=bridge.authority.verify()['reservations'];records=[]
    def stop(*a,**kw):
        records.append(bridge.authority.verify()['reservations'])
        raise OSError('disclosed-stop-after-original-typed-prepayment')
    monkeypatch.setattr(module,'prepare_forward_snapshot',stop)
    try:
        with pytest.raises((ServiceDeadlineClosed,OSError)):
            with _service_step_scope(service):
                protocol.prepare_fit(recipe.pipeline_id,start,gates={'directory':'fixture','artifacts':{}})
        assert bridge.authority.verify()['reservations']==before+int(remaining>302)
        assert len(records)==int(remaining>302)
        if records:
            with bridge.authority._transaction() as db:
                rows=bridge.authority._verify(db)['reservations'].values()
                paid=[r for r in rows if r['work_key'].startswith('algorithm_fit:')]
            assert len(paid)==1 and paid[0]['lease']['lease_seconds']==300
    finally:service.close(close_runtime=False)



def test_actual_long_running_campaign_subclass_retains_original_global_caps(actual_service,monkeypatch):
    from scripts import v7_long_running as long
    c=actual_service
    c.service.close(close_runtime=False)
    monkeypatch.setattr(c.service,'close',lambda:None)
    end=(pd.Timestamp.now(tz='UTC')+pd.Timedelta(days=1)).isoformat()
    plan=long.CampaignPlan(end_at=end,max_sessions=2,max_reserved_seconds=800,
        max_reserved_steps=4,max_fit_attempts=2,session_seconds=400,session_steps=2,
        forward_directory=str(c.runtime.directory),feed_directory=str(c.runtime.feed.directory),
        research={'backend':'disabled','research':{'evaluation_mode':'joint','search_mode':'local'}},
        lifecycle={'source_id':c.runtime.lifecycle.plan.source_id})
    store=long.CampaignStore(c.runtime.directory.parent/'campaign',plan)
    session=store.reserve_session(); assert session
    service=long.CampaignService(c.runtime,store,
        config=ServiceConfig(session_id=session['session_id'],wall_seconds=400,max_steps=2,poll=False),
        guard=lambda:{'allowed':True},work_authority=c.wallet)
    try:
        other=SimpleNamespace(runtime=c.runtime,service=service)
        quiet_step(other,monkeypatch)
        original=copy.deepcopy(store.load())
        assert service.step(defer_fits=True)['status']=='stepped'
        assert service.state['native_service_clock']['service_seconds']==400
        current=store.load()
        assert current['sessions']==original['sessions'] and current['fit_attempts']==original['fit_attempts']
        with pytest.raises(TypeError):
            with _service_step_scope(SimpleNamespace(**service.__dict__)):pass
    finally:service.close();store.close()


@pytest.mark.parametrize('actual_service',['neural'],indirect=True)
def test_original_whole_policy_contract_and_historical_ticket_context(actual_service,monkeypatch):
    from factorlab.discovery import typed_policy_confirmation as policy
    from factorlab.discovery.claim_protocol import ClaimProtocol,FrozenClaimEndpoint
    from factorlab.discovery.neural_forward_contract import assert_frozen_forward_fit
    from factorlab.discovery.neural_forward_ticket import _verify_historical_context
    from factorlab.discovery.joint_pool import FrozenPool
    c=actual_service;lc=c.runtime.lifecycle
    start=lc.now().normalize().replace(day=1)+pd.DateOffset(months=1)
    k0=c.runtime.recipe_for(FrozenPool())
    protocol=ClaimProtocol.freeze(lc,[FrozenClaimEndpoint.library(start,k0,
        {'pipeline_id':fingerprint('benchmark'),'benchmark':'explicit_artificial'}),
        FrozenClaimEndpoint.algorithm(start,k0)])
    monkeypatch.setattr(lc,'clock',lambda:start)
    pipeline=c.runtime.recipe_for(c.recipe)
    # Only selector outcome is artificial. Original frozen six-month endpoints,
    # dual selection records/LC event, job and historical binding are real.
    monkeypatch.setattr(policy,'_selection',lambda *a:(pipeline,{'private_artificial_selector':True}))
    assert policy.candidate_pipeline(lc,start,k0)==pipeline
    job=lc.plan_fit(start,policy.TRIAL_STREAM,pipeline)
    context=assert_frozen_forward_fit(lc,job,c.recipe,purpose='policy_trial')['binding']
    assert context['adoption']['claim_protocol']==protocol.digest
    _verify_historical_context(lc,context,c.recipe)
    for key,value in [('purpose','A0'),('stream','A0'),('serving_eligible',False),('adoption',{})]:
        damaged=copy.deepcopy(context);damaged[key]=value
        with pytest.raises(ValueError):_verify_historical_context(lc,damaged,c.recipe)
    with pytest.raises(ValueError):assert_frozen_forward_fit(lc,job,c.recipe,purpose='A0')


@pytest.mark.parametrize('actual_service',['standard'],indirect=True)
@pytest.mark.parametrize('slow',[False,True])
def test_actual_autonomous_tail_and_pre_spawn_follow_original_service_clock(actual_service,monkeypatch,slow):
    from factorlab.discovery import v7_autonomous as module
    from factorlab.discovery.monthly_schedule import default_schedule
    from factorlab.discovery.v7_run import V7Config
    from test_v7_autonomous import Runner
    c=actual_service;quiet_step(c,monkeypatch)
    config=module.AutonomousConfig(backend='disabled',research=V7Config(evaluation_mode='joint',search_mode='local'))
    runner=Runner() # Explicit no-process transport; production scheduling is real.
    scheduler=module.ResearchScheduler(c.runtime,config=config,runner=runner,
        authority=c.wallet,schedule=default_schedule(config.research,month=c.cutoff.strftime('%Y-%m')))
    # This artificial prepaid fixture must leave headroom at the original
    # native origin, including real constructor/scheduler setup already spent.
    # No clock is injected or renewed. The deliberate delay is sampled only
    # after the actual durable spawn intent exists.
    if slow:
        origin=c.service._native_construction_origin
        elapsed=max(time.monotonic()-origin.monotonic_started,
                    time.time()-pd.Timestamp(origin.wall_started_at).timestamp(),0.)
        c.service.state['spent_seconds']=7200-elapsed-310.
    else:
        c.service.state['spent_seconds']=6800.
    c.service._save('original_charge_fixture')
    original_save=scheduler._save; delayed=[]
    def save(state):
        original_save(state)
        if slow and state['status']=='running':
            from factorlab.discovery.forward_service_scope import _ACTIVE
            scope=_ACTIVE.get()
            assert scope is not None
            remaining=min(scope.ceiling.monotonic_deadline-time.monotonic(),
                pd.Timestamp(scope.ceiling.wall_deadline).timestamp()-time.time())
            assert remaining>302., 'Test setup expired before the intended delayed-save boundary'
            delayed.append(remaining)
            time.sleep(max(0.,remaining-301.8))
    monkeypatch.setattr(scheduler,'_save',save)
    try:
        if slow:
            with pytest.raises(ServiceDeadlineClosed):module.AutonomousService(c.service,scheduler).step()
            state=scheduler._load(c.cutoff.strftime('%Y-%m'))
            assert len(delayed)==1 and len(state['attempts'])==1 and state['attempts'][0]['process'] is None
            assert not runner.starts
        else:
            result=module.AutonomousService(c.service,scheduler).step()
            assert result['research']['actions'][-1]['status']=='spawned'
            state=scheduler._load(c.cutoff.strftime('%Y-%m'));job=module._read(state['job_path'])
            assert job['version']==3 and job['outer_clock']['seconds']==config.deadline_seconds
            from factorlab.discovery.s_outer_context import SOuterContext,effective_deadlines,verify_job_clock
            outer=SOuterContext.autonomous(state['job_path'],0,state['attempts'][0]['token'])
            bound=outer._verify();effective=effective_deadlines(bound['clock'])
            assert effective['monotonic_deadline']<bound['clock']['monotonic_deadline']
            damaged=copy.deepcopy(job);damaged['version']=2
            del damaged['outer_clock']['version'];del damaged['outer_clock']['service_restriction']
            with pytest.raises(ValueError,match='cannot be removed'):verify_job_clock(damaged,state['job_path'])
            # Actual separate interpreter prepares exactly one paid S baseline;
            # it reads the service restriction as a descendant, never adopts
            # the supervisor. It stops before source loading or research work.
            import subprocess,sys
            child="""
import sys,json
from pathlib import Path
import factorlab.protocol as protocol
protocol.ROOT=Path(sys.argv[1])
from factorlab.discovery.v7_autonomous import _read
from factorlab.discovery.s_outer_context import SOuterContext,effective_deadlines
from factorlab.discovery.neural_forward_ticket import open_existing_authority
from factorlab.discovery.s_preparation import SPreparationEnvelope
from factorlab.discovery.scheduled_search import FrozenSWorkSchedule
job=_read(Path(sys.argv[2]));outer=SOuterContext.autonomous(sys.argv[2],0,sys.argv[3])
verified=outer._verify();authority=open_existing_authority(job['work_authority']['directory'])
preparation=SPreparationEnvelope.prepare(authority,job['run_directory'],job['month'],
    config=job['config']['research'],spec=job['rolling'],source_reference=verified['source_reference'],
    schedule_hash=FrozenSWorkSchedule.from_dict(job['work_schedule']).digest,
    clock_scope='before_feed_snapshot',outer_context=outer)
record=preparation.baseline_record();bound=effective_deadlines(verified['clock'])
assert record['lease']['lease_seconds']==300
assert preparation.identity['execution_envelope']['monotonic_deadline']<=bound['monotonic_deadline']
print(json.dumps({'paid_original_lease':300,'effective_monotonic_deadline':bound['monotonic_deadline']}))
"""
            process=subprocess.run([sys.executable,'-c',child,str(c.runtime.directory.parent),
                state['job_path'],state['attempts'][0]['token']],capture_output=True,text=True,timeout=30)
            assert process.returncode==0,process.stdout+process.stderr
            assert json.loads(process.stdout)['paid_original_lease']==300
            original=c.service.directory/'research-restriction.json' 
            original.rename(original.with_suffix('.retained'))
            with pytest.raises((ValueError,OSError)):outer._verify()
    finally:scheduler.close()


@pytest.mark.parametrize('actual_service',['standard'],indirect=True)
def test_research_baseline_preissue_rechecks_original_effective_headroom(actual_service,monkeypatch):
    from factorlab.discovery import v7_autonomous as module
    from factorlab.discovery.monthly_schedule import default_schedule
    from factorlab.discovery.v7_run import V7Config
    from factorlab.discovery.s_outer_context import SOuterContext
    from factorlab.discovery.s_preparation import SPreparationEnvelope
    from test_v7_autonomous import Runner
    c=actual_service;quiet_step(c,monkeypatch)
    config=module.AutonomousConfig(backend='disabled',research=V7Config(evaluation_mode='joint',search_mode='local'))
    schedule=default_schedule(config.research,month=c.cutoff.strftime('%Y-%m'))
    scheduler=module.ResearchScheduler(c.runtime,config=config,runner=Runner(),authority=c.wallet,schedule=schedule)
    c.service.state['spent_seconds']=7200-304.;c.service._save('existing_charge_fixture')
    try:
        module.AutonomousService(c.service,scheduler).step()
        state=scheduler._load(c.cutoff.strftime('%Y-%m'));job=module._read(state['job_path'])
        outer=SOuterContext.autonomous(state['job_path'],0,state['attempts'][0]['token'])
        verified=outer._verify();before=c.wallet.verify()
        time.sleep(4.2)
        monkeypatch.setattr(FrozenWorkLease,'issue',lambda *a,**kw:pytest.fail('Late baseline lease issued'))
        with pytest.raises(TimeoutError,match='full unchanged work lease'):
            SPreparationEnvelope.prepare(c.wallet,job['run_directory'],job['month'],
                config=job['config']['research'],spec=job['rolling'],source_reference=verified['source_reference'],
                schedule_hash=schedule.digest,clock_scope='before_feed_snapshot',outer_context=outer)
        assert c.wallet.verify()==before
    finally:scheduler.close()


def test_dead_original_supervisor_process_cannot_reopen_its_service(actual_service,monkeypatch):
    import subprocess,sys
    c=actual_service;root=c.runtime.directory.parent
    c.service.close();monkeypatch.setattr(c.service,'close',lambda:None)
    child="""
import sys
from pathlib import Path
import factorlab.protocol as p
p.ROOT=Path(sys.argv[1])
from factorlab.discovery.v7_forward import ForwardRuntime
from factorlab.discovery.v7_service import ForwardService,ServiceConfig
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
r=ForwardRuntime(p.ROOT/'r',p.ROOT/'f')
w=MonthlyWorkAuthority(p.ROOT/'w',run_id='headroom',task='volatility',universe=['BTCUSDT'])
s=ForwardService(r,config=ServiceConfig(session_id='original-child',poll=False,wall_seconds=7200),guard=lambda:{'allowed':True},work_authority=w)
try:
    try:s.step(defer_fits=True)
    except Exception as error:assert type(error).__name__=='SnapshotUnavailable'
    assert (s.directory/'native-clock.json').exists()
finally:s.close()
"""
    process=subprocess.run([sys.executable,'-c',child,str(root)],capture_output=True,text=True,timeout=30)
    assert process.returncode==0,process.stdout+process.stderr
    runtime=ForwardRuntime(root/'r',root/'f')
    wallet=MonthlyWorkAuthority(root/'w',run_id='headroom',task='volatility',universe=['BTCUSDT'])
    service=ForwardService(runtime,config=ServiceConfig(session_id='original-child',poll=False,wall_seconds=7200),
        guard=lambda:{'allowed':True},work_authority=wallet)
    try:
        original=(service.directory/'native-clock.json').read_bytes()
        with pytest.raises(ValueError,match='scope identity changed') as rejected:
            service.step(defer_fits=True)
        assert type(rejected.value) is ValueError
        assert service.state['spent_seconds']>=7200
        assert (service.directory/'native-clock.json').read_bytes()==original
        assert service.step(defer_fits=True)['status']=='budget_exhausted'
    finally:service.close()
