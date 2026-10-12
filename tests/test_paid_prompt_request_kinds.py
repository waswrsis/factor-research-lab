"""Real original wallet/bridge, bounded synthetic replies; no provider or fitting."""
import copy
import json
from dataclasses import asdict, replace
from pathlib import Path
import sys

import pytest

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0,str(ROOT/'tests'))
from test_v7_prompt_builder import inputs, response
from test_v7_run_factor_pool import source_root
from test_v7_s_controller_integration import case as integration_case, staged
from factorlab.discovery import prompt_request_kinds as kinds
from factorlab.discovery.joint_pool import FrozenPool
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
from factorlab.discovery.prompt_builder import build_formula_prompt
from factorlab.discovery.rolling import RollingSpec
from factorlab.discovery.scheduled_search import FrozenSWorkSchedule, ScheduledSearch
from factorlab.discovery.v7_agent import BackendReply, propose_batch as original_propose_batch
from factorlab.discovery.v7_memory import search_exposure_context
from factorlab.discovery.v7_run import V7Config, RunStore, monthly_budget
from factorlab.discovery.work_authority_bridge import WorkAuthorityBridge
from factorlab.protocol import fingerprint


def propose_batch(packet, *, audit, persist_budget, budget, **kwargs):
    local=copy.deepcopy(audit)
    def save(value):
        budget.bridge.state['agent_audit']=copy.deepcopy(local)
        persist_budget(value)
    return original_propose_batch(packet,audit=local,persist_budget=save,budget=budget,**kwargs)


def request_structure(packet, *, audit, persist_budget, budget, **kwargs):
    local=copy.deepcopy(audit)
    def save(value):
        budget.bridge.state['agent_audit']=copy.deepcopy(local)
        persist_budget(value)
    return kinds.request_structure(packet,audit=local,persist_budget=save,budget=budget,**kwargs)


@pytest.fixture
def case(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    config=V7Config(search_mode='gpt',evaluation_mode='joint',nonlinear_mode='isolated')
    store=RunStore(tmp_path/'r',config,RollingSpec())
    authority=MonthlyWorkAuthority(tmp_path/'w',run_id='request-test',task='volatility',universe=['BTCUSDT'],partition_policy='standard')
    month='2026-08'
    exp=search_exposure_context('source1',month+'-01T00:00:00+00:00',['BTCUSDT'])
    store.exposure.record(**{k:v for k,v in exp.items() if k!='data_id'},role='S')
    state={'status':'running','cutoff':month+'-01T00:00:00+00:00','data_id':'source1',
        'search_exposure':exp,'budget':monthly_budget(config,month).to_dict(),
        'agent_audit':[],'frozen_reference_pool':FrozenPool().to_dict(),'nonlinear_input_data_id':'artificial-input'}
    store.save(month,state,'original')
    def persist(value):
        state['budget']=value
        store.save(month,state,'projection')
    bridge=WorkAuthorityBridge(store,month,state,authority,persist)
    def run(family, operation):
        schedule=FrozenSWorkSchedule.freeze(config=asdict(config),limits=monthly_budget(config,month).limits,
            slots=[{'key':f'family:{family}','stage':'family','counts':{'gpt_attempt':2,'family':1,'ast':32},'seconds':300}])
        scheduler=ScheduledSearch(bridge,schedule)
        return scheduler.run(f'family:{family}',operation,input_binding={'index':family})
    yield bridge,run,persist
    store.close()


def first(budget, persist, *, invalid=False):
    kwargs=inputs();kwargs['budget']=budget
    packet=build_formula_prompt(**kwargs)
    value=response(kwargs)
    if invalid:
        value['identity']['task_id']='wrong'
    reply=BackendReply(json.dumps(value))
    try:
        batch=propose_batch(packet,backend=lambda *a,**k:reply,budget=budget,
            feature_names=('returns',),allowed_parent_ids={'p0','p1'},
            audit=budget.bridge.state['agent_audit'],persist_budget=persist)
    except ValueError:
        if not invalid: raise
        batch=None
    return packet,reply,batch


def structural_reply(packet, **kwargs):
    return BackendReply(json.dumps({'version':kinds.VERSION,
        'recipe_id':next(iter(json.loads(packet.body)['recipes'])),
        'mechanism':'Past S persistence.', 'competing_explanation':'Redundant information.',
        'falsification':'Independent original recipe gates fail.'}))


def test_actual_paid_structural_reply_original_completed_readback(case):
    bridge,run,persist=case
    def operation(budget):
        original,_,_=first(budget,persist)
        packet=kinds.structural_packet(original,budget)
        original_static=json.loads(original.prefix.split('\nSTATIC_CONTRACT_JSON\n')[1])
        retained=json.loads(packet.prefix.split('STATIC_TASK_CONTRACT_JSON\n')[1])
        assert retained=={k:original_static[k] for k in ('task_id','rolling_id','feature_contract_id',
            'training','task_semantics','feature_schema','input_semantics')}
        assert retained['feature_schema'][0]['unit']=='log ratio'
        assert retained['training']=={'calendar_months':3,'timezone':'UTC','initialization':'cold'}
        assert 'output_contract' not in retained
        request_structure(packet,budget=budget,backend=structural_reply,
            audit=bridge.state['agent_audit'],persist_budget=persist)
        with pytest.raises(ValueError,match='second family slot'):
            request_structure(packet,budget=budget,backend=structural_reply,
                audit=bridge.state['agent_audit'],persist_budget=persist)
        return {'structural_audit_hash':fingerprint(bridge.state['agent_audit'][-1])}
    run(2,operation)
    read=kinds.verified_recommendation(bridge.store,bridge.state,bridge.identity)
    assert read['recipe_id'] and read['reservation_id']
    assert bridge.actual_counts['gpt_attempt']==2
    with bridge.authority._transaction() as db:
        row=next(iter(bridge.authority._verify(db)['reservations'].values()))
    assert row['lease']['lease_seconds']==300
    assert row['counts']['gpt_attempt']==2 and row['actual_counts']['gpt_attempt']==2


def test_semantic_revision_is_new_ast_and_failed_attempt_stays(case):
    bridge,run,persist=case
    def operation(budget):
        original,reply,_=first(budget,persist,invalid=True)
        packet=kinds.semantic_packet(original,reply,budget)
        assert packet.telemetry()['request_kind']=='semantic_review'
        kwargs=inputs();value=response(kwargs)
        value['family_id']='newfamily'
        value['programs'][0]['expressions']=['mean(f("returns"))']
        value['template']['base']=copy.deepcopy(value['programs'][0])
        result=propose_batch(packet,backend=lambda *a,**k:BackendReply(json.dumps(value)),budget=budget,
            feature_names=('returns',),allowed_parent_ids={'p0','p1'},audit=bridge.state['agent_audit'],persist_budget=persist)
        assert result.family_id=='newfamily'
        return {'new_id':result.programs[0].digest}
    run(0,operation)
    assert [a['status'] for a in bridge.state['agent_audit']]==['failed','validated']
    assert bridge.actual_counts['gpt_attempt']==2 and bridge.actual_counts['ast']==2


@pytest.mark.parametrize('mutation',['same_ast','same_family'])
def test_semantic_does_not_relabel_rejected_candidate(case,mutation):
    bridge,run,persist=case
    def operation(budget):
        original,reply,_=first(budget,persist,invalid=True)
        packet=kinds.semantic_packet(original,reply,budget)
        value=response(inputs())
        if mutation=='same_ast': value['family_id']='different'
        else:
            value['programs'][0]['expressions']=['mean(f("returns"))']
            value['template']['base']=copy.deepcopy(value['programs'][0])
        with pytest.raises(ValueError,match='new AST'):
            propose_batch(packet,backend=lambda *a,**k:BackendReply(json.dumps(value)),budget=budget,
                feature_names=('returns',),allowed_parent_ids={'p0','p1'},audit=bridge.state['agent_audit'],persist_budget=persist)
        return {}
    run(0,operation)
    assert [a['status'] for a in bridge.state['agent_audit']]==['failed','failed']
    assert bridge.actual_counts['gpt_attempt']==2 and not bridge._budget.families


@pytest.mark.parametrize('role',['V','C'])
def test_specialized_packet_rejects_cross_exposure(case,role):
    bridge,run,persist=case
    def operation(budget):
        original,_,_=first(budget,persist)
        body=json.loads(original.body);body['parents'][0]['evidence']['role']=role
        forged=replace(original,body=json.dumps(body))
        with pytest.raises(ValueError): kinds.structural_packet(forged,budget)
        return {}
    run(2,operation)
    assert bridge.actual_counts['gpt_attempt']==1


def test_no_paid_budget_or_unregistered_kind():
    packet=build_formula_prompt(**inputs())
    with pytest.raises(TypeError,match='paid S family'):
        kinds.structural_packet(packet,inputs()['budget'])
    class Spoof(type(packet)): pass
    with pytest.raises(TypeError,match='Unregistered'):
        kinds.validate_request(Spoof(packet.prefix,packet.body,packet.identity),inputs()['budget'])


@pytest.mark.parametrize('mutation',['illegal_recipe','extra_permission','late','wrong_family'])
def test_rejected_structure_no_retry_no_refund(case,mutation,monkeypatch):
    bridge,run,persist=case
    def operation(budget):
        original,_,_=first(budget,persist)
        if mutation=='wrong_family':
            with pytest.raises(ValueError,match='second family slot'): kinds.structural_packet(original,budget)
            return {}
        packet=kinds.structural_packet(original,budget)
        original_check=budget.check_time
        def reply(p,**kw):
            value=json.loads(structural_reply(p).payload)
            if mutation=='illegal_recipe':value['recipe_id']='caller-recipe'
            elif mutation=='extra_permission':value['serving_eligible']=True
            elif mutation=='late':budget.check_time=lambda:(_ for _ in ()).throw(TimeoutError('original deadline'))
            return BackendReply(json.dumps(value))
        with pytest.raises((ValueError,TimeoutError)):
            request_structure(packet,budget=budget,backend=reply,audit=bridge.state['agent_audit'],persist_budget=persist)
        if mutation=='late': budget.check_time=original_check
        return {'structural_audit_hash':fingerprint(bridge.state['agent_audit'][-1])}
    run(1 if mutation=='wrong_family' else 2,operation)
    assert bridge.actual_counts['gpt_attempt']==(1 if mutation=='wrong_family' else 2)
    if mutation!='wrong_family': assert bridge.state['agent_audit'][-1]['status']=='failed'


def test_real_paid_review_uses_recommendation_only_after_terminal_feedback(case,monkeypatch):
    from factorlab.discovery import structural_evolution as evolution
    from factorlab.discovery.neural_forward import FrozenNeuralRecipe
    from factorlab.discovery.monthly_schedule import default_schedule
    bridge,run,persist=case
    month='2026-07';store=bridge.store
    config=V7Config(**store.identity['config'])
    exp=search_exposure_context('prior-input',month+'-01T00:00:00+00:00',['BTCUSDT'])
    store.exposure.record(**{k:v for k,v in exp.items() if k!='data_id'},role='S')
    prior={'status':'running','cutoff':month+'-01T00:00:00+00:00','data_id':'prior-input',
        'search_exposure':exp,'budget':monthly_budget(config,month).to_dict(),'agent_audit':[],
        'frozen_reference_pool':FrozenPool().to_dict(),'nonlinear_input_data_id':'prior-artificial'}
    store.save(month,prior,'prior-input')
    def prior_persist(value):
        prior['budget']=value;store.save(month,prior,'prior-projection')
    old=WorkAuthorityBridge(store,month,prior,bridge.authority,prior_persist)
    ScheduledSearch(old,default_schedule(config,month=month))
    base=FrozenNeuralRecipe()
    evolution.select_structure(old,base)
    def operation(budget):
        original,_,_=first(budget,persist)
        packet=kinds.structural_packet(original,budget)
        def choose(p,**kw):
            value=json.loads(structural_reply(p).payload)
            value['recipe_id']=next(k for k,v in json.loads(p.body)['recipes'].items()
                if v['config']['structures'][0]['width']==4 and v['output']=='residual')
            return BackendReply(json.dumps(value))
        request_structure(packet,budget=budget,backend=choose,audit=bridge.state['agent_audit'],persist_budget=persist)
        return {'structural_audits_only':True,'structural_audit_hash':fingerprint(bridge.state['agent_audit'][-1])}
    run(2,operation)
    def feedback(b,*args):
        row=b.authority.get(b._active.record['reservation_id'])
        assert row['status']=='running' and row['counts']['solve']==1 and row['lease']['lease_seconds']==300
        return {'terminal':True,'eligible':True,'reason':'explicit artificial feedback seam; no trained evidence'}
    monkeypatch.setattr(evolution,'_feedback',feedback)
    selected=evolution.select_structure(bridge,base)
    assert selected['body']['gpt_proposed'] is True
    assert selected['body']['recipe']['config']['structures'][0]['width']==4
    assert selected['body']['serving_eligible'] is False
    recipe=FrozenNeuralRecipe.from_dict(selected['body']['recipe'])
    assert evolution.assert_selected_structure(store,bridge.state,bridge.identity,recipe)
    assert evolution.select_structure(bridge,base)==selected


def test_direct_specialized_backend_cannot_bypass_prepayment(case,tmp_path):
    from factorlab.discovery.v7_agent import CodexBatchBackend
    bridge,run,persist=case
    def operation(budget):
        original,_,_=first(budget,persist)
        packet=kinds.structural_packet(original,budget)
        with pytest.raises(ValueError,match='prepaid invocation'):
            CodexBatchBackend(tmp_path/'not-created')(packet,model=packet.model,reasoning=packet.reasoning)
        assert not (tmp_path/'not-created').exists()
        return {}
    run(2,operation)


def test_backend_preserves_structural_typed_schema_and_wire(case,tmp_path,monkeypatch):
    from factorlab.discovery.v7_agent import CodexBatchBackend
    bridge,run,persist=case
    def operation(budget):
        original,_,_=first(budget,persist)
        packet=kinds.structural_packet(original,budget)
        backend=CodexBatchBackend(tmp_path/'raw',executable='never-executed')
        def local_transport(command,text,runtime):
            schema=json.loads((runtime/'response-schema.json').read_text())
            assert 'recipe_id' in schema['properties'] and 'programs' not in schema['properties']
            return '\n'.join(json.dumps(e) for e in [{'type':'turn.started'},
                {'type':'item.completed','item':{'type':'agent_message','text':structural_reply(packet).payload}},
                {'type':'turn.completed'}])
        monkeypatch.setattr(backend,'_run',local_transport)
        request_structure(packet,budget=budget,backend=backend,audit=bridge.state['agent_audit'],persist_budget=persist)
        return {'structural_audit_hash':fingerprint(bridge.state['agent_audit'][-1])}
    run(2,operation)
    assert kinds.verified_recommendation(bridge.store,bridge.state,bridge.identity)


@pytest.mark.parametrize('mutation',['missing_mirror','changed_audit','missing_audit'])
def test_original_structural_recommendation_cannot_be_rebuilt(case,mutation):
    bridge,run,persist=case
    def operation(budget):
        original,_,_=first(budget,persist)
        packet=kinds.structural_packet(original,budget)
        request_structure(packet,budget=budget,backend=structural_reply,audit=bridge.state['agent_audit'],persist_budget=persist)
        return {'structural_audit_hash':fingerprint(bridge.state['agent_audit'][-1])}
    run(2,operation)
    assert kinds.verified_recommendation(bridge.store,bridge.state,bridge.identity)
    if mutation=='missing_mirror':
        path=bridge.anchor/'specialized_requests/structural_probe.json'
        path.rename(path.with_suffix('.retained'))
    else:
        if mutation=='changed_audit':bridge.state['agent_audit'][-1]['structural_reply']['mechanism']='Rehashed different reason.'
        else:bridge.state['agent_audit'].pop()
        persist(bridge.state['budget'])
    with pytest.raises(ValueError): kinds.verified_recommendation(bridge.store,bridge.state,bridge.identity)


def test_recommendation_read_inside_original_wallet_verification_uses_same_transaction(case,monkeypatch):
    from factorlab.discovery import monthly_work_authority as wallet
    bridge,run,persist=case
    def operation(budget):
        original,_,_=first(budget,persist)
        packet=kinds.structural_packet(original,budget)
        request_structure(packet,budget=budget,backend=structural_reply,audit=bridge.state['agent_audit'],persist_budget=persist)
        return {'structural_audit_hash':fingerprint(bridge.state['agent_audit'][-1])}
    run(2,operation)
    original=wallet.allocation_route
    seen=[]
    def routed(*args,**kwargs):
        assert wallet._IDENTITY_READ_SCOPE.get()['db'].in_transaction
        seen.append(kinds.verified_recommendation(bridge.store,bridge.state,bridge.identity)['reservation_id'])
        return original(*args,**kwargs)
    monkeypatch.setattr(wallet,'allocation_route',routed)
    assert bridge.authority.verify()['reservations']==1
    assert seen


def test_standard_kinds_do_not_change_control_allocations():
    from factorlab.discovery.monthly_schedule import default_schedule,default_control_schedules
    schedule=default_schedule(V7Config(search_mode='gpt',evaluation_mode='joint',nonlinear_mode='isolated'),month='2026-08').to_dict()
    assert schedule['allocated_counts']['gpt_attempt']==6
    assert schedule['allocation']['request_kind_policy']==kinds.POLICY
    for arm,control in default_control_schedules(symbols=['BTCUSDT'],month='2026-08').items():
        assert control.to_dict()['allocated_counts']['gpt_attempt']==2
        assert 'request_kind_policy' not in control.to_dict()['allocation']


def test_actual_monthly_loop_dispatches_both_paid_request_kinds(integration_case,monkeypatch):
    from test_v7_s_controller_integration import FakeBatch,run,read_state
    c=integration_case
    c.config=replace(c.config,nonlinear_mode='isolated')
    schedule=c.schedule.to_dict()
    schedule['allocation']['nonlinear_seconds_holdback']=900
    c.schedule=FrozenSWorkSchedule.freeze(config=asdict(c.config),limits=schedule['limits'],
        slots=schedule['slots'],allocation=schedule['allocation'])
    formula=FakeBatch(first_invalid=True)
    seen=[]
    def backend(packet,**kwargs):
        seen.append(packet.telemetry()['request_kind'])
        if type(packet) is kinds.StructuralPacket: return structural_reply(packet)
        return formula(packet,**kwargs)
    readbacks=[]
    def no_training(bridge,*args,**kwargs):
        readbacks.append(kinds.verified_recommendation(bridge.store,bridge.state,bridge.identity))
        return {'status':'test_only_no_nonlinear_training'}
    monkeypatch.setattr('factorlab.discovery.s_nonlinear_controller.run_shared_nonlinear',no_training)
    run(c,backend)
    assert seen==['formula_family','semantic_review','formula_family','formula_family','structural_probe']
    state=read_state(c)
    assert state['budget']['counts']['gpt_attempt']==5 and readbacks[0]['recipe_id']
    assert [a['status'] for a in state['agent_audit']]==['failed','validated','validated','validated','validated']
    with c.authority._transaction() as db:
        families=[r for r in c.authority._verify(db)['reservations'].values()
            if r['binding'].get('work',{}).get('stage')=='family']
    assert [r['actual_counts']['gpt_attempt'] for r in families]==[2,1,2]
    assert all(r['lease']['lease_seconds']==300 and r['counts']['gpt_attempt']==2 for r in families)
