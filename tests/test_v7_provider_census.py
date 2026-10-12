"""Original journal + real raw verifier; paid-month/wallet lookups are premise fixtures.

No provider command is executed. These reader-boundary tests do not create an
accepted full registry or scientific certificate.
"""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from factorlab.protocol import fingerprint
from factorlab.discovery.evidence import file_hash
from factorlab.discovery.prompt_builder import PromptPacket, PromptIdentity
from factorlab.discovery.v7_agent import CodexBatchBackend
from factorlab.discovery.work_lease import FrozenWorkLease,_observe
from factorlab.discovery.v7_run import RunStore,V7Config
from factorlab.discovery.rolling import RollingSpec
from factorlab.discovery import full_mechanism_contract as mechanism


@pytest.fixture
def module(monkeypatch):
    name='factorlab.discovery.full_trial_certificate'
    source=Path(__file__).with_name('full_trial_certificate.py')
    if not source.is_file():
        return __import__(name,fromlist=['_search'])
    spec=importlib.util.spec_from_file_location(name,source)
    result=importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules,name,result)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def case(tmp_path_factory,monkeypatch,module):
    root=tmp_path_factory.mktemp('c')
    paths={k:root/k for k in ('research','runtime','provider','authority')}
    for path in paths.values(): path.mkdir()
    store=RunStore(paths['research'],V7Config(evaluation_mode='joint',search_mode='gpt'),RollingSpec())
    states={}; authority_records={}
    policy={'backend':'codex','model_requested':'gpt-6-astra','reasoning_requested':'high',
            'context_policy':'fresh_codex_per_trial_v1'}
    trial={'trial_id':'synthetic-reader-boundary'}
    executable=str(Path(sys.executable).resolve())
    contract={'provider_executable':executable,'provider_executable_sha256':file_hash(executable),
              'reported_resolved_model':'fixture-model','identity_scope':'local_raw_fixture_only'}
    permit=SimpleNamespace(paths=paths,trial=trial,
        registry=SimpleNamespace(plan=SimpleNamespace(provider_policy=policy,mechanism_contract=contract)))
    # Only original complete paid-month readback and wallet lookup are replaced.
    # _search_journal/RunStore.verify and verify_provider_attempt remain real.
    def monthly(directory,month):
        state=states[month]
        audit=state.get('agent_audit',[])
        return {'provider_attempt_records':copy.deepcopy(audit),'agent_attempts':len(audit)}
    monkeypatch.setattr('factorlab.discovery.calibration_coverage._monthly_evidence',monthly)
    monkeypatch.setattr('factorlab.discovery.neural_forward_ticket.open_existing_authority',
                        lambda directory:SimpleNamespace(get=lambda key:authority_records[key]))
    class Case:
        def save(self,month,audit,*,status='complete',charged=None):
            state={'status':status,'agent_audit':copy.deepcopy(audit),
                   'budget':{'counts':{'gpt_attempt':len(audit) if charged is None else charged}}}
            states[month]=state; store.save(month,state,'fixture_original_audit')
        def add(self,month,number,*,text='original prompt',resolved='fixture-model',
                audit_status='validated',raw_status='completed',orphan=False):
            jobpath=paths['runtime']/'autonomous/jobs'/month/'job.json'
            jobpath.parent.mkdir(parents=True,exist_ok=True)
            job={'month':month,'original_fixture':True}
            jobpath.write_text(json.dumps(job),'utf-8')
            from factorlab.discovery.full_trial_controller import VERSION as CONTROLLER_VERSION
            context={'version':CONTROLLER_VERSION,'trial':trial,
                'provider_policy':policy,'job_hash':fingerprint(job),'job_path':str(jobpath),
                'context_policy':'ephemeral_no_resume_no_user_rules_no_host_memory',
                'provider_authenticity_certified':False}
            parent=paths['provider']/month; parent.mkdir(exist_ok=True)
            (parent/'context.json').write_text(json.dumps(context),'utf-8')
            packet=PromptPacket(text,'{}',PromptIdentity('t','r','p','f'))
            audit={**packet.telemetry(),'status':audit_status,'attempt':number}
            path=parent/'attempts'/f'{packet.request_hash[:16]}-{number:032x}'
            path.mkdir(parents=True)
            (path/'stdin.txt').write_text(packet.text,'utf-8')
            (path/'response-schema.json').write_text('{}','utf-8')
            (path/'stderr.log').write_text('','utf-8')
            event={'type':'turn.started'}
            if resolved is not None: event['model']=resolved
            (path/'events.jsonl').write_text(json.dumps(event)+'\n'+json.dumps({'type':'turn.completed'}),'utf-8')
            if not orphan:
                (path/'attempt.json').write_text(json.dumps({**packet.telemetry(),
                    'status':raw_status,'model_resolved':resolved}),'utf-8')
            lease=FrozenWorkLease.issue(300); rid=month+':'+str(number)
            authority_records[rid]={'lease_hash':lease.digest,'lease':lease.to_dict(),
                'binding':{'work':{'stage':'family'}}}
            original={'version':mechanism.VERSION,'trial':trial,'physical_start':_observe(lease),
                'context_digest':fingerprint(context),'job_digest':fingerprint(job),
                'reservation_id':rid,'lease_hash':lease.digest,
                'command':CodexBatchBackend(path.parent,executable=executable)._command(path,path/'response-schema.json'),
                'executable_sha256':file_hash(executable),'stdin_sha256':file_hash(path/'stdin.txt'),
                'schema_sha256':file_hash(path/'response-schema.json')}
            (path/'invocation-original.json').write_text(json.dumps(original),'utf-8')
            (path/'invocation-output.json').write_text(json.dumps({'original_digest':fingerprint(original),
                'files':{name:file_hash(path/name) for name in ('events.jsonl','stderr.log')},
                'physical_finish':_observe(lease)}),'utf-8')
            return audit,path
        def read(self): return module._search(permit)
        def preserve(self,path): path.rename(root/('retained-'+path.name))
    value=Case(); value.store=store; value.paths=paths; value.states=states
    try: yield value
    finally: store.close()


def test_known_positive_and_duplicate_request_multiset(case):
    one,_=case.add('2026-10',1); two,_=case.add('2026-10',2)
    assert one['request_hash']==two['request_hash']
    case.save('2026-10',[one,two])
    value=case.read()
    assert value['provider_attempt_coverage']['complete']
    assert value['provider_attempt_coverage']['expected_attempts']==2
    assert value['provider_mechanism_verified']


def test_preserved_unknown_directory_cannot_shrink_denominator(case):
    one,_=case.add('2026-10',1); two,path=case.add('2026-10',2,resolved=None,audit_status='failed')
    case.save('2026-10',[one,two])
    full=case.read()
    assert full['provider_attempt_coverage']['complete']
    assert not full['provider_mechanism_verified']
    case.preserve(path)
    reduced=case.read()
    assert not reduced['provider_mechanism_verified']
    assert not reduced['all_resolved_models_known']
    assert not reduced['provider_attempt_coverage']['complete']
    assert reduced['provider_attempt_coverage']['charged_attempts']==2
    assert reduced['provider_attempt_coverage']['missing'][0]['count']==1


@pytest.mark.parametrize('fault',['wrong_request','wrong_metadata','extra_raw','missing_audit','charged_gap','orphan','started'])
def test_equal_counts_and_other_original_coverage_failures(case,fault):
    audit,path=case.add('2026-10',1,orphan=fault=='orphan',audit_status='started' if fault=='started' else 'validated')
    rows=[audit]; charged=1
    if fault=='wrong_request': rows[0]['request_hash']=fingerprint('different original request')
    if fault=='wrong_metadata': rows[0]['prefix_hash']=fingerprint('different prefix')
    if fault=='extra_raw': case.add('2026-10',2)
    if fault=='missing_audit': rows=[]
    if fault=='charged_gap': charged=2
    case.save('2026-10',rows,charged=charged)
    value=case.read()
    assert not value['provider_mechanism_verified']
    if fault!='started': assert not value['provider_attempt_coverage']['complete']


def test_raw_tag_cannot_relabel_original_stdin_even_with_matching_audit(case):
    audit,path=case.add('2026-10',1)
    audit['request_hash']=fingerprint('relabeled')
    raw=json.loads((path/'attempt.json').read_text('utf-8')); raw['request_hash']=audit['request_hash']
    (path/'attempt.json').write_text(json.dumps(raw),'utf-8')
    case.save('2026-10',[audit])
    value=case.read()
    assert not value['provider_attempt_coverage']['complete']
    assert not value['provider_mechanism_verified']


def test_cross_month_same_count_cannot_exchange_original_request(case):
    first,_=case.add('2026-10',1,text='October request')
    second,_=case.add('2026-11',2,text='November request')
    case.save('2026-10',[second]); case.save('2026-11',[first])
    value=case.read()
    assert not value['provider_attempt_coverage']['complete']
    assert len(value['provider_attempt_coverage']['missing'])==2
    assert not value['provider_mechanism_verified']


def test_incomplete_original_audit_and_pre_backend_no_raw_remain_counted(case):
    first,_=case.add('2026-10',1)
    second={**first,'request_hash':fingerprint('before backend'),'status':'started'}
    case.save('2026-10',[first,second],status='running')
    value=case.read()
    assert value['months'][0]['state']=='incomplete'
    assert value['months'][0]['agent_attempts']==2
    assert value['months'][0]['charged_agent_attempts']==2
    assert not value['provider_attempt_coverage']['complete']
    assert not value['provider_mechanism_verified']


def test_unfinished_month_journal_tamper_is_rejected(case):
    first,_=case.add('2026-10',1)
    case.save('2026-10',[first],status='running')
    altered={**case.states['2026-10'],'agent_audit':[]}
    with case.store.db:
        case.store.db.execute('UPDATE months SET value=?,hash=? WHERE month=?',
            (json.dumps(altered),fingerprint(altered),'2026-10'))
    with pytest.raises(ValueError,match='checkpoint'):
        case.read()


def test_original_complete_month_readback_cannot_supply_another_audit(case,monkeypatch):
    audit,_=case.add('2026-10',1)
    case.save('2026-10',[audit])
    monkeypatch.setattr('factorlab.discovery.calibration_coverage._monthly_evidence',
        lambda *args:{'provider_attempt_records':[],'agent_attempts':0})
    with pytest.raises(ValueError,match='journal audit'):
        case.read()


def test_job_month_and_context_directory_must_agree(case):
    audit,_=case.add('2026-10',1); case.save('2026-10',[audit])
    path=case.paths['runtime']/'autonomous/jobs/2026-10/job.json'
    job=json.loads(path.read_text('utf-8')); job['month']='2026-11'
    path.write_text(json.dumps(job),'utf-8')
    context_path=case.paths['provider']/'2026-10/context.json'
    context=json.loads(context_path.read_text('utf-8')); context['job_hash']=fingerprint(job)
    context_path.write_text(json.dumps(context),'utf-8')
    with pytest.raises(ValueError,match='job month'):
        case.read()


def test_missing_original_search_journal_cannot_erase_the_audit(case):
    from factorlab.discovery.calibration_trial_context import TrialIsolationError
    audit,_=case.add('2026-10',1); case.save('2026-10',[audit])
    case.store.close()
    path=case.paths['research']/'state.sqlite'
    path.rename(path.with_suffix('.retained'))
    with pytest.raises(TrialIsolationError,match='without its original journal'):
        case.read()
