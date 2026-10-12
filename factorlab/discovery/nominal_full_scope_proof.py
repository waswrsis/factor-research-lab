"""Prospective original full-policy native execution; never nominal truth.

One source-fixed unit per original coordinate. Proof <=300 is INSIDE the
original <=7200 service; no fit/GPT/month wallet is created by this module.
"""
from __future__ import annotations
from datetime import datetime, timedelta
from fractions import Fraction
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace
import importlib
import json
import os
import sys
import threading
import psutil
from factorlab.protocol import fingerprint, project_path

VERSION='nominal-full-scope-native-proof-v2-registered-risk-readback'
RELATIVE='factorlab/discovery/nominal_full_scope_proof.py'
WORKER='factorlab.discovery.nominal_full_scope_worker'
CLAIMS=('library_skill','factor1','factor2','factor3','algorithm')
_KEY=object()
_PROOF=ContextVar('full_scope_original_proof_task',default=None)

def _plain(v):return json.loads(json.dumps(v,sort_keys=True,allow_nan=False))
def _api():
    from . import nominal_systemd_unit as units
    from . import native_verification_terminal as kernel
    return units,kernel
def _path(value):
    from .calibration_trial_context import _no_redirect
    root=project_path('.');path=Path(value)
    if not path.is_absolute():path=root/path
    _no_redirect(path,root)
    return project_path(path)
def _write(path,value):
    from .v7_storage import immutable_record
    immutable_record(_path(path),{'record':value,'hash':fingerprint(value)})
def _read(path):
    from .nominal_cold_conformance import _read as read
    return read(_path(path))
def _pair(roots,name):
    a,b=(_read(p/name)for p in roots)
    if a!=b:raise ValueError('Original full-scope mirror differs: '+name)
    return a
def _write_pair(roots,name,value):
    for root in roots:_write(root/name,value)
    if _pair(roots,name)!=value:raise ValueError('Original publication changed')
def _installed():
    from .monthly_work_authority import source_files
    from .evidence import file_hash
    files=source_files()
    if Path(__file__).resolve()!=project_path(RELATIVE).resolve() or files.get(RELATIVE)!=file_hash(__file__):
        raise ValueError('New canonical full-scope proof source required')
    for name in ('nominal_full_scope_worker','nominal_algorithm_producer','nominal_algorithm_execution','nominal_algorithm_policy',
                 'nominal_algorithm_total_risk','nominal_total_loss','full_trial_controller',
                 'full_calibration_registry','nominal_systemd_unit','native_verification_terminal',
                 'nominal_algorithm_risk_reader'):
        module=importlib.import_module('factorlab.discovery.'+name)
        relative='factorlab/discovery/'+name+'.py'
        if Path(module.__file__).resolve()!=project_path(relative).resolve() or files.get(relative)!=file_hash(module.__file__):
            raise ValueError('Loaded original full-policy dependency differs: '+name)
    return files

class FullScopeProofPlan:
    """All original N and whole-algorithm oracle coordinates, frozen once."""
    @classmethod
    def freeze(cls,design):
        from .nominal_algorithm_total_risk import AlgorithmTotalRiskRegistry,_fresh_parent
        from .nominal_algorithm_policy import NominalAlgorithmPolicy
        from filelock import FileLock
        if type(design)is not AlgorithmTotalRiskRegistry:raise TypeError('Exact new whole-policy design required')
        files=_installed();design.verify();policy=NominalAlgorithmPolicy.inspect(design)
        parent=design.parent
        with FileLock(str(parent.directory/'.full-scope-proof-creation.lock'),timeout=0):
            _fresh_parent(parent)
            directory=parent.directory/'nominal_full_scope_proof'
            if (parent.directory/'nominal_full_scope_proof.json').exists():raise ValueError('Original proof anchor already exists')
            directory.mkdir(exist_ok=False)
            value=cls._expected(design,policy,files)
            _write_pair((directory,parent.directory),'nominal_full_scope_proof.json',value)
        return cls.inspect(parent)
    @staticmethod
    def _expected(design,policy,files):
        from .nominal_full_five_consumer import descriptor
        seconds=design.identity['full_schedule']['service']['wall_seconds']
        if type(seconds)not in(int,float) or not 300<=seconds<=7200:
            raise ValueError('Original service must retain its own <=7200 ceiling and room for a proof task')
        cells=[{'name':c.name,'N':c.repetitions,'scenario_digest':c.scenario_digest}for c in design.parent.plan.acceptance.cells]
        return {'version':VERSION,'directory':str(design.parent.directory/'nominal_full_scope_proof'),
            'parent_directory':str(design.parent.directory),'parent_digest':design.parent.digest,
            'full_five_consumer':descriptor(design),
            'design_digest':design.digest,'policy_digest':policy.digest,'source_files':files,
            'cells':cells,'original_N':sum(c['N']for c in cells),'oracle_paths':design.parent.plan.oracle_paths,
            'roles':['validation','oracle'],'claims':list(CLAIMS),'fixed_estimand_replicas_included':False,
            'original_full_plan_digest':design.parent.plan.digest,'original_service_seconds':seconds,
            'proof_seconds':300,'proof_time':'inside_min_original_controller_and_OS_service_deadlines',
            'algorithm_risk_readback': {
                'version': 'nominal-algorithm-registered-risk-readback-v1',
                'allocation': 'validation_original_single_proof300_remaining',
                'oracle_paths': design.parent.plan.oracle_paths,
                'extra_fit_GPT_or_task_credit': False, 'provider_IID_granted': False},
            'fit_GPT_wallet':'existing_original_only_no_new_credit','per_coordinate_attempts':1,
            'failed_coordinate':{'retain_N':True,'possible_error':1,'power':0},
            'worker':WORKER,'native_scope':'actual_full_policy_trajectory_not_nominal_risk',
            'strict_monthly_cold_certified':False,
            'formal_acceptance':False}
    @classmethod
    def inspect(cls,parent):
        from .full_calibration_registry import FullTrialRegistry
        if type(parent)is not FullTrialRegistry:raise TypeError('Exact original full registry required')
        obj=cls.__new__(cls);obj.parent=parent
        obj.directory=parent.directory/'nominal_full_scope_proof'
        obj.identity=_read(obj.directory/'nominal_full_scope_proof.json');obj.digest=fingerprint(obj.identity)
        obj.verify();return obj
    def verify(self):
        from .nominal_algorithm_total_risk import AlgorithmTotalRiskRegistry
        from .nominal_algorithm_policy import NominalAlgorithmPolicy
        files=_installed();self.parent.verify()
        design=AlgorithmTotalRiskRegistry.inspect(self.parent);policy=NominalAlgorithmPolicy.inspect(design)
        expected=self._expected(design,policy,files)
        if (self.identity!=expected or self.digest!=fingerprint(expected)
            or self.directory!=self.parent.directory/'nominal_full_scope_proof'
            or _pair((self.directory,self.parent.directory),'nominal_full_scope_proof.json')!=expected):
            raise ValueError('Original prospective full-N proof plan changed')
        return expected
    def coordinate(self,cell,index,oracle_index=None):
        self.verify()
        return self.parent.identity_for(cell,index,oracle_index=oracle_index)
    def roots(self,coordinate):
        expected=self.parent.identity_for(coordinate['cell'],coordinate['index'],oracle_index=coordinate['oracle_index'])
        if coordinate!=expected or coordinate.get('estimand_slot')is not None:raise ValueError('Original full-policy coordinate required')
        root=self.parent.paths(coordinate['cell'],coordinate['index'],oracle_index=coordinate['oracle_index'])['root']
        return (_path(root/'native-full-scope'),_path(self.directory/'attempts'/coordinate['trial_id']))

def unit_name(plan,coordinate):return 'factorresearch-nf-'+fingerprint([plan.digest,coordinate])+'.service'
def command(plan,coordinate):
    plan.roots(coordinate)
    oracle='validation' if coordinate['oracle_index']is None else str(coordinate['oracle_index'])
    return [sys.executable,'-B','-m',WORKER,'--registry',str(plan.parent.directory),
        '--cell',coordinate['cell'],'--trial-index',str(coordinate['index']),'--oracle-index',oracle]

def _unit_contract_view(raw,seconds):
    """Reuse the common fixed7200 validator without raising an earlier limit.

    The ACTUAL property must equal the original scope limit first. Only a local
    copy of that checked property is adapted for the existing validator; every
    retained observation and all deadline arithmetic keep the actual raw value.
    """
    if (type(seconds)not in(int,float)or not 300<=seconds<=7200
        or type(raw.get('RuntimeMaxUSec'))is not int or raw['RuntimeMaxUSec']!=seconds*10**6):
        raise ValueError('Actual OS service ceiling differs from the original frozen limit')
    return {**raw,'RuntimeMaxUSec':7200*10**6}

def _expired(identity,observed):
    seconds=identity['service_seconds']
    if (observed['boot_id']!=identity['boot_id']
        or Fraction(observed['monotonic'])>=Fraction(identity['unit_start_monotonic_us'],10**6)+seconds
        or Fraction(observed['wall'])>=Fraction(identity['unit_start_realtime_us'],10**6)+seconds):
        raise TimeoutError('Original external unit clock expired or changed boot')

def _live(identity,clock=None,proof=None):
    units,kernel=_api();_expired(identity,units._observe())
    if clock is not None:clock.assert_parent_alive()
    if proof is not None:proof.assert_parent_alive()
    raw=units._unit(identity['unit'],None if clock is None else SimpleNamespace(clock=clock))
    units._same(identity,_unit_contract_view(raw,identity['service_seconds']))
    if (raw['ActiveState']!='active' or raw['SubState']!='running' or raw['MainPID']!=os.getpid()
        or raw['ExecMainPID']!=os.getpid() or os.environ.get('INVOCATION_ID')!=identity['invocation_id']
        or kernel._proc(os.getpid())['start_ticks']!=identity['parent_start_ticks']
        or kernel._group(os.getpid())!=Path(identity['owned_group']['path'])):
        raise ValueError('Original full-scope process/unit identity lost')
    _expired(identity,units._observe())
    if clock is not None:clock.assert_parent_alive()
    if proof is not None:proof.assert_parent_alive()
    return raw

def _bootstrap(plan,coordinate):
    units,kernel=_api()
    if sys.platform!='linux' or threading.current_thread()is not threading.main_thread():
        raise OSError('Original Linux main-thread unit required')
    raw=units._unit(unit_name(plan,coordinate))
    units._configuration(_unit_contract_view(raw,plan.identity['original_service_seconds']),command(plan,coordinate))
    if any(k in os.environ for k in units.UNSET_ENVIRONMENT):raise ValueError('Inherited source injection refused')
    if raw['MainPID']!=os.getpid() or raw['ExecMainPID']!=os.getpid():raise ValueError('Fixed current main PID required')
    owned=units._group(raw);observed=units._observe();proc=kernel._proc(os.getpid())
    from .evidence import file_hash
    value={'version':VERSION,'plan_digest':plan.digest,'coordinate':coordinate,
        'source_files':plan.identity['source_files'],'service_seconds':plan.identity['original_service_seconds'],
        'unit':raw['Id'],'object_path':raw['object_path'],'invocation_id':units._invocation(raw),
        'exec_main_pid':os.getpid(),'parent_start_ticks':proc['start_ticks'],
        'parent_created':psutil.Process(os.getpid()).create_time(),'boot_id':observed['boot_id'],
        'unit_start_monotonic_us':raw['ExecMainStartTimestampMonotonic'],
        'unit_start_realtime_us':raw['ExecMainStartTimestamp'],'command':command(plan,coordinate),
        'owned_group':owned,'shared_group':units._shared(Path(owned['path'])),
        'executable_sha256':file_hash(sys.executable),'busctl_sha256':file_hash(units.BUSCTL)}
    _live(value);return value

def _fail(roots,error):
    value={'version':VERSION,'state':'failed_or_unknown_consumed','retain_N':True,
        'possible_error':1,'power':0,'retry_allowed':False,'reason':type(error).__name__+':'+str(error)[:900]}
    for p in roots:
        try:
            if p.is_dir()and not(p/'failure.json').exists():_write(p/'failure.json',value)
        except BaseException as secondary:error.add_note('Failure publication unavailable: '+type(secondary).__name__)
def _no_failure(roots):
    if any((p/'failure.json').exists()for p in roots):raise ValueError('Original failed coordinate remains consumed')

def _reap_natural():
    """No signals, no waits for live children, and no retry/cleanup success."""
    records=[]
    for _ in range(64):
        try:pid,status=os.waitpid(-1,os.WNOHANG)
        except ChildProcessError:return records
        if pid==0:raise ValueError('Original subprocess still alive; cannot publish completion')
        if not os.WIFEXITED(status)or os.WEXITSTATUS(status)!=0:raise ValueError('Unreaped child did not exit naturally zero')
        records.append({'pid':pid,'status':status,'natural_zero':True})
    raise ValueError('Original bounded child reap exceeded task limit')

def _producer_publication(plan,coordinate):
    """Small original producer records, with fixed paths and no re-computation."""
    from .full_calibration_registry import QuarantinedTrialPermit
    from .nominal_algorithm_producer import _roots,VERSION as producer_version
    from .calibration_oracle_evidence import _read as plain_read
    permit=QuarantinedTrialPermit(plan.parent,coordinate);roots=_roots(permit)
    identity=_pair(roots,'identity.json')
    intent=_pair(tuple(p/'proof'for p in roots),'intent.json')
    proof=_pair(tuple(p/'proof'for p in roots),'result.json')
    result=_pair(tuple(p/'attempt'for p in roots),'result.json')
    terminal=plain_read(_path(permit.paths['root']/'controller/steps/-002-terminal.json'))
    terminal_mirror=plain_read(_path(plan.parent.directory/'controller_steps'/coordinate['trial_id']/'-002-terminal.json'))
    if (any((p/'attempt/failure.json').exists()for p in roots)or identity['version']!=producer_version
        or identity['trial']!=coordinate or identity['source_files']!=plan.identity['source_files']
        or intent!={'identity_hash':fingerprint(identity),'lease':result['proof_lease'],'original_clock':identity['original_service_clock']}
        or result['identity_hash']!=fingerprint(identity)or result['trial']!=coordinate or result['proof']!=proof
        or result['controller_terminal_hash']!=proof['controller_terminal_hash']
        or result['publication_state']!='provisional_requires_original_native_completion'
        or result['provider_IID_certified']is not False or result['formal_acceptance']is not False
        or terminal!=terminal_mirror or fingerprint(terminal)!=result['controller_terminal_hash']):
        raise ValueError('Original paid producer proof/publication/terminal changed')
    return {'identity':identity,'intent':intent,'proof':proof,'result':result}

class _ProofTask:
    def __init__(self,*a,**k):raise TypeError('Only the fixed full-scope worker may acquire proof work')
    def check(self):
        if (self.key is not _KEY or _PROOF.get()is not self
            or self.owner!=(os.getpid(),threading.get_ident())or self.closed):
            raise ValueError('Original active source-bound proof task required')
        _no_failure(self.roots);_live(self.identity,self.clock,self.lease)

def _bind_producer(plan,coordinate):
    """Open original pre-enrolled stores; bind clock INSIDE actual unit."""
    from .full_calibration_registry import QuarantinedTrialPermit
    from .nominal_algorithm_total_risk import AlgorithmTotalRiskRegistry
    from .nominal_algorithm_policy import NominalAlgorithmPolicy
    from .nominal_algorithm_producer import AlgorithmNominalProducer
    from .full_mechanism_contract import verify_mechanism_contract
    from .v7_forward import ForwardRuntime,ForwardConfig
    from .v7_lifecycle import LifecyclePlan
    from .claim_protocol import FrozenClaimEndpoint
    from .neural_forward_ticket import open_existing_authority
    from .calibration_oracle_evidence import _read as plain_read
    permit=QuarantinedTrialPermit(plan.parent,coordinate);permit.verify()
    for p in (permit.paths['runtime']/'forward.json',permit.paths['runtime']/'lifecycle.sqlite',
              permit.paths['feed']/'identity.json',permit.paths['feed']/'receipts.sqlite'):
        if not p.is_file():raise ValueError('Original pre-enrolled store missing; never repair or initialize')
    if (permit.paths['root']/'controller').exists() or (permit.paths['root']/'nominal_algorithm_execution').exists():
        raise ValueError('Old or partially bound execution cannot receive a new native attempt')
    authority=open_existing_authority(permit.paths['authority'])
    if authority.verify()['reservations']:raise ValueError('Previously used original wallet cannot restart')
    design=AlgorithmTotalRiskRegistry.inspect(plan.parent);policy=NominalAlgorithmPolicy.inspect(design)
    generators=verify_mechanism_contract(plan.parent.plan.mechanism_contract,plan.parent.plan.acceptance)
    generator=generators[coordinate['scenario_digest']]
    value=plain_read(permit.paths['runtime']/'forward.json')['identity']
    config={**value['config'],'symbols':tuple(value['config']['symbols'])}
    lifecycle={**value['lifecycle'],'symbols':tuple(value['lifecycle']['symbols'])}
    runtime=ForwardRuntime(permit.paths['runtime'],permit.paths['feed'],config=ForwardConfig(**config),
                           lifecycle_plan=LifecyclePlan(**lifecycle))
    try:
        return AlgorithmNominalProducer.bind(design,permit,runtime=runtime,authority=authority,
            generator=generator,endpoints=tuple(FrozenClaimEndpoint.from_dict(v)for v in policy.identity['endpoints']))
    finally:runtime.close()

def run_coordinate(directory,cell,index,oracle_index=None):
    """The only fixed worker issuer: original run, metered proof, quiet return."""
    from .full_calibration_registry import FullTrialRegistry
    from .full_trial_controller import ControllerStepLease
    from .work_lease import FrozenWorkLease,_observe
    if _PROOF.get()is not None:raise ValueError('Nested full-scope proof refused')
    parent=FullTrialRegistry.inspect(directory);plan=FullScopeProofPlan.inspect(parent)
    coordinate=plan.coordinate(cell,index,oracle_index);roots=plan.roots(coordinate)
    bootstrap=_bootstrap(plan,coordinate)
    if any(p.exists()for p in roots):raise ValueError('Original full-policy native coordinate already consumed')
    task=None;token=None
    try:
        for p in roots:p.mkdir(parents=True,exist_ok=False)
        _write_pair(roots,'intent.json',{'version':VERSION,'plan_digest':plan.digest,
            'coordinate':coordinate,'retry_allowed':False})
        _live(bootstrap)
        producer=_bind_producer(plan,coordinate)
        clock=ControllerStepLease.from_dict(producer.identity['original_service_clock'],producer.transport.controller.schedule)
        if (clock.service_seconds!=plan.identity['original_service_seconds']
            or clock.issuer['parent_pid']!=bootstrap['exec_main_pid']
            or clock.issuer['parent_created']!=bootstrap['parent_created']or clock.issuer['boot_id']!=bootstrap['boot_id']
            or Fraction(clock.issuer['monotonic_started'])<Fraction(bootstrap['unit_start_monotonic_us'],10**6)):
            raise ValueError('Original transport clock differs from actual native parent')
        identity={**bootstrap,'clock':clock.to_dict(),'producer_hash':fingerprint(producer.identity)}
        _write_pair(roots,'native-unit.json',identity);_live(identity,clock)
        result=producer.run_once()  # Exact fixed new mode; never bypass its private owner context.
        _live(identity,clock)
        if (result['identity_hash']!=fingerprint(producer.identity)
            or result['publication_state']!='provisional_requires_original_native_completion' or result['formal_acceptance']is not False):
            raise ValueError('Original actual transport did not return its provisional complete trace')
        # The producer already issued the ONLY proof300 before heavy reads.
        # Loading it does not renew its clock; wrapper closure uses its remainder.
        publication=_producer_publication(plan,coordinate)
        if publication['identity']!=producer.identity or publication['result']!=result:
            raise ValueError('Returned result is not the original dual producer publication')
        lease=FrozenWorkLease.from_dict(publication['intent']['lease'])
        if (lease.lease_seconds!=300 or any(getattr(lease,k)!=clock.issuer[k]for k in ('parent_pid','parent_created','boot_id'))):
            raise ValueError('Original single proof task belongs to another issuer')
        task=object.__new__(_ProofTask);task.key=_KEY;task.owner=(os.getpid(),threading.get_ident())
        task.producer=producer;task.roots=roots;task.identity=identity;task.clock=clock
        task.lease=lease;task.result=result;task.closed=False;token=_PROOF.set(task)
        task.check();_write_pair(roots,'proof-intent.json',{'version':VERSION,'native_binding_hash':fingerprint(identity),'lease':lease.to_dict()})
        risk_readback=None
        if coordinate['oracle_index'] is None:
            from .nominal_algorithm_total_risk import AlgorithmTotalRiskRegistry
            risk_readback=AlgorithmTotalRiskRegistry.inspect(parent).certified_risk_interval(cell,index)
            task.check()
        full_five_readback=None
        if coordinate['oracle_index'] is None:
            from .nominal_full_five_consumer import consume_original_validation
            full_five_readback=consume_original_validation(AlgorithmTotalRiskRegistry.inspect(parent),cell,index)
            task.check()
        from .nominal_algorithm_total_risk import AlgorithmTotalRiskRegistry
        from .nominal_full_five_consumer import final_original_hook
        final_full_N=final_original_hook(AlgorithmTotalRiskRegistry.inspect(parent)) if coordinate['oracle_index'] is not None else None
        task.check()
        actual=publication['proof']['actual_execution']
        summary={**actual,'producer_hash':fingerprint(producer.identity),
            'producer_publication_hash':fingerprint(publication),'actual_complete_trajectory':True,
            'nominal_truth_certified':False,'strict_monthly_cold_certified':False,
            'provider_IID_certified':False,'formal_acceptance':False,
            'algorithm_risk_readback':risk_readback,'full_five_readback':full_five_readback,'final_full_N':final_full_N}
        task.check()
        proof={'version':VERSION,'native_binding_hash':fingerprint(identity),'lease':lease.to_dict(),
            'summary':summary,'observed':_observe(lease),'provisional_only':True}
        _write_pair(roots,'proof.json',proof);task.check()
        reaped={'version':VERSION,'native_binding_hash':fingerprint(identity),
            'events':_reap_natural(),'cleanup_signals':False}
        _write_pair(roots,'reaped.json',reaped);task.check()
        units,kernel=_api();tree=kernel._subtree(Path(identity['owned_group']['path']))
        units._quiescent_tree(identity,tree)
        quiet={'version':VERSION,'native_binding_hash':fingerprint(identity),'proof_hash':fingerprint(proof),
            'reaped_hash':fingerprint(reaped),'tree':tree,'observed':_observe(lease),'cleanup_signals':False}
        _write_pair(roots,'quiescence.json',quiet)
        task.check();units._quiescent_tree(identity,kernel._subtree(Path(identity['owned_group']['path'])))
        plan.verify()
        if _producer_publication(plan,coordinate)!=publication:raise ValueError('Original producer changed across native closure')
        task.check();task.closed=True
        return {'version':VERSION,'coordinate':coordinate,'proof_hash':fingerprint(proof),
            'provisional_only':True,'native_completed':False,'nominal_truth_certified':False}
    except BaseException as error:
        _fail(roots,error);raise
    finally:
        if task is not None:task.closed=True
        if token is not None:_PROOF.reset(token)

def _exit_deadline(identity,raw):
    seconds=identity['service_seconds'];issuer=identity['clock']['issuer']
    start_m=Fraction(raw['ExecMainStartTimestampMonotonic'],10**6);start_w=Fraction(raw['ExecMainStartTimestamp'],10**6)
    end_m=Fraction(raw['ExecMainExitTimestampMonotonic'],10**6);end_w=Fraction(raw['ExecMainExitTimestamp'],10**6)
    began=datetime.fromisoformat(issuer['wall_started_at'])
    dm=min(start_m+seconds,Fraction(issuer['monotonic_started'])+seconds,Fraction(issuer['monotonic_started']+seconds))
    dw=min(start_w+seconds,Fraction(began.timestamp())+seconds,Fraction((began+timedelta(seconds=seconds)).timestamp()))
    if (identity['clock']['service_seconds']!=seconds or issuer['boot_id']!=identity['boot_id']
        or not 0<start_m<=Fraction(issuer['monotonic_started'])<end_m<dm
        or not 0<start_w<=Fraction(began.timestamp())<end_w<dw):
        raise ValueError('Original unit/controller absolute deadline failed')

class FullScopeNativeObservation:
    def __init__(self,*a,**k):raise TypeError('Use original retained full-scope unit reader')
    def __reduce__(self):raise TypeError('Kernel observation cannot serialize as authority')
    @property
    def identity(self):return _plain(self._identity)
    def assert_current(self):
        if self._key is not _KEY:raise TypeError('Original typed native observation required')
        fresh=read_retained(self._plan,self._coordinate)
        if fresh._identity!=self._identity or fresh._raw!=self._raw:raise ValueError('Original retained full-scope observation changed')
        return self

def read_retained(plan,coordinate):
    """Small retained OS/immutable proof read. Never re-runs semantic fits/risks.

    This witnesses the past metered trajectory proof. A later conformance caller
    must match its original execution records to these digests under its own
    registered cost; no actual record mutation is repaired here.
    """
    if type(plan)is not FullScopeProofPlan:raise TypeError('Exact new full-scope plan required')
    plan.verify();roots=plan.roots(coordinate);_no_failure(roots)
    identity=_pair(roots,'native-unit.json');proof=_pair(roots,'proof.json');quiet=_pair(roots,'quiescence.json')
    intent=_pair(roots,'intent.json');task=_pair(roots,'proof-intent.json');reaped=_pair(roots,'reaped.json')
    if (intent!={'version':VERSION,'plan_digest':plan.digest,'coordinate':coordinate,'retry_allowed':False}
        or identity['coordinate']!=coordinate or identity['plan_digest']!=plan.digest
        or identity['source_files']!=plan.identity['source_files'] or identity['command']!=command(plan,coordinate)
        or identity['unit']!=unit_name(plan,coordinate)or identity['service_seconds']!=plan.identity['original_service_seconds']
        or task!={'version':VERSION,'native_binding_hash':fingerprint(identity),'lease':proof['lease']}
        or proof['native_binding_hash']!=fingerprint(identity)or proof['provisional_only']is not True
        or quiet['proof_hash']!=fingerprint(proof)or quiet['reaped_hash']!=fingerprint(reaped)
        or quiet['native_binding_hash']!=fingerprint(identity)or quiet['cleanup_signals']is not False
        or reaped['cleanup_signals']is not False or reaped['native_binding_hash']!=fingerprint(identity)):
        raise ValueError('Original full-scope proof closure differs')
    from .work_lease import FrozenWorkLease,_evaluate
    from .full_trial_controller import ControllerStepLease
    lease=FrozenWorkLease.from_dict(proof['lease'])
    clock=ControllerStepLease.from_dict(identity['clock'],SimpleNamespace(service={'wall_seconds':identity['service_seconds']}))
    if (lease.lease_seconds!=300 or any(getattr(lease,k)!=clock.issuer[k]for k in ('parent_pid','parent_created','boot_id'))
        or clock.issuer['parent_pid']!=identity['exec_main_pid']or clock.issuer['parent_created']!=identity['parent_created']
        or not _evaluate(lease,proof['observed']).allowed or not clock.evaluate(proof['observed']).allowed
        or not _evaluate(lease,quiet['observed'],proof['observed']).allowed
        or not clock.evaluate(quiet['observed'],proof['observed']).allowed):
        raise ValueError('Original metered proof crossed its task/service clock')
    summary=proof['summary']
    if (set(summary['claims'])!=set(CLAIMS)or any(v['state']!='complete'for v in summary['claims'].values())
        or summary['actual_complete_trajectory']is not True or summary['nominal_truth_certified']is not False
        or summary['strict_monthly_cold_certified']is not False
        or summary['provider_IID_certified']is not False or summary['formal_acceptance']is not False
        or summary['producer_hash']!=identity['producer_hash']
        or any(type(v)is not dict or set(v)!={'pid','status','natural_zero'}or type(v['pid'])is not int or v['pid']<=0
            or v['status']!=0 or v['natural_zero']is not True for v in reaped['events'])):
        raise ValueError('Original actual-only full-scope semantic proof differs')
    publication=_producer_publication(plan,coordinate)
    if (summary['producer_publication_hash']!=fingerprint(publication)
        or publication['identity']['original_service_clock']!=identity['clock']
        or publication['intent']['lease']!=proof['lease']
        or fingerprint(publication['identity'])!=identity['producer_hash']
        or any(summary.get(k)!=v for k,v in publication['proof']['actual_execution'].items())):
        raise ValueError('Native proof differs from the original single metered producer')
    risk=summary.get('algorithm_risk_readback')
    if coordinate['oracle_index'] is None:
        from .nominal_algorithm_total_risk import AlgorithmTotalRiskRegistry
        from .nominal_algorithm_risk_reader import validate_retained_readback
        validate_retained_readback(risk,AlgorithmTotalRiskRegistry.inspect(plan.parent),coordinate,
                                  binding_hash=fingerprint(identity),lease=proof['lease'])
    elif risk is not None:
        raise ValueError('Oracle native task must not borrow validation risk-readback allocation')
    full_five=summary.get('full_five_readback')
    if coordinate['oracle_index'] is None:
        from .nominal_full_five_consumer import validate_retained_consumer
        validate_retained_consumer(full_five,AlgorithmTotalRiskRegistry.inspect(plan.parent),coordinate,
                                    binding_hash=fingerprint(identity),lease=proof['lease'])
    elif full_five is not None:
        raise ValueError('Oracle task cannot borrow validation full-five allocation')
    from .nominal_algorithm_total_risk import AlgorithmTotalRiskRegistry
    from .nominal_full_five_consumer import validate_retained_full_N
    validate_retained_full_N(summary.get('final_full_N'),AlgorithmTotalRiskRegistry.inspect(plan.parent),coordinate,
                             binding_hash=fingerprint(identity),lease=proof['lease'])
    completed=publication['proof']['proof_completed_observation']
    if (not _evaluate(lease,completed).allowed or not clock.evaluate(completed).allowed
        or not _evaluate(lease,proof['observed'],completed).allowed):
        raise ValueError('Native closure preceded or outlived original metered proof')
    units,kernel=_api();observed=units._observe()
    from .evidence import file_hash
    if (observed['boot_id']!=identity['boot_id']or identity['executable_sha256']!=file_hash(sys.executable)
        or identity['busctl_sha256']!=file_hash(units.BUSCTL)):raise ValueError('Original boot/executable unavailable')
    raw=units._unit(identity['unit']);view=_unit_contract_view(raw,identity['service_seconds']);units._same_invocation(identity,view)
    if (raw['ActiveState']!='active'or raw['SubState']!='exited'or raw['MainPID']!=0
        or raw['Result']!='success'or raw['ExecMainCode']!=1 or raw['ExecMainStatus']!=0):
        raise ValueError('No original natural-zero full-policy exit')
    _exit_deadline(identity,raw);units._historical_group(identity,view);units._quiescent_tree(identity,quiet['tree'])
    if (Fraction(quiet['observed']['monotonic'])>=Fraction(raw['ExecMainExitTimestampMonotonic'],10**6)
        or Fraction(quiet['observed']['wall'])>=Fraction(raw['ExecMainExitTimestamp'],10**6)
        or Fraction(raw['ExecMainExitTimestampMonotonic'],10**6)>Fraction(observed['monotonic'])
        or Fraction(raw['ExecMainExitTimestamp'],10**6)>Fraction(observed['wall'])):
        raise ValueError('Original quiescence/exit clock order differs')
    if (Fraction(raw['ExecMainExitTimestampMonotonic'],10**6)>=Fraction(lease.monotonic_deadline)
        or Fraction(raw['ExecMainExitTimestamp'],10**6)>=Fraction(datetime.fromisoformat(lease.wall_deadline).timestamp())):
        raise ValueError('Original proof lease expired before natural completion')
    hashes={name:fingerprint(_pair(roots,name))for name in ('intent.json','native-unit.json','proof-intent.json','proof.json','reaped.json','quiescence.json')}
    if hashes!={name:fingerprint(v)for name,v in zip(hashes,(intent,identity,task,proof,reaped,quiet))}:
        raise ValueError('Original proof records changed during read')
    plan.verify();_no_failure(roots)
    again=units._unit(identity['unit']);view=_unit_contract_view(again,identity['service_seconds'])
    units._same_invocation(identity,view);units._historical_group(identity,view)
    if again!=raw:raise ValueError('Original retained invocation replaced during read')
    if hashes!={name:fingerprint(_pair(roots,name))for name in hashes}:raise ValueError('Original proof bytes changed at final read')
    if _producer_publication(plan,coordinate)!=publication:raise ValueError('Original producer publication changed at final read')
    _no_failure(roots)
    obj=object.__new__(FullScopeNativeObservation);obj._key=_KEY;obj._plan=plan;obj._coordinate=_plain(coordinate)
    obj._raw=_plain(raw);obj._identity={'version':VERSION,'plan_digest':plan.digest,'coordinate':coordinate,
        'native_binding_hash':fingerprint(identity),'proof_hash':fingerprint(proof),'actual_proof':proof['summary'],
        'native_completed':True,'nominal_truth_certified':False,'strict_monthly_cold_certified':False,'formal_acceptance':False}
    return obj
