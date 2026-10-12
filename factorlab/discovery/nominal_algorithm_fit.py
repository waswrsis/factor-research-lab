"""Trusted final numerical fit inability in the new whole-policy invocation.

No generic exception classification, model substitution, new allowance, solver
replay, or fixed-factor authority. Persistent records are provisional until the
original full-scope native service authenticates the completed producer.
"""
from __future__ import annotations
from contextvars import ContextVar
from pathlib import Path
import importlib
import json
import os
import sys
import threading
import weakref
from factorlab.protocol import fingerprint,project_path

VERSION='nominal-whole-policy-numerical-fit-v1'
RELATIVE='factorlab/discovery/nominal_algorithm_fit.py'
_KEY=object()
_KERNEL=ContextVar('original_whole_policy_numeric_kernel',default=None)
_EVENTS=weakref.WeakKeyDictionary()
_ERRORS=weakref.WeakSet()

def descriptor():
    return {'version':VERSION,'scope':'original_known_or_pool_final_escaped_exact_LinAlgError',
        'profile_recovery':'original_exception_and_profile_control_flow_unchanged',
        'cold_unavailability':'from_next_original_prefix_until_later_same_stream_paid_cold_model_visible',
        'old_model':'retained_in_original_ledger_but_not_served_as_replacement_for_numeric_failed_cold',
        'fit_accounting':'original_attempt_and_actual_counters_failed_consumed_no_refund',
        'nominal_loss':'full_original_support_B_only_for_authenticated_numeric_unavailability',
        'infrastructure_failure':'unknown_error_one_power_zero_full_N',
        'extra_fit_or_proof_allowance':0,'history':'original_fullscope_native_unit_not_JSON_or_fixedfactor',
        'formal_acceptance':False}

def _owner():return os.getpid(),threading.get_ident()
def _plain(v):return json.loads(json.dumps(v,sort_keys=True,allow_nan=False))
def _api():
    from . import nominal_algorithm_producer as producer
    return producer
def _producer(runtime=None):
    module=_api();p=module._ACTIVE.get()
    if p is None:
        if runtime is not None:module._active(runtime)
        return None
    if type(p)is not module.AlgorithmNominalProducer or p.owner!=_owner():raise ValueError('Exact original whole-policy owner required')
    p.check()
    if runtime is not None and module._active(runtime)is not p:raise ValueError('Original whole-policy runtime required')
    return p
def _installed(p):
    from .monthly_work_authority import source_files
    from .evidence import file_hash
    files=source_files()
    for name in ('nominal_algorithm_fit','nominal_algorithm_producer','nominal_algorithm_policy','nominal_fit_outcome',
        'known_linear_forward','pool_forward','convex_eval','v7_forward','reserved_work_budget','nominal_full_scope_proof'):
        m=importlib.import_module('factorlab.discovery.'+name);rel='factorlab/discovery/'+name+'.py'
        if Path(m.__file__).resolve()!=project_path(rel).resolve()or files.get(rel)!=file_hash(m.__file__):
            raise ValueError('Canonical whole-policy numeric dependency changed: '+name)
    if files!=p.identity['source_files']or p.identity.get('numeric_fit_policy')!=descriptor():
        raise ValueError('Original preregistered whole-policy numeric source required')
    return files
def _paths(p,job):return tuple(root/'numeric-fits'/job for root in p.roots)
def _read_pair(roots,name):
    from .nominal_algorithm_execution import _pair
    return _pair(roots,name)
def _write_pair(roots,name,value):
    from .nominal_algorithm_execution import _write_pair
    _write_pair(roots,name,value)
    if _read_pair(roots,name)!=value:raise ValueError('Original numerical mirrors changed')

class AlgorithmNumericalFitFailure(ValueError):
    def __init__(self,key,scope,event):
        if key is not _KEY or type(self)is not AlgorithmNumericalFitFailure:raise TypeError('Only original final kernel may emit this failure')
        self._scope,self._event,self._owner=scope,event,_owner()
        self._token=object()
        self._digest=fingerprint(event);_ERRORS.add(self)
        super().__init__('original_numeric_linalg:'+self._digest)
    def __reduce__(self):raise TypeError('Original numerical failure cannot serialize/restore')
    def _check(self):
        if self not in _ERRORS or self._owner!=_owner()or fingerprint(self._event)!=self._digest:
            raise ValueError('Original live numerical failure identity lost')
        self._scope.producer.check()
        if _producer()is not self._scope.producer:raise ValueError('Different original producer cannot borrow failure')

class _Scope:
    def __init__(self,key,producer,runtime,authority,job,ticket,snapshot,data,recipe,budget,first):
        if key is not _KEY:raise TypeError('Original paid call only')
        self.producer,self.runtime,self.authority,self.job=producer,runtime,authority,job
        # Never retain full feature matrices or traceback frames in the
        # service-wide event map. Actual input hashes live in the event.
        self.ticket,self.snapshot,self.budget,self.first=ticket,snapshot,budget,first
        self.owner=_owner();self.calls=0;self.last=None;self.active=True
        self.roots=_paths(producer,job)
        self.files=_installed(producer)
        from .nominal_algorithm_policy import _live_prefix
        self.prefix=_live_prefix(runtime.lifecycle)
        self.binding={'version':VERSION,'producer_hash':fingerprint(producer.identity),'trial':producer.permit.trial,
            'job':job,'source_files':self.files,'ticket':{'path':str(ticket.path),'digest':ticket.digest},
            'snapshot':snapshot,'recipe':_plain(recipe.to_dict()),'prefix':self.prefix,
            'budget_binding_hash':budget._binding_hash,'lease':budget._lease.to_dict(),
            'reservation_id':budget._binding['reservation_id'],'runtime_directory':str(runtime.directory)}
        self.digest=fingerprint(self.binding)
        from .nominal_algorithm_execution import _exclusive
        _exclusive(self.roots);_write_pair(self.roots,'intent.json',self.binding)
    def check(self):
        if not self.active or _KERNEL.get()is not self or self.owner!=_owner()or _producer()is not self.producer:
            raise ValueError('Original active numerical kernel scope lost')
        self.producer.check();self.budget.check_time()
        if (fingerprint(self.binding)!=self.digest or self.budget._binding_hash!=self.binding['budget_binding_hash']
            or self.budget._lease.to_dict()!=self.binding['lease']or _read_pair(self.roots,'intent.json')!=self.binding):
            raise ValueError('Original numeric source/lease/budget binding changed')
    def capture(self,error,site,inputs):
        self.check()
        event={'version':VERSION,'binding_hash':self.digest,'kernel_site':site,'kernel_ordinal':self.calls,
            'inputs':inputs,'exception_type':'numpy.linalg.LinAlgError','exception_message':str(error)[:1024],
            'budget':self.budget.to_dict(),'source_files':self.files,'historical_authority':False}
        _write_pair(self.roots,f'kernel-{self.calls:06d}.json',event)
        self.check();self.last=(error,event)

def kernel_scope():
    scope=_KERNEL.get()
    if scope is None:return None
    if type(scope)is not _Scope:raise ValueError('Private whole-policy kernel scope required')
    scope.check();return scope

def fit_original_forward(runtime,authority,job,ticket,snapshot,data,recipe,budget,first):
    """Fixed issuer call; original fit and original exception remain unscoped."""
    from . import known_linear_forward as known,pool_forward as pool
    from .known_linear_reference import fit_known_linear
    from .joint_pool import fit_pool
    caller=sys._getframe(1).f_code
    if type(ticket)is known.KnownForwardTicket and caller is known.fit_known_forward.__code__:
        kind='known'
    elif type(ticket)is pool.PoolForwardTicket and caller is pool.fit_pool_forward.__code__:
        kind='pool'
    else:raise ValueError('Only the actual known/pool issuer call site may enter')
    p=_producer(runtime)
    if p is None:
        return (known.fit_known_linear(data,snapshot['cutoff'],recipe,budget=budget)if kind=='known'
                else pool.fit_pool(data,snapshot['cutoff'],recipe.pool,budget=budget))
    from .monthly_work_authority import MonthlyWorkAuthority
    from .reserved_work_budget import ReservedChannelBudget
    if type(authority)is not MonthlyWorkAuthority or type(budget)is not ReservedChannelBudget or budget.test_only or budget.executor_mode!='issuer':
        raise ValueError('Exact original prepaid issuer budget required')
    if _KERNEL.get()is not None:raise ValueError('Nested numerical fit scope refused')
    budget.check_time();record=authority.get(ticket.value['reservation_id'])
    if (record['status']!='running'or record['reservation_id']!=budget._binding['reservation_id']
        or record['lease_hash']!=budget._lease.digest or record['lease']!=budget._lease.to_dict()
        or record['process_identity']!={'pid':os.getpid(),'created':budget._lease.parent_created}
        or ticket.value['context']['job']!=job or snapshot['snapshot_id']!=ticket.value['snapshot_ref']['snapshot_id']
        or ticket.value['context']['lifecycle_directory']!=str(runtime.directory)
        or data.training_identity(snapshot['cutoff'])!=snapshot['training']
        or authority.manifest['source_files']!=p.identity['source_files']):
        raise ValueError('Original actual forward ticket/source/snapshot/lease differs')
    events=_EVENTS.setdefault(p,{})
    if job in events:raise ValueError('Original numerical job cannot retry')
    scope=_Scope(_KEY,p,runtime,authority,job,ticket,snapshot,data,recipe,budget,first)
    token=_KERNEL.set(scope)
    try:
        scope.check()
        from .nominal_fit_outcome import _LINALG_ERROR
        try:
            model=(fit_known_linear(data,snapshot['cutoff'],recipe,budget=budget)if kind=='known'
                else fit_pool(data,snapshot['cutoff'],recipe.pool,budget=budget))
        except _LINALG_ERROR as error:
            if type(error)is not _LINALG_ERROR or scope.last is None or scope.last[0]is not error:raise
            scope.check();event=scope.last[1]
            budget.finalize();actual=ReservedChannelBudget.verify_counts(authority,record['reservation_id'])
            if actual['test_only']or not actual['finalized']or actual['actual_counts']!=event['budget']['actual_counts']:
                raise ValueError('Original actual numerical debit differs')
            from .nonlinear_launcher import _kernel_snapshot,_validate_kernel
            from .neural_forward_launcher import _identity,_member,_observation
            last=_kernel_snapshot();_validate_kernel(first,last);_member(last,_identity())
            numeric=AlgorithmNumericalFitFailure(_KEY,scope,event)
            failed=authority.fail_consumed(record['reservation_id'],str(numeric))
            if failed['status']!='failed_consumed':raise ValueError('Interrupted/late work is unknown, not numerical inability')
            terminal={'version':VERSION,'binding_hash':scope.digest,'event_hash':fingerprint(event),
                'reservation':failed,'actual_accounting':actual,'kernel_before':first,'kernel_after':last,
                'observation':_observation(budget._lease),'model':None,'serving_completion':False,'formal_acceptance':False}
            _write_pair(scope.roots,'terminal.json',terminal);scope.check()
            events[job]={'token':numeric._token,'scope':scope,'event':event,'event_hash':numeric._digest,
                'binding':scope.binding,'terminal':terminal,'accepted':None}
            raise numeric from error
        scope.check()
        # Profile search can consume real failed lstsq calls and still succeed.
        # Those events remain diagnostics; no failed-job outcome is inserted.
        _write_pair(scope.roots,'model-return.json',{'binding_hash':scope.digest,'model_digest':model.digest})
        scope.check()
        return model
    finally:
        scope.active=False;scope.last=None;_KERNEL.reset(token)

def record_failed_job(runtime,job,error):
    """Called only by original monthly runtime after issuer's failure tail."""
    from .v7_forward import ForwardRuntime
    if (sys._getframe(1).f_code not in (ForwardRuntime._fit_known_month.__code__,ForwardRuntime._fit_pool_month.__code__)
        or type(error)is not AlgorithmNumericalFitFailure):raise ValueError('Original runtime typed failure branch required')
    error._check();p=_producer(runtime);state=_EVENTS.get(p,{}).get(job)
    if state is None or state['token']is not error._token or state['accepted']is not None:raise ValueError('Original numerical outcome absent/reused')
    scope=error._scope
    if scope.runtime is not runtime or scope.job!=job:raise ValueError('Numeric outcome borrowed another actual job')
    # Original issuer catch writes exactly this failure after this typed error.
    module=importlib.import_module('factorlab.discovery.known_linear_forward'if type(scope.ticket).__name__=='KnownForwardTicket'else'factorlab.discovery.pool_forward')
    failure=module._record(scope.ticket.path.parent/'launch/failure.json')
    if failure!={'reason':'AlgorithmNumericalFitFailure:'+str(error)}:raise ValueError('Original issuer failure differs from exact live numeric event')
    scope.budget._lease.assert_parent_alive();p.check()
    runtime.lifecycle.fail_fit(job,'numerical_failure')
    row=runtime.lifecycle.db.execute('SELECT status,model,failure FROM jobs WHERE id=?',(job,)).fetchone()
    if row is None or row[0]!='planned'or row[1]is not None or row[2]!='numerical_failure':raise ValueError('Original failed cold job unexpectedly admitted a model')
    from .evidence import file_hash
    from .nonlinear_launcher import _kernel_snapshot,_validate_kernel
    from .neural_forward_launcher import _identity,_member,_observation
    last=_kernel_snapshot();_validate_kernel(state['terminal']['kernel_after'],last);_member(last,_identity())
    accepted={'version':VERSION,'binding_hash':scope.digest,'terminal_hash':fingerprint(state['terminal']),
        'issuer_failure_sha256':file_hash(scope.ticket.path.parent/'launch/failure.json'),
        'job':job,'job_state':list(row),'prefix':scope.prefix,'numeric_event_hash':error._digest,
        'kernel':last,'observation':_observation(scope.budget._lease),
        'nominal_only':True,'actual_failure':{'error':1,'power':0,'retain_N':True}}
    _write_pair(scope.roots,'accepted.json',accepted)
    scope.budget._lease.assert_parent_alive();p.check();state['accepted']=accepted
    scope.runtime=None

def _verify_job(p,lc,job):
    if _producer()is not p:raise ValueError('Original live producer proof required; no restored failure JSON authority')
    state=_EVENTS.get(p,{}).get(job)
    if state is None or state['accepted']is None:raise ValueError('No original live accepted numerical emission')
    scope=state['scope']
    if type(scope)is not _Scope or scope.owner!=_owner()or scope.producer is not p or fingerprint(state['event'])!=state['event_hash']:
        raise ValueError('Original private live numerical emission changed')
    if str(lc.directory)!=state['binding']['runtime_directory']:raise ValueError('Numerical lifecycle differs')
    for name,value in [('intent.json',state['binding']),('terminal.json',state['terminal']),('accepted.json',state['accepted'])]:
        if _read_pair(scope.roots,name)!=value:raise ValueError('Original numerical chain changed')
    event=_read_pair(scope.roots,f"kernel-{state['event']['kernel_ordinal']:06d}.json")
    if event!=state['event']:raise ValueError('Original trusted numeric event changed')
    from .evidence import file_hash
    if file_hash(scope.ticket.path.parent/'launch/failure.json')!=state['accepted']['issuer_failure_sha256']:
        raise ValueError('Original issuer failure bytes changed')
    row=lc.db.execute('SELECT status,model,failure FROM jobs WHERE id=?',(job,)).fetchone()
    if row is None or list(row)!=state['accepted']['job_state']:raise ValueError('Original failed job became a model/changed')
    from .reserved_work_budget import ReservedChannelBudget
    record=scope.authority.get(scope.binding['reservation_id'])
    if record!=state['terminal']['reservation']or record['status']!='failed_consumed':raise ValueError('Original failure accounting changed')
    actual=ReservedChannelBudget.verify_counts(scope.authority,record['reservation_id'])
    if actual!=state['terminal']['actual_accounting']:raise ValueError('Original actual debit chain changed')
    from .neural_forward_admission import _kernel,_observation
    terminal=state['terminal'];accepted=state['accepted'];lease=scope.budget._lease
    parent=record['process_identity']
    _kernel(terminal['kernel_before'],terminal['kernel_before'],terminal['kernel_after'],lease,parent,parent)
    _kernel(terminal['kernel_before'],terminal['kernel_after'],accepted['kernel'],lease,parent,parent)
    _observation(terminal['observation'],lease);_observation(accepted['observation'],lease,previous=terminal['observation'])
    p.check()
    return {'job':job,'binding_hash':scope.digest,'event_hash':state['event_hash'],'terminal_hash':fingerprint(state['terminal']),
        'accepted_hash':fingerprint(state['accepted']),'prefix':scope.prefix,'cutoff':scope.snapshot['cutoff'],
        'stream':scope.ticket.value['context']['stream'],'pipeline':scope.ticket.value['context']['pipeline']}

def verify_failed_jobs(p,lc):
    rows=lc.db.execute('SELECT id,failure FROM jobs WHERE failure IS NOT NULL ORDER BY id').fetchall()
    states=_EVENTS.get(p,{})
    if {r[0]for r in rows}!=set(states)or any(r[1]!='numerical_failure'for r in rows):
        raise ValueError('Unclassified actual fit failure remains unknown/full N')
    return [_verify_job(p,lc,job)for job,_ in rows]

def numeric_unavailability(runtime,stream,origin,expected_pipeline):
    p=_producer(runtime)
    if p is None:return None
    return _unavailability(p,runtime.lifecycle,stream,origin,expected_pipeline)

def _unavailability(p,lc,stream,origin,expected_pipeline):
    from .v7_lifecycle import stamp
    from .quarantined_observation import publication_visible
    expected=None if expected_pipeline is None else {k:v for k,v in expected_pipeline.items()if k!='complexity'}
    candidates=[]
    for job,state in _EVENTS.get(p,{}).items():
        if state['accepted']is None:continue
        scope=state['scope'];c=scope.ticket.value['context']
        if c['stream']!=stream or expected is not None and c['pipeline']!=expected:continue
        if stamp(scope.snapshot['cutoff'])<=stamp(origin)and stamp(scope.prefix['logical_at'])<stamp(origin):
            candidates.append((stamp(scope.snapshot['cutoff']),job))
    if not candidates:return None
    failed_cutoff,job=max(candidates)
    # A later original paid same-recipe cold model must actually be visible;
    # a mere new plan, uncommitted model or current-prefix completion is not it.
    rows=lc.db.execute('SELECT id,cutoff,pipeline FROM jobs WHERE stream=? AND failure IS NULL '
        'AND status IN (?,?) AND cutoff>? AND cutoff<=? AND completed<=?',
        (stream,'verified','activated',failed_cutoff.isoformat(),stamp(origin).isoformat(),stamp(origin).isoformat())).fetchall()
    for newer,cutoff,pipeline in rows:
        if (expected is None or json.loads(pipeline)==expected)and publication_visible(lc,'model',newer,origin)is True:return None
    return _verify_job(p,lc,job)

def verify_numeric_row(p,lc,record,row):
    expected=_unavailability(p,lc,row['stream'],record['prefix']['logical_at'],record['recipes'][row['stream']])
    if row.get('numeric_failure')!=expected:raise ValueError('Original numerical cold unavailability changed')
    if expected is not None and (row['reason']not in ('original_numeric_cold_fit_unavailable','original_source_no_current_origin')or row['prediction']is not None or row['model_id']is not None):
        raise ValueError('Old model cannot replace original numeric-failed cold model')
    if expected is None and row['reason']=='original_numeric_cold_fit_unavailable':raise ValueError('Unproved numeric absence cannot become B')
