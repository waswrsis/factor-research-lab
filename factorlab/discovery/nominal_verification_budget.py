"""New nominal verification quota, never a fit wallet or truth certificate.

Only resource allocation/bookkeeping is implemented here. Canonical real law,
readback and report consumers and external OS supervision remain separate work.
"""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
from contextvars import ContextVar
import os
import threading
import hashlib
from filelock import FileLock
from factorlab.protocol import project_path, fingerprint
from .evidence import file_hash
from .monthly_work_authority import source_files, DEFAULT_LIMITS
from .monthly_partition_policy import frozen_policy
from .work_lease import FrozenWorkLease, _observe, _evaluate
from .full_trial_controller import ControllerStepLease
from .v7_storage import immutable_record

VERSION='new-nominal-verification-quota-v1'
RELATIVE='factorlab/discovery/nominal_verification_budget.py'
TASK_SECONDS=300
SERVICE_SECONDS=7200
TASKS_PER_SERVICE=SERVICE_SECONDS//TASK_SECONDS-1
RESOURCE_CONTRACT={'shared_cpu_fraction':[1,4],'shared_memory_soft_bytes':640*2**20,
 'shared_memory_hard_bytes':768*2**20,'swap_bytes':0,'tasks_max':64,
 'scope':'aggregate_all_concurrent_services_not_per_service_multiplication',
 'wall_seconds':7200,'external_OS_enforcement':'required_not_certified_by_this_component'}

def _path(value):
 from .calibration_trial_context import _no_redirect
 p=Path(value);p=p if p.is_absolute() else project_path('.')/p
 _no_redirect(p,project_path('.'));return project_path(p)

def _write(path,value):immutable_record(_path(path),{'record':value,'hash':fingerprint(value)})

def _read(path):
 p=_path(path)
 if not p.is_file():
  _remember_witness_read(p,_witness_path_state(p))
  raise ValueError('Bounded original quota record required')
 if p.stat().st_size>2**20:raise ValueError('Bounded original quota record required')
 def unique(pairs):
  out={}
  for k,v in pairs:
   if k in out:raise ValueError('Duplicate quota key')
   out[k]=v
  return out
 raw=p.read_bytes()
 v=json.loads(raw.decode('utf8'),object_pairs_hook=unique,
  parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
 if set(v)!={'record','hash'} or fingerprint(v['record'])!=v['hash']:raise ValueError('Quota record changed')
 _remember_witness_read(p,hashlib.sha256(raw).hexdigest())
 return v['record']

def _installed():
 from . import nominal_effect_design as design
 files=source_files()
 if Path(__file__).resolve()!=project_path(RELATIVE).resolve() or files.get(RELATIVE)!=file_hash(__file__):
  raise ValueError('Canonical new verification quota source required')
 relative='factorlab/discovery/nominal_effect_design.py'
 if Path(design.__file__).resolve()!=project_path(relative).resolve() or files.get(relative)!=file_hash(design.__file__):
  raise ValueError('Canonical prospective fixed effect design required')
 return files

def _acceptance(value):
 from .nominal_actual_trial import _acceptance as parse
 return parse(value)

def _scope(census,acceptance,target_directory):
 from .nominal_cold_conformance import NominalCensus
 from .calibration_acceptance import AcceptancePlan
 from .nominal_effect_design import require_full_library_grid,scenario_design,DESIGNS
 if type(census) is not NominalCensus or type(acceptance) is not AcceptancePlan:
  raise TypeError('Original typed census and whole prospective AcceptancePlan required')
 # No derived law object exists yet; heavy truth work is always registered.
 p=census.verify();files=_installed();b=p['profile_scope']['work_budget']
 if (p['source_files']!=files or p['atoms']!=list(range(128))
     or b['limits']!=DEFAULT_LIMITS or b['monthly_seconds']!=SERVICE_SECONDS
     or b['partition_policy']!=frozen_policy('standard')):
  raise ValueError('Actual complete128 standard original profile differs')
 require_full_library_grid(acceptance,census.digest)
 return {'census_directory':str(census.directory),'census_digest':census.digest,
  'census_identity_sha256':file_hash(census.directory/'identity.json'),
  'target_directory':str(target_directory),
  'library_designs':[scenario_design(census.digest,n) for n,_,_ in DESIGNS],
  'source_files':files,'profile_scope':p['profile_scope'],
  'acceptance':json.loads(json.dumps(acceptance.to_dict())),'acceptance_digest':acceptance.digest}

class FrozenVerificationPlan:
 """Compact exact task enumeration; caller cannot add work or reduce N."""
 def __init__(self,binding):
  self.binding=binding;self.acceptance=_acceptance(binding['acceptance'])
  from .nominal_effect_design import require_full_library_grid,scenario_design,DESIGNS
  require_full_library_grid(self.acceptance,binding['census_digest'])
  if binding['library_designs']!=[scenario_design(binding['census_digest'],n) for n,_,_ in DESIGNS]:
   raise ValueError('Fixed prospective design map changed')
  if self.acceptance.digest!=binding['acceptance_digest']:raise ValueError('Whole original acceptance changed')
  self.total_N=sum(c.repetitions for c in self.acceptance.cells)
  self.total_tasks=self.total_N+len(self.acceptance.cells)+133
  self.phases=[];service_start=0
  for name,start,end in [('registration',0,1),('cold_grid_proof',1,131),
                        ('actual_full_N_readback',131,131+self.total_N),
                        ('final_verification_and_report',131+self.total_N,self.total_tasks)]:
   count=1 if start==0 else (end-start+TASKS_PER_SERVICE-1)//TASKS_PER_SERVICE
   self.phases.append({'phase':name,'task_start':start,'task_end_exclusive':end,
                      'service_start':service_start,'service_end_exclusive':service_start+count})
   service_start+=count
  self.total_services=service_start
 def task(self,index):
  if type(index) is not int or not 0<=index<self.total_tasks:raise ValueError('Registered integer task required')
  if index==0:kind,unit='registration_header_validation',None
  elif index<=128:kind,unit='cold_atom_evidence_validation',{'atom':index-1}
  elif index==129:kind,unit='law_reduction',None
  elif index==130:kind,unit='full_law_verify_before_actual_registry',None
  elif index<131+self.total_N:
   i=index-131
   for c in self.acceptance.cells:
    if i<c.repetitions:kind,unit='actual_coordinate_readback',{'cell':c.name,'index':i};break
    i-=c.repetitions
  elif index==131+self.total_N:kind,unit='full_law_verify_before_final_report',None
  elif index<self.total_tasks-1:
   kind,unit='acceptance_cell_aggregation',{'cell':self.acceptance.cells[index-(132+self.total_N)].name}
  else:kind,unit='final_full_N_report',None
  phase=next(p for p in self.phases if p['task_start']<=index<p['task_end_exclusive'])
  return {'index':index,'kind':kind,'unit':unit,'service':phase['service_start']+(index-phase['task_start'])//TASKS_PER_SERVICE,
   'seconds':TASK_SECONDS,'retry_allowed':False,'grants_fit_or_GPT_credit':False}
 def service(self,index):
  if type(index) is not int or not 0<=index<self.total_services:raise ValueError('Registered integer service required')
  phase=next(p for p in self.phases if p['service_start']<=index<p['service_end_exclusive'])
  start=phase['task_start']+(index-phase['service_start'])*TASKS_PER_SERVICE
  end=min(start+TASKS_PER_SERVICE,phase['task_end_exclusive'])
  return {'index':index,'task_start':start,'task_end_exclusive':end,'prepaid_seconds':(end-start)*TASK_SECONDS,
   'phase':phase['phase'],
   'wall_seconds':SERVICE_SECONDS,'nonspendable_headroom_seconds':SERVICE_SECONDS-(end-start)*TASK_SECONDS,
   'retry_allowed':False,'resource_contract':RESOURCE_CONTRACT}
 def to_dict(self):
  return {'version':VERSION,'binding':self.binding,'atoms':128,'total_N':self.total_N,
   'total_tasks':self.total_tasks,'total_prepaid_seconds':TASK_SECONDS*self.total_tasks,
   'total_services':self.total_services,'tasks_per_nonregistration_service':TASKS_PER_SERVICE,
   'fixed_phase_service_ranges':self.phases,
   'service_count_formula':'1+ceil(130/23)+ceil(N/23)+ceil((C+2)/23)',
   'enumeration':'task(index) canonical source only; every original cell/index retained',
   'registration_service_separate':True,'failed_or_missing':'consumed_or_unknown_never_removed_from_N',
   'quota':'new_user_approved_nominal_pure_verification_not_old_fit_wallet',
   'actual_consumers_integrated':False,'semantic_proof_certified':False}

def _scope_bytes(value):
 return json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode('utf8')

class _WitnessReadScope:
 """Private invocation snapshot, never a lease, proof or writable authority."""
 __slots__=('registry','directory','digest','identity','source','owner','active','reads')
 def __init__(self,registry):
  self.registry=registry;self.directory=str(registry.directory);self.digest=registry.digest
  self.identity=_scope_bytes(registry.identity)
  self.source=_scope_bytes(registry.plan.binding['source_files'])
  self.owner=(os.getpid(),threading.get_ident());self.active=True;self.reads={}

_WITNESS_READ=ContextVar('original_nominal_verification_witness_read',default=None)

def _forbid_witness_mutation():
 if _WITNESS_READ.get() is not None:
  raise ValueError('Readonly witness scope cannot allocate, finish or close work')

def _validate_witness_scope(registry,scope):
 if (type(scope) is not _WitnessReadScope or not scope.active
     or type(registry) is not VerificationRegistry or scope.registry is not registry
     or scope.owner!=(os.getpid(),threading.get_ident())
     or scope.directory!=str(registry.directory) or scope.digest!=registry.digest
     or scope.identity!=_scope_bytes(registry.identity)):
  raise ValueError('Original active same-thread same-registry witness scope required')

def _witness_path_state(path):
 path=_path(path)
 if not path.exists():return None
 if not path.is_file():return 'existing-nonfile'
 if path.stat().st_size>2**20:raise ValueError('Bounded original witness record required')
 return hashlib.sha256(path.read_bytes()).hexdigest()

def _remember_witness_read(path,digest):
 scope=_WITNESS_READ.get()
 if scope is None:return
 _validate_witness_scope(scope.registry,scope)
 key=str(path)
 if key in scope.reads and scope.reads[key]!=digest:
  raise ValueError('Quota bytes or failure marker changed inside witness scope')
 scope.reads[key]=digest

def _witness_failure_present(paths):
 found=False
 for path in paths:
  state=_witness_path_state(path)
  _remember_witness_read(path,state)
  if state is not None:found=True
 return found

def _verify_witness_reads(scope):
 for path,expected in scope.reads.items():
  if _witness_path_state(Path(path))!=expected:
   raise ValueError('Witness-read quota bytes/failure markers changed before exit')

def _witness_source(registry):
 scope=_WITNESS_READ.get()
 if scope is None:return _installed()
 _validate_witness_scope(registry,scope)
 # Only source bytes already checked at this invocation's entry are reused.
 # Every quota/census/clock/failure record is still read by _witness_verify.
 return json.loads(scope.source)

class VerificationRegistry:
 @classmethod
 def freeze(cls,directory,*,census,acceptance,target_directory):
  # Meter begins before source/header checks, never after full truth validation.
  _forbid_witness_mutation()
  from .nominal_cold_conformance import NominalCensus
  from .calibration_acceptance import AcceptancePlan
  if type(census) is not NominalCensus or type(acceptance) is not AcceptancePlan:
   raise TypeError('Original typed parents required')
  lease=FrozenWorkLease.issue(TASK_SECONDS)
  directory=_path(directory);anchor=_path(census.directory/'nominal_verification_registration')
  target_directory=_path(target_directory)
  allocation=target_directory.with_name(target_directory.name+'.verification-allocation.json')
  global_lock=FileLock(str(project_path('work/nominal_verification_service.lock')),timeout=0)
  global_lock.acquire()
  try:
   with FileLock(str(anchor)+'.creation.lock',timeout=0):
    if directory.exists() or anchor.exists() or target_directory.exists() or allocation.exists():raise ValueError('Census/target verification already registered/attempted; no reset')
    for p in (directory,anchor):p.mkdir(parents=True)
    intent={'version':VERSION,'directory':str(directory),'anchor':str(anchor),'census_directory':str(census.directory),
     'claimed_census_digest':census.digest,'acceptance':json.loads(json.dumps(acceptance.to_dict())),
     'target_directory':str(target_directory),
     'registration_lease':lease.to_dict(),'retry_allowed':False,'full_law_verified':False}
    for p in (directory,anchor):_write(p/'intent.json',intent)
    _write(allocation,intent)
    try:
     lease.assert_parent_alive();binding=_scope(census,acceptance,target_directory)
     plan=FrozenVerificationPlan(binding).to_dict()
     identity={'intent_hash':fingerprint(intent),'directory':str(directory),'anchor':str(anchor),'plan':plan}
     for p in (directory,anchor):_write(p/'identity.json',identity)
     observed=_observe(lease)
     if not _evaluate(lease,observed).allowed:raise ValueError('Registration original deadline lost')
     terminal={'task':FrozenVerificationPlan(binding).task(0),'lease':lease.to_dict(),'observed':observed,
      'identity_hash':fingerprint(identity),'state':'resource_completed_not_truth_certificate'}
     for p in (directory,anchor):_write(p/'registration-terminal.json',terminal)
     successor={'terminal_hash':fingerprint(terminal),'observed':_observe(lease)}
     for p in (directory,anchor):_write(p/'registration-successor.json',successor)
     lease.assert_parent_alive()
    except BaseException as error:
     for p in (directory,anchor):_write(p/'registration-failure.json',{'consumed':True,'reason':str(error)[:1000]})
     raise
  finally:global_lock.release()
  return cls(directory)
 def __init__(self,directory):
  self.directory=_path(directory);self.identity=_read(self.directory/'identity.json')
  self.anchor=_path(self.identity['anchor']);self.digest=fingerprint(self.identity)
  self.plan=FrozenVerificationPlan(self.identity['plan']['binding']);self.verify()
 def verify(self):
  if any((p/'registration-failure.json').exists() for p in (self.directory,self.anchor)):raise ValueError('Registration consumed failure')
  for name in ('intent.json','identity.json','registration-terminal.json','registration-successor.json'):
   if _read(self.directory/name)!=_read(self.anchor/name):raise ValueError('Original registry dual anchors differ')
  intent=_read(self.directory/'intent.json');identity=_read(self.directory/'identity.json')
  b=self.plan.binding
  if (identity!=self.identity or identity['directory']!=str(self.directory) or identity['anchor']!=str(self.anchor)
      or identity['intent_hash']!=fingerprint(intent) or self.plan.to_dict()!=identity['plan']
      or b['source_files']!=_installed() or intent['acceptance']!=b['acceptance']
      or intent['census_directory']!=b['census_directory'] or intent['claimed_census_digest']!=b['census_digest']
      or self.anchor!=_path(Path(b['census_directory'])/'nominal_verification_registration')
      or intent['target_directory']!=b['target_directory']
      or file_hash(Path(b['census_directory'])/'identity.json')!=b['census_identity_sha256']):
   raise ValueError('Frozen quota source/parents/wholeN changed')
  target=_path(b['target_directory'])
  if _read(target.with_name(target.name+'.verification-allocation.json'))!=intent:raise ValueError('Original grid target allocation changed')
  r=_read(self.directory/'registration-terminal.json');lease=FrozenWorkLease.from_dict(intent['registration_lease'])
  successor=_read(self.directory/'registration-successor.json')
  if (r['lease']!=lease.to_dict() or r['identity_hash']!=self.digest or r['task']!=self.plan.task(0)
      or r['state']!='resource_completed_not_truth_certificate' or not _evaluate(lease,r['observed']).allowed
      or successor['terminal_hash']!=fingerprint(r) or not _evaluate(lease,successor['observed'],r['observed']).allowed):
   raise ValueError('Original registration resource witness differs')
  return identity
 def _witness_verify(self):
  if _witness_failure_present([p/'registration-failure.json' for p in (self.directory,self.anchor)]):raise ValueError('Registration consumed failure')
  for name in ('intent.json','identity.json','registration-terminal.json','registration-successor.json'):
   if _read(self.directory/name)!=_read(self.anchor/name):raise ValueError('Original registry dual anchors differ')
  intent=_read(self.directory/'intent.json');identity=_read(self.directory/'identity.json')
  b=self.plan.binding
  if (identity!=self.identity or identity['directory']!=str(self.directory) or identity['anchor']!=str(self.anchor)
      or identity['intent_hash']!=fingerprint(intent) or self.plan.to_dict()!=identity['plan']
      or b['source_files']!=_witness_source(self) or intent['acceptance']!=b['acceptance']
      or intent['census_directory']!=b['census_directory'] or intent['claimed_census_digest']!=b['census_digest']
      or self.anchor!=_path(Path(b['census_directory'])/'nominal_verification_registration')
      or intent['target_directory']!=b['target_directory']
      or file_hash(Path(b['census_directory'])/'identity.json')!=b['census_identity_sha256']):
   raise ValueError('Frozen quota source/parents/wholeN changed')
  target=_path(b['target_directory'])
  if _read(target.with_name(target.name+'.verification-allocation.json'))!=intent:raise ValueError('Original grid target allocation changed')
  r=_read(self.directory/'registration-terminal.json');lease=FrozenWorkLease.from_dict(intent['registration_lease'])
  successor=_read(self.directory/'registration-successor.json')
  if (r['lease']!=lease.to_dict() or r['identity_hash']!=self.digest or r['task']!=self.plan.task(0)
      or r['state']!='resource_completed_not_truth_certificate' or not _evaluate(lease,r['observed']).allowed
      or successor['terminal_hash']!=fingerprint(r) or not _evaluate(lease,successor['observed'],r['observed']).allowed):
   raise ValueError('Original registration resource witness differs')
  return identity
 @contextmanager
 def witness_read_scope(self):
  """Readonly witness snapshot; public authority methods never borrow it.

  Entry and exit run the original complete verify/source checks. Intermediate
  witnesses reread original headers, census, clocks, dual anchors and failures;
  only the validated source inventory is reused. Source mutation can therefore
  be detected at exit rather than immediately. Consumers must not publish or
  grant work until this context exits successfully. No OS immutability claim.
  """
  if type(self) is not VerificationRegistry:raise TypeError('Exact original registry required')
  parent=_WITNESS_READ.get()
  if parent is not None:_validate_witness_scope(self,parent)
  self.verify()
  scope=_WitnessReadScope(self);token=_WITNESS_READ.set(scope)
  try:
   yield
  finally:
   # Exit audit is unconditional, including body exceptions. Clear this scope
   # first, so no consumer can use it after exit, even from a copied context.
   scope.active=False;_WITNESS_READ.reset(token)
   self.verify()
   _verify_witness_reads(scope)
   if parent is not None:
    for path,digest in scope.reads.items():_remember_witness_read(Path(path),digest)
 def service(self,index):return VerificationService(self,index)
 def service_witness(self,index):
  self._witness_verify();plan=self.plan.service(index)
  if index==0:return self.task_witness(0)
  roots=[p/'services'/f'{index:06d}' for p in (self.directory,self.anchor)]
  if _witness_failure_present([p/'failure.json' for p in roots]):raise ValueError('Service failure consumed')
  for n in ('intent.json','terminal.json','successor.json'):
   if _read(roots[0]/n)!=_read(roots[1]/n):raise ValueError('Service completion anchors differ')
  i=_read(roots[0]/'intent.json');t=_read(roots[0]/'terminal.json')
  successor=_read(roots[0]/'successor.json')
  clock=ControllerStepLease.from_dict(i['clock'],SimpleNamespace(service={'wall_seconds':SERVICE_SECONDS}))
  if (i['registry_digest']!=self.digest or i['plan']!=plan or t['binding_hash']!=fingerprint(i)
      or i['resource_contract']!=RESOURCE_CONTRACT or i['external_supervision_pending'] is not True
      or not clock.evaluate(t['observed']).allowed or t['semantic_proof_certified'] is not False
      or t['unused_allocations_consumed'] is not True or successor['terminal_hash']!=fingerprint(t)
      or not clock.evaluate(successor['observed'],t['observed']).allowed):raise ValueError('Original service witness changed')
  states=[]
  for k in range(plan['task_start'],plan['task_end_exclusive']):
   try:self.task_witness(k);states.append('resource_completed_not_truth_certificate')
   except (ValueError,OSError):states.append('failed_or_missing_consumed')
  if states!=t['states']:raise ValueError('Task state changed after service completion')
  return t
 def task_witness(self,index):
  self._witness_verify();task=self.plan.task(index)
  if index==0:return _read(self.directory/'registration-terminal.json')
  service_roots=[p/'services'/f"{task['service']:06d}" for p in (self.directory,self.anchor)]
  if _witness_failure_present([p/'failure.json' for p in service_roots]):raise ValueError('Any original service failure consumes its tasks')
  roots=[p/'tasks'/f'{index:06d}' for p in (self.directory,self.anchor)]
  if _witness_failure_present([p/'failure.json' for p in roots]):raise ValueError('Task failure remains consumed')
  for n in ('intent.json','terminal.json','successor.json'):
   if _read(roots[0]/n)!=_read(roots[1]/n):raise ValueError('Task original dual anchors differ')
  i=_read(roots[0]/'intent.json');t=_read(roots[0]/'terminal.json');s=_read(roots[0]/'successor.json')
  lease=FrozenWorkLease.from_dict(i['lease'])
  if (i['registry_digest']!=self.digest or i['task']!=task or t['intent_hash']!=fingerprint(i)
      or t['state']!='resource_completed_not_truth_certificate' or s['terminal_hash']!=fingerprint(t)
      or lease.lease_seconds!=TASK_SECONDS or t['semantic_proof_certified'] is not False
      or t['actual_consumer_integrated'] is not False
      or not _evaluate(lease,t['observed']).allowed or not _evaluate(lease,s['observed'],t['observed']).allowed):
   raise ValueError('Original nonrenewable task witness invalid')
  service_roots=[p/'services'/f"{task['service']:06d}" for p in (self.directory,self.anchor)]
  sv=_read(service_roots[0]/'intent.json')
  if (sv!=_read(service_roots[1]/'intent.json') or i['service_binding']!=fingerprint(sv)
      or sv['resource_contract']!=RESOURCE_CONTRACT or sv['external_supervision_pending'] is not True):raise ValueError('Task service anchor changed')
  clock=ControllerStepLease.from_dict(sv['clock'],SimpleNamespace(service={'wall_seconds':SERVICE_SECONDS}))
  if (sv['plan']!=self.plan.service(task['service']) or not clock.evaluate(t['service_observed']).allowed
      or any(getattr(lease,k)!=clock.issuer[k] for k in ('parent_pid','parent_created','boot_id'))
      or lease.monotonic_started<clock.issuer['monotonic_started']
      or lease.monotonic_deadline>clock.supervisor().monotonic_deadline):
   raise ValueError('Task outside original service resource envelope')
  return t

class VerificationService:
 def __init__(self,registry,index):
  _forbid_witness_mutation()
  if type(registry) is not VerificationRegistry:raise TypeError('Typed original verification registry required')
  registry.verify();self.registry=registry;self.plan=registry.plan.service(index)
  if index==0:raise ValueError('Registration service already consumed at freeze')
  self.roots=[p/'services'/f'{index:06d}' for p in (registry.directory,registry.anchor)]
  self.lock=FileLock(str(project_path('work/nominal_verification_service.lock')),timeout=0);self.lock.acquire()
  self.closed=False;self.active_task=None
  from threading import RLock
  self.task_lock=RLock()
  try:
   if any(p.exists() for p in self.roots):raise ValueError('Service already attempted; no renewal or replacement')
   self.clock=ControllerStepLease.issue(SimpleNamespace(service={'wall_seconds':SERVICE_SECONDS}))
   self.binding={'registry_digest':registry.digest,'plan':self.plan,'clock':self.clock.to_dict(),
    'external_supervision_pending':True,'resource_contract':RESOURCE_CONTRACT}
   for p in self.roots:p.mkdir(parents=True);_write(p/'intent.json',self.binding)
   self.check()
  except BaseException:self.closed=True;self.lock.release();raise
 def check(self):
  if self.closed:raise ValueError('Service is closed')
  if any((p/'failure.json').exists() for p in self.roots):raise ValueError('Any original service failure consumes the service')
  if any(_read(p/'intent.json')!=self.binding for p in self.roots):raise ValueError('Service original anchors differ')
  return self.clock.assert_parent_alive()
 def acquire(self,index):
  _forbid_witness_mutation()
  with self.task_lock:return VerificationTask(self,index)
 def close(self):
  _forbid_witness_mutation()
  if self.closed:raise ValueError('Original service already closed')
  try:
   self.check()
   if self.active_task is not None and not self.active_task.closed:
    self.active_task.fail('service_closed_with_unfinished_task')
   self.check();states=[]
   for i in range(self.plan['task_start'],self.plan['task_end_exclusive']):
    try:self.registry.task_witness(i);states.append('resource_completed_not_truth_certificate')
    except (ValueError,OSError):states.append('failed_or_missing_consumed')
   observed=self.clock.observe()
   record={'binding_hash':fingerprint(self.binding),'states':states,'observed':observed,
    'unused_allocations_consumed':True,'semantic_proof_certified':False}
   for p in self.roots:_write(p/'terminal.json',record)
   successor={'terminal_hash':fingerprint(record),'observed':self.clock.observe()}
   for p in self.roots:_write(p/'successor.json',successor)
   self.check()
  except BaseException as error:
   failure={'consumed':True,'reason':str(error)[:1000],'retain_all_N':True}
   for p in self.roots:
    try:
     if not (p/'failure.json').exists():_write(p/'failure.json',failure)
    except (ValueError,OSError) as recording_error:
     error.add_note('Failure recording could not complete: '+type(recording_error).__name__)
   raise
  finally:self.closed=True;self.lock.release()

class VerificationTask:
 def __init__(self,service,index):
  _forbid_witness_mutation()
  if type(service) is not VerificationService:raise TypeError('Original active service required')
  service.check();service.registry.verify();task=service.registry.plan.task(index)
  if service.active_task is not None and not service.active_task.closed:raise ValueError('One task at a time inside shared service')
  if task['service']!=service.plan['index']:raise ValueError('Task cannot move to another service')
  self.service=service;self.task=task
  self.roots=[p/'tasks'/f'{index:06d}' for p in (service.registry.directory,service.registry.anchor)]
  if any(p.exists() for p in self.roots):raise ValueError('Task allocated before; no retry')
  if service.check().remaining_seconds<=TASK_SECONDS:raise ValueError('Complete task lease no longer fits service')
  self.lease=FrozenWorkLease.issue(TASK_SECONDS);self.closed=False
  self.intent={'registry_digest':service.registry.digest,'task':task,'lease':self.lease.to_dict(),
   'service_binding':fingerprint(service.binding)}
  # Same-thread issuance is serialized by the service; filesystem exclusivity
  # rejects another invocation even if it obtained the same live object.
  for p in self.roots:p.mkdir(parents=True);_write(p/'intent.json',self.intent)
  service.active_task=self
  self.check()
 def check(self):
  if self.closed:raise ValueError('Task already terminal')
  self.service.check()
  if any((p/'failure.json').exists() or _read(p/'intent.json')!=self.intent for p in self.roots):raise ValueError('Original task differs/failed')
  return self.lease.assert_parent_alive()
 def finish(self):
  _forbid_witness_mutation()
  if self.closed:raise ValueError('Task already terminal; never overwrite')
  try:
   self.check();self.service.registry.verify()
   value={'intent_hash':fingerprint(self.intent),'observed':_observe(self.lease),
    'service_observed':self.service.clock.observe(),'state':'resource_completed_not_truth_certificate',
    'semantic_proof_certified':False,'actual_consumer_integrated':False}
   for p in self.roots:_write(p/'terminal.json',value)
   self.check();successor={'terminal_hash':fingerprint(value),'observed':_observe(self.lease)}
   for p in self.roots:_write(p/'successor.json',successor)
   self.check();self.closed=True
   return self.service.registry.task_witness(self.task['index'])
  except BaseException as error:
   self.fail(type(error).__name__+':'+str(error));raise
 def fail(self,reason):
  _forbid_witness_mutation()
  record={'consumed':True,'reason':str(reason)[:1000],'retain_all_N':True,'retry_allowed':False}
  for p in self.roots:_write(p/'failure.json',record)
  self.closed=True
