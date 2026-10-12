"""Finite original-service owner for <=23 live task completions.

Snapshots are provisional. History requires a separate retained OS service
observation. No JSON, timestamp, or success boolean constructs live authority.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import threading
from factorlab.protocol import fingerprint, project_path

RELATIVE='factorlab/discovery/native_verification_batch.py'
VERSION='native-verification-batch-v1'
TASK_VERSION='native-verification-task-provisional-v1'
MAX_TASKS=23
_KEY=object()
_CURRENT=None

def _plain(value):
    return json.loads(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False))

def _api():
    from . import nominal_verification_budget as quota
    from . import nominal_systemd_unit as units
    return quota,units

def _task_types():
    from .nominal_verification_deadline import OriginalTaskDeadline
    from .native_verification_launcher import NativeVerificationLaunch
    return OriginalTaskDeadline,NativeVerificationLaunch

def _source(service):
    from .monthly_work_authority import source_files
    from .evidence import file_hash
    files=source_files()
    if (Path(__file__).resolve()!=project_path(RELATIVE).resolve()
        or files.get(RELATIVE)!=file_hash(__file__)
        or files!=service.registry.plan.binding['source_files']):
        raise ValueError('Canonical batch and exact original frozen source required')
    service.registry.verify()
    return files

class BatchTaskObservation:
    def __init__(self,key,batch,index,snapshot,native):
        if key is not _KEY:raise TypeError('Only original live batch may issue an observation')
        self._batch,self._index=batch,index
        self._snapshot,self._native=_plain(snapshot),_plain(native)
    @property
    def snapshot(self):return _plain(self._snapshot)
    @property
    def native_observation(self):return _plain(self._native)
    def assert_current(self):return self._batch.read_task(self._index)
    def __reduce__(self):raise TypeError('Live batch observation cannot serialize or restore')

def current_batch(quota,service_index):
    """Only an existing in-memory original owner; never reconstruct from files."""
    book,_=_api()
    if type(quota) is not book.VerificationRegistry or type(service_index) is not int:
        raise TypeError('Exact original registry and integer service index required')
    batch=_CURRENT
    if batch is None:return None
    batch._owner()
    if (batch.closed or batch.failed or batch.index!=service_index
        or batch.service.registry.digest!=quota.digest
        or batch.service.registry.directory!=quota.directory
        or batch.service.registry.anchor!=quota.anchor):return None
    return batch

class VerificationBatch:
    def __init__(self,service):
        global _CURRENT
        book,units=_api()
        if type(service) is not book.VerificationService:
            raise TypeError('Exact already-acquired original VerificationService required')
        if _CURRENT is not None:raise ValueError('Original live service batch already retained')
        start,end=service.plan['task_start'],service.plan['task_end_exclusive']
        if (type(start) is not int or type(end) is not int or not 0<end-start<=MAX_TASKS):
            raise ValueError('Original fixed nonempty service range of at most23 required')
        self.service=service
        self.owner=(os.getpid(),threading.get_ident())
        self.index=service.plan['index'];self.indices=tuple(range(start,end))
        self.slots={};self.snapshots={};self.committed=set();self.observations={}
        self.closed=False;self.failed=False;self.finishing=False
        self.roots=[p/'native-batch' for p in service.roots]
        self.binding=None
        _CURRENT=self  # Consumed even if directory, unit admission, or IO fails.
        try:
            service.check();self.sources=_source(service)
            for p in self.roots:p.mkdir(exist_ok=False)
            self.binding=units.bind_current_service(service)
            if type(self.binding) is not units.RetainedServiceBinding:
                raise TypeError('Actual original retained-unit binding required')
            self.identity={'version':VERSION,'registry_digest':service.registry.digest,
                'service_index':self.index,'service_binding':_plain(service.binding),
                'task_indices':list(self.indices),'source_files':self.sources,
                'unit_binding':self.binding.identity,'unit_binding_digest':self.binding.digest,
                'max_retained_children':MAX_TASKS,'provisional_only':True}
            self.digest=fingerprint(self.identity)
            for p in self.roots:book._write(p/'intent.json',self.identity)
            self._live()
        except BaseException as error:
            self._fail(error);raise

    def __reduce__(self):raise TypeError('Live batch cannot serialize or resume')
    def _owner(self):
        if self.owner!=(os.getpid(),threading.get_ident()) or _CURRENT is not self:
            raise ValueError('Original live parent/thread batch required')
    def _anchors(self):
        book,_=_api()
        if any((p/'failure.json').exists() for p in (*self.roots,*self.service.roots)):
            raise ValueError('Any original batch/service failure dominates')
        if any(book._read(p/'intent.json')!=self.identity for p in self.roots):
            raise ValueError('Original dual batch intent changed')
        if _source(self.service)!=self.sources:raise ValueError('Original batch source changed')
    def _live(self):
        self._owner()
        if self.failed or self.closed or self.finishing:raise ValueError('Batch no longer open')
        self._anchors();self.binding.check_live()
    def _closing(self):
        self._owner()
        if self.failed or self.closed:raise ValueError('Batch failed or closed')
        self._anchors();self.binding.check_closing()
        self.service.clock.assert_parent_alive()
    def _fail(self,error):
        self.failed=True
        book,_=_api()
        value={'consumed':True,'reason':str(error)[:900],'retry_allowed':False,'retain_all_N':True}
        for p in self.roots:
            try:
                if p.is_dir() and not (p/'failure.json').exists():book._write(p/'failure.json',value)
            except BaseException as failure:error.add_note('Batch failure record unavailable: '+type(failure).__name__)
    def _claim_task(self,task,guard):
        book,_=_api()
        deadline,_=_task_types()
        if type(task) is not book.VerificationTask or task.service is not self.service:
            raise TypeError('Exact task acquired from this original service required')
        if (type(guard) is not deadline or guard.task is not task or guard.closed
            or guard.owner!=self.owner):raise TypeError('Exact active original task deadline owner required')
        self._live();i=task.task['index']
        if (type(i) is not int or i not in self.indices or i in self.slots
            or len(self.slots)>=MAX_TASKS or any(k not in self.committed for k in self.slots)
            or self.service.active_task is not task):raise ValueError('Fixed unused original task slot required')
        task.check();self.slots[i]={'task':task,'guard':guard,'launch':None}
    def _retain_task(self,launch):
        _,launcher=_task_types()
        if type(launch) is not launcher or launch.batch is not self:
            raise TypeError('Exact original launcher object for this batch required')
        self._owner();slot=self.slots[launch.task.task['index']]
        if slot['task'] is not launch.task or slot['guard'] is not launch.guard or slot['launch'] is not None:
            raise ValueError('Original once-only retained launch required')
        slot['launch']=launch
    def run_task(self,task):
        self._owner()
        from .native_verification_launcher import launch_original_task
        try:
            launch=launch_original_task(task,batch=self)
            launch.serve_until_completion()
            return self.observations[task.task['index']]
        except BaseException as error:
            self._fail(error);raise
    def _publish_task(self,launch,observed):
        book,_=_api();self._live()
        task=launch.task;i=task.task['index']
        if self.slots[i]['launch'] is not launch or launch.guard.closed:
            raise ValueError('Original active guarded task required')
        current=launch.attempt.read_completion()
        record=book._read(task.roots[0]/'semantic-output.json')
        ack=book._read(task.roots[0]/'semantic-ack.json')
        value={'version':TASK_VERSION,'registry_digest':self.service.registry.digest,
            'service_index':self.index,'task_index':i,'batch_intent_hash':self.digest,
            'task_intent':_plain(task.intent),'task_intent_hash':fingerprint(task.intent),
            'ticket_digest':launch.attempt.digest,'ticket':_plain(launch.attempt.identity),
            'lease':task.lease.to_dict(),'source_files':self.sources,
            'unit_binding_digest':self.binding.digest,'target':ack['target'],
            'semantic_payload_hash':fingerprint(record['payload']),
            'publication':_plain(launch.attempt.publication_snapshot),
            'terminal_binding':_plain(launch.handle._binding),
            'native_observation':_plain(observed),'native_recheck':_plain(current),
            'native_observation_digest':fingerprint(observed),'provisional_only':True}
        paths=[p/'native-provisional.json' for p in task.roots]
        for p in paths:book._write(p,value)
        if any(book._read(p)!=value for p in paths):raise ValueError('Provisional task mirrors changed')
        final_native=launch.attempt.read_completion()  # Live kernel and byte closure after IO.
        self.observations[i]=BatchTaskObservation(_KEY,self,i,value,final_native)
        launch._remaining()  # Strict original task and service check BEFORE timer close.
        self.snapshots[i]=value
    def _commit_task(self,launch):
        self._owner();i=launch.task.task['index']
        if self.failed or not launch.guard.closed or i not in self.snapshots:
            raise ValueError('Original guarded provisional publication incomplete')
        self.committed.add(i)
    def read_task(self,index):
        self._live()
        if type(index) is not int or index not in self.committed:
            raise ValueError('No original completed live task')
        book,_=_api();slot=self.slots[index];launch=slot['launch']
        if launch is None or launch.aborted or not launch.completed:
            raise ValueError('Original retained task launch unavailable')
        value=self.snapshots[index]
        if any(book._read(p/'native-provisional.json')!=value for p in slot['task'].roots):
            raise ValueError('Original provisional task bytes changed')
        native=launch.read_completion()
        self._live()
        return BatchTaskObservation(_KEY,self,index,value,native)
    def finish(self):
        """Close quota, publish inventory, then explicitly reap natural children.

        The returned files remain provisional until the separate original unit
        naturally exits zero within its original service deadline.
        """
        global _CURRENT
        self._live();book,_=_api()
        try:
            if self.committed!=set(self.indices):raise ValueError('All original service slots required; no deletion or backfill')
            rows=[]
            for i in self.indices:
                observation=self.read_task(i)
                rows.append({'task_index':i,'snapshot_hash':fingerprint(observation.snapshot)})
            self.finishing=True
            self.service.close()
            witness=self.service.registry.service_witness(self.index)
            inventory={'version':VERSION,'registry_digest':self.service.registry.digest,
                'service_index':self.index,'batch_intent_hash':self.digest,
                'unit_binding_digest':self.binding.digest,'source_files':self.sources,
                'records':rows,'service_witness_hash':fingerprint(witness),
                'provisional_only':True,'reap_pending':True}
            for p in self.roots:book._write(p/'inventory.json',inventory)
            if any(book._read(p/'inventory.json')!=inventory for p in self.roots):raise ValueError('Batch inventory mirrors differ')
            self._closing()
            reaped=[]
            for i in self.indices:
                launch=self.slots[i]['launch']
                live=launch.read_completion()
                wait=live['native_diagnostic']['sample']['wait']
                launch.handle.invalidate('explicit_natural_reap_after_batch_inventory')
                kind=getattr(os,'P_PIDFD',None)
                identifier=launch.pidfd if kind is not None else launch.process.pid
                if kind is None:kind=os.P_PID
                event=os.waitid(kind,identifier,os.WEXITED|os.WNOHANG)
                got=None if event is None else {k:int(getattr(event,k)) for k in ('si_pid','si_uid','si_signo','si_status','si_code')}
                if got!=wait or got['si_code']!=1 or got['si_status']!=0:
                    raise ValueError('Original natural-zero child reap unavailable')
                launch.process.returncode=0  # Already reaped explicitly; no destructor wait.
                launch.handle.close();os.close(launch.pidfd);launch.pidfd=None
                launch.channel.close()
                reaped.append({'task_index':i,'wait':got,'natural_reap':True,'cleanup_signal':False})
                self._closing()
            result={'version':VERSION,'batch_intent_hash':self.digest,
                'inventory_hash':fingerprint(inventory),'reaped':reaped,
                'provisional_only':True,'cleanup_signals':False}
            for p in self.roots:book._write(p/'reaped.json',result)
            if any(book._read(p/'reaped.json')!=result for p in self.roots):raise ValueError('Reap mirrors differ')
            self.binding.publish_quiescence()
            self._closing()
            self.closed=True;_CURRENT=None
            return _plain(result)
        except BaseException as error:
            self._fail(error);raise
