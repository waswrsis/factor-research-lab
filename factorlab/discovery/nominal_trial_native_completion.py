"""Original nominal physical-trial units, distinct from verification units.

Only an actual retained invocation with timely natural zero exit authorizes
history. Snapshot files and the live worker's return remain provisional.
"""
from __future__ import annotations
from contextvars import ContextVar
from datetime import datetime,timezone,timedelta
from fractions import Fraction
import json
import os
from pathlib import Path
import re
import sys
import threading
import psutil
from factorlab.protocol import fingerprint,project_path

RELATIVE='factorlab/discovery/nominal_trial_native_completion.py'
WORKER='factorlab.discovery.nominal_trial_native_worker'
WORKER_RELATIVE='factorlab/discovery/nominal_trial_native_worker.py'
VERSION='nominal-trial-retained-unit-v1'
RETURN_VERSION='nominal-trial-native-return-v1'
_KEY=object()
_RUN=ContextVar('original_nominal_physical_unit',default=None)

def _plain(v):return json.loads(json.dumps(v,sort_keys=True,allow_nan=False))
def _api():
    from . import nominal_systemd_unit as units
    from . import nominal_actual_trial as actual
    from . import nominal_trial_controller as controller
    from . import native_verification_terminal as kernel
    return units,actual,controller,kernel
def _installed(identity):
    from .monthly_work_authority import source_files
    from .evidence import file_hash
    files=source_files()
    if sys.platform!='linux':raise OSError('Original Linux actual-trial unit required')
    if Path(__file__).resolve()!=project_path(RELATIVE).resolve():raise ValueError('Canonical actual-native producer required')
    for p in (RELATIVE,WORKER_RELATIVE,'factorlab/discovery/nominal_trial_controller.py',
              'factorlab/discovery/nominal_systemd_unit.py'):
        if files.get(p)!=file_hash(project_path(p)):raise ValueError('Canonical actual-native source closure incomplete')
    if files!=identity['source_files']:raise ValueError('Original actual registry source differs')
    return files
def _lexical(directory,cell,index,batch_index):
    if type(cell)is not str or re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_-]{0,79}',cell)is None:
        raise ValueError('Fixed supported original cell name required')
    if type(index)is not int or not 0<=index<2000 or type(batch_index)is not int or batch_index not in (0,1):
        raise ValueError('Original two-physical-month simulation coordinates required')
    directory=project_path(directory)
    from .calibration_trial_context import _no_redirect
    _no_redirect(directory,project_path('.'))
    return directory
def trial_command(directory,cell,index,batch_index):
    directory=_lexical(directory,cell,index,batch_index)
    return [sys.executable,'-B','-m',WORKER,'--registry',str(directory),'--cell',cell,
            '--trial-index',str(index),'--batch-index',str(batch_index)]
def trial_unit_environment():return _api()[0].unit_environment()
def _name(coordinate,batch_index):
    return 'factorresearch-nt-'+fingerprint(coordinate)+'-'+f'{batch_index:02d}'+'.service'
def trial_unit_name(registry,cell,index,batch_index):
    _,actual,_,_=_api()
    if type(registry)is not actual.NominalActualRegistry:raise TypeError('Exact original actual registry required')
    _lexical(registry.directory,cell,index,batch_index)
    coordinate=registry.coordinate(cell,index)
    if batch_index>=len(registry.identity['work_batches']):raise ValueError('Original registered batch absent')
    return _name(coordinate,batch_index)
def _roots(trial,batch_index):
    return (trial.paths['root']/'controller_batches'/f'{batch_index:02d}',
        trial.registry.directory/'controller_batches'/trial.digest/f'{batch_index:02d}')
def _attempt_roots(value):
    c=value['coordinate'];directory=project_path(value['registry_directory'])
    root=directory/'trials'/f"v{c['cell_ordinal']:02d}-{c['index']:04d}"
    paths=(root/'native-unit-attempts'/f"{value['batch_index']:02d}",
        directory/'native_unit_attempts'/fingerprint(c)/f"{value['batch_index']:02d}")
    from .calibration_trial_context import _no_redirect
    for p in paths:_no_redirect(p,directory)
    return root,paths
def _claim_bootstrap(value):
    root,paths=_attempt_roots(value);controller=_api()[2]
    if not root.is_dir() or not (root/'draw-result.json').is_file():raise ValueError('Existing original drawn trial required')
    if (root/'failure.json').exists() or any(p.exists()for p in paths):raise ValueError('Actual native attempt already consumed')
    for p in paths:p.mkdir(parents=True,exist_ok=False);controller._write(p/'intent.json',value)
    _check_attempt(value)
def _check_attempt(value):
    root,paths=_attempt_roots(value)
    if (root/'failure.json').exists() or any((p/'failure.json').exists()for p in paths):
        raise ValueError('Actual native bootstrap failure remains consumed')
    if _mirrors(paths,'intent.json')!=value:raise ValueError('Original actual native bootstrap mirrors differ')
def _fail_bootstrap(value,error):
    root,paths=_attempt_roots(value);controller=_api()[2]
    failure={'consumed':True,'reason':type(error).__name__+':'+str(error)[:900],
             'family_error':1,'power':0,'retain_N':True,'retry_allowed':False}
    for p in (root,*paths):
        try:
            if p.is_dir() and not (p/'failure.json').exists():controller._write(p/'failure.json',failure)
        except BaseException as e:error.add_note('Actual bootstrap failure unavailable: '+type(e).__name__)
def _mirrors(roots,name):
    read=_api()[1]._read
    value=read(roots[0]/name)
    if read(roots[1]/name)!=value:raise ValueError('Original actual native mirrors differ: '+name)
    return value
def _fail(trial,batch_index,error):
    _,_,controller,_=_api()
    value={'consumed':True,'reason':type(error).__name__+':'+str(error)[:900],
        'family_error':1,'power':0,'retain_N':True,'retry_allowed':False}
    for p in (trial.paths['root'],*_roots(trial,batch_index)):
        try:
            if p.is_dir() and not (p/'failure.json').exists():controller._write(p/'failure.json',value)
        except BaseException as e:error.add_note('Original actual failure unavailable: '+type(e).__name__)
def _failed(trial,roots):
    if any((p/n).exists() for p in (trial.paths['root'],*roots) for n in ('failure.json','prefix-session-failure.json')):
        raise ValueError('Original failed actual trial remains consumed')

def _exit_deadline(identity,raw):
    """Both original lease AND real ExecMainStart deadlines, strictly earlier."""
    issuer=identity['clock']['issuer']
    start_m=Fraction(raw['ExecMainStartTimestampMonotonic'],10**6)
    start_w=Fraction(raw['ExecMainStartTimestamp'],10**6)
    end_m=Fraction(raw['ExecMainExitTimestampMonotonic'],10**6)
    end_w=Fraction(raw['ExecMainExitTimestamp'],10**6)
    wall=datetime.fromisoformat(issuer['wall_started_at'])
    dm=min(Fraction(issuer['monotonic_started'])+7200,Fraction(issuer['monotonic_started']+7200),start_m+7200)
    dw=min(Fraction(wall.timestamp())+7200,Fraction((wall+timedelta(seconds=7200)).timestamp()),start_w+7200)
    if (identity['clock']['service_seconds']!=7200 or issuer['boot_id']!=identity['boot_id']
        or not 0<start_m<=Fraction(issuer['monotonic_started'])<end_m<dm
        or not 0<start_w<=Fraction(wall.timestamp())<end_w<dw
        or datetime.fromtimestamp(float(start_w),timezone.utc).strftime('%Y-%m')!=identity['batch']['actual_UTC_month']
        or datetime.fromtimestamp(float(end_w),timezone.utc).strftime('%Y-%m')!=identity['batch']['actual_UTC_month']):
        raise ValueError('Actual native exit outside original unit/lease/month bounds')

def _live_unit(identity,batch=None):
    units,_,_,kernel=_api()
    observed=units._observe()
    if observed['boot_id']!=identity['boot_id']:raise ValueError('Original actual native boot lost')
    if (Fraction(observed['monotonic'])>=Fraction(identity['unit_start_monotonic_us'],10**6)+7200
        or Fraction(observed['wall'])>=Fraction(identity['unit_start_realtime_us'],10**6)+7200):
        raise TimeoutError('Original unit-start absolute7200 reached')
    if datetime.fromtimestamp(observed['wall'],timezone.utc).strftime('%Y-%m')!=identity['batch']['actual_UTC_month']:
        raise ValueError('Original actual unit crossed its physical month')
    raw=units._unit(identity['unit'],batch)
    units._same(identity,raw)
    if (raw['ActiveState']!='active' or raw['SubState']!='running'
        or not raw.get('ControlGroup')
        or raw['MainPID']!=os.getpid() or raw['ExecMainPID']!=os.getpid()
        or os.environ.get('INVOCATION_ID')!=identity['invocation_id']
        or kernel._proc(os.getpid())['start_ticks']!=identity['parent_start_ticks']
        or kernel._group(os.getpid())!=Path(identity['owned_group']['path'])):
        raise ValueError('Original actual live main process unavailable')
    if batch is not None:batch.clock.assert_parent_alive()
    after=units._observe()
    if (after['boot_id']!=identity['boot_id'] or after['monotonic']<observed['monotonic'] or after['wall']<observed['wall']
        or Fraction(after['monotonic'])>=Fraction(identity['unit_start_monotonic_us'],10**6)+7200
        or Fraction(after['wall'])>=Fraction(identity['unit_start_realtime_us'],10**6)+7200
        or datetime.fromtimestamp(after['wall'],timezone.utc).strftime('%Y-%m')!=identity['batch']['actual_UTC_month']):
        raise TimeoutError('Original actual unit expired/changed during live verification')
    return raw

def _quiescence(identity,batch):
    units,_,_,kernel=_api()
    group=kernel._group_path(Path(identity['owned_group']['path']))
    if group.stat().st_ino!=identity['owned_group']['inode']:raise ValueError('Actual owned group changed before return')
    tree=kernel._subtree(group)
    if tree['members']!=[identity['exec_main_pid']]:raise ValueError('Actual worker descendants remain before natural exit')
    observed=batch.clock.observe()
    if (not batch.clock.evaluate(observed).allowed or observed['boot_id']!=identity['boot_id']
        or Fraction(observed['monotonic'])>=Fraction(identity['unit_start_monotonic_us'],10**6)+7200
        or Fraction(observed['wall'])>=Fraction(identity['unit_start_realtime_us'],10**6)+7200):
        raise TimeoutError('Original actual unit deadline reached during quiescence read')
    return {'owned_group':identity['owned_group'],'subtree':tree,'observed':observed,
            'original_main_pid':identity['exec_main_pid'],'cleanup_signals':False}

def _released_group(identity):
    """OS release is admissible ONLY beside the actual same natural-zero unit."""
    units,_,_,kernel=_api()
    group=Path(identity['owned_group']['path']);root=Path('/sys/fs/cgroup')
    if not group.is_absolute()or not group.is_relative_to(root)or group.name!=identity['unit']or '..'in group.parts:
        raise ValueError('Original released group path identity invalid')
    for parent in group.parents:
        if parent==root.parent:break
        if parent.is_symlink():raise ValueError('Released group ancestor alias refused')
    try:group.lstat()
    except FileNotFoundError:pass
    else:raise ValueError('Empty ControlGroup with existing/replaced original group refused')
    shared=kernel._group_path(Path(identity['shared_group']['path']))
    if shared.stat().st_ino!=identity['shared_group']['inode']or units._shared(group)!=identity['shared_group']:
        raise ValueError('Original shared aggregate identity/limits changed after OS release')

def _historical_same(identity,raw):
    units,_,_,_=_api()
    if raw.get('ControlGroup')!='':
        units._same(identity,raw)
        return False
    # This branch cannot authorize running/failed/restarted/replaced units.
    if (raw['ActiveState']!='active'or raw['SubState']!='exited'or raw['MainPID']!=0
        or raw['Id']!=identity['unit']
        or raw['Result']!='success'or raw['ExecMainCode']!=1 or raw['ExecMainStatus']!=0
        or units._invocation(raw)!=identity['invocation_id']or raw['object_path']!=identity['object_path']
        or raw['ExecMainPID']!=identity['exec_main_pid']
        or raw['ExecMainStartTimestampMonotonic']!=identity['unit_start_monotonic_us']
        or raw['ExecMainStartTimestamp']!=identity['unit_start_realtime_us']):
        raise ValueError('Only the same retained original natural-zero unit can release its group')
    units._configuration(raw,identity['command']);_released_group(identity)
    return True

def _bootstrap(directory,cell,index,batch_index):
    """Bounded header is design identity only, not full law/registry truth."""
    units,actual,_,kernel=_api()
    if threading.current_thread()is not threading.main_thread():raise ValueError('Original actual worker main thread required')
    directory=_lexical(directory,cell,index,batch_index)
    header=actual._read(directory/'identity.json')
    if header['directory']!=str(directory):raise ValueError('Original actual registry directory changed')
    files=_installed(header)
    plan=actual._acceptance(header['acceptance'])
    ordinal=next((j for j,c in enumerate(plan.cells) if c.name==cell),None)
    if ordinal is None or index>=plan.cells[ordinal].repetitions:raise ValueError('Coordinate outside full registered N')
    coordinate={'registry':fingerprint(header),'cell':cell,'cell_ordinal':ordinal,'index':index,
                'scenario_digest':plan.cells[ordinal].scenario_digest}
    if header['actual_observation']!='registered_nominal_simulation' or len(header['work_batches'])!=2:
        raise ValueError('Original registered nominal two-physical-month simulation only')
    batch=header['work_batches'][batch_index]
    if batch['index']!=batch_index or batch['service_seconds_ceiling']!=7200:raise ValueError('Original7200 batch required')
    command=trial_command(directory,cell,index,batch_index)
    raw=units._unit(_name(coordinate,batch_index));units._configuration(raw,command)
    if any(k in os.environ for k in units.UNSET_ENVIRONMENT):raise ValueError('Inherited injection environment refused')
    if raw['MainPID']!=os.getpid() or raw['ExecMainPID']!=os.getpid():raise ValueError('Original actual worker must be current main PID')
    owned=units._group(raw);observed=units._observe();proc=kernel._proc(os.getpid())
    from .evidence import file_hash
    value={'version':VERSION,'coordinate':coordinate,'registry_directory':str(directory),
        'batch_index':batch_index,'batch':batch,'source_files':files,'boot_id':observed['boot_id'],
        'unit':raw['Id'],'object_path':raw['object_path'],'invocation_id':units._invocation(raw),
        'exec_main_pid':os.getpid(),'parent_start_ticks':proc['start_ticks'],
        'parent_created':psutil.Process(os.getpid()).create_time(),
        'unit_start_monotonic_us':raw['ExecMainStartTimestampMonotonic'],
        'unit_start_realtime_us':raw['ExecMainStartTimestamp'],'command':command,
        'owned_group':owned,'shared_group':units._shared(Path(owned['path'])),
        'executable_sha256':file_hash(sys.executable),'busctl_sha256':file_hash(units.BUSCTL)}
    _live_unit(value)
    return value

class _Run:
    def __init__(self,value):
        self.owner=(os.getpid(),threading.get_ident());self.bootstrap=_plain(value)
        self.binding=None;self.started=False
    def check(self):
        if self.owner!=(os.getpid(),threading.get_ident()) or _RUN.get()is not self:
            raise ValueError('Original actual worker invocation required')

class LiveTrialBatchBinding:
    def __init__(self,*a,**k):raise TypeError('Original controller admission only')
    def __reduce__(self):raise TypeError('Actual unit binding cannot serialize')
    @property
    def identity(self):return _plain(self._identity)
    def check_live(self):
        self._run.check();self._batch.clock.assert_parent_alive()
        _check_attempt(self._run.bootstrap)
        roots=(self._batch.directory,self._batch.mirror);_failed(self._batch.trial,roots)
        if _mirrors(roots,'native-unit.json')!=self._identity:raise ValueError('Original actual unit binding changed')
        if _installed(self._batch.trial.registry.identity)!=self._identity['source_files']:raise ValueError('Original actual source changed')
        return _live_unit(self._identity,self._batch)
    def publish_completed(self,terminal,successor):
        self.check_live();roots=(self._batch.directory,self._batch.mirror)
        if _mirrors(roots,'terminal.json')!=terminal or _mirrors(roots,'successor.json')!=successor:
            raise ValueError('Original actual controller completion differs')
        value={**self._identity,'controller_terminal_hash':fingerprint(terminal),
            'controller_successor_hash':fingerprint(successor),'provisional_only':True}
        controller=_api()[2]
        for p in roots:controller._write(p/'native-completion.json',value)
        if _mirrors(roots,'native-completion.json')!=value:raise ValueError('Native completion mirrors differ')
        self.check_live()
        return value

def bind_current_batch(batch):
    units,_,controller,_=_api()
    if type(batch)is not controller._PhysicalBatch:raise TypeError('Exact original physical batch required')
    run=_RUN.get()
    if type(run)is not _Run:raise ValueError('Fixed native actual worker entry required')
    run.check()
    if run.binding is not None:raise ValueError('Actual batch invocation already bound')
    value=run.bootstrap
    if (batch.trial.coordinate!=value['coordinate'] or batch.plan!=value['batch']
        or batch.trial.registry.identity['source_files']!=value['source_files']):
        raise ValueError('Actual worker coordinate/source/batch differs')
    issuer=batch.clock.issuer
    if (issuer['parent_pid']!=os.getpid() or issuer['parent_created']!=value['parent_created']
        or issuer['boot_id']!=value['boot_id'] or Fraction(issuer['monotonic_started'])<Fraction(value['unit_start_monotonic_us'],10**6)):
        raise ValueError('Original actual clock/parent differs from live unit')
    roots=(batch.directory,batch.mirror)
    identity={**value,'clock':batch.clock.to_dict(),'controller_roots':[str(p)for p in roots],
        'controller_binding_hash':fingerprint(batch.binding)}
    if any((p/'native-unit.json').exists()for p in roots):raise ValueError('Actual native unit binding already consumed')
    for p in roots:controller._write(p/'native-unit.json',identity)
    bound=object.__new__(LiveTrialBatchBinding)
    bound._run,bound._batch,bound._identity=run,batch,_plain(identity)
    run.binding=bound;bound.check_live()
    return bound

def run_original_batch(directory,cell,index,batch_index):
    """Fixed worker entry. Only the enclosing actual unit's zero exit authorizes."""
    if _RUN.get()is not None:raise ValueError('No nested actual worker invocation')
    value=_bootstrap(directory,cell,index,batch_index)
    if any(p.exists()for p in _attempt_roots(value)[1]):raise ValueError('Actual native attempt already consumed; no retry')
    run=_Run(value);token=_RUN.set(run);trial=None
    try:
        _claim_bootstrap(value)  # Durable once-only admission before full parent truth reads.
        _,actual,controller,_=_api()
        registry=actual.NominalActualRegistry(directory)
        if registry.digest!=value['coordinate']['registry'] or _installed(registry.identity)!=value['source_files']:
            raise ValueError('Full original registry differs from bootstrap design')
        trial=actual.NominalActualTrial(registry,cell,index)
        _live_unit(value)
        run.started=True
        terminal=controller.run_batch(trial,batch_index)
        if type(run.binding)is not LiveTrialBatchBinding:raise ValueError('Controller did not bind original actual native unit')
        roots=_roots(trial,batch_index);completion=_mirrors(roots,'native-completion.json')
        successor=_mirrors(roots,'successor.json')
        result={'version':RETURN_VERSION,'coordinate':trial.coordinate,'batch_index':batch_index,
            'controller_terminal_hash':fingerprint(terminal),'controller_successor_hash':fingerprint(successor),
            'native_binding_hash':fingerprint(run.binding.identity),'provisional_only':True,'cleanup_signals':False}
        if completion!={**run.binding.identity,'controller_terminal_hash':fingerprint(terminal),
                'controller_successor_hash':fingerprint(successor),'provisional_only':True}:
            raise ValueError('Original actual completion closure differs')
        for p in roots:controller._write(p/'native-return.json',result)
        if _mirrors(roots,'native-return.json')!=result:raise ValueError('Original actual return mirrors differ')
        run.binding.check_live()
        quiet={'version':VERSION,'native_binding_hash':fingerprint(run.binding.identity),
            'native_return_hash':fingerprint(result),'observation':_quiescence(run.binding.identity,run.binding._batch),
            'provisional_only':True}
        for p in roots:controller._write(p/'native-quiescence.json',quiet)
        if _mirrors(roots,'native-quiescence.json')!=quiet:raise ValueError('Original quiescence mirrors differ')
        run.binding.check_live()
        _quiescence(run.binding.identity,run.binding._batch)
        return result
    except BaseException as error:
        if trial is not None:_fail(trial,batch_index,error)
        _fail_bootstrap(value,error)
        raise
    finally:_RUN.reset(token)

def _retained(registry,cell,index,batch_index):
    units,actual,controller,kernel=_api()
    if type(registry)is not actual.NominalActualRegistry:raise TypeError('Exact original actual registry required')
    files=_installed(registry.verify());trial=actual.NominalActualTrial(registry,cell,index)
    _lexical(registry.directory,cell,index,batch_index);roots=_roots(trial,batch_index);_failed(trial,roots)
    identity=_mirrors(roots,'native-unit.json');binding=_mirrors(roots,'intent.json')
    batch=registry.identity['work_batches'][batch_index]
    if (identity['version']!=VERSION or identity['coordinate']!=trial.coordinate or identity['batch_index']!=batch_index
        or identity['batch']!=batch or identity['clock']!=binding['clock']
        or identity['controller_binding_hash']!=fingerprint(binding)
        or identity['controller_roots']!=[str(p)for p in roots]
        or identity['source_files']!=files or identity['registry_directory']!=str(registry.directory)
        or identity['unit']!=_name(trial.coordinate,batch_index)
        or identity['command']!=trial_command(registry.directory,cell,index,batch_index)):
        raise ValueError('Original actual-trial unit identity differs')
    issuer=identity['clock']['issuer']
    bootstrap={k:v for k,v in identity.items()if k not in ('clock','controller_roots','controller_binding_hash')}
    _check_attempt(bootstrap)
    if issuer['parent_pid']!=identity['exec_main_pid'] or issuer['parent_created']!=identity['parent_created']:
        raise ValueError('Actual controller parent differs from bound unit')
    from .evidence import file_hash
    if identity['executable_sha256']!=file_hash(sys.executable) or identity['busctl_sha256']!=file_hash(units.BUSCTL):
        raise ValueError('Original actual native executable changed')
    observed=units._observe()
    if observed['boot_id']!=identity['boot_id']:raise ValueError('Original actual boot unavailable')
    raw=units._unit(identity['unit']);released=_historical_same(identity,raw)
    if (raw['ActiveState']!='active' or raw['SubState']!='exited' or raw['MainPID']!=0
        or raw['Result']!='success' or raw['ExecMainCode']!=1 or raw['ExecMainStatus']!=0):
        raise ValueError('Actual original physical-batch unit has no natural-zero exit')
    _exit_deadline(identity,raw)
    if (Fraction(raw['ExecMainExitTimestampMonotonic'],10**6)>Fraction(observed['monotonic'])
        or Fraction(raw['ExecMainExitTimestamp'],10**6)>Fraction(observed['wall'])):
        raise ValueError('Actual native exit appears after its current OS observation')
    if not released:
        subtree=kernel._subtree(Path(identity['owned_group']['path']))
        if subtree['members'] or subtree['populated']:raise ValueError('Actual original unit subtree not empty')
    terminal=controller.verify_completed_batch(trial,batch_index);successor=_mirrors(roots,'successor.json')
    expected={**identity,'controller_terminal_hash':fingerprint(terminal),
        'controller_successor_hash':fingerprint(successor),'provisional_only':True}
    if _mirrors(roots,'native-completion.json')!=expected:raise ValueError('Original actual native completion changed')
    returned={'version':RETURN_VERSION,'coordinate':trial.coordinate,'batch_index':batch_index,
        'controller_terminal_hash':fingerprint(terminal),'controller_successor_hash':fingerprint(successor),
        'native_binding_hash':fingerprint(identity),'provisional_only':True,'cleanup_signals':False}
    if _mirrors(roots,'native-return.json')!=returned:raise ValueError('Original fixed actual worker did not finish publication')
    quiet=_mirrors(roots,'native-quiescence.json');q=quiet['observation']
    from .full_trial_controller import ControllerStepLease
    from types import SimpleNamespace
    clock=ControllerStepLease.from_dict(identity['clock'],SimpleNamespace(service={'wall_seconds':7200}))
    if (set(quiet)!= {'version','native_binding_hash','native_return_hash','observation','provisional_only'}
        or quiet['version']!=VERSION or quiet['native_binding_hash']!=fingerprint(identity)
        or quiet['native_return_hash']!=fingerprint(returned)or quiet['provisional_only']is not True
        or q['owned_group']!=identity['owned_group']or q['original_main_pid']!=identity['exec_main_pid']
        or q['cleanup_signals']is not False or q['subtree']['members']!=[identity['exec_main_pid']]
        or not clock.evaluate(q['observed']).allowed
        or Fraction(q['observed']['monotonic'])>=Fraction(raw['ExecMainExitTimestampMonotonic'],10**6)
        or Fraction(q['observed']['wall'])>=Fraction(raw['ExecMainExitTimestamp'],10**6)):
        raise ValueError('Original actual worker did not prove quiescence before its natural exit')
    _installed(registry.verify());_failed(trial,roots);_check_attempt(bootstrap)
    again=units._unit(identity['unit']);_historical_same(identity,again)
    if again!=raw:raise ValueError('Actual retained invocation changed across completion read')
    return expected,raw,units._observe()

class RetainedTrialBatchObservation:
    def __init__(self,*a,**k):raise TypeError('Use actual retained physical-batch reader; JSON is not authority')
    def __reduce__(self):raise TypeError('Actual retained unit cannot serialize or resume')
    @property
    def identity(self):return _plain(self._identity)
    def assert_current(self):
        if self._key is not _KEY:raise TypeError('Original typed actual retained observation required')
        identity,raw,now=_retained(self._registry,self._cell,self._index,self._batch_index)
        if (identity!=self._identity or raw!=self._raw or now['boot_id']!=self._observed['boot_id']
            or now['monotonic']<self._observed['monotonic'] or now['wall']<self._observed['wall']):
            raise ValueError('Original actual retained invocation changed')
        return self
def read_retained_batch(registry,cell,index,batch_index):
    identity,raw,observed=_retained(registry,cell,index,batch_index)
    result=object.__new__(RetainedTrialBatchObservation)
    result._key,result._registry,result._cell,result._index,result._batch_index=_KEY,registry,cell,index,batch_index
    result._identity,result._raw,result._observed=_plain(identity),_plain(raw),observed
    return result
