"""Fixed prospective ten-second draw service and retained native observation.

No lease is issued here. The sole original draw lease and earlier OS ten-second
deadline both apply. Native completion is execution evidence, never five-claim
truth or statistical acceptance. There is no retry, refund or new fit wallet.
"""
from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import importlib
import json
import os
import sys
import threading

from factorlab.protocol import fingerprint, project_path

RELATIVE='factorlab/discovery/nominal_draw_native.py'
WORKER='factorlab.discovery.nominal_draw_worker'
VERSION='nominal-draw-native-v1'
ACTUAL_VERSION='v7-nominal-actual-trial-v5-native-draw'
_KEY=object()
_ACTIVE=ContextVar('original_native_draw_owner',default=None)

def _api():
    from . import nominal_actual_trial as original
    from . import nominal_systemd_unit as units
    from . import native_verification_terminal as kernel
    return original,units,kernel
def _plain(v):return json.loads(json.dumps(v,sort_keys=True,allow_nan=False))
def _pair(roots,name):
    original,_,_=_api();a,b=(original._read(p/name)for p in roots)
    if a!=b:raise ValueError('Original native draw mirrors differ: '+name)
    return a
def _write_pair(roots,name,value):
    original,_,_=_api()
    for p in roots:original._write(p/name,value)
    if _pair(roots,name)!=value:raise ValueError('Original native publication changed')
def _installed():
    from .monthly_work_authority import source_files
    from .evidence import file_hash
    files=source_files()
    if Path(__file__).resolve()!=project_path(RELATIVE).resolve() or files.get(RELATIVE)!=file_hash(__file__):
        raise ValueError('Fresh canonical native draw source required')
    for name in ('nominal_actual_trial','nominal_draw_worker','nominal_draw_diagnostic','nominal_draw_measurement',
                 'nominal_systemd_unit','native_verification_terminal','work_lease','rational_sampling'):
        module=importlib.import_module('factorlab.discovery.'+name);rel='factorlab/discovery/'+name+'.py'
        if Path(module.__file__).resolve()!=project_path(rel).resolve() or files.get(rel)!=file_hash(module.__file__):
            raise ValueError('Original loaded native draw dependency differs')
    return files
def _header(directory):
    original,_,_=_api()
    registry=object.__new__(original.NominalActualRegistry)
    registry.directory=project_path(directory)
    registry.identity=original._read(registry.directory/'identity.json');registry.digest=fingerprint(registry.identity)
    if original.VERSION!=ACTUAL_VERSION or registry.identity['version']!=ACTUAL_VERSION:
        raise ValueError('New prospective native draw registry required')
    return registry

class NativeDrawPlan:
    @classmethod
    def freeze(cls,registry):
        from filelock import FileLock
        original,units,_=_api()
        from .evidence import file_hash
        if cls is not NativeDrawPlan or type(registry)is not original.NominalActualRegistry:raise TypeError('Exact original registry required')
        if sys.platform!='linux':raise OSError('Register actual Linux source/executables prospectively')
        registry.verify();files=_installed()
        if files!=registry.identity['source_files']:raise ValueError('Original whole source differs')
        from .nominal_verification_budget import VerificationRegistry
        quota=VerificationRegistry(registry.identity['grid']['verification_registry_directory'])
        if quota.digest!=registry.identity['grid']['verification_registry_digest']:raise ValueError('Original paid law quota changed')
        authority=units.read_retained_service(quota,quota.plan.task(130)['service'])
        if type(authority)is not units.RetainedServiceObservation:raise TypeError('Original typed retained cold-law service required')
        shared=authority.identity['shared_group'];authority.assert_current()
        directory=registry.directory/'native-draw'
        with FileLock(str(registry.directory/'.native-draw-registration.lock'),timeout=0):
            if any((registry.directory/n).exists()for n in ('trials','draw_allocations','native-draw','native-draw-plan.json')):
                raise ValueError('Prior allocation/native plan cannot be reset')
            directory.mkdir(exist_ok=False)
            (directory/'start-claims').mkdir(exist_ok=False)
            (registry.directory/'native-draw-start-claims').mkdir(exist_ok=False)
            value={'version':VERSION,'registry_directory':str(registry.directory),'registry_digest':registry.digest,
                'source_files':files,'acceptance_digest':registry.identity['acceptance_digest'],
                'registered_N':registry.identity['registered_N'],'original_seconds':10,
                'worker':WORKER,'python':sys.executable,'python_sha256':file_hash(sys.executable),
                'busctl_sha256':file_hash(units.BUSCTL),'shared_slice':units.SHARED_SLICE,
                'shared_group':shared,'original_shared_service_binding_hash':fingerprint(authority.identity),
                'per_coordinate_attempts':1,'retry_allowed':False,'extra_fit_GPT_credit':False,
                'failed_coordinate':{'retain_N':True,'family_error':1,'power':0},'formal_acceptance':False}
            _write_pair((directory,registry.directory),'native-draw-plan.json',value)
        return cls.inspect(registry)
    @classmethod
    def inspect(cls,registry):
        original,_,_=_api()
        if cls is not NativeDrawPlan or type(registry)is not original.NominalActualRegistry:raise TypeError('Exact original registry required')
        obj=object.__new__(cls);obj.registry=registry;obj.directory=registry.directory/'native-draw'
        obj.identity=_pair((obj.directory,registry.directory),'native-draw-plan.json');obj.digest=fingerprint(obj.identity)
        obj.verify();return obj
    def verify(self):
        original,units,kernel=_api();from .evidence import file_hash
        p=self.identity;r=self.registry
        if (type(self)is not NativeDrawPlan or type(r)is not original.NominalActualRegistry
            or self.directory!=r.directory/'native-draw' or r.identity['directory']!=str(r.directory)
            or p!=_pair((self.directory,r.directory),'native-draw-plan.json') or self.digest!=fingerprint(p)
            or r.identity!=original._read(r.directory/'identity.json') or fingerprint(r.identity)!=r.digest
            or p['registry_directory']!=str(r.directory) or p['registry_digest']!=r.digest
            or r.identity['version']!=ACTUAL_VERSION or original.VERSION!=ACTUAL_VERSION
            or p['source_files']!=r.identity['source_files'] or p['source_files']!=_installed()
            or p['acceptance_digest']!=r.identity['acceptance_digest'] or p['registered_N']!=r.identity['registered_N']
            or p['version']!=VERSION or p['original_seconds']!=10 or type(p['original_seconds'])is not int
            or p['worker']!=WORKER or p['python']!=sys.executable or p['python_sha256']!=file_hash(sys.executable)
            or p['busctl_sha256']!=file_hash(units.BUSCTL) or p['shared_slice']!=units.SHARED_SLICE
            or p['per_coordinate_attempts']!=1 or p['retry_allowed']is not False or p['extra_fit_GPT_credit']is not False
            or p['failed_coordinate']!={'retain_N':True,'family_error':1,'power':0} or p['formal_acceptance']is not False):
            raise ValueError('Original prospective native draw descriptor changed')
        shared=kernel._group_path(Path(p['shared_group']['path']))
        if (shared.name!=units.SHARED_SLICE or shared.stat().st_ino!=p['shared_group']['inode']
            or not kernel._limits(kernel._values(shared))):raise ValueError('Original shared aggregate inode/limits changed')
        return _plain(p)
    def coordinate(self,cell,index):return self.registry._allocation_header(cell,index)[0]
    def roots(self,coordinate):
        c,paths=self.registry._allocation_header(coordinate['cell'],coordinate['index'])
        if coordinate!=c:raise ValueError('Original registered coordinate required')
        key=fingerprint(c)
        return (self.directory/'attempts'/key,self.registry.directory/'native-draw-attempts'/key)

def unit_name(plan,c):return 'factorresearch-nd-'+fingerprint([plan.digest,c])+'.service'
def command(plan,c):
    plan.roots(c)
    return [sys.executable,'-B','-m',WORKER,'--registry',str(plan.registry.directory),'--cell',c['cell'],'--index',str(c['index'])]
def _quote(value):
    if type(value)is not str or any(ch in value for ch in '\n\r\x00'):raise ValueError('One bounded unit argument required')
    return '"'+value.replace('\\','\\\\').replace('"','\\"').replace('%','%%').replace('$','$$')+'"'
def unit_text(plan,c):
    plan.verify();_,units,_=_api()
    lines=['[Unit]','Description=Original once-only ten-second nominal draw',
        '[Service]','Type=exec','ExitType=main','RemainAfterExit=yes','Restart=no','KillMode=control-group',
        'Delegate=yes','OOMPolicy=kill','RuntimeMaxSec=10','TimeoutStopSec=1','Slice='+units.SHARED_SLICE,
        'WorkingDirectory='+_quote(str(project_path('.'))),'ExecStart='+' '.join(_quote(x)for x in command(plan,c))]
    lines+=['Environment='+_quote(k+'='+v)for k,v in units.unit_environment().items()]
    lines+=['UnsetEnvironment='+' '.join(units.UNSET_ENVIRONMENT)]
    return '\n'.join(lines)+'\n'
def prepare_unit(plan,cell,index):
    """Durable once-only descriptor. Return offline unit text; do not start it."""
    from filelock import FileLock
    if type(plan)is not NativeDrawPlan:raise TypeError('Exact fixed draw plan required')
    plan.verify();c=plan.coordinate(cell,index);roots=plan.roots(c)
    with FileLock(str(roots[0])+'.lock',timeout=0):
        if any(p.exists()for p in roots):raise ValueError('Coordinate consumed; never prepare again')
        for p in roots:p.mkdir(parents=True,exist_ok=False)
        value={'version':VERSION,'plan_digest':plan.digest,'coordinate':c,'command':command(plan,c),
            'unit':unit_name(plan,c),'unit_text_hash':fingerprint(unit_text(plan,c)),
            'source_files':plan.identity['source_files'],'retain_N':True,'retry_allowed':False}
        _write_pair(roots,'intent.json',value)
    return {'unit':value['unit'],'text':unit_text(plan,c),'coordinate':c,'provisional_only':True}
def launch_once(plan,cell,index):
    """Fixed StartUnit only after durable consumed launch marker. No retry."""
    from filelock import FileLock
    if sys.platform!='linux' or type(plan)is not NativeDrawPlan:raise TypeError('Actual Linux fixed plan required')
    plan.verify();c=plan.coordinate(cell,index);roots=plan.roots(c);intent=_pair(roots,'intent.json')
    if intent!=_intent(plan,c):raise ValueError('Original prepared unit descriptor differs')
    # Review/install the exact unit file separately; this API cannot install or
    # replace arbitrary service configuration. Worker validates loaded properties.
    path=Path.home()/'.config/systemd/user'/unit_name(plan,c)
    if path.is_symlink() or path.resolve()!=path or path.read_text('utf8')!=unit_text(plan,c):raise ValueError('Install the exact reviewed unit first')
    with FileLock(str(roots[0])+'.launch.lock',timeout=0):
        if any((p/'launch.json').exists()for p in roots):raise ValueError('Original launch consumed; never repeat')
        _write_pair(roots,'launch.json',{'version':VERSION,'intent_hash':fingerprint(intent),'retained_N':True,'retry_allowed':False})
        _,units,_=_api()
        return units._message(units._call(['/org/freedesktop/systemd1','org.freedesktop.systemd1.Manager',
            'StartUnit','ss',unit_name(plan,c),'fail']),'o')
def _intent(plan,c):
    return {'version':VERSION,'plan_digest':plan.digest,'coordinate':c,'command':command(plan,c),
        'unit':unit_name(plan,c),'unit_text_hash':fingerprint(unit_text(plan,c)),
        'source_files':plan.identity['source_files'],'retain_N':True,'retry_allowed':False}
def _view(raw):
    if type(raw.get('RuntimeMaxUSec'))is not int or raw['RuntimeMaxUSec']!=10*10**6:
        raise ValueError('Actual original OS ten-second ceiling required')
    return {**raw,'RuntimeMaxUSec':7200*10**6}
def _deadline(identity,raw=None,observed=None):
    lease=identity['lease'];sm=Fraction(identity['unit_start_monotonic_us'],10**6);sw=Fraction(identity['unit_start_realtime_us'],10**6)
    lm=Fraction(lease['monotonic_started']);lw=Fraction(datetime.fromisoformat(lease['wall_started_at']).timestamp())
    dm=min(sm+10,Fraction(lease['monotonic_deadline']));dw=min(sw+10,Fraction(datetime.fromisoformat(lease['wall_deadline']).timestamp()))
    if lease['lease_seconds']!=10 or not 0<sm<=lm<dm or not 0<sw<=lw<dw:raise ValueError('Original lease must begin within original OS runtime')
    if raw is not None:
        em=Fraction(raw['ExecMainExitTimestampMonotonic'],10**6);ew=Fraction(raw['ExecMainExitTimestamp'],10**6)
        if not lm<em<dm or not lw<ew<dw:raise ValueError('Natural exit missed original earlier ten-second bound')
    if observed is not None:
        if observed['boot_id']!=identity['boot_id'] or not lm<=Fraction(observed['monotonic'])<dm or not lw<=Fraction(observed['wall'])<dw:
            raise TimeoutError('Original earlier ten-second draw deadline expired')
def _live(identity,lease):
    _,units,kernel=_api();lease.assert_parent_alive();_deadline(identity,observed=units._observe())
    raw=units._unit(identity['unit'],SimpleNamespace(clock=lease));units._same(identity,_view(raw))
    if (raw['ActiveState']!='active' or raw['SubState']!='running' or raw['MainPID']!=os.getpid()
        or raw['ExecMainPID']!=os.getpid() or os.environ.get('INVOCATION_ID')!=identity['invocation_id']
        or kernel._proc(os.getpid())['start_ticks']!=identity['parent_start_ticks']
        or kernel._group(os.getpid())!=Path(identity['owned_group']['path'])):raise ValueError('Original live draw owner changed')
    lease.assert_parent_alive();_deadline(identity,observed=units._observe());return raw
def _fail(roots,error):
    original,_,_=_api()
    for p in roots:
        try:
            if not(p/'failure.json').exists():original._write(p/'failure.json',{'state':'failed_or_unknown_consumed',
                'retain_N':True,'family_error':1,'power':0,'retry_allowed':False,'reason':type(error).__name__})
        except BaseException as secondary:error.add_note('Native failure marker unavailable: '+type(secondary).__name__)
def _no_failure(roots,paths,mirror):
    if any((p/'failure.json').exists()for p in (*roots,paths['root'],mirror)):raise ValueError('Original failure remains consumed')
def _claims(registry,cell,index):
    from .nominal_draw_worker import claim_paths
    from .calibration_trial_context import _no_redirect
    for path in claim_paths(str(registry.directory),cell,index):
        _no_redirect(path,project_path('.'))
        if (path.is_symlink() or not path.is_file() or path.stat().st_size!=len(b'original-draw-started-once\n')
            or path.read_bytes()!=b'original-draw-started-once\n'):raise ValueError('Original exclusive worker claim required')
def _publication(paths,mirror):
    original,_,_=_api();names=(('draw-intent.json','intent.json'),('draw-result.json','result.json'),('draw-seal.json','seal.json'))
    values={}
    for left,right in names:
        a=original._read(paths['root']/left)
        if a!=original._read(mirror/right):raise ValueError('Original draw publication mirrors differ')
        values[left]=a
    from .calibration_trial_context import _no_redirect
    from .evidence import file_hash
    for name in ('allocation.json','events.jsonl','terminal.json'):
        path=paths['entropy']/name;_no_redirect(path,project_path('.'))
        if path.is_symlink() or not path.is_file() or path.stat().st_size>16*2**20:raise ValueError('Original bounded entropy publication required')
        values['entropy/'+name]={'sha256':file_hash(path),'bytes':path.stat().st_size}
    return values

class _Owner:
    def __init__(self,key,registry,cell,index):
        if key is not _KEY:raise TypeError('Fixed worker only')
        self.registry=registry;self.cell=cell;self.index=index;self.owner=(os.getpid(),threading.get_ident())
        self.identity=None;self.lease=None;self.active=True
    def check(self):
        if not self.active or self.owner!=(os.getpid(),threading.get_ident()):raise ValueError('Original draw owner context required')

@contextmanager
def native_draw_scope(registry,intent,paths,mirror,lease):
    """Called by original draw AFTER its only lease and both durable intents."""
    from .work_lease import FrozenWorkLease,_observe
    from .evidence import file_hash
    ctx=_ACTIVE.get()
    if type(ctx)is not _Owner or ctx.registry is not registry:raise ValueError('Use fixed native draw worker')
    ctx.check()
    if type(lease)is not FrozenWorkLease or lease.lease_seconds!=10 or ctx.lease is not None:raise ValueError('One original draw lease only')
    plan=NativeDrawPlan.inspect(registry);c=plan.coordinate(ctx.cell,ctx.index);roots=plan.roots(c)
    original,units,kernel=_api()
    try:
        lease.assert_parent_alive()
        if (intent['coordinate']!=c or intent['lease']!=lease.to_dict()
            or _pair(roots,'intent.json')!=_intent(plan,c)
            or _pair(roots,'launch.json')!={'version':VERSION,'intent_hash':fingerprint(_intent(plan,c)),'retained_N':True,'retry_allowed':False}
            or original._read(paths['root']/'draw-intent.json')!=intent or original._read(mirror/'intent.json')!=intent):
            raise ValueError('Actual original allocation/launch must precede native metadata')
        _no_failure(roots,paths,mirror)
        _claims(registry,ctx.cell,ctx.index)
        raw=units._unit(unit_name(plan,c),SimpleNamespace(clock=lease));units._configuration(_view(raw),command(plan,c))
        if raw['MainPID']!=os.getpid() or raw['ExecMainPID']!=os.getpid():raise ValueError('Actual systemd main draw PID required')
        if any(k in os.environ for k in units.UNSET_ENVIRONMENT):raise ValueError('Source injection environment refused')
        owned=units._group(raw);observed=units._observe();proc=kernel._proc(os.getpid())
        if units._shared(Path(owned['path']))!=plan.identity['shared_group']:raise ValueError('Draw left the original shared aggregate')
        identity={'version':VERSION,'plan_digest':plan.digest,'coordinate':c,'source_files':plan.identity['source_files'],
            'unit':raw['Id'],'object_path':raw['object_path'],'invocation_id':units._invocation(raw),
            'exec_main_pid':os.getpid(),'parent_start_ticks':proc['start_ticks'],'parent_created':lease.parent_created,
            'boot_id':observed['boot_id'],'unit_start_monotonic_us':raw['ExecMainStartTimestampMonotonic'],
            'unit_start_realtime_us':raw['ExecMainStartTimestamp'],'command':command(plan,c),
            'owned_group':owned,'shared_group':units._shared(Path(owned['path'])),
            'executable_sha256':file_hash(sys.executable),'busctl_sha256':file_hash(units.BUSCTL),
            'lease':lease.to_dict(),'intent_hash':fingerprint(intent)}
        if lease.parent_pid!=os.getpid() or lease.boot_id!=identity['boot_id']:raise ValueError('Actual original lease issuer differs')
        _live(identity,lease);_write_pair(roots,'native-unit.json',identity)
        ctx.identity=identity;ctx.lease=lease
        yield
        _live(identity,lease);_no_failure(roots,paths,mirror)
        publication=_publication(paths,mirror)
        diagnostic_hash=None
        try:
            diagnostic=original._read(paths['root']/'draw-diagnostic.json')
            if diagnostic==original._read(mirror/'diagnostic.json'):diagnostic_hash=fingerprint(diagnostic)
        except Exception:
            pass  # Optional measurement is never a scientific permission.
        value={'version':VERSION,'native_binding_hash':fingerprint(identity),'publication':{k:fingerprint(v)for k,v in publication.items()},
            'observed':_observe(lease),'diagnostic_hash':diagnostic_hash,'provisional_only':True,'native_completed':False,'formal_acceptance':False}
        _write_pair(roots,'return.json',value)
        tree=kernel._subtree(Path(owned['path']));units._quiescent_tree(identity,tree)
        quiet={'version':VERSION,'native_binding_hash':fingerprint(identity),'return_hash':fingerprint(value),
            'tree':tree,'observed':_observe(lease),'cleanup_signals':False}
        _write_pair(roots,'quiescence.json',quiet)
        _live(identity,lease);units._quiescent_tree(identity,kernel._subtree(Path(owned['path'])))
        plan.verify()
        if _publication(paths,mirror)!=publication:raise ValueError('Original draw changed during native closure')
        lease.assert_parent_alive();_deadline(identity,observed=units._observe())
    except BaseException as error:
        _fail(roots,error);raise

def require_native_draw(trial):
    """Internal provisional read only in exact worker; external reads need OS."""
    original,_,_=_api()
    if type(trial)is not original.NominalActualTrial:raise TypeError('Exact original trial required')
    ctx=_ACTIVE.get()
    if ctx is not None:
        if type(ctx)is not _Owner or ctx.registry is not trial.registry or ctx.cell!=trial.cell or ctx.index!=trial.index:
            raise ValueError('Foreign native draw context')
        ctx.check()
        if ctx.identity is None or ctx.lease is None:raise ValueError('Actual native metadata not yet bound')
        ctx.lease.assert_parent_alive();_deadline(ctx.identity,observed=_api()[1]._observe())
        return  # Internal construction only. No public native-success object.
    return read_retained_draw(NativeDrawPlan.inspect(trial.registry),trial.cell,trial.index)

def run_worker(directory,cell,index):
    """Fixed CLI body; no provider, fit, arbitrary command or second lease."""
    if sys.platform!='linux' or threading.current_thread()is not threading.main_thread() or _ACTIVE.get()is not None:
        raise OSError('One original Linux main-thread draw worker required')
    registry=_header(directory);ctx=_Owner(_KEY,registry,cell,index);token=_ACTIVE.set(ctx)
    try:
        registry.draw_trial(cell,index)
        if ctx.identity is None or ctx.lease is None:raise ValueError('Original native scope never completed')
        ctx.lease.assert_parent_alive();_deadline(ctx.identity,observed=_api()[1]._observe())
        return 0  # Still provisional; only retained OS reader can grant native completion.
    finally:ctx.active=False;_ACTIVE.reset(token)

class NativeDrawObservation:
    def __init__(self,*a,**k):raise TypeError('Use actual retained native draw reader')
    def __reduce__(self):raise TypeError('Native observation cannot serialize as authority')
    @property
    def identity(self):return _plain(self._identity)
    def assert_current(self):
        if self._key is not _KEY:raise TypeError('Original native observation required')
        fresh=read_retained_draw(self._plan,self._cell,self._index)
        if fresh._identity!=self._identity or fresh._raw!=self._raw:raise ValueError('Original retained draw changed')
        return self
    def diagnostic_snapshot(self):
        """Optional counters bound by original native return; no new authority."""
        from .nominal_draw_diagnostic import read_diagnostic
        self.assert_current()
        if self._identity['diagnostic_hash']is None:raise ValueError('Original optional diagnostic was unavailable')
        packet=read_diagnostic(self._plan.registry,self._cell,self._index)
        if fingerprint(packet)!=self._identity['diagnostic_hash']:raise ValueError('Original measured counter publication changed')
        self.assert_current()
        return {'native_observation':self.identity,'diagnostic':packet,'diagnostic_only':True,'formal_acceptance':False}

def read_retained_draw(plan,cell,index):
    """Authenticate real same-invocation natural exit; no mtime or JSON grants."""
    from .work_lease import FrozenWorkLease,_evaluate
    from .evidence import file_hash
    if type(plan)is not NativeDrawPlan or sys.platform!='linux':raise TypeError('Exact original Linux native draw plan required')
    plan.verify();c,paths=plan.registry._allocation_header(cell,index);roots=plan.roots(c)
    original,units,kernel=_api();mirror=plan.registry.directory/'draw_allocations'/fingerprint(c)
    _no_failure(roots,paths,mirror);_claims(plan.registry,cell,index)
    records={n:_pair(roots,n)for n in ('intent.json','launch.json','native-unit.json','return.json','quiescence.json')}
    identity=records['native-unit.json'];result=records['return.json'];quiet=records['quiescence.json'];publication=_publication(paths,mirror)
    lease=FrozenWorkLease.from_dict(identity['lease'])
    if (identity['version']!=VERSION or records['intent.json']!=_intent(plan,c)
        or records['launch.json']!={'version':VERSION,'intent_hash':fingerprint(records['intent.json']),'retained_N':True,'retry_allowed':False}
        or identity['plan_digest']!=plan.digest or identity['coordinate']!=c or identity['source_files']!=plan.identity['source_files']
        or identity['command']!=command(plan,c) or identity['unit']!=unit_name(plan,c)
        or identity['shared_group']!=plan.identity['shared_group']
        or identity['lease']!=publication['draw-intent.json']['lease'] or identity['intent_hash']!=fingerprint(publication['draw-intent.json'])
        or lease.parent_pid!=identity['exec_main_pid'] or lease.parent_created!=identity['parent_created'] or lease.boot_id!=identity['boot_id']
        or result['version']!=VERSION or result['native_binding_hash']!=fingerprint(identity)
        or result['publication']!={k:fingerprint(v)for k,v in publication.items()}
        or result['provisional_only']is not True or result['native_completed']is not False or result['formal_acceptance']is not False
        or quiet['version']!=VERSION or quiet['native_binding_hash']!=fingerprint(identity)
        or quiet['return_hash']!=fingerprint(result) or quiet['cleanup_signals']is not False
        or not _evaluate(lease,result['observed']).allowed or not _evaluate(lease,quiet['observed'],result['observed']).allowed):
        raise ValueError('Original source/lease/publication closure differs')
    observed=units._observe()
    if observed['boot_id']!=identity['boot_id'] or file_hash(sys.executable)!=identity['executable_sha256'] or file_hash(units.BUSCTL)!=identity['busctl_sha256']:
        raise ValueError('Original boot or executable changed')
    raw=units._unit(identity['unit']);view=_view(raw);units._same_invocation(identity,view)
    _natural(raw);_deadline(identity,raw=raw);units._historical_group(identity,view);units._quiescent_tree(identity,quiet['tree'])
    if not (Fraction(quiet['observed']['monotonic'])<Fraction(raw['ExecMainExitTimestampMonotonic'],10**6)<=Fraction(observed['monotonic'])
            and Fraction(quiet['observed']['wall'])<Fraction(raw['ExecMainExitTimestamp'],10**6)<=Fraction(observed['wall'])):
        raise ValueError('Original quiescence/natural exit observation order differs')
    plan.verify();_no_failure(roots,paths,mirror)
    again=units._unit(identity['unit']);units._same_invocation(identity,_view(again));units._historical_group(identity,_view(again));_natural(again)
    if again!=raw or records!={n:_pair(roots,n)for n in records} or publication!=_publication(paths,mirror):
        raise ValueError('Original source/native/publication changed during read')
    plan.verify();_no_failure(roots,paths,mirror)
    obj=object.__new__(NativeDrawObservation);obj._key=_KEY;obj._plan=plan;obj._cell=cell;obj._index=index;obj._raw=_plain(raw)
    obj._identity={'version':VERSION,'registry_digest':plan.registry.digest,'coordinate':c,'native_binding_hash':fingerprint(identity),
        'publication':result['publication'],'return_hash':fingerprint(result),'diagnostic_hash':result['diagnostic_hash'],
        'unit':identity['unit'],'invocation_id':identity['invocation_id'],'exec_main_pid':identity['exec_main_pid'],
        'boot_id':identity['boot_id'],'lease':identity['lease'],
        'unit_start_monotonic_us':raw['ExecMainStartTimestampMonotonic'],'unit_exit_monotonic_us':raw['ExecMainExitTimestampMonotonic'],
        'unit_start_realtime_us':raw['ExecMainStartTimestamp'],'unit_exit_realtime_us':raw['ExecMainExitTimestamp'],
        'native_elapsed_microseconds':raw['ExecMainExitTimestampMonotonic']-raw['ExecMainStartTimestampMonotonic'],
        'native_completed':True,'original_seconds':10,'formal_acceptance':False}
    return obj
def _natural(raw):
    if (raw['ActiveState']!='active' or raw['SubState']!='exited' or raw['MainPID']!=0 or raw['Result']!='success'
        or raw['ExecMainCode']!=1 or raw['ExecMainStatus']!=0):raise ValueError('Original natural CLD_EXITED/zero status required')
