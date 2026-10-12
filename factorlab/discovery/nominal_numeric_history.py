"""Prospective original fixed-pair execution witness, never a second fit budget.

Only the still-retained original systemd invocation can authenticate historical
numeric failure. This module never reconstructs authority from a JSON receipt,
repeats a solver, creates a model, launches a unit, or renews the original lease.
"""
from __future__ import annotations
from contextvars import ContextVar
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import importlib
import json
import os
import math
import re
import sys
import threading
import weakref
import psutil
from factorlab.protocol import fingerprint, project_path

VERSION='nominal-original-pair-native-v1'
WORKER='factorlab.discovery.nominal_numeric_worker'
RELATIVE='factorlab/discovery/nominal_numeric_history.py'
_KEY=object()
_RUN=ContextVar('original_fixed_pair_native',default=None)
_LIVE=weakref.WeakSet()

def _plain(value):return json.loads(json.dumps(value,sort_keys=True,allow_nan=False))
def _api():
    from . import nominal_systemd_unit as units,native_verification_terminal as native
    return units,native
def _read(path):
    # Original raw/row artifacts have the existing 64 MiB limit; the small
    # verification-quota reader's 1 MiB limit is not their schema.
    from .nominal_cold_conformance import _read as read
    return read(_path(path))
def _write(path,value):
    from .nominal_cold_conformance import _write as write
    return write(_path(path),value)
def _path(path):
    from .calibration_trial_context import _no_redirect
    value=project_path(path);_no_redirect(value,project_path('.'));return value
def _pair(roots,name):
    a,b=(_read(p/name)for p in roots)
    if a!=b:raise ValueError('Original native pair mirrors differ: '+name)
    return a
def _publish(roots,name,value):
    for p in roots:_write(p/name,value)
    if _pair(roots,name)!=value:raise ValueError('Original native pair publication differs')

def descriptor():
    return {'version':VERSION,'worker':WORKER,'unit_seconds':300,'original_pair_seconds':300,
        'quota':'original_prelisted_fixed_replica_only','extra_fit_or_proof_credit':0,
        'attempts_per_cutoff':1,'numeric_reservation':'failed_consumed_not_completed',
        'history_authority':'same_retained_OS_invocation_natural_zero_inside_both_deadlines',
        'legacy_failure':'unknown_full_N','formal_acceptance':False}

def _installed():
    from .monthly_work_authority import source_files
    from .evidence import file_hash
    files=source_files()
    for name in ('nominal_numeric_history','nominal_numeric_worker','nominal_fit_outcome',
                 'nominal_factor_census','convex_eval','nominal_total_loss',
                 'nominal_systemd_unit','native_verification_terminal'):
        module=importlib.import_module('factorlab.discovery.'+name)
        rel='factorlab/discovery/'+name+'.py'
        if Path(module.__file__).resolve()!=project_path(rel).resolve() or files.get(rel)!=file_hash(module.__file__):
            raise ValueError('Canonical original numeric execution source required: '+name)
    return files

def _runtime_identity():
    from .evidence import file_hash
    import numpy
    modules=[importlib.import_module(n)for n in ('numpy._core._multiarray_umath','numpy.linalg._umath_linalg')]
    return {'python':{'path':sys.executable,'sha256':file_hash(sys.executable),'version':sys.version},
        'numpy':numpy.__version__,'kernel_binaries':{m.__name__:{'path':str(Path(m.__file__).resolve()),
            'sha256':file_hash(m.__file__)}for m in modules},'busctl_sha256':file_hash(_api()[0].BUSCTL)}

def command(directory,cutoff):
    directory=_path(directory)
    if type(cutoff)is not str or datetime.fromisoformat(cutoff).isoformat()!=cutoff:
        raise ValueError('Original canonical cutoff required')
    if datetime.fromisoformat(cutoff).utcoffset()!=__import__('datetime').timedelta(0):
        raise ValueError('Original UTC cutoff required')
    return [sys.executable,'-B','-m',WORKER,'--protocol',str(directory),'--cutoff',cutoff]

def unit_name(protocol_digest,cutoff):return 'factorresearch-np-'+fingerprint([protocol_digest,cutoff])+'.service'

def _configuration(raw,expected):
    units,_=_api()
    # Reuse the exact original environment/command/container checks. Only the
    # service duration differs: observed actual RuntimeMax must be 300 first.
    if type(raw.get('RuntimeMaxUSec'))is not int or raw['RuntimeMaxUSec']!=300*10**6:
        raise ValueError('Fixed original pair unit must enforce 300 seconds')
    adapted={**raw,'RuntimeMaxUSec':7200*10**6}
    units._configuration(adapted,expected)

def _same(identity,raw):
    units,_=_api();_configuration(raw,identity['command'])
    if (raw['Id']!=identity['unit'] or units._invocation(raw)!=identity['invocation_id']
        or raw['object_path']!=identity['object_path'] or raw['ExecMainPID']!=identity['exec_main_pid']
        or raw['ExecMainStartTimestampMonotonic']!=identity['unit_start_monotonic_us']
        or raw['ExecMainStartTimestamp']!=identity['unit_start_realtime_us']):
        raise ValueError('Original fixed fit invocation identity differs')

def _unit_bound(identity,observed):
    if (observed['boot_id']!=identity['boot_id']
        or not Fraction(identity['unit_start_monotonic_us'],10**6)<=Fraction(observed['monotonic'])<Fraction(identity['unit_start_monotonic_us'],10**6)+300
        or not Fraction(identity['unit_start_realtime_us'],10**6)<=Fraction(observed['wall'])<Fraction(identity['unit_start_realtime_us'],10**6)+300):
        raise TimeoutError('Original fixed pair unit deadline or boot differs')

def _live(identity,lease=None):
    units,native=_api();_unit_bound(identity,units._observe())
    if lease is not None:lease.assert_parent_alive()
    raw=units._unit(identity['unit'],None if lease is None else SimpleNamespace(clock=lease));_same(identity,raw)
    if (raw['ActiveState']!='active' or raw['SubState']!='running' or raw['MainPID']!=os.getpid()
        or identity['exec_main_pid']!=os.getpid() or os.environ.get('INVOCATION_ID')!=identity['invocation_id']
        or native._proc(os.getpid())['start_ticks']!=identity['parent_start_ticks']
        or native._group(os.getpid())!=Path(identity['owned_group']['path'])
        or units._group(raw)!=identity['owned_group']
        or units._shared(Path(identity['owned_group']['path']))!=identity['shared_group']):
        raise ValueError('Original live fixed-pair owner/group differs')
    _unit_bound(identity,units._observe())
    if lease is not None:lease.assert_parent_alive()
    return raw

def _roots(header,cutoff):
    digest=fingerprint(header);key=fingerprint([digest,cutoff])
    return (_path(header['directory'])/'native-pairs'/key,
        _path(header['authority_directory'])/'nominal_numeric_units'/key)

def _failed(roots,folder=None):
    paths=roots if folder is None else (*roots,folder)
    if any((p/'failure.json').exists()for p in paths):raise ValueError('Original infrastructure failure remains unknown/consumed')

def _fail(roots,error):
    value={'version':VERSION,'state':'infrastructure_unknown','retain_N':True,'error':1,'power':0,
        'retry':False,'reason':type(error).__name__+':'+str(error)[:900]}
    for p in roots:
        try:
            if p.is_dir()and not(p/'failure.json').exists():_write(p/'failure.json',value)
        except BaseException as other:error.add_note('Original failure publication unavailable: '+type(other).__name__)

def _scope(protocol,cutoff):
    from .nominal_factor_census import NominalFactorProtocol
    active=_RUN.get()
    if (type(protocol)is not NominalFactorProtocol or type(active)is not dict
        or active['owner']!=(os.getpid(),threading.get_ident()) or active['protocol']is not protocol
        or active['cutoff']!=cutoff):raise ValueError('Original prospective fixed worker scope required')
    _failed(active['roots']);_live(active['identity'],active['lease']);return active

def require_original_pair(protocol,cutoff):
    from . import nominal_factor_census as producer
    if sys._getframe(1).f_code is not producer._produce_month.__code__:raise ValueError('Original paid producer only')
    active=_scope(protocol,cutoff)
    if active['lease']is not None:raise ValueError('Original pair cannot acquire twice')

def bind_original_lease(protocol,cutoff,lease):
    from . import nominal_factor_census as producer
    from .work_lease import FrozenWorkLease
    if sys._getframe(1).f_code is not producer._produce_month.__code__:raise ValueError('Original paid producer lease only')
    active=_scope(protocol,cutoff)
    if type(lease)is not FrozenWorkLease or lease.lease_seconds!=300 or active['lease']is not None:
        raise ValueError('One actual original 300-second lease required')
    if (lease.parent_pid!=active['identity']['exec_main_pid']or lease.parent_created!=active['identity']['parent_created']
        or lease.boot_id!=active['identity']['boot_id']):raise ValueError('Original fit issuer differs from native owner')
    active['lease']=lease
    _publish(active['roots'],'lease.json',{'identity_hash':fingerprint(active['identity']),'lease':lease.to_dict()})
    _live(active['identity'],lease)

def _bootstrap(directory,cutoff):
    if sys.platform!='linux' or threading.current_thread()is not threading.main_thread():
        raise OSError('Original Linux fixed worker main thread required')
    directory=_path(directory);header=_read(directory/'identity.json')
    if (header['directory']!=str(directory)or header.get('numeric_native')!=descriptor()
        or cutoff not in header['schedule']['cutoffs']):raise ValueError('Prospective original native protocol required')
    files=_installed()
    if files!=header['source_files']:raise ValueError('New whole source registration required')
    units,native=_api();cmd=command(directory,cutoff);raw=units._unit(unit_name(fingerprint(header),cutoff))
    _configuration(raw,cmd)
    if any(k in os.environ for k in units.UNSET_ENVIRONMENT):raise ValueError('Inherited code injection refused')
    if raw['MainPID']!=os.getpid()or raw['ExecMainPID']!=os.getpid():raise ValueError('Original worker main PID required')
    owned=units._group(raw);now=units._observe()
    identity={'version':VERSION,'protocol_digest':fingerprint(header),'protocol_directory':str(directory),'cutoff':cutoff,
        'source_files':files,'unit':raw['Id'],'object_path':raw['object_path'],'invocation_id':units._invocation(raw),
        'exec_main_pid':os.getpid(),'parent_start_ticks':native._proc(os.getpid())['start_ticks'],
        'parent_created':psutil.Process(os.getpid()).create_time(),'boot_id':now['boot_id'],
        'unit_start_monotonic_us':raw['ExecMainStartTimestampMonotonic'],'unit_start_realtime_us':raw['ExecMainStartTimestamp'],
        'command':cmd,'owned_group':owned,'shared_group':units._shared(Path(owned['path'])),'runtime':_runtime_identity()}
    _live(identity);return header,identity

def run_original_pair(directory,cutoff):
    """Fixed worker only. Success return is provisional until real OS zero exit."""
    from . import nominal_numeric_worker as worker
    if sys._getframe(1).f_code is not worker.main.__code__:raise ValueError('Fixed native worker entry required')
    from filelock import FileLock
    if sys.platform!='linux':raise OSError('Original Linux fixed worker required')
    directory=_path(directory);command(directory,cutoff)
    header=_read(directory/'identity.json')
    # Allocation is not truth. Freeze the attempted coordinate before full
    # parent/source/runtime checks, so their failure cannot make it reusable.
    if (header['directory']!=str(directory)or header.get('numeric_native')!=descriptor()
        or cutoff not in header['schedule']['cutoffs']
        or _path(header['mirror'])!=_path(header['authority_directory'])/'nominal_factor_protocol'/'identity.json'
        or _read(header['mirror'])!=header):raise ValueError('Original anchored design coordinate required')
    roots=_roots(header,cutoff)
    roots[1].parent.mkdir(parents=True,exist_ok=True)
    with FileLock(str(roots[1])+'.lock',timeout=0):
        if any(p.exists()for p in roots):raise ValueError('Original native pair attempt already consumed')
        token=None;runtime=None
        try:
            for p in roots:p.mkdir(parents=True,exist_ok=False)
            _publish(roots,'allocation.json',{'version':VERSION,'protocol_digest':fingerprint(header),
                'protocol_directory':str(directory),'cutoff':cutoff,'source_files':header['source_files'],'retry':False})
            verified_header,identity=_bootstrap(directory,cutoff)
            if verified_header!=header:raise ValueError('Original header changed after durable allocation')
            _publish(roots,'identity.json',identity)
            from .nominal_factor_census import NominalFactorProtocol,produce_selected_month
            from .nominal_fit_outcome import NominalNumericalFailure
            from .v7_forward import ForwardRuntime,ForwardConfig
            from .v7_lifecycle import LifecyclePlan
            protocol=NominalFactorProtocol(directory)
            if protocol.identity!=header:raise ValueError('Original protocol changed after native bootstrap')
            active={'owner':(os.getpid(),threading.get_ident()),'protocol':protocol,'cutoff':cutoff,
                'identity':identity,'roots':roots,'lease':None}
            token=_RUN.set(active)
            runtime_root=_path(header['runtime_directory'])
            # Existing immutable runtime only; the constructor must not create an
            # alternative config/feed or a fresh wallet.
            envelope=json.loads((runtime_root/'forward.json').read_text('utf8'));ri=envelope['identity']
            if envelope['sha256']!=fingerprint(ri)or ri['source_files']!=identity['source_files']:raise ValueError('Original runtime identity differs')
            cfg=dict(ri['config']);cfg['symbols']=tuple(cfg['symbols'])
            lp=dict(ri['lifecycle']);lp['symbols']=tuple(lp['symbols'])
            _live(identity)
            runtime=ForwardRuntime(runtime_root,ri['feed_directory'],config=ForwardConfig(**cfg),lifecycle_plan=LifecyclePlan(**lp))
            outcome=produce_selected_month(protocol,runtime=runtime,cutoff=cutoff)
            lease=active['lease']
            if lease is None:raise ValueError('Original paid lease absent')
            if type(outcome)is NominalNumericalFailure:
                records=_numeric_records(protocol,cutoff)
                status='numeric_failed_consumed'
            elif type(outcome)is dict and set(outcome)=={'reference','candidate'}:
                records={'completed_roles':outcome};status='models_completed'
            else:raise ValueError('Unknown original producer result')
            runtime.close();runtime=None
            _live(identity,lease)
            if _installed()!=identity['source_files']:raise ValueError('Original numeric source changed before publication')
            publication={'version':VERSION,'identity_hash':fingerprint(identity),'lease_hash':lease.digest,
                'status':status,'records_hash':fingerprint(records),'provisional_only':True,'cleanup_signals':False}
            _publish(roots,'completion.json',publication)
            units,native=_api();tree=native._subtree(Path(identity['owned_group']['path']))
            units._quiescent_tree(identity,tree);_live(identity,lease)
            quiet={'identity_hash':fingerprint(identity),'completion_hash':fingerprint(publication),'tree':tree,'observed':units._observe()}
            _unit_bound(identity,quiet['observed']);lease.assert_parent_alive()
            _publish(roots,'quiescence.json',quiet)
            _failed(roots);_live(identity,lease)
            return {'version':VERSION,'status':status,'protocol_digest':protocol.digest,'cutoff':cutoff,
                'completion_hash':fingerprint(publication),'provisional_only':True,'cleanup_signals':False}
        except BaseException as error:
            _fail(roots,error);raise
        finally:
            if runtime is not None:runtime.close()
            if token is not None:_RUN.reset(token)

def _numeric_records(protocol,cutoff):
    """Byte/accounting consistency only; no solver or stored-authority restore."""
    from .neural_forward_ticket import open_existing_authority
    from .reserved_work_budget import ReservedChannelBudget
    from .nominal_factor_census import validate_nominal_factor_route
    from .work_lease import FrozenWorkLease
    from .neural_forward_admission import _kernel,_observation
    p=protocol.verify();plan=protocol.pair_plan(p['atom'],cutoff)
    folder=protocol.directory/'fits'/fingerprint(plan);intent=_read(folder/'intent.json')
    authority=open_existing_authority(p['authority_directory']);authority.verify()
    mirror=authority.directory/'nominal_factor_intents'/fingerprint(intent['work_key'])
    _failed((folder,mirror));roots=(folder/'numerical',mirror/'numerical')
    prep=_pair(roots,'preparation.json');terminal=_pair(roots,'numerical-terminal.json');publication=_pair(roots,'numerical-publication.json')
    event=_pair(tuple(r/terminal['failed_role']for r in roots),'numeric-failure.json')
    _event_shape(event)
    if event!=_pair(tuple(r/terminal['failed_role']for r in roots),f"kernel-{event['kernel_ordinal']:06d}.json"):
        raise ValueError('Original exact failed kernel event missing')
    lease=FrozenWorkLease.from_dict(intent['lease']);record=authority.get(prep['reservation_id'])
    validate_nominal_factor_route(authority,intent['work_key'],intent['binding'],lease,p['counts_per_pair'])
    actual=ReservedChannelBudget.verify_counts(authority,record['reservation_id'])
    raw=_read(folder/'raw.json')
    if (_read(mirror/'intent.json')!=intent or terminal['status']!='failed_consumed_numeric_provisional'
        or terminal['reservation']!=record or record['status']!='failed_consumed'
        or record.get('reason')!='nominal_numeric_linalg:'+fingerprint(event)
        or prep['plan']!=plan or prep['protocol_digest']!=protocol.digest or prep['source_files']!=p['source_files']
        or prep['authority_manifest_hash']!=authority.manifest_hash or prep['lease_hash']!=lease.digest
        or record['lease']!=lease.to_dict()or record['counts']!=p['counts_per_pair']
        or terminal['actual_accounting']!=actual or actual['test_only']or not actual['finalized']
        or any(actual['actual_counts'][k]>record['counts'][k]for k in record['counts'])
        or terminal['preparation_hash']!=fingerprint(prep)or terminal['numeric_event_hash']!=fingerprint(event)
        or event['preparation_hash']!=fingerprint(prep)or event['original_lease']!=lease.to_dict()
        or event['source_files']!=p['source_files']or event['role']!=terminal['failed_role']
        or event['budget']['actual_counts']!=actual['actual_counts']
        or event['exception_type']!='numpy.linalg.LinAlgError' or event['kernel_site']not in ('ridge_augmented','residual_projection')
        or terminal['raw_hash']!=fingerprint(raw)or prep['raw_hash']!=fingerprint(raw)or event['raw_hash']!=fingerprint(raw)
        or publication['terminal_hash']!=fingerprint(terminal)or terminal['serving_completion']is not False
        or terminal['later_roles']!='not_computed_after_original_abort'):
        raise ValueError('Original paid numeric event/accounting chain differs')
    parent=record['process_identity']
    if parent!={'pid':lease.parent_pid,'created':lease.parent_created}:
        raise ValueError('Original numerical issuer is not this native lease owner')
    _kernel(terminal['kernel_before'],terminal['kernel_before'],terminal['kernel_after'],lease,parent,parent)
    _kernel(terminal['kernel_before'],terminal['kernel_after'],publication['kernel'],lease,parent,parent)
    _observation(terminal['observation'],lease);_observation(publication['observation'],lease,previous=terminal['observation'])
    roles=terminal['nominal_roles']
    if set(roles)!={'reference','candidate'}or set(terminal['completed_roles'])!=set(('reference',)if terminal['failed_role']=='candidate'else()):
        raise ValueError('Original abort ordering differs')
    from .evidence import file_hash
    model_hashes={}
    for role in ('reference','candidate'):
        output=roles[role];roleplan=protocol.plan(p['atom'],cutoff,role)
        if output['plan']!=roleplan or output['raw_hash']!=fingerprint(raw):raise ValueError('Original role plan/raw differs')
        if role in terminal['completed_roles']:
            if output!=terminal['completed_roles'][role]:raise ValueError('Completed original role changed')
            model_hashes[role]=file_hash(folder/(role+'-model.json'))
        elif output['model_digest']is not None or any(r['prediction']is not None for r in output['rows']):
            raise ValueError('Numeric failure must not invent a model/prediction')
    return {'intent':intent,'preparation':prep,'event':event,'terminal':terminal,'publication':publication,
        'raw_hash':fingerprint(raw),'model_hashes':model_hashes,'roles':roles}

def _event_shape(event):
    """Finite kernel metadata consistency; only OS execution grants authority."""
    from .nominal_fit_outcome import VERSION as EVENT_VERSION,MAX_ROWS,MAX_COLS,MAX_KERNEL_CALLS
    if (event['version']!=EVENT_VERSION or event['status']!='executed_linalg_error_provisional'
        or event['historical_truth_certificate']is not False or type(event['kernel_ordinal'])is not int
        or not 1<=event['kernel_ordinal']<=MAX_KERNEL_CALLS or event['budget']['test_only']is not False):
        raise ValueError('Original exact numeric event schema required')
    values=event['inputs']
    if set(values)!={'matrix','response','rcond'}:raise ValueError('Original kernel input metadata required')
    for name,dimensions in (('matrix',2),('response',1)):
        value=values[name];shape=value['shape']
        if (set(value)!={'shape','dtype','sha256'}or value['dtype']!='binary64'or type(shape)is not list
            or len(shape)!=dimensions or any(type(n)is not int or n<1 for n in shape)
            or shape[0]>MAX_ROWS or dimensions==2 and shape[1]>MAX_COLS
            or type(value['sha256'])is not str or re.fullmatch('[0-9a-f]{64}',value['sha256'])is None):
            raise ValueError('Original finite bounded binary64 array metadata required')
    if values['matrix']['shape'][0]!=values['response']['shape'][0]:raise ValueError('Original actual matrix dimensions differ')
    rcond=values['rcond']
    if event['kernel_site']=='ridge_augmented':
        if rcond is not None:raise ValueError('Original ridge rcond changed')
    elif event['kernel_site']=='residual_projection':
        if type(rcond)is not str or not math.isfinite(float.fromhex(rcond)) or float.fromhex(rcond).hex()!=rcond:
            raise ValueError('Original actual finite residual rcond required')
    else:raise ValueError('Unknown original numerical kernel site')

def _exit_bounds(identity,raw,lease):
    starts=(Fraction(raw['ExecMainStartTimestampMonotonic'],10**6),Fraction(raw['ExecMainStartTimestamp'],10**6))
    ends=(Fraction(raw['ExecMainExitTimestampMonotonic'],10**6),Fraction(raw['ExecMainExitTimestamp'],10**6))
    ls=(Fraction(lease.monotonic_started),Fraction(datetime.fromisoformat(lease.wall_started_at).timestamp()))
    deadlines=(Fraction(lease.monotonic_deadline),Fraction(datetime.fromisoformat(lease.wall_deadline).timestamp()))
    if lease.lease_seconds!=300 or lease.boot_id!=identity['boot_id']or lease.parent_pid!=identity['exec_main_pid']or lease.parent_created!=identity['parent_created']:
        raise ValueError('Original paid lease owner differs')
    if any(not 0<s<=a<e<min(s+300,d)for s,a,e,d in zip(starts,ls,ends,deadlines)):
        raise ValueError('Original native pair did not finish strictly inside both deadlines')
    if datetime.fromtimestamp(float(ends[1]),timezone.utc).strftime('%Y-%m')!=datetime.fromisoformat(lease.wall_started_at).strftime('%Y-%m'):
        raise ValueError('Original physical wallet month changed')

def _retained(protocol,cutoff):
    from .nominal_factor_census import NominalFactorProtocol
    from .work_lease import FrozenWorkLease
    if type(protocol)is not NominalFactorProtocol:raise TypeError('Exact original fixed protocol required')
    if sys.platform!='linux':raise OSError('Actual retained original Linux unit required')
    p=protocol.verify();files=_installed();roots=_roots(p,cutoff);_failed(roots)
    identity=_pair(roots,'identity.json')
    allocation={'version':VERSION,'protocol_digest':protocol.digest,'protocol_directory':str(protocol.directory),
        'cutoff':cutoff,'source_files':files,'retry':False}
    if _pair(roots,'allocation.json')!=allocation:raise ValueError('Original attempt allocation absent or changed')
    if (p.get('numeric_native')!=descriptor()or identity['source_files']!=files or identity['protocol_digest']!=protocol.digest
        or identity['protocol_directory']!=str(protocol.directory)or identity['cutoff']!=cutoff
        or identity['command']!=command(protocol.directory,cutoff)or identity['unit']!=unit_name(protocol.digest,cutoff)
        or identity['runtime']!=_runtime_identity()):raise ValueError('Original prospective native source/coordinate differs')
    original=_pair(roots,'lease.json');lease=FrozenWorkLease.from_dict(original['lease'])
    if original['identity_hash']!=fingerprint(identity):raise ValueError('Original native/lease binding differs')
    units,native=_api();now=units._observe()
    if now['boot_id']!=identity['boot_id']:raise ValueError('Original boot authority lost')
    raw=units._unit(identity['unit']);_same(identity,raw)
    if (raw['ActiveState']!='active'or raw['SubState']!='exited'or raw['MainPID']!=0
        or raw['Result']!='success'or raw['ExecMainCode']!=1 or raw['ExecMainStatus']!=0):
        raise ValueError('Original fit unit has no natural-zero execution completion')
    _exit_bounds(identity,raw,lease)
    if (Fraction(raw['ExecMainExitTimestampMonotonic'],10**6)>Fraction(now['monotonic'])
        or Fraction(raw['ExecMainExitTimestamp'],10**6)>Fraction(now['wall'])):raise ValueError('Exit occurs after OS observation')
    if raw['ControlGroup']=='':units._released_group(identity)
    else:
        if units._group(raw)!=identity['owned_group']or units._shared(Path(identity['owned_group']['path']))!=identity['shared_group']:
            raise ValueError('Original retained group changed')
        tree=native._subtree(Path(identity['owned_group']['path']))
        if tree['members']or tree['populated']:raise ValueError('Original retained subtree not empty')
    records=_numeric_records(protocol,cutoff)
    if records['intent']['lease']!=lease.to_dict():raise ValueError('Native unit does not own the original paid fit')
    expected={'version':VERSION,'identity_hash':fingerprint(identity),'lease_hash':lease.digest,
        'status':'numeric_failed_consumed','records_hash':fingerprint(records),'provisional_only':True,'cleanup_signals':False}
    if _pair(roots,'completion.json')!=expected:raise ValueError('Original worker did not finish numeric publication')
    quiet=_pair(roots,'quiescence.json');observed=quiet['observed'];units._quiescent_tree(identity,quiet['tree'])
    if quiet['identity_hash']!=fingerprint(identity)or quiet['completion_hash']!=fingerprint(expected):raise ValueError('Quiescence closure differs')
    _unit_bound(identity,observed)
    if (not lease.monotonic_started<=observed['monotonic']<lease.monotonic_deadline
        or not datetime.fromisoformat(lease.wall_started_at).timestamp()<=observed['wall']<datetime.fromisoformat(lease.wall_deadline).timestamp()
        or Fraction(observed['monotonic'])>=Fraction(raw['ExecMainExitTimestampMonotonic'],10**6)
        or Fraction(observed['wall'])>=Fraction(raw['ExecMainExitTimestamp'],10**6)):
        raise ValueError('Original quiescence was not inside lease and before exit')
    if _installed()!=files:raise ValueError('Original source changed during read')
    _failed(roots);again=units._unit(identity['unit']);_same(identity,again)
    if again!=raw:raise ValueError('Original invocation changed during historical read')
    return identity,records,raw,units._observe()

class RetainedNumericObservation:
    def __init__(self,*args,**kwargs):raise TypeError('Use actual original retained fit reader')
    def __reduce__(self):raise TypeError('Original OS authority cannot serialize/restore')
    @property
    def identity(self):return _plain(self._identity)
    def assert_current(self):
        if self not in _LIVE or self._owner!=(os.getpid(),threading.get_ident()):raise ValueError('Original typed OS observation owner lost')
        identity,records,raw,observed=_retained(self._protocol,self._cutoff)
        if (identity!=self._identity or records!=self._records or raw!=self._raw
            or observed['boot_id']!=self._observed['boot_id']or observed['monotonic']<self._observed['monotonic']or observed['wall']<self._observed['wall']):
            raise ValueError('Original numerical execution witness changed')
        return self
    def role(self,role):
        if role not in ('reference','candidate'):raise ValueError('Original role required')
        self.assert_current();value=_plain(self._records['roles'][role])
        value['paid_pair_evidence']={'native_numeric_identity':fingerprint(self._identity),
            'records_hash':fingerprint(self._records),'reservation_id':self._records['terminal']['reservation']['reservation_id'],
            'status':'failed_consumed','native_completion_verified':True,'serving_completion':False,'formal_acceptance':False}
        return value

def read_retained_numeric(protocol,cutoff):
    identity,records,raw,now=_retained(protocol,cutoff)
    result=object.__new__(RetainedNumericObservation)
    result._protocol,result._cutoff,result._identity,result._records,result._raw,result._observed=protocol,cutoff,identity,records,raw,now
    result._owner=(os.getpid(),threading.get_ident());_LIVE.add(result);return result
