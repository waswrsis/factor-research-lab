"""Once-only fixed full-scope unit preparation and launch; no new work credit.

Administrative unit installation is separate from a paid execution. Existing
original enrolled stores are required and never repaired or initialized here.
The actual fixed worker still owns every clock, wallet and native completion.
"""
from __future__ import annotations
from pathlib import Path
import importlib
import os
import re
import sys
from factorlab.protocol import fingerprint, project_path

VERSION='nominal-original-fullscope-unit-launcher-v1'
RELATIVE='factorlab/discovery/nominal_full_scope_launcher.py'


def _installed(plan):
    from .nominal_full_scope_proof import FullScopeProofPlan
    from .monthly_work_authority import source_files
    from .evidence import file_hash
    if type(plan) is not FullScopeProofPlan:
        raise TypeError('Exact frozen original full-scope plan required')
    plan.verify()
    files=source_files()
    if files != plan.identity['source_files']:
        raise ValueError('Original registered complete source changed')
    for name in ('nominal_full_scope_launcher','nominal_full_scope_proof',
                 'nominal_full_scope_worker','nominal_algorithm_producer',
                 'nominal_capacity_preflight','nominal_systemd_unit',
                 'full_calibration_registry','nominal_algorithm_total_risk'):
        module=importlib.import_module('factorlab.discovery.'+name)
        rel='factorlab/discovery/'+name+'.py'
        if (Path(module.__file__).resolve()!=project_path(rel).resolve()
                or files.get(rel)!=file_hash(module.__file__)):
            raise ValueError('Launcher dependency is not the exact registered source: '+name)
    return files


def _capacity(plan):
    from .nominal_algorithm_total_risk import AlgorithmTotalRiskRegistry
    from .nominal_algorithm_policy import NominalAlgorithmPolicy
    from .nominal_capacity_preflight import NominalCapacityPreflight
    policy=NominalAlgorithmPolicy.inspect(AlgorithmTotalRiskRegistry.inspect(plan.parent))
    observation=NominalCapacityPreflight.analyze(policy)
    observation.assert_current()
    observation.require_not_proven_impossible()
    return observation.report


def _roots(plan, coordinate):
    plan.roots(coordinate)  # Exact registered coordinate; rejects alternate slots.
    original=plan.parent.paths(coordinate['cell'],coordinate['index'],
                              oracle_index=coordinate['oracle_index'])['root']
    return (project_path(original/'native-full-launch'),
            project_path(plan.directory/'launches'/coordinate['trial_id']))


def _quote(value, *, exec_argument=False):
    if type(value) is not str or any(ord(c)<32 or ord(c)==127 for c in value):
        raise ValueError('One original fixed unit argument required')
    escaped=value.replace('\\','\\\\').replace('"','\\"').replace('%','%%')
    if exec_argument:escaped=escaped.replace('$','$$')
    return '"'+escaped+'"'


def _working_directory(value):
    value = value.as_posix()
    if not value or not value.isprintable() or any(c in value for c in '\\\"\''):
        raise ValueError('Invalid systemd working directory')
    # WorkingDirectory is a raw path with specifiers, not command argv.
    # The same workspace is an executable path: quotes/backslashes above
    # are rejected by systemd regardless of argv escaping.
    if value[-1].isspace():
        value += '/.'  # Preserve final whitespace through INI parsing.
    return value.replace('%', '%%')


def _sync_directory(path):
    """A consumed hardlink/name must be durable before an external side effect."""
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def _durable_pair(roots,name,value):
    from .nominal_full_scope_proof import _write_pair
    _write_pair(roots,name,value)
    # Original immutable_record already fsyncs each actual record payload.
    # Here Linux side effects additionally require the directory entries durable.
    if sys.platform=='linux':
        for root in roots:
            _sync_directory(root)
            _sync_directory(root.parent)


def unit_text(plan, coordinate):
    from .nominal_full_scope_proof import command
    from . import nominal_systemd_unit as units
    _installed(plan);plan.roots(coordinate)
    seconds=plan.identity['original_service_seconds']
    if type(seconds) not in (int,float) or not 300<=seconds<=7200:
        raise ValueError('Original service ceiling must remain 300..7200 seconds')
    lines=['[Unit]','Description=Original once-only complete nominal policy',
        '[Service]','Type=exec','ExitType=main','RemainAfterExit=yes','Restart=no',
        'KillMode=control-group','Delegate=yes','OOMPolicy=kill',
        'RuntimeMaxSec='+str(seconds),'TimeoutStopSec=1','Slice='+units.SHARED_SLICE,
        'WorkingDirectory='+_working_directory(project_path('.')),
        'ExecStart='+' '.join(_quote(value,exec_argument=True) for value in command(plan,coordinate))]
    lines+=['Environment='+_quote(key+'='+value) for key,value in units.unit_environment().items()]
    lines+=['UnsetEnvironment='+' '.join(units.UNSET_ENVIRONMENT)]
    return '\n'.join(lines)+'\n'


def _descriptor(plan, coordinate):
    from .nominal_full_scope_proof import unit_name, command
    return {'version':VERSION,'plan_digest':plan.digest,'coordinate':coordinate,
            'unit':unit_name(plan,coordinate),'command':command(plan,coordinate),
            'unit_text_hash':fingerprint(unit_text(plan,coordinate)),
            'source_files':plan.identity['source_files'],'retry_allowed':False,
            'new_lease_or_wallet':False,'native_completed':False}


def _unused_original_stores(plan, coordinate):
    from .full_calibration_registry import QuarantinedTrialPermit
    from .neural_forward_ticket import open_existing_authority
    permit=QuarantinedTrialPermit(plan.parent,coordinate)
    permit.verify()
    required=(permit.paths['runtime']/'forward.json',permit.paths['runtime']/'lifecycle.sqlite',
              permit.paths['feed']/'identity.json',permit.paths['feed']/'receipts.sqlite')
    if any(not path.is_file() or path.is_symlink() for path in required):
        raise ValueError('Original pre-enrolled stores required; no repair or initialization')
    consumed=(permit.paths['root']/'controller',permit.paths['root']/'nominal_algorithm_execution',
              permit.paths['root']/'nominal_algorithm_producer',*plan.roots(coordinate))
    if any(path.exists() for path in consumed):
        raise ValueError('Original coordinate already attempted; no new native launch')
    authority=open_existing_authority(permit.paths['authority'])
    if authority.verify()['reservations']:
        raise ValueError('Original wallet already consumed')
    return permit


def prepare_unit(plan, cell, index, *, oracle_index=None):
    """Prepare fixed descriptor for a fresh enrolled coordinate, without launch."""
    from filelock import FileLock
    from .nominal_full_scope_proof import _write_pair
    _installed(plan)
    capacity=_capacity(plan)
    coordinate=plan.coordinate(cell,index,oracle_index)
    roots=_roots(plan,coordinate)
    with FileLock(str(roots[0])+'.lock',timeout=0):
        _unused_original_stores(plan,coordinate)
        if any(root.exists() for root in roots):
            raise ValueError('Partial or previous launch preparation is consumed')
        for root in roots:root.mkdir(parents=True,exist_ok=False)
        value=_descriptor(plan,coordinate)
        _durable_pair(roots,'intent.json',value)
        _installed(plan)
    return {'unit':value['unit'],'text':unit_text(plan,coordinate),
            'coordinate':coordinate,'capacity_necessary_only':capacity,
            'native_completed':False,'formal_acceptance':False}


def _prepared(plan, cell, index, oracle_index):
    from .nominal_full_scope_proof import _pair
    _installed(plan)
    coordinate=plan.coordinate(cell,index,oracle_index)
    roots=_roots(plan,coordinate)
    value=_pair(roots,'intent.json')
    if value!=_descriptor(plan,coordinate):
        raise ValueError('Original prepared launcher identity changed')
    return coordinate,roots,value


def _unit_file(value):
    if re.fullmatch(r'factorresearch-nf-[0-9a-f]{64}\.service',value['unit']) is None:
        raise ValueError('Only exact fixed full-scope unit names are installable')
    directory=Path.home()/'.config/systemd/user'
    path=directory/value['unit']
    # No alternate directory, symlink replacement or user command is accepted.
    for parent in (path,*path.parents):
        if parent.is_symlink():raise ValueError('Native unit installation cannot traverse symlinks')
    if path.is_absolute() is not True or path.resolve()!=path:
        raise ValueError('Fixed real user unit path required')
    return path


def install_unit_once(plan, cell, index, *, oracle_index=None):
    """Install only the exact prepared unit; never overwrite an existing file."""
    from filelock import FileLock
    from .nominal_full_scope_proof import _write_pair
    from . import nominal_systemd_unit as units
    if sys.platform!='linux':raise OSError('Actual Linux user systemd required')
    coordinate,roots,value=_prepared(plan,cell,index,oracle_index)
    with FileLock(str(roots[0])+'.install.lock',timeout=0):
        if any((root/name).exists() for root in roots for name in ('install.json','install-ready.json')):
            raise ValueError('Original installation attempt consumed')
        path=_unit_file(value)
        if path.exists():raise ValueError('Existing native unit must not be replaced')
        _durable_pair(roots,'install.json',{'version':VERSION,'intent_hash':fingerprint(value),
                                        'unit_file':str(path),'retry_allowed':False})
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x',encoding='utf8',newline='\n') as stream:
            stream.write(unit_text(plan,coordinate));stream.flush();os.fsync(stream.fileno())
        _sync_directory(path.parent)
        if path.read_text('utf8')!=unit_text(plan,coordinate):raise ValueError('Original unit readback changed')
        units._call(['/org/freedesktop/systemd1','org.freedesktop.systemd1.Manager','Reload'])
        _installed(plan)
        if path.read_text('utf8')!=unit_text(plan,coordinate):raise ValueError('Reload changed original unit bytes')
        _durable_pair(roots,'install-ready.json',{'version':VERSION,'intent_hash':fingerprint(value),
            'unit_file':str(path),'unit_text_hash':value['unit_text_hash'],'reload_completed':True})
    return {'unit':value['unit'],'installed':True,'launched':False}


def launch_once(plan, cell, index, *, oracle_index=None):
    """Durable consumed marker before fixed StartUnit, with no retry or new credit."""
    from filelock import FileLock
    from .nominal_full_scope_proof import _pair,_write_pair
    from . import nominal_systemd_unit as units
    if sys.platform!='linux':raise OSError('Actual Linux user systemd required')
    coordinate,roots,value=_prepared(plan,cell,index,oracle_index)
    capacity=_capacity(plan)
    path=_unit_file(value)
    installed={'version':VERSION,'intent_hash':fingerprint(value),'unit_file':str(path),'retry_allowed':False}
    ready={'version':VERSION,'intent_hash':fingerprint(value),'unit_file':str(path),
           'unit_text_hash':value['unit_text_hash'],'reload_completed':True}
    if (_pair(roots,'install.json')!=installed or _pair(roots,'install-ready.json')!=ready
            or path.read_text('utf8')!=unit_text(plan,coordinate)):
        raise ValueError('Exact once-installed original unit required')
    with FileLock(str(roots[0])+'.launch.lock',timeout=0):
        _unused_original_stores(plan,coordinate)
        if any((root/'launch.json').exists() for root in roots):
            raise ValueError('Original native launch consumed; never restart')
        _durable_pair(roots,'launch.json',{'version':VERSION,'intent_hash':fingerprint(value),
            'capacity_hash':fingerprint(capacity),'retain_N':True,'retry_allowed':False})
        _installed(plan)
        if path.read_text('utf8')!=unit_text(plan,coordinate):raise ValueError('Original unit changed before StartUnit')
        return units._message(units._call(['/org/freedesktop/systemd1','org.freedesktop.systemd1.Manager',
            'StartUnit','ss',value['unit'],'fail']),'o')
