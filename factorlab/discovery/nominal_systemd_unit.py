"""Actual retained systemd service observations; JSON never restores authority.

Current original issuers bind their own unit. Historical reads require that same
retained invocation to have exited naturally, without restart, before the original
service deadlines. No start/stop/load/reset, task, model or time allocation occurs.
Released groups require source-bound pre-exit quiescence and the original live
retained unit's natural-zero exit. Stored quiescence never restores authority.
"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import threading
import time
from datetime import datetime
from fractions import Fraction

from factorlab.protocol import fingerprint, project_path

RELATIVE = 'factorlab/discovery/nominal_systemd_unit.py'
WORKER_RELATIVE = 'factorlab/discovery/nominal_verification_service_worker.py'
WORKER = 'factorlab.discovery.nominal_verification_service_worker'
VERSION = 'original-retained-verification-service-v1'
BUSCTL = '/usr/bin/busctl'
SHARED_SLICE = 'factorresearch-nv-shared.slice'
MAX_BYTES = 1024 * 1024
_KEY = object()
QUIESCENCE_VERSION = 'native-verification-service-quiescence-v1'

UNIT_FIELDS = {'Id': 's', 'LoadState': 's', 'ActiveState': 's', 'SubState': 's',
               'InvocationID': 'ay'}
SERVICE_FIELDS = {'Type': 's', 'ExitType': 's', 'Restart': 's', 'KillMode': 's',
    'OOMPolicy': 's', 'RemainAfterExit': 'b', 'Delegate': 'b', 'ControlGroup': 's',
    'Result': 's', 'NRestarts': 'u', 'MainPID': 'u', 'ExecMainPID': 'u',
    'ExecMainCode': 'i', 'ExecMainStatus': 'i', 'RuntimeMaxUSec': 't',
    'TimeoutStopUSec': 't', 'WorkingDirectory': 's', 'Environment': 'as',
    'UnsetEnvironment': 'as', 'PassEnvironment': 'as', 'EnvironmentFiles': 'a(sb)',
    'ExecMainStartTimestamp': 't', 'ExecMainStartTimestampMonotonic': 't',
    'ExecMainExitTimestamp': 't', 'ExecMainExitTimestampMonotonic': 't',
    'ExecStart': 'a(sasbttttuii)', 'ExecStartPre': 'a(sasbttttuii)',
    'ExecStartPost': 'a(sasbttttuii)', 'ExecStop': 'a(sasbttttuii)',
    'ExecStopPost': 'a(sasbttttuii)'}
UNSET_ENVIRONMENT = ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP', 'PYTHONINSPECT',
                     'PYTHONUSERBASE', 'LD_PRELOAD', 'LD_LIBRARY_PATH')


def _plain(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _json(raw):
    if type(raw) is not bytes or len(raw) > MAX_BYTES:
        raise ValueError('Bounded native bus reply required')
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError('Duplicate native bus key')
            out[key] = value
        return out
    def finite_float(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError('Nonfinite native bus numeric value')
        return value
    return json.loads(raw.decode('utf8'), object_pairs_hook=unique, parse_float=finite_float,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def _message(raw, signature):
    value = _json(raw)
    if (type(value) is not dict or set(value) != {'type', 'data'}
            or value['type'] != signature or type(value['data']) is not list
            or len(value['data']) != 1):
        raise ValueError('Exact original bus message signature required')
    return value['data'][0]


def _properties(raw, fields):
    values = _message(raw, 'a{sv}')
    if type(values) is not dict:
        raise ValueError('Original property map required')
    result = {}
    for name, signature in fields.items():
        v = values.get(name)
        if type(v) is not dict or set(v) != {'type', 'data'} or v['type'] != signature:
            raise ValueError('Required native property missing or mistyped: ' + name)
        data = v['data']  # A variant contains its value, not another message tuple.
        if signature == 's' and type(data) is not str:
            raise ValueError('Native string property required')
        if signature == 'b' and type(data) is not bool:
            raise ValueError('Native Boolean property required')
        if signature in {'u', 'i', 't'}:
            limit = 2**64 if signature == 't' else 2**32
            lower = -(2**31) if signature == 'i' else 0
            upper = 2**31 if signature == 'i' else limit
            if type(data) is not int or not lower <= data < upper:
                raise ValueError('Native bounded integer property required')
        if signature.startswith('a') and type(data) is not list:
            raise ValueError('Native array property required')
        if signature == 'as' and any(type(v) is not str for v in data):
            raise ValueError('Native string array property required')
        result[name] = data
    return result


def unit_environment():
    return {'FACTOR_RESEARCH_ROOT': str(project_path('.')),
        'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1',
        'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1', 'PATH': '/usr/bin:/bin',
        'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8', 'TZ': 'UTC',
        'TMPDIR': str(project_path('work/tmp')), 'TMP': str(project_path('work/tmp')),
        'TEMP': str(project_path('work/tmp')), 'XDG_CACHE_HOME': str(project_path('.cache')),
        'HF_HOME': str(project_path('.cache/huggingface')), 'TORCH_HOME': str(project_path('.cache/torch'))}


def service_unit_name(quota, index):
    from .nominal_verification_budget import VerificationRegistry
    if type(quota) is not VerificationRegistry:
        raise TypeError('Exact original verification quota required')
    plan = quota.plan.service(index)
    if plan['index'] == 0:
        raise ValueError('Registration service already consumed')
    return 'factorresearch-nv-' + quota.digest + '-' + f'{index:06d}' + '.service'


def service_command(quota, index):
    service_unit_name(quota, index)
    return [sys.executable, '-B', '-m', WORKER, '--registry', str(quota.directory),
            '--service-index', str(index)]


def _installed(quota):
    from .nominal_verification_budget import VerificationRegistry
    from .monthly_work_authority import source_files
    from .evidence import file_hash
    if type(quota) is not VerificationRegistry:
        raise TypeError('Exact original verification registry required')
    if sys.platform != 'linux':
        raise OSError('Retained Linux systemd unit required')
    files = source_files()
    if (Path(__file__).resolve() != project_path(RELATIVE).resolve()
            or files.get(RELATIVE) != file_hash(__file__)
            or files != quota.plan.binding['source_files']
            or WORKER_RELATIVE not in files
            or files[WORKER_RELATIVE] != file_hash(project_path(WORKER_RELATIVE))):
        raise ValueError('Whole canonical source and fixed service worker required')
    quota.verify()
    return files


def _remaining(service=None):
    if service is None:
        return 2.0  # Readonly inspector cap; not a service or task allowance.
    value = service.clock.assert_parent_alive().remaining_seconds
    if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
        raise TimeoutError('Original service expired')
    return min(2.0, value)


def _call(args, service=None):
    if sys.platform != 'linux':
        raise OSError('Native Linux bus unavailable')
    runtime = Path('/run/user') / str(os.getuid())
    bus = runtime / 'bus'
    if (runtime.resolve() != runtime or runtime.stat().st_uid != os.getuid()
            or bus.is_symlink() or bus.stat().st_uid != os.getuid()
            or not stat.S_ISSOCK(bus.stat().st_mode)):
        raise ValueError('Actual original user bus socket required')
    # Validate socket peer rather than inherited DBUS_SESSION_BUS_ADDRESS.
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
        peer.settimeout(_remaining(service))
        peer.connect(str(bus))
        import struct
        pid, uid, gid = struct.unpack('3i', peer.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        if pid <= 0 or uid != os.getuid() or gid < 0:
            raise ValueError('Actual original user bus peer required')
    cmd = [BUSCTL, '--address=unix:path=' + str(bus), '--json=short', '--no-pager',
           '--timeout=2s', 'call', 'org.freedesktop.systemd1', *args]
    result = subprocess.run(cmd, shell=False, check=True, capture_output=True,
        timeout=_remaining(service), env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8'})
    if service is not None:
        service.clock.assert_parent_alive()
    if len(result.stdout) > MAX_BYTES:
        raise ValueError('Native bus reply exceeds original read bound')
    return result.stdout


def _unit(name, service=None):
    # GetUnit only: never LoadUnit/StartUnit or replacement of missing history.
    path = _message(_call(['/org/freedesktop/systemd1', 'org.freedesktop.systemd1.Manager',
                          'GetUnit', 's', name], service), 'o')
    if type(path) is not str or not path.startswith('/org/freedesktop/systemd1/unit/'):
        raise ValueError('Original loaded unit object required')
    values = {'object_path': path}
    for suffix, fields in (('Unit', UNIT_FIELDS), ('Service', SERVICE_FIELDS)):
        raw = _call([path, 'org.freedesktop.DBus.Properties', 'GetAll', 's',
                     'org.freedesktop.systemd1.' + suffix], service)
        values.update(_properties(raw, fields))
    if values['Id'] != name:
        raise ValueError('Original unit identity differs')
    # GetAll calls are separate observations: re-read ID/invocation/main PID at
    # the end; an overlapping replacement must never combine two invocations.
    again = _properties(_call([path, 'org.freedesktop.DBus.Properties', 'GetAll', 's',
                              'org.freedesktop.systemd1.Unit'], service), UNIT_FIELDS)
    if any(again[key] != values[key] for key in UNIT_FIELDS):
        raise ValueError('Original unit changed across native property reads')
    return values


def _invocation(raw):
    value = raw['InvocationID']
    if (type(value) is not list or len(value) != 16
            or any(type(v) is not int or not 0 <= v <= 255 for v in value)
            or not any(value)):
        raise ValueError('Original nonzero invocation ID required')
    return bytes(value).hex()


def _command(raw):
    rows = raw['ExecStart']
    if type(rows) is not list or len(rows) != 1:
        raise ValueError('One original main command required')
    row = rows[0]
    if (type(row) is not list or len(row) != 10 or type(row[0]) is not str
            or type(row[1]) is not list or any(type(v) is not str for v in row[1])
            or row[2] is not False or any(type(v) is not int or v < 0 for v in row[3:8])
            or any(type(v) is not int for v in row[8:])):
        raise ValueError('Original nonignored structured command required')
    if row[0] != row[1][0]:
        raise ValueError('Main executable and argv differ')
    return row[1]


def _configuration(raw, expected_command):
    exact = {'LoadState': 'loaded', 'Type': 'exec', 'ExitType': 'main',
        'RemainAfterExit': True, 'Restart': 'no', 'KillMode': 'control-group',
        'Delegate': True, 'OOMPolicy': 'kill', 'RuntimeMaxUSec': 7200 * 10**6,
        'TimeoutStopUSec': 10**6, 'WorkingDirectory': str(project_path('.')), 'NRestarts': 0}
    if any(raw.get(k) != v or type(raw.get(k)) is not type(v) for k, v in exact.items()):
        raise ValueError('Original retained-unit containment contract changed')
    if _command(raw) != expected_command:
        raise ValueError('Fixed original service command differs')
    if any(raw[k] != [] for k in ('ExecStartPre', 'ExecStartPost', 'ExecStop', 'ExecStopPost')):
        raise ValueError('Auxiliary unit commands refused')
    env = raw['Environment']
    if len(env) != len(set(env)) or sorted(env) != sorted(k + '=' + v for k, v in unit_environment().items()):
        raise ValueError('Fixed source-root/thread environment required')
    if (sorted(raw['UnsetEnvironment']) != sorted(UNSET_ENVIRONMENT)
            or raw['PassEnvironment'] != [] or raw['EnvironmentFiles'] != []):
        raise ValueError('Import/library injection environment not excluded')
    _invocation(raw)


def _observe():
    from .work_lease import _boot_id
    return {'boot_id': _boot_id(), 'monotonic': time.monotonic(), 'wall': time.time()}


def _shared(group):
    from . import native_verification_terminal as native
    root = Path('/sys/fs/cgroup')
    for candidate in group.parents:
        if not candidate.is_relative_to(root):
            break
        try:
            raw = native._values(candidate)
        except OSError:
            continue
        if native._limits(raw) and candidate.name == SHARED_SLICE:
            return {'path': str(candidate), 'inode': candidate.stat().st_ino}
    raise ValueError('Original aggregate resource ancestor unavailable')


def _group(raw):
    from . import native_verification_terminal as native
    value = raw['ControlGroup']
    if (type(value) is not str or not value.startswith('/') or '..' in Path(value).parts
            or Path(value).name != raw['Id']):
        raise ValueError('Original owned service cgroup required')
    path = native._group_path(Path('/sys/fs/cgroup') / value.lstrip('/'))
    return {'path': str(path), 'inode': path.stat().st_ino}


def _same_invocation(identity, raw):
    if (_invocation(raw) != identity['invocation_id'] or raw['object_path'] != identity['object_path']
            or raw['ExecMainPID'] != identity['exec_main_pid']
            or raw['ExecMainStartTimestampMonotonic'] != identity['unit_start_monotonic_us']
            or raw['ExecMainStartTimestamp'] != identity['unit_start_realtime_us']):
        raise ValueError('Original invocation/process/cgroup identity unavailable')
    _configuration(raw, identity['command'])


def _same(identity, raw):
    _same_invocation(identity, raw)
    if (_group(raw) != identity['owned_group']
            or _shared(Path(identity['owned_group']['path'])) != identity['shared_group']):
        raise ValueError('Original live cgroup identity unavailable')


def _released_group(identity):
    """Only actual ENOENT, with existing unredirected parents, is release."""
    from . import native_verification_terminal as native
    path = Path(identity['owned_group']['path'])
    root = Path('/sys/fs/cgroup')
    if (not path.is_absolute() or not path.is_relative_to(root)
            or '..' in path.parts or path.name != identity['unit']):
        raise ValueError('Original absolute owned path required')
    # Check parent independently: a dangling alias or lost ancestor is not an
    # absent original leaf. Permission and other IO errors propagate.
    native._group_path(path.parent)
    try:
        path.lstat()
    except FileNotFoundError:
        pass
    else:
        raise ValueError('Released original path exists or was recreated')
    if _shared(path) != identity['shared_group']:
        raise ValueError('Original shared aggregate changed after release')


def _historical_group(identity, raw):
    from . import native_verification_terminal as native
    if raw['ControlGroup'] == '':
        _released_group(identity)
    else:
        _same(identity, raw)
        tree = native._subtree(Path(identity['owned_group']['path']))
        if tree['members'] or tree['populated']:
            raise ValueError('Original retained subtree still populated')


def _quiescent_tree(identity, tree):
    """Stored numeric consistency, never a kernel authority constructor."""
    pid = identity['exec_main_pid'];owned = identity['owned_group']
    if (type(pid) is not int or pid <= 0 or type(tree) is not dict
            or set(tree) != {'nodes', 'members', 'populated'}
            or tree['members'] != [pid] or tree['populated'] is not True
            or type(tree['nodes']) is not list or not tree['nodes']):
        raise ValueError('Only original main process may remain before exit')
    seen = set();main = 0
    for row in tree['nodes']:
        path = Path(row['path'])
        if (row['path'] in seen or not path.is_relative_to(Path(owned['path']))
                or '..' in path.parts or type(row['inode']) is not int or row['inode'] <= 0):
            raise ValueError('Original quiescent subtree paths differ')
        seen.add(row['path'])
        if row['path'] == owned['path']:
            main += 1
            if row['inode'] != owned['inode'] or row['members'] != [pid] or row['populated'] is not True:
                raise ValueError('Original main group differs at quiescence')
        elif row['members'] != [] or row['populated'] is not False:
            raise ValueError('Descendant members remain before service exit')
        if [int(v) for v in row['members_raw'].split()] != row['members']:
            raise ValueError('Original raw member observation differs')
        pairs = [v.split() for v in row['events'].splitlines()]
        if (any(len(v) != 2 for v in pairs) or len(pairs) != len({v[0] for v in pairs})
                or dict(pairs).get('populated') != ('1' if row['populated'] else '0')):
            raise ValueError('Original raw population observation differs')
    if main != 1:
        raise ValueError('Original owned main group missing')


def _batch_closure(quota, index, identity):
    """Recompute original whole-batch records; no verify_task recursion or OS claim."""
    from . import nominal_native_completion as completion
    from . import native_verification_terminal as native
    from .nominal_verification_budget import _read
    reader = completion.NativeCompletionReader(quota)
    files = reader._original()
    # The exact source map is already checked by _read_binding; explicitly bind
    # these numeric readers as well before using their private record validators.
    if (completion._module('nominal_native_completion', files) is not completion
            or completion._module('native_verification_terminal', files) is not native
            or files != identity['source_files']):
        raise ValueError('Original closure readers required')
    roots = [p/'services'/f'{index:06d}' for p in (quota.directory, quota.anchor)]
    batches = [p/'native-batch' for p in roots]
    if any((p/'failure.json').exists() for p in (*roots, *batches)):
        raise ValueError('Any original batch/service failure dominates')
    batch = completion._mirrors(batches, 'intent.json')
    inventory = completion._mirrors(batches, 'inventory.json')
    reaped = completion._mirrors(batches, 'reaped.json')
    plan = quota.plan.service(index)
    indices = list(range(plan['task_start'], plan['task_end_exclusive']))
    if not 0 < len(indices) <= 23:
        raise ValueError('Original bounded complete task range required')
    rows = [];events = []
    with quota.witness_read_scope():
        witness = quota.service_witness(index)
        for k in indices:
            task = quota.plan.task(k);target = completion._target(quota, task)
            data = reader._records(k, target, fingerprint(_read(target)['payload']))
            actual_roots, actual_batch, actual_unit = reader._batch_records(data, files)
            if actual_batch != batch or actual_unit != identity or tuple(actual_roots) != tuple(batches):
                raise ValueError('Original complete task batch/unit differs')
            snapshot = completion._mirrors(data['roots'], 'native-provisional.json')
            wait = completion._snapshot_check(snapshot, data, batch, files, native)
            rows.append({'task_index': k, 'snapshot_hash': fingerprint(snapshot)})
            events.append({'task_index': k, 'wait': wait, 'natural_reap': True, 'cleanup_signal': False})
    expected = {'version': 'native-verification-batch-v1', 'registry_digest': quota.digest,
        'service_index': index, 'batch_intent_hash': fingerprint(batch),
        'unit_binding_digest': fingerprint(identity), 'source_files': files, 'records': rows,
        'service_witness_hash': fingerprint(witness), 'provisional_only': True, 'reap_pending': True}
    if (inventory != expected or reaped != {'version': 'native-verification-batch-v1',
            'batch_intent_hash': fingerprint(batch), 'inventory_hash': fingerprint(inventory),
            'reaped': events, 'provisional_only': True, 'cleanup_signals': False}):
        raise ValueError('Original whole inventory/natural reap differs')
    return {'batch_intent_hash': fingerprint(batch), 'inventory_hash': fingerprint(inventory),
            'reaped_hash': fingerprint(reaped), 'service_witness_hash': fingerprint(witness)}


def _quiescence_base(quota, index, identity, closure):
    return {'version': QUIESCENCE_VERSION, 'registry_digest': quota.digest, 'service_index': index,
        'source_files': identity['source_files'], 'unit_binding_hash': fingerprint(identity),
        'owned_group': identity['owned_group'], 'shared_group': identity['shared_group'],
        'original_main_pid': identity['exec_main_pid'], 'closure': closure,
        'provisional_only': True}


def _read_quiescence(quota, index, identity, closure, raw):
    from .nominal_verification_budget import _read
    from .full_trial_controller import ControllerStepLease
    from types import SimpleNamespace
    roots = [p/'services'/f'{index:06d}' for p in (quota.directory, quota.anchor)]
    value = _read(roots[0]/'native-quiescence.json')
    if value != _read(roots[1]/'native-quiescence.json'):
        raise ValueError('Original quiescence mirrors differ')
    base = _quiescence_base(quota, index, identity, closure)
    if (type(value) is not dict or set(value) != set(base) | {'subtree', 'observed'}
            or any(value[k] != v for k, v in base.items())):
        raise ValueError('Original quiescence closure changed')
    _quiescent_tree(identity, value['subtree'])
    clock = ControllerStepLease.from_dict(identity['clock'], SimpleNamespace(service={'wall_seconds': 7200}))
    witness = quota.service_witness(index)
    if (not clock.evaluate(value['observed'], witness['observed']).allowed
            or Fraction(value['observed']['monotonic']) >= Fraction(raw['ExecMainExitTimestampMonotonic'], 10**6)
            or Fraction(value['observed']['wall']) >= Fraction(raw['ExecMainExitTimestamp'], 10**6)):
        raise ValueError('Original quiescence must precede natural exit inside original clock')
    return value


def _deadline(identity, raw):
    clock = identity['clock']
    issuer = clock['issuer']
    end_m = Fraction(raw['ExecMainExitTimestampMonotonic'], 10**6)
    end_w = Fraction(raw['ExecMainExitTimestamp'], 10**6)
    start_m = Fraction(raw['ExecMainStartTimestampMonotonic'], 10**6)
    start_w = Fraction(raw['ExecMainStartTimestamp'], 10**6)
    # Honor the earlier of exact arithmetic and the original floating-point
    # supervisor deadline; do not add nanoseconds through a new rounding rule.
    deadline_m = min(start_m + 7200,
                     Fraction(issuer['monotonic_started']) + 7200,
                     Fraction(issuer['monotonic_started'] + 7200))
    start = datetime.fromisoformat(issuer['wall_started_at'])
    from datetime import timedelta
    deadline_w = min(start_w + 7200,
                     Fraction(start.timestamp()) + 7200,
                     Fraction((start + timedelta(seconds=7200)).timestamp()))
    if (clock['service_seconds'] != 7200 or issuer['boot_id'] != identity['boot_id']
            or not 0 < start_m <= Fraction(issuer['monotonic_started']) < end_m < deadline_m
            or not 0 < start_w <= Fraction(datetime.fromisoformat(issuer['wall_started_at']).timestamp()) < end_w < deadline_w):
        raise ValueError('Actual original service exit missed absolute deadlines')


def _read_binding(quota, index):
    from .nominal_verification_budget import _read
    roots = [p / 'services' / f'{index:06d}' for p in (quota.directory, quota.anchor)]
    if any((p / 'failure.json').exists() for p in roots):
        raise ValueError('Original service failure dominates unit completion')
    identity = _read(roots[0] / 'native-unit.json')
    if identity != _read(roots[1] / 'native-unit.json'):
        raise ValueError('Original native-unit mirrors differ')
    binding = _read(roots[0] / 'intent.json')
    if (binding != _read(roots[1] / 'intent.json') or identity['version'] != VERSION
            or identity['registry_digest'] != quota.digest or identity['service_index'] != index
            or identity['service_plan'] != quota.plan.service(index)
            or identity['service_binding_hash'] != fingerprint(binding)
            or identity['clock'] != binding['clock'] or identity['unit'] != service_unit_name(quota, index)
            or identity['command'] != service_command(quota, index)
            or identity['source_files'] != _installed(quota)):
        raise ValueError('Original service/unit/source binding differs')
    from .evidence import file_hash
    if (identity['executable_sha256'] != file_hash(sys.executable)
            or identity['busctl_sha256'] != file_hash(BUSCTL)):
        raise ValueError('Original Python or native bus executable changed')
    return identity


class RetainedServiceBinding:
    def __init__(self, *args, **kwargs):
        raise TypeError('Use bind_current_service with original service')
    def __reduce__(self):
        raise TypeError('Live unit binding cannot serialize as authority')
    @property
    def identity(self):
        return _plain(self._identity)
    @property
    def digest(self):
        return fingerprint(self._identity)
    def _check(self, closing):
        if (self._key is not _KEY or self._owner != (os.getpid(), threading.get_ident())
                or self._identity['exec_main_pid'] != os.getpid()):
            raise ValueError('Original service owner required')
        if closing:
            self._service.clock.assert_parent_alive()
        else:
            self._service.check()
        if _read_binding(self._service.registry, self._service.plan['index']) != self._identity:
            raise ValueError('Original live binding changed')
        raw = _unit(self._identity['unit'], self._service)
        _same(self._identity, raw)
        if raw['ActiveState'] != 'active' or raw['SubState'] != 'running' or raw['MainPID'] != os.getpid():
            raise ValueError('Original live service unavailable')
        self._service.clock.assert_parent_alive()
        return _plain(raw)
    def check_live(self):
        return self._check(False)
    def check_closing(self):
        return self._check(True)
    def publish_quiescence(self):
        """After original natural reaps, observe the real remaining main process."""
        from .nominal_verification_budget import VerificationService, _read, _write
        from . import native_verification_terminal as native
        if (type(self) is not RetainedServiceBinding or type(self._service) is not VerificationService
                or self._service.closed is not True):
            raise TypeError('Exact original closed service and live binding required')
        self.check_closing()
        service = self._service;index = service.plan['index'];identity = self._identity
        paths = [p/'native-quiescence.json' for p in service.roots]
        if any(p.exists() for p in paths):
            raise ValueError('Original quiescence already attempted; no replacement')
        closure = _batch_closure(service.registry, index, identity)
        self.check_closing()
        tree = native._subtree(Path(identity['owned_group']['path']))
        _quiescent_tree(identity, tree)
        observed = service.clock.observe()
        if not service.clock.evaluate(observed).allowed:
            raise ValueError('Quiescence missed original service clock')
        value = {**_quiescence_base(service.registry, index, identity, closure),
                 'subtree': tree, 'observed': observed}
        for p in paths:
            _write(p, value)
        if any(_read(p) != value for p in paths):
            raise ValueError('Original quiescence publication changed')
        if _batch_closure(service.registry, index, identity) != closure:
            raise ValueError('Original batch closure changed during publication')
        self.check_closing()
        _quiescent_tree(identity, native._subtree(Path(identity['owned_group']['path'])))
        self.check_closing()
        service.clock.assert_parent_alive()
        return _plain(value)


def bind_current_service(service):
    from .nominal_verification_budget import VerificationService, _write
    from . import native_verification_terminal as native
    from .evidence import file_hash
    if type(service) is not VerificationService:
        raise TypeError('Exact original acquired service required')
    service.check()
    files = _installed(service.registry)
    raw = _unit(service_unit_name(service.registry, service.plan['index']), service)
    command = service_command(service.registry, service.plan['index'])
    _configuration(raw, command)
    if any(name in os.environ for name in UNSET_ENVIRONMENT):
        raise ValueError('Original service inherited import/library injection')
    if (raw['ActiveState'] != 'active' or raw['SubState'] != 'running'
            or raw['MainPID'] != os.getpid() or raw['ExecMainPID'] != os.getpid()
            or service.clock.issuer['parent_pid'] != os.getpid()
            or os.environ.get('INVOCATION_ID') != _invocation(raw)):
        raise ValueError('Current original unit/main process required')
    owned = _group(raw)
    if native._group(os.getpid()) != Path(owned['path']):
        raise ValueError('Original issuer did not join its own fixed service group')
    observed = _observe()
    if (observed['boot_id'] != service.clock.issuer['boot_id']
            or raw['ExecMainStartTimestampMonotonic'] <= 0
            or Fraction(raw['ExecMainStartTimestampMonotonic'], 10**6) > Fraction(service.clock.issuer['monotonic_started'])):
        raise ValueError('Original boot/unit issuance order differs')
    identity = {'version': VERSION, 'registry_digest': service.registry.digest,
        'service_index': service.plan['index'], 'service_plan': service.plan,
        'service_binding_hash': fingerprint(service.binding), 'clock': service.clock.to_dict(),
        'boot_id': observed['boot_id'], 'unit': raw['Id'], 'object_path': raw['object_path'],
        'invocation_id': _invocation(raw), 'exec_main_pid': os.getpid(),
        'unit_start_monotonic_us': raw['ExecMainStartTimestampMonotonic'],
        'unit_start_realtime_us': raw['ExecMainStartTimestamp'],
        'command': command, 'owned_group': owned, 'shared_group': _shared(Path(owned['path'])),
        'executable_sha256': file_hash(sys.executable), 'busctl_sha256': file_hash(BUSCTL),
        'source_files': files}
    if any((p / 'native-unit.json').exists() for p in service.roots):
        raise ValueError('Original unit binding already attempted; no replacement')
    for path in service.roots:
        _write(path / 'native-unit.json', identity)
    service.check()
    result = object.__new__(RetainedServiceBinding)
    result._key = _KEY
    result._service = service
    result._owner = (os.getpid(), threading.get_ident())
    result._identity = _plain(identity)
    result.check_live()
    return result


class RetainedServiceObservation:
    def __init__(self, *args, **kwargs):
        raise TypeError('Use read_retained_service; copied observations are not authority')
    def __reduce__(self):
        raise TypeError('Retained observation cannot be restored from JSON')
    @property
    def identity(self):
        return _plain(self._identity)
    @property
    def properties(self):
        return _plain(self._raw)
    @property
    def observed(self):
        return _plain(self._observed)
    def assert_current(self):
        if self._key is not _KEY:
            raise ValueError('Original typed retained observation required')
        identity, raw, observed = _retained(self._quota, self._index)
        if identity != self._identity or raw != self._raw or observed['boot_id'] != self._observed['boot_id']:
            raise ValueError('Original retained unit changed during evidence read')
        if observed['monotonic'] < self._observed['monotonic'] or observed['wall'] < self._observed['wall']:
            raise ValueError('Physical observation clock regressed')
        return self


def _retained(quota, index):
    identity = _read_binding(quota, index)
    quota.service_witness(index)
    observed = _observe()
    if observed['boot_id'] != identity['boot_id']:
        raise ValueError('Original retained service boot lost')
    raw = _unit(identity['unit'])
    _same_invocation(identity, raw)
    if (raw['ActiveState'] != 'active' or raw['SubState'] != 'exited'
            or raw['MainPID'] != 0 or raw['Result'] != 'success'
            or raw['ExecMainCode'] != 1 or raw['ExecMainStatus'] != 0):
        raise ValueError('Original naturally exited retained service required')
    _deadline(identity, raw)
    _historical_group(identity, raw)
    closure = _batch_closure(quota, index, identity)
    quiescence = _read_quiescence(quota, index, identity, closure, raw)
    quota.verify()
    # Final fresh OS and record reads: restored tokens or old JSON are not an
    # observation of this retained invocation. Any mid-read change rejects.
    if _read_binding(quota, index) != identity:
        raise ValueError('Original retained binding changed during read')
    if (_batch_closure(quota, index, identity) != closure
            or _read_quiescence(quota, index, identity, closure, raw) != quiescence):
        raise ValueError('Original quiescence closure changed during read')
    _historical_group(identity, raw)
    if _unit(identity['unit']) != raw:
        raise ValueError('Original retained unit changed during record/group reads')
    final = _observe()
    if (final['boot_id'] != observed['boot_id'] or final['monotonic'] < observed['monotonic']
            or final['wall'] < observed['wall']):
        raise ValueError('Original native observation clock changed')
    return identity, raw, final


def read_retained_service(quota, index):
    identity, raw, observed = _retained(quota, index)
    result = object.__new__(RetainedServiceObservation)
    result._key, result._quota, result._index = _KEY, quota, index
    result._identity, result._raw, result._observed = _plain(identity), _plain(raw), observed
    return result
