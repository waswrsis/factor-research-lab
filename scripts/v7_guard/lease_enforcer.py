"""STAGE: dedicated issuer supervision; no lease, quota or admission grant.

The supervisor owns a Python issuer created by this fixed bootstrap. The issuer
keeps its actual PID and executes its existing numeric/provider/publication path.
Only an existing canonical reservation can arm a deadline. Windows uses a Job
Object; Linux requires the supervisor to be the actual bounded systemd service
MainPID, with control-group teardown and writable cgroup.kill. No POSIX-group
fallback claims containment of a provider that starts a separate session.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import runpy
import subprocess
import sys
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from factorlab import protocol
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
from factorlab.discovery.work_lease import FrozenWorkLease, LeaseCheck, _boot_id, _evaluate


def _write(path, value):
    path = Path(path)
    raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.pending')
    try:
        with temporary.open('xb') as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        # A visible handshake must already be complete. link is atomic and
        # refuses replacement of an existing immutable record on both hosts.
        os.link(temporary, path)
        if os.name != 'nt':
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _read(path):
    path = Path(path)
    if path.stat().st_size > 65536:
        raise ValueError('Bounded guard record required')
    return json.loads(path.read_text())


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _identity(process):
    return {'pid': process.pid, 'created': process.create_time()}


def _ancestry(process, supervisor):
    values = []
    for parent in process.parents():
        values.append(_identity(parent))
        if _identity(parent) == supervisor:
            return values
    raise ValueError('Issuer is not a child of its actual supervisor')


def _authority(directory):
    root = Path(directory).resolve()
    wrapper = _read(root / 'manifest.json')
    if wrapper['hash'] != protocol.fingerprint(wrapper['record']):
        raise ValueError('Authority manifest changed')
    obj = MonthlyWorkAuthority.__new__(MonthlyWorkAuthority)
    obj.directory, obj.database = root, root / 'authority.sqlite'
    obj.manifest, obj.manifest_hash = wrapper['record'], wrapper['hash']
    obj.limits = obj.manifest['limits'].copy()
    obj.index_path = protocol.project_path('work/monthly_work_authority_index') / (
        protocol.fingerprint(obj.manifest['scope']) + '.json')
    obj.verify()
    return obj


class _WindowsJob:
    """Kernel-owned membership includes later children; breakaway is not enabled."""
    def __init__(self):
        from ctypes import wintypes as w
        class Basic(ctypes.Structure):
            _fields_ = [('per_process', ctypes.c_int64), ('per_job', ctypes.c_int64),
                        ('flags', w.DWORD), ('min_ws', ctypes.c_size_t), ('max_ws', ctypes.c_size_t),
                        ('active', w.DWORD), ('affinity', ctypes.c_size_t),
                        ('priority', w.DWORD), ('scheduling', w.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in ('ro', 'wo', 'oo', 'rb', 'wb', 'ob')]
        class Extended(ctypes.Structure):
            _fields_ = [('basic', Basic), ('io', IO), ('process_memory', ctypes.c_size_t),
                        ('job_memory', ctypes.c_size_t), ('peak_process', ctypes.c_size_t),
                        ('peak_job', ctypes.c_size_t)]
        self.dll = ctypes.WinDLL('kernel32', use_last_error=True)
        self.dll.CreateJobObjectW.argtypes = (ctypes.c_void_p, w.LPCWSTR)
        self.dll.CreateJobObjectW.restype = w.HANDLE
        self.dll.SetInformationJobObject.argtypes = (w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD)
        self.dll.AssignProcessToJobObject.argtypes = (w.HANDLE, w.HANDLE)
        self.dll.TerminateJobObject.argtypes = (w.HANDLE, w.UINT)
        self.dll.QueryInformationJobObject.argtypes = (w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD, ctypes.c_void_p)
        self.dll.CloseHandle.argtypes = (w.HANDLE,)
        self.handle = self.dll.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        settings = Extended()
        settings.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, no breakaway.
        if not self.dll.SetInformationJobObject(self.handle, 9, ctypes.byref(settings), ctypes.sizeof(settings)):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())

    def attach(self, process):
        # Popen owns this exact native process handle, so PID reuse is irrelevant.
        if not self.dll.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())
        # Windows venv python.exe is a redirector which creates another native
        # process. Assign the suspended root BEFORE any redirector can fork.
        from ctypes import wintypes as w
        threads = psutil.Process(process.pid).threads()
        if len(threads) != 1:
            raise RuntimeError('Suspended bootstrap has unexpected native threads')
        self.dll.OpenThread.argtypes = (w.DWORD, w.BOOL, w.DWORD)
        self.dll.OpenThread.restype = w.HANDLE
        self.dll.ResumeThread.argtypes = (w.HANDLE,)
        self.dll.ResumeThread.restype = w.DWORD
        handle = self.dll.OpenThread(0x0002, False, threads[0].id)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if self.dll.ResumeThread(handle) == 0xffffffff:
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            self.dll.CloseHandle(handle)

    def members(self):
        class Members(ctypes.Structure):
            _fields_ = [('assigned', ctypes.c_uint32), ('listed', ctypes.c_uint32),
                        ('pids', ctypes.c_size_t * 256)]
        result = Members()
        if not self.dll.QueryInformationJobObject(self.handle, 3, ctypes.byref(result), ctypes.sizeof(result), None):
            raise ctypes.WinError(ctypes.get_last_error())
        return list(result.pids[:result.listed])

    def stop(self):
        if not self.dll.TerminateJobObject(self.handle, 124):
            raise ctypes.WinError(ctypes.get_last_error())
        end = time.monotonic() + 5
        while self.members() and time.monotonic() < end:
            time.sleep(.01)
        if self.members():
            raise RuntimeError('Owned Job Object still contains running processes')

    def close(self):
        if self.handle:
            self.dll.CloseHandle(self.handle)
            self.handle = None


class _LinuxService:
    """Strict production prerequisite, not exercised by Windows acceptance."""
    def __init__(self):
        line = Path('/proc/self/cgroup').read_text().strip()
        if not line.startswith('0::/') or '\n' in line:
            raise ValueError('Actual unified owned service cgroup required')
        self.relative = line[3:]
        self.path = Path('/sys/fs/cgroup') / self.relative.lstrip('/')
        if not self.path.name.endswith('.service') or self.path.resolve() != self.path:
            raise ValueError('An exact service leaf is required')
        self.inode, self.boot = self.path.stat().st_ino, _boot_id()
        props = subprocess.check_output(['systemctl', '--user', 'show', self.path.name,
            '--property=MainPID,KillMode,Restart'], text=True, timeout=2)
        values = dict(row.split('=', 1) for row in props.splitlines() if '=' in row)
        if values != {'MainPID': str(os.getpid()), 'KillMode': 'control-group', 'Restart': 'no'}:
            raise ValueError('Guard must be the actual non-restarting service MainPID')
        self.check()
        if self.members() != [os.getpid()]:
            raise ValueError('Dedicated service must start without other live workloads')
        # Open before workload launch; no privileged escalation or ancestor write.
        self.kill_fd = os.open(self.path / 'cgroup.kill', os.O_WRONLY)

    def check(self):
        if self.path.stat().st_ino != self.inode or _boot_id() != self.boot:
            raise ValueError('Owned service identity changed')
        def read(name):
            return (self.path / name).read_text().strip()
        quota, period = read('cpu.max').split()
        if (quota == 'max' or int(period) <= 0 or int(quota) > .25 * int(period)
                or not 0 < int(read('memory.max')) <= 768 * 2**20
                or not 0 < int(read('memory.high')) <= 640 * 2**20
                or int(read('memory.swap.max')) != 0 or not 0 < int(read('pids.max')) <= 64):
            raise ValueError('Original hard resource envelope is missing or wider')
        if os.getpid() not in self.members():
            raise ValueError('Guard left its actual owned service')

    def attach(self, process):
        if Path(f'/proc/{process.pid}/cgroup').read_text().strip() != '0::' + self.relative:
            raise ValueError('Issuer does not inherit the owned whole-service boundary')
        self.check()

    def members(self):
        return [int(v) for v in (self.path / 'cgroup.procs').read_text().split()]

    def stop(self):
        # Use the descriptor opened on the original leaf. A widened/malformed
        # current limit must not prevent terminating the already-owned unit.
        os.write(self.kill_fd, b'1')  # Includes the guard; systemd owns postmortem evidence.
        os._exit(125)

    def close(self):
        os.close(self.kill_fd)


def _lease_check(lease, issuer, previous=None):
    observation = {'wall': datetime.now(UTC).timestamp(), 'monotonic': time.monotonic(),
        'boot_id': _boot_id(), 'pid': lease.parent_pid, 'ppid': os.getpid(),
        'parent_pid': issuer.pid, 'parent_created': issuer.create_time(),
        'parent_alive': issuer.is_running() and issuer.status() not in {psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD}}
    return _evaluate(lease, observation, previous), observation


def _completed_clock(lease, previous):
    """No work permission: keep timeout protection while an exited issuer is reaped."""
    wall, mono, boot = datetime.now(UTC).timestamp(), time.monotonic(), _boot_id()
    observation = {'wall': wall, 'monotonic': mono, 'boot_id': boot,
                   'parent_alive': False, 'completed_issuer_exited': True}
    if boot != lease.boot_id:
        return LeaseCheck('boot_changed'), observation
    if (wall < datetime.fromisoformat(lease.wall_started_at).timestamp() or mono < lease.monotonic_started
            or (previous and (wall < previous['wall'] or mono < previous['monotonic']))):
        return LeaseCheck('clock_rollback'), observation
    remaining = min(datetime.fromisoformat(lease.wall_deadline).timestamp() - wall,
                    lease.monotonic_deadline - mono)
    return LeaseCheck('ok', remaining) if remaining > 0 else LeaseCheck('deadline'), observation


class IssuerGuard:
    """Use only inside the fixed dedicated bootstrap; cannot attach another PID.

    Arm immediately after canonical prepayment, before snapshot/numeric/provider
    work. Disarm only after existing publication and registration return. This
    one-work stage intentionally rejects any second arm in the same issuer.
    """
    def __init__(self, authority, reservation_id):
        if type(authority) is not MonthlyWorkAuthority:
            raise TypeError('Actual canonical monthly authority required')
        directory = os.environ.get('V7_STAGE_GUARD_DIRECTORY')
        if not directory:
            raise ValueError('Dedicated supervised issuer required; never attach interactive caller')
        self.directory = Path(directory).resolve()
        marker = _read(self.directory / 'bootstrap.json')
        me = psutil.Process(os.getpid())
        if (marker['issuer'] != _identity(me)
                or _ancestry(me, marker['supervisor']) != marker['ancestry']
                or marker['guard_source'] != _hash(__file__)):
            raise ValueError('Dedicated issuer or supervisor identity changed')
        self.authority, self.rid = authority, reservation_id
        self.original = authority.get(reservation_id)
        self.lease = FrozenWorkLease.from_dict(self.original['lease'])
        if (self.lease.parent_pid != me.pid or self.lease.parent_created != me.create_time()
                or self.original['status'] not in {'reserved', 'running'}):
            raise ValueError('Only this original reserved issuer can arm work')
        self.lease.assert_parent_alive()
        _write(self.directory / 'arm.json', {'authority_directory': str(authority.directory),
            'authority_manifest_hash': authority.manifest_hash, 'reservation_id': reservation_id,
            'lease': self.lease.to_dict(), 'lease_hash': self.lease.digest})
        self._wait('armed.json')

    def _wait(self, name):
        while not (self.directory / name).exists():
            self.lease.assert_parent_alive()
            if not psutil.pid_exists(_read(self.directory / 'bootstrap.json')['supervisor']['pid']):
                os._exit(125)
            time.sleep(.01)
        if _read(self.directory / name)['lease_hash'] != self.lease.digest:
            raise ValueError('Guard acknowledgement belongs to another original lease')

    def disarm(self):
        record = self.authority.get(self.rid)
        if record['lease'] != self.original['lease'] or record['status'] != 'completed':
            raise ValueError('Disarm requires original completed accounting, not unconsumed failure')
        self.lease.assert_parent_alive()
        # Completed forward work must already possess its original positive SQL
        # publication. Guard completion itself never grants model admission.
        purpose = record['binding'].get('purpose')
        if purpose in {'known_reference', 'pool_forward'}:
            from factorlab.discovery.forward_publication_witness import ForwardPublicationWitness
            from factorlab.discovery.known_linear_forward import KnownForwardTicket
            from factorlab.discovery.pool_forward import PoolForwardTicket
            cls = PoolForwardTicket if purpose == 'pool_forward' else KnownForwardTicket
            folder = 'pool_forward_tickets' if purpose == 'pool_forward' else 'known_forward_tickets'
            work = record['work_key'].split(':', 1)[1]
            ticket = cls.load(self.authority.directory / folder / work / 'ticket.json')
            proof = ForwardPublicationWitness.read_completed(ticket)
            from factorlab.discovery.neural_forward_admission import (
                _observation as verify_observation,
            )
            verify_observation(proof['observation'], self.lease)
            if ticket.value['lease'] != self.lease.to_dict():
                raise ValueError('Forward publication changed original issuer lease')
        self.lease.assert_parent_alive()
        _write(self.directory / 'disarm.json', {'reservation_id': self.rid, 'lease_hash': self.lease.digest})
        self._wait('disarmed.json')
        self.lease.assert_parent_alive()


def supervise(script, arguments, *, directory, source_root):
    """Fixed Python bootstrap; trusted local code only, no provider program input.

    The ten-second pre-arm abort is a liveness cap, NOT a spend grant. Production
    startup still belongs to the immutable outer controller envelope.
    """
    directory, script = Path(directory).resolve(), Path(script).resolve()
    if directory.exists() or not script.is_file():
        raise ValueError('Fresh supervisor directory and existing trusted entry required')
    directory.mkdir(parents=True)
    request = {'script': str(script), 'script_sha256': _hash(script), 'arguments': list(arguments),
        'source_root': str(Path(source_root).resolve()), 'guard_source': _hash(__file__)}
    _write(directory / 'request.json', request)
    boundary = _WindowsJob() if os.name == 'nt' else _LinuxService()
    worker = None
    lease = authority = record = previous = None
    state, reason, observed = 'failed', 'startup_failed', None
    terminated_members = []
    watcher = None
    watcher_stop = threading.Event()
    completion_ready = threading.Event()
    watch_failure = {}
    try:
        env = dict(os.environ, V7_STAGE_GUARD_DIRECTORY=str(directory))
        with (directory / 'worker.log').open('wb') as log:
            worker = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
                'bootstrap', str(directory)], stdout=log, stderr=subprocess.STDOUT, env=env,
                cwd=ROOT, creationflags=(getattr(subprocess, 'CREATE_NO_WINDOW', 0)
                                        | (0x0004 if os.name == 'nt' else 0)))
            boundary.attach(worker)
            start_limit = time.monotonic() + 10
            while not (directory / 'hello.json').exists():
                if worker.poll() is not None or time.monotonic() >= start_limit:
                    raise RuntimeError('Fixed bootstrap never identified its actual native process')
                time.sleep(.01)
            hello = _read(directory / 'hello.json')
            issuer = psutil.Process(hello['pid'])
            supervisor = _identity(psutil.Process(os.getpid()))
            ancestry = _ancestry(issuer, supervisor)
            if (hello != _identity(issuer) or issuer.pid not in boundary.members()
                    or not (worker.pid == issuer.pid or any(p['pid'] == worker.pid for p in ancestry))):
                raise ValueError('Bootstrap native PID does not belong to the launched owned tree')
            marker = {'issuer': _identity(issuer), 'supervisor': supervisor, 'ancestry': ancestry,
                      'guard_source': _hash(__file__), 'membership': boundary.members()}
            _write(directory / 'bootstrap.json', marker)
            while True:
                if lease is None and (directory / 'arm.json').exists():
                    arm = _read(directory / 'arm.json')
                    proposed = FrozenWorkLease.from_dict(arm['lease'])
                    if proposed.parent_pid != issuer.pid or proposed.parent_created != issuer.create_time():
                        raise ValueError('Proposed lease cannot name any unowned process')
                    # This watchdog belongs to the separate supervisor, never
                    # the issuer doing NumPy/Codex work. Start BEFORE canonical
                    # SQL/hash verification, so slow storage there cannot leave
                    # the owned issuer running beyond even the proposed lease.
                    def watch(original_lease=proposed):
                        prior = None
                        try:
                            while not watcher_stop.is_set():
                                try:
                                    checked, seen = _lease_check(original_lease, issuer, prior)
                                except psutil.NoSuchProcess:
                                    if completion_ready.is_set():
                                        checked, seen = _completed_clock(original_lease, prior)
                                    else:
                                        raise
                                if checked.reason == 'parent_lost' and completion_ready.is_set():
                                    checked, seen = _completed_clock(original_lease, prior)
                                if not checked.allowed:
                                    watch_failure.update(reason=checked.reason, observation=seen)
                                    boundary.stop()
                                    return
                                prior = seen
                                watcher_stop.wait(min(.02, checked.remaining_seconds))
                        except BaseException:  # noqa: BLE001 -- native ownership close is the fallback
                            os._exit(125)  # Job handle/systemd MainPID loss tears down owned work.
                    watcher = threading.Thread(target=watch, name='external-original-deadline', daemon=True)
                    watcher.start()
                    protocol.ROOT = Path(request['source_root'])
                    authority = _authority(arm['authority_directory'])
                    record = authority.get(arm['reservation_id'])
                    lease = FrozenWorkLease.from_dict(record['lease'])
                    if (arm != {'authority_directory': str(authority.directory),
                            'authority_manifest_hash': authority.manifest_hash,
                            'reservation_id': record['reservation_id'], 'lease': lease.to_dict(),
                            'lease_hash': lease.digest}
                            or record['status'] not in {'reserved', 'running'}
                            or lease.parent_pid != issuer.pid or lease.parent_created != issuer.create_time()):
                        raise ValueError('Guard request does not match original canonical prepayment')
                    check, observed = _lease_check(lease, issuer)
                    if not check.allowed:
                        reason = check.reason
                        break
                    previous = observed
                    _write(directory / 'armed.json', {'lease_hash': lease.digest,
                        'observation': observed, 'membership': boundary.members()})
                if lease is not None:
                    check, observed = _lease_check(lease, issuer, previous)
                    if not check.allowed:
                        reason = check.reason
                        break
                    previous = observed
                    if (directory / 'disarm.json').exists():
                        if _read(directory / 'disarm.json') != {
                                'reservation_id': record['reservation_id'], 'lease_hash': lease.digest}:
                            raise ValueError('Disarm identity changed')
                        current = authority.get(record['reservation_id'])
                        if current['status'] != 'completed' or current['lease'] != lease.to_dict():
                            raise ValueError('Canonical work did not finish on its original lease')
                        check, observed = _lease_check(lease, issuer, previous)
                        if not check.allowed:
                            reason = check.reason
                            break
                        completion_ready.set()
                        _write(directory / 'disarmed.json', {'lease_hash': lease.digest, 'observation': observed})
                        # Shutdown belongs to the same original deadline too.
                        worker.wait(timeout=min(2, check.remaining_seconds))
                        watcher_stop.set()
                        watcher.join(timeout=1)
                        if watcher.is_alive() or watch_failure:
                            raise RuntimeError('Original deadline watchdog closed during disarm')
                        if worker.returncode != 0:
                            raise RuntimeError('Issuer failed after disarming')
                        if (time.monotonic() >= lease.monotonic_deadline
                                or datetime.now(UTC) >= datetime.fromisoformat(lease.wall_deadline)):
                            raise RuntimeError('Issuer shutdown passed original deadline')
                        state, reason = 'completed', 'original_lease_disarmed'
                        break
                if worker.poll() is not None:
                    reason = 'issuer_exited_without_completion'
                    break
                if lease is None and time.monotonic() >= start_limit:
                    reason = 'no_canonical_arm'
                    break
                time.sleep(.02)
    except BaseException as error:  # noqa: BLE001 -- cleanup must cover native launch/control failures
        reason = type(error).__name__ + ':' + str(error)[:1000]
    finally:
        watcher_stop.set()
        if watcher is not None:
            watcher.join(timeout=6)
            if watcher.is_alive():
                os._exit(125)
        if watch_failure:
            state, reason, observed = 'failed', watch_failure['reason'], watch_failure['observation']
        # Stop before logging or accounting I/O: a failed receipt cannot defer kill.
        try:
            terminated_members = boundary.members()
            if (state != 'completed' or (os.name == 'nt' and terminated_members)
                    or (os.name != 'nt' and set(terminated_members) != {os.getpid()})):
                boundary.stop()
            if worker is not None and worker.poll() is None:
                worker.wait(timeout=5)
        finally:
            boundary.close()  # Windows kernel also kills the job if guard is killed.
        if authority is not None and record is not None and state != 'completed':
            try:
                current = authority.get(record['reservation_id'])
                if current['status'] in {'reserved', 'running'}:
                    authority.fail_consumed(record['reservation_id'], 'hard_guard:' + reason)
            except BaseException as error:  # noqa: BLE001 -- no annotation failure can undo debit
                reason += ';failure_annotation=' + type(error).__name__
        result = {'status': state, 'reason': reason, 'lease': None if lease is None else lease.to_dict(),
            'lease_hash': None if lease is None else lease.digest, 'observation': observed,
            'owned_members_at_cleanup': terminated_members,
            'platform': sys.platform, 'publication_authority': False}
        _write(directory / 'result.json', result)
    return result


def _bootstrap(directory):
    directory = Path(directory)
    if os.name != 'nt':
        parent = os.getppid()
        if ctypes.CDLL(None).prctl(1, 9, 0, 0, 0) != 0 or os.getppid() != parent:
            os._exit(125)
    end = time.monotonic() + 10
    parent = psutil.Process(os.getppid())
    _write(directory / 'hello.json', _identity(psutil.Process(os.getpid())))
    while not (directory / 'bootstrap.json').exists():
        if not parent.is_running() or time.monotonic() >= end:
            os._exit(125)
        time.sleep(.01)
    request = _read(directory / 'request.json')
    if request['guard_source'] != _hash(__file__) or request['script_sha256'] != _hash(request['script']):
        raise ValueError('Fixed issuer entry/source changed')
    protocol.ROOT = Path(request['source_root'])
    sys.argv = [request['script'], *request['arguments']]
    runpy.run_path(request['script'], run_name='__main__')


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == 'bootstrap':
        _bootstrap(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == 'supervise':
        args = _read(sys.argv[2])
        result = supervise(**args)
        raise SystemExit(0 if result['status'] == 'completed' else 124)
    else:
        raise SystemExit('Fixed supervisor/bootstrap entry required')
