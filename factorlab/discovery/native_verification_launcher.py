"""One original task per fixed VerificationBatch slot; no durable proof.

The containing service must already have externally enforced shared limits.
This launcher does not grant time, create quota, reap, or enable controllers.
The typed original batch retains at most 23 children until service closure.
"""
from __future__ import annotations
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import threading

from factorlab.protocol import fingerprint, project_path

RELATIVE = 'factorlab/discovery/native_verification_launcher.py'
WORKER = 'factorlab.discovery.native_verification_worker'
_KEY = object()


class LaunchFailure(RuntimeError):
    """Consumed failure; .launch retains the original process and live handles."""
    def __init__(self, error, launch):
        super().__init__(str(error))
        self.launch = launch


def _components():
    from . import native_verification_terminal as terminal
    from .native_verification_bridge import NativeVerificationAttempt
    return terminal, NativeVerificationAttempt


def _original_deadline(task):
    from .nominal_verification_budget import VerificationTask
    from .nominal_verification_deadline import OriginalTaskDeadline
    if type(task) is not VerificationTask:
        raise TypeError('Exact original acquired VerificationTask required')
    return OriginalTaskDeadline(task)


def _source(task):
    from .monthly_work_authority import source_files
    from .evidence import file_hash
    terminal, bridge = _components()
    files = terminal._installed(task)
    modules = ((RELATIVE, __file__),
        ('factorlab/discovery/native_verification_bridge.py', sys.modules[bridge.__module__].__file__),
        ('factorlab/discovery/native_verification_terminal.py', terminal.__file__))
    if files != source_files():
        raise ValueError('Original source inventory changed')
    for relative, path in modules:
        if Path(path).resolve() != project_path(relative).resolve() or files.get(relative) != file_hash(path):
            raise ValueError('Canonical launcher/bridge/terminal source required')
    if 'factorlab/discovery/native_verification_worker.py' not in files:
        raise ValueError('Fixed worker source absent')
    task.check()


def _environment():
    # No inherited import injection or credentials. Runtime is the same executable.
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'LC_ALL', 'TZ') if k in os.environ}
    env.update({'FACTOR_RESEARCH_ROOT': str(project_path('.')),
        'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1',
        'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1'})
    return env


def _new_group(task, terminal):
    parent = terminal._group(os.getpid())
    root = Path('/sys/fs/cgroup')
    shared = None
    for candidate in (parent, *parent.parents):
        if not candidate.is_relative_to(root):
            break
        try:
            if terminal._limits(terminal._values(candidate)):
                shared = candidate
                break
        except OSError:
            continue
    if shared is None:
        raise ValueError('Existing aggregate CPU100/640soft768hard/swap0/tasks64 required')
    # Ancestor resource limits already govern the inclusive subtree. Enabling
    # child controllers is unnecessary and would conflict with an issuer
    # residing directly in this domain (the no-internal-process constraint).
    if terminal._bounded(shared / 'cgroup.type').strip() != 'domain':
        raise ValueError('Original shared domain cgroup required')
    # Keep the worker inside the original unit for KillMode=control-group.
    name = 'verification-' + str(task.task['index']) + '-' + fingerprint(task.intent)
    worker = parent / name
    worker.mkdir(mode=0o700, exist_ok=False)  # No reuse, even after a partial failure.
    terminal._group_path(worker)
    actual = terminal._shared(worker, os.getpid())
    if actual['path'] != str(shared):
        raise ValueError('Shared enforcement identity changed')
    if terminal._subtree(worker)['members']:
        raise ValueError('New exclusive worker cgroup is not empty')
    return worker, worker.stat().st_ino, actual


def _join(launch):
    terminal = launch.terminal
    worker = terminal._group_path(launch.worker_cgroup)
    if worker.stat().st_ino != launch.group_inode:
        raise ValueError('Exclusive group identity changed')
    proc = terminal._proc(launch.process.pid)
    if proc['ppid'] != launch.owner[0] or proc['state'] == 'Z':
        raise ValueError('Original live direct child required')
    launch.child_identity = proc
    # O_NOFOLLOW protects the actual mutation, rather than a preceding path check.
    directory_fd = os.open(worker, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        if os.fstat(directory_fd).st_ino != launch.group_inode:
            raise ValueError('Exclusive group descriptor identity changed')
        fd = os.open('cgroup.procs', os.O_WRONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        try:
            raw = (str(launch.process.pid) + '\n').encode('ascii')
            if os.write(fd, raw) != len(raw):
                raise OSError('Incomplete cgroup admission')
        finally:
            os.close(fd)
    finally:
        os.close(directory_fd)
    after = terminal._proc(launch.process.pid)
    if (after['ppid'], after['start_ticks']) != (proc['ppid'], proc['start_ticks']):
        raise ValueError('Direct child changed during admission')
    if terminal._group(launch.process.pid) != worker or worker.stat().st_ino != launch.group_inode:
        raise ValueError('Actual worker cgroup admission missing')
    if terminal._shared(worker, launch.owner[0]) != launch.shared:
        raise ValueError('Actual aggregate enforcement changed')


class NativeVerificationLaunch:
    def __init__(self, key, task, guard, batch):
        if key is not _KEY:
            raise TypeError('Use launch_original_task with an acquired original task')
        self.task = task
        self.guard = guard
        self.batch = batch
        self.owner = (os.getpid(), threading.get_ident())
        self.terminal = self.attempt = self.process = self.handle = None
        self.channel = self.child_channel = None
        self.worker_cgroup = self.group_inode = self.shared = None
        self.pidfd = self.pidfd_identity = self.child_identity = None
        self.aborted = False
        self.completed = False
        self.cleanup_errors = []

    def __reduce__(self):
        raise TypeError('Live launch authority cannot be serialized')

    def _owner(self):
        if self.owner != (os.getpid(), threading.get_ident()):
            raise ValueError('Original parent and thread required')

    def _remaining(self):
        self._owner()
        if self.aborted:
            raise ValueError('Failed launch remains consumed')
        return min(self.task.lease.assert_parent_alive().remaining_seconds,
                   self.task.service.check().remaining_seconds)

    def abort(self, reason):
        """Irreversible poison before exact pidfd-only signal; never reap."""
        self._owner()
        try:
            self._abort_contained(reason)
        finally:
            # Keep the original absolute timer active throughout failure writes
            # and containment cleanup. Deletion failure never grants success.
            self.guard.close()

    def _abort_contained(self, reason):
        self.aborted = True
        if self.attempt is not None:
            self.attempt.state.poisoned = True
        if self.handle is not None:
            self.handle.invalidate(str(reason))
        try:
            self.task.fail(str(reason)[:900])
        except BaseException as error:
            self.cleanup_errors.append('failure_record:' + type(error).__name__)
        # pidfd identity stays open for this process lifetime; no numeric-PID kill.
        if self.pidfd is not None:
            try:
                st = os.fstat(self.pidfd)
                if (st.st_dev, st.st_ino) != self.pidfd_identity:
                    raise ValueError('Retained cleanup pidfd identity changed')
                signal.pidfd_send_signal(self.pidfd, signal.SIGKILL, None, 0)
            except ProcessLookupError:
                pass
            except BaseException as error:
                self.cleanup_errors.append('pidfd_signal:' + type(error).__name__)
        # Descendants are not silently declared cleaned: the external aggregate
        # supervisor remains responsible if a fixed worker unexpectedly forks.
        for channel in (self.channel, self.child_channel):
            if channel is not None:
                try:
                    channel.close()
                except OSError as error:
                    self.cleanup_errors.append('channel_close:' + type(error).__name__)

    def serve_until_completion(self):
        """Exactly check/consume/ready, then retained kernel readiness and proof.

        Parent semantic execution remains inside its ORIGINAL absolute OS timer.
        The enclosing service still requires external aggregate supervision;
        timer staging is not evidence of tested native scheduling or WCET.
        """
        self._owner()
        if self.completed or self.aborted:
            raise ValueError('One consumed launch execution only')
        try:
            for expected in ('check', 'consume', 'publication_ready'):
                self._remaining()
                if self.attempt.serve_one() != expected:
                    raise ValueError('Fixed original worker operation sequence required')
            remaining = self._remaining()
            ready, _, _ = select.select([self.pidfd], [], [], remaining)
            self._remaining()
            if ready != [self.pidfd]:
                raise TimeoutError('Original task deadline reached without child exit')
            result = self.attempt.observe_completion()
            if result.get('status') != 'single_task_publication_bound_to_live_exit':
                raise ValueError('Original live completion unavailable')
            self.batch._publish_task(self, result)
            self.guard.close()
            self.batch._commit_task(self)
            self.completed = True
            return result  # No required evidence writes after this observation.
        except BaseException as error:
            try:
                self.abort(type(error).__name__ + ':' + str(error))
            except BaseException as cleanup_error:
                error.add_note('Original deadline/cleanup failed: ' + type(cleanup_error).__name__)
            raise LaunchFailure(error, self) from error

    def read_completion(self):
        self._owner()
        if self.aborted or not self.completed:
            raise ValueError('No completed live launch')
        return self.attempt.read_completion()


def launch_original_task(task, *, batch):
    """Linux-only source-bound launcher; never allocate or renew any task."""
    if sys.platform != 'linux' or not all(hasattr(os, n) for n in ('pidfd_open', 'waitid')):
        raise RuntimeError('Linux live pidfd/waitid required; Windows unsupported')
    if not hasattr(signal, 'pidfd_send_signal'):
        raise RuntimeError('Exact pidfd cleanup API required')
    guard = _original_deadline(task)  # Exact type + original source-bound issuer.
    launch = None
    claimed = False
    try:
        from .native_verification_batch import VerificationBatch
        if type(batch) is not VerificationBatch:
            raise TypeError('Exact original VerificationBatch required')
        batch._claim_task(task, guard)
        claimed = True
        launch = NativeVerificationLaunch(_KEY, task, guard, batch)
        batch._retain_task(launch)
        _source(task)  # All heavy source/parent verification is under the timer.
        terminal, bridge = _components()
        launch.terminal = terminal
        launch.worker_cgroup, launch.group_inode, launch.shared = _new_group(task, terminal)
        launch.attempt = bridge.prepare(task, worker_cgroup=launch.worker_cgroup)
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        launch.channel, launch.child_channel = parent, child
        for channel in (parent, child):
            channel.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
        child_fd = child.fileno()
        launch._remaining()
        launch.process = subprocess.Popen(launch.attempt.expected_argv(child_fd),
            shell=False, pass_fds=(child_fd,), close_fds=True,
            cwd=str(project_path('.')), env=_environment(), stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Pin immediately; the child waits for the authenticated original gate.
        launch.pidfd = os.pidfd_open(launch.process.pid, 0)
        st = os.fstat(launch.pidfd)
        launch.pidfd_identity = (st.st_dev, st.st_ino)
        child.close()
        launch.child_channel = None
        _join(launch)
        launch._remaining()
        launch.attempt.bind_child(launch.process, parent, child_fd=child_fd)
        launch.handle = terminal.attach_original_child(task, launch.process,
            worker_cgroup=launch.worker_cgroup)
        launch.attempt.bind_terminal_handle(launch.handle)
        launch.attempt.release_start_gate()
        return launch
    except BaseException as error:
        try:
            if launch is not None:
                launch.abort(type(error).__name__ + ':' + str(error))
            else:
                try:
                    if claimed:
                        task.fail(type(error).__name__ + ':' + str(error)[:900])
                finally:
                    guard.close()
        except BaseException as cleanup_error:
            error.add_note('Original deadline/cleanup failed: ' + type(cleanup_error).__name__)
        raise LaunchFailure(error, launch) from error
