"""Linux x86_64 original-issuer absolute monotonic termination timer.

Default SIGALRM terminates the bounded issuer if its original task expires,
even when a synchronous consumer never returns to Python deadline checks.
No new work time, child launch, model quota or semantic acceptance is created.
Kernel scheduling is not a hard-real-time guarantee; acceptance still requires
actual completion observations strictly before both original deadlines.
"""
from __future__ import annotations
import ctypes
from fractions import Fraction
import math
import os
from pathlib import Path
import signal
import sys
import threading

RELATIVE = 'factorlab/discovery/nominal_verification_deadline.py'
CLOCK_MONOTONIC = 1
TIMER_ABSTIME = 1
_ACTIVE = None


def _absolute_ns(value):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value > 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError('Original finite positive absolute deadline required')
    # Round toward an earlier deadline, never grant rounding time.
    exact = Fraction(value) * 10**9
    return exact.numerator // exact.denominator


class _Timespec(ctypes.Structure):
    _fields_ = [('tv_sec', ctypes.c_int64), ('tv_nsec', ctypes.c_int64)]


class _Itimerspec(ctypes.Structure):
    _fields_ = [('it_interval', _Timespec), ('it_value', _Timespec)]


def _spec(value):
    ns = _absolute_ns(value)
    seconds, nanos = divmod(ns, 10**9)
    if seconds >= 2**63:
        raise ValueError('Original deadline outside fixed native ABI')
    return _Itimerspec(_Timespec(0, 0), _Timespec(seconds, nanos))


def _libc():
    if sys.platform != 'linux' or os.uname().machine != 'x86_64':
        raise OSError('Only registered Linux x86_64 timer ABI is supported')
    lib = ctypes.CDLL(None, use_errno=True)
    lib.timer_create.argtypes = [ctypes.c_int, ctypes.c_void_p,
                                ctypes.POINTER(ctypes.c_void_p)]
    lib.timer_create.restype = ctypes.c_int
    lib.timer_settime.argtypes = [ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(_Itimerspec), ctypes.c_void_p]
    lib.timer_settime.restype = ctypes.c_int
    lib.timer_delete.argtypes = [ctypes.c_void_p]
    lib.timer_delete.restype = ctypes.c_int
    return lib


def _check_native(rc, operation):
    if rc != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, 'Original verification ' + operation + ' failed')


class OriginalTaskDeadline:
    """One task-owned absolute OS timer, no callable handler or override.

    This intentionally terminates only its own bounded verification issuer.
    It refuses an existing alarm, blocked/custom SIGALRM, nested ownership or
    unsupported host. The full enclosing service still needs original7200
    supervision and shared resource limits, separately from this task timer.
    """
    def __init__(self, task):
        global _ACTIVE
        from .nominal_verification_budget import VerificationTask
        from factorlab.protocol import project_path
        from .evidence import file_hash
        if (type(task) is not VerificationTask or sys.platform != 'linux'
                or threading.current_thread() is not threading.main_thread()
                or os.getpid() != task.lease.parent_pid):
            raise TypeError('Exact original bounded Linux issuer main thread required')
        if _ACTIVE is not None:
            raise ValueError('Original task timer already owned; no nested reset')
        if (Path(__file__).resolve() != project_path(RELATIVE).resolve()
                or task.service.registry.plan.binding['source_files'].get(RELATIVE) != file_hash(__file__)):
            raise ValueError('Canonical original source-bound task timer required')
        if (signal.getsignal(signal.SIGALRM) != signal.SIG_DFL
                or signal.SIGALRM in signal.pthread_sigmask(signal.SIG_BLOCK, set())
                or signal.getitimer(signal.ITIMER_REAL) != (0., 0.)):
            raise ValueError('Foreign, ignored or blocked alarm cannot be replaced')
        task.lease.assert_parent_alive()
        service_check = task.service.check()
        if service_check.remaining_seconds <= 0:
            raise ValueError('Original service deadline reached')
        deadline = min(task.lease.monotonic_deadline,
                       task.service.clock.supervisor().monotonic_deadline)
        spec = _spec(deadline)
        self.task = task
        self.owner = (os.getpid(), threading.get_ident())
        self.closed = False
        self._lib = _libc()
        self._timer = ctypes.c_void_p()
        # NULL sigevent means SIGALRM with default disposition on this target;
        # no Python signal callback can defer expiration until consumer return.
        _check_native(self._lib.timer_create(CLOCK_MONOTONIC, None,
                                            ctypes.byref(self._timer)), 'timer_create')
        _ACTIVE = self
        try:
            _check_native(self._lib.timer_settime(self._timer, TIMER_ABSTIME,
                                                 ctypes.byref(spec), None), 'timer_settime')
            task.lease.assert_parent_alive()
            task.service.check()
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                error.add_note('Original timer cleanup failed: ' + type(cleanup_error).__name__)
            raise

    def close(self):
        global _ACTIVE
        if self.closed:
            return
        if self.owner != (os.getpid(), threading.get_ident()) or _ACTIVE is not self:
            raise ValueError('Only original task timer owner may release it')
        # Failure to delete does not reset ownership or imply safe completion.
        _check_native(self._lib.timer_delete(self._timer), 'timer_delete')
        self.closed = True
        _ACTIVE = None

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()

    def __reduce__(self):
        raise TypeError('Original OS timer ownership cannot serialize or resume')
