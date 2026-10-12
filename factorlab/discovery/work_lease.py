"""Original work leases, not permission to spend a budget or launch a model.

Issue in the launcher BEFORE its reservation commits. The authority must bind
the exact serialized lease; a checksum alone is not authorization. Loading or
starting a worker never grants time. No Torch imports or process-tree signals.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psutil


def _boot_id():
    if sys.platform == 'linux':
        value = Path('/proc/sys/kernel/random/boot_id').read_text('ascii').strip()
        return 'linux:' + str(uuid.UUID(value))
    if sys.platform == 'win32':
        return 'windows:kuser-boot-v1:' + str(_windows_boot_sequence())
    raise OSError('Unsupported kernel boot identity')


def _windows_boot_sequence():
    """Read the OS loader's boot counter, not a wall-clock/uptime estimate.

    Microsoft ntddk.h documents BootId as incremented on each boot attempt.
    Its AMD64 KUSER_SHARED_DATA layout places the ULONG at 0x2c4; the
    user mapping is read-only at 0x7ffe0000 (Microsoft MSRC, 2022-04-05).
    Restrict this adapter to native 64-bit NT 10.0 / AMD64. No fallback,
    cache, randomness, registry writes, or inferred boot time is permitted.
    """
    import ctypes
    import struct

    if sys.platform != 'win32' or ctypes.sizeof(ctypes.c_void_p) != 8:
        raise OSError('Unsupported Windows boot identity architecture')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    read = kernel.ReadProcessMemory
    read.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                     ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
    read.restype = ctypes.c_int
    data = ctypes.create_string_buffer(0x2c8)
    size = ctypes.c_size_t()
    # -1 is GetCurrentProcess's documented pseudo-handle; no handle is opened.
    if not read(ctypes.c_void_p(-1), 0x7ffe0000, data, len(data), ctypes.byref(size)):
        raise OSError(ctypes.get_last_error(), 'Windows boot identity read failed')
    if size.value != len(data):
        raise OSError('Incomplete Windows boot identity read')
    architecture = struct.unpack_from('<H', data, 0x26a)[0]
    major, minor = struct.unpack_from('<II', data, 0x26c)
    sequence = struct.unpack_from('<I', data, 0x2c4)[0]
    if architecture != 9 or (major, minor) != (10, 0) or sequence == 0:
        raise OSError('Unsupported Windows boot identity layout or counter')
    return sequence


def _utc(value):
    if not isinstance(value, str):
        raise ValueError('Lease timestamps must be canonical UTC strings')  # noqa: TRY004
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset() != timedelta(0) or result.isoformat() != value:
        raise ValueError('Lease timestamps must be canonical UTC strings')
    return result


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


@dataclass(frozen=True)
class LeaseCheck:
    reason: str
    remaining_seconds: float = 0.

    @property
    def allowed(self):
        return self.reason == 'ok'

    @property
    def exit_code(self):
        return 0 if self.allowed else 124 if self.reason == 'deadline' else 125


class LeaseClosed(RuntimeError):
    def __init__(self, check):
        self.check = check
        super().__init__('Original work lease closed: ' + check.reason)


@dataclass(frozen=True)
class FrozenWorkLease:
    lease_seconds: int
    wall_started_at: str
    wall_deadline: str
    monotonic_started: float
    monotonic_deadline: float
    boot_id: str
    parent_pid: int
    parent_created: float
    version: str = 'work-lease-v1'

    def __post_init__(self):
        start, end = _utc(self.wall_started_at), _utc(self.wall_deadline)
        if (self.version != 'work-lease-v1' or type(self.lease_seconds) is not int
                or not 1 <= self.lease_seconds <= 300
                or end != start + timedelta(seconds=self.lease_seconds)
                or not _finite(self.monotonic_started) or self.monotonic_started < 0
                or not _finite(self.monotonic_deadline)
                or self.monotonic_deadline != self.monotonic_started + self.lease_seconds
                or type(self.parent_pid) is not int or self.parent_pid <= 0
                or not _finite(self.parent_created) or self.parent_created <= 0
                or self.parent_created > start.timestamp()
                or not isinstance(self.boot_id, str)):
            raise ValueError('Invalid original work lease')
        if self.boot_id.startswith('linux:'):
            valid = self.boot_id == 'linux:' + str(uuid.UUID(self.boot_id[6:]))
        elif self.boot_id.startswith('windows:kuser-boot-v1:'):
            text = self.boot_id[len('windows:kuser-boot-v1:'):]
            valid = (text.isascii() and text.isdigit() and str(int(text)) == text
                     and 0 < int(text) <= 0xffffffff)
        elif self.boot_id.startswith('windows:'):
            # Preserve old records for audit, never issue or continue this identity.
            text = self.boot_id[8:]
            valid = text.isascii() and text.isdigit() and str(int(text)) == text and int(text) > 0
        else:
            valid = False
        if not valid:
            raise ValueError('Invalid lease boot identity')

    @classmethod
    def issue(cls, seconds=300):
        """Trusted issuer process becomes the subsequent worker's exact parent."""
        if type(seconds) is not int or not 1 <= seconds <= 300:
            raise ValueError('Work lease requires integer seconds in 1..300')
        began = time.monotonic()  # Sample first; system inspection consumes lease time.
        wall = datetime.now(UTC)
        process = psutil.Process(os.getpid())
        return cls(seconds, wall.isoformat(), (wall + timedelta(seconds=seconds)).isoformat(),
                   began, began + seconds, _boot_id(), process.pid, process.create_time())

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError('Unknown or missing original lease fields')
        return cls(**value)

    @property
    def digest(self):
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        return hashlib.sha256(encoded).hexdigest()

    def check(self):
        """Read-only check for issuer or its direct worker; never renew time."""
        return _evaluate(self, _observe(self))

    def assert_parent_alive(self, *, as_worker=False):
        observation = _observe(self)
        check = _evaluate(self, observation)
        if as_worker and observation.get('pid') == self.parent_pid:
            check = LeaseCheck('wrong_process')
        if not check.allowed:
            raise LeaseClosed(check)
        return check

    def as_worker(self):
        return LeaseWatchdog(self)


WorkLease = FrozenWorkLease


def _observe(lease):
    try:
        parent = psutil.Process(lease.parent_pid)
        # Match issue()/serialized UTC microseconds. Mixing time.time() floats
        # with datetime's microsecond representation can invent a 1-ULP rollback.
        # This changes representation, not the strict deadline/rollback policy.
        return {'wall': datetime.now(UTC).timestamp(), 'monotonic': time.monotonic(), 'boot_id': _boot_id(),
                'pid': os.getpid(), 'ppid': os.getppid(), 'parent_pid': parent.pid,
                'parent_created': parent.create_time(),
                'parent_alive': parent.is_running() and parent.status() not in
                    {psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD}}
    except (OSError, ValueError, psutil.Error):
        return {'unavailable': True}


def _evaluate(lease, observation, previous=None):
    """Private pure test seam; production observations always come from _observe."""
    if observation.get('unavailable'):
        return LeaseCheck('identity_unavailable')
    wall, mono = observation['wall'], observation['monotonic']
    if not _finite(wall) or not _finite(mono):
        return LeaseCheck('clock_invalid')
    if ((lease.boot_id.startswith('windows:') and lease.boot_id[8:].isdigit())
            or observation['boot_id'] != lease.boot_id):
        return LeaseCheck('boot_changed')
    if (wall < _utc(lease.wall_started_at).timestamp() or mono < lease.monotonic_started
            or (previous is not None and (wall < previous['wall'] or mono < previous['monotonic']))):
        return LeaseCheck('clock_rollback')
    remaining = min(_utc(lease.wall_deadline).timestamp() - wall, lease.monotonic_deadline - mono)
    if remaining <= 0:
        return LeaseCheck('deadline')
    if (observation['parent_pid'] != lease.parent_pid or observation['parent_created'] != lease.parent_created
            or observation['parent_alive'] is not True):
        return LeaseCheck('parent_lost')
    if observation['pid'] != lease.parent_pid and observation['ppid'] != lease.parent_pid:
        return LeaseCheck('wrong_process')
    return LeaseCheck('ok', remaining)


class LeaseWatchdog:
    """Direct worker self-exit only; issuer cannot accidentally guard itself.

    This Python thread is not an OS hard-real-time guarantee. The enclosing
    launcher/cgroup must enforce its own deadline and resource policy as well.
    No public clock/identity/exit callback injection is accepted.
    """
    def __init__(self, lease):
        if not isinstance(lease, FrozenWorkLease):
            raise TypeError('A frozen original work lease is required')
        self.lease = lease
        self._stop = threading.Event()
        self._thread = None
        self._entered = False
        self._previous = None

    def _check(self):
        observation = _observe(self.lease)
        check = _evaluate(self.lease, observation, self._previous)
        if check.allowed:
            self._previous = observation
        return check

    def _watch(self):
        while not self._stop.is_set():
            check = self._check()
            if self._stop.is_set():
                return
            if not check.allowed:
                os._exit(check.exit_code)  # Only this worker, never parent/other PIDs.
            self._stop.wait(min(.05, check.remaining_seconds))

    def __enter__(self):
        if self._entered or os.getpid() == self.lease.parent_pid:
            raise ValueError('Work lease watchdog is single-use and direct-worker only')
        self._entered = True
        check = self._check()
        if not check.allowed:
            os._exit(check.exit_code)
        self._thread = threading.Thread(target=self._watch, name='original-work-lease', daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=.5)
