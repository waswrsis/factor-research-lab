"""Windows boot-instance identity and unchanged lease boundaries; no research."""
import ctypes
import json
import struct
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from factorlab.discovery import work_lease as module


def kernel_page(monkeypatch, *, sequence=108, architecture=9, major=10, minor=0,
                success=True, complete=True):
    data = bytearray(0x2c8)
    struct.pack_into('<H', data, 0x26a, architecture)
    struct.pack_into('<II', data, 0x26c, major, minor)
    struct.pack_into('<I', data, 0x2c4, sequence)

    def read(handle, address, destination, length, count):
        assert handle.value == ctypes.c_void_p(-1).value
        assert address == 0x7ffe0000 and length == len(data)
        ctypes.memmove(destination, bytes(data), len(data))
        ctypes.cast(count, ctypes.POINTER(ctypes.c_size_t))[0] = length if complete else length - 1
        return int(success)

    monkeypatch.setattr(module, 'sys', SimpleNamespace(platform='win32'))
    monkeypatch.setattr(ctypes, 'WinDLL', lambda *a, **k: SimpleNamespace(ReadProcessMemory=read), raising=False)
    monkeypatch.setattr(ctypes, 'get_last_error', lambda: 5, raising=False)


def lease(boot='windows:kuser-boot-v1:108'):
    return module.FrozenWorkLease(30, '2026-01-01T00:00:00+00:00',
                                 '2026-01-01T00:00:30+00:00', 100., 130., boot, 123, 1.)


def observed(value, **changes):
    result = {'wall': datetime.fromisoformat(value.wall_started_at).timestamp(),
                  'monotonic': 100., 'boot_id': value.boot_id, 'pid': 124, 'ppid': 123,
                  'parent_pid': 123, 'parent_created': 1., 'parent_alive': True}
    return dict(result, **changes)


def test_wall_minus_uptime_drift_cannot_change_kernel_boot_identity(monkeypatch):
    kernel_page(monkeypatch)
    estimates = iter([1790773011.4840178, 1790773009.739192, 1790773000.0])
    monkeypatch.setattr(module.psutil, 'boot_time', lambda: next(estimates))
    identities = []
    for _ in range(3):
        module.psutil.boot_time()  # Artificially drift the old estimator.
        identities.append(module._boot_id())
    assert identities == ['windows:kuser-boot-v1:108'] * 3
    value = lease(identities[0])
    assert module._evaluate(value, observed(value, boot_id=identities[-1])).allowed


def test_new_loader_boot_sequence_invalidates_old_lease_without_renewal(monkeypatch):
    kernel_page(monkeypatch, sequence=108)
    value = lease(module._boot_id())
    original = value.to_dict()
    kernel_page(monkeypatch, sequence=109)
    assert module._evaluate(value, observed(value, boot_id=module._boot_id())).reason == 'boot_changed'
    assert value.to_dict() == original


@pytest.mark.parametrize('change', [{'success': False}, {'complete': False}, {'sequence': 0},
                                   {'architecture': 12}, {'major': 11}, {'minor': 1}])
def test_unknown_or_failed_kernel_read_refuses_identity(monkeypatch, change):
    kernel_page(monkeypatch, **change)
    with pytest.raises(OSError):
        module._boot_id()
    # A read failure closes both the original observer and a new issuance.
    actual = module.FrozenWorkLease(30, '2026-01-01T00:00:00+00:00',
        '2026-01-01T00:00:30+00:00', 100., 130., 'windows:kuser-boot-v1:108',
        module.os.getpid(), 1.)
    assert module._observe(actual) == {'unavailable': True}
    with pytest.raises(OSError):
        module.FrozenWorkLease.issue(30)


def test_unknown_pointer_width_refuses_identity(monkeypatch):
    kernel_page(monkeypatch)
    monkeypatch.setattr(ctypes, 'sizeof', lambda _: 4)
    with pytest.raises(OSError, match='architecture'):
        module._boot_id()


def test_legacy_numeric_windows_lease_remains_readable_but_never_allowed():
    value = lease('windows:1790773011')
    record = value.to_dict()
    restored = module.FrozenWorkLease.from_dict(json.loads(json.dumps(record)))
    assert restored.to_dict() == record and restored.digest == value.digest
    assert module._evaluate(restored, observed(restored)).reason == 'boot_changed'
    assert module._evaluate(restored, observed(restored, boot_id='windows:kuser-boot-v1:108')).reason == 'boot_changed'


@pytest.mark.parametrize('suffix', ['0', '-1', '+1', '01', '4294967296', '1.0', '', '１２'])
def test_new_identity_is_a_canonical_nonzero_uint32(suffix):
    with pytest.raises(ValueError):
        lease('windows:kuser-boot-v1:' + suffix)


def test_wall_rollback_deadline_and_parent_guards_still_close():
    value = lease()
    wall = datetime.fromisoformat(value.wall_started_at).timestamp()
    for changes, reason in [({'wall': wall - 1}, 'clock_rollback'),
                            ({'monotonic': 99}, 'clock_rollback'),
                            ({'wall': wall + 30}, 'deadline'),
                            ({'monotonic': 130}, 'deadline'),
                            ({'parent_created': 2.}, 'parent_lost'),
                            ({'parent_alive': False}, 'parent_lost'),
                            ({'ppid': 999}, 'wrong_process')]:
        assert module._evaluate(value, observed(value, **changes)).reason == reason
    assert value.lease_seconds == 30 and value.monotonic_deadline == 130.


def test_linux_proc_uuid_branch_remains_canonical(monkeypatch):
    monkeypatch.setattr(module, 'sys', SimpleNamespace(platform='linux'))
    seen = []

    def read(path, encoding):
        seen.append((str(path), encoding))
        return '12345678-1234-1234-1234-123456789abc\n'

    monkeypatch.setattr(module.Path, 'read_text', read)
    assert module._boot_id() == 'linux:12345678-1234-1234-1234-123456789abc'
    assert seen == [(str(Path('/proc/sys/kernel/random/boot_id')), 'ascii')]


@pytest.mark.skipif(sys.platform != 'win32', reason='Actual Windows shared kernel page')
def test_actual_kernel_identity_agrees_in_fresh_processes_without_clock_estimator():
    expected = module._boot_id()
    code = "from factorlab.discovery.work_lease import _boot_id; import psutil; psutil.boot_time=lambda: (_ for _ in ()).throw(AssertionError('no estimator')); print(_boot_id())"
    for _ in range(2):
        result = subprocess.run([sys.executable, '-B', '-c', code],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True,
                                text=True, check=True, timeout=20)
        assert result.stdout.strip() == expected
    assert expected.startswith('windows:kuser-boot-v1:')
