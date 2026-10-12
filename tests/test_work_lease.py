"""Small CPU process checks and pure lease boundaries; no Torch/network/market."""
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psutil
import pytest

from factorlab.discovery import work_lease as module
from factorlab.discovery.work_lease import FrozenWorkLease, LeaseClosed, LeaseWatchdog


def observation(lease):
    return {'wall': datetime.fromisoformat(lease.wall_started_at).timestamp(),
            'monotonic': lease.monotonic_started, 'boot_id': lease.boot_id,
            'pid': lease.parent_pid + 1, 'ppid': lease.parent_pid, 'parent_pid': lease.parent_pid,
            'parent_created': lease.parent_created, 'parent_alive': True}


def test_issue_identity_comes_from_system_and_roundtrip_does_not_renew():
    lease = FrozenWorkLease.issue(10)
    assert lease.parent_pid == os.getpid()
    assert lease.parent_created == psutil.Process().create_time()
    assert lease.check().allowed and lease.assert_parent_alive().allowed
    value = json.loads(json.dumps(lease.to_dict()))
    restored = FrozenWorkLease.from_dict(value)
    assert restored == lease and restored.digest == lease.digest
    assert restored.monotonic_deadline == lease.monotonic_deadline
    with pytest.raises(FrozenInstanceError):
        restored.lease_seconds = 300
    with pytest.raises(LeaseClosed, match='wrong_process'):
        lease.assert_parent_alive(as_worker=True)
    with pytest.raises(ValueError, match='direct-worker'), LeaseWatchdog(lease):
        pytest.fail('Issuer cannot accidentally install a self-killing fixture')


def test_same_wall_instant_with_time_float_one_ulp_below_canonical_is_not_rollback(monkeypatch):
    wall = [datetime.now(UTC)]

    class CanonicalClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return wall[0]

    monkeypatch.setattr(module, 'datetime', CanonicalClock)
    # Reproduce the observed Windows representation boundary deterministically:
    # time.time's float is 238 ns lower while datetime's canonical instant is unchanged.
    monkeypatch.setattr(module.time, 'time', lambda: math.nextafter(wall[0].timestamp(), -math.inf))
    lease = FrozenWorkLease.issue(10)
    observed = module._observe(lease)
    assert module.time.time() < datetime.fromisoformat(lease.wall_started_at).timestamp()
    assert observed['wall'] == datetime.fromisoformat(lease.wall_started_at).timestamp()
    assert module._evaluate(lease, observed).allowed
    original = lease.to_dict()
    wall[0] -= timedelta(microseconds=1)
    assert module._evaluate(lease, module._observe(lease)).reason == 'clock_rollback'
    assert lease.to_dict() == original  # No tolerance, renewal or deadline mutation.


def test_one_thousand_real_issue_observe_samples_have_no_precision_only_rollback():
    deltas, failures = [], []
    for index in range(1000):
        lease = FrozenWorkLease.issue(300)
        observed = module._observe(lease)
        check = module._evaluate(lease, observed)
        deltas.append(observed['wall'] - datetime.fromisoformat(lease.wall_started_at).timestamp())
        if not check.allowed:
            failures.append({'iteration': index, 'reason': check.reason, 'wall_delta': deltas[-1]})
    receipt = {'samples': 1000, 'min_wall_delta': min(deltas),
               'max_wall_delta': max(deltas), 'failures': failures}
    print(json.dumps(receipt, sort_keys=True))
    assert not failures, receipt
    assert min(deltas) >= 0


@pytest.mark.parametrize('seconds', [True, False, 0, -1, 301, 1.5, float('nan'), '10'])
def test_invalid_lease_durations_are_rejected(seconds):
    with pytest.raises(ValueError):
        FrozenWorkLease.issue(seconds)


@pytest.mark.parametrize('damage', ['unknown', 'missing', 'wall_extension', 'mono_extension', 'naive',
                                   'nonutc', 'nan', 'bool_pid', 'negative_created', 'boot', 'version'])
def test_serialized_original_contract_is_strict(damage):
    value = FrozenWorkLease.issue(10).to_dict()
    if damage == 'unknown':
        value['verified'] = True
    elif damage == 'missing':
        del value['boot_id']
    elif damage == 'wall_extension':
        value['wall_deadline'] = (datetime.fromisoformat(value['wall_deadline']) + timedelta(seconds=1)).isoformat()
    elif damage == 'mono_extension':
        value['monotonic_deadline'] += 1
    elif damage == 'naive':
        value['wall_started_at'] = '2026-01-01T00:00:00'
    elif damage == 'nonutc':
        value['wall_started_at'] = '2026-01-01T00:00:00+01:00'
    elif damage == 'nan':
        value['monotonic_started'] = float('nan')
    elif damage == 'bool_pid':
        value['parent_pid'] = True
    elif damage == 'negative_created':
        value['parent_created'] = -1
    elif damage == 'boot':
        value['boot_id'] = 'linux:caller-says-verified'
    else:
        value['version'] = 'renewable'
    with pytest.raises(ValueError):
        FrozenWorkLease.from_dict(value)


@pytest.mark.parametrize(('changed', 'reason'), [
    ({'boot_id': 'other-boot'}, 'boot_changed'), ({'parent_alive': False}, 'parent_lost'),
    ({'parent_created': 0.}, 'parent_lost'), ({'ppid': 0}, 'wrong_process'),
    ({'wall': float('inf')}, 'clock_invalid'), ({'unavailable': True}, 'identity_unavailable'),
])
def test_pure_system_observation_closes_identity_failures(changed, reason):
    lease = FrozenWorkLease.issue(10)
    check = module._evaluate(lease, observation(lease) | changed)
    assert not check.allowed and check.reason == reason and check.exit_code == 125


def test_original_deadlines_and_rollback_before_first_watch_cannot_extend_work():
    lease = FrozenWorkLease.issue(10)
    first = observation(lease)
    for clock in ('wall', 'monotonic'):
        at_end = first | {clock: first[clock] + 10}
        assert module._evaluate(lease, at_end).reason == 'deadline'
        before = first | {clock: first[clock] + 9.9}
        assert 0 < module._evaluate(lease, before).remaining_seconds < .11
        assert module._evaluate(lease, first | {clock: first[clock] - 1}).reason == 'clock_rollback'
    # Later rollback still above the original start is visible to the watchdog.
    previous = first | {'wall': first['wall'] + 5, 'monotonic': first['monotonic'] + 5}
    next_observation = first | {'wall': first['wall'] + 4, 'monotonic': first['monotonic'] + 6}
    assert module._evaluate(lease, next_observation, previous).reason == 'clock_rollback'


def test_no_caller_verified_or_public_observer_callback_can_authorize_a_lease():
    with pytest.raises(TypeError):
        FrozenWorkLease.issue(1, verified=True)
    with pytest.raises(TypeError):
        LeaseWatchdog(FrozenWorkLease.issue(1), exit_process=lambda _: None)


CHILD = '''
import json,sys,time,os
from pathlib import Path
import psutil
from factorlab.discovery.work_lease import FrozenWorkLease
lease=FrozenWorkLease.from_dict(json.loads(sys.argv[1]))
mode=sys.argv[2]
try:
    with lease.as_worker() as guard:
        assert 'torch' not in sys.modules
        if mode=='parent':
            Path(sys.argv[3]).write_text(json.dumps({'pid':os.getpid(),'created':psutil.Process().create_time(),
                'parent_pid':os.getppid(),'parent_created':psutil.Process(os.getppid()).create_time()}))
        if mode=='normal':
            time.sleep(.02)
        elif mode=='exception':
            raise RuntimeError('synthetic')
        else:
            time.sleep(20)
except RuntimeError:
    assert mode=='exception'
assert not guard._thread.is_alive()
time.sleep(max(0,lease.monotonic_deadline-time.monotonic())+.1)
assert 'torch' not in sys.modules
'''


def child_env():
    return {**os.environ, 'PYTHONPATH': os.pathsep.join((str(Path(__file__).resolve().parents[1]),
             str(Path(psutil.__file__).resolve().parents[1]))), 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}


@pytest.mark.parametrize(('mode', 'expected'), [('deadline', 124), ('normal', 0), ('exception', 0)])
def test_real_child_expiry_or_clean_context_shutdown(mode, expected):
    executable = getattr(sys, '_base_executable', sys.executable)
    lease = FrozenWorkLease.issue(2)
    result = subprocess.run([executable, '-B', '-c', CHILD, json.dumps(lease.to_dict()), mode],
        env=child_env(), capture_output=True, timeout=7, check=False)
    assert result.returncode == expected, result.stderr.decode(errors='replace')


def test_real_delayed_child_cannot_restart_an_already_expired_lease():
    executable = getattr(sys, '_base_executable', sys.executable)
    lease = FrozenWorkLease.issue(1)
    time.sleep(1.05)
    result = subprocess.run([executable, '-B', '-c', CHILD, json.dumps(lease.to_dict()), 'deadline'],
        env=child_env(), capture_output=True, timeout=3, check=False)
    assert result.returncode == 124, result.stderr.decode(errors='replace')


def test_exact_parent_death_causes_owned_worker_self_exit(tmp_path):
    executable = getattr(sys, '_base_executable', sys.executable)
    marker = tmp_path / 'ready.json'
    parent_code = '''
import json,subprocess,sys
from factorlab.discovery.work_lease import FrozenWorkLease
lease=FrozenWorkLease.issue(10)
p=subprocess.Popen([sys.executable,'-B','-c',sys.argv[1],json.dumps(lease.to_dict()),'parent',sys.argv[2]])
p.wait(timeout=15)
'''
    parent = subprocess.Popen([executable, '-B', '-c', parent_code, CHILD, str(marker)],
        env=child_env(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    child = None
    try:
        until = time.monotonic() + 4
        while not marker.exists() and time.monotonic() < until:
            time.sleep(.02)
        assert marker.exists(), 'Owned child failed to start'
        identity = json.loads(marker.read_text())
        assert identity['parent_pid'] == parent.pid
        assert identity['parent_created'] == psutil.Process(parent.pid).create_time()
        child = psutil.Process(identity['pid'])
        assert child.create_time() == identity['created']
        parent.kill()  # Retained Popen handle only; no process search.
        parent.wait(timeout=2)
        code = child.wait(timeout=3)
        # POSIX non-child reaping can return None; Windows returns actual exit.
        assert code in (125, None)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=2)
        if child is not None:
            try:
                if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                    child.kill()
                    child.wait(timeout=2)
            except psutil.NoSuchProcess:
                pass
