"""Small owned CPU children only; no Torch, kernel trial, GPT or market work."""
import datetime
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from factorlab.discovery.nonlinear_launcher import WorkerLifetime
from factorlab.discovery.process_policy import isolated_process_kwargs

CHILD = r'''
import datetime,json,os,sys,time
from pathlib import Path
import psutil
from factorlab.discovery.nonlinear_launcher import WorkerLifetime
mode, directory = sys.argv[1], Path(sys.argv[2])
parent = psutil.Process(os.getppid())
started_wall, started_mono = time.time(), time.monotonic()
reservation = {'started_at': datetime.datetime.fromtimestamp(
    started_wall - (3 if mode == 'already_expired' else 0), datetime.UTC).isoformat(),
    'started_monotonic': started_mono, 'launcher_pid': parent.pid,
    'launcher_created': parent.create_time()}
def stop(code):
    (directory/'exit.json').write_text(json.dumps({'code':code,'torch_loaded':'torch' in sys.modules}))
    os._exit(code)
guard = WorkerLifetime(reservation, 5 if mode == 'parent_loss' else 1,
    exit_process=stop, wall_clock=(lambda: time.time()-3600) if mode=='rollback' else None)
with guard:
    (directory/'ready.json').write_text(json.dumps({'pid':os.getpid(),'created':psutil.Process().create_time(),
        'parent_pid':parent.pid,'parent_created':parent.create_time()}))
    time.sleep(20)  # A blocking native wait cannot defer the independent watcher.
'''


def child_env():
    return {**os.environ, 'PYTHONPATH': os.pathsep.join((str(Path(__file__).resolve().parents[1]),
                                                       str(Path(psutil.__file__).resolve().parents[1]))),
            'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}


@pytest.mark.parametrize('mode', ['deadline', 'rollback', 'already_expired'])
def test_owned_child_exits_at_original_deadline_even_when_wall_clock_rolls_back(tmp_path, mode):
    before = time.monotonic()
    result = subprocess.run([sys.executable, '-B', '-c', CHILD, mode, str(tmp_path)],
        env=child_env(), capture_output=True, timeout=8, check=False, **isolated_process_kwargs())
    assert result.returncode == 124, result.stderr.decode(errors='replace')
    assert json.loads((tmp_path / 'exit.json').read_text()) == {'code': 124, 'torch_loaded': False}
    assert time.monotonic() - before < 8


def _wait_file(path, timeout=5):
    deadline = time.monotonic() + timeout
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(.02)
    assert path.exists(), f'Owned child did not publish {path.name}'
    return json.loads(path.read_text())


def test_actual_registered_parent_death_stops_only_owned_child(tmp_path):
    # The intermediate process is deliberately killed through its retained
    # Popen handle. The NN-like child must self-exit, not scan/kill other PIDs.
    parent_code = 'import subprocess,sys; p=subprocess.Popen(sys.argv[1:]); p.wait(timeout=15)'
    # Windows venv redirectors own kill-on-close job objects. Use the real
    # interpreter so this test exercises the watcher, not that OS side effect.
    executable = getattr(sys, '_base_executable', sys.executable)
    parent = subprocess.Popen([executable, '-B', '-c', parent_code,
        executable, '-B', '-c', CHILD, 'parent_loss', str(tmp_path)],
        env=child_env(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        **isolated_process_kwargs())
    identity, registered_parent = None, None
    try:
        identity = _wait_file(tmp_path / 'ready.json')
        began = time.monotonic()
        # Windows venv executables may be redirector processes. Kill the exact
        # parent that the worker actually registered, never assume Popen.pid.
        registered_parent = psutil.Process(identity['parent_pid'])
        assert registered_parent.create_time() == identity['parent_created']
        registered_parent.kill()
        parent.wait(timeout=3)
        assert _wait_file(tmp_path / 'exit.json', timeout=3) == {'code': 125, 'torch_loaded': False}
        assert time.monotonic() - began < 3
    finally:
        if registered_parent is not None and registered_parent.is_running():
            registered_parent.kill()
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=3)
        if identity is not None:
            try:
                child = psutil.Process(identity['pid'])
                if child.create_time() == identity['created'] and child.is_running():
                    child.wait(timeout=1)
            except psutil.TimeoutExpired:
                # Failure cleanup only: this is our exact recorded child.
                child.kill()
                child.wait(timeout=3)
            except psutil.NoSuchProcess:
                pass


def reservation():
    parent = psutil.Process(os.getppid())
    return {'started_at': datetime.datetime.now(datetime.UTC).isoformat(),
            'started_monotonic': time.monotonic(), 'launcher_pid': parent.pid,
            'launcher_created': parent.create_time()}


def test_reused_parent_identity_is_closed_without_killing_test_host():
    value, reasons, finished = reservation(), [], threading.Event()
    impostor = SimpleNamespace(is_running=lambda: True, status=lambda: psutil.STATUS_RUNNING,
                               create_time=lambda: value['launcher_created'] + 1)

    def record(code):
        reasons.append(code)
        finished.set()

    with WorkerLifetime(value, 1, parent_factory=lambda _: impostor, exit_process=record):
        assert finished.wait(timeout=1)
    assert reasons == [125]


def test_normal_exit_stops_watcher_and_later_clock_does_not_kill_host():
    value, reasons = reservation(), []
    now = [value['started_monotonic']]
    lifetime = WorkerLifetime(value, 1, exit_process=reasons.append, monotonic=lambda: now[0])
    with lifetime:
        pass
    now[0] += 500
    assert lifetime._thread is not None and not lifetime._thread.is_alive()
    assert not reasons


def test_future_monotonic_origin_cannot_extend_a_ticket():
    value = reservation()
    value['started_monotonic'] += 500
    with pytest.raises(ValueError, match='lifetime reservation'):
        WorkerLifetime(value, 1, exit_process=lambda _: pytest.fail('Must reject without self-exit'))


def test_worker_exception_still_stops_its_lifetime_thread():
    lifetime = WorkerLifetime(reservation(), 1, exit_process=lambda _: pytest.fail('Test host must not exit'))
    with pytest.raises(RuntimeError, match='synthetic work failed'), lifetime:
        raise RuntimeError('synthetic work failed')
    assert not lifetime._thread.is_alive()
