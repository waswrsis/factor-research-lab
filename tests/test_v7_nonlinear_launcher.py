"""Engineering boundary checks only; mocked kernel counters are NOT Linux proof."""
import copy
import json
import time
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from test_v7_nonlinear_orchestration import owner as _owner_fixture

from factorlab.discovery import nonlinear_launcher as launcher
from factorlab.discovery import nonlinear_orchestration as orchestration
from factorlab.discovery import nonlinear_runtime
from factorlab.protocol import fingerprint


@pytest.fixture
def owner(tmp_path, monkeypatch):
    yield from _owner_fixture.__wrapped__(tmp_path, monkeypatch)


def kernel():
    events = {'low': 0, 'high': 0, 'max': 0, 'oom': 0, 'oom_kill': 0}
    return {'path': '/fixture-not-a-linux-proof', 'inode': 12, 'boot_id': 'fixture',
            'scope': 'whole_cgroup', 'platform': 'linux', 'peak': 100 * 2**20,
            'events': events, 'raw': {'memory.max': str(768 * 2**20),
                'memory.high': str(640 * 2**20), 'memory.swap.max': '0', 'pids.max': '64',
                'cpu.max': '25000 100000', 'memory.peak': str(100 * 2**20),
                'memory.events': '\n'.join(f'{k} {v}' for k, v in events.items())}}


@pytest.mark.parametrize('damage', ['peak_limit', 'peak_forged', 'oom', 'events_forged',
                                    'cpu', 'swap', 'boundary', 'missing_oom'])
def test_kernel_acceptance_rejects_resource_or_counter_mismatch(damage):
    first, last = kernel(), kernel()
    if damage == 'peak_limit':
        last['peak'] = 600 * 2**20
        last['raw']['memory.peak'] = str(last['peak'])
    elif damage == 'peak_forged':
        last['raw']['memory.peak'] = str(700 * 2**20)
    elif damage == 'oom':
        last['events']['oom_kill'] = 1
    elif damage == 'events_forged':
        last['raw']['memory.events'] = 'oom 1\noom_kill 0'
    elif damage == 'cpu':
        last['raw']['cpu.max'] = '50000 100000'
    elif damage == 'swap':
        last['raw']['memory.swap.max'] = 'max'
    elif damage == 'boundary':
        last['inode'] += 1
    else:
        del last['events']['oom']
    with pytest.raises(ValueError):
        launcher._validate_kernel(first, last)


def job_for(owner):
    bridge, data, cutoff, _folder, epoch, _ = owner
    gates = {'protocol': epoch.value['protocol_receipt'], 'probes': ['unavailable'],
             'positive_control': 'unavailable'}
    reservation = orchestration.reserve_neural_ticket(bridge, epoch, fingerprint(gates),
                                                       mode='resource_trial', seconds=1)
    directory = bridge.store.directory / 'nonlinear_launches' / reservation['ticket_id']
    job = {'ticket': reservation['ticket'], 'ticket_id': reservation['ticket_id'],
           'context': nonlinear_runtime._context(data, cutoff, orchestration.epoch_pool(), bridge.source_id),
           'gate_artifacts': gates, 'mode': 'resource_trial', 'seconds': 1,
           'launch_directory': str(directory)}
    return directory, job


def test_hand_authored_verified_boolean_never_opens_production_resource_gate():
    with pytest.raises(ValueError, match='not installed'):
        nonlinear_runtime._external_resource_verifier({'verified': True, 'platform': 'linux'}, {})


def test_ticket_has_actual_global_channel_source_and_event_chain(owner):
    bridge, *_ = owner
    _, job = job_for(owner)
    assert launcher._ticket(job)['status'] == 'reserved'
    assert bridge.state['nonlinear']['channel']['spent_seconds'] == 1
    altered = copy.deepcopy(job)
    altered['gate_artifacts']['probes'] = ['forged']
    with pytest.raises(ValueError, match='canonical'):
        launcher._ticket(altered)
    bridge.store.db.execute('DELETE FROM events WHERE seq=0')
    bridge.store.db.commit()
    with pytest.raises(ValueError, match='journal'):
        launcher._ticket(job)


def test_operational_source_files_are_in_the_same_receipt_identity(owner):
    bridge, *_ = owner
    assert nonlinear_runtime._source()['files'] == bridge.store.identity['source_files']
    script = bridge.store.directory.parent / 'scripts' / 'v7_public_fetch.py'
    script.parent.mkdir()
    script.write_text('version = 1\n')
    with pytest.raises(ValueError, match='active monthly'):
        bridge.verify()


def test_ticket_rejects_missing_budget_even_when_receipt_claims_reserved(owner):
    bridge, *_ = owner
    _, job = job_for(owner)
    bridge.state['budget']['counts']['solve'] = 0
    bridge.store.save(bridge.month, bridge.state, 'fixture_corruption')
    with pytest.raises(ValueError, match='canonical'):
        launcher._ticket(job)


def test_changed_archived_source_rejected(owner):
    bridge, *_ = owner
    _, job = job_for(owner)
    (bridge.store.directory / 'source' / 'factorlab' / 'v7.py').write_text('changed')
    with pytest.raises(ValueError, match='archive changed'):
        launcher._ticket(job)


def test_launch_directory_cannot_be_changed_to_reuse_consumed_ticket(owner):
    directory, job = job_for(owner)
    with pytest.raises(ValueError, match='one fixed launch directory'):
        launcher.launch(directory.parent / 'alternative', job)


def test_timeout_consumes_ticket_and_second_launch_does_not_spawn(owner, monkeypatch):
    bridge, *_ = owner
    directory, job = job_for(owner)
    calls, stopped = [], []

    class Child:
        pid = 987654
        returncode = None

        def poll(self):
            return None

    def spawn(*args, **kwargs):
        calls.append((args, kwargs))
        return Child()

    monkeypatch.setattr(launcher, '_kernel_snapshot', kernel)
    monkeypatch.setattr(launcher.importlib.util, 'find_spec', lambda _: object())
    monkeypatch.setattr(launcher.psutil, 'Process', lambda *args: SimpleNamespace(create_time=lambda: 42.))
    monkeypatch.setattr(launcher.subprocess, 'Popen', spawn)
    monkeypatch.setattr(launcher, 'terminate_process_tree', lambda child: stopped.append(child.pid))
    moments = iter([0., 2.])
    monkeypatch.setattr(launcher.time, 'monotonic', lambda: next(moments))
    result = launcher.launch(directory, job)
    assert result['status'] == 'failed_consumed' and result['receipt'] is None
    assert stopped == [987654] and len(calls) == 1
    assert launcher.launch(directory, job)['status'] == 'already_consumed'
    assert len(calls) == 1 and bridge.budget.counts['solve'] == 21
    assert not (directory / 'receipt.json').exists()
    assert launcher._read(directory / 'failure.json')['error_type'] == 'TimeoutError'
    assert calls[0][0][0][2:4] == ['-m', 'factorlab.discovery.nonlinear_launcher']
    assert calls[0][1]['shell'] is False


def test_worker_requires_its_live_parent_and_runs_at_most_once(owner, monkeypatch):
    directory, job = job_for(owner)
    directory.mkdir(parents=True)
    launcher._write(directory / 'job.json', job)
    launcher._write(directory / 'reservation.json', {'job_hash': fingerprint(job),
        'launcher_pid': 12345, 'launcher_created': 42.,
        'started_at': launcher._now().isoformat(), 'started_monotonic': time.monotonic()})
    monkeypatch.setattr(launcher.os, 'getppid', lambda: 54321)
    with pytest.raises(ValueError, match='actually live'):
        launcher.worker(directory / 'job.json', _lifetime_factory=lambda *_: nullcontext())
    assert not (directory / 'worker-started.json').exists()
    monkeypatch.setattr(launcher.os, 'getppid', lambda: 12345)
    monkeypatch.setattr(launcher.psutil, 'Process', lambda *args: SimpleNamespace(create_time=lambda: 42.))
    calls = []
    monkeypatch.setattr(launcher, '_run_worker', lambda path: calls.append(path))
    launcher.worker(directory / 'job.json', _lifetime_factory=lambda *_: nullcontext())
    with pytest.raises(ValueError, match='permanently consumed'):
        launcher.worker(directory / 'job.json', _lifetime_factory=lambda *_: nullcontext())
    assert len(calls) == 1


def test_failed_canonical_ticket_cannot_start_the_worker(owner, monkeypatch):
    bridge, *_ = owner
    directory, job = job_for(owner)
    directory.mkdir(parents=True)
    launcher._write(directory / 'job.json', job)
    bridge.state['nonlinear']['attempts'][job['ticket_id']]['status'] = 'failed_consumed'
    bridge.commit()
    monkeypatch.setattr(launcher, '_kernel_snapshot', lambda: pytest.fail('consumed ticket touched resource producer'))
    with pytest.raises(ValueError, match='already consumed'):
        launcher._run_worker(directory / 'job.json')


def test_receipt_hash_alone_does_not_create_canonical_work(owner):
    directory, job = job_for(owner)
    directory.mkdir(parents=True)
    launcher._write(directory / 'job.json', job)
    launcher._write(directory / 'receipt.json', {'status': 'completed', 'verified': True})
    with pytest.raises((KeyError, ValueError)):
        launcher.verify_launcher_artifact({'launcher_receipt': str(directory / 'receipt.json')}, {})
    assert json.loads((directory / 'receipt.json').read_text())['value']['verified']
