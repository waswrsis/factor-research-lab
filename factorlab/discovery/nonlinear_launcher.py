"""Fixed NN subprocesses with parent-produced, run-bound Linux cgroup receipts.

The collector reads kernel files itself and journals a reserved fixed command.
It is local execution provenance, NOT administrator-resistant authentication.
Windows, missing Torch, missing kernel counters, incomplete work, and unverifiable
cgroup membership fail closed. No arbitrary command or receipt importer exists.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from filelock import FileLock

from factorlab.protocol import fingerprint, project_path

from .evidence import file_hash
from .process_policy import isolated_process_kwargs, restricted_environment, terminate_process_tree
from .v7_storage import immutable_record


def _now():
    return pd.Timestamp.now(tz='UTC')


def _write(path, value):
    immutable_record(path, {'value': value, 'sha256': fingerprint(value)})


def _read(path):
    result = json.loads(project_path(path).read_text('utf-8'))
    if set(result) != {'value', 'sha256'} or fingerprint(result['value']) != result['sha256']:
        raise ValueError('Launcher artifact integrity mismatch')
    return result['value']


class WorkerLifetime:
    """Self-exit only: exact registered parent and the original absolute lease.

    The monotonic timestamp is sampled in the same host before reservation
    publication; startup and a wall-clock rollback cannot grant extra time.
    Injected observers/exit are an internal unit-test seam, never job fields.
    """
    def __init__(self, reservation, seconds, *, parent_factory=None, exit_process=None,
                 wall_clock=None, monotonic=None):
        self._wall = wall_clock or time.time
        self._monotonic = monotonic or time.monotonic
        self._exit = exit_process or os._exit
        started = pd.Timestamp(reservation['started_at'])
        origin = reservation['started_monotonic']
        pid, created = reservation['launcher_pid'], reservation['launcher_created']
        if (started.tzinfo is None or pd.isna(started)
                or type(origin) not in (int, float) or not math.isfinite(origin)
                or origin > self._monotonic()
                or type(seconds) is not int or not 1 <= seconds <= 300
                or type(pid) is not int or pid <= 0 or pid == os.getpid()
                or type(created) not in (int, float) or not math.isfinite(created)):
            raise ValueError('Invalid fixed worker lifetime reservation')
        self._parent = (parent_factory or psutil.Process)(pid)
        self._created, self._pid = created, pid
        self._wall_deadline, self._deadline = started.timestamp() + seconds, origin + seconds
        self._stop = threading.Event()
        self._thread = None

    def _reason(self):
        if self._wall() >= self._wall_deadline or self._monotonic() >= self._deadline:
            return 124
        try:
            if (os.getppid() != self._pid or not self._parent.is_running()
                    or self._parent.create_time() != self._created
                    or self._parent.status() == psutil.STATUS_ZOMBIE):
                return 125
        except (psutil.Error, OSError):
            return 125  # Parent identity became unavailable; fail closed.
        return None

    def _watch(self):
        while not self._stop.is_set():
            reason = self._reason()
            if reason is not None:
                self._exit(reason)  # No kill(), process tree lookup or PID search.
                return  # A test exit callback may deliberately return.
            self._stop.wait(min(.05, max(.001, self._deadline - self._monotonic())))

    def __enter__(self):
        self._thread = threading.Thread(target=self._watch, name='neural-worker-lifetime', daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=.5)


def _kernel_snapshot():
    if sys.platform != 'linux':
        raise ValueError('linux_resource_proof_unavailable')
    root = Path('/sys/fs/cgroup').resolve()
    line = next((s for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::')), None)
    if line is None:
        raise ValueError('cgroup_v2_unavailable')
    leaf = (root / line[3:].lstrip('/')).resolve()
    if not leaf.is_relative_to(root):
        raise ValueError('Invalid kernel cgroup path')
    # The common resource boundary must contain the supervisor and NN child.
    # Require the exact envelope on one ancestor; never sum process RSS.
    expected = {'memory.max': 768 * 2**20, 'memory.high': 640 * 2**20,
                'memory.swap.max': 0, 'pids.max': 64}
    for node in (leaf, *leaf.parents):
        if not node.is_relative_to(root):
            break
        if not all((node / name).exists() for name in (*expected, 'cpu.max')):
            continue
        values = {name: (node / name).read_text().strip() for name in expected}
        cpu = (node / 'cpu.max').read_text().split()
        if (values != {k: str(v) for k, v in expected.items()} or len(cpu) != 2
                or cpu[0] == 'max' or int(cpu[1]) <= 0 or int(cpu[0]) / int(cpu[1]) != 1.0):
            continue
        for name in ('memory.peak', 'memory.current', 'memory.events', 'cpu.stat', 'cgroup.procs'):
            values[name] = (node / name).read_text().strip()
        values['cpu.max'] = ' '.join(cpu)
        events = {key: int(value) for key, value in (line.split() for line in values['memory.events'].splitlines())}
        return {'path': str(node), 'inode': node.stat().st_ino,
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'raw': values, 'peak': int(values['memory.peak']), 'events': events,
                'observed_at': _now().isoformat(), 'scope': 'whole_cgroup', 'platform': 'linux'}
    raise ValueError('Exact shared CPU100/RAM768/high640/swap0/PIDs64 cgroup is required')


def _same_boundary(first, last):
    return all(first.get(key) == last.get(key) for key in ('path', 'inode', 'boot_id', 'scope', 'platform'))


def _validate_kernel(first, last):
    if (not _same_boundary(first, last) or last['peak'] < first['peak']
            or not 0 < last['peak'] < 600 * 2**20
            or any(snapshot['events'].get(name, -1) != 0
                   for snapshot in (first, last) for name in ('oom', 'oom_kill'))):
        raise ValueError('Whole-cgroup peak/OOM/boundary acceptance failed')
    required = {'memory.max': str(768 * 2**20), 'memory.high': str(640 * 2**20),
                'memory.swap.max': '0', 'pids.max': '64'}
    for snapshot in (first, last):
        parsed_events = {key: int(value) for key, value in
                         (line.split() for line in snapshot['raw']['memory.events'].splitlines())}
        if (type(snapshot['peak']) is not int or int(snapshot['raw']['memory.peak']) != snapshot['peak']
                or parsed_events != snapshot['events']):
            raise ValueError('Kernel summary differs from its raw counters')
        cpu = snapshot['raw']['cpu.max'].split()
        if (any(snapshot['raw'].get(k) != v for k, v in required.items())
                or len(cpu) != 2 or cpu[0] == 'max' or int(cpu[1]) <= 0
                or int(cpu[0]) / int(cpu[1]) != 1.0):
            raise ValueError('Kernel envelope changed during the trial')


def availability():
    """Read-only preflight; never installs Torch or creates a cgroup/service."""
    if sys.platform != 'linux':
        return {'status': 'linux_resource_proof_unavailable'}
    if importlib.util.find_spec('torch') is None:
        return {'status': 'dependency_unavailable', 'dependency': 'torch'}
    from .v7_service import HostGuard
    guard = HostGuard()()
    if not guard['allowed']:
        return {'status': 'host_guard_closed', 'reasons': guard['reasons']}
    try:
        snapshot = _kernel_snapshot()
        _validate_kernel(snapshot, snapshot)
    except (OSError, ValueError) as exc:
        return {'status': 'linux_resource_proof_unavailable', 'reason': str(exc)[:400]}
    return {'status': 'available', 'resource_scope': 'actual_common_cgroup_not_process_RSS'}


def _ticket(job):
    """Read the canonical RunStore reservation, not applicant-provided counters."""
    run = project_path(job['ticket']['run_directory'])
    manifest = json.loads((run / 'manifest.json').read_text('utf-8'))
    if (fingerprint(manifest['identity']) != manifest['hash']
            or manifest['identity']['source_files'] != job['context']['source']['files']):
        raise ValueError('Launch source differs from RunStore')
    if any(file_hash(project_path(run / 'source' / name)) != digest
           for name, digest in manifest['identity']['source_files'].items()):
        raise ValueError('Frozen RunStore source archive changed')
    with sqlite3.connect((run / 'state.sqlite').as_uri() + '?mode=ro', uri=True) as db:
        found = db.execute('SELECT value,hash FROM months WHERE month=?', (job['ticket']['month'],)).fetchone()
        if found is None:
            raise ValueError('Missing authoritative neural reservation')
        state = json.loads(found[0])
        if fingerprint(state) != found[1]:
            raise ValueError('Neural monthly reservation checksum mismatch')
        latest, previous, sequence = None, '0' * 64, 0
        for number, text, digest in db.execute('SELECT seq,value,hash FROM events ORDER BY seq'):
            event = json.loads(text)
            if (number != sequence or event['sequence'] != sequence or event['previous'] != previous
                    or fingerprint(event) != digest):
                raise ValueError('Neural monthly reservation journal mismatch')
            if event['month'] == job['ticket']['month']:
                latest = event
            previous, sequence = digest, sequence + 1
        if latest is None or latest['state_hash'] != found[1]:
            raise ValueError('Neural monthly reservation journal mismatch')
    row = state.get('nonlinear', {}).get('attempts', {}).get(job['ticket_id'])
    if (not row or row['ticket'] != job['ticket'] or fingerprint(row['ticket']) != job['ticket_id']
            or row['status'] not in {'reserved', 'running', 'completed', 'failed_consumed'}
            or state['data_id'] != job['ticket']['controller_data_id']
            or state.get('nonlinear_input_data_id') != job['ticket'].get('input_data_id')
            or fingerprint(job['gate_artifacts']) != row['ticket']['gate_id']
            or job['mode'] != row['ticket']['mode']
            or state['nonlinear']['epoch_id'] != job['ticket']['epoch_id']
            or any(state['nonlinear']['channel']['counts'][name] < amount
                   for name, amount in job['ticket']['limits'].items())
            or state['budget']['counts']['solve'] < job['ticket']['limits']['solve']
            or state['nonlinear']['channel']['spent_seconds'] < job['ticket']['seconds']):
        raise ValueError('Missing or changed canonical neural ticket')
    from .exposure import ExposureLedger
    exposure_path = project_path(manifest['identity']['exposure_path'])
    if not exposure_path.is_file():
        raise ValueError('Missing authoritative S exposure ledger')
    # Open only the actual manifest-bound ledger, without creating schema or
    # allowing a new empty sibling to replace protected V/C state.
    exposure = ExposureLedger.__new__(ExposureLedger)
    exposure.clock = _now
    exposure.path = exposure_path
    exposure.db = sqlite3.connect(exposure_path.as_uri() + '?mode=ro', uri=True)
    try:
        exposed = state['search_exposure']
        if exposed.get('data_id') != state['data_id'] or exposed['source'] != job['context']['source_id']:
            raise ValueError('Neural worker S exposure identity mismatch')
        exposure.assert_prompt_search(**{k: v for k, v in exposed.items() if k != 'data_id'})
    finally:
        exposure.close()
    return row


def save_worker_data(directory, data, *, cutoff):
    """Nonexecutable arrays and metadata; pickle is never accepted."""
    from .nonlinear_orchestration import causal_worker_data
    data = causal_worker_data(data, cutoff)
    directory = project_path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays, metadata = directory / 'data.npz', directory / 'metadata.parquet'
    with arrays.open('xb') as handle:
        np.savez_compressed(handle, values=data.frame.values, target=data.target,
                            observed=data.observed, maturity=data.maturity.asi8)
        handle.flush()
        os.fsync(handle.fileno())
    data.frame.metadata.to_parquet(metadata, index=False)
    return {'arrays': str(arrays), 'arrays_hash': file_hash(arrays),
            'metadata': str(metadata), 'metadata_hash': file_hash(metadata),
            'feature_names': list(data.frame.feature_names), 'bypass_mask': list(data.frame.bypass_mask),
            'schema_hash': data.frame.schema_hash, 'availability_mode': data.availability_mode}


def _load_data(job):
    from .known_library import FactorFrame
    from .rolling import RollingData, RollingSpec
    record = job['data']
    for kind in ('arrays', 'metadata'):
        if file_hash(project_path(record[kind])) != record[kind + '_hash']:
            raise ValueError('Frozen worker data changed')
    with np.load(project_path(record['arrays']), allow_pickle=False) as arrays:
        frame = FactorFrame(arrays['values'], pd.read_parquet(project_path(record['metadata'])),
                            tuple(record['feature_names']), tuple(record['bypass_mask']), record['schema_hash'])
        return RollingData(frame, arrays['target'], pd.to_datetime(arrays['maturity'], utc=True),
                           arrays['observed'], RollingSpec(**job['rolling']), record['availability_mode'])


def launch(directory, job):
    """Only the fixed trusted worker is launchable; all failures consume the ticket."""
    directory = project_path(directory)
    expected_directory = project_path(job['ticket']['run_directory']) / 'nonlinear_launches' / job['ticket_id']
    if directory != expected_directory.resolve() or job.get('launch_directory') != str(directory):
        raise ValueError('Each nonrefundable ticket has exactly one fixed launch directory')
    if (type(job['seconds']) not in (int, float) or not math.isfinite(job['seconds'])
            or not 0 < job['seconds'] <= 300 or job['seconds'] != job['ticket']['seconds']
            or job['mode'] != job['ticket']['mode']):
        raise ValueError('Invalid frozen neural worker deadline or mode')
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / 'launch.lock'), timeout=0):
        if (directory / 'reservation.json').exists():
            return {'status': 'already_consumed', 'receipt': None}
        if _ticket(job)['status'] != 'reserved':
            return {'status': 'already_consumed', 'receipt': None}
        first = _kernel_snapshot()
        if importlib.util.find_spec('torch') is None:
            return {'status': 'dependency_unavailable', 'dependency': 'torch', 'receipt': None}
        if job['mode'] not in {'resource_trial', 'residual'}:
            raise ValueError('Unknown fixed neural worker mode')
        _write(directory / 'job.json', job)
        began = time.monotonic()
        started = _now()
        command = [sys.executable, '-B', '-m', 'factorlab.discovery.nonlinear_launcher',
                   '--job', str(directory / 'job.json')]
        reservation = {'job_hash': fingerprint(job), 'command': command,
                       'started_at': started.isoformat(), 'started_monotonic': began, 'kernel_before': first,
                       'launcher_pid': os.getpid(), 'launcher_created': psutil.Process().create_time()}
        _write(directory / 'reservation.json', reservation)
        environment = restricted_environment()
        environment.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                           CUDA_VISIBLE_DEVICES='')
        process, failure, identity = None, None, None
        try:
            with (directory / 'worker.log').open('xb') as output:
                process = subprocess.Popen(command, cwd=project_path('.'), env=environment,
                    stdin=subprocess.DEVNULL, stdout=output, stderr=output, shell=False,
                    **isolated_process_kwargs())
                identity = {'pid': process.pid, 'created_at': psutil.Process(process.pid).create_time()}
                while process.poll() is None:
                    if time.monotonic() - began >= job['seconds']:
                        raise TimeoutError('neural_ticket_deadline_consumed')
                    if output.tell() > 1_048_576:
                        raise ValueError('Neural worker diagnostic budget exceeded')
                    time.sleep(.05)
                if process.returncode != 0:
                    raise ValueError('neural_worker_failed')
            after = _kernel_snapshot()
            _validate_kernel(first, after)
            ready, result = _read(directory / 'membership.json'), _read(directory / 'result.json')
            if (ready['job_hash'] != fingerprint(job) or ready['pid'] != identity['pid']
                    or not _same_boundary(first, ready['kernel'])
                    or result['job_hash'] != fingerprint(job)
                    or result['result']['status'] != 'completed'
                    or time.monotonic() - began >= job['seconds']):
                raise ValueError('Worker identity, completion or deadline mismatch')
            receipt = {'status': 'completed', 'job_hash': fingerprint(job), 'process': identity,
                       'started_at': started.isoformat(), 'finished_at': _now().isoformat(),
                       'kernel_before': first, 'kernel_after': after,
                       'membership_hash': file_hash(directory / 'membership.json'),
                       'result_hash': file_hash(directory / 'result.json'),
                       'reservation_hash': file_hash(directory / 'reservation.json'),
                       'test_only': False, 'authentication': 'local_run_bound_collector_not_admin_resistant'}
            _write(directory / 'receipt.json', receipt)
            return {'status': 'completed', 'receipt': str(directory / 'receipt.json')}
        except (ValueError, OSError, TimeoutError, RuntimeError) as exc:
            failure = {'status': 'failed_consumed', 'error_type': type(exc).__name__, 'error': str(exc)[:400]}
            _write(directory / 'failure.json', failure)
            return {**failure, 'receipt': None}
        finally:
            if process is not None and process.poll() is None:
                terminate_process_tree(process)


def artifact_from_receipt(path):
    path = project_path(path)
    receipt, job = _read(path), _read(path.parent / 'job.json')
    after = receipt['kernel_after']
    return {'launcher_receipt': str(path), 'launcher_receipt_hash': file_hash(path),
            'platform': 'linux', 'scope': 'whole_cgroup', 'oom': 0, 'oom_kill': 0,
            'memory_max': 768 * 2**20, 'memory_high': 640 * 2**20, 'swap_max': 0,
            'cpu_quota_fraction': 1.0, 'memory_peak': after['peak'],
            'trial_context': job['resource_context'], 'status': receipt['status'],
            'raw_receipt_hash': fingerprint([receipt['kernel_before'], after]),
            'workload_result_hash': receipt['result_hash'],
            **job['capacity'],
            'trial_started_at': receipt['started_at'], 'trial_finished_at': receipt['finished_at']}


def verify_launcher_artifact(artifact, expected):
    path = project_path(artifact['launcher_receipt'])
    receipt, job = _read(path), _read(path.parent / 'job.json')
    ticket = _ticket(job)
    if (artifact != artifact_from_receipt(path) or artifact['trial_context'] != expected
            or receipt['job_hash'] != fingerprint(job) or job['mode'] != 'resource_trial'
            or receipt['status'] != 'completed' or receipt['test_only'] is not False
            or ticket['status'] != 'completed'
            or path.parent != (project_path(job['ticket']['run_directory']) / 'nonlinear_launches' / job['ticket_id']).resolve()):
        raise ValueError('Resource artifact is not the registered completed launcher trial')
    _validate_kernel(receipt['kernel_before'], receipt['kernel_after'])
    for filename, field in [('result.json', 'result_hash'), ('membership.json', 'membership_hash'),
                            ('reservation.json', 'reservation_hash')]:
        if file_hash(path.parent / filename) != receipt[field]:
            raise ValueError('Launcher producer evidence changed')
    membership = _read(path.parent / 'membership.json')
    if (membership['job_hash'] != fingerprint(job) or membership['pid'] != receipt['process']['pid']
            or not _same_boundary(membership['kernel'], receipt['kernel_before'])):
        raise ValueError('Worker was not a member of the measured whole cgroup')
    result = _read(path.parent / 'result.json')
    if (result['job_hash'] != fingerprint(job) or result['result']['status'] != 'completed'
            or result['capacity'] != job['capacity'] or result['automatic_adoption'] is not False):
        raise ValueError('Resource trial did not complete its frozen workload')
    from .v7_run import source_files
    if source_files() != job['context']['source']['files']:
        raise ValueError('Resource workload source changed')
    return {'verified': True, 'test_only': False,
            'scope': 'actual_run_bound_kernel_receipt_not_remote_identity_authentication'}


def _run_worker(job_path):
    from .joint_pool import FrozenPool
    from .nonlinear_channel import GATE_FIELDS, ChannelBudget, NeuralGate, research_residual
    from .nonlinear_orchestration import _config, input_data_identity
    from .nonlinear_runtime import verify_gate_artifacts, verify_pre_resource_artifacts
    from .v7_run import source_files
    job_path = project_path(job_path)
    job = _read(job_path)
    if _ticket(job)['status'] != 'reserved':
        raise ValueError('Neural worker ticket is already consumed')
    if source_files() != job['context']['source']['files']:
        raise ValueError('Frozen neural worker source changed')
    kernel = _kernel_snapshot()
    _write(job_path.parent / 'membership.json', {'job_hash': fingerprint(job), 'pid': os.getpid(), 'kernel': kernel})
    data, config, pool = _load_data(job), _config(job['config']), FrozenPool.from_dict(job['pool'])
    from .nonlinear_runtime import workload_capacity
    capacity = workload_capacity(data, job['cutoff'])
    from .nonlinear_runtime import _context
    if (_context(data, job['cutoff'], pool, job['context']['source_id']) != job['context']
            or config.digest != job['ticket']['config_id']
            or input_data_identity(data, job['cutoff']) != job['ticket']['input_data_id']
            or (data.frame.metadata.available_at >= pd.Timestamp(job['cutoff'])).any()
            or (data.observed & np.asarray(data.maturity > pd.Timestamp(job['cutoff']))).any()
            or job_path.parent != project_path(job['launch_directory']) or capacity != job['capacity']):
        raise ValueError('Worker data, configuration or launch directory changed')
    verifier = verify_pre_resource_artifacts if job['mode'] == 'resource_trial' else verify_gate_artifacts
    gate = verifier(job['artifact_directory'], data, job['cutoff'], pool, job['gate_artifacts'],
                    source_id=job['context']['source_id'], config=config)
    if gate['test_only']:
        raise ValueError('Synthetic gate fixtures cannot launch production NN work')
    limits = job['ticket']['limits']

    class TicketBudget(ChannelBudget):
        @property
        def limits(self):
            return {**super().limits, 'solve': limits['solve']}

    budget = TicketBudget(max_neural_fits=limits['neural_fit'],
                          max_epochs=max(1, limits['epoch']), max_seconds=job['seconds'])
    permission = NeuralGate(**{key: True for key in GATE_FIELDS},
                            evidence_ids={key: gate['gate_id'] for key in GATE_FIELDS})
    result = research_residual(data, job['cutoff'], pool, permission, config, budget)
    if any(budget.counts[k] > amount for k, amount in limits.items()):
        raise ValueError('Worker exceeded its nonrefundable monthly ticket')
    _write(job_path.parent / 'result.json', {'job_hash': fingerprint(job), 'result': result,
        'capacity': capacity,
        'scope': 'resource_rehearsal_not_market_evidence' if job['mode'] == 'resource_trial' else 'S_research_only',
        'automatic_adoption': False})


def worker(job_path, *, _lifetime_factory=None):
    job_path = project_path(job_path)
    with FileLock(str(job_path.parent / 'worker.lock'), timeout=0):
        marker = job_path.parent / 'worker-started.json'
        if marker.exists():
            raise ValueError('Neural worker attempt is permanently consumed')
        job, reservation = _read(job_path), _read(job_path.parent / 'reservation.json')
        if (reservation['job_hash'] != fingerprint(job) or reservation['launcher_pid'] != os.getppid()
                or psutil.Process(os.getppid()).create_time() != reservation['launcher_created']):
            raise ValueError('Worker requires its actually live reserved launcher parent')
        _write(marker, {'job_hash': fingerprint(job), 'pid': os.getpid(), 'started_at': _now().isoformat()})
        # Only fixed CLI execution uses the production default. In-process
        # engineering tests inject a context manager rather than risk exiting
        # the pytest host. No JSON/GPT field can supply this factory.
        lifetime = (_lifetime_factory or WorkerLifetime)(reservation, job['ticket']['seconds'])
        with lifetime:
            _run_worker(job_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    worker(parser.parse_args().job)
