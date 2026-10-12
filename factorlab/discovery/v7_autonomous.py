"""Monthly S research subprocesses joined to the receipt-based forward service.

This scheduler inherits external resource limits. It never installs a service,
polls a network source, promotes A0, or invents a fresh run after interruption.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import uuid4

import pandas as pd
import psutil
from filelock import FileLock

from factorlab.protocol import fingerprint, project_path
from factorlab.time_utils import month_start

from .evidence import file_hash
from .joint_pool import FrozenPool, PoolMember
from .process_policy import isolated_process_kwargs, restricted_environment
from .rolling import RollingSpec
from .v7_feed import ReceiptArchive
from .v7_lifecycle import stamp
from .v7_run import RunStore, V7Config, _verify_month_artifacts, run_month, source_files
from .v7_storage import immutable_record

FULL_TRIAL_BACKEND_DISPATCH = 'v7-original-full-trial-controller-v2'


@dataclass(frozen=True)
class AutonomousConfig:
    backend: str = 'codex'
    max_attempts: int = 3
    max_jobs: int = 96
    max_validation_batches: int = 3
    deadline_seconds: int = 7200
    research: V7Config = field(default_factory=lambda: V7Config(
        evaluation_mode='joint', search_mode='gpt', profile_candidates=6))
    not_before_month: str | None = None

    def __post_init__(self):
        if self.not_before_month is not None and (
                not isinstance(self.not_before_month, str)
                or not re.fullmatch(r'[0-9]{4}-(?:0[1-9]|1[0-2])', self.not_before_month)
                or self.not_before_month.startswith('0000-')):
            raise ValueError('Research not_before_month must be a canonical UTC YYYY-MM month')
        if (self.backend not in {'codex', 'disabled'} or self.max_attempts not in (1, 2, 3)
                or self.max_jobs != 96 or self.max_validation_batches != 3
                or type(self.deadline_seconds) is not int or not 30 <= self.deadline_seconds <= 7200
                or self.research.evaluation_mode != 'joint' or self.research.formal_confirmation_enabled):
            raise ValueError('Invalid frozen autonomous research policy')
        if self.backend == 'codex' and self.research.search_mode != 'gpt':
            raise ValueError('Explicit Codex backend requires the GPT research policy')


def _read(path):
    value = json.loads(Path(path).read_text('utf-8'))
    if set(value) != {'value', 'sha256'} or fingerprint(value['value']) != value['sha256']:
        raise ValueError('Autonomous artifact integrity failure')
    return value['value']


def _write(path, value):
    return immutable_record(path, {'value': value, 'sha256': fingerprint(value)})


def inherited_limits():
    """Fail closed outside an already constrained Linux cgroup v2 ancestor chain."""
    if sys.platform != 'linux':
        raise ValueError('Real autonomous workers require the externally constrained Linux host')
    line = next((line for line in Path('/proc/self/cgroup').read_text().splitlines()
                 if line.startswith('0::')), None)
    if line is None:
        raise ValueError('A unified inherited cgroup is required')
    root = Path('/sys/fs/cgroup').resolve()
    leaf = (root / line[3:].lstrip('/')).resolve()
    if not leaf.is_relative_to(root):
        raise ValueError('Invalid inherited cgroup path')
    limits = {'cpu': float('inf'), 'memory': float('inf'), 'high': float('inf'),
              'swap': float('inf'), 'pids': float('inf')}
    for node in (leaf, *leaf.parents):
        if not node.is_relative_to(root):
            break
        cpu = (node / 'cpu.max').read_text().split() if (node / 'cpu.max').exists() else ['max']
        if cpu[0] != 'max':
            limits['cpu'] = min(limits['cpu'], int(cpu[0]) / int(cpu[1]))
        for name, filename in [('memory', 'memory.max'), ('high', 'memory.high'),
                               ('swap', 'memory.swap.max'), ('pids', 'pids.max')]:
            value = (node / filename).read_text().strip() if (node / filename).exists() else 'max'
            if value != 'max':
                limits[name] = min(limits[name], int(value))
    if not (0 < limits['cpu'] <= .25 and 0 < limits['memory'] <= 768 * 2**20
            and 0 < limits['high'] <= 640 * 2**20
            and limits['swap'] == 0 and 0 < limits['pids'] <= 64):
        raise ValueError('Research requires shared CPU25%/RAM768MiB/no-swap/PIDs64 limits')
    return {'cgroup': str(leaf), 'effective_limits': limits, 'scope': 'inherited_ancestor_limits'}


class SubprocessRunner:
    """No detached service. Every child retains its parent's constrained cgroup."""
    def __init__(self):
        self.children = {}

    @staticmethod
    def command(job_path, attempt):
        return [sys.executable, '-B', '-m', 'factorlab.discovery.v7_autonomous', 'worker',
                '--job', str(job_path), '--attempt', str(attempt['number']), '--token', attempt['token']]

    def find(self, job_path, attempt):
        command = self.command(job_path, attempt)
        matches = []
        for process in psutil.process_iter(['pid', 'cmdline', 'create_time']):
            if process.info['cmdline'] == command:
                matches.append({'pid': process.pid, 'created': process.info['create_time'], 'command': command})
        if len(matches) > 1:
            raise ValueError('Multiple workers for one reserved attempt')
        return matches[0] if matches else None

    def status(self, identity):
        child = self.children.get(identity['pid'])
        if child is not None and child.poll() is not None:
            return 'exited'
        try:
            process = psutil.Process(identity['pid'])
            if (process.create_time() != identity['created'] or process.cmdline() != identity['command']):
                return 'identity_mismatch'
            return 'exited' if process.status() == psutil.STATUS_ZOMBIE else 'running'
        except psutil.NoSuchProcess:
            return 'exited'

    def start(self, job_path, attempt):
        inherited_limits()
        command = self.command(job_path, attempt)
        root = project_path('.')
        environment = restricted_environment()
        temporary = Path(job_path).parent / 'tmp'
        temporary.mkdir(exist_ok=True)
        environment.update(TMP=str(temporary), TEMP=str(temporary), TMPDIR=str(temporary),
                           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with (Path(job_path).parent / f'worker-{attempt["number"]}.log').open('ab') as log:
            process = subprocess.Popen(command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=log, shell=False, **isolated_process_kwargs())
        self.children[process.pid] = process
        return {'pid': process.pid, 'created': psutil.Process(process.pid).create_time(), 'command': command}

    def stop(self, identity):
        try:
            # Keep the exact validated Process instance. Reconstructing it after
            # status() could capture a reused PID and then signal another job.
            process = psutil.Process(identity['pid'])
            if (process.create_time() != identity['created'] or process.cmdline() != identity['command']
                    or process.status() == psutil.STATUS_ZOMBIE):
                return {'status': 'not_running_or_identity_changed', 'killed': []}
        except psutil.NoSuchProcess:
            return {'status': 'not_running_or_identity_changed', 'killed': []}
        try:
            descendants = process.children(recursive=True)
        except psutil.NoSuchProcess:
            return {'status': 'exited_during_teardown', 'killed': []}
        killed = []
        for target in reversed([process, *descendants]):
            try:
                target.kill()  # psutil verifies PID reuse on signalling operations.
                killed.append(target.pid)
            except psutil.NoSuchProcess:
                pass
        return {'status': 'owned_tree_signalled', 'killed': killed, 'escaped_descendants_verified': False}

    def reap_stopped(self, identity):
        """Bounded teardown of the exact previously owned child, never new work."""
        if self.status(identity) == 'identity_mismatch':
            return {'status': 'identity_lost_no_signal'}
        child = self.children.get(identity['pid'])
        if child is not None:
            child.wait(timeout=5)
        if self.status(identity) == 'running':
            process = psutil.Process(identity['pid'])
            if process.create_time() != identity['created'] or process.cmdline() != identity['command']:
                raise ValueError('Owned worker changed during teardown')
            process.wait(timeout=5)
        if self.status(identity) == 'running':
            raise ValueError('Original worker teardown is unresolved')


class ResearchScheduler:
    def __init__(self, runtime, *, config=None, runner=None, authority=None, schedule=None):
        self.runtime, self.config = runtime, config or AutonomousConfig()
        if tuple(self.config.research.symbols) != tuple(runtime.config.symbols):
            raise ValueError('Research must share the actual forward universe')
        from .monthly_work_authority import MonthlyWorkAuthority
        from .neural_forward_ticket import _LifecycleReader, _verify_authority_binding
        from .scheduled_search import FrozenSWorkSchedule
        if type(authority) is not MonthlyWorkAuthority or type(schedule) is not FrozenSWorkSchedule:
            raise ValueError('Autonomous S requires the actual service wallet and immutable work schedule')
        from .scheduled_search import validate_schedule_authority
        validate_schedule_authority(authority, schedule)
        if fingerprint(schedule.to_dict()['config']) != fingerprint(asdict(self.config.research)):
            raise ValueError('Autonomous S schedule differs from the research algorithm')
        reader = _LifecycleReader(runtime.directory)
        try:
            _verify_authority_binding(reader, authority, reader.verify()['operations'])
        finally:
            reader.close()
        self.authority, self.schedule = authority, schedule
        self.runner = runner or SubprocessRunner()
        self.directory = project_path(runtime.directory / 'autonomous')
        self.directory.mkdir(parents=True, exist_ok=True)
        self.exposure_path = runtime.directory / 'exposure.sqlite'
        if runtime.identity['source_files'] != source_files():
            raise ValueError('Research cannot change a frozen forward source identity')
        self.identity = {'version': 2, 'work_authority': {'directory': str(authority.directory),
                         'manifest_hash': authority.manifest_hash}, 'work_schedule': schedule.to_dict(), 'config': asdict(self.config), 'feed_id': runtime.feed.identity,
                         'forward_identity': fingerprint(runtime.identity), 'source_files': source_files(),
                         'exposure_path': str(self.exposure_path), 'rolling': asdict(runtime.spec)}
        _write(self.directory / 'identity.json', self.identity)
        self.db = sqlite3.connect(self.directory / 'scheduler.sqlite')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS jobs (month TEXT PRIMARY KEY, value TEXT, hash TEXT)')
        self.db.commit()

    def close(self):
        self.db.close()

    def _accept_exited_outcome(self, state, job, attempt, outcome):
        from .s_outer_context import check_clock
        if (outcome['job_hash'] != state['job_hash'] or outcome['token'] != attempt['token']
                or outcome['attempt'] != attempt['number']
                or outcome['status'] not in {'complete', 'failed', 'resource_incomplete', 'deadline_consumed'}
                or job['version'] != (3 if 'service_restriction' in job['outer_clock'] else 2)
                or job['deadline'] != job['outer_clock']['wall_deadline']):
            raise ValueError('Worker outcome differs from original attempt')
        # This runs AFTER reading the entire original outcome. Being able to
        # inspect it in a later process is not permission to credit late work.
        try:
            from .s_outer_context import verify_job_clock
            verify_job_clock(job,state['job_path'])
            if job['outer_clock']['issuer']['parent_pid'] != os.getpid():
                raise ValueError('Original research issuer differs from this controller')
        except (ValueError, TimeoutError) as error:
            state.update(status='deadline_consumed' if isinstance(error, TimeoutError)
                         else 'interrupted_consumed', stop_reason='original_outer_clock_lost_at_readback')
        else:
            state['status'] = ('deadline_consumed' if stamp(outcome['finished_at']) > stamp(job['deadline'])
                               else outcome['status'])
        state['outcome'] = outcome

    def settle_running(self):
        """Observe/close existing attempts only; never plan, spawn, queue or retry.

        Used while the original controller issuer remains alive. A dead or
        different issuer cannot be replaced by the process reading this state.
        """
        from .s_outer_context import check_clock
        with FileLock(str(self.directory / '.lock'), timeout=0):
            if source_files() != self.identity['source_files']:
                raise ValueError('Frozen autonomous source changed')
            self.authority.verify()
            if self.identity['work_authority'] != {
                    'directory': str(self.authority.directory), 'manifest_hash': self.authority.manifest_hash}:
                raise ValueError('Original autonomous authority changed')
            months = [row[0] for row in self.db.execute('SELECT month FROM jobs ORDER BY month')]
            if {path.parent.name for path in (self.directory / 'jobs').glob('*/job.json')} != set(months):
                raise ValueError('Research job index differs from immutable reservations; never reset attempts')
            actions, pending = [], False
            for month in months:
                state = self._load(month)
                if state['status'] not in {'planned', 'running', 'interrupted'}:
                    continue
                prior_hash = fingerprint(state)
                job = _read(state['job_path'])
                if (fingerprint(job) != state['job_hash'] or job['month'] != month
                        or job['source_files'] != self.identity['source_files']):
                    raise ValueError('Frozen job differs from scheduler checkpoint')
                attempt = state['attempts'][-1] if state['attempts'] else None
                identity = attempt.get('process') if attempt else None
                if attempt and identity is None:
                    identity = self.runner.find(state['job_path'], attempt)
                    attempt['process'] = identity
                status = self.runner.status(identity) if identity else 'exited'
                if status == 'running':
                    try:
                        check_clock(job['outer_clock'])
                        if job['outer_clock']['issuer']['parent_pid'] != os.getpid():
                            raise ValueError('Original research issuer differs from this controller')
                    except (ValueError, TimeoutError) as error:
                        state['termination'] = self.runner.stop(identity)
                        self.runner.reap_stopped(identity)
                        state.update(status='deadline_consumed' if isinstance(error, TimeoutError)
                                     else 'interrupted_consumed', stop_reason='original_outer_clock_lost')
                    else:
                        pending = True
                elif status == 'identity_mismatch':
                    # Never signal a reused PID or a process outside this reservation.
                    state.update(status='process_identity_lost', stop_reason='original_process_identity_lost')
                else:
                    outcome_path = (Path(state['job_path']).parent / f'outcome-{attempt["number"]}.json'
                                    if attempt else None)
                    if outcome_path is not None and outcome_path.is_file():
                        outcome = _read(outcome_path)
                        self._accept_exited_outcome(state, job, attempt, outcome)
                    else:
                        state.update(status='interrupted_consumed', stop_reason=(
                            'original_worker_exited_without_outcome' if attempt
                            else 'original_plan_has_no_reserved_attempt'))
                if fingerprint(state) != prior_hash:
                    self._save(state)
                actions.append({'month': month, 'status': state['status'],
                                'attempt': attempt['number'] if attempt else None})
            return {'status': 'pending' if pending else 'settled', 'actions': actions}

    def stop_and_reap(self, reason='supervisor_stopped'):
        """Close only this scheduler's exact reserved process identities."""
        identities = []
        for month, in self.db.execute('SELECT month FROM jobs').fetchall():
            state = self._load(month)
            if state['status'] != 'complete' and state['attempts']:
                attempt = state['attempts'][-1]
                identity = attempt.get('process') or self.runner.find(state['job_path'], attempt)
                if identity is not None:
                    identities.append(identity)
        result = self.stop(reason)
        for identity in identities:
            self.runner.reap_stopped(identity)
        return result

    def worker_active(self):
        """Read-only fit deferral; an unresolved reserved spawn is conservatively busy."""
        for month, in self.db.execute('SELECT month FROM jobs').fetchall():
            state = self._load(month)
            if state['status'] != 'running':
                continue
            attempt = state['attempts'][-1]
            identity = attempt.get('process')
            if identity is None or self.runner.status(identity) == 'running':
                return True
        return False

    def _load(self, month):
        row = self.db.execute('SELECT value,hash FROM jobs WHERE month=?', (month,)).fetchone()
        if row is None:
            return None
        result = json.loads(row[0])
        if fingerprint(result) != row[1] or result['month'] != month:
            raise ValueError('Research scheduler checkpoint identity failure')
        return result

    def _save(self, state):
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO jobs VALUES (?,?,?)',
                            (state['month'], json.dumps(state, sort_keys=True), fingerprint(state)))

    def _plan(self, cutoff, now):
        from .forward_service_scope import check_service_work
        check_service_work(self.runtime)
        from .s_outer_context import original_clock
        outer = original_clock(self.config.deadline_seconds)
        month = cutoff.strftime('%Y-%m')
        if self.db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] >= self.config.max_jobs:
            raise ValueError('Monthly research archive capacity reached; explicit migration required')
        directory = self.directory / 'jobs' / month
        directory.mkdir(parents=True, exist_ok=True)
        # Quarantine is derived from the original lifecycle enrollment; no
        # caller can redirect its memory chain to another repetition.
        from .quarantine_execution import permit_for
        trial_permit = permit_for(self.runtime.lifecycle)
        research_directory = (trial_permit.paths['research'] if trial_permit is not None
                              else self.directory / 'research_run')
        manifest = {'version': 3 if 'service_restriction' in outer else 2, 'outer_clock': outer,
                    'work_authority': self.identity['work_authority'], 'work_schedule': self.schedule.to_dict(), 'month': month, 'cutoff': cutoff.isoformat(),
                    'snapshot_start': (cutoff - (pd.DateOffset(months=6) if self.runtime.spec.window_days is None
                        else pd.Timedelta(days=180)) - pd.Timedelta(hours=1234)).isoformat(),
                    'feed_directory': str(self.runtime.feed.directory), 'feed_id': self.runtime.feed.identity,
                    'run_directory': str(research_directory), 'config': asdict(self.config),
                    'rolling': asdict(self.runtime.spec), 'source_files': self.identity['source_files'],
                    'exposure_path': str(self.exposure_path), 'reserved_at': outer['issuer']['wall_started_at'],
                    'deadline': outer['wall_deadline']}
        if trial_permit is not None:
            from .quarantined_observation import current_binding
            manifest['observation_prefix'] = current_binding(self.runtime.lifecycle)
        check_service_work(self.runtime)
        from .s_outer_context import verify_job_clock
        verify_job_clock(manifest,directory/'job.json')
        _write(directory / 'job.json', manifest)
        state = {'month': month, 'job_path': str(directory / 'job.json'), 'job_hash': fingerprint(manifest),
                 'status': 'planned', 'attempts': [], 'validation': None}
        self._save(state)
        return state

    def _queue(self, state, now, service):
        if state['validation'] is not None or service is None:
            return
        manifest = _read(state['job_path'])
        store = RunStore(manifest['run_directory'], self.config.research, self.runtime.spec,
                         exposure_path=self.exposure_path, feed_identity=self.runtime.feed.identity,
                         mode='authority_scheduled_research_v1', work_contract={
                             'schedule': manifest['work_schedule'],
                             'authority_manifest_hash': manifest['work_authority']['manifest_hash']})
        try:
            measured = store.get(state['month'])
            if measured is None or measured['status'] != 'complete':
                raise ValueError('Successful worker lacks its complete frozen research checkpoint')
            if fingerprint(measured['summary']) != state['outcome']['summary_hash']:
                raise ValueError('Worker completion differs from its research summary')
            _verify_month_artifacts(Path(manifest['run_directory']), state['month'], measured)
            context = measured['search_exposure']
            if context['source'] != self.runtime.feed.identity or context['end'] != manifest['cutoff']:
                raise ValueError('Research result does not bind actual feed/cutoff exposure')
            store.exposure.assert_prompt_search(source=context['source'], symbols=context['symbols'],
                start=context['start'], end=context['end'], batch=context['batch'])
            reference = FrozenPool.from_dict(measured['frozen_reference_pool'])
            candidates = []
            for row in measured['summary']['shortlist']:
                record = measured['pool']['records'][row['formula_id']]
                if row['program'] != record['program'] or row['lookback'] != record['lookback']:
                    raise ValueError('Shortlist differs from its measured candidate')
                member = PoolMember(row['program'], lookback=row['lookback'],
                                    parameter_mode='profile' if row['program'].get('parameters') else 'fixed',
                                    family=record['family'])
                if len(reference.members) == 8 or member.member_id in {m.member_id for m in reference.members}:
                    continue
                pool = FrozenPool((*reference.members, member), version='V_' + member.member_id[:16],
                    reference_pipeline_id=reference.pipeline_id, mode='raw', role='confirmation',
                    max_profile_calls=reference.max_profile_calls, reference_pool=reference)
                recipe = self.runtime.recipe_for(pool)
                recipe['complexity'] = sum(_node_count(m.program) for m in pool.members)
                if recipe['pipeline_id'] not in {c['pipeline_id'] for c in candidates}:
                    candidates.append(recipe)
                if len(candidates) == 3:
                    break
            if not candidates:
                state['validation'] = {'status': 'no_candidate', 'A0_changed': False}
            else:
                occupied = {row[0] for row in self.runtime.lifecycle.db.execute('SELECT ordinal FROM batches')}
                queued = [item for item in service.state['freeze_requests'].values()
                          if item['status'] == 'pending']
                occupied.update(item['request']['ordinal'] for item in queued)
                free_slots = sorted(set(range(1, self.config.max_validation_batches + 1)) - occupied)
                if not free_slots:
                    state['validation'] = {'status': 'validation_slots_exhausted', 'A0_changed': False}
                else:
                    start = month_start(stamp(manifest['cutoff'])) + pd.DateOffset(months=1)
                    if start <= now:
                        state['validation'] = {'status': 'missed_next_month', 'A0_changed': False}
                    else:
                        request = {'batch': context['batch'], 'ordinal': free_slots[0],
                                   'start': start.isoformat(), 'candidates': candidates,
                                   'reference': self.runtime.recipe_for(reference),
                                   'evidence_sources': [{'start': context['start'], 'end': context['end']}]}
                        # Durable intent makes queue publication idempotent after a crash.
                        state['validation'] = {'status': 'queue_pending', 'request': request,
                                               'freeze_at': now.isoformat(), 'A0_changed': False}
                        self._save(state)
        finally:
            store.close()
        self._save(state)

    def _queue_algorithm(self, state, service):
        if service is None or state.get('algorithm_validation') is not None:
            return
        from .structural_validation import queue_structural_validation
        manifest = _read(state['job_path'])
        if (fingerprint(manifest) != state['job_hash']
                or manifest['source_files'] != self.identity['source_files']
                or manifest['work_authority']['manifest_hash'] != self.authority.manifest_hash):
            raise ValueError('Algorithm queue original scheduler job changed')
        result = queue_structural_validation(self.runtime, self.authority,
                                             manifest['run_directory'], state['month'])
        if state.get('algorithm_validation') not in (None, result):
            raise ValueError('Original algorithm queue result cannot change on restart')
        state['algorithm_validation'] = result
        self._save(state)

    def _publish_queue(self, state, service):
        item = state.get('validation')
        if item and item['status'] == 'queue_pending' and service is not None:
            existing = service.state['freeze_requests'].get(item['request']['batch'])
            when = existing['freeze_at'] if existing else self.runtime.lifecycle.now().isoformat()
            if stamp(item['request']['start']) <= self.runtime.lifecycle.now() and existing is None:
                item['status'] = 'missed_next_month'
                self._save(state)
                return
            service.queue_batch(freeze_at=when, **item['request'])
            item['freeze_at'] = when
            item['status'] = 'queued_V_only'
            self._save(state)

    def step(self, *, service=None):
        if service is None:
            return self._step_owned(service=None)
        from .forward_service_scope import _service_followup_scope
        with _service_followup_scope(service):
            return self._step_owned(service=service)

    def _step_owned(self, *, service=None):
        """Bounded status/spawn/queue work; numerical research always in the child."""
        from .forward_service_scope import check_service_work
        check_service_work(self.runtime)
        with FileLock(str(self.directory / '.lock'), timeout=0):
            if source_files() != self.identity['source_files']:
                raise ValueError('Frozen autonomous source changed')
            now = self.runtime.lifecycle.now()
            current_month = month_start(now).strftime('%Y-%m')
            months = [row[0] for row in self.db.execute('SELECT month FROM jobs ORDER BY month')]
            disk_months = {path.parent.name for path in (self.directory / 'jobs').glob('*/job.json')}
            if disk_months != set(months):
                raise ValueError('Research job index differs from immutable reservations; never reset attempts')
            stop = (self.directory / 'STOP').exists()
            actions, busy = [], False
            for month in months:
                state = self._load(month)
                manifest = _read(state['job_path'])
                if (fingerprint(manifest) != state['job_hash'] or manifest['month'] != month
                        or manifest['source_files'] != self.identity['source_files']):
                    raise ValueError('Frozen job differs from scheduler checkpoint')
                if state['status'] in {'planned', 'running', 'interrupted'}:
                    attempt = state['attempts'][-1] if state['attempts'] else None
                    identity = attempt.get('process') if attempt else None
                    if attempt and identity is None:
                        identity = self.runner.find(state['job_path'], attempt)
                        attempt['process'] = identity
                        self._save(state)
                    status = self.runner.status(identity) if identity else 'exited'
                    expired = now >= stamp(manifest['deadline'])
                    outcome_path = Path(state['job_path']).parent / f'outcome-{attempt["number"]}.json' if attempt else None
                    if outcome_path and outcome_path.exists() and status == 'exited':
                        outcome = _read(outcome_path)
                        self._accept_exited_outcome(state, manifest, attempt, outcome)
                    elif stop or expired:
                        if identity and status == 'running':
                            state['termination'] = self.runner.stop(identity)
                        state['status'] = 'stopped' if stop else 'deadline_consumed'
                    elif status == 'identity_mismatch':
                        state['status'] = 'process_identity_lost'
                    elif status == 'running':
                        busy = True
                    elif attempt and identity is None and (now - stamp(attempt['reserved_at'])).total_seconds() < 15:
                        busy = True  # A committed spawn may still be acquiring its process identity.
                    else:
                        state['status'] = 'interrupted' if attempt else 'planned'
                    self._save(state)
                if state['status'] == 'complete' and not stop:
                    check_service_work(self.runtime)
                    self._queue(state, now, service)
                    check_service_work(self.runtime)
                    self._publish_queue(state, service)
                    check_service_work(self.runtime)
                    self._queue_algorithm(state, service)
                actions.append({'month': month, 'status': state['status'], 'validation': state['validation']})
            if stop:
                return {'status': 'stopped', 'actions': actions}
            if self.config.not_before_month is not None and current_month < self.config.not_before_month:
                # Existing ownership, outcomes, STOP and queues were checked
                # above. This gate grants no new job or replacement attempt.
                return {'status': 'waiting_not_before', 'not_before_month': self.config.not_before_month,
                        'actions': actions, 'A0_changed': False}
            if not busy:
                state = self._load(current_month)
                if state is None:
                    check_service_work(self.runtime,new_work=True)
                    if self.runtime.spec.window_days is not None:
                        from .v7_feed import SnapshotUnavailable
                        try:
                            self.runtime.refresh_training_view()
                        except SnapshotUnavailable:
                            return {'status':'waiting_data_readiness','actions':actions,'A0_changed':False}
                    cutoff = month_start(now) if self.runtime.spec.window_days is None else now
                    state = self._plan(cutoff, now)
                if state['status'] in {'planned', 'interrupted'}:
                    if len(state['attempts']) >= self.config.max_attempts:
                        state['status'] = 'attempts_consumed'
                        self._save(state)
                    else:
                        check_service_work(self.runtime)
                        attempt = {'number': len(state['attempts']), 'token': uuid4().hex,
                                   'reserved_at': now.isoformat(), 'process': None}
                        state['attempts'].append(attempt)
                        state['status'] = 'running'
                        self._save(state)  # Reserve before spawn; never refund an unobserved launch.
                        check_service_work(self.runtime,new_work=True)
                        from .s_outer_context import check_clock
                        check_clock(_read(state['job_path'])['outer_clock'])
                        attempt['process'] = self.runner.start(state['job_path'], attempt)
                        self._save(state)
                        actions.append({'month': current_month, 'status': 'spawned', 'attempt': attempt['number']})
            return {'status': 'stepped', 'actions': actions, 'A0_changed': False}

    def stop(self, reason='operator_stop'):
        if reason not in {'operator_stop', 'supervisor_stopped', 'host_guard', 'supervisor_budget'}:
            raise ValueError('Unknown typed research stop')
        with FileLock(str(self.directory / '.lock'), timeout=0):
            stopped = []
            for month, in self.db.execute('SELECT month FROM jobs').fetchall():
                state = self._load(month)
                if state['status'] not in {'planned', 'running', 'interrupted'}:
                    continue
                attempt = state['attempts'][-1] if state['attempts'] else None
                identity = attempt.get('process') if attempt else None
                if attempt and identity is None:
                    identity = self.runner.find(state['job_path'], attempt)
                if identity and self.runner.status(identity) == 'running':
                    state['termination'] = self.runner.stop(identity)
                state.update(status='stopped', stop_reason=reason)
                self._save(state)
                stopped.append(month)
            return {'status': 'stopped', 'reason': reason, 'months': stopped}


def _node_count(value):
    if isinstance(value, dict):
        return int('op' in value) + sum(_node_count(v) for v in value.values())
    return sum(_node_count(v) for v in value) if isinstance(value, list) else 0


def sealed_snapshot(job_path):
    """ReceiptArchive as-of selection happens once, with explicit attrs restored."""
    job_path = Path(job_path)
    job = _read(job_path)
    path, receipt = job_path.parent / 'snapshot.parquet', job_path.parent / 'snapshot.json'
    if receipt.exists():
        identity = _read(receipt)
        if identity['job_hash'] != fingerprint(job) or not path.is_file() or file_hash(path) != identity['file_hash']:
            raise ValueError('Research snapshot file/receipt mismatch')
        frame = pd.read_parquet(path)
        if (list(frame.columns) != identity['columns'] or len(frame) != identity['rows']
                or {str(k): str(v) for k, v in frame.dtypes.items()} != identity['dtypes']):
            raise ValueError('Research snapshot schema or datetime precision changed')
        frame.attrs = identity['attrs']
        if job.get('observation_prefix') is not None:
            original = _observation_snapshot(job_path, job)
            if not original.equals(frame) or original.attrs != frame.attrs:
                raise ValueError('Original logical S snapshot prefix/frame changed')
        return frame
    if job.get('observation_prefix') is not None:
        frame = _observation_snapshot(job_path, job)
    else:
        feed = ReceiptArchive(job['feed_directory'], tuple(job['config']['research']['symbols']))
        try:
            if feed.identity != job['feed_id']:
                raise ValueError('Research feed identity changed')
            frame = feed.snapshot(job['cutoff'], start=job['snapshot_start'],
                                  archive_delay_seconds=job['rolling']['availability_delay_seconds'])
        finally:
            feed.close()
    if frame.attrs.get('feed_identity') != job['feed_id'] or frame.attrs.get('snapshot_asof') != job['cutoff']:
        raise ValueError('Snapshot lost receipt provenance')
    # An orphan parquet can only be accepted if exactly regenerated from the
    # same immutable receipt boundary; later receipts never enter this snapshot.
    temporary = path.with_suffix('.parquet.tmp')
    frame.to_parquet(temporary, index=False)
    if path.exists() and file_hash(path) != file_hash(temporary):
        temporary.unlink()
        raise ValueError('Uncommitted research snapshot cannot be replaced')
    os.replace(temporary, path)
    _write(receipt, {'job_hash': fingerprint(job), 'file_hash': file_hash(path), 'rows': len(frame),
                     'attrs': frame.attrs, 'columns': list(frame.columns), 'cutoff': job['cutoff'],
                     'dtypes': {str(k): str(v) for k, v in frame.dtypes.items()}})
    return frame


def _observation_snapshot(job_path, job):
    from .neural_forward_ticket import _LifecycleReader
    from .quarantined_observation import snapshot_for_prefix
    reader = _LifecycleReader(Path(job_path).parents[3])
    try:
        return snapshot_for_prefix(reader, job['observation_prefix'], job['cutoff'],
                                   months=6, context_hours=1234)
    finally:
        reader.close()


def run_worker(job_path, attempt, token, *, backend=None, enforce_resources=True):
    job_path = project_path(job_path)
    job = _read(job_path)
    with sqlite3.connect(job_path.parents[2] / 'scheduler.sqlite') as db:
        row = db.execute('SELECT value,hash FROM jobs WHERE month=?', (job['month'],)).fetchone()
    if row is None:
        raise ValueError('Worker has no durable scheduler reservation')
    reserved = json.loads(row[0])
    if (fingerprint(reserved) != row[1] or reserved['job_hash'] != fingerprint(job)
            or reserved['job_path'] != str(job_path) or reserved['status'] != 'running'
            or not reserved['attempts'] or reserved['attempts'][-1]['number'] != attempt
            or reserved['attempts'][-1]['token'] != token):
        raise ValueError('Worker launch differs from its durable reservation')
    from .s_outer_context import verify_job_clock
    verify_job_clock(job,job_path)
    if source_files() != job['source_files']:
        raise ValueError('Worker source differs from frozen job')
    if (job_path.parent / f'outcome-{attempt}.json').exists():
        raise ValueError('Closed autonomous attempt is readback only')
    if enforce_resources:
        inherited_limits()
    with FileLock(str(job_path.parent / '.worker.lock'), timeout=0):
        identity = {'job_hash': fingerprint(job), 'token': token, 'attempt': attempt}
        outcome = {**identity, 'status': 'failed'}
        finished = threading.Event()
        deadline = stamp(job['deadline']).timestamp()
        terminal_lock = threading.Lock()
        terminal, published = None, False

        def publish_terminal(value, *, expired=False):
            nonlocal terminal, published
            with terminal_lock:
                if terminal is None:
                    # The last clock check and terminal choice share the writer
                    # lock. A deadline decision never retains a summary hash.
                    from .s_outer_context import check_clock
                    try:check_clock(job['outer_clock'])
                    except (ValueError,TimeoutError):expired=True
                    terminal = ({**identity, 'status': 'deadline_consumed'}
                                if expired or time.time() >= deadline else dict(value))
                    terminal['finished_at'] = pd.Timestamp.now(tz='UTC').isoformat()
                if not published:
                    # Keep this exact payload even if publication raises. An
                    # interrupted writer may already have committed the file.
                    _write(job_path.parent / f'outcome-{attempt}.json', terminal)
                    published = True
                    finished.set()
                return terminal

        def watchdog():
            from .s_outer_context import _check_clock_live
            expired = False
            while not finished.wait(.2):
                try:
                    _check_clock_live(job['outer_clock'])
                except (ValueError, TimeoutError):
                    expired = True
                    break
            if expired:
                must_exit = True
                try:
                    decision = publish_terminal(identity, expired=True)
                    # A successfully published pre-deadline result can win the
                    # race with the timeout wakeup. Never rewrite that receipt.
                    must_exit = decision['status'] == 'deadline_consumed'
                finally:
                    if must_exit:
                        try:
                            for child in psutil.Process().children(recursive=True):
                                try:
                                    child.kill()
                                except psutil.NoSuchProcess:
                                    pass
                        finally:
                            # Disk errors, immutable-write conflicts or process
                            # inspection/kill failures cannot skip hard exit.
                            os._exit(124)

        if enforce_resources:
            threading.Thread(target=watchdog, daemon=True).start()
        try:
            if time.time() >= deadline:
                raise TimeoutError('Frozen research deadline already consumed')
            config_value = dict(job['config']['research'])
            config_value['symbols'] = tuple(config_value['symbols'])
            config = V7Config(**config_value)
            from .neural_forward_ticket import open_existing_authority
            from .s_outer_context import SOuterContext
            from .s_preparation import SPreparationEnvelope
            from .scheduled_search import FrozenSWorkSchedule
            from .work_lease import FrozenWorkLease
            # The original child baseline lease begins BEFORE reading receipts.
            outer = SOuterContext.autonomous(job_path, attempt, token)
            from .s_outer_context import require_lease_headroom
            require_lease_headroom(outer._verify()['clock'])
            origin = FrozenWorkLease.issue(300)
            authority = open_existing_authority(job['work_authority']['directory'])
            if authority.manifest_hash != job['work_authority']['manifest_hash']:
                raise ValueError('Autonomous shared wallet identity changed')
            schedule = FrozenSWorkSchedule.from_dict(job['work_schedule'])
            from .scheduled_search import validate_schedule_authority
            validate_schedule_authority(authority, schedule)
            source_reference = {'feed_identity': job['feed_id'], 'feed_directory': job['feed_directory'],
                                'exposure_path': job['exposure_path'], 'cutoff': job['cutoff']}
            if job.get('observation_prefix') is not None:
                source_reference['observation_prefix'] = job['observation_prefix']
            preparation = SPreparationEnvelope.prepare(authority, job['run_directory'], job['month'],
                config=asdict(config), spec=job['rolling'], schedule_hash=schedule.digest,
                source_reference=source_reference,
                clock_scope='before_feed_snapshot', lease=origin, outer_context=outer)
            if not preparation._executable:
                raise ValueError('Original autonomous preparation is readback only')
            try:
                bars = sealed_snapshot(job_path)
                preparation.check()
                origin.assert_parent_alive()
            except BaseException as error:
                try:
                    authority.fail_consumed(preparation.baseline_record()['reservation_id'], 'snapshot_failed')
                except BaseException as accounting_error:  # noqa: BLE001 -- preserve source failure.
                    error.add_note('Snapshot failure accounting: ' + type(accounting_error).__name__)
                raise
            if config.search_mode == 'gpt' and backend is None:
                if job['config']['backend'] != 'codex':
                    raise ValueError('No explicitly configured research backend')
                from .full_trial_controller import backend_for_job
                backend = backend_for_job(job_path)
                if backend is None:
                    from .v7_agent import CodexBatchBackend
                    backend = CodexBatchBackend(job_path.parent / 'provider')
            elif backend is not None:
                from .full_trial_controller import backend_for_job
                if backend_for_job(job_path) is not None:
                    raise ValueError('Registered full trial cannot inject a provider callback')
            summary = run_month(job['run_directory'], bars, job['cutoff'], config=config,
                spec=RollingSpec(**job['rolling']), backend=backend, exposure_path=job['exposure_path'],
                feed_directory=job['feed_directory'], authority=authority, schedule=schedule, preparation=preparation)
            outer._verify()
            if time.time() >= deadline:
                raise TimeoutError('Research completed after its frozen deadline')
            outcome.update(status='complete', summary_hash=fingerprint(summary))
            return summary
        except (MemoryError, ValueError, OSError, TimeoutError) as exc:
            outcome.update(status='resource_incomplete' if isinstance(exc, MemoryError) else 'failed',
                           error_type=type(exc).__name__, error=str(exc)[:300])
            raise
        finally:
            decision = publish_terminal(outcome)
            if decision['status'] == 'deadline_consumed':
                raise TimeoutError('Frozen research deadline consumed')


class AutonomousService:
    def __init__(self, service, scheduler):
        if service.runtime is not scheduler.runtime:
            raise ValueError('Autonomous and forward scheduling require the same runtime')
        self.service, self.scheduler = service, scheduler

    def step(self, *, session=None):
        forward = self.service.step(session=session, defer_fits=self.scheduler.worker_active())
        if forward.get('tick', {}).get('serving_only_isolated_jobs'):
            research = self.scheduler.stop('supervisor_stopped')
        elif forward['status'] == 'stepped':
            research = self.scheduler.step(service=self.service)
        else:
            reason = ('host_guard' if forward['status'] == 'blocked_by_host_guard' else
                      'supervisor_stopped' if forward['status'] == 'stopped' else 'supervisor_budget')
            research = self.scheduler.stop(reason)
        return {'forward': forward, 'research': research}

    def run(self, *, session=None):
        try:
            return self._run(session=session)
        finally:
            # CLI shutdown, including an exception or keyboard interrupt, must
            # not leave research running after the supervisor has stopped.
            self.scheduler.stop('supervisor_stopped')

    def _run(self, *, session=None):
        """Minute cadence with the forward service's same durable wall budget."""
        service = self.service
        began = service.monotonic()
        last = None
        while service.monotonic() - began < service.config.wall_seconds:
            last = self.step(session=session)
            if last['forward']['status'] != 'stepped':
                return last
            remaining = min(service.config.wall_seconds - (service.monotonic() - began),
                            service.config.wall_seconds - service.state['spent_seconds'])
            if remaining <= 0:
                break
            slept = service.monotonic()
            service.state['active_since'] = service.lifecycle.now().isoformat()
            service._save('autonomous_sleep_started')
            try:
                service.sleeper(min(service.config.tick_seconds, remaining))
            finally:
                service.state['spent_seconds'] += max(0., service.monotonic() - slept)
                service.state['active_since'] = None
                service._save('autonomous_sleep_finished')
        return {'status': 'invocation_wall_limit', 'last_step': last,
                'research': self.scheduler.stop('supervisor_budget')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['worker'])
    parser.add_argument('--job', required=True)
    parser.add_argument('--attempt', required=True, type=int)
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    run_worker(args.job, args.attempt, args.token)


if __name__ == '__main__':
    main()
