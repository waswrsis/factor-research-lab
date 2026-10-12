"""Explicit finite campaign around unchanged V7 core; never installs a service.

Run this controller inside an already constrained Linux cgroup. The one owned
campaign worker rotates bounded internal sessions without stopping its research
child at each session boundary. Research keeps the original scheduler/run and
budgets. Every session prepays its full wall/step allowance without refunds.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

import pandas as pd
import psutil
from filelock import FileLock

from factorlab.discovery.evidence import file_hash
from factorlab.discovery.process_policy import isolated_process_kwargs, restricted_environment
from factorlab.discovery.v7_autonomous import (
    AutonomousConfig,
    AutonomousService,
    ResearchScheduler,
    SubprocessRunner,
    inherited_limits,
)
from factorlab.discovery.v7_feed import TransientFeedError
from factorlab.discovery.v7_forward import ForwardConfig, ForwardRuntime
from factorlab.discovery.v7_lifecycle import LifecyclePlan, canonical, stamp
from factorlab.discovery.v7_run import V7Config, source_files
from factorlab.discovery.v7_service import ForwardService, HostGuard, ServiceConfig
from factorlab.discovery.v7_storage import immutable_record
from factorlab.protocol import fingerprint, project_path


def now():
    return pd.Timestamp.now(tz='UTC')


class SessionWindowClosed(RuntimeError):
    """Insufficient prepaid time for another complete work lease; rotate only."""


def boot_identity():
    """Linux monotonic clocks survive process restart, never machine reboot."""
    if sys.platform == 'linux':
        value = Path('/proc/sys/kernel/random/boot_id').read_text('ascii').strip()
        if not re.fullmatch(r'[0-9a-f-]{36}', value):
            raise ValueError('Unverified Linux boot identity')
        return value
    # Production workers already refuse non-Linux hosts. Synthetic unit tests
    # may use this process-local identity; no cross-process clock claim there.
    return f'nonlinux-test-process-{os.getpid()}'


@dataclass(frozen=True)
class CampaignPlan:
    end_at: str
    max_sessions: int
    max_reserved_seconds: int
    max_reserved_steps: int
    max_fit_attempts: int
    forward_directory: str
    feed_directory: str
    research: dict
    lifecycle: dict
    symbols: tuple[str, ...] = ('BTCUSDT',)
    session_seconds: int = 7200
    session_steps: int = 120
    tick_seconds: int = 60
    polling_pages: int = 4
    step_limit_seconds: int = 300
    max_consecutive_transport_failures: int = 5
    old_unit: str = 'fr-regression-active.service'
    old_exec_start: str = '/tools/long80/supervise_iteration.py pilot'
    work_authority: dict | None = None
    work_schedule: dict | None = None
    old_service_receipt: str | None = None
    old_service_receipt_sha256: str | None = None
    research_spec: dict | None = None

    def __post_init__(self):
        stamp(self.end_at)
        for key, ceiling in {'max_sessions': 5000, 'max_reserved_seconds': 366 * 86400,
                'max_reserved_steps': 1_000_000, 'max_fit_attempts': 10000,
                'session_seconds': 7200, 'session_steps': 7200, 'tick_seconds': 300,
                'step_limit_seconds': 300, 'max_consecutive_transport_failures': 5}.items():
            value = getattr(self, key)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError(f'Invalid finite campaign {key}')
        if (self.session_seconds < 4 or self.session_seconds > self.max_reserved_seconds or self.session_steps > self.max_reserved_steps
                or not re.fullmatch(r'[A-Za-z0-9_.@-]+[.]service', self.old_unit)
                or not re.search(r'/tools/long80/supervise_iteration[.]py(?:\s|$)', self.old_exec_start)):
            raise ValueError('Invalid campaign session allowance or frozen legacy unit identity')
        ForwardConfig(symbols=tuple(self.symbols), polling_pages=self.polling_pages)
        research = research_config(self.research, self.symbols)
        if self.work_authority is not None or self.work_schedule is not None:
            from factorlab.discovery.scheduled_search import FrozenSWorkSchedule
            if (not isinstance(self.work_authority, dict)
                    or set(self.work_authority) != {'directory', 'manifest_hash'}
                    or not re.fullmatch('[0-9a-f]{64}', self.work_authority['manifest_hash'])):
                raise ValueError('Actual existing campaign wallet reference required')
            schedule = FrozenSWorkSchedule.from_dict(self.work_schedule)
            if fingerprint(schedule.to_dict()['config']) != fingerprint(asdict(research.research)):
                raise ValueError('Campaign work schedule differs from research config')
        lifecycle = dict(self.lifecycle)
        lifecycle['symbols'] = tuple(lifecycle.get('symbols', self.symbols))
        if lifecycle['symbols'] != tuple(self.symbols):
            raise ValueError('Campaign lifecycle universe differs')

        checked_plan = LifecyclePlan(**lifecycle)
        if self.research_spec is not None:
            from factorlab.discovery.layered_research import LayeredResearchSpec
            if not isinstance(self.research_spec, dict):
                raise ValueError('Explicit layered research spec object required')
            LayeredResearchSpec(**self.research_spec)
            if ((checked_plan.training_window_days, checked_plan.refit_interval_days) != (90, 15)
                    or checked_plan.calibration_directory is not None
                    or checked_plan.calibration_binding is not None
                    or checked_plan.confirmation_months != 3 or checked_plan.algorithm_months != 6):
                raise ValueError('Layered campaign cannot mix calendar mode or legacy certificates')
        if project_path(self.forward_directory) == project_path(self.feed_directory):
            raise ValueError('Campaign feed and forward directories must differ')
        if bool(self.old_service_receipt) != bool(self.old_service_receipt_sha256):
            raise ValueError('Legacy operator receipt needs its independently frozen file hash')
        if self.old_service_receipt_sha256 is not None and not re.fullmatch('[0-9a-f]{64}', self.old_service_receipt_sha256):
            raise ValueError('Invalid frozen legacy receipt hash')

    @property
    def effective_step_seconds(self):
        # Short sessions reserve half the non-cleanup window for setup/idle
        # overhead: consuming the entire window would make any real startup
        # latency deny every step. This is frozen, never a deadline extension.
        return min(self.step_limit_seconds, (self.session_seconds - 2) // 2)


def campaign_plan_fields(plan):
    fields = asdict(plan)
    if plan.research_spec is None:
        fields.pop('research_spec')  # Preserve the legacy default manifest shape.
    return fields


def campaign_runtime_kwargs(plan, clock):
    lifecycle = dict(plan.lifecycle)
    lifecycle['symbols'] = tuple(lifecycle.get('symbols', plan.symbols))
    result = {'config': ForwardConfig(symbols=tuple(plan.symbols), polling_pages=plan.polling_pages,
                    training_window_days=lifecycle.get('training_window_days'),
                    refit_interval_days=lifecycle.get('refit_interval_days')),
              'lifecycle_plan': LifecyclePlan(**lifecycle), 'clock': clock}
    if plan.research_spec is not None:
        from factorlab.discovery.layered_research import LayeredResearchSpec
        result['research_spec'] = LayeredResearchSpec(**plan.research_spec)
    return result


def research_config(value, symbols):
    fields = copy.deepcopy(value)
    research = fields.pop('research', {})
    research.setdefault('symbols', tuple(symbols))
    research['symbols'] = tuple(research['symbols'])
    fields['research'] = V7Config(**research)
    result = AutonomousConfig(**fields)
    if tuple(result.research.symbols) != tuple(symbols):
        raise ValueError('Campaign research universe differs')
    return result



def actual_work_contract(plan):
    """Legacy metadata may be inspected; execution never creates a new wallet."""
    from factorlab.discovery.neural_forward_ticket import open_existing_authority
    from factorlab.discovery.scheduled_search import FrozenSWorkSchedule
    if plan.work_authority is None or plan.work_schedule is None:
        raise ValueError('Campaign execution requires actual shared authority and frozen S schedule')
    authority = open_existing_authority(plan.work_authority['directory'])
    if authority.manifest_hash != plan.work_authority['manifest_hash']:
        raise ValueError('Campaign shared authority identity changed')
    schedule = FrozenSWorkSchedule.from_dict(plan.work_schedule)
    from factorlab.discovery.scheduled_search import validate_schedule_authority
    validate_schedule_authority(authority, schedule)
    if fingerprint(schedule.to_dict()['config']) != fingerprint(asdict(research_config(plan.research, plan.symbols).research)):
        raise ValueError('Campaign schedule no longer matches its immutable config')
    return authority, schedule

def _enveloped(path):
    record = json.loads(Path(path).read_text('utf-8'))
    if set(record) != {'value', 'sha256'} or fingerprint(record['value']) != record['sha256']:
        raise ValueError('Inherited research receipt integrity failure')
    return record['value']


def _research_anchor(state):
    attempts = state['attempts']
    prefix = []
    if len(attempts) > 3:
        raise ValueError('Research attempt anchor exceeds its frozen maximum')
    for index, attempt in enumerate(attempts):
        if attempt['number'] != index or not isinstance(attempt['token'], str) or not attempt['token']:
            raise ValueError('Research attempt anchor identity differs')
        prefix.append({key: attempt[key] for key in ('number', 'token', 'reserved_at')})
    return {'job_hash': state['job_hash'], 'attempts': prefix, 'status': state['status'],
            'outcome_hash': fingerprint(state['outcome']) if state.get('outcome') is not None else None}


def _advance_research_anchor(old, new):
    if (new['job_hash'] != old['job_hash'] or len(new['attempts']) < len(old['attempts'])
            or new['attempts'][:len(old['attempts'])] != old['attempts']
            or (old['outcome_hash'] is not None and old['outcome_hash'] != new['outcome_hash'])
            or (old['status'] not in {'planned', 'running', 'interrupted'} and old['status'] != new['status'])):
        raise ValueError('Inherited research reservation/attempt/result anchor rewound or changed')


def _historical_outer_matches(job, seconds):
    """Read immutable original job timing without renewing a historical clock."""
    from factorlab.discovery.work_lease import FrozenWorkLease
    outer = job['outer_clock']
    origin = FrozenWorkLease.from_dict(outer['issuer'])
    return (job.get('version') == 2 and outer['seconds'] == seconds
            and outer['monotonic_deadline'] == origin.monotonic_started + seconds
            and stamp(outer['wall_deadline']) == stamp(origin.wall_started_at) + pd.Timedelta(seconds=seconds)
            and job['deadline'] == outer['wall_deadline'])


def inherited_research_months(plan, core, observed, *, include_anchors=False):
    """Freeze existing month identities once; never grant them fresh budgets."""
    forward = project_path(plan.forward_directory)
    autonomous = forward / 'autonomous'
    database = autonomous / 'scheduler.sqlite'
    if not database.exists():
        if autonomous.exists() and any(autonomous.iterdir()):
            raise ValueError('Inherited research scheduler journal missing')
        return {} if include_anchors else []
    runtime = json.loads((forward / 'forward.json').read_text('utf-8'))
    identity = _enveloped(autonomous / 'identity.json')
    if (fingerprint(runtime['identity']) != runtime['sha256']
            or runtime['identity']['source_files'] != core or identity['source_files'] != core
            or identity['forward_identity'] != fingerprint(runtime['identity'])
            or identity['feed_id'] != plan.lifecycle['source_id']
            or identity['feed_id'] != runtime['identity']['lifecycle']['source_id']
            or project_path(runtime['identity']['feed_directory']) != project_path(plan.feed_directory)
            or project_path(identity['exposure_path']) != forward / 'exposure.sqlite'
            or fingerprint(identity['config']) != fingerprint(asdict(research_config(plan.research, plan.symbols)))
            or identity.get('work_authority') != plan.work_authority
            or identity.get('work_schedule') != plan.work_schedule
            or runtime['identity'].get('research_spec') != plan.research_spec):
        raise ValueError('Inherited research runtime/source/config identity differs from campaign')
    db = sqlite3.connect(f'{database.as_uri()}?mode=ro', uri=True)
    try:
        months, anchors = [], {}
        for month, text, digest in db.execute('SELECT month,value,hash FROM jobs ORDER BY month'):
            state = json.loads(text)
            if (not re.fullmatch(r'\d{4}-(0[1-9]|1[0-2])', month)
                    or fingerprint(state) != digest or state['month'] != month):
                raise ValueError('Inherited research job/month integrity failure')
            job_path = project_path(state['job_path'])
            job = _enveloped(job_path)
            cutoff = pd.Timestamp(month + '-01', tz='UTC')
            if (job_path != autonomous / 'jobs' / month / 'job.json' or job['month'] != month
                    or stamp(job['cutoff']) != cutoff or cutoff > stamp(observed)
                    or fingerprint(job) != state['job_hash'] or job['source_files'] != core
                    or job['feed_id'] != identity['feed_id']
                    or project_path(job['feed_directory']) != project_path(plan.feed_directory)
                    or project_path(job['run_directory']) != autonomous / 'research_run'
                    or project_path(job['exposure_path']) != forward / 'exposure.sqlite'
                    or fingerprint(job['rolling']) != fingerprint(runtime['identity']['rolling'])
                    or not cutoff <= stamp(job['reserved_at']) <= stamp(observed)
                    or not _historical_outer_matches(job, identity['config']['deadline_seconds'])
                    or fingerprint(job['config']) != fingerprint(identity['config'])
                    or job.get('work_authority') != plan.work_authority
                    or job.get('work_schedule') != plan.work_schedule):
                raise ValueError('Inherited monthly research reservation differs from frozen identity')
            if state.get('outcome') is not None:
                attempt = state['attempts'][-1]
                result = _enveloped(job_path.parent / f'outcome-{attempt["number"]}.json')
                if (result != state['outcome'] or result['job_hash'] != state['job_hash']
                        or result['token'] != attempt['token']
                        or (result['status'] == 'complete' and stamp(result['finished_at']) > stamp(job['deadline']))):
                    raise ValueError('Inherited research result differs from saved attempt receipt')
            elif state['status'] == 'complete':
                raise ValueError('Inherited completed research has no retained outcome')
            months.append(month)
            anchors[month] = _research_anchor(state)
        if set(months) != {p.parent.name for p in (autonomous / 'jobs').glob('*/job.json')}:
            raise ValueError('Inherited monthly journal differs from on-disk reservations')
        return anchors if include_anchors else months
    finally:
        db.close()


class CampaignStore:
    """One hash-chained transaction journal shared by parent and its worker."""
    def __init__(self, directory, plan, *, clock=now):
        self.directory, self.plan, self.clock = project_path(directory), plan, clock
        self.directory.mkdir(parents=True, exist_ok=True)
        manifest_path, db_path = self.directory / 'campaign.json', self.directory / 'campaign.sqlite'
        existed = manifest_path.exists()
        prior = json.loads(manifest_path.read_text('utf-8')) if existed else None
        if prior and fingerprint(prior['identity']) != prior['sha256']:
            raise ValueError('Campaign immutable identity integrity failure')
        began_at = prior['identity']['began_at'] if prior else stamp(clock()).isoformat()
        first_month = stamp(began_at).normalize().replace(day=1)
        months = [t.strftime('%Y-%m') for t in pd.date_range(first_month, stamp(plan.end_at), freq='MS', inclusive='left')]
        core = source_files()
        inherited_anchors = (prior['identity']['inherited_research_anchors'] if prior else
                             inherited_research_months(plan, core, stamp(clock()), include_anchors=True))
        inherited = sorted(inherited_anchors)
        budget_months = sorted(set(months) | set(inherited))
        self.identity = {'version': 1, 'plan': campaign_plan_fields(plan), 'core': core,
                         'adapter_sha256': file_hash(Path(__file__)),
                         'python': str(Path(sys.executable).resolve()), 'began_at': began_at,
                         'eligible_research_months': months,
                         'inherited_research_months': inherited, 'research_budget_months': budget_months,
                         'inherited_research_anchors': inherited_anchors,
                         'research_cumulative_ceiling': {key: len(budget_months) * value for key, value in
                             {'gpt_attempt': 6, 'family': 3, 'ast': 96, 'f': 24, 'solve': 1600}.items()},
                         'research_budget_policy': 'same_existing_monthly_ledger_no_additional_allowance'}
        if existed and not db_path.exists():
            raise ValueError('Campaign journal missing; cannot reset consumed budget')
        if not existed and any(p.name != '.owner.lock' for p in self.directory.iterdir()):
            raise ValueError('Campaign identity missing from nonempty directory')
        if not existed and not stamp(clock()) < stamp(plan.end_at) <= stamp(clock()) + pd.Timedelta(days=366):
            raise ValueError('New campaign deadline must be within the next 366 days')
        immutable_record(manifest_path, {'identity': self.identity, 'sha256': fingerprint(self.identity)})
        immutable_record(project_path(plan.forward_directory) / 'long-campaign.json',
                         {'directory': str(self.directory), 'campaign_identity': fingerprint(self.identity)})
        self.db = sqlite3.connect(db_path, timeout=10)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY,value TEXT,hash TEXT);'
                             'CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY,value TEXT,hash TEXT);')
        if self.db.execute('SELECT COUNT(*) FROM state').fetchone()[0] == 0:
            if existed or self.db.execute('SELECT COUNT(*) FROM events').fetchone()[0]:
                self.db.close()
                raise ValueError('Campaign checkpoint missing from existing journal')
            self._write({'sessions': [], 'launches': [], 'fit_attempts': {}, 'freeze_requests': {},
                         'research_anchors': copy.deepcopy(inherited_anchors),
                         'active_step': None, 'last_interrupted_step': None,
                         'consecutive_transport_failures': 0, 'total_transport_failures': 0,
                         'terminal': None, 'last_at': stamp(clock()).isoformat()}, 'created')
        self._verified_head = None
        self.last_verification_rows = 0
        try:
            saved_state = self.load()
            if existed:
                self.verify_research_anchors(saved_state)
        except BaseException:
            self.db.close()
            raise

    def close(self):
        self.db.close()

    def verify_research_anchors(self, saved_state=None):
        current = inherited_research_months(self.plan, self.identity['core'], stamp(self.clock()), include_anchors=True)
        saved = (self.load() if saved_state is None else saved_state)['research_anchors']
        if not set(current) <= set(self.identity['research_budget_months']):
            raise ValueError('Research reservation month is outside the frozen campaign budget union')
        for month, old in saved.items():
            if month not in current:
                raise ValueError('Inherited research reservation anchor missing from scheduler journal')
            _advance_research_anchor(old, current[month])
        if current != saved:
            self.update('research_progress_anchored', lambda state: state.update(research_anchors=copy.deepcopy(current)))

    def record_research(self, state):
        new = _research_anchor(state)
        month = state['month']
        if month not in self.identity['research_budget_months']:
            raise ValueError('Research month lies outside the frozen campaign budget union')

        def change(campaign):
            old = campaign['research_anchors'].get(month)
            if old is not None:
                _advance_research_anchor(old, new)
            campaign['research_anchors'][month] = copy.deepcopy(new)
        self.update('research_reservation_anchored_before_local_save', change)

    def load(self):
        owns_read = not self.db.in_transaction
        if owns_read:
            self.db.execute('BEGIN')
        try:
            return self._load_snapshot()
        finally:
            if owns_read:
                self.db.rollback()  # End a consistent read snapshot; never mutate.

    def _load_snapshot(self):
        row = self.db.execute('SELECT value,hash FROM state WHERE id=1').fetchone()
        if not row or fingerprint(json.loads(row[0])) != row[1]:
            raise ValueError('Campaign checkpoint integrity failure')
        head = self.db.execute('SELECT seq,hash FROM events ORDER BY seq DESC LIMIT 1').fetchone()
        cursor, previous, last = self._verified_head or (-1, '0' * 64, None)
        if (head is None or head[0] < cursor
                or (head[0] == cursor and head[1] != previous)):
            raise ValueError('Campaign journal head missing, rewound or changed')
        if cursor >= 0:
            anchor = self.db.execute('SELECT value,hash FROM events WHERE seq=?', (cursor,)).fetchone()
            if anchor is None or anchor[1] != previous or fingerprint(json.loads(anchor[0])) != previous:
                raise ValueError('Campaign verified journal anchor missing or changed')
        self.last_verification_rows = 0
        for seq, text, digest in self.db.execute('SELECT seq,value,hash FROM events WHERE seq>? ORDER BY seq', (cursor,)):
            event = json.loads(text)
            if (seq != cursor + 1 or event['seq'] != seq or event['previous'] != previous
                    or fingerprint(event) != digest):
                raise ValueError('Campaign journal integrity failure')
            cursor, previous, last = seq, digest, event['state_hash']
            self.last_verification_rows += 1
        if last != row[1] or (cursor, previous) != head:
            raise ValueError('Campaign checkpoint differs from journal head')
        self._verified_head = (cursor, previous, last)
        return json.loads(row[0])

    def _write(self, state, kind):
        digest = fingerprint(state)
        row = self.db.execute('SELECT seq,hash FROM events ORDER BY seq DESC LIMIT 1').fetchone()
        event = {'seq': row[0] + 1 if row else 0, 'previous': row[1] if row else '0' * 64,
                 'kind': kind, 'state_hash': digest}
        self.db.execute('INSERT INTO events VALUES (?,?,?)', (event['seq'], canonical(event), fingerprint(event)))
        self.db.execute('INSERT OR REPLACE INTO state VALUES (1,?,?)', (canonical(state), digest))
        self.db.commit()

    def update(self, kind, change):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            state = self.load()
            observed = stamp(self.clock())
            if observed < stamp(state['last_at']):
                raise ValueError('Campaign clock moved backward')
            result = change(state)
            state['last_at'] = observed.isoformat()
            self._write(state, kind)
            return result
        except BaseException:
            self.db.rollback()
            raise

    def terminal(self, reason):
        def change(state):
            if state['terminal'] is None:
                state['terminal'] = {'reason': reason, 'at': stamp(self.clock()).isoformat()}
            return state['terminal']
        return self.update('terminal', change)

    def reserve_session(self, launch_token=None):
        def change(state):
            seconds = sum(s['wall_seconds'] for s in state['sessions'])
            steps = sum(s['max_steps'] for s in state['sessions'])
            if (state['terminal'] or stamp(self.clock()) >= stamp(self.plan.end_at)
                    or (stamp(self.plan.end_at) - stamp(self.clock())).total_seconds() < self.plan.effective_step_seconds + 2
                    or len(state['sessions']) >= self.plan.max_sessions
                    or seconds + self.plan.session_seconds > self.plan.max_reserved_seconds
                    or steps + self.plan.session_steps > self.plan.max_reserved_steps):
                return None
            item = {'session_id': f'campaign-{fingerprint(self.identity)[:12]}-{len(state["sessions"])}',
                    'wall_seconds': self.plan.session_seconds, 'max_steps': self.plan.session_steps,
                    'reserved_at': stamp(self.clock()).isoformat(), 'launch_token': launch_token}
            state['sessions'].append(item)
            if state['active_step'] is not None:
                state['last_interrupted_step'] = state['active_step']
                state['active_step'] = None
            return copy.deepcopy(item)
        return self.update('session_reserved_no_refund', change)

    def begin_step(self, session_id):
        def change(state):
            if not state['sessions'] or state['sessions'][-1]['session_id'] != session_id:
                raise ValueError('Step requires the current prepaid session')
            session = state['sessions'][-1]
            began_monotonic = time.monotonic()
            deadline = min(stamp(session['reserved_at']) + pd.Timedelta(seconds=session['wall_seconds']),
                           stamp(self.plan.end_at))
            observed = stamp(self.clock())
            if state['active_step'] is not None or state['terminal']:
                raise RuntimeError('No work may start after its prepaid session deadline')
            if (deadline - observed).total_seconds() < self.plan.effective_step_seconds + 2:
                raise SessionWindowClosed('Insufficient complete work lease before session deadline')
            deadline = min(deadline, observed + pd.Timedelta(seconds=self.plan.effective_step_seconds))
            if not math.isfinite(began_monotonic):
                raise ValueError('Invalid original monotonic work clock')
            active = {'id': uuid4().hex, 'session_id': session_id, 'deadline': deadline.isoformat(),
                      'launch_token': session['launch_token'], 'began_at': observed.isoformat(),
                      'boot_identity': boot_identity(), 'monotonic_started': began_monotonic,
                      'monotonic_deadline': began_monotonic + (deadline - observed).total_seconds()}
            state['active_step'] = active
            return copy.deepcopy(active)
        return self.update('step_reserved_with_deadline', change)

    def record_transport_outcome(self, error=None):
        """Only a completed good tick resets a durable consecutive-failure cap."""
        def change(state):
            if error is None:
                state['consecutive_transport_failures'] = 0
            else:
                state['consecutive_transport_failures'] += 1
                state['total_transport_failures'] += 1
                if state['consecutive_transport_failures'] >= self.plan.max_consecutive_transport_failures:
                    state['terminal'] = {'reason': 'transient_transport_allowance_consumed',
                                         'at': stamp(self.clock()).isoformat()}
            return {'consecutive': state['consecutive_transport_failures'],
                    'total': state['total_transport_failures'], 'error': error,
                    'exhausted': state['terminal'] is not None}
        return self.update('transport_tick_failed' if error else 'transport_tick_succeeded', change)

    def end_step(self, identifier):
        def change(state):
            active = state['active_step']
            if active is None or active['id'] != identifier:
                raise ValueError('Step completion differs from the active reservation')
            expired = active_step_expired(state, active['launch_token'], {}, clock=self.clock)
            if expired and state['terminal'] is None:
                state['terminal'] = {'reason': 'session_step_deadline', 'at': stamp(self.clock()).isoformat()}
            state['active_step'] = None
            return expired
        expired = self.update('step_finished', change)
        if expired:
            raise TimeoutError('Work finished after its prepaid session deadline')

    def reserve_fit(self, job, value):
        def change(state):
            previous = state['fit_attempts'].get(job, {'count': 0, 'last_at': None})
            observed = stamp(self.clock())
            requested_at = stamp(value['last_at'])
            if (type(value['count']) is not int or value['count'] != previous['count'] + 1 or value['count'] > 3
                    or requested_at > observed
                    or (previous['last_at'] is not None and requested_at < stamp(previous['last_at']))):
                raise ValueError('Global fit reservation differs from durable per-job attempts')
            if (state['terminal'] or stamp(self.clock()) >= stamp(self.plan.end_at)
                    or sum(v['count'] for v in state['fit_attempts'].values()) >= self.plan.max_fit_attempts):
                raise RuntimeError('Global fit attempt allowance consumed')
            reserved = {'count': value['count'], 'last_at': observed.isoformat()}
            state['fit_attempts'][job] = reserved
            return copy.deepcopy(reserved)
        return self.update('fit_reserved_no_refund', change)

    def save_queues(self, requests):
        if requests == self.load()['freeze_requests']:
            return
        def change(state):
            for key, old in state['freeze_requests'].items():
                new = requests.get(key)
                if (new is None or new['request'] != old['request'] or new['freeze_at'] != old['freeze_at']
                        or (old['status'] != 'pending' and new['status'] != old['status'])):
                    raise ValueError('Carried V/C freeze identity or final status cannot change')
            state['freeze_requests'] = copy.deepcopy(requests)
        self.update('freeze_queue_carried', change)


class CampaignScheduler(ResearchScheduler):
    """The global reservation survives deletion/rollback of a scheduler row."""
    def __init__(self, runtime, store, **kwargs):
        self.campaign = store
        super().__init__(runtime, **kwargs)

    def _save(self, state):
        self.campaign.record_research(state)
        super()._save(state)


class CampaignService(ForwardService):
    """Global reservation first; local crash cannot return a fit attempt."""
    def __init__(self, runtime, store, **kwargs):
        self.campaign = store
        super().__init__(runtime, **kwargs)
        try:
            self._carry()
            super()._save('global_attempts_and_queue_carried')
        except BaseException:
            self.close(close_runtime=False)
            raise

    def _carry(self):
        shared = self.campaign.load()
        for job, local in self.state['fit_attempts'].items():
            if shared['fit_attempts'].get(job, {}).get('count', 0) < local['count']:
                raise ValueError('Local fit count exceeds authoritative global reservation')
        for key, local in self.state['freeze_requests'].items():
            old = shared['freeze_requests'].get(key)
            if old is None or old['request'] != local['request'] or old['freeze_at'] != local['freeze_at']:
                raise ValueError('Local queue differs from globally frozen V/C identity')
        self.state['fit_attempts'] = copy.deepcopy(shared['fit_attempts'])
        self.state['freeze_requests'] = copy.deepcopy(shared['freeze_requests'])

    def _save(self, event):
        if event in {'created', 'interrupted_time_charged'}:
            self._carry()
        elif event == 'fit_attempt_reserved':
            global_counts = self.campaign.load()['fit_attempts']
            changed = [(job, value) for job, value in self.state['fit_attempts'].items()
                       if value != global_counts.get(job)]
            if len(changed) != 1:
                raise ValueError('Exactly one global fit reservation required')
            try:
                job, value = changed[0]
                self.state['fit_attempts'][job] = self.campaign.reserve_fit(job, value)
            except RuntimeError:
                self.state['fit_attempts'] = copy.deepcopy(global_counts)
                self.campaign.terminal('global_fit_allowance_consumed')
                raise
        self.campaign.save_queues(self.state['freeze_requests'])
        super()._save(event)


def _owned_memory_snapshot():
    """Read credit only from this process's directly capped service cgroup."""
    if sys.platform != 'linux':
        raise ValueError('Owned service headroom requires real Linux cgroup v2')
    root = Path('/sys/fs/cgroup').resolve()
    lines = Path('/proc/self/cgroup').read_text().splitlines()
    unified = [line[3:] for line in lines if line.startswith('0::')]
    if len(unified) != 1:
        raise ValueError('One actual unified process cgroup is required')
    node = (root / unified[0].lstrip('/')).resolve()
    if (not node.is_relative_to(root) or node == root
            or not node.name.endswith('.service') or node.name.startswith('user@')):
        raise ValueError('Headroom credit requires the actual leaf service, not a shared ancestor')
    maximum_text = (node / 'memory.max').read_text().strip()
    current_text = (node / 'memory.current').read_text().strip()
    if not maximum_text.isdecimal() or not current_text.isdecimal():
        raise ValueError('Owned service memory readings must be finite nonnegative integers')
    maximum, current = int(maximum_text), int(current_text)
    if not 0 <= current <= maximum <= 768 * 2**20 or maximum == 0:
        raise ValueError('Owned memory current/max exceeds the frozen resource envelope')
    pids = [int(value) for value in (node / 'cgroup.procs').read_text().split()]
    if os.getpid() not in pids:
        raise ValueError('Current process is not in the credited service boundary')
    boot = boot_identity()
    return {'path': str(node), 'inode': node.stat().st_ino, 'boot_id': boot,
            'pid': os.getpid(), 'memory_current_bytes': current,
            'memory_max_bytes': maximum}


def _headroom_host_guard():
    """Preserve process/RSS blocks, credit only observed owned current memory."""
    before = _owned_memory_snapshot()
    # HostGuard still performs its complete live-process and RSS inspection.
    # Its minimum is the irreducible host reserve; the remaining owned capacity
    # is checked only after the bracketing kernel observations are available.
    host = HostGuard(minimum_available_mib=128)()
    after = _owned_memory_snapshot()
    if any(before[key] != after[key] for key in ('path', 'inode', 'boot_id', 'pid', 'memory_max_bytes')):
        raise ValueError('Owned service boundary changed during host headroom inspection')
    credit = min(before['memory_current_bytes'], after['memory_current_bytes'])
    required = max(128 * 2**20, 768 * 2**20 - credit + 128 * 2**20)
    result = {**host, 'host_headroom': {'policy': 'owned_current_remaining_capacity_plus_128MiB',
        'before': before, 'after': after, 'credited_current_bytes': credit,
        'required_available_bytes': required, 'reserve_bytes': 128 * 2**20}}
    if (host['allowed'] and host['available_mib'] * 2**20 < required):
        result.update(allowed=False, reasons=[*host['reasons'], 'insufficient_available_memory'])
    return result


def production_guard(plan):
    """Read-only preflight; missing/active old unit always blocks, never kills."""
    try:
        limits = inherited_limits()
        result = subprocess.run(['systemctl', '--user', 'show', plan.old_unit,
            '--property=LoadState', '--property=ActiveState', '--property=ExecStart'],
            capture_output=True, text=True, timeout=5, check=False)
        fields = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        if (result.returncode or fields.get('LoadState') != 'loaded'
                or fields.get('ActiveState') not in {'inactive', 'failed'}):
            raise ValueError('Legacy unit active, missing or unverified')
        actual = fields.get('ExecStart', '').strip()
        if not actual:
            if plan.old_service_receipt is None:
                raise ValueError('Empty inactive ExecStart needs a prior frozen operator inspection receipt')
            path = project_path(plan.old_service_receipt)
            if file_hash(path) != plan.old_service_receipt_sha256:
                raise ValueError('Legacy operator receipt bytes differ from the frozen hash')
            receipt = json.loads(path.read_text('utf-8'))
            fields = {key: receipt.get(key) for key in ('unit', 'scope', 'exec_start')}
            if (set(receipt) != {'unit', 'scope', 'exec_start', 'sha256'}
                    or fields != {'unit': plan.old_unit, 'scope': 'user', 'exec_start': plan.old_exec_start}
                    or receipt['sha256'] != fingerprint(fields)):
                raise ValueError('Legacy operator receipt scope, unit, executable or hash differs')
            actual = fields['exec_start']
        if plan.old_exec_start not in actual:
            raise ValueError('Legacy unit executable differs from frozen identity')
        host = _headroom_host_guard()
        if not host['allowed']:
            return host
        return {**host, 'hard_limits_verified': True, 'limits': limits, 'legacy': fields}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {'allowed': False, 'reason': 'campaign_preflight_failed', 'error_type': type(exc).__name__}


class CampaignRunner(SubprocessRunner):
    @staticmethod
    def command(directory, launch):
        return [sys.executable, '-B', '-m', 'scripts.v7_long_running', 'worker',
                '--directory', str(directory), '--token', launch['token']]

    def start(self, directory, launch):
        inherited_limits()
        command = self.command(directory, launch)
        temporary = Path(directory) / 'tmp'
        temporary.mkdir(exist_ok=True)
        environment = restricted_environment()
        environment.update(TMP=str(temporary), TEMP=str(temporary), TMPDIR=str(temporary),
                           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with (Path(directory) / f'worker-{launch["token"]}.log').open('ab') as log:
            child = subprocess.Popen(command, cwd=project_path('.'), env=environment, shell=False,
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, **isolated_process_kwargs())
        self.children[child.pid] = child
        return {'pid': child.pid, 'created': psutil.Process(child.pid).create_time(), 'command': command}


class LongController:
    def __init__(self, directory, plan, *, clock=now, runner=None, guard=None, sleeper=time.sleep,
                 research_runner=None):
        # Only trusted Python entry code supplies the fixed guarded producer.
        # No CLI command/factory or caller-selected argv grants stop authority.
        if research_runner is not None and not isinstance(research_runner, SubprocessRunner):
            raise TypeError('An actual fixed SubprocessRunner implementation is required')
        self.research_runner = research_runner if research_runner is not None else SubprocessRunner()
        self.directory = project_path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.owner = FileLock(str(self.directory / '.owner.lock'), timeout=0)
        self.owner.acquire()
        try:
            self.store = CampaignStore(directory, plan, clock=clock)
        except BaseException:
            self.owner.release()
            raise
        self.clock, self.runner = clock, runner or CampaignRunner()
        self.guard, self.sleeper = guard or (lambda: production_guard(plan)), sleeper

    def close(self):
        self.store.close()
        self.owner.release()

    def _owned(self, launch):
        identity = launch.get('process')
        return identity if identity is not None else self.runner.find(self.directory, launch)

    def _stop(self, reason):
        terminal = self.store.terminal(reason)
        state = self.store.load()
        if state['launches']:
            identity = self._owned(state['launches'][-1])
            if identity and self.runner.status(identity) == 'running':
                self.runner.stop(identity)
                for _ in range(20):
                    if self.runner.status(identity) != 'running':
                        break
                    self.sleeper(.1)
                if self.runner.status(identity) == 'running':
                    raise RuntimeError('Owned campaign worker termination unverified')
        self._stop_registered_research()
        return {'status': 'terminal', **terminal}

    def _stop_registered_research(self):
        # A worker killed externally may leave its registered research child.
        # Validate its exact manifest command and PID birth time before signalling.
        path = project_path(self.store.plan.forward_directory) / 'autonomous/scheduler.sqlite'
        if not path.exists():
            return
        runner = self.research_runner
        db = sqlite3.connect(f'{path.as_uri()}?mode=ro', uri=True)
        try:
            for month, text, digest in db.execute('SELECT month,value,hash FROM jobs'):
                state = json.loads(text)
                if fingerprint(state) != digest or state['month'] != month:
                    raise ValueError('Research ownership journal is corrupt; cannot guess processes')
                if not state['attempts']:
                    continue
                job_path = project_path(state['job_path'])
                expected = path.parent / 'jobs' / month / 'job.json'
                if job_path != expected:
                    raise ValueError('Research ownership path differs from campaign runtime')
                attempt = state['attempts'][-1]
                identity = attempt.get('process') or runner.find(job_path, attempt)
                if identity:
                    if identity['command'] != runner.command(job_path, attempt):
                        raise ValueError('Research ownership command differs from reservation')
                    runner.stop(identity)
        finally:
            db.close()

    def step(self):
        state = self.store.load()
        if source_files() != self.store.identity['core'] or file_hash(Path(__file__)) != self.store.identity['adapter_sha256']:
            return self._stop('source_changed')
        if (self.directory / 'STOP').exists():
            return self._stop('operator_stop')
        if state['terminal']:
            return self._stop(state['terminal']['reason'])
        if stamp(self.clock()) >= stamp(self.store.plan.end_at):
            return self._stop('calendar_deadline')
        observation = self.guard()
        if observation.get('allowed') is not True:
            if not state['launches'] and not state['sessions']:
                return {'status': 'blocked', 'reason': 'host_guard', 'guard': observation}
            return {**self._stop('host_guard'), 'guard': observation}
        if state['launches']:
            launch = state['launches'][-1]
            active = state['active_step']
            if active is not None and active_step_expired(state, launch['token'], {}, clock=self.clock):
                return self._stop('step_work_deadline_consumed')
            identity = self._owned(launch)
            status = self.runner.status(identity) if identity else 'exited'
            if status == 'identity_mismatch':
                return self._stop('process_identity_lost')
            if status == 'running':
                return {'status': 'running', 'process': identity}
            if (stamp(self.clock()) - stamp(launch['reserved_at'])).total_seconds() < 15:
                return {'status': 'awaiting_reserved_spawn'}
        if (len(state['launches']) >= self.store.plan.max_sessions
                or len(state['sessions']) >= self.store.plan.max_sessions
                or sum(s['wall_seconds'] for s in state['sessions']) + self.store.plan.session_seconds > self.store.plan.max_reserved_seconds
                or sum(s['max_steps'] for s in state['sessions']) + self.store.plan.session_steps > self.store.plan.max_reserved_steps):
            return self._stop('campaign_allowance_consumed')
        launch = {'token': uuid4().hex, 'reserved_at': stamp(self.clock()).isoformat(), 'process': None}
        self.store.update('worker_launch_reserved', lambda s: s['launches'].append(launch))
        identity = self.runner.start(self.directory, launch)
        self.store.update('worker_identity_recorded', lambda s: s['launches'][-1].update(process=identity))
        return {'status': 'spawned', 'process': identity}

    def run(self):
        try:
            while True:
                result = self.step()
                if result['status'] == 'terminal':
                    return result
                remaining = (stamp(self.store.plan.end_at) - stamp(self.clock())).total_seconds()
                self.sleeper(max(0., min(10., remaining)))
        finally:
            self._stop('controller_exit')


def read_plan(directory):
    manifest = json.loads((project_path(directory) / 'campaign.json').read_text('utf-8'))
    if fingerprint(manifest['identity']) != manifest['sha256']:
        raise ValueError('Campaign manifest integrity failure')
    fields = manifest['identity']['plan']
    fields['symbols'] = tuple(fields['symbols'])
    return CampaignPlan(**fields)


def inspect_campaign(directory):
    """Existing campaign only: no mkdir, lockfile, pragma, identity or DB writes."""
    directory = project_path(directory)
    manifest_path, db_path = directory / 'campaign.json', directory / 'campaign.sqlite'
    if not manifest_path.is_file() or not db_path.is_file():
        raise ValueError('Status requires an existing complete campaign; it never creates one')
    manifest = json.loads(manifest_path.read_text('utf-8'))
    identity = manifest['identity']
    if (fingerprint(identity) != manifest['sha256'] or identity['core'] != source_files()
            or identity['adapter_sha256'] != file_hash(Path(__file__))):
        raise ValueError('Campaign identity/source integrity failure')
    store = CampaignStore.__new__(CampaignStore)
    store._verified_head = None
    store.last_verification_rows = 0
    wal = db_path.with_name(db_path.name + '-wal')
    # A closed/checkpointed DB can be inspected without creating WAL/SHM files.
    # An active WAL uses SQLite's existing shared-memory read coordination only.
    mode = 'mode=ro' if wal.exists() and wal.stat().st_size else 'mode=ro&immutable=1'
    store.db = sqlite3.connect(f'{db_path.as_uri()}?{mode}', uri=True)
    try:
        return store.load()
    finally:
        store.close()


def active_step_expired(state, token, cache, *, clock=now, monotonic=None):
    """Idle/rollover is not a numerical step; no old deadline kills a new slot."""
    active = state['active_step']
    if active is None or active['launch_token'] != token:
        return False
    observed = stamp(clock())
    measured = (monotonic or time.monotonic)()
    began, deadline = active['monotonic_started'], active['monotonic_deadline']
    if (not all(type(value) in (int, float) and math.isfinite(value) for value in (began, deadline, measured))
            or deadline <= began):
        raise ValueError('Invalid original monotonic work lease')
    cache.update(id=active['id'], deadline=deadline)  # Original reservation, never a new wall-derived lease.
    return (boot_identity() != active['boot_identity'] or measured < began or measured >= deadline
            or observed < stamp(active['began_at']) or observed >= stamp(active['deadline']))


def unit_template(directory, *, clock=now):
    """Emit only, for an already frozen campaign. No systemd mutation."""
    directory = project_path(directory)
    inspect_campaign(directory)
    plan = read_plan(directory)
    seconds = int((stamp(plan.end_at) - stamp(clock())).total_seconds())
    if not 1 <= seconds <= 366 * 86400:
        raise ValueError('Frozen campaign has no valid remaining runtime')
    root = project_path('.')

    def quote(value):
        value = value.as_posix() if isinstance(value, Path) else str(value)
        if not value.isprintable():
            raise ValueError('Invalid systemd argument')
        return '"' + value.replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%') + '"'

    def working_directory(value):
        value = value.as_posix()
        if not value or not value.isprintable() or any(c in value for c in '\\\"\''):
            raise ValueError('Invalid systemd working directory')
        # Path setting, not argv: $ stays literal. The workspace also appears
        # in executable names, where systemd rejects backslashes and quotes.
        # Protect trailing whitespace from config trimming without changing cwd.
        if value[-1].isspace():
            value += '/.'
        return value.replace('%', '%%')

    entry = root / 'scripts/v7_guard/campaign_guard_entry.py'
    guards = ('lease_enforcer.py', 'multiwork_enforcer.py', 'tree_work_enforcer.py', 'guarded_runtime.py', 'campaign_guard_entry.py')
    if any(not (entry.parent / name).is_file() for name in guards):
        raise ValueError('Complete fixed guardian sources required')
    guard = directory / 'whole_tree_guard'
    if guard.exists() or inspect_campaign(directory).get('whole_tree_guard') is not None:
        raise ValueError('Original guard attempt is consumed; no restart template')
    command = [root / '.venv/bin/python', '-B', entry, 'guard', directory, guard]
    postmortem = [root / '.venv/bin/python', '-B', entry, 'postmortem', directory, guard]
    return '\n'.join(['[Unit]', 'Description=Frozen finite V7 campaign (no trading)', '', '[Service]',
        'Type=exec', 'WorkingDirectory=' + working_directory(root),
        'ExecStart=:' + ' '.join(map(quote, command)),  # ':' disables argv environment expansion.
        'ExecStopPost=:' + ' '.join(map(quote, postmortem)),
        'CPUAccounting=yes', 'MemoryAccounting=yes', 'TasksAccounting=yes', 'CPUQuota=25%',
        'MemoryHigh=640M', 'MemoryMax=768M', 'MemorySwapMax=0', 'TasksMax=64',
        f'RuntimeMaxSec={seconds}', 'Restart=no', 'KillMode=control-group', 'TimeoutStopSec=20',
        'NoNewPrivileges=yes', 'UMask=0077', 'StandardOutput=journal', 'StandardError=journal',
        'Environment=OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1',
        '', '# Output only: requires explicit operator installation and observed resource acceptance.', ''])


def campaign_step(store, autonomous, session_id):
    """One hard-deadline work lease; transient failures retain every charge."""
    active = store.begin_step(session_id)
    try:
        try:
            result = autonomous.step()
        except TransientFeedError as exc:
            state = store.record_transport_outcome(exc.code)
            return {'forward': {'status': 'transient_transport_allowance_consumed' if state['exhausted']
                                else 'transient_transport_failure', 'transport': state},
                    'research': {'status': 'existing_worker_preserved'}}
        except Exception as exc:
            # Preserve the exception, but prevent a new campaign worker from
            # turning auth/schema/integrity/resource errors into fresh retries.
            store.terminal(f'nontransient_step_failure:{type(exc).__name__}')
            raise
        if result['forward']['status'] == 'stepped':
            store.record_transport_outcome()
        return result
    finally:
        store.end_step(active['id'])


def run_worker(directory, token, *, clock=now, runtime_factory=ForwardRuntime,
               scheduler_factory=None, guard=None, sleeper=time.sleep,
               enforce_watchdog=True):
    directory = project_path(directory)
    plan = read_plan(directory)
    guard = guard or (lambda: production_guard(plan))
    with FileLock(str(directory / '.worker.lock'), timeout=0):
        store = CampaignStore(directory, plan, clock=clock)
        runtime = scheduler = service = None
        finished = threading.Event()
        if enforce_watchdog:
            parent = psutil.Process().parent()
            parent_created = parent.create_time() if parent is not None else None
            deadline = time.monotonic() + max(0., (stamp(plan.end_at) - stamp(clock())).total_seconds())

            def watch():
                step_cache = {}
                monitor = sqlite3.connect(f'{(directory / "campaign.sqlite").as_uri()}?mode=ro', uri=True)
                while not finished.wait(1.):
                    try:
                        alive = parent is not None and parent.is_running() and parent.create_time() == parent_created
                        expired = time.monotonic() >= deadline or stamp(clock()) >= stamp(plan.end_at)
                        stop = (directory / 'STOP').exists()
                        row = monitor.execute('SELECT value,hash FROM state WHERE id=1').fetchone()
                        state = json.loads(row[0])
                        if fingerprint(state) != row[1]:
                            raise ValueError('Watchdog campaign state integrity failure')
                        expired = expired or active_step_expired(state, token, step_cache, clock=clock)
                        stop = stop or state['terminal'] is not None
                    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error, psutil.Error):
                        alive, expired, stop = False, False, False
                    if not alive or expired or stop:
                        try:
                            for child in reversed(psutil.Process().children(recursive=True)):
                                try:
                                    child.kill()
                                except psutil.NoSuchProcess:
                                    pass
                        finally:
                            os._exit(124)
                monitor.close()
            threading.Thread(target=watch, daemon=True).start()
        try:
            launches = store.load()['launches']
            if not launches or launches[-1]['token'] != token or store.load()['terminal']:
                raise ValueError('Worker has no current durable launch reservation')
            if guard().get('allowed') is not True:
                store.terminal('host_guard')
                return
            authority, schedule = actual_work_contract(plan)
            reserved = store.reserve_session(token)
            if reserved is None:
                store.terminal('campaign_allowance_consumed')
                return
            lifecycle = dict(plan.lifecycle)
            lifecycle['symbols'] = tuple(lifecycle.get('symbols', plan.symbols))
            runtime = runtime_factory(plan.forward_directory, plan.feed_directory,
                **campaign_runtime_kwargs(plan, clock))
            from factorlab.discovery.neural_forward_ticket import _bind_authority
            _bind_authority(runtime, authority)
            scheduler = (CampaignScheduler(runtime, store, config=research_config(plan.research, plan.symbols),
                                           authority=authority, schedule=schedule)
                         if scheduler_factory is None else
                         scheduler_factory(runtime, config=research_config(plan.research, plan.symbols), authority=authority, schedule=schedule))
            while not store.load()['terminal']:
                if (directory / 'STOP').exists() or stamp(clock()) >= stamp(plan.end_at):
                    store.terminal('operator_stop_or_calendar_deadline')
                    break
                if (stamp(plan.end_at) - stamp(clock())).total_seconds() < plan.effective_step_seconds + 2:
                    store.terminal('calendar_work_window_closed')
                    break
                if reserved is None:
                    store.terminal('campaign_allowance_consumed')
                    break
                config = ServiceConfig(**{k: reserved[k] for k in ('session_id', 'wall_seconds', 'max_steps')},
                    tick_seconds=plan.tick_seconds, poll=True)
                service = CampaignService(runtime, store, config=config, guard=guard, work_authority=authority,
                    stop_requested=lambda: ((directory / 'STOP').exists() or bool(store.load()['terminal'])
                                            or stamp(clock()) >= stamp(plan.end_at)))
                autonomous = AutonomousService(service, scheduler)
                # Deliberately do not call AutonomousService.run(): its pilot
                # shutdown stops the research child at every session boundary.
                while (service.state['steps'] < config.max_steps
                       and service.state['spent_seconds'] <= config.wall_seconds - plan.effective_step_seconds - 2
                       and (stamp(clock()) - stamp(reserved['reserved_at'])).total_seconds()
                           <= config.wall_seconds - plan.effective_step_seconds - 2
                       and (stamp(plan.end_at) - stamp(clock())).total_seconds() >= plan.effective_step_seconds + 2):
                    try:
                        result = campaign_step(store, autonomous, config.session_id)
                    except SessionWindowClosed:
                        break  # Admission clock advanced since the loop check: normal rollover.
                    if result['forward']['status'] not in {'stepped', 'transient_transport_failure'}:
                        store.terminal(result['forward']['status'])
                        break
                    if service.state['steps'] >= config.max_steps:
                        break
                    remaining = min(config.wall_seconds - service.state['spent_seconds'],
                                    config.wall_seconds - (stamp(clock()) - stamp(reserved['reserved_at'])).total_seconds(),
                                    (stamp(plan.end_at) - stamp(clock())).total_seconds())
                    if remaining <= 2:
                        break
                    began = service.monotonic()
                    service.state['active_since'] = stamp(clock()).isoformat()
                    service._save('campaign_sleep_started')
                    try:
                        sleeper(min(config.tick_seconds, remaining - 2))
                    finally:
                        service.state['spent_seconds'] += max(0., service.monotonic() - began)
                        service.state['active_since'] = None
                        service._save('campaign_sleep_finished')
                service.close(close_runtime=False)
                service = None
                reserved = store.reserve_session(token)
        finally:
            try:
                if scheduler:
                    scheduler.stop('supervisor_stopped')
            finally:
                if service:
                    service.close(close_runtime=False)
                if scheduler:
                    scheduler.close()
                if runtime:
                    runtime.close()
                store.close()
                finished.set()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['init', 'run', 'status', 'unit-template', 'worker'])
    parser.add_argument('--directory', required=True)
    parser.add_argument('--plan')
    parser.add_argument('--token')
    args = parser.parse_args(argv)
    if args.operation == 'worker':
        if not args.token:
            raise ValueError('Worker token required')
        run_worker(args.directory, args.token)
        return
    if args.operation == 'status':
        if args.plan is not None:
            raise ValueError('Read-only status takes no new plan')
        print(json.dumps(inspect_campaign(args.directory), indent=2))
        return
    if args.operation == 'unit-template':
        if args.plan is not None:
            raise ValueError('Unit template requires an already frozen campaign, not a new plan')
        print(unit_template(args.directory), end='')
        return
    if args.plan:
        fields = json.loads(project_path(args.plan).read_text('utf-8'))
        fields['symbols'] = tuple(fields.get('symbols', ('BTCUSDT',)))
        plan = CampaignPlan(**fields)
    else:
        plan = read_plan(args.directory)
    controller = LongController(args.directory, plan)
    try:
        result = ({'status': 'initialized_not_running', 'identity_hash': fingerprint(controller.store.identity)}
                  if args.operation == 'init' else controller.run())
        print(json.dumps(result, indent=2))
    finally:
        controller.close()


if __name__ == '__main__':
    main()
