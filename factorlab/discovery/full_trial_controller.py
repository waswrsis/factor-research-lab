"""Original full-pipeline execution driver; no new wallets or synthetic results.

Physical and original journal-bound observation call the actual AutonomousService.
Logical execution requires the source-bound worker, receipt and activation bridge;
an absent bridge is an explicit pending dependency. This module never accepts a
backend, command, loss array, successful-outcome flag or public clock.
"""
from __future__ import annotations

import json
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from filelock import FileLock

from factorlab.protocol import fingerprint
from factorlab.discovery.calibration_runtime_contract import _original_operations
from factorlab.discovery.calibration_trial_context import _no_redirect, verify_memory_namespace
from factorlab.discovery.evidence import file_hash
from factorlab.discovery.full_calibration_registry import (
    FullTrialRegistry, QuarantinedTrialPermit, _read,
)
from factorlab.discovery.monthly_work_authority import source_files
from factorlab.discovery.prompt_builder import DEFAULT_MODEL, DEFAULT_REASONING
from factorlab.discovery.v7_storage import immutable_record

VERSION = 'v7-original-full-trial-controller-v2'
BOUND = 'full_trial_controller_bound'


@dataclass(frozen=True)
class ControllerStepLease:
    """Physical supervisor deadline; never a spendable model work lease.

    Every child still needs its separate original <=300s prepaid WorkLease.
    Supervisor time comes exclusively from the frozen cumulative service budget.
    """
    issuer: dict
    service_seconds: float

    @classmethod
    def issue(cls, schedule):
        from .work_lease import FrozenWorkLease
        return cls(FrozenWorkLease.issue(300).to_dict(), schedule.service['wall_seconds'])

    @classmethod
    def from_dict(cls, value, schedule):
        from .work_lease import FrozenWorkLease
        if set(value) != {'issuer', 'service_seconds'} or value['service_seconds'] != schedule.service['wall_seconds']:
            raise ValueError('Controller deadline changed its original service budget')
        FrozenWorkLease.from_dict(value['issuer'])
        return cls(**value)

    def to_dict(self):
        return asdict(self)

    def supervisor(self):
        from datetime import timedelta
        from types import SimpleNamespace
        from .work_lease import FrozenWorkLease, _utc
        original = FrozenWorkLease.from_dict(self.issuer)
        fields = original.to_dict()
        fields['wall_deadline'] = (_utc(original.wall_started_at)+timedelta(seconds=self.service_seconds)).isoformat()
        fields['monotonic_deadline'] = original.monotonic_started+self.service_seconds
        return SimpleNamespace(**fields)

    def observe(self):
        from .work_lease import _observe
        return _observe(self.supervisor())

    def evaluate(self, observation, previous=None):
        from .work_lease import _evaluate, LeaseCheck
        check = _evaluate(self.supervisor(), observation, previous)
        if check.allowed and observation['pid'] != self.issuer['parent_pid']:
            return LeaseCheck('wrong_process')
        return check

    def assert_parent_alive(self):
        from .work_lease import LeaseClosed
        check = self.evaluate(self.observe())
        if not check.allowed:
            raise LeaseClosed(check)
        return check

    def elapsed(self, observation):
        from .work_lease import _utc
        if observation.get('unavailable'):
            return self.service_seconds
        return max(0., observation['wall'] - _utc(self.issuer['wall_started_at']).timestamp(),
                   observation['monotonic'] - self.issuer['monotonic_started'])


def step_completed(row):
    """Credit only an original result observed by a separate successor call.

    Callers must first reconstruct the original files and LC journal via verify.
    A witness proves the predecessor's IO ended; it makes no claim about its own
    later commit/return. That tail is charged to the same absolute service clock.
    """
    from .work_lease import FrozenWorkLease
    attempt, result, witness = row['attempt'], row.get('result'), row.get('witness')
    if result is None or witness is None:
        return False
    FrozenWorkLease.from_dict(attempt['lease']['issuer'])
    lease = ControllerStepLease(**attempt['lease'])
    return (result['state'] == 'awaiting_successor_witness' and witness['state'] == 'confirmed'
            and witness['attempt_hash'] == fingerprint(attempt)
            and witness['result_hash'] == fingerprint(result)
            and lease.evaluate(witness['observed'], result['physical_finish']).allowed)


def _plain(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


@dataclass(frozen=True)
class PendingDependency:
    code: str
    required: str


def _provider_policy(permit):
    expected = {'backend': 'codex', 'model_requested': DEFAULT_MODEL,
                'reasoning_requested': DEFAULT_REASONING,
                'context_policy': 'fresh_codex_per_trial_v1'}
    if permit.registry.plan.provider_policy != expected:
        raise ValueError('Full trial provider policy differs from the actual restricted worker')
    return expected


def _permit(permit):
    if type(permit) is not QuarantinedTrialPermit:
        raise TypeError('Original registered trial permit required')
    permit.verify()
    _provider_policy(permit)
    from .nominal_algorithm_execution import require_registered_mode
    require_registered_mode(permit)
    return permit


@dataclass(frozen=True)
class FullControllerSchedule:
    """Every processing opportunity is fixed before the first work reservation.

    UTC dates are observation opportunities, never wallet months. Existing work
    kernels alone choose and debit the actual UTC authority month. A step lease
    is a wall-time cap, not a second budget or a renewal of child work leases.
    """
    due_at: tuple[str, ...]
    service: dict
    autonomous: dict
    search_schedule: dict
    observation_mode: str = 'physical_UTC'

    def __post_init__(self):
        from factorlab.discovery.v7_service import ServiceConfig
        from factorlab.discovery.v7_autonomous import AutonomousConfig
        from factorlab.discovery.v7_run import V7Config
        from factorlab.discovery.scheduled_search import FrozenSWorkSchedule
        if (type(self.due_at) is not tuple or not 1 <= len(self.due_at) <= 7200
                or self.observation_mode not in {'physical_UTC', 'bound_logical_observation'}):
            raise ValueError('Bounded immutable observation schedule required')
        dates = [pd.Timestamp(x) for x in self.due_at]
        if any(x.tzinfo is None or x.isoformat() != s for x, s in zip(dates, self.due_at)):
            raise ValueError('Canonical aware observation timestamps required')
        if any(x.utcoffset().total_seconds() != 0 for x in dates) or dates != sorted(set(dates)):
            raise ValueError('Strictly increasing UTC observation opportunities required')
        service = ServiceConfig(**self.service)
        if (service.poll or service.max_fits_per_step != 1 or service.max_fit_attempts != 1
                or service.max_steps != len(dates)):
            raise ValueError('Full trials require local receipts, one fit and no free retries')
        fields = _plain(self.autonomous)
        fields['research']['symbols'] = tuple(fields['research']['symbols'])
        fields['research'] = V7Config(**fields['research'])
        autonomous = AutonomousConfig(**fields)
        schedule = FrozenSWorkSchedule.from_dict(self.search_schedule)
        if (autonomous.backend != 'codex' or autonomous.max_attempts != 1
                or schedule.to_dict()['config'] != _plain(asdict(autonomous.research))):
            raise ValueError('Original Codex search schedule and once-only worker required')
        for name in ('service', 'autonomous', 'search_schedule'):
            object.__setattr__(self, name, _plain(getattr(self, name)))

    def to_dict(self):
        return _plain(asdict(self))

    @classmethod
    def from_dict(cls, value):
        value = _plain(value)
        value['due_at'] = tuple(value['due_at'])
        return cls(**value)

    @property
    def digest(self):
        return fingerprint(self.to_dict())


class FullTrialController:
    """Freeze first; open only existing original stores; execute fixed code only."""
    step_completed = staticmethod(step_completed)

    @classmethod
    def freeze(cls, permit, runtime, authority, schedule):
        from factorlab.discovery.v7_forward import ForwardRuntime
        from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
        from factorlab.discovery.scheduled_search import FrozenSWorkSchedule, validate_schedule_authority
        permit = _permit(permit)
        if (type(runtime) is not ForwardRuntime or type(authority) is not MonthlyWorkAuthority
                or type(schedule) is not FullControllerSchedule):
            raise TypeError('Actual runtime, original authority and typed full schedule required')
        permit.verify(runtime.lifecycle)
        if authority.directory != permit.paths['authority']:
            raise ValueError('Controller cannot borrow another wallet')
        validate_schedule_authority(authority, FrozenSWorkSchedule.from_dict(schedule.search_schedule))
        protocol = runtime.claim_protocol()
        if protocol is None:
            raise ValueError('Original two six-month endpoints must be frozen before the controller')
        value = {'version': VERSION, 'trial': permit.trial, 'schedule': schedule.to_dict(),
                 'schedule_digest': schedule.digest, 'source_files': source_files(),
                 'implementation_hash': file_hash(__file__),
                 'authority_manifest_hash': authority.manifest_hash,
                 'runtime_identity_hash': fingerprint(runtime.identity),
                 'claim_protocol_digest': protocol.digest,
                 'provider_policy': _provider_policy(permit),
                 'work_clock': 'physical_UTC_original_wallet_and_leases',
                 'service_clock': 'continuous_original_dual_clock_including_idle_no_restart',
                 'completion_policy': 'successor_observes_original_pending_result_no_self_completion',
                 'formal_gate_eligible': False}
        directory = permit.paths['root'] / 'controller'
        directory.mkdir(parents=True, exist_ok=True)
        with FileLock(str(directory / '.lock'), timeout=0):
            existing = [o for o in _original_operations(runtime.lifecycle) if o['kind'] == BOUND]
            if existing:
                result = cls.inspect(permit)
                if result.identity != value:
                    raise ValueError('Original controller schedule cannot be replaced')
                return result
            if ((directory/'identity.json').exists() or authority.verify()['reservations']
                    or runtime.lifecycle.db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]
                    or runtime.lifecycle.forecasts.db.execute('SELECT COUNT(*) FROM forecasts').fetchone()[0]
                    or (runtime.directory/'autonomous').exists() or (runtime.directory/'service').exists()):
                raise ValueError('Controller cannot enroll retrospectively or recover ambiguous freeze')
            mirror = permit.registry.directory/'controller_plans'/(permit.trial['trial_id']+'.json')
            immutable_record(directory/'identity.json', value)
            immutable_record(mirror, value)
            with runtime.lifecycle.db:
                runtime.lifecycle._event(BOUND, {'identity_hash': fingerprint(value)})
        return cls.inspect(permit)

    @classmethod
    def inspect(cls, permit):
        obj = cls.__new__(cls)
        obj.permit = _permit(permit)
        obj.directory = permit.paths['root']/'controller'
        obj.identity = _read(_no_redirect(obj.directory/'identity.json', permit.registry.directory))
        obj.schedule = FullControllerSchedule.from_dict(obj.identity['schedule'])
        obj.verify()
        return obj

    def verify(self):
        self.permit.verify()
        from .nominal_algorithm_execution import require_registered_mode
        require_registered_mode(self.permit)
        value = self.identity
        mirror = self.permit.registry.directory/'controller_plans'/(self.permit.trial['trial_id']+'.json')
        if (_read(self.directory/'identity.json') != value or _read(mirror) != value
                or value['version'] != VERSION or value['trial'] != self.permit.trial
                or value['schedule_digest'] != self.schedule.digest
                or value['source_files'] != source_files() or value['implementation_hash'] != file_hash(__file__)
                or value['provider_policy'] != _provider_policy(self.permit)
                or value.get('service_clock') != 'continuous_original_dual_clock_including_idle_no_restart'
                or value.get('completion_policy') != 'successor_observes_original_pending_result_no_self_completion'
                or value['formal_gate_eligible'] is not False):
            raise ValueError('Original full controller/source/provider schedule changed')
        with self.permit.open_lifecycle() as lc:
            bound = [o['detail'] for o in _original_operations(lc) if o['kind'] == BOUND]
            if bound != [{'identity_hash': fingerprint(value)}]:
                raise ValueError('Controller has no original lifecycle binding')
            from factorlab.discovery.claim_protocol import ClaimProtocol
            if ClaimProtocol(lc).digest != value['claim_protocol_digest']:
                raise ValueError('Original five-claim protocol changed')
        return self._steps()

    def _paths(self, index, suffix):
        filename = f'{index:04d}-{suffix}.json'
        return (self.directory/'steps'/filename,
                self.permit.registry.directory/'controller_steps'/self.permit.trial['trial_id']/filename)

    def _write_pair(self, index, suffix, value):
        for path in self._paths(index, suffix):
            immutable_record(_no_redirect(path, self.permit.registry.directory), value)
        # Bind both immutable files to the original lifecycle journal. A second
        # locally rehashed pair cannot rewrite the actual dispatch history.
        import sqlite3
        from contextlib import closing
        from factorlab.discovery.v7_lifecycle import canonical
        database = self.permit.paths['runtime']/'lifecycle.sqlite'
        if not database.is_file():
            raise ValueError('Original controller lifecycle journal missing')
        detail = {'controller_hash': fingerprint(self.identity), 'index': index,
                  'phase': suffix, 'content_hash': fingerprint(value)}
        with closing(sqlite3.connect(database)) as db, db:
            db.execute('PRAGMA synchronous=FULL')
            db.execute('BEGIN IMMEDIATE')
            previous, sequence, found = '0'*64, 0, []
            for seq, text, digest in db.execute('SELECT * FROM operations ORDER BY seq'):
                operation = json.loads(text)
                if (seq != sequence+1 or operation['seq'] != seq or operation['previous'] != previous
                        or fingerprint(operation) != digest):
                    raise ValueError('Original controller lifecycle chain changed')
                previous, sequence = digest, seq
                if operation['kind'] == 'full_trial_controller_step' and operation['detail']['index'] == index and operation['detail']['phase'] == suffix:
                    found.append(operation['detail'])
            if found:
                if found != [detail]:
                    raise ValueError('Original controller step journal binding changed')
                return
            record = {'seq': sequence+1, 'previous': previous, 'kind': 'full_trial_controller_step',
                      'at': pd.Timestamp.now(tz='UTC').isoformat(), 'detail': detail}
            db.execute('INSERT INTO operations VALUES (?,?,?)',
                       (sequence+1, canonical(record), fingerprint(record)))

    def _execution_clock(self):
        paths = self._paths(-1, 'clock')
        if not any(p.exists() for p in paths):
            return None
        value = _read(paths[0])
        if (value != _read(paths[1]) or value['controller_hash'] != fingerprint(self.identity)
                or value['version'] != VERSION):
            raise ValueError('Original controller execution clock changed')
        ControllerStepLease.from_dict(value['lease'], self.schedule)
        return value

    def _start_execution(self, initial=None):
        """Bind a pre-IO sample only for a genuinely new original execution.

        Clock loss cannot renew a session: the old journal anchor rejects the
        replacement pair before any execution is permitted. A half-pair fails.
        The original clock is not a model lease or another wallet allowance.
        """
        # Public mutating entrypoints sample before even lock/clock-path IO.
        # This candidate is never a replacement for an existing original. A
        # candidate carries no authority and is not a spendable child lease.
        from .nominal_algorithm_execution import require_execution_scope
        require_execution_scope(self.permit)
        initial = ControllerStepLease.issue(self.schedule) if initial is None else initial
        value = self._execution_clock()
        if value is None:
            from types import SimpleNamespace
            prior = _original_operations(SimpleNamespace(directory=self.permit.paths['runtime']))
            if any(o['kind'] == 'full_trial_controller_step' for o in prior):
                raise ValueError('Original controller clock anchor exists without its files; never renew')
            value = {'version': VERSION, 'controller_hash': fingerprint(self.identity),
                     'lease': initial.to_dict()}
            self._write_pair(-1, 'clock', value)
        return ControllerStepLease.from_dict(value['lease'], self.schedule)

    def _terminal(self):
        paths = self._paths(-2, 'terminal')
        return _read(paths[0]) if paths[0].exists() else None

    def _close_execution(self, lease, reason):
        existing = self._terminal()
        if existing is not None:
            return existing
        result = {'state': 'controller_service_closed', 'reason': reason,
                  'controller_hash': fingerprint(self.identity),
                  'clock_hash': fingerprint(self._execution_clock()),
                  'observed': lease.observe(), 'formal_gate_eligible': False}
        self._write_pair(-2, 'terminal', result)
        return result

    def _steps(self):
        # Pure readback only. Paid production must still open and close its own
        # source/prefix audit; this context never spans dispatch or publication.
        if self.schedule.observation_mode != 'bound_logical_observation':
            return self._read_steps()
        from .quarantined_observation import _original_observation_read_scope
        with self.permit.open_lifecycle() as lifecycle:
            with _original_observation_read_scope(lifecycle):
                return self._read_steps()

    def _read_steps(self):
        rows = []
        previous = '0'*64
        roots = [self.directory/'steps', self.permit.registry.directory/'controller_steps'/self.permit.trial['trial_id']]
        files = [{p.name for p in root.glob('*.json')} for root in roots]
        if files[0] != files[1]:
            raise ValueError('Original controller step mirror missing; never replay work')
        expected = set()
        anchors = []
        clock = self._execution_clock()
        if clock is not None:
            expected.add(self._paths(-1, 'clock')[0].name)
            anchors.append({'controller_hash': fingerprint(self.identity), 'index': -1,
                            'phase': 'clock', 'content_hash': fingerprint(clock)})
        lease = ControllerStepLease.from_dict(clock['lease'], self.schedule) if clock else None
        prior_observation = None
        for index, due in enumerate(self.schedule.due_at):
            attempts = self._paths(index, 'attempt')
            results = self._paths(index, 'result')
            if not attempts[0].exists():
                break
            expected.add(attempts[0].name)
            attempt = _read(attempts[0])
            anchors.append({'controller_hash': fingerprint(self.identity), 'index': index,
                            'phase': 'attempt', 'content_hash': fingerprint(attempt)})
            if (attempt != _read(attempts[1]) or attempt['index'] != index
                    or attempt['due_at'] != due or attempt['previous'] != previous
                    or attempt['controller_hash'] != fingerprint(self.identity)):
                raise ValueError('Original controller step identity changed')
            if (clock is None or attempt['lease'] != clock['lease']
                    or attempt['clock_hash'] != fingerprint(clock)
                    or not lease.evaluate(attempt['physical_start'], prior_observation).allowed):
                raise ValueError('Controller opportunity renewed or lost its original physical clock')
            prefix = attempt.get('observation_prefix')
            if prefix is not None:
                from factorlab.discovery.quarantined_observation import QuarantinedObservation
                with self.permit.open_lifecycle() as lc:
                    observer = QuarantinedObservation.from_lifecycle(lc)
                    if observer is None:
                        raise ValueError('Original controller observation journal missing')
                    observer.verify_binding(prefix)
            observed_at = (pd.Timestamp(prefix['logical_at']) if prefix is not None else
                           pd.Timestamp(attempt['physical_start']['wall'], unit='s', tz='UTC'))
            if observed_at < pd.Timestamp(due):
                raise ValueError('Controller step preceded frozen observation opportunity')
            result = None
            if results[0].exists():
                expected.add(results[0].name)
                result = _read(results[0])
                anchors.append({'controller_hash': fingerprint(self.identity), 'index': index,
                                'phase': 'result', 'content_hash': fingerprint(result)})
                if (result != _read(results[1]) or result['attempt_hash'] != fingerprint(attempt)
                        or result['state'] not in {'awaiting_successor_witness', 'failed_consumed', 'interrupted_consumed'}):
                    raise ValueError('Original controller result changed')
                if (result['state'] == 'awaiting_successor_witness'
                        and not lease.evaluate(result['physical_finish'], attempt['physical_start']).allowed):
                    raise ValueError('Pending controller result has no original timely dispatch observation')
            witness = None
            witnesses = self._paths(index, 'witness')
            if witnesses[0].exists():
                expected.add(witnesses[0].name)
                witness = _read(witnesses[0])
                anchors.append({'controller_hash': fingerprint(self.identity), 'index': index,
                                'phase': 'witness', 'content_hash': fingerprint(witness)})
                if (result is None or result['state'] != 'awaiting_successor_witness'
                        or witness != _read(witnesses[1])
                        or witness['attempt_hash'] != fingerprint(attempt)
                        or witness['result_hash'] != fingerprint(result)
                        or witness['state'] not in {'confirmed', 'failed_or_unknown_consumed'}):
                    raise ValueError('Original controller successor witness changed')
                if witness['state'] == 'confirmed':
                    if (not lease.evaluate(witness['observed'], result['physical_finish']).allowed
                            or any(max(p.stat().st_mtime_ns, p.stat().st_ctime_ns) > witness['observed']['wall']*1e9
                                   for p in results)):
                        raise ValueError('Original controller predecessor was not timely observed after publication')
            row = {'attempt': attempt, 'result': result, 'witness': witness}
            previous = fingerprint(row)
            rows.append(row)
            if not step_completed(row):
                break
            prior_observation = witness['observed']
        terminal = self._terminal()
        if terminal is not None:
            paths = self._paths(-2, 'terminal')
            expected.add(paths[0].name)
            if (clock is None or terminal != _read(paths[1])
                    or terminal['controller_hash'] != fingerprint(self.identity)
                    or terminal['clock_hash'] != fingerprint(clock)
                    or terminal['state'] != 'controller_service_closed'):
                raise ValueError('Original terminal service boundary changed')
            anchors.append({'controller_hash': fingerprint(self.identity), 'index': -2,
                            'phase': 'terminal', 'content_hash': fingerprint(terminal)})
        if files[0] != expected:
            raise ValueError('Unregistered, truncated or out-of-order controller steps')
        from factorlab.discovery.neural_forward_ticket import _LifecycleReader
        reader = _LifecycleReader(self.permit.paths['runtime'])
        try:
            operations = [o for o in _original_operations(reader)
                          if o['kind'] == 'full_trial_controller_step']
            original = [o['detail'] for o in operations]
        finally:
            reader.close()
        if original != anchors:
            raise ValueError('Original controller step files differ from lifecycle journal anchors')
        for row in rows:
            if step_completed(row):
                event = next(o for o in operations if o['detail']['index'] == row['attempt']['index']
                             and o['detail']['phase'] == 'result')
                if pd.Timestamp(event['at']).timestamp() > row['witness']['observed']['wall']:
                    raise ValueError('Successor preceded original predecessor journal commit')
        return rows

    def pending_dependencies(self):
        missing = []
        if self.schedule.observation_mode != 'physical_UTC':
            try:
                from factorlab.discovery.quarantined_observation import current_binding
            except ImportError:
                current_binding = None
            with self.permit.open_lifecycle() as lc:
                prefix = current_binding(lc) if current_binding is not None else None
            if prefix is None:
                missing.append(PendingDependency('bound_observation_clock',
                    'Original observation clock must bind runtime, feed, worker and activation while UTC wallet/leases stay physical'))
        # Source integration is explicit. An ordinary Codex worker is not a
        # substitute for canonical trial context merely because its name matches.
        from factorlab.discovery import v7_autonomous
        if getattr(v7_autonomous, 'FULL_TRIAL_BACKEND_DISPATCH', None) != VERSION:
            missing.append(PendingDependency('registered_worker_backend_dispatch',
                'run_worker must select backend_for_job for the original quarantine permit before run_month'))
        return tuple(missing)

    def _confirm_predecessor(self, index, row, lease):
        """A separate locked invocation observes the already-published result.

        No source, receipt, queue, dispatch, or opportunity is created here.
        Confirmation's own return tail is not self-certified and cannot extend
        the first absolute clock or authorize a subsequent expired opportunity.
        """
        observed = lease.observe()
        allowed = lease.evaluate(observed, row['result']['physical_finish']).allowed
        if allowed:
            allowed = all(max(p.stat().st_mtime_ns, p.stat().st_ctime_ns) <= observed['wall']*1e9
                          for p in self._paths(index, 'result'))
        witness = {'attempt_hash': fingerprint(row['attempt']),
                   'result_hash': fingerprint(row['result']), 'observed': observed,
                   'state': 'confirmed' if allowed else 'failed_or_unknown_consumed'}
        self._write_pair(index, 'witness', witness)
        return {'state': 'completed_step' if allowed else 'failed_or_unknown_consumed',
                'index': index, 'witness': witness, 'output': row['result']['output'],
                'formal_gate_eligible': False}

    def step(self, index):
        """One original opportunity, initially uncredited pending successor IO.

        Reopening a result never confirms it implicitly. Use pump/run for the
        separate confirmation-only invocation; old failures remain consumed.
        """
        if type(index) is not int or not 0 <= index < len(self.schedule.due_at):
            raise ValueError('Step outside the original controller schedule')
        initial = ControllerStepLease.issue(self.schedule)
        with FileLock(str(self.directory/'.pump.lock'), timeout=0), FileLock(str(self.directory/'.lock'), timeout=0):
            lease = self._start_execution(initial)
            rows = self.verify()
            return self._step_locked(index, rows, lease)

    def _step_locked(self, index, rows, lease):
        if index < len(rows):
            row = rows[index]
            if row['result'] is not None:
                return row['result']
            result = {'attempt_hash': fingerprint(row['attempt']), 'state': 'interrupted_consumed',
                      'reason': 'original_dispatch_outcome_unknown_no_replay', 'formal_gate_eligible': False}
            self._write_pair(index, 'result', result)
            return result
        if index != len(rows):
            raise ValueError('Cannot skip original controller opportunities')
        if self._terminal() is not None:
            return self._terminal()
        if rows and not step_completed(rows[-1]):
            return {'state': 'predecessor_unconfirmed_or_consumed', 'formal_gate_eligible': False}
        observed = lease.observe()
        check = lease.evaluate(observed, rows[-1]['witness']['observed'] if rows else None)
        if not check.allowed:
            return self._close_execution(lease, check.reason)
        dependencies = self.pending_dependencies()
        if dependencies:
            return {'state': 'pending_dependencies', 'dependencies': [asdict(d) for d in dependencies],
                    'formal_gate_eligible': False, 'budget_granted': False}
        prefix = None
        observed_at = pd.Timestamp(observed['wall'], unit='s', tz='UTC')
        if self.schedule.observation_mode == 'bound_logical_observation':
            from factorlab.discovery.quarantined_observation import current_binding
            with self.permit.open_lifecycle() as lc:
                prefix = current_binding(lc)
            observed_at = pd.Timestamp(prefix['logical_at'])
        if observed_at < pd.Timestamp(self.schedule.due_at[index]):
            return {'state': 'pending_observation_time', 'formal_gate_eligible': False}
        # Resample AFTER dependencies/prefix reads, with the same original lease.
        observed = lease.observe()
        check = lease.evaluate(observed, rows[-1]['witness']['observed'] if rows else None)
        if not check.allowed:
            return self._close_execution(lease, check.reason)
        attempt = {'index': index, 'due_at': self.schedule.due_at[index],
                   'controller_hash': fingerprint(self.identity), 'lease': lease.to_dict(),
                   'clock_hash': fingerprint(self._execution_clock()), 'physical_start': observed,
                   'observation_prefix': prefix, 'previous': fingerprint(rows[-1]) if rows else '0'*64}
        self._write_pair(index, 'attempt', attempt)
        try:
            lease.assert_parent_alive()
            output = self._dispatch()
            finished = lease.observe()
            from .work_lease import LeaseClosed
            check = lease.evaluate(finished, observed)
            if not check.allowed:
                raise LeaseClosed(check)
            result = {'attempt_hash': fingerprint(attempt), 'state': 'awaiting_successor_witness',
                      'physical_finish': finished, 'output': _plain(output), 'formal_gate_eligible': False}
        except BaseException as exc:
            result = {'attempt_hash': fingerprint(attempt), 'state': 'failed_consumed',
                      'error_type': type(exc).__name__, 'physical_finish': lease.observe(),
                      'formal_gate_eligible': False}
            self._write_pair(index, 'result', result)
            raise
        self._write_pair(index, 'result', result)
        return result

    def pump(self):
        """Original source/work pump OR predecessor confirmation, never both."""
        initial = ControllerStepLease.issue(self.schedule)
        with FileLock(str(self.directory/'.pump.lock'), timeout=0), FileLock(str(self.directory/'.lock'), timeout=0):
            lease = self._start_execution(initial)
            rows = self.verify()
            if self._terminal() is not None:
                return self._terminal()
            if rows and rows[-1]['result'] is None:
                return self._step_locked(len(rows)-1, rows, lease)
            if rows and rows[-1]['result']['state'] == 'awaiting_successor_witness' and rows[-1]['witness'] is None:
                return self._confirm_predecessor(len(rows)-1, rows[-1], lease)
            if rows and not step_completed(rows[-1]):
                return {'state': 'interrupted_consumed', 'retry_allowed': False, 'formal_gate_eligible': False}
            check = lease.evaluate(lease.observe(), rows[-1]['witness']['observed'] if rows else None)
            if not check.allowed:
                return self._close_execution(lease, check.reason)
            if len(rows) == len(self.schedule.due_at):
                return {'state': 'controller_schedule_exhausted', 'formal_gate_eligible': False}
            if self.schedule.observation_mode == 'physical_UTC':
                return self._step_locked(len(rows), rows, lease)
            from factorlab.discovery.quarantined_observation import QuarantinedObservation
            from factorlab.discovery.finite_market_oracle import publish_due
            from factorlab.discovery.v7_forward import ForwardRuntime, ForwardConfig
            from factorlab.discovery.v7_lifecycle import LifecyclePlan
            runtime = None
            try:
                value = _read(self.permit.paths['runtime']/'forward.json')['identity']
                config = {**value['config'], 'symbols': tuple(value['config']['symbols'])}
                plan = {**value['lifecycle'], 'symbols': tuple(value['lifecycle']['symbols'])}
                runtime = ForwardRuntime(self.permit.paths['runtime'], self.permit.paths['feed'],
                    config=ForwardConfig(**config), lifecycle_plan=LifecyclePlan(**plan))
                lease.assert_parent_alive()
                observer = QuarantinedObservation.from_lifecycle(runtime.lifecycle)
                if observer is None:
                    return {'state': 'pending_dependencies', 'dependencies': ['bound_observation_clock'],
                            'formal_gate_eligible': False}
                state = observer.inspect(runtime.lifecycle)
                due = pd.Timestamp(self.schedule.due_at[len(rows)])
                if state['logical_at'] is None or pd.Timestamp(state['logical_at']) < due:
                    lease.assert_parent_alive()
                    publish_due(observer.generator, self.permit, max_new_receipts=1)
                    lease.assert_parent_alive()
                    observer.advance(runtime)
            except BaseException as exc:
                self._close_execution(lease, type(exc).__name__)
                raise
            finally:
                if runtime is not None:
                    runtime.close()
            return self._step_locked(len(rows), rows, lease)

    def run(self, *, max_pumps=1):
        """Same issuer, one absolute deadline, including the final witness pump.

        Confirmation never adds a source release, work attempt, or schedule slot.
        The original max_pumps bound applies to source/work invocations; a final
        confirmation-only invocation must drain their pending publication.
        """
        if type(max_pumps) is not int or not 1 <= max_pumps <= 7200:
            raise ValueError('Bounded positive original pump count required')
        progress = []
        for _ in range(max_pumps):
            result = self.pump()
            progress.append(result)
            if result['state'] == 'awaiting_successor_witness':
                result = self.pump()
                progress.append(result)
            if result['state'] in {'pending_dependencies', 'controller_schedule_exhausted',
                                   'failed_consumed', 'interrupted_consumed', 'controller_service_closed',
                                   'failed_or_unknown_consumed', 'predecessor_unconfirmed_or_consumed'}:
                break
            if result['state'] == 'pending_observation_time' and self.schedule.observation_mode == 'physical_UTC':
                break
            if (result['state'] == 'completed_step'
                    and result['output'].get('forward', {}).get('status') != 'stepped'):
                break
            if result['state'] == 'completed_step' and result['index'] == len(self.schedule.due_at)-1:
                break
        self.verify()
        return {'state': progress[-1]['state'] if progress else 'controller_schedule_exhausted',
                'progress': progress, 'formal_gate_eligible': False}

    def _settle_research(self, scheduler, service, started, spent):
        """No new controller tick, source release, queue or research reservation."""
        lease = ControllerStepLease.from_dict(self._steps()[-1]['attempt']['lease'], self.schedule)
        while True:
            remaining = min(lease.assert_parent_alive().remaining_seconds,
                            service.config.wall_seconds - spent - (time.monotonic() - started))
            if remaining <= 0:
                return scheduler.stop_and_reap('supervisor_budget')
            if service.stop_requested() or (service.directory / 'STOP').exists() or (scheduler.directory / 'STOP').exists():
                return scheduler.stop_and_reap('supervisor_stopped')
            guard = service.guard()
            if guard.get('allowed') is not True:
                return scheduler.stop_and_reap('host_guard')
            result = scheduler.settle_running()
            if result['status'] != 'pending':
                return result
            # This is physical paid supervisor time, never a logical date jump.
            time.sleep(min(.2, remaining))

    def _dispatch(self):
        """Concrete production route. No injected service, clock, command or runner."""
        from factorlab.discovery.v7_forward import ForwardConfig, ForwardRuntime
        from factorlab.discovery.v7_lifecycle import LifecyclePlan
        from factorlab.discovery.neural_forward_ticket import open_existing_authority
        from factorlab.discovery.v7_service import ForwardService, ServiceConfig
        from factorlab.discovery.v7_autonomous import AutonomousConfig, AutonomousService, ResearchScheduler
        from factorlab.discovery.v7_run import V7Config
        from factorlab.discovery.scheduled_search import FrozenSWorkSchedule
        paths = self.permit.paths
        # Constructors may initialize stores: forbid their use to repair missing originals.
        for path in (paths['runtime']/'forward.json', paths['runtime']/'lifecycle.sqlite',
                     paths['feed']/'identity.json', paths['feed']/'receipts.sqlite'):
            if not path.is_file():
                raise ValueError('Original runtime/feed store missing; no recreation')
        wrapper = _read(paths['runtime']/'forward.json')['identity']
        config = {**wrapper['config'], 'symbols': tuple(wrapper['config']['symbols'])}
        plan = {**wrapper['lifecycle'], 'symbols': tuple(wrapper['lifecycle']['symbols'])}
        authority = open_existing_authority(paths['authority'])
        if authority.manifest_hash != self.identity['authority_manifest_hash']:
            raise ValueError('Original controller wallet changed')
        runtime = ForwardRuntime(paths['runtime'], paths['feed'], config=ForwardConfig(**config),
                                 lifecycle_plan=LifecyclePlan(**plan))
        service = scheduler = None
        started = spent = None
        try:
            self.permit.verify(runtime.lifecycle)
            config = _plain(self.schedule.autonomous)
            config['research']['symbols'] = tuple(config['research']['symbols'])
            config['research'] = V7Config(**config['research'])
            service = ForwardService(runtime, config=ServiceConfig(**self.schedule.service), work_authority=authority)
            scheduler = ResearchScheduler(runtime, config=AutonomousConfig(**config), authority=authority,
                schedule=FrozenSWorkSchedule.from_dict(self.schedule.search_schedule))
            if service.state.get('controller_research_active') is not None:
                # An interrupted drain cannot reconstruct elapsed time from a
                # historical logical date or acquire a replacement issuer.
                scheduler.stop_and_reap('supervisor_stopped')
                service.state['spent_seconds'] = max(service.state['spent_seconds'], service.config.wall_seconds)
                service.state['controller_research_active'] = None
                service.state['active_since'] = None
                service._save('controller_interrupted_research_consumed')
            lease = ControllerStepLease.from_dict(self._execution_clock()['lease'], self.schedule)
            lease.assert_parent_alive()
            # Durable spent_seconds is a lower bound only. Admission always
            # uses the first absolute clock, including validation/setup, idle,
            # prior SQLite/close/return tails, and no renewal after restart.
            service.state['spent_seconds'] = max(service.state['spent_seconds'], lease.elapsed(lease.observe()))
            started, spent = time.monotonic(), service.state['spent_seconds']
            service.state['controller_research_active'] = {
                'controller_hash': fingerprint(self.identity),
                'attempt_hash': fingerprint(self._steps()[-1]['attempt']),
                'spent_before': spent}
            service._save('controller_research_started')
            # This production composite performs receipt tick, typed cold fits,
            # V/C single-look endpoints, A0 waiting/adoption, then actual S worker.
            observer = None
            if self.schedule.observation_mode == 'bound_logical_observation':
                from factorlab.discovery.quarantined_observation import (
                    QuarantinedObservation, current_binding, _READ_SCOPE)
                if _READ_SCOPE.get() is not None:
                    raise ValueError('Full controller cannot inherit a paid observation read scope')
                observer = QuarantinedObservation.from_lifecycle(runtime.lifecycle)
                if observer is None:
                    raise ValueError('Bound logical observation disappeared before dispatch')
            # Every supported observation-prefix writer first takes this LC
            # lock. Same-instance tick/service sections can reenter it. Do not
            # install an observation read scope around paid production: each
            # ticket must close its OWN full audit before its original 300s
            # assertion. The outer controller's longer clock cannot cover it.
            with (runtime.lifecycle.controller_lock() if observer else nullcontext()):
                if observer is not None:
                    attempt = self._steps()[-1]['attempt']
                    if current_binding(runtime.lifecycle) != attempt['observation_prefix']:
                        raise ValueError('Observation advanced after original controller reservation')
                lease.assert_parent_alive()
                service.state['spent_seconds'] = max(service.state['spent_seconds'], lease.elapsed(lease.observe()))
                # The service scope must inspect the durably charged state.
                # Persist this later native observation before its checkpoint
                # verification; never hide setup/lock time or relax equality.
                service._save('controller_dispatch_time_observed')
                output = AutonomousService(service, scheduler).step()
                if observer is not None and current_binding(runtime.lifecycle) != attempt['observation_prefix']:
                    raise ValueError('Observation changed inside original controller dispatch')
            # Release the lifecycle lock before waiting: the original child
            # must independently reconstruct its captured immutable prefix.
            # This controller does not advance that prefix until it is settled.
            output['research_terminal'] = self._settle_research(scheduler, service, started, spent)
            return output
        finally:
            try:
                if scheduler is not None:
                    scheduler.stop_and_reap('supervisor_stopped')
                if service is not None and started is not None:
                    service.state['spent_seconds'] = max(service.state['spent_seconds'],
                        spent + max(0., time.monotonic() - started))
                    service.state['controller_research_active'] = None
                    service.state['active_since'] = None
                    service._save('controller_research_finished')
            finally:
                if scheduler is not None:
                    scheduler.close()
                if service is not None:
                    service.close(close_runtime=False)
                runtime.close()


def backend_for_job(job_path):
    """Fixed worker integration: None for an ordinary nonquarantine job.

    Call after the existing worker has verified its launch token and outer
    lease. A missing original binding is an error, never a fresh provider path.
    """
    from factorlab.discovery.neural_forward_ticket import _LifecycleReader
    from factorlab.discovery.quarantine_execution import permit_for
    path = Path(job_path)
    runtime = path.parents[3]  # forward/autonomous/jobs/YYYY-MM/job.json
    reader = _LifecycleReader(runtime)
    try:
        permit = permit_for(reader)
    finally:
        reader.close()
    if permit is None:
        return None
    return RegisteredTrialCodexBackend(permit, path)


class RegisteredTrialCodexBackend:
    """Actual restricted provider; only the original active family debit calls it."""
    def __init__(self, permit, job_path):
        self.permit = _permit(permit)
        self.job_path = _no_redirect(Path(job_path), permit.paths['runtime'])
        self.job = _read(self.job_path)
        expected = permit.paths['runtime']/'autonomous'/'jobs'/self.job['month']/'job.json'
        if (self.job_path != expected or Path(self.job['run_directory']) != permit.paths['research']
                or Path(self.job['feed_directory']) != permit.paths['feed']
                or self.job['work_authority']['directory'] != str(permit.paths['authority'])
                or self.job['source_files'] != source_files()):
            raise ValueError('Original worker research/feed/wallet namespace differs from registered trial')
        import sqlite3
        from contextlib import closing
        database = permit.paths['runtime']/'autonomous'/'scheduler.sqlite'
        with closing(sqlite3.connect(database.as_uri()+'?mode=ro', uri=True)) as db:
            row = db.execute('SELECT value,hash FROM jobs WHERE month=?', (self.job['month'],)).fetchone()
        if row is None:
            raise ValueError('Original registered provider worker reservation missing')
        reservation = json.loads(row[0])
        if (fingerprint(reservation) != row[1] or reservation['job_hash'] != fingerprint(self.job)
                or reservation['job_path'] != str(self.job_path) or reservation['status'] != 'running'):
            raise ValueError('Provider job differs from original running worker reservation')
        self.context = {'version': VERSION, 'trial': permit.trial, 'provider_policy': _provider_policy(permit),
                        'job_hash': fingerprint(self.job), 'job_path': str(self.job_path),
                        'context_policy': 'ephemeral_no_resume_no_user_rules_no_host_memory',
                        'provider_authenticity_certified': False}
        self.directory = permit.paths['provider']/self.job['month']
        context_path = self.directory/'context.json'
        if not context_path.exists() and self.directory.exists() and any(self.directory.iterdir()):
            raise ValueError('Original provider context missing; cannot replace attempt history')
        immutable_record(context_path, self.context)
        self.verify()

    def verify(self):
        self.permit.verify()
        if (_read(self.job_path) != self.job
                or _read(self.directory/'context.json') != self.context):
            raise ValueError('Original trial provider/job context changed')
        verify_memory_namespace(self.permit.paths['research'])

    def __call__(self, *args, **kwargs):
        raise ValueError('Trial provider requires original active reserved family work')

    def call_reserved(self, packet, *, model, reasoning, work_budget):
        from factorlab.discovery.work_authority_bridge import AuthorityWorkBudget
        from factorlab.discovery.v7_agent import CodexBatchBackend
        if (type(work_budget) is not AuthorityWorkBudget or work_budget.bridge._active is not work_budget
                or work_budget.bridge.store.directory != self.permit.paths['research']
                or work_budget.bridge.authority.directory != self.permit.paths['authority']
                or work_budget.record['binding']['work']['stage'] != 'family'):
            raise ValueError('Trial provider requires its original active shared family reservation')
        if (model, reasoning) != (DEFAULT_MODEL, DEFAULT_REASONING):
            raise ValueError('Original requested provider identity differs')
        self.verify()
        work_budget.check_time()

        original_owner = self

        class OriginalLeaseBackend(CodexBatchBackend):
            def _run(self, command, text, runtime):
                work_budget.check_time()
                self.timeout_seconds = min(180., work_budget.lease.assert_parent_alive().remaining_seconds)
                from .full_mechanism_contract import _begin_provider, _finish_provider
                original = _begin_provider(original_owner, work_budget, command, text, runtime)
                try:
                    return super()._run(command, text, runtime)
                finally:
                    _finish_provider(runtime, original, work_budget)

        backend = OriginalLeaseBackend(self.directory/'attempts',
            timeout_seconds=min(180., work_budget.lease.assert_parent_alive().remaining_seconds))
        reply = backend(packet, model=model, reasoning=reasoning)
        work_budget.check_time()
        self.verify()
        return reply
