"""Invocation-only restriction from the actual service's original physical time.

This never grants work, changes a child lease, or makes historical artifacts
eligible. Standalone producers retain their original contracts. Within an actual
service step, fresh 300s issuers and final serving publication additionally obey
this original native dual-clock ceiling. Cooperative Python checks do not promise
hard real-time preemption during blocking OS/native calls.
"""
import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
from datetime import timedelta
from threading import get_ident
from types import SimpleNamespace

from factorlab.protocol import fingerprint, project_path

from .work_lease import FrozenWorkLease, LeaseClosed, _evaluate, _observe, _utc

_ACTIVE = ContextVar('original_forward_service_scope', default=None)
VERSION = 'v7-continuous-native-service-ceiling-v1'
CONTROLLER_CLOCK_VERSION = 'v7-original-controller-service-clock-v2'


class ServiceDeadlineClosed(ValueError):
    pass


def _controller_clock(lifecycle, service_identity, runtime_identity, *, active=None, full=True):
    """Reconstruct an original registered C clock; never accept a supplied clock.

    Full entry/paid reads verify the permit, wallet, controller and all steps.
    Watchdog restriction reads retain immutable mirrors and original LC anchors,
    but do not repeat disk-wide source hashing. This path grants no work.
    """
    from .calibration_runtime_contract import _original_operations
    from .full_trial_controller import BOUND, VERSION as CV, FullTrialController, FullControllerSchedule, ControllerStepLease
    from .full_calibration_registry import EVENT
    operations = _original_operations(lifecycle)
    bound = [o['detail'] for o in operations if o['kind']==BOUND]
    enrolled = [o for o in operations if o['kind']==EVENT]
    if not bound:
        if active is not None or (enrolled and (lifecycle.directory.parent/'controller').exists()):
            raise ValueError('Original controller service binding disappeared')
        return None
    if len(bound)!=1 or len(enrolled)!=1:
        raise ValueError('Original controller enrollment/binding is ambiguous')
    runtime = project_path(lifecycle.directory)
    root, registry = runtime.parent, runtime.parent.parent.parent
    read = lambda path: json.loads(path.read_text('utf-8'))
    identity = read(root/'controller'/'identity.json')
    intent = enrolled[0]['detail']
    trial_id = identity['trial']['trial_id']
    schedule = FullControllerSchedule.from_dict(identity['schedule'])
    clock = read(root/'controller'/'steps'/'-001-clock.json')
    binding = {'controller_hash':fingerprint(identity), 'clock_hash':fingerprint(clock),
               'schedule_digest':schedule.digest}
    anchors = [o['detail'] for o in operations if o['kind']=='full_trial_controller_step'
               and o['detail']['phase']=='clock']
    if (bound != [{'identity_hash':binding['controller_hash']}]
            or identity != read(registry/'controller_plans'/(trial_id+'.json'))
            or identity['version']!=CV or identity['schedule_digest']!=schedule.digest
            or identity['runtime_identity_hash']!=fingerprint(runtime_identity)
            or identity['source_files']!=runtime_identity['source_files']
            or identity['authority_manifest_hash']!=intent['authority_manifest_hash']
            or service_identity.get('work_authority')!={'directory':intent['paths']['authority'],
                                                       'manifest_hash':identity['authority_manifest_hash']}
            or service_identity['config']!=schedule.service
            or intent['paths']['runtime']!=str(runtime) or intent['paths']['root']!=str(root)
            or any(intent.get(k)!=v for k,v in identity['trial'].items())
            or read(root/'enrollment_intent.json')!=intent
            or read(root/'enrollment.json')!={'intent':intent,'original_event_hash':fingerprint(enrolled[0])}
            or clock!=read(registry/'controller_steps'/trial_id/'-001-clock.json')
            or clock.get('version')!=CV or clock.get('controller_hash')!=binding['controller_hash']
            or anchors!=[{'controller_hash':binding['controller_hash'],'index':-1,
                         'phase':'clock','content_hash':binding['clock_hash']}]
            or identity.get('service_clock')!='continuous_original_dual_clock_including_idle_no_restart'
            or identity.get('completion_policy')!='successor_observes_original_pending_result_no_self_completion'
            or identity.get('formal_gate_eligible') is not False):
        raise ValueError('Original controller clock/runtime/wallet binding changed')
    lease = ControllerStepLease.from_dict(clock['lease'],schedule)
    native = FrozenWorkLease.from_dict(service_identity['native_execution_origin'])
    outer = lease.supervisor()
    if (native.parent_pid!=outer.parent_pid or native.parent_created!=outer.parent_created
            or native.boot_id!=outer.boot_id
            or native.monotonic_started<outer.monotonic_started
            or _utc(native.wall_started_at)<_utc(outer.wall_started_at)):
        raise ValueError('Original service/controller issuer cannot be replaced')
    if full:
        from .quarantine_execution import permit_for
        permit = permit_for(lifecycle)
        if permit is None:
            raise ValueError('Original controller permit missing')
        controller = FullTrialController.inspect(permit)
        # inspect already verifies the complete original step/mirror/LC chain.
        if controller.identity!=identity or controller._execution_clock()!=clock:
            raise ValueError('Original controller changed during restriction read')
    if active is not None:
        attempts = [o['detail'] for o in operations if o['kind']=='full_trial_controller_step'
                    and o['detail']['phase']=='attempt']
        if (type(active) is not dict or not attempts or active.get('controller_hash')!=binding['controller_hash']
                or active.get('attempt_hash')!=attempts[-1]['content_hash']):
            raise ValueError('Service scope changed its original controller attempt')
    return {'version':CONTROLLER_CLOCK_VERSION,'identity_hash':fingerprint(service_identity),
            'issuer':clock['lease']['issuer'],'spent_before':0.,
            'service_seconds':schedule.service['wall_seconds'],'controller_binding':binding}


class _Scope:
    def __init__(self, service, origin=None):
        from .v7_service import ForwardService
        if not isinstance(service, ForwardService) or not service._owner_lock.is_locked:
            raise TypeError('Original owned ForwardService step required')
        # Sample native physical clocks before any identity/ledger read. The
        # injectable service.monotonic is deliberately not a production clock.
        origin = service._native_construction_origin if origin is None else origin
        if type(origin) is not FrozenWorkLease:
            raise TypeError('Original native step sample required')
        self.service = service
        self.thread, self.pid = get_ident(), os.getpid()
        self.runtime = project_path(service.runtime.directory)
        self.identity = json.loads((service.directory/'identity.json').read_text('utf-8'))
        if (self.identity['config'] != asdict(service.config)
                or self.identity['lifecycle'] != service.lifecycle.plan.digest
                or self.identity.get('native_clock_protocol')!=VERSION
                or self.identity.get('native_execution_origin')!=origin.to_dict()):
            raise ValueError('Service scope changed the original persisted identity')
        database = [r[2] for r in service.db.execute('PRAGMA database_list') if r[1]=='main']
        if len(database)!=1 or project_path(database[0])!=service.directory/'state.sqlite':
            raise ValueError('Service scope is not its original state database')
        row = service.db.execute('SELECT value,hash FROM checkpoint WHERE id=1').fetchone()
        if row is None or json.loads(row[0])!=service.state or fingerprint(service.state)!=row[1]:
            raise ValueError('Service scope lost its original charged checkpoint')
        service._verify(row[1])
        active = service.state.get('controller_research_active')
        controller_clock = _controller_clock(service.lifecycle, self.identity, service.runtime.identity, active=active)
        if controller_clock is not None and active is None:
            raise ValueError('Registered controller execution cannot drop its original active attempt')
        path = service.directory/'native-clock.json'
        existing = service.state.get('native_service_clock')
        if path.exists() != (existing is not None):
            raise ValueError('Original native service clock/state mirror missing; never renew')
        if existing is None:
            if (service.state['steps'] or any(json.loads(r[0])['event']=='native_service_clock_bound'
                    for r in service.db.execute('SELECT value FROM events'))):
                raise ValueError('Original service execution cannot restart without its native clock')
            existing = controller_clock or {'version': VERSION, 'identity_hash': fingerprint(self.identity),
                        'issuer': origin.to_dict(), 'spent_before': service.state['spent_seconds'],
                        'service_seconds': service.config.wall_seconds}
            from .v7_storage import immutable_record
            immutable_record(path, existing)
            service.state['native_service_clock'] = existing
            service._save('native_service_clock_bound')
        if (json.loads(path.read_text('utf-8'))!=existing
                or (controller_clock is not None and existing!=controller_clock)
                or (controller_clock is None and (set(existing)!={'version','identity_hash','issuer','spent_before','service_seconds'}
                    or existing['version']!=VERSION or existing['issuer']!=origin.to_dict()))
                or existing['identity_hash']!=fingerprint(self.identity)
                or existing['service_seconds']!=service.config.wall_seconds):
            raise ValueError('Original native service clock changed')
        self.clock = existing
        self.origin = FrozenWorkLease.from_dict(existing['issuer'])
        self.spent = existing['spent_before']
        remaining = service.config.wall_seconds-self.spent
        values = self.origin.to_dict()
        values['wall_deadline'] = (_utc(self.origin.wall_started_at)+timedelta(seconds=remaining)).isoformat()
        values['monotonic_deadline'] = self.origin.monotonic_started+remaining
        self.ceiling = SimpleNamespace(**values)
        self.previous = None
        self.rejection = None
        # A registered primary uses the original C start exactly once. An
        # ordinary service retains its own immutable native origin and debit.
        # No old clock record is migrated or interpreted under the new schema.

    def check(self, runtime, *, seconds=0, lease=None):
        if (get_ident()!=self.thread or os.getpid()!=self.pid or self.pid!=self.origin.parent_pid or project_path(runtime.directory)!=self.runtime
                or self.service.state['spent_seconds']<self.spent
                or self.service.state.get('native_service_clock')!=self.clock
                or json.loads((self.service.directory/'native-clock.json').read_text('utf-8'))!=self.clock
                or json.loads((self.service.directory/'identity.json').read_text('utf-8'))!=self.identity):
            raise ValueError('Original service deadline scope identity changed')
        observation = _observe(self.ceiling)
        check = self.rejection or _evaluate(self.ceiling, observation, self.previous)
        if not check.allowed:
            self.rejection = check
            if check.reason != 'deadline':
                raise LeaseClosed(check)
        if observation.get('pid')!=self.pid and check.allowed:
            raise ValueError('Original service deadline scope identity changed')
        if (not check.allowed or check.remaining_seconds<seconds
                or self.service.state['spent_seconds']>=self.service.config.wall_seconds
                or self.service.config.wall_seconds-self.service.state['spent_seconds']<seconds):
            raise ServiceDeadlineClosed('Original service deadline has insufficient full-work headroom')
        self.previous = observation
        if lease is not None:
            if (type(lease) is not FrozenWorkLease or lease.lease_seconds!=300
                    or lease.parent_pid!=self.origin.parent_pid or lease.parent_created!=self.origin.parent_created
                    or lease.boot_id!=self.origin.boot_id):
                raise ValueError('New work lease differs from original service issuer or unchanged 300s contract')
            if (_utc(lease.wall_deadline)>_utc(self.ceiling.wall_deadline)
                    or lease.monotonic_deadline>self.ceiling.monotonic_deadline):
                raise ServiceDeadlineClosed('New unchanged 300s lease exceeds original service deadline')

    def finish(self):
        # Observe after the step body's durable final write. This next write's
        # own tail is deliberately not self-certified: all later execution uses
        # the same first absolute clock, so that tail/idle time never refunds it.
        def failure(observation, previous):
            check = self.rejection or _evaluate(self.ceiling, observation, previous)
            if not check.allowed and check.reason != 'deadline':
                return LeaseClosed(check), check.reason
            if observation.get('pid')!=self.pid or self.pid!=self.origin.parent_pid:
                return ValueError('Original service deadline scope identity changed'), 'scope_identity_changed'
            if not check.allowed or self.service.state['spent_seconds']>=self.service.config.wall_seconds:
                return (ServiceDeadlineClosed('Original native service deadline consumed; no new session issuer'),
                        'original_native_service_deadline_consumed')
            return None, None

        def consume(reason):
            self.service.state['spent_seconds']=max(self.service.state['spent_seconds'],
                                                   self.service.config.wall_seconds)
            self.service.state['last_result']={'status':'budget_exhausted','reason':reason}
            self.service._save('native_service_deadline_consumed' if reason=='original_native_service_deadline_consumed'
                               else 'native_service_scope_closed')

        observation = _observe(self.ceiling)
        error, reason = failure(observation, self.previous)
        elapsed = (self.ceiling.monotonic_deadline-self.origin.monotonic_started
                   if observation.get('unavailable') else
                   max(0., observation['monotonic']-self.origin.monotonic_started,
                       observation['wall']-_utc(self.origin.wall_started_at).timestamp()))
        self.service.state['spent_seconds']=max(self.service.state['spent_seconds'],self.spent+elapsed,
            self.service.config.wall_seconds if error is not None else 0.)
        if error is not None:
            consume(reason)
        else:
            self.service._save('native_service_time_observed')
        # A late observed tail is negative evidence, never a renewed positive
        # certificate. Persist it before returning to a composite scheduler.
        after = _observe(self.ceiling)
        if error is None:
            error, reason = failure(after, observation)
            if error is not None:
                consume(reason)
        if error is not None:
            raise error


@contextmanager
def _service_step_scope(service, *, origin=None):
    if _ACTIVE.get() is not None:
        raise ValueError('Service steps cannot nest or renew the original deadline')
    scope = _Scope(service, origin)
    token = _ACTIVE.set(scope)
    try:
        try:
            scope.check(service.runtime)
            yield
        except BaseException as error:  # Finish on all exits while preserving the original failure.
            try:
                scope.finish()
            except BaseException as closing:  # noqa: BLE001 - retain cleanup evidence without replacing the root failure
                # Keep the step's original failure primary, with the closing
                # failure in its context, including a genuinely late deadline.
                # Splice out the back-link before re-raising: Python's cycle
                # breaking would otherwise discard the original older context.
                tail = closing
                while tail.__context__ is not None and tail.__context__ is not error:
                    tail = tail.__context__
                if tail.__context__ is error:
                    tail.__context__ = error.__context__
                raise error
            raise
        else:
            scope.finish()
    finally:
        _ACTIVE.reset(token)


def check_service_work(runtime, *, lease=None, new_work=False):
    """Restriction only; normal standalone issuer/admission checks still apply."""
    scope = _ACTIVE.get()
    if scope is not None:
        scope.check(runtime, seconds=302 if new_work else 0, lease=lease)



@contextmanager
def _service_followup_scope(service):
    # A followup is never a first execution and can never renew the clock.
    if service.state.get('native_service_clock') is None:
        raise ValueError('Research requires the original already-entered service clock')
    with _service_step_scope(service):
        yield


def current_service_reference():
    scope=_ACTIVE.get()
    if scope is None:
        return None
    scope.check(scope.service.runtime)
    value={'version':VERSION,'directory':str(scope.service.directory),
           'identity_hash':fingerprint(scope.identity),'clock_hash':fingerprint(scope.clock),
           'runtime_identity_hash':fingerprint(scope.service.runtime.identity),
           'wall_deadline':scope.ceiling.wall_deadline,
           'monotonic_deadline':scope.ceiling.monotonic_deadline}
    previous=scope.service.state.get('research_service_restriction')
    path=scope.service.directory/'research-restriction.json'
    if path.exists() != (previous is not None):
        raise ValueError('Original research service restriction mirror missing')
    if previous is None:
        from .v7_storage import immutable_record
        immutable_record(path,value)
        scope.service.state['research_service_restriction']=value
        scope.service._save('original_research_service_restriction_bound')
    elif previous!=value or json.loads(path.read_text('utf-8'))!=value:
        raise ValueError('Original research service restriction cannot change')
    scope.check(scope.service.runtime)
    return value


def verify_service_reference(value, *, _verify_source=True):
    return _verify_service_reference(value, _verify_source=_verify_source, historical=False)


def verify_historical_service_reference(value):
    """Read durable original evidence only; grants no live execution authority."""
    return _verify_service_reference(value, _verify_source=True, historical=True)


def _verify_service_reference(value, *, _verify_source, historical):
    # Read-only restriction usable by the original issuer/direct child. It
    # reconstructs the original service chain; no runtime/service is created.
    import sqlite3
    from .v7_run import source_files
    if (type(value) is not dict or set(value)!={'version','directory','identity_hash','clock_hash',
            'runtime_identity_hash','wall_deadline','monotonic_deadline'} or value['version']!=VERSION):
        raise ValueError('Unknown original service restriction')
    directory=project_path(value['directory'])
    identity=json.loads((directory/'identity.json').read_text('utf-8'))
    clock=json.loads((directory/'native-clock.json').read_text('utf-8'))
    with sqlite3.connect((directory/'state.sqlite').as_uri()+'?mode=ro',uri=True) as db:
        db.execute('BEGIN')  # One coherent snapshot while parent commits its next state.
        row=db.execute('SELECT value,hash FROM checkpoint WHERE id=1').fetchone()
        if row is None:raise ValueError('Missing original service checkpoint')
        state=json.loads(row[0]);prior='0'*64;last=None
        for seq,raw,digest in db.execute('SELECT seq,value,hash FROM events ORDER BY seq'):
            event=json.loads(raw)
            if event['sequence']!=seq or event['previous']!=prior or fingerprint(event)!=digest:
                raise ValueError('Original service event chain changed')
            prior,last=digest,event['state_hash']
    wrapper=json.loads((directory.parents[1]/'forward.json').read_text('utf-8'))
    from .v7_lifecycle import LifecyclePlan
    lifecycle_fields = dict(wrapper['identity']['lifecycle'])
    lifecycle_fields['symbols'] = tuple(lifecycle_fields['symbols'])
    lifecycle_digest = LifecyclePlan(**lifecycle_fields).digest
    if (fingerprint(state)!=row[1] or last!=row[1] or state.get('native_service_clock')!=clock
            or state.get('research_service_restriction')!=value
            or json.loads((directory/'research-restriction.json').read_text('utf-8'))!=value
            or fingerprint(identity)!=value['identity_hash'] or fingerprint(clock)!=value['clock_hash']
            or clock['identity_hash']!=value['identity_hash']
            or clock['version'] not in (VERSION,CONTROLLER_CLOCK_VERSION)
            or clock['service_seconds']!=identity['config']['wall_seconds']
            or identity.get('native_clock_protocol')!=VERSION
            or (clock['version']==VERSION and identity.get('native_execution_origin')!=clock['issuer'])
            or wrapper!={'identity':wrapper['identity'],'sha256':fingerprint(wrapper['identity'])}
            or wrapper['sha256']!=value['runtime_identity_hash']
            or (_verify_source and wrapper['identity']['source_files']!=source_files())
            or lifecycle_digest!=identity['lifecycle']):
        raise ValueError('Research lost its original service/runtime/source restriction')
    from .neural_forward_ticket import _LifecycleReader
    reader=_LifecycleReader(directory.parents[1])
    try:
        expected=_controller_clock(reader,identity,wrapper['identity'],full=_verify_source)
    finally:
        reader.close()
    if ((expected is not None and clock!=expected)
            or (expected is None and clock['version']!=VERSION)):
        raise ValueError('Research lost its canonical original controller clock')
    original=FrozenWorkLease.from_dict(clock['issuer'])
    remaining=clock['service_seconds']-clock['spent_before']
    if (expected is not None and (value['wall_deadline']!=(_utc(original.wall_started_at)+timedelta(seconds=remaining)).isoformat()
            or value['monotonic_deadline']!=original.monotonic_started+remaining)):
        raise ValueError('Research cannot replace its canonical original controller deadline')
    if (_utc(value['wall_deadline'])>_utc(original.wall_started_at)+timedelta(seconds=remaining)
            or value['monotonic_deadline']>original.monotonic_started+remaining):
        raise ValueError('Research cannot extend its original service deadline')
    ceiling=SimpleNamespace(**{**original.to_dict(),'wall_deadline':value['wall_deadline'],
                              'monotonic_deadline':value['monotonic_deadline']})
    if historical:
        return ceiling  # All immutable mirrors/chain checked above; never renew the clock.
    observation=_observe(ceiling)
    checked=_evaluate(ceiling,observation)
    # This is read-only restriction inspection, including neural grandchildren.
    # It grants no issuer ownership; actual work leases retain their own parent.
    if state['spent_seconds']>=clock['service_seconds'] or (not checked.allowed and checked.reason!='wrong_process'):
        raise TimeoutError('Original service research restriction consumed')
    return ceiling
