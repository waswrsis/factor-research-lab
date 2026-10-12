"""STAGE: dedicated issuer, original typed outer scope and sequential prepaid work.
No grant is created here. Scope and work sources are reopened at every handshake.
Not yet wired into CLI or the combined campaign service.
"""
from __future__ import annotations

import ctypes
import inspect
import json
import os
import runpy
import subprocess
import sys
import threading
import time
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lease_enforcer as base

from factorlab import protocol
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority, BudgetExhausted
from factorlab.discovery.s_outer_context import SOuterContext
from factorlab.discovery.work_lease import FrozenWorkLease, _boot_id


def _write(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    base._write(path,value)


def _copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _reservation_request(value, lease):
    """Canonical pre-reserve inputs; normalization grants no work or allowance."""
    from factorlab.discovery import monthly_work_authority as wallet
    if type(value) is not dict or set(value) != {'work_key', 'counts', 'binding', 'lease_seconds', 'retry_id'}:
        raise ValueError('Incomplete original reservation request')
    work_key = wallet._text(value['work_key'], 'work_key')
    counts, binding = wallet._counts(value['counts']), wallet._object(value['binding'], 'binding')
    if value['retry_id'] is not None:
        wallet._text(value['retry_id'], 'retry_id')
    seconds = value['lease_seconds']
    if (not any(counts.values()) or type(seconds) is not int or not 1 <= seconds <= 300
            or seconds != lease.lease_seconds):
        raise ValueError('Original reservation counts or lease changed')
    return {'work_key': work_key, 'counts': counts, 'binding': binding,
            'lease_seconds': seconds, 'retry_id': value['retry_id']}


def _verify_unfunded(scope, pending, error_message, issuer):
    """Independent read-only proof of actual canonical quota refusal."""
    from factorlab.discovery import monthly_work_authority as wallet
    lease = FrozenWorkLease.from_dict(pending['lease'])
    request = _reservation_request(pending['request'], lease)
    if (request != pending['request'] or lease.parent_pid != issuer.pid
            or lease.parent_created != issuer.create_time()
            or {'directory': pending['authority_directory'],
                'manifest_hash': pending['authority_manifest_hash']} != scope['authority']):
        raise ValueError('Unfunded request changed its watched issuer or wallet')
    _clock(scope['clock'])
    check, _ = base._lease_check(lease, issuer, None)
    if not check.allowed:
        raise TimeoutError('Original unfunded work ' + check.reason)
    authority = base._authority(pending['authority_directory'])
    if authority.manifest_hash != pending['authority_manifest_hash']:
        raise ValueError('Unfunded authority manifest changed')
    _role(scope, {**request, 'lease': lease.to_dict()})
    reason = None
    canonical_reasons = {'Monthly prepaid work quota exhausted'}
    if authority.manifest['partition_policy'] is not None:
        canonical_reasons.update('Nontransferable monthly ' + lane + ' quota exhausted'
                                 for lane in authority.manifest['partition_policy']['lanes'])
    if error_message not in canonical_reasons:
        raise ValueError('Unregistered canonical quota exception')
    with authority._transaction() as db:
        verified = authority._verify(db)
        if any(row['work_key'] == request['work_key'] for row in verified['reservations'].values()):
            raise ValueError('A reserved work cannot close as unfunded')
        month = lease.wall_started_at[:7]
        if verified['wallets'] and month < max(verified['wallets']):
            raise ValueError('Unfunded request moved its work month backwards')
        ceiling = (authority._month_ceiling(month) if hasattr(authority, '_month_ceiling') else
                   {'counts': authority.limits, 'seconds': authority.manifest['monthly_seconds']})
        current = verified['wallets'].get(month, {'counts': wallet._counts({}), 'seconds': 0})
        if (any(current['counts'][key] + request['counts'][key] > ceiling['counts'][key]
                for key in ceiling['counts'])
                or current['seconds'] + request['lease_seconds'] > ceiling['seconds']):
            reason = 'Monthly prepaid work quota exhausted'
        elif authority.manifest['partition_policy'] is not None:
            with authority._identity_read_scope(db):
                allocation = wallet.allocation_route(authority, request['work_key'], request['binding'],
                    current=True, lease=lease, debit=request['counts'])
            try:
                wallet.charge_partition(authority.manifest['partition_policy'],
                    _copy(verified['partitions']), _copy(verified['owners']), month,
                    allocation, request['counts'], request['lease_seconds'])
            except BudgetExhausted as error:
                if type(error) is not BudgetExhausted:
                    raise
                reason = str(error)
        # Another owned producer can consume more allowance after the original
        # refusal. The first failing quota may change; no allowance can return.
        if reason is None:
            raise ValueError('Canonical wallet does not prove this quota refusal')
    if scope != _scope(scope['reference'], base._identity(issuer)):
        raise ValueError('Outer authorization changed during unfunded verification')
    _clock(scope['clock'])
    check, _ = base._lease_check(lease, issuer, None)
    if not check.allowed:
        raise TimeoutError('Late unfunded acknowledgement')
    return reason


def _campaign_platform():
    return sys.platform


def _scope(reference, issuer):
    """Trusted durable S or actual prepaid Campaign active step; no float grant."""
    if reference['kind'] == 'S':
        value = SOuterContext.verify_bound(reference['bound'])
        clock = value['clock']
        origin = FrozenWorkLease.from_dict(clock['issuer'])
        return {'reference': _copy(reference), 'clock': {'wall_started': origin.wall_started_at,
            'wall_deadline': clock['wall_deadline'], 'monotonic_started': origin.monotonic_started,
            'monotonic_deadline': clock['monotonic_deadline'], 'boot_id': origin.boot_id,
            'parent': {'pid': origin.parent_pid, 'created': origin.parent_created}},
            'authority': value['authority'], 'directory': value['directory'],
            'role': 'S', 'source_reference': value['source_reference'], 'scope_hash': value['scope_hash']}
    if reference['kind'] != 'campaign':
        raise ValueError('Actual typed S or prepaid Campaign step required')
    from scripts import v7_long_running as campaign
    directory = protocol.project_path(reference['directory'])
    state = campaign.inspect_campaign(directory)
    plan = campaign.read_plan(directory)
    active = state['active_step']
    if (state['terminal'] or (directory/'STOP').exists() or not active
            or active['id'] != reference['step_id'] or not state['sessions'] or not state['launches']):
        raise ValueError('Campaign step is not its original active reservation')
    session, launch = state['sessions'][-1], state['launches'][-1]
    if (active['session_id'] != session['session_id'] or active['launch_token'] != launch['token']
            or session['launch_token'] != launch['token'] or launch['process'] is None
            or {k: launch['process'][k] for k in ('pid','created')} != issuer
            or datetime.fromisoformat(active['deadline']) > datetime.fromisoformat(plan.end_at)
            or campaign.active_step_expired(state, launch['token'], {})):
        raise ValueError('Campaign original process/session/deadline changed')
    # Linux boot is globally observable. Non-Linux Campaign clocks intentionally
    # have process-local test identities: this path is rejected, not weakened.
    if _campaign_platform() != 'linux' or active['boot_identity'] != campaign.boot_identity():
        raise ValueError('Production Campaign outer scope requires original Linux boot')
    from factorlab.discovery.neural_forward_ticket import _LifecycleReader
    reader = _LifecycleReader(plan.forward_directory)
    try:
        operations = reader.verify()['operations']
        rows = [o['detail'] for o in operations if o['kind'] == 'monthly_work_authority_bound']
        if len(rows) != 1:
            raise ValueError('Campaign runtime needs exactly one actual monthly wallet')
        from factorlab.discovery.neural_forward_ticket import (
            _verify_authority_binding,
            open_existing_authority,
        )
        actual = open_existing_authority(rows[0]['authority_directory'])
        _verify_authority_binding(reader,actual,operations)
        authority = {'directory':str(actual.directory),'manifest_hash':actual.manifest_hash}
    finally:
        reader.close()
    remaining_session = min(datetime.fromisoformat(session['reserved_at']).timestamp()+session['wall_seconds'],
                            datetime.fromisoformat(plan.end_at).timestamp())
    idle_clock = {'wall_started':active['began_at'], 'wall_deadline':datetime.fromtimestamp(remaining_session,UTC).isoformat(),
        'monotonic_started':active['monotonic_started'],
        'monotonic_deadline':active['monotonic_started']+remaining_session-datetime.fromisoformat(active['began_at']).timestamp(),
        'boot_id':_boot_id(),'parent':issuer}
    return {'reference': _copy(reference), 'idle_clock':idle_clock, 'clock': {'wall_started': active['began_at'],
        'wall_deadline': active['deadline'], 'monotonic_started': active['monotonic_started'],
        'monotonic_deadline': active['monotonic_deadline'], 'boot_id': _boot_id(), 'parent': issuer},
        'authority': authority, 'directory': plan.forward_directory, 'role': 'forward',
        'scope_hash': protocol.fingerprint({'session':session,'active_step':active})}


def _clock(clock, previous=None, *, parent_required=True):
    wall, mono = datetime.now(UTC), time.monotonic()
    if (_boot_id() != clock['boot_id'] or mono < clock['monotonic_started']
            or wall < datetime.fromisoformat(clock['wall_started'])
            or (previous and (wall.timestamp() < previous[0] or mono < previous[1]))):
        raise ValueError('Original clock or boot moved backward')
    if parent_required:
        parent = psutil.Process(clock['parent']['pid'])
        if base._identity(parent) != clock['parent'] or parent.status() == psutil.STATUS_ZOMBIE:
            raise ValueError('Original outer issuer disappeared')
    if mono >= clock['monotonic_deadline'] or wall >= datetime.fromisoformat(clock['wall_deadline']):
        raise TimeoutError('Original outer deadline consumed')
    return wall.timestamp(), mono


def _role(scope, record):
    binding = record['binding']
    if scope['role'] == 'S':
        if binding.get('role') != 'S':
            raise ValueError('Forward work cannot borrow an S outer scope')
        expected = protocol.fingerprint({'directory': scope['directory'],
            'replay_month': scope['source_reference']['cutoff'][:7]})
        key='bridge_id'
        if 'research_bridge_id' in binding:
            if set(binding)!={'ticket_terms','research_bridge_id','role','input_id','pipeline_id','serving_eligible'} or binding['serving_eligible'] is not False:
                raise ValueError('Full S research requires its exact typed binding')
            key='research_bridge_id'
        if 'nonlinear_bridge_id' in binding:
            if (set(binding)!={'S_nonlinear_terms','nonlinear_bridge_id','role','kind','plan_hash','input_id','pipeline_id','serving_eligible'}
                    or binding['kind'] not in {'quarterly_probe','resource_trial'} or binding['serving_eligible'] is not False):
                raise ValueError('S nonlinear work requires its exact typed binding')
            key='nonlinear_bridge_id'
        if binding.get(key) != expected:
            raise ValueError('S reservation belongs to another original search run')
    elif binding.get('role') == 'S_typed_queue':
        from factorlab.discovery.structural_queue_review import validate_queue_route

        from factorlab.discovery.neural_forward_ticket import open_existing_authority
        actual = open_existing_authority(scope['authority']['directory'])
        if (actual.manifest_hash != scope['authority']['manifest_hash']
                or binding.get('runtime_directory') != scope['directory']
                or scope['reference']['kind'] != 'campaign'):
            raise ValueError('Typed queue cannot borrow another Campaign/runtime/wallet')
        validate_queue_route(actual, record['work_key'], binding,
                             FrozenWorkLease.from_dict(record['lease']), record['counts'])
    elif binding.get('purpose') == 'policy_trial':
        # A whole-policy trial is nonserving work in its original algorithm
        # claim, never an enum-based permission to borrow a Campaign scope.
        from factorlab.discovery.monthly_partition_policy import _neural_forward
        from factorlab.discovery.neural_forward_ticket import open_existing_authority, read_prepaid_intent

        if scope['role'] != 'forward' or scope['reference']['kind'] != 'campaign':
            raise ValueError('Policy trial requires its original Campaign scope')
        actual = open_existing_authority(scope['authority']['directory'])
        if actual.manifest_hash != scope['authority']['manifest_hash']:
            raise ValueError('Policy trial Campaign wallet changed')
        lease = FrozenWorkLease.from_dict(record['lease'])
        intent = read_prepaid_intent(actual, record['work_key'], binding, lease, record['counts'])
        context = intent['context']
        if (context['purpose'] != 'policy_trial' or context['stream'] != 'A0_policy_trial'
                or context['lifecycle_directory'] != scope['directory']):
            raise ValueError('Policy trial cannot borrow another stream or Campaign runtime')
        # Reconstruct the original lifecycle job, complete recipe, source and
        # causal selection/claim binding through the wallet's canonical route.
        owner = _neural_forward(actual, record['work_key'], binding, lease, record['counts'])
        if owner != {'lane': 'forward', 'owner': {
                'runtime_directory': scope['directory'], 'plan_id': context['plan_id']}}:
            raise ValueError('Policy trial lost its original forward owner')
    elif binding.get('purpose') not in {'pool_forward','known_reference','A0','V','C','resource_trial'} and binding.get('kind') != 'v7-algorithm-validation-fit-v2':
        raise ValueError('Campaign scope cannot authorize S or untyped work')


def _closed_work(record,closure):
    if closure=='completed':return record['status']=='completed'
    if record['status']!='failed_consumed':return False
    binding=record['binding'];reason=record.get('reason','')
    if closure=='local_quota':
        return binding.get('role')=='S' and 'bridge_id' in binding and reason.startswith('local_prepaid_quota:')
    if closure=='closed_nonlinear_gate':
        return (binding.get('role')=='S' and binding.get('serving_eligible') is False
            and (('nonlinear_bridge_id' in binding and reason=='S_nonlinear_preparation:NonlinearGateClosed')
                 or ('research_bridge_id' in binding and reason=='S_preparation:NonlinearGateClosed')))
    if closure=='unavailable_snapshot':
        return (binding.get('purpose') in {'pool_forward','known_reference'} and 'prepare_terms' in binding
                and reason=='forward_preparation:SnapshotUnavailable')
    return False


class IssuerScope:
    """Each operation covers reserve return through actual final publication IO."""
    def __init__(self, reference):
        root = os.environ.get('V7_MULTI_GUARD_DIRECTORY')
        if not root:
            raise ValueError('Dedicated multiwork issuer required')
        self.root = Path(root)
        marker = base._read(self.root/'bootstrap.json')
        me = base._identity(psutil.Process())
        if (marker['issuer'] != me or base._ancestry(psutil.Process(),marker['supervisor']) != marker['ancestry']
                or marker['guard_source'] != base._hash(__file__)):
            raise ValueError('Dedicated guard ownership changed')
        self.scope = _scope(reference,me)
        parent = self.root/'scopes'
        entries=sorted(parent.glob('*'),key=lambda p:int(p.name)) if parent.exists() else []
        if any(not (p/'closed.json').exists() for p in entries):
            raise ValueError('An original scope is still unfinished')
        if len(entries)>=8192:raise ValueError('Bounded supervisor scope capacity consumed')
        self.directory = parent/str(len(entries))
        self.closed, self.active, self.sequence = False, False, 0
        _write(self.directory/'request.json',self.scope)
        self._wait(self.directory/'ready.json', self.scope['clock'])

    def _wait(self,path,clock):
        while not path.exists():
            _clock(clock)
            supervisor=base._read(self.root/'bootstrap.json')['supervisor']
            if base._identity(psutil.Process(supervisor['pid'])) != supervisor:
                os._exit(125)
            time.sleep(.01)
        result=base._read(path)
        if result['scope_hash'] != self.scope['scope_hash']:
            raise ValueError('Acknowledgement changed original scope')
        return result

    @contextmanager
    def installed(self, authority):
        """Scope-local canonical post-reserve hook, before producer IO continues."""
        if self.closed or self.active or type(authority) is not MonthlyWorkAuthority:
            raise ValueError('One actual issuer hook may execute at a time')
        if self.scope != _scope(self.scope['reference'],base._identity(psutil.Process())):
            raise ValueError('Outer context changed before work')
        if self.scope['authority'] != {'directory':str(authority.directory),'manifest_hash':authority.manifest_hash}:
            raise ValueError('Wrong shared wallet')
        if '_reserve' in authority.__dict__:
            raise ValueError('Original authority reservation method was replaced')
        self.active=True
        self.current=None
        self.authority=authority
        original=authority._reserve
        def prepay(*args,**kwargs):
            if self.current is not None or self.sequence>=8192:
                raise ValueError('Prior work must finish without creating another lease')
            # Freeze the sole original clock before canonical reservation IO.
            # The preliminary acknowledgement is containment only; no numeric
            # work proceeds until the actual committed record is reread below.
            if kwargs.get('lease') is None:
                kwargs={**kwargs,'lease':FrozenWorkLease.issue(kwargs['lease_seconds'])}
            lease=kwargs['lease']
            if (type(lease) is not FrozenWorkLease or lease.parent_pid!=os.getpid()
                    or lease.parent_created!=psutil.Process().create_time()):
                raise ValueError('Original reservation belongs to another issuer')
            bound = inspect.signature(original).bind(*args, **kwargs).arguments
            request = _reservation_request({key: value for key, value in bound.items() if key != 'lease'}, lease)
            path=self.directory/'work'/str(self.sequence)
            pending = {'lease':lease.to_dict(),'authority_directory':str(authority.directory),
                'authority_manifest_hash':authority.manifest_hash, 'request': request}
            _write(path/'pending.json', pending)
            watching = self._wait(path/'watching.json',self.scope['clock'])
            if watching != {'scope_hash': self.scope['scope_hash'], 'pending_hash': protocol.fingerprint(pending)}:
                raise ValueError('Guard did not watch the original reservation request')
            try:
                record=original(*args,**kwargs)
            except BudgetExhausted as error:
                if type(error) is not BudgetExhausted:
                    raise
                lease.assert_parent_alive()
                _clock(self.scope['clock'])
                if any((path/name).exists() for name in ('arm.json', 'armed.json', 'done.json', 'released.json')):
                    raise ValueError('Unfunded request conflicts with prepaid work') from error
                refused = {'pending_hash': protocol.fingerprint(pending),
                           'exception_type': 'canonical_wallet_budget_exhausted', 'reason': str(error)}
                _write(path/'unfunded.json', refused)
                acknowledged = self._wait(path/'unfunded_ack.json', self.scope['clock'])
                quota_reason = acknowledged.get('quota_reason')
                quota_reasons = {'Monthly prepaid work quota exhausted'}
                if authority.manifest['partition_policy'] is not None:
                    quota_reasons.update('Nontransferable monthly ' + lane + ' quota exhausted'
                                         for lane in authority.manifest['partition_policy']['lanes'])
                if acknowledged != {'scope_hash': self.scope['scope_hash'],
                        'pending_hash': refused['pending_hash'], 'lease_hash': lease.digest,
                        'closure': 'unfunded_no_reservation', 'quota_reason': quota_reason} or quota_reason not in quota_reasons:
                    raise ValueError('Unfunded acknowledgement changed original request') from error
                lease.assert_parent_alive()
                _clock(self.scope['clock'])
                self.sequence += 1
                raise
            self.current=record
            if record['lease']!=lease.to_dict() or record['status'] not in {'reserved','running'}:
                raise ValueError('Original prepayment differs from the already watched clock')
            _write(path/'arm.json',{'reservation_id':record['reservation_id'],'lease':record['lease'],
                'authority_directory':str(authority.directory),'authority_manifest_hash':authority.manifest_hash})
            self._wait(path/'armed.json',self.scope['clock'])
            lease.assert_parent_alive()
            return record
        authority._reserve=prepay
        try:
            yield
            if self.current is not None:
                raise ValueError('Original work still active when producer returned')
        except BaseException:  # noqa: BLE001 -- no resumed execution after a failed work
            self.closed=True
            os._exit(126)
        finally:
            del authority.__dict__['_reserve']
            self.active=False

    def finish_work(self, expected_id, *, quota_consumed=False, closure=None):
        """Called only AFTER a trusted producer publishes and rechecks its result."""
        if not self.active or self.current is None or self.current['reservation_id']!=expected_id:
            raise ValueError('No matching original work to release')
        current=self.authority.get(expected_id)
        if closure is not None and quota_consumed:raise ValueError('Ambiguous original closure')
        closure=closure or ('local_quota' if quota_consumed else 'completed')
        if (not _closed_work(current,closure) or current['lease']!=self.current['lease']):
            raise ValueError('Only genuine completion or original local quota can release work')
        FrozenWorkLease.from_dict(current['lease']).assert_parent_alive()
        path=self.directory/'work'/str(self.sequence)
        _write(path/'done.json',{'reservation_id':expected_id,'completed_record_hash':protocol.fingerprint(current),
            'closure':closure})
        self._wait(path/'released.json',self.scope['clock'])
        self.sequence+=1;self.current=None

    @contextmanager
    def operation(self, authority):
        with self.installed(authority):
            yield
            if self.current is None:raise ValueError('No original prepaid work was executed')
            self.finish_work(self.current['reservation_id'])

    def close(self):
        if self.closed or self.active or (not self.sequence and self.scope['role']!='forward'):
            raise ValueError('Original scope cannot close with unfinished work')
        if self.scope != _scope(self.scope['reference'],base._identity(psutil.Process())):
            raise ValueError('Original scope changed before close')
        _write(self.directory/'done.json',{'scope_hash':self.scope['scope_hash'],'works':self.sequence})
        self._wait(self.directory/'closed.json',self.scope['clock'])
        self.closed=True


def supervise(script, arguments, *, directory, source_root, _ownership=None, _actor=None):
    directory,script=Path(directory).resolve(),Path(script).resolve()
    borrowed=_ownership is not None
    if borrowed != (_actor is not None):raise ValueError('Exact shared ownership and actor required')
    if directory.exists() and not borrowed: raise ValueError('Guard attempts cannot reset')
    directory.mkdir(parents=True,exist_ok=borrowed)
    request={'script':str(script),'arguments':list(arguments),'script_hash':base._hash(script),
        'source_root':str(Path(source_root).resolve()),'guard_source':base._hash(__file__),
        'base_source':base._hash(base.__file__),
        'adapter_source':base._hash(Path(__file__).with_name('guarded_runtime.py'))}
    _write(directory/'request.json',request)
    boundary=_ownership if borrowed else (base._WindowsJob() if os.name=='nt' else base._LinuxService())
    worker=psutil.Process(_actor['pid']) if borrowed else None; active=None; scope=None; record=None; authority=None; sequence=0
    active_pending_hash=None
    scope_index=0; idle_clock=None; seen_scopes=set(); total_works=0; scope_ready=False
    stop=threading.Event(); closing=threading.Event(); failed=[]; lock=threading.Lock()
    start=time.monotonic(); status='failed'; reason='startup'; observation=None
    def alive():
        if borrowed:
            return worker.is_running() and base._identity(worker)==_actor and worker.status()!=psutil.STATUS_ZOMBIE
        return worker.poll() is None
    def watch():
        prior=None
        try:
            while not stop.is_set():
                with lock: current,work,idle=scope,active,idle_clock
                if not scope_ready and idle is None and time.monotonic()-start>60:
                    raise TimeoutError('Original scope was not authenticated during startup')
                if current is None:
                    if idle is not None:
                        prior=_clock(idle,prior,parent_required=not closing.is_set())
                    elif time.monotonic()-start>60: raise TimeoutError('No trusted original outer scope')
                else:
                    prior=_clock(current['clock'],prior,parent_required=not closing.is_set())
                    if work is not None:
                        check,_=base._lease_check(work,psutil.Process(work.parent_pid),None)
                        if not check.allowed: raise TimeoutError('Original work '+check.reason)
                stop.wait(.01)
        except BaseException as error:  # noqa: BLE001 -- enforce termination even after native/IO failure
            failed.append(type(error).__name__+':'+str(error))
            try:
                boundary.stop()
            except BaseException:  # noqa: BLE001 -- guard death closes actual kernel ownership
                os._exit(125)
    thread=threading.Thread(target=watch,daemon=True)
    try:
        env=dict(os.environ,V7_MULTI_GUARD_DIRECTORY=str(directory))
        with (nullcontext(None) if borrowed else (directory/'worker.log').open('xb')) as log:
            if borrowed:
                hello=dict(_actor);issuer=worker
                if not alive() or issuer.pid not in boundary.members():raise ValueError('Unowned registered actor')
            else:
                worker=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'bootstrap',str(directory)],
                    cwd=base.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)|(4 if os.name=='nt' else 0))
                boundary.attach(worker)
                while not (directory/'hello.json').exists():
                    if not alive():raise RuntimeError('Issuer died during bootstrap')
                    if time.monotonic()-start>60:raise TimeoutError('Unidentified issuer startup')
                    time.sleep(.01)
                hello=base._read(directory/'hello.json');issuer=psutil.Process(hello['pid'])
                if hello!=base._identity(issuer) or issuer.pid not in boundary.members():raise ValueError('Unowned actual issuer')
            thread.start()
            supervisor=base._identity(psutil.Process())
            _write(directory/'bootstrap.json',{'issuer':hello,'supervisor':supervisor,
                'ancestry':base._ancestry(issuer,supervisor),'guard_source':base._hash(__file__)})
            protocol.ROOT=Path(request['source_root'])
            while alive() and not failed:
                root=directory/'scopes'/str(scope_index)
                if scope is None and (root/'request.json').exists():
                    proposed=base._read(root/'request.json')
                    with lock:scope=proposed  # Tighten clock BEFORE slow canonical reread.
                    if proposed!=_scope(proposed['reference'],hello):raise ValueError('Outer request differs from canonical source')
                    key=protocol.fingerprint(proposed['reference'])
                    if key in seen_scopes:raise ValueError('An original outer scope cannot be reopened')
                    seen_scopes.add(key)
                    scope_ready=True
                    _write(root/'ready.json',{'scope_hash':scope['scope_hash']})
                if scope is not None:
                    path=root/'work'/str(sequence)
                    if active is None and (path/'pending.json').exists():
                        pending=base._read(path/'pending.json');lease=FrozenWorkLease.from_dict(pending['lease'])
                        if (set(pending) != {'lease', 'authority_directory', 'authority_manifest_hash', 'request'}
                                or _reservation_request(pending['request'], lease) != pending['request']
                                or lease.parent_pid!=issuer.pid or lease.parent_created!=issuer.create_time()
                                or {'directory':pending['authority_directory'],'manifest_hash':pending['authority_manifest_hash']}!=scope['authority']):
                            raise ValueError('Pre-reserve clock belongs to another original issuer/wallet')
                        with lock:active=lease
                        active_pending_hash=protocol.fingerprint(pending)
                        _write(path/'watching.json',{'scope_hash':scope['scope_hash'],
                            'pending_hash':active_pending_hash})
                    if active is not None and (path/'unfunded.json').exists():
                        refused = base._read(path/'unfunded.json')
                        pending = base._read(path/'pending.json')
                        if (set(refused) != {'pending_hash', 'exception_type', 'reason'}
                                or refused['pending_hash'] != protocol.fingerprint(pending)
                                or refused['pending_hash'] != active_pending_hash
                                or refused['exception_type'] != 'canonical_wallet_budget_exhausted'
                                or pending['lease'] != active.to_dict()
                                or any((path/name).exists() for name in
                                       ('arm.json', 'armed.json', 'done.json', 'released.json'))):
                            raise ValueError('Unfunded closure conflicts with original pending work')
                        quota_reason = _verify_unfunded(scope, pending, refused['reason'], issuer)
                        if (base._read(path/'pending.json') != pending
                                or base._read(path/'unfunded.json') != refused
                                or any((path/name).exists() for name in
                                       ('arm.json', 'armed.json', 'done.json', 'released.json'))):
                            raise ValueError('Unfunded closure changed during canonical verification')
                        _write(path/'unfunded_ack.json', {'scope_hash': scope['scope_hash'],
                            'pending_hash': refused['pending_hash'], 'lease_hash': active.digest,
                            'closure': 'unfunded_no_reservation', 'quota_reason': quota_reason})
                        _clock(scope['clock'])
                        check, observation = base._lease_check(active, issuer, None)
                        if not check.allowed:
                            raise TimeoutError('Unfunded acknowledgement exceeded original lease')
                        with lock:active=None
                        active_pending_hash=None
                        sequence += 1
                    if active is not None and not (path/'armed.json').exists() and (path/'arm.json').exists():
                        arm=base._read(path/'arm.json');lease=active
                        authority=base._authority(arm['authority_directory']);record=authority.get(arm['reservation_id'])
                        if (record['lease']!=arm['lease'] or record['lease']!=active.to_dict() or record['status'] not in {'reserved','running'}
                                or scope['authority']!={'directory':str(authority.directory),'manifest_hash':authority.manifest_hash}
                                or arm['authority_manifest_hash']!=authority.manifest_hash):raise ValueError('Original wallet differs')
                        pending = base._read(path/'pending.json')
                        if (protocol.fingerprint(pending) != active_pending_hash
                                or pending['lease'] != active.to_dict()
                                or any(record[key] != pending['request'][key] for key in ('work_key', 'counts', 'binding'))
                                or (path/'unfunded.json').exists() or (path/'unfunded_ack.json').exists()):
                            raise ValueError('Prepayment changed its original request or conflicts with refusal')
                        _role(scope,record)
                        # Both original clocks stay watched. The effective
                        # deadline is their minimum; never rewrite a signed
                        # 300s work lease merely because the step ends sooner.
                        if scope!=_scope(scope['reference'],hello):raise ValueError('Outer authorization changed')
                        _write(path/'armed.json',{'scope_hash':scope['scope_hash'],'lease_hash':lease.digest})
                    if active is not None and (path/'done.json').exists():
                        current=authority.get(record['reservation_id']);done=base._read(path/'done.json')
                        closure=done.get('closure')
                        if (not _closed_work(current,closure) or current['lease']!=active.to_dict()
                                or done!={'reservation_id':record['reservation_id'],'completed_record_hash':protocol.fingerprint(current),'closure':closure}):
                            raise ValueError('Original work is not genuinely completed')
                        check,observation=base._lease_check(active,issuer,None)
                        if not check.allowed:raise TimeoutError('Late work completion')
                        # Only known released work leaves the watchdog set. The
                        # immutable outer remains active across all work gaps.
                        with lock:active=None
                        active_pending_hash=None
                        _write(path/'released.json',{'scope_hash':scope['scope_hash']})
                        sequence+=1
                    if (root/'done.json').exists():
                        if active is not None or base._read(root/'done.json')!={'scope_hash':scope['scope_hash'],'works':sequence}:
                            raise ValueError('Scope closed with incomplete original work')
                        if scope!=_scope(scope['reference'],hello):raise ValueError('Scope authority changed')
                        _clock(scope['clock'])
                        acknowledged={'scope_hash':scope['scope_hash']}
                        with lock:
                            idle_clock=scope.get('idle_clock',scope['clock'])
                            scope=None
                        _write(root/'closed.json',acknowledged)
                        total_works+=sequence;sequence=0;scope_index+=1
                if (directory/'finished.json').exists():
                    if scope is not None or not scope_index or base._read(directory/'finished.json')!={'scopes':scope_index}:
                        raise ValueError('Dedicated issuer cannot finish with active work')
                    _clock(idle_clock);closing.set()
                    _write(directory/'finished_ack.json',{'scopes':scope_index})
                    worker.wait(timeout=min(2,idle_clock['monotonic_deadline']-time.monotonic()))
                    # A non-parent guardian cannot know exit status. Original
                    # actual scheduler owns that result; this only confirms exit.
                    if not borrowed and worker.returncode:raise RuntimeError('Issuer failed after scope completion')
                    status,reason='completed','original_scopes_completed';break
                time.sleep(.01)
            if failed:raise RuntimeError(failed[0])
            if status!='completed':raise RuntimeError('Issuer disappeared without completed scope')
    except BaseException as error:  # noqa: BLE001 -- mandatory owned-tree cleanup
        reason=type(error).__name__+':'+str(error)
    finally:
        stop.set()
        if thread.is_alive():thread.join(timeout=1)
        # No failure-log or accounting write is allowed to precede teardown.
        try:
            if status!='completed' or (not borrowed and (set(boundary.members())-({os.getpid()} if os.name!='nt' else set()))):boundary.stop()
        finally:
            if not borrowed:boundary.close()
        if authority is not None and record is not None and status!='completed':
            try:
                if authority.get(record['reservation_id'])['status'] in {'reserved','running'}:
                    authority.fail_consumed(record['reservation_id'],'multiwork_guard:'+reason)
            except BaseException as error:  # noqa: BLE001 -- annotation cannot undo prepayment
                reason += ';annotation_failed='+type(error).__name__
        result={'status':status,'reason':reason,'released_works':total_works+sequence,'completed_scopes':scope_index,'scope':scope,
            'observation':observation,'admission_authority':False}
        _write(directory/'result.json',result)
    return result


def finish():
    root=Path(os.environ['V7_MULTI_GUARD_DIRECTORY'])
    entries=list((root/'scopes').glob('*'))
    if not entries or any(not (p/'closed.json').exists() for p in entries):
        raise ValueError('Cannot finish with unclosed original scopes')
    _write(root/'finished.json',{'scopes':len(entries)})
    while not (root/'finished_ack.json').exists():
        owner=base._read(root/'bootstrap.json')['supervisor']
        if base._identity(psutil.Process(owner['pid']))!=owner:os._exit(125)
        time.sleep(.01)


def _bootstrap(directory):
    directory=Path(directory)
    if os.name!='nt':
        parent=os.getppid()
        if ctypes.CDLL(None).prctl(1,9,0,0,0)!=0 or os.getppid()!=parent:os._exit(125)
    _write(directory/'hello.json',base._identity(psutil.Process()))
    while not (directory/'bootstrap.json').exists():time.sleep(.01)
    request=base._read(directory/'request.json')
    if (request['guard_source']!=base._hash(__file__) or request['base_source']!=base._hash(base.__file__)
            or request['script_hash']!=base._hash(request['script'])
            or request['adapter_source']!=base._hash(Path(__file__).with_name('guarded_runtime.py'))):raise ValueError('Frozen issuer source changed')
    protocol.ROOT=Path(request['source_root'])
    sys.argv=[request['script'],*request['arguments']]
    runpy.run_path(request['script'],run_name='__main__')


if __name__=='__main__':
    if sys.argv[1]=='bootstrap':_bootstrap(sys.argv[2])
    elif sys.argv[1]=='supervise':
        result=supervise(**base._read(sys.argv[2]));sys.exit(0 if result['status']=='completed' else 124)
    else:raise ValueError('Fixed dedicated entry required')
