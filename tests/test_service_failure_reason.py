"""Artificial scope observations test classification, never elapsed-time evidence."""
import copy
import json
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from threading import get_ident
from types import SimpleNamespace

import pytest

from factorlab.discovery import forward_service_scope as module
from factorlab.discovery.work_lease import FrozenWorkLease, LeaseCheck, LeaseClosed


@pytest.fixture
def scope(tmp_path, monkeypatch):
    from factorlab import protocol
    monkeypatch.setattr(protocol, 'ROOT', tmp_path)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    origin = FrozenWorkLease(300, start.isoformat(), (start+timedelta(seconds=300)).isoformat(),
                             100., 400., 'linux:12345678-1234-1234-1234-123456789abc', os.getpid(), 1000.)
    s = module._Scope.__new__(module._Scope)
    s.origin, s.pid, s.thread = origin, os.getpid(), get_ident()
    s.runtime, s.identity, s.clock = tmp_path/'runtime', {'identity': 'original'}, {'issuer': origin.to_dict()}
    s.spent, s.previous, s.rejection = 20., None, None
    s.ceiling = SimpleNamespace(**{**origin.to_dict(),
        'wall_deadline': (start+timedelta(seconds=580)).isoformat(), 'monotonic_deadline': 680.})
    events = []
    s.service = SimpleNamespace(directory=tmp_path, runtime=SimpleNamespace(directory=s.runtime),
        config=SimpleNamespace(wall_seconds=600), state={'spent_seconds':20.,
        'native_service_clock':s.clock, 'last_result':{'status':'stepped'}, 'fit_attempts':{'paid':1}},
        _save=lambda event: events.append((event, copy.deepcopy(s.service.state))))
    (tmp_path/'identity.json').write_text(json.dumps(s.identity), encoding='utf-8')
    (tmp_path/'native-clock.json').write_text(json.dumps(s.clock), encoding='utf-8')
    s.events = events
    monkeypatch.setattr(FrozenWorkLease, 'issue', lambda *a, **k: pytest.fail('Never issue a renewed lease'))
    return s


def observation(scope, elapsed=10., **changes):
    return {'wall':datetime.fromisoformat(scope.origin.wall_started_at).timestamp()+elapsed,
            'monotonic':scope.origin.monotonic_started+elapsed, 'boot_id':scope.origin.boot_id,
            'pid':scope.pid, 'ppid':0, 'parent_pid':scope.origin.parent_pid,
            'parent_created':scope.origin.parent_created, 'parent_alive':True, **changes}


def rejected_observation(scope, reason):
    if reason=='identity_unavailable': return {'unavailable':True}
    return observation(scope, **{
        'boot_changed':{'boot_id':'linux:87654321-4321-4321-4321-cba987654321'},
        'parent_lost':{'parent_alive':False},
        'wrong_process':{'pid':scope.pid+1, 'ppid':scope.pid+2},
        'clock_rollback':{'monotonic':99.},
        'clock_invalid':{'wall':float('nan')},
        'unknown_reason':{},
    }[reason])


def assert_consumed(scope, reason, *, clock=None):
    assert scope.service.state['spent_seconds']>=600
    assert scope.service.state['last_result']=={'status':'budget_exhausted','reason':reason}
    assert scope.events[-1][1]==scope.service.state
    assert scope.service.state['fit_attempts']=={'paid':1}
    assert json.loads((scope.service.directory/'native-clock.json').read_text())==(scope.clock if clock is None else clock)
    assert scope.ceiling.monotonic_deadline==680.
    assert scope.origin.monotonic_started==100.


@pytest.mark.parametrize('reason', ['boot_changed','parent_lost','identity_unavailable',
                                  'wrong_process','clock_rollback','clock_invalid','unknown_reason'])
@pytest.mark.parametrize('boundary', ['check','finish','after_save'])
def test_nondeadline_reasons_survive_and_consume_without_renewal(scope, monkeypatch, reason, boundary):
    bad = rejected_observation(scope, reason)
    good = observation(scope)
    unknown = LeaseCheck(reason)
    if reason=='unknown_reason':
        evaluate = module._evaluate
        monkeypatch.setattr(module, '_evaluate', lambda lease, sample, previous=None:
            unknown if sample is bad else evaluate(lease, sample, previous))
    samples = iter([good, bad] if boundary=='after_save' else [bad, good, good])
    monkeypatch.setattr(module, '_observe', lambda ceiling: next(samples))
    if boundary=='check':
        with pytest.raises(LeaseClosed) as first:
            scope.check(scope.service.runtime)
        assert first.value.check.reason==reason
        # A later healthy sample must not erase an already observed rejection.
    with pytest.raises(LeaseClosed) as rejected:
        scope.finish()
    assert rejected.value.check.reason==reason
    if reason=='unknown_reason': assert rejected.value.check is unknown
    assert_consumed(scope, reason)
    assert scope.events[-1][0]=='native_service_scope_closed'
    if boundary=='after_save': assert scope.events[0][0]=='native_service_time_observed'


@pytest.mark.parametrize('clock', ['wall','monotonic','both'])
@pytest.mark.parametrize('boundary', ['check','finish','after_save'])
def test_real_dual_clock_deadline_keeps_original_type_and_messages(scope, monkeypatch, clock, boundary):
    late = observation(scope)
    if clock in ('wall','both'): late['wall']+=570.
    if clock in ('monotonic','both'): late['monotonic']+=570.
    samples = iter([observation(scope),late] if boundary=='after_save' else [late,late,late])
    monkeypatch.setattr(module, '_observe', lambda ceiling: next(samples))
    if boundary=='check':
        with pytest.raises(module.ServiceDeadlineClosed) as rejected:
            scope.check(scope.service.runtime)
        assert str(rejected.value)=='Original service deadline has insufficient full-work headroom'
    with pytest.raises(module.ServiceDeadlineClosed) as rejected:
        scope.finish()
    assert str(rejected.value)=='Original native service deadline consumed; no new session issuer'
    assert_consumed(scope, 'original_native_service_deadline_consumed')
    assert scope.events[-1][0]=='native_service_deadline_consumed'


@pytest.mark.parametrize('damage', ['runtime','identity','clock','state','refund','thread',
                                  'lease_type','lease_seconds','lease_parent','lease_boot'])
def test_structural_rejection_is_value_error_and_late_cleanup_preserves_it(scope, monkeypatch, damage):
    samples = iter([observation(scope), observation(scope,580.), observation(scope,580.)]
                   if damage.startswith('lease_') else [observation(scope,580.), observation(scope,580.)])
    monkeypatch.setattr(module, '_observe', lambda ceiling: next(samples))
    runtime, lease = scope.service.runtime, None
    if damage=='runtime': runtime=SimpleNamespace(directory=scope.runtime/'foreign')
    elif damage=='identity': (scope.service.directory/'identity.json').write_text('{}')
    elif damage=='clock': (scope.service.directory/'native-clock.json').write_text('{}')
    elif damage=='state': scope.service.state['native_service_clock']={}
    elif damage=='refund': scope.service.state['spent_seconds']=19.
    elif damage=='thread': scope.thread=-1
    elif damage=='lease_type': lease=object()
    elif damage=='lease_seconds':
        lease=replace(scope.origin, lease_seconds=299, monotonic_deadline=399.,
                      wall_deadline=(datetime.fromisoformat(scope.origin.wall_started_at)+timedelta(seconds=299)).isoformat())
    elif damage=='lease_parent': lease=replace(scope.origin, parent_created=1001.)
    elif damage=='lease_boot': lease=replace(scope.origin, boot_id='linux:87654321-4321-4321-4321-cba987654321')
    original_check=scope.check
    monkeypatch.setattr(scope, 'check', lambda ignored: original_check(runtime, lease=lease))
    monkeypatch.setattr(module, '_Scope', lambda *args:scope)
    with pytest.raises(ValueError) as rejected, module._service_step_scope(scope.service):
        pytest.fail('Invalid scope entered')
    assert type(rejected.value) is ValueError
    assert ('300s contract' if damage.startswith('lease_') else 'scope identity changed') in str(rejected.value)
    assert type(rejected.value.__context__) is module.ServiceDeadlineClosed
    assert rejected.value.__context__.__context__ is None
    assert module._ACTIVE.get() is None
    assert_consumed(scope, 'original_native_service_deadline_consumed', clock={} if damage=='clock' else None)


def test_body_error_and_save_error_both_remain_visible(scope, monkeypatch):
    monkeypatch.setattr(module, '_observe', lambda ceiling:observation(scope,580.))
    monkeypatch.setattr(scope, 'check', lambda *args:None)
    monkeypatch.setattr(module, '_Scope', lambda *args:scope)
    failure=OSError('disclosed durable save failure')
    monkeypatch.setattr(scope.service, '_save', lambda event:(_ for _ in ()).throw(failure))
    original=ValueError('disclosed original body failure')
    with pytest.raises(ValueError) as rejected, module._service_step_scope(scope.service):
        raise original
    assert rejected.value is original and original.__context__ is failure
    assert scope.service.state['spent_seconds']>=600
    assert module._ACTIVE.get() is None


def test_original_and_cleanup_nested_contexts_remain_complete(scope, monkeypatch):
    monkeypatch.setattr(scope, 'check', lambda *args:None)
    monkeypatch.setattr(module, '_Scope', lambda *args:scope)
    oldest, original = KeyError('original cause'), ValueError('body failed')
    closing_inner, closing = RuntimeError('closing cause'), OSError('closing failed')
    def finish():
        try: raise closing_inner
        except RuntimeError: raise closing
    monkeypatch.setattr(scope, 'finish', finish)
    with pytest.raises(ValueError) as rejected, module._service_step_scope(scope.service):
        try: raise oldest
        except KeyError: raise original
    assert rejected.value is original
    assert original.__context__ is closing
    assert closing.__context__ is closing_inner
    assert closing_inner.__context__ is oldest
    assert oldest.__context__ is None
    assert module._ACTIVE.get() is None


def test_prior_sample_rollback_is_preserved_after_durable_save(scope, monkeypatch):
    samples=iter([observation(scope,10.),observation(scope,9.)])
    monkeypatch.setattr(module, '_observe', lambda ceiling:next(samples))
    with pytest.raises(LeaseClosed) as rejected: scope.finish()
    assert rejected.value.check.reason=='clock_rollback'
    assert_consumed(scope, 'clock_rollback')


def test_nondeadline_root_survives_later_physical_deadline(scope, monkeypatch):
    samples=iter([rejected_observation(scope,'boot_changed'),observation(scope,580.),observation(scope,580.)])
    monkeypatch.setattr(module, '_observe', lambda ceiling:next(samples))
    monkeypatch.setattr(module, '_Scope', lambda *args:scope)
    with pytest.raises(LeaseClosed) as rejected, module._service_step_scope(scope.service):
        pytest.fail('Closed scope entered')
    assert rejected.value.check.reason=='boot_changed'
    assert type(rejected.value.__context__) is LeaseClosed
    assert rejected.value.__context__.check.reason=='boot_changed'
    assert rejected.value.__context__.__context__ is None
    assert_consumed(scope, 'boot_changed')


def test_lease_deadline_overrun_and_insufficient_headroom_still_reject(scope, monkeypatch):
    monkeypatch.setattr(module, '_observe', lambda ceiling:observation(scope,300.))
    with pytest.raises(module.ServiceDeadlineClosed,match='insufficient full-work headroom'):
        scope.check(scope.service.runtime, seconds=302)
    later=replace(scope.origin, monotonic_started=400., monotonic_deadline=700.,
        wall_started_at=(datetime.fromisoformat(scope.origin.wall_started_at)+timedelta(seconds=300)).isoformat(),
        wall_deadline=(datetime.fromisoformat(scope.origin.wall_started_at)+timedelta(seconds=600)).isoformat())
    with pytest.raises(module.ServiceDeadlineClosed,match='New unchanged 300s lease exceeds original service deadline'):
        scope.check(scope.service.runtime,lease=later)
    scope.finish()
    assert scope.service.state['spent_seconds']==320.
    assert scope.service.state['last_result']=={'status':'stepped'}
    assert scope.rejection is None and scope.ceiling.monotonic_deadline==680.


def test_healthy_scope_keeps_original_clock_and_monotone_spend(scope, monkeypatch):
    samples=iter([observation(scope,10.),observation(scope,12.),observation(scope,13.)])
    monkeypatch.setattr(module, '_observe', lambda ceiling:next(samples))
    scope.check(scope.service.runtime, seconds=302, lease=scope.origin)
    scope.service.state['spent_seconds']=80.
    scope.finish()
    assert scope.service.state['spent_seconds']==80.
    assert scope.events[-1][0]=='native_service_time_observed'
    assert scope.service.state['native_service_clock']==scope.clock
    assert scope.rejection is None and scope.ceiling.monotonic_deadline==680.
