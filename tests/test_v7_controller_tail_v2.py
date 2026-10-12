"""Real clocks/PIDs, dual immutable files and original lifecycle journal.

Registration and numerical/provider dispatch are explicit no-work boundary
fixtures. These tests establish controller timing/accounting, not market,
provider identity, model acceptance, or the formal statistical certificate.
"""
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from factorlab.discovery.full_trial_controller import FullTrialController
from factorlab.discovery.v7_lifecycle import Lifecycle, LifecyclePlan


@pytest.fixture(autouse=True)
def restore_project_root(monkeypatch):
    from factorlab import protocol
    monkeypatch.setattr(protocol, 'ROOT', protocol.ROOT)


def boundary(root, seconds=2., *, create=True):
    from factorlab import protocol
    protocol.ROOT = root
    runtime = root/'trial/runtime'
    if create:
        runtime.mkdir(parents=True)
        lc = Lifecycle(runtime, LifecyclePlan(source_id="artificial-controller-clock-fixture"))
        lc.close()
    obj = FullTrialController.__new__(FullTrialController)
    obj.directory = root/'trial/controller'
    obj.directory.mkdir(parents=True, exist_ok=True)
    obj.permit = SimpleNamespace(registry=SimpleNamespace(directory=root),
        paths={'runtime': runtime}, trial={'trial_id': 'original'})
    obj.identity = {'engineering_registration_boundary': True}
    obj.schedule = SimpleNamespace(due_at=('2020-01-01T00:00:00+00:00', '2020-01-02T00:00:00+00:00'),
        observation_mode='physical_UTC', service={'wall_seconds': seconds})
    obj.verify = obj._steps
    obj.pending_dependencies = lambda: ()
    obj._dispatch = lambda: {'forward': {'status': 'stepped'}, 'no_provider_or_fit': True}
    return obj


def test_pending_never_credited_successor_only_and_run_final_drain(tmp_path):
    obj = boundary(tmp_path, 5.)
    calls=[]
    obj._dispatch=lambda: calls.append(os.getpid()) or {'forward': {'status': 'stepped'}}
    first = obj.step(0)
    assert first['state'] == 'awaiting_successor_witness'
    assert not obj.step_completed(obj.verify()[0])
    assert obj.step(1)['state'] == 'predecessor_unconfirmed_or_consumed'
    assert obj.pump()['state'] == 'completed_step'
    assert calls == [os.getpid()]
    result = obj.run(max_pumps=1)
    assert [r['state'] for r in result['progress']] == ['awaiting_successor_witness', 'completed_step']
    rows = obj.verify()
    assert len(rows) == 2 and all(obj.step_completed(r) for r in rows)
    assert rows[0]['attempt']['lease'] == rows[1]['attempt']['lease']
    assert obj.step(0) == first and len(calls) == 2


def test_actual_result_pair_tail_is_pending_then_consumed(tmp_path):
    obj = boundary(tmp_path, 1.)
    original = obj._write_pair
    def slow(index, phase, value):
        original(index, phase, value)
        if phase == 'result':
            time.sleep(1.1)
    obj._write_pair = slow
    started=time.monotonic()
    result=obj.step(0)
    assert time.monotonic()-started > 1.
    assert result['state']=='awaiting_successor_witness'
    assert obj.pump()['state']=='failed_or_unknown_consumed'
    assert not obj.step_completed(obj.verify()[0])
    assert obj.pump()['state']=='interrupted_consumed'
    assert len(obj.verify())==1


def test_clock_initial_commit_tail_is_charged_before_validation(tmp_path):
    obj=boundary(tmp_path,1.)
    original=obj._write_pair
    def slow(index,phase,value):
        original(index,phase,value)
        if phase=='clock': time.sleep(1.1)
    obj._write_pair=slow
    obj._dispatch=lambda: pytest.fail('expired initial IO must not dispatch')
    result=obj.pump()
    assert result['state']=='controller_service_closed'
    assert obj.verify()==[]
    original_clock=obj._execution_clock()
    assert obj.pump()==result and obj._execution_clock()==original_clock


@pytest.mark.parametrize('entry',['step','pump'])
@pytest.mark.parametrize('slow_boundary',['clock_lookup','lock_acquisition'])
def test_first_lookup_and_lock_cost_precede_no_new_budget(tmp_path,monkeypatch,entry,slow_boundary):
    obj=boundary(tmp_path,1.)
    delayed=[]
    if slow_boundary=='clock_lookup':
        original=obj._execution_clock
        def lookup():
            if not delayed:
                delayed.append(True);time.sleep(1.1)
            return original()
        monkeypatch.setattr(obj,'_execution_clock',lookup)
    else:
        from filelock import FileLock
        original=FileLock.__enter__
        def acquire(lock):
            if not delayed:
                delayed.append(True);time.sleep(1.1)
            return original(lock)
        monkeypatch.setattr(FileLock,'__enter__',acquire)
    obj._dispatch=lambda:pytest.fail('initial lookup/lock IO consumed original clock')
    result=obj.step(0) if entry=='step' else obj.pump()
    assert result['state']=='controller_service_closed' and obj.verify()==[]
    assert obj._execution_clock()['lease']['service_seconds']==1.


def test_existing_clock_ignores_later_real_candidate_origins(tmp_path,monkeypatch):
    from factorlab.discovery.full_trial_controller import ControllerStepLease
    original=ControllerStepLease.issue.__func__;samples=[]
    def issue(cls,schedule):
        sample=original(cls,schedule);samples.append(sample.to_dict());return sample
    monkeypatch.setattr(ControllerStepLease,'issue',classmethod(issue))
    obj=boundary(tmp_path,1.)
    obj.run(max_pumps=1)
    frozen=obj._execution_clock()
    time.sleep(1.05)
    assert obj.pump()['state']=='controller_service_closed'
    assert len(samples)>=3 and samples[-1]['issuer']['monotonic_started']>samples[0]['issuer']['monotonic_started']
    assert obj._execution_clock()==frozen and frozen['lease']==samples[0]


def test_continuous_clock_charges_idle_and_prior_return_tail(tmp_path):
    obj=boundary(tmp_path,1.)
    assert obj.run(max_pumps=1)['state']=='completed_step'
    original=obj._execution_clock()
    time.sleep(1.05)
    obj._dispatch=lambda: pytest.fail('must not renew next opportunity')
    assert obj.pump()['state']=='controller_service_closed'
    assert obj._execution_clock()==original and len(obj.verify())==1


def test_orphan_attempt_never_replayed(tmp_path):
    obj=boundary(tmp_path,5.)
    original=obj._write_pair
    def crash(index,phase,value):
        original(index,phase,value)
        if phase=='attempt': raise KeyboardInterrupt('after durable reservation')
    obj._write_pair=crash
    with pytest.raises(KeyboardInterrupt): obj.step(0)
    obj._write_pair=original
    obj._dispatch=lambda: pytest.fail('orphan cannot replay')
    assert obj.pump()['state']=='interrupted_consumed'
    assert obj.pump()['state']=='interrupted_consumed'
    assert len(obj.verify())==1


def test_missing_clock_cannot_renew_original_lifecycle_anchor(tmp_path):
    obj=boundary(tmp_path,5.)
    obj.step(0)
    for p in obj._paths(-1,'clock'): p.rename(p.with_suffix('.retained'))
    obj._dispatch=lambda: pytest.fail('clock deletion cannot grant replacement time')
    with pytest.raises(ValueError,match='clock anchor exists'): obj.pump()


def test_lost_witness_pair_cannot_be_reconfirmed(tmp_path):
    obj=boundary(tmp_path,5.)
    obj.run(max_pumps=1)
    for p in obj._paths(0,'witness'): p.rename(p.with_suffix('.retained'))
    with pytest.raises(ValueError,match='journal anchors'): obj.pump()


def test_expired_dispatch_tail_is_failed_and_not_credited(tmp_path):
    obj=boundary(tmp_path,1.)
    def close_tail():
        # Real SQLite commit/close followed by a slow return. The real dispatch
        # has this same finally/close boundary; no numerical work is simulated.
        with sqlite3.connect(tmp_path/'close.sqlite') as db:
            db.execute('CREATE TABLE original (value TEXT)')
            db.execute('INSERT INTO original VALUES (?)',('paid original close',))
        time.sleep(1.1)
        return {'forward': {'status':'stepped'}}
    obj._dispatch=close_tail
    from factorlab.discovery.work_lease import LeaseClosed
    with pytest.raises(LeaseClosed): obj.step(0)
    rows=obj.verify()
    assert rows[0]['result']['state']=='failed_consumed' and not obj.step_completed(rows[0])
    assert obj.pump()['state']=='interrupted_consumed'


def test_dead_original_issuer_cannot_recover_new_session(tmp_path):
    # The child issues and durably anchors its actual clock, then exits. The
    # parent must neither act as that issuer nor allocate a replacement epoch.
    code=('import sys; '
          f'sys.path.insert(0,{str(Path(__file__).resolve().parent)!r}); '
          'from pathlib import Path; from test_v7_controller_tail_v2 import boundary; '
          'from filelock import FileLock; '
          f'o=boundary(Path({str(tmp_path)!r}),30.); '
          'o._start_execution()')
    child=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,
                         creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),
                         cwd=Path(sys.modules[FullTrialController.__module__].__file__).resolve().parents[2])
    assert child.returncode==0,child.stderr
    obj=boundary(tmp_path,30.,create=False)
    clock=obj._execution_clock()
    assert clock['lease']['issuer']['parent_pid']!=os.getpid()
    obj._dispatch=lambda: pytest.fail('dead issuer cannot be replaced')
    assert obj.pump()['state']=='controller_service_closed'
    assert obj._execution_clock()==clock and obj.verify()==[]


@pytest.mark.parametrize('value',[0,7201,True,1.5])
def test_run_preserves_frozen_invocation_bounds(tmp_path,value):
    obj=boundary(tmp_path)
    with pytest.raises(ValueError):obj.run(max_pumps=value)
    assert obj._execution_clock() is None

