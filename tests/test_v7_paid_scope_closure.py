"""Actual lifecycle locks and canonical paid tickets on artificial receipts.

No provider/model training/Linux admission is asserted. The 300-second test
first proves timely ticket issuance including its own full closing audit, then
delays the first unscoped verification audit until that real lease expires.
It does not attribute that delayed call specifically to the closing callback;
the companion one-second diagnostic isolates that exact callback separately.
"""
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from filelock import Timeout

from factorlab.discovery import quarantined_observation as observation
from factorlab.discovery.v7_lifecycle import Lifecycle, LifecyclePlan
from factorlab.discovery.neural_forward_ticket import _LifecycleReader
from test_v7_quarantined_observation import factory


@pytest.fixture
def tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp('p')


def test_same_instance_thread_reentry_and_other_instances_processes_excluded(tmp_path,monkeypatch):
    from factorlab import protocol
    monkeypatch.setattr(protocol,'ROOT',tmp_path)
    lc=Lifecycle(tmp_path/'lc',LifecyclePlan(source_id='explicit-artificial-lock-test'))
    other=Lifecycle(lc.directory,lc.plan)
    try:
        lock=lc.controller_lock()
        with lock:
            with lc.controller_lock():
                assert lc.controller_lock() is lock
                lc.verify()
            with pytest.raises(Timeout):
                with other.controller_lock():pass
            def competing_thread():
                try:
                    with lc.controller_lock():return 'incorrectly-acquired'
                except Timeout:return 'blocked'
            with ThreadPoolExecutor(max_workers=1) as pool:
                assert pool.submit(competing_thread).result(timeout=5)=='blocked'
            code=('from pathlib import Path;from filelock import Timeout;'
                  'from factorlab import protocol;'
                  'from factorlab.discovery.neural_forward_ticket import _LifecycleReader;'
                  f'protocol.ROOT=Path({str(tmp_path)!r});r=_LifecycleReader(Path({str(lc.directory)!r}));'
                  '\ntry:\n with r.controller_lock():raise AssertionError("foreign process entered")'
                  '\nexcept Timeout:print("blocked")\nfinally:r.close()')
            child=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,
                cwd=Path(protocol.__file__).resolve().parents[1],
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),timeout=15)
            assert child.returncode==0 and child.stdout.strip()=='blocked',child.stderr
        with other.controller_lock():other.verify()
        reader=_LifecycleReader(lc.directory)
        try:
            with reader.controller_lock():reader.verify()
        finally:reader.close()
    finally:
        other.close();lc.close()


def test_full_registered_dispatch_does_not_enclose_paid_ticket_audit(tmp_path,monkeypatch):
    # Execute the existing actual finite/feed/claim/controller/wallet fixture.
    # Hook only observes the paid-call boundary, then calls the original method.
    from factorlab.discovery.v7_forward import ForwardRuntime
    from test_v7_bound_logical_driver import test_actual_bound_logical_controller_with_full_service_headroom
    original=ForwardRuntime.fit_month;seen=[]
    def observed(self,*args,**kwargs):
        assert observation._READ_SCOPE.get() is None
        assert self.lifecycle.controller_lock().is_locked
        reader=_LifecycleReader(self.directory)
        try:
            with pytest.raises(Timeout):
                with reader.controller_lock():pass
        finally:reader.close()
        seen.append(True)
        return original(self,*args,**kwargs)
    monkeypatch.setattr(ForwardRuntime,'fit_month',observed)
    test_actual_bound_logical_controller_with_full_service_headroom(tmp_path,monkeypatch)
    assert seen==[True]


def test_real_300s_ticket_closing_audit_cannot_borrow_controller_lock(factory,monkeypatch):
    from factorlab.discovery import finite_market_oracle as finite
    from factorlab.discovery.known_linear_forward import pipeline_for,issue_known_ticket
    from factorlab.discovery.known_linear_reference import FrozenLinearRecipe
    from factorlab.discovery.work_lease import FrozenWorkLease,LeaseClosed
    gen,permit,runtime,wallet=factory(law_start='2025-12-20T00:00:00Z',hours=1033)
    clock=observation.QuarantinedObservation.bind(runtime,gen,permit)
    issued=0
    while issued<1033:
        rows=finite.publish_due(gen,permit,max_new_receipts=96)['issued']
        assert rows;issued+=len(rows)
    consumed=0
    while consumed<1032:
        rows=clock.advance_batch(runtime,max_events=min(96,1032-consumed))
        consumed+=rows['events_committed']
    prefix=clock.binding(1031)
    recipe=FrozenLinearRecipe('K0',runtime.spec)
    job=runtime.lifecycle.plan_fit('2026-02-01T00:00:00Z','A0',pipeline_for(recipe))
    inspections=[];slow={'lease':None,'delayed':False}
    original=observation.QuarantinedObservation.inspect
    def audited(self,*args,**kwargs):
        state=original(self,*args,**kwargs)
        scoped=observation._READ_SCOPE.get() is not None
        inspections.append(scoped)
        if slow['lease'] is not None and not scoped and not slow['delayed']:
            slow['delayed']=True
            time.sleep(max(0.,slow['lease'].monotonic_deadline-time.monotonic())+.03)
        return state
    monkeypatch.setattr(observation.QuarantinedObservation,'inspect',audited)
    # This is the exact serialization lock used by the full controller. The
    # producer opens/closes its OWN original source scope while it is held.
    with runtime.lifecycle.controller_lock():
        assert observation._READ_SCOPE.get() is None
        ticket=issue_known_ticket(runtime,wallet,job,recipe)
        assert True in inspections and inspections[-1] is False
        lease=FrozenWorkLease.from_dict(ticket.value['lease'])
        assert lease.lease_seconds==300 and lease.assert_parent_alive().allowed
        before=wallet.verify()
        slow['lease']=lease;inspections.clear()
        with pytest.raises(LeaseClosed):ticket.verify(status='reserved')
        assert slow['delayed'] and inspections[-1] is False
        assert observation._READ_SCOPE.get() is None
        assert wallet.verify()==before and before['reservations']==1
        assert ticket.value['context']['observation_prefix']==prefix
    # Closing failure leaves no read-scope grant and grants no replacement work.
    assert observation._READ_SCOPE.get() is None
    assert clock.binding(1031)==prefix
    with pytest.raises(LeaseClosed):ticket.verify(status='reserved')
