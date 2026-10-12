"""Real original source closing and a one-second boundary diagnostic.

This is not a paid model ticket or evidence of 300-second model admission.
The separate 1033-hour ticket test exercises the actual 300-second producer.
"""
import sys
import time
import pytest
from factorlab.discovery import quarantined_observation as observation
from factorlab.discovery.work_lease import FrozenWorkLease, LeaseClosed
from test_v7_quarantined_observation import factory, ready

@pytest.fixture
def tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp('b')

def test_original_closing_full_audit_precedes_real_lease_assertion(factory,monkeypatch):
    clock,gen,permit,runtime,wallet=ready(factory)
    prefix=clock.advance(runtime)
    original=observation.QuarantinedObservation.inspect
    calls=[];state={}
    def audited(self,*args,**kwargs):
        caller=sys._getframe(1).f_code.co_name
        result=original(self,*args,**kwargs)
        if caller=='_original_observation_read_scope' and observation._READ_SCOPE.get() is None:
            calls.append('original_closing_finished')
            lease=state['lease']
            time.sleep(max(0.,lease.monotonic_deadline-time.monotonic())+.03)
        return result
    monkeypatch.setattr(observation.QuarantinedObservation,'inspect',audited)
    before=wallet.verify()
    with runtime.lifecycle.controller_lock():
        with pytest.raises(LeaseClosed):
            with observation._original_observation_read_scope(runtime.lifecycle):
                assert observation.current_binding(runtime.lifecycle)==prefix
                # Short real lease isolates the final full-read boundary. No
                # paid work is reserved and no caller clock is injected.
                state['lease']=FrozenWorkLease.issue(1)
                assert state['lease'].assert_parent_alive().allowed
            calls.append('assert_after_closing')
            state['lease'].assert_parent_alive()
        assert calls==['original_closing_finished','assert_after_closing']
        assert observation._READ_SCOPE.get() is None
    assert wallet.verify()==before
    # Removal of the scope grant restores ordinary original reconstruction.
    assert observation.current_binding(runtime.lifecycle)==prefix
