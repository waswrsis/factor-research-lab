"""Actual physical service identity readback; artificial empty feed, no provider."""
import json
from types import SimpleNamespace
import pytest
import test_controller_native_fixture as fixture
from test_controller_native_accounting import manual
from factorlab.discovery import forward_service_scope as scope
from factorlab.discovery.v7_forward import ForwardConfig


@pytest.fixture(params=['calendar', 'elapsed90'])
def coupled_case(tmp_path, monkeypatch, request):
    if request.param=='elapsed90':
        original=fixture.ForwardRuntime
        def elapsed(*args, **kwargs):
            assert 'config' not in kwargs
            return original(*args, config=ForwardConfig(training_window_days=90,refit_interval_days=15), **kwargs)
        monkeypatch.setattr(fixture,'ForwardRuntime',elapsed)
    generator=fixture.case.__wrapped__(tmp_path,monkeypatch,SimpleNamespace(param=300))
    value=next(generator)
    try:yield value
    finally:
        try:next(generator)
        except StopIteration:pass


def test_actual_reference_preserves_original_versioned_lifecycle_and_deadline(coupled_case):
    _,runtime,wallet,_,driver=coupled_case
    service,lease=manual(coupled_case)
    before=wallet.verify()
    try:
        with scope._service_step_scope(service):reference=scope.current_service_reference()
        assert json.loads((service.directory/'identity.json').read_text())['lifecycle']==runtime.lifecycle.plan.digest
        observed=scope.verify_service_reference(reference)
        assert observed.monotonic_deadline==lease.supervisor().monotonic_deadline
        assert observed.wall_deadline==lease.supervisor().wall_deadline
        assert wallet.verify()==before
    finally:service.close()


def test_rehashed_runtime_cadence_tampering_does_not_replace_original_service_identity(coupled_case):
    from factorlab.protocol import fingerprint
    _,runtime,wallet,_,driver=coupled_case
    service,lease=manual(coupled_case)
    before=wallet.verify()
    try:
        with scope._service_step_scope(service):reference=scope.current_service_reference()
        path=runtime.directory/'forward.json';original=path.read_bytes()
        value=json.loads(original)
        fields=value['identity']['lifecycle']
        fields['training_window_days']=None if fields['training_window_days']==90 else 90
        fields['refit_interval_days']=None if fields['refit_interval_days']==15 else 15
        value['sha256']=fingerprint(value['identity']);path.write_text(json.dumps(value))
        try:
            with pytest.raises(ValueError):scope.verify_service_reference(reference)
            assert wallet.verify()==before
        finally:path.write_bytes(original)
    finally:service.close()
