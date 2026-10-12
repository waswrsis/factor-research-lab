"""Synthetic local fixtures; no provider call, trained model or statistical proof."""
import copy
from dataclasses import replace
from pathlib import Path
import pytest
from factorlab.protocol import fingerprint
from factorlab.discovery import full_trial_certificate as certificate
from factorlab.discovery import full_mechanism_contract as mechanism
from factorlab.discovery.full_adaptive_risk import reconstruct_full_risk
from factorlab.discovery.calibration_acceptance import AcceptanceCell, AcceptancePlan, CLAIMS
from factorlab.discovery.finite_market_oracle import FiniteGenerator
from factorlab.discovery.pipeline_calibration import verify_calibration_certificate
from test_v7_full_trial_controller import case
from test_v7_quarantined_observation import factory, ready


def test_original_pending_certificate_real_lifecycle_route_and_changed_binding(case,tmp_path):
    registry,runtime,wallet,permit,driver=case
    before=wallet.verify()
    made=certificate.produce_full_certificate(registry)
    linked=certificate.publish_lifecycle_certificate(registry,tmp_path/'link',artifact_digest=made['artifact_digest'])
    result=verify_calibration_certificate(tmp_path/'link',expected_binding=linked['calibration_binding'])
    assert result['formal_gate_eligible'] is False
    claims, scope = certificate._claims(permit, include_scope=True)
    assert result['six_month_claim_scope'] == scope
    assert result['six_month_claim_scope_hash'] == fingerprint(scope)
    assert scope['runtime_scope_hash'] == fingerprint(registry.plan.runtime_scope)
    assert set(scope['endpoints']) == {'library_skill', 'algorithm'}
    assert all(e['nominal_UTC_calendar_months'] == 6 for e in scope['endpoints'].values())
    assert result['numeric_acceptance']['cells'][0]['possible_family_errors']==1000
    assert 'prospective_joint_mechanism_contract' in result['pending_dependencies']
    with pytest.raises(ValueError,match='does not bind'):
        verify_calibration_certificate(tmp_path/'link',expected_binding='f'*64)
    assert reconstruct_full_risk(registry,'mixed',0)['claims']==dict.fromkeys(CLAIMS)
    with pytest.raises(TypeError):
        reconstruct_full_risk({'success':True},'mixed',0)
    assert wallet.verify()==before


def test_mechanism_real_source_freeze_rejects_boolean_identity_and_binary_replacement(factory,tmp_path):
    gen,permit,runtime,wallet=factory()
    roles={r:r for r in ('null','boundary','positive','mixed')}
    acceptance=AcceptancePlan(tuple(AcceptanceCell(r,gen.law.digest,1000,CLAIMS if r=='positive' else ()) for r in roles))
    generator=FiniteGenerator(tmp_path/'mechanism-generator',gen.law,acceptance)
    executable=tmp_path/'artificial-provider-binary';executable.write_bytes(b'not an executable; test never invokes it')
    value=mechanism.freeze_mechanism_contract([generator],provider_executable=executable,
        resolved_model='synthetic-reported-model',assumptions=mechanism.ASSUMPTIONS,scenario_roles=roles)
    assert mechanism.verify_mechanism_contract(value,acceptance)[gen.law.digest].digest==generator.digest
    assert value['assumption_status']=='declared_modelling_conditions_not_empirically_proved'
    changed={**value,'provider_authenticity_certified':True}
    with pytest.raises(ValueError,match='changed'):
        mechanism.verify_mechanism_contract(changed,acceptance)
    # No retrospective contract can be grafted onto an already frozen registry.
    with pytest.raises(ValueError):
        permit.registry.plan=replace(permit.registry.plan,mechanism_contract=value)
    executable.write_bytes(b'changed original binary')
    with pytest.raises(ValueError,match='changed'):
        mechanism.verify_mechanism_contract(value,acceptance)


def test_six_month_consumer_requires_exact_original_scope(case, monkeypatch):
    """Consumer branch check, not a manufactured positive full certificate."""
    from factorlab.discovery import quarantine_execution as qe
    from factorlab.discovery import pipeline_calibration as calibration
    from factorlab.discovery.claim_protocol import ClaimProtocol
    from factorlab.discovery.v7_lifecycle import Lifecycle
    registry, runtime, wallet, permit, driver = case
    protocol = ClaimProtocol(runtime.lifecycle)
    scope = qe.six_month_claim_scope(protocol)
    result = {'formal_gate_eligible': True}
    # This synthetic seam represents only the preceding independently verified
    # full/runtime gate. It grants no real production result or permission.
    monkeypatch.setattr(qe, 'permit_for', lambda lc: None)
    monkeypatch.setattr(Lifecycle, 'calibration_ready', lambda lc: True)
    monkeypatch.setattr(calibration, 'verify_calibration_certificate', lambda *a, **k: result)
    from types import SimpleNamespace
    original_plan = runtime.lifecycle.plan
    runtime.lifecycle.plan = SimpleNamespace(calibration_directory='synthetic-consumer',
        calibration_binding='synthetic-consumer')
    # The exact original scope was read before this consumer-only prerequisite
    # seam; preserve its actual value rather than inventing endpoint recipes.
    monkeypatch.setattr(qe, 'six_month_claim_scope', lambda p: scope)
    try:
        assert qe.claim_formal_ready(protocol) is False
        result['six_month_claim_scope_hash'] = fingerprint(scope)
        assert qe.claim_formal_ready(protocol) is True
        changed = copy.deepcopy(scope)
        changed['endpoints']['algorithm']['rule']['alpha'] = .02
        result['six_month_claim_scope_hash'] = fingerprint(changed)
        assert qe.claim_formal_ready(protocol) is False
    finally:
        runtime.lifecycle.plan = original_plan


def test_original_allocator_clock_accepts_prefix_and_refuses_missing_changed_late(factory):
    from factorlab.discovery.monthly_partition_policy import _algorithm_allocation_time
    from factorlab.discovery.work_lease import FrozenWorkLease
    clock,gen,permit,runtime,wallet=ready(factory)
    before=wallet.verify()
    early=FrozenWorkLease.issue(300)
    prefix=clock.advance(runtime)
    later=FrozenWorkLease.issue(300)
    import pandas as pd
    assert _algorithm_allocation_time(runtime.directory,later,prefix)==pd.Timestamp(prefix['logical_at'])
    assert pd.Timestamp(later.wall_started_at)>pd.Timestamp(prefix['logical_at'])
    with pytest.raises(ValueError,match='unavailable'):
        _algorithm_allocation_time(runtime.directory,early,prefix)
    with pytest.raises(ValueError,match='lost'):
        _algorithm_allocation_time(runtime.directory,later,None)
    changed=copy.deepcopy(prefix);changed['logical_at']='2020-01-01T00:00:00+00:00'
    with pytest.raises(ValueError,match='prefix'):
        _algorithm_allocation_time(runtime.directory,later,changed)
    assert wallet.verify()==before
