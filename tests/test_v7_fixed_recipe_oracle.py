"""Tiny real source/wallet plumbing with an explicitly synthetic selection seam.

No original full selection, six-month coverage, provider or Linux proof claimed.
"""
from dataclasses import replace
from types import SimpleNamespace
import json
import pytest

from factorlab.protocol import fingerprint
from factorlab.discovery.calibration_oracle_evidence import OracleClaimSpec
from factorlab.discovery.full_calibration_registry import QuarantinedTrialPermit
from factorlab.discovery.fixed_recipe_oracle import FixedRecipeOracle, fixed_enrollment_preflight
from factorlab.discovery.known_linear_reference import FrozenLinearRecipe
from test_v7_quarantined_observation import factory, tmp_path


@pytest.fixture
def replica(factory, monkeypatch):
    gen, original, runtime, wallet=factory()
    registry=original.registry
    validation=registry.identity_for('finite',0)
    spec=OracleClaimSpec('library_skill','2026-02-01T00:00:00+00:00','2026-08-01T00:00:00+00:00',
        'benchmark','K0',runtime.recipe_for(FrozenLinearRecipe(rolling_spec=runtime.spec)),runtime.recipe_for())
    # Only original-selection/controller-read prerequisites are synthetic here;
    # replica identity, wallets, actual physical source, observation and tickets
    # below are the real production implementation.
    from factorlab.discovery import calibration_oracle_evidence as evidence
    monkeypatch.setattr(evidence,'_frozen_estimands',lambda permit: ({'validation_trial':validation},[spec]))
    from factorlab.discovery.full_trial_controller import FullTrialController
    controller=SimpleNamespace(identity={'engineering_fixture':True},
        schedule=SimpleNamespace(service={'wall_seconds':3000.}))
    monkeypatch.setattr(FullTrialController,'inspect',classmethod(lambda cls,permit:controller))
    permit=QuarantinedTrialPermit(registry,registry.identity_for('finite',0,oracle_index=0,estimand_slot='library_skill'))
    from factorlab.discovery.v7_cli import _enroll_original_trial, _original_trial_runtime
    _enroll_original_trial(registry,'finite',0,0,estimand_slot='library_skill')
    own=_original_trial_runtime(permit)
    try:
        driver=FixedRecipeOracle.freeze(permit,own,gen)
        yield driver,own,permit,spec
    finally:
        own.close()


def test_independent_identity_real_failed_fit_is_prepaid_and_not_retried(replica):
    driver,runtime,permit,spec=replica
    from factorlab.discovery.neural_forward_ticket import open_existing_authority
    authority=open_existing_authority(permit.paths['authority'])
    before=authority.verify()
    result=driver.pump()
    assert result['state']=='failed_or_late_consumed'  # one bar cannot fit 240-hour minimum
    rows=driver.verify()
    assert len(rows)==1 and rows[0]['result']['observation_prefix']['hour']==0
    assert rows[0]['completion']['physical_finish']['monotonic']>=rows[0]['result']['physical_finish']['monotonic']
    after=authority.verify()
    assert after['reservations']>before['reservations']
    assert driver.pump()['retry_allowed'] is False
    assert authority.verify()==after
    assert driver.identity['selection']['original_estimand']['end']==spec.end
    assert len(driver.identity['support'])==128


def test_fixed_replica_rejects_wrong_slot_clock_and_orphan_intent(replica):
    driver,runtime,permit,spec=replica
    with pytest.raises(TypeError):
        driver.pump(now='2099-01-01')
    with pytest.raises(ValueError):
        permit.registry.identity_for('finite',0,oracle_index=0,estimand_slot='algorithm')
    driver.pump()
    path=driver.directory/'steps/00000-attempt.json'
    # Preserve the original as a retained nonmatching name, never delete evidence.
    path.rename(path.with_suffix('.retained'))
    with pytest.raises(ValueError,match='anchor|intent'):
        FixedRecipeOracle.inspect(permit)


def test_fixed_replica_wrong_recipe_is_not_authorized(replica):
    driver,runtime,permit,spec=replica
    driver.pump()
    from factorlab.discovery.fixed_recipe_oracle import job_context
    row=runtime.lifecycle.db.execute('SELECT id,pipeline FROM jobs ORDER BY cutoff,stream').fetchone()
    assert row is not None
    pipeline=json.loads(row[1])
    assert job_context(runtime.lifecycle,row[0],pipeline)['original_estimand_hash']==spec.digest
    with pytest.raises(ValueError,match='recipe'):
        job_context(runtime.lifecycle,row[0],{**pipeline,'pipeline_id':'f'*64})


def test_design_labels_do_not_replace_original_truth_patterns():
    from factorlab.discovery.calibration_acceptance import AcceptanceCell, AcceptancePlan, CLAIMS, OracleRiskEnvelope, ClaimObservation, TrialObservation
    from factorlab.discovery.full_trial_certificate import _truth_pattern_coverage
    roles={r:r for r in ('null','boundary','positive','mixed')}
    plan=AcceptancePlan(tuple(AcceptanceCell(r,fingerprint('same-law'),1000,CLAIMS) for r in roles))
    envelope=OracleRiskEnvelope((1.,1.),(.5,.5),'finite_exhaustive_risk_enclosure',fingerprint('numeric-only'),plan.oracle_risk_alpha)
    rows=[TrialObservation(r,0,'complete',fingerprint(r),tuple(ClaimObservation(s,True,envelope) for s in CLAIMS)) for r in roles]
    result=_truth_pattern_coverage(plan,rows,{'scenario_roles':roles})
    assert 'original_all_five_known_null_truth_coverage' in result['pending']
    assert 'original_mixed_null_alternative_truth_coverage' in result['pending']
    assert 'original_boundary_truth_at_delta0_not_identified' in result['pending']


def test_service_one_second_cannot_open_a_new_300_second_fit():
    from factorlab.discovery.fixed_recipe_oracle import _fit_headroom
    from factorlab.discovery.work_lease import FrozenWorkLease
    lease=FrozenWorkLease.issue(300)
    with pytest.raises(ValueError,match='complete 300s fit'):
        _fit_headroom({'physical_lease':lease.to_dict(),'remaining_service_seconds':1.})
    _fit_headroom({'physical_lease':lease.to_dict(),'remaining_service_seconds':600.})


def _terminal_primitive(tmp_path, seconds):
    """Real clock, dual immutable IO and LC anchor; no selection/fit evidence."""
    import sqlite3
    from factorlab.discovery.fixed_recipe_oracle import _write_step
    from factorlab.discovery.work_lease import FrozenWorkLease, _observe
    root=tmp_path/'registry'; runtime=root/'trial/runtime'
    runtime.mkdir(parents=True)
    with sqlite3.connect(runtime/'lifecycle.sqlite') as db:
        db.execute('CREATE TABLE operations (seq INTEGER PRIMARY KEY, record TEXT, hash TEXT)')
    permit=SimpleNamespace(registry=SimpleNamespace(directory=root),
        paths={'root':root/'trial','runtime':runtime},trial={'trial_id':'original-test-trial'})
    lease=FrozenWorkLease.issue(300)
    attempt={'physical_lease':lease.to_dict(),'remaining_service_seconds':seconds}
    completion={'state':'awaiting_successor_witness','physical_finish':_observe(lease),
                'attempt_hash':fingerprint(attempt),'elapsed_seconds':0.}
    _write_step(permit,0,'commit',completion)
    return permit,{'attempt':attempt,'completion':completion}


def test_terminal_tail_065_over_original_05_cannot_be_confirmed_or_refunded(tmp_path):
    import time
    from factorlab.discovery.fixed_recipe_oracle import _witness_predecessor, _confirmed, _charged_seconds
    permit,row=_terminal_primitive(tmp_path,.5)
    # Actual post-publication delay, the reproduced old durable-complete hole.
    time.sleep(.65)
    assert not _confirmed(row['attempt'],row['completion'],None)
    witness=_witness_predecessor(permit,0,row)
    assert witness['state']=='failed_or_unknown_consumed'
    assert not _confirmed(row['attempt'],row['completion'],witness)
    assert _charged_seconds(permit,0,row['attempt'],row['completion'])==.5
    # Original negative witness is immutable; a later call cannot overwrite it.
    with pytest.raises(ValueError,match='Immutable'):
        _witness_predecessor(permit,0,row)


def test_terminal_step_needs_real_successor_and_preserves_original_start(tmp_path):
    from factorlab.discovery.fixed_recipe_oracle import _witness_predecessor, _confirmed, _charged_seconds
    permit,row=_terminal_primitive(tmp_path,30.)
    original=dict(row['attempt']['physical_lease'])
    assert not _confirmed(row['attempt'],row['completion'],None)
    assert _charged_seconds(permit,0,row['attempt'],row['completion'])==30.
    witness=_witness_predecessor(permit,0,row)
    assert _confirmed(row['attempt'],row['completion'],witness)
    assert row['attempt']['physical_lease']==original
    assert _charged_seconds(permit,0,row['attempt'],row['completion'])>0
    changed={**witness,'observed':{**witness['observed'],'boot_id':'windows:1'}}
    assert not _confirmed(row['attempt'],row['completion'],changed)
    changed={**witness,'observed':{**witness['observed'],'parent_alive':False}}
    assert not _confirmed(row['attempt'],row['completion'],changed)
    # Mirror loss cannot create a refund or a valid replacement witness.
    mirror=permit.registry.directory/'fixed_oracle_steps'/permit.trial['trial_id']/'00000-witness.json'
    mirror.rename(mirror.with_suffix('.retained'))
    assert _charged_seconds(permit,0,row['attempt'],row['completion'])==30.


def test_confirm_only_pump_does_not_open_work_or_issue_replacement_start(tmp_path,monkeypatch):
    from factorlab.discovery import fixed_recipe_oracle as module
    from factorlab.discovery.work_lease import FrozenWorkLease
    permit,row=_terminal_primitive(tmp_path,30.)
    module._write_step(permit,0,'attempt',row['attempt'])
    # This bounded orchestration test supplies an already-verified row; actual
    # full provenance is exercised by the separate registry/replica fixture.
    row.update(result={'state':'complete'},witness=None)
    driver=FixedRecipeOracle.__new__(FixedRecipeOracle)
    driver.permit=permit; driver.directory=permit.paths['root']/'fixed_oracle'
    driver.identity={'service_seconds':30.}
    monkeypatch.setattr(driver,'verify',lambda:[row])
    def forbidden(*args,**kwargs):
        raise AssertionError('Confirmation must not open work or a new start')
    monkeypatch.setattr(module,'_open_runtime',forbidden)
    monkeypatch.setattr(FrozenWorkLease,'issue',forbidden)
    assert driver.pump()['state']=='predecessor_confirmed'


@pytest.mark.parametrize('terminal',[
    'original_source_exhausted','original_window_exhausted','failed_or_late_consumed',
    'service_budget_exhausted','failed_or_unknown_consumed','interrupted_consumed'])
def test_same_issuer_run_stops_on_every_terminal_without_retry(terminal,monkeypatch):
    import os
    from factorlab.discovery.work_lease import FrozenWorkLease
    driver=FixedRecipeOracle.__new__(FixedRecipeOracle)
    driver.identity={'source_hours':2}
    calls=[]
    states=iter(['awaiting_successor_witness','predecessor_confirmed',terminal])
    def pump():
        calls.append(os.getpid())
        return {'state':next(states)}
    def no_new_lease(*args,**kwargs):
        raise AssertionError('run must only dispatch the original pump')
    monkeypatch.setattr(driver,'pump',pump)
    monkeypatch.setattr(FrozenWorkLease,'issue',no_new_lease)
    result=driver.run()
    assert result=={'state':terminal,'pump_calls':3,'last':{'state':terminal},'formal_gate_eligible':False}
    assert calls==[os.getpid()]*3


def test_same_issuer_run_is_bounded_and_includes_final_confirmation(monkeypatch):
    driver=FixedRecipeOracle.__new__(FixedRecipeOracle)
    driver.identity={'source_hours':1}
    states=iter(['awaiting_successor_witness','predecessor_confirmed','original_source_exhausted'])
    monkeypatch.setattr(driver,'pump',lambda:{'state':next(states)})
    assert driver.run()['pump_calls']==3
    calls=[]
    def nonterminal():
        calls.append(1)
        return {'state':'awaiting_successor_witness'}
    monkeypatch.setattr(driver,'pump',nonterminal)
    result=driver.run()
    assert result['state']=='invocation_bound_reached'
    assert result['pump_calls']==len(calls)==3
