"""Real source/runtime/enrollment/ledger/statistic/selection/publication chain.

Artificial future clocks and pre-recorded forecasts are explicit fixture input.
No model fit, provider, original work grant or genuine certificate is simulated.
This exercises endpoint/adoption kernels, not a full model/provider trial.
"""
from pathlib import Path

import pandas as pd
import pytest

from factorlab.discovery.calibration_acceptance import AcceptanceCell, AcceptancePlan
from factorlab.discovery.calibration_runtime_contract import runtime_calibration_contract
from factorlab.discovery.claim_protocol import ClaimProtocol, FrozenClaimEndpoint
from factorlab.discovery.full_calibration_registry import FullTrialPlan, FullTrialRegistry
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
from factorlab.discovery.neural_forward_ticket import _bind_authority
from factorlab.discovery.quarantine_execution import permit_for, verify_decision
from factorlab.discovery.v7_forward import ForwardRuntime
from factorlab.discovery.v7_service import ForwardService
from factorlab.protocol import fingerprint


@pytest.fixture
def case(tmp_path, monkeypatch):
    from factorlab import protocol
    workspace = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
    monkeypatch.setattr(protocol, 'ROOT', workspace)
    clock = [pd.Timestamp('2026-10-01T00:00:05Z')]
    baseline = ForwardRuntime(tmp_path/'profile', tmp_path/'profile-feed', clock=lambda: clock[0])
    wallet = MonthlyWorkAuthority(tmp_path/'profile-wallet', run_id=str(tmp_path),
        task='volatility', universe=['BTCUSDT'], partition_policy='standard')
    _bind_authority(baseline, wallet)
    scope = runtime_calibration_contract(baseline.lifecycle)
    baseline.close()
    plan = FullTrialPlan(AcceptancePlan((AcceptanceCell('mixed', fingerprint('mixed'), 1000, ('factor1',)),)),
        scope, {'backend': 'codex', 'model_requested': 'gpt-6-astra', 'reasoning_requested': 'high',
                'context_policy': 'fresh_codex_per_trial_v1'},
        {key+'_digest': fingerprint(key) for key in ('design', 'validation', 'oracle')})
    registry = FullTrialRegistry(tmp_path/'registry', plan)
    paths = registry.paths('mixed', 0)
    runtime = ForwardRuntime(paths['runtime'], paths['feed'], clock=lambda: clock[0])
    trial = registry.identity_for('mixed', 0)
    wallet = MonthlyWorkAuthority(paths['authority'],
        run_id=fingerprint({'registry': registry.digest, 'trial': trial['trial_id']}),
        task='volatility', universe=['BTCUSDT'], partition_policy='standard')
    permit = registry.enroll(runtime, wallet, 'mixed', 0)
    try:
        yield registry, runtime, wallet, permit, clock
    finally:
        runtime.close()


def batch(case):
    _, runtime, _, _, _ = case
    lc = runtime.lifecycle
    lc.exposure.record(source=lc.plan.source_id, symbols=lc.plan.symbols,
        start='2026-07-01T00:00:00Z', end='2026-10-01T00:00:00Z', role='S', batch='A')
    lc.freeze_batch(batch='A', ordinal=1, start='2026-11-01T00:00:00Z',
        candidates=[{'pipeline_id': fingerprint('candidate'), 'artificial': 'candidate', 'complexity': 1}],
        reference={'pipeline_id': fingerprint('reference'), 'artificial': 'reference'},
        evidence_sources=[{'start': '2026-07-01T00:00:00Z', 'end': '2026-10-01T00:00:00Z'}])
    return lc


def recorded_pairs(lc, start, end):
    identity, _, _ = lc._batch('A')
    for stream, pipeline, value in [('A:reference', identity['reference']['pipeline_id'], .05),
            ('A:'+identity['candidates'][0]['pipeline_id'], identity['candidates'][0]['pipeline_id'], .09)]:
        for hour in pd.date_range(start, end, inclusive='left', freq='h'):
            origin = hour+pd.Timedelta(seconds=1)
            window = origin.ceil('h')
            prediction = {'value': value, 'received_at': origin.isoformat(),
                          'forecast_window_start': window.isoformat(), 'issued_at': origin.isoformat(),
                          'model_id': 'artificial-pre-recorded', 'pipeline_id': pipeline,
                          'availability': 'artificial_fixture_recorded_receipt'}
            lc.forecasts.put(run=lc.plan.digest, stream=stream, origin=origin,
                symbol='BTCUSDT', task='volatility', model_id=prediction['model_id'],
                pipeline_id=pipeline, prediction=prediction)
            maturity = window+pd.Timedelta(hours=12)
            lc.settle(stream=stream, origin=origin, symbol='BTCUSDT', task='volatility',
                label=.1, maturity=maturity, received_at=maturity)


def test_actual_C_kernel_quarantine_adoption_and_readonly_recovery(case, monkeypatch):
    registry, runtime, wallet, permit, clock = case
    lc = batch(case)
    original_budget = wallet.verify()
    clock[0] = pd.Timestamp('2026-12-01T00:00:05Z')
    recorded_pairs(lc, '2026-11-01T00:00:00Z', '2026-11-30T11:00:00Z')
    selected = lc.select_validation('A')['selected']
    assert selected and not lc.calibration_ready()
    # Actual service endpoint route, without poll/model-fit side effects.
    service = ForwardService(runtime, work_authority=wallet)
    try:
        actions = []
        service._endpoints(clock[0], actions)
    finally:
        service.close(close_runtime=False)
    assert lc.exposure.db.execute('SELECT COUNT(*) FROM confirmations').fetchone()[0] == 1
    clock[0] = pd.Timestamp('2027-03-03T00:00:00Z')
    recorded_pairs(lc, '2026-12-01T01:00:00Z', '2027-03-01T00:00:00Z')
    original_publish = lc.exposure.finish_confirmation
    calls = []

    def interrupted(*args):
        calls.append(True)
        raise OSError('fixture failure after real score commit before exposure publication')

    monkeypatch.setattr(lc.exposure, 'finish_confirmation', interrupted)
    with pytest.raises(OSError, match='fixture failure'):
        lc.conclude_confirmation('A')
    monkeypatch.setattr(lc.exposure, 'finish_confirmation', original_publish)
    committed = lc._batch_operation('A', 'confirmation_score_committed')['result']
    monkeypatch.setattr(lc, '_losses', lambda *a, **k: (_ for _ in ()).throw(AssertionError('must not rescore')))
    result = lc.recover_confirmation('A')
    assert result == committed and len(calls) == 1
    assert not result['supported'] and not result['calibration_passed'] and not result['formal_eligible']
    assert result['quarantined_decision']['accepted'] is True
    assert result['quarantined_decision']['trial_id'] == permit.trial['trial_id']
    for field, value in [('trial_id', fingerprint('borrowed_trial')),
                         ('accepted', False), ('registry_digest', fingerprint('other_registry'))]:
        forged = {**result, 'quarantined_decision': {**result['quarantined_decision'], field: value}}
        with pytest.raises(ValueError, match='outside'):
            verify_decision(lc, forged)
    with pytest.raises(ValueError, match='formal'):
        verify_decision(lc, {**result, 'supported': True})
    clock[0] = pd.Timestamp('2027-04-01T00:00:00Z')
    fallback = {'pipeline_id': fingerprint('fallback'), 'artificial': 'fallback'}
    assert lc.adoption_pipeline('2027-03-01T00:00:00Z', fallback) == fallback
    assert lc.adoption_pipeline(clock[0], fallback)['pipeline_id'] == selected['candidate_id']
    # Read-only scope reconstitution uses an artificial future observation clock,
    # never grants a lease or changes the registry/work clock.
    with permit.open_lifecycle() as reader:
        reader.clock = lambda: clock[0]
        assert reader.forecasts.verify()['forecasts'] == 5732
        assert reader.adoption_pipeline(clock[0], fallback)['pipeline_id'] == selected['candidate_id']
        assert not reader.calibration_ready()
    assert wallet.verify() == original_budget
    assert registry.verify()['formal_gate_eligible'] is False


def test_no_caller_or_foreign_permission_and_missing_original_enrollment(case):
    _, runtime, _, permit, _ = case
    assert permit_for(runtime.lifecycle).trial == permit.trial
    marker = runtime.directory.parent/'enrollment.json'
    marker.rename(marker.with_suffix('.preserved'))
    with pytest.raises(ValueError, match='missing'):
        runtime.lifecycle.confirmation_execution_ready()


def test_ordinary_runtime_still_rejects_quarantine_metadata_without_enrollment(case):
    registry, _, _, _, clock = case
    other = ForwardRuntime(registry.directory.parent/'ordinary', registry.directory.parent/'ordinary-feed',
                           clock=lambda: clock[0])
    try:
        assert permit_for(other.lifecycle) is None
        assert not other.lifecycle.confirmation_execution_ready()
        with pytest.raises(ValueError, match='outside'):
            verify_decision(other.lifecycle, {'supported': False, 'quarantined_decision': {
                'trial_id': 'handwritten', 'accepted': True}})
    finally:
        other.close()


def test_six_month_quarantine_empty_coverage_is_not_acceptance(case):
    _, runtime, _, permit, clock = case
    lc = runtime.lifecycle
    k0 = runtime.recipe_for()
    reference = {'pipeline_id': fingerprint('benchmark'), 'artificial': 'benchmark'}
    endpoints = [FrozenClaimEndpoint.library('2026-11-01T00:00:00Z', k0, reference),
                 FrozenClaimEndpoint.algorithm('2026-11-01T00:00:00Z', k0)]
    claims = ClaimProtocol.freeze(lc, endpoints)
    clock[0] = pd.Timestamp('2027-05-03T00:00:00Z')
    for slot in ('library_skill', 'algorithm'):
        result = claims.finalize(slot)
        assert not result['formal_eligible'] and not result['result']['supported']
        assert result['result']['quarantined_decision']['trial_id'] == permit.trial['trial_id']
        assert not result['result']['quarantined_decision']['accepted']
        assert result['result']['eligible'] == 181*24
        assert result['result']['coverage'] == 0
        assert claims.finalize(slot) == result
    assert not claims.verify()['formal_eligible']
