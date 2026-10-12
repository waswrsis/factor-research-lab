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
from factorlab.discovery.full_calibration_registry import FullTrialPlan, FullTrialRegistry
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
from factorlab.discovery.neural_forward_ticket import _bind_authority
from factorlab.discovery.v7_forward import ForwardRuntime
from factorlab.protocol import fingerprint


@pytest.fixture
def case(tmp_path, monkeypatch):
    from factorlab import protocol
    workspace = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
    monkeypatch.setattr(protocol, 'ROOT', workspace)
    clock = [pd.Timestamp.now(tz='UTC').normalize().replace(day=1)-pd.DateOffset(months=6)]
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

