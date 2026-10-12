"""Bounded engineering fixtures. Empty/artificial feed; no provider/model run.

Staged modules are loaded under their production import names. The actual
service test executes empty receipt ticks and plans only; the once-only tests
replace the private dispatch to emulate crashes, explicitly not runtime proof.
"""
import importlib.util
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import pytest

from factorlab.discovery import full_calibration_registry as registry_module
from factorlab.discovery import full_trial_controller as controller
from factorlab.discovery import full_trial_certificate as certificate
from factorlab.discovery import v7_autonomous as autonomous

from factorlab.discovery.calibration_acceptance import AcceptancePlan, AcceptanceCell
from factorlab.discovery.calibration_runtime_contract import runtime_calibration_contract
from factorlab.discovery.claim_protocol import ClaimProtocol, FrozenClaimEndpoint
from factorlab.discovery.full_calibration_registry import FullTrialPlan, FullTrialRegistry
from factorlab.discovery.known_linear_reference import FrozenLinearRecipe
from factorlab.discovery.monthly_schedule import default_schedule
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
from factorlab.discovery.neural_forward_ticket import _bind_authority
from factorlab.discovery.v7_forward import ForwardRuntime
from factorlab.discovery.v7_service import ServiceConfig
from factorlab.protocol import fingerprint


@pytest.fixture
def case(tmp_path, monkeypatch, request):
    from factorlab import protocol
    monkeypatch.setattr(protocol, 'ROOT', tmp_path)
    (tmp_path/'factorlab').mkdir()
    (tmp_path/'factorlab'/'artificial.py').write_text('FIXTURE = True\n')
    # Bound canonical source paths for these exact staged implementations.
    # The rest of this small workspace remains an explicitly synthetic fixture.
    (tmp_path/'factorlab'/'discovery').mkdir()
    for name in ('full_calibration_registry', 'full_trial_controller', 'full_trial_certificate', 'v7_autonomous', 'forward_service_scope', 'v7_service', 'v7_forward'):
        (tmp_path/'factorlab'/'discovery'/(name+'.py')).write_bytes(Path(__import__('factorlab.discovery.'+name,fromlist=[name]).__file__).read_bytes())
    (tmp_path/'pyproject.toml').write_text('[project]\n')
    (tmp_path/'uv.lock').write_text('version = 1\n')
    baseline = ForwardRuntime(tmp_path/'profile', tmp_path/'profile-feed')
    wallet = MonthlyWorkAuthority(tmp_path/'profile-wallet', run_id='profile',
        task='volatility', universe=['BTCUSDT'], partition_policy='standard')
    _bind_authority(baseline, wallet)
    scope = runtime_calibration_contract(baseline.lifecycle)
    baseline.close()
    plan = FullTrialPlan(AcceptancePlan((AcceptanceCell('mixed', fingerprint('mixed'), 1000, ('algorithm',)),)),
        scope, {'backend': 'codex', 'model_requested': 'gpt-6-astra', 'reasoning_requested': 'high',
                'context_policy': 'fresh_codex_per_trial_v1'},
        {key+'_digest': fingerprint(key) for key in ('design', 'validation', 'oracle')})
    registry = FullTrialRegistry(tmp_path/'registry', plan)
    paths = registry.paths('mixed', 0)
    runtime = ForwardRuntime(paths['runtime'], paths['feed'])
    trial = registry.identity_for('mixed', 0)
    wallet = MonthlyWorkAuthority(paths['authority'],
        run_id=fingerprint({'registry': registry.digest, 'trial': trial['trial_id']}),
        task='volatility', universe=['BTCUSDT'], partition_policy='standard')
    permit = registry.enroll(runtime, wallet, 'mixed', 0)
    start = (pd.Timestamp.now(tz='UTC').normalize().replace(day=1)+pd.DateOffset(months=1)).isoformat()
    k0 = runtime.recipe_for()
    ClaimProtocol.freeze(runtime.lifecycle, [FrozenClaimEndpoint.library(start, k0,
        runtime.recipe_for(FrozenLinearRecipe(rolling_spec=runtime.spec))), FrozenClaimEndpoint.algorithm(start, k0)])
    auto = autonomous.AutonomousConfig(max_attempts=1, not_before_month='2099-01')
    search = default_schedule(auto.research, month='2026-10')
    schedule = controller.FullControllerSchedule(
        tuple((pd.Timestamp.now(tz='UTC')-pd.Timedelta(minutes=2-i)).isoformat() for i in range(2)),
        asdict(ServiceConfig(session_id='registered', wall_seconds=getattr(request, "param", 300), max_steps=2,
                             max_fit_attempts=1, poll=False)), asdict(auto), search.to_dict())
    driver = controller.FullTrialController.freeze(permit, runtime, wallet, schedule)
    try:
        yield registry, runtime, wallet, permit, driver
    finally:
        runtime.close()

