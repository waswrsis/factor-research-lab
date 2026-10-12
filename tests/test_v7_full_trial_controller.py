"""Bounded engineering fixtures. Empty/artificial feed; no provider/model run.

Staged modules are loaded under their production import names. The actual
service test executes empty receipt ticks and plans only; the once-only tests
replace the private dispatch to emulate crashes, explicitly not runtime proof.
The normal composite tick uses an artificial healthy host observation; actual
host health and OS limits require the separate platform acceptance.
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
def case(tmp_path, monkeypatch):
    from factorlab import protocol
    monkeypatch.setattr(protocol, 'ROOT', tmp_path)
    (tmp_path/'factorlab').mkdir()
    (tmp_path/'factorlab'/'artificial.py').write_text('FIXTURE = True\n')
    # Bound canonical source paths for these exact staged implementations.
    # The rest of this small workspace remains an explicitly synthetic fixture.
    (tmp_path/'factorlab'/'discovery').mkdir()
    for name in ('full_calibration_registry', 'full_trial_controller', 'full_trial_certificate', 'v7_autonomous'):
        (tmp_path/'factorlab'/'discovery'/(name+'.py')).write_bytes(Path(sys.modules['factorlab.discovery.'+name].__file__).read_bytes())
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
        asdict(ServiceConfig(session_id='registered', wall_seconds=300, max_steps=2,
                             max_fit_attempts=1, poll=False)), asdict(auto), search.to_dict())
    driver = controller.FullTrialController.freeze(permit, runtime, wallet, schedule)
    try:
        yield registry, runtime, wallet, permit, driver
    finally:
        runtime.close()


def test_original_schedule_freeze_readback_and_scope(case):
    registry, runtime, wallet, permit, driver = case
    original = wallet.verify()
    assert controller.FullTrialController.inspect(permit).identity == driver.identity
    assert controller.FullTrialController.freeze(permit, runtime, wallet, driver.schedule).identity == driver.identity
    with pytest.raises(ValueError, match='cannot be replaced'):
        controller.FullTrialController.freeze(permit, runtime, wallet,
            replace(driver.schedule, due_at=tuple((pd.Timestamp(d)+pd.Timedelta(seconds=1)).isoformat() for d in driver.schedule.due_at)))
    assert wallet.verify() == original
    assert 'factorlab/discovery/full_trial_controller.py' in driver.identity['source_files']
    with pytest.raises(TypeError, match='permit'):
        controller.FullTrialController.inspect({'verified': True})


def test_actual_production_composite_tick_no_provider_or_fit(case, monkeypatch):
    import copy
    from factorlab.discovery.v7_service import HostGuard
    host_observation = {'rss_mib': 0., 'available_mib': 9000., 'processes': []}
    monkeypatch.setattr(HostGuard, '_inspect', staticmethod(lambda: copy.deepcopy(host_observation)))
    _, runtime, wallet, _, driver = case
    (runtime.directory / 'artificial-host-boundary.json').write_text(json.dumps({
        'host_observation': host_observation,
        'actual_host_health_or_hard_limits_proved': False,
        'native_clocks_wallets_and_controller_verifiers_replaced': False,
    }, indent=2), encoding='utf8')
    before = wallet.verify()
    now = pd.Timestamp.now(tz='UTC')
    runtime.feed.receive(pd.DataFrame([{'symbol': 'BTCUSDT', 'open_time': now.floor('h')-pd.Timedelta(hours=2),
        'open': 100., 'high': 101., 'low': 99., 'close': 100., 'volume': 10.}]),
        received_at=now, response_hash=fingerprint('explicit single synthetic engineering candle'))
    result = driver.step(0)
    assert result['state'] == 'awaiting_successor_witness'
    assert driver.pump()['state'] == 'completed_step'
    assert driver.step_completed(driver.verify()[0])
    assert 'forward' in result['output'] and 'research' in result['output']
    assert (runtime.directory/'service'/'registered'/'state.sqlite').exists()
    assert (runtime.directory/'autonomous'/'scheduler.sqlite').exists()
    # Original C300 cannot enclose a fresh 300-second paid lease plus its
    # unchanged 2-second publication margin. The actual service must refuse
    # before reserving work; the one-row feed never grants a special exception.
    assert before['reservations'] == 0
    assert wallet.verify()['reservations'] == 0
    assert any(a.get('kind') == 'fit' and a.get('status') == 'insufficient_service_headroom'
               for a in result['output']['forward']['actions'])
    after = wallet.verify()
    assert driver.step(0) == result
    assert wallet.verify() == after
    assert len(driver.verify()) == 1


def test_once_only_private_crash_fixture_and_later_slots(case, monkeypatch):
    _, _, wallet, permit, driver = case
    calls = []
    monkeypatch.setattr(controller.FullTrialController, '_dispatch', lambda self: calls.append(1) or {'synthetic': True})
    first = driver.step(0)
    again = controller.FullTrialController.inspect(permit)
    assert again.step(0) == first and calls == [1]
    assert again.pump()['state'] == 'completed_step' and calls == [1]
    assert again.step(1)['state'] == 'awaiting_successor_witness' and calls == [1, 1]
    assert again.pump()['state'] == 'completed_step'
    assert wallet.verify()['reservations'] == 0


def test_failed_dispatch_is_consumed_not_rerun(case, monkeypatch):
    _, _, _, permit, driver = case
    calls = []
    def fail(self):
        calls.append(1)
        raise OSError('synthetic crash')
    monkeypatch.setattr(controller.FullTrialController, '_dispatch', fail)
    with pytest.raises(OSError, match='synthetic'):
        driver.step(0)
    assert controller.FullTrialController.inspect(permit).step(0)['state'] == 'failed_consumed'
    assert calls == [1]


def test_orphan_original_attempt_is_retained_without_dispatch(case, monkeypatch):
    _, _, _, permit, driver = case
    from factorlab.discovery.work_lease import FrozenWorkLease
    lease = driver._start_execution()
    attempt = {'index': 0, 'due_at': driver.schedule.due_at[0],
        'controller_hash': fingerprint(driver.identity), 'lease': lease.to_dict(),
        'clock_hash': fingerprint(driver._execution_clock()), 'physical_start': lease.observe(),
        'observation_prefix': None, 'previous': '0'*64}
    driver._write_pair(0, 'attempt', attempt)
    monkeypatch.setattr(controller.FullTrialController, '_dispatch', lambda self: pytest.fail('must not replay'))
    result = controller.FullTrialController.inspect(permit).step(0)
    assert result['state'] == 'interrupted_consumed'
    assert driver.step(0) == result


def test_source_and_missing_original_step_anchor_reject(case, monkeypatch):
    _, _, _, permit, driver = case
    monkeypatch.setattr(controller.FullTrialController, '_dispatch', lambda self: {'synthetic': True})
    driver.step(0)
    driver._paths(0, 'result')[1].unlink()
    with pytest.raises(ValueError, match='mirror missing'):
        controller.FullTrialController.inspect(permit)


def test_provider_and_public_callback_are_not_permissions(case):
    _, _, _, permit, driver = case
    original = permit.registry.plan.provider_policy.copy()
    permit.registry.plan.provider_policy['model_requested'] = 'other-provider'
    with pytest.raises(ValueError):
        controller.FullTrialController.inspect(permit)
    permit.registry.plan.provider_policy.clear()
    permit.registry.plan.provider_policy.update(original)
    with pytest.raises(TypeError):
        driver.step(0, backend=lambda: {'success': True})
    with pytest.raises(ValueError, match='reserved family'):
        controller.RegisteredTrialCodexBackend.__new__(controller.RegisteredTrialCodexBackend)(None)


def test_unbound_logical_clock_remains_typed_pending(case):
    _, _, _, _, driver = case
    # No changed original schedule is written or dispatched. This inspects the
    # explicit dependency contract only, with a fresh dataclass fixture.
    driver.schedule = replace(driver.schedule, observation_mode='bound_logical_observation')
    assert any(d.code == 'bound_observation_clock' for d in driver.pending_dependencies())


def test_full_certificate_keeps_original_denominator_and_formal_pending(case):
    registry, _, wallet, _, _ = case
    before = wallet.verify()
    made = certificate.produce_full_certificate(registry)
    record = certificate.verify_full_certificate(registry, artifact_digest=made['artifact_digest'],
        expected_runtime_scope_hash=fingerprint(registry.plan.runtime_scope), require_formal=False)
    assert record['registered_trials'] == 1000
    cell = record['numeric_acceptance']['cells'][0]
    assert cell['missing'] == 999 and cell['failed'] == 1
    assert cell['possible_family_errors'] == 1000
    assert cell['discovery_power']['algorithm']['denominator'] == 1000
    assert record['full_V7'] is False
    with pytest.raises(certificate.FullScopePending):
        certificate.verify_full_certificate(registry, artifact_digest=made['artifact_digest'],
            expected_runtime_scope_hash=fingerprint(registry.plan.runtime_scope))
    assert wallet.verify() == before


def test_rehashed_complete_certificate_and_changed_availability_rejected(case):
    registry, _, _, _, _ = case
    made = certificate.produce_full_certificate(registry)
    path = registry.directory/'full_certificates'/(made['artifact_digest']+'.json')
    forged = json.loads(path.read_text())
    forged['full_V7'] = forged['formal_gate_eligible'] = True
    forged['pending_dependencies'] = []
    digest = fingerprint(forged)
    (path.parent/(digest+'.json')).write_text(json.dumps(forged))
    with pytest.raises(ValueError, match='Complete original certificate'):
        certificate.verify_full_certificate(registry, artifact_digest=digest,
            expected_runtime_scope_hash=fingerprint(registry.plan.runtime_scope))
    with pytest.raises(ValueError, match='availability'):
        certificate.verify_full_certificate(registry, artifact_digest=made['artifact_digest'],
            expected_runtime_scope_hash=fingerprint('different delay'))
    with pytest.raises(TypeError, match='no arrays'):
        certificate.reconstruct_full_certificate([{'loss': 0, 'success': True}])
