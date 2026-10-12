"""Original real runtime/wallet preregistration, never full statistical credit."""
import json
from dataclasses import replace

import pytest
from test_v7_run_factor_pool import source_root

from factorlab.discovery import full_calibration_registry as module
from factorlab.discovery.calibration_acceptance import AcceptanceCell, AcceptancePlan
from factorlab.discovery.calibration_runtime_contract import runtime_calibration_contract
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
from factorlab.discovery.neural_forward_ticket import _bind_authority
from factorlab.discovery.v7_forward import ForwardRuntime
from factorlab.protocol import fingerprint


@pytest.fixture
def case(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    baseline = ForwardRuntime(tmp_path / 'profile-runtime', tmp_path / 'profile-feed')
    wallet = MonthlyWorkAuthority(tmp_path / 'profile-wallet', run_id='profile',
        task='volatility', universe=['BTCUSDT'], partition_policy='standard')
    _bind_authority(baseline, wallet)
    try:
        scope = runtime_calibration_contract(baseline.lifecycle)
    finally:
        baseline.close()
    plan = module.FullTrialPlan(
        AcceptancePlan((AcceptanceCell('null', fingerprint('null'), 1000),
                        AcceptanceCell('mixed', fingerprint('mixed'), 1000, ('factor1',)))),
        scope, {'backend': 'codex', 'model_requested': 'gpt-6-astra',
                'reasoning_requested': 'high', 'context_policy': 'fresh_codex_per_trial_v1'},
        {'design_digest': fingerprint('design'), 'validation_digest': fingerprint('validation'),
         'oracle_digest': fingerprint('oracle')})
    registry = module.FullTrialRegistry(tmp_path / 'registry', plan)
    paths = registry.paths('null', 0)
    runtime = ForwardRuntime(paths['runtime'], paths['feed'])
    trial = registry.identity_for('null', 0)
    wallet = MonthlyWorkAuthority(paths['authority'],
        run_id=fingerprint({'registry': registry.digest, 'trial': trial['trial_id']}),
        task='volatility', universe=['BTCUSDT'], partition_policy='standard')
    try:
        yield registry, runtime, wallet
    finally:
        runtime.close()


def test_whole_grid_freezes_without_executing_or_materializing_all_trials(case):
    registry, runtime, wallet = case
    assert registry.verify()['planned_validation_repetitions'] == 2000
    assert module.FullTrialRegistry.inspect(registry.directory).digest == registry.digest
    assert wallet.verify()['reservations'] == 0
    assert not (runtime.directory.parent / 'enrollment.json').exists()
    assert registry.paths('null', 0) != registry.paths('null', 1)
    assert registry.paths('null', 0) != registry.paths('null', 0, oracle_index=0)
    assert (registry.identity_for('null', 0)['market_seed_material']
            != registry.identity_for('null', 0, oracle_index=0)['market_seed_material'])
    with pytest.raises(ValueError, match='outside'):
        registry.paths('null', 1000)


def test_actual_original_enrollment_no_budget_or_formal_permission_and_readonly_history(case):
    registry, runtime, wallet = case
    before = wallet.verify()
    permit = registry.enroll(runtime, wallet, 'null', 0)
    value = permit.verify(runtime.lifecycle)
    assert value['execution_scope'] == 'quarantined_simulation_only'
    assert value['formal_gate_eligible'] is False
    assert value['full_pipeline_execution_verified'] is False
    assert runtime.lifecycle.calibration_ready() is False
    assert wallet.verify() == before
    assert permit.verify() == value
    with permit.open_lifecycle() as reader:
        assert permit.verify(reader) == value
        with pytest.raises(Exception, match='readonly'):
            reader.db.execute("DELETE FROM operations")
    operations = runtime.lifecycle.db.execute('SELECT COUNT(*) FROM operations').fetchone()[0]
    assert registry.enroll(runtime, wallet, 'null', 0).verify() == value
    assert runtime.lifecycle.db.execute('SELECT COUNT(*) FROM operations').fetchone()[0] == operations


def test_permission_cannot_be_used_for_foreign_actual_lifecycle(case):
    registry, runtime, wallet = case
    permit = registry.enroll(runtime, wallet, 'null', 0)
    other = ForwardRuntime(registry.directory.parent / 'foreign', registry.directory.parent / 'other-feed')
    try:
        with pytest.raises(ValueError, match='foreign lifecycle'):
            permit.verify(other.lifecycle)
    finally:
        other.close()


def test_enrollment_after_real_planned_work_rejects_without_minting_marker(case):
    registry, runtime, wallet = case
    runtime.lifecycle.plan_fit('2026-10-01T00:00:00Z', 'A0', runtime.recipe_for())
    with pytest.raises(ValueError, match='before any production job'):
        registry.enroll(runtime, wallet, 'null', 0)
    assert not (runtime.directory.parent / 'enrollment_intent.json').exists()
    assert wallet.verify()['reservations'] == 0


def test_missing_original_enrollment_cannot_be_recreated_by_inspect_or_enroll(case):
    registry, runtime, wallet = case
    permit = registry.enroll(runtime, wallet, 'null', 0)
    path = runtime.directory.parent / 'enrollment.json'
    path.unlink()
    with pytest.raises(ValueError, match='missing'):
        permit.verify()
    with pytest.raises(ValueError, match='marker missing'):
        registry.enroll(runtime, wallet, 'null', 0)
    assert not path.exists()


def test_changed_source_and_original_enrollment_event_rejected(case):
    registry, runtime, wallet = case
    permit = registry.enroll(runtime, wallet, 'null', 0)
    event = runtime.lifecycle.db.execute("SELECT seq,record FROM operations WHERE "
        "json_extract(record,'$.kind')=?", (module.EVENT,)).fetchone()
    original = json.loads(event[1])
    changed = json.loads(event[1])
    changed['detail']['index'] = 1
    with runtime.lifecycle.db:
        runtime.lifecycle.db.execute('UPDATE operations SET record=?,hash=? WHERE seq=?',
                                    (json.dumps(changed), fingerprint(changed), event[0]))
    with pytest.raises(ValueError, match='enrollment|chain|scope'):
        permit.verify(runtime.lifecycle)
    with runtime.lifecycle.db:
        runtime.lifecycle.db.execute('UPDATE operations SET record=?,hash=? WHERE seq=?',
                                    (event[1], fingerprint(original), event[0]))
    (registry.directory.parent / 'factorlab/v7.py').write_text('version=8')
    with pytest.raises(ValueError, match='registry changed'):
        permit.verify()


def test_other_cell_namespace_is_rejected_before_enrollment(case):
    registry, runtime, wallet = case
    with pytest.raises(ValueError, match='exact quarantined trial'):
        registry.enroll(runtime, wallet, 'mixed', 0)
    assert not (runtime.directory.parent / 'enrollment.json').exists()


def test_standard_budget_and_separate_generator_domains_are_mandatory(case):
    registry, _runtime, _wallet = case
    value = registry.plan.to_dict()
    value['runtime_scope']['work_budget']['partition_policy'] = None
    with pytest.raises(ValueError, match='standard shared wallet'):
        module.FullTrialPlan.from_dict(value)
    with pytest.raises(ValueError, match='Separate original'):
        replace(registry.plan, generator_design={k: fingerprint('same')
                for k in ('design_digest', 'validation_digest', 'oracle_digest')})


def test_inspect_rejects_flipped_authenticity_or_clock_flags_and_never_rewrites(case):
    registry, _runtime, _wallet = case
    path = registry.directory / 'identity.json'
    original = path.read_bytes()
    changed = json.loads(original)
    changed['provider_authenticity_certified'] = True
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='changed or self-authorized'):
        module.FullTrialRegistry.inspect(registry.directory)
    assert json.loads(path.read_text())['provider_authenticity_certified'] is True
    path.write_bytes(original)
    changed['provider_authenticity_certified'] = False
    changed['work_clock'] = 'virtual_month_reset'
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='changed or self-authorized'):
        module.FullTrialRegistry.inspect(registry.directory)


@pytest.mark.parametrize('change', ['unchanged', 'lost_directory', 'tampered_receipt'])
def test_readonly_permit_checks_actual_nonempty_forecast_receipts(case, change):
    registry, runtime, wallet = case
    permit = registry.enroll(runtime, wallet, 'null', 0)
    ledger = runtime.lifecycle.forecasts
    ledger.put(run=runtime.lifecycle.plan.digest, stream='K0',
        origin='2026-10-01T00:00:00Z', symbol='BTCUSDT', task='volatility',
        model_id='explicit-artificial-model', pipeline_id='explicit-artificial-pipeline',
        prediction={'variance': .01})
    receipts = ledger.records_directory
    if change == 'lost_directory':
        receipts.rename(receipts.with_name('forecasts.records.preserved'))
    elif change == 'tampered_receipt':
        receipt = next(receipts.glob('*.json'))
        value = json.loads(receipt.read_text())
        value['receipt']['prediction']['variance'] = .02
        receipt.write_text(json.dumps(value))
    if change == 'unchanged':
        before = {p.name: p.read_bytes() for p in receipts.glob('*.json')}
        with permit.open_lifecycle() as reader:
            assert reader.forecasts.verify() == {'forecasts': 1}
            with pytest.raises(Exception, match='readonly'):
                reader.forecasts.db.execute('DELETE FROM forecasts')
        assert {p.name: p.read_bytes() for p in receipts.glob('*.json')} == before
    else:
        with pytest.raises(ValueError, match='forecast'), permit.open_lifecycle():
            pass
