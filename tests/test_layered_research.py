"""Real lifecycle bookkeeping with synthetic loss fixtures; no market evidence."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path('D:/FactorResearch')
sys.path.insert(0, str(ROOT))
NAME = 'factorlab.discovery._layered_research_staged'
PATH = Path(__file__).resolve().parents[1] / 'factorlab/discovery/layered_research.py'
SPEC = importlib.util.spec_from_file_location(NAME, PATH)
m = importlib.util.module_from_spec(SPEC)
sys.modules[NAME] = m
SPEC.loader.exec_module(m)
from factorlab.discovery.confirmatory import paired_effect_interval
from factorlab.discovery.v7_lifecycle import Lifecycle, LifecyclePlan


@pytest.fixture
def research(tmp_path, monkeypatch):
    frozen = m.source_files()
    monkeypatch.setattr(m, 'source_files', lambda: frozen)
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    clock = [pd.Timestamp('2026-10-04T03:00:05Z')]
    spec = m.LayeredResearchSpec(m.fingerprint(frozen))
    plan = LifecyclePlan('actual-feed', training_window_days=90, refit_interval_days=15)
    lc = m.LayeredResearchLifecycle(tmp_path / 'run', plan, research_spec=spec, clock=lambda: clock[0])
    yield lc, clock, spec
    lc.close()


def synthetic_score(effect=.02):
    origins = pd.date_range('1970-01-01', periods=2160, freq='h', tz='UTC')
    result = paired_effect_interval(origins, np.ones(2160), np.full(2160, 1-effect))
    return {**result, 'formal_eligible': False}


def test_new_enrollment_allows_measurement_without_changing_legacy_gate(research):
    lc, _, _ = research
    assert lc.confirmation_execution_ready()
    assert not lc.calibration_ready()
    assert not lc.verify()['formal_gate_ready']
    assessment = lc.research_assessment(synthetic_score())
    assert assessment['status'] == 'eligible_for_unconfirmed_research_retention'
    assert assessment['whole_agent_power_certified'] is False
    assert assessment['market_alpha_certified'] is False
    assert lc.research_assessment(synthetic_score(0))['status'] == 'not_supported_by_this_endpoint'


@pytest.mark.parametrize('field,value', [('supported', True), ('formal_eligible', True),
    ('calibration_passed', True), ('quarantined_decision', {'accepted': True}),
    ('interval', [-.001, .02]), ('rule_id', 'foreign')])
def test_diagnostics_cannot_inherit_authority_or_forge_interval(research, field, value):
    lc, _, _ = research
    result = synthetic_score()
    result[field] = value
    with pytest.raises(ValueError):
        lc.research_assessment(result)


def test_deleted_or_changed_manifest_fails_without_recreation(research):
    lc, _, _ = research
    path = lc.directory / 'layered-research.json'
    original = path.read_bytes()
    path.write_text('{}')
    with pytest.raises(ValueError):
        lc.verify()
    path.write_bytes(original)
    path.unlink()
    with pytest.raises(ValueError):
        lc.verify()
    assert not path.exists()


def test_source_change_or_missing_enrollment_fails_closed(research, monkeypatch):
    lc, _, _ = research
    monkeypatch.setattr(m, 'source_files', lambda: {'changed': '0'*64})
    with pytest.raises(ValueError):
        lc.confirmation_execution_ready()


def test_old_lifecycle_cannot_be_retrofitted(research):
    lc, clock, spec = research
    old = Lifecycle(lc.directory.parent / 'legacy', lc.plan, clock=lambda: clock[0])
    old.due_fit_cutoff()  # An exposed old cadence anchor must never be converted.
    old.close()
    with pytest.raises(ValueError, match='fresh'):
        m.LayeredResearchLifecycle(lc.directory.parent / 'legacy', lc.plan,
                                    research_spec=spec, clock=lambda: clock[0])
    assert not (lc.directory.parent / 'legacy/layered-research.json').exists()


def test_plan_cannot_mix_legacy_certificates_or_relax_protocol(research):
    lc, clock, spec = research
    for override in ({'minimum_gain': .003}, {'refit_interval_days': 15.},
                     {'statistical_claim': 'whole_agent_certified'}):
        with pytest.raises(ValueError):
            m.LayeredResearchSpec(spec.source_hash, **override)
    mixed = LifecyclePlan('actual-feed', training_window_days=90, refit_interval_days=15,
                          calibration_directory='old', calibration_binding='old')
    with pytest.raises(ValueError, match='no legacy'):
        m.LayeredResearchLifecycle(lc.directory.parent / 'mixed', mixed,
                                    research_spec=spec, clock=lambda: clock[0])


def test_one_look_C_commit_recovery_and_causal_unconfirmed_retention(research, monkeypatch):
    lc, clock, _spec = research
    lc.due_fit_cutoff()
    lc.exposure.record(source=lc.plan.source_id, symbols=lc.plan.symbols,
                       start='2026-07-01T00:00:00Z', end='2026-10-01T00:00:00Z', role='S', batch='A')
    lc.freeze_batch(batch='A', ordinal=1, start='2026-11-01T00:00:00Z',
        candidates=[{'pipeline_id': 'factor-A', 'complexity': 3}], reference={'pipeline_id': 'K0'},
        evidence_sources=[{'start': '2026-07-01T00:00:00Z', 'end': '2026-10-01T00:00:00Z'}])
    def losses(stream, start, end, as_of, *, expected_pipeline=None):
        assert pd.Timestamp(end) <= pd.Timestamp(as_of)
        value = 1. if stream.endswith(':reference') else .98
        return {(hour, 'BTCUSDT'): (value, .01, 'fixed-revision')
                for hour in pd.date_range(start, end, inclusive='left', freq='h')}
    monkeypatch.setattr(lc, '_losses', losses)
    clock[0] = pd.Timestamp('2026-12-01T00:00:00Z')
    assert lc.select_validation('A')['selected'] is not None
    lc.reserve_selected_confirmation('A')
    clock[0] = pd.Timestamp('2027-03-02T12:00:00Z')
    result = lc.conclude_confirmation('A')
    assert result['supported'] is False and result['diagnostic_status'] == 'supported_target'
    monkeypatch.setattr(lc, '_losses', lambda *a, **k: pytest.fail('Recovery cannot look again'))
    assert lc.recover_confirmation('A') == result
    assert lc.verify()['formal_gate_ready'] is False
    assert lc.exposure.db.execute('SELECT COUNT(*) FROM confirmations WHERE consumed=1').fetchone()[0] == 1
    # Next elapsed fit cutoff after actual publication; earlier cutoffs retain K0.
    earlier = lc.fit_cutoff(lc._cadence_anchor())
    assert lc.adoption_pipeline(earlier, {'pipeline_id': 'K0'})['pipeline_id'] == 'K0'
    clock[0] = pd.Timestamp('2027-03-20T12:00:00Z')
    cutoff = lc.due_fit_cutoff()
    assert lc.adoption_pipeline(cutoff, {'pipeline_id': 'K0'})['pipeline_id'] == 'factor-A'
    # Pool/NN ticket validation uses this shared binding; it must permit
    # operational retention without inventing the old confirmed-policy seal.
    monkeypatch.setitem(sys.modules, 'factorlab.discovery.layered_research', m)
    typed_name = 'factorlab.discovery._layered_factor_typed_stage'
    typed_spec = importlib.util.spec_from_file_location(typed_name, PATH.parent/'typed_policy_confirmation.py')
    typed_module = importlib.util.module_from_spec(typed_spec)
    sys.modules[typed_name] = typed_module
    typed_spec.loader.exec_module(typed_module)
    monkeypatch.setitem(sys.modules, 'factorlab.discovery.typed_policy_confirmation', typed_module)
    contract_name = 'factorlab.discovery._layered_adoption_contract_stage'
    contract_spec = importlib.util.spec_from_file_location(contract_name,
        PATH.parent/'neural_forward_contract.py')
    contract = importlib.util.module_from_spec(contract_spec)
    sys.modules[contract_name] = contract
    contract_spec.loader.exec_module(contract)
    bound = contract._confirmed_adoption_binding(lc, cutoff, {'pipeline_id': 'factor-A'})
    assert bound['version'] == 'v7-unconfirmed-research-adoption-v1'
    assert bound['formal_gate_eligible'] is False
    assert bound['whole_agent_power_certified'] is False
    assert contract._confirmed_adoption_binding(lc, cutoff, {'pipeline_id': 'factor-A'},
        expected=bound, historical=True) == bound
    with pytest.raises(ValueError, match='differs'):
        contract._confirmed_adoption_binding(lc, cutoff, {'pipeline_id': 'factor-A'},
            expected={**bound, 'formal_gate_eligible': True}, historical=True)
    assert not lc.calibration_ready()
    assert lc.research_assessment(result)['formal_gate_eligible'] is False
