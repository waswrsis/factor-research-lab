"""Bounded original registry adapters; no policy/provider execution is claimed."""
import pytest
from test_conditional_cold_refit import case
from factorlab.discovery.adaptive_policy_execution import read_policy_execution, advance_original_policy
from factorlab.discovery.typed_cold_selection import _runtime_reader, select_typed_neural
from factorlab.discovery.full_calibration_registry import QuarantinedTrialPermit


def test_whole_original_policy_denominator_and_no_new_enrollment(case):
    _, registry, _, validation, _, _ = case
    before = sorted(str(p) for p in registry.directory.rglob('enrollment.json'))
    report = read_policy_execution(registry, 'cold', 0)
    assert report['registered_paths'] == 128
    assert [p['atom'] for p in report['paths']] == list(range(128))
    assert {p['state'] for p in report['paths']} == {'missing_original_enrollment'}
    assert report['complete_execution_paths'] == 0
    assert report['full_adaptive_risk_truth'] is None and not report['formal_gate_eligible']
    assert before == sorted(str(p) for p in registry.directory.rglob('enrollment.json'))
    with pytest.raises(TypeError, match='oracle-index'):
        advance_original_policy(validation)
    with pytest.raises(ValueError, match='Bounded'):
        advance_original_policy(validation, max_pumps=0)
    # Original unregistered path must fail rather than receive a new wallet.
    original = QuarantinedTrialPermit(registry, registry.identity_for('cold', 0, oracle_index=0))
    with pytest.raises((ValueError, FileNotFoundError)):
        advance_original_policy(original)
    assert before == sorted(str(p) for p in registry.directory.rglob('enrollment.json'))


def test_original_runtime_readonly_no_repair_and_public_typed_selector(case):
    _, registry, _, validation, _, _ = case
    before = sorted(str(p) for p in validation.paths['root'].rglob('*'))
    with _runtime_reader(validation) as runtime:
        runtime.lifecycle.verify(); runtime.feed.verify()
        assert runtime.directory == validation.paths['runtime']
        assert runtime.feed.directory == validation.paths['feed']
    assert before == sorted(str(p) for p in validation.paths['root'].rglob('*'))
    with pytest.raises(ValueError, match='name its actual'):
        select_typed_neural(validation, 'typed_neural_caller_winner')
    with pytest.raises((ValueError, FileNotFoundError)):
        select_typed_neural(validation, 'typed_neural_'+'0'*64)
    assert before == sorted(str(p) for p in validation.paths['root'].rglob('*'))
