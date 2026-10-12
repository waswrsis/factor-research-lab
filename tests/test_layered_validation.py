"""Staged semantic checks; no certificate, provider or market performance."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path('D:/FactorResearch')
sys.path.insert(0, str(ROOT))
PATH = Path(__file__).resolve().parents[1] / 'factorlab/discovery/layered_validation.py'
NAME = 'factorlab.discovery._layered_validation_staged'
SPEC = importlib.util.spec_from_file_location(NAME, PATH)
m = importlib.util.module_from_spec(SPEC)
sys.modules[NAME] = m
SPEC.loader.exec_module(m)


def test_boundary_is_null_under_actual_binary64_loss_law():
    plan = m.LayeredPlan(repetitions=20)
    for scenario in m.SCENARIOS:
        boundary = m.law_risks(plan, scenario, '0.005')
        assert not any(r['alternative'] for claim in boundary for r in claim)
        assert all(r['relative_gain'] <= .005 for claim in boundary for r in claim)
        positive = m.law_risks(plan, scenario, '0.01')
        assert all(sum(r['alternative'] for r in claim) == 4 for claim in positive)


def test_generator_risk_enumerates_its_ordered_binary64_operations():
    from fractions import Fraction
    plan = m.LayeredPlan(repetitions=20)
    center = m._centers('0.02')
    amplitude = .002 * 1.25
    values = [float(float(center + common * amplitude) + individual * .001)
              for common in (-1, 1) for individual in (-1, 1)]
    exact = sum(map(Fraction.from_float, values), Fraction()) / 4
    risk = m.law_risks(plan, 'serial', '0.02')[0][0]['candidate']
    assert Fraction(*risk) == exact
    assert m._stage(np.random.default_rng(1), plan, 'serial', '0.02', 720).shape == (720, 5, 24)


def test_phase_domain_and_law_differ_before_any_trial():
    design = m.LayeredPlan(repetitions=20, phase='design')
    validation = m.LayeredPlan(repetitions=20, phase='validation')
    assert design.digest != validation.digest
    assert not np.array_equal(m._amplitudes(design, 'regime', 90),
                              m._amplitudes(validation, 'regime', 90))


def test_missing_all_trials_remain_in_registered_denominator():
    plan = m.LayeredPlan(repetitions=20)
    report = m.summarize(plan, [])
    assert report['registered_trials'] == 3 * 6 * 20
    assert report['retained_trials'] == 0
    assert all(c['possible_family_errors'] == c['denominator'] == 20 for c in report['cells'])
    assert all(v['lower'] == 0 for c in report['cells'] for v in c['power'].values())
    assert not report['criteria_met'] and not report['formal_gate_eligible']


def test_deadline_failure_is_consumed_without_shortening_or_new_clock():
    plan = m.LayeredPlan(repetitions=20)
    result = m.trial(plan, 'serial', '0.01', 0, clock=lambda: 2., deadline=1.)
    assert result['state'] == 'failed_consumed'
    assert result['possible_family_error'] is True
    assert result['discovery_successes'] == [False] * 5
    report = m.summarize(plan, [result])
    cell = next(c for c in report['cells'] if c['scenario'] == 'serial' and c['effect'] == '0.01')
    assert cell['failures_in_denominator'] == 20
    assert all(v['denominator'] == 20 and v['successes'] == 0 for v in cell['power'].values())


def test_duplicate_and_changed_trial_are_rejected():
    plan = m.LayeredPlan(repetitions=20)
    row = m.trial(plan, 'serial', '0', 0, clock=lambda: 1., deadline=0.)
    with pytest.raises(ValueError, match='duplicate'):
        m.summarize(plan, [row, row])
    with pytest.raises(ValueError, match='changed'):
        m.summarize(plan, [{**row, 'plan_digest': '0' * 64}])


def test_actual_confirmation_kernel_used_and_selected_risk_retained():
    plan = m.LayeredPlan(repetitions=20)
    result = m.trial(plan, 'serial', '0.02', 0)
    assert result['state'] == 'complete'
    assert len(result['claims']) == 5
    for claim in result['claims']:
        assert claim['selected'] is not None
        assert claim['known_law_selected_risk']['alternative']
        assert claim['score_readout']['rule_id'] == plan.rule.digest
        assert claim['score_readout']['calibration_passed'] is False
        assert claim['score_readout']['status'] == 'uncalibrated_diagnostic'


def test_one_synthetic_execution_fault_is_not_dropped_or_retried(monkeypatch):
    plan = m.LayeredPlan(repetitions=20)
    class RNG:
        def integers(self, *args):
            return 0
    monkeypatch.setattr(m.np.random, 'default_rng', lambda *args: RNG())
    result = m.trial(plan, 'execution_failures', 'mixed', 0)
    assert result['state'] == 'failed_consumed'
    assert result['claims'] == [] and result['possible_family_error']
    assert result['reason'].startswith('OSError:')


def test_scope_cannot_impersonate_a_full_policy_certificate():
    plan = m.LayeredPlan(repetitions=20)
    report = m.summarize(plan, [])
    assert report['scope'] == 'frozen_candidate_loss_scoring_and_selection'
    assert not report['full_V7_five_claim_pipeline_calibrated']
    assert not report['production_certificate_issued']
    assert 'whole_autonomous_algorithm_power' in report['excluded']
    assert 'algorithm' not in m.CLAIMS
    assert report['failure_loss'] == 2**20
    assert plan.rule.delta0 == .005 and plan.rule.alpha == .01
    assert plan.rule.bootstrap_samples == 10000


def test_high_noise_is_an_explicit_new_bound_scope():
    low = m.LayeredPlan(repetitions=20)
    high = m.LayeredPlan(repetitions=20, noise_multiplier=10.)
    assert low.digest != high.digest
    assert m._amplitudes(high, 'serial', 90)[0] == 10 * m._amplitudes(low, 'serial', 90)[0]
    assert m.summarize(high, [])['law_assumptions']['noise_multiplier'] == 10.
    assert m.summarize(high, [])['law_assumptions']['real_market_law_match_verified'] is False
    family = m.LayeredPlan(repetitions=20, noise_multiplier=10., confidence_family_size=3)
    assert m.summarize(family, [])['tail_alpha'] == m.summarize(high, [])['tail_alpha'] / 3


def test_readback_recomputes_and_rejects_changed_decision_and_missing_trial(tmp_path):
    import json
    plan = m.LayeredPlan(repetitions=20, phase='design', wall_seconds=1)
    directory = tmp_path / 'finite'
    original = m.run(directory, plan)
    assert m.verify_report(directory) == original
    report = directory / 'report.json'
    content = json.loads(report.read_text('utf-8'))
    content['formal_gate_eligible'] = True
    report.write_text(json.dumps(content), encoding='utf-8')
    with pytest.raises(ValueError, match='reproduced'):
        m.verify_report(directory)
    report.write_text(json.dumps(original), encoding='utf-8')
    trial = next(p for p in directory.glob('serial-*.json'))
    saved = trial.read_bytes()
    trial.unlink()
    with pytest.raises(ValueError, match='reproduced'):
        m.verify_report(directory)
    trial.write_bytes(saved)
    assert m.verify_report(directory) == original
    with pytest.raises(FileExistsError):
        m.run(directory, plan)


def test_cli_planning_does_not_create_run_or_call_worker(tmp_path, monkeypatch, capsys):
    import json
    from dataclasses import asdict
    monkeypatch.setitem(sys.modules, 'factorlab.discovery.layered_validation', m)
    script = Path(__file__).resolve().parents[1] / 'scripts/v7_layered_validation.py'
    spec = importlib.util.spec_from_file_location('_layered_cli_test', script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text(json.dumps(asdict(m.LayeredPlan(repetitions=20))), encoding='utf-8')
    target = tmp_path / 'not-created'
    assert cli.main(['plan', '--plan', str(plan_path), '--directory', str(target)]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value['status'] == 'PLANNING_ONLY' and value['work_started'] is False
    assert value['GPT_calls'] == value['fits'] == 0 and not target.exists()
    plan_path.write_text('{"repetitions": 19}', encoding='utf-8')
    with pytest.raises(ValueError, match='protocol'):
        cli.main(['run', '--plan', str(plan_path), '--directory', str(target)])
    assert not target.exists()
