import functools
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from test_v7_run_factor_pool import source_root

from factorlab.discovery import pipeline_calibration as calibration
from factorlab.discovery.confirmatory import ConfirmationRule
from factorlab.protocol import fingerprint


def _fresh_calibration_process(function):
    """Numerical CPU cases retain their 768MiB guard in a fresh interpreter.

    The complete legacy suite loads neural weights into its parent process.
    That parent is not the CPU runtime being validated by these three cases.
    """
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        key = 'FACTOR_V7_ISOLATED_CALIBRATION_CASE'
        if os.environ.get(key) == function.__name__:
            return function(*args, **kwargs)
        environment = {**os.environ, key: function.__name__, 'CUDA_VISIBLE_DEVICES': '',
                       'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        base = Path('work/v7c/ci') / f't-{time.time_ns()}'
        base.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
            f'tests/test_v7_pipeline_calibration.py::{function.__name__}', '--basetemp',
            str(base)],
            env=environment, capture_output=True, text=True, timeout=180, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
    return wrapped


def fake_evaluator(plan, job, emit):
    emit({'event': 'synthetic_test_step', 'seed': job['seed']})
    return {'claims': [{'slot': i, 'family': 'null' if i in (0, 3) else 'signal',
                       'truth': 'exact_null' if i in (0, 3) else 'unidentified',
                       'accepted': False, 'effect_estimate': None} for i in range(5)]}


@pytest.fixture
def fast_plan():
    return calibration.CalibrationPlan(calibration_repetitions=1, validation_repetitions=1,
                                      rule=ConfirmationRule(bootstrap_samples=1000))


def test_complete_test_job_seed_separation_resume_and_no_certificate_promotion(tmp_path, monkeypatch, fast_plan):
    source_root(tmp_path, monkeypatch)
    directory = tmp_path / 'job'
    arguments = {'evaluator': fake_evaluator, 'evaluator_id': 'test-v1', 'max_new_trials': 30}
    result = calibration.run_pipeline_calibration(directory, fast_plan, **arguments)
    assert result['complete'] and result['trials'] == 30
    assert not result['formal_gate_eligible'] and not result['effect_truth_calibrated']
    repeated = calibration.run_pipeline_calibration(directory, fast_plan, **arguments)
    assert repeated == result
    verified = calibration.verify_calibration_certificate(directory, expected_binding=result['binding_id'])
    assert verified == result
    with sqlite3.connect(directory / 'calibration.sqlite') as db:
        rows = list(calibration._read_trials(db))
    seeds = {phase: {row['job']['seed'] for row in rows if row['job']['phase'] == phase}
             for phase in ('calibration', 'validation')}
    assert len(seeds['calibration']) == 15 and len(seeds['validation']) == 15
    assert seeds['calibration'].isdisjoint(seeds['validation'])
    assert all(row['receipt_count'] == 1 for row in rows)
    freeze = json.loads((directory / 'validation_freeze.json').read_text('utf-8'))
    assert freeze['rule'] == fast_plan.rule.to_dict()
    assert 'held_out' in freeze['nuisance_settings']


def test_reservation_and_step_persist_before_work_interruption_is_not_retried(tmp_path, monkeypatch, fast_plan):
    source_root(tmp_path, monkeypatch)
    directory = tmp_path / 'job'

    class Crash(BaseException):
        pass

    seen = []

    def interrupted(plan, job, emit):
        with sqlite3.connect(directory / 'calibration.sqlite') as db:
            value = json.loads(db.execute('SELECT value FROM trials WHERE id=?', (job['id'],)).fetchone()[0])
        assert value['status'] == 'reserved'
        seen.append(job['id'])
        emit({'event': 'reserved_before_work'})
        raise Crash

    with pytest.raises(Crash):
        calibration.run_pipeline_calibration(directory, fast_plan, evaluator=interrupted, evaluator_id='failure-v1')
    result = calibration.run_pipeline_calibration(directory, fast_plan, evaluator=fake_evaluator,
                                                  evaluator_id='failure-v1')
    assert result['trials'] == 2
    with sqlite3.connect(directory / 'calibration.sqlite') as db:
        rows = list(calibration._read_trials(db))
    prior = next(r for r in rows if r['job']['id'] == seen[0])
    assert prior['status'] == 'interrupted_consumed'
    assert prior['receipt_count'] == 1
    assert sum(c['resource_or_interrupted_failures'] for c in result['curve']) == 1


@pytest.mark.parametrize('corruption', ['checkpoint_missing', 'receipt_missing', 'receipt_changed'])
def test_corrupt_recovery_evidence_is_rejected(tmp_path, monkeypatch, fast_plan, corruption):
    source_root(tmp_path, monkeypatch)
    directory = tmp_path / 'job'
    arguments = {'evaluator': fake_evaluator, 'evaluator_id': 'test-v1'}
    calibration.run_pipeline_calibration(directory, fast_plan, **arguments)
    with sqlite3.connect(directory / 'calibration.sqlite') as db:
        if corruption == 'checkpoint_missing':
            db.execute('DELETE FROM trials')
        elif corruption == 'receipt_missing':
            db.execute('DELETE FROM receipts')
        else:
            db.execute('UPDATE receipts SET value=?', ('{}',))
    with pytest.raises((ValueError, KeyError), match='.*'):
        calibration.run_pipeline_calibration(directory, fast_plan, **arguments)


def test_certificate_gate_cannot_be_forged_even_with_new_payload_hash(tmp_path, monkeypatch, fast_plan):
    source_root(tmp_path, monkeypatch)
    directory = tmp_path / 'job'
    calibration.run_pipeline_calibration(directory, fast_plan, evaluator=fake_evaluator,
                                          evaluator_id='test-v1', max_new_trials=30)
    path = directory / 'certificate.json'
    value = json.loads(path.read_text('utf-8'))
    value['report']['formal_gate_eligible'] = True
    value['formal_gate_eligible'] = True
    value['digest'] = fingerprint({k: v for k, v in value.items() if k != 'digest'})
    path.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError, match='reproduced'):
        calibration.verify_calibration_certificate(directory)


def test_code_and_protocol_binding_cannot_change_on_resume(tmp_path, monkeypatch, fast_plan):
    source_root(tmp_path, monkeypatch)
    directory = tmp_path / 'job'
    calibration.run_pipeline_calibration(directory, fast_plan, evaluator=fake_evaluator, evaluator_id='test-v1')
    (tmp_path / 'factorlab' / 'v7.py').write_text('version=8', encoding='utf-8')
    with pytest.raises(ValueError, match='Immutable'):
        calibration.run_pipeline_calibration(directory, fast_plan, evaluator=fake_evaluator, evaluator_id='test-v1')


def test_validation_freeze_is_verified_not_just_present(tmp_path, monkeypatch, fast_plan):
    source_root(tmp_path, monkeypatch)
    directory = tmp_path / 'job'
    calibration.run_pipeline_calibration(directory, fast_plan, evaluator=fake_evaluator,
                                          evaluator_id='test-v1', max_new_trials=30)
    path = directory / 'validation_freeze.json'
    value = json.loads(path.read_text('utf-8'))
    value['validation_seed_domain'] = [fast_plan.seed, 0]
    path.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError, match='validation freeze'):
        calibration.verify_calibration_certificate(directory)


@_fresh_calibration_process
def test_all_candidate_failures_remain_attributable(monkeypatch, fast_plan):
    def fail(*args, **kwargs):
        raise ValueError('injected_formula_support_failure')
    monkeypatch.setattr(calibration, 'evaluate_formula', fail)
    events = []
    result = calibration.actual_pipeline_trial(fast_plan, next(calibration._jobs(fast_plan)), events.append)
    assert all(c['outcome'] == 'all_candidates_failed' and not c['accepted'] for c in result['claims'])
    assert sum(r['event'] == 'candidate_failed' for r in events) == 6
    assert all(c['failures'] for c in result['claims'])


def test_binomial_unit_is_whole_five_claim_trial_and_failures_are_conservative(fast_plan):
    job = next(calibration._jobs(fast_plan))
    claims = fake_evaluator(fast_plan, job, lambda _: None)['claims']
    for claim in claims:
        claim['accepted'] = True
    rows = [{'job': job, 'status': 'complete', 'result': {'claims': claims}},
            {'job': job, 'status': 'failed_consumed'}]
    report = calibration._aggregate(fast_plan, rows, test_only=True)
    group = next(c for c in report['curve'] if c['repetitions'])
    assert group['repetitions'] == 2
    assert group['known_null_false_trials'] == 1
    assert group['conservative_possible_false_trials'] == 2
    assert group['acceptance_rate'] == .5
    assert not report['formal_gate_eligible']


def test_ohlc_generator_is_causal_bounded_and_validation_nuisance_is_held_out():
    bars = calibration.synthetic_bars(seed=12, scenario='regime', effect_target=.01)
    duplicate = calibration.synthetic_bars(seed=12, scenario='regime', effect_target=.01)
    np.testing.assert_array_equal(bars.close, duplicate.close)
    assert len(bars) < 9000
    assert (bars.high >= bars[['open', 'close']].max(axis=1)).all()
    assert (bars.low <= bars[['open', 'close']].min(axis=1)).all()
    validation = calibration.synthetic_bars(seed=12, scenario='regime', effect_target=.01, phase='validation')
    assert not np.array_equal(bars.close, validation.close)
    missing = calibration.synthetic_bars(seed=12, scenario='missing_history', effect_target=.01)
    assert len(missing) == len(bars) - 3


@_fresh_calibration_process
def test_actual_ohlc_fits_S_V_and_three_cold_C_months(monkeypatch, fast_plan):
    # Force a V winner only to cover the C fitting branch with tiny smoke data.
    # All numerical fits, features, labels and paired intervals remain real;
    # this override is a test fixture, never a calibration certificate.
    monkeypatch.setattr(calibration, 'select_on_validation', lambda rows: rows[0] if rows else None)
    events = []
    job = next(calibration._jobs(fast_plan))
    result = calibration.actual_pipeline_trial(fast_plan, job, events.append)
    assert result['evaluator'] == 'actual_pipeline'
    assert 0 < result['solves'] < fast_plan.max_solves_per_trial
    assert not result['effect_is_known_truth']
    stages = [r['event'] for r in events]
    assert stages.count('candidate_S') == 6
    assert stages.count('C_month_fitted') == 6  # Shared family winner for five dependent slots.
    assert stages.count('C_fixed_endpoint') == 2
    assert all(len(c['monthly']) == 3 for c in result['claims'])
    assert all(c['interval']['calibration_passed'] is False for c in result['claims'])
    assert abs(result['claims'][0]['effect_estimate']) < 1e-10
    reservations = [r['solves'] for r in events if r['event'] == 'solve_reserved']
    assert reservations == list(range(1, result['solves'] + 1))


@_fresh_calibration_process
def test_budget_exhaustion_returns_explicit_failure_and_consumes_trial(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    plan = calibration.CalibrationPlan(calibration_repetitions=1, validation_repetitions=1,
                                        max_solves_per_trial=1, rule=ConfirmationRule(bootstrap_samples=1000))
    report = calibration.run_pipeline_calibration(tmp_path / 'job', plan)
    assert report['trials'] == 1 and not report['formal_gate_eligible']
    with sqlite3.connect(tmp_path / 'job' / 'calibration.sqlite') as db:
        row = next(calibration._read_trials(db))
    assert row['status'] == 'failed_consumed'
    assert row['error'] == 'solve_limit'
    assert row['receipt_count'] >= 2
