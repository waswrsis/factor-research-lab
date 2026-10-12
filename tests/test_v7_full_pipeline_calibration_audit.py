"""Independent adversarial accounting tests; no market or acceptance evidence."""
import itertools
import json
import sqlite3
from dataclasses import replace

import pytest
from test_v7_full_pipeline_calibration import fake
from test_v7_run_factor_pool import source_root

from factorlab.discovery import full_pipeline_calibration as full
from factorlab.discovery.confirmatory import ConfirmationRule


@pytest.fixture
def fixture_run(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    plan = full.FullCalibrationPlan(rule=ConfirmationRule(bootstrap_samples=1000), candidates=2)
    return tmp_path / 'full-audit', plan


def run(directory, plan, **kwargs):
    return full.run_full_pipeline_calibration(directory, plan, evaluator=fake,
                                               evaluator_id='independent-fixture', **kwargs)


def test_missing_identity_cannot_rebind_old_units_to_new_seed(fixture_run):
    directory, plan = fixture_run
    run(directory, plan)
    (directory / 'identity.json').unlink()
    with pytest.raises((ValueError, FileNotFoundError), match='identity|binding|Missing|missing|frozen'):
        run(directory, replace(plan, seed=plan.seed + 1))


@pytest.mark.parametrize('damage', ['missing', 'changed'])
def test_completed_resume_still_verifies_independent_validation_freeze(fixture_run, damage):
    directory, plan = fixture_run
    result = run(directory, plan, max_new_trials=100)
    assert result['complete']
    freeze = directory / 'validation_freeze.json'
    if damage == 'missing':
        freeze.unlink()
    else:
        value = json.loads(freeze.read_text('utf-8'))
        value['calibration_hash'] = '0' * 64
        freeze.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises((ValueError, FileNotFoundError), match='freeze|validation|Missing|missing|changed'):
        run(directory, plan)


def test_existing_report_prevents_refunding_lost_checkpoint_and_reservation(fixture_run):
    directory, plan = fixture_run
    run(directory, plan)
    assert list((directory / 'reports').glob('*.json'))
    with sqlite3.connect(directory / 'calibration.sqlite') as db:
        db.execute('DELETE FROM units')
        db.execute('DELETE FROM journal')
    for path in (directory / 'reservations').glob('*.json'):
        path.unlink()
    with pytest.raises(ValueError, match='checkpoint|report|reservation|missing|consumed'):
        run(directory, plan)


def test_failed_oracle_paths_cannot_be_dropped_to_certify_survivor_risk(fixture_run):
    _, base_plan = fixture_run
    plan = replace(base_plan, oracle_paths=129)
    jobs = list(itertools.islice(full._jobs(plan), 130))
    rows = [{'job': jobs[0], 'status': 'complete', 'result': {
        'claims': [{'slot': slot, 'accepted': True, 'selected': 'original-frozen-formula'}
                   for slot in full.SLOTS]}}]
    # The 128 easy paths all support +2%; the difficult path did not produce a
    # risk. Its unobserved risk can be arbitrarily adverse. Conditioning on
    # successful fits cannot establish the original path distribution's effect.
    rows.extend({'job': job, 'status': 'complete',
                 'result': {'risks': {slot: [1., .98] for slot in full.SLOTS}}}
                for job in jobs[1:-1])
    rows.append({'job': jobs[-1], 'status': 'failed_consumed', 'error': 'resource_or_numerical_failure'})
    report = full._report(plan, rows, test_only=True)
    outcome = report['outcomes'][0]
    assert outcome['all_oracles_terminal']
    assert all(claim['truth']['status'] == 'uncertain' for claim in outcome['claims'])
    assert outcome['possible_false_trial']


def test_five_duplicate_slot_names_are_not_five_claims(fixture_run):
    directory, plan = fixture_run

    def duplicates(plan, job, emit, experiment=None):
        emit({'event': 'solve_reserved', 'solves': 1})
        return {'claims': [{'slot': 'factor1', 'accepted': True}] * 5}

    result = full.run_full_pipeline_calibration(directory, plan, evaluator=duplicates,
                                                evaluator_id='duplicates')
    with sqlite3.connect(directory / 'calibration.sqlite') as db:
        rows = list(full._read(db))
    assert rows[0]['status'] == 'failed_consumed'
    assert rows[0]['solves_reserved'] == 1
    assert result['outcomes'][0]['possible_false_trial']
    assert not result['formal_gate_eligible'] and not result['full_V7']


def test_completed_synthetic_run_never_publishes_formal_certificate(fixture_run):
    directory, plan = fixture_run
    result = run(directory, plan, max_new_trials=100)
    assert result['complete'] and result['test_only']
    assert not result['formal_gate_eligible'] and not result['full_V7']
    assert not result['effect_truth_certified']
    assert not list(directory.rglob('*certificate*.json'))
