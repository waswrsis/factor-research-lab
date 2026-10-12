"""Synthetic construction and fault injection; never statistical acceptance."""
import json
import os
import sqlite3
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_v7_run_factor_pool import source_root

from factorlab.discovery import full_pipeline_calibration as full
from factorlab.discovery.confirmatory import ConfirmationRule


@pytest.fixture
def plan():
    return full.FullCalibrationPlan(rule=ConfirmationRule(bootstrap_samples=1000), candidates=2)


def fake(plan, job, emit, experiment=None):
    emit({'event': 'solve_reserved', 'solves': 1})
    if job['kind'] == 'oracle':
        return {'risks': {slot: [1., .98] for slot in full.SLOTS}}
    return {'claims': [{'slot': slot, 'accepted': False, 'selected': 'fixture',
                        'base_mean': 1., 'candidate_mean': .98} for slot in full.SLOTS],
            'frozen_factors': {f'factor{i}': None for i in range(1, 4)}}


def test_seed_domains_and_unit_budgets(plan):
    jobs = list(full._jobs(plan))
    assert len(jobs) == 60
    assert len({tuple(j['seed_key']) for j in jobs}) == len(jobs)
    assert len({j['seed'] for j in jobs}) == len(jobs)
    assert plan.unit_seconds <= 300 and plan.memory_mb <= 768
    for phase in ('calibration', 'validation'):
        group = [j for j in jobs if j['phase'] == phase]
        assert {j['seed'] for j in group if j['kind'] == 'experiment'}.isdisjoint(
            j['seed'] for j in group if j['kind'] == 'oracle')


def test_incremental_work_never_promotes_test_callback(tmp_path, monkeypatch, plan):
    source_root(tmp_path, monkeypatch)
    run = tmp_path / 'full'
    first = full.run_full_pipeline_calibration(run, plan, evaluator=fake, evaluator_id='fixture')
    assert first['work_units'] == 1 and not first['complete']
    assert all(c['truth']['status'] == 'uncertain' for c in first['outcomes'][0]['claims'])
    result = full.run_full_pipeline_calibration(run, plan, evaluator=fake, evaluator_id='fixture', max_new_trials=100)
    assert result['complete'] and result['work_units'] == 60
    assert not result['formal_gate_eligible'] and not result['full_V7']
    assert result['test_only'] and not result['effect_truth_certified']
    assert not (run / 'certificate.json').exists()
    assert full.run_full_pipeline_calibration(run, plan, evaluator=fake, evaluator_id='fixture') == result
    assert (run / 'validation_freeze.json').exists()
    assert full.inspect_full_pipeline_calibration(run) == result


def test_crash_reservation_and_spending_are_not_retried(tmp_path, monkeypatch, plan):
    source_root(tmp_path, monkeypatch)
    run = tmp_path / 'crash'

    class Crash(BaseException):
        pass

    def crash(plan, job, emit, experiment=None):
        emit({'event': 'solve_reserved', 'solves': 7})
        raise Crash

    with pytest.raises(Crash):
        full.run_full_pipeline_calibration(run, plan, evaluator=crash, evaluator_id='same')
    result = full.run_full_pipeline_calibration(run, plan, evaluator=fake, evaluator_id='same')
    assert result['work_units'] == 2
    with sqlite3.connect(run / 'calibration.sqlite') as db:
        rows = list(full._read(db))
    original = next(r for r in rows if r['job']['kind'] == 'experiment')
    assert original['status'] == 'interrupted_consumed' and original['solves_reserved'] == 7
    assert next(r for r in rows if r['job']['kind'] == 'oracle')['status'] == 'failed_consumed'


@pytest.mark.parametrize('damage', ['checkpoint', 'journal', 'reservation', 'archive', 'source'])
def test_recovery_rejects_missing_or_changed_evidence(tmp_path, monkeypatch, plan, damage):
    source_root(tmp_path, monkeypatch)
    run = tmp_path / 'damage'
    full.run_full_pipeline_calibration(run, plan, evaluator=fake, evaluator_id='fixture')
    if damage in ('checkpoint', 'journal'):
        with sqlite3.connect(run / 'calibration.sqlite') as db:
            db.execute('DELETE FROM ' + ('units' if damage == 'checkpoint' else 'journal'))
    elif damage == 'reservation':
        next((run / 'reservations').glob('*.json')).write_text('{}')
    elif damage == 'archive':
        next((run / 'source' / 'factorlab').glob('*.py')).write_text('changed=1')
    else:
        next((tmp_path / 'factorlab').glob('*.py')).write_text('changed=1')
    with pytest.raises((ValueError, KeyError)):
        full.run_full_pipeline_calibration(run, plan, evaluator=fake, evaluator_id='fixture')


def test_oracle_gain_comes_from_fitted_risks_not_amplitude(plan):
    assert full.oracle_truth(plan, [[1., .98]] * 2, slot='algorithm')['status'] == 'uncertain'
    positive = full.oracle_truth(plan, [[1., .98]] * 128, slot='algorithm')
    assert positive['status'] == 'alternative' and positive['gain'] == pytest.approx(.02)
    assert not positive['certified_population_truth']
    assert full.oracle_truth(plan, [[1., .997]] * 128, slot='factor1')['status'] == 'null'
    assert full.oracle_truth(plan, [[1., .98], [1., 1.02]] * 64, slot='factor2')['status'] == 'uncertain'


def test_last_item_resource_overrun_rejected_before_result(plan, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(full.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(full.psutil, 'Process', lambda: SimpleNamespace(
        memory_info=lambda: SimpleNamespace(rss=128 * 2**20)))
    budget = full.UnitBudget(plan, lambda _: None)
    budget.consume('solve')
    clock[0] = 301.
    with pytest.raises(RuntimeError, match='wall'):
        budget.check()
    assert budget.solves == 1


def _fresh_cpu(tmp_path, plan, body):
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text(json.dumps(plan.to_dict()), encoding='utf-8')
    script = (
        'import json,sys\nfrom pathlib import Path\n'
        'from factorlab import protocol\n'
        'from factorlab.discovery import full_pipeline_calibration as full\n'
        f'root = Path({str(tmp_path)!r})\nprotocol.ROOT = root\n'
        'plan = full.FullCalibrationPlan.from_dict(json.loads((root / "plan.json").read_text()))\n'
        'assert "torch" not in sys.modules\n' + body + '\nassert "torch" not in sys.modules\n'
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    completed = subprocess.run([sys.executable, '-X', 'faulthandler', '-c', script], env=env, capture_output=True,
                               text=True, timeout=330, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_real_five_slot_smoke_has_distinct_pipeline_steps(tmp_path, monkeypatch, plan):
    source_root(tmp_path, monkeypatch)
    run = tmp_path / 'actual'
    _fresh_cpu(tmp_path, plan,
        'result = full.run_full_pipeline_calibration(root / "actual", plan, max_new_trials=1)\n'
        '(root / "result.json").write_text(json.dumps(result))')
    result = json.loads((tmp_path / 'result.json').read_text())
    assert not result['formal_gate_eligible'] and not result['full_V7']
    with sqlite3.connect(run / 'calibration.sqlite') as db:
        rows = list(full._read(db))
        events = [json.loads(payload)['payload'] for (payload,) in db.execute('SELECT value FROM journal')]
    assert len(rows) == 1
    assert rows[0]['status'] == 'complete', rows[0].get('error')
    actual = rows[0]['result']
    assert [c['slot'] for c in actual['claims']] == list(full.SLOTS)
    assert sum(e.get('event') == 'S_three_fold' for e in events) == 6
    assert sum(e.get('event') == 'V_shortlist_frozen' for e in events) == 3
    assert sum(e.get('event') == 'C_frozen' for e in events) == 3
    assert any(e.get('event') == 'cold_month' and e['library'] for e in events)
    assert len(actual['claims'][-1]['adoption_trace']) == 6
    assert all(c['formal_claim'] is False for c in actual['claims'])
    assert actual['elapsed_seconds'] < 300 and actual['peak_rss_bytes'] <= 768 * 2**20
    assert 0 < actual['solves'] <= 1600


def test_real_independent_oracle_cold_fits_original_formulas(tmp_path, monkeypatch, plan):
    source_root(tmp_path, monkeypatch)
    _fresh_cpu(tmp_path, plan,
        'jobs = list(full._jobs(plan))\n'
        'assert jobs[1]["kind"] == "oracle" and jobs[1]["seed"] != jobs[0]["seed"]\n'
        'programs = full.candidate_programs()[0]\n'
        'original = {"frozen_factors": {f"factor{i}": full.PoolMember(programs[(i-1)%3]).to_dict() for i in range(1,4)}}\n'
        'receipts = []\nresult = full.actual_work_unit(plan, jobs[1], receipts.append, experiment=original)\n'
        '(root / "oracle_result.json").write_text(json.dumps(result))\n'
        '(root / "oracle_receipts.json").write_text(json.dumps(receipts))\n'
        '(root / "oracle_original.json").write_text(json.dumps(original))')
    result = json.loads((tmp_path / 'oracle_result.json').read_text())
    receipts = json.loads((tmp_path / 'oracle_receipts.json').read_text())
    original = json.loads((tmp_path / 'oracle_original.json').read_text())
    assert set(result['risks']) == set(full.SLOTS)
    assert all(result['risks'][slot] is not None for slot in full.SLOTS)
    assert result['original_frozen_hash'] == full.fingerprint(original['frozen_factors'])
    assert sum(e.get('event') == 'cold_month' and e['member'] is not None for e in receipts) >= 9
    assert result['elapsed_seconds'] < 300 and result['peak_rss_bytes'] <= 768 * 2**20


def test_source_change_mid_callback_never_starts_another_unit(tmp_path, monkeypatch, plan):
    source_root(tmp_path, monkeypatch)
    calls = []

    def change(plan, job, emit, experiment=None):
        calls.append(job['id'])
        next((tmp_path / 'factorlab').glob('*.py')).write_text('changed_during_fit=1')
        return fake(plan, job, emit, experiment)

    with pytest.raises(ValueError, match='Source changed'):
        full.run_full_pipeline_calibration(tmp_path / 'change', plan, evaluator=change,
                                           evaluator_id='change', max_new_trials=2)
    assert len(calls) == 1


def test_algorithm_adopts_only_completed_future_C_whole_pipeline(plan, monkeypatch):
    members = {f'factor{i}': full.PoolMember(p) for i, p in enumerate(full.candidate_programs()[0][:3], 1)}
    monkeypatch.setattr(full, '_search', lambda data, start, plan, budget, emit, slot: (members[slot], []))
    monkeypatch.setattr(full, '_month', lambda *args, **kwargs: (['test-origin'], [1.], [.99]))
    monkeypatch.setattr(full, '_claim', lambda slot, pair, start, months, plan: {
        'slot': slot, 'accepted': slot in {'factor1', 'factor3'}, 'formal_claim': False})
    result = full._experiment(None, plan, None, lambda _: None)
    trace = result['claims'][-1]['adoption_trace']
    # Oct–Dec precede factor1's C endpoint plus36h; factor2 never passes.
    assert [r['pipeline_member'] for r in trace] == [None, None, None,
        members['factor1'].member_id, members['factor1'].member_id, members['factor3'].member_id]


def test_memory_limit_is_enforced_before_solve_reservation(plan, monkeypatch):
    monkeypatch.setattr(full.psutil, 'Process', lambda: SimpleNamespace(
        memory_info=lambda: SimpleNamespace(rss=769 * 2**20)))
    events = []
    budget = full.UnitBudget(plan, events.append)
    with pytest.raises(RuntimeError, match='memory_budget'):
        budget.consume('solve')
    assert budget.solves == 0 and not events


def test_publication_rechecks_wall_deadline_and_preserves_spending(tmp_path, monkeypatch, plan):
    source_root(tmp_path, monkeypatch)
    clock = [0.]
    monkeypatch.setattr(full.time, 'monotonic', lambda: clock[0])

    def late(plan, job, emit, experiment=None):
        result = fake(plan, job, emit, experiment)
        clock[0] = 301.
        return result

    run = tmp_path / 'late'
    full.run_full_pipeline_calibration(run, plan, evaluator=late, evaluator_id='late')
    with sqlite3.connect(run / 'calibration.sqlite') as db:
        row = next(full._read(db))
    assert row['status'] == 'failed_consumed' and row['solves_reserved'] == 1
    assert row['elapsed_seconds_charged'] >= 301 and 'wall_budget' in row['error']
    assert 'result' not in row


def test_oracle_conditional_mixture_excludes_no_selection_only_from_effect_bin(plan):
    plan = replace(plan, oracle_paths=128, validation_repetitions=3)
    rows = []

    def experiment(scenario, effect, repetition, gain, accepted, no_selection=False):
        first = full._job(plan, 1, scenario, effect, repetition, 0)
        claims = [{'slot': slot, 'accepted': accepted,
                   'selected': None if no_selection and slot.startswith('factor') else 'selected'}
                  for slot in full.SLOTS]
        rows.append({'job': first, 'status': 'complete', 'result': {'claims': claims}})
        for unit in range(1, 129):
            rows.append({'job': full._job(plan, 1, scenario, effect, repetition, unit), 'status': 'complete',
                         'result': {'risks': {slot: [1., 1 - gain] for slot in full.SLOTS}}})

    experiment(0, 0, 0, 0., False)
    experiment(1, 3, 0, .01, True)
    experiment(1, 3, 1, .01, False)
    experiment(1, 3, 2, 0., False, no_selection=True)
    report = full._report(plan, iter(sorted(rows, key=lambda r: r['job']['id'])), test_only=True)
    point = next(p for p in report['validation_mixture_conditional_power']
                 if p['slot'] == 'factor1' and p['learner_effect_target'] == .01)
    assert point['n'] == 2 and point['accepted'] == 1 and point['rate'] == .5
    null = next(p for p in report['validation_scenario_conditional_power']
                if p['scenario'] == 'global_null_design' and p['slot'] == 'factor1'
                and p['learner_effect_target'] == .01)
    assert null['status'] == 'unavailable' and null['n'] == 0 and null['lower'] is None
    cell = next(c for c in report['curve'] if c['phase'] == 'validation'
                and c['scenario'] == 'mixed_regime_missing' and c['amplitude'] == .01)
    discovery = next(p for p in cell['discovery_acceptance'] if p['slot'] == 'factor1')
    assert discovery['n'] == 3 and discovery['accepted'] == 1 and discovery['rate'] == pytest.approx(1/3)
    assert not report['power_is_end_to_end_discovery_probability']
    assert 'unsupported_effect_bins' in report['reasons']
    assert not report['formal_gate_eligible'] and not report['full_V7']
