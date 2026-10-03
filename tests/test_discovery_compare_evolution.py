"""Compare complete sequential procedures without trimming inconvenient origins."""
import json
import sqlite3
from dataclasses import replace

import numpy as np
import pytest
from test_discovery_evolution import fixture, good_builder, run

from factorlab.discovery.compare_evolution import compare_evolutions
from factorlab.discovery.contracts import TaskDataset, TaskSpec
from factorlab.discovery.evolution import EvolutionPolicy
from factorlab.discovery.learning import fit


def pair(tmp_path, *, evolve=False, candidate_policy=None):
    data, spec, model = fixture()
    fixed = EvolutionPolicy(mode='fixed', shadow_hours=16, max_cycles=1, update_every_hours=24)
    candidate_policy = candidate_policy or replace(fixed, mode='evolve' if evolve else 'fixed')
    run(tmp_path, 'reference', data, spec, model, policy=fixed)
    run(tmp_path, 'candidate', data, spec, model, policy=candidate_policy,
        **({'candidate_builder': good_builder} if evolve else {}))
    return tmp_path / 'candidate', tmp_path / 'reference', tmp_path / 'comparison.json'


def test_identical_full_outputs_have_zero_improvement_and_no_confirmation(tmp_path):
    paths = pair(tmp_path)
    result = compare_evolutions(*paths)
    assert result['primary_comparison']['estimate'] == 0
    assert result['primary_comparison']['status'] == 'target_not_met'
    assert result['primary_comparison']['family_size'] == 3
    assert result['predictive_claim_supported'] is False
    assert result['independent_confirmation'] is False
    assert result['equal_actual_resource_use_claimed'] is False
    assert result['equal_policy_resource_ceilings'] is True
    assert result['paired_forecast_samples'] == 81
    assert result['per_symbol']['TEST']['return']['support'] == 80
    assert sum(segment['tasks']['return']['total_rows'] for segment in result['calendar_segments']) == 81
    with pytest.raises(ValueError, match='already exists'):
        compare_evolutions(*paths)


def test_known_signal_improves_whole_procedure_after_gate_not_shadow_only(tmp_path):
    paths = pair(tmp_path, evolve=True)
    result = compare_evolutions(*paths)
    primary = result['primary_comparison']
    assert primary['estimate'] > .5
    assert primary['operational_development_improvement_supported'] is True
    assert primary['predictive_claim_supported'] is False
    assert result['candidate']['resources']['promotions'] == 1
    assert result['candidate']['resources']['cycles_attempted'] == 1
    # Pre-promotion incumbent forecasts remain, reducing whole-procedure improvement.
    assert primary['estimate'] < 1
    assert result['policy_differences']['mode'] == {'candidate': 'evolve', 'reference': 'fixed'}


def test_failed_research_retains_all_hours_in_procedure_comparison(tmp_path):
    data, spec, model = fixture()
    fixed = EvolutionPolicy(mode='fixed', shadow_hours=16, max_cycles=1)
    run(tmp_path, 'reference', data, spec, model, policy=fixed)
    def failed(*_args):
        raise RuntimeError('controlled failed candidate')
    run(tmp_path, 'candidate', data, spec, model, policy=replace(fixed, mode='evolve'),
        candidate_builder=failed)
    result = compare_evolutions(tmp_path / 'candidate', tmp_path / 'reference', tmp_path / 'out.json')
    assert result['primary_comparison']['estimate'] == 0
    assert result['paired_forecast_samples'] == 81
    assert result['candidate']['resources']['failed_or_interrupted_cycles'] == 1


@pytest.mark.parametrize('change', ['drop_hour', 'drop_task', 'settlement', 'target', 'loss', 'origin'])
def test_inconsistent_predictions_or_settlements_cannot_be_trimmed(tmp_path, change):
    candidate, reference, output = pair(tmp_path)
    with sqlite3.connect(candidate / 'state.sqlite') as db:
        if change in {'drop_hour', 'drop_task'}:
            db.execute("DELETE FROM predictions WHERE sample=(SELECT sample FROM predictions LIMIT 1)")
        elif change == 'settlement':
            db.execute("UPDATE predictions SET settled=0,status=NULL,target=NULL,loss=NULL WHERE sample='TEST:64'")
        elif change == 'target':
            # Internally consistent loss, but paired targets disagree.
            db.execute("UPDATE predictions SET target=2,loss=4 WHERE sample='TEST:64'")
        elif change == 'loss':
            db.execute("UPDATE predictions SET loss=999 WHERE sample='TEST:64'")
        else:
            db.execute("UPDATE predictions SET origin='2024-01-04T00:00:00+00:00' WHERE sample='TEST:64'")
    with pytest.raises(ValueError):
        compare_evolutions(candidate, reference, output)
    assert not output.exists()


def test_policy_budget_differences_are_visible_not_claimed_matched(tmp_path):
    policy = EvolutionPolicy(mode='fixed', shadow_hours=32, max_cycles=2, update_every_hours=48)
    paths = pair(tmp_path, candidate_policy=policy)
    result = compare_evolutions(*paths)
    assert result['equal_policy_resource_ceilings'] is False
    assert set(result['policy_differences']) == {'shadow_hours', 'max_cycles', 'update_every_hours'}


@pytest.mark.parametrize('field', ['visible_hash', 'last_timestamp'])
def test_changed_visible_data_or_information_cutoff_rejects_comparison(tmp_path, field):
    candidate, reference, output = pair(tmp_path)
    with sqlite3.connect(candidate / 'state.sqlite') as db:
        state = json.loads(db.execute('SELECT value FROM state WHERE id=1').fetchone()[0])
        state[field] = 'other-hash' if field == 'visible_hash' else '2024-01-06T23:00:00+00:00'
        db.execute('UPDATE state SET value=? WHERE id=1', (json.dumps(state),))
    with pytest.raises(ValueError, match='integrity'):
        compare_evolutions(candidate, reference, output)


def test_valid_procedures_on_different_observed_data_are_not_comparable(tmp_path):
    data, spec, model = fixture()
    policy = EvolutionPolicy(mode='fixed')
    run(tmp_path, 'reference', data, spec, model, policy=policy)
    data.x[0, 0, 0] += 1
    run(tmp_path, 'candidate', data, spec, model, policy=policy)
    with pytest.raises(ValueError, match='visible_hash'):
        compare_evolutions(tmp_path / 'candidate', tmp_path / 'reference', tmp_path / 'out.json')


def test_initial_digest_and_protocol_must_match_even_when_predictions_match(tmp_path):
    data, spec, model = fixture()
    policy = EvolutionPolicy(mode='fixed')
    run(tmp_path, 'reference', data, spec, model, policy=policy)
    model.metadata['diagnostics']['test_marker'] = 'different-initial-contract'
    run(tmp_path, 'candidate', data, spec, model, policy=policy)
    with pytest.raises(ValueError, match='initial_digest'):
        compare_evolutions(tmp_path / 'candidate', tmp_path / 'reference', tmp_path / 'initial.json')
    different = replace(spec, minimum_increment=.02)
    run(tmp_path, 'protocol', data, different, model, policy=policy)
    with pytest.raises(ValueError, match='protocol'):
        compare_evolutions(tmp_path / 'protocol', tmp_path / 'reference', tmp_path / 'protocol.json')


def test_tampered_model_is_rejected_by_initial_inspection(tmp_path):
    candidate, reference, output = pair(tmp_path)
    next((candidate / 'models').rglob('model.pt')).write_bytes(b'broken')
    with pytest.raises(ValueError, match='artifact'):
        compare_evolutions(candidate, reference, output)


def test_all_tasks_and_equal_symbol_segments_are_retained(tmp_path):
    original, config, _ = fixture()
    tasks = (*config.tasks, TaskSpec('state', 'state', 2))
    config = replace(config, tasks=tasks, symbols=('ALT', 'TEST'))
    x = np.repeat(original.x, 2, axis=0)
    meta = original.meta.loc[original.meta.index.repeat(2)].reset_index(drop=True)
    meta['symbol'] = ['ALT', 'TEST'] * len(original.x)
    meta['sample_id'] = [f'{symbol}:{i // 2}' for i, symbol in enumerate(meta.symbol)]
    target = np.tile([2., 1.], len(original.x))
    data = TaskDataset(x, {'return': target, 'state': x.mean((1, 2))},
        {task.name: np.ones(len(x), dtype=bool) for task in tasks},
        {task.name: np.repeat(original.maturity['return'], 2) for task in tasks},
        meta, original.features, tasks)
    model = fit(data.subset(np.arange(70)), data.subset(np.arange(80, 120)), None, config.training)
    model.state['return'][:] = 0
    mean, std = model.metadata['target_scales']['return']
    model.state['return'][-1] = -mean / std  # prediction zero for both symbols
    policy = EvolutionPolicy(mode='fixed')
    run(tmp_path, 'reference', data, config, model, policy=policy)
    run(tmp_path, 'candidate', data, config, model, policy=policy)
    result = compare_evolutions(tmp_path / 'candidate', tmp_path / 'reference', tmp_path / 'out.json')
    assert set(result['tasks']) == {'return', 'state'}
    assert result['tasks']['state']['descriptive_only'] is True
    assert result['per_symbol']['ALT']['return']['reference_loss'] == pytest.approx(4)
    assert result['per_symbol']['TEST']['return']['reference_loss'] == pytest.approx(1)
    assert result['calendar_segments'][0]['tasks']['return']['reference_loss'] == pytest.approx(2.5)
    assert result['paired_forecast_samples'] == 162
