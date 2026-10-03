"""Causal promotion, rejection, persistence, and recovery tests with real checkpoints."""
import sqlite3
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from factorlab.discovery.contracts import ExperimentSpec, TaskDataset, TaskSpec, TrainingSpec
from factorlab.discovery.evolution import EvolutionPolicy, inspect_evolution, run_evolution
from factorlab.discovery.learning import fit


def fixture():
    times = pd.date_range('2024-01-01', periods=160, freq='h', tz='UTC')
    rng = np.random.default_rng(41)
    x = rng.normal(size=(160, 2, 2)).astype(np.float32)
    spec = ExperimentSpec(tasks=(TaskSpec('return', 'return', 2),), primary_task='return',
        lookback=2, symbols=('TEST',), start=times[0].isoformat(), train_end=times[64].isoformat(),
        dev_end=times[145].isoformat(), calibration_end=times[150].isoformat(), end=times[-1].isoformat(),
        training=TrainingSpec(epochs=1, max_seconds=10), training_seeds=(17,), candidates=2,
        search_seconds=30, min_blocks=2, bootstrap_block_hours=4, bootstrap_samples=100,
        minimum_increment=.01)
    data = TaskDataset(x, {'return': np.ones(160)}, {'return': np.ones(160, dtype=bool)},
        {'return': (times + pd.Timedelta(hours=2)).to_numpy(dtype='datetime64[ns]')},
        pd.DataFrame({'sample_id': [f'TEST:{i}' for i in range(160)], 'symbol': 'TEST',
                      'available_at': times}), ('a', 'b'), spec.tasks)
    model = fit(data.subset(np.arange(35)), data.subset(np.arange(40, 60)), None, spec.training)
    model.state['return'][:] = 0
    model.state['return'][-1] = -1
    return data, spec, model


def good_builder(train, dev, incumbent, spec, policy, directory):
    assert max(train.maturity['return']) <= dev.meta.available_at.min().to_datetime64()
    assert max(dev.maturity['return']) <= pd.Timestamp(spec.dev_end).to_datetime64()
    candidate = fit(train, dev, None, spec.training)
    return candidate, {'test': 'known_signal_candidate'}


def bad_builder(train, dev, incumbent, spec, policy, directory):
    model, details = good_builder(train, dev, incumbent, spec, policy, directory)
    model.state['return'][:] = 0
    model.state['return'][-1] = -5
    return model, details


def run(tmp_path, name, data, spec, model, **kwargs):
    policy = kwargs.pop('policy', EvolutionPolicy(shadow_hours=16, update_every_hours=24, max_cycles=1))
    return run_evolution(data, model, spec, tmp_path / name, policy=policy,
        memory_path=tmp_path / 'memory.sqlite', registry_path=tmp_path / 'evidence.sqlite',
        evidence_source='synthetic', **kwargs)


def predictions(path):
    with sqlite3.connect(path / 'state.sqlite') as db:
        return db.execute('SELECT sample,task,model,origin,value,role,settled,loss,status,target '
                          'FROM predictions ORDER BY origin,sample,model,task').fetchall()


def test_improvement_requires_mature_shadow_then_promotes_once(tmp_path):
    data, spec, model = fixture()
    partial = run(tmp_path, 'evo', data, spec, model, candidate_builder=good_builder, max_ticks=10)
    assert partial['promotions'] == 0
    assert partial['cycles'][0]['status'] == 'shadow'
    done = run(tmp_path, 'evo', data, spec, None, candidate_builder=good_builder, resume=True)
    assert done['promotions'] == 1
    assert done['cycles'][0]['gate']['status'] == 'supported'
    assert done['cycles'][0]['gate']['operational_development_improvement_supported'] is True
    assert done['cycles'][0]['gate']['predictive_claim_supported'] is False
    assert done['cycles'][0]['gate']['independent_confirmation'] is False
    assert pd.Timestamp(done['cycles'][0]['decided_at']) >= pd.Timestamp(done['cycles'][0]['decision_at'])
    assert inspect_evolution(tmp_path / 'evo')['event_head'] == done['event_head']


def test_worse_candidate_is_not_installed_after_successful_fit(tmp_path):
    data, spec, model = fixture()
    done = run(tmp_path, 'bad', data, spec, model, candidate_builder=bad_builder)
    assert done['incumbent'] == model.digest
    assert done['cycles'][0]['status'] == 'retained'
    assert done['cycles'][0]['gate']['status'] == 'target_not_met'


def test_resume_is_idempotent_and_matches_uninterrupted_predictions(tmp_path):
    data, spec, model = fixture()
    policy = EvolutionPolicy(mode='fixed')
    run(tmp_path, 'full', data, spec, model, policy=policy)
    run(tmp_path, 'split', data, spec, model, policy=policy, max_ticks=17)
    result = run(tmp_path, 'split', data, spec, None, policy=policy, resume=True)
    before = predictions(tmp_path / 'split')
    again = run(tmp_path, 'split', data, spec, None, policy=policy, resume=True)
    assert predictions(tmp_path / 'full') == before == predictions(tmp_path / 'split')
    assert result == again


def test_future_targets_never_reach_current_search_or_change_past(tmp_path):
    data, spec, model = fixture()
    run(tmp_path, 'original', data, spec, model, candidate_builder=good_builder, max_ticks=8)
    other = deepcopy(data)
    future = other.meta.available_at >= pd.Timestamp(spec.train_end) + pd.Timedelta(hours=8)
    other.targets['return'][future] = -10000
    run(tmp_path, 'future', other, spec, model, candidate_builder=good_builder, max_ticks=8)
    # Wall-clock fitting diagnostics affect model identities, not causal predictions.
    def outputs(name):
        return sorted(tuple(row[:2] + row[3:]) for row in predictions(tmp_path / name))
    assert outputs('original') == outputs('future')


def test_resume_rejects_changed_past_protocol_or_checkpoint(tmp_path):
    data, spec, model = fixture()
    policy = EvolutionPolicy(mode='fixed')
    run(tmp_path, 'safe', data, spec, model, policy=policy, max_ticks=4)
    other = deepcopy(data)
    other.x[0, 0, 0] += 2
    with pytest.raises(ValueError, match='Previously observed'):
        run(tmp_path, 'safe', other, spec, None, policy=policy, resume=True)
    with pytest.raises(ValueError, match='identity changed'):
        run(tmp_path, 'safe', data, spec, None, policy=replace(policy, shadow_hours=1), resume=True)
    artifact = next((tmp_path / 'safe' / 'models').rglob('model.pt'))
    artifact.write_bytes(b'broken')
    with pytest.raises(ValueError, match='artifact changed'):
        run(tmp_path, 'safe', data, spec, None, policy=policy, resume=True)


def test_future_fitted_initial_model_is_rejected(tmp_path):
    data, spec, model = fixture()
    model.metadata['fit_label_cutoff'] = spec.dev_end
    with pytest.raises(ValueError, match='cutoff'):
        run(tmp_path, 'future-model', data, spec, model)


def test_default_refit_runs_real_weight_update_but_obeys_shadow(tmp_path):
    data, spec, model = fixture()
    result = run(tmp_path, 'refit', data, spec, model,
                 policy=EvolutionPolicy(mode='refit', max_cycles=1, shadow_hours=16))
    assert result['cycles'][0]['status'] == 'promoted'


def test_invalid_candidate_search_records_failure_and_preserves_model(tmp_path):
    data, spec, model = fixture()

    def fail(*_args):
        raise RuntimeError('controlled training failure')

    result = run(tmp_path, 'failed', data, spec, model, candidate_builder=fail)
    assert result['incumbent'] == model.digest
    assert result['cycles'][0]['status'] == 'failed'
    assert result['ticks'] > 1


def test_builtin_evolution_search_uses_persistent_memory_across_cycles(tmp_path):
    data, spec, model = fixture()
    result = run(tmp_path, 'population', data, spec, model,
                 policy=EvolutionPolicy(max_cycles=2, shadow_hours=16, update_every_hours=24))
    assert len(result['cycles']) == 2
    assert all(c['status'] in {'promoted', 'retained', 'unchanged_retained'}
               for c in result['cycles']), result['cycles']
    assert result['cycles'][1]['research']['development_memory']
    with sqlite3.connect(tmp_path / 'memory.sqlite') as connection:
        assert connection.execute('SELECT COUNT(*) FROM action_results').fetchone()[0] >= 2


def test_partial_feed_can_resume_when_more_data_arrives(tmp_path):
    data, spec, model = fixture()
    policy = EvolutionPolicy(mode='fixed')
    early = data.subset(np.arange(80))
    for name in early.targets:
        early.observed[name] &= early.maturity[name] <= early.meta.available_at.max().to_datetime64()
        early.targets[name][~early.observed[name]] = np.nan
    partial = run(tmp_path, 'feed', early, spec, model, policy=policy)
    assert pd.Timestamp(partial['last_timestamp']) == early.meta.available_at.max()
    done = run(tmp_path, 'feed', data, spec, None, policy=policy, resume=True)
    assert done['ticks'] > partial['ticks']
