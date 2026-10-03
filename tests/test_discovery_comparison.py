"""Replay comparison checks use actual archived, verified replay outputs."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from factorlab.discovery import comparison
from factorlab.discovery.comparison import compare_replays
from factorlab.discovery.contracts import ExperimentSpec, TaskDataset, TaskSpec, TrainingSpec
from factorlab.discovery.learning import fit
from factorlab.discovery.replay import load_replay, run_replay


@pytest.fixture
def replay_pair(tmp_path):
    rng = np.random.default_rng(314)
    times = pd.date_range('2024-01-01', periods=96, freq='h', tz='UTC')
    task = TaskSpec('return_value', 'return', 1)
    spec = ExperimentSpec(tasks=(task,), primary_task=task.name, lookback=2,
                          symbols=('CONTROLLED',), start='2024-01-01', train_end='2024-01-02',
                          dev_end='2024-01-03', calibration_end='2024-01-04', end='2024-01-05',
                          training=TrainingSpec(epochs=1, max_seconds=10),
                          bootstrap_block_hours=6, bootstrap_samples=100, min_blocks=2)
    x = rng.normal(size=(96, 2, 1)).astype(np.float32)
    y = .5 * x[:, -1, 0] + .1 * rng.normal(size=96)
    data = TaskDataset(x, {task.name: y}, {task.name: np.ones(96, dtype=bool)},
                       {task.name: (times + pd.Timedelta(hours=1)).to_numpy(dtype='datetime64[ns]')},
                       pd.DataFrame({'sample_id': [f's{i}' for i in range(96)],
                                     'symbol': 'CONTROLLED', 'available_at': times}), ('x',), spec.tasks)
    train = data.subset(np.arange(96) < 12)
    dev = data.subset((np.arange(96) >= 12) & (np.arange(96) < 23))
    initial = fit(train, dev, None, spec.training)
    fixed_dir, refit_dir = tmp_path / 'fixed', tmp_path / 'refit'
    kwargs = {'update_every_hours': 6, 'registry_path': tmp_path / 'registry.sqlite',
              'evidence_source': 'controlled-replay-comparison'}
    run_replay(data, initial, spec, fixed_dir, update_mode='fixed', **kwargs)
    run_replay(data, initial, spec, refit_dir, update_mode='refit', **kwargs)
    return data, spec, initial, fixed_dir, refit_dir, kwargs


def test_fixed_vs_refit_aligns_entire_scope_and_reports_development_cost(replay_pair, tmp_path):
    _, spec, _, fixed_dir, refit_dir, _ = replay_pair
    result = compare_replays(refit_dir, fixed_dir, tmp_path / 'comparison.json')
    assert result['evidence_stage'] == 'development_only'
    assert not result['independent_confirmation']
    assert result['paired_prediction_samples'] == 24
    entry = result['comparisons'][spec.primary_task]
    assert entry['support'] == 24 and entry['coverage'] == 1
    assert not entry['predictive_claim_supported']
    fixed, refit = load_replay(fixed_dir), load_replay(refit_dir)
    reference_losses = [r['value']['loss'] for r in fixed['records'] if r['event'] == 'settlement']
    candidate_losses = [r['value']['loss'] for r in refit['records'] if r['event'] == 'settlement']
    assert entry['estimate'] == pytest.approx(np.mean(reference_losses) - np.mean(candidate_losses))
    assert result['reference']['cost']['update_attempts'] == 0
    assert result['candidate']['cost']['successful_updates'] == 3
    assert result['candidate']['cost']['known_successful_update_fit_seconds'] >= 0
    assert result['candidate']['cost']['cost_complete']
    assert json.loads((tmp_path / 'comparison.json').read_text())['paired_settlements'] == 24


def test_comparison_is_repeatable_and_does_not_overwrite_different_result(replay_pair, tmp_path):
    _, spec, _, fixed_dir, _, _ = replay_pair
    output = tmp_path / 'comparison.json'
    result = compare_replays(fixed_dir, fixed_dir, output)
    record = result['comparisons'][spec.primary_task]
    assert record['estimate'] == record['ci_lower'] == record['ci_upper'] == 0
    assert compare_replays(fixed_dir, fixed_dir, output) == result
    output.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='Immutable'):
        compare_replays(fixed_dir, fixed_dir, output)


def test_corrupted_archive_is_rejected_before_comparison(replay_pair, tmp_path):
    _, _, _, fixed_dir, refit_dir, _ = replay_pair
    journal = fixed_dir / 'records.jsonl'
    journal.write_text(journal.read_text().splitlines()[0] + '\n')
    with pytest.raises(ValueError, match='integrity'):
        compare_replays(refit_dir, fixed_dir, tmp_path / 'comparison.json')


def test_declared_initial_identity_must_match_the_loaded_checkpoint(replay_pair, tmp_path, monkeypatch):
    _, _, _, fixed_dir, refit_dir, _ = replay_pair
    c, r = load_replay(refit_dir), load_replay(fixed_dir)
    c['manifest']['initial_model_digest'] = 'different-checkpoint'
    monkeypatch.setattr(comparison, 'load_replay', lambda directory: c if directory == refit_dir else r)
    with pytest.raises(ValueError, match='initial model identity'):
        compare_replays(refit_dir, fixed_dir, tmp_path / 'comparison.json')


def test_different_initial_fitted_state_is_not_called_an_update_effect(replay_pair, tmp_path):
    data, spec, initial, fixed_dir, _, kwargs = replay_pair
    altered = copy.deepcopy(initial)
    altered.state[spec.primary_task][0] += .1
    altered_dir = tmp_path / 'altered_initial'
    run_replay(data, altered, spec, altered_dir, update_mode='fixed', **kwargs)
    with pytest.raises(ValueError, match='Initial fitted states'):
        compare_replays(altered_dir, fixed_dir, tmp_path / 'comparison.json')


def test_equal_initial_weights_are_allowed_with_different_recorded_wall_time(replay_pair, tmp_path):
    data, spec, initial, fixed_dir, _, kwargs = replay_pair
    altered = copy.deepcopy(initial)
    altered.metadata['diagnostics']['elapsed_seconds'] += 1.
    altered_dir = tmp_path / 'same_weights_new_clock'
    run_replay(data, altered, spec, altered_dir, update_mode='fixed', **kwargs)
    result = compare_replays(altered_dir, fixed_dir, tmp_path / 'comparison.json')
    assert result['comparisons'][spec.primary_task]['estimate'] == 0
    assert result['candidate']['initial_model_digest'] != result['reference']['initial_model_digest']


@pytest.mark.parametrize('field', ['data_sha256', 'source_files', 'protocol', 'start', 'schema'])
def test_protocol_source_data_and_time_mismatch_are_rejected(replay_pair, tmp_path, monkeypatch, field):
    _, _, _, fixed_dir, refit_dir, _ = replay_pair
    c, r = load_replay(refit_dir), load_replay(fixed_dir)
    c['manifest'][field] = 'different'
    monkeypatch.setattr(comparison, 'load_replay', lambda directory: c if directory == refit_dir else r)
    with pytest.raises(ValueError, match=f'different {field}'):
        compare_replays(refit_dir, fixed_dir, tmp_path / 'comparison.json')


@pytest.mark.parametrize('fault', ['missing_prediction', 'missing_settlement', 'different_target',
                                   'different_maturity', 'bad_loss', 'duplicate_prediction'])
def test_semantic_alignment_does_not_trim_bad_or_missing_samples(replay_pair, tmp_path, monkeypatch, fault):
    _, _, _, fixed_dir, refit_dir, _ = replay_pair
    c, r = load_replay(refit_dir), load_replay(fixed_dir)
    prediction = next(v for v in c['records'] if v['event'] == 'prediction')
    settlement = next(v for v in c['records'] if v['event'] == 'settlement')
    if fault == 'missing_prediction':
        c['records'].remove(prediction)
    elif fault == 'missing_settlement':
        c['records'].remove(settlement)
    elif fault == 'different_target':
        settlement['value']['target'] += 1.
    elif fault == 'different_maturity':
        settlement['value']['maturity'] = '2024-01-01T00:00Z'
    elif fault == 'bad_loss':
        settlement['value']['loss'] += 1.
    else:
        c['records'].append(copy.deepcopy(prediction))
    monkeypatch.setattr(comparison, 'load_replay', lambda directory: c if directory == refit_dir else r)
    with pytest.raises(ValueError):
        compare_replays(refit_dir, fixed_dir, tmp_path / 'comparison.json')


def test_missing_head_is_unavailable_without_selecting_successful_rows(replay_pair, tmp_path, monkeypatch):
    _, spec, _, fixed_dir, refit_dir, _ = replay_pair
    c, r = load_replay(refit_dir), load_replay(fixed_dir)
    first = next(v for v in c['records'] if v['event'] == 'prediction')
    first['value']['predictions'][spec.primary_task] = None
    settled = next(v for v in c['records'] if v['event'] == 'settlement'
                   and v['value']['sample_id'] == first['value']['sample_id'])
    settled['value'].update(status='unavailable_prediction', loss=None)
    monkeypatch.setattr(comparison, 'load_replay', lambda directory: c if directory == refit_dir else r)
    result = compare_replays(refit_dir, fixed_dir, tmp_path / 'comparison.json')
    entry = result['comparisons'][spec.primary_task]
    assert entry['status'] == 'unavailable' and entry['support'] == 0
    assert entry['total_rows'] == entry['eligible_support'] == 24
    assert entry['candidate_prediction_support'] == 23
    assert entry['reference_prediction_support'] == 24
