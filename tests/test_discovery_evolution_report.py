"""Validate exported multi-task scores and first-event bins against the actual ledger."""
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from factorlab.discovery.contracts import ExperimentSpec, TaskDataset, TaskSpec, TrainingSpec
from factorlab.discovery.evidence import file_hash
from factorlab.discovery.evolution import EvolutionPolicy, run_evolution
from factorlab.discovery.evolution_report import export_evolution
from factorlab.discovery.learning import fit
from factorlab.discovery.scoring import adaptive_horizons, per_sample_loss


def multitask_fixture():
    times = pd.date_range('2024-01-01', periods=160, freq='h', tz='UTC')
    rng = np.random.default_rng(125)
    tasks = (TaskSpec('return', 'return', 2), TaskSpec('touch', 'first_event', 2, (1, 2)))
    spec = ExperimentSpec(tasks=tasks, primary_task='return', lookback=2, symbols=('TEST',),
        start=times[0].isoformat(), train_end=times[64].isoformat(), dev_end=times[145].isoformat(),
        calibration_end=times[150].isoformat(), end=times[-1].isoformat(),
        training=TrainingSpec(epochs=1, max_seconds=5), training_seeds=(17,),
        candidates=2, search_seconds=4, min_blocks=2, bootstrap_block_hours=4, bootstrap_samples=100)
    data = TaskDataset(rng.normal(size=(160, 2, 2)).astype(np.float32),
        {'return': rng.normal(size=160), 'touch': np.arange(160, dtype=float) % tasks[1].output_size},
        {task.name: np.ones(160, dtype=bool) for task in tasks},
        {task.name: (times + pd.Timedelta(hours=2)).to_numpy(dtype='datetime64[ns]') for task in tasks},
        pd.DataFrame({'sample_id': [f'TEST:{i}' for i in range(160)], 'symbol': 'TEST',
                      'available_at': times}), ('a', 'b'), tasks)
    model = fit(data.subset(np.arange(35)), data.subset(np.arange(40, 60)), None, spec.training)
    return data, spec, model


def test_export_scores_cover_every_task_and_loss_matches_each_task_definition(tmp_path):
    data, spec, model = multitask_fixture()
    directory = tmp_path / 'run'
    summary = run_evolution(data, model, spec, directory, policy=EvolutionPolicy(mode='fixed'),
        memory_path=tmp_path / 'memory.sqlite', registry_path=tmp_path / 'evidence.sqlite',
        evidence_source='synthetic_report')
    output = tmp_path / 'export'
    exported = export_evolution(directory, output)
    scores = pd.read_parquet(output / 'scores.parquet')
    assert set(scores.task) == {task.name for task in spec.tasks}
    assert len(scores) == summary['ticks'] * len(spec.tasks) == exported['rows']
    lookup = {sample: i for i, sample in enumerate(data.meta.sample_id)}
    for task in spec.tasks:
        selected = scores[scores.task == task.name]
        settled = selected[selected.status == 'scored']
        predictions = np.asarray([json.loads(value) for value in settled.value])
        target = np.asarray([data.targets[task.name][lookup[sample]] for sample in settled['sample']])
        expected = per_sample_loss(task, target, predictions)
        assert np.allclose(settled.loss.to_numpy(dtype=float), expected)
        assert summary['tasks'][task.name]['support'] == len(settled)
        assert summary['tasks'][task.name]['loss'] == pytest.approx(float(expected.mean()))
        assert summary['tasks'][task.name]['pending'] == int((selected.settled == 0).sum())
        maturity = pd.to_datetime(selected.origin, utc=True) + pd.Timedelta(hours=task.horizon_hours)
        assert summary['tasks'][task.name]['pending'] == int((maturity > pd.Timestamp(summary['last_timestamp'])).sum())
        assert (pd.to_datetime(settled.origin, utc=True) + pd.Timedelta(hours=2)
                <= pd.Timestamp(summary['last_timestamp'])).all()
    with sqlite3.connect(directory / 'state.sqlite') as connection:
        connection.row_factory = sqlite3.Row
        original = [dict(row) for row in connection.execute(
            "SELECT * FROM predictions WHERE role='incumbent' ORDER BY origin,symbol,task")]
    assert scores.astype(object).where(scores.notna(), None).to_dict('records') == original
    document = (output / 'report.html').read_text('utf-8')
    assert '<td>return</td>' in document and '<td>touch</td>' in document
    manifest = json.loads((output / 'manifest.json').read_text('utf-8'))
    assert manifest['event_head'] == summary['event_head']
    for filename, digest in manifest['files'].items():
        assert file_hash(output / filename) == digest


def test_export_event_distribution_preserves_joint_classes_bins_and_horizon_rule(tmp_path):
    data, spec, model = multitask_fixture()
    directory = tmp_path / 'run'
    run_evolution(data, model, spec, directory, policy=EvolutionPolicy(mode='fixed'),
        memory_path=tmp_path / 'memory.sqlite', registry_path=tmp_path / 'evidence.sqlite',
        evidence_source='synthetic_report')
    output = tmp_path / 'export'
    export_evolution(directory, output)
    scores = pd.read_parquet(output / 'scores.parquet')
    event_rows = scores[scores.task == 'touch']
    distributions = json.loads((output / 'event_distributions.json').read_text('utf-8'))
    task = spec.tasks[1]
    probabilities = np.asarray([json.loads(value) for value in event_rows.value])
    expected = adaptive_horizons(probabilities, task)
    assert len(distributions) == len(event_rows) and len(distributions) > 10
    assert probabilities.shape == (len(event_rows), 5)
    assert np.allclose(probabilities.sum(axis=1), 1)
    for index, (distribution, row) in enumerate(zip(distributions, event_rows.itertuples(), strict=True)):
        assert distribution['sample'] == row.sample
        assert distribution['symbol'] == row.symbol
        assert distribution['origin'] == row.origin
        assert distribution['model'] == row.model
        assert distribution['task'] == 'touch'
        assert distribution['bins'] == [1, 2]
        assert distribution['probabilities'] == probabilities[index].tolist()
        assert distribution['selected_horizon_hours'] == int(expected['horizon_hours'][index])
        assert distribution['coverage_achieved'] == bool(expected['coverage_achieved'][index])
        assert distribution['concentration_is_confidence'] is False
        # Complete no-event is category zero; horizon coverage uses unconditional
        # up/down event mass, never renormalizes after dropping no-event.
        first_bin_mass = probabilities[index, 1] + probabilities[index, 2]
        full_event_mass = probabilities[index, 1:].sum()
        manual_horizon = 1 if first_bin_mass >= .8 else 2
        assert distribution['selected_horizon_hours'] == manual_horizon
        assert distribution['coverage_achieved'] == (full_event_mass >= .8)


def test_export_rejects_changed_model_artifact_before_writing_report(tmp_path):
    data, spec, model = multitask_fixture()
    directory = tmp_path / 'run'
    run_evolution(data, model, spec, directory, policy=EvolutionPolicy(mode='fixed'),
        memory_path=tmp_path / 'memory.sqlite', registry_path=tmp_path / 'evidence.sqlite',
        evidence_source='synthetic_report', max_ticks=3)
    artifact = next((directory / 'models').rglob('model.pt'))
    artifact.write_bytes(b'invalid checkpoint')
    with pytest.raises(ValueError, match='artifact changed'):
        export_evolution(directory, tmp_path / 'invalid-export')
    assert not (tmp_path / 'invalid-export').exists()
