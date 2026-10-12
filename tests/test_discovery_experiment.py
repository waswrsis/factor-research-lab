from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from factorlab.discovery.contracts import ExperimentSpec, TaskDataset, TaskSpec, TrainingSpec
from factorlab.discovery.evidence import EvidenceRegistry
from factorlab.discovery.experiment import ModelGroup, run_experiment
from factorlab.discovery.report import render_report
from factorlab.discovery.search import ResearchProposal, _clean_history, action_menu, mutate


def controlled_data():
    rng = np.random.default_rng(312)
    dates = pd.date_range('2024-01-01', periods=960, freq='h', tz='UTC')
    x = rng.normal(size=(len(dates), 8, 3)).astype(np.float32)
    y = np.exp(-4 + 0.6 * x[:, -1, 0] + 0.03 * rng.normal(size=len(x)))
    spec = ExperimentSpec(tasks=(TaskSpec(),), lookback=8, symbols=('TEST',),
        start='2024-01-01', train_end='2024-01-15', dev_end='2024-01-22',
        calibration_end='2024-01-29', end='2024-02-10',
        training=TrainingSpec(epochs=2, patience=2, batch_size=128, max_seconds=20),
        training_seeds=(17,), candidates=1, bootstrap_samples=50, min_blocks=2,
        bootstrap_block_hours=24)
    data = TaskDataset(x, {'volatility': y}, {'volatility': np.ones(len(y), dtype=bool)},
        {'volatility': (dates + pd.Timedelta(hours=12)).to_numpy(dtype='datetime64[ns]')},
        pd.DataFrame({'sample_id': [f'TEST:{i}' for i in range(len(dates))],
                      'symbol': 'TEST', 'available_at': dates}), ('a', 'b', 'c'), spec.tasks)
    return data, spec


@pytest.mark.usefixtures('_preserve_native_thread_count')
def test_full_research_probes_confirmation_and_bundle_reload(tmp_path):
    import torch
    torch.set_num_threads(2)
    data, spec = controlled_data()
    directory, registry = tmp_path / 'run', tmp_path / 'registry.sqlite'
    result = run_experiment(data, spec, directory, data_hash='controlled-fixture',
                            modes=('random',), confirm=True, registry_path=registry,
                            evidence_source='controlled-fixture')
    assert result['status'] == 'confirmation_complete'
    assert set(result['confirmation']['comparisons']) == {
        'system_vs_baseline', 'representation_addition', 'matched_dimension_control',
        'single_vs_baseline', 'fixed_vs_single', 'random_vs_fixed'}
    model = ModelGroup.load(directory / 'bundle' / 'candidate')
    assert np.isfinite(model.predict(data)['volatility']).all()
    assert model.encode(data).shape[0] == len(data.x)
    assert render_report(directory).is_file()
    from factorlab.discovery.catalog import QualityCatalog
    catalog = QualityCatalog(tmp_path / 'catalog.sqlite')
    entry = catalog.register(directory)
    assert catalog.verify(entry['entry_id'])['valid']
    with pytest.raises(ValueError, match='Previously exposed'):
        EvidenceRegistry(registry).consume(source='controlled-fixture', symbols=spec.symbols,
            start=spec.calibration_end, end=spec.end, run='another', claim='changed-target')


def test_protocol_collections_cannot_change_after_freeze():
    with pytest.raises(ValueError, match='immutable'):
        ExperimentSpec(symbols=['BTCUSDT'])
    with pytest.raises(ValueError, match='immutable'):
        TaskSpec(event_bins=[4, 12])


def test_shared_menu_has_no_noop_after_learning_rate_change():
    from factorlab.discovery.contracts import RepresentationSpec
    spec, parent = ExperimentSpec(), RepresentationSpec()
    for lr in (0.001, 0.0001):
        training = replace(spec.training, learning_rate=lr)
        for action in action_menu(parent, spec, training):
            p = ResearchProposal(spec.digest, parent.digest, spec.dev_end,
                action['operation'], 'hypothesis', 'reject without improvement', 'test', action['arguments'])
            child, settings = mutate(parent, training, p, spec)
            assert child != parent or settings != training


def test_agent_packet_does_not_forward_nested_non_development_values():
    spec = ExperimentSpec()
    history = [{'partition': 'development', 'evidence_cutoff': spec.dev_end,
                'status': 'ok', 'metrics': {'volatility': {'loss': 0.3, 'confirmation_loss': -99}},
                'training': [{'test': -999}]}]
    clean = _clean_history(history, spec)
    assert clean[0]['metrics'] == {'volatility': {'loss': 0.3}}
    assert 'training' not in clean[0]


def test_empty_confirmation_is_insufficient_and_stays_consumed(tmp_path):
    data, spec = controlled_data()
    data = data.subset((data.meta.available_at < pd.Timestamp(spec.calibration_end, tz='UTC')).to_numpy())
    result = run_experiment(data, spec, tmp_path / 'empty', data_hash='fixture',
        modes=(), confirm=True, registry_path=tmp_path / 'registry.sqlite', evidence_source='fixture')
    assert result['confirmation']['accepted_system'] is False
    assert all(c['status'] == 'insufficient_evidence'
               for c in result['confirmation']['comparisons'].values())
    assert result['confirmation']['candidate_predictions']['rows'] == 0


def test_failed_pipeline_records_failure_without_overwriting_old_run(tmp_path):
    import json
    data, spec = controlled_data()
    data = data.subset((data.meta.available_at < pd.Timestamp(spec.dev_end, tz='UTC')).to_numpy())
    target = tmp_path / 'failed'
    with pytest.raises(ValueError, match='at least 32'):
        run_experiment(data, spec, target, data_hash='fixture', modes=(),
                       registry_path=tmp_path / 'registry.sqlite', evidence_source='fixture')
    assert json.loads((target / 'failure.json').read_text())['status'] == 'process_invalid'
    before = (target / 'journal.jsonl').read_bytes()
    with pytest.raises(FileExistsError):
        run_experiment(data, spec, target, data_hash='fixture', modes=(),
                       registry_path=tmp_path / 'registry.sqlite', evidence_source='fixture')
    assert before == (target / 'journal.jsonl').read_bytes()


@pytest.fixture
def _preserve_native_thread_count():
    """Only this test's native thread choice; restore even after a failed assertion."""
    import torch
    original_threads = torch.get_num_threads()
    try:
        yield
    finally:
        torch.set_num_threads(original_threads)
