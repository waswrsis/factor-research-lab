"""Production persistence, chooser and operator routing on bounded synthetic data."""
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch
from test_discovery_experiment import controlled_data

from factorlab.discovery import experiment
from factorlab.discovery.contracts import ModuleSpec, RepresentationSpec, partition
from factorlab.discovery.evolution import EvolutionPolicy, _memory_context_hash, _research_candidate
from factorlab.discovery.experiment import ModelGroup, fit_group
from factorlab.discovery.operator_search import seed_program
from factorlab.discovery.research_memory import ResearchMemory, to_development_history
from factorlab.discovery.robust_selection import RobustSelectionSpec
from factorlab.protocol import utc


def fixture(candidates=1):
    torch.set_num_threads(2)
    data, spec = controlled_data()
    return data, replace(spec, training=replace(spec.training, selection_task=spec.primary_task,
                                                epochs=1, patience=1),
                         candidates=candidates, search_seconds=120,
                         selection=RobustSelectionSpec(mode='walk_forward', min_train_hours=120,
                                                       min_validation_hours=48))


@pytest.mark.usefixtures('_preserve_native_thread_count')
def test_real_production_pipeline_retains_every_forward_model_and_forecast(tmp_path):
    data, spec = fixture()
    directory = tmp_path / 'production'
    result = experiment.run_experiment(data, spec, directory, data_hash='synthetic-forward-fixture',
                 modes=('diagnostic',), confirm=False, registry_path=tmp_path / 'registry.sqlite',
                 evidence_source='synthetic-forward-fixture')
    assert result['status'] == 'development_complete'
    assert result['selection']['selected_on'] == 'cold_walk_forward_robust_rank'
    assert result['selection']['arm'] == 'search_diagnostic'
    search = json.loads((directory / 'search_diagnostic' / 'results.json').read_text('utf-8'))
    assert search['fit_attempts'] == 4 and search['final_refit']['status'] == 'ok'
    assert len(search['simple_benchmarks']) == 3
    assert all(row['controls']['training_constant']['metrics']['volatility']['status'] == 'ok'
               for row in search['simple_benchmarks'])
    folders = sorted((directory / 'search_diagnostic' / 'forward_folds').glob('*/fold-*'))
    assert len(folders) == 3
    for folder in folders:
        model = ModelGroup.load(folder / 'model')
        fold = json.loads((folder / 'fold.json').read_text('utf-8'))
        rows = pd.read_parquet(folder / 'ensemble-forecasts.parquet')
        assert len(rows) == fold['split']['validation_rows']
        assert model.digest == fold['model_digest']
        assert rows.volatility_observed.sum() > 0
        assert np.isfinite(rows.loc[rows.volatility_observed, 'volatility_loss']).all()
        assert (rows.loc[rows.volatility_observed, 'volatility_maturity']
                <= utc(fold['split']['validation_end_exclusive'])).all()
    reloaded = ModelGroup.load(directory / 'bundle' / 'candidate')
    assert reloaded.digest == search['final_refit']['digest']
    assert reloaded.digest not in {fold['digest'] for fold in search['records'][0]['folds']}


@pytest.mark.usefixtures('_preserve_native_thread_count')
def test_production_chooser_uses_forward_rank_even_when_single_dev_prefers_other_arm(tmp_path, monkeypatch):
    data, spec = fixture()
    original_score = experiment._score

    def fake_search(train, dev, parent, protocol, directory, *, mode, **kwargs):
        model = fit_group(train, dev, parent, protocol.training, protocol.training_seeds)
        model.selection_test_mode = mode
        return model, {'selection_rank': .2 if mode == 'diagnostic' else .1}

    def adverse_dev_loss(model, development, primary):
        loss, metrics = original_score(model, development, primary)
        if hasattr(model, 'selection_test_mode'):
            loss = 10.0 if model.selection_test_mode == 'diagnostic' else .000001
        return loss, metrics

    monkeypatch.setattr(experiment, 'search_candidates', fake_search)
    monkeypatch.setattr(experiment, '_score', adverse_dev_loss)
    result = experiment.run_experiment(data, spec, tmp_path / 'rank', data_hash='synthetic-chooser',
                 modes=('random', 'diagnostic'), registry_path=tmp_path / 'registry.sqlite',
                 evidence_source='synthetic-chooser')
    assert result['selection']['arm'] == 'search_diagnostic'
    assert result['selection']['selected_on'] == 'cold_walk_forward_robust_rank'


@pytest.mark.usefixtures('_preserve_native_thread_count')
def test_production_fails_closed_when_all_forward_searches_are_invalid(tmp_path, monkeypatch):
    data, spec = fixture()
    monkeypatch.setattr(experiment, 'search_candidates', lambda *args, **kwargs:
                        (None, {'selection_rank': None, 'fallback_reason': 'reference_fold_fit_failed'}))
    directory = tmp_path / 'invalid'
    with pytest.raises(RuntimeError, match='walk-forward'):
        experiment.run_experiment(data, spec, directory, data_hash='synthetic-invalid-search',
                 modes=('diagnostic',), registry_path=tmp_path / 'registry.sqlite',
                 evidence_source='synthetic-invalid-search')
    failure = json.loads((directory / 'failure.json').read_text('utf-8'))
    assert failure['status'] == 'process_invalid'
    assert not (directory / 'bundle' / 'candidate').exists()


@pytest.mark.usefixtures('_preserve_native_thread_count')
def test_real_evolution_research_keeps_programs_fold_artifacts_and_gate_memory(tmp_path):
    data, spec = fixture(candidates=2)
    train, development = partition(data, spec, 'train'), partition(data, spec, 'development')
    program = seed_program(train.features, 0, spec.lookback)
    architecture = RepresentationSpec((ModuleSpec('indicator', 'program', dim=2, width=4,
                          scale=spec.lookback, features=tuple(program['feature_names']), program=program),))
    initial = fit_group(train, development, architecture, spec.training, spec.training_seeds)
    policy = EvolutionPolicy(search_mode='diagnostic', program_search=True, inherit_weights=True)
    directory, memory_path = tmp_path / 'cycle' / 'research', tmp_path / 'memory.sqlite'
    candidate, result = _research_candidate(train, development, initial, spec, policy, directory,
                                            utc(spec.dev_end), 'cpu', memory_path, None)
    assert candidate is not None and result['program_search']
    assert result['inherit_weights'] is False
    assert result['fit_attempts'] == 7
    assert len(list((directory / 'forward_folds').glob('*/fold-*'))) == 6
    assert all(not record['reused'] for record in result['records'] if 'architecture' in record)
    facts = ResearchMemory(memory_path).retrieve(spec.tasks[0], spec.dev_end)
    assert len(facts) == 1
    assert type(facts[0]['metrics']['robust_eligible']) is bool
    assert facts[0]['metrics']['selection_reason'] in {'passed', 'minimum_relative_improvement_not_met',
                 'winning_fold_fraction_not_met', 'primary_fold_degradation', 'stability_penalty_not_met'}
    history = to_development_history(facts, spec)
    assert 'robustness_rank' in history[0] and 'selection_reason' in history[0]
    assert all(any(module['kind'] == 'program' for module in record['architecture']['modules'])
               for record in result['records'] if 'architecture' in record)


@pytest.mark.usefixtures('_preserve_native_thread_count')
def test_memory_context_changes_with_frozen_selection_and_checkpoint_policy():
    data, spec = fixture()
    changed = replace(spec, selection=replace(spec.selection, stability_penalty=2))
    assert _memory_context_hash(data, spec) != _memory_context_hash(data, changed)
    legacy = replace(spec, selection=RobustSelectionSpec())
    joint = replace(legacy, training=replace(legacy.training, selection_task=None))
    assert _memory_context_hash(data, legacy) != _memory_context_hash(data, joint)
    later = replace(spec, train_end='2024-01-16', dev_end='2024-01-23')
    assert _memory_context_hash(data, spec) == _memory_context_hash(data, later)


@pytest.fixture
def _preserve_native_thread_count():
    """Only this test's native thread choice; restore even after a failed assertion."""
    import torch
    original_threads = torch.get_num_threads()
    try:
        yield
    finally:
        torch.set_num_threads(original_threads)
