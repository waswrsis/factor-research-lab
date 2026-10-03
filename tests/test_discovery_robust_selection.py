"""Synthetic forward-fold tests establish mechanics, not market efficacy."""
from dataclasses import FrozenInstanceError, asdict, replace

import numpy as np
import pandas as pd
import pytest

from factorlab.discovery import population, search
from factorlab.discovery.contracts import (
    ExperimentSpec,
    ModuleSpec,
    RepresentationSpec,
    TaskDataset,
    TaskSpec,
    TrainingSpec,
)
from factorlab.discovery.robust_selection import (
    RobustSelectionSpec,
    compare_folds,
    macro_metrics,
    walk_forward_parts,
)
from factorlab.protocol import fingerprint, utc


def fixture(candidates=2, seeds=(17, 29)):
    tasks = (TaskSpec('ret', 'return', 1), TaskSpec('aux', 'return', 1), TaskSpec('state', 'state', 1))
    dates = pd.date_range('2024-01-01', periods=600, freq='h', tz='UTC')
    settings = TrainingSpec(epochs=1, batch_size=256, max_seconds=10, selection_task='ret')
    spec = ExperimentSpec(tasks=tasks, primary_task='ret', symbols=('TEST',), lookback=4,
                          start=dates[0].isoformat(), train_end=dates[300].isoformat(),
                          dev_end=(dates[-1] + pd.Timedelta(hours=1)).isoformat(),
                          calibration_end='2024-02-01', end='2024-03-01', training=settings,
                          training_seeds=seeds, candidates=candidates, search_seconds=600,
                          selection=RobustSelectionSpec(mode='walk_forward'))
    data = TaskDataset(np.zeros((600, 4, 1), dtype=np.float32),
                       {task.name: np.zeros(600) for task in tasks},
                       {task.name: np.ones(600, dtype=bool) for task in tasks},
                       {task.name: (dates + pd.Timedelta(hours=task.horizon_hours)).to_numpy(dtype='datetime64[ns]')
                        for task in tasks},
                       pd.DataFrame({'sample_id': [str(i) for i in range(600)], 'symbol': 'TEST',
                                     'available_at': dates}), ('x',), tasks)
    root = RepresentationSpec((ModuleSpec('root', 'summary', 2, 4, 4),))
    other = RepresentationSpec((ModuleSpec('other', 'summary', 2, 5, 4),))
    return data.subset(np.arange(300)), data.subset(np.arange(300, 600)), root, other, spec


class FakeModel:
    def __init__(self, data, architecture, seed, losses, origin, stop_reason='epoch_budget'):
        self.schema = data.schema
        self.metadata = {'settings': {'seed': seed}}
        self.diagnostics = {'stop_reason': stop_reason, 'optimizer_steps': 1,
                            'weight_transfer': {'requested': False}}
        self.parameter_count = 1
        self.digest = fingerprint([architecture.digest, seed, origin.isoformat()])
        self.losses = losses

    def predict(self, data):
        assert all(np.isnan(values).all() for values in data.targets.values())
        assert all(not values.any() for values in data.observed.values())
        assert all(np.isnat(values).all() for values in data.maturity.values())
        hour = (utc(data.meta.available_at.min()) - utc('2024-01-01')).total_seconds() / 3600
        index = min(2, max(0, int((hour - 300) // 100)))
        return {name: np.full(len(data.meta), np.sqrt(values[index]))
                for name, values in self.losses.items()}


def fake_fitter(calls, losses, stop_reason=None):
    def fit(train, early, architecture, settings, seeds, device, *, initial_model=None):
        assert initial_model is None and len(seeds) == 1
        origin = utc(early.meta.available_at.max()) + pd.Timedelta(hours=1)
        calls.append({'architecture': architecture.digest, 'seed': seeds[0],
                      'train': train, 'early': early, 'origin': origin})
        return FakeModel(train, architecture, seeds[0], losses[architecture.digest], origin,
                         stop_reason.get(architecture.digest, 'epoch_budget') if stop_reason else 'epoch_budget')
    return fit


def score(primary, auxiliary=1, state=1, support=10):
    return {name: {'status': 'ok', 'support': support, 'loss': loss, 'n_hours': support,
                   'aggregation': 'equal_symbol_within_hour_then_equal_hour'}
            for name, loss in (('ret', primary), ('aux', auxiliary), ('state', state))}


def test_policy_frozen_validated_and_protocol_roundtrip():
    _, _, _, _, spec = fixture()
    assert ExperimentSpec.from_dict(spec.to_dict()) == spec
    with pytest.raises(FrozenInstanceError):
        spec.selection.folds = 2
    for value in ({'mode': 'bogus'}, {'folds': 1}, {'folds': True}, {'stability_penalty': float('nan')},
                  {'min_winning_fold_fraction': 0}, {'minimum_relative_improvement': -1}):
        with pytest.raises(ValueError):
            RobustSelectionSpec(**value)


def test_folds_are_forward_purged_and_clear_unmatured_labels():
    train, dev, _, _, spec = fixture()
    parts = walk_forward_parts(train, dev, spec)
    assert len(parts) == 3
    for fitting, early, validation, detail in parts:
        origin, stop = utc(detail['information_cutoff']), utc(detail['validation_end_exclusive'])
        assert fitting.meta.available_at.max() < early.meta.available_at.min() < origin
        assert validation.meta.available_at.min() >= origin
        assert validation.meta.available_at.max() < stop
        for task in spec.tasks:
            assert (fitting.maturity[task.name] <= early.meta.available_at.min().to_datetime64()).all()
            assert (early.maturity[task.name] <= origin.to_datetime64()).all()
            assert (validation.maturity[task.name][validation.observed[task.name]] <= stop.to_datetime64()).all()
    assert [len(fitting.meta) + len(early.meta) for fitting, early, _, _ in parts] == [300, 400, 500]
    # Future auxiliary labels at a fold endpoint are removed before scoring.
    changed = replace(dev, maturity={**dev.maturity, 'aux': dev.maturity['aux'] + np.timedelta64(20, 'h')})
    part = walk_forward_parts(train, changed, spec)[0][2]
    assert (~part.observed['aux']).sum() == 20
    assert np.isnan(part.targets['aux'][~part.observed['aux']]).all()


def test_macro_metrics_symbols_then_hours_prevents_duplicate_overweight():
    task = TaskSpec('ret', 'return', 1)
    dates = pd.to_datetime(['2024-01-01'] * 4 + ['2024-01-01T01:00'], utc=True, format='ISO8601')
    data = TaskDataset(np.zeros((5, 1, 1), dtype=np.float32), {'ret': np.zeros(5)},
                       {'ret': np.ones(5, dtype=bool)},
                       {'ret': (dates + pd.Timedelta(hours=1)).to_numpy(dtype='datetime64[ns]')},
                       pd.DataFrame({'sample_id': list('abcde'), 'symbol': ['A', 'A', 'A', 'B', 'A'],
                                     'available_at': dates}), ('x',), (task,))
    metrics = macro_metrics(data, {'ret': np.sqrt([4, 4, 4, 0, 0])})
    assert metrics['ret']['loss'] == 1  # ((4 + 0)/2 + 0)/2.
    assert metrics['ret']['n_hours'] == 2


def test_guards_primary_tail_auxiliary_and_missing_support_ignore_state():
    _, _, _, _, spec = fixture()
    reference = [score(1)] * 3
    assert compare_folds([score(.8, state=100)] * 3, reference, spec.tasks, 'ret', spec.selection)['eligible']
    assert 'auxiliary_degradation:aux' in compare_folds([score(.8, auxiliary=1.11)] * 3,
                     reference, spec.tasks, 'ret', spec.selection)['reasons']
    result = compare_folds([score(.5), score(.5), score(1.11)], reference, spec.tasks, 'ret', spec.selection)
    assert 'primary_fold_degradation' in result['reasons']
    result = compare_folds([score(.8), score(.8), score(.8, support=0)], reference, spec.tasks, 'ret', spec.selection)
    assert not result['eligible'] and any('unsupported' in reason for reason in result['reasons'])
    zero = compare_folds([score(.8, auxiliary=1e-13)] * 3, [score(1, auxiliary=0)] * 3,
                         spec.tasks, 'ret', spec.selection)
    assert zero['eligible']


def test_negative_stability_rank_cannot_displace_reference_and_support_must_match():
    _, _, _, _, spec = fixture()
    result = compare_folds([score(.9), score(.9), score(1.09)], [score(1)] * 3,
                           spec.tasks, 'ret', spec.selection)
    assert result['mean_relative_gain'] > 0 and result['rank_score'] < 0
    assert result['winning_fold_fraction'] == 2 / 3
    assert result['reasons'] == ['stability_penalty_not_met']
    result = compare_folds([score(.8, support=9)] * 3, [score(1)] * 3,
                           spec.tasks, 'ret', spec.selection)
    assert not result['eligible']
    assert all('incomparable_support_or_aggregation' in reason for reason in result['reasons'])
    result = compare_folds([score(1e308)] * 3, [score(0)] * 3,
                           spec.tasks, 'ret', spec.selection)
    assert not result['eligible']
    import json
    json.dumps(result, allow_nan=False)


def test_partial_epoch_inside_history_is_resource_failure():
    from factorlab.discovery.robust_selection import _validate_fit
    train, _, root, _, _ = fixture()
    model = FakeModel(train, root, 17, {}, utc('2024-01-01'))
    model.diagnostics['history'] = [{'partial_epoch': True}]
    with pytest.raises(TimeoutError, match='Resource-truncated'):
        _validate_fit(model)


def test_robustness_feedback_is_typed_and_survives_durable_memory(tmp_path):
    from factorlab.discovery.research_memory import (
        ResearchMemory,
        clean_metrics,
        to_development_history,
    )
    _, _, root, _, spec = fixture()
    fields = {'robustness_rank': -.05, 'mean_relative_gain': .03, 'relative_gain_std': .08,
              'winning_fold_fraction': 2 / 3, 'worst_primary_relative_degradation': .09,
              'robust_eligible': False, 'selection_reason': 'stability_penalty_not_met'}
    row = {'partition': 'development', 'evidence_cutoff': spec.dev_end, 'status': 'ok',
           'protocol_hash': spec.digest, 'metrics': {'ret': score(1)['ret'], **fields}, **fields}
    clean = search._clean_history([row], spec)[0]
    assert {key: clean[key] for key in fields} == fields
    assert {key: clean['metrics'][key] for key in fields} == fields
    assert search._clean_history([clean], spec) == [clean]
    memory = ResearchMemory(tmp_path / 'robust-memory.sqlite')
    memory.record_action_result(task=spec.tasks[0], evidence_cutoff=spec.dev_end,
                                protocol_hash=spec.digest, parent_hash=root.digest, candidate_id='candidate',
                                action={'operation': 'width', 'arguments': {'name': 'root', 'width': 5}},
                                structure=asdict(root), metrics=clean['metrics'])
    retrieved = memory.retrieve(spec.tasks[0], spec.dev_end)
    assert retrieved[0]['metrics'] == clean['metrics']
    history = to_development_history(retrieved, spec)
    assert {key: history[0][key] for key in fields} == fields
    assert {key: search._clean_history(history, spec)[0][key] for key in fields} == fields
    for invalid in ({'robustness_rank': True}, {'robustness_rank': float('inf')},
                    {'selection_reason': 'confirmation loss improved'}, {'robust_eligible': 1},
                    {'winning_fold_fraction': 2}, {'relative_gain_std': -1},
                    {'robustness_rank': {'future_loss': .1}}):
        with pytest.raises(ValueError):
            search._clean_history([{**row, **invalid}], spec)
        with pytest.raises(ValueError):
            clean_metrics(invalid)
    with pytest.raises(ValueError, match='mature development'):
        search._clean_history([{**row, 'evidence_cutoff': spec.end}], spec)


def test_external_request_worker_preserves_only_typed_robustness_feedback(tmp_path):
    import json

    from factorlab.discovery.agent_worker import sanitize_request
    _, _, root, _, spec = fixture()
    fields = {'robustness_rank': -.05, 'mean_relative_gain': .03, 'relative_gain_std': .08,
              'winning_fold_fraction': 2 / 3, 'worst_primary_relative_degradation': .09,
              'robust_eligible': False, 'selection_reason': 'stability_penalty_not_met'}
    row = {'partition': 'development', 'evidence_cutoff': spec.dev_end, 'status': 'ok',
           'protocol_hash': spec.digest, 'metrics': {'ret': score(1)['ret']}, **fields}
    with pytest.raises(TimeoutError):
        search.propose(root, spec, [row], 0, 'external', tmp_path, timeout=0, features=('x',))
    packet = json.loads((tmp_path / 'request-0.json').read_text('utf-8'))
    sanitized = sanitize_request(packet)
    assert {key: sanitized['history'][0][key] for key in fields} == fields
    packet['history'][0]['selection_reason'] = 'future market results'
    packet['request_hash'] = fingerprint({key: value for key, value in packet.items() if key != 'request_hash'})
    with pytest.raises(ValueError, match='bounded reason'):
        sanitize_request(packet)


def test_fold_callbacks_include_reference_and_rejected_candidate(monkeypatch):
    train, dev, root, other, spec = fixture()
    calls, folds = [], []
    losses = {root.digest: {'ret': [1] * 3, 'aux': [1] * 3},
              other.digest: {'ret': [.8] * 3, 'aux': [2] * 3}}

    def retain(model, candidate, fold, validation):
        assert model.digest == fold['digest'] and fold['status'] == 'ok'
        assert len(validation.meta) == fold['validation_rows']
        folds.append((candidate['architecture_hash'], fold['fold']))

    _, result = population.search_population(train, dev, (root, other), spec,
                 fit_candidate=fake_fitter(calls, losses), on_fold=retain)
    assert folds == [(architecture.digest, index) for architecture in (root, other) for index in range(3)]
    assert result['records'][1]['selection_reason'] == 'auxiliary_degradation'
    assert result['fallback_to_root']


def test_fold_callback_failure_invalidates_candidate():
    train, dev, root, other, spec = fixture()
    losses = {root.digest: {'ret': [1] * 3, 'aux': [1] * 3},
              other.digest: {'ret': [.8] * 3, 'aux': [1] * 3}}

    def failure(model, candidate, fold, validation):
        if candidate['architecture_hash'] == other.digest:
            raise RuntimeError('declared fold persistence failure')

    _, result = population.search_population(train, dev, (root, other), spec,
                  fit_candidate=fake_fitter([], losses), on_fold=failure)
    assert result['records'][1]['status'] == 'failed'
    assert result['fallback_to_root'] and result['architecture'] == asdict(root)


def test_population_coldfits_each_forward_fold_all_seeds_then_final_refit():
    train, dev, root, other, spec = fixture()
    calls, saved = [], []
    losses = {root.digest: {'ret': [1] * 3, 'aux': [1] * 3},
              other.digest: {'ret': [.8] * 3, 'aux': [1] * 3}}
    best, result = population.search_population(train, dev, (root, other), spec,
                         fit_candidate=fake_fitter(calls, losses), initial_models={root.digest: object()},
                         inherit_weights=True, on_candidate=lambda model, record: saved.append((model, record)))
    assert best is not None and result['architecture'] == asdict(other)
    assert len(calls) == result['fit_attempts'] == result['successful_fits'] == 14
    assert result['budget_seed_fits'] == 14
    assert [row['seed'] for row in calls] == list(spec.training_seeds) * 7
    assert all(row['early'].meta.available_at.max() < utc('2024-01-01') + pd.Timedelta(hours=stop)
               for row, stop in zip(calls[:6], (300, 300, 400, 400, 500, 500)))
    assert calls[-1]['train'] is train and calls[-1]['early'] is dev
    assert len(saved) == 1 and saved[0][0] is best
    assert result['initial_models_reused'] is False and result['inherit_weights'] is False
    assert result['records'][1]['comparison']['eligible']
    assert all(row['status'] == 'ok' for row in result['records'][1]['folds'])
    assert result['final_refit']['status'] == 'ok'


def test_rank_prefers_stability_and_reference_never_changes():
    train, dev, root, stable, spec = fixture(candidates=3, seeds=(17,))
    volatile = RepresentationSpec((ModuleSpec('volatile', 'summary', 2, 6, 4),))
    calls = []
    losses = {root.digest: {'ret': [1] * 3, 'aux': [1] * 3},
              stable.digest: {'ret': [.8] * 3, 'aux': [1] * 3},
              volatile.digest: {'ret': [.1, .1, 1.05], 'aux': [1] * 3}}
    _, result = population.search_population(train, dev, (root, stable, volatile), spec,
                                             fit_candidate=fake_fitter(calls, losses))
    assert result['architecture'] == asdict(stable)
    assert result['records'][2]['comparison']['relative_gains'] == pytest.approx([.9, .9, -.05])
    assert result['records'][2]['comparison']['eligible']


def test_failed_seed_rejects_entire_candidate_and_falls_back_to_cold_root():
    train, dev, root, other, spec = fixture()
    calls = []
    losses = {root.digest: {'ret': [1] * 3, 'aux': [1] * 3},
              other.digest: {'ret': [.8] * 3, 'aux': [1] * 3}}
    fit = fake_fitter(calls, losses)

    def failure(*args, **kwargs):
        model = fit(*args, **kwargs)
        if args[2] == other and args[4] == (29,):
            raise RuntimeError('declared failed seed')
        return model

    _, result = population.search_population(train, dev, (root, other), spec, fit_candidate=failure)
    assert result['architecture'] == asdict(root)
    assert result['fallback_to_root'] and result['fallback_reason'] == 'no_candidate_passed_robust_gates'
    assert result['records'][1]['status'] == 'failed' and len(result['records'][1]['folds']) == 1
    assert result['fit_attempts'] == 10 and result['successful_fits'] == 9


def test_resource_truncated_candidate_invalid_and_failed_reference_returns_no_model():
    train, dev, root, other, spec = fixture()
    losses = {root.digest: {'ret': [1] * 3, 'aux': [1] * 3},
              other.digest: {'ret': [.8] * 3, 'aux': [1] * 3}}
    _, result = population.search_population(train, dev, (root, other), spec,
                    fit_candidate=fake_fitter([], losses, {other.digest: 'time_budget'}))
    assert result['records'][1]['status'] == 'failed'
    assert result['architecture'] == asdict(root)
    best, result = population.search_population(train, dev, (root, other), spec,
                    fit_candidate=fake_fitter([], losses, {root.digest: 'time_budget'}))
    assert best is None and result['fallback_reason'] == 'reference_fold_fit_failed'
    assert result['newly_fitted'] == result['fit_attempts'] == 1


def test_final_refit_failure_returns_no_fold_model():
    train, dev, root, _, spec = fixture(candidates=1)
    losses = {root.digest: {'ret': [1] * 3, 'aux': [1] * 3}}
    fit = fake_fitter([], losses)

    def failure(*args, **kwargs):
        if args[0] is train:
            raise TimeoutError('declared final refit failure')
        return fit(*args, **kwargs)

    best, result = population.search_population(train, dev, root, spec, fit_candidate=failure)
    assert best is None and result['final_refit']['status'] == 'failed'
    assert result['best_candidate_id'] is None


def test_program_proposal_path_clean_history_remains_typed(monkeypatch):
    train, dev, root, _, spec = fixture(seeds=(17,))
    calls = []

    def propose(parent, protocol, history, iteration, *args, **kwargs):
        assert kwargs['program_search'] is True
        assert search._clean_history(history, protocol) == history
        assert all('folds' not in row and 'fit_attempts' not in row for row in history)
        return search.ResearchProposal(protocol.digest, parent.digest, protocol.dev_end,
                    'width', 'bounded change', 'reject if unstable', 'test', {'name': 'root', 'width': 5})

    child = replace(root, modules=(replace(root.modules[0], width=5),))
    losses = {root.digest: {'ret': [1] * 3, 'aux': [1] * 3},
              child.digest: {'ret': [.8] * 3, 'aux': [1] * 3}}
    monkeypatch.setattr(search, 'propose', propose)
    _, result = population.search_population(train, dev, root, spec,
                     fit_candidate=fake_fitter(calls, losses), program_search=True)
    assert result['architecture'] == asdict(child) and result['proposal_sources'] == ['test']


def test_real_tiny_forward_fits_have_three_distinct_cold_fit_cutoffs():
    train, dev, root, _, spec = fixture(candidates=1, seeds=(17,))
    best, result = population.search_population(train, dev, root, spec)
    assert best is not None
    folds = result['records'][0]['folds']
    assert len({fold['digest'] for fold in folds}) == 3
    for fold in folds:
        diagnostic = fold['training'][0]
        assert not diagnostic['weight_transfer']['requested']
        assert utc(fold['fit_metadata'][0]['development_feature_cutoff']) < utc(fold['information_cutoff'])
    assert result['fit_attempts'] == 4 and result['final_refit']['status'] == 'ok'
