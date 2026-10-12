"""Artificial G02 numerical checks; no production S admission or Linux proof."""
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_v7_neural_forward import fitted as fitted  # noqa: PLC0414
from test_v7_neural_forward_admission import evidence as evidence  # noqa: PLC0414
from test_v7_nonlinear_channel import artificial_data

from factorlab.discovery import neural_forward
from factorlab.discovery import neural_research as module
from factorlab.discovery.joint_pool import FrozenPool, PoolMember
from factorlab.discovery.neural_forward import FrozenNeuralRecipe, NeuralForwardModel, _joint
from factorlab.discovery.nonlinear_channel import ChannelBudget, ResidualConfig, StructureSpec


def recipe_for(data, *, epochs=1, output='residual'):
    return FrozenNeuralRecipe(rolling_spec=data.spec, output=output,
        config=ResidualConfig(enabled=True, structures=(StructureSpec(4, 2),), seeds=(17, 29), epochs=epochs))


@pytest.fixture(scope='module')
def research(tmp_path_factory):
    data = artificial_data()
    recipe = recipe_for(data)
    directory = tmp_path_factory.mktemp('neural-research') / 'attempt'
    budget = ChannelBudget()
    result = module._evaluate_artificial(data, '2025-04-01T00:00Z', recipe,
        symbols=('X',), directory=directory, budget=budget)
    return data, recipe, result, directory, budget


def test_three_cold_folds_save_full_recipe_and_all_actual_costs(research):
    data, recipe, result, directory, budget = research
    assert result['status'] == 'completed', result['folds']
    assert result['role'] == 'S' and result['test_only'] and not result['formal_claim']
    assert result['pipeline_id'] == recipe.pipeline_id
    assert [f['fit_cutoff'] for f in result['folds']] == [
        '2025-01-01T00:00:00+00:00', '2025-02-01T00:00:00+00:00', '2025-03-01T00:00:00+00:00']
    assert budget.counts == module.research_capacity(recipe) == {
        'tree_fit': 0, 'neural_fit': 6, 'epoch': 6, 'solve': 18}
    assert result['nominal_rows'] == 90 * 24 and result['coverage'] < .1
    for index, fold in enumerate(result['folds']):
        model = NeuralForwardModel.load(directory / f'fold-{index}-model.json')
        assert model.receipt['fit'] == 'cold' and model.receipt['cutoff'] == fold['fit_cutoff']
        assert model.recipe.pipeline_id == recipe.pipeline_id
        assert fold['seed_checkpoints'][0]['seed'] == 17
        assert model.receipt['training']['rows'] == int(data.train_mask(fold['fit_cutoff']).sum())


def test_reload_recomputes_exact_hourly_ensemble_and_controls(research):
    data, _, result, directory, _ = research
    for i, fold in enumerate(result['folds']):
        model = NeuralForwardModel.load(directory / f'fold-{i}-model.json')
        k0 = _joint(json.loads((directory / f'fold-{i}-k0.json').read_text()))
        scored = module.score_research_fold(data, model, k0,
            start=fold['fit_cutoff'], end=fold['score_end'], symbols=('X',))
        assert scored['hourly'] == fold['hourly']
        assert scored['predictions'] == fold['predictions']
        assert set(scored['S']) == {'K0', 'Pv'}
        assert {'initial_encoder_ridge', 'trained_encoder_ridge'} <= set(scored['losses'])
    hours = [h for f in result['folds'] for h in f['hourly']]
    assert result['losses']['ensemble'] == np.mean([h['losses']['ensemble'] for h in hours])


def test_ensemble_loss_is_not_average_seed_loss_when_errors_cancel(research):
    data, _, result, directory, _ = research
    fold = result['folds'][0]
    value = NeuralForwardModel.load(directory / 'fold-0-model.json').to_dict()
    for seed, bias in zip(value['seeds'], (-1., 1.), strict=True):
        seed['selected_epoch'] = 1
        seed['training_loss'] = seed['epoch0_loss'] / 2  # Synthetic checkpoint metadata, not a fitted-performance claim.
        seed['head']['weight'] = [[0., 0.]]
        seed['head']['bias'] = [bias]
    model = NeuralForwardModel.from_dict(value)
    target = data.target.copy()
    score = data.score_mask(fold['fit_cutoff'], fold['score_end'], fold['score_end'])
    reference, _ = model.baseline.predict_history(data.frame)
    target[score] = reference[score]
    changed = replace(data, target=target)
    k0 = _joint(json.loads((directory / 'fold-0-k0.json').read_text()))
    scored = module.score_research_fold(changed, model, k0,
        start=fold['fit_cutoff'], end=fold['score_end'], symbols=('X',))
    assert scored['losses']['ensemble'] < 1e-25
    assert scored['mean_seed_loss_diagnostic'] > .99


def test_model_from_another_fold_cannot_be_relabelled(research):
    data, _, result, directory, _ = research
    model = NeuralForwardModel.load(directory / 'fold-0-model.json')
    k0 = _joint(json.loads((directory / 'fold-0-k0.json').read_text()))
    fold = result['folds'][1]
    with pytest.raises(ValueError, match='own cold cutoff'):
        module.score_research_fold(data, model, k0, start=fold['fit_cutoff'], end=fold['score_end'], symbols=('X',))


def test_future_and_outside_first_training_window_do_not_change_first_cold_model(research, tmp_path):
    data, recipe, result, _, _ = research
    target = data.target.copy()
    target[~data.train_mask('2025-01-01T00:00Z')] *= 7
    changed = replace(data, target=target)
    again = module._evaluate_artificial(changed, '2025-04-01T00:00Z', recipe,
        symbols=('X',), directory=tmp_path / 'changed', budget=ChannelBudget())
    assert again['folds'][0]['model_id'] == result['folds'][0]['model_id']
    assert again['folds'][0]['predictions']['ensemble'] == result['folds'][0]['predictions']['ensemble']


def test_after_development_cutoff_labels_do_not_enter_manifest_or_any_fit(research, tmp_path):
    data, recipe, result, _, _ = research
    target = data.target.copy()
    future = np.asarray(data.maturity > pd.Timestamp('2025-04-01T00:00Z'))
    target[future] *= 19
    changed = replace(data, target=target)
    again = module._evaluate_artificial(changed, '2025-04-01T00:00Z', recipe,
        symbols=('X',), directory=tmp_path / 'future', budget=ChannelBudget())
    assert again['input_id'] == result['input_id']
    assert [f['model_id'] for f in again['folds']] == [f['model_id'] for f in result['folds']]
    assert again['losses'] == result['losses']


def test_every_fit_receives_only_its_past_mature_labels(research, tmp_path, monkeypatch):
    data, recipe, _, _, _ = research
    original = neural_forward._fit_neural_forward_cold
    seen = []
    def fit(cropped, cutoff, recipe, budget):
        cutoff = pd.Timestamp(cutoff)
        assert (cropped.frame.metadata.available_at < cutoff).all()
        assert (cropped.maturity[cropped.observed] <= cutoff).all()
        assert (cropped.target[~cropped.observed] == 0).all()
        assert cropped.maturity[~cropped.observed].isna().all()
        seen.append(cutoff)
        return original(cropped, cutoff, recipe, budget)
    monkeypatch.setattr(neural_forward, '_fit_neural_forward_cold', fit)
    module._evaluate_artificial(data, '2025-04-01T00:00Z', recipe,
        symbols=('X',), directory=tmp_path / 'cropped', budget=ChannelBudget())
    assert len(seen) == 3 and len(set(seen)) == 3


def test_failed_fold_keeps_three_nominal_months_and_consumed_work(research, tmp_path, monkeypatch):
    data, recipe, _, _, _ = research
    original = neural_forward._fit_neural_forward_cold
    def fail_middle(data, cutoff, recipe, budget):
        if pd.Timestamp(cutoff).month == 2:
            budget.consume('neural_fit')
            raise RuntimeError('artificial failed middle fold')
        return original(data, cutoff, recipe, budget)
    monkeypatch.setattr(neural_forward, '_fit_neural_forward_cold', fail_middle)
    budget = ChannelBudget()
    result = module._evaluate_artificial(data, '2025-04-01T00:00Z', recipe,
        symbols=('X',), directory=tmp_path / 'failed', budget=budget)
    assert result['status'] == 'incomplete' and result['losses'] is None
    assert len(result['folds']) == 3 and result['folds'][1]['coverage'] == 0
    assert result['nominal_rows'] == 90 * 24 and budget.counts['neural_fit'] == 5
    assert result['support'] == sum(f['support'] for f in result['folds'])
    with pytest.raises(ValueError, match='cannot silently restart'):
        module._evaluate_artificial(data, '2025-04-01T00:00Z', recipe,
            symbols=('X',), directory=tmp_path / 'failed', budget=budget)


def test_capacity_includes_k0_and_refuses_before_any_fit(research, tmp_path, monkeypatch):
    data, recipe, _, _, _ = research
    monkeypatch.setattr(neural_forward, '_fit_neural_forward_cold', lambda *a: pytest.fail('Must reject before fit'))
    counts = {'tree_fit': 0, 'neural_fit': 0, 'epoch': 0, 'solve': 1583}
    with pytest.raises(ValueError, match='three complete cold folds and K0'):
        module._evaluate_artificial(data, '2025-04-01T00:00Z', recipe,
            symbols=('X',), directory=tmp_path / 'insufficient', budget=ChannelBudget(counts=counts))
    assert not (tmp_path / 'insufficient').exists()


def test_epoch_zero_preserves_reference_but_controls_remain_separate(tmp_path):
    data = artificial_data()
    recipe = recipe_for(data, epochs=0)
    result = module._evaluate_artificial(data, '2025-04-01T00:00Z', recipe,
        symbols=('X',), directory=tmp_path / 'epoch0', budget=ChannelBudget())
    assert result['status'] == 'completed'
    for fold in result['folds']:
        assert fold['predictions']['ensemble'] == fold['predictions']['Pv']
        assert all(s['epoch'] == 0 for s in fold['seed_checkpoints'])


def test_missing_rows_and_symbols_do_not_shrink_coverage_denominator(research):
    data, _, result, directory, _ = research
    fold = result['folds'][0]
    model = NeuralForwardModel.load(directory / 'fold-0-model.json')
    k0 = _joint(json.loads((directory / 'fold-0-k0.json').read_text()))
    keep = np.ones(len(data.target), bool)
    keep[np.flatnonzero(data.score_mask(fold['fit_cutoff'], fold['score_end'], fold['score_end']))[:3]] = False
    missing = replace(data, frame=data.frame.take(keep), target=data.target[keep],
                      maturity=data.maturity[keep], observed=data.observed[keep])
    scored = module.score_research_fold(missing, model, k0,
        start=fold['fit_cutoff'], end=fold['score_end'], symbols=('X', 'ABSENT'))
    assert scored['nominal_rows'] == 31 * 24 * 2
    assert scored['support'] == fold['support'] - 3


def test_public_api_requires_actual_ticket_and_worker_cannot_use_local_budget(tmp_path):
    with pytest.raises(ValueError, match='Missing or oversized S evidence'):
        module.evaluate_neural_research(ticket_path=tmp_path / 'forward-ticket.json')
    with pytest.raises(TypeError, match='actual child-only'):
        module._run_research_worker(None, None, None, symbols=('X',), directory=tmp_path, budget=ChannelBudget())


def test_output_modes_have_distinct_predeclared_identity(research):
    _, recipe, _, _, _ = research
    assert replace(recipe, output='encoder_ridge').pipeline_id != recipe.pipeline_id
    assert replace(recipe, config=replace(recipe.config, seeds=(29, 17))).pipeline_id != recipe.pipeline_id


def test_nonempty_frozen_pv_is_actual_baseline_and_not_k0(tmp_path):
    data = artificial_data()
    formula = {'version': 'causal-operator-program-v1', 'feature_names': list(data.frame.feature_names),
        'parameters': {}, 'outputs': [{'op': 'tanh', 'arg': {'op': 'last',
            'arg': {'op': 'feature', 'name': 'rv_24'}}}]}
    pool = FrozenPool((PoolMember(formula, 1),), version='fixed-Pv-fixture')
    recipe = replace(recipe_for(data, epochs=0), pool=pool, output='encoder_ridge')
    result = module._evaluate_artificial(data, '2025-04-01T00:00Z', recipe,
        symbols=('X',), directory=tmp_path / 'pv', budget=ChannelBudget())
    assert result['status'] == 'completed', result['folds']
    for fold in result['folds']:
        model = NeuralForwardModel.load(fold['model']['path'])
        assert model.baseline.pool.digest == pool.digest
        assert fold['pv_model_id'] != fold['k0_model_id']
        assert len(model.baseline.predictor.feature_names) == len(data.frame.feature_names) + 1


def test_unequal_symbols_per_hour_still_weights_hours_equally(research):
    data, _, result, directory, _ = research
    fold = result['folds'][0]
    model = NeuralForwardModel.load(directory / 'fold-0-model.json')
    k0 = _joint(json.loads((directory / 'fold-0-k0.json').read_text()))
    index = np.flatnonzero(data.score_mask(fold['fit_cutoff'], fold['score_end'], fold['score_end']))[0]
    metadata = data.frame.metadata.copy()
    extra = metadata.iloc[[index]].copy()
    extra['symbol'] = 'Y'
    extra['sample_id'] = 'second-symbol-on-one-hour'
    frame = replace(data.frame, values=np.vstack([data.frame.values, data.frame.values[index]]),
                    metadata=pd.concat([metadata, extra], ignore_index=True))
    extended = replace(data, frame=frame, target=np.append(data.target, data.target[index] * np.exp(4)),
        observed=np.append(data.observed, True), maturity=data.maturity.append(data.maturity[[index]]))
    scored = module.score_research_fold(extended, model, k0,
        start=fold['fit_cutoff'], end=fold['score_end'], symbols=('X', 'Y'))
    hourly = np.mean([h['losses']['ensemble'] for h in scored['hourly']])
    rows = np.square(np.log(np.asarray(scored['targets']) + 1e-8)
                     - np.log(np.asarray(scored['predictions']['ensemble']) + 1e-8)).mean()
    assert scored['losses']['ensemble'] == hourly
    assert not np.isclose(hourly, rows)
    assert scored['nominal_rows'] == 31 * 24 * 2


def test_research_origin_cannot_borrow_real_forward_admission_fixture(evidence):
    from factorlab.discovery.neural_forward_admission import verify_forward_admission
    value = evidence.model.to_dict()
    value['receipt'].update(version='v7-neural-research-payload-v1', payload_origin='isolated_research_worker')
    research_model = NeuralForwardModel.from_dict(value)
    assert research_model.pipeline_id == evidence.model.pipeline_id
    with pytest.raises(ValueError, match='production admission'):
        verify_forward_admission(research_model, evidence.cert)
