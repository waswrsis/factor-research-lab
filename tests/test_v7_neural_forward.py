"""Artificial cold-fit and safe complete-artifact checks; never promotion evidence."""
import copy
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_v7_nonlinear_channel import artificial_data

from factorlab.discovery import neural_forward as module
from factorlab.discovery.joint_pool import FrozenPool, PoolMember
from factorlab.discovery.known_library import FactorFrame
from factorlab.discovery.neural_forward import FrozenNeuralRecipe, NeuralForwardModel
from factorlab.discovery.nonlinear_channel import (
    ChannelBudget,
    ResidualConfig,
    StructureSpec,
    _train_residual,
)
from factorlab.protocol import fingerprint


@pytest.fixture(scope='module')
def fitted():
    data = artificial_data()
    recipe = FrozenNeuralRecipe(rolling_spec=data.spec,
        config=ResidualConfig(enabled=True, structures=(StructureSpec(4, 2, 1),), seeds=(17, 29), epochs=2))
    budget = ChannelBudget()
    model = module._fit_neural_forward_cold(data, '2025-04-01T00:00:00Z', recipe, budget)
    return data, recipe, model, budget


def test_public_fit_cannot_bypass_missing_actual_forward_ticket():
    with pytest.raises(TypeError):
        module.fit_neural_forward(None, None, None, verified=True)


def test_recipe_roundtrip_and_output_identity(fitted):
    _, recipe, _, _ = fitted
    restored = FrozenNeuralRecipe.from_dict(json.loads(json.dumps(recipe.to_dict())))
    assert restored.pipeline_id == recipe.pipeline_id
    assert replace(recipe, output='encoder_ridge').pipeline_id != recipe.pipeline_id
    assert replace(recipe, config=replace(recipe.config, seeds=(29, 17))).pipeline_id != recipe.pipeline_id
    assert recipe.budget_limits == {'solve': 5, 'neural_fit': 2, 'epoch': 4, 'seconds': 300}


def test_multi_expert_and_multi_structure_recipe_are_not_admitted(fitted):
    _, recipe, _, _ = fitted
    for structures in [(StructureSpec(4, 2, 2),), (StructureSpec(4, 2), StructureSpec(8, 2))]:
        with pytest.raises(ValueError, match='single-expert'):
            replace(recipe, config=replace(recipe.config, structures=structures))


def test_actual_cold_fit_uses_exact_three_months_and_charged_seed_work(fitted):
    data, recipe, model, budget = fitted
    assert model.receipt['training']['train_start'] == '2025-01-01T00:00:00+00:00'
    assert model.receipt['cutoff'] == '2025-04-01T00:00:00+00:00'
    assert model.receipt['training']['rows'] == int(data.train_mask(model.receipt['cutoff']).sum())
    assert budget.counts == {'tree_fit': 0, 'neural_fit': 2, 'epoch': 4, 'solve': 5}
    assert model.pipeline_id == recipe.pipeline_id
    assert model.task_support == ('volatility',) and model.receipt['test_only']
    assert not model.receipt['runtime_admitted'] and not model.receipt['automatic_adoption']


def test_save_restore_predictions_are_identical_and_artifact_is_json(fitted, tmp_path):
    data, _, model, _ = fitted
    path = tmp_path / 'complete.json'
    model.save(path)
    restored = NeuralForwardModel.load(path)
    assert restored.digest == model.digest and restored.schema_hash == model.schema_hash
    assert np.array_equal(restored.predict(data.frame), model.predict(data.frame), equal_nan=True)
    assert json.loads(path.read_text())['model']['baseline']['kind'] == 'v7_joint_pool'


def test_future_and_old_outside_window_labels_cannot_change_cold_model(fitted):
    data, recipe, model, _ = fitted
    target = data.target.copy()
    target[~data.train_mask('2025-04-01T00:00:00Z')] *= 1000
    altered = replace(data, target=target)
    repeated = module._fit_neural_forward_cold(altered, '2025-04-01T00:00:00Z', recipe, ChannelBudget())
    assert repeated.digest == model.digest


def test_epoch_zero_returns_exact_reference_for_both_frozen_outputs(fitted):
    data, recipe, _, _ = fitted
    for output in ('residual', 'encoder_ridge'):
        frozen = replace(recipe, config=replace(recipe.config, epochs=0), output=output)
        model = module._fit_neural_forward_cold(data, '2025-04-01T00:00:00Z', frozen, ChannelBudget())
        actual, mask = model.predict_history(data.frame)
        expected, reference_mask = model.baseline.predict_history(data.frame)
        assert np.array_equal(mask, reference_mask)
        assert np.array_equal(actual, expected, equal_nan=True)
        assert all(row['selected_epoch'] == 0 for row in model.seeds)


def test_numpy_residual_prediction_matches_torch_float64(fitted):
    import torch
    data, recipe, model, _ = fitted
    design, mask = model.baseline.feature_history(data.frame)
    X = model.baseline.predictor.scaler.transform(design.take(mask))
    base = model.baseline.predictor.predict(design.take(mask))
    components = []
    for row in model.seeds:
        if row['selected_epoch'] == 0:
            components.append(base)
            continue
        network = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 4), torch.nn.Tanh(),
                                      torch.nn.Linear(4, 2), torch.nn.Tanh()).double()
        network.load_state_dict({k: torch.tensor(v, dtype=torch.float64) for k, v in row['encoder'].items()})
        with torch.no_grad():
            code = network(torch.tensor(X, dtype=torch.float64))
            correction = (code @ torch.tensor(row['head']['weight'], dtype=torch.float64).T).numpy().ravel()
        components.append(module._decode(np.log(base + 1e-8) + correction + row['head']['bias'][0])[0])
    assert any(row['selected_epoch'] > 0 for row in model.seeds)
    expected = module._decode(np.log(np.asarray(components) + 1e-8).mean(axis=0))[0]
    np.testing.assert_allclose(model.predict_history(data.frame)[0][mask], expected, rtol=1e-12, atol=1e-12)
    assert recipe.output == 'residual'


def test_cold_training_matches_existing_residual_algorithm_for_each_seed(fitted):
    import torch
    data, recipe, model, _ = fitted
    design, valid = model.baseline.feature_history(data.frame)
    mask = data.train_mask(model.receipt['cutoff']) & valid
    reference = model.baseline.predict_history(data.frame)[0]
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for row in model.seeds:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(row['seed'])
                # Training-as-score here is solely a private-algorithm parity
                # test, never a held-out score or reported S candidate result.
                compared = _train_residual(torch, data, {'fit_cutoff': model.receipt['cutoff']},
                    model.baseline, design, reference, mask, mask, recipe.config.structures[0],
                    row['seed'], recipe.config, ChannelBudget())
            assert compared['selected_epoch'] == row['selected_epoch']
            assert compared['data_order_hashes'] == row['data_order_hashes']
            assert compared['fitted_parameters']['encoder'] == row['encoder']
            assert compared['fitted_parameters']['experts']['0.weight'] == row['head']['weight']
    finally:
        torch.set_num_threads(old_threads)


def test_encoder_readout_is_serialized_and_predicts_through_same_convex_family(fitted):
    data, recipe, _, _ = fitted
    recipe = replace(recipe, output='encoder_ridge')
    model = module._fit_neural_forward_cold(data, '2025-04-01T00:00:00Z', recipe, ChannelBudget())
    predictions, mask = model.predict_history(data.frame)
    assert np.isfinite(predictions[mask]).all()
    assert all(s['trained_readout']['kind'] == s['initial_readout']['kind'] == 'v7_log_ridge' for s in model.seeds)
    for state in model.seeds:
        assert len(state['trained_readout']['coefficients']) == len(data.frame.feature_names) + 2


@pytest.mark.parametrize('damage', ['weight_shape', 'extra_op', 'readout_shape', 'epoch0_head', 'wrong_cutoff', 'admitted'])
def test_recomputed_outer_hash_does_not_bypass_complete_model_validation(fitted, damage):
    _, _, model, _ = fitted
    value = model.to_dict()
    seed = value['seeds'][0]
    if damage == 'weight_shape':
        seed['encoder']['0.weight'] = [[1.]]
    elif damage == 'extra_op':
        seed['encoder']['python_eval'] = 'not executable'
    elif damage == 'readout_shape':
        seed['trained_readout']['coefficients'] = [1.]
    elif damage == 'epoch0_head':
        seed['selected_epoch'] = 0
        seed['head']['bias'] = [1.]
    elif damage == 'wrong_cutoff':
        value['receipt']['cutoff'] = '2025-05-01T00:00:00+00:00'
    else:
        value['receipt']['test_only'] = False
        value['receipt']['runtime_admitted'] = True
    with pytest.raises(ValueError):
        NeuralForwardModel.from_dict(value)


def test_nested_artifact_mutation_cannot_silently_change_predictions(fitted):
    data, _, model, _ = fitted
    copy_model = NeuralForwardModel.from_dict(model.to_dict())
    copy_model.seeds[0]['head']['bias'][0] += 1
    with pytest.raises(ValueError, match='modified'):
        copy_model.predict_history(data.frame)


def test_bad_input_order_or_gap_is_never_silently_filled(fitted):
    data, _, model, _ = fitted
    wrong = FactorFrame(data.frame.values[:, ::-1], data.frame.metadata,
                        data.frame.feature_names[::-1], data.frame.bypass_mask[::-1], data.frame.schema_hash)
    with pytest.raises(ValueError, match='feature order'):
        model.predict_history(wrong)
    values = data.frame.values.copy()
    values[10] = np.nan
    predictions, mask = model.predict_history(replace(data.frame, values=values))
    assert not mask[10] and np.isnan(predictions[10])


def test_insufficient_budget_fails_before_any_cold_work(fitted):
    data, recipe, _, _ = fitted
    budget = ChannelBudget(counts={'tree_fit': 0, 'neural_fit': 0, 'epoch': 0, 'solve': 1599})
    before = copy.deepcopy(budget.counts)
    with pytest.raises(ValueError, match='Insufficient nonrefundable'):
        module._fit_neural_forward_cold(data, '2025-04-01T00:00:00Z', recipe, budget)
    assert budget.counts == before


def test_failed_neural_fit_retains_reservations_and_restores_torch_global_state(fitted, monkeypatch):
    import torch
    data, recipe, _, _ = fitted
    before = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled())
    monkeypatch.setattr(module, '_fit_seed', lambda *a: (_ for _ in ()).throw(RuntimeError('synthetic failure')))
    budget = ChannelBudget()
    with pytest.raises(RuntimeError, match='synthetic failure'):
        module._fit_neural_forward_cold(data, '2025-04-01T00:00:00Z', recipe, budget)
    assert budget.counts['solve'] == budget.counts['neural_fit'] == 1
    assert (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()) == before


def test_loading_and_prediction_in_fresh_process_do_not_import_torch(fitted, tmp_path):
    _, _, model, _ = fitted
    path = tmp_path / 'portable.json'
    model.save(path)
    script = '''
import sys,numpy as np,pandas as pd
from factorlab.discovery.neural_forward import NeuralForwardModel
from factorlab.discovery.known_library import FactorFrame,FEATURE_NAMES,library_schema
from factorlab.protocol import fingerprint
m=NeuralForwardModel.load(sys.argv[1])
f=FactorFrame(np.ones((2,len(FEATURE_NAMES))),pd.DataFrame({'sample_id':['a','b'],'symbol':['X','X'],
    'available_at':pd.date_range('2025-04-02',periods=2,freq='h',tz='UTC')}),schema_hash=fingerprint(library_schema()))
assert np.isfinite(m.predict(f)).all()
assert 'torch' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', script, str(path)], capture_output=True,
        env={**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1])}, timeout=15, check=False)
    assert result.returncode == 0, result.stderr.decode(errors='replace')


def test_checksums_and_strict_recipe_fields_are_verified(fitted, tmp_path):
    _, recipe, model, _ = fitted
    changed = recipe.to_dict()
    changed['unregistered_rule'] = True
    with pytest.raises(ValueError, match='fields'):
        FrozenNeuralRecipe.from_dict(changed)
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps({'model': model.to_dict(), 'digest': fingerprint('incorrect')}))
    with pytest.raises(ValueError, match='checksum'):
        NeuralForwardModel.load(path)


@pytest.mark.parametrize('damage', ['not_mapping', 'bool', 'negative', 'missing', 'oversized',
                                  'missing_solve', 'missing_fit', 'missing_epoch'])
def test_complete_artifact_requires_plausible_nonrefundable_budget_counts(fitted, damage):
    _, _, model, _ = fitted
    value = model.to_dict()
    counts = value['receipt']['budget_counts']
    if damage == 'not_mapping':
        value['receipt']['budget_counts'] = []
    elif damage == 'bool':
        counts['tree_fit'] = False
    elif damage == 'negative':
        counts['tree_fit'] = -1
    elif damage == 'missing':
        del counts['epoch']
    elif damage == 'oversized':
        counts['solve'] = 1601
    else:
        counts[damage.removeprefix('missing_').replace('fit', 'neural_fit')] = 0
    with pytest.raises(ValueError, match='budget receipt'):
        NeuralForwardModel.from_dict(value)


def test_nonempty_frozen_formula_pool_is_part_of_complete_forward_model(fitted, tmp_path):
    data, recipe, _, _ = fitted
    formula = {'version': 'causal-operator-program-v1', 'feature_names': list(data.frame.feature_names),
               'parameters': {'alpha': {'lower': .1, 'upper': 2., 'initial': .5}},
               'outputs': [{'op': 'mul', 'left': {'op': 'param', 'name': 'alpha'},
                            'right': {'op': 'tanh', 'arg': {'op': 'last',
                                'arg': {'op': 'feature', 'name': 'rv_24'}}}}]}
    pool = FrozenPool((PoolMember(formula, 1, parameters={'alpha': 1.2}),), version='frozen-fixture')
    changed = replace(recipe, pool=pool)
    model = module._fit_neural_forward_cold(data, '2025-04-01T00:00:00Z', changed, ChannelBudget())
    assert len(model.baseline.predictor.feature_names) == len(data.frame.feature_names) + 1
    assert model.baseline.parameters == ({'alpha': 1.2},)
    assert model.pipeline_id != recipe.pipeline_id
    path = model.save(tmp_path / 'nonempty.json')
    restored = NeuralForwardModel.load(path)
    np.testing.assert_array_equal(restored.predict(data.frame), model.predict(data.frame))
