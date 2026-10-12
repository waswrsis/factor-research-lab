"""Artificial fixed-control engineering checks; not forward/market evidence."""
import copy
import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_v7_nonlinear_channel import artificial_data

from factorlab.discovery import known_linear_reference as module
from factorlab.discovery.convex_eval import fit_ridge
from factorlab.discovery.joint_pool import hour_equal_weights
from factorlab.discovery.known_library import FEATURE_NAMES, FactorFrame
from factorlab.discovery.nonlinear_channel import ChannelBudget
from factorlab.protocol import fingerprint


def fit(data, name='HAR_cal', **kwargs):
    return module.fit_known_linear(data, '2025-04-01T00:00Z',
        module.FrozenLinearRecipe(name, data.spec), budget=kwargs.get('budget', ChannelBudget()))


@pytest.mark.parametrize('name', ['HAR_cal', 'K0'])
def test_exact_existing_ladder_ridge_and_immutable_roundtrip(name, tmp_path):
    data = artificial_data()
    budget = ChannelBudget()
    model = fit(data, name, budget=budget)
    columns = module.HAR_COLUMNS if name == 'HAR_cal' else FEATURE_NAMES
    indices = [FEATURE_NAMES.index(n) for n in columns]
    design = FactorFrame(data.frame.values[:, indices], data.frame.metadata, columns,
        tuple(data.frame.bypass_mask[i] for i in indices),
        fingerprint({'library': data.frame.schema_hash, 'columns': columns}))
    mask = data.train_mask('2025-04-01T00:00Z')
    expected = fit_ridge(design.take(mask), data.target[mask], ridge=data.spec.ridge,
                         sample_weight=hour_equal_weights(data.frame.metadata.loc[mask]))
    np.testing.assert_array_equal(model.predictor.coefficients, expected.coefficients)
    np.testing.assert_allclose(model.predict(data.frame), expected.predict(design), rtol=1e-14, atol=0)
    assert budget.counts['solve'] == 1
    assert model.pipeline_id == model.recipe.pipeline_id
    assert model.receipt['training']['train_start'] == '2025-01-01T00:00:00+00:00'
    assert 'serving' not in model.receipt and 'formal' not in model.receipt
    path = model.save(tmp_path / 'model.json')
    restored = module.KnownLinearPredictor.load(path)
    assert restored.digest == model.digest and restored.schema_hash == model.schema_hash
    np.testing.assert_array_equal(restored.predict(data.frame), model.predict(data.frame))
    model.save(path)
    with pytest.raises(ValueError, match='Immutable'):
        fit(replace(data, target=data.target * 2), name).save(path)


def test_outside_window_future_and_immature_labels_cannot_change_cold_fit():
    data = artificial_data()
    original = fit(data)
    mask = data.train_mask('2025-04-01T00:00Z')
    target = data.target.copy()
    target[~mask] *= 999
    values = data.frame.values.copy()
    values[~mask] *= 3
    changed = replace(data, target=target, frame=replace(data.frame, values=values))
    assert fit(changed).digest == original.digest
    maturity = data.maturity.to_series(index=np.arange(len(mask)))
    maturity.loc[mask] = pd.Timestamp('2025-05-01T00:00Z')
    with pytest.raises(ValueError, match='Insufficient mature'):
        fit(replace(data, maturity=pd.DatetimeIndex(maturity)))


def test_har_retains_full_k0_support_even_unused_column_missing():
    data = artificial_data()
    values = data.frame.values.copy()
    values[250, FEATURE_NAMES.index('rv_720')] = np.nan
    changed = replace(data, frame=replace(data.frame, values=values))
    model = fit(changed)
    pred, mask = model.predict_history(changed.frame)
    assert not mask[250] and np.isnan(pred[250])
    assert model.receipt['training']['rows'] == int(changed.train_mask('2025-04-01T00:00Z').sum())


def test_unequal_symbol_support_uses_equal_hours():
    data = artificial_data()
    idx = np.arange(len(data.target))
    extra = idx[::3]
    meta = data.frame.metadata.iloc[np.r_[idx, extra]].reset_index(drop=True)
    meta.loc[len(idx):, 'symbol'] = 'Y'
    meta.loc[len(idx):, 'sample_id'] += '-Y'
    frame = replace(data.frame, values=data.frame.values[np.r_[idx, extra]], metadata=meta)
    data = replace(data, frame=frame, target=np.r_[data.target, data.target[extra] * 3],
                   maturity=data.maturity[np.r_[idx, extra]], observed=data.observed[np.r_[idx, extra]])
    model = fit(data)
    mask = data.train_mask('2025-04-01T00:00Z')
    design = module._design(frame, model.recipe).take(mask)
    weighted = fit_ridge(design, data.target[mask], sample_weight=hour_equal_weights(design.metadata))
    unweighted = fit_ridge(design, data.target[mask])
    np.testing.assert_array_equal(model.predictor.coefficients, weighted.coefficients)
    assert not np.allclose(model.predictor.coefficients, unweighted.coefficients)


def test_budget_required_and_failure_never_refunds(monkeypatch):
    data = artificial_data()
    recipe = module.FrozenLinearRecipe(rolling_spec=data.spec)
    with pytest.raises(TypeError, match='budget'):
        module.fit_known_linear(data, '2025-04-01T00:00Z', recipe)
    with pytest.raises(ValueError, match='budget'):
        module.fit_known_linear(data, '2025-04-01T00:00Z', recipe, budget=None)
    budget = ChannelBudget()
    def failure(*args, **kwargs):
        raise RuntimeError('synthetic numerical failure')
    monkeypatch.setattr(module, 'fit_ridge', failure)
    with pytest.raises(RuntimeError):
        fit(data, budget=budget)
    assert budget.counts['solve'] == 1


@pytest.mark.parametrize('mutation', ['columns', 'support', 'extra'])
def test_recipe_cannot_silently_change_contract(mutation):
    value = module.FrozenLinearRecipe().to_dict()
    if mutation == 'columns':
        value['columns'].reverse()
    else:
        value[mutation] = 'changed'
    with pytest.raises(ValueError):
        module.FrozenLinearRecipe.from_dict(value)


def test_schema_and_postload_mutation_rejected():
    data = artificial_data()
    model = fit(data)
    with pytest.raises(ValueError, match='raw K0'):
        model.predict(replace(data.frame, schema_hash='standardized-legacy'))
    model.receipt['solve_count'] = 0
    with pytest.raises(ValueError, match='mutated'):
        model.predict(data.frame)


@pytest.mark.parametrize('mutation', ['negative_ridge', 'shape', 'extra', 'receipt'])
def test_rehashed_unsupported_saved_model_rejected(mutation, tmp_path):
    value = copy.deepcopy(fit(artificial_data()).to_dict())
    if mutation == 'negative_ridge':
        value['predictor']['ridge'] = -1
    elif mutation == 'shape':
        value['predictor']['coefficients'].pop()
    elif mutation == 'extra':
        value['runtime_admitted'] = True
    else:
        value['receipt']['training']['train_start'] = '2024-12-01T00:00:00+00:00'
    path = tmp_path / 'corrupt.json'
    path.write_text(json.dumps({'model': value, 'digest': fingerprint(value)}))
    with pytest.raises(ValueError):
        module.KnownLinearPredictor.load(path)


def test_import_and_model_load_do_not_load_torch(tmp_path):
    path = fit(artificial_data()).save(tmp_path / 'model.json')
    code = ('import sys; from factorlab.discovery.known_linear_reference import KnownLinearPredictor; '
            'KnownLinearPredictor.load(sys.argv[1]); assert "torch" not in sys.modules')
    result = subprocess.run([sys.executable, '-c', code, str(path)], capture_output=True,
                            text=True, timeout=20, check=False)
    assert result.returncode == 0, result.stderr
