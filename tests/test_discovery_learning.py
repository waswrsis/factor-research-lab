"""Small actual fits: these test mechanics, not market efficacy."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from factorlab.discovery import learning
from factorlab.discovery.contracts import (
    ModuleSpec,
    RepresentationSpec,
    TaskDataset,
    TaskSpec,
    TrainingSpec,
)
from factorlab.discovery.learning import (
    FittedProbe,
    FittedResearchModel,
    _event_objective,
    event_log_probabilities,
    event_nll,
    fit,
    fit_probe,
    summary_features,
)
from factorlab.discovery.scoring import evaluate


def dataset(n=64, seed=1, shift=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 6, 2)).astype(np.float32) + shift
    tasks = (TaskSpec('vol', 'volatility', 2), TaskSpec('ret', 'return', 2),
             TaskSpec('event', 'first_event', 2, (1, 2)), TaskSpec('state', 'state'))
    times = pd.date_range('2024-01-01', periods=n, freq='h', tz='UTC')
    maturity = {t.name: (times + pd.Timedelta(hours=2)).to_numpy(dtype='datetime64[ns]')
                for t in tasks}
    return TaskDataset(x, {'vol': np.exp(0.2 * x[:, -1, 0]),
                          'ret': 2 * x[:, -1, 0] - x[:, -1, 1],
                          'event': np.where(x[:, -1, 0] > 0, 1, 0),
                          'state': x.mean((1, 2))},
                       {t.name: np.ones(n, dtype=bool) for t in tasks}, maturity,
                       pd.DataFrame({'sample_id': [f'{seed}-{i}' for i in range(n)],
                                     'symbol': 'TEST', 'available_at': times}),
                       ('a', 'b'), tasks)


def settings(**kw):
    return TrainingSpec(**{'epochs': 12, 'batch_size': 32, 'learning_rate': 0.02,
                           'l2': 0.001, 'patience': 4, 'max_seconds': 30, **kw})


def test_linear_ridge_scales_residual_roundtrip():
    train, dev = dataset(), dataset(32, 2)
    model = fit(train, dev, None, settings())
    prediction = model.predict(dev)
    assert np.mean((prediction['ret'] - dev.targets['ret']) ** 2) < 0.01
    assert (prediction['vol'] >= 0).all()
    assert prediction['event'].shape == (32, 5)
    np.testing.assert_allclose(prediction['event'].sum(1), 1, atol=1e-12)
    assert model.diagnostics['optimizer_results']['ret']['gradient_residual'] < 1e-10
    mean = np.asarray(model.metadata['input_mean'])
    np.testing.assert_allclose(mean, summary_features(train).mean(0))
    path = model.save('work/tests/discovery-linear-roundtrip.pt')
    loaded = FittedResearchModel.load(path)
    assert model.digest == loaded.digest
    for key in prediction:
        np.testing.assert_array_equal(prediction[key], loaded.predict(dev)[key])


@pytest.mark.parametrize('combination', ['concat', 'gated'])
def test_learned_modules_actual_training_and_checkpoint(combination):
    train, dev = dataset(), dataset(32, 2)
    architecture = RepresentationSpec(tuple(ModuleSpec(kind, kind, 3, 5, 4)
                                             for kind in ('summary', 'temporal', 'mlp')),
                                      combination)
    model = fit(train, dev, architecture, settings(epochs=4))
    assert model.encode(dev).shape == (32, 9)
    prediction = model.predict(dev)
    assert np.isfinite(prediction['ret']).all()
    np.testing.assert_allclose(prediction['event'].sum(1), 1, atol=1e-10)
    diagnostics = model.diagnostics
    assert diagnostics['optimizer_steps'] > 0
    assert diagnostics['history'][0]['preclip_gradient_norm'] > 0
    assert diagnostics['history'][0]['actual_update_norm'] > 0
    assert diagnostics['best_epoch'] >= 0
    assert diagnostics['best_epoch'] == int(np.argmin([
        row['development_loss'] for row in diagnostics['history']]))
    assert diagnostics['global_optimum_claimed'] is False
    assert diagnostics['stop_reason'] in {'development_patience', 'epoch_budget'}
    path = model.save(f'work/tests/discovery-{combination}-roundtrip.pt')
    loaded = FittedResearchModel.load(path)
    np.testing.assert_array_equal(loaded.encode(dev), model.encode(dev))
    np.testing.assert_array_equal(loaded.predict(dev)['ret'], prediction['ret'])


def test_event_extremes_full_survival_nll_and_gradient():
    logits = torch.tensor([[1000., -1000., -1000., 1000.]] * 5,
                          dtype=torch.float64, requires_grad=True)
    labels = torch.arange(5)
    loss = event_nll(logits, labels)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert torch.isfinite(logits.grad).all()
    np.testing.assert_allclose(event_log_probabilities(logits.detach()).exp().sum(1), 1)
    zeros = np.zeros((5, 4))
    probabilities = np.exp(event_log_probabilities(zeros))
    np.testing.assert_allclose(probabilities[0], [.25, .25, .25, .125, .125])
    np.testing.assert_allclose(event_nll(zeros, np.arange(5)), -np.log(probabilities[0]))
    assert event_log_probabilities(np.zeros((0, 4))).shape == (0, 5)


def test_full_event_nll_analytic_gradient_matches_autograd():
    rng = np.random.default_rng(8)
    design = rng.normal(size=(9, 4))
    labels = np.array([0, 1, 2, 3, 4, 0, 0, 4, 2])
    beta = rng.normal(size=(4, 4))
    loss, gradient = _event_objective(beta.ravel(), design, labels, 2, .03)
    coefficients = torch.tensor(beta, dtype=torch.float64, requires_grad=True)
    objective = event_nll(torch.tensor(design) @ coefficients, torch.tensor(labels)).mean()
    objective = objective + .03 * coefficients.square().sum()
    objective.backward()
    np.testing.assert_allclose(loss, objective.detach().numpy(), rtol=1e-12)
    np.testing.assert_allclose(gradient.reshape(beta.shape), coefficients.grad.numpy(), atol=1e-12)


def test_probe_learns_frozen_representation_and_restores():
    train, dev = dataset(), dataset(32, 2)
    z, dz = train.x[:, -1].copy(), dev.x[:, -1].copy()
    model = fit_probe(z, train, dz, dev, settings())
    prediction = model.predict_encoded(dz, dev)
    assert np.mean((prediction['ret'] - dev.targets['ret']) ** 2) < .001
    path = model.save('work/tests/discovery-probe-roundtrip.pt')
    loaded = FittedProbe.load(path)
    assert isinstance(loaded, FittedProbe)
    np.testing.assert_array_equal(loaded.predict_encoded(dz, dev)['ret'], prediction['ret'])
    with pytest.raises(ValueError, match='shape'):
        loaded.predict_encoded(dz[:, :1], dev)


def test_training_only_scales_and_cutoffs_and_model_integrity():
    train, dev = dataset(), dataset(32, 2, shift=20)
    train.observed['state'][:] = False
    train.maturity['state'][:] = np.datetime64('2099-01-01')
    model = fit(train, dev, None, settings(epochs=2))
    np.testing.assert_allclose(model.metadata['input_mean'], summary_features(train).mean(0))
    assert model.metadata['fit_feature_cutoff'].startswith('2024-01-03T15:00:00')
    assert model.metadata['fit_label_cutoff'].startswith('2024-01-03T17:00:00')
    assert 'state' not in model.metadata['fit_label_cutoffs_by_task']
    path = model.save('work/tests/discovery-integrity.pt')
    payload = torch.load(path, weights_only=True)
    payload['state']['ret'][0] += 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match='hash mismatch'):
        FittedResearchModel.load(path)


def test_development_patience_and_parameter_budget_on_learned_fit():
    train, dev = dataset(), dataset(32, 2)
    architecture = RepresentationSpec((ModuleSpec('small', 'temporal', 2, 4, 3),))
    with pytest.raises(ValueError, match='Parameter budget'):
        fit(train, dev, architecture, settings(max_parameters=1))
    model = fit(train, dev, architecture, settings(min_delta=1e10, patience=1, epochs=5))
    assert model.diagnostics['stop_reason'] == 'development_patience'
    assert len(model.diagnostics['history']) == 2


def test_missing_heads_masked_nan_and_nonfinite_rejection():
    train, dev = dataset(), dataset(32, 2)
    train.observed['state'][:] = False
    train.targets['state'][:] = np.nan
    train.observed['ret'][::2] = False
    train.targets['ret'][::2] = np.nan
    model = fit(train, dev, None, settings())
    assert 'state' not in model.predict(dev)
    assert model.diagnostics['unsupported_tasks'] == ['state']
    assert model.diagnostics['train_support']['ret'] == 32
    bad = dataset()
    bad.x[0, 0, 0] = np.nan
    with pytest.raises(FloatingPointError, match='Nonfinite'):
        fit(bad, dev, None, settings())
    dev.targets['ret'][0] = np.inf
    with pytest.raises(FloatingPointError, match='Nonfinite'):
        model.predict(dev)


def test_schema_and_budget_failures():
    train, dev = dataset(), dataset(32, 2)
    with pytest.raises(ValueError, match='Parameter budget'):
        fit(train, dev, None, settings(max_parameters=1))
    model = fit(train, dev, None, settings())
    mismatch = replace(dev, features=('b', 'a'))
    with pytest.raises(ValueError, match='schema'):
        model.predict(mismatch)


@pytest.mark.parametrize('mode', ['linear', 'learned', 'probe'])
def test_empty_prediction_encoding_and_confirmation_scoring_keep_schema(mode):
    train, dev = dataset(), dataset(32, 2)
    if mode == 'probe':
        model = fit_probe(train.x[:, -1], train, dev.x[:, -1], dev, settings(epochs=2))
        dimension = 2
    elif mode == 'learned':
        architecture = RepresentationSpec((ModuleSpec('small', 'temporal', 3, 4, 3),))
        model = fit(train, dev, architecture, settings(epochs=2))
        dimension = 3
    else:
        model = fit(train, dev, None, settings(epochs=2))
        dimension = 12
    empty = dev.subset(np.zeros(len(dev.meta), dtype=bool))
    assert model.encode(empty).shape == (0, dimension)
    predictions = model.predict_encoded(np.empty((0, dimension)), empty) if mode == 'probe' else model.predict(empty)
    for task in empty.tasks:
        expected = (0, task.output_size) if task.kind == 'first_event' else (0,)
        assert predictions[task.name].shape == expected
    results = evaluate(empty, predictions)
    assert all(value['status'] == 'insufficient_evidence' for value in results.values())
    wrong_schema = replace(empty, features=('b', 'a'))
    with pytest.raises(ValueError, match='schema'):
        model.predict(wrong_schema)
    with pytest.raises(ValueError, match='schema'):
        model.encode(wrong_schema)
    with pytest.raises(ValueError, match='nonempty'):
        fit(empty, dev, None, settings())


@pytest.mark.parametrize('status,message', [(2, 'ABNORMAL_TERMINATION_IN_LNSRCH'),
                                          (3, 'ROUNDING ERRORS PREVENT PROGRESS')])
def test_event_numerical_solver_failure_rejects_even_previous_finite_iterate(monkeypatch, status, message):
    train, dev = dataset(), dataset(32, 2)

    def failed_solver(_objective, initial, *, callback, **_kwargs):
        callback(initial.copy())  # A finite previous checkpoint cannot hide failed solver status.
        return SimpleNamespace(x=initial.copy(), success=False, status=status, message=message, nit=1)

    monkeypatch.setattr(learning, 'minimize', failed_solver)
    with pytest.raises(RuntimeError, match=f'status={status}'):
        fit(train, dev, None, settings())


@pytest.mark.parametrize('message,expected', [('STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT', 'iteration_budget'),
                                             ('STOP: TOTAL NO. OF F AND G EVALUATIONS EXCEEDS LIMIT', 'evaluation_budget')])
def test_event_solver_resource_limits_record_exact_reason(monkeypatch, message, expected):
    train, dev = dataset(), dataset(32, 2)
    monkeypatch.setattr(learning, 'minimize', lambda _objective, initial, **_kwargs:
                        SimpleNamespace(x=initial.copy(), success=False, status=1, message=message, nit=2))
    model = fit(train, dev, None, settings())
    record = model.diagnostics['optimizer_results']['event']
    assert record['stop_reason'] == expected
    assert record['status'] == 1 and record['message'] == message
    assert record['optimizer_success'] is False
    assert model.metadata['fit_feature_cutoff'] == train.meta.available_at.max().isoformat()
    assert model.metadata['fit_label_cutoff'] == pd.Timestamp(train.maturity['vol'].max(), tz='UTC').isoformat()
