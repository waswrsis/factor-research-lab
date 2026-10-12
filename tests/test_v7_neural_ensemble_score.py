"""Full-learner arithmetic and support tests on artificial cold models."""
from dataclasses import replace

import numpy as np
import pytest
from test_v7_neural_forward import fitted as fitted  # noqa: PLC0414

from factorlab.discovery.convex_eval import log_mse
from factorlab.discovery.neural_ensemble_score import score_complete_neural_fold
from factorlab.discovery.neural_forward import NeuralForwardModel
from factorlab.protocol import fingerprint


def score(data, model, symbols=('X',)):
    return score_complete_neural_fold(data, model, start='2025-04-01T00:00Z',
                                      end='2025-04-03T00:00Z', as_of='2025-04-04T00:00Z', symbols=symbols)


def test_complete_model_own_predictions_define_its_s_score(fitted):
    data, _, model, _ = fitted
    result = score(data, model)
    expected = log_mse(result['target'], result['predictions'])
    assert result['loss'] == expected
    assert result['support'] == 4 and result['nominal_rows'] == 48
    assert result['coverage'] == 4 / 48
    assert result['pipeline_id'] == model.recipe.pipeline_id
    assert not result['formal_claim'] and not result['automatic_adoption']
    value = dict(result); value.pop('score_id')
    assert result['score_id'] == fingerprint(value)


def test_save_reload_produces_exact_score_and_does_not_train(fitted, tmp_path):
    data, _, model, budget = fitted
    before = dict(budget.counts)
    path = tmp_path / 'model.json'
    model.save(path)
    assert score(data, NeuralForwardModel.load(path)) == score(data, model)
    assert budget.counts == before


def test_missing_symbols_and_labels_do_not_shrink_nominal_denominator(fitted):
    data, _, model, _ = fitted
    result = score(data, model, ('X', 'Y'))
    assert result['nominal_rows'] == 96 and result['coverage'] == 4 / 96
    observed = data.observed.copy()
    observed[data.frame.metadata.available_at >= '2025-04-01T00:00Z'] = False
    missing = replace(data, observed=observed)
    result = score(missing, model)
    assert result['support'] == 0 and result['coverage'] == 0
    assert result['loss'] is None and result['gain'] is None and result['hourly'] == []


def test_changed_training_source_is_rejected_without_refitting(fitted):
    data, _, model, _ = fitted
    target = data.target.copy()
    target[data.train_mask('2025-04-01T00:00Z')] *= 2
    with pytest.raises(ValueError, match='training source'):
        score(replace(data, target=target), model)


def test_unused_future_or_old_labels_cannot_change_complete_fold_score(fitted):
    data, _, model, _ = fitted
    target = data.target.copy()
    unused = ((data.frame.metadata.available_at < '2025-01-01T00:00Z')
              | (data.frame.metadata.available_at >= '2025-04-03T00:00Z')).to_numpy()
    target[unused] *= 100
    assert score(replace(data, target=target), model) == score(data, model)


def test_old_month_fit_cannot_impersonate_current_cold_model(fitted):
    data, _, model, _ = fitted
    with pytest.raises(ValueError, match='cold cutoff'):
        score_complete_neural_fold(data, model, start='2025-04-02T00:00Z',
            end='2025-04-03T00:00Z', as_of='2025-04-04T00:00Z', symbols=('X',))


def test_log_ensemble_loss_is_not_average_seed_loss():
    # +a and -a log errors cancel in the ensemble. Their separate MSEs do not.
    target = np.full(3, 1.)
    seeds = np.exp(np.array([[-.4, -.4, -.4], [.4, .4, .4]])) - 1e-8
    ensemble = np.exp(np.log(seeds + 1e-8).mean(axis=0)) - 1e-8
    actual = log_mse(target, ensemble)
    per_seed_mean = np.mean([log_mse(target, seed) for seed in seeds])
    assert actual < 1e-15 and per_seed_mean > .15


@pytest.mark.parametrize('symbols', [('X', 'X'), (), 'X', ('X', 1)])
def test_universe_must_be_explicit_unique_symbols(fitted, symbols):
    data, _, model, _ = fitted
    with pytest.raises(ValueError, match='universe'):
        score(data, model, symbols)
