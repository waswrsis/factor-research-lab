from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from factorlab.discovery.benchmarks import fit_simple_benchmarks, macro_weights
from factorlab.discovery.contracts import TaskDataset, TaskSpec
from factorlab.discovery.replay import _prediction_inputs
from factorlab.discovery.scoring import VOLATILITY_FLOOR


def fixture():
    times = pd.date_range('2025-01-01', periods=12, freq='h', tz='UTC')
    tasks = (TaskSpec(), TaskSpec('event', 'first_event'))
    x = np.full((12, 24, 1), .01, dtype=np.float32)
    data = TaskDataset(x, {'volatility': np.geomspace(.01, .1, 12), 'event': np.arange(12) % 5},
        {t.name: np.ones(12, dtype=bool) for t in tasks},
        {t.name: (times + pd.Timedelta(hours=12)).to_numpy(dtype='datetime64[ns]') for t in tasks},
        pd.DataFrame({'sample_id': list(map(str, range(12))), 'symbol': 'X', 'available_at': times}),
        ('return_1h',), tasks)
    return data, times[-1] + pd.Timedelta(hours=13)


def test_constant_uses_scored_space_and_causal_forecasts():
    data, cutoff = fixture()
    model = fit_simple_benchmarks(data, cutoff)['training_constant']
    expected = np.exp(np.log(data.targets['volatility'] + VOLATILITY_FLOOR).mean()) - VOLATILITY_FLOOR
    assert model.constants['volatility'] == pytest.approx(expected)
    forecast = model.predict(_prediction_inputs(data))
    changed = deepcopy(data)
    changed.targets['volatility'][:] = 999
    changed.targets['event'][:] = 0
    for name, value in forecast.items():
        np.testing.assert_array_equal(model.predict(changed)[name], value)
    np.testing.assert_allclose(forecast['event'].sum(axis=1), 1)
    assert (forecast['event'] > 0).all()


def test_calibration_is_train_only_log_offset():
    data, cutoff = fixture()
    data.targets['volatility'][:] = .07
    models = fit_simple_benchmarks(data, cutoff)
    np.testing.assert_allclose(models['calibrated_rolling_24h'].predict(_prediction_inputs(data))['volatility'], .07)
    assert models['rolling_24h'].predict(data)['volatility'][0] == pytest.approx(np.log1p(.01) * np.sqrt(12))


def test_maturity_and_rolling_schema_are_required():
    data, cutoff = fixture()
    with pytest.raises(ValueError, match='mature'):
        fit_simple_benchmarks(data, cutoff - pd.Timedelta(hours=10))
    data.x = data.x[:, :2]
    with pytest.raises(ValueError, match='24 causal'):
        fit_simple_benchmarks(data, cutoff)
    assert set(fit_simple_benchmarks(data, cutoff, include_rolling=False)) == {'training_constant'}


def test_macro_weights_do_not_overweight_duplicate_symbol_rows():
    data, _ = fixture()
    data.meta.loc[0:2, 'available_at'] = data.meta.available_at.iloc[0]
    data.meta.loc[0:1, 'symbol'] = 'A'
    data.meta.loc[2, 'symbol'] = 'B'
    weights = macro_weights(data, np.ones(12, dtype=bool))
    assert weights[0] + weights[1] == pytest.approx(weights[2])
    assert weights[:3].sum() == pytest.approx(weights[3])
    assert weights.sum() == pytest.approx(1)


def test_negative_volatility_is_rejected_before_fitting():
    data, cutoff = fixture()
    data.targets['volatility'][0] = -.1
    with pytest.raises(ValueError, match='nonnegative'):
        fit_simple_benchmarks(data, cutoff)


def test_event_smoothing_is_invariant_to_duplicate_observations():
    data, cutoff = fixture()
    indices = np.repeat(np.arange(len(data.meta)), 2)
    meta = data.meta.iloc[indices].reset_index(drop=True)
    meta['sample_id'] = list(map(str, range(len(meta))))
    repeated = TaskDataset(data.x[indices], {k: v[indices] for k, v in data.targets.items()},
        {k: v[indices] for k, v in data.observed.items()},
        {k: v[indices] for k, v in data.maturity.items()}, meta, data.features, data.tasks)
    first = fit_simple_benchmarks(data, cutoff)['training_constant']
    second = fit_simple_benchmarks(repeated, cutoff)['training_constant']
    np.testing.assert_allclose(first.constants['event'], second.constants['event'])
