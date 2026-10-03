"""Synthetic checkpoint regressions; no inference about market performance."""
from dataclasses import asdict

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
from factorlab.discovery.learning import FittedResearchModel, fit, fit_probe
from factorlab.discovery.scoring import per_sample_loss


def data(n=8):
    tasks = (TaskSpec('vol', 'volatility', 2), TaskSpec('aux', 'return', 2),
             TaskSpec('state', 'state'))
    times = pd.date_range('2024-01-01', periods=n, freq='h', tz='UTC')
    return TaskDataset(np.ones((n, 2, 1), dtype=np.float32),
                       {'vol': np.ones(n), 'aux': np.zeros(n), 'state': np.zeros(n)},
                       {t.name: np.ones(n, dtype=bool) for t in tasks},
                       {t.name: (times + pd.Timedelta(hours=2)).to_numpy(dtype='datetime64[ns]')
                        for t in tasks},
                       pd.DataFrame({'sample_id': [str(i) for i in range(n)],
                                     'symbol': 'TEST', 'available_at': times}), ('a',), tasks)


ARCHITECTURE = RepresentationSpec((ModuleSpec('small', 'temporal', 2, 4, 2),))


@pytest.mark.parametrize('value', ['', 'not a name', 'hyphen-name', 3, True, [], {}])
def test_selection_task_contract_rejects_invalid_names(value):
    with pytest.raises(ValueError, match='selection_task'):
        TrainingSpec(selection_task=value)


def test_legacy_settings_deserialize_without_selection_task():
    previous = asdict(TrainingSpec())
    previous.pop('selection_task')
    assert TrainingSpec(**previous).selection_task is None


@pytest.mark.parametrize('problem', ['unknown', 'descriptive', 'training', 'development'])
def test_primary_checkpoint_requires_predictive_train_and_development_support(problem):
    train, dev = data(), data()
    task = 'missing' if problem == 'unknown' else 'state' if problem == 'descriptive' else 'vol'
    if problem in {'training', 'development'}:
        sample = train if problem == 'training' else dev
        sample.observed['vol'][:] = False
        sample.targets['vol'][:] = np.nan
    with pytest.raises(ValueError, match='selection_task'):
        fit(train, dev, ARCHITECTURE, TrainingSpec(selection_task=task))


@pytest.mark.parametrize('mode', ['linear', 'probe'])
def test_selection_setting_is_explicitly_rejected_for_independent_linear_heads(mode):
    train, dev = data(), data()
    settings = TrainingSpec(selection_task='vol')
    with pytest.raises(ValueError, match='no joint checkpoint'):
        if mode == 'linear':
            fit(train, dev, None, settings)
        else:
            fit_probe(train.x[:, -1], train, dev.x[:, -1], dev, settings)


def test_original_score_uses_equal_symbols_within_utc_hour_then_equal_hours():
    dev = data(6)
    # UTC hour 00: A has 3 duplicates with mean 2; B has mean 10.
    # UTC hour 01: A has mean 20. Equal-hour result = (6 + 20) / 2 = 13.
    dev.meta['available_at'] = pd.to_datetime([
        '2024-01-01T09:00:00+09:00', '2024-01-01T00:10:00Z',
        '2024-01-01T00:20:00Z', '2024-01-01T00:30:00Z',
        '2024-01-01T01:00:00Z', '2024-01-01T01:10:00Z'], utc=True)
    dev.meta['symbol'] = ['A', 'A', 'A', 'B', 'A', 'A']
    dev.observed['aux'][-1] = False
    dev.targets['aux'][-1] = np.nan
    loss = np.array([0., 2., 4., 10., 20., 1000.])
    task = dev.tasks[1]
    # Nonunit training target transform ensures selection is in original units.
    scales = {'aux': [3., 7.]}
    prediction = np.sqrt(loss)
    result = learning._selection_losses({'aux': ((prediction - 3) / 7)[:, None]},
                                        dev, [task], scales)
    assert result['aux'] == pytest.approx(13)
    assert result['aux'] != pytest.approx(loss[:5].mean())


def test_primary_volatility_selection_uses_decoded_floor_aware_benchmark_score():
    dev = data()
    raw = {'vol': np.full((len(dev.x), 1), -1000.)}
    scales = {'vol': [0., 1.]}
    task = dev.tasks[0]
    result = learning._selection_losses(raw, dev, [task], scales)
    expected = per_sample_loss(task, dev.targets['vol'], np.zeros(len(dev.x))).mean()
    assert result['vol'] == pytest.approx(expected)
    assert result['vol'] < 1000  # Decoded zero/floor error, not standardized 1e6.


def test_primary_event_selection_uses_benchmark_probability_floor():
    fixture = data()
    task = TaskSpec('event', 'first_event', 2, (1, 2))
    dev = TaskDataset(fixture.x, {'event': np.ones(len(fixture.x))},
                      {'event': np.ones(len(fixture.x), dtype=bool)},
                      {'event': fixture.maturity['vol']}, fixture.meta, fixture.features, (task,))
    raw = {'event': np.tile([-1000., -1000., 0., 0.], (len(dev.x), 1))}
    result = learning._selection_losses(raw, dev, [task], {})
    assert result['event'] == pytest.approx(-np.log(1e-12))


def test_primary_trend_controls_best_checkpoint_patience_and_restoration(monkeypatch):
    train, dev = data(), data()
    # All trainable parameters are real; one controlled optimizer step per epoch
    # supplies conflicting main/auxiliary trends without stochastic convergence.
    primary = [3., 2., 3., 4.]
    auxiliary = [100., 10., 0., 0.]
    stepped_states = []

    def forward(network, x):
        return {name: head.bias[0].expand(len(x), 1)
                for name, head in zip(network.task_names, network.heads)}

    def step(optimizer, closure=None):
        epoch = len(stepped_states)
        with torch.no_grad():
            for group in optimizer.param_groups:
                for parameter in group['params']:
                    parameter.zero_()
            # Network head order follows the declared tasks.
            params = optimizer.param_groups[0]['params']
            params[-5].fill_(np.sqrt(primary[epoch]))
            params[-3].fill_(np.sqrt(auxiliary[epoch]))
        stepped_states.append([p.detach().clone() for p in params])

    monkeypatch.setattr(learning._Network, 'forward', forward)
    monkeypatch.setattr(torch.optim.Adam, 'step', step)
    settings = TrainingSpec(epochs=4, batch_size=8, patience=1, min_delta=.01,
                            selection_task='vol', max_seconds=30)
    model = fit(train, dev, ARCHITECTURE, settings)
    history = model.diagnostics['history']
    assert model.diagnostics['stop_reason'] == 'development_patience'
    assert len(history) == 3
    assert [row['selection_loss'] for row in history] == pytest.approx(primary[:3])
    assert history[2]['development_loss'] < history[1]['development_loss']
    assert model.diagnostics['best_epoch'] == 1
    assert model.diagnostics['selection_loss'] == pytest.approx(2.)
    assert model.diagnostics['development_loss'] == history[1]['development_loss']
    assert model.diagnostics['selection_task'] == 'vol'
    assert model.diagnostics['selection_aggregation'] == learning.SELECTION_AGGREGATION
    # Restoration includes auxiliary weights from the same selected epoch.
    prediction = model.predict(dev)
    assert per_sample_loss(train.tasks[0], dev.targets['vol'], prediction['vol']).mean() == pytest.approx(2.)
    assert per_sample_loss(train.tasks[1], dev.targets['aux'], prediction['aux']).mean() == pytest.approx(10.)
    path = model.save('work/v6_improvement/tests/primary-selection.pt')
    loaded = FittedResearchModel.load(path)
    assert loaded.digest == model.digest
    assert loaded.diagnostics == model.diagnostics
    assert loaded.metadata['settings']['selection_task'] == 'vol'
    np.testing.assert_array_equal(loaded.predict(dev)['vol'], prediction['vol'])


def test_old_checkpoint_metadata_keeps_digest_and_load_compatibility():
    model = fit(data(), data(), ARCHITECTURE,
                TrainingSpec(epochs=1, batch_size=8, max_seconds=30))
    model.metadata['settings'].pop('selection_task')
    for record in [model.diagnostics, *model.diagnostics['history']]:
        for key in list(record):
            if key.startswith('selection_') or key == 'development_loss_metric':
                record.pop(key)
    previous_digest = model.digest
    path = model.save('work/v6_improvement/tests/legacy-selection.pt')
    loaded = FittedResearchModel.load(path)
    assert loaded.digest == previous_digest
    assert 'selection_task' not in loaded.metadata['settings']
    np.testing.assert_array_equal(loaded.predict(data())['vol'], model.predict(data())['vol'])
