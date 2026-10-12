"""Independent synthetic failure injections for the V7 development controller.

No market evidence or provider calls. Assertions describe the required contract,
including recovery across writes that succeed immediately before a process dies.
"""
import json
import sqlite3

import numpy as np
import pytest
from test_v7_rolling_registry import candles
from test_v7_run_factor_pool import source_root

from factorlab.discovery import v7_run
from factorlab.discovery.factor_pool import fit_formula
from factorlab.discovery.local_mutation import MonthlyBudget
from factorlab.discovery.rolling import RollingData, RollingSpec, build_rolling_data
from factorlab.protocol import fingerprint


class InjectedCrash(BaseException):
    pass


@pytest.fixture
def tiny_run(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    bars = candles('2024-08-01', '2025-04-02')
    data = build_rolling_data(bars)
    monkeypatch.setattr(v7_run, 'build_rolling_data', lambda *_: data)
    monkeypatch.setattr(v7_run, '_baseline_readout', lambda *_: {'loss': {'K0': 1.}})

    def score(data, cutoff, program, *, lookback, **kwargs):
        return {'role': 'S', 'reference': 'K0', 'aggregation': 'equal_fold_mean_loss',
                'formula_id': fingerprint(v7_run.formula_identity(program, lookback)),
                'folds': [], 'channels': {c: {'gain': .01, 'loss': .99, 'reference_loss': 1.}
                                        for c in ('F', 'S')},
                'evidence_status': 'exposed_development_only'}

    monkeypatch.setattr(v7_run, 'evaluate_formula', score)
    return tmp_path / 'run', bars, v7_run.V7Config(formulas_per_family=1), data


def _run(tiny_run):
    directory, bars, config, _ = tiny_run
    return v7_run.run_month(directory, bars, '2025-04-01T00:00:00Z', config=config)


def _state(directory):
    with sqlite3.connect(directory / 'state.sqlite') as db:
        return json.loads(db.execute('SELECT value FROM months WHERE month=?', ('2025-04',)).fetchone()[0])


def test_missing_checkpoint_cannot_reset_month_budget(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    store = v7_run.RunStore(tmp_path / 'run', v7_run.V7Config(), RollingSpec())
    try:
        store.save('2025-04', {'budget': {'solve': 100}}, 'budget_reserved')
        store.db.execute('DELETE FROM months WHERE month=?', ('2025-04',))
        store.db.commit()
        with pytest.raises(ValueError, match='integrity|checkpoint|month'):
            store.verify()
    finally:
        store.close()


@pytest.mark.parametrize('target', ['factorlab/v7.py', 'manifest.json'])
def test_saved_source_corruption_rejected(tmp_path, monkeypatch, target):
    source_root(tmp_path, monkeypatch)
    directory = tmp_path / 'run'
    v7_run.RunStore(directory, v7_run.V7Config(), RollingSpec()).close()
    (directory / 'source' / target).write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='source.*integrity'):
        v7_run.RunStore(directory, v7_run.V7Config(), RollingSpec())


def test_family_reservation_crash_can_resume_without_reset(tiny_run, monkeypatch):
    original = v7_run.RunStore.save
    fired = False

    def crash_after_reservation(self, month, state, event):
        nonlocal fired
        original(self, month, state, event)
        if not fired and event == 'budget_reserved' and state['budget']['counts']['family'] == 1:
            fired = True
            raise InjectedCrash

    monkeypatch.setattr(v7_run.RunStore, 'save', crash_after_reservation)
    with pytest.raises(InjectedCrash):
        _run(tiny_run)
    assert _state(tiny_run[0])['budget']['counts']['family'] == 1
    monkeypatch.setattr(v7_run.RunStore, 'save', original)
    result = _run(tiny_run)
    assert result['budget']['counts']['family'] <= 3
    assert not result['formal_confirmation_enabled']


@pytest.mark.parametrize('position', ['before', 'after'])
def test_publication_crash_recovers_exact_published_summary(tiny_run, monkeypatch, position):
    original = v7_run.immutable_json
    published = None

    def crash_after_publication(path, value):
        nonlocal published
        if path.parent.name == 'months' and position == 'before':
            published = value
            raise InjectedCrash
        result = original(path, value)
        if path.parent.name == 'months':
            published = value
            raise InjectedCrash
        return result

    monkeypatch.setattr(v7_run, 'immutable_json', crash_after_publication)
    with pytest.raises(InjectedCrash):
        _run(tiny_run)
    monkeypatch.setattr(v7_run, 'immutable_json', original)
    assert _run(tiny_run) == published


def test_f_reservation_is_durable_before_candidate_and_survives_crash(tiny_run, monkeypatch):
    original = v7_run.evaluate_formula

    def crash_during_candidate(*args, **kwargs):
        state = _state(tiny_run[0])
        assert state['budget']['counts']['f'] == 1
        assert state['active_candidate'] is not None
        raise InjectedCrash

    monkeypatch.setattr(v7_run, 'evaluate_formula', crash_during_candidate)
    with pytest.raises(InjectedCrash):
        _run(tiny_run)
    monkeypatch.setattr(v7_run, 'evaluate_formula', original)
    result = _run(tiny_run)
    assert result['budget']['counts']['f'] == 3
    assert sum(a['status'] == 'interrupted_consumed' for a in result['attempts']) == 1
    assert sum(a['status'] == 'scored' for a in result['attempts']) == 2


@pytest.mark.parametrize('target', ['model', 'result', 'summary'])
def test_completed_resume_checks_referenced_artifacts(tiny_run, target):
    summary = _run(tiny_run)
    directory = tiny_run[0]
    if target == 'model':
        path = directory / 'models' / summary['baseline_model'] / 'model.json'
    elif target == 'result':
        path = next((directory / 'results').glob('*.json'))
    else:
        path = directory / 'months' / '2025-04.json'
    path.write_text('{}', encoding='utf-8')
    with pytest.raises((ValueError, KeyError), match='.*'):
        _run(tiny_run)


def test_completed_resume_rejects_missing_model(tiny_run):
    summary = _run(tiny_run)
    path = tiny_run[0] / 'models' / summary['baseline_model'] / 'model.json'
    path.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        _run(tiny_run)


def test_dataset_identity_preserves_provenance_even_with_same_numbers(tiny_run):
    _run(tiny_run)
    tiny_run[1].attrs['feed_identity'] = 'different-archive'
    with pytest.raises(ValueError, match='Actual receipt S requires the shared authority and fixed work schedule'):
        _run(tiny_run)


def test_month_data_identity_ignores_unseen_future_but_rejects_seen_change(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    monkeypatch.setattr(v7_run, '_baseline_readout', lambda *_: {'loss': {'K0': 1.}})
    # Fail before search, leaving a durable as-of data identity without fitting.
    def stop(*args, **kwargs):
        raise InjectedCrash
    monkeypatch.setattr(v7_run, 'fit_ridge', stop)
    bars = candles('2024-08-01', '2025-04-03')
    directory = tmp_path / 'run'
    with pytest.raises(InjectedCrash):
        v7_run.run_month(directory, bars, '2025-04-01T00:00:00Z')
    future = bars.copy()
    future.loc[future.open_time >= '2025-04-01T00:00:00Z', ['open', 'high', 'low', 'close']] *= 2
    with pytest.raises(InjectedCrash):
        v7_run.run_month(directory, future, '2025-04-01T00:00:00Z')
    past = bars.copy()
    past.loc[past.open_time < '2025-01-01T00:00:00Z', ['open', 'high', 'low', 'close']] *= 2
    with pytest.raises(ValueError, match='dataset'):
        v7_run.run_month(directory, past, '2025-04-01T00:00:00Z')
    assert _state(directory)['budget']['counts']['solve'] == 2


def test_profile_fit_does_not_read_future_labels():
    data = build_rolling_data(candles('2024-08-01', '2025-04-02'))
    program = v7_run.literature_seeds()[0]
    program['parameters'] = {'alpha': {'initial': .3, 'lower': .05, 'upper': .95}}
    program['outputs'][0] = {'op': 'last', 'arg': {'op': 'ema',
        'arg': {'op': 'feature', 'name': 'rv_6'}, 'alpha': {'op': 'param', 'name': 'alpha'}}}
    cutoff = '2025-03-01T00:00:00Z'
    altered = data.target.copy()
    altered[~data.train_mask(cutoff) & data.observed] *= 50
    changed = RollingData(data.frame, altered, data.maturity, data.observed, data.spec, data.availability_mode)
    budgets = [MonthlyBudget('2025-03') for _ in range(2)]
    left = fit_formula(data, cutoff, program, channel='S', profile=True, max_profile_calls=5, budget=budgets[0])
    right = fit_formula(changed, cutoff, program, channel='S', profile=True, max_profile_calls=5, budget=budgets[1])
    assert left[0].digest == right[0].digest
    assert left[3] == right[3]
    np.testing.assert_array_equal(left[1], right[1])
    assert budgets[0].counts['solve'] == left[3]['calls'] + 1
    assert budgets[0].to_dict() == budgets[1].to_dict()
