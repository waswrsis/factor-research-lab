import json

import numpy as np
import pytest
from test_v7_rolling_registry import candles

from factorlab.discovery.causal_numpy import execute_formula
from factorlab.discovery.factor_pool import (
    FactorPipeline,
    FactorPool,
    evaluate_formula,
    fit_formula,
)
from factorlab.discovery.local_mutation import MonthlyBudget
from factorlab.discovery.model_registry import ModelRegistry
from factorlab.discovery.rolling import build_rolling_data
from factorlab.discovery.v7_run import (
    RunStore,
    V7Config,
    literature_seeds,
    replay_baseline,
    run_month,
)


def source_root(tmp_path, monkeypatch):
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    (tmp_path / 'factorlab').mkdir()
    (tmp_path / 'factorlab' / 'v7.py').write_text('version=7', encoding='utf-8')
    (tmp_path / 'pyproject.toml').write_text('[project]', encoding='utf-8')
    (tmp_path / 'uv.lock').write_text('version=1', encoding='utf-8')


def test_f_s_marginal_attribution_pool_restore_and_duplicate():
    data = build_rolling_data(candles('2024-08-01', '2025-04-02'))
    program = literature_seeds()[0]
    budget = MonthlyBudget('2025-04')
    result = evaluate_formula(data, '2025-04-01T00:00:00Z', program, budget=budget)
    assert len(result['folds']) == 3
    assert all(f['coverage'] == 1 for f in result['folds'])
    assert budget.counts['solve'] == 9
    assert result['evidence_status'] == 'exposed_development_only'
    pool = FactorPool()
    assert pool.add(program, result, family='family')
    assert not pool.add(program, result, family='family')
    restored = FactorPool.from_dict(pool.to_dict())
    assert restored.digest == pool.digest
    assert restored.shortlist()[0][1]['program'] == program


def test_profile_bounds_training_only_and_bundle_registry(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    data = build_rolling_data(candles('2024-08-01', '2025-04-02'))
    program = literature_seeds()[0]
    program['parameters'] = {'alpha': {'initial': .3, 'lower': .05, 'upper': .95}}
    program['outputs'][0] = {'op': 'last', 'arg': {'op': 'ema',
                              'arg': {'op': 'feature', 'name': 'rv_6'},
                              'alpha': {'op': 'param', 'name': 'alpha'}}}
    budget = MonthlyBudget('2025-04')
    bundle, values, support, profile = fit_formula(data, '2025-04-01T00:00:00Z', program,
                                                  channel='S', profile=True, max_profile_calls=5, budget=budget)
    assert .05 <= bundle.parameters['alpha'] <= .95
    assert profile['calls'] <= 5
    assert budget.counts['solve'] == profile['calls'] + 1
    registry = ModelRegistry(tmp_path / 'models')
    identity = registry.save(bundle, {'pipeline': bundle.pipeline_id})
    loaded, _ = registry.load(identity)
    assert isinstance(loaded, FactorPipeline)
    np.testing.assert_array_equal(bundle.predict(data.frame.take(support), values[support]),
                                  loaded.predict(data.frame.take(support), values[support]))
    assert np.isfinite(execute_formula(program, data.frame, parameters=bundle.parameters)[support]).all()


def test_month_controller_resume_no_duplicate_and_frozen_source(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    bars = candles('2024-08-01', '2025-04-02')
    config = V7Config(formulas_per_family=1)
    summary = run_month(tmp_path / 'run', bars, '2025-04-01T00:00:00Z', config=config)
    assert summary['budget']['counts']['family'] == 3
    assert summary['budget']['counts']['f'] == 3
    assert len(summary['shortlist']) <= 3
    assert not summary['formal_confirmation_enabled']
    repeated = run_month(tmp_path / 'run', bars, '2025-04-01T00:00:00Z', config=config)
    assert summary == repeated
    changed = bars.copy()
    changed.loc[changed.open_time < '2025-01-01T00:00:00Z', ['open', 'high', 'low', 'close']] *= 2
    with pytest.raises(ValueError, match='dataset'):
        run_month(tmp_path / 'run', changed, '2025-04-01T00:00:00Z', config=config)
    (tmp_path / 'factorlab' / 'v7.py').write_text('version=8', encoding='utf-8')
    with pytest.raises(ValueError, match='source/config'):
        run_month(tmp_path / 'run', bars, '2025-04-01T00:00:00Z', config=config)


def test_three_month_baseline_replay_restart(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    bars = candles('2024-08-01', '2025-05-02')
    cutoffs = ['2025-02-01T00:00:00Z', '2025-03-01T00:00:00Z', '2025-04-01T00:00:00Z']
    first = replay_baseline(tmp_path / 'replay', bars, cutoffs)
    second = replay_baseline(tmp_path / 'replay', bars, cutoffs)
    assert len({row['model'] for row in first}) == 3
    assert all(row['new_predictions'] > 600 for row in first)
    assert all(row['new_predictions'] == 0 for row in second)
    assert [row['model'] for row in first] == [row['model'] for row in second]


def test_state_event_chain_and_manifest_integrity(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    from factorlab.discovery.rolling import RollingSpec
    store = RunStore(tmp_path / 'run', V7Config(), RollingSpec())
    store.save('2025-04', {'counter': 1}, 'test')
    assert store.verify()['months'] == 1
    store.db.execute('UPDATE months SET value=?', (json.dumps({'counter': 2}),))
    store.db.commit()
    with pytest.raises(ValueError, match='integrity'):
        store.verify()
    store.close()
