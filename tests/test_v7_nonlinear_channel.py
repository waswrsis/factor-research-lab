"""Bounded synthetic nonlinear research checks, never market-efficacy evidence."""
import copy
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factorlab.discovery.known_library import FEATURE_NAMES, FactorFrame, library_schema
from factorlab.discovery.nonlinear_channel import (
    GATE_FIELDS,
    ChannelBudget,
    NeuralGate,
    ProbeConfig,
    ResidualConfig,
    StructureSpec,
    research_residual,
    run_probe,
    structure_suggestions,
)
from factorlab.discovery.rolling import RollingData, RollingSpec
from factorlab.protocol import fingerprint


def artificial_data():
    times = pd.date_range('2024-09-01', '2025-04-03', freq='12h', tz='UTC')
    rng = np.random.default_rng(77)
    values = rng.normal(size=(len(times), len(FEATURE_NAMES)))
    values[:, :8] = .1 + np.abs(values[:, :8])
    meta = pd.DataFrame({'symbol': 'X', 'available_at': times,
                         'feature_open_time': times - pd.Timedelta(hours=1),
                         'forecast_window_start': times, 'sample_id': [f's{i}' for i in range(len(times))]})
    frame = FactorFrame(values, meta, schema_hash=fingerprint(library_schema()))
    target = np.exp(.2 + .3 * np.tanh(values[:, 8] * values[:, 9]))
    return RollingData(frame, target, times + pd.Timedelta(hours=12), np.ones(len(times), bool),
                        RollingSpec(minimum_train_rows=12), 'assumed_historical_delay')


def gate():
    return NeuralGate(**{name: True for name in GATE_FIELDS},
                      evidence_ids={name: f'synthetic-{name}' for name in GATE_FIELDS})


def test_disabled_or_closed_gate_does_not_load_torch():
    script = (
        'import sys; from types import SimpleNamespace; '
        'from factorlab.discovery.rolling import RollingSpec; '
        'from factorlab.discovery.nonlinear_channel import research_residual,run_probe; '
        'assert research_residual(None,None)["status"]=="disabled"; '
        'data=SimpleNamespace(spec=RollingSpec()); '
        'assert run_probe(data,"2025-05-01T00:00:00Z")["status"]=="not_due"; '
        'assert "torch" not in sys.modules'
    )
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('cutoff',['2025-05-01T00:00:00Z','2025-05-16T00:00:00Z'])
def test_elapsed_ninety_day_closed_channels_do_not_load_torch(cutoff):
    script=(
        'import sys; from types import SimpleNamespace; '
        'from factorlab.discovery.rolling import RollingSpec; '
        'from factorlab.discovery.nonlinear_channel import research_residual,run_probe; '
        'assert research_residual(None,None)["status"]=="disabled"; '
        'data=SimpleNamespace(spec=RollingSpec(window_days=90)); '
        f'assert run_probe(data,{cutoff!r})["status"]=="not_due"; '
        'assert "torch" not in sys.modules'
    )
    result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True,check=False)
    assert result.returncode==0,result.stderr


def test_tree_fixed_quarterly_probe_same_support_and_future_target_independence():
    data = artificial_data()
    result = run_probe(data, '2025-04-01T00:00:00Z')
    changed = data.target.copy()
    changed[data.frame.metadata.available_at >= pd.Timestamp('2025-04-01T00:00:00Z')] *= 100
    future = replace(data, target=changed)
    again = run_probe(future, '2025-04-01T00:00:00Z')
    assert result['status'] == 'completed' and not result['formal_claim']
    assert result['budget']['counts']['tree_fit'] == 3
    assert result['budget']['counts']['solve'] == 3
    for first, second in zip(result['folds'], again['folds'], strict=True):
        assert first['sample_ids'] == second['sample_ids']
        assert first['model_id'] == second['model_id']
        assert first['fit']['fit'] == 'cold'
        assert first['reference_model_id'] == second['reference_model_id']
        assert first['predictions'] == second['predictions']
    with pytest.raises(ValueError, match='frozen'):
        ProbeConfig(max_depth=4)


def test_scheduled_tree_really_runs_without_loading_torch():
    source = str(Path(__file__).resolve())
    script = (f'import runpy,sys; ns=runpy.run_path({source!r}); '
              'result=ns["run_probe"](ns["artificial_data"](),"2025-04-01T00:00:00Z"); '
              'assert result["status"]=="completed"; assert "torch" not in sys.modules')
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_missing_feature_support_is_common_to_tree_and_reference():
    data = artificial_data()
    values = data.frame.values.copy()
    missing = np.flatnonzero(data.frame.metadata.available_at >= pd.Timestamp('2025-01-01T00:00:00Z'))[5]
    values[missing, -1] = np.nan
    frame = FactorFrame(values, data.frame.metadata, schema_hash=data.frame.schema_hash)
    modified = replace(data, frame=frame)
    result = run_probe(modified, '2025-04-01T00:00:00Z')
    excluded = frame.metadata.sample_id.iloc[missing]
    assert all(excluded not in row['sample_ids'] for row in result['folds'])
    for row in result['folds']:
        expected = modified.score_mask(row['fit_cutoff'], row['score_end'], row['score_end'])
        assert row['support'] == int(expected.sum())


@pytest.mark.parametrize('missing', GATE_FIELDS)
def test_each_gate_is_independently_required(missing):
    values = {name: name != missing for name in GATE_FIELDS}
    blocked = NeuralGate(**values, evidence_ids={name: name for name in GATE_FIELDS})
    with pytest.raises(ValueError, match='every gate'):
        research_residual(artificial_data(), '2025-04-01T00:00:00Z', gate=blocked,
                           config=ResidualConfig(enabled=True))


def test_epoch_zero_exact_reference_common_seeds_and_training_order():
    data = artificial_data()
    config = ResidualConfig(enabled=True, structures=(StructureSpec(4, 2, 1), StructureSpec(8, 2, 1)),
                            seeds=(17, 29), epochs=0)
    result = research_residual(data, '2025-04-01T00:00:00Z', gate=gate(), config=config)
    assert result['status'] == 'completed'
    assert len(result['records']) == 12
    assert all(r['selected_epoch'] == 0 and r['loss'] == r['epoch0_loss']
               and r['data_order_hashes'] == [] for r in result['records'])
    assert all(r['convex_readouts']['trained_encoder']['loss']
               == r['convex_readouts']['initial_encoder_dimension_control']['loss']
               for r in result['records'])
    assert result['budget']['counts']['neural_fit'] == 12
    assert result['budget']['counts']['solve'] == 27
    assert not result['formal_claim']


def test_one_epoch_uses_common_seed_orders_across_structures_and_finite_outputs():
    config = ResidualConfig(enabled=True, structures=(StructureSpec(4, 2, 1), StructureSpec(8, 2, 1)),
                            seeds=(17, 29), epochs=1)
    result = research_residual(artificial_data(), '2025-04-01T00:00:00Z', gate=gate(), config=config)
    assert result['status'] == 'completed'
    grouped = {}
    for row in result['records']:
        key = (row['fit_cutoff'], row['seed'])
        if key in grouped:
            assert row['data_order_hashes'] == grouped[key]
        grouped[key] = row['data_order_hashes']
        assert np.isfinite(row['predictions']).all()
    assert result['budget']['counts']['epoch'] == 12
    data = artificial_data()
    future_target = data.target.copy()
    future_target[data.frame.metadata.available_at >= pd.Timestamp('2025-04-01T00:00:00Z')] *= 100
    rerun = research_residual(replace(data, target=future_target), '2025-04-01T00:00:00Z',
                              gate=gate(), config=config)
    assert [r['model_id'] for r in result['records']] == [r['model_id'] for r in rerun['records']]
    assert [r['predictions'] for r in result['records']] == [r['predictions'] for r in rerun['records']]


def test_budget_restore_structure_bounds_and_isolated_task_rejection():
    saved = []
    budget = ChannelBudget(persist=lambda value: saved.append(copy.deepcopy(value)))
    budget.consume('neural_fit', 2)
    restored = ChannelBudget.from_dict(saved[-1])
    assert restored.counts['neural_fit'] == 2
    with pytest.raises(ValueError):
        restored.consume('neural_fit', 17)
    with pytest.raises(ValueError):
        StructureSpec(width=128)
    proposals = structure_suggestions()
    assert len({p['structure_id'] for p in proposals}) == 3
    assert all(p['status'] == 'proposed' for p in proposals)
    wrong = replace(artificial_data(), spec=RollingSpec(task='first_event', horizon_hours=20,
                                                       minimum_train_rows=12))
    with pytest.raises(ValueError, match='isolates'):
        run_probe(wrong, '2025-04-01T00:00:00Z')
