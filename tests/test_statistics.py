import numpy as np
import pandas as pd
import pytest

from factorlab.statistics import paired_block_bootstrap, risk_metrics


def test_identical_paired_paths_have_exact_zero_interval():
    path = np.random.default_rng(2).normal(0, .01, 240)
    result = paired_block_bootstrap(path, path, n_bootstrap=100, risk_aversion=10)
    assert result["status"] == "ok"
    assert result["estimate"] == result["ci_lower"] == result["ci_upper"] == 0


def test_constant_paired_improvement_and_tail_insufficiency():
    reference = np.arange(250) / 1e5
    result = paired_block_bootstrap(reference + .001, reference, n_bootstrap=100)
    assert result["estimate"] == pytest.approx(.001)
    assert result["ci_lower"] == pytest.approx(.001)
    assert result["ci_upper"] == pytest.approx(.001)
    assert result["dropped_observations"] == 10
    assert paired_block_bootstrap(reference[:100], reference[:100])["status"] == "insufficient_sample"


def test_unaligned_gapped_and_asset_matrix_paths_rejected():
    index = pd.date_range("2025-01-01", periods=240, freq="h", tz="UTC")
    path = pd.Series(0., index=index)
    with pytest.raises(ValueError, match="identical"):
        paired_block_bootstrap(path, path.shift(1).dropna())
    with pytest.raises(ValueError, match="gaps"):
        paired_block_bootstrap(path.drop(index[10]), path.drop(index[10]))
    with pytest.raises(ValueError, match="one-dimensional"):
        paired_block_bootstrap(np.zeros((240, 3)), np.zeros((240, 3)))


def test_initial_drawdown_and_return_compounding():
    result = risk_metrics([-.1, .1], equity=[900, 990])
    assert result["max_drawdown"] == pytest.approx(.1)
    assert result["total_return"] == pytest.approx(-.01)
    assert risk_metrics([])["n_observations"] == 0
    with pytest.raises(ValueError):
        risk_metrics([np.nan])
