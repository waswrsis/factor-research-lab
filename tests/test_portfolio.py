import numpy as np
import pandas as pd
import pytest

from factorlab.portfolio import PortfolioConfig, drift_weights, optimize_portfolio


def inputs():
    symbols = ["BTC", "ETH", "SOL", "X"]
    return (pd.Series([.1, -.1, .08, .09], index=symbols),
            pd.DataFrame(np.eye(4) * .01, index=symbols, columns=symbols),
            pd.Series(0., index=symbols))


def test_constraints_liquidity_and_cash_solver_failure():
    mean, cov, prev = inputs()
    caps = pd.Series([.05, .1, .2, 0], index=mean.index)
    result = optimize_portfolio(mean, cov, prev, liquidity_caps=caps)
    assert result.status == "ok"
    assert (result.weights.abs() <= caps + 1e-8).all()
    assert result.weights.abs().sum() <= 1
    assert result.weights.iloc[0] > 0 and result.weights.iloc[1] < 0
    failed = optimize_portfolio(mean, cov, prev, config=PortfolioConfig(solver="MISSING_SOLVER"))
    assert failed.status == "cash_fallback" and (failed.weights == 0).all()


def test_horizon_scaling_and_input_nonmutation():
    mean, cov, prev = inputs()
    original = mean.copy()
    adjusted = optimize_portfolio(mean, cov, prev, config=PortfolioConfig(expected_return_horizon_hours=1,
                                                                           covariance_horizon_hours=1))
    equivalent = optimize_portfolio(mean * 12, cov * 12, prev)
    pd.testing.assert_series_equal(mean, original)
    assert adjusted.weights.to_numpy() == pytest.approx(equivalent.weights.to_numpy(), abs=1e-7)


def test_indefinite_covariance_projected_and_zero_forecast_chooses_cash():
    mean, cov, prev = inputs()
    cov.iloc[0, 1] = cov.iloc[1, 0] = .05
    result = optimize_portfolio(mean, cov, prev, config=PortfolioConfig(covariance_shrinkage=0))
    assert result.status == "ok"
    assert result.diagnostics["minimum_eigenvalue_before_projection"] < 0
    result = optimize_portfolio(mean * 0, cov, prev)
    assert result.weights.abs().max() < 1e-5


def test_drift_uses_nav_including_cash_and_optional_costs():
    weights = pd.Series([.5, -.2], index=["BTC", "ETH"])
    returns = pd.Series([.1, -.1], index=weights.index)
    assert drift_weights(weights, returns).to_numpy() == pytest.approx([.55 / 1.07, -.18 / 1.07])
    assert drift_weights(weights, returns, portfolio_return=.05).iloc[0] == pytest.approx(.55 / 1.05)
    with pytest.raises(ValueError):
        drift_weights(weights, returns, portfolio_return=-1)
