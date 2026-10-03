"""Convex portfolio allocation in one declared return/risk horizon."""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioConfig:
    risk_aversion: float = 10.
    turnover_penalty: float = .001
    gross_limit: float = 1.
    single_name_limit: float = .3
    covariance_shrinkage: float = .1
    objective_horizon_hours: float = 12.
    expected_return_horizon_hours: float = 12.
    covariance_horizon_hours: float = 12.
    solver: str = "CLARABEL"

    def __post_init__(self):
        numeric = [self.risk_aversion, self.turnover_penalty, self.gross_limit,
                   self.single_name_limit, self.covariance_shrinkage,
                   self.objective_horizon_hours, self.expected_return_horizon_hours,
                   self.covariance_horizon_hours]
        if not np.isfinite(numeric).all():
            raise ValueError("Portfolio settings must be finite")
        if (self.risk_aversion < 0 or self.turnover_penalty < 0
                or not 0 < self.single_name_limit <= self.gross_limit <= 1
                or not 0 <= self.covariance_shrinkage <= 1
                or min(numeric[-3:]) <= 0):
            raise ValueError("Invalid portfolio settings")


@dataclass(frozen=True)
class PortfolioResult:
    weights: pd.Series
    status: str
    objective: float | None
    diagnostics: dict


def drift_weights(weights: pd.Series, asset_returns: pd.Series, *,
                  portfolio_return: float | None = None) -> pd.Series:
    """Derivative exposure drift; pass portfolio_return to include fees/funding."""
    if not weights.index.equals(asset_returns.index) or weights.index.has_duplicates:
        raise ValueError("Weights and asset returns must have identical unique symbols")
    w, r = weights.to_numpy(float), asset_returns.to_numpy(float)
    if not np.isfinite(w).all() or not np.isfinite(r).all() or (r <= -1).any():
        raise ValueError("Invalid weights or returns")
    nav_return = float(w @ r) if portfolio_return is None else float(portfolio_return)
    if not np.isfinite(nav_return) or nav_return <= -1:
        raise ValueError("Portfolio NAV must remain positive")
    return pd.Series(w * (1 + r) / (1 + nav_return), index=weights.index)


def optimize_portfolio(expected_returns: pd.Series, covariance: pd.DataFrame,
                       previous_weights: pd.Series, *, config=None,
                       liquidity_caps=None) -> PortfolioResult:
    config = PortfolioConfig() if config is None else config
    symbols = expected_returns.index
    if (not len(symbols) or symbols.has_duplicates
            or not symbols.equals(previous_weights.index)
            or not symbols.equals(covariance.index) or not symbols.equals(covariance.columns)):
        raise ValueError("Mean, covariance and previous weights require identical unique symbols")
    mu = expected_returns.to_numpy(float, copy=True)
    raw = covariance.to_numpy(float)
    previous = previous_weights.to_numpy(float)
    if not np.isfinite(mu).all() or not np.isfinite(raw).all() or not np.isfinite(previous).all():
        raise ValueError("Portfolio inputs must be finite")
    if not np.allclose(raw, raw.T, atol=1e-10):
        raise ValueError("Covariance must be symmetric")
    if (np.diag(raw) < 0).any():
        raise ValueError("Covariance variances cannot be negative")
    caps = np.full(len(symbols), config.single_name_limit)
    if liquidity_caps is not None:
        if not isinstance(liquidity_caps, pd.Series) or not symbols.equals(liquidity_caps.index):
            raise ValueError("Liquidity caps require identical symbols")
        liquid = liquidity_caps.to_numpy(float)
        if not np.isfinite(liquid).all() or (liquid < 0).any():
            raise ValueError("Liquidity caps must be finite nonnegative NAV fractions")
        caps = np.minimum(caps, liquid)
    mu *= config.objective_horizon_hours / config.expected_return_horizon_hours
    raw = raw * config.objective_horizon_hours / config.covariance_horizon_hours
    shrunk = (1 - config.covariance_shrinkage) * raw + config.covariance_shrinkage * np.diag(np.diag(raw))
    eigenvalues, eigenvectors = np.linalg.eigh(shrunk)
    psd = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T
    diagnostics = {"objective_horizon_hours": config.objective_horizon_hours,
                   "minimum_eigenvalue_before_projection": float(eigenvalues.min()),
                   "covariance_shrinkage": config.covariance_shrinkage,
                   "horizon_scaling_assumption": "mean_and_covariance_scale_linearly_with_hours",
                   "cash_allowed": True}
    w = cp.Variable(len(symbols))
    objective = mu @ w - config.risk_aversion / 2 * cp.quad_form(w, cp.psd_wrap(psd))
    objective -= config.turnover_penalty * cp.norm1(w - previous)
    problem = cp.Problem(cp.Maximize(objective), [cp.norm1(w) <= config.gross_limit,
                                                 cp.abs(w) <= caps])
    try:
        problem.solve(solver=config.solver)
        values = None if w.value is None else np.asarray(w.value).reshape(-1)
        valid = (problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and values is not None
                 and np.isfinite(values).all() and np.abs(values).sum() <= config.gross_limit + 1e-6
                 and (np.abs(values) <= caps + 1e-6).all())
        if valid:
            # Remove numerical constraint overshoots conservatively.
            values = np.clip(values, -caps, caps)
            if np.abs(values).sum() > config.gross_limit:
                values *= config.gross_limit / np.abs(values).sum()
            diagnostics["solver_status"] = problem.status
            attained = (mu @ values - config.risk_aversion / 2 * values @ psd @ values
                        - config.turnover_penalty * np.abs(values - previous).sum())
            return PortfolioResult(pd.Series(values, index=symbols), "ok", float(attained), diagnostics)
        diagnostics["solver_status"] = problem.status
    except cp.error.SolverError as exc:
        diagnostics["solver_error"] = str(exc)
    return PortfolioResult(pd.Series(0., index=symbols), "cash_fallback", None, diagnostics)
