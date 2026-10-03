"""Risk summaries and paired, non-overlapping time-block uncertainty estimates."""
from __future__ import annotations

import numpy as np
import pandas as pd


def risk_metrics(returns, equity=None, periods_per_year=8760) -> dict:
    r = np.asarray(returns, dtype=float).reshape(-1)
    if not np.isfinite(r).all() or (r <= -1).any():
        raise ValueError("Returns must be finite and greater than -1")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    if not len(r):
        return {"n_observations": 0, "total_return": None, "annualized_return": None,
                "annualized_volatility": None, "sharpe": None, "max_drawdown": None}
    wealth = np.r_[1., np.cumprod(1 + r)]
    if equity is not None:
        observed = np.asarray(equity, dtype=float).reshape(-1)
        if len(observed) != len(r) or not np.isfinite(observed).all() or (observed <= 0).any():
            raise ValueError("Equity must be positive, finite and aligned with returns")
        # The initial baseline is reconstructed from the first observed return.
        wealth = np.r_[observed[0] / (1 + r[0]), observed]
    vol = float(r.std(ddof=1)) if len(r) > 1 else 0.
    annual_log = float(np.log1p(r).sum() * periods_per_year / len(r))
    return {"n_observations": len(r), "total_return": float(np.prod(1 + r) - 1),
            "annualized_return": float(np.expm1(annual_log)) if annual_log < 700 else None,
            "annualized_volatility": float(vol * np.sqrt(periods_per_year)),
            "sharpe": float(r.mean() / vol * np.sqrt(periods_per_year)) if vol > 0 else None,
            "max_drawdown": float(-(wealth / np.maximum.accumulate(wealth) - 1).min())}


def paired_block_bootstrap(candidate_returns, reference_returns, block_length=24,
                           n_bootstrap=2000, confidence=.95, seed=0, min_blocks=8,
                           risk_aversion=0) -> dict:
    """Resample the same disjoint time blocks for both policies; never asset rows.

    Statistic is mean return minus risk_aversion / 2 * population variance per
    observation. The trailing incomplete block is explicitly excluded. This CI
    describes this one paired policy comparison, with no multiple-search correction.
    """
    if block_length < 1 or int(block_length) != block_length or n_bootstrap < 1:
        raise ValueError("Block length and bootstrap count must be positive integers")
    if int(n_bootstrap) != n_bootstrap or min_blocks < 2 or int(min_blocks) != min_blocks:
        raise ValueError("Invalid bootstrap count or minimum block count")
    if not 0 < confidence < 1 or not np.isfinite(risk_aversion) or risk_aversion < 0:
        raise ValueError("Invalid confidence or risk aversion")
    if isinstance(candidate_returns, pd.Series) or isinstance(reference_returns, pd.Series):
        if not isinstance(candidate_returns, pd.Series) or not isinstance(reference_returns, pd.Series):
            raise TypeError("Both paired inputs must be Series or both arrays")
        if (not candidate_returns.index.equals(reference_returns.index)
                or candidate_returns.index.has_duplicates
                or not candidate_returns.index.is_monotonic_increasing):
            raise ValueError("Paired returns require the identical ordered, unique time index")
        if isinstance(candidate_returns.index, pd.DatetimeIndex) and len(candidate_returns) > 2:
            intervals = np.diff(candidate_returns.index.asi8)
            if not (intervals == intervals[0]).all():
                raise ValueError("Time gaps must be resolved explicitly before block resampling")
    candidate = np.asarray(candidate_returns, dtype=float)
    reference = np.asarray(reference_returns, dtype=float)
    if candidate.ndim != 1 or reference.ndim != 1 or candidate.shape != reference.shape:
        raise ValueError("Inputs must be aligned one-dimensional portfolio return paths")
    if not np.isfinite(candidate).all() or not np.isfinite(reference).all():
        raise ValueError("Paired paths cannot contain missing or infinite returns")
    blocks = len(candidate) // int(block_length)
    used = blocks * int(block_length)
    result = {"status": "insufficient_sample", "n_blocks": blocks, "block_length": block_length,
              "n_observations": len(candidate), "used_observations": used,
              "dropped_observations": len(candidate) - used, "confidence": confidence,
              "estimate": None, "ci_lower": None, "ci_upper": None,
              "statistic": "mean_minus_half_risk_aversion_times_variance_difference"}
    if blocks < min_blocks:
        return result
    c = candidate[:used].reshape(blocks, int(block_length))
    r = reference[:used].reshape(blocks, int(block_length))

    def utility(x):
        return x.mean() - risk_aversion / 2 * x.var()

    samples = np.empty(int(n_bootstrap))
    rng = np.random.default_rng(seed)
    for i in range(int(n_bootstrap)):
        selected = rng.integers(0, blocks, size=blocks)
        samples[i] = utility(c[selected]) - utility(r[selected])
    alpha = (1 - confidence) / 2
    result.update(status="ok", estimate=float(utility(c) - utility(r)),
                  ci_lower=float(np.quantile(samples, alpha)),
                  ci_upper=float(np.quantile(samples, 1 - alpha)), seed=seed,
                  n_bootstrap=int(n_bootstrap))
    return result
