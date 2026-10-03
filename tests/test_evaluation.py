"""Independent timing/unit checks without market data or model fitting."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from factorlab.backtest import BacktestResult
from factorlab.dataset import Dataset
from factorlab.evaluation import confirmation_gate, evaluate_predictions
from factorlab.protocol import Protocol


def test_covariance_funding_and_execution_inputs_use_only_past(monkeypatch):
    from factorlab import evaluation

    protocol = Protocol(symbols=("BTCUSDT",))
    times = pd.date_range("2025-03-20", periods=210, freq="h", tz="UTC")
    start, end = times[-6], times[-1] + pd.Timedelta(hours=1)
    prices = 100 * np.exp(np.arange(len(times)) * 0.001 + np.sin(np.arange(len(times))) * 0.01)
    bars = pd.DataFrame({"open_time": times, "symbol": "BTCUSDT", "open": prices})
    funding_times = [
        start - pd.Timedelta(hours=25),
        start - pd.Timedelta(hours=17),
        start - pd.Timedelta(hours=9),
        start + pd.Timedelta(hours=8),
    ]
    funding = pd.DataFrame(
        {
            "symbol": "BTCUSDT",
            "funding_time": funding_times,
            "available_at": [x + pd.Timedelta(seconds=5) for x in funding_times],
            "funding_rate": [0.001, 0.001, 0.001, 99.0],
        }
    )
    entries = times[-6:-1]
    meta = pd.DataFrame(
        {
            "entry_time": entries,
            "symbol": "BTCUSDT",
            "available_at": entries - pd.Timedelta(minutes=1),
        }
    )
    ds = Dataset(np.zeros((5, 48, 1), np.float32), np.zeros(5), np.ones(5), meta, ("ret",))
    captured = []

    def allocate(mean, covariance, previous, *, config):
        captured.append((mean.copy(), covariance.copy(), previous.copy(), config))
        return SimpleNamespace(status="ok", weights=pd.Series(0.0, index=mean.index))

    monkeypatch.setattr(evaluation, "optimize_portfolio", allocate)
    result = evaluate_predictions(
        ds, np.full(5, 0.02), bars, bars.copy(), funding, start, end, protocol
    )
    expected = pd.Series(prices, index=times).pct_change().loc[lambda x: x.index < start].tail(168)
    assert captured[0][1].iloc[0, 0] == pytest.approx(expected.var() * 12)
    assert captured[0][0].iloc[0] == pytest.approx(0.02 - 0.001 * 12 / 8)
    assert captured[0][3].expected_return_horizon_hours == 12
    assert captured[0][3].covariance_horizon_hours == 12
    assert captured[0][3].turnover_penalty == pytest.approx(0.001 + 0.0008)
    # Future price revisions cannot change the first allocation covariance.
    altered = bars.copy()
    altered.loc[altered.open_time >= start, "open"] *= 100
    captured.clear()
    evaluate_predictions(
        ds, np.full(5, 0.02), altered, altered.copy(), funding, start, end, protocol
    )
    assert captured[0][1].iloc[0, 0] == pytest.approx(expected.var() * 12)
    assert result.metrics["final_quantities"]["BTCUSDT"] == 0


def test_confirmation_insufficient_time_blocks_cannot_promote():
    times = pd.date_range("2025-03-25", periods=168, freq="h", tz="UTC")
    candidate = BacktestResult(
        pd.DataFrame({"return": 0.01}, index=times),
        pd.DataFrame(),
        {"max_drawdown": 0, "solver_failures": 0},
        (),
    )
    baseline = BacktestResult(
        pd.DataFrame({"return": 0.0}, index=times),
        pd.DataFrame(),
        {"max_drawdown": 0, "solver_failures": 0},
        (),
    )
    result = confirmation_gate(candidate, baseline, Protocol())
    assert not result["passed"]
    assert result["decision"] == "retain_baseline"
    assert result["paired"]["status"] == "insufficient_sample"
    assert result["paired"]["n_blocks"] == 3
