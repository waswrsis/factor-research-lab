"""Independent synthetic proofs of the P1 evaluation/execution contract."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from factorlab import evaluation
from factorlab.dataset import Dataset
from factorlab.portfolio import PortfolioResult
from factorlab.protocol import Protocol


def fixture_inputs(signal_hours=range(60, 66)):
    index = pd.date_range("2025-01-01", periods=90, freq="h", tz="UTC")
    # Nonconstant history gives an identifiable covariance; no fitted model is used.
    prices = 100 * np.cumprod(1 + .001 + .0005 * np.sin(np.arange(90)))
    bars = pd.DataFrame({"symbol": "BTC", "open_time": index, "open": prices})
    marks = bars.copy()
    meta = pd.DataFrame({"entry_time": index[list(signal_hours)], "symbol": "BTC"})
    meta["available_at"] = meta.entry_time - pd.Timedelta(minutes=59, seconds=55)
    n = len(meta)
    data = Dataset(np.zeros((n, 1, 1), np.float32), np.zeros(n), np.ones(n), meta, ("f",))
    funding = pd.DataFrame({"symbol": ["BTC"] * 3,
                            "funding_time": index[[16, 24, 61]],
                            "funding_rate": [.001, .003, .02]})
    funding["available_at"] = funding.funding_time + pd.Timedelta(seconds=5)
    protocol = replace(Protocol(), symbols=("BTC",))
    return data, bars, marks, funding, protocol, index


def constant_allocator(monkeypatch, captured, status="ok"):
    def allocate(mean, covariance, previous, *, config):
        captured.append((mean.copy(), covariance.copy(), previous.copy(), config))
        weights = pd.Series(.2 if status == "ok" else 0., index=mean.index)
        return PortfolioResult(weights, status, 0. if status == "ok" else None, {})
    monkeypatch.setattr(evaluation, "optimize_portfolio", allocate)


def evaluate(inputs, *, start_hour=60, end_hour=66):
    data, bars, marks, funding, protocol, index = inputs
    return evaluation.evaluate_predictions(data, np.full(len(data.y), .05), bars,
                                           marks, funding, index[start_hour], index[end_hour],
                                           protocol)


def test_forecast_uses_past_funding_but_payment_uses_actual_future_event(monkeypatch):
    inputs = fixture_inputs()
    captured = []
    constant_allocator(monkeypatch, captured)
    result = evaluate(inputs)
    # Two historical rates average .002 over 8h; forecasted 12h charge is .003.
    assert captured[0][0]["BTC"] == pytest.approx(.05 - .003)
    assert captured[1][0]["BTC"] == pytest.approx(.05 - .003)
    # Event at 61h is charged on the position held before 61h's rebalance.
    opening_qty = result.trades.iloc[0].position_quantity
    event_mark = inputs[2].iloc[61].open
    assert result.equity.funding.loc[inputs[-1][61]] == pytest.approx(-opening_qty * event_mark * .02)
    changed = list(inputs)
    changed[3] = inputs[3].copy()
    changed[3].loc[2, "funding_rate"] = .07
    captured.clear()
    altered = evaluate(tuple(changed))
    assert captured[0][0]["BTC"] == pytest.approx(.05 - .003)
    assert captured[1][0]["BTC"] == pytest.approx(.05 - .003)
    assert altered.equity.funding.loc[inputs[-1][61]] == pytest.approx(-opening_qty * event_mark * .07)


def test_current_and_future_opens_do_not_change_first_covariance(monkeypatch):
    inputs = fixture_inputs()
    captured = []
    constant_allocator(monkeypatch, captured)
    evaluate(inputs)
    earlier = captured[0][1]
    changed = list(inputs)
    for item in (1, 2):
        changed[item] = inputs[item].copy()
        changed[item].loc[60:, "open"] *= 1.5
    captured.clear()
    evaluate(tuple(changed))
    pd.testing.assert_frame_equal(captured[0][1], earlier)


def test_terminal_closure_pays_costs_and_reconciles_mark_basis(monkeypatch):
    inputs = list(fixture_inputs())
    inputs[2] = inputs[2].copy()
    inputs[2]["open"] *= 1.01
    captured = []
    constant_allocator(monkeypatch, captured)
    result = evaluate(tuple(inputs))
    assert result.metrics["final_quantities"] == {"BTC": 0.0}
    terminal = result.trades[result.trades.timestamp == inputs[-1][65]]
    assert len(terminal) == 1
    assert terminal.fee.iloc[0] > 0
    assert terminal.slippage.iloc[0] > 0
    change = result.equity.equity.diff()
    change.iloc[0] = result.equity.equity.iloc[0] - inputs[4].initial_equity
    components = (result.equity.price_pnl + result.equity.execution_basis_pnl
                  + result.equity.funding - result.equity.fees - result.equity.slippage)
    np.testing.assert_allclose(change, components, rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(1 + result.equity["return"],
                               result.equity.equity / np.r_[inputs[4].initial_equity,
                                                            result.equity.equity.iloc[:-1]])
    assert result.metrics["total_return"] == pytest.approx(
        result.equity.equity.iloc[-1] / inputs[4].initial_equity - 1)


def test_missing_signal_goes_flat_but_hourly_cash_timeline_is_preserved(monkeypatch):
    inputs = fixture_inputs(signal_hours=[60, 62])
    captured = []
    constant_allocator(monkeypatch, captured)
    result = evaluate(inputs)
    pd.testing.assert_index_equal(result.equity.index, inputs[-1][60:66], check_names=False)
    closed = result.trades[result.trades.timestamp == inputs[-1][61]]
    assert closed.position_quantity.iloc[0] == 0
    assert result.equity.gross_exposure.loc[inputs[-1][64]] == 0
    assert result.equity["return"].loc[inputs[-1][64]] == 0
    assert len(captured) == 2


def test_solver_failure_is_visible_and_confirmation_cannot_pass(monkeypatch):
    inputs = fixture_inputs()
    captured = []
    constant_allocator(monkeypatch, captured, status="cash_fallback")
    result = evaluate(inputs)
    assert result.trades.empty
    assert result.metrics["solver_failures"] == 5
    assert result.metrics["final_quantities"] == {"BTC": 0.0}
    # Force a favorable synthetic CI to prove solver failures independently veto promotion.
    monkeypatch.setattr(evaluation, "paired_block_bootstrap",
                        lambda *a, **k: {"status": "ok", "ci_lower": .001})
    gate = evaluation.confirmation_gate(result, result, inputs[4])
    assert gate["passed"] is False
    assert gate["decision"] == "retain_baseline"


def test_failure_after_open_closes_existing_position_with_real_costs(monkeypatch):
    inputs = fixture_inputs()
    calls = []

    def allocate(mean, covariance, previous, *, config):
        status = "ok" if not calls else "cash_fallback"
        calls.append(previous.copy())
        return PortfolioResult(pd.Series(.2 if status == "ok" else 0., index=mean.index),
                               status, None, {})

    monkeypatch.setattr(evaluation, "optimize_portfolio", allocate)
    result = evaluate(inputs)
    close = result.trades[result.trades.timestamp == inputs[-1][61]]
    assert len(close) == 1
    assert calls[1]["BTC"] > 0
    assert close.position_quantity.iloc[0] == 0
    assert close.quantity.iloc[0] < 0
    assert close.fee.iloc[0] > 0
    assert close.slippage.iloc[0] > 0
    assert result.equity.funding.loc[inputs[-1][61]] < 0
    assert result.metrics["solver_failures"] == 4


def test_evaluator_rejects_hourly_market_gap(monkeypatch):
    inputs = list(fixture_inputs())
    captured = []
    constant_allocator(monkeypatch, captured)
    inputs[1] = inputs[1].drop(63)
    inputs[2] = inputs[2].drop(63)
    with pytest.raises(ValueError, match="gaps"):
        evaluate(tuple(inputs))
