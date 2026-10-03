import numpy as np
import pandas as pd
import pytest

from factorlab.backtest import BacktestConfig, run_backtest


def prices(values):
    return pd.DataFrame({"BTC": values}, index=pd.date_range("2025-01-01", periods=len(values), freq="h", tz="UTC"))


def config(**kwargs):
    return BacktestConfig(initial_equity=1000, fee_bps=0, slippage_bps=0,
                          single_name_limit=1, **kwargs)


def test_manual_variation_margin_and_drift_rebalance():
    p = prices([100, 110, 121])
    target = pd.DataFrame({"BTC": [.5, .5]}, index=p.index[:2])
    result = run_backtest(p, target, config=config())
    # 5 contracts earn 50. Rebalance to 525/110 contracts; next mark earns 52.5.
    assert result.equity.equity.tolist() == pytest.approx([1000, 1050, 1102.5])
    assert result.trades.quantity.tolist() == pytest.approx([5, 525 / 110 - 5])
    assert result.equity["return"].tolist() == pytest.approx([0, .05, .05])
    assert result.metrics["final_quantities"]["BTC"] == pytest.approx(525 / 110)


def test_actual_funding_precedes_boundary_rebalance_and_has_signed_exposure():
    p = prices([100, 110, 110])
    targets = pd.DataFrame({"BTC": [.5, -.5]}, index=p.index[:2])
    funding = pd.DataFrame({"symbol": ["BTC", "BTC"], "funding_time": [p.index[1], p.index[2]],
                            "rate": [.01, .02]})
    result = run_backtest(p, targets, funding_events=funding, config=config())
    # Long pays 5*110*.01 before becoming short; short receives 4.7477*110*.02.
    assert result.equity.equity.tolist() == pytest.approx([1000, 1044.5, 1054.945])
    assert result.equity.funding.tolist() == pytest.approx([0, -5.5, 10.445])


def test_intrahour_funding_uses_past_mark_never_future_mark():
    p = prices([100, 200])
    funding = pd.DataFrame({"symbol": ["BTC"], "funding_time": [p.index[0] + pd.Timedelta(minutes=30)],
                            "rate": [.01]})
    targets = pd.DataFrame({"BTC": [.5]}, index=p.index[:1])
    result = run_backtest(p, targets, funding_events=funding, config=config())
    assert result.equity.funding.iloc[-1] == -5
    assert result.equity.equity.iloc[-1] == 1495


def test_manual_slippage_fee_reversal_and_mark_execution_basis():
    p = prices([100, 100])
    targets = pd.DataFrame({"BTC": [.5, -.5]}, index=p.index)
    cfg = BacktestConfig(initial_equity=1000, fee_bps=10, slippage_bps=20, single_name_limit=1)
    result = run_backtest(p, targets, config=cfg)
    first_nav = 1000 - 5 * .2 - 501 * .001
    delta = -.5 * first_nav / 100 - 5
    second_nav = first_nav - abs(delta) * .2 - abs(delta) * 99.8 * .001
    assert result.equity.equity.tolist() == pytest.approx([first_nav, second_nav])
    marks = prices([105, 105])
    marked = run_backtest(p.iloc[:1], targets.iloc[:1], mark_prices=marks.iloc[:1], config=config())
    assert marked.equity.equity.iloc[0] == 1025  # New contracts have immediate mark/fill basis.
    assert marked.equity.execution_basis_pnl.iloc[0] == 25
    accounted = (result.equity.price_pnl + result.equity.execution_basis_pnl
                 + result.equity.funding - result.equity.fees - result.equity.slippage)
    assert result.equity.equity.diff().fillna(result.equity.equity.iloc[0] - 1000).to_numpy() == pytest.approx(accounted.to_numpy())


def test_missing_prices_and_gaps_fail_closed():
    p = prices([100, np.nan])
    targets = pd.DataFrame({"BTC": [.5, .5]}, index=p.index)
    with pytest.raises(ValueError, match="missing"):
        run_backtest(p, targets, config=config())
    p = prices([100, 100, 100]).iloc[[0, 2]]
    with pytest.raises(ValueError, match="gaps"):
        run_backtest(p, targets.iloc[:0], config=config())


def test_prefix_causality_and_policy_sees_actual_ledger():
    p = prices([100, 110, 121, 150])
    targets = pd.DataFrame({"BTC": [.5, .5, .5, -.5]}, index=p.index)
    full = run_backtest(p, targets, config=config())
    prefix = run_backtest(p.iloc[:3], targets.iloc[:3], config=config())
    pd.testing.assert_frame_equal(full.equity.iloc[:3], prefix.equity)
    seen = []

    def policy(t, drifted, nav):
        seen.append((t, drifted["BTC"], nav))
        return pd.Series(.5, index=p.columns)

    run_backtest(p.iloc[:2], weight_policy=policy, config=config())
    assert seen[0][1:] == (0, 1000)
    assert seen[1][1:] == pytest.approx((550 / 1050, 1050))


def test_lot_minimum_notional_and_maintenance_stress():
    p = prices([100, 110])
    targets = pd.DataFrame({"BTC": [.35, .01]}, index=p.index)
    result = run_backtest(p, targets, config=config(lot_sizes={"BTC": 1}, min_notionals={"BTC": 400}))
    assert result.trades.empty
    p = prices([100, 160])
    targets = pd.DataFrame({"BTC": [-1.]}, index=p.index[:1])
    with pytest.raises(ValueError, match="Maintenance stress"):
        run_backtest(p, targets, config=config(maintenance_margin_rate=.5))


def test_no_implicit_funding_and_duplicate_event_rejected():
    p = prices([100] * 10)
    targets = pd.DataFrame({"BTC": [.5]}, index=p.index[:1])
    assert run_backtest(p, targets, config=config()).equity.funding.sum() == 0
    event = {"symbol": "BTC", "funding_time": p.index[1], "rate": .01}
    with pytest.raises(ValueError, match="duplicate"):
        run_backtest(p, targets, funding_events=pd.DataFrame([event, event]), config=config())
