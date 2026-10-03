"""Offline linear-perpetual cash ledger, with explicit mark and funding conventions."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .statistics import risk_metrics


@dataclass(frozen=True)
class BacktestConfig:
    initial_equity: float = 100_000.
    fee_bps: float = 5.
    slippage_bps: float = 3.
    gross_limit: float = 1.
    single_name_limit: float = .3
    maintenance_margin_rate: float = 0.
    lot_sizes: Mapping[str, float] | None = None
    min_notionals: Mapping[str, float] | None = None

    def __post_init__(self):
        vals = [self.initial_equity, self.fee_bps, self.slippage_bps,
                self.gross_limit, self.single_name_limit, self.maintenance_margin_rate]
        if (not np.isfinite(vals).all() or self.initial_equity <= 0
                or self.fee_bps < 0 or not 0 <= self.slippage_bps < 10_000
                or not 0 < self.single_name_limit <= self.gross_limit <= 1
                or not 0 <= self.maintenance_margin_rate < 1):
            raise ValueError("Invalid cash, cost or exposure settings")
        for rules, strictly_positive in ((self.lot_sizes, True), (self.min_notionals, False)):
            if rules is not None and any(not np.isfinite(v) or v < 0 or (strictly_positive and v == 0)
                                         for v in rules.values()):
                raise ValueError("Lot sizes must be positive and minimum notionals nonnegative")


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict
    assumptions: tuple[str, ...]


def _time_index(index, label):
    if (not isinstance(index, pd.DatetimeIndex) or index.tz is None
            or str(index.tz) != "UTC" or index.has_duplicates
            or not index.is_monotonic_increasing or index.hasnans):
        raise ValueError(f"{label} requires a sorted, unique UTC DatetimeIndex")


def _prices(frame, label):
    _time_index(frame.index, label)
    if not len(frame) or not len(frame.columns) or frame.columns.has_duplicates:
        raise ValueError(f"{label} must have rows and unique symbols")
    values = frame.to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"{label} has missing/nonpositive prices; gaps and delistings require explicit treatment")


def run_backtest(execution_prices: pd.DataFrame, target_weights: pd.DataFrame | None = None, *,
                 funding_events=None, mark_prices=None, config=None,
                 weight_policy=None) -> BacktestResult:
    """Targets are already causally scheduled by the caller at execution timestamps.

    Each bar: variation-margin PnL, funding held before that timestamp's rebalance,
    rebalance with adverse fills, then fees. Derivative principal never leaves cash.
    No hidden price filling, assumed periodic funding, or end-of-test liquidation.
    """
    config = BacktestConfig() if config is None else config
    _prices(execution_prices, "Execution prices")
    index, symbols = execution_prices.index, execution_prices.columns
    if not index.equals(index.floor("h")):
        raise ValueError("Execution prices must be at whole-hour boundaries")
    if len(index) > 1 and not (np.diff(index.asi8) == pd.Timedelta(hours=1).value).all():
        raise ValueError("Execution price gaps are unsupported; supply every hourly bar")
    if target_weights is None:
        if weight_policy is None:
            raise ValueError("Supply target_weights or weight_policy")
        target_weights = pd.DataFrame(0., index=index, columns=symbols)
    _time_index(target_weights.index, "Targets")
    if not symbols.equals(target_weights.columns) or not target_weights.index.isin(index).all():
        raise ValueError("Target symbols must match prices and timestamps must be execution bars")

    def validate_weights(values):
        if (not np.isfinite(values).all()
                or (np.abs(values) > config.single_name_limit + 1e-10).any()
                or (np.abs(values).sum(axis=-1) > config.gross_limit + 1e-10).any()):
            raise ValueError("Targets must be finite and satisfy gross and single-name limits")

    validate_weights(target_weights.to_numpy(float))
    for rules in (config.lot_sizes, config.min_notionals):
        if rules is not None and not set(rules).issubset(set(symbols)):
            raise ValueError("Execution rules refer to unknown symbols")
    marks = execution_prices if mark_prices is None else mark_prices
    _prices(marks, "Mark prices")
    if not symbols.equals(marks.columns) or not index.equals(marks.index):
        raise ValueError("Mark prices must explicitly cover the identical hourly bars and symbols")
    if funding_events is None:
        events = pd.DataFrame(columns=["symbol", "funding_time", "rate"])
    else:
        events = funding_events.copy()
        if not {"symbol", "funding_time", "rate"}.issubset(events.columns):
            raise ValueError("Funding events require symbol, funding_time and rate")
        # Do not coerce naïve timestamps into UTC: this would conceal an ambiguous source clock.
        for value in events.funding_time:
            stamp = pd.Timestamp(value)
            if stamp.tzinfo is None or stamp.utcoffset() != pd.Timedelta(0):
                raise ValueError("Funding timestamps must be UTC aware")
        events["funding_time"] = pd.to_datetime(events.funding_time, utc=True)
        if (not events.symbol.isin(symbols).all() or not np.isfinite(events.rate.to_numpy(float)).all()
                or events.duplicated(["symbol", "funding_time"]).any()):
            raise ValueError("Funding symbols/rates must be valid with no duplicate events")
        events = events.loc[events.funding_time.between(index[0], index[-1])].sort_values("funding_time")
    event_rows = list(events.itertuples(index=False))
    event_cursor = 0
    qty = np.zeros(len(symbols))
    cash = config.initial_equity
    previous_nav = cash
    previous_mark = marks.iloc[0].to_numpy(float)
    rows, trades = [], []
    for timestamp in index:
        mark = marks.loc[timestamp].to_numpy(float)
        price_pnl = float(qty @ (mark - previous_mark))
        cash += price_pnl
        funding = 0.
        while event_cursor < len(event_rows) and event_rows[event_cursor].funding_time <= timestamp:
            event = event_rows[event_cursor]
            j = symbols.get_loc(event.symbol)
            asof = marks.index.searchsorted(event.funding_time, side="right") - 1
            payment = -qty[j] * float(marks.iloc[asof, j]) * float(event.rate)
            cash += payment
            funding += payment
            event_cursor += 1
        gross_before = float(np.abs(qty * mark).sum())
        if cash <= 0 or cash < config.maintenance_margin_rate * gross_before:
            raise ValueError(f"Maintenance stress breach at {timestamp}; exchange liquidation is unsupported")
        fees = slippage = turnover = execution_basis_pnl = 0.
        if timestamp in target_weights.index:
            execution = execution_prices.loc[timestamp].to_numpy(float)
            desired = target_weights.loc[timestamp].to_numpy(float)
            if weight_policy is not None:
                actual = pd.Series(qty * execution / cash, index=symbols)
                recommendation = weight_policy(timestamp, actual, cash)
                if not isinstance(recommendation, pd.Series) or not recommendation.index.equals(symbols):
                    raise ValueError("weight_policy must return a Series with identical symbols")
                desired = recommendation.to_numpy(float)
                validate_weights(desired)
            target = desired * cash / execution
            for j, symbol in enumerate(symbols):
                lot = None if config.lot_sizes is None else config.lot_sizes.get(symbol)
                if lot is not None:
                    target[j] = np.trunc(target[j] / lot) * lot
                delta = target[j] - qty[j]
                minimum = 0. if config.min_notionals is None else config.min_notionals.get(symbol, 0.)
                if abs(delta * execution[j]) < minimum or abs(delta) < 1e-14:
                    continue
                fill = execution[j] * (1 + np.sign(delta) * config.slippage_bps / 10_000)
                notional = abs(delta * fill)
                fee = notional * config.fee_bps / 10_000
                slip = abs(delta) * execution[j] * config.slippage_bps / 10_000
                # New contracts enter marked at this boundary; variation vs. fill is immediate.
                cash -= delta * (fill - mark[j]) + fee
                execution_basis_pnl += delta * (mark[j] - execution[j])
                qty[j] += delta
                fees += fee
                slippage += slip
                turnover += abs(delta * execution[j])
                trades.append({"timestamp": timestamp, "symbol": symbol, "quantity": delta,
                               "execution_price": execution[j], "fill_price": fill,
                               "notional": notional, "fee": fee, "slippage": slip,
                               "position_quantity": qty[j]})
        gross = float(np.abs(qty * mark).sum())
        breach = cash <= 0 or cash < config.maintenance_margin_rate * gross
        if breach:
            raise ValueError(f"Maintenance stress breach after execution at {timestamp}")
        rows.append({"timestamp": timestamp, "equity": cash, "cash": cash,
                     "return": cash / previous_nav - 1, "price_pnl": price_pnl,
                     "funding": funding, "fees": fees, "slippage": slippage,
                     "execution_basis_pnl": execution_basis_pnl,
                     "turnover": turnover, "gross_exposure": gross / cash,
                     "net_exposure": float(qty @ mark) / cash, "maintenance_breach": breach})
        previous_nav, previous_mark = cash, mark
    equity = pd.DataFrame(rows).set_index("timestamp")
    columns = ["timestamp", "symbol", "quantity", "execution_price", "fill_price", "notional",
               "fee", "slippage", "position_quantity"]
    trades_frame = pd.DataFrame(trades, columns=columns)
    metrics = risk_metrics(equity["return"], equity.equity)
    metrics.update(total_fees=float(equity.fees.sum()), total_slippage=float(equity.slippage.sum()),
                   total_funding=float(equity.funding.sum()), total_turnover=float(equity.turnover.sum()),
                   final_equity=float(cash),
                   final_quantities={str(symbol): float(qty[j]) for j, symbol in enumerate(symbols)})
    assumptions = (
        "Offline linear perpetuals; quantity times mark change is settled into cash once per hourly bar.",
        "Caller schedules causally available predictions at the next eligible whole-hour execution bar.",
        "Funding uses actual supplied events and pre-rebalance positions; positive rates charge longs.",
        "Intrahour funding uses the last observed hourly boundary mark, with no future interpolation.",
        "Targets use pre-trade NAV and execution prices; costs can slightly raise post-trade exposure ratios.",
        "Hourly marks must be observable at their timestamp; execution prices proxy marks if omitted.",
        "Funding outside the simulated closed interval is excluded; no missing funding events are synthesized.",
        "No hidden price filling, delisting assumption, final liquidation, or live exchange connections.",
        "Maintenance is a simple stress threshold; a breach raises rather than simulates exchange liquidation.",
    )
    return BacktestResult(equity, trades_frame, metrics, assumptions)
