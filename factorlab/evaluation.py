"""Causal portfolio evaluation separate from forecasting and search selection."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, BacktestResult, run_backtest
from .dataset import Dataset
from .portfolio import PortfolioConfig, optimize_portfolio
from .protocol import Protocol
from .statistics import paired_block_bootstrap


def evaluate_predictions(data: Dataset, prediction: np.ndarray, bars: pd.DataFrame,
                          marks: pd.DataFrame, funding: pd.DataFrame,
                          start: pd.Timestamp, end: pd.Timestamp,
                          protocol: Protocol, cost_multiplier: float = 1.0) -> BacktestResult:
    if len(data.y) != len(prediction) or not np.isfinite(prediction).all():
        raise ValueError('Finite predictions must align with sample metadata')
    signals = data.meta[['entry_time', 'symbol', 'available_at']].copy()
    if not (signals.available_at < signals.entry_time).all():
        raise ValueError('Signals must arrive strictly before execution')
    signals['prediction'] = prediction
    if signals.duplicated(['entry_time', 'symbol']).any():
        raise ValueError('Duplicate decision for an execution boundary')
    signals = signals.pivot(index='entry_time', columns='symbol', values='prediction')
    prices = bars.pivot(index='open_time', columns='symbol', values='open').sort_index()
    mark_prices = marks.pivot(index='open_time', columns='symbol', values='open').sort_index()
    symbols = pd.Index(protocol.symbols)
    prices = prices.reindex(columns=symbols)
    mark_prices = mark_prices.reindex(columns=symbols)
    timeline = prices.index[(prices.index >= start) & (prices.index < end)]
    if len(timeline) < 2:
        raise ValueError('Evaluation interval is too short')
    signals = signals.reindex(columns=symbols)
    price_returns = prices.pct_change(fill_method=None)
    pc = PortfolioConfig(risk_aversion=protocol.risk_aversion,
                         turnover_penalty=protocol.turnover_penalty + cost_multiplier * (
                             protocol.fee_bps + protocol.slippage_bps)/10_000,
                         gross_limit=protocol.gross_limit,
                         single_name_limit=protocol.single_name_limit)
    bc = BacktestConfig(initial_equity=protocol.initial_equity,
                        fee_bps=protocol.fee_bps * cost_multiplier,
                        slippage_bps=protocol.slippage_bps * cost_multiplier,
                        gross_limit=protocol.gross_limit,
                        single_name_limit=protocol.single_name_limit)
    failures = []

    def policy(timestamp, actual_weights, equity):
        # Close positions at the final boundary, including liquidation costs.
        if timestamp == timeline[-1] or timestamp not in signals.index:
            return pd.Series(0.0, index=symbols)
        signal = signals.loc[timestamp]
        if signal.isna().any():
            return pd.Series(0.0, index=symbols)
        # Only boundaries strictly before execution; last used close preceded
        # the model's assumed signal arrival. No current-hour range/close.
        history = price_returns.loc[price_returns.index < timestamp].tail(168).dropna()
        if len(history) < 48:
            return pd.Series(0.0, index=symbols)
        covariance = history.cov() * protocol.horizon_hours
        expected_funding = pd.Series(0.0, index=symbols)
        for symbol in symbols:
            past = funding.loc[(funding.symbol == symbol) &
                               (funding.available_at < timestamp-pd.Timedelta(hours=1))].tail(9)
            if len(past) >= 2:
                interval = past.funding_time.diff().dt.total_seconds().dropna().median()/3600
                expected_funding.loc[symbol] = past.funding_rate.mean() * protocol.horizon_hours / interval
        allocation = optimize_portfolio(signal-expected_funding, covariance, actual_weights, config=pc)
        if allocation.status != 'ok':
            failures.append({'time': str(timestamp), 'status': allocation.status})
        return allocation.weights

    result = run_backtest(prices.loc[timeline], funding_events=funding.rename(
        columns={'funding_rate': 'rate'}), mark_prices=mark_prices.loc[timeline],
        config=bc, weight_policy=policy)
    result.metrics['solver_failures'] = len(failures)
    result.metrics['utility_hourly'] = float(result.equity['return'].mean() -
                                           protocol.risk_aversion/2 * result.equity['return'].var(ddof=0))
    result.metrics['cost_multiplier'] = cost_multiplier
    return result


def confirmation_gate(candidate: BacktestResult, reference: BacktestResult,
                       protocol: Protocol) -> dict:
    paired = paired_block_bootstrap(candidate.equity['return'], reference.equity['return'],
                                    block_length=protocol.bootstrap_block_hours,
                                    n_bootstrap=protocol.bootstrap_samples,
                                    min_blocks=protocol.promotion_min_blocks,
                                    risk_aversion=protocol.risk_aversion, seed=17)
    passed = (paired['status'] == 'ok' and paired['ci_lower'] > 0 and
              candidate.metrics['max_drawdown'] <= protocol.max_drawdown and
              candidate.metrics.get('solver_failures', 0) == 0)
    return {'passed': bool(passed), 'paired': paired,
            'decision': 'eligible_for_shadow_only' if passed else 'retain_baseline',
            'limitation': 'Historical confirmation is not authorization for live trading.'}
