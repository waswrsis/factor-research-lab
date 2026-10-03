"""Supervised samples with delayed execution and explicitly mature labels."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, build_features
from .protocol import Protocol, utc


@dataclass
class Dataset:
    x: np.ndarray
    y: np.ndarray
    scale: np.ndarray
    meta: pd.DataFrame
    features: tuple[str, ...]

    def subset(self, mask: np.ndarray) -> Dataset:
        return Dataset(self.x[mask], self.y[mask], self.scale[mask],
                       self.meta.loc[mask].reset_index(drop=True), self.features)


def make_dataset(bars: pd.DataFrame, protocol: Protocol) -> Dataset:
    unexpected = set(bars['symbol']) - set(protocol.symbols)
    if unexpected:
        raise ValueError(f'Assets outside the frozen universe: {sorted(unexpected)}')
    if not (bars['available_at'] == bars['close_time'] + pd.Timedelta(
        seconds=protocol.availability_delay_seconds
    )).all():
        raise ValueError('Input availability differs from frozen latency assumption')
    bars = bars.loc[(bars.open_time >= utc(protocol.start)) &
                    (bars.open_time < utc(protocol.end))].copy()
    frame = build_features(bars)
    samples, targets, scales, metadata = [], [], [], []
    lookback = protocol.lookback
    for symbol, g in frame.groupby('symbol', sort=True):
        g = g.sort_values('open_time').reset_index(drop=True)
        prices = g.set_index('open_time')['open']
        values = g[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
        for i in range(lookback - 1, len(g)):
            window = values[i-lookback+1:i+1]
            row = g.iloc[i]
            # The bar's close is unavailable at its printed boundary.
            entry = row['available_at'].ceil('h')
            exit_time = entry + pd.Timedelta(hours=protocol.horizon_hours)
            if entry not in prices.index or exit_time not in prices.index:
                continue
            if not np.isfinite(window).all():
                continue
            if row['open_time'] - g.iloc[i-lookback+1]['open_time'] != pd.Timedelta(
                hours=lookback-1
            ):
                continue
            # A missing hour anywhere in the outcome interval invalidates it.
            if len(prices.loc[entry:exit_time]) != protocol.horizon_hours + 1:
                continue
            scale = max(float(row['volatility_24h']) * np.sqrt(protocol.horizon_hours),
                        protocol.volatility_floor)
            samples.append(window)
            targets.append(float(prices.loc[exit_time] / prices.loc[entry] - 1))
            scales.append(scale)
            metadata.append({'symbol': symbol, 'feature_open_time': row['open_time'],
                             'available_at': row['available_at'], 'entry_time': entry,
                             'label_end': exit_time, 'label_available_at': exit_time +
                             pd.Timedelta(seconds=protocol.availability_delay_seconds)})
    if not samples:
        raise ValueError('No complete causal supervised samples')
    meta = pd.DataFrame(metadata)
    order = np.lexsort((meta.symbol.to_numpy(), meta.available_at.astype('int64').to_numpy()))
    return Dataset(np.stack(samples)[order], np.asarray(targets, dtype=np.float32)[order],
                   np.asarray(scales, dtype=np.float32)[order],
                   meta.iloc[order].reset_index(drop=True), tuple(FEATURE_COLUMNS))
