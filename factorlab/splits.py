"""Time boundaries use when outcomes are observable, not row offsets."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .protocol import Protocol, utc


@dataclass(frozen=True)
class Fold:
    name: str
    validation_start: pd.Timestamp
    confirmation_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def outer_folds(protocol: Protocol) -> list[Fold]:
    ends = (*protocol.test_starts[1:], protocol.end)
    return [Fold(f'fold-{i}', utc(t) - pd.Timedelta(days=28),
                 utc(t) - pd.Timedelta(days=7), utc(t), utc(e))
            for i, (t, e) in enumerate(zip(protocol.test_starts, ends, strict=True))]


def segment_masks(meta: pd.DataFrame, fold: Fold) -> dict[str, np.ndarray]:
    """Purged chronological training, search, one-shot confirmation, next-block test.

    Labels ending exactly at a boundary are excluded because the outcome may
    only arrive after its timestamp. Test labels also must mature before end.
    """
    t = pd.to_datetime(meta['available_at'], utc=True)
    end = pd.to_datetime(meta['label_available_at'], utc=True)
    return {
        'train': ((t < fold.validation_start) & (end < fold.validation_start)).to_numpy(),
        'search': ((t >= fold.validation_start) &
                   (end < fold.confirmation_start)).to_numpy(),
        'confirmation': ((t >= fold.confirmation_start) &
                         (end < fold.test_start)).to_numpy(),
        'test': ((t >= fold.test_start) & (end < fold.test_end)).to_numpy(),
    }


def chronological_oof(meta: pd.DataFrame, cutoff: pd.Timestamp, parts: int = 3,
                      min_train_hours: int = 168):
    """OOF train/validation masks with strict label maturity for downstream stacking."""
    t = pd.to_datetime(meta['available_at'], utc=True)
    maturity = pd.to_datetime(meta['label_available_at'], utc=True)
    eligible = (maturity < cutoff)
    times = pd.DatetimeIndex(sorted(t[eligible].unique()))
    if len(times) <= min_train_hours + parts:
        raise ValueError('Insufficient chronological OOF history')
    edges = np.linspace(min_train_hours, len(times), parts + 1, dtype=int)
    for i in range(parts):
        start = times[edges[i]]
        end = times[edges[i + 1]] if edges[i + 1] < len(times) else cutoff
        train = (t < start) & (maturity < start)
        valid = (t >= start) & (t < end) & (maturity < cutoff)
        yield train.to_numpy(), valid.to_numpy()
