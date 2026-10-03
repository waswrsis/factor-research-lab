"""Small causal benchmarks fitted only to a declared historical training prefix.

These are comparison models, not proof that a candidate uses its inputs. The
constant is fitted in the scored target space, avoiding an arithmetic-mean
volatility benchmark that is mismatched to log squared error.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from factorlab.protocol import fingerprint, utc

from .contracts import TaskDataset, TaskSpec
from .scoring import VOLATILITY_FLOOR


def macro_weights(data: TaskDataset, mask: np.ndarray) -> np.ndarray:
    """Weights equal symbols within UTC hour, then equal hours."""
    rows = data.meta.loc[mask, ['symbol', 'available_at']].copy()
    if rows.empty:
        raise ValueError('Benchmark requires observed training labels')
    rows['hour'] = pd.to_datetime(rows.available_at, utc=True).dt.floor('h')
    samples = rows.groupby(['hour', 'symbol']).symbol.transform('size').to_numpy()
    symbols = rows.groupby('hour').symbol.transform('nunique').to_numpy()
    weights = 1.0 / (samples * symbols * rows.hour.nunique())
    if not np.isclose(weights.sum(), 1):
        raise ValueError('Invalid macro weights')
    return weights


def _rolling(data: TaskDataset, task: TaskSpec) -> np.ndarray:
    if 'return_1h' not in data.features or data.x.shape[1] < 24:
        raise ValueError('Rolling benchmark requires 24 causal return_1h observations')
    returns = data.x[:, -24:, data.features.index('return_1h')].astype(np.float64)
    if (returns <= -1).any():
        raise ValueError('Simple returns must exceed -1')
    # Targets are square roots of future log-return energy, not demeaned std.
    logs = np.log1p(returns)
    return np.sqrt(np.square(logs).mean(axis=1) * task.horizon_hours)


@dataclass
class SimpleBenchmark:
    name: str
    schema: str
    tasks: tuple[TaskSpec, ...]
    constants: dict
    log_offsets: dict
    provenance: dict

    @property
    def digest(self):
        return fingerprint(self.to_dict())

    def to_dict(self):
        return asdict(self)

    def predict(self, data: TaskDataset) -> dict[str, np.ndarray]:
        if data.schema != self.schema:
            raise ValueError('Benchmark input schema changed')
        n = len(data.meta)
        result = {}
        for task in self.tasks:
            if task.kind == 'first_event':
                result[task.name] = np.tile(self.constants[task.name], (n, 1))
            elif task.kind == 'volatility' and self.name != 'training_constant':
                raw = _rolling(data, task)
                if self.name == 'calibrated_rolling_24h':
                    raw = np.maximum(0, np.exp(np.log(raw + VOLATILITY_FLOOR)
                                              + self.log_offsets[task.name]) - VOLATILITY_FLOOR)
                result[task.name] = raw
            else:
                result[task.name] = np.full(n, self.constants[task.name], dtype=np.float64)
            if not np.isfinite(result[task.name]).all():
                raise ValueError('Nonfinite benchmark forecast')
        return result


def fit_simple_benchmarks(train: TaskDataset, cutoff, *, include_rolling=True):
    """Fit all predeclared controls, never select one using evaluation labels.

    Rolling models use the same train-fitted constants for non-volatility tasks.
    The calibrated model learns only an additive log-volatility offset.
    """
    boundary = utc(cutoff)
    if not len(train.meta) or (train.meta.available_at >= boundary).any():
        raise ValueError('Benchmark training features must precede cutoff')
    constants, offsets, supports = {}, {}, {}
    for task in train.tasks:
        mask = train.task_mask(task.name)
        if not mask.any() or (train.maturity[task.name][mask] > boundary.to_datetime64()).any():
            raise ValueError('Benchmark labels must be supported and mature by cutoff')
        weights = macro_weights(train, mask)
        y = train.targets[task.name][mask]
        supports[task.name] = int(mask.sum())
        if task.kind == 'first_event':
            # One symmetric pseudo-hour per category. Duplicating a symbol's
            # observation inside an hour must not weaken this regularization.
            hours = pd.to_datetime(train.meta.loc[mask, 'available_at'], utc=True).dt.floor('h').nunique()
            counts = np.bincount(y.astype(int), weights=weights * hours,
                                 minlength=task.output_size) + 1.0
            constants[task.name] = (counts / counts.sum()).tolist()
        elif task.kind == 'volatility':
            if (y < 0).any():
                raise ValueError('Volatility benchmark labels must be nonnegative')
            log_y = np.log(y + VOLATILITY_FLOOR)
            constants[task.name] = float(max(0, np.exp(weights @ log_y) - VOLATILITY_FLOOR))
            if include_rolling:
                log_prediction = np.log(_rolling(train, task)[mask] + VOLATILITY_FLOOR)
                offsets[task.name] = float(weights @ (log_y - log_prediction))
        else:
            constants[task.name] = float(weights @ y)
    provenance = {
        'fit_cutoff': boundary.isoformat(),
        'fit_feature_cutoff': train.meta.available_at.max().isoformat(),
        'fit_label_cutoffs': {t.name: pd.Timestamp(train.maturity[t.name][train.task_mask(t.name)].max(),
                                                  tz='UTC').isoformat() for t in train.tasks},
        'training_support': supports, 'fitting_aggregation': 'equal_symbol_within_hour_then_equal_hour',
        'calibration_partition': 'training_only', 'evaluation_targets_used': False,
        'state_interpretation': 'descriptive_only',
        'event_smoothing': 'one_symmetric_pseudo_hour_per_category_not_independent_observation_count',
    }
    names = ['training_constant']
    if include_rolling and any(t.kind == 'volatility' for t in train.tasks):
        names += ['rolling_24h', 'calibrated_rolling_24h']
    return {name: SimpleBenchmark(name, train.schema, train.tasks, constants.copy(),
                                   offsets.copy() if name == 'calibrated_rolling_24h' else {},
                                   provenance.copy()) for name in names}
