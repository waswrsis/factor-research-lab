"""Frozen task, architecture and experiment contracts for research-only workflows."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from factorlab.protocol import fingerprint, utc

from .robust_selection import RobustSelectionSpec


def _positive(value: int, name: str, maximum: int = 1_000_000) -> None:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f'{name} must be an integer in 1..{maximum}')


@dataclass(frozen=True)
class TaskSpec:
    name: str = 'volatility'
    kind: str = 'volatility'
    horizon_hours: int = 12
    event_bins: tuple[int, ...] = (4, 12)
    barrier_multiplier: float = 1.0

    def __post_init__(self):
        if not isinstance(self.event_bins, tuple):
            raise ValueError('Event bins must be an immutable tuple')  # noqa: TRY004
        if not self.name.isidentifier() or self.kind not in {
            'volatility', 'first_event', 'state', 'return'
        }:
            raise ValueError('Invalid task name or kind')
        _positive(self.horizon_hours, 'horizon_hours', 168)
        if (not math.isfinite(self.barrier_multiplier) or self.barrier_multiplier <= 0):
            raise ValueError('Barrier multiplier must be positive')
        if self.kind == 'first_event' and (
            not self.event_bins or tuple(sorted(set(self.event_bins))) != self.event_bins
            or any(type(x) is not int or x <= 0 for x in self.event_bins)
            or self.event_bins[-1] != self.horizon_hours
        ):
            raise ValueError('Event bins must increase and end at task horizon')

    @property
    def output_size(self) -> int:
        return 2 * len(self.event_bins) + 1 if self.kind == 'first_event' else 1

    @property
    def predictive(self) -> bool:
        return self.kind != 'state'


@dataclass
class TaskDataset:
    x: np.ndarray
    targets: dict[str, np.ndarray]
    observed: dict[str, np.ndarray]
    maturity: dict[str, np.ndarray]
    meta: pd.DataFrame
    features: tuple[str, ...]
    tasks: tuple[TaskSpec, ...]

    def __post_init__(self):
        n = len(self.meta)
        names = {t.name for t in self.tasks}
        if (self.x.ndim != 3 or self.x.shape[0] != n
            or self.x.shape[2] != len(self.features) or self.x.dtype != np.float32
            or not np.isfinite(self.x).all()):
            raise ValueError('X must be finite float32 [N, lookback, features]')
        if (len(names) != len(self.tasks) or not names or len(set(self.features)) != len(self.features)
            or set(self.targets) != names or set(self.observed) != names
            or set(self.maturity) != names):
            raise ValueError('Task and feature schemas must be unique and complete')
        if not {'sample_id', 'symbol', 'available_at'} <= set(self.meta):
            raise ValueError('Missing metadata columns')
        if self.meta.sample_id.duplicated().any() or self.meta.available_at.isna().any():
            raise ValueError('Sample identity must be unique and time must exist')
        for task in self.tasks:
            y, valid, ready = (self.targets[task.name], self.observed[task.name],
                               self.maturity[task.name])
            if (y.shape != (n,) or valid.shape != (n,) or valid.dtype != np.bool_
                or ready.shape != (n,) or ready.dtype.kind != 'M'):
                raise ValueError('Task targets, masks and maturity must align')
            if not np.isfinite(y[valid]).all() or np.isnat(ready[valid]).any():
                raise ValueError('Observed labels must be finite and have maturity')
            if task.kind == 'first_event' and (
                (y[valid] != np.floor(y[valid])).any()
                or (y[valid] < 0).any() or (y[valid] >= task.output_size).any()
            ):
                raise ValueError('Event categories outside task support')
            if (task.predictive and valid.any()
                and (ready[valid] <= self.meta.available_at.to_numpy(dtype='datetime64[ns]')[valid]).any()):
                raise ValueError('Predictive targets must mature after observation')

    def subset(self, mask: np.ndarray) -> TaskDataset:
        indices = np.asarray(mask)
        if indices.dtype == np.bool_ and indices.shape != (len(self.meta),):
            raise ValueError('Boolean subset mask must align')
        return TaskDataset(self.x[indices], {k: v[indices] for k, v in self.targets.items()},
                           {k: v[indices] for k, v in self.observed.items()},
                           {k: v[indices] for k, v in self.maturity.items()},
                           self.meta.iloc[indices].reset_index(drop=True), self.features, self.tasks)

    def task_mask(self, name: str, cutoff: str | pd.Timestamp | None = None) -> np.ndarray:
        valid = self.observed[name].copy()
        if cutoff is not None:
            valid &= self.maturity[name] <= utc(cutoff).to_datetime64()
        return valid

    @property
    def schema(self) -> str:
        return fingerprint({'features': self.features, 'lookback': self.x.shape[1],
                            'tasks': [asdict(t) for t in self.tasks]})


@dataclass(frozen=True)
class ModuleSpec:
    name: str = 'temporal'
    kind: str = 'temporal'
    dim: int = 8
    width: int = 16
    scale: int = 12
    features: tuple[str, ...] = ()
    depth: int = 1
    program: dict[str, Any] | None = None

    def __post_init__(self):
        if not isinstance(self.features, tuple):
            raise ValueError('Module features must be an immutable tuple')  # noqa: TRY004
        if (not isinstance(self.name, str) or not self.name.isidentifier()
            or not isinstance(self.kind, str) or self.kind not in {'summary', 'temporal', 'mlp', 'program'}):
            raise ValueError('Invalid representation module')
        for key, maximum in [('dim', 64), ('width', 128), ('scale', 672), ('depth', 3)]:
            _positive(getattr(self, key), key, maximum)
        if any(not isinstance(f, str) or not f for f in self.features):
            raise ValueError('Module features must be nonempty strings')
        if len(set(self.features)) != len(self.features):
            raise ValueError('Duplicate module features')
        if self.kind == 'program':
            from .operator_program import validate_program

            object.__setattr__(self, 'program', validate_program(
                self.program, features=self.features or None))
        elif self.program is not None:
            raise ValueError('Only program modules may declare an operator program')


@dataclass(frozen=True)
class RepresentationSpec:
    modules: tuple[ModuleSpec, ...] = field(default_factory=lambda: (ModuleSpec(),))
    combination: str = 'concat'

    def __post_init__(self):
        if not isinstance(self.modules, tuple) or not all(isinstance(m, ModuleSpec) for m in self.modules):
            raise ValueError('Modules must be a tuple of ModuleSpec')
        if (not 1 <= len(self.modules) <= 6
            or len({m.name for m in self.modules}) != len(self.modules)
            or self.combination not in {'concat', 'gated'}):
            raise ValueError('Invalid module set or combination')
        if self.combination == 'gated' and len({m.dim for m in self.modules}) != 1:
            raise ValueError('Gated representations require equal dimensions')

    @property
    def digest(self) -> str:
        value = asdict(self)
        # Depth one is the historical two-linear-layer architecture. Preserve
        # identities of frozen pre-depth architecture records.
        for module in value['modules']:
            if module['depth'] == 1:
                module.pop('depth')
            # Historical modules had no program field. Preserve all prior
            # checkpoint and architecture identities when no DSL is present.
            if module['program'] is None:
                module.pop('program')
        return fingerprint(value)

    @classmethod
    def from_dict(cls, value: dict) -> RepresentationSpec:
        value = dict(value)
        value['modules'] = tuple(ModuleSpec(**{**m, 'features': tuple(m.get('features', ()))})
                                 for m in value['modules'])
        return cls(**value)


@dataclass(frozen=True)
class TrainingSpec:
    epochs: int = 8
    batch_size: int = 256
    learning_rate: float = 0.001
    l2: float = 0.001
    patience: int = 3
    min_delta: float = 0.0001
    clip_norm: float = 5.0
    seed: int = 17
    max_parameters: int = 200_000
    max_seconds: int = 120
    selection_task: str | None = None

    def __post_init__(self):
        for name in ('epochs', 'batch_size', 'patience', 'max_parameters', 'max_seconds'):
            _positive(getattr(self, name), name)
        for name in ('learning_rate', 'l2', 'clip_norm'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError('Invalid minimum training improvement')
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError('Invalid seed')
        if self.selection_task is not None and (
            not isinstance(self.selection_task, str) or not self.selection_task.isidentifier()
        ):
            raise ValueError('selection_task must be a valid task name or None')


@dataclass(frozen=True)
class ExperimentSpec:
    version: str = 'discovery-v3'
    tasks: tuple[TaskSpec, ...] = field(default_factory=lambda: (
        TaskSpec(), TaskSpec('first_event', 'first_event'), TaskSpec('state', 'state'),
    ))
    primary_task: str = 'volatility'
    lookback: int = 48
    symbols: tuple[str, ...] = ('BTCUSDT', 'ETHUSDT', 'SOLUSDT')
    start: str = '2025-01-01'
    train_end: str = '2025-07-01'
    dev_end: str = '2025-09-01'
    calibration_end: str = '2025-10-01'
    end: str = '2026-01-01'
    training: TrainingSpec = field(default_factory=TrainingSpec)
    training_seeds: tuple[int, ...] = (17, 29)
    candidates: int = 4
    search_seconds: int = 900
    bootstrap_block_hours: int = 48
    bootstrap_samples: int = 1000
    min_blocks: int = 20
    minimum_increment: float = 0.005
    confidence: float = 0.95
    seed: int = 17
    availability_delay_seconds: int = 5
    selection: RobustSelectionSpec = field(default_factory=RobustSelectionSpec)

    def __post_init__(self):
        if (not isinstance(self.tasks, tuple) or not all(isinstance(t, TaskSpec) for t in self.tasks)
            or not isinstance(self.symbols, tuple) or not isinstance(self.training_seeds, tuple)
            or not isinstance(self.training, TrainingSpec)
            or not isinstance(self.selection, RobustSelectionSpec)):
            raise ValueError('Protocol collections must be immutable typed tuples')
        if not self.tasks or len({t.name for t in self.tasks}) != len(self.tasks):
            raise ValueError('Tasks must be nonempty and unique')
        if self.primary_task not in {t.name for t in self.tasks}:
            raise ValueError('Primary task missing')
        if not next(t for t in self.tasks if t.name == self.primary_task).predictive:
            raise ValueError('Primary discovery claim requires a predictive task')
        if self.selection.mode == 'walk_forward' and self.training.selection_task != self.primary_task:
            raise ValueError('Walk-forward checkpoint selection must match primary_task')
        if not self.symbols or len(set(self.symbols)) != len(self.symbols):
            raise ValueError('Universe must be nonempty and unique')
        if (not self.training_seeds or len(set(self.training_seeds)) != len(self.training_seeds)
            or any(type(s) is not int or not 0 <= s < 2**31 for s in self.training_seeds)):
            raise ValueError('Training seeds must be unique nonnegative integers')
        for name in ('lookback', 'candidates', 'search_seconds', 'bootstrap_block_hours',
                     'bootstrap_samples', 'min_blocks', 'availability_delay_seconds'):
            _positive(getattr(self, name), name)
        if self.lookback > 672 or self.candidates > 100:
            raise ValueError('Local research resource limit exceeded')
        boundaries = [utc(getattr(self, key)) for key in
                      ('start', 'train_end', 'dev_end', 'calibration_end', 'end')]
        if boundaries != sorted(set(boundaries)):
            raise ValueError('Strict chronological partitions required')
        if not 0.5 < self.confidence < 1 or not math.isfinite(self.minimum_increment):
            raise ValueError('Invalid confirmation criteria')
        if self.minimum_increment < 0 or type(self.seed) is not int or self.seed < 0:
            raise ValueError('Invalid increment or seed')

    @property
    def digest(self) -> str:
        return fingerprint(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> ExperimentSpec:
        value = dict(value)
        if 'tasks' in value:
            value['tasks'] = tuple(TaskSpec(**{**t, 'event_bins': tuple(t.get('event_bins', (4, 12)))})
                                   for t in value['tasks'])
        if 'training' in value:
            value['training'] = TrainingSpec(**value['training'])
        if 'selection' in value:
            value['selection'] = RobustSelectionSpec(**value['selection'])
        if 'symbols' in value:
            value['symbols'] = tuple(value['symbols'])
        if 'training_seeds' in value:
            value['training_seeds'] = tuple(value['training_seeds'])
        return cls(**value)


def partition(data: TaskDataset, spec: ExperimentSpec, name: str) -> TaskDataset:
    boundaries = {'train': (spec.start, spec.train_end),
                  'development': (spec.train_end, spec.dev_end),
                  'calibration': (spec.dev_end, spec.calibration_end),
                  'confirmation': (spec.calibration_end, spec.end)}
    start, end = boundaries[name]
    subset = data.subset(((data.meta.available_at >= utc(start)) &
                         (data.meta.available_at < utc(end))).to_numpy())
    for task in subset.tasks:
        subset.observed[task.name] &= subset.task_mask(task.name, end)
    return subset
