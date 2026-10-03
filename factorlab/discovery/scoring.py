"""Task-specific losses and conditional, dependence-aware research evidence.

Training diagnostics never enter these scores. A missing task is unavailable;
invalid predictions anywhere inside a returned task are errors, even where the
label is unobserved. Volatility uses the same log(y + floor) target as training.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from factorlab.discovery.contracts import ExperimentSpec, TaskDataset, TaskSpec
from factorlab.protocol import project_path, save_json

VOLATILITY_FLOOR = 1e-8
PROBABILITY_FLOOR = 1e-12


def _probabilities(prediction: np.ndarray, classes: int) -> np.ndarray:
    p = np.asarray(prediction, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != classes:
        raise ValueError(f'Joint probabilities must have shape [N, {classes}]')
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Joint probabilities must be finite and in [0, 1]')
    mass = p.sum(axis=1)
    if not np.allclose(mass, 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError('Joint probability mass must sum to one')
    return p / mass[:, None]


def per_sample_loss(task: TaskSpec, target: np.ndarray,
                    prediction: np.ndarray) -> np.ndarray:
    """Score complete event outcomes, or task-defined regression errors."""
    y = np.asarray(target, dtype=np.float64)
    if y.ndim != 1 or not np.isfinite(y).all():
        raise ValueError('Targets must be finite [N]')
    if task.kind == 'first_event':
        p = _probabilities(prediction, task.output_size)
        if len(p) != len(y):
            raise ValueError('Predictions and targets must align')
        if ((y != np.floor(y)).any() or (y < 0).any()
                or (y >= task.output_size).any()):
            raise ValueError('Event target outside joint category support')
        return -np.log(np.maximum(p[np.arange(len(y)), y.astype(np.int64)],
                                  PROBABILITY_FLOOR))
    p = np.asarray(prediction, dtype=np.float64)
    if p.shape == (len(y), 1):
        p = p[:, 0]
    if p.shape != y.shape or not np.isfinite(p).all():
        raise ValueError('Regression predictions must be finite and align with targets')
    if task.kind == 'volatility' and ((p < 0).any() or (y < 0).any()):
        raise ValueError('Volatility predictions and targets must be nonnegative')
    with np.errstate(over='ignore', invalid='ignore'):
        if task.kind == 'volatility':
            loss = (np.log(p + VOLATILITY_FLOOR)
                    - np.log(y + VOLATILITY_FLOOR)) ** 2
        else:
            loss = (p - y) ** 2
    if not np.isfinite(loss).all():
        raise ValueError('Nonfinite task loss')
    return loss


def _aligned_prediction(data: TaskDataset, task: TaskSpec,
                        predictions: dict[str, np.ndarray]) -> np.ndarray:
    if task.name not in predictions:
        raise ValueError(f'Missing task prediction: {task.name}')
    p = np.asarray(predictions[task.name], dtype=np.float64)
    n = len(data.meta)
    shapes = {(n, task.output_size)} if task.kind == 'first_event' else {(n,), (n, 1)}
    if p.shape not in shapes:
        raise ValueError(f'Predictions for {task.name} must align with all dataset rows')
    if not np.isfinite(p).all():
        raise ValueError(f'Predictions for {task.name} must be finite on all dataset rows')
    if task.kind == 'first_event':
        _probabilities(p, task.output_size)
    elif task.kind == 'volatility' and (p < 0).any():
        raise ValueError('Volatility predictions must be nonnegative on all dataset rows')
    return p


def _probability_diagnostics(y: np.ndarray, p: np.ndarray) -> dict:
    target = np.eye(p.shape[1])[y.astype(np.int64)]
    confidence = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    bins = []
    ece = 0.0
    for i in range(10):
        lo, hi = i / 10, (i + 1) / 10
        mask = (confidence >= lo) & ((confidence < hi) if i < 9 else (confidence <= hi))
        count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else None
        mean_confidence = float(confidence[mask].mean()) if count else None
        if count:
            ece += count / len(y) * abs(accuracy - mean_confidence)
        bins.append({'lower': lo, 'upper': hi, 'support': count,
                     'mean_probability': mean_confidence, 'observed_frequency': accuracy})
    return {'brier': float(np.square(p - target).sum(axis=1).mean()),
            'calibration_kind': 'top_label', 'calibration_bins': bins,
            'expected_calibration_error': float(ece),
            'calibration_is_diagnostic_only': True}


def evaluate(data: TaskDataset, predictions: dict[str, np.ndarray]) -> dict:
    """Evaluate every declared task, retaining unavailable and unobserved tasks."""
    if set(predictions) - {t.name for t in data.tasks}:
        raise ValueError('Predictions contain an undeclared task')
    results = {}
    total = len(data.meta)
    for task in data.tasks:
        mask = data.task_mask(task.name)
        eligible = int(mask.sum())
        record = {'kind': task.kind, 'predictive': task.predictive,
                  'interpretation': 'future_target' if task.predictive else 'descriptive_only',
                  'eligible_support': eligible, 'support': 0, 'loss': None,
                  'coverage': 0.0, 'label_coverage': eligible / total if total else 0.0,
                  'coverage_scope': 'provided_dataset_observed_task_labels',
                  'total_rows': total}
        if task.name not in predictions:
            record['status'] = 'unavailable'
        else:
            p = _aligned_prediction(data, task, predictions)
            if not eligible:
                record['status'] = 'insufficient_evidence'
            else:
                y = data.targets[task.name][mask]
                losses = per_sample_loss(task, y, p[mask])
                record.update(status='ok', support=eligible, loss=float(losses.mean()),
                              coverage=eligible / total)
                if task.kind == 'first_event':
                    record.update(_probability_diagnostics(y, _probabilities(p[mask],
                                                                           task.output_size)))
        results[task.name] = record
    return results


def paired_comparison(data: TaskDataset, candidate: dict[str, np.ndarray],
                      reference: dict[str, np.ndarray], task_name: str,
                      spec: ExperimentSpec, family_size: int = 1) -> dict:
    """Compare paired losses, averaging simultaneous symbols before block resampling.

    Nonoverlapping calendar blocks preserve all shared-symbol shocks and within-block
    dependence. Empty hours are not compressed into adjacent observations. The
    interval is conditional on this dataset's eligible scope, not a universe claim.
    """
    if type(family_size) is not int or family_size < 1:
        raise ValueError('Comparison family size must be a positive integer')
    task = next((t for t in data.tasks if t.name == task_name), None)
    if task is None:
        raise ValueError('Comparison task is not in the dataset')
    frozen = next((t for t in spec.tasks if t.name == task_name), None)
    if task != frozen:
        raise ValueError('Comparison task must match the frozen experiment task')
    cp = _aligned_prediction(data, task, candidate)
    rp = _aligned_prediction(data, task, reference)
    mask = data.task_mask(task_name)
    n = int(mask.sum())
    adjusted = 1 - (1 - spec.confidence) / family_size
    result = {'task': task_name, 'predictive': task.predictive,
              'support': n, 'total_rows': len(data.meta),
              'coverage': n / len(data.meta) if len(data.meta) else 0.0,
              'coverage_scope': 'provided_dataset_observed_task_labels',
              'conditional_scope_only': True,
              'minimum_increment': spec.minimum_increment,
              'confidence': spec.confidence, 'family_size': family_size,
              'adjusted_confidence': adjusted, 'estimate': None,
              'ci_lower': None, 'ci_upper': None, 'n_blocks': 0,
              'n_hours': 0, 'status': 'insufficient_evidence',
              'predictive_claim_supported': False,
              'aggregation': 'equal_symbol_within_hour_then_equal_hour',
              'bootstrap': 'nonoverlapping_calendar_time_blocks',
              'block_hours': spec.bootstrap_block_hours,
              'bootstrap_samples': spec.bootstrap_samples}
    if not n:
        result['reason'] = 'no_observed_task_labels'
        return result
    y = data.targets[task_name][mask]
    improvements = per_sample_loss(task, y, rp[mask]) - per_sample_loss(task, y, cp[mask])
    rows = data.meta.loc[mask, ['symbol', 'available_at']].copy()
    rows['hour'] = pd.to_datetime(rows.available_at, utc=True).dt.floor('h')
    rows['improvement'] = improvements
    # Avoid letting repeated decisions for one symbol overweight a shared hour.
    hourly = rows.groupby(['hour', 'symbol'], observed=True).improvement.mean()
    hourly = hourly.groupby(level='hour').mean().sort_index()
    hour_id = hourly.index.asi8 // (3600 * 1_000_000_000)
    blocks = pd.DataFrame({'value': hourly.to_numpy(),
                           'block': hour_id // spec.bootstrap_block_hours}).groupby('block').value
    sums, counts = blocks.sum().to_numpy(), blocks.count().to_numpy()
    count = len(sums)
    result.update(estimate=float(hourly.mean()), n_blocks=count, n_hours=len(hourly),
                  symbols=sorted(rows.symbol.unique().tolist()),
                  start=hourly.index[0].isoformat(), end=hourly.index[-1].isoformat())
    if count < spec.min_blocks:
        result['reason'] = 'minimum_time_blocks_not_met'
        return result
    rng = np.random.default_rng(spec.seed)
    boot = np.empty(spec.bootstrap_samples, dtype=np.float64)
    for begin in range(0, spec.bootstrap_samples, 128):
        size = min(128, spec.bootstrap_samples - begin)
        sampled = rng.integers(0, count, size=(size, count))
        boot[begin:begin + size] = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    tail = (1 - spec.confidence) / family_size / 2
    lower, upper = np.quantile(boot, [tail, 1 - tail])
    if lower > spec.minimum_increment:
        status = 'supported'
    elif upper <= spec.minimum_increment:
        status = 'target_not_met'
    else:
        status = 'insufficient_evidence'
    result.update(ci_lower=float(lower), ci_upper=float(upper), status=status,
                  predictive_claim_supported=task.predictive and status == 'supported',
                  reason='interval_relative_to_frozen_minimum_increment')
    return result


def _temperature(p: np.ndarray, temperature: float) -> np.ndarray:
    if temperature == 1.0:
        return p.copy()
    logits = np.log(np.maximum(p, PROBABILITY_FLOOR)) / temperature
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


class ProbabilityCalibrator:
    """One bounded positive temperature per complete joint event distribution."""

    def __init__(self, *, min_support: int = 20,
                 temperature_bounds: tuple[float, float] = (0.05, 20.0)):
        if type(min_support) is not int or min_support < 1:
            raise ValueError('Minimum calibration support must be positive')
        bounds = tuple(float(v) for v in temperature_bounds)
        if (len(bounds) != 2 or not all(math.isfinite(v) for v in bounds)
                or not 0 < bounds[0] <= 1 <= bounds[1] or bounds[0] == bounds[1]):
            raise ValueError('Temperature bounds must be positive, ordered and contain identity')
        self.min_support = min_support
        self.temperature_bounds = bounds
        self.tasks: tuple[TaskSpec, ...] = ()
        self.temperatures: dict[str, float] = {}
        self.diagnostics: dict[str, dict] = {}
        self.fitted = False

    def fit(self, calibration_data: TaskDataset,
            predictions: dict[str, np.ndarray]) -> ProbabilityCalibrator:
        if set(predictions) - {t.name for t in calibration_data.tasks}:
            raise ValueError('Calibration predictions contain an undeclared task')
        self.tasks = calibration_data.tasks
        self.temperatures = {}
        self.diagnostics = {}
        self.fitted = False
        for task in self.tasks:
            raw = (_aligned_prediction(calibration_data, task, predictions)
                   if task.name in predictions else None)
            if task.kind != 'first_event':
                self.diagnostics[task.name] = {'status': 'not_applicable',
                                               'transform': 'identity'}
                continue
            mask = calibration_data.task_mask(task.name)
            support = int(mask.sum())
            self.temperatures[task.name] = 1.0
            record = {'support': support, 'temperature': 1.0,
                      'before_nll': None, 'after_nll': None}
            self.diagnostics[task.name] = record
            if task.name not in predictions:
                record['status'] = 'unavailable'
                continue
            # Returned, eligible probabilities must remain valid even when support is small.
            p = _probabilities(raw[mask], task.output_size)
            y = calibration_data.targets[task.name][mask]
            loss = per_sample_loss(task, y, p)
            if support < self.min_support:
                record['status'] = 'no_support' if not support else 'insufficient_support'
                if support:
                    record.update(before_nll=float(loss.mean()), after_nll=float(loss.mean()))
                continue
            before = float(loss.mean())

            def objective(log_temperature: float, probabilities=p,
                          current_task=task, targets=y) -> float:
                transformed = _temperature(probabilities, math.exp(log_temperature))
                return float(per_sample_loss(current_task, targets, transformed).mean())

            fitted = minimize_scalar(objective, method='bounded',
                                     bounds=tuple(math.log(v) for v in self.temperature_bounds))
            temperature = math.exp(float(fitted.x))
            after = objective(float(fitted.x))
            if fitted.success and math.isfinite(after) and after < before:
                self.temperatures[task.name] = temperature
                record.update(status='fitted', temperature=temperature,
                              before_nll=before, after_nll=after)
            else:
                record.update(status='identity_retained', before_nll=before, after_nll=before)
        self.fitted = True
        return self

    def transform(self, predictions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if not self.fitted:
            raise ValueError('Calibrator must be fitted or loaded before use')
        if set(predictions) - {t.name for t in self.tasks}:
            raise ValueError('Predictions contain an undeclared calibration task')
        output = {name: np.asarray(value).copy() for name, value in predictions.items()}
        for task in self.tasks:
            if task.kind == 'first_event' and task.name in output:
                p = _probabilities(output[task.name], task.output_size)
                temperature = self.temperatures[task.name]
                if temperature != 1.0:
                    output[task.name] = _temperature(p, temperature)
        return output

    def to_dict(self) -> dict:
        if not self.fitted:
            raise ValueError('Cannot save an unfitted calibrator')
        return {'version': 1, 'min_support': self.min_support,
                'temperature_bounds': list(self.temperature_bounds),
                'tasks': [asdict(t) for t in self.tasks],
                'temperatures': dict(self.temperatures),
                'diagnostics': json.loads(json.dumps(self.diagnostics, allow_nan=False))}

    @classmethod
    def from_dict(cls, value: dict) -> ProbabilityCalibrator:
        if (set(value) != {'version', 'min_support', 'temperature_bounds', 'tasks',
                           'temperatures', 'diagnostics'} or value['version'] != 1):
            raise ValueError('Unknown calibrator format')
        result = cls(min_support=value['min_support'],
                     temperature_bounds=tuple(value['temperature_bounds']))
        result.tasks = tuple(TaskSpec(**{**t, 'event_bins': tuple(t['event_bins'])})
                             for t in value['tasks'])
        if not result.tasks or len({t.name for t in result.tasks}) != len(result.tasks):
            raise ValueError('Invalid calibrator task schema')
        event_names = {t.name for t in result.tasks if t.kind == 'first_event'}
        if (set(value['temperatures']) != event_names
                or set(value['diagnostics']) != {t.name for t in result.tasks}):
            raise ValueError('Incomplete calibrator task state')
        for name, temperature in value['temperatures'].items():
            if (not isinstance(temperature, (int, float)) or not math.isfinite(temperature)
                    or not result.temperature_bounds[0] <= temperature <= result.temperature_bounds[1]):
                raise ValueError(f'Invalid temperature for {name}')
        result.temperatures = {k: float(v) for k, v in value['temperatures'].items()}
        result.diagnostics = json.loads(json.dumps(value['diagnostics'], allow_nan=False))
        result.fitted = True
        return result

    def save(self, path: str | Path) -> Path:
        return save_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> ProbabilityCalibrator:
        return cls.from_dict(json.loads(project_path(path).read_text(encoding='utf-8')))


def adaptive_horizons(probabilities: np.ndarray, task: TaskSpec, *,
                      coverage_threshold: float = 0.8) -> dict:
    """Derive fixed-rule horizons; concentration is a diagnostic, not confidence.

    At each bin, no_event includes events after that bin plus complete no-event.
    Horizon selection uses unconditional cumulative event mass. Full-joint NLL
    remains the common evaluation target regardless of the selected horizon.
    """
    if task.kind != 'first_event':
        raise ValueError('Adaptive horizons require a first-event task')
    if not math.isfinite(coverage_threshold) or not 0 < coverage_threshold <= 1:
        raise ValueError('Coverage threshold must be in (0, 1]')
    p = _probabilities(probabilities, task.output_size)
    up = np.cumsum(p[:, 1::2], axis=1)
    down = np.cumsum(p[:, 2::2], axis=1)
    event = up + down
    no_event = np.clip(1 - event, 0, 1)
    reached = event >= coverage_threshold
    selected = np.where(reached.any(axis=1), reached.argmax(axis=1), len(task.event_bins) - 1)
    row = np.arange(len(p))
    event_mass = p[:, 1:].sum(axis=1)
    bin_mass = p[:, 1::2] + p[:, 2::2]
    concentration = np.divide(bin_mass.max(axis=1), event_mass,
                              out=np.zeros(len(p)), where=event_mass > 0)
    return {'horizon_hours': np.asarray(task.event_bins)[selected],
            'selected_up_probability': up[row, selected],
            'selected_down_probability': down[row, selected],
            'selected_no_event_probability': no_event[row, selected],
            'cumulative_up': up, 'cumulative_down': down, 'no_event_by_horizon': no_event,
            'coverage_achieved': reached.any(axis=1),
            'event_concentration': concentration,
            'joint_concentration': np.square(p).sum(axis=1),
            'concentration_is_confidence': False,
            'rule': {'coverage_threshold': float(coverage_threshold),
                     'mass': 'unconditional_cumulative_event',
                     'event_bins': list(task.event_bins), 'fallback_hours': task.horizon_hours},
            'evaluation_target': 'complete_joint_first_event_distribution'}
