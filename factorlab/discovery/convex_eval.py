"""Cold, training-only statistical fits on two-dimensional feature frames.

Fixed-feature ridge is convex. Parameter profiling is bounded local search,
not a global or convex guarantee. This module never imports neural machinery.
"""
from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import numpy as np
from scipy.optimize import minimize

from factorlab.protocol import fingerprint, project_path

from .known_library import FactorFrame

VOLATILITY_FLOOR = 1e-8


def _matrix(value) -> np.ndarray:
    result = np.asarray(value.values if isinstance(value, FactorFrame) else value,
                        dtype=np.float64)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError('Expected finite two-dimensional features; select valid support explicitly')
    return result


def _weights(n, sample_weight=None):
    if n <= 0:
        raise ValueError('At least one training row is required')
    weights = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if (weights.shape != (n,) or not np.isfinite(weights).all()
            or (weights < 0).any() or not weights.max() > 0):
        raise ValueError('Weights must be finite, nonnegative and have positive total mass')
    # Normalize before summation so large but finite weights cannot overflow.
    weights = weights / weights.max()
    return weights / weights.sum()


def _target(y, n):
    target = np.asarray(y, dtype=np.float64)
    if target.shape != (n,) or not np.isfinite(target).all() or (target < 0).any():
        raise ValueError('Volatility target must be a finite nonnegative vector')
    return target


def _positive(value, name):
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')


@dataclass(frozen=True)
class TrainScaler:
    center: np.ndarray
    scale: np.ndarray
    bypass_mask: tuple[bool, ...]

    def __post_init__(self):
        for name in ('center', 'scale'):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @classmethod
    def fit(cls, X, sample_weight=None, bypass_mask=None):
        matrix = _matrix(X)
        weights = _weights(len(matrix), sample_weight)
        bypass = tuple(False for _ in range(matrix.shape[1])) if bypass_mask is None else tuple(bypass_mask)
        if len(bypass) != matrix.shape[1] or any(type(v) is not bool for v in bypass):
            raise ValueError('Scaler bypass mask must align with feature columns')
        center = weights @ matrix
        scale = np.sqrt(weights @ np.square(matrix - center))
        scale[scale <= np.finfo(float).eps] = 1.0
        center[np.asarray(bypass, dtype=bool)] = 0.0
        scale[np.asarray(bypass, dtype=bool)] = 1.0
        return cls(center, scale, bypass)

    def transform(self, X):
        matrix = _matrix(X)
        if matrix.shape[1] != len(self.center):
            raise ValueError('Scaler feature count mismatch')
        output = (matrix - self.center) / self.scale
        if not np.isfinite(output).all():
            raise ValueError('Nonfinite scaled features')
        return output

    def to_dict(self):
        return {'center': self.center.tolist(), 'scale': self.scale.tolist(),
                'bypass_mask': list(self.bypass_mask)}

    @classmethod
    def from_dict(cls, value):
        center = np.asarray(value['center'], dtype=float)
        scale = np.asarray(value['scale'], dtype=float)
        bypass = tuple(value['bypass_mask'])
        if (center.ndim != 1 or scale.shape != center.shape or len(bypass) != len(center)
                or any(type(v) is not bool for v in bypass) or not np.isfinite(center).all()
                or not np.isfinite(scale).all() or (scale <= 0).any()):
            raise ValueError('Invalid saved scaler')
        return cls(center, scale, bypass)


@dataclass(frozen=True)
class RidgePredictor:
    scaler: TrainScaler
    coefficients: np.ndarray
    intercept: float
    ridge: float
    feature_names: tuple[str, ...]
    input_schema_hash: str | None = None
    provenance: dict = field(default_factory=dict)
    kind: str = field(default='v7_log_ridge', init=False)
    version: int = field(default=1, init=False)
    task_support: tuple[str, ...] = field(default=('volatility',), init=False)

    def __post_init__(self):
        coefficients = np.array(self.coefficients, dtype=float, copy=True)
        coefficients.setflags(write=False)
        object.__setattr__(self, 'coefficients', coefficients)
        object.__setattr__(self, 'provenance', deepcopy(self.provenance))

    @property
    def schema(self):
        return {'feature_names': list(self.feature_names),
                'bypass_mask': list(self.scaler.bypass_mask),
                'input_schema_hash': self.input_schema_hash, 'dtype': 'float64',
                'target': 'log(y+1e-8)', 'decode': 'max(exp(q)-1e-8,0)'}

    def _input(self, X):
        if isinstance(X, FactorFrame):
            if X.feature_names != self.feature_names:
                raise ValueError('Feature names/order differ from fitted schema')
            if self.input_schema_hash is not None and X.schema_hash != self.input_schema_hash:
                raise ValueError('Feature policy differs from fitted schema')
        matrix = _matrix(X)
        if matrix.shape[1] != len(self.coefficients):
            raise ValueError('Predictor feature count mismatch')
        return matrix

    def predict_log(self, X):
        output = self.intercept + self.scaler.transform(self._input(X)) @ self.coefficients
        if not np.isfinite(output).all():
            raise ValueError('Nonfinite log prediction')
        return output

    def predict_with_diagnostics(self, X):
        q = self.predict_log(X)
        with np.errstate(over='ignore', invalid='ignore'):
            raw = np.exp(q) - VOLATILITY_FLOOR
        if not np.isfinite(raw).all():
            raise ValueError('Nonfinite decoded prediction')
        prediction = np.maximum(raw, 0.0)
        return prediction, {'clipped_count': int((raw < 0).sum()),
                            'clipped_fraction': float(np.mean(raw < 0)) if len(raw) else 0.0}

    def predict(self, X):
        return self.predict_with_diagnostics(X)[0]

    def to_dict(self):
        return {'kind': self.kind, 'version': self.version, 'scaler': self.scaler.to_dict(),
                'coefficients': self.coefficients.tolist(), 'intercept': self.intercept,
                'ridge': self.ridge, 'feature_names': list(self.feature_names),
                'input_schema_hash': self.input_schema_hash, 'provenance': deepcopy(self.provenance),
                'target_floor': VOLATILITY_FLOOR}

    @property
    def digest(self):
        return fingerprint(self.to_dict())

    def save(self, path: str | Path) -> Path:
        """Save a single immutable, hash-checked model JSON inside the project."""
        target = project_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {'digest': self.digest, 'model': self.to_dict()}
        if target.exists():
            if type(self).load(target).digest == self.digest:
                return target
            raise FileExistsError('Refusing to overwrite a different fitted model')
        temporary = target.with_name(f'.{target.name}.{uuid4().hex}.tmp')
        try:
            with temporary.open('x', encoding='utf-8', newline='\n') as stream:
                json.dump(payload, stream, sort_keys=True, allow_nan=False, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            # Publication must not replace a concurrently published model.
            try:
                os.link(temporary, target)
            except FileExistsError:
                if type(self).load(target).digest != self.digest:
                    raise
        finally:
            temporary.unlink(missing_ok=True)
        return target

    @classmethod
    def load(cls, path: str | Path):
        payload = json.loads(project_path(path).read_text('utf-8'))
        model = payload['model']
        if payload.get('digest') != fingerprint(model):
            raise ValueError('Fitted model integrity mismatch')
        if (model.get('kind') != 'v7_log_ridge' or model.get('version') != 1
                or model.get('target_floor') != VOLATILITY_FLOOR):
            raise ValueError('Unsupported model kind/version/target')
        coefficients = np.asarray(model['coefficients'], dtype=float)
        names = tuple(model['feature_names'])
        scaler = TrainScaler.from_dict(model['scaler'])
        if (coefficients.shape != (len(names),) or len(set(names)) != len(names)
                or len(scaler.center) != len(names) or not np.isfinite(coefficients).all()
                or not math.isfinite(model['intercept'])):
            raise ValueError('Invalid fitted model shape or numeric state')
        _positive(model['ridge'], 'ridge')
        result = cls(scaler, coefficients, model['intercept'], model['ridge'], names,
                     model['input_schema_hash'], model['provenance'])
        if result.digest != payload['digest']:
            raise ValueError('Saved model fields are not a lossless supported schema')
        return result


def fit_ridge(X, y, *, ridge=1e-3, sample_weight=None, bypass_mask=None,
              feature_names=None, provenance=None) -> RidgePredictor:
    """Minimize weighted mean log-square loss + ridge*||w||², cold each call.

    Zero-weight rows are permitted; NaN rows are never silently dropped. Arrays
    carry positional semantics; FactorFrame additionally binds policy and order.
    """
    _positive(ridge, 'ridge')
    matrix = _matrix(X)
    target = _target(y, len(matrix))
    weights = _weights(len(matrix), sample_weight)
    if isinstance(X, FactorFrame):
        if feature_names is not None and tuple(feature_names) != X.feature_names:
            raise ValueError('Explicit feature names conflict with FactorFrame')
        if bypass_mask is not None and tuple(bypass_mask) != X.bypass_mask:
            raise ValueError('Explicit bypass mask conflicts with FactorFrame')
        feature_names, bypass_mask = X.feature_names, X.bypass_mask
    names = tuple(f'x{i}' for i in range(matrix.shape[1])) if feature_names is None else tuple(feature_names)
    if len(names) != matrix.shape[1] or len(set(names)) != len(names):
        raise ValueError('Feature names must align and be unique')
    scaler = TrainScaler.fit(matrix, weights, bypass_mask)
    design = scaler.transform(matrix)
    q = np.log(target + VOLATILITY_FLOOR)
    xbar, qbar = weights @ design, float(weights @ q)
    root_weights = np.sqrt(weights)
    # Augmented least squares avoids explicitly forming or inverting X'X.
    augmented = np.vstack(((design - xbar) * root_weights[:, None],
                           math.sqrt(ridge) * np.eye(design.shape[1])))
    response = np.concatenate(((q - qbar) * root_weights, np.zeros(design.shape[1])))
    from .nominal_fit_outcome import ridge_lstsq
    coefficients = ridge_lstsq(augmented, response)[0]
    intercept = qbar - float(xbar @ coefficients)
    model = RidgePredictor(scaler, coefficients, intercept, float(ridge), names,
                           X.schema_hash if isinstance(X, FactorFrame) else None,
                           {} if provenance is None else dict(provenance))
    # Validate JSON provenance and all fitted values before returning a model.
    _ = model.digest
    return model


def log_mse(y, prediction, sample_weight=None):
    """Canonical score on decoded volatility, including the nonnegative boundary."""
    pred = np.asarray(prediction, dtype=float)
    target = _target(y, len(pred))
    if pred.shape != target.shape or not np.isfinite(pred).all() or (pred < 0).any():
        raise ValueError('Predictions must be finite nonnegative volatility')
    weights = _weights(len(pred), sample_weight)
    return float(weights @ np.square(np.log(target + VOLATILITY_FLOOR)
                                     - np.log(pred + VOLATILITY_FLOOR)))


@dataclass(frozen=True)
class ResidualProjector:
    coefficients: np.ndarray
    rank: int
    rcond: float
    residual_scale: float
    residual_scale_ratio: float
    near_zero: bool
    condition_number: float | None

    @classmethod
    def fit(cls, B, z, *, sample_weight=None, rcond=1e-12, near_zero_ratio=1e-8,
            absolute_floor=1e-12):
        """Weighted training projection on [1,B], with explicit SVD tolerance."""
        _positive(rcond, 'rcond')
        _positive(near_zero_ratio, 'near_zero_ratio')
        _positive(absolute_floor, 'absolute_floor')
        matrix = _matrix(B)
        feature = np.asarray(z, dtype=float)
        if feature.shape != (len(matrix),) or not np.isfinite(feature).all():
            raise ValueError('Residual candidate must be one finite scalar per row')
        weights = _weights(len(matrix), sample_weight)
        design = np.column_stack((np.ones(len(matrix)), matrix))
        from .nominal_fit_outcome import residual_lstsq
        coefficients, _, rank, singular = residual_lstsq(
            design * np.sqrt(weights)[:, None], feature * np.sqrt(weights), rcond)
        residual = feature - design @ coefficients
        scale = float(np.sqrt(weights @ np.square(residual)))
        feature_scale = float(np.sqrt(weights @ np.square(feature - weights @ feature)))
        ratio = scale / max(feature_scale, absolute_floor)
        zero = scale <= absolute_floor or ratio <= near_zero_ratio
        condition = (float(singular[0] / singular[-1])
                     if len(singular) and singular[-1] > rcond * singular[0] else None)
        return cls(coefficients, int(rank), float(rcond), scale, ratio, bool(zero), condition)

    def transform(self, B, z, *, normalize=False):
        matrix = _matrix(B)
        feature = np.asarray(z, dtype=float)
        if (feature.shape != (len(matrix),) or not np.isfinite(feature).all()
                or matrix.shape[1] + 1 != len(self.coefficients)):
            raise ValueError('Residual transform shape/support mismatch')
        residual = feature - np.column_stack((np.ones(len(matrix)), matrix)) @ self.coefficients
        if normalize and not self.near_zero:
            residual = residual / self.residual_scale
        if not np.isfinite(residual).all():
            raise ValueError('Nonfinite projected feature')
        return residual

    def augment(self, B: FactorFrame, z, *, name='candidate_residual') -> FactorFrame:
        """Safe F-pipeline input: preserve B and bypass a second residual scaling.

        Normalization uses this training projection's scale. In particular a
        near-zero residual remains tiny; fit_ridge must not standardize it back
        to unit variance. Evaluation calls reuse the very same projector.
        """
        if not isinstance(B, FactorFrame) or name in B.feature_names:
            raise ValueError('Residual augmentation requires a FactorFrame and a new column name')
        residual = self.transform(B, z, normalize=True)
        policy = fingerprint({'base_schema': B.schema_hash, 'projection': self.to_dict(),
                              'name': name, 'normalization': 'training_only_near_zero_bypass'})
        return FactorFrame(np.column_stack((B.values, residual)), B.metadata,
                           (*B.feature_names, name), (*B.bypass_mask, True), policy)

    def to_dict(self):
        return {'coefficients': self.coefficients.tolist(), 'rank': self.rank,
                'rcond': self.rcond, 'residual_scale': self.residual_scale,
                'residual_scale_ratio': self.residual_scale_ratio,
                'near_zero': self.near_zero, 'condition_number': self.condition_number}

    @classmethod
    def from_dict(cls, value):
        coefficients = np.asarray(value['coefficients'], dtype=float)
        if coefficients.ndim != 1 or not np.isfinite(coefficients).all():
            raise ValueError('Invalid saved residual projection')
        fingerprint(value)
        return cls(coefficients, **{k: v for k, v in value.items() if k != 'coefficients'})


@dataclass(frozen=True)
class ProfileResult:
    parameters: np.ndarray
    predictor: RidgePredictor
    objective: float
    calls: int
    history: tuple[dict, ...]
    status: str


class ProfileFitError(ValueError):
    """All attempts failed; charged calls and failure evidence remain inspectable."""

    def __init__(self, history):
        self.history = tuple(history)
        self.calls = len(history)
        super().__init__(f'No valid profile fit in {self.calls} charged calls')


def profile_parameters(feature_factory, y, *, bounds, initializations, max_calls,
                       ridge=1e-3, sample_weight=None, bypass_mask=None,
                       provenance=None) -> ProfileResult:
    """Bounded local profiling using TRAINING data only, one solve per call.

    ``max_calls`` is a global budget across all starts (and all numeric-search
    calls), including failures. Each caller/fold must share the outer run ledger
    if it needs a global monthly cap. No validation labels enter this API.
    """
    if type(max_calls) is not int or max_calls < 1:
        raise ValueError('max_calls must be a positive integer')
    limits = np.asarray(bounds, dtype=float)
    starts = np.asarray(initializations, dtype=float)
    if (limits.ndim != 2 or limits.shape[1] != 2 or not len(limits)
            or not np.isfinite(limits).all() or (limits[:, 0] >= limits[:, 1]).any()):
        raise ValueError('Bounds must be nonempty finite strictly ordered intervals')
    if (starts.ndim != 2 or starts.shape[1] != len(limits) or not len(starts)
            or not np.isfinite(starts).all() or (starts < limits[:, 0]).any()
            or (starts > limits[:, 1]).any()):
        raise ValueError('Every initialization must lie within all parameter bounds')
    _positive(ridge, 'ridge')
    target = _target(y, len(y))
    weights = _weights(len(target), sample_weight)
    history, best = [], None

    class BudgetReached(Exception):
        pass

    def objective(theta):
        nonlocal best
        if len(history) >= max_calls:
            raise BudgetReached
        theta = np.asarray(theta, dtype=float)
        record = {'call': len(history) + 1, 'parameters': theta.tolist(), 'status': 'started'}
        history.append(record)
        try:
            if (theta < limits[:, 0]).any() or (theta > limits[:, 1]).any():
                raise ValueError('Optimizer proposed parameters outside the registered bounds')
            features = feature_factory(theta.copy())
            model = fit_ridge(features, target, ridge=ridge, sample_weight=weights,
                              bypass_mask=bypass_mask, provenance=provenance)
            loss = float(weights @ np.square(np.log(target + VOLATILITY_FLOOR)
                                              - model.predict_log(features))
                         + ridge * np.square(model.coefficients).sum())
            if not math.isfinite(loss):
                raise ValueError('Nonfinite profile objective')
            record.update(status='ok', objective=loss)
            if best is None or loss < best[0]:
                best = (loss, theta.copy(), model)
            return loss
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            record.update(status='failed', error_type=type(exc).__name__)
            return float('inf')

    # All starts get an initial evaluation before local refinement can consume
    # the remaining budget. No uncounted final refit is performed.
    for start in starts:
        if len(history) == max_calls:
            break
        objective(start)
    for start in starts:
        if len(history) == max_calls:
            break
        try:
            minimize(objective, start, method='Powell', bounds=limits,
                     options={'maxfev': max_calls - len(history), 'xtol': 1e-5, 'ftol': 1e-7})
        except BudgetReached:
            break
    if best is None:
        raise ProfileFitError(history)
    return ProfileResult(best[1], best[2], best[0], len(history), tuple(history),
                         'budget_exhausted' if len(history) == max_calls else 'optimizer_stopped')
