"""Independent seven-class first-event baseline, without neural imports.

Class 0 is no event. Classes 1..6 are up/down in the three frozen event bins.
Reference-class logits identify the model; positive L2 includes free intercepts
so training sets with missing classes still have a finite, unique optimum.
"""
from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from factorlab.protocol import fingerprint, project_path

from .convex_eval import TrainScaler, _matrix, _positive, _weights
from .known_library import FactorFrame, library_schema

EVENT_CLASSES = 7
PROBABILITY_FLOOR = 1e-12
RATIO_FLOOR = 1e-12
EVENT_FEATURE_NAMES = ('rv_6_over_24', 'rv_24_over_168',
                       'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos')


def build_event_library(frame: FactorFrame) -> FactorFrame:
    """Derive two raw RV ratios plus unchanged forecast-start calendar columns."""
    if not isinstance(frame, FactorFrame) or frame.schema_hash != fingerprint(library_schema()):
        raise ValueError('Event library requires the frozen raw K0 feature policy')
    index = {name: i for i, name in enumerate(frame.feature_names)}
    needed = ('rv_6', 'rv_24', 'rv_168', *EVENT_FEATURE_NAMES[2:])
    if not set(needed) <= set(index):
        raise ValueError('Event library requires RV and UTC calendar inputs')
    x = frame.values
    values = np.column_stack((
        x[:, index['rv_6']] / np.maximum(x[:, index['rv_24']], RATIO_FLOOR),
        x[:, index['rv_24']] / np.maximum(x[:, index['rv_168']], RATIO_FLOOR),
        *(x[:, index[name]] for name in EVENT_FEATURE_NAMES[2:]),
    ))
    schema = {'kind': 'K0_event_ratios', 'version': 1, 'source_schema': frame.schema_hash,
              'feature_names': EVENT_FEATURE_NAMES, 'ratio_denominator_floor': RATIO_FLOOR,
              'ratios': 'rv6/max(rv24,floor);rv24/max(rv168,floor)',
              'calendar': 'unchanged UTC forecast_window_start',
              'bypass_mask': [False, False, True, True, True, True]}
    return FactorFrame(values, frame.metadata, EVENT_FEATURE_NAMES,
                       (False, False, True, True, True, True), fingerprint(schema))


def _labels(y, n, classes):
    values = np.asarray(y)
    if (values.shape != (n,) or not np.issubdtype(values.dtype, np.number)
            or not np.isfinite(values).all() or (values != np.floor(values)).any()
            or (values < 0).any() or (values >= classes).any()):
        raise ValueError('Event labels must be integer codes in the complete class domain')
    return values.astype(np.int64)


@dataclass(frozen=True)
class EventPredictor:
    scaler: TrainScaler
    parameters: np.ndarray  # [intercept + feature, free class]
    ridge: float
    feature_names: tuple[str, ...]
    input_schema_hash: str | None
    provenance: dict = field(default_factory=dict)
    optimizer: dict = field(default_factory=dict)
    kind: str = field(default='v7_event_logit', init=False)
    version: int = field(default=1, init=False)
    task_support: tuple[str, ...] = field(default=('first_event',), init=False)
    classes: int = field(default=EVENT_CLASSES, init=False)

    def __post_init__(self):
        parameters = np.array(self.parameters, dtype=float, copy=True)
        if parameters.shape != (len(self.feature_names) + 1, EVENT_CLASSES - 1):
            raise ValueError('Event coefficient shape must retain all seven classes')
        if not np.isfinite(parameters).all():
            raise ValueError('Nonfinite event coefficient')
        parameters.setflags(write=False)
        object.__setattr__(self, 'parameters', parameters)
        object.__setattr__(self, 'provenance', deepcopy(self.provenance))
        object.__setattr__(self, 'optimizer', deepcopy(self.optimizer))

    @property
    def schema(self):
        return {'feature_names': list(self.feature_names),
                'bypass_mask': list(self.scaler.bypass_mask),
                'input_schema_hash': self.input_schema_hash, 'dtype': 'float64',
                'classes': EVENT_CLASSES, 'reference_class': 0,
                'class_semantics': '0=no_event;1+2*bin+down for three frozen bins',
                'regularization': 'ridge*sum(all_free_coefficients_and_intercepts^2)',
                'score_probability_floor': PROBABILITY_FLOOR}

    def predict_log_proba(self, X):
        if isinstance(X, FactorFrame):
            if X.feature_names != self.feature_names:
                raise ValueError('Feature names/order differ from event model schema')
            if self.input_schema_hash is not None and X.schema_hash != self.input_schema_hash:
                raise ValueError('Feature policy differs from event model schema')
        matrix = _matrix(X)
        if matrix.shape[1] != len(self.feature_names):
            raise ValueError('Event feature count mismatch')
        design = np.column_stack((np.ones(len(matrix)), self.scaler.transform(matrix)))
        logits = np.column_stack((np.zeros(len(matrix)), design @ self.parameters))
        if not np.isfinite(logits).all():
            raise ValueError('Nonfinite event logits')
        return logits - logsumexp(logits, axis=1, keepdims=True)

    def predict(self, X):
        return np.exp(self.predict_log_proba(X))

    def to_dict(self):
        return {'kind': self.kind, 'version': self.version, 'scaler': self.scaler.to_dict(),
                'parameters': self.parameters.tolist(), 'ridge': self.ridge,
                'feature_names': list(self.feature_names),
                'input_schema_hash': self.input_schema_hash,
                'provenance': deepcopy(self.provenance), 'optimizer': deepcopy(self.optimizer),
                'schema': self.schema}

    @property
    def digest(self):
        return fingerprint(self.to_dict())

    def save(self, path: str | Path):
        target = project_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if type(self).load(target).digest == self.digest:
                return target
            raise FileExistsError('Refusing to overwrite a different event model')
        temporary = target.with_name(f'.{target.name}.{uuid4().hex}.tmp')
        try:
            with temporary.open('x', encoding='utf-8', newline='\n') as stream:
                json.dump({'digest': self.digest, 'model': self.to_dict()}, stream,
                          sort_keys=True, allow_nan=False, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
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
        value = payload['model']
        if fingerprint(value) != payload.get('digest'):
            raise ValueError('Event model integrity mismatch')
        if value.get('kind') != 'v7_event_logit' or value.get('version') != 1:
            raise ValueError('Unsupported event model version')
        _positive(value['ridge'], 'ridge')
        scaler = TrainScaler.from_dict(value['scaler'])
        names = tuple(value['feature_names'])
        if len(set(names)) != len(names) or len(scaler.center) != len(names):
            raise ValueError('Invalid event feature schema')
        result = cls(scaler, value['parameters'], value['ridge'], names,
                     value['input_schema_hash'], value['provenance'], value['optimizer'])
        if result.digest != payload['digest']:
            raise ValueError('Event model is not a lossless supported schema')
        return result


def fit_event(X, y, *, classes=7, ridge=1e-3, sample_weight=None, bypass_mask=None,
              feature_names=None, provenance=None, max_iter=1000, tolerance=1e-9):
    """Cold multinomial fit; ridge penalizes ALL free coefficients and intercepts."""
    if type(classes) is not int or classes != EVENT_CLASSES:
        raise ValueError('This version freezes seven joint first-event classes')
    if type(max_iter) is not int or max_iter < 1:
        raise ValueError('max_iter must be positive')
    _positive(ridge, 'ridge')
    _positive(tolerance, 'tolerance')
    matrix = _matrix(X)
    labels = _labels(y, len(matrix), classes)
    weights = _weights(len(matrix), sample_weight)
    if isinstance(X, FactorFrame):
        if feature_names is not None and tuple(feature_names) != X.feature_names:
            raise ValueError('Explicit names conflict with FactorFrame')
        if bypass_mask is not None and tuple(bypass_mask) != X.bypass_mask:
            raise ValueError('Explicit bypass conflicts with FactorFrame')
        feature_names, bypass_mask = X.feature_names, X.bypass_mask
    names = tuple(f'x{i}' for i in range(matrix.shape[1])) if feature_names is None else tuple(feature_names)
    if len(names) != matrix.shape[1] or len(set(names)) != len(names):
        raise ValueError('Feature names must align and be unique')
    scaler = TrainScaler.fit(matrix, weights, bypass_mask)
    design = np.column_stack((np.ones(len(matrix)), scaler.transform(matrix)))
    one_hot = np.eye(classes)[labels]
    shape = (design.shape[1], classes - 1)

    def objective(flat):
        parameters = flat.reshape(shape)
        logits = np.column_stack((np.zeros(len(matrix)), design @ parameters))
        log_prob = logits - logsumexp(logits, axis=1, keepdims=True)
        loss = -float(weights @ log_prob[np.arange(len(matrix)), labels])
        loss += ridge * float(np.square(parameters).sum())
        gradient = design.T @ (weights[:, None] * (np.exp(log_prob) - one_hot)[:, 1:])
        gradient += 2 * ridge * parameters
        if not np.isfinite(loss) or not np.isfinite(gradient).all():
            raise ValueError('Nonfinite multinomial objective')
        return loss, gradient.ravel()

    result = minimize(objective, np.zeros(np.prod(shape)), method='L-BFGS-B', jac=True,
                      options={'maxiter': max_iter, 'gtol': tolerance, 'ftol': 1e-14})
    if not result.success:
        raise ValueError(f'Event solver did not converge: {result.message}')
    model = EventPredictor(scaler, result.x.reshape(shape), float(ridge), names,
                           X.schema_hash if isinstance(X, FactorFrame) else None,
                           {} if provenance is None else provenance,
                           {'method': 'L-BFGS-B', 'initialization': 'zero',
                            'max_iter': max_iter, 'tolerance': float(tolerance),
                            'iterations': int(result.nit), 'evaluations': int(result.nfev),
                            'objective': float(result.fun),
                            'gradient_max_abs': float(np.abs(result.jac).max())})
    _ = model.digest
    return model


def event_nll(y, probabilities, sample_weight=None):
    p = np.asarray(probabilities, dtype=float)
    if (p.ndim != 2 or p.shape[1] != EVENT_CLASSES or not np.isfinite(p).all()
            or (p < 0).any() or (p > 1).any()
            or not np.allclose(p.sum(axis=1), 1., rtol=1e-6, atol=1e-8)):
        raise ValueError('Expected complete normalized seven-class probabilities')
    labels = _labels(y, len(p), EVENT_CLASSES)
    weights = _weights(len(p), sample_weight)
    return -float(weights @ np.log(np.maximum(p[np.arange(len(p)), labels], PROBABILITY_FLOOR)))
