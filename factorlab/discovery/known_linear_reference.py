"""Frozen V7 linear controls, not a serving authorization or variance-HAR model.

HAR_cal is the existing raw K0 subset regressed on log future volatility.
Both controls retain full K0 support and fit cold on three UTC calendar months.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field

import numpy as np

from factorlab.protocol import fingerprint, project_path
from factorlab.time_utils import calendar_window

from .convex_eval import RidgePredictor, TrainScaler, fit_ridge
from .joint_pool import hour_equal_weights
from .known_library import FEATURE_NAMES, FactorFrame, library_schema
from .rolling import RollingSpec
from .v7_storage import immutable_record

HAR_COLUMNS = ('rv_1', 'rv_24', 'rv_168', 'hour_sin', 'hour_cos',
               'weekday_sin', 'weekday_cos')


@dataclass(frozen=True)
class FrozenLinearRecipe:
    name: str = 'HAR_cal'
    rolling_spec: RollingSpec = field(default_factory=RollingSpec)

    def __post_init__(self):
        if self.name not in {'HAR_cal', 'K0'}:
            raise ValueError('Unknown fixed linear benchmark')
        if (not isinstance(self.rolling_spec, RollingSpec)
                or self.rolling_spec.task != 'volatility'
                or self.rolling_spec.horizon_hours != 12):
            raise ValueError('Linear benchmark requires the 12-hour volatility protocol')

    @property
    def columns(self):
        return HAR_COLUMNS if self.name == 'HAR_cal' else FEATURE_NAMES

    def to_dict(self):
        spec = asdict(self.rolling_spec)
        spec['event_bins'] = list(spec['event_bins'])
        return {'version': 'v7-known-linear-recipe-v1', 'name': self.name,
                'rolling_spec': spec, 'columns': list(self.columns),
                'library': library_schema(), 'fit': 'cold',
                'support': 'full_raw_K0_finite', 'weighting': 'equal_symbol_then_hour',
                'target': 'log(y+1e-8)', 'decode': 'max(exp(q)-1e-8,0)'}

    @property
    def pipeline_id(self):
        return fingerprint(self.to_dict())

    @classmethod
    def from_dict(cls, value):
        spec = dict(value['rolling_spec'])
        spec['event_bins'] = tuple(spec['event_bins'])
        result = cls(value['name'], RollingSpec(**spec))
        if result.to_dict() != value:
            raise ValueError('Unsupported frozen linear recipe fields')
        return result


def _design(frame, recipe):
    expected = library_schema()
    if (not isinstance(frame, FactorFrame) or frame.feature_names != FEATURE_NAMES
            or frame.schema_hash != fingerprint(expected)
            or frame.bypass_mask != tuple(expected['bypass_mask'])):
        raise ValueError('Linear benchmark requires the exact raw K0 schema')
    indices = [FEATURE_NAMES.index(name) for name in recipe.columns]
    policy = fingerprint({'library': frame.schema_hash, 'columns': recipe.columns})
    return FactorFrame(frame.values[:, indices], frame.metadata, recipe.columns,
                       tuple(frame.bypass_mask[i] for i in indices), policy)


def _ridge(value):
    model = RidgePredictor(TrainScaler.from_dict(value['scaler']), value['coefficients'],
        value['intercept'], value['ridge'], tuple(value['feature_names']),
        value['input_schema_hash'], value['provenance'])
    n = len(model.feature_names)
    if (model.to_dict() != value or model.coefficients.shape != (n,)
            or len(model.scaler.center) != n or len(set(model.feature_names)) != n
            or not np.isfinite(model.coefficients).all()
            or not np.isfinite(model.intercept) or not np.isfinite(model.ridge)
            or model.ridge <= 0):
        raise ValueError('Invalid frozen linear ridge state')
    return model


@dataclass(frozen=True)
class KnownLinearPredictor:
    recipe: FrozenLinearRecipe
    predictor: RidgePredictor
    receipt: dict
    kind: str = field(default='v7_known_linear', init=False)
    task_support: tuple[str, ...] = field(default=('volatility',), init=False)
    _sealed_digest: str = field(default='', init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, 'predictor', _ridge(self.predictor.to_dict()))
        object.__setattr__(self, 'receipt', copy.deepcopy(self.receipt))
        receipt, recipe = self.receipt, self.recipe
        expected_schema = fingerprint({'library': fingerprint(library_schema()),
                                       'columns': recipe.columns})
        bypass = tuple(library_schema()['bypass_mask'][FEATURE_NAMES.index(n)] for n in recipe.columns)
        if (self.predictor.feature_names != recipe.columns
                or self.predictor.scaler.bypass_mask != bypass
                or self.predictor.input_schema_hash != expected_schema
                or self.predictor.ridge != recipe.rolling_spec.ridge):
            raise ValueError('Ridge differs from frozen linear recipe')
        if set(receipt) != {'version', 'pipeline_id', 'training', 'weighting', 'solve_count'}:
            raise ValueError('Unsupported linear training receipt')
        training = receipt['training']
        start, end = recipe.rolling_spec.window(training['cutoff'])
        if (receipt['version'] != 'v7-known-linear-fit-v1'
                or receipt['pipeline_id'] != recipe.pipeline_id
                or receipt['weighting'] != 'equal_symbol_then_hour'
                or type(receipt['solve_count']) is not int or receipt['solve_count'] != 1
                or training['rolling_hash'] != recipe.rolling_spec.digest
                or training['fit'] != 'cold' or training['train_start'] != start.isoformat()
                or training['cutoff'] != end.isoformat()
                or type(training['rows']) is not int
                or training['rows'] < recipe.rolling_spec.minimum_train_rows
                or self.predictor.provenance != receipt):
            raise ValueError('Linear training receipt differs from recipe or ridge')
        object.__setattr__(self, '_sealed_digest', fingerprint(self.to_dict()))

    def _assert_integrity(self):
        if fingerprint(self.to_dict()) != self._sealed_digest:
            raise ValueError('Frozen linear model was mutated')

    @property
    def pipeline_id(self):
        self._assert_integrity()
        return self.recipe.pipeline_id

    @property
    def schema(self):
        return {'input': library_schema(), 'recipe': self.recipe.to_dict(),
                'readout': self.predictor.schema, 'pipeline_id': self.pipeline_id}

    @property
    def schema_hash(self):
        return fingerprint(self.schema)

    def predict_history(self, frame):
        self._assert_integrity()
        design = _design(frame, self.recipe)
        mask = frame.valid_rows
        predictions = np.full(len(mask), np.nan)
        if mask.any():
            predictions[mask] = self.predictor.predict(design.take(mask))
        return predictions, mask

    def predict(self, frame):
        return self.predict_history(frame)[0]

    def to_dict(self):
        return {'kind': self.kind, 'version': 1, 'recipe': self.recipe.to_dict(),
                'predictor': self.predictor.to_dict(), 'receipt': copy.deepcopy(self.receipt)}

    @property
    def digest(self):
        self._assert_integrity()
        return self._sealed_digest

    def save(self, path):
        return immutable_record(path, {'model': self.to_dict(), 'digest': self.digest})

    @classmethod
    def from_dict(cls, value):
        result = cls(FrozenLinearRecipe.from_dict(value['recipe']),
                     _ridge(value['predictor']), value['receipt'])
        if result.to_dict() != value:
            raise ValueError('Unsupported linear model schema')
        return result

    @classmethod
    def load(cls, path):
        record = json.loads(project_path(path).read_text('utf-8'))
        if set(record) != {'model', 'digest'} or fingerprint(record['model']) != record['digest']:
            raise ValueError('Linear model integrity mismatch')
        return cls.from_dict(record['model'])


def fit_known_linear(data, cutoff, recipe, *, budget):
    """One charged cold solve; receipt proves no serving or statistical approval."""
    if budget is None or not callable(getattr(budget, 'consume', None)):
        raise ValueError('Explicit solve budget is required')
    if not isinstance(recipe, FrozenLinearRecipe) or data.spec != recipe.rolling_spec:
        raise ValueError('Data protocol differs from frozen linear recipe')
    design = _design(data.frame, recipe)
    training = data.training_identity(cutoff)
    mask = data.train_mask(cutoff)
    weights = hour_equal_weights(design.metadata.loc[mask])
    receipt = {'version': 'v7-known-linear-fit-v1', 'pipeline_id': recipe.pipeline_id,
               'training': training, 'weighting': 'equal_symbol_then_hour', 'solve_count': 1}
    budget.consume('solve')
    model = fit_ridge(design.take(mask), data.target[mask], ridge=recipe.rolling_spec.ridge,
                      sample_weight=weights, provenance=receipt)
    return KnownLinearPredictor(recipe, model, receipt)
