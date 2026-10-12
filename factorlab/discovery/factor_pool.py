"""V7 frozen-formula F/P/S attribution with explicit common support.

F measures a fixed formula's marginal linear information; S measures the whole
fitted pipeline. Both are development estimates here, not independent evidence.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass

import numpy as np

from factorlab.protocol import fingerprint, project_path

from .causal_numpy import execute_formula, formula_identity
from .convex_eval import (
    ResidualProjector,
    RidgePredictor,
    TrainScaler,
    fit_ridge,
    log_mse,
    profile_parameters,
)
from .known_library import FactorFrame
from .rolling import RollingData, development_folds
from .v7_storage import immutable_record as immutable_json


def append_feature(frame, values, identity, name):
    if name in frame.feature_names or len(values) != len(frame.values):
        raise ValueError('Unaligned/duplicate appended factor')
    return FactorFrame(np.column_stack((frame.values, values)), frame.metadata,
                       (*frame.feature_names, name), (*frame.bypass_mask, False),
                       fingerprint({'base': frame.schema_hash, 'factor': identity, 'name': name}))


def _ridge_from_dict(value):
    if value['kind'] != 'v7_log_ridge' or value['version'] != 1 or value['target_floor'] != 1e-8:
        raise ValueError('Unsupported bundled ridge')
    return RidgePredictor(TrainScaler.from_dict(value['scaler']), np.asarray(value['coefficients']),
                          value['intercept'], value['ridge'], tuple(value['feature_names']),
                          value['input_schema_hash'], value['provenance'])


@dataclass(frozen=True)
class FactorPipeline:
    predictor: RidgePredictor
    formula: dict
    parameters: dict
    lookback: int
    projector: ResidualProjector | None
    recipe: dict
    kind = 'v7_factor_pipeline'
    task_support = ('volatility',)

    @property
    def pipeline_id(self):
        # Numerical fitted parameters/coefs belong to model digest. The frozen
        # fitting recipe is shared across future cold monthly fits.
        return fingerprint({'formula': formula_identity(self.formula, self.lookback),
                            'recipe': self.recipe, 'channel': 'F' if self.projector else 'S'})

    @property
    def schema(self):
        return {'input': formula_identity(self.formula, self.lookback)['library'],
                'fitted_input': self.predictor.schema, 'pipeline_id': self.pipeline_id}

    def feature_frame(self, frame, values):
        if self.projector:
            return self.projector.augment(frame, values, name='candidate')
        return append_feature(frame, values, self.pipeline_id, 'candidate')

    def predict(self, frame, values):
        return self.predictor.predict(self.feature_frame(frame, values))

    def to_dict(self):
        return {'kind': self.kind, 'version': 1, 'predictor': self.predictor.to_dict(),
                'formula': self.formula, 'parameters': self.parameters, 'lookback': self.lookback,
                'projector': self.projector.to_dict() if self.projector else None, 'recipe': self.recipe}

    @property
    def digest(self):
        return fingerprint(self.to_dict())

    def save(self, path):
        return immutable_json(path, {'model': self.to_dict(), 'digest': self.digest})

    @classmethod
    def load(cls, path):
        record = json.loads(project_path(path).read_text('utf-8'))
        value = record['model']
        if record['digest'] != fingerprint(value) or value['kind'] != cls.kind or value['version'] != 1:
            raise ValueError('Factor pipeline integrity/schema mismatch')
        result = cls(_ridge_from_dict(value['predictor']), value['formula'], value['parameters'], value['lookback'],
                     ResidualProjector.from_dict(value['projector']) if value['projector'] else None, value['recipe'])
        if result.digest != record['digest']:
            raise ValueError('Factor pipeline schema is not losslessly supported')
        return result


def _solve(budget):
    if budget is not None:
        budget.consume('solve')


def relative_gain(reference_loss, candidate_loss):
    return 1 - candidate_loss / reference_loss if reference_loss > 0 else None


def fit_formula(data: RollingData, cutoff, program, *, channel='F', lookback=48,
                profile=False, max_profile_calls=60, budget=None):
    """Cold fit on the mature three-month support, with bounded training-only P.

    The caller persists budget reservations before work. With profile=False,
    parameter initials are frozen, giving the fixed-formula F experiment.
    """
    if data.spec.task != 'volatility' or channel not in {'F', 'S'}:
        raise ValueError('V7 formula fitting currently isolates volatility F/S')
    program = copy.deepcopy(program)
    identity = formula_identity(program, lookback)
    theta = {k: v['initial'] for k, v in program['parameters'].items()}
    features = execute_formula(program, data.frame, lookback=lookback, parameters=theta)
    support = data.train_mask(cutoff) & np.isfinite(features)
    if support.sum() < data.spec.minimum_train_rows:
        raise ValueError('Insufficient common formula training support')
    B, y = data.frame.take(support), data.target[support]
    provenance = {**data.training_identity(cutoff), 'channel': channel,
                  'common_train_rows': int(support.sum()), 'formula': fingerprint(identity), 'profile': profile}
    recipe = {'rolling_hash': data.spec.digest, 'ridge': data.spec.ridge,
              'profile': profile, 'max_profile_calls': max_profile_calls if profile else 0,
              'channel': channel, 'fit': 'cold', 'projection': 'train_weighted_svd' if channel == 'F' else None}
    pipeline_id = fingerprint({'formula': identity, 'recipe': recipe, 'channel': channel})

    def make_frame(values):
        z = execute_formula(program, data.frame, lookback=lookback, parameters=values)[support]
        if not np.isfinite(z).all():
            raise ValueError('Parameter fit changed finite support; candidate rejected')
        if channel == 'F':
            projector = ResidualProjector.fit(B, z)
            return projector.augment(B, z, name='candidate')
        return append_feature(B, z, pipeline_id, 'candidate')

    profile_receipt = None
    if profile and theta:
        keys = sorted(theta)
        bounds = [(program['parameters'][k]['lower'], program['parameters'][k]['upper']) for k in keys]
        initial = [theta[k] for k in keys]
        starts = [initial, [lo + .25 * (hi - lo) for lo, hi in bounds],
                  [lo + .75 * (hi - lo) for lo, hi in bounds]]

        def factory(values):
            _solve(budget)
            return make_frame(dict(zip(keys, values, strict=True)))

        result = profile_parameters(factory, y, bounds=bounds, initializations=starts,
                                    max_calls=max_profile_calls, ridge=data.spec.ridge)
        theta = dict(zip(keys, result.parameters.tolist(), strict=True))
        profile_receipt = {'calls': result.calls, 'objective': result.objective, 'status': result.status}
    final_values = execute_formula(program, data.frame, lookback=lookback, parameters=theta)
    train_z = final_values[support]
    projector = ResidualProjector.fit(B, train_z) if channel == 'F' else None
    fitted_frame = (projector.augment(B, train_z, name='candidate') if projector else
                    append_feature(B, train_z, pipeline_id, 'candidate'))
    _solve(budget)
    model = fit_ridge(fitted_frame, y, ridge=data.spec.ridge, provenance=provenance)
    bundle = FactorPipeline(model, program, theta, lookback, projector, recipe)
    assert bundle.pipeline_id == pipeline_id
    return bundle, final_values, support, profile_receipt


def evaluate_formula(data, cutoff, program, *, lookback=48, budget=None, profile=False):
    """Three cold chronological folds; report F and raw-feature S separately."""
    folds = []
    for fold in development_folds(cutoff, spec=data.spec):
        base_train = data.train_mask(fold['fit_cutoff'])
        raw = execute_formula(program, data.frame, lookback=lookback)
        common_train = base_train & np.isfinite(raw)
        if common_train.sum() < data.spec.minimum_train_rows:
            raise ValueError('Insufficient development common train support')
        _solve(budget)
        base = fit_ridge(data.frame.take(common_train), data.target[common_train], ridge=data.spec.ridge)
        eligible = data.score_mask(fold['fit_cutoff'], fold['score_end'], fold['score_end'])
        score = eligible & np.isfinite(raw)
        if not score.any():
            raise ValueError('Empty mature development score support')
        B, y = data.frame.take(score), data.target[score]
        base_loss = log_mse(y, base.predict(B))
        record = {**fold, 'reference': 'K0', 'support': int(score.sum()),
                  'eligible': int(eligible.sum()), 'coverage': float(score.sum() / eligible.sum()),
                  'reference_loss': base_loss, 'channels': {}}
        for channel in ('F', 'S'):
            bundle, values, _, receipt = fit_formula(data, fold['fit_cutoff'], program,
                                                    channel=channel, lookback=lookback, budget=budget,
                                                    profile=profile and channel == 'S')
            if not np.isfinite(values[score]).all():
                raise ValueError('Fitted parameters changed score support')
            prediction = bundle.predict(B, values[score])
            loss = log_mse(y, prediction)
            record['channels'][channel] = {'loss': loss, 'gain': relative_gain(base_loss, loss),
                'absolute_delta': base_loss - loss,
                'near_zero': bundle.projector.near_zero if bundle.projector else None,
                'profile': receipt, 'pipeline_id': bundle.pipeline_id}
        folds.append(record)
    return {'role': 'S', 'reference': 'K0', 'aggregation': 'equal_fold_mean_loss',
            'formula_id': fingerprint(formula_identity(program, lookback)), 'folds': folds,
            'channels': {channel: {'loss': float(np.mean([f['channels'][channel]['loss'] for f in folds])),
                'reference_loss': float(np.mean([f['reference_loss'] for f in folds])),
                'gain': relative_gain(float(np.mean([f['reference_loss'] for f in folds])),
                                      float(np.mean([f['channels'][channel]['loss'] for f in folds])))} for channel in ('F', 'S')},
            'evidence_status': 'exposed_development_only'}


class FactorPool:
    """Bounded hypothesis archive; adding records does not change accepted A0."""
    def __init__(self):
        self.records = {}

    def add(self, program, result, *, family, lookback=48):
        identifier = fingerprint(formula_identity(program, lookback))
        if identifier in self.records:
            return False
        if len(self.records) >= 96 or result['role'] != 'S' or result['formula_id'] != identifier:
            raise ValueError('Factor pool budget/attribution mismatch')
        self.records[identifier] = {'program': copy.deepcopy(program), 'lookback': lookback,
                                    'result': copy.deepcopy(result), 'family': family}
        return True

    @property
    def digest(self):
        return fingerprint(self.records)

    def shortlist(self, limit=3):
        if limit != 3:
            raise ValueError('V7 freezes at most three validation candidates')
        # Nonlinear discoveries need not have independent linear F evidence.
        defined = [(k, v) for k, v in self.records.items() if v['result']['channels']['S']['gain'] is not None]
        ranked = sorted(defined, key=lambda p: (-p[1]['result']['channels']['S']['gain'], p[0]))
        return [(key, value) for key, value in ranked[:limit]]

    def to_dict(self):
        return {'version': 1, 'records': copy.deepcopy(self.records), 'digest': self.digest}

    @classmethod
    def from_dict(cls, value):
        result = cls()
        if value.get('version') != 1 or value.get('digest') != fingerprint(value['records']):
            raise ValueError('Factor pool integrity mismatch')
        for identifier, record in value['records'].items():
            result.add(record['program'], record['result'], family=record['family'], lookback=record['lookback'])
            if identifier not in result.records:
                raise ValueError('Factor pool formula identity mismatch')
        return result
