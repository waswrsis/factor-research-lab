"""Frozen multi-formula pools, cold fitted systems and development attribution.

Every executable member uses the raw causal K0 coordinate contract. Pool
selection is development-only; this module never mutates a serving A0 or a
frozen confirmation pool and never labels development scores as confirmation.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from factorlab.protocol import fingerprint, project_path
from factorlab.time_utils import hour_ids

from .causal_numpy import execute_formula, formula_identity, _ScalarProfileCache
from .convex_eval import (
    ResidualProjector,
    RidgePredictor,
    TrainScaler,
    fit_ridge,
    log_mse,
    profile_parameters,
)
from .known_library import FactorFrame, library_schema
from .rolling import development_folds
from .v7_storage import immutable_record

MAX_MEMBERS = 8
MAX_ARCHIVE = 96


def _receipt(stage, **facts):
    body = {'stage': stage, 'version': 1, **facts}
    return {**body, 'receipt_id': fingerprint(body)}


def _aware(value):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None or pd.isna(stamp):
        raise ValueError('Pool evidence timestamps must be aware and nonmissing')
    return stamp.tz_convert('UTC')


def _month(value):
    stamp = _aware(value)
    if stamp.day != 1 or stamp != stamp.normalize():
        raise ValueError('Pool month must be a UTC calendar-month boundary')
    return stamp.isoformat()


def hour_equal_weights(metadata):
    """One row per symbol/hour; symbols equal inside hours, then hours equal."""
    if not {'available_at', 'symbol'} <= set(metadata):
        raise ValueError('Pool scores require symbol and origin metadata')
    hours = hour_ids(metadata.available_at)
    symbols = metadata.symbol.tolist()
    if len(set(zip(hours.tolist(), symbols, strict=True))) != len(hours):
        raise ValueError('Resolve duplicate symbol/hour origins before fitting or scoring')
    _, inverse, counts = np.unique(hours, return_inverse=True, return_counts=True)
    return 1. / counts[inverse]


@dataclass(frozen=True)
class PoolMember:
    program: dict
    lookback: int = 48
    parameter_mode: str = 'fixed'
    parameters: dict | None = None
    family: str = 'local'
    parent_ids: tuple[str, ...] = ()

    def __post_init__(self):
        program = copy.deepcopy(self.program)
        formula_identity(program, self.lookback)
        if self.parameter_mode not in {'fixed', 'profile'}:
            raise ValueError('Pool parameter mode must be fixed or profile')
        specs = program.get('parameters', {})
        initial = {k: v['initial'] for k, v in specs.items()}
        values = initial if self.parameters is None else dict(self.parameters)
        if set(values) != set(specs):
            raise ValueError('Pool member parameter names differ from formula contract')
        for name, spec in specs.items():
            lo, hi, value = spec['lower'], spec['upper'], values[name]
            if (not all(np.isfinite(v) for v in (lo, hi, value)) or not lo < hi
                    or not lo <= value <= hi):
                raise ValueError('Pool parameters require finite ordered bounds and valid initials')
        if not isinstance(self.family, str) or not self.family or len(self.parent_ids) > 8:
            raise ValueError('Invalid pool member family/parent lineage')
        object.__setattr__(self, 'program', program)
        object.__setattr__(self, 'parameters', values)
        object.__setattr__(self, 'parent_ids', tuple(self.parent_ids))
        fingerprint(self.to_dict())

    @property
    def member_id(self):
        # Mechanism provenance does not turn identical executable formulas into
        # different members, nor let repeated aliases split their L2 penalty.
        return fingerprint({'formula': formula_identity(self.program, self.lookback),
                            'parameters': self.parameters, 'parameter_mode': self.parameter_mode})

    @property
    def structure_id(self):
        return fingerprint(formula_identity(self.program, self.lookback))

    def to_dict(self):
        return {'program': copy.deepcopy(self.program), 'lookback': self.lookback,
                'parameter_mode': self.parameter_mode, 'parameters': dict(self.parameters),
                'family': self.family, 'parent_ids': list(self.parent_ids)}

    @classmethod
    def from_dict(cls, value):
        return cls(**copy.deepcopy(value))


@dataclass(frozen=True)
class FrozenPool:
    members: tuple[PoolMember, ...] = ()
    version: str = 'P_dev_0'
    reference_pipeline_id: str = 'K0_v1'
    mode: str = 'raw'
    max_profile_calls: int = 60
    role: str = 'P_dev'
    reference_pool: FrozenPool | None = None
    access_modes: tuple[str, ...] = ()

    def __post_init__(self):
        members = tuple(self.members)
        if (len(members) > MAX_MEMBERS or any(not isinstance(m, PoolMember) for m in members)
                or len({m.member_id for m in members}) != len(members)):
            raise ValueError('Pool requires at most eight unique executable members')
        if (self.mode not in {'raw', 'residual'} or self.role not in {'P_dev', 'A0', 'confirmation', 'reference'}
                or type(self.max_profile_calls) is not int or not 1 <= self.max_profile_calls <= 60
                or not isinstance(self.version, str) or not self.version
                or not isinstance(self.reference_pipeline_id, str) or not self.reference_pipeline_id):
            raise ValueError('Invalid frozen pool identity or fitting policy')
        object.__setattr__(self, 'members', members)
        access = tuple(self.access_modes)
        if access and (len(access) != len(members) or any(m not in {'raw', 'residual'} for m in access)):
            raise ValueError('Explicit member access modes must align with the pool')
        object.__setattr__(self, 'access_modes', access)
        if self.reference_pool is not None:
            ref = self.reference_pool
            if (not isinstance(ref, FrozenPool) or len(ref.members) + 1 != len(members)
                    or tuple(m.member_id for m in ref.members) != tuple(m.member_id for m in members[:-1])
                    or self.reference_pipeline_id != ref.pipeline_id
                    or (access and access[:-1] != tuple(ref.member_access(i) for i in range(len(ref.members))))):
                raise ValueError('Conditional pool must freeze its exact prefix reference pipeline')

    def to_dict(self):
        return {'version': self.version, 'members': [m.to_dict() for m in self.members],
                'reference_pipeline_id': self.reference_pipeline_id, 'mode': self.mode,
                'max_profile_calls': self.max_profile_calls, 'role': self.role,
                'reference_pool': self.reference_pool.to_dict() if self.reference_pool else None,
                'access_modes': list(self.access_modes)}

    @classmethod
    def from_dict(cls, value):
        value = copy.deepcopy(value)
        value['members'] = tuple(PoolMember.from_dict(m) for m in value['members'])
        if value.get('reference_pool') is not None:
            value['reference_pool'] = cls.from_dict(value['reference_pool'])
        return cls(**value)

    @property
    def digest(self):
        return fingerprint(self.to_dict())

    @property
    def pipeline_id(self):
        return fingerprint({'version': 'v7_joint_pool_v1', 'pool': self.to_dict(),
                            'library': library_schema(), 'fit': 'cold_hour_equal_ridge',
                            'transform': 'ordered_scalar_residuals' if self.mode == 'residual' else 'raw'})

    def member_access(self, index):
        if self.access_modes:
            return self.access_modes[index]
        if self.reference_pool is not None and index < len(self.reference_pool.members):
            return self.reference_pool.member_access(index)
        return self.mode


def _append(frame, values, member, parameters):
    name = 'factor_' + member.member_id[:16]
    return FactorFrame(np.column_stack((frame.values, values)), frame.metadata,
                       (*frame.feature_names, name), (*frame.bypass_mask, False),
                       fingerprint({'base': frame.schema_hash, 'member': member.member_id,
                                    'parameters': parameters, 'mode': 'raw'}))


def _vectors(frame, pool, parameters, *, scalar_cache=None):
    return [execute_formula(m.program, frame, lookback=m.lookback, parameters=p, _scalar_cache=scalar_cache)
            for m, p in zip(pool.members, parameters, strict=True)]


def _valid(frame, vectors):
    valid = frame.valid_rows.copy()
    for vector in vectors:
        valid &= np.isfinite(vector)
    return valid


def _design(frame, vectors, pool, parameters, projectors=None, weights=None):
    result, projections = frame, []
    for i, (member, vector, theta) in enumerate(zip(pool.members, vectors, parameters, strict=True)):
        if pool.member_access(i) == 'residual':
            projector = (ResidualProjector.fit(result, vector, sample_weight=weights)
                         if projectors is None else projectors[i])
            result = projector.augment(result, vector, name='factor_' + member.member_id[:16])
        else:
            projector = None
            result = _append(result, vector, member, theta)
        projections.append(projector)
    return result, tuple(projections)


def _ridge_from_dict(value):
    if value.get('kind') != 'v7_log_ridge' or value.get('version') != 1 or value.get('target_floor') != 1e-8:
        raise ValueError('Unsupported bundled joint ridge')
    result = RidgePredictor(TrainScaler.from_dict(value['scaler']), np.asarray(value['coefficients']),
                            value['intercept'], value['ridge'], tuple(value['feature_names']),
                            value['input_schema_hash'], value['provenance'])
    if fingerprint(result.to_dict()) != fingerprint(value):
        raise ValueError('Unsupported bundled ridge fields')
    return result


@dataclass(frozen=True)
class JointPredictor:
    pool: FrozenPool
    parameters: tuple[dict, ...]
    projectors: tuple[ResidualProjector | None, ...]
    predictor: RidgePredictor
    receipt: dict
    kind: str = field(default='v7_joint_pool', init=False)
    task_support: tuple[str, ...] = field(default=('volatility',), init=False)
    _sealed_digest: str = field(default='', init=False, repr=False)

    def __post_init__(self):
        if len(self.parameters) != len(self.pool.members) or len(self.projectors) != len(self.pool.members):
            raise ValueError('Joint predictor member state is unaligned')
        object.__setattr__(self, 'pool', FrozenPool.from_dict(self.pool.to_dict()))
        object.__setattr__(self, 'parameters', tuple(copy.deepcopy(p) for p in self.parameters))
        object.__setattr__(self, 'receipt', copy.deepcopy(self.receipt))
        if self.receipt.get('pool_hash') != self.pool.digest:
            raise ValueError('Joint predictor pool differs from its training receipt')
        object.__setattr__(self, '_sealed_digest', fingerprint(self.to_dict()))

    def _assert_integrity(self):
        if fingerprint(self.to_dict()) != self._sealed_digest:
            raise ValueError('Frozen joint predictor state changed after fitting')

    @property
    def pipeline_id(self):
        self._assert_integrity()
        return fingerprint({'pool_template': self.pool.pipeline_id,
                            'rolling_hash': self.receipt['training']['rolling_hash'],
                            'ridge': self.predictor.ridge,
                            'weighting': 'equal_symbol_then_hour'})

    @property
    def schema(self):
        return {'input': library_schema(), 'readout': self.predictor.schema,
                'pool_hash': self.pool.digest, 'pipeline_id': self.pipeline_id,
                'reference_pipeline_id': (self.receipt['frozen_reference_fit']['pipeline_id']
                                          if self.receipt.get('frozen_reference_fit') else self.pool.reference_pipeline_id),
                'reference_pipeline_template_id': self.pool.reference_pipeline_id,
                'members': [m.member_id for m in self.pool.members]}

    def feature_history(self, frame, *, _scalar_cache=None):
        """Aligned fixed-pool inputs for same-information nonlinear comparisons."""
        self._assert_integrity()
        if frame.schema_hash != fingerprint(library_schema()):
            raise ValueError('Joint history must use the frozen raw K0 coordinate system')
        vectors = _vectors(frame, self.pool, self.parameters, scalar_cache=_scalar_cache)
        mask = _valid(frame, vectors)
        values = np.full((len(frame.values), len(self.predictor.feature_names)), np.nan)
        if mask.any():
            design, _ = _design(frame.take(mask), [v[mask] for v in vectors], self.pool,
                                self.parameters, self.projectors)
            values[mask] = design.values
        aligned = FactorFrame(values, frame.metadata, self.predictor.feature_names,
                              self.predictor.scaler.bypass_mask, self.predictor.input_schema_hash)
        return aligned, mask

    def predict_history(self, frame):
        design, mask = self.feature_history(frame)
        predictions = np.full(len(frame.values), np.nan)
        if mask.any():
            predictions[mask] = self.predictor.predict(design.take(mask))
        return predictions, mask

    def predict(self, frame):
        """Aligned historical prediction; unsupported warmup rows remain NaN."""
        return self.predict_history(frame)[0]

    def to_dict(self):
        return {'kind': self.kind, 'version': 1, 'pool': self.pool.to_dict(),
                'parameters': list(copy.deepcopy(self.parameters)),
                'projectors': [p.to_dict() if p is not None else None for p in self.projectors],
                'predictor': self.predictor.to_dict(), 'receipt': copy.deepcopy(self.receipt)}

    @property
    def digest(self):
        self._assert_integrity()
        return self._sealed_digest

    def save(self, path):
        return immutable_record(path, {'model': self.to_dict(), 'digest': self.digest})

    @classmethod
    def load(cls, path):
        record = json.loads(project_path(path).read_text('utf-8'))
        value = record['model']
        if record.get('digest') != fingerprint(value) or value.get('kind') != 'v7_joint_pool' or value.get('version') != 1:
            raise ValueError('Joint predictor integrity or version mismatch')
        result = cls(FrozenPool.from_dict(value['pool']), tuple(value['parameters']),
                     tuple(ResidualProjector.from_dict(p) if p is not None else None for p in value['projectors']),
                     _ridge_from_dict(value['predictor']), value['receipt'])
        if result.digest != record['digest']:
            raise ValueError('Joint predictor round-trip changes identity')
        return result


def _charge(budget):
    if budget is not None:
        budget.consume('solve')


class _ReservationAbort(Exception):
    """Carry budget failures through optimizers that treat ValueError as bad theta."""


def _fit_pool(data, cutoff, pool, budget, *, support=None, parameters=None, profile=True,
              reference_model=None, scalar_cache=None):
    if data.spec.task != 'volatility':
        raise ValueError('Joint pool isolates the volatility task')
    theta = [dict(m.parameters) for m in pool.members] if parameters is None else copy.deepcopy(parameters)
    if scalar_cache is None and profile:
        scalar_cache = _ScalarProfileCache()
    vectors = _vectors(data.frame, pool, theta, scalar_cache=scalar_cache)
    available = data.train_mask(cutoff) & _valid(data.frame, vectors)
    if support is not None:
        support = np.asarray(support, dtype=bool)
        if support.shape != available.shape or (support & ~available).any():
            raise ValueError('Requested common training support is not mature and finite')
        available = support
    if available.sum() < data.spec.minimum_train_rows:
        raise ValueError('Insufficient common mature pool training support')
    training_identity = data.training_identity(cutoff)
    frame, target = data.frame.take(available), data.target[available]
    weights = hour_equal_weights(frame.metadata)
    calls, profile_receipt = 0, None
    prefix_receipt = None
    prefix_count = 0
    if pool.reference_pool is not None:
        if reference_model is None:
            reference_model, _ = _fit_pool(data, cutoff, pool.reference_pool, budget,
                                           support=available, profile=profile, scalar_cache=scalar_cache)
            calls += reference_model.receipt['solve_calls']
        elif (reference_model.pool.digest != pool.reference_pool.digest
                or reference_model.receipt['training']['common_support_hash'] != fingerprint(np.flatnonzero(available).tolist())
                or reference_model.receipt['cutoff'] != _aware(cutoff).isoformat()
                or any(reference_model.receipt['training'].get(key) != value
                       for key, value in training_identity.items())):
            raise ValueError('Cached reference fit differs from conditional common support or cutoff')
        prefix_count = len(pool.reference_pool.members)
        theta[:prefix_count] = copy.deepcopy(reference_model.parameters)
        prefix_receipt = {'model_digest': reference_model.digest, 'pipeline_id': reference_model.pipeline_id,
                          'receipt_id': reference_model.receipt['receipt_id']}
        vectors = _vectors(data.frame, pool, theta, scalar_cache=scalar_cache)
    profile_keys = [(i, name) for i, m in enumerate(pool.members)
                    if i >= prefix_count and m.parameter_mode == 'profile'
                    for name in sorted(m.parameters)] if profile else []
    if profile_keys:
        bounds = [(pool.members[i].program['parameters'][name]['lower'],
                   pool.members[i].program['parameters'][name]['upper']) for i, name in profile_keys]
        starts = [[theta[i][name] for i, name in profile_keys],
                  [lo + .25 * (hi - lo) for lo, hi in bounds],
                  [lo + .75 * (hi - lo) for lo, hi in bounds]]

        def factory(values):
            nonlocal calls
            try:
                _charge(budget)
            except Exception as exc:
                raise _ReservationAbort(exc) from exc
            calls += 1
            proposal = copy.deepcopy(theta)
            for (i, name), value in zip(profile_keys, values, strict=True):
                proposal[i][name] = float(value)
            features = _vectors(data.frame, pool, proposal, scalar_cache=scalar_cache)
            if any(not np.isfinite(v[available]).all() for v in features):
                raise ValueError('Profile changed registered finite training support')
            design, _ = _design(frame, [v[available] for v in features], pool, proposal, weights=weights)
            return design

        try:
            result = profile_parameters(factory, target, bounds=bounds, initializations=starts,
                                        max_calls=pool.max_profile_calls, ridge=data.spec.ridge,
                                        sample_weight=weights)
        except _ReservationAbort as exc:
            raise exc.args[0] from exc
        for (i, name), value in zip(profile_keys, result.parameters, strict=True):
            theta[i][name] = float(value)
        profile_receipt = _receipt('P_training_profile', calls=result.calls,
                                   parameters=copy.deepcopy(theta), objective=result.objective,
                                   status=result.status, history=list(result.history),
                                   access=pool.mode, support_hash=fingerprint(np.flatnonzero(available).tolist()))
        vectors = _vectors(data.frame, pool, theta, scalar_cache=scalar_cache)
    design, projections = _design(frame, [v[available] for v in vectors], pool, theta, weights=weights)
    provenance = {**training_identity, 'pool_hash': pool.digest,
                  'common_train_rows': int(available.sum()),
                  'common_support_hash': fingerprint(np.flatnonzero(available).tolist()),
                  'aggregation': 'equal_symbol_within_hour_then_equal_hour'}
    _charge(budget)
    calls += 1
    model = fit_ridge(design, target, ridge=data.spec.ridge, sample_weight=weights, provenance=provenance)
    bound_pipeline = fingerprint({'pool_template': pool.pipeline_id,
                                  'rolling_hash': provenance['rolling_hash'], 'ridge': model.ridge,
                                  'weighting': 'equal_symbol_then_hour'})
    receipt = _receipt('cold_pool_fit', pool_hash=pool.digest, pipeline_id=bound_pipeline,
                       cutoff=_aware(cutoff).isoformat(), solve_calls=calls, training=provenance,
                       parameter_profile=profile_receipt, frozen_reference_fit=prefix_receipt,
                       near_zero=[p.near_zero if p is not None else None for p in projections])
    return JointPredictor(pool, tuple(theta), projections, model, receipt), available


def fit_pool(data, cutoff, pool: FrozenPool, budget=None) -> JointPredictor:
    """Cold monthly fit. Profiling uses only mature three-month training labels."""
    return _fit_pool(data, cutoff, pool, budget)[0]


def _conditional_pool(reference, member, *, mode, version, max_profile_calls=60):
    return FrozenPool((*reference.members, member), version, reference.pipeline_id, mode,
                       max_profile_calls, 'P_dev', reference)


def _loss(data, predictions, mask):
    weights = hour_equal_weights(data.frame.metadata.loc[mask])
    return log_mse(data.target[mask], predictions[mask], weights)


def _comparison(reference, candidate):
    return {'reference_loss': float(reference), 'candidate_loss': float(candidate),
            'absolute_delta': float(reference - candidate),
            'gain': float(1 - candidate / reference) if reference > 0 else None}


def frame_alignment_id(frame):
    from .frame_identity import frame_identity
    columns = ['symbol', 'available_at', 'feature_open_time']
    return fingerprint({'rule': 'typed-frame-columns-v1', 'columns': frame_identity(frame.metadata[columns])})


def evaluate_candidate(data, cutoff, pool: FrozenPool, candidate: PoolMember, *, budget=None,
                        old_root=None, reserve_candidate=False, context=None):
    """Three-fold F/P/S; one candidate parameter profile per fold, at most 60 calls.

    F separately measures fixed theta0 residual information against K0 and Pv.
    P compares theta0 versus fitted theta under the SAME raw conditional-pool
    access; reference pool theta/transform are frozen to its same-support fit.
    S compares the full fitted raw pool with K0, Pv, and optional archived root
    predictions. F->S differences are never reported as a parameter-fit effect.

    The controller normally reserves one F slot before calling; alternatively
    reserve_candidate=True charges it here. Every ridge solve is charged here.
    old_root is an optional per-fit-cutoff map of aligned immutable predictions,
    with fit_cutoff, pipeline_id, alignment_id and artifact_sha256 evidence.
    Persistent evidence requires a predeclared archive context. Its external
    data/source/protocol hashes belong to the controller's frozen manifest;
    this evaluator binds them before work, never permits later relabelling.
    """
    if context is not None:
        context = copy.deepcopy(context)
        expected = DevelopmentPoolArchive.context(pool, cutoff=cutoff, data_hash=context['data_hash'],
                                                  protocol_hash=context['protocol_hash'], source_hash=context['source_hash'])
        if context != expected:
            raise ValueError('Evaluation context differs from the frozen pool and cutoff')
    if reserve_candidate and budget is not None:
        budget.consume('f', family_id=candidate.family)
    if len(pool.members) >= MAX_MEMBERS:
        raise ValueError('Pool capacity reached; predeclared pruning/replacement must precede admission')
    if candidate.structure_id in {m.structure_id for m in pool.members}:
        raise ValueError('Exact existing formula is not a new pool candidate')
    if candidate.parameters and candidate.parameter_mode != 'profile':
        raise ValueError('Parameterized F/P/S candidates must declare profile admission policy')
    fixed = replace(candidate, parameter_mode='fixed')
    optimized = replace(candidate, parameter_mode='profile')
    anchor = FrozenPool(version='K0_v1', reference_pipeline_id='K0_v1', role='reference')
    f0_pool = _conditional_pool(anchor, fixed, mode='residual', version=pool.version + ':F_K0')
    fp_pool = _conditional_pool(pool, fixed, mode='residual', version=pool.version + ':F_Pv')
    raw_pool = _conditional_pool(pool, fixed, mode='raw', version=pool.version + ':theta0')
    p_pool = _conditional_pool(pool, optimized, mode='raw', version=pool.version + ':theta_hat',
                               max_profile_calls=pool.max_profile_calls)
    # One bounded X-only cache for this invocation; every model/solve is still cold.
    scalar_cache = _ScalarProfileCache()
    initial_vectors = _vectors(data.frame, pool, [m.parameters for m in pool.members],
                               scalar_cache=scalar_cache)
    initial_vectors.append(execute_formula(candidate.program, data.frame, lookback=candidate.lookback,
                                           parameters=candidate.parameters, _scalar_cache=scalar_cache))
    finite = _valid(data.frame, initial_vectors)
    folds = []
    alignment_id = frame_alignment_id(data.frame)
    for fold in development_folds(cutoff, spec=data.spec):
        train = data.train_mask(fold['fit_cutoff']) & finite
        score = data.score_mask(fold['fit_cutoff'], fold['score_end'], fold['score_end']) & finite
        if train.sum() < data.spec.minimum_train_rows or not score.any():
            raise ValueError('Insufficient common candidate/reference development support')
        k0, _ = _fit_pool(data, fold['fit_cutoff'], anchor, budget, support=train, scalar_cache=scalar_cache)
        reference, _ = _fit_pool(data, fold['fit_cutoff'], pool, budget, support=train, scalar_cache=scalar_cache)
        fitted_f0, _ = _fit_pool(data, fold['fit_cutoff'], f0_pool, budget, support=train, reference_model=k0, scalar_cache=scalar_cache)
        fitted_fp, _ = _fit_pool(data, fold['fit_cutoff'], fp_pool, budget, support=train, reference_model=reference, scalar_cache=scalar_cache)
        raw, _ = _fit_pool(data, fold['fit_cutoff'], raw_pool, budget, support=train, reference_model=reference, scalar_cache=scalar_cache)
        if candidate.parameters:
            fitted_p, _ = _fit_pool(data, fold['fit_cutoff'], p_pool, budget, support=train, reference_model=reference, scalar_cache=scalar_cache)
        else:
            fitted_p = raw
        models = {'K0': k0, 'Pv': reference, 'F_K0': fitted_f0, 'F_Pv': fitted_fp,
                  'theta0_raw': raw, 'theta_hat_raw': fitted_p}
        predictions, losses, diagnostics = {}, {}, {}
        for name, model in models.items():
            features, mask = model.feature_history(data.frame, _scalar_cache=scalar_cache)
            if (score & ~mask).any():
                raise ValueError('Fitted parameter or pipeline changed common scoring support')
            pred = np.full(len(mask), np.nan)
            pred[score], diagnostics[name] = model.predictor.predict_with_diagnostics(features.take(score))
            predictions[name] = pred
            losses[name] = _loss(data, pred, score)
        old_comparison = {'status': 'not_recomputed', 'comparison': None}
        supplied = None if old_root is None else old_root.get(fold['fit_cutoff'])
        if supplied is not None:
            if (supplied.get('fit_cutoff') != fold['fit_cutoff']
                    or supplied.get('alignment_id') != alignment_id
                    or not supplied.get('pipeline_id') or len(supplied.get('artifact_sha256', '')) != 64):
                raise ValueError('Old root predictions lack the matching frozen origin/fit identity')
            previous = np.asarray(supplied['predictions'], dtype=float)
            previous_mask = np.asarray(supplied['mask'], dtype=bool)
            if previous.shape != data.target.shape or previous_mask.shape != score.shape:
                raise ValueError('Old root predictions are not sample-aligned')
            paired = score & previous_mask
            if not paired.any() or not np.isfinite(previous[paired]).all() or (previous[paired] < 0).any():
                raise ValueError('Old root has no valid common nonnegative support')
            old_comparison = {'status': 'paired_archived_prediction_comparison',
                              'pipeline_id': supplied['pipeline_id'],
                              'artifact_sha256': supplied['artifact_sha256'],
                              'support': int(paired.sum()),
                              'comparison': _comparison(_loss(data, previous, paired),
                                                        _loss(data, predictions['theta_hat_raw'], paired))}
        hours = len(np.unique(hour_ids(data.frame.metadata.loc[score, 'available_at'])))
        record = {**fold, 'support_rows': int(score.sum()), 'support_hours': hours,
                  'common_support_hash': fingerprint(np.flatnonzero(score).tolist()),
                  'train_support_hash': fingerprint(np.flatnonzero(train).tolist()),
                  'F': {'K0': _comparison(losses['K0'], losses['F_K0']),
                        'Pv': _comparison(losses['Pv'], losses['F_Pv']),
                        'theta': dict(fixed.parameters),
                        'near_zero_K0': fitted_f0.projectors[-1].near_zero,
                        'near_zero_Pv': fitted_fp.projectors[-1].near_zero},
                  'P': {**_comparison(losses['theta0_raw'], losses['theta_hat_raw']),
                        'access': 'same_raw_conditional_pool', 'theta0': dict(fixed.parameters),
                        'theta_hat': dict(fitted_p.parameters[-1]),
                        'reference_theta_unchanged': True,
                        'optimized': bool(candidate.parameters)},
                  'S': {'K0': _comparison(losses['K0'], losses['theta_hat_raw']),
                        'Pv': _comparison(losses['Pv'], losses['theta_hat_raw']),
                        'old_root': old_comparison},
                  'models': {name: {'digest': model.digest, 'pipeline_id': model.pipeline_id,
                                    'scoring_decode': diagnostics[name],
                                    'receipt': model.receipt} for name, model in models.items()}}
        folds.append(record)
    weights = np.asarray([f['support_hours'] for f in folds], dtype=float)

    def aggregate(branch, reference=None):
        entries = [f[branch][reference] if reference else f[branch] for f in folds]
        ref = float(np.average([e['reference_loss'] for e in entries], weights=weights))
        cand = float(np.average([e['candidate_loss'] for e in entries], weights=weights))
        return {**_comparison(ref, cand),
                'winning_fold_fraction': float(np.mean([e['absolute_delta'] > 0 for e in entries]))}

    return {'role': 'S', 'evidence_status': 'exposed_development_only',
            'context': context,
            'candidate_id': candidate.member_id, 'pool_hash': pool.digest,
            'reference_pipeline_id': folds[0]['models']['Pv']['pipeline_id'],
            'reference_pipeline_template_id': pool.pipeline_id,
            'aggregation': 'equal_symbol_within_hour_then_equal_hour_across_folds',
            'F': {name: aggregate('F', name) for name in ('K0', 'Pv')},
            'P': {**aggregate('P'), 'access': 'same_raw_conditional_pool'},
            'S': {name: aggregate('S', name) for name in ('K0', 'Pv')},
            'old_root': [f['S']['old_root'] for f in folds], 'folds': folds,
            'receipt': _receipt('F_P_S_three_fold_evaluation', pool_hash=pool.digest,
                                candidate_id=candidate.member_id, cutoff=_aware(cutoff).isoformat(),
                                f_slot='charged_here' if reserve_candidate else 'caller_reserved',
                                fit_receipts=[m['receipt']['receipt_id'] for f in folds
                                              for m in f['models'].values()])}


def prescreen_candidates(data, cutoff, pool, candidates, *, limit=24, blind_spot_slots=6):
    """Training-X-only conditional novelty scheduling, never target screening.

    Lower linear novelty may hide interactions. A deterministic hash sample of
    the remaining candidates therefore retains blind-spot slots; all deferred
    candidates stay in the archive. No V/C labels or scores enter this function.
    """
    candidates = tuple(candidates)
    if (len(candidates) > MAX_ARCHIVE or len({c.member_id for c in candidates}) != len(candidates)
            or type(limit) is not int or not 1 <= limit <= 24
            or type(blind_spot_slots) is not int or not 0 <= blind_spot_slots <= limit):
        raise ValueError('Invalid bounded prescreen candidate/slot contract')
    reference_vectors = _vectors(data.frame, pool, [m.parameters for m in pool.members])
    train = data.train_mask(cutoff) & _valid(data.frame, reference_vectors)
    entries = []
    for candidate in candidates:
        z = execute_formula(candidate.program, data.frame, lookback=candidate.lookback,
                            parameters=candidate.parameters)
        support = train & np.isfinite(z)
        if support.sum() < data.spec.minimum_train_rows:
            entries.append({'candidate_id': candidate.member_id, 'status': 'insufficient_support',
                            'novelty_ratio': None})
            continue
        B, _ = _design(data.frame.take(support), [v[support] for v in reference_vectors], pool,
                       [m.parameters for m in pool.members], weights=hour_equal_weights(data.frame.metadata.loc[support]))
        projection = ResidualProjector.fit(B, z[support], sample_weight=hour_equal_weights(B.metadata))
        entries.append({'candidate_id': candidate.member_id, 'status': 'scored_X_only',
                        'novelty_ratio': projection.residual_scale_ratio,
                        'near_zero': projection.near_zero, 'support': int(support.sum())})
    valid = [e for e in entries if e['novelty_ratio'] is not None]
    ranked = sorted(valid, key=lambda e: (-e['novelty_ratio'], e['candidate_id']))
    leading = ranked[:max(0, limit - blind_spot_slots)]
    used = {e['candidate_id'] for e in leading}
    remaining = sorted((e for e in valid if e['candidate_id'] not in used),
                       key=lambda e: fingerprint({'candidate': e['candidate_id'], 'pool': pool.digest,
                                                   'cutoff': _aware(cutoff).isoformat(), 'purpose': 'interaction_blind_spot'}))
    selected = leading + remaining[:limit - len(leading)]
    return {'selected_ids': [e['candidate_id'] for e in selected], 'entries': entries,
            'blind_spot_ids': [e['candidate_id'] for e in selected[len(leading):]],
            'label_values_used': False, 'hard_rejection_by_correlation': False,
            'receipt': _receipt('training_only_conditional_prescreen', pool_hash=pool.digest,
                                cutoff=_aware(cutoff).isoformat(), candidates=len(candidates),
                                projection_calls=len(valid), selected=[e['candidate_id'] for e in selected])}


def stability_select(results, pool, *, minimum_winning_fraction=2/3, minimum_gain=0., limit=1):
    """Descriptive S stability gate, not a confidence interval or promotion."""
    if limit != 1 or minimum_winning_fraction != 2/3 or not np.isfinite(minimum_gain) or minimum_gain < 0:
        raise ValueError('First-version pool admission allows one stable monthly structure')
    if len(results) > MAX_ARCHIVE or len({r['candidate_id'] for r in results}) != len(results):
        raise ValueError('Stability selection exceeds bounded archive')
    eligible = []
    for result in results:
        if result['role'] != 'S' or result['pool_hash'] != pool.digest or len(result['folds']) != 3:
            raise ValueError('Stability result has a different pool/context or evidence role')
        metric = result['F']['Pv']
        if (not np.isfinite(metric['winning_fold_fraction'])
                or not 0 <= metric['winning_fold_fraction'] <= 1):
            raise ValueError('Invalid descriptive winning-fold fraction')
        if (metric['gain'] is not None and np.isfinite(metric['gain'])
                and metric['gain'] > minimum_gain
                and metric['winning_fold_fraction'] >= minimum_winning_fraction):
            eligible.append(result)
    selected = sorted(eligible, key=lambda r: (-r['F']['Pv']['gain'], r['candidate_id']))[:limit]
    return {'selected_ids': [r['candidate_id'] for r in selected], 'role': 'S',
            'formal_support_claimed': False,
            'receipt': _receipt('descriptive_stability_selection', pool_hash=pool.digest,
                                candidates=len(results), selected=[r['candidate_id'] for r in selected],
                                minimum_winning_fraction=minimum_winning_fraction,
                                minimum_gain=minimum_gain)}


def _flat_subset(pool, indices, version):
    return FrozenPool(tuple(pool.members[i] for i in indices), version, pool.pipeline_id, pool.mode,
                       pool.max_profile_calls, 'P_dev', None,
                       tuple(pool.member_access(i) for i in indices))


def backward_prune(data, cutoff, pool, *, budget=None, max_remove=1, tolerance=0.):
    """Bounded cold backward S ablation on identical original-pool support.

    Ablation freezes each formula/access policy and explicitly refits the whole
    remaining system. It returns a new development pool only. A conditional
    prefix fitting graph is flattened into a declared joint-refit pruning
    experiment for BOTH full and reduced controls, never reusing old scores.
    """
    if pool.role != 'P_dev' or type(max_remove) is not int or not 0 <= max_remove <= MAX_MEMBERS:
        raise ValueError('Pruning is bounded and may modify development pools only')
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Pruning loss tolerance must be finite and nonnegative')
    initial_vectors = _vectors(data.frame, pool, [m.parameters for m in pool.members])
    finite = _valid(data.frame, initial_vectors)
    folds = development_folds(cutoff, spec=data.spec)
    supports = [(data.train_mask(f['fit_cutoff']) & finite,
                 data.score_mask(f['fit_cutoff'], f['score_end'], f['score_end']) & finite) for f in folds]
    current = _flat_subset(pool, list(range(len(pool.members))), pool.version + ':prune-control')
    steps = []

    def score(candidate_pool):
        values, hours, receipts = [], [], []
        for fold, (train, evaluation) in zip(folds, supports, strict=True):
            if not evaluation.any():
                raise ValueError('Pruning requires common mature scoring support')
            fitted, _ = _fit_pool(data, fold['fit_cutoff'], candidate_pool, budget, support=train)
            prediction, mask = fitted.predict_history(data.frame)
            if (evaluation & ~mask).any():
                raise ValueError('Pruning changed the common finite evaluation support')
            values.append(_loss(data, prediction, evaluation))
            hours.append(len(np.unique(hour_ids(data.frame.metadata.loc[evaluation, 'available_at']))))
            receipts.append(fitted.receipt['receipt_id'])
        return float(np.average(values, weights=hours)), values, receipts

    for _ in range(min(max_remove, len(current.members))):
        baseline, fold_baseline, baseline_receipts = score(current)
        trials = []
        for i, member in enumerate(current.members):
            reduced = _flat_subset(current, [j for j in range(len(current.members)) if j != i],
                                   current.version + ':without:' + member.member_id[:12])
            loss, fold_loss, receipts = score(reduced)
            wins = float(np.mean(np.asarray(fold_loss) <= np.asarray(fold_baseline) + tolerance))
            trials.append((loss, member.member_id, reduced, wins, receipts))
        eligible = [t for t in trials if t[0] <= baseline + tolerance and t[3] >= 2/3]
        chosen = min(eligible, key=lambda t: (t[0], t[1])) if eligible else None
        steps.append(_receipt('backward_joint_refit_step', reference_pool=current.digest,
                              baseline_loss=baseline, baseline_fit_receipts=baseline_receipts,
                              trials=[{'removed': t[1], 'loss': t[0], 'winning_fold_fraction': t[3],
                                       'fit_receipts': t[4]} for t in trials],
                              removed=chosen[1] if chosen else None))
        if chosen is None:
            break
        current = chosen[2]
    removed_ids = sorted({m.member_id for m in pool.members} - {m.member_id for m in current.members})
    # No accepted removal must not silently replace the original fitting graph.
    result = current if removed_ids else pool
    return result, _receipt('development_backward_pruning', role='S', pool_hash=pool.digest,
                             result_pool_hash=result.digest, removed_ids=removed_ids, steps=steps,
                             reference_refit_policy='same_flat_joint_refit_for_full_and_reduced',
                             affects_A0=False, affects_confirmation=False)


class DevelopmentPoolArchive:
    """A bounded persistent archive shard with explicit old-shard references.

    At 96 context records, write/save this shard and roll over. Prior results
    remain in the immutable predecessor file; only its active parent records
    are copied into the next shard. Scores never cross a context identity.
    """
    def __init__(self, epoch, *, predecessor=None):
        self.epoch = _month(epoch)
        self.predecessor = copy.deepcopy(predecessor)
        self.records = {}
        self.lifecycle = {}
        self.admissions = {}

    @staticmethod
    def context(pool, *, cutoff, data_hash, protocol_hash, source_hash):
        if not all(isinstance(v, str) and v for v in (data_hash, protocol_hash, source_hash)):
            raise ValueError('Archive context requires data/protocol/source identities')
        return {'pool_hash': pool.digest, 'reference_pipeline_id': pool.pipeline_id,
                'cutoff': _aware(cutoff).isoformat(), 'data_hash': data_hash,
                'protocol_hash': protocol_hash, 'source_hash': source_hash}

    def record(self, member, result, context, *, state='active_parent'):
        if (result.get('role') != 'S' or result.get('pool_hash') != context['pool_hash']
                or result.get('candidate_id') != member.member_id
                or result.get('receipt', {}).get('cutoff') != _aware(context['cutoff']).isoformat()
                or result.get('context') != context
                or state not in {'active_parent', 'dormant', 'retired', 'pending_evidence'}):
            raise ValueError('Archive result is not matching development evidence')
        _aware(context['cutoff']).isoformat()
        key = fingerprint({'member': member.member_id, 'context': context})
        entry = {'member': member.to_dict(), 'result': copy.deepcopy(result),
                 'context': copy.deepcopy(context), 'state': state}
        if key in self.records:
            if self.records[key] != entry:
                raise ValueError('An archived formula/context result cannot be rewritten')
            return key
        if len(self.records) >= MAX_ARCHIVE:
            raise ValueError('Archive shard full: persist and roll over without deleting evidence')
        self.records[key] = entry
        self.lifecycle.setdefault(member.member_id, {'state': state, 'decay': []})
        return key

    def result_for(self, member_id, context):
        key = fingerprint({'member': member_id, 'context': context})
        return copy.deepcopy(self.records[key]['result']) if key in self.records else None

    def active_parents(self, context, *, limit=4):
        if type(limit) is not int or not 1 <= limit <= 4:
            raise ValueError('Each parent request may include one to four programs')
        newest = {}
        for entry in self.records.values():
            member = PoolMember.from_dict(entry['member'])
            if (self.lifecycle[member.member_id]['state'] != 'active_parent'
                    or _aware(entry['context']['cutoff']) > _aware(context['cutoff'])):
                continue
            old = newest.get(member.member_id)
            if old is None or entry['context']['cutoff'] > old['context']['cutoff']:
                newest[member.member_id] = entry
        ordered = sorted(newest.items(), key=lambda pair: (pair[1]['context']['cutoff'], pair[0]), reverse=True)
        return [{'member_id': identifier, 'member': copy.deepcopy(entry['member']),
                 'origin_context': copy.deepcopy(entry['context']),
                 'current_context_result': copy.deepcopy(entry['result']) if entry['context'] == context else None,
                 'prior_evidence_only': entry['context'] != context}
                for identifier, entry in ordered[:limit]]

    def monitor_decay(self, member_id, *, month, k0_gain, pool_gain, quality_ok=True):
        month = _month(month)
        if member_id not in self.lifecycle or type(quality_ok) is not bool:
            raise ValueError('Unknown archive member or data quality state')
        for value in (k0_gain, pool_gain):
            if value is not None and not np.isfinite(value):
                raise ValueError('Decay gains must be finite or explicitly unavailable')
        history = self.lifecycle[member_id]['decay']
        row = {'month': month, 'k0_gain': k0_gain, 'pool_gain': pool_gain, 'quality_ok': quality_ok}
        if history and _aware(month) <= _aware(history[-1]['month']):
            if history[-1] == row:
                return _receipt('development_decay', member_id=member_id, repeated=True, state=self.lifecycle[member_id]['state'])
            raise ValueError('Decay checks must be immutable increasing monthly endpoints')
        history.append(row)
        del history[:-3]
        consecutive = len(history) == 3 and all(
            _aware(history[i]['month']) + pd.DateOffset(months=1) == _aware(history[i + 1]['month']) for i in range(2))
        negative = consecutive and all(r['quality_ok'] and r['pool_gain'] is not None
                                       and r['pool_gain'] < 0 for r in history)
        reason = 'data_quality' if not quality_ok else 'insufficient_negative_checks'
        if negative:
            self.lifecycle[member_id]['state'] = 'pending_retirement'
            reason = ('relationship_weakened' if all(r['k0_gain'] is not None and r['k0_gain'] < 0 for r in history)
                      else 'possibly_replaced_by_pool')
        return _receipt('development_decay', member_id=member_id, checks=copy.deepcopy(history),
                        state=self.lifecycle[member_id]['state'], reason=reason, automatic_exit=False,
                        affects_A0=False)

    def admit(self, pool, member, *, month, selection):
        month = _month(month)
        if (pool.role != 'P_dev' or member.member_id not in selection.get('selected_ids', [])
                or selection.get('role') != 'S'
                or selection.get('receipt', {}).get('pool_hash') != pool.digest):
            raise ValueError('Admission requires matching stable development selection')
        if month in self.admissions:
            raise ValueError('At most one new development structure is admitted per month')
        new = _conditional_pool(pool, member, mode='raw', version=pool.version + ':admit:' + month,
                                max_profile_calls=pool.max_profile_calls)
        self.admissions[month] = {'member_id': member.member_id, 'pool_hash': new.digest}
        return new, _receipt('development_admission', month=month, pool_hash=pool.digest,
                              result_pool_hash=new.digest, member_id=member.member_id, affects_A0=False)

    def exit_members(self, pool, member_ids, *, selection):
        ids = set(member_ids)
        if (pool.role != 'P_dev' or selection.get('role') not in {'S', 'V'}
                or selection.get('pool_hash') != pool.digest or selection.get('decision') != 'retire'
                or not ids or not ids <= {m.member_id for m in pool.members}):
            raise ValueError('Retirement requires matching explicit development selection')
        new = _flat_subset(pool, [i for i, m in enumerate(pool.members) if m.member_id not in ids],
                            pool.version + ':retire')
        for member_id in ids:
            if member_id in self.lifecycle:
                self.lifecycle[member_id]['state'] = 'retired'
        return new, _receipt('development_retirement', pool_hash=pool.digest, result_pool_hash=new.digest,
                              member_ids=sorted(ids), selection=copy.deepcopy(selection), affects_A0=False)

    def to_dict(self):
        return {'version': 1, 'epoch': self.epoch, 'predecessor': copy.deepcopy(self.predecessor),
                'records': copy.deepcopy(self.records), 'lifecycle': copy.deepcopy(self.lifecycle),
                'admissions': copy.deepcopy(self.admissions)}

    @property
    def digest(self):
        return fingerprint(self.to_dict())

    def save(self, path):
        return immutable_record(path, {'archive': self.to_dict(), 'digest': self.digest})

    @classmethod
    def load(cls, path):
        record = json.loads(project_path(path).read_text('utf-8'))
        value = record['archive']
        return cls.from_dict(value, expected_digest=record.get('digest'))

    @classmethod
    def from_dict(cls, value, *, expected_digest=None):
        value = copy.deepcopy(value)
        if ((expected_digest is not None and expected_digest != fingerprint(value))
                or value.get('version') != 1 or len(value['records']) > MAX_ARCHIVE):
            raise ValueError('Archive integrity/version/bounds failure')
        archive = cls(value['epoch'], predecessor=value['predecessor'])
        archive.records, archive.lifecycle, archive.admissions = value['records'], value['lifecycle'], value['admissions']
        for key, entry in archive.records.items():
            member = PoolMember.from_dict(entry['member'])
            if key != fingerprint({'member': member.member_id, 'context': entry['context']}):
                raise ValueError('Archive member/context identity mismatch')
        return archive

    def rollover(self, next_month, saved_path):
        saved = type(self).load(saved_path)
        if saved.digest != self.digest or _aware(next_month) <= _aware(self.epoch):
            raise ValueError('Persist the exact old shard before forward rollover')
        new = type(self)(next_month, predecessor={'path': str(project_path(saved_path)), 'digest': self.digest})
        recent = sorted(self.records.items(), key=lambda pair: (pair[1]['context']['cutoff'], pair[0]), reverse=True)
        kept = set()
        for key, entry in recent:
            member_id = PoolMember.from_dict(entry['member']).member_id
            if member_id not in kept and self.lifecycle[member_id]['state'] == 'active_parent':
                new.records[key] = copy.deepcopy(entry)
                new.lifecycle[member_id] = copy.deepcopy(self.lifecycle[member_id])
                kept.add(member_id)
                if len(kept) == MAX_MEMBERS:
                    break
        return new

    def verify_history(self, *, max_shards=120):
        """Validate all predecessor files explicitly; load verifies one shard only."""
        if type(max_shards) is not int or not 1 <= max_shards <= 120:
            raise ValueError('History verification is bounded to 120 shards')
        record, seen, digests = self.predecessor, set(), []
        while record is not None:
            path = str(project_path(record['path']))
            if path in seen or len(seen) >= max_shards:
                raise ValueError('Cyclic or oversized pool predecessor history')
            seen.add(path)
            previous = type(self).load(path)
            if previous.digest != record['digest']:
                raise ValueError('Pool predecessor digest mismatch')
            digests.append(previous.digest)
            record = previous.predecessor
        return _receipt('verified_pool_history', shard_count=len(digests), digests=digests)
