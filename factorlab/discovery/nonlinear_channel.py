"""Scheduled tree diagnostics and explicitly gated, isolated neural research.

These are S system comparisons, not F novelty attribution or forward evidence.
Importing this module, running the tree, or disabling neural research loads no
torch. No model is promoted here and no shared event/state head is trained.
"""
from __future__ import annotations

import copy
import hashlib
import pickle
import time
from dataclasses import asdict, dataclass, field

import numpy as np
from threadpoolctl import threadpool_limits

from factorlab.protocol import fingerprint
from factorlab.time_utils import calendar_window

from .convex_eval import VOLATILITY_FLOOR, fit_ridge, log_mse
from .known_library import FactorFrame
from .rolling import development_folds


@dataclass(frozen=True)
class ProbeConfig:
    version: str = 'v7-quarterly-tree-v1'
    anchor_month: int = 1
    max_iter: int = 32
    max_leaf_nodes: int = 7
    max_depth: int = 3
    min_samples_leaf: int = 32
    learning_rate: float = .05
    l2_regularization: float = .1
    random_state: int = 17

    def __post_init__(self):
        if any(type(getattr(self, name)) is not type(spec.default)
               or getattr(self, name) != spec.default
               for name, spec in self.__dataclass_fields__.items()):
            raise ValueError('Tree hyperparameters are frozen; a change requires a new protocol')

    @property
    def digest(self):
        return fingerprint(asdict(self))


@dataclass
class ChannelBudget:
    """Independent expensive-fit counters plus an optional shared solve ledger.

    The controller persists to_dict after each reservation with persist=. Failed
    work consumes reservations. Construct/restore one ledger per research batch.
    """
    max_tree_fits: int = 3
    max_neural_fits: int = 18
    max_epochs: int = 180
    max_seconds: float = 3600.
    counts: dict = field(default_factory=lambda: {'tree_fit': 0, 'neural_fit': 0, 'epoch': 0, 'solve': 0})
    shared: object = field(default=None, repr=False)
    persist: object = field(default=None, repr=False)
    spent_seconds: float = 0.
    _deadline: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        for name, maximum in [('max_tree_fits', 3), ('max_neural_fits', 18), ('max_epochs', 360)]:
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= maximum:
                raise ValueError('Invalid independent nonlinear budget')
        if not np.isfinite(self.max_seconds) or not 0 < self.max_seconds <= 7200:
            raise ValueError('Invalid nonlinear wall budget')
        limits = self.limits
        if (set(self.counts) != set(limits) or any(type(v) is not int or not 0 <= v <= limits[k]
                                                for k, v in self.counts.items())
                or not np.isfinite(self.spent_seconds) or self.spent_seconds < 0):
            raise ValueError('Invalid nonlinear budget state')

    @property
    def limits(self):
        return {'tree_fit': self.max_tree_fits, 'neural_fit': self.max_neural_fits,
                'epoch': self.max_epochs, 'solve': 1600}

    def consume(self, kind, amount=1):
        self.check_time()
        if (kind not in self.limits or type(amount) is not int or amount < 1
                or self.counts[kind] + amount > self.limits[kind]
                or self.spent_seconds >= self.max_seconds):
            raise ValueError('Nonlinear resource budget exhausted')
        if kind == 'solve' and self.shared is not None:
            self.shared.consume('solve', amount)
        self.counts[kind] += amount
        self._persist()

    def begin(self):
        self._deadline = time.monotonic() + max(0., self.max_seconds - self.spent_seconds)
        self.check_time()

    def check_time(self):
        if self._deadline is not None and time.monotonic() >= self._deadline:
            raise ValueError('Nonlinear wall-time budget exhausted')

    def charge_time(self, seconds):
        self.spent_seconds += seconds
        self._persist()

    def _persist(self):
        if self.persist is not None:
            self.persist(self.to_dict())

    def to_dict(self):
        return {'version': 'v7-nonlinear-budget-v1', 'max_tree_fits': self.max_tree_fits,
                'max_neural_fits': self.max_neural_fits, 'max_epochs': self.max_epochs,
                'max_seconds': self.max_seconds, 'counts': dict(self.counts),
                'spent_seconds': self.spent_seconds}

    @classmethod
    def from_dict(cls, value, **callbacks):
        value = dict(value)
        if value.pop('version', None) != 'v7-nonlinear-budget-v1':
            raise ValueError('Invalid nonlinear budget version')
        return cls(**value, **callbacks)


def _budget(value):
    return value if isinstance(value, ChannelBudget) else ChannelBudget(shared=value)


def _task(data, cutoff):
    _, end = data.spec.window(cutoff)
    if data.spec.months != 3 or data.spec.task != 'volatility' or data.spec.horizon_hours != 12:
        raise ValueError('Nonlinear channel isolates three-month volatility12h research')
    return end


def _fold(data, fold, pool, budget):
    from .joint_pool import FrozenPool, fit_pool
    pool = FrozenPool() if pool is None else pool
    fitted = fit_pool(data, fold['fit_cutoff'], pool, budget=budget)
    frame, feature_mask = fitted.feature_history(data.frame)
    baseline, prediction_mask = fitted.predict_history(data.frame)
    valid = np.asarray(feature_mask) & np.asarray(prediction_mask) & frame.valid_rows
    train = data.train_mask(fold['fit_cutoff']) & valid
    score = data.score_mask(fold['fit_cutoff'], fold['score_end'], fold['score_end']) & valid
    if train.sum() < data.spec.minimum_train_rows or not score.any():
        raise ValueError('Insufficient same-input common nonlinear support')
    if not np.isfinite(baseline[train | score]).all() or (baseline[train | score] < 0).any():
        raise ValueError('Invalid frozen baseline predictions')
    return fitted, frame, baseline, train, score


def _decode(q):
    with np.errstate(over='ignore', invalid='ignore'):
        raw = np.exp(q) - VOLATILITY_FLOOR
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite nonlinear forecast')
    return np.maximum(raw, 0.), int((raw < 0).sum())


def run_probe(data, cutoff, pool=None, budget=None):
    """Jan/Apr/Jul/Oct diagnostic, with three predeclared cold development folds."""
    end = _task(data, cutoff)
    config, budget = ProbeConfig(), _budget(budget)
    if (end.month - config.anchor_month) % 3:
        return {'status': 'not_due', 'cutoff': end.isoformat(), 'protocol_id': config.digest,
                'role': 'S', 'formal_claim': False}
    budget.begin()
    from sklearn import __version__ as sklearn_version
    from sklearn.ensemble import HistGradientBoostingRegressor

    from .joint_pool import hour_equal_weights
    records = []
    for fold in development_folds(end, spec=data.spec):
        started = time.monotonic()
        budget.consume('tree_fit')
        try:
            fitted, frame, reference, train, score = _fold(data, fold, pool, budget)
            parameters = {k: v for k, v in asdict(config).items() if k not in {'version', 'anchor_month'}}
            tree = HistGradientBoostingRegressor(**parameters, early_stopping=False)
            train_weights = hour_equal_weights(frame.metadata.loc[train])
            score_weights = hour_equal_weights(frame.metadata.loc[score])
            with threadpool_limits(limits=1):
                tree.fit(frame.values[train], np.log(data.target[train] + VOLATILITY_FLOOR),
                         sample_weight=train_weights)
                prediction, clipped = _decode(tree.predict(frame.values[score]))
            base_loss = log_mse(data.target[score], reference[score], score_weights)
            loss = log_mse(data.target[score], prediction, score_weights)
            records.append({**fold, 'status': 'scored', 'fit': data.training_identity(fold['fit_cutoff']),
                            'pool_id': fitted.pool.digest, 'feature_schema': frame.schema_hash,
                            'model_id': hashlib.sha256(pickle.dumps(tree, protocol=5)).hexdigest(),
                            'reference_model_id': fitted.digest, 'support': int(score.sum()),
                            'sample_ids': frame.metadata.sample_id[score].tolist(),
                            'reference_loss': base_loss, 'loss': loss,
                            'gain': 1 - loss / base_loss if base_loss > 0 else None,
                            'clipped_count': clipped, 'predictions': prediction.tolist()})
        finally:
            budget.charge_time(time.monotonic() - started)
    return {'status': 'completed', 'role': 'S', 'formal_claim': False,
            'protocol_id': config.digest, 'config': asdict(config), 'sklearn_version': sklearn_version,
            'folds': records, 'budget': budget.to_dict(),
            'repeatable_development_direction': all(r['gain'] is not None and r['gain'] > 0 for r in records),
            'interpretation': 'Same-input system diagnostic; positive development direction is not confirmation.'}


GATE_FIELDS = ('probe_repeatability', 'positive_control', 'resource_verified',
               'protocol_verified', 'independent_budget')


@dataclass(frozen=True)
class NeuralGate:
    probe_repeatability: bool = False
    positive_control: bool = False
    resource_verified: bool = False
    protocol_verified: bool = False
    independent_budget: bool = False
    evidence_ids: dict = field(default_factory=dict)

    def validate(self):
        if (any(getattr(self, name) is not True for name in GATE_FIELDS)
                or set(self.evidence_ids) != set(GATE_FIELDS)
                or any(type(v) is not str or not v or len(v) > 128 for v in self.evidence_ids.values())):
            raise ValueError('Neural research requires every gate and its frozen evidence receipt')


@dataclass(frozen=True)
class StructureSpec:
    width: int = 8
    representation_dim: int = 2
    experts: int = 1

    def __post_init__(self):
        if (type(self.width) is not int or self.width not in (4, 8, 16, 32)
                or type(self.representation_dim) is not int or self.representation_dim not in (2, 4, 8)
                or type(self.experts) is not int or self.experts not in (1, 2)
                or self.representation_dim > self.width):
            raise ValueError('Structure is outside the registered bounded grammar')

    @property
    def digest(self):
        return fingerprint({'version': 'v7-small-residual-v1', **asdict(self)})


def structure_suggestions():
    """Explicit registrations, never executable code generation or auto-promotion."""
    return [{'structure_id': s.digest, 'structure': asdict(s), 'status': 'proposed',
             'requires_gate': list(GATE_FIELDS)} for s in
            (StructureSpec(8, 2, 1), StructureSpec(16, 4, 1), StructureSpec(16, 4, 2))]


@dataclass(frozen=True)
class ResidualConfig:
    enabled: bool = False
    structures: tuple[StructureSpec, ...] = (StructureSpec(),)
    seeds: tuple[int, ...] = (17, 29, 43)
    epochs: int = 4
    batch_size: int = 64
    learning_rate: float = .001

    def __post_init__(self):
        if (type(self.enabled) is not bool or not 1 <= len(self.structures) <= 2
                or any(not isinstance(s, StructureSpec) for s in self.structures)
                or len({s.digest for s in self.structures}) != len(self.structures)
                or not 2 <= len(self.seeds) <= 3 or len(set(self.seeds)) != len(self.seeds)
                or any(type(s) is not int or not 0 <= s < 2**31 for s in self.seeds)
                or type(self.epochs) is not int or not 0 <= self.epochs <= 20
                or type(self.batch_size) is not int or not 1 <= self.batch_size <= 128
                or not np.isfinite(self.learning_rate) or not 0 < self.learning_rate <= .01):
            raise ValueError('Invalid bounded residual experiment')

    @property
    def digest(self):
        return fingerprint({'version': 'v7-neural-S-channel-v1', **asdict(self),
                            'checkpoint': 'training_loss_only_including_exact_reference_epoch0',
                            'readout': 'same_ridge_plus_equal_dim_initial_encoder_control'})


def _encoded_frame(frame, embedding, model_id):
    return FactorFrame(np.column_stack((frame.values, embedding)), frame.metadata,
                       (*frame.feature_names, *(f'encoder_{i}' for i in range(embedding.shape[1]))),
                       (*frame.bypass_mask, *(False for _ in range(embedding.shape[1]))),
                       fingerprint({'base': frame.schema_hash, 'encoder': model_id}))


def _train_residual(torch, data, fold, fitted, frame, reference, train, score,
                    structure, seed, config, budget):
    from .joint_pool import hour_equal_weights

    indices = np.flatnonzero(train | score)
    X = fitted.predictor.scaler.transform(frame.take(indices))
    local_train, local_score = train[indices], score[indices]
    Xt = torch.tensor(X[local_train], dtype=torch.float64)
    y = torch.tensor(np.log(data.target[train] + VOLATILITY_FLOOR), dtype=torch.float64)
    base = torch.tensor(np.log(reference[train] + VOLATILITY_FLOOR), dtype=torch.float64)
    train_weights = hour_equal_weights(frame.metadata.loc[train])
    score_weights = hour_equal_weights(frame.metadata.loc[score])
    weights = torch.tensor(train_weights, dtype=torch.float64)
    network = torch.nn.Sequential(torch.nn.Linear(X.shape[1], structure.width), torch.nn.Tanh(),
                                  torch.nn.Linear(structure.width, structure.representation_dim),
                                  torch.nn.Tanh()).double()
    heads = torch.nn.ModuleList([torch.nn.Linear(structure.representation_dim, 1).double()
                                 for _ in range(structure.experts)])
    for head in heads:
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
    optimizer = torch.optim.Adam([*network.parameters(), *heads.parameters()], lr=config.learning_rate)
    with torch.no_grad():
        initial_embedding = network(torch.tensor(X, dtype=torch.float64)).numpy().copy()
    best_loss, best_epoch = log_mse(data.target[train], reference[train], train_weights), 0
    best_state = (copy.deepcopy(network.state_dict()), copy.deepcopy(heads.state_dict()))
    orders, rng = [], np.random.default_rng(seed)
    for epoch in range(1, config.epochs + 1):
        budget.consume('epoch')
        order = rng.permutation(len(Xt))
        orders.append(fingerprint(order.tolist()))
        for left in range(0, len(order), config.batch_size):
            budget.check_time()
            chosen = order[left:left + config.batch_size]
            code = network(Xt[chosen])
            correction = torch.stack([head(code).squeeze(-1) for head in heads]).mean(0)
            loss = (weights[chosen] * torch.square(base[chosen] + correction - y[chosen])).sum() / weights[chosen].sum()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite residual training loss')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            code = network(Xt)
            correction = torch.stack([head(code).squeeze(-1) for head in heads]).mean(0).numpy()
        prediction, _ = _decode(base.numpy() + correction)
        candidate_loss = log_mse(data.target[train], prediction, train_weights)
        if candidate_loss < best_loss:
            best_loss, best_epoch = candidate_loss, epoch
            best_state = (copy.deepcopy(network.state_dict()), copy.deepcopy(heads.state_dict()))
    network.load_state_dict(best_state[0])
    heads.load_state_dict(best_state[1])
    with torch.no_grad():
        code = network(torch.tensor(X, dtype=torch.float64))
        embedding = code.numpy()
        correction = torch.stack([head(code).squeeze(-1) for head in heads]).mean(0).numpy()
    prediction, clipped = _decode(np.log(reference[score] + VOLATILITY_FLOOR) + correction[local_score])
    if best_epoch == 0:
        prediction, clipped = reference[score].copy(), 0
    parameters = {'encoder': {k: v.numpy().tolist() for k, v in best_state[0].items()},
                  'experts': {k: v.numpy().tolist() for k, v in best_state[1].items()}}
    model_id = fingerprint({'structure': structure.digest, 'seed': seed, 'parameters': parameters,
                            'reference_model': fitted.digest, 'protocol': config.digest})
    readouts = {}
    for label, codes in [('trained_encoder', embedding), ('initial_encoder_dimension_control', initial_embedding)]:
        extended = _encoded_frame(frame.take(indices), codes, model_id + ':' + label)
        budget.consume('solve')
        model = fit_ridge(extended.take(local_train), data.target[train], ridge=data.spec.ridge,
                          sample_weight=train_weights)
        readouts[label] = {'loss': log_mse(data.target[score], model.predict(extended.take(local_score)), score_weights),
                           'model_id': model.digest, 'dimensions': codes.shape[1]}
    base_loss = log_mse(data.target[score], reference[score], score_weights)
    loss = log_mse(data.target[score], prediction, score_weights)
    return {**fold, 'seed': seed, 'structure_id': structure.digest, 'model_id': model_id,
            'reference_model_id': fitted.digest, 'fit': data.training_identity(fold['fit_cutoff']),
            'selected_epoch': best_epoch, 'epoch0_loss': base_loss, 'loss': loss,
            'gain': 1 - loss / base_loss if base_loss > 0 else None,
            'train_objective': best_loss, 'data_order_hashes': orders,
            'support': int(score.sum()), 'sample_ids': frame.metadata.sample_id[score].tolist(),
            'predictions': prediction.tolist(), 'clipped_count': clipped,
            'convex_readouts': readouts, 'fitted_parameters': parameters,
            'attribution': 'S system comparison; extra dimensions do not establish F novelty'}


def research_residual(data, cutoff, pool=None, gate=None, config=None, budget=None):
    config = config or ResidualConfig()
    if not config.enabled:
        return {'status': 'disabled', 'formal_claim': False, 'protocol_id': config.digest}
    if any(structure.experts != 1 for structure in config.structures):
        raise ValueError('Multiple experts require an independently verified MoE gate; not enabled')
    end = _task(data, cutoff)
    if not isinstance(gate, NeuralGate):
        raise TypeError('An explicit neural research gate is required')
    gate.validate()
    budget = _budget(budget)
    budget.begin()
    if (3 * len(config.structures) * len(config.seeds) > budget.max_neural_fits - budget.counts['neural_fit']
            or 3 * len(config.structures) * len(config.seeds) * config.epochs > budget.max_epochs - budget.counts['epoch']):
        raise ValueError('Insufficient independent budget for all candidate/seed/fold fits')
    import torch
    old_threads = torch.get_num_threads()
    records = []
    try:
        torch.set_num_threads(1)
        for fold in development_folds(end, spec=data.spec):
            began = time.monotonic()
            try:
                fitted, frame, reference, train, score = _fold(data, fold, pool, budget)
            finally:
                budget.charge_time(time.monotonic() - began)
            for structure in config.structures:
                for seed in config.seeds:
                    budget.consume('neural_fit')
                    started = time.monotonic()
                    try:
                        with torch.random.fork_rng(devices=[]):
                            torch.manual_seed(seed)
                            record = _train_residual(torch, data, fold, fitted, frame, reference,
                                                     train, score, structure, seed, config, budget)
                        records.append({'status': 'scored', **record})
                    except (ValueError, RuntimeError, FloatingPointError) as exc:
                        records.append({**fold, 'structure_id': structure.digest, 'seed': seed,
                                        'status': 'failed', 'error': str(exc)[:400]})
                    finally:
                        budget.charge_time(time.monotonic() - started)
    finally:
        torch.set_num_threads(old_threads)
    return {'status': 'completed' if all(r['status'] == 'scored' for r in records) else 'incomplete',
            'role': 'S', 'formal_claim': False, 'protocol_id': config.digest,
            'config': asdict(config), 'gate_evidence': dict(gate.evidence_ids),
            'records': records, 'budget': budget.to_dict(), 'torch_version': torch.__version__,
            'interpretation': 'Cold task-isolated research; no shared state/event head, no automatic promotion.'}
