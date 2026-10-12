"""Frozen complete neural recipes and NumPy-only forward artifacts.

Model contents never confer admission. Artificial fits keep their test-only
identity; isolated workers emit neutral payloads that need a separate verified
admission certificate. S seed losses do not score the fixed log ensemble.
"""
from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np
from threadpoolctl import threadpool_limits

from factorlab.protocol import fingerprint, project_path
from factorlab.time_utils import calendar_window

from .convex_eval import VOLATILITY_FLOOR, ResidualProjector, TrainScaler, fit_ridge, log_mse
from .joint_pool import FrozenPool, JointPredictor, _ridge_from_dict, fit_pool, hour_equal_weights
from .known_library import FEATURE_NAMES, library_schema
from .nonlinear_channel import ChannelBudget, ResidualConfig, StructureSpec, _encoded_frame
from .nonlinear_orchestration import causal_worker_data
from .rolling import RollingSpec
from .v7_storage import immutable_record

RULES = {'fit': 'cold_three_UTC_calendar_months', 'target': 'log(volatility12h+1e-8)',
         'optimizer': 'Adam_default_betas_eps_weight_decay0', 'dtype': 'float64_CPU_single_thread',
         'checkpoint': 'strict_training_logMSE_improvement_including_exact_reference_epoch0',
         'aggregation': 'mean_log_of_decoded_seed_predictions', 'weighting': 'equal_symbol_then_hour',
         'seed_order': 'recipe_order_numpy_default_rng_per_seed', 'epoch0': 'exact_reference_for_both_outputs',
         'controls': 'trained_and_initial_encoder_same_ridge_same_support',
         'primary_loss': 'canonical_logMSE', 'automatic_adoption': False}


def _config(value):
    values = dict(value)
    values['structures'] = tuple(StructureSpec(**s) for s in values['structures'])
    values['seeds'] = tuple(values['seeds'])
    return ResidualConfig(**values)


def _spec(value):
    values = dict(value)
    values['event_bins'] = tuple(values['event_bins'])
    return RollingSpec(**values)


@dataclass(frozen=True)
class FrozenNeuralRecipe:
    pool: FrozenPool = field(default_factory=FrozenPool)
    rolling_spec: RollingSpec = field(default_factory=RollingSpec)
    config: ResidualConfig = field(default_factory=lambda: ResidualConfig(enabled=True))
    output: str = 'residual'
    _sealed_digest: str = field(default='', init=False, repr=False)

    def __post_init__(self):
        if (not isinstance(self.pool, FrozenPool) or not isinstance(self.rolling_spec, RollingSpec)
                or not isinstance(self.config, ResidualConfig) or not self.config.enabled
                or len(self.config.structures) != 1 or self.config.structures[0].experts != 1
                or self.output not in {'residual', 'encoder_ridge'}
                or self.rolling_spec.task != 'volatility' or self.rolling_spec.horizon_hours != 12
                or self.rolling_spec.months != 3):
            raise ValueError('Neural recipe freezes one single-expert volatility12h structure and cold three-month fit')
        object.__setattr__(self, 'pool', FrozenPool.from_dict(self.pool.to_dict()))
        object.__setattr__(self, '_sealed_digest', fingerprint(self.to_dict()))

    def to_dict(self):
        rules = copy.deepcopy(RULES)
        if self.rolling_spec.window_days is not None:
            rules['fit'] = 'cold_continuous_90_days_at_original_15_day_cutoff'
        return {'version': 'v7-neural-forward-recipe-v1', 'pool': self.pool.to_dict(),
                'rolling_spec': asdict(self.rolling_spec), 'config': asdict(self.config),
                'output': self.output, 'input': library_schema(), 'rules': rules}

    @property
    def pipeline_id(self):
        if fingerprint(self.to_dict()) != self._sealed_digest:
            raise ValueError('Frozen neural recipe changed')
        return self._sealed_digest

    @classmethod
    def from_dict(cls, value):
        if set(value) != {'version', 'pool', 'rolling_spec', 'config', 'output', 'input', 'rules'}:
            raise ValueError('Unsupported neural recipe fields')
        if (value['version'] != 'v7-neural-forward-recipe-v1' or value['input'] != library_schema()):
            raise ValueError('Unknown neural recipe policy or input')
        result = cls(FrozenPool.from_dict(value['pool']), _spec(value['rolling_spec']),
                     _config(value['config']), value['output'])
        if fingerprint(result.to_dict()) != fingerprint(value):
            raise ValueError('Neural recipe is not a lossless supported schema')
        return result

    @property
    def budget_limits(self):
        def baseline_solves(pool):
            prefix = len(pool.reference_pool.members) if pool.reference_pool else 0
            calls = 1 + (pool.max_profile_calls if any(
                m.parameter_mode == 'profile' and m.parameters for m in pool.members[prefix:]) else 0)
            return calls + (baseline_solves(pool.reference_pool) if pool.reference_pool else 0)
        _ = self.pipeline_id
        seeds = len(self.config.seeds)
        return {'solve': baseline_solves(self.pool) + 2 * seeds, 'neural_fit': seeds,
                'epoch': seeds * self.config.epochs, 'seconds': 300}


def _array(value, shape, label):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f'Invalid finite neural {label} shape')
    return array


def _encoder(parameters, inputs, structure):
    shapes = {'0.weight': (structure.width, inputs), '0.bias': (structure.width,),
              '2.weight': (structure.representation_dim, structure.width),
              '2.bias': (structure.representation_dim,)}
    if set(parameters) != set(shapes):
        raise ValueError('Unsupported encoder parameter names')
    return {name: _array(parameters[name], shape, name) for name, shape in shapes.items()}


def _embedding(X, parameters, structure):
    weights = _encoder(parameters, X.shape[1], structure)
    return np.tanh(np.tanh(X @ weights['0.weight'].T + weights['0.bias'])
                   @ weights['2.weight'].T + weights['2.bias'])


def _ridge(value):
    model = _ridge_from_dict(value)
    n = len(model.feature_names)
    if (len(set(model.feature_names)) != n or model.coefficients.shape != (n,)
            or not np.isfinite(model.coefficients).all() or not math.isfinite(model.intercept)
            or not math.isfinite(model.ridge) or model.ridge <= 0 or len(model.scaler.center) != n):
        raise ValueError('Invalid complete convex readout')
    TrainScaler.from_dict(model.scaler.to_dict())
    return model


def _joint(value):
    if set(value) != {'kind', 'version', 'pool', 'parameters', 'projectors', 'predictor', 'receipt'}:
        raise ValueError('Unsupported baseline model fields')
    if value['kind'] != 'v7_joint_pool' or value['version'] != 1:
        raise ValueError('Neural reference must be a complete frozen joint model')
    pool = FrozenPool.from_dict(value['pool'])
    if len(value['parameters']) != len(pool.members):
        raise ValueError('Unaligned frozen pool parameters')
    for member, parameters in zip(pool.members, value['parameters'], strict=True):
        if set(parameters) != set(member.program['parameters']):
            raise ValueError('Unknown frozen pool parameter')
        for key, number in parameters.items():
            bounds = member.program['parameters'][key]
            if not math.isfinite(number) or not bounds['lower'] <= number <= bounds['upper']:
                raise ValueError('Frozen pool parameters exceed declared bounds')
    result = JointPredictor(pool, tuple(value['parameters']), tuple(
        ResidualProjector.from_dict(p) if p is not None else None for p in value['projectors']),
        _ridge(value['predictor']), value['receipt'])
    if fingerprint(result.to_dict()) != fingerprint(value):
        raise ValueError('Baseline model schema changed on restoration')
    return result


def _decode(q):
    with np.errstate(over='ignore', invalid='ignore'):
        raw = np.exp(q) - VOLATILITY_FLOOR
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite neural prediction')
    return np.maximum(raw, 0.), int((raw < 0).sum())


@dataclass(frozen=True)
class NeuralForwardModel:
    recipe: FrozenNeuralRecipe
    baseline: JointPredictor
    seeds: tuple[dict, ...]
    receipt: dict
    kind: str = field(default='v7_neural_forward', init=False)
    task_support: tuple[str, ...] = field(default=('volatility',), init=False)
    _sealed_digest: str = field(default='', init=False, repr=False)

    def __post_init__(self):
        recipe = FrozenNeuralRecipe.from_dict(self.recipe.to_dict())
        baseline = _joint(self.baseline.to_dict())
        object.__setattr__(self, 'recipe', recipe)
        object.__setattr__(self, 'baseline', baseline)
        object.__setattr__(self, 'seeds', tuple(copy.deepcopy(s) for s in self.seeds))
        object.__setattr__(self, 'receipt', copy.deepcopy(self.receipt))
        start, end = recipe.rolling_spec.window(baseline.receipt['cutoff'])
        receipt_fields = {'version', 'pipeline_id', 'baseline_id', 'training', 'cutoff', 'train_start',
                          'fit', 'test_only', 'runtime_admitted', 'formal_claim', 'automatic_adoption',
                          'torch_version', 'numpy_version', 'budget_counts', 'interpretation'}
        artificial = (self.receipt.get('version') == 'v7-neural-cold-fit-v1'
                      and set(self.receipt) == receipt_fields
                      and self.receipt.get('test_only') is True
                      and self.receipt.get('runtime_admitted') is False)
        neutral_fields = (receipt_fields - {'test_only', 'runtime_admitted'}) | {
            'payload_origin', 'admission_required'}
        neutral_origin = {
            'v7-neural-cold-payload-v2': 'isolated_forward_worker',
            'v7-neural-research-payload-v1': 'isolated_research_worker',
        }
        neutral = (self.receipt.get('version') in neutral_origin
                   and set(self.receipt) == neutral_fields
                   and self.receipt.get('payload_origin') == neutral_origin.get(self.receipt.get('version'))
                   and self.receipt.get('admission_required') is True)
        if (baseline.pool.digest != recipe.pool.digest or baseline.predictor.ridge != recipe.rolling_spec.ridge
                or baseline.receipt['training']['rolling_hash'] != recipe.rolling_spec.digest
                or self.receipt.get('pipeline_id') != recipe.pipeline_id
                or self.receipt.get('baseline_id') != baseline.digest
                or self.receipt.get('training') != baseline.receipt['training']
                or self.receipt.get('formal_claim') is not False
                or self.receipt.get('automatic_adoption') is not False
                or not (artificial or neutral)
                or self.receipt['fit'] != 'cold' or self.receipt['cutoff'] != end.isoformat()
                or self.receipt['train_start'] != start.isoformat()
                or baseline.receipt['training']['cutoff'] != end.isoformat()):
            raise ValueError('Complete neural artifact lacks matching cold reference or admission boundary')
        self._validate_seeds()
        counts = self.receipt['budget_counts']
        ceilings = {'tree_fit': 3, 'neural_fit': 18, 'epoch': 360, 'solve': 1600}
        solves = baseline.receipt['solve_calls']
        if (not isinstance(counts, dict) or set(counts) != set(ceilings)
                or any(type(v) is not int or not 0 <= v <= ceilings[k] for k, v in counts.items())
                or type(solves) is not int or solves < 1
                or counts['solve'] < solves + 2 * len(self.seeds)
                or counts['neural_fit'] < len(self.seeds)
                or counts['epoch'] < len(self.seeds) * recipe.config.epochs):
            raise ValueError('Invalid cumulative neural budget receipt or missing completed work')
        object.__setattr__(self, '_sealed_digest', fingerprint(self.to_dict()))

    def _validate_seeds(self):
        config, width = self.recipe.config, len(self.baseline.predictor.feature_names)
        structure = config.structures[0]
        if tuple(s.get('seed') for s in self.seeds) != config.seeds:
            raise ValueError('Seed ensemble differs from its frozen recipe')
        required = {'seed', 'selected_epoch', 'encoder', 'head', 'initial_encoder', 'trained_readout',
                    'initial_readout', 'encoding_id', 'initial_encoding_id', 'data_order_hashes',
                    'training_loss', 'epoch0_loss'}
        for seed in self.seeds:
            if (set(seed) != required or type(seed['selected_epoch']) is not int
                    or not 0 <= seed['selected_epoch'] <= config.epochs
                    or len(seed['data_order_hashes']) != config.epochs
                    or any(not isinstance(h, str) or len(h) != 64 for h in seed['data_order_hashes'])
                    or any(not math.isfinite(seed[k]) or seed[k] < 0 for k in ('training_loss', 'epoch0_loss'))
                    or seed['training_loss'] > seed['epoch0_loss']):
                raise ValueError('Invalid frozen seed checkpoint or training control')
            for key in ('encoder', 'initial_encoder'):
                _encoder(seed[key], width, structure)
            if set(seed['head']) != {'weight', 'bias'}:
                raise ValueError('Only the registered one-head residual is supported')
            _array(seed['head']['weight'], (1, structure.representation_dim), 'head')
            _array(seed['head']['bias'], (1,), 'head bias')
            if seed['selected_epoch'] == 0 and (np.any(seed['head']['weight']) or np.any(seed['head']['bias'])):
                raise ValueError('Epoch zero must be the exact frozen reference')
            if (seed['selected_epoch'] == 0 and (seed['training_loss'] != seed['epoch0_loss']
                    or seed['encoder'] != seed['initial_encoder'])):
                raise ValueError('Epoch zero cannot contain trained encoder parameters')
            if seed['selected_epoch'] > 0 and seed['training_loss'] >= seed['epoch0_loss']:
                raise ValueError('Selected trained epoch must strictly improve its training reference')
            for label, encoder in [('trained', 'encoder'), ('initial', 'initial_encoder')]:
                identity_key = 'encoding_id' if label == 'trained' else 'initial_encoding_id'
                expected_id = fingerprint({'pipeline': self.recipe.pipeline_id, 'seed': seed['seed'],
                                           'encoder': seed[encoder], 'label': label})
                readout = _ridge(seed[label + '_readout'])
                expected_names = (*self.baseline.predictor.feature_names,
                                  *(f'encoder_{i}' for i in range(structure.representation_dim)))
                expected_schema = fingerprint({'base': self.baseline.predictor.input_schema_hash,
                                               'encoder': expected_id})
                if (seed[identity_key] != expected_id or readout.feature_names != expected_names
                        or readout.input_schema_hash != expected_schema or readout.ridge != self.recipe.rolling_spec.ridge
                        or readout.scaler.bypass_mask != (*self.baseline.predictor.scaler.bypass_mask,
                                                         *((False,) * structure.representation_dim))):
                    raise ValueError('Encoder and same-convex readout are not aligned')

    def _assert_integrity(self):
        _ = self.recipe.pipeline_id
        if fingerprint(self.to_dict()) != self._sealed_digest:
            raise ValueError('Frozen neural model was modified')

    @property
    def pipeline_id(self):
        self._assert_integrity()
        return self.recipe.pipeline_id

    @property
    def digest(self):
        self._assert_integrity()
        return self._sealed_digest

    @property
    def schema(self):
        return {'input': library_schema(), 'pipeline_id': self.pipeline_id, 'output': self.recipe.output,
                'dtype': 'float64_CPU', 'task': 'volatility12h', 'pool_hash': self.recipe.pool.digest,
                'aggregation': RULES['aggregation'], 'seed_count': len(self.seeds)}

    @property
    def schema_hash(self):
        return fingerprint(self.schema)

    def predict_history(self, frame):
        self._assert_integrity()
        if (frame.feature_names != FEATURE_NAMES or frame.schema_hash != fingerprint(library_schema())
                or frame.bypass_mask != tuple(library_schema()['bypass_mask'])):
            raise ValueError('Neural history must use the exact raw K0 feature order and policy')
        design, support = self.baseline.feature_history(frame)
        predictions = np.full(len(frame.values), np.nan)
        if not support.any():
            return predictions, support
        selected = design.take(support)
        with threadpool_limits(limits=1):
            reference = self.baseline.predictor.predict(selected)
            if all(s['selected_epoch'] == 0 for s in self.seeds):
                predictions[support] = reference  # Preserve epoch-zero bit identity.
                return predictions, support
            X = self.baseline.predictor.scaler.transform(selected)
            outputs = []
            for seed in self.seeds:
                if seed['selected_epoch'] == 0:
                    outputs.append(reference)
                    continue
                codes = _embedding(X, seed['encoder'], self.recipe.config.structures[0])
                if self.recipe.output == 'residual':
                    q = np.log(reference + VOLATILITY_FLOOR) + (
                        codes @ np.asarray(seed['head']['weight']).T).reshape(-1) + seed['head']['bias'][0]
                else:
                    readout = _ridge(seed['trained_readout'])
                    q = readout.predict_log(_encoded_frame(selected, codes, seed['encoding_id']))
                outputs.append(_decode(q)[0])
            values = np.asarray(outputs)
            predictions[support] = _decode(np.mean(np.log(values + VOLATILITY_FLOOR), axis=0))[0]
            predictions[np.flatnonzero(support)[np.all(values == 0, axis=0)]] = 0.
        return predictions, support

    def predict(self, frame):
        return self.predict_history(frame)[0]

    def to_dict(self):
        return {'kind': self.kind, 'version': 1, 'recipe': self.recipe.to_dict(),
                'baseline': self.baseline.to_dict(), 'seeds': copy.deepcopy(list(self.seeds)),
                'receipt': copy.deepcopy(self.receipt)}

    def save(self, path):
        return immutable_record(path, {'model': self.to_dict(), 'digest': self.digest})

    @classmethod
    def from_dict(cls, value):
        if (set(value) != {'kind', 'version', 'recipe', 'baseline', 'seeds', 'receipt'}
                or value['kind'] != 'v7_neural_forward' or value['version'] != 1):
            raise ValueError('Unknown complete neural model schema')
        result = cls(FrozenNeuralRecipe.from_dict(value['recipe']), _joint(value['baseline']),
                     tuple(value['seeds']), value['receipt'])
        if result.digest != fingerprint(value):
            raise ValueError('Neural artifact cannot round-trip its identity')
        return result

    @classmethod
    def load(cls, path):
        path = project_path(path)
        if path.stat().st_size > 4 * 2**20:
            raise ValueError('Complete neural JSON exceeds the bounded artifact size')
        wrapper = json.loads(path.read_text('utf-8'))
        if set(wrapper) != {'model', 'digest'} or wrapper['digest'] != fingerprint(wrapper['model']):
            raise ValueError('Complete neural artifact checksum mismatch')
        return cls.from_dict(wrapper['model'])


def fit_neural_forward(*, ticket_path):
    """Execute one canonical ticket, or restore its already completed artifact.

    No data, fit callback, command, validator, or admission flag is accepted.
    The separate verifier reads original evidence even on a cached completion.
    This entry produces a model binding; it does not adopt or activate it.
    """
    from .neural_forward_admission import _record, verify_completed_artifact
    from .neural_forward_launcher import launch_forward_fit
    from .neural_forward_ticket import ForwardFitTicket, open_existing_authority
    ticket = ForwardFitTicket.load(ticket_path)
    authority = open_existing_authority(ticket.value['authority_directory'])
    record = authority.get(ticket.value['reservation_id'])
    if record['status'] == 'completed':
        certificate = _record(ticket.path.parent / 'launch/certificate.json')
    elif record['status'] == 'reserved':
        certificate = launch_forward_fit(ticket.path)['certificate']
    else:
        raise ValueError('Forward ticket is already attempted; no replacement fit or lease')
    if certificate.get('ticket') != {'path': str(ticket.path), 'digest': ticket.digest}:
        raise ValueError('Completed certificate belongs to another requested ticket')
    result = verify_completed_artifact(certificate, allow_resource_trial=True)
    if result['ticket_digest'] != ticket.digest:
        raise ValueError('Completed artifact differs from the requested ticket')
    return {**result, 'certificate': certificate, 'cached': record['status'] == 'completed'}


def _fit_neural_forward_cold(data, cutoff, recipe, budget):
    """Artificial engineering primitive only; outputs are permanently test-only.

    This is not the production fit entry, a resource verifier or permission to
    train inside ForwardRuntime. A future trusted child needs real admission.
    """
    return _fit_neural_numerical(data, cutoff, recipe, budget, artificial=True)


def _fit_neural_forward_payload(data, cutoff, recipe, budget):
    """Trusted isolated worker's neutral numerical payload, never admission."""
    from .reserved_work_budget import ReservedChannelBudget
    if (not isinstance(budget, ReservedChannelBudget) or budget.test_only
            or budget.executor_mode != 'child'):
        raise TypeError('Neutral forward payloads require the canonical prepaid worker budget')
    budget.check_time()
    return _fit_neural_numerical(data, cutoff, recipe, budget, artificial=False)


def _fit_neural_research_payload(data, cutoff, recipe, budget):
    """Registered research child's same cold kernel; never a forward receipt."""
    from .reserved_work_budget import ReservedChannelBudget
    if (not isinstance(budget, ReservedChannelBudget) or budget.test_only
            or budget.executor_mode != 'child'):
        raise TypeError('Research payloads require the canonical prepaid worker budget')
    budget.check_time()
    return _fit_neural_numerical(data, cutoff, recipe, budget, artificial=False, research=True)


def _fit_neural_numerical(data, cutoff, recipe, budget, *, artificial, research=False):
    if research and artificial:
        raise ValueError('Research worker and artificial receipt modes cannot be combined')
    if not isinstance(recipe, FrozenNeuralRecipe) or not isinstance(budget, ChannelBudget):
        raise TypeError('Engineering cold fits require a frozen recipe and explicit channel budget')
    _ = recipe.pipeline_id
    start, end = recipe.rolling_spec.window(cutoff)
    if data.spec.digest != recipe.rolling_spec.digest:
        raise ValueError('Training protocol differs from the frozen neural recipe')
    if (data.frame.feature_names != FEATURE_NAMES or data.frame.schema_hash != fingerprint(library_schema())
            or data.frame.bypass_mask != tuple(library_schema()['bypass_mask'])):
        raise ValueError('Training input differs from the frozen raw K0 coordinate system')
    data = causal_worker_data(data, end)
    limits = recipe.budget_limits
    if any(budget.counts[k] + amount > budget.limits[k] for k, amount in limits.items() if k != 'seconds'):
        raise ValueError('Insufficient nonrefundable budget for the complete seed ensemble')
    budget.begin()
    began, initial_time = time.monotonic(), budget.spent_seconds
    import torch
    original_threads, original_deterministic = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        baseline = fit_pool(data, end, recipe.pool, budget)
        design, support = baseline.feature_history(data.frame)
        train = data.train_mask(end) & support
        if train.sum() < data.spec.minimum_train_rows:
            raise ValueError('Insufficient mature three-month common neural support')
        frame = design.take(train)
        X = baseline.predictor.scaler.transform(frame)
        reference = baseline.predictor.predict(frame)
        target = data.target[train]
        weights = hour_equal_weights(frame.metadata)
        seeds = []
        for seed in recipe.config.seeds:
            budget.consume('neural_fit')
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                seeds.append(_fit_seed(torch, X, frame, target, reference, weights, recipe, seed, budget))
        budget.check_time()
        receipt = {'version': 'v7-neural-cold-fit-v1', 'pipeline_id': recipe.pipeline_id,
            'baseline_id': baseline.digest, 'training': copy.deepcopy(baseline.receipt['training']),
            'cutoff': end.isoformat(), 'train_start': start.isoformat(), 'fit': 'cold',
            'test_only': True, 'runtime_admitted': False, 'formal_claim': False, 'automatic_adoption': False,
            'torch_version': torch.__version__, 'numpy_version': np.__version__,
            'budget_counts': dict(budget.counts), 'interpretation': 'Artificial model engineering; new ensemble needs S/V/C evaluation'}
        if not artificial:
            receipt.pop('test_only')
            receipt.pop('runtime_admitted')
            receipt.update(version='v7-neural-cold-payload-v2', payload_origin='isolated_forward_worker',
                           admission_required=True,
                           interpretation='Neutral numerical payload; external admission and independent S/V/C required')
            if research:
                receipt.update(version='v7-neural-research-payload-v1',
                    payload_origin='isolated_research_worker',
                    interpretation='S research only; this receipt cannot admit a forward or serving model')
        return NeuralForwardModel(recipe, baseline, tuple(seeds), receipt)
    finally:
        torch.use_deterministic_algorithms(original_deterministic)
        torch.set_num_threads(original_threads)
        budget.charge_time(max(0., initial_time + time.monotonic() - began - budget.spent_seconds))


def _fit_seed(torch, X, frame, target, reference, weights, recipe, seed, budget):
    config, structure = recipe.config, recipe.config.structures[0]
    network = torch.nn.Sequential(torch.nn.Linear(X.shape[1], structure.width), torch.nn.Tanh(),
        torch.nn.Linear(structure.width, structure.representation_dim), torch.nn.Tanh()).double()
    head = torch.nn.Linear(structure.representation_dim, 1).double()
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    optimizer = torch.optim.Adam([*network.parameters(), *head.parameters()], lr=config.learning_rate)
    Xt, y, base, w = (torch.tensor(v, dtype=torch.float64) for v in
                     (X, np.log(target + VOLATILITY_FLOOR), np.log(reference + VOLATILITY_FLOOR), weights))
    initial = copy.deepcopy(network.state_dict())
    best, best_epoch = log_mse(target, reference, weights), 0
    epoch0 = best
    best_state = (copy.deepcopy(initial), copy.deepcopy(head.state_dict()))
    orders, rng = [], np.random.default_rng(seed)
    for epoch in range(1, config.epochs + 1):
        budget.consume('epoch')
        order = rng.permutation(len(X))
        orders.append(fingerprint(order.tolist()))
        for left in range(0, len(order), config.batch_size):
            budget.check_time()
            indices = order[left:left + config.batch_size]
            correction = head(network(Xt[indices])).squeeze(-1)
            loss = (w[indices] * (base[indices] + correction - y[indices]).square()).sum() / w[indices].sum()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite cold neural objective')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            prediction = _decode((base + head(network(Xt)).squeeze(-1)).numpy())[0]
        measured = log_mse(target, prediction, weights)
        if measured < best:
            best, best_epoch = measured, epoch
            best_state = (copy.deepcopy(network.state_dict()), copy.deepcopy(head.state_dict()))
    parameters = {k: v.numpy().tolist() for k, v in best_state[0].items()}
    original = {k: v.numpy().tolist() for k, v in initial.items()}
    result = {'seed': seed, 'selected_epoch': best_epoch, 'encoder': parameters,
        'initial_encoder': original, 'head': {k: v.numpy().tolist() for k, v in best_state[1].items()},
        'data_order_hashes': orders, 'training_loss': best, 'epoch0_loss': epoch0}
    for label, encoder in [('trained', parameters), ('initial', original)]:
        identifier = fingerprint({'pipeline': recipe.pipeline_id, 'seed': seed, 'encoder': encoder, 'label': label})
        embedding = _embedding(X, encoder, structure)
        extended = _encoded_frame(frame, embedding, identifier)
        budget.consume('solve')
        readout = fit_ridge(extended, target, ridge=recipe.rolling_spec.ridge, sample_weight=weights,
                            provenance={'pipeline': recipe.pipeline_id, 'seed': seed, 'support': fingerprint(frame.metadata.sample_id.tolist())})
        result[label + '_readout'] = readout.to_dict()
        result['encoding_id' if label == 'trained' else 'initial_encoding_id'] = identifier
    budget.check_time()
    return result
