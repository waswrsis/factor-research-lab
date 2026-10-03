"""Bounded research training; learned representations and fixed linear probes.

No label, maturity, symbol or time metadata enters a model. Scalers are fitted
on training inputs/observed training targets only. Event heads parameterize a
joint first-hit distribution with hazards and conditional directions.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from torch import nn
from torch.nn import functional as F

from factorlab.protocol import fingerprint, project_path, utc

from .contracts import RepresentationSpec, TaskDataset, TaskSpec, TrainingSpec

VOLATILITY_FLOOR = 1e-8
FORMAT_VERSION = 'discovery-learning-v1'
SELECTION_AGGREGATION = 'equal_symbol_within_hour_then_equal_hour'


def _finite(value, context):
    ok = torch.isfinite(value).all().item() if torch.is_tensor(value) else np.isfinite(value).all()
    if not ok:
        raise FloatingPointError(f'Nonfinite {context}; training/prediction invalid')


def _check_data(data, *, allow_empty=False):
    _finite(data.x, 'inputs')
    if ((not allow_empty and len(data.x) == 0)
            or data.x.shape[1] == 0 or data.x.shape[2] == 0):
        raise ValueError('A nonempty input dataset is required')
    for task in data.tasks:
        y = data.targets[task.name][data.task_mask(task.name)]
        _finite(y, f'targets for {task.name}')
        if task.kind == 'volatility' and (y < 0).any():
            raise ValueError('Volatility targets must be nonnegative')
        if task.kind == 'first_event' and (
            (y != np.floor(y)).any() or (y < 0).any() or (y >= task.output_size).any()
        ):
            raise ValueError('Invalid first-event category')


def _summary(x):
    """Last, mean, standard deviation, extrema and endpoint change per input."""
    return np.concatenate((x[:, -1], x.mean(1), x.std(1), x.min(1), x.max(1),
                           x[:, -1] - x[:, 0]), axis=1)


def summary_features(data: TaskDataset) -> np.ndarray:
    _finite(data.x, 'summary inputs')
    result = _summary(data.x.astype(np.float64))
    _finite(result, 'summary features')
    return result


def _scale(x):
    mean, std = x.mean(0), x.std(0)
    std = np.where(std > 1e-8, std, 1.0)
    _finite(mean, 'input means')
    _finite(std, 'input scales')
    return mean, std


def _task_setup(train):
    scales, active = {}, []
    for task in train.tasks:
        mask = train.task_mask(task.name)
        if not mask.any():
            continue
        active.append(task)
        if task.kind != 'first_event':
            y = train.targets[task.name][mask].astype(np.float64)
            if task.kind == 'volatility':
                y = np.log(y + VOLATILITY_FLOOR)
            scales[task.name] = [float(y.mean()), float(y.std()) if y.std() > 1e-8 else 1.0]
    if not active:
        raise ValueError('No observed training task has support')
    return active, scales


def _target(data, task, scales):
    mask = data.task_mask(task.name)
    # Masked NaN labels are allowed by the dataset contract and never used.
    y = np.zeros(len(data.x), dtype=np.float64)
    y[mask] = data.targets[task.name][mask]
    if task.kind == 'first_event':
        return y.astype(np.int64), mask
    if task.kind == 'volatility':
        y[mask] = np.log(y[mask] + VOLATILITY_FLOOR)
    mean, std = scales[task.name]
    y[mask] = (y[mask] - mean) / std
    _finite(y, f'transformed target {task.name}')
    return y, mask


def event_log_probabilities(logits):
    """Stable joint log probabilities: 0=no hit, 1+2*b=up, 2+2*b=down.

    Logits contain K hazards followed by K conditional up-direction logits.
    This uses the complete no-event likelihood and survival before the hit.
    """
    if logits.ndim != 2 or logits.shape[1] == 0 or logits.shape[1] % 2:
        raise ValueError('Event logits must have shape [N, 2*K]')
    _finite(logits, 'event logits')
    k = logits.shape[1] // 2
    h, d = logits[:, :k], logits[:, k:]
    if torch.is_tensor(logits):
        survival = F.logsigmoid(-h)
        previous = torch.cat((torch.zeros_like(survival[:, :1]),
                              survival.cumsum(1)[:, :-1]), 1)
        hit = previous + F.logsigmoid(h)
        pairs = torch.stack((hit + F.logsigmoid(d), hit + F.logsigmoid(-d)), 2)
        return torch.cat((survival.sum(1, keepdim=True), pairs.flatten(1)), 1)
    survival = -np.logaddexp(0.0, h)
    previous = np.concatenate((np.zeros_like(survival[:, :1]),
                               survival.cumsum(1)[:, :-1]), 1)
    hit = previous - np.logaddexp(0.0, -h)
    pairs = np.stack((hit - np.logaddexp(0.0, -d), hit - np.logaddexp(0.0, d)), 2)
    return np.concatenate((survival.sum(1, keepdims=True), pairs.reshape(len(logits), 2 * k)), 1)


def event_nll(logits, categories):
    """Per-row full NLL; callers select observed rows before calling."""
    logp = event_log_probabilities(logits)
    if torch.is_tensor(logits):
        return -logp.gather(1, categories.long()[:, None]).squeeze(1)
    return -logp[np.arange(len(logits)), np.asarray(categories, dtype=np.int64)]


def _event_objective(beta, design, y, k, penalty):
    beta = beta.reshape(design.shape[1], 2 * k)
    logits = design @ beta
    loss = float(event_nll(logits, y).mean() + penalty * np.square(beta).sum())
    bins = (y - 1) // 2
    event = y > 0
    at_risk = (~event[:, None]) | (np.arange(k)[None, :] <= bins[:, None])
    hit = event[:, None] & (np.arange(k)[None, :] == bins[:, None])
    up = ((y % 2) == 1)[:, None]
    grad_logits = np.concatenate(((expit(logits[:, :k]) - hit) * at_risk,
                                  (expit(logits[:, k:]) - up) * hit), axis=1)
    gradient = design.T @ grad_logits / len(y) + 2 * penalty * beta
    _finite(loss, 'linear event objective')
    _finite(gradient, 'linear event gradient')
    return loss, gradient.ravel()


class _Module(nn.Module):
    def __init__(self, spec, features, lookback):
        super().__init__()
        selected = list(spec.features or features)
        if not set(selected) <= set(features):
            raise ValueError(f'Unknown features in module {spec.name}')
        self.register_buffer('indices', torch.tensor([features.index(x) for x in selected]))
        self.kind, self.scale = spec.kind, min(spec.scale, lookback)
        f = len(selected)
        if spec.kind == 'temporal':
            # Each channel has learned historical weights; mixing learns interactions.
            self.time_logits = nn.Parameter(torch.zeros(f, self.scale))
        input_size = 6 * f if spec.kind == 'summary' else self.scale * f if spec.kind == 'mlp' else f
        if spec.kind == 'program':
            from .operator_program import ProgramModule

            # Programs see only the declared feature subset and trailing scale
            # of training-standardized input. Their scalar roots are projected
            # through the same trainable MLP used by the existing modules.
            self.program = ProgramModule(spec.program, tuple(selected), lookback=self.scale)
            input_size = self.program.output_dim
        layers = [nn.Linear(input_size, spec.width), nn.Tanh()]
        for _ in range(spec.depth - 1):
            layers.extend((nn.Linear(spec.width, spec.width), nn.Tanh()))
        self.network = nn.Sequential(*layers, nn.Linear(spec.width, spec.dim))

    def forward(self, x):
        x = x[:, -self.scale:, self.indices]
        if self.kind == 'summary':
            x = torch.cat((x[:, -1], x.mean(1), x.std(1, correction=0),
                           x.amin(1), x.amax(1), x[:, -1] - x[:, 0]), 1)
        elif self.kind == 'mlp':
            x = x.flatten(1)
        elif self.kind == 'program':
            x = self.program(x)
            _finite(x, 'operator program outputs')
        else:
            x = (x.transpose(1, 2) * self.time_logits.softmax(1)).sum(2)
        return self.network(x)


class _Network(nn.Module):
    def __init__(self, architecture, features, lookback, tasks):
        super().__init__()
        self.modules_by_name = nn.ModuleList([_Module(m, features, lookback)
                                             for m in architecture.modules])
        self.combination = architecture.combination
        self.dim = sum(m.dim for m in architecture.modules)
        if self.combination == 'gated':
            self.gate = nn.Linear(self.dim, len(architecture.modules))
            output_dim = architecture.modules[0].dim
        else:
            output_dim = self.dim
        # Integer keys allow valid Python task names that are Module attribute names.
        self.heads = nn.ModuleList([nn.Linear(output_dim, 2 * len(t.event_bins)
                                               if t.kind == 'first_event' else 1) for t in tasks])
        self.task_names = [t.name for t in tasks]

    def encode(self, x):
        return torch.cat([module(x) for module in self.modules_by_name], 1)

    def forward(self, x):
        z = self.encode(x)
        if self.combination == 'gated':
            weights = self.gate(z).softmax(1)
            z = (z.reshape(len(z), len(self.modules_by_name), -1) * weights[:, :, None]).sum(1)
        return {name: head(z) for name, head in zip(self.task_names, self.heads)}


def _norm(values):
    return float(np.sqrt(sum(float(torch.sum(v.detach().double() ** 2)) for v in values)))


def _tensor_state(network):
    return {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}


def _decode(raw, tasks, scales):
    output = {}
    for task in tasks:
        value = raw[task.name]
        _finite(value, f'raw prediction {task.name}')
        if task.kind == 'first_event':
            logp = event_log_probabilities(value)
            value = np.exp(logp - logsumexp(logp, axis=1, keepdims=True))
        else:
            mean, std = scales[task.name]
            value = value.reshape(-1) * std + mean
            if task.kind == 'volatility':
                with np.errstate(over='ignore'):
                    value = np.maximum(np.exp(value) - VOLATILITY_FLOOR, 0)
        _finite(value, f'prediction {task.name}')
        output[task.name] = value
    return output


class FittedResearchModel:
    """Serializable fitted model. Missing training heads are explicitly unavailable."""
    def __init__(self, metadata, state):
        self.metadata, self.state = metadata, state
        self.diagnostics = metadata['diagnostics']
        self.schema = metadata['schema']
        self.tasks = tuple(TaskSpec(**{**t, 'event_bins': tuple(t['event_bins'])})
                           for t in metadata['active_tasks'])
        self.network = None
        if metadata['mode'] == 'learned':
            self.network = _Network(RepresentationSpec.from_dict(metadata['architecture']),
                                    tuple(metadata['features']), metadata['lookback'], self.tasks)
            self.network.load_state_dict(state)
            self.network.eval()

    @property
    def digest(self):
        h = hashlib.sha256(fingerprint(self.metadata).encode())
        for key, value in sorted(self.state.items()):
            array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
            h.update(key.encode())
            h.update(str(array.dtype).encode())
            h.update(str(array.shape).encode())
            h.update(array.tobytes())
        return h.hexdigest()

    @property
    def parameter_count(self):
        return self.diagnostics['parameter_count']

    @property
    def program_parameters(self):
        """Bounded indicator values at the selected fitted checkpoint."""
        if self.network is None:
            return {}
        architecture = RepresentationSpec.from_dict(self.metadata['architecture'])
        return {spec.name: {name: float(value.detach().cpu())
                            for name, value in module.program.parameter_values().items()}
                for spec, module in zip(architecture.modules, self.network.modules_by_name)
                if spec.kind == 'program'}

    def _check(self, data):
        if data.schema != self.schema:
            raise ValueError('Dataset schema differs from fitted model')
        _check_data(data, allow_empty=True)

    def encode(self, data: TaskDataset):
        self._check(data)
        if len(data.x) == 0:
            dimension = self.network.dim if self.network is not None else self.metadata['encoded_dim']
            return np.empty((0, dimension), dtype=np.float64)
        if self.metadata['mode'] == 'probe':
            raise ValueError('Probe requires external representations, use predict_encoded')
        if self.network is None:
            z = (summary_features(data) - np.asarray(self.metadata['input_mean'])) / np.asarray(
                self.metadata['input_std'])
        else:
            z_parts = []
            for start in range(0, len(data.x), self.metadata['settings']['batch_size']):
                x = self._input(data.x[start:start + self.metadata['settings']['batch_size']])
                with torch.no_grad():
                    z_parts.append(self.network.encode(x).cpu().numpy())
            z = np.concatenate(z_parts)
        _finite(z, 'encoded representations')
        return z

    def _input(self, x):
        x = (x.astype(np.float64) - np.asarray(self.metadata['input_mean'])) / np.asarray(
            self.metadata['input_std'])
        _finite(x, 'scaled inputs')
        value = torch.as_tensor(x, dtype=torch.float32)
        _finite(value, 'float32 scaled inputs')
        return value

    def predict(self, data: TaskDataset):
        self._check(data)
        if len(data.x) == 0:
            return {task.name: np.empty((0, task.output_size) if task.kind == 'first_event' else (0,),
                                       dtype=np.float64) for task in self.tasks}
        if self.network is None:
            z = self.encode(data)
            raw = self._linear_raw(z)
        else:
            chunks = {task.name: [] for task in self.tasks}
            for start in range(0, len(data.x), self.metadata['settings']['batch_size']):
                x = self._input(data.x[start:start + self.metadata['settings']['batch_size']])
                with torch.no_grad():
                    for name, value in self.network(x).items():
                        chunks[name].append(value.cpu().numpy())
            raw = {name: np.concatenate(parts).astype(np.float64) for name, parts in chunks.items()}
        return _decode(raw, self.tasks, self.metadata['target_scales'])

    def _linear_raw(self, z):
        design = np.column_stack((z, np.ones(len(z))))
        return {t.name: design @ np.asarray(self.state[t.name]) for t in self.tasks}

    def save(self, path: str | Path):
        path = project_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {k: v if torch.is_tensor(v) else torch.from_numpy(np.asarray(v).copy())
                 for k, v in self.state.items()}
        payload = {'metadata_json': json.dumps(self.metadata, allow_nan=False),
                   'state': state, 'sha256': self.digest}
        temp = path.with_suffix(path.suffix + '.tmp')
        torch.save(payload, temp)
        temp.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path):
        payload = torch.load(project_path(path), map_location='cpu', weights_only=True)
        metadata = json.loads(payload['metadata_json'])
        if metadata.get('format') != FORMAT_VERSION:
            raise ValueError('Unsupported research model format')
        state = payload['state']
        if metadata['mode'] != 'learned':
            state = {k: v.numpy() for k, v in state.items()}
        model_class = FittedProbe if metadata['mode'] == 'probe' else FittedResearchModel
        model = model_class(metadata, state)
        if model.digest != payload['sha256']:
            raise ValueError('Research model hash mismatch')
        return model


class FittedProbe(FittedResearchModel):
    """Linear head on frozen externally supplied representations."""
    def predict_encoded(self, z, data: TaskDataset):
        self._check(data)
        z = np.asarray(z, dtype=np.float64)
        if z.shape != (len(data.x), self.metadata['encoded_dim']):
            raise ValueError('Encoded probe input shape mismatch')
        _finite(z, 'probe inputs')
        z = (z - np.asarray(self.metadata['input_mean'])) / np.asarray(self.metadata['input_std'])
        return _decode(self._linear_raw(z), self.tasks, self.metadata['target_scales'])


def _metadata(train, development, active, scales, settings, architecture, mode, mean, std):
    metadata = {'format': FORMAT_VERSION, 'schema': train.schema, 'mode': mode,
            'features': list(train.features), 'lookback': train.x.shape[1],
            'active_tasks': [asdict(t) for t in active], 'target_scales': scales,
            'settings': asdict(settings),
            'architecture': asdict(architecture) if architecture is not None else None,
            'input_mean': np.asarray(mean).tolist(), 'input_std': np.asarray(std).tolist(),
            'volatility_floor': VOLATILITY_FLOOR,
            'diagnostics': {'train_support': {t.name: int(train.task_mask(t.name).sum())
                                              for t in train.tasks},
                            'development_support': {t.name: int(development.task_mask(t.name).sum())
                                                    for t in train.tasks},
                            'unsupported_tasks': [t.name for t in train.tasks if t not in active],
                            'global_optimum_claimed': False}}
    event_support = {}
    for task in active:
        if task.kind != 'first_event':
            continue
        labels = train.targets[task.name][train.task_mask(task.name)].astype(np.int64)
        bins = (labels - 1) // 2
        event_support[task.name] = {
            'at_risk_by_bin': [int(((labels == 0) | (bins >= b)).sum())
                               for b in range(len(task.event_bins))],
            'up_by_bin': [int((labels == 1 + 2 * b).sum()) for b in range(len(task.event_bins))],
            'down_by_bin': [int((labels == 2 + 2 * b).sum()) for b in range(len(task.event_bins))],
            'complete_no_event': int((labels == 0).sum())}
    metadata['diagnostics']['train_event_support'] = event_support
    # Conservative feature bounds include every row actually supplied to input
    # scaling/prediction. Label bounds include only supported, observed heads.
    for prefix, data in (('fit', train), ('development', development)):
        metadata[f'{prefix}_feature_cutoff'] = utc(data.meta.available_at.max()).isoformat()
        task_cutoffs = {}
        for task in active:
            mask = data.task_mask(task.name)
            if mask.any():
                timestamp = data.maturity[task.name][mask].max()
                task_cutoffs[task.name] = utc(str(timestamp)).isoformat()
        metadata[f'{prefix}_label_cutoffs_by_task'] = task_cutoffs
        metadata[f'{prefix}_label_cutoff'] = max(task_cutoffs.values()) if task_cutoffs else None
    return metadata


class _NumericalStop(Exception):
    pass


def _inherit_weights(network, metadata, initial_model):
    """Copy complete semantic modules, never similarly shaped latent slices.

    Optimizer state is deliberately fresh. Head weights require the entire
    ordered latent basis and the target transform to retain their meanings.
    """
    parameters = dict(network.named_parameters())
    report = {'requested': initial_model is not None, 'source_digest': None,
              'copied_parameters': 0, 'total_parameters': sum(p.numel() for p in parameters.values()),
              'coverage': 0.0, 'copied_tensors': [], 'skipped': {},
              'optimizer_state': 'fresh'}
    if initial_model is None:
        report['skipped']['all'] = 'cold_start'
        return report
    if not isinstance(initial_model, FittedResearchModel):
        raise TypeError('initial_model must be one fitted research model')
    report['source_digest'] = initial_model.digest
    source = initial_model.metadata
    if source['mode'] != 'learned' or source['schema'] != metadata['schema']:
        report['skipped']['all'] = 'source_mode_or_input_task_schema_changed'
        return report
    for key in ('fit_feature_cutoff', 'fit_label_cutoff',
                'development_feature_cutoff', 'development_label_cutoff'):
        if source.get(key) is not None and (
            metadata.get(key) is None or utc(source[key]) > utc(metadata[key])
        ):
            raise ValueError('Initial model contains later fit or development evidence')
    if not (np.array_equal(source['input_mean'], metadata['input_mean'])
            and np.array_equal(source['input_std'], metadata['input_std'])):
        report['skipped']['all'] = 'input_standardization_changed'
        return report
    old_arch = RepresentationSpec.from_dict(source['architecture'])
    new_arch = RepresentationSpec.from_dict(metadata['architecture'])
    previous = {m.name: (i, m) for i, m in enumerate(old_arch.modules)}
    copied_modules = set()

    def semantics(module):
        return (module.kind, module.dim, module.width, module.depth,
                min(module.scale, metadata['lookback']),
                tuple(module.features or metadata['features']),
                fingerprint(module.program))

    def copy_tensor(target_name, source_name):
        target, value = parameters[target_name], initial_model.state[source_name]
        if target.shape != value.shape:
            report['skipped'][target_name] = 'tensor_shape_changed'
            return
        _finite(value, 'inherited weights')
        with torch.no_grad():
            target.copy_(value.to(device=target.device, dtype=target.dtype))
        report['copied_tensors'].append(target_name)
        report['copied_parameters'] += target.numel()

    for index, module in enumerate(new_arch.modules):
        old = previous.get(module.name)
        prefix = f'modules_by_name.{index}.'
        if old is None or semantics(old[1]) != semantics(module):
            report['skipped'][module.name] = 'new_module_or_module_semantics_changed'
            continue
        for name in parameters:
            if name.startswith(prefix):
                copy_tensor(name, f'modules_by_name.{old[0]}.' + name[len(prefix):])
        copied_modules.add(module.name)
    same_basis = (old_arch.combination == new_arch.combination
                  and len(old_arch.modules) == len(new_arch.modules)
                  and tuple(m.name for m in old_arch.modules) == tuple(m.name for m in new_arch.modules)
                  and len(copied_modules) == len(new_arch.modules))
    old_tasks = {t.name: (i, t) for i, t in enumerate(initial_model.tasks)}
    for index, task in enumerate(metadata['active_tasks']):
        old = old_tasks.get(task['name'])
        prefix = f'heads.{index}.'
        if (not same_basis or old is None or asdict(old[1]) != task
            or source['target_scales'].get(task['name']) != metadata['target_scales'].get(task['name'])):
            report['skipped'][prefix] = 'latent_basis_or_target_semantics_changed'
            continue
        for name in parameters:
            if name.startswith(prefix):
                copy_tensor(name, f'heads.{old[0]}.' + name[len(prefix):])
    if new_arch.combination == 'gated':
        for name in parameters:
            if name.startswith('gate.'):
                if same_basis:
                    copy_tensor(name, name)
                else:
                    report['skipped'][name] = 'latent_basis_changed'
    report['coverage'] = report['copied_parameters'] / report['total_parameters']
    return report


def _linear_fit(z, train, development, settings, mode, started, dev_z):
    if settings.selection_task is not None:
        raise ValueError('selection_task requires learned training; linear/probe fitting has no joint checkpoint')
    active, scales = _task_setup(train)
    mean, std = _scale(z)
    z = (z - mean) / std
    _finite(z, 'scaled linear features')
    design = np.column_stack((z, np.ones(len(z))))
    metadata = _metadata(train, development, active, scales, settings, None, mode, mean, std)
    metadata['encoded_dim'] = z.shape[1]
    params = sum(design.shape[1] * (2 * len(t.event_bins) if t.kind == 'first_event' else 1)
                 for t in active)
    if params > settings.max_parameters:
        raise ValueError('Parameter budget exceeded before fitting')
    diagnostics = metadata['diagnostics']
    diagnostics.update(parameter_count=params, history=[], best_epoch=0,
                       stop_reason='closed_form', optimizer_results={},
                       checkpoint_selection='closed_form_or_per_head_development_nll')
    state, gradients, init_norm = {}, [], 0.0
    # Equal task means plus L2 over EVERY coefficient/bias: independent head
    # solves use H*l2, exactly matching that jointly regularized objective.
    penalty = settings.l2 * len(active)
    for task in active:
        if time.monotonic() - started >= settings.max_seconds:
            raise TimeoutError('Linear training budget exhausted before all heads fitted')
        y, mask = _target(train, task, scales)
        a, y = design[mask], y[mask]
        if task.kind == 'first_event':
            k = len(task.event_bins)
            initial = np.zeros(a.shape[1] * 2 * k)
            dev_target, dev_mask = _target(development, task, scales)
            dev_design = np.column_stack(((dev_z - mean) / std, np.ones(len(dev_z))))[dev_mask]
            iterate, best, records = [initial.copy()], [initial.copy()], []
            selection_loss, patience_reference, stale, reason = [np.inf], [np.inf], [0], [None]
            best_iteration = [0]

            def callback(beta, iterate=iterate, a=a, y=y, k=k, dev_design=dev_design,
                         dev_target=dev_target, dev_mask=dev_mask, records=records,
                         selection_loss=selection_loss, best=best, best_iteration=best_iteration,
                         stale=stale, reason=reason, patience_reference=patience_reference):
                previous = iterate[0]
                iterate[0] = beta.copy()
                objective, residual = _event_objective(beta, a, y, k, penalty)
                value = float(event_nll(dev_design @ beta.reshape(a.shape[1], 2 * k),
                                       dev_target[dev_mask]).mean()) if dev_mask.any() else None
                records.append({'iteration': len(records) + 1,
                                'preclip_gradient_norm': float(np.linalg.norm(residual)),
                                'parameter_norm': float(np.linalg.norm(beta)),
                                'actual_update_norm': float(np.linalg.norm(beta - previous)),
                                'training_regularized_loss': objective,
                                'development_loss': value})
                if value is None or value < selection_loss[0]:
                    best[0] = beta.copy()
                    selection_loss[0] = value if value is not None else np.inf
                    best_iteration[0] = len(records)
                if value is None or value < patience_reference[0] - settings.min_delta:
                    patience_reference[0] = value if value is not None else np.inf
                    stale[0] = 0
                else:
                    stale[0] += 1
                if time.monotonic() - started >= settings.max_seconds:
                    reason[0] = 'time_budget'
                    raise _NumericalStop()
                if dev_mask.any() and stale[0] >= settings.patience:
                    reason[0] = 'development_patience'
                    raise _NumericalStop()

            result = None
            try:
                result = minimize(_event_objective, initial, args=(a, y, k, penalty), jac=True,
                                  method='L-BFGS-B', callback=callback,
                                  options={'maxiter': settings.epochs, 'gtol': 1e-7, 'ftol': 1e-12})
            except _NumericalStop:
                pass
            if reason[0] == 'time_budget':
                raise TimeoutError('Linear event fitting exhausted declared time budget')
            if result is not None and not result.success:
                message = str(result.message)
                # L-BFGS-B status 1 is a declared numerical resource limit, not
                # status 2's line-search/rounding failure. Never promote a failed
                # solver merely because an earlier finite iterate exists.
                if int(result.status) != 1:
                    raise RuntimeError(f'Event numerical solver failed: status={result.status}; {message}')
                upper_message = message.upper()
                if 'ITERATION' in upper_message:
                    reason[0] = 'iteration_budget'
                elif 'EVALUATION' in upper_message:
                    reason[0] = 'evaluation_budget'
                else:
                    raise RuntimeError(f'Unrecognized event solver limit: status={result.status}; {message}')
            selected = best[0] if records else result.x
            beta = selected.reshape(a.shape[1], 2 * k)
            _, gradient = _event_objective(selected, a, y, k, penalty)
            init_gradient = _event_objective(initial, a, y, k, penalty)[1]
            stop = reason[0] or 'numerical_tolerance'
            diagnostics['optimizer_results'][task.name] = {
                'optimizer_success': bool(result.success) if result is not None else False,
                'status': int(result.status) if result is not None else None,
                'message': str(result.message) if result is not None else stop,
                'iterations': int(result.nit) if result is not None else len(records),
                'stop_reason': stop, 'best_iteration': best_iteration[0], 'history': records,
                'gradient_residual': float(np.linalg.norm(gradient)) / len(active)}
            diagnostics['stop_reason'] = stop
        else:
            gram = a.T @ a / len(a) + penalty * np.eye(a.shape[1])
            rhs = a.T @ y / len(a)
            beta = np.linalg.solve(gram, rhs)
            gradient = 2 * (a.T @ (a @ beta - y) / len(a) + penalty * beta)
            init_gradient = -2 * rhs
            diagnostics['optimizer_results'][task.name] = {
                'gradient_residual': float(np.linalg.norm(gradient)) / len(active),
                'method': 'positive_ridge_all_coefficients_and_bias'}
        _finite(beta, f'linear coefficients {task.name}')
        state[task.name] = beta
        gradients.append(float(np.square(gradient / len(active)).sum()))
        init_norm += float(np.square(init_gradient / len(active)).sum())
    diagnostics['gradient_residual'] = float(np.sqrt(sum(gradients)))
    pnorm = float(np.sqrt(sum(float(np.square(v).sum()) for v in state.values())))
    model_class = FittedProbe if mode == 'probe' else FittedResearchModel
    model = model_class(metadata, state)
    dz = (dev_z - mean) / std
    dev_raw = model._linear_raw(dz)
    losses = _numpy_losses(dev_raw, development, active, scales)
    diagnostics['history'].append({'epoch': 0, 'preclip_gradient_norm': float(np.sqrt(init_norm)),
                                   'parameter_norm': pnorm, 'actual_update_norm': pnorm,
                                   'development_task_losses': losses})
    diagnostics['development_loss'] = float(np.mean(list(losses.values()))) if losses else None
    diagnostics['elapsed_seconds'] = time.monotonic() - started
    if diagnostics['elapsed_seconds'] > settings.max_seconds:
        raise TimeoutError('Linear fitting exceeded declared time budget')
    return model


def _numpy_losses(raw, data, active, scales):
    losses = {}
    for task in active:
        _finite(raw[task.name], f'development head {task.name}')
        y, mask = _target(data, task, scales)
        if not mask.any():
            continue
        value = (event_nll(raw[task.name][mask], y[mask]) if task.kind == 'first_event'
                 else (raw[task.name].reshape(-1)[mask] - y[mask]) ** 2)
        _finite(value, f'task loss {task.name}')
        losses[task.name] = float(value.mean())
    return losses


def _selection_task(train, development, settings):
    """A requested predictive head must be usable; never fall back to auxiliaries."""
    if settings.selection_task is None:
        return None
    task = next((t for t in train.tasks if t.name == settings.selection_task), None)
    if task is None:
        raise ValueError(f'Unknown selection_task: {settings.selection_task}')
    if not task.predictive:
        raise ValueError('selection_task must be predictive')
    for name, data in (('training', train), ('development', development)):
        if not data.task_mask(task.name).any():
            raise ValueError(f'selection_task {task.name} has no observed {name} support')
    return task


def _selection_losses(raw, data, active, scales):
    """Original task scores using the benchmark's symbol/hour weighting.

    Duplicate observations for a symbol within a UTC hour first share that
    symbol's weight. Symbols then share each hour, and hours share the result.
    This differs from the standardized row means used by the training objective.
    """
    from .scoring import per_sample_loss

    predictions = _decode(raw, active, scales)
    losses = {}
    for task in active:
        mask = data.task_mask(task.name)
        if not mask.any():
            continue
        rows = data.meta.loc[mask, ['available_at', 'symbol']].copy()
        rows['hour'] = pd.to_datetime(rows.available_at, utc=True).dt.floor('h')
        rows['loss'] = per_sample_loss(task, data.targets[task.name][mask],
                                       predictions[task.name][mask])
        if rows.symbol.isna().any():
            raise ValueError('Checkpoint selection requires symbol metadata on observed rows')
        hourly = rows.groupby(['hour', 'symbol'], observed=True).loss.mean()
        losses[task.name] = float(hourly.groupby(level='hour').mean().mean())
        _finite(losses[task.name], f'selection task loss {task.name}')
    return losses


def fit(train: TaskDataset, development: TaskDataset,
        architecture: RepresentationSpec | None, settings: TrainingSpec,
        device='cpu', *, initial_model: FittedResearchModel | None = None,
        retain_initial_scalers: bool = False) -> FittedResearchModel:
    started = time.monotonic()
    _check_data(train)
    _check_data(development)
    if train.schema != development.schema:
        raise ValueError('Training and development schemas differ')
    if type(retain_initial_scalers) is not bool:
        raise ValueError('retain_initial_scalers must be boolean')
    if architecture is None:
        result = _linear_fit(summary_features(train), train, development, settings,
                             'linear', started, summary_features(development))
        if initial_model is not None:
            result.diagnostics['weight_transfer'] = {
                'requested': True, 'coverage': 0.0, 'copied_parameters': 0,
                'source_digest': initial_model.digest, 'skipped': {'all': 'linear_closed_form'},
                'optimizer_state': 'fresh'}
        return result
    selection_task = _selection_task(train, development, settings)
    active, scales = _task_setup(train)
    mean, std = _scale(train.x.astype(np.float64).reshape(-1, train.x.shape[2]))
    scaler_transfer = {'scaler_policy': 'fit_current_training_inputs_and_targets',
                       'retained_target_tasks': []}
    if (retain_initial_scalers and initial_model is not None
        and isinstance(initial_model, FittedResearchModel)
        and initial_model.metadata['mode'] == 'learned' and initial_model.schema == train.schema):
        source = initial_model.metadata
        mean, std = np.asarray(source['input_mean']), np.asarray(source['input_std'])
        if mean.shape != (train.x.shape[2],) or std.shape != mean.shape or (std <= 0).any():
            raise ValueError('Initial input standardization is invalid')
        _finite(mean, 'retained input mean')
        _finite(std, 'retained input scales')
        for task in active:
            if task in initial_model.tasks and task.name in source['target_scales']:
                scale = source['target_scales'][task.name]
                if len(scale) != 2 or not np.isfinite(scale).all() or scale[1] <= 0:
                    raise ValueError('Initial target standardization is invalid')
                scales[task.name] = list(scale)
                scaler_transfer['retained_target_tasks'].append(task.name)
        scaler_transfer.update(scaler_policy='retain_compatible_historical_input_and_target_scalers',
                               scaler_source_digest=initial_model.digest,
                               scaler_fit_feature_cutoff=source['fit_feature_cutoff'],
                               scaler_fit_label_cutoff=source['fit_label_cutoff'])
    metadata = _metadata(train, development, active, scales, settings, architecture,
                         'learned', mean, std)
    torch.manual_seed(settings.seed)
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('Requested CUDA device is unavailable')
    network = _Network(architecture, train.features, train.x.shape[1], active).to(device)
    count = sum(p.numel() for p in network.parameters())
    if count > settings.max_parameters:
        raise ValueError('Parameter budget exceeded before training')
    diagnostics = metadata['diagnostics']
    diagnostics.update(parameter_count=count, history=[], stop_reason='epoch_budget', best_epoch=-1)
    diagnostics['weight_transfer'] = _inherit_weights(network, metadata, initial_model)
    diagnostics['weight_transfer'].update(scaler_transfer)
    program_groups = {
        module.name: [(name, parameter) for name, parameter in network.named_parameters()
                      if name.startswith(f'modules_by_name.{index}.program.')]
        for index, module in enumerate(architecture.modules) if module.kind == 'program'}
    initial_program_state = {name: parameter.detach().clone()
                             for group in program_groups.values() for name, parameter in group}
    if program_groups:
        diagnostics['programs'] = {
            module.name: {'program_digest': fingerprint(module.program),
                          'trainable_parameters': sum(p.numel() for _, p in program_groups[module.name]),
                          'input_semantics': 'training_standardized_selected_features',
                          'selected_features': list(module.features or train.features),
                          'effective_scale': min(module.scale, train.x.shape[1])}
            for module in architecture.modules if module.kind == 'program'}
    train_x = torch.as_tensor((train.x.astype(np.float64) - mean) / std,
                              dtype=torch.float32, device=device)
    dev_x = torch.as_tensor((development.x.astype(np.float64) - mean) / std,
                            dtype=torch.float32, device=device)
    _finite(train_x, 'training float32 inputs')
    _finite(dev_x, 'development float32 inputs')
    targets, masks = {}, {}
    for task in active:
        y, mask = _target(train, task, scales)
        targets[task.name] = torch.as_tensor(y, dtype=torch.long if task.kind == 'first_event'
                                            else torch.float32, device=device)
        masks[task.name] = torch.as_tensor(mask, dtype=torch.bool, device=device)
    if not any(development.task_mask(t.name).any() for t in active):
        raise ValueError('Development data has no supported task for checkpoint selection')
    optimizer = torch.optim.Adam(network.parameters(), lr=settings.learning_rate)
    rng = np.random.default_rng(settings.seed)
    best_loss, patience_reference, best_state, stale, steps = np.inf, np.inf, None, 0, 0
    for epoch in range(settings.epochs):
        if time.monotonic() - started >= settings.max_seconds:
            diagnostics['stop_reason'] = 'time_budget'
            break
        network.train()
        order = rng.permutation(len(train_x))
        gn, pn, un = [], [], []
        program_history = {name: {'preclip_gradient_norm': [], 'actual_update_norm': []}
                           for name in program_groups}
        interrupted = False
        for start in range(0, len(order), settings.batch_size):
            if time.monotonic() - started >= settings.max_seconds:
                interrupted = True
                diagnostics['stop_reason'] = 'time_budget'
                break
            ix = torch.as_tensor(order[start:start + settings.batch_size], device=device)
            raw = network(train_x[ix])
            for name, value in raw.items():
                _finite(value, f'training head {name}')
            losses = []
            for task in active:
                mask = masks[task.name][ix]
                if mask.any():
                    y = targets[task.name][ix][mask]
                    value = (event_nll(raw[task.name][mask], y) if task.kind == 'first_event'
                             else (raw[task.name].reshape(-1)[mask] - y) ** 2)
                    # Unbiased batch estimate of each globally supported task mean,
                    # then equal weighting across heads (not across labels).
                    losses.append(value.sum() * len(train_x) /
                                  (len(ix) * diagnostics['train_support'][task.name]))
            objective = sum(losses) / len(active) + settings.l2 * sum(
                p.square().sum() for p in network.parameters())
            _finite(objective, 'training objective')
            optimizer.zero_grad()
            objective.backward()
            gradients = [p.grad for p in network.parameters() if p.grad is not None]
            for gradient in gradients:
                _finite(gradient, 'preclip gradients')
            gn.append(_norm(gradients))
            for name, group in program_groups.items():
                program_history[name]['preclip_gradient_norm'].append(
                    _norm([p.grad for _, p in group if p.grad is not None]))
            program_before = {name: parameter.detach().clone()
                              for group in program_groups.values() for name, parameter in group}
            before = [p.detach().clone() for p in network.parameters()]
            torch.nn.utils.clip_grad_norm_(network.parameters(), settings.clip_norm,
                                           error_if_nonfinite=True)
            optimizer.step()
            for name, group in program_groups.items():
                program_history[name]['actual_update_norm'].append(
                    _norm([p.detach() - program_before[key] for key, p in group]))
            for p in network.parameters():
                _finite(p, 'updated parameters')
            pn.append(_norm(list(network.parameters())))
            un.append(_norm([p.detach() - old for p, old in zip(network.parameters(), before)]))
            steps += 1
        network.eval()
        raw_chunks = {task.name: [] for task in active}
        with torch.no_grad():
            for start in range(0, len(dev_x), settings.batch_size):
                for name, value in network(dev_x[start:start + settings.batch_size]).items():
                    raw_chunks[name].append(value.cpu().numpy())
        raw = {k: np.concatenate(v).astype(np.float64) for k, v in raw_chunks.items()}
        losses = _numpy_losses(raw, development, active, scales)
        loss = float(np.mean(list(losses.values())))
        selection_losses = (_selection_losses(raw, development, active, scales)
                            if selection_task is not None else losses)
        selection_loss = (selection_losses[selection_task.name]
                          if selection_task is not None else loss)
        aggregation = (SELECTION_AGGREGATION if selection_task is not None
                       else 'observed_development_equal_task_mean')
        record = {'epoch': epoch, 'development_loss': loss, 'development_task_losses': losses,
                  'development_loss_metric': 'standardized_observed_row_equal_task_mean',
                  'selection_loss': selection_loss, 'selection_task': settings.selection_task,
                  'selection_aggregation': aggregation, 'selection_task_losses': selection_losses,
                  'selection_task_loss_metric': ('original_per_sample_loss' if selection_task is not None
                                                 else 'standardized_observed_row_task_mean'),
                  'preclip_gradient_norm': max(gn) if gn else 0.0,
                  'parameter_norm': pn[-1] if pn else _norm(list(network.parameters())),
                  'actual_update_norm': max(un) if un else 0.0, 'steps': len(un),
                  'partial_epoch': interrupted}
        if program_groups:
            record['program_parameter_updates'] = {
                name: {key: max(values) if values else 0.0 for key, values in measurements.items()}
                for name, measurements in program_history.items()}
        diagnostics['history'].append(record)
        if selection_loss < best_loss:
            best_loss, best_state = selection_loss, _tensor_state(network)
            diagnostics['best_epoch'] = epoch
        if selection_loss < patience_reference - settings.min_delta:
            patience_reference, stale = selection_loss, 0
        else:
            stale += 1
        if time.monotonic() - started >= settings.max_seconds:
            interrupted = True
            diagnostics['stop_reason'] = 'time_budget'
        if interrupted:
            break
        if stale >= settings.patience:
            diagnostics['stop_reason'] = 'development_patience'
            break
    if best_state is None or steps == 0:
        raise TimeoutError('Training ended before a fitted checkpoint could be selected')
    selected_record = diagnostics['history'][diagnostics['best_epoch']]
    diagnostics.update(development_loss=selected_record['development_loss'],
                       development_task_losses=selected_record['development_task_losses'],
                       development_loss_metric='standardized_observed_row_equal_task_mean',
                       selection_loss=best_loss, selection_task=settings.selection_task,
                       selection_aggregation=selected_record['selection_aggregation'],
                       selection_task_losses=selected_record['selection_task_losses'],
                       selection_task_loss_metric=selected_record['selection_task_loss_metric'],
                       elapsed_seconds=time.monotonic() - started,
                       optimizer_steps=steps,
                       checkpoint_selection=('observed_development_primary_task_original_loss'
                                             if selection_task is not None
                                             else 'observed_development_equal_task_mean'),
                       gradient_residual=None)
    for name, group in program_groups.items():
        diagnostics['programs'][name]['selected_raw_update_norm'] = _norm([
            best_state[key].to(initial_program_state[key].device) - initial_program_state[key]
            for key, _ in group])
    # Best state is restored, never the final stale iterate.
    result = FittedResearchModel(metadata, best_state)
    for name, values in result.program_parameters.items():
        diagnostics['programs'][name]['selected_parameter_values'] = values
    return result


def fit_probe(train_z, train_data: TaskDataset, dev_z, dev_data: TaskDataset,
              settings: TrainingSpec) -> FittedProbe:
    started = time.monotonic()
    _check_data(train_data)
    _check_data(dev_data)
    if train_data.schema != dev_data.schema:
        raise ValueError('Training and development schemas differ')
    train_z, dev_z = np.asarray(train_z, dtype=np.float64), np.asarray(dev_z, dtype=np.float64)
    if (train_z.ndim != 2 or dev_z.ndim != 2 or len(train_z) != len(train_data.x)
        or len(dev_z) != len(dev_data.x) or train_z.shape[1] != dev_z.shape[1]
        or train_z.shape[1] == 0):
        raise ValueError('Probe representation dimensions must align')
    _finite(train_z, 'training probe representation')
    _finite(dev_z, 'development probe representation')
    return _linear_fit(train_z, train_data, dev_data, settings, 'probe', started, dev_z)
