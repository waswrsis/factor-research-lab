"""Frozen-representation diagnostics; these are development evidence, not claims.

Zero-ablation measures dependence of the existing head on a module, including
gating effects. It is not the benefit of retraining an architecture without it.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch

from factorlab.protocol import utc

from .contracts import RepresentationSpec, TaskDataset
from .learning import FittedResearchModel, _decode
from .scoring import evaluate


def _members(model):
    members = getattr(model, 'models', [model])
    if not members or not all(isinstance(m, FittedResearchModel) for m in members):
        raise ValueError('Diagnostics require a fitted model or fitted seed group')
    if len({m.schema for m in members}) != 1:
        raise ValueError('Diagnostic model schemas differ')
    return members


def _architecture(member):
    value = member.metadata.get('architecture')
    return RepresentationSpec.from_dict(value) if value is not None else None


def _zero_prediction(member, data, module_index):
    """Forward frozen gate/heads from a copied encoding, never modify a model."""
    architecture = _architecture(member)
    encoded = member.encode(data).copy()
    begin = sum(m.dim for m in architecture.modules[:module_index])
    encoded[:, begin:begin + architecture.modules[module_index].dim] = 0.0
    network = member.network
    if network is None:
        raise ValueError('Module ablation requires learned representation')
    raw = {task.name: [] for task in member.tasks}
    batch_size = member.metadata['settings']['batch_size']
    with torch.no_grad():
        for start in range(0, len(encoded), batch_size):
            z = torch.as_tensor(encoded[start:start + batch_size], dtype=torch.float32)
            if network.combination == 'gated':
                weights = network.gate(z).softmax(1)
                z = (z.reshape(len(z), len(architecture.modules), -1)
                     * weights[:, :, None]).sum(1)
            for name, head in zip(network.task_names, network.heads):
                raw[name].append(head(z).cpu().numpy())
    output = {name: np.concatenate(parts).astype(np.float64) for name, parts in raw.items()}
    return _decode(output, member.tasks, member.metadata['target_scales'])


def _mean_predictions(values):
    if any(set(v) != set(values[0]) for v in values):
        raise ValueError('Seed supervision supports differ')
    return {name: np.mean([v[name] for v in values], axis=0) for name in values[0]}


def _delta(reference, changed):
    return {name: {'support': value['support'], 'reference_loss': value['loss'],
        'ablated_loss': changed[name]['loss'],
        'loss_increase': (changed[name]['loss'] - value['loss']
                          if value['loss'] is not None and changed[name]['loss'] is not None else None)}
        for name, value in reference.items()}


def _redundancy(left, right):
    if len(left) < 3:
        return {'status': 'insufficient_evidence', 'support': len(left),
                'maximum_absolute_coordinate_correlation': None,
                'mean_absolute_coordinate_correlation': None}
    left = left.astype(np.float64) - left.mean(0)
    right = right.astype(np.float64) - right.mean(0)
    ln, rn = np.linalg.norm(left, axis=0), np.linalg.norm(right, axis=0)
    valid_left, valid_right = ln > 1e-12, rn > 1e-12
    if not valid_left.any() or not valid_right.any():
        return {'status': 'constant_representation', 'support': len(left),
                'maximum_absolute_coordinate_correlation': None,
                'mean_absolute_coordinate_correlation': None}
    correlation = (left[:, valid_left].T @ right[:, valid_right]
                   / np.outer(ln[valid_left], rn[valid_right]))
    absolute = np.clip(np.abs(correlation), 0, 1)
    return {'status': 'ok', 'support': len(left),
            'maximum_absolute_coordinate_correlation': float(absolute.max()),
            'mean_absolute_coordinate_correlation': float(absolute.mean()),
            'scope': 'within_seed_coordinates_not_aligned_across_refits'}


def representation_diagnostics(model, data: TaskDataset, *, cutoff=None,
                               state_feature='return_24h', state_threshold=0.01) -> dict:
    """Module contribution/redundancy and trailing-input-state performance.

    Caller must supply development data only. With cutoff, later origins are
    excluded and only mature observed labels are scored. State uses the origin's
    trailing input, never target labels, full-sample quantiles, or future states.
    No fitting, sorting of latent coordinates, or model mutation is performed.
    """
    if not np.isfinite(state_threshold) or state_threshold <= 0:
        raise ValueError('State threshold must be frozen, finite and positive')
    members = _members(model)
    digest = model.digest
    if cutoff is not None:
        limit = utc(cutoff)
        data = data.subset((data.meta.available_at <= limit).to_numpy())
        for task in data.tasks:
            data.observed[task.name] &= data.task_mask(task.name, limit)
    if not len(data.meta):
        return {'status': 'insufficient_evidence', 'rows': 0, 'modules': [],
                'redundancy': [], 'states': {}, 'model_digest': digest}
    architectures = [_architecture(member) for member in members]
    if any(a != architectures[0] for a in architectures):
        raise ValueError('Seed group must share one frozen architecture')
    predictions = _mean_predictions([member.predict(data) for member in members])
    reference = evaluate(data, predictions)
    ablations, redundancy, encoded = [], [], [member.encode(data) for member in members]
    architecture = architectures[0]
    if architecture is not None:
        for i, module in enumerate(architecture.modules):
            changed = _mean_predictions([_zero_prediction(member, data, i) for member in members])
            ablations.append({'name': module.name, 'module': asdict(module),
                              'tasks': _delta(reference, evaluate(data, changed))})
        offsets = np.cumsum([0, *[m.dim for m in architecture.modules]])
        for i, left in enumerate(architecture.modules):
            for j in range(i + 1, len(architecture.modules)):
                per_seed = [_redundancy(z[:, offsets[i]:offsets[i + 1]],
                                        z[:, offsets[j]:offsets[j + 1]]) for z in encoded]
                finite = [value['maximum_absolute_coordinate_correlation'] for value in per_seed
                          if value['maximum_absolute_coordinate_correlation'] is not None]
                redundancy.append({'left': left.name, 'right': architecture.modules[j].name,
                    'per_seed': per_seed,
                    'mean_seed_maximum_absolute_correlation': float(np.mean(finite)) if finite else None})
    states = {}
    if state_feature in data.features:
        value = data.x[:, -1, data.features.index(state_feature)]
        masks = {'down': value < -state_threshold, 'flat': np.abs(value) <= state_threshold,
                 'up': value > state_threshold}
        observed_name = state_feature + '_observed'
        if observed_name in data.features:
            valid = data.x[:, -1, data.features.index(observed_name)] > .5
            masks = {name: mask & valid for name, mask in masks.items()}
            masks['unknown'] = ~valid
        for name, mask in masks.items():
            part = data.subset(mask)
            states[name] = {'rows': len(part.meta), 'metrics': evaluate(
                part, {task: value[mask] for task, value in predictions.items()})}
    if model.digest != digest:
        raise ValueError('Frozen model changed during diagnostics')
    return {'status': 'ok', 'format': 'representation-diagnostics-v1', 'rows': len(data.meta),
        'model_digest': digest, 'cutoff': utc(cutoff).isoformat() if cutoff is not None else None,
        'metrics': reference, 'modules': ablations, 'redundancy': redundancy, 'states': states,
        'state_policy': {'feature': state_feature, 'threshold': state_threshold,
                         'basis': 'trailing_input_at_origin'},
        'scope': 'provided_development_only_no_independent_claim',
        'contribution_semantics': 'frozen_head_zero_ablation_not_retrained_architecture',
        'coordinate_identity': 'within_frozen_version_only'}
