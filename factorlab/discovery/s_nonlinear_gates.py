"""Shared-wallet nonlinear gates verified from fixed producer tickets, never booleans.

Two distinct quarterly receipts must precede the target cutoff. A new resource
trial can open only a subsequent month. Epoch identity includes the entire pool,
recipe, raw coordinate system, source code and actual receipt source.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd

from factorlab.protocol import fingerprint, project_path

from .neural_forward import FrozenNeuralRecipe
from .neural_research_ticket import _canonical, _context
from .nonlinear_channel import ProbeConfig
from .nonlinear_orchestration import causal_worker_data, input_data_identity
from .v7_storage import immutable_record
from .work_authority_bridge import WorkAuthorityBridge

VERSION = 'v7-shared-nonlinear-epoch-v1'


class NonlinearGateClosed(ValueError):
    """Expected lack of effect/capacity, distinct from damaged evidence or IO."""


def _terms(context, recipe, data, source_hash):
    return {'version': VERSION, 'recipe': _canonical(recipe.to_dict()),
            'source_id': context['source_id'], 'symbols': context['symbols'],
            'source_hash': source_hash, 'schema_hash': data.frame.schema_hash,
            'feature_names': list(data.frame.feature_names), 'bypass_mask': list(data.frame.bypass_mask),
            'probe': _canonical(ProbeConfig().__dict__), 'role': 'S', 'serving_eligible': False,
            'criterion': 'two_distinct_quarters_all_three_paired_gains_positive;positive_control_mean_gain_positive',
            'resource_rule': 'prior_cutoff_exact_recipe_actual_cgroup_whole_workload_and_capacity'}


def freeze_shared_epoch(bridge, data, recipe):
    if type(bridge) is not WorkAuthorityBridge or type(recipe) is not FrozenNeuralRecipe:
        raise TypeError('Actual shared controller and exact frozen learner required')
    bridge._guard()
    context = _context(bridge.store, bridge.month, bridge.state, bridge.identity, recipe)
    if input_data_identity(data, context['cutoff']) != context['input_id']:
        raise ValueError('Epoch input does not match sealed S input')
    terms = _terms(context, recipe, data, fingerprint(bridge.authority.manifest['source_files']))
    key = fingerprint(terms)
    root = bridge.authority.directory / 'nonlinear_epochs' / key
    primary, mirror = root / 'epoch.json', bridge.authority.directory / 'nonlinear_epoch_anchors' / (key + '.json')
    if primary.exists() or mirror.exists():
        if not primary.is_file() or not mirror.is_file():
            raise ValueError('Nonlinear epoch anchor lost; no regeneration')
        value = json.loads(primary.read_text('utf-8'))
        if value != json.loads(mirror.read_text('utf-8')) or value['terms'] != terms:
            raise ValueError('Original nonlinear epoch changed')
    else:
        value = {'terms': terms, 'epoch_id': key, 'authority_directory': str(bridge.authority.directory),
                 'authority_manifest_hash': bridge.authority.manifest_hash,
                 'available_at': datetime.now(UTC).isoformat()}
        immutable_record(primary, value)
        immutable_record(mirror, value)
    return {'path': str(primary), 'digest': fingerprint(value)}


def _epoch(reference, context, recipe, data, source_hash):
    if type(reference) is not dict or set(reference) != {'path', 'digest'}:
        raise ValueError('Registered exact nonlinear epoch reference required')
    path = project_path(reference['path'])
    value = json.loads(path.read_text('utf-8'))
    if (fingerprint(value) != reference['digest'] or value['epoch_id'] != fingerprint(value['terms'])
            or value['terms'] != _terms(context, recipe, data, source_hash)):
        raise ValueError('Nonlinear epoch differs from current pool/source/full recipe/input schema')
    root = project_path(value['authority_directory'])
    if (path != root / 'nonlinear_epochs' / value['epoch_id'] / 'epoch.json'
            or json.loads((root / 'nonlinear_epoch_anchors' / (value['epoch_id'] + '.json')).read_text('utf-8')) != value):
        raise ValueError('Nonlinear epoch canonical double anchor differs')
    from .neural_forward_ticket import open_existing_authority
    authority = open_existing_authority(root)
    if authority.manifest_hash != value['authority_manifest_hash']:
        raise ValueError('Epoch shared authority changed')
    return value


def verify_shared_gates(gates, data, recipe, context, *, source_hash, require_resource=False):
    expected = {'epoch', 'probes', 'resource'} if require_resource else {'epoch', 'probes'}
    if type(gates) is not dict or set(gates) != expected:
        raise ValueError('Actual shared producer tickets and registered epoch required')
    epoch = _epoch(gates['epoch'], context, recipe, data, source_hash)
    if not isinstance(gates['probes'], list) or len(gates['probes']) != 2:
        raise ValueError('Exactly two predeclared distinct quarterly producer references required')
    from .s_nonlinear_launcher import verify_completed_nonlinear
    quarters = set()
    cutoff = pd.Timestamp(context['cutoff'])
    physical_cutoff = cutoff
    if context.get('observation_prefix') is not None:
        if context.get('role') == 'S':
            from .s_observation_source import physical_cutoff as original_S_physical_cutoff
            physical_cutoff = original_S_physical_cutoff(context)
        else:
            from .nonlinear_runtime import _context as gate_context, _gate_physical_cutoff
            from .neural_forward_ticket import _LifecycleReader
            from .quarantined_observation import job_binding
            reader = _LifecycleReader(context['lifecycle_directory'])
            try:
                if job_binding(reader, context['job']) != context['observation_prefix']:
                    raise ValueError('Shared S gate lost original forward job observation')
            finally:
                reader.close()
            physical_cutoff = _gate_physical_cutoff(
                gate_context(data, cutoff, recipe.pool, context['source_id']),
                {'lifecycle_directory': context['lifecycle_directory'], 'job': context['job']})
    reports = []
    def physically_visible(report):
        if pd.Timestamp(report['publication']['observation']['wall']) > physical_cutoff:
            return False
        if context.get('observation_prefix') is not None:
            path = report['ticket'].path.parent/'launch/publication/publication.sqlite'
            stat = path.stat()
            if max(stat.st_mtime_ns, stat.st_ctime_ns) > physical_cutoff.value:
                return False
        return True
    for path in gates['probes']:
        report = verify_completed_nonlinear(path)
        ticket = report['ticket']
        origin = ticket.value['context']
        quarter = pd.Timestamp(origin['cutoff'])
        if (ticket.value['kind'] != 'quarterly_probe'
                or ticket.value['recipe'] != _canonical(recipe.to_dict())
                or origin['source_id'] != context['source_id'] or origin['symbols'] != context['symbols']
                or ticket.value['source_hash'] != source_hash
                or ticket.value['authority_directory'] != epoch['authority_directory']
                or pd.Timestamp(ticket.value['lease']['wall_started_at']) < pd.Timestamp(epoch['available_at'])
                or not physically_visible(report)
                or quarter >= cutoff or (quarter.month - 1) % 3):
            raise ValueError('Probe is not prior immutable same-epoch shared S evidence')
        quarters.add(quarter.isoformat())
        folds = report['result']['probe']['folds']
        positive = report['result']['positive_control']['relative_gain']
        if not all(f['gain'] is not None and f['gain'] > 0 for f in folds) or positive is None or positive <= 0:
            raise NonlinearGateClosed('Repeated same-support direction/positive-control gate did not pass')
        reports.append(report)
    if len(quarters) != 2:
        raise ValueError('Duplicate quarterly evidence is not repeatability')
    result = {'permission': 'controlled_resource_trial_only', 'epoch_id': epoch['epoch_id'],
              'role': 'S', 'serving_eligible': False, 'research_enabled': False, 'test_only': False}
    if require_resource:
        resource = verify_completed_nonlinear(gates['resource'])
        ticket = resource['ticket']
        if (ticket.value['kind'] != 'resource_trial' or ticket.value['recipe'] != _canonical(recipe.to_dict())
                or ticket.value['source_hash'] != source_hash
                or ticket.value['authority_directory'] != epoch['authority_directory']
                or ticket.value['gates'] != {'epoch': gates['epoch'], 'probes': gates['probes']}
                or not physically_visible(resource)):
            raise ValueError('Resource proof is not prior same-recipe shared production evidence')
        trial_data = resource['data']
        actual = causal_worker_data(data, cutoff)
        trial = trial_data.frame.values
        current = actual.frame.values
        if (trial.shape[0] < current.shape[0] or trial.shape[1] != current.shape[1]
                or trial.nbytes < current.nbytes or trial_data.frame.schema_hash != actual.frame.schema_hash):
            raise NonlinearGateClosed('Measured resource trial does not cover actual causal input capacity')
        result.update(permission='complete_three_fold_S_only', research_enabled=True)
    return result
