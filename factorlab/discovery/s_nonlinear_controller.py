"""Actual shared S nonlinear opportunity controller; never a serving/adoption path."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from factorlab.protocol import fingerprint

from .algorithm_shortlist import AlgorithmCatalogue
from .joint_pool import FrozenPool
from .neural_forward import FrozenNeuralRecipe
from .neural_research_launcher import launch_research
from .neural_research_ticket import issue_research_ticket
from .nonlinear_orchestration import input_data_identity
from .s_nonlinear_gates import NonlinearGateClosed, freeze_shared_epoch
from .s_nonlinear_launcher import launch_nonlinear, verify_completed_nonlinear
from .s_nonlinear_ticket import capacity, freeze_nonlinear_plan, issue_nonlinear_ticket
from .v7_storage import immutable_record
from .work_authority_bridge import WorkAuthorityBridge
from .work_lease import FrozenWorkLease

VERSION = 'v7-shared-S-nonlinear-controller-v3-prepaid-evolution'


def _history(authority, epoch):
    # This index chooses candidates for later paid verification; it cannot grant
    # an effect/resource gate. Every chosen ticket is independently replayed in
    # issue/worker/parent verification inside the original prepaid work lease.
    directory = authority.directory / 'nonlinear_epochs' / epoch / 'producer_refs'
    paths = sorted(directory.glob('*.json')) if directory.is_dir() else []
    if len(paths) > 120:
        raise ValueError('Nonlinear epoch retained producer bound reached')
    records = []
    for path in paths:
        value = json.loads(path.read_text('utf-8'))
        if set(value) != {'record', 'hash'} or fingerprint(value['record']) != value['hash']:
            raise ValueError('Nonlinear producer index changed')
        record = value['record']
        if path.stem != record['ticket_digest'][:16] or record['epoch_id'] != epoch:
            raise ValueError('Nonlinear producer index identity changed')
        records.append(record)
    return records


def _register(authority, epoch, ticket):
    lease = FrozenWorkLease.from_dict(ticket.value['lease'])
    lease.assert_parent_alive()
    report = verify_completed_nonlinear(ticket.path)
    record = {'epoch_id': epoch, 'kind': ticket.value['kind'], 'ticket': str(ticket.path),
              'ticket_digest': ticket.digest, 'recipe_id': fingerprint(ticket.value['recipe']),
              'cutoff': ticket.value['context']['cutoff'],
              'available_at': report['publication']['observation']['wall']}
    immutable_record(authority.directory / 'nonlinear_epochs' / epoch / 'producer_refs' / (ticket.digest[:16] + '.json'),
                     {'record': record, 'hash': fingerprint(record)})
    lease.assert_parent_alive()
    return record


def run_shared_nonlinear(bridge, data, cutoff, *, mode, feed_directory):
    if type(bridge) is not WorkAuthorityBridge or mode not in {'probes', 'isolated'}:
        raise TypeError('Actual shared S bridge and explicitly frozen nonlinear mode required')
    bridge._guard()
    if bridge.store.identity['config']['nonlinear_mode'] != mode:
        raise ValueError('Nonlinear mode differs from the original frozen controller')
    if bridge._active is not None or bridge.state['cutoff'] != pd.Timestamp(cutoff).isoformat():
        raise ValueError('Nonlinear work must follow closed parent work at original cutoff')
    if bridge.state.get('nonlinear_report') is not None:
        raise ValueError('Completed nonlinear opportunities are readback only; no second run')
    identity = input_data_identity(data, cutoff)
    if bridge.state.get('nonlinear_input_data_id', identity) != identity:
        raise ValueError('Actual nonlinear controller input changed')
    bridge.state['nonlinear_input_data_id'] = identity
    bridge.store.save(bridge.month, bridge.state, 'shared_nonlinear_actual_input')
    recipe = FrozenNeuralRecipe(pool=FrozenPool.from_dict(bridge.state['frozen_reference_pool']), rolling_spec=data.spec)
    from .structural_evolution import select_structure
    structural = select_structure(bridge, recipe)
    if structural['body']['recipe'] is None:
        result = {'version': VERSION, 'role': 'S', 'serving_eligible': False,
                  'formal_claim': False, 'status': structural['body']['status'],
                  'structure_search': structural, 'work': [], 'deferred': []}
        bridge.state['nonlinear_report'] = result
        bridge.store.save(bridge.month, bridge.state, 'structural_search_deferred')
        bridge._guard()
        return result
    recipe = FrozenNeuralRecipe.from_dict(structural['body']['recipe'])
    epoch = freeze_shared_epoch(bridge, data, recipe)
    epoch_value = json.loads(Path(epoch['path']).read_text('utf-8'))
    from .neural_research_ticket import _context
    from .s_observation_source import physical_cutoff
    context = _context(bridge.store, bridge.month, bridge.state, bridge.identity, recipe)
    available_before = physical_cutoff(context)
    prior = [r for r in _history(bridge.authority, epoch_value['epoch_id'])
             if r['recipe_id'] == recipe.pipeline_id
             and pd.Timestamp(r['cutoff']) < pd.Timestamp(cutoff)
             and pd.Timestamp(r['available_at']) <= available_before]
    quarters = {}
    for r in sorted(prior, key=lambda v: (v['cutoff'], v['ticket_digest'])):
        if r['kind'] == 'quarterly_probe':
            quarters.setdefault(r['cutoff'], r)
    probes = [r['ticket'] for r in list(quarters.values())[-2:]]
    resource = next((r['ticket'] for r in reversed(prior) if r['kind'] == 'resource_trial'), None)
    due = (pd.Timestamp(cutoff).month - 1) % 3 == 0
    kinds = ['quarterly_probe'] if due else []
    gates = {'epoch': epoch, 'probes': probes}
    reviewed = structural['body'].get('review_reservation') is not None
    if mode == 'isolated' and len(probes) == 2 and (resource is None or reviewed):
        kinds.append('complete_neural_S' if resource else 'resource_trial')
    scheduled = json.loads((bridge.directory / 'schedule/identity.json').read_text('utf-8'))['schedule']
    limits = scheduled['allocation']['nonlinear_max_counts']
    selected, deferred = [], []
    if not reviewed:
        deferred.append({'kind': 'structure_review', 'reason': 'quarterly_probe_priority',
            'earliest_cutoff': (pd.Timestamp(cutoff)+pd.DateOffset(months=1)).isoformat()})
        deferred.append({'kind': 'complete_neural_S', 'reason': 'full_structure_review_not_completed'})
    if mode == 'probes':
        deferred.append({'kind': 'complete_neural_S', 'reason': 'probe_only_mode_has_no_neural_permission'})
    elif len(probes) < 2:
        deferred.append({'kind': 'complete_neural_S', 'reason': 'independent_recipe_prior_quarter_gates_missing',
                         'available_quarters': len(probes), 'required_quarters': 2,
                         'pipeline_id': recipe.pipeline_id})
    elif resource is None:
        deferred.append({'kind': 'complete_neural_S', 'reason': 'prior_same_recipe_resource_proof_missing',
                         'pipeline_id': recipe.pipeline_id})
    from .structural_review import combined_capacity
    for kind in kinds:
        proposal = selected + ([kind, 'typed_queue_review'] if kind == 'complete_neural_S' else [kind])
        if (any(sum(capacity(recipe, k).get(counter, 0) for k in proposal) + (1 if reviewed and counter == 'solve' else 0) > limit
                for counter, limit in limits.items())
                or (len(proposal) + int(reviewed)) * 300 > scheduled['allocation']['nonlinear_seconds_holdback']):
            deferred.append({'kind': kind, 'reason': 'full_recipe_exceeds_original_pool_dependent_holdback',
                             'required': capacity(recipe, kind)})
        else:
            selected = proposal
    allocation_used = combined_capacity(recipe, selected, scheduled['allocation'], review=reviewed)
    result = {'version': VERSION, 'role': 'S', 'serving_eligible': False, 'formal_claim': False,
              'combined_allocation': allocation_used,
              'recipe': recipe.to_dict(), 'structure_search': structural, 'epoch': epoch, 'work': [], 'deferred': deferred,
              'status': 'pending_gates', 'prior_quarters': len(probes),
              'requires_prior_resource': resource is None, 'new_evidence_earliest_use': 'subsequent_month_cutoff'}
    if selected:
        freeze_nonlinear_plan(bridge, recipe, kinds=tuple(selected))
    for kind in selected:
        if kind == 'typed_queue_review':
            result['deferred'].append({'kind': kind, 'reason': 'preallocated_actual_scheduler_queue_producer'})
            continue
        bridge._guard()
        catalogue = None
        ticket = None
        try:
            if kind == 'complete_neural_S':
                catalogue = AlgorithmCatalogue.freeze(bridge, (recipe,))
                ticket = issue_research_ticket(bridge, data, recipe, gates={**gates, 'resource': resource},
                                               feed_directory=feed_directory)
                try:
                    launch_research(ticket.path)
                finally:
                    # A failed consumed ticket remains part of the catalogue denominator.
                    catalogue.record_ticket(ticket.path)
                result['catalogue'] = catalogue.seal_shortlist()
                result['work'].append({'kind': kind, 'ticket': str(ticket.path), 'status': 'completed'})
            else:
                ticket = issue_nonlinear_ticket(bridge, data, recipe, kind=kind,
                    gates=gates if kind == 'resource_trial' else None, feed_directory=feed_directory)
                launch_nonlinear(ticket.path)
                result['work'].append(_register(bridge.authority, epoch_value['epoch_id'], ticket))
        except NonlinearGateClosed as error:
            reservation_id = getattr(error, 'reservation_id', None)
            if reservation_id is None and ticket is not None:
                reservation_id = ticket.value['reservation_id']
            record = bridge.authority.get(reservation_id)
            if record['status'] != 'failed_consumed':
                raise ValueError('Expected closed gate lacks its original consumed work') from error
            result['work'].append({'kind': kind, 'status': 'gate_closed_consumed',
                'reservation_id': record['reservation_id'], 'prepaid_counts': record['counts'],
                'reason': str(error)})
            if catalogue is not None:
                result['catalogue'] = catalogue.seal_shortlist()
    if result['work'] and all(w.get('status') == 'gate_closed_consumed' for w in result['work']):
        result['status'] = 'gate_closed_consumed'
    elif result['work']:
        result['status'] = 'completed_available_later' if 'catalogue' not in result else 'completed_S_only'
    bridge._guard()
    result = json.loads(json.dumps(result, allow_nan=False))
    bridge.state['nonlinear_report'] = result
    bridge.store.save(bridge.month, bridge.state, 'shared_nonlinear_closed')
    return result
