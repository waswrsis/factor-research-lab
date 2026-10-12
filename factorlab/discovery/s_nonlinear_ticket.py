"""Two fixed S-only work producers on the controller's original shared wallet.

This does not create A0 jobs, accept executable callbacks, or confer serving
rights. Every cold tree/resource worker is prepaid before payload preparation.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil
from filelock import FileLock

from factorlab.protocol import fingerprint, project_path

from .neural_forward import FrozenNeuralRecipe
from .neural_forward_ticket import open_existing_authority, runtime_identity
from .neural_research_ticket import (
    _bridge_identity,
    _canonical,
    _context,
    _input,
    _outer,
    _reader,
)
from .nonlinear_launcher import _load_data, save_worker_data
from .v7_storage import immutable_record
from .work_authority_bridge import WorkAuthorityBridge
from .work_lease import FrozenWorkLease

VERSION = 'v7-authority-S-nonlinear-ticket-v1'
KINDS = ('quarterly_probe', 'resource_trial')
PLAN_KINDS = (*KINDS, 'complete_neural_S', 'typed_queue_review')


def _read(path):
    path = project_path(path)
    if not path.is_file() or path.stat().st_size > 16 * 2**20:
        raise ValueError('Missing/bounded S nonlinear artifact')
    return json.loads(path.read_text('utf-8'))


def capacity(recipe, kind):
    if type(recipe) is not FrozenNeuralRecipe or kind not in PLAN_KINDS:
        raise ValueError('Only exact fixed S nonlinear work kinds are permitted')
    if kind == 'typed_queue_review':
        return {'solve': 1}
    if kind == 'complete_neural_S':
        from .neural_research import research_capacity
        return research_capacity(recipe)
    bound = recipe.budget_limits
    baseline = bound['solve'] - 2 * len(recipe.config.seeds)
    return ({'tree_fit': 3, 'solve': 3 * baseline + 6} if kind == 'quarterly_probe' else
            {k: bound[k] for k in ('solve', 'neural_fit', 'epoch')})


def _plan(bridge_identity, state, recipe, kind, store=None):
    plan = state.get('nonlinear_work_plan')
    if not isinstance(plan, dict) or set(plan) != {'body', 'hash'} or fingerprint(plan['body']) != plan['hash']:
        raise ValueError('Actual controller must freeze the complete nonlinear work plan before any producer')
    if store is None:
        raise TypeError('Actual readonly controller required for plan allocation verification')
    if state.get('structural_plan') is not None:
        from .structural_evolution import assert_selected_structure
        assert_selected_structure(store, state, bridge_identity, recipe)
    body = plan['body']
    local = store.directory / 'work_authority' / state['cutoff'][:7]
    authority_root = project_path(bridge_identity['authority_directory']) / 'search_bridges' / bridge_identity['bridge_id']
    for root in (local, authority_root):
        if _read(root / 'nonlinear_plan.json') != plan:
            raise ValueError('Nonlinear immutable plan missing or changed')
    schedule = _read(local / 'schedule/identity.json')
    if schedule != _read(authority_root / 'schedule/identity.json'):
        raise ValueError('Shared original schedule anchors differ')
    from .scheduled_search import FrozenSWorkSchedule
    frozen = FrozenSWorkSchedule.from_dict(schedule['schedule'])
    allocation = frozen.to_dict()['allocation']
    if (schedule['schedule_hash'] != frozen.digest
            or schedule['bridge_identity_hash'] != fingerprint(bridge_identity)
            or body['schedule_hash'] != frozen.digest
            or body['allocation'] != allocation
            or body['bridge_identity_hash'] != fingerprint(bridge_identity)
            or body['recipe'] != _canonical(recipe.to_dict()) or body['serving_eligible'] is not False):
        raise ValueError('Nonlinear plan differs from original shared S recipe/bridge/allocation')
    if store.identity['config']['budget_policy'] == 'controls_third' and store.identity['config']['search_mode'] != 'gpt':
        raise ValueError('Controls nonlinear holdback has only one predeclared owner: gpt arm')
    structure = state.get('structural_plan')
    if structure is not None:
        from .structural_review import combined_capacity, verify_review_completion
        reviewed = structure['body'].get('review_reservation') is not None
        expected_review = {'reservation_id': structure['body'].get('review_reservation'),
                           'counts': {'solve': 1} if reviewed else {}, 'seconds': 300 if reviewed else 0}
        if body.get('structure_review') != expected_review:
            raise ValueError('Nonlinear plan lost its original prepaid structure review')
        combined_capacity(recipe, [w['kind'] for w in body['work']], allocation, review=reviewed)
        if kind in {'complete_neural_S', 'typed_queue_review'}:
            verify_review_completion(bridge_identity['authority_directory'], structure)
    limits = allocation['nonlinear_max_counts']
    work = body['work']
    if (not work or len({w['kind'] for w in work}) != len(work)
            or any(w['kind'] not in PLAN_KINDS or w['counts'] != capacity(recipe, w['kind'])
                   or w['seconds'] != 300 for w in work)
            or any(sum(w['counts'].get(k, 0) for w in work) > limits.get(k, 0)
                   for k in {key for w in work for key in w['counts']})
            or sum(w['seconds'] for w in work) > allocation['nonlinear_seconds_holdback']):
        raise ValueError('Nonlinear works exceed their original shared holdback')
    if len([w for w in work if w['kind'] == kind]) != 1:
        raise ValueError('Work is absent from original fixed nonlinear allocation')
    return plan


def freeze_nonlinear_plan(bridge, recipe, *, kinds):
    """Freeze exact bounded work opportunities; this never passes any effect gate."""
    if type(bridge) is not WorkAuthorityBridge or type(recipe) is not FrozenNeuralRecipe:
        raise TypeError('Actual shared controller and complete recipe required')
    bridge._guard()
    if bridge.state.get('structural_plan') is not None:
        from .structural_evolution import assert_selected_structure
        assert_selected_structure(bridge.store, bridge.state, bridge.identity, recipe)
    if bridge._active is not None:
        raise ValueError('Close parent work before freezing child opportunity plan')
    if (not isinstance(kinds, tuple) or not kinds or len(set(kinds)) != len(kinds)
            or any(kind not in PLAN_KINDS for kind in kinds)):
        raise ValueError('Exact fixed nonlinear opportunity kinds required')
    if 'quarterly_probe' in kinds and (int(bridge.state['cutoff'][5:7]) - 1) % 3:
        raise ValueError('Fixed quarterly probe is not due this month')
    schedule = _read(bridge.directory / 'schedule/identity.json')
    body = {'version': VERSION, 'bridge_identity_hash': fingerprint(bridge.identity),
            'schedule_hash': schedule['schedule_hash'], 'allocation': schedule['schedule']['allocation'],
            'recipe': _canonical(recipe.to_dict()), 'serving_eligible': False,
            'work': [{'kind': k, 'counts': capacity(recipe, k), 'seconds': 300} for k in kinds]}
    structure = bridge.state.get('structural_plan')
    if structure is not None:
        reviewed = structure['body'].get('review_reservation') is not None
        body['structure_review'] = {'reservation_id': structure['body'].get('review_reservation'),
            'counts': {'solve': 1} if reviewed else {}, 'seconds': 300 if reviewed else 0}
    plan = {'body': body, 'hash': fingerprint(body)}
    # Writing an oversized proposal can only fail closed; it grants no lease.
    for root in (bridge.directory, bridge.anchor):
        immutable_record(root / 'nonlinear_plan.json', plan)
    prior = bridge.state.get('nonlinear_work_plan')
    if prior is not None and prior != plan:
        raise ValueError('Existing nonlinear plan cannot be replaced')
    bridge.state['nonlinear_work_plan'] = plan
    _plan(bridge.identity, bridge.state, recipe, kinds[0], bridge.store)
    bridge.store.save(bridge.month, bridge.state, 'fixed_nonlinear_opportunities')
    return plan


def _gates(gates, data, recipe, context):
    from .s_nonlinear_gates import verify_shared_gates
    from .v7_run import source_files
    return verify_shared_gates(gates, data, recipe, context,
                              source_hash=fingerprint(source_files()), require_resource=False)


def _binding(base):
    return {'S_nonlinear_terms': fingerprint(base), 'role': 'S',
            'nonlinear_bridge_id': base['context']['bridge_id'], 'kind': base['kind'],
            'plan_hash': base['plan']['hash'], 'pipeline_id': fingerprint(base['recipe']),
            'input_id': base['context']['input_id'], 'serving_eligible': False}


def _key(context, kind):
    # Equivalent gates/seconds/recipes cannot create a second work slot.
    return fingerprint({'bridge': context['bridge_identity_hash'], 'kind': kind})


@dataclass(frozen=True)
class SNonlinearTicket:
    path: Path
    value: dict

    def __post_init__(self):
        object.__setattr__(self, 'path', project_path(self.path))
        object.__setattr__(self, 'value', _canonical(copy.deepcopy(self.value)))
        expected = {'version', 'kind', 'context', 'bridge', 'plan', 'recipe', 'gates',
                    'feed_directory', 'feed_identity', 'source_hash', 'runtime', 'data',
                    'authority_directory', 'authority_manifest_hash', 'reservation_id',
                    'terms_hash', 'lease', 'lease_hash'}
        value = self.value
        if set(value) != expected or value['version'] != VERSION or value['kind'] not in KINDS:
            raise ValueError('Unknown S nonlinear ticket schema or producer')
        lease = FrozenWorkLease.from_dict(value['lease'])
        if lease.lease_seconds != 300 or lease.digest != value['lease_hash']:
            raise ValueError('Fixed nonlinear work requires its original 300-second reservation')
        expected_path = project_path(value['authority_directory']) / 'S_nonlinear_tickets' / _key(value['context'], value['kind']) / 'ticket.json'
        if self.path != expected_path:
            raise ValueError('Noncanonical nonlinear ticket path')
        FrozenNeuralRecipe.from_dict(value['recipe'])
        object.__setattr__(self, '_digest', fingerprint(value))

    @property
    def digest(self):
        if fingerprint(self.value) != self._digest:
            raise ValueError('S nonlinear ticket mutated')
        return self._digest

    @classmethod
    def load(cls, path):
        value = _read(path)
        if set(value) != {'value', 'sha256'} or fingerprint(value['value']) != value['sha256']:
            raise ValueError('S nonlinear ticket checksum changed')
        return cls(path, value['value'])

    def verify(self, *, mode='launch'):
        value = self.value
        if mode not in {'launch', 'worker', 'running_parent', 'completed'}:
            raise ValueError('Unknown exact S nonlinear worker phase')
        if _read(self.path) != {'value': value, 'sha256': self.digest}:
            raise ValueError('Persisted nonlinear ticket changed')
        authority = open_existing_authority(value['authority_directory'])
        record = authority.get(value['reservation_id'])
        recipe = FrozenNeuralRecipe.from_dict(value['recipe'])
        base = {k: value[k] for k in value if k not in {'authority_manifest_hash', 'reservation_id',
            'terms_hash', 'lease', 'lease_hash', 'data'}}
        desired = {'launch': 'reserved', 'worker': 'running', 'running_parent': 'running', 'completed': 'completed'}[mode]
        if (authority.manifest_hash != value['authority_manifest_hash'] or record['status'] != desired
                or record['binding'] != _binding(base)
                or any(record[k] != value[k] for k in ('terms_hash', 'lease', 'lease_hash'))
                or record['counts'] != {k: capacity(recipe, value['kind']).get(k, 0) for k in record['counts']}
                or fingerprint(authority.manifest['source_files']) != value['source_hash']
                or value['runtime'] != runtime_identity()
                or _read(self.path.parent / 'preparation.json') != {
                    'reservation_id': record['reservation_id'], 'lease': record['lease'], 'terms_hash': record['terms_hash']}):
            raise ValueError('S nonlinear work differs from canonical original reservation/runtime')
        lease = FrozenWorkLease.from_dict(value['lease'])
        _outer(value['bridge'], lease, current=mode != 'completed')
        if mode != 'completed':
            lease.assert_parent_alive(as_worker=mode == 'worker')
            if mode != 'worker' and os.getpid() != lease.parent_pid:
                raise ValueError('Only original S issuer may launch this producer')
        if mode == 'worker' and record['process_identity'] != {'pid': os.getpid(), 'created': psutil.Process().create_time()}:
            raise ValueError('Only the canonical registered direct child may compute')
        reader = _reader(value['context']['run_directory'])
        try:
            month = value['context']['replay_month']
            state = copy.deepcopy(reader.get(month))
            if mode == 'completed' and state['status'] in {'complete', 'finalizing'}:
                state['status'] = 'running'
            if (reader.identity['source_files'] != authority.manifest['source_files']
                    or reader.identity['feed_identity'] != value['feed_identity']
                    or _bridge_identity(reader.directory / 'work_authority' / month / 'identity.json') != value['bridge']
                    or _bridge_identity(authority.directory / 'search_bridges' / value['context']['bridge_id'] / 'identity.json') != value['bridge']
                    or _context(reader, month, state, value['bridge'], recipe) != value['context']
                    or _plan(value['bridge'], state, recipe, value['kind'], reader) != value['plan']):
                raise ValueError('Actual S source/controller/plan changed')
            if (authority.manifest['scope']['task'] != 'volatility'
                    or authority.manifest['scope']['universe'] != value['context']['symbols']):
                raise ValueError('Nonlinear work belongs to another task/universe wallet')
            for key in ('arrays', 'metadata'):
                path = project_path(value['data'][key])
                if path.parent != self.path.parent / 'payload' or path.stat().st_size > 16 * 2**20:
                    raise ValueError('Nonlinear payload escaped original ticket')
            data = _load_data({'data': value['data'], 'rolling': asdict(recipe.rolling_spec)})
            data = _input(data, recipe, value['context'], value['feed_directory'], value['feed_identity'])
            if value['kind'] == 'resource_trial':
                _gates(value['gates'], data, recipe, value['context'])
            elif value['gates'] is not None:
                raise ValueError('Quarterly producer cannot import external claimed gates')
        finally:
            reader.close()
        if mode != 'completed':
            lease.assert_parent_alive(as_worker=mode == 'worker')
        return authority, record, data, recipe


def issue_nonlinear_ticket(bridge, data, recipe, *, kind, gates=None, feed_directory=None):
    if type(bridge) is not WorkAuthorityBridge or type(recipe) is not FrozenNeuralRecipe:
        raise TypeError('Actual shared bridge and exact complete recipe required')
    bridge._guard()
    if bridge._active is not None:
        raise ValueError('An existing parent work must close before this fixed child work')
    context = _context(bridge.store, bridge.month, bridge.state, bridge.identity, recipe)
    plan = _plan(bridge.identity, bridge.state, recipe, kind, bridge.store)
    folder = bridge.authority.directory / 'S_nonlinear_tickets' / _key(context, kind)
    path = folder / 'ticket.json'
    with FileLock(str(folder) + '.lock', timeout=0):
        if folder.exists():
            raise ValueError('Original nonlinear work already prepared or consumed; no retry')
        lease = FrozenWorkLease.issue(300)
        _outer(bridge.identity, lease, current=True)
        base = {'version': VERSION, 'kind': kind, 'context': context, 'bridge': _canonical(bridge.identity),
                'plan': plan, 'recipe': _canonical(recipe.to_dict()), 'gates': _canonical(gates),
                'feed_directory': str(project_path(feed_directory)) if feed_directory is not None else None,
                'feed_identity': bridge.store.identity['feed_identity'],
                'source_hash': fingerprint(bridge.store.identity['source_files']), 'runtime': runtime_identity(),
                'authority_directory': str(bridge.authority.directory)}
        # Prepay BEFORE data reconstruction, gate checks and payload serialization.
        record = bridge.authority._reserve_with_lease('S_nonlinear:' + _key(context, kind),
            counts=capacity(recipe, kind), binding=_binding(base), lease=lease)
        if record['status'] != 'reserved' or record['lease'] != lease.to_dict():
            raise ValueError('A missing nonlinear preparation cannot renew the original work')
        try:
            immutable_record(folder / 'preparation.json', {'reservation_id': record['reservation_id'],
                'lease': record['lease'], 'terms_hash': record['terms_hash']})
            data = _input(data, recipe, context, base['feed_directory'], base['feed_identity'])
            if kind == 'resource_trial':
                _gates(gates, data, recipe, context)
            elif gates is not None:
                raise ValueError('Quarterly producer cannot accept external claimed gates')
            payload = save_worker_data(folder / 'payload', data, cutoff=context['cutoff'])
            value = {**base, 'data': payload, **{k: record[k] for k in (
                'authority_manifest_hash', 'reservation_id', 'terms_hash', 'lease', 'lease_hash')}}
            ticket = SNonlinearTicket(path, value)
            immutable_record(path, {'value': ticket.value, 'sha256': ticket.digest})
            bridge._guard()
            lease.assert_parent_alive()
            return ticket
        except BaseException as error:
            from .s_nonlinear_gates import NonlinearGateClosed
            if isinstance(error, NonlinearGateClosed):
                error.reservation_id = record['reservation_id']
            try:
                bridge.authority.fail_consumed(record['reservation_id'], 'S_nonlinear_preparation:' + type(error).__name__)
            except BaseException as other:  # noqa: BLE001 -- retain original failure.
                error.add_note('Nonlinear failure accounting: ' + type(other).__name__)
            raise
