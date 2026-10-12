"""Frozen nontransferable lanes in one wallet; evidence derives routes.

These checks authenticate local trusted-producer records, not administrators.
There is no public caller-selected lane or second balance. Historical routes
verify original immutable provenance without requiring a live old issuer.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing

from factorlab.protocol import fingerprint, project_path

KINDS = ('ast', 'f', 'gpt_attempt', 'family', 'solve', 'neural_fit', 'epoch', 'tree_fit')
VERSION = 'v7-monthly-partitions-v1'


def counts(**values):
    return {k: values.get(k, 0) for k in KINDS}


def frozen_policy(mode):
    if mode not in ('standard', 'controls'):
        raise ValueError('Unknown frozen monthly allocation')
    if mode == 'standard':
        lanes = {'S': {'counts': counts(ast=64, f=23, gpt_attempt=4, family=2, solve=717), 'seconds': 1890}}
    else:
        lanes = {f'S:{arm}': {'counts': counts(ast=32, f=8, gpt_attempt=2, family=1, solve=337), 'seconds': 1440}
                 for arm in ('local', 'random', 'gpt')}
    lanes.update(nonlinear={'counts': counts(solve=240, neural_fit=12, epoch=48, tree_fit=3), 'seconds': 900},
                 forward={'counts': counts(solve=412 if mode == 'standard' else 349, neural_fit=6, epoch=132),
                          'seconds': 1320 if mode == 'standard' else 1980})
    return {'version': VERSION, 'mode': mode, 'lanes': lanes, 'unused_transfer': False,
            'owner_rule': 'actual_month_unique_S_run_or_controls_members_and_single_forward_runtime',
            'predecessor_consumption': ({'S': {'counts': counts(ast=32, f=1, gpt_attempt=2, family=1, solve=226), 'seconds': 1590},
                                        'forward': {'counts': counts(solve=5), 'seconds': 1500}}
                                       if mode == 'standard' else {})}


def canonical_policy(value):
    if value is None:
        return None  # Explicit unpartitioned compatibility; never a holdback claim.
    if not isinstance(value, (str, dict)):
        raise TypeError('A frozen monthly partition policy is required')
    expected = frozen_policy(value if isinstance(value, str) else value.get('mode'))
    if not isinstance(value, str) and value != expected:
        raise ValueError('Frozen monthly partition policy cannot change')
    return expected


def read(path, wrapper=None):
    path = project_path(path)
    if not path.is_file() or path.stat().st_size > 2**20:
        raise ValueError('Missing/oversized original allocation evidence: '+str(path))
    value = json.loads(path.read_text('utf-8'))
    if wrapper is not None:
        key, digest = wrapper
        if set(value) != {key, digest} or fingerprint(value[key]) != value[digest]:
            raise ValueError('Allocation evidence hash differs')
        value = value[key]
    return value


def _pair(left, right, wrapper):
    value = read(left, wrapper)
    if value != read(right, wrapper):
        raise ValueError('Original allocation evidence mirrors differ')
    return value


def _controls(authority, prep, directory, *, current):
    outer = prep.get('outer_scope')
    if not isinstance(outer, dict) or outer.get('reference', {}).get('kind') != 'controls':
        raise ValueError('Controls owner needs its actual original three-arm executor')
    ref = outer['reference']
    root = project_path(ref['path'])
    value = read(root/'shared_controls_identity.json', ('value', 'sha256'))
    canonical = authority.directory/'shared_S_controls'/fingerprint({'directory': str(root), 'cutoff': value['source_reference']['cutoff']})
    if (project_path(value['root']) != canonical or value != read(canonical/'identity.json', ('value', 'sha256'))
            or value['directory'] != str(root)
            or value['authority'] != {'directory': str(authority.directory), 'manifest_hash': authority.manifest_hash}
            or fingerprint(value) != outer['scope_hash'] or value['clock'] != outer['clock']
            or value['arm_order'] != ['local', 'random', 'gpt']):
        raise ValueError('Controls owner differs from actual shared executor journal')
    arm = ref['arm']
    if arm not in value['arm_order'] or directory != root/'arms'/arm:
        raise ValueError('Controls work is not the original fixed arm directory')
    index = value['arm_order'].index(arm)
    claim = _pair(canonical/f'{index}-claim.json', root/f'{arm}-claim.json', ('value', 'sha256'))
    if claim != {'identity_hash': fingerprint(value), 'arm': arm, 'index': index}:
        raise ValueError('Missing genuine original controls claim')
    if (fingerprint(value['schedules'][arm]) != prep['work_schedule_hash']
            or value['schedules'][arm]['config'] != prep['config'] or value['spec'] != prep['spec']
            or value['source_reference'] != prep['source_reference']):
        raise ValueError('Controls member changed source or full schedule')
    if current and (canonical/f'{index}-outcome.json').exists():
        raise ValueError('Completed control member cannot reserve new work')
    return f'S:{arm}', {'controls_group': fingerprint(value)}


def _search(authority, bridge_id, *, current, lease, preparation_only=False, verification_cache=None):
    if not isinstance(bridge_id, str) or len(bridge_id) != 64 or any(c not in '0123456789abcdef' for c in bridge_id):
        raise ValueError('Real canonical S bridge required')
    prep_path = authority.directory/'S_preparations'/bridge_id/'identity.json'
    prep = None
    if prep_path.exists():
        prep = read(prep_path, ('record', 'hash'))
        directory = project_path(prep['directory'])
        if (prep != read(directory/'S_preparations'/prep['month']/'identity.json', ('record', 'hash'))
                or prep['key'] != bridge_id
                or bridge_id != fingerprint({'directory': str(directory), 'replay_month': prep['month']})
                or prep['authority_manifest_hash'] != authority.manifest_hash
                or prep['source_hash'] != fingerprint(authority.manifest['source_files'])):
            raise ValueError('S preparation owner/source changed')
        config, spec = prep['config'], prep['spec']
    if preparation_only:
        if prep is None:
            raise ValueError('Baseline has no actual S preparation')
        envelope = prep['execution_envelope']
    else:
        anchor = authority.directory/'search_bridges'/bridge_id/'identity.json'
        bridge = read(anchor, ('record', 'hash'))
        directory = project_path(bridge['run_directory'])
        if (bridge != read(directory/'work_authority'/bridge['replay_month']/'identity.json', ('record', 'hash'))
                or bridge['bridge_id'] != bridge_id
                or bridge_id != fingerprint({'directory': str(directory), 'replay_month': bridge['replay_month']})
                or bridge['authority_manifest_hash'] != authority.manifest_hash
                or bridge['authority_directory'] != str(authority.directory)
                or bridge['source_hash'] != fingerprint(authority.manifest['source_files'])):
            raise ValueError('S bridge owner/source changed')
        manifest = read(directory/'manifest.json', ('identity', 'hash'))
        if (fingerprint(manifest) != bridge['run_identity_hash']
                or manifest['source_files'] != authority.manifest['source_files']):
            raise ValueError('S owner is not its real frozen RunStore')
        config, spec = manifest['config'], manifest['rolling_spec']
        envelope = bridge['execution_envelope']
        # The current authoritative RunStore checkpoint must still contain the
        # original input/exposure. A caller-provided role is insufficient.
        from .neural_research_ticket import _reader
        # All per-row anchors/manifest/schema checks above and input/status/lease
        # checks below still execute. Reuse only the already verified real
        # RunStore snapshot within this single wallet verification transaction.
        cache_key = ('run_store_state', authority.manifest_hash, str(directory), bridge_id,
            bridge['replay_month'], bridge['run_identity_hash'], bridge['source_hash'],
            bridge.get('preparation_hash'), 'S', current)
        if verification_cache is not None and cache_key in verification_cache:
            state = verification_cache[cache_key]
        else:
            store = _reader(directory)
            try:
                state = store.get(bridge['replay_month'])
                if state is not None:
                    state = {k: state[k] for k in ('data_id', 'search_exposure', 'status')}
            finally:
                store.close()
            if (verification_cache is not None and len(verification_cache) < 32
                    and len(json.dumps(state, ensure_ascii=False).encode('utf-8')) <= 65536):
                # Keep only required scalars/exposure, at most 32 bounded entries.
                verification_cache[cache_key] = state
        if (state is None or state['data_id'] != bridge['data_id']
                or state['search_exposure'] != bridge['search_exposure']
                or current and state['status'] != 'running'):
            raise ValueError('S owner monthly journal/input changed or is not running')
        if prep is not None and bridge.get('preparation_hash') != fingerprint(prep):
            raise ValueError('S owner swapped its preparation')
    if (sorted(config['symbols']) != authority.manifest['scope']['universe']
            or spec['task'] != authority.manifest['scope']['task']):
        raise ValueError('S owner is outside the actual authority scope')
    from .work_lease import FrozenWorkLease
    original = FrozenWorkLease.from_dict(envelope['issuer'])
    if (lease.parent_pid != original.parent_pid or lease.parent_created != original.parent_created
            or lease.boot_id != original.boot_id or lease.wall_started_at[:7] != original.wall_started_at[:7]
            or lease.monotonic_started < original.monotonic_started
            or lease.monotonic_deadline > envelope['monotonic_deadline']
            or lease.wall_deadline > envelope['wall_deadline']):
        raise ValueError('Partition work cannot replace its original S execution clock')
    policy = authority.manifest['partition_policy']
    extra = {}
    if policy['mode'] == 'controls':
        if prep is None:
            raise ValueError('Controls require the original preparation/executor journal')
        lane, extra = _controls(authority, prep, directory, current=current)
    else:
        if config['budget_policy'] != 'standard':
            raise ValueError('A controls member cannot borrow a standard S partition')
        lane = 'S'
    return {'lane': lane, 'owner': {'bridge_id': bridge_id, 'run_directory': str(directory)}, **extra}


def _runtime(authority, directory, job=None):
    from .neural_forward_ticket import _LifecycleReader, _verify_authority_binding
    directory = project_path(directory)
    reader = _LifecycleReader(directory)
    try:
        history = reader.verify()
        _verify_authority_binding(reader, authority, history['operations'])
        identity = read(directory/'forward.json', ('identity', 'sha256'))
        # ForwardRuntime serializes its complete source identity in this file.
        body = identity.get('identity', identity)
        if body.get('source_files') != authority.manifest['source_files']:
            raise ValueError('Forward owner source differs from the shared wallet')
        if job is not None and reader.db.execute('SELECT 1 FROM jobs WHERE id=?', (job,)).fetchone() is None:
            raise ValueError('Forward allocation is not an actual lifecycle job')
        return {'runtime_directory': str(directory), 'plan_id': reader.plan.digest}
    finally:
        reader.close()


def _neural_forward(authority, work_key, binding, lease, debit):
    from .neural_forward import FrozenNeuralRecipe
    from .neural_forward_ticket import (
        _LifecycleReader,
        _verify_historical_context,
        read_prepaid_intent,
    )
    intent = read_prepaid_intent(authority, work_key, binding, lease, debit)
    context = intent['context']
    directory = project_path(context['lifecycle_directory'])
    owner = _runtime(authority, directory, context['job'])
    reader = _LifecycleReader(directory)
    try:
        _verify_historical_context(reader, context, FrozenNeuralRecipe.from_dict(intent['recipe']))
    finally:
        reader.close()
    return {'lane': 'forward', 'owner': owner}


def _algorithm_allocation_time(runtime_directory, lease, prefix):
    from .v7_lifecycle import stamp
    from .neural_forward_ticket import _LifecycleReader
    allocation_at = stamp(lease.wall_started_at)
    if prefix is not None:
        from .quarantined_observation import QuarantinedObservation
        reader = _LifecycleReader(runtime_directory)
        try:
            journal = QuarantinedObservation.from_lifecycle(reader)
            if journal is None:
                raise ValueError('Algorithm prefix requires original enrolled observation')
            journal.verify_binding(prefix)
            if stamp(prefix['physical_observation']['wall']) > allocation_at:
                raise ValueError('Algorithm original observation was unavailable at real allocation')
            allocation_at = stamp(prefix['logical_at'])
        finally:
            reader.close()
    else:
        from .quarantined_observation import QuarantinedObservation
        reader = _LifecycleReader(runtime_directory)
        try:
            if QuarantinedObservation.from_lifecycle(reader) is not None:
                raise ValueError('Bound algorithm allocation lost its original observation prefix')
        finally:
            reader.close()
    return allocation_at


def _algorithm_fit(authority, work_key, binding, lease, debit):
    from types import SimpleNamespace

    import pandas as pd

    from .algorithm_validation_producer import _events
    from .neural_forward import FrozenNeuralRecipe
    from .neural_forward_ticket import _LifecycleReader
    from .v7_lifecycle import stamp
    intent = read(authority.directory/'algorithm_work_intents'/(fingerprint(work_key)+'.json'), ('value', 'hash'))
    if set(intent) != {'version', 'plan', 'validation_directory', 'binding', 'work_key', 'lease', 'observation_prefix'}:
        raise ValueError('Unknown dedicated algorithm allocation schema')
    directory = project_path(intent['validation_directory'])
    identity = read(directory/'identity.json', ('value', 'hash'))
    validation = fingerprint(identity)
    with closing(sqlite3.connect((directory/'validation.sqlite').as_uri()+'?mode=ro', uri=True)) as db:
        if db.execute('SELECT * FROM state').fetchall() != [(1, 'frozen', validation)]:
            raise ValueError('Dedicated algorithm state was lost or replaced')
    plan = intent['plan']
    pipeline, stage, cutoff = plan['pipeline_id'], plan['stage'], stamp(plan['cutoff'])
    expected_key = 'algorithm_fit:'+fingerprint({'validation': validation, 'pipeline': pipeline,
        'cutoff': plan['cutoff'], 'stage': stage})
    recipe = FrozenNeuralRecipe.from_dict(identity['recipes'][pipeline])
    limits = counts(**{k: v for k, v in recipe.budget_limits.items() if k != 'seconds'})
    limits['solve'] += 1
    if (intent != read(directory/'fits'/fingerprint(work_key)[:16]/'allocation.json', ('value', 'hash'))
            or identity != read(authority.directory/'algorithm_validations'/(identity['catalogue_id']+'.json'), ('value', 'hash'))
            or identity['directory'] != str(directory) or identity['authority_directory'] != str(authority.directory)
            or identity['authority_manifest_hash'] != authority.manifest_hash
            or identity['source_hash'] != fingerprint(authority.manifest['source_files'])
            or intent['version'] != 'algorithm-work-allocation-v1' or intent['work_key'] != work_key
            or work_key != expected_key or binding != intent['binding']
            or set(binding) != {'kind', 'validation_id', 'fit_plan_hash', 'pipeline_id', 'gates_hash'}
            or binding['kind'] != 'v7-algorithm-validation-fit-v2' or binding['validation_id'] != validation
            or binding['fit_plan_hash'] != fingerprint(plan) or binding['pipeline_id'] != pipeline
            or lease.to_dict() != intent['lease'] or lease.lease_seconds != 300
            or debit != limits or debit != identity['work_counts']):
        raise ValueError('Dedicated algorithm allocation changed its original work')
    if stage == 'V':
        allowed = cutoff.isoformat() == identity['start']
        start, end = identity['start'], identity['V_end']
    elif stage == 'qualifier':
        from .neural_forward_ticket import _LifecycleReader
        reader = _LifecycleReader(identity['runtime_directory'])
        try:
            # This is allocation identity reconstruction inside wallet.verify.
            # Original model admission separately verifies the full artifacts;
            # recursively opening that same wallet here is not an authority.
            events = _events(SimpleNamespace(authority=authority, directory=directory, digest=validation,
                runtime=SimpleNamespace(lifecycle=reader)), verify_models=False)
        finally:
            reader.close()
        results = [e for e in events if e['kind'] == 'endpoint_result' and e['payload']['stage'] == 'V']
        allowed = (len(results) == 1 and results[0]['payload']['score']['selected'] == pipeline
            and stamp(results[0]['recorded_at']) < stamp(identity['qualifier_start'])
            and stamp(identity['qualifier_start']) <= cutoff < stamp(identity['qualifier_end']))
        start, end = cutoff.isoformat(), (cutoff+pd.DateOffset(months=1)).isoformat()
    else:
        raise ValueError('Unknown dedicated algorithm stage')
    expected_plan = {'version': binding['kind'], 'validation_id': validation, 'stage': stage,
        'cutoff': cutoff.isoformat(), 'train_start': (cutoff-pd.DateOffset(months=3)).isoformat(),
        'prediction_start': start, 'prediction_end': end, 'pipeline_id': pipeline,
        'recipe': json.loads(json.dumps(recipe.to_dict())), 'counts': limits, 'lease_seconds': 300,
        'source_id': identity['source_id'], 'plan_id': identity['plan_id'],
        'stream': identity['batch']+':'+stage+':'+pipeline, 'formal_claim': False,
        'serving_eligible': False, 'factor_claim_eligible': False, 'producer': identity['producer']}
    allocation_at = _algorithm_allocation_time(identity['runtime_directory'], lease, intent['observation_prefix'])
    if (not allowed or cutoff != cutoff.normalize().replace(day=1) or plan != expected_plan
            or not stamp(start) <= allocation_at < stamp(end)):
        raise ValueError('Algorithm allocation is outside the frozen original stage/recipe')
    owner = _runtime(authority, identity['runtime_directory'])
    reader = _LifecycleReader(identity['runtime_directory'])
    try:
        operations = reader.verify()['operations']
        bindings = [o['detail'] for o in operations if o['kind'] == 'algorithm_validation_bound'
                    and o['detail']['catalogue_id'] == identity['catalogue_id']]
        if bindings != [{'catalogue_id': identity['catalogue_id'], 'directory': str(directory), 'identity_hash': validation}]:
            raise ValueError('Dedicated algorithm lacks its actual lifecycle binding')
        for batch in (identity['batch'], identity['originating_S_batch']):
            row = reader.exposure.db.execute('SELECT identity FROM validations WHERE batch=?', (batch,)).fetchone()
            expected = {'batch': batch, 'source': identity['source_id'], 'symbols': sorted(identity['symbols']),
                'start': identity['start'], 'end': identity['qualifier_end'], 'candidate_ids': list(identity['recipes']),
                'reference_id': identity['references']['Pv']['pipeline_id'], 'frozen_at': identity['frozen_at']}
            if not row or json.loads(row[0]) != expected:
                raise ValueError('Dedicated algorithm lost actual protected exposure')
    finally:
        reader.close()
    return {'lane': 'forward', 'owner': owner}


def route(authority, work_key, binding, *, current, lease, debit, verification_cache=None):
    """Derive a route from canonical original producer evidence, never role alone."""
    if work_key.startswith('prompt_efficiency:'):
        from .prompt_efficiency import validate_allocation
        return validate_allocation(authority, work_key, binding, lease, debit, current=current,
                                   verification_cache=verification_cache)
    if work_key.startswith('conditional_cold_refit:'):
        from .conditional_cold_refit import validate_allocation
        return validate_allocation(authority, work_key, binding, lease, debit)
    if work_key.startswith('typed_queue_review:'):
        from .structural_queue_review import validate_queue_route
        return validate_queue_route(authority, work_key, binding, lease, debit)
    if work_key.startswith('S-baseline:') and set(binding) == {'role', 'stage', 'preparation_hash', 'bridge_id', 'source_reference'}:
        owner = _search(authority, binding['bridge_id'], current=current, lease=lease, preparation_only=True, verification_cache=verification_cache)
        prep = read(authority.directory/'S_preparations'/binding['bridge_id']/'identity.json', ('record', 'hash'))
        if (work_key != 'S-baseline:'+binding['bridge_id'] or binding['role'] != 'S'
                or binding['stage'] != 'baseline' or binding['preparation_hash'] != fingerprint(prep)
                or binding['source_reference'] != prep['source_reference'] or debit != counts(solve=7)
                or lease.to_dict() != prep['execution_envelope']['issuer']):
            raise ValueError('Baseline allocation intent changed')
        return owner
    if 'bridge_id' in binding and binding.get('role') == 'S':
        owner = _search(authority, binding['bridge_id'], current=current, lease=lease, verification_cache=verification_cache)
        bridge = read(authority.directory/'search_bridges'/binding['bridge_id']/'identity.json', ('record', 'hash'))
        if (set(binding) != {'bridge_id', 'bridge_identity_hash', 'role', 'replay_month', 'work'}
                or binding['bridge_identity_hash'] != fingerprint(bridge)
                or binding['replay_month'] != bridge['replay_month']
                or not work_key.startswith('work-authority-bridge-v1:'+binding['bridge_id']+':')):
            raise ValueError('S allocation is not the actual original bridge')
        if binding['work'].get('kind') == 'structure_review':
            from .structural_review import validate_review_route
            validate_review_route(authority, work_key, binding, bridge, lease, debit)
            return {**owner, 'S_lane': owner['lane'], 'lane': 'nonlinear'}
        return owner
    for key, prefix in [('research_bridge_id', 'neural_S:'), ('nonlinear_bridge_id', 'S_nonlinear:')]:
        if key in binding and work_key.startswith(prefix) and binding.get('role') == 'S' and binding.get('serving_eligible') is False:
            owner = _search(authority, binding[key], current=current, lease=lease, verification_cache=verification_cache)
            from .neural_forward import FrozenNeuralRecipe
            from .neural_research_ticket import _reader
            from .s_nonlinear_ticket import _plan
            bridge = read(authority.directory/'search_bridges'/binding[key]/'identity.json', ('record', 'hash'))
            store = _reader(bridge['run_directory'])
            try:
                state = store.get(bridge['replay_month'])
                plan = state.get('nonlinear_work_plan')
                if not isinstance(plan, dict) or 'body' not in plan:
                    raise ValueError('Nonlinear lane needs the actual frozen controller plan')
                recipe = FrozenNeuralRecipe.from_dict(plan['body']['recipe'])
                kind = 'complete_neural_S' if key == 'research_bridge_id' else binding.get('kind')
                plan = _plan(bridge, state, recipe, kind, store)
                selected = next(w for w in plan['body']['work'] if w['kind'] == kind)
                if (debit != counts(**selected['counts']) or lease.lease_seconds != selected['seconds']
                        or binding.get('input_id') != state.get('nonlinear_input_data_id')
                        or binding.get('pipeline_id') != recipe.pipeline_id):
                    raise ValueError('Nonlinear debit differs from its complete frozen recipe/work')
                expected_key = (fingerprint({'bridge': fingerprint(bridge), 'kind': kind})
                    if key == 'nonlinear_bridge_id' else fingerprint({'bridge': fingerprint(bridge),
                        'controller_data': bridge['data_id'], 'input': state['nonlinear_input_data_id'],
                        'pipeline': recipe.pipeline_id}))
                if work_key != prefix+expected_key or key == 'nonlinear_bridge_id' and binding.get('plan_hash') != plan['hash']:
                    raise ValueError('Nonlinear stable work key or original plan changed')
            finally:
                store.close()
            return {**owner, 'S_lane': owner['lane'], 'lane': 'nonlinear'}
    if work_key.startswith('nominal_factor_cold:'):
        from .nominal_factor_census import validate_nominal_factor_route
        return validate_nominal_factor_route(authority, work_key, binding, lease, debit)
    if work_key.startswith('nominal_cold:'):
        from .nominal_cold_conformance import validate_nominal_route
        return validate_nominal_route(authority, work_key, binding, lease, debit)
    if work_key.startswith('forward_preparation:'):
        suffix = work_key.removeprefix('forward_preparation:')
        intent = read(authority.directory/'forward_preparations'/suffix/'intent.json', ('value', 'sha256'))
        if (intent != read(project_path(intent['runtime_directory'])/'forward_preparations'/suffix/'intent.json', ('value', 'sha256'))
                or intent['authority_manifest_hash'] != authority.manifest_hash
                or debit != intent['counts'] or lease.to_dict() != intent['lease']
                or binding != {'prepare_terms': fingerprint(intent), 'job': intent['job'],
                               'pipeline_id': intent['pipeline']['pipeline_id'], 'purpose': intent['purpose']}):
            raise ValueError('Forward preparation cannot borrow another allocation')
        return {'lane': 'forward', 'owner': _runtime(authority, intent['runtime_directory'], intent['job'])}
    if work_key.startswith('neural_forward:'):
        return _neural_forward(authority, work_key, binding, lease, debit)
    if work_key.startswith('algorithm_fit:'):
        return _algorithm_fit(authority, work_key, binding, lease, debit)
    raise ValueError('Unknown work provenance cannot select a monthly partition')


def charge(policy, partitions, owners, month, allocation, debit, seconds):
    """Pure deterministic replay/precheck; called inside the wallet transaction."""
    lane = allocation['lane']
    if lane not in policy['lanes']:
        raise ValueError('Unknown frozen partition')
    lane_key = month+'|'+lane
    old = partitions.get(lane_key, {'counts': counts(), 'seconds': 0})
    limit = policy['lanes'][lane]
    new = {'counts': {k: old['counts'][k]+debit[k] for k in KINDS}, 'seconds': old['seconds']+seconds}
    if any(new['counts'][k] > limit['counts'][k] for k in KINDS) or new['seconds'] > limit['seconds']:
        from .monthly_work_authority import BudgetExhausted
        raise BudgetExhausted('Nontransferable monthly '+lane+' quota exhausted')
    claims = {month+'|'+(allocation.get('S_lane', lane) if lane != 'forward' else 'forward'): allocation['owner']}
    if 'controls_group' in allocation:
        claims[month+'|controls_group'] = allocation['controls_group']
    for key, owner in claims.items():
        if key in owners and owners[key] != owner:
            raise ValueError('Another actual owner already holds this monthly partition')
        if lane == 'nonlinear' and key not in owners:
            raise ValueError('Nonlinear work requires its already charged actual S owner')
    for key, owner in claims.items():
        owners[key] = owner
    partitions[lane_key] = new
