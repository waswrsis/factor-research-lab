"""S-only complete-recipe tickets backed by the actual shared work authority.

No serving role, forward ticket or self-declared resource certificate is accepted.
Historical exposed inputs require the controller's sealed component identity;
live inputs additionally re-read the actual receipt archive as of the cutoff.
"""
from __future__ import annotations

import copy
import json
import os
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import psutil
from filelock import FileLock

from factorlab.protocol import fingerprint, project_path

from .exposure import ExposureLedger
from .neural_forward import FrozenNeuralRecipe
from .neural_forward_ticket import open_existing_authority, runtime_identity
from .neural_research import research_capacity
from .nonlinear_launcher import _load_data, save_worker_data
from .nonlinear_orchestration import causal_worker_data, input_data_identity
from .rolling import build_rolling_data
from .v7_feed import ReceiptArchive
from .v7_run import RunStore, source_files
from .v7_storage import immutable_record
from .work_authority_bridge import WorkAuthorityBridge
from .work_lease import FrozenWorkLease, _boot_id

VERSION = 'v7-neural-S-ticket-v1'


def _read(path):
    path = project_path(path)
    if not path.is_file() or path.stat().st_size > 16 * 2**20:
        raise ValueError('Missing or oversized S evidence')
    return json.loads(path.read_text('utf-8'))


def _canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _bridge_identity(path):
    value = _read(path)
    if set(value) != {'record', 'hash'} or value['hash'] != fingerprint(value['record']):
        raise ValueError('S bridge identity checksum changed')
    return value['record']


def _reader(directory):
    """Open actual RunStore and exposure read-only; never initialize anything."""
    result = RunStore.__new__(RunStore)
    result.directory = project_path(directory)
    manifest = _read(result.directory / 'manifest.json')
    if set(manifest) != {'identity', 'hash'} or fingerprint(manifest['identity']) != manifest['hash']:
        raise ValueError('RunStore manifest changed')
    result.identity = manifest['identity']
    result.db = sqlite3.connect((result.directory / 'state.sqlite').as_uri() + '?mode=ro', uri=True)
    result.exposure = ExposureLedger.__new__(ExposureLedger)
    result.exposure.path = project_path(result.identity['exposure_path'])
    result.exposure.clock = lambda: pd.Timestamp.now(tz='UTC')
    try:
        result.exposure.db = sqlite3.connect(result.exposure.path.as_uri() + '?mode=ro', uri=True)
        result.verify()
        if result.identity['source_files'] != source_files():
            raise ValueError('S RunStore source changed')
        return result
    except BaseException:
        result.db.close()
        if hasattr(result.exposure, 'db'):
            result.exposure.close()
        raise


def _context(store, month, state, bridge, recipe):
    exposed = state['search_exposure']
    if (state['status'] != 'running' or state['cutoff'][:7] != month
            or exposed['data_id'] != state['data_id']
            or state.get('nonlinear_input_data_id') is None
            or _canonical(state.get('frozen_reference_pool')) != _canonical(recipe.pool.to_dict())
            or recipe.rolling_spec.digest != store.identity['rolling_hash']
            or bridge['run_identity_hash'] != fingerprint(store.identity)
            or bridge['data_id'] != state['data_id'] or bridge['search_exposure'] != exposed
            or state.get('authority_bridge', {}).get('identity_hash') != fingerprint(bridge)):
        raise ValueError('S recipe/input/bridge differs from the actual running controller')
    store.exposure.assert_prompt_search(**{k: v for k, v in exposed.items() if k != 'data_id'})
    result = {'role': 'S', 'run_directory': str(store.directory), 'replay_month': month,
            'cutoff': state['cutoff'], 'controller_data_id': state['data_id'],
            'input_id': state['nonlinear_input_data_id'], 'source_id': exposed['source'],
            'symbols': sorted(store.identity['config']['symbols']), 'exposure': exposed,
            'bridge_id': bridge['bridge_id'], 'bridge_identity_hash': fingerprint(bridge),
            'run_identity_hash': fingerprint(store.identity), 'serving_eligible': False,
            'automatic_adoption': False, 'formal_claim': False}
    from .s_observation_source import preparation_context
    result.update(preparation_context(store, month, bridge))
    return result


def _gates(gates, data, recipe, context):
    from .s_nonlinear_gates import verify_shared_gates
    return verify_shared_gates(gates, data, recipe, context,
                              source_hash=fingerprint(source_files()), require_resource=True)


def _input(data, recipe, context, feed_directory, feed_identity):
    data = causal_worker_data(data, context['cutoff'])
    if input_data_identity(data, context['cutoff']) != context['input_id']:
        raise ValueError('S payload differs from the checkpointed controller input')
    if feed_identity is not None:
        if feed_directory is None:
            raise ValueError('Live S needs its actual immutable receipt archive')
        archive = ReceiptArchive(feed_directory, tuple(context['symbols']))
        try:
            if archive.identity != feed_identity or archive.identity != context['source_id']:
                raise ValueError('S archive source identity differs from shared exposure')
            cutoff = pd.Timestamp(context['cutoff'])
            if context.get('observation_prefix') is not None:
                from .s_observation_source import snapshot_from_context
                bars = snapshot_from_context(context, feed_directory)
            else:
                bars = archive.snapshot(cutoff, start=cutoff-(pd.DateOffset(months=6) if recipe.rolling_spec.window_days is None
                    else pd.Timedelta(days=180))-pd.Timedelta(hours=1234),
                                        archive_delay_seconds=recipe.rolling_spec.availability_delay_seconds)
            actual = build_rolling_data(bars, recipe.rolling_spec)
            if input_data_identity(actual, cutoff) != context['input_id']:
                raise ValueError('S payload is not the real as-of receipt snapshot')
        finally:
            archive.close()
    elif feed_directory is not None:
        raise ValueError('Historical controller cannot borrow an unrelated receipt archive')
    return data


def _outer(bridge, lease, *, current):
    envelope = bridge['execution_envelope']
    original = FrozenWorkLease.from_dict(envelope['issuer'])
    if (lease.parent_pid != original.parent_pid or lease.parent_created != original.parent_created
            or lease.boot_id != original.boot_id or lease.wall_started_at[:7] != original.wall_started_at[:7]
            or lease.monotonic_started < original.monotonic_started
            or lease.monotonic_deadline > envelope['monotonic_deadline']
            or pd.Timestamp(lease.wall_deadline) > pd.Timestamp(envelope['wall_deadline'])):
        raise ValueError('S ticket would replace or extend the original execution envelope')
    if current and (_boot_id() != original.boot_id or pd.Timestamp.now(tz='UTC') >= pd.Timestamp(envelope['wall_deadline'])):
        raise ValueError('Original S envelope no longer permits work')


@dataclass(frozen=True)
class ResearchFitTicket:
    path: Path
    value: dict
    _digest: str = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, 'path', project_path(self.path))
        object.__setattr__(self, 'value', _canonical(copy.deepcopy(self.value)))
        expected = {'version', 'authority_directory', 'authority_manifest_hash', 'reservation_id',
                    'terms_hash', 'lease', 'lease_hash', 'context', 'bridge', 'recipe', 'gates',
                    'runtime', 'source_hash', 'feed_directory', 'feed_identity', 'data'}
        if set(self.value) != expected or self.value['version'] != VERSION:
            raise ValueError('Unknown S ticket schema; forward tickets cannot authorize research')
        lease = FrozenWorkLease.from_dict(self.value['lease'])
        if lease.lease_seconds != 300 or lease.digest != self.value['lease_hash']:
            raise ValueError('Full three-fold S requires its single original 300-second lease')
        recipe = FrozenNeuralRecipe.from_dict(self.value['recipe'])
        key = _key(self.value['context'], recipe)
        if self.path != project_path(self.value['authority_directory']) / 'research_tickets' / key / 'ticket.json':
            raise ValueError('Noncanonical S ticket path')
        object.__setattr__(self, '_digest', fingerprint(self.value))

    @property
    def digest(self):
        if fingerprint(self.value) != self._digest:
            raise ValueError('Frozen S ticket mutated')
        return self._digest

    @classmethod
    def load(cls, path):
        value = _read(path)
        if set(value) != {'value', 'sha256'} or fingerprint(value['value']) != value['sha256']:
            raise ValueError('S ticket checksum changed')
        return cls(path, value['value'])

    def verify(self, *, mode='launch'):
        if mode not in {'launch', 'worker', 'running_parent', 'completed'}:
            raise ValueError('Unknown S ticket verification role')
        value = self.value
        if _read(self.path) != {'value': value, 'sha256': self.digest}:
            raise ValueError('Persisted S ticket changed')
        if _read(self.path.parent / 'preparation.json') != {
                'reservation_id': value['reservation_id'], 'lease': value['lease']}:
            raise ValueError('Original prepaid S preparation evidence changed')
        authority = open_existing_authority(value['authority_directory'])
        record = authority.get(value['reservation_id'])
        recipe = FrozenNeuralRecipe.from_dict(value['recipe'])
        base = {k: value[k] for k in value if k not in {
            'data', 'authority_manifest_hash', 'reservation_id', 'terms_hash', 'lease', 'lease_hash'}}
        if (authority.manifest_hash != value['authority_manifest_hash']
                or fingerprint(authority.manifest['source_files']) != value['source_hash']
                or record['binding'] != _binding(base, recipe)
                or any(record[k] != value[k] for k in ('terms_hash', 'lease', 'lease_hash'))
                or record['counts'] != {k: research_capacity(recipe).get(k, 0) for k in record['counts']}
                or runtime_identity() != value['runtime']):
            raise ValueError('S ticket differs from canonical prepaid work/runtime')
        wanted = {'launch': 'reserved', 'worker': 'running', 'running_parent': 'running', 'completed': 'completed'}[mode]
        if record['status'] != wanted:
            raise ValueError('S work is consumed or not in the required canonical state')
        lease = FrozenWorkLease.from_dict(value['lease'])
        _outer(value['bridge'], lease, current=mode != 'completed')
        if mode != 'completed':
            lease.assert_parent_alive(as_worker=mode == 'worker')
            if mode != 'worker' and os.getpid() != lease.parent_pid:
                raise ValueError('Only the original S issuer can launch/verify parent completion')
        if mode == 'worker' and record['process_identity'] != {'pid': os.getpid(), 'created': psutil.Process().create_time()}:
            raise ValueError('S worker is not the registered original direct child')
        reader = _reader(value['context']['run_directory'])
        try:
            month = value['context']['replay_month']
            state = reader.get(month)
            bridge_path = reader.directory / 'work_authority' / month / 'identity.json'
            if (_bridge_identity(bridge_path) != value['bridge']
                    or _bridge_identity(authority.directory / 'search_bridges' / value['bridge']['bridge_id'] / 'identity.json') != value['bridge']):
                raise ValueError('S original bridge anchor changed')
            # Historical completion may survive the controller's finalization;
            # all other context fields and the original exposure still match.
            state = copy.deepcopy(state)
            from .s_nonlinear_ticket import _plan
            _plan(value['bridge'], state, recipe, 'complete_neural_S', reader)
            if mode == 'completed' and state['status'] in {'complete', 'finalizing'}:
                state['status'] = 'running'
            if _context(reader, month, state, value['bridge'], recipe) != value['context']:
                raise ValueError('Actual S controller context changed')
            if (value['bridge']['authority_manifest_hash'] != authority.manifest_hash
                    or value['bridge']['authority_directory'] != str(authority.directory)
                    or reader.identity['source_files'] != authority.manifest['source_files']
                    or authority.manifest['scope']['task'] != 'volatility'
                    or authority.manifest['scope']['universe'] != value['context']['symbols']
                    or reader.identity['feed_identity'] != value['feed_identity']):
                raise ValueError('S uses another wallet, universe or actual source')
            for key in ('arrays', 'metadata'):
                path = project_path(value['data'][key])
                if path.parent != self.path.parent / 'payload' or path.stat().st_size > 16 * 2**20:
                    raise ValueError('S payload path/size is not canonical')
            data = _load_data({'data': value['data'], 'rolling': asdict(recipe.rolling_spec)})
            data = _input(data, recipe, value['context'], value['feed_directory'], value['feed_identity'])
            _gates(value['gates'], data, recipe, value['context'])
        finally:
            reader.close()
        if mode != 'completed':
            lease.assert_parent_alive(as_worker=mode == 'worker')
        return authority, record, data, recipe


def _key(context, recipe):
    return fingerprint({'bridge': context['bridge_identity_hash'], 'controller_data': context['controller_data_id'],
                        'input': context['input_id'], 'pipeline': recipe.pipeline_id})


def _binding(base, recipe):
    return {'ticket_terms': fingerprint(base), 'research_bridge_id': base['context']['bridge_id'],
            'role': 'S', 'input_id': base['context']['input_id'], 'pipeline_id': recipe.pipeline_id,
            'serving_eligible': False}


def issue_research_ticket(bridge, data, recipe, *, gates, feed_directory=None):
    if type(bridge) is not WorkAuthorityBridge or type(recipe) is not FrozenNeuralRecipe:
        raise TypeError('Real S authority bridge and complete frozen recipe required')
    bridge._guard()
    if bridge._active is not None:
        raise ValueError('Finish parent S work before reserving an isolated neural work item')
    context = _context(bridge.store, bridge.month, bridge.state, bridge.identity, recipe)
    from .s_nonlinear_ticket import _plan
    _plan(bridge.identity, bridge.state, recipe, 'complete_neural_S', bridge.store)
    folder = bridge.authority.directory / 'research_tickets' / _key(context, recipe)
    path = folder / 'ticket.json'
    gates = _canonical(gates)
    feed_directory = str(project_path(feed_directory)) if feed_directory is not None else None
    with FileLock(str(folder) + '.lock', timeout=0):
        if path.exists():
            existing = ResearchFitTicket.load(path)
            if existing.value['gates'] != gates or existing.value['feed_directory'] != feed_directory:
                raise ValueError('S retry cannot change gates/source or obtain a fresh lease')
            bridge.authority.get(existing.value['reservation_id'])
            return existing
        if folder.exists():
            raise ValueError('Partial S preparation cannot reset its prepaid work')
        lease = FrozenWorkLease.issue(300)
        _outer(bridge.identity, lease, current=True)
        base = {'version': VERSION, 'authority_directory': str(bridge.authority.directory),
                'context': context, 'bridge': _canonical(bridge.identity), 'recipe': _canonical(recipe.to_dict()),
                'gates': gates, 'runtime': runtime_identity(),
                'source_hash': fingerprint(bridge.store.identity['source_files']),
                'feed_directory': feed_directory, 'feed_identity': bridge.store.identity['feed_identity']}
        record = bridge.authority._reserve_with_lease('neural_S:' + _key(context, recipe),
            counts=research_capacity(recipe), binding=_binding(base, recipe), lease=lease)
        if record['status'] != 'reserved' or record['lease'] != lease.to_dict():
            raise ValueError('Existing S reservation cannot be revived after missing preparation')
        try:
            folder.mkdir(parents=True)
            immutable_record(folder / 'preparation.json', {'reservation_id': record['reservation_id'], 'lease': record['lease']})
            data = _input(data, recipe, context, feed_directory, bridge.store.identity['feed_identity'])
            _gates(gates, data, recipe, context)
            payload = save_worker_data(folder / 'payload', data, cutoff=context['cutoff'])
            value = {**base, 'data': payload, **{k: record[k] for k in (
                'authority_manifest_hash', 'reservation_id', 'terms_hash', 'lease', 'lease_hash')}}
            ticket = ResearchFitTicket(path, value)
            immutable_record(path, {'value': ticket.value, 'sha256': ticket.digest})
            bridge._guard()
            lease.assert_parent_alive()
            return ticket
        except BaseException as error:
            from .s_nonlinear_gates import NonlinearGateClosed
            if isinstance(error, NonlinearGateClosed):
                error.reservation_id = record['reservation_id']
            try:
                bridge.authority.fail_consumed(record['reservation_id'], 'S_preparation:' + type(error).__name__)
            except BaseException as accounting:  # noqa: BLE001 -- preserve original failure.
                error.add_note('Consumed S accounting: ' + type(accounting).__name__)
            raise
