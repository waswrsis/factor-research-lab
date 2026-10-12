"""Dedicated algorithm shadow producer; no factor, serving or alpha permission.

The fixed child trains one complete ensemble and same-cutoff Pv/K0 references.
The parent publishes only through a pending SQLite witness under the original
lease. Historical admission checks original observations, never renews work.
Hashes establish local trusted-producer provenance, not administrator security.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock
from threadpoolctl import threadpool_limits


def _code_root():
    """Both the isolated staging location and eventual canonical package path."""
    here = Path(__file__).resolve()
    root = here.parents[2] if here.parent.name == 'discovery' else here.parents[3]
    if not (root/'factorlab/discovery/neural_forward.py').is_file():
        raise ValueError('Fixed algorithm worker is outside its project source tree')
    return root


# The staged worker loads only its fixed sibling, never caller-supplied code.
if __package__ in (None, ''):
    sys.path.insert(0, str(_code_root()))
    __package__ = 'factorlab.discovery'
if 'factorlab.discovery.algorithm_validation' not in sys.modules:
    _spec = importlib.util.spec_from_file_location('factorlab.discovery.algorithm_validation',
        Path(__file__).with_name('algorithm_validation.py'))
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _mod
    _spec.loader.exec_module(_mod)

from factorlab.protocol import fingerprint, project_path

from . import algorithm_validation as protocol_module
from .algorithm_shortlist import _copy, _read, _write
from .evidence import file_hash
from .joint_pool import FrozenPool, fit_pool
from .known_library import build_feature_history
from .monthly_work_authority import DEFAULT_LIMITS, source_files
from .neural_forward import (
    FrozenNeuralRecipe,
    NeuralForwardModel,
    _fit_neural_forward_payload,
    _joint,
)
from .neural_forward_admission import _kernel as _strict_kernel
from .neural_forward_admission import _observation as _verify_observation
from .neural_forward_launcher import _environment, _identity, _member, _observation
from .neural_forward_ticket import (
    _LifecycleReader,
    _snapshot_from_actual_source,
    open_existing_authority,
    runtime_identity,
)
from .nonlinear_launcher import _kernel_snapshot, _load_data, _validate_kernel
from .nonlinear_orchestration import input_data_identity
from .process_policy import isolated_process_kwargs, terminate_process_tree
from .reserved_work_budget import ReservedChannelBudget
from .v7_lifecycle import stamp
from .v7_storage import immutable_record
from .work_lease import FrozenWorkLease

VERSION = 'v7-algorithm-shadow-producer-v1'
MAX_EVENTS = 50000


def verify_gates(gates, data, recipe, plan, *, forward_job=None):
    from .nonlinear_runtime import verify_pre_resource_artifacts
    context = None
    if forward_job is not None:
        from .quarantined_observation import job_binding
        reader = _LifecycleReader(forward_job['lifecycle_directory'])
        try:
            prefix = job_binding(reader, forward_job['job'])
            row = reader.db.execute('SELECT cutoff,stream,pipeline FROM jobs WHERE id=?', (forward_job['job'],)).fetchone()
            if (not row or row[0] != plan['cutoff'] or row[1] != plan['stream']
                    or json.loads(row[2])['recipe'] != _copy(recipe.to_dict())):
                raise ValueError('Typed gates differ from the original complete job')
            context = {'cutoff': plan['cutoff'], 'source_id': plan['source_id'],
                       'symbols': list(reader.plan.symbols), **forward_job}
            if prefix is not None:
                context['observation_prefix'] = prefix
        finally:
            reader.close()
    if type(gates) is dict and set(gates) == {'shared_S'}:
        if context is None:
            raise ValueError('Shared S typed gates require the original job')
        from .s_nonlinear_gates import verify_shared_gates
        report = verify_shared_gates(gates['shared_S'], data, recipe, context,
                                     source_hash=fingerprint(source_files()), require_resource=False)
    elif type(gates) is dict and set(gates) == {'directory', 'artifacts'}:
        report = verify_pre_resource_artifacts(gates['directory'], data, plan['cutoff'], recipe.pool,
            gates['artifacts'], source_id=plan['source_id'], config=recipe.config,
            forward_job=forward_job if context is not None and context.get('observation_prefix') is not None else None)
    else:
        raise ValueError('Algorithm work requires actual quarterly and known-component gates')
    if report['test_only']:
        raise ValueError('Artificial gates cannot authorize algorithm production')
    return report


def _key(plan):
    return 'algorithm_fit:' + fingerprint({'validation': plan['validation_id'],
        'pipeline': plan['pipeline_id'], 'cutoff': plan['cutoff'], 'stage': plan['stage']})


def _fit_path(protocol, plan):
    return protocol.directory / 'fits' / fingerprint(_key(plan))[:16] / 'prepared.json'


class _Ticket:
    def __init__(self, path):
        self.path = project_path(path)
        self.value = _read(self.path)
        self.digest = fingerprint(self.value)

    def verify(self, statuses):
        v, p = self.value, self.value['plan']
        identity_path = project_path(v['validation_directory']) / 'identity.json'
        identity = _read(identity_path)
        if (set(v) != {'version', 'plan', 'job', 'authority_directory', 'authority_manifest_hash',
                'reservation_id', 'terms_hash', 'lease', 'lease_hash', 'validation_directory',
                'gates', 'data', 'snapshot_ref', 'shadow_only', 'serving_eligible', 'formal_claim'}
                or v['version'] != protocol_module.TICKET_VERSION
                or v['shadow_only'] is not True or v['formal_claim'] is not False
                or v['serving_eligible'] is not False
                or _read(self.path) != v or p['validation_id'] != fingerprint(identity)
                or identity['implementation'] != protocol_module._implementation()
                or identity['source_hash'] != fingerprint(source_files())
                or identity['numeric_runtime'] != runtime_identity()
                or _read(project_path(identity['anchor'])) != identity
                or self.path != identity_path.parent / 'fits' / fingerprint(_key(p))[:16] / 'prepared.json'):
            raise ValueError('Typed algorithm ticket identity/path/source changed')
        recipe = FrozenNeuralRecipe.from_dict(identity['recipes'][p['pipeline_id']])
        limits = {k: recipe.budget_limits.get(k, 0) for k in DEFAULT_LIMITS}
        limits['solve'] += 1
        if (p['recipe'] != _copy(recipe.to_dict()) or p['counts'] != limits or limits != identity['work_counts']
                or p['lease_seconds'] != 300 or p['source_id'] != identity['source_id']
                or p['plan_id'] != identity['plan_id'] or p['stage'] not in {'V', 'qualifier'}):
            raise ValueError('Algorithm recipe/schedule changed')
        authority = open_existing_authority(v['authority_directory'])
        record = authority.get(v['reservation_id'])
        binding = {'kind': protocol_module.TICKET_VERSION, 'validation_id': p['validation_id'],
            'fit_plan_hash': fingerprint(p), 'pipeline_id': p['pipeline_id'], 'gates_hash': fingerprint(v['gates'])}
        if (record['status'] not in statuses or record['binding'] != binding
                or record['work_key'] != _key(p) or record['counts'] != limits
                or any(v[k] != record[k] for k in ('authority_directory', 'authority_manifest_hash',
                    'terms_hash', 'lease', 'lease_hash'))
                or identity['authority_directory'] != v['authority_directory']
                or identity['authority_manifest_hash'] != v['authority_manifest_hash']
                or _read(self.path.parent / 'preparation.json') != {
                    'reservation_id': record['reservation_id'], 'lease': record['lease']}):
            raise ValueError('Algorithm reservation is not original canonical prepaid work')
        reader = _LifecycleReader(identity['runtime_directory'])
        try:
            snap, actual, _ = _snapshot_from_actual_source(reader, v['snapshot_ref'], recipe)
            if fingerprint(snap) != v['snapshot_ref']['record_hash'] or snap['job'] != v['job']:
                raise ValueError('Algorithm actual snapshot changed')
        finally:
            reader.close()
        for key, name in (('arrays', 'data.npz'), ('metadata', 'metadata.parquet')):
            if project_path(v['data'][key]) != self.path.parent / 'payload' / name:
                raise ValueError('Algorithm payload escaped its fixed directory')
        data = _load_data({'data': v['data'], 'rolling': asdict(recipe.rolling_spec)})
        if input_data_identity(data, p['cutoff']) != input_data_identity(actual, p['cutoff']):
            raise ValueError('Algorithm payload differs from actual cutoff snapshot')
        verify_gates(v['gates'], actual, recipe, p,
            forward_job={'lifecycle_directory': identity['runtime_directory'], 'job': v['job']})
        return authority, record, actual, recipe


class _Witness:
    def __init__(self, ticket, *, prepare=False):
        self.ticket = ticket
        self.directory = ticket.path.parent / 'publication'
        self.path = self.directory / 'witness.sqlite'
        self.anchor = ticket.path.parent / 'publication_anchor.json'
        self.identity = {'version': VERSION, 'ticket_digest': ticket.digest,
            'reservation_id': ticket.value['reservation_id'], 'lease_hash': ticket.value['lease_hash'],
            'authority_manifest_hash': ticket.value['authority_manifest_hash'],
            'directory': str(self.directory)}
        if prepare:
            if self.directory.exists() or self.anchor.exists():
                raise ValueError('Algorithm publication already attempted; cannot reset')
            _write(self.anchor, self.identity)
            _write(self.directory / 'identity.json', self.identity)
            with closing(sqlite3.connect(self.path)) as db, db:
                db.execute('PRAGMA synchronous=FULL')
                db.execute('CREATE TABLE witness(id INTEGER PRIMARY KEY CHECK(id=1), identity TEXT, status TEXT, proof TEXT, hash TEXT)')
                db.execute('INSERT INTO witness VALUES(1,?,\'pending\',NULL,NULL)', (fingerprint(self.identity),))
        self._row()

    def _row(self):
        if (_read(self.anchor) != self.identity or _read(self.directory / 'identity.json') != self.identity
                or not self.path.is_file()):
            raise ValueError('Algorithm publication anchor/database missing or changed')
        with closing(sqlite3.connect(self.path.as_uri()+'?mode=ro', uri=True)) as db:
            rows = db.execute('SELECT * FROM witness').fetchall()
        if len(rows) != 1 or rows[0][:2] != (1, fingerprint(self.identity)):
            raise ValueError('Algorithm publication singleton changed')
        row = rows[0]
        if row[2] == 'pending' and row[3:] == (None, None):
            return row
        if row[2] != 'completed' or fingerprint(json.loads(row[3])) != row[4]:
            raise ValueError('Algorithm publication proof changed')
        return row

    def complete(self, proof):
        self._row()
        if set(proof) != {'receipt_hash', 'model_digest', 'k0_digest', 'actual_counts_hash', 'kernel', 'observation'}:
            raise ValueError('Unknown algorithm publication proof')
        with closing(sqlite3.connect(self.path.as_uri()+'?mode=rw', uri=True)) as db, db:
            db.execute('PRAGMA synchronous=FULL')
            db.execute('BEGIN IMMEDIATE')
            changed = db.execute("UPDATE witness SET status='completed',proof=?,hash=? WHERE id=1 AND identity=? AND status='pending'",
                (json.dumps(proof, sort_keys=True, allow_nan=False), fingerprint(proof), fingerprint(self.identity)))
            if changed.rowcount != 1:
                raise ValueError('Algorithm publication is once only')
            FrozenWorkLease.from_dict(self.ticket.value['lease']).assert_parent_alive()

    def read(self):
        row = self._row()
        if row[2] != 'completed':
            raise ValueError('Algorithm artifact has no completed publication witness')
        return json.loads(row[3])


def _outputs(ticket, record, first, last):
    directory = ticket.path.parent / 'launch'
    result, membership = _read(directory / 'result.json'), _read(directory / 'membership.json')
    lease = FrozenWorkLease.from_dict(record['lease'])
    parent, process = {'pid': lease.parent_pid, 'created': lease.parent_created}, record['process_identity']
    if (result.get('version') != VERSION or result.get('ticket_digest') != ticket.digest
            or result.get('process') != process or result.get('lease_hash') != lease.digest
            or membership != {'ticket_digest': ticket.digest, 'process': process, 'parent': parent,
                'kernel': membership.get('kernel')}):
        raise ValueError('Algorithm result/membership belongs to different work')
    for sample in (membership['kernel'], result['kernel']):
        _strict_kernel(first, sample, last, lease, parent, process)
    authority = open_existing_authority(ticket.value['authority_directory'])
    counts = ReservedChannelBudget.verify_counts(authority, record['reservation_id'])
    if counts['test_only'] or not counts['finalized'] or counts['actual_counts'] != result['actual_counts']:
        raise ValueError('Algorithm costs differ from durable actual child chain')
    model = NeuralForwardModel.load(directory / 'model.json')
    k0 = _joint(json.loads((directory / 'k0.json').read_text('utf-8')))
    recipe = FrozenNeuralRecipe.from_dict(ticket.value['plan']['recipe'])
    if (model.digest != result['model_digest'] or k0.digest != result['k0_digest']
            or model.pipeline_id != recipe.pipeline_id
            or model.receipt.get('payload_origin') != 'isolated_forward_worker'
            or model.baseline.pool.to_dict() != recipe.pool.to_dict()
            or k0.pool.to_dict() != FrozenPool().to_dict()
            or model.receipt['cutoff'] != ticket.value['plan']['cutoff']
            or k0.receipt['cutoff'] != ticket.value['plan']['cutoff']):
        raise ValueError('Algorithm model/reference recipe or cutoff differs')
    return model, k0, counts['actual_counts'], result


def launch(protocol, prepared):
    protocol.verify()
    p = prepared['plan']
    if protocol.read_prepared(p['pipeline_id'], p['cutoff'], stage=p['stage']) != _copy(prepared):
        raise ValueError('Only the actual original prepared algorithm ticket is launchable')
    ticket = _Ticket(_fit_path(protocol, p))
    authority, record, _, _ = ticket.verify(('reserved',))
    lease = FrozenWorkLease.from_dict(record['lease'])
    directory, process = ticket.path.parent / 'launch', None
    with FileLock(str(ticket.path.parent / 'launch.lock'), timeout=0):
        if directory.exists() or (ticket.path.parent / 'publication_anchor.json').exists():
            raise ValueError('Algorithm launch already attempted; no execution retry')
        directory.mkdir()
        try:
            lease.assert_parent_alive()
            witness = _Witness(ticket, prepare=True)
            first = _kernel_snapshot()
            _validate_kernel(first, first)
            _member(first, _identity())
            command = [sys.executable, '-B', str(Path(__file__).resolve()), '--ticket', str(ticket.path)]
            with (directory / 'worker.log').open('xb') as log:
                process = subprocess.Popen(command, cwd=str(_code_root()),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    env=_environment(), **isolated_process_kwargs())
                identity = _identity(process.pid)
                authority.start(record['reservation_id'], identity)
                _write(directory / 'start.json', {'ticket_digest': ticket.digest, 'process': identity, 'lease_hash': lease.digest})
                while process.poll() is None:
                    lease.assert_parent_alive()
                    if log.tell() > 2**20:
                        raise ValueError('Algorithm child output exceeded bound')
                    time.sleep(min(.05, lease.check().remaining_seconds))
                if process.returncode != 0 or log.tell() > 2**20:
                    raise ValueError('Algorithm child failed or exceeded output bound')
            lease.assert_parent_alive()
            _, record, _, _ = ticket.verify(('running',))
            last = _kernel_snapshot()
            model, k0, counts, result = _outputs(ticket, record, first, last)
            completed = authority.complete(record['reservation_id'], result={'ticket_digest': ticket.digest,
                'model_digest': model.digest, 'k0_digest': k0.digest}, actual_counts=counts)
            final = _kernel_snapshot()
            _validate_kernel(last, final)
            receipt = {'version': VERSION, 'ticket_digest': ticket.digest,
                'reservation_hash': fingerprint(completed), 'model_hash': file_hash(directory/'model.json'),
                'k0_hash': file_hash(directory/'k0.json'), 'result_hash': fingerprint(result),
                'kernel_before': first, 'kernel_after': final, 'observation': _observation(lease),
                'shadow_only': True, 'formal_claim': False, 'serving_eligible': False}
            _write(directory / 'receipt.json', receipt)
            proof = {'receipt_hash': file_hash(directory/'receipt.json'), 'model_digest': model.digest,
                'k0_digest': k0.digest, 'actual_counts_hash': fingerprint(counts),
                'kernel': _kernel_snapshot(), 'observation': _observation(lease)}
            _validate_kernel(final, proof['kernel'])
            _member(proof['kernel'], _identity())
            witness.complete(proof)
            return {'status': 'completed', 'ticket': str(ticket.path), 'ticket_digest': ticket.digest,
                'shadow_only': True, 'formal_claim': False, 'serving_eligible': False}
        except BaseException:
            # Pending witness, not this optional failure file, controls admission.
            if authority.get(record['reservation_id'])['status'] in {'reserved', 'running'}:
                authority.fail_consumed(record['reservation_id'], 'algorithm_shadow_failed')
            raise
        finally:
            if process is not None and process.poll() is None:
                terminate_process_tree(process)


def worker(path):
    ticket = _Ticket(path)
    lease = FrozenWorkLease.from_dict(ticket.value['lease'])
    directory = ticket.path.parent / 'launch'
    with lease.as_worker(), threadpool_limits(limits=1):
        while not (directory/'start.json').exists():
            lease.assert_parent_alive(as_worker=True)
            time.sleep(.01)
        identity = _identity()
        if _read(directory/'start.json') != {'ticket_digest': ticket.digest,
                'process': identity, 'lease_hash': lease.digest}:
            raise ValueError('Algorithm child start identity differs')
        authority, record, data, recipe = ticket.verify(('running',))
        budget = ReservedChannelBudget.from_reservation(authority, record['reservation_id'])
        first = _kernel_snapshot()
        parent = {'pid': lease.parent_pid, 'created': lease.parent_created}
        _member(first, parent, identity)
        _write(directory/'membership.json', {'ticket_digest': ticket.digest,
            'process': identity, 'parent': parent, 'kernel': first})
        model = _fit_neural_forward_payload(data, ticket.value['plan']['cutoff'], recipe, budget)
        k0 = fit_pool(data, ticket.value['plan']['cutoff'], FrozenPool(), budget)
        model.save(directory/'model.json')
        immutable_record(directory/'k0.json', k0.to_dict())
        if NeuralForwardModel.load(directory/'model.json').digest != model.digest:
            raise ValueError('Algorithm model save/load differs')
        if _joint(json.loads((directory/'k0.json').read_text())).digest != k0.digest:
            raise ValueError('Algorithm K0 save/load differs')
        budget.finalize()
        last = _kernel_snapshot()
        _validate_kernel(first, last)
        _member(last, parent, identity)
        _write(directory/'result.json', {'version': VERSION, 'ticket_digest': ticket.digest,
            'process': identity, 'lease_hash': lease.digest, 'model_digest': model.digest,
            'k0_digest': k0.digest, 'actual_counts': budget.actual_counts, 'kernel': last})
        budget.check_time()


def admit(protocol, pipeline_id, cutoff, *, stage='V'):
    """Historical shadow-only admission; no new lease, registry or factor slot."""
    prepared = protocol.read_prepared(pipeline_id, cutoff, stage=stage, statuses=('completed',))
    ticket = _Ticket(_fit_path(protocol, prepared['plan']))
    return _admit_ticket(ticket)


def _admit_ticket(ticket):
    _, record, _, _ = ticket.verify(('completed',))
    directory = ticket.path.parent/'launch'
    proof = _Witness(ticket).read()
    receipt = _read(directory/'receipt.json')
    lease = FrozenWorkLease.from_dict(record['lease'])
    model, k0, counts, result = _outputs(ticket, record, receipt['kernel_before'], proof['kernel'])
    _validate_kernel(receipt['kernel_after'], proof['kernel'])
    _verify_observation(receipt['observation'], lease)
    _verify_observation(proof['observation'], lease, previous=receipt['observation'])
    if (receipt['version'] != VERSION or receipt['ticket_digest'] != ticket.digest
            or receipt['reservation_hash'] != fingerprint(record)
            or receipt['result_hash'] != fingerprint(result)
            or receipt['model_hash'] != file_hash(directory/'model.json')
            or receipt['k0_hash'] != file_hash(directory/'k0.json')
            or receipt['shadow_only'] is not True or receipt['formal_claim'] is not False
            or receipt['serving_eligible'] is not False
            or proof != {'receipt_hash': file_hash(directory/'receipt.json'), 'model_digest': model.digest,
                'k0_digest': k0.digest, 'actual_counts_hash': fingerprint(counts),
                'kernel': proof['kernel'], 'observation': proof['observation']}):
        raise ValueError('Algorithm shadow admission proof does not match completed work')
    return {'model': model, 'K0': k0, 'Pv': model.baseline, 'ticket_digest': ticket.digest,
        'completed_at': proof['observation']['wall'], 'shadow_only': True,
        'formal_claim': False, 'serving_eligible': False}


def _model_visible(protocol, bundle, pipeline, cutoff, stage, prefix, origin):
    if prefix is None:
        return stamp(bundle['completed_at']) <= origin
    from .nonlinear_runtime import physical_bound_for_prefix
    physical = physical_bound_for_prefix(protocol.runtime.lifecycle, prefix)
    path = _fit_path(protocol, {'validation_id': protocol.digest, 'pipeline_id': pipeline,
                              'cutoff': cutoff, 'stage': stage})
    files = (path, path.parent/'publication'/'witness.sqlite',
             *(path.parent/'launch'/name for name in ('model.json', 'k0.json', 'receipt.json', 'result.json')))
    return (stamp(bundle['completed_at']) <= physical and all(
        max(p.stat().st_mtime_ns, p.stat().st_ctime_ns) <= physical.value for p in files))


def _events(protocol, *, verify_models=True):
    """Bounded hash chain plus immutable external anchors; no outcome callbacks."""
    anchors = protocol.authority.directory/'algorithm_validation_events'/protocol.digest
    with closing(sqlite3.connect((protocol.directory/'validation.sqlite').as_uri()+'?mode=ro', uri=True)) as db:
        rows = db.execute('SELECT seq,body,hash FROM events ORDER BY seq').fetchall()
    if len(rows) > MAX_EVENTS:
        raise ValueError('Algorithm event limit exceeded')
    if set(anchors.glob('*.json')) != {anchors/f'{i}.json' for i in range(1, len(rows)+1)}:
        raise ValueError('Algorithm event anchors lost or uncommitted; cannot reset/relook')
    values, previous, admitted = [], protocol.digest, {}
    from .quarantined_observation import current_binding, QuarantinedObservation
    current_prefix = current_binding(protocol.runtime.lifecycle)
    observer = QuarantinedObservation.from_lifecycle(protocol.runtime.lifecycle) if current_prefix is not None else None
    for i, (seq, text, digest) in enumerate(rows, 1):
        value = json.loads(text)
        if (seq != i or len(text) > 16384 or value['previous'] != previous
                or value['seq'] != i or value['validation_id'] != protocol.digest
                or fingerprint(value) != digest or _read(anchors/f'{i}.json') != value):
            raise ValueError('Algorithm event sequence/source changed')
        prefix = value.get('observation_prefix')
        if (prefix is None) != (current_prefix is None):
            raise ValueError('Typed event lost or invented original observation provenance')
        if prefix is not None:
            from .nonlinear_runtime import physical_bound_for_prefix
            physical_bound_for_prefix(protocol.runtime.lifecycle, prefix)
            if value['recorded_at'] != prefix['logical_at']:
                raise ValueError('Algorithm event is not its original logical observation')
            if prefix['hour'] < current_prefix['hour']:
                next_prefix = observer.binding(prefix['hour']+1, protocol.runtime.lifecycle)
                commit = (anchors/f'{i}.json').stat()
                if max(commit.st_mtime_ns, commit.st_ctime_ns) > stamp(next_prefix['physical_observation']['wall']).value:
                    raise ValueError('Typed event was written after its original observation advanced')
            if value['kind'] == 'forecast' and verify_models:
                payload = value['payload']
                cutoff = protocol.runtime.lifecycle.fit_cycle_start(payload['origin']).isoformat()
                for pid, digest in payload['tickets'].items():
                    key = (pid, cutoff, payload['stage'])
                    if key not in admitted:
                        path = _fit_path(protocol, {'validation_id': protocol.digest, 'pipeline_id': pid,
                            'cutoff': cutoff, 'stage': payload['stage']})
                        admitted[key] = _admit_ticket(_Ticket(path))
                    bundle = admitted[key]
                    if (bundle['ticket_digest'] != digest or not _model_visible(protocol, bundle,
                            pid, cutoff, payload['stage'], prefix, stamp(payload['origin']))):
                        raise ValueError('Typed forecast used a model unpublished at its original prefix')
        values.append(value)
        previous = digest
    return values


def _append(protocol, kind, payload, *, guard=None):
    old = _events(protocol)
    if len(old) >= MAX_EVENTS:
        raise ValueError('Algorithm event budget consumed')
    value = {'seq': len(old)+1, 'validation_id': protocol.digest, 'kind': kind,
        'payload': payload, 'recorded_at': protocol_module.observation_now(protocol.runtime).isoformat(),
        'previous': fingerprint(old[-1]) if old else protocol.digest}
    from .quarantined_observation import current_binding
    prefix = current_binding(protocol.runtime.lifecycle)
    if prefix is not None:
        value['observation_prefix'] = prefix
    text = json.dumps(value, sort_keys=True, allow_nan=False)
    if len(text) > 16384:
        raise ValueError('Algorithm event exceeds fixed bound')
    with closing(sqlite3.connect((protocol.directory/'validation.sqlite').as_uri()+'?mode=rw', uri=True)) as db, db:
        db.execute('PRAGMA synchronous=FULL')
        db.execute('BEGIN IMMEDIATE')
        if db.execute('SELECT COUNT(*) FROM events').fetchone()[0] != len(old):
            raise ValueError('Concurrent algorithm event append')
        # Intent first. Crash before SQL commit closes this protocol rather than
        # creating a second forecast or a second endpoint look.
        _write(protocol.authority.directory/'algorithm_validation_events'/protocol.digest/f'{value["seq"]}.json', value)
        db.execute('INSERT INTO events VALUES(?,?,?)', (value['seq'], text, fingerprint(value)))
        if guard:
            guard()  # Internal only; no public callback/evidence admission API.
        if prefix != current_binding(protocol.runtime.lifecycle):
            raise ValueError('Typed event observation advanced before durable commit')
    return value


def endpoint_result(protocol, stage):
    protocol.verify()
    found = [v for v in _events(protocol) if v['kind'] == 'endpoint_result' and v['payload']['stage'] == stage]
    if len(found) > 1:
        raise ValueError('Algorithm endpoint was replayed')
    return {**found[0]['payload'], 'recorded_at': found[0]['recorded_at']} if found else None


def _stage(protocol, stage):
    identity = protocol.identity
    if stage == 'V':
        return stamp(identity['start']), stamp(identity['V_end']), list(identity['recipes'])
    if stage != 'qualifier':
        raise ValueError('Unknown algorithm stage')
    result = endpoint_result(protocol, 'V')
    if not result or not result['score']['selected'] or stamp(result['recorded_at']) >= stamp(identity['qualifier_start']):
        raise ValueError('Qualifier requires the timely single V winner')
    return stamp(identity['qualifier_start']), stamp(identity['qualifier_end']), [result['score']['selected']]


def tick(protocol):
    """Publish only current actual origins; settle immutable forecasts once."""
    protocol.verify()
    with FileLock(str(protocol.directory/'events.lock'), timeout=0):
        events, now = _events(protocol), protocol_module.observation_now(protocol.runtime)
        from .quarantined_observation import current_binding, snapshot_for_prefix
        prefix = current_binding(protocol.runtime.lifecycle)
        bars = (protocol.runtime._snapshot(now) if prefix is None else snapshot_for_prefix(
            protocol.runtime.lifecycle, prefix, now, context_hours=protocol.runtime.config.context_hours))
        basis = 'actual_receipt' if prefix is None else 'quarantined_finite_observation'
        forecasts = {v['payload']['key']: v['payload'] for v in events if v['kind'] == 'forecast'}
        settled = {v['payload']['forecast_key'] for v in events if v['kind'] == 'settlement'}
        closed = {v['payload']['stage'] for v in events if v['kind'] == 'endpoint_read'}
        settlements = 0
        for key, f in forecasts.items():
            if key in settled or f['stage'] in closed:
                continue
            start = stamp(f['window_start'])
            end = start+pd.Timedelta(hours=12)
            subset = bars[(bars.symbol == f['symbol']) & (bars.open_time >= start-pd.Timedelta(hours=1)) & (bars.open_time < end)].sort_values('open_time')
            if (len(subset) != 13 or list(subset.open_time) != list(pd.date_range(start-pd.Timedelta(hours=1), periods=13, freq='h'))
                    or not (subset.availability_basis == basis).all()):
                continue
            mature = max(end+pd.Timedelta(seconds=protocol.runtime.spec.availability_delay_seconds), subset.available_at.max())
            cutoff = stamp(protocol.identity[f['stage']+'_scoring_cutoff'])
            if mature > now or now > cutoff:
                continue
            target = float(np.sqrt(np.sum(np.diff(np.log(subset.close.to_numpy(float)))**2)))
            label_id = fingerprint({'source': protocol.identity['source_id'], 'start': start.isoformat(),
                'symbol': f['symbol'], 'revisions': subset.revision.tolist(), 'target': target})
            def settlement_guard(cutoff=cutoff, mature=mature):
                current = protocol_module.observation_now(protocol.runtime)
                if current < max(now, mature) or current > cutoff:
                    raise ValueError('Algorithm settlement missed its frozen scoring cutoff')
            _append(protocol, 'settlement', {'forecast_key': key, 'target': target,
                'label_identity': label_id, 'matured_at': mature.isoformat(), 'stage': f['stage']},
                guard=settlement_guard)
            settlements += 1
        stage = ('V' if stamp(protocol.identity['start']) <= now < stamp(protocol.identity['V_end']) else
                 'qualifier' if stamp(protocol.identity['qualifier_start']) <= now < stamp(protocol.identity['qualifier_end']) else None)
        if not stage or stage in closed:
            return {'predictions': 0, 'settlements': settlements}
        _, _, candidates = _stage(protocol, stage)
        cutoff = protocol.runtime.lifecycle.fit_cycle_start(now)
        bundles = {}
        for candidate in candidates:
            path = _fit_path(protocol, protocol.fit_plan(candidate, cutoff, stage=stage))
            if not path.exists():
                return {'predictions': 0, 'settlements': settlements, 'status': 'waiting_all_models'}
            state = protocol.authority.get(_read(path)['reservation_id'])['status']
            if state != 'completed':
                return {'predictions': 0, 'settlements': settlements, 'status': 'model_work_'+state}
            bundles[candidate] = admit(protocol, candidate, cutoff, stage=stage)
        history = build_feature_history(bars)
        predictions = {}
        with threadpool_limits(limits=1):
            for candidate, bundle in bundles.items():
                predictions[candidate] = bundle['model'].predict_history(history)
            first = next(iter(bundles.values()))
            for ref in ('K0', 'Pv'):
                if any(b[ref].digest != first[ref].digest for b in bundles.values()):
                    raise ValueError('Candidates did not fit identical same-cutoff references')
                predictions[ref] = first[ref].predict_history(history)
        published = 0
        for symbol, group in bars.groupby('symbol', sort=True):
            latest = group.sort_values('open_time').iloc[-1]
            origin = stamp(latest.available_at)
            if (latest.availability_basis != basis or not latest.prediction_origin
                    or latest.open_time != origin.floor('h')-pd.Timedelta(hours=1)
                    or not 0 <= (now-origin).total_seconds() <= protocol.runtime.lifecycle.plan.prediction_max_lag_seconds
                    or origin.ceil('h') <= now
                    or any(not _model_visible(protocol, b, pid, cutoff.isoformat(), stage, prefix, origin)
                           for pid, b in bundles.items())):
                continue
            key = fingerprint({'stage': stage, 'hour': origin.floor('h').isoformat(), 'symbol': symbol})
            if key in forecasts:
                continue
            mask = (history.metadata.symbol == symbol) & (history.metadata.feature_open_time == latest.open_time)
            indexes = np.flatnonzero(mask)
            if len(indexes) != 1:
                continue
            index = indexes[0]
            if any(not support[index] or not math.isfinite(values[index]) or values[index] < 0 for values, support in predictions.values()):
                continue
            def guard(origin=origin):
                current = protocol_module.observation_now(protocol.runtime)
                if (current < now or current >= origin.ceil('h')
                        or (current-origin).total_seconds() > protocol.runtime.lifecycle.plan.prediction_max_lag_seconds):
                    raise ValueError('Algorithm forecast completed too late; no backdating')
            _append(protocol, 'forecast', {'key': key, 'stage': stage, 'symbol': symbol,
                'hour': origin.floor('h').isoformat(), 'origin': origin.isoformat(),
                'window_start': origin.ceil('h').isoformat(),
                'predictions': {name: float(values[index]) for name, (values, _) in predictions.items()},
                'tickets': {name: b['ticket_digest'] for name, b in bundles.items()},
                'revision': latest.revision}, guard=guard)
            published += 1
        return {'predictions': published, 'settlements': settlements}


def _endpoint_score(protocol, stage, events):
    start, end, candidates = _stage(protocol, stage)
    rows = {name: [] for name in [*candidates, 'K0', 'Pv']}
    forecasts = {v['payload']['key']: v for v in events if v['kind'] == 'forecast' and v['payload']['stage'] == stage}
    cutoff = stamp(protocol.identity[stage+'_scoring_cutoff'])
    seen = set()
    for event in events:
        if event['kind'] != 'settlement' or event['payload']['stage'] != stage:
            continue
        s = event['payload']
        if s['forecast_key'] in seen or s['forecast_key'] not in forecasts:
            raise ValueError('Algorithm settlement is duplicate or orphan')
        seen.add(s['forecast_key'])
        f = forecasts[s['forecast_key']]['payload']
        if stamp(event['recorded_at']) > cutoff or stamp(s['matured_at']) > cutoff:
            continue
        if set(f['predictions']) != set(rows):
            raise ValueError('Endpoint omits a frozen model or reference')
        for name, values in rows.items():
            loss = (math.log(f['predictions'][name]+1e-8)-math.log(s['target']+1e-8))**2
            values.append({'hour': f['hour'], 'symbol': f['symbol'], 'loss': loss,
                'label_identity': s['label_identity']})
    rule = protocol_module.AlgorithmValidationRule(**{k: protocol.identity['rule'][k]
        for k in ('minimum_coverage', 'minimum_gain')})
    return protocol_module._paired_selection(rows, candidates, start=start, end=end,
        symbols=protocol.identity['symbols'], rule=rule)


def conclude(protocol, stage):
    protocol.verify()
    _stage(protocol, stage)
    with FileLock(str(protocol.directory/'events.lock'), timeout=0):
        events = _events(protocol)
        old = [v for v in events if v['kind'] == 'endpoint_result' and v['payload']['stage'] == stage]
        if old:
            if len(old) != 1:
                raise ValueError('Duplicate algorithm endpoint')
            return {**old[0]['payload'], 'recorded_at': old[0]['recorded_at']}
        if any(v['kind'] == 'endpoint_read' and v['payload']['stage'] == stage for v in events):
            raise ValueError('Algorithm endpoint read consumed without completed score; cannot relook')
        if protocol_module.observation_now(protocol.runtime) < stamp(protocol.identity[stage+'_scoring_cutoff']):
            raise ValueError('Algorithm endpoint not matured')
        _append(protocol, 'endpoint_read', {'stage': stage, 'evidence_hash': fingerprint(events)})
        score = _endpoint_score(protocol, stage, events)
        def result_guard():
            # A positive winner must actually be committed before qualifier
            # starts. Slow disk cannot turn a late selection into a prior one.
            if (stage == 'V' and score['selected'] is not None
                    and protocol_module.observation_now(protocol.runtime) >= stamp(protocol.identity['qualifier_start'])):
                raise ValueError('Algorithm winner was not published before qualifier start')
        result = _append(protocol, 'endpoint_result', {'stage': stage, 'score': score,
            'evidence_hash': fingerprint(events), 'formal_claim': False, 'serving_eligible': False},
            guard=result_guard)
        return {**result['payload'], 'recorded_at': result['recorded_at']}


def controller_step(protocol, *, deadline_monotonic, max_fits=1):
    """Bounded autonomous shadow dispatch; artifacts remain the gate authority.

    Missing real gate references yield waiting status. They cannot be replaced
    by a flag, and a consumed stable work never retries. Original issuer/300s
    checks remain in prepare/launch. No inference is replayed for a missed month.
    """
    if (type(max_fits) is not int or not 0 <= max_fits <= 3
            or type(deadline_monotonic) not in (int, float) or not math.isfinite(deadline_monotonic)):
        raise ValueError('Invalid bounded algorithm controller dispatch')
    protocol.verify()
    actions = []
    now = protocol_module.observation_now(protocol.runtime)
    with FileLock(str(protocol.directory/'controller.lock'), timeout=0):
        if stamp(protocol.identity['start']) <= now <= stamp(protocol.identity['qualifier_scoring_cutoff']):
            actions.append({'kind': 'algorithm_shadow_tick', **tick(protocol)})
        for stage in ('V', 'qualifier'):
            try:
                start, end, candidates = _stage(protocol, stage)
            except ValueError:
                if stage == 'qualifier' and endpoint_result(protocol, 'V') is not None:
                    actions.append({'kind': 'algorithm_qualifier', 'status': 'no_timely_winner'})
                    continue
                if stage == 'qualifier':
                    continue
                raise
            if now >= stamp(protocol.identity[stage+'_scoring_cutoff']):
                result = conclude(protocol, stage)
                actions.append({'kind': 'algorithm_endpoint', 'stage': stage, 'result': result})
                continue
            if not start <= now < end:
                continue
            cutoff = protocol.runtime.lifecycle.fit_cycle_start(now)
            for pipeline in candidates:
                if max_fits <= 0 or time.monotonic()+302 > deadline_monotonic:
                    actions.append({'kind': 'algorithm_fit', 'status': 'insufficient_step_headroom'})
                    break
                plan = protocol.fit_plan(pipeline, cutoff, stage=stage)
                key = _key(plan)
                with protocol.authority._transaction() as db:
                    old = [r for r in protocol.authority._verify(db)['reservations'].values() if r['work_key'] == key]
                if old:
                    if len(old) != 1:
                        raise ValueError('Algorithm work key has multiple reservations')
                    actions.append({'kind': 'algorithm_fit', 'pipeline_id': pipeline, 'status': old[0]['status'],
                        'reservation_id': old[0]['reservation_id'], 'replayed': False})
                    continue
                gates = [e['payload'] for e in _events(protocol) if e['kind'] == 'fit_gates' and e['payload']['work_key'] == key]
                if not gates:
                    actions.append({'kind': 'algorithm_fit', 'pipeline_id': pipeline, 'status': 'waiting_real_gate_artifacts'})
                    continue
                if len(gates) != 1 or gates[0]['plan_hash'] != fingerprint(plan):
                    raise ValueError('Algorithm gate registration changed')
                # Counts/seconds prepay occurs inside prepare before source IO.
                prepared = protocol.prepare_fit(pipeline, cutoff, gates=gates[0]['gates'], stage=stage)
                max_fits -= 1
                actions.append({'kind': 'algorithm_fit', **launch(protocol, prepared)})
    return {'actions': actions, 'formal_claim': False, 'serving_eligible': False, 'automatic_adoption': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fixed typed algorithm shadow worker')
    parser.add_argument('--ticket', required=True)
    worker(parser.parse_args().ticket)
