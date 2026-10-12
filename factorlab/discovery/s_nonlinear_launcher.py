"""Fixed isolated quarterly tree/positive-control or S resource trial producer.

The whole Linux cgroup is measured, including parent revalidation. Each work
shares its original prepaid 300-second lease. Receipts are S evidence
only; no forward admission certificate is produced or accepted.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from datetime import datetime
from pathlib import Path

import numpy as np
from filelock import FileLock
from threadpoolctl import threadpool_limits

from factorlab.protocol import fingerprint, project_path

from .convex_eval import log_mse
from .evidence import file_hash
from .joint_pool import hour_equal_weights
from .neural_forward import NeuralForwardModel, _fit_neural_research_payload
from .neural_forward_admission import _kernel as _strict_kernel
from .neural_forward_launcher import _environment, _identity, _member, _observation
from .neural_research_ticket import _canonical
from .nonlinear_channel import run_probe
from .nonlinear_launcher import _kernel_snapshot, _read, _same_boundary, _validate_kernel, _write
from .numeric_replay import RULE as NUMERIC_REPLAY_RULE
from .numeric_replay import assert_replayed
from .process_policy import isolated_process_kwargs, terminate_process_tree
from .reserved_work_budget import ReservedChannelBudget
from .rolling import development_folds
from .s_nonlinear_ticket import SNonlinearTicket
from .v7_controls import known_component_recovery_probe
from .v7_storage import immutable_record
from .work_lease import FrozenWorkLease

VERSION = 'v7-authority-S-nonlinear-launch-v1'
LOG_LIMIT = 2**20


def _paths(ticket):
    root = ticket.path.parent / 'launch'
    return {name: root / (name + '.json') for name in ('start', 'membership', 'worker_result', 'receipt', 'failure')}


class _Publication:
    """Pending survives failure-log errors; COMMIT is the logical publication point."""
    def __init__(self, ticket, *, prepare=False):
        if type(ticket) is not SNonlinearTicket:
            raise TypeError('S publication requires its actual typed ticket')
        self.ticket = ticket
        self.directory = ticket.path.parent / 'launch/publication'
        self.path = self.directory / 'publication.sqlite'
        self.anchor = ticket.path.parent / 'publication_anchor.json'
        self.identity = {'version': VERSION, 'ticket_digest': ticket.digest,
                         'reservation_id': ticket.value['reservation_id'],
                         'authority_manifest_hash': ticket.value['authority_manifest_hash'],
                         'source_hash': ticket.value['source_hash'], 'lease_hash': ticket.value['lease_hash'],
                         'directory': str(self.directory)}
        if prepare:
            if self.anchor.exists() or self.directory.exists():
                raise ValueError('S publication already prepared; cannot reset/retry')
            immutable_record(self.anchor, self.identity)
            self.directory.mkdir(parents=True)
            immutable_record(self.directory / 'identity.json', self.identity)
            with closing(sqlite3.connect(self.path)) as db, db:
                db.execute('PRAGMA journal_mode=DELETE')
                db.execute('PRAGMA synchronous=FULL')
                db.execute('CREATE TABLE publication(id INTEGER PRIMARY KEY CHECK(id=1), identity TEXT, status TEXT, proof TEXT, hash TEXT)')
                db.execute('INSERT INTO publication VALUES(1,?,\'pending\',NULL,NULL)', (fingerprint(self.identity),))
            FrozenWorkLease.from_dict(ticket.value['lease']).assert_parent_alive()
        self._verify()

    def _verify(self):
        if (not self.path.is_file() or json.loads(self.anchor.read_text()) != self.identity
                or json.loads((self.directory / 'identity.json').read_text()) != self.identity):
            raise ValueError('S publication identity/database missing or changed')
        with closing(sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True)) as db:
            rows = db.execute('SELECT * FROM publication').fetchall()
        if len(rows) != 1 or rows[0][0:2] != (1, fingerprint(self.identity)):
            raise ValueError('S publication singleton changed')
        row = rows[0]
        if row[2] == 'pending':
            if row[3:] != (None, None):
                raise ValueError('Pending publication has an uncommitted proof')
        elif row[2] == 'completed':
            if fingerprint(json.loads(row[3])) != row[4]:
                raise ValueError('S publication proof changed')
        else:
            raise ValueError('Unknown S publication status')
        return row

    def complete(self, proof):
        self._verify()
        if set(proof) != {'receipt_hash', 'result_hash', 'actual_counts_hash', 'kernel', 'observation'}:
            raise ValueError('Unknown S publication proof')
        with closing(sqlite3.connect(self.path.as_uri() + '?mode=rw', uri=True)) as db, db:
            db.execute('PRAGMA synchronous=FULL')
            db.execute('BEGIN IMMEDIATE')
            changed = db.execute("UPDATE publication SET status='completed',proof=?,hash=? WHERE id=1 AND identity=? AND status='pending'",
                (json.dumps(proof, sort_keys=True, allow_nan=False), fingerprint(proof), fingerprint(self.identity)))
            if changed.rowcount != 1:
                raise ValueError('S publication cannot replay a completed work')
            # Any failure here rolls back to pending even if failure.json cannot
            # be written. No application-side check is added after COMMIT.
            FrozenWorkLease.from_dict(self.ticket.value['lease']).assert_parent_alive()

    def read_completed(self):
        row = self._verify()
        if row[2] != 'completed':
            raise ValueError('S result has no committed publication witness')
        return json.loads(row[3])


def _research_result(ticket, data, recipe):
    """Replay scores/weights without granting a serving or neural-effect gate."""
    directory = ticket.path.parent / 'launch/research'
    path = directory / 'result.json'
    if not path.is_file() or path.stat().st_size > 16 * 2**20:
        raise ValueError('Missing or oversized fixed producer result')
    result = json.loads(path.read_text('utf-8'))
    if (result.get('version') != VERSION or result.get('role') != 'S'
            or result.get('kind') != ticket.value['kind'] or result.get('status') != 'completed'
            or result.get('ticket_digest') != ticket.digest
            or result.get('recipe') != _canonical(recipe.to_dict())
            or result.get('input_id') != ticket.value['context']['input_id']
            or result.get('test_only') is not False or result.get('serving_eligible') is not False):
        raise ValueError('Fixed nonlinear output does not match its original S ticket')
    cutoff = ticket.value['context']['cutoff']
    if ticket.value['kind'] == 'resource_trial':
        model_path = directory / 'model.json'
        model = NeuralForwardModel.load(model_path)
        if (result.get('model') != {'path': str(model_path), 'sha256': file_hash(model_path), 'digest': model.digest}
                or model.recipe.pipeline_id != recipe.pipeline_id
                or model.receipt.get('payload_origin') != 'isolated_research_worker'
                or any(model.receipt['training'].get(k) != v
                       for k, v in data.training_identity(cutoff).items())):
            raise ValueError('Resource model differs from the cold complete S recipe')
        pred, mask = model.predict_history(data.frame)
        replay = {'sample_ids': data.frame.metadata.sample_id[mask].tolist(),
                  'predictions': pred[mask].tolist()}
        assert_replayed(result['prediction_replay'], replay, roundoff=True)
    else:
        probe, positive = result['probe'], result['positive_control']
        expected = list(development_folds(cutoff))
        if probe.get('status') != 'completed' or len(probe['folds']) != 3 or len(positive['folds']) != 3:
            raise ValueError('Quarterly producer must complete all fixed cold folds and controls')
        if (positive.get('scope') != 'fixed_S_known_component_recovery_diagnostic'
                or positive.get('component') != 'rv_24' or positive.get('solves') != {'removed': 3, 'restored': 3}
                or positive.get('known_population_effect') is not False
                or positive.get('opens_NN_or_confirmation_gate') is not False):
            raise ValueError('Positive control changed its fixed work or claimed population evidence')
        metadata = data.frame.metadata
        for fold, control, contract in zip(probe['folds'], positive['folds'], expected, strict=True):
            if any(fold.get(k) != v or control.get(k) != v for k, v in contract.items()):
                raise ValueError('Quarterly cold-fold chronology changed')
            if fold['fit'] != data.training_identity(contract['fit_cutoff']) or fold['pool_id'] != recipe.pool.digest:
                raise ValueError('Quarterly fit borrowed another pool or training window')
            ids = fold['sample_ids']
            mask = metadata.sample_id.isin(ids).to_numpy()
            legal = data.score_mask(contract['fit_cutoff'], contract['score_end'], contract['score_end'])
            if (len(set(ids)) != len(ids) or metadata.sample_id[mask].tolist() != ids
                    or not np.all(legal[mask]) or len(ids) != fold['support'] or not ids):
                raise ValueError('Quarterly score support escaped its causal fold')
            pred = np.asarray(fold['predictions'], dtype=float)
            if pred.shape != (len(ids),) or not np.isfinite(pred).all() or (pred < 0).any():
                raise ValueError('Quarterly predictions are invalid')
            loss = log_mse(data.target[mask], pred, hour_equal_weights(metadata.loc[mask]))
            assert_replayed(fold['loss'], loss, roundoff=True)
            base = fold['reference_loss']
            if type(base) not in (int, float) or not np.isfinite(base) or base < 0:
                raise ValueError('Invalid same-support reference loss')
            assert_replayed(fold['gain'], 1 - loss / base if base else None, roundoff=True)
            common = data.frame.valid_rows
            control_train = data.train_mask(contract['fit_cutoff']) & common
            control_score = legal & common
            if (control['train_rows'] != int(control_train.sum()) or control['score_rows'] != int(control_score.sum())
                    or control['support_hash'] != fingerprint(np.flatnonzero(control_score).tolist())
                    or control['solves_per_arm'] != 1
                    or any(not np.isfinite(v) or v < 0 for v in control['losses'].values())
                    or set(control['losses']) != {'removed', 'restored'}):
                raise ValueError('Fixed positive-control common support changed')
        removed = float(np.mean([f['losses']['removed'] for f in positive['folds']]))
        restored = float(np.mean([f['losses']['restored'] for f in positive['folds']]))
        assert_replayed(positive['relative_gain'], 1 - restored / removed if removed else None, roundoff=True)
    return result, {'path': str(path), 'sha256': file_hash(path), 'digest': fingerprint(result)}


def _produce_fixed(ticket, data, recipe, budget, directory):
    # Only the actual registered child calls this branch. No caller callback or
    # executable expression selects the numerical producer.
    if type(budget) is not ReservedChannelBudget or budget.test_only or budget.executor_mode != 'child':
        raise TypeError('Fixed producer requires the canonical registered child budget')
    budget.check_time()
    cutoff = ticket.value['context']['cutoff']
    result = {'version': VERSION, 'role': 'S', 'kind': ticket.value['kind'], 'status': 'completed',
              'ticket_digest': ticket.digest, 'recipe': _canonical(recipe.to_dict()),
              'input_id': ticket.value['context']['input_id'], 'test_only': False, 'serving_eligible': False}
    if ticket.value['kind'] == 'quarterly_probe':
        result['probe'] = run_probe(data, cutoff, pool=recipe.pool, budget=budget)
        budget.consume('solve', 6)
        result['positive_control'] = known_component_recovery_probe(data, cutoff)
    elif ticket.value['kind'] == 'resource_trial':
        model = _fit_neural_research_payload(data, cutoff, recipe, budget)
        path = directory / 'model.json'
        model.save(path)
        predictions, mask = model.predict_history(data.frame)
        result['model'] = {'path': str(path), 'sha256': file_hash(path), 'digest': model.digest}
        result['prediction_replay'] = {'sample_ids': data.frame.metadata.sample_id[mask].tolist(),
                                       'predictions': predictions[mask].tolist()}
    else:
        raise ValueError('Unknown fixed S nonlinear producer')
    budget.check_time()
    result['budget_counts'] = budget.counts
    immutable_record(directory / 'result.json', result)
    return result


def _verify_outputs(ticket, record, data, recipe, first, last):
    paths = _paths(ticket)
    membership, worker_result = _read(paths['membership']), _read(paths['worker_result'])
    lease = FrozenWorkLease.from_dict(record['lease'])
    process = record['process_identity']
    parent = {'pid': lease.parent_pid, 'created': lease.parent_created}
    if (membership.get('version') != VERSION or membership.get('ticket_digest') != ticket.digest
            or membership.get('process') != process or membership.get('parent') != parent
            or worker_result.get('version') != VERSION or worker_result.get('ticket_digest') != ticket.digest
            or worker_result.get('process') != process or worker_result.get('lease_hash') != lease.digest):
        raise ValueError('S worker/membership differs from canonical child')
    for sample in (membership['kernel'], worker_result['kernel']):
        if not _same_boundary(first, sample):
            raise ValueError('S worker escaped its shared resource boundary')
        _member(sample, parent, process)
        _validate_kernel(first, sample)
        _validate_kernel(sample, last)
        _strict_kernel(first, sample, last, lease, parent, process)
    _member(first, parent)
    _member(last, parent)
    _validate_kernel(first, last)
    counts = ReservedChannelBudget.verify_counts(open_authority(ticket), record['reservation_id'])
    if counts['test_only'] or not counts['finalized'] or counts['actual_counts'] != worker_result['actual_counts']:
        raise ValueError('S child actual accounting differs from durable prepaid evidence')
    result, reference = _research_result(ticket, data, recipe)
    if (worker_result['research'] != reference or result['budget_counts'] != counts['counts']):
        raise ValueError('Complete S models used different actual counters/results')
    return result, reference, counts['actual_counts']


def open_authority(ticket):
    from .neural_forward_ticket import open_existing_authority
    return open_existing_authority(ticket.value['authority_directory'])


def launch_nonlinear(ticket_path):
    ticket = SNonlinearTicket.load(project_path(ticket_path))
    paths = _paths(ticket)
    directory = paths['start'].parent
    directory.mkdir(parents=True, exist_ok=True)
    lease = FrozenWorkLease.from_dict(ticket.value['lease'])
    process, authority = None, None
    with FileLock(str(directory / 'launch.lock'), timeout=0):
        if any(p.exists() for p in paths.values()) or (ticket.path.parent / 'publication_anchor.json').exists():
            raise ValueError('S launch already attempted; no retry or refund')
        try:
            authority, record, data, recipe = ticket.verify(mode='launch')
            publication = _Publication(ticket, prepare=True)
            first = _kernel_snapshot()
            _validate_kernel(first, first)
            _member(first, _identity())
            if ticket.value['kind'] == 'resource_trial' and importlib.util.find_spec('torch') is None:
                raise ValueError('Torch unavailable; complete S remains closed')
            command = [sys.executable, '-B', '-m', 'factorlab.discovery.s_nonlinear_launcher', '--ticket', str(ticket.path)]
            with (directory / 'worker.log').open('xb') as log:
                process = subprocess.Popen(command, cwd=str(Path(__file__).resolve().parents[2]),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    env=_environment(), **isolated_process_kwargs())
                identity = _identity(process.pid)
                running = authority.start(record['reservation_id'], identity)
                if running['status'] != 'running':
                    raise ValueError('Original S reservation did not start')
                _write(paths['start'], {'ticket_digest': ticket.digest, 'process': identity, 'lease_hash': lease.digest})
                while process.poll() is None:
                    lease.assert_parent_alive()
                    if log.tell() > LOG_LIMIT:
                        raise ValueError('S worker output bound exceeded')
                    time.sleep(min(.05, lease.check().remaining_seconds))
                if process.returncode != 0:
                    raise RuntimeError(f'Complete S worker exited {process.returncode}')
                if log.tell() > LOG_LIMIT:
                    raise ValueError('S worker output bound exceeded')
            _, running, data, recipe = ticket.verify(mode='running_parent')
            last = _kernel_snapshot()
            _result, reference, counts = _verify_outputs(ticket, running, data, recipe, first, last)
            completed = authority.complete(record['reservation_id'],
                result={'ticket_digest': ticket.digest, 'research_digest': reference['digest']}, actual_counts=counts)
            if completed['status'] != 'completed':
                raise ValueError('S original lease closed before completion')
            final_kernel = _kernel_snapshot()
            _validate_kernel(last, final_kernel)
            _member(final_kernel, _identity())
            receipt = {'version': VERSION, 'role': 'S', 'status': 'completed', 'workload': ticket.value['kind'],
                'ticket': {'path': str(ticket.path), 'digest': ticket.digest}, 'research': reference,
                'reservation_id': record['reservation_id'], 'completed_record_hash': fingerprint(completed),
                'actual_counts': counts, 'process': identity, 'kernel_before': first, 'kernel_after': final_kernel,
                'context': ticket.value['context'], 'source_hash': ticket.value['source_hash'],
                'parent_evaluation': {'digest': fingerprint(_result), 'numeric_replay': NUMERIC_REPLAY_RULE},
                'serving_eligible': False, 'formal_claim': False, 'automatic_adoption': False,
                'observation': _observation(lease), 'authentication': 'local_trusted_producer_not_admin_authentication'}
            _write(paths['receipt'], receipt)
            kernel = _kernel_snapshot()
            _validate_kernel(final_kernel, kernel)
            _member(kernel, _identity())
            proof = {'receipt_hash': file_hash(paths['receipt']), 'result_hash': reference['sha256'],
                     'actual_counts_hash': fingerprint(counts), 'kernel': kernel, 'observation': _observation(lease)}
            response = {'status': 'completed', 'role': 'S', 'research': reference,
                        'receipt': {'path': str(paths['receipt']), 'sha256': proof['receipt_hash']},
                        'serving_eligible': False, 'formal_claim': False, 'automatic_adoption': False}
            publication.complete(proof)
            return response
        except BaseException as error:
            try:
                _write(paths['failure'], {'ticket_digest': ticket.digest,
                    'reason': type(error).__name__ + ':' + str(error)[:1500]})
            except BaseException as persistence_error:  # noqa: BLE001 - preserve original failure without refund
                error.add_note('S failure record unavailable: ' + type(persistence_error).__name__)
            try:
                if authority is not None and authority.get(ticket.value['reservation_id'])['status'] in {'reserved', 'running'}:
                    authority.fail_consumed(ticket.value['reservation_id'], type(error).__name__ + ':' + str(error)[:800])
            except BaseException as accounting_error:  # noqa: BLE001 - preserve original failure without refund
                error.add_note('S consumed accounting closure unavailable: ' + type(accounting_error).__name__)
            raise
        finally:
            if process is not None and process.poll() is None:
                terminate_process_tree(process)


def verify_completed_nonlinear(ticket_path):
    ticket = SNonlinearTicket.load(project_path(ticket_path))
    _, record, data, recipe = ticket.verify(mode='completed')
    paths = _paths(ticket)
    if paths['failure'].exists():
        raise ValueError('Failed S launch cannot become completed research')
    proof = _Publication(ticket).read_completed()
    receipt = _read(paths['receipt'])
    if (receipt['version'] != VERSION or receipt['role'] != 'S'
            or receipt['ticket'] != {'path': str(ticket.path), 'digest': ticket.digest}
            or receipt['completed_record_hash'] != fingerprint(record)
            or receipt['serving_eligible'] is not False or receipt['formal_claim'] is not False
            or receipt['automatic_adoption'] is not False or file_hash(paths['receipt']) != proof['receipt_hash']):
        raise ValueError('S completion receipt changed or borrowed a serving role')
    result, reference, counts = _verify_outputs(ticket, record, data, recipe, receipt['kernel_before'], receipt['kernel_after'])
    if (reference != receipt['research'] or proof['result_hash'] != reference['sha256']
            or fingerprint(counts) != proof['actual_counts_hash'] or receipt['actual_counts'] != counts
            or receipt.get('parent_evaluation') != {'digest': fingerprint(result), 'numeric_replay': NUMERIC_REPLAY_RULE}):
        raise ValueError('S publication does not bind the original actual work')
    _validate_kernel(receipt['kernel_after'], proof['kernel'])
    _member(proof['kernel'], {'pid': record['lease']['parent_pid'], 'created': record['lease']['parent_created']})
    lease = FrozenWorkLease.from_dict(record['lease'])
    parent = {'pid': lease.parent_pid, 'created': lease.parent_created}
    member = _read(paths['membership'])['kernel']
    _strict_kernel(receipt['kernel_before'], member, proof['kernel'], lease, parent, record['process_identity'])
    previous_wall, previous_mono = datetime.fromisoformat(lease.wall_started_at), lease.monotonic_started
    for observation in (receipt['observation'], proof['observation']):
        wall = datetime.fromisoformat(observation['wall'])
        mono = observation['monotonic']
        if (observation['parent'] != parent or observation['boot_id'] != lease.boot_id
                or not previous_wall <= wall < datetime.fromisoformat(lease.wall_deadline)
                or type(mono) not in (int, float) or not np.isfinite(mono)
                or not previous_mono <= mono < lease.monotonic_deadline):
            raise ValueError('S publication observer differs from its original lease/chronology')
        previous_wall, previous_mono = wall, mono
    return {'ticket': ticket, 'reservation': record, 'data': data, 'result': result, 'receipt': receipt,
            'publication': proof, 'role': 'S', 'serving_eligible': False, 'formal_claim': False}


def worker(ticket_path):
    ticket = SNonlinearTicket.load(project_path(ticket_path))
    lease = FrozenWorkLease.from_dict(ticket.value['lease'])
    paths = _paths(ticket)
    with lease.as_worker():
        while not paths['start'].exists():
            lease.assert_parent_alive(as_worker=True)
            time.sleep(.01)
        process = _identity()
        if _read(paths['start']) != {'ticket_digest': ticket.digest, 'process': process, 'lease_hash': lease.digest}:
            raise ValueError('S fixed-worker handshake differs from its original child identity')
        authority, _, data, recipe = ticket.verify(mode='worker')
        budget = ReservedChannelBudget.from_reservation(authority, ticket.value['reservation_id'])
        parent = {'pid': lease.parent_pid, 'created': lease.parent_created}
        kernel = _kernel_snapshot()
        _member(kernel, parent, process)
        _write(paths['membership'], {'version': VERSION, 'ticket_digest': ticket.digest,
            'process': process, 'parent': parent, 'kernel': kernel})
        # Enforce the same BLAS path as parent replay even when a trusted
        # supervisor imported numerical libraries before launching this child.
        # Environment variables alone do not reconfigure an already loaded BLAS.
        with threadpool_limits(limits=1):
            result = _produce_fixed(ticket, data, recipe, budget, paths['start'].parent / 'research')
        if result['status'] != 'completed':
            raise ValueError('At least one complete cold S fold failed; no completed workload receipt')
        budget.check_time()
        budget.finalize()
        last = _kernel_snapshot()
        _validate_kernel(kernel, last)
        _member(last, parent, process)
        path = paths['start'].parent / 'research/result.json'
        reference = {'path': str(path), 'sha256': file_hash(path), 'digest': fingerprint(result)}
        _write(paths['worker_result'], {'version': VERSION, 'ticket_digest': ticket.digest,
            'process': process, 'lease_hash': lease.digest, 'research': reference,
            'actual_counts': budget.actual_counts, 'kernel': last})
        budget.check_time()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Fixed shared-authority S nonlinear worker')
    parser.add_argument('--ticket', required=True)
    worker(parser.parse_args(argv).ticket)


if __name__ == '__main__':
    main()
