"""Fixed isolated complete S learner, with a durable publication linearization.

The whole Linux cgroup is measured, including parent revalidation. All three
cold folds share the original prepaid 300-second lease. Receipts are S evidence
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

from .evidence import file_hash
from .neural_forward import NeuralForwardModel, _joint
from .neural_forward_admission import _kernel as _strict_kernel
from .neural_forward_launcher import _environment, _identity, _member, _observation
from .neural_research import _run_research_worker, score_research_fold
from .neural_research_ticket import ResearchFitTicket, _canonical
from .nonlinear_launcher import _kernel_snapshot, _read, _same_boundary, _validate_kernel, _write
from .numeric_replay import RULE as NUMERIC_REPLAY_RULE
from .numeric_replay import assert_replayed
from .process_policy import isolated_process_kwargs, terminate_process_tree
from .reserved_work_budget import ReservedChannelBudget
from .rolling import development_folds
from .v7_storage import immutable_record
from .work_lease import FrozenWorkLease

VERSION = 'v7-complete-neural-S-launch-v2'
LOG_LIMIT = 2**20


def _paths(ticket):
    root = ticket.path.parent / 'launch'
    return {name: root / (name + '.json') for name in ('start', 'membership', 'worker_result', 'receipt', 'failure')}


class _Publication:
    """Pending survives failure-log errors; COMMIT is the logical publication point."""
    def __init__(self, ticket, *, prepare=False):
        if type(ticket) is not ResearchFitTicket:
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
    """Parent independently reconstructs paired hourly scores from saved weights."""
    directory = ticket.path.parent / 'launch/research'
    result_path = directory / 'result.json'
    if not result_path.is_file() or result_path.stat().st_size > 16 * 2**20:
        raise ValueError('Missing or oversized complete S result')
    result = json.loads(result_path.read_text('utf-8'))
    raw_digest = fingerprint(result)
    if (result.get('version') != 'v7-complete-neural-S-research-v1' or result.get('role') != 'S'
            or result.get('status') != 'completed' or result.get('test_only') is not False
            or result.get('automatic_adoption') is not False or result.get('formal_claim') is not False
            or result.get('recipe') != _canonical(recipe.to_dict())
            or result.get('input_id') != ticket.value['context']['input_id']
            or result.get('symbols') != ticket.value['context']['symbols']):
        raise ValueError('S result is not the complete original recipe/input')
    manifest = json.loads((directory / 'manifest.json').read_text('utf-8'))
    if fingerprint(manifest) != result.get('manifest_id') or any(result.get(k) != v for k, v in manifest.items() if k != 'folds'):
        raise ValueError('S result differs from its original numerical manifest')
    expected_folds = list(development_folds(ticket.value['context']['cutoff'], spec=recipe.rolling_spec))
    if len(result['folds']) != 3 or manifest['folds'] != expected_folds:
        raise ValueError('S result lacks all three fixed cold folds')
    hours, support, nominal = [], 0, 0
    with threadpool_limits(limits=1):
        for i, (fold, expected) in enumerate(zip(result['folds'], expected_folds, strict=True)):
            if (fold['status'] != 'completed' or fold.get('role') != 'S'
                    or any(fold.get(k) != v for k, v in expected.items() if k != 'role')):
                raise ValueError('Incomplete or changed cold-fold S evaluation')
            model_path, k0_path = directory / f'fold-{i}-model.json', directory / f'fold-{i}-k0.json'
            for reference, path in ((fold['model'], model_path), (fold['k0'], k0_path)):
                if reference != {'path': str(path), 'sha256': file_hash(path)} or path.stat().st_size > 16 * 2**20:
                    raise ValueError('Saved S model path/hash differs from its own fold')
            model = NeuralForwardModel.load(model_path)
            if (model.recipe.pipeline_id != recipe.pipeline_id
                    or model.receipt.get('payload_origin') != 'isolated_research_worker'):
                raise ValueError('S model borrowed a forward/artificial learner')
            k0 = _joint(json.loads(k0_path.read_text('utf-8')))
            recomputed = score_research_fold(data, model, k0, start=expected['fit_cutoff'],
                end=expected['score_end'], symbols=ticket.value['context']['symbols'])
            expected_fields = set(expected) | set(recomputed) | {'status', 'model', 'k0', 'budget_counts'}
            if set(fold) != expected_fields:
                raise ValueError('Parent-recomputed S fold schema differs from worker')
            numerical = {'predictions', 'hourly', 'losses', 'S', 'mean_seed_loss_diagnostic'}
            try:
                for key, value in recomputed.items():
                    assert_replayed(fold[key], value, roundoff=key in numerical)
            except ValueError as error:
                raise ValueError('Parent-recomputed S losses/predictions differ from worker') from error
            # A tolerated child ULP is never used to decide gain or eligibility.
            # The raw on-disk artifact and its original digest remain untouched.
            fold.update(recomputed)
            hours.extend(recomputed['hourly'])
            support += recomputed['support']
            nominal += recomputed['nominal_rows']
    losses = {name: float(np.mean([h['losses'][name] for h in hours])) for name in hours[0]['losses']} if hours else {}
    if not hours:
        raise ValueError('S aggregate does not match the complete paired hourly evidence')
    aggregate = {'support': support, 'nominal_rows': nominal, 'supported_hours': len(hours),
                 'nominal_hours': sum(f['nominal_hours'] for f in result['folds']), 'coverage': support / nominal}
    assert_replayed({k: result[k] for k in aggregate}, aggregate)
    assert_replayed(result['losses'], losses, roundoff=True)
    assert_replayed(result['conditional_losses'], losses, roundoff=True)
    result['losses'] = losses
    result['conditional_losses'] = losses
    result['numeric_replay'] = NUMERIC_REPLAY_RULE.copy()
    return result, {'path': str(result_path), 'sha256': file_hash(result_path), 'digest': raw_digest}


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


def launch_research(ticket_path):
    ticket = ResearchFitTicket.load(project_path(ticket_path))
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
            if importlib.util.find_spec('torch') is None:
                raise ValueError('Torch unavailable; complete S remains closed')
            command = [sys.executable, '-B', '-m', 'factorlab.discovery.neural_research_launcher', '--ticket', str(ticket.path)]
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
            receipt = {'version': VERSION, 'role': 'S', 'status': 'completed', 'workload': 'complete_three_fold_S',
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


def verify_completed_research(ticket_path):
    ticket = ResearchFitTicket.load(project_path(ticket_path))
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
    return {'ticket': ticket, 'reservation': record, 'result': result, 'receipt': receipt,
            'publication': proof, 'role': 'S', 'serving_eligible': False, 'formal_claim': False}


def worker(ticket_path):
    ticket = ResearchFitTicket.load(project_path(ticket_path))
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
            result = _run_research_worker(data, ticket.value['context']['cutoff'], recipe,
                symbols=ticket.value['context']['symbols'], directory=paths['start'].parent / 'research', budget=budget)
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
    parser = argparse.ArgumentParser(description='Fixed complete three-fold S neural worker')
    parser.add_argument('--ticket', required=True)
    worker(parser.parse_args(argv).ticket)


if __name__ == '__main__':
    main()
