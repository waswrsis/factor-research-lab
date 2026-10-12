"""Fixed CPU full-model worker; local provenance is not admin authentication.

The original authority lease includes startup and every artifact write. This
module produces evidence, never a registry admission or an adoption decision.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil
from filelock import FileLock

from factorlab.protocol import fingerprint, project_path

from .evidence import file_hash
from .nonlinear_launcher import _kernel_snapshot, _read, _same_boundary, _validate_kernel, _write
from .process_policy import isolated_process_kwargs, restricted_environment, terminate_process_tree
from .work_lease import FrozenWorkLease

RECEIPT_VERSION = 'v7-neural-forward-launch-v1'
RESULT_VERSION = 'v7-neural-forward-worker-v1'
MEMBERSHIP_VERSION = 'v7-neural-forward-membership-v1'
LOG_LIMIT = 2**20


def _identity(pid=None):
    process = psutil.Process(os.getpid() if pid is None else pid)
    return {'pid': process.pid, 'created': process.create_time()}


def _load_ticket(path):
    from .neural_forward_ticket import ForwardFitTicket
    return ForwardFitTicket.load(project_path(path))


def _model(path):
    from .neural_forward import NeuralForwardModel
    return NeuralForwardModel.load(path)


def _verified_counts(authority, reservation_id):
    from .reserved_work_budget import ReservedChannelBudget
    result = ReservedChannelBudget.verify_counts(authority, reservation_id)
    if result['test_only'] or not result['finalized']:
        raise ValueError('Full worker requires final production prepaid accounting')
    return result['actual_counts']


def _prepare_publication(ticket):
    from .forward_publication_witness import ForwardPublicationWitness
    return ForwardPublicationWitness.prepare(ticket)


def _member(kernel, *identities):
    members = {int(value) for value in kernel['raw']['cgroup.procs'].split()}
    if any(identity['pid'] not in members for identity in identities):
        raise ValueError('Exact launcher/worker common cgroup membership is unavailable')


def _paths(ticket):
    root = ticket.path.parent / 'launch'
    return {name: root / (name + '.json') for name in
            ('model', 'result', 'membership', 'receipt', 'failure', 'start')}


def _observation(lease):
    lease.assert_parent_alive()
    # Same canonical microsecond wall representation as the original lease.
    from .work_lease import _boot_id
    return {'wall': datetime.now(UTC).isoformat(), 'monotonic': time.monotonic(),
            'boot_id': _boot_id(), 'parent': _identity()}


def _file(path):
    return {'path': str(path), 'sha256': file_hash(path)}


def _environment():
    env = restricted_environment()
    env.update({'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
                'CUDA_VISIBLE_DEVICES': '', 'PYTHONNOUSERSITE': '1'})
    # A caller's Python import overrides must not redirect the fixed worker.
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    return env


def _verify_outputs(ticket, record, process, first, last):
    paths = _paths(ticket)
    if paths['failure'].exists():
        raise ValueError('A consumed failed launch cannot become a completed artifact')
    membership, result = _read(paths['membership']), _read(paths['result'])
    lease = FrozenWorkLease.from_dict(record['lease'])
    parent = {'pid': lease.parent_pid, 'created': lease.parent_created}
    if (membership.get('version') != MEMBERSHIP_VERSION
            or record.get('process_identity') != process
            or membership.get('ticket_digest') != ticket.digest
            or membership.get('process') != process or membership.get('parent') != parent
            or result.get('version') != RESULT_VERSION or result.get('status') != 'completed'
            or result.get('ticket_digest') != ticket.digest or result.get('process') != process
            or result.get('lease_hash') != lease.digest):
        raise ValueError('Worker result or exact membership identity differs from ticket')
    for sample in (membership['kernel'], result['kernel']):
        if not _same_boundary(first, sample):
            raise ValueError('Worker escaped the registered common resource boundary')
        _member(sample, parent, process)
        _validate_kernel(first, sample)
        _validate_kernel(sample, last)
    _member(first, parent)
    _member(last, parent)
    _validate_kernel(first, last)
    model = _model(paths['model'])
    if (model.digest != result.get('model_digest') or model.pipeline_id != fingerprint(ticket.value['recipe'])
            or model.receipt.get('version') != 'v7-neural-cold-payload-v2'):
        raise ValueError('Saved full model differs from the frozen worker recipe/result')
    return model, result


def launch_forward_fit(ticket_path):
    """Spend one already-reserved fixed ticket; retries never issue a new lease.

    Exceptions fail closed and persist failure.json. Existing success is read
    through the separate admission verifier, never by re-running this entry.
    """
    ticket = _load_ticket(ticket_path)
    paths = _paths(ticket)
    directory = paths['model'].parent
    directory.mkdir(parents=True, exist_ok=True)
    lease = FrozenWorkLease.from_dict(ticket.value['lease'])
    process = None
    authority = None
    reservation_id = ticket.value['reservation_id']
    with FileLock(str(directory / 'launch.lock'), timeout=0):
        if any(paths[name].exists() for name in paths):
            raise ValueError('Forward ticket has already been attempted; no retry or refund')
        try:
            authority, record, _, _ = ticket.verify_for_launch()
            if record['status'] != 'reserved':
                raise ValueError('Forward ticket has already been attempted; no retry or refund')
            lease.assert_parent_alive()
            publication = _prepare_publication(ticket)
            first = _kernel_snapshot()
            _validate_kernel(first, first)
            _member(first, _identity())
            if importlib.util.find_spec('torch') is None:
                raise ValueError('Torch is unavailable; complete neural channel remains closed')
            command = [sys.executable, '-B', '-m', 'factorlab.discovery.neural_forward_launcher',
                       '--ticket', str(ticket.path)]
            with (directory / 'worker.log').open('xb') as log:
                process = subprocess.Popen(command, cwd=str(Path(__file__).resolve().parents[2]),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    env=_environment(), **isolated_process_kwargs())
                identity = _identity(process.pid)
                running = authority.start(reservation_id, identity)
                if running['status'] != 'running':
                    raise ValueError('Authority did not start the original live reservation')
                _write(paths['start'], {'ticket_digest': ticket.digest, 'process': identity,
                                       'lease_hash': lease.digest})
                while process.poll() is None:
                    lease.assert_parent_alive()
                    if log.tell() > LOG_LIMIT:
                        raise ValueError('Fixed worker exceeded its bounded output allowance')
                    time.sleep(min(.05, lease.check().remaining_seconds))
                if process.returncode != 0:
                    raise RuntimeError(f'Fixed full model worker exited {process.returncode}')
                if log.tell() > LOG_LIMIT:
                    raise ValueError('Fixed worker exceeded its bounded output allowance')
            lease.assert_parent_alive()
            # Revalidate source, frozen job, data and reservation after the worker.
            _, current, _, _ = ticket.verify_running_parent()
            last = _kernel_snapshot()
            model, result = _verify_outputs(ticket, current, identity, first, last)
            if _verified_counts(authority, reservation_id) != result['actual_counts']:
                raise ValueError('Worker actual costs differ from durable prepaid accounting')
            worker_result = _file(paths['result'])
            model_ref = {**_file(paths['model']), 'digest': model.digest}
            membership_ref = _file(paths['membership'])
            completed = authority.complete(reservation_id, result={
                'ticket_digest': ticket.digest, 'model_digest': model.digest,
                'worker_result_hash': worker_result['sha256']}, actual_counts=result['actual_counts'])
            if completed['status'] != 'completed':
                raise ValueError('Original reservation closed before accounting completion')
            # These are final reads of all workload/accounting inputs. The receipt
            # write that follows is itself checked under the SAME original lease.
            if (authority.get(reservation_id) != completed
                    or file_hash(paths['model']) != model_ref['sha256']
                    or file_hash(paths['result']) != worker_result['sha256']
                    or file_hash(paths['membership']) != membership_ref['sha256']):
                raise ValueError('Full model completion artifacts changed during final IO')
            final_kernel = _kernel_snapshot()
            _validate_kernel(last, final_kernel)
            _member(final_kernel, _identity())
            last = final_kernel  # Includes parent payload reload and accounting verification.
            receipt = {'version': RECEIPT_VERSION, 'status': 'completed', 'workload': 'full_forward',
                'ticket': {'path': str(ticket.path), 'digest': ticket.digest}, 'model': model_ref,
                'worker_result': worker_result, 'membership': membership_ref,
                'reservation_id': reservation_id, 'authority_manifest_hash': ticket.value['authority_manifest_hash'],
                'completed_record_hash': fingerprint(completed), 'lease_hash': lease.digest,
                'source_hash': ticket.value['source_hash'], 'context': ticket.value['context'],
                'process': identity, 'kernel_before': first, 'kernel_after': last,
                'final_io_observation': _observation(lease),
                'authentication': 'local_run_bound_collector_not_admin_resistant'}
            _write(paths['receipt'], receipt)
            reference = _file(paths['receipt'])
            certificate = {'version': 'v7-neural-forward-admission-v1',
                'launcher_receipt': reference, 'ticket': receipt['ticket'], 'payload_digest': model.digest}
            _write(directory / 'certificate.json', certificate)
            publication_kernel = _kernel_snapshot()
            _validate_kernel(last, publication_kernel)
            _member(publication_kernel, _identity())
            proof = {'certificate_digest': fingerprint(certificate), 'receipt_hash': reference['sha256'],
                'model_digest': model.digest, 'actual_counts_hash': fingerprint(result['actual_counts']),
                'kernel': publication_kernel, 'observation': _observation(lease)}
            response = {'status': 'completed', 'ticket_digest': ticket.digest,
                        'model': model_ref, 'receipt': reference, 'certificate': certificate}
            # The trusted witness rechecks this original lease immediately before
            # SQL COMMIT. That commit is publication's linearization point; no
            # further application-side check can invent another final-IO window.
            publication.complete(proof)
            return response
        except BaseException as error:
            # Never turn an already-written late receipt into usable success.
            try:
                _write(paths['failure'], {'version': 'v7-neural-forward-failure-v1',
                    'ticket_digest': ticket.digest, 'reason': type(error).__name__ + ':' + str(error)[:2000]})
            except BaseException as persistence_error:  # noqa: BLE001 - preserve/rethrow original failure below
                error.add_note('Failure evidence could not persist: ' + repr(persistence_error))
            try:
                if authority is not None:
                    state = authority.get(reservation_id)
                    if state['status'] in {'reserved', 'running'}:
                        authority.fail_consumed(reservation_id, str(error)[:2000])
            except BaseException as accounting_error:  # noqa: BLE001 - preserve/rethrow original failure below
                error.add_note('Nonrefundable authority closure failed: ' + repr(accounting_error))
            raise
        finally:
            if process is not None and process.poll() is None:
                terminate_process_tree(process)


def worker(ticket_path):
    """Fixed CLI worker. No fit function, command or evidence may be supplied."""
    ticket = _load_ticket(ticket_path)
    lease = FrozenWorkLease.from_dict(ticket.value['lease'])
    paths = _paths(ticket)
    with lease.as_worker():
        while not paths['start'].exists():
            lease.assert_parent_alive(as_worker=True)
            time.sleep(.01)
        start = _read(paths['start'])
        process = _identity()
        if start != {'ticket_digest': ticket.digest, 'process': process, 'lease_hash': lease.digest}:
            raise ValueError('Fixed worker start handshake differs from its real identity')
        authority, _, data, recipe = ticket.verify_for_launch(as_worker=True)
        from .neural_forward import _fit_neural_forward_payload
        from .reserved_work_budget import ReservedChannelBudget
        budget = ReservedChannelBudget.from_reservation(authority, ticket.value['reservation_id'])
        parent = {'pid': lease.parent_pid, 'created': lease.parent_created}
        kernel = _kernel_snapshot()
        _member(kernel, parent, process)
        _write(paths['membership'], {'version': MEMBERSHIP_VERSION, 'ticket_digest': ticket.digest,
                                    'process': process, 'parent': parent, 'kernel': kernel})
        model = _fit_neural_forward_payload(data, ticket.value['context']['cutoff'], recipe, budget)
        model.save(paths['model'])
        restored = _model(paths['model'])
        if restored.digest != model.digest:
            raise ValueError('Saved full model did not restore exactly')
        budget.check_time()
        budget.finalize()
        kernel_after = _kernel_snapshot()
        _validate_kernel(kernel, kernel_after)
        _member(kernel_after, parent, process)
        _write(paths['result'], {'version': RESULT_VERSION, 'status': 'completed',
            'ticket_digest': ticket.digest, 'model_digest': restored.digest, 'process': process,
            'lease_hash': lease.digest, 'actual_counts': budget.actual_counts, 'kernel': kernel_after})
        budget.check_time()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Fixed complete neural model worker')
    parser.add_argument('--ticket', required=True)
    args = parser.parse_args(argv)
    worker(args.ticket)


if __name__ == '__main__':
    main()
