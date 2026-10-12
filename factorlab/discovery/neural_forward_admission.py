"""Read-only, run-bound admission for a complete neural forward artifact.

Hashes link immutable evidence; they are not a remote signature or permission
from a caller. Only the fixed ticket/launcher protocol can supply the evidence.
Loading an old completed artifact never renews its original work lease.
"""
from __future__ import annotations

import json
import math
import uuid
from datetime import UTC, datetime
from pathlib import PurePosixPath

from factorlab.protocol import fingerprint, project_path

from .evidence import file_hash
from .monthly_work_authority import DEFAULT_LIMITS
from .neural_forward import NeuralForwardModel
from .nonlinear_launcher import _same_boundary, _validate_kernel
from .work_lease import FrozenWorkLease

VERSION = 'v7-neural-forward-admission-v1'
CERTIFICATE_FIELDS = {'version', 'launcher_receipt', 'ticket', 'payload_digest'}
RECEIPT_FIELDS = {'version', 'status', 'workload', 'ticket', 'model', 'worker_result',
                  'membership', 'reservation_id', 'authority_manifest_hash',
                  'completed_record_hash', 'lease_hash', 'source_hash', 'context',
                  'process', 'kernel_before', 'kernel_after', 'final_io_observation',
                  'authentication'}


def _fields(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError('Invalid ' + label + ' schema')
    return value


def _hash(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in '0123456789abcdef' for c in value)):
        raise ValueError('Invalid evidence digest')
    return value


def _read(path, *, maximum=4 * 2**20):
    path = project_path(path)
    if not path.is_file() or path.stat().st_size > maximum:
        raise ValueError('Missing or oversized admission evidence')

    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('Duplicate evidence JSON key')
            value[key] = item
        return value

    def invalid(_):
        raise ValueError('Nonfinite evidence JSON')

    return json.loads(path.read_text('utf-8'), object_pairs_hook=unique, parse_constant=invalid)


def _utc(value):
    if not isinstance(value, str):
        raise ValueError('Admission observation requires UTC text')  # noqa: TRY004
    moment = datetime.fromisoformat(value)
    if moment.tzinfo is None or moment.utcoffset() != UTC.utcoffset(moment):
        raise ValueError('Admission observation requires UTC time')
    return moment


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _observation(value, lease, *, previous=None):
    """Validate an ORIGINAL collector observation, not today's load clock."""
    _fields(value, {'wall', 'monotonic', 'boot_id', 'parent'}, 'completion observation')
    parent = _identity(value['parent'], label='observation parent')
    if parent != {'pid': lease.parent_pid, 'created': lease.parent_created}:
        raise ValueError('Completion observer is not the original launcher')
    wall, mono = _utc(value['wall']), value['monotonic']
    if (not _finite(mono) or value['boot_id'] != lease.boot_id
            or not _utc(lease.wall_started_at) <= wall < _utc(lease.wall_deadline)
            or not lease.monotonic_started <= mono < lease.monotonic_deadline):
        raise ValueError('Evidence is outside the original work lease')
    if previous and (wall < _utc(previous['wall']) or mono < previous['monotonic']):
        raise ValueError('Evidence clock moved backwards')
    return value


def _identity(value, *, label):
    _fields(value, {'pid', 'created'}, label)
    if type(value['pid']) is not int or value['pid'] <= 0 or not _finite(value['created']) or value['created'] <= 0:
        raise ValueError('Invalid original process identity')
    return value


def _file_deadline(paths, lease):
    # Local filesystem metadata is an extra engineering anchor, not a signature.
    delta = _utc(lease.wall_deadline) - datetime(1970, 1, 1, tzinfo=UTC)
    deadline_ns = ((delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds) * 1000
    for path in paths:
        stat = path.stat()
        if stat.st_mtime_ns >= deadline_ns or stat.st_ctime_ns >= deadline_ns:
            raise ValueError('Evidence file was created/modified after the original deadline')


def _kernel(first, member, last, lease, parent, worker):
    for sample in (first, member, last):
        path = sample.get('path')
        if (sample.get('scope') != 'whole_cgroup' or sample.get('platform') != 'linux'
                or type(sample.get('inode')) is not int or sample['inode'] <= 0
                or not isinstance(path, str) or not path.startswith('/sys/fs/cgroup/')
                or '..' in PurePosixPath(path).parts or str(PurePosixPath(path)) != path
                or str(uuid.UUID(sample['boot_id'])) != sample['boot_id']):
            raise ValueError('Invalid original Linux whole-cgroup boundary')
    _validate_kernel(first, last)
    _validate_kernel(first, member)
    _validate_kernel(member, last)
    if not _same_boundary(first, member):
        raise ValueError('Worker belonged to another resource boundary')
    if 'linux:' + first['boot_id'] != lease.boot_id:
        raise ValueError('Kernel and original lease boot identities differ')
    times = [_utc(k['observed_at']) for k in (first, member, last)]
    if not _utc(lease.wall_started_at) <= times[0] <= times[1] <= times[2] < _utc(lease.wall_deadline):
        raise ValueError('Kernel observations are outside the original lease')
    for snapshot, required in ((first, {parent['pid']}), (member, {parent['pid'], worker['pid']}),
                               (last, {parent['pid']})):
        try:
            members = {int(pid) for pid in snapshot['raw']['cgroup.procs'].split()}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('Missing original cgroup process membership') from exc
        if not required <= members:
            raise ValueError('Original parent/worker not in measured whole cgroup')


def _record(path):
    wrapped = _fields(_read(path), {'value', 'sha256'}, 'launcher record wrapper')
    if fingerprint(wrapped['value']) != wrapped['sha256']:
        raise ValueError('Launcher record checksum changed')
    return wrapped['value']


def _reference(value, path, *, extra=()):
    _fields(value, {'path', 'sha256', *extra}, 'fixed file reference')
    if project_path(value['path']) != path or file_hash(path) != _hash(value['sha256']):
        raise ValueError('Evidence file changed or is outside its fixed launch directory')


def _child_accounting(authority, reservation_id):
    from .reserved_work_budget import ReservedChannelBudget
    return ReservedChannelBudget.verify_counts(authority, reservation_id)


def _publication_proof(ticket):
    from .forward_publication_witness import ForwardPublicationWitness
    return ForwardPublicationWitness.read_completed(ticket)


def verify_completed_artifact(certificate, *, allow_resource_trial=False):
    """Verify fixed completed evidence, including historical authority/source.

    This reads the original completion clocks and does NOT require an old
    issuer to remain alive. There are no caller-provided validators or clocks.
    Resource trials only become usable resource evidence, never serving models.
    """
    from .neural_forward_ticket import ForwardFitTicket

    if type(allow_resource_trial) is not bool:
        raise ValueError('Resource permission must be explicit boolean')
    cert = _fields(certificate, CERTIFICATE_FIELDS, 'admission certificate')
    if cert['version'] != VERSION:
        raise ValueError('Unknown forward admission version')
    ticket_ref = _fields(cert['ticket'], {'path', 'digest'}, 'ticket reference')
    ticket_path = project_path(ticket_ref['path'])
    ticket = ForwardFitTicket.load(ticket_path)
    bound = ticket.verify_completed_binding()
    if bound['ticket_digest'] != _hash(ticket_ref['digest']):
        raise ValueError('Certificate borrows another ticket')
    context, record, snapshot = bound['context'], bound['reservation'], bound['snapshot']
    recipe = bound['recipe']
    directory = ticket_path.parent / 'launch'
    if any((directory / name).exists() for name in ('failure.json', 'late.json', 'late-result.json')):
        raise ValueError('Failed or late launch cannot be admitted')
    if _record(directory / 'certificate.json') != cert:
        raise ValueError('Certificate publication is missing or differs from the fixed launcher')
    _reference(cert['launcher_receipt'], directory / 'receipt.json')
    receipt = _fields(_record(directory / 'receipt.json'), RECEIPT_FIELDS, 'forward launcher receipt')
    if (receipt['version'] != 'v7-neural-forward-launch-v1' or receipt['status'] != 'completed'
            or receipt['workload'] != 'full_forward'
            or receipt['authentication'] != 'local_run_bound_collector_not_admin_resistant'
            or receipt['ticket'] != ticket_ref or receipt['context'] != context
            or receipt['source_hash'] != bound['source_hash']
            or receipt['reservation_id'] != record['reservation_id']
            or receipt['authority_manifest_hash'] != record['authority_manifest_hash']
            or receipt['completed_record_hash'] != fingerprint(record)
            or record['status'] != 'completed'):
        raise ValueError('Launcher receipt is not its original completed full-forward workload')
    trial = context['purpose'] == 'resource_trial'
    # Context comes only from the original ticket's completed-binding verifier,
    # which replays policy_context against the original frozen claim/selection.
    # This distinct prediction stream is not operational A0 authority.
    if (context['purpose'] not in {'resource_trial', 'A0', 'V', 'C', 'policy_trial'}
            or ((context['purpose'] == 'policy_trial')
                != (context.get('stream') == 'A0_policy_trial'))
            or type(context['serving_eligible']) is not bool
            or context['serving_eligible'] == trial or (trial and not allow_resource_trial)):
        raise ValueError('Resource trial or unpermitted purpose cannot become a serving model')
    _reference(receipt['model'], directory / 'model.json', extra={'digest'})
    _reference(receipt['worker_result'], directory / 'result.json')
    _reference(receipt['membership'], directory / 'membership.json')
    model = NeuralForwardModel.load(directory / 'model.json')
    if (model.digest != _hash(cert['payload_digest']) or model.digest != receipt['model']['digest']
            or model.pipeline_id != recipe.pipeline_id or model.recipe.to_dict() != recipe.to_dict()
            or model.receipt.get('version') != 'v7-neural-cold-payload-v2'
            or model.receipt.get('payload_origin') != 'isolated_forward_worker'
            or model.receipt.get('admission_required') is not True
            or any(key in model.receipt for key in ('test_only', 'runtime_admitted', 'verified'))
            or model.receipt['training'] != snapshot['training']
            or model.receipt['cutoff'] != context['cutoff']
            or snapshot['pipeline_id'] != model.pipeline_id
            or snapshot['feed'] != context['source_id']
            or snapshot['source_hash'] != bound['source_hash']
            or snapshot['availability'] != 'actual_receipts'):
        raise ValueError('Model payload is not the ticket-bound production cold learner')
    result = _fields(_record(directory / 'result.json'),
        {'version', 'status', 'ticket_digest', 'model_digest', 'actual_counts', 'process', 'kernel', 'lease_hash'},
        'forward worker result')
    membership = _fields(_record(directory / 'membership.json'),
        {'version', 'ticket_digest', 'process', 'parent', 'kernel'}, 'forward worker membership')
    worker = _identity(receipt['process'], label='worker identity')
    lease = FrozenWorkLease.from_dict(record['lease'])
    parent = {'pid': lease.parent_pid, 'created': lease.parent_created}
    if _record(directory / 'start.json') != {
            'ticket_digest': bound['ticket_digest'], 'process': worker, 'lease_hash': lease.digest}:
        raise ValueError('Original worker start handshake changed')
    if (worker['pid'] == parent['pid'] or worker != record['process_identity']
            or result['version'] != 'v7-neural-forward-worker-v1' or result['status'] != 'completed'
            or membership['version'] != 'v7-neural-forward-membership-v1'
            or result['ticket_digest'] != bound['ticket_digest']
            or membership['ticket_digest'] != bound['ticket_digest']
            or result['process'] != worker or membership['process'] != worker
            or membership['parent'] != parent or result['model_digest'] != model.digest
            or any(value != lease.digest for value in (record['lease_hash'], receipt['lease_hash'], result['lease_hash']))):
        raise ValueError('Worker identity or original lease does not match its authority')
    actual = _fields(result['actual_counts'], DEFAULT_LIMITS, 'actual work counts')
    if (any(type(value) is not int or value < 0 or value > record['counts'][key]
            for key, value in actual.items()) or actual != record['actual_counts']
            or any(model.receipt['budget_counts'][key] != actual[key] for key in model.receipt['budget_counts'])):
        raise ValueError('Completed model work differs from its nonrefundable accounting')
    expected_result = {'model_digest': model.digest, 'worker_result_hash': receipt['worker_result']['sha256'],
                       'ticket_digest': bound['ticket_digest']}
    if record['result'] != expected_result:
        raise ValueError('Authority did not complete these exact worker bytes')
    child = _child_accounting(bound['authority'], record['reservation_id'])
    if (child.get('test_only') is not False or child.get('finalized') is not True
            or child.get('actual_counts') != actual):
        raise ValueError('Original finalized child accounting differs from completed work')
    first, last = receipt['kernel_before'], receipt['kernel_after']
    _kernel(first, membership['kernel'], last, lease, parent, worker)
    _kernel(first, result['kernel'], last, lease, parent, worker)
    if _utc(result['kernel']['observed_at']) < _utc(membership['kernel']['observed_at']):
        raise ValueError('Worker result predates its membership')
    final = _observation(receipt['final_io_observation'], lease)
    if (_utc(final['wall']) < max(_utc(last['observed_at']), _utc(record['completed_at']))
            or not _utc(lease.wall_started_at).timestamp() <= worker['created'] <= _utc(membership['kernel']['observed_at']).timestamp()):
        raise ValueError('Original completion/process chronology is invalid')
    proof = _fields(_publication_proof(ticket), {'certificate_digest', 'receipt_hash', 'model_digest',
        'actual_counts_hash', 'kernel', 'observation'}, 'committed forward publication proof')
    if (proof['certificate_digest'] != fingerprint(cert)
            or proof['receipt_hash'] != cert['launcher_receipt']['sha256']
            or proof['model_digest'] != model.digest or proof['actual_counts_hash'] != fingerprint(actual)):
        raise ValueError('Committed forward publication belongs to other artifacts or work')
    _kernel(first, membership['kernel'], proof['kernel'], lease, parent, worker)
    _kernel(first, result['kernel'], proof['kernel'], lease, parent, worker)
    _validate_kernel(last, proof['kernel'])
    if _utc(proof['kernel']['observed_at']) < _utc(last['observed_at']):
        raise ValueError('Committed resource observation predates final launcher IO')
    observation = _observation(proof['observation'], lease, previous=final)
    if _utc(observation['wall']) < _utc(proof['kernel']['observed_at']):
        raise ValueError('Committed observation predates its resource measurement')
    _file_deadline([ticket_path, *(directory / name for name in
                    ('model.json', 'result.json', 'membership.json', 'receipt.json',
                     'start.json', 'certificate.json'))], lease)
    return {**bound, 'model': model, 'receipt': receipt, 'publication': proof,
            'certificate_digest': fingerprint(cert),
            'serving_eligible': not trial, 'evidence_scope': 'local_run_bound_not_admin_resistant'}


def verify_forward_admission(model, certificate, *, expected_pipeline=None, expected_job=None):
    """Validate a serving binding; never turn an S/test/resource artifact into it."""
    if not isinstance(model, NeuralForwardModel):
        raise TypeError('A complete neural forward model is required')
    if model.receipt.get('version') != 'v7-neural-cold-payload-v2':
        raise ValueError('Artificial/test-only neural artifacts have no production admission')
    validated = verify_completed_artifact(certificate)
    context = validated['context']
    if model.digest != validated['model'].digest:
        raise ValueError('Admission belongs to another model payload')
    if expected_job is not None and context['job'] != expected_job:
        raise ValueError('Admission belongs to another lifecycle job')
    if expected_pipeline is not None:
        expected = {k: v for k, v in expected_pipeline.items() if k != 'complexity'}
        if context['pipeline'] != json.loads(json.dumps(expected, allow_nan=False)):
            raise ValueError('Admission differs from the expected complete pipeline')
    return validated

