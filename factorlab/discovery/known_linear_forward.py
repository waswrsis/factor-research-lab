"""Fixed linear forward work on the runtime's one prepaid monthly authority.

This is trusted local accounting/publication evidence, not an administrator-
resistant signature, predictive skill, or an independent statistical gate.
Only the frozen K0 fallback and registered claim reference streams qualify.
Numeric work, reload and publication share the original 300 second lease.
"""
from __future__ import annotations

import copy
import json
import os
import platform
from dataclasses import dataclass
from importlib import metadata

from filelock import FileLock

from factorlab.protocol import fingerprint, project_path

from .evidence import file_hash
from .known_linear_reference import FrozenLinearRecipe, KnownLinearPredictor, fit_known_linear
from .monthly_work_authority import DEFAULT_LIMITS, MonthlyWorkAuthority
from .neural_forward_admission import _file_deadline, _kernel
from .neural_forward_admission import _observation as verify_observation
from .neural_forward_launcher import _identity, _member, _observation
from .neural_forward_ticket import (
    _bind_authority,
    _LifecycleReader,
    _read,
    _snapshot_from_actual_source,
    _verify_authority_binding,
    open_existing_authority,
)
from .nonlinear_launcher import _kernel_snapshot, _validate_kernel
from .reserved_work_budget import ReservedChannelBudget
from .v7_lifecycle import stamp
from .v7_storage import immutable_record
from .work_lease import FrozenWorkLease

VERSION = 'v7-known-forward-ticket-v2'
CERTIFICATE_VERSION = 'v7-known-forward-admission-v1'
FIELDS = {'version', 'authority_directory', 'context', 'snapshot_ref', 'preparation_ref', 'recipe', 'runtime',
          'source_hash', 'reservation_id', 'authority_manifest_hash', 'terms_hash', 'lease', 'lease_hash'}


def runtime_identity():
    """Fixed convex dependencies; a missing optional Torch is irrelevant here."""
    packages = ('numpy', 'pandas', 'scipy', 'scikit-learn', 'pyarrow', 'filelock', 'psutil')
    return {'version': 'v7-known-linear-runtime-v1', 'python': platform.python_version(),
            'packages': {name: metadata.version(name) for name in packages}}


def pipeline_for(recipe):
    return {'kind': 'v7_known_linear', 'pipeline_id': recipe.pipeline_id,
            'recipe': recipe.to_dict(), 'task': 'volatility', 'rolling_hash': recipe.rolling_spec.digest}


def _context(lifecycle, job, recipe):
    lifecycle.verify()
    operations = [json.loads(row[0]) for row in lifecycle.db.execute('SELECT record FROM operations ORDER BY seq')]
    row = lifecycle.db.execute('SELECT cutoff,stream,pipeline FROM jobs WHERE id=?', (job,)).fetchone()
    pipeline = pipeline_for(recipe)
    if row is None or json.loads(row[2]) != pipeline:
        raise ValueError('Fixed linear work requires its authoritative complete recipe')
    cutoff, stream = stamp(row[0]), row[1]
    lifecycle.fit_cutoff(cutoff)
    if recipe.rolling_spec.window_days != lifecycle.plan.training_window_days:
        raise ValueError('Fixed reference changed the lifecycle training window')
    from .fixed_recipe_oracle import job_context
    fixed_oracle = job_context(lifecycle, job, pipeline)
    from .nominal_actual_trial import job_context as nominal_job_context
    nominal_actual = nominal_job_context(lifecycle, job, pipeline)
    if fixed_oracle is not None and nominal_actual is not None:
        raise ValueError('Old oracle and new actual trial cannot share a runtime')
    from .nominal_algorithm_warmboot import job_context as warmboot_job_context
    warmboot = warmboot_job_context(lifecycle, job, pipeline)
    if warmboot is not None:
        if nominal_actual is not None or fixed_oracle is not None:
            raise ValueError('Warmboot cannot borrow old fixed/library authority')
        binding = warmboot
    elif nominal_actual is not None:
        binding = nominal_actual
    elif fixed_oracle is not None:
        binding = fixed_oracle
    elif stream == 'A0':
        if recipe.name != 'K0':
            raise ValueError('Only the frozen K0 fallback may use the reference A0 path')
        selected = lifecycle.adoption_pipeline(cutoff, pipeline)
        if {k: v for k, v in selected.items() if k != 'complexity'} != pipeline:
            raise ValueError('Fixed K0 cannot replace an already confirmed month-start learner')
        from .claim_protocol import ClaimProtocol
        if ClaimProtocol._bindings(lifecycle):
            protocol = ClaimProtocol(lifecycle)
            if protocol.endpoints['algorithm'].reference != pipeline:
                raise ValueError('Fixed K0 cannot change the registered algorithm fallback identity')
        binding = {'role': 'K0_fallback'}
    else:
        from .claim_protocol import ClaimProtocol
        protocol = ClaimProtocol(lifecycle)
        matches = [e for e in protocol.endpoints.values()
                   if stamp(e.start) <= cutoff < stamp(e.end)
                   and ((e.reference_stream == stream and e.reference == pipeline)
                        or (e.slot == 'library_skill' and e.candidate_stream == stream and e.candidate == pipeline))]
        if not matches:
            raise ValueError('Fixed linear stream is not a registered six-month claim reference')
        binding = {'role': 'fixed_claim_reference', 'protocol_digest': protocol.digest,
                   'slots': sorted(e.slot for e in matches)}
    if not any(o['kind'] == 'fit_planned' and o['detail']['job'] == job for o in operations):
        raise ValueError('Fixed linear job lacks its original planning event')
    result = {'plan_id': lifecycle.plan.digest, 'lifecycle_directory': str(lifecycle.directory),
            'job': job, 'cutoff': cutoff.isoformat(), 'stream': stream, 'pipeline': pipeline, **binding}
    from .quarantined_observation import job_binding
    prefix = job_binding(lifecycle, job)
    if prefix is not None:
        result['observation_prefix'] = prefix
    return result


def _write(path, value):
    return immutable_record(path, {'value': value, 'sha256': fingerprint(value)})


def _record(path):
    value = _read(path)
    if set(value) != {'value', 'sha256'} or value['sha256'] != fingerprint(value['value']):
        raise ValueError('Fixed linear record checksum changed')
    return value['value']


@dataclass(frozen=True)
class KnownForwardTicket:
    path: object
    value: dict

    def __post_init__(self):
        object.__setattr__(self, 'path', project_path(self.path))
        object.__setattr__(self, 'value', copy.deepcopy(self.value))
        if set(self.value) != FIELDS or self.value['version'] != VERSION:
            raise ValueError('Unsupported fixed linear ticket schema')
        recipe = FrozenLinearRecipe.from_dict(self.value['recipe'])
        if self.value['context']['pipeline'] != pipeline_for(recipe):
            raise ValueError('Fixed linear ticket lost its full recipe')
        object.__setattr__(self, '_digest', fingerprint(self.value))

    @property
    def digest(self):
        if fingerprint(self.value) != self._digest:
            raise ValueError('Fixed linear ticket was mutated')
        return self._digest

    @classmethod
    def load(cls, path):
        value = _read(path)
        if set(value) != {'value', 'sha256'} or fingerprint(value['value']) != value['sha256']:
            raise ValueError('Fixed linear ticket checksum changed')
        return cls(path, value['value'])

    def verify(self, *, status):
        if self.load(self.path).digest != self.digest:
            raise ValueError('Fixed linear ticket differs from original frozen bytes')
        v = self.value
        authority = open_existing_authority(v['authority_directory'])
        work = fingerprint({'plan': v['context']['plan_id'], 'job': v['context']['job']})
        if self.path != authority.directory / 'known_forward_tickets' / work / 'ticket.json':
            raise ValueError('Fixed linear ticket moved outside its canonical work directory')
        record = authority.get(v['reservation_id'])
        from .forward_preparation import ForwardPreparation
        counts = {k: int(k == 'solve') for k in DEFAULT_LIMITS}
        if (record['binding']['purpose'] != 'known_reference'
                or record['counts'] != counts or record['status'] != status
                or any(record[k] != v[k] for k in (
                    'authority_manifest_hash', 'terms_hash', 'lease', 'lease_hash'))
                or v['runtime'] != runtime_identity()
                or v['source_hash'] != fingerprint(authority.manifest['source_files'])):
            raise ValueError('Fixed linear ticket differs from canonical authority, runtime or work state')
        lease = FrozenWorkLease.from_dict(record['lease'])
        # Reject already closed work before reconstructing expensive inputs.
        # A successful verification still requires the full closing source
        # audit and the original final physical check below.
        if status != 'completed':
            lease.assert_parent_alive()
        recipe = FrozenLinearRecipe.from_dict(v['recipe'])
        reader = _LifecycleReader(v['context']['lifecycle_directory'])
        try:
            from .quarantined_observation import _original_observation_read_scope
            with _original_observation_read_scope(reader):
                prepared_record = ForwardPreparation.verify_ticket_reference(self.path, v)
                if prepared_record != record:
                    raise ValueError('Fixed linear ticket differs from canonical authority, runtime or work state')
                result = reader.verify()
                _verify_authority_binding(reader, authority, result['operations'])
                if _context(reader, v['context']['job'], recipe) != v['context']:
                    raise ValueError('Fixed linear job or protected claim reference changed')
                snapshot, data, sources = _snapshot_from_actual_source(
                    reader, v['snapshot_ref'], recipe, expected_pipeline=pipeline_for(recipe))
                if (sources != authority.manifest['source_files']
                        or snapshot['snapshot_id'] != v['snapshot_ref']['snapshot_id']):
                    raise ValueError('Fixed linear snapshot belongs to another source or cutoff')
        finally:
            reader.close()
        from .nominal_short_fit_policy import verify_registered_lease
        verify_registered_lease(self, recipe, record)
        if status != 'completed':
            lease.assert_parent_alive()
            if os.getpid() != lease.parent_pid:
                raise ValueError('Fixed convex reference may run only in its original issuer')
        return authority, record, snapshot, data, recipe


def issue_known_ticket(runtime, authority, job, recipe):
    if type(authority) is not MonthlyWorkAuthority or type(recipe) is not FrozenLinearRecipe:
        raise TypeError('Actual monthly authority and fixed linear recipe are required')
    work = fingerprint({'plan': runtime.lifecycle.plan.digest, 'job': job})
    folder = authority.directory / 'known_forward_tickets' / work
    path = folder / 'ticket.json'
    # Existing work is found before any new lease or input preparation.
    with FileLock(str(folder) + '.lock', timeout=0):
        if path.exists():
            ticket = KnownForwardTicket.load(path)
            if ticket.value['recipe'] != json.loads(json.dumps(recipe.to_dict())):
                raise ValueError('Fixed monthly work cannot change its recipe')
            return ticket
        if folder.exists():
            raise ValueError('Partial fixed reference ticket cannot reset its original work')
        from .forward_service_scope import check_service_work
        check_service_work(runtime, new_work=True)
        from .nominal_short_fit_policy import issue_registered_lease
        lease = issue_registered_lease(runtime, job, recipe)
        if lease is None:
            lease = FrozenWorkLease.issue(300)
        check_service_work(runtime, lease=lease)
        from .quarantined_observation import _original_observation_read_scope
        with _original_observation_read_scope(runtime.lifecycle):
            from .nominal_algorithm_warmboot import require_fit_opportunity
            require_fit_opportunity(runtime, job)
            lease.assert_parent_alive()
            context = _context(runtime.lifecycle, job, recipe)
            if runtime.identity['source_files'] != authority.manifest['source_files']:
                raise ValueError('Fixed reference and authority have different frozen sources')
            from .forward_preparation import ForwardPreparation
            _bind_authority(runtime, authority)
            folder.mkdir(parents=True)
            immutable_record(folder / 'preparation.json', {'job': job, 'lease': lease.to_dict()})
            preparation = ForwardPreparation.begin(runtime, authority, job, recipe, lease=lease)
            prepared = preparation.snapshot()
            prepared.verify()
            base = {'version': VERSION, 'authority_directory': str(authority.directory),
                    'context': context, 'snapshot_ref': preparation.snapshot_ref,
                    'preparation_ref': preparation.reference,
                    'recipe': json.loads(json.dumps(recipe.to_dict())), 'runtime': runtime_identity(),
                    'source_hash': fingerprint(runtime.identity['source_files'])}
            record = preparation.record
            ticket = KnownForwardTicket(path, {**base, **{k: record[k] for k in (
                'reservation_id', 'authority_manifest_hash', 'terms_hash', 'lease', 'lease_hash')}})
            _write(path, ticket.value)
            preparation.seal_ticket(path, ticket.value)
        lease.assert_parent_alive()
        return ticket


def _reference(path, *, digest=None):
    value = {'path': str(path), 'sha256': file_hash(path)}
    return {**value, 'digest': digest} if digest is not None else value


def _verify_reference(reference, path, *, digest=None):
    if reference != _reference(path, digest=digest):
        raise ValueError('Fixed reference completion artifact changed')


def verify_known_admission(model, certificate, *, expected_pipeline, expected_job):
    """Historical readback; no training, mutable receipt repair or renewed time."""
    from .forward_publication_witness import ForwardPublicationWitness
    cert = certificate
    if (not isinstance(cert, dict) or set(cert) != {'version', 'ticket', 'launcher_receipt', 'payload_digest'}
            or cert['version'] != CERTIFICATE_VERSION or cert['payload_digest'] != model.digest):
        raise ValueError('Invalid fixed reference admission certificate')
    ticket = KnownForwardTicket.load(cert['ticket']['path'])
    if cert['ticket'] != {'path': str(ticket.path), 'digest': ticket.digest}:
        raise ValueError('Fixed reference admission borrows another ticket')
    authority, record, snapshot, _, recipe = ticket.verify(status='completed')
    if (type(model) is not KnownLinearPredictor or pipeline_for(recipe) != expected_pipeline
            or model.recipe != recipe or ticket.value['context']['job'] != expected_job
            or model.receipt['training'] != snapshot['training']):
        raise ValueError('Fixed reference model differs from actual job, training or complete recipe')
    directory = ticket.path.parent / 'launch'
    if (directory / 'failure.json').exists() or _record(directory / 'certificate.json') != cert:
        raise ValueError('Fixed reference publication failed or lacks its original certificate')
    receipt_path = directory / 'receipt.json'
    _verify_reference(cert['launcher_receipt'], receipt_path)
    receipt = _record(receipt_path)
    fields = {'version', 'ticket', 'model', 'completed_record_hash', 'actual_counts_hash',
              'snapshot_id', 'kernel_before', 'kernel_after', 'final_io_observation', 'authentication'}
    actual = ReservedChannelBudget.verify_counts(authority, record['reservation_id'])
    expected_counts = {k: int(k == 'solve') for k in DEFAULT_LIMITS}
    if (set(receipt) != fields or receipt['version'] != 'v7-known-forward-completion-v1'
            or receipt['ticket'] != cert['ticket'] or receipt['completed_record_hash'] != fingerprint(record)
            or receipt['snapshot_id'] != snapshot['snapshot_id']
            or receipt['actual_counts_hash'] != fingerprint(actual['actual_counts'])
            or actual['actual_counts'] != expected_counts or actual['test_only']
            or record['result'] != {'ticket_digest': ticket.digest, 'model_digest': model.digest,
                                    'snapshot_id': snapshot['snapshot_id']}
            or receipt['authentication'] != 'local_run_bound_collector_not_admin_resistant'):
        raise ValueError('Fixed reference completion accounting differs from actual work')
    model_path = directory / 'model.json'
    _verify_reference(receipt['model'], model_path, digest=model.digest)
    if KnownLinearPredictor.load(model_path).digest != model.digest:
        raise ValueError('Fixed reference payload differs from completed artifact')
    lease = FrozenWorkLease.from_dict(record['lease'])
    parent = {'pid': lease.parent_pid, 'created': lease.parent_created}
    if record['process_identity'] != parent:
        raise ValueError('Fixed reference was not fitted by its original issuer')
    _kernel(receipt['kernel_before'], receipt['kernel_before'], receipt['kernel_after'], lease, parent, parent)
    verify_observation(receipt['final_io_observation'], lease)
    proof = ForwardPublicationWitness.read_completed(ticket)
    if (proof['certificate_digest'] != fingerprint(cert) or proof['receipt_hash'] != cert['launcher_receipt']['sha256']
            or proof['model_digest'] != model.digest or proof['actual_counts_hash'] != fingerprint(expected_counts)):
        raise ValueError('Fixed reference lacks its committed final publication')
    _kernel(receipt['kernel_before'], receipt['kernel_after'], proof['kernel'], lease, parent, parent)
    verify_observation(proof['observation'], lease, previous=receipt['final_io_observation'])
    _file_deadline([model_path, receipt_path, directory / 'certificate.json'], lease)
    return {'reservation': record, 'receipt': receipt, 'snapshot': snapshot, 'ticket': ticket,
            'proof': proof}


def fit_known_forward(runtime, authority, job, recipe):
    """Exactly one prepaid fixed fit; only positive completed artifacts recover."""
    from .forward_publication_witness import ForwardPublicationWitness
    ticket = issue_known_ticket(runtime, authority, job, recipe)
    directory = ticket.path.parent / 'launch'
    original = authority.get(ticket.value['reservation_id'])
    if original['status'] == 'completed':
        model = KnownLinearPredictor.load(directory / 'model.json')
        certificate = _record(directory / 'certificate.json')
        verify_known_admission(model, certificate, expected_pipeline=pipeline_for(recipe), expected_job=job)
        return {'model': model, 'certificate': certificate, 'cached': True,
                'train': model.receipt['training'], 'ticket_digest': ticket.digest}
    with FileLock(str(directory) + '.lock', timeout=0):
        if directory.exists():
            raise ValueError('Fixed reference was already attempted; no retry or refund')
        directory.mkdir()
        try:
            authority, record, snapshot, data, recipe = ticket.verify(status='reserved')
            lease = FrozenWorkLease.from_dict(record['lease'])
            publication = ForwardPublicationWitness.prepare(ticket)
            first = _kernel_snapshot()
            _validate_kernel(first, first)
            _member(first, _identity())
            running = authority.start(record['reservation_id'], _identity())
            if running['status'] != 'running':
                raise ValueError('Fixed reference reservation closed before fitting')
            budget = ReservedChannelBudget.from_issuer_reservation(authority, record['reservation_id'])
            from .nominal_algorithm_fit import fit_original_forward
            model = fit_original_forward(runtime,authority,job,ticket,snapshot,data,recipe,budget,first)
            model.save(directory / 'model.json')
            restored = KnownLinearPredictor.load(directory / 'model.json')
            if restored.digest != model.digest:
                raise ValueError('Fixed reference save/reload changed the model')
            budget.finalize()
            actual = ReservedChannelBudget.verify_counts(authority, record['reservation_id'])
            if actual['test_only'] or actual['actual_counts'] != {k: int(k == 'solve') for k in DEFAULT_LIMITS}:
                raise ValueError('Fixed reference actual work differs from its prepaid solve')
            # Reconstruct actual receipt inputs after numeric work, before debit completion.
            ticket.verify(status='running')
            completed = authority.complete(record['reservation_id'], result={
                'ticket_digest': ticket.digest, 'model_digest': model.digest,
                'snapshot_id': snapshot['snapshot_id']}, actual_counts=actual['actual_counts'])
            if completed['status'] != 'completed' or authority.get(record['reservation_id']) != completed:
                raise ValueError('Fixed reference accounting did not complete')
            last = _kernel_snapshot()
            _validate_kernel(first, last)
            _member(last, _identity())
            receipt = {'version': 'v7-known-forward-completion-v1',
                'ticket': {'path': str(ticket.path), 'digest': ticket.digest},
                'model': _reference(directory / 'model.json', digest=model.digest),
                'completed_record_hash': fingerprint(completed),
                'actual_counts_hash': fingerprint(actual['actual_counts']), 'snapshot_id': snapshot['snapshot_id'],
                'kernel_before': first, 'kernel_after': last, 'final_io_observation': _observation(lease),
                'authentication': 'local_run_bound_collector_not_admin_resistant'}
            _write(directory / 'receipt.json', receipt)
            certificate = {'version': CERTIFICATE_VERSION, 'ticket': receipt['ticket'],
                           'launcher_receipt': _reference(directory / 'receipt.json'), 'payload_digest': model.digest}
            _write(directory / 'certificate.json', certificate)
            final = _kernel_snapshot()
            _validate_kernel(last, final)
            _member(final, _identity())
            response = {'model': model, 'certificate': certificate, 'cached': False,
                        'train': snapshot['training'], 'ticket_digest': ticket.digest}
            publication.complete({'certificate_digest': fingerprint(certificate),
                'receipt_hash': certificate['launcher_receipt']['sha256'], 'model_digest': model.digest,
                'actual_counts_hash': fingerprint(actual['actual_counts']), 'kernel': final,
                'observation': _observation(lease)})
            return response
        except BaseException as error:
            try:
                _write(directory / 'failure.json', {'reason': type(error).__name__ + ':' + str(error)[:2000]})
            except BaseException as failure_error:  # noqa: BLE001
                error.add_note('Failure evidence could not persist: ' + repr(failure_error))
            try:
                record = authority.get(ticket.value['reservation_id'])
                if record['status'] in {'reserved', 'running'}:
                    authority.fail_consumed(record['reservation_id'], str(error)[:1000])
            except BaseException as authority_error:  # noqa: BLE001
                error.add_note('Nonrefundable authority closure failed: ' + repr(authority_error))
            raise
