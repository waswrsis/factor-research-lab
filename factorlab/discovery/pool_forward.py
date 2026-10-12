"""Frozen pool forward work on the runtime's one prepaid monthly authority.

This is trusted local accounting/publication evidence, not an administrator-
resistant signature, predictive skill, or an independent statistical gate.
Roles come from actual A0 adoption, protected V/C windows or frozen claim streams.
Numeric work, reload and publication share the original 300 second lease.
"""
from __future__ import annotations

import copy
import json
import os
import platform
from dataclasses import asdict, dataclass
from importlib import metadata

import numpy as np
from filelock import FileLock

from factorlab.protocol import fingerprint, project_path

from .evidence import file_hash
from .joint_pool import FrozenPool, JointPredictor, _valid, _vectors, fit_pool
from .monthly_work_authority import DEFAULT_LIMITS, MonthlyWorkAuthority
from .neural_forward import _joint
from .neural_forward_admission import _file_deadline, _kernel
from .neural_forward_admission import _observation as verify_observation
from .neural_forward_contract import _confirmed_adoption_binding
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
from .program_contract import Program
from .reserved_work_budget import ReservedChannelBudget
from .rolling import RollingSpec
from .v7_forward import ForwardRuntime
from .v7_lifecycle import stamp
from .v7_storage import immutable_record
from .work_lease import FrozenWorkLease

VERSION = 'v7-pool-forward-ticket-v2'
CERTIFICATE_VERSION = 'v7-pool-forward-admission-v1'
FIELDS = {'version', 'authority_directory', 'context', 'snapshot_ref', 'preparation_ref', 'recipe', 'runtime',
          'source_hash', 'reservation_id', 'authority_manifest_hash', 'terms_hash', 'lease', 'lease_hash'}


def runtime_identity():
    """Fixed convex dependencies; a missing optional Torch is irrelevant here."""
    packages = ('numpy', 'pandas', 'scipy', 'scikit-learn', 'pyarrow', 'filelock', 'psutil')
    return {'version': 'v7-pool-runtime-v1', 'python': platform.python_version(),
            'producer_sha256': file_hash(__file__),
            'packages': {name: metadata.version(name) for name in packages}}


@dataclass(frozen=True)
class PoolForwardRecipe:
    """Ticket adapter only: preserves the existing executable pool pipeline ID."""
    pool: FrozenPool
    rolling_spec: RollingSpec

    def __post_init__(self):
        if type(self.pool) is not FrozenPool or type(self.rolling_spec) is not RollingSpec:
            raise TypeError('Complete frozen pool and rolling specification required')
        if self.rolling_spec.task != 'volatility' or self.rolling_spec.horizon_hours != 12:
            raise ValueError('Pool forward uses the frozen 12h volatility task')
        for member in self.pool.members:
            program = Program.from_dict(member.program)
            if (program.to_dict() != member.program
                    or program._analyze()['required_lookback'] > member.lookback):
                raise ValueError('Frozen pool formula exceeds its exact causal language/window')
        object.__setattr__(self, 'pool', FrozenPool.from_dict(self.pool.to_dict()))
        object.__setattr__(self, '_hash', fingerprint(self.to_dict()))

    def to_dict(self):
        return json.loads(json.dumps({'version': 'v7-pool-forward-recipe-v1',
            'pool': self.pool.to_dict(), 'rolling_spec': asdict(self.rolling_spec)}))

    @classmethod
    def from_dict(cls, value):
        spec = dict(value['rolling_spec'])
        spec['event_bins'] = tuple(spec['event_bins'])
        result = cls(FrozenPool.from_dict(value['pool']), RollingSpec(**spec))
        if result.to_dict() != value:
            raise ValueError('Unknown pool forward recipe fields')
        return result

    @property
    def pipeline_id(self):
        return pipeline_for(self)['pipeline_id']

    @property
    def solve_limit(self):
        def count(pool):
            prefix = len(pool.reference_pool.members) if pool.reference_pool else 0
            current = 1 + (pool.max_profile_calls if any(
                m.parameter_mode == 'profile' and m.parameters for m in pool.members[prefix:]) else 0)
            return current + (count(pool.reference_pool) if pool.reference_pool else 0)
        if fingerprint(self.to_dict()) != self._hash:
            raise ValueError('Pool forward recipe mutated')
        return count(self.pool)


def pipeline_for(recipe):
    if fingerprint(recipe.to_dict()) != recipe._hash:
        raise ValueError('Pool forward recipe mutated')
    runtime = ForwardRuntime.__new__(ForwardRuntime)
    runtime.spec = recipe.rolling_spec
    return runtime.recipe_for(recipe.pool)


def _context(lifecycle, job, recipe, *, expected=None, active=False):
    lifecycle.verify()
    operations = [json.loads(row[0]) for row in lifecycle.db.execute('SELECT record FROM operations ORDER BY seq')]
    row = lifecycle.db.execute('SELECT cutoff,stream,pipeline FROM jobs WHERE id=?', (job,)).fetchone()
    pipeline = pipeline_for(recipe)
    if row is None or json.loads(row[2]) != pipeline:
        raise ValueError('Frozen pool work requires its authoritative complete recipe')
    cutoff, stream = stamp(row[0]), row[1]
    observed = lifecycle.now() if expected is None else stamp(expected['authorized_at'])
    historical = expected is not None
    lifecycle.fit_cutoff(cutoff)
    if recipe.rolling_spec.window_days != lifecycle.plan.training_window_days or cutoff > observed:
        raise ValueError('Pool forward changed the original fit cadence or training window')
    from .claim_protocol import ClaimProtocol
    protocol = ClaimProtocol(lifecycle) if ClaimProtocol._bindings(lifecycle) else None
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
    elif stream == 'A0_policy_trial':
        from .typed_policy_confirmation import nominal_pool_context
        binding = nominal_pool_context(lifecycle, job, recipe,
            authorized_at=observed, active=active or not historical)
    elif stream == 'A0':
        fallback = pipeline_for(PoolForwardRecipe(FrozenPool(role='reference', version='K0_v1'),
                                                 recipe.rolling_spec))
        if protocol is not None and protocol.endpoints['algorithm'].reference != fallback:
            raise ValueError('Algorithm fallback differs from original empty-pool K0')
        selected = lifecycle.adoption_pipeline(cutoff, fallback)
        if {k: v for k, v in selected.items() if k != 'complexity'} != pipeline:
            raise ValueError('Pool A0 differs from the actual month-start adoption policy')
        adoption = None if pipeline == fallback else _confirmed_adoption_binding(
            lifecycle, cutoff, pipeline, expected=expected.get('adoption') if historical else None,
            historical=historical)
        binding = {'role': 'K0_fallback' if adoption is None else 'A0_adopted', 'adoption': adoption}
    else:
        matches = [e for e in protocol.endpoints.values()
                   if lifecycle.fit_cycle_start(e.start) <= cutoff <= observed < stamp(e.end)
                   and stamp(e.start) <= observed
                   and ((e.reference_stream == stream and e.reference == pipeline)
                        or (e.slot == 'library_skill' and e.candidate_stream == stream and e.candidate == pipeline))
                   ] if protocol is not None else []
        if matches:
            if active and any(lifecycle.now() >= stamp(e.end) for e in matches):
                raise ValueError('Frozen claim work is no longer open')
            binding = {'role': 'fixed_claim_reference', 'protocol_digest': protocol.digest,
                       'slots': sorted(e.slot for e in matches)}
        else:
            batch, separator, role = stream.rpartition(':')
            row_batch = lifecycle.db.execute('SELECT identity,hash,status,selected FROM batches WHERE id=?',
                                             (batch,)).fetchone()
            if not separator or not row_batch or role not in {'reference', recipe.pipeline_id}:
                raise ValueError('Pool stream lacks an actual protected batch or frozen claim')
            identity = json.loads(row_batch[0])
            chosen = identity['reference'] if role == 'reference' else next(
                (c for c in identity['candidates'] if c['pipeline_id'] == role), None)
            if chosen is None or {k: v for k, v in chosen.items() if k != 'complexity'} != pipeline:
                raise ValueError('Pool differs from complete frozen protected recipe')
            confirmation = None
            if (lifecycle.fit_cycle_start(identity['start']) <= cutoff <= observed < stamp(identity['v_end'])
                    and stamp(identity['start']) <= observed):
                phase = 'V'
                if (not historical or active) and (row_batch[2] != 'frozen_V'
                                                   or lifecycle.now() >= stamp(identity['v_end'])):
                    raise ValueError('V work is no longer open')
            else:
                phase = 'C'
                found = lifecycle.exposure.db.execute(
                    'SELECT identity,consumed,result FROM confirmations WHERE batch=?', (batch,)).fetchone()
                if not found:
                    raise ValueError('C work lacks actual confirmation reservation')
                confirmation = json.loads(found[0])
                start = stamp(confirmation['actual_start'])
                if (not lifecycle.fit_cycle_start(start) <= cutoff < stamp(confirmation['end'])
                        or not start <= observed < stamp(confirmation['end'])
                        or role not in {'reference', confirmation['candidate_id']}):
                    raise ValueError('C work is outside actual selected confirmation')
                if (not historical or active) and (row_batch[2] != 'selected_C' or found[1] or found[2]
                        or not start <= lifecycle.now() < stamp(confirmation['end'])):
                    raise ValueError('C work is already consumed or closed')
            binding = {'role': phase, 'batch': batch, 'batch_hash': row_batch[1],
                       'confirmation': confirmation}
    if not any(o['kind'] == 'fit_planned' and o['detail']['job'] == job for o in operations):
        raise ValueError('Frozen pool job lacks its original planning event')
    result = {'plan_id': lifecycle.plan.digest, 'lifecycle_directory': str(lifecycle.directory),
              'job': job, 'cutoff': cutoff.isoformat(), 'stream': stream, 'pipeline': pipeline,
              'authorized_at': observed.isoformat(), **binding}
    from .quarantined_observation import job_binding
    prefix = job_binding(lifecycle, job)
    if prefix is not None:
        result['observation_prefix'] = prefix
    if historical and result != expected:
        raise ValueError('Pool historical authorization changed')
    return result


def _counts(solves):
    if type(solves) is not int or solves < 1:
        raise ValueError('Pool solve count must be a positive integer')
    return {k: solves if k == 'solve' else 0 for k in DEFAULT_LIMITS}


def _load_model(path):
    value = _read(path)
    if set(value) != {'model', 'digest'} or fingerprint(value['model']) != value['digest']:
        raise ValueError('Pool forward model bytes changed')
    return _joint(value['model'])


def _verify_model(model, recipe, snapshot, data):
    strict = _joint(model.to_dict())
    receipt = strict.receipt
    mask = data.train_mask(snapshot['cutoff']) & _valid(data.frame, _vectors(
        data.frame, recipe.pool, [dict(m.parameters) for m in recipe.pool.members]))
    training = {**snapshot['training'], 'pool_hash': recipe.pool.digest,
                'common_train_rows': int(mask.sum()),
                'common_support_hash': fingerprint(np.flatnonzero(mask).tolist()),
                'aggregation': 'equal_symbol_within_hour_then_equal_hour'}
    calls = receipt.get('solve_calls')
    if (strict.digest != model.digest or strict.pool.digest != recipe.pool.digest
            or strict.pipeline_id != recipe.pipeline_id or receipt.get('pipeline_id') != recipe.pipeline_id
            or strict.predictor.ridge != recipe.rolling_spec.ridge
            or receipt.get('cutoff') != snapshot['cutoff'] or receipt.get('stage') != 'cold_pool_fit'
            or receipt.get('version') != 1 or receipt.get('training') != training
            or strict.predictor.provenance != training
            or type(calls) is not int or not 1 <= calls <= recipe.solve_limit
            or receipt.get('receipt_id') != fingerprint({k: v for k, v in receipt.items() if k != 'receipt_id'})):
        raise ValueError('Pool model differs from actual causal snapshot, support or recipe')
    return strict


def _write(path, value):
    return immutable_record(path, {'value': value, 'sha256': fingerprint(value)})


def _record(path):
    value = _read(path)
    if set(value) != {'value', 'sha256'} or value['sha256'] != fingerprint(value['value']):
        raise ValueError('Frozen pool record checksum changed')
    return value['value']


@dataclass(frozen=True)
class PoolForwardTicket:
    path: object
    value: dict

    def __post_init__(self):
        object.__setattr__(self, 'path', project_path(self.path))
        object.__setattr__(self, 'value', copy.deepcopy(self.value))
        if set(self.value) != FIELDS or self.value['version'] != VERSION:
            raise ValueError('Unsupported frozen pool ticket schema')
        recipe = PoolForwardRecipe.from_dict(self.value['recipe'])
        if self.value['context']['pipeline'] != pipeline_for(recipe):
            raise ValueError('Frozen pool ticket lost its full recipe')
        object.__setattr__(self, '_digest', fingerprint(self.value))

    @property
    def digest(self):
        if fingerprint(self.value) != self._digest:
            raise ValueError('Frozen pool ticket was mutated')
        return self._digest

    @classmethod
    def load(cls, path):
        value = _read(path)
        if set(value) != {'value', 'sha256'} or fingerprint(value['value']) != value['sha256']:
            raise ValueError('Frozen pool ticket checksum changed')
        return cls(path, value['value'])

    def verify(self, *, status):
        if self.load(self.path).digest != self.digest:
            raise ValueError('Frozen pool ticket differs from original frozen bytes')
        v = self.value
        authority = open_existing_authority(v['authority_directory'])
        work = fingerprint({'plan': v['context']['plan_id'], 'job': v['context']['job']})
        if self.path != authority.directory / 'pool_forward_tickets' / work / 'ticket.json':
            raise ValueError('Frozen pool ticket moved outside its canonical work directory')
        record = authority.get(v['reservation_id'])
        from .forward_preparation import ForwardPreparation
        prepared_record = ForwardPreparation.verify_ticket_reference(self.path, v)
        recipe = PoolForwardRecipe.from_dict(v['recipe'])
        counts = _counts(recipe.solve_limit)
        if (prepared_record != record or record['binding']['purpose'] != 'pool_forward'
                or record['counts'] != counts or record['status'] != status
                or any(record[k] != v[k] for k in (
                    'authority_manifest_hash', 'terms_hash', 'lease', 'lease_hash'))
                or v['runtime'] != runtime_identity()
                or v['source_hash'] != fingerprint(authority.manifest['source_files'])):
            raise ValueError('Frozen pool ticket differs from canonical authority, runtime or work state')
        recipe = PoolForwardRecipe.from_dict(v['recipe'])
        reader = _LifecycleReader(v['context']['lifecycle_directory'])
        try:
            from .quarantined_observation import _original_observation_read_scope
            with _original_observation_read_scope(reader):
                result = reader.verify()
                _verify_authority_binding(reader, authority, result['operations'])
                if _context(reader, v['context']['job'], recipe, expected=v['context'],
                            active=status != 'completed') != v['context']:
                    raise ValueError('Frozen pool job or protected claim reference changed')
                snapshot, data, sources = _snapshot_from_actual_source(
                    reader, v['snapshot_ref'], recipe, expected_pipeline=pipeline_for(recipe))
                if (sources != authority.manifest['source_files']
                        or snapshot['snapshot_id'] != v['snapshot_ref']['snapshot_id']):
                    raise ValueError('Frozen pool snapshot belongs to another source or cutoff')
        finally:
            reader.close()
        from .nominal_short_fit_policy import verify_registered_lease
        verify_registered_lease(self, recipe, record)
        lease = FrozenWorkLease.from_dict(record['lease'])
        if status != 'completed':
            lease.assert_parent_alive()
            if os.getpid() != lease.parent_pid:
                raise ValueError('Fixed convex reference may run only in its original issuer')
        return authority, record, snapshot, data, recipe


def issue_pool_ticket(runtime, authority, job, pool):
    recipe = PoolForwardRecipe(pool, runtime.spec)
    if type(authority) is not MonthlyWorkAuthority or type(pool) is not FrozenPool:
        raise TypeError('Actual monthly authority and frozen pool recipe are required')
    work = fingerprint({'plan': runtime.lifecycle.plan.digest, 'job': job})
    folder = authority.directory / 'pool_forward_tickets' / work
    path = folder / 'ticket.json'
    # Existing work is found before any new lease or input preparation.
    with FileLock(str(folder) + '.lock', timeout=0):
        if path.exists():
            ticket = PoolForwardTicket.load(path)
            if ticket.value['recipe'] != json.loads(json.dumps(recipe.to_dict())):
                raise ValueError('Fixed monthly work cannot change its recipe')
            return ticket
        if folder.exists():
            raise ValueError('Partial pool forward ticket cannot reset its original work')
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
                raise ValueError('Pool forward and authority have different frozen sources')
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
            ticket = PoolForwardTicket(path, {**base, **{k: record[k] for k in (
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
        raise ValueError('Pool forward completion artifact changed')


def verify_pool_admission(model, certificate, *, expected_pipeline, expected_job):
    """Historical readback; no training, mutable receipt repair or renewed time."""
    from .forward_publication_witness import ForwardPublicationWitness
    cert = certificate
    if (not isinstance(cert, dict) or set(cert) != {'version', 'ticket', 'launcher_receipt', 'payload_digest'}
            or cert['version'] != CERTIFICATE_VERSION or cert['payload_digest'] != model.digest):
        raise ValueError('Invalid pool forward admission certificate')
    ticket = PoolForwardTicket.load(cert['ticket']['path'])
    if cert['ticket'] != {'path': str(ticket.path), 'digest': ticket.digest}:
        raise ValueError('Pool forward admission borrows another ticket')
    authority, record, snapshot, data, recipe = ticket.verify(status='completed')
    if (type(model) is not JointPredictor or pipeline_for(recipe) != expected_pipeline
            or model.pool.digest != recipe.pool.digest or ticket.value['context']['job'] != expected_job):
        raise ValueError('Pool forward model differs from actual job, training or complete recipe')
    _verify_model(model, recipe, snapshot, data)
    directory = ticket.path.parent / 'launch'
    if (directory / 'failure.json').exists() or _record(directory / 'certificate.json') != cert:
        raise ValueError('Pool forward publication failed or lacks its original certificate')
    receipt_path = directory / 'receipt.json'
    _verify_reference(cert['launcher_receipt'], receipt_path)
    receipt = _record(receipt_path)
    fields = {'version', 'ticket', 'model', 'completed_record_hash', 'actual_counts_hash',
              'snapshot_id', 'kernel_before', 'kernel_after', 'final_io_observation', 'authentication'}
    actual = ReservedChannelBudget.verify_counts(authority, record['reservation_id'])
    expected_counts = _counts(model.receipt['solve_calls'])
    if (set(receipt) != fields or receipt['version'] != 'v7-pool-forward-completion-v1'
            or receipt['ticket'] != cert['ticket'] or receipt['completed_record_hash'] != fingerprint(record)
            or receipt['snapshot_id'] != snapshot['snapshot_id']
            or receipt['actual_counts_hash'] != fingerprint(actual['actual_counts'])
            or actual['actual_counts'] != expected_counts or actual['test_only']
            or record['result'] != {'ticket_digest': ticket.digest, 'model_digest': model.digest,
                                    'snapshot_id': snapshot['snapshot_id']}
            or receipt['authentication'] != 'local_run_bound_collector_not_admin_resistant'):
        raise ValueError('Pool forward completion accounting differs from actual work')
    model_path = directory / 'model.json'
    _verify_reference(receipt['model'], model_path, digest=model.digest)
    if _load_model(model_path).digest != model.digest:
        raise ValueError('Pool forward payload differs from completed artifact')
    lease = FrozenWorkLease.from_dict(record['lease'])
    parent = {'pid': lease.parent_pid, 'created': lease.parent_created}
    if record['process_identity'] != parent:
        raise ValueError('Pool forward was not fitted by its original issuer')
    _kernel(receipt['kernel_before'], receipt['kernel_before'], receipt['kernel_after'], lease, parent, parent)
    verify_observation(receipt['final_io_observation'], lease)
    proof = ForwardPublicationWitness.read_completed(ticket)
    if (proof['certificate_digest'] != fingerprint(cert) or proof['receipt_hash'] != cert['launcher_receipt']['sha256']
            or proof['model_digest'] != model.digest or proof['actual_counts_hash'] != fingerprint(expected_counts)):
        raise ValueError('Pool forward lacks its committed final publication')
    _kernel(receipt['kernel_before'], receipt['kernel_after'], proof['kernel'], lease, parent, parent)
    verify_observation(proof['observation'], lease, previous=receipt['final_io_observation'])
    _file_deadline([model_path, receipt_path, directory / 'certificate.json'], lease)
    return {'reservation': record, 'receipt': receipt, 'snapshot': snapshot, 'ticket': ticket,
            'proof': proof}


def fit_pool_forward(runtime, authority, job, pool):
    """Exactly one prepaid fixed fit; only positive completed artifacts recover."""
    from .forward_publication_witness import ForwardPublicationWitness
    ticket = issue_pool_ticket(runtime, authority, job, pool)
    recipe = PoolForwardRecipe(pool, runtime.spec)
    directory = ticket.path.parent / 'launch'
    original = authority.get(ticket.value['reservation_id'])
    if original['status'] == 'completed':
        model = _load_model(directory / 'model.json')
        certificate = _record(directory / 'certificate.json')
        verify_pool_admission(model, certificate, expected_pipeline=pipeline_for(recipe), expected_job=job)
        return {'model': model, 'certificate': certificate, 'cached': True,
                'train': model.receipt['training'], 'ticket_digest': ticket.digest}
    with FileLock(str(directory) + '.lock', timeout=0):
        if directory.exists():
            raise ValueError('Pool forward was already attempted; no retry or refund')
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
                raise ValueError('Pool forward reservation closed before fitting')
            budget = ReservedChannelBudget.from_issuer_reservation(authority, record['reservation_id'])
            from .nominal_algorithm_fit import fit_original_forward
            model = fit_original_forward(runtime,authority,job,ticket,snapshot,data,recipe,budget,first)
            model.save(directory / 'model.json')
            restored = _load_model(directory / 'model.json')
            _verify_model(restored, recipe, snapshot, data)
            if restored.digest != model.digest:
                raise ValueError('Pool forward save/reload changed the model')
            budget.finalize()
            actual = ReservedChannelBudget.verify_counts(authority, record['reservation_id'])
            if actual['test_only'] or actual['actual_counts'] != _counts(model.receipt['solve_calls']):
                raise ValueError('Pool forward actual work differs from its prepaid solve')
            # Reconstruct actual receipt inputs after numeric work, before debit completion.
            ticket.verify(status='running')
            completed = authority.complete(record['reservation_id'], result={
                'ticket_digest': ticket.digest, 'model_digest': model.digest,
                'snapshot_id': snapshot['snapshot_id']}, actual_counts=actual['actual_counts'])
            if completed['status'] != 'completed' or authority.get(record['reservation_id']) != completed:
                raise ValueError('Pool forward accounting did not complete')
            last = _kernel_snapshot()
            _validate_kernel(first, last)
            _member(last, _identity())
            receipt = {'version': 'v7-pool-forward-completion-v1',
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
