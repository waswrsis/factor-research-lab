"""Admission verifier counterexamples, NOT real Linux producer acceptance.

Ticket, child-accounting and publication-proof boundaries are private test
monkeypatches; accounting/source use a real local authority. Synthetic kernel
records do not certify a production workload.
"""
import copy
import json
import os
import sqlite3
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_v7_neural_forward import fitted as fitted  # noqa: PLC0414
from test_v7_neural_forward_launcher import real_ticket_case as real_ticket_case  # noqa: PLC0414

from factorlab.discovery import neural_forward_admission as admission
from factorlab.discovery.evidence import file_hash
from factorlab.discovery.monthly_work_authority import DEFAULT_LIMITS, MonthlyWorkAuthority
from factorlab.discovery.neural_forward import NeuralForwardModel
from factorlab.discovery.neural_forward_contract import neural_pipeline
from factorlab.protocol import fingerprint


def write(path, value):
    path.write_text(json.dumps({'value': value, 'sha256': fingerprint(value)}), encoding='utf-8')


def read(path):
    return json.loads(path.read_text('utf-8'))['value']


@pytest.fixture
def evidence(tmp_path, monkeypatch, fitted):
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    (tmp_path / 'factorlab').mkdir()
    (tmp_path / 'factorlab/frozen.py').write_text('version=1')
    (tmp_path / 'pyproject.toml').write_text('[project]')
    (tmp_path / 'uv.lock').write_text('version=1')
    authority = MonthlyWorkAuthority(tmp_path / 'authority', run_id='test', task='volatility', universe=['X'])
    _, recipe, artificial, _ = fitted
    model_value = artificial.to_dict()
    r = model_value['receipt']
    r.pop('test_only'); r.pop('runtime_admitted')
    r.update(version='v7-neural-cold-payload-v2', payload_origin='isolated_forward_worker', admission_required=True)
    model = NeuralForwardModel.from_dict(model_value)
    counts = {key: model.receipt['budget_counts'].get(key, 0) for key in DEFAULT_LIMITS}
    child = {'test_only': False, 'finalized': True, 'actual_counts': counts.copy()}
    # This fixture has no real spawned child; production verification itself
    # is exercised by the missing-child-chain counterexample below.
    monkeypatch.setattr(admission, '_child_accounting', lambda *_: copy.deepcopy(child))
    reserved = authority.reserve('job', counts=counts, binding={'job': 'job'})
    lease = reserved['lease']
    # Kernel mocks are Linux-shaped; original authoritative lease remains the
    # host lease. This private seam only bridges the test host boot label.
    monkeypatch.setattr(admission, '_kernel', lambda *args: kernel_checks(*args))
    process = {'pid': os.getpid() + 100, 'created': datetime.now(UTC).timestamp()}
    parent = {'pid': lease['parent_pid'], 'created': lease['parent_created']}
    directory = tmp_path / 'ticket'; directory.mkdir()
    ticket_path = directory / 'ticket.json'
    ticket_value = {'fixture': 'fixed-ticket-boundary', 'reservation_id': reserved['reservation_id']}
    ticket_path.write_text(json.dumps(ticket_value))
    ticket_digest = fingerprint(ticket_value)
    launch = directory / 'launch'; launch.mkdir()
    context = {'purpose': 'V', 'serving_eligible': True, 'job': 'job', 'source_id': 'actual-feed',
               'cutoff': model.receipt['cutoff'], 'pipeline': neural_pipeline(recipe, recipe.rolling_spec)}
    source_hash = fingerprint(authority.manifest['source_files'])
    snapshot = {'training': model.receipt['training'], 'pipeline_id': recipe.pipeline_id,
                'feed': 'actual-feed', 'source_hash': source_hash, 'availability': 'actual_receipts'}

    def kernel(pids):
        raw = {'memory.max': str(768 * 2**20), 'memory.high': str(640 * 2**20), 'memory.swap.max': '0',
               'pids.max': '64', 'cpu.max': '25000 100000', 'memory.peak': str(100 * 2**20),
               'memory.events': 'oom 0\noom_kill 0', 'cgroup.procs': '\n'.join(map(str, pids))}
        return {'path': '/sys/fs/cgroup/test.service', 'inode': 100, 'boot_id': 'test-boot',
                'scope': 'whole_cgroup', 'platform': 'linux', 'peak': 100 * 2**20,
                'events': {'oom': 0, 'oom_kill': 0}, 'raw': raw, 'observed_at': datetime.now(UTC).isoformat()}

    first = kernel([parent['pid']])
    authority.start(reserved['reservation_id'], process)
    write(launch / 'start.json', {'ticket_digest': ticket_digest, 'process': process, 'lease_hash': reserved['lease_hash']})
    member = {'version': 'v7-neural-forward-membership-v1', 'ticket_digest': ticket_digest,
              'process': process, 'parent': parent, 'kernel': kernel([parent['pid'], process['pid']])}
    model.save(launch / 'model.json')
    result = {'version': 'v7-neural-forward-worker-v1', 'status': 'completed', 'ticket_digest': ticket_digest,
              'model_digest': model.digest, 'actual_counts': counts, 'process': process,
              'kernel': kernel([parent['pid'], process['pid']]), 'lease_hash': reserved['lease_hash']}
    write(launch / 'membership.json', member); write(launch / 'result.json', result)
    final = authority.complete(reserved['reservation_id'], result={'model_digest': model.digest,
        'worker_result_hash': file_hash(launch / 'result.json'), 'ticket_digest': ticket_digest}, actual_counts=counts)
    after = kernel([parent['pid']])
    ticket_ref = {'path': str(ticket_path), 'digest': ticket_digest}
    receipt = {'version': 'v7-neural-forward-launch-v1', 'status': 'completed', 'workload': 'full_forward',
        'ticket': ticket_ref, 'model': {'path': str(launch / 'model.json'), 'sha256': file_hash(launch / 'model.json'),
                                    'digest': model.digest},
        'worker_result': {'path': str(launch / 'result.json'), 'sha256': file_hash(launch / 'result.json')},
        'membership': {'path': str(launch / 'membership.json'), 'sha256': file_hash(launch / 'membership.json')},
        'reservation_id': final['reservation_id'], 'authority_manifest_hash': authority.manifest_hash,
        'completed_record_hash': fingerprint(final), 'lease_hash': final['lease_hash'], 'source_hash': source_hash,
        'context': context, 'process': process, 'kernel_before': first, 'kernel_after': after,
        'final_io_observation': {'wall': datetime.now(UTC).isoformat(), 'monotonic': time.monotonic(),
                                 'boot_id': lease['boot_id'], 'parent': parent},
        'authentication': 'local_run_bound_collector_not_admin_resistant'}
    write(launch / 'receipt.json', receipt)

    def verify():
        assert json.loads(ticket_path.read_text()) == ticket_value
        current = authority.get(reserved['reservation_id'])
        return {'authority': authority, 'reservation': current, 'recipe': recipe, 'context': copy.deepcopy(context),
                'snapshot': copy.deepcopy(snapshot), 'ticket_digest': ticket_digest, 'source_hash': source_hash,
                'data_contract': {'fixture_only': True}, 'runtime': None}

    ticket = SimpleNamespace(verify_completed_binding=verify)
    def load(path):
        if Path(path) != ticket_path:
            raise ValueError('Unknown fixture ticket')
        return ticket
    monkeypatch.setitem(sys.modules, 'factorlab.discovery.neural_forward_ticket',
                        SimpleNamespace(ForwardFitTicket=SimpleNamespace(load=load)))
    cert = {'version': admission.VERSION, 'launcher_receipt': {'path': str(launch / 'receipt.json'),
            'sha256': file_hash(launch / 'receipt.json')}, 'ticket': ticket_ref, 'payload_digest': model.digest}
    write(launch / 'certificate.json', cert)
    proof = {'certificate_digest': fingerprint(cert), 'receipt_hash': cert['launcher_receipt']['sha256'],
             'model_digest': model.digest, 'actual_counts_hash': fingerprint(counts),
             'kernel': kernel([parent['pid']]),
             'observation': {'wall': datetime.now(UTC).isoformat(), 'monotonic': time.monotonic(),
                             'boot_id': lease['boot_id'], 'parent': parent}}
    monkeypatch.setattr(admission, '_publication_proof', lambda _: copy.deepcopy(proof))
    return SimpleNamespace(model=model, artificial=artificial, cert=cert, receipt=receipt, launch=launch,
                           authority=authority, context=context, snapshot=snapshot, lease=lease, child=child,
                           proof=proof)


def kernel_checks(first, member, last, lease, parent, worker):
    # Exercise all production checks except mapping this Windows fixture's boot
    # label to its intentionally artificial Linux kernel records.
    first, member, last = map(copy.deepcopy, (first, member, last))
    from dataclasses import replace
    boot = '00000000-0000-0000-0000-000000000001'
    for k in (first, member, last):
        if k['boot_id'] != 'test-boot':
            raise ValueError('Fixture kernel boot changed')
        k['boot_id'] = boot
    original_kernel(first, member, last, replace(lease, boot_id='linux:' + boot), parent, worker)


original_kernel = admission._kernel
original_child_accounting = admission._child_accounting
original_publication_proof = admission._publication_proof


def reseal(evidence):
    write(evidence.launch / 'receipt.json', evidence.receipt)
    evidence.cert['launcher_receipt']['sha256'] = file_hash(evidence.launch / 'receipt.json')
    write(evidence.launch / 'certificate.json', evidence.cert)
    evidence.proof['certificate_digest'] = fingerprint(evidence.cert)
    evidence.proof['receipt_hash'] = evidence.cert['launcher_receipt']['sha256']


def test_complete_fixed_evidence_binds_model_job_and_pipeline(evidence):
    value = admission.verify_forward_admission(evidence.model, evidence.cert,
        expected_job='job', expected_pipeline=evidence.context['pipeline'])
    assert value['serving_eligible'] and value['model'].digest == evidence.model.digest
    assert value['publication'] == evidence.proof
    with pytest.raises(ValueError, match='job'):
        admission.verify_forward_admission(evidence.model, evidence.cert, expected_job='another')


def test_v1_artificial_payload_cannot_be_admitted(evidence):
    with pytest.raises(ValueError, match='Artificial'):
        admission.verify_forward_admission(evidence.artificial, evidence.cert)


@pytest.mark.parametrize('damage', ['workload', 'completed_hash', 'source', 'lease', 'pid', 'late_wall',
                                   'late_mono', 'boot', 'context', 'peak', 'cpu', 'oom', 'member'])
def test_recomputed_receipt_hash_is_not_authority(evidence, damage):
    r = evidence.receipt
    if damage == 'workload': r['workload'] = 'S_research_only'
    elif damage == 'completed_hash': r['completed_record_hash'] = '0' * 64
    elif damage == 'source': r['source_hash'] = '0' * 64
    elif damage == 'lease': r['lease_hash'] = '0' * 64
    elif damage == 'pid': r['process'] = {**r['process'], 'pid': 1}
    elif damage == 'late_wall': r['final_io_observation']['wall'] = evidence.lease['wall_deadline']
    elif damage == 'late_mono': r['final_io_observation']['monotonic'] = evidence.lease['monotonic_deadline']
    elif damage == 'boot': r['final_io_observation']['boot_id'] = 'other'
    elif damage == 'context': r['context'] = {**r['context'], 'job': 'borrowed'}
    elif damage == 'peak': r['kernel_after']['peak'] = 600 * 2**20
    elif damage == 'cpu': r['kernel_after']['raw']['cpu.max'] = '50000 100000'
    elif damage == 'oom': r['kernel_after']['events']['oom'] = 1
    elif damage == 'member': r['kernel_after']['raw']['cgroup.procs'] = '1'
    reseal(evidence)
    with pytest.raises(ValueError):
        admission.verify_forward_admission(evidence.model, evidence.cert)


def test_changed_model_file_rejected_even_with_new_file_reference(evidence):
    path = evidence.launch / 'model.json'
    wrapper = json.loads(path.read_text()); wrapper['model']['seeds'][0]['head']['bias'][0] += .01
    wrapper['digest'] = fingerprint(wrapper['model']); path.write_text(json.dumps(wrapper))
    evidence.receipt['model']['sha256'] = file_hash(path)
    reseal(evidence)
    with pytest.raises(ValueError):
        admission.verify_forward_admission(evidence.model, evidence.cert)


@pytest.mark.parametrize('name', ['failure.json', 'late.json', 'late-result.json'])
def test_late_or_failure_anchor_prevents_partial_publication_recovery(evidence, name):
    (evidence.launch / name).write_text('{}')
    with pytest.raises(ValueError, match='Failed or late'):
        admission.verify_forward_admission(evidence.model, evidence.cert)


def test_missing_completed_mirror_requires_authority_recovery_not_forged_flags(evidence):
    (evidence.authority.directory / 'events/0000000003.json').unlink()
    with pytest.raises(ValueError, match='mirror'):
        admission.verify_forward_admission(evidence.model, evidence.cert)


@pytest.mark.parametrize('name', ['certificate.json', 'start.json', 'membership.json', 'result.json'])
def test_partial_publication_is_not_relabelled_completed(evidence, name):
    (evidence.launch / name).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        admission.verify_forward_admission(evidence.model, evidence.cert)


def test_payload_post_deadline_file_timestamp_rejected(evidence):
    path = evidence.launch / 'receipt.json'
    late = datetime.fromisoformat(evidence.lease['wall_deadline']).timestamp() + 1
    os.utime(path, (late, late))
    with pytest.raises(ValueError, match='deadline'):
        admission.verify_forward_admission(evidence.model, evidence.cert)


def test_resource_trial_may_supply_capacity_but_never_serving(evidence):
    evidence.context.update(purpose='resource_trial', serving_eligible=False)
    evidence.receipt['context'] = copy.deepcopy(evidence.context); reseal(evidence)
    with pytest.raises(ValueError, match='Resource trial'):
        admission.verify_forward_admission(evidence.model, evidence.cert)
    assert not admission.verify_completed_artifact(evidence.cert, allow_resource_trial=True)['serving_eligible']


def test_only_reference_certificate_schema_is_accepted(evidence):
    evidence.cert['verified'] = True
    with pytest.raises(ValueError, match='schema'):
        admission.verify_forward_admission(evidence.model, evidence.cert)


def test_historical_load_does_not_require_old_parent_to_remain_alive(evidence, monkeypatch):
    from factorlab.discovery.work_lease import FrozenWorkLease
    monkeypatch.setattr(FrozenWorkLease, 'check', lambda *_: pytest.fail('Completed load cannot renew/check live lease'))
    assert admission.verify_forward_admission(evidence.model, evidence.cert)['serving_eligible']


def test_missing_original_child_actual_chain_cannot_be_replaced_by_summary(evidence, monkeypatch):
    monkeypatch.setattr(admission, '_child_accounting', original_child_accounting)
    with pytest.raises((ValueError, FileNotFoundError)):
        admission.verify_forward_admission(evidence.model, evidence.cert)


@pytest.mark.parametrize('damage', ['test_only', 'not_finalized', 'counts'])
def test_child_accounting_must_be_production_final_and_identical(evidence, damage):
    if damage == 'test_only':
        evidence.child['test_only'] = True
    elif damage == 'not_finalized':
        evidence.child['finalized'] = False
    else:
        evidence.child['actual_counts']['solve'] += 1
    with pytest.raises(ValueError, match='child accounting'):
        admission.verify_forward_admission(evidence.model, evidence.cert)


def test_public_completed_cache_cannot_borrow_another_valid_ticket(evidence, monkeypatch, tmp_path):
    from factorlab.discovery.neural_forward import fit_neural_forward
    module = sys.modules['factorlab.discovery.neural_forward_ticket']
    original_load = module.ForwardFitTicket.load
    requested = tmp_path / 'other-ticket' / 'ticket.json'
    requested.parent.mkdir()
    (requested.parent / 'launch').mkdir()
    requested.write_text('{}')
    write(requested.parent / 'launch/certificate.json', evidence.cert)
    record = evidence.authority.get(evidence.receipt['reservation_id'])
    different = SimpleNamespace(path=requested, digest='e' * 64, value={
        'authority_directory': str(evidence.authority.directory),
        'reservation_id': record['reservation_id']})
    monkeypatch.setattr(module.ForwardFitTicket, 'load',
        lambda path: different if Path(path) == requested else original_load(path))
    monkeypatch.setattr(module, 'open_existing_authority', lambda _: evidence.authority, raising=False)
    monkeypatch.setattr('factorlab.discovery.neural_forward_launcher.launch_forward_fit',
        lambda *_: pytest.fail('Completed cache cannot invoke training'))
    with pytest.raises(ValueError, match='ticket|Ticket'):
        fit_neural_forward(ticket_path=requested)


@pytest.mark.parametrize('stage', ['manifest', 'completion', 'transaction'])
def test_registry_failed_final_check_and_failed_marker_never_leave_usable_model(evidence, monkeypatch, tmp_path, stage):
    from factorlab.discovery import model_registry
    from factorlab.discovery.work_lease import FrozenWorkLease
    registry = model_registry.ModelRegistry(tmp_path / 'models')
    original_check = FrozenWorkLease.assert_parent_alive
    original_write = model_registry.immutable_json
    original_database = registry._publication_db
    calls = []
    updates = []
    def traced_database():
        db = original_database()
        db.set_trace_callback(lambda sql: updates.append(sql) if sql.startswith('UPDATE publications SET') else None)
        return db
    def fail_final_check(self, **kwargs):
        calls.append(1)
        if ((stage == 'manifest' and len(calls) == 3)
                or (stage == 'completion' and list(registry.directory.glob('*/registration_completed.json')))
                or (stage == 'transaction' and updates)):
            raise ValueError('original-monotonic-deadline')
        return original_check(self, **kwargs)
    def missing_failure_marker(path, value):
        if Path(path).name == 'registration_failed.json':
            raise OSError('failure-marker-persistence-unavailable')
        return original_write(path, value)
    monkeypatch.setattr(FrozenWorkLease, 'assert_parent_alive', fail_final_check)
    monkeypatch.setattr(model_registry, 'immutable_json', missing_failure_marker)
    monkeypatch.setattr(registry, '_publication_db', traced_database)
    with pytest.raises((ValueError, OSError)):
        registry.save(evidence.model, evidence.context['pipeline'], certificate=evidence.cert, expected_job='job')
    manifest = next(registry.directory.glob('*/manifest.json'), None)
    if manifest is not None:
        with pytest.raises((ValueError, FileNotFoundError)):
            registry.load(manifest.parent.name)


@pytest.mark.parametrize('damage', ['database_missing', 'anchor_missing', 'row_missing',
                                   'binding', 'completion', 'pending'])
def test_registry_publication_witness_cannot_be_missing_reset_or_rebound(evidence, tmp_path, damage):
    from factorlab.discovery.model_registry import ModelRegistry
    registry = ModelRegistry(tmp_path / 'models')
    identifier = registry.save(evidence.model, evidence.context['pipeline'],
                               certificate=evidence.cert, expected_job='job')
    database = registry.directory / 'neural_publications.sqlite'
    if damage == 'database_missing':
        database.unlink()
    elif damage == 'anchor_missing':
        (registry.directory / 'neural_publications_identity.json').unlink()
    else:
        with sqlite3.connect(database) as db:
            if damage == 'row_missing':
                db.execute('DELETE FROM publications WHERE id=?', (identifier,))
            elif damage == 'binding':
                db.execute('UPDATE publications SET binding=? WHERE id=?', ('{}', identifier))
            elif damage == 'completion':
                db.execute('UPDATE publications SET completion=? WHERE id=?', ('0' * 64, identifier))
            else:
                db.execute('UPDATE publications SET status=? WHERE id=?', ('pending', identifier))
    with pytest.raises((ValueError, FileNotFoundError)):
        registry.load(identifier)
    with pytest.raises((ValueError, FileNotFoundError)):
        registry.save(evidence.model, evidence.context['pipeline'], certificate=evidence.cert, expected_job='job')
    if damage == 'database_missing':
        assert not database.exists()


@pytest.mark.parametrize('damage', ['certificate_digest', 'receipt_hash', 'model_digest',
                                   'actual_counts_hash', 'kernel_peak', 'late', 'reordered'])
def test_committed_publication_proof_must_match_artifacts_resources_and_original_time(evidence, damage):
    proof = evidence.proof
    if damage.endswith(('_hash', '_digest')):
        proof[damage] = '0' * 64
    elif damage == 'kernel_peak':
        proof['kernel']['peak'] = 600 * 2**20
        proof['kernel']['raw']['memory.peak'] = str(600 * 2**20)
    elif damage == 'late':
        proof['observation']['monotonic'] = evidence.lease['monotonic_deadline']
    else:
        proof['observation']['wall'] = evidence.lease['wall_started_at']
    with pytest.raises(ValueError):
        admission.verify_forward_admission(evidence.model, evidence.cert)


def test_uncommitted_publication_is_not_replaced_by_complete_files(evidence, monkeypatch):
    def uncommitted(_):
        raise ValueError('Original publication transaction is pending')
    monkeypatch.setattr(admission, '_publication_proof', uncommitted)
    with pytest.raises(ValueError, match='pending'):
        admission.verify_forward_admission(evidence.model, evidence.cert)


@pytest.mark.parametrize('stage', ['final_resource', 'transaction'])
def test_real_launcher_wallet_and_witness_reject_last_check_with_unwritable_failure_log(
        real_ticket_case, monkeypatch, stage):
    """Actual ticket/wallet/witness; numerical result, OS process and kernel are stubs."""
    from test_v7_neural_forward_launcher import _launch_artificial_full

    from factorlab.discovery import forward_publication_witness as publication
    from factorlab.discovery import neural_forward_launcher as launcher
    from factorlab.discovery.work_lease import FrozenWorkLease, LeaseCheck, LeaseClosed
    ticket, authority = real_ticket_case.ticket, real_ticket_case.authority
    original_write, original_connect = launcher._write, publication._connect
    original_check = FrozenWorkLease.assert_parent_alive
    original = authority.get(ticket.value['reservation_id'])
    before = authority.status(original['actual_month'])['counts']
    updates = []

    def connect(path, *, readonly=False):
        db = original_connect(path, readonly=readonly)
        if not readonly:
            db.set_trace_callback(lambda sql: updates.append(sql)
                                  if sql.startswith('UPDATE publication SET') else None)
        return db

    def check(self, **kwargs):
        if stage == 'transaction' and updates:
            raise LeaseClosed(LeaseCheck('deadline'))
        return original_check(self, **kwargs)

    def final_resource_failure():
        raise ValueError('Final real-resource acceptance unavailable')

    def write(path, value):
        if path.name == 'failure.json':
            raise OSError('Failure log unavailable')
        result = original_write(path, value)
        if stage == 'final_resource' and path.name == 'certificate.json':
            monkeypatch.setattr(launcher, '_kernel_snapshot', final_resource_failure)
        return result

    monkeypatch.setattr(publication, '_connect', connect)
    monkeypatch.setattr(FrozenWorkLease, 'assert_parent_alive', check)
    monkeypatch.setattr(launcher, '_write', write)
    with pytest.raises((ValueError, LeaseClosed)):
        _launch_artificial_full(real_ticket_case, monkeypatch)
    assert (ticket.path.parent / 'launch/certificate.json').is_file()
    assert not (ticket.path.parent / 'launch/failure.json').exists()
    assert authority.get(ticket.value['reservation_id'])['status'] == 'completed'
    assert authority.status(original['actual_month'])['counts'] == before
    if stage == 'transaction':
        assert updates  # The failing lease check really follows the uncommitted UPDATE.
    with pytest.raises(ValueError, match='pending'):
        publication.ForwardPublicationWitness.read_completed(ticket)
