"""Synthetic process seams: these tests confer no Linux resource admission."""
from __future__ import annotations

import copy
import json
import multiprocessing
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from factorlab.discovery import neural_forward_launcher as module
from factorlab.discovery.work_lease import FrozenWorkLease, LeaseCheck, LeaseClosed
from factorlab.protocol import fingerprint


@pytest.fixture
def launch_case(tmp_path, monkeypatch):
    lease = FrozenWorkLease.issue()
    parent = {'pid': lease.parent_pid, 'created': lease.parent_created}
    child = {'pid': lease.parent_pid + 100000, 'created': lease.parent_created + 1}
    counts = {k: 0 for k in ('gpt_attempt', 'family', 'ast', 'f', 'solve', 'tree_fit', 'neural_fit', 'epoch')}
    record = {'status': 'reserved', 'lease': lease.to_dict(), 'reservation_id': 'reserved-test'}
    class Authority:
        def start(self, reservation_id, identity):
            assert reservation_id == record['reservation_id']
            record.update(status='running', process_identity=identity)
            return copy.deepcopy(record)
        def complete(self, reservation_id, *, result, actual_counts):
            record.update(status='completed', result=result, actual_counts=actual_counts)
            return copy.deepcopy(record)
        def get(self, reservation_id):
            return copy.deepcopy(record)
        def fail_consumed(self, reservation_id, reason):
            record.update(status='failed_consumed', reason=reason)
    authority = Authority()
    ticket = SimpleNamespace(path=tmp_path / 'ticket.json', digest='ticket-test', value={
        'lease': lease.to_dict(), 'reservation_id': record['reservation_id'],
        'authority_manifest_hash': 'manifest-test', 'source_hash': 'source-test',
        'context': {'purpose': 'resource_trial', 'cutoff': '2026-10-01T00:00:00+00:00'}, 'recipe': {'test': True}})
    ticket.verify_for_launch = lambda **kwargs: (authority, copy.deepcopy(record), None, None)
    ticket.verify_running_parent = lambda: (authority, copy.deepcopy(record), None, None)
    ticket.path.write_text('{}')
    model = SimpleNamespace(digest='model-test', pipeline_id=fingerprint(ticket.value['recipe']),
                            receipt={'version': 'v7-neural-cold-payload-v2'})
    raw = {'memory.max': str(768 * 2**20), 'memory.high': str(640 * 2**20),
           'memory.swap.max': '0', 'pids.max': '64', 'cpu.max': '25000 100000',
           'memory.peak': str(50 * 2**20), 'memory.events': 'oom 0\noom_kill 0',
           'cgroup.procs': f'{parent["pid"]}\n{child["pid"]}'}
    kernel = {'path': '/synthetic', 'inode': 123, 'boot_id': 'test', 'scope': 'whole_cgroup',
              'platform': 'linux', 'raw': raw, 'peak': 50 * 2**20, 'events': {'oom': 0, 'oom_kill': 0}}
    monkeypatch.setattr(module, '_load_ticket', lambda path: ticket)
    monkeypatch.setattr(module, '_model', lambda path: model)
    monkeypatch.setattr(module, '_verified_counts', lambda authority, reservation_id: counts)
    monkeypatch.setattr(module, '_prepare_publication', lambda ticket: SimpleNamespace(complete=lambda proof: None))
    monkeypatch.setattr(module, '_identity', lambda pid=None: parent if pid is None else child)
    monkeypatch.setattr(module, '_kernel_snapshot', lambda: copy.deepcopy(kernel))
    monkeypatch.setattr(module.importlib.util, 'find_spec', lambda name: object())
    captured = {}
    class Process:
        pid = child['pid']
        returncode = 0
        def __init__(self, command, **kwargs):
            captured.update(command=command, kwargs=kwargs, calls=captured.get('calls', 0) + 1)
            self.written = False
        def poll(self):
            if not self.written:
                assert record['status'] == 'running'
                paths = module._paths(ticket)
                assert module._read(paths['start'])['process'] == child
                paths['model'].write_text(json.dumps({'engineering': True}))
                module._write(paths['membership'], {'version': module.MEMBERSHIP_VERSION,
                    'ticket_digest': ticket.digest, 'process': child, 'parent': parent, 'kernel': kernel})
                module._write(paths['result'], {'version': module.RESULT_VERSION, 'status': 'completed',
                    'ticket_digest': ticket.digest, 'process': child, 'model_digest': model.digest,
                    'lease_hash': lease.digest, 'actual_counts': counts, 'kernel': kernel})
                self.written = True
            return 0
    monkeypatch.setattr(module.subprocess, 'Popen', Process)
    return SimpleNamespace(ticket=ticket, record=record, authority=authority, kernel=kernel,
        captured=captured, model=model, lease=lease, parent=parent, child=child)


def test_fixed_launch_binds_completed_artifacts_and_original_lease(launch_case):
    case = launch_case
    result = module.launch_forward_fit(case.ticket.path)
    receipt = module._read(module._paths(case.ticket)['receipt'])
    assert result['certificate']['payload_digest'] == case.model.digest
    assert receipt['lease_hash'] == case.lease.digest
    assert receipt['completed_record_hash'] == fingerprint(case.record)
    assert case.record['result'] == {'ticket_digest': case.ticket.digest, 'model_digest': case.model.digest,
                                    'worker_result_hash': receipt['worker_result']['sha256']}
    assert case.captured['command'][1:5] == ['-B', '-m', 'factorlab.discovery.neural_forward_launcher', '--ticket']
    assert case.captured['kwargs']['env']['CUDA_VISIBLE_DEVICES'] == ''


def test_successful_ticket_replay_does_not_poison_or_rerun(launch_case):
    module.launch_forward_fit(launch_case.ticket.path)
    with pytest.raises(ValueError, match='already been attempted'):
        module.launch_forward_fit(launch_case.ticket.path)
    assert launch_case.captured['calls'] == 1
    assert not module._paths(launch_case.ticket)['failure'].exists()


def test_missing_real_kernel_closes_before_spawn_and_consumes(launch_case, monkeypatch):
    def unavailable():
        raise ValueError('linux_resource_proof_unavailable')
    monkeypatch.setattr(module, '_kernel_snapshot', unavailable)
    with pytest.raises(ValueError, match='linux_resource'):
        module.launch_forward_fit(launch_case.ticket.path)
    assert launch_case.record['status'] == 'failed_consumed'
    assert not launch_case.captured
    assert module._paths(launch_case.ticket)['failure'].exists()


def test_saved_model_cannot_substitute_another_recipe(launch_case):
    launch_case.model.pipeline_id = 'wrong'
    with pytest.raises(ValueError, match='frozen worker recipe'):
        module.launch_forward_fit(launch_case.ticket.path)
    assert launch_case.record['status'] == 'failed_consumed'


def test_late_certificate_io_leaves_failure_even_after_accounting_complete(launch_case, monkeypatch):
    original = module._write
    def write(path, value):
        original(path, value)
        if path.name == 'certificate.json':
            def closed(*args, **kwargs):
                raise LeaseClosed(LeaseCheck('deadline'))
            monkeypatch.setattr(FrozenWorkLease, 'assert_parent_alive', closed)
    monkeypatch.setattr(module, '_write', write)
    with pytest.raises(LeaseClosed):
        module.launch_forward_fit(launch_case.ticket.path)
    assert launch_case.record['status'] == 'completed'
    assert module._paths(launch_case.ticket)['failure'].exists()


@pytest.mark.parametrize('stage', ['reload', 'publication'])
def test_parent_verification_and_publication_peak_are_part_of_resource_gate(launch_case, monkeypatch, stage):
    def spike():
        launch_case.kernel['peak'] = 601 * 2**20
        launch_case.kernel['raw']['memory.peak'] = str(launch_case.kernel['peak'])
    if stage == 'reload':
        def model(path):
            spike()
            return launch_case.model
        monkeypatch.setattr(module, '_model', model)
    else:
        original = module._write
        def write(path, value):
            original(path, value)
            if path.name == 'certificate.json':
                spike()
        monkeypatch.setattr(module, '_write', write)
    with pytest.raises(ValueError, match='peak|memory|resource|headroom'):
        module.launch_forward_fit(launch_case.ticket.path)
    assert module._paths(launch_case.ticket)['failure'].exists()


def test_kernel_oom_rejected_before_worker(launch_case):
    launch_case.kernel['events']['oom'] = 1
    with pytest.raises(ValueError):
        module.launch_forward_fit(launch_case.ticket.path)
    assert not launch_case.captured


def test_environment_drops_credentials_and_import_redirection(monkeypatch):
    for key in ('OPENAI_API_KEY', 'MY_TOKEN', 'PYTHONPATH', 'PYTHONHOME'):
        monkeypatch.setenv(key, 'never-copy')
    env = module._environment()
    assert not {'OPENAI_API_KEY', 'MY_TOKEN', 'PYTHONPATH', 'PYTHONHOME'} & set(env)


def test_cli_rejects_caller_command_or_mode():
    with pytest.raises(SystemExit):
        module.main(['--ticket', 'unused', '--command', 'anything'])
    with pytest.raises(SystemExit):
        module.main(['--ticket', 'unused', '--mode', 'forward'])


def test_test_only_model_cannot_be_presented_as_full_production_payload(launch_case):
    launch_case.model.receipt['version'] = 'v7-neural-cold-fit-v1'
    with pytest.raises(ValueError, match='frozen worker recipe'):
        module.launch_forward_fit(launch_case.ticket.path)
    assert launch_case.record['status'] == 'failed_consumed'


def test_worker_cannot_self_report_lower_than_durable_costs(launch_case, monkeypatch):
    monkeypatch.setattr(module, '_verified_counts', lambda *args: {'solve': 99})
    with pytest.raises(ValueError, match='durable prepaid accounting'):
        module.launch_forward_fit(launch_case.ticket.path)
    assert launch_case.record['status'] == 'failed_consumed'


def test_worker_watchdog_refuses_in_process_issuer_fixture(launch_case):
    # This executes the real guard, which must reject before starting any exit
    # thread; pytest is the issuer, not its registered direct child.
    with pytest.raises(ValueError, match='direct-worker only'):
        module.worker(launch_case.ticket.path)
    assert launch_case.record['status'] == 'reserved'


def test_failed_child_exit_is_consumed_without_model_or_receipt(launch_case, monkeypatch):
    class FailedProcess:
        pid = launch_case.child['pid']
        returncode = 19
        def __init__(self, *args, **kwargs):
            pass
        def poll(self):
            return 19
    monkeypatch.setattr(module.subprocess, 'Popen', FailedProcess)
    with pytest.raises(RuntimeError, match='exited 19'):
        module.launch_forward_fit(launch_case.ticket.path)
    assert launch_case.record['status'] == 'failed_consumed'
    assert not module._paths(launch_case.ticket)['receipt'].exists()


def test_registered_child_substitution_is_rejected(launch_case, monkeypatch):
    original = launch_case.ticket.verify_running_parent
    def verify():
        launch_case.record['process_identity'] = {'pid': 1, 'created': 1.}
        return original()
    launch_case.ticket.verify_running_parent = verify
    with pytest.raises(ValueError, match='exact membership identity'):
        module.launch_forward_fit(launch_case.ticket.path)


def test_parent_tears_down_only_owned_process_on_original_deadline(launch_case, monkeypatch):
    stopped = []
    class RunningProcess:
        pid = launch_case.child['pid']
        returncode = None
        def __init__(self, *args, **kwargs):
            pass
        def poll(self):
            return None
    monkeypatch.setattr(module.subprocess, 'Popen', RunningProcess)
    monkeypatch.setattr(module, 'terminate_process_tree', lambda process: stopped.append(process.pid))
    real_start = launch_case.authority.start
    def start(*args):
        result = real_start(*args)
        def closed(*args, **kwargs):
            raise LeaseClosed(LeaseCheck('deadline'))
        monkeypatch.setattr(FrozenWorkLease, 'assert_parent_alive', closed)
        return result
    launch_case.authority.start = start
    with pytest.raises(LeaseClosed):
        module.launch_forward_fit(launch_case.ticket.path)
    assert stopped == [launch_case.child['pid']]
    assert launch_case.record['status'] == 'failed_consumed'


def _direct_worker_fixture(root, ticket_value):
    """Actual child/lease/accounting; synthetic numerics and kernel, test only."""
    from factorlab.discovery import neural_forward
    from factorlab.discovery.neural_forward_ticket import open_existing_authority
    patch = pytest.MonkeyPatch()
    patch.setattr('factorlab.protocol.ROOT', Path(root))
    authority = open_existing_authority(Path(root) / 'authority')
    lease = FrozenWorkLease.from_dict(ticket_value['lease'])
    ticket = SimpleNamespace(path=Path(root) / 'ticket.json', digest=fingerprint(ticket_value), value=ticket_value)
    def verify(**kwargs):
        assert kwargs == {'as_worker': True}
        lease.assert_parent_alive(as_worker=True)
        record = authority.get(ticket_value['reservation_id'])
        assert record['status'] == 'running' and record['process_identity']['pid'] == os.getpid()
        return authority, record, None, None
    ticket.verify_for_launch = verify
    patch.setattr(module, '_load_ticket', lambda path: ticket)
    model = SimpleNamespace(digest='synthetic-model')
    model.save = lambda path: path.write_text('{"synthetic":true}')
    def fit(data, cutoff, recipe, budget):
        budget.begin()
        budget.consume('solve')
        return model
    patch.setattr(neural_forward, '_fit_neural_forward_payload', fit)
    patch.setattr(module, '_model', lambda path: model)
    raw = {'memory.max': str(768 * 2**20), 'memory.high': str(640 * 2**20),
        'memory.swap.max': '0', 'pids.max': '64', 'cpu.max': '25000 100000',
        'memory.peak': str(50 * 2**20), 'memory.events': 'oom 0\noom_kill 0',
        'cgroup.procs': f'{lease.parent_pid}\n{os.getpid()}'}
    kernel = {'path': '/synthetic', 'inode': 123, 'boot_id': 'test', 'scope': 'whole_cgroup',
              'platform': 'linux', 'raw': raw, 'peak': 50 * 2**20, 'events': {'oom': 0, 'oom_kill': 0}}
    patch.setattr(module, '_kernel_snapshot', lambda: kernel)
    module.worker(ticket.path)


def test_actual_direct_child_uses_original_lease_and_durable_once_accounting(tmp_path, monkeypatch):
    from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
    from factorlab.discovery.reserved_work_budget import ReservedChannelBudget
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    (tmp_path / 'factorlab').mkdir()
    (tmp_path / 'factorlab/frozen.py').write_text('test_only=1')
    (tmp_path / 'pyproject.toml').write_text('[project]')
    (tmp_path / 'uv.lock').write_text('version=1')
    authority = MonthlyWorkAuthority(tmp_path / 'authority', run_id='engineering', task='volatility', universe=['X'])
    reservation = authority.reserve('synthetic-direct-worker', counts={'solve': 1}, binding={'test_only': True}, lease_seconds=30)
    value = {'lease': reservation['lease'], 'reservation_id': reservation['reservation_id'],
             'context': {'cutoff': '2025-04-01T00:00:00+00:00'}}
    (tmp_path / 'ticket.json').write_text(json.dumps(value))
    (tmp_path / 'launch').mkdir()
    child = multiprocessing.get_context('spawn').Process(target=_direct_worker_fixture, args=(str(tmp_path), value))
    child.start()
    try:
        identity = module._identity(child.pid)
        authority.start(reservation['reservation_id'], identity)
        module._write(tmp_path / 'launch/start.json', {'ticket_digest': fingerprint(value),
            'process': identity, 'lease_hash': reservation['lease_hash']})
        child.join(15)
        assert child.exitcode == 0
        actual = ReservedChannelBudget.verify_counts(authority, reservation['reservation_id'])
        assert actual['actual_counts']['solve'] == 1 and actual['lease_hash'] == reservation['lease_hash']
        result = module._read(tmp_path / 'launch/result.json')
        assert result['process'] == identity and result['actual_counts'] == actual['actual_counts']
        assert authority.get(reservation['reservation_id'])['status'] == 'running'
        assert time.monotonic() < reservation['lease']['monotonic_deadline']
    finally:
        if child.is_alive():
            child.terminate()
            child.join(5)


@pytest.fixture
def real_ticket_case(tmp_path, monkeypatch):
    """Real receipt archive/job/payload/authority; pre-resource gate is synthetic."""
    import pandas as pd
    from test_v7_feed_audit import Clock
    from test_v7_rolling_registry import candles
    from test_v7_run_factor_pool import source_root

    from factorlab.discovery import neural_forward_ticket as tickets
    from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
    from factorlab.discovery.neural_forward import FrozenNeuralRecipe
    from factorlab.discovery.v7_forward import ForwardRuntime

    source_root(tmp_path, monkeypatch)
    clock = Clock('2026-09-30T23:30:00Z')
    runtime = ForwardRuntime(tmp_path / 'run', tmp_path / 'feed', clock=clock)
    try:
        path = tmp_path / 'artificial.parquet'
        candles('2026-05-01T00:00Z', '2026-09-30T23:00Z').to_parquet(path)
        runtime.feed.import_seed(path)
        clock.now = pd.Timestamp('2026-10-01T00:00:10Z')
        recipe = FrozenNeuralRecipe(rolling_spec=runtime.spec)
        job = runtime.lifecycle.plan_fit('2026-10-01T00:00:00Z', 'A0', runtime.recipe_for(recipe))
        authority = MonthlyWorkAuthority(tmp_path / 'authority', run_id='full-launch-fixture',
                                        task='volatility', universe=list(runtime.config.symbols))
        # No real quarterly/resource evidence exists in this artificial test.
        monkeypatch.setattr(tickets, '_verify_gates', lambda *args, **kwargs: None)
        ticket = tickets.issue_forward_ticket(runtime, authority, job, recipe,
                                             purpose='resource_trial', gates={})
        yield SimpleNamespace(ticket=ticket, authority=authority, runtime=runtime, recipe=recipe)
    finally:
        runtime.close()


def test_real_ticket_parent_stage_must_not_reuse_reserved_only_verifier(real_ticket_case):
    case = real_ticket_case
    authority, original, data, recipe = case.ticket.verify_for_launch()
    assert recipe.pipeline_id == case.recipe.pipeline_id and len(data.frame.values) > 0
    identity = {'pid': os.getpid() + 100000, 'created': time.time()}
    authority.start(original['reservation_id'], identity)
    with pytest.raises(ValueError, match='consumed|launchable'):
        case.ticket.verify_for_launch()
    _, running, repeated, _ = case.ticket.verify_running_parent()
    assert running['status'] == 'running' and running['lease'] == original['lease']
    assert repeated.frame.schema_hash == data.frame.schema_hash


def test_real_ticket_launcher_preflight_failure_keeps_prepaid_costs(real_ticket_case, monkeypatch):
    case = real_ticket_case
    reserved = case.authority.get(case.ticket.value['reservation_id'])
    before = case.authority.status(reserved['actual_month'])
    def unavailable():
        raise ValueError('synthetic-host-no-real-Linux-proof')
    monkeypatch.setattr(module, '_kernel_snapshot', unavailable)
    with pytest.raises(ValueError, match='no-real-Linux-proof'):
        module.launch_forward_fit(case.ticket.path)
    after = case.authority.status(reserved['actual_month'])
    assert after['counts'] == before['counts']
    assert case.authority.get(reserved['reservation_id'])['status'] == 'failed_consumed'
    assert module._paths(case.ticket)['failure'].exists()


def _launch_artificial_full(case, monkeypatch):
    ticket, authority = case.ticket, case.authority
    reservation = authority.get(ticket.value['reservation_id'])
    lease = FrozenWorkLease.from_dict(reservation['lease'])
    parent = {'pid': lease.parent_pid, 'created': lease.parent_created}
    child = {'pid': os.getpid() + 100000, 'created': time.time()}
    counts = reservation['counts']
    model = SimpleNamespace(digest=fingerprint('artificial-full-result'), pipeline_id=case.recipe.pipeline_id,
                            receipt={'version': 'v7-neural-cold-payload-v2'})
    raw = {'memory.max': str(768 * 2**20), 'memory.high': str(640 * 2**20),
        'memory.swap.max': '0', 'pids.max': '64', 'cpu.max': '25000 100000',
        'memory.peak': str(50 * 2**20), 'memory.events': 'oom 0\noom_kill 0',
        'cgroup.procs': f'{parent["pid"]}\n{child["pid"]}'}
    kernel = {'path': '/synthetic', 'inode': 123, 'boot_id': 'test', 'scope': 'whole_cgroup',
              'platform': 'linux', 'raw': raw, 'peak': 50 * 2**20, 'events': {'oom': 0, 'oom_kill': 0}}
    monkeypatch.setattr(module, '_kernel_snapshot', lambda: kernel)
    monkeypatch.setattr(module, '_identity', lambda pid=None: parent if pid is None else child)
    monkeypatch.setattr(module, '_model', lambda path: model)
    monkeypatch.setattr(module, '_verified_counts', lambda *args: counts)
    monkeypatch.setattr(module.importlib.util, 'find_spec', lambda *args: object())
    class CompletedProcess:
        pid = child['pid']
        returncode = 0
        def __init__(self, *args, **kwargs):
            self.written = False
        def poll(self):
            if not self.written:
                assert authority.get(reservation['reservation_id'])['status'] == 'running'
                paths = module._paths(ticket)
                paths['model'].write_text('{"artificial":true}')
                module._write(paths['membership'], {'version': module.MEMBERSHIP_VERSION,
                    'ticket_digest': ticket.digest, 'process': child, 'parent': parent, 'kernel': kernel})
                module._write(paths['result'], {'version': module.RESULT_VERSION, 'status': 'completed',
                    'ticket_digest': ticket.digest, 'process': child, 'model_digest': model.digest,
                    'lease_hash': lease.digest, 'actual_counts': counts, 'kernel': kernel})
                self.written = True
            return 0
    monkeypatch.setattr(module.subprocess, 'Popen', CompletedProcess)
    result = module.launch_forward_fit(ticket.path)
    return result, reservation


def test_real_ticket_launcher_completion_uses_running_parent_verification(real_ticket_case, monkeypatch):
    case = real_ticket_case
    result, reservation = _launch_artificial_full(case, monkeypatch)
    authority, ticket = case.authority, case.ticket
    completed = authority.get(reservation['reservation_id'])
    assert result['status'] == completed['status'] == 'completed'
    assert completed['lease'] == reservation['lease'] and completed['actual_counts'] == reservation['counts']
    assert not module._paths(ticket)['failure'].exists()


def test_real_publication_remains_closed_when_final_lease_and_failure_log_fail(real_ticket_case, monkeypatch):
    from factorlab.discovery.forward_publication_witness import ForwardPublicationWitness
    original_prepare, original_write = module._prepare_publication, module._write
    original_assert = FrozenWorkLease.assert_parent_alive
    checks = []
    def prepare(ticket):
        actual = original_prepare(ticket)
        def complete(proof):
            def closed(lease, *args, **kwargs):
                checks.append('check')
                if len(checks) == 2:  # SQL UPDATE happened; final pre-COMMIT check fails.
                    raise LeaseClosed(LeaseCheck('deadline'))
                return original_assert(lease, *args, **kwargs)
            monkeypatch.setattr(FrozenWorkLease, 'assert_parent_alive', closed)
            return actual.complete(proof)
        return SimpleNamespace(complete=complete)
    def write(path, value):
        if path.name == 'failure.json':
            raise OSError('Synthetic failure evidence filesystem denial')
        return original_write(path, value)
    monkeypatch.setattr(module, '_prepare_publication', prepare)
    monkeypatch.setattr(module, '_write', write)
    with pytest.raises(LeaseClosed):
        _launch_artificial_full(real_ticket_case, monkeypatch)
    ticket = real_ticket_case.ticket
    assert (ticket.path.parent / 'launch/certificate.json').exists()
    assert not module._paths(ticket)['failure'].exists()
    assert real_ticket_case.authority.get(ticket.value['reservation_id'])['status'] == 'completed'
    assert len(checks) == 2
    with pytest.raises(ValueError):
        ForwardPublicationWitness.read_completed(ticket)


def test_real_publication_success_binds_actual_final_proof(real_ticket_case, monkeypatch):
    from factorlab.discovery.forward_publication_witness import ForwardPublicationWitness
    result, reservation = _launch_artificial_full(real_ticket_case, monkeypatch)
    proof = ForwardPublicationWitness.read_completed(real_ticket_case.ticket)
    assert proof['certificate_digest'] == fingerprint(result['certificate'])
    assert proof['receipt_hash'] == result['receipt']['sha256']
    assert proof['actual_counts_hash'] == fingerprint(reservation['counts'])
