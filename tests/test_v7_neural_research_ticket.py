"""Artificial real-wallet S ticket tests; private gate seam is NOT gate evidence."""
import copy
import json
import multiprocessing
import os
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_v7_nonlinear_channel import artificial_data

from factorlab.discovery import neural_research_launcher as launcher
from factorlab.discovery import neural_research_ticket as module
from factorlab.discovery.joint_pool import FrozenPool
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
from factorlab.discovery.neural_forward import FrozenNeuralRecipe
from factorlab.discovery.nonlinear_channel import ResidualConfig
from factorlab.discovery.nonlinear_orchestration import input_data_identity
from factorlab.discovery.v7_memory import search_exposure_context
from factorlab.discovery.v7_run import RunStore, V7Config, monthly_budget
from factorlab.discovery.work_authority_bridge import WorkAuthorityBridge
from factorlab.discovery.work_lease import FrozenWorkLease


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    (tmp_path / 'factorlab').mkdir()
    (tmp_path / 'factorlab/frozen.py').write_text('source=1')
    (tmp_path / 'pyproject.toml').write_text('[project]')
    (tmp_path / 'uv.lock').write_text('version=1')
    data = artificial_data()
    config = V7Config(evaluation_mode='joint', symbols=('X',))
    store = RunStore(tmp_path / 'research', config, data.spec)
    authority = MonthlyWorkAuthority(tmp_path / 'wallet', run_id='campaign', task='volatility', universe=['X'], partition_policy='standard')
    month, cutoff = '2025-04', '2025-04-01T00:00:00+00:00'
    exposed = search_exposure_context('controller-data', cutoff, ['X'])
    store.exposure.record(**{k: v for k, v in exposed.items() if k != 'data_id'}, role='S')
    state = {'status': 'running', 'cutoff': cutoff, 'data_id': 'controller-data',
        'search_exposure': exposed, 'budget': monthly_budget(config, month).to_dict(),
        'frozen_reference_pool': FrozenPool().to_dict(),
        'nonlinear_input_data_id': input_data_identity(data, cutoff)}
    store.save(month, state, 'frozen_controller_input')

    def persist(value):
        state['budget'] = value
        store.save(month, state, 'budget_projection')

    bridge = WorkAuthorityBridge(store, month, state, authority, persist)
    recipe = FrozenNeuralRecipe(rolling_spec=data.spec, config=ResidualConfig(enabled=True, epochs=4))
    # This is the newly declared 4-epoch complete recipe, not the former
    # 10-epoch learner or inherited performance evidence. Its full three folds
    # are retained and the shared default allocation is explicit.
    from factorlab.discovery.monthly_schedule import default_schedule
    from factorlab.discovery.s_nonlinear_ticket import freeze_nonlinear_plan
    from factorlab.discovery.scheduled_search import ScheduledSearch
    ScheduledSearch(bridge, default_schedule(config, month=month))
    freeze_nonlinear_plan(bridge, recipe, kinds=('quarterly_probe', 'resource_trial', 'complete_neural_S'))
    # Explicit prepaid artificial baseline establishes the genuine S owner; no fit claimed.
    bridge.begin_work('baseline', counts={'solve': 7}, lease_seconds=300,
        binding={'stage': 'baseline'}).complete({'artificial_owner_setup': True, 'training_performed': False})
    gates = {'directory': str(tmp_path / 'artificial-gate-placeholder'),
             'artifacts': {'protocol': 'p', 'probes': ['q1', 'q2'], 'positive_control': 'c'}}
    monkeypatch.setattr(module, '_gates', lambda *args: {'private_test_seam': True})
    yield bridge, data, recipe, gates
    store.close()


def issue(case):
    b, data, recipe, gates = case
    return module.issue_research_ticket(b, data, recipe, gates=gates)


def test_true_ticket_prepays_all_three_folds_once_and_binds_actual_controller(case):
    ticket = issue(case)
    authority, record, data, recipe = ticket.verify()
    assert record['counts']['neural_fit'] == 9
    assert record['counts']['epoch'] == 36
    assert record['lease']['lease_seconds'] == 300
    assert record['actual_month'] != ticket.value['context']['replay_month']
    assert ticket.value['context']['serving_eligible'] is False
    assert recipe.pipeline_id == case[2].pipeline_id
    assert input_data_identity(data, case[0].state['cutoff']) == case[0].state['nonlinear_input_data_id']
    assert issue(case).digest == ticket.digest
    assert authority.verify()['reservations'] == 2


def test_current_S_deadline_cannot_be_extended_for_neural_work(case):
    # The immutable envelope cannot be edited; a smaller original one must be
    # created at construction rather than granting a new long lease here.
    b = case[0]
    b.identity['execution_envelope']['monotonic_deadline'] -= 7199
    with pytest.raises(ValueError, match='envelope changed'):
        issue(case)
    assert b.authority.verify()['reservations'] == 1


def test_changed_payload_rejected_after_original_prepaid_input_work(case):
    b, data, recipe, gates = case
    target = data.target.copy()
    target[0] += .5
    with pytest.raises(ValueError, match='checkpointed'):
        module.issue_research_ticket(b, replace(data, target=target), recipe, gates=gates)
    assert b.authority.verify()['reservations'] == 2
    with b.authority._transaction() as db:
        record = next(r for r in b.authority._verify(db)['reservations'].values()
                      if r['work_key'].startswith('neural_S:'))
    assert record['status'] == 'failed_consumed'


def test_missing_controller_seal_or_wrong_Pv_cannot_claim_self_reported_input(case):
    b = case[0]
    del b.state['nonlinear_input_data_id']
    b.store.save(b.month, b.state, 'removed_seal')
    with pytest.raises(ValueError, match='actual running controller'):
        issue(case)


def test_payload_serialization_crash_keeps_full_prepaid_cost_and_no_new_lease(case, monkeypatch):
    monkeypatch.setattr(module, 'save_worker_data', lambda *args, **kwargs: (_ for _ in ()).throw(OSError('write failed')))
    with pytest.raises(OSError, match='write failed'):
        issue(case)
    authority = case[0].authority
    assert authority.verify()['reservations'] == 2
    with pytest.raises(ValueError, match='Partial S preparation'):
        issue(case)
    assert authority.verify()['reservations'] == 2


def test_payload_tamper_is_rejected_by_real_reload(case):
    ticket = issue(case)
    Path(ticket.value['data']['arrays']).write_bytes(b'damaged')
    with pytest.raises(ValueError, match='worker data changed'):
        ticket.verify()


def test_actual_same_batch_V_exposure_closes_S_launch(case):
    ticket = issue(case)
    exposed = case[0].state['search_exposure']
    case[0].store.exposure.record(**{k: v for k, v in exposed.items() if k != 'data_id'}, role='V')
    with pytest.raises(ValueError):
        ticket.verify()


def test_windows_resource_unavailable_consumes_attempt_without_spawning(case, monkeypatch):
    ticket = issue(case)
    monkeypatch.setattr(launcher, '_kernel_snapshot', lambda: (_ for _ in ()).throw(ValueError('linux unavailable')))
    monkeypatch.setattr(launcher.subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('must not spawn'))
    with pytest.raises(ValueError, match='linux unavailable'):
        launcher.launch_research(ticket.path)
    assert case[0].authority.get(ticket.value['reservation_id'])['status'] == 'failed_consumed'
    with pytest.raises(ValueError, match='already attempted'):
        launcher.launch_research(ticket.path)
    with pytest.raises(ValueError, match='required canonical state'):
        launcher.verify_completed_research(ticket.path)


def test_canonical_registered_child_required_even_for_original_issuer(case):
    ticket = issue(case)
    case[0].authority.start(ticket.value['reservation_id'], {'pid': 123, 'created': 1.})
    with pytest.raises(RuntimeError, match='wrong_process'):
        ticket.verify(mode='worker')
    ticket.verify(mode='running_parent')


def test_publication_final_SQL_check_failure_rolls_back_without_failure_log(case, monkeypatch):
    ticket = issue(case)
    publication = launcher._Publication(ticket, prepare=True)
    proof = {'receipt_hash': 'a' * 64, 'result_hash': 'b' * 64, 'actual_counts_hash': 'c' * 64,
             'kernel': {'synthetic': True}, 'observation': {'synthetic': True}}
    monkeypatch.setattr(FrozenWorkLease, 'assert_parent_alive',
                        lambda self, **kwargs: (_ for _ in ()).throw(RuntimeError('final deadline')))
    with pytest.raises(RuntimeError, match='final deadline'):
        publication.complete(proof)
    assert publication._verify()[2] == 'pending'
    with pytest.raises(ValueError, match='no committed'):
        publication.read_completed()


@pytest.mark.parametrize('path', ['publication.sqlite', 'identity.json', 'anchor'])
def test_missing_publication_evidence_never_resets(case, path):
    ticket = issue(case)
    publication = launcher._Publication(ticket, prepare=True)
    target = publication.anchor if path == 'anchor' else publication.directory / path
    target.unlink()
    with pytest.raises((ValueError, OSError)):
        launcher._Publication(ticket)
    with pytest.raises(ValueError, match='already prepared'):
        launcher._Publication(ticket, prepare=True)


def test_S_cannot_be_loaded_as_forward_ticket(case):
    from factorlab.discovery.neural_forward_ticket import ForwardFitTicket
    ticket = issue(case)
    with pytest.raises(ValueError, match='forward ticket schema'):
        ForwardFitTicket.load(ticket.path)


def test_forged_result_receipt_cannot_bypass_pending_witness(case):
    ticket = issue(case)
    publication = launcher._Publication(ticket, prepare=True)
    (publication.directory.parent / 'receipt.json').write_text(json.dumps({'fake': 'success'}))
    with pytest.raises(ValueError, match='no committed'):
        publication.read_completed()


def _synthetic_kernel(pids):
    # Private engineering substitute; never actual Linux/cgroup acceptance.
    raw = {'memory.max': str(768 * 2**20), 'memory.high': str(640 * 2**20),
           'memory.swap.max': '0', 'pids.max': '64', 'cpu.max': '25000 100000',
           'memory.peak': str(50 * 2**20), 'memory.events': 'oom 0\noom_kill 0',
           'cgroup.procs': '\n'.join(str(p) for p in pids)}
    return {'path': '/synthetic', 'inode': 123, 'boot_id': 'synthetic', 'scope': 'whole_cgroup',
            'platform': 'linux', 'raw': raw, 'peak': 50 * 2**20, 'events': {'oom': 0, 'oom_kill': 0},
            'observed_at': datetime.now(UTC).isoformat()}


def _actual_child(root, ticket_path):
    import factorlab.protocol
    factorlab.protocol.ROOT = Path(root)
    module._gates = lambda *args: {'private_test_seam': True}
    launcher._kernel_snapshot = lambda: _synthetic_kernel([os.getppid(), os.getpid()])
    launcher.worker(ticket_path)


@pytest.mark.parametrize('publication_failure', [False, True])
def test_true_registered_child_runs_three_models_parent_recomputes_and_S_only(case, monkeypatch, publication_failure):
    b, data, recipe, gates = case
    ticket = module.issue_research_ticket(b, data, recipe, gates=gates)
    processes = []

    class ActualChild:
        def __init__(self, command, **kwargs):
            assert command[1:5] == ['-B', '-m', 'factorlab.discovery.neural_research_launcher', '--ticket']
            assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
            self.inner = multiprocessing.get_context('spawn').Process(target=_actual_child,
                args=(str(b.store.directory.parent), str(ticket.path)))
            self.inner.start()
            self.pid = self.inner.pid
            processes.append(self)

        @property
        def returncode(self):
            return self.inner.exitcode

        def poll(self):
            return self.inner.exitcode

        def wait(self, timeout=None):
            self.inner.join(timeout)
            return self.inner.exitcode

    monkeypatch.setattr(launcher.subprocess, 'Popen', ActualChild)
    monkeypatch.setattr(launcher, '_kernel_snapshot', lambda: _synthetic_kernel(
        [os.getpid()] + [p.pid for p in processes if p.poll() is None]))
    monkeypatch.setattr(launcher, '_strict_kernel', lambda *args: None)
    began = time.monotonic()
    if publication_failure:
        original_complete, original_write = launcher._Publication.complete, launcher._write

        def final_check_fails(self, proof):
            with monkeypatch.context() as patch:
                patch.setattr(FrozenWorkLease, 'assert_parent_alive',
                    lambda self, **kwargs: (_ for _ in ()).throw(RuntimeError('after SQL update')))
                original_complete(self, proof)

        def failure_log_unwritable(path, value):
            if path.name == 'failure.json':
                raise OSError('failure log unavailable')
            return original_write(path, value)

        monkeypatch.setattr(launcher._Publication, 'complete', final_check_fails)
        monkeypatch.setattr(launcher, '_write', failure_log_unwritable)
        with pytest.raises(RuntimeError, match='after SQL update'):
            launcher.launch_research(ticket.path)
        assert b.authority.get(ticket.value['reservation_id'])['status'] == 'completed'
        assert not launcher._paths(ticket)['failure'].exists()
        with pytest.raises(ValueError, match='no committed'):
            launcher.verify_completed_research(ticket.path)
        return
    response = launcher.launch_research(ticket.path)
    assert time.monotonic() - began < 300
    assert processes[0].returncode == 0
    result = launcher.verify_completed_research(ticket.path)
    assert response['role'] == result['role'] == 'S'
    assert not response['serving_eligible'] and not response['formal_claim']
    assert result['reservation']['actual_counts']['neural_fit'] == 9
    assert result['reservation']['actual_counts']['epoch'] == 36
    assert len(result['result']['folds']) == 3
    assert all(f['status'] == 'completed' for f in result['result']['folds'])
    with monkeypatch.context() as patch:
        patch.setattr(FrozenWorkLease, 'assert_parent_alive',
            lambda self, **kwargs: (_ for _ in ()).throw(RuntimeError('old parent lost')))
        assert launcher.verify_completed_research(ticket.path)['role'] == 'S'
    saved = copy.deepcopy(result['result'])
    saved['folds'][0]['hourly'][0]['losses']['ensemble'] += 1
    path = Path(response['research']['path'])
    path.write_text(json.dumps(saved), encoding='utf-8')
    with pytest.raises(ValueError, match='Parent-recomputed'):
        launcher._research_result(ticket, data, recipe)
