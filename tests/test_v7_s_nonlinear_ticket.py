"""Real wallet/lease/RunStore tests; no Linux or market resource evidence."""
import importlib.util
import json
import multiprocessing
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from test_v7_neural_research_ticket import case as original_case  # noqa: F401 -- pytest fixture

from factorlab.discovery import s_nonlinear_launcher as launcher
from factorlab.discovery import s_nonlinear_ticket as module
from factorlab.discovery.monthly_schedule import default_schedule
from factorlab.discovery.scheduled_search import ScheduledSearch
from factorlab.discovery.v7_run import V7Config
from factorlab.discovery.work_lease import FrozenWorkLease


@pytest.fixture
def case(original_case, monkeypatch):  # noqa: F811 -- pytest fixture injection
    bridge, data, recipe, gates = original_case
    recipe = replace(recipe, config=replace(recipe.config, epochs=4))
    config = V7Config(**{**bridge.store.identity['config'], 'symbols': tuple(bridge.store.identity['config']['symbols'])})
    schedule = default_schedule(config, month=bridge.month)
    ScheduledSearch(bridge, schedule)
    monkeypatch.setattr(module, '_gates', lambda *args: {'private_engineering_test_seam': True})
    module.freeze_nonlinear_plan(bridge, recipe, kinds=('quarterly_probe', 'resource_trial', 'complete_neural_S'))
    return bridge, data, recipe, gates


def issue(case, kind='quarterly_probe'):
    bridge, data, recipe, gates = case
    return module.issue_nonlinear_ticket(bridge, data, recipe, kind=kind,
                                         gates=gates if kind == 'resource_trial' else None)


def test_quarterly_original_prepaid_tree_and_positive_counts(case):
    ticket = issue(case)
    _, record, _, recipe = ticket.verify()
    assert record['counts']['tree_fit'] == 3
    assert record['counts']['solve'] == 9
    assert record['counts']['neural_fit'] == 0
    assert record['lease']['lease_seconds'] == 300
    assert recipe.pipeline_id == case[2].pipeline_id
    with pytest.raises(ValueError, match='no retry'):
        issue(case)
    assert case[0].authority.verify()['reservations'] == 2


def test_resource_trial_uses_same_wallet_without_A0_job(case):
    ticket = issue(case, 'resource_trial')
    _, record, _, _ = ticket.verify()
    assert record['counts']['neural_fit'] == 3
    assert record['counts']['epoch'] == 12
    assert record['counts']['solve'] == 7
    assert record['binding']['role'] == 'S'
    assert record['binding']['serving_eligible'] is False
    assert 'job' not in ticket.value['context']


def test_input_mismatch_is_charged_before_numerical_reconstruction(case):
    bridge, data, recipe, _ = case
    changed = data.target.copy()
    changed[0] += .5
    with pytest.raises(ValueError, match='checkpointed'):
        module.issue_nonlinear_ticket(bridge, replace(data, target=changed), recipe, kind='quarterly_probe')
    assert bridge.authority.verify()['reservations'] == 2
    intent = json.loads(next((bridge.authority.directory / 'S_nonlinear_tickets').glob('*/preparation.json')).read_text())
    assert bridge.authority.get(intent['reservation_id'])['status'] == 'failed_consumed'
    with pytest.raises(ValueError, match='no retry'):
        issue(case)


def test_payload_failure_does_not_refund_or_issue_replacement(case, monkeypatch):
    monkeypatch.setattr(module, 'save_worker_data', lambda *a, **k: (_ for _ in ()).throw(OSError('disk')))
    with pytest.raises(OSError, match='disk'):
        issue(case)
    assert case[0].authority.verify()['reservations'] == 2
    with pytest.raises(ValueError, match='no retry'):
        issue(case)


def test_missing_plan_anchor_and_V_exposure_fail_closed(case):
    ticket = issue(case)
    path = case[0].anchor / 'nonlinear_plan.json'
    path.rename(path.with_suffix('.removed'))
    with pytest.raises(ValueError, match='Missing'):
        ticket.verify()
    path.with_suffix('.removed').rename(path)
    exposure = case[0].state['search_exposure']
    case[0].store.exposure.record(**{k: v for k, v in exposure.items() if k != 'data_id'}, role='V')
    with pytest.raises(ValueError):
        ticket.verify()


def test_missing_true_linux_boundary_consumes_without_spawn(case, monkeypatch):
    ticket = issue(case)
    monkeypatch.setattr(launcher, '_kernel_snapshot', lambda: (_ for _ in ()).throw(ValueError('linux unavailable')))
    monkeypatch.setattr(launcher.subprocess, 'Popen', lambda *a, **k: pytest.fail('No spawn'))
    with pytest.raises(ValueError, match='linux unavailable'):
        launcher.launch_nonlinear(ticket.path)
    assert case[0].authority.get(ticket.value['reservation_id'])['status'] == 'failed_consumed'
    with pytest.raises(ValueError, match='already attempted'):
        launcher.launch_nonlinear(ticket.path)


def test_publication_final_deadline_failure_keeps_pending(case, monkeypatch):
    ticket = issue(case)
    publication = launcher._Publication(ticket, prepare=True)
    monkeypatch.setattr(FrozenWorkLease, 'assert_parent_alive',
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError('deadline')))
    proof = {'receipt_hash': 'a'*64, 'result_hash': 'b'*64, 'actual_counts_hash': 'c'*64,
             'kernel': {'artificial_fixture': True}, 'observation': {'artificial_fixture': True}}
    with pytest.raises(RuntimeError, match='deadline'):
        publication.complete(proof)
    with pytest.raises(ValueError, match='no committed'):
        publication.read_completed()


def test_original_parent_cannot_claim_registered_child_role(case):
    ticket = issue(case)
    case[0].authority.start(ticket.value['reservation_id'], {'pid': 123, 'created': 1.})
    with pytest.raises(RuntimeError, match='wrong_process'):
        ticket.verify(mode='worker')
    ticket.verify(mode='running_parent')


def _actual_fixed_child(root, ticket_path, research=False):
    from test_v7_neural_research_ticket import _synthetic_kernel

    import factorlab.discovery
    import factorlab.protocol
    factorlab.protocol.ROOT = Path(root)
    here = Path(__file__).parent
    for name in ('monthly_schedule', 's_nonlinear_ticket', 's_nonlinear_launcher'):
        full = 'factorlab.discovery.' + name
        # Staging overlays live beside these tests; after promotion import the
        # real package normally. Neither branch alters the production loader.
        if (here / (name + '.py')).is_file():
            spec = importlib.util.spec_from_file_location(full, here / (name + '.py'))
            value = importlib.util.module_from_spec(spec)
            sys.modules[full] = value
            setattr(factorlab.discovery, name, value)
            spec.loader.exec_module(value)
        else:
            importlib.import_module(full)
    if research:
        from factorlab.discovery import neural_research_launcher as fixed
        from factorlab.discovery import neural_research_ticket as rt
        rt._gates = lambda *a: {'private_engineering_test_seam': True}
    else:
        fixed = sys.modules['factorlab.discovery.s_nonlinear_launcher']
    sys.modules['factorlab.discovery.s_nonlinear_ticket']._gates = lambda *a: {'private_engineering_test_seam': True}
    fixed._kernel_snapshot = lambda: _synthetic_kernel([os.getppid(), os.getpid()])
    fixed.worker(ticket_path)


@pytest.mark.parametrize('kind', ['quarterly_probe', 'resource_trial'])
def test_actual_registered_child_fixed_numerics_and_parent_replay(case, monkeypatch, kind):
    from test_v7_neural_research_ticket import _synthetic_kernel
    bridge, data, recipe, _ = case
    ticket = issue(case, kind)
    processes = []

    class ActualChild:
        def __init__(self, command, **kwargs):
            assert command[1:5] == ['-B', '-m', 'factorlab.discovery.s_nonlinear_launcher', '--ticket']
            assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
            self.inner = multiprocessing.get_context('spawn').Process(target=_actual_fixed_child,
                args=(str(bridge.store.directory.parent), str(ticket.path)))
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
    monkeypatch.setattr(launcher, '_strict_kernel', lambda *a: None)
    response = launcher.launch_nonlinear(ticket.path)
    restored = launcher.verify_completed_nonlinear(ticket.path)
    assert response['role'] == restored['role'] == 'S'
    assert restored['serving_eligible'] is False
    counts = restored['reservation']['actual_counts']
    if kind == 'quarterly_probe':
        assert counts['tree_fit'] == 3 and counts['solve'] == 9
        assert counts['neural_fit'] == 0
        assert len(restored['result']['probe']['folds']) == 3
        changed = restored['result']
        changed['probe']['folds'][0]['loss'] += 1
        Path(response['research']['path']).write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            launcher._research_result(ticket, data, recipe)
    else:
        assert counts['neural_fit'] == 3 and counts['epoch'] == 12 and counts['solve'] == 7
        assert restored['result']['model']['digest']


def test_actual_complete_three_fold_S_retains_separate_catalogue_and_shared_wallet(case, monkeypatch):
    from test_v7_neural_research_ticket import _synthetic_kernel

    from factorlab.discovery import neural_research_launcher as full
    from factorlab.discovery import neural_research_ticket as tickets
    from factorlab.discovery.algorithm_shortlist import AlgorithmCatalogue
    bridge, data, recipe, gates = case
    catalogue = AlgorithmCatalogue.freeze(bridge, (recipe,))
    # Explicit private gate seam: this test proves engineering connectivity only.
    monkeypatch.setattr(tickets, '_gates', lambda *a: {'private_engineering_test_seam': True})
    ticket = tickets.issue_research_ticket(bridge, data, recipe, gates=gates)
    processes = []

    class ActualChild:
        def __init__(self, command, **kwargs):
            assert command[1:5] == ['-B', '-m', 'factorlab.discovery.neural_research_launcher', '--ticket']
            self.inner = multiprocessing.get_context('spawn').Process(target=_actual_fixed_child,
                args=(str(bridge.store.directory.parent), str(ticket.path), True))
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
    monkeypatch.setattr(full.subprocess, 'Popen', ActualChild)
    monkeypatch.setattr(full, '_kernel_snapshot', lambda: _synthetic_kernel(
        [os.getpid()] + [p.pid for p in processes if p.poll() is None]))
    monkeypatch.setattr(full, '_strict_kernel', lambda *a: None)
    full.launch_research(ticket.path)
    verified = full.verify_completed_research(ticket.path)
    catalogue.record_ticket(ticket.path)
    shortlist = catalogue.seal_shortlist()
    assert verified['reservation']['actual_counts']['neural_fit'] == 9
    assert verified['reservation']['actual_counts']['epoch'] == 36
    assert verified['reservation']['actual_counts']['solve'] == 24
    assert len(verified['result']['folds']) == 3
    assert shortlist['all_candidates'][0]['status'] == 'completed'
    assert shortlist['serving_eligible'] is False
    assert shortlist['formal_claim'] is False
