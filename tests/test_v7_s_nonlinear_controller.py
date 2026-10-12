"""Controller producer wiring, using actual wallet and registered child numerics."""
import multiprocessing
import os
from pathlib import Path

import pytest
from test_v7_neural_research_ticket import _synthetic_kernel
from test_v7_nonlinear_channel import artificial_data
from test_v7_run_factor_pool import source_root
from test_v7_s_nonlinear_ticket import _actual_fixed_child

from factorlab.discovery import s_nonlinear_controller as controller
from factorlab.discovery import s_nonlinear_launcher as launcher
from factorlab.discovery.joint_pool import FrozenPool
from factorlab.discovery.monthly_schedule import default_schedule
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
from factorlab.discovery.nonlinear_orchestration import input_data_identity
from factorlab.discovery.scheduled_search import ScheduledSearch
from factorlab.discovery.v7_memory import search_exposure_context
from factorlab.discovery.v7_run import RunStore, V7Config, monthly_budget
from factorlab.discovery.work_authority_bridge import WorkAuthorityBridge


@pytest.fixture
def setup(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    stores = []
    def make(month):
        data = artificial_data()
        config = V7Config(evaluation_mode='joint', symbols=('X',), nonlinear_mode='isolated')
        store = RunStore(tmp_path / month, config, data.spec)
        stores.append(store)
        authority = MonthlyWorkAuthority(tmp_path / 'wallet', run_id='actual-controller-test', task='volatility', universe=['X'], partition_policy='standard')
        cutoff = month + '-01T00:00:00+00:00'
        exposed = search_exposure_context('controller-data', cutoff, ['X'])
        store.exposure.record(**{k: v for k, v in exposed.items() if k != 'data_id'}, role='S')
        state = {'status': 'running', 'cutoff': cutoff, 'data_id': 'controller-data', 'search_exposure': exposed,
                 'budget': monthly_budget(config, month).to_dict(), 'frozen_reference_pool': FrozenPool().to_dict(),
                 'nonlinear_input_data_id': input_data_identity(data, cutoff)}
        store.save(month, state, 'frozen_controller_input')
        def persist(value):
            state['budget'] = value
            store.save(month, state, 'budget_projection')
        bridge = WorkAuthorityBridge(store, month, state, authority, persist)
        ScheduledSearch(bridge, default_schedule(config, month=month))
        # Explicit prepaid artificial baseline establishes the genuine S owner; no fit claimed.
        bridge.begin_work('baseline', counts={'solve': 7}, lease_seconds=300,
            binding={'stage': 'baseline'}).complete({'artificial_owner_setup': True, 'training_performed': False})
        return bridge, data, cutoff
    yield make
    for store in stores:
        store.close()


def _assert_one_prepaid_structure_review(bridge, result):
    from factorlab.discovery.monthly_partition_policy import counts
    from factorlab.discovery.structural_review import verify_review_completion
    selection = result['structure_search']
    reservation_id = selection['body']['review_reservation']
    assert reservation_id is not None
    verify_review_completion(bridge.authority, selection)
    review = bridge.authority.get(reservation_id)
    assert review['status'] == 'completed'
    assert review['binding']['work']['kind'] == 'structure_review'
    assert review['counts'] == counts(solve=1)
    assert review['actual_counts'] == counts(solve=1)
    assert review['lease']['lease_seconds'] == 300
    assert review['result']['result']['training_performed'] is False
    assert review['result']['result']['structural_plan_hash'] == selection['hash']
    assert bridge._active is None
    return review


def test_nondue_month_with_no_real_evidence_is_pending_and_spends_no_fits(setup):
    bridge, data, cutoff = setup('2025-05')
    original_reservations = bridge.authority.verify()['reservations']
    assert original_reservations == 1
    result = controller.run_shared_nonlinear(bridge, data, cutoff, mode='isolated', feed_directory=None)
    assert result['status'] == 'pending_gates' and result['prior_quarters'] == 0
    assert result['work'] == [] and result['serving_eligible'] is False
    review = _assert_one_prepaid_structure_review(bridge, result)
    assert bridge.authority.verify()['reservations'] == original_reservations + 1
    from factorlab.discovery.monthly_partition_policy import counts
    assert bridge.authority.status(review['actual_month'])['counts'] == counts(solve=8)
    before_readback = bridge.authority.verify()
    with pytest.raises(ValueError, match='readback only'):
        controller.run_shared_nonlinear(bridge, data, cutoff, mode='isolated', feed_directory=None)
    assert bridge.authority.verify() == before_readback


def test_quarter_controller_runs_registered_child_and_indexes_only_closed_actual_proof(setup, monkeypatch):
    bridge, data, cutoff = setup('2025-04')
    original_reservations = bridge.authority.verify()['reservations']
    assert original_reservations == 1
    processes = []
    class ActualChild:
        def __init__(self, command, **kwargs):
            self.inner = multiprocessing.get_context('spawn').Process(target=_actual_fixed_child,
                args=(str(bridge.store.directory.parent), command[-1]))
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
    result = controller.run_shared_nonlinear(bridge, data, cutoff, mode='isolated', feed_directory=None)
    assert result['status'] == 'completed_available_later'
    assert len(result['work']) == 1 and result['work'][0]['kind'] == 'quarterly_probe'
    assert result['prior_quarters'] == 0 and result['requires_prior_resource'] is True
    refs = list((bridge.authority.directory / 'nonlinear_epochs').glob('*/producer_refs/*.json'))
    assert len(refs) == 1
    verified = launcher.verify_completed_nonlinear(result['work'][0]['ticket'])
    assert verified['reservation']['actual_counts']['tree_fit'] == 3
    assert verified['reservation']['actual_counts']['solve'] == 9
    assert Path(result['work'][0]['ticket']).is_file()
    assert result['structure_search']['body']['review_reservation'] is None
    assert bridge.authority.verify()['reservations'] == original_reservations + 1
    from factorlab.discovery.monthly_partition_policy import counts
    assert bridge.authority.status(verified['reservation']['actual_month'])['counts'] == counts(solve=16, tree_fit=3)


def test_nonpositive_gate_closes_only_original_opportunity_and_retains_prepaid_counts(setup, monkeypatch):
    from factorlab.discovery import s_nonlinear_ticket
    from factorlab.discovery.s_nonlinear_gates import NonlinearGateClosed
    bridge, data, cutoff = setup('2025-05')
    original_reservations = bridge.authority.verify()['reservations']
    assert original_reservations == 1
    from factorlab.discovery.neural_forward import FrozenNeuralRecipe
    recipe = FrozenNeuralRecipe(pool=FrozenPool.from_dict(bridge.state['frozen_reference_pool']), rolling_spec=data.spec)
    # Private metadata enumeration seam only. Actual issue still prepays and
    # checks the real controller/data; no Linux or positive effect is claimed.
    monkeypatch.setattr(controller, '_history', lambda *a: [
        {'recipe_id': recipe.pipeline_id, 'available_at': '2025-04-02T00:00:00+00:00',
         'kind': 'quarterly_probe', 'cutoff': f'2025-{month}-01T00:00:00+00:00',
         'ticket_digest': month*32, 'ticket': month} for month in ('01', '04')])
    monkeypatch.setattr(s_nonlinear_ticket, '_gates',
        lambda *a: (_ for _ in ()).throw(NonlinearGateClosed('not positive')))
    result = controller.run_shared_nonlinear(bridge, data, cutoff, mode='isolated', feed_directory=None)
    assert result['status'] == 'gate_closed_consumed'
    row = bridge.authority.get(result['work'][0]['reservation_id'])
    assert row['status'] == 'failed_consumed' and row['counts']['neural_fit'] == 3
    assert row['counts']['solve'] == 7 and row['counts']['epoch'] == 12
    assert result['serving_eligible'] is False
    review = _assert_one_prepaid_structure_review(bridge, result)
    assert bridge.authority.verify()['reservations'] == original_reservations + 2
    from factorlab.discovery.monthly_partition_policy import counts
    assert bridge.authority.status(review['actual_month'])['counts'] == counts(solve=15, neural_fit=3, epoch=12)
