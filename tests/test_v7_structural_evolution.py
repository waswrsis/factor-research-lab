"""Real metadata/ledger, private feedback seams, artificial CPU numerics only."""
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file())
sys.path.insert(0, str(ROOT / 'tests'))
from test_v7_nonlinear_channel import artificial_data
from test_v7_run_factor_pool import source_root

from factorlab.discovery.joint_pool import FrozenPool
from factorlab.discovery.monthly_schedule import default_schedule
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
from factorlab.discovery.neural_forward import FrozenNeuralRecipe, NeuralForwardModel
from factorlab.discovery.nonlinear_channel import ChannelBudget
from factorlab.discovery.nonlinear_orchestration import input_data_identity
from factorlab.discovery.scheduled_search import ScheduledSearch
from factorlab.discovery.v7_memory import search_exposure_context
from factorlab.discovery.v7_run import RunStore, V7Config, monthly_budget
from factorlab.discovery.work_authority_bridge import WorkAuthorityBridge


@pytest.fixture
def module():
    from factorlab.discovery import structural_evolution
    return structural_evolution


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    data = artificial_data()
    config = V7Config(evaluation_mode='joint', symbols=('X',), nonlinear_mode='isolated')
    store = RunStore(tmp_path / 'research', config, data.spec)
    authority = MonthlyWorkAuthority(tmp_path / 'wallet', run_id='structural-artificial',
                                     task='volatility', universe=['X'], partition_policy='standard')
    def make(month):
        cutoff = month + '-01T00:00:00+00:00'
        exposed = search_exposure_context('artificial-context-' + month, cutoff, ['X'])
        store.exposure.record(**{k: v for k, v in exposed.items() if k != 'data_id'}, role='S')
        state = {'status': 'running', 'cutoff': cutoff, 'data_id': exposed['data_id'], 'search_exposure': exposed,
                 'budget': monthly_budget(config, month).to_dict(), 'frozen_reference_pool': FrozenPool().to_dict(),
                 'nonlinear_input_data_id': input_data_identity(data, cutoff)}
        store.save(month, state, 'artificial_original_input')
        def persist(value):
            state['budget'] = value
            store.save(month, state, 'original_projection')
        bridge = WorkAuthorityBridge(store, month, state, authority, persist)
        ScheduledSearch(bridge, default_schedule(config, month=month))
        return bridge, FrozenNeuralRecipe(rolling_spec=data.spec)
    yield make, data, store, authority
    store.close()












def test_finite_mutation_graph_is_executable_and_never_changes_training_grammar(module):
    root = FrozenNeuralRecipe()
    seen = {root.pipeline_id: root}
    pending = [root]
    while pending:
        parent = pending.pop()
        for child, _ in module.local_mutations(parent, eligible=True):
            assert child.config.structures[0].experts == 1
            assert child.config.seeds == root.config.seeds and child.config.epochs == 4
            assert child.pool.to_dict() == root.pool.to_dict()
            if child.pipeline_id not in seen:
                seen[child.pipeline_id] = child
                pending.append(child)
    assert len(seen) == 22


def test_mutated_width_changes_real_trained_matrix_and_full_three_fold_artifact(module, tmp_path):
    from factorlab.discovery.neural_research import _evaluate_artificial, research_capacity
    data = artificial_data()
    base = FrozenNeuralRecipe(rolling_spec=data.spec)
    child, edge = module.local_mutations(base, eligible=True)[0]
    assert edge == 'width' and child.config.structures[0].width == 16
    budget = ChannelBudget()
    result = _evaluate_artificial(data, '2025-04-01T00:00Z', child, symbols=('X',),
                                 directory=tmp_path / 'numeric', budget=budget)
    assert result['status'] == 'completed' and result['test_only']
    assert len(result['folds']) == 3 and result['pipeline_id'] == child.pipeline_id
    assert budget.counts == research_capacity(child)
    assert budget.counts['neural_fit'] == 9 and budget.counts['epoch'] == 36
    for i in range(3):
        model = NeuralForwardModel.load(tmp_path / 'numeric' / f'fold-{i}-model.json')
        assert model.recipe.pipeline_id == child.pipeline_id
        assert len(model.seeds) == 3
        assert len(model.seeds[0]['encoder']['0.weight']) == 16


def test_changed_seed_or_epoch_not_silently_relabelled_as_current_algorithm(module, campaign):
    make, _, _, _ = campaign
    bridge, recipe = make('2025-05')
    for config in (replace(recipe.config, epochs=2), replace(recipe.config, seeds=(17, 29))):
        with pytest.raises(ValueError, match='preserves'):
            module.select_structure(bridge, replace(recipe, config=config))








from factorlab.discovery.structural_review import combined_capacity, freeze_review_intent
from factorlab.discovery.monthly_partition_policy import counts
from factorlab.discovery.work_lease import FrozenWorkLease

def own(bridge):
    bridge.begin_work('baseline', counts={'solve': 7}, lease_seconds=300,
        binding={'stage': 'baseline'}).complete({'artificial_owner_setup': True})


def test_quarter_defers_numeric_until_next_month_on_same_actual_wallet(module, campaign, monkeypatch):
    make, _, _, authority = campaign
    first, base = make('2025-04')
    initial = module.select_structure(first, base)
    assert initial['body']['review_reservation'] is None
    seen = []
    def feedback(bridge, *args):
        work = bridge._active
        row = authority.get(work.record['reservation_id'])
        assert row['status'] == 'running' and row['allocation']['lane'] == 'nonlinear'
        assert row['counts'] == counts(solve=1) and row['lease']['lease_seconds'] == 300
        seen.append(row['reservation_id'])
        return {'terminal': True, 'eligible': True, 'reason': 'private_artificial_feedback'}
    monkeypatch.setattr(module, '_feedback', feedback)
    second, base = make('2025-05')
    own(second)
    result = module.select_structure(second, base)
    assert result['body']['recipe']['config']['structures'][0]['width'] == 16
    assert result['body']['parent_pipeline_id'] == initial['body']['pipeline_id']
    assert seen == [result['body']['review_reservation']]
    row = authority.get(seen[0]); assert row['status'] == 'completed'
    assert row['actual_counts'] == counts(solve=1)
    assert row['authority_directory'] == str(authority.directory)
    before = authority.status(row['actual_month'])
    assert module.select_structure(second, base) == result
    assert authority.status(row['actual_month']) == before
    assert authority.verify()['reservations'] == 2  # Original S owner plus same-wallet review.


def test_quarter_does_not_read_completed_models_or_cancel_probe(module, campaign, monkeypatch):
    make, _, _, authority = campaign
    first, base = make('2025-01'); module.select_structure(first, base)
    monkeypatch.setattr(module, '_feedback', lambda *a: pytest.fail('unpaid numerical feedback'))
    second, base = make('2025-04')
    result = module.select_structure(second, base)
    assert result['body']['pipeline_id'] == base.pipeline_id
    assert result['body']['feedback']['reason'] == 'quarter_probe_priority_review_deferred'
    assert authority.verify()['reservations'] == 0
    from factorlab.discovery.s_nonlinear_ticket import freeze_nonlinear_plan
    plan = freeze_nonlinear_plan(second, base, kinds=('quarterly_probe',))
    assert [w['kind'] for w in plan['body']['work']] == ['quarterly_probe']


def test_review_original_deadline_failure_charged_and_not_reissued(module, campaign, monkeypatch):
    make, _, _, authority = campaign
    first, base = make('2025-04'); module.select_structure(first, base)
    second, base = make('2025-05'); own(second)
    original = FrozenWorkLease.assert_parent_alive
    expired = {'digest': None}
    def checked(self, *a, **kw):
        if self.digest == expired['digest']:
            raise TimeoutError('original 300s review expired')
        return original(self, *a, **kw)
    monkeypatch.setattr(FrozenWorkLease, 'assert_parent_alive', checked)
    def feedback(bridge, *a):
        expired['digest'] = bridge._active.lease.digest
        return {'terminal': True, 'eligible': True, 'reason': 'private delayed readback'}
    monkeypatch.setattr(module, '_feedback', feedback)
    with pytest.raises(TimeoutError, match='original 300s'):
        module.select_structure(second, base)
    with authority._transaction() as db:
        rows = list(authority._verify(db)['reservations'].values())
    review = next(r for r in rows if r['binding'].get('work', {}).get('kind') == 'structure_review')
    assert review['status'] == 'failed_consumed' and review['counts'] == counts(solve=1)
    before = authority.status(review['actual_month'])
    with pytest.raises(ValueError, match='already prepaid|closed'):
        module.select_structure(second, base)
    assert authority.status(review['actual_month']) == before


def test_combined_three_works_bound_and_fourth_cannot_borrow_forward(module, campaign):
    make, _, _, _ = campaign
    bridge, recipe = make('2025-05')
    allocation = json.loads((bridge.directory/'schedule/identity.json').read_text())['schedule']['allocation']
    value = combined_capacity(recipe, ['resource_trial', 'complete_neural_S'], allocation, review=True)
    assert value['seconds'] == 900 and value['counts']['neural_fit'] == 12 and value['counts']['epoch'] == 48
    with pytest.raises(ValueError, match='original nonlinear capacity'):
        combined_capacity(recipe, ['quarterly_probe', 'resource_trial', 'complete_neural_S'], allocation, review=True)
    assert allocation['nonlinear_seconds_holdback'] == 900 and allocation['forward_seconds_holdback'] == 2820


@pytest.mark.parametrize('fault', ['missing_mirror', 'wrong_work_key', 'wrong_debit', 'wrong_lease'])
def test_typed_review_cannot_self_label_foreign_grant(module, campaign, fault):
    make, _, _, authority = campaign
    bridge, base = make('2025-05'); own(bridge)
    intent = freeze_review_intent(bridge, base)
    if fault == 'missing_mirror':
        (bridge.anchor/'structure_review.json').unlink()
    with pytest.raises(ValueError):
        bridge.begin_work('other' if fault == 'wrong_work_key' else 'structure_review',
            counts={'solve': 2 if fault == 'wrong_debit' else 1},
            lease_seconds=299 if fault == 'wrong_lease' else 300,
            binding={'kind': 'structure_review', 'intent_hash': intent['hash']})
    assert authority.verify()['reservations'] == 1


def test_quarter_unreviewed_recipe_cannot_run_fullS(module, campaign):
    from factorlab.discovery.s_nonlinear_ticket import freeze_nonlinear_plan
    make, _, _, authority = campaign
    bridge, base = make('2025-04'); module.select_structure(bridge, base)
    with pytest.raises(ValueError, match='completed structure review'):
        freeze_nonlinear_plan(bridge, base, kinds=('complete_neural_S',))
    assert authority.verify()['reservations'] == 0


def test_guard_releases_only_after_complete_review_and_no_release_on_exception(module, campaign):
    make, _, _, authority = campaign
    bridge, base = make('2025-05'); own(bridge)
    path = ROOT/'scripts/v7_guard/guarded_runtime.py'
    spec = importlib.util.spec_from_file_location('review_guard_test', path)
    guard = importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)
    calls = []
    class Scope:
        @property
        def current(self):
            with authority._transaction() as db:
                rows = authority._verify(db)['reservations'].values()
            found = [r for r in rows if r['binding'].get('work', {}).get('kind') == 'structure_review']
            return found[-1] if found else None
        def finish_work(self, rid):
            row = authority.get(rid)
            assert row['status'] == 'completed'
            assert bridge.state['structural_plan']['hash'] == row['result']['result']['structural_plan_hash']
            calls.append(rid)
    with guard.nonlinear_publication(Scope(), authority):
        result = module.select_structure(bridge, base)
    assert calls == [result['body']['review_reservation']]

def test_guard_keeps_failed_review_under_original_supervision(module, campaign, monkeypatch):
    make, _, _, authority = campaign
    first, base = make('2025-04'); module.select_structure(first, base)
    bridge, base = make('2025-05'); own(bridge)
    path = ROOT/'scripts/v7_guard/guarded_runtime.py'
    spec = importlib.util.spec_from_file_location('review_guard_failure_test', path)
    guard = importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)
    class Scope:
        def __init__(self):
            self.current = {'reservation_id': 'only a failure-test sentinel; never a grant'}
        def finish_work(self, rid):
            pytest.fail('A failed/unpublished review must not release hard supervision')
    def feedback(*args):
        raise OSError('artificial historical artifact IO failed')
    monkeypatch.setattr(module, '_feedback', feedback)
    with guard.nonlinear_publication(Scope(), authority), pytest.raises(OSError, match='artifact IO'):
        module.select_structure(bridge, base)
    with authority._transaction() as db:
        rows = list(authority._verify(db)['reservations'].values())
    assert rows[-1]['status'] == 'failed_consumed' and rows[-1]['counts'] == counts(solve=1)


def test_completed_row_without_exact_review_result_cannot_release_candidate(module, campaign, monkeypatch):
    from factorlab.discovery.structural_review import verify_review_completion
    make, _, _, authority = campaign
    bridge, base = make('2025-05'); own(bridge)
    selected = module.select_structure(bridge, base)
    before = authority.get(selected['body']['review_reservation'])
    # Private supplied-return corruption tests the terminal binding, not a fake authority acceptance.
    monkeypatch.setattr(authority, 'get', lambda rid: {**before, 'result': {'result': {'kind': 'structure_review'}}})
    with pytest.raises(ValueError, match='fully completed'):
        verify_review_completion(authority, selected)

def test_actual_controller_reviews_mutated_recipe_then_waits_for_its_own_gates(module, campaign, monkeypatch):
    from factorlab.discovery.s_nonlinear_controller import run_shared_nonlinear
    make, data, _, authority = campaign
    first, base = make('2025-04'); original = module.select_structure(first, base)
    second, _ = make('2025-05'); own(second)
    monkeypatch.setattr(module, '_feedback', lambda *a: {'terminal': True, 'eligible': True,
        'reason': 'private artificial class; no market effect claim'})
    result = run_shared_nonlinear(second, data, second.state['cutoff'], mode='isolated', feed_directory=None)
    assert result['recipe']['config']['structures'][0]['width'] == 16
    assert result['structure_search']['body']['parent_pipeline_id'] == original['body']['pipeline_id']
    assert result['prior_quarters'] == 0 and result['work'] == []
    assert result['combined_allocation']['seconds'] == 300
    assert any(d['reason'] == 'independent_recipe_prior_quarter_gates_missing' for d in result['deferred'])
    assert authority.verify()['reservations'] == 2

def test_unpartitioned_diagnostic_wallet_cannot_buy_a_production_queue(campaign):
    from factorlab.discovery.structural_queue_review import produce_typed_queue_review

    from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
    make, _, store, authority = campaign
    make('2025-05')
    diagnostic = MonthlyWorkAuthority(authority.directory.parent/'unpartitioned', run_id='diagnostic',
        task='volatility', universe=['X'], partition_policy=None)
    with pytest.raises(ValueError, match='nontransferable monthly partitions'):
        produce_typed_queue_review(None, diagnostic, store.directory, '2025-05')
    assert diagnostic.verify()['reservations'] == 0
