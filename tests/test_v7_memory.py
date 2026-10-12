"""Real local stores with tiny synthetic result records; no market/GPT calls."""
import copy
import json

import pytest
from test_v7_hypotheses import program
from test_v7_run_factor_pool import source_root

from factorlab.discovery.exposure import ExposureLedger
from factorlab.discovery.factor_pool import FactorPool, formula_identity
from factorlab.discovery.local_mutation import MonthlyBudget
from factorlab.discovery.rolling import RollingSpec, development_folds
from factorlab.discovery.v7_memory import ResearchMemory, search_exposure_context
from factorlab.discovery.v7_run import DurableBudget, RunStore, V7Config
from factorlab.discovery.v7_storage import immutable_record
from factorlab.protocol import fingerprint


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    memory = ResearchMemory.initial('2026-10-01T00:00Z', [
        {'program': program(), 'citation': 'test literature A'},
        {'program': program({'op': 'const', 'value': 1.}), 'citation': 'test literature B'}])
    store = RunStore(tmp_path / 'run', V7Config(), RollingSpec(),
                     exposure_path=tmp_path / 'exposure.sqlite')
    exposure = ExposureLedger(tmp_path / 'exposure.sqlite', clock=lambda: '2026-10-04T00:00Z')
    cutoff, month, data_id = '2026-10-01T00:00:00+00:00', '2026-10', 'synthetic_data'
    identifier = fingerprint(formula_identity(program(), 48))
    result = {'role': 'S', 'formula_id': identifier, 'folds': list(development_folds(cutoff)),
              'channels': {'F': {'gain': .01}, 'S': {'gain': .02}},
              'evidence_status': 'synthetic_test_record'}
    pool = FactorPool()
    pool.add(program(), result, family='family1')
    context = search_exposure_context(data_id, cutoff, ['BTCUSDT'])
    exposure.record(**{k: v for k, v in context.items() if k != 'data_id'}, role='S')
    state = {'cutoff': cutoff, 'data_id': data_id, 'pool': pool.to_dict(), 'status': 'running',
             'search_exposure': context}
    immutable_record(store.directory / 'results' / f'{month}-{identifier}.json', result)
    store.save(month, state, 'synthetic_fixture')
    yield memory, store, exposure, identifier, tmp_path
    store.close()
    exposure.close()


def proposal(parents=None):
    return {'family_id': 'family1', 'parent_ids': parents or ['refinement:seed:0'],
            'hypothesis': 'test causal hypothesis', 'mechanism': 'test past signal',
            'competing_explanation': 'duplicate scale', 'falsification': 'zero conditional gain'}


def measured(fixture):
    memory, store, exposure, identifier, _ = fixture
    version = memory.register_proposal(proposal(), island='refinement', next_experiment='test',
        minimum_progress=.001, remaining_budget=8, stop_conditions='two stalled months')
    evidence = memory.bind_result(store, exposure, '2026-10', identifier)
    parent = memory.register_measured_parent(family_id='family1', evidence_id=evidence,
                                              store=store, exposure=exposure)
    return version, evidence, parent


def test_actual_checkpoint_artifact_and_exposure_resolve_as_prior_only(fixture):
    memory, store, exposure, _, _ = fixture
    _, evidence, parent = measured(fixture)
    result = memory.prompt_parents(store, exposure, island='refinement', current_context={'pool': 'new'})
    resolved = next(r for r in result['ancestry_evidence'] if r['parent_id'] == parent)['evidence'][0]
    assert resolved['evidence_id'] == evidence and resolved['prior_gain_S'] == .02
    assert resolved['current_gain'] is None and resolved['scope'] == 'prior_context_only'
    assert all(p['current_gain'] is None for p in result['parents'])


@pytest.mark.parametrize('damage', ['missing_exposure', 'wrong_data_source', 'V_exposure', 'result_file', 'state_hash'])
def test_self_reported_S_does_not_bypass_actual_source_or_artifact(fixture, damage):
    memory, store, exposure, identifier, _ = fixture
    if damage == 'missing_exposure':
        exposure.db.execute('DELETE FROM evidence')
        exposure.db.commit()
    elif damage == 'V_exposure':
        context = store.get('2026-10')['search_exposure']
        exposure.record(**{k: v for k, v in context.items() if k != 'data_id'}, role='V')
    elif damage == 'wrong_data_source':
        state = store.get('2026-10')
        state['search_exposure']['source'] = 'otherdata'
        store.save('2026-10', state, 'bad_source')
    elif damage == 'result_file':
        (store.directory / 'results' / f'2026-10-{identifier}.json').write_text('{}')
    else:
        store.db.execute("UPDATE months SET hash='damaged'")
        store.db.commit()
    with pytest.raises(ValueError):
        memory.bind_result(store, exposure, '2026-10', identifier)


def test_ancestor_is_rechecked_after_initial_bind_and_all_later_use(fixture):
    memory, store, exposure, _, _ = fixture
    _, _, parent = measured(fixture)
    exposure.db.execute("UPDATE evidence SET role='C'")
    exposure.db.commit()
    with pytest.raises(ValueError, match='integrity'):
        memory.prompt_parents(store, exposure, island='refinement', current_context={})
    budget = MonthlyBudget('2026-10')
    budget.register_family('family1')
    durable = DurableBudget(budget, lambda _: None)
    with pytest.raises(ValueError, match='integrity'):
        memory.materialize(parent, [{'kind': 'replace', 'path': ['outputs', 0],
            'value': {'op': 'const', 'value': 2.}}], family_id='family1', budget=durable,
            store=store, exposure=exposure)
    assert budget.counts['ast'] == 0


def test_parent_family_mismatch_and_forged_parent_are_rejected(fixture):
    memory, store, exposure, identifier, _ = fixture
    with pytest.raises(ValueError, match='existing'):
        memory.register_proposal(proposal(['fake']), island='refinement', next_experiment='test',
            minimum_progress=.001, remaining_budget=8, stop_conditions='stop')
    altered = proposal()
    altered['family_id'] = 'another'
    memory.register_proposal(altered, island='refinement', next_experiment='test',
        minimum_progress=.001, remaining_budget=8, stop_conditions='stop')
    evidence = memory.bind_result(store, exposure, '2026-10', identifier)
    with pytest.raises(ValueError, match='another'):
        memory.register_measured_parent(family_id='another', evidence_id=evidence, store=store, exposure=exposure)


def test_save_restart_order_rollover_and_immutable_predecessor(fixture):
    memory, _, _, _, root = fixture
    measured(fixture)
    path = memory.save(root / 'memory.json')
    restored = ResearchMemory.from_state(json.loads(path.read_text('utf-8')))
    assert restored.to_state() == memory.to_state()
    assert list(restored.book.parents_by_id) == list(memory.book.parents_by_id)
    november = restored.rollover('2026-11-01T00:00Z', path)
    assert november.book.predecessor['digest'] == memory.book.digest
    assert november.book.verify_history()['shard_count'] == 1
    assert november.evidence == memory.evidence
    assert ResearchMemory.from_state(json.loads(path.read_text('utf-8'))).book.epoch == memory.book.epoch


def test_lane_reservations_persist_and_do_not_reset_on_restart(fixture):
    memory, _, _, _, _ = fixture
    budget = MonthlyBudget('2026-10')
    for family in ['one', 'two', 'three']:
        budget.register_family(family)
    candidates = [{'candidate_id': f'c{i}', 'family_id': ['one', 'two', 'three'][i % 3],
                   'lane': 'refine' if i < 12 else 'new' if i < 18 else 'blind'} for i in range(24)]
    checkpoints = []
    assert len(memory.reserve_schedule(candidates, budget, persist=checkpoints.append)['scheduled']) == 24
    restored = ResearchMemory.from_state(checkpoints[-1])
    assert not restored.reserve_schedule(candidates, budget, persist=checkpoints.append)['scheduled']
    assert restored.lanes == {'refine': 12, 'new': 6, 'blind': 6}
    assert budget.counts['f'] == 0  # Controller reserves F separately, immediately before actual work.


def test_local_mutation_reserves_ast_durably_even_on_invalid_output(fixture):
    memory, store, exposure, _, _ = fixture
    budget, states = MonthlyBudget('2026-10'), []
    budget.register_family('one')
    attempts = [{'kind': 'replace', 'path': ['outputs', 0, 'op'], 'value': 'unknown'}]
    with pytest.raises(TypeError, match='durable'):
        memory.materialize('refinement:seed:0', attempts, family_id='one', budget=budget,
                           store=store, exposure=exposure)
    records = memory.materialize('refinement:seed:0', attempts, family_id='one',
        budget=DurableBudget(budget, states.append), store=store, exposure=exposure)
    assert records[0]['status'] == 'invalid' and states[0]['counts']['ast'] == 1


def test_state_tampering_is_detected_and_capacity_is_explicit(fixture):
    memory, _, _, _, root = fixture
    state = memory.to_state()
    state['memory']['lanes']['blind'] = 5
    with pytest.raises(ValueError, match='integrity'):
        ResearchMemory.from_state(state)
    memory.proposals = {str(i): {} for i in range(96)}
    path = memory.save(root / 'full.json')
    with pytest.raises(ValueError, match='capacity exhausted'):
        memory.rollover('2026-11-01T00:00Z', path)


def test_partial_symbol_exposure_cannot_authorize_multi_symbol_memory(fixture):
    memory, store, exposure, identifier, _ = fixture
    # Isolate coverage from the source/config immutability tests using the genuine ledger API.
    context = copy.deepcopy(store.get('2026-10')['search_exposure'])
    with pytest.raises(ValueError, match='Each symbol'):
        exposure.assert_prompt_search(**{k: v for k, v in context.items() if k not in {'data_id', 'symbols'}},
                                      symbols=['BTCUSDT', 'ETHUSDT'])
    assert memory.bind_result(store, exposure, '2026-10', identifier)


def test_dormant_parent_cannot_bypass_prompt_filter_via_direct_local_mutation(fixture):
    memory, store, exposure, _, _ = fixture
    version, evidence, parent = measured(fixture)
    memory.observe(version, store=store, exposure=exposure, month='2026-10-01T00:00Z',
                   progress=0., spent=8, evidence_sources=[evidence])
    assert memory.book.versions[version]['state'] == 'dormant'
    budget = MonthlyBudget('2026-10')
    budget.register_family('one')
    with pytest.raises(ValueError, match='Dormant'):
        memory.materialize(parent, [{'kind': 'replace', 'path': ['outputs', 0],
            'value': {'op': 'const', 'value': 1.}}], family_id='one',
            budget=DurableBudget(budget, lambda _: None), store=store, exposure=exposure)
    descendant = proposal([parent])
    descendant['family_id'] = 'descendant'
    with pytest.raises(ValueError, match='Dormant'):
        memory.register_proposal(descendant, island='refinement', next_experiment='test',
            minimum_progress=.001, remaining_budget=8, stop_conditions='stop')
    assert budget.counts['ast'] == 0
