"""Original source/preparation/ledger boundaries with synthetic old S readouts.
No provider, score fitting, serving grant, or market claim is exercised.
"""
import copy
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from test_v7_hypotheses import program
from test_v7_s_observation_source import factory, original_S_source
from factorlab.discovery.factor_pool import FactorPool, formula_identity
from factorlab.discovery.rolling import development_folds
from factorlab.discovery.s_preparation import SPreparationEnvelope
from factorlab.discovery.v7_memory import ResearchMemory, search_exposure_context
from factorlab.discovery.v7_run import RunStore, V7Config, monthly_budget
from factorlab.discovery.v7_storage import immutable_record
from factorlab.discovery.work_authority_bridge import WorkAuthorityBridge
from factorlab.protocol import fingerprint


@pytest.fixture
def revival(factory, request):
    _, runtime, wallet, prefix, cutoff, _, reference = original_S_source(factory)
    config = V7Config(symbols=('BTCUSDT',), evaluation_mode='joint')
    store = RunStore(runtime.directory/'revival-S', config, runtime.spec,
        exposure_path=runtime.lifecycle.exposure.path, feed_identity=runtime.feed.identity)
    old_cutoff = cutoff - pd.DateOffset(months=1)
    scenario = getattr(request, 'param', None)
    old_data_id = 'new-artificial-data' if scenario == 'same_data' else 'old-artificial-data'
    memory = ResearchMemory.initial(old_cutoff, [
        {'program': program(), 'citation': 'Artificial seed A'},
        {'program': program({'op': 'const', 'value': 1.}), 'citation': 'Artificial seed B'}])
    try:
        value = program(); identifier = fingerprint(formula_identity(value, 48))
        result = {'role': 'S', 'formula_id': identifier, 'folds': list(development_folds(old_cutoff)),
                  'channels': {'F': {'gain': 0.}, 'S': {'gain': 0.}},
                  'evidence_status': 'synthetic_record_not_computed_score'}
        pool = FactorPool(); pool.add(value, result, family='old-family')
        old_context = search_exposure_context(old_data_id, old_cutoff, ['BTCUSDT'], runtime.feed.identity)
        store.exposure.record(**{k: v for k, v in old_context.items() if k != 'data_id'}, role='S')
        old_state = {'cutoff': old_cutoff.isoformat(), 'data_id': old_data_id,
                     'search_exposure': old_context, 'pool': pool.to_dict(), 'status': 'running'}
        old_month = old_cutoff.strftime('%Y-%m')
        immutable_record(store.directory/'results'/f'{old_month}-{identifier}.json', result)
        store.save(old_month, old_state, 'synthetic_original_readout')
        version = memory.register_proposal({'family_id': 'old-family', 'parent_ids': ['refinement:seed:0'],
            'hypothesis': 'Original artificial hypothesis', 'mechanism': 'Past volatility',
            'competing_explanation': 'Scale duplicate', 'falsification': 'No conditional increment'},
            island='refinement', next_experiment='Fixed data-trigger review', minimum_progress=.005,
            remaining_budget=1, stop_conditions='Two stalls or exhausted')
        evidence = memory.bind_result(store, store.exposure, old_month, identifier)
        parent = memory.register_measured_parent(family_id='old-family', evidence_id=evidence,
            store=store, exposure=store.exposure)
        memory.observe(version, store=store, exposure=store.exposure,
            month=cutoff if scenario == 'same_observation_month' else old_cutoff,
            progress=0., evidence_sources=[evidence], spent=1, failure='Artificial stall')
        previous = memory.to_state()
        saved = memory.save(store.directory/'memories'/f'{old_month}.json')
        old_state.update(status='complete', research_memory=previous)
        store.save(old_month, old_state, 'original_memory_closed')
        memory = memory.rollover(cutoff, saved)
        context = search_exposure_context('new-artificial-data', cutoff, ['BTCUSDT'], runtime.feed.identity)
        store.exposure.record(**{k: v for k, v in context.items() if k != 'data_id'}, role='S')
        state = {'cutoff': cutoff.isoformat(), 'data_id': context['data_id'], 'search_exposure': context,
                 'pool': FactorPool().to_dict(), 'status': 'running',
                 'budget': monthly_budget(config, cutoff.strftime('%Y-%m')).to_dict(),
                 'research_memory': memory.to_state()}
        store.save(cutoff.strftime('%Y-%m'), state, 'current_original_S')
        prep = SPreparationEnvelope.prepare(wallet, store.directory, cutoff.strftime('%Y-%m'),
            config=asdict(config), spec=asdict(runtime.spec), source_reference=reference, schedule_hash='a'*64)
        prep.start_preprocessing(); prep.seal_input(data_id=state['data_id'], search_exposure=context)
        def persist_budget(value):
            state['budget'] = value
            store.save(cutoff.strftime('%Y-%m'), state, 'real_bridge_projection')
        bridge = WorkAuthorityBridge(store, cutoff.strftime('%Y-%m'), state, wallet, persist_budget, preparation=prep)
        checkpoints = []
        def persist(value):
            checkpoints.append(copy.deepcopy(value)); state['research_memory'] = value
            store.save(cutoff.strftime('%Y-%m'), state, 'automatic_revival_checkpoint')
        yield SimpleNamespace(memory=memory, store=store, wallet=wallet, runtime=runtime,
            prefix=prefix, cutoff=cutoff, prep=prep, bridge=bridge, state=state, previous=previous,
            old_version=version, old_parent=parent, evidence=evidence, saved=saved,
            persist=persist, checkpoints=checkpoints)
    finally:
        store.close()


def test_new_original_S_data_revives_once_preserves_ancestry_and_all_credits(revival):
    c = revival; before = c.wallet.verify(); projected = copy.deepcopy(c.state['budget'])
    old_bytes = c.saved.read_bytes()
    outcome = c.memory.automatic_revive(c.store, c.store.exposure, island='refinement', persist=c.persist)
    assert outcome['status'] == 'revived' and outcome['adds_work_credit'] is False
    child = c.memory.book.get_version(outcome['version'])
    assert child['parent_version'] == c.old_version and child['history'] == []
    assert child['remaining_budget'] == 1 and child['previous_failures'] == [{'month': '2025-12-01T00:00:00+00:00', 'failure': 'Artificial stall'}]
    assert c.memory.book.get_version(c.old_version)['state'] == 'dormant'
    parent = c.memory.book.get_parent(outcome['parent_id'])
    assert parent['ancestor_ids'] == [c.old_parent] and parent['evidence_sources'] == [c.evidence]
    trigger = child['revival_trigger']['verified_evidence'][0]
    assert trigger['scope'] == 'data_trigger_only_not_a_score' and trigger['current_gain'] is None
    assert trigger['source_reference']['observation_prefix'] == c.prefix
    assert trigger['preparation_hash'] == c.prep.digest
    restored = ResearchMemory.from_state(c.checkpoints[-1])
    assert restored.automatic_revive(c.store, c.store.exposure, island='refinement', persist=c.persist)['status'] == 'already_considered'
    prompt = restored.prompt_parents(c.store, c.store.exposure, island='refinement', current_context={'new': True})
    assert outcome['parent_id'] in [p['parent_id'] for p in prompt['parents']]
    assert all(p['current_gain'] is None for p in prompt['parents'])
    assert len(c.checkpoints) == 1 and c.wallet.verify() == before and c.state['budget'] == projected
    assert c.saved.read_bytes() == old_bytes


@pytest.mark.parametrize('damage', ['missing_preparation', 'V', 'C', 'changed_input', 'foreign_context'])
def test_revival_rejects_missing_original_or_nonS_current_context(revival, damage):
    c = revival; before = c.memory.to_state(); budget = c.wallet.verify()
    if damage == 'missing_preparation':
        original = c.prep.root/'input.json'; original.rename(original.with_name('retained-input.json'))
    elif damage in {'V', 'C'}:
        ctx = c.state['search_exposure']
        c.store.exposure.record(**{k: v for k, v in ctx.items() if k != 'data_id'}, role=damage)
    else:
        c.state['data_id'] = 'forged' if damage == 'changed_input' else c.state['data_id']
        if damage == 'foreign_context': c.state['search_exposure']['source'] = 'foreign'
        c.store.save('2026-01', c.state, 'artificial_tamper')
    with pytest.raises((ValueError, OSError)):
        c.memory.automatic_revive(c.store, c.store.exposure, island='refinement', persist=c.persist)
    assert c.memory.to_state() == before and not c.checkpoints
    if damage not in {'missing_preparation', 'changed_input', 'foreign_context'}:
        assert c.wallet.verify() == budget


def test_persisted_trigger_is_reverified_before_prompt_reuse(revival):
    c = revival
    c.memory.automatic_revive(c.store, c.store.exposure, island='refinement', persist=c.persist)
    ctx = c.state['search_exposure']
    c.store.exposure.record(**{k: v for k, v in ctx.items() if k != 'data_id'}, role='V')
    with pytest.raises(ValueError):
        c.memory.prompt_parents(c.store, c.store.exposure, island='refinement', current_context={})


@pytest.mark.parametrize('revival', ['same_data', 'same_observation_month'], indirect=True)
def test_predefined_new_data_and_later_month_are_both_required(revival):
    c = revival; before = c.memory.to_state(); budget = c.wallet.verify()
    result = c.memory.automatic_revive(c.store, c.store.exposure, island='refinement', persist=c.persist)
    assert result['status'] == 'new_data_trigger_not_met'
    assert c.memory.to_state() == before and not c.checkpoints and c.wallet.verify() == budget


def test_closed_S_month_cannot_retroactively_create_revival(revival):
    c = revival; before = c.memory.to_state()
    c.state['status'] = 'complete'; c.store.save('2026-01', c.state, 'closed_synthetic_boundary')
    with pytest.raises(ValueError, match='active original S month'):
        c.memory.automatic_revive(c.store, c.store.exposure, island='refinement', persist=c.persist)
    assert c.memory.to_state() == before and not c.checkpoints


def test_default_paid_family_wires_revival_before_GPT_and_local_parent_reads():
    import ast
    from factorlab.discovery import v7_run
    tree = ast.parse(Path(v7_run.__file__).read_text('utf-8'))
    scheduled = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_run_month_scheduled_impl')
    family = next(n for n in ast.walk(scheduled) if isinstance(n, ast.FunctionDef) and n.name == 'family_work')
    calls = [n for n in ast.walk(family) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    revive = [n for n in calls if n.func.attr == 'automatic_revive']
    parents = [n for n in calls if n.func.attr == 'prompt_parents']
    assert len(revive) == 1 and parents and all(revive[0].lineno < n.lineno for n in parents)
    assert any(k.arg == 'persist' and isinstance(k.value, ast.Name) and k.value.id == 'persist_memory' for k in revive[0].keywords)
