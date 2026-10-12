"""Capacity/ancestry controls use genuine S stores but synthetic numeric records."""
import json

import pandas as pd
import pytest
import test_v7_memory as memory_tests
from test_v7_hypotheses import program, spec

from factorlab.discovery.factor_pool import FactorPool, formula_identity
from factorlab.discovery.hypotheses import HypothesisBook
from factorlab.discovery.local_mutation import MonthlyBudget
from factorlab.discovery.rolling import development_folds
from factorlab.discovery.v7_memory import ResearchMemory, search_exposure_context
from factorlab.discovery.v7_storage import immutable_record
from factorlab.protocol import fingerprint

memory_fixture = memory_tests.fixture


def test_six_months_144_measured_parents_keep_bounded_live_index_and_resolve_old_S(memory_fixture):
    memory, store, exposure, _, root = memory_fixture
    first_parent = first_evidence = first_version = first_memory_path = None
    last_parent = {'refinement': 'refinement:seed:0', 'exploration': 'exploration:seed:0'}
    total = 0
    for cycle in range(6):
        cutoff = pd.Timestamp('2026-10-01T00:00Z') + pd.DateOffset(months=cycle)
        month = cutoff.strftime('%Y-%m')
        exposure.clock = lambda when=cutoff: when + pd.Timedelta(days=4)
        pool, entries = FactorPool(), []
        families = [f'{month}-family-{i}' for i in range(3)]
        for index in range(24):
            formula = program({'op': 'const', 'value': .01 * (cycle * 24 + index + 1)})
            identifier = fingerprint(formula_identity(formula, 48))
            result = {'role': 'S', 'formula_id': identifier, 'folds': list(development_folds(cutoff)),
                'channels': {'F': {'gain': .01}, 'S': {'gain': .02}}, 'scope': 'synthetic_test_only'}
            family = families[index // 8]
            pool.add(formula, result, family=family)
            immutable_record(store.directory / 'results' / f'{month}-{identifier}.json', result)
            entries.append((identifier, family))
        context = search_exposure_context(f'data-{cycle}', cutoff, ['BTCUSDT'])
        exposure.record(**{k: v for k, v in context.items() if k != 'data_id'}, role='S')
        store.save(month, {'cutoff': cutoff.isoformat(), 'data_id': f'data-{cycle}', 'pool': pool.to_dict(),
            'status': 'running', 'search_exposure': context}, 'synthetic_month')
        versions = {}
        for i, family in enumerate(families):
            island = 'refinement' if i % 2 == 0 else 'exploration'
            proposal = {'family_id': family, 'parent_ids': [last_parent[island]],
                'hypothesis': 'Synthetic capacity hypothesis', 'mechanism': 'Past observations',
                'competing_explanation': 'Redundant statistic', 'falsification': 'No conditional information'}
            versions[family] = memory.register_proposal(proposal, island=island, next_experiment='bounded control',
                minimum_progress=.001, remaining_budget=8, stop_conditions='two stalled cycles')
            for identifier, owner in entries:
                if owner != family:
                    continue
                evidence = memory.bind_result(store, exposure, month, identifier)
                last_parent[island] = memory.register_measured_parent(
                    family_id=family, evidence_id=evidence, store=store, exposure=exposure)
                total += 1
                if first_parent is None:
                    first_parent, first_evidence, first_version = last_parent[island], evidence, versions[family]
            memory.observe(versions[family], store=store, exposure=exposure, month=cutoff,
                           progress=.01, spent=1, evidence_sources=[evidence])
        # Unparented active hypotheses also require bounded indexing and retained
        # stopping state; they cannot accumulate forever in the live dictionary.
        for i in range(4):
            memory.book.add(spec(f'{month}-unparented-{i}'))
        budget = MonthlyBudget(month)
        for family in families:
            budget.register_family(family)
        candidates = [{'candidate_id': identifier, 'family_id': family,
            'lane': 'refine' if i < 12 else 'new' if i < 18 else 'blind'}
            for i, (identifier, family) in enumerate(entries)]
        saved_states = []
        assert len(memory.reserve_schedule(candidates, budget, persist=saved_states.append)['scheduled']) == 24
        memory = ResearchMemory.from_state(saved_states[-1])
        assert not memory.reserve_schedule(candidates, budget, persist=saved_states.append)['scheduled']
        assert memory.lanes == {'refine': 12, 'new': 6, 'blind': 6}
        path = memory.save(root / f'memory-{month}.json')
        first_memory_path = first_memory_path or path
        memory = memory.rollover(cutoff + pd.DateOffset(months=1), path)
        memory = ResearchMemory.from_state(memory.to_state())
        assert len(memory.book.parents_by_id) <= 20
        assert len(memory.book.versions) <= 32
        assert len(memory.evidence) <= 16 and len(memory.proposals) <= 32
        assert memory.lanes == {'refine': 0, 'new': 0, 'blind': 0}
    assert total == 144
    assert first_parent not in memory.book.parents_by_id
    assert first_evidence not in memory.evidence
    assert first_version not in memory.book.versions
    assert memory.book.get_version(first_version)['remaining_budget'] == 7
    assert memory.get_proposal('2026-10-family-0')['version'] == first_version
    parent = memory.book.get_parent(first_parent, resolver=memory.resolver(store, exposure))
    assert parent['evidence_sources'] == [first_evidence]
    output = memory.prompt_parents(store, exposure, island='refinement', current_context={'new': True})
    assert all(p['current_gain'] is None for p in output['parents'])
    assert memory.verify_history()['shard_count'] == 6
    # Bringing an old version back into the writable index retains its quota.
    memory.observe(first_version, store=store, exposure=exposure, month=memory.book.epoch,
                   progress=.01, spent=1, evidence_sources=[first_evidence])
    assert memory.book.versions[first_version]['remaining_budget'] == 6
    # An old lineage identifier cannot be recycled as a new hypothesis/family.
    with pytest.raises(ValueError, match='versioned revival'):
        memory.book.add(spec('2026-10-family-0'))
    # The genuine original S ledger remains authoritative after index eviction.
    first_context = store.get('2026-10')['search_exposure']
    exposure.record(**{k: v for k, v in first_context.items() if k != 'data_id'}, role='V')
    with pytest.raises(ValueError, match='Protected'):
        memory.book.get_parent(first_parent, resolver=memory.resolver(store, exposure))
    # Archived memory bytes are checked even when the requested live parent exists.
    saved = json.loads(first_memory_path.read_text('utf-8'))
    saved['memory']['evidence'][first_evidence]['data_id'] = 'tampered'
    first_memory_path.write_text(json.dumps(saved), encoding='utf-8')
    with pytest.raises(ValueError, match='predecessor'):
        ResearchMemory.from_state(memory.to_state())


def test_hashbound_missing_ancestor_and_cycle_are_never_trusted(memory_fixture):
    memory, _, _, _, root = memory_fixture
    path = memory.save(root / 'old.json')
    new = memory.rollover('2026-11-01T00:00Z', path)
    with pytest.raises(ValueError, match='Unresolved'):
        new.book.get_parent('invented-ancestor')
    # A predecessor cannot point forward/back to the current epoch even when
    # its payload has a self-consistent content hash.
    later = HypothesisBook('2026-12-01T00:00Z')
    later_path = later.save(root / 'future-book.json')
    new.book.predecessor = {'path': str(later_path), 'digest': later.digest}
    with pytest.raises(ValueError, match='chronology'):
        new.book.verify_history()


def test_predecessor_bound_rejects_more_than_120_shards(tmp_path):
    predecessor = None
    for i in range(121):
        book = HypothesisBook(pd.Timestamp('2000-01-01T00:00Z') + pd.DateOffset(months=i),
                              predecessor=predecessor)
        path = book.save(tmp_path / f'{i}.json')
        predecessor = {'path': str(path), 'digest': book.digest}
    current = HypothesisBook('2011-01-01T00:00Z', predecessor=predecessor)
    with pytest.raises(ValueError, match='oversized'):
        current.verify_history()
