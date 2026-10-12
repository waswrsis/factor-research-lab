import copy

import pytest

from factorlab.discovery.hypotheses import (
    HypothesisBook,
    materialize_mutations,
    schedule_candidates,
)
from factorlab.discovery.known_library import library_schema
from factorlab.discovery.local_mutation import BudgetExhausted, MonthlyBudget


def spec(identifier='h1', island='refinement'):
    return {'hypothesis_id': identifier, 'island': island, 'mechanism': 'short shock memory',
            'competing_explanation': 'only a scale change', 'falsification': 'no conditional gain',
            'next_experiment': 'fixed scale duplicate control', 'minimum_progress': .001,
            'remaining_budget': 8, 'stop_conditions': 'two stalled monthly cycles'}


def program(expression=None, parameters=None):
    return {'version': 'causal-operator-program-v1', 'feature_names': library_schema()['feature_names'],
            'parameters': parameters or {}, 'outputs': [expression or
                {'op': 'last', 'arg': {'op': 'feature', 'name': 'rv_1'}}]}


def resolver(identifier):
    return {'evidence_id': identifier, 'role': 'S', 'batch': 'search-july'}


def test_stall_sleep_explicit_version_revival_and_immutable_old_evidence(tmp_path):
    book = HypothesisBook('2026-07-01T00:00Z')
    original = book.add(spec())
    for month in ('2026-07-01T00:00Z', '2026-08-01T00:00Z'):
        receipt = book.observe(original, month=month, progress=0, evidence_sources=['source1'],
                               resolver=resolver, failure='duplicate scaling', spent=1)
    assert receipt['state'] == 'dormant' and receipt['reason'] == 'two_stalled_cycles'
    saved_old = copy.deepcopy(book.versions[original])
    revived = book.revive(original, trigger={'kind': 'new_input', 'evidence_id': 'newinput',
                                           'description': 'registered new causal feature'},
                           resolver=resolver, remaining_budget=4)
    assert revived != original and book.versions[original] == saved_old
    assert book.versions[revived]['parent_version'] == original
    assert len(book.versions[revived]['previous_failures']) == 2
    with pytest.raises(ValueError, match='successor'):
        book.revive(original, trigger={'kind': 'new_data', 'evidence_id': 'newdata',
                                      'description': 'new observations'}, resolver=resolver, remaining_budget=4)
    assert HypothesisBook.load(book.save(tmp_path / 'book.json')).digest == book.digest


def test_missing_month_and_new_mechanism_do_not_count_as_stagnation():
    book = HypothesisBook('2026-07-01T00:00Z')
    version = book.add(spec())
    for month in ('2026-07-01T00:00Z', '2026-09-01T00:00Z'):
        receipt = book.observe(version, month=month, progress=0, evidence_sources=['e'], resolver=resolver)
        assert receipt['state'] == 'active'
    receipt = book.observe(version, month='2026-10-01T00:00Z', progress=-1., new_mechanism=True,
                           evidence_sources=['e'], resolver=resolver)
    assert receipt['state'] == 'active'
    with pytest.raises(ValueError, match='increase'):
        book.observe(version, month='2026-09-01T00:00Z', progress=0, evidence_sources=['e'], resolver=resolver)


def parent(identifier, *, kind='literature_seed', ancestors=None, sources=None, version=None):
    return {'parent_id': identifier, 'island': 'refinement', 'program': program(),
            'ancestor_ids': ancestors or [], 'evidence_sources': sources or [], 'kind': kind,
            'citation': 'registered reference' if kind == 'literature_seed' else None,
            'hypothesis_version': version}


def test_every_parent_ancestor_requires_resolved_S_and_dormant_is_inactive():
    book = HypothesisBook('2026-07-01T00:00Z')
    version = book.add(spec())
    book.register_parent(parent('seed1'))
    book.register_parent(parent('seed2'))
    book.register_parent(parent('ancestor', kind='research', sources=['ancestorS'], version=version))
    book.register_parent(parent('descendant', kind='research', ancestors=['ancestor'], sources=['ownS'], version=version))
    calls = []

    def checked(identifier):
        calls.append(identifier)
        return resolver(identifier)

    result = book.parents('refinement', resolver=checked, count=2)
    assert len(result['parents']) == 2 and set(calls) == {'ancestorS', 'ownS'}

    def contaminated(identifier):
        return {**resolver(identifier), 'role': 'C' if identifier == 'ancestorS' else 'S'}

    with pytest.raises(ValueError, match='ancestor'):
        book.parents('refinement', resolver=contaminated, count=2)
    book.observe(version, month='2026-07-01T00:00Z', progress=0, evidence_sources=['e'], resolver=resolver, spent=8)
    assert [p['parent_id'] for p in book.parents('refinement', resolver=resolver)['parents']] == ['seed1', 'seed2']


def test_parent_alias_and_cycle_cannot_overwrite_history():
    book = HypothesisBook('2026-07-01T00:00Z')
    book.register_parent(parent('seed1'))
    with pytest.raises(ValueError, match='immutable'):
        book.register_parent(parent('seed1'))
    with pytest.raises(ValueError, match='ancestry'):
        book.register_parent(parent('cycle', ancestors=['cycle']))
    with pytest.raises(ValueError, match='literature seeds'):
        book.parents('refinement', resolver=resolver)


def test_schedule_quotas_and_family_caps_never_create_families_or_force_fill():
    budget = MonthlyBudget('2026-07')
    budget.register_family('one')
    budget.register_family('two')
    before = budget.digest
    candidates = [{'candidate_id': f'c{i}', 'family_id': 'one' if i % 2 else 'two',
                   'lane': 'refine' if i < 20 else 'blind'} for i in range(30)]
    result = schedule_candidates(candidates, budget)
    assert result['receipt']['actual'] == {'refine': 12, 'new': 0, 'blind': 4}
    assert len(result['scheduled']) == 16 and budget.digest == before
    assert len(budget.families) == 2 and result['receipt']['reserves_f'] is False
    repeated = schedule_candidates(candidates, budget, used_by_lane=result['receipt']['cumulative_reserved'])
    assert repeated['receipt']['actual']['refine'] == 0
    with pytest.raises(ValueError, match='unregistered'):
        schedule_candidates([{'candidate_id': 'x', 'family_id': 'invented', 'lane': 'new'}], budget)


def test_mutation_reserves_invalid_duplicate_and_preserves_integer_window():
    base = program({'op': 'last', 'arg': {'op': 'rolling_mean', 'window': 2,
                    'arg': {'op': 'feature', 'name': 'rv_1'}}})
    budget = MonthlyBudget('2026-07')
    budget.register_family('one')
    attempts = [{'kind': 'replace', 'path': ['outputs', 0, 'arg', 'window'], 'value': value}
                for value in (3, 3, 2.9, 0)]
    records = materialize_mutations(base, attempts, family_id='one', budget=budget)
    assert [r['status'] for r in records] == ['valid', 'duplicate', 'invalid', 'invalid']
    assert budget.counts['ast'] == 4 and budget.families['one']['ast'] == 4
    assert base['outputs'][0]['arg']['window'] == 2


def test_crossover_type_domains_and_explicit_unused_parameter_declaration():
    base = program({'op': 'mul', 'left': {'op': 'last', 'arg': {'op': 'feature', 'name': 'rv_1'}},
                    'right': {'op': 'param', 'name': 'scale'}},
                    {'scale': {'initial': .5, 'lower': .1, 'upper': 1.}})
    conflicting = copy.deepcopy(base)
    conflicting['parameters']['scale']['upper'] = 2.
    budget = MonthlyBudget('2026-07')
    budget.register_family('one')
    records = materialize_mutations(base, [
        {'kind': 'crossover', 'path': ['outputs', 0, 'right'], 'donor': conflicting,
         'donor_path': ['outputs', 0, 'right']},
        {'kind': 'crossover', 'path': ['outputs', 0, 'right'], 'donor': program(),
         'donor_path': ['outputs', 0, 'arg']},
        {'kind': 'replace', 'path': ['parameters', 'scale', 'initial'], 'value': 9.},
        {'kind': 'crossover', 'path': ['outputs', 0, 'right'], 'donor': program(),
         'donor_path': ['outputs', 0]}], family_id='one', budget=budget)
    assert [r['status'] for r in records] == ['invalid', 'invalid', 'invalid', 'valid']
    assert records[-1]['removed_unused_declarations'] == ['scale']
    assert records[-1]['program']['parameters'] == {}


def test_budget_denies_before_materialization_and_no_validation_free_attempts():
    budget = MonthlyBudget('2026-07')
    budget.register_family('one')
    budget.consume('ast', 32, family_id='one')
    with pytest.raises(BudgetExhausted):
        materialize_mutations({}, [{'kind': 'replace'}], family_id='one', budget=budget)
    assert budget.counts['ast'] == 32


def test_rollover_preserves_old_shard_and_stopping_lookback(tmp_path):
    book = HypothesisBook('2026-07-01T00:00Z')
    version = book.add(spec())
    book.observe(version, month='2026-07-01T00:00Z', progress=0, evidence_sources=['e'], resolver=resolver)
    path = book.save(tmp_path / 'july.json')
    august = book.rollover('2026-08-01T00:00Z', path)
    receipt = august.observe(version, month='2026-08-01T00:00Z', progress=0,
                              evidence_sources=['e'], resolver=resolver)
    assert receipt['state'] == 'dormant'
    assert HypothesisBook.load(path).versions[version]['state'] == 'active'
    assert august.predecessor['digest'] == book.digest
    assert august.verify_history()['shard_count'] == 1
    path.write_text('{}')
    with pytest.raises((KeyError, ValueError)):
        august.verify_history()
