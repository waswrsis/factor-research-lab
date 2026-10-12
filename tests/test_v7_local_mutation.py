"""Budget failures and exact semantic identities, without market evaluation."""
import copy

import pytest

from factorlab.discovery.local_mutation import (
    BudgetExhausted,
    MonthlyBudget,
    ProgramTemplate,
    expand_template,
)
from factorlab.discovery.operator_program import VERSION, Program
from factorlab.discovery.program_syntax import CompactProgram, parse_expression


def template(values=(2, 3)):
    p = Program.from_dict({'version': VERSION, 'feature_names': ['x'], 'parameters': {},
                           'outputs': [parse_expression('last(rolling_mean(f("x"),2))')]})
    return ProgramTemplate.from_dict({'base': CompactProgram.from_program(p).to_dict(),
                                      'slots': [{'path': ['outputs', 0, 'arg', 'window'],
                                                 'values': list(values)}]})


def test_invalid_and_duplicate_variants_consume_budget_and_keep_reason():
    budget = MonthlyBudget('2026-10')
    budget.register_family('h1')
    rows = list(expand_template(template((2, 2, 0, 3)), family_id='h1', budget=budget))
    assert [r['status'] for r in rows] == ['valid', 'duplicate', 'invalid', 'valid']
    assert 'window' in rows[2]['reason']
    assert rows[0]['digest'] == rows[1]['digest']
    assert budget.counts['ast'] == 4
    assert budget.families['h1']['ast'] == 4
    assert budget.counts['f'] == budget.counts['solve'] == 0


def test_persistence_preserves_limits_consumption_and_digest():
    budget = MonthlyBudget('2026-10')
    budget.consume('gpt_attempt', 2)
    budget.consume('ast', 2)  # Invalid/unattributed seed proposals still consume quota.
    budget.register_family('h1')
    list(expand_template(template(), family_id='h1', budget=budget))
    budget.consume('f', family_id='h1')
    budget.consume('solve', 6)
    restored = MonthlyBudget.from_dict(budget.to_dict())
    assert restored.to_dict() == budget.to_dict()
    assert restored.digest == budget.digest
    changed = budget.to_dict()
    changed['limits']['solve'] -= 1
    assert MonthlyBudget.from_dict(changed).digest != budget.digest
    restored.consume('gpt_attempt', 4)
    with pytest.raises(BudgetExhausted):
        restored.consume('gpt_attempt')
    assert restored.counts['gpt_attempt'] == 6


def test_monthly_and_family_limits_are_atomic():
    b = MonthlyBudget('2026-10')
    for name in ('a', 'b', 'c'):
        b.register_family(name)
        b.consume('ast', 32, family_id=name)
        b.consume('f', 8, family_id=name)
    b.consume('solve', 1600)
    before = b.to_dict()
    for operation in [lambda: b.register_family('d'), lambda: b.consume('ast', family_id='a'),
                      lambda: b.consume('f', family_id='b'), lambda: b.consume('solve')]:
        with pytest.raises(BudgetExhausted):
            operation()
        assert b.to_dict() == before


def test_untrusted_restore_rejects_oversized_limits_or_inconsistent_counters():
    value = MonthlyBudget('2026-10').to_dict()
    for modified in [dict(value, counts=dict(value['counts'], family=1)),
                     dict(value, limits=dict(value['limits'], ast=97)),
                     dict(value, month_id='2026-13')]:
        with pytest.raises(ValueError):
            MonthlyBudget.from_dict(modified)


def test_no_cartesian_explosion_or_type_coercion():
    base = template().to_dict()
    oversized = copy.deepcopy(base)
    oversized['slots'][0]['values'] = list(range(33))
    with pytest.raises(ValueError):
        ProgramTemplate.from_dict(oversized)
    base['slots'][0]['values'] = [2.5]
    with pytest.raises(ValueError, match='type'):
        ProgramTemplate.from_dict(base)


def test_overlap_and_operator_domain_mutation_rejected():
    base = template().to_dict()
    base['slots'].append(copy.deepcopy(base['slots'][0]))
    with pytest.raises(ValueError, match='Overlapping'):
        ProgramTemplate.from_dict(base)
    base['slots'] = [{'path': ['outputs', 0, 'op'], 'values': ['mean']}]
    with pytest.raises(ValueError, match='frozen'):
        ProgramTemplate.from_dict(base)


def test_distinct_nearly_identical_constants_are_not_correlation_deduplicated():
    p = Program.from_dict({'version': VERSION, 'feature_names': ['x'], 'parameters': {},
                           'outputs': [parse_expression('add(last(f("x")),1.0)')]})
    t = ProgramTemplate.from_dict({'base': CompactProgram.from_program(p).to_dict(),
                                   'slots': [{'path': ['outputs', 0, 'right', 'value'],
                                              'values': [1., 1.0000000001]}]})
    b = MonthlyBudget('2026-10')
    b.register_family('h')
    assert [r['status'] for r in expand_template(t, family_id='h', budget=b)] == ['valid', 'valid']


def test_template_membership_checks_unmodified_fields_and_slot_values():
    t = template()
    assert t.contains(t.instantiate([1]))
    p = t.instantiate([0]).to_dict()
    p['outputs'][0]['arg']['window'] = 4
    assert not t.contains(Program.from_dict(p))
