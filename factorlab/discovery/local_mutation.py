"""Finite, recorded local mutations and shared V7 monthly research limits.

Every materialized variant consumes budget, including invalid and duplicate ones.
No label scoring, correlation filtering, semantic repair or parameter clipping.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .program_syntax import CompactProgram

if TYPE_CHECKING:
    from .program_contract import Program


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _id(value):
    if type(value) is not str or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', value):
        raise ValueError('Expected a bounded identifier')
    return value


class BudgetExhausted(ValueError):
    pass


@dataclass
class MonthlyBudget:
    """Serializable single-owner ledger; callers atomically persist after spending.

    Reservations precede work and are never refunded. It is intentionally not a
    concurrent database; the run controller owns locking and durable recovery.
    """
    month_id: str
    limits: dict = field(default_factory=lambda: {
        'gpt_attempt': 6, 'family': 3, 'ast': 96, 'f': 24, 'solve': 1600,
        'family_ast': 32, 'family_f': 8})
    counts: dict = field(default_factory=lambda: {
        'gpt_attempt': 0, 'family': 0, 'ast': 0, 'f': 0, 'solve': 0})
    families: dict = field(default_factory=dict)

    def __post_init__(self):
        if type(self.month_id) is not str or not re.fullmatch(r'\d{4}-(0[1-9]|1[0-2])', self.month_id):
            raise ValueError('month_id must be YYYY-MM UTC')
        maxima = {'gpt_attempt': 6, 'family': 3, 'ast': 96, 'f': 24, 'solve': 1600,
                  'family_ast': 32, 'family_f': 8}
        if set(self.limits) != set(maxima) or any(
            type(v) is not int or not 1 <= v <= maxima[k] for k, v in self.limits.items()
        ):
            raise ValueError('Invalid limits or limits exceed V7 contract')
        keys = set(maxima) - {'family_ast', 'family_f'}
        if set(self.counts) != keys or any(
            type(v) is not int or not 0 <= v <= self.limits[k] for k, v in self.counts.items()
        ):
            raise ValueError('Invalid budget counters')
        if type(self.families) is not dict or len(self.families) != self.counts['family']:
            raise ValueError('Inconsistent family count')
        for name, counts in self.families.items():
            _id(name)
            if type(counts) is not dict or set(counts) != {'ast', 'f'} or any(
                type(v) is not int or not 0 <= v <= self.limits[f'family_{k}']
                for k, v in counts.items()
            ):
                raise ValueError('Invalid family counters')
        if (sum(c['ast'] for c in self.families.values()) > self.counts['ast']
                or sum(c['f'] for c in self.families.values()) != self.counts['f']):
            raise ValueError('Family and global counters disagree')

    def register_family(self, family_id):
        self.__post_init__()
        _id(family_id)
        if family_id in self.families:
            raise ValueError('Family already registered')
        if self.counts['family'] >= self.limits['family']:
            raise BudgetExhausted('Monthly family limit')
        self.families[family_id] = {'ast': 0, 'f': 0}
        self.counts['family'] += 1

    def consume(self, kind, amount=1, *, family_id=None):
        self.__post_init__()
        if kind not in {'gpt_attempt', 'ast', 'f', 'solve'} or type(amount) is not int or amount < 1:
            raise ValueError('Invalid budget reservation')
        if self.counts[kind] + amount > self.limits[kind]:
            raise BudgetExhausted(f'Monthly {kind} limit')
        if kind == 'ast' and family_id is None:
            # Reserve untrusted seed ASTs before validation. Failed responses have
            # no successful family, but their concrete proposals still cost ASTs.
            pass
        elif kind in {'ast', 'f'}:
            if family_id not in self.families:
                raise ValueError('Register family before candidate work')
            if self.families[family_id][kind] + amount > self.limits[f'family_{kind}']:
                raise BudgetExhausted(f'Family {kind} limit')
            self.families[family_id][kind] += amount
        elif family_id is not None:
            raise ValueError('Family attribution applies to ast/f reservations only')
        self.counts[kind] += amount

    def attribute_ast(self, family_id, amount):
        """Attach already-reserved seed ASTs to a successfully validated family."""
        self.__post_init__()
        if (family_id not in self.families or type(amount) is not int or amount < 1
                or amount > self.counts['ast'] - sum(c['ast'] for c in self.families.values())):
            raise ValueError('Invalid AST attribution')
        if self.families[family_id]['ast'] + amount > self.limits['family_ast']:
            raise BudgetExhausted('Family ast limit')
        self.families[family_id]['ast'] += amount

    def to_dict(self):
        self.__post_init__()
        return copy.deepcopy({'version': 'v7-monthly-budget-v1', 'month_id': self.month_id,
                              'limits': self.limits, 'counts': self.counts,
                              'families': self.families})

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or set(value) != {
            'version', 'month_id', 'limits', 'counts', 'families'
        } or value['version'] != 'v7-monthly-budget-v1':
            raise ValueError('Invalid monthly budget schema')
        return cls(**copy.deepcopy({k: v for k, v in value.items() if k != 'version'}))

    @property
    def digest(self):
        return _hash(self.to_dict())


def _get(record, path):
    try:
        for component in path:
            if type(record) is list:
                if type(component) is not int or not 0 <= component < len(record):
                    raise ValueError('Invalid template index')
            elif type(record) is not dict or type(component) is not str or component not in record:
                raise ValueError('Invalid template field')
            record = record[component]
        return record
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError('Invalid template path') from exc


def _put(record, path, value):
    _get(record, path)
    parent = _get(record, path[:-1])
    parent[path[-1]] = copy.deepcopy(value)


@dataclass(frozen=True)
class ProgramTemplate:
    """Slots target canonical AST leaves/subtrees, with finite explicit choices."""
    _json: str

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or set(value) != {'base', 'slots'}:
            raise ValueError('Template requires base and slots')
        base = CompactProgram.from_dict(value['base'])
        slots = value['slots']
        if type(slots) is not list or len(slots) > 4:
            raise ValueError('Template has too many slots')
        paths = []
        size = 1
        record = base.to_program().to_dict()
        for slot in slots:
            if type(slot) is not dict or set(slot) != {'path', 'values'}:
                raise ValueError('Invalid template slot')
            path, choices = slot['path'], slot['values']
            if (type(path) is not list or not 2 <= len(path) <= 16
                    or path[0] not in {'outputs', 'parameters'}):
                raise ValueError('Only existing output/parameter paths can vary')
            original = _get(record, path)
            if path[0] == 'parameters' and (len(path) != 3 or path[-1] != 'initial'):
                raise ValueError('Only parameter initial values may vary; domains are frozen')
            if path[-1] == 'op' or path[-1] in {'lower', 'upper'}:
                raise ValueError('Operator names and parameter domains are frozen')
            if type(choices) is not list or not 1 <= len(choices) <= 32:
                raise ValueError('Slot choices must be bounded')
            if any(type(v) is not type(original) for v in choices):
                raise ValueError('Slot must preserve the exact AST field type')
            if any(tuple(path[:len(p)]) == p or tuple(p[:len(path)]) == tuple(path) for p in paths):
                raise ValueError('Overlapping template paths')
            paths.append(tuple(path))
            size *= len(choices)
        if size > 32:
            raise ValueError('Template exceeds 32 materializations per family')
        try:
            serialized = json.dumps({'base': base.to_dict(), 'slots': slots}, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False)
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError('Invalid template values') from exc
        if len(serialized.encode()) > 100_000:
            raise ValueError('Template exceeds byte budget')
        return cls(serialized)

    def to_dict(self):
        return json.loads(self._json)

    @property
    def size(self):
        return math.prod(len(s['values']) for s in self.to_dict()['slots'])

    def instantiate(self, indices) -> Program:
        from .program_contract import Program
        value = self.to_dict()
        if len(indices) != len(value['slots']):
            raise ValueError('Incorrect number of template choices')
        record = CompactProgram.from_dict(value['base']).to_program().to_dict()
        for index, slot in zip(indices, value['slots'], strict=True):
            if type(index) is not int or not 0 <= index < len(slot['values']):
                raise ValueError('Invalid template choice')
            _put(record, slot['path'], slot['values'][index])
        return Program.from_dict(record)

    def contains(self, program: Program) -> bool:
        from .program_contract import Program
        value = self.to_dict()
        base = CompactProgram.from_dict(value['base']).to_program().to_dict()
        candidate = program.to_dict()
        for slot in value['slots']:
            actual = _get(candidate, slot['path'])
            if not any(_hash(actual) == _hash(choice) for choice in slot['values']):
                return False
            _put(base, slot['path'], actual)
        return Program.from_dict(base).digest == program.digest


def expand_template(template: ProgramTemplate, *, family_id: str,
                    budget: MonthlyBudget, seen: set[str] | None = None):
    """Yield audit records, reserving one AST before each attempted construction.

    ``seen`` contains exact Program digests, never output correlations or scores.
    Controller must persist reservations/results before proceeding to evaluation.
    """
    template = ProgramTemplate.from_dict(template.to_dict())
    seen = set() if seen is None else seen
    slots = template.to_dict()['slots']
    for indices in itertools.product(*(range(len(s['values'])) for s in slots)):
        budget.consume('ast', family_id=family_id)
        try:
            program = template.instantiate(indices)
            if len(program.outputs) != 1:
                raise ValueError('V7 requires one scalar output per formula')
        except ValueError as exc:
            yield {'family_id': family_id, 'choices': list(indices), 'status': 'invalid',
                   'reason': str(exc), 'program': None, 'digest': None}
            continue
        duplicate = program.digest in seen
        seen.add(program.digest)
        yield {'family_id': family_id, 'choices': list(indices),
               'status': 'duplicate' if duplicate else 'valid', 'reason': None,
               'program': program.to_dict(), 'digest': program.digest}
