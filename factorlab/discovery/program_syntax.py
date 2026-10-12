"""Compact transport for the existing causal Program, never executable source.

The canonical Program JSON remains the execution identity. Python's parser is
used only as a tokenizer/tree builder; no eval, imports or attribute lookup occur.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .program_contract import Program

SYNTAX_VERSION = 'compact-causal-v1'
MAX_TEXT = 20_000
BINARY = frozenset({'add', 'sub', 'mul', 'safe_div'})
UNARY = frozenset({'tanh', 'log_abs', 'sqrt_abs', 'abs', 'last', 'mean', 'std', 'max', 'min'})


def strict_json(text: str, *, max_bytes=100_000):
    if type(text) is not str or len(text.encode('utf-8')) > max_bytes:
        raise ValueError('Invalid or oversized JSON')

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError(f'Nonfinite JSON number: {value}')

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=nonfinite)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError('Invalid JSON') from exc


def parse_expression(expression: str) -> dict:
    """Parse one expression; Program.from_dict subsequently checks types/domains."""
    if type(expression) is not str or not expression.strip() or len(expression) > MAX_TEXT:
        raise ValueError('Invalid or oversized compact expression')
    try:
        tree = ast.parse(expression.strip(), mode='eval')
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ValueError('Invalid compact syntax') from exc
    if sum(1 for _ in ast.walk(tree)) > 512:
        raise ValueError('Compact syntax tree exceeds budget')

    def literal(node, kind):
        if isinstance(node, ast.Constant) and type(node.value) in kind:
            return node.value
        if (kind == (int, float) and isinstance(node, ast.UnaryOp)
                and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant)
                and type(node.operand.value) in kind):
            return -node.operand.value
        raise ValueError('Expected a literal of the declared type')

    def visit(node, depth=1):
        if depth > 12:
            raise ValueError('Compact expression exceeds depth budget')
        if isinstance(node, (ast.Constant, ast.UnaryOp)):
            return {'op': 'const', 'value': literal(node, (int, float))}
        if (not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name)
                or node.keywords):
            raise ValueError('Only registered positional function calls are allowed')
        op, args = node.func.id, node.args
        arity = 2 if op in BINARY | {'lag', 'rolling_mean', 'rolling_std', 'ema'} else 1
        if len(args) != arity:
            raise ValueError('Incorrect operator arity')
        if op in {'f', 'p'}:
            return {'op': 'feature' if op == 'f' else 'param',
                    'name': literal(args[0], (str,))}
        if op in BINARY:
            return {'op': op, 'left': visit(args[0], depth + 1),
                    'right': visit(args[1], depth + 1)}
        if op in UNARY | {'lag', 'rolling_mean', 'rolling_std', 'ema'}:
            result = {'op': op, 'arg': visit(args[0], depth + 1)}
            if op == 'ema':
                result['alpha'] = visit(args[1], depth + 1)
            elif op == 'lag' or op.startswith('rolling_'):
                result['steps' if op == 'lag' else 'window'] = literal(args[1], (int,))
            return result
        raise ValueError(f'Unregistered operator: {op}')

    return visit(tree.body)


def render_expression(node: dict) -> str:
    """Render a validated Program node without algebraic rearrangement."""
    op = node['op']
    if op in {'feature', 'param'}:
        return f'{"f" if op == "feature" else "p"}({json.dumps(node["name"])})'
    if op == 'const':
        return repr(node['value'])
    if op in BINARY:
        return f'{op}({render_expression(node["left"])},{render_expression(node["right"])})'
    suffix = (f',{node["steps"]}' if op == 'lag' else f',{node["window"]}'
              if op.startswith('rolling_') else f',{render_expression(node["alpha"])}'
              if op == 'ema' else '')
    return f'{op}({render_expression(node["arg"])}{suffix})'


@dataclass(frozen=True)
class CompactProgram:
    _canonical_json: str

    @classmethod
    def from_program(cls, program: Program):
        from .program_contract import Program
        program = Program.from_dict(program.to_dict())
        return cls(json.dumps({'syntax_version': SYNTAX_VERSION,
                              'feature_names': list(program.feature_names),
                              'expressions': [render_expression(n) for n in program.outputs],
                              'parameters': program.parameters},
                             sort_keys=True, separators=(',', ':'), allow_nan=False))

    @classmethod
    def from_dict(cls, value: dict):
        from .program_contract import VERSION, Program
        if type(value) is not dict or set(value) != {
            'syntax_version', 'feature_names', 'expressions', 'parameters'
        } or value['syntax_version'] != SYNTAX_VERSION:
            raise ValueError('Invalid compact program schema/version')
        roots = value['expressions']
        if type(roots) is not list or not 1 <= len(roots) <= 16:
            raise ValueError('Invalid compact output count')
        program = Program.from_dict({'version': VERSION, 'feature_names': value['feature_names'],
                                     'parameters': value['parameters'],
                                     'outputs': [parse_expression(e) for e in roots]})
        return cls.from_program(program)

    def to_dict(self):
        return strict_json(self._canonical_json)

    def to_program(self) -> Program:
        from .program_contract import VERSION, Program
        value = self.to_dict()
        return Program.from_dict({'version': VERSION, 'feature_names': value['feature_names'],
                                  'parameters': value['parameters'],
                                  'outputs': [parse_expression(e) for e in value['expressions']]})

    @property
    def digest(self):
        return self.to_program().digest


def compact_program(program: Program) -> dict:
    return CompactProgram.from_program(program).to_dict()


def parse_compact_program(value: dict) -> Program:
    return CompactProgram.from_dict(value).to_program()
