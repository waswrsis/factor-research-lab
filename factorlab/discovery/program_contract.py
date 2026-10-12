"""Torch-free trusted syntax contract for the existing causal formula language.

This is a bounded expression language, not arbitrary Python execution or an OS
sandbox. New formulas compose primitives; they are not selected from an indicator
catalog. Inputs are floating [batch, past_time, feature] windows. Outputs are
[batch, number_of_scalar_roots]. All reductions are within one sample. Leading
positions without enough history are excluded, never padded into aggregates.
"""
from __future__ import annotations

import copy
import ctypes
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

VERSION = 'causal-operator-program-v1'
MAX_NODES = 64
MAX_DEPTH = 12
MAX_OUTPUTS = 16
MAX_LOOKBACK = 512
MAX_COST = 4096
EPSILON = 1e-6
AMPLITUDE_BOUND = 1e6
_BINARY = {'add', 'sub', 'mul', 'safe_div'}
_UNARY = {'tanh', 'log_abs', 'sqrt_abs', 'abs'}
_REDUCE = {'last', 'mean', 'std', 'max', 'min'}


@lru_cache(maxsize=1)
def _native_expf():
    # The old scalar CPU kernel uses the platform float exp, not NumPy's
    # vectorized exp approximation. Their one-ULP differences can change the
    # strict representability decision for very narrow parameter intervals.
    name = ('ucrtbase.dll' if sys.platform == 'win32' else
            '/usr/lib/libSystem.B.dylib' if sys.platform == 'darwin' else 'libm.so.6')
    try:
        function = ctypes.CDLL(name).expf
    except (OSError, AttributeError) as exc:
        raise RuntimeError('Native scalar expf unavailable; float32 contract cannot be verified') from exc
    function.argtypes = (ctypes.c_float,)
    function.restype = ctypes.c_float
    return function


def float32_parameter_initial(lower: float, upper: float, ratio: float) -> float:
    """Emulate the old CPU scalar float32 sigmoid and separate affine operations.

    Each operation rounds to float32 as the old scalar Torch tensor did. In
    particular converting the bounds only AFTER the affine expression would
    silently accept parameters that the trainable float32 representation cannot
    place strictly inside their original double-precision interval.
    """
    logit = np.float32(math.log(ratio / (1 - ratio)))
    exponential = np.float32(_native_expf()(float(-logit)))
    sigmoid = np.float32(1.) / np.float32(np.float32(1.) + exponential)
    scaled = np.float32(upper - lower) * sigmoid
    return float(np.float32(lower) + scaled)


def _number(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{label} must be a finite number')
    if abs(value) > AMPLITUDE_BOUND:
        raise ValueError(f'{label} exceeds amplitude bound')
    return float(value)


def _fields(value: Any, expected: set[str], label: str) -> None:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f'{label} requires exactly {sorted(expected)}')


def _integer(value: Any, low: int, high: int, label: str) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{label} must be an integer in [{low}, {high}]')
    return value


@dataclass(frozen=True)
class Program:
    """Validated JSON program; use ``from_dict`` for an untrusted proposal.

    Parameter initial values must be strictly inside finite bounds. Validation
    is repeated at module construction, so direct dataclass construction cannot
    bypass it. Serialized dictionaries are defensive copies.
    """

    feature_names: tuple[str, ...]
    outputs: tuple[dict, ...]
    parameters: dict[str, dict[str, float]]
    version: str = VERSION

    @classmethod
    def from_dict(cls, value: dict) -> Program:
        _fields(value, {'version', 'feature_names', 'outputs', 'parameters'}, 'program')
        if value['version'] != VERSION:
            raise ValueError('Unknown operator program version')
        features = value['feature_names']
        if (type(features) is not list or not 1 <= len(features) <= 256
                or any(type(f) is not str or not f or len(f) > 128 for f in features)
                or len(set(features)) != len(features)):
            raise ValueError('feature_names must be unique nonempty strings')
        roots = value['outputs']
        if type(roots) is not list or not 1 <= len(roots) <= MAX_OUTPUTS:
            raise ValueError('Invalid output count')
        parameters = value['parameters']
        if type(parameters) is not dict or len(parameters) > MAX_NODES:
            raise ValueError('Invalid parameters')
        specs = {}
        for name, spec in parameters.items():
            if (type(name) is not str or not name.isidentifier()
                    or len(name) > 64 or '.' in name):
                raise ValueError('Invalid parameter name')
            _fields(spec, {'initial', 'lower', 'upper'}, 'parameter')
            clean = {k: _number(v, f'parameter {k}') for k, v in spec.items()}
            if not clean['lower'] < clean['initial'] < clean['upper']:
                raise ValueError('Parameter initial must be strictly inside bounds')
            ratio = (clean['initial'] - clean['lower']) / (clean['upper'] - clean['lower'])
            if not 1e-6 < ratio < 1 - 1e-6:
                raise ValueError('Parameter initial is too close to a bound for stable training')
            effective = float32_parameter_initial(clean['lower'], clean['upper'], ratio)
            if not clean['lower'] < float(effective) < clean['upper']:
                raise ValueError('Parameter initial is not representable inside float32 bounds')
            specs[name] = clean
        program = cls(tuple(features), tuple(copy.deepcopy(roots)), specs)
        program._analyze()
        return program

    @classmethod
    def from_json(cls, value: str) -> Program:
        if type(value) is not str or len(value) > 100_000:
            raise ValueError('Invalid or oversized program JSON')

        def unique_pairs(pairs):
            result = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError('Duplicate JSON key')
                result[key] = item
            return result

        try:
            decoded = json.loads(value, object_pairs_hook=unique_pairs)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise ValueError('Invalid program JSON') from exc
        return cls.from_dict(decoded)

    def to_dict(self) -> dict:
        return copy.deepcopy({'version': self.version, 'feature_names': list(self.feature_names),
                              'outputs': list(self.outputs), 'parameters': self.parameters})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode('utf-8')).hexdigest()

    def _analyze(self) -> dict:
        count = 0
        cost = 0
        used = set()

        def visit(node, depth=1):
            nonlocal count, cost
            count += 1
            cost += 1
            if count > MAX_NODES or depth > MAX_DEPTH:
                raise ValueError('Node/depth budget exceeded')
            if type(node) is not dict or type(node.get('op')) is not str:
                raise ValueError('Every node requires a string op')
            op = node['op']
            if op == 'feature':
                _fields(node, {'op', 'name'}, op)
                if type(node['name']) is not str or node['name'] not in self.feature_names:
                    raise ValueError('Unknown feature')
                return 'series', 1
            if op == 'const':
                _fields(node, {'op', 'value'}, op)
                _number(node['value'], 'constant')
                return 'scalar', 1
            if op == 'param':
                _fields(node, {'op', 'name'}, op)
                if type(node['name']) is not str or node['name'] not in self.parameters:
                    raise ValueError('Unknown parameter')
                used.add(node['name'])
                return 'scalar', 1
            if op in _BINARY:
                _fields(node, {'op', 'left', 'right'}, op)
                left, lh = visit(node['left'], depth + 1)
                right, rh = visit(node['right'], depth + 1)
                return ('series' if 'series' in (left, right) else 'scalar'), max(lh, rh)
            if op in _UNARY | _REDUCE | {'lag', 'rolling_mean', 'rolling_std', 'ema'}:
                extra = ({'steps'} if op == 'lag' else {'window'} if op.startswith('rolling_')
                         else {'alpha'} if op == 'ema' else set())
                _fields(node, {'op', 'arg'} | extra, op)
                kind, history = visit(node['arg'], depth + 1)
                if op in _UNARY:
                    return kind, history
                if kind != 'series':
                    raise ValueError(f'{op} requires a series')
                if op in _REDUCE:
                    return 'scalar', history
                if op == 'lag':
                    history += _integer(node['steps'], 0, MAX_LOOKBACK - 1, 'lag steps')
                elif op.startswith('rolling_'):
                    window = _integer(node['window'], 1, MAX_LOOKBACK, 'rolling window')
                    history += window - 1
                    cost += window
                else:
                    alpha_kind, _ = visit(node['alpha'], depth + 1)
                    if alpha_kind != 'scalar':
                        raise ValueError('ema alpha must be scalar')
                    if not self._unit_interval(node['alpha']):
                        raise ValueError('ema alpha must be const or bounded param in (0, 1]')
                if history > MAX_LOOKBACK:
                    raise ValueError('Lookback budget exceeded')
                return 'series', history
            raise ValueError(f'Unknown operator: {op}')

        histories = []
        for root in self.outputs:
            kind, history = visit(root)
            if kind != 'scalar':
                raise ValueError('Output roots must be scalar; use last or another aggregate')
            histories.append(history)
        if cost > MAX_COST:
            raise ValueError('Operation cost budget exceeded')
        if used != set(self.parameters):
            raise ValueError('Unused parameter declaration')
        return {'nodes': count, 'cost': cost, 'required_lookback': max(histories)}

    def _unit_interval(self, node):
        if node['op'] == 'const':
            return 0 < node['value'] <= 1
        if node['op'] == 'param':
            spec = self.parameters[node['name']]
            return 0 <= spec['lower'] < spec['upper'] <= 1
        return False

    @property
    def required_lookback(self) -> int:
        """Minimum input length, not a maximum dependency horizon.

        EMA and whole-window aggregates use all supplied valid past positions.
        Lag and rolling consume additional history; insufficient windows fail.
        """
        return self._analyze()['required_lookback']

    @property
    def complexity(self) -> dict:
        return self._analyze()

    @property
    def formulas(self) -> tuple[str, ...]:
        def render(node):
            op = node['op']
            if op == 'feature':
                return f'feature[{node["name"]!r}]'
            if op == 'param':
                return f'parameter[{node["name"]!r}]'
            if op == 'const':
                return repr(node['value'])
            if op in _BINARY:
                return f'{op}({render(node["left"])}, {render(node["right"])})'
            suffix = (f', {node["steps"]}' if op == 'lag' else f', {node["window"]}'
                      if op.startswith('rolling_') else f', {render(node["alpha"])}'
                      if op == 'ema' else '')
            return f'{op}({render(node["arg"])}{suffix})'
        self._analyze()
        return tuple(render(root) for root in self.outputs)

    @property
    def source(self) -> str:
        """Trusted compiler's readable listing, for review only; never executed."""
        return '\n'.join(f'factor_{i} = {formula}' for i, formula in enumerate(self.formulas))

def validate_program(value: Program | dict, features=None) -> dict:
    """Return canonical defensive JSON data; optionally require exact input order."""
    program = Program.from_dict(value.to_dict() if isinstance(value, Program) else value)
    if features is not None and tuple(features) != program.feature_names:
        raise ValueError('Feature order differs from program schema')
    return program.to_dict()


def program_language_spec() -> dict:
    """Compact proposal-worker grammar, including a valid learnable example."""
    return {
        'version': VERSION,
        'top_level': ['version', 'feature_names', 'parameters', 'outputs'],
        'input': 'float32/float64 [B,L,F], past-only; feature_names is exact F order',
        'output': '[B,K], K scalar roots; wrap a series in last/mean/std/max/min',
        'nodes': {
            'feature': {'op': 'feature', 'name': 'declared feature name'},
            'const': {'op': 'const', 'value': 'finite number'},
            'param': {'op': 'param', 'name': 'declared parameter name'},
            'binary': {'op': sorted(_BINARY), 'left': 'node', 'right': 'node'},
            'unary': {'op': sorted(_UNARY), 'arg': 'node'},
            'aggregate': {'op': sorted(_REDUCE), 'arg': 'series node'},
            'lag': {'op': 'lag', 'arg': 'series node', 'steps': 'integer >= 0'},
            'rolling': {'op': ['rolling_mean', 'rolling_std'], 'arg': 'series node',
                        'window': 'integer >= 1'},
            'ema': {'op': 'ema', 'arg': 'series node',
                    'alpha': 'const in (0,1] or param with bounds inside [0,1]'},
        },
        'parameter': {'initial': 'lower < initial < upper', 'lower': 'finite bound',
                      'upper': 'finite bound'},
        'semantics': {'std': 'population standard deviation (correction=0)',
                      'log_abs': 'log1p(abs(arg))', 'sqrt_abs': 'sqrt(abs(arg)+epsilon)',
                      'safe_div': 'preserve denominator sign; magnitude >= epsilon',
                      'ema_seed': 'first valid past value',
                      'warmup': 'incomplete leading history excluded from aggregates'},
        'limits': {'nodes': MAX_NODES, 'depth': MAX_DEPTH, 'outputs': MAX_OUTPUTS,
                   'lookback': MAX_LOOKBACK, 'cost': MAX_COST, 'epsilon': EPSILON,
                   'amplitude': AMPLITUDE_BOUND},
        'example': {'version': VERSION, 'feature_names': ['x'],
                    'parameters': {'alpha': {'initial': 0.5, 'lower': 0.0, 'upper': 1.0}},
                    'outputs': [{'op': 'last', 'arg': {'op': 'ema',
                                 'arg': {'op': 'feature', 'name': 'x'},
                                 'alpha': {'op': 'param', 'name': 'alpha'}}}]},
    }

