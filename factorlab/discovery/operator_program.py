"""Typed, differentiable causal formulas interpreted by trusted Torch operations.

This is a bounded expression language, not arbitrary Python execution or an OS
sandbox. New formulas compose primitives; they are not selected from an indicator
catalog. Inputs are floating [batch, past_time, feature] windows. Outputs are
[batch, number_of_scalar_roots]. All reductions are within one sample. Leading
positions without enough history are excluded, never padded into aggregates.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

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
            effective = clean['lower'] + (clean['upper'] - clean['lower']) * torch.sigmoid(
                torch.tensor(math.log(ratio / (1 - ratio)), dtype=torch.float32))
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


@dataclass
class _Value:
    data: torch.Tensor
    series: bool = False
    start: int = 0


class ProgramModule(nn.Module):
    """Differentiable evaluator with bounded learnable parameters.

    ``end_index`` is a zero-based inclusive prediction origin. The evaluator
    slices before validating/evaluating, so appended future observations cannot
    affect that origin. Nonfinite consumed input or intermediates fail explicitly;
    finite amplitudes are clipped to the public AMPLITUDE_BOUND at every node.
    """

    def __init__(self, program: Program | dict, feature_names=None, lookback=None):
        super().__init__()
        self.program = Program.from_dict(program.to_dict() if isinstance(program, Program)
                                         else program)
        if feature_names is not None and tuple(feature_names) != self.program.feature_names:
            raise ValueError('Feature order differs from program schema')
        if lookback is not None:
            _integer(lookback, self.program.required_lookback, MAX_LOOKBACK, 'lookback')
        self.raw_parameters = nn.ParameterDict()
        self._parameter_keys = {}
        for index, (name, spec) in enumerate(sorted(self.program.parameters.items())):
            key = f'p{index}'
            self._parameter_keys[name] = key
            ratio = (spec['initial'] - spec['lower']) / (spec['upper'] - spec['lower'])
            self.raw_parameters[key] = nn.Parameter(torch.tensor(math.log(ratio / (1 - ratio))))

    @property
    def output_dim(self) -> int:
        return len(self.program.outputs)

    def parameter_values(self) -> dict[str, torch.Tensor]:
        return {name: spec['lower'] + (spec['upper'] - spec['lower']) * torch.sigmoid(raw)
                for name, key in self._parameter_keys.items()
                for raw, spec in [(self.raw_parameters[key], self.program.parameters[name])]}

    def to_program(self) -> Program:
        """Export fitted parameter initials while preserving expression semantics."""
        record = self.program.to_dict()
        for name, value in self.parameter_values().items():
            spec = record['parameters'][name]
            # A saturated machine value must not produce an invalid exported spec.
            initial = float(value.detach().cpu())
            margin = (spec['upper'] - spec['lower']) * 1e-5
            spec['initial'] = min(spec['upper'] - margin,
                                  max(spec['lower'] + margin, initial))
        return Program.from_dict(record)

    def forward(self, inputs: torch.Tensor, *, end_index: int | None = None) -> torch.Tensor:
        if (not isinstance(inputs, torch.Tensor) or inputs.ndim != 3
                or inputs.dtype not in (torch.float32, torch.float64) or inputs.shape[0] < 1
                or inputs.shape[2] != len(self.program.feature_names)):
            raise ValueError('Expected floating inputs [B,L,F] matching feature_names')
        if end_index is not None:
            _integer(end_index, 0, inputs.shape[1] - 1, 'end_index')
            inputs = inputs[:, :end_index + 1]
        if not self.program.required_lookback <= inputs.shape[1] <= MAX_LOOKBACK:
            raise ValueError('Input window violates lookback bounds')
        if not torch.isfinite(inputs).all():
            raise ValueError('Consumed inputs contain nonfinite values')
        if any(not torch.isfinite(raw).all() for raw in self.raw_parameters.values()):
            raise ValueError('Nonfinite raw learned parameter')
        values = self.parameter_values()
        for value in values.values():
            if not torch.isfinite(value).all():
                raise ValueError('Nonfinite learned parameter')
        inputs = inputs.clamp(-AMPLITUDE_BOUND, AMPLITUDE_BOUND)

        def checked(data, series=False, start=0):
            if not torch.isfinite(data).all():
                raise ValueError('Nonfinite operator result')
            return _Value(data.clamp(-AMPLITUDE_BOUND, AMPLITUDE_BOUND), series, start)

        def evaluate(node):
            op = node['op']
            if op == 'feature':
                return _Value(inputs[:, :, self.program.feature_names.index(node['name'])], True)
            if op in {'const', 'param'}:
                value = (inputs.new_tensor(node['value']) if op == 'const'
                         else values[node['name']].to(inputs))
                return checked(value.expand(inputs.shape[0]))
            if op in _BINARY:
                left, right = evaluate(node['left']), evaluate(node['right'])
                series = left.series or right.series
                a = left.data.unsqueeze(1) if series and not left.series else left.data
                b = right.data.unsqueeze(1) if series and not right.series else right.data
                if op == 'add':
                    result = a + b
                elif op == 'sub':
                    result = a - b
                elif op == 'mul':
                    result = a * b
                else:
                    denominator = torch.where(b < 0, -b.abs().clamp_min(EPSILON),
                                              b.abs().clamp_min(EPSILON))
                    result = a / denominator
                return checked(result, series, max(left.start, right.start))
            arg = evaluate(node['arg'])
            if op in _UNARY:
                result = (torch.tanh(arg.data) if op == 'tanh' else
                          torch.log1p(arg.data.abs()) if op == 'log_abs' else
                          torch.sqrt(arg.data.abs() + EPSILON) if op == 'sqrt_abs'
                          else arg.data.abs())
                return checked(result, arg.series, arg.start)
            valid = arg.data[:, arg.start:]
            if op in _REDUCE:
                result = (valid[:, -1] if op == 'last' else valid.mean(1) if op == 'mean'
                          else valid.std(1, correction=0) if op == 'std'
                          else valid.max(1).values if op == 'max' else valid.min(1).values)
                return checked(result)
            if op == 'lag':
                steps = node['steps']
                result = arg.data if steps == 0 else torch.cat(
                    (torch.zeros_like(arg.data[:, :steps]), arg.data[:, :-steps]), dim=1)
                return checked(result, True, arg.start + steps)
            if op.startswith('rolling_'):
                window = node['window']
                patches = valid.unfold(1, window, 1)
                result = (patches.mean(-1) if op == 'rolling_mean'
                          else patches.std(-1, correction=0))
                start = arg.start + window - 1
                result = torch.cat((torch.zeros_like(arg.data[:, :start]), result), dim=1)
                return checked(result, True, start)
            alpha = evaluate(node['alpha']).data[:, None]
            if not ((alpha > 0) & (alpha <= 1)).all():
                raise ValueError('ema alpha is outside (0, 1]')
            terms = [valid[:, :1]]
            for step in range(1, valid.shape[1]):
                terms.append(alpha * valid[:, step:step + 1] + (1 - alpha) * terms[-1])
            result = torch.cat((torch.zeros_like(arg.data[:, :arg.start]), *terms), dim=1)
            return checked(result, True, arg.start)

        return torch.stack([evaluate(root).data for root in self.program.outputs], dim=1)


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
