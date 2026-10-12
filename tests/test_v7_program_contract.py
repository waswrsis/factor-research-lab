"""Differential syntax and float32 boundary evidence against the frozen Torch API."""
import copy
import math
import subprocess
import sys

import numpy as np
import pytest
import torch

from factorlab.discovery import operator_program as legacy
from factorlab.discovery import program_contract as contract


def record(root=None, parameter=None):
    return {'version': contract.VERSION, 'feature_names': ['x', 'y'],
            'outputs': [root or {'op': 'last', 'arg': {'op': 'feature', 'name': 'x'}}],
            'parameters': {} if parameter is None else {'p': parameter}}


def outcome(parser, value):
    try:
        result = parser.from_dict(value)
        return ('accepted', result.to_json(), result.digest, result.complexity, result.formulas, result.source)
    except ValueError as exc:
        return ('rejected', str(exc))


def test_contract_import_parse_and_grammar_do_not_load_torch():
    code = ('import sys; from factorlab.discovery.program_contract import Program,program_language_spec; '
            'Program.from_dict(program_language_spec()["example"]); assert "torch" not in sys.modules')
    child = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, check=False)
    assert child.returncode == 0, child.stderr


def test_every_operator_and_serialized_identity_match_legacy():
    feature = {'op': 'feature', 'name': 'x'}
    series = [feature]
    series += [{'op': op, 'arg': feature} for op in ('tanh', 'log_abs', 'sqrt_abs', 'abs')]
    series += [{'op': op, 'left': feature, 'right': {'op': 'const', 'value': 2.}}
               for op in ('add', 'sub', 'mul', 'safe_div')]
    series += [{'op': 'lag', 'arg': feature, 'steps': 511},
               {'op': 'rolling_mean', 'arg': feature, 'window': 512},
               {'op': 'rolling_std', 'arg': feature, 'window': 2},
               {'op': 'ema', 'arg': feature, 'alpha': {'op': 'const', 'value': .4}}]
    for child in series:
        for aggregate in ('last', 'mean', 'std', 'min', 'max'):
            value = record({'op': aggregate, 'arg': child})
            assert outcome(contract.Program, value) == outcome(legacy.Program, value)
    assert contract.program_language_spec() == legacy.program_language_spec()
    parsed = contract.Program.from_dict(contract.program_language_spec()['example'])
    assert contract.Program.from_json(parsed.to_json()).digest == parsed.digest
    assert contract.validate_program(parsed, ['x']) == parsed.to_dict()
    with pytest.raises(ValueError, match='Feature order'):
        contract.validate_program(parsed, ['y'])


@pytest.mark.parametrize('root', [
    {'op': 'python', 'source': 'not executable'},
    {'op': 'const', 'value': True},
    {'op': 'const', 'value': float('nan')},
    {'op': 'const', 'value': 1000001},
    {'op': 'mean', 'arg': {'op': 'const', 'value': 1}},
    {'op': 'last', 'arg': {'op': 'feature', 'name': 'unknown'}},
    {'op': 'last', 'arg': {'op': 'lag', 'arg': {'op': 'feature', 'name': 'x'}, 'steps': -1}},
    {'op': 'last', 'arg': {'op': 'rolling_mean', 'arg': {'op': 'feature', 'name': 'x'}, 'window': 2.5}},
    {'op': 'last', 'arg': {'op': 'rolling_mean', 'arg': {'op': 'feature', 'name': 'x'}, 'window': True}},
    {'op': 'last', 'arg': {'op': 'ema', 'arg': {'op': 'feature', 'name': 'x'},
                          'alpha': {'op': 'const', 'value': 0}}},
    {'op': 'feature', 'name': 'x'},
    {'op': 'const', 'value': 1, 'extra': 0},
])
def test_invalid_ast_rejections_preserve_exact_messages(root):
    value = record(root)
    assert outcome(contract.Program, value) == outcome(legacy.Program, value)
    assert outcome(contract.Program, value)[0] == 'rejected'


def test_scalar_expf_and_float32_affine_match_20001_legacy_initials_exactly():
    rng = np.random.default_rng(662)
    ratios = np.r_[np.linspace(1.000001e-6, 1-1.000001e-6, 10001),
                   rng.uniform(1e-6, 1-1e-6, 10000)]
    for ratio in ratios:
        # Deliberately scalar: the legacy validator constructs one scalar tensor.
        old = torch.sigmoid(torch.tensor(math.log(ratio/(1-ratio)), dtype=torch.float32))
        assert contract.float32_parameter_initial(0., 1., ratio) == float(old)
    for lower, upper in [(-1e6, 1e6), (1., 1.+1e-7), (-1e-20, 1e-20), (.1, .9)]:
        for ratio in ratios[::137]:
            old = lower + (upper-lower) * torch.sigmoid(
                torch.tensor(math.log(ratio/(1-ratio)), dtype=torch.float32))
            assert contract.float32_parameter_initial(lower, upper, ratio) == float(old)


def test_parameter_acceptance_narrow_float32_and_ratio_edges_match():
    intervals = [(0., 1.), (1., 1.+1e-8), (1., 1.+1e-7), (-1e6, 1e6),
                 (0., 1e-45), (-1e-45, 1e-45), (.1, .10000001)]
    ratios = [0., 1e-6, np.nextafter(1e-6, 1.), .1415271860463708, .5,
              np.nextafter(1-1e-6, 0.), 1-1e-6, 1.]
    for lo, hi in intervals:
        for ratio in ratios:
            value = record({'op': 'param', 'name': 'p'},
                           {'initial': float(lo+(hi-lo)*ratio), 'lower': lo, 'upper': hi})
            assert outcome(contract.Program, value) == outcome(legacy.Program, value)


def test_json_duplicates_and_defensive_copies_match():
    for text in ['{"version":1,"version":2}', '[' * 1200, '{}', 'x', ' ' * 100001]:
        outcomes = []
        for parser in (contract.Program, legacy.Program):
            with pytest.raises(ValueError) as failure:
                parser.from_json(text)
            outcomes.append(str(failure.value))
        assert outcomes[0] == outcomes[1]
    value = contract.program_language_spec()['example']
    original = copy.deepcopy(value)
    parsed = contract.Program.from_dict(value)
    value['outputs'][0]['op'] = 'bad'
    assert parsed.to_dict() == original
    returned = parsed.to_dict()
    returned['parameters']['alpha']['upper'] = 999.
    assert parsed.to_dict() == original
