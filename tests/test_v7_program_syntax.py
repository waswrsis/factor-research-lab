"""Artificial execution-equivalence and parser rejection controls."""
import copy
import json
import subprocess
import sys

import pytest
import torch

from factorlab.discovery.operator_program import VERSION, Program, ProgramModule
from factorlab.discovery.program_syntax import CompactProgram, parse_expression, strict_json


def program():
    return Program.from_dict({
        'version': VERSION, 'feature_names': ['x', 'y'],
        'parameters': {'a': {'initial': .312345678901, 'lower': 0., 'upper': 1.}},
        'outputs': [parse_expression(e) for e in [
            'last(ema(rolling_mean(f("x"),3),p("a")))',
            'sub(mean(lag(f("x"),1)),safe_div(std(f("y")),-0.0))',
            'mul(max(abs(f("x"))),min(tanh(f("y"))))',
            'add(log_abs(-2),sqrt_abs(last(rolling_std(f("x"),2))))']]})


def test_all_registered_operations_preserve_ast_digest_and_execution():
    original = program()
    compact = CompactProgram.from_program(original)
    restored = CompactProgram.from_dict(compact.to_dict()).to_program()
    assert restored.to_json() == original.to_json()
    assert restored.digest == original.digest == compact.digest
    values = torch.arange(24, dtype=torch.float64).reshape(2, 6, 2) / 3 - 2
    torch.testing.assert_close(ProgramModule(original)(values), ProgramModule(restored)(values),
                               rtol=0, atol=0)
    assert original.parameters['a']['initial'] == restored.parameters['a']['initial']


@pytest.mark.parametrize('expression', [
    '__import__("os").system("echo unsafe")', 'f("x")[0]', 'x + 1',
    '[f("x") for x in y]', 'last(arg=f("x"))', 'last(*[f("x")])',
    'lambda: 1', 'f.__class__', 'True', 'None', 'rolling_mean(f("x"),2.0)',
    'sumsq(f("x"),24)', 'tanh(f("x"),1)', '1; 2'])
def test_non_language_input_is_rejected(expression):
    with pytest.raises(ValueError):
        parse_expression(expression)


@pytest.mark.parametrize('expression', ['last(rolling_mean(f("x"),0))',
                                         'last(ema(f("x"),2))', 'last(f("unknown"))',
                                         'last(1)', '1e999'])
def test_invalid_semantics_are_rejected_without_clamping(expression):
    value = CompactProgram.from_program(program()).to_dict()
    value['expressions'] = [expression]
    value['parameters'] = {}
    with pytest.raises(ValueError):
        CompactProgram.from_dict(value)


def test_feature_order_and_float_order_are_identity():
    p = program()
    value = p.to_dict()
    value['feature_names'].reverse()
    assert CompactProgram.from_program(Program.from_dict(value)).digest != p.digest
    swapped = copy.deepcopy(value)
    root = swapped['outputs'][-1]
    root['left'], root['right'] = root['right'], root['left']
    assert CompactProgram.from_program(Program.from_dict(swapped)).digest != Program.from_dict(value).digest


def test_strict_json_rejects_duplicate_and_nonfinite_keys():
    for value in ['{"a":1,"a":2}', '{"x":NaN}', '{"x":Infinity}']:
        with pytest.raises(ValueError):
            strict_json(value)
    assert strict_json(json.dumps({'x': 1})) == {'x': 1}


def test_importing_new_protocol_modules_does_not_import_torch():
    result = subprocess.run([sys.executable, '-c',
                             ('import sys; import factorlab.discovery.v7_agent; '
                              'assert "torch" not in sys.modules')],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
