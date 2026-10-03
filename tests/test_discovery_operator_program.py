"""Artificial formula semantics and fitting evidence; no market efficacy claim."""
import copy
import json

import pytest
import torch

from factorlab.discovery.operator_program import (
    AMPLITUDE_BOUND,
    MAX_NODES,
    VERSION,
    Program,
    ProgramModule,
    program_language_spec,
    validate_program,
)


def feature(name='x'):
    return {'op': 'feature', 'name': name}


def unary(op, arg, **kwargs):
    return {'op': op, 'arg': arg, **kwargs}


def constant(value):
    return {'op': 'const', 'value': value}


def binary(op, left, right):
    return {'op': op, 'left': left, 'right': right}


def program(roots=None, parameters=None):
    return Program.from_dict({'version': VERSION, 'feature_names': ['x', 'y'],
                              'outputs': roots or [unary('last', feature())],
                              'parameters': parameters or {}})


def test_hand_calculated_rolling_lag_ema_and_scalar_broadcast():
    x = torch.tensor([[[1., 2.], [2., 4.], [3., 6.], [4., 8.]]])
    p = program([
        unary('last', unary('rolling_mean', feature(), window=3)),
        unary('mean', unary('lag', feature(), steps=2)),
        unary('last', unary('ema', feature(), alpha=constant(0.5))),
        unary('last', unary('rolling_std', feature(), window=2)),
        unary('mean', binary('add', feature(), unary('last', feature('y')))),
        unary('std', feature()), unary('max', feature()), unary('min', feature()),
    ])
    expected = torch.tensor([[3., 1.5, 3.125, .5, 10.5, 1.25**.5, 4., 1.]])
    torch.testing.assert_close(ProgramModule(p)(x), expected)
    assert p.required_lookback == 3


def test_nested_warmup_and_series_alignment():
    p = program([unary('mean', binary('add',
        unary('rolling_mean', unary('lag', feature(), steps=1), window=2), feature()))])
    x = torch.arange(1., 6.).view(1, 5, 1).expand(1, 5, 2)
    assert p.required_lookback == 3
    # Valid t=2..4: current x + mean(x[t-2:t]) = 4.5, 6.5, 8.5.
    torch.testing.assert_close(ProgramModule(p)(x), torch.tensor([[6.5]]))
    with pytest.raises(ValueError, match='lookback'):
        ProgramModule(p)(x[:, :2])


def test_origin_prefix_is_causal_including_nonfinite_future():
    p = program([unary('last', unary('ema', unary('rolling_mean', feature(), window=2),
                                    alpha=constant(.3)))])
    x = torch.arange(12., dtype=torch.float32).reshape(1, 6, 2)
    original = ProgramModule(p)(x, end_index=3)
    changed = x.clone()
    changed[:, 4:] = float('nan')
    torch.testing.assert_close(ProgramModule(p)(changed, end_index=3), original)
    torch.testing.assert_close(ProgramModule(p)(x[:, :4]), original)


def test_batch_independence_and_input_gradients():
    module = ProgramModule(program([binary('safe_div', unary('mean', feature()),
                                          unary('std', feature('y')))]))
    x = torch.randn(4, 8, 2, requires_grad=True)
    output = module(x)
    torch.testing.assert_close(output, torch.cat([module(row[None]) for row in x]))
    output[0].sum().backward()
    assert torch.count_nonzero(x.grad[1:]) == 0
    assert torch.count_nonzero(x.grad[0]) > 0


def test_learned_formula_fits_known_synthetic_data_and_state_roundtrip():
    p = program([binary('add', binary('mul', {'op': 'param', 'name': 'keys'},
                                      unary('last', feature())),
                        {'op': 'param', 'name': 'items'})],
                {'keys': {'initial': .2, 'lower': -4., 'upper': 4.},
                 'items': {'initial': -.2, 'lower': -2., 'upper': 2.}})
    module = ProgramModule(p)
    x = torch.linspace(-1., 1., 64).view(64, 1, 1).expand(64, 3, 2)
    target = 1.7 * x[:, -1, 0:1] + .6
    optimizer = torch.optim.Adam(module.parameters(), lr=.08)
    initial = torch.nn.functional.mse_loss(module(x), target).item()
    for step in range(180):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(module(x), target)
        loss.backward()
        if step == 0:
            assert all(v.grad is not None and v.grad.abs().sum() > 0
                       for v in module.parameters())
        optimizer.step()
    assert loss.item() < initial * .001
    assert loss.item() < 1e-6
    clone = ProgramModule(p)
    clone.load_state_dict(module.state_dict())
    torch.testing.assert_close(clone(x), module(x), rtol=0, atol=0)
    reordered = p.to_dict()
    reordered['parameters'] = dict(reversed(list(reordered['parameters'].items())))
    reordered_module = ProgramModule(reordered)
    assert reordered_module.program.digest == p.digest
    reordered_module.load_state_dict(module.state_dict())
    torch.testing.assert_close(reordered_module(x), module(x), rtol=0, atol=0)
    torch.testing.assert_close(ProgramModule(module.to_program())(x), module(x), atol=1e-6, rtol=1e-6)


def test_ema_alpha_receives_gradient():
    module = ProgramModule(program_language_spec()['example'])
    x = torch.tensor([[[0.], [1.], [3.], [7.]]])
    module(x).sum().backward()
    assert next(module.parameters()).grad.abs().item() > 0


def test_json_roundtrip_digest_and_defensive_copies():
    record = program_language_spec()['example']
    p = Program.from_dict(record)
    expected = p.digest
    record['parameters']['alpha']['initial'] = .7
    record['outputs'][0]['arg']['alpha']['name'] = 'bad'
    assert p.digest == expected
    assert Program.from_json(p.to_json()).digest == expected
    assert Program.from_dict(json.loads(p.to_json())).digest == expected
    assert 'factor_0 = last(ema(' in p.source
    assert 'parameter' in p.formulas[0]
    assert validate_program(p, features=('x',)) == p.to_dict()
    reordered = p.to_dict()
    reordered['parameters'] = dict(reversed(list(reordered['parameters'].items())))
    assert Program.from_dict(reordered).digest == expected


@pytest.mark.parametrize('root', [
    {'op': 'python', 'code': '__import__("os")'},
    unary('lag', feature(), steps=-1),
    unary('rolling_mean', feature(), window=True),
    unary('mean', constant(1)),
    unary('last', unary('ema', feature(), alpha=constant(0))),
    unary('last', unary('ema', feature(), alpha=binary('add', constant(.2), constant(.3)))),
    {'op': 'const', 'value': float('nan')},
    {'op': 'param', 'name': 'undeclared'},
    {'op': 'feature', 'name': 'future_close'},
    {'op': 'const', 'value': 1, 'code': 'ignored?'},
    feature(),
])
def test_bad_nodes_rejected(root):
    with pytest.raises(ValueError):
        program([root])


def test_structural_budgets_and_bad_json_rejected():
    for encoded in ['{', '[]', '{"version":1,"version":2}', 'NaN']:
        with pytest.raises(ValueError):
            Program.from_json(encoded)
    node = feature()
    for _ in range(14):
        node = unary('abs', node)
    with pytest.raises(ValueError, match='budget'):
        program([unary('last', node)])
    node = unary('last', feature())
    for _ in range(8):
        node = binary('add', copy.deepcopy(node), copy.deepcopy(node))
    with pytest.raises(ValueError, match='budget'):
        program([node])
    assert MAX_NODES == 64
    with pytest.raises(ValueError):
        program([constant(0)] * 17)
    with pytest.raises(ValueError, match='Lookback'):
        program([unary('last', unary('lag', unary('rolling_mean', feature(), window=512),
                                    steps=1))])
    with pytest.raises(ValueError):
        program([{'op': 'param', 'name': 'a'}],
                {'a': {'initial': 1. - 1e-12, 'lower': 0., 'upper': 1.}})
    with pytest.raises(ValueError):
        program(parameters={'unused': {'initial': 0., 'lower': -1., 'upper': 1.}})


def test_runtime_finite_bounds_dtype_and_schema_checks():
    module = ProgramModule(program([binary('safe_div', unary('last', feature()), constant(0))]))
    x = torch.ones(1, 4, 2)
    assert module(x).item() == AMPLITUDE_BOUND
    with pytest.raises(ValueError, match='nonfinite'):
        module(x * float('inf'))
    with pytest.raises(ValueError):
        module(x.half())
    with pytest.raises(ValueError):
        ProgramModule(program(), feature_names=('y', 'x'))
    with pytest.raises(ValueError):
        validate_program(program(), features=('y', 'x'))
    with pytest.raises(ValueError):
        ProgramModule(program([unary('last', unary('lag', feature(), steps=3))]), lookback=3)
    with pytest.raises(ValueError):
        module(x, end_index=-1)
    with pytest.raises(ValueError):
        module(torch.ones(1, 513, 2))
    parameterized = ProgramModule(program_language_spec()['example'])
    with torch.no_grad():
        next(parameterized.parameters()).fill_(float('inf'))
    with pytest.raises(ValueError, match='Nonfinite raw'):
        parameterized(torch.ones(1, 3, 1))
