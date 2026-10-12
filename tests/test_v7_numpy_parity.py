"""Independent artificial parity audit: same raw K0 coordinates and fixed theta.

This is not a claim that old standardized neural factors have the same meaning.
Old ProgramModule stores trainable raw parameters in float32 by default; exact
theta comparisons therefore explicitly pass its effective values to NumPy.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from factorlab.discovery.causal_numpy import execute_formula, formula_identity
from factorlab.discovery.known_library import FEATURE_NAMES, FactorFrame, library_schema
from factorlab.discovery.operator_program import VERSION, Program, ProgramModule
from factorlab.discovery.program_syntax import parse_expression
from factorlab.protocol import fingerprint


def make_program(expression, parameters=None):
    return Program.from_dict({'version': VERSION, 'feature_names': list(FEATURE_NAMES),
                              'outputs': [parse_expression(expression)],
                              'parameters': parameters or {}})


def make_frame(n=30, *, values=None, metadata=None):
    if values is None:
        t = np.arange(n, dtype=np.float64)
        values = np.column_stack([.01 * (j + 1) + .005 * np.sin(t / (j + 1))
                                  for j in range(len(FEATURE_NAMES))])
        values[:, -4:] = np.column_stack([np.sin(t), np.cos(t), np.sin(t / 7), np.cos(t / 7)])
    if metadata is None:
        opens = pd.date_range('2026-01-01', periods=n, freq='h', tz='UTC')
        metadata = pd.DataFrame({'symbol': 'ARTIFICIAL', 'feature_open_time': opens,
                                 'available_at': opens + pd.Timedelta(hours=1),
                                 'forecast_window_start': opens + pd.Timedelta(hours=1),
                                 'sample_id': [f'artificial-{i}' for i in range(n)]})
    return FactorFrame(values, metadata, schema_hash=fingerprint(library_schema()))


def effective_theta(module):
    return {name: float(value.detach()) for name, value in module.parameter_values().items()}


def torch_expected(module, frame, *, lookback):
    """Independent row-by-row support construction, no NumPy evaluator internals."""
    output = np.full(len(frame.values), np.nan)
    meta = frame.metadata
    for destination, row in meta.iterrows():
        expected_times = pd.date_range(end=row.feature_open_time, periods=lookback, freq='h')
        selected = meta[(meta.symbol == row.symbol) & meta.feature_open_time.isin(expected_times)]
        selected = selected.sort_values('feature_open_time')
        if (len(selected) != lookback
                or not (selected.available_at <= row.available_at).all()):
            continue
        inputs = frame.values[selected.index.to_numpy()]
        if not np.isfinite(inputs).all():
            continue
        with torch.no_grad():
            output[destination] = float(module(torch.tensor(inputs[None], dtype=torch.float64))[0, 0])
    return output


@pytest.mark.parametrize('expression', [
    'last(f("rv_1"))', 'mean(lag(f("rv_6"),2))',
    'last(lag(f("rv_24"),0))',
    'mean(rolling_mean(lag(f("rv_48"),2),3))',
    'std(rolling_std(rolling_mean(f("rv_168"),2),3))',
    'last(ema(lag(rolling_mean(f("rv_720"),3),1),0.25))',
    'mean(add(rolling_mean(lag(f("rv_1"),1),3),f("rv_6")))',
    'std(sub(rolling_mean(f("rv_24"),2),lag(f("rv_48"),4)))',
    'mean(add(f("parkinson_24"),last(f("parkinson_168"))))',
    'max(tanh(f("hour_sin")))', 'min(abs(f("hour_cos")))',
    'last(sqrt_abs(log_abs(f("weekday_sin"))))',
    'std(f("weekday_cos"))',
    'last(safe_div(f("rv_1"),sub(f("hour_sin"),f("hour_sin"))))',
    'safe_div(mean(f("rv_1")),-0.0000000001)',
])
def test_numpy_matches_old_interpreter_on_identical_raw_float64_windows(expression):
    p, frame = make_program(expression), make_frame()
    module = ProgramModule(p).double()
    actual = execute_formula(p.to_dict(), frame, lookback=8, chunk_size=3)
    expected = torch_expected(module, frame, lookback=8)
    np.testing.assert_allclose(actual, expected, rtol=5e-13, atol=5e-13, equal_nan=True)
    assert np.isnan(actual[:7]).all()


def test_bounded_parameters_ema_safe_div_and_scalar_broadcast_use_same_effective_theta():
    p = make_program('mean(safe_div(ema(rolling_std(f("rv_24"),3),p("a")),p("d")))',
                     {'a': {'initial': .312345678901, 'lower': 0., 'upper': 1.},
                      'd': {'initial': -.0234567890123, 'lower': -.1, 'upper': .1}})
    module, frame = ProgramModule(p).double(), make_frame()
    actual = execute_formula(p.to_dict(), frame, lookback=9, parameters=effective_theta(module))
    np.testing.assert_allclose(actual, torch_expected(module, frame, lookback=9),
                               rtol=5e-13, atol=5e-13, equal_nan=True)


def test_parameter_default_contract_uses_declared_initial_not_hidden_float32_rounding():
    p = make_program('p("a")', {'a': {'initial': .312345678901, 'lower': 0., 'upper': 1.}})
    frame = make_frame()
    actual = execute_formula(p.to_dict(), frame, lookback=4)
    np.testing.assert_array_equal(actual[3:], np.full(len(actual) - 3, p.parameters['a']['initial']))
    # Different parameterization precision is explicit. It is not blamed on the
    # arithmetic interpreter or silently treated as an exact old-checkpoint load.
    assert effective_theta(ProgramModule(p).double())['a'] != p.parameters['a']['initial']


def test_full_valid_history_mask_excludes_prefix_padding_in_nested_reductions():
    frame = make_frame(n=6)
    values = frame.values.copy()
    values[:, 0] = np.arange(1., 7.)
    frame = make_frame(n=6, values=values)
    p = make_program('mean(add(rolling_mean(lag(f("rv_1"),1),2),f("rv_1")))')
    actual = execute_formula(p.to_dict(), frame, lookback=6)
    # Eligible t=2..5: (1+2)/2+3, ..., (4+5)/2+6; padded prefix is excluded.
    assert actual[-1] == 7.5
    np.testing.assert_allclose(actual, torch_expected(ProgramModule(p).double(), frame, lookback=6),
                               equal_nan=True)


def test_future_pollution_cannot_change_earlier_outputs():
    p = make_program('last(ema(rolling_mean(f("rv_1"),3),0.25))')
    original = make_frame()
    polluted_values = original.values.copy()
    polluted_values[20:] = np.nan
    polluted = make_frame(values=polluted_values)
    before = execute_formula(p.to_dict(), original, lookback=8)
    after = execute_formula(p.to_dict(), polluted, lookback=8)
    np.testing.assert_array_equal(before[:20], after[:20])
    module = ProgramModule(p).double()
    inputs = torch.tensor(original.values[None], dtype=torch.float64)
    changed = torch.tensor(polluted.values[None], dtype=torch.float64)
    torch.testing.assert_close(module(inputs, end_index=19), module(changed, end_index=19))
    assert np.isnan(after[20:]).all()


def test_gaps_missing_feature_and_late_receipts_have_identical_support():
    original = make_frame(n=36)
    retained = [i for i in range(36) if i != 10]
    meta = original.metadata.iloc[retained].reset_index(drop=True).copy()
    values = original.values[retained].copy()
    # One unused K0 column is missing: complete-window contract still rejects it.
    values[20, -1] = np.nan
    meta.loc[4, 'available_at'] += pd.Timedelta(hours=5)
    frame = make_frame(values=values, metadata=meta)
    p = make_program('mean(lag(f("rv_1"),1))')
    actual = execute_formula(p.to_dict(), frame, lookback=4, chunk_size=2)
    expected = torch_expected(ProgramModule(p).double(), frame, lookback=4)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13, equal_nan=True)
    assert np.isnan(actual[10:13]).all()  # full support restarts after missing hour
    assert np.isnan(actual[20:24]).all()
    assert np.isnan(actual[5:8]).all()  # delayed historical row unavailable at these origins


def test_symbol_and_row_order_do_not_change_result_or_chunk_policy():
    left = make_frame(18)
    right_meta = left.metadata.copy()
    right_meta['symbol'] = 'OTHER'
    right_meta['sample_id'] = [f'other-{i}' for i in range(18)]
    merged_meta = pd.concat([left.metadata, right_meta], ignore_index=True)
    merged_values = np.concatenate([left.values, left.values * 2])
    order = np.random.default_rng(77).permutation(36)
    frame = make_frame(values=merged_values[order], metadata=merged_meta.iloc[order])
    p = make_program('last(rolling_std(f("rv_6"),3))')
    expected = torch_expected(ProgramModule(p).double(), frame, lookback=6)
    for chunk_size in (1, 5, 128):
        actual = execute_formula(p.to_dict(), frame, lookback=6, chunk_size=chunk_size)
        np.testing.assert_allclose(actual, expected, rtol=5e-13, atol=5e-13, equal_nan=True)


def test_clipping_and_zero_signed_denominator_match_registered_semantics():
    frame = make_frame(n=10)
    values = frame.values.copy()
    values[:, 0] = np.linspace(-1e9, 1e9, 10)
    p = make_program('safe_div(last(mul(f("rv_1"),1000000)),-0.0)')
    frame = make_frame(n=10, values=values)
    actual = execute_formula(p.to_dict(), frame, lookback=4)
    expected = torch_expected(ProgramModule(p).double(), frame, lookback=4)
    np.testing.assert_array_equal(actual, expected)


def test_coordinate_identity_cannot_be_confused_with_old_program_digest():
    p = make_program('last(f("rv_1"))')
    identity = formula_identity(p.to_dict(), 8)
    assert identity['version'] == 'v7-raw-K0-window-v1'
    assert fingerprint(identity) != p.digest
    assert fingerprint(formula_identity(p.to_dict(), 9)) != fingerprint(identity)
    record = p.to_dict()
    record['feature_names'][0] = 'standardized_return'
    with pytest.raises(ValueError, match='raw K0'):
        formula_identity(record, 8)
