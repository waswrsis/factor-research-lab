"""Bounded NumPy interpreter for V7 raw-unit formula features.

Validation is performed by the proposal boundary's existing Program language.
This interpreter never evaluates Python. The V7 raw K0 coordinate contract is
different from V5/V6 standardized neural windows, even for the same AST.
"""
from __future__ import annotations

import copy
import hashlib
from collections import OrderedDict

import numpy as np

from factorlab.protocol import fingerprint
from factorlab.time_utils import HOUR

from .known_library import FactorFrame, library_schema

EPSILON = 1e-6
AMPLITUDE_BOUND = 1e6
COORDINATE_VERSION = 'v7-raw-K0-window-v1'


def formula_identity(program: dict, lookback=48):
    if type(lookback) is not int or not 1 <= lookback <= 512:
        raise ValueError('Invalid formula lookback')
    # The caller supplies canonical validated data, not source code.
    if (program.get('version') != 'causal-operator-program-v1'
            or len(program.get('outputs', [])) != 1
            or tuple(program.get('feature_names', [])) != tuple(library_schema()['feature_names'])):
        raise ValueError('V7 formula requires one scalar output on the raw K0 feature contract')
    return {'version': COORDINATE_VERSION, 'program': copy.deepcopy(program),
            'library': library_schema(), 'lookback': lookback,
            'epsilon': EPSILON, 'amplitude_bound': AMPLITUDE_BOUND,
            'window_policy': 'full finite consecutive observable K0 rows'}



class _ScalarProfileCache:
    """Fit-local LRU of parameter-independent scalar vectors, never windows.

    Keys include the exact clipped input bytes, shape, dtype, program identity
    and node path. Changed inputs cannot reuse earlier data. All support and
    parameter checks still execute normally. This cache never contains labels,
    fitted coefficients or a solver result and grants no budget exemption.
    """
    def __init__(self, max_bytes=8 * 1024 * 1024, max_entries=4096):
        if (type(max_bytes) is not int or max_bytes < 0
                or type(max_entries) is not int or max_entries < 0):
            raise ValueError('Invalid scalar cache bounds')
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self.entries = OrderedDict()
        self.bytes = self.hits = self.misses = 0

    def get(self, key):
        value = self.entries.get(key)
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
            self.entries.move_to_end(key)
        return value

    def put(self, key, value):
        if key in self.entries or value.nbytes > self.max_bytes or not self.max_entries:
            return
        while self.entries and (self.bytes + value.nbytes > self.max_bytes
                                or len(self.entries) >= self.max_entries):
            _, old = self.entries.popitem(last=False)
            self.bytes -= old.nbytes
        stored = value.copy()
        stored.flags.writeable = False
        self.entries[key] = stored
        self.bytes += stored.nbytes


def _scalar_plan(program):
    """Locate maximal fixed scalar subtrees without changing AST evaluation."""
    nodes = {}
    def visit(node, path):
        if type(node) is not dict:
            return 0, 0, False, True
        op = node.get('op')
        if op in {'feature', 'const', 'param'}:
            layout = (1, 1, op != 'param', op == 'feature')
        else:
            if op in {'add', 'sub', 'mul', 'safe_div'}:
                keys = ('left', 'right')
            elif op in {'tanh', 'log_abs', 'sqrt_abs', 'abs', 'last', 'mean',
                        'std', 'max', 'min', 'lag', 'rolling_mean', 'rolling_std'}:
                keys = ('arg',)
            elif op == 'ema':
                keys = ('arg', 'alpha')
            else:
                return 0, 0, False, True
            children = [visit(node.get(k), path + (k,)) for k in keys]
            if any(c[0] == 0 for c in children):
                return 0, 0, False, True
            series = (False if op in {'last', 'mean', 'std', 'max', 'min'} else
                      True if op in {'lag', 'rolling_mean', 'rolling_std', 'ema'} else
                      any(c[3] for c in children))
            layout = (1 + sum(c[0] for c in children),
                      1 + max(c[1] for c in children), all(c[2] for c in children), series)
        nodes[id(node)] = (path, *layout)
        return layout
    visit(program['outputs'][0], ())
    eligible = {path for path, count, height, fixed, series in nodes.values()
                if fixed and not series and count > 1}
    return {key: (path, count, height) for key, (path, count, height, fixed, series) in nodes.items()
            if path in eligible and not any(path[:i] in eligible for i in range(len(path)))}


def _features_covered(program, plan):
    def covered(node):
        if type(node) is not dict:
            return False
        if id(node) in plan:
            return True
        op = node.get('op')
        if op in {'const', 'param'}:
            return True
        if op in {'add', 'sub', 'mul', 'safe_div'}:
            return covered(node.get('left')) and covered(node.get('right'))
        if op in {'tanh', 'log_abs', 'sqrt_abs', 'abs', 'last', 'mean', 'std',
                  'max', 'min', 'lag', 'rolling_mean', 'rolling_std', 'ema'}:
            return covered(node.get('arg')) and (op != 'ema' or covered(node.get('alpha')))
        return False
    return covered(program['outputs'][0])

def _execute(program, inputs, parameters, *, scalar_cache=None, cache_namespace=None, cache_input_key=None):
    features = program['feature_names']
    n = len(inputs)
    count = 0
    plan = _scalar_plan(program) if scalar_cache is not None else {}
    input_key = cache_input_key

    def evaluate(node, depth=1):
        nonlocal count, input_key
        entry = plan.get(id(node))
        if entry is None:
            return compute(node, depth)
        path, node_count, height = entry
        if input_key is None:
            contiguous = np.ascontiguousarray(inputs)
            input_key = (cache_namespace or fingerprint(program), inputs.shape,
                         inputs.dtype.str, hashlib.sha256(memoryview(contiguous).cast('B')).hexdigest())
        key = (input_key, path)
        cached = scalar_cache.get(key)
        if cached is not None:
            count += node_count
            if count > 64 or depth + height - 1 > 12:
                raise ValueError('Formula depth/node limit')
            return cached, False, 0
        result = compute(node, depth)
        if not result[1]:
            scalar_cache.put(key, result[0])
        return result

    def checked(data, series=False, start=0):
        value = np.asarray(data, dtype=float)
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite formula intermediate')
        return np.clip(value, -AMPLITUDE_BOUND, AMPLITUDE_BOUND), series, start

    def compute(node, depth=1):
        nonlocal count
        count += 1
        if count > 64 or depth > 12 or type(node) is not dict:
            raise ValueError('Formula depth/node limit')
        op = node.get('op')
        if op == 'feature':
            return checked(inputs[:, :, features.index(node['name'])], True)
        if op in {'const', 'param'}:
            value = node['value'] if op == 'const' else parameters[node['name']]
            return checked(np.full(n, value))
        if op in {'add', 'sub', 'mul', 'safe_div'}:
            a, sa, ia = evaluate(node['left'], depth + 1)
            b, sb, ib = evaluate(node['right'], depth + 1)
            series = sa or sb
            if series and not sa:
                a = a[:, None]
            if series and not sb:
                b = b[:, None]
            if op == 'safe_div':
                b = np.where(b < 0, -np.maximum(np.abs(b), EPSILON), np.maximum(np.abs(b), EPSILON))
            result = a + b if op == 'add' else a - b if op == 'sub' else a * b if op == 'mul' else a / b
            return checked(result, series, max(ia, ib))
        data, series, start = evaluate(node['arg'], depth + 1)
        if op in {'tanh', 'log_abs', 'sqrt_abs', 'abs'}:
            result = (np.tanh(data) if op == 'tanh' else np.log1p(np.abs(data)) if op == 'log_abs'
                      else np.sqrt(np.abs(data) + EPSILON) if op == 'sqrt_abs' else np.abs(data))
            return checked(result, series, start)
        if not series or start >= inputs.shape[1]:
            raise ValueError('Series/history required by formula')
        valid = data[:, start:]
        if op in {'last', 'mean', 'std', 'max', 'min'}:
            result = (valid[:, -1] if op == 'last' else valid.mean(1) if op == 'mean' else valid.std(1)
                      if op == 'std' else valid.max(1) if op == 'max' else valid.min(1))
            return checked(result)
        if op == 'lag':
            steps = node['steps']
            if type(steps) is not int or not 0 <= steps < inputs.shape[1] or start + steps >= inputs.shape[1]:
                raise ValueError('Lag exceeds window')
            output = data if steps == 0 else np.concatenate((np.zeros_like(data[:, :steps]), data[:, :-steps]), axis=1)
            return checked(output, True, start + steps)
        if op in {'rolling_mean', 'rolling_std'}:
            width = node['window']
            if type(width) is not int or not 1 <= width <= valid.shape[1]:
                raise ValueError('Rolling window exceeds history')
            patches = np.lib.stride_tricks.sliding_window_view(valid, width, axis=1)
            rolled = patches.mean(-1) if op == 'rolling_mean' else patches.std(-1)
            start += width - 1
            return checked(np.concatenate((np.zeros_like(data[:, :start]), rolled), axis=1), True, start)
        if op == 'ema':
            alpha, alpha_series, _ = evaluate(node['alpha'], depth + 1)
            if alpha_series or ((alpha <= 0) | (alpha > 1)).any():
                raise ValueError('Invalid EMA alpha')
            output = np.zeros_like(data)
            output[:, start] = valid[:, 0]
            for step in range(start + 1, inputs.shape[1]):
                output[:, step] = alpha * data[:, step] + (1 - alpha) * output[:, step - 1]
            return checked(output, True, start)
        raise ValueError(f'Unsupported operator {op!r}')

    value, series, _ = evaluate(program['outputs'][0])
    if series or value.shape != (n,):
        raise ValueError('V7 supports exactly one scalar per formula')
    return value


def execute_formula(program: dict, frame: FactorFrame, *, lookback=48, parameters=None, chunk_size=128, _scalar_cache=None):
    """Return aligned scalar features, with missing warmup/gap support as NaN.

    Sliding views and bounded chunks avoid allocating an entire 3D tensor.
    Feature input rows after the origin are never evaluated, including receipts
    delivered out of open-time order. Params must remain inside their bounds.
    """
    identity = formula_identity(program, lookback)
    if frame.schema_hash != fingerprint(library_schema()) or frame.feature_names != tuple(program['feature_names']):
        raise ValueError('Formula frame differs from raw K0 coordinate schema')
    if type(chunk_size) is not int or not 1 <= chunk_size <= 4096:
        raise ValueError('Invalid evaluator chunk')
    specs = program.get('parameters', {})
    theta = {key: value['initial'] for key, value in specs.items()} if parameters is None else parameters
    if set(theta) != set(specs) or any(not np.isfinite(theta[k]) or not v['lower'] <= theta[k] <= v['upper']
                                     for k, v in specs.items()):
        raise ValueError('Formula parameters outside declared finite bounds')
    cache_namespace = fingerprint(identity)
    output = np.full(len(frame.values), np.nan)
    metadata = frame.metadata
    plan = _scalar_plan(program) if _scalar_cache is not None else {}
    frame_key = None
    if _scalar_cache is not None:
        from .frame_identity import frame_identity
        try:
            frame_key = (cache_namespace, frame.values.shape, frame.values.dtype.str,
                hashlib.sha256(memoryview(np.ascontiguousarray(frame.values)).cast('B')).hexdigest(),
                frame_identity(metadata[['symbol', 'feature_open_time', 'available_at']]),
                fingerprint(metadata.index.tolist()))
        except (TypeError, ValueError, KeyError):
            # Unsupported metadata retains the original evaluator/error path.
            frame_key = None
    covered = frame_key is not None and _features_covered(program, plan)
    for positions in metadata.groupby('symbol', sort=False).groups.values():
        positions = np.asarray(positions)
        ordered = positions[np.argsort(metadata.iloc[positions].feature_open_time.to_numpy(), kind='stable')]
        group = metadata.iloc[ordered]
        breaks = group.feature_open_time.diff().ne(HOUR).cumsum().to_numpy()
        for segment in np.unique(breaks):
            indices = ordered[breaks == segment]
            if len(indices) < lookback:
                continue
            matrix = frame.values[indices]
            windows = np.lib.stride_tricks.sliding_window_view(matrix, lookback, axis=0).transpose(0, 2, 1)
            avail = metadata.iloc[indices].available_at.to_numpy(dtype='datetime64[ns]')
            receipts = np.lib.stride_tricks.sliding_window_view(avail, lookback)
            finite_windows = (np.lib.stride_tricks.sliding_window_view(
                np.isfinite(matrix).all(axis=1), lookback).all(axis=1)
                if _scalar_cache is not None else None)
            for left in range(0, len(windows), chunk_size):
                batch = windows[left:left + chunk_size]
                valid = (np.isfinite(batch).all(axis=(1, 2)) if finite_windows is None
                         else finite_windows[left:left + len(batch)].copy())
                visible = receipts[left:left + len(batch)]
                valid &= visible.max(axis=1) <= visible[:, -1]
                if valid.any():
                    destinations = indices[lookback - 1 + left:lookback - 1 + left + len(batch)][valid]
                    input_key = ((frame_key, tuple(destinations.tolist())) if frame_key is not None else None)
                    warm = covered and all((input_key, path) in _scalar_cache.entries
                                           for path, _, _ in plan.values())
                    # A broadcast placeholder is safe only after each actual feature
                    # subtree is present, keyed to all original data and receipts.
                    inputs = (np.broadcast_to(np.array(0., dtype=float),
                              (len(destinations), lookback, matrix.shape[1])) if warm else
                              np.clip(batch[valid], -AMPLITUDE_BOUND, AMPLITUDE_BOUND))
                    result = _execute(program, inputs, theta, scalar_cache=_scalar_cache,
                                      cache_namespace=cache_namespace, cache_input_key=input_key)
                    output[destinations] = result
    return output
