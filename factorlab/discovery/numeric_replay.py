"""Versioned float64 replay checks; never a score or selection tolerance.

Only callers' explicitly selected numerical fields should enable roundoff.
Schema, primitive types, shapes and every non-floating leaf stay exact. Inputs
are bounded JSON-compatible trees; arrays must first be serialized by their
typed producer. This intentionally does not coerce dtype, bool, integer or NaN.
"""
from __future__ import annotations

import math
import sys

RULE = {'version': 'v7-float64-replay-roundoff-v1', 'epsilon_multiplier': 64,
        'epsilon': sys.float_info.epsilon, 'scale': 'max(1,abs(actual),abs(expected))',
        'finite_only': True, 'schema_types_shape_nonfloat': 'exact',
        'scientific_scores': 'parent_recomputed_no_selection_tolerance'}


def assert_replayed(actual, expected, *, roundoff=False, _path='$'):
    """Reject changed structure/values; optional tolerance is float-float only."""
    if type(actual) is not type(expected):
        raise ValueError('Numeric replay dtype changed at ' + _path)
    kind = type(expected)
    if kind is dict:
        if set(actual) != set(expected) or any(type(k) is not str for k in expected):
            raise ValueError('Numeric replay fields changed at ' + _path)
        for key in expected:
            assert_replayed(actual[key], expected[key], roundoff=roundoff, _path=_path + '.' + key)
    elif kind in (list, tuple):
        if len(actual) != len(expected):
            raise ValueError('Numeric replay shape changed at ' + _path)
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            assert_replayed(left, right, roundoff=roundoff, _path=f'{_path}[{index}]')
    elif kind is float:
        if not math.isfinite(actual) or not math.isfinite(expected):
            raise ValueError('Nonfinite numeric replay at ' + _path)
        limit = 64 * sys.float_info.epsilon * max(1., abs(actual), abs(expected)) if roundoff else 0.
        if abs(actual-expected) > limit:
            raise ValueError('Numeric replay exceeds float64 roundoff at ' + _path)
    elif kind in (str, int, bool, type(None)):
        if actual != expected:
            raise ValueError('Numeric replay exact identity changed at ' + _path)
    else:
        raise ValueError('Unsupported numeric replay dtype at ' + _path)
