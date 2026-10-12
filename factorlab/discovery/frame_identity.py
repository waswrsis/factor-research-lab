"""Typed column fingerprints without pandas datetime/string formatting.

Binary numeric and UTC-time coordinates retain precision. Column/row order
matter; the incidental pandas index does not. The tagged format cannot confuse
a literal string with a number or missing value. This is a new identity rule;
frozen records using another rule must not be silently rewritten.
"""
from __future__ import annotations

import hashlib
import struct
from datetime import datetime

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_unsigned_integer_dtype,
)

from factorlab.protocol import fingerprint


def _scalar(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return {'null': True}
    if isinstance(value, str):
        return {'string': value}
    if isinstance(value, (bool, np.bool_)):
        return {'boolean': bool(value)}
    if isinstance(value, (int, np.integer)):
        return {'integer': str(int(value))}
    if isinstance(value, (float, np.floating)):
        return {'float64_bits': struct.pack('<d', float(value)).hex()}
    if isinstance(value, (pd.Timestamp, datetime, np.datetime64)):
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            return {'null': True}
        return {'time_ns': str(stamp.value), 'timezone': str(stamp.tz)}
    raise TypeError('Unsupported dataframe identity scalar: ' + type(value).__name__)


def frame_identity(frame):
    """Canonical supported dataframe content, without astype(str)/to_dict."""
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError('Frame identity requires a dataframe with unique columns')
    columns = []
    for name in frame.columns:
        if not isinstance(name, str):
            raise TypeError('Frame identity requires named string columns')
        series = frame[name]
        dtype = series.dtype
        if is_datetime64_any_dtype(dtype):
            values = series.array.as_unit('ns').asi8.astype('<i8', copy=False)
            content = {'kind': 'epoch_nanoseconds', 'timezone': str(getattr(dtype, 'tz', None)),
                       'sha256': hashlib.sha256(values.tobytes()).hexdigest()}
        elif (is_bool_dtype(dtype) or is_integer_dtype(dtype) or is_float_dtype(dtype)) and not series.isna().any():
            if getattr(dtype, 'itemsize', 8) > 8:
                raise TypeError('Frame identity does not narrow numeric types wider than 64 bits')
            kind = ('bool' if is_bool_dtype(dtype) else 'unsigned_integer' if is_unsigned_integer_dtype(dtype)
                    else 'integer' if is_integer_dtype(dtype) else 'float64')
            target = ('u1' if kind == 'bool' else '<f8' if kind == 'float64'
                      else '<u8' if kind == 'unsigned_integer' else '<i8')
            values = series.to_numpy(dtype=target)
            content = {'kind': kind, 'sha256': hashlib.sha256(values.tobytes()).hexdigest()}
        else:
            # The NumPy object view already contains primitive Python values;
            # do not ask pandas to stringify or box each datetime/native cell.
            content = {'kind': 'tagged_scalars', 'sha256': fingerprint([
                _scalar(value) for value in series.to_numpy(dtype=object).tolist()])}
        columns.append({'name': name, **content})
    return fingerprint({'version': 'typed-frame-columns-v1', 'rows': len(frame), 'columns': columns})
