"""Explicit UTC time arithmetic independent of datetime storage resolution."""
from __future__ import annotations

import numpy as np
import pandas as pd

EPOCH = pd.Timestamp('1970-01-01', tz='UTC')
HOUR = pd.Timedelta(hours=1)


def utc_index(values) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(values)
    if index.tz is None or index.hasnans:
        raise ValueError('Time arithmetic requires timezone-aware, nonmissing timestamps')
    return index.tz_convert('UTC')


def hour_ids(values) -> np.ndarray:
    """Floor elapsed UTC hours, including dates before the Unix epoch."""
    return np.asarray((utc_index(values) - EPOCH) // HOUR, dtype=np.int64)


def hourly_contiguous(values) -> bool:
    index = utc_index(values)
    return bool(len(index) < 2 or ((index[1:] - index[:-1]) == HOUR).all())


def constant_intervals(values) -> bool:
    index = utc_index(values)
    if len(index) < 3:
        return True
    intervals = index[1:] - index[:-1]
    return bool((intervals == intervals[0]).all())


def month_start(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError('Calendar boundaries require timezone-aware timestamps')
    stamp = stamp.tz_convert('UTC')
    return pd.Timestamp(year=stamp.year, month=stamp.month, day=1, tz='UTC')


def calendar_window(cutoff, months=3) -> tuple[pd.Timestamp, pd.Timestamp]:
    end = pd.Timestamp(cutoff)
    if type(months) is not int or months <= 0:
        raise ValueError('Calendar months must be a positive integer')
    if end.tzinfo is None or end.tz_convert('UTC') != month_start(end):
        raise ValueError('Rolling cutoff must be the first UTC instant of a calendar month')
    end = end.tz_convert('UTC')
    return end - pd.DateOffset(months=months), end
