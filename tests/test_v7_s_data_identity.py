"""S integration delegates dtype/NaT rules to shared frame_identity tests."""
from dataclasses import replace

import numpy as np
import pandas as pd
from test_v7_nonlinear_channel import artificial_data

from factorlab.discovery import s_data_identity as module


def raw(data):
    return pd.DataFrame({'symbol': data.frame.metadata.symbol,
                         'available_at': data.frame.metadata.available_at,
                         'open_time': data.frame.metadata.feature_open_time,
                         'close': data.frame.values[:, 0]})


def test_stage_identity_avoids_formatter_and_future_labels(monkeypatch):
    data = artificial_data()
    selected = raw(data)
    cutoff = pd.Timestamp('2025-04-01', tz='UTC')
    expected = module.typed_search_data_id(data, selected, cutoff)
    def forbidden(*args, **kwargs):
        raise AssertionError('String formatter/dict path must not be used')
    monkeypatch.setattr(pd.DataFrame, 'astype', forbidden)
    monkeypatch.setattr(pd.DataFrame, 'to_dict', forbidden)
    assert module.typed_search_data_id(data, selected, cutoff) == expected
    target = data.target.copy()
    target[data.maturity > cutoff] *= 100
    assert module.typed_search_data_id(replace(data, target=target), selected, cutoff) == expected
    modified = selected.copy()
    modified.loc[modified.available_at > cutoff, 'close'] *= 100
    assert module.typed_search_data_id(data, modified, cutoff) == expected


def test_sample_order_and_typed_value_precision_are_bound():
    data = artificial_data()
    selected = raw(data)
    cutoff = pd.Timestamp('2025-04-01', tz='UTC')
    expected = module.typed_search_data_id(data, selected, cutoff)
    index = np.arange(len(data.target))[::-1]
    changed = replace(data, frame=data.frame.take(index), target=data.target[index],
                      maturity=data.maturity[index], observed=data.observed[index])
    assert module.typed_search_data_id(changed, selected, cutoff) != expected
    altered = selected.copy()
    altered.loc[0, 'available_at'] += pd.Timedelta(nanoseconds=1)
    assert module.typed_search_data_id(data, altered, cutoff) != expected
    # Raw rows/columns are explicitly canonicalized, unlike feature sample order.
    assert module.typed_search_data_id(data, selected.iloc[::-1, ::-1], cutoff) == expected
