"""Precision/type and causal raw-frame hashing invariants."""
import numpy as np
import pandas as pd
import pytest

from factorlab.discovery.frame_identity import frame_identity


def frame():
    return pd.DataFrame({'t': pd.date_range('2026-10-01', periods=3, freq='ns', tz='UTC'),
                         'x': [1., 2., 3.], 's': ['a', 'b', None], 'b': [True, False, True]})


def test_hash_keeps_nanosecond_and_single_ulp_numeric_changes():
    original = frame()
    digest = frame_identity(original)
    changed = original.copy()
    changed.loc[0, 't'] += pd.Timedelta(nanoseconds=1)
    assert frame_identity(changed) != digest
    changed = original.copy()
    changed.loc[0, 'x'] = np.nextafter(1., 2.)
    assert frame_identity(changed) != digest


def test_content_order_and_type_matter_but_incidental_index_does_not():
    original = frame()
    digest = frame_identity(original)
    changed = original.set_axis([10, 11, 12])
    assert frame_identity(changed) == digest
    assert frame_identity(original.iloc[::-1]) != digest
    assert frame_identity(original[original.columns[::-1]]) != digest
    assert frame_identity(pd.DataFrame({'x': [1., None]})) != frame_identity(pd.DataFrame({'x': ['1.0', 'nan']}))


def test_no_native_datetime_formatting_or_dataframe_dict_conversion(monkeypatch):
    original = frame()
    expected = frame_identity(original)
    def forbidden(*a, **k):
        pytest.fail('Typed hashing must not invoke pandas string/dict formatting')
    monkeypatch.setattr(pd.DataFrame, 'astype', forbidden)
    monkeypatch.setattr(pd.DataFrame, 'to_dict', forbidden)
    assert frame_identity(original) == expected


def test_parquet_roundtrip_and_missing_time_are_stable(tmp_path):
    original = frame()
    original.loc[1, 't'] = pd.NaT
    original.to_parquet(tmp_path / 'x.parquet')
    assert frame_identity(pd.read_parquet(tmp_path / 'x.parquet')) == frame_identity(original)


@pytest.mark.parametrize('bad', [pd.DataFrame([[1, 2]], columns=['x', 'x']), pd.DataFrame({'x': [{}]})])
def test_unsupported_or_ambiguous_schema_is_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        frame_identity(bad)


@pytest.mark.parametrize('unsigned,signed', [('uint64', 'int64'), ('UInt64', 'Int64')])
def test_unsigned_max_cannot_collide_with_negative_signed_value(unsigned, signed):
    high = pd.DataFrame({'x': pd.Series([2**64 - 1], dtype=unsigned)})
    negative = pd.DataFrame({'x': pd.Series([-1], dtype=signed)})
    assert frame_identity(high) != frame_identity(negative)


def test_object_datetime_retains_timezone_and_missing_time_is_null():
    aware = pd.DataFrame({'t': pd.Series([pd.Timestamp('2026-01-01', tz='UTC')], dtype=object)})
    naive = pd.DataFrame({'t': pd.Series([pd.Timestamp('2026-01-01')], dtype=object)})
    assert frame_identity(aware) != frame_identity(naive)
    missing = pd.DataFrame({'t': pd.Series([np.datetime64('NaT')], dtype=object)})
    null = pd.DataFrame({'t': pd.Series([None], dtype=object)})
    assert frame_identity(missing) == frame_identity(null)
