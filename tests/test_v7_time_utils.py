import numpy as np
import pandas as pd
import pytest

from factorlab.time_utils import calendar_window, constant_intervals, hour_ids, hourly_contiguous


@pytest.mark.parametrize('unit', ['ns', 'us', 'ms'])
def test_calendar_blocks_do_not_depend_on_storage_unit(unit):
    index = pd.date_range('2025-01-01', periods=240, freq='h', tz='UTC').as_unit(unit)
    assert len(np.unique(hour_ids(index) // 24)) == 10
    np.testing.assert_array_equal(hour_ids(index), hour_ids(index.as_unit('ns')))
    assert hourly_contiguous(index) and constant_intervals(index)
    assert not hourly_contiguous(index.delete(20))
    assert not constant_intervals(index.delete(20))


@pytest.mark.parametrize(('end', 'start'), [
    ('2024-03-01', '2023-12-01'), ('2024-06-01', '2024-03-01'),
    ('2025-01-01', '2024-10-01'), ('2027-04-01', '2027-01-01')])
def test_three_calendar_months(end, start):
    left, right = calendar_window(pd.Timestamp(end, tz='UTC'))
    assert left == pd.Timestamp(start, tz='UTC')
    assert right == pd.Timestamp(end, tz='UTC')


def test_time_contracts_reject_missing_and_nonboundary_inputs():
    with pytest.raises(ValueError):
        hour_ids(pd.date_range('2025-01-01', periods=2, freq='h'))
    with pytest.raises(ValueError):
        hour_ids(pd.DatetimeIndex([pd.NaT], tz='UTC'))
    with pytest.raises(ValueError):
        calendar_window(pd.Timestamp('2025-04-02', tz='UTC'))
    assert hour_ids(pd.DatetimeIndex([pd.Timestamp('1969-12-31 23:30', tz='UTC')]))[0] == -1
