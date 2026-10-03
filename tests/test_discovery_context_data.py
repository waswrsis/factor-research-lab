"""Point-in-time context checks, including late receipts and future appends."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_discovery_tasks import bars, index_at, spec

from factorlab.discovery.context_data import (
    build_context_dataset,
    fetch_context,
    funding_archive_context,
    validate_context_table,
)


def table(rows):
    return pd.DataFrame(rows, columns=['symbol', 'observed_at', 'available_at', 'value'])


def value(data, index, feature):
    return data.x[index, -1, data.features.index(feature)]


def test_delayed_receipt_missing_masks_and_each_window_step():
    source = bars()
    rows = table([('BTCUSDT', '2025-01-03T10:00:00Z', '2025-01-03T12:00:00Z', .002)])
    result = build_context_dataset(source, spec(), funding=rows, cross_assets=False)
    i = index_at(result, 58)  # origin 11:00:05: event occurred but not received
    assert value(result, i, 'funding') == 0
    assert value(result, i, 'funding_observed') == 0
    i = index_at(result, 59)  # origin 12:00:05
    assert value(result, i, 'funding') == pytest.approx(.002)
    assert value(result, i, 'funding_observed') == 1
    np.testing.assert_array_equal(result.x[i, :-1, result.features.index('funding_observed')], 0)
    stale = index_at(result, 90)
    assert value(result, stale, 'funding_observed') == 0
    assert np.isfinite(result.x).all()


def test_future_append_does_not_rewrite_past_with_delayed_older_event():
    source = bars()
    initial = table([('BTCUSDT', '2025-01-03T10:00:00Z', '2025-01-03T11:00:00Z', .001)])
    more = pd.concat([initial, table([
        ('BTCUSDT', '2025-01-03T09:00:00Z', '2025-01-03T13:00:00Z', .009),
        ('BTCUSDT', '2025-01-04T00:00:00Z', '2025-01-04T01:00:00Z', .003)])], ignore_index=True)
    old = build_context_dataset(source.iloc[:71], spec(), funding=initial, cross_assets=False)
    new = build_context_dataset(source, spec(), funding=more, cross_assets=False)
    np.testing.assert_array_equal(old.x, new.x[:len(old.x)])
    assert value(new, index_at(new, 62), 'funding') == pytest.approx(.001)


def test_revisions_only_change_after_receipt_and_outside_universe_rejected():
    rows = table([
        ('BTCUSDT', '2025-01-03T10:00:00Z', '2025-01-03T11:00:00Z', .001),
        ('BTCUSDT', '2025-01-03T10:00:00Z', '2025-01-03T14:00:00Z', .002)])
    result = build_context_dataset(bars(), spec(), funding=rows, cross_assets=False)
    assert value(result, index_at(result, 60), 'funding') == pytest.approx(.001)
    assert value(result, index_at(result, 63), 'funding') == pytest.approx(.002)
    rows.loc[0, 'symbol'] = 'OTHER'
    with pytest.raises(ValueError, match='outside'):
        build_context_dataset(bars(), spec(), funding=rows)


@pytest.mark.parametrize('change', ['naive', 'future_event', 'nonfinite', 'negative_oi', 'duplicate'])
def test_context_rejects_invalid_source_assertions(change):
    rows = table([('BTCUSDT', '2025-01-03T10:00:00Z', '2025-01-03T11:00:00Z', 1.)])
    if change == 'naive':
        rows.loc[0, 'available_at'] = '2025-01-03T11:00:00'
    elif change == 'future_event':
        rows.loc[0, 'observed_at'] = '2025-01-03T12:00:00Z'
    elif change == 'nonfinite':
        rows.loc[0, 'value'] = np.nan
    elif change == 'negative_oi':
        rows.loc[0, 'value'] = -1.
    else:
        rows = pd.concat([rows, rows], ignore_index=True)
    with pytest.raises(ValueError):
        validate_context_table(rows, 'open_interest')


def test_cross_asset_features_are_causal_and_self_peer_explicitly_missing():
    btc, eth = bars(), bars(symbol='ETHUSDT')
    eth.loc[:, ['open', 'high', 'low', 'close']] *= np.exp(.0003 * np.arange(len(eth)))[:, None]
    source = pd.concat([btc, eth], ignore_index=True)
    config = spec(symbols=('BTCUSDT', 'ETHUSDT'))
    result = build_context_dataset(source, config)
    i = index_at(result, 70)
    expected = eth.close.iloc[70] / eth.close.iloc[69] - 1
    assert value(result, i, 'peer_ETHUSDT_return_1h') == pytest.approx(expected)
    assert value(result, i, 'peer_ETHUSDT_return_1h_observed') == 1
    assert value(result, i, 'peer_BTCUSDT_return_1h_observed') == 0
    assert value(result, i, 'peer_ETHUSDT_correlation_24h_observed') == 1
    prefix = pd.concat([btc.iloc[:71], eth.iloc[:71]], ignore_index=True)
    past = build_context_dataset(prefix, config)
    np.testing.assert_array_equal(past.x, result.x[:len(past.x)])
    changed = source.copy()
    changed.loc[(changed.symbol == 'ETHUSDT') & (changed.open_time > btc.open_time.iloc[70]),
                ['open', 'high', 'low', 'close']] *= 2
    altered = build_context_dataset(changed, config)
    np.testing.assert_array_equal(past.x, altered.x[:len(past.x)])


def test_cross_asset_gap_masks_missing_and_restarts_correlation():
    btc, eth = bars(), bars(skip=(70,), symbol='ETHUSDT')
    source = pd.concat([btc, eth], ignore_index=True)
    result = build_context_dataset(source, spec(symbols=('BTCUSDT', 'ETHUSDT')))
    i = index_at(result, 72)
    assert value(result, i, 'peer_ETHUSDT_correlation_24h_observed') == 0
    assert np.isfinite(result.x).all()


def test_archive_conversion_does_not_reuse_assumed_historical_publication():
    frame = pd.DataFrame({'symbol': ['BTCUSDT'],
        'funding_time': pd.to_datetime(['2025-01-01'], utc=True), 'funding_rate': [.001]})
    frame.attrs['availability_basis'] = 'assumed_live_delay'
    result = funding_archive_context(frame, '2026-01-01T00:00:00Z')
    assert result.available_at.iloc[0] == pd.Timestamp('2026-01-01T00:00:00Z')
    asserted = table([('BTCUSDT', '2025-01-01T00:00:00Z', '2025-01-01T00:00:05Z', .001)])
    asserted.attrs['availability_basis'] = 'assumed_live_delay'
    with pytest.raises(ValueError, match='Assumed'):
        validate_context_table(asserted, 'funding')


def test_bounded_fetch_records_receipt_and_refuses_historical_oi():
    calls = []
    def get(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: [
            {'symbol': 'BTCUSDT', 'fundingTime': 1735689600000, 'fundingRate': '.001'}])
    result = fetch_context('funding', 'BTCUSDT', start='2025-01-01', end='2025-01-02',
                           limit=3, session=SimpleNamespace(get=get))
    assert len(calls) == 1
    assert calls[0][1]['params']['limit'] == 3
    assert result.available_at.iloc[0] > result.observed_at.iloc[0]
    assert result.attrs['availability_basis'] == 'actual_http_receipt'
    with pytest.raises(ValueError, match='historical'):
        fetch_context('open_interest', 'BTCUSDT', start='2025-01-01')


def test_fine_event_bins_are_opt_in_new_schema_without_mutating_old_protocol():
    old = spec()
    events = tuple(replace(t, event_bins=(1, 2, 3, 4)) if t.kind == 'first_event' else t
                   for t in old.tasks)
    new = replace(old, tasks=events)
    assert old.tasks[2].event_bins == (2, 4)
    result = build_context_dataset(bars(), new, cross_assets=False)
    assert result.tasks[2].output_size == 9
    assert new.digest != old.digest
    assert result.schema != build_context_dataset(bars(), old, cross_assets=False).schema
