"""Substantive task-time, leakage, censoring and ingestion checks on tiny inputs."""
import hashlib
import io
import json
import zipfile
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from factorlab import data
from factorlab.discovery import tasks
from factorlab.discovery.contracts import ExperimentSpec, TaskSpec, partition
from factorlab.features import FEATURE_COLUMNS, build_features


def bars(hours=180, skip=(), symbol='BTCUSDT'):
    time = pd.date_range('2025-01-01', periods=hours, freq='h', tz='UTC')
    idx = np.arange(hours)
    close = 100 * np.exp(.0002 * idx + .004 * np.sin(idx * .7))
    frame = pd.DataFrame({
        'symbol': symbol, 'open_time': time, 'close_time': time + data.HOUR,
        'available_at': time + data.HOUR + pd.Timedelta(seconds=5),
        'open': close, 'high': close * 1.00001, 'low': close * .99999,
        'close': close, 'volume': 10. + idx % 13,
    })
    return frame.drop(index=list(skip)).reset_index(drop=True)


def spec(**kwargs):
    base = ExperimentSpec(
        tasks=(TaskSpec('vol', 'volatility', 4), TaskSpec('ret', 'return', 4),
               TaskSpec('event', 'first_event', 4, (2, 4), 3.),
               TaskSpec('state', 'state', 4)),
        primary_task='vol', lookback=4, symbols=('BTCUSDT',),
        start='2025-01-01', train_end='2025-01-03', dev_end='2025-01-05',
        calibration_end='2025-01-07', end='2025-01-09',
    )
    return replace(base, **kwargs)


def index_at(dataset, hour, symbol='BTCUSDT'):
    time = pd.Timestamp('2025-01-01', tz='UTC') + hour * data.HOUR
    return np.flatnonzero((dataset.meta.feature_open_time == time) &
                          (dataset.meta.symbol == symbol))[0]


def test_hand_calculated_origin_targets_and_feature_schema():
    source, config = bars(), spec()
    result = tasks.build_task_dataset(source, config)
    i = index_at(result, 70)
    future = np.diff(np.log(source.close.iloc[71:76].to_numpy()))
    assert result.targets['vol'][i] == pytest.approx(np.linalg.norm(future))
    assert result.targets['ret'][i] == pytest.approx(np.log(source.close.iloc[75] / source.close.iloc[71]))
    assert result.meta.forecast_window_start.iloc[i] == source.open_time.iloc[72]
    assert result.targets['state'][i] == pytest.approx(source.close.iloc[70] / source.close.iloc[46] - 1)
    assert result.features == FEATURE_COLUMNS
    assert not any('state' in name or 'label' in name or 'target' in name for name in result.features)
    np.testing.assert_allclose(result.x[i], build_features(source).loc[67:70, FEATURE_COLUMNS])
    assert result.x.dtype == np.float32
    assert result.meta.feature_open_time.iloc[0] == source.open_time.iloc[51]
    assert result.meta.sample_id.is_unique


def test_future_perturbations_change_labels_without_changing_known_features_or_barriers():
    source, config = bars(), spec()
    baseline = tasks.build_task_dataset(source, config)
    changed = source.copy()
    changed.loc[72:, ['open', 'high', 'low', 'close']] *= 2
    changed.loc[72:, 'volume'] *= 3
    altered = tasks.build_task_dataset(changed, config)
    count = index_at(baseline, 70) + 1
    np.testing.assert_array_equal(baseline.x[:count], altered.x[:count])
    np.testing.assert_array_equal(baseline.targets['state'][:count], altered.targets['state'][:count])
    np.testing.assert_array_equal(baseline.meta.event_barrier_log_width_event[:count],
                                  altered.meta.event_barrier_log_width_event[:count])
    i = count - 1
    assert baseline.targets['vol'][i] != altered.targets['vol'][i]
    assert baseline.targets['ret'][i] != altered.targets['ret'][i]
    # Truncating all future data leaves the same model input with unobserved labels.
    prefix = tasks.build_task_dataset(source.iloc[:71], config)
    np.testing.assert_array_equal(prefix.x[-1], baseline.x[i])
    assert not prefix.observed['vol'][-1]
    assert prefix.observed['state'][-1]


def test_gap_invalidates_horizon_and_restarts_every_feature_and_window():
    source = bars(200, skip=(75,))
    result = tasks.build_task_dataset(source, spec(end='2025-01-10'))
    for hour in range(71, 75):
        i = index_at(result, hour)
        assert not result.observed['vol'][i]
        assert not result.observed['ret'][i]
        assert not result.observed['event'][i]
        assert result.observed['state'][i]
    post_gap = result.meta[result.meta.feature_open_time > pd.Timestamp('2025-01-04T03:00Z')]
    assert post_gap.feature_open_time.iloc[0] == pd.Timestamp('2025-01-06T07:00Z')  # 76+48+3
    assert post_gap.window_start_open_time.iloc[0] == pd.Timestamp('2025-01-06T04:00Z')


@pytest.mark.parametrize('step,up,expected', [(1, True, 1), (2, False, 2), (3, True, 3), (4, False, 4)])
def test_first_event_uses_frozen_log_boundaries_and_bins(step, up, expected):
    source, config = bars(), spec()
    result = tasks.build_task_dataset(source, config)
    i = index_at(result, 70)
    width = result.meta.event_barrier_log_width_event.iloc[i]
    center = source.close.iloc[70]
    # Set all future candles safely inside frozen boundaries, then force one touch.
    source.loc[72:75, ['open', 'high', 'low', 'close']] = center
    source.loc[71 + step, 'high' if up else 'low'] = center * np.exp((1 if up else -1) * width * 1.01)
    result = tasks.build_task_dataset(source, config)
    i = index_at(result, 70)
    assert result.observed['event'][i]
    assert result.targets['event'][i] == expected
    trailing = np.diff(np.log(source.close.iloc[46:71]))
    assert width == pytest.approx(3 * np.std(trailing, ddof=0) * np.sqrt(4))
    assert result.maturity['event'][i] == source.available_at.iloc[75].to_datetime64()


def test_double_touch_is_ambiguous_and_no_touch_requires_full_horizon():
    source, config = bars(), spec()
    reference = tasks.build_task_dataset(source, config)
    width = reference.meta.event_barrier_log_width_event.iloc[index_at(reference, 70)]
    center = source.close.iloc[70]
    source.loc[72:75, ['open', 'high', 'low', 'close']] = center
    clear = tasks.build_task_dataset(source, config)
    i = index_at(clear, 70)
    assert clear.observed['event'][i] and clear.targets['event'][i] == 0
    source.loc[72, 'high'] = center * np.exp(width * 1.01)
    source.loc[72, 'low'] = center * np.exp(-width * 1.01)
    ambiguous = tasks.build_task_dataset(source, config)
    i = index_at(ambiguous, 70)
    assert not ambiguous.observed['event'][i]
    assert np.isnan(ambiguous.targets['event'][i])
    assert ambiguous.meta.label_status_event.iloc[i] == 'ambiguous_double_touch'
    assert ambiguous.observed['vol'][i]  # Ambiguity censors only the event task.
    tail = tasks.build_task_dataset(source.iloc[:72], config)
    assert not tail.observed['event'][index_at(tail, 70)]


def test_first_event_ignores_later_double_touch_after_known_first_event():
    source, config = bars(), spec()
    reference = tasks.build_task_dataset(source, config)
    center = source.close.iloc[70]
    width = reference.meta.event_barrier_log_width_event.iloc[index_at(reference, 70)]
    source.loc[72:75, ['open', 'high', 'low', 'close']] = center
    source.loc[72, 'high'] = center * np.exp(width * 1.01)
    source.loc[73, 'high'] = center * np.exp(width * 1.01)
    source.loc[73, 'low'] = center * np.exp(-width * 1.01)
    result = tasks.build_task_dataset(source, config)
    i = index_at(result, 70)
    assert result.observed['event'][i] and result.targets['event'][i] == 1


def test_availability_overlap_hour_is_excluded_from_event_window():
    source, config = bars(), spec()
    reference = tasks.build_task_dataset(source, config)
    center = source.close.iloc[70]
    width = reference.meta.event_barrier_log_width_event.iloc[index_at(reference, 70)]
    source.loc[72:75, ['open', 'high', 'low', 'close']] = center
    # The next bar began five seconds before availability: both touches ignored.
    source.loc[71, 'high'] = center * np.exp(width * 2)
    source.loc[71, 'low'] = center * np.exp(-width * 2)
    result = tasks.build_task_dataset(source, config)
    i = index_at(result, 70)
    assert result.observed['event'][i] and result.targets['event'][i] == 0
    assert result.meta.forecast_window_start.iloc[i] > result.meta.available_at.iloc[i]
    assert result.maturity['event'][i] == (
        result.meta.forecast_window_start.iloc[i] + 4 * data.HOUR + pd.Timedelta(seconds=5)
    ).to_datetime64()


@pytest.mark.parametrize('up,expected', [(True, 1), (False, 2)])
def test_future_window_open_beyond_barrier_has_known_direction_despite_double_extremes(up, expected):
    source, config = bars(), spec()
    reference = tasks.build_task_dataset(source, config)
    center = source.close.iloc[70]
    width = reference.meta.event_barrier_log_width_event.iloc[index_at(reference, 70)]
    source.loc[72:75, ['open', 'high', 'low', 'close']] = center
    source.loc[72, 'open'] = center * np.exp((1 if up else -1) * width * 1.1)
    source.loc[72, 'high'] = center * np.exp(width * 2)
    source.loc[72, 'low'] = center * np.exp(-width * 2)
    result = tasks.build_task_dataset(source, config)
    i = index_at(result, 70)
    assert result.observed['event'][i]
    assert result.targets['event'][i] == expected


def test_unknown_pre_window_close_is_used_only_as_label_baseline():
    source, config = bars(), spec()
    reference = tasks.build_task_dataset(source, config)
    changed = source.copy()
    changed.loc[71, ['open', 'high', 'low', 'close']] *= 1.02
    result = tasks.build_task_dataset(changed, config)
    i = index_at(result, 70)
    np.testing.assert_array_equal(reference.x[i], result.x[i])
    assert reference.meta.event_barrier_log_width_event.iloc[i] == result.meta.event_barrier_log_width_event.iloc[i]
    assert result.targets['ret'][i] == pytest.approx(np.log(source.close.iloc[75] / changed.close.iloc[71]))
    assert result.targets['ret'][i] != reference.targets['ret'][i]


def test_maturity_tail_masks_and_partition_purge():
    source, config = bars(), spec()
    result = tasks.build_task_dataset(source, config)
    for task in config.tasks:
        if task.predictive:
            assert not result.observed[task.name][-5:].any()
            assert result.observed[task.name][-6]
            assert result.maturity[task.name][-1] == (
                source.available_at.iloc[-1] + 5 * data.HOUR
            ).to_datetime64()
            origin = index_at(result, 70)
            ready = source.available_at.iloc[75]
            assert not result.task_mask(task.name, ready - pd.Timedelta(nanoseconds=1))[origin]
            assert result.task_mask(task.name, ready)[origin]
    train = partition(result, config, 'train')
    assert len(train.meta) == 0  # Warm-up ends after the short train interval.
    dev = partition(result, config, 'development')
    assert (dev.maturity['vol'][dev.observed['vol']] <= pd.Timestamp(config.dev_end).to_datetime64()).all()
    assert len(dev.meta) > np.count_nonzero(dev.observed['vol'])
    assert not next(t for t in result.tasks if t.name == 'state').predictive


def test_symbol_isolation_sorting_and_rejected_input():
    btc, eth = bars(), bars(symbol='ETHUSDT')
    eth[['open', 'high', 'low', 'close']] *= 4
    config = spec(symbols=('ETHUSDT', 'BTCUSDT'))
    combined = pd.concat([eth, btc]).sort_values('open_time', kind='stable')
    combined.index = np.zeros(len(combined), dtype=int)
    result = tasks.build_task_dataset(combined, config)
    assert result.meta.equals(result.meta.sort_values(['available_at', 'symbol']).reset_index(drop=True))
    single = tasks.build_task_dataset(btc, spec())
    np.testing.assert_array_equal(result.x[result.meta.symbol == 'BTCUSDT'], single.x)
    wrong_delay = btc.copy()
    wrong_delay['available_at'] += pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match='availability'):
        tasks.build_task_dataset(wrong_delay, spec())
    with pytest.raises(ValueError, match='universe'):
        tasks.build_task_dataset(btc, config)
    with pytest.raises(ValueError, match='Out-of-order'):
        tasks.build_task_dataset(btc.iloc[::-1], spec())


def test_no_finite_windows_produces_valid_empty_dataset():
    source = bars(100)
    source['volume'] = 0.
    result = tasks.build_task_dataset(source, spec())
    assert result.x.shape == (0, 4, len(FEATURE_COLUMNS))
    assert all(v.shape == (0,) for v in result.targets.values())


def archive_fixture(monkeypatch, tmp_path):
    source = bars(240)
    lines = []
    for row in source.itertuples():
        start = int(row.open_time.timestamp() * 1000)
        lines.append(f'{start},{row.open},{row.high},{row.low},{row.close},{row.volume},'
                     f'{start + 3599999},0,1,0,0,0')
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('BTCUSDT-1h-2025-01.csv', '\n'.join(lines))
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    calls = []

    def fetch(url):
        calls.append(url)
        return (f'{digest}  BTCUSDT-1h-2025-01.zip\n'.encode()
                if url.endswith('.CHECKSUM') else payload)

    monkeypatch.setattr(data, '_fetch', fetch)
    original = data.download_month
    monkeypatch.setattr(tasks, 'download_month', lambda symbol, month, kind:
                        original(symbol, month, kind, root=tmp_path / 'archives'))
    config = spec(start='2025-01-04', train_end='2025-01-05', dev_end='2025-01-06',
                  calibration_end='2025-01-07', end='2025-01-08')
    return config, calls


def test_prepare_verified_kline_only_manifest_reuse_and_frozen_coverage(monkeypatch, tmp_path):
    config, calls = archive_fixture(monkeypatch, tmp_path)
    output = tmp_path / 'prepared'
    first = tasks.prepare_bars(config, output)
    assert len(calls) == 2 and all('/klines/' in url for url in calls)
    manifest = json.loads((output / 'bars_manifest.json').read_text())
    assert manifest['parquet_sha256'] == hashlib.sha256((output / 'bars.parquet').read_bytes()).hexdigest()
    assert len(manifest['archives']) == 1
    assert manifest['audit']['missing_hours_in_requested_calendar']['BTCUSDT'] == 0
    cached = tasks.prepare_bars(config, output)
    pd.testing.assert_frame_equal(first, cached)
    assert len(calls) == 2
    with pytest.raises(ValueError, match='manifest differs'):
        tasks.prepare_bars(replace(config, end='2025-01-09'), output)
    assert len(calls) == 2
    (output / 'bars.parquet').write_bytes((output / 'bars.parquet').read_bytes() + b'tampered')
    with pytest.raises(ValueError, match='checksum'):
        tasks.prepare_bars(config, output)


def test_prepare_cached_source_tampering_is_rejected(monkeypatch, tmp_path):
    config, _ = archive_fixture(monkeypatch, tmp_path)
    output = tmp_path / 'prepared'
    tasks.prepare_bars(config, output)
    manifest = json.loads((output / 'bars_manifest.json').read_text())
    from pathlib import Path
    archive = Path(manifest['archives'][0]['path'])
    archive.write_bytes(archive.read_bytes() + b'tamper')
    with pytest.raises(ValueError, match='checksum'):
        tasks.prepare_bars(config, output)


def test_prepare_rejects_c_drive_before_network(monkeypatch):
    monkeypatch.setattr(tasks, 'download_month', lambda *_: pytest.fail('unexpected network'))
    with pytest.raises(ValueError, match='Project path'):
        tasks.prepare_bars(spec(), 'C:/FactorResearch/forbidden')
