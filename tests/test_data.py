import hashlib
import io
import json
import zipfile

import pandas as pd
import pytest

from factorlab import data


def kline_payload(hours=4, skip=(), header=True, price=100):
    lines = []
    if header:
        lines.append('open_time,open,high,low,close,volume,close_time,quote_volume,'
                     'count,taker_volume,taker_quote,ignore')
    start = pd.Timestamp('2025-01-01', tz='UTC')
    for i in range(hours):
        if i in skip:
            continue
        t = int((start + pd.Timedelta(hours=i)).timestamp() * 1000)
        p = price + i
        lines.append(f'{t},{p},{p+2},{p-2},{p+1},{10+i},{t+3599999},0,1,0,0,0')
    return ('\n'.join(lines) + '\n').encode()


def zipped(payload):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('BTCUSDT-1h-2025-01.csv', payload)
    return buffer.getvalue()


def source_mock(monkeypatch, payload):
    archive = zipped(payload)
    digest = hashlib.sha256(archive).hexdigest()
    calls = []

    def fetch(url):
        calls.append(url)
        if url.endswith('.CHECKSUM'):
            return f'{digest}  BTCUSDT-1h-2025-01.zip\n'.encode()
        return archive

    monkeypatch.setattr(data, '_fetch', fetch)
    return digest, calls


@pytest.mark.parametrize('header', [True, False])
def test_parse_times_and_gap_audit(header):
    frame = data.parse_klines_csv(kline_payload(5, skip=(2,), header=header), 'BTCUSDT')
    assert frame['close_time'].iloc[0] == pd.Timestamp('2025-01-01T01:00:00Z')
    assert frame['available_at'].iloc[0] == pd.Timestamp('2025-01-01T01:00:05Z')
    audit = data.audit_dataset(frame)
    assert audit['missing_hours'] == 1
    assert audit['symbols']['BTCUSDT']['gap_events'] == 1
    assert not audit['historical_universe_complete']
    assert len(frame) == 4  # Never fill the missing hour.


@pytest.mark.parametrize('failure', ['duplicate', 'order', 'inf', 'negative_price',
                                    'bad_ohlc', 'bad_time', 'early_available',
                                    'availability_order'])
def test_reject_corrupt_canonical(failure):
    frame = data.parse_klines_csv(kline_payload(), 'BTCUSDT')
    if failure == 'duplicate':
        frame = pd.concat([frame, frame.iloc[[0]]])
    elif failure == 'order':
        frame = frame.iloc[::-1]
    elif failure == 'inf':
        frame.loc[0, 'volume'] = float('inf')
    elif failure == 'negative_price':
        frame.loc[0, 'close'] = -1
    elif failure == 'bad_ohlc':
        frame.loc[0, 'high'] = 1
    elif failure == 'bad_time':
        frame.loc[0, 'close_time'] += pd.Timedelta(seconds=1)
    elif failure == 'early_available':
        frame.loc[0, 'available_at'] = frame.loc[0, 'open_time']
    elif failure == 'availability_order':
        frame.loc[0, 'available_at'] = frame['available_at'].iloc[-1] + pd.Timedelta(hours=1)
    with pytest.raises(ValueError):
        data.validate_klines(frame)


def test_reject_non_hour_source_and_microsecond_units():
    raw = kline_payload(header=False).replace(b'1735693199999', b'1735693199998')
    with pytest.raises(ValueError, match='duration'):
        data.parse_klines_csv(raw, 'BTCUSDT')
    raw = kline_payload(header=False).replace(b'1735689600000', b'1735689600000000')
    with pytest.raises((ValueError, OverflowError)):
        data.parse_klines_csv(raw, 'BTCUSDT')


def test_funding_preserves_real_times_and_interval():
    payload = b'calc_time,funding_interval_hours,last_funding_rate\n1735689600000,8,0.001\n1735704000000,4,-0.002\n'
    frame = data.parse_funding_csv(payload, 'BTCUSDT')
    assert frame['funding_time'].iloc[1] - frame['funding_time'].iloc[0] == pd.Timedelta(hours=4)
    assert frame['funding_interval_hours'].tolist() == [8, 4]
    assert frame['funding_rate'].tolist() == [.001, -.002]
    assert frame['available_at'].iloc[1] == frame['funding_time'].iloc[1] + pd.Timedelta(seconds=5)
    with pytest.raises(ValueError, match='out-of-order'):
        data.parse_funding_csv(payload + b'1735689600000,8,0.001\n', 'BTCUSDT')


def test_versioned_download_cache_and_revision(monkeypatch, tmp_path):
    digest, calls = source_mock(monkeypatch, kline_payload())
    path = data.download_month('BTCUSDT', '2025-01', root=tmp_path)
    assert digest in path.name
    assert len(calls) == 2
    first_bytes = path.read_bytes()
    frame = data.load_archive(path, 'BTCUSDT')
    assert len(frame) == 4
    assert 'ingested_at' in frame.attrs['provenance']
    assert frame['available_at'].max() < pd.Timestamp(frame.attrs['provenance']['ingested_at'])
    assert data.download_month('BTCUSDT', '2025-01', root=tmp_path) == path
    assert len(calls) == 2
    _, revision_calls = source_mock(monkeypatch, kline_payload(price=200))
    revision = data.download_month('BTCUSDT', '2025-01', root=tmp_path, refresh=True)
    assert len(revision_calls) == 2
    assert revision != path
    assert path.read_bytes() == first_bytes
    metadata = json.loads(revision.with_suffix('.zip.json').read_text())
    assert path.name in metadata['observed_prior_versions']
    assert data.load_archive(path, 'BTCUSDT')['open'].iloc[0] == 100


def test_checksum_failure_on_download_and_cached_load(monkeypatch, tmp_path):
    source_mock(monkeypatch, kline_payload())
    path = data.download_month('BTCUSDT', '2025-01', root=tmp_path)
    path.write_bytes(path.read_bytes() + b'tamper')
    with pytest.raises(ValueError, match='checksum'):
        data.load_archive(path, 'BTCUSDT')
    with pytest.raises(ValueError, match='checksum'):
        data.download_month('BTCUSDT', '2025-01', root=tmp_path)
    monkeypatch.setattr(data, '_fetch', lambda url: (
        b'0' * 64 + b'  BTCUSDT-1h-2025-01.zip\n' if url.endswith('.CHECKSUM') else b'wrong'
    ))
    with pytest.raises(ValueError, match='checksum'):
        data.download_month('BTCUSDT', '2025-01', root=tmp_path, refresh=True)


def test_archive_month_symbol_kind_and_source_names(monkeypatch, tmp_path):
    source_mock(monkeypatch, kline_payload())
    path = data.download_month('BTCUSDT', '2025-01', root=tmp_path)
    with pytest.raises(ValueError, match='symbol'):
        data.load_archive(path, 'ETHUSDT')
    with pytest.raises(ValueError, match='kind'):
        data.load_archive(path, 'BTCUSDT', 'mark')
    assert '/markPriceKlines/BTCUSDT/1h/' in data._source('BTCUSDT', '2025-01', 'mark')[0]
    assert '/fundingRate/BTCUSDT/BTCUSDT-fundingRate-' in data._source('BTCUSDT', '2025-01', 'funding')[0]
    with pytest.raises(ValueError):
        data.download_month('../BTCUSDT', '2025-01', root=tmp_path)
    with pytest.raises(ValueError):
        data.download_month('BTCUSDT', '2025-13', root=tmp_path)


def test_c_drive_writes_rejected(monkeypatch):
    monkeypatch.setattr(data, '_fetch', lambda _: pytest.fail('Must reject before network'))
    with pytest.raises(ValueError, match='Project path'):
        data.download_month('BTCUSDT', '2025-01', root='C:/FactorResearch/data')
