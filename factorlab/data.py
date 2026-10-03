"""Explicit, versioned public-data ingestion; importing this module never downloads."""
from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from filelock import FileLock

from .protocol import project_path

BASE_URL = 'https://data.binance.vision/data/futures/um/monthly'
HOUR = pd.Timedelta(hours=1)
CANONICAL_COLUMNS = ('symbol', 'open_time', 'close_time', 'available_at',
                     'open', 'high', 'low', 'close', 'volume')
KINDS = {'klines': 'klines', 'mark': 'markPriceKlines', 'funding': 'fundingRate'}
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024


def _source(symbol: str, month: str, kind: str) -> tuple[str, str]:
    if not re.fullmatch(r'[A-Z0-9]{3,30}', symbol):
        raise ValueError('Expected an uppercase Binance symbol')
    if not re.fullmatch(r'\d{4}-(0[1-9]|1[0-2])', month):
        raise ValueError('Expected YYYY-MM')
    if kind not in KINDS:
        raise ValueError(f'Unknown data kind: {kind}')
    directory = KINDS[kind]
    suffix = 'fundingRate' if kind == 'funding' else '1h'
    name = f'{symbol}-{suffix}-{month}.zip'
    interval = '' if kind == 'funding' else '/1h'
    return f'{BASE_URL}/{directory}/{symbol}{interval}/{name}', name


def _checksum(text: str, expected_name: str | None = None) -> str:
    fields = text.strip().split()
    if len(fields) != 2 or not re.fullmatch(r'[a-fA-F0-9]{64}', fields[0]):
        raise ValueError('Invalid SHA-256 checksum file')
    if expected_name is not None and fields[1].lstrip('*') != expected_name:
        raise ValueError('Checksum names another source archive')
    return fields[0].lower()


def _verify(path: Path) -> dict:
    metadata_path = path.with_suffix(path.suffix + '.json')
    checksum_path = path.with_suffix(path.suffix + '.CHECKSUM')
    if not metadata_path.is_file() or not checksum_path.is_file():
        raise ValueError('Archive requires provenance and source checksum sidecars')
    metadata = json.loads(metadata_path.read_text('utf-8'))
    expected = _checksum(checksum_path.read_text('utf-8'), metadata['source_filename'])
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected or actual != metadata['sha256']:
        raise ValueError('Archive checksum mismatch')
    if metadata['bytes'] != path.stat().st_size:
        raise ValueError('Archive length mismatch')
    if f'.{actual}.zip' not in path.name:
        raise ValueError('Archive filename is not SHA-versioned')
    return metadata


def _fetch(url: str) -> bytes:
    """Network access is called exclusively by the explicit download_month operation."""
    with requests.get(url, timeout=(15, 90), stream=True) as response:
        response.raise_for_status()
        chunks, size = [], 0
        for chunk in response.iter_content(1024 * 1024):
            size += len(chunk)
            if size > MAX_ARCHIVE_BYTES:
                raise ValueError('Archive exceeds bounded download size')
            chunks.append(chunk)
    return b''.join(chunks)


def download_month(symbol: str, month: str, kind: str = 'klines',
                   root: Path = Path('D:/FactorResearch/data'), refresh: bool = False) -> Path:
    """Verify source SHA-256, store immutable content versions, return selected archive.

    refresh=False pins the cached version. refresh=True checks the current source
    and keeps every previously observed version instead of overwriting history.
    """
    url, source_name = _source(symbol, month, kind)
    directory = project_path(root) / 'binance_um' / kind / symbol / month
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / '.download.lock')):
        index_path = directory / 'latest.json'
        if index_path.exists() and not refresh:
            index = json.loads(index_path.read_text('utf-8'))
            selected = directory / index['filename']
            if selected.parent.resolve() != directory.resolve():
                raise ValueError('Invalid cached archive path')
            _verify(selected)
            return selected
        checksum_payload = _fetch(url + '.CHECKSUM')
        digest = _checksum(checksum_payload.decode('utf-8'), source_name)
        payload = _fetch(url)
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError('Downloaded archive checksum mismatch')
        # Reject invalid/unsafe containers before accepting an immutable version.
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = archive.infolist()
            if len(members) != 1 or members[0].filename != Path(source_name).with_suffix('.csv').name:
                raise ValueError('Archive does not contain its declared source CSV')
            if members[0].file_size > MAX_ARCHIVE_BYTES:
                raise ValueError('Expanded archive exceeds bounded size')
            if archive.testzip() is not None:
                raise ValueError('Archive CRC mismatch')
        path = directory / f'{Path(source_name).stem}.{digest}.zip'
        revisions = sorted(p.name for p in directory.glob('*.zip') if p != path)
        if path.exists():
            _verify(path)
        else:
            metadata = {'schema_version': 1, 'symbol': symbol, 'month': month, 'kind': kind,
                        'source_url': url, 'source_filename': source_name, 'sha256': digest,
                        'bytes': len(payload), 'ingested_at': datetime.now(UTC).isoformat(),
                        'observed_prior_versions': revisions,
                        'availability_basis': 'assumed_historical_live_delay_not_archive_publish_time'}
            staged = path.with_suffix('.zip.tmp')
            staged.write_bytes(payload)
            for suffix, content in (('.zip.CHECKSUM', checksum_payload),
                                    ('.zip.json', json.dumps(metadata, indent=2).encode())):
                sidecar = path.with_suffix(suffix)
                sidecar_stage = path.with_suffix(suffix + '.tmp')
                sidecar_stage.write_bytes(content)
                sidecar_stage.replace(sidecar)
            # A committed archive always has both committed sidecars, even if interrupted.
            staged.replace(path)
        latest = {'filename': path.name, 'sha256': digest}
        temporary = directory / 'latest.json.tmp'
        temporary.write_text(json.dumps(latest, indent=2), 'utf-8')
        temporary.replace(index_path)
        return path


def _csv(payload: bytes, header_keys: tuple[str, ...]) -> pd.DataFrame:
    if not payload.strip():
        raise ValueError('Empty CSV')
    first = payload.decode('utf-8-sig').splitlines()[0].split(',')[0].strip().lower()
    header = 0 if first in header_keys else None
    frame = pd.read_csv(io.BytesIO(payload), header=header)
    if frame.empty:
        raise ValueError('CSV has no observations')
    return frame


def _timestamps(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors='raise')
    if not np.isfinite(numeric).all() or not (numeric == np.floor(numeric)).all():
        raise ValueError('Timestamp must be finite integer milliseconds')
    result = pd.to_datetime(numeric, unit='ms', utc=True, errors='raise')
    if not result.between(pd.Timestamp('2017-01-01', tz='UTC'),
                          pd.Timestamp('2100-01-01', tz='UTC')).all():
        raise ValueError('Timestamp outside USD-M archive bounds; milliseconds required')
    return result


def _delay(value: float) -> pd.Timedelta:
    if not np.isfinite(value) or value < 0:
        raise ValueError('Availability delay must be finite and nonnegative')
    return pd.Timedelta(seconds=value)


def parse_klines_csv(payload: bytes, symbol: str,
                     availability_delay_seconds: int = 5) -> pd.DataFrame:
    """Parse USD-M 1h CSV; source closes are inclusive millisecond timestamps."""
    raw = _csv(payload, ('open_time', 'open time'))
    if raw.shape[1] != 12:
        raise ValueError('USD-M kline CSV must have twelve source columns')
    frame = pd.DataFrame({'symbol': symbol, 'open_time': _timestamps(raw.iloc[:, 0])})
    frame['close_time'] = frame['open_time'] + HOUR
    source_close = _timestamps(raw.iloc[:, 6])
    if not (source_close == frame['close_time'] - pd.Timedelta(milliseconds=1)).all():
        raise ValueError('Source candle duration is not exactly one hour')
    frame['available_at'] = frame['close_time'] + _delay(availability_delay_seconds)
    for i, name in enumerate(('open', 'high', 'low', 'close', 'volume'), start=1):
        frame[name] = pd.to_numeric(raw.iloc[:, i], errors='raise').astype(float)
    validate_klines(frame)
    frame.attrs['availability_basis'] = 'assumed_live_delay'
    frame.attrs['availability_delay_seconds'] = availability_delay_seconds
    return frame


def parse_funding_csv(payload: bytes, symbol: str,
                      availability_delay_seconds: int = 5) -> pd.DataFrame:
    """Funding event timestamps are observed source events, never an 8-hour schedule."""
    raw = _csv(payload, ('calc_time', 'funding_time', 'fundingtime'))
    if raw.shape[1] != 3:
        raise ValueError('Funding CSV must have time, interval-hours, rate columns')
    frame = pd.DataFrame({'symbol': symbol, 'funding_time': _timestamps(raw.iloc[:, 0])})
    frame['funding_interval_hours'] = pd.to_numeric(raw.iloc[:, 1], errors='raise').astype(float)
    frame['funding_rate'] = pd.to_numeric(raw.iloc[:, 2], errors='raise').astype(float)
    if not np.isfinite(frame[['funding_interval_hours', 'funding_rate']]).all().all():
        raise ValueError('Nonfinite funding values')
    if not (frame['funding_interval_hours'] > 0).all():
        raise ValueError('Funding interval must be positive')
    if frame['funding_time'].duplicated().any() or not frame['funding_time'].is_monotonic_increasing:
        raise ValueError('Duplicate or out-of-order funding events')
    frame['available_at'] = frame['funding_time'] + _delay(availability_delay_seconds)
    frame.attrs['availability_basis'] = 'assumed_live_delay'
    return frame


def load_archive(path: Path | str, symbol: str, kind: str = 'klines',
                 availability_delay_seconds: int = 5) -> pd.DataFrame:
    path = Path(path)
    metadata = _verify(path)
    if metadata['symbol'] != symbol or metadata['kind'] != kind:
        raise ValueError('Archive symbol or data kind differs from requested values')
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) != 1 or not members[0].filename.endswith('.csv'):
            raise ValueError('Archive must contain exactly one CSV')
        if members[0].file_size > MAX_ARCHIVE_BYTES:
            raise ValueError('Expanded archive exceeds bounded size')
        payload = archive.read(members[0])
    if kind == 'funding':
        frame = parse_funding_csv(payload, symbol, availability_delay_seconds)
        event_time = frame['funding_time']
    elif kind in ('klines', 'mark'):
        frame = parse_klines_csv(payload, symbol, availability_delay_seconds)
        event_time = frame['open_time']
    else:
        raise ValueError('Unknown data kind')
    if not (event_time.dt.strftime('%Y-%m') == metadata['month']).all():
        raise ValueError('Archive observations do not match declared month')
    frame.attrs['provenance'] = metadata
    frame.attrs['kind'] = kind
    return frame


def validate_klines(frame: pd.DataFrame) -> dict:
    missing = set(CANONICAL_COLUMNS).difference(frame.columns)
    if missing or frame.empty:
        raise ValueError(f'Empty or incomplete canonical klines: {sorted(missing)}')
    if frame['symbol'].isna().any() or not frame['symbol'].map(
        lambda value: isinstance(value, str) and bool(re.fullmatch(r'[A-Z0-9]{3,30}', value))
    ).all():
        raise ValueError('Invalid symbol')
    for name in ('open_time', 'close_time', 'available_at'):
        if not isinstance(frame[name].dtype, pd.DatetimeTZDtype):
            raise ValueError(f'{name} must be timezone-aware UTC')  # noqa: TRY004
        if str(frame[name].dt.tz) != 'UTC' or frame[name].isna().any():
            raise ValueError(f'{name} must be nonmissing UTC')
    if frame.duplicated(['symbol', 'open_time']).any():
        raise ValueError('Duplicate symbol/hour observation')
    if not (frame['open_time'] == frame['open_time'].dt.floor('h')).all():
        raise ValueError('Open times must align with UTC hours')
    if not (frame['close_time'] == frame['open_time'] + HOUR).all():
        raise ValueError('Canonical close time must be exclusive open plus one hour')
    if not (frame['available_at'] >= frame['close_time']).all():
        raise ValueError('Candle cannot be available before close')
    values = frame[['open', 'high', 'low', 'close', 'volume']].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not (values[:, :4] > 0).all():
        raise ValueError('Prices/volume must be finite; prices strictly positive')
    if (frame['volume'] < 0).any():
        raise ValueError('Volume cannot be negative')
    if (frame['high'] < frame[['open', 'close', 'low']].max(axis=1)).any() or (
        frame['low'] > frame[['open', 'close', 'high']].min(axis=1)
    ).any():
        raise ValueError('OHLC range inconsistent')
    summaries, total_gaps = {}, 0
    for symbol, group in frame.groupby('symbol', sort=False):
        if not group['open_time'].is_monotonic_increasing:
            raise ValueError('Out-of-order symbol/hour observation')
        if not group['available_at'].is_monotonic_increasing:
            raise ValueError('Availability out of order')
        delta = group['open_time'].diff().dropna()
        missing_hours = int(((delta / HOUR) - 1).clip(lower=0).sum())
        total_gaps += missing_hours
        summaries[symbol] = {'rows': len(group), 'missing_hours': missing_hours,
                             'gap_events': int((delta > HOUR).sum()),
                             'start': group['open_time'].iloc[0].isoformat(),
                             'end_exclusive': group['close_time'].iloc[-1].isoformat()}
    return {'rows': len(frame), 'missing_hours': total_gaps, 'symbols': summaries}


def audit_dataset(frame: pd.DataFrame) -> dict:
    result = validate_klines(frame)
    result.update({'universe_mode': 'fixed_survivors_engineering_demo',
                   'historical_universe_complete': False,
                   'gap_policy': 'preserve_missing_hours_no_fill',
                   'historical_availability': 'assumed_close_plus_delay_not_observed',
                   'revision_policy': 'SHA_versions_preserved_when_observed',
                   'limitations': ['Current archives may contain historical revisions unavailable then.',
                                   'Fixed surviving assets do not represent historical listings/delistings.',
                                   'Assumed live delay is not evidence of actual historical delivery latency.',
                                   'Boundary gaps outside the observed first/last hour need expected calendar.']})
    return result
