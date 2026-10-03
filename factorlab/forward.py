"""One-shot, public USD-M observation; no scheduling, credentials or order API."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from filelock import FileLock

from .data import HOUR, parse_klines_csv, validate_klines
from .dataset import Dataset
from .features import FEATURE_COLUMNS, build_features
from .protocol import Protocol, fingerprint
from .shadow import ShadowLedger, _safe_path, _time, verify_bundle
from .training import FittedModel

BASE_URL = 'https://fapi.binance.com'


def wall_now():
    return pd.Timestamp.now(tz='UTC')


def load_frozen_model(bundle, expected_hash: str, protocol: Protocol):
    manifest = verify_bundle(bundle)  # Before any model deserialization.
    if manifest['bundle_hash'] != expected_hash:
        raise ValueError('Bundle differs from independently retained expected hash')
    if manifest['metadata'].get('protocol_hash') != protocol.digest:
        raise ValueError('Frozen model protocol_hash differs from observation protocol')
    root = _safe_path(bundle)
    path = _safe_path(root / manifest['metadata'].get('model_directory', '.'))
    if not path.is_relative_to(root):
        raise ValueError('Model directory must be inside verified bundle')
    model = FittedModel.load(path)
    if not set(model.feature_names).issubset(FEATURE_COLUMNS):
        raise ValueError('Model features differ from causal inference contract')
    if model.spec is not None and model.spec.lookback != protocol.lookback:
        raise ValueError('Model lookback differs from frozen protocol')
    if model.diagnostics['prediction_clip'] != protocol.prediction_clip:
        raise ValueError('Model prediction limit differs from frozen protocol')
    return model, manifest


def canonical_closed(rows, symbol, server_time, received_at, protocol):
    """Filter forming bars, reject future opens, require fresh contiguous history."""
    server, received = _time(server_time), _time(received_at)
    if received > wall_now() or server > received + pd.Timedelta(seconds=5):
        raise ValueError('Future receipt or exchange clock ahead of local clock')
    if received - server > pd.Timedelta(seconds=60):
        raise ValueError('Stale exchange clock')
    if not isinstance(rows, list) or not rows or any(
        not isinstance(row, list) or len(row) != 12 for row in rows
    ):
        raise ValueError('Expected public twelve-column kline rows')
    payload = ('\n'.join(','.join(map(str, row)) for row in rows)).encode()
    frame = parse_klines_csv(payload, symbol, protocol.availability_delay_seconds)
    if (frame.open_time > min(server, received).floor('h')).any():
        raise ValueError('Future candle open')
    frame = frame.loc[frame.available_at <= min(server, received)].copy()
    minimum = max(200, protocol.lookback + 49)
    if len(frame) < minimum:
        raise ValueError('Insufficient closed hourly history')
    expected_close = (min(server, received) - pd.Timedelta(
        seconds=protocol.availability_delay_seconds)).floor('h')
    if frame.close_time.iloc[-1] != expected_close:
        raise ValueError('Latest eligible closed candle is missing')
    # All downloaded past candles were actually observed at receipt, not in the past.
    frame['available_at'] = received
    if validate_klines(frame)['missing_hours']:
        raise ValueError('Gap in forward inference history')
    frame.attrs['provenance'] = {'symbol': symbol, 'source_url': BASE_URL + '/fapi/v1/klines',
                               'response_sha256': fingerprint(rows),
                               'server_time': server.isoformat(),
                               'received_at': received.isoformat(), 'rows': len(frame)}
    return frame


class PublicBinance:
    """Fixed-host GET-only client; redirects and all nonpublic endpoints prohibited."""

    def _get(self, endpoint, params=None):
        if endpoint not in ('/fapi/v1/time', '/fapi/v1/klines'):
            raise ValueError('Only public time/klines endpoints are implemented')
        with requests.get(BASE_URL + endpoint, params=params, timeout=(10, 20),
                          allow_redirects=False) as response:
            if 300 <= response.status_code < 400:
                raise ValueError('Public endpoint redirect rejected')
            response.raise_for_status()  # Regional restrictions fail without fallback.
            if len(response.content) > 2_000_000:
                raise ValueError('Public response exceeds size bound')
            result = response.json()
        return result, wall_now()

    def _server(self):
        value, received = self._get('/fapi/v1/time')
        if type(value.get('serverTime')) is not int:
            raise ValueError('Expected integer exchange server milliseconds')
        server = pd.Timestamp(value['serverTime'], unit='ms', tz='UTC')
        if abs(server - received) > pd.Timedelta(seconds=5):
            raise ValueError('Exchange/local clock skew exceeds five seconds')
        return server

    def closed_bars(self, symbol, protocol):
        if not re.fullmatch(r'[A-Z0-9]{3,30}', symbol):
            raise ValueError('Invalid public market symbol')
        server = self._server()
        boundary = (server - pd.Timedelta(seconds=protocol.availability_delay_seconds)).floor('h')
        rows, received = self._get('/fapi/v1/klines', {
            'symbol': symbol, 'interval': '1h', 'endTime': boundary.value // 1_000_000 - 1,
            'limit': max(200, protocol.lookback + 49)})
        return canonical_closed(rows, symbol, server, received, protocol)

    def outcome(self, symbol, entry, end, delay):
        server = self._server()
        if end + pd.Timedelta(seconds=delay) > min(server, wall_now()):
            raise ValueError('Outcome has not matured at exchange/local clock')
        rows, received = self._get('/fapi/v1/klines', {
            'symbol': symbol, 'interval': '1h', 'startTime': entry.value // 1_000_000,
            'endTime': end.value // 1_000_000, 'limit': 13})
        if not isinstance(rows, list) or any(len(row) != 12 for row in rows):
            raise ValueError('Invalid outcome candle response')
        times = [pd.Timestamp(row[0], unit='ms', tz='UTC') for row in rows]
        if times != list(pd.date_range(entry, end, freq='h')):
            raise ValueError('Incomplete/gapped outcome open prices')
        prices = np.asarray([float(row[1]) for row in rows])
        if not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError('Invalid outcome open prices')
        return float(prices[-1] / prices[0] - 1), {
            'source_url': BASE_URL + '/fapi/v1/klines', 'symbol': symbol,
            'entry_time': entry.isoformat(), 'label_end': end.isoformat(),
            'server_time': server.isoformat(), 'received_at': received.isoformat(),
            'response_sha256': fingerprint(rows), 'entry_open': prices[0], 'end_open': prices[-1]}


def inference_dataset(bars, protocol, observed_at):
    """Latest causal windows, without outcome prices, labels or training calendar."""
    observed = _time(observed_at)
    if observed > wall_now() or (bars.available_at > observed).any():
        raise ValueError('Inference availability exceeds actual observation')
    if set(bars.symbol) != set(protocol.symbols):
        raise ValueError('Inference requires the exact frozen asset universe')
    if (bars.available_at < bars.close_time + pd.Timedelta(
        seconds=protocol.availability_delay_seconds)).any():
        raise ValueError('Inference input violates frozen close delay')
    if validate_klines(bars)['missing_hours']:
        raise ValueError('Gap in inference history')
    frame = build_features(bars)
    samples, scales, metadata = [], [], []
    entry = observed.floor('h') + HOUR
    for symbol, group in frame.groupby('symbol', sort=True):
        window = group[list(FEATURE_COLUMNS)].tail(protocol.lookback).to_numpy(np.float32)
        if window.shape != (protocol.lookback, len(FEATURE_COLUMNS)) or not np.isfinite(window).all():
            raise ValueError('Latest causal feature window is incomplete/nonfinite')
        row = group.iloc[-1]
        samples.append(window)
        scales.append(max(float(row.volatility_24h) * np.sqrt(12), protocol.volatility_floor))
        metadata.append({'symbol': symbol, 'feature_open_time': row.open_time,
                         'feature_close_time': row.close_time, 'available_at': observed,
                         'entry_time': entry, 'label_end': entry + 12 * HOUR})
    return Dataset(np.stack(samples), np.zeros(len(samples), np.float32),
                   np.asarray(scales, np.float32), pd.DataFrame(metadata), FEATURE_COLUMNS)


def _durable_publish(path, writer, *, binary=False):
    # A terminated writer leaves only an ignored staging file, never a partial final.
    with tempfile.NamedTemporaryFile(mode='wb' if binary else 'w',
            encoding=None if binary else 'utf-8', prefix='.prepared-', suffix='.tmp',
            dir=path.parent, delete=False) as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
        stage = Path(stream.name)
    stage.rename(path)  # Adapter lock serializes; Windows refuses existing destinations.


def _journal(ledger, content):
    body = {**content, 'protocol_hash': ledger.config['protocol_hash'],
            'model_hash': ledger.config['model_hash'], 'observation_mode': 'forward'}
    digest = fingerprint(body)
    name = ('prediction-' + body['decision_id']) if body['kind'] == 'prediction' else digest
    path = _safe_path(ledger.directory / 'provenance' / (name + '.json'))
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        if json.loads(path.read_text('utf-8')) != {**body, 'sha256': digest}:
            raise ValueError('Existing prepared provenance differs; review interrupted attempt')
    else:
        _durable_publish(path, lambda stream: json.dump(
            {**body, 'sha256': digest}, stream, indent=2, allow_nan=False))
    return path


def _input_artifact(ledger, decision, x=None, scale=None):
    path = _safe_path(ledger.directory / 'inputs' / (decision + '.npz'))
    path.parent.mkdir(exist_ok=True)
    if not path.exists() and x is not None:
        _durable_publish(path, lambda stream: np.savez(stream, x=x, scale=scale), binary=True)
    if not path.is_file() or path.stat().st_size > 1_000_000:
        raise ValueError('Missing or oversized inference tensor artifact')
    with np.load(path, allow_pickle=False) as arrays:
        stored_x, stored_scale = arrays['x'], arrays['scale']
        if (stored_x.dtype != np.float32 or stored_scale.dtype != np.float32 or
                stored_scale.shape != () or not np.isfinite(stored_x).all() or
                not np.isfinite(stored_scale) or stored_scale <= 0):
            raise ValueError('Invalid inference tensor artifact')
        if x is not None and (not np.array_equal(stored_x, x) or not np.array_equal(stored_scale, scale)):
            raise ValueError('Existing inference tensor differs from retry')
        input_hash = hashlib.sha256(stored_x.tobytes() + stored_scale.tobytes()).hexdigest()
    return {'input_artifact': path.relative_to(ledger.directory).as_posix(),
            'input_artifact_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'input_hash': input_hash}


def _known_retry(ledger, row, decision, input_hash, prediction, feature_close):
    path = _safe_path(ledger.directory / 'provenance' / ('prediction-' + decision + '.json'))
    if not path.is_file():
        raise ValueError('Existing decision has no known inference provenance')
    body = json.loads(path.read_text('utf-8'))
    digest = body.pop('sha256', None)
    expected = {'kind': 'prediction', 'decision_id': decision, 'input_hash': input_hash,
                'prediction': prediction, 'symbol': row['symbol'],
                'available_at': pd.Timestamp(row['available_at'], tz='UTC').isoformat(),
                'entry_time': pd.Timestamp(row['entry_time'], tz='UTC').isoformat(),
                'label_end': pd.Timestamp(row['label_end'], tz='UTC').isoformat(),
                'feature_close_time': feature_close.isoformat(), 'observation_mode': 'forward',
                'model_hash': ledger.config['model_hash'], 'protocol_hash': ledger.config['protocol_hash']}
    artifact = _input_artifact(ledger, decision)
    if artifact['input_hash'] != input_hash:
        raise ValueError('Stored inference tensor differs from known retry')
    expected.update(artifact)
    if (fingerprint(body) != digest or row['prediction'] != prediction or
            any(body.get(key) != value for key, value in expected.items())):
        raise ValueError('Existing decision is not an exact known inference retry')


def observe_once(bundle, expected_hash, ledger_directory, protocol, *, client=None, settle=False):
    model, manifest = load_frozen_model(bundle, expected_hash, protocol)
    client = client or PublicBinance()
    ledger = ShadowLedger(ledger_directory, protocol.digest, manifest['bundle_hash'],
                          observation_mode='forward',
                          availability_delay_seconds=protocol.availability_delay_seconds)
    with FileLock(str(_safe_path(ledger.directory / '.forward.lock'))):
        settled = []
        if settle:
            with ledger._connect() as db:
                pending = db.execute('SELECT p.* FROM predictions p LEFT JOIN outcomes o '
                                     'USING(decision_id) WHERE o.decision_id IS NULL '
                                     'AND p.label_end + ? <= ? ORDER BY p.label_end LIMIT 20',
                                     (protocol.availability_delay_seconds * 1_000_000_000,
                                      wall_now().value)).fetchall()
            for row in pending:
                entry, end = (_time(pd.Timestamp(row[name], tz='UTC'))
                              for name in ('entry_time', 'label_end'))
                value, provenance = client.outcome(row['symbol'], entry, end,
                                                    protocol.availability_delay_seconds)
                _journal(ledger, {'kind': 'outcome', 'decision_id': row['decision_id'],
                                  'realized_return': value, 'source': provenance})
                settled.append(ledger.append_outcome(row['decision_id'], value,
                    end + pd.Timedelta(seconds=protocol.availability_delay_seconds), wall_now()))
        frames = [client.closed_bars(symbol, protocol) for symbol in protocol.symbols]
        sources = [frame.attrs['provenance'] for frame in frames]
        observed = wall_now()  # Common conservative availability after every response.
        data = inference_dataset(pd.concat(frames, ignore_index=True), protocol, observed)
        values = model.predict(data)
        if values.shape != (len(data.meta),) or not np.isfinite(values).all() or (values < -1).any():
            raise ValueError('Model returned invalid simple-return forecasts')
        appended, skipped = [], []
        for index, row in data.meta.iterrows():
            decision = fingerprint({'model_hash': manifest['bundle_hash'], 'symbol': row.symbol,
                                    'feature_close_time': row.feature_close_time.isoformat()})
            input_hash = hashlib.sha256(data.x[index].tobytes() + data.scale[index].tobytes()).hexdigest()
            with ledger._connect() as db:
                existing = db.execute('SELECT * FROM predictions WHERE decision_id=?', (decision,)).fetchone()
            if existing is not None:
                if existing['symbol'] != row.symbol:
                    raise ValueError('Existing decision is not an exact known inference retry')
                _known_retry(ledger, existing, decision, input_hash, float(values[index]), row.feature_close_time)
                skipped.append(decision)
                continue
            artifact = _input_artifact(ledger, decision, data.x[index], data.scale[index])
            evidence = _safe_path(ledger.directory / 'provenance' / ('prediction-' + decision + '.json'))
            available, entry, end = observed, row.entry_time, row.label_end
            if evidence.exists():  # Retry a prepared attempt interrupted before its commit.
                body = json.loads(evidence.read_text('utf-8'))
                available, entry, end = map(_time, (body['available_at'], body['entry_time'], body['label_end']))
                prepared = {'symbol': row.symbol, 'prediction': float(values[index]),
                            'available_at': available.value, 'entry_time': entry.value, 'label_end': end.value}
                _known_retry(ledger, prepared, decision, input_hash, float(values[index]), row.feature_close_time)
            else:
                _journal(ledger, {'kind': 'prediction', 'decision_id': decision, **artifact,
                                  'symbol': row.symbol, 'entry_time': entry.isoformat(),
                                  'label_end': end.isoformat(),
                                  'feature_close_time': row.feature_close_time.isoformat(),
                                  'available_at': available.isoformat(), 'prediction': float(values[index]),
                                  'sources': sources})
            appended.append(ledger.append_prediction(decision, row.symbol, available, entry,
                                                      end, float(values[index]), wall_now()))
        summary = ledger.summary(wall_now())
        return {'appended': appended, 'idempotent_skips': skipped, 'settled': settled,
                'settlement_limit': 20, 'mature_pending_remaining': summary['mature_without_outcome'],
                'summary': summary}
