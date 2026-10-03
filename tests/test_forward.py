import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from factorlab import forward, shadow
from factorlab.features import FEATURE_COLUMNS
from factorlab.forward import canonical_closed, inference_dataset, load_frozen_model, observe_once
from factorlab.protocol import Protocol
from factorlab.shadow import ShadowLedger, freeze_bundle
from factorlab.training import FittedModel


@pytest.fixture
def clock(monkeypatch):
    clock = [pd.Timestamp('2026-10-03T10:30:00Z')]
    monkeypatch.setattr(forward, 'wall_now', lambda: clock[0])
    monkeypatch.setattr(shadow, '_wall_now', lambda: clock[0])
    return clock


@pytest.fixture
def protocol():
    return replace(Protocol(), symbols=('BTCUSDT',))


def rows(close='2026-10-03T10:00:00Z', count=200):
    result = []
    for i, stamp in enumerate(pd.date_range(end=pd.Timestamp(close) - pd.Timedelta(hours=1),
                                          periods=count, freq='h')):
        price = 100 + i * .1 + np.sin(i) * .2
        result.append([stamp.value // 1_000_000, str(price), str(price + 1), str(price - 1),
                       str(price + .1), str(100 + i % 17),
                       (stamp + pd.Timedelta(hours=1)).value // 1_000_000 - 1,
                       '0', 10, '0', '0', '0'])
    return result


def bars(protocol, clock):
    return canonical_closed(rows(), 'BTCUSDT', clock[0] - pd.Timedelta(seconds=1),
                            clock[0], protocol)


@pytest.fixture
def bundle(tmp_path, protocol):
    source = tmp_path / 'model'
    n = len(FEATURE_COLUMNS)
    FittedModel(None, FEATURE_COLUMNS, np.zeros(n, np.float32), np.ones(n, np.float32),
                np.zeros(n, np.float32), .1, None, 0,
                {'prediction_clip': protocol.prediction_clip}).save(source)
    destination = tmp_path / 'frozen'
    manifest = freeze_bundle(source, destination, {'protocol_hash': protocol.digest})
    return destination, manifest['bundle_hash']


def test_verify_before_load_and_trusted_reference(bundle, protocol, monkeypatch):
    path, digest = bundle
    calls = []
    monkeypatch.setattr(FittedModel, 'load', lambda path: calls.append(path))
    with pytest.raises(ValueError, match='expected hash'):
        load_frozen_model(path, 'f' * 64, protocol)
    with pytest.raises(ValueError, match='protocol_hash'):
        load_frozen_model(path, digest, Protocol())
    (path / 'arrays.npz').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='tree/hash'):
        load_frozen_model(path, digest, protocol)
    assert calls == []


def test_closed_only_and_actual_receipt_availability(protocol, clock):
    source = rows(count=201)
    # Append the presently forming 10:00 candle; its values must not be features.
    source += rows(close='2026-10-03T11:00:00Z', count=1)
    result = canonical_closed(source, 'BTCUSDT', clock[0], clock[0], protocol)
    assert len(result) == 201
    assert result.close_time.max() == pd.Timestamp('2026-10-03T10:00:00Z')
    assert (result.available_at == clock[0]).all()
    assert result.attrs['provenance']['received_at'] == clock[0].isoformat()


def test_future_gap_stale_and_delay_fail_closed(protocol, clock):
    with pytest.raises(ValueError, match='Future receipt'):
        canonical_closed(rows(), 'BTCUSDT', clock[0], clock[0] + pd.Timedelta(seconds=1), protocol)
    with pytest.raises(ValueError, match='Future candle'):
        canonical_closed(rows() + rows(close='2026-10-03T12:00:00Z', count=1),
                         'BTCUSDT', clock[0], clock[0], protocol)
    source = rows(count=201)
    source.pop(100)
    with pytest.raises(ValueError, match='Gap'):
        canonical_closed(source, 'BTCUSDT', clock[0], clock[0], protocol)
    with pytest.raises(ValueError, match='Latest eligible'):
        canonical_closed(rows(close='2026-10-03T09:00:00Z'), 'BTCUSDT', clock[0], clock[0], protocol)
    clock[0] = pd.Timestamp('2026-10-03T10:00:04Z')
    result = canonical_closed(rows(count=201), 'BTCUSDT', clock[0], clock[0], protocol)
    assert result.close_time.max() == pd.Timestamp('2026-10-03T09:00:00Z')


def test_inference_has_no_observed_future_labels_or_training_calendar(protocol, clock):
    frame = bars(protocol, clock)
    data = inference_dataset(frame, protocol, clock[0])
    assert data.x.shape == (1, 48, len(FEATURE_COLUMNS)) and np.isfinite(data.x).all()
    assert np.array_equal(data.y, [0])  # Shape-only dummy: never read by predict.
    assert data.meta.entry_time.iloc[0] == pd.Timestamp('2026-10-03T11:00:00Z')
    assert data.meta.label_end.iloc[0] == pd.Timestamp('2026-10-03T23:00:00Z')
    assert 'label_available_at' not in data.meta
    assert data.scale[0] > 0
    frame.loc[frame.index[-1], 'available_at'] += pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match='availability'):
        inference_dataset(frame, protocol, clock[0])


def test_repeat_same_candle_is_known_idempotent_retry(bundle, protocol, clock, tmp_path):
    class Client:
        def closed_bars(self, symbol, protocol):
            return bars(protocol, clock)
    path, digest = bundle
    first = observe_once(path, digest, tmp_path / 'ledger', protocol, client=Client())
    assert len(first['appended']) == 1 and first['summary']['predictions'] == 1
    artifact = tmp_path / 'ledger' / 'inputs' / (first['appended'][0]['decision_id'] + '.npz')
    with np.load(artifact, allow_pickle=False) as stored:
        expected = inference_dataset(bars(protocol, clock), protocol, clock[0])
        np.testing.assert_array_equal(stored['x'], expected.x[0])
        np.testing.assert_array_equal(stored['scale'], expected.scale[0])
    clock[0] += pd.Timedelta(minutes=2)
    second = observe_once(path, digest, tmp_path / 'ledger', protocol, client=Client())
    assert second['appended'] == [] and len(second['idempotent_skips']) == 1
    assert second['summary']['predictions'] == 1
    provenance = tmp_path / 'ledger' / 'provenance' / ('prediction-' + first['appended'][0]['decision_id'] + '.json')
    content = json.loads(provenance.read_text('utf-8'))
    content['input_hash'] = 'f' * 64
    provenance.write_text(json.dumps(content), 'utf-8')
    with pytest.raises(ValueError, match='exact known'):
        observe_once(path, digest, tmp_path / 'ledger', protocol, client=Client())


def test_crash_after_durable_prediction_commit_recovers_idempotently(bundle, protocol, clock, tmp_path, monkeypatch):
    class Client:
        def closed_bars(self, symbol, protocol):
            return bars(protocol, clock)
    original = ShadowLedger.append_prediction
    def commit_then_crash(self, *args):
        original(self, *args)
        raise RuntimeError('simulated process exit immediately after commit')
    monkeypatch.setattr(ShadowLedger, 'append_prediction', commit_then_crash)
    path, digest = bundle
    with pytest.raises(RuntimeError, match='process exit'):
        observe_once(path, digest, tmp_path / 'crash', protocol, client=Client())
    monkeypatch.setattr(ShadowLedger, 'append_prediction', original)
    clock[0] += pd.Timedelta(minutes=1)
    recovered = observe_once(path, digest, tmp_path / 'crash', protocol, client=Client())
    assert recovered['appended'] == [] and len(recovered['idempotent_skips']) == 1
    assert recovered['summary']['predictions'] == 1


def test_crash_after_provenance_before_commit_preserves_actual_first_receipt(bundle, protocol, clock, tmp_path, monkeypatch):
    class Client:
        def closed_bars(self, symbol, protocol):
            return bars(protocol, clock)
    original = ShadowLedger.append_prediction
    def crash_before_commit(self, *args):
        raise RuntimeError('simulated interruption before ledger commit')
    monkeypatch.setattr(ShadowLedger, 'append_prediction', crash_before_commit)
    path, digest = bundle
    first_receipt = clock[0]
    with pytest.raises(RuntimeError, match='interruption'):
        observe_once(path, digest, tmp_path / 'prepared', protocol, client=Client())
    monkeypatch.setattr(ShadowLedger, 'append_prediction', original)
    clock[0] += pd.Timedelta(minutes=1)
    recovered = observe_once(path, digest, tmp_path / 'prepared', protocol, client=Client())
    assert len(recovered['appended']) == 1 and recovered['idempotent_skips'] == []
    assert recovered['appended'][0]['available_at'] == first_receipt.isoformat()
    assert recovered['appended'][0]['recorded_at'] == clock[0].isoformat()


def test_exact_settlement_maturity(bundle, protocol, clock, tmp_path):
    class Client:
        outcome_calls = 0
        def closed_bars(self, symbol, protocol):
            close = (clock[0] - pd.Timedelta(seconds=protocol.availability_delay_seconds)).floor('h')
            source = rows(close=close.isoformat())
            return canonical_closed(source, symbol, clock[0], clock[0], protocol)
        def outcome(self, symbol, entry, end, delay):
            self.outcome_calls += 1
            assert clock[0] >= end + pd.Timedelta(seconds=delay)
            return .02, {'received_at': clock[0].isoformat()}
    client = Client()
    path, digest = bundle
    directory = tmp_path / 'settlement'
    observe_once(path, digest, directory, protocol, client=client)
    clock[0] = pd.Timestamp('2026-10-03T23:00:04.999999999Z')
    before = observe_once(path, digest, directory, protocol, client=client, settle=True)
    assert before['settled'] == [] and client.outcome_calls == 0
    clock[0] = pd.Timestamp('2026-10-03T23:00:05Z')
    at = observe_once(path, digest, directory, protocol, client=client, settle=True)
    assert len(at['settled']) == 1 and client.outcome_calls == 1
    assert at['settled'][0]['label_available_at'] == clock[0].isoformat()
    assert at['settled'][0]['recorded_at'] == clock[0].isoformat()
    assert at['summary']['mature'] == 1 and at['summary']['invalid'] == 0


def test_public_outcome_uses_exact_open_prices_only(protocol, clock, monkeypatch):
    client = forward.PublicBinance()
    entry = clock[0].floor('h') - 12 * pd.Timedelta(hours=1)
    end = entry + 12 * pd.Timedelta(hours=1)
    response = rows(close=(end + pd.Timedelta(hours=1)).isoformat(), count=13)
    monkeypatch.setattr(client, '_server', lambda: clock[0])
    monkeypatch.setattr(client, '_get', lambda *args: (response, clock[0]))
    value, provenance = client.outcome('BTCUSDT', entry, end, 5)
    assert value == pytest.approx(float(response[-1][1]) / float(response[0][1]) - 1)
    assert provenance['received_at'] == clock[0].isoformat()
    response.pop(6)
    with pytest.raises(ValueError, match='gapped'):
        client.outcome('BTCUSDT', entry, end, 5)


def test_interrupted_tensor_publication_leaves_no_final_and_retry_recovers(bundle, protocol, clock, tmp_path, monkeypatch):
    class Client:
        def closed_bars(self, symbol, protocol):
            return bars(protocol, clock)
    original = forward.np.savez
    def partial_write_then_crash(stream, **arrays):
        stream.write(b'partial archive')
        raise RuntimeError('simulated exit during tensor write')
    monkeypatch.setattr(forward.np, 'savez', partial_write_then_crash)
    path, digest = bundle
    directory = tmp_path / 'partial'
    with pytest.raises(RuntimeError, match='during tensor'):
        observe_once(path, digest, directory, protocol, client=Client())
    assert list((directory / 'inputs').glob('*.npz')) == []
    monkeypatch.setattr(forward.np, 'savez', original)
    recovered = observe_once(path, digest, directory, protocol, client=Client())
    assert len(recovered['appended']) == 1 and recovered['summary']['predictions'] == 1


def test_client_rejects_nonpublic_endpoints():
    with pytest.raises(ValueError, match='Only public'):
        forward.PublicBinance()._get('/fapi/v1/order')
