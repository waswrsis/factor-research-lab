"""Independent forward-adapter revision, evidence and deadline counterexamples."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from factorlab import forward, shadow
from factorlab.features import FEATURE_COLUMNS
from factorlab.protocol import Protocol
from factorlab.shadow import ShadowLedger, freeze_bundle
from factorlab.training import FittedModel


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    clock = [pd.Timestamp("2026-10-03T10:30:00Z")]
    monkeypatch.setattr(forward, "wall_now", lambda: clock[0])
    monkeypatch.setattr(shadow, "_wall_now", lambda: clock[0])
    protocol = replace(Protocol(), symbols=("BTCUSDT",))
    times = pd.date_range(end="2026-10-03T09:00:00Z", periods=200, freq="h")
    response = []
    for i, stamp in enumerate(times):
        price = 100 + 0.2 * i + np.sin(i) * 0.1
        response.append(
            [
                stamp.value // 1_000_000,
                price,
                price + 1,
                price - 1,
                price + 0.1,
                100 + i % 17,
                (stamp + pd.Timedelta(hours=1)).value // 1_000_000 - 1,
                0,
                10,
                0,
                0,
                0,
            ]
        )
    width = len(FEATURE_COLUMNS)
    source = tmp_path / "source"
    FittedModel(
        None,
        FEATURE_COLUMNS,
        np.zeros(width, np.float32),
        np.ones(width, np.float32),
        np.zeros(width, np.float32),
        0.0,
        None,
        0.0,
        {"prediction_clip": protocol.prediction_clip},
    ).save(source)
    destination = tmp_path / "bundle"
    manifest = freeze_bundle(source, destination, {"protocol_hash": protocol.digest})

    class Client:
        def closed_bars(self, symbol, protocol):
            return forward.canonical_closed(response, symbol, clock[0], clock[0], protocol)

    return clock, protocol, response, destination, manifest["bundle_hash"], Client()


def test_revised_closed_history_is_rejected_even_when_forecast_is_identical(scenario, tmp_path):
    clock, protocol, response, bundle, digest, client = scenario
    ledger = tmp_path / "revision"
    first = forward.observe_once(bundle, digest, ledger, protocol, client=client)
    assert first["appended"][0]["prediction"] == 0.0
    clock[0] += pd.Timedelta(minutes=1)
    response[-1][4] += 0.2  # Still inside valid OHLC bounds; same closed candle ID.
    with pytest.raises(ValueError, match="tensor differs|exact known"):
        forward.observe_once(bundle, digest, ledger, protocol, client=client)
    store = ShadowLedger(ledger, protocol.digest, digest, observation_mode="forward")
    assert store.summary(clock[0])["predictions"] == 1


def test_changed_durable_tensor_cannot_become_a_known_retry(scenario, tmp_path):
    clock, protocol, _, bundle, digest, client = scenario
    ledger = tmp_path / "tensor"
    first = forward.observe_once(bundle, digest, ledger, protocol, client=client)
    decision = first["appended"][0]["decision_id"]
    artifact = ledger / "inputs" / (decision + ".npz")
    with np.load(artifact, allow_pickle=False) as stored:
        x, scale = stored["x"].copy(), stored["scale"].copy()
    x[0, 0] += 0.5
    np.savez(artifact, x=x, scale=scale)
    clock[0] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="tensor differs|exact known"):
        forward.observe_once(bundle, digest, ledger, protocol, client=client)


def test_prepared_forecast_cannot_be_retimed_after_its_entry(scenario, tmp_path, monkeypatch):
    clock, protocol, _, bundle, digest, client = scenario
    ledger = tmp_path / "deadline"
    original = ShadowLedger.append_prediction

    def interrupt_before_commit(self, *args):
        raise RuntimeError("interrupted before commit")

    monkeypatch.setattr(ShadowLedger, "append_prediction", interrupt_before_commit)
    with pytest.raises(RuntimeError):
        forward.observe_once(bundle, digest, ledger, protocol, client=client)
    original_input = client.closed_bars("BTCUSDT", protocol)
    monkeypatch.setattr(ShadowLedger, "append_prediction", original)
    # Supply identical original evidence to isolate retry's entry-deadline check.
    monkeypatch.setattr(client, "closed_bars", lambda *args: original_input.copy())
    clock[0] = pd.Timestamp("2026-10-03T11:00:01Z")
    with pytest.raises(ValueError, match="precede"):
        forward.observe_once(bundle, digest, ledger, protocol, client=client)
    store = ShadowLedger(ledger, protocol.digest, digest, observation_mode="forward")
    assert store.summary(clock[0])["predictions"] == 0
