"""Four zero-fit clock/ledger checks; model and feature values are controlled stubs."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from factorlab.discovery import nominal_algorithm_producer, quarantined_observation, v7_forward
from factorlab.discovery.v7_lifecycle import ForecastExpired, Lifecycle, LifecyclePlan


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    origin = pd.Timestamp('2026-10-12T06:01:21.370500Z')
    clock = [origin + pd.Timedelta(seconds=1)]
    lifecycle = Lifecycle(tmp_path / 'run', LifecyclePlan('zero-fit-clock-fixture'), clock=lambda: clock[0])
    bars = pd.DataFrame([{'symbol': 'BTCUSDT', 'open_time': origin.floor('h') - pd.Timedelta(hours=1),
                          'available_at': origin, 'received_at': origin,
                          'availability_basis': 'actual_receipt', 'prediction_origin': True}])
    calls = []
    state = {'delay': pd.Timedelta(0)}

    def predict(history):
        calls.append(clock[0])
        clock[0] += state['delay']
        return np.array([.02]), np.array([True])

    model = {'model_id': 'controlled-model', 'pipeline': {'pipeline_id': 'controlled-pipeline'},
             'activation': origin.isoformat(), 'model': SimpleNamespace(predict_history=predict)}
    monkeypatch.setattr(lifecycle, 'active_model', lambda stream: model)
    runtime = object.__new__(v7_forward.ForwardRuntime)
    runtime.lifecycle = lifecycle
    runtime.config = SimpleNamespace(symbols=('BTCUSDT',))
    monkeypatch.setattr(runtime, '_snapshot', lambda now: bars.copy())
    monkeypatch.setattr(runtime, '_settle_available', lambda rows: 0)
    monkeypatch.setattr(runtime, '_serving_recipes', lambda *args: {'A0': None})
    monkeypatch.setattr(runtime, '_activate_ready', lambda *args, **kwargs: model)
    monkeypatch.setattr(v7_forward, 'build_feature_history', lambda rows: SimpleNamespace(
        metadata=pd.DataFrame({'symbol': rows.symbol, 'feature_open_time': rows.open_time})))
    yield SimpleNamespace(runtime=runtime, lifecycle=lifecycle, clock=clock, bars=bars,
                          calls=calls, state=state, model=model, origin=origin)
    lifecycle.close()


def test_verified_receipt_recovery_skips_duplicate_inference_and_rejects_missing_receipt(case):
    c = case
    assert c.runtime.tick()['new_predictions'] == 1
    receipt = next(c.lifecycle.forecasts.records_directory.glob('*.json'))
    original_bytes = receipt.read_bytes()
    original_prediction = json.loads(original_bytes)['receipt']['prediction']
    # Real crash boundary: receipt committed, SQL row not committed.
    with c.lifecycle.forecasts.db:
        c.lifecycle.forecasts.db.execute('DELETE FROM forecasts')
    c.clock[0] = c.origin + pd.Timedelta(seconds=293)
    c.state['delay'] = pd.Timedelta(seconds=30)
    result = c.runtime.tick()
    assert result['new_predictions'] == 0 and result['skipped'] == 1
    assert len(c.calls) == 1 and receipt.read_bytes() == original_bytes
    rows = list(c.lifecycle.forecasts.iter_records())
    assert len(rows) == 1 and json.loads(rows[0]['prediction']) == original_prediction
    # A missing durable receipt is corruption, never a duplicate shortcut.
    receipt.unlink()
    with pytest.raises(ValueError, match='receipt/index integrity'):
        c.runtime.tick()
    assert len(c.calls) == 1


def test_new_forecast_expiry_skips_without_backfill_and_exact_lag_boundary_is_valid(case):
    c = case
    c.clock[0] = c.origin + pd.Timedelta(seconds=293)
    c.state['delay'] = pd.Timedelta(seconds=30)
    result = c.runtime.tick()
    assert result['new_predictions'] == 0 and result['skipped'] == 1
    assert list(c.lifecycle.forecasts.iter_records()) == []
    assert list(c.lifecycle.forecasts.records_directory.glob('*.json')) == []
    # The next-hour boundary is independently closed, even while lag < 300s.
    c.origin = pd.Timestamp('2026-10-12T06:59:58Z')
    c.bars.loc[0, ['available_at', 'received_at']] = c.origin
    c.clock[0] = c.origin + pd.Timedelta(seconds=1)
    c.state['delay'] = pd.Timedelta(seconds=1)
    result = c.runtime.tick()
    assert result['new_predictions'] == 0 and result['skipped'] == 1
    assert list(c.lifecycle.forecasts.iter_records()) == []
    # Fresh next-hour evidence, with the unchanged inclusive 300s lag boundary.
    c.origin = pd.Timestamp('2026-10-12T07:01:00Z')
    c.bars.loc[0, ['available_at', 'received_at']] = c.origin
    c.bars.loc[0, 'open_time'] = c.origin.floor('h') - pd.Timedelta(hours=1)
    c.clock[0] = c.origin + pd.Timedelta(seconds=299)
    c.state['delay'] = pd.Timedelta(seconds=1)
    assert c.runtime.tick()['new_predictions'] == 1
    assert len(list(c.lifecycle.forecasts.iter_records())) == 1


def test_expiry_does_not_hide_invalid_receipt_window_or_model(case):
    c = case
    c.clock[0] = c.origin + pd.Timedelta(seconds=293)
    c.state['delay'] = pd.Timedelta(seconds=30)
    c.bars.loc[0, 'received_at'] = c.origin + pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match='timely real receipt') as error:
        c.runtime.tick()
    assert type(error.value) is ValueError
    row = {'stream': 'A0', 'origin': c.origin, 'symbol': 'BTCUSDT', 'task': 'volatility',
           'prediction': .02, 'received_at': c.origin, 'feature_open_time': c.bars.iloc[0].open_time,
           'forecast_window_start': c.origin.ceil('h'), 'model_id': c.model['model_id'],
           'pipeline_id': c.model['pipeline']['pipeline_id']}
    for change in ({'forecast_window_start': c.origin.ceil('h') + pd.Timedelta(hours=1)},
                   {'model_id': 'foreign-model'}):
        with pytest.raises(ValueError) as error:
            c.lifecycle.record_forecast(**(row | change))
        assert type(error.value) is ValueError
    with pytest.raises(ForecastExpired, match='timely real receipt'):
        c.lifecycle.record_forecast(**row)
    assert list(c.lifecycle.forecasts.iter_records()) == []


def test_nominal_capture_keeps_inference_and_expiry_is_not_success(case, monkeypatch):
    c = case
    assert c.runtime.tick()['new_predictions'] == 1
    receipt = next(c.lifecycle.forecasts.records_directory.glob('*.json'))
    original = receipt.read_bytes()
    ticket = object()
    completed = []
    monkeypatch.setattr(nominal_algorithm_producer, 'capture_tick', lambda *a, **kw: ticket)
    monkeypatch.setattr(nominal_algorithm_producer, 'skip_nonfinite_output', lambda *a: False)
    monkeypatch.setattr(nominal_algorithm_producer, 'finish_tick', completed.append)
    c.clock[0] = c.origin + pd.Timedelta(seconds=293)
    c.state['delay'] = pd.Timedelta(seconds=30)
    with pytest.raises(ForecastExpired):
        c.runtime.tick()
    assert len(c.calls) == 2 and completed == []
    assert receipt.read_bytes() == original
    # A quarantined context without a nominal ticket cannot borrow the live skip.
    monkeypatch.setattr(nominal_algorithm_producer, 'capture_tick', lambda *a, **kw: None)
    monkeypatch.setattr(quarantined_observation, 'current_binding', lambda lc: {'logical_at': c.origin.isoformat()})
    monkeypatch.setattr(c.lifecycle, 'now', lambda: c.clock[0])
    c.bars.loc[0, 'availability_basis'] = 'quarantined_finite_observation'
    c.clock[0] = c.origin + pd.Timedelta(seconds=293)
    with pytest.raises(ForecastExpired):
        c.runtime.tick()
    assert len(c.calls) == 3 and completed == []
    assert receipt.read_bytes() == original
