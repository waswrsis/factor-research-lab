import numpy as np
import pandas as pd
import pytest

from factorlab.discovery.convex_eval import fit_ridge
from factorlab.discovery.exposure import ExposureLedger, select_on_validation
from factorlab.discovery.model_registry import ForecastLedger, ModelRegistry
from factorlab.discovery.rolling import RollingSpec, build_rolling_data, development_folds


def candles(start='2024-12-01', end='2025-05-02'):
    times = pd.date_range(start, end, freq='h', tz='UTC', inclusive='left')
    price = np.exp(4 + np.cumsum(np.sin(np.arange(len(times)) / 5) * .002 + .0001))
    frame = pd.DataFrame({'symbol': 'BTCUSDT', 'open_time': times,
                          'available_at': times + pd.Timedelta(hours=1, seconds=1),
                          'open': price, 'close': price,
                          'high': price * 1.003, 'low': price / 1.003})
    return frame


def test_three_month_maturity_cold_identity_and_future_mutation():
    bars = candles()
    data = build_rolling_data(bars)
    cutoff = pd.Timestamp('2025-04-01', tz='UTC')
    mask = data.train_mask(cutoff)
    meta = data.frame.metadata.loc[mask]
    assert meta.available_at.min() >= pd.Timestamp('2025-01-01', tz='UTC')
    assert meta.available_at.max() < cutoff
    assert (data.maturity[mask] <= cutoff).all()
    assert mask.sum() > 2000
    assert not mask[-1]
    future = bars.open_time >= cutoff
    bars.loc[future, ['open', 'high', 'low', 'close']] *= 7
    changed = build_rolling_data(bars)
    assert data.training_identity(cutoff) == changed.training_identity(cutoff)
    model = fit_ridge(data.frame.take(mask), data.target[mask])
    assert model.digest == fit_ridge(data.frame.take(mask), data.target[mask]).digest
    assert len(development_folds('2025-04-01T00:00:00Z')) == 3


def test_gap_warmup_and_bad_source():
    bars = candles().drop(index=[1000]).reset_index(drop=True)
    data = build_rolling_data(bars)
    index = data.frame.metadata.feature_open_time.searchsorted(bars.open_time.iloc[1000])
    assert not data.frame.valid_rows[index:index + 720].any()
    assert data.frame.valid_rows[index + 720]
    bad = candles()
    bad.loc[20, 'close'] = np.nan
    with pytest.raises(ValueError):
        build_rolling_data(bad)
    unordered = candles().sample(frac=1, random_state=1)
    ordered = build_rolling_data(candles())
    repaired = build_rolling_data(unordered)
    np.testing.assert_equal(ordered.frame.values, repaired.frame.values)


def test_observed_receipt_cannot_be_backdated():
    bars = candles()
    bars.attrs['availability_mode'] = 'observed_rest_feed'
    bars['received_at'] = bars.available_at
    bars['prediction_origin'] = True
    bars.loc[20, 'received_at'] = bars.open_time.iloc[20]
    with pytest.raises(ValueError, match='receipt'):
        build_rolling_data(bars)


def test_event_full_horizon_and_invalid_ohlc():
    data = build_rolling_data(candles(), RollingSpec(task='first_event', horizon_hours=20))
    assert np.isin(data.target[data.observed], np.arange(7)).all()
    assert (~data.observed[-20:]).all()
    bad = candles()
    bad.loc[100, 'open'] = 0
    with pytest.raises(ValueError, match='OHLC'):
        build_rolling_data(bad, RollingSpec(task='first_event', horizon_hours=20))


def test_registry_restore_tamper_and_logical_forecast_binding(tmp_path, monkeypatch):
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    data = build_rolling_data(candles())
    mask = data.train_mask('2025-04-01T00:00:00Z')
    model = fit_ridge(data.frame.take(mask), data.target[mask])
    registry = ModelRegistry(tmp_path / 'models')
    pipeline = data.training_identity('2025-04-01T00:00:00Z')
    identifier = registry.save(model, pipeline)
    restored, _ = registry.load(identifier, expected_pipeline=pipeline)
    np.testing.assert_array_equal(model.predict(data.frame.take(mask)), restored.predict(data.frame.take(mask)))
    with pytest.raises(ValueError, match='Pipeline'):
        registry.load(identifier, expected_pipeline={})
    ledger = ForecastLedger(tmp_path / 'forecast.sqlite')
    row = {'run': 'run', 'stream': 'A0', 'origin': '2025-04-01T01:00:01Z', 'symbol': 'BTC',
           'task': 'volatility', 'model_id': identifier, 'pipeline_id': 'K0', 'prediction': 0.1}
    assert ledger.put(**row)
    assert not ledger.put(**row)
    with pytest.raises(ValueError, match='rebound'):
        ledger.put(**{**row, 'model_id': 'changed'})
    ledger.close()
    artifact = tmp_path / 'models' / identifier / 'model.json'
    artifact.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='integrity'):
        registry.load(identifier)


def test_confirmation_single_look_future_and_renaming(tmp_path, monkeypatch):
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    clock = [pd.Timestamp('2025-04-01T00:00:00Z')]
    ledger = ExposureLedger(tmp_path / 'exposure.sqlite', clock=lambda: clock[0])
    ticket = {'batch': 'b1', 'source': 'exchange', 'symbols': ['BTC'], 'start': '2025-05-01T00:00:00Z',
              'end': '2025-08-01T00:00:00Z', 'candidate_id': 'candidate', 'reference_id': 'K0',
              'rule': {'horizon_hours': 12}, 'frozen_at': clock[0], 'cohort': 1, 'claim_slot': 'factor1'}
    ledger.reserve_confirmation(**ticket)
    clock[0] = pd.Timestamp('2025-08-04T00:00:00Z')
    with pytest.raises(ValueError, match='matured'):
        ledger.consume_confirmation('b1', as_of='2025-08-01T01:00:00Z')
    ledger.consume_confirmation('b1', as_of='2025-08-03T00:00:00Z')
    with pytest.raises(ValueError, match='consumed'):
        ledger.consume_confirmation('b1', as_of='2025-08-04T00:00:00Z')
    ledger.finish_confirmation('b1', {'supported': False})
    with pytest.raises(ValueError, match='slot'):
        ledger.reserve_confirmation(**{**ticket, 'batch': 'b2', 'candidate_id': 'renamed'})
    with pytest.raises(ValueError, match='search'):
        ledger.assert_prompt_search(source='exchange', symbols=['BTC'], start=ticket['start'],
                                    end=ticket['end'], batch='b2')
    ledger.close()


def test_validation_is_budgeted_common_support_point_gate():
    row = {'role': 'V', 'common_support': True, 'coverage': 1., 'gain': .006,
           'complexity': 3, 'candidate_id': 'a'}
    assert select_on_validation([row]) == row
    assert select_on_validation([{**row, 'gain': .005}]) is None
    with pytest.raises(ValueError, match='budget'):
        select_on_validation([row] * 4)
    with pytest.raises(ValueError, match='support'):
        select_on_validation([{**row, 'common_support': False}])
