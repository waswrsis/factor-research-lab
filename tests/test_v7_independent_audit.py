"""Independent small-input audit of integrated V7 causal/evidence contracts."""
import json

import numpy as np
import pandas as pd
import pytest

from factorlab.discovery.convex_eval import fit_ridge
from factorlab.discovery.event_eval import fit_event
from factorlab.discovery.exposure import ExposureLedger
from factorlab.discovery.known_library import FactorFrame
from factorlab.discovery.known_library import library_schema as k0_schema
from factorlab.discovery.model_registry import ForecastLedger, ModelRegistry
from factorlab.discovery.rolling import RollingSpec, build_rolling_data
from factorlab.protocol import fingerprint


def bars(n=2400):
    times = pd.date_range('2026-07-01', periods=n, freq='h', tz='UTC')
    increments = .0001 + .001 * np.sin(np.arange(n) / 3)
    close = 100 * np.exp(np.cumsum(increments))
    op = np.r_[close[0], close[:-1]]
    return pd.DataFrame({'symbol': 'BTCUSDT', 'open_time': times,
                         'available_at': times + pd.Timedelta(hours=1, seconds=1),
                         'open': op, 'close': close,
                         'high': np.maximum(op, close) * np.exp(.0002),
                         'low': np.minimum(op, close) * np.exp(-.0002)})


def observed_bars(n=1100):
    frame = bars(n)
    frame['received_at'] = frame.available_at
    frame['prediction_origin'] = True
    frame.attrs['availability_mode'] = 'observed_rest_feed'
    return frame


def test_independent_future_mutation_cannot_change_training_identity_or_model():
    source = bars()
    spec = RollingSpec(minimum_train_rows=2)
    cutoff = pd.Timestamp('2026-10-01T00:00Z')
    before = build_rolling_data(source, spec)
    future = source.open_time >= cutoff
    source.loc[future, ['open', 'high', 'low', 'close']] *= 1.7
    after = build_rolling_data(source, spec)
    mask = before.train_mask(cutoff)
    assert mask.sum() > 1000
    np.testing.assert_equal(mask, after.train_mask(cutoff))
    np.testing.assert_equal(before.frame.values[mask], after.frame.values[mask])
    np.testing.assert_equal(before.target[mask], after.target[mask])
    assert before.training_identity(cutoff) == after.training_identity(cutoff)
    left = fit_ridge(before.frame.take(mask), before.target[mask])
    right = fit_ridge(after.frame.take(mask), after.target[mask])
    assert left.digest == right.digest


def test_independent_late_preforecast_close_delays_label_maturity():
    source = observed_bars()
    origin = 1000
    # Forecast starts at open[1002]. Its first return uses close[1001], which
    # arrives after all twelve forecast candles. It is not a model input.
    late = source.available_at[1025] + pd.Timedelta(seconds=2)
    source.loc[origin + 1, ['available_at', 'received_at']] = late
    source.loc[origin + 1, 'prediction_origin'] = False
    result = build_rolling_data(source, RollingSpec(minimum_train_rows=2))
    row = result.frame.metadata.index[
        result.frame.metadata.feature_open_time == source.open_time[origin]][0]
    assert result.frame.valid_rows[row] and result.observed[row]
    assert result.maturity[row] == late
    start = result.frame.metadata.available_at[row]
    end = start + pd.Timedelta(seconds=1)
    assert not result.score_mask(start, end, late - pd.Timedelta(seconds=1))[row]
    assert result.score_mask(start, end, late)[row]


def test_independent_unsorted_rows_and_datetime_units_keep_labels():
    source = bars(800)
    expected = build_rolling_data(source)
    source = source.sample(frac=1, random_state=71).reset_index(drop=True)
    source['open_time'] = source.open_time.dt.as_unit('ms')
    source['available_at'] = source.available_at.dt.as_unit('us')
    actual = build_rolling_data(source)
    np.testing.assert_equal(actual.observed, expected.observed)
    np.testing.assert_equal(actual.target, expected.target)
    np.testing.assert_equal(actual.frame.values, expected.frame.values)


def test_independent_future_gap_prevents_label_and_resets_feature_support():
    source = bars(1600)
    gap = 780
    result = build_rolling_data(source.drop(index=gap))
    index = result.frame.metadata.set_index('feature_open_time').index
    before_gap = index.get_loc(source.open_time[770])
    assert result.frame.valid_rows[before_gap]
    assert not result.observed[before_gap]
    after_gap = index.get_loc(source.open_time[800])
    assert not result.frame.valid_rows[after_gap]
    resumed = index.get_loc(source.open_time[1501])
    assert result.frame.valid_rows[resumed]


@pytest.mark.parametrize('column,value', [('open', np.nan), ('high', 0), ('low', -1)])
def test_independent_invalid_future_event_candle_never_becomes_no_event(column, value):
    source = bars(800)
    source.loc[750, column] = value
    with pytest.raises(ValueError, match='OHLC'):
        build_rolling_data(source, RollingSpec(task='first_event', horizon_hours=20))


def test_independent_double_touch_is_unobserved_but_open_hit_has_known_direction():
    source = bars(800)
    origin = 740
    first = origin + 2
    # Put the first future candle open inside both barriers, with both extremes
    # outside. Later candles cannot resolve first intrabar direction.
    source.loc[first, 'open'] = source.close[origin]
    source.loc[first, 'high'] = source.close[origin] * 1.2
    source.loc[first, 'low'] = source.close[origin] * .8
    ambiguous = build_rolling_data(source, RollingSpec(task='first_event', horizon_hours=20))
    assert ambiguous.frame.metadata.label_status[origin] == 'ambiguous_double_touch'
    assert not ambiguous.observed[origin] and np.isnan(ambiguous.target[origin])
    source.loc[first, 'open'] = source.close[origin] * 1.1
    known = build_rolling_data(source, RollingSpec(task='first_event', horizon_hours=20))
    assert known.observed[origin] and known.target[origin] == 1
    assert known.maturity[origin] == known.frame.metadata.forecast_window_start[origin] + pd.Timedelta(hours=20, seconds=1)


def test_independent_registry_roundtrip_both_kinds_and_pipeline_binding(tmp_path):
    frame = FactorFrame(np.arange(30, dtype=float).reshape(10, 3),
                        pd.DataFrame({'id': range(10)}), ('a', 'b', 'c'), (False,) * 3)
    models = [fit_ridge(frame, np.linspace(.01, .02, 10)),
              fit_event(frame, np.arange(10) % 7)]
    registry = ModelRegistry(tmp_path / 'models')
    for model in models:
        pipeline = {'role': 'test', 'task': model.task_support[0]}
        identity = registry.save(model, pipeline)
        assert identity == registry.save(model, pipeline)
        loaded, manifest = registry.load(identity, expected_pipeline=pipeline)
        assert loaded.digest == model.digest and manifest['schema'] == loaded.schema
        np.testing.assert_equal(loaded.predict(frame), model.predict(frame))
        with pytest.raises(ValueError, match='Pipeline'):
            registry.load(identity, expected_pipeline={'role': 'different'})
    folder = registry.directory / identity
    raw = json.loads((folder / 'manifest.json').read_text())
    raw['identity']['task_support'] = ['volatility']
    (folder / 'manifest.json').write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='identity'):
        registry.load(identity)


def test_independent_forecast_recovery_cannot_rebind_same_logical_origin(tmp_path):
    path = tmp_path / 'forecasts.sqlite'
    arguments = {'run': 'r', 'stream': 'candidate', 'origin': '2026-10-01T00:00Z',
                 'symbol': 'BTCUSDT', 'task': 'volatility', 'model_id': 'old',
                 'pipeline_id': 'p', 'prediction': .1}
    ledger = ForecastLedger(path)
    assert ledger.put(**arguments)
    ledger.close()
    ledger = ForecastLedger(path)
    assert ledger.put(**arguments) is False
    with pytest.raises(ValueError, match='rebound'):
        ledger.put(**{**arguments, 'model_id': 'new'})
    assert ledger.put(**{**arguments, 'stream': 'reference', 'model_id': 'reference'})
    assert len(list(ledger.iter_records())) == 2
    ledger.close()


def test_independent_exposure_prompt_needs_full_coverage_each_symbol(tmp_path):
    now = [pd.Timestamp('2027-02-01T00:00Z')]
    ledger = ExposureLedger(tmp_path / 'exposure.sqlite', clock=lambda: now[0])
    common = {'source': 'market', 'batch': 'new-research'}
    ledger.record(**common, symbols=['BTC'], start='2027-01-01T00:00Z',
                  end='2027-01-15T00:00Z', role='S')
    with pytest.raises(ValueError):
        ledger.assert_prompt_search(**common, symbols=['BTC'], start='2027-01-01T00:00Z',
                                    end='2027-02-01T00:00Z')
    ledger.record(**common, symbols=['BTC'], start='2027-01-15T00:00Z',
                  end='2027-02-01T00:00Z', role='S')
    assert ledger.assert_prompt_search(**common, symbols=['BTC'], start='2027-01-01T00:00Z',
                                       end='2027-02-01T00:00Z')
    with pytest.raises(ValueError):
        ledger.assert_prompt_search(**common, symbols=['BTC', 'ETH'], start='2027-01-01T00:00Z',
                                    end='2027-02-01T00:00Z')
    ledger.close()


def test_independent_frozen_overlapping_confirmations_survive_later_other_batch_reuse(tmp_path):
    now = [pd.Timestamp('2027-01-01T00:00Z')]
    ledger = ExposureLedger(tmp_path / 'overlap.sqlite', clock=lambda: now[0])
    base = {'source': 'market', 'symbols': ['BTC'], 'reference_id': 'reference',
            'rule': {'horizon_hours': 12}, 'cohort': 1}
    ledger.reserve_confirmation(**base, batch='A', candidate_id='candidate_A',
                                start='2027-02-01T00:00Z', end='2027-04-01T00:00Z',
                                frozen_at=now[0], claim_slot='factor1')
    now[0] = pd.Timestamp('2027-02-01T00:00Z')
    ledger.reserve_confirmation(**base, batch='B', candidate_id='candidate_B',
                                start='2027-03-01T00:00Z', end='2027-05-01T00:00Z',
                                frozen_at=now[0], claim_slot='factor2')
    now[0] = pd.Timestamp('2027-04-03T00:00Z')
    result_a = ledger.consume_confirmation('A', as_of=now[0])
    assert pd.Timestamp(result_a['scoring_cutoff']) == pd.Timestamp('2027-04-02T12:00Z')
    # Already observed March may become a later batch's explicit S input.
    ledger.record(source='market', symbols=['BTC'], batch='later', role='S',
                  start='2027-03-01T00:00Z', end='2027-04-01T00:00Z')
    assert ledger.assert_prompt_search(source='market', symbols=['BTC'], batch='later',
                                       start='2027-03-01T00:00Z', end='2027-04-01T00:00Z')
    with pytest.raises(ValueError):
        ledger.assert_prompt_search(source='market', symbols=['BTC'], batch='B',
                                    start='2027-03-01T00:00Z', end='2027-04-01T00:00Z')
    now[0] = pd.Timestamp('2027-05-05T00:00Z')
    result_b = ledger.consume_confirmation('B', as_of=now[0])
    assert pd.Timestamp(result_b['scoring_cutoff']) == pd.Timestamp('2027-05-02T12:00Z')
    with pytest.raises(ValueError):
        ledger.consume_confirmation('B', as_of=now[0])
    ledger.close()


def test_independent_same_batch_feedback_cannot_keep_clean_confirmation(tmp_path):
    now = [pd.Timestamp('2027-01-01T00:00Z')]
    ledger = ExposureLedger(tmp_path / 'contamination.sqlite', clock=lambda: now[0])
    reserve = {'batch': 'A', 'source': 'market', 'symbols': ['BTC'],
               'start': '2027-02-01T00:00Z', 'end': '2027-03-01T00:00Z',
               'candidate_id': 'c', 'reference_id': 'r', 'rule': {'horizon_hours': 12},
               'frozen_at': now[0], 'cohort': 1, 'claim_slot': 'factor1'}
    identity = ledger.reserve_confirmation(**reserve)
    assert identity == ledger.reserve_confirmation(**reserve)
    with pytest.raises(ValueError):
        ledger.reserve_confirmation(**{**reserve, 'batch': 'slot-renaming'})
    now[0] = pd.Timestamp('2027-02-15T00:00Z')
    ledger.record(batch='A', source='market', symbols=['BTC'], role='S',
                  start='2027-02-01T00:00Z', end='2027-02-14T00:00Z')
    now[0] = pd.Timestamp('2027-03-03T00:00Z')
    with pytest.raises(ValueError):
        ledger.consume_confirmation('A', as_of=now[0])
    ledger.close()


def test_independent_zero_reference_loss_has_no_percentage_gain():
    from factorlab.discovery.factor_pool import evaluate_formula
    from factorlab.discovery.rolling import RollingData

    time = pd.date_range('2026-01-01', '2026-07-01', freq='h', tz='UTC', inclusive='left')
    meta = pd.DataFrame({'symbol': 'BTC', 'available_at': time + pd.Timedelta(seconds=1),
                         'feature_open_time': time - pd.Timedelta(hours=1),
                         'forecast_window_start': time + pd.Timedelta(hours=1)})
    schema = k0_schema()
    frame = FactorFrame(np.zeros((len(time), 12)), meta, tuple(schema['feature_names']),
                        tuple(schema['bypass_mask']), fingerprint(schema))
    # log(y+epsilon)=0 exactly, giving an exact constant zero-loss reference.
    data = RollingData(frame, np.full(len(time), 1 - 1e-8), time + pd.Timedelta(hours=13),
                       np.ones(len(time), dtype=bool), RollingSpec(minimum_train_rows=2),
                       'assumed_historical_delay')
    formula = {'version': 'causal-operator-program-v1', 'feature_names': schema['feature_names'],
               'parameters': {}, 'outputs': [{'op': 'last', 'arg': {'op': 'feature', 'name': 'rv_6'}}]}
    result = evaluate_formula(data, '2026-07-01T00:00Z', formula, lookback=1)
    assert result['channels']['F']['gain'] is None
    assert result['channels']['S']['gain'] is None


def test_independent_confirmation_does_not_silently_overweight_duplicate_hours():
    from factorlab.discovery.confirmatory import ConfirmationRule, paired_effect_interval

    time = pd.date_range('2026-01-01', periods=12, freq='h', tz='UTC')
    origins = time.repeat([10] + [1] * 11)
    base = np.ones(len(origins))
    candidate = np.r_[np.zeros(10), np.full(11, 2.)]
    rule = ConfirmationRule(block_hours=1, bootstrap_samples=1000)
    try:
        result = paired_effect_interval(origins, base, candidate, rule=rule)
    except ValueError as error:
        # A single-symbol API may explicitly require preaggregated unique hours.
        assert 'hour' in str(error).lower() or 'duplicate' in str(error).lower()
    else:
        expected = np.mean(np.r_[.995, np.full(11, -1.005)])
        assert result['d_mean'] == pytest.approx(expected)
