import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from factorlab.discovery.event_eval import (
    EventPredictor,
    build_event_library,
    event_nll,
    fit_event,
)
from factorlab.discovery.known_library import FactorFrame, build_known_library


def test_missing_classes_have_finite_unique_fit_and_full_probabilities():
    x = np.linspace(-1, 1, 60)[:, None]
    y = np.where(x[:, 0] > 0, 2, 0)
    model = fit_event(x, y, ridge=.01)
    probabilities = model.predict(x)
    assert probabilities.shape == (60, 7)
    assert np.isfinite(probabilities).all() and (probabilities > 0).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert event_nll(y, probabilities) < np.log(7)
    assert model.digest == fit_event(x, y, ridge=.01).digest
    assert model.schema['reference_class'] == 0
    assert 'intercepts' in model.schema['regularization']


def test_all_no_event_intercept_only_model_and_weighted_nll():
    model = fit_event(np.empty((20, 0)), np.zeros(20), ridge=.01)
    p = model.predict(np.empty((3, 0)))
    assert p.shape == (3, 7)
    assert np.all(p[:, 0] > p[:, 1:].max(axis=1))
    y, weights = [0, 3, 6], [1, 2, 3]
    expected = -np.average(np.log(p[np.arange(3), y]), weights=weights)
    assert event_nll(y, p, weights) == pytest.approx(expected)


def test_frame_scaler_and_policy_are_frozen(tmp_path):
    x = np.column_stack((np.arange(40), np.sin(np.arange(40))))
    frame = FactorFrame(x, pd.DataFrame({'id': range(40)}), ('ratio', 'hour'), (False, True))
    model = fit_event(frame, np.arange(40) % 7, provenance={'bins': [4, 12, 20]})
    assert model.scaler.center[1] == 0 and model.scaler.scale[1] == 1
    path = model.save(tmp_path / 'event.json')
    restored = EventPredictor.load(path)
    assert restored.digest == model.digest
    np.testing.assert_equal(restored.predict(frame), model.predict(frame))
    assert model.save(path) == path
    with pytest.raises(FileExistsError):
        fit_event(frame, np.zeros(40)).save(path)
    different = FactorFrame(x, frame.metadata, frame.feature_names, frame.bypass_mask, 'different')
    with pytest.raises(ValueError, match='policy'):
        restored.predict(different)
    raw = json.loads(path.read_text())
    raw['model']['parameters'][0][0] += 1
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='integrity'):
        EventPredictor.load(path)


def test_raw_event_library_ratios_calendar_and_invalid_warmup():
    time = pd.date_range('2024-01-01', periods=730, freq='h', tz='UTC')
    close = np.exp(np.arange(730) * .001)
    bars = pd.DataFrame({'symbol': 'BTCUSDT', 'open_time': time,
                         'available_at': time + pd.Timedelta(hours=1, seconds=1),
                         'close': close, 'high': close * 1.01, 'low': close / 1.01})
    base = build_known_library(bars)
    event = build_event_library(base)
    assert event.values.shape == (730, 6)
    assert not event.valid_rows[:720].any()
    np.testing.assert_allclose(event.values[720:, :2], np.tile([6/24, 24/168], (10, 1)))
    np.testing.assert_equal(event.values[:, 2:], base.values[:, 8:])
    assert event.schema_hash != base.schema_hash


def test_invalid_labels_and_solver_failure_are_not_silently_accepted():
    x = np.arange(10)[:, None]
    for y in ([0]*9 + [7], [0]*9 + [.5], [0]*9 + [np.nan]):
        with pytest.raises(ValueError, match='integer codes'):
            fit_event(x, y)
    with pytest.raises(ValueError, match='seven'):
        fit_event(x, np.zeros(10), classes=3)
    with pytest.raises(ValueError, match='converge'):
        fit_event(x, np.arange(10) % 7, max_iter=1)
    with pytest.raises(ValueError, match='normalized'):
        event_nll([0], np.ones((1, 7)))


def test_event_module_is_torch_free():
    command = ('import sys; import factorlab.discovery.event_eval; '
               'assert "torch" not in sys.modules')
    result = subprocess.run([sys.executable, '-B', '-c', command], capture_output=True,
                            text=True, check=False)
    assert result.returncode == 0, result.stderr
