import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from factorlab.discovery.convex_eval import (
    ProfileFitError,
    ResidualProjector,
    RidgePredictor,
    TrainScaler,
    fit_ridge,
    log_mse,
    profile_parameters,
)
from factorlab.discovery.known_library import FactorFrame


def test_weighted_ridge_matches_independent_normal_equations_and_cold_fit():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(70, 3))
    y = np.exp(.3 + X @ np.array([.2, -.4, .1])) - 1e-8
    weights = np.arange(1, 71, dtype=float)
    model = fit_ridge(X, y, ridge=.02, sample_weight=weights)
    design = np.column_stack((np.ones(len(X)), model.scaler.transform(X)))
    normalized = weights / weights.sum()
    penalty = np.diag([0, .02, .02, .02])
    expected = np.linalg.solve(design.T @ (normalized[:, None] * design) + penalty,
                               design.T @ (normalized * np.log(y + 1e-8)))
    np.testing.assert_allclose(np.r_[model.intercept, model.coefficients], expected, atol=1e-12)
    second = fit_ridge(X, y, ridge=.02, sample_weight=weights)
    assert model.digest == second.digest
    assert log_mse(y, model.predict(X), weights) < .002


def test_train_only_scaling_and_calendar_bypass():
    X = np.column_stack((np.arange(10), np.sin(np.arange(10))))
    frame = FactorFrame(X, pd.DataFrame({'id': range(10)}), ('energy', 'calendar'), (False, True))
    model = fit_ridge(frame, np.linspace(.1, .2, 10))
    np.testing.assert_allclose(model.scaler.center, [4.5, 0])
    assert model.scaler.scale[1] == 1
    before = model.digest
    model.predict(np.array([[100., .5]]))
    assert model.digest == before
    wrong = FactorFrame(X[:, ::-1], frame.metadata, ('calendar', 'energy'), (True, False))
    with pytest.raises(ValueError, match='names/order'):
        model.predict(wrong)


def test_boundary_decode_and_empty_feature_constant_model():
    model = fit_ridge(np.empty((10, 0)), np.ones(10))
    np.testing.assert_allclose(model.predict(np.empty((2, 0))), 1)
    low = RidgePredictor(TrainScaler(np.array([0.]), np.array([1.]), (False,)),
                         np.array([0.]), -30., .1, ('x',))
    prediction, diagnostic = low.predict_with_diagnostics(np.zeros((2, 1)))
    np.testing.assert_equal(prediction, 0)
    assert diagnostic['clipped_fraction'] == 1
    high = RidgePredictor(low.scaler, low.coefficients, 1000., .1, ('x',))
    with pytest.raises(ValueError, match='decoded'):
        high.predict(np.zeros((1, 1)))


def test_save_load_digest_and_tamper_rejection(tmp_path):
    X = np.arange(12, dtype=float).reshape(6, 2)
    model = fit_ridge(X, np.arange(6) / 10, provenance={'training_end': '2026-10-01Z'})
    path = model.save(tmp_path / 'model.json')
    loaded = RidgePredictor.load(path)
    assert loaded.digest == model.digest
    np.testing.assert_equal(loaded.predict(X), model.predict(X))
    model.save(path)
    with pytest.raises(FileExistsError):
        fit_ridge(X, np.ones(6)).save(path)
    raw = json.loads(path.read_text())
    raw['model']['intercept'] += .1
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='integrity'):
        RidgePredictor.load(path)


def test_weighted_projection_duplicate_and_conditional_residual():
    rng = np.random.default_rng(42)
    B = rng.normal(size=(50, 2))
    weight = np.linspace(.2, 2, 50)
    duplicate = 3 + B @ np.array([2., -5.])
    projector = ResidualProjector.fit(B, duplicate, sample_weight=weight)
    assert projector.near_zero
    assert np.max(np.abs(projector.transform(B, duplicate, normalize=True))) < 1e-12
    z = duplicate + rng.normal(scale=.1, size=50)
    projector = ResidualProjector.fit(B, z, sample_weight=weight)
    residual = projector.transform(B, z)
    np.testing.assert_allclose(np.column_stack((np.ones(50), B)).T @ (weight * residual),
                               0, atol=1e-11)
    assert not projector.near_zero
    saved = ResidualProjector.from_dict(projector.to_dict())
    np.testing.assert_equal(saved.transform(B[:3] + 10, z[:3]),
                            projector.transform(B[:3] + 10, z[:3]))


def test_near_zero_residual_is_not_amplified():
    B = np.arange(20, dtype=float)[:, None]
    z = B[:, 0] + 1e-10 * np.sin(B[:, 0])
    projector = ResidualProjector.fit(B, z)
    assert projector.near_zero
    assert np.max(np.abs(projector.transform(B, z, normalize=True))) < 1e-8
    frame = FactorFrame(B, pd.DataFrame({'id': range(20)}), ('base',), (False,))
    augmented = projector.augment(frame, z)
    model = fit_ridge(augmented, np.exp(.01 * B[:, 0]))
    assert model.scaler.scale[-1] == 1
    assert model.scaler.center[-1] == 0
    assert abs(model.coefficients[-1]) < 1e-8
    baseline = fit_ridge(frame, np.exp(.01 * B[:, 0]))
    np.testing.assert_allclose(model.predict(augmented), baseline.predict(frame), atol=1e-12)


def test_profile_counts_every_solve_failure_and_respects_bounds():
    x = np.linspace(0, 2, 30)
    seen = []

    def factory(theta):
        seen.append(theta.copy())
        if theta[0] == .5:
            raise ValueError('synthetic rejected candidate')
        return np.sin(theta[0] * x)[:, None]

    result = profile_parameters(factory, np.exp(.2*np.sin(x)), bounds=[(.5, 2)],
                                initializations=[[.5], [1.], [1.8]], max_calls=9)
    assert result.calls == len(seen) == len(result.history) == 9
    assert result.history[0]['status'] == 'failed'
    assert result.status == 'budget_exhausted'
    assert all(.5 <= theta[0] <= 2 for theta in seen)
    assert np.isfinite(result.predictor.predict(factory(result.parameters))).all()


def test_invalid_training_support_and_failed_profile_evidence():
    with pytest.raises(ValueError, match='finite'):
        fit_ridge(np.array([[np.nan]]), [.2])
    with pytest.raises(ValueError, match='nonnegative'):
        fit_ridge(np.zeros((2, 1)), [1., -1.])
    with pytest.raises(ValueError, match='positive'):
        fit_ridge(np.zeros((2, 1)), [1., 1.], sample_weight=[0, 0])
    with pytest.raises(ProfileFitError) as caught:
        profile_parameters(lambda _: np.array([[np.nan]]), [.1], bounds=[(0., 1.)],
                           initializations=[[.2]], max_calls=3)
    assert caught.value.calls == 3
    assert all(r['status'] == 'failed' for r in caught.value.history)


def test_modules_do_not_load_torch():
    check = subprocess.run([sys.executable, '-B', '-c',
        ('import sys; import factorlab.discovery.convex_eval; '
         'assert "torch" not in sys.modules')], capture_output=True, text=True, check=False)
    assert check.returncode == 0, check.stderr
