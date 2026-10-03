from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from factorlab.dataset import Dataset
from factorlab.graph import ExpertSpec, GraphSpec
from factorlab.protocol import Protocol
from factorlab.research_agent import diagnostic_proposal
from factorlab.training import FittedModel, fit_model


@pytest.fixture
def artificial_data():
    rng = np.random.default_rng(31)
    x = rng.normal(size=(480, 48, 2)).astype(np.float32)
    scale = np.linspace(0.002, 0.006, len(x), dtype=np.float32)
    y = (0.3 * x[:, -1, 0] + 0.1 * x[:, -1, 1] ** 2) * scale
    times = pd.date_range("2025-01-01", periods=len(x), freq="h", tz="UTC")
    meta = pd.DataFrame(
        {"available_at": times, "label_available_at": times + pd.Timedelta(hours=13)}
    )
    return Dataset(x, y, scale, meta, ("ret", "vol"))


def spec():
    return GraphSpec(
        ("ret", "vol"),
        (ExpertSpec("time", width=4, scales=(4, 12)), ExpertSpec("interaction", "mlp", width=4)),
        gating="dynamic",
    )


def test_train_only_scaling_and_chronological_calibration(artificial_data):
    data = artificial_data
    train = np.arange(len(data.y)) < 420
    protocol = Protocol(epochs=1, batch_size=128)
    fitted = fit_model(data, train, protocol, spec(), seed=7)
    split = data.meta.available_at[336]
    fit_indices = np.flatnonzero(train & (data.meta.label_available_at < split).to_numpy())
    assert fitted.diagnostics["fit_samples"] == len(fit_indices)
    assert fitted.diagnostics["calibration_samples"] == 84
    assert pd.Timestamp(fitted.diagnostics["fit_label_cutoff"]) < split
    np.testing.assert_allclose(fitted.mean, data.x[fit_indices].mean(axis=(0, 1)), atol=1e-7)
    np.testing.assert_allclose(fitted.std, data.x[fit_indices].std(axis=(0, 1)), rtol=1e-5)
    # Change every row excluded from parameter fitting, including calibration
    # features and labels. Only held-out alpha is allowed to depend on them.
    changed = Dataset(data.x.copy(), data.y.copy(), data.scale.copy(), data.meta, data.features)
    excluded = np.ones(len(data.y), dtype=bool)
    excluded[fit_indices] = False
    changed.x[excluded] = 100
    changed.y[excluded] = -0.03
    second = fit_model(changed, train, protocol, spec(), seed=7)
    np.testing.assert_array_equal(fitted.mean, second.mean)
    np.testing.assert_array_equal(fitted.std, second.std)
    np.testing.assert_array_equal(fitted.coef, second.coef)
    assert all(
        torch.equal(v, second.residual.state_dict()[k])
        for k, v in fitted.residual.state_dict().items()
    )
    assert 0 <= fitted.alpha <= 1 and 0 <= second.alpha <= 1


@pytest.mark.parametrize("with_network", [False, True])
def test_fitted_artifact_roundtrip_and_schema_order(artificial_data, tmp_path, with_network):
    data = artificial_data
    fitted = fit_model(
        data,
        np.arange(len(data.y)) < 420,
        Protocol(epochs=1),
        spec() if with_network else None,
        seed=19,
    )
    if with_network:
        # Ensure the residual branch is exercised even if held-out alpha rejects it.
        fitted.alpha = 0.5
    fitted.save(tmp_path / "artifact")
    loaded = FittedModel.load(tmp_path / "artifact")
    np.testing.assert_array_equal(fitted.predict(data), loaded.predict(data))
    reordered = Dataset(
        data.x[:, :, ::-1].copy(), data.y, data.scale, data.meta, data.features[::-1]
    )
    np.testing.assert_array_equal(fitted.predict(data), loaded.predict(reordered))


def test_normalized_baseline_and_residual_share_economic_units(artificial_data):
    data = artificial_data

    class ConstantResidual(torch.nn.Module):
        def forward(self, x):
            return torch.ones(x.shape[0], dtype=x.dtype, device=x.device) * 0.5

    fitted = FittedModel(
        spec(),
        data.features,
        np.zeros(2, dtype=np.float32),
        np.ones(2, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        0.25,
        ConstantResidual(),
        0.4,
        {"prediction_clip": 0.1},
    )
    np.testing.assert_allclose(fitted.predict(data), (0.25 + 0.4 * 0.5) * data.scale)


def test_custom_protocol_parameter_limit_is_enforced(artificial_data):
    with pytest.raises(ValueError, match="parameter"):
        fit_model(
            artificial_data,
            np.arange(len(artificial_data.y)) < 420,
            replace(Protocol(epochs=1), max_parameters=10),
            spec(),
        )


def test_custom_protocol_prediction_clip_is_enforced(artificial_data):
    data = artificial_data
    train = np.arange(len(data.y)) < 420
    fitted = fit_model(data, train, Protocol(epochs=1, prediction_clip=0.0001))
    assert np.max(np.abs(fitted.predict(data))) <= 0.0001 + 1e-10


@pytest.mark.parametrize(
    "item",
    [
        {"alpha": 0.01},
        {"partition": "search", "alpha": 0.01},
        {"evidence_cutoff": "2025-02-01", "alpha": 0.01},
        {"partition": "confirmation", "evidence_cutoff": "2025-02-01", "alpha": 0.01},
        {"partition": "search", "evidence_cutoff": "2025-04-01", "alpha": 0.01},
    ],
)
def test_consumed_agent_memory_requires_mature_search_provenance(item):
    with pytest.raises(ValueError):
        diagnostic_proposal("parent", Protocol(), "2025-03-01", [item], 0)


def test_valid_agent_memory_can_generate_a_proposal():
    protocol = Protocol()
    item = {"partition": "search", "evidence_cutoff": "2025-02-01", "alpha": 0.01}
    proposal = diagnostic_proposal("parent", protocol, "2025-03-01", [item], 0)
    assert proposal.operation == "drop_expert"
    proposal.validate(protocol, "2025-03-01", "parent")


def test_unknown_agent_cutoff_is_rejected():
    protocol = Protocol()
    proposal = diagnostic_proposal("parent", protocol, "2025-03-01", [], 0)
    with pytest.raises(ValueError):
        replace(proposal, evidence_cutoff="NaT").validate(protocol, "2025-03-01", "parent")
    with pytest.raises(ValueError):
        diagnostic_proposal(
            "parent",
            protocol,
            "2025-03-01",
            [{"partition": "search", "evidence_cutoff": "NaT", "alpha": 0}],
            0,
        )


def test_empty_prediction_and_invalid_causal_scales(artificial_data):
    data = artificial_data
    fitted = fit_model(data, np.arange(len(data.y)) < 420, Protocol(epochs=1))
    empty = data.subset(np.zeros(len(data.y), dtype=bool))
    assert fitted.predict(empty).shape == (0,)
    for invalid in [0, -1, float("nan"), float("inf")]:
        changed = Dataset(data.x, data.y, data.scale.copy(), data.meta, data.features)
        changed.scale[0] = invalid
        with pytest.raises(ValueError):
            fitted.predict(changed)
    for invalid in [float("nan"), float("inf")]:
        changed = Dataset(data.x.copy(), data.y, data.scale, data.meta, data.features)
        changed.x[0, -1, 0] = invalid
        with pytest.raises(ValueError):
            fitted.predict(changed)
