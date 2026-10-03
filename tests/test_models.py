from dataclasses import replace

import pytest
import torch

from factorlab.graph import ExpertSpec, FeatureOp, GraphSpec, mutate, parameter_count
from factorlab.models import build_model, inherit_compatible


def graph(gating="dynamic"):
    return GraphSpec(
        ("ret", "vol"),
        (ExpertSpec("time", scales=(4, 12)), ExpertSpec("nonlinear", "mlp")),
        gating=gating,
    )


@pytest.mark.parametrize("gating", ["static", "dynamic"])
def test_shape_count_gradient_and_actual_training(gating):
    model = build_model(graph(gating), seed=2)
    assert sum(p.numel() for p in model.parameters()) == parameter_count(model.spec)
    x = torch.randn(16, 48, 2)
    target = x[:, -1, 0] + 0.4 * x[:, -4:, 1].mean(dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    before = ((model(x) - target) ** 2).mean().item()
    for _ in range(12):
        optimizer.zero_grad()
        prediction = model(x)
        assert prediction.shape == (16,)
        loss = ((prediction - target) ** 2).mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert all(p.grad.abs().sum() > 0 for p in model.parameters())
        optimizer.step()
    assert ((model(x) - target) ** 2).mean().item() < before


def test_independent_filters_and_structural_parameter_changes():
    spec = graph()
    model = build_model(spec)
    first = model.experts["time"].filters["4"].weight
    second = model.experts["time"].filters["12"].weight
    assert first.data_ptr() != second.data_ptr()
    for op in ["add_expert", "drop_expert", "width", "depth", "scales", "add_interaction"]:
        changed = mutate(spec, op)
        changed_model = build_model(changed)
        assert sum(p.numel() for p in changed_model.parameters()) == parameter_count(changed)
        assert parameter_count(changed) != parameter_count(spec)


def test_causal_endpoint_and_batch_independence():
    # A stream wrapper slices only the window ending at t. Appending/changing
    # future observations therefore cannot change any earlier model endpoint.
    model = build_model(graph())
    stream = torch.randn(90, 2)
    t = 60
    before = model(stream[t - 47 : t + 1].unsqueeze(0))
    altered = stream.clone()
    altered[t + 1 :] = 1000
    assert torch.equal(before, model(altered[t - 47 : t + 1].unsqueeze(0)))
    batch = torch.randn(4, 48, 2)
    original = model(batch)[0]
    batch[1:] *= 100
    assert torch.equal(original, model(batch)[0])
    assert torch.allclose(model(batch[:1]), model(batch)[:1], atol=1e-7)


def test_gate_allocations_and_state_contract():
    model = build_model(graph())
    x = torch.randn(3, 48, 2)
    weights = model.gate_weights(x)
    assert weights.shape == (3, 2)
    assert torch.allclose(weights.sum(dim=1), torch.ones(3))
    assert torch.allclose(model(x), model(x, state=x[:, -1]))
    with pytest.raises(ValueError, match="state"):
        model(x, state=torch.randn(3, 3))
    with pytest.raises(ValueError):
        build_model(graph("static"))(x, state=x[:, -1])
    with pytest.raises(ValueError):
        model(torch.randn(3, 47, 2))
    x[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(x)


def test_seed_and_feature_operations():
    spec = replace(
        graph(),
        feature_ops=(
            FeatureOp("cross", "product", ("ret", "vol")),
            FeatureOp("squared", "square", ("cross",)),
        ),
    )
    torch.manual_seed(99)
    state = torch.random.get_rng_state().clone()
    first, second = build_model(spec, 17), build_model(spec, 17)
    assert torch.equal(state, torch.random.get_rng_state())
    assert all(torch.equal(v, second.state_dict()[k]) for k, v in first.state_dict().items())
    x = torch.randn(2, 48, 2)
    augmented = first._augment(x)
    assert torch.equal(augmented[:, :, 2], x[:, :, 0] * x[:, :, 1])
    assert torch.equal(augmented[:, :, 3], augmented[:, :, 2].square())
    assert first(x).shape == (2,)


def test_inheritance_reports_and_rejects_semantic_layout_change():
    parent = build_model(graph(), 1)
    child = build_model(graph(), 2)
    report = inherit_compatible(parent, child)
    assert report["copied"] and not report["skipped"]
    assert all(torch.equal(v, child.state_dict()[k]) for k, v in parent.state_dict().items())
    reordered = build_model(replace(graph(), features=("vol", "ret")), 2)
    original = {k: v.clone() for k, v in reordered.state_dict().items()}
    report = inherit_compatible(parent, reordered)
    assert not report["copied"] and report["skipped"]
    assert all(torch.equal(v, original[k]) for k, v in reordered.state_dict().items())
    wider = build_model(mutate(graph(), {"operation": "width", "expert": "time", "value": 32}))
    report = inherit_compatible(parent, wider)
    assert any(k.startswith("experts.nonlinear") for k in report["copied"])
    assert all(not k.startswith("experts.time") for k in report["copied"])
    assert any(k.startswith("experts.time") for k in report["skipped"])
