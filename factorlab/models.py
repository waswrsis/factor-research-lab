"""Trainable causal-window experts with a relative-allocation gate."""

from __future__ import annotations

import torch
from torch import nn

from .graph import GraphSpec, validate_graph


def _head(incoming: int, width: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(incoming, width), nn.Tanh()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(width, width), nn.Tanh()])
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


class TemporalExpert(nn.Module):
    def __init__(self, features: int, width: int, depth: int, scales: tuple[int, ...]):
        super().__init__()
        self.filters = nn.ModuleDict({str(s): nn.Conv1d(features, width, s) for s in scales})
        self.head = _head(width * len(scales), width, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Each filter sees only its trailing scale and emits exactly one endpoint.
        values = [
            torch.tanh(layer(x[:, -int(s) :].transpose(1, 2))).squeeze(-1)
            for s, layer in self.filters.items()
        ]
        return self.head(torch.cat(values, dim=-1)).squeeze(-1)


class MLPExpert(nn.Module):
    def __init__(self, features: int, width: int, depth: int):
        super().__init__()
        self.head = _head(2 * features, width, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([x[:, -1], x.mean(dim=1)], dim=-1)).squeeze(-1)


class FactorNetwork(nn.Module):
    def __init__(self, spec: GraphSpec):
        super().__init__()
        self.spec = validate_graph(spec)
        features = len(spec.features) + len(spec.feature_ops)
        self.experts = nn.ModuleDict(
            {
                e.name: (
                    TemporalExpert(features, e.width, e.depth, e.scales)
                    if e.kind == "temporal"
                    else MLPExpert(features, e.width, e.depth)
                )
                for e in spec.experts
            }
        )
        if spec.gating == "dynamic":
            self.gate = nn.Linear(len(spec.features), len(spec.experts))
        else:
            self.gate_logits = nn.Parameter(torch.zeros(len(spec.experts)))

    def _check(self, x: torch.Tensor, state: torch.Tensor | None) -> None:
        if x.ndim != 3 or x.shape[1:] != (self.spec.lookback, len(self.spec.features)):
            raise ValueError("X must have shape [B, lookback, len(raw features)]")
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError("X must contain finite floating-point values")
        if state is not None:
            if self.spec.gating != "dynamic":
                raise ValueError("optional state is only used by dynamic gating")
            if state.shape != (x.shape[0], len(self.spec.features)):
                raise ValueError("gate state must have shape [B, len(raw features)]")
            if (
                state.device != x.device
                or state.dtype != x.dtype
                or not torch.isfinite(state).all()
            ):
                raise ValueError("gate state must be finite and share X dtype/device")

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        values = {name: x[:, :, i] for i, name in enumerate(self.spec.features)}
        ordered = list(values.values())
        for op in self.spec.feature_ops:
            inputs = [values[name] for name in op.inputs]
            if op.kind == "product":
                value = inputs[0] * inputs[1]
            elif op.kind == "difference":
                value = inputs[0] - inputs[1]
            elif op.kind == "square":
                value = inputs[0].square()
            else:
                value = inputs[0].tanh()
            values[op.name] = value
            ordered.append(value)
        augmented = torch.stack(ordered, dim=-1)
        if not torch.isfinite(augmented).all():
            raise ValueError("feature operations produced nonfinite values")
        return augmented

    def gate_weights(self, x: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        """Relative expert allocations; these are not predictive confidence scores."""
        self._check(x, state)
        if self.spec.gating == "dynamic":
            logits = self.gate(x[:, -1] if state is None else state)
        else:
            logits = self.gate_logits.unsqueeze(0).expand(x.shape[0], -1)
        return logits.softmax(dim=-1)

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        weights = self.gate_weights(x, state)
        augmented = self._augment(x)
        values = torch.stack([e(augmented) for e in self.experts.values()], dim=-1)
        return (weights * values).sum(dim=-1)


def build_model(spec: GraphSpec, seed: int = 0) -> FactorNetwork:
    # Construction is deterministic without resetting the caller's RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return FactorNetwork(spec)


def inherit_compatible(parent: FactorNetwork, child: FactorNetwork) -> dict[str, list[str]]:
    """Copy full tensors only when shape AND semantic feature/expert layout match.

    This is an initialization aid, not a function-preserving transformation.
    """
    copied, skipped = [], []
    source = parent.state_dict()
    target = child.state_dict()
    layout_ok = (
        parent.spec.features == child.spec.features
        and parent.spec.feature_ops == child.spec.feature_ops
        and parent.spec.lookback == child.spec.lookback
    )
    old_experts = {e.name: e for e in parent.spec.experts}
    new_experts = {e.name: e for e in child.spec.experts}
    gate_ok = (
        layout_ok
        and parent.spec.gating == child.spec.gating
        and tuple(old_experts) == tuple(new_experts)
    )
    with torch.no_grad():
        for name, tensor in target.items():
            compatible = layout_ok
            if name.startswith("experts."):
                expert_name = name.split(".")[1]
                # The whole expert layout must agree, including scale order/head layout.
                compatible = compatible and old_experts.get(expert_name) == new_experts[expert_name]
            else:
                compatible = compatible and gate_ok
            if compatible and name in source and source[name].shape == tensor.shape:
                tensor.copy_(source[name].to(device=tensor.device, dtype=tensor.dtype))
                copied.append(name)
            else:
                skipped.append(name)
    return {"copied": copied, "skipped": skipped}
