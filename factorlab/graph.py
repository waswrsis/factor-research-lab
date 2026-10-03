"""Typed, bounded structure proposals. No operation may change the evaluation task."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any

from torch import nn

MAX_PARAMETERS = 1_000_000
EXPERT_RESERVED_NAMES = frozenset(dir(nn.ModuleDict()))


@dataclass(frozen=True)
class ExpertSpec:
    name: str
    kind: str = "temporal"
    width: int = 16
    depth: int = 1
    scales: tuple[int, ...] = (4, 12, 48)


@dataclass(frozen=True)
class FeatureOp:
    name: str
    kind: str
    inputs: tuple[str, ...]


@dataclass(frozen=True)
class GraphSpec:
    features: tuple[str, ...]
    experts: tuple[ExpertSpec, ...]
    lookback: int = 48
    horizon_hours: int = 12
    output_unit: str = "vol_scaled_simple_return"
    target_unit: str = "simple_return"
    gating: str = "static"
    loss: str = "mse"
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    feature_ops: tuple[FeatureOp, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return to_dict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GraphSpec:
        return from_dict(value)


def _name(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and value.isidentifier()


def _integer(value: Any, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


def _parameter_count(spec: GraphSpec) -> int:
    f = len(spec.features) + len(spec.feature_ops)
    total = 0
    for expert in spec.experts:
        if expert.kind == "temporal":
            total += sum(expert.width * (f * s + 1) for s in expert.scales)
            incoming = len(expert.scales) * expert.width
        else:
            incoming = 2 * f
        total += (incoming + 1) * expert.width
        total += (expert.depth - 1) * (expert.width + 1) * expert.width
        total += expert.width + 1
    if spec.gating == "dynamic":
        total += (len(spec.features) + 1) * len(spec.experts)
    else:
        total += len(spec.experts)
    return total


def validate_graph(spec: GraphSpec) -> GraphSpec:
    """Raise ValueError on invalid/over-budget graphs and return the validated spec."""
    if not isinstance(spec, GraphSpec):
        raise ValueError("expected GraphSpec")  # noqa: TRY004 - proposal validation API
    if not isinstance(spec.features, tuple) or not spec.features:
        raise ValueError("features must be a nonempty tuple")
    if not all(_name(f) for f in spec.features) or len(set(spec.features)) != len(spec.features):
        raise ValueError("feature names must be unique identifiers")
    if not _integer(spec.lookback, 1, 4096) or not _integer(spec.horizon_hours, 12, 12):
        raise ValueError("lookback must be 1..4096 and task horizon must remain 12 hours")
    if spec.output_unit != "vol_scaled_simple_return" or spec.target_unit != "simple_return":
        raise ValueError("output/target units cannot change")
    if spec.gating not in {"static", "dynamic"} or spec.loss not in {"mse", "huber"}:
        raise ValueError("unsupported gating or loss")
    if spec.optimizer not in {"adam", "adamw"}:
        raise ValueError("unsupported optimizer")
    if (
        isinstance(spec.learning_rate, bool)
        or not isinstance(spec.learning_rate, (int, float))
        or not math.isfinite(spec.learning_rate)
        or not 1e-6 <= spec.learning_rate <= 0.1
    ):
        raise ValueError("learning_rate must be finite and in [1e-6, 0.1]")
    if not isinstance(spec.experts, tuple) or not 1 <= len(spec.experts) <= 16:
        raise ValueError("one to sixteen experts required")
    names = set()
    for e in spec.experts:
        if (
            not isinstance(e, ExpertSpec)
            or not _name(e.name)
            or e.name in names
            or e.name in EXPERT_RESERVED_NAMES
        ):
            raise ValueError("expert names must be unique identifiers")
        names.add(e.name)
        if e.kind not in {"temporal", "mlp"}:
            raise ValueError("unsupported expert kind")
        if not _integer(e.width, 1, 128) or not _integer(e.depth, 1, 4):
            raise ValueError("expert width/depth outside bounds")
        if (
            not isinstance(e.scales, tuple)
            or not e.scales
            or len(set(e.scales)) != len(e.scales)
            or not all(_integer(s, 1, spec.lookback) for s in e.scales)
        ):
            raise ValueError("scales must be unique positive integers within lookback")
    if not isinstance(spec.feature_ops, tuple) or len(spec.feature_ops) > 64:
        raise ValueError("feature_ops must be a tuple with at most 64 operations")
    available = set(spec.features)
    for op in spec.feature_ops:
        if not isinstance(op, FeatureOp) or not _name(op.name) or op.name in available:
            raise ValueError("operation names must be unique feature identifiers")
        arity = {"product": 2, "difference": 2, "square": 1, "tanh": 1}.get(op.kind)
        if (
            arity is None
            or not isinstance(op.inputs, tuple)
            or len(op.inputs) != arity
            or not all(i in available for i in op.inputs)
        ):
            raise ValueError("operation inputs must reference existing preceding features")
        available.add(op.name)
    if _parameter_count(spec) > MAX_PARAMETERS:
        raise ValueError(f"graph exceeds {MAX_PARAMETERS} parameter budget")
    return spec


def parameter_count(spec: GraphSpec) -> int:
    validate_graph(spec)
    return _parameter_count(spec)


def to_dict(spec: GraphSpec) -> dict[str, Any]:
    validate_graph(spec)
    result = asdict(spec)
    result["features"] = list(spec.features)
    result["experts"] = [{**asdict(e), "scales": list(e.scales)} for e in spec.experts]
    result["feature_ops"] = [{**asdict(op), "inputs": list(op.inputs)} for op in spec.feature_ops]
    return result


def from_dict(value: Mapping[str, Any]) -> GraphSpec:
    if not isinstance(value, Mapping):
        raise ValueError("graph must be a mapping")  # noqa: TRY004 - proposal validation API
    try:
        fields = dict(value)
        fields["features"] = tuple(fields["features"])
        fields["experts"] = tuple(
            ExpertSpec(**{**e, "scales": tuple(e.get("scales", (4, 12, 48)))})
            for e in fields["experts"]
        )
        fields["feature_ops"] = tuple(
            FeatureOp(**{**op, "inputs": tuple(op["inputs"])})
            for op in fields.get("feature_ops", ())
        )
        return validate_graph(GraphSpec(**fields))
    except (KeyError, TypeError) as exc:
        raise ValueError(f"malformed graph: {exc}") from exc


def mutate(
    spec: GraphSpec,
    operation: str | Mapping[str, Any],
    seed: int = 0,
    feature_pool: tuple[str, ...] = (),
) -> GraphSpec:
    """Apply one bounded edit; mapping keys select an expert/value explicitly.

    A string selects deterministic seeded defaults. Invalid and no-op edits fail,
    so they can be recorded as failed proposals rather than scored as new graphs.
    """
    validate_graph(spec)
    rng = random.Random(seed)
    op = {"operation": operation} if isinstance(operation, str) else dict(operation)
    kind = op.get("operation", op.get("op"))
    aliases = {
        "remove_expert": "drop_expert",
        "change_width": "width",
        "change_depth": "depth",
        "change_scales": "scales",
        "gate": "gating",
        "change_gate": "gating",
        "change_loss": "loss",
        "change_optimizer": "optimizer",
        "learningrate": "learning_rate",
        "rawfeature": "add_feature",
        "interaction": "add_interaction",
        "remove_feature": "drop_feature",
    }
    kind = aliases.get(kind, kind)
    allowed = {"operation", "op", "expert", "value", "name", "kind", "inputs", "scales"}
    if set(op) - allowed:
        raise ValueError("unknown proposal fields (evaluation protocol edits are forbidden)")
    experts = list(spec.experts)
    selectable = [e for e in experts if e.kind == "temporal"] if kind == "scales" else experts
    if not selectable:
        raise ValueError("scale mutations require a temporal expert")
    selected = op.get("expert", rng.choice(selectable).name)
    index = next((i for i, e in enumerate(experts) if e.name == selected), None)
    if kind == "add_expert":
        name = op.get(
            "name",
            next(
                f"expert_{i}" for i in range(100) if f"expert_{i}" not in {e.name for e in experts}
            ),
        )
        experts.append(
            ExpertSpec(
                name,
                kind=op.get("kind", rng.choice(["temporal", "mlp"])),
                scales=tuple(
                    op.get("scales", tuple(s for s in (4, 12, 48) if s <= spec.lookback) or (1,))
                ),
            )
        )
        child = replace(spec, experts=tuple(experts))
    elif kind == "drop_expert":
        if index is None:
            raise ValueError("unknown expert")
        experts.pop(index)
        child = replace(spec, experts=tuple(experts))
    elif kind in {"width", "depth", "scales"}:
        if index is None:
            raise ValueError("unknown expert")
        e = experts[index]
        if kind == "scales" and e.kind != "temporal":
            raise ValueError("scale mutations require a temporal expert")
        scale_choices = [
            tuple(sorted({min(s, spec.lookback) for s in scales}))
            for scales in [(1,), (4, 12), (4, 12, 48)]
        ]
        choices = {"width": [8, 16, 32, 64], "depth": [1, 2, 3, 4], "scales": scale_choices}
        valid = [v for v in choices[kind] if v != getattr(e, kind)]
        if not valid and "value" not in op:
            raise ValueError("no alternative layout within bounds")
        value = op["value"] if "value" in op else rng.choice(valid)
        if kind == "scales":
            value = tuple(value)
        experts[index] = replace(e, **{kind: value})
        child = replace(spec, experts=tuple(experts))
    elif kind in {"gating", "loss", "optimizer", "learning_rate"}:
        choices = {
            "gating": ["static", "dynamic"],
            "loss": ["mse", "huber"],
            "optimizer": ["adam", "adamw"],
            "learning_rate": [1e-4, 3e-4, 1e-3, 3e-3],
        }
        value = op.get("value", rng.choice([v for v in choices[kind] if v != getattr(spec, kind)]))
        child = replace(spec, **{kind: value})
    elif kind == "add_feature":
        available = sorted(set(feature_pool) - set(spec.features))
        if not available:
            raise ValueError("no observed feature available to add")
        value = op.get("value", op.get("name", rng.choice(available)))
        if value not in available:
            raise ValueError("raw feature must be new and from the observed feature pool")
        child = replace(spec, features=spec.features + (value,))
    elif kind == "drop_feature":
        value = op.get("value", rng.choice(spec.features))
        if value not in spec.features:
            raise ValueError("unknown feature")
        child = replace(spec, features=tuple(f for f in spec.features if f != value))
    elif kind == "add_interaction":
        name = op.get(
            "name",
            next(
                f"interaction_{i}"
                for i in range(100)
                if f"interaction_{i}" not in set(spec.features) | {x.name for x in spec.feature_ops}
            ),
        )
        operation_kind = op.get("kind", "product")
        inputs = tuple(op.get("inputs", (rng.choice(spec.features), rng.choice(spec.features))))
        child = replace(
            spec, feature_ops=spec.feature_ops + (FeatureOp(name, operation_kind, inputs),)
        )
    elif kind == "drop_interaction":
        if not spec.feature_ops:
            raise ValueError("no interaction to drop")
        name = op.get("name", rng.choice(spec.feature_ops).name)
        child = replace(spec, feature_ops=tuple(o for o in spec.feature_ops if o.name != name))
    else:
        raise ValueError(f"unsupported operation: {kind}")
    validate_graph(child)
    if child == spec:
        raise ValueError("proposal did not change graph")
    return child
