from dataclasses import replace

import pytest

from factorlab.graph import (
    ExpertSpec,
    FeatureOp,
    GraphSpec,
    from_dict,
    mutate,
    parameter_count,
    to_dict,
    validate_graph,
)


def graph():
    return GraphSpec(("ret", "vol"), (ExpertSpec("time"), ExpertSpec("interaction", "mlp")))


def test_round_trip_and_parameter_count():
    spec = replace(graph(), feature_ops=(FeatureOp("cross", "product", ("ret", "vol")),))
    assert from_dict(to_dict(spec)) == spec
    assert spec.from_dict(spec.to_dict()) == spec
    assert parameter_count(spec) > parameter_count(graph())


@pytest.mark.parametrize(
    "operation",
    [
        "add_expert",
        "drop_expert",
        "width",
        "depth",
        "scales",
        "gating",
        "loss",
        "optimizer",
        "learning_rate",
        "add_feature",
        "add_interaction",
        "drop_feature",
    ],
)
def test_mutation_changes_spec(operation):
    parent = graph()
    child = mutate(parent, operation, feature_pool=("ret", "vol", "funding"))
    assert child != parent
    assert child.horizon_hours == parent.horizon_hours
    assert child.target_unit == parent.target_unit
    assert child.output_unit == parent.output_unit
    assert mutate(parent, operation, feature_pool=("ret", "vol", "funding")) == child


def test_unsafe_or_invalid_edits_rejected():
    with pytest.raises(ValueError, match="observed"):
        mutate(graph(), {"operation": "add_feature", "value": "future"}, feature_pool=("funding",))
    with pytest.raises(ValueError, match="protocol"):
        mutate(graph(), {"operation": "width", "horizon_hours": 1})
    with pytest.raises(ValueError):
        mutate(graph(), {"operation": "width", "value": 999})
    with pytest.raises(ValueError):
        validate_graph(replace(graph(), horizon_hours=1))
    with pytest.raises(ValueError):
        validate_graph(replace(graph(), output_unit="price"))
    with pytest.raises(ValueError, match="parameter budget"):
        validate_graph(
            GraphSpec(tuple(f"f{i}" for i in range(300)), (ExpertSpec("big", width=128),))
        )


def test_feature_dag_validation_and_dependent_removal():
    spec = replace(
        graph(),
        feature_ops=(
            FeatureOp("cross", "product", ("ret", "vol")),
            FeatureOp("squash", "tanh", ("cross",)),
        ),
    )
    validate_graph(spec)
    with pytest.raises(ValueError, match="preceding"):
        mutate(spec, {"operation": "drop_feature", "value": "ret"})
    with pytest.raises(ValueError, match="preceding"):
        mutate(spec, {"operation": "drop_interaction", "name": "cross"})
    assert len(mutate(spec, {"operation": "drop_interaction", "name": "squash"}).feature_ops) == 1
    with pytest.raises(ValueError):
        from_dict({**to_dict(graph()), "cost": 0})


@pytest.mark.parametrize(
    "change",
    [{"width": True}, {"depth": 0}, {"scales": (49,)}, {"scales": (4, 4)}, {"kind": "unknown"}],
)
def test_expert_limits(change):
    with pytest.raises(ValueError):
        validate_graph(replace(graph(), experts=(replace(ExpertSpec("bad"), **change),)))


def test_scale_mutations_target_actual_temporal_structure():
    with pytest.raises(ValueError, match="temporal"):
        mutate(graph(), {"operation": "scales", "expert": "interaction", "value": [1]})
    short = GraphSpec(("ret",), (ExpertSpec("short", scales=(1,)),), lookback=2)
    child = mutate(short, "scales")
    assert child.experts[0].scales == (2,)
    assert parameter_count(short) != parameter_count(child)
    with pytest.raises(ValueError):
        validate_graph(replace(graph(), horizon_hours=12.0))


@pytest.mark.parametrize("name", ["training", "forward", "keys", "_modules"])
def test_expert_module_reserved_names_rejected(name):
    with pytest.raises(ValueError, match="expert names"):
        validate_graph(replace(graph(), experts=(ExpertSpec(name),)))
