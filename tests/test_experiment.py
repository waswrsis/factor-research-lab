"""Bounded policy orchestration regressions; no neural training or market download."""

import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from factorlab.backtest import BacktestResult
from factorlab.dataset import Dataset
from factorlab.graph import ExpertSpec, GraphSpec, mutate
from factorlab.protocol import Protocol
from factorlab.research_agent import diagnostic_proposal, read_proposal
from factorlab.splits import outer_folds


@pytest.mark.parametrize(
    "change", [{"unknown_override": True}, {"hypothesis": None}, {"seed": "17"}, {"seed": 1.5}]
)
def test_bad_external_proposals_are_recordable_value_errors(tmp_path, change):
    protocol = Protocol()
    value = asdict(diagnostic_proposal("parent", protocol, "2025-03-01", [], 0))
    value.update(change)
    path = tmp_path / "proposal.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        read_proposal(path, protocol, "2025-03-01", "parent")


def test_external_agent_can_specify_a_concrete_structural_edit(tmp_path):
    protocol = Protocol()
    value = asdict(diagnostic_proposal("parent", protocol, "2025-03-01", [], 0))
    value.update(
        operation="width", source="external_agent", arguments={"expert": "time", "value": 32}
    )
    path = tmp_path / "proposal.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    proposal = read_proposal(path, protocol, "2025-03-01", "parent")
    parent = GraphSpec(("ret",), (ExpertSpec("time", width=4, scales=(4, 12)),))
    child = mutate(parent, {"operation": proposal.operation, **proposal.arguments}, proposal.seed)
    assert child.experts[0].width == 32
    assert child.features == parent.features
    assert child.horizon_hours == 12
    assert child.target_unit == "simple_return"


def test_search_requests_never_contain_confirmation_or_test_evidence(monkeypatch, tmp_path):
    from factorlab import experiment

    protocol = Protocol(candidates_per_search=2, seeds=(17, 29), bootstrap_samples=20)
    fold = outer_folds(protocol)[0]
    decisions = [
        fold.validation_start - pd.Timedelta(days=2),
        fold.validation_start + pd.Timedelta(hours=1),
        fold.confirmation_start + pd.Timedelta(hours=1),
        fold.test_start + pd.Timedelta(hours=1),
    ]
    meta = pd.DataFrame(
        {
            "available_at": decisions,
            "label_available_at": [x + pd.Timedelta(hours=13) for x in decisions],
            "entry_time": [x + pd.Timedelta(hours=1) for x in decisions],
            "symbol": "BTCUSDT",
        }
    )
    dataset = Dataset(np.zeros((4, 48, 1), np.float32), np.zeros(4), np.ones(4), meta, ("ret",))
    base = GraphSpec(("ret",), (ExpertSpec("time", width=4, scales=(4, 12)),), gating="dynamic")
    fitted = []
    prediction_calls = []
    proposed = []

    class Model:
        alpha = 0.5

        def __init__(self, spec):
            self.spec = spec
            self.diagnostics = {"synthetic_audit_stub": True}

        def predict(self, ds, device):
            prediction_calls.append((self.spec, ds.meta.available_at.iloc[0]))
            return np.full(len(ds.y), 0.01 if self.spec else 0)

        def save(self, path):
            return path

    def fit(data, train, protocol, spec=None, seed=17, device="cpu"):
        assert data.meta.label_available_at[train].max() < fold.validation_start
        assert not any(x[1] >= fold.confirmation_start for x in prediction_calls)
        fitted.append((spec, seed))
        return Model(spec)

    def evaluate(ds, predictions, bars, marks, funding, start, end, protocol, cost_multiplier=1):
        times = pd.date_range(start, end, freq="h", inclusive="left")
        value = float(predictions.mean())
        equity = pd.DataFrame({"return": value, "equity": 100_000.0}, index=times)
        return BacktestResult(
            equity,
            pd.DataFrame(),
            {"utility_hourly": value, "max_drawdown": 0, "solver_failures": 0},
            (),
        )

    original = experiment.diagnostic_proposal

    def propose(parent_id, protocol, cutoff, history, iteration):
        assert all(x["partition"] == "search" for x in history)
        assert all(pd.Timestamp(x["evidence_cutoff"]) < fold.confirmation_start for x in history)
        assert not any("confirmation" in x or "test" in x for x in history)
        assert not any(x[1] >= fold.confirmation_start for x in prediction_calls)
        proposed.append(iteration)
        return original(parent_id, protocol, cutoff, history, iteration)

    monkeypatch.setattr(
        experiment,
        "load_market",
        lambda p: (
            {"klines": pd.DataFrame(), "mark": pd.DataFrame(), "funding": pd.DataFrame()},
            {},
        ),
    )
    monkeypatch.setattr(experiment, "make_dataset", lambda bars, p: dataset)
    monkeypatch.setattr(experiment, "starter_graph", lambda data, p: base)
    monkeypatch.setattr(experiment, "fit_model", fit)
    monkeypatch.setattr(experiment, "evaluate_predictions", evaluate)
    monkeypatch.setattr(experiment, "diagnostic_proposal", propose)
    result = experiment.run_experiment(protocol, tmp_path / "run", fold_limit=1)
    assert proposed == [0, 1]
    assert len(fitted) <= 1 + len(protocol.seeds) * (1 + 2 * protocol.candidates_per_search)
    for arm in ("C_random_search", "D_diagnostic_agent"):
        records = result["folds"][0]["arms"][arm]["search_records"]
        assert len(records) == protocol.candidates_per_search
        assert all(x["status"] in ("failed", "evaluated") for x in records)
    assert all(not arm["confirmation"]["passed"] for arm in result["folds"][0]["arms"].values())
    for arm in result["folds"][0]["arms"].values():
        assert arm["deployed_policy"]["utility_hourly"] == 0
    with pytest.raises(ValueError, match="already used"):
        experiment.run_experiment(protocol, tmp_path / "run", fold_limit=1)
