"""Independent hook counterexample. Fake gates, no NN, market fit or network."""
from dataclasses import replace

import pandas as pd
import pytest
from test_v7_nonlinear_orchestration import owner as owner_fixture

from factorlab.discovery import nonlinear_orchestration as orchestration
from factorlab.discovery import nonlinear_runtime as runtime
from factorlab.discovery import v7_run


@pytest.fixture
def owner(tmp_path, monkeypatch):
    yield from owner_fixture.__wrapped__(tmp_path, monkeypatch)


def test_controller_resource_selection_uses_exact_same_causal_worker_frame(owner, monkeypatch):
    bridge, raw, cutoff, *_ = owner
    # run_month admits raw available_at <= cutoff. A zero-delay candle can
    # yield exactly this boundary row; the worker correctly uses strict <.
    chosen = (raw.frame.metadata.available_at <= pd.Timestamp(cutoff)).to_numpy()
    data = replace(raw, frame=raw.frame.take(chosen), target=raw.target[chosen],
                   observed=raw.observed[chosen], maturity=raw.maturity[chosen])
    causal = orchestration.causal_worker_data(data, cutoff)
    assert len(data.frame.values) == len(causal.frame.values) + 1
    bridge.state['nonlinear_input_data_id'] = orchestration.input_data_identity(data, cutoff)
    bridge.commit()
    folder = bridge.store.directory / 'nonlinear_epoch'
    context = runtime._context(causal, cutoff, orchestration.epoch_pool(), bridge.source_id)
    artifact = {**runtime.workload_capacity(causal, cutoff), 'scope': 'artificial-verifier-fixture-only'}
    runtime._capacity(artifact, causal, cutoff)  # Actual worker workload fits.
    monkeypatch.setattr(runtime, '_now', lambda: pd.Timestamp(cutoff))
    receipt_store = runtime._Store(folder)
    try:
        identifier = runtime._register(receipt_store, 'resource', context, artifact)
    finally:
        receipt_store.close()
    monkeypatch.setattr(runtime, '_resource', lambda *a: {'verified': True})
    monkeypatch.setattr(runtime, 'verify_gate_artifacts', lambda *a, **kw: {'test_only': False})
    monkeypatch.setattr(orchestration, 'produce_quarterly_artifacts', lambda *a: {'status': 'cached'})
    monkeypatch.setattr(orchestration, 'prior_gate_artifacts',
        lambda *a: {'status': 'research_evidence_verified', 'artifacts': {}})
    launches = []
    monkeypatch.setattr(orchestration, 'run_isolated_research',
        lambda *a, **kw: launches.append(kw) or {'status': 'no-work-performed'})

    def checkpoint(event):
        bridge.store.save(bridge.month, bridge.state, event)

    result = v7_run._nonlinear_month(bridge.store, bridge.month, bridge.state, bridge.budget,
        data, pd.Timestamp(cutoff), 'isolated', 3600., checkpoint)
    assert not result['resource_capacity_rejections']
    assert launches == [{'resource_receipt_id': identifier, 'resource_trial': False}]
