"""Actual lifecycle-chain adoption plumbing; private losses/gate are artificial.

These tests do not certify calibration or a predictive effect. They exercise
the first A0 fit permission after the ordinary C completion/publish path.
"""
import copy
from dataclasses import replace

import pandas as pd
import pytest
from test_v7_lifecycle_audit import _loss_rows
from test_v7_lifecycle_audit import audit_store as audit_store  # noqa: PLC0414
from test_v7_neural_forward_contract import freeze, plan

from factorlab.discovery.neural_forward import FrozenNeuralRecipe
from factorlab.discovery.neural_forward_contract import (
    _confirmed_adoption_binding,
    assert_frozen_forward_fit,
    neural_pipeline,
)


def _confirmed(store, clock, monkeypatch, *, supported=True):
    recipe = FrozenNeuralRecipe()
    freeze(store, recipe)
    clock[0] = pd.Timestamp('2026-12-01T00:00:05Z')
    monkeypatch.setattr(store, '_losses', lambda stream, start, end, as_of, **kwargs:
        _loss_rows(start, end, 1. if stream.endswith(':reference') else .8))
    store.select_validation('A')
    monkeypatch.setattr(store, 'calibration_ready', lambda: True)  # Plumbing seam only.
    store.reserve_selected_confirmation('A')
    # Make unsupported C deterministic without altering the selection step.
    if not supported:
        monkeypatch.setattr(store, '_losses', lambda stream, start, end, as_of, **kwargs:
            _loss_rows(start, end, 1.))
    clock[0] = pd.Timestamp('2027-03-03T00:00:00Z')
    result = store.conclude_confirmation('A')
    assert result['supported'] is supported
    return recipe, result


def test_first_nn_a0_permission_comes_from_completed_c_without_active_nn(audit_store, monkeypatch):
    store, clock = audit_store
    recipe, result = _confirmed(store, clock, monkeypatch)
    assert store.active_model('A0') is None
    clock[0] = pd.Timestamp('2027-04-01T00:00:05Z')
    job = plan(store, recipe, '2027-04-01T00:00Z', 'A0')
    context = assert_frozen_forward_fit(store, job, recipe, purpose='A0')['binding']
    proof = context['adoption']
    assert proof['batch'] == 'A' and proof['completed_at'] == result['completed_at']
    assert context['serving_eligible']
    assert _confirmed_adoption_binding(store, context['cutoff'], context['pipeline'],
        expected=proof, historical=True) == proof


def test_c_finishing_after_month_boundary_cannot_authorize_that_month(audit_store, monkeypatch):
    store, clock = audit_store
    recipe, _ = _confirmed(store, clock, monkeypatch)
    job = plan(store, recipe, '2027-03-01T00:00Z', 'A0')
    with pytest.raises(ValueError, match='adoption authority'):
        assert_frozen_forward_fit(store, job, recipe, purpose='A0')


def test_unsupported_c_cannot_authorize_new_a0(audit_store, monkeypatch):
    store, clock = audit_store
    recipe, _ = _confirmed(store, clock, monkeypatch, supported=False)
    clock[0] = pd.Timestamp('2027-04-01T00:00:05Z')
    job = plan(store, recipe, '2027-04-01T00:00Z', 'A0')
    with pytest.raises(ValueError, match='adoption authority'):
        assert_frozen_forward_fit(store, job, recipe, purpose='A0')


def test_full_recipe_mismatch_and_disabled_calibration_cannot_borrow_c(audit_store, monkeypatch):
    store, clock = audit_store
    recipe, _ = _confirmed(store, clock, monkeypatch)
    clock[0] = pd.Timestamp('2027-04-01T00:00:05Z')
    other = neural_pipeline(replace(recipe, output='encoder_ridge'), recipe.rolling_spec)
    with pytest.raises(ValueError, match='adoption authority'):
        _confirmed_adoption_binding(store, clock[0].floor('D'), other)
    monkeypatch.setattr(store, 'calibration_ready', lambda: False)
    with pytest.raises(ValueError, match='adoption authority'):
        _confirmed_adoption_binding(store, clock[0].floor('D'), neural_pipeline(recipe, recipe.rolling_spec))


def test_historical_proof_must_match_original_completion_and_reservation(audit_store, monkeypatch):
    store, clock = audit_store
    recipe, _ = _confirmed(store, clock, monkeypatch)
    clock[0] = pd.Timestamp('2027-04-01T00:00:05Z')
    pipeline = neural_pipeline(recipe, recipe.rolling_spec)
    proof = _confirmed_adoption_binding(store, '2027-04-01T00:00Z', pipeline)
    bad = copy.deepcopy(proof)
    bad['result_hash'] = '0' * 64
    with pytest.raises(ValueError, match='adoption authority'):
        _confirmed_adoption_binding(store, '2027-04-01T00:00Z', pipeline, expected=bad, historical=True)

