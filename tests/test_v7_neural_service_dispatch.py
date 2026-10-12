"""Typed scheduling on artificial actual-feed stores; no production worker claim."""
import copy

import pytest
from test_v7_neural_forward_launcher import real_ticket_case as real_ticket_case  # noqa: PLC0414
from test_v7_dynamic_forward_gates import fit_pending_in_original_service_scope

from factorlab.discovery.neural_forward import FrozenNeuralRecipe
from factorlab.discovery.v7_service import ForwardService, ServiceConfig


def service(case, **kwargs):
    return ForwardService(case.runtime, config=ServiceConfig(poll=False),
                          guard=lambda: {'allowed': True}, **kwargs)


def test_service_keeps_full_neural_identity_instead_of_extracting_pool(real_ticket_case):
    c = real_ticket_case
    s = service(c)
    try:
        pipeline = c.runtime.recipe_for(c.recipe)
        learner = s._pool({**pipeline, 'complexity': 3})
        assert isinstance(learner, FrozenNeuralRecipe)
        assert learner.pipeline_id == c.recipe.pipeline_id
        assert c.runtime.recipe_for(learner) == pipeline
        assert c.runtime.recipe_for(learner.pool)['pipeline_id'] != pipeline['pipeline_id']
        changed = copy.deepcopy(pipeline)
        changed['recipe']['output'] = 'encoder_ridge'
        with pytest.raises(ValueError, match='runtime fitting policy'):
            s._pool(changed)
    finally:
        s.close()


def test_monthly_dispatch_cannot_train_neural_without_actual_authority(real_ticket_case, monkeypatch):
    c = real_ticket_case
    monkeypatch.setattr('factorlab.discovery.pool_forward.fit_pool',
                        lambda *a, **k: pytest.fail('Neural recipes must not degrade into convex fits'))
    monkeypatch.setattr('factorlab.discovery.neural_forward.fit_pool',
                        lambda *a, **k: pytest.fail('Neural recipes must not degrade into convex fits'))
    with pytest.raises(ValueError, match='actual authority'):
        c.runtime.fit_month('2026-10-01T00:00:00Z', pool=c.recipe)


def test_service_calls_full_neural_entry_once_with_frozen_gate_references(real_ticket_case, monkeypatch):
    c = real_ticket_case
    gates = {c.recipe.pipeline_id: {'fixture': 'no-production-resource-approval'}}
    s = service(c, neural_authority=c.authority, neural_gates=gates)
    calls = []
    def dispatch(cutoff, **kwargs):
        calls.append((cutoff, kwargs))
        c.runtime.lifecycle.fail_fit(c.ticket.value['context']['job'], 'integrity_failure')
        raise ValueError('engineering-stop-before-numerical-fit')
    monkeypatch.setattr(c.runtime, 'fit_month', dispatch)
    try:
        actions = []
        fit_pending_in_original_service_scope(s, c.runtime.lifecycle.now(), actions)
        assert len(calls) == 1
        assert calls[0][1]['pool'].pipeline_id == c.recipe.pipeline_id
        assert calls[0][1]['authority'] is c.authority
        assert calls[0][1]['gates'] == gates[c.recipe.pipeline_id]
        fit_pending_in_original_service_scope(s, c.runtime.lifecycle.now(), [])
        assert len(calls) == 1
    finally:
        s.close()


def test_mutated_gate_map_is_not_a_new_service_permission(real_ticket_case, monkeypatch):
    c = real_ticket_case
    s = service(c, neural_authority=c.authority, neural_gates={c.recipe.pipeline_id: {}})
    monkeypatch.setattr(c.runtime, 'fit_month', lambda *a, **k: pytest.fail('Mutated gates cannot launch'))
    try:
        s.neural_gates[c.recipe.pipeline_id]['verified'] = True
        with pytest.raises(ValueError, match='static authority/gate identity changed'):
            fit_pending_in_original_service_scope(s, c.runtime.lifecycle.now(), [])
        assert not s.state['fit_attempts']
    finally:
        s.close()


def test_neural_worker_runtime_failure_retains_controller_and_old_model(real_ticket_case, monkeypatch):
    c = real_ticket_case
    s = service(c, neural_authority=c.authority, neural_gates={c.recipe.pipeline_id: {}})
    job = c.ticket.value['context']['job']
    def failure(*a, **k):
        c.runtime.lifecycle.fail_fit(job, 'integrity_failure')
        raise RuntimeError('Fixed full worker exited 19')
    monkeypatch.setattr(c.runtime, 'fit_month', failure)
    try:
        actions = []
        fit_pending_in_original_service_scope(s, c.runtime.lifecycle.now(), actions)
        assert actions[0]['status'] == 'integrity_failure'
    finally:
        s.close()


@pytest.mark.parametrize('positive_cache', [False, True])
def test_attempt_ceiling_allows_only_verified_original_cache_recovery(real_ticket_case, monkeypatch, positive_cache):
    c = real_ticket_case
    config = ServiceConfig(poll=False, max_fit_attempts=1)
    kwargs = {'config': config, 'guard': lambda: {'allowed': True},
              'neural_authority': c.authority, 'neural_gates': {c.recipe.pipeline_id: {}}}
    s = ForwardService(c.runtime, **kwargs)
    job = c.ticket.value['context']['job']
    s.state['fit_attempts'][job] = {'count': 1, 'last_at': None}
    s._save('artificial_crash_after_registration_before_lifecycle_seal')
    s.close(close_runtime=False)
    resumed = ForwardService(c.runtime, **kwargs)
    calls = []
    # Scheduling seam only. Real admission/registry corruption is tested
    # separately; these flags do not exist in the production cache probe API.
    monkeypatch.setattr(c.runtime, 'has_registered_neural_completion', lambda *a: positive_cache)
    def restore(*a, **k):
        calls.append(k)
        return {'job': job, 'model_id': 'original-artificial-model', 'cached': True}
    monkeypatch.setattr(c.runtime, 'fit_month', restore)
    try:
        actions = []
        resumed._fit_pending(c.runtime.lifecycle.now(), actions, resumed.monotonic() + 10)
        assert len(calls) == int(positive_cache)
        assert resumed.state['fit_attempts'][job]['count'] == 1
    finally:
        resumed.close()


def test_cache_probe_does_not_promote_reserved_or_unregistered_worker(real_ticket_case):
    c = real_ticket_case
    assert not c.runtime.has_registered_neural_completion(c.ticket.value['context']['job'], c.recipe)
