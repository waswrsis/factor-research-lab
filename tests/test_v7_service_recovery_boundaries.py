"""Recipe-transition scheduling cuts and original-service restart boundaries.

Scheduling fixtures do not establish model admission. Actual service-store tests
use a real source-bound runtime and empty real receipt archive, without fitting.
"""
import json

import pandas as pd
import pytest

from factorlab.discovery.v7_forward import ForwardRuntime
from factorlab.discovery.v7_service import ForwardService, ServiceConfig


def _activation_cut(tmp_path, *, stream='A0_policy_trial', cutoff='2026-09-01T00:00:00+00:00',
                    with_service=False):
    # Real storage, runtime identity and original service ownership/clock. The
    # rows and active-model callback below are scheduling seams, not admissions.
    now = pd.Timestamp('2026-10-01T00:00:05+00:00')
    runtime = ForwardRuntime(tmp_path/'forward', tmp_path/'feed', clock=lambda: now)
    if with_service:
        # Construct before synthetic scheduling rows: constructor verifies the
        # actual empty store and establishes its original durable execution.
        runtime._boundary_service = ForwardService(runtime, config=ServiceConfig(poll=False),
                                                   guard=lambda: {'allowed': True})
    db = runtime.lifecycle.db
    prior, selected = {'pipeline_id': 'prior-K0'}, {'pipeline_id': 'new-typed-NN'}
    db.execute('INSERT INTO jobs(id,cutoff,stream,pipeline,status,failure,completed) '
               'VALUES (?,?,?,?,?,?,?)', ('old', cutoff, stream,
        json.dumps(prior), 'activated', None, cutoff))
    db.execute('INSERT INTO active(stream,job,generation) VALUES (?,?,?)', (stream, 'old', 1))
    db.commit()
    checked = []
    active = {'pipeline': prior, 'generation': 1}
    def original_active(name):
        checked.append(name)
        return active
    runtime.lifecycle.active_model = original_active
    return runtime, now, prior, selected, checked

def test_new_month_policy_recipe_waits_without_relabelling_old_activation(tmp_path):
    runtime, now, prior, selected, checked = _activation_cut(tmp_path)
    try:
        before = runtime.lifecycle.db.execute('SELECT * FROM active').fetchall()
        assert runtime._activate_ready('A0_policy_trial', now, expected_pipeline=selected) is None
        assert checked == ['A0_policy_trial']
        assert runtime.lifecycle.db.execute('SELECT * FROM active').fetchall() == before
        assert runtime._activate_ready('A0_policy_trial', now, expected_pipeline=prior)['pipeline'] == prior
        assert runtime._activate_ready('A0_policy_trial', now, expected_pipeline=None)['pipeline'] == prior
    finally:
        runtime.close()


@pytest.mark.parametrize(('stream', 'cutoff'), [
    ('A0_policy_trial', '2026-10-01T00:00:00+00:00'),
    ('batch:fixed_candidate', '2026-09-01T00:00:00+00:00'),
])
def test_mismatch_is_still_rejected_for_current_policy_or_fixed_stream(tmp_path, stream, cutoff):
    runtime, now, _, selected, _ = _activation_cut(tmp_path, stream=stream, cutoff=cutoff)
    try:
        with pytest.raises(ValueError, match='Serving model differs'):
            runtime._activate_ready(stream, now, expected_pipeline=selected)
    finally:
        runtime.close()


def test_policy_transition_reaches_normal_service_plan_and_fit(tmp_path):
    runtime, now, _, selected, checked = _activation_cut(tmp_path, with_service=True)
    service = runtime._boundary_service
    def scheduling_tick(**kwargs):
        activation = runtime._activate_ready('A0_policy_trial', now, expected_pipeline=selected)
        return {'at': now.isoformat(), 'activation': activation, 'forecasts': 0}
    runtime.tick = scheduling_tick
    calls = []
    service._freeze_due = service._endpoints = lambda *args: None
    service._plan_month = lambda now: calls.append('plan')
    service._algorithm_step = lambda *args, **kwargs: 0
    service._fit_pending = lambda *args, **kwargs: calls.append('fit')
    try:
        result = service.step()
        assert result['status'] == 'stepped'
        assert result['tick']['activation'] is None  # No old model relabelled.
        assert result['tick']['forecasts'] == 0
        assert service.state['steps'] == 1
        assert service.state['active_since'] is None
        assert checked == ['A0_policy_trial']
        assert calls == ['plan', 'fit']
    finally:
        service.close(close_runtime=False)
        runtime.close()


@pytest.mark.parametrize('has_active', [False, True])
def test_previous_verified_policy_recipe_is_not_activated_as_new_recipe(tmp_path, has_active):
    runtime, now, prior, selected, _ = _activation_cut(tmp_path, cutoff='2026-08-01T00:00:00+00:00')
    db = runtime.lifecycle.db
    if not has_active:
        db.execute('DELETE FROM active')
        runtime.lifecycle.active_model = lambda stream: None
    db.execute('INSERT INTO jobs(id,cutoff,stream,pipeline,status,failure,completed) '
               'VALUES (?,?,?,?,?,?,?)', ('unactivated',
        '2026-09-01T00:00:00+00:00', 'A0_policy_trial', json.dumps(prior), 'verified', None,
        '2026-09-01T00:00:05+00:00'))
    runtime.lifecycle.activate = lambda *a, **k: pytest.fail('Old recipe must stay unactivated')
    try:
        assert runtime._activate_ready('A0_policy_trial', now, expected_pipeline=selected) is None
        assert db.execute('SELECT status FROM jobs WHERE id=?', ('unactivated',)).fetchone() == ('verified',)
    finally:
        runtime.close()


@pytest.mark.parametrize(('stream', 'cutoff'), [
    ('A0_policy_trial', '2026-10-01T00:00:00+00:00'),
    ('batch:fixed_candidate', '2026-09-01T00:00:00+00:00'),
])
def test_current_or_fixed_verified_wrong_recipe_still_fails(tmp_path, monkeypatch, stream, cutoff):
    runtime, now, _, selected, _ = _activation_cut(tmp_path, stream=stream, cutoff=cutoff)
    runtime.lifecycle.db.execute("UPDATE jobs SET status='verified'")
    runtime.lifecycle.db.execute('DELETE FROM active')
    runtime.lifecycle.active_model = lambda name: None
    # Publication visibility only; this scheduling test creates no admission.
    monkeypatch.setattr('factorlab.discovery.quarantined_observation.publication_visible', lambda *a: True)
    try:
        with pytest.raises(ValueError, match='Serving model differs'):
            runtime._activate_ready(stream, now, expected_pipeline=selected)
    finally:
        runtime.close()


@pytest.fixture
def actual_runtime(tmp_path):
    runtime = ForwardRuntime(tmp_path/'forward', tmp_path/'feed')
    try:
        yield runtime
    finally:
        runtime.close()


def test_actual_service_fresh_and_intact_restart_preserve_budget(actual_runtime):
    config = ServiceConfig(poll=False, max_steps=1)
    service = ForwardService(actual_runtime, config=config)
    try:
        before = dict(service.state)
        assert before['steps'] == 0 and before['fit_attempts'] == {}
    finally:
        service.close(close_runtime=False)
    reopened = ForwardService(actual_runtime, config=config)
    try:
        assert reopened.state == before
    finally:
        reopened.close(close_runtime=False)


def test_actual_service_missing_database_cannot_reset_original_budget(actual_runtime, tmp_path):
    config = ServiceConfig(poll=False, max_steps=1)
    service = ForwardService(actual_runtime, config=config)
    # Persist an already consumed original supervisor projection; there is no
    # fit work, model, admission, kernel or provider substitute in this test.
    service.state.update(steps=1, spent_seconds=17.)
    service._save('bounded_test_original_consumed_supervisor_step')
    path = service.directory/'state.sqlite'
    service.close(close_runtime=False)
    assert path.resolve().is_relative_to(tmp_path.resolve())
    original = path.with_name('preserved_original_state.sqlite')
    path.rename(original)
    with pytest.raises(ValueError, match='state database missing; cannot reset original budget'):
        ForwardService(actual_runtime, config=config)
    assert not path.exists() and original.is_file()


def test_actual_service_empty_existing_checkpoint_cannot_initialize(actual_runtime):
    service = ForwardService(actual_runtime, config=ServiceConfig(poll=False))
    try:
        # Even complete loss of both SQL projections is not first initialization.
        service.db.execute('DELETE FROM checkpoint')
        service.db.execute('DELETE FROM events')
        service.db.commit()
    finally:
        service.close(close_runtime=False)
    with pytest.raises(ValueError, match='checkpoint missing'):
        ForwardService(actual_runtime, config=ServiceConfig(poll=False))


@pytest.mark.parametrize('also_state', [False, True])
def test_actual_service_missing_identity_or_both_files_cannot_recreate(actual_runtime, tmp_path, also_state):
    config = ServiceConfig(poll=False)
    service = ForwardService(actual_runtime, config=config)
    directory = service.directory
    service.close(close_runtime=False)
    assert directory.resolve().is_relative_to(tmp_path.resolve())
    (directory/'identity.json').rename(directory/'preserved_original_identity.json')
    if also_state:
        (directory/'state.sqlite').rename(directory/'preserved_original_state.sqlite')
    with pytest.raises(ValueError, match='identity missing from existing namespace'):
        ForwardService(actual_runtime, config=config)
    assert not (directory/'identity.json').exists()
    assert (directory/'state.sqlite').exists() is (not also_state)
