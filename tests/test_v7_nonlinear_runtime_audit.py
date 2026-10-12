"""Independent synthetic fault injections; no trained model or external verifier."""
import pytest
from test_v7_nonlinear_runtime import prepared as prepared_fixture

from factorlab.discovery import nonlinear_runtime as runtime


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    return prepared_fixture.__wrapped__(tmp_path, monkeypatch)


def execute(prepared):
    directory, data, pool, config, gates, _ = prepared
    return runtime.run_gated_residual(directory, data, '2025-05-01T00:00:00Z', pool,
                                      gates, source_id='artificial', config=config)


def test_missing_nonlinear_budget_checkpoint_cannot_repeat_completed_work(prepared, monkeypatch):
    calls = []

    def producer(*args):
        args[-1].consume('solve')
        calls.append(1)
        return {'status': 'completed'}

    monkeypatch.setattr(runtime, 'research_residual', producer)
    execute(prepared)
    store = runtime._Store(prepared[0])
    store.db.execute("DELETE FROM records WHERE kind='budget'")
    store.db.commit()
    store.close()
    with pytest.raises(ValueError, match='budget|checkpoint|orphan|missing|consumed'):
        execute(prepared)
    assert len(calls) == 1


def test_last_work_item_past_deadline_cannot_commit_completed_result(prepared, monkeypatch):
    elapsed = [0.]
    monkeypatch.setattr(runtime.time, 'monotonic', lambda: elapsed[0])

    def producer(*args):
        args[-1].consume('solve')
        elapsed[0] = 3601.
        return {'status': 'completed'}

    monkeypatch.setattr(runtime, 'research_residual', producer)
    with pytest.raises(ValueError, match='time|budget|deadline'):
        execute(prepared)
    store = runtime._Store(prepared[0])
    state = store.get('budget', '2025-05')
    store.close()
    assert state['channel']['counts']['solve'] == 1
    assert state['channel']['spent_seconds'] >= 3601
    assert next(iter(state['attempts'].values()))['status'] == 'failed_consumed'


def test_uninstalled_resource_verifier_fails_before_neural_fit(prepared, monkeypatch):
    def unavailable(*args):
        raise ValueError('Independent verifier unavailable')

    monkeypatch.setattr(runtime, '_external_resource_verifier', unavailable)
    monkeypatch.setattr(runtime, 'research_residual', lambda *args: pytest.fail('Unverified fit'))
    with pytest.raises(ValueError, match='Independent verifier unavailable'):
        execute(prepared)
    store = runtime._Store(prepared[0])
    assert store.get('budget', '2025-05') is None
    store.close()
