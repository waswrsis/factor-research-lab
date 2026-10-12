"""Actual Lifecycle window/restore checks; fake selection losses are plumbing only."""
import json

import pandas as pd
import pytest
from test_v7_forward_audit import _receive
from test_v7_forward_audit import runtime as runtime  # noqa: PLC0414
from test_v7_lifecycle_audit import _freeze, _selected
from test_v7_lifecycle_audit import audit_store as audit_store  # noqa: PLC0414

from factorlab.discovery.joint_pool import FrozenPool
from factorlab.discovery.v7_forward import ForwardRuntime
from factorlab.discovery.v7_lifecycle import Lifecycle


def view(store):
    result = ForwardRuntime.__new__(ForwardRuntime)
    result.lifecycle = store
    return result


@pytest.mark.parametrize(('at', 'permitted'), [
    ('2026-10-31T23:59:59Z', False), ('2026-11-01T00:00:00Z', True),
    ('2026-11-30T23:59:59Z', True), ('2026-12-01T00:00:00Z', False),
    ('2027-11-01T00:00:05Z', False),
])
def test_validation_exact_origin_window_and_a0_continuity(audit_store, at, permitted):
    store, clock = audit_store
    _freeze(store)
    clock[0] = pd.Timestamp(at)
    store.verify()
    recipes = view(store)._serving_recipes(clock[0], clock[0])
    assert set(recipes) == ({'A0', 'A:reference', 'A:factor-A'} if permitted else {'A0'})


def test_late_issue_cannot_extend_validation_window(audit_store):
    store, clock = audit_store
    _freeze(store)
    clock[0] = pd.Timestamp('2026-12-01T00:00:01Z')
    store.verify()
    assert view(store)._serving_recipes(clock[0] - pd.Timedelta(seconds=2), clock[0]) == {'A0': None}


@pytest.mark.parametrize(('offset', 'permitted'), [(-1, False), (0, True), (1, True)])
def test_confirmation_real_reservation_actual_start_boundary(audit_store, monkeypatch, offset, permitted):
    store, clock = audit_store
    _selected(store, clock, monkeypatch)
    receipt = json.loads(store.exposure.db.execute('SELECT identity FROM confirmations').fetchone()[0])
    actual = pd.Timestamp(receipt['actual_start'])
    assert actual > pd.Timestamp(receipt['nominal_start'])
    clock[0] = actual + pd.Timedelta(seconds=offset)
    store.verify()
    recipes = view(store)._serving_recipes(clock[0], clock[0])
    assert set(recipes) == ({'A0', 'A:reference', 'A:factor-A'} if permitted else {'A0'})


def test_confirmation_ends_at_origin_end_not_scoring_grace_and_restart_does_not_revive(audit_store, monkeypatch):
    store, clock = audit_store
    _selected(store, clock, monkeypatch)
    receipt = json.loads(store.exposure.db.execute('SELECT identity FROM confirmations').fetchone()[0])
    end = pd.Timestamp(receipt['end'])
    clock[0] = end - pd.Timedelta(seconds=5)
    store.verify()
    assert 'A:factor-A' in view(store)._serving_recipes(clock[0], clock[0])
    # Origin is inside C although the forecast's next-hour start + 12 h is outside.
    assert clock[0].ceil('h') + pd.Timedelta(hours=12) > end
    for moment in (end, end + pd.Timedelta(hours=1), pd.Timestamp(receipt['scoring_cutoff'])):
        clock[0] = moment
        restored = Lifecycle(store.directory, store.plan, clock=lambda: clock[0])
        try:
            restored.verify()
            assert view(restored)._serving_recipes(moment, moment) == {'A0': None}
        finally:
            restored.close()


def test_selected_without_confirmation_publication_does_not_serve_c(audit_store, monkeypatch):
    store, clock = audit_store
    _freeze(store)
    clock[0] = pd.Timestamp('2026-12-01T00:00:05Z')
    from test_v7_lifecycle_audit import _loss_rows
    monkeypatch.setattr(store, '_losses', lambda stream, start, end, as_of, **kwargs:
        _loss_rows(start, end, 1. if stream.endswith(':reference') else .9))
    assert store.select_validation('A')['selected']
    store.verify()
    assert view(store)._serving_recipes(clock[0], clock[0]) == {'A0': None}


def freeze_runtime(store, clock):
    reference = store.recipe_for()
    pool = FrozenPool(version='frozen-candidate')
    candidate = store.recipe_for(pool)
    store.lifecycle.exposure.record(source=store.feed.identity, symbols=store.config.symbols,
        start='2026-07-01T00:00:00Z', end='2026-09-01T00:00:00Z', role='S', batch='live')
    store.lifecycle.freeze_batch(batch='live', ordinal=1, start='2026-10-01T00:00:00Z',
        candidates=[candidate | {'complexity': 0}], reference=reference,
        evidence_sources=[{'start': '2026-07-01T00:00:00Z', 'end': '2026-09-01T00:00:00Z'}])
    clock.now = pd.Timestamp('2026-10-01T00:00:10Z')
    for stream, selected_pool in [('A0', None), ('live:reference', None),
                                  (f'live:{candidate["pipeline_id"]}', pool)]:
        store.fit_month('2026-10-01T00:00:00Z', stream=stream, pool=selected_pool,
                        authority=store.test_authority)
    # Unknown streams are now rejected before fitting, not merely filtered at serving.
    with pytest.raises(ValueError, match='protected batch|frozen claim'):
        store.fit_month('2026-10-01T00:00:00Z', stream='unknown:orphan',
                        authority=store.test_authority)
    assert store.lifecycle.active_model('unknown:orphan') is None
    return candidate


def test_tick_stops_expired_and_unknown_stream_inference_but_retains_a0_and_old_settlement(runtime, monkeypatch):
    store, clock, history, _ = runtime
    candidate = freeze_runtime(store, clock)
    _receive(runtime, '2026-09-30T23:00Z', '2026-10-01T01:00Z', '2026-10-01T01:00:05Z')
    assert store.tick()['new_predictions'] == 3
    assert store.lifecycle.active_model(f'live:{candidate["pipeline_id"]}')
    # Mature actual labels remain settleable after the serving window expires.
    clock.now = pd.Timestamp('2026-11-01T01:00:05Z')
    segment = history.loc[(history.open_time >= '2026-10-01T01:00Z')
                          & (history.open_time < '2026-11-01T01:00Z')]
    store.feed.receive(segment, received_at=clock(), response_hash='b' * 64)
    calls, original = [], store._activate_ready

    def observed(stream, origin, **kwargs):
        calls.append(stream)
        return original(stream, origin, **kwargs)

    monkeypatch.setattr(store, '_activate_ready', observed)
    result = store.tick()
    assert result['new_predictions'] == 1 and result['new_settlements'] == 3
    assert calls == ['A0']
    assert store.lifecycle.active_model(f'live:{candidate["pipeline_id"]}')  # State is not permission.
    records = list(store.lifecycle.forecasts.iter_records())
    assert len(records) == 4 and records[-1]['stream'] == 'A0'


def test_corrupt_frozen_batch_fails_before_any_tick_prediction(runtime):
    store, clock, _, _ = runtime
    freeze_runtime(store, clock)
    store.lifecycle.db.execute("UPDATE batches SET identity='{}'")
    store.lifecycle.db.commit()
    with pytest.raises((ValueError, KeyError), match='.*'):
        store.tick()
    assert not list(store.lifecycle.forecasts.iter_records())


def test_same_pipeline_id_cannot_hide_different_frozen_pool_recipe(runtime):
    store, clock, _, _ = runtime
    candidate = freeze_runtime(store, clock)
    clock.now = pd.Timestamp('2026-10-01T01:00:05Z')
    forged = dict(candidate, pool=FrozenPool(version='other-month-pool').to_dict())
    with pytest.raises(ValueError, match='frozen pipeline recipe'):
        store._activate_ready(f'live:{candidate["pipeline_id"]}', clock(), expected_pipeline=forged)
