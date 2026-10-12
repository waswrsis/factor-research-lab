import json

import numpy as np
import pandas as pd
import pytest

from factorlab.discovery.convex_eval import fit_ridge
from factorlab.discovery.v7_lifecycle import Lifecycle, LifecyclePlan


@pytest.fixture
def controller(tmp_path, monkeypatch):
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    clock = [pd.Timestamp('2026-10-04T03:00:05Z')]
    value = Lifecycle(tmp_path / 'run', LifecyclePlan('feed-identity'), clock=lambda: clock[0])
    yield value, clock
    value.close()


def ready(controller, stream='A0', cutoff='2026-10-01T00:00:00Z', pipeline='K0'):
    model = fit_ridge(np.arange(60).reshape(-1, 1), np.linspace(.001, .03, 60))
    job = controller.plan_fit(cutoff, stream, {'pipeline_id': pipeline})
    controller.advance_fit(job, 'snapshot_sealed', output='sha256-data')
    controller.advance_fit(job, 'features_ready', output='sha256-features')
    identifier = controller.seal_model(job, model)
    return job, identifier


def test_completion_activation_cas_failure_and_restart(controller):
    store, clock = controller
    job, model = ready(store)
    with pytest.raises(ValueError, match='completion'):
        store.activate(job, clock[0] - pd.Timedelta(seconds=1))
    assert store.activate(job, clock[0]) == 1
    assert store.activate(job, clock[0]) == 1
    clock[0] = pd.Timestamp('2026-11-01T00:00:05Z')
    newer = store.plan_fit('2026-11-01T00:00:00Z', 'A0', {'pipeline_id': 'K0'})
    store.fail_fit(newer, 'resource_incomplete')
    assert store.active_model('A0')['model_id'] == model
    store.advance_fit(newer, 'snapshot_sealed')
    store.advance_fit(newer, 'features_ready')
    store.seal_model(newer, fit_ridge(np.arange(60).reshape(-1, 1), np.linspace(.002, .02, 60)))
    with pytest.raises(ValueError, match='generation'):
        store.activate(newer, clock[0], expected_generation=0)
    assert store.activate(newer, clock[0], expected_generation=1) == 2
    assert store.verify()['jobs'] == 2
    with pytest.raises(ValueError, match='reactivated'):
        store.activate(job, clock[0])


def test_receipt_forecast_no_backfill_no_rebinding_and_settlement(controller):
    store, clock = controller
    job, model = ready(store)
    store.activate(job, clock[0])
    row = {'stream': 'A0', 'origin': clock[0], 'symbol': 'BTCUSDT', 'task': 'volatility', 'prediction': .02,
        'received_at': clock[0], 'feature_open_time': '2026-10-04T02:00:00Z',
        'forecast_window_start': '2026-10-04T04:00:00Z', 'model_id': model, 'pipeline_id': 'K0'}
    assert store.record_forecast(**row)
    clock[0] += pd.Timedelta(seconds=1)
    assert not store.record_forecast(**row)
    with pytest.raises(ValueError, match='rebound'):
        store.record_forecast(**(row | {'prediction': .03}))
    with pytest.raises(ValueError, match='timely'):
        store.record_forecast(**(row | {'received_at': '2026-10-04T02:00:00Z'}))
    clock[0] = pd.Timestamp('2026-10-04T16:00:06Z')
    with pytest.raises(ValueError, match='timely'):
        store.record_forecast(**row)
    settlement = {'stream': 'A0', 'origin': row['origin'], 'symbol': 'BTCUSDT', 'task': 'volatility',
        'label': .025, 'maturity': '2026-10-04T16:00:05Z', 'received_at': clock[0]}
    assert store.settle(**settlement)
    assert not store.settle(**settlement)
    with pytest.raises(ValueError, match='overwritten'):
        store.settle(**(settlement | {'label': .04}))
    assert store.verify()['settlements'] == 1


def test_month_boundary_and_no_gate_or_pool_side_adoption(controller):
    store, clock = controller
    with pytest.raises(ValueError, match='exactly'):
        store.plan_fit(clock[0], 'A0', {'pipeline_id': 'K0'})
    assert store.adoption_pipeline('2026-10-01T00:00:00Z', {'pipeline_id': 'K0'}) == {'pipeline_id': 'K0'}
    store.exposure.record(source='feed-identity', symbols=['BTCUSDT'], start='2026-07-01T00:00:00Z',
                          end='2026-10-01T00:00:00Z', role='S', batch='A')
    kwargs = {'batch': 'A', 'ordinal': 1, 'start': '2026-11-01T00:00:00Z',
        'candidates': [{'pipeline_id': 'factor-A', 'complexity': 3}], 'reference': {'pipeline_id': 'K0'},
        'evidence_sources': [{'start': '2026-07-01T00:00:00Z', 'end': '2026-10-01T00:00:00Z'}]}
    store.freeze_batch(**kwargs)
    with pytest.raises(ValueError, match='fixed monthly'):
        store.select_validation('A')
    clock[0] = pd.Timestamp('2026-12-01T00:00:00Z')
    value = store.select_validation('A')
    assert value['selected'] is None and value['records'][0]['coverage'] == 0
    with pytest.raises(ValueError, match='fixed monthly'):
        store.select_validation('A')
    with pytest.raises(ValueError, match='Formal C'):
        store.reserve_selected_confirmation('A')
    assert store.adoption_pipeline('2026-12-01T00:00:00Z', {'pipeline_id': 'K0'})['pipeline_id'] == 'K0'


def test_chain_and_model_corruption_rejected(controller):
    store, _ = controller
    _job, _model = ready(store)
    assert store.verify()['jobs'] == 1
    with store.db:
        store.db.execute('UPDATE operations SET record=? WHERE seq=1', (json.dumps({'bad': True}),))
    with pytest.raises((ValueError, KeyError)):
        store.verify()
