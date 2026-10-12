"""Independent artificial production-boundary counterexamples; no NN/GPT/network."""
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest
from test_v7_nonlinear_launcher import job_for
from test_v7_nonlinear_orchestration import owner as owner_fixture

from factorlab.discovery import nonlinear_launcher as launcher
from factorlab.discovery import nonlinear_orchestration as orchestration
from factorlab.discovery import nonlinear_runtime as runtime
from factorlab.discovery.known_library import FactorFrame
from factorlab.discovery.rolling import development_folds


@pytest.fixture
def owner(tmp_path, monkeypatch):
    yield from owner_fixture.__wrapped__(tmp_path, monkeypatch)


def capacity(data, cutoff):
    return {'train_row_capacity': max(int(data.train_mask(f['fit_cutoff']).sum())
                                     for f in development_folds(cutoff)),
            'score_row_capacity': max(int(data.score_mask(f['fit_cutoff'], f['score_end'], f['score_end']).sum())
                                     for f in development_folds(cutoff)),
            'frame_row_capacity': len(data.frame.values), 'feature_capacity': data.frame.values.shape[1],
            'frame_byte_capacity': data.frame.values.nbytes}


def padded_future(data, copies=8):
    count = len(data.frame.values) * copies
    dates = pd.date_range('2026-01-01', periods=count, freq='h', tz='UTC')
    meta = pd.DataFrame({'symbol': 'X', 'available_at': dates,
        'feature_open_time': dates - pd.Timedelta(hours=1), 'forecast_window_start': dates,
        'sample_id': [f'future-{i}' for i in range(count)]})
    values = np.tile(data.frame.values, (copies, 1))
    frame = FactorFrame(np.vstack([data.frame.values, values]),
        pd.concat([data.frame.metadata, meta], ignore_index=True), data.frame.feature_names,
        data.frame.bypass_mask, data.frame.schema_hash)
    return replace(data, frame=frame, target=np.r_[data.target, np.ones(count)],
        observed=np.r_[data.observed, np.ones(count, bool)],
        maturity=data.maturity.append(dates + pd.Timedelta(hours=12)))


def test_resource_trial_does_not_certify_larger_unscored_frame(owner):
    _, data, cutoff, _, _, _ = owner
    expanded = padded_future(data)
    assert capacity(data, cutoff)['train_row_capacity'] == capacity(expanded, cutoff)['train_row_capacity']
    assert capacity(data, cutoff)['score_row_capacity'] == capacity(expanded, cutoff)['score_row_capacity']
    assert len(expanded.frame.values) == 9 * len(data.frame.values)
    with pytest.raises(ValueError, match='capacity|frame|rows|bytes'):
        runtime._capacity(capacity(data, cutoff), expanded, cutoff)


def test_capacity_rejects_more_feature_columns_even_with_same_fold_counts(owner):
    _, data, cutoff, _, _, _ = owner
    frame = FactorFrame(np.c_[data.frame.values, np.ones(len(data.frame.values))], data.frame.metadata,
        (*data.frame.feature_names, 'extra'), (*data.frame.bypass_mask, False), data.frame.schema_hash)
    changed = replace(data, frame=frame)
    with pytest.raises(ValueError, match='capacity|feature|schema|bytes'):
        runtime._capacity(capacity(data, cutoff), changed, cutoff)


def test_worker_ticket_rechecks_shared_s_exposure_after_reservation(owner):
    bridge, *_ = owner
    _, job = job_for(owner)
    assert launcher._ticket(job)['status'] == 'reserved'
    exposed = bridge.state['search_exposure']
    bridge.store.exposure.record(**{k: v for k, v in exposed.items() if k != 'data_id'}, role='V')
    with pytest.raises(ValueError, match='exposure|protected|role|S|V'):
        launcher._ticket(job)


def test_same_neural_experiment_cannot_be_repeated_by_changing_seconds(owner):
    bridge, _, _, _, epoch, _ = owner
    first = orchestration.reserve_neural_ticket(bridge, epoch, 'same-gate', seconds=30)
    assert first['status'] == 'reserved'
    # Explicitly failed work remains charged; a duration change is not new evidence.
    bridge.state['nonlinear']['attempts'][first['ticket_id']]['status'] = 'failed_consumed'
    bridge.commit()
    before = bridge.budget.counts['solve']
    second = orchestration.reserve_neural_ticket(bridge, epoch, 'same-gate', seconds=29)
    assert second['status'] == 'already_consumed'
    assert bridge.budget.counts['solve'] == before


def test_same_month_producer_cannot_substitute_data_after_epoch_freeze(owner):
    bridge, data, cutoff, folder, epoch, _ = owner
    target = data.target.copy()
    mature = data.train_mask(cutoff)
    target[mature] *= 7
    changed = replace(data, target=target)
    reached = []

    def produce(ledger):
        reached.append(True)
        ledger.consume('solve')
        return {'status': 'completed'}

    with pytest.raises(ValueError, match='data|snapshot|context|identity'):
        orchestration._produce(bridge, folder, changed, cutoff, epoch, 'probe', produce)
    assert not reached


def test_atomic_global_commit_failure_preserves_both_charges(owner):
    bridge, _, _, _, epoch, _ = owner
    ledger = bridge.channel(epoch)
    original = bridge.budget.persist

    def commit_then_die(value):
        original(value)
        raise OSError('synthetic death after actual authoritative commit')

    bridge.budget.persist = commit_then_die
    with pytest.raises(OSError, match='authoritative commit'):
        ledger.consume('solve', 2)
    saved = bridge.store.get(bridge.month)
    assert saved['budget']['counts']['solve'] == saved['nonlinear']['channel']['counts']['solve'] == 2


def test_source_change_rejects_before_any_quarterly_producer(owner, monkeypatch):
    bridge, data, cutoff, folder, epoch, _ = owner
    (bridge.store.directory.parent / 'factorlab/v7.py').write_text('version=changed')
    monkeypatch.setattr(orchestration, 'run_probe', lambda *args: pytest.fail('Wrong source reached producer'))
    with pytest.raises(ValueError, match='active monthly'):
        orchestration.produce_quarterly_artifacts(folder, bridge, data, cutoff, epoch)
    assert bridge.budget.counts['solve'] == 0


def test_future_rows_and_unmatured_labels_are_not_transferred_to_child(owner):
    bridge, data, cutoff, folder, _, _ = owner
    maturity = list(data.maturity)
    last_visible = np.flatnonzero(data.frame.metadata.available_at < pd.Timestamp(cutoff))[-1]
    maturity[last_visible] = pd.Timestamp(cutoff) + pd.Timedelta(hours=1)
    data = replace(data, maturity=pd.DatetimeIndex(maturity))
    expanded = padded_future(data, copies=1)
    context = runtime._context(data, cutoff, orchestration.epoch_pool(), bridge.source_id)
    assert runtime._context(expanded, cutoff, orchestration.epoch_pool(), bridge.source_id) == context
    saved = launcher.save_worker_data(folder / 'raw-transfer-evidence', expanded, cutoff=cutoff)
    restored = launcher._load_data({'data': saved, 'rolling': asdict(data.spec)})
    protected = restored.frame.metadata.available_at >= pd.Timestamp(cutoff)
    assert not protected.any()
    immature = ~restored.observed
    assert immature.any()
    assert (restored.target[immature] == 0).all()
    assert restored.maturity[immature].isna().all()


def test_actual_availability_cannot_be_backdated_for_retroactive_quarter(owner):
    bridge, data, cutoff, folder, _, _ = owner
    store = runtime._Store(folder)
    try:
        context = runtime._context(data, cutoff, orchestration.epoch_pool(), bridge.source_id)
        identifier = runtime._register(store, 'probe', context, {'status': 'completed'})
        # Producer clock is April 2; April 1 cannot consume this artifact.
        with pytest.raises(ValueError, match='availability'):
            runtime._read(store, identifier, context, pd.Timestamp(cutoff))
    finally:
        store.close()


def test_parent_resource_boolean_is_not_production_evidence():
    with pytest.raises(ValueError, match='verifier'):
        runtime._external_resource_verifier({'verified': True, 'scope': 'whole_cgroup'}, {})
