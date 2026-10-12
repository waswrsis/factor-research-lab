"""Actual-feed snapshot contracts using artificial candles, no remote calls."""
import copy
import json
from dataclasses import replace

import pandas as pd
import pytest
from test_v7_feed_audit import Clock
from test_v7_rolling_registry import candles
from test_v7_run_factor_pool import source_root

from factorlab.discovery.forward_fit_snapshot import verify_prepared_forward_snapshot
from factorlab.discovery.v7_forward import ForwardRuntime


@pytest.fixture
def setup(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    clock = Clock('2026-09-30T23:30:00Z')
    runtime = ForwardRuntime(tmp_path / 'run', tmp_path / 'feed', clock=clock)
    path = tmp_path / 'seed.parquet'
    candles('2026-05-01T00:00Z', '2026-09-30T23:00Z').to_parquet(path)
    runtime.feed.import_seed(path)
    clock.now = pd.Timestamp('2026-10-01T00:00:10Z')
    job = runtime.lifecycle.plan_fit('2026-10-01T00:00:00Z', 'A0', runtime.recipe_for())
    yield runtime, clock, job
    runtime.close()


def test_snapshot_seals_actual_publication_but_keeps_seed_provenance(setup):
    runtime, _, job = setup
    prepared = runtime.prepare_fit_snapshot(job)
    assert prepared.record['availability'] == 'actual_receipts'
    assert prepared.data.availability_mode == 'mixed_seed_and_observed_feed'
    assert prepared.record['train_start'] == '2026-07-01T00:00:00+00:00'
    assert prepared.record['publication_lineage']
    assert all(pd.Timestamp(r['published_at']) < pd.Timestamp(prepared.record['cutoff'])
               for r in prepared.record['publication_lineage'])
    assert runtime.lifecycle.db.execute('SELECT status FROM jobs WHERE id=?', (job,)).fetchone()[0] == 'features_ready'
    assert verify_prepared_forward_snapshot(runtime, job, prepared) == prepared.record


def test_preparation_does_not_fit_and_recovery_is_exact(setup, monkeypatch):
    runtime, _, job = setup
    # Trap each real numerical producer after the old unmetered alias was removed.
    for path in ('factorlab.discovery.pool_forward.fit_pool',
                 'factorlab.discovery.neural_forward.fit_pool',
                 'factorlab.discovery.known_linear_forward.fit_known_linear'):
        monkeypatch.setattr(path, lambda *_, **__: pytest.fail('Snapshot preparation cannot train'))
    first = runtime.prepare_fit_snapshot(job)
    second = runtime.prepare_fit_snapshot(job)
    assert first.verify() == second.verify()
    assert runtime.lifecycle.db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == 1


def test_context_is_retained_but_only_three_months_are_training_labels(setup):
    runtime, _, job = setup
    prepared = runtime.prepare_fit_snapshot(job)
    origins = prepared.data.frame.metadata.available_at
    assert origins.min() < pd.Timestamp(prepared.record['train_start'])
    mask = prepared.data.train_mask(prepared.record['cutoff'])
    assert (origins.loc[mask] >= pd.Timestamp(prepared.record['train_start'])).all()
    assert (origins.loc[mask] < pd.Timestamp(prepared.record['cutoff'])).all()
    unavailable = ~prepared.data.observed
    assert unavailable.any()
    assert (prepared.data.target[unavailable] == 0).all()
    assert prepared.data.maturity[unavailable].isna().all()


@pytest.mark.parametrize('kind', ['future_target', 'mature_target', 'maturity', 'record', 'stored'])
def test_payload_or_snapshot_changes_are_detected(setup, kind):
    runtime, _, job = setup
    prepared = runtime.prepare_fit_snapshot(job)
    target = prepared.data.target.copy()
    if kind == 'future_target':
        target[~prepared.data.observed] = 10
        prepared = replace(prepared, data=replace(prepared.data, target=target))
    elif kind == 'mature_target':
        target[prepared.data.observed] *= 2
        prepared = replace(prepared, data=replace(prepared.data, target=target))
    elif kind == 'maturity':
        maturity = prepared.data.maturity.where(prepared.data.observed,
                                               pd.Timestamp('2026-11-01T00:00:00Z'))
        prepared = replace(prepared, data=replace(prepared.data, maturity=maturity))
    elif kind == 'record':
        record = copy.deepcopy(prepared.record)
        record['cutoff'] = '2026-11-01T00:00:00+00:00'
        prepared = replace(prepared, record=record)
    else:
        path = runtime.directory / 'snapshots' / f'{job}.json'
        record = json.loads(path.read_text())
        record['raw_asof_hash'] = '0' * 64
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='snapshot|identity|answer|Snapshot'):
        verify_prepared_forward_snapshot(runtime, job, prepared)


def test_later_receipt_cannot_rewrite_cutoff_snapshot(setup):
    runtime, clock, job = setup
    first = runtime.prepare_fit_snapshot(job)
    clock.now = pd.Timestamp('2026-10-01T01:00:05Z')
    runtime.feed.receive(candles('2026-09-30T23:00Z', '2026-10-01T01:00Z'),
                         received_at=clock(), response_hash='a' * 64)
    assert runtime.prepare_fit_snapshot(job).verify() == first.verify()


def test_edited_frozen_source_cannot_prepare_worker_inputs(setup):
    runtime, _, job = setup
    (runtime.directory / 'source/factorlab/v7.py').write_text('tampered')
    with pytest.raises(ValueError, match='source archive'):
        runtime.prepare_fit_snapshot(job)


def test_changed_publication_index_cannot_supply_worker_provenance(setup):
    runtime, _, job = setup
    runtime.feed.db.execute("UPDATE publications SET published_at='2026-09-01T00:00:00+00:00'")
    runtime.feed.db.commit()
    with pytest.raises(ValueError, match='publication|Publication'):
        runtime.prepare_fit_snapshot(job)


def test_unknown_job_and_changed_rolling_recipe_are_rejected(setup):
    runtime, _, _ = setup
    with pytest.raises(ValueError, match='authoritative'):
        runtime.prepare_fit_snapshot('0' * 64)
    recipe = runtime.recipe_for()
    recipe['rolling_hash'] = '0' * 64
    job = runtime.lifecycle.plan_fit('2026-09-01T00:00:00Z', 'alternate', recipe)
    with pytest.raises(ValueError, match='rolling rule'):
        runtime.prepare_fit_snapshot(job)
