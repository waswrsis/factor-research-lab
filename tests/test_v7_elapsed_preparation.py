"""Actual paid preparation on artificial receipts, not market or native proof."""
from types import SimpleNamespace
import pandas as pd
import pytest
from factorlab.discovery.forward_preparation import ForwardPreparation
from factorlab.discovery.known_linear_reference import FrozenLinearRecipe
from factorlab.discovery.v7_forward import ForwardRuntime,ForwardConfig
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
from factorlab.discovery.work_lease import FrozenWorkLease
from test_v7_run_factor_pool import source_root
from test_v7_feed_audit import Clock
from test_v7_rolling_registry import candles


@pytest.fixture
def case(tmp_path,monkeypatch):
    source_root(tmp_path,monkeypatch)
    clock=Clock('2026-10-10T06:34:39.609948Z')
    runtime=ForwardRuntime(tmp_path/'runtime',tmp_path/'feed',clock=clock,
        config=ForwardConfig(training_window_days=90,refit_interval_days=15))
    try:
        path=tmp_path/'artificial.parquet';candles('2026-02-01T00:00Z','2026-10-10T05:00Z').to_parquet(path)
        runtime.feed.import_seed(path);view=runtime.refresh_training_view()
        authority=MonthlyWorkAuthority(tmp_path/'authority',run_id='elapsed-preparation-artificial',
            task='volatility',universe=list(runtime.config.symbols))
        recipe=FrozenLinearRecipe('K0',runtime.spec)
        cutoff=runtime.lifecycle.due_fit_cutoff()
        job=runtime.lifecycle.plan_fit(cutoff,'A0',runtime.recipe_for(recipe))
        yield SimpleNamespace(runtime=runtime,authority=authority,clock=clock,recipe=recipe,job=job,
            cutoff=cutoff,view=view)
    finally:runtime.close()


def test_actual_elapsed_first_preparation_prepays_before_snapshot_and_preserves_original_lease(case):
    lease=FrozenWorkLease.issue(300)
    prep=ForwardPreparation.begin(case.runtime,case.authority,case.job,case.recipe,lease=lease)
    record=prep.record
    assert record['status']=='reserved' and record['lease']==lease.to_dict()
    assert prep._header['cutoff']==case.cutoff.isoformat()
    assert case.cutoff!=case.cutoff.normalize().replace(day=1)
    assert case.authority.status(record['actual_month'])['counts']['solve']==1
    snapshot=prep.snapshot();snapshot.verify()
    assert snapshot.record['training']['train_start']==(case.cutoff-pd.Timedelta(days=90)).isoformat()
    assert prep.record['status']=='reserved' and prep.record['lease']==lease.to_dict()
    with pytest.raises(ValueError,match='untouched planned'):
        ForwardPreparation.begin(case.runtime,case.authority,case.job,case.recipe)
    assert case.authority.status(record['actual_month'])['counts']['solve']==1


def test_next_elapsed_slot_uses_same_anchor_and_independent_work_key(case):
    case.clock.now=case.cutoff+pd.Timedelta(days=15)
    cutoff=case.runtime.lifecycle.due_fit_cutoff()
    job=case.runtime.lifecycle.plan_fit(cutoff,'A0',case.runtime.recipe_for(case.recipe))
    prep=ForwardPreparation.begin(case.runtime,case.authority,job,case.recipe)
    assert cutoff==case.cutoff+pd.Timedelta(days=15)
    assert prep._header['cutoff']==cutoff.isoformat() and prep.record['lease']['lease_seconds']==300
    assert job!=case.job and case.runtime.lifecycle._cadence_anchor()==case.cutoff


def test_off_grid_job_tampering_cannot_buy_any_preparation(case):
    # Deliberate corrupted SQL cannot bypass its original hashed fit event.
    with case.runtime.lifecycle.db:
        case.runtime.lifecycle.db.execute('UPDATE jobs SET cutoff=? WHERE id=?',
            ((case.cutoff+pd.Timedelta(days=1)).isoformat(),case.job))
    with pytest.raises(ValueError):
        ForwardPreparation.begin(case.runtime,case.authority,case.job,case.recipe)
    assert case.authority.verify()['reservations']==0
