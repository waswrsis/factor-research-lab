"""Elapsed-cadence engineering checks on real SQLite stores and causal kernels.

Synthetic prices and controlled test clocks are explicitly not forward evidence.
"""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factorlab.protocol import fingerprint
from factorlab.discovery.v7_lifecycle import Lifecycle, LifecyclePlan, canonical
from factorlab.discovery.v7_forward import ForwardConfig, ForwardRuntime
from factorlab.discovery.rolling import RollingSpec, build_rolling_data
from factorlab.discovery.known_linear_reference import FrozenLinearRecipe, fit_known_linear
from factorlab.discovery.local_mutation import MonthlyBudget
from factorlab.discovery.neural_forward_ticket import _LifecycleReader

HERE = Path(__file__).resolve().parents[1]


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    now = [pd.Timestamp('2026-10-10T06:34:39.609948Z')]
    plan = LifecyclePlan('synthetic-grid', training_window_days=90, refit_interval_days=15)
    value = Lifecycle(tmp_path/'lifecycle', plan, clock=lambda: now[0])
    yield value,now
    value.close()


def test_actual_first_anchor_is_not_calendar_month_start(store):
    lifecycle,now=store
    assert lifecycle.due_fit_cutoff()==now[0]
    assert lifecycle.verify()['jobs']==0


def test_first_plan_requires_original_anchor(store):
    lifecycle,now=store
    with pytest.raises(ValueError,match='anchor required'):
        lifecycle.plan_fit(now[0],'A0',{'pipeline_id':'test-only'})
    assert lifecycle.db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]==0


def test_two_fit_slots_share_same_plan_and_do_not_reset_on_calendar_month(store):
    lifecycle,now=store
    anchor=lifecycle.due_fit_cutoff()
    first=lifecycle.plan_fit(anchor,'A0',{'pipeline_id':'test-only'})
    now[0]+=pd.Timedelta(days=31)
    cutoff=lifecycle.due_fit_cutoff()
    assert cutoff==anchor+pd.Timedelta(days=30)
    second=lifecycle.plan_fit(cutoff,'A0',{'pipeline_id':'test-only'})
    assert first!=second
    assert lifecycle.verify()['jobs']==2
    assert lifecycle.db.execute('SELECT COUNT(*) FROM operations WHERE json_extract(record,\'$.kind\')=?',
                                ('fit_cadence_anchored',)).fetchone()[0]==1


def test_reopen_retains_anchor_instead_of_reanchoring(store):
    lifecycle,now=store
    anchor=lifecycle.due_fit_cutoff()
    now[0]+=pd.Timedelta(days=16)
    other=Lifecycle(lifecycle.directory,lifecycle.plan,clock=lambda:now[0])
    try:
        assert other.due_fit_cutoff()==anchor+pd.Timedelta(days=15)
        assert other._cadence_anchor()==anchor
        other.verify()
    finally:other.close()


def test_wrong_grid_is_rejected_before_job_write(store):
    lifecycle,now=store
    anchor=lifecycle.due_fit_cutoff()
    for value in (anchor-pd.Timedelta(seconds=1),anchor+pd.Timedelta(days=14),
                  anchor+pd.Timedelta(days=15,seconds=1)):
        with pytest.raises(ValueError,match='15-day grid'):
            lifecycle.plan_fit(value,'A0',{'pipeline_id':'test-only'})
    assert lifecycle.verify()['jobs']==0


def test_future_valid_grid_cannot_be_fitted(store):
    lifecycle,now=store
    anchor=lifecycle.due_fit_cutoff()
    with pytest.raises(ValueError,match='snapshot'):
        lifecycle.plan_fit(anchor+pd.Timedelta(days=15),'A0',{'pipeline_id':'test-only'})


def test_failure_does_not_grant_replacement_job_or_new_anchor(store):
    lifecycle,now=store
    cutoff=lifecycle.due_fit_cutoff()
    job=lifecycle.plan_fit(cutoff,'A0',{'pipeline_id':'test-only'})
    lifecycle.fail_fit(job,'data_incomplete')
    now[0]+=pd.Timedelta(days=1)
    assert lifecycle.plan_fit(cutoff,'A0',{'pipeline_id':'test-only'})==job
    assert lifecycle.db.execute('SELECT failure FROM jobs WHERE id=?',(job,)).fetchone()[0]=='data_incomplete'
    assert lifecycle.due_fit_cutoff()==cutoff
    assert lifecycle.verify()['jobs']==1


def test_duplicate_anchor_rejected_by_full_and_readonly_verifiers(store):
    lifecycle,_=store
    lifecycle.due_fit_cutoff()
    original=json.loads(lifecycle.db.execute('SELECT record FROM operations ORDER BY seq LIMIT 1').fetchone()[0])
    with lifecycle.db:lifecycle._event('fit_cadence_anchored',original['detail'])
    with pytest.raises(ValueError,match='anchor required'):lifecycle.verify()
    reader=_LifecycleReader(lifecycle.directory)
    try:
        with pytest.raises(ValueError,match='anchor required'):reader.verify()
    finally:reader.close()


def test_changed_anchor_plan_rejected_even_with_recomputed_chain(store):
    lifecycle,_=store
    lifecycle.due_fit_cutoff()
    event=json.loads(lifecycle.db.execute('SELECT record FROM operations').fetchone()[0])
    event['detail']['plan_id']='0'*64
    with lifecycle.db:
        lifecycle.db.execute('UPDATE operations SET record=?,hash=?',(canonical(event),fingerprint(event)))
    with pytest.raises(ValueError,match='anchor changed'):lifecycle.verify()


def test_naive_anchor_rejected_even_with_recomputed_chain(store):
    lifecycle,_=store
    lifecycle.due_fit_cutoff()
    event=json.loads(lifecycle.db.execute('SELECT record FROM operations').fetchone()[0])
    event['detail']['anchor']='2026-10-10T06:34:39'
    with lifecycle.db:
        lifecycle.db.execute('UPDATE operations SET record=?,hash=?',(canonical(event),fingerprint(event)))
    with pytest.raises(ValueError,match='aware'):lifecycle.verify()


def test_wrong_grid_event_and_index_rejected_on_readonly_reload(store):
    lifecycle,_=store
    cutoff=lifecycle.due_fit_cutoff()
    job=lifecycle.plan_fit(cutoff,'A0',{'pipeline_id':'test-only'})
    row=lifecycle.db.execute('SELECT seq,record FROM operations WHERE seq=2').fetchone()
    event=json.loads(row[1]);event['detail']['cutoff']=(cutoff-pd.Timedelta(seconds=1)).isoformat()
    with lifecycle.db:
        lifecycle.db.execute('UPDATE operations SET record=?,hash=? WHERE seq=?',
                             (canonical(event),fingerprint(event),row[0]))
        lifecycle.db.execute('UPDATE jobs SET cutoff=? WHERE id=?',(event['detail']['cutoff'],job))
    reader=_LifecycleReader(lifecycle.directory)
    try:
        with pytest.raises(ValueError,match='15-day grid'):reader.verify()
    finally:reader.close()


def test_legacy_plan_hash_is_unchanged():
    plan=LifecyclePlan('legacy')
    fields=asdict(plan);fields.pop('training_window_days');fields.pop('refit_interval_days')
    assert plan.digest==fingerprint({'version':'v7-lifecycle-v1',**fields})


@pytest.mark.parametrize('values',[(90,None),(None,15),(90,1),(True,15),(90,True)])
def test_period_pair_is_atomic_and_exact(values):
    kwargs=dict(training_window_days=values[0],refit_interval_days=values[1])
    with pytest.raises(ValueError):LifecyclePlan('bad',**kwargs)
    with pytest.raises(ValueError):ForwardConfig(**kwargs)


def test_runtime_rejects_different_lifecycle_cadence_before_binding(tmp_path,monkeypatch):
    from factorlab.discovery.v7_feed import ReceiptArchive
    monkeypatch.setattr('factorlab.protocol.ROOT',tmp_path)
    feed=ReceiptArchive(tmp_path/'feed',('BTCUSDT',));identity=feed.identity;feed.close()
    with pytest.raises(ValueError,match='actual feed'):
        ForwardRuntime(tmp_path/'run',tmp_path/'feed',
            config=ForwardConfig(training_window_days=90,refit_interval_days=15),
            lifecycle_plan=LifecyclePlan(identity))
    assert not (tmp_path/'run'/'forward.json').exists()


@pytest.fixture
def data():
    times=pd.date_range('2026-05-20',periods=4300,freq='h',tz='UTC')
    increments=.0003*np.sin(np.arange(len(times))/19)+.001*np.cos(np.arange(len(times))/43)
    price=30000*np.exp(np.cumsum(increments))
    bars=pd.DataFrame({'symbol':'BTCUSDT','open_time':times,
        'available_at':times+pd.Timedelta(hours=1,seconds=5),
        'open':price,'close':price,'high':price*1.001,'low':price*.999,'volume':10.})
    bars.attrs={'availability_mode':'assumed_historical_delay'}
    return build_rolling_data(bars,RollingSpec(window_days=90,availability_delay_seconds=5))


def test_linear_cold_fit_and_reload_use_actual_90day_start(data):
    cutoff=pd.Timestamp('2026-10-10T06:34:39.609948Z')
    recipe=FrozenLinearRecipe('K0',data.spec)
    budget=MonthlyBudget('2026-10')
    predictor=fit_known_linear(data,cutoff,recipe,budget=budget)
    restored=type(predictor).from_dict(predictor.to_dict())
    assert restored.digest==predictor.digest
    assert predictor.receipt['training']['train_start']==(cutoff-pd.Timedelta(days=90)).isoformat()
    assert predictor.receipt['training']['cutoff']==cutoff.isoformat()
    assert predictor.receipt['training']['rows']==int(data.train_mask(cutoff).sum())
    assert budget.counts['solve']==1


@pytest.fixture
def runtime(tmp_path,monkeypatch):
    from shutil import copy2
    from factorlab.discovery.evidence import source_paths
    from factorlab.discovery.v7_feed import ReceiptArchive
    # Copy every current staged executable dependency, not a two-file source stub.
    for source in source_paths(HERE):
        target=tmp_path/source.relative_to(HERE)
        target.parent.mkdir(parents=True,exist_ok=True);copy2(source,target)
    monkeypatch.setattr('factorlab.protocol.ROOT',tmp_path)
    now=[pd.Timestamp('2026-10-10T06:34:39.609948Z')]
    config=ForwardConfig(training_window_days=90,refit_interval_days=15)
    value=ForwardRuntime(tmp_path/'run',tmp_path/'feed',config=config,clock=lambda:now[0])
    yield value,now,tmp_path
    value.close()


def synthetic_seed(value,root):
    times=pd.date_range('2026-05-20',periods=3250,freq='h',tz='UTC')
    prices=30000*np.exp(np.cumsum(.001*np.sin(np.arange(len(times))/31)))
    bars=pd.DataFrame({'symbol':'BTCUSDT','open_time':times,'open':prices,'close':prices,
                       'high':prices*1.002,'low':prices*.998,'volume':10.})
    path=root/'synthetic-only.parquet';bars.to_parquet(path,index=False)
    value.feed.import_seed(path)


def test_daily_refresh_waits_for_import_before_first_anchor(runtime):
    value,_,root=runtime
    from factorlab.discovery.v7_feed import SnapshotUnavailable
    with pytest.raises(SnapshotUnavailable):value.refresh_training_view()
    events=[json.loads(r[0]) for r in value.lifecycle.db.execute('SELECT record FROM operations')]
    assert not any(r['kind']=='fit_cadence_anchored' for r in events)
    synthetic_seed(value,root)
    view=value.refresh_training_view()
    assert value.lifecycle._cadence_anchor()==pd.Timestamp(view['observed_at'])
    assert view['fit_permission'] is False and view['search_exposure'] is False
    assert view['training']['train_start']=='2026-07-12T06:34:39.609948+00:00'
    assert view['training']['cutoff']==view['observed_at']
    assert value.lifecycle.verify()['jobs']==0


def test_refresh_is_once_per_elapsed_day_and_independent_of_fit_grid(runtime):
    value,now,root=runtime
    synthetic_seed(value,root)
    first=value.refresh_training_view()
    fit=value.lifecycle.due_fit_cutoff()
    now[0]+=pd.Timedelta(hours=23)
    assert value.refresh_training_view()==first
    now[0]+=pd.Timedelta(hours=2)
    second=value.refresh_training_view()
    assert second['refresh_slot']!=(first['refresh_slot'])
    assert value.lifecycle.due_fit_cutoff()==fit
    assert second['observed_at']==now[0].isoformat()
    assert value.lifecycle.verify()['jobs']==0


def test_refresh_does_not_backfill_missed_days(runtime):
    value,now,root=runtime
    synthetic_seed(value,root);value.refresh_training_view()
    now[0]+=pd.Timedelta(days=4,hours=2)
    value.refresh_training_view()
    records=[json.loads(r[0]) for r in value.lifecycle.db.execute('SELECT record FROM operations')]
    assert sum(r['kind']=='training_view_refreshed' for r in records)==2
    assert value.lifecycle.verify()['jobs']==0


def test_daily_view_artifact_loss_blocks_full_and_readonly_reload(runtime):
    value,_,root=runtime
    synthetic_seed(value,root);view=value.refresh_training_view()
    path=value.lifecycle.directory/'training_views'/(fingerprint(view['refresh_slot'])+'.json')
    original=path.read_bytes()
    try:
        path.write_bytes(b'{}')
        with pytest.raises(ValueError,match='artifact'):value.lifecycle.verify()
        reader=_LifecycleReader(value.directory)
        try:
            with pytest.raises(ValueError,match='artifact'):reader.verify()
        finally:reader.close()
    finally:path.write_bytes(original)


def test_complete_neural_kernel_and_reload_use_90day_rule(data):
    from factorlab.discovery.neural_forward import FrozenNeuralRecipe,NeuralForwardModel,_fit_neural_forward_cold
    from factorlab.discovery.nonlinear_channel import ChannelBudget,ResidualConfig,StructureSpec
    recipe=FrozenNeuralRecipe(rolling_spec=data.spec,
        config=ResidualConfig(enabled=True,structures=(StructureSpec(4,2,1),),seeds=(17,29),epochs=1))
    budget=ChannelBudget()
    model=_fit_neural_forward_cold(data,'2026-10-10T06:34:39.609948Z',recipe,budget)
    assert model.receipt['train_start']=='2026-07-12T06:34:39.609948+00:00'
    assert model.receipt['training']['rolling_hash']==data.spec.digest
    restored=NeuralForwardModel.from_dict(model.to_dict())
    assert restored.digest==model.digest
    assert restored.recipe.to_dict()['rules']['fit']=='cold_continuous_90_days_at_original_15_day_cutoff'
    assert budget.counts['neural_fit']==2 and budget.counts['epoch']==2
    # This numerical engineering fit still cannot act as a serving certificate.
    assert model.receipt['test_only'] is True and model.receipt['runtime_admitted'] is False


def test_fifteen_day_slots_share_actual_wallet_and_failures_never_refund(runtime):
    from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority,BudgetExhausted
    value,now,root=runtime
    synthetic_seed(value,root);value.refresh_training_view()
    authority=MonthlyWorkAuthority(root/'wallet',run_id='cadence-engineering',task='volatility',
        universe=('BTCUSDT',),limits={'solve':2},monthly_seconds=2)
    cutoff=value.lifecycle.due_fit_cutoff()
    first=authority.reserve('grid:'+cutoff.isoformat(),counts={'solve':1},binding={'cutoff':cutoff.isoformat()},
                            lease_seconds=1)
    authority.fail_consumed(first['reservation_id'],'engineering_failure')
    # A diagnostic retry name cannot create a new lease or restore credits.
    again=authority.reserve(first['work_key'],counts={'solve':1},binding=first['binding'],
                            lease_seconds=1,retry_id='retry-label')
    assert again['lease']==first['lease'] and again['status']=='failed_consumed'
    now[0]+=pd.Timedelta(days=15)
    second_cutoff=value.lifecycle.due_fit_cutoff()
    second=authority.reserve('grid:'+second_cutoff.isoformat(),counts={'solve':1},
        binding={'cutoff':second_cutoff.isoformat()},lease_seconds=1)
    assert first['actual_month']==second['actual_month']
    status=authority.status(first['actual_month'])
    assert status['counts']['solve']==2 and status['reserved_seconds']==2
    with pytest.raises(BudgetExhausted):
        authority.reserve('third-grid',counts={'solve':1},binding={'cutoff':'unusable'},lease_seconds=1)
    assert authority.status(first['actual_month'])['counts']['solve']==2


def test_search_context_keeps_elapsed_cutoff_and_calendar_budget_identity():
    from factorlab.discovery.v7_memory import search_exposure_context
    cutoff=pd.Timestamp('2026-10-10T06:34:39.609948Z')
    context=search_exposure_context('engineering-data',cutoff,('BTCUSDT',),spec=RollingSpec(window_days=90))
    assert context['end']==cutoff.isoformat()
    assert context['batch']=='2026-10'
    assert context['start']==(cutoff-pd.Timedelta(days=180,hours=1234)).isoformat()


def test_search_readouts_are_three_30day_blocks_with_90day_cold_training():
    from factorlab.discovery.rolling import development_folds
    spec=RollingSpec(window_days=90)
    cutoff=pd.Timestamp('2024-03-10T12:00:00Z')
    folds=development_folds(cutoff,spec=spec)
    assert len(folds)==3 and folds[-1]['score_end']==cutoff.isoformat()
    for fold in folds:
        start,end=pd.Timestamp(fold['fit_cutoff']),pd.Timestamp(fold['score_end'])
        assert end-start==pd.Timedelta(days=30)
        assert spec.window(start)[1]-spec.window(start)[0]==pd.Timedelta(days=90)


def test_elapsed_claim_policy_identity_does_not_shorten_six_month_confirmation():
    from factorlab.discovery.claim_protocol import FrozenClaimEndpoint
    runtime=ForwardRuntime.__new__(ForwardRuntime);runtime.spec=RollingSpec(window_days=90)
    k0=runtime.recipe_for()
    endpoint=FrozenClaimEndpoint.algorithm('2026-11-01T00:00:00Z',k0)
    assert endpoint.candidate['version']=='v7-adaptive-elapsed90-refit15-v3'
    assert endpoint.candidate['refit']=='cold_continuous_90_days_on_original_15_day_grid'
    assert endpoint.end=='2027-05-01T00:00:00+00:00'
    assert FrozenClaimEndpoint.from_dict(endpoint.to_dict()).digest==endpoint.digest


def test_legacy_claim_policy_cannot_be_attached_to_elapsed_reference():
    from factorlab.discovery.claim_protocol import FrozenClaimEndpoint
    from factorlab.discovery.typed_policy_confirmation import POLICY,TRIAL_STREAM
    runtime=ForwardRuntime.__new__(ForwardRuntime);runtime.spec=RollingSpec(window_days=90)
    k0=runtime.recipe_for()
    with pytest.raises(ValueError,match='registered A0 policy'):
        FrozenClaimEndpoint('algorithm','2026-11-01T00:00:00Z',
            {**POLICY,'initial_pipeline':k0},k0,TRIAL_STREAM,'K0')


def test_same_month_policy_selections_have_distinct_fifteen_day_paths(store):
    from factorlab.discovery.typed_policy_confirmation import _selection_paths
    lifecycle,_=store
    first=lifecycle.due_fit_cutoff();second=first+pd.Timedelta(days=15)
    assert first.month==second.month
    assert _selection_paths(lifecycle,first)!=_selection_paths(lifecycle,second)
