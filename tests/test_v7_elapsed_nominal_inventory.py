"""Engineering boundary checks; no market, nominal truth or admission proof."""
from copy import deepcopy
from types import SimpleNamespace
from dataclasses import asdict
import pandas as pd
import pytest
from test_v7_elapsed_calibration_cadence import runtime
from factorlab.discovery.rolling import RollingSpec
from factorlab.discovery.rolling_cadence import RollingCadence
from factorlab.discovery.v7_forward import ForwardRuntime
from factorlab.discovery.known_linear_reference import FrozenLinearRecipe
from factorlab.discovery.joint_pool import FrozenPool
from factorlab.discovery.nominal_factor_census import _schedule
from factorlab.discovery.nominal_full_scope_preparation import _configuration
from factorlab.discovery.nominal_algorithm_warmboot import _preendpoint_cutoffs,_allowed_cutoff,ELAPSED_VERSION
from factorlab.discovery.nominal_capacity_preflight import _derive
from factorlab.discovery.nominal_cold_conformance import selected_cutoff
from factorlab.discovery.calibration_runtime_contract import runtime_calibration_contract
from test_v7_run_factor_pool import source_root
from factorlab.discovery.v7_forward import ForwardConfig
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
from factorlab.discovery.monthly_partition_policy import frozen_policy
from factorlab.discovery.neural_forward_ticket import _bind_authority


@pytest.fixture
def standard_runtime(tmp_path,monkeypatch):
    source_root(tmp_path,monkeypatch)
    value=ForwardRuntime(tmp_path/'runtime',tmp_path/'feed',
        config=ForwardConfig(training_window_days=90,refit_interval_days=15))
    wallet=MonthlyWorkAuthority(tmp_path/'wallet',run_id='elapsed-fullscope',
        task='volatility',universe=['BTCUSDT'],partition_policy=frozen_policy('standard'))
    _bind_authority(value,wallet)
    try:yield value,wallet
    finally:value.close()


@pytest.mark.parametrize('phase',[0,1,14,15,29,30])
def test_half_open_inventory_preserves_original_phase_and_counts(phase):
    cadence=RollingCadence();start=pd.Timestamp('2026-05-01T00:00:00Z')
    anchor=start-pd.Timedelta(days=31)+pd.Timedelta(days=phase,seconds=5)
    end=start+pd.DateOffset(months=6)
    jobs=cadence.fit_cutoffs(anchor,start,end,include_current_cycle=False)
    assert len(jobs)>=int((end-start)//pd.Timedelta(days=15))
    assert all(start<=t<end for t in jobs)
    assert all(cadence.slot_identity(anchor,t)['anchor']==anchor.isoformat() for t in jobs)
    assert all(b-a==pd.Timedelta(days=15) for a,b in zip(jobs,jobs[1:]))
    assert cadence.fit_cutoffs(anchor,anchor,anchor+pd.Timedelta(days=15))==(anchor,)
    assert cadence.fit_cutoffs(anchor,anchor+pd.Timedelta(days=15),anchor+pd.Timedelta(days=30))==(anchor+pd.Timedelta(days=15),)


def test_pure_inventory_late_anchor_and_current_cycle_are_explicit():
    cadence=RollingCadence();anchor=pd.Timestamp('2026-10-10T06:34:39.609948Z')
    assert cadence.fit_cutoffs(anchor,anchor-pd.Timedelta(days=2),anchor)==()
    assert cadence.fit_cutoffs(anchor,anchor+pd.Timedelta(days=1),anchor+pd.Timedelta(days=15))==(anchor,)
    assert cadence.fit_cutoffs(anchor,anchor+pd.Timedelta(days=1),anchor+pd.Timedelta(days=15),include_current_cycle=False)==()
    with pytest.raises(ValueError,match='Nonempty'):
        cadence.fit_cutoffs(anchor,anchor,anchor)
    with pytest.raises(ValueError,match='bounded'):
        from datetime import timedelta
        old=pd.Timestamp('1900-01-01T00:00:00Z')
        cadence.fit_cutoffs(old,old,pd.Timestamp(old.to_pydatetime()+timedelta(days=15*8193)))


def test_original_runtime_configuration_binds_both_elapsed_fields(standard_runtime):
    value,wallet=standard_runtime;scope=runtime_calibration_contract(value.lifecycle)
    before=wallet.verify();config,lifecycle,work=_configuration(scope)
    assert config.training_window_days==90 and config.refit_interval_days==15
    assert lifecycle['training_window_days']==90 and lifecycle['refit_interval_days']==15
    assert work==scope['work_budget'] and wallet.verify()==before
    bad=deepcopy(scope);bad['lifecycle_policy']['training_window_days']=None
    bad['lifecycle_policy']['refit_interval_days']=None
    with pytest.raises(ValueError,match='rolling/universe'):_configuration(bad)


def test_preendpoint_inventory_uses_original_durable_lifecycle_anchor(runtime):
    value,wallet=runtime
    # Explicit artificial test clock anchor, not an actual data-ready assertion.
    anchor=value.lifecycle.due_fit_cutoff();end=anchor+pd.Timedelta(days=31)
    warm={'version':ELAPSED_VERSION,'endpoint_start':end.isoformat()}
    before=wallet.verify()
    expected=tuple((anchor+pd.Timedelta(days=n)).isoformat() for n in (0,15,30))
    assert _preendpoint_cutoffs(value.lifecycle,warm)==expected
    assert all(_allowed_cutoff(value.lifecycle,warm,c) for c in expected)
    assert _allowed_cutoff(value.lifecycle,warm,end.isoformat()) is False
    with pytest.raises(ValueError,match='15-day grid'):
        _allowed_cutoff(value.lifecycle,warm,(anchor+pd.Timedelta(days=1)).isoformat())
    assert wallet.verify()==before and before['reservations']==0
    assert value.lifecycle.db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]==0


def test_census_retains_every_hour_and_original_anchor_with_role_ready_delays():
    spec=SimpleNamespace(start='2026-05-01T00:00:00Z',end='2026-11-01T00:00:00Z')
    rolling=RollingSpec(window_days=90);anchor=pd.Timestamp('2026-04-02T06:34:39.609948Z')
    schedule=_schedule(spec,rolling_spec=rolling,cadence_anchor=anchor)
    assert schedule['hours']==184*24
    assert schedule['fit_schedule']['anchor']==anchor.isoformat()
    cutoffs=list(map(pd.Timestamp,schedule['cutoffs']))
    assert pd.Timestamp(schedule['raw_start'])==cutoffs[0]-pd.Timedelta(days=90,hours=1234)
    for i,role in enumerate(('reference','candidate')):
        for c in cutoffs:
            assert pd.Timestamp(schedule['ready'][role][c.isoformat()])==c+pd.Timedelta(hours=i+1,seconds=5)
        selected=[selected_cutoff(schedule,role,t) for t in pd.date_range(spec.start,spec.end,freq='h',inclusive='left')+pd.Timedelta(seconds=5)]
        assert len(selected)==schedule['hours'] and set(selected)<=set(schedule['cutoffs'])
    with pytest.raises(ValueError,match='anchor required'):_schedule(spec,rolling_spec=rolling)


def capacity_inputs():
    rolling=RollingSpec(window_days=90);runtime=ForwardRuntime.__new__(ForwardRuntime);runtime.spec=rolling
    har=runtime.recipe_for(FrozenLinearRecipe('HAR_cal',rolling));k0=runtime.recipe_for(FrozenPool())
    start=pd.Timestamp('2026-05-01T00:00:00Z');end=start+pd.DateOffset(months=6)
    due=tuple(t.isoformat() for t in pd.date_range(start-pd.DateOffset(months=1),end+pd.Timedelta(hours=13),freq='h')+pd.Timedelta(seconds=5))
    endpoints=[{'slot':'library_skill','start':start.isoformat(),'end':end.isoformat(),
        'reference_stream':'HAR','candidate_stream':'K0','reference':har,'candidate':k0},
        {'slot':'algorithm','start':start.isoformat(),'end':end.isoformat(),
        'reference_stream':'K0','candidate_stream':'A0_policy_trial','reference':k0,'candidate':{'policy':'adaptive'}}]
    warm={'version':ELAPSED_VERSION,'cutoff':None,'endpoint_start':start.isoformat(),
        'streams':{'HAR':har,'K0':k0,'A0_policy_trial':k0},
        'opportunities':[t for t in due if pd.Timestamp(t)<start],
        'fit_schedule':{'cadence_hash':RollingCadence().digest,'training_days':90,'refit_days':15,'refresh_days':1}}
    search={'slots':3,'reserved_seconds':3480,'allocated_counts':{}}
    return endpoints,due,warm,rolling,search


def test_capacity_counts_at_least_39_original_jobs_not_old_21():
    endpoints,due,warm,rolling,search=capacity_inputs()
    report=_derive(endpoints,due,warm,rolling,{'known':300,'pool':300,'neural':300},search,{},
        {'wall_seconds':7200,'max_fits_per_step':1})
    assert report['required_cold_lower_bound']['jobs']==39
    assert report['required_cold_lower_bound']['seconds']==11700
    assert report['status']=='rejected_necessary_capacity'
    assert 'required_cold_prepaid_seconds_exceed_even_two_physical_months' in report['proven_required_violations']
    assert report['jobs']is None and report['ordinary_A0_planned_jobs']is None
    assert report['anchor']=='unknown_until_original_actual_data_ready_prefix'
    assert report['same_physical_month_forward_limit']['seconds']==2820
    assert report['execution_authorized']is False and report['formal_acceptance']is False
    assert report['runtime_performance_verified']is False


def test_shorter_prospective_ceiling_cannot_claim_capacity_acceptance():
    endpoints,due,warm,rolling,search=capacity_inputs()
    report=_derive(endpoints,due,warm,rolling,{'known':40,'pool':40,'neural':300},search,{},
        {'wall_seconds':7200,'max_fits_per_step':1})
    assert report['required_cold_lower_bound']['seconds']==1560
    assert report['status']=='necessary_only_unknown' and report['proven_required_violations']==[]
    assert report['full_cost_enumerated']is False and report['execution_authorized']is False
    assert report['all_planned_model_lower_bound']is None
    assert report['search']['owner_key']=='actual_UTC_month|S'


def test_elapsed_capacity_cannot_borrow_calendar_cutoff_or_other_cadence():
    endpoints,due,warm,rolling,search=capacity_inputs()
    warm['cutoff']='2026-04-01T00:00:00Z'
    with pytest.raises(ValueError,match='elapsed warmboot'):
        _derive(endpoints,due,warm,rolling,{'known':300,'pool':300,'neural':300},search,{},
            {'wall_seconds':7200,'max_fits_per_step':1})


def test_numeric_failed_fit_keeps_full_hourly_support_and_90_day_loss(monkeypatch):
    import numpy as np
    from factorlab.discovery import nominal_factor_census as census,nominal_total_loss as loss
    from factorlab.discovery.nominal_fit_outcome import _numeric_rows
    # Arithmetic seam only: this cannot produce a paid event or certificate.
    monkeypatch.setattr(census,'_loss_module',lambda:loss)
    rolling=RollingSpec(window_days=90)
    schedule=_schedule(SimpleNamespace(start='2026-05-01T00:00:00Z',end='2026-06-01T00:00:00Z'),
        rolling_spec=rolling,cadence_anchor='2026-04-02T00:00:00Z')
    origin=pd.Timestamp('2026-05-01T00:00:05Z');cutoff=selected_cutoff(schedule,'reference',origin)
    data=SimpleNamespace(spec=rolling,frame=SimpleNamespace(metadata=pd.DataFrame([{
        'available_at':origin,'symbol':'BTCUSDT','feature_open_time':origin.floor('h')-pd.Timedelta(hours=1),
        'forecast_window_start':origin.ceil('h')}])),target=np.array([1.]),observed=np.array([True]),
        maturity=[origin.ceil('h')+pd.Timedelta(hours=12)])
    preparation=SimpleNamespace(data=data,protocol_identity={'schedule':schedule,'law':{'symbols':['BTCUSDT']}},cutoff=cutoff)
    rows=_numeric_rows(preparation,'reference')
    expected=[t for t in pd.date_range(schedule['start'],schedule['end'],freq='h',inclusive='left')+pd.Timedelta(seconds=5)
        if selected_cutoff(schedule,'reference',t)==cutoff]
    assert len(rows)==len(expected)>1
    assert all(r['loss']==2**20 and r['prediction']is None for r in rows)
    assert all(pd.Timestamp(r['train_start'])==pd.Timestamp(cutoff)-pd.Timedelta(days=90) for r in rows)
    assert rows[0]['label']==1. and all(r['label']is None for r in rows[1:])


def test_policy_readonly_reader_retains_elapsed_spec_and_does_not_mutate_wallet(standard_runtime):
    from factorlab.discovery.typed_policy_confirmation import _runtime
    original,wallet=standard_runtime;before=wallet.verify()
    with _runtime(original.lifecycle) as (restored,authority):
        assert restored.spec==original.spec and restored.spec.window_days==90
        assert restored.lifecycle is original.lifecycle
        assert restored.feed.db.execute('PRAGMA query_only').fetchone()[0]==1
        assert authority.verify()==before
    assert wallet.verify()==before


@pytest.mark.parametrize('reader',['policy','typed'])
def test_readonly_readers_reject_rehashed_calendar_substitution_before_feed_access(standard_runtime,reader):
    import json
    from factorlab.protocol import fingerprint
    from factorlab.discovery.typed_policy_confirmation import _runtime
    from factorlab.discovery.typed_cold_selection import _runtime_reader
    original,wallet=standard_runtime;before=wallet.verify()
    path=original.directory/'forward.json';content=json.loads(path.read_text('utf-8'))
    content['identity']['rolling']['window_days']=None
    content['sha256']=fingerprint(content['identity']);path.write_text(json.dumps(content),encoding='utf-8')
    context=(_runtime(original.lifecycle) if reader=='policy' else _runtime_reader(
        SimpleNamespace(paths={'runtime':original.directory,'feed':original.feed.directory})))
    with pytest.raises(ValueError,match='rolling cadence'):
        with context:raise AssertionError('Changed scope must never reach a reader body')
    assert wallet.verify()==before
