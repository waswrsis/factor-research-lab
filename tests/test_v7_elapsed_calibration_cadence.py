"""Actual component scope and pure nominal arithmetic; no truth certificate."""
from dataclasses import asdict
import json
import pandas as pd
import pytest
from test_v7_run_factor_pool import source_root
from factorlab.protocol import fingerprint
from factorlab.discovery.calibration_runtime_contract import runtime_calibration_contract
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
from factorlab.discovery.neural_forward_ticket import _bind_authority
from factorlab.discovery.v7_forward import ForwardRuntime,ForwardConfig
from factorlab.discovery.nominal_algorithm_policy import cold_window,model_visible_at,c_first_origin,ELAPSED_VERSION


@pytest.fixture
def runtime(tmp_path,monkeypatch):
    source_root(tmp_path,monkeypatch)
    value=ForwardRuntime(tmp_path/'runtime',tmp_path/'feed',
        config=ForwardConfig(training_window_days=90,refit_interval_days=15))
    wallet=MonthlyWorkAuthority(tmp_path/'wallet',run_id='elapsed-scope',
        task='volatility',universe=['BTCUSDT'])
    _bind_authority(value,wallet)
    try:yield value,wallet
    finally:value.close()


def test_actual_registered_90_15_scope_is_readonly_and_preserves_calendar_wallet(runtime):
    value,wallet=runtime;before=wallet.verify()
    result=runtime_calibration_contract(value.lifecycle)
    assert result['version']=='v7-actual-elapsed-runtime-calibration-component-v2'
    assert result['rolling']['window_days']==90
    assert result['rolling_hash']==value.spec.digest
    assert result['forward']['refit_interval_days']==15
    assert result['lifecycle_policy']['refit_interval_days']==15
    assert result['work_budget']['monthly_seconds']==7200
    assert result['work_budget']['limits']==wallet.limits
    assert result['scope']=='actual_runtime_component_only'
    assert 'formal_gate_eligible' not in result
    assert runtime_calibration_contract(value.lifecycle)==result
    assert wallet.verify()==before and before['reservations']==0
    assert value.lifecycle.db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]==0


def test_calendar_and_elapsed_runtimes_cannot_share_calibration_identity(runtime):
    value,wallet=runtime
    other=ForwardRuntime(value.directory.parent/'calendar',value.feed.directory)
    try:
        _bind_authority(other,wallet)
        first=runtime_calibration_contract(value.lifecycle)
        second=runtime_calibration_contract(other.lifecycle)
        assert first['rolling_hash']!=second['rolling_hash']
        assert fingerprint(first)!=fingerprint(second)
    finally:other.close()


def test_rehashed_manifest_cannot_claim_a_calendar_window_for_elapsed_configuration(runtime):
    value,wallet=runtime
    path=value.directory/'forward.json';content=json.loads(path.read_text('utf-8'))
    content['identity']['rolling']['window_days']=None
    content['sha256']=fingerprint(content['identity'])
    path.write_text(json.dumps(content),encoding='utf-8')
    with pytest.raises(ValueError,match='rolling/availability'):
        runtime_calibration_contract(value.lifecycle)
    assert wallet.verify()['reservations']==0


@pytest.mark.parametrize('days',[0,15,30,45,60])
def test_nominal_window_is_exactly_90_days_on_original_grid(days):
    anchor=pd.Timestamp('2026-10-10T06:34:39.609948Z')
    cutoff=anchor+pd.Timedelta(days=days)
    start,end=cold_window(cutoff.isoformat(),cadence_anchor=anchor.isoformat())
    assert pd.Timestamp(end)==cutoff
    assert pd.Timestamp(end)-pd.Timestamp(start)==pd.Timedelta(days=90)


@pytest.mark.parametrize('shift',[pd.Timedelta(days=-15),pd.Timedelta(days=14),pd.Timedelta(days=15,seconds=1)])
def test_nominal_window_cannot_move_the_original_anchor(shift):
    anchor=pd.Timestamp('2026-10-10T06:34:39.609948Z')
    with pytest.raises(ValueError,match='15-day grid'):
        cold_window((anchor+shift).isoformat(),cadence_anchor=anchor.isoformat())


def test_pure_visibility_uses_same_grid_without_granting_model_admission():
    anchor='2026-10-10T06:34:39.609948+00:00'
    kwargs={'cutoff':anchor,'cadence_anchor':anchor,'same_recipe':True,
            'completion_prefix':'2026-10-10T07:00:05+00:00'}
    assert model_visible_at(origin='2026-10-10T08:00:05+00:00',**kwargs)
    assert not model_visible_at(origin='2026-10-10T07:00:05+00:00',**kwargs)
    assert not model_visible_at(origin='2026-10-10T08:00:05+00:00',**{**kwargs,'same_recipe':False})


def test_legacy_arithmetic_and_confirmation_activation_are_preserved():
    assert cold_window('2026-11-01T00:00:00+00:00')==(
        '2026-08-01T00:00:00+00:00','2026-11-01T00:00:00+00:00')
    with pytest.raises(ValueError,match='monthly cold cutoff'):
        cold_window('2026-11-10T00:00:00+00:00')
    origin='2026-11-01T00:00:05+00:00'
    assert c_first_origin('2026-11-01T00:00:00+00:00',origin,5,policy=ELAPSED_VERSION)==origin
