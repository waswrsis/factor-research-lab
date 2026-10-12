"""Real six-month claim/ledger/adoption; artificial forecasts and selector seam.
No fitted neural/provider/Linux/full-calibration evidence is claimed.
"""
import json
import os
import numpy as np
import pandas as pd
import pytest
from factorlab.protocol import fingerprint
from factorlab.discovery import typed_policy_confirmation as policy
from factorlab.discovery.claim_protocol import ClaimProtocol, FrozenClaimEndpoint
from factorlab.discovery.confirmatory import ConfirmationRule
from factorlab.discovery.convex_eval import fit_ridge
from factorlab.discovery.joint_pool import FrozenPool
from factorlab.discovery.v7_forward import ForwardRuntime
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
from factorlab.discovery.neural_forward_ticket import _bind_authority
from factorlab.discovery.calibration_runtime_contract import runtime_calibration_contract
from factorlab.discovery.calibration_acceptance import AcceptanceCell, AcceptancePlan
from factorlab.discovery.full_calibration_registry import FullTrialPlan, FullTrialRegistry


def test_real_algorithm_slot_confirmation_next_month_and_tamper(tmp_path_factory, monkeypatch):
    root = tmp_path_factory.mktemp('pc')
    (root/'factorlab').mkdir(); (root/'factorlab/source.py').write_text('engineering=1')
    (root/'pyproject.toml').write_text('[project]'); (root/'uv.lock').write_text('version=1')
    monkeypatch.setattr('factorlab.protocol.ROOT', root)
    clock=[pd.Timestamp('2026-12-31T12:00:00Z')]
    profile=ForwardRuntime(root/'p',root/'pf',clock=lambda:clock[0])
    wallet=MonthlyWorkAuthority(root/'w',run_id='profile',task='volatility',universe=['BTCUSDT'],partition_policy='standard')
    _bind_authority(profile,wallet)
    registry=FullTrialRegistry(root/'r',FullTrialPlan(
        AcceptancePlan((AcceptanceCell('test',fingerprint('law'),1000),)),runtime_calibration_contract(profile.lifecycle),
        {'backend':'codex','model_requested':'gpt-6-astra','reasoning_requested':'high','context_policy':'fresh_codex_per_trial_v1'},
        {k+'_digest':fingerprint(k) for k in ('design','validation','oracle')}))
    paths=registry.paths('test',0); runtime=ForwardRuntime(paths['runtime'],paths['feed'],clock=lambda:clock[0])
    trial=registry.identity_for('test',0)
    authority=MonthlyWorkAuthority(paths['authority'],run_id=fingerprint({'registry':registry.digest,'trial':trial['trial_id']}),
        task='volatility',universe=['BTCUSDT'],partition_policy='standard')
    registry.enroll(runtime,authority,'test',0)
    lc=runtime.lifecycle; k0=runtime.recipe_for(FrozenPool()); candidate=runtime.recipe_for(FrozenPool(version='artificial_candidate'))
    rule=ConfirmationRule(bootstrap_samples=1000)
    protocol=ClaimProtocol.freeze(lc,[FrozenClaimEndpoint.library('2027-01-01T00:00:00Z',k0,
        {'pipeline_id':fingerprint('benchmark'),'benchmark':'artificial'},rule=rule),
        FrozenClaimEndpoint.algorithm('2027-01-01T00:00:00Z',k0,rule=rule)])
    try:
        assert protocol.endpoints['algorithm'].candidate_stream == 'A0_policy_trial'
        assert protocol.endpoints['algorithm'].end == '2027-07-01T00:00:00+00:00'
        identities=[json.loads(r[0]) for r in lc.exposure.db.execute('SELECT identity FROM confirmations')]
        assert {i['claim_slot'] for i in identities} == {'algorithm','library_skill'}
        assert all(i['alpha']==.01 for i in identities)
        clock[0]=pd.Timestamp('2027-01-01T00:00:00Z')
        with policy._original_reader(lc) as reader:
            assert policy._selection(reader,clock[0],k0)[0]==k0
        # Canonical selector itself is separately covered by typed V tests. This
        # explicit seam isolates the new whole-policy claim/adoption contract.
        monkeypatch.setattr(policy,'_selection',lambda *a:(candidate,{'private_artificial_selector':True}))
        assert policy.adopted_pipeline(lc,clock[0],k0)==k0
        assert policy.candidate_pipeline(lc,clock[0],k0)==candidate
        model=fit_ridge(np.arange(60).reshape(-1,1),np.linspace(.001,.03,60))
        monkeypatch.setattr(lc,'_work_authority_bound',lambda:False)  # artificial model admission seam only
        for stream,pipeline,value in [('K0',k0,.05),(policy.TRIAL_STREAM,candidate,.09)]:
            clock[0]=pd.Timestamp('2027-01-01T00:00:00Z')
            job=lc.plan_fit(clock[0],stream,pipeline)
            lc.advance_fit(job,'snapshot_sealed',output='artificial');lc.advance_fit(job,'features_ready',output='artificial')
            # Explicit artificial model publication seam; no admitted cold fit claimed.
            with monkeypatch.context() as fixture_patch:
                fixture_patch.setattr(lc,'_work_authority_bound',lambda:False)
                model_id=lc.seal_model(job,model)
                lc.activate(job,clock[0])
            for hour in pd.date_range('2027-01-01','2027-07-01',inclusive='left',freq='h',tz='UTC'):
                origin=hour+pd.Timedelta(seconds=1);window=origin.ceil('h'); maturity=window+pd.Timedelta(hours=12)
                forecast={'value':value,'received_at':origin.isoformat(),'forecast_window_start':window.isoformat(),
                    'issued_at':origin.isoformat(),'model_id':model_id,'pipeline_id':pipeline['pipeline_id'],
                    'availability':'explicit_artificial_pre_recorded_fixture'}
                lc.forecasts.put(run=lc.plan.digest,stream=stream,origin=origin,symbol='BTCUSDT',task='volatility',
                    model_id=model_id,pipeline_id=pipeline['pipeline_id'],prediction=forecast)
                clock[0]=maturity
                lc.settle(stream=stream,origin=origin,symbol='BTCUSDT',task='volatility',label=.1,maturity=maturity,received_at=maturity)
        clock[0]=pd.Timestamp(protocol.endpoints['algorithm'].scoring_cutoff)
        score=protocol.finalize('algorithm')
        assert score['result']['quarantined_decision']['accepted']
        assert score['result']['supported'] is False and score['formal_eligible'] is False
        assert policy.adopted_pipeline(lc,'2027-07-01T00:00:00Z',k0)==k0
        clock[0]=pd.Timestamp('2027-08-01T00:00:00Z')
        assert policy.adopted_pipeline(lc,clock[0],k0)==candidate
        adopted=policy.canonical_adoption(lc,clock[0],candidate)
        def no_reentry(*args,**kwargs):
            raise AssertionError('historical admission must not recurse into model or selector work')
        with monkeypatch.context() as guard:
            guard.setattr(lc,'verify',no_reentry)
            guard.setattr(policy,'_selection',no_reentry)
            assert policy.canonical_adoption(lc,clock[0],candidate)==adopted
        assert adopted['claim_slot']=='algorithm' and adopted['no_individual_neural_claim']
        assert policy.adopted_pipeline(lc,clock[0],k0)==candidate
        assert lc.db.execute('SELECT COUNT(*) FROM batches').fetchone()[0]==0
        assert lc.exposure.db.execute('SELECT COUNT(*) FROM confirmations').fetchone()[0]==2
        path=protocol.directory/'algorithm/score.json';raw=path.read_bytes();path.unlink()
        with pytest.raises(ValueError,match='publication missing'):
            policy.canonical_adoption(lc,clock[0],candidate)
        path.write_bytes(raw)
        late=(clock[0]+pd.Timedelta(seconds=1)).value;os.utime(path,ns=(late,late))
        assert policy.canonical_adoption(lc,clock[0],candidate) is None
        os.utime(path,None)
        assert policy.canonical_adoption(lc,clock[0],candidate)==adopted
        inputs=protocol.directory/'algorithm/inputs.json';inputs.write_text('{}')
        with pytest.raises(ValueError,match='paired-loss'):
            policy.canonical_adoption(lc,clock[0],candidate)
        (root/'numeric.json').write_text(json.dumps({'score':score,'adoption':adopted,'scope':'artificial selector/forecasts, actual claim/ledger/statistic; no neural/provider fit'},indent=2))
    finally:
        runtime.close();profile.close()


def test_old_policy_cannot_migrate_by_relabeling():
    old={'version':'v7-adaptive-algorithm-v1','selection':'latest_ordinal_supported_C_completed_before_month_start',
         'refit':'cold_three_calendar_months','no_selection':'retain_previous_or_initial_K0',
         'integrity_failure':'fallback_K0','development_retirement':'does_not_change_A0',
         'initial_pipeline':{'pipeline_id':'0'*64,'kind':'toy'}}
    with pytest.raises(ValueError,match='complete registered'):
        FrozenClaimEndpoint('algorithm','2027-01-01T00:00:00Z',old,old['initial_pipeline'],'A0','K0')
