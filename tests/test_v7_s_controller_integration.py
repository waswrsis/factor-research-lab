"""Core real-wallet/actual-receipt controller tests on artificial candles only."""
import copy
import json
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

REPO = next(p for p in Path(__file__).resolve().parents
            if (p / 'pyproject.toml').is_file() and (p / 'factorlab').is_dir())
sys.path.insert(0, str(REPO / 'tests'))


@pytest.fixture
def staged():
    from factorlab.discovery import v7_run
    return v7_run


@pytest.fixture
def case(tmp_path, monkeypatch, staged, request):
    from test_v7_feed_audit import Clock
    from test_v7_rolling_registry import candles
    from test_v7_run_factor_pool import source_root

    from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
    from factorlab.discovery.neural_forward_ticket import _bind_authority
    from factorlab.discovery.scheduled_search import FrozenSWorkSchedule
    from factorlab.discovery.v7_forward import ForwardRuntime
    from factorlab.discovery.v7_s_controller import standard_slots
    source_root(tmp_path, monkeypatch)
    clock = Clock('2026-09-30T23:30:00Z')
    runtime = ForwardRuntime(tmp_path/'runtime',tmp_path/'feed',clock=clock)
    try:
        path=tmp_path/'artificial.parquet'
        candles('2026-02-01T00:00Z','2026-09-30T23:00Z').to_parquet(path)
        runtime.feed.import_seed(path)
        clock.now=pd.Timestamp('2026-10-01T00:00:10Z')
        authority=MonthlyWorkAuthority(tmp_path/'wallet',run_id='staged-S-artificial',task='volatility',universe=['BTCUSDT'], partition_policy=getattr(request, 'param', None))
        _bind_authority(runtime,authority)
        config=staged.V7Config(evaluation_mode='joint',search_mode='gpt',lookback=4)
        cutoff=pd.Timestamp('2026-10-01T00:00:00Z')
        bars=runtime.feed.snapshot(cutoff,start=cutoff-pd.DateOffset(months=6)-pd.Timedelta(hours=1234),
            archive_delay_seconds=runtime.spec.availability_delay_seconds)
        slots,allocation=standard_slots(candidate_caps=[15]*24,candidate_seconds=[90]*24,
            pruning_cap=27,pruning_seconds=120,forward_solve_holdback=100)
        allocation.update(forward_seconds_holdback=600,nonlinear_seconds_holdback=0)
        schedule=FrozenSWorkSchedule.freeze(config=asdict(config),limits=staged.monthly_budget(config,'2026-10').limits,
            slots=slots,allocation=allocation)
        yield SimpleNamespace(root=tmp_path,module=staged,runtime=runtime,authority=authority,
            config=config,cutoff=cutoff,bars=bars,schedule=schedule,directory=tmp_path/'research')
    finally:
        runtime.close()


class FakeBatch:
    def __init__(self, first_invalid=False):
        self.calls=[]
        self.first_invalid=first_invalid
    def __call__(self, packet, *, model, reasoning):
        from factorlab.discovery.v7_agent import BackendReply
        body=json.loads(packet.body)
        self.calls.append({'model':model,'reasoning':reasoning,'body':body})
        if self.first_invalid and len(self.calls)==1:
            return BackendReply('{}')
        from factorlab.discovery.program_contract import Program
        from factorlab.discovery.program_syntax import CompactProgram, parse_expression
        program=CompactProgram.from_dict(body['parents'][0]['program']).to_program().to_dict()
        program['parameters']={}
        program['outputs']=[parse_expression(f'mul(last(f("rv_6")),mul(last(f("rv_6")),{len(self.calls)}))')]
        compact=CompactProgram.from_program(Program.from_dict(program)).to_dict()
        value={'version':'v7-proposal-batch-v1','identity':body['identity'],
            'family_id':f'fixture-family-{len(self.calls)}','hypothesis':'Artificial nonlinear persistence.',
            'mechanism':'Quadratic past variation can differ from linear K0.',
            'competing_explanation':'No conditional information beyond K0.',
            'falsification':'Same common-support conditional three-fold losses show no support.',
            'parent_ids':[body['parents'][0]['parent_id']],
            'template':{'base':copy.deepcopy(compact),'slots':[]},'programs':[compact]}
        return BackendReply(json.dumps(value))


def run(case, backend=None, **kwargs):
    return case.module.run_month(case.directory,case.bars,case.cutoff,config=case.config,
        spec=case.runtime.spec,backend=backend or FakeBatch(),exposure_path=case.runtime.lifecycle.exposure.path,
        feed_directory=case.runtime.feed.directory,authority=case.authority,schedule=case.schedule,**kwargs)


def read_state(case):
    with sqlite3.connect(case.directory/'state.sqlite') as db:
        return json.loads(db.execute('SELECT value FROM months WHERE month=?',('2026-10',)).fetchone()[0])


def test_real_three_family_loop_same_wallet_and_seven_prepaid_baseline(case,monkeypatch):
    original=case.module.build_rolling_data
    observed=[]
    def build(*args,**kwargs):
        with case.authority._transaction() as db:
            reservations=case.authority._verify(db)['reservations']
        assert len(reservations)==1
        first=next(iter(reservations.values()))
        assert first['counts']['solve']==7 and first['status']=='running'
        observed.append(first)
        return original(*args,**kwargs)
    monkeypatch.setattr(case.module,'build_rolling_data',build)
    backend=FakeBatch()
    result=run(case,backend)
    assert len(backend.calls)==3
    assert all(c['model']=='gpt-6-astra' and c['reasoning']=='high' for c in backend.calls)
    assert result['accounting']=='shared_authority_fixed_schedule_v1'
    assert result['serving_eligible'] is False and result['hard_deadline_enforced'] is False
    state=read_state(case)
    assert state['status']=='complete' and state['baseline'] is not None
    assert state['budget']['counts']['solve']>=52
    assert state['budget']['counts']['f']==3
    assert all(len(item['result']['folds'])==3 for item in state['pool']['records'].values())
    assert state['budget']['counts']['family']==3
    assert state['budget']['counts']['gpt_attempt']==3
    from factorlab.discovery.v7_memory import ResearchMemory
    assert len(ResearchMemory.from_state(state['research_memory']).scheduled)<=24
    assert observed[0]['lease']==json.loads((case.directory/'S_preparations/2026-10/identity.json').read_text())['record']['execution_envelope']['issuer']
    scheduled=list((case.directory/'work_authority/2026-10/schedule/results').glob('*.json'))
    assert len(scheduled)==29
    assert case.runtime.lifecycle.active_model('A0') is None
    spent=case.authority.status(observed[0]['actual_month'])
    with pytest.raises(ValueError,match='readback|twice'):
        run(case,backend)
    assert case.authority.status(observed[0]['actual_month'])==spent


def test_feature_failure_spends_original_baseline_without_second_wallet(case,monkeypatch):
    def bad(*args,**kwargs):
        raise OSError('artificial feature input failure')
    monkeypatch.setattr(case.module,'build_rolling_data',bad)
    with pytest.raises(OSError,match='feature input'):
        run(case)
    with case.authority._transaction() as db:
        rows=list(case.authority._verify(db)['reservations'].values())
    assert len(rows)==1 and rows[0]['status']=='failed_consumed' and rows[0]['counts']['solve']==7
    with pytest.raises(ValueError,match='readback|twice'):
        run(case)
    assert case.authority.verify()['reservations']==1


def test_missing_forward_time_allocation_rejects_before_any_reservation(case):
    from factorlab.discovery.scheduled_search import FrozenSWorkSchedule
    body=case.schedule.to_dict()
    del body['allocation']['forward_seconds_holdback']
    case.schedule=FrozenSWorkSchedule.freeze(config=body['config'],limits=body['limits'],
        slots=body['slots'],allocation=body['allocation'])
    with pytest.raises(ValueError,match='holdbacks'):
        run(case)
    assert case.authority.verify()['reservations']==0


def test_actual_feed_cannot_fall_back_to_offline_wallet(case):
    with pytest.raises(ValueError,match='shared authority'):
        case.module.run_month(case.directory,case.bars,case.cutoff,config=case.config)
    assert case.authority.verify()['reservations']==0


def recapped(case, cap):
    from factorlab.discovery.scheduled_search import FrozenSWorkSchedule
    body=case.schedule.to_dict()
    for slot in body['slots']:
        if slot['stage']=='candidate':
            slot['counts']['solve']=cap
    body['allocation']['S_solve']=7+24*cap+27
    case.schedule=FrozenSWorkSchedule.freeze(config=body['config'],limits=body['limits'],slots=body['slots'],allocation=body['allocation'])


def test_two_attempts_share_original_family_and_quota_does_not_stop_later_families(case):
    recapped(case,1)
    backend=FakeBatch(first_invalid=True)
    result=run(case,backend)
    assert len(backend.calls)==4
    assert result['budget']['counts']['solve']==7
    assert result['budget']['counts']['gpt_attempt']==4
    assert sum(a['status']=='quota_consumed' for a in result['attempts'])==3
    with case.authority._transaction() as db:
        rows=list(case.authority._verify(db)['reservations'].values())
    families=[r for r in rows if r['binding'].get('work',{}).get('stage')=='family']
    assert len(families)==3
    assert [r['actual_counts']['gpt_attempt'] for r in families]==[2,1,1]
    assert all(r['counts']['gpt_attempt']==2 and r['lease']['lease_seconds']==300 for r in families)
    failed=[r for r in rows if r['status']=='failed_consumed']
    assert len(failed)==3 and all(r['counts']['solve']==1 for r in failed)
    assert all('cold-fold solve bound' in r['reason'] for r in failed)


def test_transport_or_io_failure_closes_s_keeps_original_family_charge(case):
    calls=[]
    def failed_backend(*args,**kwargs):
        calls.append(1)
        raise OSError('synthetic transport IO failure')
    with pytest.raises(OSError,match='transport IO'):
        run(case,failed_backend)
    with case.authority._transaction() as db:
        rows=list(case.authority._verify(db)['reservations'].values())
    assert len(rows)==2 and len(calls)==1
    assert rows[-1]['counts']['gpt_attempt']==2 and rows[-1]['status']=='failed_consumed'
    with pytest.raises(ValueError,match='readback|twice'):
        run(case,failed_backend)
    assert len(calls)==1


def test_frozen_same_batch_validation_cannot_enter_feature_builder(case,monkeypatch):
    case.runtime.lifecycle.exposure.record(source=case.runtime.feed.identity,batch='2026-10',
        symbols=['BTCUSDT'],start='2026-09-01T00:00:00Z',end='2026-10-01T00:00:00Z',role='V')
    monkeypatch.setattr(case.module,'build_rolling_data',lambda *_: pytest.fail('Protected V entered features'))
    with pytest.raises(ValueError,match='Protected V/C'):
        run(case)
    assert case.authority.verify()['reservations']==1


def test_controls_third_does_not_create_another_free_wallet(case):
    from dataclasses import replace
    case.config=replace(case.config,budget_policy='controls_third',research_seconds=2400,profile_candidates=2)
    with pytest.raises(ValueError,match='S schedule config differs from RunStore contract'):
        run(case)
    assert case.authority.verify()['reservations']==0


def test_supplied_preparation_cannot_pay_for_another_run_directory(case,monkeypatch):
    from factorlab.discovery.s_preparation import SPreparationEnvelope
    source={'feed_identity':case.bars.attrs['feed_identity'],
            'feed_directory':str(case.runtime.feed.directory),
            'exposure_path':str(case.runtime.lifecycle.exposure.path),'cutoff':case.cutoff.isoformat()}
    prep=SPreparationEnvelope.prepare(case.authority,case.root/'different-run','2026-10',
        config=asdict(case.config),spec=asdict(case.runtime.spec),source_reference=source,
        schedule_hash=case.schedule.digest)
    monkeypatch.setattr(case.module,'build_rolling_data',lambda *_: pytest.fail('Wrong-run preparation spent'))
    with pytest.raises(ValueError,match='actual invocation'):
        run(case,preparation=prep)
    assert prep.baseline_record()['status']=='reserved'
