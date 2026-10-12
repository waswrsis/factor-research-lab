"""Staged CLI uses real local authority/receipt fixtures, before any fitting."""
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

REPO=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file() and (p/'factorlab').is_dir())
sys.path.insert(0,str(REPO/'tests'))
from test_v7_s_controller_integration import case as case  # noqa: PLC0414
from test_v7_s_controller_integration import staged as staged  # noqa: PLC0414


@pytest.fixture
def cli():
    from factorlab.discovery import v7_cli
    return v7_cli


def arguments(case, *, source='bars.parquet'):
    config=case.root/'config.json'; config.write_text(json.dumps(asdict(case.config)))
    schedule=case.root/'schedule.json'; schedule.write_text(json.dumps(case.schedule.to_dict()))
    return ['research','--bars',str(case.root/source),'--run',str(case.directory),'--cutoff',case.cutoff.isoformat(),
        '--config',str(config),'--work-authority',str(case.authority.directory),'--work-schedule',str(schedule),
        '--feed-directory',str(case.runtime.feed.directory),'--exposure-path',str(case.runtime.lifecycle.exposure.path),
        '--availability-delay-seconds',str(case.runtime.spec.availability_delay_seconds)]


class BeforeFit(BaseException):
    pass


def test_cli_original_prepay_exists_before_first_local_read(case,cli,monkeypatch):
    case.bars.to_parquet(case.root/'bars.parquet')
    actual=cli.pd.read_parquet
    reads=[]
    def read(*args,**kwargs):
        with case.authority._transaction() as db:
            rows=list(case.authority._verify(db)['reservations'].values())
        assert len(rows)==1 and rows[0]['status']=='reserved'
        assert rows[0]['counts']['solve']==7
        reads.append(rows[0]['lease'])
        return actual(*args,**kwargs)
    monkeypatch.setattr(cli.pd,'read_parquet',read)
    def before_fit(*args,**kwargs):
        raise BeforeFit
    monkeypatch.setattr(case.module,'_baseline_readout',before_fit)
    with pytest.raises(BeforeFit):
        cli.main(arguments(case))
    assert len(reads)==1
    prep=json.loads((case.directory/'S_preparations/2026-10/identity.json').read_text())['record']
    assert prep['clock_scope']=='before_local_source_load'
    assert prep['execution_envelope']['issuer']==reads[0]


def test_cli_failed_source_read_keeps_original_cost_and_refuses_reload(case,cli):
    with pytest.raises((OSError,ValueError)):
        cli.main(arguments(case,source='missing.parquet'))
    with case.authority._transaction() as db:
        rows=list(case.authority._verify(db)['reservations'].values())
    assert len(rows)==1 and rows[0]['status']=='failed_consumed' and rows[0]['counts']['solve']==7
    with pytest.raises(ValueError,match='readback only'):
        cli.main(arguments(case,source='missing.parquet'))
    assert case.authority.verify()['reservations']==1


def test_cli_never_implicitly_falls_back_to_legacy_wallet(case,cli,monkeypatch):
    args=arguments(case)
    index=args.index('--work-authority'); del args[index:index+2]
    monkeypatch.setattr(cli.pd,'read_parquet',lambda *_: pytest.fail('unpaid source read'))
    with pytest.raises(ValueError,match='actual work authority'):
        cli.main(args)
    assert case.authority.verify()['reservations']==0
