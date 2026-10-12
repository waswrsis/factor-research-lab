"""Actual default-allocation entry contracts; no GPT/network/service launch."""
import json
from dataclasses import asdict, replace

import pytest
from test_v7_s_cli_entries import arguments
from test_v7_s_controller_integration import case as case  # noqa: PLC0414
from test_v7_s_controller_integration import staged as staged  # noqa: PLC0414

from factorlab.discovery.monthly_schedule import VERSION, default_schedule

pytestmark = pytest.mark.parametrize('case', ['standard'], indirect=True)


@pytest.fixture
def long_module():
    from scripts import v7_long_running as module
    return module


def test_explicit_default_CLI_is_expanded_before_prepaid_source_load(case, monkeypatch):
    from factorlab.discovery import v7_cli
    args = arguments(case)
    args[args.index('--work-schedule') + 1] = VERSION
    observed = []
    def read(path):
        with case.authority._transaction() as db:
            rows = list(case.authority._verify(db)['reservations'].values())
        assert len(rows) == 1 and rows[0]['counts']['solve'] == 7
        anchor = json.loads((case.directory / 'S_preparations/2026-10/identity.json').read_text())['record']
        assert anchor['work_schedule_hash'] == default_schedule(case.config, month='2026-10').digest
        observed.append(rows[0]['lease'])
        raise OSError('intentional artificial source failure')
    monkeypatch.setattr(v7_cli.pd, 'read_parquet', read)
    with pytest.raises(OSError, match='artificial source'):
        v7_cli.main(args)
    assert len(observed) == 1
    assert case.authority.verify()['reservations'] == 1


def test_actual_long_campaign_reuses_declared_wallet_and_exact_schedule(case, long_module):
    plan = long_module.CampaignPlan(end_at='2027-01-01T00:00:00Z',
        max_sessions=2, max_reserved_seconds=14400, max_reserved_steps=14400, max_fit_attempts=10,
        forward_directory=str(case.runtime.directory), feed_directory=str(case.runtime.feed.directory),
        research={'backend': 'disabled', 'research': asdict(case.config)},
        lifecycle={'source_id': case.runtime.feed.identity},
        work_authority={'directory': str(case.authority.directory), 'manifest_hash': case.authority.manifest_hash},
        work_schedule=default_schedule(case.config, month='2026-10').to_dict())
    authority, schedule = long_module.actual_work_contract(plan)
    assert authority.directory == case.authority.directory
    assert schedule.digest == default_schedule(case.config, month='2026-10').digest
    assert authority.verify()['reservations'] == 0
    with pytest.raises(ValueError, match='wallet|authority identity'):
        long_module.actual_work_contract(replace(plan, work_authority={
            'directory': str(case.authority.directory), 'manifest_hash': 'a'*64}))
    old = replace(plan, work_authority=None, work_schedule=None)
    with pytest.raises(ValueError, match='actual shared authority'):
        long_module.actual_work_contract(old)
    assert authority.verify()['reservations'] == 0
