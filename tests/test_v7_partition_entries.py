"""Actual wallets; early named-policy rejection before any source read/debit."""
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from test_monthly_partitions import root as root  # noqa: PLC0414 -- pytest fixture

from factorlab.discovery import v7_cli
from factorlab.discovery.monthly_schedule import (
    VERSION,
    default_control_schedules,
    default_schedule,
)
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
from factorlab.discovery.rolling import RollingSpec
from factorlab.discovery.shared_controls_executor import SharedControlsExecutor
from factorlab.discovery.v7_autonomous import AutonomousConfig, ResearchScheduler
from factorlab.discovery.v7_run import V7Config


def diagnostic_wallet(root):
    return MonthlyWorkAuthority(root/'wallet', run_id='diagnostic-only', task='volatility',
                                universe=['BTCUSDT'], partition_policy=None)


def test_cli_rejects_default_schedule_on_diagnostic_wallet_before_read_or_debit(root, monkeypatch):
    wallet = diagnostic_wallet(root)
    config = V7Config(evaluation_mode='joint')
    path = root/'config.json'
    path.write_text(json.dumps(asdict(config)))
    def forbid(*a, **k):
        raise AssertionError('Policy mismatch reached numerical source read')
    monkeypatch.setattr(v7_cli.pd, 'read_parquet', forbid)
    args = SimpleNamespace(config=str(path), availability_delay_seconds=1, window_days=None,
        offline_diagnostic=False, work_authority=str(wallet.directory), work_schedule=VERSION,
        feed_directory=str(root/'unopened-feed'), exposure_path=str(root/'unopened-exposure'),
        cutoff='2026-10-01T00:00:00Z', run=str(root/'research'), bars=str(root/'unopened.parquet'))
    with pytest.raises(ValueError, match='authoritative monthly partition'):
        v7_cli._research_from_file(args)
    assert wallet.verify()['reservations'] == 0
    assert not (root/'research').exists()


def test_scheduler_rejects_named_policy_before_creating_runtime_journal(root):
    wallet = diagnostic_wallet(root)
    config = V7Config(evaluation_mode='joint')
    runtime = SimpleNamespace(config=SimpleNamespace(symbols=('BTCUSDT',)), directory=root/'forward')
    with pytest.raises(ValueError, match='authoritative monthly partition'):
        ResearchScheduler(runtime, config=AutonomousConfig(research=config, backend='disabled'), authority=wallet,
                          schedule=default_schedule(config, month='2026-10'))
    assert wallet.verify()['reservations'] == 0
    assert not runtime.directory.exists()


def test_controls_reject_wrong_policy_before_opening_feed_or_creating_claim(root):
    wallet = MonthlyWorkAuthority(root/'wallet', run_id='standard', task='volatility',
                                 universe=['BTCUSDT'], partition_policy='standard')
    with pytest.raises(ValueError, match='authoritative monthly partition'):
        SharedControlsExecutor(root/'controls', authority=wallet,
            schedules=default_control_schedules(symbols=('BTCUSDT',), month='2026-10'),
            feed_directory=root/'unopened-feed', exposure_path=root/'unopened-exposure',
            cutoff='2026-10-01T00:00:00Z', spec=RollingSpec())
    assert wallet.verify()['reservations'] == 0
    assert not (root/'controls').exists()


def test_direct_run_month_rejects_default_on_diagnostic_wallet_before_input_work(root):
    import pandas as pd

    from factorlab.discovery.v7_run import run_month
    wallet = diagnostic_wallet(root)
    config = V7Config(evaluation_mode='joint')
    bars = pd.DataFrame()
    bars.attrs['feed_identity'] = 'unopened-fixture'
    with pytest.raises(ValueError, match='authoritative monthly partition'):
        run_month(root/'research', bars, '2026-10-01T00:00:00Z', config=config,
            authority=wallet, schedule=default_schedule(config, month='2026-10'),
            feed_directory=root/'unopened-feed', exposure_path=root/'unopened-exposure')
    assert wallet.verify()['reservations'] == 0
    assert not (root/'research').exists()


@pytest.mark.parametrize('mode', ['standard', 'controls', 'diagnostic'])
def test_authority_init_freezes_explicit_policy_without_starting_work(root, mode, capsys):
    v7_cli.main(['work-authority-init', '--run', str(root/'forward'), '--feed', str(root/'feed'),
                '--symbol', 'BTCUSDT', '--work-authority', str(root/'wallet'), '--partition-policy', mode])
    result = json.loads(capsys.readouterr().out)
    assert result['work_started'] is False
    assert (result['partition_policy'] is None if mode == 'diagnostic'
            else result['partition_policy']['mode'] == mode)
    from factorlab.discovery.neural_forward_ticket import open_existing_authority
    assert open_existing_authority(root/'wallet').verify()['reservations'] == 0


def test_authority_init_cannot_choose_a_policy_implicitly(root):
    with pytest.raises(SystemExit) as error:
        v7_cli.main(['work-authority-init', '--run', str(root/'forward'), '--feed', str(root/'feed'),
                    '--symbol', 'BTCUSDT', '--work-authority', str(root/'wallet')])
    assert error.value.code == 2
    assert not (root/'wallet').exists()


def test_real_three_family_S_completes_in_one_partitioned_wallet(root):
    """Actual synthetic receipt/numerical loop, FakeBatch explicitly uncertified."""
    import pandas as pd
    from test_v7_feed_audit import Clock
    from test_v7_rolling_registry import candles
    from test_v7_s_controller_integration import FakeBatch

    from factorlab.discovery.neural_forward_ticket import _bind_authority
    from factorlab.discovery.v7_forward import ForwardRuntime
    from factorlab.discovery.v7_run import run_month
    clock = Clock('2026-09-30T23:30:00Z')
    runtime = ForwardRuntime(root/'forward', root/'feed', clock=clock)
    try:
        datafile = root/'artificial.parquet'
        candles('2026-02-01T00:00Z', '2026-09-30T23:00Z').to_parquet(datafile)
        runtime.feed.import_seed(datafile)
        clock.now = pd.Timestamp('2026-10-01T00:00:10Z')
        authority = MonthlyWorkAuthority(root/'wallet', run_id='actual-partitioned-S-artificial',
            task='volatility', universe=['BTCUSDT'], partition_policy='standard')
        _bind_authority(runtime, authority)
        config = V7Config(evaluation_mode='joint', search_mode='gpt', lookback=4)
        cutoff = pd.Timestamp('2026-10-01T00:00:00Z')
        bars = runtime.feed.snapshot(cutoff, start=cutoff-pd.DateOffset(months=6)-pd.Timedelta(hours=1234),
            archive_delay_seconds=runtime.spec.availability_delay_seconds)
        schedule = default_schedule(config, month='2026-10')
        backend = FakeBatch()
        result = run_month(root/'research', bars, cutoff, config=config, spec=runtime.spec,
            backend=backend, exposure_path=runtime.lifecycle.exposure.path,
            feed_directory=runtime.feed.directory, authority=authority, schedule=schedule)
        assert result['serving_eligible'] is False
        assert result['budget']['counts']['gpt_attempt'] == 3
        actual_month = pd.Timestamp.now(tz='UTC').strftime('%Y-%m')
        status = authority.status(actual_month)
        assert status['partitions']['S']['counts']['gpt_attempt'] == 6
        assert status['partitions']['S']['counts']['solve'] <= 943
        assert status['owners']['S']['run_directory'] == str(root/'research')
        assert 'forward' not in status['partitions']
        before = authority.verify()
        with pytest.raises(ValueError, match='readback only|readback-only'):
            run_month(root/'research', bars, cutoff, config=config, spec=runtime.spec,
                backend=backend, exposure_path=runtime.lifecycle.exposure.path,
                feed_directory=runtime.feed.directory, authority=authority, schedule=schedule)
        assert authority.verify() == before and len(backend.calls) == 3
    finally:
        runtime.close()
