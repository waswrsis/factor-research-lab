import json

import pytest

from factorlab.discovery import shared_controls, v7_cli, v7_seed


def test_shared_control_status_only_reads_existing_artifact(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(shared_controls, 'inspect_shared_controls',
                        lambda directory: seen.append(directory) or {'completed_budget_comparison': False})
    assert v7_cli.main(['shared-controls-status', '--run', 'work/control-status']) == 0
    assert seen == ['work/control-status']
    assert not json.loads(capsys.readouterr().out)['completed_budget_comparison']


def test_shared_controls_require_explicit_gpt_switch_before_data_read():
    with pytest.raises(SystemExit) as exc:
        v7_cli.main(['shared-controls', '--run', 'work/control', '--bars', 'missing',
                     '--cutoff', '2025-04-01T00:00Z', '--symbol', 'BTCUSDT'])
    assert exc.value.code == 2


def test_seed_bridge_policy_is_an_explicit_opt_in(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(v7_seed, 'prepare_seed',
                        lambda *args, **kwargs: seen.append(kwargs) or {'status': 'pending'})
    assert v7_cli.main(['prepare-seed', '--run', 'work/seed-cli', '--symbol', 'BTCUSDT',
                       '--allow-daily-bridge']) == 0
    assert seen[0]['allow_daily_bridge'] is True
    assert json.loads(capsys.readouterr().out)['status'] == 'pending'
