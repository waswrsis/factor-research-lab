"""Explicit dispatch contracts; no network, model calls or synthetic alpha."""
import json

from factorlab.discovery import full_pipeline_calibration, v7_cli, v7_seed


def test_public_seed_cli_forwards_bounded_arguments_without_starting_a_feed(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(v7_seed, 'prepare_seed', lambda *args, **kwargs:
                        seen.append((args, kwargs)) or {'status': 'pending'})
    assert v7_cli.main(['prepare-seed', '--run', 'work/seed-cli', '--symbol', 'BTCUSDT',
        '--as-of', '2026-10-04T12:00:00Z', '--months', '8', '--max-new-archives', '1']) == 0
    assert seen == [(('work/seed-cli',), {'symbols': ('BTCUSDT',), 'as_of': '2026-10-04T12:00:00Z',
                                        'months': 8, 'max_new_archives': 1})]
    assert json.loads(capsys.readouterr().out) == {'status': 'pending'}


def test_five_slot_cli_uses_single_work_unit_and_never_old_certificate_api(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(full_pipeline_calibration, 'run_full_pipeline_calibration',
        lambda *args, **kwargs: seen.append((args, kwargs)) or {'formal_gate_eligible': False})
    assert v7_cli.main(['full-calibrate', '--run', 'work/five-cli']) == 0
    assert seen[0][0] == ('work/five-cli',)
    assert seen[0][1]['max_new_trials'] == 1
    assert seen[0][1]['plan'].oracle_paths == 2
    assert json.loads(capsys.readouterr().out) == {'formal_gate_eligible': False}


def test_five_slot_status_cli_only_reads_existing_evidence(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(full_pipeline_calibration, 'inspect_full_pipeline_calibration',
        lambda path: seen.append(path) or {'complete': False})
    assert v7_cli.main(['full-calibration-status', '--run', 'work/five-cli']) == 0
    assert seen == ['work/five-cli']
    assert json.loads(capsys.readouterr().out) == {'complete': False}
