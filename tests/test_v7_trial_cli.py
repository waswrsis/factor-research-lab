"""Real original registry inspection: no hidden initialization or work."""
import json
import pytest

from test_v7_full_calibration_registry import case
from factorlab.discovery.v7_cli import main


def _contents(directory):
    return {str(p.relative_to(directory)): p.read_bytes()
            # SQLite's shared-memory lock area changes even on read-only opens.
            # Durable database/WAL bytes and the complete file set stay checked.
            for p in directory.rglob('*') if p.is_file() and not p.name.endswith('-shm')}


def test_registry_status_reads_actual_frozen_grid_without_writes_or_charge(case, capsys):
    registry, runtime, wallet = case
    charged = wallet.verify()
    original = _contents(registry.directory)
    operations = runtime.lifecycle.db.execute('SELECT COUNT(*) FROM operations').fetchone()[0]
    assert main(['trial-registry-status', '--registry', str(registry.directory)]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value['planned_validation_repetitions'] == 2000
    assert value['read_only'] and not value['work_started'] and not value['formal_gate_eligible']
    assert _contents(registry.directory) == original
    assert wallet.verify() == charged
    assert runtime.lifecycle.db.execute('SELECT COUNT(*) FROM operations').fetchone()[0] == operations


def test_registry_status_missing_directory_is_not_initialized(tmp_path, monkeypatch):
    from factorlab import protocol
    monkeypatch.setattr(protocol, 'ROOT', tmp_path)
    missing = tmp_path / 'not-created'
    with pytest.raises((ValueError, FileNotFoundError)):
        main(['trial-registry-status', '--registry', str(missing)])
    assert not missing.exists()


def test_freeze_invalid_plan_does_not_create_registry(tmp_path, monkeypatch):
    from factorlab import protocol
    monkeypatch.setattr(protocol, 'ROOT', tmp_path)
    plan = tmp_path / 'invalid.json'
    plan.write_text('{"acceptance": {}, "acceptance": {}}', encoding='utf-8')
    target = tmp_path / 'not-created'
    with pytest.raises(ValueError, match='Duplicate'):
        main(['trial-registry-freeze', '--registry', str(target), '--plan', str(plan)])
    assert not target.exists()


def test_original_policy_census_keeps_all_missing_paths_without_initialization(case, capsys):
    registry, runtime, wallet = case
    charged = wallet.verify()
    original = _contents(registry.directory)
    assert main(['policy-execution-status', '--registry', str(registry.directory),
                 '--cell', 'null', '--repetition', '0']) == 0
    value = json.loads(capsys.readouterr().out)
    assert value['registered_paths'] == 128 and len(value['paths']) == 128
    assert value['complete_execution_paths'] == 0 and value['read_only']
    assert {p['state'] for p in value['paths']} == {'missing_original_enrollment'}
    assert _contents(registry.directory) == original and wallet.verify() == charged


def test_explicit_enrollment_uses_frozen_scope_and_cannot_reset_a_partial_trial(case, capsys):
    from factorlab.discovery.full_calibration_registry import QuarantinedTrialPermit
    from factorlab.discovery.neural_forward_ticket import open_existing_authority
    registry, _, _ = case
    args = ['trial-enroll', '--registry', str(registry.directory),
            '--cell', 'null', '--repetition', '1']
    assert main(args) == 0
    value = json.loads(capsys.readouterr().out)
    assert not value['work_started'] and not value['formal_gate_eligible']
    permit = QuarantinedTrialPermit(registry, registry.identity_for('null', 1))
    permit.verify()
    wallet = open_existing_authority(permit.paths['authority'])
    assert wallet.verify()['reservations'] == 0
    original = _contents(permit.paths['root'])
    assert main(args) == 0
    capsys.readouterr()
    assert _contents(permit.paths['root']) == original
    # Losing an original enrollment marker cannot recreate the runtime/wallet.
    marker = permit.paths['root'] / 'enrollment.json'
    marker.unlink()
    partial = _contents(permit.paths['root'])
    with pytest.raises(ValueError, match='Partial original'):
        main(args)
    assert not marker.exists() and _contents(permit.paths['root']) == partial
    assert wallet.verify()['reservations'] == 0


@pytest.mark.parametrize('command', ['cold-refit-status', 'cold-refit-run'])
def test_cold_cli_missing_original_never_initializes(tmp_path, monkeypatch, command):
    from factorlab import protocol
    monkeypatch.setattr(protocol, 'ROOT', tmp_path)
    missing = tmp_path / 'missing-cold'
    args = [command, '--directory', str(missing)]
    if command == 'cold-refit-run':
        args += ['--atom', '0']
    with pytest.raises((ValueError, FileNotFoundError)):
        main(args)
    assert not missing.exists()


def test_policy_run_requires_original_oracle_coordinate_before_work(tmp_path):
    missing = tmp_path / 'missing-registry'
    with pytest.raises(SystemExit) as error:
        main(['policy-execution-run', '--registry', str(missing),
              '--cell', 'null', '--repetition', '0'])
    assert error.value.code == 2 and not missing.exists()
