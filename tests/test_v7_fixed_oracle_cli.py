"""Fixed-estimand CLI rejection/read-only boundaries; no oracle success claim."""
import pytest

from factorlab.discovery.v7_cli import main
from test_v7_full_calibration_registry import case


@pytest.mark.parametrize('command', ('fixed-oracle-status', 'fixed-oracle-report', 'fixed-oracle-pump', 'fixed-oracle-run'))
def test_missing_fixed_oracle_registry_is_not_initialized(tmp_path, monkeypatch, command):
    from factorlab import protocol
    monkeypatch.setattr(protocol, 'ROOT', tmp_path)
    missing = tmp_path / 'missing'
    with pytest.raises((ValueError, FileNotFoundError)):
        main([command, '--registry', str(missing), '--cell', 'null', '--repetition', '0',
              '--oracle-index', '0', '--slot', 'factor1'])
    assert not missing.exists()


def test_fixed_oracle_does_not_accept_algorithm_or_arbitrary_recipe(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(['fixed-oracle-freeze', '--registry', str(tmp_path / 'r'), '--cell', 'null',
              '--repetition', '0', '--oracle-index', '0', '--slot', 'algorithm',
              '--generator', str(tmp_path / 'g')])
    assert exc.value.code == 2
    assert not (tmp_path / 'r').exists()
    assert not (tmp_path / 'g').exists()


def test_fixed_replica_requires_original_oracle_coordinate_before_initialization(case):
    registry, runtime, wallet = case
    original = {str(p.relative_to(registry.directory)): p.read_bytes()
                for p in registry.directory.rglob('*')
                if p.is_file() and not p.name.endswith('-shm')}
    charged = wallet.verify()
    with pytest.raises(ValueError):
        main(['trial-enroll', '--registry', str(registry.directory), '--cell', 'null',
              '--repetition', '0', '--estimand-slot', 'factor1'])
    assert wallet.verify() == charged
    assert {str(p.relative_to(registry.directory)): p.read_bytes()
            for p in registry.directory.rglob('*')
            if p.is_file() and not p.name.endswith('-shm')} == original
