"""Operational executables belong to the same immutable research identity."""
import pytest
from test_v7_run_factor_pool import source_root

from factorlab.discovery.evidence import snapshot_source
from factorlab.discovery.evolution import _current_source_files
from factorlab.discovery.rolling import RollingSpec
from factorlab.discovery.v7_run import RunStore, V7Config, source_files
from factorlab.protocol import fingerprint


@pytest.mark.parametrize('name', ['v7_public_fetch.py', 'v7_long_running.py', 'v7_linux_bootstrap.sh'])
def test_operational_worker_mutation_invalidates_research_identity(tmp_path, monkeypatch, name):
    source_root(tmp_path, monkeypatch)
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    worker = scripts / name
    worker.write_text('frozen_worker = 1\n')
    first = source_files()
    assert f'scripts/{name}' in first
    saved = snapshot_source('archived-source')
    assert saved['files'] == first
    assert _current_source_files() == first
    run = RunStore('research', V7Config(), RollingSpec())
    run.close()
    worker.write_text('frozen_worker = 2\n')
    assert fingerprint(source_files()) != fingerprint(first)
    assert (tmp_path / 'archived-source/scripts' / name).read_text() == 'frozen_worker = 1\n'
    with pytest.raises(ValueError):
        RunStore('research', V7Config(), RollingSpec())


def test_removed_operational_source_does_not_match_frozen_identity(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    worker = scripts / 'v7_public_fetch.py'
    worker.write_text('worker = 1\n')
    first = source_files()
    worker.unlink()
    assert source_files() != first
