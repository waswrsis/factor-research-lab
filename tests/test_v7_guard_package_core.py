"""Current-core source freezing, package boundaries and genuine local kill recovery."""
import json
import os
import runpy
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from factorlab.discovery import evidence

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'work/v7g/guard_core_acceptance'
BOOTSTRAP = runpy.run_path(str(ROOT / 'scripts/v7_linux_bootstrap.sh'))


def run(*args, cwd=ROOT, timeout=45):
    result = subprocess.run([sys.executable, '-B', *map(str, args)], cwd=cwd,
                            capture_output=True, text=True, timeout=timeout, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_exact_core_package_verify_unpack_roundtrip():
    root = HERE / ('package-' + uuid.uuid4().hex[:8])
    run(HERE / 'make_fixture.py', root.name)
    original = json.loads((root / 'core_fixture_manifest.json').read_text())
    assert not original['explicit_artificial_changes']
    result = BOOTSTRAP['pack'](root, root.with_suffix('.tar'))
    manifest, entries = BOOTSTRAP['verify_package'](result['package'], result['package_sha256'])
    assert set(BOOTSTRAP['GUARD_FILES']) <= set(entries)
    for name in ('scripts/v7_guard/extra.py', 'work/secret.json',
                 'scripts/v7_guard/../v7_long_running.py'):
        assert not BOOTSTRAP['allowed_name'](name)
    unpacked = root.with_name(root.name + '-unpacked')
    BOOTSTRAP['unpack'](result['package'], result['package_sha256'], unpacked)
    for name in BOOTSTRAP['GUARD_FILES']:
        assert (unpacked / name).read_bytes() == (ROOT / name).read_bytes()
    assert manifest['source_hash'] == result['source_hash']


def test_core_guard_identity_all_or_none_and_snapshot_detects_mutation(tmp_path, monkeypatch):
    from factorlab import protocol
    monkeypatch.setattr(protocol, 'ROOT', tmp_path)
    assert not any('v7_guard' in str(p) for p in evidence.source_paths(tmp_path))
    guards = BOOTSTRAP['GUARD_FILES']
    for name in (*guards, 'pyproject.toml', 'uv.lock'):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('original frozen bytes')
    identity = evidence.snapshot_source(tmp_path / 'snapshot')['files']
    for name in guards:
        assert (tmp_path / 'snapshot' / name).read_bytes() == (tmp_path / name).read_bytes()
        assert identity[name] == evidence.file_hash(tmp_path / name)
    # Rename rather than delete; the original evidence remains available.
    (tmp_path / guards[0]).rename(tmp_path / 'original-guard.preserved')
    with pytest.raises(ValueError, match='Incomplete'):
        evidence.source_paths(tmp_path)
    (tmp_path / guards[0]).write_text('different source')
    assert identity[guards[0]] != evidence.file_hash(tmp_path / guards[0])


@pytest.mark.skipif(os.name != 'nt', reason='Windows Job test; actual Linux acceptance is separate')
def test_core_native_timeout_then_no_refund_postmortem():
    root = HERE / ('native-' + uuid.uuid4().hex[:8])
    fixture = ROOT / 'work/v7g/guard_fixture_forward_authorization_bf1ca245_20261010'
    import hashlib
    assert hashlib.sha256((fixture / 'binding.json').read_bytes()).hexdigest() == '82aa748992cc2dc7019c3c3a7cfd10e1d563daf7bc77c05f42bc927016db2f1e'
    run(fixture / 'make_native_fixture.py', root.name)
    run(root / 'setup_fixture.py', 'native_block', cwd=root)
    campaign, guard = root / 'fixture/campaign', root / 'fixture/guard'
    entry = root / 'scripts/v7_guard/campaign_guard_entry.py'
    with (root / 'supervisor.log').open('wb') as log:
        child = subprocess.Popen([sys.executable, '-B', str(entry), 'guard', str(campaign), str(guard)],
            cwd=root, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            child.wait(timeout=85)
        finally:
            if child.poll() is None:
                (campaign / 'STOP').write_text('Bounded own fixture cleanup')
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
    assert child.returncode != 0
    assert (root / 'fixture/native-block.json').exists(), (root / 'supervisor.log').read_text()
    run(ROOT / 'work/v7g/guard_package_stage/check_postmortem.py', root, cwd=root)
    report = json.loads((root / 'fixture/postmortem-test.json').read_text())
    assert report['status'] == 'passed' and report['actual_windows_job']
    assert not report['Linux_resource_acceptance'] and not report['actual_GPT']
