"""Local package safety/recipe checks. No Linux deployment or download is performed."""
import importlib.machinery
import importlib.metadata
import importlib.util
import io
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/v7_linux_bootstrap.sh'
loader = importlib.machinery.SourceFileLoader('v7_linux_recipe_test', str(SCRIPT))
spec = importlib.util.spec_from_loader(loader.name, loader)
recipe = importlib.util.module_from_spec(spec)
loader.exec_module(recipe)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'source'
    for name in (*recipe.FIXED, 'factorlab/__init__.py', 'factorlab/discovery/v7_cli.py'):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# synthetic source\n')
    for name in ('data/private.csv', 'runs/old/results.json', '.env', '.venv/config.py',
                 'work/old/secret.txt', 'outputs/results.txt', 'auth.json', 'factorlab/secret.key'):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('excluded content')
    return root


def test_pinned_cpu_versions_match_current_environment_and_have_no_cuda_chain():
    entries = [line.strip() for line in Path('config/v7-linux-requirements.txt').read_text().splitlines()
               if line.strip() and not line.startswith('#')]
    names = set()
    for entry in entries:
        name, version = entry.split('==')
        assert importlib.metadata.version(name) == version
        names.add(name)
    assert len(names) == 19
    assert 'cloudpickle' in names
    assert not names & {'torch', 'pyqlib', 'mlflow', 'nvidia-cuda-runtime-cu12'}


def test_cpu_runtime_pins_cover_all_active_transitive_dependencies():
    """The --no-deps install must work without relying on developer packages."""
    entries = [line.strip() for line in Path('config/v7-linux-requirements.txt').read_text().splitlines()
               if line.strip() and not line.startswith('#')]
    pins = {canonicalize_name(name): version for name, version in
            (entry.split('==') for entry in entries)}
    for name in pins:
        for text in importlib.metadata.requires(name) or []:
            dependency = Requirement(text)
            if dependency.marker and not dependency.marker.evaluate({'extra': '', 'sys_platform': 'linux'}):
                continue
            key = canonicalize_name(dependency.name)
            assert key in pins, f'{name} needs an unpinned dependency: {text}'
            assert dependency.specifier.contains(pins[key]), f'{name} dependency version differs: {text}'


def test_polyglot_script_python_cli_help_is_offline():
    result = subprocess.run([sys.executable, '-B', str(SCRIPT), '--help'],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert 'bootstrap' in result.stdout and 'unit-template' in result.stdout
    assert SCRIPT.read_text().splitlines()[1] == "'''exec' \"${V7_PYTHON:-python3.12}\" \"$0\" \"$@\""


def test_deterministic_allowlisted_package_and_safe_new_workspace(source, tmp_path):
    first, second = tmp_path / 'one.tar', tmp_path / 'two.tar'
    receipt = recipe.pack(source, first)
    assert recipe.pack(source, second)['package_sha256'] == receipt['package_sha256']
    manifest, entries = recipe.verify_package(first, receipt['package_sha256'])
    assert set(entries) == set(recipe.FIXED) | {'factorlab/__init__.py', 'factorlab/discovery/v7_cli.py'}
    assert manifest['source_hash'] == receipt['source_hash']
    destination = tmp_path / 'research-v7-new'
    assert recipe.unpack(first, receipt['package_sha256'], destination)['source_hash'] == receipt['source_hash']
    assert (destination / 'factorlab/discovery/v7_cli.py').read_text() == '# synthetic source\n'
    with pytest.raises(ValueError, match='existing'):
        recipe.unpack(first, receipt['package_sha256'], destination)
    with pytest.raises(ValueError, match='hash mismatch'):
        recipe.verify_package(first, '0' * 64)


@pytest.mark.parametrize('name', ['/absolute.py', '../outside.py', 'factorlab/../outside.py',
                                  'factorlab\\unsafe.py', 'C:/unsafe.py', './factorlab/a.py',
                                  'data/secret.csv', 'factorlab//a.py', '.env'])
def test_tar_unsafe_paths_are_rejected_before_any_extraction(tmp_path, name):
    package = tmp_path / 'unsafe.tar'
    with tarfile.open(package, 'w') as archive:
        item = tarfile.TarInfo(name)
        item.size = 1
        archive.addfile(item, io.BytesIO(b'x'))
    with pytest.raises(ValueError, match='Unsafe'):
        recipe.verify_package(package, recipe.digest_file(package))


@pytest.mark.parametrize('kind', [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.DIRTYPE])
def test_tar_link_device_directory_entries_are_rejected(tmp_path, kind):
    package = tmp_path / 'special.tar'
    with tarfile.open(package, 'w') as archive:
        item = tarfile.TarInfo('factorlab/a.py')
        item.type, item.linkname = kind, '../../outside'
        archive.addfile(item)
    with pytest.raises(ValueError, match='Unsafe'):
        recipe.verify_package(package, recipe.digest_file(package))


def test_duplicate_and_changed_manifest_content_rejected(source, tmp_path):
    package = tmp_path / 'original.tar'
    receipt = recipe.pack(source, package)
    with tarfile.open(package) as archive:
        contents = [(member.name, archive.extractfile(member).read()) for member in archive]
    for kind in ('duplicate', 'changed'):
        changed = tmp_path / f'{kind}.tar'
        records = contents + [contents[0]] if kind == 'duplicate' else [(contents[0][0], b'changed'), *contents[1:]]
        with tarfile.open(changed, 'w') as archive:
            for name, content in records:
                item = tarfile.TarInfo(name)
                item.size = len(content)
                archive.addfile(item, io.BytesIO(content))
        with pytest.raises(ValueError):
            recipe.verify_package(changed, recipe.digest_file(changed))
    assert recipe.verify_package(package, receipt['package_sha256'])


def test_workspace_rejects_oldlong80_and_symlink(tmp_path):
    with pytest.raises(ValueError, match='dedicated'):
        recipe.workspace_path(tmp_path / 'research-long80/subdir')
    link = tmp_path / 'linked'
    try:
        link.symlink_to(tmp_path, target_is_directory=True)
    except OSError as exc:
        # A privilege-limited Windows host can still create an actual junction.
        # Exercise real ancestor alias rejection; never replace it with a mock.
        if sys.platform != 'win32' or getattr(exc, 'winerror', None) != 1314:
            raise
        import shutil
        shell = shutil.which('pwsh') or shutil.which('powershell')
        assert shell is not None, 'A real directory alias is required for this test'
        script = tmp_path / 'create-owned-junction.ps1'
        script.write_text(
            'param([string]$OwnedAliasPath, [string]$OwnedAliasTarget)\n'
            'New-Item -ItemType Junction -Path $OwnedAliasPath '
            '-Target $OwnedAliasTarget -ErrorAction Stop | Out-Null\n',
            encoding='utf-8')
        result = subprocess.run(
            [shell, '-NoProfile', '-NonInteractive', '-File', str(script),
             '-OwnedAliasPath', str(link), '-OwnedAliasTarget', str(tmp_path)],
            capture_output=True, text=True, timeout=15, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
        assert link.is_junction() and link.resolve() == tmp_path.resolve()
    with pytest.raises(ValueError, match='symlink'):
        recipe.workspace_path(link / 'new-v7')


def test_preflight_refuses_active_old_service_and_low_disk(source, monkeypatch):
    monkeypatch.setattr(recipe.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(recipe.platform, 'machine', lambda: 'x86_64')
    monkeypatch.setattr(recipe.platform, 'libc_ver', lambda: ('glibc', '2.36'))
    monkeypatch.setattr(recipe, 'systemctl', lambda *args: 'LoadState=loaded\nActiveState=active\n')
    with pytest.raises(RuntimeError, match='active'):
        recipe.preflight(source, 'fr-regression-active.service', scope='user')
    executable = '/home/plague/factorresearch-regression/tools/long80/supervise_iteration.py pilot'
    monkeypatch.setattr(recipe, 'systemctl', lambda *args: f'LoadState=loaded\nActiveState=inactive\nExecStart={executable}\n')
    monkeypatch.setattr(recipe.shutil, 'disk_usage', lambda path: SimpleNamespace(free=100))
    with pytest.raises(RuntimeError, match='4 GiB'):
        recipe.preflight(source, 'exact-long80.service')
    with pytest.raises(ValueError, match='exact existing'):
        recipe.preflight(source, 'bad/unit.service')


def test_old_service_identity_empty_inactive_needs_exact_frozen_receipt(tmp_path):
    import hashlib
    import json

    fields = {'unit': 'fr-regression-active.service', 'scope': 'user',
              'exec_start': '/home/plague/factorresearch-regression/tools/long80/supervise_iteration.py pilot'}
    with pytest.raises(RuntimeError, match='frozen operator'):
        recipe._old_service_identity({'ExecStart': ''}, fields['unit'], 'user')
    receipt = tmp_path / 'old-service.json'
    receipt.write_text(json.dumps({**fields, 'sha256': hashlib.sha256(recipe.canonical(fields)).hexdigest()}))
    assert recipe._old_service_identity({'ExecStart': ''}, fields['unit'], 'user', receipt) == fields
    with pytest.raises(ValueError, match='identity/hash'):
        recipe._old_service_identity({'ExecStart': ''}, fields['unit'], 'system', receipt)
    with pytest.raises(RuntimeError, match='verified long80'):
        recipe._old_service_identity({'ExecStart': '/some/other/service'}, fields['unit'], 'user')


def test_systemd_template_fixes_limits_and_has_no_automatic_service_mutation(source):
    # Template-only fixture; these placeholder files never authorize execution.
    campaign = source / 'runs/test'
    campaign.mkdir(parents=True)
    (campaign / 'campaign.json').write_text('{}')
    (campaign / 'campaign.sqlite').write_bytes(b'')
    command = [str(source / '.venv/bin/python'), '-B',
               str(source / 'scripts/v7_guard/campaign_guard_entry.py'), 'guard',
               str(campaign), str(campaign / 'whole_tree_guard')]
    text = recipe.systemd_template(source, 'old-long80.service', 'user', command)
    for value in ('CPUQuota=25%', 'MemoryHigh=640M', 'MemoryMax=768M', 'MemorySwapMax=0',
                  'TasksMax=64', 'RuntimeMaxSec=7200', 'KillMode=control-group', 'Restart=no',
                  'NoNewPrivileges=yes', 'ExecStopPost=', 'guard-exec'):
        assert value in text
    assert 'sudo' not in text and 'daemon-reload' not in text
    with pytest.raises(ValueError, match='fixed guarded campaign entry'):
        recipe.systemd_template(source, 'old-long80.service', 'user', ['/usr/bin/python3', '-c', 'pass'])


def test_read_only_cgroup_receipt_rejects_widened_limits(tmp_path):
    for name, value in recipe.LIMITS.items():
        (tmp_path / name).write_text(str(value))
    for name, value in {'cpu.max': '25000 100000', 'memory.current': '1024', 'memory.peak': '2048',
                         'memory.events': 'oom 0\noom_kill 0', 'cpu.stat': 'usage_usec 1', 'pids.current': '2'}.items():
        (tmp_path / name).write_text(value)
    assert recipe.resource_snapshot(tmp_path)['limits_checked']
    (tmp_path / 'memory.max').write_text(str(9 * 1024**3))
    with pytest.raises(RuntimeError, match='resource envelope'):
        recipe.resource_snapshot(tmp_path)
    (tmp_path / 'memory.max').write_text(str(recipe.LIMITS['memory.max']))
    (tmp_path / 'cpu.max').write_text('100000 100000')
    with pytest.raises(RuntimeError, match='25%'):
        recipe.resource_snapshot(tmp_path)


def test_bootstrap_recipe_forbids_source_builds_unpinned_resolution_and_global_sync():
    text = SCRIPT.read_text()
    assert "'--only-binary=:all:'" in text and "'--no-deps'" in text
    assert "'--report'" in text and "'--cache-dir', str(root / '.cache/pip')" in text
    assert "'pip', 'check'" in text
    assert "'venv', str(venv)" in text
    assert "'sudo'" not in text and "'uv', 'sync'" not in text
    assert not any('extractall(' in line and not line.lstrip().startswith('#') for line in text.splitlines())


def test_workspace_rejects_noncanonical_parent_components(tmp_path):
    # This positive control prevents canonical-directory rejection being
    # implemented by forbidding every new directory.
    ordinary = tmp_path / 'ordinary'
    ordinary.mkdir()
    assert recipe.workspace_path(ordinary / 'new-v7') == ordinary / 'new-v7'
    with pytest.raises(ValueError, match='dedicated'):
        recipe.workspace_path(ordinary / '..' / 'new-v7')
