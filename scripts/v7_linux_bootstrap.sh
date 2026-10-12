#!/bin/sh
'''exec' "${V7_PYTHON:-python3.12}" "$0" "$@"
' '''
"""POSIX-shell/Python polyglot. Packaging also runs with project Python on Windows.

No command installs, starts or stops a system service. bootstrap is the only
network/install command; unpack refuses every existing destination. All package
paths and hashes are validated before extraction; no tar extractall is used.
"""
import argparse
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

MANIFEST = 'V7_PACKAGE_MANIFEST.json'
FIXED = ('AGENTS.md', 'pyproject.toml', 'uv.lock',
         'config/discovery_v7_local.json', 'config/discovery_v7_gpt.json',
         'config/v7_service_pilot.json', 'config/v7_autonomous_pilot.json',
         'config/discovery_v7_full_gpt.json', 'config/v7_autonomous_full.json',
         'config/v7-linux-requirements.txt', 'config/v7-linux-neural-requirements.txt',
         'config/v7-linux-neural-lock.json', 'scripts/v7_linux_bootstrap.sh',
         'scripts/v7_public_fetch.py', 'scripts/v7_long_running.py',
         'scripts/v7_layered_validation.py',
         'tools/linux_full_path/linux_production_path.py', 'tools/linux_full_path/README.txt',
         'docs/V7_LINUX_OPERATIONS.txt', 'docs/V7_RUNTIME_GUIDE.txt', 'docs/V7_LONG_RUNNING.txt')
GUARD_FILES = ('scripts/v7_guard/lease_enforcer.py', 'scripts/v7_guard/multiwork_enforcer.py', 'scripts/v7_guard/tree_work_enforcer.py', 'scripts/v7_guard/guarded_runtime.py', 'scripts/v7_guard/campaign_guard_entry.py')
FIXED += GUARD_FILES
MAX_FILES, MAX_FILE_BYTES, MAX_PACKAGE_BYTES = 2048, 8 * 1024**2, 64 * 1024**2
LIMITS = {'memory.high': 640 * 1024**2, 'memory.max': 768 * 1024**2,
          'memory.swap.max': 0, 'pids.max': 64}


def digest_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def unique_pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('Duplicate manifest JSON key')
        result[key] = value
    return result


def allowed_name(name):
    path = PurePosixPath(name)
    if (not name or '\\' in name or ':' in name or path.is_absolute()
            or any(part in {'', '.', '..'} for part in name.split('/'))):
        return False
    return (name in FIXED or name == MANIFEST or
            (len(path.parts) >= 2 and path.parts[0] == 'factorlab' and path.suffix == '.py'
             and not any(part.startswith('.') or part == '__pycache__' for part in path.parts)))


def source_manifest(source):
    source = Path(source).resolve(strict=True)
    paths = [*(source / 'factorlab').rglob('*.py'), *(source / item for item in FIXED)]
    records = {}
    for path in sorted(paths):
        relative = path.relative_to(source).as_posix()
        if (not allowed_name(relative) or path.is_symlink() or not path.is_file()
                or not path.resolve().is_relative_to(source)
                or any(parent.is_symlink() for parent in path.parents if parent != source)):
            raise ValueError(f'Unsafe or missing source file: {relative}')
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError('Source file exceeds bounded package size')
        records[relative] = {'sha256': digest_file(path), 'bytes': path.stat().st_size}
    if not (source / 'factorlab/discovery/v7_cli.py').is_file() or len(records) > MAX_FILES:
        raise ValueError('Missing V7 CLI or oversized source list')
    if sum(item['bytes'] for item in records.values()) > MAX_PACKAGE_BYTES:
        raise ValueError('Source package exceeds size cap')
    return {'version': 1, 'kind': 'v7_cpu_source', 'files': records,
            'source_hash': hashlib.sha256(canonical(records)).hexdigest(),
            'excluded': ['credentials', 'data', 'runs', 'work', 'outputs', '.venv', 'caches', 'tests']}


def pack(source, output):
    source, output = Path(source).resolve(strict=True), Path(output).absolute()
    manifest = source_manifest(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as destination, tarfile.open(fileobj=destination, mode='w') as archive:
        for name, record in manifest['files'].items():
            content = (source / name).read_bytes()
            if hashlib.sha256(content).hexdigest() != record['sha256']:
                raise ValueError('Source changed during packaging; discard incomplete package')
            item = tarfile.TarInfo(name)
            item.size, item.mode, item.mtime = len(content), 0o644, 0
            archive.addfile(item, io.BytesIO(content))
        content = canonical(manifest)
        item = tarfile.TarInfo(MANIFEST)
        item.size, item.mode = len(content), 0o644
        archive.addfile(item, io.BytesIO(content))
    receipt = {'package': str(output), 'package_sha256': digest_file(output),
               'source_hash': manifest['source_hash'], 'files': len(manifest['files'])}
    with output.with_suffix(output.suffix + '.sha256.json').open('x', encoding='utf-8') as handle:
        json.dump(receipt, handle, sort_keys=True, indent=2)
    return receipt


def verify_package(package, expected_hash):
    package = Path(package)
    if not re.fullmatch('[0-9a-f]{64}', expected_hash) or digest_file(package) != expected_hash:
        raise ValueError('Source package transport hash mismatch')
    if package.stat().st_size > MAX_PACKAGE_BYTES + 4 * 1024**2:
        raise ValueError('Tar package too large')
    with tarfile.open(package, mode='r:') as archive:
        entries, size = {}, 0
        for item in archive:
            if (not item.isfile() or item.issym() or item.islnk() or not allowed_name(item.name)
                    or item.name in entries or not 0 <= item.size <= MAX_FILE_BYTES
                    or len(entries) >= MAX_FILES + 1):
                raise ValueError('Unsafe, duplicate, special or oversized tar entry')
            size += item.size
            if size > MAX_PACKAGE_BYTES:
                raise ValueError('Tar expansion exceeds bounded size')
            entries[item.name] = archive.extractfile(item).read()
    if MANIFEST not in entries:
        raise ValueError('Package has no source manifest')
    manifest = json.loads(entries.pop(MANIFEST), object_pairs_hook=unique_pairs)
    if (manifest.get('kind') != 'v7_cpu_source' or manifest.get('version') != 1
            or set(manifest['files']) != set(entries)
            or not set(FIXED) <= set(entries)
            or manifest['source_hash'] != hashlib.sha256(canonical(manifest['files'])).hexdigest()):
        raise ValueError('Package source manifest is incomplete or inconsistent')
    for name, content in entries.items():
        if manifest['files'][name] != {'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}:
            raise ValueError('Packaged source content does not match its manifest')
    return manifest, entries


def workspace_path(value, *, must_exist=False):
    path = Path(value)
    if (not path.is_absolute() or path.parent == path or len(path.parts) < 3
            or any('long80' in part.lower() for part in path.parts)
            or path.resolve() != path
            or any(parent.is_symlink() for parent in [path, *path.parents])):
        raise ValueError('Use a new dedicated absolute V7 workspace, without symlink or long80 ancestry')
    if must_exist and not path.is_dir():
        raise ValueError('Workspace is absent')
    return path


def unpack(package, expected_hash, workspace):
    root = workspace_path(workspace)
    manifest, entries = verify_package(package, expected_hash)
    if root.exists():
        raise ValueError('Refuse to overwrite an existing workspace')
    root.mkdir(parents=False, exist_ok=False)
    # No links or executable tar metadata are honored. A failure leaves an
    # incomplete new workspace for inspection; it never deletes an old path.
    for name, content in entries.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Extraction path escaped workspace')
        with path.open('xb') as stream:
            stream.write(content)
    (root / MANIFEST).write_bytes(canonical(manifest))
    return {'workspace': str(root), 'source_hash': manifest['source_hash']}


def systemctl(scope, *args):
    command = ['systemctl', *(['--user'] if scope == 'user' else []), *args]
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=15)
    if result.returncode:
        raise RuntimeError('Cannot inspect systemd; do not bypass the preflight')
    return result.stdout


def _old_service_identity(values, old_unit, scope, receipt_path=None):
    """Use actual ExecStart; an empty inactive transient needs a frozen prior receipt."""
    executable = values.get('ExecStart', '').strip()
    if not executable:
        if receipt_path is None:
            raise RuntimeError('Empty old ExecStart requires a frozen operator service receipt')
        receipt = json.loads(Path(receipt_path).read_text('utf-8'), object_pairs_hook=unique_pairs)
        fields = {key: receipt.get(key) for key in ('unit', 'scope', 'exec_start')}
        if (set(receipt) != {'unit', 'scope', 'exec_start', 'sha256'}
                or fields['unit'] != old_unit or fields['scope'] != scope
                or receipt['sha256'] != hashlib.sha256(canonical(fields)).hexdigest()):
            raise ValueError('Frozen old service receipt identity/hash mismatch')
        executable = fields['exec_start']
    if (not isinstance(executable, str)
            or not re.search(r'/tools/long80/supervise_iteration[.]py(?:\s|[;}\"]|$)', executable)):
        raise RuntimeError('Old service ExecStart is not the verified long80 supervisor')
    return {'unit': old_unit, 'scope': scope, 'exec_start': executable}


def preflight(workspace, old_unit, scope='user', old_service_receipt=None):
    if platform.system() != 'Linux' or sys.version_info[:2] != (3, 12):
        raise ValueError('Runtime requires Linux CPython 3.12')
    if platform.machine() not in {'x86_64', 'AMD64'}:
        raise ValueError('This frozen wheel recipe targets Linux x86_64')
    libc, version = platform.libc_ver()
    if libc != 'glibc' or tuple(map(int, version.split('.')[:2])) < (2, 28):
        raise ValueError('PyArrow wheel requires Linux glibc >= 2.28')
    root = workspace_path(workspace, must_exist=True)
    if not re.fullmatch('[A-Za-z0-9_.@-]+[.]service', old_unit):
        raise ValueError('Supply the exact existing long80 service unit')
    status = systemctl(scope, 'show', old_unit, '--property=LoadState', '--property=ActiveState', '--property=ExecStart')
    values = dict(line.split('=', 1) for line in status.splitlines() if '=' in line)
    if values.get('LoadState') != 'loaded' or values.get('ActiveState') not in {'inactive', 'failed'}:
        raise RuntimeError('Old long80 service is active, transitional or unverified; no automatic stop')
    service_identity = _old_service_identity(values, old_unit, scope, old_service_receipt)
    free = shutil.disk_usage(root).free
    if free < 4 * 1024**3:
        raise RuntimeError('Need at least 4 GiB free space before install/run')
    if not (Path('/sys/fs/cgroup') / 'cgroup.controllers').is_file():
        raise RuntimeError('cgroup v2 required for shared memory/CPU acceptance')
    manifest = json.loads((root / MANIFEST).read_text('utf-8'))
    actual = source_manifest(root)
    if actual != manifest:
        raise RuntimeError('Workspace source differs from the sealed package')
    return {'old_service': values, 'old_service_identity': service_identity,
            'disk_free_bytes': free, 'source_hash': manifest['source_hash'],
            'python': sys.version, 'platform': platform.platform()}


def contained_environment(root):
    root = Path(root)
    directories = {'TMPDIR': root / 'tmp', 'TEMP': root / 'tmp', 'TMP': root / 'tmp',
                   'PIP_CACHE_DIR': root / '.cache/pip', 'XDG_CACHE_HOME': root / '.cache',
                   'MPLCONFIGDIR': root / '.cache/matplotlib', 'NUMBA_CACHE_DIR': root / '.cache/numba'}
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for name in ('PYTHONPATH', 'PYTHONHOME', 'PIP_EXTRA_INDEX_URL', 'PIP_INDEX_URL'):
        env.pop(name, None)
    env.update({k: str(v) for k, v in directories.items()})
    env.update(FACTOR_RESEARCH_ROOT=str(root), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
               PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK='1')
    return env


def neural_runtime_lock(root):
    """Validate the dedicated CPU wheel contract before creating an environment."""
    root = Path(root)
    lock = json.loads((root / 'config/v7-linux-neural-lock.json').read_bytes(), object_pairs_hook=unique_pairs)
    required = root / 'config/v7-linux-neural-requirements.txt'
    if (lock.get('version') != 1 or lock.get('kind') != 'v7-linux-cpu-neural-runtime'
            or lock.get('target') != {'python': '3.12', 'implementation': 'cpython',
                                      'platform': 'linux_x86_64', 'glibc_minimum': '2.28'}
            or digest_file(required) != lock.get('requirements_sha256')
            or digest_file(root / 'config/v7-linux-requirements.txt') != lock.get('baseline_requirements_sha256')):
        raise ValueError('CPU neural runtime requirements/target changed')
    packages = lock.get('packages')
    if not isinstance(packages, list) or not 8 <= len(packages) <= 64:
        raise ValueError('Invalid bounded CPU dependency inventory')
    names, expected = set(), []
    for item in packages:
        name = item['name']
        if (not re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*', name) or name in names
                or name.startswith(('nvidia-', 'cuda', 'triton'))
                or not re.fullmatch('[0-9a-f]{64}', item['sha256'])
                or type(item['bytes']) is not int or not 0 < item['bytes'] <= 512 * 1024**2
                or not re.fullmatch('[0-9A-Za-z.+-]+', item['version'])
                or '/' in item['wheel'] or not item['wheel'].endswith('.whl')
                or unquote(urlparse(item['url']).path.rsplit('/', 1)[-1]) != item['wheel']
                or not (item['url'].startswith('https://files.pythonhosted.org/packages/')
                        or item['url'].startswith('https://download-r2.pytorch.org/whl/cpu/'))
                or not isinstance(item['active_requires_dist'], list)):
            raise ValueError('Invalid CPU-only locked wheel identity')
        names.add(name)
        requirement = 'torch @ ' + item['url'] if name == 'torch' else name + '==' + item['version']
        expected.append(requirement + ' --hash=sha256:' + item['sha256'])
    torch = next((item for item in packages if item['name'] == 'torch'), None)
    if (torch is None or torch['version'] != '2.11.0+cpu'
            or 'cp312-cp312-manylinux_2_28_x86_64.whl' not in torch['wheel']
            or not torch['url'].startswith('https://download-r2.pytorch.org/whl/cpu/')):
        raise ValueError('Exact Linux CPython 3.12 CPU Torch wheel required')
    lines = [line.strip() for line in required.read_text().splitlines()
             if line.strip() and not line.startswith('#')]
    if lines != ['--require-hashes', '--only-binary=:all:', *expected]:
        raise ValueError('CPU neural requirements differ from locked wheel inventory')
    pins = {item['name']: item['version'] for item in packages}
    for line in (root / 'config/v7-linux-requirements.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        name, version = line.strip().split('==')
        if pins.get(name.lower().replace('_', '-')) != version:
            raise ValueError('CPU neural profile must retain every original convex dependency pin')
    if (lock['total_wheel_bytes'] != sum(item['bytes'] for item in packages)
            or lock['total_wheel_bytes'] > 1024**3):
        raise ValueError('CPU neural wheel size inventory changed')
    return lock


def bootstrap(workspace, old_unit, scope, old_service_receipt=None, *, neural=False):
    receipt = preflight(workspace, old_unit, scope, old_service_receipt)
    root = workspace_path(workspace, must_exist=True)
    if neural:
        neural_runtime_lock(root)
    env = contained_environment(root)
    venv = root / '.venv'
    if venv.exists():
        raise ValueError('Do not overwrite an existing environment; create a new workspace')
    subprocess.run([sys.executable, '-m', 'venv', str(venv)], env=env, check=True, timeout=120)
    python = venv / 'bin/python'
    report = root / 'runtime/install-report.json'
    report.parent.mkdir(exist_ok=True)
    requirements = 'v7-linux-neural-requirements.txt' if neural else 'v7-linux-requirements.txt'
    subprocess.run([str(python), '-m', 'pip', '--isolated', 'install', '--index-url', 'https://pypi.org/simple',
                    '--cache-dir', str(root / '.cache/pip'),
                    '--only-binary=:all:', '--no-deps', '--requirement', str(root / 'config' / requirements),
                    '--report', str(report)], env=env, cwd=root, check=True, timeout=1200)
    subprocess.run([str(python), '-m', 'pip', 'check'], env=env, cwd=root, check=True, timeout=30)
    code = ('import sys,json,importlib.metadata as m; '
            'from factorlab.discovery.program_contract import Program,program_language_spec; '
            'from factorlab.discovery import v7_run,v7_forward; '
            'Program.from_dict(program_language_spec()["example"]); '
            'assert "torch" not in sys.modules; '
            'print(json.dumps({d.metadata["Name"]:d.version for d in m.distributions()},sort_keys=True))')
    packages = subprocess.run([str(python), '-B', '-c', code], env=env, cwd=root,
                              check=True, capture_output=True, text=True, timeout=60)
    if neural:
        # Separate process: normal registry/service imports remain Torch-free.
        subprocess.run([str(python), '-B', '-c',
            ('import torch; assert torch.__version__ == "2.11.0+cpu"; '
            'assert torch.version.cuda is None; assert torch.version.hip is None; '
            'torch.set_num_threads(1); torch.set_num_interop_threads(1); '
            'x=torch.tensor([1.,2.],requires_grad=True); (x*x).sum().backward(); '
            'assert x.grad.tolist()==[2.,4.]')],
            env=env, cwd=root, check=True, timeout=60)
    receipt.update(packages=json.loads(packages.stdout), install_report_sha256=digest_file(report),
                   runtime_profile='cpu_neural_locked' if neural else 'convex_cpu',
                   neural_lock_sha256=digest_file(root / 'config/v7-linux-neural-lock.json') if neural else None,
                   requirements_sha256=digest_file(root / 'config' / requirements),
                   runtime_status='import_smoke_only_not_resource_acceptance')
    (root / 'runtime/environment.json').write_bytes(canonical(receipt))
    return receipt


def resource_snapshot(cgroup=None, *, enforce=True):
    if cgroup is None:
        rows = Path('/proc/self/cgroup').read_text().splitlines()
        match = next((r.split(':', 2)[2] for r in rows if r.startswith('0::')), None)
        if match is None:
            raise RuntimeError('Cannot locate unified process cgroup')
        cgroup = Path('/sys/fs/cgroup') / match.lstrip('/')
    root = Path(cgroup)
    values = {name: (root / name).read_text().strip() for name in
              (*LIMITS, 'cpu.max', 'memory.current', 'memory.peak', 'memory.events', 'cpu.stat', 'pids.current')}
    if enforce:
        for name, cap in LIMITS.items():
            value = values[name]
            if value == 'max' or not value.isdigit() or int(value) > cap:
                raise RuntimeError(f'Effective {name} exceeds the frozen resource envelope')
        quota, period = values['cpu.max'].split()
        if quota == 'max' or int(period) <= 0 or int(quota) <= 0 or int(quota) / int(period) > .25:
            raise RuntimeError('Effective CPU quota exceeds 25%')
    return {'cgroup': str(root), 'values': values, 'limits_checked': enforce}



def guarded_command(root, command):
    """Production unit executes only the frozen whole-service guardian."""
    prefix = [str(root / '.venv/bin/python'), '-B',
              str(root / 'scripts/v7_guard/campaign_guard_entry.py'), 'guard']
    if len(command) != 6 or list(map(str, command[:4])) != prefix:
        raise ValueError('Exact fixed guarded campaign entry required')
    campaign = Path(command[4]).resolve(strict=True)
    guard = Path(command[5]).resolve()
    if (not campaign.is_relative_to(root) or guard != campaign / 'whole_tree_guard'
            or guard.exists() or not (campaign / 'campaign.json').is_file()
            or not (campaign / 'campaign.sqlite').is_file()):
        raise ValueError('Existing frozen campaign and unused canonical guard path required')
    for name in GUARD_FILES:
        path = root / name
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('Missing or unsafe fixed guard source')
    return campaign


def systemd_template(workspace, old_unit, scope, command, old_service_receipt=None):
    root = workspace_path(workspace, must_exist=True)
    guarded_command(root, command)
    if not re.fullmatch('[A-Za-z0-9_.@-]+[.]service', old_unit):
        raise ValueError('Invalid old unit identity')

    def quote(value):
        if not str(value).isprintable():
            raise ValueError('Invalid systemd argument')
        return '"' + str(value).replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%') + '"'

    def working_directory(value):
        value = value.as_posix()
        if not value or not value.isprintable() or any(c in value for c in '\\\"\''):
            raise ValueError('Invalid systemd working directory')
        # WorkingDirectory is a raw path with specifiers, not command argv.
        # The same workspace is an executable path: quotes/backslashes above
        # are rejected by systemd regardless of argv escaping.
        if value[-1].isspace():
            value += '/.'  # Preserve final whitespace through INI parsing.
        return value.replace('%', '%%')

    executable = [(root / '.venv/bin/python').as_posix(), (root / 'scripts/v7_linux_bootstrap.sh').as_posix(),
                  'guard-exec', '--workspace', root.as_posix(), '--old-unit', old_unit, '--scope', scope]
    if old_service_receipt is not None:
        executable += ['--old-service-receipt', str(Path(old_service_receipt).resolve())]
    executable += ['--', *command]
    lines = ['[Unit]', 'Description=Frozen V7 CPU research (no trading)', '', '[Service]', 'Type=exec',
             'WorkingDirectory=' + working_directory(root), 'ExecStart=:' + ' '.join(map(quote, executable)),
             'ExecStopPost=:' + ' '.join(map(quote, [(root / '.venv/bin/python').as_posix(),
                                                  (root / 'scripts/v7_guard/campaign_guard_entry.py').as_posix(),
                                                  'postmortem', command[4], command[5]])),
             'CPUAccounting=yes', 'MemoryAccounting=yes', 'TasksAccounting=yes', 'CPUQuota=25%',
             'MemoryHigh=640M', 'MemoryMax=768M', 'MemorySwapMax=0', 'TasksMax=64',
             'RuntimeMaxSec=7200', 'TimeoutStopSec=20', 'KillMode=control-group', 'Restart=no',
             'NoNewPrivileges=yes', 'UMask=0077', 'StandardOutput=journal', 'StandardError=journal',
             'Environment=OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1',
             '', '# Template only: inspect before installing; no automatic restart or service installation.']
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    package = commands.add_parser('pack')
    package.add_argument('--source', required=True)
    package.add_argument('--output', required=True)
    for action in ('verify', 'unpack'):
        sub = commands.add_parser(action)
        sub.add_argument('--package', required=True)
        sub.add_argument('--sha256', required=True)
        if action == 'unpack':
            sub.add_argument('--workspace', required=True)
    for action in ('preflight', 'bootstrap', 'unit-template', 'guard-exec'):
        sub = commands.add_parser(action)
        sub.add_argument('--workspace', required=True)
        sub.add_argument('--old-unit', required=True)
        sub.add_argument('--scope', choices=('user', 'system'), default='user')
        sub.add_argument('--old-service-receipt')
        if action == 'bootstrap':
            sub.add_argument('--neural', action='store_true', help='Install hash-locked CPU Torch profile; no training/startup')
        if action in ('unit-template', 'guard-exec'):
            sub.add_argument('command', nargs=argparse.REMAINDER)
    commands.add_parser('resources')
    args = parser.parse_args(argv)
    if args.action == 'pack':
        result = pack(args.source, args.output)
    elif args.action == 'verify':
        result = verify_package(args.package, args.sha256)[0]
    elif args.action == 'unpack':
        result = unpack(args.package, args.sha256, args.workspace)
    elif args.action == 'preflight':
        result = preflight(args.workspace, args.old_unit, args.scope, args.old_service_receipt)
    elif args.action == 'bootstrap':
        result = bootstrap(args.workspace, args.old_unit, args.scope, args.old_service_receipt, neural=args.neural)
    elif args.action == 'resources':
        result = resource_snapshot()
    else:
        command = args.command[1:] if args.command[:1] == ['--'] else args.command
        if args.action == 'unit-template':
            print(systemd_template(args.workspace, args.old_unit, args.scope, command, args.old_service_receipt), end='')
            return 0
        preflight(args.workspace, args.old_unit, args.scope, args.old_service_receipt)
        resource_snapshot()
        root = workspace_path(args.workspace, must_exist=True)
        guarded_command(root, command)
        os.chdir(root)
        os.execve(command[0], command, contained_environment(root))
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
