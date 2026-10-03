"""Stage a reviewed Linux profile on a NEW deployment copy, never on Windows.

Default: check and write proposed files beneath work/linux_migration/staged.
Use --apply on the Linux copy before creating any new research run.
This is a one-time migration, not an updater or a resume-identity converter.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import tomllib
from pathlib import Path

RUNTIME = '''"""Linux deployment controls; versioned with research source."""
import os
import signal


def torch_threads():
    value = int(os.environ.get('FACTOR_RESEARCH_TORCH_THREADS', '2'))
    if not 1 <= value <= 9:
        raise ValueError('FACTOR_RESEARCH_TORCH_THREADS must be in 1..9')
    return value


def install_stop_handler():
    def stop(signum, frame):
        # A second TERM must not interrupt finally cleanup.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt('Deployment stop requested')
    signal.signal(signal.SIGTERM, stop)


def terminate_posix_tree(process):
    """Cancel the known descendant tree; the outer group is the final backstop."""
    import psutil
    try:
        parent = psutil.Process(process.pid)
        victims = parent.children(recursive=True) + [parent]
    except psutil.NoSuchProcess:
        return
    for victim in victims:
        try:
            victim.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(victims, timeout=4)
    process.wait(timeout=5)
'''


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Expected exactly one patch anchor: {old[:100]!r}')
    return source.replace(old, new, 1)


def proposed(root: Path):
    paths = ['factorlab/protocol.py', 'factorlab/data.py',
             'factorlab/experiment.py', 'factorlab/discovery/__main__.py',
             'factorlab/discovery/evolution_cli.py', 'factorlab/discovery/agent_worker.py',
             'pyproject.toml', '.python-version', 'AGENTS.md']
    before = {p: (root / p).read_bytes() for p in paths}
    after = {p: b.decode('utf-8-sig').replace('\r\n', '\n') for p, b in before.items()}
    p = 'factorlab/protocol.py'
    after[p] = replace_once(after[p], 'import math\n', 'import math\nimport os\n')
    after[p] = replace_once(after[p], 'from pathlib import Path\n',
                            'from pathlib import Path, PureWindowsPath\n')
    after[p] = replace_once(after[p], "ROOT = Path('D:/FactorResearch')", '''_configured_root = os.environ.get('FACTOR_RESEARCH_ROOT')
if _configured_root:
    ROOT = Path(_configured_root).expanduser()
    if not ROOT.is_absolute():
        raise ValueError('FACTOR_RESEARCH_ROOT must be absolute')
    ROOT = ROOT.resolve()
else:
    ROOT = Path(__file__).resolve().parents[1]''')
    after[p] = replace_once(after[p], '    value = Path(path)\n',
        "    if os.name != 'nt' and PureWindowsPath(str(path)).drive:\n"
        "        raise ValueError('Windows drive paths require an explicit migration mapping')\n"
        '    value = Path(path)\n')
    p = 'factorlab/data.py'
    after[p] = replace_once(after[p], "root: Path = Path('D:/FactorResearch/data')",
                            "root: Path = Path('data')")
    for p in ['factorlab/experiment.py', 'factorlab/discovery/__main__.py',
              'factorlab/discovery/evolution_cli.py']:
        after[p] = replace_once(after[p], 'import torch\n',
            'import torch\n\nfrom factorlab.deployment_runtime import torch_threads\n')
        after[p] = replace_once(after[p], 'torch.set_num_threads(4)',
                                'torch.set_num_threads(torch_threads())')
        after[p] = after[p].replace(
            'from factorlab.deployment_runtime import torch_threads\n\nfrom factorlab.protocol',
            'from factorlab.deployment_runtime import torch_threads\nfrom factorlab.protocol')
    for p in ['factorlab/discovery/__main__.py', 'factorlab/discovery/agent_worker.py']:
        after[p] = replace_once(after[p], 'def main(argv=None):\n',
            'def main(argv=None):\n'
            '    from factorlab.deployment_runtime import install_stop_handler\n'
            '    install_stop_handler()\n')
    p = 'factorlab/discovery/evolution_cli.py'
    after[p] = replace_once(after[p], "'--backend', args.worker_backend, '--model', 'gpt-6.1-sol',",
        "'--backend', args.worker_backend, '--model', 'gpt-6-astra',\n"
        "            '--reasoning-effort', 'high',")
    p = 'factorlab/discovery/agent_worker.py'
    after[p] = replace_once(after[p], "    if os.name == 'nt':\n",
        "    if os.name != 'nt':\n"
        '        from factorlab.deployment_runtime import terminate_posix_tree\n'
        '        terminate_posix_tree(process)\n'
        '        return\n'
        "    if os.name == 'nt':\n")
    p = 'pyproject.toml'
    for old, new in [
        ('torch==2.11.0+cu128', 'torch==2.11.0+cpu'),
        ("sys_platform == 'win32' and platform_machine == 'AMD64'",
         "sys_platform == 'linux' and platform_machine == 'x86_64'"),
        ('https://download.pytorch.org/whl/cu128', 'https://download.pytorch.org/whl/cpu'),
        ('  "filelock>=3.16,<3.30",', '  "filelock>=3.16,<3.30",\n  "psutil==7.2.2",'),
    ]:
        after[p] = replace_once(after[p], old, new)
    if after[p].count('pytorch-cu128') != 2:
        raise ValueError('Unexpected torch index/source count')
    after[p] = after[p].replace('pytorch-cu128', 'pytorch-cpu')
    tomllib.loads(after[p])
    after['.python-version'] = '3.12.14\n'
    after['AGENTS.md'] = '''# Linux deployment copy instructions

This is a new deployment identity copied from D:/FactorResearch. The historical
Windows workspace notes below are retained as provenance. For operations in this
Linux copy, these instructions supersede the Windows path, PowerShell, CUDA,
executable-path and old research-agent-model instructions below.

- Use the absolute FACTOR_RESEARCH_ROOT selected on the project overlay.
- Source outputs/dot-deployment-kit/environment.sh in Bash before project commands.
- Use .venv/bin/python; keep project data, temporary files and caches in this root.
- The deployed research proposer uses gpt-6-astra with high reasoning.
- Preserve the operator's existing HOME and CODEX_HOME and never archive credentials.
- Preserve old evidence, rejected candidates and exposure ledgers. Windows runs
  cannot be resumed by rewriting identities. Create new Linux runs.
- Follow outputs/DOT_DEPLOYMENT_MANUAL.md. Do not claim Linux acceptance, model
  access, persistence or new alpha until the corresponding checks have passed.
- No live exchange trading. Do not relax research gates or enable extra proposer
  capabilities to bypass deployment failures.

## Historical Windows instructions

''' + after['AGENTS.md']
    new = 'factorlab/deployment_runtime.py'
    if (root / new).exists():
        raise ValueError('Deployment runtime already exists; do not patch twice')
    after[new] = RUNTIME
    for name, source in after.items():
        if name.endswith('.py'):
            ast.parse(source, filename=name)
    return before, {p: text.encode('utf-8') for p, text in after.items()}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if args.apply and sys.platform != 'linux':
        parser.error('--apply is permitted only on the new Linux copy')
    base = root / 'work/linux_migration'
    if (base / 'applied.json').exists():
        raise SystemExit('Already applied. Preserve this version; do not apply again.')
    before, after = proposed(root)
    record = {'root': str(root), 'model': 'gpt-6-astra', 'reasoning_effort': 'high',
              'python': '3.12.14', 'files': {p: {'before': digest(before[p]) if p in before else None,
              'after': digest(b)} for p, b in after.items()},
              'next': 'Generate a new Linux uv.lock, install, and run target acceptance.'}
    for name, payload in after.items():
        path = base / 'staged' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    base.mkdir(parents=True, exist_ok=True)
    (base / 'proposed.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    if args.apply:
        original = base / 'original'
        if original.exists():
            raise SystemExit('Original backup already exists: possible partial apply; inspect, do not retry blindly.')
        # Back up both original source and Windows lock before any product write.
        originals = dict(before)
        originals['uv.lock'] = (root / 'uv.lock').read_bytes()
        for name, payload in originals.items():
            path = original / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        for name, payload in before.items():
            if (root / name).read_bytes() != payload:
                raise SystemExit('Source changed during staging; no apply permitted.')
        for name, payload in after.items():
            path = root / name
            temp = path.with_name(path.name + '.linux-profile.tmp')
            with temp.open('xb') as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
        (base / 'applied.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps({'applied': args.apply, 'files': len(after), 'record': str(base)}))


if __name__ == '__main__':
    main()
