"""Parse the actual project Codex argument vector using --help; no inference."""
import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

root = Path(os.environ['FACTOR_RESEARCH_ROOT']).resolve(strict=True)
sys.path.insert(0, str(root))
from factorlab.discovery.agent_worker import ResearchWorker, WorkerConfig

directory = root / 'work/deployment' / ('codex-probe-' + uuid4().hex[:10])
directory.mkdir(parents=True)
worker = ResearchWorker(WorkerConfig(exchange_dirs=(directory / 'exchange',),
    state_dir=directory / 'state', backend='codex', model='gpt-6-astra', reasoning_effort='high'))
command = worker._codex_command(directory)
with (directory / 'stdout.txt').open('wb') as out, (directory / 'stderr.txt').open('wb') as err:
    result = subprocess.run(command[:-1] + ['--help'], stdout=out, stderr=err, timeout=30,
                            cwd=root, check=False)
(directory / 'probe.json').write_text(json.dumps({'argv': command[:-1] + ['--help'],
    'returncode': result.returncode,
    'scope': 'argument parsing only; model access/authentication/event enforcement unverified'}, indent=2),
    encoding='utf-8')
print(json.dumps({'directory': str(directory), 'returncode': result.returncode}))
raise SystemExit(result.returncode)
