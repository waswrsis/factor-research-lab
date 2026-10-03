"""Write NEW bounded CPU configurations. No training and no data download."""
import argparse
import copy
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    args = p.parse_args()
    root = args.root.resolve(strict=True)
    spec = json.loads((root / 'config/discovery_v6.json').read_text('utf-8-sig'))
    policy = json.loads((root / 'config/evolution_v6.json').read_text('utf-8-sig'))
    full = copy.deepcopy(spec)
    full['version'] = 'discovery-v6-linux-cpu-astra-high-v1'
    full['training']['batch_size'] = 128
    full['search_seconds'] = 1800
    smoke = copy.deepcopy(full)
    smoke.update(version='discovery-v6-linux-deployment-smoke-v1', candidates=2,
                 train_end='2025-03-01', dev_end='2025-04-05',
                 calibration_end='2025-05-01', end='2025-06-01')
    smoke['training']['epochs'] = 4
    small_policy = dict(policy, max_cycles=1, population_size=2)
    files = {'config/discovery_dot_smoke.json': smoke,
             'config/evolution_dot_smoke.json': small_policy,
             'config/discovery_dot_cpu.json': full,
             'config/evolution_dot_cpu.json': policy}
    if any((root / name).exists() for name in files):
        raise SystemExit('Destination configuration already exists; preserve it, do not overwrite.')
    for name, value in files.items():
        (root / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'created': list(files), 'note': '2025 is exposed development data; not a new benchmark.'}))


if __name__ == '__main__':
    main()
