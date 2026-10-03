"""User entrypoints for persistent research; all state remains in the workspace."""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from uuid import uuid4

import pandas as pd
import torch

from factorlab.protocol import project_path, utc

from .contracts import ModuleSpec, RepresentationSpec
from .evidence import EvidenceRegistry, immutable_json, verify_bundle
from .evolution import EvolutionPolicy, EvolutionStore, _architecture, _causal_model, run_evolution
from .experiment import ModelGroup
from .learning import FittedResearchModel, fit
from .replay import _training_parts


def add_arguments(parser):
    parser.add_argument('--output', required=True)
    parser.add_argument('--policy', default='config/evolution.json')
    parser.add_argument('--mode', choices=['fixed', 'refit', 'evolve'])
    parser.add_argument('--search', choices=['random', 'diagnostic', 'external'])
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--initial-run', help='Use a saved search bundle with verified past-only training')
    parser.add_argument('--initial-checkpoint', help='Load the same saved initial weights for paired modes')
    parser.add_argument('--initialize-only', action='store_true', help='Save a causal common initial model, without replay')
    parser.add_argument('--initial-model', choices=['accepted', 'candidate', 'baseline'], default='accepted')
    parser.add_argument('--max-ticks', type=int)
    parser.add_argument('--as-of', help='Optional causal processing cutoff')
    parser.add_argument('--cross-assets', action='store_true')
    parser.add_argument('--funding', help='Point-in-time funding Parquet/CSV table')
    parser.add_argument('--open-interest', help='Point-in-time open-interest Parquet/CSV table')
    parser.add_argument('--context-max-age-hours', type=float, default=24)
    parser.add_argument('--memory', default='data/research_memory.sqlite')
    parser.add_argument('--watch', action='store_true', help='Reload an append-only bars.parquet feed and resume')
    parser.add_argument('--refresh-feed', action='store_true', help='Fetch bounded public closed candles into an explicit live feed')
    parser.add_argument('--watch-seconds', type=int, default=3600)
    parser.add_argument('--poll-seconds', type=int, default=30)
    parser.add_argument('--worker-backend', choices=['rules', 'codex'], help='Start a bounded proposal consumer')


def _data(args, spec):
    from factorlab.data import validate_klines

    from .context_data import build_context_dataset, read_context_table
    from .tasks import build_task_dataset
    bars_path = project_path(args.data) / 'bars.parquet'
    if (bars_path.parent / 'feed_manifest.json').exists():
        from .feed import load_feed
        bars = load_feed(bars_path.parent)
    else:
        bars = pd.read_parquet(bars_path)
    validate_klines(bars)
    if args.cross_assets or args.funding or args.open_interest:
        data = build_context_dataset(bars, spec,
            funding=read_context_table(args.funding, 'funding') if args.funding else None,
            open_interest=read_context_table(args.open_interest, 'open_interest') if args.open_interest else None,
            cross_assets=args.cross_assets, max_age_hours=args.context_max_age_hours)
    else:
        data = build_task_dataset(bars, spec)
    return data


def _initial(args, data, spec):
    if args.resume:
        return None
    if getattr(args, 'initial_checkpoint', None):
        if args.initial_run:
            raise ValueError('Choose one initial model source')
        path = project_path(args.initial_checkpoint)
        return ModelGroup.load(path) if path.is_dir() else FittedResearchModel.load(path)
    if args.initial_run:
        folder = project_path(args.initial_run)
        claim = json.loads((folder / 'frozen_claim.json').read_text('utf-8'))
        verify_bundle(folder / 'bundle', claim)
        result = json.loads((folder / 'results.json').read_text('utf-8'))
        name = args.initial_model
        if name == 'accepted':
            selection_cutoff = claim['protocol']['end'] if result.get('confirmation') else claim['protocol']['dev_end']
            if utc(selection_cutoff) > utc(spec.train_end):
                raise ValueError('Accepted-model selection used evidence after this evolution start')
            name = 'candidate' if (result.get('confirmation') or {}).get('accepted_system') else 'baseline'
        return ModelGroup.load(folder / 'bundle' / name)
    train, dev, _ = _training_parts(data, utc(spec.train_end))
    structure = RepresentationSpec((ModuleSpec('temporal', scale=min(12, spec.lookback)),
                                    ModuleSpec('interaction', 'mlp', scale=min(12, spec.lookback))))
    from dataclasses import replace
    return ModelGroup([fit(train, dev, structure, replace(spec.training, seed=seed), args.device)
                       for seed in spec.training_seeds])


def run_cli(args, spec):
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; choose cpu')
    torch.set_num_threads(4)
    if not 1 <= args.poll_seconds <= 60 or args.watch_seconds < 1:
        raise ValueError('Watch needs a positive budget and 1..60 second polling')
    value = json.loads(project_path(args.policy).read_text('utf-8-sig'))
    if args.mode:
        value['mode'] = args.mode
    if args.search:
        value['search_mode'] = args.search
    policy = EvolutionPolicy(**value)
    if args.worker_backend and policy.search_mode != 'external':
        raise ValueError('A proposal worker requires --search external')
    if args.refresh_feed:
        from .feed import refresh_feed
        refresh_feed(args.data, spec.symbols)
    data = _data(args, spec)
    # Reserve research exposure before any initial fitting, including failures.
    registry = EvidenceRegistry()
    registry.import_prior_runs()
    registry.consume(source='binance_um_1h', symbols=spec.symbols,
        start=(utc(spec.start) - pd.Timedelta(hours=spec.lookback + 49)).isoformat(), end=spec.dev_end,
        run=str(project_path(args.output)), claim='evolution-cli-' + spec.digest, role='development')
    initial = _initial(args, data, spec)
    if args.initialize_only:
        if initial is None:
            raise ValueError('Initialization cannot use --resume')
        _causal_model(initial, data, spec.train_end)
        folder = project_path(args.output)
        folder.mkdir(parents=True, exist_ok=False)
        path = folder / ('initial-model' if isinstance(initial, ModelGroup) else 'initial-model.pt')
        initial.save(path)
        result = {'checkpoint': str(path), 'digest': initial.digest, 'schema': initial.schema,
                  'cutoff': spec.train_end, 'protocol': asdict(spec)}
        immutable_json(folder / 'initial.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result
    process = None
    log = None
    stop_file = None
    if args.worker_backend:
        import subprocess
        import sys
        folder = project_path(args.output)
        folder.mkdir(parents=True, exist_ok=True)
        stop_file = folder / 'worker' / f'stop-{uuid4().hex}.json'
        log = (folder / 'worker.log').open('ab')
        command = [sys.executable, '-m', 'factorlab.discovery.agent_worker',
            '--state-dir', str(folder / 'worker'),
            '--backend', args.worker_backend, '--model', 'gpt-6.1-sol',
            '--stop-file', str(stop_file),
            '--max-requests', str(policy.max_cycles * max(16, 8 * spec.candidates) * 3),
            '--max-seconds', str(max(args.watch_seconds, policy.max_cycles * spec.search_seconds + 60))]
        for cycle_id in range(1, policy.max_cycles + 1):
            command += ['--exchange', str(folder / 'research' / f'cycle-{cycle_id:04d}' / 'exchange')]
        try:
            process = subprocess.Popen(command, cwd=project_path('.'), stdout=log, stderr=log,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except (OSError, ValueError):
            log.close()
            raise
    deadline = time.monotonic() + args.watch_seconds
    resume = args.resume
    try:
        while True:
            result = run_evolution(data, initial, spec, args.output, policy=policy, device=args.device,
                resume=resume, as_of=args.as_of, max_ticks=args.max_ticks, memory_path=args.memory)
            folder = project_path(args.output)
            immutable_json(folder / 'snapshots' / f"summary-{result['ticks']:07d}-{result['event_head'][:12]}.json", result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if (not args.watch or time.monotonic() >= deadline
                    or utc(result['last_timestamp']) >= utc(spec.dev_end)):
                break
            if process is not None and process.poll() is not None:
                raise RuntimeError('Proposal worker exited; inspect the persisted worker diagnostics')
            time.sleep(min(args.poll_seconds, max(0, deadline - time.monotonic())))
            if args.refresh_feed:
                from .feed import refresh_feed
                refresh_feed(args.data, spec.symbols)
            data, initial, resume = _data(args, spec), None, True
        return result
    finally:
        try:
            if process is not None:
                from .agent_worker import stop_worker
                stop_worker(process, stop_file, timeout=10)
        finally:
            if log is not None:
                log.close()


def diagnostics_cli(directory, data, output):
    from .diagnostics import representation_diagnostics
    store = EvolutionStore(directory)
    try:
        state = store.state()
        model = store.load_model(state['incumbent'])
        result = representation_diagnostics(model, data, cutoff=state['last_timestamp'])
        result['architecture'] = asdict(_architecture(model)) if _architecture(model) else None
        immutable_json(project_path(output), result)
        return result
    finally:
        store.close()
