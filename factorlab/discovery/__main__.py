"""Local entry point: python -m factorlab.discovery --help."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from factorlab.protocol import project_path, utc

from .contracts import ExperimentSpec, partition
from .evidence import EvidenceRegistry, file_hash, immutable_json, verify_bundle
from .experiment import ModelGroup, run_experiment
from .learning import fit
from .tasks import build_task_dataset, prepare_bars


def load_spec(path):
    value = json.loads(project_path(path).read_text('utf-8-sig'))
    return ExperimentSpec.from_dict(value.get('protocol', value))


def read_bars(directory):
    directory = project_path(directory)
    manifest = json.loads((directory / 'bars_manifest.json').read_text('utf-8'))
    # Schema owned by tasks.prepare_bars; reuse validates all source archives too.
    return directory, manifest


def causal_initial(data, spec, architecture, device):
    """Initial replay selection uses history strictly before the replay starts."""
    past = partition(data, spec, 'train')
    times = sorted(past.meta.available_at.unique())
    if len(times) < 64:
        raise ValueError('Insufficient history for chronological initial-model selection')
    split = utc(times[int(len(times) * 0.8)])
    train = past.subset((past.meta.available_at < split).to_numpy())
    for task in train.tasks:
        train.observed[task.name] &= train.task_mask(task.name, split)
    dev = past.subset((past.meta.available_at >= split).to_numpy())
    return fit(train, dev, architecture, spec.training, device)


def main(argv=None):
    parser = argparse.ArgumentParser(description='自适应因子研究：数据、训练、对照、评分与逐步回放')
    parser.add_argument('--config', default='config/discovery.json')
    parser.add_argument('--data', default='data/discovery/2025-hourly')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare', help='准备并校验公开小时数据，只下载所需K线')
    run = sub.add_parser('run', help='训练、结构搜索、固定表示对照；确认须显式选择')
    run.add_argument('--output', required=True)
    run.add_argument('--search', nargs='*', choices=['random', 'diagnostic', 'external'],
                     default=['random', 'diagnostic'])
    run.add_argument('--exchange', help='外部研究agent的请求与提案目录')
    run.add_argument('--confirm', action='store_true', help='消耗预先保留的确认区间，重复使用会被拒绝')
    run.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    replay = sub.add_parser('replay', help='逐小时输出、成熟后评分、按冻结周期更新')
    replay.add_argument('--output', required=True)
    replay.add_argument('--update', choices=['fixed', 'refit'], default='fixed')
    replay.add_argument('--update-every-hours', type=int, default=168)
    replay.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    from .evolution_cli import add_arguments
    add_arguments(sub.add_parser('evolve', help='持续研究、影子比较、结构演化及中断恢复'))
    evolution_inspect = sub.add_parser('inspect-evolution', help='核验持续研究状态与模型产物')
    evolution_inspect.add_argument('run')
    evolution_compare = sub.add_parser('compare-evolution', help='比较同条件固定/重训/演化研究程序')
    evolution_compare.add_argument('--candidate', required=True)
    evolution_compare.add_argument('--reference', required=True)
    evolution_compare.add_argument('--output', required=True)
    evolution_report = sub.add_parser('evolution-report', help='导出已核验的研究评分、事件分布和本地报告')
    evolution_report.add_argument('--run', required=True)
    evolution_report.add_argument('--output', required=True)
    feed_init = sub.add_parser('feed-init', help='将明确的历史种子复制到新的实时采集目录')
    feed_init.add_argument('--seed', required=True)
    feed_init.add_argument('--output', required=True)
    feed_refresh = sub.add_parser('feed-refresh', help='有预算采集公开已收盘小时K线')
    feed_refresh.add_argument('--feed', required=True)
    feed_refresh.add_argument('--symbols', nargs='+', required=True)
    feed_refresh.add_argument('--max-pages', type=int, default=1)
    inspect = sub.add_parser('inspect', help='查看一个已完成研究的主要结论')
    inspect.add_argument('run')
    score = sub.add_parser('score', help='用完整冻结版本对指定资料生成研究评分')
    score.add_argument('--run', required=True)
    score.add_argument('--output', required=True)
    score.add_argument('--model', choices=['accepted', 'candidate', 'baseline'], default='accepted')
    sub.add_parser('evidence', help='查看资料使用记录')
    catalog = sub.add_parser('catalog', help='登记、列出或核验完整因子质量档案')
    catalog.add_argument('--register', help='登记已完成实验目录')
    catalog.add_argument('--verify', help='核验指定档案ID与其冻结文件')
    catalog.add_argument('--export', help='将当前质量档案导出为不可覆盖的JSON')
    controls = sub.add_parser('controls', help='检验已知信号和无信息对照，不使用市场资料')
    controls.add_argument('--output', required=True)
    comparison = sub.add_parser('compare-replay', help='核验并比较同条件固定与更新回放')
    comparison.add_argument('--candidate', required=True)
    comparison.add_argument('--reference', required=True)
    comparison.add_argument('--output', required=True)
    report_parser = sub.add_parser('report', help='为已完成实验生成一次结果页')
    report_parser.add_argument('run')
    args = parser.parse_args(argv)
    if args.command == 'feed-init':
        from .feed import initialize_feed
        print(json.dumps(initialize_feed(args.output, args.seed), ensure_ascii=False))
        return
    if args.command == 'feed-refresh':
        from .feed import refresh_feed
        print(json.dumps(refresh_feed(args.feed, tuple(args.symbols), max_pages=args.max_pages), ensure_ascii=False))
        return
    if args.command == 'evolve':
        from .evolution_cli import run_cli
        run_cli(args, load_spec(args.config))
        return
    if args.command == 'inspect-evolution':
        from .evolution import inspect_evolution
        print(json.dumps(inspect_evolution(args.run), ensure_ascii=False, indent=2))
        return
    if args.command == 'compare-evolution':
        from .compare_evolution import compare_evolutions
        print(json.dumps(compare_evolutions(args.candidate, args.reference, args.output), ensure_ascii=False, indent=2))
        return
    if args.command == 'evolution-report':
        from .evolution_report import export_evolution
        print(json.dumps(export_evolution(args.run, args.output), ensure_ascii=False, indent=2))
        return
    if args.command == 'report':
        from .report import render_report
        print(json.dumps({'report': str(render_report(args.run))}, ensure_ascii=False))
        return
    if args.command == 'compare-replay':
        from .comparison import compare_replays
        result = compare_replays(args.candidate, args.reference, args.output)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    if args.command == 'controls':
        from .controls import run_controls
        result = run_controls(args.output)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if not result['passed']:
            raise RuntimeError('Controlled pipeline checks failed')
        return
    if args.command == 'evidence':
        print(json.dumps(EvidenceRegistry().records(), indent=2, ensure_ascii=False))
        return
    if args.command == 'catalog':
        from .catalog import QualityCatalog
        library = QualityCatalog()
        if args.register:
            value = library.register(args.register)
        elif args.verify:
            value = library.verify(args.verify)
        elif args.export:
            value = {'export': str(library.export(args.export))}
        else:
            value = library.list()
        print(json.dumps(value, indent=2, ensure_ascii=False))
        return
    if args.command == 'inspect':
        result = json.loads((project_path(args.run) / 'results.json').read_text('utf-8'))
        print(json.dumps({k: result.get(k) for k in
                          ('status', 'selection', 'confirmation', 'limitations')},
                         indent=2, ensure_ascii=False))
        return
    spec = load_spec(args.config)
    bars = prepare_bars(spec, args.data)
    if args.command == 'prepare':
        print(json.dumps({'data': str(project_path(args.data)), 'rows': len(bars),
                          'symbols': list(spec.symbols)}, ensure_ascii=False))
        return
    data = build_task_dataset(bars, spec)
    if args.command in {'run', 'replay'}:
        if args.device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable; choose cpu explicitly')
        torch.set_num_threads(4)
    if args.command == 'run':
        result = run_experiment(data, spec, args.output,
             data_hash=file_hash(project_path(args.data) / 'bars.parquet'), modes=tuple(args.search),
             exchange=args.exchange, confirm=args.confirm, device=args.device)
        from .catalog import QualityCatalog
        from .report import render_report
        entry = QualityCatalog().register(args.output)
        report = render_report(args.output)
        print(json.dumps({'status': result['status'], 'selection': result['selection'],
                          'quality_record': entry['entry_id'],
                          'report': str(report)}, ensure_ascii=False))
    elif args.command == 'replay':
        from .contracts import ModuleSpec, RepresentationSpec
        from .replay import run_replay
        registry = EvidenceRegistry()
        registry.import_prior_runs()
        registry.consume(source='binance_um_1h', symbols=spec.symbols,
            start=(utc(spec.start) - pd.Timedelta(hours=spec.lookback + 48)).isoformat(),
            end=spec.dev_end, run=str(project_path(args.output)), claim=spec.digest,
            role='development')
        architecture = RepresentationSpec((ModuleSpec(scale=min(12, spec.lookback)),))
        model = causal_initial(data, spec, architecture, args.device)
        result = run_replay(data, model, spec, args.output, architecture=architecture,
                           update_every_hours=args.update_every_hours,
                           update_mode=args.update, device=args.device)
        print(json.dumps({key: result[key] for key in
            ('status', 'scope', 'update_mode', 'predictions', 'scored_tasks', 'censored_tasks',
             'pending_tasks', 'tasks')}, indent=2, ensure_ascii=False))
    elif args.command == 'score':
        directory = project_path(args.run)
        claim = json.loads((directory / 'frozen_claim.json').read_text('utf-8'))
        verify_bundle(directory / 'bundle', claim)
        result = json.loads((directory / 'results.json').read_text('utf-8'))
        registry = EvidenceRegistry()
        registry.import_prior_runs()
        registry.consume(source='binance_um_1h', symbols=spec.symbols,
            start=(utc(spec.start) - pd.Timedelta(hours=spec.lookback + 48)).isoformat(),
            end=spec.end, run=str(project_path(args.output)), claim=claim['digest'],
            role='development')
        name = args.model
        if name == 'accepted':
            name = ('candidate' if (result.get('confirmation') or {}).get('accepted_system')
                    else 'baseline')
        from .scoring import ProbabilityCalibrator, adaptive_horizons
        model = ModelGroup.load(directory / 'bundle' / name)
        output = ProbabilityCalibrator.load(directory / 'bundle' / f'{name}_calibration.json').transform(
            model.predict(data))
        target = project_path(args.output)
        target.mkdir(parents=True, exist_ok=False)
        rows = data.meta[['sample_id', 'symbol', 'available_at']].copy()
        for key, values in output.items():
            if values.ndim == 1:
                rows[key] = values
            else:
                for j in range(values.shape[1]):
                    rows[f'{key}_prob_{j}'] = values[:, j]
                task = next(t for t in data.tasks if t.name == key)
                adaptive = adaptive_horizons(values, task)
                for field in ('horizon_hours', 'selected_up_probability', 'selected_down_probability',
                              'selected_no_event_probability', 'coverage_achieved'):
                    rows[f'{key}_{field}'] = adaptive[field]
        rows.to_parquet(target / 'scores.parquet', index=False)
        z = model.encode(data)
        np.save(target / 'representations.npy', z, allow_pickle=False)
        immutable_json(target / 'manifest.json', {'model': name, 'model_hash': model.digest,
            'frozen_claim': claim['digest'], 'rows': len(rows), 'task_schema': data.schema,
            'scores_sha256': file_hash(target / 'scores.parquet'),
            'representation_sha256': file_hash(target / 'representations.npy'),
            'evidence_exposure': {'source': 'binance_um_1h', 'intervals': [
                {'symbol': symbol,
                 'start': (utc(spec.start) - pd.Timedelta(hours=spec.lookback + 48)).isoformat(),
                 'end': utc(spec.end).isoformat()} for symbol in spec.symbols]},
            'tasks': [{'name': t.name, 'kind': t.kind, 'horizon_hours': t.horizon_hours,
                       'event_bins': t.event_bins} for t in spec.tasks],
            'scope': 'research scores, no trading actions; historical scoring may be in-sample'})
        print(json.dumps({'scores': str(target), 'model': name, 'rows': len(rows)}))


if __name__ == '__main__':
    main()
