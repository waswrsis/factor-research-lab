"""Export verified research scores and a compact local evolution report."""
from __future__ import annotations

import html
import json

import numpy as np
import pandas as pd

from factorlab.protocol import project_path

from .contracts import ExperimentSpec
from .evidence import file_hash, immutable_json
from .evolution import EvolutionStore, inspect_evolution
from .scoring import adaptive_horizons


def export_evolution(directory, output):
    summary = inspect_evolution(directory)
    destination = project_path(output)
    destination.mkdir(parents=True, exist_ok=False)
    store = EvolutionStore(directory)
    try:
        state = store.state()
        spec = ExperimentSpec.from_dict(state['identity']['protocol'])
        rows = [dict(row) for row in store.db.execute(
            "SELECT * FROM predictions WHERE role='incumbent' ORDER BY origin,symbol,task")]
        scores = pd.DataFrame(rows)
        scores.to_parquet(destination / 'scores.parquet', index=False)
        distributions = []
        for task in spec.tasks:
            if task.kind != 'first_event':
                continue
            selected = scores[scores.task == task.name]
            if selected.empty:
                continue
            probabilities = np.asarray([json.loads(value) for value in selected.value])
            horizons = adaptive_horizons(probabilities, task)
            for index, row in enumerate(selected.itertuples()):
                distributions.append({'sample': row.sample, 'symbol': row.symbol, 'origin': row.origin,
                    'model': row.model, 'task': task.name,
                    'selected_horizon_hours': int(horizons['horizon_hours'][index]),
                    'coverage_achieved': bool(horizons['coverage_achieved'][index]),
                    'probabilities': probabilities[index].tolist(), 'bins': list(task.event_bins),
                    'concentration_is_confidence': False})
        immutable_json(destination / 'event_distributions.json', distributions)
        immutable_json(destination / 'summary.json', summary)
        e = lambda text: html.escape(str(text))
        task_rows = ''.join(f'<tr><td>{e(name)}</td><td>{value["support"]}</td>'
                           f'<td>{e(value["loss"])}</td><td>{value["pending"]}</td></tr>'
                           for name, value in summary['tasks'].items())
        cycle_rows = ''
        for cycle in summary['cycles']:
            gate = cycle.get('gate', {})
            actions = [record.get('proposal', {}).get('operation')
                       for record in cycle.get('research', {}).get('records', [])
                       if record.get('proposal')]
            cycle_rows += (f'<tr><td>{cycle["id"]}</td><td>{e(cycle["cutoff"])}</td>'
                f'<td>{e(cycle["status"])}</td><td>{e(", ".join(actions))}</td>'
                f'<td>{e(gate.get("estimate", "—"))}</td>'
                f'<td>{e(gate.get("ci_lower", "—"))} / {e(gate.get("ci_upper", "—"))}</td></tr>')
        report = f'''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>持续因子研究报告</title><style>
body{{max-width:1150px;margin:40px auto;font:16px/1.65 system-ui;background:#f6f8fb;color:#172334;padding:0 24px}}
table{{width:100%;border-collapse:collapse;background:white;margin:20px 0}}td,th{{padding:10px;border-bottom:1px solid #dce3ec;text-align:left}}
.card{{padding:22px;background:white;border-radius:12px;margin:20px 0}}small{{color:#567}}
</style><h1>持续因子研究 · {e(summary['mode'])}</h1>
<div class="card">已处理 <b>{summary['ticks']}</b> 个预测时刻；研究 <b>{len(summary['cycles'])}</b> 轮；
晋级 <b>{summary['promotions']}</b> 次。<br>最后信息时刻：{e(summary['last_timestamp'])}<br>
这是开发资料上的顺序研究；晋级表示满足本轮运作规则，不等于独立确认或收益保证。</div>
<h2>当前输出质量</h2><table><tr><th>任务</th><th>已成熟样本</th><th>平均损失（越低越好）</th><th>待成熟</th></tr>{task_rows}</table>
<h2>结构与版本迭代</h2><table><tr><th>轮次</th><th>研究截止</th><th>结果</th><th>尝试动作</th><th>影子增量</th><th>区间下界 / 上界</th></tr>{cycle_rows}</table>
<div class="card">流程：成熟反馈 → 有预算搜索 → 保存候选 → 并行记录未来表现 → 单次比较 → 替换或保留。
失败与中断会保留现用版本；历史输出不会按新模型回写。事件输出是首次触边时间的分箱概率，概率集中度不当作可靠性证明。</div>
<p><a href="scores.parquet">完整研究评分</a> · <a href="event_distributions.json">事件分布与所选期限</a> · <a href="summary.json">完整研究记录</a></p>
<small>已核验预测/结算账本、模型文件和冻结身份。检验不防御拥有整个工作区写权限的恶意全面重写。</small></html>'''
        (destination / 'report.html').write_text(report, encoding='utf-8')
        manifest = {'source_run': str(project_path(directory)), 'event_head': summary['event_head'],
                    'scope': summary['scope'], 'rows': len(scores), 'files': {
                        path.name: file_hash(path) for path in destination.iterdir()
                        if path.is_file() and path.suffix != '.lock'}}
        immutable_json(destination / 'manifest.json', manifest)
        return {'report': str(destination / 'report.html'), 'rows': len(scores), 'manifest': manifest}
    finally:
        store.close()
