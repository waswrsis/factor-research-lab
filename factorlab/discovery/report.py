"""A self-contained, read-only research report. No execution or trading controls."""
from __future__ import annotations

import html
import json

from factorlab.protocol import project_path

from .evidence import file_hash, immutable_json


def render_report(run_directory):
    directory = project_path(run_directory)
    result = json.loads((directory / 'results.json').read_text('utf-8'))
    esc = lambda value: html.escape(str(value))
    labels = {'baseline': '正则线性基准', 'single': '单个可学习模块',
              'fixed_experts': '固定多模块', 'search_random': '随机结构搜索',
              'search_diagnostic': '本地诊断规则搜索', 'search_external': 'LLM 结构搜索',
              'system_vs_baseline': '完整候选 对 简单基准',
              'representation_addition': '加入冻结表示的增量',
              'matched_dimension_control': '冻结表示 对 同维数对照',
              'single_vs_baseline': '单模块 对 简单基准',
              'fixed_vs_single': '多模块 对 单模块',
              'random_vs_fixed': '随机搜索 对 固定多模块',
              'external_vs_fixed': 'LLM搜索 对 固定多模块',
              'external_vs_random': 'LLM搜索 对 随机搜索',
              'supported': '获得限定支持', 'target_not_met': '未达预定增量',
              'insufficient_evidence': '证据不足', 'unavailable': '无有效候选'}
    arm_rows = []
    for name, arm in result['arms'].items():
        loss = arm.get('development_loss')
        arm_rows.append('<tr><td>' + esc(labels.get(name, name)) + '</td><td>' +
                        (f'{loss:.6f}' if loss is not None else '没有有效候选') + '</td><td>' +
                        esc(arm.get('parameter_count', '见候选记录')) + '</td></tr>')
    confirmation = result.get('confirmation')
    headline = ('候选通过本次确认，结论限于当前任务与样本' if confirmation and confirmation['accepted_system']
                else '候选尚未获得确认支持，默认保留简单基准' if confirmation
                else '开发实验已完成，尚未进行独立确认')
    rows = []
    for name, comparison in (confirmation or {}).get('comparisons', {}).items():
        cells = [labels.get(comparison['status'], comparison['status'])]
        cells += [f'{comparison[key]:.6f}' if comparison.get(key) is not None else '—'
                  for key in ('estimate', 'ci_lower', 'ci_upper')]
        cells += [comparison.get('n_blocks', '—'), labels.get(name, name)]
        rows.append('<tr>' + ''.join(f'<td>{esc(cell)}</td>' for cell in cells) + '</tr>')
    limitations = ''.join(f'<li>{esc(item)}</li>' for item in result['limitations'])
    content = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>因子研究结果</title><style>
body{{margin:0;background:#f5f7fa;color:#17283b;font:16px/1.7 system-ui,sans-serif}}
main{{max-width:1080px;margin:40px auto;padding:0 24px}}section{{background:white;padding:24px;margin:20px 0;border-radius:12px}}
h1{{font-size:30px}}h2{{font-size:20px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border-bottom:1px solid #dde4eb;padding:10px;text-align:left;overflow-wrap:anywhere}}
.state{{color:#0d6072;font-weight:600}}details{{margin:16px 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}button{{padding:8px 14px;cursor:pointer}}
@media(max-width:650px){{main{{padding:0 12px}}section{{padding:14px}}table{{font-size:12px}}td,th{{padding:5px}}}}
</style><main><h1>因子研究结果</h1><p class="state">{esc(headline)}</p>
<p>主任务：{esc(result['primary_task'])}。本页呈现研究评分与证据，不产生交易动作。</p>
<section><h2>开发阶段比较</h2><p>开发成绩用于选择候选，不能替代独立确认。</p>
<table><thead><tr><th>方法</th><th>主要损失 越低越好</th><th>参数数</th></tr></thead><tbody>{''.join(arm_rows)}</tbody></table>
<p>选中的候选：{esc(labels.get(result['selection']['arm'], result['selection']['arm']))}。选择依据为预先规定的开发主任务损失。</p></section>
<section><h2>独立确认</h2><p>{'已经消耗确认资料；结果不能反馈后再沿用为未见证据。' if confirmation else '本次只完成开发比较，没有消耗确认区间。'}</p>
<table><thead><tr><th>结论</th><th>增量</th><th>区间下界</th><th>区间上界</th><th>时间块数</th><th>比较对象</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p>supported 表示在限定任务和范围内得到支持；target_not_met 表示未达到预定增量；insufficient_evidence 表示证据不足。</p></section>
<section><h2>解释范围</h2><ul>{limitations}</ul>
<p>表示身份绑定完整生成器版本。模型组合权重与潜在维度，不直接等同于因子质量或因果贡献。</p></section>
<details><summary>查看完整实验记录</summary><pre>{esc(json.dumps(result, ensure_ascii=False, indent=2))}</pre></details>
<button onclick="window.print()">打印或保存</button></main></html>'''
    target = directory / 'report.html'
    with target.open('x', encoding='utf-8') as stream:
        stream.write(content)
    immutable_json(directory / 'report_manifest.json', {'sha256': file_hash(target)})
    return target
