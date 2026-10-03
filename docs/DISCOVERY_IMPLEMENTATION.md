# 第三版方案实施与验收台账

目标依据：outputs/factor_research_plan_v3_and_reassessment.md。
实现位置：factorlab/discovery，独立于旧版交易回测流程。旧模块与历史记录保留。
本文件记录工作范围与验收证据，不把待实现项目记作完成。

## 必须交付的能力

1. 有时点和标签成熟规则的数据集，波动为首个任务，首次事件为独立任务；描述任务不混作未来信息。
2. 固定特征正则基准、可学习表示和组合、多任务监督、因子组身份、结构增删合并及维数修改。
3. 数值健康、停止原因、分项监督与梯度更新诊断、最佳检查点、独立校准、可验证模型保存加载。
4. 冻结表示的实际增量和容量对照、完整体系对照、共同覆盖限制与依赖保留的不确定性估计。
5. 当前研究评分和因子质量分开；最小增量、未达目标和证据不足均可表达。
6. 成熟标签的顺序回放，固定和更新算法可比较，自适应期限固定共同评价对象。
7. 统一实验记录、不可变历史、完整版本包、开发和确认隔离、持久的已用资料登记。
8. 有预算结构搜索与 agent 提案，保留简单搜索对照；有限 JSON 动作，不执行 agent 任意代码。
9. 真实数据端到端运行、受控正负检查、代码与语义独立审查、可操作入口和说明。

## 协作接口

共用 contracts.py 由主代理负责，其他代理不修改。

- TaskSpec(name, kind, horizon_hours, event_bins, barrier_multiplier)。kind 为 volatility、first_event、state、return。
- TaskDataset(x float32[N,L,F], targets dict[str,array[N]], observed dict[str,bool[N]], maturity dict[str,datetime64ns[N]], meta DataFrame, features tuple, tasks tuple)。meta 必须有 sample_id、symbol、available_at，按时间再标的排序；事件类别为 0=完整无事件，1+2*b=第 b 段向上，2+2*b=向下；缺失/歧义用 observed=False，不能按无事件处理。
- TaskDataset.subset(mask) 与 task_mask(name, cutoff=None)；标签及诊断不得进入模型特征。
- ModuleSpec(name, kind, dim, width, scale, features)，kind summary/temporal/mlp，features 空表示全部。
- RepresentationSpec(modules tuple, combination concat/gated)；表示可独立导出，组合是下游使用方法而非因子质量。
- TrainingSpec(epochs, batch_size, learning_rate, l2, patience, min_delta, clip_norm, seed, max_parameters, max_seconds)，所有预算运行前冻结。
- ExperimentSpec(tasks, lookback, symbols, start, train_end, dev_end, calibration_end, end, training, candidates, bootstrap_block_hours, bootstrap_samples, min_blocks, minimum_increment, confidence, seed)，确认区间 calibration_end 到 end。配置读取严格拒绝未知字段。

各模块需要自行检查输入和返回值，但不各自修改评价协议。

## 文件归属

- 主代理：contracts、evidence、search、experiment、CLI、报告与集成。
- 数据任务代理：tasks.py、tests/test_discovery_tasks.py。
- 表示训练代理：learning.py、tests/test_discovery_learning.py。
- 评价代理：scoring.py、tests/test_discovery_scoring.py。

## 完成证据（2026-10-03）

当前实现与本计划九项能力均已接通。项目全套298项测试通过（23.88秒），
factorlab与tests的Ruff检查通过。Python测试范围固定为项目tests，第三方参考源码不参与收集。

| 能力 | 实现与实际验收 |
| --- | --- |
| 数据和任务 | tasks/contracts；26,568根真实小时K线；未来完整窗口、缺口、歧义、未成熟分别覆盖测试 |
| 表示与组合 | learning；线性、单模块、多模块、两种子；实际搜索执行替换、门控、删除、压缩维数；所有注册动作单测 |
| 优化诊断 | 支持度、梯度、更新、停止、最佳checkpoint；数值失败拒绝；加载哈希校验及零样本评分测试 |
| 两条评价 | 全系统、加入冻结表示、同维数对照；正式确认6,585个成熟标的时点、46个48小时块 |
| 两层评分 | 任务评分导出26,280行；catalog存系统与表示独立质量结论，默认保留baseline |
| 顺序回放 | fixed/refit各4,464次预测，13,314次任务结算，78个未成熟任务；8次重训成功 |
| 版本与隔离 | 模型/校准/探针/源码冻结；日志链、确认先consume；历史回放/评分/失败暴露可从已知产物恢复 |
| agent与预算 | GPT-6.1 Sol/High真实提供4轮JSON提案；随机搜索同为4次；确认包含方法对照 |
| 综合验收 | 人工AR正负控制、真实数据运行、两轮独立语义审查、CLI与页面核验、保存后重载和评分 |

真实运行：runs/discovery-v3-development-001；runs/discovery-v3-agent-confirm-001。
回放：runs/discovery-v3-replay-fixed-001 与 runs/discovery-v3-replay-refit-001。
受控记录：outputs/discovery_controls_v3.json。
回放比较：outputs/discovery_replay_comparison_v3_verified.json。
评分：outputs/discovery-v3-scores-001；使用说明：docs/DISCOVERY_GUIDE.txt。

正式确认总耗时291.578秒；random搜索28.031秒，external搜索245.718秒（含提案等待）。
实际agent服务共307秒，含首请求前等待，计时范围不同。没有独立API费用记录或FLOPs测量。
正式确认选中external候选，但系统增量为-0.010282，调整区间[-0.036582,0.009311]，证据不足；
表示增量、容量对照与agent对随机搜索也未得到支持。不能据此宣称市场有效性已经成立。

已解决的审查发现：预测前几秒被纳入事件标签、预热与旧数据暴露遗漏、回放/评分未登记、
菜单学习率无效动作、可变协议集合、嵌套确认资料进入agent记忆、接近零的损失公式差异、
求解器失败误归预算、空确认数据崩溃、未观察标签行的非有限预测被忽略、uv.lock登记误排除。
最后一项仅影响已完成结果的档案登记；修复后登记与核验成功，没有重跑或重新消费确认。

数据/学习代理与评价代理独立复核了冻结时点、任务掩码、方法对照、失败规则与资料使用路径，
未报告剩余阻断问题。数值健康不等于全局收敛；这次小预算研究未证明复杂模型优于基准。
实测覆盖当前加密永续小时数据；传统期货适配、持续无人值守API agent及未来真实时间绩效
不在本次已验证能力之内。外部agent接口需要运行中的提案消费者。
