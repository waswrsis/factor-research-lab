# FactorResearch

小时级因子研究原型：由 GPT 提出因果公式和受限结构修改，训练器拟合参数，再按跨时段评分和后续影子比较决定是否采用候选。产物是因子表示、任务预测和研究证据，不包含交易执行。

当前研究版本为 V6。已有公式编写、参数拟合、模块增删、开发记忆、历史结构复用和有限预算演化；尚未证明新方法稳健超过原 benchmark，也不具备无限运行及任意算法源码自行重写能力。

## 获取部署版本

请使用 [deployment-2026-10-03-v1 Release](https://github.com/waswrsis/FactorResearch/releases/tag/deployment-2026-10-03-v1)。其中的 `factorresearch-portable-20261003-v1.zip` 是完整部署迁移资料，大小 237,295,877 字节，包含主程序、数据、模型、记忆库、实验结果和审计记录。

Git 仓库维护源码、配置、测试、说明与部署工具。完整历史数据和模型通过 Release 附件提供；GitHub 自动生成的 Source code 压缩包只含 Git 跟踪的文件。

**部署入口：[GitHub Release 部署说明](docs/GITHUB_RELEASE_DEPLOYMENT.md)**。它说明如何在云端用 Git 获取源码、下载 Release、校验并解包到项目磁盘，再进行 Linux 适配。

## 目标环境

推荐首个云端验收环境：Debian 13 x86_64、Python 3.12.14、CPU、约 10 GiB 内存，研究文件放在项目 overlay。安装时重建 Linux CPU 依赖；项目中的原 Windows 虚拟环境和凭据没有发布。

目标已有 Codex CLI 时先核验版本、认证和完整参数兼容性。Linux 部署模板选择 `gpt-6-astra`、推理强度 `high`。历史实验使用的模型设置保持原样。

## 验证状态

- 完整迁移包包含 14,343 个文件；已在 Windows 完成全量解包和每个文件的 SHA256/CRC 校验。
- 恢复副本中的 22 个正式 SQLite 数据库通过完整性检查，包括一个带 WAL 的状态库。
- 原 Windows V6 工程记录为 548 项测试通过；这是历史版本验收，不是目标 Linux 验收。
- 96 次市场拟合没有取得合格的基准改善；真实 GPT 生成的两个新公式均被拒绝。2025 数据已经用于研究开发，不能重新称为独立测试资料。
- 目标 Linux 的依赖安装、完整测试、进程生命周期和 Astra 实际请求仍待执行。发布 Release 不代表已经部署到 dot 云端。

详细部署步骤见 [原迁移手册](outputs/DOT_DEPLOYMENT_MANUAL.md)，研究限制和结果见 [V6 实测报告](outputs/ROBUST_V6_IMPROVEMENT_REPORT.txt)。校验记录保存在 [docs/release](docs/release)。

## 目录

| 路径 | 用途 |
|---|---|
| `factorlab/` | 研究程序及因果算子解释器 |
| `config/` | 冻结任务和演化预算配置 |
| `tests/` | 工程测试 |
| `scripts/` | 原 Windows 辅助入口 |
| `outputs/dot-deployment-kit/` | Linux 适配、传输校验和有界启动工具 |
| Release 中的 `data/` | 行情缓存、准备数据和曝光账本 |
| Release 中的 `runs/` | 真实模型、研究记忆、候选、预测和源代码快照 |
| Release 中的 `work/` | 历史审计与验收资料 |

迁移到 Linux 会产生新的代码与环境身份。保留旧 run 作为证据，创建新的 Linux run；不得改写旧摘要以绕过恢复校验。
