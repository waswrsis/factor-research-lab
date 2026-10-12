【历史 V5/V6 交接包，2026-10-03；不是当前 V7 安装入口】
下列工具、重新生成依赖锁、GPT 试运行及“全部测试”要求仅属于旧方案。
不得将其作为当前 V7 的默认启动或全量 pytest 步骤；先读项目 README，等待最终 P6 交付绑定。
本历史包不证明当前 V7 完整部署，原工具与报告仅为留档。

FactorResearch dot 部署交接包

先阅读 outputs/DOT_DEPLOYMENT_MANUAL.md。

这个交接包只包含手册、六个辅助工具和本地审查记录。
它不包含 FactorResearch 主程序、市场数据、模型、研究记录或 Python 环境。
完整项目迁移包需要在 D:/FactorResearch 停止所有写入后，按手册第4节制作。

目标：Debian 13 / Python 3.12.14 / CPU / 9.7 GiB RAM / 项目 overlay。
研究 proposer：gpt-6-astra，high，复用目标已有 Codex CLI。
部署助手可以使用本手册末尾的交接指令。

工具：
transfer.py              跨平台停机封包与校验解包
apply_linux_profile.py   仅在新 Linux 副本上应用适配，原件留档
environment.sh          Bash 研究环境；先指定真实 overlay 根目录
configure.py            创建新的有限预算配置
probe_codex.py          完整 CLI 参数解析检查，不调用模型
run_bounded.py          Linux 有界进程启动与清理

本地传输回归、语法、配置契约和静态检查不代表目标 Linux 验收。
还需在目标验证：依赖安装、全部测试、进程退出、GPT 权限和真实研究、备份恢复。
旧 Windows run 与历史证据保留；新 Linux 源码和依赖需创建新 run。
禁止改写旧摘要绕过 resume，也不把新目录或部署成功当成独立预测证据。
