# FactorResearch 迁移到 dot 云端电脑的部署手册

本手册把 `D:/FactorResearch` 的程序、数据、实验记录和审计资料迁移到 dot 云端电脑，并在新的 Linux 部署身份下启动有预算的因子研究。目标是自动提出算子与结构候选、拟合参数、回放评分、进行影子比较和记录淘汰理由；不包括交易执行。

**推荐路线：保留 Windows 原件 → 完整封包与校验 → 在项目 overlay 中解包 → 应用 Linux CPU 适配 → 创建新的依赖锁 → 完成环境与 GPT 验收 → 新建 Linux 研究运行 → 有界后台运行与异地备份。** 研究 Agent 按用户最新要求使用 **`gpt-6-astra`，推理强度 `high`**。

编制日期：2026 年 10 月 3 日。依据为当前本地源码、V5/V6 实验档案、用户提供的目标环境和官方文档。本次只制作手册与附件，**没有连接目标机器、安装目标依赖、调用目标 GPT 或完成云端部署**。附件在本地做了有限校验；Linux 实际运行及完整依赖解析必须在目标机器验收。

## 1 目标环境与部署判断

| 项目 | 用户提供的实测值 | 本次部署选择 |
|---|---|---|
| 系统 | Debian GNU/Linux 13，x86_64 | Linux 新环境，不复制 Windows 虚拟环境 |
| CPU | 9 个可见共享逻辑处理器 | 单个研究进程，Torch 与常见数学库先用 2 线程 |
| 内存 | 9.7 GiB，总体可用约 8.6 GiB | 单个候选依次训练，batch 128 起步，监测整个进程树 |
| Swap | 无 | OOM 可能直接中断，没有保证可执行清理逻辑 |
| GPU | 未确认 | 显式 `--device cpu`，安装 CPU Torch |
| 根目录与 `/tmp` | 约 4.9 GiB tmpfs | 不放项目环境、研究缓存、数据和日志 |
| 项目工作目录 | 32 GB overlay，约 30 GB 可用 | 项目及其虚拟环境放这里；持久性另行确认 |
| Python | 3.12.14 | 符合项目 `>=3.12,<3.13`；部署版本文件同步改为此版本 |
| Node.js | 24.19.0 | 已有 Codex 的运行条件之一，研究核心不需要 Node 构建 |
| Codex | 已安装 CLI | 复用，先检查版本、参数、登录和模型权限 |

这套配置适合小规模 CPU 研究与逐轮搜索，不能根据配置表承诺训练速度或内存一定够用。首轮先用三标的、小时数据、小候选集；线程数、批量和时间预算是初始工程选择，需用实际峰值与每轮耗时验证。

tmpfs 的容量上限不是预先占用的内存，也不是独立硬盘；文件会使用内存，并在有 swap 时可能换出。因此不能把 4.9 GiB 简单再加到 9.7 GiB 内存上。overlay 的名称和 `df` 容量也不能证明数据永久保存。参见 [Linux tmpfs 文档](https://docs.kernel.org/filesystems/tmpfs.html)。

## 2 迁移前需要知道的限制

当前代码不能直接复制后运行，存在下列适配点。附件 `apply_linux_profile.py` 仅用于**新的 Linux 副本**，会保存修改前文件和每个文件的摘要。

| 当前行为 | Linux 适配 |
|---|---|
| `factorlab/protocol.py` 固定 `D:/FactorResearch` | 根目录读取绝对路径 `FACTOR_RESEARCH_ROOT`，保留目录边界校验 |
| 下载器默认 `D:/FactorResearch/data` | 改为项目内相对 `data` |
| Windows 路径在 POSIX 会变成奇怪的相对路径 | 明确拒绝含 Windows 盘符的旧路径，不做偷偷映射 |
| 锁文件限定 win32/AMD64，Torch 为 CUDA 12.8 Windows 轮子 | 改 Linux/x86_64 与 `torch==2.11.0+cpu`，目标机重新生成 `uv.lock` |
| 三处入口固定 Torch 4 线程 | 改为读取 `FACTOR_RESEARCH_TORCH_THREADS`，默认 2，校验 1 至 9 |
| 演化入口固定 `gpt-6.1-sol` | 实际 worker 参数改为 `gpt-6-astra` 和 `high` |
| POSIX 只杀直接子进程 | 增加后代清理，并由外层独立进程组作最后清理 |
| SIGTERM 默认直接结束 | 主 CLI 与 worker 增加合作退出处理 |
| `AGENTS.md` 仍要求 D 盘和 PowerShell | 在目标副本加入优先的 Linux 操作说明，保留旧说明作历史 |

原演化运行绑定源码、依赖锁、协议、数据语义和绝对记忆库路径。跨平台修改这些内容后，**旧 Windows run 不能直接 `--resume`**。不要改旧 manifest、替换 SQLite 中路径或重算旧摘要来绕过检查。旧运行完整保留，新 Linux run 使用新目录和新记忆库。若将来需要无缝接续旧运行，须另做明确的迁移兼容契约。

旧数据准备 manifest 也含 D 盘绝对路径。保留这些原件；在新目录重新准备 Linux 数据 manifest。质量档案索引中也有旧绝对路径，因此不保证所有旧 `catalog verify` 或路径相关检查在 Linux 原样通过。原文件摘要、原报告与 D 盘原件仍是历史核对依据。

截至本手册编制时，本地 V6 记录有 548 项测试通过；冻结市场对照的 96 次拟合没有获得合格的基准改善。曾完成的真实 GPT 验收使用 Sol，而非本次部署的 Astra。**这些是旧 Windows 版本证据，不是新 Linux/Astra 版本的验收结果。** 2025 年资料已是暴露的开发资料，搬家或换模型不会使它重新成为独立测试集。

## 3 附件和执行约定

附件目录为 `outputs/dot-deployment-kit/`，随项目整体迁移：

| 文件 | 用途 |
|---|---|
| `transfer.py` | 停机后封包、逐文件摘要、校验后解包到新目录 |
| `apply_linux_profile.py` | 预览或一次性应用上述 Linux 适配；不安装依赖 |
| `environment.sh` | 将研究环境和缓存指向项目目录，设置 CPU 线程 |
| `configure.py` | 创建新的小规模验收与三轮 CPU 配置，不覆盖旧配置 |
| `probe_codex.py` | 用项目生成的完整 CLI 参数做 `--help` 兼容检查，不调用模型 |
| `run_bounded.py` | 限时运行、单进程锁、停止文件、日志与退出记录 |

Windows 命令用 PowerShell；目标命令用 Bash。下文 `/实际项目挂载点` 是待替换的占位符，不是已发现的 dot 路径。选择已确认的 32 GB 项目 overlay 所在目录，不使用 `/tmp`、tmpfs 根目录或没有核实的默认 home。

不要全局更改 `HOME` 或 `CODEX_HOME`，也不迁移 Windows 的 Codex 安装和凭据。附件只管理研究项目；Codex 自身的认证及全局运行文件继续由目标已有 CLI 管理，其所在文件系统也应留意空间与重建后的登录状态。

## 4 停机和制作迁移包

### 4.1 停止全部写入

在本地停止研究主进程、独立 worker 和 feed 更新进程。前台先按一次 Ctrl+C，等待退出和清理。核对任务管理器或原终端确认这些项目进程都已结束，然后再封包。

必须一起保留 `data/`、`runs/`、`models/`、`outputs/` 和 `work/` 的证据。特别是曝光账本 `data/discovery_evidence.sqlite`，以及各个 run 中的记忆库、状态库、候选、失败记录、source 快照、模型和预测记录。实际记忆库不一定在默认的 `data/research_memory.sqlite`，有的在 run 目录内。

SQLite 采用 WAL 时，已提交数据可能尚在 `-wal` 中。不要运行中只复制 `.sqlite` 主文件，也不要手动删除 `-wal` 或 `-shm`。停机后整目录封包可以一起保留这些文件；单库 online backup 不能代替多个库、模型和 feed 的共同停机一致性。

### 4.2 在 D 盘封包

```powershell
Set-Location -LiteralPath D:/FactorResearch
. ./scripts/Enter-Research.ps1 -Quiet
New-Item -ItemType Directory -Force -Path work/migration_export | Out-Null
$archive = 'D:/FactorResearch/work/migration_export/factorresearch-portable-20261003.zip'
./.venv/Scripts/python.exe outputs/dot-deployment-kit/transfer.py pack D:/FactorResearch $archive
Get-FileHash -LiteralPath $archive -Algorithm SHA256
```

如果同名包已存在，使用新日期或后缀，不覆盖旧包。工具排除可重建的 `.venv`、`.bootstrap`、`.tools`、`.cache`、`.git`、常见构建缓存、自己的导出目录和已知凭据文件；保留源码、未被 Git 跟踪的程序文件及研究档案。它不做任意文本的秘密识别，封包前仍应检查自写脚本或日志中是否曾明文保存凭据。不要把项目 zip 公开发布。

工具拒绝链接类文件与危险路径；遇到 junction/symlink 应先核对目标，明确决定保留方式，不能让封包跟随到项目外。排除项目中的第三方 `.git` 不会排除其源码。Git clone 不能替代此次封包，因为当前项目有大量未跟踪文件与实验资料。

先前统计的可迁移项目内容约为 1 GiB 以下的逻辑文件规模；这不是实测 zip 大小，也不含依赖环境。Windows CUDA Torch 本身约 4 GiB，不应带到 Linux。

### 4.3 上传和接收

将项目 zip、它的 `.sha256`、附件工具 `transfer.py` 上传到目标 **overlay 中的接收目录**。可使用 dot 的文件上传或已经可用的文件传输方式；本手册不假设有 SSH、sudo、Docker 或固定网络端口。

```bash
set -euo pipefail
export TRANSFER_DIR='/实际项目挂载点/transfer'
export FACTOR_RESEARCH_ROOT='/实际项目挂载点/FactorResearch'
cd "$TRANSFER_DIR"
sha256sum factorresearch-portable-20261003.zip
# 将输出与 D 盘 Get-FileHash 的结果独立核对，一致后继续。
PYTHON_BIN="$(command -v python3.12 || command -v python3)"
"$PYTHON_BIN" --version
"$PYTHON_BIN" transfer.py unpack factorresearch-portable-20261003.zip "$FACTOR_RESEARCH_ROOT"
cd "$FACTOR_RESEARCH_ROOT"
mkdir -p work/deployment
cp --no-clobber _TRANSFER_MANIFEST.json work/deployment/received-payload-manifest-20261003.json
```

解包目标必须不存在；不要覆盖旧部署或运行中的目录。内部逐文件校验可检查内容完整性，外部与 D 盘核对的 zip 摘要用于确认拿到的是预期的整个包。根目录 `_TRANSFER_MANIFEST.json` 是本次接收清单，工具在下次封包时重新生成根清单，不把旧根清单当作项目内容；上面的副本会随项目保留，重复迁移时使用新的留档文件名。原 zip 与摘要也应留存。失败后保留错误与不完整目录，换一个全新的目标路径重新恢复，不自动删除原资料。

## 5 核验目标挂载和资源

在目标执行，保存输出到项目内。若实际可用资源比用户给出的数据更小，先调整计划，不能按旧配置硬跑。

```bash
mkdir -p work/deployment
{
  date -u
  uname -a
  cat /etc/os-release
  id
  nproc
  free -h
  df -hT "$FACTOR_RESEARCH_ROOT" /tmp /
  findmnt -T "$FACTOR_RESEARCH_ROOT"
  findmnt -T /tmp
  cat /proc/self/cgroup
  "$PYTHON_BIN" --version
  node --version
  command -v codex
  codex --version
} > work/deployment/target-preflight.txt 2>&1
cat work/deployment/target-preflight.txt
```

同时检查当前进程所属 cgroup 的 `memory.max`、`memory.current`、`memory.events`、`cpu.max`。容器挂载可能把所属 cgroup 映射为 `/sys/fs/cgroup` 根，也可能是子目录；根据 `/proc/self/cgroup` 和挂载实际定位，不能读取宿主无关层级然后宣称那是本容器配额。`nproc` 只说明可见处理器，不等于有相同数量的独占核。

执行前确认项目路径确实在预期 overlay，并向平台确认停止、重启和重建后的保存规则。不明确时按可丢失环境管理，定期导出到实例外。不要在未知后端 overlay 上自行加 swap，或把同一 overlay 内第二份 zip 当作异地备份。

建议初始运行保留至少 2 GiB 实际可用内存；磁盘先保留约 6 GiB 空间给安装临时文件和新产物。这是保守启动策略，不是经过该服务器实测的硬需求。安装过程中可同时存在下载缓存、展开文件和新环境，所需峰值大于最终 `.venv`。

## 6 应用 Linux 副本适配

先做预览，再检查记录并应用。预览不改产品文件，仅在新副本的 `work/linux_migration/staged/` 写出提案。

```bash
"$PYTHON_BIN" outputs/dot-deployment-kit/apply_linux_profile.py --root "$FACTOR_RESEARCH_ROOT"
cat work/linux_migration/proposed.json
diff -u factorlab/protocol.py work/linux_migration/staged/factorlab/protocol.py || test "$?" -eq 1
diff -u pyproject.toml work/linux_migration/staged/pyproject.toml || test "$?" -eq 1
"$PYTHON_BIN" outputs/dot-deployment-kit/apply_linux_profile.py --root "$FACTOR_RESEARCH_ROOT" --apply
source outputs/dot-deployment-kit/environment.sh
```

适配工具通过精确文本匹配识别本版源文件；锚点不符就停止，不猜测如何修改别的版本。原源码、`AGENTS.md`、Python 版本文件与 Windows `uv.lock` 保存在 `work/linux_migration/original/`。若中途失败，查看备份、暂存文件和当前摘要；不要盲目重复应用。最清楚的恢复方式是从原包解到另一新目录重做。

该适配没有改写历史 `runs/**/source/`、历史 manifest 或账本。它会改变当前算法源码身份，所以从这一步起只能创建新的 Linux run。`environment.sh` 中的根目录变量必须在任何 `factorlab` 导入前设置。

## 7 创建 CPU Python 环境

### 7.1 选择解释器和 uv

沿用目标已安装的 Python 3.12.14。验证精确版本，避免把 Debian 系统的另一个 Python 版本误装进项目。

```bash
"$PYTHON_BIN" -c 'import sys; print(sys.version); assert sys.version_info[:3] == (3,12,14)'
if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
  "$UV_BIN" --version
else
  "$PYTHON_BIN" -m venv "$FACTOR_RESEARCH_ROOT/.bootstrap"
  "$FACTOR_RESEARCH_ROOT/.bootstrap/bin/python" -m pip install 'uv==0.12.22'
  UV_BIN="$FACTOR_RESEARCH_ROOT/.bootstrap/bin/uv"
fi
export UV_BIN
```

`uv 0.12.22` 的公开包在编制时可获取；已有 uv 可先复用并记录版本，若命令不兼容再处理。若现有 Python 没有 `venv`/`ensurepip`，先请目标管理员提供与 **这个 Python 3.12** 匹配的组件，或使用经确认的 uv 安装方法并把二进制与临时文件放在项目下。不要认为 Debian 的通用 `python3-venv` 必然对应此自带 3.12 解释器。环境未建好时停在此处，不转用 Windows `.exe`。

### 7.2 重新解析并安装完整依赖

```bash
"$UV_BIN" lock --python "$PYTHON_BIN"
"$UV_BIN" sync --locked --python "$PYTHON_BIN" --no-build
"$UV_BIN" pip check --python "$FACTOR_RESEARCH_ROOT/.venv/bin/python"
.venv/bin/python -c 'import torch; print(torch.__version__); assert torch.version.cuda is None'
```

新的 `uv.lock` 必须由 Linux 解析器生成，不能手工把 Windows marker 替换后宣称锁文件有效。`--locked` 要求已有锁与项目一致；`--no-build` 让缺少二进制轮子的包明确失败，避免在 10 GiB 机器上意外启动大规模源码编译。详见 [uv 项目指南](https://docs.astral.sh/uv/guides/projects/) 与 [锁定和同步规则](https://docs.astral.sh/uv/concepts/projects/sync/)。

完整依赖默认保留 Qlib、MLflow、Optuna、LightGBM、CVXPY、Jupyter 等当前项目组件，满足“整个程序迁移”。它们不意味着每次搜索都同时运行。若某个包没有可用轮子，记录包名和报错，再评估必要的构建工具或新的依赖版本；修改依赖须在新研究冻结前完成，不能直接删依赖而宣称完整环境已部署。

`external/qlib` 与 `external/rd-agent` 随包保存为源代码参考。可导入的 Qlib 由项目依赖安装；本手册启动的是 `factorlab.discovery` 自主研究程序，没有启动 RD-Agent 官方服务，也不要求另开 Docker、Jupyter 或 MLflow 服务器。

已核对 PyTorch 官方 CPU 索引含 `torch-2.11.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl`，但这仅证明轮子列出，不代表整套依赖已在目标安装通过。参见 [PyTorch CPU 安装说明](https://pytorch.org/get-started/locally/#no-cuda) 和 [官方 CPU 轮子索引](https://download.pytorch.org/whl/cpu/torch/)。若 LightGBM 提示缺少 `libgomp.so.1`，再核查 `libgomp1`；如有权限由系统包管理器安装，无权限则交由平台处理。不要用放宽版本约束来掩盖系统库问题。

## 8 检查已有 Codex 与 Astra 接入

### 8.1 CLI 参数和登录

```bash
command -v codex
codex --version
codex --help > work/deployment/codex-help.txt
codex exec --help > work/deployment/codex-exec-help.txt
codex features list > work/deployment/codex-features.txt
codex login status
.venv/bin/python outputs/dot-deployment-kit/probe_codex.py
```

本地 Windows CLI 为 0.160.0；目标是否兼容必须看上述结果。项目使用 `--no-daemon`、`--ignore-user-config`、`--ignore-rules`、只读 sandbox、JSON 事件和 schema、以及一组禁用能力参数。官方介绍了 `codex exec` 的 JSON 和结构化输出，但不能据此假定每个版本都接受本项目全部参数。参见 [Codex CLI](https://learn.chatgpt.com/docs/codex/cli) 与 [非交互运行](https://learn.chatgpt.com/docs/non-interactive-mode)。

probe 会保存实际参数、stdout、stderr 和返回码到新建的 `work/deployment/codex-probe-*/`。非零返回码或未知选项需要先解决版本或适配兼容问题；不要删除能力限制，也不要切换到宽权限任意命令后端来绕过。返回零只表示帮助参数解析通过，仍须审查启动警告，并用后续真实一次研究验证认证、模型可用性及事件格式。

如果尚未登录，按已有 CLI 的方式登录；无浏览器环境可在账户允许时执行 `codex login --device-auth`，由账户持有人完成授权。不要把凭据、验证码或 `auth.json` 写进本手册、项目包或日志。详见 [官方认证说明](https://learn.chatgpt.com/docs/auth)。

**当前 worker 会过滤 `OPENAI_API_KEY`、`CODEX_API_KEY` 以及变量名含 TOKEN/SECRET/PASSWORD 的环境变量。** 因而不能把“export 一个 API key”当作当前集成的即插即用路线。本手册复用同一目标用户的 CLI 已保存登录。若要改成独立 API 后端，需另做受审适配与验收。

### 8.2 确认配置真正生效

Linux 适配直接修改演化入口传给 worker 的参数：

```text
--backend codex --model gpt-6-astra --reasoning-effort high
```

仅修改用户 `config.toml` 不够，因为当前调用显式选择模型并忽略用户配置。`gpt-6-astra` 与 `high` 为官方模型接口支持的选择，但 API 文档不保证目标 Codex 账户具有相同访问权限；以目标真实请求结果为准。参见 [GPT 6 Astra 模型资料](https://developers.openai.com/api/docs/models/gpt-6-astra)。不可静默回落到其他模型并把结果记作 Astra。

研究 worker 的 GPT 职责是提出结构和受限表达式程序，实际计算由可信解释器与训练代码执行。它不会像部署助手一样自由读写项目、安装依赖或修改评分协议。部署 Codex 和研究 worker 是两个不同角色。

## 9 环境验收和冻结

先生成新配置，再做工程检查；此时不运行行情下载和正式研究。

```bash
.venv/bin/python outputs/dot-deployment-kit/configure.py --root "$FACTOR_RESEARCH_ROOT"
.venv/bin/python -c 'import torch,numpy,pandas,scipy,sklearn,qlib,mlflow,pyarrow,duckdb,optuna,backtesting,cvxpy,lightgbm,numba,matplotlib,requests,httpx,pydantic,yaml,psutil; print("imports OK")'
.venv/bin/python -m factorlab.discovery --help
.venv/bin/python -m factorlab.discovery.agent_worker --help
.venv/bin/python -m compileall -q factorlab outputs/dot-deployment-kit
.venv/bin/python -m ruff check factorlab tests
.venv/bin/python -m pytest -q --basetemp=work/deployment/pytest-linux-v1
```

测试使用人工小数据，仍会产生测试临时文件并消耗计算；它不是正式市场训练。源码含历史 Windows 假设，目标完整测试是否通过目前未知。若失败，保存失败项，先判断平台假设还是真实逻辑缺陷。不得把一批失败直接标记 skip 后声称与原 548 项同等通过。需要修复时，在首个 Linux 研究 run 前改好，重新检查并记录；如果已经开始研究，修复后另建部署与 run 身份。

还要实际测试 Linux 停机流程：用外层启动器包一个只睡眠的 Python 进程及其子进程，分别测试正常退出、超时、停止文件和 SIGTERM，检查没有残留子进程。`run_bounded.py --help` 给出参数；下例应约数秒后开始停止并返回 124，最长再等待清理宽限时间：

```bash
set +e
.venv/bin/python outputs/dot-deployment-kit/run_bounded.py --seconds 3 --label lifecycle-timeout-v1 -- \
  .venv/bin/python -c 'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c","import time; time.sleep(600)"]); time.sleep(600)'
rc=$?
set -e
test "$rc" -eq 124
```

这只验证普通同组后代。若程序主动新建会话、daemon 脱离、外层被 SIGKILL/OOM 杀死或整个容器消失，Python 清理无法保证生效。需要更强约束时使用平台的进程组/cgroup 管理或可用的 systemd；保留 Codex 的 no-daemon 约束。

全部源码适配、CLI兼容检查和依赖解析完成后，记录部署文件摘要与版本。不要把凭据或整个环境变量列表放入记录。

```bash
{
  date -u
  .venv/bin/python --version
  "$UV_BIN" --version
  codex --version
  printf 'research_model=gpt-6-astra\nreasoning_effort=high\ndevice=cpu\ntorch_threads=2\n'
} > work/deployment/versions.txt
sha256sum pyproject.toml uv.lock .python-version AGENTS.md config/discovery_dot_*.json config/evolution_dot_*.json \
  outputs/dot-deployment-kit/*.py outputs/dot-deployment-kit/environment.sh > work/deployment/deployment-files.sha256
find factorlab -name '*.py' -print0 | sort -z | xargs -0 sha256sum > work/deployment/source-files.sha256
```

附件与 `.python-version` 并非都被现有研究 source 冻结自动涵盖，所以另存这些部署摘要。更换启动器、运行参数、模型和资源预算时也保留版本记录。

## 10 准备新的 Linux 历史数据目录

先重用随包迁移且带 checksum 的 Binance 原始月档，在新目录形成 Linux manifest：

```bash
.venv/bin/python -m factorlab.discovery --config config/discovery_v6.json \
  --data data/discovery/linux-2025-v1 prepare
```

该命令会读取并校验已有缓存，缺失时访问公开数据下载。不是纯环境检查。新的输出目录不要与旧 `data/discovery/2025-hourly` 混用，不修改原 manifest。若协议或下载请求不同，程序拒绝复用是正常保护行为。

检查新 `bars_manifest.json` 的审计结果，核对标的、时间范围、缺口、重复与校验摘要。原始价格数据被重新整理不代表新样本；曝光账本必须一直保留。跨平台重新生成 Parquet 文件也不要求其文件字节与原件完全一致，但行数据、时间语义与来源需核对。

## 11 进行两级小规模研究验收

`configure.py` 创建的 smoke 配置保持 3 个任务、3 个标的、48 小时回看、3 个时间折与 2 个训练种子，把候选降到 2、epochs 降到 4、batch 设为 128、搜索时间设为 1800 秒、研究循环设为 1。训练截止 2025-03-01，开发回放截止 2025-04-05，足以容纳一次 336 小时影子比较及标签延迟；它不是独立基准。

### 11.1 先用规则提案走通整个链路

```bash
.venv/bin/python outputs/dot-deployment-kit/run_bounded.py --seconds 7200 --label rules-smoke-v1 -- \
  .venv/bin/python -m factorlab.discovery --config config/discovery_dot_smoke.json \
  --data data/discovery/linux-2025-v1 evolve --policy config/evolution_dot_smoke.json \
  --output runs/dot-rules-smoke-v1 --memory data/dot-rules-smoke-v1-memory.sqlite \
  --device cpu --search external --worker-backend rules
```

检查生成的模型、研究循环、预测和任务评分，确认退出后主程序与 worker 都结束。规则后端只能证明本地管线，不证明 GPT 调用可用。

### 11.2 再用 Astra 完成一次真实提案验收

这一步会调用目标账户的模型额度，并进行市场历史开发数据训练。前一步通过、登录与模型权限检查就绪后再执行：

```bash
.venv/bin/python outputs/dot-deployment-kit/run_bounded.py --seconds 7200 --label astra-smoke-v1 -- \
  .venv/bin/python -m factorlab.discovery --config config/discovery_dot_smoke.json \
  --data data/discovery/linux-2025-v1 evolve --policy config/evolution_dot_smoke.json \
  --output runs/dot-astra-smoke-v1 --memory data/dot-astra-smoke-v1-memory.sqlite \
  --device cpu --search external --worker-backend codex
```

这个 smoke 的 worker 总请求尝试上限由当前代码计算为 `1 × max(16, 8 × 2) × 3 = 48`，并受 worker 时间、单次 120 秒超时和提案需求共同限制；**不是只会调用一次 GPT，也不是预付费硬上限**。通常实际调用少于上限，以保存的 attempts 和 usage 为准。若需要更低的总请求或 token/金额上限，应在开始运行前加入受审预算配置；当前项目没有精确金额封顶接口。Astra high 若在目标经常超过单次超时，应记录失败并在新版本预先修改预算，不能事后无限重试。

写入停止后检查：

```bash
.venv/bin/python -m factorlab.discovery inspect-evolution runs/dot-astra-smoke-v1 \
  > work/deployment/astra-smoke-inspection.json
.venv/bin/python -m factorlab.discovery evolution-report --run runs/dot-astra-smoke-v1 \
  --output outputs/dot-astra-smoke-v1-report-001
```

验收至少包括：worker 有成功完成记录；提案来源确为 `codex_cli_llm_worker`；模型请求配置为 Astra/high；事件审计没有被拒绝的工具行为；生成的结构或公式通过校验并实际得到训练与 fold 评分。检查 `pending_cycle`、任务 pending 和拒绝原因。只出现一份 JSON 提案，或所有拟合都超时，不能判定完整研究链路通过。

候选全部被拒绝、基准被保留可以是有效研究结果。提升不是部署验收的必要条件。若确实改善，也只能先称为这段开发资料上的结果，不能直接称为可交易因子或独立样本外收益。

## 12 开始三轮有预算的历史演化

小规模验收通过后使用新配置与新目录：

```bash
.venv/bin/python outputs/dot-deployment-kit/run_bounded.py --seconds 14400 --label astra-history-v1 -- \
  .venv/bin/python -m factorlab.discovery --config config/discovery_dot_cpu.json \
  --data data/discovery/linux-2025-v1 evolve --policy config/evolution_dot_cpu.json \
  --output runs/dot-astra-history-v1 --memory data/dot-astra-history-v1-memory.sqlite \
  --device cpu --search external --worker-backend codex
```

CPU 配置保持 V6 的 4 个候选、3 折、2 种子、3 个研究周期及原评分门槛，batch 改为 128、每次搜索预算改为 1800 秒。这些预算变化会影响模型选择，故它是新的 CPU 研究条件，不能直接把分数与旧 900 秒 Windows 配置作公平性能比较。

每轮搜索上限计算涉及 `(候选数 × 折数 + 1) × 种子数`，此处是 26 个拟合预算单位；初始化、回放和其他开销还要另计，不能把 1800 秒当作完整程序总时长。worker 请求尝试上限为 `3 × max(16, 8 × 4) × 3 = 288`，实际仍受时间与请求生成限制。外层 14400 秒是额外的工程停止预算，带清理宽限，不保证能完成三轮。

若资源不足或超时，保留失败记录。分析瓶颈后在新配置/新 run 降低规模或增加预先声明的预算；不要清空已消费轮次、不断重开相同数据搜索直至得到好分数。达到 `max_cycles` 只停止新增搜索，程序仍可能继续预测和结算已有标签。

## 13 需要新行情时的额外流程

历史回放与在线逐步接收行情不同。前面的历史研究不需要 `--watch`。只有数据会继续追加时才使用 feed 和 watch。

### 13.1 先准备近期历史种子

当前 `prepare` 下载 Binance USD-M 小时 K 线的 **monthly** 档案，没有 daily/REST 自动补洞后备路径。不要拿包含未来终点的 live 配置跑 prepare，否则会请求不存在的月份。

以 2026-10-03 为例，可创建单独 seed 配置，设置以下日期，其他 schema 沿用 `discovery_dot_cpu.json`：

```json
{
  "version": "dot-seed-through-202609-v1",
  "start": "2026-08-01",
  "train_end": "2026-08-15",
  "dev_end": "2026-09-01",
  "calibration_end": "2026-09-15",
  "end": "2026-10-01"
}
```

这只是要覆盖到完整配置中的日期字段，不能把这个片段单独当完整配置。生成方式：

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('config/discovery_dot_seed_202609.json')
assert not p.exists(), 'Use a new config name'
v=json.loads(Path('config/discovery_dot_cpu.json').read_text())
v.update(version='dot-seed-through-202609-v1', start='2026-08-01', train_end='2026-08-15',
         dev_end='2026-09-01', calibration_end='2026-09-15', end='2026-10-01')
p.write_text(json.dumps(v,indent=2)+'\n')
PY
.venv/bin/python -m factorlab.discovery --config config/discovery_dot_seed_202609.json \
  --data data/discovery/dot-seed-202609-v1 prepare
.venv/bin/python -m factorlab.discovery feed-init \
  --seed data/discovery/dot-seed-202609-v1/bars.parquet --output data/feeds/dot-live-v1
.venv/bin/python -m factorlab.discovery feed-refresh --feed data/feeds/dot-live-v1 \
  --symbols BTCUSDT ETHUSDT SOLUSDT --max-pages 4
```

实际还会请求回看预热涉及的更早月档。部署时须确认 9 月档已经发布；404 应停止诊断，不能用不完整月档或旧 2025 seed 冒充近期连续数据。日期过期时重新选择已发布月份，并生成新配置、新 seed 和新 feed。

一次手动 refresh 每标的最多 4 页、每页 96 小时，即 384 小时。它不能补一年空缺，也不会自动修补所有内部缺口。重复请求最近窗口不会使更早缺口自动消失。如果 seed 与当前行情差距超过此边界，先补做可靠的历史数据适配或等待可衔接月档，再新建 feed；不要伪造接收时间。

watch 内置 refresh 默认只有 1 页，即最近 96 小时。每批 REST 新数据的可用时间记录为实际收到时刻，补齐的历史 K 线不是曾经真实发布过的预测；每标的最新一根才成为本次新的预测起点。保留 seed、所有 snapshots、journal 与 feed manifest 整套资料。

### 13.2 创建在线研究协议

先检查 seed/feed 连续性、成熟标签和真实接收时间。把 `train_end` 设为部署时刻之后的下一个完整 UTC 小时作为研究起点，先等到该时刻与相应闭合行情到达再启动。示例生成器使用当前 UTC 时钟，**不要对过去回填接收时间**：

```bash
.venv/bin/python - <<'PY'
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
p=Path('config/discovery_dot_live_v1.json')
assert not p.exists(), 'Freeze once; use a new name for a changed protocol'
v=json.loads(Path('config/discovery_dot_cpu.json').read_text())
t=datetime.now(timezone.utc).replace(minute=0,second=0,microsecond=0)+timedelta(hours=1)
v.update(version='dot-live-v1', start='2026-08-01', train_end=t.isoformat(),
         dev_end=(t+timedelta(days=60)).isoformat(),
         calibration_end=(t+timedelta(days=61)).isoformat(), end=(t+timedelta(days=62)).isoformat())
p.write_text(json.dumps(v,indent=2)+'\n')
print('Earliest launch UTC:',t.isoformat())
PY
```

这里 `start` 必须对应实际使用的 seed 起点，执行日期改变时一并调整。60 天开发窗口和后续字段是运行协议示例，不承诺自动完成独立确认；`evolve` 主要在开发窗口内进行研究。创建后把新配置摘要另行记录，再开始新 run。

### 13.3 有界在线运行

```bash
.venv/bin/python outputs/dot-deployment-kit/run_bounded.py --seconds 25200 --label live-window-001 \
  --stop-file work/deployment/STOP-live-window-001 -- \
  .venv/bin/python -m factorlab.discovery --config config/discovery_dot_live_v1.json \
  --data data/feeds/dot-live-v1 evolve --policy config/evolution_dot_cpu.json \
  --output runs/dot-live-v1 --memory data/dot-live-v1-memory.sqlite \
  --device cpu --search external --worker-backend codex \
  --watch --refresh-feed --watch-seconds 21600 --poll-seconds 60
```

这是一段约 6 小时 watch 加外层 7 小时停止预算的示例。`--watch-seconds` 只在一次内部调用返回后检查，初始化和一轮搜索可能额外耗时，所以有外层限制。`--max-ticks` 是每次内部调用的限制，加 watch 后不等于整个服务只处理这么多条。

同一 Linux 版本续跑时，保持源码、依赖锁、协议、数据/feed 身份及**同一记忆路径**，给上面 evolve 参数追加 `--resume`；换新的 launcher `--label` 和 `--stop-file`，原输出目录不变。resume 不会补发中断前没能及时发布的真实预测，也不保证重做已消费的搜索槽。

模型标签要等未来 12 或 20 小时成熟，影子比较需 336 小时。运行一个晚上可能只能验证健康状态与增量预测，不能宣布影子胜出。某个运行退出时仍可有未成熟标签或 pending cycle；不要修改原 `dev_end` 来强行延长，后续窗口另建协议。

## 14 后台启动与停止

### 14.1 优先使用平台提供的进程管理

如果 dot 有明确支持的后台任务或服务管理功能，按它的规则运行相同命令，并记录工作目录、用户、环境、最长运行时间和重启行为。本手册未核实 dot 的保活、休眠、重建或自动重启承诺。

没有服务管理器时，可以把第 13.3 节命令放进项目内 `work/deployment/launch-live.sh`，开头显式设置正确的根路径并 source 环境，然后：

```bash
nohup bash work/deployment/launch-live.sh > work/deployment/launcher-window-001.log 2>&1 < /dev/null &
printf '%s\n' "$!" > work/deployment/launcher-window-001.pid
```

这是有限时长的后台进程。`nohup` 不抵抗 OOM、容器回收、平台休眠、重启或配额限制。记录 PID 只能辅助排查，不要隔很久后未经身份核对直接对旧 PID 发送信号。

推荐使用启动时指定的停止文件请求结束：

```bash
touch work/deployment/STOP-live-window-001
```

外层启动器先通知主进程合作退出，再在宽限后清理自己创建的进程组。等待日志/receipt 确认所有后代结束，再做导出与检查。停止文件每次用新名称，不能把旧 STOP 文件带入新启动。

### 14.2 只有确实支持时才使用 systemd

Debian 容器不等于运行着 systemd；先检查 `systemctl --user status` 是否实际可用，以及退出登录后 user service 是否会保留。若无权限或无用户服务管理器，不强行安装它。

可用时建立用户 unit，下面路径都要换成真实绝对路径。`launch-live.sh` 必须包含确定的根目录、正确的 Codex PATH、source 环境和一次有界命令；首次启动和 resume 脚本不能混淆：

```ini
[Unit]
Description=Bounded FactorResearch session

[Service]
Type=exec
WorkingDirectory=/实际项目挂载点/FactorResearch
ExecStart=/bin/bash /实际项目挂载点/FactorResearch/work/deployment/launch-live.sh
KillMode=mixed
TimeoutStopSec=60
Restart=no
MemoryAccounting=yes
MemoryMax=7G
CPUQuota=200%

[Install]
WantedBy=default.target
```

`7G` 和 200% 是保守初始限制，须根据实际 cgroup 上限调整；限制内存不会让大任务自动缩小，触顶可能 OOM。用户服务只有在相应控制器被委派时才会实际执行限制，核对 `systemctl --user show` 和实际 cgroup 文件，不只看 unit 语法。

`KillMode=mixed` 给主进程先处理 TERM，再清理组内残余进程，`TimeoutStopSec` 留清理时间。这里刻意使用 `Restart=no`，避免重复异常、重新消耗预算或把初次启动不断重放。参见 [Debian 13 systemd.kill](https://manpages.debian.org/trixie/systemd/systemd.kill.5.en.html) 和 [systemd.service](https://manpages.debian.org/trixie/systemd/systemd.service.5.en.html)。

项目现有程序具备有预算的多轮研究、状态记录和同身份 resume，**尚不是无限运行、自动补齐任意断档、自动开新协议并永久运维的完整服务**。不要用无限 `while true` 和不断创建新 run 伪装该能力。

## 15 监测和资源调整

同时查看 `work/deployment/logs/` 下外层日志与 receipt、run 的 `worker.log`、worker 状态文件、research 候选结果与模型记录。检查汇总报告前先停止写入，避免把写入中的多个文件当作一致快照。

| 观察项 | 需要判断的内容 | 动作 |
|---|---|---|
| 完整退出记录 | 返回码、timeout/stop/正常结束、是否清理后代 | 失败保留原运行，先诊断，不无界重试 |
| 内存 | 整个进程树 RSS 与容器 memory.current/peak/events | 留头部空间；RSS 相加可能重复计算共享页，不能冒充实际容器占用 |
| 磁盘 | `.venv`、`.cache`、runs、feed snapshots 的增长 | 定期导出完整档案；不自动删哈希链引用的快照 |
| 数据进度 | 最新接收时间、缺口、最新预测起点 | 检查行情请求和 feed 连续性；不能按进程存活判定正常 |
| GPT | attempts/completed/errors、usage、启动警告 | 确认是模型提案而非规则回退；额度与频率限制另行处理 |
| 训练 | 成功 fold 数、超时/数值失败、候选通过率 | 全部失败是管线未验收；无提升不等于失败 |
| 研究 | 基准与候选损失、稳定性、影子门槛、pending | 维持冻结标准，保留拒绝记录 |

按 3 标的、1 年小时数据、48 回看、12 特征、float32，仅输入窗口约 58 MiB；36 特征约 173 MiB。**这不是内存峰值估计**：数据子集复制、多折数据、float64 标准化、Torch 张量、模型和依赖都额外占用。仅减小 batch 不会消除整个数据矩阵的占用。

先单进程、2 线程、短数据与小候选集，测出峰值后再扩大。9 个共享 CPU 不适合据此直接开 9 份并行训练。不要一次安装 GPU 全家桶、同时运行 Jupyter/MLflow 服务与多份研究进程来验证最小部署。

## 16 备份 恢复和升级

1. 停止所有研究、worker 与 feed 写入，核对退出记录和后代清理。
2. 使用 `transfer.py pack` 封包完整当前部署，目标放在 `work/migration_export/` 下的新名字。此工具在 Linux 也可使用；不要用它封包一个仍在写入的在线运行。
3. 将包和摘要移到实例外的持久存储，独立核对摘要。当前 overlay 内副本不能覆盖实例丢失风险。
4. 在全新目录试解包。逐文件摘要通过后检查所有实际 SQLite 的 `PRAGMA integrity_check`，必须返回 `ok`，并核对模型、source 和 feed 文件完整。
5. 恢复依赖使用此 Linux 部署的已冻结 `uv.lock` 和 Python 版本，不重新解析；重新登录或检查现有 Codex，验证模型可用性。
6. **同一 Linux 部署恢复到原绝对路径**且身份核验通过后，才使用原参数加 `--resume`。恢复到不同绝对路径会受现有身份契约限制，先新 run 或开发明确迁移支持，不能承诺换路径无缝恢复。

SQLite 校验可在停机后的恢复副本上遍历执行；不要把存在 `.sqlite` 文件当作它已通过校验。下面核验 `data/runs/models` 中的 `.sqlite`、`.sqlite3`、`.db`（包括 MLflow 库）；执行者还须根据档案清单，把 `work/` 中实际正式库逐个加入 `extra_databases`。人工损坏测试库应由明确的文件清单区分，不可因为路径包含 `fixture` 或 `pytest` 就跳过正式库。

```bash
.venv/bin/python - <<'PY'
import sqlite3
from pathlib import Path
bad=[]
paths=set()
extra_databases=[]  # 在这里加入经档案清单确认的正式 work 数据库相对路径
for base in ('data','runs','models'):
    for pattern in ('*.sqlite','*.sqlite3','*.db'):
        paths.update(Path(base).rglob(pattern))
paths.update(Path(p) for p in extra_databases)
for p in sorted(paths):
    try:
        con=sqlite3.connect(p.resolve().as_uri()+'?mode=ro',uri=True)
        try: rows=con.execute('PRAGMA integrity_check').fetchall()
        finally: con.close()
        if rows != [('ok',)]: bad.append((str(p),rows))
    except sqlite3.Error as exc:
        bad.append((str(p),type(exc).__name__))
print({'checked':len(paths),'failed':bad})
assert not bad
PY
```

如果 SQLite、模型摘要、feed 链或演化 manifest 检查失败，保留整个故障副本并停止续跑。优先从最近已验证的完整停机备份恢复到新的检查目录；不要删除损坏状态库后在原 run 中重建，也不要改摘要使校验通过。原运行无法恢复时，保留损坏证据，在新 run 从经过验证的数据重新开始。

新增算法、改 Python/依赖、换 Agent 模型或调整评价窗口应形成新版本与新 run，保留旧部署包。回滚时恢复旧源码及其锁文件、原路径和参数；不把新状态库交给旧代码盲目续跑。环境重建后 Codex 登录可能失效，恢复验收不可省略。

## 17 常见故障与处理顺序

| 现象 | 优先检查 | 处理原则 |
|---|---|---|
| `Project path must remain within` 或 Windows drive 拒绝 | 环境是否先 source；是否使用旧绝对 manifest | 新目录生成 Linux 数据准备记录，不改旧身份 |
| `Frozen evolution identity changed on resume` | 源码、锁文件、协议、记忆路径是否改变 | 恢复完全相同部署，或新建 run |
| uv 不支持平台或锁过期 | 是否还在使用 Windows profile/lock | 保存原锁后在目标 Linux 正式解析新锁 |
| 编译耗尽内存、轮子不存在 | 失败包的 Linux/CPython 3.12 轮子 | 明确评估构建依赖，不无约束升级全部包 |
| `BackendExitCode_2` | probe 的 stderr 与完整 CLI 参数 | 先修兼容；不删除隔离和 schema 参数 |
| `CodexCLIUnavailable` | 服务用户的 PATH 与实际 codex 路径 | 服务显式保留已验证路径，不复制 Windows exe |
| 模型无权限或超时 | 同一用户登录、Astra 可用性、120 秒单次上限 | 不静默换模型，不无限重试 |
| 候选全部 fit 失败 | CPU预算、数据连续性、数值错误 | 保留失败证据；新 run 预先改规模或预算 |
| 137 或 OOM | cgroup memory.events、日志、临时目录是否 tmpfs | 降低规模，检查半写产物；不把退出当正常完成 |
| watch 结束却仍 pending | 影子期、标签成熟时间、协议终点 | 如实记录未完成证据，不追改未来标签 |
| feed 跟不上或有长缺口 | seed日期、96/384小时窗口限制 | 先完成可靠数据衔接，不反复抓最近窗口假装补齐 |
| 进程正常但没有新评分 | 闭合行情、成熟标签、最新预测起点 | 区分数据等待与程序停滞 |

## 18 最终验收清单

部署执行者应保存以下结果，逐项注明通过、失败或未执行：

- 整包外部 SHA256 与内部逐文件校验通过；重要 SQLite、原始归档、模型和审计齐全。
- 工作目录确在目标项目 overlay；真实 CPU/内存/cgroup 配额已记录；异地备份位置明确。
- Linux 路径适配、CPU依赖锁、解释器版本和线程配置已核对；完整 imports、lint、tests 有目标日志。
- 目标 Codex 实际参数兼容、登录有效；一次 Astra/high 真实研究的事件与 usage 有记录。
- 小规模规则链路和 GPT 链路都有实际拟合与评分；候选被拒绝也完整保留。
- 正常退出、超时、主动停止及后代清理在 Linux 做过实际验证；不存在残留研究进程。
- 新 run 身份和参数被冻结；2025 数据仍标记为已暴露开发资料；没有把部署成功写成预测有效。
- 在线运行若启用，seed衔接、真实接收时间、pending状态及 336 小时影子期被正确理解。
- 停机备份、实例外保存和一次新目录恢复检查完成；同路径 resume 前再做身份核验。

可以把下述要求交给目标机器中的 Codex 作为部署任务：

> 阅读本手册和附件。先核实真实项目 overlay 路径与资源，按手册迁移并建立新的 Linux CPU 部署，研究 proposer 使用 gpt-6-astra/high。保留全部旧证据，不改旧 run 摘要，不复制 Windows 环境或凭据。按顺序执行路径适配、依赖锁定、CLI兼容、工程测试、停机测试和两级有界研究验收；失败时记录原因并解决，不放宽评价门槛或自动换模型。最终交付部署版本摘要、日志、资源峰值、实际评分结果、停止与恢复证据及未完成事项。持续运行与独立因子有效性须分别报告。

本手册的源码核对材料在 `work/deployment_manual/runtime_review.txt` 和 `data_review.txt`；旧研究结果见 `outputs/ROBUST_V6_IMPROVEMENT_REPORT.txt` 与 `work/v6_improvement/`。附件本地检查和交付审查记录也保存于 `work/deployment_manual/`，它们不替代目标 Linux 验收。
