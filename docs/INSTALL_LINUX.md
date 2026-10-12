> 2026-10-12：原服务器已按用户后续授权启动。本文的“不启动”步骤仍是第三方安装默认行为；原安装快照保留，最新启动快照见 STARTUP_STATUS.json。

# Linux安装：完整程序就位，服务保持关闭

对应 `v7-rolling-cpu1-2026-10-12`，仓库 `waswrsis/factor-research-lab`。源码、依赖与配置已安装，原服务器后续按用户授权启动；第三方安装默认保持关闭。[原安装摘要](INSTALLATION_SUMMARY.json)与[启动快照](STARTUP_STATUS.json)分别记录当时状态，不代表完整科研验收。剩余前向/GPT对照已取消，不能记成通过。

## 环境与源码

目标：Linux x86_64、glibc ≥2.28、已有CPython3.12及venv支持。实际安装使用Python3.12.3。以下命令供操作者安装使用，均不运行拟合/GPT或启动服务；选择一个源码获取方式并使用全新目录。

从仓库固定tag获取（tag发布后可用）：

```sh
git clone --branch v7-rolling-cpu1-2026-10-12 --depth 1 https://github.com/waswrsis/factor-research-lab.git factor-research-lab
cd factor-research-lab
```

或从同一Release下载 `source-core-rolling-cpu1.tar`、`SHA256SUMS`、源包manifest和安装指南。对照Release公布的校验值；校验文件本身须来自可信发布位置，而非未知第三方。

```sh
curl --fail --location --remote-name https://github.com/waswrsis/factor-research-lab/releases/download/v7-rolling-cpu1-2026-10-12/source-core-rolling-cpu1.tar
curl --fail --location --remote-name https://github.com/waswrsis/factor-research-lab/releases/download/v7-rolling-cpu1-2026-10-12/SHA256SUMS
sha256sum --check --ignore-missing SHA256SUMS
```

core tar SHA-256为 `927baff95d80441ccc13441c0f93448a9ad6ee8d1dbde797b67ce5939034ce56`，4,669,440字节、234文件。至少确认该tar的校验行显示OK；不能把跳过缺少附件当作全部附件验证。源包map hash `f5842afbf3b21f859d8d4f998918305ca993bf606b9da92392aaa23cb6502ce6` 与研究身份 `4b5f44c951ef5f0c58592fe12557908b1d12774a20f22583755125d046fdb88a` 含义不同：前者覆盖打包记录，后者覆盖220项研究资产。

推荐用固定tag中已经审阅的 `scripts/v7_linux_bootstrap.sh` 做纯包验证和安全解包。其 `verify`/`unpack` 不安装服务；目标目录必须尚不存在，父目录须存在。

```sh
python3.12 /path/to/tag-checkout/scripts/v7_linux_bootstrap.sh verify --package /path/to/source-core-rolling-cpu1.tar --sha256 927baff95d80441ccc13441c0f93448a9ad6ee8d1dbde797b67ce5939034ce56
python3.12 /path/to/tag-checkout/scripts/v7_linux_bootstrap.sh unpack --package /path/to/source-core-rolling-cpu1.tar --sha256 927baff95d80441ccc13441c0f93448a9ad6ee8d1dbde797b67ce5939034ce56 --workspace /absolute/new-workspace
cd /absolute/new-workspace
```

仓库包含测试源码；core包有意不含测试、历史实验、数据、模型、钱包、认证或环境缓存。私有历史校准资料不是完整主程序的必需组成部分。若单独提供有限公开训练样本，它是有来源/hash的历史输入，不是实时feed、已登记campaign或真实前向证据。

## 独立CPU环境与最低限度入口检查

在选定源码根目录中操作，`.venv`必须是新环境。不要执行根目录的Windows/CUDA `uv sync` 或 `pip install -e .`。完整CPU神经清单已包含基础CPU依赖及明确wheel哈希；普通源码安装直接使用独立venv，不依赖旧服务器preflight。

```sh
export FACTOR_RESEARCH_ROOT="$PWD"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p .cache/pip tmp
export TMPDIR="$PWD/tmp"
export PIP_CACHE_DIR="$PWD/.cache/pip"
python3.12 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: --no-deps --require-hashes -r config/v7-linux-neural-requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -B -s -m factorlab.discovery.v7_cli --help
.venv/bin/python -B -s -m scripts.v7_long_running --help
```

`config/v7-linux-requirements.txt` 是固定基础CPU清单；上述神经清单包含这些依赖，并与 `config/v7-linux-neural-lock.json` 对应。缺少适配wheel、hash不符或依赖不一致就保留错误，不放宽锁、不改版本、不临时编译来伪装相同环境。无需运行pytest、市场拟合或GPT调用。入口返回0只说明入口可用，不是端到端科研通过。

历史bootstrap的 `bootstrap`/服务生成步骤以及旧P6初始化工具绑定原服务器旧服务和STOP。第三方不要伪造这些回执或钱包，也不要用移除检查的方法使它们通过。上面的普通源码安装不使用这些特定服务器认证流程。

## 安装后不启动

原安装快照中的unit为loaded、inactive、MainPID=0、disabled；后续授权启动记录另见STARTUP_STATUS.json。公开附件不提供已登记的私人campaign或认证资料，也不要求安装者复制它们。第三方安装步骤不包含服务start/enable或长期研究命令。

首次激活需另行授权，使用自己的真实feed、明确的新配置、source身份和额度冻结；历史样本导入不能伪装实时收件，首次实际数据就绪才确定锚点。研究模式为90日训练窗、15经过日拟合、每日视图刷新；分钟轮询处理小时闭合数据，12小时标签真实成熟后才结算。UTC自然月额度不因安装或重启重发。服务资源安排须核验实际systemd/cgroup，不以配置文本代替运行证据。

模型为gpt-6-astra/high，认证由操作员自行管理，不随包发布；知识截止日期未知。项目不执行交易，旧校准未通过，不声称可靠alpha或完整算法统计功效。Windows首次获取uv的步骤仍不完整，封存fixture缺省仅限制历史验收复现。

当前运行配置使用最多1逻辑CPU核、768MiB内存；CPU25%的旧验收资料不能充作这个配置的完整验收。首次连续90日训练在真实导入后启动，研究cutoff按实际调度时刻冻结。forward_directory与feed_directory须是规范绝对路径。失败尝试及其预付额度保留；本版本保守地在所有月份扣除前驱已消费的forward4solve/1200秒和S21solve/900秒，实际共享月上限1575solve/5100秒，部分后续研究槽可能因额度不足跳过。原安装Release保留；新版实际状态见STARTUP_STATUS.json，不声称有可靠alpha。
