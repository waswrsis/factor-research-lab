# 历史 V6 Release 部署记录（2026-10-03）

**本页只适用于下述旧 V6 包，不是当前 V7 的安装或上线步骤。** 不要把旧包的身份、锁文件改写、测试或研究启动命令用于当前V7源码。
当前版本的完整 Release/实际部署尚待最终交付绑定；入口与状态见根目录 README.md、START_HERE.txt。以下旧版本包尺寸、摘要与执行记录保留原义。

此指南对应公开仓库 `waswrsis/FactorResearch` 的 `deployment-2026-10-03-v1` Release。目标是取得完整项目，在 dot 云端项目磁盘中完成新的 Linux CPU 部署。部署助手应先读本指南，再读完整包中的 `outputs/DOT_DEPLOYMENT_MANUAL.md`。

Release 完整包为 237,295,877 字节，文件名 `factorresearch-portable-20261003-v1.zip`，SHA256 为：

```text
ea2b5a4dc795b480059d4747dbb0e56b6a8a3aec8bc8bcb562f5f66ccae4b453
```

已有 14,343 文件的校验和 22 个正式数据库的本地恢复检查。Linux 安装、Linux 退出测试、Astra 实际调用与长期运行尚未在目标验收。完整包保留 Windows 原始源码与锁文件，因此需要下面的 Linux 适配步骤。

## 1 选择正确的磁盘目录

以下命令在目标 Bash 中执行。把第一行替换成实际 32 GB 项目 overlay 的挂载路径；不要使用 `/tmp` 或 tmpfs 根目录。预留至少约 1 GiB 给包与解压文件，另为 Python 环境、安装缓存和后续实验留足数 GiB 空间。

```bash
set -euo pipefail
cd '/实际项目overlay挂载点'
pwd
df -hT .
findmnt -T .
command -v git
command -v curl
export PYTHON_BIN="$(command -v python3.12 || command -v python3)"
"$PYTHON_BIN" --version
```

需要 Git、curl 和目标 Python 3.12。公开仓库与公开附件可直接读取，不需要把 GitHub token 写进命令。下载不到时先检查出站网络、代理和平台限制。

## 2 用 Git 获取对应版本和下载完整附件

三个目录 `FactorResearch-source`、`factorresearch-download`、`FactorResearch` 都应使用新的位置。不要把解包覆盖到已存在的项目上。

```bash
git clone --depth 1 --branch deployment-2026-10-03-v1 \
  https://github.com/waswrsis/FactorResearch.git FactorResearch-source

mkdir factorresearch-download
curl --fail --location --retry 3 \
  --output factorresearch-download/factorresearch-portable-20261003-v1.zip \
  https://github.com/waswrsis/FactorResearch/releases/download/deployment-2026-10-03-v1/factorresearch-portable-20261003-v1.zip
curl --fail --location --retry 3 \
  --output factorresearch-download/factorresearch-portable-20261003-v1.zip.sha256 \
  https://github.com/waswrsis/FactorResearch/releases/download/deployment-2026-10-03-v1/factorresearch-portable-20261003-v1.zip.sha256

(cd factorresearch-download && sha256sum --check factorresearch-portable-20261003-v1.zip.sha256)
```

`git clone` 获得程序源码和部署工具；上面的 Release 附件补齐全部数据、模型与实验档案。不要仅下载 GitHub 自动提供的 `Source code (zip)` 后就开始寻找数据。

如果目标已安装 GitHub CLI，也可用以下命令代替两个 curl 下载命令，选择一种方式即可：

```bash
gh release download deployment-2026-10-03-v1 --repo waswrsis/FactorResearch \
  --pattern 'factorresearch-portable-20261003-v1.zip*' --dir factorresearch-download
```

参见 [GitHub CLI 的 Release 下载说明](https://cli.github.com/manual/gh_release_download)。附件发布方式符合 [GitHub Releases 文档](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)，无需把完整实验二进制纳入 Git 历史。

## 3 校验并恢复到运行目录

```bash
"$PYTHON_BIN" FactorResearch-source/outputs/dot-deployment-kit/transfer.py unpack \
  factorresearch-download/factorresearch-portable-20261003-v1.zip FactorResearch
cd FactorResearch
mkdir -p work/deployment
cp --no-clobber _TRANSFER_MANIFEST.json work/deployment/received-release-20261003-v1.json
export FACTOR_RESEARCH_ROOT="$PWD"
```

恢复工具会核对整个压缩包的外部摘要和每个文件的内部摘要，拒绝不安全路径和覆盖已有目录。完整原始资料约 653 MB；不要删除原包和 checksum，直至备份与目标验收完成。

运行目录是新解出的 `FactorResearch`。`FactorResearch-source` 作为带 Git 的源代码参照保留，它不含完整实验档案。后续所有研究操作在运行目录进行。

## 4 应用 Linux CPU 配置

先确认 Python 版本，再预览适配。适配工具只允许在 Linux 应用；不会把 D 盘旧 run 改成可恢复的新身份。

```bash
"$PYTHON_BIN" -c 'import sys; print(sys.version); assert sys.version_info[:3] == (3,12,14)'
"$PYTHON_BIN" outputs/dot-deployment-kit/apply_linux_profile.py --root "$FACTOR_RESEARCH_ROOT"
cat work/linux_migration/proposed.json
"$PYTHON_BIN" outputs/dot-deployment-kit/apply_linux_profile.py --root "$FACTOR_RESEARCH_ROOT" --apply
source outputs/dot-deployment-kit/environment.sh
```

适配包括：正确的 Linux 根路径、CPU Torch 来源、2 线程起点、合作退出与子进程清理，以及研究 proposer 的 `gpt-6-astra / high` 参数。原文件与 Windows 锁保存在 `work/linux_migration/original/`。

实际目标 Python 若不是 3.12.14，应停止核对配置；不要跳过版本断言并宣称验收完成。已有 Codex CLI 的安装、登录位置不迁移，保持 `HOME` 和 `CODEX_HOME`。

## 5 按迁移手册完成安装与验收

从 `outputs/DOT_DEPLOYMENT_MANUAL.md` 第 7 节继续，依次执行：

1. 在项目内建立 uv/CPU Python 环境，在 Linux 重新生成 `uv.lock` 并锁定安装。
2. 核验 Codex 版本、登录、完整参数兼容性；确认目标账户可调用 Astra/high。
3. 工程 imports、lint、tests 和 Linux 停机测试；保存真实检查日志。
4. 生成新的 Linux 数据准备目录和有限预算配置。
5. 先规则提案验收，再真实 GPT 有界研究验收。后者会消耗模型额度并执行历史训练。
6. 通过后再安排三轮研究或接入近期行情 feed，设置停止预算与异地备份。

旧 Windows 虚拟环境未包含在 Release，不能直接运行 `.venv/Scripts/python.exe`。新 Linux 入口使用 `.venv/bin/python`。历史曝光账本必须保留，旧 run 与 rejected 候选完整保留作证据；不要为让新模型通过而清空记忆、删除失败或降低原门槛。

## 6 交给目标 Codex 的执行要求

> 本次部署使用 waswrsis/FactorResearch 的 deployment-2026-10-03-v1 Release。按本指南校验、解包，在实际项目 overlay 中完成新的 Linux CPU 部署，研究 proposer 使用 gpt-6-astra/high。先安装和工程验收，再执行有界研究验收。保留原始档案与曝光账本，不修改旧摘要，不自动换模型，不做交易。最终报告目标环境、源码和依赖摘要、测试结果、实际模型调用、资源峰值、研究评分、停机恢复记录及未完成事项。发布包或进程启动成功不代表已发现有效因子。
