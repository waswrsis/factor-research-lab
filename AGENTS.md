# Published source snapshot

This Git repository is the source and deployment-guide portion of the project.
For a requested Linux/cloud deployment, follow docs/GITHUB_RELEASE_DEPLOYMENT.md
first: clone to a source reference directory, download and verify the complete
Release asset, and unpack into a separate new runtime root on the project overlay.
The runtime root is FACTOR_RESEARCH_ROOT. Its Linux migration profile supersedes
the historical Windows/D-drive setup instructions preserved below. Do not create
a fake D: directory on Linux. Preserve all prior research evidence and create a
new Linux run identity. A source clone alone does not include the full data/models.
For operations on the original Windows workstation, retain D:/FactorResearch.

## Historical workspace instructions

# Project workspace

The user requested that this project's files and subsequent work live on drive D.

- Use `D:/FactorResearch` as the working directory for project commands.
- Keep project scratch files in `work/` and user-facing deliverables in `outputs/`.
- If later authorized to install project dependencies or download research data, keep project-specific environments, caches, datasets, checkpoints, and logs under this directory.
- The previous C-drive workspace is a retained backup. An automatic approval check rejected its cleanup; no compatibility junctions were created.
- Do not relocate shared Codex runtimes, application installations, Windows fonts, or global configuration as part of this project migration.
- `work/feasibility_audit/` contains an earlier, unexecuted exploratory draft. It is not evidence of a completed data audit or an approved training run. The current design report remains paper research.
- This migration does not itself authorize model training, bulk data downloads, or live trading.

Migration verification is recorded in `work/migration/migration_manifest.json`.

## Environment setup authorization and operation

The user subsequently authorized subagents, installation of open-source projects,
and environment preparation while preserving the D-drive workspace.

- Before project Python/package commands, dot-source `scripts/Enter-Research.ps1`
  so temporary files and library caches are directed to this project.
- Use `.venv/Scripts/python.exe` and `.tools/uv/uv.exe`; the managed Python base
  is also on drive D under `.tools/python/`.
- Dependency choices are recorded in `pyproject.toml`, with exact resolutions
  and package hashes in `uv.lock`. Reapply with `scripts/Sync-Research.ps1`.
- `external/qlib` is a pinned Qlib source reference; the importable package is
  installed in `.venv`. `external/rd-agent` is source reference only because
  its official runtime requires Linux. Do not claim the autonomous agent is running.
- Only environment smoke checks on tiny artificial inputs have been authorized
  by the current setup scope. No market dataset, fitted strategy, or live
  execution is implied by a successful environment check.

## Phased implementation authorization (latest user goal)

The latest user goal authorizes phased implementation of the research platform,
with semantic checks and code audits at the end of every phase. This supersedes
prior environment-only / paper-only scope above. Public market-data ingestion,
local model training, offline backtests, architecture search and research-agent
integration are within this implementation scope. No live exchange trading is
requested or authorized.

- All code, data, caches, experiment artifacts and audit reports remain on D.
- Use small bounded experiments appropriate for 32 GB RAM / 8 GB GPU initially.
- Freeze evaluation protocols before results; never silently alter cost, split,
  risk, horizon or promotion criteria to improve a score.
- Every phase needs semantic and code review evidence, tests, resolved findings,
  and explicit limitations. Synthetic checks are not market evidence.
- Never claim historical profitability, general asset-universe validity, live
  operation or future forward performance without actual supporting evidence.
- Existing historical-source download drafts are not approved implementations.
- Subagents may implement separate owned modules and independently review work.

## Subagent preference

The user explicitly requests `gpt-6.1-sol` with `high` reasoning for subagents.
When spawning a new subagent, set that model and reasoning effort explicitly,
and use fork_turns="none" or a bounded history count as the tool requires.
The user's wording "gpt66.1sol high" was interpreted as GPT-6.1 Sol / High.

## Continuous evolution extension

The user subsequently requested completion of the iterative/evolving research loop.
Current work adds bounded population search, development-only research memory,
an independent Codex proposal worker, causal shadow promotion, recoverable state,
and optional public closed-candle data feeds. See docs/EVOLUTION_IMPLEMENTATION.md
for requirement-by-requirement acceptance, not the old paper/environment-only notes.
Keep new feed directories separate from immutable historical archive caches.
Recorded API receipt time must not be backdated into historical prediction origins.
Do not alter frozen protocol/source identity in order to resume an old algorithm
under changed code. Preserve old results; new algorithm versions need new runs.

## Autonomous operator research extension

The user clarified that autonomy means no manual intervention per research iteration,
with GPT checking and revising candidates. It does not mean label-free training.
Version 5 adds new causal expression programs, fitted internal parameters, typed
repair feedback and compatible program revival. See docs/AUTONOMOUS_OPERATOR_GUIDE.txt
and outputs/OPERATOR_V5_ACCEPTANCE.txt for actual scope and evidence.
Generated programs remain data executed by a trusted bounded interpreter; readable
formula listings are not executable Python. Do not claim arbitrary source rewriting,
unlimited research, independent predictive benefit or live trading from this feature.
Preserve version 4 acceptance records. Keep evaluation and evidence boundaries frozen.

## Version 6 robust research validation

Version 6 adds optional primary-task checkpoints, cold chronological fold selection,
auxiliary shadow safeguards, simple controls, and typed robustness feedback to GPT.
See docs/ROBUST_RESEARCH_V6.txt and outputs/ROBUST_V6_IMPROVEMENT_REPORT.txt.
The 96-fit frozen market comparison retained the original baseline; the combined
method did not improve the benchmark. A checkpoint-only branch had a 0.14% point
gain without supported improvement. All 2025 periods remain exposed development data.
The separate real-GPT check generated and fitted two programs, rejected both, and
retained its reference. Synthetic controls are engineering evidence, not market alpha.
Keep all V5 and V6 artifacts, including rejected candidates and invalidated untrained
preparation directories. Further algorithm changes require new frozen run identities.
Completion requirements and independent audits are in work/v6_improvement/.
