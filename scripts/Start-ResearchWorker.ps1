param(
    [Parameter(Mandatory=$true)][string[]]$Exchange,
    [ValidateSet('rules','command','codex')][string]$Backend = 'codex',
    [string]$StateDir = 'work/research_worker',
    [string]$MemoryPath = 'data/discovery_research_memory.sqlite',
    [string]$Model = 'gpt-6.1-sol',
    [string]$CommandJson,
    [string]$StopFile,
    [int]$MaxRequests = 100,
    [int]$MaxSeconds = 3600,
    [switch]$Once
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/Enter-Research.ps1" -Quiet
$researchWorkerArgs = @('-m', 'factorlab.discovery.agent_worker', '--backend', $Backend,
    '--state-dir', $StateDir, '--memory-path', $MemoryPath,
    '--max-requests', $MaxRequests, '--max-seconds', $MaxSeconds)
foreach ($researchExchange in $Exchange) { $researchWorkerArgs += @('--exchange', $researchExchange) }
if ($Backend -eq 'codex') { $researchWorkerArgs += @('--model', $Model, '--reasoning-effort', 'high') }
if ($CommandJson) { $researchWorkerArgs += @('--command-json', $CommandJson) }
if ($StopFile) { $researchWorkerArgs += @('--stop-file', $StopFile) }
if ($Once) { $researchWorkerArgs += '--once' }
& 'D:/FactorResearch/.venv/Scripts/python.exe' @researchWorkerArgs
exit $LASTEXITCODE
