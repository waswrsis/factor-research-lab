param(
    [ValidateSet('Help','Run','Resume','Inspect','Report')][string]$Mode = 'Help',
    [string]$Output = '',
    [string]$ReportOutput = '',
    [string]$Config = 'config/discovery_v6.json',
    [string]$Policy = 'config/evolution_v6.json',
    [string]$Data = 'data/discovery/2025-hourly',
    [string]$Memory = 'data/research_memory_v6.sqlite',
    [ValidateSet('codex','rules')][string]$Backend = 'codex',
    [ValidateSet('cpu','cuda')][string]$Device = 'cpu'
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Enter-Research.ps1') -Quiet
if ($Mode -eq 'Help') {
    Write-Output 'Version 6: primary-task checkpoints, cold walk-forward search and auxiliary shadow safeguards.'
    Write-Output 'Read docs/ROBUST_RESEARCH_V6.txt before starting a new bounded run.'
    Write-Output 'Run: ./scripts/Start-RobustResearch.ps1 -Mode Run -Output runs/robust-v6-001'
    Write-Output 'Resume: ./scripts/Start-RobustResearch.ps1 -Mode Resume -Output runs/robust-v6-001'
    Write-Output 'Inspect: ./scripts/Start-RobustResearch.ps1 -Mode Inspect -Output runs/robust-v6-001'
    Write-Output 'Report: ./scripts/Start-RobustResearch.ps1 -Mode Report -Output runs/robust-v6-001 -ReportOutput outputs/robust-v6-001'
    Write-Output 'Three forward folds, two seeds, four candidate suites, 900 seconds per search including final cold refit.'
    Write-Output 'Two-week shadow evaluation; the default GPT worker uses the existing Codex login, gpt-6.1-sol/high.'
    Write-Output 'The default data are historically exposed development data; no independent performance claim is implied.'
    return
}
& (Join-Path $PSScriptRoot 'Start-AutonomousResearch.ps1') -Mode $Mode -Output $Output `
    -ReportOutput $ReportOutput -Config $Config -Policy $Policy -Data $Data -Memory $Memory `
    -Backend $Backend -Device $Device
if ($LASTEXITCODE -ne 0) { throw "Robust research exited with code $LASTEXITCODE" }
