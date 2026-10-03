param(
    [ValidateSet('Help','Run','Resume','Inspect','Report')][string]$Mode = 'Help',
    [string]$Output = '',
    [string]$ReportOutput = '',
    [string]$Config = 'config/discovery_v5.json',
    [string]$Policy = 'config/evolution_v5.json',
    [string]$Data = 'data/discovery/2025-hourly',
    [string]$Memory = 'data/research_memory_v5.sqlite',
    [ValidateSet('codex','rules')][string]$Backend = 'codex',
    [ValidateSet('cpu','cuda')][string]$Device = 'cpu'
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Enter-Research.ps1') -Quiet
$autonomousPython = Join-Path $env:VIRTUAL_ENV 'Scripts/python.exe'

if ($Mode -ne 'Help' -and -not $Output) {
    throw 'Specify -Output for the research run directory.'
}

switch ($Mode) {
    'Help' {
        Write-Output 'Autonomous causal-operator research, version 5. Default action: help only.'
        Write-Output 'Run starts a new research directory; Resume continues its frozen source/config identity.'
        Write-Output 'Default worker: Codex gpt-6.1-sol / high, using the existing Codex login.'
        Write-Output 'Rules is a deterministic fallback and does not call a language model.'
        Write-Output 'Run:     ./scripts/Start-AutonomousResearch.ps1 -Mode Run -Output outputs/autonomous-v5-001'
        Write-Output 'Resume:  ./scripts/Start-AutonomousResearch.ps1 -Mode Resume -Output outputs/autonomous-v5-001'
        Write-Output 'Inspect: ./scripts/Start-AutonomousResearch.ps1 -Mode Inspect -Output outputs/autonomous-v5-001'
        Write-Output 'Report:  ./scripts/Start-AutonomousResearch.ps1 -Mode Report -Output outputs/autonomous-v5-001 -ReportOutput outputs/autonomous-v5-report-001'
        Write-Output 'Optional: -Device cuda, -Backend rules, -Config, -Policy, -Data, -Memory.'
        Write-Output 'Default limits: 3 cycles, 4 fitted candidates per cycle, 2 training seeds, 600 seconds per search, 120 seconds per seed fit.'
        Write-Output 'The existing launcher bounds the worker to 288 requests and 3600 seconds, and stops it when the research process exits.'
        Write-Output 'This entry processes the configured local dataset; it does not enable a live feed watch.'
        & $autonomousPython -m factorlab.discovery evolve --help
    }
    { $_ -in 'Run','Resume' } {
        $autonomousArguments = @('-m','factorlab.discovery','--config',$Config,'--data',$Data,
            'evolve','--output',$Output,'--policy',$Policy,'--device',$Device,
            '--memory',$Memory,'--search','external','--worker-backend',$Backend)
        if ($Mode -eq 'Resume') { $autonomousArguments += '--resume' }
        & $autonomousPython @autonomousArguments
    }
    'Inspect' {
        & $autonomousPython -m factorlab.discovery inspect-evolution $Output
    }
    'Report' {
        if (-not $ReportOutput) { throw 'Specify -ReportOutput for a new report directory.' }
        & $autonomousPython -m factorlab.discovery evolution-report --run $Output --output $ReportOutput
    }
}

if ($LASTEXITCODE -ne 0) { throw "Autonomous research exited with code $LASTEXITCODE" }
