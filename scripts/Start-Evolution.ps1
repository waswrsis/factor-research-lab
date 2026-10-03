param(
    [ValidateSet('Help','Run','Resume','Inspect','Report')][string]$Mode = 'Help',
    [string]$Output = '',
    [string]$ReportOutput = '',
    [string]$Config = 'config/discovery_v4.json',
    [string]$Policy = 'config/evolution_v4.json',
    [string]$Data = 'data/discovery/2025-hourly',
    [switch]$Agent,
    [ValidateSet('cpu','cuda')][string]$Device = 'cpu'
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Enter-Research.ps1') -Quiet
$evolutionPython = Join-Path $env:VIRTUAL_ENV 'Scripts/python.exe'
if ($Mode -ne 'Help' -and -not $Output) { throw '请指定研究目录。' }
switch ($Mode) {
    'Help' { & $evolutionPython -m factorlab.discovery evolve --help }
    { $_ -in 'Run','Resume' } {
        $evolutionArguments = @('-m','factorlab.discovery','--config',$Config,'--data',$Data,
            'evolve','--output',$Output,'--policy',$Policy,'--device',$Device)
        if ($Mode -eq 'Resume') { $evolutionArguments += '--resume' }
        if ($Agent) { $evolutionArguments += @('--search','external','--worker-backend','codex') }
        & $evolutionPython @evolutionArguments
    }
    'Inspect' { & $evolutionPython -m factorlab.discovery inspect-evolution $Output }
    'Report' {
        if (-not $ReportOutput) { throw '请指定新的报告输出目录。' }
        & $evolutionPython -m factorlab.discovery evolution-report --run $Output --output $ReportOutput
    }
}
if ($LASTEXITCODE -ne 0) { throw "持续研究程序退出：$LASTEXITCODE" }
