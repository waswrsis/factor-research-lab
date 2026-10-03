param(
    [ValidateSet('Help','Prepare','Research','Confirm','Replay','Inspect')]
    [string]$Mode = 'Help',
    [string]$Output = '',
    [string]$Config = 'config/discovery.json',
    [string]$Data = 'data/discovery/2025-hourly',
    [ValidateSet('cpu','cuda')][string]$Device = 'cpu'
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Enter-Research.ps1') -Quiet
$discoveryPython = Join-Path $env:VIRTUAL_ENV 'Scripts/python.exe'
switch ($Mode) {
    'Help' { & $discoveryPython -m factorlab.discovery --help }
    'Prepare' { & $discoveryPython -m factorlab.discovery --config $Config --data $Data prepare }
    'Research' {
        if (-not $Output) { throw '请指定新的实验输出目录。' }
        & $discoveryPython -m factorlab.discovery --config $Config --data $Data run --output $Output --device $Device
    }
    'Confirm' {
        if (-not $Output) { throw '请指定新的实验输出目录。确认资料只能独立使用一次。' }
        & $discoveryPython -m factorlab.discovery --config $Config --data $Data run --output $Output --device $Device --confirm
    }
    'Replay' {
        if (-not $Output) { throw '请指定新的回放输出目录。' }
        & $discoveryPython -m factorlab.discovery --config $Config --data $Data replay --output $Output --device $Device --update refit
    }
    'Inspect' { & $discoveryPython -m factorlab.discovery inspect $Output }
}
if ($LASTEXITCODE -ne 0) { throw "研究程序失败，退出码 $LASTEXITCODE" }
