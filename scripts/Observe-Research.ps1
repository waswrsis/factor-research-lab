param([switch]$Settle)
$ErrorActionPreference = 'Stop'
. 'D:\FactorResearch\scripts\Enter-Research.ps1' -Quiet
$observerConfig = Get-Content -LiteralPath 'D:\FactorResearch\config\shadow_observer.json' -Raw | ConvertFrom-Json
$observerArgs = @('scripts/observe_shadow.py', '--bundle', $observerConfig.bundle,
    '--expected-hash', $observerConfig.expected_hash, '--ledger', $observerConfig.ledger,
    '--protocol', $observerConfig.protocol, '--once')
if ($Settle) { $observerArgs += '--settle' }
& 'D:\FactorResearch\.venv\Scripts\python.exe' @observerArgs
if ($LASTEXITCODE -ne 0) { throw 'Shadow observation failed. Existing records have been retained.' }
