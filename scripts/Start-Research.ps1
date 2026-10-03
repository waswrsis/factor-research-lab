param(
    [ValidateSet('Check', 'Notebook', 'Shell')]
    [string]$Mode = 'Check'
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Enter-Research.ps1') -Quiet
$researchPython = Join-Path $env:FACTOR_RESEARCH_ROOT '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $researchPython)) {
    throw 'The project Python environment is missing. Run scripts\Sync-Research.ps1 first.'
}

switch ($Mode) {
    'Check' {
        & $researchPython (Join-Path $PSScriptRoot 'verify_environment.py') --require-cuda
        if ($LASTEXITCODE -ne 0) { throw 'Core environment checks failed.' }
        & $researchPython (Join-Path $PSScriptRoot 'verify_qlib.py')
        if ($LASTEXITCODE -ne 0) { throw 'Qlib component checks failed.' }
    }
    'Notebook' {
        # Loopback only; retain Jupyter's normal authentication defaults.
        & $researchPython -m jupyterlab '--ServerApp.ip=127.0.0.1' '--ServerApp.root_dir=D:\FactorResearch' --no-browser
        if ($LASTEXITCODE -ne 0) { throw 'JupyterLab exited with an error.' }
    }
    'Shell' {
        Write-Output 'Workspace and project-local caches are active in this PowerShell process.'
        Write-Output "Python: $researchPython"
        Write-Output "Working directory: $((Get-Location).Path)"
    }
}
