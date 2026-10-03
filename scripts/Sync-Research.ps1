param()

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Enter-Research.ps1') -Quiet
$researchUv = Join-Path $env:FACTOR_RESEARCH_ROOT '.tools\uv\uv.exe'
if (-not (Test-Path -LiteralPath $researchUv)) {
    throw 'Project-local uv is missing. Restore the verified tool recorded in work\environment\uv-provenance.json.'
}
$researchPythonVersion = (Get-Content -LiteralPath (Join-Path $env:FACTOR_RESEARCH_ROOT '.python-version') -Raw).Trim()
& $researchUv python install $researchPythonVersion --no-bin --no-registry
if ($LASTEXITCODE -ne 0) { throw 'Managed Python setup failed.' }
& $researchUv sync --locked
if ($LASTEXITCODE -ne 0) { throw 'Locked dependency synchronization failed.' }
& $researchUv pip check --python (Join-Path $env:FACTOR_RESEARCH_ROOT '.venv\Scripts\python.exe')
if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed.' }
Write-Output 'The project environment matches uv.lock.'
