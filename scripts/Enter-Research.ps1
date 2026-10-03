param([switch]$Quiet)

$ErrorActionPreference = 'Stop'
$researchRoot = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot)).TrimEnd('\')
if ($researchRoot -ne 'D:\FactorResearch') {
    throw 'This project is configured for D:\FactorResearch. Review paths before relocating it.'
}
Set-Location -LiteralPath $researchRoot

# Process-local settings only: do not modify the user profile or machine PATH.
$researchPaths = @{
    FACTOR_RESEARCH_ROOT = $researchRoot
    UV_PROJECT_ENVIRONMENT = (Join-Path $researchRoot '.venv')
    UV_CACHE_DIR = (Join-Path $researchRoot '.cache\uv')
    UV_PYTHON_INSTALL_DIR = (Join-Path $researchRoot '.tools\python')
    UV_PYTHON_BIN_DIR = (Join-Path $researchRoot '.tools\python-bin')
    UV_TOOL_DIR = (Join-Path $researchRoot '.tools\uv-tools')
    UV_TOOL_BIN_DIR = (Join-Path $researchRoot '.tools\bin')
    PIP_CACHE_DIR = (Join-Path $researchRoot '.cache\pip')
    TORCH_HOME = (Join-Path $researchRoot '.cache\torch')
    TORCH_EXTENSIONS_DIR = (Join-Path $researchRoot '.cache\torch-extensions')
    TORCHINDUCTOR_CACHE_DIR = (Join-Path $researchRoot '.cache\torch-inductor')
    NUMBA_CACHE_DIR = (Join-Path $researchRoot '.cache\numba')
    JOBLIB_TEMP_FOLDER = (Join-Path $researchRoot '.cache\joblib')
    PYTHONPYCACHEPREFIX = (Join-Path $researchRoot '.cache\pycache')
    RUFF_CACHE_DIR = (Join-Path $researchRoot '.cache\ruff')
    MPLCONFIGDIR = (Join-Path $researchRoot '.cache\matplotlib')
    HF_HOME = (Join-Path $researchRoot '.cache\huggingface')
    HF_HUB_CACHE = (Join-Path $researchRoot '.cache\huggingface\hub')
    XDG_CACHE_HOME = (Join-Path $researchRoot '.cache\xdg\cache')
    XDG_CONFIG_HOME = (Join-Path $researchRoot '.cache\xdg\config')
    XDG_DATA_HOME = (Join-Path $researchRoot '.cache\xdg\data')
    TEMP = (Join-Path $researchRoot '.cache\tmp')
    TMP = (Join-Path $researchRoot '.cache\tmp')
    IPYTHONDIR = (Join-Path $researchRoot '.cache\ipython')
    JUPYTER_CONFIG_DIR = (Join-Path $researchRoot '.cache\jupyter\config')
    JUPYTER_DATA_DIR = (Join-Path $researchRoot '.cache\jupyter\data')
    JUPYTER_RUNTIME_DIR = (Join-Path $researchRoot '.cache\jupyter\runtime')
}
foreach ($researchEntry in $researchPaths.GetEnumerator()) {
    if ($researchEntry.Key -ne 'UV_PROJECT_ENVIRONMENT') {
        New-Item -ItemType Directory -Path $researchEntry.Value -Force | Out-Null
    }
    [System.Environment]::SetEnvironmentVariable($researchEntry.Key, $researchEntry.Value, 'Process')
}
$env:UV_MANAGED_PYTHON = 'true'
$env:UV_NO_MODIFY_PATH = '1'
$env:UV_LINK_MODE = 'hardlink'
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONUTF8 = '1'
$env:MLFLOW_TRACKING_URI = 'sqlite:///D:/FactorResearch/runs/mlflow.db'
$env:MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING = 'false'
$env:MLFLOW_DISABLE_TELEMETRY = 'true'
$env:DO_NOT_TRACK = 'true'
$env:VIRTUAL_ENV = Join-Path $researchRoot '.venv'
$researchBin = Join-Path $researchRoot '.venv\Scripts'
$researchUvBin = Join-Path $researchRoot '.tools\uv'
$env:PATH = "$researchBin;$researchUvBin;$env:PATH"
if (-not $Quiet) {
    Write-Output "Research workspace: $researchRoot"
    Write-Output 'Project Python, dependencies, caches, datasets, and experiment output stay on drive D.'
}
