# Source from Bash after setting FACTOR_RESEARCH_ROOT to the target overlay directory.
# HOME and CODEX_HOME intentionally retain the existing operator's authentication.
if [[ -z "${FACTOR_RESEARCH_ROOT:-}" || "$FACTOR_RESEARCH_ROOT" != /* ]]; then
  printf '%s\n' 'Set an absolute FACTOR_RESEARCH_ROOT on the project overlay first.' >&2
  return 1
fi
if [[ ! -f "$FACTOR_RESEARCH_ROOT/pyproject.toml" ]]; then
  printf '%s\n' 'Project pyproject.toml is absent.' >&2
  return 1
fi
export FACTOR_RESEARCH_ROOT
export UV_PROJECT_ENVIRONMENT="$FACTOR_RESEARCH_ROOT/.venv"
export UV_CACHE_DIR="$FACTOR_RESEARCH_ROOT/.cache/uv"
export PIP_CACHE_DIR="$FACTOR_RESEARCH_ROOT/.cache/pip"
export TMPDIR="$FACTOR_RESEARCH_ROOT/.cache/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export TORCH_HOME="$FACTOR_RESEARCH_ROOT/.cache/torch"
export TORCH_EXTENSIONS_DIR="$FACTOR_RESEARCH_ROOT/.cache/torch-extensions"
export TORCHINDUCTOR_CACHE_DIR="$FACTOR_RESEARCH_ROOT/.cache/torch-inductor"
export NUMBA_CACHE_DIR="$FACTOR_RESEARCH_ROOT/.cache/numba"
export JOBLIB_TEMP_FOLDER="$FACTOR_RESEARCH_ROOT/.cache/joblib"
export MPLCONFIGDIR="$FACTOR_RESEARCH_ROOT/.cache/matplotlib"
export PYTHONPYCACHEPREFIX="$FACTOR_RESEARCH_ROOT/.cache/pycache"
export PYTHONNOUSERSITE=1 PYTHONUTF8=1
export UV_PYTHON_DOWNLOADS=never
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=false MLFLOW_DISABLE_TELEMETRY=true DO_NOT_TRACK=true
export MLFLOW_TRACKING_URI="sqlite:///$FACTOR_RESEARCH_ROOT/runs/mlflow.db"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 NUMBA_NUM_THREADS=2
export FACTOR_RESEARCH_TORCH_THREADS=2
export PATH="$FACTOR_RESEARCH_ROOT/.venv/bin:$FACTOR_RESEARCH_ROOT/.bootstrap/bin:$PATH"
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR" "$TORCH_HOME" \
  "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$NUMBA_CACHE_DIR" \
  "$JOBLIB_TEMP_FOLDER" "$MPLCONFIGDIR" "$PYTHONPYCACHEPREFIX" \
  "$FACTOR_RESEARCH_ROOT/work/deployment/logs" "$FACTOR_RESEARCH_ROOT/runs"
cd "$FACTOR_RESEARCH_ROOT" || return 1
