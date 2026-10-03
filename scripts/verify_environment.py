#!/usr/bin/env python
"""Run offline smoke checks for the D-drive FactorResearch environment.

This script does not download data, inspect credentials, train a model, or
backtest a strategy. Run it through the project launcher so import caches and
temporary files remain within the project. It never changes environment vars.

Exit codes: 0 = passed (possibly with warnings), 1 = failed checks,
            2 = invalid invocation or inability to save the report.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import io
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("D:/FactorResearch")
DEFAULT_REPORT = PROJECT_ROOT / "work/environment/verification.json"
PACKAGES = (
    ("numpy", "numpy", True),
    ("pandas", "pandas", True),
    ("scipy", "scipy", True),
    ("sklearn", "scikit-learn", True),
    ("torch", "torch", True),
    ("optuna", "optuna", True),
    ("duckdb", "duckdb", True),
    ("pyarrow", "pyarrow", True),
    ("qlib", "pyqlib", False),
    ("numba", "numba", False),
    ("vectorbt", "vectorbt", False),
    ("backtesting", "backtesting", False),
)
# Only non-secret, path-valued variables used for project-local scratch space.
CACHE_VARIABLES = (
    "PIP_CACHE_DIR",
    "UV_CACHE_DIR",
    "XDG_CACHE_HOME",
    "HF_HOME",
    "TORCH_HOME",
    "TORCH_EXTENSIONS_DIR",
    "NUMBA_CACHE_DIR",
    "MPLCONFIGDIR",
    "JOBLIB_TEMP_FOLDER",
    "PYTHONPYCACHEPREFIX",
    "TMP",
    "TEMP",
)


def is_inside(path: Path, root: Path) -> bool:
    """Resolve paths before containment checks, including Windows junctions."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError, RuntimeError):
        return False


def error_detail(exc: Exception) -> dict[str, str]:
    return {"error_type": type(exc).__name__, "error": str(exc)}


def add_check(
    report: dict[str, Any],
    name: str,
    status: str,
    **details: Any,
) -> None:
    report["checks"].append({"name": name, "status": status, **details})


def check_workspace(report: dict[str, Any], report_path: Path) -> bool:
    paths = {
        "cwd": Path.cwd(),
        "python_executable": Path(sys.executable),
        "script": Path(__file__),
        "report": report_path,
    }
    safe = True
    for name, path in paths.items():
        contained = is_inside(path, PROJECT_ROOT)
        safe = safe and contained
        add_check(
            report,
            "path_" + name,
            "passed" if contained else "failed",
            path=str(path.resolve()),
            required_root=str(PROJECT_ROOT.resolve()),
        )

    cache_paths: dict[str, Any] = {}
    outside = []
    missing = []
    for name in CACHE_VARIABLES:
        value = os.environ.get(name)
        if not value:
            missing.append(name)
            continue
        path = Path(value)
        contained = path.is_absolute() and is_inside(path, PROJECT_ROOT)
        cache_paths[name] = {"path": value, "inside_project": contained}
        if not contained:
            outside.append(name)
    add_check(
        report,
        "launcher_cache_paths",
        "failed" if outside else ("warning" if missing else "passed"),
        configured=cache_paths,
        not_configured=missing,
        outside_project=outside,
        note=(
            "The launcher must redirect package caches and temporary files. "
            "Missing settings are not proof that a package will write outside "
            "the project; configured paths outside the project are rejected."
        ),
    )
    safe = safe and not outside
    # Reject external cache paths before importing third-party libraries.
    return safe


def check_packages(report: dict[str, Any]) -> dict[str, Any]:
    modules: dict[str, Any] = {}
    for module_name, distribution, required in PACKAGES:
        try:
            module = importlib.import_module(module_name)
            modules[module_name] = module
            try:
                version = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                version = str(getattr(module, "__version__", "unknown"))
            add_check(
                report,
                "package_" + module_name,
                "passed",
                required=required,
                distribution=distribution,
                version=version,
                module_path=getattr(module, "__file__", None),
            )
        except Exception as exc:
            absent = (
                isinstance(exc, ModuleNotFoundError)
                and getattr(exc, "name", None) == module_name
            )
            status = "failed" if required else ("skipped" if absent else "warning")
            add_check(
                report,
                "package_" + module_name,
                status,
                required=required,
                distribution=distribution,
                **error_detail(exc),
            )
    return modules


def operator_smoke(torch: Any, device: str) -> dict[str, Any]:
    """Check one tiny matrix product and its gradient, not model training."""
    generator = torch.Generator(device=device)
    generator.manual_seed(1729)
    x = torch.randn(
        (8, 8), generator=generator, device=device, requires_grad=True
    )
    weights = torch.randn((8, 4), generator=generator, device=device)
    result = x @ weights
    objective = result.square().mean()
    objective.backward()
    expected_gradient = (2.0 / result.numel()) * (result.detach() @ weights.T)
    if x.grad is None or not bool(torch.isfinite(x.grad).all().item()):
        raise RuntimeError("The matrix-product gradient is missing or non-finite.")
    if not bool(torch.allclose(x.grad, expected_gradient, rtol=1e-4, atol=1e-6)):
        raise RuntimeError("Autograd disagrees with the analytic gradient.")
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    return {
        "device": device,
        "input_shape": [8, 8],
        "weight_shape": [8, 4],
        "output_shape": list(result.shape),
        "objective_value": float(objective.detach().cpu().item()),
        "gradient_matches_analytic": True,
        "scope": "single matrix product and backward pass; no model or optimizer",
    }


def check_torch(report: dict[str, Any], modules: dict[str, Any], require_cuda: bool) -> None:
    torch = modules.get("torch")
    if torch is None:
        add_check(report, "torch_operators", "skipped", reason="torch import failed")
        return
    try:
        add_check(report, "torch_cpu_operators", "passed", **operator_smoke(torch, "cpu"))
    except Exception as exc:
        add_check(report, "torch_cpu_operators", "failed", **error_detail(exc))
    try:
        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count())
        details: dict[str, Any] = {
            "available": available,
            "device_count": count,
            "torch_cuda_runtime": torch.version.cuda,
            "required": require_cuda,
            "devices": [],
        }
        if available:
            for index in range(count):
                props = torch.cuda.get_device_properties(index)
                free_bytes, total_bytes = torch.cuda.mem_get_info(index)
                details["devices"].append(
                    {
                        "index": index,
                        "name": props.name,
                        "compute_capability": [props.major, props.minor],
                        "total_memory_bytes": int(props.total_memory),
                        "runtime_total_memory_bytes": int(total_bytes),
                        "currently_free_memory_bytes": int(free_bytes),
                    }
                )
            add_check(report, "cuda_detection", "passed", **details)
            add_check(
                report, "torch_cuda_operators", "passed",
                **operator_smoke(torch, "cuda:0")
            )
        else:
            add_check(
                report, "cuda_detection",
                "failed" if require_cuda else "warning", **details
            )
            add_check(
                report, "torch_cuda_operators", "skipped",
                reason="CUDA is unavailable; CPU checks still apply."
            )
    except Exception as exc:
        # A detected GPU that cannot execute an operator is a real failure,
        # even when CPU-only installations were allowed by the invocation.
        add_check(report, "cuda_runtime_or_operators", "failed", **error_detail(exc))


def check_roundtrip(report: dict[str, Any], modules: dict[str, Any]) -> None:
    required = ("numpy", "pandas", "duckdb", "pyarrow")
    missing = [name for name in required if name not in modules]
    if missing:
        add_check(report, "parquet_duckdb_roundtrip", "skipped", missing_packages=missing)
        return
    pd = modules["pandas"]
    np = modules["numpy"]
    connection = None
    try:
        frame = pd.DataFrame(
            {
                "row_id": np.arange(4, dtype=np.int64),
                "value": np.array([0.125, -0.5, 1.25, 2.0], dtype=np.float64),
                "label": ["a", "b", "c", "d"],
            }
        )
        buffer = io.BytesIO()
        frame.to_parquet(buffer, engine="pyarrow", index=False)
        parquet_size = buffer.tell()
        buffer.seek(0)
        decoded = pd.read_parquet(buffer, engine="pyarrow")
        pd.testing.assert_frame_equal(frame, decoded)
        connection = modules["duckdb"].connect(database=":memory:")
        connection.register("sample", decoded)
        queried = connection.execute(
            "SELECT row_id, value, label FROM sample ORDER BY row_id"
        ).fetchdf()
        pd.testing.assert_frame_equal(frame, queried)
        total = connection.execute("SELECT SUM(value) FROM sample").fetchone()[0]
        if not math.isclose(float(total), float(frame["value"].sum())):
            raise RuntimeError("DuckDB aggregate did not match the input values.")
        add_check(
            report, "parquet_duckdb_roundtrip", "passed", rows=len(frame),
            parquet_bytes=parquet_size, storage="in-memory only"
        )
    except Exception as exc:
        add_check(report, "parquet_duckdb_roundtrip", "failed", **error_detail(exc))
    finally:
        if connection is not None:
            connection.close()


def check_optuna(report: dict[str, Any], modules: dict[str, Any]) -> None:
    optuna = modules.get("optuna")
    if optuna is None:
        add_check(report, "optuna_scalar_smoke", "skipped", reason="optuna import failed")
        return
    try:
        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.RandomSampler(seed=1729)
        )

        def objective(trial: Any) -> float:
            value = trial.suggest_float("x", -1.0, 1.0)
            return float((value - 0.25) ** 2)

        study.optimize(objective, n_trials=3, n_jobs=1, show_progress_bar=False)
        if len(study.trials) != 3 or not math.isfinite(float(study.best_value)):
            raise RuntimeError("The in-memory scalar study did not complete.")
        add_check(
            report, "optuna_scalar_smoke", "passed", trials=len(study.trials),
            best_value=float(study.best_value), storage="in-memory only",
            scope="three evaluations of (x - 0.25)^2; no financial objective"
        )
    except Exception as exc:
        add_check(report, "optuna_scalar_smoke", "failed", **error_detail(exc))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--require-cuda", action="store_true",
        help="Return failure if CUDA is unavailable."
    )
    args = parser.parse_args()
    report_path = args.report.resolve()
    if not is_inside(report_path, PROJECT_ROOT):
        parser.error("The JSON report must stay inside D:/FactorResearch.")

    report: dict[str, Any] = {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT.resolve()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "scope": {
            "network_access": False,
            "real_data_download": False,
            "model_training": False,
            "strategy_backtest": False,
            "credential_inspection": False,
            "environment_variables_modified": False,
        },
        "checks": [],
    }
    if check_workspace(report, report_path):
        modules = check_packages(report)
        check_torch(report, modules, args.require_cuda)
        check_roundtrip(report, modules)
        check_optuna(report, modules)
    else:
        add_check(
            report, "dependency_smoke_checks", "skipped",
            reason="Workspace or configured cache path checks failed before imports."
        )

    counts = {status: 0 for status in ("passed", "failed", "warning", "skipped")}
    for check in report["checks"]:
        counts[check["status"]] += 1
    report["summary"] = counts
    report["status"] = (
        "failed" if counts["failed"] else ("passed_with_warnings" if counts["warning"] else "passed")
    )
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    except OSError as exc:
        print(json.dumps({"status": "report_write_failed", **error_detail(exc)}))
        return 2
    print(json.dumps({"status": report["status"], "summary": counts, "report": str(report_path)}))
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
