#!/usr/bin/env python
"""Offline Qlib operator and local recorder checks, without data or training.

Run from D:/FactorResearch with its project Python. The only artifacts are a
tiny local Qlib run and work/environment/qlib-verification.json. Ancillary
source/environment capture is deliberately excluded from this controlled test.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import socket
import sys
import tempfile
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlparse
from urllib.request import url2pathname


ROOT = Path("D:/FactorResearch")
WORK = ROOT / "work/environment"
PROVIDER = ROOT / "data/qlib"
TRACKING = WORK / "qlib-mlruns"
REPORT = WORK / "qlib-verification.json"


def within(path: Path, root: Path = ROOT) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError, RuntimeError):
        return False


def add(report: dict, name: str, **details) -> None:
    report["checks"].append({"name": name, "status": "passed", **details})


def verify_rolling(report: dict) -> None:
    import numpy as np
    import pandas as pd
    from qlib.data._libs.rolling import rolling_mean

    values = np.array([1.0, 3.0, np.nan, 7.0, -2.0, 0.0, 5.0], dtype=np.float64)
    expected = pd.Series(values).rolling(window=3, min_periods=1).mean().to_numpy()
    actual = rolling_mean(values, 3)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
    add(report, "qlib_cython_rolling_mean", window=3, samples=len(values),
        actual=actual.tolist(), expected=expected.tolist(),
        max_absolute_error=float(np.max(np.abs(actual - expected))))


def verify_recorder(report: dict) -> None:
    import mlflow
    import qlib
    from qlib.config import C
    from qlib.workflow import R
    from qlib.workflow.recorder import MLflowRecorder

    tracking_uri = TRACKING.resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    qlib.init(
        provider_uri=str(PROVIDER.resolve()),
        region="us",
        expression_cache=None,
        dataset_cache=None,
        calendar_cache=None,
        local_cache_path=str((WORK / "qlib-cache").resolve()),
        provider="LocalProvider",
        calendar_provider="LocalCalendarProvider",
        instrument_provider="LocalInstrumentProvider",
        feature_provider="LocalFeatureProvider",
        pit_provider="LocalPITProvider",
        expression_provider="LocalExpressionProvider",
        dataset_provider="LocalDatasetProvider",
        kernels=1,
        joblib_backend="threading",
        auto_mount=False,
        mount_path=None,
        logging_level=logging.WARNING,
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {"uri": tracking_uri, "default_exp_name": "environment-check"},
        },
    )
    if C.expression_cache is not None or C.dataset_cache is not None:
        raise RuntimeError("Qlib disk caches were unexpectedly enabled")
    if C.get_kernels("day") != 1:
        raise RuntimeError("Qlib worker count is not one")
    for configured_path in C.provider_uri.values():
        if Path(configured_path).resolve() != PROVIDER.resolve():
            raise RuntimeError("Qlib provider escaped the explicit local data directory")
    add(report, "qlib_local_configuration", provider_uri=str(PROVIDER),
        tracking_uri=tracking_uri, expression_cache=None, dataset_cache=None,
        calendar_cache=None, n_jobs=1, qlib_worker_setting="kernels=1",
        note="Region initializes defaults only; no calendar or market data is read.")

    expected = {"kind": "environment-check", "values": [1, 2, 3],
                "trained": False, "market_data_used": False}
    # This hook records unrelated workspace contents; core recorder APIs remain intact.
    with patch.object(MLflowRecorder, "_log_uncommitted_code", lambda self: None):
        with R.start(experiment_name="environment-check", recorder_name="local-artifact-smoke",
                     uri=tracking_uri):
            recorder = R.get_recorder()
            artifact_uri = recorder.get_artifact_uri()
            parsed = urlparse(artifact_uri)
            if parsed.scheme != "file" or parsed.netloc:
                raise RuntimeError("Recorder artifact storage is not a local file URI")
            artifact_dir = Path(url2pathname(parsed.path)).resolve()
            if not within(artifact_dir, TRACKING):
                raise RuntimeError("Recorder artifact storage escaped the project tracking directory")
            R.log_metrics(environment_check=1.0)
            R.save_objects(**{"environment-smoke.pkl": expected})
            restored = recorder.load_object("environment-smoke.pkl")
            if restored != expected:
                raise AssertionError("Recorder object round-trip mismatch")

    # Leaving R.start flushes Qlib's asynchronous metric writer.
    metrics = recorder.list_metrics()
    if metrics.get("environment_check") != 1.0:
        raise AssertionError("Recorded metric did not survive the run flush")
    artifact_file = artifact_dir / "environment-smoke.pkl"
    if not artifact_file.is_file():
        raise AssertionError("Expected local artifact file is missing")
    add(report, "qlib_mlflow_local_round_trip", run_id=recorder.id,
        experiment_id=recorder.experiment_id, artifact_uri=artifact_uri,
        artifact_bytes=artifact_file.stat().st_size, metrics=metrics,
        dictionary_equal=True, run_status=recorder.status)


def main() -> int:
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Controlled offline Qlib environment checks; no production workflow validation",
        "project_root": str(ROOT),
        "checks": [],
        "restrictions": {
            "market_data_downloaded": False,
            "model_training": False,
            "strategy_backtest": False,
            "source_snapshot_hook_disabled": True,
            "environment_capture": "Only an explicit safe environment view is visible; no _QLIB_ variables",
            "network_connections": "Blocked during third-party imports and verification",
            "credential_inspection": False,
        },
    }
    try:
        if Path.cwd().resolve() != ROOT.resolve():
            raise RuntimeError("Run this script with D:/FactorResearch as the working directory")
        for path in (Path(__file__), Path(sys.executable), WORK, PROVIDER, TRACKING, REPORT):
            if not within(path):
                raise RuntimeError(f"Path is outside the project: {path}")
        for path in (WORK, PROVIDER, TRACKING, WORK / "tmp", WORK / "qlib-cache",
                     WORK / "cache", WORK / "config", WORK / "matplotlib"):
            if not within(path):
                raise RuntimeError(f"Output directory is outside the project: {path}")
            path.mkdir(parents=True, exist_ok=True)

        # Preserve non-secret Windows loader/home paths without changing the user's
        # home directory. Qlib resolves some unused defaults during import, even
        # when all active provider, cache, temporary, and tracking paths are on D.
        # No credentials or unlisted environment variables are enumerated.
        safe_env = {name: os.environ[name] for name in
                    ("SystemRoot", "WINDIR", "PATH", "COMSPEC", "PATHEXT", "PROCESSOR_ARCHITECTURE",
                     "USERPROFILE", "HOMEDRIVE", "HOMEPATH")
                    if name in os.environ}
        safe_env.update({
            "USERNAME": "factor-environment-check",
            "TMP": str(WORK / "tmp"), "TEMP": str(WORK / "tmp"),
            "XDG_CACHE_HOME": str(WORK / "cache"),
            "XDG_CONFIG_HOME": str(WORK / "config"),
            "MPLCONFIGDIR": str(WORK / "matplotlib"),
            "MLFLOW_TRACKING_URI": TRACKING.resolve().as_uri(),
            "MLFLOW_REGISTRY_URI": TRACKING.resolve().as_uri(),
            "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING": "false",
            "MLFLOW_ENABLE_ASYNC_LOGGING": "false",
            "MLFLOW_ENABLE_ASYNC_TRACE_LOGGING": "false",
            "MLFLOW_ENABLE_OTLP_EXPORTER": "false",
            "MLFLOW_DISABLE_TELEMETRY": "true",
            "DO_NOT_TRACK": "true",
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        })
        blocked_attempts = []

        def deny_network(*args, **kwargs):
            # Record code locations only: never serialize destinations, call
            # arguments, local variables, source text, or environment values.
            stack = []
            frame = sys._getframe(1)
            try:
                while frame is not None and len(stack) < 20:
                    stack.append({"file": frame.f_code.co_filename,
                                  "function": frame.f_code.co_name,
                                  "line": frame.f_lineno})
                    frame = frame.f_back
            finally:
                del frame
            blocked_attempts.append({"stack": stack})
            raise RuntimeError("Network access is disabled for this offline environment check")

        with ExitStack() as stack:
            stack.enter_context(patch("os.environ", safe_env))
            stack.enter_context(patch.object(tempfile, "tempdir", str(WORK / "tmp")))
            stack.enter_context(patch.object(socket.socket, "connect", deny_network))
            stack.enter_context(patch.object(socket.socket, "connect_ex", deny_network))
            stack.enter_context(patch.object(socket, "create_connection", deny_network))
            stack.enter_context(patch.object(socket, "getaddrinfo", deny_network))
            report["versions"] = {name: importlib.metadata.version(name)
                                  for name in ("pyqlib", "mlflow", "numpy", "pandas")}
            for name, check in (("qlib_cython_rolling_mean", verify_rolling),
                                ("qlib_mlflow_local_round_trip", verify_recorder)):
                try:
                    check(report)
                except Exception as exc:
                    report["checks"].append({"name": name, "status": "failed",
                                             "error_type": type(exc).__name__, "error": str(exc)})
        report["blocked_network_attempts"] = len(blocked_attempts)
        report["blocked_network_call_stacks"] = blocked_attempts
        if blocked_attempts:
            report["checks"].append({"name": "no_network_attempts", "status": "failed",
                                     "attempt_count": len(blocked_attempts)})
        else:
            add(report, "no_network_attempts", attempt_count=0)
    except Exception as exc:
        report["checks"].append({"name": "setup", "status": "failed",
                                 "error_type": type(exc).__name__, "error": str(exc)})

    report["status"] = "failed" if any(c["status"] == "failed" for c in report["checks"]) else "passed"
    if not within(REPORT):
        print("Refusing to write report outside D:/FactorResearch", file=sys.stderr)
        return 2
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(REPORT)}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
