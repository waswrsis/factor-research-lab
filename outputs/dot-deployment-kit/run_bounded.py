#!/usr/bin/env python3
r"""Run one trusted Linux command in an owned process group with bounded cleanup.

Example (target Linux only; these are prospective lifecycle acceptance commands):

  export FACTOR_RESEARCH_ROOT=/workspace/FactorResearch
  PY="$FACTOR_RESEARCH_ROOT/.venv/bin/python"
  RUNNER="$FACTOR_RESEARCH_ROOT/outputs/dot-deployment-kit/run_bounded.py"
  # If installed elsewhere, set RUNNER to that deployed copy's absolute path.
  "$PY" "$RUNNER" --seconds 10 --label check-exit -- \
      "$PY" -c 'print("lifecycle check")'
  "$PY" "$RUNNER" --seconds 1 --label check-timeout -- \
      "$PY" -c 'import time; time.sleep(60)'
  # Expected: first exit 0; second exit 124. Use a fresh label for every check.

The work deadline includes launcher startup; cleanup can add up to 40 seconds:
main PID TERM / 30 seconds, then owned group TERM / 5 seconds, KILL / 5 seconds.
Children must inherit the owned process group. A daemon that calls setsid or
changes group can escape; use an available cgroup supervisor for that guarantee.
The flock excludes only other instances of this launcher at the same root.
Log output streams to disk without a memory buffer or an automatic disk quota.
RSS is sampled from /proc, summed over the current descendant tree plus remaining
owned-group members. Shared pages can be counted more than once and short peaks
can be missed. It is not a cgroup memory measurement. Receipt records command
arguments: never put authentication secrets in argv.

Exit: 124 deadline, 130 signal/stop file, otherwise child status (signals map to
128+signal). Launcher setup/internal errors use 2/1, exec failures use 127/126.
Existing labels are never overwritten. Setup refusal before label reservation
does not create a receipt. No restart/retry, package setup, or research is implicit.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

POLL_SECONDS = 0.2
MAIN_GRACE_SECONDS = 30.0
GROUP_TERM_SECONDS = 5.0
GROUP_KILL_SECONDS = 5.0
LABEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")


def positive_seconds(value: str) -> int:
    if not re.fullmatch(r"[0-9]+", value):
        raise argparse.ArgumentTypeError("--seconds must be a positive integer")
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("--seconds must be a positive integer")
    return result


def safe_label(value: str) -> str:
    if not LABEL_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError("--label must be 1..80 letters/digits/._-, beginning with a letter or digit")
    return value


def confined(root: Path, value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    candidate = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Path must remain within FACTOR_RESEARCH_ROOT: {value}")
    return candidate


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def proc_snapshot() -> dict[int, tuple[int, int, str, int | None]] | None:
    """pid -> (ppid, pgrp, state, RSS bytes); missing processes are normal races."""
    try:
        entries = list(Path("/proc").iterdir())
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        return None
    result = {}
    for path in entries:
        if not path.name.isdecimal():
            continue
        try:
            stat = (path / "stat").read_text(encoding="ascii")
            # comm (field 2) may itself contain spaces and parentheses.
            fields = stat[stat.rfind(")") + 2:].split()
            state, ppid, pgrp = fields[0], int(fields[1]), int(fields[2])
            try:
                rss = int((path / "statm").read_text(encoding="ascii").split()[1]) * page_size
            except (OSError, ValueError, IndexError):
                rss = None
            result[int(path.name)] = (ppid, pgrp, state, rss)
        except (OSError, ValueError, IndexError):
            continue
    return result


class Sampler:
    def __init__(self):
        self.pid: int | None = None
        self.pgid: int | None = None
        self.samples = 0
        self.peak_bytes = 0
        self.peak_members = 0
        self.unreadable_samples = 0
        self.unreadable_process_samples = 0

    def sample(self):
        if self.pid is None:
            return
        snapshot = proc_snapshot()
        if snapshot is None:
            self.unreadable_samples += 1
            return
        members = {self.pid}
        while True:
            added = {pid for pid, row in snapshot.items() if row[0] in members} - members
            if not added:
                break
            members.update(added)
        members.update(pid for pid, row in snapshot.items() if row[1] == self.pgid)
        rows = [snapshot[pid] for pid in members if pid in snapshot and snapshot[pid][2] not in {"Z", "X"}]
        self.samples += 1
        self.unreadable_process_samples += sum(row[3] is None for row in rows)
        self.peak_bytes = max(self.peak_bytes, sum(row[3] or 0 for row in rows))
        self.peak_members = max(self.peak_members, len(rows))

    def receipt(self):
        return {
            "peak_sampled_process_tree_rss_bytes": self.peak_bytes if self.samples else None,
            "peak_sampled_process_count": self.peak_members if self.samples else None,
            "sample_count": self.samples,
            "unreadable_snapshot_count": self.unreadable_samples,
            "unreadable_process_sample_count": self.unreadable_process_samples,
            "nominal_poll_seconds": POLL_SECONDS,
            "scope": "sum_RSS_of_live_descendants_and_remaining_owned_group_members",
            "limitations": "Not cgroup memory; shared pages may be double-counted; brief peaks and inaccessible RSS may be missed.",
        }


def group_live_pids(pgid: int) -> list[int] | None:
    snapshot = proc_snapshot()
    if snapshot is None:
        return None
    return sorted(pid for pid, row in snapshot.items() if row[1] == pgid and row[2] not in {"Z", "X"})


def group_alive(pgid: int) -> bool:
    snapshot = proc_snapshot()
    if snapshot is not None:
        rows = [row for row in snapshot.values() if row[1] == pgid]
        if rows:
            # Zombies are already dead and cannot be removed with another signal.
            return any(row[2] not in {"Z", "X"} for row in rows)
        # No visible members can also mean restricted /proc; do not assume empty.
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def send_group(pgid: int, sig: int) -> bool:
    try:
        os.killpg(pgid, sig)
        return True
    except ProcessLookupError:
        return False


def wait_main(process: subprocess.Popen, seconds: float, sampler: Sampler):
    deadline = time.monotonic() + seconds
    while process.poll() is None:
        sampler.sample()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(POLL_SECONDS, remaining))
    sampler.sample()


def wait_group(process: subprocess.Popen, pgid: int, seconds: float, sampler: Sampler):
    deadline = time.monotonic() + seconds
    while group_alive(pgid):
        process.poll()  # Reap the direct child promptly, even if only descendants remain.
        sampler.sample()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(POLL_SECONDS, remaining))
    process.poll()
    sampler.sample()


def cleanup(process: subprocess.Popen, pgid: int, sampler: Sampler) -> dict:
    """Only signal the group created by this launcher's start_new_session Popen."""
    if pgid != process.pid or pgid == os.getpgrp():
        raise RuntimeError("Refusing to signal an unowned process group")
    began = time.monotonic()
    stages = []
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        stages.append("main_SIGTERM")
        wait_main(process, MAIN_GRACE_SECONDS, sampler)
    # A main process exiting is not evidence that its owned group has disappeared.
    if group_alive(pgid):
        if send_group(pgid, signal.SIGTERM):
            stages.append("group_SIGTERM")
        wait_group(process, pgid, GROUP_TERM_SECONDS, sampler)
    if group_alive(pgid):
        if send_group(pgid, signal.SIGKILL):
            stages.append("group_SIGKILL")
        wait_group(process, pgid, GROUP_KILL_SECONDS, sampler)
    process.poll()
    survivors = group_live_pids(pgid)
    still_alive = group_alive(pgid)
    return {
        "stages": stages,
        "elapsed_seconds": time.monotonic() - began,
        "live_owned_group_pids_after_cleanup": survivors,
        "live_owned_group_remaining": still_alive,
        "limits_seconds": {"main_grace": MAIN_GRACE_SECONDS, "group_TERM": GROUP_TERM_SECONDS, "group_KILL": GROUP_KILL_SECONDS},
        "boundary": "owned_POSIX_process_group; setsid/setpgid_escape_not_guaranteed",
    }


def exclusive_json(path: Path, value: dict):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(temporary, flags, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic publish without replacing any existing label receipt.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def child_exit_code(returncode: int | None) -> int:
    if returncode is None:
        return 1
    return returncode if returncode >= 0 else 128 - returncode


def main(argv=None) -> int:
    began = time.monotonic()  # Includes root/lock/log setup and process startup.
    started_at = utc_now()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=positive_seconds, required=True)
    parser.add_argument("--label", type=safe_label, required=True)
    parser.add_argument("--stop-file", help="Optional confined path; presence requests a stop, including before spawn")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- COMMAND [ARG ...], passed without a shell")
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not command[0]:
        parser.error("Specify -- COMMAND [ARG ...]")
    if os.name != "posix" or not sys.platform.startswith("linux"):
        print("run_bounded.py requires Linux POSIX; --help can be inspected on other systems.", file=sys.stderr)
        return 2
    import fcntl  # Linux-only; importing this script for syntax/help does not require it.

    lock_fd = None
    log_handle = None
    process = None
    pgid = None
    receipt_path = None
    previous_handlers = {}
    stopped_signal = [None]
    sampler = Sampler()
    reason = "setup"
    exit_code = 1
    cleanup_record = None
    errors = []
    root = None
    log_path = None
    stop_file = None

    def signal_handler(signum, frame):
        if stopped_signal[0] is None:
            stopped_signal[0] = signum

    try:
        configured = os.environ.get("FACTOR_RESEARCH_ROOT")
        if not configured:
            raise ValueError("Export an absolute FACTOR_RESEARCH_ROOT before invoking the launcher")
        candidate_root = Path(configured).expanduser()
        if not candidate_root.is_absolute():
            raise ValueError("FACTOR_RESEARCH_ROOT must be an absolute Linux path")
        root = candidate_root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("FACTOR_RESEARCH_ROOT must be an existing directory")
        deployment = confined(root, "work/deployment")
        deployment.mkdir(parents=True, exist_ok=True)
        log_dir = confined(root, deployment / "logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        lock_path = confined(root, deployment / ".run_bounded.lock")
        lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise ValueError("Another run_bounded instance already holds the project lock") from exc
            raise
        log_path = log_dir / f"{args.label}.log"
        candidate_receipt = log_dir / f"{args.label}.json"
        if os.path.lexists(log_path) or os.path.lexists(candidate_receipt):
            raise ValueError("Label already exists; choose a fresh --label (existing files are preserved)")
        if args.stop_file:
            stop_file = confined(root, args.stop_file)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(log_path, flags, 0o600)
        log_handle = os.fdopen(fd, "wb", buffering=0)
        receipt_path = candidate_receipt  # Exclusive log reservation now owns this label.
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[sig] = signal.signal(sig, signal_handler)
        deadline = began + args.seconds
        if stopped_signal[0] is not None or (stop_file is not None and os.path.lexists(stop_file)):
            reason, exit_code = "stopped_before_spawn", 130
        elif time.monotonic() >= deadline:
            reason, exit_code = "deadline_before_spawn", 124
        else:
            reason = "starting"
            process = subprocess.Popen(
                command, cwd=root, stdin=subprocess.DEVNULL,
                stdout=log_handle, stderr=log_handle, shell=False,
                start_new_session=True, close_fds=True,
            )
            pgid = process.pid  # setsid() makes this child its process-group leader.
            sampler.pid, sampler.pgid = process.pid, pgid
            reason = "running"
            while True:
                sampler.sample()
                if stopped_signal[0] is not None or (stop_file is not None and os.path.lexists(stop_file)):
                    reason, exit_code = "stopped", 130
                    break
                if time.monotonic() >= deadline:
                    reason, exit_code = "deadline", 124
                    break
                if process.poll() is not None:
                    reason, exit_code = "child_exited", child_exit_code(process.returncode)
                    break
                time.sleep(min(POLL_SECONDS, max(0.0, deadline - time.monotonic())))
    except FileNotFoundError as exc:
        reason = "exec_not_found" if reason == "starting" else "setup_error"
        exit_code = 127 if reason == "exec_not_found" else 2
        errors.append(f"{type(exc).__name__}: {exc}")
    except PermissionError as exc:
        reason = "exec_not_permitted" if reason == "starting" else "launcher_error"
        exit_code = 126 if reason == "exec_not_permitted" else (2 if receipt_path is None else 1)
        errors.append(f"{type(exc).__name__}: {exc}")
    except BaseException as exc:  # noqa: BLE001 - supervisor must reach cleanup even on interruption
        reason = "launcher_error"
        exit_code = 2 if receipt_path is None else 1
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if process is not None and pgid is not None:
            try:
                cleanup_record = cleanup(process, pgid, sampler)
                if cleanup_record["live_owned_group_remaining"]:
                    errors.append("Owned group still has live members after the bounded KILL wait")
                    if exit_code not in {124, 130}:
                        exit_code = 1
            except BaseException as exc:  # noqa: BLE001 - cleanup faults must still trigger final group kill
                errors.append(f"CleanupError: {type(exc).__name__}: {exc}")
                # A fault during sampling/waiting must not skip the last owned-group kill.
                try:
                    if pgid == process.pid and pgid != os.getpgrp():
                        send_group(pgid, signal.SIGKILL)
                        process.wait(timeout=GROUP_KILL_SECONDS)
                except BaseException as final_exc:  # noqa: BLE001 - supervisor preserves receipt after final kill failure
                    errors.append(f"FinalKillError: {type(final_exc).__name__}: {final_exc}")
                if exit_code not in {124, 130}:
                    exit_code = 1
        if stopped_signal[0] is not None and exit_code != 124:
            reason, exit_code = "stopped", 130
        if log_handle is not None:
            try:
                log_handle.close()
            except OSError as exc:
                errors.append(f"LogCloseError: {exc}")
        if receipt_path is not None:
            receipt = {
                "format": "factor-research-bounded-launch-v1",
                "label": args.label,
                "root": str(root),
                "command": command,
                "started_at": started_at,
                "finished_at": utc_now(),
                "elapsed_seconds": time.monotonic() - began,
                "work_deadline_seconds": args.seconds,
                "shutdown_allowance_seconds": MAIN_GRACE_SECONDS + GROUP_TERM_SECONDS + GROUP_KILL_SECONDS,
                "reason": reason,
                "exit_code": exit_code,
                "child_pid": process.pid if process is not None else None,
                "owned_pgid": pgid,
                "child_returncode": process.returncode if process is not None else None,
                "stop_signal": signal.Signals(stopped_signal[0]).name if stopped_signal[0] is not None else None,
                "stop_file": str(stop_file) if stop_file is not None else None,
                "log": str(log_path),
                "rss_sampling": sampler.receipt(),
                "cleanup": cleanup_record,
                "errors": errors,
            }
            try:
                exclusive_json(receipt_path, receipt)
            except BaseException as exc:  # noqa: BLE001 - receipt failure must not bypass handler/lock release
                print(f"Receipt could not be published: {type(exc).__name__}: {exc}", file=sys.stderr)
                if exit_code not in {124, 130}:
                    exit_code = 1
        for sig, previous in previous_handlers.items():
            signal.signal(sig, previous)
        if lock_fd is not None:
            try:
                os.close(lock_fd)  # Releasing this fd releases flock, including on errors.
            except OSError:
                pass
    for error in errors:
        print(error, file=sys.stderr)
    if receipt_path is not None:
        print(f"Receipt: {receipt_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
