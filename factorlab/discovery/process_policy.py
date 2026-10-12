"""Torch-free restricted proposer process policy; not an operating-system sandbox.

The disable list is a versioned compatibility contract with agent_worker. CLI
rejection of a flag must fail closed, never retry with fewer restrictions.
On POSIX, callers must launch with isolated_process_kwargs() for tree teardown.
"""
from __future__ import annotations

import math
import os
import signal
import subprocess
from pathlib import Path

DISABLED_FEATURES = ('shell_tool', 'unified_exec', 'shell_snapshot', 'apps', 'plugins',
                     'remote_plugin', 'multi_agent', 'multi_agent_v2', 'hooks', 'memories',
                     'browser_use', 'browser_use_external', 'browser_use_full_cdp_access',
                     'computer_use', 'image_generation', 'in_app_browser', 'in_app_chat',
                     'in_app_local_automation', 'view_image', 'workspace_dependencies',
                     'worktrees', 'skill_search', 'skill_mcp_dependency_install',
                     'tool_suggest', 'code_mode', 'code_mode_host', 'goals', 'sleep_tool')

RESTRICTED_OVERRIDES = ('web_search="disabled"', 'tools.view_image=false', 'mcp_servers={}',
                        'project_doc_max_bytes=0', 'features.skip_host_skill_discovery=true',
                        'shell_environment_policy.inherit="none"')


def proposer_restriction_args():
    """Fresh argv entries, to append without interpolation or permissive fallback."""
    return ([part for feature in DISABLED_FEATURES for part in ('--disable', feature)]
            + [part for value in RESTRICTED_OVERRIDES for part in ('-c', value)])


def restricted_environment(environment=None):
    """Match old worker secret filtering; source environment remains unchanged."""
    environment = os.environ if environment is None else environment
    return {key: value for key, value in environment.items()
            if key.upper() not in {'OPENAI_API_KEY', 'CODEX_API_KEY'}
            and not any(word in key.upper() for word in ('TOKEN', 'SECRET', 'PASSWORD'))}


def isolated_process_kwargs():
    """Arguments for Popen: POSIX dedicated session, Windows hidden process group."""
    if os.name == 'nt':
        return {'creationflags': (getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
                                  | getattr(subprocess, 'CREATE_NO_WINDOW', 0))}
    return {'start_new_session': True}


def terminate_process_tree(process, *, timeout=5.):
    """Bounded teardown of an owned, still-running proposer and its process group.

    Windows uses the native PID tree command without a shell. POSIX kills only
    a dedicated group led by this exact Popen PID, never the caller's group.
    Detached/escaped descendants and children of an already-exited wrapper are
    not claimed verified. This helper is lifecycle control, not containment.
    """
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 10:
        raise ValueError('Process teardown timeout must be in 0..10 seconds')
    if type(process.pid) is not int or process.pid <= 0:
        raise ValueError('Teardown requires a valid owned process PID')
    if process.poll() is not None:
        return {'status': 'already_exited', 'returncode': process.returncode,
                'tree_command_completed': False, 'escaped_descendants_verified': False}
    if os.name == 'nt':
        taskkill = Path(os.environ.get('SystemRoot', 'C:/Windows')) / 'System32' / 'taskkill.exe'
        try:
            outcome = subprocess.run([str(taskkill), '/PID', str(process.pid), '/T', '/F'],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
                                     timeout=timeout,
                                     creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except (OSError, subprocess.TimeoutExpired):
            if process.poll() is None:
                process.kill()
            process.wait(timeout=timeout)
            raise
        tree_done = outcome.returncode == 0
    else:
        try:
            group = os.getpgid(process.pid)
        except ProcessLookupError:
            process.wait(timeout=timeout)
            return {'status': 'exited_during_teardown', 'returncode': process.returncode,
                    'tree_command_completed': False, 'escaped_descendants_verified': False}
        if group != process.pid or group == os.getpgrp():
            # Refuse broad group signals. End the owned root, then fail loudly
            # because no descendant teardown can be verified for this launch.
            process.kill()
            process.wait(timeout=timeout)
            raise RuntimeError('POSIX proposer was not launched in a dedicated session')
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        tree_done = True
    if process.poll() is None:
        process.kill()
    process.wait(timeout=timeout)
    if not tree_done:
        raise RuntimeError('Native process-tree termination failed; owned root was stopped')
    return {'status': 'terminated', 'returncode': process.returncode,
            'tree_command_completed': tree_done, 'escaped_descendants_verified': False}
