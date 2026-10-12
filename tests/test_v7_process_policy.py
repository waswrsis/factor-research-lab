import ast
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from factorlab.discovery import process_policy as policy


def test_disable_list_matches_old_worker_without_importing_it():
    tree = ast.parse(Path('factorlab/discovery/agent_worker.py').read_text('utf-8'))
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(name, ast.Name) and name.id == 'DISABLED_FEATURES' for name in node.targets)]
    assert policy.DISABLED_FEATURES == ast.literal_eval(assignments[0].value)
    args = policy.proposer_restriction_args()
    assert args.count('--disable') == len(policy.DISABLED_FEATURES)
    assert 'mcp_servers={}' in args and 'web_search="disabled"' in args
    args.clear()
    assert policy.proposer_restriction_args()


def test_secret_filter_and_torch_free_import():
    env = {'PATH': 'x', 'OPENAI_API_KEY': 'one', 'Codex_Api_Key': 'two',
           'TOKEN_ID': 'three', 'OTHER_SECRET': 'four', 'PASSWORD': 'five', 'TEMP': 'D:/tmp'}
    assert policy.restricted_environment(env) == {'PATH': 'x', 'TEMP': 'D:/tmp'}
    assert len(env) == 7
    child = subprocess.run([sys.executable, '-B', '-c',
                            'import sys; import factorlab.discovery.process_policy; assert "torch" not in sys.modules'],
                           capture_output=True, text=True, check=False)
    assert child.returncode == 0, child.stderr


def test_owned_idle_process_is_stopped_without_shell():
    process = subprocess.Popen([sys.executable, '-B', '-c',
                                'import time; print("ready",flush=True); time.sleep(30)'],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                               **policy.isolated_process_kwargs())
    try:
        assert process.stdout.readline().strip() == 'ready'
        result = policy.terminate_process_tree(process)
        assert process.poll() is not None and result['tree_command_completed']
        assert result['escaped_descendants_verified'] is False
        assert policy.terminate_process_tree(process)['status'] == 'already_exited'
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        process.stdout.close()


@pytest.mark.skipif(os.name != 'nt', reason='Windows native handle verifies descendant identity')
def test_windows_termination_stops_actual_grandchild():
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    child_code = 'import time; print("ready",flush=True); time.sleep(30)'
    parent_code = ('import subprocess,sys,time; '
                   f'p=subprocess.Popen([sys.executable,"-B","-c",{child_code!r}],'
                   'stdout=subprocess.PIPE,text=True,creationflags=subprocess.CREATE_NO_WINDOW); '
                   'assert p.stdout.readline().strip()=="ready"; '
                   'print(p.pid,flush=True); time.sleep(30)')
    process = subprocess.Popen([sys.executable, '-B', '-c', parent_code],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                               **policy.isolated_process_kwargs())
    handle = None
    try:
        child_pid = int(process.stdout.readline().strip())
        handle = kernel.OpenProcess(0x00100000, False, child_pid)  # SYNCHRONIZE only.
        assert handle
        assert kernel.WaitForSingleObject(handle, 0) == 258  # WAIT_TIMEOUT: still alive.
        policy.terminate_process_tree(process)
        assert kernel.WaitForSingleObject(handle, 5000) == 0  # Same process handle exited.
    finally:
        if process.poll() is None:
            policy.terminate_process_tree(process)
        process.wait(timeout=5)
        process.stdout.close()
        if handle:
            kernel.CloseHandle(handle)


@pytest.mark.parametrize('timeout', [0, -1, float('inf'), 11, True])
def test_timeout_validation_happens_before_process_action(timeout):
    process = Mock(pid=123)
    with pytest.raises(ValueError, match='timeout'):
        policy.terminate_process_tree(process, timeout=timeout)
    process.poll.assert_not_called()


@pytest.mark.skipif(os.name == 'nt', reason='POSIX-only dedicated-session contract')
def test_nonisolated_posix_refuses_broad_group_signal():
    process = subprocess.Popen([sys.executable, '-B', '-c', 'import time; time.sleep(30)'])
    try:
        with pytest.raises(RuntimeError, match='dedicated session'):
            policy.terminate_process_tree(process)
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
