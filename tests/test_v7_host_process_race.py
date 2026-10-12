"""Terminated children must not block a supervisor; unknown living ones must."""
import os
from types import SimpleNamespace

import psutil
import pytest

from factorlab.discovery.v7_service import HostGuard


class Process:
    def __init__(self, pid, command, status, name='python'):
        self.info = {'pid': pid, 'cmdline': command, 'name': name, 'status': status}
        self.result = status

    def status(self):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def observe(monkeypatch, processes):
    monkeypatch.setattr(psutil, 'process_iter', lambda *a: iter(processes))
    monkeypatch.setattr(psutil, 'Process', lambda: SimpleNamespace(
        memory_info=lambda: SimpleNamespace(rss=20 * 2**20)))
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: SimpleNamespace(available=1000 * 2**20))
    return HostGuard()()


@pytest.mark.parametrize('command', [None, []])
def test_confirmed_zombie_is_not_a_running_service(monkeypatch, command):
    result = observe(monkeypatch, [Process(42, command, psutil.STATUS_ZOMBIE)])
    assert result['allowed'] is True
    assert result['blocking_pids'] == []


def test_disappeared_python_is_ignored_after_positive_no_such_process(monkeypatch):
    assert observe(monkeypatch, [Process(42, None, psutil.NoSuchProcess(42))])['allowed'] is True


@pytest.mark.parametrize('command', [None, []])
def test_unreadable_living_python_still_blocks_with_pid_diagnostic(monkeypatch, command):
    result = observe(monkeypatch, [Process(42, command, psutil.STATUS_SLEEPING)])
    assert result['allowed'] is False
    assert result['unreadable_process'] == {'pid': 42, 'status': psutil.STATUS_SLEEPING}


def test_denied_status_still_blocks(monkeypatch):
    result = observe(monkeypatch, [Process(42, None, psutil.AccessDenied(42))])
    assert result['allowed'] is False
    assert result['error_type'] == 'AccessDenied'


def test_readable_legacy_python_still_blocks(monkeypatch):
    result = observe(monkeypatch, [Process(42, ['python', '/long80/run.py'], psutil.STATUS_SLEEPING)])
    assert result['allowed'] is False
    assert result['blocking_pids'] == [42]


def test_own_pid_does_not_false_block(monkeypatch):
    assert observe(monkeypatch, [Process(os.getpid(), None, psutil.STATUS_SLEEPING)])['allowed'] is True
