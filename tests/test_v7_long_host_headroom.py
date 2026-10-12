import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import v7_long_running as candidate

REAL_GUARD = candidate.HostGuard
MIB = 2**20


def snapshot(current=500, **changes):
    return {'path': '/sys/fs/cgroup/app.slice/owned.service', 'inode': 7, 'boot_id': 'actual-boot',
            'pid': os.getpid(), 'memory_current_bytes': current*MIB,
            'memory_max_bytes': 768*MIB} | changes


def host(monkeypatch, available, *, before=500, after=None, processes=(), rss=20):
    readings = iter([snapshot(before), snapshot(before if after is None else after)])
    monkeypatch.setattr(candidate, '_owned_memory_snapshot', lambda: next(readings))
    monkeypatch.setattr(candidate, 'HostGuard', lambda **kwargs: REAL_GUARD(**kwargs, inspect=lambda:
        {'rss_mib': rss, 'available_mib': available, 'processes': list(processes)}))
    return candidate._headroom_host_guard()


def test_current_owned_500_allows_available_640_without_changing_rss_process_guard(monkeypatch):
    result = host(monkeypatch, 640)
    assert result['allowed']
    assert result['host_headroom']['required_available_bytes'] == 396*MIB


@pytest.mark.parametrize(('current', 'available', 'allowed'), [(0, 895, False), (0, 896, True),
    (768, 127, False), (768, 128, True), (500, 395, False), (500, 396, True)])
def test_exact_headroom_boundaries(monkeypatch, current, available, allowed):
    assert host(monkeypatch, available, before=current)['allowed'] is allowed


def test_smaller_bracketing_credit_is_used(monkeypatch):
    result = host(monkeypatch, 400, before=500, after=400)
    assert not result['allowed']
    assert result['host_headroom']['credited_current_bytes'] == 400*MIB
    assert result['host_headroom']['required_available_bytes'] == 496*MIB


@pytest.mark.parametrize('key', ['path', 'inode', 'boot_id', 'pid', 'memory_max_bytes'])
def test_changed_resource_boundary_rejected(monkeypatch, key):
    readings = iter([snapshot(), snapshot(**{key: 'changed'})])
    monkeypatch.setattr(candidate, '_owned_memory_snapshot', lambda: next(readings))
    monkeypatch.setattr(candidate, 'HostGuard', lambda **kwargs: lambda: {'allowed': True, 'available_mib': 999, 'reasons': []})
    with pytest.raises(ValueError, match='boundary changed'):
        candidate._headroom_host_guard()


def test_legacy_rss_and_failed_inspection_are_not_relaxed(monkeypatch):
    assert 'legacy_long80_active' in host(monkeypatch, 999, processes=[{'pid': 42, 'command': 'python /long80/run.py'}])['reasons']
    assert 'rss_limit' in host(monkeypatch, 999, rss=768)['reasons']
    readings = iter([snapshot(), snapshot()])
    monkeypatch.setattr(candidate, '_owned_memory_snapshot', lambda: next(readings))
    blocked = {'allowed': False, 'reasons': ['host_inspection_failed'], 'error_type': 'AccessDenied'}
    monkeypatch.setattr(candidate, 'HostGuard', lambda **kwargs: lambda: blocked)
    result = candidate._headroom_host_guard()
    assert not result['allowed'] and result['error_type'] == 'AccessDenied'


@pytest.fixture
def kernel(tmp_path, monkeypatch):
    root = tmp_path/'sys-fs-cgroup'
    node = root/'app.slice/owned.service'
    node.mkdir(parents=True)
    (node/'memory.max').write_text(str(768*MIB))
    (node/'memory.current').write_text(str(500*MIB))
    (node/'cgroup.procs').write_text(str(os.getpid()))
    proc = tmp_path/'self-cgroup'
    proc.write_text('0::/app.slice/owned.service\n')
    boot = tmp_path/'boot'
    boot.write_text('1c4e0a00-36fc-498b-91d2-3cdce77ac26d')
    mapping = {'/sys/fs/cgroup': root, '/proc/self/cgroup': proc, '/proc/sys/kernel/random/boot_id': boot}
    monkeypatch.setattr(candidate, 'Path', lambda name: mapping.get(str(name), Path(name)))
    monkeypatch.setattr(candidate, 'sys', SimpleNamespace(platform='linux'))
    return node, proc


def test_kernel_reads_exact_current_leaf(kernel):
    node, _ = kernel
    observed = candidate._owned_memory_snapshot()
    assert observed['path'] == str(node.resolve()) and observed['memory_current_bytes'] == 500*MIB


def test_kernel_invalid_boot_identity_fails_closed(kernel):
    candidate.Path('/proc/sys/kernel/random/boot_id').write_text('unverified')
    with pytest.raises(ValueError, match='boot identity'):
        candidate._owned_memory_snapshot()


@pytest.mark.parametrize(('filename', 'value'), [('memory.max', 'max'), ('memory.max', str(769*MIB)),
    ('memory.max', '0'), ('memory.current', '-1'), ('memory.current', str(769*MIB)),
    ('memory.current', 'not-known'), ('cgroup.procs', '999999999')])
def test_unknown_or_unowned_kernel_credit_is_rejected(kernel, filename, value):
    node, _ = kernel
    (node/filename).write_text(value)
    with pytest.raises(ValueError):
        candidate._owned_memory_snapshot()


@pytest.mark.parametrize('path', ['/app.slice', '/user@1000.service', '/session-12.scope', '/app.slice/sibling.service'])
def test_cannot_borrow_shared_ancestor_or_other_service(kernel, path):
    _, proc = kernel
    proc.write_text('0::'+path+'\n')
    with pytest.raises((ValueError, FileNotFoundError)):
        candidate._owned_memory_snapshot()


def test_production_hard_caps_still_fail_before_credit(monkeypatch):
    def deny():
        raise ValueError('Invalid CPU/high/max/swap/pids kernel envelope')
    monkeypatch.setattr(candidate, 'inherited_limits', deny)
    monkeypatch.setattr(candidate, '_headroom_host_guard', lambda: pytest.fail('Must not credit before hard-cap check'))
    assert candidate.production_guard(SimpleNamespace())['allowed'] is False


def test_production_uses_new_credit_and_preserves_verified_limits(monkeypatch):
    monkeypatch.setattr(candidate, 'inherited_limits', lambda: {'scope': 'actual-kernel'})
    monkeypatch.setattr(candidate.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=0,
        stdout='LoadState=loaded\nActiveState=inactive\nExecStart=/tools/long80/supervise_iteration.py pilot\n'))
    monkeypatch.setattr(candidate, '_headroom_host_guard', lambda: {'allowed': True, 'host_headroom': {'credit': 'kernel'}})
    plan = SimpleNamespace(old_unit='old.service', old_exec_start='/tools/long80/supervise_iteration.py pilot')
    assert candidate.production_guard(plan)['hard_limits_verified']
