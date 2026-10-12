"""Real temporary file-tree checks; no Linux or resource acceptance is implied."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def boundary(tmp_path, monkeypatch):
    guard = Path(__file__).resolve().parents[1]/'scripts/v7_guard'
    monkeypatch.syspath_prepend(str(guard))
    spec = importlib.util.spec_from_file_location('postmortem_control_test', guard/'tree_work_enforcer.py')
    tree = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tree)
    relative = '/user.slice/owned.service'
    leaf = tmp_path/'sys/fs/cgroup'/relative.lstrip('/')
    proc = tmp_path/'proc/self/cgroup'
    proc.parent.mkdir(parents=True)
    proc.write_text('0::'+relative+'/.control')

    def group(path, members='', populated='0'):
        path.mkdir(parents=True, exist_ok=True)
        (path/'cgroup.procs').write_text(members)
        (path/'cgroup.events').write_text('populated '+populated+'\nfrozen 0\n')
        return path

    group(leaf, populated='1')
    control = group(leaf/'.control', str(os.getpid()), '1')
    def mapped(path):
        text = str(path)
        return tmp_path/text.lstrip('/') if text.startswith(('/sys/', '/proc/')) else Path(path)
    monkeypatch.setattr(tree, 'Path', mapped)
    monkeypatch.setattr(tree, 'os', SimpleNamespace(name='posix', getpid=os.getpid, scandir=os.scandir))
    monkeypatch.setattr(tree.multi, '_boot_id', lambda: 'linux:original')
    monkeypatch.setattr(tree, '_original_dead', lambda identity: True)
    props = {'MainPID': '0', 'KillMode': 'control-group', 'Restart': 'no', 'ControlGroup': relative}
    def systemctl(*args, **kwargs):
        return '\n'.join(k+'='+v for k, v in props.items())
    monkeypatch.setattr(tree.subprocess, 'check_output', systemctl)
    value = {'platform': 'posix', 'boot_id': 'linux:original', 'guardian': {'pid': 123, 'created': 1.},
             'relative': relative, 'cgroup': '/sys/fs/cgroup'+relative,
             'inode': leaf.stat().st_ino, 'unit': leaf.name}
    return SimpleNamespace(tree=tree, leaf=leaf, control=control, proc=proc, value=value,
                           group=group, props=props, systemctl=systemctl)


def test_direct_control_inherits_original_service_with_empty_siblings(boundary):
    b = boundary
    b.group(b.leaf/'retired'/'empty')
    b.group(b.leaf/'retired')
    before = {p: p.read_bytes() for p in b.leaf.rglob('*') if p.is_file()}
    b.tree._postmortem_boundary(b.value)
    assert {p: p.read_bytes() for p in before} == before


def test_original_service_placement_still_passes(boundary):
    b = boundary
    b.proc.write_text('0::'+b.value['relative'])
    b.group(b.leaf, str(os.getpid()), '1')
    b.group(b.control)
    b.tree._postmortem_boundary(b.value)


@pytest.mark.parametrize('change', [
    'foreign', 'other_child', 'nested_control', 'live_root', 'live_sibling', 'live_nested',
    'extra_control_pid', 'missing_pid', 'missing_events', 'unknown_population',
    'duplicate_population', 'unaccounted_population', 'root_population_zero',
    'oversize_members', 'too_deep', 'too_many_groups', 'too_many_entries',
    'recreated_root', 'live_guardian', 'boot_changed', 'main_alive', 'foreign_unit',
])
def test_control_boundary_refuses_unknown_or_other_live_processes(boundary, monkeypatch, change):
    b = boundary
    if change in {'foreign', 'other_child', 'nested_control'}:
        suffix = {'foreign': '/other.service/.control', 'other_child': '/user.slice/owned.service/child',
                  'nested_control': b.value['relative']+'/.control/nested'}[change]
        b.proc.write_text('0::'+suffix)
    elif change == 'live_root': b.group(b.leaf, '987', '1')
    elif change == 'live_sibling': b.group(b.leaf/'sibling', '987', '1')
    elif change == 'live_nested':
        b.group(b.leaf/'sibling', populated='1')
        b.group(b.leaf/'sibling'/'nested', '987', '1')
    elif change == 'extra_control_pid': b.group(b.control, str(os.getpid())+'\n987', '1')
    elif change == 'missing_pid': b.group(b.control, '', '1')
    elif change == 'missing_events': (b.control/'cgroup.events').unlink()
    elif change == 'unknown_population': b.group(b.control, str(os.getpid()), 'unknown')
    elif change == 'duplicate_population': (b.control/'cgroup.events').write_text('populated 1\npopulated 1')
    elif change == 'unaccounted_population': b.group(b.leaf/'sibling', populated='1')
    elif change == 'root_population_zero': b.group(b.leaf)
    elif change == 'oversize_members': (b.control/'cgroup.procs').write_text('1 '*2049)
    elif change == 'too_deep':
        path = b.leaf
        for i in range(9): path = b.group(path/str(i))
    elif change == 'too_many_groups':
        for i in range(64): b.group(b.leaf/str(i))
    elif change == 'too_many_entries':
        for i in range(257): (b.control/str(i)).touch()
    elif change == 'recreated_root': b.value['inode'] += 1
    elif change == 'live_guardian': monkeypatch.setattr(b.tree, '_original_dead', lambda _: False)
    elif change == 'boot_changed': b.value['boot_id'] = 'linux:old'
    elif change == 'main_alive': b.props['MainPID'] = '987'
    elif change == 'foreign_unit': b.props['ControlGroup'] += '/different'
    with pytest.raises((ValueError, OSError)):
        b.tree._postmortem_boundary(b.value)


@pytest.mark.parametrize('change', ['new_member', 'membership', 'moved_to_root',
                                   'replace_control', 'new_sibling', 'boot'])
def test_changes_during_unit_readback_are_rejected(boundary, monkeypatch, change):
    b = boundary
    def changing_systemctl(*args, **kwargs):
        if change == 'new_member': b.group(b.control, str(os.getpid())+'\n987', '1')
        elif change == 'membership': b.proc.write_text('0::'+b.value['relative'])
        elif change == 'moved_to_root':
            b.proc.write_text('0::'+b.value['relative'])
            b.group(b.leaf, str(os.getpid()), '1')
            b.group(b.control)
        elif change == 'replace_control':
            b.control.rename(b.leaf/'old-control')
            b.group(b.leaf/'old-control')
            b.group(b.control, str(os.getpid()), '1')
        elif change == 'new_sibling': b.group(b.leaf/'new')
        elif change == 'boot': monkeypatch.setattr(b.tree.multi, '_boot_id', lambda: 'linux:new')
        return b.systemctl(*args, **kwargs)
    monkeypatch.setattr(b.tree.subprocess, 'check_output', changing_systemctl)
    with pytest.raises(ValueError):
        b.tree._postmortem_boundary(b.value)


@pytest.mark.skipif(os.name == 'nt', reason='POSIX-only real cgroup symlink boundary')
def test_linked_control_is_rejected(boundary):
    b = boundary
    destination = b.leaf.parent/'foreign'
    b.control.rename(destination)
    b.control.symlink_to(destination, target_is_directory=True)
    with pytest.raises(ValueError):
        b.tree._postmortem_boundary(b.value)
