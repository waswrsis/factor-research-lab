"""Pure finite filesystem/proc counterexamples, NOT Linux resource acceptance."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT/'scripts/v7_guard'


@pytest.fixture
def tree(monkeypatch):
    monkeypatch.syspath_prepend(str(GUARD))
    spec = importlib.util.spec_from_file_location('guard_contract_test', GUARD/'tree_work_enforcer.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('change', ['foreign_leaf', 'wrong_inode', 'live_main',
                                   'other_member', 'new_boot', 'wrong_restart'])
def test_linux_postmortem_rejects_unowned_or_live_boundary(tree, monkeypatch, tmp_path, change):
    leaf = tmp_path/'sys/fs/cgroup/user.slice/owned.service'
    leaf.mkdir(parents=True)
    proc = tmp_path/'proc/self'
    proc.mkdir(parents=True)
    relative = '/user.slice/owned.service'
    (proc/'cgroup').write_text('0::'+relative)
    (leaf/'cgroup.procs').write_text(str(tree.os.getpid()))
    (leaf/'cgroup.events').write_text('populated 1\nfrozen 0\n')
    identity = {'pid': 999999999, 'created': 1.0}
    value = {'platform': 'posix', 'boot_id': 'linux:observed', 'guardian': identity,
             'relative': relative, 'cgroup': '/sys/fs/cgroup'+relative,
             'inode': leaf.stat().st_ino, 'unit': leaf.name}
    # Real file reads/stats; only namespace root/kernel properties are synthetic.
    actual_path = Path

    def mapped(path):
        text = str(path)
        return actual_path(tmp_path/text.lstrip('/')) if text.startswith(('/sys/', '/proc/')) else actual_path(path)

    # The production check explicitly compares resolved paths; use a small wrapper
    # preserving displayed proc namespace while delegating all I/O to real files.
    class NamespacePath:
        def __init__(self, path):
            self.path = str(path)
        def __truediv__(self, item):
            return NamespacePath(self.path+'/'+str(item))
        def __eq__(self, other):
            return isinstance(other, NamespacePath) and self.path == other.path
        def __str__(self):
            return self.path
        def __fspath__(self):
            return str(mapped(self.path))
        @property
        def name(self):
            return actual_path(self.path).name
        def resolve(self):
            return self
        def stat(self):
            return mapped(self.path).stat()
        def read_text(self):
            return mapped(self.path).read_text()
        def open(self):
            return mapped(self.path).open()
        def is_symlink(self):
            return mapped(self.path).is_symlink()

    monkeypatch.setattr(tree, 'Path', NamespacePath)
    monkeypatch.setattr(tree, 'os', SimpleNamespace(name='posix', getpid=tree.os.getpid,
                                                   scandir=tree.os.scandir))
    monkeypatch.setattr(tree.multi, '_boot_id', lambda: 'linux:observed')
    properties = {'MainPID': '0', 'KillMode': 'control-group', 'Restart': 'no', 'ControlGroup': relative}
    monkeypatch.setattr(tree.subprocess, 'check_output', lambda *a, **k:
                        '\n'.join(k+'='+v for k, v in properties.items()))
    tree._postmortem_boundary(value)
    if change == 'foreign_leaf':
        (proc/'cgroup').write_text('0::/user.slice/other.service')
    elif change == 'wrong_inode':
        value['inode'] += 1
    elif change == 'live_main':
        properties['MainPID'] = '12'
    elif change == 'other_member':
        (leaf/'cgroup.procs').write_text(str(tree.os.getpid())+'\n12')
    elif change == 'new_boot':
        value['boot_id'] = 'linux:old'
    else:
        properties['Restart'] = 'always'
    with pytest.raises(ValueError):
        tree._postmortem_boundary(value)

