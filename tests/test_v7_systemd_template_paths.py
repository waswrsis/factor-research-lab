"""Path settings and argv have different systemd grammars; no service execution."""
import ast
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest


class TemplatePath(PurePosixPath):
    """Filesystem seam for grammar-only tests; not guard acceptance evidence."""

    def is_file(self):
        return self.name in {'lease_enforcer.py', 'multiwork_enforcer.py',
                             'tree_work_enforcer.py', 'guarded_runtime.py',
                             'campaign_guard_entry.py'}

    def exists(self):
        return False


def render(script, root):
    root = TemplatePath(root)
    source = Path(script).read_text('utf-8')
    name = 'unit_template' if script.endswith('.py') else 'systemd_template'
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {'Path': TemplatePath, 'now': None, 're': __import__('re'),
        'project_path': lambda p: root if p == '.' else TemplatePath(p),
        'inspect_campaign': lambda p: {},
        # Only formatting is isolated here. Real guard gates are tested separately.
        'guarded_command': lambda root, command: None,
        'read_plan': lambda p: SimpleNamespace(end_at=60),
        'stamp': lambda p: SimpleNamespace(__unused__=p),
        'workspace_path': lambda *a, **kw: root}

    class Time:
        def __sub__(self, other):
            return SimpleNamespace(total_seconds=lambda: 60)
    namespace['stamp'] = lambda _: Time()
    exec(compile(ast.Module(body=[node], type_ignores=[]), script, 'exec'), namespace)  # noqa: S102
    if name == 'unit_template':
        return namespace[name]('/campaign', clock=lambda: None)
    campaign = root / 'runs/test'
    command = [root / '.venv/bin/python', '-B',
               root / 'scripts/v7_guard/campaign_guard_entry.py', 'guard',
               campaign, campaign / 'whole_tree_guard']
    return namespace[name](root, 'old.service', 'user', command)


SCRIPTS = ['scripts/v7_long_running.py', 'scripts/v7_linux_bootstrap.sh']


@pytest.mark.parametrize('script', SCRIPTS)
@pytest.mark.parametrize('name', ['plain', 'with space', 'percent%h', '$DOLLAR', 'trailing '])
def test_path_setting_preserves_literal_characters_without_argv_quotes(script, name):
    root = PurePosixPath('/tmp') / name
    unit = render(script, root)
    line = next(line for line in unit.splitlines() if line.startswith('WorkingDirectory='))
    value = line.removeprefix('WorkingDirectory=')
    assert value.startswith('/') and not value.startswith('"')
    expected = str(root) + ('/.' if name[-1] in ' \\' else '')
    assert value.replace('%%', '%') == expected
    if '$' in name:
        assert '$$' not in value  # No environment expansion in this path setting.
        assert '$$DOLLAR' not in unit and 'ExecStart=:' in unit
    if name.endswith('\\'):
        assert not line.endswith('\\')  # Cannot continue into ExecStart.
    assert sum(line.startswith('ExecStart=') for line in unit.splitlines()) == 1


@pytest.mark.parametrize('script', SCRIPTS)
@pytest.mark.parametrize('name', [r'back\slash', 'trailing\\', 'quote"', "quote'"])
def test_systemd_unsupported_executable_workspace_characters_fail_explicitly(script, name):
    with pytest.raises(ValueError, match='Invalid systemd working directory'):
        render(script, PurePosixPath('/tmp') / name)


@pytest.mark.parametrize('script', SCRIPTS)
@pytest.mark.parametrize('bad', ['\n', '\r', '\t', '\x00', '\x1b', '\x7f', '\x85', '\u2028'])
def test_control_characters_cannot_inject_or_mutate_a_unit(script, bad):
    with pytest.raises(ValueError, match='Invalid systemd'):
        render(script, PurePosixPath('/tmp') / ('prefix' + bad + 'suffix'))
