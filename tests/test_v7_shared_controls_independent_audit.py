"""Independent receipts/recovery counterexamples; no GPT or numerical training."""
import json

import pandas as pd
import pytest
from test_v7_controls import StubBackend
from test_v7_shared_controls import _stub
from test_v7_shared_controls import fixture as fixture_source

from factorlab.discovery import shared_controls as controls
from factorlab.protocol import fingerprint


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    return fixture_source.__wrapped__(tmp_path, monkeypatch)


def test_missing_allocation_cannot_reset_deadline_after_reserved_arm(fixture, monkeypatch):
    root, bars, cutoff = fixture
    calls = []
    _stub(monkeypatch, calls)
    original = controls.run_month

    class Crash(BaseException):
        pass

    def crashed(*args, **kwargs):
        original(*args, **kwargs)
        raise Crash()

    monkeypatch.setattr(controls, 'run_month', crashed)
    with pytest.raises(Crash):
        controls.run_shared_controls(root / 'controls', bars, cutoff, backend=StubBackend())
    path = root / 'controls/allocation.json'
    old = controls._read(path)
    path.unlink()
    monkeypatch.setattr(controls, '_now', lambda: pd.Timestamp(old['deadline']) + pd.Timedelta(hours=1))
    monkeypatch.setattr(controls, 'run_month', original)
    with pytest.raises(ValueError, match='allocation|reservation|reset|missing'):
        controls.run_shared_controls(root / 'controls', bars, cutoff, backend=StubBackend())
    assert len(calls) == 1


def test_failed_arm_cannot_be_relabelled_completed_in_report(fixture, monkeypatch):
    root, bars, cutoff = fixture
    _stub(monkeypatch, [], fail='random')
    result = controls.run_shared_controls(root / 'controls', bars, cutoff, backend=StubBackend())
    assert not result['completed_budget_comparison']
    path = root / 'controls/report.json'
    result['completed_budget_comparison'] = True
    path.write_text(json.dumps({'value': result, 'sha256': fingerprint(result)}))
    with pytest.raises(ValueError, match='report|comparison|completed|derived'):
        controls.inspect_shared_controls(root / 'controls')


@pytest.mark.parametrize('late_arm', ['local', 'gpt'])
def test_monotonic_late_arm_is_failed_even_if_wall_remains_inside_lease(fixture, monkeypatch, late_arm):
    root, bars, cutoff = fixture
    calls = []
    _stub(monkeypatch, calls)
    original = controls.run_month
    wall = pd.Timestamp('2026-10-05T00:00:00Z')
    mono = [100.]
    monkeypatch.setattr(controls, '_now', lambda: wall)
    monkeypatch.setattr(controls.time, 'monotonic', lambda: mono[0])

    def late_return(*args, **kwargs):
        original(*args, **kwargs)
        if kwargs['config'].search_mode == late_arm:
            mono[0] = kwargs['external_deadline'] + 1

    monkeypatch.setattr(controls, 'run_month', late_return)
    report = controls.run_shared_controls(root / 'controls', bars, cutoff, backend=StubBackend())
    assert len(calls) == (1 if late_arm == 'local' else 3)
    failure = report['arms'][late_arm]['failure']
    assert failure is not None and failure['error_type'] == 'SharedDeadline'
    assert not report['completed_budget_comparison']


def test_resume_wall_rollback_inside_started_interval_cannot_extend_original_monotonic(fixture, monkeypatch):
    root, bars, cutoff = fixture
    calls = []
    _stub(monkeypatch, calls)
    original = controls.run_month
    wall = [pd.Timestamp('2026-10-05T00:00:00Z')]
    mono = [100.]
    monkeypatch.setattr(controls, '_now', lambda: wall[0])
    monkeypatch.setattr(controls.time, 'monotonic', lambda: mono[0])

    class Crash(BaseException):
        pass

    def crash(*a, **kw):
        original(*a, **kw)
        raise Crash()

    monkeypatch.setattr(controls, 'run_month', crash)
    with pytest.raises(Crash):
        controls.run_shared_controls(root / 'controls', bars, cutoff, backend=StubBackend())
    wall[0] += pd.Timedelta(seconds=10)  # Real monotonic time advanced much further.
    mono[0] += 1000
    monkeypatch.setattr(controls, 'run_month', original)
    controls.run_shared_controls(root / 'controls', bars, cutoff, backend=StubBackend())
    assert {call[2] for call in calls} == {7300.}


def test_reboot_cannot_resume_old_monotonic_allocation(fixture, monkeypatch):
    root, bars, cutoff = fixture
    calls = []
    _stub(monkeypatch, calls)
    original = controls.run_month
    boot = ['windows:kuser-boot-v1:100']
    monkeypatch.setattr(controls, '_boot_id', lambda: boot[0])

    class Crash(BaseException):
        pass

    def crash(*a, **kw):
        original(*a, **kw)
        raise Crash()

    monkeypatch.setattr(controls, 'run_month', crash)
    with pytest.raises(Crash):
        controls.run_shared_controls(root / 'controls', bars, cutoff, backend=StubBackend())
    boot[0] = 'windows:kuser-boot-v1:101'
    monkeypatch.setattr(controls, 'run_month', original)
    with pytest.raises(ValueError, match='host clock'):
        controls.run_shared_controls(root / 'controls', bars, cutoff, backend=StubBackend())
    assert len(calls) == 1
