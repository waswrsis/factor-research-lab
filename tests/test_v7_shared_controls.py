"""Shared-credit controls: real synthetic fits and independent failure boundaries."""
import copy

import pandas as pd
import pytest
from test_v7_controls import StubBackend
from test_v7_rolling_registry import candles
from test_v7_run_factor_pool import source_root

from factorlab.discovery import shared_controls as controls
from factorlab.discovery.local_mutation import BudgetExhausted
from factorlab.discovery.rolling import RollingSpec
from factorlab.discovery.v7_run import V7Config, monthly_budget, run_month


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    return tmp_path, candles('2024-08-01', '2025-04-02'), '2025-04-01T00:00:00Z'


def test_real_three_arms_share_one_ceiling_and_readonly_resume(fixture):
    root, bars, cutoff = fixture
    backend = StubBackend()
    bars['available_at'] += pd.Timedelta(seconds=4)
    spec = RollingSpec(availability_delay_seconds=5)
    report = controls.run_shared_controls(root / 'controls', bars, cutoff, backend=backend, spec=spec)
    assert report['completed_budget_comparison']
    assert not report['formal_claims_enabled']
    assert report['reserved_total_counts'] == {'gpt_attempt': 6, 'family': 3, 'ast': 96, 'f': 24, 'solve': 1599}
    for key, maximum in report['reserved_total_counts'].items():
        assert report['actual_total_counts'][key] <= maximum
    assert all(a['readout']['actual_budget']['limits']['ast'] == 32 for a in report['arms'].values())
    assert all(a['readout']['actual_budget']['limits']['f'] == 8 for a in report['arms'].values())
    assert all(a['failure'] is None for a in report['arms'].values())
    calls = len(backend.packets)
    assert controls.inspect_shared_controls(root / 'controls')['completed_budget_comparison']
    assert controls.run_shared_controls(root / 'controls', bars, cutoff, backend=backend, spec=spec) == report
    assert len(backend.packets) == calls
    with pytest.raises(ValueError, match='Immutable'):
        controls.run_shared_controls(root / 'controls', bars, cutoff, backend=backend)


def _stub(monkeypatch, calls, fail=None):
    def run(directory, bars, cutoff, *, config, backend, external_deadline):
        calls.append((directory, config, external_deadline))
        if config.search_mode == fail:
            raise RuntimeError('failed original arm')
    monkeypatch.setattr(controls, 'run_month', run)
    def readout(directory, config, month):
        return {'status': 'complete', 'actual_budget': monthly_budget(config, month).to_dict()}
    monkeypatch.setattr(controls, '_readout', readout)


def test_failed_arm_does_not_refund_or_get_an_extra_envelope(fixture, monkeypatch):
    root, bars, cutoff = fixture
    calls = []
    _stub(monkeypatch, calls, 'random')
    result = controls.run_shared_controls(root / 'controls', bars, cutoff, backend=StubBackend())
    assert len(calls) == 3 and not result['completed_budget_comparison']
    assert result['arms']['random']['failure']['error_type'] == 'RuntimeError'
    assert len({call[2] for call in calls}) == 1
    assert controls.run_shared_controls(root / 'controls', bars, cutoff, backend=StubBackend()) == result
    assert len(calls) == 3


def test_crash_resumes_same_arm_and_original_deadline(fixture, monkeypatch):
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
    allocation = controls._read(root / 'controls/allocation.json')
    monkeypatch.setattr(controls, 'run_month', original)
    result = controls.run_shared_controls(root / 'controls', bars, cutoff, backend=StubBackend())
    assert result['completed_budget_comparison']
    assert calls[0][0] == calls[1][0]
    assert len(list(calls[0][0].glob('invocation-*.json'))) == 2
    assert controls._read(root / 'controls/allocation.json') == allocation


def test_expired_global_budget_closes_all_arms_without_training(fixture, monkeypatch):
    root, bars, cutoff = fixture
    calls = []
    _stub(monkeypatch, calls)
    original = controls.run_month
    class Crash(BaseException):
        pass
    monkeypatch.setattr(controls, 'run_month', lambda *args, **kwargs: (_ for _ in ()).throw(Crash()))
    with pytest.raises(Crash):
        controls.run_shared_controls(root / 'controls', bars, cutoff, backend=StubBackend())
    monkeypatch.setattr(controls, 'run_month', original)
    allocation = controls._read(root / 'controls/allocation.json')
    monkeypatch.setattr(controls, '_now', lambda: pd.Timestamp(allocation['deadline']) + pd.Timedelta(seconds=1))
    report = controls.run_shared_controls(root / 'controls', bars, cutoff, backend=StubBackend())
    assert not calls and not report['completed_budget_comparison']
    assert all(o['failure']['error_type'] == 'SharedDeadline' for o in report['arms'].values())


def test_shared_policy_cannot_silently_change_per_arm_settings():
    with pytest.raises(ValueError, match='third-month'):
        V7Config(budget_policy='controls_third')
    budget = monthly_budget(controls.control_config(), '2025-04')
    budget.register_family('only')
    with pytest.raises(BudgetExhausted):
        budget.register_family('extra')
    budget.consume('solve', 533)
    with pytest.raises(BudgetExhausted):
        budget.consume('solve')


def test_shared_source_delay_cannot_change_other_fit_rules(tmp_path):
    with pytest.raises(ValueError, match='every fitting rule'):
        controls.run_shared_controls(tmp_path / 'unused', None, '2025-04-01T00:00Z',
            backend=StubBackend(), spec=RollingSpec(ridge=.002))
    assert not (tmp_path / 'unused').exists()


def test_expired_external_deadline_precedes_any_source_or_data_access(tmp_path):
    with pytest.raises(BudgetExhausted, match='External'):
        run_month(tmp_path / 'run', None, None, external_deadline=0.)
    assert not (tmp_path / 'run').exists()


def test_readonly_inspect_rejects_full_budget_substitution(fixture, monkeypatch):
    root, bars, cutoff = fixture
    calls = []
    _stub(monkeypatch, calls)
    controls.run_shared_controls(root / 'controls', bars, cutoff, backend=StubBackend())
    saved = controls._read(root / 'controls/local-outcome.json')['readout']
    corrupted = copy.deepcopy(saved)
    corrupted['actual_budget']['limits']['ast'] = 96
    monkeypatch.setattr(controls, '_readout', lambda *args: corrupted)
    with pytest.raises(ValueError, match='differs|budget'):
        controls.inspect_shared_controls(root / 'controls')


def _crashed_allocation(fixture, monkeypatch):
    root, bars, cutoff = fixture
    calls = []
    _stub(monkeypatch, calls)
    original = controls.run_month
    class Crash(BaseException):
        pass
    def crash(*args, **kwargs):
        original(*args, **kwargs)
        raise Crash()
    monkeypatch.setattr(controls, 'run_month', crash)
    with pytest.raises(Crash):
        controls.run_shared_controls(root / 'controls', bars, cutoff, backend=StubBackend())
    monkeypatch.setattr(controls, 'run_month', original)
    return root, bars, cutoff, calls


def test_boot_time_estimate_drift_does_not_renew_shared_allocation(fixture, monkeypatch):
    import psutil
    monkeypatch.setattr(controls, '_boot_id', lambda: 'windows:kuser-boot-v1:100')
    root, bars, cutoff, calls = _crashed_allocation(fixture, monkeypatch)
    allocation_bytes = (root / 'controls/allocation.json').read_bytes()
    monkeypatch.setattr(psutil, 'boot_time', lambda: (_ for _ in ()).throw(AssertionError('Estimated boot time used')))
    report = controls.run_shared_controls(root / 'controls', bars, cutoff, backend=StubBackend())
    assert report['completed_budget_comparison'] and len(calls) == 4
    assert (root / 'controls/allocation.json').read_bytes() == allocation_bytes
    assert len({call[2] for call in calls}) == 1


@pytest.mark.parametrize('damage', ['legacy', 'missing_version', 'empty', 'bool', 'noncanonical', 'extra'])
def test_invalid_boot_allocation_cannot_resume_or_spend(fixture, monkeypatch, damage):
    import json
    monkeypatch.setattr(controls, '_boot_id', lambda: 'windows:kuser-boot-v1:100')
    root, bars, cutoff, calls = _crashed_allocation(fixture, monkeypatch)
    path = root / 'controls/allocation.json'
    from factorlab.protocol import fingerprint
    value = controls._read(path)
    if damage == 'legacy':
        value['host_boot_time'] = 100.
        del value['host_boot_id']
    elif damage == 'missing_version':
        del value['version']
    elif damage == 'extra':
        value['additional_lease'] = 7200
    else:
        value['host_boot_id'] = {'empty': '', 'bool': True, 'noncanonical': 'windows:kuser-boot-v1:0100'}[damage]
    path.write_text(json.dumps({'value': value, 'sha256': fingerprint(value)}))
    before = path.read_bytes()
    with pytest.raises(ValueError, match='allocation'):
        controls.run_shared_controls(root / 'controls', bars, cutoff, backend=StubBackend())
    assert len(calls) == 1 and path.read_bytes() == before
    assert len(list((root / 'controls/arms/local').glob('invocation-*.json'))) == 1


def test_unknown_boot_identity_does_not_start_a_new_arm(fixture, monkeypatch):
    monkeypatch.setattr(controls, '_boot_id', lambda: 'windows:kuser-boot-v1:100')
    root, bars, cutoff, calls = _crashed_allocation(fixture, monkeypatch)
    allocation_bytes = (root / 'controls/allocation.json').read_bytes()
    def unavailable():
        raise OSError('Kernel identity unavailable')
    monkeypatch.setattr(controls, '_boot_id', unavailable)
    with pytest.raises(OSError, match='unavailable'):
        controls.run_shared_controls(root / 'controls', bars, cutoff, backend=StubBackend())
    assert len(calls) == 1 and (root / 'controls/allocation.json').read_bytes() == allocation_bytes
