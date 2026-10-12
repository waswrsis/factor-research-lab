"""Staged constructor integration, no fit/network/prediction performance claim."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path('D:/FactorResearch')
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parents[1] / 'factorlab/discovery'


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


policy = load('factorlab.discovery._layered_forward_policy_stage', 'layered_research.py')
forward = load('factorlab.discovery._layered_forward_runtime_stage', 'v7_forward.py')
from factorlab.discovery.v7_lifecycle import Lifecycle


def source_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    (tmp_path/'factorlab').mkdir()
    (tmp_path/'factorlab/fixture.py').write_text('SYNTHETIC_SOURCE = True\n')
    (tmp_path/'pyproject.toml').write_text('[project]\n')
    (tmp_path/'uv.lock').write_text('version=1\n')


def test_forward_new_mode_identity_reopen_and_no_legacy_retrofit(tmp_path, monkeypatch):
    source_fixture(tmp_path, monkeypatch)
    monkeypatch.setitem(sys.modules, 'factorlab.discovery.layered_research', policy)
    clock = lambda: pd.Timestamp('2026-10-10T16:00:00Z')
    config = forward.ForwardConfig(training_window_days=90, refit_interval_days=15)
    spec = policy.LayeredResearchSpec(policy.fingerprint(policy.source_files()))
    runtime = forward.ForwardRuntime(tmp_path/'new', tmp_path/'feed',
        config=config, clock=clock, research_spec=spec)
    assert isinstance(runtime.lifecycle, policy.LayeredResearchLifecycle)
    assert runtime.identity['statistical_mode'] == 'prospective_uncertified_research'
    assert runtime.identity['research_spec_hash'] == spec.digest
    assert set(runtime.identity['research_implementations']) == {'forward', 'research_policy'}
    assert runtime.lifecycle.confirmation_execution_ready()
    assert not runtime.lifecycle.calibration_ready()
    runtime.close()
    reopened = forward.ForwardRuntime(tmp_path/'new', tmp_path/'feed',
        config=config, clock=clock, research_spec=spec)
    assert reopened.lifecycle.verify()['jobs'] == 0
    assert len(reopened.lifecycle._research_events()) == 1
    reopened.close()
    snapshot = tmp_path/'new/runtime_implementations/forward.json'
    saved = snapshot.read_bytes()
    snapshot.write_text('{}')
    with pytest.raises(ValueError, match='snapshot'):
        forward.ForwardRuntime(tmp_path/'new', tmp_path/'feed',
            config=config, clock=clock, research_spec=spec)
    snapshot.write_bytes(saved)
    # Omitting the mode cannot reinterpret the existing new run as a legacy run.
    with pytest.raises(ValueError, match='Immutable'):
        forward.ForwardRuntime(tmp_path/'new', tmp_path/'feed', config=config, clock=clock)


def test_legacy_forward_identity_gate_and_new_mode_are_separate(tmp_path, monkeypatch):
    source_fixture(tmp_path, monkeypatch)
    monkeypatch.setitem(sys.modules, 'factorlab.discovery.layered_research', policy)
    clock = lambda: pd.Timestamp('2026-10-10T16:00:00Z')
    config = forward.ForwardConfig(training_window_days=90, refit_interval_days=15)
    old = forward.ForwardRuntime(tmp_path/'old', tmp_path/'feed', config=config, clock=clock)
    assert type(old.lifecycle) is Lifecycle
    assert 'research_spec' not in old.identity
    assert not old.lifecycle.confirmation_execution_ready()
    old.close()
    spec = policy.LayeredResearchSpec(policy.fingerprint(policy.source_files()))
    with pytest.raises(ValueError, match='Immutable'):
        forward.ForwardRuntime(tmp_path/'old', tmp_path/'feed',
            config=config, clock=clock, research_spec=spec)


def test_invalid_mode_has_no_lifecycle_enrollment(tmp_path, monkeypatch):
    source_fixture(tmp_path, monkeypatch)
    monkeypatch.setitem(sys.modules, 'factorlab.discovery.layered_research', policy)
    config = forward.ForwardConfig(training_window_days=90, refit_interval_days=15)
    with pytest.raises(ValueError, match='immutable'):
        forward.ForwardRuntime(tmp_path/'invalid', tmp_path/'feed', config=config, research_spec={})
    assert not (tmp_path/'invalid/lifecycle.sqlite').exists()
    spec = policy.LayeredResearchSpec(policy.fingerprint(policy.source_files()))
    with pytest.raises(ValueError, match='elapsed'):
        forward.ForwardRuntime(tmp_path/'wrong-cadence', tmp_path/'feed', research_spec=spec)
    assert not (tmp_path/'wrong-cadence/forward.json').exists()


@pytest.mark.parametrize('damage', ['missing', 'changed'])
def test_reopen_never_recreates_or_repairs_research_manifest(tmp_path, monkeypatch, damage):
    source_fixture(tmp_path, monkeypatch)
    monkeypatch.setitem(sys.modules, 'factorlab.discovery.layered_research', policy)
    clock = lambda: pd.Timestamp('2026-10-10T16:00:00Z')
    config = forward.ForwardConfig(training_window_days=90, refit_interval_days=15)
    spec = policy.LayeredResearchSpec(policy.fingerprint(policy.source_files()))
    runtime = forward.ForwardRuntime(tmp_path/'new', tmp_path/'feed',
        config=config, clock=clock, research_spec=spec)
    runtime.close()
    manifest = tmp_path/'new/layered-research.json'
    if damage == 'missing':
        manifest.unlink()
    else:
        manifest.write_text('{}')
    with pytest.raises(ValueError, match='manifest'):
        forward.ForwardRuntime(tmp_path/'new', tmp_path/'feed',
            config=config, clock=clock, research_spec=spec)
    if damage == 'missing':
        assert not manifest.exists()
    else:
        assert manifest.read_text() == '{}'
