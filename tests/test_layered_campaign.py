"""Finite controller wiring and identity; never launches a campaign/provider."""
import hashlib
import importlib.util
import runpy
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

ROOT = Path('D:/FactorResearch')
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


policy = load('factorlab.discovery._layered_campaign_policy_stage',
              HERE/'factorlab/discovery/layered_research.py')
campaign = load('_layered_campaign_controller_stage', HERE/'scripts/v7_long_running.py')


def plan():
    return campaign.CampaignPlan(end_at='2027-04-01T00:00:00Z', max_sessions=2,
        max_reserved_seconds=14400, max_reserved_steps=240, max_fit_attempts=20,
        forward_directory='runs/staged-test-forward', feed_directory='runs/staged-test-feed',
        research={'backend': 'disabled', 'research': {'evaluation_mode': 'joint', 'search_mode': 'local'}},
        lifecycle={'source_id': 'actual-feed', 'training_window_days': 90,
                                'refit_interval_days': 15})


def test_legacy_default_shape_and_runtime_keywords_unchanged():
    original = plan()
    expected = asdict(original)
    expected.pop('research_spec')
    assert campaign.campaign_plan_fields(original) == expected
    args = campaign.campaign_runtime_kwargs(original, lambda: None)
    assert set(args) == {'config', 'lifecycle_plan', 'clock'}


def test_layered_mode_reaches_worker_with_original_budgets(monkeypatch):
    monkeypatch.setitem(sys.modules, 'factorlab.discovery.layered_research', policy)
    original = plan()
    spec = policy.LayeredResearchSpec('a'*64)
    changed = replace(original, research_spec=asdict(spec))
    assert campaign.campaign_plan_fields(changed)['research_spec'] == asdict(spec)
    args = campaign.campaign_runtime_kwargs(changed, lambda: None)
    assert isinstance(args['research_spec'], policy.LayeredResearchSpec)
    assert args['research_spec'].digest == spec.digest
    assert args['lifecycle_plan'].training_window_days == 90
    assert args['lifecycle_plan'].refit_interval_days == 15
    assert changed.max_sessions == original.max_sessions
    assert changed.max_fit_attempts == original.max_fit_attempts
    assert changed.max_reserved_seconds == original.max_reserved_seconds


def test_mixed_mode_and_legacy_certificate_rejected_before_worker(monkeypatch):
    monkeypatch.setitem(sys.modules, 'factorlab.discovery.layered_research', policy)
    original = plan()
    spec = asdict(policy.LayeredResearchSpec('a'*64))
    for fields in ({'source_id': 'actual-feed'},
                   {**original.lifecycle, 'calibration_directory': 'old', 'calibration_binding': 'old'},
                   {**original.lifecycle, 'confirmation_months': 6}):
        with pytest.raises(ValueError, match='cannot mix'):
            replace(original, lifecycle=fields, research_spec=spec)
    with pytest.raises(ValueError, match='object'):
        replace(original, research_spec=True)


def test_calibration_CLI_is_in_new_executable_source_binding(tmp_path):
    evidence = load('factorlab.discovery._layered_evidence_source_stage',
                    HERE/'factorlab/discovery/evidence.py')
    (tmp_path/'scripts').mkdir()
    cli = tmp_path/'scripts/v7_layered_validation.py'
    cli.write_text('SYNTHETIC_CLI=True\n')
    assert cli in evidence.source_paths(tmp_path)


def test_linux_package_contains_every_frozen_executable_asset(tmp_path):
    bootstrap = runpy.run_path(str(HERE/'scripts/v7_linux_bootstrap.sh'))
    evidence = load('factorlab.discovery._layered_pack_evidence_stage',
                    HERE/'factorlab/discovery/evidence.py')
    source = tmp_path/'source'
    for name in (*bootstrap['FIXED'], 'factorlab/discovery/v7_cli.py',
                 'factorlab/discovery/layered_research.py'):
        path = source/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('SYNTHETIC_PACKAGE_BYTES = True\n')
    receipt = bootstrap['pack'](source, tmp_path/'source.tar')
    _, entries = bootstrap['verify_package'](receipt['package'], receipt['package_sha256'])
    assert 'scripts/v7_layered_validation.py' in entries
    frozen = {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in evidence.source_paths(source)}
    assert set(frozen) <= set(entries)
    assert all(hashlib.sha256(entries[n]).hexdigest() == h for n, h in frozen.items())
