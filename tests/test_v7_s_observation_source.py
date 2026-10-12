"""Real local physical writes; no provider, training, Linux or formal evidence."""
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

from factorlab.discovery import finite_market_oracle as finite
from factorlab.discovery.calibration_acceptance import AcceptanceCell, AcceptancePlan
from factorlab.discovery.calibration_runtime_contract import _original_operations, runtime_calibration_contract
from factorlab.discovery.full_calibration_registry import FullTrialPlan, FullTrialRegistry
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
from factorlab.discovery.neural_forward_ticket import _bind_authority
from factorlab.discovery.v7_forward import ForwardRuntime
from factorlab.protocol import fingerprint

# Canonical package name, relative production imports and real dependencies.
MODULE = 'factorlab.discovery.quarantined_observation'
local_source = Path(__file__).with_name('quarantined_observation.py')
if local_source.is_file():
    spec = importlib.util.spec_from_file_location(MODULE, local_source)
    observation = importlib.util.module_from_spec(spec)
    sys.modules[MODULE] = observation
    spec.loader.exec_module(observation)
else:
    observation = importlib.import_module(MODULE)
Q = observation.QuarantinedObservation


@pytest.fixture
def tmp_path(tmp_path_factory):
    # Preserve actual Windows path semantics without exceeding MAX_PATH.
    return tmp_path_factory.mktemp('t')


@pytest.fixture
def factory(tmp_path, monkeypatch):
    from factorlab import protocol
    repo = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
    originals = {p: (repo/p).read_bytes() for p in finite.PROOF_FILES}
    for name in ('v7_lifecycle', 'forward_fit_snapshot', 'neural_forward_ticket', 'neural_forward_contract',
                 'v7_forward', 'rolling', 'pool_forward', 'known_linear_forward', 'quarantined_observation',
                 'finite_market_oracle', 'nonlinear_runtime', 's_nonlinear_gates', 'forward_preparation', 'v7_run', 'v7_autonomous', 's_outer_context', 's_nonlinear_controller',
                 's_nonlinear_gates', 'neural_research_ticket', 'structural_validation', 's_observation_source'):
        path = Path(importlib.import_module('factorlab.discovery.'+name).__file__)
        originals['factorlab/discovery/'+path.name] = path.read_bytes()
    monkeypatch.setattr(protocol, 'ROOT', tmp_path)
    (tmp_path/'factorlab').mkdir()
    (tmp_path/'factorlab/v7.py').write_text('version=7')
    (tmp_path/'pyproject.toml').write_text('[project]')
    (tmp_path/'uv.lock').write_text('version=1')
    for name, body in originals.items():
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    profile = ForwardRuntime(tmp_path/'profile', tmp_path/'profile-feed')
    wallet = MonthlyWorkAuthority(tmp_path/'profile-wallet', run_id='profile', task='volatility',
                                  universe=['BTCUSDT'], partition_policy='standard')
    _bind_authority(profile, wallet)
    scope = runtime_calibration_contract(profile.lifecycle)
    profile.close()
    opened = []

    def make(index=0, gap_probability=0., *, future=False, law_start='2026-01-31T12:00:00Z', hours=24):
        root = tmp_path/f'case-{index}'
        start = (pd.Timestamp.now(tz='UTC').floor('h')+pd.Timedelta(days=2) if future
                 else pd.Timestamp(law_start))
        law = finite.FiniteMarketLaw(start.isoformat(), hours=hours, gap_probability=gap_probability)
        acceptance = AcceptancePlan((AcceptanceCell('finite', law.digest, 1000),))
        gen = finite.FiniteGenerator(root/'generator', law, acceptance)
        plan = FullTrialPlan(acceptance, scope,
            {'backend': 'codex', 'model_requested': 'gpt-6-astra', 'reasoning_requested': 'high',
             'context_policy': 'fresh_codex_per_trial_v1'}, law.domains)
        registry = FullTrialRegistry(root/'registry', plan)
        paths = registry.paths('finite', 0, oracle_index=0)
        runtime = ForwardRuntime(paths['runtime'], paths['feed'])
        trial = registry.identity_for('finite', 0, oracle_index=0)
        wallet = MonthlyWorkAuthority(paths['authority'],
            run_id=fingerprint({'registry': registry.digest, 'trial': trial['trial_id']}),
            task='volatility', universe=['BTCUSDT'], partition_policy='standard')
        permit = registry.enroll(runtime, wallet, 'finite', 0, oracle_index=0)
        opened.append(runtime)
        return gen, permit, runtime, wallet
    yield make
    for runtime in opened:
        runtime.close()


def ready(factory):
    gen, permit, runtime, wallet = factory()
    clock = Q.bind(runtime, gen, permit)
    finite.publish_due(gen, permit)
    return clock, gen, permit, runtime, wallet



def original_S_source(factory):
    gen, permit, runtime, wallet = factory(law_start='2025-12-31T22:00:00Z')
    journal = Q.bind(runtime, gen, permit)
    finite.publish_due(gen, permit, max_new_receipts=3)
    journal.advance_batch(runtime, max_events=2)
    prefix = observation.current_binding(runtime.lifecycle)
    cutoff = pd.Timestamp('2026-01-01T00:00:00Z')
    bars = observation.snapshot_for_prefix(runtime.lifecycle, prefix, cutoff, months=6, context_hours=1234)
    reference = {'feed_identity': runtime.feed.identity, 'feed_directory': str(runtime.feed.directory),
                 'exposure_path': str(runtime.lifecycle.exposure.path), 'cutoff': cutoff.isoformat(),
                 'observation_prefix': prefix}
    return journal, runtime, wallet, prefix, cutoff, bars, reference


def test_logical_S_source_accepts_only_original_frame_and_source(factory):
    from factorlab.discovery.v7_run import V7Config, _verified_S_receipt_source
    from factorlab.discovery.s_observation_source import validate_reference
    _, runtime, _, prefix, cutoff, bars, reference = original_S_source(factory)
    config = V7Config(symbols=('BTCUSDT',), evaluation_mode='joint')
    selected, identity = _verified_S_receipt_source(bars, cutoff, config, runtime.spec,
        runtime.lifecycle.exposure.path, runtime.feed.directory)
    assert identity == runtime.feed.identity
    assert selected.equals(bars)
    assert selected.iloc[0].received_at > selected.iloc[0].available_at
    assert validate_reference(reference) == prefix
    with pytest.raises(ValueError):
        validate_reference({**reference, 'feed_directory': str(runtime.directory)})
    with pytest.raises(ValueError):
        validate_reference({k: v for k, v in reference.items() if k != 'observation_prefix'})
    fabricated = bars.copy()
    fabricated.loc[0, 'received_at'] = fabricated.loc[0, 'available_at']
    with pytest.raises(ValueError, match='original observed law'):
        _verified_S_receipt_source(fabricated, cutoff, config, runtime.spec,
            runtime.lifecycle.exposure.path, runtime.feed.directory)


def test_S_preparation_binds_physical_boundary_without_changing_logical_cutoff(factory):
    from dataclasses import asdict
    from types import SimpleNamespace
    from factorlab.discovery.v7_run import V7Config
    from factorlab.discovery.s_preparation import SPreparationEnvelope
    from factorlab.discovery.s_observation_source import preparation_context, physical_cutoff, snapshot_from_context
    _, runtime, wallet, prefix, cutoff, bars, reference = original_S_source(factory)
    config = V7Config(symbols=('BTCUSDT',), evaluation_mode='joint')
    directory = runtime.directory/'original_S'
    prep = SPreparationEnvelope.prepare(wallet, directory, '2026-01', config=asdict(config),
        spec=asdict(runtime.spec), source_reference=reference, schedule_hash='a'*64)
    store = SimpleNamespace(directory=directory, exposure=runtime.lifecycle.exposure,
        identity={'feed_identity': runtime.feed.identity})
    bridge = {'authority_directory': str(wallet.directory), 'authority_manifest_hash': wallet.manifest_hash,
              'preparation_hash': prep.digest}
    binding = preparation_context(store, '2026-01', bridge)
    context = {**binding, 'cutoff': cutoff.isoformat(), 'source_id': runtime.feed.identity,
               'run_directory': str(directory), 'replay_month': '2026-01'}
    bound = physical_cutoff(context)
    assert bound == pd.Timestamp(prefix['physical_observation']['wall'])
    assert bound > cutoff
    assert snapshot_from_context(context, runtime.feed.directory).equals(bars)
    assert physical_cutoff({'cutoff': cutoff.isoformat()}) == cutoff
    with pytest.raises(ValueError, match='original source prefix'):
        physical_cutoff({**context, 'source_id': 'foreign'})
    with pytest.raises(ValueError, match='original source prefix'):
        physical_cutoff({**context, 'cutoff': '2026-02-01T00:00:00+00:00'})
