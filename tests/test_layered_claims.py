"""Prospective endpoint and selector bookkeeping; only artificial clock/losses."""
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path('D:/FactorResearch')
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parents[1]/'factorlab/discovery'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_legacy_oracle_policy_import_remains_available():
    """Existing oracle imports retain the original policy, not the new variant."""
    from factorlab.discovery import calibration_oracle_evidence, claim_protocol
    from factorlab.discovery import typed_policy_confirmation

    assert claim_protocol.POLICY is typed_policy_confirmation.POLICY
    assert calibration_oracle_evidence.POLICY is claim_protocol.POLICY
    assert claim_protocol.POLICY['version'] == 'v7-adaptive-algorithm-v2'


@pytest.fixture
def layered(tmp_path, monkeypatch):
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    (tmp_path/'factorlab').mkdir()
    (tmp_path/'factorlab/fixture.py').write_text('SYNTHETIC_SOURCE=True\n')
    (tmp_path/'pyproject.toml').write_text('[project]\n')
    (tmp_path/'uv.lock').write_text('version=1\n')
    policy = load('factorlab.discovery._layered_claims_policy_stage', HERE/'layered_research.py')
    monkeypatch.setitem(sys.modules, 'factorlab.discovery.layered_research', policy)
    typed = load('factorlab.discovery._layered_claims_typed_stage', HERE/'typed_policy_confirmation.py')
    monkeypatch.setitem(sys.modules, 'factorlab.discovery.typed_policy_confirmation', typed)
    claims = load('factorlab.discovery._layered_claims_protocol_stage', HERE/'claim_protocol.py')
    monkeypatch.setitem(sys.modules, 'factorlab.discovery.claim_protocol', claims)
    forward = load('factorlab.discovery._layered_claims_forward_stage', HERE/'v7_forward.py')
    monkeypatch.setitem(sys.modules, 'factorlab.discovery.v7_forward', forward)
    clock = [pd.Timestamp('2026-10-10T16:00:00Z')]
    spec = policy.LayeredResearchSpec(policy.fingerprint(policy.source_files()))
    runtime = forward.ForwardRuntime(tmp_path/'runtime', tmp_path/'feed',
        config=forward.ForwardConfig(training_window_days=90, refit_interval_days=15),
        research_spec=spec, clock=lambda: clock[0])
    yield runtime, clock, policy, typed, claims
    runtime.close()


def endpoints(runtime, claims, spec_hash):
    from factorlab.discovery.known_linear_reference import FrozenLinearRecipe
    k0 = runtime.recipe_for()
    baseline = runtime.recipe_for(FrozenLinearRecipe('HAR_cal', runtime.spec))
    return [claims.FrozenClaimEndpoint.library('2026-11-01T00:00:00Z', k0, baseline),
            claims.FrozenClaimEndpoint.algorithm('2026-11-01T00:00:00Z', k0, research_spec_hash=spec_hash)]


def test_new_policy_hash_is_frozen_and_old_endpoint_cannot_be_enrolled(layered):
    runtime, _, _, typed, claims = layered
    wrong = endpoints(runtime, claims, None)
    with pytest.raises(ValueError, match='enrollment'):
        claims.ClaimProtocol.freeze(runtime.lifecycle, wrong)
    assert not (runtime.directory/'claim_protocol').exists()
    correct = endpoints(runtime, claims, runtime.lifecycle.research_spec.digest)
    algorithm = correct[1]
    assert algorithm.candidate['version'] == 'v7-layered-elapsed-research-policy-v4'
    assert algorithm.candidate['statistical_status'] == 'descriptive_one_endpoint_no_population_power_certificate'
    assert claims.FrozenClaimEndpoint.from_dict(algorithm.to_dict()).digest == algorithm.digest
    protocol = claims.ClaimProtocol.freeze(runtime.lifecycle, correct)
    assert typed._protocol(runtime.lifecycle).digest == protocol.digest
    assert protocol.verify()['formal_eligible'] is False
    changed = algorithm.to_dict()
    changed['research_spec_hash'] = 'b'*64
    with pytest.raises(ValueError):
        claims.FrozenClaimEndpoint.from_dict(changed)


def test_six_month_finalization_retains_one_look_and_no_whole_agent_certificate(layered, monkeypatch):
    runtime, clock, _, _, claims = layered
    protocol = claims.ClaimProtocol.freeze(runtime.lifecycle,
        endpoints(runtime, claims, runtime.lifecycle.research_spec.digest))
    assert protocol.finalize('algorithm')['status'] == 'pending'
    clock[0] = pd.Timestamp('2027-05-02T12:00:00Z')
    def losses(endpoint, stream, pipeline):
        value = 1. if stream == endpoint.reference_stream else .98
        return {(hour.isoformat(), 'BTCUSDT'): (value, .01, 'synthetic-fixed-revision')
                for hour in pd.date_range(endpoint.start, endpoint.end, inclusive='left', freq='h')}
    monkeypatch.setattr(protocol, '_losses', losses)
    score = protocol.finalize('algorithm')
    assert score['formal_eligible'] is False
    assert score['result']['supported'] is False
    assert score['result']['diagnostic_status'] == 'supported_target'
    monkeypatch.setattr(protocol, '_losses', lambda *a, **k: pytest.fail('No second endpoint read'))
    assert protocol.finalize('algorithm') == score
    assert protocol.verify()['formal_eligible'] is False


def test_readonly_selection_preserves_research_identity_and_never_initializes_new_files(layered):
    runtime, clock, policy, typed, claims = layered
    lc = runtime.lifecycle
    lc.due_fit_cutoff()
    claims.ClaimProtocol.freeze(lc, endpoints(runtime, claims, lc.research_spec.digest))
    clock[0] = pd.Timestamp('2026-11-09T16:00:00Z')
    cutoff = lc.due_fit_cutoff()
    # No typed catalogue or accepted factor: original K0 is still selected.
    from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
    from factorlab.discovery.neural_forward_ticket import _bind_authority
    wallet = MonthlyWorkAuthority(lc.directory.parent/'wallet', run_id='synthetic-layered-selector',
        task='volatility', universe=['BTCUSDT'], partition_policy='standard')
    _bind_authority(runtime, wallet)
    selected = typed.candidate_pipeline(lc, cutoff, runtime.recipe_for())
    assert selected == runtime.recipe_for()
    paths_before = {str(p.relative_to(lc.directory)) for p in lc.directory.rglob('*') if p.is_file()}
    with typed._original_reader(lc) as reader:
        assert policy.verified_research_spec(reader).digest == lc.research_spec.digest
        assert typed._protocol(reader).endpoints['algorithm'].research_spec_hash == lc.research_spec.digest
    assert {str(p.relative_to(lc.directory)) for p in lc.directory.rglob('*') if p.is_file()} == paths_before
    assert lc.adoption_pipeline(cutoff, runtime.recipe_for()) == runtime.recipe_for()
    assert not lc.calibration_ready()


def test_measured_factor_reaches_typed_selector_and_operational_binding_without_certificate(layered, monkeypatch):
    runtime, clock, policy, typed, claims = layered
    lc = runtime.lifecycle
    lc.due_fit_cutoff()
    claims.ClaimProtocol.freeze(lc, endpoints(runtime, claims, lc.research_spec.digest))
    from factorlab.discovery.joint_pool import FrozenPool
    from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
    from factorlab.discovery.neural_forward_ticket import _bind_authority
    wallet = MonthlyWorkAuthority(lc.directory.parent/'wallet', run_id='synthetic-measured-selector',
        task='volatility', universe=['BTCUSDT'], partition_policy='standard')
    _bind_authority(runtime, wallet)
    candidate = runtime.recipe_for(FrozenPool(version='synthetic_candidate'))
    lc.exposure.record(source=lc.plan.source_id, symbols=lc.plan.symbols,
        start='2026-07-01T00:00:00Z', end='2026-10-01T00:00:00Z', role='S', batch='A')
    lc.freeze_batch(batch='A', ordinal=1, start='2026-11-01T00:00:00Z',
        candidates=[{**candidate, 'complexity': 3}], reference=runtime.recipe_for(),
        evidence_sources=[{'start':'2026-07-01T00:00:00Z','end':'2026-10-01T00:00:00Z'}])
    def losses(stream, start, end, as_of, *, expected_pipeline=None):
        value = 1. if stream.endswith(':reference') else .98
        return {(hour, 'BTCUSDT'): (value, .01, 'fixed-synthetic-revision')
                for hour in pd.date_range(start, end, inclusive='left', freq='h')}
    monkeypatch.setattr(lc, '_losses', losses)
    clock[0] = pd.Timestamp('2026-12-01T00:00:00Z')
    lc.select_validation('A'); lc.reserve_selected_confirmation('A')
    clock[0] = pd.Timestamp('2027-03-02T12:00:00Z')
    result = lc.conclude_confirmation('A')
    assert policy.research_retention_decision(lc, result)['formal_gate_eligible'] is False
    clock[0] = pd.Timestamp('2027-03-20T12:00:00Z')
    cutoff = lc.due_fit_cutoff()
    assert lc.adoption_pipeline(cutoff, runtime.recipe_for()) == candidate
    selected = typed._read_selection(lc, cutoff, runtime.recipe_for())
    assert selected['selection']['kind'] == 'unconfirmed_research_factor_C'
    bound = typed.canonical_adoption(lc, cutoff, candidate)
    assert bound['version'] == 'v7-unconfirmed-layered-policy-adoption-v1'
    assert bound['whole_agent_power_certified'] is False
    assert bound['formal_gate_eligible'] is False
    assert typed.canonical_adoption(lc, cutoff, candidate, expected=bound, historical=True) == bound
    with pytest.raises(ValueError, match='binding'):
        typed.canonical_adoption(lc, cutoff, candidate,
            expected={**bound, 'whole_agent_power_certified':True}, historical=True)
    # Exercise the existing convex-fit admission context; a selector alone
    # must not leave the real producer rejecting the new research binding.
    from factorlab.discovery.pool_forward import PoolForwardRecipe, _context
    recipe = PoolForwardRecipe(FrozenPool(version='synthetic_candidate'), runtime.spec)
    job = lc.plan_fit(cutoff, 'A0', candidate)
    context = _context(lc, job, recipe)
    assert context['adoption']['version'] == 'v7-unconfirmed-layered-policy-adoption-v1'
    assert context['adoption']['formal_gate_eligible'] is False


def test_CLI_status_and_claim_freeze_propagate_new_scope(layered, monkeypatch, capsys):
    runtime, _, _, _, _ = layered
    cli = load('factorlab.discovery._layered_cli_stage', HERE/'v7_cli.py')
    from dataclasses import asdict
    spec_path = runtime.directory.parent/'spec.json'
    spec_path.write_text(json.dumps(asdict(runtime.lifecycle.research_spec)), encoding='utf-8')
    base = ['--run', str(runtime.directory), '--feed', str(runtime.feed.directory),
            '--symbol','BTCUSDT','--window-days','90','--research-spec',str(spec_path)]
    assert cli.main(['forward-status', *base]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status['statistical_mode'] == 'prospective_uncertified_research'
    assert status['whole_agent_power_certified'] is False
    assert cli.main(['claims-freeze', *base, '--start','2026-11-01T00:00:00Z']) == 0
    value = json.loads(capsys.readouterr().out)
    assert value['formal_claims_enabled'] is False
    algorithm = next(e for e in value['endpoints'] if e['slot']=='algorithm')
    assert algorithm['research_spec_hash'] == runtime.lifecycle.research_spec.digest
