"""Finite synthetic generator/proof engineering, not provider or market evidence."""
import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from factorlab.discovery import finite_market_oracle as m
from factorlab.discovery.calibration_acceptance import AcceptanceCell, AcceptancePlan
from factorlab.protocol import fingerprint


@pytest.fixture
def generator(tmp_path):
    plan = AcceptancePlan((AcceptanceCell('finite', fingerprint('finite'), 1000, ('factor1',)),))
    return m.FiniteGenerator(tmp_path/'generator', m.FiniteMarketLaw('2026-01-01T00:00:00Z'), plan)


def test_registered_finite_population_reconstructs_exactly_with_persistence_and_regimes(generator):
    value = generator.produce_atom(0)
    assert generator.read_atom(0) == value
    assert len(value['rows']) == 72
    assert len({r['regime'] for r in value['schedule']}) == 2
    assert all(2**-16 <= r['close'] <= 2**16 for r in value['rows'])
    assert value['formal_gate_eligible'] is False
    assert generator.reconstruct_atom(0, population='design') != value
    assert len(set(generator.law.domains.values())) == 3
    assert generator.identity['proof']['loss_upper_bound'] == 2**20


def test_finite_support_shared_across_validation_and_oracle_not_separate_laws(generator):
    atom = generator.atom_for('validation', {'trial': 'one'})
    assert generator.produce_atom(atom) == generator.read_atom(atom)
    assert generator.identity['domains'] == generator.law.domains
    assert 'no_upper_clip' in generator.identity['proof']['decoder_audit']


def test_rehashed_market_atom_tamper_is_reconstructed_and_refused(generator):
    generator.produce_atom(0)
    path = generator.directory/'atoms/evaluation/0000.json'
    saved = json.loads(path.read_text()); saved['record']['rows'][0]['close'] *= 1.01
    saved['hash'] = fingerprint(saved['record']); path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='atom changed'):
        generator.read_atom(0)


def test_missing_identity_cannot_reset_existing_population(generator):
    p = generator.directory/'identity.json'
    p.rename(p.with_suffix('.retained'))
    with pytest.raises(ValueError, match='cannot reset'):
        m.FiniteGenerator(generator.directory, generator.law, generator.acceptance)


def test_original_loss_is_unclipped_and_bound_covers_extreme_finite_predictions():
    high = float(np.finfo(np.float64).max)
    for a, b in ((0., high), (high, 0.), (0., 0.), (.01, .1)):
        actual = float((np.log(a+1e-8)-np.log(b+1e-8))**2)
        assert m._finite_loss(a, b) == actual
        assert actual < m.LOSS_BOUND
    assert m._finite_loss(0., high) > 500000  # never clip loss to an observed scale
    for bad in (float('inf'), float('nan'), -1.):
        with pytest.raises(ValueError, match='Undefined predictions'):
            m._finite_loss(.01, bad)


def test_128_atom_census_numeric_enclosure_exact_rational_mean(generator):
    # Numeric proof kernel only; no claim these invented rows are actual pipeline outputs.
    rows = [{'atom': i, 'reference': .1+i/1000, 'candidate': .08+i/1200, 'defined': True}
            for i in range(128)]
    result = m._enclose_census(rows, generator=generator, oracle_digest=fingerprint(rows))
    for key in ('reference', 'candidate'):
        exact = sum((Fraction.from_float(r[key]) for r in rows), Fraction())/128
        low, high = getattr(result, key)
        assert Fraction.from_float(low) <= exact <= Fraction.from_float(high)
    assert result.method == 'finite_exhaustive_risk_enclosure'
    assert result.per_risk_tail_alpha == generator.acceptance.oracle_risk_alpha


def test_any_failed_atom_prevents_enclosure_no_survivor_truth(generator):
    rows = [{'atom': i, 'reference': 1., 'candidate': .8, 'defined': True} for i in range(128)]
    rows[-1].update(defined=False, reference=None, candidate=None)
    assert m._enclose_census(rows, generator=generator, oracle_digest=fingerprint(rows)) is None
    with pytest.raises(ValueError, match='Exact original census'):
        m._enclose_census(rows[:-1], generator=generator, oracle_digest=fingerprint(rows))
    rows[-1] = dict(rows[0])
    with pytest.raises(ValueError, match='Exact original census'):
        m._enclose_census(rows, generator=generator, oracle_digest=fingerprint(rows))


def test_two_path_engineering_grid_cannot_certify_128(generator, tmp_path):
    small = m.FiniteGenerator(tmp_path/'smoke', replace(generator.law, support_size=2), generator.acceptance)
    rows = [{'atom': i, 'reference': 1., 'candidate': .8, 'defined': True} for i in range(2)]
    assert small.identity['test_grid_only'] is True
    assert m._enclose_census(rows, generator=small, oracle_digest=fingerprint(rows)) is None


def test_changed_proof_source_or_grid_cannot_resume(generator):
    generator.identity['proof']['loss_upper_bound'] = 1.
    with pytest.raises(ValueError, match='proof/grid changed'):
        generator.verify()

@pytest.fixture
def registered(tmp_path, monkeypatch):
    """Real registry/wallet/source anchors with a private future clock only in tests."""
    from factorlab.discovery.calibration_runtime_contract import runtime_calibration_contract
    from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
    from factorlab.discovery.neural_forward_ticket import _bind_authority
    from factorlab.discovery.v7_forward import ForwardRuntime
    repo = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
    originals = {p: (repo/p).read_bytes() for p in m.PROOF_FILES}
    from test_v7_run_factor_pool import source_root
    source_root(tmp_path, monkeypatch)
    for name, body in originals.items():
        path = tmp_path/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(body)
    from factorlab.discovery import full_calibration_registry as reg
    original_scope = ForwardRuntime(tmp_path/'scope', tmp_path/'scope-feed')
    wallet = MonthlyWorkAuthority(tmp_path/'scope-wallet', run_id='scope', task='volatility',
                                   universe=['BTCUSDT'], partition_policy='standard')
    _bind_authority(original_scope, wallet)
    scope = runtime_calibration_contract(original_scope.lifecycle); original_scope.close()
    law = m.FiniteMarketLaw((m._now().floor('h') + m.pd.Timedelta(hours=1)).isoformat(),
                            hours=24, gap_probability=0.)
    acceptance = AcceptancePlan((AcceptanceCell('finite', law.digest, 1000),))
    gen = m.FiniteGenerator(tmp_path/'generator', law, acceptance)
    plan = reg.FullTrialPlan(acceptance, scope,
        {'backend': 'codex', 'model_requested': 'gpt-6-astra', 'reasoning_requested': 'high',
         'context_policy': 'fresh_codex_per_trial_v1'}, law.domains)
    registry = reg.FullTrialRegistry(tmp_path/'registry', plan)
    paths = registry.paths('finite', 0, oracle_index=0)
    runtime = ForwardRuntime(paths['runtime'], paths['feed'])
    trial = registry.identity_for('finite', 0, oracle_index=0)
    wallet = MonthlyWorkAuthority(paths['authority'],
        run_id=fingerprint({'registry': registry.digest, 'trial': trial['trial_id']}), task='volatility',
        universe=['BTCUSDT'], partition_policy='standard')
    permit = registry.enroll(runtime, wallet, 'finite', 0, oracle_index=0)
    clock = [m.pd.Timestamp(law.start)]
    monkeypatch.setattr(m, '_now', lambda: clock[0])
    try:
        yield gen, registry, permit, runtime, wallet, clock
    finally:
        runtime.close()


def test_actual_receipt_publish_has_original_current_time_and_no_historical_asof(registered):
    from factorlab.discovery.v7_feed import SnapshotUnavailable
    gen, _, permit, runtime, wallet, clock = registered
    original_budget = wallet.verify()
    pending = m.publish_due(gen, permit)
    assert pending['issued'] == []
    clock[0] += m.pd.Timedelta(hours=1, seconds=5)
    result = m.publish_due(gen, permit)
    assert len(result['issued']) == 1 and result['issued'][0]['received_at'] == clock[0].isoformat()
    check = m.verify_publications(gen, permit)
    assert check['completed_hours'] == [0] and not check['all_registered_receipts_complete']
    runtime.feed.clock = lambda: clock[0]
    with pytest.raises(SnapshotUnavailable):
        runtime.feed.snapshot(clock[0]-m.pd.Timedelta(seconds=1))
    snapshot = runtime.feed.snapshot(clock[0])
    assert len(snapshot) == 1
    assert snapshot.iloc[0].received_at == clock[0]
    assert m.publish_due(gen, permit)['issued'] == []
    assert wallet.verify() == original_budget


def test_publication_failure_consumed_without_retry_or_refreshed_receipt(registered, monkeypatch):
    gen, _, permit, runtime, _, clock = registered
    clock[0] += m.pd.Timedelta(hours=1, seconds=5)
    write = m._write
    def failed(path, value):
        if path.name.endswith('-complete.json'):
            raise OSError('original completion publication interrupted')
        return write(path, value)
    monkeypatch.setattr(m, '_write', failed)
    with pytest.raises(OSError):
        m.publish_due(gen, permit)
    monkeypatch.setattr(m, '_write', write)
    receipts = runtime.feed.db.execute('SELECT id FROM batches').fetchall()
    clock[0] += m.pd.Timedelta(seconds=2)
    with pytest.raises(ValueError, match='interrupted; cannot renew'):
        m.publish_due(gen, permit)
    assert runtime.feed.db.execute('SELECT id FROM batches').fetchall() == receipts
    assert m.verify_publications(gen, permit)['completed_hours'] == []


def test_missing_publication_attempt_anchor_refused_without_feed_reset(registered):
    gen, _, permit, runtime, _, clock = registered
    clock[0] += m.pd.Timedelta(hours=1, seconds=5)
    m.publish_due(gen, permit)
    path = permit.paths['root']/'generator_publications/00000-attempt.json'
    path.rename(path.with_suffix('.retained'))
    with pytest.raises(ValueError, match='anchors missing'):
        m.verify_publications(gen, permit)
    assert runtime.feed.db.execute('SELECT COUNT(*) FROM batches').fetchone()[0] == 1


def test_missing_census_produces_no_envelope_and_artifact_is_reconstructed(registered, monkeypatch):
    gen, registry, _, _, _, _ = registered
    # All 128 actual paths are absent. The canonical oracle module is loaded,
    # but no original loss input or completion is invented.
    result = m.produce_finite_envelopes(gen, registry, 'finite', 0)
    assert len(result['record']['missing_atoms']) == 128
    assert all(v is None for v in result['record']['risk_envelopes'].values())
    assert result['record']['algorithm']['risk_envelope'] is None
    assert m.verify_finite_envelopes(gen, registry, 'finite', 0,
                                     artifact_digest=result['digest']) == result['record']
    target = gen.directory/'envelopes'/f"{result['digest']}.json"
    saved = json.loads(target.read_text()); saved['record']['formal_gate_eligible'] = True
    saved['hash'] = fingerprint(saved['record']); target.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='original census reconstruction'):
        m.verify_finite_envelopes(gen, registry, 'finite', 0, artifact_digest=result['digest'])


def test_runtime_version_change_cannot_silently_change_finite_population(generator, monkeypatch):
    original = m.runtime_identity
    monkeypatch.setattr(m, 'runtime_identity', lambda: {**original(), 'python': [0, 0, 0]})
    with pytest.raises(ValueError, match='generator/source/proof/grid changed'):
        generator.verify()
