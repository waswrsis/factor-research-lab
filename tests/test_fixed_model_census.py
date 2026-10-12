"""Actual sealed convex prediction census; two atoms are diagnostic only."""
import json

import pandas as pd
import pytest

from factorlab.discovery import finite_market_oracle as finite
from factorlab.discovery import fixed_model_census as m
from factorlab.discovery.calibration_acceptance import AcceptanceCell, AcceptancePlan
from factorlab.discovery.joint_pool import FrozenPool, fit_pool
from factorlab.discovery.local_mutation import MonthlyBudget
from factorlab.discovery.rolling import RollingSpec, build_rolling_data
from factorlab.discovery.v7_forward import ForwardRuntime
from factorlab.discovery.v7_lifecycle import Lifecycle, LifecyclePlan
from factorlab.protocol import fingerprint


@pytest.fixture
def actual(tmp_path):
    law = finite.FiniteMarketLaw('2025-12-01T00:00:00Z', hours=1300, support_size=2, gap_probability=0.)
    plan = AcceptancePlan((AcceptanceCell('finite', law.digest, 1000),))
    generator = finite.FiniteGenerator(tmp_path/'generator', law, plan)
    # Genuine offline design-domain data and cold convex fit. This is not a
    # protected production fit or a wallet admission fixture.
    atom = generator.produce_atom(0, population='design')
    frame = pd.DataFrame(atom['rows'])
    frame['open_time'] = pd.to_datetime(frame.open_time, utc=True)
    frame['available_at'] = frame.open_time+pd.Timedelta(hours=1, seconds=5)
    spec = RollingSpec(availability_delay_seconds=5)
    data = build_rolling_data(frame, spec)
    budget = MonthlyBudget('2026-02')
    lc = Lifecycle(tmp_path/'original', LifecyclePlan('offline-artificial-design'),
                   clock=lambda: pd.Timestamp('2026-02-01T00:00:00Z'))
    descriptor = ForwardRuntime.__new__(ForwardRuntime); descriptor.spec = spec
    identifiers = []
    for i in range(2):
        pool = FrozenPool(role='reference', version='fixed-census-'+str(i))
        model = fit_pool(data, '2026-02-01T00:00:00Z', pool, budget=budget)
        pipeline = descriptor.recipe_for(pool)
        job = lc.plan_fit('2026-02-01T00:00:00Z', 'model'+str(i), pipeline)
        lc.advance_fit(job, 'snapshot_sealed', output='artificial-design-frozen')
        lc.advance_fit(job, 'features_ready', output='actual-finite-library')
        identifiers.append(lc.seal_model(job, model))
    try:
        yield generator, lc, identifiers, budget
    finally:
        lc.close()


def test_actual_sealed_models_generate_two_real_atoms_but_no_128_envelope(actual, tmp_path):
    generator, lc, ids, budget = actual
    census = m.FixedModelCensus(tmp_path/'census', generator, lc.directory, *ids)
    progress = census.run(max_new_atoms=2)
    assert set(progress['states'].values()) == {'complete'}
    report = census.report()
    assert report['completed_atoms'] == 2 and report['missing_atoms'] == 0
    assert report['undefined_or_failed_atoms'] == []
    assert report['registered_atoms'] == 2 and report['risk_envelope'] is None
    assert report['original_cold_refit_claim_scope_eligible'] is False
    with pytest.raises(ValueError, match='cannot replace original cold-refit'):
        census.original_claim_envelope()
    before = budget.to_dict()
    assert census.run(max_new_atoms=2)['new_atoms'] == 0
    assert census.report() == report
    assert budget.to_dict() == before
    # Every result came from actual source-bound predictors, not injected losses.
    result = finite._read(census.directory/'units/0/result.json')
    assert result['actual_pairs'] == result['expected_pairs'] == 52
    assert result['reference'] >= 0 and result['candidate'] >= 0


def test_rehashed_conditional_loss_cannot_replace_original_model_replay(actual, tmp_path):
    generator, lc, ids, _ = actual
    census = m.FixedModelCensus(tmp_path/'census', generator, lc.directory, *ids)
    census.run()
    path = census.directory/'units/0/result.json'
    record = finite._read(path); record['candidate'] = 0.
    path.write_text(json.dumps({'record': record, 'hash': fingerprint(record)}))
    import sqlite3
    with sqlite3.connect(census.directory/'units.sqlite') as db:
        db.execute('UPDATE units SET record=?,hash=? WHERE atom=0',
                    (json.dumps(record, sort_keys=True), fingerprint(record)))
    with pytest.raises(ValueError, match='actual sealed-model prediction replay'):
        census.report()


def test_model_census_journal_loss_cannot_reissue_inference(actual, tmp_path):
    generator, lc, ids, _ = actual
    census = m.FixedModelCensus(tmp_path/'census', generator, lc.directory, *ids)
    census.run()
    path = census.directory/'units.sqlite'; path.rename(path.with_suffix('.retained'))
    with pytest.raises(ValueError, match='journal missing'):
        m.FixedModelCensus(census.directory, generator, lc.directory, *ids)
    assert not path.exists()


def test_original_undefined_atom_is_not_retried_under_another_call(actual, tmp_path, monkeypatch):
    generator, lc, ids, _ = actual
    census = m.FixedModelCensus(tmp_path/'census', generator, lc.directory, *ids)
    def fail(*args):
        raise ValueError('injected numerical failure')
    monkeypatch.setattr(m, '_atom_risk', fail)
    first = census.run(max_new_atoms=1)
    assert first['states'] == {'0': 'failed_consumed'}
    second = census.run(max_new_atoms=1)
    assert second['states'] == {'0': 'failed_consumed', '1': 'failed_consumed'}
    report = census.report()
    assert report['undefined_or_failed_atoms'] == [0, 1] and report['risk_envelope'] is None


def test_gapped_population_preserves_undefined_atoms_in_denominator(actual, tmp_path):
    from dataclasses import replace
    generator, lc, ids, _ = actual
    gapped = finite.FiniteGenerator(tmp_path/'gapped', replace(generator.law, gap_probability=.1), generator.acceptance)
    census = m.FixedModelCensus(tmp_path/'gapped-census', gapped, lc.directory, *ids)
    census.run(max_new_atoms=2)
    report = census.report()
    assert report['registered_atoms'] == 2 and report['missing_atoms'] == 0
    assert report['undefined_or_failed_atoms'] == [0, 1]
    assert report['risk_envelope'] is None
