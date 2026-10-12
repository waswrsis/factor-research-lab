"""Artificial actual-ledger oracle engineering; not an iid/market certificate."""
import json

import numpy as np
import pandas as pd
import pytest

from factorlab.discovery import calibration_oracle_evidence as m
from factorlab.discovery.convex_eval import fit_ridge
from factorlab.discovery.v7_lifecycle import Lifecycle, LifecyclePlan
from factorlab.protocol import fingerprint


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    now = [pd.Timestamp('2026-01-01T00:00:00Z')]
    lc = Lifecycle(tmp_path / 'lc', LifecyclePlan('artificial-feed', symbols=('BTCUSDT', 'ETHUSDT')),
                   clock=lambda: now[0])
    recipes = {s: {'pipeline_id': fingerprint(s), 'artificial_learner': s} for s in ('base', 'candidate')}
    model = fit_ridge(np.arange(60).reshape(-1, 1), np.linspace(.001, .03, 60))
    models = {}
    for stream, recipe in recipes.items():
        job = lc.plan_fit(now[0], stream, recipe)
        lc.advance_fit(job, 'snapshot_sealed', output='artificial-data')
        lc.advance_fit(job, 'features_ready', output='artificial-features')
        models[stream] = lc.seal_model(job, model)
        lc.activate(job, now[0])
    # Unequal symbol counts per hour exposes accidental row-weighting.
    for hour, symbols in [(0, ('BTCUSDT', 'ETHUSDT')), (1, ('BTCUSDT',))]:
        now[0] = pd.Timestamp('2026-01-01T00:00:01Z') + pd.Timedelta(hours=hour)
        for symbol in symbols:
            for stream in recipes:
                value = .1 if stream == 'base' else (.12 if hour == 0 else .2)
                lc.record_forecast(stream=stream, origin=now[0], symbol=symbol, task='volatility',
                    prediction=value, received_at=now[0], feature_open_time=now[0].floor('h')-pd.Timedelta(hours=1),
                    forecast_window_start=now[0].ceil('h'), model_id=models[stream],
                    pipeline_id=recipes[stream]['pipeline_id'])
    now[0] = pd.Timestamp('2026-01-02T00:00:00Z')
    for row in lc.forecasts.iter_records():
        lc.settle(stream=row['stream'], origin=row['origin'], symbol=row['symbol'], task='volatility',
            label=.13, maturity=pd.Timestamp(row['origin']).ceil('h')+pd.Timedelta(hours=12), received_at=now[0])
    spec = m.OracleClaimSpec('factor1', '2026-01-01T00:00:00Z', '2026-01-01T02:00:00Z',
                              'base', 'candidate', recipes['base'], recipes['candidate'])
    yield lc, spec
    lc.close()


def test_reconstructs_real_receipts_equal_hour_and_nominal_missingness(case):
    lc, spec = case
    result = m.reconstruct_claim(lc.directory, spec)
    a = (np.log(.13+1e-8)-np.log(.12+1e-8))**2
    b = (np.log(.13+1e-8)-np.log(.2+1e-8))**2
    assert result['path_risks']['candidate'] == pytest.approx((a+b)/2)
    assert result['path_risks']['candidate'] != pytest.approx((2*a+b)/3)
    assert result['hour_support'] == 2 and result['pair_support'] == 3
    assert result['coverage'] == 1 and result['pair_coverage'] == .75
    assert result['status'] == 'incomplete_support'
    assert result['truth'] == 'uncertain' and result['formal_gate_eligible'] is False


def test_read_only_inspection_does_not_repair_missing_forecast_index(case):
    lc, spec = case
    with lc.forecasts.db:
        lc.forecasts.db.execute('DELETE FROM forecasts WHERE stream=?', ('candidate',))
    with pytest.raises(ValueError, match='missing from logical ledger'):
        m.reconstruct_claim(lc.directory, spec)
    assert lc.forecasts.db.execute('SELECT COUNT(*) FROM forecasts').fetchone()[0] == 3


def test_original_receipt_file_tamper_refused(case):
    lc, spec = case
    p = next(lc.forecasts.records_directory.glob('*.json'))
    saved = json.loads(p.read_text())
    saved['receipt']['prediction']['value'] = .17
    p.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        m.reconstruct_claim(lc.directory, spec)


def test_label_revision_mismatch_not_dropped(case):
    lc, spec = case
    row = lc.db.execute('SELECT rowid,record FROM settlement WHERE stream=? LIMIT 1', ('candidate',)).fetchone()
    value = json.loads(row[1]); value['revision'] = 'different-revision'
    with lc.db:
        lc.db.execute('UPDATE settlement SET record=?,hash=? WHERE rowid=?',
                      (json.dumps(value), fingerprint(value), row[0]))
    with pytest.raises(ValueError, match='different actual labels'):
        m.reconstruct_claim(lc.directory, spec)


def test_fixed_pipeline_rebinding_refused(case):
    lc, spec = case
    other = m.OracleClaimSpec(spec.slot, spec.start, spec.end, spec.reference_stream,
        spec.candidate_stream, spec.reference, {**spec.candidate, 'artificial_learner': 'changed'})
    with pytest.raises(ValueError, match='fixed learner changed'):
        m.reconstruct_claim(lc.directory, other)


def test_nonselection_not_replaced_by_reference(case):
    lc, spec = case
    absent = m.OracleClaimSpec(spec.slot, spec.start, spec.end, spec.reference_stream,
                              spec.candidate_stream, spec.reference, None)
    result = m.reconstruct_claim(lc.directory, absent)
    assert result['status'] == 'not_selected' and result['path_risks'] is None
    assert result['truth'] == 'uncertain'


def test_algorithm_cannot_mean_last_fitted_winner(case):
    _, spec = case
    with pytest.raises(ValueError, match='entire original A0'):
        m.OracleClaimSpec('algorithm', spec.start, spec.end, 'base', 'A0', spec.reference, spec.candidate)


def test_mutated_frozen_estimand_is_refused(case):
    lc, spec = case
    spec.reference['changed'] = True
    with pytest.raises(ValueError, match='estimand mutated'):
        m.reconstruct_claim(lc.directory, spec)


def test_missing_lifecycle_database_is_not_recreated(case):
    lc, spec = case
    directory = lc.directory / 'missing'
    directory.mkdir()
    (directory/'lifecycle.json').write_bytes((lc.directory/'lifecycle.json').read_bytes())
    with pytest.raises(ValueError, match='Missing original oracle ledger'):
        m.reconstruct_claim(directory, spec)
    assert not (directory/'lifecycle.sqlite').exists()

@pytest.fixture
def raw_case(tmp_path, monkeypatch):
    from factorlab.discovery.v7_feed import ReceiptArchive
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    now = [pd.Timestamp('2026-01-01T13:00:05Z')]
    feed = ReceiptArchive(tmp_path/'feed', clock=lambda: now[0])
    times = pd.date_range('2026-01-01T00:00:00Z', periods=13, freq='h')
    prices = np.exp(np.linspace(4., 4.12, 13))
    frame = pd.DataFrame({'symbol': 'BTCUSDT', 'open_time': times,
        'open': prices, 'close': prices, 'high': prices+1, 'low': prices-1, 'volume': 10.})
    feed.receive(frame, received_at=now[0], response_hash='a'*64)
    snap = feed.snapshot(now[0])
    row = {'symbol': 'BTCUSDT', 'window': '2026-01-01T01:00:00+00:00',
        'received_at': now[0].isoformat(), 'maturity': now[0].isoformat(),
        'label': float(np.sqrt(np.square(np.diff(np.log(prices))).sum())),
        'revision': fingerprint(snap.revision.tolist())}
    yield feed, now, frame, row
    feed.close()


def test_raw_original_13_candle_label_reconstruction(raw_case):
    feed, _, _, row = raw_case
    assert m._raw_labels(feed.directory, [row], source_id=feed.identity, symbols=['BTCUSDT'], delay=5) is None


def test_later_true_revision_does_not_change_old_label(raw_case):
    feed, now, frame, row = raw_case
    now[0] += pd.Timedelta(hours=1)
    frame.loc[12, ['open', 'high', 'low', 'close']] *= 1.01
    feed.receive(frame, received_at=now[0], response_hash='b'*64)
    m._raw_labels(feed.directory, [row], source_id=feed.identity, symbols=['BTCUSDT'], delay=5)


def test_consistently_rehashed_false_settlement_still_fails_raw_truth(raw_case):
    feed, _, _, row = raw_case
    with pytest.raises(ValueError, match='differs from original production raw'):
        m._raw_labels(feed.directory, [{**row, 'label': row['label']*1.02}],
                       source_id=feed.identity, symbols=['BTCUSDT'], delay=5)


def test_missing_raw_index_not_recreated(raw_case):
    feed, _, _, row = raw_case
    target = feed.directory/'unavailable'
    target.mkdir()
    (target/'identity.json').write_bytes((feed.directory/'identity.json').read_bytes())
    with pytest.raises(ValueError, match='Missing original raw feed index'):
        m._raw_labels(target, [row], source_id=feed.identity, symbols=['BTCUSDT'], delay=5)
    assert not (target/'receipts.sqlite').exists()

@pytest.fixture
def registry_case(tmp_path, monkeypatch):
    """Actual frozen registry, runtime, source, wallet and original enrollment.

    No provider, market generator, simulated activation or training is executed.
    """
    from test_v7_run_factor_pool import source_root

    from factorlab.discovery.calibration_acceptance import AcceptanceCell, AcceptancePlan
    from factorlab.discovery.calibration_runtime_contract import runtime_calibration_contract
    from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
    from factorlab.discovery.neural_forward_ticket import _bind_authority
    from factorlab.discovery.v7_forward import ForwardRuntime
    source_root(tmp_path, monkeypatch)
    from factorlab.discovery import full_calibration_registry as registry_module
    baseline = ForwardRuntime(tmp_path/'profile-runtime', tmp_path/'profile-feed')
    wallet = MonthlyWorkAuthority(tmp_path/'profile-wallet', run_id='profile', task='volatility',
                                   universe=['BTCUSDT'], partition_policy='standard')
    _bind_authority(baseline, wallet)
    scope = runtime_calibration_contract(baseline.lifecycle)
    baseline.close()
    plan = registry_module.FullTrialPlan(
        AcceptancePlan((AcceptanceCell('null', fingerprint('null'), 1000),)), scope,
        {'backend': 'codex', 'model_requested': 'gpt-6-astra', 'reasoning_requested': 'high',
         'context_policy': 'fresh_codex_per_trial_v1'},
        {'design_digest': fingerprint('design'), 'validation_digest': fingerprint('validation'),
         'oracle_digest': fingerprint('oracle')})
    registry = registry_module.FullTrialRegistry(tmp_path/'registry', plan)
    opened = []
    def enroll(oracle_index=None):
        paths = registry.paths('null', 0, oracle_index=oracle_index)
        runtime = ForwardRuntime(paths['runtime'], paths['feed'])
        opened.append(runtime)
        trial = registry.identity_for('null', 0, oracle_index=oracle_index)
        wallet = MonthlyWorkAuthority(paths['authority'],
            run_id=fingerprint({'registry': registry.digest, 'trial': trial['trial_id']}),
            task='volatility', universe=['BTCUSDT'], partition_policy='standard')
        return registry.enroll(runtime, wallet, 'null', 0, oracle_index=oracle_index), runtime, wallet
    try:
        yield registry_module, registry, enroll
    finally:
        for runtime in opened:
            runtime.close()


def test_typed_real_permit_rejects_incomplete_original_trial_without_artifact(registry_case):
    _, _, enroll = registry_case
    permit, runtime, wallet = enroll()
    before = wallet.verify()
    with pytest.raises((ValueError, FileNotFoundError)):
        m.freeze_oracle_estimands(permit)
    assert wallet.verify() == before
    assert not (runtime.directory.parent/'oracle_estimands.json').exists()
    with pytest.raises(ValueError, match='wrong registered trial domain'):
        m.produce_oracle_path(permit)
    with pytest.raises(TypeError, match='Actual preregistered'):
        m.produce_oracle_path({'trial': permit.trial})


def _engineering_specs(runtime):
    """Only a private original-selection seam; NOT real completed validation."""
    reference = runtime.recipe_for()
    start, end = '2026-01-01T00:00:00Z', '2026-01-01T02:00:00Z'
    return [m.OracleClaimSpec('library_skill', start, end, 'benchmark', 'K0',
                 {'pipeline_id': fingerprint('benchmark'), 'kind': 'artificial'}, reference),
            *[m.OracleClaimSpec('factor'+str(i), start, end, 'reference'+str(i), 'absent'+str(i), reference, None)
              for i in range(1, 4)],
            m.OracleClaimSpec('algorithm', start, end, 'K0', m.POLICY.get('evaluation_stream', 'A0'), reference,
                               {**m.POLICY, 'initial_pipeline': reference})]


def test_actual_registry_empty_path_artifact_readback_no_truth_or_fee(registry_case, monkeypatch):
    _, registry, enroll = registry_case
    validation, runtime, _ = enroll()
    specs = _engineering_specs(runtime)
    # The only seam bypasses original completed-selection extraction. Actual
    # permit/dual anchors/source/SQLite/forecast/feed checks are untouched.
    monkeypatch.setattr(m, '_original_specs', lambda _: specs)
    frozen = m.freeze_oracle_estimands(validation)
    oracle, runtime, wallet = enroll(0)
    before = wallet.verify()
    value = m.produce_oracle_path(oracle)
    assert m.read_oracle_path(oracle) == value
    assert value['estimand_hash'] == fingerprint(frozen)
    assert tuple(c['estimand']['slot'] for c in value['claims']) == m.CLAIMS
    assert value['risk_envelope'] is None and value['truth'] == 'uncertain'
    assert not value['formal_gate_eligible'] and not value['full_pipeline_execution_verified']
    assert wallet.verify() == before
    report = m.aggregate_oracle_paths(registry, 'null', 0)
    assert report['missing_paths'] == 127 and report['incomplete_paths'] == 1
    assert all(c['truth'] == 'uncertain' for c in report['claims'].values())


def test_original_oracle_artifact_tamper_reconstructed_not_trusted(registry_case, monkeypatch):
    _, _, enroll = registry_case
    validation, runtime, _ = enroll()
    monkeypatch.setattr(m, '_original_specs', lambda _: _engineering_specs(runtime))
    m.freeze_oracle_estimands(validation)
    oracle, _, _ = enroll(0)
    m.produce_oracle_path(oracle)
    path = oracle.paths['root']/'oracle_path.json'
    saved = json.loads(path.read_text()); saved['record']['truth'] = 'alternative'
    saved['hash'] = fingerprint(saved['record']); path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='differs from reconstructed'):
        m.read_oracle_path(oracle)


def test_oracle_enrollment_before_estimand_freeze_is_rejected(registry_case, monkeypatch):
    _, _, enroll = registry_case
    validation, runtime, _ = enroll()
    monkeypatch.setattr(m, '_original_specs', lambda _: _engineering_specs(runtime))
    enroll(0)
    with pytest.raises(ValueError, match='after oracle execution enrollment'):
        m.freeze_oracle_estimands(validation)


def test_missing_estimand_dual_anchor_cannot_reset(registry_case, monkeypatch):
    _, _, enroll = registry_case
    validation, runtime, _ = enroll()
    monkeypatch.setattr(m, '_original_specs', lambda _: _engineering_specs(runtime))
    m.freeze_oracle_estimands(validation)
    first, _ = m._estimand_paths(validation)
    first.rename(first.with_suffix('.retained'))
    with pytest.raises(ValueError, match='Missing or oversized'):
        m.freeze_oracle_estimands(validation)
    assert not first.exists()
