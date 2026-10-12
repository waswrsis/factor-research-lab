import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from factorlab.discovery.joint_pool import (
    DevelopmentPoolArchive,
    FrozenPool,
    JointPredictor,
    PoolMember,
    backward_prune,
    evaluate_candidate,
    fit_pool,
    frame_alignment_id,
    hour_equal_weights,
    prescreen_candidates,
    stability_select,
)
from factorlab.discovery.known_library import FactorFrame, library_schema
from factorlab.discovery.rolling import RollingData, RollingSpec, development_folds
from factorlab.protocol import fingerprint


def program(expression, parameters=None):
    return {'version': 'causal-operator-program-v1', 'feature_names': library_schema()['feature_names'],
            'parameters': parameters or {}, 'outputs': [expression]}


def last(name):
    return {'op': 'last', 'arg': {'op': 'feature', 'name': name}}


def member(expression, *, parameters=None, lookback=2, parameter_mode='fixed'):
    return PoolMember(program(expression, parameters), lookback, parameter_mode)


@pytest.fixture(scope='module')
def data():
    times = pd.date_range('2026-01-01', '2026-07-05', freq='h', tz='UTC', inclusive='left')
    rng = np.random.default_rng(144)
    x = rng.uniform(.5, 1.5, (len(times), 12))
    x[:, 8] = np.sin(2*np.pi*times.hour/24)
    x[:, 9] = np.cos(2*np.pi*times.hour/24)
    x[:, 10] = np.sin(2*np.pi*times.dayofweek/7)
    x[:, 11] = np.cos(2*np.pi*times.dayofweek/7)
    metadata = pd.DataFrame({'symbol': 'BTC', 'available_at': times + pd.Timedelta(seconds=1),
                             'feature_open_time': times - pd.Timedelta(hours=1),
                             'forecast_window_start': times + pd.Timedelta(hours=1)})
    schema = library_schema()
    frame = FactorFrame(x, metadata, tuple(schema['feature_names']), tuple(schema['bypass_mask']), fingerprint(schema))
    y = np.exp(-2 + .3*x[:, 0] + .8*x[:, 0]*x[:, 1]) - 1e-8
    return RollingData(frame, y, times + pd.Timedelta(hours=13), np.ones(len(times), dtype=bool),
                       RollingSpec(minimum_train_rows=24), 'assumed_historical_delay')


class Budget:
    def __init__(self, limit=1000):
        self.calls = []
        self.limit = limit

    def consume(self, kind, **kwargs):
        if len(self.calls) >= self.limit:
            raise RuntimeError('synthetic hard budget')
        self.calls.append((kind, kwargs))


def test_joint_fit_history_roundtrip_and_model_identity(data, tmp_path):
    a = member({'op': 'mul', 'left': last('rv_1'), 'right': last('rv_6')})
    b = member({'op': 'tanh', 'arg': last('rv_24')})
    pool = FrozenPool((a, b), version='P_dev_July')
    budget = Budget()
    fitted = fit_pool(data, '2026-07-01T00:00Z', pool, budget)
    pred, mask = fitted.predict_history(data.frame)
    assert not mask[0] and mask[1:].all() and np.isnan(pred[0])
    assert fitted.kind == 'v7_joint_pool' and len(budget.calls) == 1
    features, feature_mask = fitted.feature_history(data.frame)
    assert features.values.shape[1] == 14
    np.testing.assert_equal(feature_mask, mask)
    np.testing.assert_equal(fitted.predictor.predict(features.take(mask)), pred[mask])
    path = fitted.save(tmp_path / 'joint.json')
    restored = JointPredictor.load(path)
    assert restored.digest == fitted.digest and restored.pipeline_id == fitted.pipeline_id
    np.testing.assert_equal(restored.predict_history(data.frame)[0], pred)
    assert fitted.receipt['solve_calls'] == 1 and fitted.receipt['receipt_id']
    other = replace(data, spec=replace(data.spec, ridge=.01))
    changed = fit_pool(other, '2026-07-01T00:00Z', pool)
    assert changed.pipeline_id != fitted.pipeline_id
    raw = json.loads(path.read_text())
    raw['model']['parameters'][0]['injected'] = 1
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='integrity'):
        JointPredictor.load(path)


def test_future_labels_and_values_do_not_change_cold_fit(data):
    candidate = member({'op': 'mul', 'left': last('rv_1'), 'right': last('rv_6')})
    pool = FrozenPool((candidate,))
    cutoff = pd.Timestamp('2026-07-01T00:00Z')
    future = data.frame.metadata.available_at >= cutoff
    values = data.frame.values.copy()
    values[future, :8] *= 9
    target = data.target.copy()
    target[future] *= 100
    frame = FactorFrame(values, data.frame.metadata, data.frame.feature_names, data.frame.bypass_mask, data.frame.schema_hash)
    changed = replace(data, frame=frame, target=target)
    original_fit = fit_pool(data, cutoff, pool)
    changed_fit = fit_pool(changed, cutoff, pool)
    assert original_fit.digest == changed_fit.digest
    before = original_fit.predict_history(data.frame)[0]
    after = changed_fit.predict_history(changed.frame)[0]
    np.testing.assert_equal(before[~future], after[~future])


def test_F_duplicate_control_P_same_access_and_S_separate(data):
    duplicate = member({'op': 'mul', 'left': last('rv_1'), 'right': {'op': 'const', 'value': 2.}})
    result = evaluate_candidate(data, '2026-07-01T00:00Z', FrozenPool(), duplicate)
    assert abs(result['F']['K0']['absolute_delta']) < 1e-12
    assert abs(result['F']['Pv']['absolute_delta']) < 1e-12
    assert result['P']['absolute_delta'] == pytest.approx(0, abs=1e-15)
    assert result['P']['access'] == 'same_raw_conditional_pool'
    assert all(f['F']['near_zero_K0'] and f['F']['near_zero_Pv'] for f in result['folds'])
    assert all(f['S']['old_root']['status'] == 'not_recomputed' for f in result['folds'])
    assert result['evidence_status'] == 'exposed_development_only'


def test_joint_conditional_P_profiles_only_candidate_and_accounts_calls(data):
    existing = member({'op': 'mul', 'left': last('rv_24'), 'right': last('rv_48')})
    candidate = member({'op': 'tanh', 'arg': {'op': 'mul', 'left': last('rv_1'),
                       'right': {'op': 'param', 'name': 'alpha'}}},
                       parameters={'alpha': {'initial': .5, 'lower': .1, 'upper': 2.}},
                       parameter_mode='profile')
    pool = FrozenPool((existing,), max_profile_calls=6)
    budget = Budget()
    result = evaluate_candidate(data, '2026-07-01T00:00Z', pool, candidate, budget=budget)
    profiles = [f['models']['theta_hat_raw']['receipt']['parameter_profile'] for f in result['folds']]
    assert sum(p['calls'] for p in profiles) == 18
    assert len(budget.calls) == 36  # Six cold readouts + six profile solves per fold.
    for fold in result['folds']:
        assert fold['P']['reference_theta_unchanged']
        assert .1 <= fold['P']['theta_hat']['alpha'] <= 2.
        f_receipt = fold['models']['F_Pv']['receipt']
        p_receipt = fold['models']['theta_hat_raw']['receipt']
        assert f_receipt['frozen_reference_fit'] == p_receipt['frozen_reference_fit']
        assert f_receipt['parameter_profile'] is None


def test_hour_equal_weights_reject_duplicate_pairs_and_do_not_overweight_assets():
    metadata = pd.DataFrame({'symbol': ['A', 'B', 'A'],
                             'available_at': pd.to_datetime(['2026-01-01T00:00Z', '2026-01-01T00:00Z', '2026-01-01T01:00Z'])})
    np.testing.assert_equal(hour_equal_weights(metadata), [.5, .5, 1.])
    metadata.loc[1, 'symbol'] = 'A'
    with pytest.raises(ValueError, match='duplicate'):
        hour_equal_weights(metadata)


def test_prescreen_never_reads_label_values_and_keeps_blind_spots(data):
    candidates = [member(last('rv_1')),
                  member({'op': 'mul', 'left': last('rv_1'), 'right': last('rv_6')}),
                  member({'op': 'tanh', 'arg': last('rv_48')}),
                  member({'op': 'const', 'value': 1.})]
    changed = replace(data, target=data.target[::-1].copy())
    left = prescreen_candidates(data, '2026-07-01T00:00Z', FrozenPool(), candidates, limit=3, blind_spot_slots=2)
    right = prescreen_candidates(changed, '2026-07-01T00:00Z', FrozenPool(), candidates, limit=3, blind_spot_slots=2)
    assert left == right and left['label_values_used'] is False
    assert len(left['selected_ids']) == 3 and len(left['blind_spot_ids']) == 2
    assert len(left['entries']) == 4 and not left['hard_rejection_by_correlation']


def test_backward_prune_is_bounded_development_only_and_retains_receipts(data):
    a = member(last('rv_1'))
    b = member({'op': 'const', 'value': 3.})
    pool = FrozenPool((a, b), mode='residual')
    pruned, receipt = backward_prune(data, '2026-07-01T00:00Z', pool, max_remove=1, tolerance=1e-10)
    assert len(pruned.members) == 1 and len(pool.members) == 2
    assert len(receipt['removed_ids']) == 1 and receipt['affects_A0'] is False
    assert receipt['steps'][0]['trials'][0]['fit_receipts']
    for protected in ('A0', 'confirmation'):
        with pytest.raises(ValueError, match='development'):
            backward_prune(data, '2026-07-01T00:00Z', replace(pool, role=protected))


def fake_result(pool, candidate):
    return {'role': 'S', 'pool_hash': pool.digest, 'candidate_id': candidate.member_id,
            'receipt': {'cutoff': '2026-07-01T00:00:00+00:00'},
            'folds': [{}, {}, {}], 'F': {'Pv': {'gain': .01, 'winning_fold_fraction': 1.}}}


def test_archive_context_parent_liveness_admission_decay_and_rollover(tmp_path):
    pool = FrozenPool()
    candidate = member(last('rv_1'))
    result = fake_result(pool, candidate)
    archive = DevelopmentPoolArchive('2026-07-01T00:00Z')
    context = archive.context(pool, cutoff='2026-07-01T00:00Z', data_hash='data',
                              protocol_hash='protocol', source_hash='source')
    result['context'] = context
    archive.record(candidate, result, context)
    assert archive.result_for(candidate.member_id, context) == result
    future_context = {**context, 'cutoff': '2026-08-01T00:00:00+00:00', 'data_hash': 'new-data'}
    assert archive.result_for(candidate.member_id, future_context) is None
    parents = archive.active_parents(future_context)
    assert parents[0]['prior_evidence_only'] and parents[0]['current_context_result'] is None
    selection = stability_select([result], pool)
    admitted, receipt = archive.admit(pool, candidate, month=context['cutoff'], selection=selection)
    assert len(admitted.members) == 1 and not pool.members and receipt['affects_A0'] is False
    with pytest.raises(ValueError, match='one new'):
        archive.admit(pool, candidate, month=context['cutoff'], selection=selection)
    path = archive.save(tmp_path / 'july.json')
    rolled = archive.rollover('2026-08-01T00:00Z', path)
    assert rolled.predecessor['digest'] == archive.digest
    assert DevelopmentPoolArchive.load(path).digest == archive.digest
    for month in ('2026-08-01T00:00Z', '2026-09-01T00:00Z', '2026-10-01T00:00Z'):
        decay = rolled.monitor_decay(candidate.member_id, month=month, k0_gain=.01, pool_gain=-.01)
    assert decay['state'] == 'pending_retirement' and decay['reason'] == 'possibly_replaced_by_pool'
    assert len(admitted.members) == 1 and decay['automatic_exit'] is False
    exit_selection = {'role': 'S', 'pool_hash': admitted.digest, 'decision': 'retire'}
    retired, receipt = rolled.exit_members(admitted, [candidate.member_id], selection=exit_selection)
    assert not retired.members and receipt['affects_A0'] is False
    with pytest.raises(ValueError, match='development'):
        rolled.exit_members(replace(admitted, role='A0'), [candidate.member_id], selection=exit_selection)


def test_pool_and_archive_bounds_and_context_rejection():
    candidates = [member({'op': 'const', 'value': float(i)}) for i in range(9)]
    with pytest.raises(ValueError, match='eight'):
        FrozenPool(tuple(candidates))
    with pytest.raises(ValueError, match='unique'):
        FrozenPool((candidates[0], candidates[0]))
    archive = DevelopmentPoolArchive('2026-07-01T00:00Z')
    pool, candidate = FrozenPool(), candidates[0]
    result = fake_result(pool, candidate)
    context = archive.context(pool, cutoff='2026-07-01T00:00Z', data_hash='d', protocol_hash='p', source_hash='s')
    result['context'] = context
    with pytest.raises(ValueError, match='matching'):
        archive.record(candidate, {**result, 'role': 'C'}, context)
    for i in range(96):
        changed_context = {**context, 'data_hash': f'd-{i}'}
        archive.record(candidate, {**result, 'context': changed_context}, changed_context)
    with pytest.raises(ValueError, match='shard full'):
        changed_context = {**context, 'data_hash': 'd-97'}
        archive.record(candidate, {**result, 'context': changed_context}, changed_context)


def test_joint_module_is_torch_free():
    command = ('import sys; import factorlab.discovery.joint_pool; '
               'assert "torch" not in sys.modules')
    result = subprocess.run([sys.executable, '-B', '-c', command], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_nested_profile_reference_survives_save_load(data, tmp_path):
    existing = member({'op': 'tanh', 'arg': {'op': 'mul', 'left': last('rv_24'),
                      'right': {'op': 'param', 'name': 'alpha'}}},
                      parameters={'alpha': {'initial': .5, 'lower': .1, 'upper': 2.}},
                      parameter_mode='profile')
    prefix = FrozenPool((existing,), max_profile_calls=3)
    added = member({'op': 'mul', 'left': last('rv_1'), 'right': last('rv_6')})
    nested = FrozenPool((existing, added), reference_pool=prefix,
                        reference_pipeline_id=prefix.pipeline_id)
    budget = Budget()
    fitted = fit_pool(data, '2026-07-01T00:00Z', nested, budget)
    assert len(budget.calls) == 5 and fitted.receipt['solve_calls'] == 5
    assert fitted.receipt['frozen_reference_fit']['pipeline_id']
    restored = JointPredictor.load(fitted.save(tmp_path / 'nested.json'))
    assert restored.pool.reference_pool.digest == prefix.digest
    np.testing.assert_equal(restored.predict_history(data.frame)[0], fitted.predict_history(data.frame)[0])


def test_real_valueerror_budget_halts_profile_immediately(data):
    from factorlab.discovery.local_mutation import BudgetExhausted

    class ExhaustedBudget:
        def __init__(self):
            self.attempts = 0

        def consume(self, *args, **kwargs):
            self.attempts += 1
            raise BudgetExhausted('test exhausted')

    candidate = member({'op': 'tanh', 'arg': {'op': 'mul', 'left': last('rv_1'),
                       'right': {'op': 'param', 'name': 'alpha'}}},
                       parameters={'alpha': {'initial': .5, 'lower': .1, 'upper': 2.}},
                       parameter_mode='profile')
    budget = ExhaustedBudget()
    with pytest.raises(BudgetExhausted, match='test exhausted'):
        fit_pool(data, '2026-07-01T00:00Z', FrozenPool((candidate,)), budget)
    assert budget.attempts == 1
    with pytest.raises(ValueError, match='profile admission policy'):
        evaluate_candidate(data, '2026-07-01T00:00Z', FrozenPool(),
                           replace(candidate, parameter_mode='fixed'))


def test_old_root_compares_only_exact_declared_cold_fold_identities(data):
    candidate = member({'op': 'mul', 'left': last('rv_1'), 'right': last('rv_6')})
    old = {fold['fit_cutoff']: {'fit_cutoff': fold['fit_cutoff'],
           'alignment_id': frame_alignment_id(data.frame), 'pipeline_id': 'archived-old-root',
           'artifact_sha256': 'a' * 64, 'predictions': np.full(len(data.target), .2),
           'mask': np.ones(len(data.target), dtype=bool)}
           for fold in development_folds('2026-07-01T00:00Z')}
    result = evaluate_candidate(data, '2026-07-01T00:00Z', FrozenPool(), candidate, old_root=old)
    assert all(r['status'] == 'paired_archived_prediction_comparison' for r in result['old_root'])
    assert result['reference_pipeline_id'] == result['folds'][0]['models']['Pv']['pipeline_id']
    old[next(iter(old))]['fit_cutoff'] = '2026-06-01T00:00:00+00:00'
    with pytest.raises(ValueError, match='frozen origin/fit identity'):
        evaluate_candidate(data, '2026-07-01T00:00Z', FrozenPool(), candidate, old_root=old)


def test_archive_gaps_quality_and_stability_fractions_cannot_force_exit():
    pool, candidate = FrozenPool(), member(last('rv_1'))
    result = fake_result(pool, candidate)
    archive = DevelopmentPoolArchive('2026-07-01T00:00Z')
    context = archive.context(pool, cutoff='2026-07-01T00:00Z', data_hash='d', protocol_hash='p', source_hash='s')
    result['context'] = context
    archive.record(candidate, result, context)
    for month, good in [('2026-07-01T00:00Z', True), ('2026-09-01T00:00Z', True),
                        ('2026-10-01T00:00Z', True), ('2026-11-01T00:00Z', False)]:
        receipt = archive.monitor_decay(candidate.member_id, month=month, k0_gain=-.01,
                                        pool_gain=-.01, quality_ok=good)
        assert receipt['state'] == 'active_parent'
    assert receipt['reason'] == 'data_quality'
    with pytest.raises(ValueError, match='bounded archive'):
        stability_select([result, result], pool)
    result['F']['Pv']['winning_fold_fraction'] = 2.
    with pytest.raises(ValueError, match='winning-fold fraction'):
        stability_select([result], pool)
