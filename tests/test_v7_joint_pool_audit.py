"""Independent invariants for joint-factor attribution and frozen evidence."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from factorlab.discovery import joint_pool as joint
from factorlab.discovery.known_library import FactorFrame, library_schema
from factorlab.discovery.local_mutation import BudgetExhausted
from factorlab.discovery.rolling import RollingData, RollingSpec
from factorlab.protocol import fingerprint


def _member(node, *, parameters=None, mode='fixed'):
    return joint.PoolMember({'version': 'causal-operator-program-v1',
        'feature_names': library_schema()['feature_names'], 'parameters': parameters or {},
        'outputs': [node]}, lookback=2, parameter_mode=mode)


def _last(name):
    return {'op': 'last', 'arg': {'op': 'feature', 'name': name}}


def _parameter_member(name='rv_1'):
    return _member({'op': 'tanh', 'arg': {'op': 'mul', 'left': _last(name),
                                        'right': {'op': 'param', 'name': 'theta'}}},
        parameters={'theta': {'initial': .6, 'lower': .1, 'upper': 1.9}}, mode='profile')


@pytest.fixture(scope='module')
def audit_data():
    time = pd.date_range('2026-01-01', '2026-07-05', inclusive='left', freq='h', tz='UTC')
    x = np.random.default_rng(887).uniform(.4, 1.6, (len(time), 12))
    metadata = pd.DataFrame({'symbol': 'A', 'available_at': time + pd.Timedelta(seconds=1),
        'feature_open_time': time - pd.Timedelta(hours=1), 'forecast_window_start': time + pd.Timedelta(hours=1)})
    schema = library_schema()
    frame = FactorFrame(x, metadata, tuple(schema['feature_names']), tuple(schema['bypass_mask']), fingerprint(schema))
    target = np.exp(-2 + .7 * x[:, 0] * x[:, 1]) - 1e-8
    return RollingData(frame, target, time + pd.Timedelta(hours=13), np.ones(len(time), dtype=bool),
                       RollingSpec(minimum_train_rows=24), 'assumed_historical_delay')


def test_prefix_access_cannot_change_under_frozen_reference():
    existing, added = _member(_last('rv_1')), _member(_last('rv_6'))
    prefix = joint.FrozenPool((existing,), mode='residual')
    with pytest.raises(ValueError, match='prefix|reference|access'):
        joint.FrozenPool((existing, added), reference_pool=prefix, reference_pipeline_id=prefix.pipeline_id,
                         access_modes=('raw', 'raw'))


@pytest.mark.parametrize('field,value', [('cutoff', '2026-08-01T00:00:00+00:00'),
                                       ('data_hash', 'changed-data'), ('source_hash', 'changed-source'),
                                       ('protocol_hash', 'changed-protocol')])
def test_old_fold_result_cannot_be_relabeled_into_new_context(audit_data, field, value):
    candidate = _member({'op': 'mul', 'left': _last('rv_1'), 'right': _last('rv_6')})
    pool = joint.FrozenPool()
    archive = joint.DevelopmentPoolArchive('2026-08-01T00:00Z')
    context = archive.context(pool, cutoff='2026-07-01T00:00Z', data_hash='data',
                              protocol_hash='protocol', source_hash='source')
    result = joint.evaluate_candidate(audit_data, '2026-07-01T00:00Z', pool, candidate, context=context)
    archive.record(candidate, result, context)
    context = {**context, field: value}
    with pytest.raises(ValueError, match='context|cutoff|identity|matching'):
        archive.record(candidate, result, context)


def test_fitted_pool_cannot_silently_follow_later_formula_mutation(audit_data):
    member = _member({'op': 'mul', 'left': _last('rv_1'), 'right': _last('rv_6')})
    pool = joint.FrozenPool((member,))
    fitted = joint.fit_pool(audit_data, '2026-07-01T00:00Z', pool)
    prediction = fitted.predict(audit_data.frame)
    member.program['outputs'][0]['left']['arg']['name'] = 'rv_24'
    try:
        repeated = fitted.predict(audit_data.frame)
    except ValueError:
        return  # Failing closed is an acceptable immutable-runtime contract.
    np.testing.assert_array_equal(repeated, prediction)


def test_ragged_clone_symbol_does_not_reweight_calendar_hours(audit_data):
    # A second identical asset exists during only part of the training period.
    # Equal-hour weighting must leave the fitted K0 function unchanged.
    positions = np.flatnonzero((audit_data.frame.metadata.available_at >= '2026-05-01T00:00Z') &
                               (audit_data.frame.metadata.available_at < '2026-05-20T00:00Z'))
    extra = audit_data.frame.metadata.iloc[positions].copy()
    extra['symbol'] = 'B'
    frame = FactorFrame(np.vstack([audit_data.frame.values, audit_data.frame.values[positions]]),
        pd.concat([audit_data.frame.metadata, extra], ignore_index=True), audit_data.frame.feature_names,
        audit_data.frame.bypass_mask, audit_data.frame.schema_hash)
    extended = RollingData(frame, np.r_[audit_data.target, audit_data.target[positions]],
        audit_data.maturity.append(audit_data.maturity[positions]),
        np.r_[audit_data.observed, audit_data.observed[positions]], audit_data.spec, audit_data.availability_mode)
    single = joint.fit_pool(audit_data, '2026-07-01T00:00Z', joint.FrozenPool())
    ragged = joint.fit_pool(extended, '2026-07-01T00:00Z', joint.FrozenPool())
    np.testing.assert_allclose(single.predict(audit_data.frame), ragged.predict(audit_data.frame), rtol=1e-12, atol=1e-14)


def test_profiled_prefix_is_identical_in_F_theta0_and_theta_hat(audit_data, monkeypatch):
    prefix = joint.FrozenPool((_parameter_member('rv_24'),), max_profile_calls=3)
    candidate = _parameter_member('rv_1')
    fits = []
    original = joint._fit_pool

    def captured(*args, **kwargs):
        result = original(*args, **kwargs)
        fits.append(result[0])
        return result

    monkeypatch.setattr(joint, '_fit_pool', captured)
    result = joint.evaluate_candidate(audit_data, '2026-07-01T00:00Z', prefix, candidate)
    by_digest = {model.digest: model for model in fits}
    for fold in result['folds']:
        reference = by_digest[fold['models']['Pv']['digest']]
        for key in ('F_Pv', 'theta0_raw', 'theta_hat_raw'):
            model = by_digest[fold['models'][key]['digest']]
            assert model.parameters[0] == reference.parameters[0]
            assert model.receipt['training']['common_support_hash'] == reference.receipt['training']['common_support_hash']
            assert model.receipt['frozen_reference_fit']['model_digest'] == reference.digest


def test_parameter_budget_failure_halts_before_any_unreserved_solve(audit_data, monkeypatch):
    attempts = []

    class Budget:
        def consume(self, kind, **kwargs):
            attempts.append(kind)
            raise BudgetExhausted('injected hard stop')

    def forbidden(*args, **kwargs):
        pytest.fail('A solver ran after reservation was rejected')

    monkeypatch.setattr('factorlab.discovery.convex_eval.fit_ridge', forbidden)
    monkeypatch.setattr(joint, 'fit_ridge', forbidden)
    with pytest.raises(BudgetExhausted, match='hard stop'):
        joint.fit_pool(audit_data, '2026-07-01T00:00Z', joint.FrozenPool((_parameter_member(),)), Budget())
    assert attempts == ['solve']


def test_near_zero_residual_and_saved_parameters_keep_predictions(audit_data, tmp_path):
    duplicate = _member({'op': 'mul', 'left': _last('rv_1'), 'right': {'op': 'const', 'value': 1e-9}})
    empty = joint.fit_pool(audit_data, '2026-07-01T00:00Z', joint.FrozenPool())
    fitted = joint.fit_pool(audit_data, '2026-07-01T00:00Z', joint.FrozenPool((duplicate,), mode='residual'))
    restored = joint.JointPredictor.load(fitted.save(tmp_path / 'tiny_duplicate.json'))
    pred, support = restored.predict_history(audit_data.frame)
    assert fitted.projectors[0].near_zero
    np.testing.assert_allclose(pred[support], empty.predict(audit_data.frame)[support], rtol=1e-12, atol=1e-13)
    np.testing.assert_array_equal(pred, fitted.predict(audit_data.frame))


def test_nontraining_label_changes_do_not_change_profile_or_parameters(audit_data):
    cutoff = '2026-06-01T00:00Z'
    altered = audit_data.target.copy()
    altered[~audit_data.train_mask(cutoff) & audit_data.observed] *= 100
    changed = replace(audit_data, target=altered)
    pool = joint.FrozenPool((_parameter_member(),), max_profile_calls=3)
    original = joint.fit_pool(audit_data, cutoff, pool)
    refit = joint.fit_pool(changed, cutoff, pool)
    assert original.digest == refit.digest


def test_fitted_parameter_mutation_is_rejected_before_predict_or_save(audit_data, tmp_path):
    fitted = joint.fit_pool(audit_data, '2026-07-01T00:00Z',
                            joint.FrozenPool((_parameter_member(),), max_profile_calls=3))
    fitted.parameters[0]['theta'] = .11 if fitted.parameters[0]['theta'] != .11 else .12
    with pytest.raises(ValueError, match='mutat|integrity|sealed|changed'):
        fitted.predict(audit_data.frame)
    with pytest.raises(ValueError, match='mutat|integrity|sealed|changed'):
        fitted.save(tmp_path / 'changed_parameter.json')
