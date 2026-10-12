"""Artificial real-ledger endpoint checks; no formal calibration certificate."""
import json

import numpy as np
import pandas as pd
import pytest

from factorlab.discovery import claim_protocol as module
from factorlab.discovery.convex_eval import fit_ridge
from factorlab.discovery.joint_pool import FrozenPool
from factorlab.discovery.rolling import RollingSpec
from factorlab.discovery.v7_forward import ForwardRuntime
from factorlab.discovery.v7_lifecycle import Lifecycle, LifecyclePlan
from factorlab.protocol import fingerprint


@pytest.fixture
def frozen(tmp_path, monkeypatch, request):
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    (tmp_path / 'factorlab').mkdir()
    (tmp_path / 'factorlab/source.py').write_text('source=1')
    (tmp_path / 'pyproject.toml').write_text('[project]')
    (tmp_path / 'uv.lock').write_text('version=1')
    clock = [pd.Timestamp('2025-12-31T23:00:00Z')]
    symbols = getattr(request, 'param', ('BTCUSDT',))
    lc = Lifecycle(tmp_path / 'run', LifecyclePlan('actual-artificial-feed', symbols=symbols), clock=lambda: clock[0])
    descriptor = ForwardRuntime.__new__(ForwardRuntime)
    descriptor.spec = RollingSpec()
    k0 = descriptor.recipe_for(FrozenPool(role='reference', version='K0_v1'))
    reference = {'pipeline_id': fingerprint('benchmark'), 'benchmark': 'frozen_artificial_baseline'}
    endpoints = [module.FrozenClaimEndpoint.library('2026-01-01T00:00:00Z', k0, reference),
                 module.FrozenClaimEndpoint.algorithm('2026-01-01T00:00:00Z', k0)]
    protocol = module.ClaimProtocol.freeze(lc, endpoints)
    # Offline endpoint fixture has no full runtime/paid selector. Keep its
    # historical K0 policy as an explicit plumbing seam, never an adoption proof.
    monkeypatch.setattr('factorlab.discovery.typed_policy_confirmation.candidate_pipeline', lambda _lc, _cutoff, k0: k0)
    yield protocol, lc, clock, endpoints
    lc.close()


def forecasts(frozen, *, mismatch=False, missing_candidate=False, late=False):
    _protocol, lc, clock, endpoints = frozen
    model = fit_ridge(np.arange(60).reshape(-1, 1), np.linspace(.001, .03, 60))
    clock[0] = pd.Timestamp('2026-01-01T00:00:00Z')
    ids = {}
    pipelines = {'K0': endpoints[0].candidate, 'benchmark': endpoints[0].reference,
                 module.TRIAL_STREAM: endpoints[1].reference}
    for stream, pipeline in pipelines.items():
        job = lc.plan_fit(clock[0], stream, pipeline)
        lc.advance_fit(job, 'snapshot_sealed', output='artificial-immutable-data')
        lc.advance_fit(job, 'features_ready', output='artificial-features')
        ids[stream] = lc.seal_model(job, model)
        lc.activate(job, clock[0])
    origin = pd.Timestamp('2026-01-01T00:00:01Z')
    clock[0] = origin
    for stream, value in [('benchmark', .1), ('K0', .12), (module.TRIAL_STREAM, .12)]:
        if missing_candidate and stream == 'K0':
            continue
        lc.record_forecast(stream=stream, origin=origin, symbol='BTCUSDT', task='volatility',
            prediction=value, received_at=origin, feature_open_time='2025-12-31T23:00:00Z',
            forecast_window_start='2026-01-01T01:00:00Z', model_id=ids[stream],
            pipeline_id=pipelines[stream]['pipeline_id'])
    clock[0] = pd.Timestamp('2026-01-01T13:00:01Z')
    if late:
        clock[0] = pd.Timestamp(endpoints[0].scoring_cutoff) + pd.Timedelta(hours=1)
    for stream in pipelines:
        if missing_candidate and stream == 'K0':
            continue
        lc.settle(stream=stream, origin=origin, symbol='BTCUSDT', task='volatility',
                  label=.14 if mismatch and stream == 'K0' else .13,
                  maturity='2026-01-01T13:00:00Z', received_at=clock[0])
    return origin


def mature(frozen):
    frozen[2][0] = pd.Timestamp(frozen[3][0].scoring_cutoff)


def test_exact_six_calendar_months_and_same_five_slot_ledger(frozen):
    protocol, lc, _, endpoints = frozen
    assert endpoints[0].end == '2026-07-01T00:00:00+00:00'
    rows = lc.exposure.db.execute('SELECT cohort,claim_slot FROM confirmations ORDER BY claim_slot').fetchall()
    assert rows == [(1, 'algorithm'), (1, 'library_skill')]
    assert protocol.finalize('algorithm')['status'] == 'pending'
    assert lc.exposure.db.execute('SELECT SUM(consumed) FROM confirmations').fetchone()[0] == 0


def test_bound_directory_loss_cannot_disable_or_restart_claims(frozen, tmp_path):
    protocol, lc, _, endpoints = frozen
    retained = protocol.directory.with_name('retained_claim_evidence')
    assert protocol.directory.resolve().is_relative_to(tmp_path.resolve())
    assert retained.resolve().is_relative_to(tmp_path.resolve())
    protocol.directory.rename(retained)
    with pytest.raises(ValueError, match='Bound claim directory missing'):
        module.ClaimProtocol.freeze(lc, endpoints)
    assert not protocol.directory.exists()
    assert lc.exposure.db.execute('SELECT COUNT(*) FROM confirmations').fetchone()[0] == 2
    assert module.ClaimProtocol._bindings(lc) == [
        {'protocol_digest': protocol.digest, 'directory': str(protocol.directory)}]


@pytest.mark.parametrize('after_start', [False, True])
def test_partial_two_slot_freeze_recovers_only_original_prospective_identity(
        frozen, tmp_path, monkeypatch, after_start):
    _, original_lc, clock, endpoints = frozen
    lc = Lifecycle(tmp_path / 'partial', original_lc.plan, clock=lambda: clock[0])
    reserve = lc.exposure.reserve_confirmation

    def interrupted(**kwargs):
        if kwargs['claim_slot'] == 'algorithm':
            raise OSError('second reservation interrupted')
        return reserve(**kwargs)

    try:
        monkeypatch.setattr(lc.exposure, 'reserve_confirmation', interrupted)
        with pytest.raises(OSError, match='second reservation'):
            module.ClaimProtocol.freeze(lc, endpoints)
        manifest = json.loads((lc.directory / 'claim_protocol/protocol.json').read_text())
        old = lc.exposure.db.execute('SELECT identity FROM confirmations').fetchall()
        assert len(old) == 1 and len(module.ClaimProtocol._bindings(lc)) == 1
        monkeypatch.setattr(lc.exposure, 'reserve_confirmation', reserve)
        clock[0] += pd.Timedelta(minutes=61 if after_start else 30)
        if after_start:
            with pytest.raises(ValueError):
                module.ClaimProtocol.freeze(lc, endpoints)
            assert lc.exposure.db.execute('SELECT identity FROM confirmations').fetchall() == old
        else:
            resumed = module.ClaimProtocol.freeze(lc, endpoints)
            assert resumed.manifest == manifest
            assert resumed.manifest['frozen_at'] == '2025-12-31T23:00:00+00:00'
            assert lc.exposure.db.execute('SELECT COUNT(*) FROM confirmations').fetchone()[0] == 2
            assert len(module.ClaimProtocol._bindings(lc)) == 1
            assert old[0] in lc.exposure.db.execute('SELECT identity FROM confirmations').fetchall()
    finally:
        lc.close()


def test_actual_losses_common_support_and_nominal_denominator_do_not_shrink(frozen):
    forecasts(frozen)
    mature(frozen)
    score = frozen[0].finalize('library_skill')
    result = score['result']
    assert result['eligible'] == 181 * 24 and result['support'] == 1
    assert result['pair_eligible'] == 181 * 24
    assert result['coverage'] == 1 / (181 * 24)
    assert result['supported'] is False and score['formal_eligible'] is False
    a = (np.log(.13 + 1e-8) - np.log(.1 + 1e-8)) ** 2
    b = (np.log(.13 + 1e-8) - np.log(.12 + 1e-8)) ** 2
    assert result['d_mean'] == pytest.approx(.995 * a - b)
    algorithm = frozen[0].finalize('algorithm')
    assert algorithm['result']['relative_gain'] == 0


def test_missing_candidate_and_whole_waiting_period_are_counted_as_missing(frozen):
    forecasts(frozen, missing_candidate=True)
    mature(frozen)
    value = frozen[0].finalize('library_skill')['result']
    assert value['eligible'] == 4344 and value['support'] == 0
    assert value['coverage'] == value['pair_coverage'] == 0


def test_label_mismatch_consumes_once_and_cannot_be_rescored(frozen):
    forecasts(frozen, mismatch=True)
    mature(frozen)
    with pytest.raises(ValueError, match='labels'):
        frozen[0].finalize('library_skill')
    assert module.ClaimProtocol(frozen[1]).finalize('library_skill')['status'] == 'interrupted_consumed'


def test_committed_score_recovers_publication_without_new_look(frozen, monkeypatch):
    forecasts(frozen)
    mature(frozen)
    original = frozen[1].exposure.finish_confirmation
    monkeypatch.setattr(frozen[1].exposure, 'finish_confirmation',
                        lambda *a: (_ for _ in ()).throw(OSError('publication unavailable')))
    with pytest.raises(OSError, match='publication unavailable'):
        frozen[0].finalize('library_skill')
    monkeypatch.setattr(frozen[1].exposure, 'finish_confirmation', original)
    monkeypatch.setattr(module, 'paired_effect_interval', lambda *a, **k: pytest.fail('Cannot rescore'))
    monkeypatch.setattr(module.ClaimProtocol, '_losses', lambda *a: pytest.fail('Cannot reread outcomes'))
    recovered = module.ClaimProtocol(frozen[1]).finalize('library_skill')
    assert recovered == frozen[0].finalize('library_skill')
    assert recovered['completed_at'] == frozen[2][0].isoformat()


def test_caller_calibration_stub_cannot_open_formal_gate(frozen, monkeypatch):
    forecasts(frozen)
    mature(frozen)
    monkeypatch.setattr(frozen[1], 'calibration_ready', lambda: True)
    result = frozen[0].finalize('algorithm')
    assert not result['formal_eligible'] and not result['result']['calibration_passed']
    assert not result['result']['supported']


def test_changed_endpoint_and_missing_index_cannot_reset_budget(frozen):
    protocol, lc, _, endpoints = frozen
    changed = module.FrozenClaimEndpoint.library('2026-02-01T00:00:00Z', endpoints[0].candidate, endpoints[0].reference)
    with pytest.raises(ValueError):
        module.ClaimProtocol.freeze(lc, [changed, endpoints[1]])
    protocol.path.unlink()
    with pytest.raises(ValueError, match='missing'):
        module.ClaimProtocol.freeze(lc, endpoints)
    assert lc.exposure.db.execute('SELECT COUNT(*) FROM confirmations').fetchone()[0] == 2


def test_same_completed_score_cannot_self_authorize_by_rehashed_json(frozen):
    forecasts(frozen)
    mature(frozen)
    value = frozen[0].finalize('library_skill')
    value['formal_eligible'] = True
    with module.closing(module.sqlite3.connect(frozen[0].path)) as db, db:
        db.execute('UPDATE claims SET score=?,hash=? WHERE slot=?',
                   (json.dumps(value), fingerprint(value), 'library_skill'))
    with pytest.raises(ValueError, match='self-authorized'):
        module.ClaimProtocol(frozen[1])


def test_frozen_source_and_loss_inputs_are_required(frozen):
    forecasts(frozen)
    mature(frozen)
    frozen[0].finalize('library_skill')
    (frozen[0].directory / 'library_skill/inputs.json').write_text('{}')
    with pytest.raises(ValueError, match='paired-loss'):
        module.ClaimProtocol(frozen[1])


def test_endpoint_mutation_and_factor_slot_substitution_rejected(frozen):
    frozen[3][0].candidate['pool']['members'].append('not-K0')
    with pytest.raises(ValueError, match='mutated|changed'):
        frozen[0].verify()
    with pytest.raises(ValueError, match='factor slots'):
        frozen[0].finalize('factor1')


@pytest.mark.parametrize('frozen', [('BTCUSDT', 'ETHUSDT')], indirect=True)
def test_missing_whole_symbol_retains_nominal_symbol_hour_denominator(frozen):
    forecasts(frozen)
    mature(frozen)
    result = frozen[0].finalize('library_skill')['result']
    assert result['eligible'] == 4344 and result['pair_eligible'] == 8688
    assert result['coverage'] == 1 / 4344 and result['pair_coverage'] == 1 / 8688


def test_late_label_receipt_cannot_enter_original_fixed_endpoint(frozen):
    forecasts(frozen, late=True)
    result = frozen[0].finalize('library_skill')['result']
    assert result['support'] == 0 and result['eligible'] == 4344


@pytest.mark.parametrize('extra', ['complexity', 'unregistered_field'])
def test_actual_a0_job_activation_strips_only_selection_complexity(frozen, monkeypatch, extra):
    """Actual typed recipe/job/activation; adoption policy is a private plumbing stub."""
    forecasts(frozen)
    mature(frozen)
    k0 = frozen[3][1].reference
    monkeypatch.setattr('factorlab.discovery.typed_policy_confirmation.candidate_pipeline', lambda *_: {**k0, extra: 7})
    if extra == 'complexity':
        assert frozen[0].finalize('algorithm')['result']['support'] == 1
    else:
        with pytest.raises(ValueError, match='causal selection rule'):
            frozen[0].finalize('algorithm')
