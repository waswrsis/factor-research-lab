"""Independent boundary checks; artificial records are not calibration evidence."""
import json

import numpy as np
import pandas as pd
import pytest

from factorlab.discovery.convex_eval import fit_ridge
from factorlab.discovery.v7_lifecycle import Lifecycle, LifecyclePlan


@pytest.fixture
def audit_store(tmp_path, monkeypatch):
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    clock = [pd.Timestamp('2026-10-01T00:00:05Z')]
    store = Lifecycle(tmp_path / 'audit', LifecyclePlan('audit-feed'), clock=lambda: clock[0])
    yield store, clock
    store.close()


def _ready(store, clock, stream='A0', pipeline='K0'):
    cutoff = clock[0].normalize().replace(day=1)
    job = store.plan_fit(cutoff, stream, {'pipeline_id': pipeline})
    store.advance_fit(job, 'snapshot_sealed')
    store.advance_fit(job, 'features_ready')
    model = fit_ridge(np.arange(40).reshape(-1, 1), np.linspace(.01, .03, 40))
    model_id = store.seal_model(job, model)
    return job, model_id


def _forecast(store, clock, stream='A0', pipeline='K0'):
    job, model_id = _ready(store, clock, stream, pipeline)
    store.activate(job, clock[0])
    origin = clock[0]
    store.record_forecast(stream=stream, origin=origin, symbol='BTCUSDT', task='volatility',
        prediction=.02, received_at=origin, feature_open_time=origin.floor('h') - pd.Timedelta(hours=1),
        forecast_window_start=origin.ceil('h'), model_id=model_id, pipeline_id=pipeline)
    return origin


def _freeze(store):
    store.exposure.record(source='audit-feed', symbols=['BTCUSDT'],
        start='2026-07-01T00:00:00Z', end='2026-10-01T00:00:00Z', role='S', batch='A')
    kwargs = {'batch': 'A', 'ordinal': 1, 'start': '2026-11-01T00:00:00Z',
        'candidates': [{'pipeline_id': 'factor-A', 'complexity': 3}],
        'reference': {'pipeline_id': 'K0'},
        'evidence_sources': [{'start': '2026-07-01T00:00:00Z', 'end': '2026-10-01T00:00:00Z'}]}
    return kwargs, store.freeze_batch(**kwargs)


def _loss_rows(start, end, loss):
    return {(t.isoformat(), 'BTCUSDT'): (loss, .03, 'first_received_v1')
            for t in pd.date_range(start, end, freq='h', inclusive='left')}


def _selected(store, clock, monkeypatch):
    _freeze(store)
    clock[0] = pd.Timestamp('2026-12-01T00:00:05Z')
    monkeypatch.setattr(store, '_losses', lambda stream, start, end, as_of, **kwargs:
        _loss_rows(start, end, 1. if stream.endswith(':reference') else .9))
    assert store.select_validation('A')['selected']['candidate_id'] == 'factor-A'
    # This isolates lifecycle plumbing; a fake gate never certifies market evidence.
    monkeypatch.setattr(store, 'calibration_ready', lambda: True)
    store.reserve_selected_confirmation('A')


def test_freeze_retry_preserves_original_time_after_clock_advances(audit_store):
    store, clock = audit_store
    kwargs, digest = _freeze(store)
    clock[0] += pd.Timedelta(seconds=1)
    assert store.freeze_batch(**kwargs) == digest


def test_failed_verified_model_cannot_activate_without_reverification(audit_store):
    store, clock = audit_store
    job, _ = _ready(store, clock)
    store.fail_fit(job, 'integrity_failure')
    with pytest.raises(ValueError):
        store.activate(job, clock[0])
    assert store.active_model('A0') is None


def test_twelve_hour_label_cannot_settle_before_forecast_window_matures(audit_store):
    store, clock = audit_store
    origin = _forecast(store, clock)
    clock[0] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError):
        store.settle(stream='A0', origin=origin, symbol='BTCUSDT', task='volatility',
            label=.03, maturity=clock[0], received_at=clock[0])
    assert store.db.execute('SELECT COUNT(*) FROM settlement').fetchone()[0] == 0


def test_frozen_validation_period_cannot_be_same_batch_prompt_search(audit_store):
    store, clock = audit_store
    _freeze(store)
    clock[0] = pd.Timestamp('2026-11-15T00:00:00Z')
    request = {'source': 'audit-feed', 'symbols': ['BTCUSDT'], 'batch': 'A',
               'start': '2026-11-01T00:00:00Z', 'end': '2026-11-10T00:00:00Z'}
    # Recording facts may remain permitted, but they may never pass the prompt guard.
    try:
        store.exposure.record(**request, role='S')
    except ValueError:
        return
    with pytest.raises(ValueError):
        store.exposure.assert_prompt_search(**request)


def test_validation_rejects_wrong_pipeline_even_when_stream_name_matches(audit_store):
    store, clock = audit_store
    _freeze(store)
    clock[0] = pd.Timestamp('2026-11-01T00:00:05Z')
    origins = {}
    for stream in ('A:reference', 'A:factor-A'):
        # Both streams actually contain K0. Naming a stream is not pipeline identity.
        try:
            origins[stream] = _forecast(store, clock, stream, 'K0')
        except ValueError:
            if stream == 'A:factor-A':
                return  # Early identity validation is also valid.
            raise
    clock[0] = pd.Timestamp('2026-11-01T13:00:05Z')
    for stream, origin in origins.items():
        store.settle(stream=stream, origin=origin, symbol='BTCUSDT', task='volatility',
            label=.03, maturity=clock[0], received_at=clock[0])
    clock[0] = pd.Timestamp('2026-12-01T00:00:00Z')
    with pytest.raises(ValueError, match='[Pp]ipeline|identity'):
        store.select_validation('A')


def test_confirmation_filters_actual_start_but_keeps_nominal_denominator(audit_store, monkeypatch):
    store, clock = audit_store
    _selected(store, clock, monkeypatch)
    captured = {}

    def score(origins, baseline, candidate, **kwargs):
        captured.update(origins=origins, **kwargs)
        return {'status': 'unresolved'}

    monkeypatch.setattr('factorlab.discovery.v7_lifecycle.paired_effect_interval', score)
    clock[0] = pd.Timestamp('2027-03-03T00:00:00Z')
    store.conclude_confirmation('A')
    assert min(map(pd.Timestamp, captured['origins'])) >= pd.Timestamp('2026-12-01T01:00:00Z')
    assert captured['eligible_count'] == 24 * 90


def test_committed_confirmation_result_recovers_without_second_score(audit_store, monkeypatch):
    store, clock = audit_store
    _selected(store, clock, monkeypatch)
    scored = []
    monkeypatch.setattr('factorlab.discovery.v7_lifecycle.paired_effect_interval',
                        lambda *a, **kw: scored.append(True) or {'status': 'unresolved'})
    import factorlab.discovery.v7_lifecycle as lifecycle

    original = lifecycle.immutable_record

    def interrupted(path, value):
        if path.parent.name == 'claims':
            raise OSError('simulated disk interruption after committed result')
        return original(path, value)

    monkeypatch.setattr(lifecycle, 'immutable_record', interrupted)
    clock[0] = pd.Timestamp('2027-03-03T00:00:00Z')
    with pytest.raises(OSError):
        store.conclude_confirmation('A')
    durable = json.loads(store.exposure.db.execute(
        'SELECT result FROM confirmations WHERE batch=?', ('A',)).fetchone()[0])
    monkeypatch.setattr(lifecycle, 'immutable_record', original)
    assert store.conclude_confirmation('A') == durable
    assert len(scored) == 1


def test_validation_uses_intersection_of_all_candidates(audit_store, monkeypatch):
    store, clock = audit_store
    kwargs, _ = _freeze(store)
    # A separate batch has a second candidate with half the support. It must not
    # let A gain selection on an easier support than B or the reference.
    store.exposure.record(source='audit-feed', symbols=['BTCUSDT'],
        start='2026-07-01T00:00:00Z', end='2026-10-01T00:00:00Z', role='S', batch='B')
    kwargs.update(batch='B', ordinal=2, candidates=kwargs['candidates'] + [
        {'pipeline_id': 'factor-B', 'complexity': 2}])
    store.freeze_batch(**kwargs)
    clock[0] = pd.Timestamp('2026-12-01T00:00:00Z')

    def losses(stream, start, end, as_of, **kwargs):
        rows = _loss_rows(start, end, 1. if stream.endswith(':reference') else .9)
        return dict(list(rows.items())[::2]) if stream.endswith(':factor-B') else rows

    monkeypatch.setattr(store, '_losses', losses)
    result = store.select_validation('B')
    assert result['selected'] is None
    assert {record['coverage'] for record in result['records']} == {.5}
    assert result['common_pairs'] == 360


def test_validation_equal_hour_weight_matches_confirmation_with_missing_symbols(tmp_path, monkeypatch):
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    clock = [pd.Timestamp('2026-10-01T00:00:05Z')]
    store = Lifecycle(tmp_path / 'multi', LifecyclePlan('audit-feed', symbols=('BTCUSDT', 'ETHUSDT')),
                      clock=lambda: clock[0])
    try:
        store.exposure.record(source='audit-feed', symbols=['BTCUSDT', 'ETHUSDT'],
            start='2026-07-01T00:00:00Z', end='2026-10-01T00:00:00Z', role='S', batch='A')
        _freeze(store)
        clock[0] = pd.Timestamp('2026-12-01T00:00:00Z')

        def losses(stream, start, end, as_of, **kwargs):
            reference = stream.endswith(':reference')
            rows = {}
            for i, hour in enumerate(pd.date_range(start, end, freq='h', inclusive='left')):
                symbols = ['BTCUSDT'] if i < 36 else ['BTCUSDT', 'ETHUSDT']
                loss = (100. if reference else 110.) if i < 36 else (1. if reference else .6)
                for symbol in symbols:
                    rows[hour.isoformat(), symbol] = (loss, .03, 'first_received_v1')
            return rows

        monkeypatch.setattr(store, '_losses', losses)
        result = store.select_validation('A')
        assert result['records'][0]['coverage'] == .975
        # Symbol-hour weighting gives +3.77%, but equal hours give -2.02%.
        expected_gain = 1 - (36 * 110 + 684 * .6) / (36 * 100 + 684)
        assert result['records'][0]['gain'] == pytest.approx(expected_gain)
        assert result['selected'] is None
    finally:
        store.close()


def test_confirmation_delay_cannot_shrink_nominal_identity_or_exceed_twenty_four_hours(audit_store):
    store, clock = audit_store
    clock[0] = pd.Timestamp('2026-12-01T23:59:00Z')
    kwargs = {'batch': 'late', 'source': 'audit-feed', 'symbols': ['BTCUSDT'],
        'start': '2026-12-01T00:00:00Z', 'end': '2027-03-01T00:00:00Z',
        'candidate_id': 'factor-A', 'reference_id': 'K0', 'rule': {'horizon_hours': 12},
        'frozen_at': clock[0], 'cohort': 1, 'claim_slot': 'factor1'}
    with pytest.raises(ValueError):
        store.exposure.reserve_confirmation(**kwargs, actual_start='2026-12-02T00:00:01Z')
    store.exposure.reserve_confirmation(**kwargs, actual_start='2026-12-02T00:00:00Z')
    identity = json.loads(store.exposure.db.execute(
        'SELECT identity FROM confirmations WHERE batch=?', ('late',)).fetchone()[0])
    assert identity['start'] == identity['nominal_start'] == '2026-12-01T00:00:00+00:00'
    assert identity['activation_delay_seconds'] == 24 * 3600
    assert identity['scoring_cutoff'] == '2027-03-02T12:00:00+00:00'
