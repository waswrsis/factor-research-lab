"""Actual fixed schedule, wallet and numeric stage tests; no provider/network."""
from datetime import UTC, datetime

import pytest
from test_work_authority_bridge import case as case  # noqa: PLC0414

from factorlab.discovery import scheduled_search as schedule_module
from factorlab.discovery import v7_s_controller as controller_module
from factorlab.discovery import work_authority_bridge as bridge_module
from factorlab.discovery.local_mutation import BudgetExhausted


def staged(case, *, bridge_cls=None):
    _, store, month, state, authority, persist = case
    bridge = (bridge_cls or bridge_module.WorkAuthorityBridge)(store, month, state, authority, persist)
    slots = [{'key': key, 'stage': stage, 'counts': {'solve': count}, 'seconds': 10}
             for key, stage, count in [('baseline', 'baseline', 1), ('candidate:0', 'candidate', 2),
                                      ('pruning', 'pruning', 1)]]
    schedule = schedule_module.FrozenSWorkSchedule.freeze(config=store.identity['config'],
        limits=bridge.identity['limits'], slots=slots)
    return schedule_module.ScheduledSearch(bridge, schedule,
                                           local_quota_type=bridge_module.LocalQuotaExhausted)


def solve(work):
    work.consume('solve')
    return {'artificial_result': True}


def exhaust(work):
    work.consume('solve')
    work.consume('solve')


def test_staged_local_quota_only_closes_current_work_and_never_refunds(case):
    scheduled = staged(case)
    original_envelope = scheduled.bridge.identity['execution_envelope'].copy()
    first = scheduled.run('baseline', exhaust, input_binding={'stage': 'artificial'})
    second = scheduled.run('candidate:0', solve, input_binding={'candidate': 'fixed'})
    assert first['status'] == 'quota_consumed' and second['status'] == 'completed'
    assert scheduled.bridge.identity['execution_envelope'] == original_envelope
    authority = case[4]
    assert authority.get(first['reservation_id'])['status'] == 'failed_consumed'
    assert authority.status(datetime.now(UTC).strftime('%Y-%m'))['counts']['solve'] == 3
    assert scheduled.bridge.actual_counts['solve'] == 2


def test_order_and_allocations_cannot_change_to_reuse_unused_credit(case):
    scheduled = staged(case)
    scheduled.skip('baseline', reason='no_candidate')
    scheduled.run('candidate:0', solve, input_binding={'candidate': 'fixed'})
    # Baseline's unused one solve is not moved to the next slot.
    result = scheduled.run('pruning', exhaust, input_binding={'pool': 'fixed'})
    assert result['status'] == 'quota_consumed'
    assert case[4].status(datetime.now(UTC).strftime('%Y-%m'))['counts']['solve'] == 3


def test_out_of_order_work_closes_s_before_any_reservation(case):
    scheduled = staged(case)
    with pytest.raises(ValueError, match='next immutable'):
        scheduled.run('candidate:0', solve, input_binding={'candidate': 'fixed'})
    assert not scheduled.bridge._reservations()
    with pytest.raises(ValueError, match='closed'):
        scheduled.run('baseline', solve, input_binding={'baseline': True})


@pytest.mark.parametrize('error', [OSError('disk'), ValueError('identity'), BudgetExhausted('original S deadline')])
def test_nonlocal_failures_close_whole_s_and_keep_authority_charge(case, error):
    scheduled = staged(case)
    def fail(work):
        work.consume('solve')
        raise error
    with pytest.raises(type(error)):
        scheduled.run('baseline', fail, input_binding={'fixed': True})
    with pytest.raises(ValueError, match='closed'):
        scheduled.run('candidate:0', solve, input_binding={'candidate': 'fixed'})
    assert case[4].status(datetime.now(UTC).strftime('%Y-%m'))['counts']['solve'] == 1


def test_restart_is_readback_only_not_a_second_s_deadline(case):
    scheduled = staged(case)
    scheduled.run('baseline', solve, input_binding={'fixed': True})
    recovered = staged(case)
    assert recovered.bridge.identity == scheduled.bridge.identity
    with pytest.raises(ValueError, match='readback only'):
        recovered.run('candidate:0', solve, input_binding={'candidate': 'fixed'})
    assert case[4].status(datetime.now(UTC).strftime('%Y-%m'))['reserved_seconds'] == 10


def test_work_leases_cannot_overrun_original_s_cutoff(case, monkeypatch):
    scheduled = staged(case)
    envelope = scheduled.bridge.identity['execution_envelope']
    monkeypatch.setattr(bridge_module.time, 'monotonic', lambda: envelope['monotonic_deadline'] + 1)
    with pytest.raises(BudgetExhausted, match='Original S'):
        scheduled.run('baseline', solve, input_binding={'fixed': True})
    assert not scheduled.bridge._reservations()


def test_schedule_freezes_before_work_and_cannot_be_replaced(case):
    scheduled = staged(case)
    original = scheduled.schedule.to_dict()
    original['slots'][1]['counts']['solve'] = 3
    altered = schedule_module.FrozenSWorkSchedule.freeze(config=original['config'],
        limits=original['limits'], slots=original['slots'])
    with pytest.raises(ValueError, match='Immutable'):
        schedule_module.ScheduledSearch(scheduled.bridge, altered,
            local_quota_type=bridge_module.LocalQuotaExhausted)


def test_independent_forward_reservation_spends_same_wallet(case):
    scheduled = staged(case)
    scheduled.run('baseline', solve, input_binding={'fixed': True})
    reservation = case[4].reserve('actual-forward', counts={'solve': 5}, lease_seconds=10,
                                  binding={'channel': 'A0', 'engineering_fixture': True})
    assert case[4].status(reservation['actual_month'])['counts']['solve'] == 6
    assert case[4].status(reservation['actual_month'])['reserved_seconds'] == 20


def test_actual_baseline_and_candidate_stages_use_shared_prepaid_wallet(case):
    from dataclasses import replace

    import numpy as np
    import pandas as pd
    from test_v7_nonlinear_channel import artificial_data

    from factorlab.discovery.joint_pool import DevelopmentPoolArchive, FrozenPool, PoolMember
    from factorlab.discovery.rolling import RollingSpec
    from factorlab.discovery.v7_run import literature_seeds
    from factorlab.protocol import fingerprint

    controller = controller_module
    _, store, month, state, authority, persist = case
    bridge = bridge_module.WorkAuthorityBridge(store, month, state, authority, persist)
    # Explicit artificial allocation, not a proposed default scientific policy.
    slots, allocation = controller.standard_slots(candidate_caps=[15]*24, candidate_seconds=[10]*24,
        pruning_cap=3, pruning_seconds=10, forward_solve_holdback=1230)
    frozen = schedule_module.FrozenSWorkSchedule.freeze(config=store.identity['config'],
        limits=bridge.identity['limits'], slots=slots, allocation=allocation)
    scheduled = schedule_module.ScheduledSearch(bridge, frozen,
        local_quota_type=bridge_module.LocalQuotaExhausted)
    original = artificial_data()
    index = np.repeat(np.arange(len(original.target)), 12)
    meta = original.frame.metadata.iloc[index].reset_index(drop=True)
    offset = pd.to_timedelta(np.tile(np.arange(12), len(original.target)), unit='h')
    for column in ('available_at', 'forecast_window_start', 'feature_open_time'):
        meta[column] += offset
    meta['symbol'] = 'BTCUSDT'
    meta['sample_id'] = [f'artificial-{i}' for i in range(len(index))]
    data = replace(original, frame=replace(original.frame, values=original.frame.values[index], metadata=meta),
        target=original.target[index], maturity=original.maturity[index] + offset,
        observed=original.observed[index], spec=RollingSpec())
    pool = FrozenPool()
    context = DevelopmentPoolArchive.context(pool, cutoff='2025-04-01T00:00Z',
        data_hash=state['data_id'], protocol_hash=data.spec.digest,
        source_hash=fingerprint(store.identity['source_files']))
    member = PoolMember(literature_seeds()[0], lookback=2)
    result = controller.staged_numeric_month(scheduled, data, '2025-04-01T00:00Z', pool=pool,
        families=[[member], [], []], context=context)
    assert result['role'] == 'S' and not result['formal_claim']
    assert result['baseline']['status'] == 'completed'
    assert result['candidates'][0]['status'] == 'completed'
    assert result['actual_counts']['solve'] == 7 + controller.candidate_solve_bound(pool, parameterized=False)
    assert result['actual_counts']['f'] == 1
    wallet = authority.status(datetime.now(UTC).strftime('%Y-%m'))
    assert wallet['counts']['solve'] == 22 and wallet['counts']['ast'] == 32
    assert wallet['counts']['gpt_attempt'] == 2 and wallet['counts']['family'] == 1
    assert result['actual_counts']['gpt_attempt'] == 0  # Nontransferable unused family prepayment.


def test_two_backend_attempts_share_original_family_remaining_time(case, monkeypatch):
    import time

    controller = controller_module
    _, store, month, state, authority, persist = case
    bridge = bridge_module.WorkAuthorityBridge(store, month, state, authority, persist)
    slots = [{'key': 'family:0', 'stage': 'family', 'seconds': 10,
              'counts': {'family': 1, 'gpt_attempt': 2, 'ast': 32}}]
    frozen = schedule_module.FrozenSWorkSchedule.freeze(config=store.identity['config'],
        limits=bridge.identity['limits'], slots=slots)
    scheduled = schedule_module.ScheduledSearch(bridge, frozen,
        local_quota_type=bridge_module.LocalQuotaExhausted)
    timeouts = []
    class FakeBackend:
        def __init__(self, directory, *, timeout_seconds):
            timeouts.append(timeout_seconds)
        def __call__(self, packet, *, model, reasoning):
            assert model == 'gpt-6-astra' and reasoning == 'high'
            time.sleep(.01)
            return {'fake': True}
    monkeypatch.setattr(controller, 'CodexBatchBackend', FakeBackend)
    def attempts(work):
        original = work.lease.digest
        adapter = controller.OriginalLeaseBackend(work, store.directory / 'agent')
        for _ in range(2):
            work.consume('gpt_attempt')
            adapter(None, model='gpt-6-astra', reasoning='high')
        assert work.lease.digest == original
        return {'fake_attempts': 2}
    scheduled.run('family:0', attempts, input_binding={'fake_request_hash': 'fixed'})
    assert 0 < timeouts[1] < timeouts[0] < 10
    current = authority.status(datetime.now(UTC).strftime('%Y-%m'))
    assert current['counts']['gpt_attempt'] == 2 and current['reserved_seconds'] == 10


def test_fixed_unfunded_slot_does_not_reallocate_or_close_cheaper_later_slot(case):
    _, store, month, state, authority, persist = case
    # Real previous forward work leaves exactly two solves in this same wallet.
    authority.reserve('prior-forward', counts={'solve': 1598}, lease_seconds=1,
                      binding={'role': 'A0', 'engineering_fixture': True})
    bridge = bridge_module.WorkAuthorityBridge(store, month, state, authority, persist)
    slots = [{'key': 'expensive', 'stage': 'candidate', 'counts': {'solve': 3}, 'seconds': 10},
             {'key': 'cheap', 'stage': 'candidate', 'counts': {'solve': 1}, 'seconds': 10}]
    frozen = schedule_module.FrozenSWorkSchedule.freeze(config=store.identity['config'],
        limits=bridge.identity['limits'], slots=slots)
    scheduled = schedule_module.ScheduledSearch(bridge, frozen,
        local_quota_type=bridge_module.LocalQuotaExhausted)
    assert scheduled.run('expensive', solve, input_binding={'id': 'expensive'})['status'] == 'unfunded_fixed_slot'
    assert scheduled.run('cheap', solve, input_binding={'id': 'cheap'})['status'] == 'completed'
    assert authority.status(datetime.now(UTC).strftime('%Y-%m'))['counts']['solve'] == 1599


def test_exhaustion_classifier_cannot_be_replaced_to_hide_integrity_failure(case):
    _, store, month, state, authority, persist = case
    bridge = bridge_module.WorkAuthorityBridge(store, month, state, authority, persist)
    frozen = schedule_module.FrozenSWorkSchedule.freeze(config=store.identity['config'],
        limits=bridge.identity['limits'], slots=[{'key': 'baseline', 'stage': 'baseline',
                                                'counts': {'solve': 1}, 'seconds': 10}])
    with pytest.raises(TypeError, match='counter-only'):
        schedule_module.ScheduledSearch(bridge, frozen, local_quota_type=ValueError)
    assert authority.verify()['reservations'] == 0


