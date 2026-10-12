"""Real local wallets and RunStores; no market/GPT/network work."""
import json
from datetime import UTC, datetime

import pytest

from factorlab.discovery import work_authority_bridge as module
from factorlab.discovery.local_mutation import BudgetExhausted
from factorlab.discovery.monthly_work_authority import DEFAULT_LIMITS, MonthlyWorkAuthority
from factorlab.discovery.rolling import RollingSpec
from factorlab.discovery.v7_memory import search_exposure_context
from factorlab.discovery.v7_run import DurableBudget, RunStore, V7Config, monthly_budget
from factorlab.discovery.work_authority_bridge import WorkAuthorityBridge
from factorlab.discovery.work_lease import FrozenWorkLease, LeaseCheck


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    (tmp_path / 'factorlab').mkdir()
    (tmp_path / 'factorlab/frozen.py').write_text('version=1', encoding='utf-8')
    (tmp_path / 'pyproject.toml').write_text('[project]', encoding='utf-8')
    (tmp_path / 'uv.lock').write_text('version=1', encoding='utf-8')
    config, month = V7Config(evaluation_mode='joint'), '2025-04'
    store = RunStore(tmp_path / 'research', config, RollingSpec())
    authority = MonthlyWorkAuthority(tmp_path / 'authority', run_id='campaign', task='volatility',
                                     universe=['BTCUSDT'])
    exposed = search_exposure_context('data-1', month + '-01T00:00:00+00:00', ['BTCUSDT'])
    store.exposure.record(**{k: v for k, v in exposed.items() if k != 'data_id'}, role='S')
    state = {'status': 'running', 'data_id': 'data-1', 'search_exposure': exposed,
             'budget': monthly_budget(config, month).to_dict()}
    store.save(month, state, 'initial')

    def persist(value):
        state['budget'] = value
        store.save(month, state, 'projection')

    yield tmp_path, store, month, state, authority, persist
    store.close()


def bridge(case, persist=None, **kwargs):
    _, store, month, state, authority, checkpoint = case
    return WorkAuthorityBridge(store, month, state, authority, persist or checkpoint, **kwargs)


def start(b, key='baseline', counts=None, seconds=30):
    return b.begin_work(key, counts=counts or {'solve': 7}, lease_seconds=seconds,
                        binding={'stage': key, 'input_id': b.state['data_id']})


def test_one_lease_many_solves_actual_month_and_forward_remainder(case):
    b = bridge(case)
    work = start(b)
    assert isinstance(work, DurableBudget)
    for _ in range(3):
        work.consume('solve')
    done = work.complete({'fit': 'tiny artificial'})
    a = case[4]
    actual_month = datetime.now(UTC).strftime('%Y-%m')
    assert done['actual_month'] == actual_month != case[2]
    assert a.status(actual_month)['reserved_seconds'] == 30
    assert a.status(actual_month)['counts']['solve'] == 7
    assert done['actual_counts'] == {k: (3 if k == 'solve' else 0) for k in DEFAULT_LIMITS}
    later = a.reserve('forward', counts={'solve': 1, 'neural_fit': 3, 'epoch': 30},
                      binding={'channel': 'forward'}, lease_seconds=10)
    assert later['actual_month'] == actual_month
    assert a.status(actual_month)['reserved_seconds'] == 40
    assert a.status(actual_month)['counts']['solve'] == 8
    assert a.status(actual_month)['counts']['neural_fit'] == 3
    assert a.status(actual_month)['counts']['epoch'] == 30


def test_family_registry_and_unassigned_ast_persist_across_work(case):
    b = bridge(case)
    first = start(b, 'proposal', {'gpt_attempt': 1, 'family': 1, 'ast': 3})
    first.consume('gpt_attempt')
    first.consume('ast', 2)
    first.register_family('family-1')
    first.attribute_ast('family-1', 2)
    first.complete({'reply': 'synthetic'})
    second = start(b, 'candidate', {'ast': 1, 'f': 1, 'solve': 2})
    second.consume('ast', family_id='family-1')
    second.consume('f', family_id='family-1')
    second.consume('solve')
    assert second.families == {'family-1': {'ast': 3, 'f': 1}}
    second.complete({'result': 'synthetic'})
    third = start(b, 'bad-duplicate', {'family': 1})
    with pytest.raises(ValueError, match='already registered'):
        third.register_family('family-1')
    assert case[4].status(datetime.now(UTC).strftime('%Y-%m'))['counts']['family'] == 2
    assert b.actual_counts['family'] == 1


def test_limits_cannot_mint_neural_or_overuse_prepaid_solves(case):
    b = bridge(case)
    with pytest.raises(ValueError, match='separate reserved child'):
        start(b, counts={'neural_fit': 1})
    work = start(b, counts={'solve': 1})
    work.consume('solve')
    with pytest.raises(BudgetExhausted, match='envelope'):
        work.consume('solve')
    assert case[4].status(datetime.now(UTC).strftime('%Y-%m'))['counts']['solve'] == 1


def test_local_quota_stops_only_current_work_without_refund_or_cap_transfer(case):
    b = bridge(case)
    work = start(b, 'candidate-1', {'solve': 1})
    work.consume('solve')
    with pytest.raises(module.LocalQuotaExhausted, match='envelope'):
        work.consume('solve')
    assert not b._closed
    failed = work.fail('local_quota')
    assert failed['status'] == 'failed_consumed'
    with pytest.raises(ValueError, match='already prepaid'):
        start(b, 'candidate-1', {'solve': 1})
    next_work = start(b, 'candidate-2', {'solve': 2})
    next_work.consume('solve')
    result = next_work.complete({'next_fixed_work': True})
    assert result['actual_counts']['solve'] == 1
    wallet = case[4].status(datetime.now(UTC).strftime('%Y-%m'))
    assert wallet['counts']['solve'] == 3 and wallet['reserved_seconds'] == 60
    assert b.actual_counts['solve'] == 2


def test_projection_failure_after_charge_recovers_actual_without_execution_or_refund(case):
    b = bridge(case)
    work = start(b)
    original = b._persist

    def crash(value):
        raise OSError('synthetic checkpoint interruption')

    b._persist = crash
    with pytest.raises(OSError, match='checkpoint'):
        work.consume('solve')
    # The actual controller reloads persisted state, not an in-memory partial value.
    case[3].clear()
    case[3].update(case[1].get(case[2]))
    recovered = bridge(case, persist=original)
    result = recovered.recover_projection()
    assert result['projection']['actual_counts']['solve'] == 1
    assert case[1].get(case[2])['budget']['counts']['solve'] == 1
    with pytest.raises(ValueError, match='readback only'):
        start(recovered)
    assert case[4].status(datetime.now(UTC).strftime('%Y-%m'))['counts']['solve'] == 7
    assert len(result['reservations']) == 1


def test_projection_failure_after_prepaid_claim_never_recharges_or_reexecutes(case):
    b = bridge(case)
    b._persist = lambda value: (_ for _ in ()).throw(OSError('claim projection fail'))
    with pytest.raises(OSError):
        start(b)
    case[3].clear()
    case[3].update(case[1].get(case[2]))
    recovered = bridge(case)
    assert recovered.actual_counts['solve'] == 0
    with pytest.raises(ValueError, match='readback only'):
        start(recovered, seconds=29)
    assert case[4].status(datetime.now(UTC).strftime('%Y-%m'))['reserved_seconds'] == 30


def test_same_key_finished_or_failed_cannot_be_retried(case):
    b = bridge(case)
    work = start(b)
    work.consume('solve')
    work.fail('synthetic failure')
    with pytest.raises(ValueError, match='already prepaid'):
        start(b, seconds=10)
    assert case[4].status(datetime.now(UTC).strftime('%Y-%m'))['counts']['solve'] == 7
    second = start(b, 'next-fixed-work', {'solve': 2})
    second.consume('solve')
    second.complete({'ok': True})


def test_no_automatic_lease_extension(case, monkeypatch):
    b = bridge(case)
    work = start(b)
    monkeypatch.setattr(FrozenWorkLease, 'check', lambda self: LeaseCheck('deadline'))
    monkeypatch.setattr(FrozenWorkLease, 'assert_parent_alive',
                        lambda self, **kwargs: (_ for _ in ()).throw(RuntimeError('deadline')))
    with pytest.raises(RuntimeError, match='deadline'):
        work.consume('solve')
    assert b.actual_counts['solve'] == 0
    assert case[4].status(datetime.now(UTC).strftime('%Y-%m'))['reserved_seconds'] == 30


def test_work_cannot_extend_original_outer_deadline(case):
    b = bridge(case, external_deadline=module.time.monotonic() + 5)
    with pytest.raises(BudgetExhausted, match='original S/month'):
        start(b, seconds=10)
    assert case[4].verify()['reservations'] == 0
    with pytest.raises(ValueError, match='deadline cannot change'):
        bridge(case, external_deadline=module.time.monotonic() + 100)


@pytest.mark.parametrize('name', ['authority.sqlite', 'bridge_identity', 'local_identity', 'local_step', 'both_steps'])
def test_missing_committed_evidence_fails_closed_or_only_repairs_mirror(case, name):
    b = bridge(case)
    work = start(b)
    work.consume('solve')
    if name == 'authority.sqlite':
        case[4].database.unlink()
    elif name == 'bridge_identity':
        (b.anchor / 'identity.json').unlink()
    elif name == 'local_identity':
        (b.directory / 'identity.json').unlink()
    elif name == 'local_step':
        (b.directory / 'steps/000002.json').unlink()
        recovered = bridge(case)
        assert recovered.actual_counts['solve'] == 1
        return
    else:
        (b.directory / 'steps/000002.json').unlink()
        (b.anchor / 'steps/000002.json').unlink()
    with pytest.raises((ValueError, sqlite_error())):
        bridge(case)


def sqlite_error():
    import sqlite3
    return sqlite3.Error


def test_old_nonzero_month_cannot_gain_new_wallet_projection(case):
    case[3]['budget']['counts']['solve'] = 1
    case[1].save(case[2], case[3], 'legacy_budget')
    with pytest.raises(ValueError, match='legacy'):
        bridge(case)


def test_changed_input_or_source_rejects_before_reservation(case):
    b = bridge(case)
    (case[0] / 'factorlab/frozen.py').write_text('version=2', encoding='utf-8')
    with pytest.raises(ValueError):
        start(b)


def test_checkpoint_cannot_replace_counts(case):
    b = bridge(case)
    work = start(b)
    fake = work.to_dict()
    fake['counts']['solve'] = 7
    with pytest.raises(ValueError, match='mint'):
        work.persist(fake)
    copied = work.counts
    copied['solve'] = 999
    assert work.counts['solve'] == 0
    with pytest.raises(AttributeError):
        work.lease = FrozenWorkLease.issue(300)
    copied = work.record
    copied['counts']['solve'] = 999
    assert work.record['counts']['solve'] == 7


def test_original_envelope_cannot_be_mutated_in_memory(case):
    b = bridge(case)
    b.identity['execution_envelope']['monotonic_deadline'] += 100
    with pytest.raises(ValueError, match='envelope changed'):
        start(b)


def test_actual_steps_tamper_rejected_even_with_rehashed_mirror(case):
    b = bridge(case)
    work = start(b)
    work.consume('solve')
    path = b.anchor / 'steps/000002.json'
    value = json.loads(path.read_text('utf-8'))
    value['record']['actual_counts']['solve'] = 0
    value['hash'] = module.fingerprint(value['record'])
    path.write_text(json.dumps(value), encoding='utf-8')
    (b.directory / 'steps/000002.json').write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError, match='not derived'):
        bridge(case)


def test_original_parent_pid_and_creation_time_are_required(case, monkeypatch):
    b = bridge(case)
    work = start(b)
    pid = module.os.getpid()
    with monkeypatch.context() as patch:
        patch.setattr(module.os, 'getpid', lambda: pid + 1)
        with pytest.raises(ValueError, match='issuer'):
            work.check_time()
    process = module.psutil.Process()
    with monkeypatch.context() as patch:
        patch.setattr(module.psutil.Process, 'create_time', lambda self: process._create_time + 1)
        with pytest.raises(ValueError, match='issuer'):
            work.check_time()
    assert b.actual_counts['solve'] == 0


def test_frozen_input_cannot_be_changed_mid_execution(case):
    b = bridge(case)
    case[3]['data_id'] = 'new-data'
    case[3]['search_exposure']['data_id'] = 'new-data'
    case[1].save(case[2], case[3], 'changed_input')
    with pytest.raises(ValueError, match='Frozen bridge input'):
        start(b)
    assert case[4].verify()['reservations'] == 0


def test_authority_commit_before_claim_crash_never_reprepaid(case, monkeypatch):
    b = bridge(case)
    original = b.authority.start

    def crash(*args, **kwargs):
        raise OSError('before start')

    monkeypatch.setattr(b.authority, 'start', crash)
    with pytest.raises(OSError, match='before start'):
        start(b)
    monkeypatch.setattr(b.authority, 'start', original)
    recovered = bridge(case)
    result = recovered.recover_projection()
    assert len(result['reservations']) == 1
    assert result['reservations'][0]['status'] == 'reserved'
    assert result['projection']['actual_counts']['solve'] == 0
    with pytest.raises(ValueError, match='readback only'):
        start(recovered)
    assert b.authority.status(datetime.now(UTC).strftime('%Y-%m'))['reserved_seconds'] == 30


def test_completed_historical_projection_does_not_require_live_original_lease(case, monkeypatch):
    b = bridge(case)
    work = start(b)
    work.consume('solve')
    work.complete({'artifact': 'artificial'})
    monkeypatch.setattr(FrozenWorkLease, 'assert_parent_alive',
                        lambda self, **kwargs: (_ for _ in ()).throw(RuntimeError('old parent lost')))
    recovered = bridge(case)
    assert recovered.recover_projection()['reservations'][0]['status'] == 'completed'
    assert recovered.actual_counts['solve'] == 1


def test_post_projection_clock_failure_keeps_counts_but_closes_execution(case, monkeypatch):
    b = bridge(case)
    work = start(b)
    original, written = b._persist, []

    def persist(value):
        original(value)
        written.append(True)

    b._persist = persist
    old_check = FrozenWorkLease.assert_parent_alive

    def check(self, **kwargs):
        if written:
            raise RuntimeError('late after durable projection')
        return old_check(self, **kwargs)

    monkeypatch.setattr(FrozenWorkLease, 'assert_parent_alive', check)
    with pytest.raises(RuntimeError, match='late after'):
        work.consume('solve')
    assert case[1].get(case[2])['budget']['counts']['solve'] == 1
    assert b.authority.status(datetime.now(UTC).strftime('%Y-%m'))['counts']['solve'] == 7
    with pytest.raises(ValueError, match='closed'):
        work.consume('solve')
