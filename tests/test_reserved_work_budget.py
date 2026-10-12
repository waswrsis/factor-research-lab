import multiprocessing
import os
from pathlib import Path

import pytest

from factorlab.discovery import reserved_work_budget as module
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
from factorlab.discovery.nonlinear_channel import ChannelBudget
from factorlab.discovery.reserved_work_budget import ReservedChannelBudget
from factorlab.discovery.work_lease import FrozenWorkLease, LeaseCheck, LeaseClosed


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    (tmp_path / 'factorlab').mkdir()
    (tmp_path / 'factorlab/frozen.py').write_text('version=1', encoding='utf-8')
    (tmp_path / 'pyproject.toml').write_text('[project]', encoding='utf-8')
    (tmp_path / 'uv.lock').write_text('version=1', encoding='utf-8')
    return tmp_path


def wallet(root):
    return MonthlyWorkAuthority(root / 'authority', run_id='run', task='volatility', universe=['BTCUSDT'])


def running(root, counts=None):
    a = wallet(root)
    r = a.reserve('job', counts=counts or {'solve': 3, 'neural_fit': 1, 'epoch': 2}, binding={'job': 'fixed'})
    a.start(r['reservation_id'], {'pid': os.getpid()})
    return a, r


def artificial(root, counts=None):
    a, r = running(root, counts)
    return a, r, ReservedChannelBudget._for_test(a, r['reservation_id'])


def test_projection_is_channel_budget_accepts_zero_and_never_global_recharges(root):
    a, r, b = artificial(root, {'solve': 2})
    before = a.verify()
    assert isinstance(b, ChannelBudget)
    assert b.limits == {'tree_fit': 0, 'neural_fit': 0, 'epoch': 0, 'solve': 2}
    b.begin()
    b.consume('solve')
    b.charge_time(.01)
    result = b.finalize()
    assert result['test_only'] is True
    assert b.test_only is True
    with pytest.raises(AttributeError):
        b.test_only = False
    assert result == ReservedChannelBudget.verify_counts(a, r['reservation_id'])
    assert a.verify() == before
    assert a.status(r['actual_month'])['counts']['solve'] == 2
    assert b.actual_counts == {'ast': 0, 'f': 0, 'gpt_attempt': 0, 'family': 0,
                               'solve': 1, 'neural_fit': 0, 'epoch': 0, 'tree_fit': 0}


def test_public_constructor_and_issuer_factory_cannot_bypass_worker_role(root):
    a, r = running(root)
    with pytest.raises(TypeError):
        ReservedChannelBudget(max_seconds=300)
    with pytest.raises(LeaseClosed, match='wrong_process'):
        ReservedChannelBudget.from_reservation(a, r['reservation_id'])
    assert not (a.directory / 'child_actual').exists()
    with pytest.raises(ValueError, match='cannot be restored'):
        ReservedChannelBudget.from_dict({'counts': {'solve': 0}})


def test_canonical_running_status_and_source_required(root):
    a = wallet(root)
    r = a.reserve('job', counts={'solve': 1}, binding={'job': 'fixed'})
    with pytest.raises(ValueError, match='running'):
        ReservedChannelBudget._for_test(a, r['reservation_id'])
    a.start(r['reservation_id'], {'pid': os.getpid()})
    (root / 'factorlab/frozen.py').write_text('version=2', encoding='utf-8')
    with pytest.raises(ValueError, match='Current source'):
        ReservedChannelBudget._for_test(a, r['reservation_id'])


def test_fake_authority_rejected(root):
    class Fake:
        def get(self, rid):
            return {'status': 'running'}

    with pytest.raises(TypeError, match='real MonthlyWorkAuthority'):
        ReservedChannelBudget.from_reservation(Fake(), 'x')


def test_original_clock_cannot_restart_by_repeated_begin(root, monkeypatch):
    _a, r, b = artificial(root)
    lease = b._lease.to_dict()
    b.begin()
    b.begin()
    assert b._lease.to_dict() == lease == r['lease']
    assert b.max_seconds == 300

    def expired(self, **kwargs):
        raise LeaseClosed(LeaseCheck('deadline'))

    monkeypatch.setattr(FrozenWorkLease, 'assert_parent_alive', expired)
    with pytest.raises(LeaseClosed):
        b.begin()
    with pytest.raises(ValueError, match='closed'):
        b.consume('solve')
    assert b.counts['solve'] == 0


def test_lease_checked_after_durable_consumption_and_no_rollback(root, monkeypatch):
    a, r, b = artificial(root)
    checks = [0]

    def expires_after_write(self, **kwargs):
        checks[0] += 1
        if checks[0] > 1:
            raise LeaseClosed(LeaseCheck('deadline'))
        return LeaseCheck('ok', 1)

    monkeypatch.setattr(FrozenWorkLease, 'assert_parent_alive', expires_after_write)
    with pytest.raises(LeaseClosed):
        b.consume('solve')
    assert b.counts['solve'] == 1
    assert len(list((b._directory / 'steps').glob('*.json'))) == 1
    with pytest.raises(ValueError):
        ReservedChannelBudget.verify_counts(a, r['reservation_id'])


@pytest.mark.parametrize('kind,amount', [('solve', 4), ('epoch', 3), ('tree_fit', 1),
                                       ('solve', True), ('solve', 0), ('unknown', 1)])
def test_actual_cannot_exceed_any_prepaid_dimension(root, kind, amount):
    a, r, b = artificial(root)
    with pytest.raises(ValueError, match='quota'):
        b.consume(kind, amount)
    assert not any(b.counts.values())
    assert a.status(r['actual_month'])['counts']['solve'] == 3


def test_duplicate_construction_and_orphan_steps_do_not_zero_execution(root):
    a, r, b = artificial(root)
    b.consume('solve')
    with pytest.raises(ValueError, match='already claimed'):
        ReservedChannelBudget._for_test(a, r['reservation_id'])
    (b._directory / 'claim.json').unlink()
    with pytest.raises(ValueError, match='already claimed'):
        ReservedChannelBudget._for_test(a, r['reservation_id'])
    assert a.status(r['actual_month'])['counts']['solve'] == 3


def test_persistence_failure_is_closed_not_refunded_or_resumable(root, monkeypatch):
    a, r, b = artificial(root)
    original = module.immutable_record

    def fail(path, value):
        if path.parent.name == 'steps':
            raise OSError('simulated full disk')
        return original(path, value)

    monkeypatch.setattr(module, 'immutable_record', fail)
    with pytest.raises(OSError):
        b.consume('solve')
    assert b.counts['solve'] == 1
    with pytest.raises(ValueError, match='closed'):
        b.finalize()
    with pytest.raises(ValueError, match='already claimed'):
        ReservedChannelBudget._for_test(a, r['reservation_id'])
    with pytest.raises(ValueError, match='Missing'):
        ReservedChannelBudget.verify_counts(a, r['reservation_id'])


@pytest.mark.parametrize('damage', ['step', 'final', 'claim', 'extra', 'missing'])
def test_readonly_parent_rejects_tamper_missing_or_extra_chain(root, damage):
    a, r, b = artificial(root)
    b.consume('solve')
    b.consume('epoch')
    b.finalize()
    path = b._directory / {'step': 'steps/000001.json', 'final': 'final.json',
                           'claim': 'claim.json', 'extra': 'steps/000003.json',
                           'missing': 'steps/000001.json'}[damage]
    if damage == 'missing':
        path.unlink()
    else:
        path.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError):
        ReservedChannelBudget.verify_counts(a, r['reservation_id'])


def test_public_counter_dict_cannot_refund_and_final_cannot_resume(root):
    a, r, b = artificial(root)
    b.consume('solve')
    counts = b.counts
    counts['solve'] = 0
    limits = b.limits
    limits['solve'] = 1600
    assert b.counts['solve'] == 1 and b.limits['solve'] == 3
    with pytest.raises(AttributeError):
        b.shared = object()
    with pytest.raises(AttributeError):
        b.counts = {'solve': 0}
    b.finalize()
    with pytest.raises(ValueError):
        b.begin()
    assert ReservedChannelBudget.verify_counts(a, r['reservation_id'])['actual_counts']['solve'] == 1


@pytest.mark.parametrize('seconds', [-1, float('inf'), float('nan'), True, 301])
def test_invalid_time_charges_cannot_extend_lease(root, seconds):
    _a, r, b = artificial(root)
    with pytest.raises(ValueError, match='time accounting'):
        b.charge_time(seconds)
    assert b._lease.to_dict() == r['lease']


def test_time_receipts_have_finite_count_bound(root):
    _a, _r, b = artificial(root)
    for _ in range(module.MAX_TIME_CHARGES):
        b.charge_time(0)
    with pytest.raises(ValueError, match='time accounting'):
        b.charge_time(0)
    assert len(list((b._directory / 'steps').glob('*.json'))) == module.MAX_TIME_CHARGES


def test_completed_parent_actual_must_match_reconstructed_child_counts(root):
    a, r, b = artificial(root)
    b.consume('solve')
    b.finalize()
    a.complete(r['reservation_id'], result={'artifact_hash': 'engineered'}, actual_counts={})
    with pytest.raises(ValueError, match='Parent completion differs'):
        ReservedChannelBudget.verify_counts(a, r['reservation_id'])


def _real_child(root, rid, ready, queue):
    import sys

    import factorlab.protocol
    factorlab.protocol.ROOT = Path(root)
    ready.wait(timeout=20)
    try:
        a = wallet(Path(root))
        b = ReservedChannelBudget.from_reservation(a, rid)
        assert b.test_only is False
        with pytest.raises(AttributeError):
            b.test_only = True
        b.begin()
        b.consume('solve', 2)
        b.consume('neural_fit')
        b.consume('epoch')
        b.charge_time(.01)
        result = b.finalize()
        assert 'torch' not in sys.modules
        queue.put(('ok', result))
    except (OSError, ValueError, RuntimeError, AssertionError) as exc:
        queue.put(('error', repr(exc)))


def test_real_direct_child_projection_and_parent_readback(root):
    a = wallet(root)
    r = a.reserve('job', counts={'solve': 3, 'neural_fit': 1, 'epoch': 2}, binding={'job': 'fixed'})
    ctx = multiprocessing.get_context('spawn')
    ready, queue = ctx.Event(), ctx.Queue()
    child = ctx.Process(target=_real_child, args=(str(root), r['reservation_id'], ready, queue))
    child.start()
    try:
        a.start(r['reservation_id'], {'pid': child.pid})
        ready.set()
        status, result = queue.get(timeout=30)
        assert status == 'ok', result
        child.join(timeout=10)
        assert child.exitcode == 0
        verified = ReservedChannelBudget.verify_counts(a, r['reservation_id'])
        assert verified == result and verified['test_only'] is False
        assert verified['actual_counts']['solve'] == 2
        assert a.status(r['actual_month'])['counts']['solve'] == 3
        a.complete(r['reservation_id'], result={'hash': 'actual-test-model'}, actual_counts=verified['actual_counts'])
        assert ReservedChannelBudget.verify_counts(a, r['reservation_id']) == verified
    finally:
        if child.is_alive():
            child.terminate()
            child.join(timeout=5)
        queue.close()


def test_missing_execution_directory_cannot_reclaim_prepaid_ticket(root):
    a, r, b = artificial(root)
    preserved = b._directory.with_name(b._directory.name + '-lost-copy')
    b._directory.rename(preserved)
    with pytest.raises(ValueError, match='already claimed'):
        ReservedChannelBudget._for_test(a, r['reservation_id'])
    assert (a.directory / 'child_actual_claims' / (r['reservation_id'] + '.json')).is_file()


def test_parent_rejects_missing_independent_execution_anchor(root):
    a, r, b = artificial(root)
    b.consume('solve')
    b.finalize()
    (a.directory / 'child_actual_claims' / (r['reservation_id'] + '.json')).unlink()
    with pytest.raises(ValueError, match='Missing'):
        ReservedChannelBudget.verify_counts(a, r['reservation_id'])


def test_parent_rejects_oversized_receipt_before_json_read(root):
    a, r, b = artificial(root)
    b.finalize()
    (b._directory / 'final.json').write_text(' ' * 32769, encoding='utf-8')
    with pytest.raises(ValueError, match='Missing/damaged'):
        ReservedChannelBudget.verify_counts(a, r['reservation_id'])


def test_spent_time_cannot_be_reopened_by_begin(root):
    _a, _r, b = artificial(root)
    with pytest.raises(ValueError, match='state changed'):
        b.charge_time(b.max_seconds)
    assert b.spent_seconds == 300
    with pytest.raises(ValueError, match='closed'):
        b.begin()
