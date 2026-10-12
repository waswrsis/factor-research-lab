"""Independent issuer-mode audit on real local wallets and original leases.

Tiny Windows/local subprocess checks are not Linux resource certification.
No numerical training, remote processes or market inputs are used here.
"""
import json
import multiprocessing
import os
import time
from pathlib import Path

import pytest
from test_reserved_work_budget import root as root  # noqa: PLC0414
from test_reserved_work_budget import wallet

from factorlab.discovery import neural_forward, neural_research
from factorlab.discovery.monthly_work_authority import DEFAULT_LIMITS
from factorlab.discovery.reserved_work_budget import ReservedChannelBudget
from factorlab.discovery.work_lease import LeaseClosed
from factorlab.protocol import fingerprint


def issuer(root, *, counts=None, seconds=300, identity=None):
    authority = wallet(root)
    reservation = authority.reserve('issuer-audit', counts={'solve': 2} if counts is None else counts,
                                    lease_seconds=seconds, binding={'work': 'fixed-linear-control'})
    process = {'pid': os.getpid(), 'created': reservation['lease']['parent_created']}
    if identity is not None:
        process.update(identity)
    authority.start(reservation['reservation_id'], process)
    return authority, reservation


def test_real_issuer_is_solve_only_and_complete_counts_remain_prepaid(root):
    authority, record = issuer(root)
    budget = ReservedChannelBudget.from_issuer_reservation(authority, record['reservation_id'])
    assert budget.executor_mode == 'issuer' and budget.test_only is False
    assert budget._lease.to_dict() == record['lease']
    before = authority.verify()
    budget.begin()
    budget.consume('solve')
    budget.charge_time(.001)
    actual = budget.finalize()
    assert actual['version'] == 'reserved-channel-budget-v2'
    assert actual == ReservedChannelBudget.verify_counts(authority, record['reservation_id'])
    assert authority.verify() == before
    authority.complete(record['reservation_id'], result={'engineering_audit': True},
                       actual_counts=actual['actual_counts'])
    assert ReservedChannelBudget.verify_counts(authority, record['reservation_id']) == actual
    assert authority.status(record['actual_month'])['counts']['solve'] == 2
    assert actual['actual_counts'] == {k: int(k == 'solve') for k in DEFAULT_LIMITS}


@pytest.mark.parametrize('identity', [{'created': 1.}, {'created': None}, {'pid': 1}])
def test_canonical_running_identity_must_equal_original_pid_and_creation(root, identity):
    authority, record = issuer(root, identity=identity)
    with pytest.raises(ValueError, match='original parent'):
        ReservedChannelBudget.from_issuer_reservation(authority, record['reservation_id'])
    assert not (authority.directory / 'child_actual').exists()


@pytest.mark.parametrize('kind', [k for k in DEFAULT_LIMITS if k != 'solve'])
def test_any_non_solve_prepaid_permission_closes_issuer_entry(root, kind):
    authority, record = issuer(root, counts={'solve': 1, kind: 1})
    with pytest.raises(ValueError, match='solve-only'):
        ReservedChannelBudget.from_issuer_reservation(authority, record['reservation_id'])
    assert authority.status(record['actual_month'])['counts'][kind] == 1
    assert not (authority.directory / 'child_actual').exists()


def test_duplicate_issuer_claim_cannot_reset_counter_even_if_directory_is_lost(root):
    authority, record = issuer(root)
    budget = ReservedChannelBudget.from_issuer_reservation(authority, record['reservation_id'])
    budget.consume('solve')
    with pytest.raises(ValueError, match='already claimed'):
        ReservedChannelBudget.from_issuer_reservation(authority, record['reservation_id'])
    budget._directory.rename(budget._directory.with_name(budget._directory.name + '-preserved'))
    with pytest.raises(ValueError, match='already claimed'):
        ReservedChannelBudget.from_issuer_reservation(authority, record['reservation_id'])
    assert authority.status(record['actual_month'])['counts']['solve'] == 2


def test_real_original_deadline_expires_without_restarting_on_begin(root):
    authority, record = issuer(root, seconds=2)
    budget = ReservedChannelBudget.from_issuer_reservation(authority, record['reservation_id'])
    budget.begin()
    original = budget._lease.to_dict()
    remaining = record['lease']['monotonic_deadline'] - time.monotonic()
    if remaining > 0:
        time.sleep(remaining + .03)
    with pytest.raises(LeaseClosed, match='deadline'):
        budget.begin()
    assert budget._lease.to_dict() == original
    with pytest.raises(ValueError, match='closed'):
        budget.consume('solve')
    assert not any(budget.counts.values())
    assert authority.status(record['actual_month'])['counts']['solve'] == 2
    assert authority.recover(record['reservation_id'])['status'] == 'interrupted_consumed'


@pytest.mark.parametrize('entry', ['forward', 'research', 'three_fold_engine'])
def test_valid_issuer_budget_is_not_a_neural_execution_capability(root, entry):
    authority, record = issuer(root)
    budget = ReservedChannelBudget.from_issuer_reservation(authority, record['reservation_id'])
    with pytest.raises(TypeError, match='worker|child'):
        if entry == 'forward':
            neural_forward._fit_neural_forward_payload(None, None, None, budget)
        elif entry == 'research':
            neural_forward._fit_neural_research_payload(None, None, None, budget)
        else:
            neural_research._run_research_worker(None, None, None, symbols=('X',),
                                                 directory=root / 'unreachable', budget=budget)
    assert not any(budget.counts.values())
    with pytest.raises(AttributeError):
        budget.executor_mode = 'child'


def _borrow_parent_budget(project_root, reservation_id, result):
    import factorlab.protocol
    factorlab.protocol.ROOT = Path(project_root)
    authority = wallet(Path(project_root))
    failures = []
    for factory in (ReservedChannelBudget.from_issuer_reservation, ReservedChannelBudget.from_reservation):
        try:
            factory(authority, reservation_id)
        except (ValueError, RuntimeError) as error:
            failures.append(type(error).__name__)
        else:
            failures.append('accepted')
    result.put(failures)


def test_actual_direct_child_cannot_borrow_issuer_reservation_through_either_factory(root):
    authority, record = issuer(root)
    context = multiprocessing.get_context('spawn')
    queue = context.Queue()
    child = context.Process(target=_borrow_parent_budget, args=(str(root), record['reservation_id'], queue))
    child.start()
    try:
        assert queue.get(timeout=20) == ['ValueError', 'ValueError']
        child.join(timeout=10)
        assert child.exitcode == 0
    finally:
        if child.is_alive():
            child.terminate()
            child.join(timeout=5)
        queue.close()
    assert not (authority.directory / 'child_actual').exists()
    budget = ReservedChannelBudget.from_issuer_reservation(authority, record['reservation_id'])
    budget.consume('solve')
    budget.finalize()
    assert ReservedChannelBudget.verify_counts(authority, record['reservation_id'])['actual_counts']['solve'] == 1


def test_rehashed_executor_mode_tamper_cannot_change_historical_role(root):
    authority, record = issuer(root)
    budget = ReservedChannelBudget.from_issuer_reservation(authority, record['reservation_id'])
    budget.consume('solve')
    budget.finalize()
    paths = [budget._directory / 'claim.json',
             authority.directory / 'child_actual_claims' / (record['reservation_id'] + '.json')]
    for path in paths:
        value = json.loads(path.read_text())
        value['record']['binding']['executor_mode'] = 'child'
        value['hash'] = fingerprint(value['record'])
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='canonical authority'):
        ReservedChannelBudget.verify_counts(authority, record['reservation_id'])


def test_current_source_change_prevents_new_issuer_claim(root):
    authority, record = issuer(root)
    (root / 'factorlab/frozen.py').write_text('version=changed')
    with pytest.raises(ValueError, match='Current source'):
        ReservedChannelBudget.from_issuer_reservation(authority, record['reservation_id'])
