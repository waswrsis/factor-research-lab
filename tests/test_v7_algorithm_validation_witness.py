"""Witness transaction mechanics with actual wallet/lease; no admission claim."""
import json
import sqlite3
from types import SimpleNamespace

import pytest
from test_v7_algorithm_validation import case, module  # noqa: F401

from factorlab.protocol import fingerprint


@pytest.fixture
def witness(case):  # noqa: F811
    bridge, _, _, _, runtime = case
    # This private SQL transaction fixture is not a standard-lane forward
    # allocation. Its isolated diagnostic wallet never enters public admission.
    from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
    authority = MonthlyWorkAuthority(runtime.directory.parent / 'witness-unit-wallet',
        run_id='private-witness-mechanics', task='volatility', universe=['X'])
    record = authority.reserve('witness-transaction-only', counts={'solve': 1},
        binding={'purpose': 'private_unit_witness_mechanics'}, lease_seconds=300)
    # Private witness unit shape does NOT pass public _Ticket.verify/admit.
    ticket = SimpleNamespace(path=runtime.directory/'unit-ticket/prepared.json', digest=fingerprint('unit'),
        value={'reservation_id': record['reservation_id'], 'lease': record['lease'],
            'lease_hash': record['lease_hash'], 'authority_manifest_hash': record['authority_manifest_hash']})
    producer = module._producer()
    proof = {'receipt_hash': fingerprint('receipt'), 'model_digest': fingerprint('model'),
        'k0_digest': fingerprint('k0'), 'actual_counts_hash': fingerprint('counts'),
        'kernel': {}, 'observation': {}}
    return producer._Witness(ticket, prepare=True), proof, producer


def test_SQL_update_then_original_lease_failure_leaves_pending_without_failure_log(witness, monkeypatch):
    item, proof, producer = witness
    def fail(*a, **k):
        raise ValueError('original deadline closed after SQL UPDATE')
    monkeypatch.setattr(producer.FrozenWorkLease, 'assert_parent_alive', fail)
    with pytest.raises(ValueError, match='after SQL'):
        item.complete(proof)
    with sqlite3.connect(item.path) as db:
        assert db.execute('SELECT status,proof,hash FROM witness').fetchall() == [('pending', None, None)]
    assert not (item.ticket.path.parent/'failure.json').exists()
    with pytest.raises(ValueError, match='no completed'):
        item.read()
    with pytest.raises(ValueError, match='cannot reset'):
        producer._Witness(item.ticket, prepare=True)


@pytest.mark.parametrize('target', ['identity', 'database', 'row'])
def test_missing_publication_state_cannot_reset(witness, target):
    item, proof, producer = witness
    item.complete(proof)
    if target == 'identity':
        (item.directory/'identity.json').unlink()
    elif target == 'database':
        item.path.unlink()
    else:
        with sqlite3.connect(item.path) as db:
            db.execute('DELETE FROM witness')
    with pytest.raises(ValueError):
        item.read()
    with pytest.raises(ValueError, match='cannot reset'):
        producer._Witness(item.ticket, prepare=True)


def test_completed_is_readonly_and_never_reissues_or_rechecks_old_lease(witness, monkeypatch):
    item, proof, producer = witness
    item.complete(proof)
    before = item.path.read_bytes()
    monkeypatch.setattr(producer.FrozenWorkLease, 'assert_parent_alive',
        lambda *a, **k: (_ for _ in ()).throw(AssertionError('historical clock checked')))
    assert item.read() == json.loads(json.dumps(proof))
    assert item.path.read_bytes() == before
    with pytest.raises(ValueError, match='once only'):
        item.complete(proof)
