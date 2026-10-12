import copy
import json
import os
import sqlite3
from contextlib import closing
from types import SimpleNamespace

import pytest
from test_v7_neural_forward_launcher import real_ticket_case as real_ticket_case  # noqa: PLC0414

from factorlab.discovery import forward_publication_witness as module
from factorlab.discovery.forward_publication_witness import ForwardPublicationWitness
from factorlab.discovery.work_lease import FrozenWorkLease, LeaseCheck, LeaseClosed
from factorlab.protocol import fingerprint


def proof():
    # Local publication tests do not claim that this synthetic kernel is Linux
    # acceptance. Admission owns independent kernel/observation verification.
    return {'certificate_digest': fingerprint('certificate'), 'receipt_hash': fingerprint('receipt'),
            'model_digest': fingerprint('model'), 'actual_counts_hash': fingerprint('actual'),
            'kernel': {'synthetic': True}, 'observation': {'synthetic': True}}


def completed(case):
    rid = case.ticket.value['reservation_id']
    case.authority.start(rid, {'pid': os.getpid() + 100000, 'created': 1.})
    case.authority.complete(rid, result={'model_digest': fingerprint('model')}, actual_counts={})


def paths(case):
    root = case.ticket.path.parent / 'launch/publication_witness'
    return root, root / 'witness.sqlite'


def test_real_authority_pending_to_publication_completed(real_ticket_case):
    c = real_ticket_case
    witness = ForwardPublicationWitness.prepare(c.ticket)
    root, dbpath = paths(c)
    with closing(sqlite3.connect(dbpath)) as db:
        assert db.execute('SELECT status,proof FROM publication').fetchone() == ('pending', None)
    before = c.authority.verify()
    with pytest.raises(ValueError, match='completed canonical accounting'):
        witness.complete(proof())
    assert c.authority.verify() == before
    completed(c)
    assert witness.complete(proof()) == proof()
    assert ForwardPublicationWitness.read_completed(c.ticket) == proof()
    assert (root / 'identity.json').is_file()
    with pytest.raises(ValueError, match='cannot be repeated'):
        witness.complete(proof())


def test_completed_history_never_requires_old_parent_or_writable_database(real_ticket_case, monkeypatch):
    c = real_ticket_case
    witness = ForwardPublicationWitness.prepare(c.ticket)
    completed(c)
    witness.complete(proof())
    monkeypatch.setattr(FrozenWorkLease, 'assert_parent_alive',
                        lambda *args, **kwargs: pytest.fail('historical proof must not revive its parent'))
    before = c.authority.verify()
    assert ForwardPublicationWitness.read_completed(c.ticket) == proof()
    assert c.authority.verify() == before


def test_sql_written_final_lease_failure_and_unwritable_log_stays_pending(real_ticket_case, monkeypatch):
    c = real_ticket_case
    witness = ForwardPublicationWitness.prepare(c.ticket)
    completed(c)
    original_connect, original_check = module._connect, FrozenWorkLease.assert_parent_alive
    written = []

    def connect(*args, **kwargs):
        db = original_connect(*args, **kwargs)
        db.set_trace_callback(lambda sql: written.append(sql) if sql.startswith('UPDATE publication') else None)
        return db

    def final_expired(self, **kwargs):
        if written:
            raise LeaseClosed(LeaseCheck('deadline'))
        return original_check(self, **kwargs)

    monkeypatch.setattr(module, '_connect', connect)
    monkeypatch.setattr(FrozenWorkLease, 'assert_parent_alive', final_expired)
    monkeypatch.setattr(module, 'immutable_record', lambda *args: (_ for _ in ()).throw(OSError('disk full')))
    with pytest.raises(LeaseClosed, match='deadline'):
        witness.complete(proof())
    with pytest.raises(OSError, match='disk full'):
        module.immutable_record(c.ticket.path.parent / 'launch/failure.json', {'failed': True})
    assert written
    with closing(sqlite3.connect(paths(c)[1])) as db:
        assert db.execute('SELECT status,proof,proof_hash FROM publication').fetchone() == ('pending', None, None)
    with pytest.raises(ValueError, match='pending'):
        ForwardPublicationWitness.read_completed(c.ticket)


def test_no_post_commit_check_or_completed_replay(real_ticket_case, monkeypatch):
    c = real_ticket_case
    witness = ForwardPublicationWitness.prepare(c.ticket)
    completed(c)
    checks = []
    original = FrozenWorkLease.assert_parent_alive

    def check(self, **kwargs):
        checks.append(1)
        if len(checks) > 2:
            pytest.fail('completion must not add a post-COMMIT clock test')
        return original(self, **kwargs)

    monkeypatch.setattr(FrozenWorkLease, 'assert_parent_alive', check)
    assert witness.complete(proof()) == proof()
    assert len(checks) == 2
    assert ForwardPublicationWitness.read_completed(c.ticket) == proof()


@pytest.mark.parametrize('target', ['identity', 'database', 'row', 'anchor'])
def test_missing_identity_database_or_row_never_reinitializes(real_ticket_case, target):
    c = real_ticket_case
    witness = ForwardPublicationWitness.prepare(c.ticket)
    root, dbpath = paths(c)
    if target == 'identity':
        (root / 'identity.json').unlink()
    elif target == 'database':
        dbpath.unlink()
    elif target == 'anchor':
        (c.ticket.path.parent / 'publication_witness_anchor.json').unlink()
    else:
        with closing(sqlite3.connect(dbpath)) as db:
            db.execute('DELETE FROM publication')
            db.commit()
    with pytest.raises(ValueError, match='prepared'):
        ForwardPublicationWitness.prepare(c.ticket)
    completed(c)
    with pytest.raises(ValueError):
        witness.complete(proof())
    with pytest.raises(ValueError):
        ForwardPublicationWitness.read_completed(c.ticket)


def test_lost_whole_launch_cannot_prepare_replacement(real_ticket_case):
    c = real_ticket_case
    ForwardPublicationWitness.prepare(c.ticket)
    folder = c.ticket.path.parent / 'launch'
    folder.rename(c.ticket.path.parent / 'launch-preserved-loss-fixture')
    with pytest.raises(ValueError, match='prepared'):
        ForwardPublicationWitness.prepare(c.ticket)


def test_identity_and_proof_damage_rejected(real_ticket_case):
    c = real_ticket_case
    witness = ForwardPublicationWitness.prepare(c.ticket)
    completed(c)
    witness.complete(proof())
    with closing(sqlite3.connect(paths(c)[1])) as db:
        db.execute("UPDATE publication SET proof_hash='changed'")
        db.commit()
    with pytest.raises(ValueError, match='integrity'):
        ForwardPublicationWitness.read_completed(c.ticket)


def test_other_ticket_or_fake_ticket_cannot_borrow_witness(real_ticket_case):
    c = real_ticket_case
    ForwardPublicationWitness.prepare(c.ticket)
    with pytest.raises(TypeError, match='real canonical'):
        ForwardPublicationWitness.prepare(SimpleNamespace(path=c.ticket.path, value=c.ticket.value, digest=c.ticket.digest))
    # Modifying the in-memory real ticket cannot relabel immutable publication.
    original = copy.deepcopy(c.ticket.value)
    c.ticket.value['source_hash'] = fingerprint('changed')
    with pytest.raises(ValueError, match='modified'):
        ForwardPublicationWitness.read_completed(c.ticket)
    c.ticket.value.clear()
    c.ticket.value.update(original)


@pytest.mark.parametrize('kind', ['field', 'hash', 'boolean', 'nan', 'oversized'])
def test_complete_requires_exact_bounded_proof_schema(real_ticket_case, kind):
    c = real_ticket_case
    witness = ForwardPublicationWitness.prepare(c.ticket)
    completed(c)
    value = proof()
    if kind == 'field':
        value['verified'] = True
    elif kind == 'hash':
        value['certificate_digest'] = 'not-a-hash'
    elif kind == 'boolean':
        value['observation'] = True
    elif kind == 'nan':
        value['kernel']['memory'] = float('nan')
    else:
        value['kernel']['data'] = 'x' * 65536
    with pytest.raises(ValueError):
        witness.complete(value)
    with pytest.raises(ValueError, match='pending'):
        ForwardPublicationWitness.read_completed(c.ticket)


def test_expired_prepare_cannot_create_any_publication_record(real_ticket_case, monkeypatch):
    c = real_ticket_case

    def expired(*args, **kwargs):
        raise LeaseClosed(LeaseCheck('deadline'))

    monkeypatch.setattr(FrozenWorkLease, 'assert_parent_alive', expired)
    with pytest.raises(LeaseClosed):
        ForwardPublicationWitness.prepare(c.ticket)
    assert not paths(c)[0].exists()
    assert not (c.ticket.path.parent / 'publication_witness_anchor.json').exists()


def test_identity_binds_all_canonical_parent_and_authority_fields(real_ticket_case):
    c = real_ticket_case
    ForwardPublicationWitness.prepare(c.ticket)
    root, _ = paths(c)
    wrapper = json.loads((root / 'identity.json').read_text())
    identity = wrapper['identity']
    assert wrapper['hash'] == fingerprint(identity)
    for key in ('authority_manifest_hash', 'reservation_id', 'lease_hash', 'source_hash'):
        assert identity[key] == c.ticket.value[key]
    assert identity['ticket_digest'] == c.ticket.digest
    assert identity['parent_pid'] == c.ticket.value['lease']['parent_pid']
    assert identity['parent_created'] == c.ticket.value['lease']['parent_created']
