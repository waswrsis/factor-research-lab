"""Transactional final publication witness for the fixed neural launcher.

Pending is committed before launch. The final proof update and original lease
check occur in one SQLite transaction, with COMMIT as the publication
linearization point. Failure before commit leaves pending without depending on
a writable failure log. This is not a nanosecond/physical-fsync deadline promise
or authentication against an administrator replacing all local trusted records.
Kernel/observation semantics are separately checked by the admission verifier.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import closing

from filelock import FileLock

from factorlab.protocol import fingerprint, project_path

from .v7_storage import immutable_record
from .work_lease import FrozenWorkLease

VERSION = 'forward-publication-witness-v1'
PROOF_FIELDS = {'certificate_digest', 'receipt_hash', 'model_digest', 'actual_counts_hash',
                'kernel', 'observation'}


def _proof(value):
    if not isinstance(value, dict) or set(value) != PROOF_FIELDS:
        raise ValueError('Publication proof requires exactly six canonical fields')
    for name in PROOF_FIELDS - {'kernel', 'observation'}:
        if not isinstance(value[name], str) or re.fullmatch('[0-9a-f]{64}', value[name]) is None:
            raise ValueError('Publication proof requires canonical SHA256 digests')
    for name in ('kernel', 'observation'):
        if not isinstance(value[name], dict) or not value[name]:
            raise ValueError('Publication proof requires kernel and clock observations')
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    if len(encoded.encode()) > 65536:
        raise ValueError('Publication proof exceeds 64 KiB')
    return json.loads(encoded)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _read(path):
    try:
        if path.stat().st_size > 65536:
            raise ValueError('Oversized witness identity')
        return json.loads(path.read_text('utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError('Missing or damaged publication witness identity') from exc


def _canonical(ticket):
    # Lazy import avoids admission/launcher import cycles. A caller cannot supply
    # a look-alike ticket with a convenient validator or self-reported identity.
    from .known_linear_forward import KnownForwardTicket
    from .neural_forward_ticket import ForwardFitTicket, open_existing_authority
    from .pool_forward import PoolForwardTicket
    if type(ticket) not in {ForwardFitTicket, KnownForwardTicket, PoolForwardTicket}:
        raise TypeError('A real canonical forward fit ticket is required')
    if type(ticket).load(ticket.path).digest != ticket.digest:
        raise ValueError('Publication ticket differs from frozen bytes')
    value = ticket.value
    authority = open_existing_authority(value['authority_directory'])
    record = authority.get(value['reservation_id'])
    if type(ticket) in {KnownForwardTicket, PoolForwardTicket}:
        from .forward_preparation import ForwardPreparation
        binding_valid = ForwardPreparation.verify_ticket_reference(ticket.path, value) == record
    else:
        from .neural_forward_ticket import verify_prepaid_ticket
        binding_valid = verify_prepaid_ticket(ticket, authority, record) == record
    if (any(record[k] != value[k] for k in ('authority_manifest_hash', 'terms_hash', 'lease', 'lease_hash'))
            or not binding_valid
            or fingerprint(authority.manifest['source_files']) != value['source_hash']):
        raise ValueError('Publication ticket differs from canonical authority')
    lease = FrozenWorkLease.from_dict(record['lease'])
    directory = ticket.path.parent / 'launch' / 'publication_witness'
    if project_path(directory) != directory:
        raise ValueError('Publication witness directory was redirected')
    anchor = ticket.path.parent / 'publication_witness_anchor.json'
    if project_path(anchor) != anchor:
        raise ValueError('Publication witness anchor was redirected')
    identity = {'version': VERSION, 'ticket_digest': ticket.digest, 'ticket_path': str(ticket.path),
                'directory': str(directory), 'authority_directory': str(authority.directory),
                'authority_manifest_hash': record['authority_manifest_hash'],
                'reservation_id': record['reservation_id'], 'lease_hash': lease.digest,
                'source_hash': value['source_hash'], 'parent_pid': lease.parent_pid,
                'parent_created': lease.parent_created}
    return identity, anchor, directory, record, lease


def _connect(path, *, readonly=False):
    if not path.is_file():
        raise ValueError('Publication witness database missing; cannot initialize on recovery')
    try:
        return sqlite3.connect(path.as_uri() + ('?mode=ro' if readonly else '?mode=rw'), uri=True, timeout=10)
    except sqlite3.Error as exc:
        raise ValueError('Publication witness database cannot be opened') from exc


def _row(db, identity):
    try:
        rows = db.execute('SELECT id,identity_hash,status,'
                          'CASE WHEN length(proof)<=65536 THEN proof ELSE NULL END,proof_hash '
                          'FROM publication LIMIT 2').fetchall()
    except sqlite3.Error as exc:
        raise ValueError('Missing or damaged publication witness table') from exc
    if len(rows) != 1 or rows[0][0] != 1 or rows[0][1] != fingerprint(identity):
        raise ValueError('Publication witness singleton missing or identity changed')
    _, _, state, text, digest = rows[0]
    if state == 'pending':
        if text is not None or digest is not None:
            raise ValueError('Pending publication contains an uncommitted proof')
        return state, None
    if state != 'completed' or not isinstance(text, str) or len(text.encode()) > 65536:
        raise ValueError('Invalid publication witness state')
    try:
        proof = _proof(json.loads(text))
    except (ValueError, TypeError) as exc:
        raise ValueError('Invalid publication witness proof') from exc
    if digest != fingerprint(proof) or text != _json(proof):
        raise ValueError('Publication witness proof integrity failure')
    return state, proof


def _files(identity, anchor, directory):
    expected = {'identity': identity, 'hash': fingerprint(identity)}
    if _read(anchor) != expected or _read(directory / 'identity.json') != expected:
        raise ValueError('Publication witness frozen identity differs')
    if not (directory / 'witness.sqlite').is_file():
        raise ValueError('Publication witness database missing; cannot reset')


class ForwardPublicationWitness:
    """Trusted launcher-only publication; historical verification is read-only."""

    def __init__(self, *args, **kwargs):
        raise TypeError('Use prepare with an actual frozen ticket')

    @classmethod
    def prepare(cls, ticket):
        identity, anchor, directory, record, lease = _canonical(ticket)
        if record['status'] not in {'reserved', 'running'}:
            raise ValueError('Publication pending must precede accounting completion')
        if os.getpid() != lease.parent_pid:
            raise ValueError('Only the original launcher may prepare publication')
        lease.assert_parent_alive()
        with FileLock(str(anchor) + '.prepare.lock', timeout=10):
            if anchor.exists() or directory.exists():
                raise ValueError('Publication witness already prepared; no reset or replay')
            wrapper = {'identity': identity, 'hash': fingerprint(identity)}
            # Independent anchor survives loss of the whole launch directory.
            immutable_record(anchor, wrapper)
            directory.mkdir(parents=True)
            immutable_record(directory / 'identity.json', wrapper)
            with closing(sqlite3.connect(directory / 'witness.sqlite')) as db:
                db.execute('PRAGMA journal_mode=DELETE')
                db.execute('PRAGMA synchronous=FULL')
                db.execute('CREATE TABLE publication (id INTEGER PRIMARY KEY CHECK(id=1), '
                           'identity_hash TEXT NOT NULL,status TEXT NOT NULL,proof TEXT,proof_hash TEXT)')
                db.execute('BEGIN IMMEDIATE')
                db.execute('INSERT INTO publication VALUES (1,?,?,NULL,NULL)', (fingerprint(identity), 'pending'))
                db.commit()
        lease.assert_parent_alive()
        result = object.__new__(cls)
        result._ticket = ticket
        result._identity = identity
        result._identity_hash = fingerprint(identity)
        return result

    def complete(self, proof):
        proof = _proof(proof)
        identity, anchor, directory, record, lease = _canonical(self._ticket)
        if identity != self._identity or fingerprint(self._identity) != self._identity_hash:
            raise ValueError('In-memory publication witness identity changed')
        _files(identity, anchor, directory)
        if record['status'] != 'completed':
            raise ValueError('Publication requires completed canonical accounting')
        if os.getpid() != lease.parent_pid:
            raise ValueError('Only the original launcher may complete publication')
        lease.assert_parent_alive()
        with closing(_connect(directory / 'witness.sqlite')) as db:
            try:
                db.execute('PRAGMA synchronous=FULL')
                db.execute('BEGIN IMMEDIATE')
                status, _ = _row(db, identity)
                if status != 'pending':
                    raise ValueError('Publication completion cannot be repeated')
                db.execute('UPDATE publication SET status=?,proof=?,proof_hash=? WHERE id=1',
                           ('completed', _json(proof), fingerprint(proof)))
                # This check is deliberately AFTER the SQL write and BEFORE
                # COMMIT. Any exception rolls back to pending even if failure
                # logging is unavailable. Do not add a post-COMMIT lease check.
                lease.assert_parent_alive()
                db.commit()
            except BaseException:
                db.rollback()
                raise
        return proof

    @classmethod
    def read_completed(cls, ticket):
        identity, anchor, directory, record, _ = _canonical(ticket)
        _files(identity, anchor, directory)
        if record['status'] != 'completed':
            raise ValueError('Publication lacks completed canonical accounting')
        with closing(_connect(directory / 'witness.sqlite', readonly=True)) as db:
            status, proof = _row(db, identity)
        if status != 'completed':
            raise ValueError('Publication is pending; historical admission is forbidden')
        return proof
