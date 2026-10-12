"""One prepaid UTC-month wallet; this module does not yet wire existing callers.

SQLite owns counters, immutable reservations and state in one IMMEDIATE transaction.
External intent precedes commit; an external event mirror follows commit. A committed
event can repair its missing mirror. An uncommitted intent instead blocks recovery:
it never grants a replacement lease or silently refunds ambiguous work. These local
anchors detect lost/rolled-back files, not a hostile administrator replacing all
copies. Work keys must be stable business identities supplied by trusted controllers.
The lease is the sole time authority; loading/completing a reservation cannot renew it.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import closing, contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from threading import get_ident

from filelock import FileLock

from factorlab.protocol import fingerprint, project_path

from .evidence import file_hash, snapshot_source, source_paths
from .monthly_partition_policy import canonical_policy
from .monthly_partition_policy import charge as charge_partition
from .monthly_partition_policy import route as allocation_route
from .v7_storage import immutable_record
from .work_lease import FrozenWorkLease

DEFAULT_LIMITS = {'ast': 96, 'f': 24, 'gpt_attempt': 6, 'family': 3,
                  'solve': 1600, 'neural_fit': 18, 'epoch': 180, 'tree_fit': 3}
TERMINAL = {'completed', 'failed_consumed', 'interrupted_consumed'}
VERSION = 'monthly-work-authority-v2'


# Only recursive immutable-identity reads may use this active verification stack.
# This is not a cached wallet result and never replaces explicit verify/get/status.
_IDENTITY_READ_SCOPE = ContextVar('monthly_authority_identity_read_scope', default=None)
_CHARGE_READ_SCOPE = ContextVar('monthly_authority_charge_read_scope', default=None)


def _charge_scope(authority):
    """Only the live, same-thread locked debit may reuse its verified inventory."""
    scope = _CHARGE_READ_SCOPE.get()
    if scope is None:
        return None
    if not scope['active'] or scope['thread'] != get_ident():
        raise ValueError('Expired or foreign-thread authority charge read')
    owner = scope['authority']
    if (authority.database != owner.database or authority.directory != owner.directory
            or authority.index_path != owner.index_path
            or authority.manifest_hash != scope['manifest_hash']
            or fingerprint(authority.manifest) != scope['manifest_hash']
            or authority.limits != scope['limits']
            or str(project_path('.')) != scope['source_root']
            or _database_identity(owner, scope['db']) != scope['database_identity']
            or scope['db'].total_changes != scope['total_changes']):
        raise ValueError('Locked authority charge identity or database changed')
    return scope


def _database_identity(authority, db):
    if not db.in_transaction:
        raise ValueError('Nested authority read lost its original transaction')
    main = [row[2] for row in db.execute('PRAGMA database_list') if row[1] == 'main']
    if len(main) != 1 or project_path(main[0]) != authority.database:
        raise ValueError('Nested authority read changed its original database')
    stat = authority.database.stat()
    return (str(authority.database), stat.st_dev, stat.st_ino)


def _nested_identity_read(authority):
    """Private identity accessor for the same locked wallet, never a work grant.

    The owning verifier still audits all rows, mirrors, allocation provenance and
    quota projections. Different wallets/threads and expired copied contexts must
    do their own ordinary verification. No caller flag or persistent cache exists.
    """
    scope = _IDENTITY_READ_SCOPE.get()
    if (scope is None or not scope['active'] or scope['thread'] != get_ident()
            or authority.database != scope['authority'].database):
        return False
    owner = scope['authority']
    if (_database_identity(owner, scope['db']) != scope['database_identity']
            or authority.directory != owner.directory
            or authority.index_path != owner.index_path
            or authority.manifest_hash != owner.manifest_hash
            or authority.manifest != owner.manifest or authority.limits != owner.limits):
        raise ValueError('Nested authority immutable identity changed')
    # The charge scope checks all source/identity bytes at its entry and exit.
    # Ordinary verification retains its original recursive identity checks.
    if _charge_scope(authority) is None:
        owner._verify_identity()
        authority._verify_identity()
    return True


class BudgetExhausted(RuntimeError):
    """No reservation was created; no work may start."""


def _charge_current_source():
    # The only owner is one original debit under the bridge lock and authority
    # transaction. Both boundary audits read current source bytes afresh.
    scope = _CHARGE_READ_SCOPE.get()
    if scope is None or scope['verified'] is None:
        return None
    checked = _charge_scope(scope['authority'])
    return dict(checked['authority'].manifest['source_files'])


def source_files(*, _fresh=False):
    cached = None if _fresh else _charge_current_source()
    if cached is not None:
        return cached
    root = project_path('.')
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(source_paths(root))}


def _text(value, label, maximum=256):
    if not isinstance(value, str) or not value or len(value) > maximum or any(ord(c) < 32 for c in value):
        raise ValueError('Invalid ' + label)
    return value


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _object(value, label):
    if not isinstance(value, dict) or not value:
        raise ValueError(label + ' must be a nonempty object')
    encoded = _json(value)
    if len(encoded.encode()) > 16384:
        raise ValueError(label + ' exceeds 16 KiB')
    return json.loads(encoded)


def _counts(value):
    if not isinstance(value, dict) or set(value) - set(DEFAULT_LIMITS):
        raise ValueError('Unknown quota counters')
    if any(type(v) is not int or v < 0 for v in value.values()):
        raise ValueError('Quota counters must be nonnegative integers')
    return {k: value.get(k, 0) for k in DEFAULT_LIMITS}


def _month(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9]{4}-(0[1-9]|1[0-2])', value) or value[:4] == '0000':
        raise ValueError('Invalid UTC work month')
    return value


def _read(path):
    try:
        return json.loads(path.read_text('utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError('Missing or damaged authority record: ' + str(path)) from exc


def _wrapped(value):
    return {'record': value, 'hash': fingerprint(value)}


class MonthlyWorkAuthority:
    """Authority scope is one run/task/universe, shared by all work channels.

    Opening another directory for the same scope is rejected by a project-wide
    immutable index. Default ceilings may only be lowered. A new UTC month gets
    its wallet only inside a committed reservation, never via a reset operation.
    """

    def __init__(self, directory, *, run_id, task, universe, limits=None, monthly_seconds=7200,
                 partition_policy=None):
        self.directory = project_path(directory)
        self.database = self.directory / 'authority.sqlite'
        if isinstance(universe, str) or not isinstance(universe, (list, tuple)) or not 1 <= len(universe) <= 3:
            raise ValueError('Universe must contain one to three unique symbols')
        symbols = sorted(_text(x, 'symbol', 32) for x in universe)
        if len(set(symbols)) != len(symbols):
            raise ValueError('Duplicate universe symbol')
        limits = DEFAULT_LIMITS.copy() if limits is None else _counts(limits)
        if any(limits[k] > DEFAULT_LIMITS[k] for k in limits):
            raise ValueError('Monthly quota exceeds frozen protocol ceiling')
        if type(monthly_seconds) is not int or not 1 <= monthly_seconds <= 7200:
            raise ValueError('Monthly prepaid seconds must be 1..7200')
        policy = canonical_policy(partition_policy)
        if policy is not None and (sum(v['seconds'] for v in policy['lanes'].values()) > monthly_seconds
                or any(sum(v['counts'][k] for v in policy['lanes'].values()) > limits[k] for k in limits)):
            raise ValueError('Frozen partitions exceed actual global wallet ceilings')
        scope = {'run_id': _text(run_id, 'run_id'), 'task': _text(task, 'task'), 'universe': symbols}
        self.index_path = project_path('work/monthly_work_authority_index') / (fingerprint(scope) + '.json')
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        expected = {'version': VERSION, 'scope': scope, 'directory': str(self.directory),
                    'limits': limits, 'monthly_seconds': monthly_seconds, 'source_files': source_files(),
                    'partition_policy': policy}
        with FileLock(str(self.index_path) + '.init.lock', timeout=30):
            manifest_path = self.directory / 'manifest.json'
            if manifest_path.exists():
                if _read(manifest_path) != _wrapped(expected):
                    raise ValueError('Authority manifest/source identity changed')
                if not self.database.is_file() or not self.index_path.is_file():
                    raise ValueError('Authority database/index missing; cannot reset wallet')
            else:
                if self.index_path.exists() or (self.directory.exists() and any(self.directory.iterdir())):
                    raise ValueError('Existing authority evidence cannot initialize a new wallet')
                self.directory.mkdir(parents=True, exist_ok=True)
                immutable_record(manifest_path, _wrapped(expected))
                immutable_record(self.index_path, {'directory': str(self.directory), 'manifest_hash': fingerprint(expected)})
                archived = snapshot_source(self.directory / 'source')
                if archived['files'] != expected['source_files']:
                    raise ValueError('Source changed while freezing authority')
                with closing(sqlite3.connect(self.database)) as db:
                    db.execute('PRAGMA journal_mode=WAL')
                    db.execute('PRAGMA synchronous=FULL')
                    db.executescript('''CREATE TABLE wallets(month TEXT PRIMARY KEY, body TEXT NOT NULL);
                        CREATE TABLE reservations(id TEXT PRIMARY KEY, work_key TEXT UNIQUE NOT NULL,
                                                  body TEXT NOT NULL);
                        CREATE TABLE partition_wallets(id TEXT PRIMARY KEY, body TEXT NOT NULL);
                        CREATE TABLE monthly_owners(id TEXT PRIMARY KEY, body TEXT NOT NULL);
                        CREATE TABLE events(seq INTEGER PRIMARY KEY, body TEXT NOT NULL);''')
                    db.commit()
            self.manifest = expected
            self.manifest_hash = fingerprint(expected)
            self.limits = limits.copy()
            with self._transaction() as db:
                self._verify(db, repair=True)

    @contextmanager
    def _transaction(self):
        if _CHARGE_READ_SCOPE.get() is not None:
            raise ValueError('No new authority transaction during a locked charge read')
        # mode=rw prevents sqlite's otherwise automatic missing-file recreation.
        if not self.database.is_file():
            raise ValueError('Authority database missing; prepaid work cannot be reset')
        db = sqlite3.connect(self.database.as_uri() + '?mode=rw', uri=True, timeout=30)
        try:
            db.execute('PRAGMA synchronous=FULL')
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @contextmanager
    def _charge_read_scope(self):
        """Fresh locked inventory with full entry/exit audits for one debit."""
        import sys
        from .work_authority_bridge import AuthorityWorkBudget
        # contextlib.__enter__ advances this generator on the original caller.
        caller = sys._getframe(2)
        budget = caller.f_locals.get('self')
        if (caller.f_code is not AuthorityWorkBudget._charge.__code__
                or type(budget) is not AuthorityWorkBudget
                or budget.bridge.authority is not self or not budget.bridge._lock.is_locked
                or _CHARGE_READ_SCOPE.get() is not None):
            raise ValueError('Only the original locked bridge debit may share authority reads')
        with self._transaction() as db:
            self._verify_identity()
            scope = {'authority': self, 'db': db, 'thread': get_ident(), 'active': True,
                     'database_identity': _database_identity(self, db),
                     'manifest_hash': self.manifest_hash, 'limits': self.limits.copy(),
                     'source_root': str(project_path('.')),
                     'total_changes': db.total_changes, 'verified': None}
            identity_token = _IDENTITY_READ_SCOPE.set(scope)
            charge_token = _CHARGE_READ_SCOPE.set(scope)
            try:
                scope['verified'] = self._verify_rows(db)
                yield
            finally:
                try:
                    try:
                        _charge_scope(self)
                        if (scope['verified'] is not None
                                and self._verify_rows(db) != scope['verified']):
                            raise ValueError('Authority ledger changed during a locked bridge debit')
                    finally:
                        self._verify_identity()
                finally:
                    scope['active'] = False
                    _CHARGE_READ_SCOPE.reset(charge_token)
                    _IDENTITY_READ_SCOPE.reset(identity_token)

    def _charge_inventory(self):
        scope = _charge_scope(self)
        if scope is None:
            return None
        if scope['verified'] is None:
            raise ValueError('Authority inventory is not verified yet')
        return json.loads(_json(scope['verified']))

    def _verify_identity(self):
        if (self.manifest.get('version') != VERSION or 'partition_policy' not in self.manifest
                or canonical_policy(self.manifest['partition_policy']) != self.manifest['partition_policy']):
            raise ValueError('Unsupported authority version or changed frozen partition policy')
        if fingerprint(self.manifest) != self.manifest_hash or self.limits != self.manifest['limits']:
            raise ValueError('In-memory authority identity changed')
        if _read(self.directory / 'manifest.json') != _wrapped(self.manifest):
            raise ValueError('Authority manifest integrity failure')
        if _read(self.index_path) != {'directory': str(self.directory), 'manifest_hash': self.manifest_hash}:
            raise ValueError('Authority scope index changed')
        if source_files(_fresh=True) != self.manifest['source_files']:
            raise ValueError('Current source differs from authority frozen source')
        if _read(self.directory / 'source/manifest.json') != self.manifest['source_files']:
            raise ValueError('Authority archived source manifest changed')
        for name, digest in self.manifest['source_files'].items():
            path = self.directory / 'source' / name
            if not path.is_file() or file_hash(path) != digest:
                raise ValueError('Authority archived source missing or changed')

    @contextmanager
    def _identity_read_scope(self, db):
        self._verify_identity()
        scope = {'authority': self, 'db': db, 'thread': get_ident(), 'active': True,
                 'database_identity': _database_identity(self, db)}
        token = _IDENTITY_READ_SCOPE.set(scope)
        try:
            yield
            if _database_identity(self, db) != scope['database_identity']:
                raise ValueError('Original authority database changed during verification')
            self._verify_identity()
        finally:
            # A copy_context retained elsewhere cannot resurrect this scope.
            scope['active'] = False
            _IDENTITY_READ_SCOPE.reset(token)

    def _verify(self, db, *, repair=False):
        with self._identity_read_scope(db):
            return self._verify_rows(db, repair=repair)

    def _verify_rows(self, db, *, repair=False):
        reservations, wallets, expected_names, work_keys = {}, {}, set(), set()
        # One verifier invocation only. Never an object/global/cross-call cache.
        provenance_cache = {}
        partitions, owners = {}, {}
        head, sequence = None, 0
        for seq, raw in db.execute('SELECT seq,body FROM events ORDER BY seq'):
            event = json.loads(raw)
            body = event.get('record', {})
            if (seq != sequence + 1 or event != _wrapped(body) or body.get('seq') != seq
                    or body.get('previous') != head or body.get('authority_manifest_hash') != self.manifest_hash):
                raise ValueError('Authority event hash chain damaged')
            name = f'{seq:010d}.json'
            expected_names.add(name)
            if _read(self.directory / 'intents' / name) != event:
                raise ValueError('Authority reservation intent differs')
            mirror = self.directory / 'events' / name
            if mirror.exists():
                if _read(mirror) != event:
                    raise ValueError('Authority external event differs')
            elif repair:
                immutable_record(mirror, event)
            else:
                raise ValueError('Committed event mirror pending; reopen/recover authority')
            record = body.get('reservation', {})
            rid = record.get('reservation_id')
            old = reservations.get(rid)
            if body.get('kind') == 'reserved':
                self._validate_reserved(record)
                if old is not None or record['work_key'] in work_keys:
                    raise ValueError('Duplicate charged work identity')
                work_keys.add(record['work_key'])
                month = record['actual_month']
                wallet = wallets.setdefault(month, {'counts': _counts({}), 'seconds': 0})
                for key in DEFAULT_LIMITS:
                    wallet['counts'][key] += record['counts'][key]
                wallet['seconds'] += record['lease']['lease_seconds']
                if (any(wallet['counts'][k] > self.limits[k] for k in self.limits)
                        or wallet['seconds'] > self.manifest['monthly_seconds']):
                    raise ValueError('Authority committed quota overflow')
                if self.manifest['partition_policy'] is not None:
                    if allocation_route(self, record['work_key'], record['binding'], current=False,
                                        lease=FrozenWorkLease.from_dict(record['lease']), debit=record['counts'],
                                        verification_cache=provenance_cache) != record['allocation']:
                        raise ValueError('Original monthly allocation provenance changed')
                    try:
                        charge_partition(self.manifest['partition_policy'], partitions, owners, month,
                                         record['allocation'], record['counts'], record['lease']['lease_seconds'])
                    except BudgetExhausted as exc:
                        raise ValueError('Committed partition quota overflow') from exc
            else:
                if old is None or record.get('terms_hash') != old['terms_hash']:
                    raise ValueError('Authority transition without original reservation')
                fixed = set(old) - {'status', 'process_identity', 'result', 'actual_counts', 'completed_at', 'reason'}
                if any(record.get(k) != old[k] for k in fixed):
                    raise ValueError('Original reservation changed during transition')
                allowed = {'reserved': {'running', 'failed_consumed', 'interrupted_consumed'},
                           'running': TERMINAL}
                if record.get('status') not in allowed.get(old['status'], set()) or body['kind'] != record['status']:
                    raise ValueError('Invalid authority state transition')
                if record['status'] == 'completed':
                    self._validate_result(record)
            reservations[rid] = record
            head, sequence = event['hash'], seq
        for folder in ('intents', 'events'):
            if {p.name for p in (self.directory / folder).glob('*.json')} != expected_names:
                raise ValueError('Orphan authority evidence or uncommitted intent; cannot reset/retry')
        actual = {rid: json.loads(raw) for rid, raw in db.execute('SELECT id,body FROM reservations')}
        if actual != reservations:
            raise ValueError('Authority reservation projection missing or changed')
        if {m: json.loads(raw) for m, raw in db.execute('SELECT month,body FROM wallets')} != wallets:
            raise ValueError('Authority wallet projection missing or changed')
        if {k: json.loads(raw) for k, raw in db.execute('SELECT id,body FROM partition_wallets')} != partitions:
            raise ValueError('Authority partition projection missing or changed')
        if {k: json.loads(raw) for k, raw in db.execute('SELECT id,body FROM monthly_owners')} != owners:
            raise ValueError('Authority monthly owner projection missing or changed')
        indexed = {rid: key for rid, key in db.execute('SELECT id,work_key FROM reservations')}
        if indexed != {rid: r['work_key'] for rid, r in reservations.items()}:
            raise ValueError('Authority work index changed')
        return {'head': head, 'sequence': sequence, 'reservations': reservations, 'wallets': wallets,
                'partitions': partitions, 'owners': owners}

    def _validate_reserved(self, r):
        lease = FrozenWorkLease.from_dict(r['lease'])
        if (r['authority_manifest_hash'] != self.manifest_hash or r['authority_directory'] != str(self.directory)
                or r['limits'] != self.limits or r['status'] != 'reserved' or r['lease_hash'] != lease.digest
                or r['actual_month'] != lease.wall_started_at[:7]
                or r['reservation_id'] != fingerprint({'authority': self.manifest_hash, 'work_key': r['work_key']})
                or r['counts'] != _counts(r['counts']) or not any(r['counts'].values())
                or r['terms_hash'] != fingerprint({k: v for k, v in r.items() if k not in {'terms_hash', 'status'}})):
            raise ValueError('Invalid original authority reservation')
        _text(r['work_key'], 'work_key')
        _object(r['binding'], 'binding')
        _month(r['actual_month'])
        if (self.manifest['partition_policy'] is None) != ('allocation' not in r):
            raise ValueError('Reservation omitted or invented its frozen allocation')

    @staticmethod
    def _validate_result(r):
        actual = _counts(r['actual_counts'])
        if actual != r['actual_counts'] or any(actual[k] > r['counts'][k] for k in actual):
            raise ValueError('Actual usage exceeds prepaid reservation')
        _object(r['result'], 'result')
        done = datetime.fromisoformat(r['completed_at'])
        if done.tzinfo is None or done.utcoffset().total_seconds() != 0:
            raise ValueError('Completion must be actual UTC time')
        if not datetime.fromisoformat(r['lease']['wall_started_at']) <= done < datetime.fromisoformat(r['lease']['wall_deadline']):
            raise ValueError('Completion outside original lease')

    def _commit_event(self, db, verified, record, kind):
        if _CHARGE_READ_SCOPE.get() is not None:
            raise ValueError('Authority mutation is forbidden during a locked charge read')
        seq = verified['sequence'] + 1
        body = {'seq': seq, 'previous': verified['head'], 'authority_manifest_hash': self.manifest_hash,
                'kind': kind, 'reservation': record}
        event = _wrapped(body)
        immutable_record(self.directory / 'intents' / f'{seq:010d}.json', event)
        if kind == 'reserved':
            month = record['actual_month']
            wallet = verified['wallets'].get(month, {'counts': _counts({}), 'seconds': 0})
            wallet = {'counts': {k: wallet['counts'][k] + record['counts'][k] for k in self.limits},
                      'seconds': wallet['seconds'] + record['lease']['lease_seconds']}
            db.execute('INSERT OR REPLACE INTO wallets VALUES (?,?)', (month, _json(wallet)))
            db.execute('INSERT INTO reservations VALUES (?,?,?)',
                       (record['reservation_id'], record['work_key'], _json(record)))
            if self.manifest['partition_policy'] is not None:
                charge_partition(self.manifest['partition_policy'], verified['partitions'], verified['owners'],
                                 month, record['allocation'], record['counts'], record['lease']['lease_seconds'])
                for key, value in verified['partitions'].items():
                    db.execute('INSERT OR REPLACE INTO partition_wallets VALUES (?,?)', (key, _json(value)))
                for key, value in verified['owners'].items():
                    db.execute('INSERT OR REPLACE INTO monthly_owners VALUES (?,?)', (key, _json(value)))
        else:
            db.execute('UPDATE reservations SET body=? WHERE id=?', (_json(record), record['reservation_id']))
        db.execute('INSERT INTO events VALUES (?,?)', (seq, _json(event)))
        if kind in {'reserved', 'running', 'completed'}:
            # Preparing the payload/intent/SQL may itself consume the lease.
            # An expired uncommitted intent blocks recovery, never grants retry.
            FrozenWorkLease.from_dict(record['lease']).assert_parent_alive()
        # Commit while retaining the connection. Its subsequent context commit is
        # a no-op. A new caller may recover the missing mirror from intent+SQL.
        db.commit()
        immutable_record(self.directory / 'events' / f'{seq:010d}.json', event)
        return json.loads(_json(record))

    def reserve(self, work_key, *, counts, binding, lease_seconds=300, retry_id=None):
        return self._reserve(work_key, counts=counts, binding=binding,
                             lease_seconds=lease_seconds, retry_id=retry_id, lease=None)

    def _reserve_with_lease(self, work_key, *, counts, binding, lease, retry_id=None):
        """Trusted launcher only: keep the lease issued before payload serialization."""
        if not isinstance(lease, FrozenWorkLease) or os.getpid() != lease.parent_pid:
            raise ValueError('Original lease must belong to this launcher process')
        lease.assert_parent_alive()
        return self._reserve(work_key, counts=counts, binding=binding,
                             lease_seconds=lease.lease_seconds, retry_id=retry_id, lease=lease)

    def _reserve(self, work_key, *, counts, binding, lease_seconds, retry_id, lease):
        work_key = _text(work_key, 'work_key')
        counts, binding = _counts(counts), _object(binding, 'binding')
        if retry_id is not None:
            _text(retry_id, 'retry_id')  # Diagnostic alias cannot change identity or charge.
        if not any(counts.values()) or type(lease_seconds) is not int or not 1 <= lease_seconds <= 300:
            raise ValueError('Positive work counters and lease 1..300 seconds required')
        with self._transaction() as db:
            verified = self._verify(db, repair=True)
            for record in verified['reservations'].values():
                if record['work_key'] == work_key:
                    if (record['counts'] != counts or record['binding'] != binding
                            or record['lease']['lease_seconds'] != lease_seconds
                            or (lease is not None and record['lease'] != lease.to_dict())):
                        raise ValueError('Existing work cannot change binding, quota or lease')
                    return json.loads(_json(record))
            lease = FrozenWorkLease.issue(lease_seconds) if lease is None else lease
            lease.assert_parent_alive()
            month = lease.wall_started_at[:7]
            # A clock rollback into a previously closed UTC month cannot mint time.
            if verified['wallets'] and month < max(verified['wallets']):
                raise ValueError('Actual work month moved backwards')
            wallet = verified['wallets'].get(month, {'counts': _counts({}), 'seconds': 0})
            if (any(wallet['counts'][k] + counts[k] > self.limits[k] for k in self.limits)
                    or wallet['seconds'] + lease_seconds > self.manifest['monthly_seconds']):
                raise BudgetExhausted('Monthly prepaid work quota exhausted')
            record = {'reservation_id': fingerprint({'authority': self.manifest_hash, 'work_key': work_key}),
                      'work_key': work_key, 'authority_manifest_hash': self.manifest_hash,
                      'authority_directory': str(self.directory), 'actual_month': month,
                      'binding': binding, 'limits': self.limits.copy(), 'counts': counts,
                      'lease': lease.to_dict(), 'lease_hash': lease.digest}
            if self.manifest['partition_policy'] is not None:
                # The complete wallet audit above has finished in this SAME
                # still-locked transaction. Routing may recursively read immutable
                # identity; it may not open a second IMMEDIATE writer or skip rows.
                with self._identity_read_scope(db):
                    record['allocation'] = allocation_route(self, work_key, binding, current=True,
                                                            lease=lease, debit=counts)
                # Precheck BEFORE the irreversible external intent. A denied
                # competing owner/partition attempt cannot poison good history.
                charge_partition(self.manifest['partition_policy'], json.loads(_json(verified['partitions'])),
                                 json.loads(_json(verified['owners'])), month, record['allocation'], counts, lease_seconds)
            record['terms_hash'] = fingerprint(record)
            record['status'] = 'reserved'
            return self._commit_event(db, verified, record, 'reserved')

    def _change(self, reservation_id, action, **payload):
        with self._transaction() as db:
            verified = self._verify(db, repair=True)
            if reservation_id not in verified['reservations']:
                raise ValueError('Unknown authority reservation')
            record = json.loads(_json(verified['reservations'][reservation_id]))
            status = record['status']
            if action == 'recover' and status in TERMINAL:
                return record
            if action == status:
                if all(record.get(k) == v for k, v in payload.items()):
                    return record
                raise ValueError('A committed transition cannot change its result')
            if status in TERMINAL:
                raise ValueError('Consumed work cannot restart or be reclassified')
            if action in {'running', 'completed'} and os.getpid() != record['lease']['parent_pid']:
                raise ValueError('Only the original launcher can start/complete work')
            check = FrozenWorkLease.from_dict(record['lease']).check()
            if not check.allowed:
                record.update(status='interrupted_consumed', reason='original_lease:' + check.reason)
                return self._commit_event(db, verified, record, record['status'])
            if action == 'recover':
                return record
            if action == 'running' and status != 'reserved':
                raise ValueError('Only reserved work can start')
            if action == 'completed':
                if status != 'running':
                    raise ValueError('Work must be running before completion')
                payload['completed_at'] = datetime.now(UTC).isoformat()
            record.update(status=action, **payload)
            if action == 'completed':
                self._validate_result(record)
            return self._commit_event(db, verified, record, action)

    def start(self, reservation_id, process_identity):
        return self._change(reservation_id, 'running', process_identity=_object(process_identity, 'process identity'))

    def complete(self, reservation_id, *, result, actual_counts):
        """Record verified artifact accounting; not an independent model admission.

        The caller must recheck this original lease after all completion I/O and
        before model admission. A slow final SQLite fsync/mirror cannot be proven
        on-time from the preparation timestamp stored here alone.
        """
        return self._change(reservation_id, 'completed', result=_object(result, 'result'), actual_counts=_counts(actual_counts))

    def fail_consumed(self, reservation_id, reason, *, interrupted=False):
        if type(interrupted) is not bool:
            raise ValueError('interrupted must be boolean')
        return self._change(reservation_id, 'interrupted_consumed' if interrupted else 'failed_consumed',
                            reason=_text(reason, 'failure reason', 1024))

    def recover(self, reservation_id):
        return self._change(reservation_id, 'recover')

    def get(self, reservation_id):
        """Verified read only; completed work does not require a still-live issuer."""
        verified = self._charge_inventory()
        if verified is not None:
            if reservation_id not in verified['reservations']:
                raise ValueError('Unknown authority reservation')
            return verified['reservations'][reservation_id]
        with self._transaction() as db:
            verified = self._verify(db)
            if reservation_id not in verified['reservations']:
                raise ValueError('Unknown authority reservation')
            return verified['reservations'][reservation_id]

    def verify(self):
        verified = self._charge_inventory()
        if verified is not None:
            return {'authority_manifest_hash': self.manifest_hash, 'head': verified['head'],
                    'events': verified['sequence'], 'reservations': len(verified['reservations'])}
        with self._transaction() as db:
            verified = self._verify(db)
            return {'authority_manifest_hash': self.manifest_hash, 'head': verified['head'],
                    'events': verified['sequence'], 'reservations': len(verified['reservations'])}

    def status(self, month):
        _month(month)
        with self._transaction() as db:
            verified = self._verify(db)
            wallet = verified['wallets'].get(month, {'counts': _counts({}), 'seconds': 0})
            return {'actual_month': month, 'counts': wallet['counts'], 'reserved_seconds': wallet['seconds'],
                    'limits': self.limits.copy(), 'monthly_seconds': self.manifest['monthly_seconds'],
                    'head': verified['head'], 'authority_manifest_hash': self.manifest_hash,
                    'partition_policy': self.manifest['partition_policy'],
                    'partitions': {k.split('|', 1)[1]: v for k, v in verified['partitions'].items() if k.startswith(month+'|')},
                    'owners': {k.split('|', 1)[1]: v for k, v in verified['owners'].items() if k.startswith(month+'|')}}
