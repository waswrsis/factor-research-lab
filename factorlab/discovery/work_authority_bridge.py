"""Prepaid S work using one MonthlyWorkAuthority, with recoverable projections.

This adapter is not yet controller wiring. A work envelope is charged once in
the authority, before any execution. Its immutable actual-use chain is an audit
projection, never a second grant of monthly credit. Reopening is readback only;
interrupted envelopes cannot execute again. The caller must supply stable work
keys and bound each blocking operation with the original work lease/watchdog.
Local checks alone are cooperative and do not interrupt a native fit.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import psutil
from filelock import FileLock

from factorlab.protocol import fingerprint

from .local_mutation import BudgetExhausted, MonthlyBudget
from .monthly_work_authority import DEFAULT_LIMITS, MonthlyWorkAuthority
from .v7_run import DurableBudget, RunStore, source_files
from .v7_storage import immutable_record
from .work_lease import FrozenWorkLease, _boot_id

VERSION = 'work-authority-bridge-v1'
S_KINDS = {'ast', 'f', 'gpt_attempt', 'family', 'solve'}


class LocalQuotaExhausted(BudgetExhausted):
    """Only a deterministic counter precheck; never identity, IO or lease failure."""


def _copy(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _counts(value):
    if (type(value) is not dict or set(value) - set(DEFAULT_LIMITS)
            or any(type(v) is not int or v < 0 for v in value.values())):
        raise ValueError('Invalid canonical work counters')
    return {k: value.get(k, 0) for k in DEFAULT_LIMITS}


def _receipt_bytes(path):
    try:
        with path.open('rb') as stream:
            raw = stream.read(65537)
        if len(raw) > 65536:
            raise ValueError('Bridge receipt exceeds 64 KiB')
        return raw
    except (OSError, ValueError) as exc:
        raise ValueError('Missing/damaged bridge receipt: ' + str(path)) from exc


def _decode_receipt(path, raw):
    try:
        value = json.loads(raw.decode('utf-8'))
        if set(value) != {'record', 'hash'} or fingerprint(value['record']) != value['hash']:
            raise ValueError('Bridge receipt checksum mismatch')
        return value['record']
    except (TypeError, KeyError, ValueError) as exc:
        raise ValueError('Missing/damaged bridge receipt: ' + str(path)) from exc


def _read(path):
    return _decode_receipt(path, _receipt_bytes(path))


@dataclass(frozen=True)
class _VerifiedBridgePrefix:
    """Process-local replay result bound to exact bytes, never mutable counters."""
    identity_hash: str
    receipts: tuple[tuple[str, bytes, bytes], ...]
    reservation_terms: tuple[tuple[str, str], ...]
    state_json: str
    state_hash: str


def _write(path, value):
    if len(json.dumps(value, allow_nan=False).encode()) > 64000:
        raise ValueError('Bridge receipt exceeds bound')
    immutable_record(path, {'record': value, 'hash': fingerprint(value)})


class WorkAuthorityBridge:
    """One frozen S execution envelope per actual RunStore replay month.

    ``persist`` is the existing controller checkpoint callback. Only its
    projection is repairable. Authority reservations and immutable actual-use
    receipts precede that callback. Existing nonzero legacy budgets cannot be
    adopted. Construction after the first is readback-only, including in the
    same process; callers keep the original object during uninterrupted work.
    """

    def __init__(self, store, month, state, authority, persist, *, external_deadline=None, preparation=None):
        if type(store) is not RunStore or type(authority) is not MonthlyWorkAuthority or not callable(persist):
            raise TypeError('Actual RunStore, MonthlyWorkAuthority and checkpoint callback required')
        self.store, self.month, self.state = store, month, state
        self.authority, self._persist = authority, persist
        self._active = None
        self._closed = False
        self.preparation = preparation
        self._validate_context()
        budget = MonthlyBudget.from_dict(state['budget'])
        if budget.month_id != month:
            raise ValueError('Projection replay month mismatch')
        self.identifier = fingerprint({'directory': str(store.directory), 'replay_month': month})
        self.directory = store.directory / 'work_authority' / month
        self.anchor = authority.directory / 'search_bridges' / self.identifier
        self._lock = FileLock(str(self.anchor) + '.lock', timeout=10)
        static = {'version': VERSION, 'bridge_id': self.identifier,
                  'run_directory': str(store.directory), 'run_identity_hash': fingerprint(store.identity),
                  'replay_month': month, 'data_id': state['data_id'],
                  'search_exposure': state['search_exposure'],
                  'authority_manifest_hash': authority.manifest_hash,
                  'authority_directory': str(authority.directory),
                  'source_hash': fingerprint(store.identity['source_files']),
                  'limits': budget.limits}
        if preparation is not None:
            from .s_preparation import SPreparationEnvelope
            if type(preparation) is not SPreparationEnvelope or preparation.authority is not authority:
                raise TypeError('A real original S preparation on this authority is required')
            prepared_envelope = preparation.verified_bridge_envelope(store, month, state)
            static['preparation_hash'] = preparation.digest
        with self._lock:
            first = not (self.anchor / 'identity.json').exists()
            if first:
                if preparation is not None and not preparation._executable:
                    raise ValueError('Recovered preparation is readback only; cannot start a bridge')
                if (self.directory.exists() or self.anchor.exists() or state.get('authority_bridge') is not None
                        or any(budget.counts.values()) or budget.families
                        or (self._reservations() and preparation is None)):
                    raise ValueError('Existing/legacy work cannot reset or adopt a bridge')
                original = (FrozenWorkLease.issue(1) if preparation is None
                            else FrozenWorkLease.from_dict(prepared_envelope['issuer']))
                seconds = store.identity['config']['research_seconds']
                deadline = original.monotonic_started + seconds
                if external_deadline is not None:
                    if (type(external_deadline) not in (int, float) or not math.isfinite(external_deadline)
                            or external_deadline <= original.monotonic_started):
                        raise ValueError('Invalid original external deadline')
                    deadline = min(deadline, external_deadline)
                envelope = {'issuer': original.to_dict(), 'seconds': seconds,
                            'monotonic_deadline': deadline,
                            'wall_deadline': (datetime.fromisoformat(original.wall_started_at)
                                + timedelta(seconds=deadline-original.monotonic_started)).isoformat()}
                if preparation is not None:
                    original_baseline = preparation.baseline_record()
                    if set(self._reservations()) != {original_baseline['reservation_id']}:
                        raise ValueError('Original S preparation contains unrelated existing work')
                    envelope = _copy(prepared_envelope)
                    if external_deadline is not None and external_deadline != envelope['monotonic_deadline']:
                        raise ValueError('Cannot replace the original preparation deadline')
                identity = {**static, 'initial_budget': budget.to_dict(), 'execution_envelope': envelope}
                _write(self.anchor / 'identity.json', identity)
                _write(self.directory / 'identity.json', identity)
            else:
                identity = _read(self.anchor / 'identity.json')
                if (any(identity.get(k) != v for k, v in static.items())
                        or _read(self.directory / 'identity.json') != identity):
                    raise ValueError('Bridge identity/source/input changed')
                if external_deadline is not None and external_deadline != identity['execution_envelope']['monotonic_deadline']:
                    raise ValueError('Original S deadline cannot change on recovery')
            self.identity = identity
            self._identity_hash = fingerprint(identity)
            self._execution_allowed = first
            self._reload(repair=True)
            self._project()

    def _validate_context(self):
        self.store.verify()
        self.authority.verify()
        scope = self.authority.manifest['scope']
        if (source_files() != self.store.identity['source_files']
                or self.authority.manifest['source_files'] != self.store.identity['source_files']
                or scope['task'] != self.store.identity['rolling_spec']['task']
                or scope['universe'] != sorted(self.store.identity['config']['symbols'])
                or self.store.get(self.month) != self.state
                or self.state.get('status') != 'running'):
            raise ValueError('Bridge requires exact active source and RunStore scope')
        if hasattr(self, 'identity') and (
                fingerprint(self.store.identity) != self.identity['run_identity_hash']
                or self.state['data_id'] != self.identity['data_id']
                or self.state['search_exposure'] != self.identity['search_exposure']):
            raise ValueError('Frozen bridge input/exposure/RunStore identity changed')
        exposed = self.state.get('search_exposure', {})
        if exposed.get('data_id') != self.state.get('data_id'):
            raise ValueError('Search snapshot differs from its actual exposure')
        if str(self.store.exposure.path) != self.store.identity['exposure_path']:
            raise ValueError('Wrong actual exposure ledger')
        self.store.exposure.assert_prompt_search(**{k: v for k, v in exposed.items() if k != 'data_id'})

    def _reservations(self):
        # Use the authority's own verifier ONCE for a consistent inventory.
        # Its get() per reservation would reverify the full wallet N times.
        # No writes, repair or counter grants are performed in this transaction.
        verified = self.authority._charge_inventory()
        if verified is None:
            with self.authority._transaction() as db:
                verified = self.authority._verify(db)
        records = _copy(list(verified['reservations'].values()))
        return {r['reservation_id']: r for r in records
                if r['binding'].get('bridge_id') == self.identifier}

    def _reload(self, *, repair=False):
        budget = MonthlyBudget.from_dict(self.identity['initial_budget'])
        actual, works, previous = _counts({}), {}, '0' * 64
        reservations = self._reservations()
        projected = self.state.get('authority_bridge')
        if projected is not None and (
                projected.get('identity_hash') != fingerprint(self.identity)
                or projected.get('authority_manifest_hash') != self.authority.manifest_hash
                or type(projected.get('steps')) is not int or projected['steps'] < 0):
            raise ValueError('RunStore bridge projection identity changed')
        files = sorted((self.anchor / 'steps').glob('*.json'))
        mirrors = {p.name for p in (self.directory / 'steps').glob('*.json')}
        if mirrors - {p.name for p in files}:
            raise ValueError('Bridge authoritative step missing')
        prefix = None if repair else getattr(self, '_verified_reload_prefix', None)
        receipts, prefix_count = [], 0
        if prefix is not None:
            if (type(prefix) is not _VerifiedBridgePrefix
                    or prefix.identity_hash != fingerprint(self.identity)
                    or len(files) < len(prefix.receipts)):
                raise ValueError('Verified bridge prefix identity or length changed')
            for index, (name, original, mirrored) in enumerate(prefix.receipts):
                path = files[index]
                if (path.name != name or _receipt_bytes(path) != original
                        or _receipt_bytes(self.directory / 'steps' / name) != mirrored):
                    raise ValueError('Verified immutable bridge prefix bytes changed')
            for rid, terms_hash in prefix.reservation_terms:
                if rid not in reservations or reservations[rid]['terms_hash'] != terms_hash:
                    raise ValueError('Verified bridge prefix lost its original authority terms')
            restored = json.loads(prefix.state_json)
            if fingerprint(restored) != prefix.state_hash:
                raise ValueError('Verified bridge prefix replay state changed')
            budget = MonthlyBudget.from_dict(restored['budget'])
            actual, works, previous = restored['actual'], restored['works'], restored['head']
            receipts, prefix_count = list(prefix.receipts), len(prefix.receipts)
            if projected is not None and 0 < projected['steps'] <= prefix_count:
                # The projection can lag the latest immutable receipt. Check its
                # own exact already-byte-verified checkpoint, not just the tip.
                name, raw, _ = receipts[projected['steps'] - 1]
                checkpoint = _decode_receipt(self.anchor / 'steps' / name, raw)
                if (projected['head'] != fingerprint(checkpoint)
                        or projected['actual_counts'] != checkpoint['actual_counts']):
                    raise ValueError('RunStore projection contradicts its immutable prefix')
        for index, path in enumerate(files[prefix_count:], prefix_count + 1):
            original = _receipt_bytes(path)
            event = _decode_receipt(path, original)
            if (path.name != f'{index:06d}.json' or event['sequence'] != index
                    or event['previous'] != previous or event['bridge_id'] != self.identifier):
                raise ValueError('Bridge actual-use chain mismatch')
            mirror = self.directory / 'steps' / path.name
            if not mirror.exists() and repair:
                _write(mirror, event)
            mirrored = _receipt_bytes(mirror)
            if _decode_receipt(mirror, mirrored) != event:
                raise ValueError('Bridge step mirror changed')
            receipts.append((path.name, original, mirrored))
            rid = event['reservation_id']
            if rid not in reservations:
                raise ValueError('Bridge work reservation is missing')
            reservation = reservations[rid]
            if (reservation['binding'].get('bridge_id') != self.identifier
                    or reservation['terms_hash'] != event['terms_hash']):
                raise ValueError('Bridge step refers to different authority work')
            work = works.setdefault(rid, {'counts': _counts({}), 'claimed': False, 'terminal': None})
            operation, args = event['operation'], event['args']
            if operation == 'claim':
                if work['claimed']:
                    raise ValueError('Repeated execution claim')
                work['claimed'] = True
            elif operation in {'complete', 'failed_consumed'}:
                if not work['claimed'] or work['terminal'] is not None:
                    raise ValueError('Invalid terminal work event')
                work['terminal'] = operation
            else:
                if not work['claimed'] or work['terminal'] is not None:
                    raise ValueError('Counters outside claimed execution')
                kind, amount = self._apply(budget, operation, args)
                if kind is not None:
                    actual[kind] += amount
                    work['counts'][kind] += amount
                    if work['counts'][kind] > reservation['counts'][kind]:
                        raise ValueError('Actual usage exceeds prepaid envelope')
            if event['budget'] != budget.to_dict() or event['actual_counts'] != actual:
                raise ValueError('Bridge projection was not derived from actual steps')
            previous = fingerprint(event)
            if projected is not None and projected['steps'] == index and (
                    projected['head'] != previous or projected['actual_counts'] != actual):
                raise ValueError('RunStore projection contradicts its immutable prefix')
        if projected is not None and (projected['steps'] > len(files) or (
                projected['steps'] == 0 and (projected['head'] != '0' * 64
                                            or projected['actual_counts'] != _counts({})))):
            raise ValueError('Committed bridge steps cannot be removed/reset')
        baseline_id = (self.preparation.baseline_record()['reservation_id']
                       if self.preparation is not None else None)
        for rid, record in reservations.items():
            prepared = (self.preparation is not None
                        and record['reservation_id'] == baseline_id
                        and record['binding'].get('preparation_hash') == self.identity.get('preparation_hash'))
            if not prepared and record['binding'].get('bridge_identity_hash') != fingerprint(self.identity):
                raise ValueError('Authority work identity differs from original S envelope')
            work = works.get(rid)
            if record['status'] == 'completed' and (work is None or record['actual_counts'] != work['counts']):
                raise ValueError('Authority completed accounting differs from bridge steps')
        self._budget, self._actual, self._works = budget, actual, works
        self._head, self._sequence = previous, len(files)
        # All bytes are reread on every use, including the pre/post-debit checks.
        # Restart/repair always performs the original complete replay. The cap
        # bounds retained bytes; exceeding it only disables this optimization.
        if len(receipts) <= 4096 and sum(len(a) + len(b) for _, a, b in receipts) <= 4 * 1024 * 1024:
            restored = {'budget': budget.to_dict(), 'actual': actual, 'works': works, 'head': previous}
            self._verified_reload_prefix = _VerifiedBridgePrefix(
                fingerprint(self.identity), tuple(receipts),
                tuple(sorted((rid, reservations[rid]['terms_hash']) for rid in works)),
                json.dumps(restored, sort_keys=True, allow_nan=False), fingerprint(restored))
        else:
            self._verified_reload_prefix = None

    @staticmethod
    def _apply(budget, operation, args):
        if operation == 'consume':
            budget.consume(**args)
            return args['kind'], args['amount']
        if operation == 'register_family':
            budget.register_family(**args)
            return 'family', 1
        if operation == 'attribute_ast':
            budget.attribute_ast(**args)
            return None, 0
        raise ValueError('Unknown actual accounting operation')

    def _append(self, reservation, operation, args):
        trial = MonthlyBudget.from_dict(self._budget.to_dict())
        actual = self._actual.copy()
        if operation not in {'claim', 'complete', 'failed_consumed'}:
            try:
                kind, amount = self._apply(trial, operation, args)
            except BudgetExhausted as exc:
                raise LocalQuotaExhausted(str(exc)) from exc
            if kind is not None:
                used = self._works[reservation['reservation_id']]['counts'][kind]
                if used + amount > reservation['counts'][kind]:
                    raise LocalQuotaExhausted('Prepaid work envelope exhausted; no automatic renewal')
                actual[kind] += amount
        event = {'sequence': self._sequence + 1, 'previous': self._head, 'bridge_id': self.identifier,
                 'reservation_id': reservation['reservation_id'], 'terms_hash': reservation['terms_hash'],
                 'operation': operation, 'args': args, 'budget': trial.to_dict(), 'actual_counts': actual}
        name = f'{event["sequence"]:06d}.json'
        _write(self.anchor / 'steps' / name, event)
        _write(self.directory / 'steps' / name, event)
        self._reload()
        self._project()

    def _project(self):
        value = {'version': VERSION, 'identity_hash': fingerprint(self.identity),
                 'authority_manifest_hash': self.authority.manifest_hash,
                 'head': self._head, 'steps': self._sequence, 'actual_counts': self._actual.copy()}
        self.state['authority_bridge'] = value
        self.state['budget'] = self._budget.to_dict()
        self._persist(self._budget.to_dict())
        if self.store.get(self.month) != self.state:
            raise ValueError('RunStore projection callback did not persist exact state')

    def _guard(self):
        if self.preparation is not None and self.preparation.identity.get('outer_scope') is not None:
            from .s_outer_context import SOuterContext
            SOuterContext.verify_bound(self.preparation.identity['outer_scope'])
        if self._closed or not self._execution_allowed:
            raise ValueError('Recovered bridge is readback only; execution cannot resume')
        self._validate_context()
        if fingerprint(self.identity) != self._identity_hash:
            raise ValueError('Original bridge identity/envelope changed')
        envelope = self.identity['execution_envelope']
        origin = FrozenWorkLease.from_dict(envelope['issuer'])
        now, mono = datetime.now(UTC), time.monotonic()
        if (os.getpid() != origin.parent_pid or psutil.Process().create_time() != origin.parent_created
                or _boot_id() != origin.boot_id or mono < origin.monotonic_started
                or now < datetime.fromisoformat(origin.wall_started_at)
                or now.strftime('%Y-%m') != origin.wall_started_at[:7]):
            raise ValueError('Original S issuer/clock/execution month changed')
        if mono >= envelope['monotonic_deadline'] or now >= datetime.fromisoformat(envelope['wall_deadline']):
            raise BudgetExhausted('Original S execution deadline consumed')

    def begin_work(self, work_key, *, counts, lease_seconds, binding):
        """Claim once; no callback is called until authority has prepaid work."""
        self._guard()
        if self._active is not None:
            raise ValueError('Finish the current work before starting another')
        counts = _counts(counts)
        if any(counts[k] for k in set(DEFAULT_LIMITS) - S_KINDS):
            raise ValueError('Neural/tree work requires the separate reserved child budget')
        if type(work_key) is not str or not work_key or len(work_key) > 128 or type(binding) is not dict:
            raise ValueError('Stable bounded work key and binding required')
        with self._lock:
            self._reload()
            key = VERSION + ':' + self.identifier + ':' + work_key
            prepared = self.preparation is not None and work_key == 'baseline'
            if any(r['work_key'] == key for r in self._reservations().values()):
                raise ValueError('Work already prepaid; recover readback only, never execute again')
            if prepared:
                record = self.preparation.baseline_record()
                if (record['reservation_id'] in self._works or record['status'] != 'running'
                        or record['counts'] != counts or record['lease']['lease_seconds'] != lease_seconds):
                    raise ValueError('Original baseline work cannot change or execute twice')
                lease = FrozenWorkLease.from_dict(record['lease'])
            else:
                lease = FrozenWorkLease.issue(lease_seconds)
            envelope = self.identity['execution_envelope']
            if (lease.monotonic_deadline > envelope['monotonic_deadline']
                    or lease.wall_deadline > envelope['wall_deadline']
                    or lease.wall_deadline[:7] != lease.wall_started_at[:7]):
                raise BudgetExhausted('Work lease would exceed the original S/month deadline')
            if not prepared:
                record = self.authority._reserve_with_lease(key, counts=counts,
                    binding={'bridge_id': self.identifier, 'bridge_identity_hash': fingerprint(self.identity),
                             'role': 'S', 'replay_month': self.month, 'work': _copy(binding)}, lease=lease)
            try:
                record = self.authority.start(record['reservation_id'],
                    {'pid': os.getpid(), 'created': psutil.Process().create_time(), 'mode': 'S_parent'})
                if record['status'] != 'running':
                    raise ValueError('Prepaid work lease closed before execution')
                self._append(record, 'claim', {})
                lease.assert_parent_alive()
            except BaseException:
                self._closed = True
                raise
            work = AuthorityWorkBudget._create(self, record)
            self._active = work
            return work

    def recover_projection(self):
        """Repair committed mirrors/checkpoints; NEVER return executable work."""
        with self._lock:
            self._execution_allowed = False
            self._reload(repair=True)
            self._project()
            return {'projection': _copy(self.state['authority_bridge']),
                    'reservations': list(self._reservations().values()), 'execution_allowed': False}

    @property
    def actual_counts(self):
        return self._actual.copy()


class AuthorityWorkBudget(DurableBudget):
    """MonthlyBudget-compatible actual projection of one prepaid S work item."""

    def __init__(self, *args, **kwargs):
        raise TypeError('Use WorkAuthorityBridge.begin_work')

    @classmethod
    def _create(cls, bridge, record):
        result = object.__new__(cls)
        result._bridge, result._record = bridge, _copy(record)
        result._lease = FrozenWorkLease.from_dict(record['lease'])
        result._closed = False
        return result

    def __getattr__(self, name):
        raise AttributeError(name)

    @property
    def bridge(self):
        return self._bridge

    @property
    def record(self):
        return _copy(self._record)

    @property
    def lease(self):
        return self._lease

    @property
    def month_id(self):
        return self.bridge.month

    @property
    def counts(self):
        return self.bridge._budget.counts.copy()

    @property
    def limits(self):
        return self.bridge._budget.limits.copy()

    @property
    def families(self):
        return _copy(self.bridge._budget.families)

    @property
    def actual_counts(self):
        return self.bridge._works[self.record['reservation_id']]['counts'].copy()

    @property
    def digest(self):
        return fingerprint(self.to_dict())

    def to_dict(self):
        return self.bridge._budget.to_dict()

    def check_time(self):
        if self._closed:
            raise ValueError('Work execution is closed')
        self.bridge._guard()
        self.lease.assert_parent_alive()
        if self.bridge.authority.get(self.record['reservation_id'])['status'] != 'running':
            raise ValueError('Canonical work is not running')

    def persist(self, value):
        if value != self.to_dict():
            raise ValueError('Cannot mint or replace projected counters')
        self.check_time()
        self.bridge._project()
        self.check_time()

    def _charge(self, operation, args):
        try:
            with self.bridge._lock:
                with self.bridge.authority._charge_read_scope():
                    self.check_time()
                    self.bridge._reload()
                    self.bridge._append(self.record, operation, args)
                    self.check_time()
                # Source verification and transaction exit also consume the
                # original lease; no numerical work may start after that tail.
                self.lease.assert_parent_alive()
        except LocalQuotaExhausted:
            self._closed = True
            raise
        except BaseException:
            self._closed = self.bridge._closed = True
            raise

    def consume(self, kind, amount=1, *, family_id=None):
        self._charge('consume', {'kind': kind, 'amount': amount, 'family_id': family_id})

    def register_family(self, family_id):
        self._charge('register_family', {'family_id': family_id})

    def attribute_ast(self, family_id, amount):
        self._charge('attribute_ast', {'family_id': family_id, 'amount': amount})

    def complete(self, result):
        try:
            self.check_time()
            record = self.bridge.authority.complete(self.record['reservation_id'],
                result={'bridge_head': self.bridge._head, 'result': _copy(result)},
                actual_counts=self.actual_counts)
            if record['status'] != 'completed':
                raise ValueError('Canonical work expired before completion')
            with self.bridge._lock:
                self.bridge._append(self.record, 'complete', {})
            self.lease.assert_parent_alive()
            return record
        finally:
            self._closed = True
            self.bridge._active = None

    def fail(self, reason):
        try:
            record = self.bridge.authority.fail_consumed(self.record['reservation_id'], reason)
            with self.bridge._lock:
                self.bridge._append(self.record, 'failed_consumed', {})
            return record
        finally:
            self._closed = True
            self.bridge._active = None
