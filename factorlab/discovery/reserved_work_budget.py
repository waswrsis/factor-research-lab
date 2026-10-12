"""An executor projection of prepaid work, never a second monthly wallet.

The fixed worker (or original solve-only issuer) constructs once from a running
canonical reservation. Every
charge is durable before work and no execution state can be restored. Parent
readback requires a complete final receipt; unknown/failed accounting never
refunds the original authority charge. This is accounting, not model admission
or independent authentication against a worker replacing all local records.
"""
from __future__ import annotations

import json
import math
import os
from itertools import islice

from filelock import FileLock

from factorlab.protocol import fingerprint

from .monthly_work_authority import DEFAULT_LIMITS, MonthlyWorkAuthority
from .nonlinear_channel import ChannelBudget
from .v7_storage import immutable_record
from .work_lease import FrozenWorkLease

KINDS = ('tree_fit', 'neural_fit', 'epoch', 'solve')
VERSION = 'reserved-channel-budget-v2'
MAX_TIME_CHARGES = 64


def _read(path):
    try:
        if path.stat().st_size > 32768:
            raise ValueError('Prepaid child receipt exceeds 32 KiB')
        value = json.loads(path.read_text('utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError('Missing/damaged prepaid child accounting receipt') from exc
    if not isinstance(value, dict) or set(value) != {'record', 'hash'} or value['hash'] != fingerprint(value['record']):
        raise ValueError('Prepaid child accounting checksum mismatch')
    return value


def _write(path, record):
    value = {'record': record, 'hash': fingerprint(record)}
    immutable_record(path, value)
    return value['hash']


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _canonical(authority, reservation_id):
    if type(authority) is not MonthlyWorkAuthority:
        raise TypeError('A real MonthlyWorkAuthority is required')
    record = authority.get(reservation_id)
    if record['reservation_id'] != reservation_id:
        raise ValueError('Canonical reservation identity mismatch')
    lease = FrozenWorkLease.from_dict(record['lease'])
    if lease.digest != record['lease_hash']:
        raise ValueError('Canonical reservation lease mismatch')
    return record, lease


def _binding(authority, record, engineering):
    return {'version': VERSION, 'reservation_id': record['reservation_id'],
            'authority_manifest_hash': authority.manifest_hash,
            'authority_directory': str(authority.directory), 'terms_hash': record['terms_hash'],
            'source_hash': fingerprint(authority.manifest['source_files']),
            'lease_hash': record['lease_hash'], 'lease': record['lease'],
            'limits': {k: record['counts'][k] for k in KINDS}, 'test_only': engineering,
            'executor_mode': ('issuer' if record.get('process_identity', {}).get('pid')
                              == record['lease']['parent_pid'] else 'child')}


class ReservedChannelBudget(ChannelBudget):
    """ChannelBudget-compatible counters, with no public arbitrary quota constructor."""

    def __init__(self, *args, **kwargs):
        raise TypeError('Use from_reservation with a canonical running authority ticket')

    @classmethod
    def from_reservation(cls, authority, reservation_id):
        return cls._create(authority, reservation_id, engineering=False)

    @classmethod
    def from_issuer_reservation(cls, authority, reservation_id):
        """Original parent only, solve-only convex work; no neural worker grant."""
        record, lease = _canonical(authority, reservation_id)
        identity = record.get('process_identity', {})
        if (identity.get('pid') != lease.parent_pid or identity.get('created') != lease.parent_created
                or any(n for key, n in record['counts'].items() if key != 'solve')):
            raise ValueError('Issuer reservation must bind its original parent and solve-only quota')
        return cls._create(authority, reservation_id, engineering=False, issuer=True)

    @classmethod
    def _for_test(cls, authority, reservation_id):
        """Private issuer-only numerical seam; its receipts remain test_only forever."""
        return cls._create(authority, reservation_id, engineering=True)

    @classmethod
    def _create(cls, authority, reservation_id, *, engineering, issuer=False):
        record, lease = _canonical(authority, reservation_id)
        if record['status'] != 'running':
            raise ValueError('Prepaid child work requires a running reservation')
        lease.assert_parent_alive(as_worker=not engineering and not issuer)
        if issuer and os.getpid() != lease.parent_pid:
            raise ValueError('Issuer work requires the original parent')
        if engineering and os.getpid() != lease.parent_pid:
            raise ValueError('Private engineering seam requires the original issuer')
        if not engineering and record.get('process_identity', {}).get('pid') != os.getpid():
            raise ValueError('Only the registered direct child may claim prepaid execution')
        directory = authority.directory / 'child_actual' / reservation_id
        anchor = authority.directory / 'child_actual_claims' / (reservation_id + '.json')
        directory.parent.mkdir(parents=True, exist_ok=True)
        binding = _binding(authority, record, engineering)
        claim = {'binding': binding, 'worker_pid': os.getpid()}
        # An existing claim OR orphan step/final prohibits a fresh execution. No
        # from_dict path may zero counters or continue an interrupted fit.
        with FileLock(str(directory) + '.claim.lock', timeout=10):
            if directory.exists() or anchor.exists():
                raise ValueError('Prepaid work has already claimed execution; cannot resume/reset')
            _write(anchor, claim)
            directory.mkdir()
            head = _write(directory / 'claim.json', claim)
        result = object.__new__(cls)
        result._directory = directory
        result._binding = binding
        result._binding_hash = fingerprint(binding)
        result._lease = lease
        result._counts = {k: 0 for k in KINDS}
        result._spent = 0.
        result._head = head
        result._sequence = 0
        result._time_charges = 0
        result._closed = False
        result._finished = False
        result._guard()
        return result

    @property
    def limits(self):
        return dict(self._binding['limits'])

    @property
    def counts(self):
        return dict(self._counts)

    @property
    def actual_counts(self):
        return {k: self._counts.get(k, 0) for k in DEFAULT_LIMITS}

    @property
    def test_only(self):
        """Factory-bound mode; callers cannot relabel engineering work as production."""
        return self._binding['test_only']

    @property
    def executor_mode(self):
        return self._binding['executor_mode']

    @property
    def spent_seconds(self):
        return self._spent

    @property
    def max_seconds(self):
        return self._lease.lease_seconds

    @property
    def max_tree_fits(self):
        return self.limits['tree_fit']

    @property
    def max_neural_fits(self):
        return self.limits['neural_fit']

    @property
    def max_epochs(self):
        return self.limits['epoch']

    @property
    def shared(self):
        return None

    @property
    def persist(self):
        return None

    def _guard(self):
        if self._closed:
            raise ValueError('Prepaid child accounting is closed after failure')
        if (fingerprint(self._binding) != self._binding_hash or self._lease.digest != self._binding['lease_hash']
                or set(self._counts) != set(KINDS)
                or any(type(v) is not int or not 0 <= v <= self.limits[k] for k, v in self._counts.items())
                or not _finite(self._spent) or not 0 <= self._spent < self.max_seconds
                or type(self._time_charges) is not int or not 0 <= self._time_charges <= MAX_TIME_CHARGES):
            self._closed = True
            raise ValueError('Prepaid child accounting identity/state changed')
        try:
            if (not self._binding['test_only'] and self.executor_mode == 'issuer'
                    and os.getpid() != self._lease.parent_pid):
                raise ValueError('Issuer accounting cannot run in a child or another parent')
            self._lease.assert_parent_alive(as_worker=(not self._binding['test_only']
                                                      and self.executor_mode == 'child'))
        except BaseException:
            self._closed = True
            raise

    def check_time(self):
        self._guard()

    def begin(self):
        if self._finished:
            raise ValueError('Finalized prepaid work cannot begin again')
        self._guard()  # Never time.monotonic()+seconds; the original lease is fixed.

    def _append(self, kind, amount):
        record = {'binding_hash': self._binding_hash, 'reservation_id': self._binding['reservation_id'],
                  'source_hash': self._binding['source_hash'], 'lease_hash': self._binding['lease_hash'],
                  'sequence': self._sequence + 1, 'previous': self._head,
                  'kind': kind, 'amount': amount, 'counts': self.counts, 'spent_seconds': self._spent}
        try:
            self._head = _write(self._directory / 'steps' / f'{record["sequence"]:06d}.json', record)
            self._sequence = record['sequence']
            self._guard()
        except BaseException:
            self._closed = True
            raise

    def consume(self, kind, amount=1):
        self._guard()
        if (self._finished or kind not in KINDS or type(amount) is not int or amount < 1
                or self._counts[kind] + amount > self.limits[kind]):
            self._closed = True
            raise ValueError('Prepaid child quota exhausted or invalid charge')
        # Actual use is reserved before numeric work. Persistence failure closes
        # execution without rolling this debit back or allowing a fresh instance.
        self._counts[kind] += amount
        self._append(kind, amount)

    def charge_time(self, seconds):
        self._guard()
        if (self._finished or not _finite(seconds) or seconds < 0 or self._spent + seconds > self.max_seconds
                or self._time_charges >= MAX_TIME_CHARGES):
            self._closed = True
            raise ValueError('Invalid/exhausted prepaid child time accounting')
        self._spent += seconds
        self._time_charges += 1
        self._append('seconds', seconds)

    def to_dict(self):
        return {'version': VERSION, 'reservation_id': self._binding['reservation_id'],
                'lease_hash': self._binding['lease_hash'], 'counts': self.counts,
                'actual_counts': self.actual_counts, 'limits': self.limits,
                'spent_seconds': self._spent, 'head': self._head, 'steps': self._sequence,
                'test_only': self._binding['test_only'], 'finalized': self._finished}

    @classmethod
    def from_dict(cls, value, **callbacks):
        raise ValueError('Prepaid child execution cannot be restored from caller state')

    def finalize(self):
        self._guard()
        if not self._finished:
            record = {'binding_hash': self._binding_hash, **self.to_dict(), 'finalized': True}
            try:
                _write(self._directory / 'final.json', record)
                self._guard()
            except BaseException:
                self._closed = True
                raise
            self._finished = True
        return self.to_dict()

    @classmethod
    def verify_counts(cls, authority, reservation_id):
        """Parent read-only accounting reconstruction; no execution recovery.

        Does not assert live-parent/lease time for historical completed receipts.
        The admission verifier separately requires original live lease at launch
        completion and all model/resource/source bindings.
        """
        canonical, _ = _canonical(authority, reservation_id)
        directory = authority.directory / 'child_actual' / reservation_id
        claim = _read(directory / 'claim.json')
        if _read(authority.directory / 'child_actual_claims' / (reservation_id + '.json')) != claim:
            raise ValueError('Prepaid child execution claim anchor differs')
        binding = claim['record'].get('binding')
        if not isinstance(binding, dict) or type(binding.get('test_only')) is not bool:
            raise ValueError('Invalid prepaid child claim binding')
        if binding != _binding(authority, canonical, binding['test_only']):
            raise ValueError('Prepaid child claim differs from canonical authority')
        worker_pid = claim['record'].get('worker_pid')
        issuer = binding['executor_mode'] == 'issuer'
        if (set(claim['record']) != {'binding', 'worker_pid'} or type(worker_pid) is not int or worker_pid <= 0
                or (not binding['test_only'] and (worker_pid != canonical.get('process_identity', {}).get('pid')
                    or (issuer and (worker_pid != canonical['lease']['parent_pid']
                        or canonical['process_identity'].get('created') != canonical['lease']['parent_created']
                        or any(n for k, n in canonical['counts'].items() if k != 'solve')))
                    or (not issuer and worker_pid == canonical['lease']['parent_pid'])))):
            raise ValueError('Prepaid child claim worker identity mismatch')
        if canonical['status'] not in {'running', 'completed', 'failed_consumed', 'interrupted_consumed'}:
            raise ValueError('Canonical authority was never running')
        counts, spent, sequence, time_charges = {k: 0 for k in KINDS}, 0., 0, 0
        head = claim['hash']
        binding_hash = fingerprint(binding)
        maximum_steps = sum(binding['limits'].values()) + MAX_TIME_CHARGES
        paths = sorted(islice((directory / 'steps').glob('*.json'), maximum_steps + 1))
        if len(paths) > maximum_steps:
            raise ValueError('Too many prepaid child accounting steps')
        for path in paths:
            event = _read(path)
            step = event['record']
            sequence += 1
            if path.name != f'{sequence:06d}.json':
                raise ValueError('Missing/reordered prepaid child step')
            kind, amount = step.get('kind'), step.get('amount')
            if kind == 'seconds':
                if not _finite(amount) or amount < 0:
                    raise ValueError('Invalid child time charge')
                spent += amount
                time_charges += 1
            elif kind in KINDS and type(amount) is int and amount >= 1:
                counts[kind] += amount
            else:
                raise ValueError('Invalid child quota charge')
            expected = {'binding_hash': binding_hash, 'reservation_id': reservation_id,
                        'source_hash': binding['source_hash'], 'lease_hash': binding['lease_hash'],
                        'sequence': sequence, 'previous': head, 'kind': kind, 'amount': amount,
                        'counts': counts, 'spent_seconds': spent}
            if (step != expected or any(counts[k] > binding['limits'][k] for k in KINDS)
                    or time_charges > MAX_TIME_CHARGES or spent > binding['lease']['lease_seconds']):
                raise ValueError('Prepaid child accounting chain/limit mismatch')
            head = event['hash']
        result = {'version': VERSION, 'reservation_id': reservation_id,
                  'lease_hash': binding['lease_hash'], 'counts': counts,
                  'actual_counts': {k: counts.get(k, 0) for k in DEFAULT_LIMITS}, 'limits': binding['limits'],
                  'spent_seconds': spent, 'head': head, 'steps': sequence,
                  'test_only': binding['test_only'], 'finalized': True}
        if _read(directory / 'final.json')['record'] != {'binding_hash': binding_hash, **result}:
            raise ValueError('Missing/inconsistent final prepaid child accounting')
        if canonical['status'] == 'completed' and canonical['actual_counts'] != result['actual_counts']:
            raise ValueError('Parent completion differs from verified child actual counts')
        return result
