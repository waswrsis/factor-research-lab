"""Fixed work scheduler over the original S authority envelope.

Operations are trusted controller functions, never provider-selected callbacks.
No scientific quota allocation is implicit: the caller freezes every slot first.
This is cooperative accounting; isolated numerical/GPT deadlines still require
the fixed launcher to enforce each original lease. The monthly controller must
explicitly integrate this scheduler before it can claim unified scheduling.
"""
import json
from dataclasses import dataclass

from factorlab.discovery.monthly_work_authority import DEFAULT_LIMITS
from factorlab.discovery.monthly_work_authority import BudgetExhausted as WalletQuotaExhausted
from factorlab.discovery.v7_storage import immutable_record
from factorlab.discovery.work_authority_bridge import LocalQuotaExhausted, WorkAuthorityBridge
from factorlab.protocol import fingerprint


@dataclass(frozen=True)
class FrozenSWorkSchedule:
    _json: str

    @classmethod
    def freeze(cls, *, config, limits, slots, allocation=None):
        if not isinstance(slots, list) or not 1 <= len(slots) <= 64:
            raise ValueError('A bounded complete work schedule is required')
        used = dict.fromkeys(DEFAULT_LIMITS, 0)
        keys, seconds = set(), 0
        for slot in slots:
            if set(slot) != {'key', 'stage', 'counts', 'seconds'}:
                raise ValueError('Work slots require exact frozen fields')
            if (not isinstance(slot['key'], str) or not slot['key'] or len(slot['key']) > 100
                    or slot['key'] in keys or any(ord(c) < 32 for c in slot['key'])
                    or slot['stage'] not in {'baseline', 'family', 'candidate', 'pruning'}
                    or type(slot['seconds']) is not int or not 1 <= slot['seconds'] <= 300):
                raise ValueError('Invalid immutable work slot')
            keys.add(slot['key'])
            seconds += slot['seconds']
            counts = slot['counts']
            if (not isinstance(counts, dict) or not counts or not any(counts.values())
                    or set(counts) - {'solve', 'ast', 'family', 'gpt_attempt', 'f'}
                    or any(type(v) is not int or v < 0 for v in counts.values())):
                raise ValueError('Unknown/noninteger S counters')
            for k, v in counts.items():
                used[k] += v
        if (seconds > config['research_seconds'] or config['research_seconds'] > 7200
                or any(used[k] > limits.get(k, 0) for k in used if used[k])):
            raise ValueError('Schedule exceeds frozen original S limits')
        value = {'version': 'v7-frozen-S-work-schedule-v1', 'config': config,
                 'limits': limits, 'slots': slots, 'reserved_seconds': seconds,
                 'allocated_counts': used, 'unused_transfer': False,
                 'allocation': allocation,
                 'algorithm_change': 'local quota closes slot; immutable later slots may continue'}
        return cls(json.dumps(value, sort_keys=True, allow_nan=False))

    def to_dict(self):
        return json.loads(self._json)

    @classmethod
    def from_dict(cls, value):
        result = cls.freeze(config=value['config'], limits=value['limits'], slots=value['slots'],
                            allocation=value['allocation'])
        if result.to_dict() != value:
            raise ValueError('Unsupported or changed frozen S schedule')
        return result

    @property
    def digest(self):
        return fingerprint(self.to_dict())


def validate_schedule_authority(authority, schedule):
    """Readonly guard for entry points before their first baseline debit.

    Legacy custom schedules with no named allocation remain diagnostic only.
    The default allocation requires the exact immutable authority policy.
    """
    from .monthly_work_authority import MonthlyWorkAuthority
    if type(authority) is not MonthlyWorkAuthority:
        raise TypeError('Actual canonical authority required')
    authority.verify()
    body = FrozenSWorkSchedule.from_dict(schedule.to_dict()).to_dict()
    allocation = body.get('allocation')
    if isinstance(allocation, dict) and allocation.get('version') == 'v7-shared-monthly-allocation-20261005-v1':
        from .monthly_partition_policy import canonical_policy
        policy = canonical_policy(authority.manifest.get('partition_policy'))
        mode = 'controls' if body['config']['budget_policy'] == 'controls_third' else 'standard'
        lane = 'S:'+body['config']['search_mode'] if mode == 'controls' else 'S'
        if (policy is None or policy['mode'] != mode or lane not in policy['lanes']
                or body['allocated_counts'] != policy['lanes'][lane]['counts']
                or body['reserved_seconds'] != policy['lanes'][lane]['seconds']):
            raise ValueError('Allocation table requires the exact authoritative monthly partition policy')
    return body


class ScheduledSearch:
    """A fixed controller schedule holding no independently spendable wallet."""

    def __init__(self, bridge, schedule, *, local_quota_type=LocalQuotaExhausted):
        if type(bridge) is not WorkAuthorityBridge or local_quota_type is not LocalQuotaExhausted:
            raise TypeError('Actual authority bridge and fixed counter-only exhaustion type required')
        self.bridge, self.schedule = bridge, FrozenSWorkSchedule.from_dict(schedule.to_dict())
        self.local_quota_type = local_quota_type
        self._closed = False
        body = schedule.to_dict()
        validate_schedule_authority(bridge.authority, schedule)
        if (fingerprint(body['config']) != fingerprint(bridge.store.identity['config'])
                or body['limits'] != bridge.identity['limits']):
            raise ValueError('Schedule differs from original RunStore contract')
        # Bind both copies before the first reservation; no schedule replacement.
        self._root = bridge.anchor / 'schedule'
        self._mirror = bridge.directory / 'schedule'
        identity = {'schedule': body, 'schedule_hash': schedule.digest,
                    'bridge_identity_hash': fingerprint(bridge.identity)}
        for root in (self._root, self._mirror):
            path = root / 'identity.json'
            if not path.exists() and bridge._reservations():
                preparation = getattr(bridge, 'preparation', None)
                initial_prepaid = (preparation is not None and bridge._execution_allowed
                    and not bridge._works
                    and preparation.identity['work_schedule_hash'] == schedule.digest
                    and set(bridge._reservations()) == {preparation.baseline_record()['reservation_id']})
                if not initial_prepaid:
                    raise ValueError('Lost schedule cannot be recreated after reserved work')
            immutable_record(path, identity)
        self._identity = identity
        self._cursor = self._read_cursor()
        if (getattr(bridge, 'preparation', None) is not None
                and bridge.preparation.identity['work_schedule_hash'] != schedule.digest):
            raise ValueError('Work schedule differs from original preparation')

    def _read_cursor(self):
        for root in (self._root, self._mirror):
            if json.loads((root / 'identity.json').read_text()) != self._identity:
                raise ValueError('Frozen S work schedule changed')
        originals = sorted((self._root / 'results').glob('*.json'))
        mirrors = sorted((self._mirror / 'results').glob('*.json'))
        if [p.name for p in originals] != [p.name for p in mirrors]:
            raise ValueError('Incomplete S work publication; cannot continue execution')
        slots = self.schedule.to_dict()['slots']
        previous = '0' * 64
        for i, (path, mirror) in enumerate(zip(originals, mirrors, strict=True)):
            value = json.loads(path.read_text())
            record = value['record']
            if (i >= len(slots) or path.name != f'{i:03d}.json'
                    or value != json.loads(mirror.read_text())
                    or value['hash'] != fingerprint(record) or record['previous'] != previous
                    or record['schedule_hash'] != self.schedule.digest or record['slot'] != slots[i]):
                raise ValueError('Fixed S work history changed')
            previous = value['hash']
        self._head = previous
        return len(originals)

    def _slot(self, key):
        if self._closed:
            raise ValueError('S execution closed')
        self.bridge._guard()
        cursor = self._read_cursor()
        slots = self.schedule.to_dict()['slots']
        if cursor != self._cursor or cursor >= len(slots) or slots[cursor]['key'] != key:
            self._closed = self.bridge._closed = True
            raise ValueError('Work differs from the next immutable S slot')
        return slots[cursor]

    def _publish(self, slot, result):
        record = {'slot': slot, 'schedule_hash': self.schedule.digest,
                  'previous': self._head, 'result': result}
        value = {'record': record, 'hash': fingerprint(record)}
        for root in (self._root, self._mirror):
            immutable_record(root / 'results' / f'{self._cursor:03d}.json', value)
        self.bridge._guard()
        self._cursor += 1
        self._head = value['hash']

    def skip(self, key, *, reason):
        """No candidate/family exists; unused allowance stays with its own slot."""
        try:
            slot = self._slot(key)
            if reason not in {'no_candidate', 'no_valid_family', 'no_pool_to_prune'}:
                raise ValueError('Unregistered skip reason')
            self._publish(slot, {'status': 'unused_nontransferable', 'reason': reason})
        except BaseException:
            self._closed = self.bridge._closed = True
            raise

    def run(self, key, operation, *, input_binding):
        """Execute one trusted stage and retain every original prepaid debit."""
        work = None
        try:
            slot = self._slot(key)
            try:
                work = self.bridge.begin_work(key, counts=slot['counts'], lease_seconds=slot['seconds'],
                    binding={'schedule_hash': self.schedule.digest, 'slot': self._cursor,
                             'stage': slot['stage'], 'input': input_binding})
            except WalletQuotaExhausted:
                # No new reservation exists. Record the unavailable opportunity;
                # later work keeps its original cap and cannot borrow this one.
                self.bridge._guard()
                result = {'status': 'unfunded_fixed_slot', 'reservation_id': None}
                self._publish(slot, result)
                return result
            try:
                value = operation(work)
            except self.local_quota_type as error:
                # Counter exhaustion only: the next declared slot may still run.
                work.fail('local_prepaid_quota:' + str(error)[:300])
                result = {'status': 'quota_consumed', 'reservation_id': work.record['reservation_id']}
            else:
                work.complete({'output_hash': fingerprint(value)})
                result = {'status': 'completed', 'value': value,
                          'reservation_id': work.record['reservation_id']}
            self._publish(slot, result)
            return result
        except BaseException as error:
            self._closed = self.bridge._closed = True
            if work is not None:
                try:
                    work.fail(type(error).__name__ + ':' + str(error)[:300])
                except BaseException as failure:  # noqa: BLE001 - preserve the original fatal exception.
                    error.add_note('S fail-consumed persistence also failed: ' + str(failure)[:200])
            raise
