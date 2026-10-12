"""Original S preparation clock and one prepaid baseline reservation.

This is a 7200s-or-shorter top-level clock, NOT a 7200s wallet lease. The
original <=300s baseline lease is prepaid before feature construction. External
file loading is included only if the trusted caller prepares before that load.
"""
import json
import os
import re
import time
from datetime import UTC, datetime, timedelta

import psutil
from filelock import FileLock

from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
from factorlab.discovery.v7_storage import immutable_record
from factorlab.discovery.work_lease import FrozenWorkLease, _boot_id
from factorlab.protocol import fingerprint, project_path


def _read(path):
    if path.stat().st_size > 65536:
        raise ValueError('S preparation receipt exceeds 64 KiB')
    value = json.loads(path.read_text('utf-8'))
    if set(value) != {'record', 'hash'} or value['hash'] != fingerprint(value['record']):
        raise ValueError('S preparation identity changed')
    return value['record']


def _write(path, value):
    if len(json.dumps(value, allow_nan=False).encode()) > 64000:
        raise ValueError('S preparation receipt exceeds bound')
    immutable_record(path, {'record': value, 'hash': fingerprint(value)})


class SPreparationEnvelope:
    def __init__(self, *args, **kwargs):
        raise TypeError('Use SPreparationEnvelope.prepare or load')

    @classmethod
    def prepare(cls, authority, directory, month, *, config, spec, source_reference,
                schedule_hash, clock_scope='function_entry', lease=None, outer_context=None):
        # This must be the first operation in the trusted entry, before data load.
        if outer_context is not None:
            from .s_outer_context import SOuterContext, require_lease_headroom
            if type(outer_context) is not SOuterContext:
                raise TypeError('Actual durable S outer context required')
            require_lease_headroom(outer_context._verify()['clock'], min(300,config['research_seconds']))
        origin = FrozenWorkLease.issue(min(300, config['research_seconds'])) if lease is None else lease
        if type(origin) is not FrozenWorkLease or origin.parent_pid != os.getpid():
            raise ValueError('S preparation requires its original issuer lease')
        origin.assert_parent_alive()
        if type(authority) is not MonthlyWorkAuthority:
            raise TypeError('Real shared authority required')
        if (clock_scope not in {'function_entry', 'before_local_source_load', 'before_feed_snapshot'}
                or not isinstance(source_reference, dict) or not source_reference
                or not 30 <= config['research_seconds'] <= 7200
                or not isinstance(schedule_hash, str) or not re.fullmatch('[0-9a-f]{64}', schedule_hash)):
            raise ValueError('Invalid original preparation contract')
        directory = project_path(directory)
        key = fingerprint({'directory': str(directory), 'replay_month': month})
        root = authority.directory / 'S_preparations' / key
        mirror = directory / 'S_preparations' / month
        outer = None
        if outer_context is not None:
            from .s_outer_context import SOuterContext
            if type(outer_context) is not SOuterContext:
                raise TypeError('Actual durable S outer context required')
            outer = outer_context.bind(authority, directory, config, spec, source_reference, schedule_hash)
        static = {'version': 'v7-S-preparation-v1', 'key': key, 'directory': str(directory),
                  'month': month, 'config': config, 'spec': spec,
                  'authority_manifest_hash': authority.manifest_hash,
                  'source_hash': fingerprint(authority.manifest['source_files']),
                  'source_reference': source_reference, 'clock_scope': clock_scope,
                  'work_schedule_hash': schedule_hash, 'outer_scope': outer}
        with FileLock(str(root) + '.lock', timeout=10):
            if root.exists() or mirror.exists():
                result = cls.load(authority, directory, month)
                if any(fingerprint(result.identity[k]) != fingerprint(v) for k, v in static.items()):
                    raise ValueError('Original S preparation identity/source cannot change on reopening')
                return result  # Loading is never another grant of execution.
            authority.verify()
            seconds = config['research_seconds']
            envelope = {'issuer': origin.to_dict(), 'seconds': seconds,
                        'monotonic_deadline': origin.monotonic_started + seconds,
                        'wall_deadline': (datetime.fromisoformat(origin.wall_started_at)
                                          + timedelta(seconds=seconds)).isoformat()}
            if outer is not None:
                from .s_outer_context import effective_deadlines
                effective=effective_deadlines(outer['clock'])
                envelope['monotonic_deadline'] = min(envelope['monotonic_deadline'], effective['monotonic_deadline'])
                envelope['wall_deadline'] = min(envelope['wall_deadline'], effective['wall_deadline'])
            if (origin.monotonic_deadline > envelope['monotonic_deadline']
                    or origin.wall_deadline > envelope['wall_deadline']):
                raise TimeoutError('Original S scope has insufficient remaining time for baseline work')
            identity = json.loads(json.dumps({**static, 'execution_envelope': envelope}, allow_nan=False))
            for path in (root / 'identity.json', mirror / 'identity.json'):
                _write(path, identity)
            result = cls._loaded(authority, directory, month, root, mirror, identity, True)
            result.check()
            # The provisional reservation binds actual source references, not an
            # invented final data_id; seal_input later adds the immutable link.
            record = authority._reserve_with_lease('S-baseline:' + key, counts={'solve': 7},
                binding={'role': 'S', 'stage': 'baseline', 'preparation_hash': result.digest,
                         'bridge_id': key, 'source_reference': source_reference}, lease=origin)
            for path in (root / 'baseline.json', mirror / 'baseline.json'):
                _write(path, {'reservation_id': record['reservation_id'], 'terms_hash': record['terms_hash']})
            origin.assert_parent_alive()
            return result

    @classmethod
    def _loaded(cls, authority, directory, month, root, mirror, identity, executable):
        obj = object.__new__(cls)
        obj.authority, obj.directory, obj.month = authority, directory, month
        obj.root, obj.mirror, obj.identity = root, mirror, identity
        obj.digest = fingerprint(identity)
        obj._executable = executable
        return obj

    @classmethod
    def load(cls, authority, directory, month):
        directory = project_path(directory)
        key = fingerprint({'directory': str(directory), 'replay_month': month})
        root, mirror = authority.directory / 'S_preparations' / key, directory / 'S_preparations' / month
        identity = _read(root / 'identity.json')
        if (identity != _read(mirror / 'identity.json') or identity['key'] != key
                or identity['authority_manifest_hash'] != authority.manifest_hash
                or identity['source_hash'] != fingerprint(authority.manifest['source_files'])):
            raise ValueError('Missing or changed original S preparation')
        return cls._loaded(authority, directory, month, root, mirror, identity, False)

    def check(self):
        if not self._executable:
            raise ValueError('Recovered S preparation is readback only')
        self.authority.verify()
        if (fingerprint(self.identity) != self.digest
                or _read(self.root / 'identity.json') != self.identity
                or _read(self.mirror / 'identity.json') != self.identity):
            raise ValueError('Original S preparation identity changed')
        if self.identity.get('outer_scope') is not None:
            from .s_outer_context import SOuterContext
            SOuterContext.verify_bound(self.identity['outer_scope'])
        envelope = self.identity['execution_envelope']
        origin = FrozenWorkLease.from_dict(envelope['issuer'])
        wall, mono = datetime.now(UTC), time.monotonic()
        if (os.getpid() != origin.parent_pid or psutil.Process().create_time() != origin.parent_created
                or _boot_id() != origin.boot_id or mono < origin.monotonic_started
                or wall < datetime.fromisoformat(origin.wall_started_at)
                or wall.strftime('%Y-%m') != origin.wall_started_at[:7]):
            raise ValueError('Original S preparation process/clock changed')
        if mono >= envelope['monotonic_deadline'] or wall >= datetime.fromisoformat(envelope['wall_deadline']):
            raise TimeoutError('Original S preparation deadline consumed')

    def baseline_record(self):
        value = _read(self.root / 'baseline.json')
        if value != _read(self.mirror / 'baseline.json'):
            raise ValueError('Original baseline reservation anchor changed')
        record = self.authority.get(value['reservation_id'])
        if (record['terms_hash'] != value['terms_hash']
                or record['binding']['preparation_hash'] != self.digest
                or record['lease'] != self.identity['execution_envelope']['issuer']):
            raise ValueError('Original baseline reservation differs from S preparation')
        return record

    def start_preprocessing(self):
        self.check()
        record = self.baseline_record()
        if record['status'] != 'reserved':
            raise ValueError('Preprocessing cannot execute or resume twice')
        record = self.authority.start(record['reservation_id'],
            {'pid': os.getpid(), 'created': psutil.Process().create_time(), 'mode': 'S_parent'})
        if record['status'] != 'running':
            raise ValueError('Original preprocessing lease already closed')
        FrozenWorkLease.from_dict(record['lease']).assert_parent_alive()
        return record

    def seal_input(self, *, data_id, search_exposure):
        self.check()
        record = self.baseline_record()
        FrozenWorkLease.from_dict(record['lease']).assert_parent_alive()
        if record['status'] != 'running' or not data_id or search_exposure.get('data_id') != data_id:
            raise ValueError('Only the original running preparation can seal its input')
        body = {'preparation_hash': self.digest, 'data_id': data_id, 'search_exposure': search_exposure}
        for root in (self.root, self.mirror):
            _write(root / 'input.json', body)
        FrozenWorkLease.from_dict(record['lease']).assert_parent_alive()
        return body

    def verified_bridge_envelope(self, store, month, state):
        if self._executable:
            self.check()
        else:
            self.authority.verify()
        body = _read(self.root / 'input.json')
        if (body != _read(self.mirror / 'input.json') or body['preparation_hash'] != self.digest
                or body['data_id'] != state['data_id'] or body['search_exposure'] != state['search_exposure']
                or month != self.month or str(store.directory) != str(self.directory)
                or fingerprint(store.identity['config']) != fingerprint(self.identity['config'])
                or fingerprint(store.identity['rolling_spec']) != fingerprint(self.identity['spec'])
                or fingerprint(store.identity['source_files']) != self.identity['source_hash']):
            raise ValueError('Sealed preparation differs from actual RunStore/input contract')
        return self.identity['execution_envelope']
