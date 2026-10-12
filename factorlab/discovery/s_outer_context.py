"""Trusted durable outer S scopes; never a new spendable wallet or lease."""
import json
import sqlite3
import time
from datetime import UTC, datetime, timedelta

import psutil

from factorlab.discovery.work_lease import FrozenWorkLease, _boot_id
from factorlab.protocol import fingerprint, project_path


def original_clock(seconds):
    if type(seconds) is not int or not 30 <= seconds <= 7200:
        raise ValueError('Invalid original S scope')
    origin = FrozenWorkLease.issue(300)
    value = {'issuer': origin.to_dict(), 'seconds': seconds,
            'wall_deadline': (datetime.fromisoformat(origin.wall_started_at)
                              + timedelta(seconds=seconds)).isoformat(),
            'monotonic_deadline': origin.monotonic_started + seconds}
    from .forward_service_scope import current_service_reference
    restriction=current_service_reference()
    if restriction is not None:
        value.update(version='original-S-service-restriction-v1',service_restriction=restriction)
    return value


def check_clock(value):
    return _check_clock(value,verify_source=True)


def _check_clock_live(value):
    # Repeated watchdog polls verify original deadline and coherent durable
    # identity; paid entry/completion still use the full source check.
    return _check_clock(value,verify_source=False)


def _check_clock(value, *, verify_source):
    required={'issuer','seconds','wall_deadline','monotonic_deadline'}
    if 'service_restriction' in value:
        required.update({'version','service_restriction'})
        if value.get('version')!='original-S-service-restriction-v1':
            raise ValueError('Unknown original S service restriction')
        from .forward_service_scope import verify_service_reference
        verify_service_reference(value['service_restriction'],_verify_source=verify_source)
    if set(value)!=required:
        raise ValueError('Original S clock fields changed')
    origin = FrozenWorkLease.from_dict(value['issuer'])
    seconds = value['seconds']
    if (type(seconds) is not int or not 30 <= seconds <= 7200
            or value['monotonic_deadline'] != origin.monotonic_started + seconds
            or datetime.fromisoformat(value['wall_deadline']) !=
            datetime.fromisoformat(origin.wall_started_at) + timedelta(seconds=seconds)):
        raise ValueError('Original outer clock changed')
    wall, mono = datetime.now(UTC), time.monotonic()
    try:
        parent = psutil.Process(origin.parent_pid)
        valid = parent.create_time() == origin.parent_created and parent.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        valid = False
    if (not valid or _boot_id() != origin.boot_id or mono < origin.monotonic_started
            or wall < datetime.fromisoformat(origin.wall_started_at)
            or wall.strftime('%Y-%m') != origin.wall_started_at[:7]):
        raise ValueError('Original outer process/clock lost')
    if mono >= value['monotonic_deadline'] or wall >= datetime.fromisoformat(value['wall_deadline']):
        raise TimeoutError('Original outer S deadline consumed')
    return value


def _read(path):
    value = json.loads(path.read_text('utf-8'))
    if set(value) != {'value', 'sha256'} or value['sha256'] != fingerprint(value['value']):
        raise ValueError('Outer S artifact changed')
    return value['value']


class SOuterContext:
    def __init__(self, *args, **kwargs):
        raise TypeError('Use a verified durable outer scope')

    @classmethod
    def autonomous(cls, job_path, attempt, token):
        obj = object.__new__(cls)
        obj.reference = {'kind': 'autonomous', 'path': str(project_path(job_path)),
                         'attempt': attempt, 'token': token}
        obj._verify()
        return obj

    @classmethod
    def controls(cls, directory, arm):
        obj = object.__new__(cls)
        obj.reference = {'kind': 'controls', 'path': str(project_path(directory)), 'arm': arm}
        obj._verify()
        return obj

    def _verify(self):
        ref = self.reference
        if ref['kind'] == 'autonomous':
            path = project_path(ref['path'])
            job = _read(path)
            db_path = path.parents[2] / 'scheduler.sqlite'
            with sqlite3.connect(db_path.as_uri() + '?mode=ro', uri=True) as db:
                row = db.execute('SELECT value,hash FROM jobs WHERE month=?', (job['month'],)).fetchone()
            if row is None:
                raise ValueError('Missing original scheduler reservation')
            state = json.loads(row[0])
            if (fingerprint(state) != row[1] or state['job_hash'] != fingerprint(job)
                    or state['job_path'] != str(path) or state['status'] != 'running'
                    or state['attempts'][-1]['number'] != ref['attempt']
                    or state['attempts'][-1]['token'] != ref['token']
                    or job['version'] != (3 if 'service_restriction' in job['outer_clock'] else 2)
                    or job['deadline'] != job['outer_clock']['wall_deadline']):
                raise ValueError('Outer scope differs from the actual running research job')
            _verify_job_clock_binding(job,path)
            result = {'clock': job['outer_clock'], 'authority': job['work_authority'],
                      'directory': job['run_directory'], 'schedule_hash': fingerprint(job['work_schedule']),
                      'config': job['config']['research'], 'spec': job['rolling'],
                      'source_reference': {'feed_identity': job['feed_id'],
                          'feed_directory': job['feed_directory'], 'exposure_path': job['exposure_path'],
                          'cutoff': job['cutoff']}, 'scope_hash': fingerprint(job)}
            if job.get('observation_prefix') is not None:
                result['source_reference']['observation_prefix'] = job['observation_prefix']
        elif ref['kind'] == 'controls':
            from .shared_controls_executor import SharedControlsExecutor
            result = SharedControlsExecutor.verify_active(ref['path'], ref['arm'])
        else:
            raise ValueError('Unknown outer scope')
        check_clock(result['clock'])
        return result

    def bind(self, authority, directory, config, spec, source_reference, schedule_hash):
        result = self._verify()
        if (result['authority'] != {'directory': str(authority.directory), 'manifest_hash': authority.manifest_hash}
                or result['directory'] != str(project_path(directory))
                or fingerprint(result['config']) != fingerprint(config)
                or fingerprint(result['spec']) != fingerprint(spec)
                or result['source_reference'] != source_reference
                or result['schedule_hash'] != schedule_hash):
            raise ValueError('Outer scope does not authorize this original S invocation')
        return {'reference': self.reference, 'scope_hash': result['scope_hash'], 'clock': result['clock']}

    @classmethod
    def verify_bound(cls, value):
        obj = object.__new__(cls)
        obj.reference = value['reference']
        verified = obj._verify()
        if value['scope_hash'] != verified['scope_hash'] or value['clock'] != verified['clock']:
            raise ValueError('Bound outer S scope changed')
        return verified



def effective_deadlines(clock):
    """Original requested S clock stays intact; only its usable envelope narrows."""
    check_clock(clock)
    wall,mono=clock['wall_deadline'],clock['monotonic_deadline']
    if 'service_restriction' in clock:
        ref=clock['service_restriction']
        wall=min(wall,ref['wall_deadline']);mono=min(mono,ref['monotonic_deadline'])
    return {'wall_deadline':wall,'monotonic_deadline':mono}


def _verify_job_clock_binding(job,path):
    runtime=project_path(path).parents[3]
    restriction=job['outer_clock'].get('service_restriction')
    if restriction is None:
        if job['version']!=2 or any((runtime/'service').glob('*/native-clock.json')):
            raise ValueError('Original service restriction cannot be removed from research')
    elif (job['version']!=3 or project_path(restriction['directory']).parents[1]!=runtime):
        raise ValueError('Research restriction belongs to another original runtime')
    if job['deadline']!=job['outer_clock']['wall_deadline']:
        raise ValueError('Research original requested deadline changed')


def verify_job_clock(job,path):
    _verify_job_clock_binding(job,path)
    check_clock(job['outer_clock'])


def require_lease_headroom(clock,seconds=300):
    bound=effective_deadlines(clock)
    if (time.monotonic()+seconds>bound['monotonic_deadline']
            or datetime.now(UTC)+timedelta(seconds=seconds)>datetime.fromisoformat(bound['wall_deadline'])):
        raise TimeoutError('Original effective S scope lacks the full unchanged work lease')
    return bound
