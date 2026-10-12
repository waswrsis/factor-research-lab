"""Fixed original finite verification service worker; no registration/resume.

Success means this worker finished its exact batch and is ready to exit zero.
Retained original systemd observations must independently authenticate that exit.
"""
from __future__ import annotations
import os
from pathlib import Path, PurePosixPath
import re
import sys
import threading

RELATIVE = 'factorlab/discovery/nominal_verification_service_worker.py'
MODULE = 'factorlab.discovery.nominal_verification_service_worker'
MAX_TASKS = 23
SERVICE_SECONDS = 7200


def _arguments(argv):
    """Bounded lexical parsing only; no filesystem, registry or allocation."""
    if (type(argv) is not list or len(argv) != 4
            or any(type(x) is not str for x in argv)
            or argv[0] != '--registry' or argv[2] != '--service-index'):
        raise ValueError('Fixed --registry ABS_PATH --service-index INDEX required')
    path, text = argv[1], argv[3]
    if (not 0 < len(path) <= 4096 or not path.startswith('/')
            or path.startswith('//') or str(PurePosixPath(path)) != path or '..' in PurePosixPath(path).parts
            or any(ord(c) < 32 or ord(c) == 127 for c in path)):
        raise ValueError('Canonical absolute original registry path required')
    if not re.fullmatch(r'[1-9][0-9]{0,11}', text):
        raise ValueError('Canonical positive service index; registration0 forbidden')
    return path, int(text)


def _canonical(registry):
    from factorlab.protocol import project_path
    from .monthly_work_authority import source_files
    from .evidence import file_hash
    from . import native_verification_batch, nominal_systemd_unit
    from . import native_verification_launcher, nominal_verification_budget
    files = source_files()
    modules = (native_verification_batch, nominal_systemd_unit,
               native_verification_launcher, nominal_verification_budget)
    for module in modules:
        relative = 'factorlab/discovery/' + module.__name__.rsplit('.', 1)[1] + '.py'
        if (Path(module.__file__).resolve() != project_path(relative).resolve()
                or files.get(relative) != file_hash(module.__file__)):
            raise ValueError('Canonical service/batch/unit dependencies required')
    if (Path(__file__).resolve() != project_path(RELATIVE).resolve()
            or files.get(RELATIVE) != file_hash(__file__)
            or files != registry.plan.binding['source_files']):
        raise ValueError('New exact source-bound service worker registration required')
    return files


def _execute(registry, index):
    from .nominal_verification_budget import VerificationRegistry
    from .native_verification_batch import VerificationBatch, VERSION
    if type(registry) is not VerificationRegistry or type(index) is not int or index <= 0:
        raise TypeError('Exact original registry and non-registration service required')
    _canonical(registry)
    registry.verify()
    plan = registry.plan.service(index)
    start, end = plan['task_start'], plan['task_end_exclusive']
    if (plan['index'] != index or type(start) is not int or type(end) is not int
            or start <= 0 or not 0 < end-start <= MAX_TASKS
            or plan['wall_seconds'] != SERVICE_SECONDS or plan['retry_allowed'] is not False
            or any(registry.plan.task(i)['service'] != index for i in range(start, end))):
        raise ValueError('Original fixed service group and7200-second envelope required')
    service = registry.service(index)  # Exactly one original allocation, never a retry.
    batch = VerificationBatch(service)  # Binds actual current retained systemd unit.
    for task_index in range(start, end):
        task = service.acquire(task_index)
        batch.run_task(task)  # Exception stops this process; no next slot or service.
    result = batch.finish()  # Owns original service.close, inventory and natural reap.
    if (type(result) is not dict or result.get('version') != VERSION
            or result.get('provisional_only') is not True
            or result.get('cleanup_signals') is not False
            or type(result.get('reaped')) is not list
            or [row.get('task_index') for row in result['reaped']] != list(range(start, end))
            or batch.closed is not True or batch.failed is not False or service.closed is not True):
        raise ValueError('Original complete batch finish unavailable; cannot exit zero')
    # Closed resource state does not renew time. This is the same original clock.
    service.clock.assert_parent_alive()
    # No service.close here: finish already called it once. No cleanup can turn
    # any exception into zero. Process failure remains an original unit failure.
    return 0


def main(argv=None):
    try:
        directory, index = _arguments(sys.argv[1:] if argv is None else argv)
    except (TypeError, ValueError) as error:
        print(type(error).__name__ + ': ' + str(error), file=sys.stderr)
        return 2
    if (sys.platform != 'linux' or os.name != 'posix'
            or threading.current_thread() is not threading.main_thread()):
        print('Original Linux issuer main-thread worker required', file=sys.stderr)
        return 2
    try:
        from factorlab.protocol import project_path
        from .nominal_verification_budget import VerificationRegistry
        path = project_path(directory)
        if str(path) != directory:
            raise ValueError('Original registry path must not redirect or alias')
        registry = VerificationRegistry(path)
        return _execute(registry, index)
    except BaseException as error:
        # Existing original registry/service/task/batch anchors preserve consumption.
        # No reset, alternate registry, next service or success-looking cleanup.
        print(type(error).__name__ + ': ' + str(error)[:1000], file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
