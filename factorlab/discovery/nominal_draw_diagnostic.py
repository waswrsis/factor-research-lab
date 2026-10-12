"""Prospective fixed draw telemetry; never native or statistical authority.

Every diagnostic operation in draw_scope is charged to the original draw lease.
Missing/invalid telemetry cannot authorize anything. It does not suppress an
original draw exception, displace a foreign profiler or retry an allocation.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import sys
import threading
import time

from factorlab.protocol import fingerprint, project_path

RELATIVE = 'factorlab/discovery/nominal_draw_diagnostic.py'
VERSION = 'nominal-draw-diagnostic-hook-v1'
ACTUAL_VERSION = 'v7-nominal-actual-trial-v5-native-draw'


def _bound(registry, intent, paths, mirror, lease):
    from . import nominal_actual_trial as original
    from .evidence import file_hash
    from .work_lease import FrozenWorkLease
    if type(registry) is not original.NominalActualRegistry or type(lease) is not FrozenWorkLease:
        raise TypeError('Exact original draw registry and original lease required')
    if (lease.lease_seconds != original.SAMPLE_SECONDS or lease.lease_seconds != 10
            or intent['lease'] != lease.to_dict()
            or intent['registry_directory'] != str(registry.directory)
            or intent['source_files'] != registry.identity['source_files']
            or intent['coordinate']['registry'] != registry.digest
            or intent['paths'] != {k: str(v) for k, v in paths.items()}
            or mirror != registry.directory/'draw_allocations'/fingerprint(intent['coordinate'])
            or original._read(paths['root']/'draw-intent.json') != intent
            or original._read(mirror/'intent.json') != intent):
        raise ValueError('Diagnostic must use both actual original allocation intents')
    for relative, module in ((RELATIVE, sys.modules[__name__]), (original.RELATIVE, original)):
        if (Path(module.__file__).resolve() != project_path(relative).resolve()
                or intent['source_files'].get(relative) != file_hash(module.__file__)):
            raise ValueError('Prospective canonical diagnostic source required')
    lease.assert_parent_alive()


def _profiler(intent):
    from .nominal_draw_measurement import TARGETS, CountsProfiler, prepare_source_plan
    # Fixed imports only; no caller module, callback, counter or command selection.
    for module, _ in TARGETS:
        importlib.import_module('factorlab.discovery.' + module)
    plan = prepare_source_plan(project_path('.'), intent['source_files'])
    return CountsProfiler.for_loaded_source(plan)


@contextmanager
def draw_scope(registry, intent, paths, mirror, lease):
    """Only the fixed original draw calls this after its two durable intents.

    Diagnostic defects invalidate telemetry, not the original scientific checks.
    Setup, counting, teardown and attempted writes still consume original time.
    The caller MUST perform its original lease check after this scope returns.
    """
    from . import nominal_actual_trial as original
    from .work_lease import _observe
    began = time.monotonic_ns()
    owner = [os.getpid(), threading.get_ident()]
    profiler = None
    entered = False
    errors = []
    body_error = None
    returned = False
    try:
        _bound(registry, intent, paths, mirror, lease)
        profiler = _profiler(intent)
        profiler.__enter__()
        entered = True
    except Exception as error:
        errors.append('setup:' + type(error).__name__)
    try:
        yield
        returned = True
    except BaseException as error:
        body_error = error
        raise
    finally:
        if entered:
            try:
                profiler.__exit__(type(body_error) if body_error else None,
                                  body_error, body_error.__traceback__ if body_error else None)
            except Exception as error:
                errors.append('teardown:' + type(error).__name__)
        counters = None
        if profiler is not None:
            try:
                counters = profiler.snapshot()
            except Exception as error:
                errors.append('snapshot:' + type(error).__name__)
        try:
            # This observation is real original-lease data, not a success flag.
            finished = _observe(lease)
            record = {'version': VERSION, 'diagnostic_only': True,
                'acceptance_evidence': False, 'native_completion': None,
                'native_completion_status': 'requires_separate_original_OS_observation',
                'registry_digest': registry.digest, 'coordinate': intent['coordinate'],
                'intent_hash': fingerprint(intent), 'lease': lease.to_dict(),
                'lease_digest': lease.digest, 'source_digest': fingerprint(intent['source_files']),
                'intent_paths': [str(paths['root']/'draw-intent.json'), str(mirror/'intent.json')],
                'owner': owner, 'start_monotonic_ns': began,
                'end_monotonic_ns': time.monotonic_ns(), 'physical_observation': finished,
                'original_scope_returned': returned,
                'original_error_type': type(body_error).__name__ if body_error else None,
                'diagnostic_errors': errors, 'counters': counters,
                'publication_and_final_lease_check_are_after_this_observation': True}
            if len(json.dumps(record, allow_nan=False).encode('utf8')) > 65536:
                raise ValueError('Bounded diagnostic record exceeded')
            original._write(paths['root']/'draw-diagnostic.json', record)
            original._write(mirror/'diagnostic.json', record)
        except Exception as error:
            if body_error is not None:
                body_error.add_note('Diagnostic unavailable: ' + type(error).__name__)
            # A diagnostic write is not a required scientific publication. The
            # caller's same original final lease assertion is never suppressed.


def read_diagnostic(registry, cell, index):
    """Read telemetry only; deliberately cannot return accepted/native=True.

    Original intent/source labels are checked, but no retained unit is consulted.
    Use the original trial/native readers for all authority. This is not a free
    replacement for paid law, source, native completion or scientific verification.
    """
    from . import nominal_actual_trial as original
    from .work_lease import FrozenWorkLease
    if type(registry) is not original.NominalActualRegistry:
        raise TypeError('Exact original registry required')
    coordinate, paths = registry._allocation_header(cell, index)
    mirror = registry.directory/'draw_allocations'/fingerprint(coordinate)
    intent = original._read(paths['root']/'draw-intent.json')
    record = original._read(paths['root']/'draw-diagnostic.json')
    lease = FrozenWorkLease.from_dict(intent['lease'])
    if (intent != original._read(mirror/'intent.json')
            or record != original._read(mirror/'diagnostic.json')
            or record['version'] != VERSION or record['diagnostic_only'] is not True
            or record['acceptance_evidence'] is not False or record['native_completion'] is not None
            or record['coordinate'] != coordinate or record['registry_digest'] != registry.digest
            or intent['source_files'] != registry.identity['source_files']
            or record['source_digest'] != fingerprint(intent['source_files'])
            or record['intent_hash'] != fingerprint(intent)
            or record['lease'] != lease.to_dict() or record['lease_digest'] != lease.digest
            or lease.lease_seconds != 10
            or record['intent_paths'] != [str(paths['root']/'draw-intent.json'), str(mirror/'intent.json')]):
        raise ValueError('Original telemetry binding changed')
    return json.loads(json.dumps(record, allow_nan=False))


def main(argv=None):
    from .nominal_draw_worker import main as fixed_main
    return fixed_main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
