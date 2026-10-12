"""Fixed two-batch nominal simulation worker; no draw, registration or retry.

The producer binds the actual original OS service before heavy registry reads.
A zero exit remains provisional until the retained original unit proves it.
"""
from __future__ import annotations
import os
from pathlib import PurePosixPath
import re
import sys
import threading

RELATIVE = 'factorlab/discovery/nominal_trial_native_worker.py'
MODULE = 'factorlab.discovery.nominal_trial_native_worker'


def _arguments(argv):
    """Lexical checks only. No filesystem, authority, clock or trial access."""
    if (type(argv) is not list or len(argv) != 8
            or any(type(v) is not str for v in argv)
            or argv[::2] != ['--registry', '--cell', '--trial-index', '--batch-index']):
        raise ValueError('Fixed registry/cell/trial-index/batch-index arguments required')
    directory, cell, index, batch = argv[1::2]
    if (not 0 < len(directory) <= 4096 or not directory.startswith('/')
            or directory.startswith('//') or str(PurePosixPath(directory)) != directory
            or '..' in PurePosixPath(directory).parts
            or any(ord(c) < 32 or ord(c) == 127 for c in directory)):
        raise ValueError('Canonical absolute original registry path required')
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_-]{0,79}', cell):
        raise ValueError('Literal registered cell name required')
    if not re.fullmatch(r'0|[1-9][0-9]{0,3}', index) or int(index) >= 2000:
        raise ValueError('Canonical original trial index 0..1999 required')
    if batch not in ('0', '1'):
        raise ValueError('Only the original two registered simulation work batches are supported')
    return directory, cell, int(index), int(batch)


def _execute(directory, cell, index, batch_index):
    # This module must keep heavyweight registry/controller imports inside the
    # producer, after its actual unit preflight. No worker-side source/quota scan.
    from .nominal_trial_native_completion import run_original_batch
    return run_original_batch(directory, cell, index, batch_index)


def _completed_result(result, cell, index, batch_index):
    """Schema consistency only; original producer and retained OS confer authority."""
    keys = {'version', 'coordinate', 'batch_index', 'controller_terminal_hash',
            'controller_successor_hash', 'native_binding_hash', 'provisional_only', 'cleanup_signals'}
    if (type(result) is not dict or set(result) != keys
            or result['version'] != 'nominal-trial-native-return-v1'
            or type(result['batch_index']) is not int or result['batch_index'] != batch_index
            or result['provisional_only'] is not True or result['cleanup_signals'] is not False):
        raise ValueError('Original completed producer result unavailable')
    c = result['coordinate']
    if (type(c) is not dict or set(c) != {'registry','cell','cell_ordinal','index','scenario_digest'}
            or c['cell'] != cell or type(c['index']) is not int or c['index'] != index
            or type(c['cell_ordinal']) is not int or not 0 <= c['cell_ordinal'] < 64):
        raise ValueError('Original result coordinate differs')
    for value in (c['registry'], c['scenario_digest'], result['controller_terminal_hash'],
                  result['controller_successor_hash'], result['native_binding_hash']):
        if type(value) is not str or not re.fullmatch(r'[0-9a-f]{64}', value):
            raise ValueError('Original bound result hashes required')


def main(argv=None):
    try:
        coordinate = _arguments(sys.argv[1:] if argv is None else argv)
    except (TypeError, ValueError) as error:
        print(type(error).__name__ + ': ' + str(error), file=sys.stderr)
        return 2
    if (sys.platform != 'linux' or os.name != 'posix'
            or threading.current_thread() is not threading.main_thread()):
        print('Original Linux main-thread nominal trial worker required', file=sys.stderr)
        return 2
    try:
        result = _execute(*coordinate)
        _completed_result(result, *coordinate[1:])
        return 0
    except BaseException as error:
        # No retry, draw, alternate service, recovery or successful cleanup.
        print(type(error).__name__ + ': ' + str(error)[:1000], file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
