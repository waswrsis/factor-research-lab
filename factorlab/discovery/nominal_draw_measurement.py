"""Diagnostic counters only. No lease, draw, authority, cache or accepted result.

Future integration must place this scope inside the original allocated 10-second
draw try/except and retain its original failure accounting. This module alone
does not execute or authorize that integration.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import types

RELATIVE = 'factorlab/discovery/nominal_draw_measurement.py'
TARGETS = (
    ('nominal_actual_trial', 'NominalActualRegistry._verify_uncached'),
    ('nominal_native_completion', 'NativeCompletionReader.verify_task'),
    ('nominal_native_completion', 'NativeCompletionReader._history'),
    ('nominal_systemd_unit', '_retained'),
    ('nominal_systemd_unit', '_batch_closure'),
    ('nominal_systemd_unit', '_unit'),
    ('nominal_systemd_unit', '_call'),
    ('monthly_work_authority', 'source_files'),
    ('nominal_verification_budget', 'VerificationRegistry.service_witness'),
    ('nominal_verification_budget', 'VerificationRegistry.task_witness'),
)
MAX_COUNT = (1 << 63) - 1
MAX_SOURCE_BYTES = 64 * 2**20
MAX_FILE_BYTES = 2 * 2**20
_KEY = object()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _path(root, name):
    if (type(name) is not str or '\\' in name or name.startswith('/')
            or '..' in Path(name).parts or ':' in name):
        raise ValueError('Relative frozen source path required')
    path = root / name
    if path.resolve() != path or not path.is_file():
        raise ValueError('Canonical nonredirected source required')
    return path


def _code(tree, qualname):
    found = []
    def walk(item):
        if item.co_qualname == qualname:
            found.append(item)
        for value in item.co_consts:
            if isinstance(value, types.CodeType):
                walk(value)
    walk(tree)
    if len(found) != 1:
        raise ValueError('One exact compiled target required: ' + qualname)
    return found[0]


def prepare_source_plan(root, files):
    """Read/hash/compile only. Does not import target modules or run any target."""
    root = Path(root).resolve()
    if type(files) is not dict or not files or len(files) > 1024:
        raise ValueError('Bounded complete source inventory required')
    frozen = dict(files)
    total = 0
    for name, digest in frozen.items():
        path = _path(root,name)
        size = path.stat().st_size
        total += size
        if size > MAX_FILE_BYTES or total > MAX_SOURCE_BYTES:
            raise ValueError('Diagnostic source read bound exceeded')
        if type(digest) is not str or len(digest) != 64 or _sha(path.read_bytes()) != digest:
            raise ValueError('Frozen source differs: ' + name)
    targets = []
    for module, qualname in TARGETS:
        relative = 'factorlab/discovery/' + module + '.py'
        if relative not in frozen:
            raise ValueError('Target missing from original source inventory')
        path = _path(root, relative)
        code = _code(compile(path.read_bytes(), str(path), 'exec', dont_inherit=True,
                             optimize=sys.flags.optimize), qualname)
        targets.append({'module': 'factorlab.discovery.'+module, 'qualname':qualname,
                        'relative':relative, 'first_line':code.co_firstlineno})
    return {'version':'nominal-draw-count-plan-v1', 'root':str(root), 'files':frozen,
            'targets':targets, 'profiler_installed_in_source':RELATIVE in frozen,
            'source_digest':_sha(json.dumps(frozen, sort_keys=True,separators=(',',':')).encode()),
            'execution_authorized':False, 'diagnostic_only':True,
            'original_draw_seconds':10, 'counter_limit':MAX_COUNT}


class CountsProfiler:
    def __init__(self, key, codes, plan=None):
        if key is not _KEY or type(self) is not CountsProfiler:
            raise TypeError('Use fixed installed source binding')
        self._codes = {id(code):(label,code) for label,code in codes}
        if len(self._codes) != len(codes):
            raise ValueError('Duplicate code object')
        self._counts = {label:[0,0] for label,code in codes}
        self._plan = plan
        self._owner = (os.getpid(), threading.get_ident())
        self._state = 'new'
        self._invalid = []
        self._callback = self._event
        self._start = self._end = None
        self._bindings = []

    @classmethod
    def for_loaded_source(cls, plan):
        if cls is not CountsProfiler:
            raise TypeError('Fixed diagnostic profiler implementation only')
        if type(plan) is not dict:
            raise TypeError('Prepared diagnostic source plan required')
        expected = prepare_source_plan(plan['root'], plan['files'])
        if expected != plan or not expected['profiler_installed_in_source']:
            raise ValueError('Fresh source must include profiler; existing frozen191 cannot be instrumented')
        own = _path(Path(plan['root']), RELATIVE)
        if Path(__file__).resolve() != own:
            raise ValueError('Profiler is not installed in that fresh canonical source')
        codes = []
        bindings = []
        for item in plan['targets']:
            module = sys.modules.get(item['module'])
            path = _path(Path(plan['root']), item['relative'])
            if type(module) is not types.ModuleType or Path(module.__file__).resolve() != path:
                raise ValueError('Original target module must already be loaded canonically')
            function = module
            for part in item['qualname'].split('.'):
                function = vars(function)[part]
            expected_code = _code(compile(path.read_bytes(), str(path), 'exec',
                dont_inherit=True, optimize=sys.flags.optimize), item['qualname'])
            if (type(function) is not types.FunctionType or function.__code__ != expected_code
                    or Path(function.__code__.co_filename).resolve() != path):
                raise ValueError('Loaded target code differs from frozen source')
            codes.append((item['module']+'.'+item['qualname'], function.__code__))
            bindings.append((module,item['qualname'],function,function.__code__))
        result = cls(_KEY,codes,json.loads(json.dumps(plan)))
        result._bindings = bindings
        return result

    def _source(self):
        if self._plan is not None:
            if prepare_source_plan(self._plan['root'],self._plan['files']) != self._plan:
                raise ValueError('Diagnostic source binding changed')
        for module,qualname,function,code in self._bindings:
            if sys.modules.get(module.__name__) is not module:
                raise ValueError('Original loaded module identity changed')
            current = module
            for part in qualname.split('.'):
                current = vars(current)[part]
            if current is not function or function.__code__ is not code:
                raise ValueError('Original function or code object changed')

    def __enter__(self):
        if self._state != 'new' or self._owner != (os.getpid(),threading.get_ident()):
            raise ValueError('One original current-thread scope only')
        if sys.getprofile() is not None:
            raise ValueError('Existing current-thread profiler must not be displaced')
        self._start = time.monotonic_ns()
        self._source()
        self._state = 'active'
        sys.setprofile(self._callback)
        return self

    def _event(self, frame, event, arg):
        # Deliberately ignore arg: no return values/arguments/locals are observed.
        # Only exact identity is counted: CodeType equality would count clones.
        try:
            if self._state != 'active' or event not in ('call','return'):
                return
            if self._owner != (os.getpid(),threading.get_ident()):
                if 'owner_changed' not in self._invalid:self._invalid.append('owner_changed')
                return
            item = self._codes.get(id(frame.f_code))
            if item is not None and item[1] is frame.f_code:
                values = self._counts[item[0]]
                index = 0 if event == 'call' else 1
                if values[index] < MAX_COUNT:
                    values[index] += 1
                elif 'counter_saturated' not in self._invalid:
                    self._invalid.append('counter_saturated')
        except BaseException:
            # A counting fault must not supply a decision or suppress a target
            # error. Its telemetry is incomplete, never a native-success signal.
            self._state = 'invalid'

    def __exit__(self, kind, error, traceback):
        try:
            if self._owner != (os.getpid(),threading.get_ident()):
                raise ValueError('Profiler exit belongs to original thread only')
            if sys.getprofile() is not self._callback:
                raise ValueError('Original profiler was replaced; do not overwrite foreign profiler')
            sys.setprofile(None)
            if self._state != 'active':
                raise ValueError('Counting callback failed')
            self._source()
        except BaseException as problem:
            self._invalid.append(type(problem).__name__+': '+str(problem))
            if error is not None:
                error.add_note('Diagnostic counter scope invalid: '+str(problem))
            else:
                raise
        finally:
            self._end = time.monotonic_ns()
            self._state = 'closed'
        return False

    def snapshot(self):
        if self._state != 'closed' or self._owner != (os.getpid(),threading.get_ident()):
            raise ValueError('Only ended original-thread diagnostic may be exported')
        return {'version':'nominal-draw-counts-diagnostic-v1','diagnostic_only':True,
                'acceptance_evidence':False,'owner_pid':self._owner[0],
                'owner_thread':self._owner[1],'start_monotonic_ns':self._start,
                'end_monotonic_ns':self._end,'invalid':list(self._invalid),
                'source_digest':self._plan['source_digest'] if self._plan else None,
                'counts':{k:{'call':v[0],'return_or_exception_unwind':v[1]}
                          for k,v in self._counts.items()},
                'scope':'Python entries of fixed native-reader wrappers; no claim each entry completed OS I/O'}
