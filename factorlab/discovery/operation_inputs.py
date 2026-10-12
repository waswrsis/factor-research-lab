"""One-operation numerical input reuse, never authorization caching.

Every caller still performs the original lifecycle, SQL, publication, source,
model and actual-feed checks. This adapter replaces only build_rolling_data;
it accepts no cached permission or caller-provided verified flag. An operation
must explicitly enter a fresh scope, and exit destroys all retained values.
Only explicit core call sites use this adapter.
"""
from __future__ import annotations

import copy
import threading
from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import asdict
from pathlib import Path

from factorlab import protocol, time_utils
from factorlab.discovery import frame_identity as identity_module
from factorlab.discovery import known_library, rolling
from factorlab.discovery.evidence import file_hash
from factorlab.discovery.frame_identity import frame_identity
from factorlab.protocol import fingerprint

_CURRENT = ContextVar('v7_operation_inputs', default=None)


def _source_identity():
    # Read bytes on each lookup: unchanged mtime/size is never evidence.
    return {Path(m.__file__).name: file_hash(Path(m.__file__))
            for m in (rolling, known_library, identity_module, protocol, time_utils)}


def _size(data):
    return (data.frame.values.nbytes + data.target.nbytes + data.observed.nbytes
            + data.maturity.nbytes
            + int(data.frame.metadata.memory_usage(index=True, deep=True).sum()))


class OperationInputs:
    """Non-reentrant, thread-owned, non-persistent, capacity-limited scope.

    Internal retained buffers default to at most 16 MiB and two snapshots.
    Returned defensive copies and the caller's frames add transient memory;
    this cache is NOT a whole-process or cgroup resource limit.
    """

    def __init__(self, *, max_bytes=16 * 2**20, max_entries=2):
        if (type(max_bytes) is not int or not 0 <= max_bytes <= 32 * 2**20
                or type(max_entries) is not int or not 0 <= max_entries <= 4):
            raise ValueError('Bounded input cache capacity required')
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self._entries, self._bytes = OrderedDict(), 0
        self._token = self._owner = None
        self._closed = False
        self.stats = {'lookups': 0, 'builds': 0, 'hits': 0, 'evictions': 0,
                      'oversize': 0, 'peak_retained_bytes': 0}

    def __enter__(self):
        if self._closed or self._token is not None or _CURRENT.get() is not None:
            raise ValueError('Fresh non-nested operation scope required')
        self._owner = threading.get_ident()
        self._token = _CURRENT.set(self)
        return self

    def __exit__(self, *exc):
        if threading.get_ident() != self._owner:
            raise ValueError('Operation scope belongs to another thread')
        _CURRENT.reset(self._token)
        self._entries.clear()
        self._bytes = 0
        self._closed = True

    @property
    def retained_bytes(self):
        return self._bytes

    def build(self, bars, spec=None):
        if self._closed or _CURRENT.get() is not self or threading.get_ident() != self._owner:
            raise ValueError('Input reuse requires its active original operation')
        spec = rolling.RollingSpec() if spec is None else spec
        if type(spec) is not rolling.RollingSpec:
            raise TypeError('Actual immutable rolling specification required')
        # Snapshot caller-owned buffers before hashing or invoking the producer.
        # Supported object cells are immutable scalar values (enforced by hash).
        source = bars.copy(deep=True)
        source.attrs = copy.deepcopy(bars.attrs)
        key = fingerprint({'raw': frame_identity(source), 'attrs': source.attrs,
                           'dtypes': [str(t) for t in source.dtypes],
                           'spec': asdict(spec), 'schema': spec.digest,
                           'producer': _source_identity()})
        self.stats['lookups'] += 1
        if key in self._entries:
            self.stats['hits'] += 1
            data, _ = self._entries[key]
            self._entries.move_to_end(key)
            return copy.deepcopy(data)
        self.stats['builds'] += 1
        data = rolling.build_rolling_data(source, spec)
        size = _size(data)
        if not self.max_entries or size > self.max_bytes:
            self.stats['oversize'] += 1
            return data
        while self._entries and (len(self._entries) >= self.max_entries or self._bytes + size > self.max_bytes):
            _, (_, prior) = self._entries.popitem(last=False)
            self._bytes -= prior
            self.stats['evictions'] += 1
        self._entries[key] = (copy.deepcopy(data), size)
        self._bytes += size
        self.stats['peak_retained_bytes'] = max(self.stats['peak_retained_bytes'], self._bytes)
        return data


def build_rolling_data(bars, spec=None):
    """Explicit call-site adapter; absent scope retains original behavior."""
    scope = _CURRENT.get()
    return rolling.build_rolling_data(bars, spec) if scope is None else scope.build(bars, spec)
