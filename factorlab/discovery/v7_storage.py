"""Atomic, immutable JSON publication inside the project workspace."""
from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

from filelock import FileLock

from factorlab.protocol import project_path


def _native_io(path):
    """Windows atomic temp names may exceed MAX_PATH before the target does."""
    if os.name != 'nt':
        return path
    value = str(path)
    if value.startswith('\\\\?\\'):
        return path
    return Path('\\\\?\\UNC\\' + value[2:] if value.startswith('\\\\') else '\\\\?\\' + value)


def immutable_record(path, value):
    target = project_path(path)
    target_io = _native_io(target)
    target_io.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)
    with FileLock(str(target_io) + '.lock'):
        if target_io.exists():
            if json.loads(target_io.read_text('utf-8')) != json.loads(payload):
                raise ValueError('Immutable V7 record differs from its frozen content')
            return target
        temporary = target.with_name(f'.{target.name}.{uuid4().hex}.tmp')
        temporary_io = _native_io(temporary)
        try:
            with temporary_io.open('x', encoding='utf-8', newline='\n') as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary_io, target_io)
        finally:
            temporary_io.unlink(missing_ok=True)
    return target
