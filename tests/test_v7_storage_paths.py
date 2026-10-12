"""Regression for the actual MAX_PATH failure in immutable receipt temp I/O."""
import os

import pytest

from factorlab.discovery.v7_storage import immutable_record


@pytest.mark.skipif(os.name != 'nt', reason='Windows MAX_PATH regression')
def test_long_atomic_temp_still_publishes_once_and_preserves_original(tmp_path, monkeypatch):
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    # Target and its lock fit the legacy limit; its random atomic temp did not.
    base_length = len(str(tmp_path))
    parent = tmp_path / ('d' * (230 - base_length - 1))
    target = parent / 'record.json'
    assert len(str(target)) < 260 < len(str(target.with_name('.record.json.' + 'x' * 32 + '.tmp')))
    immutable_record(target, {'original': 1})
    assert immutable_record(target, {'original': 1}) == target
    with pytest.raises(ValueError, match='differs'):
        immutable_record(target, {'original': 2})
    assert '"original": 1' in target.read_text('utf-8')
    assert not list(parent.glob('.*.tmp'))



@pytest.mark.parametrize('length', [254, 255, 275, 290])
def test_record_and_lock_length_boundary_preserves_immutable_publication(tmp_path, monkeypatch, length):
    import json
    from factorlab.discovery.v7_storage import _native_io
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    filename = 'preparation.json'
    parent = tmp_path / ('p' * (length - len(str(tmp_path)) - len(filename) - 2))
    target = parent / filename
    assert len(str(target)) == length
    value = {'original': length}
    assert immutable_record(target, value) == target
    assert immutable_record(target, value) == target
    with pytest.raises(ValueError, match='differs'):
        immutable_record(target, {'replacement': length})
    assert json.loads(_native_io(target).read_text('utf-8')) == value
    assert not list(_native_io(parent).glob('.*.tmp'))


def test_concurrent_long_record_writers_keep_one_original(tmp_path, monkeypatch):
    import json
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from factorlab.discovery.v7_storage import _native_io
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    filename = 'shared.json'
    parent = tmp_path / ('q' * (275 - len(str(tmp_path)) - len(filename) - 2))
    target = parent / filename
    gate = Barrier(2)
    def write(value):
        gate.wait(timeout=5)
        try:
            immutable_record(target, {'writer': value})
            return 'published', value
        except ValueError as error:
            assert 'differs' in str(error)
            return 'refused', value
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(write, [1, 2]))
    assert sorted(result[0] for result in results) == ['published', 'refused']
    winner = next(value for status, value in results if status == 'published')
    assert json.loads(_native_io(target).read_text('utf-8')) == {'writer': winner}
    assert not list(_native_io(parent).glob('.*.tmp'))


def test_storage_long_path_still_rejects_outside_project_root(tmp_path, monkeypatch):
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    outside = tmp_path.parent / ('outside-' + tmp_path.name) / 'record.json'
    with pytest.raises(ValueError, match='remain within'):
        immutable_record(outside, {'must_not_publish': True})
    assert not outside.exists() and not outside.parent.exists()
