"""Zero-fit source archive semantics for the isolated Linux repair variant."""
import json
import os

import pytest

from factorlab.discovery.evidence import file_hash, immutable_json
from factorlab.discovery.evolution import _verify_source_archive
from factorlab.protocol import fingerprint


def archive(tmp_path):
    folder = tmp_path / 'source'
    folder.mkdir()
    names = ['factorlab/features.py', 'factorlab/data.py', 'factorlab/protocol.py',
             'pyproject.toml', 'uv.lock']
    for name in names:
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('original:' + name, encoding='utf-8')
    files = {name: file_hash(folder / name) for name in names}
    immutable_json(folder / 'manifest.json', files)
    # This suite runs only in the actual Linux engineering envelope.
    assert os.name == 'posix'
    assert (folder / 'manifest.json.lock').is_file()
    identity = {'source_files': files, 'source_archive': {'path': 'source', 'digest': fingerprint(files)}}
    return folder, identity


def test_generated_manifest_lock_is_not_source_and_uv_lock_remains_bound(tmp_path):
    folder, identity = archive(tmp_path)
    lock = folder / 'manifest.json.lock'
    before = lock.read_bytes()
    result = _verify_source_archive(tmp_path, identity)
    assert result['archive_verified'] is True and result['files'] == 5
    assert lock.read_bytes() == before
    (folder / 'uv.lock').write_text('changed dependency', encoding='utf-8')
    with pytest.raises(ValueError, match='source archive integrity'):
        _verify_source_archive(tmp_path, identity)


@pytest.mark.parametrize('damage', ['corrupt_file', 'missing_uv_lock', 'extra_file',
                                  'other_lock', 'nested_manifest_lock', 'manifest', 'identity'])
def test_integrity_failures_remain_rejected(tmp_path, damage):
    folder, identity = archive(tmp_path)
    if damage == 'corrupt_file':
        (folder / 'factorlab/features.py').write_text('corrupt', encoding='utf-8')
    elif damage == 'missing_uv_lock':
        (folder / 'uv.lock').rename(tmp_path / 'retained-uv.lock')
    elif damage in {'extra_file', 'other_lock', 'nested_manifest_lock'}:
        name = {'extra_file': 'extra.py', 'other_lock': 'another.lock',
                'nested_manifest_lock': 'factorlab/manifest.json.lock'}[damage]
        (folder / name).write_text('unexpected', encoding='utf-8')
    elif damage == 'manifest':
        (folder / 'manifest.json').write_text('{}', encoding='utf-8')
    else:
        identity['source_archive']['digest'] = '0' * 64
    with pytest.raises(ValueError, match='source archive'):
        _verify_source_archive(tmp_path, identity)


@pytest.mark.parametrize('destination', ['internal', 'external'])
def test_manifest_lock_symlink_is_never_exempt(tmp_path, destination):
    folder, identity = archive(tmp_path)
    lock = folder / 'manifest.json.lock'
    lock.rename(tmp_path / 'retained-manifest-lock')
    target = folder / 'uv.lock' if destination == 'internal' else tmp_path / 'external'
    if destination == 'external':
        target.write_text('foreign', encoding='utf-8')
    lock.symlink_to(target)
    with pytest.raises(ValueError, match='source archive'):
        _verify_source_archive(tmp_path, identity)
