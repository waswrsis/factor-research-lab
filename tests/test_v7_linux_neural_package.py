"""Hash-locked CPU dependency closure and package inclusion; no Linux execution."""
import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from test_v7_linux_package import recipe


def test_neural_cpu_lock_has_complete_active_dependencies_and_no_cuda():
    lock = recipe.neural_runtime_lock(Path.cwd())
    pins = {item['name']: item['version'] for item in lock['packages']}
    assert pins['torch'] == '2.11.0+cpu'
    assert len(pins) == 28
    for item in lock['packages']:
        for text in item['active_requires_dist']:
            dependency = Requirement(text)
            name = canonicalize_name(dependency.name)
            assert name in pins
            assert dependency.specifier.contains(pins[name])
    assert {'config/v7-linux-neural-requirements.txt', 'config/v7-linux-neural-lock.json'} <= set(recipe.FIXED)


@pytest.mark.parametrize('damage', ['requirements', 'baseline', 'target', 'cuda', 'torch_url', 'size'])
def test_neural_install_contract_rejects_changed_dependency_or_target(tmp_path, damage):
    config = tmp_path / 'config'
    config.mkdir()
    for name in ('v7-linux-neural-requirements.txt', 'v7-linux-neural-lock.json', 'v7-linux-requirements.txt'):
        shutil.copyfile(Path('config') / name, config / name)
    lock = copy.deepcopy(recipe.neural_runtime_lock(tmp_path))
    if damage == 'requirements':
        with (config / 'v7-linux-neural-requirements.txt').open('ab') as stream:
            stream.write(b'nvidia-cuda-runtime-cu12==12.8\n')
    elif damage == 'baseline':
        with (config / 'v7-linux-requirements.txt').open('ab') as stream:
            stream.write(b'unregistered-package==1\n')
    elif damage == 'target':
        lock['target']['python'] = '3.13'
    elif damage == 'cuda':
        lock['packages'][0]['name'] = 'nvidia-cuda-runtime-cu12'
    elif damage == 'torch_url':
        torch = next(item for item in lock['packages'] if item['name'] == 'torch')
        torch['url'] = 'https://files.pythonhosted.org/packages/unapproved/' + torch['wheel']
    else:
        lock['total_wheel_bytes'] += 1
    (config / 'v7-linux-neural-lock.json').write_text(json.dumps(lock))
    with pytest.raises(ValueError):
        recipe.neural_runtime_lock(tmp_path)


def test_downloaded_linux_wheels_match_lock_bytes_hash_and_cpu_torch_metadata():
    import email
    import zipfile
    lock = recipe.neural_runtime_lock(Path.cwd())
    wheelhouse = Path('.cache/v7-linux-cpu-wheels')
    if not all((wheelhouse / item['wheel']).is_file() for item in lock['packages']):
        pytest.skip('Optional downloaded Linux wheel artifact audit requires the D-drive wheel cache')
    for item in lock['packages']:
        path = wheelhouse / item['wheel']
        with path.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == item['sha256']
        assert path.stat().st_size == item['bytes']
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist()
                     if name.count('/') == 1 and name.endswith('.dist-info/METADATA')]
            assert len(names) == 1
            metadata = email.message_from_bytes(archive.read(names[0]))
            assert canonicalize_name(metadata['Name']) == item['name']
            assert metadata['Version'] == item['version']
            if item['name'] == 'torch':
                assert not any(text.startswith(('nvidia', 'triton'))
                               for text in metadata.get_all('Requires-Dist', []))
