"""Archived source/dependency identities and honest historical legacy inspection."""
import json
from pathlib import Path

import pytest
from test_discovery_evolution import fixture, run

from factorlab.discovery import evolution
from factorlab.discovery.evolution import EvolutionPolicy, EvolutionStore, inspect_evolution
from factorlab.protocol import fingerprint


def initialized(tmp_path):
    data, spec, model = fixture()
    policy = EvolutionPolicy(mode='fixed')
    result = run(tmp_path, 'source', data, spec, model, policy=policy, max_ticks=2)
    return data, spec, policy, result


def test_new_run_archives_complete_program_and_locked_dependencies(tmp_path):
    _, _, _, result = initialized(tmp_path)
    root = tmp_path / 'source'
    manifest = json.loads((root / 'manifest.json').read_text('utf-8'))['manifest']
    files = manifest['source_files']
    assert {'factorlab/features.py', 'factorlab/data.py', 'factorlab/protocol.py',
            'factorlab/discovery/evolution.py', 'pyproject.toml', 'uv.lock'} <= set(files)
    assert files == evolution._current_source_files()
    assert manifest['source_archive'] == {'path': 'source', 'digest': fingerprint(files)}
    archived = json.loads((root / 'source' / 'manifest.json').read_text('utf-8'))
    assert archived == files
    assert result['source_evidence']['archive_verified'] is True
    assert result['source_evidence']['runtime_binary_snapshot_claimed'] is False
    assert inspect_evolution(root)['source_evidence'] == result['source_evidence']


@pytest.mark.parametrize('path', ['factorlab/features.py', 'factorlab/data.py',
                                 'factorlab/protocol.py', 'pyproject.toml', 'uv.lock'])
def test_resume_rejects_changes_to_full_current_source_but_history_remains_inspectable(
        tmp_path, monkeypatch, path):
    data, spec, policy, before = initialized(tmp_path)
    changed = evolution._current_source_files()
    changed[path] = '0' * 64
    monkeypatch.setattr(evolution, '_current_source_files', lambda: changed)
    # Inspection proves archived bytes. It need not equal today's implementation.
    assert inspect_evolution(tmp_path / 'source')['event_head'] == before['event_head']
    with pytest.raises(ValueError, match='identity changed'):
        run(tmp_path, 'source', data, spec, None, policy=policy, resume=True)


@pytest.mark.parametrize('change', ['corrupt_file', 'missing_file', 'extra_file', 'manifest'])
def test_source_archive_corruption_or_incomplete_coverage_is_rejected(tmp_path, change):
    data, spec, policy, _ = initialized(tmp_path)
    root = tmp_path / 'source' / 'source'
    if change == 'corrupt_file':
        (root / 'factorlab' / 'features.py').write_text('changed archival bytes', encoding='utf-8')
    elif change == 'missing_file':
        # Moving one explicitly named test artifact keeps this reversible.
        (root / 'uv.lock').rename(tmp_path / 'saved-uv.lock')
    elif change == 'extra_file':
        (root / 'unexpected.py').write_text('extra file', encoding='utf-8')
    else:
        (root / 'manifest.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='source archive'):
        inspect_evolution(tmp_path / 'source')
    with pytest.raises(ValueError, match='source archive'):
        run(tmp_path, 'source', data, spec, None, policy=policy, resume=True)


def test_source_change_during_initial_snapshot_cannot_commit_run(tmp_path, monkeypatch):
    data, spec, model = fixture()
    real_snapshot = evolution.snapshot_source
    def changed_snapshot(directory):
        result = real_snapshot(directory)
        result['files']['factorlab/features.py'] = '0' * 64
        return result
    monkeypatch.setattr(evolution, 'snapshot_source', changed_snapshot)
    with pytest.raises(ValueError, match='changed during initial snapshot'):
        run(tmp_path, 'race', data, spec, model, policy=EvolutionPolicy(mode='fixed'), max_ticks=1)
    store = EvolutionStore(tmp_path / 'race')
    try:
        assert store.state() is None
        assert not (tmp_path / 'race' / 'manifest.json').exists()
    finally:
        store.close()


def make_legacy(root):
    """A test-only pre-hardening fixture; production legacy runs are never relabeled."""
    store = EvolutionStore(root)
    try:
        state = store.state()
        identity = state['identity']
        identity.pop('source_archive')
        identity['source_files'] = {Path(key).name: value for key, value in identity['source_files'].items()
                                    if key.startswith('factorlab/discovery/')}
        manifest = {'manifest': {**identity, 'evidence_exposure': {}},
                    'sha256': fingerprint({**identity, 'evidence_exposure': {}})}
        (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        with store.db:
            store.save_state(state)
    finally:
        store.close()


def test_legacy_hash_only_inspection_is_explicit_and_does_not_invent_archive(tmp_path):
    data, spec, policy, _ = initialized(tmp_path)
    root = tmp_path / 'source'
    make_legacy(root)
    result = inspect_evolution(root)
    assert result['source_evidence']['mode'] == 'legacy_discovery_hash_only'
    assert result['source_evidence']['archive_verified'] is False
    assert result['can_resume'] is False
    with pytest.raises(ValueError, match='Legacy hash-only'):
        run(tmp_path, 'source', data, spec, None, policy=policy, resume=True)
    manifest = json.loads((root / 'manifest.json').read_text('utf-8'))
    assert 'source_archive' not in manifest['manifest']


def test_source_archive_identity_cannot_be_rebound_with_only_manifest_edits(tmp_path):
    initialized(tmp_path)
    path = tmp_path / 'source' / 'manifest.json'
    record = json.loads(path.read_text('utf-8'))
    record['manifest']['source_archive']['digest'] = '0' * 64
    record['sha256'] = fingerprint(record['manifest'])
    path.write_text(json.dumps(record), encoding='utf-8')
    with pytest.raises(ValueError, match='manifest identity'):
        inspect_evolution(tmp_path / 'source')
