"""Registry + real authority admission tests with explicit artificial kernel/ticket seam."""
import json

import pytest
from test_v7_neural_forward import fitted as fitted  # noqa: PLC0414
from test_v7_neural_forward_admission import evidence as evidence  # noqa: PLC0414

from factorlab.discovery.model_registry import ModelRegistry
from factorlab.discovery.work_lease import FrozenWorkLease


def test_neutral_or_artificial_models_cannot_register_without_external_certificate(evidence, tmp_path):
    registry = ModelRegistry(tmp_path / 'models')
    for model in (evidence.model, evidence.artificial):
        with pytest.raises(ValueError, match='certificate and job'):
            registry.save(model, evidence.context['pipeline'])
    assert not list(registry.directory.iterdir())


def test_registry_requires_matching_job_and_pipeline(evidence, tmp_path):
    registry = ModelRegistry(tmp_path / 'models')
    with pytest.raises(ValueError, match='job'):
        registry.save(evidence.model, evidence.context['pipeline'],
                      certificate=evidence.cert, expected_job='another-job')
    with pytest.raises(ValueError, match='pipeline'):
        registry.save(evidence.model, {}, certificate=evidence.cert, expected_job='job')


def test_first_publish_requires_live_original_lease(evidence, tmp_path, monkeypatch):
    registry = ModelRegistry(tmp_path / 'models')
    def expired(*args, **kwargs):
        raise ValueError('original-lease-expired')
    monkeypatch.setattr(FrozenWorkLease, 'assert_parent_alive', expired)
    with pytest.raises(ValueError, match='original-lease-expired'):
        registry.save(evidence.model, evidence.context['pipeline'],
                      certificate=evidence.cert, expected_job='job')
    assert not list(registry.directory.glob('*/manifest.json'))


def test_historical_restore_revalidates_admission_without_renewing_lease(evidence, tmp_path, monkeypatch):
    registry = ModelRegistry(tmp_path / 'models')
    identifier = registry.save(evidence.model, evidence.context['pipeline'],
                               certificate=evidence.cert, expected_job='job')
    monkeypatch.setattr(FrozenWorkLease, 'assert_parent_alive',
                        lambda *a, **k: pytest.fail('Historical restore cannot renew the old lease'))
    loaded, identity = registry.load(identifier, expected_pipeline=evidence.context['pipeline'])
    assert loaded.digest == evidence.model.digest and identity['version'] == 2
    assert registry.save(evidence.model, evidence.context['pipeline'],
                         certificate=evidence.cert, expected_job='job') == identifier
    (evidence.launch / 'failure.json').write_text('{}')
    with pytest.raises(ValueError, match='Failed or late'):
        registry.load(identifier)


def test_post_publication_expiry_permanently_rejects_partial_registration(evidence, tmp_path, monkeypatch):
    registry = ModelRegistry(tmp_path / 'models')
    original = FrozenWorkLease.assert_parent_alive
    calls = []
    def expires(self, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise ValueError('expired-during-manifest-write')
        return original(self, **kwargs)
    monkeypatch.setattr(FrozenWorkLease, 'assert_parent_alive', expires)
    with pytest.raises(ValueError, match='expired-during-manifest-write'):
        registry.save(evidence.model, evidence.context['pipeline'],
                      certificate=evidence.cert, expected_job='job')
    folder = next(registry.directory.glob('*/registration_failed.json')).parent
    assert (folder / 'manifest.json').exists()
    with pytest.raises(ValueError, match='Failed neural'):
        registry.load(folder.name)


def test_registry_checksum_cannot_replace_external_admission(evidence, tmp_path):
    from factorlab.protocol import fingerprint
    registry = ModelRegistry(tmp_path / 'models')
    identifier = registry.save(evidence.model, evidence.context['pipeline'],
                               certificate=evidence.cert, expected_job='job')
    path = registry.directory / identifier / 'manifest.json'
    value = json.loads(path.read_text())
    value['identity'].pop('admission')
    other = fingerprint(value['identity'])
    other_folder = registry.directory / other
    other_folder.mkdir()
    (other_folder / 'model.json').write_bytes((path.parent / 'model.json').read_bytes())
    value['id'] = other
    (other_folder / 'manifest.json').write_text(json.dumps(value))
    with pytest.raises(ValueError, match='external admission'):
        registry.load(other)
