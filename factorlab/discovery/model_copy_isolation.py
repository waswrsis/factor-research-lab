"""Keep an intact active A0 available after one unactivated registry copy fails.

This grants no model admission or research permission. Original paid launch
artifacts and their full admission remain mandatory. The damaged copy is never
read as a predictor, repaired, republished or removed from the audit denominator.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from factorlab.protocol import fingerprint, project_path
from .evidence import file_hash
from .v7_storage import immutable_record

VERSION = 'unactivated-registry-copy-isolation-v2'
EVENT = 'model_registry_copy_isolated'


def _read(path):
    return json.loads(Path(path).read_text('utf-8'))


def _artifact(path):
    if path.resolve() != path or path.is_symlink():
        raise ValueError('Registry artifact redirection is not an isolated copy fault')
    if not path.exists():
        return {'path': str(path), 'sha256': None}
    if not path.is_file():
        raise ValueError('Registry artifact is not a regular file')
    return {'path': str(path), 'sha256': file_hash(path)}


def _envelope(lifecycle, job, row):
    """Full original admitted launch validation; only registry model bytes differ."""
    identifier, pipeline = row[4], json.loads(row[2])
    folder = lifecycle.registry.directory / identifier
    manifest = _read(folder / 'manifest.json')
    identity = manifest['identity']
    if (set(manifest) != {'identity', 'id', 'artifact_sha256'}
            or manifest['id'] != identifier or fingerprint(identity) != identifier
            or identity['pipeline'] != pipeline
            or (folder / 'registration_failed.json').exists()):
        raise ValueError('Original registry envelope changed or registration failed')
    actual = _artifact(folder / 'model.json')
    if actual['sha256'] == manifest['artifact_sha256']:
        raise ValueError('No damaged registry copy to isolate; no automatic recovery')
    kind = identity['kind']
    versions = {'v7_known_linear': 3, 'v7_joint_pool': 4, 'v7_neural_forward': 2}
    if identity['version'] == 1 and not lifecycle._work_authority_bound():
        # Offline historical Lifecycle components have no production wallet.
        # A ForwardRuntime can never enter this branch, including pre-wallet.
        if kind not in {'v7_log_ridge', 'v7_event_logit', 'v7_factor_pipeline', 'v7_joint_pool'}:
            raise ValueError('Unadmitted model kind cannot isolate a production copy')
        proof = {'scope': 'standalone_legacy_not_forward_admission'}
    else:
        if (kind not in versions or identity['version'] != versions[kind]
                or set(identity) != {'version', 'kind', 'model_digest', 'schema', 'task_support',
                                     'pipeline', 'admission', 'job'} or identity['job'] != job):
            raise ValueError('Original paid registry admission required')
        certificate = identity['admission']
        ticket_path = project_path(certificate['ticket']['path'])
        original_path = ticket_path.parent / 'launch' / 'model.json'
        # A second copy is not an authority: the canonical verifier below
        # binds this fixed path, ticket, wallet, original source and completion.
        if original_path == folder / 'model.json':
            raise ValueError('Registry copy cannot authenticate itself')
        original = lifecycle.registry._loader(kind)(original_path)
        if (original.digest != identity['model_digest'] or original.schema != identity['schema']
                or list(original.task_support) != identity['task_support']
                or file_hash(original_path) != manifest['artifact_sha256']):
            raise ValueError('Original launch payload differs from registered bytes')
        if kind == 'v7_known_linear':
            from .known_linear_forward import verify_known_admission as verifier
        elif kind == 'v7_joint_pool':
            from .pool_forward import verify_pool_admission as verifier
        else:
            from .neural_forward_admission import verify_forward_admission as verifier
        binding = verifier(original, certificate, expected_pipeline=pipeline, expected_job=job)
        from .work_lease import FrozenWorkLease
        from .neural_forward_admission import _observation, _file_deadline
        lease = FrozenWorkLease.from_dict(binding['reservation']['lease'])
        started_path, completed_path = folder / 'registration_started.json', folder / 'registration_completed.json'
        started, completed = _read(started_path), _read(completed_path)
        expected = {'identity': identifier, 'lease_hash': lease.digest,
                    'certificate_digest': fingerprint(certificate)}
        if (started != expected or set(completed) != {*expected, 'artifact_sha256', 'manifest_sha256', 'observation'}
                or {key: completed[key] for key in expected} != expected
                or completed['artifact_sha256'] != manifest['artifact_sha256']
                or completed['manifest_sha256'] != file_hash(folder / 'manifest.json')):
            raise ValueError('Original registry completion envelope changed')
        _observation(completed['observation'], lease, previous=binding['receipt']['final_io_observation'])
        _file_deadline([folder / 'manifest.json', started_path, completed_path], lease)
        lifecycle.registry._verify_publication(identifier, started, completed)
        proof = {'scope': 'original_paid_launch_verified', 'ticket': certificate['ticket'],
                 'certificate_digest': fingerprint(certificate),
                 'original_model': {'path': str(original_path), 'sha256': file_hash(original_path)},
                 'registration_started_sha256': file_hash(started_path),
                 'registration_completed_sha256': file_hash(completed_path)}
    return {'job': job, 'model': identifier, 'stream': row[1],
            'manifest_sha256': file_hash(folder / 'manifest.json'),
            'expected_artifact_sha256': manifest['artifact_sha256'],
            'damaged_copy': actual, 'original_evidence': proof}


def candidate(lifecycle, job, row, *, prior=None):
    """A verified but NEVER activated candidate, with the intact operational A0."""
    if row[3] != 'verified' or not row[4] or row[6] is not None or row[7] not in (None, 'integrity_failure'):
        raise ValueError('Only an unactivated completed candidate can isolate a copy')
    # Degraded tick serves only A0. A first prospective V/C/policy stream has
    # no same-stream incumbent; another stream's incumbent is not a fallback.
    # Cutoffs in different streams can coincide and do not order availability.
    active = lifecycle.db.execute('SELECT j.id,j.model,j.pipeline,j.activation,a.generation,j.stream,j.status '
        'FROM active a JOIN jobs j ON a.job=j.id WHERE a.stream=?', ('A0',)).fetchone()
    if (active is None or active[0] == job or active[3] is None
            or active[5] != 'A0' or active[6] != 'activated'):
        raise ValueError('An intact existing operational A0 model is required')
    lifecycle._load_job_model(active[0], active[1], json.loads(active[2]))
    value = _envelope(lifecycle, job, row)
    value['retained_active'] = {'stream': 'A0', 'job': active[0], 'model': active[1],
                                'activation': active[3], 'generation': active[4]}
    if prior is not None and value != prior['evidence']:
        raise ValueError('Isolated fault/original evidence or retained active changed')
    return value


def read_anchors(lifecycle, jobs, operations):
    events = [op for op in operations if op['kind'] == EVENT]
    found = {}
    for event in events:
        detail = event['detail']
        if set(detail) != {'job', 'proof_hash'} or detail['job'] in found or detail['job'] not in jobs:
            raise ValueError('Isolation event identity changed or duplicated')
        job, row = detail['job'], jobs[detail['job']]
        path = lifecycle.directory / 'model_isolations' / f'{job}.json'
        mirror = lifecycle.registry.directory / row[4] / 'serving_isolation.json'
        value = _read(path)
        if (value != _read(mirror) or set(value) != {'version', 'event_head', 'observed_at', 'evidence'}
                or value['version'] != VERSION or fingerprint(value) != detail['proof_hash']
                or value['event_head'] != event['previous'] or row[7] != 'integrity_failure'
                or pd.Timestamp(value['observed_at']).tzinfo is None):
            raise ValueError('Original copy-isolation mirrors/chain/failure changed')
        candidate(lifecycle, job, row, prior=value)
        found[job] = value
    expected = {lifecycle.directory / 'model_isolations' / f'{job}.json' for job in found}
    actual = set((lifecycle.directory / 'model_isolations').glob('*.json'))
    mirrors = {lifecycle.registry.directory / jobs[job][4] / 'serving_isolation.json' for job in found}
    if actual != expected or set(lifecycle.registry.directory.glob('*/serving_isolation.json')) != mirrors:
        raise ValueError('Orphan or missing isolation anchor; no reset or automatic repair')
    return found


def record(lifecycle, evidence):
    job = evidence['job']
    previous = lifecycle.db.execute('SELECT hash FROM operations ORDER BY seq DESC LIMIT 1').fetchone()[0]
    value = {'version': VERSION, 'event_head': previous,
             'observed_at': pd.Timestamp.now(tz='UTC').isoformat(), 'evidence': evidence}
    # No partial write is recoverable as permission: read_anchors requires
    # both immutable files and the original chain's failure/event pair.
    immutable_record(lifecycle.directory / 'model_isolations' / f'{job}.json', value)
    immutable_record(lifecycle.registry.directory / evidence['model'] / 'serving_isolation.json', value)
    with lifecycle.db:
        lifecycle.db.execute('BEGIN IMMEDIATE')
        actual = lifecycle.db.execute('SELECT hash FROM operations ORDER BY seq DESC LIMIT 1').fetchone()[0]
        if actual != previous:
            raise ValueError('Lifecycle changed during original fault observation')
        lifecycle._event(EVENT, {'job': job, 'proof_hash': fingerprint(value)})
        lifecycle.db.execute('UPDATE jobs SET failure=? WHERE id=?', ('integrity_failure', job))
        lifecycle._event('fit_failed_old_model_retained', {'job': job, 'code': 'integrity_failure'})
    return value


def has_isolations(lifecycle):
    # This is restriction-only. It is never a substitute for read_anchors.
    return any(json.loads(row[0])['kind'] == EVENT
               for row in lifecycle.db.execute('SELECT record FROM operations ORDER BY seq'))


def needs_serving_check(lifecycle):
    """Cheap constructor trigger, never permission or a replacement verifier."""
    if (has_isolations(lifecycle)
            or any((lifecycle.directory / 'model_isolations').glob('*.json'))
            or any(lifecycle.registry.directory.glob('*/serving_isolation.json'))):
        return True
    for (identifier,) in lifecycle.db.execute(
            "SELECT model FROM jobs WHERE status='verified' AND model IS NOT NULL"):
        folder = lifecycle.registry.directory / identifier
        manifest = _read(folder / 'manifest.json')
        if _artifact(folder / 'model.json')['sha256'] != manifest['artifact_sha256']:
            return True
    return False
