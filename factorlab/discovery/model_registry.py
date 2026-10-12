"""V7 immutable convex artifacts and logical forecast identity.

Model digest is an immutable binding, never a way to obtain a second forecast
for the same origin. Old Torch artifacts keep their existing legacy loader.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing

import pandas as pd
from filelock import FileLock

from factorlab.protocol import fingerprint, project_path

from .evidence import file_hash
from .v7_storage import immutable_record as immutable_json


class ModelRegistry:
    def __init__(self, directory):
        self.directory = project_path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _publication_db(self):
        """A separate durable witness: files alone never finish registration.

        SQL COMMIT is the publication linearization point. The original lease
        is checked after all artifact writes while completion is uncommitted.
        This is local transaction evidence, not nanosecond real-time assurance
        or authentication against replacement of every store by an operator.
        """
        path = self.directory / 'neural_publications.sqlite'
        anchor = self.directory / 'neural_publications_identity.json'
        expected = {'version': 'v7-neural-publications-v1', 'directory': str(self.directory)}
        with FileLock(str(path) + '.init.lock'):
            if not path.exists():
                if anchor.exists() or any(self.directory.glob('*/registration_started.json')):
                    raise ValueError('Neural publication witness missing; cannot reset')
                immutable_json(anchor, expected)
                with closing(sqlite3.connect(path)) as db:
                    db.execute('PRAGMA journal_mode=WAL')
                    db.execute('PRAGMA synchronous=FULL')
                    db.execute('CREATE TABLE publications (id TEXT PRIMARY KEY, status TEXT, binding TEXT, completion TEXT)')
                    db.commit()
            if not anchor.is_file() or json.loads(anchor.read_text('utf-8')) != expected:
                raise ValueError('Neural publication witness identity changed')
        db = sqlite3.connect(path.as_uri() + '?mode=rw', uri=True, timeout=30)
        db.execute('PRAGMA synchronous=FULL')
        return db

    def _verify_publication(self, identifier, started, completed):
        path = self.directory / 'neural_publications.sqlite'
        anchor = self.directory / 'neural_publications_identity.json'
        expected = {'version': 'v7-neural-publications-v1', 'directory': str(self.directory)}
        if (not path.is_file() or not anchor.is_file()
                or json.loads(anchor.read_text('utf-8')) != expected):
            raise ValueError('Missing neural publication witness')
        with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as db:
            row = db.execute('SELECT status,binding,completion FROM publications WHERE id=?', (identifier,)).fetchone()
        if row != ('completed', json.dumps(started, sort_keys=True), fingerprint(completed)):
            raise ValueError('Neural registration lacks committed completion evidence')

    def save(self, model, pipeline: dict, *, certificate=None, expected_job=None):
        kind = model.kind
        if kind == 'v7_neural_forward':
            return self.save_neural(model, pipeline, certificate=certificate, expected_job=expected_job)
        if kind == 'v7_known_linear' and (certificate is not None or expected_job is not None):
            from .known_linear_forward import verify_known_admission
            return self._save_admitted(model, pipeline, certificate=certificate, expected_job=expected_job,
                                       verifier=verify_known_admission, version=3)
        if kind == 'v7_joint_pool' and (certificate is not None or expected_job is not None):
            from .pool_forward import verify_pool_admission
            return self._save_admitted(model, pipeline, certificate=certificate, expected_job=expected_job,
                                       verifier=verify_pool_admission, version=4)
        if pipeline.get('kind') == 'v7_neural_forward':
            raise ValueError('A neural pipeline requires a complete neural model')
        if certificate is not None or expected_job is not None:
            raise ValueError('Neural admission arguments cannot authorize another model kind')
        if ((pipeline.get('kind') == 'v7_known_linear' or kind == 'v7_known_linear')
                and (kind != 'v7_known_linear' or pipeline != self._linear_pipeline(model))):
            raise ValueError('Fixed linear pipeline requires its complete matching model and recipe')
        if kind not in {'v7_log_ridge', 'v7_event_logit', 'v7_factor_pipeline', 'v7_joint_pool', 'v7_known_linear'}:
            raise ValueError('Unregistered V7 model kind')
        identity = {'version': 1, 'kind': kind, 'model_digest': model.digest,
                    'schema': model.schema, 'task_support': list(model.task_support),
                    'pipeline': pipeline}
        identifier = fingerprint(identity)
        folder = self.directory / identifier
        with FileLock(str(folder) + '.lock'):
            if (folder / 'registration_failed.json').exists() or (folder / 'serving_isolation.json').exists():
                raise ValueError('Failed neural registration cannot be restored')
            folder.mkdir(parents=True, exist_ok=True)
            artifact = folder / 'model.json'
            if not artifact.exists():
                model.save(artifact)
            else:
                restored = self._loader(kind)(artifact)
                if restored.digest != model.digest:
                    raise ValueError('Immutable model artifact conflict')
            immutable_json(folder / 'manifest.json', {
                'identity': identity, 'id': identifier, 'artifact_sha256': file_hash(artifact)})
        return identifier

    def save_neural(self, model, pipeline, *, certificate, expected_job):
        """Publish verified neutral payloads; a receipt flag is never admission.

        The first registry write remains within the original launcher lease.
        An already-published identical registration can be loaded historically
        without fitting, charging again, or reviving its old parent process.
        """
        from .neural_forward_admission import verify_forward_admission
        return self._save_admitted(model, pipeline, certificate=certificate, expected_job=expected_job,
                                   verifier=verify_forward_admission, version=2)

    def _save_admitted(self, model, pipeline, *, certificate, expected_job, verifier, version):
        """Share publication mechanics; each learner supplies its strict verifier."""
        from .work_lease import FrozenWorkLease
        if certificate is None or not isinstance(expected_job, str) or not expected_job:
            raise ValueError('Complete neural registration requires a fixed certificate and job')
        binding = verifier(model, certificate,
            expected_pipeline=pipeline, expected_job=expected_job)
        identity = {'version': version, 'kind': model.kind, 'model_digest': model.digest,
                    'schema': model.schema, 'task_support': list(model.task_support),
                    'pipeline': pipeline, 'admission': certificate, 'job': expected_job}
        identifier = fingerprint(identity)
        folder = self.directory / identifier
        with FileLock(str(folder) + '.lock'):
            if (folder / 'registration_failed.json').exists() or (folder / 'serving_isolation.json').exists():
                raise ValueError('Failed neural registration cannot be restored')
            if (folder / 'manifest.json').exists():
                restored, actual = self.load(identifier, expected_pipeline=pipeline)
                if restored.digest != model.digest or actual != identity:
                    raise ValueError('Immutable neural registration conflict')
                return identifier
            lease = FrozenWorkLease.from_dict(binding['reservation']['lease'])
            lease.assert_parent_alive()
            folder.mkdir(parents=True, exist_ok=True)
            artifact = folder / 'model.json'
            started = {
                'identity': identifier, 'lease_hash': lease.digest,
                'certificate_digest': fingerprint(certificate)}
            # The pending state is committed before publishing any model file.
            # All later files may exist after a crash; readers still reject them
            # until the separate transaction commits after the last lease check.
            db = self._publication_db()
            try:
                with db:
                    db.execute('BEGIN IMMEDIATE')
                    existing = db.execute('SELECT status,binding,completion FROM publications WHERE id=?', (identifier,)).fetchone()
                    pending = ('pending', json.dumps(started, sort_keys=True), None)
                    if existing is None:
                        db.execute('INSERT INTO publications VALUES (?,?,?,?)', (identifier, *pending))
                    elif existing != pending:
                        raise ValueError('Neural publication cannot replace an existing completion')
                immutable_json(folder / 'registration_started.json', started)
                db.execute('BEGIN IMMEDIATE')
                if not artifact.exists():
                    model.save(artifact)
                restored = self._loader(model.kind)(artifact)
                if restored.digest != model.digest:
                    raise ValueError('Immutable neural model artifact conflict')
                lease.assert_parent_alive()
                immutable_json(folder / 'manifest.json', {
                    'identity': identity, 'id': identifier, 'artifact_sha256': file_hash(artifact)})
                lease.assert_parent_alive()
                # A manifest alone is only a prepared publication. This second
                # positive anchor is absent if the final check or failure-log
                # persistence fails; historical readers default to rejection.
                from .neural_forward_launcher import _observation
                completed = {
                    'identity': identifier, 'lease_hash': lease.digest,
                    'certificate_digest': fingerprint(certificate),
                    'artifact_sha256': file_hash(artifact),
                    'manifest_sha256': file_hash(folder / 'manifest.json'),
                    'observation': _observation(lease)}
                immutable_json(folder / 'registration_completed.json', completed)
                lease.assert_parent_alive()
                db.execute('UPDATE publications SET status=?,completion=? WHERE id=?',
                           ('completed', fingerprint(completed), identifier))
                lease.assert_parent_alive()
                db.commit()  # Linearization; no fallible application work follows.
            except BaseException as error:
                db.rollback()
                try:
                    immutable_json(folder / 'registration_failed.json', {
                        'identity': identifier, 'reason': type(error).__name__ + ':' + str(error)[:2000]})
                except BaseException as persistence_error:  # noqa: BLE001 - preserve the original failed publication
                    error.add_note('Registration failure persistence failed: ' + repr(persistence_error))
                raise
            finally:
                db.close()
        return identifier

    @staticmethod
    def _linear_pipeline(model):
        return {'kind': 'v7_known_linear', 'pipeline_id': model.pipeline_id,
                'recipe': model.recipe.to_dict(), 'task': 'volatility',
                'rolling_hash': model.recipe.rolling_spec.digest}

    @staticmethod
    def _loader(kind):
        from .convex_eval import RidgePredictor
        if kind == 'v7_log_ridge':
            return RidgePredictor.load
        if kind == 'v7_event_logit':
            from .event_eval import EventPredictor
            return EventPredictor.load
        if kind == 'v7_factor_pipeline':
            from .factor_pool import FactorPipeline
            return FactorPipeline.load
        if kind == 'v7_joint_pool':
            from .joint_pool import JointPredictor
            return JointPredictor.load
        if kind == 'v7_known_linear':
            from .known_linear_reference import KnownLinearPredictor
            return KnownLinearPredictor.load
        if kind == 'v7_neural_forward':
            from .neural_forward import NeuralForwardModel
            return NeuralForwardModel.load
        raise ValueError('Unregistered model kind')

    def load(self, identifier, *, expected_pipeline=None):
        if len(identifier) != 64 or any(c not in '0123456789abcdef' for c in identifier):
            raise ValueError('Invalid model identifier')
        folder = self.directory / identifier
        if (folder / 'registration_failed.json').exists() or (folder / 'serving_isolation.json').exists():
            raise ValueError('Failed neural registration cannot be restored')
        record = json.loads((folder / 'manifest.json').read_text('utf-8'))
        identity = record['identity']
        if record['id'] != identifier or fingerprint(identity) != identifier:
            raise ValueError('Model registry identity mismatch')
        if expected_pipeline is not None and identity['pipeline'] != expected_pipeline:
            raise ValueError('Pipeline identity mismatch')
        if (identity['pipeline'].get('kind') == 'v7_neural_forward'
                and identity['kind'] != 'v7_neural_forward'):
            raise ValueError('A neural pipeline requires a complete neural model')
        if file_hash(folder / 'model.json') != record['artifact_sha256']:
            raise ValueError('Model artifact integrity failure')
        model = self._loader(identity['kind'])(folder / 'model.json')
        if (model.digest != identity['model_digest'] or model.schema != identity['schema']
                or list(model.task_support) != identity['task_support']):
            raise ValueError('Restored model semantic identity mismatch')
        if ((identity['kind'] == 'v7_known_linear' or identity['pipeline'].get('kind') == 'v7_known_linear')
                and (identity['kind'] != 'v7_known_linear' or identity['pipeline'] != self._linear_pipeline(model))):
            raise ValueError('Restored linear pipeline differs from its complete model recipe')
        admitted_versions = {'v7_neural_forward': 2, 'v7_known_linear': 3, 'v7_joint_pool': 4}
        if identity['kind'] == 'v7_neural_forward' or (
                identity['kind'] in admitted_versions and identity['version'] == admitted_versions[identity['kind']]):
            if set(identity) != {'version', 'kind', 'model_digest', 'schema', 'task_support',
                                 'pipeline', 'admission', 'job'} or identity['version'] != admitted_versions[identity['kind']]:
                raise ValueError('Neural registry manifest lacks external admission')
            from .neural_forward_admission import (
                _file_deadline,
                _observation,
                verify_forward_admission,
            )
            from .work_lease import FrozenWorkLease
            verifier = verify_forward_admission
            if identity['kind'] == 'v7_known_linear':
                from .known_linear_forward import verify_known_admission
                verifier = verify_known_admission
            elif identity['kind'] == 'v7_joint_pool':
                from .pool_forward import verify_pool_admission
                verifier = verify_pool_admission
            binding = verifier(model, identity['admission'],
                expected_pipeline=identity['pipeline'], expected_job=identity['job'])
            lease = FrozenWorkLease.from_dict(binding['reservation']['lease'])
            started_path = folder / 'registration_started.json'
            completed_path = folder / 'registration_completed.json'
            if not started_path.is_file() or not completed_path.is_file():
                raise ValueError('Neural registration lacks positive completion evidence')
            started = json.loads(started_path.read_text('utf-8'))
            expected_start = {'identity': identifier, 'lease_hash': lease.digest,
                              'certificate_digest': fingerprint(identity['admission'])}
            completed = json.loads(completed_path.read_text('utf-8'))
            if (started != expected_start
                    or set(completed) != {*expected_start, 'artifact_sha256', 'manifest_sha256', 'observation'}
                    or {k: completed[k] for k in expected_start} != expected_start
                    or completed['artifact_sha256'] != file_hash(folder / 'model.json')
                    or completed['manifest_sha256'] != file_hash(folder / 'manifest.json')):
                raise ValueError('Neural registration completion binding changed')
            _observation(completed['observation'], lease, previous=binding['receipt']['final_io_observation'])
            _file_deadline([folder / 'model.json', folder / 'manifest.json', started_path, completed_path], lease)
            self._verify_publication(identifier, started, completed)
        return model, identity


class ForecastLedger:
    def __init__(self, path):
        path = project_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.records_directory = path.with_suffix('.records')
        self.records_directory.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('''CREATE TABLE IF NOT EXISTS forecasts (
            run TEXT, stream TEXT, origin TEXT, symbol TEXT, task TEXT,
            binding TEXT NOT NULL, prediction TEXT NOT NULL,
            PRIMARY KEY(run,stream,origin,symbol,task))''')
        self.recover_receipts()

    def close(self):
        self.db.close()

    def put(self, *, run, stream, origin, symbol, task, model_id, pipeline_id, prediction):
        stamp = pd.Timestamp(origin)
        if stamp.tzinfo is None:
            raise ValueError('Forecast origin must be timezone-aware')
        if not all(isinstance(v, str) and v for v in (run, stream, symbol, task, model_id, pipeline_id)):
            raise ValueError('Incomplete logical forecast identity')
        key = (run, stream, stamp.tz_convert('UTC').isoformat(), symbol, task)
        binding = fingerprint({'model': model_id, 'pipeline': pipeline_id})
        value = json.dumps(prediction, sort_keys=True, allow_nan=False, separators=(',', ':'))
        receipt = {'key': list(key), 'binding': binding, 'prediction': json.loads(value)}
        with self.db:
            row = self.db.execute('SELECT binding,prediction FROM forecasts WHERE '
                                  'run=? AND stream=? AND origin=? AND symbol=? AND task=?', key).fetchone()
            if row is not None:
                if row != (binding, value):
                    raise ValueError('A logical forecast cannot be rebound or revised')
                return False
            immutable_json(self.records_directory / f'{fingerprint(list(key))}.json',
                           {'receipt': receipt, 'hash': fingerprint(receipt)})
            self.db.execute('INSERT INTO forecasts VALUES (?,?,?,?,?,?,?)', (*key, binding, value))
        return True

    def _read_receipt(self, path):
        try:
            wrapper = json.loads(path.read_text('utf-8'))
            receipt = wrapper['receipt']
            key = receipt['key']
            if (set(wrapper) != {'receipt', 'hash'} or set(receipt) != {'key', 'binding', 'prediction'}
                    or type(key) is not list or len(key) != 5
                    or any(type(v) is not str or not v for v in key)
                    or fingerprint(receipt) != wrapper['hash'] or fingerprint(key) != path.stem
                    or pd.Timestamp(key[2]).tzinfo is None
                    or pd.Timestamp(key[2]).tz_convert('UTC').isoformat() != key[2]):
                raise ValueError('Invalid original forecast receipt')
            return receipt
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError('Original forecast receipt/index integrity failure') from exc

    def recover_receipts(self):
        """Complete a recorded issuance; never evaluate a model or change issued_at."""
        with FileLock(str(self.records_directory) + '.lock'), self.db:
            for path in self.records_directory.glob('*.json'):
                receipt = self._read_receipt(path)
                key = receipt['key']
                value = json.dumps(receipt['prediction'], sort_keys=True, allow_nan=False, separators=(',', ':'))
                row = self.db.execute('SELECT binding,prediction FROM forecasts WHERE '
                    'run=? AND stream=? AND origin=? AND symbol=? AND task=?', key).fetchone()
                if row is not None and row != (receipt['binding'], value):
                    raise ValueError('Original forecast recovery cannot overwrite ledger content')
                if row is None:
                    self.db.execute('INSERT INTO forecasts VALUES (?,?,?,?,?,?,?)',
                                    (*key, receipt['binding'], value))

    def verify(self):
        count = 0
        keys = set()
        for record in self.iter_records():
            key = [record[name] for name in ('run', 'stream', 'origin', 'symbol', 'task')]
            identifier = fingerprint(key)
            saved = self._read_receipt(self.records_directory / f'{identifier}.json')
            expected = {'key': key, 'binding': record['binding'],
                        'prediction': json.loads(record['prediction'])}
            if saved != expected:
                raise ValueError('Original forecast content or logical index integrity failure')
            keys.add(identifier)
            count += 1
        for path in self.records_directory.glob('*.json'):
            if path.stem not in keys:
                raise ValueError('Original forecast missing from logical ledger')
        return {'forecasts': count}

    def iter_records(self):
        for row in self.db.execute('SELECT * FROM forecasts ORDER BY origin,symbol,task,stream'):
            yield dict(zip(('run', 'stream', 'origin', 'symbol', 'task', 'binding', 'prediction'), row, strict=True))
