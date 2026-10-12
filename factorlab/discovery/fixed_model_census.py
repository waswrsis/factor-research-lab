"""Exact finite-population risk of two sealed fitted models, NOT cold refitting.

This oracle evaluates original numeric model instances on a complete registered
finite synthetic population. Training is never rerun; service/activation clocks
and provider randomness are not in this estimand. Consequently its envelopes
must never fill the five original cold-refit/A0 calibration truth slots.
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock
from threadpoolctl import threadpool_limits

from factorlab.discovery.calibration_oracle_evidence import _lifecycle
from factorlab.discovery.evidence import file_hash
from factorlab.discovery.finite_market_oracle import (
    FiniteGenerator,
    _enclose_census,
    _finite_loss,
    _plain,
    _read,
    _write,
)
from factorlab.discovery.monthly_work_authority import source_files
from factorlab.discovery.rolling import RollingSpec, build_rolling_data
from factorlab.discovery.v7_storage import immutable_record
from factorlab.protocol import fingerprint, project_path

VERSION = 'v7-sealed-fitted-model-finite-census-v1'


def _models(directory, identifiers):
    from factorlab.discovery.joint_pool import JointPredictor
    from factorlab.discovery.known_linear_reference import KnownLinearPredictor
    from factorlab.discovery.neural_forward import NeuralForwardModel
    with _lifecycle(directory) as lc:
        lc.verify()
        models = []
        bindings = []
        for identifier in identifiers:
            rows = lc.db.execute('SELECT pipeline FROM jobs WHERE model=? AND status IN (?,?)',
                                 (identifier, 'verified', 'activated')).fetchall()
            if not rows or len({r[0] for r in rows}) != 1:
                raise ValueError('Conditional oracle needs an original verified lifecycle model')
            model, identity = lc.registry.load(identifier, expected_pipeline=json.loads(rows[0][0]))
            if type(model) not in (JointPredictor, KnownLinearPredictor, NeuralForwardModel):
                raise TypeError('Only audited actual pool/known/neural prediction implementations')
            models.append(model)
            bindings.append({'id': identifier, 'identity': identity,
                             'manifest_hash': file_hash(lc.registry.directory/identifier/'manifest.json'),
                             'artifact_hash': file_hash(lc.registry.directory/identifier/'model.json')})
        return models, {'lifecycle_digest': lc.plan.digest, 'models': _plain(bindings)}


def _atom_risk(generator, atom, models, score_start, score_end):
    value = generator.read_atom(atom)
    gaps = {s['hour'] for s in value['schedule'] if s['gap']}
    n = len(generator.law.symbols)
    frame = pd.DataFrame([row for i, row in enumerate(value['rows']) if i//n not in gaps])
    frame['open_time'] = pd.to_datetime(frame.open_time, utc=True)
    frame['available_at'] = frame.open_time+pd.Timedelta(hours=1, seconds=5)
    # This is an explicitly assumed, frozen computational availability schedule,
    # never an asserted actual historical receipt or a source for live forecasts.
    data = build_rolling_data(frame, RollingSpec(availability_delay_seconds=5))
    with threadpool_limits(limits=1):
        predictions = [model.predict_history(data.frame) for model in models]
    metadata = data.frame.metadata
    origin = pd.to_datetime(metadata.feature_open_time, utc=True)
    start = pd.Timestamp(generator.law.start)+pd.Timedelta(hours=score_start)
    end = pd.Timestamp(generator.law.start)+pd.Timedelta(hours=score_end)
    nominal = (score_end-score_start)*len(generator.law.symbols)
    selected = (origin >= start) & (origin < end)
    mask = selected.to_numpy() & data.observed & predictions[0][1] & predictions[1][1]
    expected = {(t.isoformat(), s) for t in pd.date_range(start, end, freq='h', inclusive='left')
                for s in generator.law.symbols}
    observed = {(origin.iloc[i].isoformat(), metadata.symbol.iloc[i]) for i in np.flatnonzero(mask)}
    if observed != expected:
        return {'atom': atom, 'defined': False, 'reference': None, 'candidate': None,
                'reason': 'missing_registered_common_support', 'expected_pairs': nominal,
                'actual_pairs': len(observed), 'market_atom_hash': fingerprint(value)}
    hourly = {}
    for i in np.flatnonzero(mask):
        y = float(data.target[i])
        losses = [_finite_loss(y, float(prediction[0][i])) for prediction in predictions]
        hourly.setdefault(origin.iloc[i].isoformat(), []).append(losses)
    risks = [math.fsum(math.fsum(pair[k] for pair in rows)/len(rows) for rows in hourly.values())/len(hourly)
             for k in range(2)]
    return {'atom': atom, 'defined': True, 'reference': risks[0], 'candidate': risks[1],
            'expected_pairs': nominal, 'actual_pairs': len(observed),
            'market_atom_hash': fingerprint(value), 'support_hash': fingerprint(sorted(observed))}


class FixedModelCensus:
    """Once-only bounded inference work units; failures retained, no hidden refit."""
    def __init__(self, directory, generator, lifecycle_directory, reference_id, candidate_id,
                 *, score_start_hour=1234, score_end_hour=None):
        if type(generator) is not FiniteGenerator:
            raise TypeError('Actual frozen finite generator required')
        generator.verify()
        end = generator.law.hours-14 if score_end_hour is None else score_end_hour
        if (type(score_start_hour) is not int or type(end) is not int
                or not 1234 <= score_start_hour < end <= generator.law.hours-14):
            raise ValueError('Frozen causal warmup and complete twelve-hour horizon required')
        self.directory = project_path(directory)
        self.generator = generator
        self.original = project_path(lifecycle_directory)
        self.identifiers = (reference_id, candidate_id)
        _, binding = _models(self.original, self.identifiers)
        self.identity = {'version': VERSION, 'directory': str(self.directory),
            'generator_directory': str(generator.directory), 'generator_digest': generator.digest,
            'lifecycle_directory': str(self.original), 'model_binding': binding,
            'reference_id': reference_id, 'candidate_id': candidate_id,
            'score_start_hour': score_start_hour, 'score_end_hour': end,
            'source_files': source_files(), 'implementation_hash': file_hash(Path(__file__)),
            'estimand': 'conditional_on_these_two_sealed_fitted_models_uniform_complete_finite_population',
            'availability': 'assumed_close_plus_5s_computational_only_not_actual_service',
            'inference_threads': 1,
            'required_atoms': generator.law.support_size, 'maximum_atom_executions': generator.law.support_size,
            'original_cold_refit_claim_scope_eligible': False, 'formal_gate_eligible': False}
        self.directory.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.directory/'.init.lock'), timeout=0):
            manifest, path = self.directory/'identity.json', self.directory/'units.sqlite'
            if not manifest.exists():
                if any(p.name != '.init.lock' for p in self.directory.iterdir()):
                    raise ValueError('Missing census identity cannot reset original inference work')
                _write(manifest, self.identity)
                with closing(sqlite3.connect(path)) as db, db:
                    db.execute('PRAGMA synchronous=FULL')
                    db.execute('CREATE TABLE units (atom INTEGER PRIMARY KEY,status TEXT,record TEXT,hash TEXT)')
            if not path.is_file():
                raise ValueError('Original census journal missing; cannot reset')
            self._verify()

    def _verify(self):
        self.generator.verify()
        if (_read(self.directory/'identity.json') != self.identity
                or self.identity['source_files'] != source_files()
                or self.identity['implementation_hash'] != file_hash(Path(__file__))):
            raise ValueError('Original model census/source changed')
        _, binding = _models(self.original, self.identifiers)
        if binding != self.identity['model_binding']:
            raise ValueError('Original sealed model changed')
        path = self.directory/'units.sqlite'
        if not path.is_file():
            raise ValueError('Original census journal missing; cannot reset')
        with closing(sqlite3.connect(path.as_uri()+'?mode=ro', uri=True)) as db, db:
            rows = db.execute('SELECT atom,status,record,hash FROM units ORDER BY atom').fetchall()
        indexed = {r[0] for r in rows}
        anchored = {int(p.parent.name) for p in (self.directory/'units').glob('*/reservation.json')}
        folders = {int(p.name) for p in (self.directory/'units').glob('*') if p.is_dir()}
        if indexed != anchored or indexed != folders or not indexed <= set(range(self.generator.law.support_size)):
            raise ValueError('Original census units/anchors missing; no retry budget')
        for atom, status, text, digest in rows:
            reservation = _read(self.directory/'units'/str(atom)/'reservation.json')
            if reservation != {'census': fingerprint(self.identity), 'atom': atom}:
                raise ValueError('Census reservation binding changed')
            if status not in {'reserved', 'complete', 'failed_consumed'}:
                raise ValueError('Unknown original census state')
            if status == 'reserved':
                if text is not None or digest is not None:
                    raise ValueError('Uncommitted census output')
            else:
                result = json.loads(text)
                if fingerprint(result) != digest or result['atom'] != atom:
                    raise ValueError('Census result/index hash changed')
                mirror = self.directory/'units'/str(atom)/'result.json'
                if not mirror.exists() or _read(mirror) != result:
                    raise ValueError('Census result mirror missing; no execution retry')
        return rows

    def run(self, *, max_new_atoms=1):
        if type(max_new_atoms) is not int or not 1 <= max_new_atoms <= 128:
            raise ValueError('Bounded original census work per call required')
        with FileLock(str(self.directory/'.run.lock'), timeout=0):
            rows = self._verify()
            spent = {r[0] for r in rows}
            new = 0
            for atom in range(self.generator.law.support_size):
                if atom in spent or new >= max_new_atoms:
                    continue
                unit = self.directory/'units'/str(atom)
                _write(unit/'reservation.json', {'census': fingerprint(self.identity), 'atom': atom})
                with closing(sqlite3.connect(self.directory/'units.sqlite')) as db, db:
                    db.execute('PRAGMA synchronous=FULL')
                    db.execute('INSERT INTO units VALUES (?, ?, NULL, NULL)', (atom, 'reserved'))
                new += 1
                started = time.monotonic()
                try:
                    self.generator.produce_atom(atom)
                    models, binding = _models(self.original, self.identifiers)
                    if binding != self.identity['model_binding']:
                        raise ValueError('Original models changed during census')
                    result = _atom_risk(self.generator, atom, models, self.identity['score_start_hour'],
                                        self.identity['score_end_hour'])
                    result.update(elapsed_seconds=time.monotonic()-started)
                    status = 'complete'
                except (ValueError, TypeError, RuntimeError, OSError, ArithmeticError, MemoryError) as exc:
                    result = {'atom': atom, 'defined': False, 'reference': None, 'candidate': None,
                              'failure_type': type(exc).__name__, 'elapsed_seconds': time.monotonic()-started}
                    status = 'failed_consumed'
                _write(unit/'result.json', result)
                with closing(sqlite3.connect(self.directory/'units.sqlite')) as db, db:
                    db.execute('PRAGMA synchronous=FULL')
                    db.execute('UPDATE units SET status=?,record=?,hash=? WHERE atom=? AND status=?',
                        (status, json.dumps(result, sort_keys=True), fingerprint(result), atom, 'reserved'))
            current = self._verify()
            return {'reserved_atoms': len(current), 'new_atoms': new,
                    'states': {str(r[0]): r[1] for r in current},
                    'formal_gate_eligible': False, 'risk_envelope': None}

    def original_claim_envelope(self):
        self._verify()
        raise ValueError('Fixed fitted-model risk cannot replace original cold-refit/A0 claim truth')

    def report(self):
        rows = self._verify()
        risks, undefined = [], []
        for atom, status, text, _ in rows:
            if status != 'complete':
                undefined.append(atom); continue
            result = json.loads(text)
            models, _ = _models(self.original, self.identifiers)
            actual = _atom_risk(self.generator, atom, models, self.identity['score_start_hour'],
                                self.identity['score_end_hour'])
            if {k: v for k, v in result.items() if k != 'elapsed_seconds'} != actual:
                raise ValueError('Original census risks differ from actual sealed-model prediction replay')
            if not actual['defined']:
                undefined.append(atom)
            risks.append({k: actual[k] for k in ('atom','reference','candidate','defined')})
        missing = self.generator.law.support_size-len(rows)
        digest = fingerprint({'identity': self.identity, 'risks': risks})
        envelope = None if missing or undefined else _enclose_census(risks, generator=self.generator,
                                                                    oracle_digest=digest)
        value = {'census_digest': fingerprint(self.identity), 'completed_atoms': len(risks),
            'missing_atoms': missing, 'undefined_or_failed_atoms': undefined,
            'registered_atoms': self.generator.law.support_size,
            'risk_envelope': None if envelope is None else _plain(asdict(envelope)),
            'estimand': self.identity['estimand'], 'original_cold_refit_claim_scope_eligible': False,
            'formal_gate_eligible': False, 'full_V7': False}
        immutable_record(self.directory/'reports'/f'{fingerprint(value)}.json', value)
        return value
