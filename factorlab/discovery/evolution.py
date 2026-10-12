"""Resumable sequential research with frozen shadow comparisons and bounded search.

The SQLite transaction is the commit boundary for a timestamp. Model files are
immutable; interrupted searches consume their slot and retain the incumbent.
Promotion is an operational research rule, never independent market evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from dataclasses import asdict, dataclass, replace
from itertools import zip_longest
from uuid import uuid4

import numpy as np
import pandas as pd
from filelock import FileLock

from factorlab.protocol import fingerprint, project_path, utc
from factorlab.time_utils import hour_ids

from .contracts import ExperimentSpec, ModuleSpec, RepresentationSpec, TaskDataset
from .evidence import EvidenceRegistry, file_hash, immutable_json, snapshot_source, source_paths
from .experiment import ModelGroup
from .learning import FittedResearchModel, fit
from .replay import _data_hash, _model_cutoffs, _prediction_inputs, _training_parts
from .scoring import evaluate, paired_comparison, per_sample_loss


@dataclass(frozen=True)
class EvolutionPolicy:
    mode: str = 'evolve'
    search_mode: str = 'diagnostic'
    update_every_hours: int = 168
    shadow_hours: int = 1008
    max_cycles: int = 6
    population_size: int = 3
    inherit_weights: bool = True
    trigger: str = 'scheduled'
    drift_window_hours: int = 48
    drift_ratio: float = 1.5
    memory_limit: int = 30
    program_search: bool = False
    revived_parents: int = 0

    def __post_init__(self):
        if self.mode not in {'fixed', 'refit', 'evolve'}:
            raise ValueError('Unknown evolution mode')
        if self.search_mode not in {'random', 'diagnostic', 'external'}:
            raise ValueError('Unknown proposal mode')
        if self.trigger not in {'scheduled', 'degradation'}:
            raise ValueError('Unknown research trigger')
        for name in ('update_every_hours', 'shadow_hours', 'max_cycles', 'population_size',
                     'drift_window_hours', 'memory_limit'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.max_cycles > 100 or self.population_size > 6:
            raise ValueError('Evolution budget exceeds local bounds')
        if not math.isfinite(self.drift_ratio) or self.drift_ratio <= 1:
            raise ValueError('Degradation ratio must exceed one')
        if type(self.inherit_weights) is not bool:
            raise ValueError('inherit_weights must be boolean')
        if type(self.program_search) is not bool or type(self.revived_parents) is not int or not 0 <= self.revived_parents <= 5:
            raise ValueError('Invalid autonomous program policy')


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


RESEARCH_INLINE_BYTES = 256 * 1024


def _canonical_chunks(value):
    return json.JSONEncoder(sort_keys=True, separators=(',', ':'), allow_nan=False).iterencode(value)


def _large_research(value):
    size = 0
    for chunk in _canonical_chunks(value):
        size += len(chunk.encode())
        if size > RESEARCH_INLINE_BYTES:
            return True
    return False


def _research_summary(value):
    """Bounded report facts; full candidate histories remain in the artifact."""
    result = {key: item for key, item in list(value.items())[:128]
              if key != '_research_ref' and len(key) <= 100
              and (item is None or type(item) in (bool, int, float)
                   or isinstance(item, str) and len(item) <= 2000)}
    records = value.get('records', [])
    operations = sorted({row.get('proposal', {}).get('operation') for row in records
                         if isinstance(row, dict) and isinstance(row.get('proposal'), dict)
                         and isinstance(row['proposal'].get('operation'), str)
                         and len(row['proposal']['operation']) <= 100})
    memory = value.get('development_memory', [])
    result.update(summary_scope='bounded_research_summary_v1', records_count=len(records),
                  operations=operations[:64], operations_truncated=len(operations) > 64,
                  development_memory_count=len(memory),
                  development_memory=[{key: row[key] for key in (
                      'candidate_id', 'status', 'iteration', 'evidence_cutoff', 'source_protocol_hash')
                      if key in row and (row[key] is None or type(row[key]) in (int, float, bool)
                                         or isinstance(row[key], str) and len(row[key]) <= 256)}
                      for row in memory[-4:] if isinstance(row, dict)])
    return result


def _research_reference_path(directory, reference):
    if (not isinstance(reference, dict) or set(reference) != {'format', 'path', 'sha256'}
            or reference['format'] != 'research-artifact-v1'
            or not isinstance(reference['sha256'], str) or len(reference['sha256']) != 64
            or any(char not in '0123456789abcdef' for char in reference['sha256'])
            or reference['path'] != f"research-objects/{reference['sha256']}.json"):
        raise ValueError('Invalid research artifact reference')
    path = project_path(directory / reference['path'])
    if not path.is_relative_to(directory.resolve()):
        raise ValueError('Research artifact reference escaped its run directory')
    return path


def _verify_research_reference(directory, research):
    if not isinstance(research, dict) or '_research_ref' not in research:
        return
    reference = research['_research_ref']
    path = _research_reference_path(directory, reference)
    if not path.is_file() or file_hash(path) != reference['sha256']:
        raise ValueError('Research artifact is missing or changed')


def _store_research(directory, research):
    if '_research_ref' in research:
        _verify_research_reference(directory, research)
        return research
    if not _large_research(research):
        return research
    folder = project_path(directory / 'research-objects')
    if not folder.is_relative_to(directory.resolve()):
        raise ValueError('Research artifact directory escaped its run directory')
    folder.mkdir(exist_ok=True)
    temporary = folder / (uuid4().hex + '.tmp')
    digest = hashlib.sha256()
    try:
        with temporary.open('xb') as stream:
            for chunk in _canonical_chunks(research):
                encoded = chunk.encode()
                digest.update(encoded)
                stream.write(encoded)
        sha256 = digest.hexdigest()
        reference = {'format': 'research-artifact-v1',
                     'path': f'research-objects/{sha256}.json', 'sha256': sha256}
        target = _research_reference_path(directory, reference)
        with FileLock(str(target) + '.lock'):
            if target.exists():
                if file_hash(target) != sha256:
                    raise ValueError('Research artifact is missing or changed')
            else:
                temporary.replace(target)
        return {**_research_summary(research), '_research_ref': reference}
    finally:
        temporary.unlink(missing_ok=True)


def _members(model):
    return model.models if isinstance(model, ModelGroup) else [model]


def _architecture(model):
    structures = [member.metadata.get('architecture') for member in _members(model)]
    if any(value != structures[0] for value in structures):
        raise ValueError('Seed group architectures disagree')
    return RepresentationSpec.from_dict(structures[0]) if structures[0] else None


def _causal_model(model, data, cutoff):
    if model.schema != data.schema:
        raise ValueError('Model and evolution input schema differ')
    return [_model_cutoffs(member, utc(cutoff)) for member in _members(model)]


def _visible(data, cutoff):
    """Only features already available and fully matured labels enter a digest."""
    result = data.subset((data.meta.available_at <= utc(cutoff)).to_numpy())
    for task in result.tasks:
        allowed = result.task_mask(task.name, cutoff)
        result.observed[task.name] = allowed
        result.targets[task.name] = result.targets[task.name].astype(float, copy=True)
        result.targets[task.name][~allowed] = np.nan
        unknown = np.isnat(result.maturity[task.name]) | (result.maturity[task.name] > utc(cutoff).to_datetime64())
        result.maturity[task.name][unknown] = np.datetime64('NaT')
    return result


def _visible_hash(data, cutoff):
    visible = _visible(data, cutoff)
    columns = [name for name in ('sample_id', 'symbol', 'available_at', 'feature_open_time',
                                'window_start_open_time') if name in visible.meta]
    return fingerprint({'arrays': _data_hash(visible),
                        'meta': visible.meta[columns].astype(str).to_dict('list')})


def _current_source_files():
    """Freeze the full project Python program and declared locked dependencies."""
    root = project_path('.')
    files = source_paths(root)
    return {str(path.relative_to(root)).replace('\\', '/'): file_hash(path)
            for path in sorted(files)}


def _verify_source_archive(directory, identity):
    """Historical inspection checks saved bytes, never today's implementation."""
    archive = identity.get('source_archive')
    if archive is None:
        return {'mode': 'legacy_discovery_hash_only', 'archive_verified': False,
                'files': len(identity.get('source_files', {})),
                'limitations': 'No source copies or complete feature/data/dependency freeze; '
                               'historical inspection only, not resumable under the archived-source contract.'}
    files = identity.get('source_files')
    required = {'factorlab/features.py', 'factorlab/data.py', 'factorlab/protocol.py',
                'pyproject.toml', 'uv.lock'}
    if (not isinstance(archive, dict) or set(archive) != {'path', 'digest'}
            or archive['path'] != 'source' or not isinstance(files, dict)
            or not required <= set(files) or fingerprint(files) != archive['digest']):
        raise ValueError('Evolution source archive identity is invalid')
    folder = project_path(directory / archive['path'])
    if not folder.is_relative_to(directory.resolve()):
        raise ValueError('Evolution source archive escaped its run directory')
    manifest_path = folder / 'manifest.json'
    if not manifest_path.is_file() or json.loads(manifest_path.read_text('utf-8')) != files:
        raise ValueError('Evolution source archive manifest is missing or changed')
    actual = {}
    for path in folder.rglob('*'):
        if not path.is_file() or path == manifest_path:
            continue
        if not path.resolve().is_relative_to(folder):
            raise ValueError('Evolution source archive contains an external file reference')
        # immutable_json's OS lock is metadata, not an archived source member.
        # Keep the external-reference check above, and never exempt a symlink.
        if path == folder / 'manifest.json.lock' and not path.is_symlink():
            continue
        actual[str(path.relative_to(folder)).replace('\\', '/')] = file_hash(path)
    if actual != files:
        raise ValueError('Evolution source archive integrity failure')
    return {'mode': 'factorlab_python_and_locked_dependencies', 'archive_verified': True,
            'files': len(files), 'digest': archive['digest'], 'path': archive['path'],
            'runtime_binary_snapshot_claimed': False}


class EvolutionStore:
    def __init__(self, directory):
        self.directory = project_path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'state.sqlite'
        self.db = sqlite3.connect(self.path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS models (id TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS cycles (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, value TEXT NOT NULL, hash TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS predictions (
                sample TEXT NOT NULL, task TEXT NOT NULL, model TEXT NOT NULL,
                origin TEXT NOT NULL, value TEXT NOT NULL, role TEXT NOT NULL, cycle INTEGER,
                symbol TEXT NOT NULL,
                settled INTEGER NOT NULL DEFAULT 0, loss REAL, status TEXT, target REAL,
                PRIMARY KEY(sample,task,model));
        ''')
        if 'symbol' not in {row[1] for row in self.db.execute('PRAGMA table_info(predictions)')}:
            self.close()
            raise ValueError('Pre-release evolution ledger lacks symbol provenance; start a new run')

    def close(self):
        self.db.close()

    def state(self):
        row = self.db.execute('SELECT value FROM state WHERE id=1').fetchone()
        return json.loads(row['value']) if row else None

    def save_state(self, state):
        self.db.execute('INSERT OR REPLACE INTO state VALUES (1,?)', (_json(state),))
        self.event('state_checkpoint', state['last_timestamp'] or state['identity']['protocol']['train_end'],
                   {'state': fingerprint(state), 'cycles': self._table_hash('cycles'),
                    'models': self._table_hash('models')})

    def _table_hash(self, table):
        if table not in {'cycles', 'models'}:
            raise ValueError('Unexpected ledger table')
        digest = hashlib.sha256(b'[')
        separator = b''
        for row in self.db.execute(f'SELECT * FROM {table} ORDER BY id'):
            digest.update(separator)
            for chunk in _canonical_chunks(dict(row)):
                digest.update(chunk.encode())
            separator = b','
        digest.update(b']')
        return digest.hexdigest()

    def event(self, event, timestamp, value):
        row = self.db.execute('SELECT seq,hash FROM events ORDER BY seq DESC LIMIT 1').fetchone()
        payload = {'sequence': row['seq'] + 1 if row else 0, 'previous': row['hash'] if row else '0' * 64,
                   'event': event, 'timestamp': utc(timestamp).isoformat(), 'value': value}
        self.db.execute('INSERT INTO events VALUES (?,?,?)',
                        (payload['sequence'], _json(payload), fingerprint(payload)))

    def iter_events(self):
        """Verify and yield one immutable event at a time, including artifact integrity."""
        previous = '0' * 64
        references = set()
        for index, row in enumerate(self.db.execute('SELECT * FROM events ORDER BY seq')):
            payload = json.loads(row['value'])
            if (payload['previous'] != previous or row['hash'] != fingerprint(payload)
                    or row['seq'] != index or payload['sequence'] != index):
                raise ValueError('Evolution event chain is invalid')
            previous = row['hash']
            research = payload['value'].get('research', {}) if isinstance(payload['value'], dict) else {}
            if isinstance(research, dict) and '_research_ref' in research:
                key = fingerprint(research['_research_ref'])
                if key not in references:
                    _verify_research_reference(self.directory, research)
                    references.add(key)
            yield {**payload, 'sha256': previous}

    def events(self, *, compact=False):
        if not compact:
            return list(self.iter_events())
        result = {'event_count': 0, 'event_head': None}
        for event in self.iter_events():
            result.update(event_count=result['event_count'] + 1, event_head=event['sha256'])
        return result

    def verify(self, *, compact=False):
        """Replay into a disposable disk ledger; compact mode retains no event list."""
        scratch_path = project_path(self.directory / ('.verify-' + uuid4().hex + '.sqlite'))
        scratch = sqlite3.connect(scratch_path)
        scratch.row_factory = sqlite3.Row
        events = [] if not compact else None
        checkpoint, count, head = None, 0, None
        try:
            scratch.execute('PRAGMA journal_mode=OFF')
            scratch.execute('PRAGMA synchronous=OFF')
            scratch.execute('PRAGMA cache_size=-1024')
            schema = self.db.execute("SELECT sql FROM sqlite_master WHERE name='predictions'").fetchone()[0]
            scratch.execute(schema)
            for event in self.iter_events():
                count, head = count + 1, event['sha256']
                if events is not None:
                    events.append(event)
                if event['event'] == 'state_checkpoint':
                    checkpoint = event['value']
                elif event['event'] == 'forecast_batch':
                    for row in event['value']['records']:
                        expected = {**row, 'settled': 0, 'loss': None, 'status': None, 'target': None}
                        columns = ('sample', 'task', 'model', 'origin', 'value', 'role', 'cycle',
                                   'symbol', 'settled', 'loss', 'status', 'target')
                        if set(expected) != set(columns):
                            raise ValueError('Evolution prediction ledger schema changed')
                        try:
                            scratch.execute('INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                                            tuple(expected[key] for key in columns))
                        except sqlite3.IntegrityError as exc:
                            raise ValueError('Duplicate forecast in immutable ledger') from exc
                elif event['event'] == 'settlement':
                    for row in event['value']['records']:
                        changed = scratch.execute('''UPDATE predictions SET settled=?,loss=?,status=?,target=?
                            WHERE sample=? AND task=? AND model=? AND settled=0''', tuple(row[key] for key in (
                                'settled', 'loss', 'status', 'target', 'sample', 'task', 'model')))
                        if changed.rowcount != 1:
                            raise ValueError('Unexpected settlement in immutable ledger')
            state = self.state()
            if state is None or checkpoint != {
                'state': fingerprint(state), 'cycles': self._table_hash('cycles'),
                'models': self._table_hash('models')
            }:
                raise ValueError('Evolution ledger state/cycle/model integrity failure')
            query = 'SELECT * FROM predictions ORDER BY sample,task,model'
            for actual, expected in zip_longest(self.db.execute(query), scratch.execute(query)):
                if actual is None or expected is None or dict(actual) != dict(expected):
                    raise ValueError('Evolution prediction/settlement ledger integrity failure')
            for row in self.db.execute('SELECT value FROM cycles'):
                _verify_research_reference(self.directory, json.loads(row['value']).get('research', {}))
        finally:
            scratch.close()
            scratch_path.unlink(missing_ok=True)
        manifest = json.loads((self.directory / 'manifest.json').read_text('utf-8'))
        identity = {key: value for key, value in manifest['manifest'].items() if key != 'evidence_exposure'}
        if (manifest['sha256'] != fingerprint(manifest['manifest'])
                or fingerprint(identity) != fingerprint(state['identity'])):
            raise ValueError('Evolution manifest identity integrity failure')
        _verify_source_archive(self.directory, state['identity'])
        return events if events is not None else {'event_count': count, 'event_head': head}

    def cycle(self, cycle_id):
        row = self.db.execute('SELECT value FROM cycles WHERE id=?', (cycle_id,)).fetchone()
        return json.loads(row['value']) if row else None

    def put_cycle(self, record):
        if isinstance(record.get('research'), dict):
            record['research'] = _store_research(self.directory, record['research'])
        self.db.execute('INSERT OR REPLACE INTO cycles VALUES (?,?)', (record['id'], _json(record)))

    def load_research(self, summary):
        """Explicit full-artifact read for audit; summaries never expand implicitly."""
        _verify_research_reference(self.directory, summary)
        if '_research_ref' not in summary:
            return summary
        return json.loads(_research_reference_path(self.directory, summary['_research_ref']).read_text('utf-8'))

    def save_model(self, model, cutoff):
        identifier = model.digest
        found = self.db.execute('SELECT value FROM models WHERE id=?', (identifier,)).fetchone()
        if found:
            return identifier
        folder = self.directory / 'models' / (identifier[:16] + '-' + uuid4().hex[:8])
        folder.mkdir(parents=True)
        if isinstance(model, ModelGroup):
            model.save(folder)
            kind = 'group'
        else:
            model.save(folder / 'model.pt')
            kind = 'single'
        files = {str(p.relative_to(folder)).replace('\\', '/'): file_hash(p)
                 for p in folder.rglob('*') if p.is_file() and p.suffix != '.lock'}
        value = {'id': identifier, 'kind': kind, 'path': str(folder.relative_to(self.directory)),
                 'files': files, 'cutoff': utc(cutoff).isoformat(),
                 'architecture': asdict(_architecture(model)) if _architecture(model) else None}
        self.db.execute('INSERT INTO models VALUES (?,?)', (identifier, _json(value)))
        return identifier

    def load_model(self, identifier):
        row = self.db.execute('SELECT value FROM models WHERE id=?', (identifier,)).fetchone()
        if not row:
            raise ValueError('Missing committed evolution model')
        value = json.loads(row['value'])
        folder = project_path(self.directory / value['path'])
        files = {str(p.relative_to(folder)).replace('\\', '/'): file_hash(p)
                 for p in folder.rglob('*') if p.is_file() and p.suffix != '.lock'}
        if files != value['files']:
            raise ValueError('Evolution model artifact changed')
        model = (ModelGroup.load(folder) if value['kind'] == 'group'
                 else FittedResearchModel.load(folder / 'model.pt'))
        if model.digest != identifier:
            raise ValueError('Evolution model identity changed')
        return model


def _fit_group(train, development, architecture, settings, seeds, device, initial_model=None):
    members = _members(initial_model) if initial_model is not None else []
    fitted = []
    for index, seed in enumerate(seeds):
        initial = members[index] if index < len(members) else None
        fitted.append(fit(train, development, architecture, replace(settings, seed=seed), device,
                          initial_model=initial, retain_initial_scalers=initial is not None))
    return ModelGroup(fitted)


def _settle(store, data, timestamp, by_id):
    tasks = {task.name: task for task in data.tasks}
    records = []
    for row in store.db.execute('SELECT * FROM predictions WHERE settled=0').fetchall():
        index = by_id.get(row['sample'])
        if index is None:
            raise ValueError('Previously predicted sample disappeared')
        ready = data.maturity[row['task']][index]
        if np.isnat(ready) or ready > timestamp.to_datetime64():
            continue
        observed = data.observed[row['task']][index]
        value, loss = None, None
        if observed:
            value = float(data.targets[row['task']][index])
            raw = np.asarray([json.loads(row['value'])], dtype=float)
            loss = float(per_sample_loss(tasks[row['task']], np.asarray([value]), raw)[0])
        status = 'scored' if observed else 'unobserved'
        store.db.execute('''UPDATE predictions SET settled=1,loss=?,status=?,target=?
                            WHERE sample=? AND task=? AND model=?''',
                         (loss, status, value, row['sample'], row['task'], row['model']))
        records.append({'sample': row['sample'], 'task': row['task'], 'model': row['model'],
                        'settled': 1, 'loss': loss, 'status': status, 'target': value})
    if records:
        store.event('settlement', timestamp, {'count': len(records), 'records': records})


def _record_predictions(store, model, data, role, cycle=None):
    clean = _prediction_inputs(data)
    predictions = model.predict(clean)
    # evaluate validates outputs even when no labels have matured.
    evaluate(clean, predictions)
    records = []
    for task in data.tasks:
        if task.name not in predictions:
            raise ValueError('Evolution requires predictions for every registered task')
        for index, row in data.meta.iterrows():
            raw = np.asarray(predictions[task.name][index]).tolist()
            store.db.execute('''INSERT INTO predictions
                (sample,task,model,origin,value,role,cycle,symbol) VALUES (?,?,?,?,?,?,?,?)''',
                (row.sample_id, task.name, model.digest, utc(row.available_at).isoformat(),
                 _json(raw), role, cycle, row.symbol))
            records.append({'sample': row.sample_id, 'task': task.name, 'model': model.digest,
                            'origin': utc(row.available_at).isoformat(), 'value': _json(raw),
                            'role': role, 'cycle': cycle, 'symbol': row.symbol})
    store.event('forecast_batch', data.meta.available_at.iloc[0], {'records': records})


def _gate(store, cycle, data, spec, policy):
    task = spec.primary_task
    rows = store.db.execute('''SELECT c.sample,c.value AS candidate,r.value AS reference,
          c.settled AS c_ready,r.settled AS r_ready FROM predictions c
          LEFT JOIN predictions r ON c.sample=r.sample AND c.task=r.task AND r.model=?
          WHERE c.cycle=? AND c.role='shadow' AND c.task=? ORDER BY c.origin,c.sample''',
          (cycle['reference'], cycle['id'], task)).fetchall()
    if not rows or any(row['reference'] is None or not row['c_ready'] or not row['r_ready'] for row in rows):
        return {'status': 'insufficient_evidence', 'reason': 'incomplete_paired_shadow',
                'predictive_claim_supported': False, 'independent_confirmation': False,
                'operational_development_improvement_supported': False,
                'scope': 'sequential_operational_gate_not_independent_confirmation',
                'support': len(rows)}
    lookup = {sample: index for index, sample in enumerate(data.meta.sample_id)}
    subset = data.subset(np.asarray([lookup[row['sample']] for row in rows]))
    result = paired_comparison(subset,
        {task: np.asarray([json.loads(row['candidate']) for row in rows])},
        {task: np.asarray([json.loads(row['reference']) for row in rows])},
        task, spec, family_size=policy.max_cycles)
    result['operational_development_improvement_supported'] = result['predictive_claim_supported']
    result.update(predictive_claim_supported=False, independent_confirmation=False,
                  scope='sequential_operational_gate_not_independent_confirmation')
    if spec.selection.mode == 'walk_forward':
        guard = _shadow_auxiliary_guard(store, cycle, spec)
        result['auxiliary_guard'] = guard
        if not guard['passed']:
            result['primary_comparison_status'] = result['status']
            result.update(status='auxiliary_guard_failed',
                          operational_development_improvement_supported=False)
    result['one_look_only'] = True
    return result


def _shadow_auxiliary_guard(store, cycle, spec):
    """A point-loss safeguard on mature pairs, not a noninferiority claim.

    Use already settled forecast losses. No candidate is re-fitted or chosen
    after reading the shadow outcomes, and descriptive state is not a forecast.
    """
    limit = spec.selection.auxiliary_max_relative_degradation
    tasks = {}
    primary_samples = {row['sample'] for row in store.db.execute(
        "SELECT sample FROM predictions WHERE cycle=? AND role='shadow' AND task=?",
        (cycle['id'], spec.primary_task)).fetchall()}
    for task in spec.tasks:
        if not task.predictive or task.name == spec.primary_task:
            continue
        rows = store.db.execute('''SELECT c.sample,c.origin,c.symbol,c.loss AS candidate_loss,
              r.loss AS reference_loss,c.settled AS c_ready,r.settled AS r_ready,
              r.sample AS reference_sample FROM predictions c
              LEFT JOIN predictions r ON c.sample=r.sample AND c.task=r.task AND r.model=?
              WHERE c.cycle=? AND c.role='shadow' AND c.task=? ORDER BY c.origin,c.sample''',
              (cycle['reference'], cycle['id'], task.name)).fetchall()
        record = {'passed': False, 'support': 0, 'relative_degradation_limit': limit}
        tasks[task.name] = record
        if not primary_samples or {row['sample'] for row in rows} != primary_samples:
            record['reason'] = 'auxiliary_origin_cohort_mismatch'
            continue
        if not rows or any(row['reference_sample'] is None or not row['c_ready']
                           or not row['r_ready'] for row in rows):
            record['reason'] = 'incomplete_mature_auxiliary_pairs'
            continue
        if any((row['candidate_loss'] is None) != (row['reference_loss'] is None) for row in rows):
            record['reason'] = 'auxiliary_pair_observation_mismatch'
            continue
        valid = [dict(row) for row in rows if row['candidate_loss'] is not None]
        if not valid:
            record['reason'] = 'no_observed_auxiliary_pairs'
            continue
        frame = pd.DataFrame(valid)
        if (not np.isfinite(frame[['candidate_loss', 'reference_loss']]).all().all()
                or (frame[['candidate_loss', 'reference_loss']] < 0).any().any()):
            record['reason'] = 'invalid_auxiliary_losses'
            continue
        frame['hour'] = pd.to_datetime(frame.origin, utc=True).dt.floor('h')
        hourly = frame.groupby(['hour', 'symbol'])[['candidate_loss', 'reference_loss']].mean()
        hourly = hourly.groupby(level='hour').mean()
        blocks = len(np.unique(hour_ids(hourly.index) // spec.bootstrap_block_hours))
        candidate, reference = hourly.mean().tolist()
        degradation = (candidate - reference) / max(abs(reference), 1e-12)
        record.update(support=len(valid), n_hours=len(hourly), n_blocks=blocks,
                      candidate_loss=float(candidate), reference_loss=float(reference),
                      relative_degradation=float(degradation),
                      aggregation='equal_symbol_within_hour_then_equal_hour')
        if blocks < spec.min_blocks:
            record['reason'] = 'insufficient_auxiliary_calendar_blocks'
        elif degradation > limit + 1e-12:
            record['reason'] = 'auxiliary_degradation_limit_exceeded'
        else:
            record.update(passed=True, reason='empirical_auxiliary_limit_met')
    return {'passed': all(row['passed'] for row in tasks.values()), 'tasks': tasks,
            'scope': 'empirical_development_safeguard_not_statistical_noninferiority',
            'one_look_only': True, 'relative_comparison_tolerance': 1e-12}


def _degradation(store, policy, spec, timestamp):
    start = timestamp - pd.Timedelta(hours=2 * policy.drift_window_hours)
    rows = store.db.execute('''SELECT origin,loss FROM predictions
        WHERE role='incumbent' AND task=? AND status='scored' AND origin>=? ORDER BY origin''',
        (spec.primary_task, start.isoformat())).fetchall()
    if not rows:
        return False, {'reason': 'no_mature_feedback'}
    frame = pd.DataFrame([dict(row) for row in rows])
    frame['origin'] = pd.to_datetime(frame.origin, utc=True)
    hourly = frame.groupby('origin').loss.mean()
    mid = timestamp - pd.Timedelta(hours=policy.drift_window_hours)
    older, newer = hourly[hourly.index < mid], hourly[hourly.index >= mid]
    enough = min(len(older), len(newer)) >= max(4, policy.drift_window_hours // 2)
    ratio = float(newer.mean() / max(float(older.mean()), 1e-12)) if enough else None
    return bool(enough and ratio >= policy.drift_ratio), {
        'reason': 'mature_loss_ratio', 'ratio': ratio, 'older_hours': len(older),
        'newer_hours': len(newer), 'threshold': policy.drift_ratio}


def _exposure(data, spec, directory, registry_path, source):
    registry = EvidenceRegistry(registry_path)
    intervals = []
    for symbol, rows in data.meta.groupby('symbol'):
        begin = utc(rows.available_at.min()) - pd.Timedelta(hours=spec.lookback + 49)
        interval = {'symbol': symbol, 'start': begin.isoformat(), 'end': utc(spec.dev_end).isoformat()}
        registry.consume(source=source, symbols=(symbol,), start=interval['start'], end=interval['end'],
                         run=str(directory), claim='evolution-' + spec.digest, role='development')
        intervals.append(interval)
    return {'source': source, 'role': 'development', 'intervals': intervals,
            'registry_path': str(registry.path), 'reservation': 'conservative_entire_declared_interval'}


def run_evolution(data: TaskDataset, initial_model, spec: ExperimentSpec, output_dir, *,
                  policy: EvolutionPolicy | None = None, device='cpu', resume=False,
                  as_of=None, max_ticks=None, memory_path='data/research_memory.sqlite',
                  registry_path='data/discovery_evidence.sqlite', evidence_source='binance_um_1h',
                  candidate_builder=None):
    """Run or resume a frozen sequential procedure on development data only.

    candidate_builder is an in-process testing/research extension; its identity
    must remain stable on resume. It receives only matured historical partitions.
    """
    policy = policy or EvolutionPolicy()
    if policy.mode != 'fixed' and (policy.shadow_hours - 1) // spec.bootstrap_block_hours < spec.min_blocks:
        raise ValueError('Shadow duration cannot provide the frozen minimum time blocks')
    directory = project_path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if data.tasks != spec.tasks or data.x.shape[1] != spec.lookback:
        raise ValueError('Evolution data disagrees with frozen protocol')
    if not set(data.meta.symbol) <= set(spec.symbols) or not len(data.meta):
        raise ValueError('Evolution requires data in the frozen universe')
    ordered = data.meta.sort_values(['available_at', 'symbol'], kind='stable').reset_index(drop=True)
    if not data.meta.equals(ordered):
        raise ValueError('Evolution samples must be chronologically ordered')
    start, end = utc(spec.train_end), utc(spec.dev_end)
    timestamp_limit = min(utc(as_of) if as_of is not None else end,
                          utc(data.meta.available_at.max()), end)
    if max_ticks is not None and (type(max_ticks) is not int or max_ticks < 1):
        raise ValueError('max_ticks must be positive')
    # Never expose confirmation suffixes to diagnostics, search, or persistence.
    data = data.subset((data.meta.available_at < end).to_numpy())
    source_files = _current_source_files()
    context_policy = data.meta.attrs.get('context_policy')
    context_rules = ({**{key: value for key, value in context_policy.items() if key != 'sources'},
                      'source_names': sorted(context_policy.get('sources', {}))} if context_policy else None)
    identity = {'format': 'discovery-evolution-v1', 'protocol': asdict(spec), 'policy': asdict(policy),
                'schema': data.schema, 'source_files': source_files,
                'source_archive': {'path': 'source', 'digest': fingerprint(source_files)},
                'context_policy': context_rules,
                'availability_mode': data.meta.attrs.get('availability_mode'),
                'feed_identity': data.meta.attrs.get('feed_identity'),
                'candidate_builder': (candidate_builder.__module__ + '.' + candidate_builder.__qualname__)
                if candidate_builder else 'builtin', 'memory_path': str(project_path(memory_path)),
                'scope': 'development_sequential_research', 'initial_digest': initial_model.digest
                if initial_model is not None else None}
    with FileLock(str(directory / '.evolution.lock'), timeout=0):
        store = EvolutionStore(directory)
        try:
            state = store.state()
            if state is None:
                if resume:
                    raise ValueError('No committed evolution state to resume')
                if initial_model is None:
                    raise ValueError('An initial model is required for a new run')
                _causal_model(initial_model, data, start)
                archived = snapshot_source(directory / 'source')
                if archived['files'] != source_files or archived['digest'] != identity['source_archive']['digest']:
                    raise ValueError('Evolution source changed during initial snapshot')
                _verify_source_archive(directory, identity)
                exposure = _exposure(data, spec, directory, registry_path, evidence_source)
                manifest = {**identity, 'evidence_exposure': exposure}
                immutable_json(directory / 'manifest.json', {'manifest': manifest, 'sha256': fingerprint(manifest)})
                with store.db:
                    incumbent = store.save_model(initial_model, start)
                    state = {'identity': identity, 'incumbent': incumbent, 'last_timestamp': None,
                             'visible_hash': None, 'next_research': start.isoformat(), 'cycles': 0,
                             'active_cycle': None, 'ticks': 0}
                    store.save_state(state)
                    store.event('started', start, {'incumbent': incumbent})
            else:
                if not resume:
                    raise ValueError('Existing evolution run requires explicit resume')
                if state['identity'].get('source_archive') is None:
                    store.verify(compact=True)
                    raise ValueError('Legacy hash-only evolution run supports historical inspection; '
                                     'start a new archived-source run to resume research')
                identity['initial_digest'] = state['identity']['initial_digest'] if initial_model is None else initial_model.digest
                if fingerprint(identity) != fingerprint(state['identity']):
                    raise ValueError('Frozen evolution identity changed on resume')
                manifest = json.loads((directory / 'manifest.json').read_text('utf-8'))
                if manifest['sha256'] != fingerprint(manifest['manifest']):
                    raise ValueError('Evolution manifest changed')
                store.verify(compact=True)
                if (state['last_timestamp'] is not None
                        and _visible_hash(data, state['last_timestamp']) != state['visible_hash']):
                    raise ValueError('Previously observed data or mature labels changed')
                for row in store.db.execute('SELECT id FROM models').fetchall():
                    store.load_model(row['id'])
                if state['active_cycle'] is not None:
                    cycle = store.cycle(state['active_cycle'])
                    if cycle['status'] == 'searching':
                        with store.db:
                            cycle.update(status='interrupted_slot_consumed', reason='resume_after_uncommitted_search')
                            store.put_cycle(cycle)
                            state['active_cycle'] = None
                            store.save_state(state)
                            store.event('interrupted_search', state['last_timestamp'] or start, cycle)
            models = {}

            def load(identifier):
                if identifier not in models:
                    models[identifier] = store.load_model(identifier)
                return models[identifier]

            by_id = {sample: index for index, sample in enumerate(data.meta.sample_id)}
            origins = pd.DatetimeIndex(data.meta.available_at.unique()).sort_values()
            origins = origins[(origins >= start) & (origins < end) & (origins <= timestamp_limit)]
            if state['last_timestamp'] is not None:
                origins = origins[origins > utc(state['last_timestamp'])]
            processed = 0
            for timestamp in origins:
                if max_ticks is not None and processed >= max_ticks:
                    break
                cycle = store.cycle(state['active_cycle']) if state['active_cycle'] is not None else None
                with store.db:
                    _settle(store, data, timestamp, by_id)
                    if cycle and cycle['status'] == 'shadow' and timestamp >= utc(cycle['decision_at']):
                        gate = _gate(store, cycle, data, spec, policy)
                        promote = gate.get('operational_development_improvement_supported', False)
                        if promote:
                            state['incumbent'] = cycle['candidate']
                        cycle.update(status='promoted' if promote else 'retained', gate=gate,
                                     decided_at=timestamp.isoformat())
                        store.put_cycle(cycle)
                        state['active_cycle'] = None
                        store.event('promotion_decision', timestamp, cycle)
                        cycle = None
                    current = data.subset((data.meta.available_at == timestamp).to_numpy())
                    incumbent = load(state['incumbent'])
                    _causal_model(incumbent, data, timestamp)
                    _record_predictions(store, incumbent, current, 'incumbent')
                    store.event('prediction', timestamp, {'incumbent': incumbent.digest,
                                                         'sample_ids': current.meta.sample_id.tolist()})
                    if cycle and cycle['status'] == 'shadow' and timestamp < utc(cycle['shadow_end']):
                        try:
                            # Savepoint prevents a malformed candidate from leaving half a paired forecast.
                            store.db.execute('SAVEPOINT shadow_prediction')
                            _record_predictions(store, load(cycle['candidate']), current, 'shadow', cycle['id'])
                            store.db.execute('RELEASE shadow_prediction')
                        except (ValueError, RuntimeError, FloatingPointError) as error:
                            store.db.execute('ROLLBACK TO shadow_prediction')
                            store.db.execute('RELEASE shadow_prediction')
                            cycle.update(status='invalid_candidate', reason=f'{type(error).__name__}: {error}')
                            store.put_cycle(cycle)
                            state['active_cycle'] = None
                            store.event('candidate_failure', timestamp, cycle)
                    state['last_timestamp'] = timestamp.isoformat()
                    state['visible_hash'] = _visible_hash(data, timestamp)
                    state['ticks'] += 1
                    store.save_state(state)
                processed += 1
                if (policy.mode == 'fixed' or state['active_cycle'] is not None
                        or state['cycles'] >= policy.max_cycles or timestamp < utc(state['next_research'])):
                    continue
                trigger, details = (True, {'reason': 'scheduled'}) if policy.trigger == 'scheduled' else _degradation(
                    store, policy, spec, timestamp)
                with store.db:
                    state['next_research'] = (timestamp + pd.Timedelta(hours=policy.update_every_hours)).isoformat()
                    store.event('research_trigger', timestamp, {'triggered': trigger, **details})
                    if trigger:
                        state['cycles'] += 1
                        cycle = {'id': state['cycles'], 'status': 'searching', 'cutoff': timestamp.isoformat(),
                                 'reference': state['incumbent'], 'trigger': details}
                        store.put_cycle(cycle)
                        state['active_cycle'] = cycle['id']
                    store.save_state(state)
                if not trigger:
                    continue
                research_started = time.perf_counter()
                training, research = None, None
                try:
                    train, development, training = _training_parts(data, timestamp)
                    candidate, research = _research_candidate(
                        train, development, load(state['incumbent']), spec, policy,
                        directory / 'research' / f"cycle-{cycle['id']:04d}", timestamp,
                        device, memory_path, candidate_builder)
                    _causal_model(candidate, data, timestamp)
                    research = dict(research)
                    if 'elapsed_seconds' in research:
                        research['candidate_search_elapsed_seconds'] = research['elapsed_seconds']
                    research.update(elapsed_seconds=time.perf_counter() - research_started,
                                    resource_scope='observed in-process wall time from partition preparation '
                                        'through candidate validation; no FLOPs or API monetary cost measurement')
                    if candidate.digest == state['incumbent']:
                        with store.db:
                            cycle.update(status='unchanged_retained', candidate=state['incumbent'],
                                         training=training, research=research,
                                         reason='development_selection_retained_current_model')
                            store.put_cycle(cycle)
                            state['active_cycle'] = None
                            store.save_state(state)
                            store.event('unchanged_retained', timestamp, cycle)
                        continue
                    with store.db:
                        candidate_id = store.save_model(candidate, timestamp)
                        cycle.update(status='shadow', candidate=candidate_id, training=training,
                                     research=research, shadow_start=timestamp.isoformat(),
                                     shadow_end=(timestamp + pd.Timedelta(hours=policy.shadow_hours)).isoformat(),
                                     decision_at=(timestamp + pd.Timedelta(hours=policy.shadow_hours +
                                         max(t.horizon_hours for t in spec.tasks) + 2)).isoformat())
                        store.put_cycle(cycle)
                        store.save_state(state)
                        store.event('candidate_started', timestamp, cycle)
                except (ValueError, RuntimeError, FloatingPointError, TimeoutError, OSError) as error:
                    research = dict(research or {})
                    if 'elapsed_seconds' in research:
                        research.setdefault('candidate_search_elapsed_seconds', research['elapsed_seconds'])
                    research.update(elapsed_seconds=time.perf_counter() - research_started,
                                    resource_scope='observed in-process wall time before caught failure; '
                                        'partial work recorded only as elapsed wall time, '
                                        'no FLOPs or API monetary cost measurement')
                    with store.db:
                        cycle.update(status='failed', reason=f'{type(error).__name__}: {error}',
                                     research=research)
                        if training is not None:
                            cycle['training'] = training
                        store.put_cycle(cycle)
                        state['active_cycle'] = None
                        store.save_state(state)
                        store.event('research_failure', timestamp, cycle)
            exhausted = len(origins) == processed
            if exhausted and (state['last_timestamp'] is None or timestamp_limit > utc(state['last_timestamp'])):
                with store.db:
                    _settle(store, data, timestamp_limit, by_id)
                    cycle = store.cycle(state['active_cycle']) if state['active_cycle'] is not None else None
                    if cycle and cycle['status'] == 'shadow' and timestamp_limit >= utc(cycle['decision_at']):
                        gate = _gate(store, cycle, data, spec, policy)
                        promote = gate.get('operational_development_improvement_supported', False)
                        if promote:
                            state['incumbent'] = cycle['candidate']
                        cycle.update(status='promoted' if promote else 'retained', gate=gate,
                                     decided_at=timestamp_limit.isoformat())
                        store.put_cycle(cycle)
                        state['active_cycle'] = None
                        store.event('promotion_decision', timestamp_limit, cycle)
                    state['last_timestamp'] = timestamp_limit.isoformat()
                    state['visible_hash'] = _visible_hash(data, timestamp_limit)
                    store.save_state(state)
            return _summary(store, spec, policy)
        finally:
            store.close()


def _memory_context_hash(data, spec):
    """Bind reusable action facts to stable input and knowability semantics.

    Rolling evidence cutoffs and context source content hashes are excluded:
    causal appends may extend one stream without changing what its inputs mean.
    Legacy unbound facts are not silently assigned to this identity.
    """
    context = data.meta.attrs.get('context_policy')
    rules = ({**{key: value for key, value in context.items() if key != 'sources'},
              'source_names': sorted(context.get('sources', {}))} if context else None)
    return fingerprint({'schema': data.schema, 'symbols': sorted(spec.symbols),
                        'research_semantics': 'causal-operator-program-v2-robust-selection',
                        'selection_policy': asdict(spec.selection),
                        'checkpoint_selection_task': spec.training.selection_task,
                        'primary_task': spec.primary_task,
                        'input_implementation': {name: file_hash(project_path('factorlab/' + name))
                            for name in ('features.py', 'discovery/tasks.py', 'discovery/context_data.py',
                                         'discovery/operator_program.py', 'discovery/learning.py',
                                         'discovery/scoring.py', 'discovery/robust_selection.py',
                                         'discovery/benchmarks.py')},
                        'lookback': spec.lookback, 'context_rules': rules,
                        'availability_delay_seconds': spec.availability_delay_seconds,
                        'availability_mode': data.meta.attrs.get('availability_mode'),
                        'feed_identity': data.meta.attrs.get('feed_identity')})


def _research_candidate(train, development, incumbent, spec, policy, directory, timestamp,
                        device, memory_path, candidate_builder):
    directory.mkdir(parents=True, exist_ok=True)
    # New research packet binds a current chronological split and current cutoff.
    current_spec = replace(spec, train_end=utc(development.meta.available_at.min()).isoformat(),
                           dev_end=timestamp.isoformat())
    if candidate_builder is not None:
        return candidate_builder(train, development, incumbent, current_spec, policy, directory)
    architecture = _architecture(incumbent)
    if policy.mode == 'refit':
        model = _fit_group(train, development, architecture, spec.training, spec.training_seeds,
                           device, incumbent if policy.inherit_weights else None)
        return model, {'method': 'same_structure_refit', 'inherited': policy.inherit_weights}
    from .population import search_population
    from .research_memory import ResearchMemory, to_development_history
    memory = ResearchMemory(memory_path)
    task = asdict(next(t for t in spec.tasks if t.name == spec.primary_task))
    context_hash = _memory_context_hash(train, current_spec)
    prior = to_development_history(memory.retrieve(
        task=task, evidence_cutoff=timestamp.isoformat(), limit=policy.memory_limit,
        context_hash=context_hash), current_spec)
    parents = [architecture] if architecture else [RepresentationSpec((ModuleSpec(scale=min(12, spec.lookback)),))]
    revived = []
    if policy.program_search:
        from .operator_search import program_contract
        program_contract(parents[0], spec.lookback, train.features)
        room = min(policy.revived_parents, policy.population_size - 1, spec.candidates - 1)
        # Prompt memory is chronological; revival still prefers the newest facts.
        for fact in reversed(prior):
            if len(revived) >= room:
                break
            if fact.get('status') != 'ok' or not fact.get('architecture'):
                continue
            saved = RepresentationSpec.from_dict(fact['architecture'])
            if not any(m.kind == 'program' for m in saved.modules) or saved.digest in {p.digest for p in parents}:
                continue
            parents.append(saved)
            revived.append(saved.digest)
    # Population owns exact current-data re-scoring, not reuse of old loss values.
    initial_models = {architecture.digest: incumbent} if architecture else None

    def save_candidate(model, record):
        folder = directory / f"candidate-{record.get('iteration', 0)}-{model.digest[:12]}"
        model.save(folder)
        programs = {}
        from .operator_program import Program
        for module in RepresentationSpec.from_dict(record['architecture']).modules:
            if module.kind != 'program':
                continue
            program = Program.from_dict(module.program)
            programs[module.name] = {'program': program.to_dict(), 'digest': program.digest,
                'source_listing': program.source, 'input_semantics': 'training_standardized_features',
                'seed_parameters': [member.program_parameters.get(module.name, {}) for member in _members(model)]}
        if programs:
            immutable_json(folder / 'operators.json', programs)
            library = directory.parent.parent / 'operator-library'
            library.mkdir(exist_ok=True)
            for value in programs.values():
                path = library / (value['digest'] + '.json')
                definition = {'program': value['program'], 'digest': value['digest'],
                              'source_listing': value['source_listing'], 'context_hash': context_hash}
                if not path.exists():
                    immutable_json(path, definition)
                elif json.loads(path.read_text('utf-8')) != definition:
                    raise ValueError('Operator library identity changed')

    def remember(record):
        proposal = record.get('proposal', {})
        if not proposal:
            return
        from .search import _clean_history
        clean = _clean_history([record], current_spec)[0]
        from .robust_selection import ROBUST_METRIC_FIELDS
        metrics = dict(clean.get('metrics', {}))
        metrics.update({key: clean[key] for key in ROBUST_METRIC_FIELDS if key in clean})
        memory.record_action_result(task=task, evidence_cutoff=timestamp.isoformat(),
            protocol_hash=current_spec.digest, parent_hash=record.get('parent_hash', proposal.get('parent_hash', 'initial')),
            candidate_id=record.get('candidate_id', fingerprint(record)),
            action={key: proposal[key] for key in ('operation', 'arguments')},
            structure=record.get('architecture'), metrics=metrics,
            elapsed_seconds=record.get('elapsed_seconds', 0), status=record.get('status', 'failed'),
            failure=clean.get('failure_code') if record.get('error') else None,
            context_hash=context_hash,
            diagnostics=clean.get('diagnostics'),
            event_id=fingerprint({'directory': str(directory), 'record': record}))

    from .validation_artifacts import persist_forward_fold
    model, result = search_population(train, development, parents, current_spec,
        mode=policy.search_mode, device=device, exchange=directory / 'exchange', history=prior,
        initial_models=initial_models, inherit_weights=policy.inherit_weights,
        population_size=policy.population_size, fit_candidate=_fit_group,
        on_candidate=save_candidate, on_record=remember, program_search=policy.program_search,
        on_fold=lambda model, record, fold, validation: persist_forward_fold(
            directory, model, record, fold, validation))
    result['revived_architectures'] = revived
    result['program_search'] = policy.program_search
    result = _store_research(directory.parent.parent, result)
    immutable_json(directory / 'result.json', result)
    if model is None:
        raise ValueError('Population search produced no valid candidate')
    return model, result


def _summary(store, spec, policy, *, verified=None):
    state = store.state()
    cycles = []
    for row in store.db.execute('SELECT value FROM cycles ORDER BY id'):
        cycle = json.loads(row['value'])
        research = cycle.get('research')
        if isinstance(research, dict) and '_research_ref' not in research and _large_research(research):
            cycle['research'] = _research_summary(research)
            cycle['research']['summary_scope'] = 'legacy_inline_research_summary_v1'
        cycles.append(cycle)
    tasks = {}
    for task in spec.tasks:
        row = store.db.execute('''SELECT COUNT(*) AS n,AVG(loss) AS loss FROM predictions
            WHERE role='incumbent' AND task=? AND status='scored' ''', (task.name,)).fetchone()
        pending = store.db.execute('''SELECT COUNT(*) AS n FROM predictions
            WHERE role='incumbent' AND task=? AND settled=0''', (task.name,)).fetchone()['n']
        tasks[task.name] = {'support': row['n'], 'loss': row['loss'], 'pending': pending}
    verified = verified if verified is not None else store.verify(compact=True)
    return {'format': 'discovery-evolution-v1', 'scope': 'development_sequential_research',
            'mode': policy.mode, 'ticks': state['ticks'], 'last_timestamp': state['last_timestamp'],
            'incumbent': state['incumbent'], 'cycles': cycles, 'tasks': tasks,
            'promotions': sum(c['status'] == 'promoted' for c in cycles),
            'event_head': verified['event_head'],
            'pending_cycle': state['active_cycle'],
            'source_evidence': _verify_source_archive(store.directory, state['identity']),
            'can_resume': state['identity'].get('source_archive') is not None}


def inspect_evolution(directory):
    store = EvolutionStore(directory)
    try:
        state = store.state()
        if state is None:
            raise ValueError('No evolution state')
        spec = ExperimentSpec.from_dict(state['identity']['protocol'])
        policy = EvolutionPolicy(**state['identity']['policy'])
        verified = store.verify(compact=True)
        for row in store.db.execute('SELECT id FROM models').fetchall():
            store.load_model(row['id'])
        return _summary(store, spec, policy, verified=verified)
    finally:
        store.close()
