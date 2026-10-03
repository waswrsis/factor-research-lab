"""Append-only local quality dossiers for completed research runs.

The catalog pins records and artifact hashes. Registration establishes local
integrity/provenance, not profitability, novelty or external authenticity.
Generator identity binds the whole frozen group, never individual latent axes.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from factorlab.protocol import fingerprint, project_path

from .contracts import ExperimentSpec
from .evidence import ExperimentJournal, file_hash, immutable_json, verify_bundle

CATALOG_VERSION = 'discovery-quality-catalog-v1'


def _json(path):
    return json.loads(Path(path).read_text('utf-8'))


def _inside(root, relative):
    if Path(relative).is_absolute():
        raise ValueError('Artifact reference must be relative to its run')
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Artifact reference escapes its run')
    return project_path(path)


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Invalid SHA256 identity')
    return value


def _read_run(directory):
    directory = project_path(directory)
    result = _json(directory / 'results.json')
    claim = _json(directory / 'frozen_claim.json')
    protocol_record = _json(directory / 'protocol.json')
    spec = ExperimentSpec.from_dict(protocol_record['protocol'])
    journal = ExperimentJournal(directory / 'journal.jsonl').read()
    if (not journal or journal[-1]['event'] != 'completed'
        or any(record.get('sequence') != i for i, record in enumerate(journal))
        or sum(record['event'] == 'completed' for record in journal) != 1):
        raise ValueError('Only a completed hash-chain run can be registered')
    if journal[-1]['value'].get('result_hash') != file_hash(directory / 'results.json'):
        raise ValueError('Completed journal does not bind current results')
    if result.get('status') not in {'development_complete', 'confirmation_complete'}:
        raise ValueError('Run is not a complete research result')
    if (protocol_record['sha256'] != spec.digest or claim['protocol_hash'] != spec.digest
        or result.get('protocol_hash') != spec.digest
        or fingerprint(claim['protocol']) != fingerprint(protocol_record['protocol'])
        or result.get('claim_hash') != claim['digest']):
        raise ValueError('Protocol/claim/result identities disagree')
    if (result.get('primary_task') != spec.primary_task
        or claim.get('primary_task') != spec.primary_task
        or project_path(result['run_directory']) != directory):
        raise ValueError('Run scope or directory identity differs')
    bundle = directory / 'bundle'
    verify_bundle(bundle, claim)
    required_comparisons = {'system_vs_baseline', 'representation_addition',
                            'matched_dimension_control'}
    if not required_comparisons <= set(claim['comparisons']):
        raise ValueError('Whole-system and representation comparisons must be frozen')
    selection = _json(bundle / 'selection.json')
    generator = _json(bundle / 'candidate' / 'group.json')
    baseline = _json(bundle / 'baseline' / 'group.json')
    probe = _json(bundle / 'probe_contract.json')
    identity = _digest(selection['candidate_digest'])
    if (result.get('selection') != selection or generator['digest'] != identity
        or probe['generator_digest'] != identity
        or baseline['digest'] != selection['baseline_digest']):
        raise ValueError('Selected whole-generator identity differs across bundle records')
    for role, group in [('candidate', generator), ('baseline', baseline)]:
        if type(group['members']) is not int or group['members'] < 1:
            raise ValueError('Fitted seed group is incomplete')
        for member in range(group['members']):
            if not (bundle / role / f'seed-{member}').is_file():
                raise ValueError('Fitted seed artifact missing')
    source_manifest = _json(directory / 'source' / 'manifest.json')
    if fingerprint(source_manifest) != claim['source_hash']:
        raise ValueError('Frozen source manifest differs from claim')
    source_files = {str(p.relative_to(directory / 'source')).replace('\\', '/')
                    for p in (directory / 'source').rglob('*') if p.is_file()
                    and str(p.relative_to(directory / 'source')).replace('\\', '/')
                    not in {'manifest.json', 'manifest.json.lock'}}
    if source_files != set(source_manifest):
        raise ValueError('Frozen source snapshot membership changed')
    for relative, expected in source_manifest.items():
        if file_hash(_inside(directory / 'source', relative)) != expected:
            raise ValueError('Frozen source snapshot changed')
    # Required files and all frozen bundle/source members are pinned. Auxiliary
    # run files are not needed to interpret the copied dossier unless referenced.
    paths = [directory / name for name in
             ('results.json', 'frozen_claim.json', 'protocol.json', 'journal.jsonl')]
    paths += [p for p in bundle.rglob('*') if p.is_file() and p.suffix != '.lock']
    paths += [directory / 'source' / 'manifest.json']
    paths += [_inside(directory / 'source', p) for p in source_manifest]
    confirmation = result.get('confirmation')
    if result['status'] == 'confirmation_complete':
        events = [record['event'] for record in journal]
        if (not isinstance(confirmation, dict) or events.count('confirmation_consumed') != 1
            or events.count('confirmation_result') != 1
            or events.index('confirmation_consumed') >= events.index('confirmation_result')):
            raise ValueError('Confirmation lacks its consumed evidence record')
        consumed = journal[events.index('confirmation_consumed')]['value']
        reported = journal[events.index('confirmation_result')]['value']
        if consumed.get('claim_hash') != claim['digest'] or reported != confirmation:
            raise ValueError('Confirmation journal and result disagree')
        if not required_comparisons <= set(confirmation['comparisons']):
            raise ValueError('Confirmation comparisons are incomplete')
        compared = confirmation['comparisons']
        supported_system = compared['system_vs_baseline']['status'] == 'supported'
        supported_representation = all(compared[k]['status'] == 'supported'
            for k in ('representation_addition', 'matched_dimension_control'))
        if (confirmation.get('accepted_system') != supported_system
            or confirmation.get('supported_representation') != supported_representation):
            raise ValueError('Conclusions disagree with their comparison evidence')
        for name in ('candidate_predictions', 'baseline_predictions'):
            artifact = confirmation[name]
            path = project_path(artifact['path'])
            if not path.is_relative_to(directory) or file_hash(path) != artifact['sha256']:
                raise ValueError('Confirmation prediction artifact identity differs')
            paths.append(path)
    elif confirmation is not None:
        raise ValueError('Development-only status cannot contain confirmation conclusions')
    artifacts = {str(p.relative_to(directory)).replace('\\', '/'): file_hash(p) for p in paths}
    return result, claim, artifacts


def _conclusions(result):
    confirmation = result.get('confirmation')
    if confirmation is None:
        return {'status': 'unconfirmed', 'comparison': None}, {
            'status': 'unconfirmed', 'comparisons': {},
            'scope': 'frozen generator group, not individual latent dimensions'}
    comparisons = confirmation['comparisons']
    system = comparisons['system_vs_baseline']
    representation = {k: comparisons[k] for k in
                       ('representation_addition', 'matched_dimension_control')}
    states = [v['status'] for v in representation.values()]
    if all(state == 'supported' for state in states):
        status = 'supported'
    elif 'flow_invalid' in states:
        status = 'flow_invalid'
    elif 'target_not_met' in states:
        status = 'target_not_met'
    else:
        status = 'insufficient_evidence'
    return {'status': system['status'], 'comparison': system}, {
        'status': status, 'comparisons': representation,
        'scope': 'frozen generator group, not individual latent dimensions'}


class QualityCatalog:
    """One SQLite file. Entries are append-only; verification rehashes artifacts."""
    def __init__(self, path: str | Path = 'data/discovery_catalog.sqlite'):
        self.path = project_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript('''
                CREATE TABLE IF NOT EXISTS entries (
                    entry_id TEXT PRIMARY KEY,
                    run_directory TEXT NOT NULL UNIQUE,
                    generator_identity TEXT NOT NULL,
                    primary_task TEXT NOT NULL,
                    evidence_state TEXT NOT NULL,
                    registered_at TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    record_hash TEXT NOT NULL);
                CREATE TRIGGER IF NOT EXISTS entries_no_update BEFORE UPDATE ON entries
                    BEGIN SELECT RAISE(ABORT, 'Catalog entries are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS entries_no_delete BEFORE DELETE ON entries
                    BEGIN SELECT RAISE(ABORT, 'Catalog entries are immutable'); END;
            ''')

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA busy_timeout=30000')
        return connection

    def register(self, run_directory: str | Path) -> dict:
        directory = project_path(run_directory)
        try:
            result, claim, artifacts = _read_run(directory)
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError('Complete run artifacts missing or invalid') from exc
        system, representation = _conclusions(result)
        protocol = claim['protocol']
        entry_id = fingerprint({'run_directory': str(directory), 'claim_hash': claim['digest'],
                                'result_hash': artifacts['results.json']})
        costs = {'total_elapsed_seconds': result['elapsed_seconds'],
                 'training_budget': protocol['training'],
                 'search_budget_seconds': protocol.get('search_seconds'),
                 'training_seeds': protocol.get('training_seeds', []),
                 'arms': {name: {k: arm[k] for k in (
                     'elapsed_seconds', 'parameter_count', 'attempts', 'budget_seconds',
                     'budget_comparison') if k in arm}
                     for name, arm in result.get('arms', {}).items()},
                 'cost_scope': 'reported time and parameters; no inferred FLOPs or dollar cost'}
        record = {'version': CATALOG_VERSION, 'entry_id': entry_id,
                  'run_directory': str(directory), 'registered_at': datetime.now(UTC).isoformat(),
                  'bundle_path': str(directory / 'bundle'), 'bundle_hash': fingerprint(claim['bundle_files']),
                  'claim_hash': claim['digest'], 'result_hash': artifacts['results.json'],
                  'generator_identity': result['selection']['candidate_digest'],
                  'generator_identity_scope': 'whole frozen generator group',
                  'primary_task': result['primary_task'],
                  'evidence_state': 'confirmation_complete' if result.get('confirmation') is not None
                                    else 'development_only',
                  'task_scope': {'tasks': protocol['tasks'], 'primary_task': protocol['primary_task'],
                      'symbols': protocol['symbols'], 'scope': claim['scope'],
                      'development_start': protocol['train_end'], 'development_end': protocol['dev_end'],
                      'confirmation_start': claim['confirmation_start'],
                      'confirmation_end': claim['confirmation_end'],
                      'coverage': result.get('confirmation', {}).get('candidate_metrics', {})
                                  if result.get('confirmation') is not None else {},
                      'limitations': result.get('limitations', [])},
                  'system_conclusion': system, 'representation_conclusion': representation,
                  'cost': costs, 'frozen_claim': claim, 'result': result, 'artifacts': artifacts}
        content = json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True)
        with self._connect() as connection:
            try:
                connection.execute('INSERT INTO entries VALUES (?,?,?,?,?,?,?,?)',
                    (entry_id, str(directory), record['generator_identity'], record['primary_task'],
                     record['evidence_state'], record['registered_at'], content, fingerprint(record)))
            except sqlite3.IntegrityError as exc:
                raise ValueError('Run already registered; existing entry cannot be overwritten') from exc
        return record

    @staticmethod
    def _record(row):
        record = json.loads(row['record_json'])
        if fingerprint(record) != row['record_hash']:
            raise ValueError('Catalog record integrity failure')
        for key in ('entry_id', 'run_directory', 'generator_identity', 'primary_task',
                    'evidence_state', 'registered_at'):
            if record[key] != row[key]:
                raise ValueError('Catalog index differs from its record')
        return record

    def get(self, entry_id: str) -> dict:
        with self._connect() as connection:
            row = connection.execute('SELECT * FROM entries WHERE entry_id=?', (entry_id,)).fetchone()
        if row is None:
            raise KeyError(entry_id)
        return self._record(row)

    def list(self, *, task: str | None = None, state: str | None = None) -> list[dict]:
        clauses, params = [], []
        for key, value in (('primary_task', task), ('evidence_state', state)):
            if value is not None:
                clauses.append(f'{key}=?')
                params.append(value)
        query = 'SELECT * FROM entries'
        if clauses:
            query += ' WHERE ' + ' AND '.join(clauses)
        query += ' ORDER BY registered_at, entry_id'
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._record(row) for row in rows]

    def verify(self, entry_id: str) -> dict:
        record = self.get(entry_id)
        directory = project_path(record['run_directory'])
        try:
            for relative, expected in record['artifacts'].items():
                if file_hash(_inside(directory, relative)) != expected:
                    raise ValueError(f'Catalog artifact changed: {relative}')
            result, claim, artifacts = _read_run(directory)
            if (result != record['result'] or claim != record['frozen_claim']
                or artifacts != record['artifacts']):
                raise ValueError('Run differs from its pinned catalog record')
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError('Catalog artifact missing or invalid') from exc
        return {'entry_id': entry_id, 'valid': True, 'artifact_count': len(record['artifacts']),
                'generator_identity': record['generator_identity']}

    def export(self, path: str | Path) -> Path:
        records = self.list()
        for record in records:
            self.verify(record['entry_id'])
        return immutable_json(path, {'version': CATALOG_VERSION, 'entries': records,
                                     'sha256': fingerprint(records)})
