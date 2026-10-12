"""Durable, retrievable action/result facts with development-only evidence boundaries.

This is research evidence, not a model's free-form diary. Unknown fields, confirmation
records, and future-cutoff facts are rejected rather than becoming prompt context.
"""
from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime

from factorlab.protocol import fingerprint, project_path, utc

from .contracts import RepresentationSpec, TaskSpec
from .robust_selection import (
    ROBUST_METRIC_FIELDS,
    TASK_METRIC_FIELDS,
    TASK_METRIC_STRINGS,
    clean_robust_metrics,
)
from .search import OPERATIONS

METRIC_FIELDS = TASK_METRIC_FIELDS | ROBUST_METRIC_FIELDS | {'primary_loss', 'parameter_count'}
STATUSES = frozenset({'ok', 'failed', 'proposed', 'rejected', 'timeout', 'duplicate'})
FORBIDDEN = frozenset({'confirmation', 'confirmation_loss', 'calibration', 'heldout', 'test',
                       'confirmation_metrics', 'confirmation_result', 'test_loss'})


def reject_forbidden(value):
    if isinstance(value, dict):
        if any(str(k).lower() in FORBIDDEN for k in value):
            raise ValueError('Non-development evidence cannot enter research memory')
        for item in value.values():
            reject_forbidden(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            reject_forbidden(item)


def task_identity(task: dict | str | TaskSpec) -> tuple[str, dict | str]:
    if isinstance(task, TaskSpec):
        task = asdict(task)
    if isinstance(task, dict):
        fields = dict(task)
        fields['event_bins'] = tuple(fields.get('event_bins', (4, 12)))
        task = json.loads(json.dumps(asdict(TaskSpec(**fields))))
    elif not isinstance(task, str) or not task.isidentifier():
        raise ValueError('Memory requires a declared task')
    return fingerprint(task), task


def clean_metrics(metrics: dict, task_names: set[str] | None = None) -> dict:
    reject_forbidden(metrics)
    if not isinstance(metrics, dict):
        raise ValueError('Metrics must be a dictionary')  # noqa: TRY004 - evidence rejection API
    result = {}
    for name, value in metrics.items():
        if name in ROBUST_METRIC_FIELDS:
            result.update(clean_robust_metrics({name: value}))
            continue
        if isinstance(value, dict):
            if task_names is not None and name not in task_names:
                raise ValueError('Metric contains an undeclared task')
            if not name.isidentifier():
                raise ValueError('Invalid metric task')
            result[name] = clean_metrics(value)
        else:
            if name not in METRIC_FIELDS or isinstance(value, (list, tuple)):
                raise ValueError('Metric field is not in the development whitelist')
            if value is not None and not isinstance(value, (str, bool, int, float)):
                raise ValueError('Metric summaries must be scalar')
            if name in TASK_METRIC_STRINGS:
                if not isinstance(value, str) or value not in TASK_METRIC_STRINGS[name]:
                    raise ValueError('Only bounded metric scope/kind strings are allowed')
            elif isinstance(value, str) and (name != 'status' or value not in {
                'ok', 'insufficient', 'insufficient_support', 'no_support', 'failed',
                'unsupported', 'supported', 'unobserved', 'not_predictive', 'descriptive',
                'empty', 'no_observations', 'unavailable', 'insufficient_evidence', 'not_applicable'
            }):
                raise ValueError('Only bounded metric status strings are allowed')
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError('Metrics must be finite')
            result[name] = value
    return result


def clean_action(action: dict) -> dict:
    reject_forbidden(action)
    if not isinstance(action, dict) or set(action) != {'operation', 'arguments'}:
        raise ValueError('Action requires operation and arguments')
    operation, arguments = action['operation'], action['arguments']
    fields = {'add': {'module'}, 'drop': {'name'}, 'merge': {'names', 'name'},
              'dimension': {'name', 'dim'}, 'replace': {'name', 'kind'},
              'combine': {'combination'}, 'learning_rate': {'value'},
              'features': {'name', 'features'}, 'scale': {'name', 'scale'},
              'width': {'name', 'width'}, 'depth': {'name', 'depth'},
              'write_program': {'name', 'program'}}
    if operation not in OPERATIONS or operation not in fields or not isinstance(arguments, dict):
        raise ValueError('Unregistered memory action')
    if set(arguments) != fields[operation]:
        raise ValueError('Action arguments differ from registered schema')
    # Arguments are structured facts, never arbitrary source/code/path strings.
    for name, value in arguments.items():
        if name in {'name', 'kind', 'combination'} and (
            not isinstance(value, str) or not value.isidentifier()
        ):
            raise ValueError('Action string must be an identifier')
        if name == 'names' and (not isinstance(value, list) or not value
                               or any(not isinstance(v, str) or not v.isidentifier() for v in value)):
            raise ValueError('Merge names must be identifiers')
        if name == 'module':
            RepresentationSpec.from_dict({'modules': [value]})
        if name == 'program':
            from .operator_program import Program
            Program.from_dict(value)
        if name == 'dim' and (type(value) is not int or not 1 <= value <= 64):
            raise ValueError('Invalid action dimension')
        if name in {'scale', 'width', 'depth'} and (
            type(value) is not int or not 1 <= value <= {'scale': 672, 'width': 128, 'depth': 3}[name]
        ):
            raise ValueError('Invalid bounded module action')
        if name == 'features' and (not isinstance(value, list) or len(set(value)) != len(value)
                                   or any(not isinstance(v, str) or not v.isidentifier() for v in value)):
            raise ValueError('Feature arguments must be unique identifiers')
        if name == 'value' and (type(value) not in (int, float)
                               or not math.isfinite(value) or not 1e-5 <= value <= 0.01):
            raise ValueError('Invalid action learning rate')
    return json.loads(json.dumps(action, allow_nan=False))


def _identity(value: str, name: str, optional=False):
    if optional and value is None:
        return None
    if not isinstance(value, str) or not value or len(value) > 160:
        raise ValueError(f'{name} must be a bounded identity')
    if any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_:-.' for ch in value):
        raise ValueError(f'{name} contains unsupported identity characters')
    return value


class ResearchMemory:
    def __init__(self, path):
        self.path = project_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute('''CREATE TABLE IF NOT EXISTS action_results (
                event_id TEXT PRIMARY KEY, task_hash TEXT NOT NULL, cutoff_ns INTEGER NOT NULL,
                protocol_hash TEXT NOT NULL, recorded_at TEXT NOT NULL, payload TEXT NOT NULL)''')
            connection.execute('CREATE INDEX IF NOT EXISTS memory_lookup '
                               'ON action_results(task_hash, cutoff_ns)')

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute('PRAGMA journal_mode=WAL')
        return connection

    def record_action_result(self, *, task, evidence_cutoff, protocol_hash, parent_hash,
                             candidate_id, action, structure=None, metrics=None,
                             elapsed_seconds=0.0, status='ok', lineage=(), failure=None,
                             event_id=None, partition='development', context_hash=None,
                             diagnostics=None):
        if partition != 'development':
            raise ValueError('Only development evidence may enter research memory')
        task_hash, task = task_identity(task)
        cutoff = utc(evidence_cutoff)
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0 or status not in STATUSES:
            raise ValueError('Invalid memory outcome')
        if structure is not None:
            structure = json.loads(json.dumps(asdict(RepresentationSpec.from_dict(structure))))
        if failure is not None:
            # Failure category only. Arbitrary exception prose may contain held-out data.
            failure = _identity(failure, 'failure')
        payload = {'task': task, 'partition': partition, 'evidence_cutoff': cutoff.isoformat(),
                   'protocol_hash': _identity(protocol_hash, 'protocol_hash'),
                   'parent_hash': _identity(parent_hash, 'parent_hash'),
                   'candidate_id': _identity(candidate_id, 'candidate_id', optional=True),
                   'action': clean_action(action), 'structure': structure,
                   'metrics': clean_metrics(metrics or {}), 'elapsed_seconds': elapsed_seconds,
                   'status': status, 'lineage': [_identity(v, 'lineage') for v in lineage],
                   'failure': failure, 'context_hash': _identity(context_hash, 'context_hash', True)}
        if diagnostics is not None:
            from .operator_search import clean_diagnostics
            payload['diagnostics'] = clean_diagnostics(diagnostics)
        reject_forbidden(payload)
        event_id = _identity(event_id or fingerprint(payload), 'event_id')
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            old = connection.execute('SELECT payload FROM action_results WHERE event_id=?',
                                     (event_id,)).fetchone()
            if old is not None and old[0] != encoded:
                raise ValueError('Memory event identity cannot be rewritten')
            connection.execute('INSERT OR IGNORE INTO action_results VALUES(?,?,?,?,?,?)',
                               (event_id, task_hash, int(cutoff.value), protocol_hash, now, encoded))
        return event_id

    def append(self, row: dict):
        """Append the exact public action/result schema; unexpected fields are rejected."""
        return self.record_action_result(**row)

    def retrieve(self, task, evidence_cutoff, *, protocol_hash=None, limit=20,
                 context_hash=None):
        """Return the latest eligible facts, newest first; history conversion reverses time."""
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError('Memory retrieval limit must be in 1..1000')
        task_hash, _ = task_identity(task)
        query = ('SELECT event_id, recorded_at, payload, protocol_hash, cutoff_ns '
                 'FROM action_results WHERE task_hash=? AND cutoff_ns<=?')
        args = [task_hash, int(utc(evidence_cutoff).value)]
        if protocol_hash is not None:
            query += ' AND protocol_hash=?'
            args.append(protocol_hash)
        query += ' ORDER BY cutoff_ns DESC, recorded_at DESC, event_id DESC'
        result = []
        with self._connect() as connection:
            for event_id, recorded_at, encoded, stored_protocol, stored_cutoff in connection.execute(query, args):
                payload = json.loads(encoded)
                expected = {'task', 'partition', 'evidence_cutoff', 'protocol_hash', 'parent_hash',
                            'candidate_id', 'action', 'structure', 'metrics', 'elapsed_seconds',
                            'status', 'lineage', 'failure', 'context_hash'}
                if 'diagnostics' in payload:
                    expected.add('diagnostics')
                    from .operator_search import clean_diagnostics
                    clean_diagnostics(payload['diagnostics'])
                if set(payload) != expected or task_identity(payload['task'])[0] != task_hash:
                    raise ValueError('Corrupt memory fact schema or task identity')
                if (payload['protocol_hash'] != stored_protocol
                    or (protocol_hash is not None and payload['protocol_hash'] != protocol_hash)):
                    raise ValueError('Corrupt memory protocol identity')
                if int(utc(payload['evidence_cutoff']).value) != stored_cutoff:
                    raise ValueError('Corrupt memory evidence cutoff identity')
                if context_hash is not None and payload.get('context_hash') != context_hash:
                    continue
                # A database is local evidence, not an instruction source. Revalidate on read.
                if payload.get('partition') != 'development':
                    raise ValueError('Corrupt memory evidence partition')
                reject_forbidden(payload)
                clean_action(payload['action'])
                clean_metrics(payload['metrics'])
                if payload['status'] not in STATUSES or not math.isfinite(payload['elapsed_seconds']) or payload['elapsed_seconds'] < 0:
                    raise ValueError('Corrupt memory outcome')
                for name in ('protocol_hash', 'parent_hash', 'candidate_id', 'context_hash', 'failure'):
                    _identity(payload[name], name, optional=name in {'candidate_id', 'context_hash', 'failure'})
                for value in payload['lineage']:
                    _identity(value, 'lineage')
                if payload['structure'] is not None:
                    RepresentationSpec.from_dict(payload['structure'])
                if utc(payload['evidence_cutoff']) > utc(evidence_cutoff):
                    raise ValueError('Corrupt memory evidence cutoff')
                result.append({**payload, 'event_id': event_id, 'recorded_at': recorded_at})
                if len(result) == limit:
                    break
        return result

    query = retrieve


def to_development_history(rows, current_spec=None):
    """Convert retrieved action/result facts to the existing search history schema.

    A recorded action is evidence of an attempted intervention, not a recovered
    free-form LLM rationale. The neutral descriptions below state that distinction.
    Retrieval is newest first to bound the selected facts, but search histories
    are oldest first so appending current outcomes and tail trimming keep the
    newest evidence. Use the same cutoff/recording/identity tie-break as retrieval.
    """
    result = []
    ordered = sorted(rows, key=lambda row: (utc(row['evidence_cutoff']),
                                            row.get('recorded_at', ''), row.get('event_id', '')))
    for index, row in enumerate(ordered):
        reject_forbidden(row)
        if row.get('partition') != 'development':
            raise ValueError('Only development memory can become agent history')
        if current_spec is not None:
            current_task = next(t for t in current_spec.tasks if t.name == current_spec.primary_task)
            if task_identity(row['task'])[0] != task_identity(current_task)[0]:
                raise ValueError('Memory task differs from current declared task')
            if utc(row['evidence_cutoff']) > utc(current_spec.dev_end):
                raise ValueError('Memory contains future evidence for the current round')
        action = clean_action(row['action'])
        metrics = clean_metrics(row['metrics'])
        history = {key: row[key] for key in ('partition', 'evidence_cutoff', 'protocol_hash',
                   'candidate_id', 'elapsed_seconds', 'status') if row.get(key) is not None}
        history['iteration'] = index
        if 'diagnostics' in row:
            from .operator_search import clean_diagnostics
            history['diagnostics'] = clean_diagnostics(row['diagnostics'])
        if current_spec is not None:
            # Derived request context is rebound, while historical experiment identity
            # remains explicit. The durable facts themselves are never rewritten.
            history['source_protocol_hash'] = row['protocol_hash']
            history['protocol_hash'] = current_spec.digest
        history['metrics'] = {key: value for key, value in metrics.items() if isinstance(value, dict)}
        for key in ('primary_loss', 'parameter_count', *sorted(ROBUST_METRIC_FIELDS)):
            if key in metrics:
                history[key] = metrics[key]
        if row.get('structure') is not None:
            history['architecture'] = asdict(RepresentationSpec.from_dict(row['structure']))
        history['proposal'] = {'protocol_hash': history['protocol_hash'], 'parent_hash': row['parent_hash'],
            'evidence_cutoff': row['evidence_cutoff'], **action,
            'hypothesis': 'A bounded action was attempted; this record stores its measured outcome.',
            'falsification': 'Use the recorded development outcome to assess the attempted action.',
            'source': 'persistent_development_action_memory'}
        if row.get('failure') is not None:
            history['error'] = _identity(row['failure'], 'failure')
        result.append(history)
    return result
