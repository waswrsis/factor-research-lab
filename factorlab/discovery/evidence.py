"""Immutable experiment records and persistent, conservative evidence-use accounting."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from filelock import FileLock

from factorlab.protocol import fingerprint, project_path, utc

from .contracts import ExperimentSpec

KNOWN_RUNS_ROOT = 'runs'
KNOWN_OUTPUTS_ROOT = 'outputs'
DEFAULT_EVIDENCE_SOURCE = 'binance_um_1h'


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def immutable_json(path: str | Path, value: object) -> Path:
    """An existing equal record is reusable; a different record is never overwritten."""
    target = project_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    with FileLock(str(target) + '.lock'):
        if target.exists():
            if json.loads(target.read_text('utf-8')) != json.loads(content):
                raise ValueError(f'Immutable record already exists: {target}')
        else:
            with target.open('x', encoding='utf-8') as stream:
                stream.write(content)
    return target


class ExperimentJournal:
    def __init__(self, path: str | Path):
        self.path = project_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        records = [json.loads(line) for line in self.path.read_text('utf-8').splitlines()]
        previous = '0' * 64
        for record in records:
            digest = record['sha256']
            payload = {k: v for k, v in record.items() if k != 'sha256'}
            if payload['previous'] != previous or fingerprint(payload) != digest:
                raise ValueError('Experiment journal integrity failure')
            previous = digest
        return records

    def append(self, event: str, value: dict) -> dict:
        with FileLock(str(self.path) + '.lock'):
            records = self.read()
            payload = {'sequence': len(records), 'time': datetime.now(UTC).isoformat(),
                       'event': event, 'value': value,
                       'previous': records[-1]['sha256'] if records else '0' * 64}
            record = {**payload, 'sha256': fingerprint(payload)}
            with self.path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
            return record


class EvidenceRegistry:
    """Exposure is keyed by market/source/time, not dataset filename or task name.

    Re-exporting the same candles or changing labels cannot reset exposure.
    A confirmation claim is consumed before scoring, including failed attempts.
    The database and known local artifacts recover common accidental deletions
    or replacement ledgers. They are not externally signed or tamper-proof;
    a project owner able to alter every artifact can also erase local history.
    """

    def __init__(self, path: str | Path = 'data/discovery_evidence.sqlite'):
        self.path = project_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute('''CREATE TABLE IF NOT EXISTS exposures (
                id INTEGER PRIMARY KEY, source TEXT NOT NULL, symbol TEXT NOT NULL,
                start TEXT NOT NULL, end TEXT NOT NULL, role TEXT NOT NULL,
                run TEXT NOT NULL, claim TEXT NOT NULL, created TEXT NOT NULL,
                UNIQUE(source, symbol, start, end, role, run, claim))''')

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute('PRAGMA busy_timeout=30000')
        return connection

    def consume(self, *, source: str, symbols: tuple[str, ...], start: str, end: str,
                run: str, claim: str, role: str = 'confirmation') -> None:
        if role not in {'development', 'confirmation', 'legacy_seen', 'sequential'}:
            raise ValueError('Unknown evidence role')
        begin, finish = utc(start).isoformat(), utc(end).isoformat()
        if begin >= finish or not symbols or not source or not claim:
            raise ValueError('Invalid evidence interval')
        with self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            for symbol in symbols:
                overlap = connection.execute(
                    'SELECT role,run FROM exposures WHERE source=? AND symbol=? '
                    'AND start<? AND end>?', (source, symbol, finish, begin)
                ).fetchall()
                if role in {'confirmation', 'sequential'} and overlap:
                    raise ValueError(f'Previously exposed evidence for {symbol}: {overlap}')
                connection.execute('INSERT OR IGNORE INTO exposures '
                                   '(source,symbol,start,end,role,run,claim,created) '
                                   'VALUES (?,?,?,?,?,?,?,?)',
                                   (source, symbol, begin, finish, role, run, claim,
                                    datetime.now(UTC).isoformat()))

    def records(self) -> list[dict]:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute('SELECT * FROM exposures ORDER BY id')]

    def import_prior_runs(self) -> None:
        """Recover known completed/failed runs and replay/score exposure manifests.

        Scan only the known local runs/outputs trees, including nested runs. Valid journals
        retain failed confirmation consumption even without results.json. Missing
        result/consumption records conservatively expose the development and
        calibration input interval. Source snapshots are never treated as runs.
        This covers ordinary missing/replaced databases, not malicious rewriting
        of the entire workspace or research artifacts outside the known trees.
        """
        roots = (project_path(KNOWN_RUNS_ROOT), project_path(KNOWN_OUTPUTS_ROOT))

        def known_paths(name):
            return sorted({path for root in roots for path in root.rglob(name)
                           if 'source' not in path.relative_to(root).parts})

        for protocol_path in known_paths('protocol.json'):
            result_path = protocol_path.parent / 'results.json'
            payload = json.loads(protocol_path.read_text('utf-8-sig'))
            if not isinstance(payload, dict):
                continue
            protocol = payload.get('protocol', payload)
            if not {'start', 'end', 'symbols'} <= set(protocol):
                continue
            is_discovery = protocol.get('version', '').startswith('discovery')
            if is_discovery:
                result = (json.loads(result_path.read_text('utf-8-sig'))
                          if result_path.exists() else {})
                consumed = bool(result.get('confirmation'))
                journal_path = protocol_path.parent / 'journal.jsonl'
                if journal_path.exists():
                    # Reject damaged recovery evidence rather than silently
                    # granting unseen status to a possibly consumed interval.
                    records = ExperimentJournal(journal_path).read()
                    consumed |= any(row['event'] == 'confirmation_consumed' for row in records)
                finish = protocol['end'] if consumed else protocol['calibration_end']
            elif protocol.get('version', '').startswith('prototype'):
                result = (json.loads(result_path.read_text('utf-8-sig'))
                          if result_path.exists() else {})
                finish = protocol['end']
            else:
                continue
            warmup_hours = int(protocol.get('lookback', 48)) + 48
            self.consume(source=result.get('evidence_source', DEFAULT_EVIDENCE_SOURCE),
                symbols=tuple(protocol['symbols']),
                start=(utc(protocol['start']) - pd.Timedelta(hours=warmup_hours)).isoformat(),
                end=finish, run=str(protocol_path.parent), role='legacy_seen',
                claim='historical-exposure-' + fingerprint(protocol))
        for manifest_path in known_paths('manifest.json'):
            payload = json.loads(manifest_path.read_text('utf-8-sig'))
            if not isinstance(payload, dict):
                continue
            manifest = payload.get('manifest', payload)
            exposure = manifest.get('evidence_exposure')
            if exposure is None:
                continue
            if 'manifest' in payload and payload.get('sha256') != fingerprint(manifest):
                raise ValueError('Historical exposure manifest integrity failure')
            intervals = exposure.get('intervals')
            if not isinstance(intervals, list) or not intervals:
                raise ValueError('Historical exposure manifest requires declared intervals')
            source = exposure.get('source', DEFAULT_EVIDENCE_SOURCE)
            claim = 'historical-manifest-exposure-' + fingerprint(exposure)
            for interval in intervals:
                self.consume(source=source, symbols=(interval['symbol'],),
                    start=interval['start'], end=interval['end'], run=str(manifest_path.parent),
                    claim=claim, role='legacy_seen')


def snapshot_source(directory: str | Path) -> dict:
    destination = project_path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    root = project_path('.')
    files = [*root.glob('factorlab/**/*.py'), root / 'pyproject.toml', root / 'uv.lock']
    hashes = {}
    for source in files:
        relative = source.relative_to(root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(source.read_bytes())
        hashes[str(relative).replace('\\', '/')] = file_hash(target)
    immutable_json(destination / 'manifest.json', hashes)
    return {'files': hashes, 'digest': fingerprint(hashes)}


def freeze_claim(directory: str | Path, spec: ExperimentSpec, bundle_directory: str | Path,
                 data_hash: str, source_hash: str, comparisons: list[str]) -> dict:
    bundle = project_path(bundle_directory)
    files = {str(path.relative_to(bundle)).replace('\\', '/'): file_hash(path)
             for path in sorted(bundle.rglob('*')) if path.is_file() and path.suffix != '.lock'}
    if not files or not comparisons:
        raise ValueError('A complete fitted bundle and declared comparisons are required')
    claim = {'protocol': asdict(spec), 'protocol_hash': spec.digest, 'bundle_files': files,
             'data_hash': data_hash, 'source_hash': source_hash, 'comparisons': comparisons,
             'primary_task': spec.primary_task, 'scope': 'fixed_task_and_covered_universe',
             'confirmation_start': spec.calibration_end, 'confirmation_end': spec.end}
    claim['digest'] = fingerprint(claim)
    immutable_json(project_path(directory) / 'frozen_claim.json', claim)
    return claim


def verify_bundle(bundle_directory: str | Path, claim: dict) -> None:
    bundle = project_path(bundle_directory)
    expected = claim['bundle_files']
    actual = {str(p.relative_to(bundle)).replace('\\', '/'): file_hash(p)
              for p in bundle.rglob('*') if p.is_file() and p.suffix != '.lock'}
    if actual != expected:
        raise ValueError('Frozen bundle was changed or is incomplete')
    if fingerprint({k: v for k, v in claim.items() if k != 'digest'}) != claim['digest']:
        raise ValueError('Claim integrity failure')
