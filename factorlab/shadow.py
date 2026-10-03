"""Frozen research artifacts and a causal, append-only paper observation ledger."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from filelock import FileLock

from .protocol import ROOT, fingerprint, project_path

MANIFEST = 'bundle_manifest.json'
MODES = ('historical_replay', 'forward')


def _safe_path(value: str | Path) -> Path:
    raw = Path(value)
    raw = raw if raw.is_absolute() else ROOT / raw
    # Check before resolving, because resolution would conceal a junction/link.
    for part in (raw, *raw.parents):
        if part.is_symlink() or part.is_junction():
            raise ValueError('Links and junctions are forbidden in artifact paths')
    return project_path(raw)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _tree(path: Path, *, manifest_allowed: bool = False) -> dict:
    if not path.is_dir():
        raise ValueError('Artifact directory does not exist')
    directories, files = [], {}
    def walk(directory):
        for child in sorted(directory.iterdir()):
            if child.is_symlink() or child.is_junction():
                raise ValueError('Links and junctions are forbidden in frozen trees')
            yield child
            if child.is_dir():
                yield from walk(child)

    for item in walk(path):
        if item.is_symlink() or item.is_junction():
            raise ValueError('Links and junctions are forbidden in frozen trees')
        project_path(item)
        name = item.relative_to(path).as_posix()
        if name == MANIFEST:
            if manifest_allowed and item.is_file():
                continue
            raise ValueError('Source contains the reserved bundle manifest name')
        if item.is_dir():
            directories.append(name)
        elif item.is_file():
            files[name] = {'sha256': _sha(item), 'bytes': item.stat().st_size}
        else:
            raise ValueError('Unsupported artifact tree entry')
    return {'directories': directories, 'files': files}


def verify_bundle(directory: str | Path) -> dict:
    """Verify the exact file/directory tree and canonical manifest digest."""
    path = _safe_path(directory)
    try:
        manifest = json.loads(_safe_path(path / MANIFEST).read_text('utf-8'))
        if set(manifest) != {'version', 'metadata', 'directories', 'files', 'bundle_hash'}:
            raise ValueError('Unexpected manifest fields')
        body = {key: value for key, value in manifest.items() if key != 'bundle_hash'}
        if manifest['version'] != 1 or fingerprint(body) != manifest['bundle_hash']:
            raise ValueError('Frozen manifest hash mismatch')
        actual = _tree(path, manifest_allowed=True)
        if actual != {key: manifest[key] for key in ('directories', 'files')}:
            raise ValueError('Frozen bundle tree/hash mismatch')
        if not actual['files']:
            raise ValueError('Frozen bundle contains no files')
    except (OSError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError('Invalid frozen bundle') from exc
    return manifest


def freeze_bundle(source_directory: str | Path, destination_directory: str | Path,
                  metadata: dict) -> dict:
    """Atomically snapshot an artifact tree; an existing bundle is never replaced."""
    source, destination = _safe_path(source_directory), _safe_path(destination_directory)
    if source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError('Source and destination trees must be disjoint')
    if not isinstance(metadata, dict):
        raise TypeError('Bundle metadata must be a JSON dictionary')
    try:
        metadata = json.loads(json.dumps(metadata, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError('Bundle metadata must be finite JSON') from exc
    body = {'version': 1, 'metadata': metadata, **_tree(source)}
    if not body['files']:
        raise ValueError('Cannot freeze an empty artifact')
    manifest = {**body, 'bundle_hash': fingerprint(body)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _safe_path(destination.with_name(destination.name + '.freeze.lock'))
    with FileLock(str(lock_path)):
        _safe_path(destination)
        if destination.exists():
            existing = verify_bundle(destination)
            if existing != manifest:
                raise ValueError('Existing frozen bundle differs; choose a new destination')
            return existing
        with tempfile.TemporaryDirectory(prefix='.freeze-', dir=destination.parent) as temporary:
            stage = Path(temporary) / 'bundle'
            stage.mkdir()
            for name in body['directories']:
                (stage / name).mkdir(parents=True, exist_ok=True)
            for name in body['files']:
                original = _safe_path(source / name)
                target = stage / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, target)
                with target.open('r+b') as stream:
                    # Flush data before publishing the snapshot directory.
                    os.fsync(stream.fileno())
            if _tree(stage) != {key: body[key] for key in ('directories', 'files')}:
                raise ValueError('Source changed during freeze')
            (stage / MANIFEST).write_text(json.dumps(manifest, indent=2, allow_nan=False), 'utf-8')
            with (stage / MANIFEST).open('r+b') as stream:
                os.fsync(stream.fileno())
            stage.rename(destination)
    return verify_bundle(destination)


def _wall_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz='UTC')


def _time(value) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError('Observation timestamps must be explicit timezone-aware times')
    return result.tz_convert('UTC')


def _number(value) -> float:
    if isinstance(value, bool):
        raise TypeError('Returns must be finite numbers')
    result = float(value)
    if not math.isfinite(result) or result < -1:
        raise ValueError('Simple returns must be finite and at least -1')
    return result


class ShadowLedger:
    """SQLite journal; prediction/outcome rows are immutable after a durable commit.

    Explicit observation mode is required. Forward predictions must be submitted
    at the actual clock with less than an hour of input age. No orders are exposed.
    """

    def __init__(self, directory: str | Path, protocol_hash: str, model_hash: str, *,
                 observation_mode: str, availability_delay_seconds: int = 5):
        if observation_mode not in MODES:
            raise ValueError('Explicit historical_replay or forward observation mode required')
        for value in (protocol_hash, model_hash):
            if (not isinstance(value, str) or len(value) != 64 or
                    any(c not in '0123456789abcdef' for c in value)):
                raise ValueError('Protocol/model references must be SHA256 digests')
        if type(availability_delay_seconds) is not int or availability_delay_seconds <= 0:
            raise ValueError('Availability delay must be a positive integer')
        self.directory = _safe_path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = _safe_path(self.directory / 'shadow.sqlite3')
        self.config = {'version': 1, 'protocol_hash': protocol_hash, 'model_hash': model_hash,
                       'observation_mode': observation_mode, 'horizon_hours': 12,
                       'unit': '12h_simple_return',
                       'availability_delay_seconds': availability_delay_seconds,
                       'entry_rule': 'first_whole_utc_hour_strictly_after_availability'}
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY '
                       'CHECK (id = 1), config TEXT NOT NULL)')
            existing = db.execute('SELECT config FROM metadata WHERE id = 1').fetchone()
            if existing is not None and json.loads(existing[0]) != self.config:
                raise ValueError('Existing ledger protocol/model/mode differs')
            db.execute('INSERT OR IGNORE INTO metadata VALUES (1, ?)',
                       (json.dumps(self.config, sort_keys=True),))
            db.execute('CREATE TABLE IF NOT EXISTS predictions ('
                       'decision_id TEXT PRIMARY KEY, symbol TEXT NOT NULL, '
                       'available_at INTEGER NOT NULL, entry_time INTEGER NOT NULL, '
                       'label_end INTEGER NOT NULL, prediction REAL NOT NULL, '
                       'recorded_at INTEGER NOT NULL, '
                       'UNIQUE (symbol, available_at))')
            db.execute('CREATE TABLE IF NOT EXISTS outcomes ('
                       'decision_id TEXT PRIMARY KEY REFERENCES predictions(decision_id), '
                       'realized_return REAL NOT NULL, label_available_at INTEGER NOT NULL, '
                       'recorded_at INTEGER NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS observations ('
                       'id INTEGER PRIMARY KEY, recorded_at INTEGER NOT NULL)')
            for table in ('metadata', 'predictions', 'outcomes', 'observations'):
                for operation in ('UPDATE', 'DELETE'):
                    db.execute(f'CREATE TRIGGER IF NOT EXISTS no_{operation.lower()}_{table} '
                               f'BEFORE {operation} ON {table} BEGIN '
                               "SELECT RAISE(ABORT, 'Shadow ledger is append-only'); END")

    @contextmanager
    def _connect(self):
        _safe_path(self.path)
        for suffix in ('-journal', '-wal', '-shm'):
            _safe_path(self.path.with_name(self.path.name + suffix))
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys = ON')
        db.execute('PRAGMA synchronous = FULL')
        try:
            with db:
                yield db
        finally:
            db.close()

    def _check_clock(self, db, now):
        now = _time(now)
        wall = _wall_now()
        if now > wall:
            raise ValueError('Observation now cannot be in the future')
        if self.config['observation_mode'] == 'forward' and wall - now > pd.Timedelta(seconds=60):
            raise ValueError('Forward observation now must match the actual wall clock')
        latest = db.execute('SELECT MAX(recorded_at) FROM ('
                            'SELECT recorded_at FROM predictions UNION ALL '
                            'SELECT recorded_at FROM outcomes UNION ALL '
                            'SELECT recorded_at FROM observations)').fetchone()[0]
        if latest is not None and now.value < latest:
            raise ValueError('Ledger observation clock cannot move backward')
        config = db.execute('SELECT config FROM metadata WHERE id=1').fetchone()
        if config is None or json.loads(config[0]) != self.config:
            raise ValueError('Ledger metadata integrity failed')
        return now

    def append_prediction(self, decision_id: str, symbol: str, available_at, entry_time,
                          label_end, prediction: float, now) -> dict:
        if not isinstance(decision_id, str) or not decision_id.strip():
            raise ValueError('A nonempty decision ID is required')
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError('A nonempty symbol is required')
        available, entry, end = map(_time, (available_at, entry_time, label_end))
        value = _number(prediction)
        if entry != available.floor('h') + pd.Timedelta(hours=1):
            raise ValueError('Entry must be the first whole hour strictly after availability')
        if end != entry + pd.Timedelta(hours=12):
            raise ValueError('Label end must equal entry plus 12 hours')
        try:
            with self._connect() as db:
                db.execute('BEGIN IMMEDIATE')
                stamp = self._check_clock(db, now)
                if available > stamp:
                    raise ValueError('Prediction input availability is in the future')
                if self.config['observation_mode'] == 'forward':
                    wall = _wall_now()
                    if wall - available >= pd.Timedelta(hours=1):
                        raise ValueError('Forward inputs must be less than one hour old')
                    if entry <= wall:
                        raise ValueError('Forward prediction must precede its hourly entry')
                db.execute('INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?)',
                           (decision_id, symbol, available.value, entry.value, end.value,
                            value, stamp.value))
        except sqlite3.IntegrityError as exc:
            raise ValueError('Duplicate decision or symbol/availability') from exc
        return {**self.config, 'decision_id': decision_id, 'symbol': symbol,
                'available_at': available.isoformat(), 'entry_time': entry.isoformat(),
                'label_end': end.isoformat(), 'prediction': value, 'recorded_at': stamp.isoformat()}

    def append_outcome(self, decision_id: str, realized_return: float,
                       label_available_at, now) -> dict:
        value, available = _number(realized_return), _time(label_available_at)
        try:
            with self._connect() as db:
                db.execute('BEGIN IMMEDIATE')
                stamp = self._check_clock(db, now)
                prediction = db.execute('SELECT * FROM predictions WHERE decision_id = ?',
                                        (decision_id,)).fetchone()
                if prediction is None:
                    raise ValueError('Outcome must reference an existing prediction')
                expected = prediction['label_end'] + (
                    self.config['availability_delay_seconds'] * 1_000_000_000)
                if available.value != expected:
                    raise ValueError('Label availability differs from frozen horizon/delay')
                if available > stamp:
                    raise ValueError('Outcome label has not matured')
                db.execute('INSERT INTO outcomes VALUES (?, ?, ?, ?)',
                           (decision_id, value, available.value, stamp.value))
        except sqlite3.IntegrityError as exc:
            raise ValueError('Duplicate outcome') from exc
        return {'decision_id': decision_id, 'realized_return': value,
                'label_available_at': available.isoformat(), 'recorded_at': stamp.isoformat(),
                'observation_mode': self.config['observation_mode']}

    def summary(self, now) -> dict:
        """Export counts and mature forecast errors, without a profitability claim."""
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            stamp = self._check_clock(db, now)
            rows = db.execute('SELECT p.*, o.realized_return, o.label_available_at, '
                              'o.recorded_at AS outcome_recorded_at FROM predictions p '
                              'LEFT JOIN outcomes o USING (decision_id)').fetchall()
            if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise ValueError('Ledger storage integrity failed')
            db.execute('INSERT INTO observations (recorded_at) VALUES (?)', (stamp.value,))
        delay = self.config['availability_delay_seconds'] * 1_000_000_000
        mature, unmature, missing, invalid, errors = 0, 0, 0, 0, []
        for row in rows:
            expected_entry = pd.Timestamp(row['available_at'], tz='UTC').floor('h').value + (
                3_600_000_000_000)
            expected_label = row['label_end'] + delay
            valid = (row['entry_time'] == expected_entry and
                     row['label_end'] == row['entry_time'] + 43_200_000_000_000 and
                     row['available_at'] <= row['recorded_at'] <= stamp.value and
                     math.isfinite(row['prediction']) and row['prediction'] >= -1)
            if row['realized_return'] is not None:
                valid &= (row['label_available_at'] == expected_label and
                          expected_label <= row['outcome_recorded_at'] <= stamp.value and
                          row['outcome_recorded_at'] >= row['recorded_at'] and
                          math.isfinite(row['realized_return']) and row['realized_return'] >= -1)
            if not valid:
                invalid += 1
            elif expected_label > stamp.value:
                unmature += 1
            elif row['realized_return'] is None:
                missing += 1
            else:
                mature += 1
                errors.append(row['prediction'] - row['realized_return'])
        return {**self.config, 'as_of': stamp.isoformat(), 'predictions': len(rows),
                'mature': mature, 'unmature': unmature, 'mature_without_outcome': missing,
                'invalid': invalid,
                'mae': sum(abs(x) for x in errors) / mature if mature else None,
                'mse': sum(x*x for x in errors) / mature if mature else None,
                'interpretation': 'forecast_error_only_not_portfolio_profitability'}

    def score(self, now) -> dict:
        return self.summary(now)
