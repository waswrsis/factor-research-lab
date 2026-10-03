"""Append-only research replay: settle, predict, then update for future origins."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock

from factorlab.protocol import fingerprint, project_path, utc

from .contracts import ExperimentSpec, RepresentationSpec, TaskDataset
from .evidence import EvidenceRegistry, file_hash, immutable_json
from .learning import FittedResearchModel, fit
from .scoring import adaptive_horizons, per_sample_loss
from .tasks import FEATURE_WARMUP_HOURS

CUTOFF_FIELDS = ('fit_feature_cutoff', 'fit_label_cutoff',
                 'development_feature_cutoff', 'development_label_cutoff')
DEFAULT_REGISTRY_PATH = 'data/discovery_evidence.sqlite'
DEFAULT_EVIDENCE_SOURCE = 'binance_um_1h'


def _model_cutoffs(model, cutoff: pd.Timestamp) -> dict:
    """Reject unverifiable or future-trained initial/update checkpoints."""
    values = {}
    for key in CUTOFF_FIELDS:
        value = model.metadata.get(key, model.diagnostics.get(key))
        if value is None:
            raise ValueError(f'Model is missing historical fit provenance: {key}')
        time = utc(value)
        if time > cutoff:
            raise ValueError(f'Model {key} exceeds replay information cutoff')
        values[key] = time.isoformat()
    return values


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _data_hash(data: TaskDataset) -> str:
    return fingerprint({
        'schema': data.schema, 'x': _array_hash(data.x),
        'sample_id': data.meta.sample_id.tolist(),
        'available_at': [utc(t).isoformat() for t in data.meta.available_at],
        'targets': {k: _array_hash(v) for k, v in data.targets.items()},
        'observed': {k: _array_hash(v) for k, v in data.observed.items()},
        'maturity': {k: _array_hash(v) for k, v in data.maturity.items()},
    })


def _prediction_inputs(data: TaskDataset) -> TaskDataset:
    """Even arbitrary predictors receive no future outcomes or label diagnostics."""
    n = len(data.meta)
    columns = [name for name in ('sample_id', 'symbol', 'available_at', 'feature_open_time',
                                'window_start_open_time') if name in data.meta]
    return TaskDataset(
        data.x.copy(), {t.name: np.full(n, np.nan) for t in data.tasks},
        {t.name: np.zeros(n, dtype=bool) for t in data.tasks},
        {t.name: np.full(n, np.datetime64('NaT'), dtype='datetime64[ns]') for t in data.tasks},
        data.meta.loc[:, columns].copy(), data.features, data.tasks,
    )


def _training_parts(data: TaskDataset, timestamp: pd.Timestamp):
    # Full periods must have elapsed for every task, including censored tasks.
    eligible = (data.meta.available_at < timestamp).to_numpy()
    for ready in data.maturity.values():
        eligible &= ~np.isnat(ready) & (ready <= timestamp.to_datetime64())
    history = data.subset(eligible)
    times = pd.DatetimeIndex(history.meta.available_at.unique()).sort_values()
    if len(times) < 2:
        raise ValueError('Insufficient mature chronological history for refit and early stopping')
    boundary = times[min(len(times) - 1, max(1, int(len(times) * .8)))]
    train_mask = (history.meta.available_at < boundary).to_numpy()
    for ready in history.maturity.values():
        train_mask &= ready <= boundary.to_datetime64()
    train = history.subset(train_mask)
    early = history.subset((history.meta.available_at >= boundary).to_numpy())
    if not len(train.meta) or not len(early.meta):
        raise ValueError('Chronological purge leaves an empty refit partition')
    # Masked outcomes are cleared as well as masked, avoiding accidental reads.
    for part in (train, early):
        for name in part.targets:
            part.targets[name] = part.targets[name].astype(np.float64, copy=True)
            part.targets[name][~part.observed[name]] = np.nan
    details = {
        'information_cutoff': timestamp.isoformat(), 'early_stopping_start': boundary.isoformat(),
        'chronological_fraction': .8, 'purge': 'train full label horizons end by early stopping start',
        'train_rows': len(train.meta), 'early_stopping_rows': len(early.meta),
        'train_sample_ids': train.meta.sample_id.tolist(),
        'early_stopping_sample_ids': early.meta.sample_id.tolist(),
        'train_sha256': _data_hash(train), 'early_stopping_sha256': _data_hash(early),
        'train_support': {t.name: int(train.observed[t.name].sum()) for t in train.tasks},
        'early_stopping_support': {t.name: int(early.observed[t.name].sum()) for t in early.tasks},
    }
    return train, early, details


def _replay_inputs(data: TaskDataset, end: pd.Timestamp) -> TaskDataset:
    """Keep history and replay origins; never hash/analyze a confirmation suffix."""
    result = data.subset((data.meta.available_at < end).to_numpy())
    for name in result.targets:
        ready = result.maturity[name]
        allowed = result.observed[name] & ~np.isnat(ready) & (ready <= end.to_datetime64())
        result.observed[name] = allowed
        result.targets[name] = result.targets[name].astype(float, copy=True)
        result.targets[name][~allowed] = np.nan
    return result


def _consume_replay_evidence(data, spec, directory, registry_path, evidence_source):
    """Conservatively mark every input candle/trailing warmup read by this replay."""
    registry = EvidenceRegistry(registry_path)
    source = evidence_source
    if not isinstance(source, str) or not source.strip():
        raise ValueError('Replay evidence source must be a nonempty source identity')
    intervals = []
    for symbol, group in data.meta.groupby('symbol', sort=True):
        if 'window_start_open_time' in group:
            first = pd.to_datetime(group.window_start_open_time, utc=True).min()
        elif 'feature_open_time' in group:
            first = pd.to_datetime(group.feature_open_time, utc=True).min() - pd.Timedelta(hours=spec.lookback - 1)
        else:
            first = pd.to_datetime(group.available_at, utc=True).min() - pd.Timedelta(
                hours=spec.lookback, seconds=spec.availability_delay_seconds)
        if pd.isna(first):
            raise ValueError('Missing input candle provenance for evidence registration')
        begin = first - pd.Timedelta(hours=FEATURE_WARMUP_HOURS)
        finish = utc(spec.dev_end)
        intervals.append({'symbol': symbol, 'start': begin.isoformat(), 'end': finish.isoformat()})
    claim = 'development-replay-' + fingerprint({
        'run': str(directory), 'protocol': spec.digest, 'schema': data.schema, 'intervals': intervals,
    })
    for interval in intervals:
        registry.consume(source=source, symbols=(interval['symbol'],), start=interval['start'],
                         end=interval['end'], run=str(directory), claim=claim, role='development')
    receipts = [row for row in registry.records()
                if row['run'] == str(directory) and row['claim'] == claim and row['source'] == source]
    if len(receipts) != len(intervals):
        raise ValueError('Replay evidence registration is incomplete')
    return {'registry_path': str(registry.path), 'source': source, 'role': 'development',
            'claim': claim, 'intervals': intervals, 'receipts': receipts,
            'warmup_policy': f'actual earliest input window plus {FEATURE_WARMUP_HOURS}h trailing feature history',
            'registered_before': 'dataset_hash_prediction_fit_and_user_results'}


class _Writer:
    """Single-writer append chain; a complete summary anchors the terminal hash."""
    def __init__(self, path: Path):
        self.path, self.sequence, self.previous = path, 0, '0' * 64
        self.stream = path.open('x', encoding='utf-8')

    def append(self, timestamp, event, value):
        payload = {'sequence': self.sequence, 'timestamp': utc(timestamp).isoformat(),
                   'event': event, 'value': value, 'previous': self.previous}
        record = {**payload, 'sha256': fingerprint(payload)}
        self.stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.sequence += 1
        self.previous = record['sha256']
        return record

    def close(self):
        self.stream.close()


def _groups(model, encoded: np.ndarray):
    architecture = model.metadata.get('architecture')
    if architecture is None:
        return {'fixed_summary': encoded.tolist()}
    groups, offset = {}, 0
    for module in architecture['modules']:
        stop = offset + module['dim']
        groups[module['name']] = encoded[offset:stop].tolist()
        offset = stop
    if offset != len(encoded):
        raise ValueError('Latent factor groups do not match checkpoint architecture')
    return groups


def _version(model, directory: Path, version: int, cutoff: pd.Timestamp, training=None):
    cutoffs = _model_cutoffs(model, cutoff)
    path = directory / 'versions' / f'{version:04d}' / 'model.pt'
    if path.exists():
        raise ValueError('Immutable replay model version already exists')
    model.save(path)
    metadata_path = path.with_name('version.json')
    value = {'version': version, 'model_digest': model.digest,
             'checkpoint': str(path.relative_to(directory)).replace('\\', '/'),
             'checkpoint_sha256': file_hash(path), 'cutoffs': cutoffs,
             'metadata': model.metadata, 'training': training,
             'effective_from': 'next prediction timestamp after update'}
    immutable_json(metadata_path, value)
    return {**value, 'version_record': str(metadata_path.relative_to(directory)).replace('\\', '/'),
            'version_record_sha256': file_hash(metadata_path)}


def run_replay(data: TaskDataset, initial_model: FittedResearchModel, spec: ExperimentSpec,
               output_dir: str | Path, *, architecture: RepresentationSpec | None = None,
               update_every_hours=168, update_mode='fixed', device='cpu',
               registry_path: str | Path | None = None, evidence_source: str | None = None) -> dict:
    """Replay only [train_end, dev_end), a development comparison, not confirmation.

    At each origin settle previously forecast mature tasks, record predictions
    with the frozen model, then perform a scheduled bounded refit. Newly fitted
    versions apply only at a later origin; failed updates retain the old version.
    The final dev_end tick settles outcomes known at that endpoint. No trading,
    positions, orders or execution simulation is part of this research replay.
    Every input/trailing warmup is registered as development exposure before
    fitting or producing outputs; callers can inject an isolated test registry.
    """
    if update_mode not in {'fixed', 'refit'}:
        raise ValueError('Update mode must be fixed or refit')
    if type(update_every_hours) is not int or update_every_hours < 1:
        raise ValueError('Update schedule must be positive integer hours')
    start, end = utc(spec.train_end), utc(spec.dev_end)
    if data.schema != initial_model.schema:
        raise ValueError('Initial checkpoint differs from the replay task schema')
    if data.tasks != spec.tasks or data.x.shape[1] != spec.lookback:
        raise ValueError('Replay dataset differs from the frozen task/lookback protocol')
    if not set(data.meta.symbol) <= set(spec.symbols):
        raise ValueError('Replay dataset contains symbols outside the frozen universe')
    _model_cutoffs(initial_model, start)
    if not data.meta.equals(data.meta.sort_values(['available_at', 'symbol'], kind='stable').reset_index(drop=True)):
        raise ValueError('Replay rows must be sorted by availability then symbol')
    data = _replay_inputs(data, end)
    if architecture is None and initial_model.metadata.get('architecture') is not None:
        architecture = RepresentationSpec.from_dict(initial_model.metadata['architecture'])
    directory = project_path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / '.replay.lock')):
        if (directory / 'manifest.json').exists() or (directory / 'records.jsonl').exists():
            raise ValueError('Replay history already exists; choose a new output directory')
        exposure = _consume_replay_evidence(
            data, spec, directory, registry_path if registry_path is not None else DEFAULT_REGISTRY_PATH,
            evidence_source if evidence_source is not None else DEFAULT_EVIDENCE_SOURCE,
        )
        sources = {}
        for source in (Path(__file__), Path(__file__).with_name('learning.py'),
                       Path(__file__).with_name('contracts.py'), Path(__file__).with_name('scoring.py'),
                       Path(__file__).with_name('evidence.py'), Path(__file__).with_name('tasks.py')):
            target = directory / 'source' / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(source.read_bytes())
            sources[str(target.relative_to(directory)).replace('\\', '/')] = file_hash(target)
        manifest = {
            'format': 'discovery-replay-v1', 'protocol': asdict(spec), 'protocol_sha256': spec.digest,
            'schema': data.schema, 'data_sha256': _data_hash(data), 'source_files': sources,
            'initial_model_digest': initial_model.digest,
            'architecture': asdict(architecture) if architecture is not None else None,
            'start': start.isoformat(), 'end_exclusive': end.isoformat(),
            'scope': 'previously_seen_development_interval_not_independent_confirmation',
            'update_mode': update_mode, 'update_every_hours': update_every_hours,
            'schedule_anchor': start.isoformat(), 'missing_ticks': 'coalesce_at_next_origin',
            'operation_order': ['settle_past', 'record_prediction', 'refit_for_next_origin'],
            'early_stopping_fraction': .2, 'adaptive_rule': 'unconditional_event_mass_threshold_0.8',
            'evidence_exposure': exposure,
        }
        manifest_record = {'manifest': manifest, 'sha256': fingerprint(manifest)}
        immutable_json(directory / 'manifest.json', manifest_record)
        version = 0
        next_version = 1
        model = initial_model
        versions = [_version(model, directory, version, start)]
        pending, losses, updates = {}, {t.name: [] for t in data.tasks}, []
        scored, censored, predictions_count = 0, 0, 0
        by_id = {sample: i for i, sample in enumerate(data.meta.sample_id)}
        task_by_name = {t.name: t for t in data.tasks}
        writer = _Writer(directory / 'records.jsonl')

        def settle(timestamp):
            nonlocal scored, censored
            for key in list(pending):
                forecast = pending[key]
                i, name = by_id[forecast['sample_id']], forecast['task']
                ready = data.maturity[name][i]
                if np.isnat(ready) or ready > timestamp.to_datetime64():
                    continue
                task = task_by_name[name]
                value = {'sample_id': forecast['sample_id'], 'task': name,
                         'prediction_sequence': forecast['prediction_sequence'],
                         'version': forecast['version'], 'maturity': pd.Timestamp(ready, tz='UTC').isoformat(),
                         'predictive': task.predictive}
                if not data.observed[name][i]:
                    value.update(status='unobserved', loss=None, target=None)
                    censored += 1
                elif forecast['prediction'] is None:
                    value.update(status='unavailable_prediction', loss=None, target=float(data.targets[name][i]))
                    censored += 1
                else:
                    target = float(data.targets[name][i])
                    prediction = np.asarray([forecast['prediction']], dtype=float)
                    loss = float(per_sample_loss(task, np.asarray([target]), prediction)[0])
                    value.update(status='scored', target=target, loss=loss,
                                 error=(None if task.kind == 'first_event' else float(prediction[0] - target)))
                    losses[name].append(loss)
                    scored += 1
                writer.append(timestamp, 'settlement', value)
                del pending[key]

        try:
            writer.append(start, 'version', {k: v for k, v in versions[0].items() if k != 'metadata'})
            interval = (data.meta.available_at >= start) & (data.meta.available_at < end)
            origins = pd.DatetimeIndex(data.meta.loc[interval, 'available_at'].unique()).sort_values()
            next_update = start + pd.Timedelta(hours=update_every_hours)
            for timestamp in origins:
                settle(timestamp)
                batch = data.subset((data.meta.available_at == timestamp).to_numpy())
                inputs = _prediction_inputs(batch)
                previous_digest = model.digest
                output = model.predict(inputs)
                encoded = np.asarray(model.encode(inputs), dtype=float)
                if model.digest != previous_digest:
                    raise ValueError('Frozen model was mutated during prediction')
                if encoded.ndim != 2 or len(encoded) != len(batch.meta) or not np.isfinite(encoded).all():
                    raise ValueError('Invalid latent group output')
                if set(output) - set(task_by_name):
                    raise ValueError('Predictor returned an undeclared task')
                for task in data.tasks:
                    if task.name not in output:
                        continue
                    values = np.asarray(output[task.name], dtype=float)
                    if task.kind != 'first_event' and values.shape == (len(batch.meta), 1):
                        values = values[:, 0]
                    shape = ((len(batch.meta), task.output_size) if task.kind == 'first_event'
                             else (len(batch.meta),))
                    if values.shape != shape or not np.isfinite(values).all():
                        raise ValueError('Task prediction is nonfinite or misaligned')
                    if task.kind == 'volatility' and (values < 0).any():
                        raise ValueError('Negative replay volatility prediction')
                    if task.kind == 'first_event':
                        adaptive_horizons(values, task)  # Validates probabilities before recording.
                    output[task.name] = values
                for j, row in batch.meta.iterrows():
                    task_values, adaptive = {}, {}
                    for task in data.tasks:
                        prediction = output[task.name][j] if task.name in output else None
                        task_values[task.name] = (
                            None if prediction is None else np.asarray(prediction).tolist())
                        if task.kind == 'first_event' and prediction is not None:
                            derived = adaptive_horizons(np.asarray(prediction)[None], task)
                            adaptive[task.name] = {
                                k: (v[0].tolist() if isinstance(v, np.ndarray) else v)
                                for k, v in derived.items()
                            }
                    value = {'sample_id': row.sample_id, 'symbol': row.symbol,
                             'available_at': timestamp.isoformat(), 'version': version,
                             'model_digest': previous_digest, 'predictions': task_values,
                             'latent_groups': _groups(model, encoded[j]), 'adaptive_horizons': adaptive}
                    record = writer.append(timestamp, 'prediction', value)
                    predictions_count += 1
                    for name, prediction in task_values.items():
                        pending[(row.sample_id, name)] = {
                            'sample_id': row.sample_id, 'task': name, 'prediction': prediction,
                            'version': version, 'prediction_sequence': record['sequence'],
                        }
                if timestamp >= next_update:
                    due = []
                    while next_update <= timestamp:
                        due.append(next_update.isoformat())
                        next_update += pd.Timedelta(hours=update_every_hours)
                    update = {'scheduled_at': due, 'information_cutoff': timestamp.isoformat(),
                              'previous_version': version, 'effective_after': timestamp.isoformat()}
                    if update_mode == 'fixed':
                        update['status'] = 'fixed_model_retained'
                    else:
                        attempted_version = next_version
                        next_version += 1
                        update['attempted_version'] = attempted_version
                        try:
                            train, early, details = _training_parts(data, timestamp)
                            candidate = fit(train, early, architecture, spec.training, device=device)
                            if candidate.schema != data.schema:
                                raise ValueError('Refitted checkpoint changed dataset schema')
                            candidate_version = _version(candidate, directory, attempted_version, timestamp, details)
                        except Exception as error:  # noqa: BLE001 - isolated update failure is recorded, old model retained
                            update.update(status='failed_old_version_retained',
                                          error_type=type(error).__name__, error=str(error))
                        else:
                            # Log the complete checkpoint before changing the active version.
                            # Journal I/O failures must abort, never masquerade as fit failures.
                            writer.append(timestamp, 'version', {
                                k: v for k, v in candidate_version.items() if k != 'metadata'
                            })
                            versions.append(candidate_version)
                            model, version = candidate, attempted_version
                            update.update(status='refitted', version=version, model_digest=model.digest,
                                          training=details)
                    updates.append(update)
                    writer.append(timestamp, 'update', update)
            settle(end)
            summary = {
                'status': 'complete', 'scope': manifest['scope'], 'update_mode': update_mode,
                'predictions': predictions_count, 'scored_tasks': scored, 'censored_tasks': censored,
                'pending_tasks': len(pending), 'pending': list(pending.values()), 'updates': updates,
                'tasks': {t.name: {'kind': t.kind, 'predictive': t.predictive,
                                   'support': len(losses[t.name]),
                                   'mean_loss': float(np.mean(losses[t.name])) if losses[t.name] else None}
                          for t in data.tasks},
                'versions': versions, 'manifest_sha256': manifest_record['sha256'],
                'journal_records': writer.sequence, 'journal_terminal_sha256': writer.previous,
                'journal_file_sha256': file_hash(writer.path),
            }
            immutable_json(directory / 'summary.json', {'summary': summary, 'sha256': fingerprint(summary)})
            return summary
        finally:
            writer.close()


def load_replay(output_dir: str | Path) -> dict:
    """Verify complete append chain, terminal anchor, sources and all model versions.

    Hashes detect accidental/casual history alteration, including tail removal;
    they are integrity evidence, not an externally signed tamper-proof ledger.
    """
    directory = project_path(output_dir)
    manifest_record = json.loads((directory / 'manifest.json').read_text('utf-8'))
    summary_record = json.loads((directory / 'summary.json').read_text('utf-8'))
    manifest, summary = manifest_record['manifest'], summary_record['summary']
    if fingerprint(manifest) != manifest_record['sha256'] or fingerprint(summary) != summary_record['sha256']:
        raise ValueError('Replay manifest or summary integrity failure')
    if summary['manifest_sha256'] != manifest_record['sha256']:
        raise ValueError('Replay summary belongs to another manifest')
    journal_path = directory / 'records.jsonl'
    if file_hash(journal_path) != summary['journal_file_sha256']:
        raise ValueError('Replay journal integrity failure')
    records, previous = [], '0' * 64
    previous_time = None
    for sequence, line in enumerate(journal_path.read_text('utf-8').splitlines()):
        record = json.loads(line)
        payload = {k: v for k, v in record.items() if k != 'sha256'}
        timestamp = utc(payload['timestamp'])
        if (payload['sequence'] != sequence or payload['previous'] != previous
                or fingerprint(payload) != record['sha256']
                or (previous_time is not None and timestamp < previous_time)):
            raise ValueError('Replay append chain integrity failure')
        previous, previous_time = record['sha256'], timestamp
        records.append(record)
    if len(records) != summary['journal_records'] or previous != summary['journal_terminal_sha256']:
        raise ValueError('Replay journal terminal anchor mismatch')
    for relative, expected in manifest['source_files'].items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory) or file_hash(path) != expected:
            raise ValueError('Replay source snapshot integrity failure')
    for version in summary['versions']:
        for key, hash_key in (('checkpoint', 'checkpoint_sha256'),
                              ('version_record', 'version_record_sha256')):
            path = (directory / version[key]).resolve()
            if not path.is_relative_to(directory) or file_hash(path) != version[hash_key]:
                raise ValueError('Replay model version integrity failure')
    return {'manifest': manifest, 'summary': summary, 'records': records}
