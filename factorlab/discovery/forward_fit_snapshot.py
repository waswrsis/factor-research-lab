"""Sealed actual-feed inputs for a cold monthly fit.

The receipt archive decides which revisions existed at the cutoff. A worker
cannot substitute a later snapshot, historical availability assumptions, or
unmatured answers. This module grants no training or adoption authority.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from factorlab.protocol import fingerprint
from factorlab.time_utils import calendar_window

from .evidence import file_hash
from .frame_identity import frame_identity
from .nonlinear_orchestration import causal_worker_data, input_data_identity
from .operation_inputs import build_rolling_data
from .rolling import RollingData
from .v7_lifecycle import stamp
from .v7_storage import immutable_record


@dataclass(frozen=True)
class PreparedForwardSnapshot:
    data: RollingData
    record: dict
    _digest: str

    def verify(self):
        value = self.record
        if fingerprint(value) != self._digest:
            raise ValueError('Sealed forward snapshot was modified')
        end = stamp(value['cutoff'])
        if (value['availability'] not in {'actual_receipts', 'quarantined_finite_observation'}
                or self.data.availability_mode != value['data_availability_mode']
                or input_data_identity(self.data, end) != value['input_data_id']
                or self.data.training_identity(end) != value['training']):
            raise ValueError('Forward causal input or training identity changed')
        visible = pd.DatetimeIndex(self.data.frame.metadata.available_at)
        unavailable = ~self.data.observed
        if (not (visible < end).all() or (self.data.target[unavailable] != 0).any()
                or not self.data.maturity[unavailable].isna().all()
                or (self.data.maturity[self.data.observed] > end).any()):
            raise ValueError('Worker input contains a future or unmatured answer')
        return self._digest


def prepare_forward_snapshot(runtime, job):
    """Prepare only the authoritative job; reuse is checked, never re-labelled.

    Caller holds the lifecycle controller lock. Preparing inputs is separate
    from reserving work; the eventual launcher must include this work in its
    original absolute lease and cannot start a new clock after serialization.
    """
    runtime.lifecycle.verify()
    from .v7_run import source_files
    if source_files() != runtime.identity['source_files']:
        raise ValueError('Forward source changed after its frozen identity')
    for relative, digest in runtime.identity['source_files'].items():
        if file_hash(runtime.directory / 'source' / relative) != digest:
            raise ValueError('Frozen forward source archive changed')
    row = runtime.lifecycle.db.execute(
        'SELECT cutoff,pipeline,status FROM jobs WHERE id=?', (job,)).fetchone()
    if row is None or row[2] not in {'planned', 'snapshot_sealed', 'features_ready', 'fitted'}:
        raise ValueError('Snapshot requires an unfinished authoritative fit job')
    start, end = runtime.spec.window(row[0])
    pipeline = json.loads(row[1])
    if pipeline.get('rolling_hash') != runtime.spec.digest:
        raise ValueError('Fit job changed the frozen rolling rule')
    runtime.feed.verify()
    from .quarantined_observation import job_binding, snapshot_for_prefix
    prefix = job_binding(runtime.lifecycle, job)
    bars = (runtime._snapshot(end) if prefix is None else snapshot_for_prefix(
        runtime.lifecycle, prefix, end, context_hours=runtime.config.context_hours))
    if (bars.attrs.get('feed_identity') != runtime.feed.identity
            or bars.attrs.get('snapshot_asof') != end.isoformat()
            or prefix is None and (pd.DatetimeIndex(bars.imported_at) > end).any()):
        raise ValueError('Snapshot lacks actual publication availability at cutoff')
    lineage = []
    for batch in sorted(set(bars.batch_id)):
        found = runtime.feed.db.execute(
            'SELECT b.hash,p.record,p.hash,p.published_at FROM batches b '
            'JOIN publications p ON p.batch=b.id WHERE b.id=?', (batch,)).fetchone()
        physical_end = end if prefix is None else stamp(prefix['physical_observation']['wall'])
        if (found is None or found[1] is None or stamp(found[3]) > physical_end
                or fingerprint(json.loads(found[1])) != found[2]):
            raise ValueError('Snapshot publication lineage is missing or changed')
        lineage.append({'batch': batch, 'batch_hash': found[0],
                        'publication_hash': found[2], 'published_at': found[3]})
    data = causal_worker_data(build_rolling_data(bars, runtime.spec), end)
    raw_hash = frame_identity(bars)
    value = {'version': 'v7-forward-fit-snapshot-v3', 'raw_hash_rule': 'typed-frame-columns-v1', 'job': job,
             'plan_id': runtime.lifecycle.plan.digest, 'pipeline_id': pipeline['pipeline_id'],
             'cutoff': end.isoformat(), 'train_start': start.isoformat(),
             'feed': runtime.feed.identity, 'feed_directory': str(runtime.feed.directory),
             'raw_asof_hash': raw_hash, 'publication_lineage': lineage,
             'availability': 'actual_receipts' if prefix is None else 'quarantined_finite_observation',
             'data_availability_mode': data.availability_mode,
             'input_data_id': input_data_identity(data, end), 'training': data.training_identity(end),
             'raw_schema': list(bars.columns), 'feature_names': list(data.frame.feature_names),
             'bypass_mask': list(data.frame.bypass_mask), 'schema_hash': data.frame.schema_hash,
             'symbols': list(runtime.config.symbols), 'context_hours': runtime.config.context_hours,
             'source_hash': fingerprint(runtime.identity['source_files'])}
    if prefix is not None:
        value['observation_prefix'] = prefix
    value['snapshot_id'] = fingerprint(value)
    path = runtime.directory / 'snapshots' / f'{job}.json'
    immutable_record(path, value)
    prepared = PreparedForwardSnapshot(data, value, fingerprint(value))
    prepared.verify()
    if row[2] == 'planned':
        runtime.lifecycle.advance_fit(job, 'snapshot_sealed', output=value['snapshot_id'])
    if row[2] in {'planned', 'snapshot_sealed'}:
        runtime.lifecycle.advance_fit(job, 'features_ready', output=value['training'])
    return prepared


def verify_prepared_forward_snapshot(runtime, job, prepared):
    """Compare sealed bytes and the actual cutoff source, including publication.

    Re-read uses the same cutoff; it cannot import new data or shift the fit
    window. A failed source check never repairs the stored snapshot.
    """
    if not isinstance(prepared, PreparedForwardSnapshot):
        raise TypeError('A prepared forward snapshot is required')
    prepared.verify()
    stored = json.loads((runtime.directory / 'snapshots' / f'{job}.json').read_text('utf-8'))
    if stored != prepared.record or stored['job'] != job:
        raise ValueError('Prepared snapshot differs from its sealed job')
    repeated = prepare_forward_snapshot(runtime, job)
    if repeated.verify() != prepared.verify():
        raise ValueError('Cutoff data changed after snapshot sealing')
    return prepared.record
