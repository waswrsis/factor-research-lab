"""Core integration: real completed wallet/models/feed; numerical reuse is not permission.

The only kernel/boot seam is the explicit artificial observation in the existing
core pool fixture. Spies below count calls and always call the actual producer.
"""
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from test_v7_pool_forward import case as case  # noqa: PLC0414
from test_v7_pool_forward import run, ticket_for

from factorlab.discovery import neural_forward_ticket, rolling
from factorlab.discovery import operation_inputs as module
from factorlab.discovery.joint_pool import FrozenPool
from factorlab.discovery.v7_feed import ReceiptArchive


def _bars(case):
    return case.runtime._snapshot('2026-10-01T00:00:00Z')


def test_scope_is_bounded_defensive_and_never_reused_between_operations(case):
    bars = _bars(case)
    with module.OperationInputs() as scope:
        first = scope.build(bars, case.runtime.spec)
        second = scope.build(bars, case.runtime.spec)
        assert scope.stats['builds'] == scope.stats['hits'] == 1
        np.testing.assert_equal(first.frame.values, second.frame.values)
        first.frame.metadata.loc[0, 'symbol'] = 'BROKEN'
        first.frame.values.setflags(write=True)
        first.frame.values[0, 0] = 999
        third = scope.build(bars, case.runtime.spec)
        assert third.frame.metadata.loc[0, 'symbol'] != 'BROKEN'
        assert third.frame.values[0, 0] != 999
        with pytest.raises(ValueError, match='non-nested'), module.OperationInputs():
            pass
    assert scope.retained_bytes == 0
    with pytest.raises(ValueError):
        scope.build(bars)
    with module.OperationInputs() as fresh:
        fresh.build(bars, case.runtime.spec)
        assert fresh.stats['builds'] == 1 and fresh.stats['hits'] == 0
    with module.OperationInputs(max_bytes=1) as small:
        small.build(bars, case.runtime.spec)
        small.build(bars, case.runtime.spec)
        assert small.stats['builds'] == 2 and small.retained_bytes == 0


def test_bytes_attrs_and_full_spec_change_key_even_with_same_object(case):
    bars = _bars(case)
    with module.OperationInputs(max_entries=1) as scope:
        scope.build(bars, case.runtime.spec)
        bars.attrs['audit_scope'] = 'changed'
        scope.build(bars, case.runtime.spec)
        scope.build(bars, replace(case.runtime.spec, ridge=.002))
        bars.loc[0, 'volume'] += 1
        scope.build(bars, case.runtime.spec)
        assert scope.stats['builds'] == 4 and scope.stats['hits'] == 0
        assert scope.stats['evictions'] == 3


def test_actual_three_stream_verification_reuses_only_numeric_build_and_preserves_all_reads(case, monkeypatch):
    runtime, lc = case.runtime, case.runtime.lifecycle
    candidate = FrozenPool(version='input-reuse-candidate')
    pipe = runtime.recipe_for(candidate)
    evidence = {'start': '2026-09-01T00:00:00+00:00', 'end': '2026-09-02T00:00:00+00:00'}
    lc.exposure.record(source=lc.plan.source_id, symbols=lc.plan.symbols,
        batch='input-reuse', role='S', **evidence)
    lc.freeze_batch(batch='input-reuse', ordinal=1, start='2026-11-01T00:00Z',
        candidates=[dict(pipe, complexity=0)], reference=runtime.recipe_for(case.pool),
        evidence_sources=[evidence])
    case.clock.now = pd.Timestamp('2026-11-01T00:00:10Z')
    monkeypatch.setattr(neural_forward_ticket, '_now', case.clock)
    for stream, pool in [('A0', case.pool), ('input-reuse:reference', case.pool),
                         ('input-reuse:' + pipe['pipeline_id'], candidate)]:
        runtime.fit_month('2026-11-01T00:00Z', stream=stream, pool=pool, authority=case.authority)

    counts = {'build': 0, 'archive_verify': 0, 'snapshot': 0, 'source_file_hash': 0}
    def spy(owner, name, counter):
        actual = getattr(owner, name)
        def measured(*args, **kwargs):
            counts[counter] += 1
            return actual(*args, **kwargs)
        monkeypatch.setattr(owner, name, measured)
    spy(rolling, 'build_rolling_data', 'build')
    # Baseline import alias captured before rolling's spy, so give both branches
    # the identical producer wrapper. This changes no returned value.
    monkeypatch.setattr(neural_forward_ticket, 'build_rolling_data', rolling.build_rolling_data)
    spy(ReceiptArchive, 'verify', 'archive_verify')
    spy(ReceiptArchive, 'snapshot', 'snapshot')
    spy(neural_forward_ticket, 'file_hash', 'source_file_hash')
    start = time.perf_counter()
    baseline = [lc.verify(), lc.verify()]
    baseline_seconds = time.perf_counter() - start
    old = dict(counts)
    for name in counts:
        counts[name] = 0
    monkeypatch.setattr(neural_forward_ticket, 'build_rolling_data', module.build_rolling_data)
    start = time.perf_counter()
    with module.OperationInputs() as scope:
        actual = [lc.verify(), lc.verify()]
        metrics = dict(scope.stats)
    scoped_seconds = time.perf_counter() - start
    assert actual == baseline
    assert old['build'] == 6 and counts['build'] == 1
    for name in ('archive_verify', 'snapshot', 'source_file_hash'):
        assert counts[name] == old[name] and old[name] > 0
    assert metrics['hits'] == 5
    assert scope.retained_bytes == 0
    cached_counts = dict(counts)
    before = counts['build']
    with module.OperationInputs():
        lc.verify()
    assert counts['build'] == before + 1
    report = {'scope': 'three actual completed model streams; two Lifecycle.verify calls',
              'baseline': old, 'cached': cached_counts, 'cache': metrics,
              'baseline_seconds': baseline_seconds, 'scoped_seconds': scoped_seconds,
              'kernel': 'synthetic observation, real validators', 'formal': False,
              'note': 'Counts in cached exclude the following fresh-operation check.'}
    (ROOT / 'work/v7g/operation-inputs-core-profile.json').write_text(json.dumps(report, indent=2))


@pytest.mark.parametrize('damage', ['frozen_source_same_stat', 'model_bytes', 'witness_pending', 'job_index'])
def test_live_permissions_and_persistent_bytes_are_rechecked_inside_same_scope(case, monkeypatch, damage):
    result = run(case)
    ticket = ticket_for(case)
    lc = case.runtime.lifecycle
    monkeypatch.setattr(neural_forward_ticket, 'build_rolling_data', module.build_rolling_data)
    with module.OperationInputs():
        lc.verify()
        if damage == 'frozen_source_same_stat':
            path = next((case.runtime.directory / 'source').rglob('*.py'))
            stat = path.stat()
            original = path.read_bytes()
            path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            assert path.stat().st_size == stat.st_size and path.stat().st_mtime_ns == stat.st_mtime_ns
        elif damage == 'model_bytes':
            path = lc.registry.directory / result['model_id'] / 'model.json'
            path.write_text('{}')
        elif damage == 'witness_pending':
            import sqlite3
            path = ticket.path.parent / 'launch/publication_witness/witness.sqlite'
            with sqlite3.connect(path) as db:
                db.execute("UPDATE publication SET status='pending'")
        else:
            lc.db.execute("UPDATE jobs SET stream='borrowed'")
            lc.db.commit()
        with pytest.raises((ValueError, KeyError, FileNotFoundError)):
            lc.verify()
