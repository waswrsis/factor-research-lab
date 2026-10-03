import json
import os
import sqlite3
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest

from factorlab import shadow
from factorlab.shadow import ShadowLedger, freeze_bundle, verify_bundle


@pytest.fixture
def artifact(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'model.json').write_text('{"features": ["ret"]}', 'utf-8')
    (source / 'checkpoint').mkdir()
    (source / 'checkpoint' / 'weights.bin').write_bytes(b'checkpoint')
    (source / 'empty').mkdir()
    return source


def test_freeze_snapshot_idempotence_and_source_independence(artifact, tmp_path):
    frozen = tmp_path / 'frozen'
    result = freeze_bundle(artifact, frozen, {'protocol_hash': 'a' * 64})
    assert len(result['bundle_hash']) == 64
    assert result == verify_bundle(frozen)
    assert result == freeze_bundle(artifact, frozen, {'protocol_hash': 'a' * 64})
    (artifact / 'checkpoint' / 'weights.bin').write_bytes(b'new')
    assert verify_bundle(frozen) == result
    with pytest.raises(ValueError, match='differs'):
        freeze_bundle(artifact, frozen, {'protocol_hash': 'a' * 64})
    assert verify_bundle(frozen) == result


@pytest.mark.parametrize('tamper', ['file', 'manifest', 'extra_file', 'extra_dir', 'missing'])
def test_verify_bundle_detects_tamper(artifact, tmp_path, tamper):
    path = tmp_path / 'frozen'
    freeze_bundle(artifact, path, {})
    if tamper == 'file':
        (path / 'model.json').write_text('changed', 'utf-8')
    elif tamper == 'manifest':
        manifest = json.loads((path / shadow.MANIFEST).read_text('utf-8'))
        manifest['metadata']['changed'] = True
        (path / shadow.MANIFEST).write_text(json.dumps(manifest), 'utf-8')
    elif tamper == 'extra_file':
        (path / 'extra').write_bytes(b'x')
    elif tamper == 'extra_dir':
        (path / 'extra').mkdir()
    else:
        (path / 'model.json').unlink()
    with pytest.raises(ValueError):
        verify_bundle(path)
    with pytest.raises(ValueError):
        freeze_bundle(artifact, path, {})


def test_freeze_rejects_overlap_reserved_name_and_non_json(artifact, tmp_path):
    with pytest.raises(ValueError, match='disjoint'):
        freeze_bundle(artifact, artifact / 'frozen', {})
    with pytest.raises(ValueError, match='finite JSON'):
        freeze_bundle(artifact, tmp_path / 'frozen', {'bad': float('nan')})
    (artifact / shadow.MANIFEST).write_text('{}', 'utf-8')
    with pytest.raises(ValueError, match='reserved'):
        freeze_bundle(artifact, tmp_path / 'frozen', {})


def test_freeze_rejects_paths_outside_workspace(artifact):
    with pytest.raises(ValueError, match='within'):
        freeze_bundle(artifact, 'C:/outside-shadow-test', {})


def test_links_rejected_without_following(artifact, tmp_path, monkeypatch):
    # Windows symlink creation often requires administrator privileges. Simulate
    # the filesystem link classification to exercise fail-closed checks portably.
    original = shadow.Path.is_symlink
    monkeypatch.setattr(shadow.Path, 'is_symlink',
                        lambda p: p.name == 'weights.bin' or original(p))
    with pytest.raises(ValueError, match='Links'):
        freeze_bundle(artifact, tmp_path / 'frozen', {})


@pytest.mark.skipif(os.name != 'nt', reason='Windows junction boundary')
def test_real_windows_junction_rejected(artifact, tmp_path):
    target = tmp_path / 'target'
    target.mkdir()
    (target / 'file').write_bytes(b'x')
    link = artifact / 'junction'
    result = subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(target)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert link.is_junction()
    with pytest.raises(ValueError, match='Links'):
        freeze_bundle(artifact, tmp_path / 'frozen', {})
    with pytest.raises(ValueError, match='Links'):
        freeze_bundle(link, tmp_path / 'frozen', {})


@pytest.fixture
def ledger(tmp_path):
    return ShadowLedger(tmp_path / 'ledger', 'a' * 64, 'b' * 64,
                        observation_mode='historical_replay')


def times():
    available = pd.Timestamp('2025-01-01T10:00:05Z')
    entry = pd.Timestamp('2025-01-01T11:00:00Z')
    end = entry + pd.Timedelta(hours=12)
    return available, entry, end


def add(ledger, decision='d1', symbol='BTCUSDT', **changes):
    available, entry, end = times()
    args = {'decision_id': decision, 'symbol': symbol, 'available_at': available,
            'entry_time': entry, 'label_end': end, 'prediction': 0.02, 'now': available}
    args.update(changes)
    return ledger.append_prediction(**args)


def test_explicit_mode_and_immutable_model_protocol_config(tmp_path, ledger):
    with pytest.raises(TypeError):
        ShadowLedger(tmp_path / 'missing', 'a' * 64, 'b' * 64)
    for changes in ({'observation_mode': 'forward'}, {'protocol_hash': 'c' * 64},
                    {'model_hash': 'c' * 64}, {'availability_delay_seconds': 6}):
        args = {'directory': ledger.directory, 'protocol_hash': 'a' * 64,
                'model_hash': 'b' * 64, 'observation_mode': 'historical_replay'}
        args.update(changes)
        with pytest.raises(ValueError, match='differs'):
            ShadowLedger(**args)


def test_prediction_record_units_duplicates_and_durability(ledger):
    result = add(ledger)
    assert result['unit'] == '12h_simple_return'
    assert result['observation_mode'] == 'historical_replay'
    with pytest.raises(ValueError, match='Duplicate'):
        add(ledger)
    with pytest.raises(ValueError, match='Duplicate'):
        add(ledger, decision='other')
    reopened = ShadowLedger(ledger.directory, 'a' * 64, 'b' * 64,
                            observation_mode='historical_replay')
    assert reopened.summary(times()[0])['predictions'] == 1


@pytest.mark.parametrize('change', [
    {'now': '2025-01-01T10:00:04Z'},
    {'entry_time': '2025-01-01T10:00:05Z'},
    {'entry_time': '2025-01-01T12:00:00Z'},
    {'label_end': '2025-01-01T22:00:00Z'},
    {'prediction': float('nan')}, {'prediction': -1.01},
    {'available_at': '2025-01-01T10:00:05'},
    {'decision_id': ''}, {'symbol': ''},
])
def test_prediction_rejects_invalid_causal_contract(ledger, change):
    with pytest.raises(ValueError):
        add(ledger, **change)
    assert ledger.summary(times()[0])['predictions'] == 0


def test_whole_hour_availability_requires_strictly_later_hour(ledger):
    add(ledger, available_at='2025-01-01T10:00:00Z', now='2025-01-01T10:00:00Z')
    assert ledger.summary(times()[0])['predictions'] == 1


def test_maturity_boundary_mature_only_scores_and_duplicate_outcome(ledger):
    add(ledger)
    _, _, end = times()
    mature = end + pd.Timedelta(seconds=5)
    assert ledger.score(mature - pd.Timedelta(nanoseconds=1))['unmature'] == 1
    with pytest.raises(ValueError, match='not matured'):
        ledger.append_outcome('d1', 0.01, mature, mature - pd.Timedelta(nanoseconds=1))
    assert ledger.score(mature)['mature_without_outcome'] == 1
    assert ledger.score(mature)['mae'] is None
    with pytest.raises(ValueError, match='availability differs'):
        ledger.append_outcome('d1', 0.01, end, mature)
    ledger.append_outcome('d1', 0.01, mature, mature)
    result = ledger.score(mature)
    assert result['mature'] == 1 and result['invalid'] == result['unmature'] == 0
    assert result['mae'] == pytest.approx(0.01)
    assert result['mse'] == pytest.approx(0.0001)
    with pytest.raises(ValueError, match='Duplicate outcome'):
        ledger.append_outcome('d1', 0.01, mature, mature)
    with pytest.raises(ValueError, match='existing prediction'):
        ledger.append_outcome('unknown', 0.01, mature, mature)
    with pytest.raises(ValueError, match='backward'):
        ledger.score(times()[0])


def test_summary_clock_cannot_move_backwards_and_invalid_rows_are_excluded(ledger):
    add(ledger)
    available, entry, end = times()
    later = available + pd.Timedelta(minutes=1)
    ledger.summary(later)
    with pytest.raises(ValueError, match='backward'):
        ledger.summary(available)
    # Direct database inserts bypass the application contract; the summary
    # nevertheless excludes an invalid horizon instead of giving it a score.
    with sqlite3.connect(ledger.path) as db:
        db.execute('INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?)',
                   ('bad', 'ETHUSDT', available.value, entry.value, end.value + 1,
                    0.02, later.value))
    result = ledger.summary(later)
    assert result['invalid'] == 1 and result['mature'] == 0


def test_ledger_respects_nondefault_frozen_delay(tmp_path):
    ledger = ShadowLedger(tmp_path / 'custom', 'a' * 64, 'b' * 64,
                          observation_mode='historical_replay', availability_delay_seconds=7)
    add(ledger)
    mature = times()[2] + pd.Timedelta(seconds=7)
    with pytest.raises(ValueError, match='availability differs'):
        ledger.append_outcome('d1', 0.01, mature - pd.Timedelta(seconds=2), mature)
    ledger.append_outcome('d1', 0.01, mature, mature)
    assert ledger.score(mature)['mature'] == 1


def test_append_only_sql_triggers(ledger):
    add(ledger)
    with sqlite3.connect(ledger.path) as db:
        for statement in ('UPDATE predictions SET prediction=0', 'DELETE FROM predictions',
                          'UPDATE metadata SET config="{}"', 'DELETE FROM metadata'):
            with pytest.raises(sqlite3.IntegrityError, match='append-only'):
                db.execute(statement)


def test_forward_rejects_old_replay_future_now_and_late_entry(tmp_path, monkeypatch):
    wall = pd.Timestamp('2025-01-01T10:30:00Z')
    monkeypatch.setattr(shadow, '_wall_now', lambda: wall)
    ledger = ShadowLedger(tmp_path / 'forward', 'a' * 64, 'b' * 64,
                          observation_mode='forward')
    with pytest.raises(ValueError, match='actual wall clock'):
        add(ledger, now='2024-01-01T10:00:05Z')
    with pytest.raises(ValueError, match='future'):
        add(ledger, now=wall + pd.Timedelta(seconds=1))
    with pytest.raises(ValueError, match='one hour old'):
        add(ledger, available_at='2025-01-01T09:30:00Z', entry_time='2025-01-01T10:00:00Z',
            label_end='2025-01-01T22:00:00Z', now=wall)
    with pytest.raises(ValueError, match='precede'):
        add(ledger, available_at='2025-01-01T09:59:59Z', entry_time='2025-01-01T10:00:00Z',
            label_end='2025-01-01T22:00:00Z', now=wall)
    result = add(ledger, now=wall)
    assert result['observation_mode'] == 'forward'


def test_historical_clock_monotonic_and_future_now_rejected(ledger, monkeypatch):
    wall = times()[0] + pd.Timedelta(minutes=20)
    monkeypatch.setattr(shadow, '_wall_now', lambda: wall)
    add(ledger, now=wall)
    with pytest.raises(ValueError, match='backward'):
        add(ledger, decision='d2', symbol='ETHUSDT', now=times()[0])
    with pytest.raises(ValueError, match='future'):
        ledger.score(wall + pd.Timedelta(nanoseconds=1))


def test_concurrent_duplicate_submissions_commit_once(ledger):
    def attempt(_):
        try:
            add(ledger)
            return True
        except ValueError:
            return False
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(attempt, range(4))) == 1
    assert ledger.summary(times()[0])['predictions'] == 1
