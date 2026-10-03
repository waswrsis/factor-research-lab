import json
from dataclasses import asdict

import pytest

from factorlab.discovery import evidence
from factorlab.discovery.contracts import ExperimentSpec
from factorlab.discovery.evidence import (
    EvidenceRegistry,
    ExperimentJournal,
    freeze_claim,
    immutable_json,
    verify_bundle,
)
from factorlab.protocol import fingerprint


@pytest.fixture(autouse=True)
def isolated_known_runs(monkeypatch, tmp_path):
    root = tmp_path / 'known_runs'
    root.mkdir()
    outputs = tmp_path / 'known_outputs'
    outputs.mkdir()
    monkeypatch.setattr(evidence, 'KNOWN_RUNS_ROOT', root)
    monkeypatch.setattr(evidence, 'KNOWN_OUTPUTS_ROOT', outputs)
    return root


def test_seen_prices_cannot_become_new_evidence_by_changing_claim(tmp_path):
    registry = EvidenceRegistry(tmp_path / 'registry.sqlite')
    record = {'source': 'binance_um_1h', 'symbols': ('BTCUSDT',), 'start': '2025-10-01',
              'end': '2026-01-01', 'run': 'one', 'claim': 'volatility'}
    registry.consume(**record)
    with pytest.raises(ValueError, match='Previously exposed'):
        registry.consume(**{**record, 'run': 'two', 'claim': 'event'})
    assert len(registry.records()) == 1
    registry.consume(**{**record, 'start': '2026-01-01', 'end': '2026-02-01'})
    assert len(registry.records()) == 2


def test_partial_symbol_overlap_rolls_back_whole_claim(tmp_path):
    registry = EvidenceRegistry(tmp_path / 'registry.sqlite')
    record = {'source': 'feed', 'symbols': ('BTC',), 'start': '2025-01-01',
              'end': '2025-02-01', 'run': 'first', 'claim': 'a'}
    registry.consume(**record, role='development')
    with pytest.raises(ValueError):
        registry.consume(**{**record, 'symbols': ('ETH', 'BTC'), 'run': 'second'})
    assert len(registry.records()) == 1


def test_journal_detects_changed_history(tmp_path):
    journal = ExperimentJournal(tmp_path / 'journal.jsonl')
    journal.append('prediction', {'p': 0.3})
    journal.append('settlement', {'y': 1})
    assert len(journal.read()) == 2
    value = journal.path.read_text().replace('0.3', '0.9')
    journal.path.write_text(value)
    with pytest.raises(ValueError, match='integrity'):
        journal.read()


def test_freeze_covers_model_and_calibration(tmp_path):
    bundle = tmp_path / 'bundle'
    immutable_json(bundle / 'model.json', {'weights': [1, 2]})
    immutable_json(bundle / 'calibration.json', {'temperature': 2.0})
    claim = freeze_claim(tmp_path, ExperimentSpec(), bundle, 'data', 'source', ['system'])
    verify_bundle(bundle, claim)
    (bundle / 'calibration.json').write_text(json.dumps({'temperature': 1.0}))
    with pytest.raises(ValueError, match='changed'):
        verify_bundle(bundle, claim)


def test_immutable_record_rejects_rewriting(tmp_path):
    path = tmp_path / 'record.json'
    immutable_json(path, {'a': 1})
    immutable_json(path, {'a': 1})
    with pytest.raises(ValueError, match='Immutable'):
        immutable_json(path, {'a': 2})


def prior_run(root, *, name='nested/failed', consumed=False, completed=False):
    directory = root / name
    directory.mkdir(parents=True)
    spec = ExperimentSpec(symbols=('BTCUSDT',))
    (directory / 'protocol.json').write_text(json.dumps({'protocol': asdict(spec)}), encoding='utf-8')
    journal = ExperimentJournal(directory / 'journal.jsonl')
    journal.append('development_fit', {'arm': 'baseline'})
    if consumed:
        journal.append('confirmation_consumed', {'claim_hash': 'frozen_claim'})
    journal.append('failed', {'status': 'process_invalid'})
    (directory / 'failure.json').write_text(json.dumps({'status': 'process_invalid'}), encoding='utf-8')
    if completed:
        (directory / 'results.json').write_text(json.dumps({'confirmation': {'accepted_system': False}}), encoding='utf-8')
    return directory, spec


def test_replaced_database_recovers_failed_confirmation_consumption_from_valid_journal(isolated_known_runs, tmp_path):
    directory, spec = prior_run(isolated_known_runs, consumed=True)
    registry = EvidenceRegistry(tmp_path / 'replacement.sqlite')
    registry.import_prior_runs()
    rows = registry.records()
    assert len(rows) == 1
    assert rows[0]['end'] == evidence.utc(spec.end).isoformat()
    assert rows[0]['source'] == 'binance_um_1h' and rows[0]['role'] == 'legacy_seen'
    assert rows[0]['run'] == str(directory)
    with pytest.raises(ValueError, match='Previously exposed'):
        registry.consume(source='binance_um_1h', symbols=spec.symbols,
                         start=spec.calibration_end, end=spec.end, run='reexported', claim='renamed_data')
    registry.import_prior_runs()
    assert len(registry.records()) == 1  # Recovery is idempotent.


def test_failed_unconsumed_run_conservatively_recovers_development_and_calibration(isolated_known_runs, tmp_path):
    _, spec = prior_run(isolated_known_runs)
    registry = EvidenceRegistry(tmp_path / 'replacement.sqlite')
    registry.import_prior_runs()
    assert registry.records()[0]['end'] == evidence.utc(spec.calibration_end).isoformat()
    assert registry.records()[0]['start'] == (
        evidence.utc(spec.start) - evidence.pd.Timedelta(hours=spec.lookback + 48)
    ).isoformat()
    # Confirmation remains unused when no consumption record exists.
    registry.consume(source='binance_um_1h', symbols=spec.symbols,
                     start=spec.calibration_end, end=spec.end, run='new', claim='unseen')


def test_corrupted_failed_run_journal_cannot_restore_unseen_status(isolated_known_runs, tmp_path):
    directory, _ = prior_run(isolated_known_runs, consumed=True)
    path = directory / 'journal.jsonl'
    path.write_text(path.read_text('utf-8').replace('confirmation_consumed', 'development_only'), encoding='utf-8')
    registry = EvidenceRegistry(tmp_path / 'replacement.sqlite')
    with pytest.raises(ValueError, match='integrity'):
        registry.import_prior_runs()
    assert not registry.records()


def test_completed_rejected_confirmation_is_still_recovered(isolated_known_runs, tmp_path):
    directory, spec = prior_run(isolated_known_runs, completed=True)
    (directory / 'journal.jsonl').unlink()  # Results alone retain the rejected attempt.
    registry = EvidenceRegistry(tmp_path / 'replacement.sqlite')
    registry.import_prior_runs()
    assert registry.records()[0]['end'] == evidence.utc(spec.end).isoformat()


@pytest.mark.parametrize('wrapped', [False, True])
def test_replay_and_score_exposure_manifests_restore_warmup_and_source(isolated_known_runs, tmp_path, wrapped):
    directory = isolated_known_runs / ('nested/replay' if wrapped else 'nested/score')
    directory.mkdir(parents=True)
    exposure = {'intervals': [{'symbol': 'BTCUSDT', 'start': '2025-01-01', 'end': '2025-03-01'},
                              {'symbol': 'ETHUSDT', 'start': '2025-01-02', 'end': '2025-03-01'}]}
    if wrapped:
        exposure['source'] = 'explicit_market_source'
    manifest = {'evidence_exposure': exposure, 'task_schema': 'source_and_time_identity_not_filename'}
    payload = {'manifest': manifest, 'sha256': fingerprint(manifest)} if wrapped else manifest
    (directory / 'manifest.json').write_text(json.dumps(payload), encoding='utf-8')
    registry = EvidenceRegistry(tmp_path / 'replacement.sqlite')
    registry.import_prior_runs()
    assert len(registry.records()) == 2
    assert {row['source'] for row in registry.records()} == {
        'explicit_market_source' if wrapped else 'binance_um_1h'}
    with pytest.raises(ValueError, match='Previously exposed'):
        registry.consume(source='explicit_market_source' if wrapped else 'binance_um_1h',
                         symbols=('BTCUSDT',), start='2025-02-01', end='2025-03-01',
                         run='renamed_export', claim='different_task')


def test_source_snapshot_protocols_are_not_mistaken_for_actual_runs(isolated_known_runs, tmp_path):
    prior_run(isolated_known_runs, name='real/source/copied_run', consumed=True)
    registry = EvidenceRegistry(tmp_path / 'replacement.sqlite')
    registry.import_prior_runs()
    assert not registry.records()


def test_tampered_wrapped_exposure_manifest_is_rejected(isolated_known_runs, tmp_path):
    directory = isolated_known_runs / 'replay'
    directory.mkdir()
    manifest = {'evidence_exposure': {'intervals': [
        {'symbol': 'BTCUSDT', 'start': '2025-01-01', 'end': '2025-02-01'}]}}
    (directory / 'manifest.json').write_text(json.dumps({'manifest': manifest, 'sha256': '0' * 64}), encoding='utf-8')
    with pytest.raises(ValueError, match='integrity'):
        EvidenceRegistry(tmp_path / 'replacement.sqlite').import_prior_runs()


def test_score_in_known_outputs_tree_is_recovered_without_scanning_actual_outputs(isolated_known_runs, tmp_path):
    directory = evidence.KNOWN_OUTPUTS_ROOT / 'scores/nested'
    directory.mkdir(parents=True)
    manifest = {'evidence_exposure': {'source': 'binance_um_1h', 'intervals': [
        {'symbol': 'BTCUSDT', 'start': '2025-02-01', 'end': '2025-04-01'}]}}
    (directory / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    registry = EvidenceRegistry(tmp_path / 'replacement.sqlite')
    registry.import_prior_runs()
    assert len(registry.records()) == 1
    assert registry.records()[0]['run'] == str(directory)
