"""Synthetic run dossiers exercise integrity; they are not market evidence."""
import json
import sqlite3

import pytest

from factorlab.discovery.catalog import QualityCatalog
from factorlab.discovery.contracts import ExperimentSpec
from factorlab.discovery.evidence import (
    ExperimentJournal,
    file_hash,
    freeze_claim,
    immutable_json,
)
from factorlab.protocol import fingerprint


def make_run(tmp_path, *, confirmed=False, system='supported', representation='target_not_met'):
    directory = tmp_path / ('confirmed' if confirmed else 'development')
    directory.mkdir()
    spec = ExperimentSpec()
    immutable_json(directory / 'protocol.json', {'protocol': spec.to_dict(), 'sha256': spec.digest})
    source = directory / 'source'
    source.mkdir()
    (source / 'small.py').write_text('# synthetic fixture, not research evidence', encoding='utf-8')
    (source / 'uv.lock').write_text('version = 1\nrevision = 1\n', encoding='utf-8')
    manifest = {'small.py': file_hash(source / 'small.py'), 'uv.lock': file_hash(source / 'uv.lock')}
    immutable_json(source / 'manifest.json', manifest)
    bundle = directory / 'bundle'
    for name in ('candidate', 'baseline'):
        target = bundle / name
        target.mkdir(parents=True)
        (target / 'seed-0').write_bytes(b'synthetic fitted-state fixture')
        immutable_json(target / 'group.json', {'members': 1, 'digest': fingerprint(name), 'schema': 'fixture'})
    selection = {'arm': 'single', 'candidate_digest': fingerprint('candidate'),
                 'baseline_digest': fingerprint('baseline'), 'selected_on': 'development_primary_loss',
                 'statement': 'synthetic test fixture only'}
    immutable_json(bundle / 'selection.json', selection)
    immutable_json(bundle / 'probe_contract.json', {'generator_digest': selection['candidate_digest']})
    claim = freeze_claim(directory, spec, bundle, fingerprint('synthetic-data'), fingerprint(manifest),
                         ['system_vs_baseline', 'representation_addition', 'matched_dimension_control'])
    result = {'status': 'confirmation_complete' if confirmed else 'development_complete',
              'run_directory': str(directory), 'protocol_hash': spec.digest, 'claim_hash': claim['digest'],
              'selection': selection, 'primary_task': spec.primary_task, 'confirmation': None,
              'evidence_source': 'synthetic-unit-test', 'elapsed_seconds': 1.2,
              'arms': {'single': {'parameter_count': 12},
                       'search_random': {'attempts': 2, 'elapsed_seconds': .5, 'budget_seconds': 3}},
              'limitations': ['Synthetic unit test; not independent market evidence']}
    journal = ExperimentJournal(directory / 'journal.jsonl')
    journal.append('development_fit', {'arm': 'single'})
    if confirmed:
        comparisons = {'system_vs_baseline': {'status': system, 'delta': .05},
                       'representation_addition': {'status': representation, 'delta': .01},
                       'matched_dimension_control': {'status': representation, 'delta': .005}}
        predictions = {}
        for key in ('candidate', 'baseline'):
            path = directory / f'{key}-scores.parquet'
            path.write_bytes(b'synthetic prediction artifact')
            predictions[key] = {'path': str(path), 'sha256': file_hash(path), 'rows': 8}
        result['confirmation'] = {
            'comparisons': comparisons, 'accepted_system': system == 'supported',
            'supported_representation': representation == 'supported',
            'candidate_metrics': {'volatility': {'coverage': 1.0}},
            'candidate_predictions': predictions['candidate'], 'baseline_predictions': predictions['baseline']}
        journal.append('confirmation_consumed', {'claim_hash': claim['digest']})
        journal.append('confirmation_result', result['confirmation'])
    immutable_json(directory / 'results.json', result)
    journal.append('completed', {'result_hash': file_hash(directory / 'results.json')})
    return directory


def test_register_get_list_verify_export_all_tmp_isolated(tmp_path):
    catalog = QualityCatalog(tmp_path / 'catalog.sqlite')
    directory = make_run(tmp_path)
    record = catalog.register(directory)
    assert record['evidence_state'] == 'development_only'
    assert record['system_conclusion']['status'] == 'unconfirmed'
    assert record['representation_conclusion']['status'] == 'unconfirmed'
    assert record['generator_identity'] == fingerprint('candidate')
    assert record['cost']['arms']['search_random']['attempts'] == 2
    assert 'source/uv.lock' in record['artifacts']
    assert catalog.get(record['entry_id']) == record
    assert catalog.list(task='volatility', state='development_only') == [record]
    assert catalog.list(task='absent') == []
    assert catalog.verify(record['entry_id'])['valid']
    export = catalog.export(tmp_path / 'export.json')
    payload = json.loads(export.read_text('utf-8'))
    assert payload['entries'] == [record]
    assert payload['sha256'] == fingerprint([record])
    with pytest.raises(KeyError):
        catalog.get('missing')


def test_snapshot_keeps_uv_lock_and_only_ignores_manifest_lock(tmp_path):
    catalog = QualityCatalog(tmp_path / 'catalog.sqlite')
    directory = make_run(tmp_path)
    (directory / 'source' / 'manifest.json.lock').write_text('generated lock', encoding='utf-8')
    record = catalog.register(directory)
    assert 'source/uv.lock' in record['artifacts']
    assert 'source/manifest.json.lock' not in record['artifacts']
    assert catalog.verify(record['entry_id'])['valid']
    (directory / 'source' / 'uv.lock').write_text('tampered dependency resolution', encoding='utf-8')
    with pytest.raises(ValueError, match='artifact changed'):
        catalog.verify(record['entry_id'])


def test_confirmation_system_and_representation_remain_independent(tmp_path):
    catalog = QualityCatalog(tmp_path / 'catalog.sqlite')
    record = catalog.register(make_run(tmp_path, confirmed=True))
    assert record['evidence_state'] == 'confirmation_complete'
    assert record['system_conclusion']['status'] == 'supported'
    assert record['representation_conclusion']['status'] == 'target_not_met'
    assert record['task_scope']['coverage']['volatility']['coverage'] == 1
    assert catalog.verify(record['entry_id'])['valid']


def test_entries_cannot_be_overwritten_or_deleted(tmp_path):
    catalog = QualityCatalog(tmp_path / 'catalog.sqlite')
    directory = make_run(tmp_path)
    record = catalog.register(directory)
    with pytest.raises(ValueError, match='already registered'):
        catalog.register(directory)
    with sqlite3.connect(catalog.path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match='immutable'):
            connection.execute('UPDATE entries SET primary_task=?', ('changed',))
        with pytest.raises(sqlite3.IntegrityError, match='immutable'):
            connection.execute('DELETE FROM entries')
    assert catalog.get(record['entry_id']) == record


@pytest.mark.parametrize('artifact', ['bundle/candidate/seed-0', 'source/small.py',
                                     'results.json', 'frozen_claim.json', 'journal.jsonl'])
def test_catalog_detects_changed_artifact(tmp_path, artifact):
    catalog = QualityCatalog(tmp_path / 'catalog.sqlite')
    directory = make_run(tmp_path)
    record = catalog.register(directory)
    (directory / artifact).write_bytes(b'tampered')
    with pytest.raises(ValueError, match='artifact changed'):
        catalog.verify(record['entry_id'])
    with pytest.raises(ValueError):
        catalog.export(tmp_path / 'invalid-export.json')


def test_reject_incomplete_journal_and_recomputed_result_without_completed_binding(tmp_path):
    catalog = QualityCatalog(tmp_path / 'catalog.sqlite')
    directory = make_run(tmp_path)
    journal = directory / 'journal.jsonl'
    rows = journal.read_text('utf-8').splitlines()
    journal.write_text(rows[0] + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='completed hash-chain'):
        catalog.register(directory)
    journal.write_text('\n'.join(rows) + '\n', encoding='utf-8')
    result_path = directory / 'results.json'
    payload = json.loads(result_path.read_text('utf-8'))
    payload['elapsed_seconds'] = 2
    result_path.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ValueError, match='bind current results'):
        catalog.register(directory)


def test_reject_hash_chain_tampering_before_registration(tmp_path):
    catalog = QualityCatalog(tmp_path / 'catalog.sqlite')
    directory = make_run(tmp_path)
    path = directory / 'journal.jsonl'
    lines = path.read_text('utf-8').splitlines()
    payload = json.loads(lines[0])
    payload['value']['arm'] = 'changed'
    lines[0] = json.dumps(payload)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='integrity'):
        catalog.register(directory)


def test_reject_added_bundle_member_and_missing_prediction(tmp_path):
    catalog = QualityCatalog(tmp_path / 'catalog.sqlite')
    directory = make_run(tmp_path, confirmed=True)
    record = catalog.register(directory)
    (directory / 'bundle' / 'unexpected').write_text('changed scope', encoding='utf-8')
    with pytest.raises(ValueError, match='bundle was changed'):
        catalog.verify(record['entry_id'])
    (directory / 'bundle' / 'unexpected').unlink()
    (directory / 'candidate-scores.parquet').unlink()
    with pytest.raises(ValueError, match='missing or invalid'):
        catalog.verify(record['entry_id'])
