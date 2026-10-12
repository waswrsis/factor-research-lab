"""Synthetic artifact integrity, canonical hashes and bounded ledger verification."""
import gc
import json
import tracemalloc
from copy import deepcopy

import pytest
from test_discovery_evolution import fixture, run

from factorlab.discovery import population
from factorlab.discovery.compare_evolution import _read, _resources
from factorlab.discovery.evidence import file_hash, immutable_json
from factorlab.discovery.evolution import EvolutionPolicy, EvolutionStore
from factorlab.discovery.evolution_report import export_evolution
from factorlab.protocol import fingerprint


def large_result():
    return {'method': 'synthetic_large_search', 'elapsed_seconds': .25,
            'resource_scope': 'synthetic wall time', 'newly_fitted': 80,
            'development_memory': [{'candidate_id': str(i), 'status': 'ok'} for i in range(30)],
            'records': [{'proposal': {'operation': 'write_program'}, 'status': 'ok',
                         'diagnostics': '证据' * 3000} for _ in range(80)]}


def initialize_state(store):
    identity = {'scope': 'synthetic_legacy_verification_fixture'}
    immutable_json(store.directory / 'manifest.json',
                   {'manifest': identity, 'sha256': fingerprint(identity)})
    state = {'identity': identity, 'last_timestamp': '2024-01-01', 'ticks': 0}
    store.save_state(state)


def test_streaming_table_hash_is_exact_legacy_fingerprint(tmp_path):
    store = EvolutionStore(tmp_path)
    try:
        for table in ('cycles', 'models'):
            assert store._table_hash(table) == fingerprint([])
            for index in range(80):
                value = json.dumps({'unicode': '中\n\\"' * 500, 'nested': [None, True, 1.25]},
                                   ensure_ascii=False)
                store.db.execute(f'INSERT INTO {table} VALUES (?,?)', (index, value))
            expected = fingerprint([dict(row) for row in store.db.execute(f'SELECT * FROM {table} ORDER BY id')])
            assert store._table_hash(table) == expected
        with pytest.raises(ValueError, match='Unexpected'):
            store._table_hash('predictions')
    finally:
        store.close()


def test_large_body_written_once_with_bounded_cycle_and_event_summaries(tmp_path):
    store = EvolutionStore(tmp_path)
    body = large_result()
    original = deepcopy(body)
    try:
        cycle = {'id': 1, 'status': 'unchanged_retained', 'research': body}
        store.put_cycle(cycle)
        store.event('candidate_started', '2024-01-01', cycle)
        store.event('unchanged_retained', '2024-01-02', cycle)
        store.put_cycle({'id': 2, 'research': body})
        summary = store.cycle(1)['research']
        assert len(list((tmp_path / 'research-objects').glob('*.json'))) == 1
        assert summary['_research_ref']['sha256'] == fingerprint(body)
        assert summary['method'] == body['method'] and summary['elapsed_seconds'] == .25
        assert summary['operations'] == ['write_program'] and summary['records_count'] == 80
        assert summary['development_memory_count'] == 30 and len(summary['development_memory']) == 4
        assert 'records' not in summary
        assert len(json.dumps(cycle).encode()) < 8000
        assert all(len(row['value'].encode()) < 9000 for row in store.db.execute('SELECT value FROM events'))
        assert store.load_research(summary) == original and body == original
        assert file_hash(tmp_path / summary['_research_ref']['path']) == fingerprint(body)
        initialize_state(store)
        assert store.verify(compact=True)['event_count'] == 3
    finally:
        store.close()


@pytest.mark.parametrize('change', ['missing', 'tampered', 'traversal', 'wrong_hash'])
def test_research_reference_integrity_rejected_on_read_and_verification(tmp_path, change):
    store = EvolutionStore(tmp_path)
    try:
        cycle = {'id': 1, 'research': large_result()}
        store.put_cycle(cycle)
        research = cycle['research']
        path = tmp_path / research['_research_ref']['path']
        if change == 'missing':
            path.unlink()
        elif change == 'tampered':
            path.write_text('{}', encoding='utf-8')
        elif change == 'traversal':
            research['_research_ref']['path'] = '../outside.json'
        else:
            research['_research_ref']['sha256'] = '0' * 64
        # Event chain remains well formed; rejecting the artifact is a separate check.
        store.event('candidate_started', '2024-01-01', cycle)
        initialize_state(store)
        with pytest.raises(ValueError, match='research artifact|Research artifact'):
            store.load_research(research)
        with pytest.raises(ValueError, match='research artifact|Research artifact'):
            store.verify(compact=True)
        assert not list(tmp_path.glob('.verify-*'))
    finally:
        store.close()


def test_legacy_inline_research_and_default_event_list_remain_readable(tmp_path):
    store = EvolutionStore(tmp_path)
    try:
        small = {'method': 'old', 'records': [{'status': 'ok'}]}
        cycle = {'id': 1, 'research': small}
        store.put_cycle(cycle)
        assert cycle['research'] == small and not (tmp_path / 'research-objects').exists()
        # Simulate an old large inline record without rewriting its table/event hashes.
        old = {'id': 2, 'research': large_result()}
        store.db.execute('INSERT INTO cycles VALUES (?,?)', (2, json.dumps(old)))
        store.event('old_inline', '2024-01-01', old)
        initialize_state(store)
        assert store.cycle(2) == old
        verified = store.verify()
        assert isinstance(verified, list) and verified == store.events()
        assert verified[0]['value'] == old
        assert store.verify(compact=True) == store.events(compact=True)
    finally:
        store.close()


def test_compact_verification_replays_on_disk_and_does_not_collect_full_events(tmp_path):
    store = EvolutionStore(tmp_path)
    try:
        for index in range(600):
            row = {'sample': str(index), 'task': 'return', 'model': 'model',
                   'origin': '2024-01-01', 'value': '0.0', 'role': 'incumbent',
                   'cycle': None, 'symbol': 'TEST'}
            store.db.execute('INSERT INTO predictions(sample,task,model,origin,value,role,cycle,symbol) '
                             'VALUES(?,?,?,?,?,?,?,?)', tuple(row.values()))
            store.event('forecast_batch', '2024-01-01', {'records': [row], 'padding': 'x' * 5000})
            settled = {key: row[key] for key in ('sample', 'task', 'model')}
            settled.update(settled=1, loss=0.0, target=0.0, status='scored')
            store.db.execute('UPDATE predictions SET settled=1,loss=0.0,target=0.0,status=? WHERE sample=?',
                             ('scored', str(index)))
            store.event('settlement', '2024-01-02', {'records': [settled]})
        initialize_state(store)
        gc.collect()
        tracemalloc.start()
        full = store.verify()
        _, full_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        expected = {'event_count': len(full), 'event_head': full[-1]['sha256']}
        del full
        gc.collect()
        tracemalloc.start()
        compact = store.verify(compact=True)
        _, compact_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert compact == expected
        assert compact_peak < full_peak / 4
        assert not list(tmp_path.glob('.verify-*'))
        store.db.execute('UPDATE predictions SET loss=123 WHERE sample=?', ('0',))
        with pytest.raises(ValueError, match='prediction/settlement'):
            store.verify(compact=True)
        assert not list(tmp_path.glob('.verify-*'))
    finally:
        store.close()


def test_builtin_large_result_exports_reference_operations_and_resource_clock(tmp_path, monkeypatch):
    data, spec, model = fixture()
    body = large_result()

    def search(*_args, **_kwargs):
        return model, deepcopy(body)

    monkeypatch.setattr(population, 'search_population', search)
    result = run(tmp_path, 'large-run', data, spec, model,
                 policy=EvolutionPolicy(max_cycles=1, shadow_hours=16), max_ticks=1)
    research = result['cycles'][0]['research']
    folder = tmp_path / 'large-run'
    artifact = json.loads((folder / 'research/cycle-0001/result.json').read_text('utf-8'))
    assert artifact['_research_ref'] == research['_research_ref']
    assert len(list((folder / 'research-objects').glob('*.json'))) == 1
    assert 'records' not in research and research['candidate_search_elapsed_seconds'] == .25
    assert research['elapsed_seconds'] > 0 and research['development_memory_count'] == 30
    assert _resources(_read(folder))['known_search_seconds'] == research['elapsed_seconds']
    exported = export_evolution(folder, tmp_path / 'report')
    assert 'write_program' in (tmp_path / 'report/report.html').read_text('utf-8')
    assert exported['rows'] == 1
