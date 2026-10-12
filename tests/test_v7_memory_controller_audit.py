"""Independent real-controller synthetic memory audit; no provider or market calls.

The tiny source-root fixture isolates frozen source identity from concurrent code
editing. Numerical ridge fits and the actual RunStore/ExposureLedger remain real.
"""
import copy
import json
import sqlite3

import pytest
from test_v7_rolling_registry import candles
from test_v7_run_factor_pool import source_root

from factorlab.discovery import v7_run
from factorlab.discovery.program_contract import Program
from factorlab.discovery.program_syntax import CompactProgram
from factorlab.discovery.rolling import RollingSpec
from factorlab.discovery.v7_agent import BATCH_VERSION, BackendReply
from factorlab.discovery.v7_memory import ResearchMemory
from factorlab.discovery.v7_run import RunStore, V7Config, run_month
from factorlab.protocol import fingerprint


class CaptureDuplicateBackend:
    """Each actual fake batch repeats one formula; all materializations cost AST."""
    def __init__(self):
        self.packets = []
        self.repeat_each_month = False

    def __call__(self, packet, **kwargs):
        body = json.loads(packet.body)
        index = len(self.packets)
        self.packets.append(copy.deepcopy(body))
        value = {'version': 'causal-operator-program-v1',
                 'feature_names': list(body['parents'][0]['program']['feature_names']),
                 'parameters': {}, 'outputs': [{'op': 'const', 'value': float(
                     index % 3 + 1 if self.repeat_each_month else index + 1)}]}
        compact = CompactProgram.from_program(Program.from_dict(value)).to_dict()
        response = {'version': BATCH_VERSION, 'identity': packet.identity.to_dict(),
                    'family_id': f'reused-family-{index % 3}',
                    'hypothesis': 'Synthetic duplicate-column control',
                    'mechanism': 'Constant scalar', 'competing_explanation': 'Intercept already spans it',
                    'falsification': 'No conditional improvement',
                    'parent_ids': [body['parents'][-1]['parent_id']],
                    'template': {'base': compact, 'slots': []}, 'programs': [compact, compact]}
        return BackendReply(json.dumps(response), {'input_tokens': 1, 'output_tokens': 1},
                            'synthetic-backend', 'audit-fixture')


class CrashAfterMeasuredResult(BaseException):
    pass


@pytest.fixture(scope='module')
def synthetic_bars():
    return candles('2024-08-01', '2025-06-03')


@pytest.fixture
def setup(tmp_path, monkeypatch, synthetic_bars):
    source_root(tmp_path, monkeypatch)
    config = V7Config(evaluation_mode='joint', search_mode='gpt',
                      formulas_per_family=1, lookback=2)
    return tmp_path / 'run', synthetic_bars, config, CaptureDuplicateBackend()


def state(directory, month):
    with sqlite3.connect(directory / 'state.sqlite') as db:
        return json.loads(db.execute('SELECT value FROM months WHERE month=?', (month,)).fetchone()[0])


def run(setup, cutoff='2025-04-01T00:00Z'):
    directory, bars, config, backend = setup
    return run_month(directory, bars, cutoff, config=config, backend=backend)


def test_two_actual_months_keep_source_bound_memory_and_namespaced_families(setup):
    directory, _, config, backend = setup
    first = run(setup)
    original = state(directory, '2025-04')
    second = run(setup, '2025-05-01T00:00Z')
    latest = state(directory, '2025-05')
    for result in (first, second):
        counts = result['budget']['counts']
        assert counts['gpt_attempt'] == 3 and counts['family'] == 3
        assert counts['ast'] == 6 and counts['f'] == 3
        assert len(result['budget']['families']) == 3
        assert all(f['ast'] == 2 and f['f'] == 1 for f in result['budget']['families'].values())
    old_memory = ResearchMemory.from_state(original['research_memory'])
    memory = ResearchMemory.from_state(latest['research_memory'])
    assert len(memory.proposals) == 6 and set(old_memory.proposals) < set(memory.proposals)
    assert all(f.startswith('2025-04:gpt:') for f in old_memory.proposals)
    assert sum(f.startswith('2025-05:gpt:') for f in memory.proposals) == 3
    assert len(memory.scheduled) == 3 and sum(memory.lanes.values()) == 3
    assert len(memory.evidence) == 6
    assert json.loads((directory / 'memories/2025-05.json').read_text()) == latest['research_memory']
    store = RunStore(directory, config, RollingSpec())
    try:
        resolver = memory.resolver(store, store.exposure)
        for identifier, reference in memory.evidence.items():
            receipt = resolver(identifier)
            origin = state(directory, reference['month'])
            assert receipt['data_id'] == origin['data_id']
            assert receipt['search_exposure'] == origin['search_exposure']
            assert receipt['scope'] == 'prior_context_only' and receipt['current_gain'] is None
    finally:
        store.close()
    assert len(backend.packets) == 6
    assert run(setup, '2025-05-01T00:00Z') == second
    assert len(backend.packets) == 6 and state(directory, '2025-04') == original


def test_actual_request_contains_bounded_verified_prior_readouts_not_current_metrics(setup):
    directory, _, _, backend = setup
    run(setup)
    run(setup, '2025-05-01T00:00Z')
    april = state(directory, '2025-04')
    may = state(directory, '2025-05')
    priors = [readout for packet in backend.packets[3:]
              for parent in packet['parents']
              for readout in parent['evidence'].get('prior_readouts', [])]
    assert priors, 'Measured ancestor results must actually reach the fake backend request'
    assert any(readout['data_id'] == april['data_id'] for readout in priors)
    all_states = {s['cutoff']: s for s in (april, may)}
    for readout in priors:
        original = all_states[readout['cutoff']]
        assert readout['data_id'] == original['data_id']
        matches = [record['result'] for record in original['pool']['records'].values()
                   if fingerprint(record['result']) == readout['result_hash']]
        assert len(matches) == 1
        assert readout['gain_F'] == matches[0]['channels']['F']['gain']
        assert readout['gain_S'] == matches[0]['channels']['S']['gain']
        assert readout['pool_id'] == matches[0]['context']['pool_hash']
    for packet in backend.packets:
        for parent in packet['parents']:
            assert parent['evidence']['metrics'] == {}
            assert len(parent['evidence'].get('prior_readouts', [])) <= 2


def test_completed_month_rejects_tampered_published_memory(setup):
    directory, _, _, _ = setup
    run(setup)
    path = directory / 'memories/2025-04.json'
    record = json.loads(path.read_text())
    record['memory']['lanes']['blind'] = 6
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='[Mm]emory|integrity|checkpoint'):
        run(setup)


def test_crash_after_measured_result_reconciles_memory_without_refitting(setup, monkeypatch):
    directory, _, _, _ = setup
    original_save = v7_run.RunStore.save
    crashed = False

    def save_then_die(self, month, record, event):
        nonlocal crashed
        original_save(self, month, record, event)
        if not crashed and event == 'measured_result_persisted':
            crashed = True
            raise CrashAfterMeasuredResult

    monkeypatch.setattr(v7_run.RunStore, 'save', save_then_die)
    with pytest.raises(CrashAfterMeasuredResult):
        run(setup)
    partial = state(directory, '2025-04')
    assert len(partial['pool']['records']) == 1
    consumed_f = partial['budget']['counts']['f']
    monkeypatch.setattr(v7_run.RunStore, 'save', original_save)
    final = run(setup)
    complete = state(directory, '2025-04')
    memory = ResearchMemory.from_state(complete['research_memory'])
    measured_ids = set(complete['pool']['records'])
    remembered_ids = {ref['formula_id'] for ref in memory.evidence.values()}
    assert measured_ids <= remembered_ids, 'Durably measured pre-crash result was lost from research memory'
    assert final['budget']['counts']['f'] == consumed_f + 2
    research_parents = [p for p in memory.book.parents_by_id.values() if p['kind'] == 'research']
    assert len(research_parents) == len(measured_ids)


def test_real_prior_ancestor_privacy_is_rechecked_before_prompt(setup):
    directory, _, config, _ = setup
    run(setup)
    run(setup, '2025-05-01T00:00Z')
    latest = state(directory, '2025-05')
    memory = ResearchMemory.from_state(latest['research_memory'])
    store = RunStore(directory, config, RollingSpec())
    try:
        clean = memory.prompt_parents(store, store.exposure, island='refinement', current_context={})
        assert clean['ancestry_evidence']
        prior = state(directory, '2025-04')['search_exposure']
        store.exposure.record(**{k: v for k, v in prior.items() if k != 'data_id'}, role='C')
        with pytest.raises(ValueError):
            memory.prompt_parents(store, store.exposure, island='refinement', current_context={})
    finally:
        store.close()


def test_real_checkpoint_source_binding_cannot_be_relabelled(setup):
    directory, _, config, _ = setup
    run(setup)
    current = state(directory, '2025-04')
    memory = ResearchMemory.from_state(current['research_memory'])
    identifier = next(iter(memory.evidence))
    store = RunStore(directory, config, RollingSpec())
    try:
        current['search_exposure']['source'] = 'different-archive'
        store.save('2025-04', current, 'synthetic_wrong_exposure_source')
        with pytest.raises(ValueError, match='exposure|identity'):
            memory.resolver(store, store.exposure)(identifier)
    finally:
        store.close()


def test_actual_remeasurement_advances_old_hypothesis_and_two_stalls_sleep(setup, monkeypatch):
    """Disable admission only to keep the diagnostic reference fixed; fits stay real."""
    from factorlab.discovery import joint_pool

    directory, _, _, backend = setup
    backend.repeat_each_month = True
    monkeypatch.setattr(joint_pool, 'stability_select', lambda results, pool, **kwargs:
                        {'selected_ids': [], 'role': 'S', 'formal_support_claimed': False,
                         'receipt': {'pool_hash': pool.digest}})
    run(setup)
    april = state(directory, '2025-04')
    april_memory = ResearchMemory.from_state(april['research_memory'])
    versions = {proposal['version'] for proposal in april_memory.proposals.values()}
    assert all(len(april_memory.book.versions[v]['history']) == 1 for v in versions)
    run(setup, '2025-05-01T00:00Z')
    may = state(directory, '2025-05')
    memory = ResearchMemory.from_state(may['research_memory'])
    assert april['data_id'] != may['data_id']
    for version in versions:
        record = memory.book.versions[version]
        assert record['state'] == 'dormant'
        assert [row['month'] for row in record['history']] == [april['cutoff'], may['cutoff']]
        second_sources = record['history'][-1]['evidence']
        assert all(row['month'] == '2025-05' and row['data_id'] == may['data_id'] for row in second_sources)
    # The original immutable April checkpoint remains one-check active evidence.
    assert state(directory, '2025-04') == april


@pytest.mark.parametrize('boundary', ['committed', 'memory_published'])
def test_finalization_crash_publishes_exact_committed_state_without_research(setup, monkeypatch, boundary):
    from factorlab.discovery import joint_pool

    directory, _, _, backend = setup
    original_save = v7_run.RunStore.save
    original_memory_save = ResearchMemory.save

    def crash_after_commit(self, month, record, event):
        original_save(self, month, record, event)
        if event == 'month_finalization_committed':
            raise CrashAfterMeasuredResult

    def crash_after_memory(self, path):
        original_memory_save(self, path)
        raise CrashAfterMeasuredResult

    if boundary == 'committed':
        monkeypatch.setattr(v7_run.RunStore, 'save', crash_after_commit)
    else:
        monkeypatch.setattr(ResearchMemory, 'save', crash_after_memory)
    with pytest.raises(CrashAfterMeasuredResult):
        run(setup)
    committed = state(directory, '2025-04')
    assert committed['status'] == 'finalizing'
    call_count = len(backend.packets)
    monkeypatch.setattr(v7_run.RunStore, 'save', original_save)
    monkeypatch.setattr(ResearchMemory, 'save', original_memory_save)

    def forbidden(*args, **kwargs):
        raise AssertionError('Committed finalization must not repeat research or fitting')

    monkeypatch.setattr(v7_run, 'fit_ridge', forbidden)
    monkeypatch.setattr(joint_pool, 'evaluate_candidate', forbidden)
    monkeypatch.setattr(joint_pool, 'backward_prune', forbidden)
    monkeypatch.setattr(joint_pool, 'stability_select', forbidden)
    assert run(setup) == committed['summary']
    complete = state(directory, '2025-04')
    assert complete == {**committed, 'status': 'complete'}
    assert len(backend.packets) == call_count
    assert json.loads((directory / 'memories' / '2025-04.json').read_text('utf-8')) == committed['research_memory']
    v7_run._verify_month_artifacts(directory, '2025-04', complete)
