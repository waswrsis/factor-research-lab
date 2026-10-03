"""Bounded worker/program/search/memory integration on artificial inputs only."""
import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from factorlab.discovery.agent_worker import ResearchWorker, WorkerConfig, sanitize_request
from factorlab.discovery.contracts import (
    ExperimentSpec,
    ModuleSpec,
    RepresentationSpec,
    TaskDataset,
    TaskSpec,
    TrainingSpec,
)
from factorlab.discovery.learning import fit
from factorlab.discovery.operator_search import (
    program_contract,
    program_decision,
    seed_program,
    training_diagnostics,
    validate_program_proposal,
)
from factorlab.discovery.population import search_population
from factorlab.discovery.research_memory import ResearchMemory, to_development_history
from factorlab.discovery.search import (
    ResearchProposal,
    _clean_history,
    action_menu,
    mutate,
    propose,
)
from factorlab.protocol import ROOT, fingerprint


def fixture():
    task = TaskSpec('ret', 'return', 1)
    training = TrainingSpec(epochs=4, batch_size=32, learning_rate=.01,
                            patience=3, l2=.0001, max_seconds=15)
    spec = ExperimentSpec(tasks=(task,), primary_task='ret', lookback=6, symbols=('TEST',),
        start='2024-01-01', train_end='2024-01-04', dev_end='2024-01-07',
        calibration_end='2024-01-09', end='2024-01-11', training=training,
        training_seeds=(17,), candidates=2, search_seconds=60)
    parent = RepresentationSpec((ModuleSpec('base', 'temporal', 3, 5, 6),))

    def dataset(start, seed):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(48, 6, 2)).astype(np.float32)
        dates = pd.date_range(start, periods=len(x), freq='h', tz='UTC')
        return TaskDataset(x, {'ret': x[:, -1, 0] - x[:, :, 0].mean(1)},
            {'ret': np.ones(len(x), dtype=bool)},
            {'ret': (dates + pd.Timedelta(hours=1)).to_numpy(dtype='datetime64[ns]')},
            pd.DataFrame({'sample_id': [f'{seed}-{i}' for i in range(len(x))],
                          'symbol': 'TEST', 'available_at': dates}), ('a', 'b'), (task,))

    return parent, spec, dataset('2024-01-01', 1), dataset('2024-01-04', 2)


def request(exchange, parent, spec, *, iteration=0, program_search=True, history=()):
    # Use the real search writer; a zero wait emits an immutable request without
    # needing a worker race or a manually reconstructed request schema.
    with pytest.raises(TimeoutError):
        propose(parent, spec, list(history), iteration, mode='external', exchange=exchange,
                timeout=0, features=('a', 'b'), program_search=program_search)
    return json.loads((exchange / f'request-{iteration}.json').read_text('utf-8'))


def worker(exchange, state, **changes):
    return ResearchWorker(WorkerConfig(exchange_dirs=(exchange,), state_dir=state,
                                       retry_seconds=0, **changes))


def test_external_menu_cannot_bypass_required_program(tmp_path):
    parent, spec, _train, _dev = fixture()
    request(tmp_path, parent, spec)
    action = action_menu(parent, spec, features=('a', 'b'))[0]
    proposal = ResearchProposal(spec.digest, parent.digest, spec.dev_end,
        action['operation'], 'Check menu bypass.', 'Reject bypass.', 'external', action['arguments'])
    (tmp_path / 'proposal-0.json').write_text(json.dumps(asdict(proposal)), encoding='utf-8')
    with pytest.raises(ValueError, match='must write a program'):
        propose(parent, spec, [], 0, mode='external', exchange=tmp_path, timeout=0,
                features=('a', 'b'), program_search=True)


def test_program_menu_respects_interpreter_maximum_window():
    _parent, spec, _train, _dev = fixture()
    program = seed_program(('a', 'b'), 0, spec.lookback)
    module = ModuleSpec('indicator', 'program', scale=6, features=('a',), program=program)
    actions = action_menu(RepresentationSpec((module,)), replace(spec, lookback=672))
    assert all(1 <= a['arguments']['scale'] <= 512 for a in actions if a['operation'] == 'scale')


def test_real_rules_worker_process_writes_outside_menu_and_recovers_without_duplicate(tmp_path):
    parent, spec, _train, _dev = fixture()
    exchange, state = tmp_path / 'exchange', tmp_path / 'state'
    packet = request(exchange, parent, spec)
    assert packet['operator_contract'] == program_contract(parent, spec.lookback, ('a', 'b'))
    command = [sys.executable, '-m', 'factorlab.discovery.agent_worker', '--exchange', str(exchange),
               '--state-dir', str(state), '--backend', 'rules', '--max-seconds', '15', '--once']
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30, check=True)
    summary = json.loads(result.stdout)
    assert summary['completed'] == 1 and summary['llm_backend'] is False
    proposal_path = exchange / 'proposal-0.json'
    before = proposal_path.read_bytes()
    proposal = propose(parent, spec, [], 0, mode='external', exchange=exchange, timeout=0,
                       features=('a', 'b'), program_search=True)
    assert proposal.operation == 'write_program'
    assert {'operation': proposal.operation, 'arguments': proposal.arguments} not in packet['actions']
    assert proposal.source == 'deterministic_worker_rules'
    child, _training = mutate(parent, spec.training, proposal, spec)
    assert child.modules[-1].kind == 'program' and child.digest != parent.digest
    restarted = worker(exchange, state).run(once=True)
    assert restarted['attempts'] == 0 and restarted['completed'] == 0
    assert proposal_path.read_bytes() == before


def test_invalid_program_is_revised_from_typed_feedback_then_fitted_and_remembered(tmp_path, monkeypatch):
    parent, spec, train, dev = fixture()
    exchange, state = tmp_path / 'exchange', tmp_path / 'state'
    request(exchange, parent, spec)
    packets = []

    def scripted(self, packet, _key):
        packets.append(deepcopy(packet))
        decision = program_decision(packet)
        if len(packets) == 1:
            invalid = json.loads(decision['program_json'])
            invalid['outputs'][0] = {'op': 'unknown_future_operator'}
            decision['program_json'] = json.dumps(invalid)
        return decision, {}

    monkeypatch.setattr(ResearchWorker, '_backend', scripted)
    first = worker(exchange, state).run(once=True)
    assert first['errors'] == 1 and first['completed'] == 0
    assert not (exchange / 'proposal-0.json').exists()
    second = worker(exchange, state).run(once=True)
    assert second['completed'] == 1 and len(packets) == 2
    assert 'revision_feedback' not in packets[0]
    feedback = packets[1]['revision_feedback']
    assert feedback['previous_attempt'] == 1
    assert feedback['error_type'] in {'ValueError', 'ProgramValidationError'}
    assert isinstance(feedback['error_code'], str) and feedback['error_code']
    assert feedback['failure_code'] == 'unknown_operator'
    assert 'unknown_future_operator' not in json.dumps(feedback)
    proposal = propose(parent, spec, [], 0, mode='external', exchange=exchange, timeout=0,
                       features=train.features, program_search=True)
    child, training = mutate(parent, spec.training, proposal, spec)
    model = fit(train, dev, child, training)
    generated = child.modules[-1].name
    assert model.diagnostics['programs'][generated]['selected_raw_update_norm'] > 0
    assert any(row['program_parameter_updates'][generated]['preclip_gradient_norm'] > 0
               for row in model.diagnostics['history'])
    diagnostic = training_diagnostics([model.diagnostics])
    assert diagnostic['program_parameter_update_norm'] > 0
    assert diagnostic['optimizer_steps'] > 0
    memory_path = tmp_path / 'memory.sqlite'
    action = {'operation': proposal.operation, 'arguments': proposal.arguments}
    kwargs = {'task': asdict(spec.tasks[0]), 'evidence_cutoff': spec.dev_end,
        'protocol_hash': spec.digest, 'parent_hash': parent.digest, 'candidate_id': model.digest,
        'action': action, 'structure': asdict(child), 'diagnostics': diagnostic,
        'metrics': {'primary_loss': model.diagnostics['development_loss'],
                 'ret': {'status': 'ok', 'loss': model.diagnostics['development_loss']}},
        'event_id': 'fitted-program-event'}
    memory = ResearchMemory(memory_path)
    assert memory.record_action_result(**kwargs) == memory.record_action_result(**kwargs)
    retrieved = ResearchMemory(memory_path).retrieve(asdict(spec.tasks[0]), spec.dev_end,
                                                     protocol_hash=spec.digest)
    assert len(retrieved) == 1
    assert retrieved[0]['action'] == action and retrieved[0]['diagnostics'] == diagnostic
    history = _clean_history(to_development_history(retrieved, spec), spec)
    assert history[0]['proposal']['operation'] == 'write_program'
    assert history[0]['diagnostics']['program_parameter_update_norm'] > 0
    next_packet = request(exchange, child, spec, iteration=1, history=history)
    clean = sanitize_request(next_packet)
    assert clean['history'][0]['diagnostics'] == diagnostic
    assert clean['history'][0]['proposal']['arguments'] == action['arguments']


@pytest.mark.parametrize('invalid_kind', ['unknown_feature', 'excess_lookback'])
def test_program_proposals_cannot_bypass_observed_schema_or_history(invalid_kind, tmp_path, monkeypatch):
    parent, spec, _train, _dev = fixture()
    exchange, state = tmp_path / 'exchange', tmp_path / 'state'
    packet = request(exchange, parent, spec)
    fields = seed_program(('ghost',) if invalid_kind == 'unknown_feature' else ('a',), 0, spec.lookback)
    if invalid_kind == 'excess_lookback':
        fields['outputs'][0] = {'op': 'last', 'arg': {'op': 'rolling_mean',
            'window': spec.lookback + 1, 'arg': fields['outputs'][0]['arg']}}
    arguments = {'name': packet['operator_contract']['targets'][-1], 'program': fields}
    proposal = ResearchProposal(spec.digest, parent.digest, spec.dev_end, 'write_program',
                                'Test constrained input access', 'Reject invalid data access',
                                'test_scripted_backend', arguments)
    with pytest.raises(ValueError):
        validate_program_proposal(proposal, parent, packet['operator_contract'])

    def invalid(self, clean, _key):
        decision = program_decision(clean)
        decision['program_json'] = json.dumps(fields)
        return decision, {}

    monkeypatch.setattr(ResearchWorker, '_backend', invalid)
    result = worker(exchange, state, max_attempts=1).run(once=True)
    assert result['errors'] == 1 and result['completed'] == 0
    assert not (exchange / 'proposal-0.json').exists()


def test_legacy_menu_remains_executable_without_program_contract(tmp_path):
    parent, spec, _train, _dev = fixture()
    exchange, state = tmp_path / 'exchange', tmp_path / 'state'
    packet = request(exchange, parent, spec, program_search=False)
    assert 'operator_contract' not in packet
    assert packet['actions'] == action_menu(parent, spec, features=('a', 'b'))
    assert worker(exchange, state).run(once=True)['completed'] == 1
    proposal = propose(parent, spec, [], 0, mode='external', exchange=exchange, timeout=0,
                       features=('a', 'b'))
    assert {'operation': proposal.operation, 'arguments': proposal.arguments} in packet['actions']
    child, _training = mutate(parent, spec.training, proposal, spec)
    assert child.digest != parent.digest
    assert all(module.program is None for module in child.modules)


def test_population_fits_generated_program_and_emits_typed_development_evidence():
    parent, spec, train, dev = fixture()
    best, result = search_population(train, dev, parent, spec, mode='diagnostic',
                                     program_search=True, inherit_weights=True)
    assert best is not None and result['newly_fitted'] == 2
    children = [row for row in result['records'] if row.get('proposal', {}).get('operation') == 'write_program']
    assert len(children) == 1 and children[0]['status'] == 'ok'
    cleaned = _clean_history([children[0]], spec)[0]
    assert cleaned['diagnostics']['optimizer_steps'] > 0
    assert cleaned['diagnostics']['program_parameter_update_norm'] > 0
    program_module = next(m for m in children[0]['architecture']['modules'] if m['kind'] == 'program')
    assert program_module['program']['parameters']


def test_language_contract_cannot_be_redefined_in_untrusted_request(tmp_path):
    parent, spec, _train, _dev = fixture()
    packet = request(tmp_path / 'exchange', parent, spec)
    packet['operator_contract']['input_semantics'] = 'future_raw_prices'
    payload = {key: value for key, value in packet.items() if key != 'request_hash'}
    packet['request_hash'] = fingerprint(payload)
    with pytest.raises(ValueError, match='contract'):
        sanitize_request(packet)
