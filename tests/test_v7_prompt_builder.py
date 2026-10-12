"""No remote calls: fixtures exercise S isolation, batch limits and fake CLI."""
import copy
import hashlib
import json
import sys

import pytest

from factorlab.discovery.local_mutation import MonthlyBudget
from factorlab.discovery.operator_program import VERSION, Program
from factorlab.discovery.program_syntax import CompactProgram, parse_expression
from factorlab.discovery.prompt_builder import PromptIdentity, build_formula_prompt
from factorlab.discovery.v7_agent import (
    BackendReply,
    CodexBatchBackend,
    ProposalBatch,
    propose_batch,
)


def inputs():
    identity = PromptIdentity('vol12h', 'rolling3utc-v1', 'pool-v1', 'raw-hourly-v1')
    p = Program.from_dict({'version': VERSION, 'feature_names': ['returns'], 'parameters': {},
                           'outputs': [parse_expression('std(f("returns"))')]})
    compact = CompactProgram.from_program(p).to_dict()
    evidence = {'evidence_id': 's1', 'role': 'S', 'task_id': identity.task_id,
                'rolling_id': identity.rolling_id, 'pool_id': identity.pool_id,
                'summary': 'Synthetic development fixture; no market claim.', 'metrics': {}}
    return {'identity': identity, 'task_semantics': 'Predict sqrt(sum of next 12 log returns squared).',
            'feature_schema': [{'name': 'returns', 'unit': 'log ratio', 'preprocessing': 'raw'}],
            'input_semantics': 'Closed, received hourly bars; raw log returns; no old standardizer.',
            'parents': [{'parent_id': f'p{i}', 'program': compact, 'evidence': evidence}
                        for i in range(2)], 'budget': MonthlyBudget('2026-10')}


def response(kwargs):
    compact = kwargs['parents'][0]['program']
    return {'version': 'v7-proposal-batch-v1', 'identity': kwargs['identity'].to_dict(),
            'family_id': 'h1', 'hypothesis': 'Recent dispersion may improve the anchor.',
            'mechanism': 'Volatility persistence.', 'competing_explanation': 'Redundant anchor.',
            'falsification': 'No S conditional gain under frozen evaluation.',
            'parent_ids': ['p0'], 'template': {'base': copy.deepcopy(compact), 'slots': []},
            'programs': [copy.deepcopy(compact)]}


def test_dynamic_context_does_not_change_prefix_and_has_no_action_menu():
    kwargs = inputs()
    first = build_formula_prompt(**kwargs)
    kwargs['failure_counts'] = {'invalid_window': 2}
    second = build_formula_prompt(**kwargs)
    assert first.prefix == second.prefix and first.prefix_hash == second.prefix_hash
    assert first.request_hash != second.request_hash
    assert 'actions' not in json.loads(first.body)
    assert first.telemetry()['usage'] is None
    assert first.telemetry()['stdin_utf8_bytes'] == len(first.text.encode())


@pytest.mark.parametrize('mutation', ['V', 'C', 'wrong_task', 'wrong_pool', 'actions', 'too_many_parents'])
def test_exposure_identity_and_size_rejection(mutation):
    kwargs = inputs()
    evidence = kwargs['parents'][0]['evidence']
    if mutation in {'V', 'C'}:
        evidence['role'] = mutation
    elif mutation == 'wrong_task':
        evidence['task_id'] = 'event20h'
    elif mutation == 'wrong_pool':
        evidence['pool_id'] = 'other'
    elif mutation == 'actions':
        kwargs['parents'][0]['actions'] = ['neural width']
    else:
        kwargs['parents'] *= 3
    with pytest.raises(ValueError):
        build_formula_prompt(**kwargs)


def test_three_calendar_months_cold_only_and_explicit_input_schema():
    with pytest.raises(ValueError):
        PromptIdentity('t', 'r', 'p', 'f', train_calendar_months=5)
    with pytest.raises(ValueError):
        PromptIdentity('t', 'r', 'p', 'f', fit_initialization='warm')
    kwargs = inputs()
    kwargs['feature_schema'][0]['name'] = 'standardized_returns'
    with pytest.raises(ValueError, match='feature contract'):
        build_formula_prompt(**kwargs)


def test_batch_checks_template_and_context_and_keeps_duplicates_visible():
    kwargs = inputs()
    value = response(kwargs)
    value['programs'] *= 2
    batch = ProposalBatch.from_dict(value, expected_identity=kwargs['identity'],
                                    feature_names=('returns',), allowed_parent_ids={'p0', 'p1'})
    assert len(batch.programs) == 2
    assert batch.programs[0].digest == batch.programs[1].digest
    for mutate in ['task', 'count', 'outside_template', 'parent']:
        bad = copy.deepcopy(value)
        if mutate == 'task':
            bad['identity']['task_id'] = 'event'
        elif mutate == 'count':
            bad['programs'] *= 5
        elif mutate == 'outside_template':
            bad['programs'][0]['expressions'] = ['mean(f("returns"))']
        else:
            bad['parent_ids'] = ['unseen']
        with pytest.raises(ValueError):
            ProposalBatch.from_dict(bad, expected_identity=kwargs['identity'],
                                    feature_names=('returns',), allowed_parent_ids={'p0', 'p1'})


def test_injected_proposer_counts_success_and_reserves_before_external_work():
    kwargs = inputs()
    packet = build_formula_prompt(**kwargs)
    persisted, audit = [], []

    def backend(packet, *, model, reasoning):
        assert persisted[-1]['counts']['gpt_attempt'] == 1
        assert (model, reasoning) == ('gpt-6-astra', 'high')
        return BackendReply(json.dumps(response(kwargs)), {'input_tokens': 50})

    result = propose_batch(packet, backend=backend, budget=kwargs['budget'],
                           feature_names=('returns',), allowed_parent_ids={'p0', 'p1'},
                           audit=audit, persist_budget=persisted.append)
    assert result.family_id == 'h1'
    assert kwargs['budget'].counts == {'gpt_attempt': 1, 'family': 1, 'ast': 1, 'f': 0, 'solve': 0}
    assert audit[-1]['status'] == 'validated'
    assert audit[-1]['model_resolved'] is None
    assert persisted[-1] == kwargs['budget'].to_dict()


def test_invalid_reply_consumes_attempt_and_materialized_ast_without_success():
    kwargs = inputs()
    bad = response(kwargs)
    bad['programs'][0]['expressions'] = ['last(rolling_mean(f("returns"),0))']
    audit = []
    with pytest.raises(ValueError):
        propose_batch(build_formula_prompt(**kwargs),
                      backend=lambda *a, **kw: BackendReply(json.dumps(bad)),
                      budget=kwargs['budget'], feature_names=('returns',),
                      allowed_parent_ids={'p0'}, audit=audit, persist_budget=lambda _: None)
    assert kwargs['budget'].counts['gpt_attempt'] == kwargs['budget'].counts['ast'] == 1
    assert kwargs['budget'].counts['family'] == 0
    assert audit[0]['status'] == 'failed'


def test_transmission_failure_does_not_retry_or_change_provider():
    kwargs = inputs()
    calls, audit = [], []

    def failed(*args, **kw):
        calls.append(kw)
        raise TimeoutError('fake transport')

    with pytest.raises(TimeoutError):
        propose_batch(build_formula_prompt(**kwargs), backend=failed, budget=kwargs['budget'],
                      feature_names=('returns',), allowed_parent_ids={'p0'}, audit=audit,
                      persist_budget=lambda _: None)
    assert len(calls) == 1
    assert kwargs['budget'].counts['gpt_attempt'] == 1
    assert audit[0]['elapsed_seconds'] >= 0


def test_audit_final_persistence_failure_preserves_original_backend_error():
    kwargs = inputs()
    audit = []

    def backend(*args, **kw):
        raise TimeoutError('original transport failure')

    def persist(_):
        if audit and audit[-1]['status'] == 'failed':
            raise OSError('artificial disk failure')

    with pytest.raises(TimeoutError, match='original transport') as error:
        propose_batch(build_formula_prompt(**kwargs), backend=backend, budget=kwargs['budget'],
                      feature_names=('returns',), allowed_parent_ids={'p0'}, audit=audit,
                      persist_budget=persist)
    assert 'OSError' in error.value.__notes__[0]


def test_success_never_returns_if_final_audit_persistence_fails():
    kwargs = inputs()
    audit = []

    def persist(_):
        if audit and audit[-1]['status'] == 'validated':
            raise OSError('artificial disk failure')

    with pytest.raises(OSError, match='disk failure'):
        propose_batch(build_formula_prompt(**kwargs),
                      backend=lambda *a, **kw: BackendReply(json.dumps(response(kwargs))),
                      budget=kwargs['budget'], feature_names=('returns',),
                      allowed_parent_ids={'p0'}, audit=audit, persist_budget=persist)


def fake_command(monkeypatch, backend, events):
    # Actual local subprocess, but no Codex executable, credentials or network.
    script = 'import sys; sys.stdin.read(); print(' + repr('\n'.join(json.dumps(e) for e in events)) + ')'
    monkeypatch.setattr(backend, '_command', lambda *args: [sys.executable, '-c', script])


def test_codex_adapter_fake_process_records_unknown_usage_and_exact_stdin(tmp_path, monkeypatch):
    kwargs = inputs()
    packet = build_formula_prompt(**kwargs)
    backend = CodexBatchBackend(tmp_path)
    fake_command(monkeypatch, backend, [
        {'type': 'turn.started'},
        {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': json.dumps(response(kwargs))}},
        {'type': 'turn.completed', 'usage': {'input_tokens': 100, 'output_tokens': 30}}])
    reply = backend(packet, model='gpt-6-astra', reasoning='high')
    assert json.loads(reply.payload)['family_id'] == 'h1'
    path = next(tmp_path.glob('*/attempt.json'))
    record = json.loads(path.read_text('utf-8'))
    assert record['reasoning_output_tokens'] is None
    assert record['cached_input_tokens'] is None
    assert record['output_tokens'] == 30
    saved = path.with_name('stdin.txt').read_bytes()
    assert saved == packet.text.encode()
    assert record['stdin_sha256'] == hashlib.sha256(saved).hexdigest()
    assert record['status'] == 'completed'
    assert record['model_resolved'] is None


def test_codex_adapter_rejects_tool_event_and_records_failure(tmp_path, monkeypatch):
    backend = CodexBatchBackend(tmp_path)
    fake_command(monkeypatch, backend, [{'type': 'turn.started'},
                                      {'type': 'item.completed', 'item': {'type': 'command_execution'}}])
    with pytest.raises(ValueError, match='ToolEvent'):
        backend(build_formula_prompt(**inputs()), model='gpt-6-astra', reasoning='high')
    record = json.loads(next(tmp_path.glob('*/attempt.json')).read_text())
    assert record['status'] == 'failed'


def test_cli_command_is_restricted_and_uses_batch_schema(tmp_path):
    backend = CodexBatchBackend(tmp_path, executable='codex-fixture')
    command = backend._command(tmp_path, tmp_path / 'response-schema.json')
    assert command[:5] == ['codex-fixture', '--no-daemon', '-a', 'never', 'exec']
    assert '--ignore-user-config' in command
    assert '--ephemeral' in command
    assert command[command.index('--sandbox') + 1] == 'read-only'
    assert 'shell_tool' in command and 'multi_agent' in command and 'apps' in command
    assert command[-3:] == ['--model', 'gpt-6-astra', '-']


@pytest.mark.parametrize('case', ['timeout', 'oversized', 'nonzero'])
def test_fake_process_resource_failures_are_recorded(tmp_path, monkeypatch, case):
    backend = CodexBatchBackend(tmp_path, timeout_seconds=.2, max_output_bytes=1024)
    scripts = {'timeout': 'import time; time.sleep(2)',
               'oversized': 'print("x" * 2048)', 'nonzero': 'raise SystemExit(7)'}
    monkeypatch.setattr(backend, '_command', lambda *args: [sys.executable, '-c', scripts[case]])
    with pytest.raises((TimeoutError, ValueError, RuntimeError)):
        backend(build_formula_prompt(**inputs()), model='gpt-6-astra', reasoning='high')
    record = json.loads(next(tmp_path.glob('*/attempt.json')).read_text())
    assert record['status'] == 'failed'
    assert record['input_tokens'] is None
