import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import pytest
from filelock import FileLock

from factorlab.discovery.agent_worker import (
    ResearchWorker,
    WorkerConfig,
    request_worker_stop,
    sanitize_request,
    stop_worker,
)
from factorlab.discovery.contracts import ExperimentSpec, RepresentationSpec
from factorlab.discovery.research_memory import ResearchMemory
from factorlab.discovery.search import action_menu
from factorlab.protocol import fingerprint


def request(directory, iteration=0, **changes):
    spec, parent = ExperimentSpec(), RepresentationSpec()
    payload = {'protocol_hash': spec.digest, 'parent_hash': parent.digest,
               'parent': asdict(parent), 'evidence_cutoff': spec.dev_end,
               'task': asdict(spec.tasks[0]), 'actions': action_menu(parent, spec),
               'history': [], 'iteration': iteration, 'instruction': 'UNTRUSTED_RAW_INSTRUCTION',
               'response_schema': {}, **changes}
    payload = json.loads(json.dumps(payload))
    payload['request_hash'] = fingerprint(payload)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f'request-{iteration}.json').write_text(json.dumps(payload), encoding='utf-8')
    return payload


def worker(tmp_path, **changes):
    return ResearchWorker(WorkerConfig(exchange_dirs=(tmp_path / 'exchange',),
                          state_dir=tmp_path / 'state', retry_seconds=0, **changes))


def test_poll_multiple_exchanges_atomic_bound_output_and_restart(tmp_path):
    one, two = tmp_path / 'one', tmp_path / 'two'
    request(one)
    request(two, 1)
    config = WorkerConfig(exchange_dirs=(one, two), state_dir=tmp_path / 'state')
    result = ResearchWorker(config).run(once=True)
    assert result['completed'] == 2 and result['llm_backend'] is False
    for folder, iteration in [(one, 0), (two, 1)]:
        proposal = json.loads((folder / f'proposal-{iteration}.json').read_text('utf-8'))
        packet = json.loads((folder / f'request-{iteration}.json').read_text('utf-8'))
        assert proposal['protocol_hash'] == packet['protocol_hash']
        assert {'operation': proposal['operation'], 'arguments': proposal['arguments']} in packet['actions']
    assert ResearchWorker(config).run(once=True)['attempts'] == 0


def test_restart_honors_persistent_retry_ceiling(tmp_path, monkeypatch):
    request(tmp_path / 'exchange')
    def failed(*args):
        raise ValueError('secret confirmation score deliberately not logged')
    monkeypatch.setattr(ResearchWorker, '_backend', failed)
    for expected in (1, 1, 0):
        result = worker(tmp_path, max_attempts=2).run(once=True)
        assert result['attempts'] == expected
    errors = sorted((tmp_path / 'exchange').glob('worker-error-*.json'))
    assert len(errors) == 2
    assert 'secret' not in ''.join(p.read_text('utf-8') for p in errors)
    assert not (tmp_path / 'exchange' / 'proposal-0.json').exists()


def test_lock_blocks_other_worker_and_releases_on_exit(tmp_path):
    request(tmp_path / 'exchange')
    lock = FileLock(str(tmp_path / 'exchange' / 'request-0.worker.lock'))
    with lock:
        assert worker(tmp_path).run(once=True)['attempts'] == 0
    assert worker(tmp_path).run(once=True)['completed'] == 1


def test_two_worker_state_dirs_cannot_double_publish(tmp_path):
    request(tmp_path / 'exchange')
    configs = [WorkerConfig(exchange_dirs=(tmp_path / 'exchange',), state_dir=tmp_path / f'state-{i}')
               for i in range(2)]
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda c: ResearchWorker(c).run(once=True), configs))
    assert sum(v['completed'] for v in results) == 1


def test_request_hash_future_history_and_confirmation_are_rejected(tmp_path):
    packet = request(tmp_path / 'exchange')
    packet['parent_hash'] = 'tampered'
    with pytest.raises(ValueError, match='integrity'):
        sanitize_request(packet)
    for history in ([{'partition': 'confirmation', 'evidence_cutoff': '2025-09-01'}],
                    [{'partition': 'development', 'evidence_cutoff': '2025-10-01'}],
                    [{'partition': 'development', 'evidence_cutoff': '2025-09-01',
                      'status': 'ok', 'metrics': {'confirmation': {'loss': 0.01}}}]):
        packet = request(tmp_path / 'exchange', history=history)
        with pytest.raises(ValueError):
            sanitize_request(packet)
    assert worker(tmp_path).run(once=True)['attempts'] == 0


def test_generic_command_only_receives_sanitized_packet_and_fixed_runtime(tmp_path):
    request(tmp_path / 'exchange', arbitrary_path='CONFIRMATION_PATH_SENTINEL')
    program = ('import json,sys,os; s=sys.stdin.read(); '
               'assert "UNTRUSTED_RAW_INSTRUCTION" not in s; '
               'assert "CONFIRMATION_PATH_SENTINEL" not in s; '
               'assert "runtime" in os.getcwd(); '
               'print(json.dumps(dict(action_index=0,hypothesis="Testable",falsification="Loss rises")))')
    result = worker(tmp_path, backend='command', command=(sys.executable, '-c', program)).run(once=True)
    assert result['completed'] == 1


def test_rules_use_retrieved_memory_to_choose_untried_action(tmp_path):
    packet = request(tmp_path / 'exchange')
    path = tmp_path / 'memory.sqlite'
    ResearchMemory(path).record_action_result(task=packet['task'], evidence_cutoff=packet['evidence_cutoff'],
        protocol_hash=packet['protocol_hash'], parent_hash=packet['parent_hash'], candidate_id='candidate',
        action=packet['actions'][0], status='ok')
    assert worker(tmp_path, memory_path=path).run(once=True)['completed'] == 1
    result = json.loads((tmp_path / 'exchange' / 'proposal-0.json').read_text('utf-8'))
    assert {'operation': result['operation'], 'arguments': result['arguments']} == packet['actions'][1]


def test_codex_disables_capabilities_uses_schema_and_validates_events(tmp_path, monkeypatch):
    request(tmp_path / 'exchange')
    monkeypatch.setattr('factorlab.discovery.agent_worker.shutil.which', lambda value: 'codex.exe')
    def completed(self, command, content, runtime, remaining):
        assert '--ignore-user-config' in command and '--ignore-rules' in command
        assert '--ephemeral' in command and '--output-schema' in command
        for feature in ('shell_tool', 'apps', 'plugins', 'browser_use', 'multi_agent'):
            assert command[command.index(feature) - 1] == '--disable'
        assert 'project_doc_max_bytes=0' in command
        assert 'UNTRUSTED_RAW_INSTRUCTION' not in content
        assert 'runtime' in str(runtime)
        assert 0 < remaining <= self.config.timeout_seconds
        message = json.dumps({'action_index': 0, 'hypothesis': 'Hypothesis', 'falsification': 'Falsification'})
        events = [{'type': 'item.completed', 'item': {'type': 'agent_message', 'text': message}},
                  {'type': 'turn.completed', 'usage': {'input_tokens': 10, 'output_tokens': 4}}]
        return '\n'.join(map(json.dumps, events))
    monkeypatch.setattr(ResearchWorker, '_run_command', completed)
    assert worker(tmp_path, backend='codex').run(once=True)['completed'] == 1


def test_budget_bounds_calls_even_when_backends_fail(tmp_path, monkeypatch):
    request(tmp_path / 'exchange')
    request(tmp_path / 'exchange', 1)
    def failed(*args):
        raise TimeoutError('timed out')
    monkeypatch.setattr(ResearchWorker, '_backend', failed)
    result = worker(tmp_path, max_requests=1, max_seconds=1).run()
    assert result['attempts'] == 1 and result['errors'] == 1


@pytest.mark.parametrize('program,expected', [
    ('import time;time.sleep(10)', 'BackendTimeout'),
    ('import sys;sys.stdout.write("x"*100000)', 'BackendOutputBudgetExceeded'),
])
def test_process_timeout_and_output_budget_are_enforced(tmp_path, program, expected):
    request(tmp_path / 'exchange')
    result = worker(tmp_path, backend='command', command=(sys.executable, '-c', program),
                    timeout_seconds=.2, max_output_bytes=1024, max_attempts=1).run(once=True)
    assert result['errors'] == 1 and result['completed'] == 0
    states = list((tmp_path / 'state').glob('*.json'))
    assert json.loads(states[0].read_text('utf-8'))['error_code'] == expected


def test_codex_startup_warning_is_allowed_but_model_tool_event_is_rejected(tmp_path, monkeypatch):
    request(tmp_path / 'exchange')
    monkeypatch.setattr('factorlab.discovery.agent_worker.shutil.which', lambda value: 'codex.exe')
    message = json.dumps({'action_index': 0, 'hypothesis': 'Hypothesis', 'falsification': 'Falsification'})
    events = [{'type': 'item.completed', 'item': {'type': 'error'}}, {'type': 'turn.started'},
              {'type': 'item.completed', 'item': {'type': 'command_execution'}},
              {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': message}},
              {'type': 'turn.completed'}]
    monkeypatch.setattr(ResearchWorker, '_run_command', lambda *args: '\n'.join(map(json.dumps, events)))
    assert worker(tmp_path, backend='codex', max_attempts=1).run(once=True)['completed'] == 0
    assert not (tmp_path / 'exchange' / 'proposal-0.json').exists()
    events.pop(2)
    successful = ResearchWorker(WorkerConfig(exchange_dirs=(tmp_path / 'exchange',),
                 state_dir=tmp_path / 'successful', backend='codex'))
    assert successful.run(once=True)['completed'] == 1


def test_typed_history_paths_and_failure_prose_do_not_reach_model(tmp_path):
    spec, parent = ExperimentSpec(), RepresentationSpec()
    packet = request(tmp_path / 'exchange', history=[{
        'partition': 'development', 'evidence_cutoff': spec.dev_end, 'status': 'failed',
        'protocol_hash': spec.digest, 'source_protocol_hash': 'a' * 64,
        'architecture': asdict(parent), 'settings': asdict(spec.training),
        'candidate_path': 'CONFIRMATION_PATH', 'error': 'CONFIRMATION_SCORE', 'metrics': {}}])
    cleaned = sanitize_request(packet)
    assert cleaned['history'][0]['architecture']['modules'][0]['name'] == 'temporal'
    assert 'CONFIRMATION' not in json.dumps(cleaned)


def test_stop_file_prevents_new_requests_and_interrupts_backend_safely(tmp_path):
    request(tmp_path / 'exchange')
    stop = tmp_path / 'stop.json'
    request_worker_stop(stop, source='test')
    result = worker(tmp_path, stop_file=stop).run()
    assert result['stopped'] == 'requested' and result['attempts'] == 0
    fresh_stop = tmp_path / 'fresh-stop.json'
    timer = threading.Timer(.2, lambda: request_worker_stop(fresh_stop, source='test'))
    timer.start()
    began = time.monotonic()
    try:
        result = worker(tmp_path, backend='command',
            command=(sys.executable, '-c', 'import time;time.sleep(10)'),
            stop_file=fresh_stop, timeout_seconds=20).run()
    finally:
        timer.join()
    assert result['stopped'] == 'requested' and result['errors'] == 1
    assert time.monotonic() - began < 5
    assert not (tmp_path / 'exchange' / 'proposal-0.json').exists()
    states = list((tmp_path / 'state').glob('*.json'))
    state = json.loads(states[0].read_text('utf-8'))
    assert state['error_code'] == 'WorkerStopRequested' and state['status'] == 'retry'


def test_shutdown_helper_records_stop_source_and_allows_worker_to_exit(tmp_path):
    class Process:
        returncode = None
        def wait(self, timeout):
            assert 0 < timeout <= 10
            assert (tmp_path / 'stop.json').exists()
            self.returncode = 0
    result = stop_worker(Process(), tmp_path / 'stop.json', source='test')
    assert result['stopped'] == 'requested' and result['returncode'] == 0
    assert json.loads((tmp_path / 'stop.json').read_text('utf-8'))['source'] == 'test'
