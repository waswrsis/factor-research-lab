"""CLI lifecycle tests using real local models and a bounded rules subprocess."""
import json
import sqlite3
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_discovery_evolution import fixture

from factorlab.discovery import evolution_cli
from factorlab.discovery.evidence import EvidenceRegistry
from factorlab.discovery.evolution import EvolutionPolicy, run_evolution
from factorlab.discovery.experiment import ModelGroup
from factorlab.discovery.learning import FittedResearchModel


def arguments(tmp_path, **updates):
    values = {'output': str(tmp_path / 'run'), 'policy': str(tmp_path / 'policy.json'),
        'data': str(tmp_path / 'data'), 'mode': None, 'search': None, 'device': 'cpu', 'resume': False,
        'initial_run': None, 'initial_model': 'accepted', 'max_ticks': None, 'as_of': None,
        'cross_assets': False, 'funding': None, 'open_interest': None, 'context_max_age_hours': 24,
        'memory': str(tmp_path / 'memory.sqlite'), 'watch': False, 'watch_seconds': 1,
        'poll_seconds': 1, 'worker_backend': None, 'initialize_only': False,
        'initial_checkpoint': None, 'refresh_feed': False}
    values.update(updates)
    return SimpleNamespace(**values)


def wire_cli(monkeypatch, tmp_path, data, model, policy):
    (tmp_path / 'policy.json').write_text(json.dumps(asdict(policy)), encoding='utf-8')
    monkeypatch.setattr(evolution_cli, '_data', lambda args, spec: data)
    monkeypatch.setattr(evolution_cli, '_initial', lambda args, data, spec: None if args.resume else model)
    class Registry(EvidenceRegistry):
        def __init__(self):
            super().__init__(tmp_path / 'cli-exposure.sqlite')
        def import_prior_runs(self):
            return 0
    monkeypatch.setattr(evolution_cli, 'EvidenceRegistry', Registry)
    def isolated_run(*args, **kwargs):
        return run_evolution(*args, registry_path=tmp_path / 'evolution-exposure.sqlite',
                             evidence_source='synthetic_cli', **kwargs)
    monkeypatch.setattr(evolution_cli, 'run_evolution', isolated_run)


def ledger(directory):
    with sqlite3.connect(Path(directory) / 'state.sqlite') as connection:
        return connection.execute('SELECT sample,task,model,origin,value,role,settled,loss,status '
                                  'FROM predictions ORDER BY origin,sample,task,model').fetchall()


@pytest.mark.parametrize('mode', ['fixed', 'refit'])
def test_cli_without_worker_runs_real_fixed_or_refit_and_resumes(tmp_path, monkeypatch, mode):
    data, spec, model = fixture()
    policy = EvolutionPolicy(mode=mode, max_cycles=1, shadow_hours=16, update_every_hours=24)
    wire_cli(monkeypatch, tmp_path, data, model, policy)
    args = arguments(tmp_path, max_ticks=10)
    first = evolution_cli.run_cli(args, spec)
    assert first['ticks'] == 10 and set(first['tasks']) == {'return'}
    if mode == 'fixed':
        assert first['cycles'] == []
        assert first['incumbent'] == model.digest
    else:
        assert first['cycles'][0]['status'] == 'shadow'
        assert first['cycles'][0]['research']['method'] == 'same_structure_refit'
        assert first['cycles'][0]['candidate'] != model.digest
    args.resume, args.max_ticks = True, None
    finished = evolution_cli.run_cli(args, spec)
    before = ledger(args.output)
    again = evolution_cli.run_cli(args, spec)
    assert finished == again
    assert before == ledger(args.output)
    assert finished['ticks'] > first['ticks']
    assert finished['tasks']['return']['support'] > 0
    assert not (Path(args.output) / 'worker.log').exists()


def test_cli_rules_worker_consumes_each_cycle_exchange_and_stops(tmp_path, monkeypatch):
    data, spec, model = fixture()
    spec = replace(spec, search_seconds=4)
    policy = EvolutionPolicy(mode='evolve', search_mode='external', max_cycles=2,
                             shadow_hours=16, update_every_hours=24)
    wire_cli(monkeypatch, tmp_path, data, model, policy)
    args = arguments(tmp_path, worker_backend='rules', max_ticks=50)
    began = time.monotonic()
    result = evolution_cli.run_cli(args, spec)
    assert time.monotonic() - began < 15
    assert len(result['cycles']) == 2
    assert all(cycle['status'] in {'promoted', 'retained', 'unchanged_retained'}
        for cycle in result['cycles']), result['cycles']
    for cycle in result['cycles']:
        cycle_folder = Path(args.output) / 'research' / f'cycle-{cycle["id"]:04d}'
        exchange = cycle_folder / 'exchange'
        research = json.loads((cycle_folder / 'result.json').read_text('utf-8'))
        assert research['agent_kind'] != 'external_llm', 'A rules worker must not be claimed as an LLM'
        requests = list(exchange.glob('request-*.json'))
        proposals = list(exchange.glob('proposal-*.json'))
        assert requests and len(proposals) == len(requests)
        for path in requests:
            packet = json.loads(path.read_text('utf-8'))
            proposal = json.loads(path.with_name(path.name.replace('request-', 'proposal-')).read_text('utf-8'))
            assert proposal['source'] == 'deterministic_worker_rules'
            assert proposal['protocol_hash'] == packet['protocol_hash']
            assert proposal['parent_hash'] == packet['parent_hash']
            assert proposal['evidence_cutoff'] == packet['evidence_cutoff']
            assert {'operation': proposal['operation'], 'arguments': proposal['arguments']} in packet['actions']
    stop_files = list((Path(args.output) / 'worker').glob('stop-*.json'))
    assert len(stop_files) == 1
    assert json.loads(stop_files[0].read_text('utf-8'))['source'] == 'evolution_cli'
    worker_logs = [json.loads(line) for line in (Path(args.output) / 'worker.log').read_text('utf-8').splitlines()]
    assert worker_logs[-1]['stopped'] == 'requested'
    assert worker_logs[-1]['completed'] >= 2
    assert worker_logs[-1]['errors'] == 0


def test_cli_worker_is_stopped_when_core_run_raises(tmp_path, monkeypatch):
    data, spec, model = fixture()
    wire_cli(monkeypatch, tmp_path, data, model,
             EvolutionPolicy(search_mode='external', max_cycles=1, shadow_hours=16))
    def fail(*args, **kwargs):
        raise RuntimeError('controlled core failure')
    monkeypatch.setattr(evolution_cli, 'run_evolution', fail)
    args = arguments(tmp_path, worker_backend='rules')
    with pytest.raises(RuntimeError, match='controlled core'):
        evolution_cli.run_cli(args, spec)
    log = [json.loads(line) for line in (Path(args.output) / 'worker.log').read_text('utf-8').splitlines()]
    assert log[-1]['stopped'] == 'requested' and log[-1]['attempts'] == 0
    assert len(list((Path(args.output) / 'worker').glob('stop-*.json'))) == 1


@pytest.mark.parametrize('confirmed', [True, False])
def test_initial_accepted_selection_using_future_evidence_is_rejected_before_load(tmp_path, monkeypatch, confirmed):
    data, spec, _ = fixture()
    folder = tmp_path / 'prior'
    folder.mkdir()
    (folder / 'frozen_claim.json').write_text(json.dumps({'protocol': spec.to_dict()}), encoding='utf-8')
    result = {'confirmation': {'accepted_system': True}} if confirmed else {'confirmation': None}
    (folder / 'results.json').write_text(json.dumps(result), encoding='utf-8')
    monkeypatch.setattr(evolution_cli, 'verify_bundle', lambda *args: None)
    def unexpected(*args):
        raise AssertionError('A future-selected model must never be loaded')
    monkeypatch.setattr(ModelGroup, 'load', unexpected)
    args = arguments(tmp_path, initial_run=str(folder))
    with pytest.raises(ValueError, match='selection.*after'):
        evolution_cli._initial(args, data, spec)


def test_initial_resume_skips_loading_or_refitting(tmp_path, monkeypatch):
    data, spec, _ = fixture()
    def unexpected(*args, **kwargs):
        raise AssertionError('Resume must rely on the committed model')
    monkeypatch.setattr(ModelGroup, 'load', unexpected)
    monkeypatch.setattr(evolution_cli, 'fit', unexpected)
    assert evolution_cli._initial(arguments(tmp_path, resume=True, initial_run='nonexistent'), data, spec) is None


def test_initialize_only_saves_verified_common_checkpoint_without_running_replay(tmp_path, monkeypatch):
    data, spec, model = fixture()
    wire_cli(monkeypatch, tmp_path, data, model, EvolutionPolicy(mode='fixed'))
    def unexpected(*args, **kwargs):
        raise AssertionError('Initialization-only must not produce evolution forecasts')
    monkeypatch.setattr(evolution_cli, 'run_evolution', unexpected)
    args = arguments(tmp_path, initialize_only=True)
    result = evolution_cli.run_cli(args, spec)
    loaded = FittedResearchModel.load(result['checkpoint'])
    assert loaded.digest == model.digest == result['digest']
    assert result['cutoff'] == spec.train_end and loaded.schema == data.schema
    assert json.loads((Path(args.output) / 'initial.json').read_text('utf-8'))['digest'] == model.digest
    assert not (Path(args.output) / 'state.sqlite').exists()


def test_common_checkpoint_gives_fixed_refit_and_evolve_the_same_initial_identity(tmp_path, monkeypatch):
    data, spec, model = fixture()
    spec = replace(spec, search_seconds=4)
    checkpoint = tmp_path / 'common-model.pt'
    model.save(checkpoint)
    real_initial = evolution_cli._initial
    identities = []
    for mode in ('fixed', 'refit', 'evolve'):
        policy = EvolutionPolicy(mode=mode, max_cycles=1, shadow_hours=16)
        wire_cli(monkeypatch, tmp_path, data, model, policy)
        monkeypatch.setattr(evolution_cli, '_initial', real_initial)
        args = arguments(tmp_path, output=str(tmp_path / mode), max_ticks=1,
                         initial_checkpoint=str(checkpoint))
        result = evolution_cli.run_cli(args, spec)
        assert result['ticks'] == 1
        with sqlite3.connect(Path(args.output) / 'state.sqlite') as connection:
            state = json.loads(connection.execute('SELECT value FROM state WHERE id=1').fetchone()[0])
        identities.append(state['identity']['initial_digest'])
    assert identities == [model.digest] * 3


def test_initialize_only_rejects_future_fitted_model_before_checkpoint_write(tmp_path, monkeypatch):
    data, spec, model = fixture()
    model.metadata['fit_label_cutoff'] = spec.dev_end
    wire_cli(monkeypatch, tmp_path, data, model, EvolutionPolicy(mode='fixed'))
    args = arguments(tmp_path, initialize_only=True)
    with pytest.raises(ValueError, match='cutoff'):
        evolution_cli.run_cli(args, spec)
    assert not Path(args.output).exists()


def test_worker_launch_failure_closes_log(tmp_path, monkeypatch):
    data, spec, model = fixture()
    wire_cli(monkeypatch, tmp_path, data, model,
             EvolutionPolicy(search_mode='external', max_cycles=1, shadow_hours=16))
    handles = []
    original_open = Path.open
    def tracked_open(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        if path.name == 'worker.log':
            handles.append(handle)
        return handle
    monkeypatch.setattr(Path, 'open', tracked_open)
    def fail(*args, **kwargs):
        raise OSError('controlled worker launch failure')
    monkeypatch.setattr(subprocess, 'Popen', fail)
    with pytest.raises(OSError, match='controlled worker launch'):
        evolution_cli.run_cli(arguments(tmp_path, worker_backend='rules'), spec)
    assert len(handles) == 1 and handles[0].closed


def test_shutdown_failure_still_closes_worker_log(tmp_path, monkeypatch):
    from factorlab.discovery import agent_worker
    data, spec, model = fixture()
    wire_cli(monkeypatch, tmp_path, data, model,
             EvolutionPolicy(search_mode='external', max_cycles=1, shadow_hours=16))
    handles = []
    original_open = Path.open
    def tracked_open(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        if path.name == 'worker.log':
            handles.append(handle)
        return handle
    monkeypatch.setattr(Path, 'open', tracked_open)
    monkeypatch.setattr(subprocess, 'Popen', lambda *args, **kwargs: object())
    summary = {'ticks': 1, 'event_head': 'a' * 64, 'last_timestamp': spec.train_end}
    monkeypatch.setattr(evolution_cli, 'run_evolution', lambda *args, **kwargs: summary)
    def fail_stop(*args, **kwargs):
        raise OSError('controlled shutdown failure')
    monkeypatch.setattr(agent_worker, 'stop_worker', fail_stop)
    with pytest.raises(OSError, match='controlled shutdown'):
        evolution_cli.run_cli(arguments(tmp_path, worker_backend='rules'), spec)
    assert len(handles) == 1 and handles[0].closed
