"""Crash injection checks publication only; no fake statistical claims are made."""
import json

import pandas as pd
import pytest
import test_v7_joint_controller_audit as joint_tests

from factorlab.discovery import joint_pool, v7_run
from factorlab.discovery.joint_pool import DevelopmentPoolArchive
from factorlab.discovery.rolling import development_folds
from factorlab.discovery.v7_memory import ResearchMemory
from factorlab.discovery.v7_run import V7Config, run_month

workspace = joint_tests.workspace
bars = joint_tests.bars


@pytest.fixture
def tracked(monkeypatch):
    calls = {'evaluate': 0, 'admit': 0, 'prune': 0}

    def diagnostic(data, cutoff, pool, candidate, *, budget, context):
        calls['evaluate'] += 1
        budget.consume('solve')
        flat = {'candidate_loss': .9, 'reference_loss': 1., 'gain': .1,
                'absolute_delta': .1, 'winning_fold_fraction': 1.}
        return {'role': 'S', 'pool_hash': pool.digest, 'candidate_id': candidate.member_id,
            'context': context, 'receipt': {'cutoff': pd.Timestamp(cutoff).isoformat()},
            'folds': list(development_folds(cutoff)), 'F': {'Pv': flat, 'K0': flat},
            'P': flat, 'S': {'Pv': flat, 'K0': flat}}

    original_admit = DevelopmentPoolArchive.admit

    def admission(*args, **kwargs):
        calls['admit'] += 1
        return original_admit(*args, **kwargs)

    def prune(data, cutoff, pool, *, budget, max_remove):
        calls['prune'] += 1
        budget.consume('solve', 2)
        return pool, {'role': 'S', 'steps': [], 'removed_ids': [], 'affects_A0': False}

    monkeypatch.setattr(joint_pool, 'evaluate_candidate', diagnostic)
    monkeypatch.setattr(DevelopmentPoolArchive, 'admit', admission)
    monkeypatch.setattr(joint_pool, 'backward_prune', prune)
    monkeypatch.setattr(joint_pool, 'stability_select', joint_tests._force_one_admission)
    return calls


@pytest.mark.parametrize(('stage', 'damage'), [('archive', False), ('memory', False),
                                               ('month', False), ('archive', True)])
def test_durable_finalization_recovers_exact_publication_without_research_again(
        workspace, bars, tracked, monkeypatch, stage, damage):
    directory = workspace / 'finalization'
    config = V7Config(evaluation_mode='joint', formulas_per_family=1, lookback=4)
    if stage == 'archive':
        owner, attribute = DevelopmentPoolArchive, 'save'
    elif stage == 'memory':
        owner, attribute = ResearchMemory, 'save'
    else:
        owner, attribute = v7_run, 'immutable_json'
    original = getattr(owner, attribute)

    def interrupted(*args, **kwargs):
        result = original(*args, **kwargs)
        path = args[0] if stage == 'month' else args[1]
        if path.parent.name == {'archive': 'archives', 'memory': 'memories', 'month': 'months'}[stage]:
            state = joint_tests._state(directory, '2025-04')
            assert state['status'] == 'finalizing'
            assert state['summary']['development_pool'] == state['development_pool']
            assert state['joint_archive'] and state['research_memory']
            raise OSError('Synthetic interruption AFTER exact file publication')
        return result

    monkeypatch.setattr(owner, attribute, interrupted)
    with pytest.raises(OSError, match='AFTER exact'):
        run_month(directory, bars, '2025-04-01T00:00Z', config=config)
    committed = joint_tests._state(directory, '2025-04')
    before = dict(tracked)
    assert before['evaluate'] and before['admit'] == before['prune'] == 1
    assert committed['status'] == 'finalizing'
    assert (directory / 'archives' / '2025-04.json').is_file()
    monkeypatch.setattr(owner, attribute, original)

    def forbidden(*args, **kwargs):
        pytest.fail('Finalization restart must not evaluate, fit, select, admit, or prune again')

    for owner, name in ((joint_pool, 'evaluate_candidate'), (joint_pool, 'backward_prune'),
                        (joint_pool, 'stability_select'), (v7_run, 'fit_ridge'),
                        (DevelopmentPoolArchive, 'admit')):
        monkeypatch.setattr(owner, name, forbidden)
    if damage:
        path = directory / 'archives' / '2025-04.json'
        record = json.loads(path.read_text('utf-8'))
        record['archive']['epoch'] = '1999-01-01T00:00:00+00:00'
        path.write_text(json.dumps(record), encoding='utf-8')
        with pytest.raises(ValueError):
            run_month(directory, bars, '2025-04-01T00:00Z', config=config)
        assert joint_tests._state(directory, '2025-04')['status'] == 'finalizing'
        return
    resumed = run_month(directory, bars, '2025-04-01T00:00Z', config=config)
    state = joint_tests._state(directory, '2025-04')
    assert resumed == committed['summary']
    assert state == {**committed, 'status': 'complete'}
    assert tracked == before
    assert DevelopmentPoolArchive.load(directory / 'archives' / '2025-04.json').to_dict() == state['joint_archive']
    assert json.loads((directory / 'memories' / '2025-04.json').read_text('utf-8')) == state['research_memory']
    assert run_month(directory, bars, '2025-04-01T00:00Z', config=config) == resumed


def test_interrupted_prune_consumes_reservation_without_repeating_fit_or_prune(
        workspace, bars, tracked, monkeypatch):
    class HardInterruption(BaseException):
        pass

    directory = workspace / 'prune-interruption'
    config = V7Config(evaluation_mode='joint', formulas_per_family=1, lookback=4)

    def interrupted(data, cutoff, pool, *, budget, max_remove):
        tracked['prune'] += 1
        budget.consume('solve')
        raise HardInterruption()

    monkeypatch.setattr(joint_pool, 'backward_prune', interrupted)
    with pytest.raises(HardInterruption):
        run_month(directory, bars, '2025-04-01T00:00Z', config=config)
    state = joint_tests._state(directory, '2025-04')
    assert state['prune_attempted'] is True and not state.get('pruning')
    assert state['status'] == 'running'
    before = dict(tracked)

    def forbidden(*args, **kwargs):
        pytest.fail('An interrupted pruning reservation cannot obtain another fit')

    monkeypatch.setattr(joint_pool, 'backward_prune', forbidden)
    monkeypatch.setattr(joint_pool, 'evaluate_candidate', forbidden)
    monkeypatch.setattr(v7_run, 'fit_ridge', forbidden)
    result = run_month(directory, bars, '2025-04-01T00:00Z', config=config)
    recovered = joint_tests._state(directory, '2025-04')
    assert recovered['pruning']['status'] == 'interrupted_consumed'
    assert result['budget'] == state['budget']
    assert tracked['prune'] == before['prune'] == 1
    assert tracked['evaluate'] == before['evaluate']
    assert len(recovered['joint_archive']['admissions']) == 1
