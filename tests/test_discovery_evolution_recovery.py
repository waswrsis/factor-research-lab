"""Crash recovery and persisted-ledger tampering regressions on artificial data."""
import json
import sqlite3
from dataclasses import replace

import numpy as np
import pytest
from test_discovery_evolution import fixture, predictions, run

from factorlab.discovery import evolution
from factorlab.discovery.contracts import TaskDataset, TaskSpec
from factorlab.discovery.evolution import EvolutionPolicy, inspect_evolution
from factorlab.discovery.learning import fit


@pytest.mark.parametrize('edit', [
    "UPDATE predictions SET value='1000000' WHERE rowid=(SELECT MIN(rowid) FROM predictions)",
    'UPDATE predictions SET loss=12345 WHERE settled=1',
    'DELETE FROM predictions WHERE rowid=(SELECT MIN(rowid) FROM predictions)',
])
def test_resume_and_inspection_reject_edited_prediction_ledger(tmp_path, edit):
    data, spec, model = fixture()
    policy = EvolutionPolicy(mode='fixed')
    run(tmp_path, 'ledger', data, spec, model, policy=policy, max_ticks=4)
    with sqlite3.connect(tmp_path / 'ledger' / 'state.sqlite') as db:
        db.execute(edit)
    with pytest.raises(ValueError, match='ledger|integrity|changed|checkpoint'):
        inspect_evolution(tmp_path / 'ledger')
    with pytest.raises(ValueError, match='ledger|integrity|changed|checkpoint'):
        run(tmp_path, 'ledger', data, spec, None, policy=policy, resume=True)


def rejected_builder(*_args):
    raise RuntimeError('declared synthetic rejection')


def test_resume_rejects_budget_counter_edit_in_state_table(tmp_path):
    data, spec, model = fixture()
    policy = EvolutionPolicy(max_cycles=1, shadow_hours=16, update_every_hours=24)
    result = run(tmp_path, 'budget', data, spec, model, policy=policy,
                 candidate_builder=rejected_builder, max_ticks=1)
    assert len(result['cycles']) == 1
    with sqlite3.connect(tmp_path / 'budget' / 'state.sqlite') as db:
        state = json.loads(db.execute('SELECT value FROM state WHERE id=1').fetchone()[0])
        state['cycles'] = 0
        db.execute('UPDATE state SET value=? WHERE id=1', (json.dumps(state),))
    with pytest.raises(ValueError, match='ledger|integrity|changed|checkpoint'):
        run(tmp_path, 'budget', data, spec, None, policy=policy,
            candidate_builder=rejected_builder, resume=True)


def test_resume_rejects_cycle_record_edit(tmp_path):
    data, spec, model = fixture()
    policy = EvolutionPolicy(max_cycles=1, shadow_hours=16)
    run(tmp_path, 'cycle', data, spec, model, policy=policy,
        candidate_builder=rejected_builder, max_ticks=1)
    with sqlite3.connect(tmp_path / 'cycle' / 'state.sqlite') as db:
        cycle = json.loads(db.execute('SELECT value FROM cycles WHERE id=1').fetchone()[0])
        cycle['status'] = 'promoted'
        db.execute('UPDATE cycles SET value=? WHERE id=1', (json.dumps(cycle),))
    with pytest.raises(ValueError, match='ledger|integrity|changed|checkpoint'):
        inspect_evolution(tmp_path / 'cycle')


def interrupted_builder(*_args):
    raise KeyboardInterrupt('declared interruption during search')


def test_search_interruption_consumes_slot_and_preserves_incumbent(tmp_path):
    data, spec, model = fixture()
    policy = EvolutionPolicy(max_cycles=1, shadow_hours=16)
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, 'interrupt', data, spec, model, policy=policy,
            candidate_builder=interrupted_builder)
    resumed = run(tmp_path, 'interrupt', data, spec, None, policy=policy,
                  candidate_builder=interrupted_builder, resume=True)
    assert len(resumed['cycles']) == 1
    assert resumed['cycles'][0]['status'] == 'interrupted_slot_consumed'
    assert resumed['incumbent'] == model.digest and resumed['pending_cycle'] is None
    assert resumed['ticks'] > 1


def test_interrupted_tick_rolls_back_prediction_and_settlement_then_resumes_exactly(tmp_path, monkeypatch):
    data, spec, model = fixture()
    policy = EvolutionPolicy(mode='fixed')
    run(tmp_path, 'full', data, spec, model, policy=policy)
    original = evolution._record_predictions
    calls = 0

    def interrupt_after_insert(*args, **kwargs):
        nonlocal calls
        original(*args, **kwargs)
        calls += 1
        if calls == 4:
            raise KeyboardInterrupt('declared interruption after row insertion')

    monkeypatch.setattr(evolution, '_record_predictions', interrupt_after_insert)
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, 'atomic', data, spec, model, policy=policy)
    with sqlite3.connect(tmp_path / 'atomic' / 'state.sqlite') as db:
        state = json.loads(db.execute('SELECT value FROM state WHERE id=1').fetchone()[0])
        assert state['ticks'] == 3
        assert db.execute('SELECT COUNT(*) FROM predictions').fetchone()[0] == 3
    monkeypatch.setattr(evolution, '_record_predictions', original)
    run(tmp_path, 'atomic', data, spec, None, policy=policy, resume=True)
    assert predictions(tmp_path / 'atomic') == predictions(tmp_path / 'full')


def missing_auxiliary_head_builder(train, dev, _incumbent, spec, _policy, _directory):
    # The dataset retains the declared schema while this candidate lacks one head.
    train.observed['state'][:] = False
    return fit(train, dev, None, spec.training), {'fixture': 'missing_auxiliary_head'}


def test_shadow_savepoint_removes_partial_rows_when_later_task_head_is_missing(tmp_path):
    data, spec, _model = fixture()
    tasks = (*data.tasks, TaskSpec('state', 'state', 2))
    data = TaskDataset(data.x, {**data.targets, 'state': np.ones(len(data.x))},
        {**data.observed, 'state': np.ones(len(data.x), dtype=bool)},
        {**data.maturity, 'state': data.maturity['return'].copy()}, data.meta, data.features, tasks)
    spec = replace(spec, tasks=tasks)
    model = fit(data.subset(np.arange(35)), data.subset(np.arange(40, 60)), None, spec.training)
    result = run(tmp_path, 'partial-shadow', data, spec, model,
                 candidate_builder=missing_auxiliary_head_builder, max_ticks=2)
    assert result['cycles'][0]['status'] == 'invalid_candidate'
    assert result['pending_cycle'] is None and result['incumbent'] == model.digest
    with sqlite3.connect(tmp_path / 'partial-shadow' / 'state.sqlite') as db:
        assert db.execute("SELECT COUNT(*) FROM predictions WHERE role='shadow'").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM predictions WHERE role='incumbent'").fetchone()[0] == 4
    assert inspect_evolution(tmp_path / 'partial-shadow')['event_head'] == result['event_head']
