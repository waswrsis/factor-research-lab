import sqlite3
from dataclasses import replace
from types import SimpleNamespace

import pytest

from factorlab.discovery.contracts import ExperimentSpec, TaskSpec, TrainingSpec
from factorlab.discovery.evolution import _shadow_auxiliary_guard
from factorlab.discovery.robust_selection import RobustSelectionSpec


def fixture(candidate_loss=1.0, reference_loss=1.0, settled=1):
    spec = ExperimentSpec(tasks=(TaskSpec(), TaskSpec('event', 'first_event'), TaskSpec('state', 'state')),
        selection=RobustSelectionSpec(mode='walk_forward'),
        training=TrainingSpec(selection_task='volatility'), bootstrap_block_hours=4, min_blocks=2)
    db = sqlite3.connect(':memory:')
    db.row_factory = sqlite3.Row
    db.execute('CREATE TABLE predictions(sample,task,model,origin,symbol,loss,settled,cycle,role)')
    for index in range(12):
        for task in ('volatility', 'event', 'state'):
            for model, loss, role in (('r', reference_loss, 'incumbent'),
                                      ('c', candidate_loss if task == 'event' else 1000000, 'shadow')):
                db.execute('INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?)',
                    (str(index), task, model, f'2025-01-01T{index:02d}:00:00+00:00', 'X', loss,
                     settled, 1 if role == 'shadow' else None, role))
    return SimpleNamespace(db=db), {'reference': 'r', 'id': 1}, spec


def test_auxiliary_degradation_blocks_even_with_primary_improvement():
    store, cycle, spec = fixture(1.31)
    result = _shadow_auxiliary_guard(store, cycle, spec)
    assert result['passed'] is False
    assert result['tasks']['event']['relative_degradation'] == pytest.approx(.31)
    assert 'state' not in result['tasks']
    store.db.close()


@pytest.mark.parametrize('candidate_loss', [1.09, 1.1])
def test_within_limit_passes_and_state_does_not_block(candidate_loss):
    store, cycle, spec = fixture(candidate_loss)
    assert _shadow_auxiliary_guard(store, cycle, spec)['passed'] is True
    store.db.close()


@pytest.mark.parametrize('loss,settled,reason', [
    (None, 1, 'auxiliary_pair_observation_mismatch'),
    (1.0, 0, 'incomplete_mature_auxiliary_pairs'),
])
def test_missing_or_immature_pairs_never_pass(loss, settled, reason):
    store, cycle, spec = fixture(loss, settled=settled)
    result = _shadow_auxiliary_guard(store, cycle, spec)
    assert not result['passed']
    assert result['tasks']['event']['reason'] == reason
    store.db.close()


def test_support_and_zero_reference_checks():
    store, cycle, spec = fixture(1.0, 0.0)
    assert not _shadow_auxiliary_guard(store, cycle, spec)['passed']
    spec = replace(spec, min_blocks=4)
    result = _shadow_auxiliary_guard(store, cycle, spec)
    assert result['tasks']['event']['reason'] == 'insufficient_auxiliary_calendar_blocks'
    store.db.close()
