"""Actual paired primary CI plus SQLite auxiliary veto; synthetic evidence only."""
import json

import numpy as np
import pandas as pd
import pytest

from factorlab.discovery.contracts import ExperimentSpec, TaskDataset, TaskSpec, TrainingSpec
from factorlab.discovery.evolution import EvolutionPolicy, EvolutionStore, _gate
from factorlab.discovery.robust_selection import RobustSelectionSpec
from factorlab.discovery.scoring import per_sample_loss


def gate_fixture(tmp_path, event_probability=.5):
    tasks = (TaskSpec('volatility', 'volatility', 2),
             TaskSpec('event', 'first_event', 2, (1, 2)), TaskSpec('state', 'state'))
    spec = ExperimentSpec(tasks=tasks, symbols=('X',), lookback=2,
        selection=RobustSelectionSpec(mode='walk_forward'),
        training=TrainingSpec(selection_task='volatility'),
        bootstrap_block_hours=4, min_blocks=2, bootstrap_samples=100,
        minimum_increment=.01)
    times = pd.date_range('2025-07-01', periods=24, freq='h', tz='UTC')
    data = TaskDataset(np.ones((24, 2, 1), dtype=np.float32),
        {'volatility': np.ones(24), 'event': np.zeros(24), 'state': np.zeros(24)},
        {t.name: np.ones(24, dtype=bool) for t in tasks},
        {t.name: (times + pd.Timedelta(hours=2)).to_numpy(dtype='datetime64[ns]')
         for t in tasks},
        pd.DataFrame({'sample_id': [str(i) for i in range(24)], 'symbol': 'X',
                      'available_at': times}), ('return_1h',), tasks)
    store = EvolutionStore(tmp_path / 'gate-ledger')
    candidate_event = [event_probability, *([(1 - event_probability) / 4] * 4)]
    for index in range(24):
        # Main candidate has uniformly lower loss: actual block CI is supported.
        # State is enormously worse and must have no predictive veto.
        for task in tasks:
            for model, role in (('reference', 'incumbent'), ('candidate', 'shadow')):
                value = (1.0 if model == 'candidate' else float(np.exp(1.0)))
                if task.kind == 'first_event':
                    value = candidate_event if model == 'candidate' else [.5, .125, .125, .125, .125]
                elif task.kind == 'state':
                    value = 1_000_000.0 if model == 'candidate' else 0.0
                loss = float(per_sample_loss(task, data.targets[task.name][index:index + 1],
                                             np.asarray([value]))[0])
                store.db.execute('''INSERT INTO predictions
                    (sample,task,model,origin,value,role,cycle,symbol,settled,loss,status,target)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''',
                    (str(index), task.name, model, times[index].isoformat(), json.dumps(value),
                     role, 1 if role == 'shadow' else None, 'X', 1, loss, 'scored',
                     float(data.targets[task.name][index])))
    return store, {'id': 1, 'reference': 'reference'}, data, spec, EvolutionPolicy(max_cycles=1)


@pytest.mark.parametrize('condition,expected', [
    ('event_degrades', False), ('only_state_degrades', True),
    ('candidate_event_immature', False), ('reference_event_immature', False),
    ('missing_reference_event', False), ('missing_candidate_event', False),
    ('insufficient_event_blocks', False),
])
def test_actual_primary_ci_cannot_override_auxiliary_veto(tmp_path, condition, expected):
    store, cycle, data, spec, policy = gate_fixture(tmp_path,
                                                   .1 if condition == 'event_degrades' else .5)
    try:
        if condition in {'candidate_event_immature', 'reference_event_immature'}:
            model = 'candidate' if condition.startswith('candidate') else 'reference'
            store.db.execute("UPDATE predictions SET settled=0,loss=NULL,status=NULL "
                             "WHERE sample='23' AND task='event' AND model=?", (model,))
        elif condition == 'missing_reference_event':
            store.db.execute("DELETE FROM predictions WHERE sample='23' AND task='event' "
                             "AND model='reference'")
        elif condition == 'missing_candidate_event':
            store.db.execute("DELETE FROM predictions WHERE sample='23' AND task='event' "
                             "AND model='candidate'")
        elif condition == 'insufficient_event_blocks':
            store.db.execute("UPDATE predictions SET loss=NULL,status='unobserved' "
                             "WHERE CAST(sample AS INTEGER)>=4 AND task='event'")
        result = _gate(store, cycle, data, spec, policy)
        # This assertion uses the real scoring/bootstrap path, not a mocked CI.
        assert result['ci_lower'] > spec.minimum_increment
        assert result.get('primary_comparison_status', result['status']) == 'supported'
        assert result['operational_development_improvement_supported'] is expected
        assert result['predictive_claim_supported'] is False
        assert result['independent_confirmation'] is False
        assert result['one_look_only'] is True
        guard = result['auxiliary_guard']
        assert guard['passed'] is expected
        assert 'state' not in guard['tasks']
        if condition == 'event_degrades':
            assert guard['tasks']['event']['relative_degradation'] > 2
            assert guard['tasks']['event']['reason'] == 'auxiliary_degradation_limit_exceeded'
        elif condition in {'candidate_event_immature', 'reference_event_immature',
                           'missing_reference_event'}:
            assert guard['tasks']['event']['reason'] == 'incomplete_mature_auxiliary_pairs'
        elif condition == 'insufficient_event_blocks':
            assert guard['tasks']['event']['reason'] == 'insufficient_auxiliary_calendar_blocks'
        elif condition == 'missing_candidate_event':
            assert guard['tasks']['event']['reason'] == 'auxiliary_origin_cohort_mismatch'
        if not expected:
            assert result['status'] == 'auxiliary_guard_failed'
    finally:
        store.close()
