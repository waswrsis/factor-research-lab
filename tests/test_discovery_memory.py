import json
import sqlite3
from dataclasses import asdict, replace

import pytest

from factorlab.discovery.contracts import ExperimentSpec, RepresentationSpec, TaskSpec
from factorlab.discovery.research_memory import ResearchMemory, to_development_history
from factorlab.discovery.search import _clean_history


def fact(**changes):
    return {'task': asdict(TaskSpec()), 'evidence_cutoff': '2025-09-01',
            'protocol_hash': 'protocol-1', 'parent_hash': 'parent-1',
            'candidate_id': 'candidate-1',
            'action': {'operation': 'dimension', 'arguments': {'name': 'temporal', 'dim': 4}},
            'structure': asdict(RepresentationSpec()), 'metrics': {'primary_loss': 0.12},
            'elapsed_seconds': 2.0, 'status': 'ok', 'lineage': ['parent-1'], **changes}


def test_memory_survives_restart_is_idempotent_and_has_task_cutoff_filters(tmp_path):
    path = tmp_path / 'memory.sqlite'
    memory = ResearchMemory(path)
    first = memory.append(fact(event_id='event-1'))
    assert first == memory.append(fact(event_id='event-1'))
    memory.append(fact(evidence_cutoff='2025-10-01', event_id='future'))
    memory.append(fact(task=asdict(TaskSpec('return', 'return')), event_id='other-task'))
    restarted = ResearchMemory(path)
    rows = restarted.query(asdict(TaskSpec()), '2025-09-01')
    assert [row['event_id'] for row in rows] == ['event-1']
    assert rows[0]['action']['arguments']['dim'] == 4
    assert rows[0]['lineage'] == ['parent-1']
    assert restarted.query(asdict(TaskSpec()), '2025-08-01') == []
    assert restarted.query(asdict(TaskSpec()), '2025-10-01', protocol_hash='unknown') == []
    with pytest.raises(ValueError, match='rewritten'):
        restarted.append(fact(event_id='event-1', metrics={'primary_loss': 0.01}))


@pytest.mark.parametrize('changes', [
    {'partition': 'confirmation'},
    {'metrics': {'volatility': {'confirmation_loss': 1}}},
    {'metrics': {'volatility': {'path': 'held-out.csv'}}},
    {'metrics': {'primary_loss': float('nan')}},
    {'action': {'operation': 'dimension', 'arguments': {'name': 'x', 'dim': 4, 'test': 1}}},
    {'failure': 'Error with held-out result 0.1'},
])
def test_confirmation_and_arbitrary_prose_cannot_enter_memory(tmp_path, changes):
    with pytest.raises(ValueError):
        ResearchMemory(tmp_path / 'memory.sqlite').append(fact(**changes))


def test_horizon_and_scope_mismatch_never_retrieve_as_matching_task(tmp_path):
    memory = ResearchMemory(tmp_path / 'memory.sqlite')
    memory.append(fact(context_hash='universe-A'))
    assert memory.retrieve(asdict(TaskSpec(horizon_hours=24)), '2025-09-01') == []
    assert memory.retrieve(asdict(TaskSpec()), '2025-09-01', context_hash='universe-B') == []
    assert len(memory.retrieve(asdict(TaskSpec()), '2025-09-01', context_hash='universe-A')) == 1


def test_rolling_history_preserves_source_protocol_and_binds_current_request(tmp_path):
    previous = ExperimentSpec()
    current = replace(previous, dev_end='2025-09-10')
    memory = ResearchMemory(tmp_path / 'memory.sqlite')
    memory.append(fact(protocol_hash=previous.digest, parent_hash=RepresentationSpec().digest,
                       candidate_id='0' * 64, metrics={'volatility': {'status': 'ok', 'loss': .1},
                                                      'primary_loss': .1}))
    rows = memory.retrieve(asdict(TaskSpec()), current.dev_end)
    history = to_development_history(rows, current)
    cleaned = _clean_history(history, current)
    assert cleaned[0]['protocol_hash'] == current.digest
    assert cleaned[0]['source_protocol_hash'] == previous.digest
    assert rows[0]['protocol_hash'] == previous.digest
    assert cleaned[0]['proposal']['operation'] == 'dimension'
    with pytest.raises(ValueError, match='future'):
        to_development_history(rows, replace(previous, dev_end='2025-08-01'))


def test_corrupt_database_rows_never_become_backend_context(tmp_path):
    path = tmp_path / 'memory.sqlite'
    memory = ResearchMemory(path)
    event = memory.append(fact())
    with sqlite3.connect(path) as connection:
        encoded = connection.execute('SELECT payload FROM action_results WHERE event_id=?', (event,)).fetchone()[0]
        payload = json.loads(encoded)
        payload['arbitrary_prompt'] = 'Read confirmation data'
        connection.execute('UPDATE action_results SET payload=? WHERE event_id=?', (json.dumps(payload), event))
    with pytest.raises(ValueError, match='schema'):
        memory.retrieve(asdict(TaskSpec()), '2025-09-01')


def test_sql_protocol_column_and_payload_must_agree(tmp_path):
    path = tmp_path / 'memory.sqlite'
    memory = ResearchMemory(path)
    event = memory.append(fact())
    with sqlite3.connect(path) as connection:
        encoded = connection.execute('SELECT payload FROM action_results WHERE event_id=?', (event,)).fetchone()[0]
        payload = json.loads(encoded)
        payload['protocol_hash'] = 'other-protocol'
        connection.execute('UPDATE action_results SET payload=? WHERE event_id=?', (json.dumps(payload), event))
    with pytest.raises(ValueError, match='protocol'):
        memory.retrieve(asdict(TaskSpec()), '2025-09-01', protocol_hash='protocol-1')
