"""Synthetic regressions for fold summaries and chronological prompt memory."""
import json
from copy import deepcopy
from dataclasses import asdict

import pandas as pd
import pytest
from test_discovery_robust_selection import fake_fitter, fixture

from factorlab.discovery import population, search
from factorlab.discovery.agent_worker import sanitize_request
from factorlab.discovery.contracts import TaskSpec
from factorlab.discovery.research_memory import (
    ResearchMemory,
    clean_metrics,
    to_development_history,
)
from factorlab.discovery.robust_selection import summarize_fold_metrics
from factorlab.protocol import fingerprint


def fold_scores(task):
    return [{task.name: {
        'kind': task.kind, 'predictive': task.predictive, 'status': 'ok',
        'support': support, 'eligible_support': support, 'total_rows': 100,
        'n_hours': support, 'loss': loss, 'coverage': support / 100,
        'label_coverage': support / 100, 'brier': brier, 'ece': brier / 2,
        'aggregation': 'equal_symbol_within_hour_then_equal_hour',
    }} for support, loss, brier in ((100, 1, .06), (80, 4, .5), (50, 9, .9))]


def emit_packet(tmp_path, spec, root, history, iteration=100):
    with pytest.raises(TimeoutError, match='allotted time'):
        search.propose(root, spec, history, iteration, mode='external', exchange=tmp_path,
                       timeout=0, features=('x',), program_search=True)
    return json.loads((tmp_path / f'request-{iteration}.json').read_text('utf-8'))


def test_fold_summary_counts_scope_and_equal_fold_loss_without_bogus_probability_pooling():
    task = TaskSpec('event', 'first_event', 12)
    scores = fold_scores(task)
    original = deepcopy(scores)
    summary = summarize_fold_metrics(scores, (task,))['event']
    assert summary['loss'] == pytest.approx(14 / 3)
    assert summary['loss'] != pytest.approx((100 + 80 * 4 + 50 * 9) / 230)
    assert summary['support'] == summary['eligible_support'] == summary['n_hours'] == 230
    assert summary['total_rows'] == 300
    assert summary['coverage'] == summary['label_coverage'] == pytest.approx(230 / 300)
    assert summary['fold_count'] == 3
    assert summary['aggregation'] == 'equal_fold_of_equal_symbol_within_hour_then_equal_hour'
    assert summary['support_scope'] == 'fold_observations_including_repeats'
    assert summary['coverage_scope'] == 'fold_rows_including_repeats'
    assert 'brier' not in summary and 'ece' not in summary
    assert scores == original
    repeated = summarize_fold_metrics([scores[0], scores[0]], (task,))['event']
    assert repeated['support'] == repeated['n_hours'] == 200
    assert repeated['support_scope'] == summary['support_scope']


def test_incomplete_descriptive_task_does_not_inherit_success_or_hours_from_first_fold():
    task = TaskSpec('state', 'state', 1)
    scores = fold_scores(task)
    scores[1]['state'].update(status='unavailable', support=0, loss=None, coverage=0)
    del scores[1]['state']['n_hours']
    row = summarize_fold_metrics(scores, (task,))['state']
    assert row['status'] == 'insufficient_evidence' and row['loss'] is None
    assert row['predictive'] is False and row['interpretation'] == 'descriptive_only'
    assert row['support'] == 150 and row['eligible_support'] == 230
    assert row['coverage'] == .5 and 'n_hours' not in row
    with pytest.raises(ValueError, match='at least one fold'):
        summarize_fold_metrics([], (task,))


def test_population_summary_and_ranking_preserved_and_roundtrip_through_memory_worker(tmp_path):
    train, dev, root, other, spec = fixture(seeds=(17,))
    # Different mature support per fold makes inheriting the first-fold counts visible.
    for index, support in enumerate((100, 80, 50)):
        dev.observed['ret'][index * 100 + support:(index + 1) * 100] = False
    losses = {root.digest: {'ret': [1, 4, 9], 'aux': [1] * 3},
              other.digest: {'ret': [.8, 3.2, 7.2], 'aux': [1] * 3}}
    _, result = population.search_population(train, dev, (root, other), spec,
                                            fit_candidate=fake_fitter([], losses))
    candidate = result['records'][1]
    assert result['architecture'] == asdict(other)
    assert candidate['comparison']['relative_gains'] == pytest.approx([.2] * 3)
    assert candidate['comparison']['rank_score'] == pytest.approx(.2)
    assert candidate['primary_loss'] == pytest.approx((.8 + 3.2 + 7.2) / 3)
    summary = candidate['metrics']['ret']
    assert summary['support'] == summary['eligible_support'] == summary['n_hours'] == 230
    assert summary['total_rows'] == 300 and summary['coverage'] == pytest.approx(230 / 300)
    assert candidate['folds'][0]['metrics']['ret']['support'] == 100
    assert candidate['folds'][0]['metrics']['ret']['aggregation'] == 'equal_symbol_within_hour_then_equal_hour'
    memory = ResearchMemory(tmp_path / 'metrics.sqlite')
    memory.record_action_result(task=spec.tasks[0], evidence_cutoff=spec.dev_end,
        protocol_hash=spec.digest, parent_hash=root.digest, candidate_id=other.digest,
        action={'operation': 'width', 'arguments': {'name': 'root', 'width': 5}},
        metrics=candidate['metrics'])
    restored = to_development_history(memory.retrieve(spec.tasks[0], spec.dev_end), spec)
    cleaned = search._clean_history(restored, spec)
    assert cleaned[0]['metrics']['ret'] == summary
    packet = emit_packet(tmp_path / 'packet', spec, root, cleaned)
    assert sanitize_request(packet)['history'][0]['metrics']['ret'] == summary
    for field in ('kind', 'aggregation', 'coverage_scope', 'support_scope'):
        invalid = {**summary, field: 'read_future_results'}
        with pytest.raises(ValueError, match='bounded metric'):
            clean_metrics({'ret': invalid})


def recorded_history(tmp_path):
    _, _, root, _, spec = fixture(seeds=(17,))
    memory = ResearchMemory(tmp_path / 'order.sqlite')
    identities = [fingerprint(['history', i]) for i in range(30)]
    for index, identity in enumerate(identities):
        memory.record_action_result(task=spec.tasks[0], evidence_cutoff=spec.dev_end,
            protocol_hash=spec.digest, parent_hash=root.digest, candidate_id=identity,
            event_id=f'event-{index:02d}',
            action={'operation': 'width', 'arguments': {'name': 'root', 'width': 5}},
            metrics={'ret': {'status': 'ok', 'loss': 1}})
    # Fix recording timestamps so ordering is deterministic and tests tie handling.
    with memory._connect() as connection:
        for index in range(30):
            stamp = (pd.Timestamp('2024-01-01', tz='UTC') + pd.Timedelta(seconds=index // 2)).isoformat()
            connection.execute('UPDATE action_results SET recorded_at=? WHERE event_id=?',
                               (stamp, f'event-{index:02d}'))
    rows = memory.retrieve(spec.tasks[0], spec.dev_end, limit=30)
    assert [row['candidate_id'] for row in rows] == identities[::-1]
    history = to_development_history(rows, spec)
    assert [row['candidate_id'] for row in history] == identities
    assert [row['iteration'] for row in history] == list(range(30))
    assert [row['candidate_id'] for row in rows] == identities[::-1], 'Conversion must not mutate retrieval'
    assert [row['candidate_id'] for row in to_development_history(rows[::2], spec)] == identities[1::2]
    return root, spec, history


def test_thirty_retrieved_facts_then_current_appends_keep_latest_twelve(tmp_path):
    root, spec, history = recorded_history(tmp_path)
    for index in range(30, 34):
        row = deepcopy(history[-1])
        row.update(iteration=index, candidate_id=fingerprint(['current', index]))
        history.append(row)
    packet = emit_packet(tmp_path / 'packet', spec, root, history)
    assert [row['candidate_id'] for row in packet['history']] == [row['candidate_id'] for row in history[-12:]]
    assert [row['iteration'] for row in packet['history']] == list(range(22, 34))
    assert len(sanitize_request(packet)['history']) == 12


def test_byte_budget_drops_oldest_whole_records_and_preserves_recent_current_facts(tmp_path):
    root, spec, history = recorded_history(tmp_path)
    for index in range(30, 34):
        row = deepcopy(history[-1])
        row.update(iteration=index, candidate_id=fingerprint(['current', index]))
        row['proposal'].update(hypothesis='证' * 3000, falsification='据' * 3000)
        history.append(row)
    packet = emit_packet(tmp_path / 'packet', spec, root, history)
    context = {key: value for key, value in packet.items() if key != 'request_hash'}
    assert len(json.dumps(context).encode()) <= 90000
    retained = packet['history']
    assert 1 < len(retained) < 4
    assert [row['candidate_id'] for row in retained] == [row['candidate_id'] for row in history[-len(retained):]]
    assert all(row['iteration'] >= 30 for row in retained)
    assert len(sanitize_request(packet)['history']) == len(retained)


def test_single_oversized_latest_fact_fails_closed_without_writing_request(tmp_path):
    root, spec, history = recorded_history(tmp_path)
    history[-1]['proposal']['hypothesis'] = '证' * 20000
    with pytest.raises(ValueError, match='byte budget'):
        search.propose(root, spec, history, 100, mode='external', exchange=tmp_path,
                       timeout=0, features=('x',), program_search=True)
    assert not (tmp_path / 'request-100.json').exists()
