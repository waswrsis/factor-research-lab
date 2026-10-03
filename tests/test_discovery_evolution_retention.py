"""Unchanged research selection and measured clocks consume the cycle budget."""
import sqlite3

import pytest
from test_discovery_evolution import fixture, run

from factorlab.discovery import evolution
from factorlab.discovery.compare_evolution import _read, _resources
from factorlab.discovery.evolution import EvolutionPolicy, inspect_evolution


def unchanged_builder(_train, _dev, incumbent, _spec, _policy, _directory):
    return incumbent, {'method': 'controlled_original_selection', 'elapsed_seconds': .125,
                       'records': [{'status': 'ok', 'reason': 'original_beat_mutations'}]}


def test_unchanged_candidate_retains_training_search_evidence_without_shadow_and_spends_cycle(tmp_path):
    data, spec, model = fixture()
    policy = EvolutionPolicy(max_cycles=1, shadow_hours=16)
    result = run(tmp_path, 'unchanged', data, spec, model, policy=policy,
                 candidate_builder=unchanged_builder)
    assert result['incumbent'] == model.digest and result['promotions'] == 0
    assert result['pending_cycle'] is None and len(result['cycles']) == 1
    cycle = result['cycles'][0]
    assert cycle['status'] == 'unchanged_retained' and cycle['candidate'] == model.digest
    assert cycle['training']['train_rows'] > 0 and cycle['training']['early_stopping_rows'] > 0
    assert cycle['research']['records'] == [{'status': 'ok', 'reason': 'original_beat_mutations'}]
    assert cycle['research']['candidate_search_elapsed_seconds'] == .125
    assert cycle['research']['elapsed_seconds'] > 0
    assert 'shadow_start' not in cycle and 'gate' not in cycle
    with sqlite3.connect(tmp_path / 'unchanged' / 'state.sqlite') as db:
        assert db.execute('SELECT COUNT(*) FROM models').fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM predictions WHERE role='shadow'").fetchone()[0] == 0
    resource = _resources(_read(tmp_path / 'unchanged'))
    assert resource['failed_or_interrupted_cycles'] == 0
    assert resource['known_search_seconds'] == cycle['research']['elapsed_seconds']
    assert resource['search_seconds_complete'] is True
    assert inspect_evolution(tmp_path / 'unchanged') == result
    resumed = run(tmp_path, 'unchanged', data, spec, None, policy=policy,
                  candidate_builder=unchanged_builder, resume=True)
    assert resumed == result


@pytest.mark.parametrize('mode', ['refit', 'evolve'])
def test_caught_candidate_failure_records_elapsed_wall_time_and_known_training(tmp_path, monkeypatch, mode):
    data, spec, model = fixture()

    def fail(*_args, **_kwargs):
        raise RuntimeError('controlled partial fitting failure')

    monkeypatch.setattr(evolution, '_research_candidate', fail)
    result = run(tmp_path, mode, data, spec, model,
                 policy=EvolutionPolicy(mode=mode, max_cycles=1, shadow_hours=16))
    cycle = result['cycles'][0]
    assert cycle['status'] == 'failed' and result['incumbent'] == model.digest
    assert cycle['training']['train_rows'] > 0
    assert cycle['research']['elapsed_seconds'] > 0
    assert 'no FLOPs or API monetary cost measurement' in cycle['research']['resource_scope']
    resource = _resources(_read(tmp_path / mode))
    assert resource['known_search_seconds'] == cycle['research']['elapsed_seconds']
    assert resource['search_seconds_complete'] is True
    assert resource['failed_or_interrupted_cycles'] == 1


def interrupted_builder(*_args):
    raise KeyboardInterrupt('controlled incomplete search clock')


def test_interrupted_candidate_clock_remains_explicitly_unmeasured(tmp_path):
    data, spec, model = fixture()
    policy = EvolutionPolicy(max_cycles=1, shadow_hours=16)
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, 'interrupted', data, spec, model, policy=policy,
            candidate_builder=interrupted_builder)
    result = run(tmp_path, 'interrupted', data, spec, None, policy=policy,
                 candidate_builder=interrupted_builder, resume=True)
    assert result['cycles'][0]['status'] == 'interrupted_slot_consumed'
    resource = _resources(_read(tmp_path / 'interrupted'))
    assert resource['known_search_seconds'] == 0 and resource['unmeasured_cycle_ids'] == [1]
    assert resource['search_seconds_complete'] is False


def test_successful_default_refit_records_wrapper_clock_for_comparison(tmp_path):
    data, spec, model = fixture()
    result = run(tmp_path, 'successful-refit', data, spec, model,
                 policy=EvolutionPolicy(mode='refit', max_cycles=1, shadow_hours=16), max_ticks=2)
    cycle = result['cycles'][0]
    assert cycle['status'] == 'shadow'
    assert cycle['research']['method'] == 'same_structure_refit'
    assert cycle['research']['elapsed_seconds'] > 0
    resource = _resources(_read(tmp_path / 'successful-refit'))
    assert resource['known_search_seconds'] == cycle['research']['elapsed_seconds']
    assert resource['unmeasured_cycle_ids'] == []
