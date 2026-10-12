"""Synthetic controls, fake-provider accounting and fixed protocol tests."""
import copy
import json
from types import SimpleNamespace

import pandas as pd
import pytest
from test_v7_memory_controller_audit import CaptureDuplicateBackend
from test_v7_rolling_registry import candles
from test_v7_run_factor_pool import source_root

from factorlab.discovery import v7_controls
from factorlab.discovery.rolling import build_rolling_data, development_folds
from factorlab.discovery.v7_agent import BackendReply
from factorlab.discovery.v7_run import V7Config


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    return tmp_path, candles('2024-08-01', '2025-04-02'), '2025-04-01T00:00:00Z'


class StubBackend(CaptureDuplicateBackend):
    """Six concrete duplicates each batch; no actual provider or known tokens."""
    def __call__(self, packet, **kwargs):
        reply = super().__call__(packet, **kwargs)
        value = json.loads(reply.payload)
        value['programs'] *= 3
        return BackendReply(json.dumps(value), None, 'synthetic-stub', 'control-smoke')


def test_real_three_arm_synthetic_smoke_same_budget_no_claim_and_no_replay(fixture):
    root, bars, cutoff = fixture
    backend = StubBackend()
    report = v7_controls.run_controls(root / 'controls', bars, cutoff, backend=backend)
    assert set(report['arms']) == {'local', 'random', 'gpt'}
    assert report['formal_claims_enabled'] is False and report['V_queries'] == report['C_queries'] == 0
    ceilings = [value['readout']['actual_budget']['limits'] for value in report['arms'].values()]
    assert ceilings[0] == ceilings[1] == ceilings[2]
    assert ceilings[0]['ast'] == 96 and ceilings[0]['f'] == 24 and ceilings[0]['solve'] == 1600
    assert all(value['failure'] is None for value in report['arms'].values())
    costs = report['arms']['gpt']['readout']['costs']
    assert 1 <= costs['provider_calls'] <= 6 and costs['input_characters'] > 0
    assert costs['known_reported_input_tokens'] is None
    assert costs['passing_per_1000_reported_input_tokens'] is None
    assert report['arms']['gpt']['readout']['duplicate_breakdown']['validated_reply_seed_duplicates'] == 15
    assert len({v['readout']['pool_reference_hash'] for v in report['arms'].values()}) == 1
    count = len(backend.packets)
    repeated = v7_controls.run_controls(root / 'controls', bars, cutoff, backend=backend)
    assert report == repeated and len(backend.packets) == count


def stub_pipeline(monkeypatch, calls, fail=None):
    def run(directory, bars, cutoff, *, config, backend):
        calls.append((directory, config))
        if fail == config.search_mode:
            raise RuntimeError('synthetic arm failure')
        return {}
    monkeypatch.setattr(v7_controls, 'run_month', run)
    monkeypatch.setattr(v7_controls, '_readout', lambda *args: {'status': 'synthetic_stub', 'costs': {}})


def test_fixed_protocol_equal_configs_retains_failure_without_replacement(fixture, monkeypatch):
    root, bars, cutoff = fixture
    calls = []
    stub_pipeline(monkeypatch, calls, fail='random')
    report = v7_controls.run_controls(root / 'controls', bars, cutoff, arms=('local', 'random'))
    assert len(calls) == 2 and calls[0][0] != calls[1][0]
    left, right = copy.copy(calls[0][1].__dict__), copy.copy(calls[1][1].__dict__)
    left.pop('search_mode')
    right.pop('search_mode')
    assert left == right and left['profile_candidates'] == 6 and left['research_seconds'] == 7200
    assert report['arms']['random']['failure']['error_type'] == 'RuntimeError'
    assert v7_controls.run_controls(root / 'controls', bars, cutoff, arms=('local', 'random')) == report
    assert len(calls) == 2
    with pytest.raises(ValueError, match='Immutable'):
        v7_controls.run_controls(root / 'controls', bars, cutoff, arms=('random', 'local'))


def test_explicit_backend_and_equal_budget_config_required_before_any_run(fixture):
    root, bars, cutoff = fixture
    with pytest.raises(ValueError, match='explicit'):
        v7_controls.run_controls(root / 'controls', bars, cutoff)
    with pytest.raises(ValueError, match='six P'):
        v7_controls.run_controls(root / 'controls', bars, cutoff,
            config=V7Config(evaluation_mode='joint', profile_candidates=0), arms=('local',))
    assert not (root / 'controls').exists()


@pytest.mark.parametrize('damage', ['data', 'source', 'source_archive'])
def test_frozen_data_source_and_archive_cannot_change_to_resume(fixture, monkeypatch, damage):
    root, bars, cutoff = fixture
    calls = []
    stub_pipeline(monkeypatch, calls)
    v7_controls.run_controls(root / 'controls', bars, cutoff, arms=('local',))
    if damage == 'data':
        bars.loc[0, 'close'] *= 2
    elif damage == 'source':
        (root / 'factorlab' / 'v7.py').write_text('changed')
    else:
        (root / 'controls' / 'source' / 'factorlab' / 'v7.py').write_text('changed')
    with pytest.raises(ValueError, match='Immutable|source|integrity'):
        v7_controls.run_controls(root / 'controls', bars, cutoff, arms=('local',))
    assert len(calls) == 1


def test_interruption_resumes_same_unique_arm_never_fresh_best_restart(fixture, monkeypatch):
    root, bars, cutoff = fixture
    calls = []
    stub_pipeline(monkeypatch, calls)
    working = v7_controls.run_month

    class Crash(BaseException):
        pass

    def fail(*args, **kwargs):
        working(*args, **kwargs)
        raise Crash

    monkeypatch.setattr(v7_controls, 'run_month', fail)
    with pytest.raises(Crash):
        v7_controls.run_controls(root / 'controls', bars, cutoff, arms=('local',))
    monkeypatch.setattr(v7_controls, 'run_month', working)
    v7_controls.run_controls(root / 'controls', bars, cutoff, arms=('local',))
    assert len(calls) == 2 and calls[0][0] == calls[1][0]
    assert len(list((calls[0][0]).glob('invocation-*.json'))) == 2


def test_known_reported_tokens_are_not_estimated_and_partial_usage_has_no_rate(tmp_path):
    packet = SimpleNamespace(request_hash='a' * 64, text='some actual input')
    replies = iter([BackendReply('{}', {'input_tokens': 100, 'output_tokens': 10}), BackendReply('{}', None)])
    callback = v7_controls._metered_backend(tmp_path, lambda *a, **k: next(replies))
    callback(packet)
    first = v7_controls._costs(tmp_path, 2)
    assert first['passing_per_1000_reported_input_tokens'] == 20
    callback(packet)
    partial = v7_controls._costs(tmp_path, 2)
    assert partial['known_reported_input_tokens'] == 100 and not partial['all_input_usage_reported']
    assert partial['passing_per_1000_reported_input_tokens'] is None
    assert partial['input_characters'] == 2 * len(packet.text)
    assert partial['output_characters_known_sum'] == 4


def test_failed_provider_call_is_preserved_as_cost_unknown(tmp_path):
    def fail(*args, **kwargs):
        raise TimeoutError('synthetic timeout')
    callback = v7_controls._metered_backend(tmp_path, fail)
    with pytest.raises(TimeoutError):
        callback(SimpleNamespace(request_hash='b' * 64, text='actual characters'))
    costs = v7_controls._costs(tmp_path, 0)
    assert costs['provider_calls'] == 1 and costs['calls'][0]['status'] == 'failed'
    assert costs['known_reported_input_tokens'] is None


def test_descriptive_pass_rule_uses_all_three_nominal_fold_coverages():
    folds = [{**f, 'support_rows': int((pd.Timestamp(f['score_end']) - pd.Timestamp(f['fit_cutoff'])).total_seconds() / 3600)}
             for f in development_folds('2025-04-01T00:00Z')]
    result = {'role': 'S', 'S': {'Pv': {'gain': .006, 'winning_fold_fraction': 2/3}}, 'folds': folds}
    assert v7_controls._passes(result, ['BTCUSDT'])
    result['folds'][0]['support_rows'] //= 2
    assert not v7_controls._passes(result, ['BTCUSDT'])
    result['S']['Pv']['gain'] = None
    assert not v7_controls._passes(result, ['BTCUSDT'])


def test_fixed_component_probe_equal_support_and_solve_budget_no_permission(fixture):
    _, bars, cutoff = fixture
    result = v7_controls.known_component_recovery_probe(build_rolling_data(bars), cutoff)
    assert result['solves'] == {'removed': 3, 'restored': 3}
    assert len(result['folds']) == 3
    assert all(f['solves_per_arm'] == 1 and set(f['losses']) == {'removed', 'restored'} for f in result['folds'])
    assert not result['known_population_effect'] and not result['opens_NN_or_confirmation_gate']


def test_reply_template_and_evaluation_duplicates_are_counted_without_guessing_deferred():
    state = {'attempts': [{'status': 'scored'}, {'status': 'duplicate'}],
             'agent_audit': [{'status': 'validated', 'concrete_programs': 6, 'unique_programs': 1},
                             {'status': 'failed'}],
             'materializations': [{'status': 'duplicate'}, {'status': 'invalid'}],
             'budget': {'counts': {'ast': 12}}, 'pool': {'records': {'one': {}}}}
    result = v7_controls._duplicates(state)
    assert result['duplicates'] == 7 and result['unscored_AST_count'] == 11
    assert result['duplicate_breakdown']['validated_reply_seed_duplicates'] == 5


def test_preexisting_arm_cannot_supply_different_initial_pool(fixture, monkeypatch):
    root, bars, cutoff = fixture
    calls = []
    stub_pipeline(monkeypatch, calls)
    path = root / 'controls' / 'arms' / 'local'
    path.mkdir(parents=True)
    (path / 'state.sqlite').write_bytes(b'preexisting arm')
    with pytest.raises(ValueError, match='preexisting'):
        v7_controls.run_controls(root / 'controls', bars, cutoff, arms=('local',))
    assert not calls


def test_cross_arm_initial_reference_guard(fixture, monkeypatch):
    root, bars, cutoff = fixture
    calls = []
    stub_pipeline(monkeypatch, calls)
    monkeypatch.setattr(v7_controls, '_readout', lambda directory, config, month: {
        'status': 'synthetic_stub', 'pool_reference_hash': config.search_mode})
    with pytest.raises(ValueError, match='initial reference pool'):
        v7_controls.run_controls(root / 'controls', bars, cutoff, arms=('local', 'random'))
    assert not (root / 'controls' / 'report.json').exists()


def test_provider_finished_receipt_must_match_reserved_request(tmp_path):
    start = {'sequence': 0, 'request_hash': 'first', 'input_characters': 10}
    v7_controls._write(tmp_path / 'provider_calls' / '0-started.json', start)
    v7_controls._write(tmp_path / 'provider_calls' / '0-finished.json',
                       {**start, 'request_hash': 'different', 'usage': None})
    with pytest.raises(ValueError, match='reserved request'):
        v7_controls._costs(tmp_path, 0)


def test_recovered_measured_result_is_not_a_failed_candidate():
    result = v7_controls._attempt_counts([{'status': s} for s in (
        'scored', 'duplicate', 'measured_recovered_budget_retained', 'invalid', 'interrupted_consumed')])
    assert result['recovered_measured_results'] == 1 and result['failures'] == 2
