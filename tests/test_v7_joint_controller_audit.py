"""Real-ridge controller integration plus isolated failure-injection controls."""
import copy
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from factorlab.discovery import joint_pool, v7_run
from factorlab.discovery.factor_pool import FactorPool
from factorlab.discovery.joint_pool import DevelopmentPoolArchive, FrozenPool, PoolMember, fit_pool
from factorlab.discovery.model_registry import ModelRegistry
from factorlab.discovery.rolling import RollingSpec, build_rolling_data, development_folds
from factorlab.discovery.v7_agent import BackendReply
from factorlab.discovery.v7_run import RunStore, V7Config, run_month
from factorlab.protocol import fingerprint


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    (tmp_path / 'factorlab').mkdir()
    (tmp_path / 'factorlab' / 'frozen.py').write_text('version=7\n', encoding='utf-8')
    (tmp_path / 'pyproject.toml').write_text('[project]\n', encoding='utf-8')
    (tmp_path / 'uv.lock').write_text('version=1\n', encoding='utf-8')
    return tmp_path


@pytest.fixture(scope='module')
def bars():
    times = pd.date_range('2024-08-01', '2025-06-03', freq='h', tz='UTC', inclusive='left')
    rng = np.random.default_rng(191)
    returns = rng.normal(size=len(times)) * (.001 + .003 * (1 + np.sin(np.arange(len(times)) / 99)))
    price = np.exp(4 + np.cumsum(returns))
    return pd.DataFrame({'symbol': 'BTCUSDT', 'open_time': times,
        'available_at': times + pd.Timedelta(hours=1, seconds=1),
        'open': price, 'close': price, 'high': price * 1.003, 'low': price / 1.003})


def _state(directory, month):
    with sqlite3.connect(directory / 'state.sqlite') as db:
        return json.loads(db.execute('SELECT value FROM months WHERE month=?', (month,)).fetchone()[0])


def _force_one_admission(results, pool, **kwargs):
    """Exercise carry mechanics, not evidence thresholds; ridge evaluation stays real."""
    ids = [row['candidate_id'] for row in results]
    return {'selected_ids': ids[:1], 'role': 'S', 'formal_support_claimed': False,
            'receipt': {'pool_hash': pool.digest}}


def test_two_real_ridge_months_carry_frozen_pool_context_archive_and_restart(workspace, bars, monkeypatch):
    monkeypatch.setattr(joint_pool, 'stability_select', _force_one_admission)
    directory = workspace / 'joint'
    config = V7Config(evaluation_mode='joint', formulas_per_family=1, lookback=4)
    first = run_month(directory, bars, '2025-04-01T00:00Z', config=config)
    a = _state(directory, '2025-04')
    first_pool = FrozenPool.from_dict(a['development_pool'])
    assert len(first_pool.members) == 1
    assert first['budget']['counts']['solve'] > 10
    second = run_month(directory, bars, '2025-05-01T00:00Z', config=config)
    b = _state(directory, '2025-05')
    assert FrozenPool.from_dict(b['frozen_reference_pool']).digest == first_pool.digest
    assert len(FrozenPool.from_dict(b['development_pool']).members) <= 2
    assert a['data_id'] != b['data_id']
    archives = [DevelopmentPoolArchive.load(directory / 'archives' / f'{month}.json')
                for month in ('2025-04', '2025-05')]
    manifest = json.loads((directory / 'manifest.json').read_text('utf-8'))
    assert archives[0].digest != archives[1].digest
    for state, archive in zip((a, b), archives, strict=True):
        assert archive.to_dict() == state['joint_archive']
        assert len(archive.admissions) <= 1
        for entry in archive.records.values():
            context, result = entry['context'], entry['result']
            # Carried evidence stays bound to the month in which it was scored.
            origin = _state(directory, pd.Timestamp(context['cutoff']).strftime('%Y-%m'))
            reference = FrozenPool.from_dict(origin['frozen_reference_pool'])
            assert context['pool_hash'] == reference.digest
            assert context['cutoff'] == origin['cutoff']
            assert pd.Timestamp(context['cutoff']) <= pd.Timestamp(state['cutoff'])
            assert context['data_hash'] == origin['data_id']
            assert context['protocol_hash'] == RollingSpec().digest
            assert context['source_hash'] == fingerprint(manifest['identity']['source_files'])
            assert result['context'] == context and result['role'] == 'S'
            assert result['channels']['F']['gain'] == result['F']['Pv']['gain']
            assert result['channels']['S']['gain'] == result['S']['Pv']['gain']
            assert origin['pool']['records'][result['formula_id']]['result'] == result
            # Old results never become current observations by changing only context.
            changed_context = {**context, 'source_hash': 'different-source'}
            assert archive.result_for(result['candidate_id'], changed_context) is None
        current = DevelopmentPoolArchive.context(FrozenPool.from_dict(state['frozen_reference_pool']),
            cutoff=state['cutoff'], data_hash=state['data_id'], protocol_hash=RollingSpec().digest,
            source_hash=fingerprint(manifest['identity']['source_files']))
        for parent in archive.active_parents(current):
            if parent['prior_evidence_only']:
                assert parent['current_context_result'] is None
                assert parent['origin_context'] != current
    assert second['A0'] == 'K0' and second['formal_confirmation_enabled'] is False
    assert run_month(directory, bars, '2025-05-01T00:00Z', config=config) == second
    assert _state(directory, '2025-04') == a


def test_joint_model_registry_roundtrip_and_wrong_pipeline_rejected(workspace, bars):
    data = build_rolling_data(bars, RollingSpec())
    member = PoolMember(v7_run.literature_seeds()[0], lookback=4)
    pool = FrozenPool((member,))
    model = fit_pool(data, '2025-04-01T00:00Z', pool)
    registry = ModelRegistry(workspace / 'models')
    pipeline = {'pipeline_id': pool.pipeline_id, 'pool_hash': pool.digest}
    identifier = registry.save(model, pipeline)
    restored, identity = registry.load(identifier, expected_pipeline=pipeline)
    assert identity['kind'] == 'v7_joint_pool'
    assert restored.digest == model.digest
    expected, mask = model.predict_history(data.frame)
    actual, restored_mask = restored.predict_history(data.frame)
    np.testing.assert_equal(actual, expected)
    np.testing.assert_equal(mask, restored_mask)
    with pytest.raises(ValueError, match='Pipeline identity'):
        registry.load(identifier, expected_pipeline={'pipeline_id': FrozenPool().pipeline_id})


@pytest.mark.parametrize('read_cutoff', ['2025-04-01T00:00Z', '2025-05-01T00:00Z'])
def test_completed_month_rejects_mutated_published_joint_archive(workspace, bars, read_cutoff):
    directory = workspace / 'archive-integrity'
    config = V7Config(evaluation_mode='joint', formulas_per_family=1, lookback=4)
    run_month(directory, bars, '2025-04-01T00:00Z', config=config)
    path = directory / 'archives' / '2025-04.json'
    record = json.loads(path.read_text('utf-8'))
    record['archive']['epoch'] = '1999-01-01T00:00:00+00:00'
    path.write_text(json.dumps(record), encoding='utf-8')
    with pytest.raises(ValueError, match='[Aa]rchive|integrity'):
        run_month(directory, bars, read_cutoff, config=config)


def test_running_predecessor_cannot_override_last_completed_pool(workspace, bars, monkeypatch):
    monkeypatch.setattr(joint_pool, 'stability_select', _force_one_admission)
    directory = workspace / 'skip-running'
    config = V7Config(evaluation_mode='joint', formulas_per_family=1, lookback=4)
    run_month(directory, bars, '2025-04-01T00:00Z', config=config)
    original = _state(directory, '2025-04')
    store = RunStore(directory, config, RollingSpec())
    try:
        # Honest incomplete checkpoint, written through the journal API. Its pool
        # is deliberately different to expose an unfiltered predecessor query.
        unfinished = copy.deepcopy(original)
        unfinished.update(status='running', cutoff='2025-05-01T00:00:00+00:00',
                          development_pool=FrozenPool(version='unfinished-May').to_dict())
        store.save('2025-05', unfinished, 'artificial_interrupted_month')
    finally:
        store.close()
    run_month(directory, bars, '2025-06-01T00:00Z', config=config)
    next_state = _state(directory, '2025-06')
    assert next_state['frozen_reference_pool'] == original['development_pool']


class ParameterBackend:
    def __init__(self):
        self.calls = 0

    def __call__(self, packet, *, model, reasoning):
        from factorlab.discovery.operator_program import Program
        from factorlab.discovery.program_syntax import CompactProgram, parse_expression
        self.calls += 1
        body = json.loads(packet.body)
        program = CompactProgram.from_dict(body['parents'][0]['program']).to_program().to_dict()
        program['outputs'] = [parse_expression('last(ema(f("rv_6"),p("alpha")))')]
        program['parameters'] = {'alpha': {'initial': .1 * self.calls, 'lower': .01, 'upper': .99}}
        compact = CompactProgram.from_program(Program.from_dict(program)).to_dict()
        value = {'version': 'v7-proposal-batch-v1', 'identity': body['identity'],
            'family_id': f'parameter-{self.calls}', 'hypothesis': 'Synthetic parameter quota check.',
            'mechanism': 'Train-only bounded averaging.', 'competing_explanation': 'Already in K0.',
            'falsification': 'No conditional improvement.',
            'parent_ids': [body['parents'][0]['parent_id']],
            'template': {'base': compact, 'slots': []}, 'programs': [compact]}
        return BackendReply(json.dumps(value), {})


def test_failed_parameter_profiles_still_consume_structure_quota(workspace, bars, monkeypatch):
    calls = []

    def failed_fit(data, cutoff, pool, candidate, *, budget, context):
        calls.append(candidate.member_id)
        budget.consume('solve')
        raise ValueError('Injected numerical failure after profile start')

    monkeypatch.setattr(joint_pool, 'evaluate_candidate', failed_fit)
    directory = workspace / 'parameter-budget'
    config = V7Config(evaluation_mode='joint', search_mode='gpt', profile_candidates=1,
                      formulas_per_family=1, lookback=4)
    result = run_month(directory, bars, '2025-04-01T00:00Z', config=config, backend=ParameterBackend())
    assert len(calls) == 1
    # Three folds x two reference readouts, one current K0, one failed profile.
    assert result['budget']['counts']['solve'] == 8
    state = _state(directory, '2025-04')
    assert any(row['status'] == 'invalid' for row in state['attempts'])
    assert fingerprint(state['joint_archive'])


def test_joint_prompt_names_the_actual_conditional_reference(workspace, bars, monkeypatch):
    """Stub scores isolate feedback semantics; this does not assert efficacy."""
    directory = workspace / 'conditional-feedback'
    config = V7Config(evaluation_mode='joint', search_mode='gpt', profile_candidates=3,
                      formulas_per_family=1, lookback=4)
    prior_pool = FrozenPool((PoolMember(v7_run.literature_seeds()[0], lookback=4),))
    archive = DevelopmentPoolArchive('2025-04-01T00:00Z')
    store = RunStore(directory, config, RollingSpec())
    try:
        archive.save(directory / 'archives' / '2025-04.json')
        store.save('2025-04', {'status': 'complete', 'pool': FactorPool().to_dict(),
            'development_pool': prior_pool.to_dict(), 'joint_archive': archive.to_dict()},
            'artificial_prior_completed')
    finally:
        store.close()

    def diagnostic(data, cutoff, pool, candidate, *, budget, context):
        budget.consume('solve')
        flat = {'candidate_loss': 8., 'reference_loss': 10., 'gain': .2,
                'absolute_delta': 2., 'winning_fold_fraction': 0.}
        return {'role': 'S', 'pool_hash': pool.digest, 'candidate_id': candidate.member_id,
            'context': context, 'receipt': {'cutoff': pd.Timestamp(cutoff).isoformat()},
            'folds': list(development_folds(cutoff)), 'F': {'Pv': flat, 'K0': flat},
            'P': flat, 'S': {'Pv': flat, 'K0': {**flat, 'gain': .5}}}

    monkeypatch.setattr(joint_pool, 'evaluate_candidate', diagnostic)

    class RecordingBackend(ParameterBackend):
        def __init__(self):
            super().__init__()
            self.bodies = []

        def __call__(self, packet, **kwargs):
            self.bodies.append(json.loads(packet.body))
            return super().__call__(packet, **kwargs)

    backend = RecordingBackend()
    run_month(directory, bars, '2025-05-01T00:00Z', config=config, backend=backend)
    baseline = _state(directory, '2025-05')['baseline']['residual_diagnostics']
    assert 1 <= len(baseline) <= 8
    for body in backend.bodies:
        assert len(body['diagnostics']) == len(baseline)
        for diagnostic, original in zip(body['diagnostics'], baseline, strict=True):
            assert diagnostic['role'] == 'S' and diagnostic['summary'].startswith('K0 squared log residual:')
            assert diagnostic['metrics'] == {'loss': original['loss'], 'support': original['support']}
            assert diagnostic['pool_id'] == prior_pool.digest
    measured = [p for body in backend.bodies for p in body['parents']
                if p['evidence'].get('prior_readouts')]
    assert measured
    for parent in measured:
        evidence = parent['evidence']
        assert evidence['pool_id'] == prior_pool.digest
        assert evidence['metrics'] == {}
        for prior in evidence['prior_readouts']:
            assert prior['gain_S'] == .2 and prior['gain_S'] != .5
            assert prior['pool_id'] == prior_pool.digest
            origin = _state(directory, pd.Timestamp(prior['cutoff']).strftime('%Y-%m'))
            assert prior['data_id'] == origin['data_id']
            assert any(fingerprint(r['result']) == prior['result_hash'] for r in origin['pool']['records'].values())
        assert 'relative to K0' not in evidence['summary']
        assert 'Pv' in evidence['summary'] or 'conditional' in evidence['summary']
