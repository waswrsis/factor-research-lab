"""Synthetic engineering receipts only; no market or Linux acceptance claim."""
import copy
import json
from dataclasses import replace

import pandas as pd
import pytest
from test_v7_nonlinear_channel import artificial_data

import factorlab.discovery.nonlinear_runtime as runtime
from factorlab.discovery.joint_pool import FrozenPool
from factorlab.discovery.local_mutation import MonthlyBudget
from factorlab.discovery.nonlinear_channel import ResidualConfig, StructureSpec
from factorlab.discovery.rolling import RollingSpec, development_folds
from factorlab.protocol import fingerprint


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    from factorlab import protocol
    monkeypatch.setattr(protocol, 'ROOT', tmp_path)
    (tmp_path / 'factorlab').mkdir()
    (tmp_path / 'factorlab' / 'frozen.py').write_text('frozen=1\n')
    (tmp_path / 'pyproject.toml').write_text('[project]\n')
    (tmp_path / 'uv.lock').write_text('version=1\n')
    directory = tmp_path / 'neural'
    data, pool = artificial_data(), FrozenPool()
    config = ResidualConfig(enabled=True, epochs=0, seeds=(17, 29))
    clock = [pd.Timestamp('2024-09-01T00:00:00Z')]
    monkeypatch.setattr(runtime, '_now', lambda: clock[0])
    protocol_id = runtime.freeze_protocol(directory, data, '2024-10-01T00:00:00Z', pool,
                                          source_id='artificial', config=config)
    store = runtime._Store(directory)
    probes = []
    for cutoff in ('2025-01-01T00:00:00Z', '2025-04-01T00:00:00Z'):
        clock[0] = pd.Timestamp(cutoff) + pd.Timedelta(days=1)
        context = runtime._context(data, cutoff, pool, 'artificial')
        result = {'status': 'completed', 'role': 'S', 'protocol_id': runtime.ProbeConfig().digest,
                  'folds': [{**fold, 'pool_id': pool.digest, 'status': 'scored',
                             'model_id': 'synthetic-tree', 'reference_model_id': 'synthetic-ridge',
                             'support': 2, 'sample_ids': ['synthetic1', 'synthetic2'],
                             'feature_schema': 'synthetic-same-input', 'loss': .9,
                             'reference_loss': 1.} for fold in development_folds(cutoff)]}
        probes.append(runtime._register(store, 'probe', context, result, test_only=True))
    context = runtime._context(data, '2025-04-01T00:00:00Z', pool, 'artificial')
    control = {'scope': 'fixed_S_known_component_recovery_diagnostic', 'component': 'rv_24',
               'folds': [{**f, 'score_rows': 2, 'support_hash': 'synthetic-only',
                          'losses': {'removed': 1., 'restored': .9}}
                         for f in development_folds(context['cutoff'])]}
    control_id = runtime._register(store, 'positive_control', context, control, test_only=True)
    store.close()
    artifact = {'platform': 'linux', 'scope': 'whole_cgroup', 'oom': 0, 'oom_kill': 0,
                'memory_max': 768 * 2**20, 'memory_high': 640 * 2**20, 'swap_max': 0,
                'cpu_quota_fraction': .25, 'memory_peak': 599 * 2**20,
                'train_row_capacity': 1000, 'score_row_capacity': 1000,
                'frame_row_capacity': 1000, 'feature_capacity': 12,
                'frame_byte_capacity': 1000 * 12 * 8,
                'trial_started_at': '2025-04-01T00:00:00Z',
                'trial_finished_at': '2025-04-02T00:00:00Z',
                'trial_context': {'context': runtime._common(context), 'config_id': config.digest},
                'status': 'completed', 'raw_receipt_hash': 'synthetic-not-Linux',
                'workload_result_hash': 'synthetic-not-market'}
    monkeypatch.setattr(runtime, '_external_resource_verifier',
                        lambda value, expected: {'verified': True, 'test_only': True})
    resource_id = runtime.record_resource_receipt(directory, data, context['cutoff'], pool,
                                                  artifact, source_id='artificial', config=config)
    clock[0] = pd.Timestamp('2025-05-01T00:00:00Z')
    gates = {'protocol': protocol_id, 'probes': probes,
             'positive_control': control_id, 'resource': resource_id}
    return directory, data, pool, config, gates, clock


def verify(prepared, **kwargs):
    directory, data, pool, config, gates, _ = prepared
    return runtime.verify_gate_artifacts(directory, data, '2025-05-01T00:00:00Z', pool,
                                          gates, source_id='artificial', config=config, **kwargs)


def mutate_receipt(prepared, kind, change):
    directory, _, _, _, gates, _ = prepared
    key = gates[kind][0] if kind == 'probes' else gates[kind]
    store = runtime._Store(directory)
    value = json.loads((directory / 'receipts' / f'{key}.json').read_text())
    change(value)
    # Even a freshly registered fixture cannot avoid semantic gate validation.
    new_id = runtime._register(store, value['kind'], value['context'], value['result'], test_only=True)
    store.close()
    if kind == 'probes':
        gates[kind][0] = new_id
    else:
        gates[kind] = new_id


def test_synthetic_gate_explicitly_stays_test_only(prepared):
    result = verify(prepared)
    assert result['test_only'] is True and result['formal_claim'] is False


@pytest.mark.parametrize('kind', ['protocol', 'positive_control', 'resource'])
def test_missing_or_wrong_producer_cannot_open(prepared, kind):
    gates = prepared[4]
    gates[kind] = gates['probes'][0]
    with pytest.raises(ValueError, match='producer'):
        verify(prepared)


def test_two_copies_of_one_quarter_are_not_repeatability(prepared):
    prepared[4]['probes'][1] = prepared[4]['probes'][0]
    with pytest.raises(ValueError, match='distinct quarters'):
        verify(prepared)


@pytest.mark.parametrize('change', [
    lambda v: v['context'].update(pool_id='wrong'),
    lambda v: v['context'].update(protocol_id='wrong'),
    lambda v: v['context'].update(source_id='other-source'),
    lambda v: v['context'].update(input_schema='different-input'),
    lambda v: v['result']['folds'][0].update(pool_id='wrong-fold-pool'),
    lambda v: v['result']['folds'][0].update(loss=1.1),
])
def test_identity_or_paired_probe_failure_rejected(prepared, change):
    mutate_receipt(prepared, 'probes', change)
    with pytest.raises(ValueError):
        verify(prepared)


def test_negative_known_component_control_cannot_open(prepared):
    mutate_receipt(prepared, 'positive_control',
                   lambda v: [f['losses'].update(restored=2.) for f in v['result']['folds']])
    with pytest.raises(ValueError, match='not positive'):
        verify(prepared)


def test_receipt_bytes_and_actual_source_manifest_rechecked(prepared):
    directory = prepared[0]
    source = directory.parent / 'factorlab' / 'frozen.py'
    source.write_text('frozen=2\n')
    with pytest.raises(ValueError, match='identity'):
        verify(prepared)
    source.write_text('frozen=1\n')
    path = directory / 'receipts' / f'{prepared[4]["resource"]}.json'
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='bytes changed'):
        verify(prepared)


def test_late_receipt_future_cutoff_and_event_task_rejected(prepared):
    prepared[5][0] = pd.Timestamp('2025-05-02T00:00:00Z')
    mutate_receipt(prepared, 'probes', lambda v: None)
    with pytest.raises(ValueError, match='availability'):
        verify(prepared)
    prepared[5][0] = pd.Timestamp('2025-04-30T00:00:00Z')
    with pytest.raises(ValueError, match='future'):
        verify(prepared)
    directory, data, pool, config, gates, _ = prepared
    events = replace(data, spec=RollingSpec(task='first_event', horizon_hours=20, minimum_train_rows=12))
    with pytest.raises(ValueError, match='isolates'):
        runtime.run_gated_residual(directory, events, '2025-05-01T00:00:00Z', pool, gates,
                                   source_id='artificial', config=config)


@pytest.mark.parametrize('field,value', [('memory_peak', 600 * 2**20), ('oom', 1),
                                        ('cpu_quota_fraction', .5), ('swap_max', 1),
                                        ('memory_high', 768 * 2**20)])
def test_linux_contract_is_checked_even_after_trusted_verifier(prepared, field, value):
    mutate_receipt(prepared, 'resource', lambda v: v['result'].update({field: value}))
    with pytest.raises(ValueError):
        verify(prepared)


def test_external_attestation_cannot_be_self_reported():
    with pytest.raises(ValueError, match='not installed'):
        runtime._external_resource_verifier({'verified': True}, {})


def test_durable_failed_work_shared_count_and_retry_are_not_refunded(prepared, monkeypatch):
    directory, data, pool, config, gates, _ = prepared
    budget = MonthlyBudget('2025-05')
    calls = []

    def fail(*args):
        ledger = args[-1]
        ledger.consume('neural_fit')
        ledger.consume('solve', 2)
        calls.append(1)
        raise RuntimeError('synthetic failure after reservation')

    monkeypatch.setattr(runtime, 'research_residual', fail)
    kwargs = {'source_id': 'artificial', 'config': config, 'budget': budget}
    with pytest.raises(RuntimeError, match='synthetic failure'):
        runtime.run_gated_residual(directory, data, '2025-05-01T00:00:00Z', pool, gates, **kwargs)
    assert budget.counts['solve'] == 2
    kwargs['budget'] = MonthlyBudget('2025-05')  # stale restart caller
    with pytest.raises(ValueError, match='already consumed'):
        runtime.run_gated_residual(directory, data, '2025-05-01T00:00:00Z', pool, gates, **kwargs)
    assert kwargs['budget'].counts['solve'] == 2 and len(calls) == 1
    store = runtime._Store(directory)
    state = store.get('budget', '2025-05')
    store.close()
    assert state['channel']['counts']['neural_fit'] == 1
    assert state['shared']['counts']['solve'] == 2


def test_completed_restart_reuses_exact_result_without_fit(prepared, monkeypatch):
    directory, data, pool, config, gates, _ = prepared
    calls = []

    def result(*args):
        args[-1].consume('neural_fit')
        args[-1].consume('solve')
        calls.append(1)
        return {'status': 'completed', 'records': [], 'formal_claim': False}

    monkeypatch.setattr(runtime, 'research_residual', result)
    first = runtime.run_gated_residual(directory, data, '2025-05-01T00:00:00Z', pool, gates,
                                       source_id='artificial', config=config)
    second = runtime.run_gated_residual(directory, data, '2025-05-01T00:00:00Z', pool, gates,
                                        source_id='artificial', config=config)
    assert first == second and len(calls) == 1
    assert first['test_only'] and not first['runtime_admitted']


def test_hard_interruption_consumes_time_and_never_retries(prepared, monkeypatch):
    directory, data, pool, config, gates, clock = prepared

    def fail(*args):
        args[-1].consume('solve')
        raise RuntimeError('stop')

    monkeypatch.setattr(runtime, 'research_residual', fail)
    with pytest.raises(RuntimeError):
        runtime.run_gated_residual(directory, data, clock[0], pool, gates,
                                   source_id='artificial', config=config)
    store = runtime._Store(directory)
    state = store.get('budget', '2025-05')
    attempt = next(iter(state['attempts'].values()))
    attempt['status'] = 'active'  # emulate process death: finally never wrote
    store.put('budget', '2025-05', state)
    store.close()
    clock[0] += pd.Timedelta(seconds=10)
    with pytest.raises(ValueError, match='already consumed'):
        runtime.run_gated_residual(directory, data, '2025-05-01T00:00:00Z', pool, gates,
                                   source_id='artificial', config=config)
    store = runtime._Store(directory)
    restored = store.get('budget', '2025-05')
    store.close()
    assert restored['channel']['spent_seconds'] >= 10
    assert next(iter(restored['attempts'].values()))['status'] == 'interrupted_consumed'


def test_multiple_experts_rejected_before_training(prepared, monkeypatch):
    directory, data, pool, config, gates, _ = prepared
    monkeypatch.setattr(runtime, 'research_residual', lambda *args: pytest.fail('Must not train'))
    config = replace(config, structures=(StructureSpec(experts=2),))
    with pytest.raises(ValueError, match='Multiple experts'):
        runtime.run_gated_residual(directory, data, '2025-05-01T00:00:00Z', pool, gates,
                                   source_id='artificial', config=config)


def test_actual_producer_executes_and_registers_once(prepared, monkeypatch):
    directory, data, pool, _, _, clock = prepared
    calls = []

    def probe(*args):
        args[-1].consume('tree_fit')
        args[-1].consume('solve')
        calls.append(1)
        return {'status': 'completed', 'synthetic_fixture': True}

    monkeypatch.setattr(runtime, 'run_probe', probe)
    kwargs = {'source_id': 'artificial'}
    first = runtime.record_probe(directory, data, clock[0], pool, **kwargs)
    assert runtime.record_probe(directory, data, clock[0], pool, **kwargs) == first
    assert len(calls) == 1
    row = json.loads((directory / 'receipts' / f'{first}.json').read_text())
    assert row['result_hash'] == fingerprint(row['result'])


def test_config_change_unregistered_json_and_budget_month_rejected(prepared, monkeypatch):
    directory, data, pool, config, gates, _ = prepared
    changed = copy.deepcopy(gates)
    changed['resource'] = '0' * 64
    with pytest.raises(ValueError, match='Unregistered'):
        runtime.run_gated_residual(directory, data, '2025-05-01T00:00:00Z', pool, changed,
                                   source_id='artificial', config=config)
    with pytest.raises(ValueError, match='configuration'):
        runtime.verify_gate_artifacts(directory, data, '2025-05-01T00:00:00Z', pool, gates,
                                       source_id='artificial', config=replace(config, epochs=1))
    with pytest.raises(ValueError, match='matching monthly'):
        runtime.run_gated_residual(directory, data, '2025-05-01T00:00:00Z', pool, gates,
                                   source_id='artificial', config=config, budget=MonthlyBudget('2025-04'))


def test_real_tiny_epoch_zero_remains_reference_and_test_only(prepared):
    directory, data, pool, config, gates, _ = prepared
    result = runtime.run_gated_residual(directory, data, '2025-05-01T00:00:00Z', pool, gates,
                                        source_id='artificial', config=config)
    assert result['status'] == 'completed'
    assert len(result['records']) == 6
    assert all(r['selected_epoch'] == 0 and r['loss'] == r['epoch0_loss'] for r in result['records'])
    assert result['test_only'] and not result['runtime_admitted'] and not result['formal_claim']


def test_restart_reconciles_all_monthly_spending_without_refund(prepared, monkeypatch):
    directory, data, pool, config, gates, _ = prepared
    shared = MonthlyBudget('2025-05')
    shared.register_family('existing')
    shared.consume('ast', 2, family_id='existing')
    shared.consume('gpt_attempt')

    def result(*args):
        args[-1].consume('solve', 2)
        return {'status': 'completed'}

    monkeypatch.setattr(runtime, 'research_residual', result)
    runtime.run_gated_residual(directory, data, '2025-05-01T00:00:00Z', pool, gates,
                               source_id='artificial', config=config, budget=shared)
    restored = runtime.restore_shared_budget(directory, '2025-05-01T00:00:00Z', MonthlyBudget('2025-05'))
    assert restored.to_dict() == shared.to_dict()
    divergent = MonthlyBudget('2025-05')
    divergent.consume('gpt_attempt', 2)
    with pytest.raises(ValueError, match='Divergent'):
        runtime.restore_shared_budget(directory, '2025-05-01T00:00:00Z', divergent)


def test_failed_final_persistence_preserves_original_exception(prepared, monkeypatch):
    directory, data, pool, config, gates, _ = prepared
    original_put = runtime._Store.put

    def put(store, kind, key, value):
        if kind == 'budget' and any(a['status'] == 'failed_consumed' for a in value['attempts'].values()):
            raise OSError('synthetic disk failure')
        return original_put(store, kind, key, value)

    def fail(*args):
        args[-1].consume('solve')
        raise RuntimeError('original training failure')

    monkeypatch.setattr(runtime._Store, 'put', put)
    monkeypatch.setattr(runtime, 'research_residual', fail)
    with pytest.raises(RuntimeError, match='original training failure') as error:
        runtime.run_gated_residual(directory, data, '2025-05-01T00:00:00Z', pool, gates,
                                   source_id='artificial', config=config)
    assert any('final checkpoint failed' in note for note in error.value.__notes__)
    store = runtime._Store(directory)
    state = store.get('budget', '2025-05')
    store.close()
    assert next(iter(state['attempts'].values()))['status'] == 'active'
    assert state['channel']['counts']['solve'] == 1


def test_future_producer_rejected_before_receipt_or_budget(prepared, monkeypatch):
    directory, data, pool, _, _, _ = prepared
    monkeypatch.setattr(runtime, 'run_probe', lambda *args: pytest.fail('Future producer ran'))
    with pytest.raises(ValueError, match='future development'):
        runtime.record_probe(directory, data, '2025-07-01T00:00:00Z', pool, source_id='artificial')
    store = runtime._Store(directory)
    assert store.get('budget', '2025-07') is None
    store.close()


def test_smaller_resource_trial_does_not_certify_larger_dataset(prepared):
    mutate_receipt(prepared, 'resource', lambda v: v['result'].update(train_row_capacity=1))
    with pytest.raises(ValueError, match='row capacity'):
        verify(prepared)


def test_missing_budget_after_failed_work_cannot_reset_via_restore(prepared, monkeypatch):
    directory, data, pool, config, gates, _ = prepared

    def fail(*args):
        args[-1].consume('solve')
        raise RuntimeError('Consumed without a result receipt')

    monkeypatch.setattr(runtime, 'research_residual', fail)
    with pytest.raises(RuntimeError):
        runtime.run_gated_residual(directory, data, '2025-05-01T00:00:00Z', pool, gates,
                                   source_id='artificial', config=config)
    store = runtime._Store(directory)
    store.db.execute("DELETE FROM records WHERE kind='budget'")
    store.db.commit()
    store.close()
    with pytest.raises(ValueError, match='Missing nonlinear budget checkpoint'):
        runtime.restore_shared_budget(directory, '2025-05-01T00:00:00Z', MonthlyBudget('2025-05'))
    with pytest.raises(ValueError, match='Missing nonlinear budget checkpoint'):
        runtime.run_gated_residual(directory, data, '2025-05-01T00:00:00Z', pool, gates,
                                   source_id='artificial', config=config)
