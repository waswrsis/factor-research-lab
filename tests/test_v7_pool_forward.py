"""Core pool forward integration on actual artificial receipt/accounting stores.

Only kernel/boot observations and the service host preflight are synthetic.
No class substitution, publication/accounting mock, provider, network, or market.
"""
import json
import os
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from factorlab.discovery import pool_forward as module
from factorlab.discovery import work_lease
from factorlab.discovery.convex_eval import fit_ridge
from factorlab.discovery.forward_publication_witness import ForwardPublicationWitness
from factorlab.discovery.joint_pool import FrozenPool, fit_pool
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
from factorlab.discovery.nonlinear_channel import ChannelBudget
from factorlab.discovery.v7_forward import ForwardRuntime
from factorlab.discovery.v7_service import ForwardService, ServiceConfig


@pytest.fixture
def case(tmp_path, monkeypatch):
    from test_v7_feed_audit import Clock
    from test_v7_rolling_registry import candles
    from test_v7_run_factor_pool import source_root

    source_root(tmp_path, monkeypatch)
    clock = Clock('2026-09-30T23:30:00Z')
    runtime = ForwardRuntime(tmp_path / 'run', tmp_path / 'feed', clock=clock)
    boot = 'd43d6a03-a96b-47b0-8e7d-e8d1bb65eaba'
    monkeypatch.setattr(work_lease, '_boot_id', lambda: 'linux:' + boot)

    def kernel():
        raw = {'memory.max': str(768 * 2**20), 'memory.high': str(640 * 2**20),
               'memory.swap.max': '0', 'pids.max': '64', 'cpu.max': '25000 100000',
               'memory.peak': str(100 * 2**20), 'memory.events': 'oom 0\noom_kill 0',
               'cgroup.procs': str(os.getpid())}
        return {'path': '/sys/fs/cgroup/artificial.service', 'inode': 321, 'boot_id': boot,
                'scope': 'whole_cgroup', 'platform': 'linux', 'raw': raw,
                'peak': 100 * 2**20, 'events': {'oom': 0, 'oom_kill': 0},
                'observed_at': datetime.now(UTC).isoformat()}

    monkeypatch.setattr(module, '_kernel_snapshot', kernel)
    try:
        path = tmp_path / 'artificial.parquet'
        candles('2026-05-01T00:00Z', '2026-09-30T23:00Z').to_parquet(path)
        runtime.feed.import_seed(path)
        clock.now = pd.Timestamp('2026-10-01T00:00:10Z')
        authority = MonthlyWorkAuthority(tmp_path / 'authority', run_id='pool-core-integration',
                                        task='volatility', universe=list(runtime.config.symbols))
        yield SimpleNamespace(runtime=runtime, authority=authority, clock=clock,
                              pool=FrozenPool(role='reference', version='K0_v1'))
    finally:
        runtime.close()


def run(case):
    return case.runtime.fit_month('2026-10-01T00:00:00Z', pool=case.pool, authority=case.authority)


def ticket_for(case):
    paths = list((case.authority.directory / 'pool_forward_tickets').glob('*/ticket.json'))
    assert len(paths) == 1
    return module.PoolForwardTicket.load(paths[0])


def wallet(case):
    with case.authority._transaction() as db:
        return case.authority._verify(db)['wallets']


def test_actual_runtime_dispatch_registry_v4_witness_and_historical_cache(case, monkeypatch):
    result = run(case)
    ticket = ticket_for(case)
    model, identity = case.runtime.lifecycle.registry.load(result['model_id'])
    assert identity['version'] == 4 and identity['job'] == result['job']
    assert identity['pipeline'] == case.runtime.recipe_for(case.pool)
    assert model.pipeline_id == identity['pipeline']['pipeline_id']
    assert ForwardPublicationWitness.read_completed(ticket)['model_digest'] == model.digest
    before = wallet(case)
    assert sum(v['counts']['solve'] for v in before.values()) >= 1
    monkeypatch.setattr(module, 'fit_pool', lambda *a, **k: pytest.fail('No cached refit'))
    monkeypatch.setattr(work_lease.FrozenWorkLease, 'assert_parent_alive',
                        lambda *a, **k: pytest.fail('No renewed historical lease'))
    again = run(case)
    assert again['cached'] and again['model_id'] == result['model_id']
    assert wallet(case) == before
    assert case.runtime.lifecycle.registry.load(result['model_id'])[0].digest == model.digest


def test_actual_forward_runtime_cannot_fit_before_its_wallet_is_bound(case, monkeypatch):
    import factorlab.discovery.v7_forward as forward

    monkeypatch.setattr(forward, 'fit_pool', lambda *a, **k: pytest.fail('Unmetered actual forward fit'),
                        raising=False)  # The fixed dispatcher no longer imports the old entry point.
    monkeypatch.setattr(module, 'fit_pool', lambda *a, **k: pytest.fail('Unmetered actual forward fit'))
    with pytest.raises(ValueError):
        case.runtime.fit_month('2026-10-01T00:00Z')
    assert not wallet(case)


def test_real_service_step_dispatches_default_pool_with_its_shared_wallet(case, monkeypatch):
    runner = ForwardService(case.runtime, work_authority=case.authority,
        config=ServiceConfig(poll=False, max_fits_per_step=1), guard=lambda: {'allowed': True})
    try:
        first = runner.step()
        fits = [item for item in first['actions'] if item['kind'] == 'fit']
        assert len(fits) == 1 and fits[0]['status'] == 'verified_waiting_for_receipt'
        _, identity = case.runtime.lifecycle.registry.load(fits[0]['model_id'])
        assert identity['version'] == 4
        ticket = ticket_for(case)
        assert ticket.value['authority_manifest_hash'] == case.authority.manifest_hash
        before = wallet(case)
        monkeypatch.setattr(module, 'fit_pool', lambda *a, **k: pytest.fail('Completed step refitted'))
        case.clock.now += pd.Timedelta(minutes=1)
        second = runner.step()
        assert not [item for item in second['actions'] if item['kind'] == 'fit']
        assert wallet(case) == before
    finally:
        runner.close()


def test_nonempty_profiled_V_pool_keeps_original_recipe_through_registry(case, monkeypatch):
    from test_v7_joint_pool import last, member

    from factorlab.discovery import neural_forward_ticket

    runtime, lc = case.runtime, case.runtime.lifecycle
    candidate = member({'op': 'mul', 'left': last('rv_1'), 'right': {'op': 'param', 'name': 'p'}},
        parameters={'p': {'lower': .5, 'upper': 2., 'initial': 1.}}, parameter_mode='profile')
    pool = FrozenPool((candidate,), reference_pool=case.pool,
                      reference_pipeline_id=case.pool.pipeline_id, max_profile_calls=9)
    pipeline = runtime.recipe_for(pool)
    evidence = {'start': '2026-09-01T00:00:00+00:00', 'end': '2026-09-02T00:00:00+00:00'}
    lc.exposure.record(source=lc.plan.source_id, symbols=lc.plan.symbols, batch='nov', role='S', **evidence)
    lc.freeze_batch(batch='nov', ordinal=1, start='2026-11-01T00:00Z',
        candidates=[{**pipeline, 'complexity': 1}], reference=runtime.recipe_for(case.pool),
        evidence_sources=[evidence])
    case.clock.now = pd.Timestamp('2026-11-01T00:00:10Z')
    monkeypatch.setattr(neural_forward_ticket, '_now', case.clock)
    result = runtime.fit_month('2026-11-01T00:00Z', stream='nov:' + pipeline['pipeline_id'],
                               pool=pool, authority=case.authority)
    model, identity = lc.registry.load(result['model_id'])
    ticket = ticket_for(case)
    assert identity['version'] == 4 and model.pool.digest == pool.digest
    assert model.pipeline_id == pipeline['pipeline_id'] == identity['pipeline']['pipeline_id']
    assert ticket.value['context']['role'] == 'V'
    record = case.authority.get(ticket.value['reservation_id'])
    assert record['counts']['solve'] == 11
    assert record['actual_counts']['solve'] == model.receipt['solve_calls']
    assert 2 < model.receipt['solve_calls'] <= 11
    before = wallet(case)
    case.clock.now = pd.Timestamp('2026-12-02T13:00Z')
    lc.select_validation('nov')  # Real empty support: concludes without selection.
    monkeypatch.setattr(module, 'fit_pool', lambda *a, **k: pytest.fail('Closed V cannot refit'))
    assert runtime.fit_month('2026-11-01T00:00Z', stream='nov:' + pipeline['pipeline_id'],
                            pool=pool, authority=case.authority)['cached']
    assert wallet(case) == before


@pytest.mark.parametrize('kind', ['joint', 'bare_ridge'])
def test_offline_models_cannot_seal_complete_pool_serving_jobs(case, kind):
    runtime = case.runtime
    job = runtime.lifecycle.plan_fit('2026-10-01T00:00Z', 'A0', runtime.recipe_for(case.pool))
    data = runtime.prepare_fit_snapshot(job).data
    model = (fit_pool(data, '2026-10-01T00:00Z', case.pool, budget=ChannelBudget()) if kind == 'joint'
             else fit_ridge(np.arange(40).reshape(-1, 1), np.linspace(.01, .1, 40)))
    with pytest.raises(ValueError):
        runtime.lifecycle.seal_model(job, model)
    assert not wallet(case)


def test_lost_runtime_manifest_cannot_turn_actual_jobs_into_offline_jobs(case):
    runtime = case.runtime
    job = runtime.lifecycle.plan_fit('2026-10-01T00:00Z', 'A0', runtime.recipe_for(case.pool))
    data = runtime.prepare_fit_snapshot(job).data
    model = fit_pool(data, '2026-10-01T00:00Z', case.pool, budget=ChannelBudget())
    (runtime.directory / 'forward.json').unlink()
    with pytest.raises(ValueError):
        runtime.lifecycle.seal_model(job, model)


@pytest.mark.parametrize('entry', ['verify', 'activate', 'active_model'])
def test_legacy_offline_joint_artifact_cannot_enter_actual_serving(case, entry):
    runtime, lc = case.runtime, case.runtime.lifecycle
    pipeline = runtime.recipe_for(case.pool)
    job = lc.plan_fit('2026-10-01T00:00Z', 'A0', pipeline)
    data = runtime.prepare_fit_snapshot(job).data
    model = fit_pool(data, '2026-10-01T00:00Z', case.pool, budget=ChannelBudget())
    identifier = lc.registry.save(model, pipeline)
    assert lc.registry.load(identifier)[1]['version'] == 1
    completed = lc.now().isoformat()
    # Reproduce historical v1 producer output, with an internally consistent chain.
    with lc.db:
        lc.db.execute('UPDATE jobs SET status=?,model=?,completed=?,failure=NULL WHERE id=?',
                      ('verified', identifier, completed, job))
        lc._event('model_verified', {'job': job, 'model': identifier, 'completed': completed})
        if entry == 'active_model':
            lc.db.execute('INSERT INTO active VALUES (?,?,?)', ('A0', job, 1))
            lc.db.execute('UPDATE jobs SET status=?,activation=? WHERE id=?', ('activated', completed, job))
            lc._event('activated', {'job': job, 'generation': 1, 'origin': completed,
                                   'activation_delay_seconds': 10., 'within_24h': True})
    with pytest.raises(ValueError):
        if entry == 'verify':
            lc.verify()
        elif entry == 'activate':
            lc.activate(job, lc.now())
        else:
            lc.active_model('A0')


def test_complete_wallet_pending_publication_and_failed_log_cannot_register(case, monkeypatch):
    original = module._write

    def inaccessible(path, value):
        if path.name == 'failure.json':
            raise OSError('failure log inaccessible')
        return original(path, value)

    monkeypatch.setattr(module, '_write', inaccessible)
    monkeypatch.setattr(ForwardPublicationWitness, 'complete',
                        lambda *a: (_ for _ in ()).throw(RuntimeError('final lease failure')))
    with pytest.raises(RuntimeError, match='final lease'):
        run(case)
    ticket = ticket_for(case)
    assert case.authority.get(ticket.value['reservation_id'])['status'] == 'completed'
    assert (ticket.path.parent / 'launch/certificate.json').exists()
    with pytest.raises(ValueError, match='pending'):
        ForwardPublicationWitness.read_completed(ticket)
    assert case.runtime.lifecycle.db.execute('SELECT model FROM jobs').fetchone()[0] is None


def test_missing_original_publication_witness_blocks_registry_restore(case):
    result = run(case)
    ticket = ticket_for(case)
    path = ticket.path.parent / 'launch/publication_witness/witness.sqlite'
    path.unlink()
    with pytest.raises(ValueError, match='missing'):
        case.runtime.lifecycle.registry.load(result['model_id'])
    assert not path.exists()


def test_preparation_failure_is_already_prepaid_and_never_refunds(case, monkeypatch):
    calls = []

    def snapshot_failure(job):
        current = wallet(case)
        assert sum(v['seconds'] for v in current.values()) > 0, 'Snapshot began before monthly time prepayment'
        calls.append(current)
        raise OSError('synthetic preparation failure')

    monkeypatch.setattr(case.runtime, 'prepare_fit_snapshot', snapshot_failure)
    with pytest.raises(OSError, match='preparation failure'):
        run(case)
    charged = wallet(case)
    assert calls and charged == calls[0]
    with pytest.raises((ValueError, OSError, RuntimeError)):
        run(case)
    assert len(calls) == 1 and wallet(case) == charged


def test_registry_v4_manifest_cannot_borrow_different_job(case):
    result = run(case)
    model, identity = case.runtime.lifecycle.registry.load(result['model_id'])
    with pytest.raises(ValueError, match='actual job'):
        module.verify_pool_admission(model, identity['admission'],
            expected_pipeline=identity['pipeline'], expected_job='0' * 64)
    path = case.runtime.lifecycle.registry.directory / result['model_id'] / 'manifest.json'
    record = json.loads(path.read_text())
    record['identity']['job'] = '0' * 64
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        case.runtime.lifecycle.registry.load(result['model_id'])
