"""Real artificial child/feed/wallet, private Windows kernel/gate/clock seams.

These exercise engineering; they are not a Linux resource or scientific gate.
"""
import multiprocessing
import os
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_v7_algorithm_validation import case, completed, module  # noqa: F401
from test_v7_neural_research_ticket import _synthetic_kernel
from test_v7_rolling_registry import candles

from factorlab.discovery import v7_service as service_module
from factorlab.discovery import work_lease
from factorlab.discovery.neural_forward_ticket import ForwardFitTicket
from factorlab.protocol import fingerprint


def _child(root, path, shift_seconds):
    import factorlab.protocol
    from factorlab.discovery import neural_forward_ticket
    factorlab.protocol.ROOT = Path(root)
    neural_forward_ticket._now = lambda: pd.Timestamp.now(tz='UTC')+pd.Timedelta(seconds=shift_seconds)
    producer = module._producer()
    original = work_lease._observe
    def observe(lease):
        value = original(lease)
        value['wall'] = (datetime.now(UTC)+pd.Timedelta(seconds=shift_seconds).to_pytimedelta()).timestamp()
        return value
    work_lease._observe = observe
    from factorlab.discovery import forward_service_scope
    forward_service_scope._observe = observe
    producer.verify_gates = lambda *a, **kw: None
    producer._kernel_snapshot = lambda: _synthetic_kernel([os.getpid(), os.getppid()])
    producer.worker(path)


def _future_clock(monkeypatch, runtime, start):
    # Import before monkeypatching, so teardown restores the physical observer.
    from factorlab.discovery import forward_service_scope
    clock = [start+pd.Timedelta(seconds=10)]
    original_issue, original_observe = work_lease.FrozenWorkLease.issue, work_lease._observe
    shifts = [pd.Timedelta(seconds=int((clock[0]-pd.Timestamp.now(tz='UTC')).total_seconds()))]
    def issue(seconds):
        lease = original_issue(seconds)
        next_shift = pd.Timedelta(seconds=int((clock[0]-pd.Timestamp.now(tz='UTC')).total_seconds()))
        # Calendar replay may advance, but issuing work cannot roll time back.
        shifts[0] = max(shifts[0], next_shift)
        return replace(lease, wall_started_at=(pd.Timestamp(lease.wall_started_at)+shifts[0]).isoformat(),
            wall_deadline=(pd.Timestamp(lease.wall_deadline)+shifts[0]).isoformat())
    def observe(lease):
        value = original_observe(lease)
        value['wall'] = (datetime.now(UTC)+shifts[0].to_pytimedelta()).timestamp()
        return value
    monkeypatch.setattr(work_lease.FrozenWorkLease, 'issue', issue)
    monkeypatch.setattr(work_lease, '_observe', observe)
    monkeypatch.setattr(forward_service_scope, '_observe', observe)
    class FutureDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.now(tz)+shifts[0].to_pytimedelta()
    monkeypatch.setattr('factorlab.discovery.monthly_work_authority.datetime', FutureDatetime)
    monkeypatch.setattr(module, '_now', lambda: clock[0])
    monkeypatch.setattr('factorlab.discovery.neural_forward_ticket._now', lambda: clock[0])
    runtime.clock = runtime.lifecycle.clock = runtime.feed.clock = lambda: clock[0]
    producer = module._producer()
    original_observation = producer._observation
    def observation(lease):
        value = original_observation(lease)
        value['wall'] = (pd.Timestamp(value['wall'])+shifts[0]).isoformat()
        return value
    monkeypatch.setattr(producer, '_observation', observation)
    return clock, shifts


def _seed(runtime, clock, start):
    clock[0] = start-pd.Timedelta(minutes=30)
    bars = candles(start-pd.DateOffset(months=5), start-pd.Timedelta(hours=1))
    bars['symbol'] = 'X'
    path = runtime.directory.parent/'actual-artificial-seed.parquet'
    bars.to_parquet(path)
    runtime.feed.import_seed(path)
    clock[0] = start+pd.Timedelta(seconds=10)


def test_real_child_publication_actual_origins_endpoint_once_and_qualifier_denied(completed, monkeypatch):  # noqa: F811
    bridge, catalogue, recipe, runtime, _ = completed
    start = module._now().normalize().replace(day=1)+pd.DateOffset(months=1)
    protocol = module.AlgorithmValidation.freeze(runtime, bridge.authority, catalogue, start=start,
        rule=module.AlgorithmValidationRule(minimum_coverage=.0001, minimum_gain=0.))
    producer = module._producer()
    clock, shift = _future_clock(monkeypatch, runtime, start)
    _seed(runtime, clock, start)
    monkeypatch.setattr(producer, 'verify_gates', lambda *a, **kw: None)
    monkeypatch.setattr(producer, '_strict_kernel', lambda *a, **kw: None)
    children = []
    class ActualChild:
        def __init__(self, command, **kwargs):
            assert Path(command[2]).name == 'algorithm_validation_producer.py'
            assert kwargs['env']['OMP_NUM_THREADS'] == '1'
            self.process = multiprocessing.get_context('spawn').Process(target=_child,
                args=(str(runtime.directory.parent), command[-1], shift[0].total_seconds()))
            self.process.start()
            self.pid = self.process.pid
            children.append(self)
        @property
        def returncode(self):
            return self.process.exitcode
        def poll(self):
            return self.process.exitcode
    monkeypatch.setattr(producer.subprocess, 'Popen', ActualChild)
    monkeypatch.setattr(producer, '_kernel_snapshot', lambda: _synthetic_kernel(
        [os.getpid()]+[c.pid for c in children if c.poll() is None]))
    protocol.register_fit_gates(recipe.pipeline_id, start,
        gates={'directory': str(runtime.directory/'private-gates'), 'artifacts': {}})
    service = service_module.ForwardService(runtime, work_authority=bridge.authority,
        config=service_module.ServiceConfig(poll=False, wall_seconds=7200), guard=lambda: {'allowed': True})
    try:
        response = service.step()
        controller = next(a for a in response['actions'] if a['kind'] == 'algorithm_controller')
        result = next(a for a in controller['actions'] if a['kind'] == 'algorithm_fit' and a.get('status') == 'completed')
        assert len([k for k in service.state['fit_attempts'] if k.startswith('algorithm:')]) == 1
        prepared = protocol.read_prepared(recipe.pipeline_id, start, statuses=('completed',))
        before = bridge.authority.status(start.strftime('%Y-%m'))
        actions = []
        service._fit_pending(clock[0], actions, time.monotonic()+300, fit_allowance=0)
        assert any(a['kind'] == 'algorithm_owned_job' and a['job'] == prepared['job'] for a in actions)
        assert bridge.authority.status(start.strftime('%Y-%m')) == before
        assert runtime.lifecycle.db.execute('SELECT failure FROM jobs WHERE id=?', (prepared['job'],)).fetchone() == (None,)
    finally:
        # Keep fixture-owned runtime open for readonly artifact/endpoint checks.
        service.db.close()
        service._owner_lock.release()
    assert result['shadow_only'] and not result['serving_eligible']
    with pytest.raises(ValueError):
        ForwardFitTicket.load(result['ticket'])
    admitted = producer.admit(protocol, recipe.pipeline_id, start)
    counts = bridge.authority.get(prepared['reservation_id'])['actual_counts']
    assert counts['neural_fit'] == 3 and counts['epoch'] == 3 and counts['solve'] == 8
    with pytest.raises(ValueError):
        protocol.launch_fit(prepared)
    # Actual new receipt; prior assumed seed produces no old origins.
    clock[0] = start+pd.Timedelta(hours=1, seconds=1)
    bars = candles(start-pd.DateOffset(months=5), start+pd.Timedelta(hours=1)).tail(2).copy()
    bars['symbol'] = 'X'
    runtime.feed.receive(bars, received_at=clock[0], response_hash=fingerprint('actual-source-one'))
    assert protocol.tick()['predictions'] == 1
    assert protocol.tick()['predictions'] == 0
    forecasts = [e['payload'] for e in producer._events(protocol) if e['kind'] == 'forecast']
    assert len(forecasts) == 1 and set(forecasts[0]['predictions']) == {recipe.pipeline_id, 'K0', 'Pv'}
    assert forecasts[0]['tickets'] == {recipe.pipeline_id: admitted['ticket_digest']}
    # Synthetic positive-control future path; no score/certificate is supplied.
    # The actual producer must still derive the target from 13 received OHLCs.
    target = forecasts[0]['predictions'][recipe.pipeline_id]
    clock[0] = start+pd.Timedelta(hours=14, seconds=1)
    future = candles(start+pd.Timedelta(hours=1), start+pd.Timedelta(hours=14))
    prices = np.exp(4+np.arange(len(future))*target/np.sqrt(12))
    future['symbol'], future['open'], future['close'] = 'X', prices, prices
    future['high'], future['low'] = prices*1.003, prices/1.003
    runtime.feed.receive(future, received_at=clock[0], response_hash=fingerprint('actual-source-label'))
    assert protocol.tick()['settlements'] == 1
    with pytest.raises(ValueError, match='not matured'):
        protocol.select_validation()
    clock[0] = pd.Timestamp(protocol.identity['V_scoring_cutoff'])
    endpoint = protocol.select_validation()
    assert endpoint['score']['common_pairs'] == 1 and endpoint['score']['nominal_hours'] >= 672
    assert endpoint['formal_claim'] is False and endpoint['serving_eligible'] is False
    assert endpoint['score']['selected'] == recipe.pipeline_id, forecasts[0]['predictions']
    assert protocol.select_validation() == endpoint
    # Irrespective of direction, recipes other than the sole selected winner
    # cannot enter the separate qualifier; no ordinary claim/adoption is created.
    with pytest.raises(ValueError):
        protocol.fit_plan('unregistered', protocol.identity['qualifier_start'], stage='qualifier')
    assert runtime.lifecycle.db.execute('SELECT COUNT(*) FROM batches').fetchone()[0] == 0
    assert runtime.lifecycle.exposure.db.execute('SELECT COUNT(*) FROM confirmations').fetchone()[0] == 0
    assert runtime.lifecycle.forecasts.db.execute('SELECT COUNT(*) FROM forecasts').fetchone()[0] == 0
    # The actual single winner now drives all three new monthly cold fits. Each
    # artificial seed is imported before its own cutoff; none creates an origin.
    for offset in range(3):
        month = pd.Timestamp(protocol.identity['qualifier_start'])+pd.DateOffset(months=offset)
        clock[0] = month-pd.Timedelta(minutes=30)
        history = candles(start-pd.DateOffset(months=5), month-pd.Timedelta(hours=1))
        history['symbol'] = 'X'
        seed_path = runtime.directory.parent/f'qualifier-training-{offset}.parquet'
        history.to_parquet(seed_path)
        runtime.feed.import_seed(seed_path)
        clock[0] = month+pd.Timedelta(seconds=10)
        plan = protocol.fit_plan(recipe.pipeline_id, month, stage='qualifier')
        assert plan['recipe'] == recipe.to_dict() and plan['stage'] == 'qualifier'
        protocol.register_fit_gates(recipe.pipeline_id, month, stage='qualifier',
            gates={'directory': str(runtime.directory/f'private-q-gates-{offset}'), 'artifacts': {}})
        work = protocol.step(deadline_monotonic=time.monotonic()+330, max_fits=1)
        assert any(a['kind'] == 'algorithm_fit' and a['status'] == 'completed' for a in work['actions'])
        assert not protocol.pending_work()
        done = protocol.read_prepared(recipe.pipeline_id, month, stage='qualifier', statuses=('completed',))
        assert bridge.authority.get(done['reservation_id'])['actual_counts']['neural_fit'] == 3
        clock[0] = month+pd.Timedelta(hours=1, seconds=1)
        actual = candles(start-pd.DateOffset(months=5), month+pd.Timedelta(hours=1)).tail(2).copy()
        actual['symbol'] = 'X'
        runtime.feed.receive(actual, received_at=clock[0], response_hash=fingerprint(['q-origin', offset]))
        assert protocol.tick()['predictions'] == 1
        forecast = [e['payload'] for e in producer._events(protocol) if e['kind'] == 'forecast'][-1]
        assert forecast['stage'] == 'qualifier' and set(forecast['tickets']) == {recipe.pipeline_id}
        clock[0] = month+pd.Timedelta(hours=14, seconds=1)
        future = candles(month+pd.Timedelta(hours=1), month+pd.Timedelta(hours=14))
        q_target = forecast['predictions'][recipe.pipeline_id]
        prices = np.exp(4+np.arange(len(future))*q_target/np.sqrt(12))
        future['symbol'], future['open'], future['close'] = 'X', prices, prices
        future['high'], future['low'] = prices*1.003, prices/1.003
        runtime.feed.receive(future, received_at=clock[0], response_hash=fingerprint(['q-label', offset]))
        assert protocol.tick()['settlements'] == 1
    clock[0] = pd.Timestamp(protocol.identity['qualifier_scoring_cutoff'])
    qualifier = protocol.conclude_qualifier()
    assert qualifier['score']['common_pairs'] == 3
    assert qualifier['score']['nominal_hours'] == int((pd.Timestamp(protocol.identity['qualifier_end'])
        -pd.Timestamp(protocol.identity['qualifier_start']))/pd.Timedelta(hours=1))
    assert qualifier['formal_claim'] is False and qualifier['serving_eligible'] is False
    assert protocol.conclude_qualifier() == qualifier
    # Merely borrowing a namespace never skips the canonical authority check.
    unknown_job = runtime.lifecycle.plan_fit(pd.Timestamp(protocol.identity['qualifier_end'])-pd.DateOffset(months=1),
        protocol.identity['batch']+':qualifier:unregistered', runtime.recipe_for(recipe))
    service = service_module.ForwardService(runtime, work_authority=bridge.authority,
        config=service_module.ServiceConfig(session_id='ownership-audit', poll=False, wall_seconds=7200),
        guard=lambda: {'allowed': True})
    try:
        with pytest.raises(ValueError, match='Unregistered algorithm namespace'):
            service._fit_pending(clock[0], [], time.monotonic()+300, fit_allowance=0)
        assert runtime.lifecycle.db.execute('SELECT failure FROM jobs WHERE id=?', (unknown_job,)).fetchone() == (None,)
    finally:
        service.db.close()
        service._owner_lock.release()
    # Completed historical read does not require the old issuer to stay alive.
    monkeypatch.setattr(work_lease.FrozenWorkLease, 'assert_parent_alive', lambda *a, **k: (_ for _ in ()).throw(AssertionError('old lease read')))
    assert producer.admit(protocol, recipe.pipeline_id, start)['ticket_digest'] == admitted['ticket_digest']


def test_endpoint_read_failure_cannot_relook_even_without_result_file(completed, monkeypatch):  # noqa: F811
    bridge, catalogue, _, runtime, _ = completed
    start = module._now().normalize().replace(day=1)+pd.DateOffset(months=1)
    protocol = module.AlgorithmValidation.freeze(runtime, bridge.authority, catalogue, start=start)
    producer = module._producer()
    monkeypatch.setattr(module, '_now', lambda: pd.Timestamp(protocol.identity['V_scoring_cutoff']))
    def failed(*args):
        raise OSError('failure after consuming one look')
    monkeypatch.setattr(producer, '_endpoint_score', failed)
    with pytest.raises(OSError):
        protocol.select_validation()
    with pytest.raises(ValueError, match='cannot relook'):
        protocol.select_validation()
    assert [e['kind'] for e in producer._events(protocol)] == ['endpoint_read']
