"""Staged typed preparation tests, artificial clocks/kernel; no V performance proof."""
import multiprocessing
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_v7_neural_research_ticket import _actual_child, _synthetic_kernel, launcher
from test_v7_nonlinear_channel import artificial_data
from test_v7_rolling_registry import candles

from factorlab.discovery import algorithm_validation as module
from factorlab.discovery import work_lease
from factorlab.discovery.algorithm_shortlist import AlgorithmCatalogue, AlgorithmShortlistRule
from factorlab.discovery.joint_pool import FrozenPool
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
from factorlab.discovery.neural_forward import FrozenNeuralRecipe
from factorlab.discovery.neural_research_ticket import issue_research_ticket
from factorlab.discovery.nonlinear_channel import ResidualConfig
from factorlab.discovery.nonlinear_orchestration import input_data_identity
from factorlab.discovery.rolling import RollingSpec
from factorlab.discovery.v7_forward import ForwardConfig, ForwardRuntime
from factorlab.discovery.v7_memory import search_exposure_context
from factorlab.discovery.v7_run import RunStore, V7Config, monthly_budget
from factorlab.discovery.work_authority_bridge import WorkAuthorityBridge
from factorlab.protocol import fingerprint


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    (tmp_path / 'factorlab').mkdir()
    (tmp_path / 'factorlab/frozen.py').write_text('source=1')
    (tmp_path / 'pyproject.toml').write_text('[project]')
    (tmp_path / 'uv.lock').write_text('version=1')
    data = artificial_data()
    # Genuine frozen production minimum=240, rather than silently changing the
    # future runtime to the small legacy S fixture's minimum_train_rows=12.
    metadata = data.frame.metadata.loc[data.frame.metadata.index.repeat(2)].reset_index(drop=True)
    times = pd.DatetimeIndex(metadata.available_at) - pd.to_timedelta(np.tile([6, 0], len(data.target)), unit='h')
    metadata['available_at'] = times
    metadata['forecast_window_start'] = times
    metadata['feature_open_time'] = times-pd.Timedelta(hours=1)
    metadata['sample_id'] = [f'synthetic-shared-{i}' for i in range(len(metadata))]
    values = np.zeros((len(metadata), data.frame.values.shape[1]))
    values[:, :8] = 1.
    values[:, 8] = np.repeat(data.frame.values[:, 8], 2)
    # Predeclared simple nonlinearity: this artificial positive signal supplies
    # a completed candidate, not evidence of performance on market data.
    target = np.exp(.8*np.tanh(2*values[:, 8]))
    data = replace(data, frame=replace(data.frame, values=values, metadata=metadata),
        target=target, observed=np.repeat(data.observed, 2),
        maturity=times+pd.Timedelta(hours=12), spec=RollingSpec())
    runtime = ForwardRuntime(tmp_path / 'forward', tmp_path / 'live-feed',
        config=ForwardConfig(symbols=('X',), archive_delay_seconds=data.spec.availability_delay_seconds))
    config = V7Config(evaluation_mode='joint', symbols=('X',))
    store = RunStore(tmp_path / 'research', config, data.spec, exposure_path=runtime.lifecycle.exposure.path)
    authority = MonthlyWorkAuthority(tmp_path / 'wallet', run_id='campaign', task='volatility', universe=['X'], partition_policy='standard')
    month, cutoff = '2025-04', '2025-04-01T00:00:00+00:00'
    exposed = search_exposure_context('controller-data', cutoff, ['X'])
    store.exposure.record(**{k: v for k, v in exposed.items() if k != 'data_id'}, role='S')
    state = {'status': 'running', 'cutoff': cutoff, 'data_id': 'controller-data',
        'search_exposure': exposed, 'budget': monthly_budget(config, month).to_dict(),
        'frozen_reference_pool': FrozenPool().to_dict(),
        'nonlinear_input_data_id': input_data_identity(data, cutoff)}
    store.save(month, state, 'frozen_controller_input')

    def persist(value):
        state['budget'] = value
        store.save(month, state, 'budget_projection')

    bridge = WorkAuthorityBridge(store, month, state, authority, persist)
    recipe = FrozenNeuralRecipe(rolling_spec=data.spec, config=ResidualConfig(enabled=True, epochs=1), output='encoder_ridge')
    from factorlab.discovery.s_nonlinear_ticket import freeze_nonlinear_plan
    from factorlab.discovery.monthly_schedule import default_schedule
    from factorlab.discovery.scheduled_search import ScheduledSearch
    ScheduledSearch(bridge, default_schedule(config, month=month))
    freeze_nonlinear_plan(bridge, recipe, kinds=('complete_neural_S',))
    bridge.begin_work('baseline', counts={'solve': 7}, lease_seconds=300,
        binding={'stage': 'baseline'}).complete({'artificial_owner_setup': True, 'training_performed': False})
    gates = {'directory': str(tmp_path / 'artificial-gate-placeholder'),
             'artifacts': {'protocol': 'p', 'probes': ['q1', 'q2'], 'positive_control': 'c'}}
    monkeypatch.setattr('factorlab.discovery.neural_research_ticket._gates', lambda *args: {'private_test_seam': True})
    yield bridge, data, recipe, gates, runtime
    store.close()
    runtime.close()


@pytest.fixture
def completed(case, monkeypatch):
    bridge, data, recipe, gates, runtime = case
    recipe = replace(recipe, config=replace(recipe.config, epochs=1))
    cat = AlgorithmCatalogue.freeze(bridge, (recipe,), rule=AlgorithmShortlistRule(
        minimum_hour_coverage=.001, minimum_symbol_hour_coverage=.001))
    ticket = issue_research_ticket(bridge, data, recipe, gates=gates)
    children = []

    class ActualChild:
        def __init__(self, command, **kwargs):
            self.process = multiprocessing.get_context('spawn').Process(target=_actual_child,
                args=(str(bridge.store.directory.parent), str(ticket.path)))
            self.process.start()
            self.pid = self.process.pid
            children.append(self)

        @property
        def returncode(self):
            return self.process.exitcode

        def poll(self):
            return self.process.exitcode

    monkeypatch.setattr(launcher.subprocess, 'Popen', ActualChild)
    monkeypatch.setattr(launcher, '_kernel_snapshot', lambda: _synthetic_kernel(
        [os.getpid()] + [p.pid for p in children if p.poll() is None]))
    monkeypatch.setattr(launcher, '_strict_kernel', lambda *args: None)
    launcher.launch_research(ticket.path)
    cat.record_ticket(ticket.path)
    assert len(cat.seal_shortlist()['algorithm_shortlist']) == 1
    yield bridge, cat, recipe, runtime, ticket


def freeze(completed):
    bridge, cat, _, runtime, _ = completed
    start = module._now().normalize().replace(day=1) + pd.DateOffset(months=1)
    return module.AlgorithmValidation.freeze(runtime, bridge.authority, cat, start=start), start


@pytest.mark.parametrize('values', [{'minimum_coverage': 0}, {'minimum_coverage': 1.1},
    {'minimum_coverage': float('nan')}, {'minimum_gain': float('inf')}, {'minimum_gain': -1}])
def test_rule_cannot_weaken_to_empty_or_nonfinite(values):
    with pytest.raises(ValueError):
        module.AlgorithmValidationRule(**values)


def test_private_endpoint_uses_common_hours_and_exact_labels_not_row_weights():
    rows = {name: [{'hour': '2026-11-01T00:00:00Z', 'symbol': 'X', 'loss': loss,
                   'label_identity': 'same-revision'}] for name, loss in [('a', .8), ('b', .8), ('K0', 1.), ('Pv', 1.)]}
    for values in rows.values():
        values.extend([{'hour': '2026-11-01T01:00:00Z', 'symbol': symbol,
                        'loss': 0., 'label_identity': 'second-hour-revision'} for symbol in ['X', 'Y']])
    score = module._paired_selection(rows, ['b', 'a'], start='2026-11-01T00:00:00Z',
        end='2026-11-01T02:00:00Z', symbols=['X', 'Y'], rule=module.AlgorithmValidationRule(minimum_coverage=.7))
    assert score['coverage'] == .75 and score['hour_coverage'] == 1.
    assert score['records'][0]['loss'] == .4  # two equal hours, not three equal rows
    assert score['selected'] == 'a'
    rows['b'][0]['label_identity'] = 'different'
    with pytest.raises(ValueError, match='label revisions'):
        module._paired_selection(rows, ['a', 'b'], start='2026-11-01T00:00:00Z',
            end='2026-11-01T02:00:00Z', symbols=['X', 'Y'], rule=module.AlgorithmValidationRule())


def test_private_endpoint_missing_candidate_cannot_shrink_nominal_denominator():
    rows = {'a': [], 'K0': [], 'Pv': []}
    result = module._paired_selection(rows, ['a'], start='2026-11-01T00:00:00Z',
        end='2026-12-01T00:00:00Z', symbols=['X'], rule=module.AlgorithmValidationRule())
    assert result['selected'] is None and result['nominal_hours'] == 720
    assert result['coverage'] == 0.


def test_full_algorithm_candidates_cannot_obtain_different_work_ceiling():
    first = FrozenNeuralRecipe(config=ResidualConfig(enabled=True, epochs=1))
    other = replace(first, output='encoder_ridge')
    counts = module._common_work_counts({r.pipeline_id: r.to_dict() for r in (first, other)})
    assert counts['neural_fit'] == 3 and counts['epoch'] == 3 and counts['solve'] == 8
    with pytest.raises(ValueError, match='identical prepaid'):
        module._common_work_counts({'one': first.to_dict(), 'two': replace(first,
            config=replace(first.config, epochs=2)).to_dict()})


def test_real_completed_catalogue_freezes_separate_context_and_no_claim_slots(completed, monkeypatch):
    bridge, cat, recipe, runtime, _ = completed
    before = bridge.authority.verify()['reservations']
    protocol, start = freeze(completed)
    assert bridge.authority.verify()['reservations'] == before
    assert runtime.lifecycle.db.execute('SELECT COUNT(*) FROM batches').fetchone()[0] == 0
    assert runtime.lifecycle.exposure.db.execute('SELECT COUNT(*) FROM confirmations').fetchone()[0] == 0
    assert module.AlgorithmValidation.freeze(runtime, bridge.authority, cat, start=start).digest == protocol.digest
    assert protocol.identity['qualifier_start'] == (start + pd.DateOffset(months=2)).isoformat()
    assert protocol.identity['qualifier_end'] == (start + pd.DateOffset(months=5)).isoformat()
    plan = protocol.fit_plan(recipe.pipeline_id, start)
    assert plan['counts']['neural_fit'] == 3 and plan['counts']['epoch'] == 3
    assert set(plan['counts']) == set(module.DEFAULT_LIMITS)
    assert plan['stage'] == 'V' and plan['serving_eligible'] is False
    assert plan['train_start'] == (start - pd.DateOffset(months=3)).isoformat()
    assert protocol.identity['references']['Pv'] == runtime.recipe_for(recipe.pool)
    with pytest.raises(ValueError, match='cannot change'):
        module.AlgorithmValidation.freeze(runtime, bridge.authority, cat, start=start,
            rule=module.AlgorithmValidationRule(minimum_gain=.02))
    with pytest.raises(ValueError, match='real execution window'):
        protocol.prepare_fit(recipe.pipeline_id, start, gates={})
    with pytest.raises(ValueError, match='schedule'):
        protocol.fit_plan(recipe.pipeline_id, start, stage='qualifier')
    with pytest.raises(ValueError, match='not matured'):
        protocol.select_validation()
    monkeypatch.setattr(module, '_now', lambda: pd.Timestamp(protocol.identity['qualifier_end']))
    empty = protocol.select_validation()
    assert empty['score']['selected'] is None and empty['score']['coverage'] == 0
    assert protocol.select_validation() == empty
    runtime.lifecycle.exposure.clock = module._now
    with pytest.raises(ValueError, match='same-batch search'):
        runtime.lifecycle.exposure.record(source=runtime.feed.identity, symbols=['X'], start=start,
            end=start+pd.DateOffset(months=1), role='S', batch=cat.identity['exposure']['batch'])
    assert bridge.authority.verify()['reservations'] == before


def test_missing_SQL_cannot_reinitialize_protocol_or_repurpose_S_witness(completed):
    bridge, cat, _, runtime, ticket = completed
    protocol, start = freeze(completed)
    (protocol.directory / 'validation.sqlite').unlink()
    with pytest.raises((ValueError, module.sqlite3.OperationalError)):
        module.AlgorithmValidation.freeze(runtime, bridge.authority, cat, start=start)
    # S and ordinary-forward ticket schemas remain distinct from prepared V work.
    from factorlab.discovery.neural_forward_ticket import ForwardFitTicket
    with pytest.raises(ValueError, match='forward ticket schema'):
        ForwardFitTicket.load(ticket.path)


@pytest.mark.parametrize('failure', [False, True])
def test_real_feed_cutoff_preparation_prepays_once_and_never_grants_launch(completed, monkeypatch, failure):
    bridge, _, recipe, runtime, _ = completed
    protocol, start = freeze(completed)
    gates = {'directory': str(runtime.directory/'private-artificial-gates'), 'artifacts': {}}
    monkeypatch.setattr(module._producer(), 'verify_gates', lambda *a, **kw: None)
    # Private future-wall seam only. Original issuer identity and monotonic
    # deadlines remain real. No external clock or test_only switch in this API.
    future = start + pd.Timedelta(seconds=10)
    original_issue, original_observe = work_lease.FrozenWorkLease.issue, work_lease._observe
    shift = pd.Timedelta(seconds=int((future-pd.Timestamp.now(tz='UTC')).total_seconds()))

    def issue(seconds):
        lease = original_issue(seconds)
        return replace(lease, wall_started_at=(pd.Timestamp(lease.wall_started_at)+shift).isoformat(),
            wall_deadline=(pd.Timestamp(lease.wall_deadline)+shift).isoformat())

    def observe(lease):
        value = original_observe(lease)
        value['wall'] = (datetime.now(UTC)+shift.to_pytimedelta()).timestamp()
        return value

    monkeypatch.setattr(module, '_now', lambda: future)
    monkeypatch.setattr('factorlab.discovery.neural_forward_ticket._now', lambda: future)
    monkeypatch.setattr(work_lease.FrozenWorkLease, 'issue', issue)
    monkeypatch.setattr(work_lease, '_observe', observe)
    clock = [start-pd.Timedelta(minutes=30)]
    runtime.clock = runtime.lifecycle.clock = runtime.feed.clock = lambda: clock[0]
    bars = candles(start-pd.DateOffset(months=5), start-pd.Timedelta(hours=1))
    bars['symbol'] = 'X'
    path = runtime.directory.parent / 'artificial-live-seed.parquet'
    bars.to_parquet(path)
    runtime.feed.import_seed(path)
    clock[0] = future
    original_prepare = module.prepare_forward_snapshot
    prepared = []

    def tracked(actual_runtime, job):
        wallet = bridge.authority.status(start.strftime('%Y-%m'))
        assert wallet['reserved_seconds'] == 300
        assert wallet['counts']['neural_fit'] == 3
        prepared.append(job)
        if failure:
            raise OSError('artificial preparation disk failure')
        return original_prepare(actual_runtime, job)

    monkeypatch.setattr(module, 'prepare_forward_snapshot', tracked)
    if failure:
        with pytest.raises(OSError, match='disk failure'):
            protocol.prepare_fit(recipe.pipeline_id, start, gates=gates)
        owner = protocol.owns_job(prepared[0])
        assert owner['status'] == 'failed_consumed'
        assert owner['pipeline_id'] == recipe.pipeline_id
    else:
        result = protocol.prepare_fit(recipe.pipeline_id, start, gates=gates)
        assert result['shadow_only'] is True and result['formal_claim'] is False
        assert protocol.read_prepared(recipe.pipeline_id) == module._copy(result)
        snapshot = module.json.loads(Path(result['snapshot_ref']['path']).read_text())
        assert snapshot['availability'] == 'actual_receipts'
        assert all(pd.Timestamp(row['published_at']) < start for row in snapshot['publication_lineage'])
        from factorlab.discovery.neural_forward_ticket import ForwardFitTicket
        incompatible = protocol.directory / 'wrong-forward.json'
        incompatible.write_text(module.json.dumps({'value': result, 'sha256': fingerprint(result)}))
        with pytest.raises(ValueError, match='forward ticket schema'):
            ForwardFitTicket.load(incompatible)
    before = bridge.authority.status(start.strftime('%Y-%m'))
    with pytest.raises(ValueError, match='already prepaid'):
        protocol.prepare_fit(recipe.pipeline_id, start, gates=gates)
    assert bridge.authority.status(start.strftime('%Y-%m')) == before
    assert len(prepared) == 1
    assert runtime.feed.db.execute('SELECT COUNT(*) FROM bars WHERE origin=1').fetchone()[0] == 0
    assert runtime.lifecycle.forecasts.db.execute('SELECT COUNT(*) FROM forecasts').fetchone()[0] == 0
