"""G03 queue engineering: real child/SQL; private input+gate+kernel seams, no Linux or data proof."""
import importlib.util
import json
import sys

ROOT = next(p for p in __import__("pathlib").Path(__file__).resolve().parents if (p/"pyproject.toml").is_file())
sys.path.insert(0, str(ROOT/"tests"))
import multiprocessing
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_v7_neural_research_ticket import _actual_child, _synthetic_kernel, launcher
from test_v7_nonlinear_channel import artificial_data

from factorlab.discovery import algorithm_validation as module
from factorlab.discovery.algorithm_shortlist import AlgorithmCatalogue, AlgorithmShortlistRule
from factorlab.discovery.joint_pool import FrozenPool
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
from factorlab.discovery.neural_forward import FrozenNeuralRecipe
from factorlab.discovery.neural_research_ticket import issue_research_ticket
from factorlab.discovery.nonlinear_channel import ResidualConfig, StructureSpec
from factorlab.discovery.nonlinear_orchestration import input_data_identity
from factorlab.discovery.rolling import RollingSpec
from factorlab.discovery.v7_forward import ForwardConfig, ForwardRuntime
from factorlab.discovery.v7_memory import search_exposure_context
from factorlab.discovery.v7_run import RunStore, V7Config, monthly_budget
from factorlab.discovery.work_authority_bridge import WorkAuthorityBridge


@pytest.fixture
def case(tmp_path, monkeypatch, request):
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
    config = V7Config(evaluation_mode='joint', symbols=('X',), nonlinear_mode='isolated')
    authority = MonthlyWorkAuthority(tmp_path / 'wallet', run_id='campaign', task='volatility', universe=['X'], partition_policy='standard')
    from factorlab.discovery.monthly_schedule import default_schedule
    from factorlab.discovery.neural_forward_ticket import _bind_authority
    _bind_authority(runtime, authority)
    original_month = pd.Timestamp.now(tz='UTC').normalize().replace(day=1)
    # Explicit artificial calendar seam: on a quarter month use the next
    # non-quarter synthetic S dataset and a prospective hypothetical V clock.
    if (original_month.month-1) % 3 == 0:
        original_month += pd.DateOffset(months=1)
    cutoff = original_month.isoformat()
    _install_artificial_exposure_clock(original_month + pd.Timedelta(days=15), monkeypatch)
    runtime.clock = lambda: original_month + pd.Timedelta(days=15)
    runtime.lifecycle.exposure.clock = runtime.clock
    monkeypatch.setattr(module, '_now', runtime.clock)
    month = cutoff[:7]
    shift = pd.Timestamp(cutoff) - pd.Timestamp('2025-04-01T00:00Z')
    metadata = data.frame.metadata.copy()
    for col in ('available_at', 'forecast_window_start', 'feature_open_time'):
        metadata[col] = metadata[col] + shift
    data = replace(data, frame=replace(data.frame, metadata=metadata), maturity=data.maturity + shift)
    schedule = default_schedule(config, month=month)
    store = RunStore(runtime.directory / 'autonomous/research_run', config, data.spec, exposure_path=runtime.lifecycle.exposure.path,
        feed_identity=runtime.feed.identity, mode='authority_scheduled_research_v1',
        work_contract={'schedule':schedule.to_dict(), 'authority_manifest_hash':authority.manifest_hash})
    store.exposure.clock = runtime.clock  # Same explicit artificial future observation clock.
    exposed = search_exposure_context('controller-data', cutoff, ['X'], runtime.feed.identity)
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
    recipe = FrozenNeuralRecipe(rolling_spec=data.spec, config=ResidualConfig(enabled=True, epochs=4, structures=(StructureSpec(16, 4),)), output='encoder_ridge')
    from factorlab.discovery.s_nonlinear_ticket import freeze_nonlinear_plan
    from factorlab.discovery.scheduled_search import ScheduledSearch
    ScheduledSearch(bridge, schedule)
    bridge.begin_work('baseline', counts={'solve':7}, lease_seconds=300, binding={'stage':'baseline'}).complete(
        {'artificial_owner_setup':True, 'training_performed':False})
    from factorlab.discovery.structural_evolution import select_structure
    if getattr(request, 'param', None) == 'legacy_non_g03':
        freeze_nonlinear_plan(bridge, recipe, kinds=('complete_neural_S',))
    else:
        select_structure(bridge, recipe)
        freeze_nonlinear_plan(bridge, recipe, kinds=('complete_neural_S', 'typed_queue_review'))
    monkeypatch.setattr('factorlab.discovery.neural_research_ticket._input', _private_input)
    gates = {'directory': str(tmp_path / 'artificial-gate-placeholder'),
             'artifacts': {'protocol': 'p', 'probes': ['q1', 'q2'], 'positive_control': 'c'}}
    monkeypatch.setattr('factorlab.discovery.neural_research_ticket._gates', lambda *args: {'private_test_seam': True})
    yield bridge, data, recipe, gates, runtime
    store.close()
    runtime.close()


@pytest.fixture
def completed(case, monkeypatch):
    bridge, data, recipe, gates, runtime = case
    recipe = replace(recipe, config=replace(recipe.config, epochs=4))
    cat = AlgorithmCatalogue.freeze(bridge, (recipe,), rule=AlgorithmShortlistRule(
        minimum_hour_coverage=.001, minimum_symbol_hour_coverage=.001))
    ticket = issue_research_ticket(bridge, data, recipe, gates=gates)
    children = []

    class ActualChild:
        def __init__(self, command, **kwargs):
            self.process = multiprocessing.get_context('spawn').Process(target=_private_child,
                args=(str(runtime.directory.parent), str(ticket.path)))
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
    closed = cat.seal_shortlist()
    bridge.state['status'] = 'complete'
    bridge.state['nonlinear_report'] = {'catalogue': closed}
    bridge.store.save(bridge.month, bridge.state, 'artificial_completed_full_S')
    yield bridge, cat, recipe, runtime, ticket



def _private_input(data, recipe, context, *args):
    # Artificial matrix is intentionally not claimed to be live feed data.
    from factorlab.discovery.nonlinear_orchestration import causal_worker_data
    data = causal_worker_data(data, context['cutoff'])
    assert input_data_identity(data, context['cutoff']) == context['input_id']
    return data


def _private_child(root, path):
    value = json.loads(Path(path).read_text('utf-8'))
    _install_artificial_exposure_clock(pd.Timestamp(value['value']['context']['cutoff'])+pd.Timedelta(days=15))
    import factorlab.discovery.neural_research_ticket as tickets
    tickets._input = _private_input
    _actual_child(root, path)


@pytest.fixture
def queue_module():
    from factorlab.discovery import structural_validation
    return structural_validation


def test_actual_registered_full_S_queues_exact_recipe_future_typed_V_and_restart_no_fee(completed, queue_module):
    bridge, catalogue, recipe, runtime, ticket = completed
    before = bridge.authority.status(ticket.value['lease']['wall_started_at'][:7])
    result = queue_module.queue_structural_validation(runtime, bridge.authority, bridge.store.directory, bridge.month)
    assert result['status'] == 'queued_typed_algorithm_V_only' and result['A0_changed'] is False
    protocol = module.AlgorithmValidation(runtime, bridge.authority, catalogue)
    assert protocol.identity['recipes'] == json.loads(json.dumps({recipe.pipeline_id: recipe.to_dict()}))
    assert pd.Timestamp(result['start']) == pd.Timestamp(bridge.state['cutoff']) + pd.DateOffset(months=1)
    assert protocol.identity['ordinary_factor_route'] is False and protocol.identity['formal_claim'] is False
    assert queue_module.queue_structural_validation(runtime, bridge.authority, bridge.store.directory, bridge.month) == result
    after = bridge.authority.status(ticket.value['lease']['wall_started_at'][:7])
    assert after['counts']['solve'] == before['counts']['solve'] + 1
    assert after['reserved_seconds'] == before['reserved_seconds'] + 300
    assert after['partitions']['nonlinear']['seconds'] == 900
    assert runtime.lifecycle.active_model('A0') is None


def test_missing_shortlist_rejected_not_regenerated(completed, queue_module):
    bridge, catalogue, _, runtime, _ = completed
    (catalogue.directory/'shortlist.json').unlink()
    with pytest.raises((ValueError, OSError)):
        queue_module.queue_structural_validation(runtime, bridge.authority, bridge.store.directory, bridge.month)
    assert not (runtime.directory/'algorithm_validation'/catalogue.digest).exists()


def test_missed_fixed_next_month_does_not_shift_window(completed, queue_module):
    bridge, catalogue, _, runtime, _ = completed
    runtime.clock = lambda: pd.Timestamp(bridge.state['cutoff']) + pd.DateOffset(months=2)
    runtime.lifecycle.clock = runtime.clock  # Same explicit hypothetical execution clock.
    result = queue_module.queue_structural_validation(runtime, bridge.authority, bridge.store.directory, bridge.month)
    assert result['status'] == 'missed_original_next_month'
    assert not (runtime.directory/'algorithm_validation'/catalogue.digest).exists()


def test_published_protocol_before_completion_failure_remains_unusable_no_refund(completed, queue_module, monkeypatch):
    bridge, _catalogue, _, runtime, ticket = completed
    original = module.AlgorithmValidation.freeze
    def publish_then_fail(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError('after exact typed publication')
    monkeypatch.setattr(module.AlgorithmValidation, 'freeze', publish_then_fail)
    with pytest.raises(OSError, match='typed publication'):
        queue_module.queue_structural_validation(runtime, bridge.authority, bridge.store.directory, bridge.month)
    before = bridge.authority.status(ticket.value['lease']['wall_started_at'][:7])
    assert before['partitions']['nonlinear']['seconds'] == 900
    with pytest.raises((ValueError, OSError), match='consumed|Missing|completed original paid queue'):
        queue_module.queue_structural_validation(runtime, bridge.authority, bridge.store.directory, bridge.month)
    with pytest.raises(ValueError, match='completed original paid queue'):
        module.AlgorithmValidation.discover(runtime, bridge.authority)
    assert bridge.authority.status(ticket.value['lease']['wall_started_at'][:7]) == before


def test_another_actual_wallet_cannot_borrow_completed_S(completed, queue_module):
    bridge, _, _, runtime, _ = completed
    other = MonthlyWorkAuthority(bridge.authority.directory.parent/'other-wallet',run_id='other',
        task='volatility',universe=['X'],partition_policy='standard')
    with pytest.raises(ValueError, match='unique monthly work authority'):
        queue_module.queue_structural_validation(runtime, other, bridge.store.directory, bridge.month)
    assert other.verify()['reservations'] == 0


def test_actual_scheduler_persists_typed_route_separately_from_factor_queue(completed, queue_module, monkeypatch):
    from factorlab.discovery import v7_autonomous as scheduler_module
    bridge, catalogue, _, runtime, ticket = completed
    from factorlab.discovery.monthly_schedule import default_schedule
    config = V7Config(**{**bridge.store.identity['config'], 'symbols': tuple(bridge.store.identity['config']['symbols'])})
    scheduler = scheduler_module.ResearchScheduler(runtime,
        config=scheduler_module.AutonomousConfig(research=config,backend='disabled'),
        authority=bridge.authority,schedule=default_schedule(config,month=bridge.month))
    try:
        state = scheduler._plan(pd.Timestamp(bridge.state['cutoff']),pd.Timestamp.now(tz='UTC'))
        # Private dispatcher completion seam; the actual S child already ran.
        # This does not pretend the scheduler launched that fixture child.
        state['status'] = 'complete'
        state['validation'] = {'status':'no_candidate','A0_changed':False}
        scheduler._save(state)
        before = bridge.authority.status(ticket.value['lease']['wall_started_at'][:7])
        scheduler._queue_algorithm(state, object())
        restored = scheduler._load(bridge.month)
        assert restored['validation'] == {'status':'no_candidate','A0_changed':False}
        assert restored['algorithm_validation']['status'] == 'queued_typed_algorithm_V_only'
        assert restored['algorithm_validation']['catalogue_id'] == catalogue.digest
        scheduler._queue_algorithm(restored, object())
        assert scheduler._load(bridge.month) == restored
        after = bridge.authority.status(ticket.value['lease']['wall_started_at'][:7])
        assert after['counts']['solve'] == before['counts']['solve'] + 1
        assert after['reserved_seconds'] == before['reserved_seconds'] + 300
    finally:
        scheduler.close()


def _install_artificial_exposure_clock(now, monkeypatch=None):
    # Only this artificial calendar fixture. The underlying real exposure
    # coverage/role checks remain active in parent and actual spawned child.
    from factorlab.discovery.exposure import ExposureLedger
    original = ExposureLedger.assert_prompt_search
    def checked(self, **kwargs):
        old = self.clock
        self.clock = lambda: now
        try:
            return original(self, **kwargs)
        finally:
            self.clock = old
    if monkeypatch is None:
        ExposureLedger.assert_prompt_search = checked
    else:
        monkeypatch.setattr(ExposureLedger, 'assert_prompt_search', checked)


def test_paid_closure_guards_bytes_sets_and_mirrors_future_reads_never_replay_numeric(completed, queue_module, monkeypatch):
    from factorlab.protocol import fingerprint
    bridge, catalogue, recipe, runtime, _ticket = completed
    result = queue_module.queue_structural_validation(runtime, bridge.authority, bridge.store.directory, bridge.month)
    def unpaid(*args, **kwargs):
        pytest.fail('Completed queue reads must not perform unbudgeted numeric replay')
    monkeypatch.setattr(AlgorithmCatalogue, 'seal_shortlist', unpaid)
    monkeypatch.setattr('factorlab.discovery.algorithm_shortlist.verify_completed_research', unpaid)
    protocol = module.AlgorithmValidation(runtime, bridge.authority, catalogue)
    # Pure guard-role classification, using the real original canonical row.
    # OS process-tree enforcement is independently tested by the guard suite.
    monkeypatch.syspath_prepend(str(ROOT/'scripts/v7_guard'))
    guard_spec = importlib.util.spec_from_file_location('queue_guard_role_test', ROOT/'scripts/v7_guard/multiwork_enforcer.py')
    guard = importlib.util.module_from_spec(guard_spec); guard_spec.loader.exec_module(guard)
    record = bridge.authority.get(result['queue_review_reservation'])
    scope = {'role': 'forward', 'reference': {'kind': 'campaign'}, 'directory': str(runtime.directory),
             'authority': {'directory': str(bridge.authority.directory), 'manifest_hash': bridge.authority.manifest_hash}}
    guard._role(scope, record)
    with pytest.raises(ValueError, match='another Campaign/runtime/wallet'):
        guard._role({**scope, 'directory': str(runtime.directory/'foreign')}, record)
    assert protocol.verify()['status'] == 'frozen'
    assert len(module.AlgorithmValidation.discover(runtime, bridge.authority)) == 1
    assert protocol.fit_plan(recipe.pipeline_id, result['start'])['pipeline_id'] == recipe.pipeline_id
    assert queue_module.queue_structural_validation(runtime, bridge.authority, bridge.store.directory, bridge.month) == result
    # Rehashing a tampered shortlist does not recreate the canonical paid proof.
    path = catalogue.directory/'shortlist.json'; raw = path.read_bytes()
    changed = json.loads(raw); changed['value']['formal_claim'] = True
    changed['hash'] = fingerprint(changed['value']); path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='artifact bytes changed'):
        protocol.verify()
    path.write_bytes(raw)
    mirror = bridge.directory/'numeric_closure.json'; raw_mirror = mirror.read_bytes(); mirror.unlink()
    with pytest.raises(ValueError, match='Missing/bounded'):
        protocol.verify()
    mirror.write_bytes(raw_mirror)
    extra = catalogue.directory/'records'/'unregistered-extra.json'; extra.write_text('{}')
    with pytest.raises(ValueError, match='artifact set changed'):
        protocol.verify()
    extra.unlink()
    assert protocol.verify()['status'] == 'frozen'
    # DELETE-journal DB timestamps independently refuse a late physical write,
    # even with unchanged rows and self-consistent JSON hashes.
    publication = bridge.authority.directory/'typed_queue_reviews'/record['work_key'].split(':', 1)[1]/'publication.sqlite'
    late_ns = pd.Timestamp(record['lease']['wall_deadline']).value + 1000
    os.utime(publication, ns=(late_ns, late_ns))
    with pytest.raises(ValueError, match='final publication IO'):
        protocol.verify()


def test_queue_expiry_after_protocol_write_cannot_authorize_discovery(completed, queue_module, monkeypatch):
    from factorlab.discovery.work_lease import FrozenWorkLease
    bridge, _catalogue, _, runtime, _ticket = completed
    original_freeze = module.AlgorithmValidation.freeze
    original_check = FrozenWorkLease.assert_parent_alive
    target = {'digest': None}
    def check(self, *args, **kwargs):
        if self.digest == target['digest']:
            raise TimeoutError('original queue lease expired')
        return original_check(self, *args, **kwargs)
    monkeypatch.setattr(FrozenWorkLease, 'assert_parent_alive', check)
    def late(*args, **kwargs):
        result = original_freeze(*args, **kwargs)
        with bridge.authority._transaction() as db:
            rows = bridge.authority._verify(db)['reservations'].values()
        target['digest'] = next(r['lease_hash'] for r in rows if r['work_key'].startswith('typed_queue_review:'))
        return result
    monkeypatch.setattr(module.AlgorithmValidation, 'freeze', late)
    with pytest.raises(TimeoutError, match='original queue lease'):
        queue_module.queue_structural_validation(runtime, bridge.authority, bridge.store.directory, bridge.month)
    with bridge.authority._transaction() as db:
        rows = bridge.authority._verify(db)['reservations'].values()
    row = next(r for r in rows if r['work_key'].startswith('typed_queue_review:'))
    assert row['status'] == 'failed_consumed' and row['counts']['solve'] == 1
    before = bridge.authority.status(row['actual_month'])
    with pytest.raises(ValueError, match='completed original paid queue'):
        module.AlgorithmValidation.discover(runtime, bridge.authority)
    assert bridge.authority.status(row['actual_month']) == before


def test_real_sql_commit_crossing_original_deadline_has_no_completed_publication(completed, queue_module, monkeypatch):
    import sqlite3

    from factorlab.discovery import structural_queue_review as queue
    from factorlab.discovery import work_lease
    bridge, _catalogue, _, runtime, _ticket = completed
    original_commit, original_observe = queue._commit_publication, work_lease._observe
    late = {'lease': None, 'committed': False}
    def commit(db):
        row = db.execute('SELECT reservation FROM publication').fetchone()
        record = bridge.authority.get(row[0])
        late['lease'] = work_lease.FrozenWorkLease.from_dict(record['lease'])
        original_commit(db)  # Real completed SQLite transaction; no fake database.
        late['committed'] = True
    def observe(lease):
        result = original_observe(lease)
        if late['committed'] and lease.digest == late['lease'].digest:
            # Private deterministic clock seam models wall+mono spent in COMMIT.
            result['wall'] = pd.Timestamp(lease.wall_deadline).timestamp() + 1
            result['monotonic'] = lease.monotonic_deadline + 1
        return result
    monkeypatch.setattr(queue, '_commit_publication', commit)
    monkeypatch.setattr(work_lease, '_observe', observe)
    with pytest.raises(work_lease.LeaseClosed, match='deadline'):
        queue_module.queue_structural_validation(runtime, bridge.authority, bridge.store.directory, bridge.month)
    assert late['committed']
    directory = bridge.authority.directory/'typed_queue_reviews'/queue._key(runtime.directory, bridge.store.directory, bridge.month)
    with sqlite3.connect(directory/'publication.sqlite') as db:
        assert db.execute('SELECT status FROM publication').fetchone() == ('completed',)
    assert not (directory/'final_io.json').exists()
    with bridge.authority._transaction() as db:
        row = next(r for r in bridge.authority._verify(db)['reservations'].values()
                   if r['work_key'].startswith('typed_queue_review:'))
    assert row['status'] == 'completed'  # Accounting completion alone is not publication permission.
    before = bridge.authority.status(row['actual_month'])
    with pytest.raises(ValueError, match='Missing/bounded'):
        module.AlgorithmValidation.discover(runtime, bridge.authority)
    with pytest.raises(ValueError, match='Missing/bounded'):
        queue_module.queue_structural_validation(runtime, bridge.authority, bridge.store.directory, bridge.month)
    assert bridge.authority.status(row['actual_month']) == before


@pytest.mark.parametrize('case', ['legacy_non_g03'], indirect=True)
def test_non_g03_keeps_actual_catalogue_replay_and_original_validation(completed, monkeypatch):
    bridge, catalogue, recipe, runtime, _ = completed
    assert 'structural_plan' not in bridge.state
    calls = []
    original = AlgorithmCatalogue.seal_shortlist
    def record_replay(self):
        calls.append(True)
        return original(self)
    monkeypatch.setattr(AlgorithmCatalogue, 'seal_shortlist', record_replay)
    start = pd.Timestamp(bridge.state['cutoff']) + pd.DateOffset(months=1)
    protocol = module.AlgorithmValidation.freeze(runtime, bridge.authority, catalogue, start=start)
    assert calls
    before = len(calls)
    protocol.verify()
    assert len(calls) > before
    assert protocol.fit_plan(recipe.pipeline_id, start)['stage'] == 'V'
    assert runtime.lifecycle.db.execute('SELECT COUNT(*) FROM batches').fetchone()[0] == 0
    assert not (bridge.directory/'typed_queue_review.json').exists()


def test_final_io_write_past_deadline_cannot_return_published_success(completed, queue_module, monkeypatch):
    from factorlab.discovery import structural_queue_review as queue
    bridge, _, _, runtime, _ = completed
    original = queue.immutable_record
    def delayed_file(path, value):
        original(path, value)
        if Path(path).name == 'final_io.json':
            record = bridge.authority.get(value['body']['reservation_id'])
            late = pd.Timestamp(record['lease']['wall_deadline']).value + 1_000_000_000
            # Real file stat seam models a final rename/write completed too late.
            os.utime(path, ns=(late, late))
    monkeypatch.setattr(queue, 'immutable_record', delayed_file)
    with pytest.raises(ValueError, match='final publication IO'):
        queue_module.queue_structural_validation(runtime, bridge.authority, bridge.store.directory, bridge.month)
    with bridge.authority._transaction() as db:
        row = next(r for r in bridge.authority._verify(db)['reservations'].values()
                   if r['work_key'].startswith('typed_queue_review:'))
    assert row['status'] == 'completed' and row['counts']['solve'] == 1
    before = bridge.authority.status(row['actual_month'])
    with pytest.raises(ValueError, match='final publication IO'):
        module.AlgorithmValidation.discover(runtime, bridge.authority)
    assert bridge.authority.status(row['actual_month']) == before
