"""Staged real-wallet tests; no numerical fits, market, GPT or network calls."""
import multiprocessing
from datetime import UTC, datetime
from pathlib import Path

import pytest

from factorlab.discovery.forward_preparation import ForwardPreparation
from factorlab.discovery.known_linear_reference import FrozenLinearRecipe
from factorlab.discovery.monthly_partition_policy import frozen_policy
from factorlab.discovery.monthly_work_authority import BudgetExhausted, MonthlyWorkAuthority
from factorlab.discovery.neural_forward_ticket import open_existing_authority
from factorlab.discovery.rolling import RollingSpec
from factorlab.discovery.v7_forward import ForwardConfig, ForwardRuntime
from factorlab.discovery.v7_memory import search_exposure_context
from factorlab.discovery.v7_run import RunStore, V7Config, monthly_budget
from factorlab.discovery.work_authority_bridge import WorkAuthorityBridge
from factorlab.protocol import fingerprint


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    (tmp_path/'factorlab').mkdir()
    (tmp_path/'factorlab/frozen.py').write_text('version=1')
    (tmp_path/'pyproject.toml').write_text('[project]')
    (tmp_path/'uv.lock').write_text('version=1')
    return tmp_path


def wallet(root, mode='standard'):
    return MonthlyWorkAuthority(root/'authority', run_id='campaign', task='volatility',
                                universe=['BTCUSDT'], partition_policy=mode)


def search(root, authority, name='S', month='2025-04'):
    config = V7Config(evaluation_mode='joint')
    store = RunStore(root/name, config, RollingSpec())
    exposed = search_exposure_context('input-'+name, month+'-01T00:00:00+00:00', ['BTCUSDT'])
    store.exposure.record(**{k: v for k, v in exposed.items() if k != 'data_id'}, role='S')
    state = {'status': 'running', 'data_id': exposed['data_id'], 'search_exposure': exposed,
             'budget': monthly_budget(config, month).to_dict(), 'cutoff': month+'-01T00:00:00+00:00'}
    store.save(month, state, 'original')
    def persist(value):
        state['budget'] = value
        store.save(month, state, 'budget_projection')
    return WorkAuthorityBridge(store, month, state, authority, persist)


def begin(bridge, key, n=1, seconds=1):
    return bridge.begin_work(key, counts={'solve': n}, lease_seconds=seconds,
                             binding={'stage': 'candidate', 'input_id': bridge.state['data_id']})


def month():
    return datetime.now(UTC).strftime('%Y-%m')


def test_real_second_RunStore_cannot_claim_same_actual_month_owner(root):
    a = wallet(root)
    first, other = search(root, a, 'first'), search(root, a, 'other')
    try:
        begin(first, 'candidate').complete({'empty_test_work': True})
        before = a.status(month())
        with pytest.raises(ValueError, match='Another actual owner'):
            begin(other, 'different-candidate')
        assert a.status(month()) == before
        assert before['owners']['S']['run_directory'] == str(root/'first')
        assert month() != first.month
    finally:
        first.store.close(); other.store.close()


def test_different_replay_month_is_not_new_actual_month_allowance(root):
    a = wallet(root)
    first, other = search(root, a, 'first', '2025-04'), search(root, a, 'other', '2025-05')
    try:
        begin(first, 'candidate').complete({'no_fit': True})
        with pytest.raises(ValueError, match='Another actual owner'):
            begin(other, 'same-execution-month')
        assert a.status(month())['counts']['solve'] == 1
    finally:
        first.store.close(); other.store.close()


def test_real_bridge_alone_cannot_label_arbitrary_solve_as_nonlinear(root):
    a = wallet(root)
    b = search(root, a)
    try:
        begin(b, 'baseline').complete({'no_fit': True})
        binding = {'ticket_terms': fingerprint('self-asserted'), 'research_bridge_id': b.identifier,
                   'role': 'S', 'input_id': 'claimed', 'pipeline_id': fingerprint('claimed'),
                   'serving_eligible': False}
        before = a.status(month())
        with pytest.raises(ValueError, match='actual frozen controller plan'):
            a.reserve('neural_S:'+fingerprint('claimed-work'), counts={'solve': 1}, binding=binding, lease_seconds=1)
        assert a.status(month()) == before
    finally:
        b.store.close()


def test_historical_completed_owner_readback_does_not_renew_old_lease(root, monkeypatch):
    from factorlab.discovery.work_lease import FrozenWorkLease
    a = wallet(root)
    b = search(root, a)
    try:
        record = begin(b, 'complete').complete({'no_fit': True})
        monkeypatch.setattr(FrozenWorkLease, 'assert_parent_alive',
                            lambda *a, **kw: (_ for _ in ()).throw(ValueError('old issuer gone')))
        assert a.get(record['reservation_id']) == record
        assert a.recover(record['reservation_id']) == record
        assert a.status(month())['counts']['solve'] == 1
    finally:
        b.store.close()


def test_S_cannot_consume_forward_or_nonlinear_holdbacks(root):
    a = wallet(root)
    b = search(root, a)
    try:
        begin(b, 'full-S-ceiling', n=943).complete({'no_fit_performed': True})
        before = a.status(month())
        with pytest.raises(BudgetExhausted, match='Nontransferable monthly S'):
            begin(b, 'cannot-borrow', n=1)
        assert a.status(month()) == before
        assert before['counts']['solve'] == 943 < before['limits']['solve']
        assert before['partitions']['S']['counts']['solve'] == 943
        with pytest.raises(ValueError, match='Unknown work provenance'):
            a.reserve('fake-forward', counts={'solve': 1}, binding={'role': 'forward'}, lease_seconds=1)
    finally:
        b.store.close()


def test_failure_consumes_partition_and_reopen_cannot_change_policy(root):
    a = wallet(root)
    b = search(root, a)
    try:
        work = begin(b, 'prepaid', n=10, seconds=30)
        original = work.fail('artificial interruption')
        assert original['status'] == 'failed_consumed'
        reopened = open_existing_authority(a.directory)
        before = reopened.status(month())
        assert before['partitions']['S']['counts']['solve'] == 10
        assert before['partitions']['S']['seconds'] == 30
        assert reopened.recover(original['reservation_id']) == original
        assert reopened.status(month()) == before
        with pytest.raises(ValueError, match='identity changed'):
            wallet(root, 'controls')
    finally:
        b.store.close()


def forward(root, a):
    runtime = ForwardRuntime(root/'forward', root/'feed', config=ForwardConfig(symbols=('BTCUSDT',)))
    recipe = FrozenLinearRecipe(rolling_spec=runtime.spec)
    cutoff = datetime.now(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
    return runtime, recipe, cutoff


def test_real_forward_still_prepaid_after_S_partition_spent(root):
    a = wallet(root)
    b = search(root, a)
    runtime, recipe, cutoff = forward(root, a)
    try:
        begin(b, 'all-S', n=943).complete({'no_fit_performed': True})
        job = runtime.lifecycle.plan_fit(cutoff, 'known:HAR_cal', runtime.recipe_for(recipe))
        prep = ForwardPreparation.begin(runtime, a, job, recipe)
        r = prep.record
        assert r['allocation']['lane'] == 'forward'
        status = a.status(month())
        assert status['counts']['solve'] == 944
        assert status['partitions']['forward']['counts']['solve'] == 1
        assert status['partitions']['S']['counts']['solve'] == 943
    finally:
        b.store.close(); runtime.close()


def _race(root, job, ready, go, result, preparation_lock):
    import factorlab.protocol
    factorlab.protocol.ROOT = Path(root)
    a = open_existing_authority(Path(root)/'authority')
    runtime, recipe, _ = forward(Path(root), a)
    # Serialize the unrelated zero-wait lifecycle binding lock, then race only
    # the two real authority transactions after BOTH original intents exist.
    original_reserve = a._reserve_with_lease
    released = False
    def reserve(*args, **kwargs):
        nonlocal released
        preparation_lock.release()
        released = True
        ready.put('ready')
        go.wait(15)
        return original_reserve(*args, **kwargs)
    a._reserve_with_lease = reserve
    preparation_lock.acquire()
    try:
        r = ForwardPreparation.begin(runtime, a, job, recipe).record
        result.put(('reserved', r['reservation_id']))
    except BudgetExhausted:
        result.put(('quota', None))
    except BaseException as exc:  # noqa: BLE001 -- return child failure to the assertion, never mask it.
        result.put(('unexpected', repr(exc)))
    finally:
        if not released:
            preparation_lock.release()
        runtime.close()


def test_real_multiprocess_race_only_one_last_forward_work_fits(root):
    a = wallet(root)
    runtime, recipe, cutoff = forward(root, a)
    try:
        jobs = [runtime.lifecycle.plan_fit(cutoff, f'known:bounded-probe-{i}', runtime.recipe_for(recipe)) for i in range(10)]
        for job in jobs[:8]:
            r = ForwardPreparation.begin(runtime, a, job, recipe).record
            a.fail_consumed(r['reservation_id'], 'prepaid allocation test, no fitting')
        assert a.status(month())['partitions']['forward']['seconds'] == 2400
        ctx = multiprocessing.get_context('spawn')
        ready, result, go, preparation_lock = ctx.Queue(), ctx.Queue(), ctx.Event(), ctx.Lock()
        workers = [ctx.Process(target=_race, args=(str(root), job, ready, go, result, preparation_lock)) for job in jobs[8:]]
        for worker in workers: worker.start()
        for _ in workers: assert ready.get(timeout=20) == 'ready'
        go.set()
        outcomes = [result.get(timeout=30) for _ in workers]
        for worker in workers:
            worker.join(10)
            assert worker.exitcode == 0
        assert sorted(v[0] for v in outcomes) == ['quota', 'reserved'], outcomes
        status = a.status(month())
        assert status['partitions']['forward']['seconds'] == 2700
        assert status['partitions']['forward']['counts']['solve'] == 9
        assert a.verify()['reservations'] == 9
    finally:
        runtime.close()


@pytest.mark.parametrize('damage', ['partition', 'owner', 'database'])
def test_missing_projection_or_wallet_does_not_reset(root, damage):
    import sqlite3
    a = wallet(root)
    b = search(root, a)
    try:
        begin(b, 'charged').complete({'no_fit_performed': True})
        if damage == 'database':
            a.database.unlink()
        else:
            with sqlite3.connect(a.database) as db:
                db.execute('DELETE FROM '+('partition_wallets' if damage == 'partition' else 'monthly_owners'))
        with pytest.raises((ValueError, sqlite3.OperationalError)):
            open_existing_authority(a.directory)
    finally:
        b.store.close()


def test_policy_exact_sum_no_transfer_and_changed_source_rejected(root):
    for mode in ('standard', 'controls'):
        p = frozen_policy(mode)
        assert sum(v['counts']['solve'] for v in p['lanes'].values()) == 1600
        assert sum(v['seconds'] for v in p['lanes'].values()) == 7200
        assert sum(v['counts']['neural_fit'] for v in p['lanes'].values()) == 18
        assert sum(v['counts']['epoch'] for v in p['lanes'].values()) == 180
    a = wallet(root)
    (root/'factorlab/frozen.py').write_text('version=changed')
    with pytest.raises(ValueError, match='source'):
        a.verify()


def test_resource_and_full_S_debits_share_single_nonlinear_ceiling(root):
    """Planned accounting, not model evidence: no gate/training result is claimed."""
    a = wallet(root)
    b = search(root, a)
    try:
        from factorlab.discovery.monthly_schedule import default_schedule
        from factorlab.discovery.neural_forward import FrozenNeuralRecipe
        from factorlab.discovery.s_nonlinear_ticket import freeze_nonlinear_plan
        from factorlab.discovery.scheduled_search import ScheduledSearch
        recipe = FrozenNeuralRecipe()
        b.state['nonlinear_input_data_id'] = fingerprint('planned-input-no-fit')
        b.store.save(b.month, b.state, 'planned_input')
        ScheduledSearch(b, default_schedule(V7Config(evaluation_mode='joint'), month=b.month))
        plan = freeze_nonlinear_plan(b, recipe, kinds=('resource_trial', 'complete_neural_S'))
        begin(b, 'baseline').complete({'no_fit_performed': True})
        base = {'role': 'S', 'nonlinear_bridge_id': b.identifier, 'kind': 'resource_trial',
                'plan_hash': plan['hash'], 'pipeline_id': recipe.pipeline_id,
                'input_id': b.state['nonlinear_input_data_id'], 'serving_eligible': False,
                'S_nonlinear_terms': fingerprint('planned-resource-terms')}
        resource_key = fingerprint({'bridge': fingerprint(b.identity), 'kind': 'resource_trial'})
        capacities = {w['kind']: w['counts'] for w in plan['body']['work']}
        resource = a.reserve('S_nonlinear:'+resource_key, counts=capacities['resource_trial'],
                             binding=base, lease_seconds=300)
        a.fail_consumed(resource['reservation_id'], 'bounded planned accounting only')
        research_binding = {'ticket_terms': fingerprint('full-S-terms'), 'research_bridge_id': b.identifier,
                            'role': 'S', 'input_id': b.state['nonlinear_input_data_id'], 'pipeline_id': base['pipeline_id'],
                            'serving_eligible': False}
        full_key = fingerprint({'bridge': fingerprint(b.identity), 'controller_data': b.state['data_id'],
                                'input': b.state['nonlinear_input_data_id'], 'pipeline': recipe.pipeline_id})
        full = a.reserve('neural_S:'+full_key, counts=capacities['complete_neural_S'],
                         binding=research_binding, lease_seconds=300)
        a.fail_consumed(full['reservation_id'], 'bounded planned accounting only')
        status = a.status(month())
        assert status['partitions']['nonlinear']['counts']['neural_fit'] == 12
        assert status['partitions']['nonlinear']['counts']['epoch'] == 48
        with pytest.raises(ValueError, match='frozen recipe/work'):
            a.reserve('neural_S:'+fingerprint('extra'), counts={'neural_fit': 1},
                      binding=research_binding, lease_seconds=1)
        assert a.status(month()) == status
        assert status['counts']['neural_fit'] < status['limits']['neural_fit']
    finally:
        b.store.close()


def test_commit_then_missing_mirror_recovers_without_refund_or_reissue(root, monkeypatch):
    import factorlab.discovery.monthly_work_authority as module
    a = wallet(root)
    b = search(root, a)
    original = module.immutable_record
    def crash(path, value):
        if path.parent == a.directory/'events':
            raise OSError('after committed transaction before mirror')
        return original(path, value)
    monkeypatch.setattr(module, 'immutable_record', crash)
    with pytest.raises(OSError, match='after committed'):
        begin(b, 'original-charge', n=5, seconds=30)
    monkeypatch.setattr(module, 'immutable_record', original)
    try:
        reopened = wallet(root)
        status = reopened.status(month())
        assert status['counts']['solve'] == status['partitions']['S']['counts']['solve'] == 5
        assert status['reserved_seconds'] == 30
        assert reopened.verify()['reservations'] == 1
        with pytest.raises(ValueError, match='closed|readback|prepaid'):
            begin(b, 'original-charge', n=5, seconds=30)
        assert reopened.status(month()) == status
    finally:
        b.store.close()


def test_shared_allocation_table_cannot_claim_guarantee_on_unpartitioned_wallet(root):
    import dataclasses

    from factorlab.discovery.scheduled_search import FrozenSWorkSchedule, ScheduledSearch
    a = MonthlyWorkAuthority(root/'authority', run_id='unpartitioned-diagnostic', task='volatility', universe=['BTCUSDT'])
    b = search(root, a)
    try:
        schedule = FrozenSWorkSchedule.freeze(config=dataclasses.asdict(V7Config(evaluation_mode='joint')),
            limits=b.identity['limits'], slots=[{'key': 'baseline', 'stage': 'baseline', 'counts': {'solve': 7}, 'seconds': 300}],
            allocation={'version': 'v7-shared-monthly-allocation-20261005-v1'})
        with pytest.raises(ValueError, match='authoritative monthly partition'):
            ScheduledSearch(b, schedule)
        assert a.verify()['reservations'] == 0
    finally:
        b.store.close()


def test_real_controls_executor_three_owners_one_group_no_second_experiment(root, monkeypatch):
    """Actual executor/preparation/RunStore; private no-fit monthly callback."""
    import pandas as pd

    from factorlab.discovery import shared_controls_executor as controls
    from factorlab.discovery.monthly_schedule import default_control_schedules
    from factorlab.discovery.neural_forward_ticket import _bind_authority
    a = wallet(root, 'controls')
    runtime, _, cutoff = forward(root, a)
    _bind_authority(runtime, a)
    schedules = default_control_schedules(symbols=['BTCUSDT'], month=cutoff[:7])
    path = root/'artificial-preparation-only.parquet'
    pd.DataFrame({'synthetic': [1., 2., 3.]}).to_parquet(path)
    saved = []
    def no_fit_month(directory, bars, replay_cutoff, *, config, spec, authority,
                     exposure_path, feed_directory, schedule, preparation, **kwargs):
        assert len(bars) == 3 and authority is a
        preparation.start_preprocessing()
        store = RunStore(directory, config, spec, mode='authority_scheduled_research_v1',
            exposure_path=exposure_path, feed_identity=runtime.feed.identity,
            work_contract={'schedule': schedule.to_dict(), 'authority_manifest_hash': a.manifest_hash})
        data_id = fingerprint({'synthetic_preparation_input': bars['synthetic'].tolist()})
        exposed = search_exposure_context(data_id, replay_cutoff, config.symbols, feed_identity=runtime.feed.identity)
        store.exposure.record(**{k: v for k, v in exposed.items() if k != 'data_id'}, role='S')
        state = {'status': 'running', 'data_id': data_id, 'search_exposure': exposed, 'cutoff': replay_cutoff,
                 'budget': monthly_budget(config, replay_cutoff[:7]).to_dict()}
        store.save(replay_cutoff[:7], state, 'artificial_input_no_fit')
        preparation.seal_input(data_id=data_id, search_exposure=exposed)
        def persist(value):
            state['budget'] = value
            store.save(replay_cutoff[:7], state, 'projected_budget')
        try:
            b = WorkAuthorityBridge(store, replay_cutoff[:7], state, a, persist, preparation=preparation)
            work = b.begin_work('baseline', counts={'solve': 7}, lease_seconds=300,
                                binding={'stage': 'baseline', 'input_id': data_id})
            work.complete({'synthetic_preparation_only': True})
            saved.append(b.identifier)
            return {'month': replay_cutoff[:7], 'status': 'artificial_no_fit', 'actual_counts': b.actual_counts}
        finally:
            store.close()
    monkeypatch.setattr(controls, 'run_month', no_fit_month)
    try:
        first = controls.SharedControlsExecutor(root/'comparison', authority=a, schedules=schedules,
            feed_directory=runtime.feed.directory, exposure_path=runtime.lifecycle.exposure.path,
            cutoff=cutoff, spec=runtime.spec)
        first.run_file(path, backend=None)
        status = a.status(month())
        assert len(saved) == 3
        assert status['counts']['solve'] == 21
        assert {k for k in status['owners'] if k.startswith('S:')} == {'S:local', 'S:random', 'S:gpt'}
        assert status['owners']['controls_group'] == fingerprint(first.identity)
        assert all(status['partitions']['S:'+arm]['counts']['solve'] == 7 for arm in controls.ARMS)
        assert a.manifest['partition_policy']['lanes']['nonlinear']['counts']['neural_fit'] == 12
        second = controls.SharedControlsExecutor(root/'second-comparison', authority=a, schedules=schedules,
            feed_directory=runtime.feed.directory, exposure_path=runtime.lifecycle.exposure.path,
            cutoff=cutoff, spec=runtime.spec)
        with pytest.raises(ValueError, match='Another actual owner'):
            second.run_file(path, backend=None)
        assert a.status(month()) == status
    finally:
        runtime.close()


def _default_policy_constructor(monkeypatch):
    original = MonthlyWorkAuthority.__init__
    def initialize(self, *args, **kwargs):
        kwargs.setdefault('partition_policy', 'standard')
        original(self, *args, **kwargs)
    monkeypatch.setattr(MonthlyWorkAuthority, '__init__', initialize)


def test_genuine_neural_ticket_exact_original_intent_and_no_relabel(tmp_path, monkeypatch):
    from test_v7_neural_forward_launcher import real_ticket_case
    _default_policy_constructor(monkeypatch)
    fixture = real_ticket_case.__wrapped__(tmp_path, monkeypatch)
    case = next(fixture)
    try:
        ticket, a = case.ticket, case.authority
        record = a.get(ticket.value['reservation_id'])
        assert record['allocation']['lane'] == 'forward'
        assert record['counts']['neural_fit'] == 3
        original = a.status(record['actual_month'])
        from factorlab.discovery.work_lease import FrozenWorkLease
        lease = FrozenWorkLease.from_dict(record['lease'])
        for change in ('counts', 'key', 'lease', 'binding'):
            key, debit, binding, owned_lease = record['work_key'], record['counts'], record['binding'], lease
            if change == 'counts': debit = {**debit, 'epoch': debit['epoch']-1}
            if change == 'key': key = 'neural_forward:'+fingerprint('renamed')
            if change == 'lease': owned_lease = FrozenWorkLease.issue(300)
            if change == 'binding': binding = {**binding, 'pipeline_id': fingerprint('fake')}
            with pytest.raises(ValueError):
                a._reserve_with_lease(key, counts=debit, binding=binding, lease=owned_lease)
        assert a.status(record['actual_month']) == original
        a.fail_consumed(record['reservation_id'], 'artificial_no_fit')
        (ticket.path.parent/'allocation.json').unlink()
        with pytest.raises(ValueError, match='Missing or oversized forward ticket evidence'):
            a.verify()
    finally:
        fixture.close()


def test_actual_completed_S_catalogue_prepares_algorithm_V_same_partitioned_wallet(tmp_path, monkeypatch):
    """Real three-fold artificial S child; future V preparation only, private gates/clock."""
    import pandas as pd
    from test_v7_algorithm_validation import case as case_fixture
    from test_v7_algorithm_validation import completed as completed_fixture
    from test_v7_algorithm_validation_producer import _future_clock, _seed

    from factorlab.discovery import algorithm_validation as validation
    from factorlab.discovery.monthly_schedule import default_schedule
    from factorlab.discovery.s_nonlinear_ticket import freeze_nonlinear_plan
    from factorlab.discovery.scheduled_search import ScheduledSearch
    _default_policy_constructor(monkeypatch)
    fixture = case_fixture.__wrapped__(tmp_path, monkeypatch)
    case = next(fixture)
    b, _, recipe, _, runtime = case
    schedule = default_schedule(V7Config(**b.store.identity['config']), month=b.month)
    ScheduledSearch(b, schedule)
    # The shared fixture already consumed the original baseline once.
    # Reuse its receipt rather than creating an illegal second execution.
    with b.authority._transaction() as db:
        baseline = [r for r in b.authority._verify(db)['reservations'].values()
                    if r['work_key'].endswith(':baseline')]
    assert len(baseline) == 1 and baseline[0]['status'] == 'completed'
    freeze_nonlinear_plan(b, recipe, kinds=('complete_neural_S',))
    completed = completed_fixture.__wrapped__(case, monkeypatch)
    try:
        _, catalogue, recipe, _, _ = next(completed)
        start = validation._now().normalize().replace(day=1)+pd.DateOffset(months=1)
        protocol = validation.AlgorithmValidation.freeze(runtime, b.authority, catalogue, start=start)
        clock, _ = _future_clock(monkeypatch, runtime, start)
        _seed(runtime, clock, start)
        monkeypatch.setattr(validation._producer(), 'verify_gates', lambda *args, **kwargs: None)
        prepared = protocol.prepare_fit(recipe.pipeline_id, start, gates={'directory': 'private', 'artifacts': {}})
        record = b.authority.get(prepared['reservation_id'])
        assert record['allocation']['lane'] == 'forward'
        assert record['counts'] == protocol.identity['work_counts']
        assert record['counts']['neural_fit'] == 3
        before = b.authority.status(record['actual_month'])
        with pytest.raises(ValueError):
            b.authority.reserve('algorithm_fit:'+fingerprint('renamed'), counts=record['counts'],
                binding=record['binding'], lease_seconds=300)
        assert b.authority.status(record['actual_month']) == before
        b.authority.fail_consumed(record['reservation_id'], 'artificial_V_preparation_only')
        path = b.authority.directory/'algorithm_work_intents'/(fingerprint(record['work_key'])+'.json')
        path.unlink()
        with pytest.raises(ValueError, match='allocation evidence'):
            b.authority.verify()
    finally:
        completed.close()
        fixture.close()


def test_public_schedule_guard_rejects_before_first_baseline_debit(root):
    from factorlab.discovery.monthly_schedule import default_schedule
    from factorlab.discovery.scheduled_search import validate_schedule_authority
    config = V7Config(evaluation_mode='joint', symbols=('BTCUSDT',))
    schedule = default_schedule(config, month='2025-04')
    legacy = MonthlyWorkAuthority(root/'legacy', run_id='diagnostic', task='volatility', universe=['BTCUSDT'])
    with pytest.raises(ValueError, match='authoritative monthly partition'):
        validate_schedule_authority(legacy, schedule)
    assert legacy.verify()['reservations'] == 0
    a = wallet(root)
    assert validate_schedule_authority(a, schedule) == schedule.to_dict()
    assert a.verify()['reservations'] == 0
