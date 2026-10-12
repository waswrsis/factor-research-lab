"""Real local journal/feed/controller integration; synthetic finite market only.

Use run_combined_overlay.py before promotion. No provider is invoked, and the
tiny 24-hour generator cannot supply a successful three-month fitted learner.
The two ordinary paths use an explicitly artificial healthy host observation;
native clocks, controller verification and wallet accounting stay real. Actual
host collection and fail-closed controller behavior have separate negative
tests in test_v7_service.py and test_controller_native_accounting.py. These
tests do not certify the developer's host health or operating-system limits.
"""
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from factorlab.protocol import fingerprint
from factorlab.discovery import finite_market_oracle as finite
from factorlab.discovery.calibration_acceptance import AcceptancePlan, AcceptanceCell
from factorlab.discovery.calibration_runtime_contract import runtime_calibration_contract
from factorlab.discovery.claim_protocol import ClaimProtocol, FrozenClaimEndpoint
from factorlab.discovery.full_calibration_registry import FullTrialPlan, FullTrialRegistry
from factorlab.discovery.full_trial_controller import FullTrialController, FullControllerSchedule
from factorlab.discovery.known_linear_reference import FrozenLinearRecipe
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
from factorlab.discovery.monthly_schedule import default_schedule
from factorlab.discovery.neural_forward_ticket import _bind_authority
from factorlab.discovery.quarantined_observation import QuarantinedObservation
from factorlab.discovery.v7_autonomous import AutonomousConfig
from factorlab.discovery.v7_forward import ForwardRuntime
from factorlab.discovery.v7_service import ServiceConfig


def test_actual_bound_logical_controller_keeps_physical_wallet_and_no_provider(tmp_path, monkeypatch):
    # Preserve the original short-service boundary: no fresh 300s loan fits.
    _actual_bound_logical_controller(tmp_path, monkeypatch, service_seconds=300, reserved=0)


def test_actual_bound_logical_controller_with_full_service_headroom(tmp_path, monkeypatch):
    # Separately registered full service budget, still one unchanged 300s paid
    # fit and the same monthly wallet. This is not evidence of short-fit success.
    _actual_bound_logical_controller(tmp_path, monkeypatch, service_seconds=7200, reserved=1)


def _actual_bound_logical_controller(tmp_path, monkeypatch, *, service_seconds, reserved):
    import copy
    import json
    from factorlab.discovery.v7_service import HostGuard
    host_observation = {'rss_mib': 0., 'available_mib': 9000., 'processes': []}
    monkeypatch.setattr(HostGuard, '_inspect', staticmethod(lambda: copy.deepcopy(host_observation)))
    (tmp_path / 'artificial-host-boundary.json').write_text(json.dumps({
        'host_observation': host_observation,
        'actual_host_health_or_hard_limits_proved': False,
        'native_clocks_wallets_and_controller_verifiers_replaced': False,
        'service_seconds': service_seconds,
    }, indent=2), encoding='utf8')
    from factorlab import protocol
    import importlib
    repo = Path(protocol.__file__).parent.parent
    sources = {name: (repo/name).read_bytes() for name in finite.PROOF_FILES}
    names = ('full_calibration_registry', 'full_trial_controller', 'full_trial_certificate',
             'v7_autonomous', 'v7_service', 'forward_gate_resolver', 'neural_forward_ticket',
             'quarantined_observation', 'v7_lifecycle', 'forward_fit_snapshot', 'rolling',
             'finite_market_oracle', 'neural_forward_contract', 'nonlinear_runtime', 'forward_service_scope')
    for name in names:
        module = importlib.import_module('factorlab.discovery.'+name)
        sources['factorlab/discovery/'+name+'.py'] = Path(module.__file__).read_bytes()
    monkeypatch.setattr(protocol, 'ROOT', tmp_path)
    (tmp_path/'factorlab').mkdir()
    (tmp_path/'pyproject.toml').write_text('[project]')
    (tmp_path/'uv.lock').write_text('version=1')
    for name, body in sources.items():
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    profile = ForwardRuntime(tmp_path/'profile', tmp_path/'profile-feed')
    authority = MonthlyWorkAuthority(tmp_path/'profile-wallet', run_id='profile',
        task='volatility', universe=['BTCUSDT'], partition_policy='standard')
    _bind_authority(profile, authority)
    scope = runtime_calibration_contract(profile.lifecycle)
    profile.close()
    law = finite.FiniteMarketLaw('2026-01-31T12:00:00+00:00', hours=24, gap_probability=0.)
    acceptance = AcceptancePlan((AcceptanceCell('finite', law.digest, 1000),))
    generator = finite.FiniteGenerator(tmp_path/'generator', law, acceptance)
    registry = FullTrialRegistry(tmp_path/'registry', FullTrialPlan(acceptance, scope,
        {'backend': 'codex', 'model_requested': 'gpt-6-astra', 'reasoning_requested': 'high',
         'context_policy': 'fresh_codex_per_trial_v1'}, law.domains))
    paths = registry.paths('finite', 0, oracle_index=0)
    runtime = ForwardRuntime(paths['runtime'], paths['feed'])
    try:
        trial = registry.identity_for('finite', 0, oracle_index=0)
        wallet = MonthlyWorkAuthority(paths['authority'],
            run_id=fingerprint({'registry': registry.digest, 'trial': trial['trial_id']}),
            task='volatility', universe=['BTCUSDT'], partition_policy='standard')
        permit = registry.enroll(runtime, wallet, 'finite', 0, oracle_index=0)
        observer = QuarantinedObservation.bind(runtime, generator, permit)
        finite.publish_due(generator, permit, max_new_receipts=1)
        first = observer.advance(runtime)
        assert runtime.lifecycle.now() == pd.Timestamp(first['logical_at'])
        assert pd.Timestamp(runtime.clock()) > runtime.lifecycle.now()
        k0 = runtime.recipe_for()
        ClaimProtocol.freeze(runtime.lifecycle, [
            FrozenClaimEndpoint.library('2026-02-01T00:00:00+00:00', k0,
                runtime.recipe_for(FrozenLinearRecipe(rolling_spec=runtime.spec))),
            FrozenClaimEndpoint.algorithm('2026-02-01T00:00:00+00:00', k0)])
        auto = AutonomousConfig(max_attempts=1, not_before_month='2026-02')
        search = default_schedule(auto.research, month='2026-01')
        due = tuple((pd.Timestamp(first['logical_at'])+pd.Timedelta(hours=i)).isoformat() for i in range(2))
        schedule = FullControllerSchedule(due, asdict(ServiceConfig(session_id='bound',
            wall_seconds=service_seconds, max_steps=2, max_fit_attempts=1, poll=False)),
            asdict(auto), search.to_dict(), observation_mode='bound_logical_observation')
        driver = FullTrialController.freeze(permit, runtime, wallet, schedule)
        from factorlab.discovery.forward_service_scope import ServiceDeadlineClosed
        from factorlab.discovery.work_lease import LeaseClosed
        original_exception_init = ServiceDeadlineClosed.__init__
        original_lease_exception_init = LeaseClosed.__init__
        exception_observations = []
        if service_seconds == 300:
            # Capture at the original raise's constructor, before finally can
            # cross the deadline or replace an earlier same-type exception.
            # No clock injection, sleep, result rewriting or exception mutation.
            def observed_exception_init(error, *args, **kwargs):
                import time
                observation = _deadline_sample()
                original_exception_init(error, *args, **kwargs)
                assert len(exception_observations) < 256
                exception_observations.append((error, observation))
            monkeypatch.setattr(ServiceDeadlineClosed, '__init__', observed_exception_init)

            def observed_lease_exception_init(error, *args, **kwargs):
                observation = _deadline_sample()
                original_lease_exception_init(error, *args, **kwargs)
                assert len(exception_observations) < 256
                exception_observations.append((error, observation))
            monkeypatch.setattr(LeaseClosed, '__init__', observed_lease_exception_init)
        try:
            result = driver.pump()
            assert result['state'] == 'awaiting_successor_witness'
            confirmed = driver.pump()
            if confirmed['state'] == 'failed_or_unknown_consumed':
                _assert_short_expired_successor(driver, result, confirmed, paths, wallet, observer,
                                                service_seconds=service_seconds, reserved=reserved)
                return
            assert confirmed['state'] == 'completed_step'
            if result['output']['forward']['status'] == 'blocked_by_host_guard':
                _assert_short_host_guard_rejection(driver, result, paths, wallet, observer,
                                                  service_seconds=service_seconds, reserved=reserved)
                return
            assert result['output']['forward']['status'] == 'stepped'
            assert result['output']['forward']['at'] == first['logical_at']
            assert not list(paths['provider'].glob('**/attempt.json'))
            import sqlite3
            from contextlib import closing
            with closing(sqlite3.connect(paths['runtime']/'autonomous'/'scheduler.sqlite')) as db:
                assert db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == 0
            charged = wallet.verify()
            assert charged['reservations'] == reserved  # no work or actual prepaid failed fit; never synthetic success
            assert driver.step(0) == result and wallet.verify() == charged
            second = driver.pump()
            assert second['state'] == 'awaiting_successor_witness'
            confirmed = driver.pump()
            if confirmed['state'] == 'failed_or_unknown_consumed':
                _assert_short_expired_successor(driver, second, confirmed, paths, wallet, observer,
                                                service_seconds=service_seconds, reserved=reserved)
                return
            assert confirmed['state'] == 'completed_step'
            assert observer.inspect()['events'] == 2
            if second['output']['forward']['status'] == 'blocked_by_host_guard':
                _assert_short_host_guard_rejection(driver, second, paths, wallet, observer,
                                                  service_seconds=service_seconds, reserved=reserved)
                return
            assert second['output']['forward']['status'] == 'stepped'
            assert second['output']['forward']['at'] == due[1]
            assert wallet.verify() == charged
            assert driver.pump()['state'] == 'controller_schedule_exhausted'
            rows = driver.verify()
            assert rows[0]['attempt']['observation_prefix'] == first
            assert all(driver.step_completed(row) for row in rows)
        except (ServiceDeadlineClosed, LeaseClosed) as original_error:
            # Only the independently registered original short-service case is
            # a no-new-fit boundary. Complete two-step execution remains required
            # by the separate 7200s positive above. Slow full audits do not grant
            # more time: verify the actual original deadline was consumed.
            if service_seconds != 300 or reserved != 0:
                raise
            import json
            import sqlite3
            from contextlib import closing
            from datetime import datetime
            clock = driver._execution_clock()
            issuer = clock['lease']['issuer']
            _assert_original_deadline_chain(original_error, exception_observations, issuer)
            native = json.loads((paths['runtime']/'service/bound/native-clock.json').read_text('utf-8'))
            assert native['version'] == 'v7-original-controller-service-clock-v2'
            assert native['issuer'] == issuer and native['spent_before'] == 0
            assert native['service_seconds'] == clock['lease']['service_seconds'] == 300
            rows = driver.verify()
            assert rows and rows[-1]['result']['state'] == 'failed_consumed'
            failed = rows[-1]['result']
            assert failed['error_type'] == type(original_error).__name__
            assert failed['physical_finish']['monotonic'] >= issuer['monotonic_started'] + 300
            assert failed['physical_finish']['wall'] >= datetime.fromisoformat(issuer['wall_started_at']).timestamp() + 300
            for key in ('parent_pid', 'parent_created', 'boot_id'):
                assert failed['physical_finish'][key] == issuer[key]
            assert all(row['attempt']['lease'] == clock['lease'] for row in rows)
            assert not list(paths['provider'].glob('**/attempt.json'))
            assert wallet.verify()['reservations'] == 0
            with closing(sqlite3.connect((paths['runtime']/'autonomous/scheduler.sqlite').as_uri()+'?mode=ro', uri=True)) as db:
                assert db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == 0
            with closing(sqlite3.connect((paths['runtime']/'service/bound/state.sqlite').as_uri()+'?mode=ro', uri=True)) as db:
                checkpoint = json.loads(db.execute('SELECT value FROM checkpoint WHERE id=1').fetchone()[0])
            assert checkpoint['native_service_clock'] == native
            assert checkpoint['spent_seconds'] >= 300
            assert driver._execution_clock() == clock
    finally:
        runtime.close()


def _deadline_sample():
    # The two clocks are sampled first, at the original exception constructor.
    # Later evidence reads cannot retroactively make an early error expired.
    import os
    import time
    import psutil
    from factorlab.discovery.work_lease import _boot_id
    sample = {'wall': time.time(), 'monotonic': time.monotonic()}
    sample.update(pid=os.getpid(), parent_created=psutil.Process().create_time(), boot_id=_boot_id())
    return sample


def _assert_short_expired_successor_records(rows, clock, native, checkpoint, result, confirmed,
                                           *, service_seconds, reserved):
    """Only a timely original dispatch followed by a genuinely expired witness.

    Production's original driver.verify must authenticate files/journal first.
    This checks a negative deadline result; it never credits a completed step.
    """
    from datetime import datetime
    from factorlab.discovery.full_trial_controller import ControllerStepLease, step_completed
    assert service_seconds == 300 and reserved == 0
    assert len(rows) in (1, 2) and rows[-1]['result'] == result
    assert all(step_completed(row) for row in rows[:-1])
    assert not step_completed(rows[-1])
    issuer = clock['lease']['issuer']
    lease = ControllerStepLease(**clock['lease'])
    assert lease.service_seconds == issuer['lease_seconds'] == 300
    assert native['version'] == 'v7-original-controller-service-clock-v2'
    assert native['issuer'] == issuer and native['spent_before'] == 0
    assert native['service_seconds'] == 300
    assert native['controller_binding']['clock_hash'] == fingerprint(clock)
    assert native['controller_binding']['controller_hash'] == clock['controller_hash']
    assert checkpoint['native_service_clock'] == native
    assert checkpoint['steps'] == len(rows) and checkpoint['fit_attempts'] == {}
    assert checkpoint['active_since'] is None and checkpoint['controller_research_active'] is None
    assert checkpoint['last_result'] == result['output']['forward']
    # This checkpoint predates the independent witness pump. Its spent time is
    # only a durable lower bound; the unchanged absolute clock includes the tail.
    assert 0 <= checkpoint['spent_seconds'] <= lease.elapsed(result['physical_finish']) < 300
    witness = rows[-1]['witness']
    assert witness['state'] == 'failed_or_unknown_consumed'
    assert confirmed == {'state':'failed_or_unknown_consumed', 'index':len(rows)-1,
                         'witness':witness, 'output':result['output'], 'formal_gate_eligible':False}
    assert witness['attempt_hash'] == fingerprint(rows[-1]['attempt'])
    assert witness['result_hash'] == fingerprint(result)
    check = lease.evaluate(witness['observed'], result['physical_finish'])
    assert not check.allowed and check.reason == 'deadline'
    assert witness['observed']['monotonic'] >= issuer['monotonic_started'] + 300
    assert witness['observed']['wall'] >= datetime.fromisoformat(issuer['wall_started_at']).timestamp() + 300
    previous = '0'*64
    for index, row in enumerate(rows):
        attempt, pending = row['attempt'], row['result']
        assert attempt['index'] == index and attempt['previous'] == previous
        assert attempt['lease'] == clock['lease'] and attempt['clock_hash'] == fingerprint(clock)
        assert set(pending) == {'attempt_hash', 'state', 'physical_finish', 'output', 'formal_gate_eligible'}
        assert set(pending['output']) == {'forward', 'research', 'research_terminal'}
        assert pending['state'] == 'awaiting_successor_witness' and pending['formal_gate_eligible'] is False
        assert pending['attempt_hash'] == fingerprint(attempt)
        assert lease.evaluate(pending['physical_finish'], attempt['physical_start']).allowed
        forward = pending['output']['forward']
        assert forward['status'] == 'stepped' and forward['at'] == attempt['due_at']
        assert forward['guard']['allowed'] is True and forward['guard']['reasons'] == []
        assert [a for a in forward['actions'] if a['kind'] == 'fit'] == [
            {'kind':'fit', 'status':'insufficient_service_headroom', 'stream':'A0'}]
        assert pending['output']['research']['status'] == 'waiting_not_before'
        assert pending['output']['research']['actions'] == []
        assert pending['output']['research_terminal'] == {'actions':[], 'status':'settled'}
        for observation in (attempt['physical_start'], pending['physical_finish'], row['witness']['observed']):
            for key in ('parent_pid', 'parent_created', 'boot_id'):
                assert observation[key] == issuer[key]
            assert observation['pid'] == issuer['parent_pid'] and observation['parent_alive'] is True
        previous = fingerprint(row)


def _assert_short_expired_successor(driver, result, confirmed, paths, wallet, observer,
                                   *, service_seconds, reserved):
    import json
    import sqlite3
    from contextlib import closing
    assert service_seconds == 300 and reserved == 0
    clock = driver._execution_clock()
    rows = driver.verify()  # Original mirrored files, observation and LC anchors.
    native = json.loads((paths['runtime']/'service/bound/native-clock.json').read_text('utf-8'))
    with closing(sqlite3.connect((paths['runtime']/'service/bound/state.sqlite').as_uri()+'?mode=ro', uri=True)) as db:
        text, digest = db.execute('SELECT value,hash FROM checkpoint WHERE id=1').fetchone()
        checkpoint = json.loads(text)
        assert fingerprint(checkpoint) == digest
        previous, last = '0'*64, None
        for sequence, text, recorded_hash in db.execute('SELECT seq,value,hash FROM events ORDER BY seq'):
            event = json.loads(text)
            assert event['sequence'] == sequence and event['previous'] == previous
            assert fingerprint(event) == recorded_hash
            previous, last = recorded_hash, event['state_hash']
        assert last == digest
    _assert_short_expired_successor_records(rows, clock, native, checkpoint, result, confirmed,
                                          service_seconds=service_seconds, reserved=reserved)
    assert not list(paths['provider'].glob('**/attempt.json'))
    charged = wallet.verify()
    assert charged['reservations'] == 0
    with closing(sqlite3.connect((paths['authority']/'authority.sqlite').as_uri()+'?mode=ro', uri=True)) as db:
        for table in ('wallets', 'reservations', 'partition_wallets', 'monthly_owners', 'events'):
            assert db.execute('SELECT COUNT(*) FROM '+table).fetchone()[0] == 0
    with closing(sqlite3.connect((paths['runtime']/'autonomous/scheduler.sqlite').as_uri()+'?mode=ro', uri=True)) as db:
        assert db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == 0
    assert observer.inspect()['events'] == len(rows)
    assert driver.step(len(rows)-1) == result and wallet.verify() == charged
    assert driver._execution_clock() == clock


def _assert_original_deadline_chain(original_error, observations, issuer):
    from datetime import datetime
    from factorlab.discovery.forward_service_scope import ServiceDeadlineClosed
    from factorlab.discovery.work_lease import LeaseClosed
    pending, visited, examined = [original_error], set(), []
    while pending:
        error = pending.pop()
        if id(error) in visited:
            continue
        visited.add(id(error))
        assert len(visited) <= 256
        assert type(error) in (ServiceDeadlineClosed, LeaseClosed), 'Non-deadline failure cannot be hidden by late cleanup'
        if type(error) is LeaseClosed:
            assert error.check.reason == 'deadline' and not error.check.allowed
        samples = [sample for observed_error, sample in observations if observed_error is error]
        assert len(samples) == 1
        sample = samples[0]
        assert sample['monotonic'] >= issuer['monotonic_started'] + 300
        assert sample['wall'] >= datetime.fromisoformat(issuer['wall_started_at']).timestamp() + 300
        assert sample['pid'] == issuer['parent_pid']
        assert sample['parent_created'] == issuer['parent_created']
        assert sample['boot_id'] == issuer['boot_id']
        examined.append(sample)
        pending.extend(e for e in (error.__context__, error.__cause__) if e is not None)
    assert examined


def _assert_short_host_guard_rejection(driver, result, paths, wallet, observer, *, service_seconds, reserved):
    # Only the original short no-new-fit case can establish this negative.
    # The independent7200s positive still requires both real stepped results.
    assert service_seconds == 300 and reserved == 0
    forward = result['output']['forward']
    assert set(forward) == {'status', 'guard'} and forward['status'] == 'blocked_by_host_guard'
    guard = forward['guard']
    assert guard['allowed'] is False and guard['reasons'] == ['rss_limit']
    assert guard['rss_mib'] >= 768 and guard['available_mib'] >= 768
    assert guard['blocking_pids'] == [] and guard['hard_limits_verified'] is False
    for key in ('research', 'research_terminal'):
        assert result['output'][key]['status'] == 'stopped'
        assert result['output'][key]['reason'] == 'host_guard'
    assert not list(paths['provider'].glob('**/attempt.json'))
    assert wallet.verify()['reservations'] == 0
    import json
    import sqlite3
    from contextlib import closing
    with closing(sqlite3.connect((paths['runtime']/'autonomous/scheduler.sqlite').as_uri()+'?mode=ro', uri=True)) as db:
        assert db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == 0
    clock = driver._execution_clock()
    issuer = clock['lease']['issuer']
    native = json.loads((paths['runtime']/'service/bound/native-clock.json').read_text('utf-8'))
    assert native['version'] == 'v7-original-controller-service-clock-v2'
    assert native['issuer'] == issuer and native['spent_before'] == 0
    assert native['service_seconds'] == clock['lease']['service_seconds'] == 300
    rows = driver.verify()
    assert rows and rows[-1]['result'] == result
    assert all(driver.step_completed(row) for row in rows)
    assert all(row['attempt']['lease'] == clock['lease'] for row in rows)
    assert observer.inspect()['events'] == len(rows)
    for key in ('parent_pid', 'parent_created', 'boot_id'):
        assert result['physical_finish'][key] == issuer[key]
    assert driver.step(len(rows)-1) == result and wallet.verify()['reservations'] == 0
    assert driver._execution_clock() == clock
