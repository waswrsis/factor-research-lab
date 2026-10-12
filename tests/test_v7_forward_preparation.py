"""Real local wallets/receipts, artificial candles; no training or Linux claim."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pandas as pd
import pytest

from factorlab.discovery import forward_preparation as module
from factorlab.discovery.forward_preparation import ForwardPreparation
from factorlab.discovery.known_linear_reference import FrozenLinearRecipe
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
from factorlab.discovery.v7_forward import ForwardRuntime
from factorlab.discovery.work_lease import FrozenWorkLease
from factorlab.protocol import fingerprint


@pytest.fixture
def case(tmp_path, monkeypatch):
    from test_v7_feed_audit import Clock
    from test_v7_rolling_registry import candles
    from test_v7_run_factor_pool import source_root
    source_root(tmp_path, monkeypatch)
    clock = Clock('2026-09-30T23:30:00Z')
    runtime = ForwardRuntime(tmp_path / 'run', tmp_path / 'feed', clock=clock)
    try:
        path = tmp_path / 'artificial.parquet'
        candles('2026-05-01T00:00Z', '2026-09-30T23:00Z').to_parquet(path)
        runtime.feed.import_seed(path)
        clock.now = pd.Timestamp('2026-10-01T00:00:10Z')
        recipe = FrozenLinearRecipe('K0', runtime.spec)
        authority = MonthlyWorkAuthority(tmp_path / 'authority', run_id='forward-preparation-artificial',
                                        task='volatility', universe=list(runtime.config.symbols))
        job = runtime.lifecycle.plan_fit('2026-10-01T00:00:00Z', 'A0', runtime.recipe_for(recipe))
        yield SimpleNamespace(runtime=runtime, recipe=recipe, authority=authority, clock=clock, job=job)
    finally:
        runtime.close()


def begin(case, **kwargs):
    return ForwardPreparation.begin(case.runtime, case.authority, case.job, case.recipe, **kwargs)


def ticket(case, prep):
    h, record = prep._header, prep.record
    value = {'version': 'v7-known-forward-ticket-v2', 'authority_directory': str(case.authority.directory),
             'context': {k: h[k] for k in ('job', 'plan_id', 'pipeline', 'cutoff', 'stream')},
             'recipe': case.recipe.to_dict(), 'snapshot_ref': prep.snapshot_ref,
             'preparation_ref': prep.reference, 'runtime': {'producer': 'test-only'},
             'source_hash': h['source_hash'], **{k: record[k] for k in
             ('reservation_id', 'authority_manifest_hash', 'terms_hash', 'lease', 'lease_hash')}}
    value['context']['lifecycle_directory'] = str(case.runtime.directory)
    path = case.authority.directory / h['ticket_directory'] / h['work'] / 'ticket.json'
    module._write(path, value)
    return path, value


def test_actual_snapshot_is_prepaid_and_retains_lease_and_reserved_state(case, monkeypatch):
    lease = FrozenWorkLease.issue(300)
    prep = begin(case, lease=lease)
    record = prep.record
    actual = case.runtime.prepare_fit_snapshot
    def snapshot(job):
        canonical = case.authority.get(record['reservation_id'])
        assert canonical['status'] == 'reserved'
        assert canonical['lease'] == lease.to_dict()
        assert case.authority.status(record['actual_month'])['counts']['solve'] == 1
        return actual(job)
    monkeypatch.setattr(case.runtime, 'prepare_fit_snapshot', snapshot)
    data = prep.snapshot()
    assert data.record['snapshot_id'] == prep.snapshot_ref['snapshot_id']
    assert prep.record == record
    path, value = ticket(case, prep)
    sealed = prep.seal_ticket(path, value)
    loaded = ForwardPreparation.load(case.authority, prep.reference)
    assert loaded.verify_ticket(path, value) == record
    assert sealed['digest'] == fingerprint(value)
    with pytest.raises(ValueError, match='read-only|already attempted'):
        loaded.snapshot()
    with pytest.raises(ValueError, match='already sealed'):
        prep.seal_ticket(path, value)
    assert case.runtime.lifecycle.active_model('A0') is None


def test_snapshot_failure_consumes_and_never_retries(case, monkeypatch):
    prep = begin(case)
    record = prep.record
    def fail(job):
        raise OSError('artificial source failure')
    monkeypatch.setattr(case.runtime, 'prepare_fit_snapshot', fail)
    with pytest.raises(OSError, match='source failure'):
        prep.snapshot()
    assert case.authority.get(record['reservation_id'])['status'] == 'failed_consumed'
    counts = case.authority.status(record['actual_month'])
    with pytest.raises(ValueError, match='already attempted'):
        begin(case)
    with pytest.raises(ValueError, match='already attempted'):
        prep.snapshot()
    assert case.authority.status(record['actual_month']) == counts


@pytest.mark.parametrize('which', ['primary', 'secondary'])
def test_missing_intent_never_rebuilds_or_refunds(case, which):
    prep = begin(case)
    record = prep.record
    root = prep._primary if which == 'primary' else prep._secondary
    (root / 'intent.json').unlink()
    with pytest.raises((ValueError, OSError)):
        ForwardPreparation.load(case.authority, prep.reference)
    with pytest.raises(ValueError, match='already attempted'):
        begin(case)
    assert case.authority.status(record['actual_month'])['counts']['solve'] == 1


@pytest.mark.parametrize('changed', ['recipe', 'job', 'runtime', 'source'])
def test_scope_changes_reject_before_snapshot(case, monkeypatch, changed):
    monkeypatch.setattr(case.runtime, 'prepare_fit_snapshot', lambda *_: pytest.fail('unpaid numerical work'))
    if changed == 'recipe':
        case.recipe = FrozenLinearRecipe('HAR_cal', case.runtime.spec)
    elif changed == 'job':
        case.job = 'f' * 64
    elif changed == 'runtime':
        case.runtime.identity['config']['context_hours'] = 999
    else:
        (case.runtime.directory / 'source/factorlab/v7.py').write_text('changed', encoding='utf-8')
    with pytest.raises((ValueError, OSError)):
        begin(case)
    assert case.authority.verify()['reservations'] == 0


def test_wrong_parent_lease_is_not_reissued(case):
    from dataclasses import replace
    lease = replace(FrozenWorkLease.issue(), parent_pid=999999)
    with pytest.raises(ValueError, match='original issuer'):
        begin(case, lease=lease)
    assert case.authority.verify()['reservations'] == 0


def test_post_snapshot_expiry_consumes_preserves_original_exception(case, monkeypatch):
    prep = begin(case)
    record = prep.record
    actual = case.runtime.prepare_fit_snapshot
    def snapshot(job):
        value = actual(job)
        def expired(*args, **kwargs):
            raise RuntimeError('synthetic original deadline')
        monkeypatch.setattr(FrozenWorkLease, 'assert_parent_alive', expired)
        return value
    monkeypatch.setattr(case.runtime, 'prepare_fit_snapshot', snapshot)
    with pytest.raises(RuntimeError, match='original deadline'):
        prep.snapshot()
    assert case.authority.get(record['reservation_id'])['status'] == 'failed_consumed'
    with pytest.raises(ValueError, match='failed or consumed'):
        ForwardPreparation.load(case.authority, prep.reference)


def test_original_deadline_before_snapshot_also_closes_reservation(case, monkeypatch):
    prep = begin(case)
    record = prep.record
    def expired(*args, **kwargs):
        raise RuntimeError('synthetic original deadline')
    monkeypatch.setattr(FrozenWorkLease, 'assert_parent_alive', expired)
    monkeypatch.setattr(case.runtime, 'prepare_fit_snapshot', lambda *_: pytest.fail('expired snapshot'))
    with pytest.raises(RuntimeError, match='original deadline'):
        prep.snapshot()
    assert case.authority.get(record['reservation_id'])['status'] == 'failed_consumed'


def test_final_ticket_cannot_replace_snapshot_or_recipe(case):
    prep = begin(case)
    record = prep.record
    prep.snapshot()
    path, value = ticket(case, prep)
    changed = copy.deepcopy(value)
    changed['recipe'] = FrozenLinearRecipe('HAR_cal', case.runtime.spec).to_dict()
    path.unlink()
    module._write(path, changed)
    with pytest.raises(ValueError, match='original preparation binding'):
        prep.seal_ticket(path, changed)
    assert case.authority.get(record['reservation_id'])['status'] == 'failed_consumed'


def test_snapshot_bytes_and_ticket_bytes_remain_immutable(case):
    prep = begin(case)
    prep.snapshot()
    path, value = ticket(case, prep)
    prep.seal_ticket(path, value)
    path.write_text(path.read_text('utf-8') + ' ', encoding='utf-8')
    with pytest.raises(ValueError, match='seal changed'):
        prep.verify_ticket(path, value)


def test_pool_recursive_cost_is_prepaid_before_features(case, monkeypatch):
    from factorlab.discovery.joint_pool import FrozenPool
    from factorlab.discovery.pool_forward import PoolForwardRecipe
    recipe = PoolForwardRecipe(FrozenPool(role='reference', version='K0_v1'), case.runtime.spec)
    job = case.runtime.lifecycle.plan_fit('2026-10-01T00:00:00Z', 'pool-test-only',
                                          case.runtime.recipe_for(recipe.pool))
    # This utility never authorizes that stream for fitting/serving. Producer
    # role checks must run before it, and the lifecycle has no admitted model.
    prep = ForwardPreparation.begin(case.runtime, case.authority, job, recipe)
    assert prep.record['counts']['solve'] == recipe.solve_limit == 1
    assert prep._header['purpose'] == 'pool_forward'
    assert case.runtime.lifecycle.active_model('pool-test-only') is None


def test_failed_failure_log_does_not_erase_spend_or_original_error(case, monkeypatch):
    prep = begin(case)
    record = prep.record
    def fail(job):
        raise OSError('actual source read failure')
    monkeypatch.setattr(case.runtime, 'prepare_fit_snapshot', fail)
    original = module._write
    def write(path, value):
        if path.name == 'failure.json':
            raise OSError('failure log unavailable')
        return original(path, value)
    monkeypatch.setattr(module, '_write', write)
    with pytest.raises(OSError, match='actual source read failure'):
        prep.snapshot()
    assert case.authority.get(record['reservation_id'])['status'] == 'failed_consumed'
    with pytest.raises(ValueError, match='failed or consumed'):
        ForwardPreparation.load(case.authority, prep.reference)


def test_readonly_open_cannot_execute_or_fail_live_reservation(case):
    prep = begin(case)
    original = prep.record
    loaded = ForwardPreparation.load(case.authority, prep.reference)
    with pytest.raises(ValueError, match='read-only'):
        loaded.snapshot()
    with pytest.raises(ValueError, match='read-only'):
        loaded.seal_ticket('missing', {})
    assert prep.record == original


def test_static_historical_verifier_never_checks_live_parent_or_reconstructs_runtime(case, monkeypatch):
    prep = begin(case)
    prep.snapshot()
    path, value = ticket(case, prep)
    prep.seal_ticket(path, value)
    record = prep.record
    case.authority.start(record['reservation_id'], {'pid': record['lease']['parent_pid']})
    completed = case.authority.complete(record['reservation_id'], result={'fixture': 'accounting-only'},
                                       actual_counts={'solve': 1})
    def forbidden(*args, **kwargs):
        pytest.fail('historical verification tried to execute or renew live work')
    monkeypatch.setattr(FrozenWorkLease, 'assert_parent_alive', forbidden)
    monkeypatch.setattr(ForwardRuntime, '__init__', forbidden)
    assert ForwardPreparation.verify_ticket_reference(path, value) == completed
    # Accounting completion in this test is NOT production model admission.
    assert case.runtime.lifecycle.active_model('A0') is None


def test_partial_snapshot_claim_never_retries_even_without_failure_log(case):
    prep = begin(case)
    record = prep.record
    module._write(prep._primary / 'snapshot_started.json', {'prepare_terms': prep._digest})
    with pytest.raises(ValueError, match='already attempted'):
        prep.snapshot()
    with pytest.raises(ValueError, match='already attempted'):
        begin(case)
    assert case.authority.get(record['reservation_id']) == record
    with pytest.raises((ValueError, OSError)):
        _ = prep.snapshot_ref


def test_snapshot_metadata_rehashed_mutation_cannot_change_fixed_job(case):
    prep = begin(case)
    prep.snapshot()
    ref = prep.snapshot_ref
    path = module.project_path(ref['path'])
    value = module._read(path)
    value['cutoff'] = '2026-11-01T00:00:00+00:00'
    value['snapshot_id'] = fingerprint({k: v for k, v in value.items() if k != 'snapshot_id'})
    path.unlink()
    module.immutable_record(path, value)
    with pytest.raises(ValueError, match='snapshot bytes'):
        _ = prep.snapshot_ref
