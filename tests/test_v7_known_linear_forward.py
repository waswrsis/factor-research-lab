"""Actual receipt/authority chain on artificial data; kernel is explicitly fake."""
from __future__ import annotations

import copy
import os
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from factorlab.discovery import known_linear_forward as module
from factorlab.discovery.forward_publication_witness import ForwardPublicationWitness
from factorlab.discovery.known_linear_reference import FrozenLinearRecipe, KnownLinearPredictor
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
from factorlab.discovery.v7_forward import ForwardRuntime
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
        authority = MonthlyWorkAuthority(tmp_path / 'authority', run_id='known-forward-artificial',
                                        task='volatility', universe=list(runtime.config.symbols))
        # An explicit engineering substitute. No actual Linux proof is generated.
        def kernel():
            return {'path': '/sys/fs/cgroup/artificial', 'inode': 321,
                    'boot_id': str(uuid.UUID(int=1)), 'scope': 'whole_cgroup', 'platform': 'linux',
                    'observed_at': datetime.now(UTC).isoformat(),
                    'raw': {'cgroup.procs': str(os.getpid())}, 'fixture': True}
        monkeypatch.setattr(module, '_kernel_snapshot', kernel)
        monkeypatch.setattr(module, '_validate_kernel', lambda *args: None)
        monkeypatch.setattr(module, '_kernel', lambda *args: None)
        # Validity of the production kernel validator has independent strict tests.
        import factorlab.discovery.neural_forward_admission as admission
        monkeypatch.setattr(admission, '_kernel', lambda *args: None)
        yield SimpleNamespace(runtime=runtime, recipe=recipe, authority=authority, clock=clock)
    finally:
        runtime.close()


def run(case):
    return case.runtime.fit_month('2026-10-01T00:00:00Z', pool=case.recipe, authority=case.authority)


def ticket_for(case):
    paths = list((case.authority.directory / 'known_forward_tickets').glob('*/ticket.json'))
    assert len(paths) == 1
    return module.KnownForwardTicket.load(paths[0])


def test_actual_chain_fits_once_and_registry_reopen_rejects_missing_witness(case, monkeypatch):
    result = run(case)
    ticket = ticket_for(case)
    reservation = case.authority.get(ticket.value['reservation_id'])
    assert reservation['status'] == 'completed'
    assert reservation['actual_counts']['solve'] == reservation['counts']['solve'] == 1
    model, identity = case.runtime.lifecycle.registry.load(result['model_id'])
    assert identity['version'] == 3 and identity['job'] == result['job']
    assert model.recipe == case.recipe
    before = case.authority.status(reservation['actual_month'])
    monkeypatch.setattr(module, 'fit_known_linear', lambda *args, **kwargs: pytest.fail('cache refitted'))
    assert run(case)['cached']
    assert case.authority.status(reservation['actual_month']) == before
    assert ForwardPublicationWitness.read_completed(ticket)['model_digest'] == model.digest
    (ticket.path.parent / 'launch/publication_witness/witness.sqlite').unlink()
    with pytest.raises(ValueError, match='witness database missing'):
        case.runtime.lifecycle.registry.load(result['model_id'])


def test_missing_authority_and_neural_gates_reject_before_numerical_work(case, monkeypatch):
    monkeypatch.setattr(module, 'fit_known_linear', lambda *args, **kwargs: pytest.fail('unpaid fit'))
    with pytest.raises(ValueError, match='actual shared'):
        case.runtime.fit_month('2026-10-01Z', pool=case.recipe)
    with pytest.raises(ValueError, match='Neural gates'):
        case.runtime.fit_month('2026-10-01Z', pool=case.recipe, authority=case.authority, gates={})
    assert case.authority.verify()['reservations'] == 0


def test_kernel_preflight_failure_consumed_and_no_retry(case, monkeypatch):
    def fail():
        raise ValueError('synthetic missing Linux capacity')
    monkeypatch.setattr(module, '_kernel_snapshot', fail)
    with pytest.raises(ValueError, match='missing Linux'):
        run(case)
    ticket = ticket_for(case)
    record = case.authority.get(ticket.value['reservation_id'])
    assert record['status'] == 'failed_consumed'
    before = case.authority.status(record['actual_month'])
    monkeypatch.setattr(module, 'fit_known_linear', lambda *args, **kwargs: pytest.fail('refit failed ticket'))
    with pytest.raises(ValueError, match='already attempted'):
        run(case)
    assert case.authority.status(record['actual_month']) == before
    assert case.runtime.lifecycle.active_model('A0') is None


def test_snapshot_byte_change_rejects_before_fit_and_remains_charged(case, monkeypatch):
    job = case.runtime.lifecycle.plan_fit('2026-10-01T00:00:00Z', 'A0', case.runtime.recipe_for(case.recipe))
    ticket = module.issue_known_ticket(case.runtime, case.authority, job, case.recipe)
    path = pd.io.common.stringify_path(ticket.value['snapshot_ref']['path'])
    with open(path, 'a', encoding='utf-8') as file:
        file.write(' ')
    monkeypatch.setattr(module, 'fit_known_linear', lambda *args, **kwargs: pytest.fail('tampered fit'))
    with pytest.raises(ValueError, match='snapshot bytes'):
        run(case)
    assert case.authority.get(ticket.value['reservation_id'])['status'] == 'failed_consumed'


def test_late_final_sql_commit_and_failed_failure_log_cannot_publish(case, monkeypatch):
    from factorlab.discovery.work_lease import FrozenWorkLease
    original_prepare = ForwardPublicationWitness.prepare
    original_write = module._write
    def prepare(ticket):
        real = original_prepare(ticket)
        def complete(proof):
            original_assert = FrozenWorkLease.assert_parent_alive
            checks = []
            def closed(lease, *args, **kwargs):
                checks.append(1)
                if len(checks) == 2:
                    raise ValueError('synthetic final deadline')
                return original_assert(lease, *args, **kwargs)
            monkeypatch.setattr(FrozenWorkLease, 'assert_parent_alive', closed)
            return real.complete(proof)
        return SimpleNamespace(complete=complete)
    monkeypatch.setattr(ForwardPublicationWitness, 'prepare', prepare)
    def write(path, value):
        if path.name == 'failure.json':
            raise OSError('synthetic read-only error log')
        return original_write(path, value)
    monkeypatch.setattr(module, '_write', write)
    with pytest.raises(ValueError, match='final deadline'):
        run(case)
    ticket = ticket_for(case)
    assert case.authority.get(ticket.value['reservation_id'])['status'] == 'completed'
    assert not (ticket.path.parent / 'launch/failure.json').exists()
    with pytest.raises(ValueError, match='pending'):
        ForwardPublicationWitness.read_completed(ticket)
    model = KnownLinearPredictor.load(ticket.path.parent / 'launch/model.json')
    certificate = module._record(ticket.path.parent / 'launch/certificate.json')
    with pytest.raises(ValueError, match='pending'):
        module.verify_known_admission(model, certificate,
            expected_pipeline=case.runtime.recipe_for(case.recipe), expected_job=ticket.value['context']['job'])


def test_har_cal_cannot_use_unregistered_a0_or_stream(case):
    har = FrozenLinearRecipe('HAR_cal', case.runtime.spec)
    with pytest.raises(ValueError, match='K0 fallback'):
        case.runtime.fit_month('2026-10-01T00:00:00Z', pool=har, authority=case.authority)
    with pytest.raises((ValueError, OSError)):
        case.runtime.fit_month('2026-10-01T00:00:00Z', stream='fake-benchmark', pool=har, authority=case.authority)
    assert case.authority.verify()['reservations'] == 0


@pytest.mark.parametrize('target', ['job', 'pipeline'])
def test_admission_cannot_borrow_job_or_full_recipe(case, target):
    result = run(case)
    model, identity = case.runtime.lifecycle.registry.load(result['model_id'])
    job, pipeline = result['job'], copy.deepcopy(identity['pipeline'])
    if target == 'job':
        job = fingerprint('unrelated-job')
    else:
        pipeline['recipe']['name'] = 'HAR_cal'
    with pytest.raises(ValueError, match='actual job'):
        module.verify_known_admission(model, identity['admission'], expected_pipeline=pipeline, expected_job=job)


def test_claim_reference_real_har_cal_stream_and_scheduler(case):
    from factorlab.discovery.claim_protocol import ClaimProtocol, FrozenClaimEndpoint
    from factorlab.discovery.v7_service import ForwardService, ServiceConfig
    case.clock.now = pd.Timestamp('2026-09-30T23:50:00Z')
    k0 = case.runtime.recipe_for()
    har = FrozenLinearRecipe('HAR_cal', case.runtime.spec)
    ClaimProtocol.freeze(case.runtime.lifecycle, [
        FrozenClaimEndpoint.library('2026-10-01T00:00:00Z', k0, case.runtime.recipe_for(har)),
        FrozenClaimEndpoint.algorithm('2026-10-01T00:00:00Z', k0)])
    case.clock.now = pd.Timestamp('2026-10-01T00:00:10Z')
    service = ForwardService(case.runtime, work_authority=case.authority,
        config=ServiceConfig(poll=False, max_fits_per_step=1), guard=lambda: {'allowed': True})
    try:
        service._plan_month(case.clock.now)
        # Isolate the genuinely planned HAR job while legacy pool dispatch is still separate.
        row = case.runtime.lifecycle.db.execute("SELECT id FROM jobs WHERE stream='benchmark'").fetchone()
        assert row is not None
        result = case.runtime.fit_month('2026-10-01T00:00:00Z', stream='benchmark', pool=har, authority=case.authority)
        model, identity = case.runtime.lifecycle.registry.load(result['model_id'])
        assert model.recipe.name == 'HAR_cal' and identity['version'] == 3
        ticket = ticket_for(case)
        assert ticket.value['context']['role'] == 'fixed_claim_reference'
        assert ticket.value['context']['slots'] == ['library_skill']
    finally:
        service.close()
