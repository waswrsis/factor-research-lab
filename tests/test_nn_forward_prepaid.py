"""Real archive/job/wallet; artificial OHLC and private pre-resource gate seam."""
import json
from pathlib import Path

import pandas as pd
import pytest
from test_v7_feed_audit import Clock
from test_v7_rolling_registry import candles
from test_v7_run_factor_pool import source_root

from factorlab.discovery import neural_forward_ticket as tickets
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
from factorlab.discovery.neural_forward import FrozenNeuralRecipe
from factorlab.discovery.v7_forward import ForwardRuntime
from factorlab.discovery.work_lease import FrozenWorkLease
from factorlab.protocol import fingerprint


@pytest.fixture
def case(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    clock = Clock('2026-09-30T23:30:00Z')
    runtime = ForwardRuntime(tmp_path/'run', tmp_path/'feed', clock=clock)
    path = tmp_path/'artificial.parquet'
    candles('2026-05-01T00:00Z', '2026-09-30T23:00Z').to_parquet(path)
    runtime.feed.import_seed(path)
    clock.now = pd.Timestamp('2026-10-01T00:00:10Z')
    recipe = FrozenNeuralRecipe(rolling_spec=runtime.spec)
    job = runtime.lifecycle.plan_fit('2026-10-01T00:00:00Z', 'A0', runtime.recipe_for(recipe))
    authority = MonthlyWorkAuthority(tmp_path/'authority', run_id='prepaid-engineering',
        task='volatility', universe=list(runtime.config.symbols), partition_policy='standard')
    monkeypatch.setattr(tickets, '_verify_gates', lambda *args, **kwargs: None)
    yield runtime, authority, job, recipe
    runtime.close()


def issue(case):
    runtime, authority, job, recipe = case
    return tickets.issue_forward_ticket(runtime, authority, job, recipe, purpose='resource_trial', gates={})


def records(authority):
    with authority._transaction() as db:
        return list(authority._verify(db)['reservations'].values())


@pytest.mark.parametrize('stage', ['snapshot', 'gates', 'payload'])
def test_every_expensive_preparation_stage_is_already_charged_and_failure_never_retries(case, monkeypatch, stage):
    runtime, authority, _, _ = case
    calls = []
    def fail(*args, **kwargs):
        rows = records(authority)
        assert len(rows) == 1 and rows[0]['status'] == 'reserved'
        assert rows[0]['lease']['lease_seconds'] == 300
        assert rows[0]['counts']['neural_fit'] == 3
        calls.append(rows[0])
        raise OSError('artificial '+stage+' failure')
    if stage == 'snapshot': monkeypatch.setattr(runtime, 'prepare_fit_snapshot', fail)
    elif stage == 'gates': monkeypatch.setattr(tickets, '_verify_gates', fail)
    else: monkeypatch.setattr(tickets, 'save_worker_data', fail)
    with pytest.raises(OSError, match='artificial'):
        issue(case)
    row = records(authority)[0]
    assert row['status'] == 'failed_consumed'
    before = authority.status(row['actual_month'])
    with pytest.raises(ValueError, match='Incomplete'):
        issue(case)
    assert len(calls) == 1 and authority.status(row['actual_month']) == before


def test_success_readback_never_reissues_or_reloads_inputs(case, monkeypatch):
    ticket = issue(case)
    _, authority, _, _ = case
    row = authority.get(ticket.value['reservation_id'])
    def prohibited(*args, **kwargs): raise AssertionError('readback restarted work')
    monkeypatch.setattr(FrozenWorkLease, 'issue', prohibited)
    monkeypatch.setattr(case[0], 'prepare_fit_snapshot', prohibited)
    assert issue(case).digest == ticket.digest
    assert authority.get(row['reservation_id']) == row
    assert row['binding']['prework_terms'] == ticket.value['preparation_ref']['digest']
    assert row['work_key'] == 'neural_forward:'+fingerprint({'plan': case[0].lifecycle.plan.digest, 'job': case[2]})


@pytest.mark.parametrize('damage', ['allocation', 'seal', 'other_seal'])
def test_missing_dual_preparation_anchor_cannot_reopen_ticket(case, damage):
    ticket = issue(case)
    runtime, authority, _, _ = case
    work = ticket.path.parent.name
    path = ticket.path.parent/('allocation.json' if damage == 'allocation' else 'seal.json')
    if damage == 'other_seal': path = runtime.directory/'forward_allocations'/work/'seal.json'
    path.unlink()
    with pytest.raises((OSError, ValueError)):
        issue(case)
    # Removing a final seal never resets the already durable charged row.
    if damage != 'allocation': assert records(authority)[0]['counts']['neural_fit'] == 3


def test_late_snapshot_keeps_original_deadline_and_fails_consumed(case, monkeypatch):
    from factorlab.discovery import work_lease
    original = case[0].prepare_fit_snapshot
    def late(job):
        prepared = original(job)
        observe = work_lease._observe
        def expired(lease):
            value = observe(lease)
            value['monotonic'] = lease.monotonic_deadline+1
            return value
        monkeypatch.setattr(work_lease, '_observe', expired)
        return prepared
    monkeypatch.setattr(case[0], 'prepare_fit_snapshot', late)
    with pytest.raises(work_lease.LeaseClosed): issue(case)
    row = records(case[1])[0]
    # Canonical authority closes a lease-expired active reservation as interrupted.
    assert row['status'] == 'interrupted_consumed' and row['lease']['lease_seconds'] == 300


def test_seal_write_failure_with_ticket_bytes_never_creates_readback_authority(case, monkeypatch):
    original = tickets.immutable_record
    def write(path, value):
        if Path(path).name == 'seal.json': raise OSError('artificial seal publication failure')
        return original(path, value)
    monkeypatch.setattr(tickets, 'immutable_record', write)
    with pytest.raises(OSError): issue(case)
    row = records(case[1])[0]
    assert row['status'] == 'failed_consumed'
    with pytest.raises((OSError, ValueError)): issue(case)
    assert len(records(case[1])) == 1


def test_rehashed_final_ticket_cannot_replace_prework_context(case):
    ticket = issue(case)
    value = dict(ticket.value)
    value['gates'] = {'invented': True}
    ticket.path.write_text(json.dumps({'value': value, 'sha256': fingerprint(value)}))
    modified = tickets.ForwardFitTicket.load(ticket.path)
    with pytest.raises(ValueError, match='preparation seal'):
        tickets.verify_prepaid_ticket(modified, case[1], records(case[1])[0])


def test_committed_debit_with_missing_mirror_is_recovered_without_preparation_retry(case, monkeypatch):
    from factorlab.discovery import monthly_work_authority as wallet_module
    original = wallet_module.immutable_record
    def fail_mirror(path, value):
        if Path(path).parent == case[1].directory/'events':
            raise OSError('artificial committed mirror failure')
        return original(path, value)
    monkeypatch.setattr(wallet_module, 'immutable_record', fail_mirror)
    with pytest.raises(OSError, match='committed mirror'):
        issue(case)
    monkeypatch.setattr(wallet_module, 'immutable_record', original)
    # Admission's readonly opener must NOT silently repair incomplete evidence.
    with pytest.raises(ValueError, match='mirror pending'):
        tickets.open_existing_authority(case[1].directory)
    restored = MonthlyWorkAuthority(case[1].directory, run_id='prepaid-engineering',
        task='volatility', universe=list(case[0].config.symbols), partition_policy='standard')
    rows = records(restored)
    assert len(rows) == 1 and rows[0]['counts']['neural_fit'] == 3
    before = restored.status(rows[0]['actual_month'])
    with pytest.raises(ValueError, match='Incomplete'):
        issue(case)
    assert restored.status(rows[0]['actual_month']) == before
    assert not (restored.directory/'forward_tickets'/rows[0]['work_key'].split(':')[1]/'payload').exists()


def test_gate_serialization_uses_the_original_clock_before_new_work(case, monkeypatch):
    calls = []
    original_issue, original_gates = FrozenWorkLease.issue, tickets._canonical_forward_gates
    def start(seconds):
        lease = original_issue(seconds)
        calls.append(lease)
        return lease
    def gates(value):
        assert len(calls) == 1
        calls[0].assert_parent_alive()
        return original_gates(value)
    monkeypatch.setattr(FrozenWorkLease, 'issue', start)
    monkeypatch.setattr(tickets, '_canonical_forward_gates', gates)
    ticket = issue(case)
    assert len(calls) == 1 and ticket.value['lease'] == calls[0].to_dict()
