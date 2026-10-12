import json
import multiprocessing
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from factorlab.discovery import monthly_work_authority as module
from factorlab.discovery.monthly_work_authority import BudgetExhausted, MonthlyWorkAuthority
from factorlab.discovery.work_lease import FrozenWorkLease, LeaseCheck


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    (tmp_path / 'factorlab').mkdir()
    (tmp_path / 'factorlab/frozen.py').write_text('version=1', encoding='utf-8')
    (tmp_path / 'pyproject.toml').write_text('[project]', encoding='utf-8')
    (tmp_path / 'uv.lock').write_text('version=1', encoding='utf-8')
    return tmp_path


def authority(root, **kwargs):
    return MonthlyWorkAuthority(root / 'wallet', run_id='run', task='volatility', universe=['BTCUSDT'], **kwargs)


def reserve(a, key='job', **kwargs):
    return a.reserve(key, counts={'solve': 1, 'neural_fit': 1, 'epoch': 2},
                     binding={'job': key, 'recipe': 'fixed'}, **kwargs)


def test_prepaid_all_categories_shared_and_no_refunds(root):
    a = authority(root)
    counts = {k: 1 for k in module.DEFAULT_LIMITS}
    r = a.reserve('S', counts=counts, binding={'channel': 'S'})
    assert r['lease_hash'] == FrozenWorkLease.from_dict(r['lease']).digest
    assert r['actual_month'] == datetime.now(UTC).strftime('%Y-%m')
    assert r['authority_manifest_hash'] == a.manifest_hash
    assert r['authority_directory'] == str(a.directory)
    assert a.start(r['reservation_id'], {'pid': 42})['status'] == 'running'
    assert a.fail_consumed(r['reservation_id'], 'synthetic failure')['status'] == 'failed_consumed'
    a.reserve('C', counts={'solve': 1}, binding={'channel': 'C'})
    status = a.status(r['actual_month'])
    assert status['counts']['solve'] == 2 and status['counts']['neural_fit'] == 1
    assert status['reserved_seconds'] == 600
    with pytest.raises(ValueError, match='Consumed'):
        a.start(r['reservation_id'], {'pid': 43})


def test_same_work_retry_alias_and_modified_terms_never_recharge(root):
    a = authority(root)
    r = reserve(a, retry_id='first')
    assert reserve(a, retry_id='different') == r
    with pytest.raises(ValueError, match='cannot change'):
        reserve(a, lease_seconds=299)
    with pytest.raises(ValueError, match='cannot change'):
        a.reserve('job', counts={'solve': 2}, binding={'job': 'job', 'recipe': 'fixed'})
    assert a.verify()['events'] == 1
    assert a.status(r['actual_month'])['counts']['solve'] == 1


def test_completed_idempotent_even_old_parent_lost(root, monkeypatch):
    a = authority(root)
    r = reserve(a)
    a.start(r['reservation_id'], {'pid': 12})
    done = a.complete(r['reservation_id'], result={'model_hash': 'sealed'}, actual_counts={'solve': 1})
    assert done['actual_counts']['neural_fit'] == 0
    monkeypatch.setattr(FrozenWorkLease, 'check', lambda _: LeaseCheck('parent_lost'))
    assert a.recover(r['reservation_id']) == done
    assert a.complete(r['reservation_id'], result={'model_hash': 'sealed'}, actual_counts={'solve': 1}) == done
    with pytest.raises(ValueError, match='committed transition'):
        a.complete(r['reservation_id'], result={'model_hash': 'different'}, actual_counts={})
    assert a.status(r['actual_month'])['counts']['neural_fit'] == 1


@pytest.mark.parametrize('reason', ['deadline', 'boot_changed', 'clock_rollback', 'parent_lost', 'wrong_process'])
def test_recover_closed_original_lease_consumes_never_reissues(root, monkeypatch, reason):
    a = authority(root)
    r = reserve(a)
    monkeypatch.setattr(FrozenWorkLease, 'check', lambda _: LeaseCheck(reason))
    recovered = a.recover(r['reservation_id'])
    assert recovered['status'] == 'interrupted_consumed'
    assert recovered['lease'] == r['lease']
    assert reserve(a, retry_id='new') == recovered
    assert a.status(r['actual_month'])['reserved_seconds'] == 300


@pytest.mark.parametrize('table', ['wallets', 'reservations', 'events'])
def test_deleted_projection_or_chain_cannot_reset(root, table):
    a = authority(root)
    reserve(a)
    with sqlite3.connect(a.database) as db:
        db.execute('DELETE FROM ' + table)
    with pytest.raises(ValueError):
        authority(root)


@pytest.mark.parametrize('name', ['authority.sqlite', 'manifest.json'])
def test_missing_authority_primary_state_fails_closed(root, name):
    a = authority(root)
    reserve(a)
    (a.directory / name).unlink()
    with pytest.raises(ValueError):
        authority(root)


def test_alternate_directory_cannot_mint_same_scope(root):
    a = authority(root)
    reserve(a)
    with pytest.raises(ValueError, match='Existing authority'):
        MonthlyWorkAuthority(root / 'renamed', run_id='run', task='volatility', universe=['BTCUSDT'])


def test_source_current_and_archive_mutation_rejected(root):
    a = authority(root)
    reserve(a)
    archived = a.directory / 'source/factorlab/frozen.py'
    archived.write_text('version=2', encoding='utf-8')
    with pytest.raises(ValueError, match='archived source'):
        a.verify()
    archived.write_text('version=1', encoding='utf-8')
    (root / 'factorlab/frozen.py').write_text('version=2', encoding='utf-8')
    with pytest.raises(ValueError, match='Current source'):
        a.verify()


def test_commit_before_mirror_crash_recovers_no_double_charge(root, monkeypatch):
    a = authority(root)
    original = module.immutable_record

    def crash(path, value):
        if path.parent.name == 'events':
            raise OSError('after database commit before external mirror')
        return original(path, value)

    with monkeypatch.context() as scoped:
        scoped.setattr(module, 'immutable_record', crash)
        with pytest.raises(OSError):
            reserve(a)
    resumed = authority(root)
    r = reserve(resumed)
    assert resumed.verify()['events'] == 1
    assert resumed.status(r['actual_month'])['reserved_seconds'] == 300
    assert len(list((a.directory / 'events').glob('*.json'))) == 1


def test_intent_before_sql_commit_crash_blocks_reissue(root, monkeypatch):
    a = authority(root)
    original = module.immutable_record

    def crash(path, value):
        result = original(path, value)
        if path.parent.name == 'intents':
            raise OSError('after intent before SQL commit')
        return result

    with monkeypatch.context() as scoped:
        scoped.setattr(module, 'immutable_record', crash)
        with pytest.raises(OSError):
            reserve(a)
    with pytest.raises(ValueError, match='Orphan'):
        authority(root)


def test_database_rollback_and_external_tamper_rejected(root):
    a = authority(root)
    reserve(a)
    event = next((a.directory / 'events').glob('*.json'))
    payload = json.loads(event.read_text())
    payload['record']['reservation']['counts']['solve'] = 0
    event.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='external event'):
        a.verify()


def test_actual_month_from_original_lease_and_same_work_cannot_refresh(root, monkeypatch):
    a = authority(root)
    original = FrozenWorkLease.issue(300)

    def issue(month):
        values = original.to_dict()
        start = datetime.fromisoformat(month + '-01T00:00:00+00:00')
        values.update(wall_started_at=start.isoformat(), wall_deadline=(start + timedelta(seconds=300)).isoformat(),
                      parent_created=start.timestamp() - 1)
        return FrozenWorkLease.from_dict(values)

    monkeypatch.setattr(FrozenWorkLease, 'issue', lambda _: issue('2026-10'))
    monkeypatch.setattr(FrozenWorkLease, 'assert_parent_alive', lambda _: LeaseCheck('ok', 300))
    first = reserve(a)
    monkeypatch.setattr(FrozenWorkLease, 'issue', lambda _: issue('2026-11'))
    assert reserve(a) == first
    new = reserve(a, 'next-business-job')
    assert new['actual_month'] == '2026-11'
    assert a.status('2026-10')['counts']['solve'] == a.status('2026-11')['counts']['solve'] == 1
    monkeypatch.setattr(FrozenWorkLease, 'issue', lambda _: issue('2026-10'))
    with pytest.raises(ValueError, match='moved backwards'):
        reserve(a, 'rollback')


@pytest.mark.parametrize('counts', [{'solve': -1}, {'solve': True}, {'unknown': 1}, {}, {'solve': 1.1}])
def test_invalid_counts_never_publish(root, counts):
    a = authority(root)
    with pytest.raises(ValueError):
        a.reserve('bad', counts=counts, binding={'job': 'bad'})
    assert a.verify()['events'] == 0


def test_actual_overrun_and_nonfinite_result_rejected(root):
    a = authority(root)
    r = reserve(a)
    a.start(r['reservation_id'], {'pid': 42})
    with pytest.raises(ValueError, match='exceeds'):
        a.complete(r['reservation_id'], result={'hash': 'x'}, actual_counts={'solve': 2})
    with pytest.raises(ValueError):
        a.complete(r['reservation_id'], result={'loss': float('nan')}, actual_counts={})
    assert a.get(r['reservation_id'])['status'] == 'running'


def _race(root, barrier, queue, key):
    # Windows spawn is a real independent interpreter and SQLite connection.
    from pathlib import Path

    import factorlab.protocol
    factorlab.protocol.ROOT = Path(root)
    a = authority(Path(root), limits={'solve': 1})
    barrier.wait(timeout=20)
    try:
        r = a.reserve(key, counts={'solve': 1}, binding={'job': key})
        queue.put(('reserved', r['reservation_id']))
    except BudgetExhausted:
        queue.put(('exhausted', key))


def test_two_real_processes_contend_last_monthly_quota(root):
    a = authority(root, limits={'solve': 1})
    ctx = multiprocessing.get_context('spawn')
    barrier, queue = ctx.Barrier(2), ctx.Queue()
    workers = [ctx.Process(target=_race, args=(str(root), barrier, queue, str(i))) for i in range(2)]
    for worker in workers:
        worker.start()
    try:
        results = [queue.get(timeout=35) for _ in workers]
        for worker in workers:
            worker.join(timeout=10)
            assert worker.exitcode == 0
        assert sorted(x[0] for x in results) == ['exhausted', 'reserved']
        assert a.verify()['reservations'] == 1
        assert a.status(datetime.now(UTC).strftime('%Y-%m'))['counts']['solve'] == 1
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
        queue.close()


def test_quota_and_seconds_exhaustion_has_no_partial_debit(root):
    a = authority(root, monthly_seconds=300)
    first = reserve(a)
    with pytest.raises(BudgetExhausted):
        reserve(a, 'second')
    assert a.status(first['actual_month'])['reserved_seconds'] == 300
    assert a.verify()['events'] == 1


def test_original_pre_serialization_lease_is_not_reissued(root, monkeypatch):
    a = authority(root)
    lease = FrozenWorkLease.issue(300)
    monkeypatch.setattr(FrozenWorkLease, 'issue', lambda _: pytest.fail('must not reissue original lease'))
    r = a._reserve_with_lease('payload', counts={'solve': 1}, binding={'payload_hash': 'frozen'}, lease=lease)
    assert r['lease'] == lease.to_dict()
    assert r['lease_hash'] == lease.digest
    assert a._reserve_with_lease('payload', counts={'solve': 1}, binding={'payload_hash': 'frozen'}, lease=lease) == r


def test_private_lease_from_other_issuer_rejected(root):
    a = authority(root)
    value = FrozenWorkLease.issue(300).to_dict()
    value['parent_pid'] += 1000
    with pytest.raises(ValueError, match='this launcher'):
        a._reserve_with_lease('other', counts={'solve': 1}, binding={'payload': 'x'},
                              lease=FrozenWorkLease.from_dict(value))
    assert a.verify()['events'] == 0


def test_slow_intent_write_cannot_return_runnable_expired_work(root, monkeypatch):
    from factorlab.discovery.work_lease import LeaseClosed

    a = authority(root)
    original = module.immutable_record
    checks = [0]

    def check(_):
        checks[0] += 1
        if checks[0] > 1:
            raise LeaseClosed(LeaseCheck('deadline'))
        return LeaseCheck('ok', 1)

    monkeypatch.setattr(FrozenWorkLease, 'assert_parent_alive', check)
    monkeypatch.setattr(module, 'immutable_record', original)
    with pytest.raises(LeaseClosed):
        reserve(a)
    # No started job or new lease on reopen: ambiguous intent conservatively blocks.
    with pytest.raises(ValueError, match='Orphan'):
        authority(root)


def test_work_key_index_tamper_rejected(root):
    a = authority(root)
    reserve(a)
    with sqlite3.connect(a.database) as db:
        db.execute("UPDATE reservations SET work_key='renamed'")
    with pytest.raises(ValueError, match='work index'):
        a.verify()


@pytest.mark.parametrize('kwargs', [{'limits': {'solve': 1601}}, {'monthly_seconds': 7201},
                                    {'monthly_seconds': True}])
def test_protocol_ceilings_cannot_be_raised(root, kwargs):
    with pytest.raises(ValueError):
        authority(root, **kwargs)


def test_get_is_copy_and_empty_future_month_does_not_initialize_wallet(root):
    a = authority(root)
    r = reserve(a)
    copied = a.get(r['reservation_id'])
    copied['counts']['solve'] = 0
    assert a.get(r['reservation_id'])['counts']['solve'] == 1
    assert a.status('2099-01')['reserved_seconds'] == 0
    with sqlite3.connect(a.database) as db:
        assert db.execute('SELECT COUNT(*) FROM wallets').fetchone()[0] == 1


def test_creation_source_archive_is_immutable_and_reopen_exact(root):
    a = authority(root)
    r = reserve(a)
    b = authority(root)
    assert b.get(r['reservation_id']) == r
    with pytest.raises(ValueError, match='identity changed'):
        authority(root, limits={'solve': 1})


def test_worker_cannot_self_certify_parent_completion(root, monkeypatch):
    a = authority(root)
    r = reserve(a)
    a.start(r['reservation_id'], {'pid': 42})
    monkeypatch.setattr(module.os, 'getpid', lambda: r['lease']['parent_pid'] + 1000)
    with pytest.raises(ValueError, match='original launcher'):
        a.complete(r['reservation_id'], result={'model': 'untrusted'}, actual_counts={})
    assert a.get(r['reservation_id'])['status'] == 'running'
