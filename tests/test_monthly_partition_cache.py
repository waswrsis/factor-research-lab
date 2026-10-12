"""Count verified reads exactly; still recheck every new verifier invocation."""
import pytest
from test_monthly_partitions import begin, search, wallet
from test_monthly_partitions import root as root  # noqa: PLC0414

from factorlab.discovery import neural_research_ticket


def test_one_real_store_read_per_wallet_verify_not_per_row(root, monkeypatch):
    a = wallet(root)
    b = search(root, a)
    try:
        for i in range(5): begin(b, 'work'+str(i), n=1).complete({'no_fit': True})
        original, calls = neural_research_ticket._reader, []
        def reader(directory):
            calls.append(str(directory))
            return original(directory)
        monkeypatch.setattr(neural_research_ticket, '_reader', reader)
        a.verify()
        assert calls == [str(b.store.directory)]
        a.verify()
        assert calls == [str(b.store.directory)]*2
        # New verification must reread actual source evidence, never trust last call.
        (b.store.directory/'source/factorlab/frozen.py').write_text('changed=1')
        with pytest.raises(ValueError): a.verify()
        assert len(calls) == 3
    finally: b.store.close()


def test_new_call_rereads_canonical_month_and_rejects_changed_input(root):
    a = wallet(root)
    b = search(root, a)
    try:
        begin(b, 'work', n=1).complete({'no_fit': True})
        a.verify()
        state = dict(b.state)
        state['data_id'] = 'changed-but-valid-new-RunStore-event'
        b.store.save(b.month, state, 'changed_data')
        with pytest.raises(ValueError, match='monthly journal/input'):
            a.verify()
    finally: b.store.close()


def test_cached_snapshot_never_skips_each_original_issuer_check(root, monkeypatch):
    from dataclasses import replace

    from factorlab.discovery.monthly_partition_policy import _search
    from factorlab.discovery.work_lease import FrozenWorkLease
    a = wallet(root)
    b = search(root, a)
    try:
        work = begin(b, 'work', n=1)
        record = work.record
        original, calls = neural_research_ticket._reader, []
        def reader(directory):
            calls.append(str(directory))
            return original(directory)
        monkeypatch.setattr(neural_research_ticket, '_reader', reader)
        lease = FrozenWorkLease.from_dict(record['lease'])
        cache = {}
        _search(a, b.identifier, current=False, lease=lease, verification_cache=cache)
        with pytest.raises(ValueError, match='original S execution clock'):
            _search(a, b.identifier, current=False, lease=replace(lease, parent_pid=lease.parent_pid+1), verification_cache=cache)
        assert calls == [str(b.store.directory)]
        assert len(cache) == 1
    finally: b.store.close()
