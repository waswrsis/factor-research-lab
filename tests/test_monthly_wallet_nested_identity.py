"""Identity recursion seams, not model/resource/calibration acceptance."""
import contextvars
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from factorlab.discovery import monthly_work_authority as module
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
from factorlab.discovery.neural_forward_ticket import open_existing_authority


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    (tmp_path / 'factorlab').mkdir()
    (tmp_path / 'factorlab/frozen.py').write_text('version=1')
    (tmp_path / 'pyproject.toml').write_text('[project]')
    (tmp_path / 'uv.lock').write_text('version=1')
    return tmp_path


def wallet(root, name='wallet'):
    return MonthlyWorkAuthority(root / name, run_id=name, task='volatility', universe=['BTCUSDT'])


def reserve(a):
    return a.reserve('original', counts={'solve': 1}, binding={'job': 'original'})


def test_same_transaction_nested_open_and_full_verify_afterward(root):
    a = wallet(root)
    reserve(a)
    with a._transaction() as db, a._identity_read_scope(db):
        nested = open_existing_authority(a.directory)
        assert nested.manifest_hash == a.manifest_hash
        assert module._nested_identity_read(nested)
    assert not module._nested_identity_read(nested)
    assert nested.verify() == a.verify()


def test_copied_context_does_not_survive_scope_or_cross_threads(root):
    a = wallet(root)
    with a._transaction() as db, a._identity_read_scope(db):
        copied = contextvars.copy_context()
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(copied.run, module._nested_identity_read, a).result() is False
        assert copied.run(module._nested_identity_read, a)
    assert copied.run(module._nested_identity_read, a) is False


def test_exception_clears_active_and_copied_context(root):
    a = wallet(root)
    with pytest.raises(RuntimeError, match='test interrupted'), a._transaction() as db, a._identity_read_scope(db):
        copied = contextvars.copy_context()
        raise RuntimeError('test interrupted')
    assert not module._nested_identity_read(a)
    assert copied.run(module._nested_identity_read, a) is False
    assert a.verify()['events'] == 0


def test_other_wallet_performs_full_verify_inside_original_scope(root, monkeypatch):
    a, b = wallet(root), wallet(root, 'other')
    original, calls = MonthlyWorkAuthority._verify_rows, []

    def observe(self, db, **kwargs):
        calls.append(str(self.directory))
        return original(self, db, **kwargs)

    monkeypatch.setattr(MonthlyWorkAuthority, '_verify_rows', observe)
    with a._transaction() as db, a._identity_read_scope(db):
        assert not module._nested_identity_read(b)
        opened = open_existing_authority(b.directory)
    assert calls == [str(b.directory)]
    assert opened.manifest_hash == b.manifest_hash


def test_explicit_verify_is_never_an_identity_only_read(root, monkeypatch):
    a = wallet(root)
    with a._transaction() as db, a._identity_read_scope(db):
        def must_open_transaction(self):
            raise RuntimeError('explicit verification still opens transaction')
        monkeypatch.setattr(MonthlyWorkAuthority, '_transaction', must_open_transaction)
        with pytest.raises(RuntimeError, match='explicit verification'):
            a.verify()


@pytest.mark.parametrize('kind', ['index', 'manifest', 'source', 'archive'])
def test_nested_immutable_tampering_is_rejected(root, kind):
    a = wallet(root)
    with pytest.raises(ValueError), a._transaction() as db, a._identity_read_scope(db):
        path = {'index': a.index_path, 'manifest': a.directory / 'manifest.json',
                'source': root / 'factorlab/frozen.py',
                'archive': a.directory / 'source/factorlab/frozen.py'}[kind]
        path.write_text('{}' if kind in {'index', 'manifest'} else 'version=2')
        open_existing_authority(a.directory)
    assert not module._nested_identity_read(a)


@pytest.mark.parametrize('kind', ['projection', 'mirror', 'event'])
def test_nested_read_does_not_excuse_outer_ledger_corruption(root, monkeypatch, kind):
    a = wallet(root)
    reserve(a)
    original = a._verify_rows
    nested_reads = []

    def probe_then_audit(db, **kwargs):
        nested_reads.append(open_existing_authority(a.directory).manifest_hash)
        if kind == 'projection':
            db.execute('DELETE FROM wallets')
        elif kind == 'event':
            db.execute("UPDATE events SET body='{}'")
        else:
            mirror = a.directory / 'events/0000000001.json'
            body = json.loads(mirror.read_text())
            body['hash'] = 'corrupted'
            mirror.write_text(json.dumps(body))
        return original(db, **kwargs)

    monkeypatch.setattr(a, '_verify_rows', probe_then_audit)
    with pytest.raises(ValueError):
        a.verify()
    assert nested_reads == [a.manifest_hash]
    assert not module._nested_identity_read(a)


def test_scope_rejects_wrong_database_connection(root):
    a, b = wallet(root), wallet(root, 'other')
    with b._transaction() as db, pytest.raises(ValueError, match='original database'), a._identity_read_scope(db):
        pytest.fail('foreign database cannot enter scope')
    assert not module._nested_identity_read(a)
