"""Retain actual SQLite handles so GC cannot mask a read-only handle leak."""
import sqlite3

import pandas as pd
import pytest

from factorlab.discovery import calibration_oracle_evidence as m
from factorlab.discovery.v7_feed import ReceiptArchive


@pytest.mark.parametrize('failure', [False, True])
def test_raw_oracle_connection_closed_after_success_or_verification_failure(tmp_path, monkeypatch, failure):
    feed = ReceiptArchive(tmp_path/'feed', ('BTCUSDT',), clock=lambda: pd.Timestamp.now(tz='UTC'))
    identity = feed.identity
    feed.db.close()
    opened = []
    real_connect = sqlite3.connect

    def tracked(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(m.sqlite3, 'connect', tracked)
    if failure:
        def reject(self):
            raise ValueError('deliberate original feed verification failure')
        monkeypatch.setattr(ReceiptArchive, 'verify', reject)
        with pytest.raises(ValueError, match='deliberate original feed'):
            m._raw_labels(tmp_path/'feed', [], source_id=identity, symbols=['BTCUSDT'], delay=1)
    else:
        m._raw_labels(tmp_path/'feed', [], source_id=identity, symbols=['BTCUSDT'], delay=1)
    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError, match='closed'):
        opened[0].execute('SELECT 1')
    original = tmp_path/'feed/receipts.sqlite'
    original.rename(original.with_suffix('.retained'))
