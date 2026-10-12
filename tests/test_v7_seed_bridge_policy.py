import pandas as pd
import pytest
from test_v7_run_factor_pool import source_root
from test_v7_seed import Session

from factorlab.discovery import v7_seed as seed


def test_explicit_bridge_can_finish_monthly_seed_without_claiming_live_ready(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    monkeypatch.setattr(seed, '_now', lambda: pd.Timestamp('2026-10-29T12:00:00Z'))
    client = Session()
    path = tmp_path / 'seed'
    result = seed.prepare_seed(path, session=client, allow_daily_bridge=True, max_new_archives=8)
    assert result['status'] == 'requires_daily_bridge' and result['seed_artifact_ready']
    assert result['counts']['archives_accepted'] == 8 and len(client.calls) == 16
    assert result['daily_bridge_start'] == '2026-10-01T00:00:00+00:00'
    assert result['daily_bridge_end'] == '2026-10-14T00:00:00+00:00'
    assert result['actual_feed_imported_at'] is None and not result['data_ready']
    assert pd.read_parquet(result['bars_path']).open_time.max() == pd.Timestamp('2026-09-30T23:00Z')
    assert seed.prepare_seed(path, session=client, allow_daily_bridge=True) == {**result,
        'counts': {**result['counts'], 'new_archive_attempts': 0}}


def test_bridge_policy_frozen_before_download_and_no_implicit_budget_change(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    monkeypatch.setattr(seed, '_now', lambda: pd.Timestamp('2026-10-29T12:00:00Z'))
    client = Session()
    path = tmp_path / 'seed'
    assert seed.prepare_seed(path, session=client)['status'] == 'requires_daily_bridge'
    assert not client.calls
    with pytest.raises(ValueError, match='Frozen'):
        seed.prepare_seed(path, session=client, allow_daily_bridge=True)
    assert not client.calls
