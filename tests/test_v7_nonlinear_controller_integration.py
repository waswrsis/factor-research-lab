"""Real local S orchestration on tiny artificial data; no Linux/GPT claims."""
import json

import pytest
from test_v7_rolling_registry import candles
from test_v7_run_factor_pool import source_root

from factorlab.discovery.v7_run import V7Config, run_month


def test_nonlinear_configuration_is_explicit_and_separate_from_adoption():
    assert V7Config().nonlinear_mode == 'disabled'
    for mode in ('probes', 'isolated'):
        with pytest.raises(ValueError, match='joint configuration'):
            V7Config(nonlinear_mode=mode)
        assert V7Config(evaluation_mode='joint', nonlinear_mode=mode).formal_confirmation_enabled is False
    with pytest.raises(ValueError):
        V7Config(evaluation_mode='joint', nonlinear_mode='unbounded')


def test_monthly_quarterly_probe_uses_actual_S_checkpoint_and_shared_budget(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    config = V7Config(evaluation_mode='joint', formulas_per_family=1, nonlinear_mode='probes')
    result = run_month('research', candles('2024-08-01', '2025-04-02'),
                       '2025-04-01T00:00:00Z', config=config)
    report = result['nonlinear']
    assert report['status'] == 'S_channel_complete'
    assert report['producers']['status'] == 'completed'
    assert report['gate']['status'] == 'pending_quarterly_evidence'
    assert report['automatic_adoption'] is False and result['A0'] == 'K0'
    assert result['formal_confirmation_enabled'] is False
    import sqlite3
    with sqlite3.connect(tmp_path / 'research/state.sqlite') as db:
        state = json.loads(db.execute('SELECT value FROM months').fetchone()[0])
    assert state['nonlinear_input_data_id']
    assert state['nonlinear']['channel']['counts']['solve'] == 9
    assert state['budget']['counts']['solve'] == result['budget']['counts']['solve']
    receipts = list((tmp_path / 'research/nonlinear_epoch/receipts').glob('*.json'))
    assert any(json.loads(path.read_text()).get('kind') == 'probe' for path in receipts)
    restored = run_month('research', candles('2024-08-01', '2025-04-02'),
                         '2025-04-01T00:00:00Z', config=config)
    assert restored == result
