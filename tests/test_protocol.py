from dataclasses import replace

import pandas as pd
import pytest

from factorlab.protocol import Protocol, project_path
from factorlab.splits import outer_folds, segment_masks


def test_frozen_protocol_and_workspace():
    p = Protocol()
    assert replace(p, fee_bps=6).digest != p.digest
    with pytest.raises(ValueError):
        project_path('C:/outside.json')
    with pytest.raises(ValueError):
        project_path('../outside.json')
    with pytest.raises(ValueError):
        Protocol(live_trading=True)


def test_purge_uses_outcome_maturity_not_decision_row():
    f = outer_folds(Protocol())[0]
    t = f.validation_start
    meta = pd.DataFrame({'available_at': [t-pd.Timedelta(hours=15), t-pd.Timedelta(hours=1), t],
                         'label_available_at': [t-pd.Timedelta(hours=1), t, t+pd.Timedelta(hours=13)]})
    masks = segment_masks(meta, f)
    assert masks['train'].tolist() == [True, False, False]
    assert masks['search'].tolist() == [False, False, True]
