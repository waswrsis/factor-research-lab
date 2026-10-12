"""Shared pool/NN C binding on an original real quarantine endpoint chain.

The endpoint fixture deliberately uses recorded artificial predictions and
future LC clocks. No model work or quota is granted by this shared read helper.
"""
import json

import pandas as pd
import pytest
from test_v7_quarantine_execution import case  # noqa: F401 -- original actual fixture
from test_v7_quarantine_execution import (
    test_actual_C_kernel_quarantine_adoption_and_readonly_recovery as _exercise_actual_case,
)

from factorlab.discovery.neural_forward_contract import _confirmed_adoption_binding
from factorlab.discovery.v7_forward import ForwardRuntime
from factorlab.protocol import fingerprint


def test_shared_pool_neural_adoption_reads_original_quarantine_C(case, monkeypatch):  # noqa: F811 -- pytest imported fixture
    _exercise_actual_case(case, monkeypatch)
    registry, runtime, wallet, permit, clock = case
    lc = runtime.lifecycle
    candidate = {k: v for k, v in lc._batch('A')[0]['candidates'][0].items() if k != 'complexity'}
    budget = wallet.verify()
    bound = _confirmed_adoption_binding(lc, clock[0], candidate)
    assert bound['batch'] == 'A'
    assert bound['pipeline_id'] == candidate['pipeline_id']
    assert not lc.calibration_ready()
    with pytest.raises(ValueError, match='completed adoption'):
        _confirmed_adoption_binding(lc, '2027-03-01T00:00:00Z', candidate)
    with pytest.raises(ValueError, match='Actual completed policy adoption cutoff/binding required'):
        _confirmed_adoption_binding(lc, '2027-05-01T00:00:00Z', candidate)
    with pytest.raises(ValueError, match='recipe differs'):
        _confirmed_adoption_binding(lc, clock[0], {**candidate, 'unexpected': 'caller_metadata'})
    with permit.open_lifecycle() as reader:
        reader.clock = lambda: pd.Timestamp('2027-05-01T00:00:00Z')
        assert _confirmed_adoption_binding(reader, clock[0], candidate, expected=bound, historical=True) == bound
        with pytest.raises(ValueError, match='differs from the original ticket'):
            _confirmed_adoption_binding(reader, clock[0], candidate,
                expected={**bound, 'result_hash': fingerprint('forged')}, historical=True)
    foreign = ForwardRuntime(registry.directory.parent/'foreign-adoption',
                             registry.directory.parent/'foreign-adoption-feed', clock=lambda: clock[0])
    try:
        with pytest.raises(ValueError, match='verified adoption'):
            _confirmed_adoption_binding(foreign.lifecycle, clock[0], candidate, expected=bound, historical=True)
    finally:
        foreign.close()
    original = lc.db.execute("SELECT result FROM batches WHERE id='A'").fetchone()[0]
    altered = {**json.loads(original), 'supported': True, 'formal_eligible': True}
    with lc.db:
        lc.db.execute("UPDATE batches SET result=? WHERE id='A'", (json.dumps(altered),))
    try:
        with pytest.raises(ValueError):
            _confirmed_adoption_binding(lc, clock[0], candidate)
    finally:
        with lc.db:
            lc.db.execute("UPDATE batches SET result=? WHERE id='A'", (original,))
    assert wallet.verify() == budget
