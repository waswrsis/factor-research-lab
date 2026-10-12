"""Pure scheduling seams: actual candidate methods, no fitting or authority.

Only orchestration dependencies are artificial. The real _fit_pending SQL,
cutoff comparison, attempt accounting, allowance and headroom checks execute.
No runtime/source identity or model acceptance is claimed by these tests.
"""
import importlib.util
import json
import sqlite3
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from factorlab.discovery.joint_pool import FrozenPool


@pytest.mark.parametrize(('future', 'defer'), [(False, False), (True, False), (False, True)])
def test_planning_clock_dispatches_only_current_due_work(monkeypatch, future, defer):
    path = Path(__file__).resolve().parents[1] / 'factorlab/discovery/v7_service.py'
    name = 'factorlab.discovery._planning_clock_candidate'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    from factorlab.discovery import forward_service_scope, nominal_algorithm_policy
    monkeypatch.setattr(forward_service_scope, 'check_service_work', lambda *a, **k: None)
    monkeypatch.setattr(nominal_algorithm_policy, 'prepare_tick', lambda *a: None)
    monkeypatch.setattr(nominal_algorithm_policy, 'after_tick', lambda *a: None)
    initial = pd.Timestamp('2026-10-12T03:38:03+00:00')
    clock = [initial]
    db = sqlite3.connect(':memory:')
    db.execute('CREATE TABLE jobs(id,cutoff,stream,pipeline,status,failure)')
    service = object.__new__(module.ForwardService)
    lifecycle = SimpleNamespace(db=db, now=lambda: clock[0], controller_lock=nullcontext)
    called, boundaries, saved = [], [], []
    service.lifecycle = lifecycle
    service.runtime = SimpleNamespace(tick=lambda **kw: {'at': initial.isoformat()},
        fit_month=lambda cutoff, **kw: called.append((cutoff, kw['stream'])) or {'job': 'first'})
    service.config = module.ServiceConfig(wall_seconds=7200, max_fits_per_step=1, max_fit_attempts=1, poll=False)
    service.guard = lambda: {'allowed': True}
    service.monotonic = lambda: 100.0
    service.state = {'steps': 0, 'spent_seconds': 0.0, 'fit_attempts': {}}
    service._save = saved.append
    service._freeze_due = lambda now, actions: boundaries.append(('freeze', now))
    service._endpoints = lambda now, actions: boundaries.append(('endpoints', now))
    service._assert_claim_protocol = lambda: None
    service._sync_algorithm_protocols = lambda: []
    service.algorithm_protocols = []
    service.work_authority = None
    service._pool = lambda recipe: FrozenPool(role='reference', version='K0_v1')
    service._algorithm_step = lambda *a, **k: 0
    def plan(now):
        assert now == initial
        clock[0] += pd.Timedelta(seconds=10)
        cutoff = clock[0] + pd.Timedelta(seconds=int(future))
        db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?)',
                   ('first', cutoff.isoformat(), 'A0', json.dumps({}), 'planned', None))
    service._plan_month = plan
    try:
        result = service._step_owned(defer_fits=defer)
        expected = not future and not defer
        assert called == ([(clock[0].isoformat(), 'A0')] if expected else [])
        assert boundaries == [('freeze', initial), ('endpoints', initial)]
        assert result['at'] == initial.isoformat()
        assert service.state['fit_attempts'] == ({'first': {'count': 1, 'last_at': clock[0].isoformat()}} if expected else {})
        assert len(result['actions']) == int(expected)
        assert service.state['steps'] == 1 and service.state['active_since'] is None
        assert saved[-1] == 'step_finished'
    finally:
        db.close()
