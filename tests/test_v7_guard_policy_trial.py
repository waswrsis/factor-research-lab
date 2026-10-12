"""Canonical prepaid policy dispatch; artificial selection, no data/model/provider.

The _role unit is called at the actual reserved preparation boundary. Native
Campaign/cgroup handshake is not simulated as passed by these Windows tests.
"""
import copy
import importlib.util
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from factorlab import protocol
from factorlab.discovery import typed_policy_confirmation as policy
from factorlab.discovery.claim_protocol import ClaimProtocol, FrozenClaimEndpoint
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
from factorlab.discovery.neural_forward import FrozenNeuralRecipe
from factorlab.discovery.neural_forward_ticket import _bind_authority, issue_forward_ticket
from factorlab.discovery.v7_forward import ForwardRuntime


@pytest.fixture
def original(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('guard_policy_test', root/'scripts/v7_guard/multiwork_enforcer.py')
    guard = importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)
    monkeypatch.setattr(protocol, 'ROOT', tmp_path)
    (tmp_path/'factorlab').mkdir(); (tmp_path/'factorlab/source.py').write_text('artificial_fixture=1')
    (tmp_path/'pyproject.toml').write_text('[project]'); (tmp_path/'uv.lock').write_text('version=1')
    clock = [pd.Timestamp('2026-10-31T00:00:00Z')]
    runtime = ForwardRuntime(tmp_path/'runtime', tmp_path/'feed', clock=lambda:clock[0])
    wallet = MonthlyWorkAuthority(tmp_path/'wallet', run_id='guard-policy', task='volatility',
                                  universe=['BTCUSDT'], partition_policy='standard')
    _bind_authority(runtime, wallet)
    recipe = FrozenNeuralRecipe(rolling_spec=runtime.spec)
    k0 = runtime.recipe_for(); candidate = runtime.recipe_for(recipe)
    claim = ClaimProtocol.freeze(runtime.lifecycle, [
        FrozenClaimEndpoint.library('2026-11-01T00:00:00Z', k0,
            {'pipeline_id':protocol.fingerprint('artificial-benchmark'),'benchmark':'fixture'}),
        FrozenClaimEndpoint.algorithm('2026-11-01T00:00:00Z', k0)])
    clock[0] = pd.Timestamp('2026-11-02T00:00:00Z')
    # Isolated selector fixture; actual selection dual files/journal, six-month
    # claim and exact full recipe context remain production verifiers.
    monkeypatch.setattr(policy, '_selection', lambda *a:(candidate, {'artificial_selector':True}))
    assert policy.candidate_pipeline(runtime.lifecycle, '2026-11-01T00:00:00Z', k0) == candidate
    job = runtime.lifecycle.plan_fit('2026-11-01T00:00:00Z', 'A0_policy_trial', candidate)
    scope = {'role':'forward', 'reference':{'kind':'campaign'}, 'directory':str(runtime.directory),
             'authority':{'directory':str(wallet.directory), 'manifest_hash':wallet.manifest_hash}}
    captured = {}
    class EndPreparation(Exception): pass
    def at_reserved(actual_job):
        assert actual_job == job
        work = 'neural_forward:'+protocol.fingerprint({'plan':runtime.lifecycle.plan.digest,'job':job})
        with sqlite3.connect(wallet.database) as db:
            rid = db.execute('SELECT id FROM reservations WHERE work_key=?', (work,)).fetchone()[0]
        record = wallet.get(rid)
        assert record['status'] == 'reserved'
        guard._role(scope, record)
        captured['record'] = record
        raise EndPreparation('no snapshot/model/provider is run')
    monkeypatch.setattr(runtime, 'prepare_fit_snapshot', at_reserved)
    try:
        with pytest.raises(EndPreparation):
            issue_forward_ticket(runtime, wallet, job, recipe, purpose='policy_trial', gates={})
        yield guard, runtime, wallet, scope, captured['record'], claim
    finally:
        runtime.close()


def test_original_policy_trial_dispatch_keeps_paid_work_and_no_A0_permission(original):
    guard, runtime, wallet, scope, record, claim = original
    assert wallet.get(record['reservation_id'])['status'] == 'failed_consumed'
    assert wallet.verify()['reservations'] == 1
    assert record['lease']['lease_seconds'] == 300
    assert claim.endpoints['algorithm'].end == '2027-05-01T00:00:00+00:00'
    assert policy.policy_context(runtime.lifecycle, record['binding']['job'],
        FrozenNeuralRecipe(rolling_spec=runtime.spec))['alpha'] == .01
    before = wallet.verify()
    with pytest.raises(ValueError, match='cannot reset or retry'):
        issue_forward_ticket(runtime, wallet, record['binding']['job'], FrozenNeuralRecipe(rolling_spec=runtime.spec),
                             purpose='policy_trial', gates={})
    assert wallet.verify() == before


@pytest.mark.parametrize('mutation', ['runtime','manifest','scope_kind','S_scope','cost','lease','prework','purpose'])
def test_policy_trial_cannot_borrow_or_relabel_original_scope(original, mutation):
    guard, runtime, wallet, scope, record, _ = original
    scope, record = copy.deepcopy(scope), copy.deepcopy(record)
    if mutation == 'runtime': scope['directory'] += '-foreign'
    elif mutation == 'manifest': scope['authority']['manifest_hash'] = '0'*64
    elif mutation == 'scope_kind': scope['reference']['kind'] = 'S'
    elif mutation == 'S_scope': scope['role'] = 'S'
    elif mutation == 'cost': record['counts']['neural_fit'] += 1
    elif mutation == 'lease': record['lease']['lease_seconds'] = 299
    elif mutation == 'prework': record['binding']['prework_terms'] = '0'*64
    else: record['binding']['purpose'] = 'caller_approved_policy'
    with pytest.raises((ValueError, KeyError)):
        guard._role(scope, record)
    assert wallet.verify()['reservations'] == 1


def test_original_policy_claim_cannot_disappear(original):
    guard, runtime, wallet, scope, record, claim = original
    path = claim.directory/'protocol.json'
    path.rename(path.with_suffix('.preserved'))
    with pytest.raises((ValueError, FileNotFoundError)):
        guard._role(scope, record)
