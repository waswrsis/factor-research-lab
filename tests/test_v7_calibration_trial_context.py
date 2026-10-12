"""Artificial isolation checks; no real Codex, provider, market or certificate.

Some cases execute the real shared-wallet S controller. Its provider is explicitly
replaced by FakeBatch; separate CLI cases use real local Python child processes.
The small memory journal fixture tests path closure, not a full RunStore proof.
"""
import json
import sqlite3
import sys
import time
from collections import Counter

import pandas as pd
import pytest
from test_v7_feed_audit import Clock
from test_v7_prompt_builder import inputs, response
from test_v7_s_controller_integration import FakeBatch

from factorlab.discovery import calibration_trial_context as context
from factorlab.discovery.calibration_coverage import CoverageHarness, CoveragePlan
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
from factorlab.discovery.neural_forward_ticket import _bind_authority
from factorlab.discovery.prompt_builder import build_formula_prompt
from factorlab.discovery.rolling import RollingSpec
from factorlab.discovery.v7_forward import ForwardRuntime
from factorlab.discovery.v7_memory import ResearchMemory, search_exposure_context
from factorlab.discovery.v7_run import RunStore, V7Config, literature_seeds, monthly_budget
from factorlab.discovery.work_authority_bridge import WorkAuthorityBridge
from factorlab.protocol import fingerprint

pytest_plugins = ['test_v7_s_controller_integration']


def harness(case):
    plan = CoveragePlan((case.cutoff.isoformat(),), trials=('rep-a', 'rep-b'),
                        input_scope='live_receipts', provider_context=context.POLICY)
    return CoverageHarness(case.root/'isolated-coverage', plan,
                           config=case.config, spec=case.runtime.spec)


def actual_input(case, h, trial):
    paths = context.trial_paths(h.directory, trial)
    clock = Clock('2026-09-30T23:30:00Z')
    runtime = ForwardRuntime(paths['exposure'].parent, paths['feed'], clock=clock)
    runtime.feed.import_seed(case.root/'artificial.parquet')
    clock.now = pd.Timestamp('2026-10-01T00:00:10Z')
    wallet = MonthlyWorkAuthority(paths['authority'], run_id='isolated-engineering-'+trial,
                                 task='volatility', universe=['BTCUSDT'])
    _bind_authority(runtime, wallet)
    bars = runtime.feed.snapshot(case.cutoff, start=case.cutoff-pd.DateOffset(months=6)-pd.Timedelta(hours=1234),
                                 archive_delay_seconds=runtime.spec.availability_delay_seconds)
    return paths, runtime, wallet, bars


def call(case, h, trial, data, **kwargs):
    _, runtime, wallet, bars = data
    return h.run_month(trial, bars, case.cutoff,
        feed_directory=runtime.feed.directory, exposure_path=runtime.lifecycle.exposure.path,
        authority=wallet, schedule=case.schedule, **kwargs)


def test_two_real_shared_research_repetitions_cannot_share_backend_or_memory(case, monkeypatch):
    h = harness(case)
    calls, fixtures = [], {}

    def artificial_provider(self, packet, *, model, reasoning):
        # Explicit provider seam; actual numeric S fits/ledgers are unchanged.
        calls.append(str(self.state_dir))
        return fixtures.setdefault(str(self.state_dir), FakeBatch())(packet, model=model, reasoning=reasoning)

    monkeypatch.setattr(context.CodexBatchBackend, '__call__', artificial_provider)
    contexts = []
    for trial in h.plan.trials:
        data = actual_input(case, h, trial)
        try:
            result = call(case, h, trial, data)
            assert result['status'] == 'complete'
            proof = result['evidence']['trial_context']
            assert proof['memory']['months'] == 1
            assert proof['provider_authenticity_certified'] is False
            assert proof['statistical_independence_certified'] is False
            contexts.append(proof['context_hash'])
            memory = json.loads((data[0]['research']/'memories/2026-10.json').read_text())['memory']
            assert memory['predecessor'] is None and memory['book']['predecessor'] is None
            before = data[2].verify()
            assert call(case, h, trial, data) == result
            assert data[2].verify() == before
        finally:
            data[1].close()
    assert len(set(contexts)) == 2
    assert sorted(Counter(calls).values()) == [3, 3]
    assert len(fixtures) == 2 and all(len(v.calls) == 3 for v in fixtures.values())
    report = h.report()
    assert report['reserved_units'] == 2 and report['exercised_paths'] == ['monthly_controller']
    assert report['formal_gate_eligible'] is report['full_V7'] is False
    # Inspection cannot recreate a missing original context marker.
    marker = context.trial_paths(h.directory, 'rep-a')['provider']/'context.json'
    marker.unlink()
    with pytest.raises(context.TrialIsolationError, match='Missing'):
        h.report()
    assert not marker.exists() and len(calls) == 6


def test_reusable_backend_or_external_input_wallet_refused_before_trial_reservation(case):
    h = harness(case)
    args = {'feed_directory': case.runtime.feed.directory,
            'exposure_path': case.runtime.lifecycle.exposure.path,
            'authority': case.authority, 'schedule': case.schedule}
    with pytest.raises(ValueError, match='reusable injected'):
        h.run_month('rep-a', case.bars, case.cutoff, backend=FakeBatch(), **args)
    with pytest.raises(ValueError, match='own its feed'):
        h.run_month('rep-a', case.bars, case.cutoff, **args)
    assert h.report()['reserved_units'] == 0 and case.authority.verify()['reservations'] == 0


def test_provider_failure_consumed_without_new_wrapper_or_call(case, monkeypatch):
    h = harness(case)
    data = actual_input(case, h, 'rep-a')
    calls = []

    def fail(self, *a, **k):
        calls.append(str(self.state_dir))
        raise RuntimeError('Explicit artificial provider outage')

    monkeypatch.setattr(context.CodexBatchBackend, '__call__', fail)
    try:
        with pytest.raises(RuntimeError, match='provider outage'):
            call(case, h, 'rep-a', data)
        before = data[2].verify()
        assert call(case, h, 'rep-a', data)['status'] == 'failed_consumed'
        assert len(calls) == 1 and data[2].verify() == before
        assert h.report()['reserved_units'] == 1
    finally:
        data[1].close()


def memory_journal(root):
    research = root/'research'
    research.mkdir()
    identity = {'scope': 'memory-path-fixture-only'}
    (research/'manifest.json').write_text(json.dumps({'identity': identity, 'hash': fingerprint(identity)}))
    seeds = [{'program': p, 'citation': 'Artificial source seed'} for p in literature_seeds()[:2]]
    first = ResearchMemory.initial('2026-09-01T00:00:00+00:00', seeds)
    first_path = research/'memories/2026-09.json'
    first.save(first_path)
    second = first.rollover('2026-10-01T00:00:00+00:00', first_path)
    second_path = research/'memories/2026-10.json'
    second.save(second_path)
    db = sqlite3.connect(research/'state.sqlite')
    db.execute('CREATE TABLE months(month TEXT PRIMARY KEY,value TEXT,hash TEXT)')
    for month, memory in [('2026-09', first), ('2026-10', second)]:
        state = {'research_memory': memory.to_state()}
        db.execute('INSERT INTO months VALUES(?,?,?)', (month, json.dumps(state), fingerprint(state)))
    db.commit()
    return research, db


def test_genuine_same_trial_rollover_retained_and_historical_scope_stays_stable(tmp_path, monkeypatch):
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    research, db = memory_journal(tmp_path)
    try:
        before = context.verify_memory_namespace(research, through_month='2026-09')
        assert before == {'months': 1, 'memory_shards': 0}
        assert context.verify_memory_namespace(research) == {'months': 2, 'memory_shards': 2}
        state = {'research_memory': None}
        db.execute('INSERT INTO months VALUES(?,?,?)', ('2026-11', json.dumps(state), fingerprint(state)))
        db.commit()
        assert context.verify_memory_namespace(research, through_month='2026-09') == before
    finally:
        db.close()


@pytest.mark.parametrize('kind', ['memory', 'book', 'foreign_evidence', 'malformed_memory'])
def test_cross_trial_history_rejected_before_foreign_file_read(tmp_path, monkeypatch, kind):
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    research, db = memory_journal(tmp_path)
    original_read, observed = context._read, []

    def read(path):
        observed.append(path)
        assert path.is_relative_to(research), 'Guard read another trial before path validation'
        return original_read(path)

    monkeypatch.setattr(context, '_read', read)
    try:
        state = json.loads(db.execute("SELECT value FROM months WHERE month='2026-10'").fetchone()[0])
        value = state['research_memory']['memory']
        if kind == 'foreign_evidence':
            ref = {'run_identity': 'f'*64}
            value['evidence'] = {fingerprint(ref): ref}
        elif kind == 'malformed_memory':
            value['book']['parents'] = 'not-a-parent-map'
        else:
            holder = value if kind == 'memory' else value['book']
            holder['predecessor']['path'] = str(tmp_path/'other-trial'/'valid-looking.json')
        state['research_memory']['digest'] = fingerprint(value)
        db.execute("UPDATE months SET value=?,hash=? WHERE month='2026-10'", (json.dumps(state), fingerprint(state)))
        db.commit()
        with pytest.raises(context.TrialIsolationError):
            context.verify_memory_namespace(research)
        assert all(p.is_relative_to(research) for p in observed)
    finally:
        db.close()


def test_actual_local_children_use_distinct_trial_cli_directories_and_preserve_unknowns(tmp_path, monkeypatch):
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    kwargs = inputs()
    packet = build_formula_prompt(**kwargs)
    events = [{'type': 'turn.started'}, {'type': 'item.completed',
              'item': {'type': 'agent_message', 'text': json.dumps(response(kwargs))}},
              {'type': 'turn.completed', 'usage': {'input_tokens': 11, 'output_tokens': 3}}]
    script = 'import sys; sys.stdin.read(); print(' + repr('\n'.join(json.dumps(e) for e in events)) + ')'
    monkeypatch.setattr(context.CodexBatchBackend, '_command', lambda *a: [sys.executable, '-c', script])
    markers = []
    for trial in ('rep-a', 'rep-b'):
        backend = context.TrialScopedCodexBackend(tmp_path/'coverage', trial, 'f'*64)
        reply = backend(packet, model='gpt-6-astra', reasoning='high')
        assert reply.usage == {'input_tokens': 11, 'output_tokens': 3}
        assert reply.model_resolved is reply.backend_version is None
        paths = context.trial_paths(tmp_path/'coverage', trial)
        record = next((paths['provider']/'attempts').glob('*/attempt.json'))
        assert json.loads(record.read_text())['status'] == 'completed'
        markers.append(record)
        assert context.TrialScopedCodexBackend.inspect_existing(tmp_path/'coverage', trial, 'f'*64)['memory']['months'] == 0
    assert markers[0].parent != markers[1].parent


def test_provider_marker_cannot_be_deleted_to_reset_history(tmp_path, monkeypatch):
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    backend = context.TrialScopedCodexBackend(tmp_path/'coverage', 'rep-a', 'f'*64)
    (backend.paths['provider']/'attempts').mkdir()
    backend.path.unlink()
    with pytest.raises(context.TrialIsolationError, match='unbound attempt'):
        context.TrialScopedCodexBackend(tmp_path/'coverage', 'rep-a', 'f'*64)
    with pytest.raises(context.TrialIsolationError, match='Missing'):
        context.TrialScopedCodexBackend.inspect_existing(tmp_path/'coverage', 'rep-a', 'f'*64)
    assert not backend.path.exists()


def test_reserved_backend_inherits_real_remaining_family_lease(tmp_path, monkeypatch):
    """Actual wallet/RunStore/clock/child; artificial reply and diagnostic input."""
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    (tmp_path/'factorlab').mkdir()
    (tmp_path/'factorlab/frozen.py').write_text('version=1')
    (tmp_path/'pyproject.toml').write_text('[project]')
    (tmp_path/'uv.lock').write_text('version=1')
    backend = context.TrialScopedCodexBackend(tmp_path/'coverage', 'rep-a', 'f'*64)
    config, month = V7Config(evaluation_mode='joint'), '2026-09'
    store = RunStore(backend.paths['research'], config, RollingSpec())
    authority = MonthlyWorkAuthority(backend.paths['authority'], run_id='lease-engineering',
                                     task='volatility', universe=['BTCUSDT'])
    exposure = search_exposure_context('artificial-data', month+'-01T00:00:00+00:00', ['BTCUSDT'])
    store.exposure.record(**{k: v for k, v in exposure.items() if k != 'data_id'}, role='S')
    state = {'status': 'running', 'data_id': 'artificial-data', 'search_exposure': exposure,
             'budget': monthly_budget(config, month).to_dict()}
    store.save(month, state, 'initial')

    def persist(value):
        state['budget'] = value
        store.save(month, state, 'projection')

    bridge = WorkAuthorityBridge(store, month, state, authority, persist)
    work = bridge.begin_work('family', counts={'gpt_attempt': 1}, lease_seconds=5,
                             binding={'stage': 'family', 'input_id': state['data_id']})
    work.consume('gpt_attempt')
    # This real delay represents time already spent in this original work.
    time.sleep(1.1)
    original_run, observed = context.CodexBatchBackend._run, []

    def observed_run(self, *args):
        observed.append(self.timeout_seconds)
        return original_run(self, *args)

    monkeypatch.setattr(context.CodexBatchBackend, '_run', observed_run)
    monkeypatch.setattr(context.CodexBatchBackend, '_command',
                        lambda *a: [sys.executable, '-c', 'import time; time.sleep(8)'])
    before = authority.verify()
    try:
        with pytest.raises(TimeoutError, match='CodexBatchTimeout'):
            backend.call_reserved(build_formula_prompt(**inputs()), model='gpt-6-astra',
                                  reasoning='high', work_budget=work)
        assert len(observed) == 1 and 0 < observed[0] < 4
        assert authority.verify() == before
        assert work.actual_counts['gpt_attempt'] == 1
        attempts = list((backend.paths['provider']/'attempts').glob('*/attempt.json'))
        assert len(attempts) == 1
        assert json.loads(attempts[0].read_text())['error_type'] == 'TimeoutError'
        # A plain duration cannot manufacture original prepaid permission.
        with pytest.raises(context.TrialIsolationError, match='original active'):
            backend.call_reserved(build_formula_prompt(**inputs()), model='gpt-6-astra',
                                  reasoning='high', work_budget=5.)
        assert len(list((backend.paths['provider']/'attempts').glob('*/attempt.json'))) == 1
    finally:
        store.close()
