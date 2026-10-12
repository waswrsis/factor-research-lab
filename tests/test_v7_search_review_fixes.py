"""Boundary regressions, no provider/S/neural execution or statistical evidence.

Real finite-prefix tests use original source, runtime and paid preparation.
Dispatch tests use disclosed thin downstream doubles to isolate routing only.
"""
import ast
import importlib.machinery
import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from test_v7_s_observation_source import factory, ready, original_S_source

ROOT = Path(__file__).resolve().parents[1]


def nested_dispatch(backend):
    from factorlab.discovery import v7_run
    source = Path(v7_run.__file__)
    tree = ast.parse(source.read_text('utf-8'))
    scope = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_run_month_scheduled_impl')
    node = next(n for n in ast.walk(scope) if isinstance(n, ast.FunctionDef) and n.name == 'bounded_backend')
    budget = SimpleNamespace(check_time=lambda: None,
        lease=SimpleNamespace(assert_parent_alive=lambda: SimpleNamespace(remaining_seconds=300)))
    env = {'__package__': 'factorlab.discovery', 'budget': budget, 'backend': backend, 'directory': ROOT/'work'}
    # Preserve the original nonlocal cell; execute the unmodified production
    # nested function inside the same minimal enclosing binding it requires.
    factory = ast.parse('def _dispatch_closure():\n    last_reply = None\n    return bounded_backend, lambda: last_reply\n').body[0]
    factory.body.insert(1, node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[])), str(source), 'exec'), env)
    dispatch, budget.reply_state = env['_dispatch_closure']()
    return dispatch, budget


@pytest.mark.parametrize('kind', ['registered', 'trial_scoped'])
def test_exact_registered_backend_receives_original_family_budget(kind, monkeypatch):
    from factorlab.discovery.full_trial_controller import RegisteredTrialCodexBackend
    from factorlab.discovery.calibration_trial_context import TrialScopedCodexBackend
    cls = RegisteredTrialCodexBackend if kind == 'registered' else TrialScopedCodexBackend
    seen = []
    def reserved(self, packet, *, model, reasoning, work_budget):
        seen.append((self, packet, model, reasoning, work_budget)); return 'routed'
    monkeypatch.setattr(cls, 'call_reserved', reserved)
    backend = object.__new__(cls)
    dispatch, budget = nested_dispatch(backend)
    assert dispatch('packet', model='gpt-6-astra', reasoning='high') == 'routed'
    assert seen == [(backend, 'packet', 'gpt-6-astra', 'high', budget)]
    assert budget.reply_state() == 'routed'


def test_reserved_dispatch_does_not_admit_lookalike_or_subclass():
    from factorlab.discovery.full_trial_controller import RegisteredTrialCodexBackend
    class Lookalike:
        def call_reserved(self, *args, **kwargs):
            pytest.fail('Unregistered reserved backend admitted')
        def __call__(self, *args, **kwargs):
            return 'ordinary_callback'
    dispatch, budget = nested_dispatch(Lookalike())
    assert dispatch(None, model='m', reasoning='r') == 'ordinary_callback'
    assert budget.reply_state() == 'ordinary_callback'
    class Subclass(RegisteredTrialCodexBackend):
        def call_reserved(self, *args, **kwargs):
            pytest.fail('Subclass admitted as registered authority')
    dispatch, budget = nested_dispatch(object.__new__(Subclass))
    with pytest.raises(ValueError, match='original active reserved family work'):
        dispatch(None, model='m', reasoning='r')
    assert budget.reply_state() is None


def test_original_prefix_controls_queue_opportunity_without_changing_wallet(factory, monkeypatch):
    from factorlab.discovery.structural_queue_review import produce_typed_queue_review
    from factorlab.discovery import neural_research_ticket
    clock, generator, permit, runtime, wallet = ready(factory)
    clock.advance(runtime)
    cutoff = runtime.lifecycle.now().normalize().replace(day=1)
    start = cutoff + pd.DateOffset(months=1)
    assert runtime.lifecycle.now() < start < pd.Timestamp(runtime.clock())
    before = wallet.verify()
    reader = SimpleNamespace(get=lambda _: {'status': 'complete', 'cutoff': cutoff.isoformat(),
        'nonlinear_report': {'catalogue': {}}}, close=lambda: None)
    monkeypatch.setattr(neural_research_ticket, '_reader', lambda _: reader)
    # Early date check must continue to actual missing bridge rejection, not
    # manufacture a paid S result or mark this future opportunity missed.
    with pytest.raises(FileNotFoundError):
        produce_typed_queue_review(runtime, wallet, permit.paths['research'], cutoff.strftime('%Y-%m'))
    assert wallet.verify() == before
    # Publish the remaining original finite releases before consuming them.
    # Publication grants no fitting budget and never backdates availability.
    from factorlab.discovery.finite_market_oracle import publish_due, verify_publications
    publish_due(generator, permit, max_new_receipts=24)
    assert verify_publications(generator, permit)['completed_hours'] == list(range(24))
    assert wallet.verify() == before
    clock.advance_batch(runtime, max_events=24)
    assert runtime.lifecycle.now() >= start
    result = produce_typed_queue_review(runtime, wallet, permit.paths['research'], cutoff.strftime('%Y-%m'))
    assert result['status'] == 'missed_original_next_month'
    assert wallet.verify() == before


def test_feedback_uses_exact_paid_S_prefix_and_preserves_future_failure_visibility(factory):
    from dataclasses import asdict
    from factorlab.discovery.v7_run import V7Config
    from factorlab.discovery.s_preparation import SPreparationEnvelope
    from factorlab.discovery.structural_evolution import _feedback_physical_cutoff, _visible_failure
    from factorlab.discovery.nonlinear_runtime import physical_bound_for_prefix
    from factorlab.discovery.quarantined_observation import current_binding
    journal, runtime, wallet, prefix, cutoff, _, reference = original_S_source(factory)
    directory = runtime.directory/'original_S'
    prep = SPreparationEnvelope.prepare(wallet, directory, '2026-01',
        config=asdict(V7Config(symbols=('BTCUSDT',), evaluation_mode='joint')),
        spec=asdict(runtime.spec), source_reference=reference, schedule_hash='a'*64)
    store = SimpleNamespace(directory=directory, exposure=runtime.lifecycle.exposure,
        identity={'feed_identity': runtime.feed.identity})
    bridge = SimpleNamespace(store=store, authority=wallet, month='2026-01',
        state={'cutoff': cutoff.isoformat(), 'search_exposure': {'source': runtime.feed.identity}},
        identity={'authority_directory': str(wallet.directory), 'authority_manifest_hash': wallet.manifest_hash,
                  'preparation_hash': prep.digest})
    before = wallet.verify()
    physical = _feedback_physical_cutoff(bridge, cutoff)
    assert physical == pd.Timestamp(prefix['physical_observation']['wall']) > cutoff
    with pytest.raises(ValueError, match='current original S cutoff'):
        _feedback_physical_cutoff(bridge, cutoff + pd.DateOffset(months=1))
    bridge.state['search_exposure']['source'] = 'foreign'
    with pytest.raises(ValueError, match='original source prefix'):
        _feedback_physical_cutoff(bridge, cutoff)
    bridge.state['search_exposure']['source'] = runtime.feed.identity
    # Actual original paid row is used only to exercise first-observation anchors;
    # this test does not claim an S failure or an eligible numerical result.
    with wallet._transaction() as db:
        row = next(iter(wallet._verify(db)['reservations'].values()))
    assert not _visible_failure(bridge, row, physical)
    journal.advance(runtime)
    later = physical_bound_for_prefix(runtime.lifecycle, current_binding(runtime.lifecycle))
    assert _visible_failure(bridge, row, later)
    assert _feedback_physical_cutoff(bridge, cutoff) == physical
    assert not _visible_failure(bridge, row, physical)
    assert wallet.verify() == before


def test_ordinary_feedback_cutoff_is_unchanged():
    from factorlab.discovery.structural_evolution import _feedback_physical_cutoff
    cutoff = pd.Timestamp('2026-10-01T00:00:00Z')
    bridge = SimpleNamespace(state={'cutoff': cutoff.isoformat(), 'search_exposure': {'source': 'ordinary'}},
        store=SimpleNamespace(directory=ROOT/'work'), month='2026-10', identity={})
    assert _feedback_physical_cutoff(bridge, cutoff) == cutoff


def load_tool(name, path):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec); loader.exec_module(module)
    return module


def test_source_pack_contains_only_owned_harness_and_readme():
    from factorlab.discovery.evidence import source_paths
    pack = load_tool('search_fix_pack', ROOT/'scripts/v7_linux_bootstrap.sh')
    harness = 'tools/linux_full_path/linux_production_path.py'
    assert ROOT/harness in source_paths(ROOT)
    manifest = pack.source_manifest(ROOT)
    assert harness in manifest['files']
    assert 'tools/linux_full_path/README.txt' in manifest['files']
    for forbidden in ('tools/other.py', 'tools/linux_full_path/secret.json',
                      'tools/linux_full_path/test_linux_production_path.py', 'tools/../credentials.json'):
        assert not pack.allowed_name(forbidden)


def test_typed_driver_uses_original_outer_headroom_and_single_fit(monkeypatch):
    from factorlab.discovery.algorithm_validation import AlgorithmValidation
    from factorlab.discovery import v7_autonomous
    driver = load_tool('search_fix_driver', ROOT/'tools/linux_full_path/linux_production_path.py')
    calls = []
    protocol = SimpleNamespace(resolve_fit_gates=lambda: None, pending_work=lambda: True,
        step=lambda **kw: calls.append(kw) or {'status': 'boundary_dispatched'})
    monkeypatch.setattr(driver, 'load_contract', lambda _: {'service': {'wall_seconds': 7200}})
    monkeypatch.setattr(driver, 'originals', lambda _: ('permit', 'generator', 'wallet'))
    monkeypatch.setattr(driver, 'runtime_for', lambda _: nullcontext('runtime'))
    monkeypatch.setattr(driver, 'sys', SimpleNamespace(platform='linux'))
    monkeypatch.setattr(driver, 'time', SimpleNamespace(monotonic=lambda: 1000))
    monkeypatch.setattr(v7_autonomous, 'inherited_limits', lambda: None)
    monkeypatch.setattr(AlgorithmValidation, 'discover', lambda *a: [protocol])
    assert driver.execute(ROOT/'work/unused', 'typed-forward') == {'status': 'boundary_dispatched'}
    assert calls == [{'deadline_monotonic': 8200, 'max_fits': 1}]
    assert calls[0]['deadline_monotonic'] > 1000 + 302
