"""Real local physical writes; no provider, training, Linux or formal evidence."""
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

from factorlab.discovery import finite_market_oracle as finite
from factorlab.discovery.calibration_acceptance import AcceptanceCell, AcceptancePlan
from factorlab.discovery.calibration_runtime_contract import _original_operations, runtime_calibration_contract
from factorlab.discovery.full_calibration_registry import FullTrialPlan, FullTrialRegistry
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
from factorlab.discovery.neural_forward_ticket import _bind_authority
from factorlab.discovery.v7_forward import ForwardRuntime
from factorlab.protocol import fingerprint

# Canonical package name, relative production imports and real dependencies.
MODULE = 'factorlab.discovery.quarantined_observation'
local_source = Path(__file__).with_name('quarantined_observation.py')
if local_source.is_file():
    spec = importlib.util.spec_from_file_location(MODULE, local_source)
    observation = importlib.util.module_from_spec(spec)
    sys.modules[MODULE] = observation
    spec.loader.exec_module(observation)
else:
    observation = importlib.import_module(MODULE)
Q = observation.QuarantinedObservation


@pytest.fixture
def tmp_path(tmp_path_factory):
    # Preserve actual Windows path semantics without exceeding MAX_PATH.
    return tmp_path_factory.mktemp('t')


@pytest.fixture
def factory(tmp_path, monkeypatch):
    from factorlab import protocol
    repo = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
    originals = {p: (repo/p).read_bytes() for p in finite.PROOF_FILES}
    for name in ('v7_lifecycle', 'forward_fit_snapshot', 'neural_forward_ticket', 'neural_forward_contract',
                 'v7_forward', 'rolling', 'pool_forward', 'known_linear_forward', 'quarantined_observation',
                 'finite_market_oracle', 'nonlinear_runtime', 's_nonlinear_gates', 'forward_preparation'):
        path = Path(importlib.import_module('factorlab.discovery.'+name).__file__)
        originals['factorlab/discovery/'+path.name] = path.read_bytes()
    monkeypatch.setattr(protocol, 'ROOT', tmp_path)
    (tmp_path/'factorlab').mkdir()
    (tmp_path/'factorlab/v7.py').write_text('version=7')
    (tmp_path/'pyproject.toml').write_text('[project]')
    (tmp_path/'uv.lock').write_text('version=1')
    for name, body in originals.items():
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    profile = ForwardRuntime(tmp_path/'profile', tmp_path/'profile-feed')
    wallet = MonthlyWorkAuthority(tmp_path/'profile-wallet', run_id='profile', task='volatility',
                                  universe=['BTCUSDT'], partition_policy='standard')
    _bind_authority(profile, wallet)
    scope = runtime_calibration_contract(profile.lifecycle)
    profile.close()
    opened = []

    def make(index=0, gap_probability=0., *, future=False, law_start='2026-01-31T12:00:00Z', hours=24):
        root = tmp_path/f'case-{index}'
        start = (pd.Timestamp.now(tz='UTC').floor('h')+pd.Timedelta(days=2) if future
                 else pd.Timestamp(law_start))
        law = finite.FiniteMarketLaw(start.isoformat(), hours=hours, gap_probability=gap_probability)
        acceptance = AcceptancePlan((AcceptanceCell('finite', law.digest, 1000),))
        gen = finite.FiniteGenerator(root/'generator', law, acceptance)
        plan = FullTrialPlan(acceptance, scope,
            {'backend': 'codex', 'model_requested': 'gpt-6-astra', 'reasoning_requested': 'high',
             'context_policy': 'fresh_codex_per_trial_v1'}, law.domains)
        registry = FullTrialRegistry(root/'registry', plan)
        paths = registry.paths('finite', 0, oracle_index=0)
        runtime = ForwardRuntime(paths['runtime'], paths['feed'])
        trial = registry.identity_for('finite', 0, oracle_index=0)
        wallet = MonthlyWorkAuthority(paths['authority'],
            run_id=fingerprint({'registry': registry.digest, 'trial': trial['trial_id']}),
            task='volatility', universe=['BTCUSDT'], partition_policy='standard')
        permit = registry.enroll(runtime, wallet, 'finite', 0, oracle_index=0)
        opened.append(runtime)
        return gen, permit, runtime, wallet
    yield make
    for runtime in opened:
        runtime.close()


def ready(factory):
    gen, permit, runtime, wallet = factory()
    clock = Q.bind(runtime, gen, permit)
    finite.publish_due(gen, permit)
    return clock, gen, permit, runtime, wallet


def test_original_logical_event_physical_receipt_and_unmodified_wallet(factory):
    clock, gen, permit, runtime, wallet = ready(factory)
    budget = wallet.verify()
    before = runtime.feed.snapshot(pd.Timestamp.now(tz='UTC'))
    binding = clock.advance(runtime)
    source = gen.read_atom(0)['schedule'][0]
    assert binding['logical_at'] == source['release_not_before']
    assert pd.Timestamp(binding['physical_observation']['wall']) > pd.Timestamp(binding['logical_at'])
    assert clock.verify_binding(binding) == binding
    assert binding['formal_gate_eligible'] is False
    assert binding['work_budget_granted'] is False
    assert wallet.verify() == budget
    after = runtime.feed.snapshot(pd.Timestamp.now(tz='UTC'))
    assert before.received_at.tolist() == after.received_at.tolist()
    assert set(inspect.signature(clock.advance).parameters) == {'runtime'}


def test_no_caller_clock_can_advance_unpublished_event(factory):
    gen, permit, runtime, _ = factory()
    clock = Q.bind(runtime, gen, permit)
    runtime.lifecycle.clock = lambda: pd.Timestamp('2099-01-01T00:00:00Z')
    runtime.feed.clock = runtime.lifecycle.clock
    with pytest.raises(ValueError, match='no completed actual receipt'):
        clock.advance(runtime)
    with pytest.raises(TypeError):
        clock.advance(runtime, logical_at='2099-01-01T00:00:00Z')
    assert clock.inspect()['events'] == 0


def test_future_original_release_cannot_be_exposed(factory):
    gen, permit, runtime, _ = factory(future=True)
    clock = Q.bind(runtime, gen, permit)
    assert finite.publish_due(gen, permit)['issued'] == []
    with pytest.raises(ValueError, match='no completed actual receipt'):
        clock.advance(runtime)


def test_reopen_is_readonly_and_never_reissues_completed_prefix(factory):
    clock, gen, permit, runtime, wallet = ready(factory)
    binding = clock.advance(runtime)
    operations = _original_operations(runtime.lifecycle)
    reopened = Q.bind(runtime, gen, permit)
    assert reopened.binding(0) == binding
    assert _original_operations(runtime.lifecycle) == operations
    with pytest.raises(ValueError, match='no completed actual receipt'):
        reopened.advance(runtime)


def test_cross_trial_binding_and_runtime_rejected(factory):
    clock, _, _, runtime, _ = ready(factory)
    binding = clock.advance(runtime)
    gen2, permit2, runtime2, _ = factory(1)
    other = Q.bind(runtime2, gen2, permit2)
    finite.publish_due(gen2, permit2)
    other.advance(runtime2)
    with pytest.raises(ValueError, match='another trial'):
        other.verify_binding(binding)
    with pytest.raises(TypeError, match='original actual runtime'):
        clock.advance(runtime2)


def test_rehashed_physical_witness_cannot_replace_original_lifecycle_anchor(factory):
    clock, _, _, runtime, _ = ready(factory)
    clock.advance(runtime)
    # Even changing both local commit copies must still match the original LC.
    for root in (clock.directory, clock.mirror):
        path = root/'commits/00000.json'
        body = json.loads(path.read_text())
        body['detail']['physical_observation']['monotonic'] -= .01
        path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match='physical witness changed'):
        clock.inspect()


def test_late_event_rewrite_even_identical_bytes_is_rejected(factory):
    clock, _, _, runtime, _ = ready(factory)
    clock.advance(runtime)
    path = clock.directory/'events/00000.json'
    old = path.stat()
    os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns+1_000_000_000))
    with pytest.raises(ValueError, match='physical publication order'):
        clock.inspect()


def test_late_source_commit_even_identical_content_is_rejected(factory):
    clock, _, permit, runtime, _ = ready(factory)
    clock.advance(runtime)
    path = permit.paths['root']/'generator_publications/00000-complete.json'
    old = path.stat()
    os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns+1_000_000_000))
    with pytest.raises(ValueError, match='source, prefix or physical witness changed'):
        clock.inspect()


def test_interrupted_after_event_write_is_not_recovered_with_new_time(factory, monkeypatch):
    clock, _, _, runtime, wallet = ready(factory)
    original = observation._event
    def crash(*args, **kwargs):
        raise OSError('interrupted before LC commit')
    monkeypatch.setattr(observation, '_event', crash)
    with pytest.raises(OSError, match='interrupted'):
        clock.advance(runtime)
    monkeypatch.setattr(observation, '_event', original)
    budget = wallet.verify()
    with pytest.raises(ValueError, match='Orphan or missing'):
        clock.advance(runtime)
    assert wallet.verify() == budget


def test_interrupted_after_sql_commit_missing_mirror_is_not_repaired(factory, monkeypatch):
    clock, _, _, runtime, _ = ready(factory)
    original = observation.immutable_record
    def crash(path, value):
        if path == clock.mirror/'commits/00000.json':
            raise OSError('interrupted mirror commit')
        return original(path, value)
    monkeypatch.setattr(observation, 'immutable_record', crash)
    with pytest.raises(OSError, match='mirror commit'):
        clock.advance(runtime)
    monkeypatch.setattr(observation, 'immutable_record', original)
    with pytest.raises(ValueError, match='Orphan or missing'):
        clock.advance(runtime)


def test_reordered_prefix_is_rejected_even_if_both_files_match(factory):
    clock, gen, permit, runtime, _ = ready(factory)
    finite.publish_due(gen, permit)
    clock.advance(runtime)
    clock.advance(runtime)
    for root in (clock.directory, clock.mirror):
        a, b = root/'events/00000.json', root/'events/00001.json'
        first, second = a.read_bytes(), b.read_bytes()
        a.write_bytes(second)
        b.write_bytes(first)
    with pytest.raises(ValueError, match='source, prefix or physical witness changed'):
        clock.inspect()


@pytest.mark.parametrize('field', ['wall', 'monotonic'])
def test_physical_clock_rollback_refused_before_new_event(factory, monkeypatch, field):
    clock, gen, permit, runtime, _ = ready(factory)
    clock.advance(runtime)
    finite.publish_due(gen, permit)
    original = observation._physical
    previous = clock.inspect()['physical_observation']
    def rollback():
        value = original()
        value[field] = ((pd.Timestamp(previous[field])-pd.Timedelta(seconds=1)).isoformat()
                        if field == 'wall' else previous[field]-1.)
        return value
    monkeypatch.setattr(observation, '_physical', rollback)
    with pytest.raises(ValueError, match='clock moved backward'):
        clock.advance(runtime)
    assert clock.inspect()['events'] == 1


def test_original_gap_advances_without_invented_receipt(factory):
    gen, permit, runtime, _ = factory(gap_probability=.1)
    clock = Q.bind(runtime, gen, permit)
    finite.publish_due(gen, permit, max_new_receipts=24)
    atom = gen.read_atom(0)
    gaps = [s['hour'] for s in atom['schedule'] if s['gap']]
    assert gaps, 'Deterministic original seed should cover an actual law gap'
    for _ in range(gaps[0]+1):
        binding = clock.advance(runtime)
    value = json.loads((clock.directory/'events'/f'{gaps[0]:05d}.json').read_text())['source']
    assert value['receipt_hash'] is None and value['physical_received_at'] is None
    assert value['source_publications'] == []
    assert binding['logical_at'] == atom['schedule'][gaps[0]]['release_not_before']


def test_retrospective_enrollment_refused(factory):
    gen, permit, runtime, _ = factory()
    finite.publish_due(gen, permit)
    with pytest.raises(ValueError, match='retrospectively'):
        Q.bind(runtime, gen, permit)


def test_physical_receipt_archive_still_refuses_logical_history(factory):
    from factorlab.discovery.v7_feed import SnapshotUnavailable
    clock, _, _, runtime, _ = ready(factory)
    binding = clock.advance(runtime)
    with pytest.raises(SnapshotUnavailable):
        runtime.feed.snapshot(binding['logical_at'])


def test_production_worker_snapshot_rejects_quarantine_availability(factory):
    from factorlab.discovery.forward_fit_snapshot import PreparedForwardSnapshot
    clock, _, _, runtime, _ = ready(factory)
    binding = clock.advance(runtime)
    record = {'cutoff': binding['logical_at'], 'availability': binding['availability']}
    with pytest.raises(ValueError, match='Forward causal input'):
        PreparedForwardSnapshot(None, record, fingerprint(record)).verify()


def test_production_ticket_replay_refuses_relabelled_observation_snapshot(factory):
    from factorlab.discovery.known_linear_reference import FrozenLinearRecipe
    from factorlab.discovery.known_linear_forward import pipeline_for
    from factorlab.discovery.neural_forward_ticket import _snapshot_from_actual_source
    from factorlab.discovery.evidence import file_hash
    clock, _, _, runtime, _ = ready(factory)
    binding = clock.advance(runtime)
    recipe = FrozenLinearRecipe('K0')
    job = runtime.lifecycle.plan_fit(pd.Timestamp(binding['logical_at']).normalize().replace(day=1),
                                     'A0', pipeline_for(recipe))
    record = {'version': 'v7-forward-fit-snapshot-v3', 'raw_hash_rule': 'typed-frame-columns-v1',
              'job': job, 'pipeline_id': recipe.pipeline_id, 'plan_id': runtime.lifecycle.plan.digest,
              'availability': binding['availability'], 'observation_prefix': binding}
    record['snapshot_id'] = fingerprint(record)
    path = runtime.directory/'snapshots'/f'{job}.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='Forward snapshot identity changed'):
        _snapshot_from_actual_source(runtime.lifecycle,
            {'job': job, 'path': str(path), 'sha256': file_hash(path)}, recipe,
            expected_pipeline=pipeline_for(recipe))


def test_canonical_lifecycle_and_runtime_read_original_prefix_not_caller_clock(factory):
    clock, _, _, runtime, _ = ready(factory)
    prefix = clock.advance(runtime)
    runtime.clock = lambda: pd.Timestamp('2099-01-01T00:00:00Z')
    runtime.lifecycle.clock = runtime.clock
    assert runtime.lifecycle.now() == pd.Timestamp(prefix['logical_at'])
    assert observation.current_binding(runtime.lifecycle) == prefix
    bars = runtime._snapshot(prefix['logical_at'])
    assert bars.attrs['availability_mode'] == 'quarantined_finite_observation'
    assert bars.iloc[0].received_at > bars.iloc[0].available_at
    assert observation.verify_frame(bars) == prefix


def test_caller_forged_frame_and_physical_provenance_rejected(factory):
    from factorlab.discovery.rolling import build_rolling_data
    clock, _, _, runtime, _ = ready(factory)
    prefix = clock.advance(runtime)
    bars = runtime._snapshot(prefix['logical_at'])
    bars.loc[0, 'received_at'] = bars.loc[0, 'available_at']
    with pytest.raises(ValueError, match='original observed law'):
        build_rolling_data(bars, runtime.spec)
    with pytest.raises(ValueError, match='exact original reconstruction'):
        bars.attrs = {'availability_mode': 'quarantined_finite_observation'}
        build_rolling_data(bars, runtime.spec)


def test_short_original_law_cannot_bypass_paid_training_support(factory):
    from factorlab.discovery.known_linear_forward import pipeline_for, _context, issue_known_ticket
    from factorlab.discovery.known_linear_reference import FrozenLinearRecipe
    from factorlab.discovery.neural_forward_ticket import _snapshot_from_actual_source
    from factorlab.discovery.evidence import file_hash
    gen, permit, runtime, wallet = factory(law_start='2026-01-31T22:00:00Z')
    clock = Q.bind(runtime, gen, permit)
    finite.publish_due(gen, permit, max_new_receipts=3)
    clock.advance(runtime)
    prefix = clock.advance(runtime)
    recipe = FrozenLinearRecipe('K0', runtime.spec)
    job = runtime.lifecycle.plan_fit('2026-02-01T00:00:00Z', 'A0', pipeline_for(recipe))
    assert observation.job_binding(runtime.lifecycle, job) == prefix
    assert _context(runtime.lifecycle, job, recipe)['observation_prefix'] == prefix
    budget = wallet.verify()
    with pytest.raises(ValueError, match='Insufficient mature, fully warmed'):
        issue_known_ticket(runtime, wallet, job, recipe)
    assert wallet.verify()['reservations'] == budget['reservations']+1


def test_real_paid_ticket_from_1033_original_receipts(factory):
    from factorlab.discovery.known_linear_forward import pipeline_for, issue_known_ticket
    from factorlab.discovery.known_linear_reference import FrozenLinearRecipe
    gen, permit, runtime, wallet = factory(law_start='2025-12-20T00:00:00Z', hours=1033)
    clock = Q.bind(runtime, gen, permit)
    issued = 0
    while issued < 1033:
        result = finite.publish_due(gen, permit, max_new_receipts=96)
        assert result['issued']
        issued += len(result['issued'])
    consumed = 0
    while consumed < 1032:
        result = clock.advance_batch(runtime, max_events=min(96, 1032-consumed))
        consumed += result['events_committed']
        assert result['lifecycle_commits'] == result['events_committed']
    prefix = clock.binding(1031)
    recipe = FrozenLinearRecipe('K0', runtime.spec)
    job = runtime.lifecycle.plan_fit('2026-02-01T00:00:00Z', 'A0', pipeline_for(recipe))
    ticket = issue_known_ticket(runtime, wallet, job, recipe)
    _, reservation, prepared, data, _ = ticket.verify(status='reserved')
    assert prepared['training']['rows'] >= 240
    assert prepared['observation_prefix'] == prefix
    assert ticket.value['context']['observation_prefix'] == prefix
    assert data.availability_mode == 'quarantined_finite_observation'
    assert reservation['actual_month'] == reservation['lease']['wall_started_at'][:7]
    assert reservation['actual_month'] != '2026-02'
    assert reservation['counts']['solve'] == 1
    clock.advance(runtime)
    assert observation.job_binding(runtime.lifecycle, job) == prefix
    # Positive original ticket reopens without another grant or a new prefix.
    assert issue_known_ticket(runtime, wallet, job, recipe).digest == ticket.digest


def test_non_enrolled_runtime_cannot_borrow_prefix(factory):
    clock, _, _, runtime, _ = ready(factory)
    prefix = clock.advance(runtime)
    foreign = ForwardRuntime(runtime.directory.parent/'foreign', runtime.feed.directory.parent/'foreign-feed')
    try:
        assert observation.current_binding(foreign.lifecycle) is None
        with pytest.raises(ValueError, match='original enrolled observation'):
            observation.snapshot_for_prefix(foreign.lifecycle, prefix, prefix['logical_at'])
    finally:
        foreign.close()


def test_physical_publication_primitive_needs_later_original_release(factory):
    # This tests the publication primitive, not a concluded C result or model.
    from factorlab.discovery.v7_storage import immutable_record
    clock, gen, permit, runtime, _ = ready(factory)
    first = clock.advance(runtime)
    key = 'publication-primitive-only'
    path = runtime.directory/'claim_publications'/f'{fingerprint(key)}.json'
    immutable_record(path, {'engineering_document': True})
    proof = observation.seal_publication(runtime.lifecycle, 'confirmation', key)
    assert proof['observation_prefix'] == first
    assert observation.publication_visible(runtime.lifecycle, 'confirmation', key, first['logical_at']) is False
    finite.publish_due(gen, permit)
    second = clock.advance(runtime)
    assert observation.publication_visible(runtime.lifecycle, 'confirmation', key, second['logical_at']) is True
    # It remains ineligible at its earlier original logical origin after reopen.
    assert observation.publication_visible(runtime.lifecycle, 'confirmation', key, first['logical_at']) is False
    old = path.stat()
    os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns+1_000_000_000))
    with pytest.raises(ValueError, match='physical publication files changed'):
        observation.read_publication(runtime.lifecycle, 'confirmation', key)


def test_neural_gate_original_job_separates_logical_context_and_physical_publication(factory):
    # Receipt-availability primitive only; no claimed positive probe or genuine
    # resource receipt. test_only remains True and cannot enable the full gate.
    from factorlab.discovery import nonlinear_runtime as gates
    from factorlab.discovery.neural_forward import FrozenNeuralRecipe
    from factorlab.discovery.neural_forward_contract import neural_pipeline
    from factorlab.discovery.rolling import build_rolling_data
    gen, permit, runtime, _ = factory(law_start='2026-01-31T22:00:00Z')
    clock = Q.bind(runtime, gen, permit)
    finite.publish_due(gen, permit, max_new_receipts=3)
    first = clock.advance(runtime)
    recipe = FrozenNeuralRecipe(rolling_spec=runtime.spec)
    data = build_rolling_data(runtime._snapshot(first['logical_at']), runtime.spec)
    cutoff = pd.Timestamp('2026-02-01T00:00:00Z')
    context = gates._context(data, cutoff, recipe.pool, runtime.lifecycle.plan.source_id)
    store = gates._Store(runtime.directory/'gate-primitives')
    try:
        receipt = gates._register(store, 'protocol', context, {'engineering_primitive': True}, test_only=True)
        future_context = {**context, 'cutoff': '2026-03-01T00:00:00+00:00'}
        future = gates._register(store, 'protocol', future_context, {'engineering_primitive': True}, test_only=True)
        second = clock.advance(runtime)
        job = runtime.lifecycle.plan_fit(cutoff, 'A0', neural_pipeline(recipe, runtime.spec))
        ref = {'lifecycle_directory': str(runtime.directory), 'job': job}
        assert gates._gate_physical_cutoff(context, ref) == pd.Timestamp(second['physical_observation']['wall'])
        assert gates._read(store, receipt, context, pd.Timestamp.now(tz='UTC'))['test_only'] is True
        with pytest.raises(ValueError, match='availability mismatch'):
            gates._read(store, receipt, context, cutoff)
        assert gates._read(store, receipt, context, cutoff, forward_job=ref)['test_only'] is True
        with pytest.raises(ValueError, match='availability mismatch'):
            gates._read(store, future, context, cutoff, forward_job=ref)
        late = gates._register(store, 'protocol', context, {'late_engineering_primitive': True}, test_only=True)
        with pytest.raises(ValueError, match='availability mismatch'):
            gates._read(store, late, context, cutoff, forward_job=ref)
        with pytest.raises(ValueError, match='not a caller clock'):
            gates._gate_physical_cutoff(context, {'clock': second['physical_observation']['wall']})
        with pytest.raises(ValueError, match='original job/source'):
            gates._gate_physical_cutoff({**context, 'source_id': 'foreign'}, ref)
    finally:
        store.close()


def test_preparation_requires_original_prefix_and_distinct_availability(factory):
    from factorlab.discovery.forward_preparation import ForwardPreparation
    from factorlab.discovery.known_linear_reference import FrozenLinearRecipe
    from factorlab.discovery.known_linear_forward import pipeline_for
    clock, _, _, runtime, _ = ready(factory)
    prefix = clock.advance(runtime)
    recipe = FrozenLinearRecipe('K0', runtime.spec)
    cutoff = pd.Timestamp(prefix['logical_at']).normalize().replace(day=1).isoformat()
    job = runtime.lifecycle.plan_fit(cutoff, 'A0', pipeline_for(recipe))
    header = {'job': job, 'plan_id': runtime.lifecycle.plan.digest, 'cutoff': cutoff,
              'feed': runtime.feed.identity, 'feed_directory': str(runtime.feed.directory),
              'source_hash': fingerprint(runtime.identity['source_files']),
              'pipeline': pipeline_for(recipe), 'runtime_directory': str(runtime.directory)}
    checker = ForwardPreparation.__new__(ForwardPreparation)
    checker._header = header
    value = {k: header[k] for k in ('job', 'plan_id', 'cutoff', 'feed', 'feed_directory', 'source_hash')}
    value.update(pipeline_id=recipe.pipeline_id, observation_prefix=prefix,
                 availability='quarantined_finite_observation')
    value['snapshot_id'] = fingerprint(value)
    checker._check_snapshot(value)
    for altered in ({**value, 'availability': 'actual_receipts'},
                    {**value, 'observation_prefix': {**prefix, 'hour': 1}}):
        altered['snapshot_id'] = fingerprint({k: v for k, v in altered.items() if k != 'snapshot_id'})
        with pytest.raises(ValueError, match='complete recipe or identity'):
            checker._check_snapshot(altered)


def test_read_invocation_rechecks_boundaries_and_has_no_cross_call_permission(factory, monkeypatch):
    clock, _, _, runtime, _ = ready(factory)
    prefix = clock.advance(runtime)
    calls = []
    original = observation._verified_publication_sources
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(observation, '_verified_publication_sources', counted)
    with observation._original_observation_read_scope(runtime.lifecycle):
        assert observation.current_binding(runtime.lifecycle) == prefix
        assert observation.current_binding(runtime.lifecycle) == prefix
        with observation._original_observation_read_scope(runtime.lifecycle):
            assert observation.current_binding(runtime.lifecycle) == prefix
        class ForeignDirectory:
            directory = runtime.directory/'foreign'
        with pytest.raises(ValueError, match='another original lifecycle prefix'):
            with observation._original_observation_read_scope(ForeignDirectory()):
                pass
    assert len(calls) == 2
    assert observation.current_binding(runtime.lifecycle) == prefix
    assert len(calls) == 3


def test_read_invocation_rejects_mid_call_original_file_rewrite(factory):
    clock, _, _, runtime, _ = ready(factory)
    prefix = clock.advance(runtime)
    with pytest.raises(ValueError, match='physical witness|physical publication'):
        with observation._original_observation_read_scope(runtime.lifecycle):
            assert observation.current_binding(runtime.lifecycle) == prefix
            path = clock.directory/'events/00000.json'
            path.write_bytes(path.read_bytes())
    # The failed closing check must not leave a reusable grant behind.
    with pytest.raises(ValueError, match='physical witness|physical publication'):
        observation.current_binding(runtime.lifecycle)
