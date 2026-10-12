"""Shared live-source S exposure audit; artificial receipts, no fitting/provider calls."""
import json
from dataclasses import asdict

import pandas as pd
import pytest
from test_v7_feed_audit import Clock
from test_v7_rolling_registry import candles
from test_v7_run_factor_pool import source_root

from factorlab.discovery import v7_run
from factorlab.discovery.exposure import ExposureLedger
from factorlab.discovery.factor_pool import FactorPool, formula_identity
from factorlab.discovery.rolling import RollingSpec, development_folds
from factorlab.discovery.v7_memory import ResearchMemory, search_exposure_context
from factorlab.discovery.v7_storage import immutable_record
from factorlab.protocol import fingerprint


class BeforeFit(BaseException):
    pass


class LiveCase(tuple):
    def __new__(cls, values, authority, schedule):
        obj = super().__new__(cls, values)
        obj.authority, obj.schedule = authority, schedule
        return obj


@pytest.fixture
def live(tmp_path, monkeypatch):
    from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
    from factorlab.discovery.neural_forward_ticket import _bind_authority
    from factorlab.discovery.scheduled_search import FrozenSWorkSchedule
    from factorlab.discovery.v7_forward import ForwardRuntime
    from factorlab.discovery.v7_s_controller import standard_slots
    source_root(tmp_path, monkeypatch)
    clock = Clock('2025-03-31T23:30:00Z')
    runtime = ForwardRuntime(tmp_path / 'forward', tmp_path / 'feed', clock=clock)
    feed = runtime.feed
    seed = candles('2024-08-01T00:00Z', '2025-03-31T23:00Z')
    path = tmp_path / 'seed.parquet'
    seed.to_parquet(path)
    feed.import_seed(path)
    clock.now = pd.Timestamp('2025-04-01T00:00:10Z')
    context = search_exposure_context('unused', '2025-04-01T00:00Z', ['BTCUSDT'], feed_identity=feed.identity)
    bars = feed.snapshot('2025-04-01T00:00Z', start=context['start'],
                         archive_delay_seconds=RollingSpec().availability_delay_seconds)
    authority = MonthlyWorkAuthority(tmp_path / 'wallet', run_id='actual-wallet-exposure-audit',
                                    task='volatility', universe=['BTCUSDT'])
    _bind_authority(runtime, authority)
    # Actual S publication is now; use that same clock when reading its ledger.
    runtime.lifecycle.exposure.clock = lambda: pd.Timestamp.now(tz='UTC')
    config = v7_run.V7Config(evaluation_mode='joint', formulas_per_family=8)
    slots, allocation = standard_slots(candidate_caps=[15]*24, candidate_seconds=[90]*24,
        pruning_cap=27, pruning_seconds=120, forward_solve_holdback=100)
    allocation.update(forward_seconds_holdback=600, nonlinear_seconds_holdback=0)
    schedule = FrozenSWorkSchedule.freeze(config=asdict(config), limits=v7_run.monthly_budget(config,'2025-04').limits,
                                         slots=slots, allocation=allocation)
    yield LiveCase((tmp_path, bars, feed.identity, runtime.lifecycle.exposure, config), authority, schedule)
    runtime.close()


def store_kwargs(live):
    return {'mode': 'authority_scheduled_research_v1', 'work_contract': {
        'schedule': live.schedule.to_dict(), 'authority_manifest_hash': live.authority.manifest_hash}}


def stop_before_fit(monkeypatch, observed):
    def stop(*args, **kwargs):
        observed.append(True)
        raise BeforeFit
    monkeypatch.setattr(v7_run, '_baseline_readout', stop)


def run(live, directory='research', **kwargs):
    root, bars, _, ledger, config = live
    return v7_run.run_month(root / directory, bars, '2025-04-01T00:00Z',
                            config=config, exposure_path=ledger.path, feed_directory=root / 'feed',
                            authority=live.authority, schedule=live.schedule, **kwargs)


def test_monthly_data_hash_changes_do_not_rename_the_shared_live_source(live):
    _, _, source, _, _ = live
    first = search_exposure_context('data-one', '2025-04-01T00:00Z', ['BTCUSDT'], feed_identity=source)
    second = search_exposure_context('data-two', '2025-05-01T00:00Z', ['BTCUSDT'], feed_identity=source)
    assert first['source'] == second['source'] == source and first['batch'] != second['batch']
    assert first['data_id'] != second['data_id']
    historical = search_exposure_context('data-one', '2025-04-01T00:00Z', ['BTCUSDT'])
    assert historical['source'] != source


def test_actual_shared_ledger_contains_S_before_any_baseline_fit(live, monkeypatch):
    root, bars, source, ledger, config = live
    checked = []

    def before_any_fit(*args, **kwargs):
        context = search_exposure_context('unused', '2025-04-01T00:00Z', ['BTCUSDT'], feed_identity=source)
        assert ledger.assert_prompt_search(**{k: v for k, v in context.items() if k != 'data_id'})
        checked.append(True)
        raise BeforeFit

    monkeypatch.setattr(v7_run, '_baseline_readout', before_any_fit)
    with pytest.raises(BeforeFit):
        run(live)
    assert checked
    assert not (root / 'research' / 'exposure.sqlite').exists()
    rows = ledger.db.execute('SELECT source,role,batch,detail FROM evidence').fetchall()
    assert len(rows) == 2 and all(row[:3] == (source, 'S', '2025-04') for row in rows)
    assert sum(json.loads(row[3]).get('scope') == 'input_preparation' for row in rows) == 1
    store = v7_run.RunStore(root / 'research', config, RollingSpec(), exposure_path=ledger.path, feed_identity=source, **store_kwargs(live))
    try:
        state = store.get('2025-04')
        assert state['search_exposure']['source'] == source
        assert [json.loads(row[3])['data_id'] for row in rows if 'data_id' in json.loads(row[3])] == [state['data_id']]
        assert store.exposure.path.resolve() == ledger.path.resolve()
        assert bars.attrs['feed_identity'] == source
    finally:
        store.close()


@pytest.mark.parametrize('protected', ['V_record', 'C_record', 'V_reservation', 'C_reservation'])
def test_same_batch_protected_source_rejected_before_even_local_fitting(live, monkeypatch, protected):
    _, _, source, ledger, _ = live
    if protected.endswith('_record'):
        ledger.record(source=source, symbols=['BTCUSDT'], start='2025-03-01T00:00Z',
                      end='2025-04-01T00:00Z', batch='2025-04', role=protected[0])
    else:
        ledger.clock = lambda: '2025-02-28T00:00Z'
        common = {'source': source, 'symbols': ['BTCUSDT'], 'start': '2025-03-01T00:00Z',
                  'end': '2025-04-01T00:00Z', 'batch': '2025-04', 'reference_id': 'reference',
                  'frozen_at': '2025-02-28T00:00Z'}
        if protected[0] == 'V':
            ledger.reserve_validation(**common, candidate_ids=['candidate'])
        else:
            ledger.reserve_confirmation(**common, candidate_id='candidate', rule={}, cohort=1, claim_slot='factor1')
        ledger.clock = lambda: pd.Timestamp.now(tz='UTC')
    called = []
    stop_before_fit(monkeypatch, called)
    with pytest.raises(ValueError, match='Protected|protected|Frozen|confirmation|feedback|expos'):
        run(live)
    assert not called


def test_other_batch_old_C_reuse_requires_new_S_and_preserves_history(live, monkeypatch):
    _, _, source, ledger, _ = live
    ledger.record(source=source, symbols=['BTCUSDT'], start='2025-03-01T00:00Z',
                  end='2025-04-01T00:00Z', role='C', batch='old-frozen-factor')
    before = ledger.db.execute("SELECT id FROM evidence WHERE role='C'").fetchone()[0]
    called = []
    stop_before_fit(monkeypatch, called)
    with pytest.raises(BeforeFit):
        run(live)
    assert called
    assert ledger.db.execute("SELECT id FROM evidence WHERE role='C'").fetchone()[0] == before
    assert set(ledger.db.execute('SELECT role,batch FROM evidence')) == {('C', 'old-frozen-factor'), ('S', '2025-04')}
    with pytest.raises(ValueError, match='Protected|protected'):
        ledger.assert_prompt_search(source=source, symbols=['BTCUSDT'], start='2025-03-01T00:00Z',
                                   end='2025-04-01T00:00Z', batch='old-frozen-factor')


def test_resume_cannot_switch_to_a_cleaner_ledger(live, monkeypatch):
    root, _, _, _, config = live
    stop_before_fit(monkeypatch, [])
    with pytest.raises(BeforeFit):
        run(live)
    with pytest.raises(ValueError, match='identity|source/config|exposure|ledger'):
        store = v7_run.RunStore(root / 'research', config, RollingSpec(), exposure_path=root / 'different.sqlite')
        store.close()


@pytest.mark.parametrize('change', ['data', 'feed_identity'])
def test_current_month_snapshot_or_feed_identity_cannot_change_on_resume(live, monkeypatch, change):
    _, bars, _, ledger, _ = live
    stop_before_fit(monkeypatch, [])
    with pytest.raises(BeforeFit):
        run(live)
    count = ledger.db.execute('SELECT COUNT(*) FROM evidence').fetchone()[0]
    if change == 'data':
        bars.loc[0, ['open', 'high', 'low', 'close']] *= 2
    else:
        bars.attrs['feed_identity'] = 'another-feed'
    with pytest.raises(ValueError, match='dataset|identity|source|feed|receipt'):
        run(live)
    assert ledger.db.execute('SELECT COUNT(*) FROM evidence').fetchone()[0] == count


def test_protected_exposure_added_after_checkpoint_is_rechecked_on_resume(live, monkeypatch):
    _, _, source, ledger, _ = live
    calls = []
    stop_before_fit(monkeypatch, calls)
    with pytest.raises(BeforeFit):
        run(live)
    ledger.record(source=source, symbols=['BTCUSDT'], start='2025-03-01T00:00Z',
                  end='2025-04-01T00:00Z', role='C', batch='2025-04')
    with pytest.raises(ValueError, match='Protected|protected|confirmation|expos'):
        run(live)
    assert len(calls) == 1


def test_live_source_cannot_use_empty_sibling_ledger_to_bypass_protected_data(live, monkeypatch):
    root, bars, source, ledger, config = live
    ledger.record(source=source, symbols=['BTCUSDT'], start='2025-03-01T00:00Z',
                  end='2025-04-01T00:00Z', role='C', batch='2025-04')
    calls = []
    stop_before_fit(monkeypatch, calls)
    with pytest.raises(ValueError, match='ledger|exposure|shared|lifecycle'):
        v7_run.run_month(root / 'new-research', bars, '2025-04-01T00:00Z', config=config,
                        exposure_path=ledger.path.parent / 'clean.sqlite', feed_directory=root / 'feed',
                        authority=live.authority, schedule=live.schedule)
    assert not calls


def test_live_source_requires_archive_and_explicit_shared_ledger(live, monkeypatch):
    root, bars, _, ledger, config = live
    calls = []
    stop_before_fit(monkeypatch, calls)
    for kwargs in ({}, {'exposure_path': ledger.path}, {'feed_directory': root / 'feed'}):
        with pytest.raises(ValueError, match='Live S|archive|shared|Shared'):
            v7_run.run_month(root / 'missing-proof', bars, '2025-04-01T00:00Z', config=config, authority=live.authority, schedule=live.schedule, **kwargs)
    assert not calls


def test_live_claimed_frame_must_match_actual_asof_receipt_bytes(live, monkeypatch):
    _, bars, _, _, _ = live
    calls = []
    stop_before_fit(monkeypatch, calls)
    bars.loc[0, 'close'] *= 1.001
    with pytest.raises(ValueError, match='snapshot|receipt'):
        run(live)
    assert not calls


def test_memory_resolver_cannot_replace_the_frozen_shared_exposure_ledger(live, monkeypatch):
    root, _, source, ledger, config = live
    stop_before_fit(monkeypatch, [])
    with pytest.raises(BeforeFit):
        run(live)
    store = v7_run.RunStore(root / 'research', config, RollingSpec(),
                           exposure_path=ledger.path, feed_identity=source, **store_kwargs(live))
    clean = ExposureLedger(root / 'unrelated-clean.sqlite')
    try:
        state = store.get('2025-04')
        program = v7_run.literature_seeds()[0]
        identifier = fingerprint(formula_identity(program, config.lookback))
        result = {'role': 'S', 'formula_id': identifier, 'folds': list(development_folds(state['cutoff'])),
                  'channels': {'F': {'gain': 0.}, 'S': {'gain': 0.}},
                  'scope': 'synthetic_checkpoint_fixture_not_fitted_market_evidence'}
        pool = FactorPool()
        pool.add(program, result, family='synthetic-only', lookback=config.lookback)
        state['pool'] = pool.to_dict()
        immutable_record(store.directory / 'results' / f'2025-04-{identifier}.json', result)
        store.save('2025-04', state, 'synthetic_audit_record')
        memory = ResearchMemory.from_state(state['research_memory'])
        evidence_id = memory.bind_result(store, store.exposure, '2025-04', identifier)
        context = state['search_exposure']
        clean.record(**{k: v for k, v in context.items() if k != 'data_id'}, role='S')
        ledger.record(source=source, symbols=['BTCUSDT'], start='2025-03-01T00:00Z',
                      end='2025-04-01T00:00Z', role='C', batch='2025-04')
        with pytest.raises(ValueError, match='ledger|exposure|path|identity'):
            memory.resolver(store, clean)(evidence_id)
    finally:
        clean.close()
        store.close()
