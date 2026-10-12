"""Actual shared-wallet control allocation; all inputs are artificial."""
import copy
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'factorlab').is_dir())
sys.path.insert(0, str(REPO / 'tests'))
from test_v7_s_controller_integration import case as case  # noqa: PLC0414


@pytest.fixture
def staged():
    from factorlab.discovery import v7_run
    return v7_run


@pytest.fixture
def executor():
    from factorlab.discovery import shared_controls_executor
    return shared_controls_executor


def schedules(case, *, holdback=100, cap=15):
    from factorlab.discovery.scheduled_search import FrozenSWorkSchedule
    base = case.schedule.to_dict()
    result = {}
    for arm in ('local', 'random', 'gpt'):
        config = replace(case.config, search_mode=arm, budget_policy='controls_third',
                         profile_candidates=2, research_seconds=2400)
        slots = copy.deepcopy(base['slots'][:10] + [base['slots'][-1]])
        for slot in slots:
            if slot['stage'] == 'candidate':
                slot['counts']['solve'] = cap
        allocation = {**base['allocation'], 'S_solve': 7 + 8*cap + 27, 'forward_solve_holdback': holdback}
        result[arm] = FrozenSWorkSchedule.freeze(config=asdict(config),
            limits=case.module.monthly_budget(config, '2026-10').limits, slots=slots, allocation=allocation)
    return result


def create(case, executor, **kwargs):
    return executor.SharedControlsExecutor(case.root / 'controls', authority=case.authority,
        schedules=schedules(case, **kwargs), feed_directory=case.runtime.feed.directory,
        exposure_path=case.runtime.lifecycle.exposure.path, cutoff=case.cutoff, spec=case.runtime.spec)


def test_three_arms_use_one_actual_wallet_before_each_file_load(case, executor, monkeypatch):
    path = case.root / 'bars.parquet'
    case.bars.to_parquet(path)
    observed = []
    actual = executor.pd.read_parquet
    def read(*args, **kwargs):
        with case.authority._transaction() as db:
            records = list(case.authority._verify(db)['reservations'].values())
        assert len(records) == len(observed) + 1
        assert records[-1]['counts']['solve'] == 7
        assert records[-1]['status'] == 'reserved'
        observed.append(records[-1])
        return actual(*args, **kwargs)
    monkeypatch.setattr(executor.pd, 'read_parquet', read)
    model_calls = []
    def run(*args, **kwargs):
        prep = kwargs['preparation']
        assert prep.identity['execution_envelope']['seconds'] == 2400
        assert prep.identity['execution_envelope']['monotonic_deadline'] <= obj.identity['clock']['monotonic_deadline']
        assert kwargs['authority'] is case.authority
        assert len(kwargs['schedule'].to_dict()['slots']) == 11
        model_calls.append(kwargs['config'].search_mode)
        prep.start_preprocessing()
        # This routing-only fixture consumes the baseline, with no claimed fit.
        case.authority.fail_consumed(prep.baseline_record()['reservation_id'], 'fixture_no_fit')
        return {'fixture': 'routing only'}
    monkeypatch.setattr(executor, 'run_month', run)
    obj = create(case, executor)
    result = obj.run_file(path, backend=object())
    assert model_calls == ['local', 'random', 'gpt']
    assert len(observed) == 3 and sum(r['counts']['solve'] for r in observed) == 21
    assert result['serving_eligible'] is False
    with pytest.raises(ValueError, match='readback only'):
        create(case, executor)


def test_aggregate_holdback_rejects_three_individually_legal_schedules(case, executor):
    # Each arm has 522 <=533 solves; their total1566 +100 holdback is illegal.
    with pytest.raises(ValueError, match='single wallet'):
        create(case, executor, cap=61)
    assert case.authority.verify()['reservations'] == 0


def test_failed_first_arm_never_loads_later_or_refunds(case, executor, monkeypatch):
    obj = create(case, executor)
    loads = []
    def broken(*args, **kwargs):
        loads.append(True)
        raise OSError('fixture read failure')
    monkeypatch.setattr(executor.pd, 'read_parquet', broken)
    with pytest.raises(OSError):
        obj.run_file(case.root / 'missing', backend=object())
    assert len(loads) == 1
    with case.authority._transaction() as db:
        record, = case.authority._verify(db)['reservations'].values()
    assert record['counts']['solve'] == 7 and record['status'] == 'failed_consumed'
    with pytest.raises(ValueError, match='replay'):
        obj.run_file(case.root / 'missing', backend=object())


def test_actual_control_loop_requires_real_shared_active_claim(case):
    from factorlab.discovery.s_preparation import SPreparationEnvelope
    schedule = schedules(case)['gpt']
    config = case.module.V7Config(**{**schedule.to_dict()['config'], 'symbols': ('BTCUSDT',)})
    source = {'feed_identity': case.runtime.feed.identity, 'feed_directory': str(case.runtime.feed.directory),
              'exposure_path': str(case.runtime.lifecycle.exposure.path), 'cutoff': case.cutoff.isoformat()}
    prep = SPreparationEnvelope.prepare(case.authority, case.directory, '2026-10',
        config=asdict(config), spec=asdict(case.runtime.spec), source_reference=source, schedule_hash=schedule.digest)
    with pytest.raises(ValueError, match='actual frozen allocation'):
        case.module.run_month(case.directory, case.bars, case.cutoff, config=config,
            authority=case.authority, schedule=schedule, preparation=prep,
            exposure_path=case.runtime.lifecycle.exposure.path, feed_directory=case.runtime.feed.directory)


def test_actual_three_arm_baselines_and_all_fixed_slots_share_authority(case, executor):
    from test_v7_s_controller_integration import FakeBatch
    path = case.root / 'bars.parquet'
    case.bars.to_parquet(path)
    backend = FakeBatch()
    obj = create(case, executor, cap=1)
    result = obj.run_file(path, backend=backend)
    assert list(result['arms']) == ['local', 'random', 'gpt']
    assert len(backend.calls) == 1
    for arm, outcome in result['arms'].items():
        assert outcome['summary']['serving_eligible'] is False
        journals = list((obj.directory / 'arms' / arm / 'work_authority/2026-10/schedule/results').glob('*.json'))
        assert len(journals) == 11
    with case.authority._transaction() as db:
        records = list(case.authority._verify(db)['reservations'].values())
    baselines = [r for r in records if r['work_key'].startswith('S-baseline:')]
    assert len(baselines) == 3 and all(r['status'] == 'completed' for r in baselines)
    assert sum(r['actual_counts']['solve'] for r in baselines) == 21
    inspected = executor.inspect_shared_authority_controls(obj.directory)
    assert len(inspected['outcomes']) == 3 and inspected['readback_only'] is True
    missing = obj.directory / 'arms' / 'local' / 'state.sqlite'
    missing.rename(missing.with_suffix('.retained'))
    with pytest.raises(Exception, match='open database|unable|no such'):
        executor.inspect_shared_authority_controls(obj.directory)
    assert not missing.exists()

