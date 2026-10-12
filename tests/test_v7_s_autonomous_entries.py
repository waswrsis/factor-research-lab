"""Real authority and scheduler, artificial receipts, no actual subprocess/GPT."""
import sys
from pathlib import Path

import pytest

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'factorlab').is_dir())
sys.path.insert(0, str(REPO / 'tests'))
from test_v7_s_controller_integration import case as case  # noqa: PLC0414
from test_v7_s_controller_integration import staged as staged  # noqa: PLC0414


@pytest.fixture
def entry(case, monkeypatch):
    from factorlab.discovery import s_outer_context, s_preparation, v7_autonomous
    modules = {'s_outer_context': s_outer_context, 's_preparation': s_preparation, 'v7_autonomous': v7_autonomous}
    scheduler = modules['v7_autonomous'].ResearchScheduler(case.runtime,
        config=modules['v7_autonomous'].AutonomousConfig(research=case.config),
        authority=case.authority, schedule=case.schedule)
    state = scheduler._plan(case.cutoff, case.runtime.clock())
    state['status'] = 'running'
    state['attempts'] = [{'number': 0, 'token': 'original', 'process': None}]
    scheduler._save(state)
    try:
        yield modules, scheduler, state
    finally:
        scheduler.close()


def test_snapshot_is_prepaid_and_same_original_job_caps_whole_S(case, entry, monkeypatch):
    modules, _scheduler, state = entry
    auto = modules['v7_autonomous']
    job = auto._read(state['job_path'])
    observed = []
    original = auto.sealed_snapshot
    def snapshot(path):
        with case.authority._transaction() as db:
            records = list(case.authority._verify(db)['reservations'].values())
        assert len(records) == 1 and records[0]['status'] == 'reserved'
        assert records[0]['counts']['solve'] == 7
        observed.append(records[0])
        return original(path)
    monkeypatch.setattr(auto, 'sealed_snapshot', snapshot)
    def run(directory, bars, cutoff, **kwargs):
        prep = kwargs['preparation']
        assert kwargs['authority'].manifest_hash == case.authority.manifest_hash
        assert kwargs['schedule'].digest == case.schedule.digest
        envelope = prep.identity['execution_envelope']
        assert envelope['monotonic_deadline'] <= job['outer_clock']['monotonic_deadline']
        assert envelope['wall_deadline'] <= job['deadline']
        assert envelope['issuer'] == observed[0]['lease']
        assert prep.identity['clock_scope'] == 'before_feed_snapshot'
        assert bars.attrs['feed_identity'] == case.runtime.feed.identity
        return {'fixture': 'no numerical fitting'}
    monkeypatch.setattr(auto, 'run_month', run)
    assert auto.run_worker(state['job_path'], 0, 'original', backend=object(), enforce_resources=False)
    assert len(observed) == 1


def test_failed_snapshot_does_not_refund_or_reload(case, entry, monkeypatch):
    modules, _, state = entry
    auto = modules['v7_autonomous']
    calls = []
    def broken(path):
        calls.append(path)
        raise OSError('artificial read failure')
    monkeypatch.setattr(auto, 'sealed_snapshot', broken)
    with pytest.raises(OSError, match='artificial'):
        auto.run_worker(state['job_path'], 0, 'original', enforce_resources=False)
    with case.authority._transaction() as db:
        record, = case.authority._verify(db)['reservations'].values()
    assert record['status'] == 'failed_consumed' and record['counts']['solve'] == 7
    with pytest.raises(ValueError, match='readback only'):
        auto.run_worker(state['job_path'], 0, 'original', enforce_resources=False)
    assert len(calls) == 1


def test_outer_rejects_changed_token_and_source_schedule_before_spending(case, entry):
    modules, _, state = entry
    cls = modules['s_outer_context'].SOuterContext
    with pytest.raises(ValueError, match='running research job'):
        cls.autonomous(state['job_path'], 0, 'different')
    context = cls.autonomous(state['job_path'], 0, 'original')
    job = modules['v7_autonomous']._read(state['job_path'])
    with pytest.raises(ValueError, match='authorize'):
        context.bind(case.authority, job['run_directory'], job['config']['research'], job['rolling'], {}, '0'*64)
    assert case.authority.verify()['reservations'] == 0


def test_original_outer_expires_without_worker_clock_reset(case, entry, monkeypatch):
    modules, _, state = entry
    outer = modules['s_outer_context']
    job = modules['v7_autonomous']._read(state['job_path'])
    monkeypatch.setattr(outer.time, 'monotonic', lambda: job['outer_clock']['monotonic_deadline'])
    with pytest.raises(TimeoutError, match='deadline'):
        outer.SOuterContext.autonomous(state['job_path'], 0, 'original')
    assert case.authority.verify()['reservations'] == 0


def test_less_than_original_baseline_work_remaining_cannot_load_snapshot(case, entry, monkeypatch):
    from dataclasses import asdict
    modules, _, state = entry
    auto, outer = modules['v7_autonomous'], modules['s_outer_context']
    context = outer.SOuterContext.autonomous(state['job_path'], 0, 'original')
    job = auto._read(state['job_path'])
    # Move only the observation forward, retaining original issued deadlines.
    from factorlab.discovery.work_lease import FrozenWorkLease
    origin = FrozenWorkLease.issue(300)
    values = origin.to_dict()
    values['monotonic_started'] = job['outer_clock']['monotonic_deadline'] - 100
    values['monotonic_deadline'] = values['monotonic_started'] + 300
    later = FrozenWorkLease.from_dict(values)
    monkeypatch.setattr(later.__class__, 'assert_parent_alive', lambda *_a, **_kw: None)
    with pytest.raises(TimeoutError, match='insufficient remaining'):
        modules['s_preparation'].SPreparationEnvelope.prepare(case.authority, job['run_directory'], job['month'],
            config=asdict(case.config), spec=job['rolling'], source_reference={
                'feed_identity': job['feed_id'], 'feed_directory': job['feed_directory'],
                'exposure_path': job['exposure_path'], 'cutoff': job['cutoff']},
            schedule_hash=case.schedule.digest, outer_context=context, lease=later)
    assert case.authority.verify()['reservations'] == 0
