"""Actual-receipt calibration calls retain the real shared S controller."""
import json

import pytest
from test_v7_s_controller_integration import FakeBatch

from factorlab.discovery import calibration_coverage as coverage

pytest_plugins = ['test_v7_s_controller_integration']


def harness(case):
    plan = coverage.CoveragePlan((case.cutoff.isoformat(),), input_scope='live_receipts')
    return coverage.CoverageHarness(case.root / 'coverage', plan,
                                    config=case.config, spec=case.runtime.spec)


def call(h, case, backend):
    return h.run_month('trial-0', case.bars, case.cutoff, backend=backend,
        exposure_path=case.runtime.lifecycle.exposure.path, feed_directory=case.runtime.feed.directory,
        authority=case.authority, schedule=case.schedule)


def test_actual_three_family_research_coverage_original_wallet_and_readonly_recovery(case, monkeypatch):
    h, backend = harness(case), FakeBatch()
    result = call(h, case, backend)
    assert result['status'] == 'complete' and len(backend.calls) == 3
    proof = result['evidence']['shared_accounting']
    assert proof['authority_manifest_hash'] == case.authority.manifest_hash
    assert proof['closed_slots'] == 29 and proof['actual_counts']['solve'] >= 52
    assert proof['actual_counts']['f'] == 3 and proof['readback_only']
    assert result['evidence']['A0'] == 'unchanged_by_S'
    before = case.authority.verify()
    monkeypatch.setattr(coverage, 'run_month', lambda *a, **k: pytest.fail('Calibrator replaced original work'))
    assert call(h, case, backend) == result
    assert len(backend.calls) == 3 and case.authority.verify() == before
    report = h.report()
    assert report['exercised_paths'] == ['monthly_controller']
    assert report['formal_gate_eligible'] is report['full_V7'] is False
    assert 'adaptive_GPT_distribution' in report['missing_requirements']
    # Losing a fixed-slot mirror cannot be healed by inspection, including if
    # the separate published monthly summary remains internally consistent.
    path = h.directory / 'trials/trial-0/research/work_authority/2026-10/schedule/results/000.json'
    path.unlink()
    with pytest.raises(ValueError, match='Incomplete S work publication'):
        h.report()
    assert not path.exists() and case.authority.verify() == before


def test_live_calibration_requires_genuine_wallet_and_schedule_before_reserving_trial(case):
    h = harness(case)
    args = {'backend': FakeBatch(), 'exposure_path': case.runtime.lifecycle.exposure.path,
            'feed_directory': case.runtime.feed.directory}
    with pytest.raises(ValueError, match='shared authority and schedule'):
        h.run_month('trial-0', case.bars, case.cutoff, **args)
    with pytest.raises(TypeError, match='Actual shared authority'):
        h.run_month('trial-0', case.bars, case.cutoff, authority=True, schedule=case.schedule, **args)
    assert h.report()['reserved_units'] == 0 and case.authority.verify()['reservations'] == 0


def test_real_prepaid_failure_cannot_be_replaced_after_calibration_failure(case, monkeypatch):
    h, backend = harness(case), FakeBatch()
    monkeypatch.setattr(case.module, 'build_rolling_data',
                        lambda *a, **k: (_ for _ in ()).throw(OSError('actual input failure')))
    with pytest.raises(OSError, match='actual input failure'):
        call(h, case, backend)
    result = call(h, case, backend)
    assert result['status'] == 'failed_consumed' and not backend.calls
    assert case.authority.verify()['reservations'] == 1
    with case.authority._transaction() as db:
        record = next(iter(case.authority._verify(db)['reservations'].values()))
    assert record['counts']['solve'] == 7 and record['status'] == 'failed_consumed'
    reservation_path = next((h.directory/'units').glob('*/reservation.json'))
    assert json.loads(reservation_path.read_text())['value']['shared_work_binding']['schedule'] == case.schedule.to_dict()
