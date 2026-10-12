"""Coverage contracts use real synthetic monthly artifacts, never a certificate."""
import json

import pytest
from test_v7_rolling_registry import candles
from test_v7_run_factor_pool import source_root

from factorlab.discovery import calibration_coverage as coverage
from factorlab.discovery.v7_run import V7Config


@pytest.fixture
def setup(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    plan = coverage.CoveragePlan(('2025-04-01T00:00:00Z',))
    return tmp_path, plan


def test_grid_is_finite_and_calendar_predeclared():
    with pytest.raises(ValueError):
        coverage.CoveragePlan(('2025-04-02T00:00:00Z',))
    with pytest.raises(ValueError):
        coverage.CoveragePlan(('2025-04-01',))
    with pytest.raises(ValueError):
        coverage.CoveragePlan(('2025-04-01T00:00:00Z',), trials=('../escape',))


def test_real_month_path_verified_without_promotion_or_full_coverage(setup, monkeypatch):
    root, plan = setup
    harness = coverage.CoverageHarness(root / 'coverage', plan)
    bars = candles('2024-08-01', '2025-04-02')
    outcome = harness.run_month('trial-0', bars, plan.cutoffs[0])
    assert outcome['status'] == 'complete'
    assert outcome['evidence']['budget']['counts']['solve'] > 0
    assert outcome['evidence']['A0'] == 'K0'
    monkeypatch.setattr(coverage, 'run_month', lambda *a, **kw: pytest.fail('No replacement fit'))
    assert harness.run_month('trial-0', bars, plan.cutoffs[0]) == outcome
    report = harness.report()
    assert report['reserved_units'] == 1
    assert report['exercised_paths'] == ['monthly_controller']
    assert set(report['missing_requirements']) == set(coverage.OBLIGATIONS) - {'monthly_controller'}
    assert report['claims'] == list(coverage.CLAIMS)
    assert report['formal_gate_eligible'] is report['full_V7'] is False
    assert report['statistical_certificate'] is None
    receipt = next((root / 'coverage/units').glob('*/outcome.json'))
    receipt.unlink()  # Death after real controller completion, before adapter publication.
    recovered = harness.run_month('trial-0', bars, plan.cutoffs[0])
    assert recovered['evidence'] == outcome['evidence']
    assert recovered['recovery'] == 'verified_existing_completion_no_work'
    # A preserved source/result chain is required on every subsequent inspection.
    path = root / 'coverage/trials/trial-0/research/months/2025-04.json'
    summary = json.loads(path.read_text())
    summary['budget']['counts']['solve'] -= 1
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='summary'):
        harness.report()


def test_external_mutation_of_plan_lists_cannot_authorize_more_trials(setup):
    root, original = setup
    trials = ['trial-0']
    plan = coverage.CoveragePlan(original.cutoffs, trials=trials)
    harness = coverage.CoverageHarness(root / 'coverage', plan)
    trials.append('trial-extra')
    with pytest.raises(ValueError, match='Unplanned'):
        harness.run_month('trial-extra', candles('2025-01-01', '2025-01-02'), plan.cutoffs[0])


def test_failure_is_consumed_and_no_new_backend_or_controller_call(setup, monkeypatch):
    root, plan = setup
    harness = coverage.CoverageHarness(root / 'coverage', plan)
    calls = []

    def fail(*a, **kw):
        calls.append(True)
        raise OSError('synthetic interrupted call')

    monkeypatch.setattr(coverage, 'run_month', fail)
    bars = candles('2025-01-01', '2025-01-02')
    with pytest.raises(OSError):
        harness.run_month('trial-0', bars, plan.cutoffs[0])
    again = harness.run_month('trial-0', bars, plan.cutoffs[0])
    assert again['status'] == 'failed_consumed' and len(calls) == 1
    assert harness.report()['reserved_units'] == 1
    assert not harness.report()['exercised_paths']


def test_missing_unit_receipt_cannot_refund_journal_reservation(setup, monkeypatch):
    root, plan = setup
    harness = coverage.CoverageHarness(root / 'coverage', plan)
    monkeypatch.setattr(coverage, 'run_month', lambda *a, **kw: (_ for _ in ()).throw(OSError('failure')))
    with pytest.raises(OSError):
        harness.run_month('trial-0', candles('2025-01-01', '2025-01-02'), plan.cutoffs[0])
    path = next((root / 'coverage/units').glob('*/reservation.json'))
    path.unlink()
    with pytest.raises(ValueError, match='journal/unit'):
        coverage.CoverageHarness(root / 'coverage', plan)


def test_lost_journal_and_reservation_cannot_hide_remaining_outcome(setup, monkeypatch):
    root, plan = setup
    harness = coverage.CoverageHarness(root / 'coverage', plan)
    monkeypatch.setattr(coverage, 'run_month', lambda *a, **kw: (_ for _ in ()).throw(OSError('failure')))
    with pytest.raises(OSError):
        harness.run_month('trial-0', candles('2025-01-01', '2025-01-02'), plan.cutoffs[0])
    (root / 'coverage/reservations.jsonl').unlink()
    next((root / 'coverage/units').glob('*/reservation.json')).unlink()
    assert list((root / 'coverage/units').glob('*/outcome.json'))
    with pytest.raises(ValueError, match='journal/unit'):
        coverage.CoverageHarness(root / 'coverage', plan)


def test_source_change_and_unplanned_month_fail_before_work(setup, monkeypatch):
    root, plan = setup
    harness = coverage.CoverageHarness(root / 'coverage', plan)
    monkeypatch.setattr(coverage, 'run_month', lambda *a, **kw: pytest.fail('Invalid input reached core'))
    bars = candles('2025-01-01', '2025-01-02')
    with pytest.raises(ValueError, match='Unplanned'):
        harness.run_month('trial-0', bars, '2025-05-01T00:00:00Z')
    (root / 'factorlab/v7.py').write_text('changed')
    with pytest.raises(ValueError, match='source'):
        harness.run_month('trial-0', bars, plan.cutoffs[0])


def test_actual_gpt_and_live_dependencies_cannot_be_implicit(setup, monkeypatch):
    root, plan = setup
    config = V7Config(evaluation_mode='joint', search_mode='gpt', formulas_per_family=1)
    harness = coverage.CoverageHarness(root / 'coverage', plan, config=config)
    monkeypatch.setattr(coverage, 'run_month', lambda *a, **kw: pytest.fail('Implicit dependency'))
    bars = candles('2025-01-01', '2025-01-02')
    with pytest.raises(ValueError, match='explicitly'):
        harness.run_month('trial-0', bars, plan.cutoffs[0])
    live = coverage.CoverageHarness(root / 'live', coverage.CoveragePlan(plan.cutoffs, input_scope='live_receipts'))
    with pytest.raises(ValueError, match='archive/shared'):
        live.run_month('trial-0', bars, plan.cutoffs[0])


def test_existing_local_diagnostic_preserves_scope_and_gets_no_production_credit(monkeypatch):
    from factorlab.discovery import full_pipeline_calibration as local
    report = {'scope': local.SCOPE, 'excluded_scope': list(local.EXCLUDED),
              'full_V7': False, 'formal_gate_eligible': False}
    monkeypatch.setattr(local, 'inspect_full_pipeline_calibration', lambda _: report)
    result = coverage.inspect_local_diagnostic('not-opened-by-stub')
    assert result['excluded_scope'] == list(local.EXCLUDED)
    assert result['production_coverage_credit'] == []
    report['full_V7'] = True
    with pytest.raises(ValueError, match='production'):
        coverage.inspect_local_diagnostic('not-opened-by-stub')


def test_shared_controls_attachment_binds_real_inspector_and_prepaid_partitions(setup, monkeypatch):
    from test_v7_controls import StubBackend
    from test_v7_shared_controls import _stub

    from factorlab.discovery import shared_controls

    root, plan = setup
    _stub(monkeypatch, [])  # Numerical work only is stubbed; all receipts/inspectors are real.
    bars = candles('2024-08-01', '2025-04-02')
    shared_controls.run_shared_controls(root / 'controls', bars, plan.cutoffs[0], backend=StubBackend())
    harness = coverage.CoverageHarness(root / 'coverage', plan)
    evidence = harness.attach_shared_controls('trial-0', root / 'controls')
    assert sum(v['budget']['limits']['solve'] for v in evidence['envelopes'].values()) == 1599
    assert sum(v['research_seconds'] for v in evidence['envelopes'].values()) == 7200
    report = harness.report()
    assert report['exercised_paths'] == ['shared_monthly_controls']
    assert 'adaptive_GPT_distribution' in report['missing_requirements']
    assert evidence['provider_authenticity_certified'] is False
    changed = bars.copy()
    changed.loc[0, 'close'] *= 1.001
    with pytest.raises(ValueError, match='different data'):
        harness.run_month('trial-0', changed, plan.cutoffs[0])


def test_failed_shared_arm_is_bound_but_does_not_cover_comparison(setup, monkeypatch):
    from test_v7_controls import StubBackend
    from test_v7_shared_controls import _stub

    from factorlab.discovery import shared_controls

    root, plan = setup
    _stub(monkeypatch, [], fail='random')
    bars = candles('2024-08-01', '2025-04-02')
    shared_controls.run_shared_controls(root / 'controls', bars, plan.cutoffs[0], backend=StubBackend())
    harness = coverage.CoverageHarness(root / 'coverage', plan)
    evidence = harness.attach_shared_controls('trial-0', root / 'controls')
    assert not evidence['completed_budget_comparison']
    assert 'shared_monthly_controls' in harness.report()['missing_requirements']
