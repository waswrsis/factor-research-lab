"""A bounded initial family, causal clock and batch-scoped information roles."""
import json

import pandas as pd
import pytest

from factorlab.discovery.exposure import CLAIM_SLOTS, ExposureLedger, select_on_validation


class Clock:
    def __init__(self, stamp='2027-01-01T00:00:00Z'):
        self.value = pd.Timestamp(stamp)

    def __call__(self):
        return self.value

    def move(self, stamp):
        self.value = pd.Timestamp(stamp)


def ticket(**changes):
    return dict(batch='A', source='exchange', symbols=['BTC'], start='2027-02-01T00:00:00Z',
                end='2027-05-01T00:00:00Z', candidate_id='candidate-A', reference_id='K0',
                rule={'horizon_hours': 12}, frozen_at='2027-01-01T00:00:00Z',
                cohort=1, claim_slot='factor1', **changes)


def changed_ticket(**changes):
    return {**ticket(), **changes}


def test_overlapping_pre_frozen_confirmations_and_later_search_are_valid(tmp_path):
    clock = Clock()
    ledger = ExposureLedger(tmp_path / 'evidence.sqlite', clock=clock)
    try:
        a = ticket()
        b = changed_ticket(batch='B', candidate_id='candidate-B', claim_slot='factor2',
                           start='2027-03-01T00:00:00Z', end='2027-06-01T00:00:00Z')
        ledger.reserve_confirmation(**a)
        ledger.reserve_confirmation(**b)
        clock.move('2027-04-01T00:00:00Z')
        ledger.record(source='exchange', symbols=['BTC'], start=a['start'],
                      end='2027-03-01T00:00:00Z', role='S', batch='later-search')
        assert ledger.assert_prompt_search(source='exchange', symbols=['BTC'], start=a['start'],
                                          end='2027-03-01T00:00:00Z', batch='later-search')
        clock.move('2027-05-03T00:00:00Z')
        first = ledger.consume_confirmation('A', as_of=clock())
        assert first['scoring_cutoff'] == '2027-05-02T12:00:00+00:00'
        clock.move('2027-07-01T00:00:00Z')
        second = ledger.consume_confirmation('B', as_of=clock())
        assert second['scoring_cutoff'] == '2027-06-02T12:00:00+00:00'
        # All facts persist; a new batch explicitly obtains old C as S.
        ledger.record(source='exchange', symbols=['BTC'], start=a['start'], end=a['end'],
                      role='S', batch='new-search')
        assert ledger.assert_prompt_search(source='exchange', symbols=['BTC'], start=a['start'],
                                          end=a['end'], batch='new-search')
        assert ledger.db.execute("SELECT COUNT(*) FROM evidence WHERE role='C'").fetchone()[0] == 2
    finally:
        ledger.close()


@pytest.mark.parametrize('role', ['S', 'V', 'C', 'legacy'])
def test_same_claim_feedback_invalidates_confirmation_without_freeing_slot(tmp_path, role):
    clock = Clock()
    ledger = ExposureLedger(tmp_path / 'evidence.sqlite', clock=clock)
    try:
        a = ticket()
        ledger.reserve_confirmation(**a)
        clock.move('2027-04-01T00:00:00Z')
        ledger.record(source='exchange', symbols=['BTC'], start=a['start'],
                      end='2027-03-01T00:00:00Z', role=role, batch='A')
        clock.move('2027-05-03T00:00:00Z')
        with pytest.raises(ValueError, match='exposed'):
            ledger.consume_confirmation('A', as_of=clock())
        with pytest.raises(ValueError, match='slot'):
            ledger.reserve_confirmation(**changed_ticket(batch='renamed'))
        assert ledger.budget_status()['reserved_slots'] == {1: 1}
    finally:
        ledger.close()


def test_five_unique_slots_immutable_reservations_and_restart(tmp_path):
    clock = Clock()
    path = tmp_path / 'evidence.sqlite'
    ledger = ExposureLedger(path, clock=clock)
    for slot in CLAIM_SLOTS:
        value = changed_ticket(batch=slot, claim_slot=slot)
        assert ledger.reserve_confirmation(**value) == ledger.reserve_confirmation(**value)
    assert ledger.budget_status()['reserved_slots'] == {1: 5}
    assert ledger.budget_status()['planned_alpha_all_cohorts'] == .05
    ledger.close()
    ledger = ExposureLedger(path, clock=clock)
    try:
        with pytest.raises(ValueError, match='slot'):
            ledger.reserve_confirmation(**changed_ticket(batch='renamed', claim_slot='factor2'))
        with pytest.raises(ValueError, match='identity'):
            ledger.reserve_confirmation(**changed_ticket(batch='factor1', candidate_id='changed'))
        for changes in ({'cohort': 2}, {'cohort': True}, {'claim_slot': 'factor4'}, {'alpha': .05}):
            with pytest.raises(ValueError):
                ledger.reserve_confirmation(**changed_ticket(**changes))
    finally:
        ledger.close()


def test_fixed_scoring_cutoff_one_look_and_immutable_result(tmp_path):
    clock = Clock()
    ledger = ExposureLedger(tmp_path / 'evidence.sqlite', clock=clock)
    try:
        ledger.reserve_confirmation(**ticket())
        clock.move('2027-06-01T00:00:00Z')
        with pytest.raises(ValueError, match='matured'):
            ledger.consume_confirmation('A', as_of='2027-05-02T11:59:59Z')
        with pytest.raises(ValueError, match='grace'):
            ledger.consume_confirmation('A', as_of=clock(), receipt_grace_hours=0)
        identity = ledger.consume_confirmation('A', as_of=clock())
        assert identity['scoring_cutoff'] == '2027-05-02T12:00:00+00:00'
        with pytest.raises(ValueError, match='consumed'):
            ledger.consume_confirmation('A', as_of=clock())
        ledger.finish_confirmation('A', {'status': 'unresolved'})
        with pytest.raises(ValueError, match='one result'):
            ledger.finish_confirmation('A', {'status': 'supported'})
        assert json.loads(ledger.db.execute('SELECT result FROM confirmations').fetchone()[0]) == {
            'status': 'unresolved'}
    finally:
        ledger.close()


def test_trusted_clock_prevents_backdated_reservation_and_future_exposure(tmp_path):
    clock = Clock('2027-03-01T00:00:00Z')
    ledger = ExposureLedger(tmp_path / 'evidence.sqlite', clock=clock)
    try:
        with pytest.raises(ValueError, match='prospectively'):
            ledger.reserve_confirmation(**ticket())
        with pytest.raises(ValueError, match='before confirmation'):
            ledger.reserve_confirmation(**changed_ticket(frozen_at='2027-03-01T00:00:00Z'))
        with pytest.raises(ValueError, match='future'):
            ledger.record(source='exchange', symbols=['BTC'], start='2027-01-01T00:00:00Z',
                          end='2027-04-01T00:00:00Z', role='S', batch='A')
        with pytest.raises(ValueError, match='future'):
            ledger.record(source='exchange', symbols=['BTC'], start='2027-01-01T00:00:00Z',
                          end='2027-02-01T00:00:00Z', role='S', batch='A',
                          observed_at='2027-04-01T00:00:00Z')
        ledger.record(source='exchange', symbols=['BTC'], start='2027-01-01T00:00:00Z',
                      end='2027-02-01T00:00:00Z', role='S', batch='A',
                      observed_at='2027-02-02T00:00:00Z')
        observed, recorded = ledger.db.execute('SELECT observed_at,recorded_at FROM evidence').fetchone()
        assert observed == '2027-02-02T00:00:00+00:00' and recorded == clock().isoformat()
    finally:
        ledger.close()


def test_full_search_coverage_is_required_per_symbol_and_batch(tmp_path):
    clock = Clock('2027-04-01T00:00:00Z')
    ledger = ExposureLedger(tmp_path / 'evidence.sqlite', clock=clock)
    request = {'source': 'exchange', 'symbols': ['BTC'], 'start': '2027-01-01T00:00:00Z',
               'end': '2027-02-01T00:00:00Z', 'batch': 'S-batch'}
    try:
        ledger.record(**{**request, 'end': '2027-01-15T00:00:00Z'}, role='S')
        with pytest.raises(ValueError, match='complete'):
            ledger.assert_prompt_search(**request)
        ledger.record(**{**request, 'start': '2027-01-15T00:00:00Z'}, role='S')
        assert ledger.assert_prompt_search(**request)
        with pytest.raises(ValueError, match='complete'):
            ledger.assert_prompt_search(**{**request, 'symbols': ['BTC', 'ETH']})
        with pytest.raises(ValueError, match='complete'):
            ledger.assert_prompt_search(**{**request, 'batch': 'renamed'})
        ledger.record(**request, role='V')
        with pytest.raises(ValueError, match='Protected'):
            ledger.assert_prompt_search(**request)
    finally:
        ledger.close()


def test_reserved_confirmation_cannot_be_relabelled_as_same_batch_search(tmp_path):
    clock = Clock()
    ledger = ExposureLedger(tmp_path / 'evidence.sqlite', clock=clock)
    try:
        ledger.reserve_confirmation(**ticket())
        clock.move('2027-03-01T00:00:00Z')
        request = {'source': 'exchange', 'symbols': ['BTC'], 'start': '2027-02-01T00:00:00Z',
                   'end': '2027-03-01T00:00:00Z', 'batch': 'A'}
        ledger.record(**request, role='S')
        with pytest.raises(ValueError, match='Frozen confirmation'):
            ledger.assert_prompt_search(**request)
    finally:
        ledger.close()


@pytest.mark.parametrize('change', [{'gain': float('nan')}, {'coverage': float('inf')},
                                    {'complexity': -1}, {'common_support': 'yes'}])
def test_validation_rejects_nonfinite_or_untyped_claims(change):
    row = {'role': 'V', 'common_support': True, 'coverage': 1., 'gain': .006,
           'complexity': 3, 'candidate_id': 'a'}
    with pytest.raises(ValueError):
        select_on_validation([{**row, **change}])


def test_undefined_validation_gain_is_not_selected_and_duplicate_ids_are_rejected():
    row = {'role': 'V', 'common_support': True, 'coverage': 1., 'gain': None,
           'complexity': 3, 'candidate_id': 'a'}
    assert select_on_validation([row]) is None
    with pytest.raises(ValueError, match='unique'):
        select_on_validation([row, row])
