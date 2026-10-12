"""Independent final V/C integrity audit; synthetic loss plumbing, no real gate."""
import json

import pandas as pd
import pytest
from test_v7_lifecycle_audit import _freeze, _loss_rows

from factorlab.discovery import v7_lifecycle
from factorlab.protocol import fingerprint


@pytest.fixture
def audit_store(tmp_path, monkeypatch):
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    clock = [pd.Timestamp('2026-10-01T00:00:05Z')]
    store = v7_lifecycle.Lifecycle(tmp_path / 'audit', v7_lifecycle.LifecyclePlan('audit-feed'),
                                  clock=lambda: clock[0])
    yield store, clock
    store.close()


def selected(store, clock, monkeypatch):
    _freeze(store)
    clock[0] = pd.Timestamp('2026-12-01T00:00:05Z')
    monkeypatch.setattr(store, '_losses', lambda stream, start, end, as_of, **kw:
        _loss_rows(start, end, 1. if stream.endswith(':reference') else .9))
    store.select_validation('A')
    monkeypatch.setattr(store, 'calibration_ready', lambda: True)
    store.reserve_selected_confirmation('A')


def concluded(store, clock, monkeypatch):
    selected(store, clock, monkeypatch)
    clock[0] = pd.Timestamp('2027-03-03T00:00:00Z')
    # Deterministic synthetic rejection. This mock is never a production gate.
    monkeypatch.setattr(v7_lifecycle, 'paired_effect_interval',
                        lambda *a, **kw: {'supported': False, 'status': 'indistinguishable'})
    result = store.conclude_confirmation('A')
    store.verify()
    return result


@pytest.mark.parametrize('target', ['batch_result', 'batch_status', 'batch_selected', 'exposure_result',
                                  'claim_file', 'validation_file', 'reservation', 'batch_delete'])
def test_final_indexes_and_artifacts_cannot_forge_a0_adoption(audit_store, monkeypatch, target):
    store, clock = audit_store
    result = concluded(store, clock, monkeypatch)
    altered = {**result, 'supported': True, 'status': 'supported_target'}
    if target == 'batch_result':
        with store.db:
            store.db.execute('UPDATE batches SET result=?', (json.dumps(altered),))
    elif target == 'batch_status':
        with store.db:
            store.db.execute("UPDATE batches SET status='selected_C'")
    elif target == 'batch_selected':
        with store.db:
            store.db.execute("UPDATE batches SET selected=NULL")
    elif target == 'exposure_result':
        with store.exposure.db:
            store.exposure.db.execute('UPDATE confirmations SET result=?', (json.dumps(altered),))
    elif target == 'claim_file':
        (store.directory / 'claims' / f'{fingerprint("A")}.json').write_text(json.dumps(altered))
    elif target == 'validation_file':
        (store.directory / 'validation' / f'{fingerprint("A")}.json').write_text('{}')
    elif target == 'reservation':
        identity = json.loads(store.exposure.db.execute('SELECT identity FROM confirmations').fetchone()[0])
        identity['candidate_id'] = 'different-candidate'
        with store.exposure.db:
            store.exposure.db.execute('UPDATE confirmations SET identity=?', (json.dumps(identity),))
    else:
        with store.db:
            store.db.execute('DELETE FROM batches')
    with pytest.raises(ValueError):
        store.verify()
    # Calling adoption directly must not bypass verify, even with a mocked gate.
    with pytest.raises(ValueError):
        store.adoption_pipeline('2027-03-01T00:00:00Z', {'pipeline_id': 'K0'})


@pytest.mark.parametrize('boundary', ['score_commit', 'exposure_finish', 'file_publish'])
def test_confirmation_committed_content_recovers_once_without_rescoring(audit_store, monkeypatch, boundary):
    store, clock = audit_store
    selected(store, clock, monkeypatch)
    clock[0] = pd.Timestamp('2027-03-03T00:00:00Z')
    calls = []
    monkeypatch.setattr(v7_lifecycle, 'paired_effect_interval', lambda *a, **kw:
                        calls.append(True) or {'supported': False, 'status': 'indistinguishable'})
    original_finish, original_publish = store.exposure.finish_confirmation, v7_lifecycle.immutable_record

    def finish(*args):
        if boundary == 'score_commit':
            raise OSError('synthetic interrupted before exposure publication')
        original_finish(*args)
        if boundary == 'exposure_finish':
            raise OSError('synthetic interrupted after exposure publication')

    def publish(path, value):
        original_publish(path, value)
        if path.parent.name == 'claims' and boundary == 'file_publish':
            raise OSError('synthetic interrupted after file publication')

    monkeypatch.setattr(store.exposure, 'finish_confirmation', finish)
    monkeypatch.setattr(v7_lifecycle, 'immutable_record', publish)
    with pytest.raises(OSError, match='interrupted'):
        store.conclude_confirmation('A')
    store.verify()  # Valid partial publication, not a second scoring permission.
    monkeypatch.setattr(store.exposure, 'finish_confirmation', original_finish)
    monkeypatch.setattr(v7_lifecycle, 'immutable_record', original_publish)
    result = store.recover_confirmation('A')
    assert len(calls) == 1 and result['supported'] is False
    assert store.conclude_confirmation('A') == result and len(calls) == 1
    assert store._batch_operation('A', 'confirmation_score_committed')['result'] == result
    store.verify()


@pytest.mark.parametrize('committed', [False, True])
def test_consumed_validation_recovers_only_existing_score(audit_store, monkeypatch, committed):
    store, clock = audit_store
    _freeze(store)
    clock[0] = pd.Timestamp('2026-12-01T00:00:00Z')
    calls = []
    original_publish = v7_lifecycle.immutable_record

    def losses(stream, start, end, as_of, **kwargs):
        calls.append(stream)
        if not committed:
            raise OSError('synthetic pre-score crash')
        return _loss_rows(start, end, 1. if stream.endswith(':reference') else .9)

    def publish(path, value):
        if path.parent.name == 'validation':
            raise OSError('synthetic post-score crash')
        return original_publish(path, value)

    monkeypatch.setattr(store, '_losses', losses)
    monkeypatch.setattr(v7_lifecycle, 'immutable_record', publish)
    with pytest.raises(OSError):
        store.select_validation('A')
    count = len(calls)
    store.verify()
    monkeypatch.setattr(v7_lifecycle, 'immutable_record', original_publish)
    result = store.recover_validation('A')
    assert (result is not None) is committed and len(calls) == count
    store.verify()


def test_confirmation_reservation_intent_survives_second_database_interruption(audit_store, monkeypatch):
    store, clock = audit_store
    _freeze(store)
    clock[0] = pd.Timestamp('2026-12-01T00:00:05Z')
    monkeypatch.setattr(store, '_losses', lambda stream, start, end, as_of, **kw:
        _loss_rows(start, end, 1. if stream.endswith(':reference') else .9))
    store.select_validation('A')
    monkeypatch.setattr(store, 'calibration_ready', lambda: True)
    original = store.exposure.reserve_confirmation
    monkeypatch.setattr(store.exposure, 'reserve_confirmation', lambda **kwargs:
                        (_ for _ in ()).throw(OSError('synthetic reserve interruption')))
    with pytest.raises(OSError):
        store.reserve_selected_confirmation('A')
    intent = store._batch_operation('A', 'confirmation_reserved')
    store.verify()
    clock[0] += pd.Timedelta(seconds=10)
    monkeypatch.setattr(store.exposure, 'reserve_confirmation', original)
    store.reserve_selected_confirmation('A')
    assert store._batch_operation('A', 'confirmation_reserved') == intent
    store.verify()


def test_actual_interval_plumbing_adopts_only_after_completion_and_with_gate(audit_store, monkeypatch):
    store, clock = audit_store
    selected(store, clock, monkeypatch)
    assert store.recover_confirmation('A') is None
    clock[0] = pd.Timestamp('2027-03-03T00:00:00Z')
    result = store.conclude_confirmation('A')
    assert result['supported'] is True  # Fixed synthetic paired losses, mocked calibration only.
    assert store.adoption_pipeline('2027-03-01T00:00Z', {'pipeline_id': 'K0'})['pipeline_id'] == 'K0'
    clock[0] = pd.Timestamp('2027-04-01T00:00:00Z')
    assert store.adoption_pipeline(clock[0], {'pipeline_id': 'K0'})['pipeline_id'] == 'factor-A'
    monkeypatch.setattr(store, 'calibration_ready', lambda: False)
    with pytest.raises(ValueError, match='certificate'):
        store.adoption_pipeline(clock[0], {'pipeline_id': 'K0'})
