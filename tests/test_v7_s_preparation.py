"""Core original-clock tests with actual local authority; no service claim."""
from dataclasses import asdict

import pytest
from test_v7_s_controller_integration import case as case  # noqa: PLC0414
from test_v7_s_controller_integration import staged as staged  # noqa: PLC0414


def prepare(case, *, lease=None):
    from factorlab.discovery.s_preparation import SPreparationEnvelope
    return SPreparationEnvelope.prepare(case.authority,case.directory,'2026-10',
        config=asdict(case.config),spec=asdict(case.runtime.spec),
        source_reference={'actual_source':case.runtime.feed.identity},schedule_hash=case.schedule.digest,lease=lease)


def test_preparation_keeps_original_lease_before_feature_and_strict_input_seal(case):
    from factorlab.discovery.work_lease import FrozenWorkLease
    original=FrozenWorkLease.issue(300)
    prep=prepare(case,lease=original)
    assert prep.baseline_record()['lease']==original.to_dict()
    assert prep.baseline_record()['counts']['solve']==7
    assert prep.start_preprocessing()['status']=='running'
    prep.seal_input(data_id='artificial',search_exposure={'data_id':'artificial'})
    with pytest.raises(ValueError,match='Immutable'):
        prep.seal_input(data_id='changed',search_exposure={'data_id':'changed'})
    assert case.authority.verify()['reservations']==1


def test_restart_never_reexecutes_preprocessing_or_reissues_baseline(case):
    prep=prepare(case)
    record=prep.start_preprocessing()
    read=prepare(case)
    assert read.baseline_record()==record
    with pytest.raises(ValueError,match='readback only'):
        read.start_preprocessing()
    assert case.authority.verify()['reservations']==1


def test_original_absolute_lease_can_expire_during_feature_build(case,monkeypatch):
    prep=prepare(case)
    record=prep.start_preprocessing()
    monkeypatch.setattr('factorlab.discovery.work_lease.time.monotonic',lambda:record['lease']['monotonic_deadline'])
    with pytest.raises(RuntimeError,match='deadline'):
        prep.seal_input(data_id='not_late',search_exposure={'data_id':'not_late'})
    assert not (prep.root/'input.json').exists()


def test_missing_anchor_cannot_create_replacement_preparation(case):
    prep=prepare(case)
    (prep.root/'identity.json').unlink()
    with pytest.raises((ValueError,OSError)):
        prepare(case)
    assert case.authority.verify()['reservations']==1
