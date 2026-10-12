"""Canonical real-wallet regressions; deterministic private final-IO clock seams."""
from pathlib import Path
import os
import pandas as pd
import pytest

from test_v7_structural_evolution import campaign, module, own
from factorlab.discovery import structural_review
from factorlab.discovery.work_authority_bridge import WorkAuthorityBridge
from factorlab.discovery.work_lease import FrozenWorkLease
from factorlab.discovery.s_nonlinear_ticket import freeze_nonlinear_plan


def test_review_postcommit_expiry_cannot_authorize_full_s(campaign, module, monkeypatch):
    make, _, _, authority = campaign
    bridge, recipe = make('2025-05')
    own(bridge)
    original_append = WorkAuthorityBridge._append
    original_check = FrozenWorkLease.assert_parent_alive
    expired = {'digest': None}

    def append(self, reservation, operation, args):
        result = original_append(self, reservation, operation, args)
        if operation == 'complete' and reservation['binding']['work'].get('kind') == 'structure_review':
            expired['digest'] = reservation['lease_hash']
        return result

    def check(self, *args, **kwargs):
        if self.digest == expired['digest']:
            raise TimeoutError('deterministic original review postcommit expiry')
        return original_check(self, *args, **kwargs)

    monkeypatch.setattr(WorkAuthorityBridge, '_append', append)
    monkeypatch.setattr(FrozenWorkLease, 'assert_parent_alive', check)
    with pytest.raises(TimeoutError, match='postcommit expiry'):
        module.select_structure(bridge, recipe)
    selection = bridge.state['structural_plan']
    row = authority.get(selection['body']['review_reservation'])
    assert row['status'] == 'completed'
    before = authority.status(row['actual_month'])
    with pytest.raises(ValueError, match='Missing/bounded'):
        structural_review.verify_review_completion(authority, selection)
    with pytest.raises(ValueError, match='Missing/bounded'):
        freeze_nonlinear_plan(bridge, recipe, kinds=('complete_neural_S', 'typed_queue_review'))
    with pytest.raises(ValueError, match='Missing/bounded'):
        module.select_structure(bridge, recipe)
    assert authority.status(row['actual_month']) == before


def test_review_final_marker_late_write_cannot_return_success(campaign, module, monkeypatch):
    make, _, _, authority = campaign
    bridge, recipe = make('2025-05')
    own(bridge)
    original = structural_review.immutable_record

    def delayed(path, value):
        original(path, value)
        if Path(path).name == 'structure_review_final_io.json':
            row = authority.get(value['body']['reservation_id'])
            late = pd.Timestamp(row['lease']['wall_deadline']).value + 1_000_000_000
            os.utime(path, ns=(late, late))

    monkeypatch.setattr(structural_review, 'immutable_record', delayed)
    with pytest.raises(ValueError, match='final publication IO'):
        module.select_structure(bridge, recipe)
    selection = bridge.state['structural_plan']
    row = authority.get(selection['body']['review_reservation'])
    assert row['status'] == 'completed'
    before = authority.status(row['actual_month'])
    with pytest.raises(ValueError, match='final publication IO'):
        structural_review.verify_review_completion(authority, selection)
    with pytest.raises(ValueError, match='final publication IO'):
        freeze_nonlinear_plan(bridge, recipe, kinds=('complete_neural_S', 'typed_queue_review'))
    assert authority.status(row['actual_month']) == before


def test_review_final_receipts_and_terminal_step_are_required(campaign, module):
    make, _, _, authority = campaign
    bridge, recipe = make('2025-05')
    own(bridge)
    selection = module.select_structure(bridge, recipe)
    row = structural_review.verify_review_completion(authority, selection)
    before = authority.status(row['actual_month'])
    assert module.select_structure(bridge, recipe) == selection
    path = bridge.directory/'structure_review_final_io.json'
    raw = path.read_bytes()
    path.unlink()
    with pytest.raises(ValueError, match='Missing/bounded'):
        structural_review.verify_review_completion(authority, selection)
    path.write_bytes(raw)
    step = bridge.directory/'steps'/f'{bridge._sequence:06d}.json'
    step_raw = step.read_bytes()
    step.unlink()
    with pytest.raises(ValueError, match='Missing/damaged bridge receipt'):
        structural_review.verify_review_completion(authority, selection)
    step.write_bytes(step_raw)
    assert structural_review.verify_review_completion(authority, selection) == row
    assert authority.status(row['actual_month']) == before
