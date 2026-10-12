"""Real artificial receipt/authority bindings; resource gates remain a private stub.

These counterexamples are engineering evidence, never Linux resource acceptance
or market alpha. The imported fixture does not replace the ticket verifier.
"""
import copy
import json
import os
import time

import pytest
from test_v7_neural_forward_launcher import real_ticket_case as real_ticket_case  # noqa: PLC0414

from factorlab.discovery import neural_forward_ticket as module
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
from factorlab.protocol import fingerprint

ORIGINAL_GATE_CHECK = module._verify_gates


def rewrite(ticket, value):
    ticket.path.write_text(json.dumps({'value': value, 'sha256': fingerprint(value)}), 'utf-8')
    return module.ForwardFitTicket.load(ticket.path)


def test_same_ticket_retry_preserves_original_reservation_and_lease(real_ticket_case):
    c = real_ticket_case
    first = c.authority.get(c.ticket.value['reservation_id'])
    before = c.authority.status(first['actual_month'])
    second = module.issue_forward_ticket(c.runtime, c.authority, first['binding']['job'], c.recipe,
                                        purpose='resource_trial', gates={})
    assert second.digest == c.ticket.digest and second.value['lease'] == first['lease']
    assert c.authority.status(first['actual_month']) == before


@pytest.mark.parametrize('field', ['cutoff', 'source_id', 'symbols', 'job', 'purpose'])
def test_rehashed_ticket_cannot_change_prepaid_context(real_ticket_case, field):
    c = real_ticket_case
    value = copy.deepcopy(c.ticket.value)
    value['context'][field] = ['ETHUSDT'] if field == 'symbols' else 'changed'
    if field == 'job':
        with pytest.raises(ValueError, match='canonical'):
            rewrite(c.ticket, value)
    else:
        changed = rewrite(c.ticket, value)
        with pytest.raises(ValueError, match='prepaid authority'):
            changed.verify_for_launch()


@pytest.mark.parametrize('kind', ['snapshot', 'source', 'payload', 'binding_file'])
def test_actual_cutoff_inputs_and_wallet_binding_cannot_be_replaced(real_ticket_case, kind):
    c = real_ticket_case
    if kind == 'snapshot':
        path = c.runtime.directory / 'snapshots' / (c.ticket.value['context']['job'] + '.json')
        value = json.loads(path.read_text())
        value['cutoff'] = '2026-09-01T00:00:00+00:00'
        path.write_text(json.dumps(value))
    elif kind == 'source':
        (c.runtime.directory / 'source/factorlab/v7.py').write_text('changed-source')
    elif kind == 'payload':
        from pathlib import Path
        Path(c.ticket.value['data']['arrays']).write_bytes(b'changed-payload')
    else:
        (c.runtime.directory / 'work_authority.json').unlink()
    with pytest.raises((ValueError, OSError)):
        c.ticket.verify_for_launch()


@pytest.mark.parametrize('scope', ['task', 'universe', 'run'])
def test_issuance_cannot_swap_wallet_scope_for_an_existing_runtime(real_ticket_case, scope):
    c = real_ticket_case
    wrong = MonthlyWorkAuthority(c.authority.directory.parent / 'other-authority',
        run_id='other-run', task='events' if scope == 'task' else 'volatility',
        universe=['ETHUSDT'] if scope == 'universe' else ['BTCUSDT'])
    with pytest.raises(ValueError, match='authority|another task'):
        module.issue_forward_ticket(c.runtime, wrong, c.ticket.value['context']['job'], c.recipe,
                                    purpose='resource_trial', gates={})
    assert all(n == 0 for n in wrong.status(c.authority.get(c.ticket.value['reservation_id'])['actual_month'])['counts'].values())


def test_completed_binding_can_verify_after_old_parent_expires_without_new_work(real_ticket_case, monkeypatch):
    c = real_ticket_case
    record = c.authority.get(c.ticket.value['reservation_id'])
    c.authority.start(record['reservation_id'], {'pid': os.getpid() + 100000, 'created': time.time()})
    c.authority.complete(record['reservation_id'], result={'fixture': 'no-model-admission'}, actual_counts={})
    before = c.authority.status(record['actual_month'])
    monkeypatch.setattr(module.FrozenWorkLease, 'assert_parent_alive',
                        lambda *a, **k: pytest.fail('Historical binding must not revive the lease'))
    actual = c.ticket.verify_completed_binding()
    assert actual['reservation']['status'] == 'completed'
    assert actual['context']['purpose'] == 'resource_trial'
    assert c.authority.status(record['actual_month']) == before


@pytest.mark.parametrize('text', ['{"x":1,"x":2}', '{"x":NaN}'])
def test_ticket_reader_rejects_ambiguous_or_nonfinite_json(real_ticket_case, text):
    c = real_ticket_case
    c.ticket.path.write_text(text)
    with pytest.raises(ValueError, match='Duplicate|Nonfinite'):
        module.ForwardFitTicket.load(c.ticket.path)


@pytest.mark.parametrize('late_part', ['accounting', 'final_io', 'publication', 'publication_store', 'none'])
def test_resource_proof_must_be_fully_published_by_next_cutoff(real_ticket_case, monkeypatch, late_part):
    c = real_ticket_case
    # Private gate/admission seams isolate this cutoff comparison, not a
    # substitute for genuine quarterly or Linux acceptance evidence.
    _, _, data, _ = c.ticket.verify_for_launch()
    source = {'frozen.py': 'a' * 64}
    monkeypatch.setattr('factorlab.discovery.nonlinear_runtime.verify_pre_resource_artifacts',
        lambda *a, **k: {'test_only': False, 'context': {'source': {'files': source}}})
    early, late = '2026-09-30T23:59:00Z', '2026-10-01T00:00:01Z'
    import pandas as pd
    monkeypatch.setattr(module, '_publication_mtime_ns',
                        lambda *a: pd.Timestamp(late if late_part == 'publication_store' else early).value)
    resource = {'context': {'purpose': 'resource_trial', 'source_id': c.runtime.feed.identity},
        'recipe': c.recipe, 'data': data, 'source_hash': fingerprint(source),
        'reservation': {'completed_at': late if late_part == 'accounting' else early},
        'receipt': {'final_io_observation': {'wall': late if late_part == 'final_io' else early}},
        'publication': {'observation': {'wall': late if late_part == 'publication' else early}}}
    monkeypatch.setattr('factorlab.discovery.neural_forward_admission.verify_completed_artifact',
                        lambda *a, **k: resource)
    context = {**c.ticket.value['context'], 'purpose': 'V'}
    gates = {'directory': str(c.runtime.directory), 'artifacts': {},
             'resource_certificate': {'ticket': {'path': str(c.ticket.path)}}}
    if late_part == 'none':
        assert ORIGINAL_GATE_CHECK(gates, data, c.recipe, context)['test_only'] is False
    else:
        with pytest.raises(ValueError, match='resource proof'):
            ORIGINAL_GATE_CHECK(gates, data, c.recipe, context)
