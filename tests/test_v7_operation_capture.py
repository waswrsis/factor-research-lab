"""Real original anchors on artificial samples; no formal statistical claim."""
from contextvars import copy_context
import json

import pytest

from test_v7_quarantined_observation import factory, ready
from factorlab.discovery import finite_market_oracle as finite
from factorlab.discovery import quarantined_observation as observation
from factorlab.protocol import fingerprint



def test_closed_context_cannot_publish_next_observation_batch(factory):
    clock, gen, permit, runtime, wallet = ready(factory)
    with observation._original_observation_read_scope(runtime.lifecycle):
        clock.inspect(runtime.lifecycle)
        saved = copy_context()
    before = wallet.verify()
    with pytest.raises(ValueError, match='expired or belongs'):
        saved.run(clock.advance_batch, runtime, max_events=1)
    assert clock.inspect()['events'] == 0 and wallet.verify() == before


def test_fresh_batch_detects_rehashed_atom_damage_after_previous_call(factory):
    gen, permit, runtime, wallet = factory()
    clock = observation.QuarantinedObservation.bind(runtime, gen, permit)
    assert len(finite.publish_due(gen, permit, max_new_receipts=24)['issued']) == 24
    clock.advance_batch(runtime, max_events=1)
    original = gen.directory/'atoms/evaluation/0000.json'
    value = json.loads(original.read_text('utf8'))
    value['record']['rows'][0]['close'] *= 1.01
    value['hash'] = fingerprint(value['record'])
    original.write_text(json.dumps(value), encoding='utf8')
    with pytest.raises(ValueError, match='atom changed'):
        clock.advance_batch(runtime, max_events=1)


def test_batch_closing_audit_rejects_mid_batch_original_damage(factory, monkeypatch):
    gen, permit, runtime, wallet = factory()
    clock = observation.QuarantinedObservation.bind(runtime, gen, permit)
    finite.publish_due(gen, permit, max_new_receipts=24)
    before = wallet.verify()
    original_write = observation.immutable_record
    damaged = []
    def write_and_damage(path, value):
        result = original_write(path, value)
        if path == clock.mirror/'commits/00000.json':
            anchor = permit.paths['root']/'generator_publications/00000-complete.json'
            saved = json.loads(anchor.read_text('utf8'))
            saved['record']['response_hash'] = '0'*64
            saved['hash'] = fingerprint(saved['record'])
            anchor.write_text(json.dumps(saved), encoding='utf8')
            damaged.append(True)
        return result
    monkeypatch.setattr(observation, 'immutable_record', write_and_damage)
    with pytest.raises(ValueError, match='differs from its mirror'):
        clock.advance_batch(runtime, max_events=1)
    assert damaged == [True] and wallet.verify() == before
    with pytest.raises(ValueError):
        clock.inspect()

