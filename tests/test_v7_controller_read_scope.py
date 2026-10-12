"""Real synthetic journal/prefix; wrapper-only controller object is disclosed.

No provider, fit, certificate, work lease replacement or production mutation.
The full original controller integration is a separate test.
"""
import contextvars
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import pytest
from test_v7_quarantined_observation import factory, ready
from factorlab.discovery import quarantined_observation as observation
from factorlab.discovery.full_trial_controller import FullTrialController


@pytest.mark.parametrize('ending', ['normal', 'error'])
def test_copied_context_cannot_resurrect_closed_observation(factory, ending):
    clock, _, _, runtime, wallet = ready(factory)
    prefix = clock.advance(runtime)
    before = wallet.verify()
    try:
        with observation._original_observation_read_scope(runtime.lifecycle):
            assert observation.current_binding(runtime.lifecycle) == prefix
            copied = contextvars.copy_context()
            if ending == 'error':
                raise RuntimeError('explicit interrupted read')
    except RuntimeError as error:
        assert ending == 'error' and str(error) == 'explicit interrupted read'
    assert observation._READ_SCOPE.get() is None
    with pytest.raises(ValueError, match='expired|another thread'):
        copied.run(observation.current_binding, runtime.lifecycle)
    assert observation.current_binding(runtime.lifecycle) == prefix
    assert wallet.verify() == before


def test_copied_context_cannot_borrow_active_observation_from_other_thread(factory):
    clock, _, _, runtime, wallet = ready(factory)
    prefix = clock.advance(runtime)
    before = wallet.verify()
    with observation._original_observation_read_scope(runtime.lifecycle):
        assert observation.current_binding(runtime.lifecycle) == prefix
        copied = contextvars.copy_context()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(copied.run, observation.current_binding, runtime.lifecycle)
            with pytest.raises(ValueError, match='expired|another thread'):
                future.result(timeout=30)
    assert observation._READ_SCOPE.get() is None and wallet.verify() == before


def test_pure_controller_readback_closes_original_audit_and_reuses_prefix(factory, monkeypatch):
    clock, _, _, runtime, wallet = ready(factory)
    prefix = clock.advance(runtime)
    before = wallet.verify()
    calls = []
    original = observation._verified_publication_sources
    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(observation, '_verified_publication_sources', counted)
    def callback():
        assert observation._READ_SCOPE.get() is not None
        assert observation.current_binding(runtime.lifecycle) == prefix
        assert observation.current_binding(runtime.lifecycle) == prefix
        return ['explicit wrapper-only result']
    obj = SimpleNamespace(schedule=SimpleNamespace(observation_mode='bound_logical_observation'),
        permit=SimpleNamespace(open_lifecycle=lambda: nullcontext(runtime.lifecycle)), _read_steps=callback)
    assert FullTrialController._steps(obj) == ['explicit wrapper-only result']
    assert len(calls) == 2  # full opening and full closing, no lifetime cache
    assert observation._READ_SCOPE.get() is None and wallet.verify() == before


@pytest.mark.parametrize('damage', ['rewrite', 'error'])
def test_pure_controller_readback_rejects_damage_and_leaves_no_scope(factory, damage):
    clock, _, _, runtime, wallet = ready(factory)
    prefix = clock.advance(runtime)
    before = wallet.verify()
    copied = []
    def callback():
        assert observation.current_binding(runtime.lifecycle) == prefix
        copied.append(contextvars.copy_context())
        if damage == 'error':
            raise RuntimeError('explicit interrupted controller read')
        path = clock.directory / 'events/00000.json'
        path.write_bytes(path.read_bytes())  # real publication metadata changes
        return ['must not escape a failed closing audit']
    obj = SimpleNamespace(schedule=SimpleNamespace(observation_mode='bound_logical_observation'),
        permit=SimpleNamespace(open_lifecycle=lambda: nullcontext(runtime.lifecycle)), _read_steps=callback)
    error, message = ((ValueError, 'physical witness|physical publication') if damage == 'rewrite'
                      else (RuntimeError, 'explicit interrupted controller read'))
    with pytest.raises(error, match=message):
        FullTrialController._steps(obj)
    assert observation._READ_SCOPE.get() is None
    with pytest.raises(ValueError, match='expired|another thread'):
        copied[0].run(observation.current_binding, runtime.lifecycle)
    assert wallet.verify() == before


def test_physical_controller_readback_does_not_open_observation_scope():
    def forbidden():
        pytest.fail('Physical controller readback must not open a logical prefix scope')
    obj = SimpleNamespace(schedule=SimpleNamespace(observation_mode='physical_UTC'),
        permit=SimpleNamespace(open_lifecycle=forbidden), _read_steps=lambda: ['physical result'])
    assert FullTrialController._steps(obj) == ['physical result']
    assert observation._READ_SCOPE.get() is None


@pytest.mark.parametrize('owner', ['thread', 'pid'])
def test_read_scope_rejects_damaged_owner_header(factory, owner):
    clock, _, _, runtime, wallet = ready(factory)
    prefix = clock.advance(runtime)
    before = wallet.verify()
    with observation._original_observation_read_scope(runtime.lifecycle):
        assert observation.current_binding(runtime.lifecycle) == prefix
        scope = observation._READ_SCOPE.get()
        original = scope[owner]
        scope[owner] = original + 1  # explicit private-header damage, no clock replacement
        try:
            with pytest.raises(ValueError, match='expired|another thread|process'):
                observation.current_binding(runtime.lifecycle)
        finally:
            scope[owner] = original
    assert observation._READ_SCOPE.get() is None
    assert wallet.verify() == before
