"""Authorization plumbing only: no model, GPT, or scientific claim."""
import json
from dataclasses import replace
import pandas as pd
import pytest

from test_v7_neural_forward_contract import freeze, plan
from test_v7_lifecycle_audit import _loss_rows
from test_v7_lifecycle_audit import audit_store as audit_store
from factorlab.discovery.neural_forward import FrozenNeuralRecipe
from factorlab.discovery.neural_forward_contract import assert_frozen_forward_fit
from factorlab.discovery.neural_forward_ticket import _verify_historical_context, _LifecycleReader


def v_context(store, clock):
    recipe=FrozenNeuralRecipe()
    freeze(store, recipe)
    clock[0]=pd.Timestamp('2026-11-01T00:00:05Z')
    job=plan(store, recipe, '2026-11-01T00:00:00Z', 'A:'+recipe.pipeline_id)
    return recipe,job,assert_frozen_forward_fit(store,job,recipe,purpose='V')['binding']


def test_original_authorization_is_stable_and_reloads_after_v_ends(audit_store):
    store,clock=audit_store
    recipe,job,context=v_context(store,clock)
    before=store.verify()
    assert context['version']=='v7-neural-forward-context-v3'
    assert context['authorized_at']==clock[0].isoformat()
    clock[0]+=pd.Timedelta(hours=1)
    assert assert_frozen_forward_fit(store,job,recipe,purpose='V')['binding']==context
    assert store.verify()==before
    clock[0]=pd.Timestamp('2027-03-01T00:00:00Z')
    _verify_historical_context(store,context,recipe)
    with pytest.raises(ValueError,match='window'):
        assert_frozen_forward_fit(store,job,recipe,purpose='V')
    assert store.verify()==before


@pytest.mark.parametrize('change', ['missing','forged','old-version'])
def test_caller_cannot_supply_or_erase_authorization(audit_store,change):
    store,clock=audit_store
    recipe,job,context=v_context(store,clock)
    if change=='missing':context.pop('authorized_at')
    elif change=='forged':context['authorized_at']='2026-11-01T00:00:06+00:00'
    else:context['version']='v7-neural-forward-context-v2'
    with pytest.raises(ValueError):_verify_historical_context(store,context,recipe)


def test_hash_chained_duplicate_authorization_is_rejected(audit_store):
    store,clock=audit_store
    recipe,job,context=v_context(store,clock)
    base={k:v for k,v in context.items() if k!='authorized_at'}
    with store.db:store._event('neural_forward_authorized',{'context':base})
    with pytest.raises(ValueError,match='authorization'):
        _verify_historical_context(store,context,recipe)
    with pytest.raises(ValueError,match='authorization'):
        assert_frozen_forward_fit(store,job,recipe,purpose='V')


def test_reader_cannot_mint_unregistered_protected_authorization(audit_store):
    store,clock=audit_store
    recipe=FrozenNeuralRecipe()
    freeze(store,recipe)
    clock[0]=pd.Timestamp('2026-11-01T00:00:05Z')
    job=plan(store,recipe,'2026-11-01T00:00:00Z','A:'+recipe.pipeline_id)
    reader=_LifecycleReader(store.directory)
    # Reader's own clock remains the actual wall clock; expose the same fixture
    # clock only for this missing-registration plumbing check.
    reader.clock=lambda:clock[0]
    before=store.verify()
    try:
        with pytest.raises(ValueError,match='original authorization'):
            assert_frozen_forward_fit(reader,job,recipe,purpose='V')
    finally:reader.close()
    assert store.verify()==before


def test_c_authorization_is_actual_activation_time_and_historical_only(audit_store,monkeypatch):
    store,clock=audit_store
    recipe=FrozenNeuralRecipe()
    freeze(store,recipe)
    clock[0]=pd.Timestamp('2026-12-01T00:00:05Z')
    monkeypatch.setattr(store,'_losses',lambda stream,start,end,as_of,**kwargs:
        _loss_rows(start,end,1. if stream.endswith(':reference') else .9))
    store.select_validation('A')
    job=plan(store,recipe,'2026-12-01T00:00:00Z','A:'+recipe.pipeline_id)
    monkeypatch.setattr(store,'calibration_ready',lambda:True)
    store.reserve_selected_confirmation('A')
    reservation=json.loads(store.exposure.db.execute('SELECT identity FROM confirmations').fetchone()[0])
    with pytest.raises(ValueError,match='window'):
        assert_frozen_forward_fit(store,job,recipe,purpose='C')
    clock[0]=pd.Timestamp(reservation['actual_start'])
    context=assert_frozen_forward_fit(store,job,recipe,purpose='C')['binding']
    assert context['authorized_at']==reservation['actual_start']
    clock[0]=pd.Timestamp(reservation['end'])+pd.Timedelta(days=1)
    _verify_historical_context(store,context,recipe)
    with pytest.raises(ValueError,match='window'):
        assert_frozen_forward_fit(store,job,recipe,purpose='C')


def test_elapsed_cutoff_before_v_boundary_keeps_actual_authorization(tmp_path,monkeypatch):
    from factorlab import protocol
    from factorlab.discovery.v7_forward import ForwardRuntime,ForwardConfig
    monkeypatch.setattr(protocol,'ROOT',tmp_path)
    (tmp_path/'factorlab').mkdir()
    (tmp_path/'factorlab'/'fixture.py').write_text('SYNTHETIC=True')
    (tmp_path/'pyproject.toml').write_text('[project]')
    (tmp_path/'uv.lock').write_text('version=1')
    clock=[pd.Timestamp('2026-10-31T12:34:56Z')]
    runtime=ForwardRuntime(tmp_path/'forward',tmp_path/'feed',
        config=ForwardConfig(training_window_days=90,refit_interval_days=15),clock=lambda:clock[0])
    try:
        store=runtime.lifecycle
        anchor=store.due_fit_cutoff()
        recipe=FrozenNeuralRecipe(rolling_spec=runtime.spec)
        store.exposure.record(source=store.plan.source_id,symbols=store.plan.symbols,
            start='2026-07-01T00:00:00Z',end='2026-10-01T00:00:00Z',role='S',batch='A')
        pipeline=runtime.recipe_for(recipe)
        store.freeze_batch(batch='A',ordinal=1,start='2026-11-01T00:00:00Z',
            candidates=[{**pipeline,'complexity':2}],reference=pipeline,
            evidence_sources=[{'start':'2026-07-01T00:00:00Z','end':'2026-10-01T00:00:00Z'}])
        clock[0]=pd.Timestamp('2026-11-01T00:00:05Z')
        job=store.plan_fit(anchor,'A:'+recipe.pipeline_id,pipeline)
        context=assert_frozen_forward_fit(store,job,recipe,purpose='V')['binding']
        assert pd.Timestamp(context['cutoff'])<pd.Timestamp('2026-11-01T00:00:00Z')
        assert context['authorized_at']==clock[0].isoformat()
        clock[0]=pd.Timestamp('2027-01-01T00:00:00Z')
        _verify_historical_context(store,context,recipe)
    finally:runtime.close()
