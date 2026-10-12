"""Pure recipe identity works inside a real restrictive runtime scope."""
import pytest
from test_policy_admission_headroom import actual_service
from factorlab.discovery.forward_service_scope import _service_step_scope, ServiceDeadlineClosed
from factorlab.discovery.pool_forward import PoolForwardRecipe, pipeline_for
from factorlab.discovery.joint_pool import FrozenPool
from factorlab.discovery.v7_forward import ForwardRuntime


def test_pure_pool_identity_inside_actual_service_grants_no_work(actual_service):
    c = actual_service
    recipe = PoolForwardRecipe(FrozenPool(role='reference', version='K0_v1'), c.runtime.spec)
    before = c.wallet.verify()
    with _service_step_scope(c.service):
        pure = pipeline_for(recipe)
        assert pure == c.runtime.recipe_for(recipe.pool)
        assert c.wallet.verify() == before
    assert c.wallet.verify() == before


def test_actual_foreign_runtime_work_still_rejected_before_planning(actual_service, tmp_path):
    c = actual_service
    foreign = ForwardRuntime(tmp_path/'foreign-runtime', tmp_path/'foreign-feed')
    try:
        before = c.wallet.verify()
        count = foreign.lifecycle.db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]
        with _service_step_scope(c.service):
            with pytest.raises(ValueError, match='scope identity changed') as rejected:
                foreign.fit_month(c.cutoff, authority=c.wallet)
            assert type(rejected.value) is ValueError
        assert foreign.lifecycle.db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == count
        assert c.wallet.verify() == before
    finally:
        foreign.close()
