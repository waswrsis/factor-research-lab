"""Bounded canonical wiring on artificial clocks/data; no confirmation claim."""
import time

import pandas as pd

from test_v7_dynamic_forward_gates import gate_case
from factorlab.discovery.claim_protocol import ClaimProtocol, FrozenClaimEndpoint
from factorlab.discovery.known_linear_reference import FrozenLinearRecipe
from factorlab.discovery.v7_service import ForwardService, ServiceConfig


def test_frozen_policy_trial_plans_a_distinct_actual_stream(gate_case):
    runtime, wallet, _, _ = gate_case
    k0 = runtime.recipe_for()
    start = '2026-12-01T00:00:00+00:00'
    ClaimProtocol.freeze(runtime.lifecycle, [FrozenClaimEndpoint.library(start, k0,
        runtime.recipe_for(FrozenLinearRecipe(rolling_spec=runtime.spec))), FrozenClaimEndpoint.algorithm(start, k0)])
    # The test already owns an explicitly artificial runtime clock. No public
    # production selector accepts it or imports a successful claim from it.
    runtime.lifecycle.clock = runtime.clock = lambda: pd.Timestamp('2026-12-02T00:00:00+00:00')
    service = ForwardService(runtime, work_authority=wallet, config=ServiceConfig(poll=False))
    try:
        service._plan_month(runtime.lifecycle.now())
        rows = runtime.lifecycle.db.execute('SELECT stream,pipeline FROM jobs WHERE cutoff=?', (start,)).fetchall()
        from factorlab.discovery.v7_lifecycle import canonical
        assert ('A0_policy_trial', canonical(k0)) in rows
        assert ('A0', canonical(k0)) in rows
        allowed = runtime.claim_streams(runtime.lifecycle.now(), runtime.lifecycle.now())
        assert allowed['A0_policy_trial'] == k0
        assert wallet.verify()['reservations'] == 0
        assert len(list(runtime.lifecycle.exposure.db.execute('SELECT identity FROM confirmations'))) == 2
    finally:
        service.close(close_runtime=False)


def test_unregistered_prospective_resource_cannot_create_a_wallet():
    import pytest
    from factorlab.discovery.prospective_resource_work import pending_prospective_resource
    with pytest.raises(TypeError, match='Original typed validation'):
        pending_prospective_resource({'selected': True, 'gates': {'shared_S': {}}})
