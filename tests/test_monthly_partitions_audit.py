"""Independent real-wallet routing counterexamples; no market or fitting."""
import pytest
from test_monthly_partitions import forward, wallet
from test_monthly_partitions import root as root  # noqa: PLC0414 -- pytest fixture

from factorlab.discovery.forward_preparation import ForwardPreparation
from factorlab.protocol import fingerprint


def test_known_job_cannot_be_relabelled_as_arbitrary_neural_forward_work(root):
    authority = wallet(root)
    runtime, recipe, cutoff = forward(root, authority)
    try:
        job = runtime.lifecycle.plan_fit(cutoff, 'known:HAR_cal', runtime.recipe_for(recipe))
        # A real known-linear intent creates the genuine runtime/wallet binding.
        preparation = ForwardPreparation.begin(runtime, authority, job, recipe)
        authority.fail_consumed(preparation.record['reservation_id'], 'artificial-no-snapshot-or-fit')
        before = authority.verify()
        binding = {'ticket_terms': fingerprint('no real neural ticket'), 'job': job,
                   'pipeline_id': fingerprint('unrelated neural recipe'),
                   'snapshot_id': fingerprint('no actual snapshot'), 'purpose': 'forward'}
        with pytest.raises(ValueError):
            authority.reserve('neural_forward:'+fingerprint('arbitrary repeated work'),
                              counts={'neural_fit': 1, 'epoch': 1, 'solve': 1},
                              lease_seconds=1, binding=binding)
        assert authority.verify() == before
    finally:
        runtime.close()
