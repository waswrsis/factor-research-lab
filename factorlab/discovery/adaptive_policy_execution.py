"""Execute already enrolled full-policy oracle paths without creating new trials.

This is a full adaptive EXECUTION census, not an oracle for the policy's risk.
Unknown provider response law and the full-policy expected-risk estimand remain
unsupported. Neither this result nor any oracle outcome is fed back to selection.
"""
from __future__ import annotations

from factorlab.protocol import fingerprint
from .full_calibration_registry import FullTrialRegistry, QuarantinedTrialPermit

VERSION = 'v7-original-adaptive-policy-execution-census-v1'


def _controller(permit):
    from .full_trial_controller import FullTrialController
    if type(permit) is not QuarantinedTrialPermit or permit.trial['role'] != 'oracle':
        raise TypeError('Actual original oracle-index trial permit required')
    permit.verify()
    return FullTrialController.inspect(permit)


def advance_original_policy(permit, *, max_pumps=7200):
    """Bounded original feed/research/service/wait/claim steps on its own wallet.

    No enrollment, new schedule, wallet creation, lease extension, retry callback,
    loss input, provider substitute or alternate command is accepted here. A
    failed/orphan original step remains consumed by FullTrialController itself.
    """
    if type(max_pumps) is not int or not 1 <= max_pumps <= 7200:
        raise ValueError('Bounded positive original pump count required')
    controller = _controller(permit)
    progress = controller.run(max_pumps=max_pumps)['progress']
    return {'version': VERSION, 'trial': permit.trial,
        'controller_digest': fingerprint(controller.identity), 'progress': progress,
        'work_clock': 'physical_UTC_original_wallet_and_leases',
        'full_adaptive_risk_truth': None, 'formal_gate_eligible': False}


def read_policy_execution(registry, cell, index):
    """Whole preregistered denominator; missing/failed paths are never deleted."""
    if type(registry) is not FullTrialRegistry:
        raise TypeError('Actual original full registry required')
    registry.verify()
    validation_trial = registry.identity_for(cell, index)
    rows = []
    from .full_trial_certificate import _claims, _search
    from .neural_forward_ticket import open_existing_authority
    for atom in range(registry.plan.oracle_paths):
        trial = registry.identity_for(cell, index, oracle_index=atom)
        paths = registry.paths(cell, index, oracle_index=atom)
        row = {'atom': atom, 'trial_id': trial['trial_id']}
        if not (paths['root']/'enrollment.json').is_file():
            row.update(state='missing_original_enrollment')
        elif not (paths['root']/'controller'/'identity.json').is_file():
            # Still verify original enrollment; do not repair a partial controller.
            QuarantinedTrialPermit(registry, trial).verify()
            row.update(state='missing_original_controller')
        else:
            permit = QuarantinedTrialPermit(registry, trial)
            controller = _controller(permit)
            steps = controller.verify()
            wallet = open_existing_authority(paths['authority']).verify()
            claims, search = _claims(permit), _search(permit)
            consumed = [s['attempt']['index'] for s in steps if not controller.step_completed(s)]
            complete = (len(steps) == len(controller.schedule.due_at) and not consumed
                        and all(c['state'] == 'complete' for c in claims.values()))
            row.update(state='complete_execution' if complete else 'incomplete_or_consumed',
                controller_digest=fingerprint(controller.identity), steps=steps,
                wallet=wallet, claims=claims, search=search, consumed_steps=consumed)
        rows.append(row)
    value = {'version': VERSION, 'validation_trial': validation_trial,
        'registered_paths': registry.plan.oracle_paths, 'paths': rows,
        'complete_execution_paths': sum(r['state'] == 'complete_execution' for r in rows),
        'scope': 'whole_original_policy_execution_only',
        'full_adaptive_risk_truth': None, 'formal_gate_eligible': False,
        'pending_truth_dependencies': [
            'Source-bound probability law for actual provider responses, not only requested model/reasoning',
            'Full-policy risk estimator covering adaptive search, waiting, activation and five claims',
            'A full-support rigorous risk enclosure under the joint market/provider law']}
    return {**value, 'evidence_digest': fingerprint(value)}
