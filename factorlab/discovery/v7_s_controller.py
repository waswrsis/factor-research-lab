"""Actual numerical/GPT stage functions; the run_month loop integrates explicitly.

Scientific allocation and nonlinear/forward reserves require a new manifest.
No entry here launches itself, publishes A0, or accepts provider-selected code.
"""
from dataclasses import replace

from factorlab.discovery.convex_eval import fit_ridge
from factorlab.discovery.joint_pool import (
    FrozenPool,
    backward_prune,
    evaluate_candidate,
    hour_equal_weights,
)
from factorlab.discovery.v7_agent import CodexBatchBackend, propose_batch
from factorlab.discovery.v7_run import _baseline_readout
from factorlab.protocol import fingerprint


def pool_solve_bound(pool):
    """Same recursive bound as FrozenNeuralRecipe; profiling is not shortened."""
    prefix = len(pool.reference_pool.members) if pool.reference_pool else 0
    own = 1 + (pool.max_profile_calls if any(
        m.parameter_mode == 'profile' and m.parameters for m in pool.members[prefix:]) else 0)
    return own + (pool_solve_bound(pool.reference_pool) if pool.reference_pool else 0)


def candidate_solve_bound(pool, *, parameterized):
    # Per fold: K0 + Pv + F_K0 + F_Pv + theta0; provided cached references
    # avoid recursive reference refits for the three conditional fits.
    return 3 * (pool_solve_bound(pool) + 4 +
                (1 + pool.max_profile_calls if parameterized else 0))


def pruning_solve_bound(pool):
    """One removal step, existing flat joint-refit control and all ablations."""
    def flat(members):
        return 1 + (pool.max_profile_calls if any(
            m.parameter_mode == 'profile' and m.parameters for m in members) else 0)
    if not pool.members:
        return 0
    return 3 * (flat(pool.members) + sum(flat(pool.members[:i] + pool.members[i+1:])
                                        for i in range(len(pool.members))))


def standard_slots(*, candidate_caps, candidate_seconds, pruning_cap,
                   pruning_seconds, forward_solve_holdback, nonlinear_solve_holdback=0):
    """Explicit allocation identity; no implicit S claim on all 1600 solves.

    The 24 positions are opportunities, not guaranteed successful fits. Each
    cap is fixed before proposal outcomes. Unused prepaid allowance never moves.
    This is the STANDARD-policy builder; controls_third needs its own identity.
    """
    if (len(candidate_caps) != 24 or len(candidate_seconds) != 24
            or any(type(v) is not int or v < 1 for v in candidate_caps)
            or type(pruning_cap) is not int or pruning_cap < 1
            or type(forward_solve_holdback) is not int or forward_solve_holdback < 1
            or type(nonlinear_solve_holdback) is not int or nonlinear_solve_holdback < 0):
        raise ValueError('Explicit 24 candidate caps and positive forward reserve required')
    total = 7 + sum(candidate_caps) + pruning_cap
    if total + forward_solve_holdback + nonlinear_solve_holdback > 1600:
        raise ValueError('S plus forward/nonlinear allocations exceed the original shared ceiling')
    slots = [{'key': 'baseline', 'stage': 'baseline', 'counts': {'solve': 7}, 'seconds': 300}]
    for family in range(3):
        slots.append({'key': f'family:{family}', 'stage': 'family',
                      'counts': {'family': 1, 'gpt_attempt': 2, 'ast': 32}, 'seconds': 300})
        for position in range(8):
            i = family * 8 + position
            slots.append({'key': f'candidate:{family}:{position}', 'stage': 'candidate',
                          'counts': {'f': 1, 'solve': candidate_caps[i]}, 'seconds': candidate_seconds[i]})
    slots.append({'key': 'pruning', 'stage': 'pruning', 'counts': {'solve': pruning_cap},
                  'seconds': pruning_seconds})
    return slots, {'version': 'v7-S-forward-allocation-v1', 'S_solve': total,
                   'forward_solve_holdback': forward_solve_holdback,
                   'nonlinear_solve_holdback': nonlinear_solve_holdback,
                   'unused_credit_transfer': False, 'shared_ceiling': 1600,
                   'baseline_solve': 7, 'per_work_seconds_max': 300,
                   'S_seconds_sum': sum(slot['seconds'] for slot in slots)}


def baseline_stage(data, cutoff, budget):
    """Unchanged actual three-fold HAR/K0 + final K0 cold fit: seven solves."""
    readout = _baseline_readout(data, cutoff, budget)
    mask = data.train_mask(cutoff)
    budget.consume('solve')
    model = fit_ridge(data.frame.take(mask), data.target[mask], ridge=data.spec.ridge,
        sample_weight=hour_equal_weights(data.frame.metadata.loc[mask]),
        provenance=data.training_identity(cutoff))
    budget.check_time()
    return {'readout': readout, 'model': model.to_dict(), 'model_digest': model.digest}


def candidate_stage(data, cutoff, pool, candidate, budget, *, budget_family_id, context):
    budget.consume('f', family_id=budget_family_id)
    result = evaluate_candidate(data, cutoff, pool, candidate, budget=budget, context=context)
    budget.check_time()
    return result


def pruning_stage(data, cutoff, pool, budget):
    pruned, receipt = backward_prune(data, cutoff, pool, budget=budget, max_remove=1)
    budget.check_time()
    return {'pool': pruned.to_dict(), 'receipt': receipt}


class OriginalLeaseBackend:
    """Both attempts share the family lease; no new overall provider deadline.

    Checks are cooperative until the fixed backend accepts the original absolute
    lease or is wrapped by the fixed external watchdog. This draft does not
    claim hard native-process enforcement merely from a relative timeout.
    """
    def __init__(self, work, directory):
        self.work, self.directory = work, directory

    def __call__(self, packet, *, model, reasoning):
        self.work.check_time()
        remaining = self.work.lease.assert_parent_alive().remaining_seconds
        backend = CodexBatchBackend(self.directory, timeout_seconds=remaining)
        result = backend(packet, model=model, reasoning=reasoning)
        self.work.check_time()
        return result


def one_proposal_attempt(packet, work, *, directory, audit, feature_names, allowed_parent_ids):
    """One existing strict parser attempt, inside a shared family envelope."""
    return propose_batch(packet, backend=OriginalLeaseBackend(work, directory), budget=work,
        feature_names=feature_names, allowed_parent_ids=allowed_parent_ids,
        audit=audit, persist_budget=work.persist)


def staged_numeric_month(scheduler, data, cutoff, *, pool, families, context):
    """Actual baseline/F/P/S/pruning integration on already frozen local families.

    No archive selection/promotion is implied. Root run_month retains its exact
    memory, prescreen, selection, archive and finalization code around these
    stage functions. GPT family routing remains separate and explicitly staged.
    """
    if len(families) != 3 or not isinstance(pool, FrozenPool):
        raise ValueError('Exactly three fixed local family opportunities required')
    out = {'role': 'S', 'formal_claim': False, 'automatic_adoption': False,
           'schedule_hash': scheduler.schedule.digest, 'families': [], 'candidates': []}
    out['baseline'] = scheduler.run('baseline', lambda b: baseline_stage(data, cutoff, b),
                                   input_binding={'data_id': scheduler.bridge.state['data_id']})
    for i, members in enumerate(families):
        if len(members) > 8:
            raise ValueError('Too many fixed family candidates')
        family_id = f'fixed-family:{i}'
        if not members:
            scheduler.skip(f'family:{i}', reason='no_valid_family')
        else:
            def materialize(budget, members=members, family_id=family_id):
                budget.register_family(family_id)
                for member in members:
                    budget.consume('ast', family_id=family_id)
                return {'family_id': family_id, 'members': [m.member_id for m in members]}
            out['families'].append(scheduler.run(f'family:{i}', materialize,
                input_binding={'members': [m.member_id for m in members]}))
        for j in range(8):
            key = f'candidate:{i}:{j}'
            if j >= len(members):
                scheduler.skip(key, reason='no_candidate')
                continue
            member = replace(members[j], family=family_id)
            out['candidates'].append(scheduler.run(key,
                lambda b, member=member, family_id=family_id: candidate_stage(
                    data, cutoff, pool, member, b, budget_family_id=family_id, context=context),
                input_binding={'formula': member.to_dict(), 'context': context}))
    if pool.members:
        out['pruning'] = scheduler.run('pruning', lambda b: pruning_stage(data, cutoff, pool, b),
                                      input_binding={'pool_hash': pool.digest})
    else:
        scheduler.skip('pruning', reason='no_pool_to_prune')
    out['actual_counts'] = scheduler.bridge.actual_counts
    out['result_hash'] = fingerprint(out)
    return out
