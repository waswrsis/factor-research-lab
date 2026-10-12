"""Bounded local full-recipe evolution; plans confer no fit or serving grant."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime

import pandas as pd

from factorlab.protocol import fingerprint, project_path

from .algorithm_shortlist import AlgorithmShortlistRule, _score
from .neural_forward import FrozenNeuralRecipe
from .neural_research_launcher import verify_completed_research
from .v7_storage import immutable_record
from .work_authority_bridge import WorkAuthorityBridge

VERSION = 'v7-local-structure-evolution-v3-paid-gpt-recommendation'
POLICY = {'version': VERSION, 'widths': [4, 8, 16, 32], 'dimensions': [2, 4, 8],
          'outputs': ['residual', 'encoder_ridge'], 'experts': 1, 'seeds': [17, 29, 43],
          'epochs': 4, 'folds': 3, 'max_recipes': 24, 'max_months': 120,
          'selection': 'continue_pending_else_feedback_ordered_unseen_local_neighbor',
          'cross_month_loss_ranking': False, 'gpt_structure_proposals': 'original_paid_S_family2_recommendation_only',
          'automatic_adoption': False, 'formal_claim': False,
          'review': 'nonquarter_only_original_300s_prepaid_1solve',
          'quarter_priority': 'fixed_probe_review_deferred_next_nonquarter'}


def _copy(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _read(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2**20:
        raise ValueError('Missing/bounded structural history anchor')
    value = json.loads(path.read_text('utf-8'))
    if set(value) != {'body', 'hash'} or fingerprint(value['body']) != value['hash']:
        raise ValueError('Structural history checksum changed')
    return value


def local_mutations(recipe, *, eligible):
    """One coordinate changes per edge; deterministic ordering is frozen policy."""
    if type(recipe) is not FrozenNeuralRecipe or type(eligible) is not bool:
        raise TypeError('Actual full recipe and verified feedback class required')
    s = recipe.config.structures[0]
    variants = []
    output = 'encoder_ridge' if recipe.output == 'residual' else 'residual'
    swap = (replace(recipe, output=output), 'switch_output')
    if not eligible:
        variants.append(swap)
    for field, values in (('width', POLICY['widths']), ('representation_dim', POLICY['dimensions'])):
        current = getattr(s, field)
        # Immediate registered neighbors only, positive feedback grows first.
        index = values.index(current)
        for offset in ((1, -1) if eligible else (-1, 1)):
            if 0 <= index + offset < len(values):
                value = values[index + offset]
                valid = value >= s.representation_dim if field == 'width' else value <= s.width
                changed = replace(s, **{field: value}) if valid else None
                if changed is not None and changed.representation_dim <= changed.width:
                    config = replace(recipe.config, structures=(changed,))
                    variants.append((replace(recipe, config=config), field))
    if eligible:
        variants.append(swap)
    return tuple(variants)


def _paths(bridge, month):
    suffix = bridge.identifier if month == bridge.month else None
    if suffix is None:
        state = bridge.store.get(month)
        ref = state.get('structural_plan') if state else None
        if ref is None:
            raise ValueError('Structural predecessor is missing its actual controller seal')
        suffix = ref['body']['bridge_id']
    return (bridge.store.directory / 'structural_evolution' / (month + '.json'),
            bridge.authority.directory / 'structural_evolution' / suffix / 'plan.json')


def assert_selected_structure(store, state, bridge_identity, recipe):
    """Worker/authority path binds paid work to the actual frozen selection."""
    from .v7_run import RunStore
    if type(store) is not RunStore or type(recipe) is not FrozenNeuralRecipe:
        raise TypeError('Actual readonly controller and full recipe required')
    value = state.get('structural_plan')
    if value is None:
        return False  # Explicit legacy/standalone diagnostics, not G03 evidence.
    body = value['body']
    primary = store.directory / 'structural_evolution' / (state['cutoff'][:7] + '.json')
    mirror = project_path(bridge_identity['authority_directory']) / 'structural_evolution' / bridge_identity['bridge_id'] / 'plan.json'
    if (_read(primary) != value or _read(mirror) != value
            or body['policy'] != POLICY or body['run_identity_hash'] != fingerprint(store.identity)
            or body['bridge_id'] != bridge_identity['bridge_id']
            or body['bridge_identity_hash'] != fingerprint(bridge_identity)
            or body['authority_manifest_hash'] != bridge_identity['authority_manifest_hash']
            or body['data_id'] != state['data_id'] or body['input_id'] != state['nonlinear_input_data_id']
            or body['cutoff'] != state['cutoff'] or body['search_exposure'] != state['search_exposure']
            or body['recipe'] != _copy(recipe.to_dict()) or body['pipeline_id'] != recipe.pipeline_id
            or body['status'] != 'selected_requires_exact_recipe_gates'):
        raise ValueError('Paid structure differs from original controller selection')
    from .prompt_request_kinds import verified_recommendation
    recommendation = verified_recommendation(store, state, bridge_identity)
    if (body.get('gpt_recommendation') != recommendation
            or (body['gpt_proposed'] and (recommendation is None
                or recipe.pipeline_id != recommendation['recipe_id']
                or body.get('review_reservation') is None))):
        raise ValueError('Structure lost its original paid GPT recommendation/review')
    if body.get('review_reservation') is not None:
        from .structural_review import verify_review_completion
        verify_review_completion(bridge_identity['authority_directory'], value)
    return True


def _history(bridge):
    bridge.store.verify()
    rows = bridge.store.db.execute('SELECT month FROM months ORDER BY month LIMIT 121').fetchall()
    if len(rows) > POLICY['max_months']:
        raise ValueError('Structural history capacity exhausted; no reset')
    history = []
    for (month,) in rows:
        state = bridge.store.get(month)
        ref = state.get('structural_plan')
        if ref is None:
            continue
        if month > bridge.month:
            raise ValueError('Structural search cannot backdate before its existing future plan')
        first, second = _paths(bridge, month)
        if _read(first) != ref or _read(second) != ref:
            raise ValueError('Structural original double anchor changed')
        body = ref['body']
        if (body['policy'] != POLICY or body['run_identity_hash'] != fingerprint(bridge.store.identity)
                or body['authority_manifest_hash'] != bridge.authority.manifest_hash
                or body['cutoff'] != state['cutoff'] or body['data_id'] != state['data_id']
                or body['input_id'] != state['nonlinear_input_data_id']
                or body['search_exposure'] != state['search_exposure']
                or body['bridge_identity_hash'] != state['authority_bridge']['identity_hash']):
            raise ValueError('Structural history lost its original S identity')
        bridge.store.exposure.assert_prompt_search(**{k: v for k, v in body['search_exposure'].items() if k != 'data_id'})
        expected = history[-1][1]['hash'] if history else None
        if body['previous_hash'] != expected:
            raise ValueError('Structural predecessor chain changed')
        from .prompt_request_kinds import verified_recommendation
        from .work_authority_bridge import _read as read_bridge
        original_identity = read_bridge(bridge.store.directory/'work_authority'/month/'identity.json')
        recommendation = verified_recommendation(bridge.store, state, original_identity)
        if (body.get('gpt_recommendation') != recommendation
                or (body['gpt_proposed'] and (recommendation is None
                    or body['pipeline_id'] != recommendation['recipe_id']
                    or body.get('review_reservation') is None))):
            raise ValueError('Historical structure lost its original paid recommendation')
        if body['recipe'] is not None:
            recipe = FrozenNeuralRecipe.from_dict(body['recipe'])
            if recipe.pipeline_id != body['pipeline_id']:
                raise ValueError('Structural recipe identity changed')
        history.append((state, ref))
    if len(history) > POLICY['max_months']:
        raise ValueError('Structural history capacity exhausted; no reset')
    return history


def _feedback_physical_cutoff(bridge, cutoff):
    """Current paid S input fixes visibility; never borrow a later wall clock."""
    from .s_observation_source import preparation_context, physical_cutoff
    if pd.Timestamp(cutoff) != pd.Timestamp(bridge.state['cutoff']):
        raise ValueError('Structural feedback changed its current original S cutoff')
    context = {'cutoff': bridge.state['cutoff'], 'run_directory': str(bridge.store.directory),
               'replay_month': bridge.month, 'source_id': bridge.state['search_exposure']['source']}
    context.update(preparation_context(bridge.store, bridge.month, bridge.identity))
    return physical_cutoff(context)


def _feedback(bridge, state, plan, cutoff):
    """Read genuine authority and producer proof, never an arbitrary score JSON."""
    body = plan['body']
    visible_through = _feedback_physical_cutoff(bridge, cutoff)
    with bridge.authority._transaction() as db:
        rows = list(bridge.authority._verify(db)['reservations'].values())
    relevant = [r for r in rows if r['binding'].get('pipeline_id') == body['pipeline_id']
                and (r['binding'].get('research_bridge_id') == body['bridge_id']
                     or r['binding'].get('nonlinear_bridge_id') == body['bridge_id'])]
    terminal = [r for r in relevant if r['status'] in {'failed_consumed', 'interrupted_consumed'}]
    if terminal and not all(_visible_failure(bridge, row, visible_through) for row in terminal):
        return {'terminal': False, 'eligible': False, 'reason': 'failure_not_yet_visible', 'blocked': True}
    if terminal:
        return {'terminal': True, 'eligible': False, 'reason': 'original_work_failed_consumed',
                'reservation_ids': sorted(r['reservation_id'] for r in terminal)}
    complete = [r for r in relevant if r['work_key'].startswith('neural_S:')]
    if len(complete) > 1:
        raise ValueError('Duplicate original full-learner attempts')
    if complete:
        row = complete[0]
        if row['status'] != 'completed':
            return {'terminal': False, 'eligible': False, 'reason': 'original_work_not_closed', 'blocked': True}
        path = bridge.authority.directory / 'research_tickets' / row['work_key'].split(':', 1)[1] / 'ticket.json'
        report = verify_completed_research(path)
        if (report['reservation'] != row or report['ticket'].value['recipe'] != body['recipe']
                or report['ticket'].value['context']['controller_data_id'] != body['data_id']):
            raise ValueError('Structural feedback belongs to another learner or original context')
        if pd.Timestamp(report['publication']['observation']['wall']) > visible_through:
            return {'terminal': False, 'eligible': False, 'reason': 'publication_not_yet_visible', 'blocked': True}
        summary = _score(report['result'], FrozenNeuralRecipe.from_dict(body['recipe']), AlgorithmShortlistRule())
        return {'terminal': True, 'eligible': summary['eligible'], 'reason': 'verified_full_S',
                'reservation_ids': [row['reservation_id']], 'evidence_hash': fingerprint(report['result'])}
    if any(r['status'] in {'reserved', 'running'} for r in relevant):
        return {'terminal': False, 'eligible': False, 'reason': 'original_work_not_closed', 'blocked': True}
    return {'terminal': False, 'eligible': False, 'reason': 'awaiting_independent_recipe_gates'}


def _visible_failure(bridge, row, cutoff):
    """Failures lack a close timestamp: first observation is prospective only.

    Neither the lease start nor its deadline is presented as publication time.
    Both immutable anchors and their actual filesystem creation/write times
    must precede the target cutoff. This is local trusted-producer evidence,
    not administrator-proof authentication or a statistical success receipt.
    """
    key = fingerprint({'run': fingerprint(bridge.store.identity), 'reservation': row['reservation_id']})
    paths = (bridge.store.directory / 'structural_feedback' / (key + '.json'),
             bridge.authority.directory / 'structural_feedback' / (key + '.json'))
    static = {'version': VERSION, 'run_identity_hash': fingerprint(bridge.store.identity),
              'authority_manifest_hash': bridge.authority.manifest_hash,
              'reservation_id': row['reservation_id'], 'reservation_hash': fingerprint(row)}
    if any(path.exists() for path in paths):
        value = _read(paths[0])
        if _read(paths[1]) != value or any(value['body'].get(k) != v for k, v in static.items()):
            raise ValueError('Original failed-work observation changed')
    else:
        body = {**static, 'observed_at': datetime.now(UTC).isoformat()}
        value = {'body': body, 'hash': fingerprint(body)}
        for path in paths:
            immutable_record(path, value)
    end = pd.Timestamp(cutoff)
    return (pd.Timestamp(value['body']['observed_at']) <= end
            and all(max(path.stat().st_mtime_ns, path.stat().st_ctime_ns) <= end.value for path in paths))


def _select_structure(bridge, base, *, review=None, quarterly=False):
    """Freeze one complete recipe before existing paid, isolated producers run.

    Reopening returns the same plan only; it does not reissue any paid work.
    A new Pv has its own recipe identities and cannot inherit another epoch.
    """
    if type(bridge) is not WorkAuthorityBridge or type(base) is not FrozenNeuralRecipe:
        raise TypeError('Actual S bridge and complete current-Pv base required')
    bridge._guard()
    if (list(base.config.seeds) != POLICY['seeds'] or base.config.epochs != POLICY['epochs']
            or base.pool.to_dict() != bridge.state['frozen_reference_pool']
            or base.rolling_spec.digest != bridge.store.identity['rolling_hash']):
        raise ValueError('Structural search preserves the full cold learner and actual Pv')
    history = _history(bridge)
    from .prompt_request_kinds import verified_recommendation
    recommendation = verified_recommendation(bridge.store, bridge.state, bridge.identity)
    gpt_selected = False
    if history and history[-1][1]['body']['cutoff'] == bridge.state['cutoff']:
        return history[-1][1]
    if len(history) >= POLICY['max_months']:
        raise ValueError('Structural history capacity exhausted; no reset')
    compatible = [(state, ref) for state, ref in history if ref['body']['recipe'] is not None
                  and ref['body']['recipe']['pool'] == _copy(base.pool.to_dict())]
    seen = {ref['body']['pipeline_id'] for _, ref in compatible}
    feedback = None
    parent, mutation, selected = None, 'seed', base
    if history and not compatible:
        ancestors = [ref['body']['pipeline_id'] for _, ref in history if ref['body']['pipeline_id'] is not None]
        parent = ancestors[-1] if ancestors else None
        mutation = 'rebase_reference_pool_resets_all_gates'
    if compatible:
        _prior_state, prior = compatible[-1]
        previous = FrozenNeuralRecipe.from_dict(prior['body']['recipe'])
        if quarterly:
            feedback = {'terminal': False, 'eligible': False, 'reason': 'quarter_probe_priority_review_deferred'}
        else:
            if review is None:
                raise ValueError('Historical numerical feedback requires original prepaid review')
            review.check_time()
            feedback = {'terminal': False, 'eligible': False, 'reason': 'awaiting_independent_recipe_gates'}
            # A quarter carry cannot erase the earlier original completed S job.
            for old_state, old_ref in reversed(compatible):
                if old_ref['body']['pipeline_id'] != previous.pipeline_id:
                    continue
                feedback = _feedback(bridge, old_state, old_ref, bridge.state['cutoff'])
                review.check_time()
                if feedback['terminal'] or feedback.get('blocked'):
                    break
        if feedback.get('blocked'):
            selected, mutation, parent = None, 'wait_original_work_evidence', previous.pipeline_id
        elif not feedback['terminal']:
            selected, mutation = previous, 'continue_independent_recipe_gates'
            parent = prior['body']['parent_pipeline_id']
        else:
            selected = None
            # Latest parent first, then bounded earlier ancestors; no ranking of
            # incompatible monthly losses. Exact duplicate programs never refit.
            parents = [previous, *(FrozenNeuralRecipe.from_dict(ref['body']['recipe'])
                       for _, ref in reversed(compatible[:-1]))]
            for candidate_parent in parents:
                choices = [(r, edge) for r, edge in local_mutations(candidate_parent, eligible=feedback['eligible'])
                           if r.pipeline_id not in seen]
                if choices:
                    selected, mutation = choices[0]
                    if recommendation is not None:
                        for choice, edge in choices:
                            if (choice.pipeline_id == recommendation['recipe_id']
                                    and _copy(choice.to_dict()) == recommendation['recipe']):
                                selected, mutation, gpt_selected = choice, edge, True
                                break
                    parent = candidate_parent.pipeline_id
                    break
    if len(seen) >= POLICY['max_recipes'] and selected is not None and selected.pipeline_id not in seen:
        selected = None
    body = {'version': VERSION, 'policy': _copy(POLICY), 'cutoff': bridge.state['cutoff'],
            'run_identity_hash': fingerprint(bridge.store.identity), 'bridge_id': bridge.identifier,
            'bridge_identity_hash': fingerprint(bridge.identity), 'authority_manifest_hash': bridge.authority.manifest_hash,
            'data_id': bridge.state['data_id'], 'input_id': bridge.state['nonlinear_input_data_id'],
            'search_exposure': _copy(bridge.state['search_exposure']),
            'previous_hash': history[-1][1]['hash'] if history else None,
            'recipe': selected.to_dict() if selected else None,
            'pipeline_id': selected.pipeline_id if selected else None,
            'parent_pipeline_id': parent, 'mutation': mutation, 'feedback': feedback,
            'status': ('selected_requires_exact_recipe_gates' if selected else
                       'deferred_original_evidence' if feedback and feedback.get('blocked') else
                       'deferred_finite_grammar_exhausted'),
            'serving_eligible': False, 'formal_claim': False, 'gpt_proposed': gpt_selected,
            'gpt_recommendation': recommendation,
            'review_reservation': review.record['reservation_id'] if review else None,
            'review_status': 'prepaid_review_pending_commit' if review else 'deferred_quarter_probe_priority'}
    value = {'body': _copy(body), 'hash': fingerprint(body)}
    first, second = _paths(bridge, bridge.month)
    # Partial publication never silently repairs a lost dual anchor.
    if first.exists() or second.exists():
        raise ValueError('Partial structural preparation is consumed/readback only')
    immutable_record(first, value)
    immutable_record(second, value)
    bridge.state['structural_plan'] = value
    bridge.store.save(bridge.month, bridge.state, 'frozen_local_structural_candidate')
    bridge._guard()
    return value


def select_structure(bridge, base):
    """Only the complete producer may expose numerical historical feedback."""
    return produce_structure_review(bridge, base)


def produce_structure_review(bridge, base):
    """Stable hard-guard hook: return only after numerical and durable closure.

    Quarter months retain the fixed probe. No numerical history is evaluated;
    mutation is explicitly deferred. Failures propagate and retain their debit.
    """
    from .structural_review import freeze_review_intent, publish_review_completion, verify_review_completion
    if type(bridge) is not WorkAuthorityBridge or type(base) is not FrozenNeuralRecipe:
        raise TypeError('Actual bridge and complete recipe required')
    bridge._guard()
    if (list(base.config.seeds) != POLICY['seeds'] or base.config.epochs != POLICY['epochs']
            or base.pool.to_dict() != bridge.state['frozen_reference_pool']
            or base.rolling_spec.digest != bridge.store.identity['rolling_hash']):
        raise ValueError('Structural search preserves the full cold learner and actual Pv')
    if bridge.state.get('structural_plan') is not None:
        value = _select_structure(bridge, base, quarterly=True)
        if value['body']['review_reservation'] is not None:
            verify_review_completion(bridge.authority, value)
        return value
    if (int(bridge.month[5:7])-1) % 3 == 0:
        return _select_structure(bridge, base, quarterly=True)
    intent = freeze_review_intent(bridge, base)
    work = bridge.begin_work('structure_review', counts={'solve': 1}, lease_seconds=300,
        binding={'kind': 'structure_review', 'intent_hash': intent['hash']})
    try:
        work.consume('solve')  # One conservative charged numerical review unit, zero model fits.
        value = _select_structure(bridge, base, review=work)
        work.check_time()
        completed = work.complete({'kind': 'structure_review', 'structural_plan_hash': value['hash'],
                       'intent_hash': intent['hash'], 'training_performed': False})
        publish_review_completion(bridge, value, completed)
        return value
    except BaseException as error:
        try:
            row = bridge.authority.get(work.record['reservation_id'])
            if row['status'] in {'reserved', 'running'}:
                work.fail(type(error).__name__ + ': structure review failed')
        except BaseException as persistence_error:  # noqa: BLE001 - retain the primary interrupted-work failure
            error.add_note('Review closure could not be persisted: '+repr(persistence_error))
        raise
