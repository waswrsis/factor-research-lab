"""Full neural recipe identities and frozen forward-fit permissions.

S diagnostics are never forward permissions. These checks read the actual
lifecycle, protected batch and confirmation reservation before any work is
charged or launched. They do not admit a model or grant statistical claims.
"""
from __future__ import annotations

import json

from factorlab.protocol import fingerprint

from .neural_forward import FrozenNeuralRecipe
from .v7_lifecycle import stamp


def neural_pipeline(recipe, rolling_spec):
    if not isinstance(recipe, FrozenNeuralRecipe):
        raise TypeError('A complete frozen neural recipe is required')
    if recipe.rolling_spec.digest != rolling_spec.digest:
        raise ValueError('Neural recipe changed the forward rolling rule')
    # Canonical wire form matters: dataclass tuples become arrays in SQLite's
    # JSON. Comparing the in-memory tuple form would reject every genuine job.
    return json.loads(json.dumps({'kind': 'v7_neural_forward', 'pipeline_id': recipe.pipeline_id,
            'recipe': recipe.to_dict(), 'task': 'volatility', 'rolling_hash': rolling_spec.digest},
            sort_keys=True, allow_nan=False))


def _confirmed_adoption_binding(lifecycle, cutoff, pipeline, *, expected=None, historical=False):
    """Read an actual concluded C authority; no active model or caller approval.

    Historical tickets keep their original selected batch even after a later
    batch concludes. New fits obey the same latest-ordinal policy as the real
    A0 scheduler. Both paths reverify the original result/reservation chain.
    """
    from .typed_policy_confirmation import canonical_adoption
    typed = canonical_adoption(lifecycle, cutoff, pipeline, expected=expected, historical=historical)
    if typed is not None:
        return typed
    lifecycle.verify()
    cutoff = stamp(cutoff)
    lifecycle.fit_cutoff(cutoff)
    if (cutoff > lifecycle.now()
            or not lifecycle.confirmation_execution_ready() or (historical and not isinstance(expected, dict))):
        raise ValueError('A neural A0 learner requires its verified adoption authority')
    eligible = []
    for batch, raw, digest, result_raw in lifecycle.db.execute(
            'SELECT id,identity,hash,result FROM batches WHERE status=? ORDER BY ordinal DESC', ('concluded',)):
        result, identity = json.loads(result_raw), json.loads(raw)
        from .quarantine_execution import verify_decision
        isolated = verify_decision(lifecycle, result)
        from .layered_research import research_retention_decision
        research = research_retention_decision(lifecycle, result)
        accepted = (result.get('supported') is True or (isolated is not None and isolated['accepted'])
                    or (research is not None and research['status'] == 'eligible_for_unconfirmed_research_retention'))
        if not accepted or stamp(result['completed_at']) > cutoff:
            continue
        if historical and batch != expected.get('batch'):
            continue
        publication = lifecycle.confirmation_publication(batch)
        publication_path = lifecycle.directory / 'claim_publications' / f'{fingerprint(batch)}.json'
        from .quarantined_observation import publication_visible
        visible = publication_visible(lifecycle, 'confirmation', batch, cutoff)
        if (stamp(publication['published_at']) > cutoff
                or (publication_path.stat().st_mtime_ns > cutoff.value if visible is None else not visible)):
            continue
        eligible.append((batch, identity, digest, result, publication))
        break
    if not eligible:
        raise ValueError('A new neural A0 learner requires its completed adoption authority')
    batch, identity, digest, result, publication = eligible[0]
    chosen = next((c for c in identity['candidates'] if c['pipeline_id'] == result['pipeline_id']), None)
    if chosen is None or {k: v for k, v in chosen.items() if k != 'complexity'} != pipeline:
        raise ValueError('Neural recipe differs from the selected adoption authority')
    found = lifecycle.exposure.db.execute(
        'SELECT identity,consumed,result FROM confirmations WHERE batch=?', (batch,)).fetchone()
    if (not found or found[1] != 1 or not found[2] or json.loads(found[2]) != result
            or json.loads(found[0])['candidate_id'] != pipeline['pipeline_id']):
        raise ValueError('Neural adoption authority lost its consumed confirmation')
    binding = {'version': 'v7-confirmed-neural-adoption-v1', 'batch': batch,
               'batch_hash': digest, 'result_hash': fingerprint(result),
               'reservation_hash': fingerprint(json.loads(found[0])),
               'completed_at': result['completed_at'], 'pipeline_id': result['pipeline_id'],
               'publication_hash': fingerprint(publication), 'published_at': publication['published_at']}
    if research is not None:
        binding['version'] = 'v7-unconfirmed-research-adoption-v1'
        binding['research_spec_hash'] = lifecycle.research_spec.digest
        binding['formal_gate_eligible'] = False
        binding['whole_agent_power_certified'] = False
    if expected is not None and expected != binding:
        raise ValueError('Neural adoption authority differs from the original ticket')
    return binding


def assert_frozen_forward_fit(lifecycle, job, recipe, *, purpose):
    """Return actual lifecycle bindings for one immutable, unfinished job.

    V/C fits obey a fixed cold-refit rule on mature past data; they cannot
    introduce a different encoder, seed, readout or formula. Resource trials
    are distinctly bound and cannot become an A0 serving model.
    """
    if purpose not in {'A0', 'V', 'C', 'resource_trial', 'policy_trial'}:
        raise ValueError('Unknown forward fit purpose')
    pipeline = neural_pipeline(recipe, recipe.rolling_spec)
    verified = lifecycle.verify()
    row = lifecycle.db.execute(
        'SELECT cutoff,stream,pipeline,status,failure FROM jobs WHERE id=?', (job,)).fetchone()
    if (row is None or row[3] not in {'planned', 'snapshot_sealed', 'features_ready', 'fitted'}
            or json.loads(row[2]) != pipeline):
        raise ValueError('Forward fit differs from the authoritative job or complete recipe')
    cutoff, stream = stamp(row[0]), row[1]
    lifecycle.fit_cutoff(cutoff)
    if recipe.rolling_spec.window_days != lifecycle.plan.training_window_days or cutoff > lifecycle.now():
        raise ValueError('Forward cutoff is not yet actually available')
    binding = {'version': 'v7-neural-forward-context-v2', 'plan_id': lifecycle.plan.digest,
               'lifecycle_directory': str(lifecycle.directory), 'job': job, 'stream': stream,
               'cutoff': cutoff.isoformat(), 'pipeline': pipeline, 'purpose': purpose,
               'source_id': lifecycle.plan.source_id, 'symbols': list(lifecycle.plan.symbols),
               'exposure_path': str(lifecycle.exposure.path)}
    from .quarantined_observation import job_binding
    prefix = job_binding(lifecycle, job)
    if prefix is not None:
        binding['observation_prefix'] = prefix
    if stream == 'A0_policy_trial':
        if purpose != 'policy_trial':
            raise ValueError('Policy trial stream cannot borrow operational A0 authority')
        from .typed_policy_confirmation import policy_context
        selected = policy_context(lifecycle, job, recipe)
        binding.update(batch=None, batch_hash=None, confirmation=None, adoption=selected, serving_eligible=True)
    elif stream.startswith('neural-resource:'):
        if purpose != 'resource_trial':
            raise ValueError('Non-serving resource work cannot borrow a serving role')
        from .forward_resource_work import verify_resource_plan
        verify_resource_plan(lifecycle, job, recipe)
        binding.update(batch=None, batch_hash=None, confirmation=None, adoption=None, serving_eligible=False)
    elif stream == 'A0':
        if purpose not in {'A0', 'resource_trial'}:
            raise ValueError('A0 cannot borrow a protected fit role')
        adoption = _confirmed_adoption_binding(lifecycle, cutoff, pipeline) if purpose == 'A0' else None
        binding.update(batch=None, batch_hash=None, confirmation=None, adoption=adoption,
                       serving_eligible=purpose != 'resource_trial')
    else:
        batch, separator, role = stream.rpartition(':')
        if not separator or not batch or role not in {'reference', recipe.pipeline_id}:
            raise ValueError('Unknown protected forward stream')
        batch_row = lifecycle.db.execute(
            'SELECT identity,hash,status,selected FROM batches WHERE id=?', (batch,)).fetchone()
        if batch_row is None:
            raise ValueError('Forward fit lacks a real frozen batch')
        identity = json.loads(batch_row[0])
        if fingerprint(identity) != batch_row[1]:
            raise ValueError('Forward frozen batch identity changed')
        expected = (identity['reference'] if role == 'reference' else next(
            (c for c in identity['candidates'] if c['pipeline_id'] == role), None))
        if expected is None or {k: v for k, v in expected.items() if k != 'complexity'} != pipeline:
            raise ValueError('Full neural recipe is not the frozen batch learner')
        confirmation = None
        if purpose == 'V':
            if (batch_row[2] != 'frozen_V'
                    or not lifecycle.fit_cycle_start(identity['start']) <= cutoff < stamp(identity['v_end'])
                    or lifecycle.now() < stamp(identity['start'])
                    or lifecycle.now() >= stamp(identity['v_end'])):
                raise ValueError('V fit is outside its frozen validation window')
        elif purpose == 'C':
            selected = json.loads(batch_row[3]) if batch_row[3] else None
            found = lifecycle.exposure.db.execute(
                'SELECT identity,consumed,result FROM confirmations WHERE batch=?', (batch,)).fetchone()
            if (batch_row[2] != 'selected_C' or not selected or not found or found[1] or found[2]
                    or role not in {'reference', selected['candidate_id']}):
                raise ValueError('C fit requires the actual selected and unconsumed confirmation')
            confirmation = json.loads(found[0])
            if (not lifecycle.fit_cycle_start(confirmation['actual_start']) <= cutoff < stamp(confirmation['end'])
                    or not stamp(confirmation['actual_start']) <= lifecycle.now() < stamp(confirmation['end'])):
                raise ValueError('C fit is outside its actual confirmation window')
        else:
            raise ValueError('A protected stream cannot borrow A0 or resource trial authority')
        binding.update(batch=batch, batch_hash=batch_row[1], confirmation=confirmation, adoption=None,
                       serving_eligible=True)
        # Protected jobs can use a cadence cutoff before a V/C boundary. The
        # cutoff therefore cannot stand in for the actual authorization time.
        # Register the first valid authorization once; later checks must retain
        # it while still obeying the current window checks above.
        binding['version'] = 'v7-neural-forward-context-v3'
        window_start = identity['start'] if purpose == 'V' else confirmation['actual_start']
        window_end = identity['v_end'] if purpose == 'V' else confirmation['end']
        authorized = _protected_authorization(lifecycle, binding, window_start, window_end)
        binding['authorized_at'] = authorized
        verified = lifecycle.verify()
    # The event head is an observation, not a stable work identity. Tickets must
    # bind this full context but re-verify the stable job/batch fields on restore.
    return {'binding': binding, 'observed_event_head': verified['event_head']}


def _protected_authorization(lifecycle, binding, start, end):
    """Original journal provenance, never a caller timestamp or reload clock."""
    from .calibration_runtime_contract import _original_operations

    def original():
        events = [o for o in _original_operations(lifecycle)
                  if o['kind'] == 'neural_forward_authorized'
                  and o['detail'].get('context', {}).get('job') == binding['job']
                  and o['detail'].get('context', {}).get('purpose') == binding['purpose']]
        if events:
            if (len(events) != 1 or events[0]['detail'] != {'context': binding}
                    or not stamp(start) <= stamp(events[0]['at']) < stamp(end)):
                raise ValueError('Protected fit lost its original authorization')
            return events[0]['at']
        return None

    existing = original()
    if existing is not None:
        return existing
    # A readonly historical reader must never create a missing authorization.
    if type(lifecycle).__name__ == '_LifecycleReader':
        raise ValueError('Protected fit lacks its original authorization')
    with lifecycle.db:
        lifecycle.db.execute('BEGIN IMMEDIATE')
        # Serialize concurrent registrations; use the transaction's own reader
        # so a second writer cannot append another first authorization.
        found = [json.loads(r[0]) for r in lifecycle.db.execute(
            "SELECT record FROM operations WHERE json_extract(record,'$.kind')=?",
            ('neural_forward_authorized',))
            if json.loads(r[0])['detail'].get('context', {}).get('job') == binding['job']
            and json.loads(r[0])['detail'].get('context', {}).get('purpose') == binding['purpose']]
        if not found:
            lifecycle._event('neural_forward_authorized', {'context': binding})
            found = [json.loads(lifecycle.db.execute(
                'SELECT record FROM operations ORDER BY seq DESC LIMIT 1').fetchone()[0])]
        if (len(found) != 1 or found[0]['detail'] != {'context': binding}
                or not stamp(start) <= stamp(found[0]['at']) < stamp(end)):
            raise ValueError('Protected fit authorization crossed its original window')
    return original()
