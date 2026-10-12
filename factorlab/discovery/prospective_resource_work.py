"""Pre-V complete resource trial from an original selected typed S learner.

No future data, model serving permission or extra wallet allocation is created.
Successful publication becomes usable only at a later original job prefix.
"""
import json
from pathlib import Path

from filelock import FileLock

from factorlab.protocol import fingerprint
from .v7_lifecycle import canonical, stamp
from .v7_storage import immutable_record
from .forward_resource_work import PREFIX, VERSION, _read, verify_resource_plan


def verify_prospective(lifecycle, value, recipe):
    ref = value['prospective']
    identity = _read(Path(ref['directory'])/'identity.json')
    source = _read(lifecycle.directory/'forward.json')['identity']
    if (fingerprint(identity) != ref['identity_hash']
            or _read(Path(identity['anchor'])) != identity
            or identity['plan_id'] != lifecycle.plan.digest
            or identity['runtime_identity_hash'] != fingerprint(source)
            or identity['source_hash'] != fingerprint(source['source_files'])
            or identity['authority_manifest_hash'] != value['authority_manifest_hash']
            or identity['recipes'].get(recipe.pipeline_id) != json.loads(canonical(recipe.to_dict()))
            or value['target_context']['cutoff'] >= identity['start']):
        raise ValueError('Resource preflight lost its original prospective algorithm identity')
    records = [json.loads(r[0]) for r in lifecycle.db.execute('SELECT record FROM operations ORDER BY seq')]
    bindings = [r['detail'] for r in records if r['kind'] == 'algorithm_validation_bound'
                and r['detail']['catalogue_id'] == identity['catalogue_id']]
    if bindings != [{'catalogue_id': identity['catalogue_id'], 'directory': ref['directory'],
                     'identity_hash': ref['identity_hash']}]:
        raise ValueError('Resource preflight is not an original registered algorithm')
    return identity


def pending_prospective_resource(protocol):
    from .algorithm_validation import AlgorithmValidation, _actual_shortlist, observation_now
    from .neural_forward import FrozenNeuralRecipe
    from .neural_research_launcher import verify_completed_research
    from .quarantined_observation import current_binding
    if type(protocol) is not AlgorithmValidation:
        raise TypeError('Original typed validation required for resource preflight')
    protocol.verify()
    runtime, authority = protocol.runtime, protocol.authority
    now = observation_now(runtime)
    if now >= stamp(protocol.identity['start']):
        return []
    cutoff = now.normalize().replace(day=1).isoformat()
    prefix = current_binding(runtime.lifecycle)
    physical = now
    if prefix is not None:
        from .nonlinear_runtime import physical_bound_for_prefix
        physical = physical_bound_for_prefix(runtime.lifecycle, prefix)
    selected = {v['pipeline_id']: v for v in _actual_shortlist(protocol.catalogue)['algorithm_shortlist']}
    plans = []
    for pid in sorted(protocol.identity['recipes']):
        recipe = FrozenNeuralRecipe.from_dict(protocol.identity['recipes'][pid])
        job = fingerprint({'run': runtime.lifecycle.plan.digest, 'cutoff': cutoff, 'stream': PREFIX+pid})
        path = runtime.directory/'resource_work'/job/'plan.json'
        if path.exists():
            plan = verify_resource_plan(runtime.lifecycle, job, recipe)
            plans.append({'job': job, 'pipeline_id': pid})
            continue
        report = verify_completed_research(selected[pid]['S_evidence']['path'])
        ticket = report['ticket']
        gates = ticket.value['gates']
        if set(gates) != {'epoch', 'probes', 'resource'}:
            continue
        if (ticket.digest != selected[pid]['S_evidence']['digest']
                or ticket.value['authority_manifest_hash'] != authority.manifest_hash
                or ticket.value['recipe'] != protocol.identity['recipes'][pid]
                or ticket.value['context']['source_id'] != runtime.lifecycle.plan.source_id):
            raise ValueError('Preflight recipe is not its genuine selected S learner')
        if stamp(report['publication']['observation']['wall']) > physical:
            continue
        target = {'purpose': 'typed_prospective', 'job': None, 'cutoff': cutoff,
                  'pipeline': runtime.recipe_for(recipe)}
        value = {'version': VERSION, 'plan_id': runtime.lifecycle.plan.digest, 'job': job,
                 'target_context': target, 'authority_directory': str(authority.directory),
                 'authority_manifest_hash': authority.manifest_hash,
                 'gates': {'shared_S': {k: gates[k] for k in ('epoch', 'probes')}, 'resource_certificate': None},
                 'research_ticket_digest': ticket.digest,
                 'prospective': {'directory': str(protocol.directory), 'identity_hash': protocol.digest}}
        lock = runtime.directory/'resource_work'/('.'+pid+'.lock')
        lock.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(lock), timeout=0):
            immutable_record(path, value)
            with runtime.lifecycle.db:
                runtime.lifecycle._event('forward_resource_planned', value)
            actual = runtime.lifecycle.plan_fit(cutoff, PREFIX+pid, target['pipeline'])
            if actual != job:
                raise ValueError('Prospective resource work identity changed')
            verify_resource_plan(runtime.lifecycle, job, recipe)
        plans.append({'job': job, 'pipeline_id': pid})
    return plans


def run_prospective_resource(protocol, job):
    from .neural_forward import FrozenNeuralRecipe
    from .forward_resource_work import _execute_plan
    plans = pending_prospective_resource(protocol)
    matches = [p for p in plans if p['job'] == job]
    if len(matches) != 1:
        raise ValueError('Unknown or expired original prospective resource work')
    recipe = FrozenNeuralRecipe.from_dict(protocol.identity['recipes'][matches[0]['pipeline_id']])
    plan = verify_resource_plan(protocol.runtime.lifecycle, job, recipe)
    lock = protocol.runtime.directory/'resource_work'/('.'+recipe.pipeline_id+'.lock')
    with FileLock(str(lock), timeout=0):
        return _execute_plan(protocol.runtime, protocol.authority, recipe, plan)
