"""Once-only full-forward resource work derived from an original eligible job.

The resource stream never serves or replaces A0. Its ordinary neural_forward
reservation consumes the existing forward lane and physical monthly wallet.
There is no callback, alternate command, caller success flag, or extra grant.
"""
from __future__ import annotations

import json

from filelock import FileLock

from factorlab.protocol import fingerprint
from .v7_storage import immutable_record

PREFIX = 'neural-resource:'
VERSION = 'v7-original-forward-resource-plan-v1'


def _read(path):
    from .calibration_trial_context import _no_redirect
    _no_redirect(path, path.parent)
    return json.loads(path.read_text('utf-8'))


def verify_resource_plan(lifecycle, job, recipe):
    """Reconstruct the original target authority and non-serving work identity."""
    path = lifecycle.directory/'resource_work'/job/'plan.json'
    value = _read(path)
    if value.get('version') != VERSION:
        raise ValueError('Unknown resource work plan')
    row = lifecycle.db.execute('SELECT cutoff,stream,pipeline FROM jobs WHERE id=?', (job,)).fetchone()
    target = value['target_context']
    from .neural_forward_contract import neural_pipeline
    from .neural_forward_ticket import _verify_historical_context
    if (target['purpose'] not in {'A0', 'V', 'C', 'policy_trial', 'typed_prospective'} or target['job'] == job
            or value['job'] != job or value['plan_id'] != lifecycle.plan.digest
            or target['pipeline'] != neural_pipeline(recipe, recipe.rolling_spec)
            or not row or list(row) != [target['cutoff'], PREFIX+recipe.pipeline_id,
                                       json.dumps(target['pipeline'], sort_keys=True, separators=(',', ':'))]):
        raise ValueError('Resource work changed original target recipe or non-serving stream')
    if target['purpose'] == 'typed_prospective':
        from .prospective_resource_work import verify_prospective
        verify_prospective(lifecycle, value, recipe)
    else:
        _verify_historical_context(lifecycle, target, recipe)
    operations = [json.loads(r[0]) for r in lifecycle.db.execute('SELECT record FROM operations ORDER BY seq')]
    anchors = [o['detail'] for o in operations if o['kind'] == 'forward_resource_planned'
               and o['detail'].get('job') == job]
    if anchors != [value]:
        raise ValueError('Resource work lost original registration')
    from pathlib import Path
    from types import SimpleNamespace
    from .neural_forward_ticket import _verify_authority_binding
    directory = Path(value['authority_directory'])
    wrapper = _read(directory/'manifest.json')
    if wrapper['hash'] != fingerprint(wrapper['record']) or wrapper['hash'] != value['authority_manifest_hash']:
        raise ValueError('Resource work wallet changed')
    # This is the wallet's work-identity verifier too. Do not recurse into
    # authority.verify() while that verifier is reconstructing this reservation.
    authority = SimpleNamespace(directory=directory, manifest=wrapper['record'], manifest_hash=wrapper['hash'])
    _verify_authority_binding(lifecycle, authority, operations)
    return value


def _plan(runtime, authority, target_job, recipe, resolved):
    from .neural_forward_contract import assert_frozen_forward_fit
    target = assert_frozen_forward_fit(runtime.lifecycle, target_job, recipe,
                                      purpose=resolved['purpose'])['binding']
    stream = PREFIX+recipe.pipeline_id
    job = fingerprint({'run': runtime.lifecycle.plan.digest, 'cutoff': target['cutoff'], 'stream': stream})
    path = runtime.directory/'resource_work'/job/'plan.json'
    if path.exists():
        return verify_resource_plan(runtime.lifecycle, job, recipe)
    value = {'version': VERSION, 'plan_id': runtime.lifecycle.plan.digest, 'job': job,
             'target_context': target, 'authority_directory': str(authority.directory),
             'authority_manifest_hash': authority.manifest_hash,
             'gates': resolved['gates'], 'research_ticket_digest': resolved['research_ticket_digest']}
    # This immutable original choice is made before allocation. A crash after
    # either half must never silently create another resource opportunity.
    immutable_record(path, value)
    with runtime.lifecycle.db:
        runtime.lifecycle._event('forward_resource_planned', value)
    actual = runtime.lifecycle.plan_fit(target['cutoff'], stream, target['pipeline'])
    if actual != job:
        raise ValueError('Resource work identity differs from lifecycle')
    return verify_resource_plan(runtime.lifecycle, job, recipe)


def produce_resource_work(runtime, authority, target_job, recipe):
    """Run or inspect one original paid full workload; never register a model."""
    from .forward_gate_resolver import resolve_forward_gates
    resolved = resolve_forward_gates(runtime, authority, target_job, recipe, shared_only=True)
    lock = runtime.directory/'resource_work'/('.'+recipe.pipeline_id+'.lock')
    lock.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock), timeout=0):
        plan = _plan(runtime, authority, target_job, recipe, resolved)
        return _execute_plan(runtime, authority, recipe, plan)


def _execute_plan(runtime, authority, recipe, plan):
    directory = runtime.directory/'resource_work'/plan['job']
    attempt, result = directory/'attempt.json', directory/'result.json'
    if result.exists():
        value = _read(result)
        if value['plan_digest'] != fingerprint(plan):
            raise ValueError('Resource result changed original plan')
        if value['status'] == 'completed':
            from .neural_forward_admission import verify_completed_artifact
            verify_completed_artifact(value['certificate'], allow_resource_trial=True)
        return {**value, 'executed': False}
    if attempt.exists():
        if _read(attempt) != {'plan_digest': fingerprint(plan)}:
            raise ValueError('Resource attempt changed')
        value = {'plan_digest': fingerprint(plan), 'status': 'interrupted_consumed'}
        immutable_record(result, value)
        return {**value, 'executed': False}
    immutable_record(attempt, {'plan_digest': fingerprint(plan)})
    from .neural_forward_ticket import issue_forward_ticket
    from .neural_forward import fit_neural_forward
    try:
        ticket = issue_forward_ticket(runtime, authority, plan['job'], recipe,
            purpose='resource_trial', gates=plan['gates'])
        fit_neural_forward(ticket_path=ticket.path)
        certificate = _read(ticket.path.parent/'launch'/'certificate.json')
        from .neural_forward_admission import verify_completed_artifact
        verify_completed_artifact(certificate, allow_resource_trial=True)
        from .forward_service_scope import check_service_work
        check_service_work(runtime)
        value = {'plan_digest': fingerprint(plan), 'status': 'completed', 'certificate': certificate}
    except (ValueError, RuntimeError, OSError, MemoryError) as exc:
        value = {'plan_digest': fingerprint(plan), 'status': 'failed_consumed',
                 'error_type': type(exc).__name__}
    immutable_record(result, value)
    from .forward_service_scope import check_service_work
    check_service_work(runtime)
    return {**value, 'executed': True}
