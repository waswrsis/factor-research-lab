"""Discover original completed gate producers for a frozen forward neural job.

No caller evidence map, hand-written success flag or extra work grant. Shared S
resource evidence remains S-only: serving still requires the separately verified
original full-forward resource workload and its publication/capacity checks.
"""
from __future__ import annotations

import json

from factorlab.protocol import fingerprint
from .calibration_trial_context import _no_redirect
from .full_calibration_registry import _read
from .monthly_work_authority import MonthlyWorkAuthority, source_files
from .neural_forward import FrozenNeuralRecipe
from .v7_forward import ForwardRuntime
from .v7_lifecycle import stamp

VERSION = 'v7-canonical-dynamic-forward-gates-v1'


class ForwardGatesPending(ValueError):
    """Original producer has not supplied the required complete workload proof."""


def resolve_forward_gates(runtime, authority, job, recipe, *, shared_only=False):
    if (type(runtime) is not ForwardRuntime or type(authority) is not MonthlyWorkAuthority
            or type(recipe) is not FrozenNeuralRecipe):
        raise TypeError('Actual runtime, original wallet and exact complete learner required')
    authority.verify()
    if authority.manifest['source_files'] != runtime.identity['source_files'] or source_files() != runtime.identity['source_files']:
        raise ValueError('Dynamic neural gates cannot cross source identities')
    from .neural_forward_ticket import _verify_authority_binding
    from .calibration_runtime_contract import _original_operations
    _verify_authority_binding(runtime.lifecycle, authority, _original_operations(runtime.lifecycle))
    row = runtime.lifecycle.db.execute('SELECT cutoff,stream,pipeline FROM jobs WHERE id=?', (job,)).fetchone()
    if row is None or json.loads(row[2]) != runtime.recipe_for(recipe):
        raise ValueError('Dynamic neural gate resolution requires the actual frozen recipe job')
    cutoff, stream = stamp(row[0]), row[1]
    from .neural_forward_ticket import ForwardFitTicket, verify_prepaid_ticket
    original_path = authority.directory/'forward_tickets'/fingerprint(
        {'plan': runtime.lifecycle.plan.digest, 'job': job})/'ticket.json'
    if original_path.is_file() and not shared_only:
        ticket = ForwardFitTicket.load(original_path)
        if (ticket.value['recipe'] != json.loads(json.dumps(recipe.to_dict(), sort_keys=True))
                or ticket.value['authority_directory'] != str(authority.directory)):
            raise ValueError('Existing prepaid forward work changed its recipe or wallet')
        record = authority.get(ticket.value['reservation_id'])
        if record['status'] == 'completed':
            ticket.verify_completed_binding()
        else:
            verify_prepaid_ticket(ticket, authority, record)
        return {'version': VERSION, 'job': job, 'pipeline_id': recipe.pipeline_id,
            'source_hash': fingerprint(source_files()), 'authority_manifest_hash': authority.manifest_hash,
            'gates': ticket.value['gates'], 'original_ticket_digest': ticket.digest,
            'authorization': 'original_ticket_verifier_still_required'}
    if stream == 'A0':
        purpose = 'A0'
    elif stream == 'A0_policy_trial':
        purpose = 'policy_trial'
    else:
        batch, separator, _ = stream.rpartition(':')
        if not separator:
            raise ValueError('Unknown neural forward stream')
        _, status, _ = runtime.lifecycle._batch(batch)
        purpose = 'V' if status == 'frozen_V' else 'C'
    # Authorize actual confirmed adoption or the actual fixed V/C role before
    # reading/choosing producer records. Typed algorithm shadows remain owned by
    # their dedicated controller and cannot borrow ordinary factor authority.
    from .neural_forward_contract import assert_frozen_forward_fit
    context = assert_frozen_forward_fit(runtime.lifecycle, job, recipe, purpose=purpose)['binding']
    from .quarantined_observation import job_binding
    prefix = job_binding(runtime.lifecycle, job)
    physical_cutoff = cutoff
    if prefix is not None:
        from .nonlinear_runtime import physical_bound_for_prefix
        physical_cutoff = physical_bound_for_prefix(runtime.lifecycle, prefix)
    expected_recipe = json.loads(json.dumps(recipe.to_dict(), sort_keys=True))
    from .neural_research_launcher import verify_completed_research
    candidates = []
    for path in sorted((authority.directory/'research_tickets').glob('*/ticket.json')):
        _no_redirect(path, authority.directory)
        value = _read(path)['value']
        if value['recipe'] != expected_recipe:
            continue
        if (value['authority_directory'] != str(authority.directory)
                or value['source_hash'] != fingerprint(source_files())
                or value['context']['source_id'] != context['source_id']):
            raise ValueError('Original same-recipe S producer has a foreign source or wallet')
        reservation = authority.get(value['reservation_id'])
        if reservation['status'] != 'completed':
            continue
        report = verify_completed_research(path)
        published = stamp(report['publication']['observation']['wall'])
        if published > physical_cutoff:
            continue
        gates = report['ticket'].value['gates']
        if set(gates) != {'epoch', 'probes', 'resource'}:
            raise ValueError('Unknown original shared S gate producer schema')
        candidates.append((published.isoformat(), report['ticket'].digest,
                           {'epoch': gates['epoch'], 'probes': gates['probes']}))
    if not candidates:
        raise ForwardGatesPending('waiting_original_completed_same_recipe_shared_S_gates')
    # Latest available original S proof, deterministic digest tie-break. This
    # selects provenance, never a more favorable model or post-hoc gate statistic.
    candidates.sort()
    _, research_digest, shared = candidates[-1]
    if shared_only:
        return {'purpose': purpose, 'research_ticket_digest': research_digest,
                'gates': {'shared_S': shared, 'resource_certificate': None}}
    from .neural_forward_admission import verify_completed_artifact
    resources = []
    for path in sorted((authority.directory/'forward_tickets').glob('*/ticket.json')):
        _no_redirect(path, authority.directory)
        value = _read(path)['value']
        if value['recipe'] != expected_recipe or value['context']['purpose'] != 'resource_trial':
            continue
        certificate_path = path.parent/'launch'/'certificate.json'
        if not certificate_path.is_file():
            continue
        certificate = _read(_no_redirect(certificate_path, authority.directory))
        report = verify_completed_artifact(certificate, allow_resource_trial=True)
        if (report['context']['source_id'] != context['source_id']
                or report['source_hash'] != fingerprint(source_files())
                or report['recipe'].pipeline_id != recipe.pipeline_id
                or report['reservation']['authority_manifest_hash'] != authority.manifest_hash):
            raise ValueError('Full forward workload proof has foreign source/recipe/wallet')
        published = stamp(report['publication']['observation']['wall'])
        if published <= physical_cutoff:
            resources.append((published.isoformat(), fingerprint(certificate), certificate))
    if not resources:
        raise ForwardGatesPending('waiting_original_same_recipe_full_forward_resource_workload')
    resources.sort(key=lambda r: (r[0], r[1]))
    resource = resources[-1]
    gates = {'shared_S': shared, 'resource_certificate': resource[2]}
    # Target-cutoff quarterly eligibility and exact actual frame capacity are
    # reverified by the fixed prepaid ticket after its original snapshot seal.
    # Resolution does not grant training or claim those checks already passed.
    return {'version': VERSION, 'job': job, 'pipeline_id': recipe.pipeline_id,
            'context_hash': fingerprint(context), 'source_hash': fingerprint(source_files()),
            'authority_manifest_hash': authority.manifest_hash,
            'research_ticket_digest': research_digest, 'resource_certificate_digest': resource[1],
            'gates': gates, 'authorization': 'original_ticket_verifier_still_required'}
