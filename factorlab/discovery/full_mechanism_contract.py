"""Prospective joint-mechanism assumptions and original local provider attestations.

Assumptions are explicit modeling conditions, never measured independence or a
server attestation. Actual fixed-command execution, binary bytes, raw reported
model, original family debit and fresh context are separately reconstructed.
"""
from pathlib import Path
import json
from dataclasses import asdict
from factorlab.protocol import fingerprint, project_path
from factorlab.discovery.evidence import file_hash
from factorlab.discovery.v7_storage import immutable_record

VERSION = 'v7-prospective-joint-mechanism-v1'
ASSUMPTIONS = (
    'domain_separated_validation_draws_modelled_independent_uniform',
    'provider_randomness_modelled_independent_across_fresh_processes_and_market_draws',
    'reported_resolved_model_and_fixed_binary_modelled_stationary_conditional_kernel',
    'physical_resource_availability_modelled_stationary_independent_across_registered_trials',
)
FIXED_REPLICA_TRANSFER = 'conditional_fixed_replica_availability_modelled_equivalent_to_original_selected_claim_including_competing_load'


def _generator(directory, acceptance):
    from .finite_market_oracle import FiniteGenerator, FiniteMarketLaw, _read
    obj = FiniteGenerator.__new__(FiniteGenerator)
    obj.directory = project_path(directory)
    obj.identity = _read(obj.directory/'identity.json')
    law = dict(obj.identity['law']); law['symbols'] = tuple(law['symbols'])
    obj.law, obj.acceptance = FiniteMarketLaw(**law), acceptance
    obj.verify()
    return obj


def freeze_mechanism_contract(generators, *, provider_executable, resolved_model, assumptions, scenario_roles):
    """Build an original plan term before registration; performs no provider call.

    The caller declares modeling assumptions verbatim. No identity/success/truth
    flags are accepted. Each finite generator and executable is read here.
    """
    from .finite_market_oracle import FiniteGenerator, proof_terms
    if (type(generators) not in (list, tuple) or not generators
            or tuple(assumptions) not in (ASSUMPTIONS, ASSUMPTIONS + (FIXED_REPLICA_TRANSFER,))
            or type(resolved_model) is not str or not resolved_model or len(resolved_model)>128):
        raise ValueError('Explicit prospective joint-mechanism assumptions and exact reported model required')
    if (type(scenario_roles) is not dict or set(scenario_roles.values()) != {'null','boundary','positive','mixed'}
            or any(type(k) is not str or not k for k in scenario_roles)):
        raise ValueError('Original null, boundary, positive and mixed design strata required')
    values = {}
    for generator in generators:
        if type(generator) is not FiniteGenerator:
            raise TypeError('Original verified finite generators required')
        generator.verify()
        if generator.law.digest in values:
            raise ValueError('Duplicate original finite law')
        values[generator.law.digest] = {'directory': str(generator.directory), 'digest': generator.digest}
    executable = Path(provider_executable).resolve(strict=True)
    if not executable.is_file():
        raise ValueError('Actual restricted executable file required')
    return {'version': VERSION, 'assumptions': list(assumptions),
        'assumption_status': 'declared_modelling_conditions_not_empirically_proved',
        'scenario_role_scope': 'preregistered_design_labels_effect_truth_recomputed_from_original_enclosures',
        'generators': values, 'scenario_roles': dict(scenario_roles), 'provider_executable': str(executable),
        'provider_executable_sha256': file_hash(executable), 'reported_resolved_model': resolved_model,
        'identity_scope': 'original_local_exec_and_raw_model_report_not_server_authentication',
        'risk_scope': 'original_selected_recipes_and_entire_preregistered_policy',
        'support_rule': 'all_original_paths_or_unknown_no_failed_path_dropping',
        'loss_proof': proof_terms(), 'implementation_sha256': file_hash(__file__)}


def verify_mechanism_contract(contract, acceptance):
    if type(contract) is not dict:
        raise ValueError('No prospectively frozen joint mechanism contract')
    generators = [_generator(v['directory'], acceptance) for v in contract.get('generators', {}).values()]
    expected = freeze_mechanism_contract(generators,
        provider_executable=contract.get('provider_executable'),
        resolved_model=contract.get('reported_resolved_model'), assumptions=contract.get('assumptions', ()), scenario_roles=contract.get('scenario_roles'))
    if set(contract['scenario_roles']) != {c.name for c in acceptance.cells}:
        raise ValueError('Mechanism strata differ from original fixed acceptance cells')
    if contract != expected or set(contract['generators']) != {c.scenario_digest for c in acceptance.cells}:
        raise ValueError('Original joint contract, source, finite law or executable changed')
    return {g.law.digest: g for g in generators}


def _begin_provider(owner, budget, command, text, runtime):
    """Private dispatch hook, called only inside the actual reserved backend."""
    from .v7_agent import CodexBatchBackend
    from .work_authority_bridge import AuthorityWorkBudget
    from .full_trial_controller import RegisteredTrialCodexBackend
    if type(owner) is not RegisteredTrialCodexBackend or type(budget) is not AuthorityWorkBudget:
        raise TypeError('Actual original provider owner and work lease required')
    owner.verify(); budget.check_time()
    if budget.bridge._active is not budget:
        raise ValueError('Original provider reservation is inactive')
    executable = str(Path(command[0]).resolve(strict=True))
    expected = CodexBatchBackend(runtime.parent, executable=executable)._command(runtime, runtime/'response-schema.json')
    if command != expected or (runtime/'stdin.txt').read_text('utf-8') != text:
        raise ValueError('Original provider command or prompt changed')
    from .work_lease import _observe, _evaluate
    observed = _observe(budget.lease)
    if not _evaluate(budget.lease, observed).allowed:
        raise ValueError('Original provider lease closed before invocation')
    value = {'version': VERSION, 'trial': owner.permit.trial, 'physical_start': observed,
        'context_digest': fingerprint(owner.context), 'job_digest': fingerprint(owner.job),
        'reservation_id': budget.record['reservation_id'], 'lease_hash': budget.record['lease_hash'],
        'command': command, 'executable_sha256': file_hash(executable),
        'stdin_sha256': file_hash(runtime/'stdin.txt'), 'schema_sha256': file_hash(runtime/'response-schema.json')}
    immutable_record(runtime/'invocation-original.json', value)
    return value


def _finish_provider(runtime, original, budget):
    # Invoked even when the real process fails. Absent/truncated raw output is
    # retained and later becomes unknown; a success flag is never an attestation.
    files = {name: file_hash(runtime/name) if (runtime/name).is_file() else None
             for name in ('events.jsonl', 'stderr.log')}
    from .work_lease import _observe
    immutable_record(runtime/'invocation-output.json', {'original_digest': fingerprint(original),
        'files': files, 'physical_finish': _observe(budget.lease)})


def verify_provider_attempt(permit, context, attempt):
    from .full_calibration_registry import _read
    from .neural_forward_ticket import open_existing_authority
    from .v7_agent import CodexBatchBackend
    contract = permit.registry.plan.mechanism_contract
    if contract is None:
        return {'status': 'unknown', 'reason': 'no_prospective_mechanism_contract'}
    if not (attempt/'invocation-original.json').is_file() or not (attempt/'invocation-output.json').is_file():
        return {'status': 'unknown', 'reason': 'original_invocation_or_completion_missing'}
    value, output = _read(attempt/'invocation-original.json'), _read(attempt/'invocation-output.json')
    executable = contract['provider_executable']
    expected_command = CodexBatchBackend(attempt.parent, executable=executable)._command(attempt, attempt/'response-schema.json')
    authority = open_existing_authority(permit.paths['authority'])
    record = authority.get(value['reservation_id'])
    if (set(value) != {'version','trial','physical_start','context_digest','job_digest','reservation_id',
                       'lease_hash','command','executable_sha256','stdin_sha256','schema_sha256'}
            or set(output) != {'original_digest','files','physical_finish'}
            or value['version'] != VERSION or value['trial'] != permit.trial
            or value['context_digest'] != fingerprint(context) or value['job_digest'] != context['job_hash']
            or value['lease_hash'] != record['lease_hash'] or record['binding']['work']['stage'] != 'family'
            or value['command'] != expected_command
            or value['executable_sha256'] != contract['provider_executable_sha256']
            or file_hash(executable) != value['executable_sha256']
            or value['stdin_sha256'] != file_hash(attempt/'stdin.txt')
            or value['schema_sha256'] != file_hash(attempt/'response-schema.json')
            or {k:output[k] for k in ('original_digest','files')} != {'original_digest': fingerprint(value), 'files': {
                n: file_hash(attempt/n) if (attempt/n).is_file() else None for n in ('events.jsonl','stderr.log')}}):
        raise ValueError('Original provider execution/physical wallet/source evidence changed')
    from .work_lease import FrozenWorkLease, _evaluate
    lease=FrozenWorkLease.from_dict(record['lease'])
    if not _evaluate(lease,value['physical_start']).allowed:
        raise ValueError('Original provider began outside its real work lease')
    if not _evaluate(lease,output['physical_finish'],value['physical_start']).allowed:
        return {'status':'unknown','reason':'original_provider_physical_lease_completion_unverified'}
    if output['files']['events.jsonl'] is None:
        return {'status': 'unknown', 'reason': 'original_provider_output_absent'}
    raw = (attempt/'events.jsonl').read_text('utf-8')
    if len(raw.encode()) > 2_097_152:
        raise ValueError('Original provider output exceeds fixed bound')
    try:
        events = [json.loads(line) for line in raw.splitlines()]
    except (ValueError, UnicodeError):
        return {'status': 'unknown', 'reason': 'original_provider_output_incomplete'}
    models = [e.get('model') for e in events if type(e) is dict and type(e.get('model')) is str]
    if not models or set(models) != {contract['reported_resolved_model']}:
        return {'status': 'unknown', 'reason': 'reported_resolved_model_unknown_or_changed'}
    return {'status': 'verified_local_mechanism', 'original_digest': fingerprint(value),
        'output_digest': fingerprint(output), 'reported_resolved_model': models[-1],
        'identity_scope': contract['identity_scope'], 'server_identity': 'unknown',
        'independence': 'prospective_modelling_assumption'}


def mechanism_domains(contract):
    """Separate original draw domains for an entire multi-law scenario grid."""
    return {role+'_digest':fingerprint({'version':VERSION,'role':role,
        'generators':contract['generators'],'scenario_roles':contract['scenario_roles']})
        for role in ('design','validation','oracle')}
