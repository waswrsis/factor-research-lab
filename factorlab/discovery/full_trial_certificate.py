"""Recomputed full-scope calibration under prospectively declared mechanism assumptions.

Every gate is derived from canonical original grid/controller/provider/oracle evidence.
Missing evidence remains pending; the original .05/.8 targets are unchanged.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from collections import Counter

from factorlab.protocol import fingerprint
from factorlab.discovery.calibration_acceptance import (
    CLAIMS, ClaimObservation, TrialObservation, OracleRiskEnvelope, acceptance_report,
)
from factorlab.discovery.calibration_trial_context import _no_redirect, verify_memory_namespace
from factorlab.discovery.full_calibration_registry import FullTrialRegistry, QuarantinedTrialPermit, _read
from factorlab.discovery.evidence import file_hash
from factorlab.discovery.v7_storage import immutable_record

VERSION = 'v7-original-full-trial-certificate-v1'


class FullScopePending(ValueError):
    """Engineering artifact verifies; its scientific scope remains unsupported."""


def _controller_type():
    # Kept a normal package import in production; tests load the staged modules
    # under that exact canonical name, not a substitute execution implementation.
    from factorlab.discovery.full_trial_controller import FullTrialController
    return FullTrialController


def _claims(permit, *, include_scope=False):
    from factorlab.discovery.calibration_oracle_evidence import _lifecycle
    from factorlab.discovery.claim_protocol import ClaimProtocol
    from factorlab.discovery.quarantine_execution import verify_decision
    records = {}
    with _lifecycle(permit.paths['runtime']) as lc:
        permit.verify(lc)
        lc.verify()
        protocol = ClaimProtocol(lc)
        # Derive the scientific endpoint contract from the original bound
        # protocol, never from a certificate caller. Enrollment has already
        # compared the entire actual runtime with the preregistered profile,
        # apart from its necessarily trial-specific receipt source identity.
        from factorlab.discovery.quarantine_execution import six_month_claim_scope
        scope = six_month_claim_scope(protocol)
        scope['runtime_scope_hash'] = fingerprint(permit.registry.plan.runtime_scope)
        for slot in ('library_skill', 'algorithm'):
            row = protocol._row(slot)
            if row is None or row[1] is None:
                records[slot] = {'state': 'pending', 'accepted': False}
                continue
            score = json.loads(row[1])
            decision = verify_decision(lc, score['result'])
            records[slot] = {'state': 'complete', 'accepted': decision['accepted'],
                             'score_digest': fingerprint(score)}
        batches = lc.db.execute('SELECT id,ordinal FROM batches ORDER BY ordinal').fetchall()
        if len(batches) > 3 or any(ordinal != i+1 for i, (_, ordinal) in enumerate(batches)):
            raise ValueError('Five-claim ledger has replaced or extra factor slots')
        for i in range(1, 4):
            if i > len(batches):
                records['factor'+str(i)] = {'state': 'pending', 'accepted': False}
                continue
            batch, _ = batches[i-1]
            identity, status, selected = lc._batch(batch)
            result = lc.db.execute('SELECT result FROM batches WHERE id=?', (batch,)).fetchone()[0]
            if status == 'concluded_no_candidate':
                records['factor'+str(i)] = {'state': 'complete', 'accepted': False,
                    'batch_digest': fingerprint(identity), 'status': status}
            elif status == 'concluded':
                score = json.loads(result)
                decision = verify_decision(lc, score)
                records['factor'+str(i)] = {'state': 'complete', 'accepted': decision['accepted'],
                    'batch_digest': fingerprint(identity), 'score_digest': fingerprint(score)}
            else:
                records['factor'+str(i)] = {'state': 'pending', 'accepted': False,
                    'batch_digest': fingerprint(identity), 'status': status}
        claims = {slot: records[slot] for slot in CLAIMS}
        return (claims, scope) if include_scope else claims


def _search_journal(root):
    """Authenticate unfinished as well as completed S audit denominators."""
    from .v7_run import RunStore, source_files
    database=root/'state.sqlite'
    if not database.exists():
        if (root/'manifest.json').exists() or (root/'source').exists():
            raise ValueError('Original S search journal missing; cannot erase its audit denominator')
        return []
    manifest=_read(root/'manifest.json')
    identity=manifest['identity']
    if manifest['hash']!=fingerprint(identity) or identity['source_files']!=source_files():
        raise ValueError('Original S search source/manifest changed')
    store=RunStore.__new__(RunStore)
    store.directory,store.identity=root,identity
    with closing(sqlite3.connect(database.as_uri()+'?mode=ro',uri=True)) as db:
        store.db=db
        db.execute('PRAGMA query_only=ON'); db.execute('BEGIN')
        store.verify()
        return [(month,json.loads(text),digest) for month,text,digest in
                db.execute('SELECT month,value,hash FROM months ORDER BY month')]


def _request_key(month, record):
    digest=record.get('request_hash') if type(record) is dict else None
    if (type(digest) is not str or len(digest)!=64
            or any(c not in '0123456789abcdef' for c in digest)):
        return None
    return month,digest


_REQUEST_FIELDS=('request_kind','template_version','prefix_hash','model_requested',
                 'reasoning','stdin_characters','stdin_utf8_bytes')


def _request_metadata(record):
    if type(record) is not dict or any(k not in record for k in _REQUEST_FIELDS):
        return None
    return fingerprint({k:record[k] for k in _REQUEST_FIELDS})


def _search(permit):
    """Read actual S evidence and all raw provider attempts; never initiate work."""
    from factorlab.discovery.calibration_coverage import _monthly_evidence
    root = permit.paths['research']
    verify_memory_namespace(root)
    months, audit_rows, pending = [], [], []
    expected=Counter(); actual=Counter(); expected_packets=Counter(); actual_packets=Counter()
    for month,state,digest in _search_journal(root):
        audit=state.get('agent_audit')
        charged=state.get('budget',{}).get('counts',{}).get('gpt_attempt')
        if type(audit) is not list:
            pending.append(month+':original_agent_audit_missing'); audit=[]
        if type(charged) is not int or charged!=len(audit):
            pending.append(month+':original_charged_attempt_count_differs_from_audit')
        for ordinal,record in enumerate(audit):
            key=_request_key(month,record)
            if key is None:
                pending.append(month+':original_audit_request_identity_missing')
            else:
                expected[key]+=1
                metadata=_request_metadata(record)
                if metadata is None:
                    pending.append(month+':original_audit_request_metadata_missing')
                else:
                    expected_packets[(*key,metadata)]+=1
            audit_rows.append({'month':month,'ordinal':ordinal,'record':record})
        item={'month':month,'state':'complete' if state['status']=='complete' else 'incomplete',
              'state_digest':digest,'provider_attempt_records':audit,
              'agent_attempts':len(audit),'charged_agent_attempts':charged}
        if state['status']=='complete':
            evidence=_monthly_evidence(root,month)
            if (evidence['provider_attempt_records']!=audit or evidence['agent_attempts']!=len(audit)):
                raise ValueError('Original complete S evidence differs from its journal audit')
            item['evidence']=evidence
        months.append(item)
    attempts, contexts, mechanisms = [], [], []
    provider = permit.paths['provider']
    for context_path in sorted(provider.glob('*/context.json')):
        _no_redirect(context_path, provider)
        context = _read(context_path)
        job_path = permit.paths['runtime']/'autonomous'/'jobs'/context_path.parent.name/'job.json'
        job = _read(job_path)
        from factorlab.discovery.full_trial_controller import VERSION as CONTROLLER_VERSION
        expected_context = {'version': CONTROLLER_VERSION, 'trial': permit.trial,
            'provider_policy': permit.registry.plan.provider_policy,
            'job_hash': fingerprint(job), 'job_path': str(job_path),
            'context_policy': 'ephemeral_no_resume_no_user_rules_no_host_memory',
            'provider_authenticity_certified': False}
        if context != expected_context:
            raise ValueError('Original provider context changed its trial or requested policy')
        if job.get('month')!=context_path.parent.name:
            raise ValueError('Original provider context differs from its job month')
        contexts.append({'month': context_path.parent.name, 'digest': fingerprint(context)})
        for attempt in sorted((context_path.parent/'attempts').glob('*')):
            _no_redirect(attempt, provider)
            if not attempt.is_dir():
                raise ValueError('Unexpected original provider attempt artifact')
            path = attempt/'attempt.json'
            record = _read(path) if path.exists() else None
            key=_request_key(context_path.parent.name,record)
            if key is None:
                pending.append(str(attempt.relative_to(provider))+':raw_request_identity_missing')
            else:
                actual[key]+=1
                metadata=_request_metadata(record)
                if metadata is None:
                    pending.append(str(attempt.relative_to(provider))+':raw_request_metadata_missing')
                else:
                    actual_packets[(*key,metadata)]+=1
                # Reconstruct the request from original stdin. Neither an
                # editable attempt tag nor equal total counts can relabel it.
                from .prompt_builder import _digest, _json
                stdin=attempt/'stdin.txt'
                if not stdin.is_file() or record.get('model_requested')!=permit.registry.plan.provider_policy.get('model_requested'):
                    pending.append(str(attempt.relative_to(provider))+':raw_request_input_or_model_missing')
                else:
                    request=_digest(_json({'text':stdin.read_text('utf-8'),
                        'model':record.get('model_requested'),'reasoning':record.get('reasoning')}))
                    if (request!=record['request_hash'] or
                            record.get('reasoning')!=permit.registry.plan.provider_policy.get('reasoning_requested')):
                        pending.append(str(attempt.relative_to(provider))+':raw_request_hash_or_reasoning_differs')
            files = {}
            for item in sorted(attempt.iterdir()):
                if item.is_file():
                    files[item.name] = file_hash(_no_redirect(item, provider))
            from .full_mechanism_contract import verify_provider_attempt
            mechanisms.append(verify_provider_attempt(permit, context, attempt))
            attempts.append({'month':context_path.parent.name,
                'relative_path': str(attempt.relative_to(provider)), 'record': record,
                'state': 'recorded' if record is not None else 'orphaned_consumed',
                'context_digest': fingerprint(context), 'files': files})
    if {p.name for p in provider.glob('*') if p.is_dir()} != {c['month'] for c in contexts}:
        raise ValueError('Original provider context missing; cannot omit orphan attempts')
    missing=expected-actual; extra=actual-expected
    if missing: pending.append('original_audit_attempts_missing_raw_directories')
    if extra: pending.append('raw_attempts_without_original_month_audit')
    if expected_packets!=actual_packets:
        pending.append('original_request_metadata_multiset_differs')
    coverage={'complete':not pending,
        'expected_attempts':len(audit_rows),'raw_attempts':len(attempts),
        'charged_attempts':sum(m['charged_agent_attempts'] for m in months)
            if all(type(m['charged_agent_attempts']) is int for m in months) else None,
        'missing':[{'month':k[0],'request_hash':k[1],'count':v} for k,v in sorted(missing.items())],
        'extra':[{'month':k[0],'request_hash':k[1],'count':v} for k,v in sorted(extra.items())],
        'audit_digest':fingerprint(audit_rows),'pending':sorted(set(pending))}
    terminal=(all(type(row['record']) is dict and row['record'].get('status') in {'validated','failed','interrupted'}
                  for row in audit_rows)
              and all((a['record'] or {}).get('status') in {'completed','failed'} for a in attempts))
    return {'months': months, 'contexts': contexts, 'attempts': attempts,
            'provider_attempt_coverage':coverage,
            'provider_mechanism_evidence': mechanisms,
            'provider_mechanism_verified': bool(attempts) and coverage['complete'] and terminal
                and all(m['status']=='verified_local_mechanism' for m in mechanisms),
            'provider_authentication': 'local_raw_exec_report_only_server_identity_unknown',
            'all_resolved_models_known': bool(attempts) and coverage['complete'] and terminal and all(
                type((a['record'] or {}).get('model_resolved')) is str and bool(a['record']['model_resolved'])
                for a in attempts)}


class _DigestCensus:
    """Streaming full-grid commitment, including every missing/failed position."""
    def __init__(self, domain):
        self.head=fingerprint({'domain':domain}); self.count=0; self.states={}

    def append(self,value):
        self.head=fingerprint({'previous':self.head,'ordinal':self.count,'original':value})
        self.count+=1
        state=value.get('state','registered')
        self.states[state]=self.states.get(state,0)+1

    def summary(self):
        return {'count':self.count,'complete_original_chain_digest':self.head,'states':self.states}


def _truth_pattern_coverage(plan, observations, mechanism):
    """Observed original truth patterns, never the caller's scenario labels."""
    from fractions import Fraction
    roles = {} if mechanism is None else mechanism['scenario_roles']
    cells = {c.name: {'role':roles.get(c.name), 'all_five_known_null':0,
        'mixed_known_null_and_alternative':0, 'known_boundary_trials':0,
        'uncertain_boundary_interval_trials':0, 'known_alternatives':dict.fromkeys(CLAIMS,0)}
        for c in plan.cells}
    for row in observations:
        if row.state != 'complete': continue
        counts=cells[row.cell]
        truth={c.slot:'uncertain' if c.oracle is None else c.oracle.truth(plan) for c in row.claims}
        counts['all_five_known_null']+=int(all(t=='null' for t in truth.values()))
        counts['mixed_known_null_and_alternative']+=int('null' in truth.values() and 'alternative' in truth.values())
        for slot in CLAIMS: counts['known_alternatives'][slot]+=int(truth[slot]=='alternative')
        known=uncertain=False
        for claim in row.claims:
            oracle=claim.oracle
            if oracle is None: continue
            b0,b1=map(Fraction.from_float,map(float,oracle.reference))
            c0,c1=map(Fraction.from_float,map(float,oracle.candidate))
            if b0>0:
                lo,hi=1-c1/b0,1-c0/b1
                known |= lo==hi==Fraction(1,200)
                uncertain |= lo<=Fraction(1,200)<=hi and lo!=hi
        counts['known_boundary_trials']+=int(known)
        counts['uncertain_boundary_interval_trials']+=int(uncertain)
    pending=[]
    if not any(v['role']=='null' and v['all_five_known_null'] for v in cells.values()):
        pending.append('original_all_five_known_null_truth_coverage')
    if not any(v['role']=='mixed' and v['mixed_known_null_and_alternative'] for v in cells.values()):
        pending.append('original_mixed_null_alternative_truth_coverage')
    for slot in CLAIMS:
        if not any(c.power_slots and slot in c.power_slots and cells[c.name]['known_alternatives'][slot]
                   for c in plan.cells):
            pending.append(slot+':original_positive_truth_coverage')
    if not any(v['role']=='boundary' and v['known_boundary_trials'] for v in cells.values()):
        pending.append('original_boundary_truth_at_delta0_not_identified')
    return {'cells':cells,'boundary_interval_is_not_known_truth':True,'pending':pending}


def reconstruct_full_certificate(registry):
    if type(registry) is not FullTrialRegistry:
        raise TypeError('Original full registry required; no arrays or success records')
    registry.verify()
    from factorlab.discovery.neural_forward_ticket import open_existing_authority
    from factorlab.discovery.full_adaptive_risk import reconstruct_full_risk
    from factorlab.discovery.full_mechanism_contract import verify_mechanism_contract
    mechanism = registry.plan.mechanism_contract
    if mechanism is not None:
        verify_mechanism_contract(mechanism, registry.plan.acceptance)
    pending = set()
    if mechanism is None:
        pending.add("prospective_joint_mechanism_contract")
    trials, observations = _DigestCensus('full_original_execution_grid'), []
    endpoint_scopes = {}
    expected_grid = _DigestCensus('full_original_registered_grid')
    for cell in registry.plan.acceptance.cells:
        for index in range(cell.repetitions):
            trial = registry.identity_for(cell.name, index)
            expected_grid.append(trial)
            paths = registry.paths(cell.name, index)
            mirror = registry.directory/'controller_plans'/(trial['trial_id']+'.json')
            if not (paths['root']/'enrollment.json').exists():
                # An incomplete enrolled namespace is retained as missing; any
                # existing evidence prevents pretending this is an unused slot.
                orphan = paths['root'].exists() and any(paths['root'].iterdir()) or mirror.exists()
                trials.append({'trial': trial, 'state': 'orphaned' if orphan else 'missing'})
                continue
            permit = QuarantinedTrialPermit(registry, trial)
            permit.verify()
            if not (paths['root']/'controller'/'identity.json').exists():
                trials.append({'trial': trial, 'state': 'pending_controller'})
                continue
            controller = _controller_type().inspect(permit)
            steps = controller.verify()
            claims, endpoint_scope = _claims(permit, include_scope=True)
            endpoint_scopes[fingerprint(endpoint_scope)] = endpoint_scope
            search = _search(permit)
            wallet = open_existing_authority(paths['authority']).verify()
            oracle = reconstruct_full_risk(registry, cell.name, index)
            pending.update(oracle['pending'])
            if not search['provider_mechanism_verified']:
                pending.add('original_local_provider_identity_and_fresh_context_evidence')
            if not search['provider_attempt_coverage']['complete']:
                pending.add('original_S_audit_and_raw_provider_attempt_multiset_coverage')
            complete = (len(steps) == len(controller.schedule.due_at)
                and all(controller.step_completed(r) for r in steps)
                and all(c['state'] == 'complete' for c in claims.values())
                and bool(search['months']) and all(m['state'] == 'complete' for m in search['months']))
            execution = {'trial': trial, 'state': 'complete' if complete else 'interrupted_consumed',
                'controller_identity_digest': fingerprint(controller.identity), 'steps': steps,
                'claims': claims, 'search': search, 'wallet': wallet, 'oracle': oracle,
                'full_pipeline_execution_verified': complete}
            digest = fingerprint(execution)
            # Keep the certificate bounded. This digest is recomputed from the
            # complete original contents above on EVERY verification; it is not
            # a caller-supplied hash or a replacement for reading the evidence.
            trials.append({'trial': trial, 'state': execution['state'],
                'complete_evidence_digest': digest, 'claims': claims,
                'controller_identity_digest': execution['controller_identity_digest'],
                'original_steps': len(steps), 'original_S_months': len(search['months']),
                'original_provider_attempts': max(search['provider_attempt_coverage']['expected_attempts'],
                    search['provider_attempt_coverage']['charged_attempts'] or 0, len(search['attempts'])),
                'original_raw_provider_attempts': len(search['attempts']),
                'original_provider_attempt_coverage': search['provider_attempt_coverage'],
                'original_wallet': wallet, 'original_oracle_digest': fingerprint(oracle)})
            observations.append(TrialObservation(cell.name, index, execution['state'], digest,
                tuple(ClaimObservation(slot, claims[slot]['accepted'] if complete else False,
                    None if oracle['claims'][slot] is None else OracleRiskEnvelope(**{
                        **oracle['claims'][slot], 'reference': tuple(oracle['claims'][slot]['reference']),
                        'candidate': tuple(oracle['claims'][slot]['candidate'])})) for slot in CLAIMS)))
    numeric = acceptance_report(registry.plan.acceptance, observations)
    truth_coverage = _truth_pattern_coverage(registry.plan.acceptance, observations, mechanism)
    pending.update(truth_coverage['pending'])
    if not numeric['all_trials_complete']:
        pending.add('entire_original_1000_to_2000_repetition_grid_completed')
    if not numeric['numeric_targets_met']:
        pending.add('original_simultaneous_one_sided_95_percent_FWER_05_and_power_80_targets')
    if set().union(*(set(c.power_slots) for c in registry.plan.acceptance.cells)) != set(CLAIMS):
        pending.add('preregistered_unconditional_power_for_all_five_claim_slots')
    if len(endpoint_scopes) != 1:
        pending.add('one_original_frozen_six_month_claim_scope_across_all_trials')
    endpoint_scope = next(iter(endpoint_scopes.values())) if len(endpoint_scopes) == 1 else None
    eligible = not pending
    pending = sorted(pending)
    record = {'version': VERSION, 'registry_digest': registry.digest, 'plan_digest': registry.plan.digest,
        'implementation_hash': file_hash(__file__), 'runtime_scope': registry.plan.runtime_scope,
        'runtime_scope_hash': fingerprint(registry.plan.runtime_scope),
        'six_month_claim_scope': endpoint_scope,
        'six_month_claim_scope_hash': fingerprint(endpoint_scope) if endpoint_scope is not None else None,
        'provider_policy': registry.plan.provider_policy, 'generator_design': registry.plan.generator_design,
        'expected_grid_digest': expected_grid.head, 'registered_trials': expected_grid.count,
        'trials': trials.summary(), 'numeric_acceptance': numeric, 'pending_dependencies': pending,
        'original_truth_pattern_coverage':truth_coverage,
        'status': 'verified_full_scope_under_declared_mechanism' if eligible else 'pending_full_scope_evidence',
        'formal_gate_eligible': eligible, 'full_V7': eligible,
        'full_V7_five_claim_pipeline_calibrated': eligible, 'covered_claim_types': list(CLAIMS),
        'test_only': False, 'rolling_hash': registry.plan.runtime_scope['rolling_hash'],
        'rule_hash': registry.plan.runtime_scope['factor_confirmation_rule_hash'],
        'mechanism_contract': mechanism,
        'truth_scope': 'original_full_path_risk_under_prospectively_declared_joint_mechanism',
        'scope': 'finite_registered_simulation_and_declared_mechanism_only_not_unbounded_market_validity'}
    return json.loads(json.dumps(record, sort_keys=True, allow_nan=False))


def produce_full_certificate(registry):
    record = reconstruct_full_certificate(registry)
    digest = fingerprint(record)
    immutable_record(registry.directory/'full_certificates'/(digest+'.json'), record)
    return {'artifact_digest': digest, 'status': record['status'],
            'formal_gate_eligible': record['formal_gate_eligible'], 'pending_dependencies': record['pending_dependencies']}


def verify_full_certificate(registry, *, artifact_digest, expected_runtime_scope_hash, require_formal=True):
    """Rebuild all content; an outer digest or changed Boolean never grants scope."""
    if (type(artifact_digest) is not str or len(artifact_digest) != 64
            or any(c not in '0123456789abcdef' for c in artifact_digest)
            or type(require_formal) is not bool):
        raise ValueError('Canonical original complete-artifact digest required')
    record = reconstruct_full_certificate(registry)
    stored = _read(registry.directory/'full_certificates'/(artifact_digest+'.json'))
    if (fingerprint(record) != artifact_digest or stored != record
            or expected_runtime_scope_hash != record['runtime_scope_hash']):
        raise ValueError('Complete original certificate, runtime availability or scope changed')
    if require_formal and (record['pending_dependencies'] or not record['formal_gate_eligible']):
        raise FullScopePending('Full calibration remains pending: '+', '.join(record['pending_dependencies']))
    return record


def publish_lifecycle_certificate(registry, directory, *, artifact_digest):
    """Publish an immutable G11 reference; verification always rereads originals."""
    from factorlab.protocol import project_path
    directory = project_path(directory)
    record = verify_full_certificate(registry, artifact_digest=artifact_digest,
        expected_runtime_scope_hash=fingerprint(registry.plan.runtime_scope), require_formal=False)
    reference = {'version': 'v7-full-original-certificate-reference-v1',
        'registry_directory': str(registry.directory), 'registry_digest': registry.digest,
        'artifact_digest': artifact_digest, 'runtime_scope_hash': record['runtime_scope_hash']}
    binding = fingerprint(reference)
    immutable_record(directory/'certificate.json', {**reference, 'binding_id': binding})
    return {'calibration_directory': str(directory), 'calibration_binding': binding,
            'formal_gate_eligible': record['formal_gate_eligible']}


def verify_lifecycle_certificate(directory, *, expected_binding=None):
    from factorlab.protocol import project_path
    value = _read(project_path(directory)/'certificate.json')
    expected = {k:value[k] for k in ('version','registry_directory','registry_digest',
                                   'artifact_digest','runtime_scope_hash')}
    if (set(value) != set(expected)|{'binding_id'}
            or expected['version'] != 'v7-full-original-certificate-reference-v1'
            or value['binding_id'] != fingerprint(expected)):
        raise ValueError('Original full calibration reference changed')
    requested = expected_binding if type(expected_binding) is str else fingerprint(expected_binding)
    if expected_binding is not None and requested != value['binding_id']:
        raise ValueError('Lifecycle does not bind this original full certificate')
    registry=FullTrialRegistry.inspect(value['registry_directory'])
    if registry.digest!=value['registry_digest']:
        raise ValueError('Lifecycle full registry identity changed')
    record=verify_full_certificate(registry,artifact_digest=value['artifact_digest'],
        expected_runtime_scope_hash=value['runtime_scope_hash'],require_formal=False)
    return {**record,'binding_id':value['binding_id']}
