"""Original paid S request slots; recommendations confer no work/admission grant."""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from contextvars import ContextVar
from pathlib import Path

from factorlab.protocol import fingerprint
from .prompt_builder import PromptPacket, clean_evidence, _json, _text
from .program_syntax import CompactProgram, strict_json
from .v7_storage import immutable_record

VERSION = 'v7-paid-request-kinds-v1'
_ACTIVE = ContextVar('original_paid_specialized_request', default=None)
POLICY = {'version': VERSION, 'family_second_request':
          ['semantic_review_after_invalid_program', 'formula_family_retry',
           'structural_probe_after_valid_formula_isolated_only'],
          'monthly_total_attempts': 6, 'semantic_max': 1, 'structural_max': 1,
          'unused_slots': 'nontransferable', 'training_or_adoption_permission': False}
SEMANTIC_PREFIX = ('Review the rejected causal formula semantics using only supplied S context. '
    'Return the ordinary complete typed proposal batch with corrected programs. Every corrected AST '
    'and family ID must differ from the rejected response. No V/C evidence, arbitrary code, new '
    'features, budget changes, or claimed test results. Explain the mechanism and falsification.\n')
STRUCTURAL_PREFIX = ('Select exactly one complete recipe_id from the supplied finite menu. '
    'Explain its mechanism using only the supplied S context. Return only version, recipe_id, '
    'mechanism, competing_explanation, falsification. This is a recommendation: original paid '
    'structural review, resource gates and independent recipe confirmation remain mandatory. '
    'Do not propose code, new experts, partial configuration, V/C evidence or training permission.\n')


@dataclass(frozen=True)
class SemanticPacket(PromptPacket):
    def telemetry(self):
        return super().telemetry() | {'request_kind': 'semantic_review', 'template_version': VERSION}


@dataclass(frozen=True)
class StructuralPacket(PromptPacket):
    def telemetry(self):
        return super().telemetry() | {'request_kind': 'structural_probe', 'template_version': VERSION}


def enabled(config):
    return config['budget_policy'] == 'standard' and config['search_mode'] == 'gpt'


def _read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2**20:
        raise ValueError('Missing bounded original request anchor')
    return strict_json(path.read_text('utf-8'))


def _binding(budget, family, *, attempts=1):
    from .work_authority_bridge import AuthorityWorkBudget
    if type(budget) is not AuthorityWorkBudget or budget.bridge._active is not budget:
        raise TypeError('Specialized requests require original active paid S family')
    budget.check_time()
    b = budget.bridge
    if not enabled(b.store.identity['config']):
        raise ValueError('Specialized requests cannot change controls or another experiment')
    b.store.exposure.assert_prompt_search(**{k: v for k, v in b.state['search_exposure'].items() if k != 'data_id'})
    paths = [b.directory/'schedule/identity.json', b.anchor/'schedule/identity.json']
    schedule = _read(paths[0])
    if (_read(paths[1]) != schedule or schedule['bridge_identity_hash'] != fingerprint(b.identity)
            or fingerprint(schedule['schedule']) != schedule['schedule_hash']):
        raise ValueError('Original request schedule identity changed')
    work = budget.record['binding']['work']
    slot = schedule['schedule']['slots'][work['slot']]
    if (work['schedule_hash'] != schedule['schedule_hash'] or work['stage'] != 'family'
            or slot['key'] != f'family:{family}' or slot['stage'] != 'family'
            or budget.record['counts']['gpt_attempt'] != 2
            or budget.actual_counts['gpt_attempt'] != attempts):
        raise ValueError('Specialized request is not its original unused second family slot')
    return {'version': VERSION, 'policy': copy.deepcopy(POLICY),
            'reservation_id': budget.record['reservation_id'], 'terms_hash': budget.record['terms_hash'],
            'bridge_identity_hash': fingerprint(b.identity), 'run_identity_hash': fingerprint(b.store.identity),
            'schedule_hash': schedule['schedule_hash'], 'family_index': family,
            'month': b.month, 'data_id': b.state['data_id'],
            'search_exposure': copy.deepcopy(b.state['search_exposure'])}


def _source(packet):
    if type(packet) is not PromptPacket:
        raise TypeError('Original formula packet required')
    body = strict_json(packet.body)
    for parent in body['parents']:
        clean_evidence(parent['evidence'], packet.identity)
    for evidence in body.get('diagnostics', []):
        clean_evidence(evidence, packet.identity)
    return body


def semantic_packet(packet, reply, budget):
    from .v7_agent import BackendReply
    binding = _binding(budget, 0)
    body = _source(packet)
    prior = budget.bridge.state['agent_audit'][-1]
    if (prior['request_hash'] != packet.request_hash or prior['status'] != 'failed'
            or prior['request_kind'] != 'formula_family' or type(reply) is not BackendReply):
        raise ValueError('Semantic review requires its actual failed original formula response')
    value = strict_json(reply.payload)
    if type(value) is not dict or type(value.get('programs', [])) is not list or len(value.get('programs', [])) > 8:
        return None  # Unparseable responses keep the dedicated slot unused.
    try:
        programs = [CompactProgram.from_dict(p).to_dict() for p in value.get('programs', [])]
    except (TypeError, ValueError, KeyError):
        return None
    rejected = {'request_hash': packet.request_hash, 'response_hash': fingerprint(value),
                'family_id': value.get('family_id'), 'programs': programs,
                'audit_hash': fingerprint(prior)}
    body.update(request_binding=binding, rejected=rejected)
    result = SemanticPacket(SEMANTIC_PREFIX + packet.prefix, _json(body), packet.identity)
    _freeze_request(result, budget)
    return result


def _recipe_menu(budget):
    from .joint_pool import FrozenPool
    from .neural_forward import FrozenNeuralRecipe
    from .rolling import RollingSpec
    from .structural_evolution import _history, local_mutations
    b = budget.bridge
    # Complete current Pv and exact frozen rolling contract, never caller choices.
    base = FrozenNeuralRecipe(pool=FrozenPool.from_dict(b.state['frozen_reference_pool']),
                             rolling_spec=RollingSpec(**b.store.identity['rolling_spec']))
    parents = [base]
    for _, ref in _history(b):
        value = ref['body']['recipe']
        if value is not None and value['pool'] == json.loads(_json(base.pool.to_dict())):
            parents.append(FrozenNeuralRecipe.from_dict(value))
    choices = {base.pipeline_id: base.to_dict()}
    for parent in parents:
        for eligible in (True, False):
            for recipe, _ in local_mutations(parent, eligible=eligible):
                choices[recipe.pipeline_id] = recipe.to_dict()
    # Max 24 historical recipes, <= 7 neighbors each. Prompt includes a bounded
    # deterministic prefix; unavailable recommendations cannot alter selection.
    return dict(sorted(choices.items())[:24])


def structural_packet(packet, budget):
    binding = _binding(budget, 2)
    b = budget.bridge
    prior = b.state['agent_audit'][-1]
    if (b.store.identity['config']['nonlinear_mode'] != 'isolated'
            or prior['request_hash'] != packet.request_hash or prior['status'] != 'validated'
            or prior['request_kind'] != 'formula_family'):
        raise ValueError('Structural request requires original successful third-family isolated S')
    body = _source(packet)
    # Preserve the original necessary semantics, units and causal input contract.
    # They live in the canonical formula's static prefix, not in its S body.
    # Only the formula-language/output schema is irrelevant to recipe-ID choice.
    parts = packet.prefix.split('\nSTATIC_CONTRACT_JSON\n')
    if len(parts) != 2:
        raise ValueError('Structural request requires original static task contract')
    static = strict_json(parts[1])
    task_contract = {k: static[k] for k in ('task_id', 'rolling_id', 'feature_contract_id',
        'training', 'task_semantics', 'feature_schema', 'input_semantics')}
    if (any(task_contract[k] != getattr(packet.identity, k)
            for k in ('task_id', 'rolling_id', 'feature_contract_id'))
            or task_contract['training'] != {'calendar_months': 3, 'timezone': 'UTC', 'initialization': 'cold'}):
        raise ValueError('Original structural task/training contract changed')
    prefix = STRUCTURAL_PREFIX + 'STATIC_TASK_CONTRACT_JSON\n' + _json(task_contract)
    body.update(request_binding=binding, recipes=_recipe_menu(budget),
                formula_request_hash=packet.request_hash, formula_audit_hash=fingerprint(prior))
    while len(_json(body).encode()) + len(prefix.encode()) > 49_000:
        if len(body['recipes']) <= 1:
            raise ValueError('Complete structural recipe exceeds original prompt byte bound')
        body['recipes'].pop(next(reversed(body['recipes'])))
    result = StructuralPacket(prefix, _json(body), packet.identity)
    _freeze_request(result, budget)
    return result


def _request_paths(budget, kind):
    b = budget.bridge
    return [root/'specialized_requests'/f'{kind}.json' for root in (b.directory, b.anchor)]


def _freeze_request(packet, budget):
    value = {'packet': {'prefix': packet.prefix, 'body': packet.body,
                       'identity': packet.identity.to_dict()}, 'telemetry': packet.telemetry()}
    for path in _request_paths(budget, packet.telemetry()['request_kind']):
        if path.exists():
            raise ValueError('Original specialized request is consumed/readback only')
    for path in _request_paths(budget, packet.telemetry()['request_kind']):
        immutable_record(path, value)
    budget.check_time()


def validate_request(packet, budget, *, _attempts=1):
    if type(packet) not in (SemanticPacket, StructuralPacket):
        if type(packet) is not PromptPacket:
            raise TypeError('Unregistered request kind')
        return
    family = 0 if type(packet) is SemanticPacket else 2
    if strict_json(packet.body)['request_binding'] != _binding(budget, family, attempts=_attempts):
        raise ValueError('Original specialized request binding changed')
    original = {'packet': {'prefix': packet.prefix, 'body': packet.body,
                          'identity': packet.identity.to_dict()}, 'telemetry': packet.telemetry()}
    if any(_read(p) != original for p in _request_paths(budget, packet.telemetry()['request_kind'])):
        raise ValueError('Original specialized request mirrors changed')


def call_backend(packet, backend, budget):
    if type(packet) not in (SemanticPacket, StructuralPacket):
        return backend(packet, model=packet.model, reasoning=packet.reasoning)
    validate_request(packet, budget, _attempts=2)
    audits = [a for a in budget.bridge.state['agent_audit'] if a['request_hash'] == packet.request_hash]
    if len(audits) != 1 or audits[0]['status'] != 'started':
        raise ValueError('Original charged specialized attempt must be persisted before backend')
    token = _ACTIVE.set((packet.request_hash, budget))
    try:
        result = backend(packet, model=packet.model, reasoning=packet.reasoning)
        budget.check_time()
        return result
    finally:
        _ACTIVE.reset(token)


def assert_backend_request(packet):
    if type(packet) in (SemanticPacket, StructuralPacket):
        current = _ACTIVE.get()
        if current is None or current[0] != packet.request_hash:
            raise ValueError('Specialized backend requires original prepaid invocation')
        validate_request(packet, current[1], _attempts=2)


def validate_semantic_result(packet, batch):
    if type(packet) is SemanticPacket:
        old = strict_json(packet.body)['rejected']
        old_ids = {CompactProgram.from_dict(p).to_program().digest for p in old['programs']}
        if batch.family_id == old['family_id'] or any(p.digest in old_ids for p in batch.programs):
            raise ValueError('Semantic revision requires new AST and candidate family identities')


def structural_schema(packet):
    if type(packet) is not StructuralPacket:
        raise TypeError('Exact structural packet required')
    text = {'type': 'string', 'minLength': 1, 'maxLength': 1200}
    properties = {'version': {'type': 'string', 'enum': [VERSION]},
                  'recipe_id': {'type': 'string', 'enum': list(strict_json(packet.body)['recipes'])},
                  'mechanism': text, 'competing_explanation': text, 'falsification': text}
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


def request_structure(packet, *, budget, backend, audit, persist_budget):
    from .v7_agent import BackendReply
    validate_request(packet, budget)
    budget.consume('gpt_attempt')
    record = packet.telemetry() | {'attempt': budget.counts['gpt_attempt'], 'status': 'started',
               'started_at_unix': time.time(), 'finished_at_unix': None, 'elapsed_seconds': None}
    audit.append(record)
    started = time.monotonic()
    try:
        persist_budget(budget.to_dict())
        reply = call_backend(packet, backend, budget)
        if type(reply) is not BackendReply:
            raise TypeError('Original typed backend reply required')
        budget.check_time()
        record.update(usage=copy.deepcopy(reply.usage), model_resolved=reply.model_resolved,
                      backend_version=reply.backend_version)
        result = strict_json(reply.payload)
        schema = structural_schema(packet)
        if (type(result) is not dict or set(result) != set(schema['properties'])
                or result['version'] != VERSION
                or result['recipe_id'] not in strict_json(packet.body)['recipes']):
            raise ValueError('Structural response must select one complete registered recipe ID')
        for field in ('mechanism', 'competing_explanation', 'falsification'):
            _text(result[field], field, 1200)
        record.update(status='validated', structural_reply=copy.deepcopy(result))
        return result
    except BaseException as error:
        record.update(status='failed' if isinstance(error, Exception) else 'interrupted',
                      error_type=type(error).__name__, error=str(error)[:400])
        raise
    finally:
        record.update(finished_at_unix=time.time(), elapsed_seconds=time.monotonic()-started)
        persist_budget(budget.to_dict())


def verified_recommendation(store, state, bridge_identity):
    """Read original completed family, audit and schedule; no arbitrary result JSON."""
    from .neural_forward_ticket import open_existing_authority
    store.verify()
    persisted = store.get(state['cutoff'][:7])
    # The original nonlinear producer adds its pending plan before its final
    # save. Verify precisely the original request inputs, not unrelated pending
    # fields; those remain subject to the producer's own publication protocol.
    if persisted is None or any(persisted.get(k) != state.get(k) for k in
            ('cutoff', 'data_id', 'search_exposure', 'agent_audit', 'frozen_reference_pool')):
        raise ValueError('Structural recommendation requires original persisted S state')
    roots = [store.directory/'work_authority'/state['cutoff'][:7],
             Path(bridge_identity['authority_directory'])/'search_bridges'/bridge_identity['bridge_id']]
    paths = [r/'specialized_requests/structural_probe.json' for r in roots]
    if not any(p.exists() for p in paths):
        if any(a['request_kind'] == 'structural_probe' for a in state.get('agent_audit', [])):
            raise ValueError('Missing original structural request')
        return None
    original = _read(paths[0])
    if _read(paths[1]) != original:
        raise ValueError('Original structural request mirrors changed')
    from .prompt_builder import PromptIdentity
    p = original['packet']
    packet = StructuralPacket(p['prefix'], p['body'], PromptIdentity.from_dict(p['identity']))
    if packet.telemetry() != original['telemetry']:
        raise ValueError('Original structural packet changed')
    body = strict_json(packet.body)
    binding = body['request_binding']
    if (binding['policy'] != POLICY or binding['family_index'] != 2
            or binding['bridge_identity_hash'] != fingerprint(bridge_identity)
            or binding['run_identity_hash'] != fingerprint(store.identity)
            or binding['data_id'] != state['data_id'] or binding['month'] != state['cutoff'][:7]
            or binding['search_exposure'] != state['search_exposure']):
        raise ValueError('Structural request changed its original S identity')
    store.exposure.assert_prompt_search(**{k: v for k, v in binding['search_exposure'].items() if k != 'data_id'})
    audits = [a for a in state['agent_audit'] if a['request_hash'] == packet.request_hash]
    if len(audits) != 1:
        raise ValueError('Original structural attempt missing or repeated')
    audit = audits[0]
    if audit['status'] != 'validated':
        return None  # Failed attempts remain charged, never become recommendations.
    authority = open_existing_authority(bridge_identity['authority_directory'])
    from .monthly_work_authority import _nested_identity_read, _IDENTITY_READ_SCOPE
    if _nested_identity_read(authority):
        # The outer canonical verifier is already auditing this exact locked
        # database/identity. Read its original transaction, never open a second
        # wallet or recurse through nonlinear allocation back into this reader.
        raw = _IDENTITY_READ_SCOPE.get()['db'].execute(
            'SELECT body FROM reservations WHERE id=?', (binding['reservation_id'],)).fetchone()
        if raw is None:
            raise ValueError('Original specialized paid family is missing')
        row = json.loads(raw[0])
    else:
        row = authority.get(binding['reservation_id'])
    if (row['terms_hash'] != binding['terms_hash'] or row['status'] != 'completed'
            or row['actual_counts']['gpt_attempt'] != 2
            or row['binding']['work']['schedule_hash'] != binding['schedule_hash']):
        raise ValueError('Structural recommendation lacks its original completed paid family')
    index = row['binding']['work']['slot']
    result_paths = [r/'schedule/results'/f'{index:03d}.json' for r in roots]
    result = _read(result_paths[0])
    record = result['record']
    output = record['result']
    if (_read(result_paths[1]) != result or fingerprint(record) != result['hash']
            or record['slot']['key'] != 'family:2' or record['schedule_hash'] != binding['schedule_hash']
            or output['status'] != 'completed' or output['reservation_id'] != row['reservation_id']
            or row['result']['result']['output_hash'] != fingerprint(output['value'])
            or output['value'].get('structural_audit_hash') != fingerprint(audit)):
        raise ValueError('Structural recommendation changed its original family publication')
    reply = audit['structural_reply']
    if reply['recipe_id'] not in body['recipes']:
        raise ValueError('Unregistered complete structural recipe')
    return {'request_hash': packet.request_hash, 'audit_hash': fingerprint(audit),
            'reservation_id': row['reservation_id'], 'recipe_id': reply['recipe_id'],
            'recipe': body['recipes'][reply['recipe_id']]}
