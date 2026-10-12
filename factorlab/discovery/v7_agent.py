"""V7 batch protocol and injectable proposer; no implicit remote backend.

An adapter callable receives (PromptPacket, *, model, reasoning) and returns
BackendReply. A controller supplies an audited Codex exec adapter explicitly;
The optional explicit Codex adapter never changes credentials or providers.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .local_mutation import MonthlyBudget, ProgramTemplate, _id
from .program_syntax import CompactProgram, strict_json
from .prompt_builder import DEFAULT_MODEL, DEFAULT_REASONING, PromptIdentity, PromptPacket, _text

BATCH_VERSION = 'v7-proposal-batch-v1'


@dataclass(frozen=True)
class ProposalBatch:
    _value: dict

    @classmethod
    def from_dict(cls, value, *, expected_identity: PromptIdentity,
                  feature_names: tuple[str, ...], allowed_parent_ids: set[str]):
        expected = {'version', 'identity', 'family_id', 'hypothesis', 'mechanism',
                    'competing_explanation', 'falsification', 'parent_ids', 'template', 'programs'}
        if type(value) is not dict or set(value) != expected or value['version'] != BATCH_VERSION:
            raise ValueError('Invalid proposal batch schema/version')
        identity = PromptIdentity.from_dict(value['identity'])
        if identity != expected_identity:
            raise ValueError('Proposal task/rolling/pool/input identity mismatch')
        _id(value['family_id'])
        for key in ('hypothesis', 'mechanism', 'competing_explanation', 'falsification'):
            _text(value[key], key, 1200)
        parents = value['parent_ids']
        if (type(parents) is not list or not 1 <= len(parents) <= 4
                or any(type(p) is not str for p in parents) or len(set(parents)) != len(parents)
                or not set(parents) <= allowed_parent_ids):
            raise ValueError('Unknown or invalid parent IDs')
        template = ProgramTemplate.from_dict(value['template'])
        programs = value['programs']
        if type(programs) is not list or not 1 <= len(programs) <= 8:
            raise ValueError('Batch must contain 1..8 concrete programs')
        clean = []
        for payload in programs:
            compact = CompactProgram.from_dict(payload)
            program = compact.to_program()
            if program.feature_names != tuple(feature_names) or len(program.outputs) != 1:
                raise ValueError('Program input schema or scalar output mismatch')
            if not template.contains(program):
                raise ValueError('Concrete program does not belong to the declared template')
            clean.append(compact.to_dict())
        # Exact duplicates remain visible; the local materialization ledger counts
        # them. This parser never turns them into independent novel proposals.
        result = copy.deepcopy(value)
        result['programs'] = clean
        result['template'] = template.to_dict()
        return cls(result)

    def to_dict(self):
        return copy.deepcopy(self._value)

    @property
    def family_id(self):
        return self._value['family_id']

    @property
    def programs(self):
        return tuple(CompactProgram.from_dict(p).to_program() for p in self._value['programs'])

    @property
    def template(self):
        return ProgramTemplate.from_dict(self._value['template'])


@dataclass(frozen=True)
class BackendReply:
    payload: str
    usage: dict | None = None
    model_resolved: str | None = None
    backend_version: str | None = None


def batch_json_schema(packet: PromptPacket) -> dict:
    """Provider output schema; semantic validation remains ProposalBatch's job."""
    body = strict_json(packet.body)
    scalar = {'type': 'number'}
    text = {'type': 'string', 'minLength': 1, 'maxLength': 1200}

    def obj(properties):
        return {'type': 'object', 'properties': properties, 'required': list(properties),
                'additionalProperties': False}

    compact = obj({
        'syntax_version': {'type': 'string', 'enum': ['compact-causal-v1']},
        'feature_names': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1,
                          'maxItems': 48},
        'expressions': {'type': 'array', 'items': {'type': 'string', 'maxLength': 20000},
                        'minItems': 1, 'maxItems': 1},
        # Strict provider schemas require closed objects. Named parameters are
        # a finite array on the wire, converted to the internal DSL dictionary.
        'parameters': {'type': 'array', 'maxItems': 64,
                       'items': obj({'name': {'type': 'string'}, 'initial': scalar,
                                     'lower': scalar, 'upper': scalar})}})
    # The finite-slot schema admits JSON data, not code. Existing Program checks
    # remain authoritative for every constructed node/subtree.
    slot = obj({'path': {'type': 'array', 'minItems': 2, 'maxItems': 16,
                         'items': {'anyOf': [{'type': 'string'}, {'type': 'integer'}]}},
                'values': {'type': 'array', 'minItems': 1, 'maxItems': 32,
                           'items': {'anyOf': [{'type': 'number'}, {'type': 'string'}]}}})
    properties = {'version': {'type': 'string', 'enum': [BATCH_VERSION]},
                  'identity': obj({k: {'type': 'integer' if type(v) is int else 'string', 'enum': [v]}
                                   for k, v in packet.identity.to_dict().items()}),
                  'family_id': {'type': 'string', 'pattern': '^[A-Za-z0-9_.:-]{1,128}$'},
                  'hypothesis': text, 'mechanism': text, 'competing_explanation': text,
                  'falsification': text,
                  'parent_ids': {'type': 'array', 'minItems': 1, 'maxItems': 4,
                                 'items': {'type': 'string', 'enum':
                                 [p['parent_id'] for p in body['parents']]}},
                  'template': obj({'base': compact, 'slots': {'type': 'array', 'maxItems': 4,
                                                              'items': slot}}),
                  'programs': {'type': 'array', 'minItems': 1, 'maxItems': 8, 'items': compact}}
    return obj(properties)


def normalize_provider_batch(message):
    """Wire representation only; local schema and DSL validation remain strict."""
    value = strict_json(message)
    if type(value) is not dict:
        raise ValueError('Provider response must be a batch object')
    programs = list(value.get('programs', []))
    template = value.get('template')
    if type(template) is dict:
        programs.append(template.get('base'))
    for compact in programs:
        if type(compact) is not dict:
            continue
        parameters = compact.get('parameters')
        if type(parameters) is list:
            named = {}
            for parameter in parameters:
                if (type(parameter) is not dict
                        or set(parameter) != {'name', 'initial', 'lower', 'upper'}
                        or type(parameter['name']) is not str or parameter['name'] in named):
                    raise ValueError('Invalid or duplicate wire parameter')
                named[parameter['name']] = {k: parameter[k] for k in ('initial', 'lower', 'upper')}
            compact['parameters'] = named
    # JSON numbers have one schema type, while the interpreter canonicalizes
    # constants/parameter initials to float. Preserve values; never coerce an
    # integer window, a name, or an operator to another AST type.
    if type(template) is dict and type(template.get('slots')) is list:
        from .local_mutation import _get, _put
        from .program_contract import Program
        base = CompactProgram.from_dict(template['base']).to_program().to_dict()
        for slot in template['slots']:
            if type(slot) is not dict or type(slot.get('path')) is not list:
                continue
            original = _get(base, slot['path'])
            if type(original) is float and slot['path'][-1] in {'value', 'initial'}:
                slot['values'] = [float(v) if type(v) is int else v for v in slot['values']]
                for compact in value.get('programs', []):
                    program = CompactProgram.from_dict(compact).to_program().to_dict()
                    actual = _get(program, slot['path'])
                    if type(actual) is int:
                        _put(program, slot['path'], float(actual))
                        compact.update(CompactProgram.from_program(Program.from_dict(program)).to_dict())
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def _save_json(path: Path, value):
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class CodexBatchBackend:
    """Explicit Codex exec adapter using current login and frozen tool restrictions.

    The callable signature matches propose_batch's injected backend. Construction
    performs no remote work. No retries, API-key fallback or model substitution.
    Programs are validated by the Torch-free trusted contract.
    """

    def __init__(self, state_dir, *, executable=None, timeout_seconds=180,
                 max_output_bytes=1_048_576):
        if (not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 7200
                or type(max_output_bytes) is not int or not 1 <= max_output_bytes <= 2_097_152):
            raise ValueError('Invalid backend resource limits')
        self.state_dir = Path(state_dir).resolve()
        self.executable = executable
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes

    def _command(self, runtime: Path, schema_path: Path):
        # This restricted process policy has no dependency on neural modules.
        from .process_policy import DISABLED_FEATURES

        executable = self.executable or shutil.which('codex')
        if executable is None:
            raise RuntimeError('CodexCLIUnavailable')
        command = [str(executable), '--no-daemon', '-a', 'never', 'exec',
                   '--ignore-user-config', '--ignore-rules', '--ephemeral',
                   '--skip-git-repo-check', '--sandbox', 'read-only', '--json',
                   '--output-schema', str(schema_path), '-C', str(runtime)]
        for feature in DISABLED_FEATURES:
            command += ['--disable', feature]
        for setting in ['web_search="disabled"', 'tools.view_image=false', 'mcp_servers={}',
                        'project_doc_max_bytes=0', 'features.skip_host_skill_discovery=true',
                        'shell_environment_policy.inherit="none"',
                        'model_reasoning_effort="high"',
                        f'projects."{runtime.as_posix()}".trust_level="untrusted"']:
            command += ['-c', setting]
        return command + ['--model', DEFAULT_MODEL, '-']

    def _run(self, command, text, runtime):
        from .process_policy import isolated_process_kwargs, terminate_process_tree

        environment = {k: v for k, v in os.environ.items()
                       if k.upper() not in {'OPENAI_API_KEY', 'CODEX_API_KEY'}
                       and not any(w in k.upper() for w in ('TOKEN', 'SECRET', 'PASSWORD'))}
        with (tempfile.TemporaryFile(dir=runtime) as stdin,
              tempfile.TemporaryFile(dir=runtime) as stdout,
              tempfile.TemporaryFile(dir=runtime) as stderr):
            stdin.write(text.encode('utf-8'))
            stdin.seek(0)
            began = time.monotonic()
            process = subprocess.Popen(command, stdin=stdin, stdout=stdout,
                                       stderr=stderr, cwd=runtime, env=environment,
                                       shell=False, **isolated_process_kwargs())
            try:
                while process.poll() is None:
                    if os.fstat(stdout.fileno()).st_size > self.max_output_bytes:
                        raise ValueError('BackendOutputBudgetExceeded')
                    if os.fstat(stderr.fileno()).st_size > self.max_output_bytes:
                        raise ValueError('BackendDiagnosticBudgetExceeded')
                    if time.monotonic() - began > self.timeout_seconds:
                        raise TimeoutError('CodexBatchTimeout')
                    try:
                        process.wait(timeout=0.05)
                    except subprocess.TimeoutExpired:
                        pass
                if process.returncode:
                    raise RuntimeError(f'BackendExitCode_{process.returncode}')
                if os.fstat(stdout.fileno()).st_size > self.max_output_bytes:
                    raise ValueError('BackendOutputBudgetExceeded')
                stdout.seek(0)
                return stdout.read(self.max_output_bytes).decode('utf-8')
            finally:
                if process.poll() is None:
                    terminate_process_tree(process)
                # Preserve bounded provider diagnostics privately. They are not
                # fed back to GPT or included in published usage summaries.
                for handle, name in ((stdout, 'events.jsonl'), (stderr, 'stderr.log')):
                    handle.seek(0)
                    (runtime / name).write_bytes(handle.read(self.max_output_bytes))

    def __call__(self, packet: PromptPacket, *, model: str, reasoning: str) -> BackendReply:
        if (model, reasoning, packet.model, packet.reasoning) != (
            DEFAULT_MODEL, DEFAULT_REASONING, DEFAULT_MODEL, DEFAULT_REASONING
        ):
            raise ValueError('Codex batch backend freezes gpt-6-astra/high')
        from .prompt_request_kinds import assert_backend_request
        assert_backend_request(packet)
        if len(packet.text.encode()) > 50_000:
            raise ValueError('Backend input byte budget exceeded')
        runtime = self.state_dir / f'{packet.request_hash[:16]}-{uuid.uuid4().hex}'
        runtime.mkdir(parents=True, exist_ok=False)
        with (runtime / 'stdin.txt').open('x', encoding='utf-8', newline='') as handle:
            handle.write(packet.text)
        schema_path = runtime / 'response-schema.json'
        from .prompt_request_kinds import StructuralPacket, structural_schema
        _save_json(schema_path, structural_schema(packet) if type(packet) is StructuralPacket else batch_json_schema(packet))
        record = packet.telemetry() | {
            'backend': 'codex_exec_batch_v1', 'backend_version': None,
            'stdin_sha256': hashlib.sha256(packet.text.encode()).hexdigest(),
            'status': 'started', 'started_at_unix': time.time(),
            'finished_at_unix': None, 'elapsed_seconds': None,
            'spawn_to_first_response_seconds': None, 'usage_raw': None,
            'input_tokens': None, 'cached_input_tokens': None, 'output_tokens': None,
            'reasoning_output_tokens': None}
        _save_json(runtime / 'attempt.json', record)
        began = time.monotonic()
        try:
            output = self._run(self._command(runtime, schema_path), packet.text, runtime)
            message, usage, resolved = None, None, None
            started, completed = False, False
            event_types = []
            for line in output.splitlines():
                event = strict_json(line, max_bytes=self.max_output_bytes)
                if type(event) is not dict:
                    raise ValueError('Invalid Codex event')
                kind = event.get('type')
                event_types.append(kind if type(kind) is str and len(kind) <= 80 else None)
                if kind == 'turn.started':
                    started = True
                if kind in {'error', 'turn.failed'}:
                    raise RuntimeError('CodexTurnFailed')
                item = event.get('item')
                if item is not None:
                    if type(item) is not dict:
                        raise ValueError('Invalid Codex item')
                    if item.get('type') == 'error' and not started:
                        continue
                    if item.get('type') not in {'agent_message', 'reasoning', 'plan_update'}:
                        raise ValueError('CodexToolEventRejected')
                    if item.get('type') == 'agent_message' and kind == 'item.completed':
                        message = item.get('text')
                if kind == 'turn.completed':
                    completed = True
                    candidate_usage = event.get('usage')
                    if candidate_usage is not None and type(candidate_usage) is not dict:
                        raise ValueError('Invalid usage object')
                    usage = candidate_usage
                if type(event.get('model')) is str:
                    resolved = event['model'][:128]
            if not started or not completed or type(message) is not str:
                raise ValueError('CodexMissingCompletedStructuredResponse')
            record.update(status='completed', usage_raw=usage, usage=usage,
                          model_resolved=resolved, event_types=event_types)
            for key in ('input_tokens', 'cached_input_tokens', 'output_tokens',
                        'reasoning_output_tokens'):
                value = (usage or {}).get(key)
                if type(value) is int and value >= 0:
                    record[key] = value
            with (runtime / 'response-wire.json').open('x', encoding='utf-8') as handle:
                handle.write(message)
            normalized = (json.dumps(strict_json(message), ensure_ascii=False, allow_nan=False)
                          if type(packet) is StructuralPacket else normalize_provider_batch(message))
            with (runtime / 'response.json').open('x', encoding='utf-8') as handle:
                handle.write(normalized)
            return BackendReply(normalized, usage, resolved, None)
        except Exception as exc:
            record.update(status='failed', error_type=type(exc).__name__, error=str(exc)[:400])
            raise
        finally:
            record.update(finished_at_unix=time.time(), elapsed_seconds=time.monotonic() - began)
            _save_json(runtime / 'attempt.json', record)


def propose_batch(packet: PromptPacket, *, backend: Callable,
                  budget: MonthlyBudget, feature_names: tuple[str, ...],
                  allowed_parent_ids: set[str], audit: list[dict],
                  persist_budget: Callable[[dict], None]) -> ProposalBatch:
    """One attempt, no retries. Persist reservations before external work.

    ``audit`` is caller-owned and must be persisted by the run controller.
    All exceptions remain failures; none trigger a different backend/model.
    """
    if packet.model != DEFAULT_MODEL or packet.reasoning != DEFAULT_REASONING:
        raise ValueError('This protocol freezes gpt-6-astra/high')
    body = strict_json(packet.body)
    packet_features = tuple(body['parents'][0]['program']['feature_names'])
    packet_parents = {p['parent_id'] for p in body['parents']}
    if tuple(feature_names) != packet_features or not allowed_parent_ids <= packet_parents:
        raise ValueError('Response validation context differs from the request')
    from .prompt_request_kinds import validate_request
    validate_request(packet, budget)
    budget.consume('gpt_attempt')
    persist_budget(budget.to_dict())
    record = packet.telemetry() | {'attempt': budget.counts['gpt_attempt'],
                                   'started_at_unix': time.time(), 'status': 'started',
                                   'finished_at_unix': None, 'elapsed_seconds': None}
    audit.append(record)
    started = time.monotonic()
    try:
        # The controller persists the shared audit list alongside the budget.
        # Publish a started receipt before the backend can be interrupted.
        persist_budget(budget.to_dict())
        from .prompt_request_kinds import call_backend
        reply = call_backend(packet, backend, budget)
        if not isinstance(reply, BackendReply):
            raise TypeError('Backend must return BackendReply')
        record.update(usage=copy.deepcopy(reply.usage), model_resolved=reply.model_resolved,
                      backend_version=reply.backend_version)
        value = strict_json(reply.payload)
        if type(value) is dict and type(value.get('programs')) is list:
            # Even an invalid overfull reply consumes available materialization
            # quota; it is never silently truncated into a valid batch.
            seeds = len(value['programs'])
            if seeds:
                budget.consume('ast', seeds)
                persist_budget(budget.to_dict())
        batch = ProposalBatch.from_dict(value,
                                        expected_identity=packet.identity,
                                        feature_names=feature_names,
                                        allowed_parent_ids=allowed_parent_ids)
        from .prompt_request_kinds import validate_semantic_result
        validate_semantic_result(packet, batch)
        budget.register_family(batch.family_id)
        # Count concrete seed materializations, including exact duplicates.
        budget.attribute_ast(batch.family_id, len(batch.programs))
        persist_budget(budget.to_dict())
        record.update(status='validated', family_id=batch.family_id,
                      concrete_programs=len(batch.programs),
                      unique_programs=len({p.digest for p in batch.programs}))
        return batch
    except BaseException as exc:
        record.update(status='failed' if isinstance(exc, Exception) else 'interrupted',
                      error_type=type(exc).__name__, error=str(exc)[:400])
        raise
    finally:
        record.update(finished_at_unix=time.time(), elapsed_seconds=time.monotonic() - started)
        original_error = sys.exception()
        try:
            persist_budget(budget.to_dict())
        except Exception as persistence_error:
            if original_error is None:
                raise  # Never return an unpersisted successful proposal.
            original_error.add_note(
                'Final proposal audit persistence also failed: '
                f'{type(persistence_error).__name__}: {persistence_error}')
