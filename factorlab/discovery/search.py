"""Bounded structure edits and validated development-only agent exchanges."""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field, replace
from itertools import combinations
from pathlib import Path

from factorlab.protocol import fingerprint, project_path, utc

from .contracts import ExperimentSpec, ModuleSpec, RepresentationSpec, TrainingSpec
from .evidence import immutable_json
from .robust_selection import ROBUST_METRIC_FIELDS, TASK_METRIC_FIELDS, clean_robust_metrics

OPERATIONS = ('add', 'drop', 'merge', 'dimension', 'replace', 'combine', 'learning_rate',
              'features', 'scale', 'width', 'depth', 'write_program')


@dataclass(frozen=True)
class ResearchProposal:
    protocol_hash: str
    parent_hash: str
    evidence_cutoff: str
    operation: str
    hypothesis: str
    falsification: str
    source: str
    arguments: dict = field(default_factory=dict)

    def validate(self, spec: ExperimentSpec, parent: RepresentationSpec) -> None:
        if self.protocol_hash != spec.digest or self.parent_hash != parent.digest:
            raise ValueError('Proposal changed protocol or parent identity')
        if utc(self.evidence_cutoff) > utc(spec.dev_end):
            raise ValueError('Proposal contains future evidence')
        if self.operation not in OPERATIONS:
            raise ValueError('Unregistered structural operation')
        if any(not isinstance(x, str) or not x.strip() for x in
               (self.hypothesis, self.falsification, self.source)):
            raise ValueError('A proposal requires source, hypothesis and falsification')
        if not isinstance(self.arguments, dict):
            raise ValueError('Arguments must be finite JSON')  # noqa: TRY004 - proposal rejection API
        json.dumps(self.arguments, allow_nan=False)


def mutate(parent: RepresentationSpec, training: TrainingSpec, proposal: ResearchProposal,
           spec: ExperimentSpec) -> tuple[RepresentationSpec, TrainingSpec]:
    proposal.validate(spec, parent)
    args = proposal.arguments
    allowed = {'add': {'module'}, 'drop': {'name'}, 'merge': {'names', 'name'},
               'dimension': {'name', 'dim'}, 'replace': {'name', 'kind'},
               'combine': {'combination'}, 'learning_rate': {'value'},
               'features': {'name', 'features'}, 'scale': {'name', 'scale'},
               'width': {'name', 'width'}, 'depth': {'name', 'depth'},
               'write_program': {'name', 'program'}}[proposal.operation]
    if set(args) != allowed:
        raise ValueError(f'Operation requires exactly {sorted(allowed)}')
    modules = list(parent.modules)
    op = proposal.operation
    updated_training = training
    if op == 'write_program':
        from .operator_search import written_module
        module = written_module(args, parent, spec.lookback)
        modules = [module if m.name == module.name else m for m in modules]
        if not any(m.name == module.name for m in parent.modules):
            modules.append(module)
    elif op == 'add':
        fields = dict(args['module'])
        fields['features'] = tuple(fields.get('features', ()))
        modules.append(ModuleSpec(**fields))
    elif op in {'drop', 'dimension', 'replace', 'features', 'scale', 'width', 'depth'}:
        matches = [i for i, m in enumerate(modules) if m.name == args['name']]
        if not matches:
            raise ValueError('Unknown representation module')
        index = matches[0]
        if op == 'drop':
            modules.pop(index)
        elif op == 'dimension':
            modules[index] = replace(modules[index], dim=args['dim'])
        elif op == 'replace':
            modules[index] = replace(modules[index], kind=args['kind'], program=None)
        elif op == 'features':
            if not isinstance(args['features'], list):
                raise ValueError('Feature subset must be a JSON list')
            modules[index] = replace(modules[index], features=tuple(args['features']))
        else:
            modules[index] = replace(modules[index], **{op: args[op]})
    elif op == 'merge':
        if not isinstance(args['names'], list) or len(set(args['names'])) < 2:
            raise ValueError('Merge requires at least two unique modules')
        selected = [m for m in modules if m.name in args['names']]
        if len(selected) != len(args['names']):
            raise ValueError('Unknown merge member')
        features = (() if any(not m.features for m in selected)
                    else tuple(sorted({f for m in selected for f in m.features})))
        merged = ModuleSpec(args['name'], 'mlp', max(m.dim for m in selected),
                            max(m.width for m in selected), max(m.scale for m in selected),
                            features)
        modules = [m for m in modules if m.name not in args['names']] + [merged]
    elif op == 'learning_rate':
        value = args['value']
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 1e-5 <= value <= 0.01:
            raise ValueError('Learning rate outside frozen operation bounds')
        updated_training = replace(training, learning_rate=value)
    child = RepresentationSpec(tuple(modules), args['combination'] if op == 'combine'
                                else parent.combination)
    if any(m.scale > spec.lookback for m in child.modules):
        raise ValueError('Scale exceeds observable lookback')
    if child == parent and updated_training == training:
        raise ValueError('No-op proposals do not count as new hypotheses')
    return child, updated_training


def action_menu(parent: RepresentationSpec, spec: ExperimentSpec,
                training: TrainingSpec | None = None,
                features: tuple[str, ...] = ()) -> list[dict]:
    """Identical public search menu for random, diagnostic and LLM proposers."""
    training = training or spec.training
    actions = []
    for module in parent.modules:
        for dim in (4, 8, 16):
            if module.dim != dim and parent.combination == 'concat':
                actions.append({'operation': 'dimension', 'arguments': {'name': module.name, 'dim': dim}})
        for kind in ('summary', 'temporal', 'mlp'):
            if kind != module.kind:
                actions.append({'operation': 'replace', 'arguments': {'name': module.name, 'kind': kind}})
        for operation, choices in (('scale', sorted({1, min(6, spec.lookback),
                                                     min(12, spec.lookback), spec.lookback})),
                                    ('width', (8, 16, 32)), ('depth', (1, 2, 3))):
            for value in choices:
                if module.kind == 'program' and operation == 'scale':
                    from .operator_program import MAX_LOOKBACK, Program
                    if not Program.from_dict(module.program).required_lookback <= value <= MAX_LOOKBACK:
                        continue
                if getattr(module, operation) != value:
                    actions.append({'operation': operation, 'arguments': {
                        'name': module.name, operation: value}})
        subsets = [()] + [(name,) for name in features]
        if len(features) > 2:
            subsets += [tuple(features[::2]), tuple(features[1::2])]
        for subset in dict.fromkeys(subsets):
            if module.kind == 'program':
                continue  # Changing program inputs requires a new, consistent program.
            if tuple(module.features or features) != tuple(subset or features):
                actions.append({'operation': 'features', 'arguments': {
                    'name': module.name, 'features': list(subset)}})
        if len(parent.modules) > 1:
            actions.append({'operation': 'drop', 'arguments': {'name': module.name}})
    if len(parent.modules) < 6:
        new_name = next(f'module_{i}' for i in range(100)
                        if f'module_{i}' not in {m.name for m in parent.modules})
        for kind in ('temporal', 'mlp'):
            module = ModuleSpec(new_name, kind, parent.modules[0].dim,
                                scale=min(12, spec.lookback))
            actions.append({'operation': 'add', 'arguments': {'module': asdict(module)}})
    if len(parent.modules) > 1:
        for pair in combinations(parent.modules, 2):
            name = 'merged_' + fingerprint([parent.digest, [m.name for m in pair]])[:8]
            actions.append({'operation': 'merge', 'arguments': {
                'names': [m.name for m in pair], 'name': name}})
        if parent.combination == 'gated' or len({m.dim for m in parent.modules}) == 1:
            actions.append({'operation': 'combine', 'arguments': {
                'combination': 'concat' if parent.combination == 'gated' else 'gated'}})
    for value in (0.0001, 0.0003, 0.001, 0.003):
        if value != training.learning_rate:
            actions.append({'operation': 'learning_rate', 'arguments': {'value': value}})
    return json.loads(json.dumps(actions))


def _clean_history(history: list[dict], spec: ExperimentSpec) -> list[dict]:
    """Clean an oldest-first history without changing append order or nesting."""
    clean = []
    for row in history:
        if row.get('partition') != 'development' or utc(row['evidence_cutoff']) > utc(spec.dev_end):
            raise ValueError('Agent memory must contain only mature development evidence')
        if any(key in row for key in ('confirmation', 'confirmation_loss', 'test', 'heldout')):
            raise ValueError('Confirmation cannot enter research memory')
        if row.get('protocol_hash', spec.digest) != spec.digest:
            raise ValueError('Memory protocol differs from frozen research protocol')
        result = {key: row[key] for key in ('iteration', 'partition', 'evidence_cutoff', 'status',
                  'primary_loss', 'parameter_count', 'elapsed_seconds', 'candidate_id',
                  'parent_id', 'architecture_hash', 'generation', 'origin', 'error',
                  'protocol_hash', 'source_protocol_hash', 'candidate_path', 'digest',
                  'rejected_initial_reason') if key in row}
        if any(isinstance(value, (dict, list, tuple)) for value in result.values()):
            raise ValueError('Development record summaries must be scalar')
        result.update(clean_robust_metrics(row))
        result['memory_version'] = 'development-action-memory-v1'
        from .operator_search import (
            clean_diagnostics,
            clean_error_type,
            failure_code,
            training_diagnostics,
        )
        if 'diagnostics' in row or 'training' in row:
            result['diagnostics'] = clean_diagnostics(row['diagnostics']) if 'diagnostics' in row else training_diagnostics(row['training'])
        error_type = clean_error_type(row.get('error_type', row.get('error')))
        if error_type:
            result['error_type'] = error_type
        if row.get('error') or row.get('failure_code'):
            result['failure_code'] = failure_code(row.get('failure_code', row.get('error')))
        for key in ('architecture', 'parent_architecture'):
            if key in row:
                result[key] = asdict(RepresentationSpec.from_dict(row[key]))
        for key in ('settings', 'parent_settings'):
            if key in row:
                result[key] = asdict(TrainingSpec(**row[key]))
        if 'proposal' in row:
            proposal = ResearchProposal(**row['proposal'])
            if (proposal.protocol_hash != spec.digest or utc(proposal.evidence_cutoff) > utc(spec.dev_end)
                or proposal.operation not in OPERATIONS):
                raise ValueError('Proposal memory contains invalid protocol or future evidence')
            if any(not isinstance(value, str) or not value.strip() for value in (
                proposal.hypothesis, proposal.falsification, proposal.source, proposal.parent_hash)):
                raise ValueError('Invalid proposal memory strings')
            # Validate action fields and bounds through the exact mutation path.
            # With a recorded parent the result is also checked structurally.
            parent_value = row.get('parent_architecture')
            _validate_memory_arguments(proposal.operation, proposal.arguments, spec)
            if parent_value is not None:
                parent = RepresentationSpec.from_dict(parent_value)
                proposal.validate(spec, parent)
            result['proposal'] = asdict(proposal)
        metrics = row.get('metrics', {})
        if set(metrics) - {task.name for task in spec.tasks} - ROBUST_METRIC_FIELDS:
            raise ValueError('Memory metrics contain undeclared tasks')
        result['metrics'] = clean_robust_metrics(metrics)
        for name, values in metrics.items():
            if name in ROBUST_METRIC_FIELDS:
                continue
            # Share the memory/worker whitelist so scope is not lost in transit.
            from .research_memory import clean_metrics
            result['metrics'][name] = clean_metrics(
                {key: values[key] for key in sorted(TASK_METRIC_FIELDS) if key in values})
            if any(isinstance(value, (dict, list)) for value in result['metrics'][name].values()):
                raise ValueError('Development metric summaries must be scalar')
        json.dumps(result, allow_nan=False)
        clean.append(result)
    return clean


def _validate_memory_arguments(operation, arguments, spec):
    """Validate older history records that did not persist parent structures."""
    keys = {'add': {'module'}, 'drop': {'name'}, 'merge': {'names', 'name'},
            'dimension': {'name', 'dim'}, 'replace': {'name', 'kind'},
            'combine': {'combination'}, 'learning_rate': {'value'},
            'features': {'name', 'features'}, 'scale': {'name', 'scale'},
            'width': {'name', 'width'}, 'depth': {'name', 'depth'},
            'write_program': {'name', 'program'}}[operation]
    if not isinstance(arguments, dict) or set(arguments) != keys:
        raise ValueError('Unregistered proposal memory arguments')
    if operation == 'write_program':
        from .operator_program import Program
        program = Program.from_dict(arguments['program'])
        module = ModuleSpec(name=arguments['name'], kind='program', program=program.to_dict(),
                           features=tuple(program.to_dict()['feature_names']), scale=spec.lookback)
        if program.required_lookback > spec.lookback:
            raise ValueError('Memory program exceeds lookback')
    elif operation == 'add':
        value = dict(arguments['module'])
        value['features'] = tuple(value.get('features', ()))
        module = ModuleSpec(**value)
    elif operation == 'merge':
        names = arguments['names']
        if (not isinstance(names, list) or not 2 <= len(names) <= 6
            or len(set(names)) != len(names)):
            raise ValueError('Invalid merge memory')
        for name in names:
            ModuleSpec(name=name)
        module = ModuleSpec(name=arguments['name'])
    elif operation in {'combine', 'learning_rate'}:
        if operation == 'combine' and arguments['combination'] not in {'concat', 'gated'}:
            raise ValueError('Invalid combination memory')
        if operation == 'learning_rate' and (type(arguments['value']) not in (int, float)
                                            or not 1e-5 <= arguments['value'] <= 0.01):
            raise ValueError('Invalid learning rate memory')
        return
    else:
        value = dict(arguments)
        if operation == 'features':
            if not isinstance(value['features'], list):
                raise ValueError('Invalid feature memory')
            value['features'] = tuple(value['features'])
        module = ModuleSpec(**value)
    if module.scale > spec.lookback and operation in {'scale', 'add'}:
        raise ValueError('Memory scale exceeds lookback')


def propose(parent: RepresentationSpec, spec: ExperimentSpec, history: list[dict], iteration: int,
            mode: str = 'random', exchange: str | Path | None = None,
            timeout: float = 300, training: TrainingSpec | None = None,
            features: tuple[str, ...] = (), program_search: bool = False) -> ResearchProposal:
    history = _clean_history(history, spec)
    menu = action_menu(parent, spec, training, features)
    context = {'protocol_hash': spec.digest, 'parent_hash': parent.digest,
               'parent': asdict(parent), 'evidence_cutoff': spec.dev_end,
               'task': asdict(next(t for t in spec.tasks if t.name == spec.primary_task)),
               'actions': menu, 'history': history[-12:] if program_search else history, 'iteration': iteration,
               'response_schema': {key: 'string' for key in (
                   'protocol_hash', 'parent_hash', 'evidence_cutoff', 'operation',
                   'hypothesis', 'falsification', 'source')},
               'instruction': 'Choose one action from actions and copy its arguments. '
                   'Return a ResearchProposal JSON with testable hypothesis and falsification. '
                   'Use only this development packet. Do not inspect confirmation data.'}
    context['response_schema']['arguments'] = 'object copied from the selected action'
    if program_search:
        from .operator_search import program_contract
        context['operator_contract'] = program_contract(parent, spec.lookback, features)
        context['instruction'] = 'Write a new or revised indicator program and review measured development feedback.'
        while len(context['history']) > 1 and len(json.dumps(context).encode()) > 90000:
            context['history'].pop(0)
        if len(json.dumps(context).encode()) > 90000:
            raise ValueError('Program request exceeds byte budget with newest history retained')
    if mode == 'external':
        if exchange is None:
            raise ValueError('External agent requires an exchange directory')
        directory = project_path(exchange)
        request = {**context, 'request_hash': fingerprint(context)}
        immutable_json(directory / f'request-{iteration}.json', request)
        response = directory / f'proposal-{iteration}.json'
        deadline = time.monotonic() + timeout
        while not response.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError('External agent did not provide a proposal in the allotted time')
            time.sleep(0.25)
        payload = json.loads(response.read_text('utf-8-sig'))
        proposal = ResearchProposal(**payload)
    else:
        if mode not in {'random', 'diagnostic'}:
            raise ValueError('Unknown research proposer')
        rng = random.Random(spec.seed + iteration)
        if program_search:
            from .operator_search import program_decision
            decision = program_decision(context)
            selected = {'operation': 'write_program', 'arguments': {
                'name': decision['target_name'], 'program': json.loads(decision['program_json'])}}
        elif mode == 'diagnostic' and history and history[-1].get('status') == 'failed':
            selected = next((a for a in menu if a['operation'] == 'learning_rate'), menu[0])
        elif mode == 'diagnostic' and iteration % 2 == 0:
            selected = next((a for a in menu if a['operation'] == 'drop'), menu[0])
        else:
            selected = rng.choice(menu)
        proposal = ResearchProposal(spec.digest, parent.digest, spec.dev_end,
            selected['operation'], 'This bounded change may improve the declared task under the same information.',
            'Reject the hypothesis if development improvement does not survive frozen confirmation.',
            'random_search' if mode == 'random' else 'local_diagnostic_rules', selected['arguments'])
    proposal.validate(spec, parent)
    if (program_search and context['operator_contract']['require_program']
            and proposal.operation != 'write_program'):
        raise ValueError('The first indicator proposal must write a program')
    if program_search and proposal.operation == 'write_program':
        from .operator_search import validate_program_proposal
        validate_program_proposal(proposal, parent, context['operator_contract'])
    elif {'operation': proposal.operation, 'arguments': proposal.arguments} not in json.loads(json.dumps(menu)):
        raise ValueError('Proposal is not in the shared frozen action menu')
    return proposal
