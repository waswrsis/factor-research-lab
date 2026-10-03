"""Bounded development-only population search with retained ancestors.

All successful candidates remain in the archive. A smaller, structurally
diverse population supplies proposals; roots remain available even when a
child has a lower loss. Search never promises monotonic child improvement.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace

import numpy as np

from factorlab.protocol import fingerprint, utc

from .contracts import RepresentationSpec
from .learning import fit
from .scoring import evaluate
from .search import _clean_history, mutate, propose


@dataclass
class PopulationCandidate:
    candidate_id: str
    architecture: RepresentationSpec
    settings: object
    model: object
    primary_loss: float
    generation: int
    parent_id: str | None
    origin: str


def _default_fit(train, development, architecture, settings, seeds, device, *, initial_model=None):
    # Deferred import allows experiment.search_candidates to use this module.
    from .experiment import ModelGroup
    members = getattr(initial_model, 'models', [initial_model]) if initial_model is not None else []
    by_seed = {model.metadata['settings']['seed']: model for model in members}
    return ModelGroup([fit(train, development, architecture, replace(settings, seed=seed), device,
                           initial_model=by_seed.get(seed),
                           retain_initial_scalers=initial_model is not None) for seed in seeds])


def _members(model):
    return getattr(model, 'models', [model])


def _check_development_inputs(train, development, spec):
    if (train.schema != development.schema or train.tasks != spec.tasks
        or train.x.shape[1] != spec.lookback):
        raise ValueError('Population datasets must match the frozen input/task schema')
    for data, start, end in ((train, spec.start, spec.train_end),
                             (development, spec.train_end, spec.dev_end)):
        if len(data.x) == 0:
            raise ValueError('Population requires nonempty training/development inputs')
        if ((data.meta.available_at < utc(start)) | (data.meta.available_at >= utc(end))).any():
            raise ValueError('Population input contains evidence outside its declared partition')
        for task in data.tasks:
            if (data.maturity[task.name][data.task_mask(task.name)] > utc(end).to_datetime64()).any():
                raise ValueError('Population labels contain immature partition evidence')


def _check_initial(model, architecture, train, spec):
    members = _members(model)
    if (len(members) != len(spec.training_seeds)
        or tuple(m.metadata['settings']['seed'] for m in members) != spec.training_seeds):
        raise ValueError('Initial candidates require the declared seed group')
    for member in members:
        metadata = member.metadata
        if (member.schema != train.schema or metadata['mode'] != 'learned'
            or RepresentationSpec.from_dict(metadata['architecture']).digest != architecture.digest):
            raise ValueError('Initial candidate differs from its declared architecture/schema')
        for key, boundary in (('fit_feature_cutoff', spec.train_end),
                              ('fit_label_cutoff', spec.train_end),
                              ('development_feature_cutoff', spec.dev_end),
                              ('development_label_cutoff', spec.dev_end)):
            if metadata.get(key) is not None and utc(metadata[key]) > utc(boundary):
                raise ValueError('Initial candidate contains future evidence')


def search_population(train, development, parents, spec, *, mode='random', device='cpu',
                      exchange=None, history=None, initial_models=None, inherit_weights=False,
                      population_size=4, fit_candidate=None, on_candidate=None, on_record=None,
                      program_search=False, on_fold=None):
    """Return (best fitted model, JSON-safe search evidence).

    ``candidates`` caps newly fitted candidates, including unfitted roots;
    reused roots cost no fit slot but their scoring consumes the wall budget.
    Callbacks cannot influence proposer context except via typed dev records.
    ``on_candidate(model, record)`` may persist a successful fitted model.
    """
    if isinstance(parents, RepresentationSpec):
        parents = (parents,)
    parents = tuple(parents)
    if (not parents or not all(isinstance(p, RepresentationSpec) for p in parents)
        or len({p.digest for p in parents}) != len(parents)):
        raise ValueError('Population roots must be nonempty and structurally unique')
    if type(population_size) is not int or not len(parents) <= population_size <= 100:
        raise ValueError('Population size must retain all roots and be at most 100')
    if mode not in {'random', 'diagnostic', 'external'}:
        raise ValueError('Unknown research proposer')
    _check_development_inputs(train, development, spec)
    if getattr(getattr(spec, 'selection', None), 'mode', 'legacy') == 'walk_forward':
        from .robust_selection import search_walk_forward_population
        return search_walk_forward_population(
            train, development, parents, spec, mode=mode, device=device,
            exchange=exchange, history=history, initial_models=initial_models,
            inherit_weights=inherit_weights, population_size=population_size,
            fit_candidate=fit_candidate, on_candidate=on_candidate, on_record=on_record,
            program_search=program_search, on_fold=on_fold)
    memory = _clean_history(history or [], spec)
    initial_models = initial_models or {}
    fitter = fit_candidate or _default_fit
    started = time.monotonic()
    records, archive, roots, seen = [], [], [], set()
    newly_fitted = 0
    proposal_attempts = 0
    previous_iterations = [row.get('iteration', -1) for row in memory]
    iteration = max(previous_iterations, default=-1) + 1

    def remaining():
        return spec.search_seconds - (time.monotonic() - started)

    def population():
        active = list(roots)
        structures = {c.architecture.digest for c in active}
        for candidate in sorted(archive, key=lambda c: (c.primary_loss, c.model.parameter_count,
                                                       c.candidate_id)):
            if len(active) >= population_size:
                break
            if candidate.architecture.digest not in structures:
                active.append(candidate)
                structures.add(candidate.architecture.digest)
        return active

    def evaluate_candidate(architecture, settings, parent=None, proposal=None, reusable=None):
        nonlocal iteration, newly_fitted
        began = time.monotonic()
        identity = fingerprint({'architecture_hash': architecture.digest, 'settings': asdict(settings)})
        record = {'iteration': iteration, 'partition': 'development', 'protocol_hash': spec.digest,
                  'evidence_cutoff': spec.dev_end, 'status': 'failed', 'candidate_id': identity,
                  'parent_id': parent.candidate_id if parent else None,
                  'generation': parent.generation + 1 if parent else 0,
                  'origin': 'mutation' if parent else 'root', 'architecture': asdict(architecture),
                  'architecture_hash': architecture.digest, 'settings': asdict(settings)}
        iteration += 1
        if proposal is not None:
            record.update(proposal=asdict(proposal), parent_architecture=asdict(parent.architecture),
                          parent_settings=asdict(parent.settings))
        try:
            if identity in seen:
                raise ValueError('Duplicate candidate')
            seen.add(identity)
            if remaining() < 1:
                raise TimeoutError('Search wall budget exhausted before evaluation')
            if reusable is not None:
                try:
                    _check_initial(reusable, architecture, train, spec)
                except ValueError as exc:
                    record['rejected_initial_reason'] = str(exc)
                    reusable = None
            if reusable is not None:
                model = reusable
                record['reused'] = True
            else:
                if newly_fitted >= spec.candidates or remaining() < len(spec.training_seeds):
                    raise TimeoutError('Candidate fit budget exhausted')
                newly_fitted += 1
                bounded = replace(settings, max_seconds=max(1, min(
                    settings.max_seconds, int(remaining() / len(spec.training_seeds)))))
                model = fitter(train, development, architecture, bounded, spec.training_seeds, device,
                               initial_model=parent.model if parent and inherit_weights else None)
                record.update(reused=False, effective_settings=asdict(bounded))
            metrics = evaluate(development, model.predict(development))
            if metrics[spec.primary_task]['status'] != 'ok':
                raise ValueError('Candidate has no supported primary task')
            loss = float(metrics[spec.primary_task]['loss'])
            if not np.isfinite(loss):
                raise FloatingPointError('Nonfinite primary loss')
            record.update(status='ok', primary_loss=loss, metrics=metrics,
                          parameter_count=model.parameter_count, digest=model.digest,
                          complexity={'modules': len(architecture.modules),
                                      'encoded_dim_per_seed': sum(m.dim for m in architecture.modules),
                                      'parameters': model.parameter_count,
                                      'maximum_depth': max(m.depth for m in architecture.modules)},
                          training=[m.diagnostics for m in _members(model)])
            if on_candidate is not None:
                on_candidate(model, record)
            candidate = PopulationCandidate(identity, architecture, settings, model, loss,
                                            record['generation'], record['parent_id'], record['origin'])
            archive.append(candidate)
            if parent is None:
                roots.append(candidate)
        except (ValueError, TypeError, KeyError, FloatingPointError, TimeoutError, RuntimeError) as exc:
            record['status'] = 'failed'
            record['error'] = f'{type(exc).__name__}: {exc}'
        record['elapsed_seconds'] = time.monotonic() - began
        records.append(record)
        memory.extend(_clean_history([record], spec))
        if on_record is not None:
            on_record(record)

    for architecture in parents:
        if remaining() < 1:
            break
        evaluate_candidate(architecture, spec.training,
                           reusable=initial_models.get(architecture.digest))
    cursor = 0
    # Bound proposal retries separately, since duplicates/rejections cost no fit slot.
    while archive and newly_fitted < spec.candidates and remaining() >= len(spec.training_seeds):
        if proposal_attempts >= max(16, 8 * spec.candidates):
            break
        active = population()
        parent = active[cursor % len(active)]
        cursor += 1
        proposal_attempts += 1
        began = time.monotonic()
        try:
            proposal = propose(parent.architecture, spec, memory, iteration, mode, exchange,
                               timeout=min(300, remaining()), training=parent.settings,
                               features=train.features, **({'program_search': True} if program_search else {}))
            child, settings = mutate(parent.architecture, parent.settings, proposal, spec)
            evaluate_candidate(child, settings, parent, proposal)
        except (ValueError, TypeError, KeyError, FloatingPointError, TimeoutError, RuntimeError) as exc:
            record = {'iteration': iteration, 'partition': 'development',
                      'protocol_hash': spec.digest, 'evidence_cutoff': spec.dev_end,
                      'status': 'failed', 'parent_id': parent.candidate_id,
                      'parent_architecture': asdict(parent.architecture),
                      'parent_settings': asdict(parent.settings),
                      'error': f'{type(exc).__name__}: {exc}',
                      'elapsed_seconds': time.monotonic() - began}
            iteration += 1
            records.append(record)
            memory.extend(_clean_history([record], spec))
            if on_record is not None:
                on_record(record)
    best = min(archive, key=lambda c: (c.primary_loss, c.model.parameter_count, c.candidate_id)) if archive else None
    result = {'mode': mode, 'records': records, 'attempts': len(records),
              'newly_fitted': newly_fitted, 'proposal_attempts': proposal_attempts,
              'elapsed_seconds': time.monotonic() - started, 'budget_seconds': spec.search_seconds,
              'budget_candidates': spec.candidates, 'population_size': population_size,
              'population_ids': [c.candidate_id for c in population()],
              'root_ids': [c.candidate_id for c in roots],
              'archive_ids': [c.candidate_id for c in archive],
              'distinct_architectures': len({c.architecture.digest for c in archive}),
              'best_loss': best.primary_loss if best else None,
              'best_candidate_id': best.candidate_id if best else None,
              'architecture': asdict(best.architecture) if best else None,
              'development_memory': memory,
              'selection': 'minimum development primary loss across retained fitted archive',
              'monotonic_improvement_claimed': False,
              'inherit_weights': inherit_weights,
              'budget_comparison': 'equal ceilings and seed count, not equal FLOPs or exact usage',
              'agent_kind': 'external_proposer' if mode == 'external' else mode,
              'proposal_sources': sorted({row['proposal']['source'] for row in records if 'proposal' in row})}
    return (best.model if best else None), result
