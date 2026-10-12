"""Frozen development-only, cold-fit chronological selection policy.

The folds are forward experiments, not slices scored by a model fitted later.
Each architecture and declared seed is fitted afresh at each fold origin.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, replace
from itertools import pairwise

import numpy as np
import pandas as pd

from factorlab.protocol import fingerprint, utc

ROBUST_METRIC_FIELDS = frozenset({'robustness_rank', 'mean_relative_gain', 'relative_gain_std',
                                 'winning_fold_fraction', 'worst_primary_relative_degradation',
                                 'robust_eligible', 'selection_reason'})
TASK_METRIC_FIELDS = frozenset({'status', 'kind', 'predictive', 'interpretation',
                                'support', 'eligible_support', 'total_rows', 'n_hours',
                                'fold_count', 'loss', 'coverage', 'label_coverage', 'brier',
                                'aggregation', 'coverage_scope', 'support_scope'})
TASK_METRIC_STRINGS = {
    'kind': frozenset({'volatility', 'return', 'first_event', 'state'}),
    'interpretation': frozenset({'future_target', 'descriptive_only'}),
    'aggregation': frozenset({'equal_symbol_within_hour_then_equal_hour',
                             'equal_fold_of_equal_symbol_within_hour_then_equal_hour'}),
    'coverage_scope': frozenset({'provided_dataset_observed_task_labels',
                                'fold_rows_including_repeats'}),
    'support_scope': frozenset({'fold_observations_including_repeats'}),
}
SELECTION_REASONS = frozenset({'passed', 'reference_root', 'incomplete_folds', 'unsupported_primary',
                               'unsupported_predictive_task', 'incomparable_support_or_aggregation',
                               'auxiliary_degradation', 'minimum_relative_improvement_not_met',
                               'winning_fold_fraction_not_met', 'primary_fold_degradation',
                               'stability_penalty_not_met', 'fit_failed', 'final_refit_failed'})


def clean_robust_metrics(values):
    """Extract only finite numeric, boolean and enumerated robustness facts."""
    result = {}
    for name in ROBUST_METRIC_FIELDS & values.keys():
        value = values[name]
        if name == 'robust_eligible':
            if type(value) is not bool:
                raise ValueError('robust_eligible must be a boolean')
        elif name == 'selection_reason':
            if not isinstance(value, str) or value not in SELECTION_REASONS:
                raise ValueError('selection_reason must be a bounded reason code')
        elif value is not None:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('Robustness summaries must be finite numbers or None')
            if name in {'relative_gain_std', 'worst_primary_relative_degradation'} and value < 0:
                raise ValueError('Robustness dispersion/degradation must be nonnegative')
            if name == 'winning_fold_fraction' and not 0 <= value <= 1:
                raise ValueError('Winning fold fraction must be in [0, 1]')
        result[name] = value
    return result


@dataclass(frozen=True)
class RobustSelectionSpec:
    mode: str = 'legacy'
    folds: int = 3
    min_train_hours: int = 168
    min_validation_hours: int = 48
    stability_penalty: float = 1.0
    min_winning_fold_fraction: float = 2 / 3
    max_primary_fold_degradation: float = .10
    auxiliary_max_relative_degradation: float = .10
    minimum_relative_improvement: float = 0.0

    def __post_init__(self):
        if self.mode not in {'legacy', 'walk_forward'}:
            raise ValueError('Unknown robust selection mode')
        for name in ('folds', 'min_train_hours', 'min_validation_hours'):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 1_000_000:
                raise ValueError(f'{name} must be a positive integer')
        if self.folds < 2 or self.folds > 20:
            raise ValueError('Walk-forward selection requires 2..20 folds')
        for name in ('stability_penalty', 'max_primary_fold_degradation',
                     'auxiliary_max_relative_degradation', 'minimum_relative_improvement'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        value = self.min_winning_fold_fraction
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError('Winning fold fraction must be in (0, 1]')


def _join_visible(train, development):
    from .contracts import TaskDataset
    data = TaskDataset(
        np.concatenate((train.x, development.x)),
        {name: np.concatenate((train.targets[name], development.targets[name])) for name in train.targets},
        {name: np.concatenate((train.observed[name], development.observed[name])) for name in train.targets},
        {name: np.concatenate((train.maturity[name], development.maturity[name])) for name in train.targets},
        pd.concat((train.meta, development.meta), ignore_index=True), train.features, train.tasks,
    )
    order = data.meta.sort_values(['available_at', 'symbol'], kind='stable').index.to_numpy()
    return data.subset(order)


def walk_forward_parts(train, development, spec):
    """First half initializes history; the last half has equal calendar folds."""
    from .replay import _training_parts
    policy = spec.selection
    data = _join_visible(train, development)
    if not set(data.meta.symbol) <= set(spec.symbols):
        raise ValueError('Walk-forward data contains undeclared symbols')
    start, end = utc(spec.start), utc(spec.dev_end)
    halfway = start + (end - start) / 2
    boundaries = [halfway + (end - halfway) * index / policy.folds
                  for index in range(policy.folds + 1)]
    parts = []
    for index, (origin, stop) in enumerate(pairwise(boundaries)):
        # _training_parts purges horizons at the inner 80/20 boundary and
        # includes only fully mature history at this outer fold's origin.
        fitting, early, details = _training_parts(data, origin)
        validation = data.subset(((data.meta.available_at >= origin)
                                  & (data.meta.available_at < stop)).to_numpy())
        mature_hours = pd.concat((fitting.meta.available_at, early.meta.available_at)).dt.floor('h').nunique()
        if mature_hours < policy.min_train_hours:
            raise ValueError('Walk-forward minimum training hours not met')
        if validation.meta.available_at.dt.floor('h').nunique() < policy.min_validation_hours:
            raise ValueError('Walk-forward minimum validation hours not met')
        for task in validation.tasks:
            mask = validation.task_mask(task.name, stop)
            validation.observed[task.name] = mask
            validation.targets[task.name] = validation.targets[task.name].astype(float, copy=True)
            validation.targets[task.name][~mask] = np.nan
        details = {**details, 'fold': index, 'validation_start': origin.isoformat(),
                   'validation_end_exclusive': stop.isoformat(),
                   'validation_rows': len(validation.meta),
                   'validation_support': {t.name: int(validation.task_mask(t.name).sum())
                                          for t in validation.tasks}}
        parts.append((fitting, early, validation, details))
    return parts


def macro_metrics(data, predictions):
    """Each symbol has equal weight within each hour, then hours are equal."""
    from .scoring import evaluate, per_sample_loss
    metrics = evaluate(data, predictions)  # Validate even predictions for masked rows.
    for task in data.tasks:
        row = metrics[task.name]
        if row['status'] != 'ok':
            continue
        mask = data.task_mask(task.name)
        values = np.asarray(predictions[task.name])
        losses = per_sample_loss(task, data.targets[task.name][mask], values[mask])
        frame = data.meta.loc[mask, ['symbol', 'available_at']].copy()
        frame['hour'] = pd.to_datetime(frame.available_at, utc=True).dt.floor('h')
        frame['loss'] = losses
        hours = frame.groupby(['hour', 'symbol'], observed=True).loss.mean().groupby(level='hour').mean()
        row.update(loss=float(hours.mean()), n_hours=len(hours),
                   aggregation='equal_symbol_within_hour_then_equal_hour')
    return metrics


def summarize_fold_metrics(scores, tasks):
    """Summarize fold occurrences, preserving the equal-fold selection loss.

    Counts include repeated rows/hours if folds overlap; they are not distinct
    evidence counts. Probability diagnostics stay in the original fold records:
    a pooled Brier/ECE is not supplied without pooled predictions and a defined
    weighting contract. Missing optional-task evidence cannot inherit fold 0's
    successful status.
    """
    if not scores:
        raise ValueError('A fold summary requires at least one fold')
    result = {}
    for task in tasks:
        rows = [score[task.name] for score in scores]
        complete = all(row['status'] == 'ok' for row in rows)
        summary = {'kind': task.kind, 'predictive': task.predictive,
                   'interpretation': 'future_target' if task.predictive else 'descriptive_only',
                   'status': 'ok' if complete else 'insufficient_evidence',
                   'fold_count': len(rows),
                   'loss': float(np.mean([row['loss'] for row in rows])) if complete else None,
                   'aggregation': 'equal_fold_of_equal_symbol_within_hour_then_equal_hour',
                   'support_scope': 'fold_observations_including_repeats',
                   'coverage_scope': 'fold_rows_including_repeats'}
        for field in ('support', 'eligible_support', 'total_rows', 'n_hours'):
            if all(field in row for row in rows):
                summary[field] = sum(row[field] for row in rows)
        if 'total_rows' in summary:
            total = summary['total_rows']
            for coverage, count in (('coverage', 'support'), ('label_coverage', 'eligible_support')):
                if count in summary:
                    summary[coverage] = summary[count] / total if total else 0.0
        result[task.name] = summary
    return result


def compare_folds(candidate, reference, tasks, primary_task, policy):
    """Fixed-root relative gains and conservative per-fold auxiliary gates."""
    if len(candidate) != policy.folds or len(reference) != policy.folds:
        return {'eligible': False, 'reasons': ['incomplete_folds']}
    reasons, gains, auxiliary = [], [], {}
    epsilon = 1e-12
    for task in tasks:
        if not task.predictive:
            continue
        task_gains = []
        for index, (current, baseline) in enumerate(zip(candidate, reference)):
            c, r = current.get(task.name, {}), baseline.get(task.name, {})
            if (c.get('status') != 'ok' or r.get('status') != 'ok'
                    or not c.get('support') or not r.get('support')
                    or c.get('loss') is None or r.get('loss') is None
                    or not np.isfinite([c['loss'], r['loss']]).all()):
                reasons.append(f'unsupported:{task.name}:fold:{index}')
                continue
            if (c['loss'] < 0 or r['loss'] < 0 or c['support'] != r['support']
                    or not c.get('n_hours') or c.get('n_hours') != r.get('n_hours')
                    or c.get('aggregation') != 'equal_symbol_within_hour_then_equal_hour'
                    or c.get('aggregation') != r.get('aggregation')):
                reasons.append(f'incomparable_support_or_aggregation:{task.name}:fold:{index}')
                continue
            gain = float((r['loss'] - c['loss']) / max(abs(r['loss']), epsilon))
            if not math.isfinite(gain):
                reasons.append(f'unsupported:{task.name}:fold:{index}')
                continue
            task_gains.append(gain)
        if task.name == primary_task:
            gains = task_gains
        else:
            auxiliary[task.name] = task_gains
            if len(task_gains) == policy.folds and min(task_gains) < -policy.auxiliary_max_relative_degradation - epsilon:
                reasons.append(f'auxiliary_degradation:{task.name}')
    if len(gains) != policy.folds:
        return {'eligible': False, 'reasons': reasons or ['unsupported_primary'],
                'auxiliary_relative_gains': auxiliary}
    scale = max(1.0, max(abs(value) for value in gains))
    normalized = np.asarray(gains) / scale
    mean, std = float(np.mean(normalized) * scale), float(np.std(normalized) * scale)
    win_fraction = float(np.mean(np.asarray(gains) > 0))
    worst_degradation = float(max(0, -min(gains)))
    if mean <= policy.minimum_relative_improvement:
        reasons.append('minimum_relative_improvement_not_met')
    if win_fraction < policy.min_winning_fold_fraction:
        reasons.append('winning_fold_fraction_not_met')
    if worst_degradation > policy.max_primary_fold_degradation + epsilon:
        reasons.append('primary_fold_degradation')
    rank = mean - policy.stability_penalty * std
    if not math.isfinite(rank):
        return {'eligible': False, 'reasons': ['unsupported_primary'],
                'auxiliary_relative_gains': auxiliary}
    if rank <= 0:
        reasons.append('stability_penalty_not_met')
    return {'eligible': not reasons, 'reasons': reasons, 'relative_gains': gains,
            'mean_relative_gain': mean, 'relative_gain_std': std,
            'rank_score': rank,
            'winning_fold_fraction': win_fraction,
            'worst_primary_relative_degradation': worst_degradation,
            'auxiliary_relative_gains': auxiliary, 'relative_loss_epsilon': epsilon}


def _validate_fit(model):
    for member in getattr(model, 'models', [model]):
        diagnostic = member.diagnostics
        if (diagnostic.get('stop_reason') == 'time_budget'
                or diagnostic.get('partial_epoch')
                or any(row.get('partial_epoch') for row in diagnostic.get('history', []))
                or diagnostic.get('resource_failure')):
            raise TimeoutError('Resource-truncated fit cannot enter robust selection')


def search_walk_forward_population(train, development, parents, spec, *, mode='random', device='cpu',
                                   exchange=None, history=None, initial_models=None, inherit_weights=False,
                                   population_size=4, fit_candidate=None, on_candidate=None, on_record=None,
                                   program_search=False, on_fold=None):
    """Cold forward-fold search; return only a fresh full train/dev refit."""
    from .benchmarks import fit_simple_benchmarks
    from .experiment import ModelGroup
    from .population import PopulationCandidate, _default_fit
    from .replay import _prediction_inputs
    from .search import _clean_history, mutate, propose

    started = time.monotonic()
    fitter = fit_candidate or _default_fit
    parts = walk_forward_parts(train, development, spec)
    policy = spec.selection
    records, archive, roots, seen = [], [], [], set()
    memory = _clean_history(history or [], spec)
    iteration = max((row.get('iteration', -1) for row in memory), default=-1) + 1
    newly_fitted = proposal_attempts = fit_attempts = successful_fits = 0
    seed_count = len(spec.training_seeds)
    # Candidate ceiling includes roots and every candidate's complete fold suite.
    # The extra group is the final cold refit, never a hidden free fit.
    fit_ceiling = (spec.candidates * policy.folds + 1) * seed_count
    per_fit_seconds = min(spec.training.max_seconds, int(spec.search_seconds / fit_ceiling))
    if per_fit_seconds < 1:
        raise ValueError('Search budget cannot cover forward folds and final refit')
    simple_benchmarks = []
    for fitting, _, validation, details in parts:
        began = time.monotonic()
        record = {'fold': details['fold'], 'information_cutoff': details['information_cutoff'],
                  'status': 'failed', 'selection_gate': False,
                  'scope': 'development_control_diagnostics_only'}
        try:
            controls = fit_simple_benchmarks(
                fitting, details['information_cutoff'],
                include_rolling='return_1h' in fitting.features and fitting.x.shape[1] >= 24)
            inputs = _prediction_inputs(validation)
            record.update(status='ok', controls={name: {'model': control.to_dict(),
                          'digest': control.digest, 'metrics': macro_metrics(validation, control.predict(inputs))}
                          for name, control in controls.items()})
        except (ValueError, TypeError, KeyError, FloatingPointError, TimeoutError, RuntimeError) as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
        record['elapsed_seconds'] = time.monotonic() - began
        simple_benchmarks.append(record)

    def remaining():
        return spec.search_seconds - (time.monotonic() - started)

    def persist(record):
        records.append(record)
        memory.extend(_clean_history([record], spec))
        if on_record is not None:
            on_record(record)

    def cold_fit(fitting, early, architecture, settings, log, *, reserve_final=True):
        nonlocal fit_attempts, successful_fits
        bounded = replace(settings, max_seconds=min(settings.max_seconds, per_fit_seconds))
        members = []
        for seed in spec.training_seeds:
            reserved = per_fit_seconds * seed_count if reserve_final else 0
            if fit_attempts >= fit_ceiling or remaining() < bounded.max_seconds + reserved:
                raise TimeoutError('Search budget exhausted with final refit reserved')
            fit_attempts += 1
            attempt = {'seed': seed, 'settings': asdict(replace(bounded, seed=seed)), 'status': 'failed'}
            log.append(attempt)
            try:
                group = fitter(fitting, early, architecture, bounded, (seed,), device, initial_model=None)
                fitted = getattr(group, 'models', [group])
                attempt['diagnostics'] = [member.diagnostics for member in fitted]
                attempt['fit_metadata'] = [member.metadata for member in fitted]
                if group.schema != fitting.schema:
                    raise ValueError('Cold fitter changed the frozen input/task schema')
                if len(fitted) != 1 or fitted[0].metadata['settings']['seed'] != seed:
                    raise ValueError('Cold fitter must return exactly the declared seed')
                _validate_fit(group)
                if remaining() < 0:
                    raise TimeoutError('Fit exceeded total search wall budget')
                attempt.update(status='ok', digest=group.digest,
                               diagnostics=[member.diagnostics for member in fitted])
                members.extend(fitted)
                successful_fits += 1
            except (ValueError, TypeError, KeyError, FloatingPointError, TimeoutError, RuntimeError) as exc:
                attempt['error'] = f'{type(exc).__name__}: {exc}'
                raise
        return ModelGroup(members)

    def active_population():
        active = list(roots)
        structures = {row.architecture.digest for row in active}
        # primary_loss is negative robust rank, so smaller is better here.
        for row in sorted(archive, key=lambda c: (c.primary_loss, c.candidate_id)):
            if len(active) >= population_size:
                break
            if row.architecture.digest not in structures:
                active.append(row)
                structures.add(row.architecture.digest)
        return active

    def evaluate_candidate(architecture, settings, parent=None, proposal=None):
        nonlocal iteration, newly_fitted
        began = time.monotonic()
        identity = fingerprint({'architecture_hash': architecture.digest, 'settings': asdict(settings)})
        record = {'iteration': iteration, 'partition': 'development', 'protocol_hash': spec.digest,
                  'evidence_cutoff': spec.dev_end, 'status': 'failed', 'candidate_id': identity,
                  'parent_id': parent.candidate_id if parent else None,
                  'generation': parent.generation + 1 if parent else 0,
                  'origin': 'mutation' if parent else 'root', 'architecture': asdict(architecture),
                  'architecture_hash': architecture.digest, 'settings': asdict(settings),
                  'reused': False, 'folds': [], 'fit_attempts': [],
                  'selection_mode': 'walk_forward', 'weights_inherited': False}
        iteration += 1
        if proposal is not None:
            record.update(proposal=asdict(proposal), parent_architecture=asdict(parent.architecture),
                          parent_settings=asdict(parent.settings))
        try:
            if identity in seen:
                raise ValueError('Duplicate candidate')
            seen.add(identity)
            if newly_fitted >= spec.candidates:
                raise TimeoutError('Candidate suite budget exhausted')
            newly_fitted += 1
            for fitting, early, validation, details in parts:
                fold = {**details, 'status': 'failed', 'fit_attempts': []}
                record['folds'].append(fold)
                try:
                    model = cold_fit(fitting, early, architecture, settings, fold['fit_attempts'])
                    before = model.digest
                    predictions = model.predict(_prediction_inputs(validation))
                    if model.digest != before:
                        raise ValueError('Fold model was mutated during prediction')
                    metrics = macro_metrics(validation, predictions)
                    if any(metrics[t.name]['status'] != 'ok' for t in spec.tasks if t.predictive):
                        raise ValueError('Fold lacks supported predictive task evidence')
                    fold.update(status='ok', metrics=metrics, digest=model.digest,
                                parameter_count=model.parameter_count,
                                fit_metadata=[member.metadata for member in model.models],
                                training=[member.diagnostics for member in model.models])
                    if on_fold is not None:
                        retained = on_fold(model, record, fold, validation)
                        if retained is not None:
                            fold['artifacts'] = retained
                finally:
                    record['fit_attempts'].extend(fold['fit_attempts'])
            scores = [fold['metrics'] for fold in record['folds']]
            is_reference = not archive
            reference_scores = scores if is_reference else [fold['metrics'] for fold in records[0]['folds']]
            comparison = compare_folds(scores, reference_scores, spec.tasks, spec.primary_task, policy)
            reason = comparison['reasons'][0].split(':')[0] if comparison['reasons'] else 'passed'
            if reason == 'unsupported':
                reason = 'unsupported_predictive_task'
            metrics = summarize_fold_metrics(scores, spec.tasks)
            record.update(status='ok', metrics=metrics, primary_loss=metrics[spec.primary_task]['loss'],
                          comparison=comparison, reference=is_reference, parameter_count=model.parameter_count,
                          digest=model.digest, digest_scope='last_forward_fold',
                          training=[member.diagnostics for member in model.models],
                          robust_eligible=comparison['eligible'],
                          robustness_rank=0.0 if is_reference else comparison.get('rank_score'),
                          mean_relative_gain=comparison.get('mean_relative_gain'),
                          relative_gain_std=comparison.get('relative_gain_std'),
                          winning_fold_fraction=comparison.get('winning_fold_fraction'),
                          worst_primary_relative_degradation=comparison.get('worst_primary_relative_degradation'),
                          selection_reason='reference_root' if is_reference else reason)
            rank = 0.0 if is_reference else comparison.get('rank_score', -1e300)
            row = PopulationCandidate(identity, architecture, settings, model, -rank,
                                      record['generation'], record['parent_id'], record['origin'])
            archive.append(row)
            if parent is None:
                roots.append(row)
        except (ValueError, TypeError, KeyError, FloatingPointError, TimeoutError, RuntimeError) as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
        record['elapsed_seconds'] = time.monotonic() - began
        persist(record)
        return record

    reference = evaluate_candidate(parents[0], spec.training)
    if reference['status'] == 'ok':
        for architecture in parents[1:]:
            if newly_fitted >= spec.candidates:
                break
            evaluate_candidate(architecture, spec.training)
        cursor = 0
        while archive and newly_fitted < spec.candidates and remaining() >= per_fit_seconds * seed_count * (policy.folds + 1):
            if proposal_attempts >= max(16, 8 * spec.candidates):
                break
            active = active_population()
            parent = active[cursor % len(active)]
            cursor += 1
            proposal_attempts += 1
            began = time.monotonic()
            try:
                proposal = propose(parent.architecture, spec, memory, iteration, mode, exchange,
                                   timeout=min(300, max(1, remaining() - per_fit_seconds * seed_count)),
                                   training=parent.settings, features=train.features,
                                   **({'program_search': True} if program_search else {}))
                child, settings = mutate(parent.architecture, parent.settings, proposal, spec)
                evaluate_candidate(child, settings, parent, proposal)
            except (ValueError, TypeError, KeyError, FloatingPointError, TimeoutError, RuntimeError) as exc:
                record = {'iteration': iteration, 'partition': 'development', 'protocol_hash': spec.digest,
                          'evidence_cutoff': spec.dev_end, 'status': 'failed', 'parent_id': parent.candidate_id,
                          'parent_architecture': asdict(parent.architecture), 'parent_settings': asdict(parent.settings),
                          'error': f'{type(exc).__name__}: {exc}', 'elapsed_seconds': time.monotonic() - began}
                iteration += 1
                persist(record)
    eligible = {row['candidate_id'] for row in records if row.get('comparison', {}).get('eligible')}
    qualified = [row for row in archive if row.candidate_id in eligible]
    selected = min(qualified, key=lambda row: (row.primary_loss, row.model.parameter_count, row.candidate_id)) if qualified else (archive[0] if archive else None)
    fallback_reason = None if qualified else ('no_candidate_passed_robust_gates' if archive else 'reference_fold_fit_failed')
    final = {'status': 'not_attempted', 'fit_attempts': []}
    best_model = None
    if selected is not None:
        final.update(candidate_id=selected.candidate_id, architecture=asdict(selected.architecture))
        try:
            best_model = cold_fit(train, development, selected.architecture, selected.settings,
                                  final['fit_attempts'], reserve_final=False)
            final.update(status='ok', digest=best_model.digest,
                         training=[member.diagnostics for member in best_model.models])
            selected_record = next(row for row in records if row.get('candidate_id') == selected.candidate_id)
            selected_record['final_refit'] = final
            if on_candidate is not None:
                on_candidate(best_model, selected_record)
        except (ValueError, TypeError, KeyError, FloatingPointError, TimeoutError, RuntimeError) as exc:
            best_model = None
            final.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            fallback_reason = 'final_refit_failed_no_model_returned'
    result = {'mode': mode, 'selection': 'cold_walk_forward_relative_to_fixed_root',
              'selection_policy': asdict(policy), 'records': records, 'attempts': len(records),
              'simple_benchmarks': simple_benchmarks,
              'benchmark_fit_groups': len(simple_benchmarks),
              'newly_fitted': newly_fitted, 'proposal_attempts': proposal_attempts,
              'fit_attempts': fit_attempts, 'successful_fits': successful_fits,
              'budget_seed_fits': fit_ceiling, 'budget_seconds_per_seed_fit': per_fit_seconds,
              'elapsed_seconds': time.monotonic() - started, 'budget_seconds': spec.search_seconds,
              'budget_candidates': spec.candidates, 'population_size': population_size,
              'population_ids': [row.candidate_id for row in active_population()],
              'root_ids': [row.candidate_id for row in roots],
              'archive_ids': [row.candidate_id for row in archive],
              'distinct_architectures': len({row.architecture.digest for row in archive}),
              'reference_candidate_id': reference['candidate_id'],
              'selection_rank': -selected.primary_loss if best_model else None,
              'selected_comparison': next((row.get('comparison') for row in records
                                          if best_model and row.get('candidate_id') == selected.candidate_id), None),
              'best_candidate_id': selected.candidate_id if best_model else None,
              'best_loss': next((row['primary_loss'] for row in records if best_model and row.get('candidate_id') == selected.candidate_id), None),
              'architecture': asdict(selected.architecture) if best_model else None,
              'development_memory': memory, 'fallback_to_root': not qualified and bool(archive),
              'fallback_reason': fallback_reason, 'final_refit': final,
              'monotonic_improvement_claimed': False, 'inherit_weights': False,
              'initial_models_reused': False, 'budget_comparison': 'equal cold seed-fold ceilings including final refit',
              'agent_kind': 'external_proposer' if mode == 'external' else mode,
              'proposal_sources': sorted({row['proposal']['source'] for row in records if 'proposal' in row})}
    return best_model, result
