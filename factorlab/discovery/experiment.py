"""Research orchestration with shared budgets, frozen probes and one-use confirmation."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, replace
from functools import wraps

import numpy as np
import pandas as pd

from factorlab.protocol import fingerprint, project_path, utc

from .contracts import ExperimentSpec, ModuleSpec, RepresentationSpec, TaskDataset, partition
from .evidence import (
    EvidenceRegistry,
    ExperimentJournal,
    file_hash,
    freeze_claim,
    immutable_json,
    snapshot_source,
    verify_bundle,
)
from .learning import FittedProbe, FittedResearchModel, fit, fit_probe, summary_features
from .population import search_population
from .scoring import ProbabilityCalibrator, adaptive_horizons, evaluate, paired_comparison


class ModelGroup:
    """All declared seeds contribute equally; none is selected after seeing results."""
    def __init__(self, models: list[FittedResearchModel]):
        if not models or len({m.schema for m in models}) != 1:
            raise ValueError('Compatible fitted seed models are required')
        self.models = models
        self.schema = models[0].schema

    @property
    def digest(self):
        return fingerprint([m.digest for m in self.models])

    @property
    def parameter_count(self):
        return sum(m.parameter_count for m in self.models)

    def predict(self, data):
        values = [model.predict(data) for model in self.models]
        if any(set(v) != set(values[0]) for v in values):
            raise ValueError('Seed models have different supervision support')
        return {k: np.mean([v[k] for v in values], axis=0) for k in values[0]}

    def encode(self, data):
        return np.concatenate([m.encode(data) for m in self.models], axis=1)

    def save(self, directory):
        target = project_path(directory)
        target.mkdir(parents=True, exist_ok=True)
        for i, model in enumerate(self.models):
            model.save(target / f'seed-{i}')
        immutable_json(target / 'group.json', {'members': len(self.models), 'digest': self.digest,
                                               'schema': self.schema})

    @classmethod
    def load(cls, directory):
        target = project_path(directory)
        record = json.loads((target / 'group.json').read_text('utf-8'))
        result = cls([FittedResearchModel.load(target / f'seed-{i}')
                      for i in range(record['members'])])
        if result.digest != record['digest'] or result.schema != record['schema']:
            raise ValueError('Seed group integrity failure')
        return result


def fit_group(train, development, architecture, settings, seeds, device='cpu'):
    return ModelGroup([fit(train, development, architecture, replace(settings, seed=seed), device)
                       for seed in seeds])


def _score(group, data, primary):
    metrics = evaluate(data, group.predict(data))
    if metrics[primary]['status'] != 'ok':
        raise ValueError('Candidate has no supported primary task')
    return float(metrics[primary]['loss']), metrics


def search_candidates(train, development, parent, spec, directory, *, mode, device='cpu',
                      exchange=None, initial_model=None, history=None, inherit_weights=False,
                      population_size=4):
    directory = project_path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    journal = ExperimentJournal(directory / 'history.jsonl')
    from .validation_artifacts import persist_forward_fold

    def save_candidate(model, record):
        target = directory / f'candidate-{record["iteration"]}'
        model.save(target)
        record['candidate_path'] = str(target)

    best, result = search_population(train, development, parent, spec, mode=mode, device=device,
        exchange=exchange, history=history, inherit_weights=inherit_weights,
        population_size=population_size,
        initial_models={parent.digest: initial_model} if initial_model is not None else None,
        on_candidate=save_candidate, on_record=lambda record: journal.append('candidate', record),
        on_fold=lambda model, record, fold, validation: persist_forward_fold(
            directory, model, record, fold, validation))
    immutable_json(directory / 'results.json', result)
    return best, result


def _save_predictions(path, data, predictions):
    rows = data.meta[['sample_id', 'symbol', 'available_at']].copy()
    for name, value in predictions.items():
        if value.ndim == 1:
            rows[f'score_{name}'] = value
        else:
            for category in range(value.shape[1]):
                rows[f'prob_{name}_{category}'] = value[:, category]
            task = next(t for t in data.tasks if t.name == name)
            rows[f'selected_horizon_{name}'] = adaptive_horizons(value, task)['horizon_hours']
    target = project_path(path)
    if target.exists():
        raise ValueError('Historical predictions cannot be overwritten')
    rows.to_parquet(target, index=False)
    return {'path': str(target), 'sha256': file_hash(target), 'rows': len(rows)}


def _record_failure(function):
    @wraps(function)
    def wrapped(data, spec, output_dir, **kwargs):
        directory = project_path(output_dir)
        already_exists = directory.exists()
        try:
            return function(data, spec, output_dir, **kwargs)
        except (Exception, KeyboardInterrupt) as exc:
            if directory.exists() and not already_exists:
                failure = {'status': 'process_invalid', 'error_type': type(exc).__name__,
                           'error': str(exc), 'protocol_hash': spec.digest,
                           'evidence_may_be_consumed': True}
                immutable_json(directory / 'failure.json', failure)
                ExperimentJournal(directory / 'journal.jsonl').append('failed', failure)
            raise
    return wrapped


@_record_failure
def run_experiment(data: TaskDataset, spec: ExperimentSpec, output_dir, *, data_hash: str,
                   modes=('random', 'diagnostic'), exchange=None, confirm=False, device='cpu',
                   registry_path='data/discovery_evidence.sqlite', evidence_source='binance_um_1h') -> dict:
    if (data.tasks != spec.tasks or data.x.shape[1] != spec.lookback
        or set(data.meta.symbol) != set(spec.symbols)):
        raise ValueError('Dataset does not match the frozen task, lookback or universe')
    if len(set(modes)) != len(modes) or set(modes) - {'random', 'diagnostic', 'external'}:
        raise ValueError('Search modes must be unique registered methods')
    method_pairs = {'single_vs_baseline': ('single', 'baseline'),
                    'fixed_vs_single': ('fixed_experts', 'single')}
    for mode in modes:
        method_pairs[f'{mode}_vs_fixed'] = (f'search_{mode}', 'fixed_experts')
    if 'random' in modes:
        for mode in modes:
            if mode != 'random':
                method_pairs[f'{mode}_vs_random'] = (f'search_{mode}', 'search_random')
    comparisons = ['system_vs_baseline', 'representation_addition',
                   'matched_dimension_control', *method_pairs]
    directory = project_path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    journal = ExperimentJournal(directory / 'journal.jsonl')
    immutable_json(directory / 'protocol.json', {'protocol': asdict(spec), 'sha256': spec.digest})
    immutable_json(directory / 'research_contract.json', {
        'search_modes': modes, 'method_pairs': method_pairs, 'comparisons': comparisons,
        'family_size': len(comparisons),
        'selection': ('cold_walk_forward_robust_rank' if spec.selection.mode == 'walk_forward'
                      else 'lowest development primary loss'),
        'unavailable_arm': 'report unavailable; do not reduce the comparison family'})
    source = snapshot_source(directory / 'source')
    registry = EvidenceRegistry(registry_path)
    registry.import_prior_runs()
    registry.consume(source=evidence_source, symbols=spec.symbols,
                     start=(utc(spec.start) - pd.Timedelta(hours=48 + spec.lookback)).isoformat(),
                     end=spec.calibration_end, run=str(directory), claim=spec.digest,
                     role='development')
    train, dev, calibration = (partition(data, spec, name)
                               for name in ('train', 'development', 'calibration'))
    if min(len(x.x) for x in (train, dev, calibration)) < 32:
        raise ValueError('Train, development and calibration each require at least 32 rows')
    primary = spec.primary_task
    started = time.monotonic()
    single = RepresentationSpec((ModuleSpec('single', 'mlp', 8, 16, min(12, spec.lookback)),))
    experts = RepresentationSpec((ModuleSpec('temporal', 'temporal', 8, 16, min(12, spec.lookback)),
                                  ModuleSpec('interaction', 'mlp', 8, 16, min(12, spec.lookback))))
    models, arms = {}, {}
    for name, architecture in (('baseline', None), ('single', single), ('fixed_experts', experts)):
        settings = spec.training if architecture is not None else replace(spec.training, selection_task=None)
        model = fit_group(train, dev, architecture, settings, spec.training_seeds, device)
        loss, metrics = _score(model, dev, primary)
        models[name] = model
        arms[name] = {'architecture': asdict(architecture) if architecture else None,
                      'development_loss': loss, 'metrics': metrics,
                      'parameter_count': model.parameter_count,
                      'seeds': [m.diagnostics for m in model.models]}
        model.save(directory / 'development_models' / name)
        journal.append('development_fit', {'arm': name, **arms[name]})
    for mode in modes:
        if mode not in {'random', 'diagnostic', 'external'}:
            raise ValueError('Unsupported search arm')
        model, result = search_candidates(train, dev, experts, spec, directory / f'search_{mode}',
                                         mode=mode, device=device, exchange=exchange,
                                         initial_model=models['fixed_experts'])
        name = f'search_{mode}'
        arms[name] = result
        if model is not None:
            loss, metrics = _score(model, dev, primary)
            models[name] = model
            arms[name].update(development_loss=loss, metrics=metrics)
    # Selection is entirely developmental, before calibration or confirmation results.
    if spec.selection.mode == 'walk_forward':
        # Never replace the forward-fold choice using a single reused dev loss.
        eligible_arms = [name for name in models if name.startswith('search_')
                         and arms[name].get('selection_rank') is not None]
        if not eligible_arms:
            raise RuntimeError('No completed walk-forward search and final refit; no candidate selected')
        selected = min(eligible_arms, key=lambda name: (-arms[name]['selection_rank'], name))
        selected_on = 'cold_walk_forward_robust_rank'
    else:
        selected = min((name for name in models if name != 'baseline'),
                       key=lambda name: arms[name]['development_loss'])
        selected_on = 'development_primary_loss'
    candidate, baseline = models[selected], models['baseline']
    bundle = directory / 'bundle'
    candidate.save(bundle / 'candidate')
    baseline.save(bundle / 'baseline')
    selection = {'arm': selected, 'candidate_digest': candidate.digest,
                 'baseline_digest': baseline.digest, 'selected_on': selected_on,
                 'statement': 'Best nonbaseline candidate; confirmation may reject it'}
    immutable_json(bundle / 'selection.json', selection)
    calibrators = {}
    for name, model in [('candidate', candidate), ('baseline', baseline)]:
        calibrator = ProbabilityCalibrator().fit(calibration, model.predict(calibration))
        calibrator.save(bundle / f'{name}_calibration.json')
        calibrators[name] = calibrator
    for name, model in models.items():
        model.save(bundle / 'methods' / name)
        ProbabilityCalibrator().fit(calibration, model.predict(calibration)).save(
            bundle / 'methods' / f'{name}_calibration.json')
    # Actual addition and matched-dimension control have identical probe input dimensions.
    base_train, base_dev = summary_features(train), summary_features(dev)
    z_train, z_dev = candidate.encode(train), candidate.encode(dev)
    mean, std = base_train.mean(0), np.maximum(base_train.std(0), 1e-8)
    rng = np.random.default_rng(spec.seed)
    projection = rng.normal(size=(base_train.shape[1], z_train.shape[1])) / np.sqrt(base_train.shape[1])
    control_train = (base_train - mean) / std @ projection
    control_dev = (base_dev - mean) / std @ projection
    np.savez(bundle / 'probe_projection.npz', mean=mean, std=std, projection=projection)
    probes = {}
    for name, x_train, x_dev in (
        ('reference', base_train, base_dev),
        ('candidate', np.column_stack([base_train, z_train]), np.column_stack([base_dev, z_dev])),
        ('capacity_control', np.column_stack([base_train, control_train]),
         np.column_stack([base_dev, control_dev])),
    ):
        # Closed-form/per-head probe optimization has no shared neural checkpoint.
        probe = fit_probe(x_train, train, x_dev, dev, replace(spec.training, selection_task=None))
        probe.save(bundle / f'probe_{name}')
        probes[name] = probe
    immutable_json(bundle / 'probe_contract.json', {
        'representation_dimension': z_train.shape[1], 'base_dimension': base_train.shape[1],
        'candidate_dimension': base_train.shape[1] + z_train.shape[1],
        'control_dimension': base_train.shape[1] + control_train.shape[1],
        'control': 'train-scaled raw summary with a fixed label-independent random projection',
        'scope': 'matched downstream dimensions; not a proof of unique or causal information',
        'checkpoint_selection': 'closed_form_or_per_head_development_nll',
        'generator_digest': candidate.digest,
    })
    immutable_json(bundle / 'method_contract.json', {'pairs': method_pairs,
        'available_models': sorted(models), 'family_size': len(comparisons)})
    claim = freeze_claim(directory, spec, bundle, data_hash, source['digest'], comparisons)
    verify_bundle(bundle, claim)
    result = {'status': 'development_complete', 'run_directory': str(directory),
              'evidence_source': evidence_source,
              'protocol_hash': spec.digest, 'claim_hash': claim['digest'], 'selection': selection,
              'arms': arms, 'primary_task': primary, 'confirmation': None,
              'representation_identity': 'whole frozen generator group, not aligned latent coordinates',
              'limitations': [
                  'Fixed surviving symbols and historical archive availability assumptions.',
                  'Candidate/seed counts and resource ceilings are matched, actual work differs.',
                  'All confirmed claims are restricted to this task, interval and observed coverage.',
                  'Predictive windows start at the next full observable hour; intrabar double hits are censored.',
                  'Historical agent proposals may contain LLM pretraining knowledge.',
              ]}
    if confirm:
        # Consume first, so an exception or failed conclusion cannot restore unseen status.
        registry.consume(source=evidence_source, symbols=spec.symbols,
                         start=spec.calibration_end, end=spec.end, run=str(directory),
                         claim=claim['digest'])
        journal.append('confirmation_consumed', {'claim_hash': claim['digest']})
        verify_bundle(bundle, claim)
        holdout = partition(data, spec, 'confirmation')
        # Re-read persisted fitted state, including calibrators and probes, before evaluation.
        candidate, baseline = ModelGroup.load(bundle / 'candidate'), ModelGroup.load(bundle / 'baseline')
        pred_c = ProbabilityCalibrator.load(bundle / 'candidate_calibration.json').transform(candidate.predict(holdout))
        pred_b = ProbabilityCalibrator.load(bundle / 'baseline_calibration.json').transform(baseline.predict(holdout))
        h_base, h_z = summary_features(holdout), candidate.encode(holdout)
        arrays = np.load(bundle / 'probe_projection.npz', allow_pickle=False)
        h_control = (h_base - arrays['mean']) / arrays['std'] @ arrays['projection']
        encoded = {'reference': h_base, 'candidate': np.column_stack([h_base, h_z]),
                   'capacity_control': np.column_stack([h_base, h_control])}
        probe_predictions = {key: FittedProbe.load(bundle / f'probe_{key}').predict_encoded(value, holdout)
                             for key, value in encoded.items()}
        family_size = len(comparisons)
        compared = {
            'system_vs_baseline': paired_comparison(holdout, pred_c, pred_b, primary, spec, family_size),
            'representation_addition': paired_comparison(holdout, probe_predictions['candidate'],
                                                       probe_predictions['reference'], primary, spec, family_size),
            'matched_dimension_control': paired_comparison(holdout, probe_predictions['candidate'],
                                                         probe_predictions['capacity_control'], primary, spec, family_size),
        }
        method_predictions = {}
        for name in models:
            loaded = ModelGroup.load(bundle / 'methods' / name)
            method_predictions[name] = ProbabilityCalibrator.load(
                bundle / 'methods' / f'{name}_calibration.json').transform(loaded.predict(holdout))
        for key, (left, right) in method_pairs.items():
            compared[key] = (paired_comparison(holdout, method_predictions[left],
                method_predictions[right], primary, spec, family_size)
                if left in method_predictions and right in method_predictions
                else {'status': 'unavailable', 'reason': 'Search did not produce a valid fitted model'})
        result['confirmation'] = {'comparisons': compared,
            'method_metrics': {name: evaluate(holdout, values)
                               for name, values in method_predictions.items()},
            'candidate_metrics': evaluate(holdout, pred_c), 'baseline_metrics': evaluate(holdout, pred_b),
            'auxiliary_metrics_status': 'descriptive_only_not_additional_confirmed_claims',
            'candidate_predictions': _save_predictions(directory / 'confirmation_scores.parquet', holdout, pred_c),
            'baseline_predictions': _save_predictions(directory / 'confirmation_reference.parquet', holdout, pred_b),
            'accepted_system': compared['system_vs_baseline']['status'] == 'supported',
            'supported_representation': all(compared[k]['status'] == 'supported'
                for k in ('representation_addition', 'matched_dimension_control'))}
        result['status'] = 'confirmation_complete'
        journal.append('confirmation_result', result['confirmation'])
    result['elapsed_seconds'] = time.monotonic() - started
    immutable_json(directory / 'results.json', result)
    journal.append('completed', {'result_hash': file_hash(directory / 'results.json')})
    return result
