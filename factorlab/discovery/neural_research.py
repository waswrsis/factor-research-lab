"""Complete-recipe three-fold S numerics with an isolated ticket entry.

The public entry requires actual S authority, exposure and registered gates.
Private artificial runs remain test-only. Research never grants serving rights.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from factorlab.protocol import fingerprint, project_path
from factorlab.time_utils import HOUR, calendar_window

from .convex_eval import VOLATILITY_FLOOR
from .evidence import file_hash
from .joint_pool import FrozenPool, JointPredictor, fit_pool, hour_equal_weights
from .neural_ensemble_score import score_complete_neural_fold
from .neural_forward import (
    FrozenNeuralRecipe,
    NeuralForwardModel,
    _decode,
    _embedding,
    _encoded_frame,
    _joint,
    _ridge,
)
from .nonlinear_channel import ChannelBudget
from .nonlinear_orchestration import causal_worker_data, input_data_identity
from .rolling import development_folds
from .v7_storage import immutable_record


def evaluate_neural_research(*, ticket_path):
    """No caller booleans, callbacks or forward ticket can authorize S work."""
    from .neural_research_launcher import launch_research
    return launch_research(ticket_path)


def research_capacity(recipe):
    """Upper bounds for all three cold learners plus one K0 solve per fold."""
    if not isinstance(recipe, FrozenNeuralRecipe):
        raise TypeError('Complete frozen neural recipe required')
    bounds = recipe.budget_limits
    return {'tree_fit': 0, 'neural_fit': 3 * bounds['neural_fit'],
            'epoch': 3 * bounds['epoch'], 'solve': 3 * (bounds['solve'] + 1)}


def _ensemble(values):
    values = np.asarray(values)
    result = _decode(np.mean(np.log(values + VOLATILITY_FLOOR), axis=0))[0]
    result[np.all(values == 0, axis=0)] = 0.
    return result


def _diagnostics(model, frame):
    """Same saved weights/readouts; no fitting or checkpoint reselection here."""
    design, support = model.baseline.feature_history(frame)
    selected = design.take(support)
    reference = model.baseline.predictor.predict(selected)
    X = model.baseline.predictor.scaler.transform(selected)
    seed_predictions, initial, trained = [], [], []
    for seed in model.seeds:
        codes = _embedding(X, seed['encoder'], model.recipe.config.structures[0])
        original = _embedding(X, seed['initial_encoder'], model.recipe.config.structures[0])
        trained_prediction = _ridge(seed['trained_readout']).predict(
            _encoded_frame(selected, codes, seed['encoding_id']))
        initial.append(_ridge(seed['initial_readout']).predict(
            _encoded_frame(selected, original, seed['initial_encoding_id'])))
        trained.append(trained_prediction)
        if seed['selected_epoch'] == 0:
            prediction = reference
        elif model.recipe.output == 'encoder_ridge':
            prediction = trained_prediction
        else:
            q = np.log(reference + VOLATILITY_FLOOR) + (
                codes @ np.asarray(seed['head']['weight']).T).reshape(-1) + seed['head']['bias'][0]
            prediction = _decode(q)[0]
        seed_predictions.append(prediction)
    values = {'initial_encoder_ridge': _ensemble(initial), 'trained_encoder_ridge': _ensemble(trained)}
    values.update({f'seed_{s["seed"]}': p for s, p in zip(model.seeds, seed_predictions, strict=True)})
    result = {}
    for name, prediction in values.items():
        result[name] = np.full(len(frame.values), np.nan)
        result[name][support] = prediction
    return result, support


def score_research_fold(data, model, k0, *, start, end, symbols):
    """Paired scores of the actual complete learner and separately named controls."""
    if not isinstance(k0, JointPredictor) or k0.pool.members:
        raise ValueError('K0 control must be a cold empty-pool model')
    basic = score_complete_neural_fold(data, model, start=start, end=end, as_of=end, symbols=symbols)
    if (pd.Timestamp(k0.receipt['cutoff']) != pd.Timestamp(start)
            or any(k0.receipt['training'].get(k) != v for k, v in data.training_identity(start).items())):
        raise ValueError('K0 control belongs to another cold fold or training source')
    predictions, support = _diagnostics(model, data.frame)
    basic_mask = (data.score_mask(start, end, end) & support
                  & data.frame.metadata.symbol.isin(symbols).to_numpy())
    if data.frame.metadata.loc[basic_mask, 'sample_id'].tolist() != basic['sample_ids']:
        raise ValueError('Complete learner and diagnostic control support differs')
    # Reuse the authoritative scorer's actual predictions. Re-running the same
    # full history neural/pool inference adds cost without new evidence.
    for name, key in (('ensemble', 'predictions'), ('Pv', 'reference_predictions')):
        predictions[name] = np.full(len(data.target), np.nan)
        predictions[name][basic_mask] = basic[key]
    predictions['K0'], k0_mask = k0.predict_history(data.frame)
    mask = basic_mask & k0_mask
    mask &= np.logical_and.reduce([np.isfinite(p) & (p >= 0) for p in predictions.values()])
    meta = data.frame.metadata.loc[mask]
    hour_equal_weights(meta) if mask.any() else None  # Reject duplicated symbol/hour origins.
    hours = pd.DatetimeIndex(meta.available_at).floor('h')
    errors = {k: np.square(np.log(data.target[mask] + VOLATILITY_FLOOR)
                          - np.log(p[mask] + VOLATILITY_FLOOR)) for k, p in predictions.items()}
    table = pd.DataFrame({'hour': hours, **errors})
    hourly = [{'hour': t.isoformat(), 'symbols_present': len(group),
               'losses': {k: float(group[k].mean()) for k in errors}}
              for t, group in table.groupby('hour', sort=True)]
    losses = {k: float(np.mean([h['losses'][k] for h in hourly])) if hourly else None for k in errors}
    comparisons = {k: {'reference_loss': losses[k], 'candidate_loss': losses['ensemble'],
                       'absolute_delta': losses[k] - losses['ensemble'] if hourly else None}
                   for k in ('K0', 'Pv')}
    return {'role': 'S', 'start': basic['start'], 'end': basic['end'],
        'model_id': model.digest, 'pipeline_id': model.pipeline_id, 'k0_model_id': k0.digest,
        'pv_model_id': model.baseline.digest, 'training': copy.deepcopy(model.receipt['training']),
        'seed_checkpoints': [{'seed': s['seed'], 'epoch': s['selected_epoch'],
                             'data_order_hashes': s['data_order_hashes']} for s in model.seeds],
        'nominal_rows': basic['nominal_rows'], 'nominal_hours': int((pd.Timestamp(end) - pd.Timestamp(start)) / HOUR),
        'support': int(mask.sum()), 'coverage': float(mask.sum() / basic['nominal_rows']),
        'common_sample_ids': meta.sample_id.tolist(), 'targets': data.target[mask].tolist(),
        'predictions': {k: p[mask].tolist() for k, p in predictions.items()}, 'hourly': hourly,
        'losses': losses, 'S': comparisons,
        'mean_seed_loss_diagnostic': float(np.mean([losses[f'seed_{s["seed"]}'] for s in model.seeds])) if hourly else None,
        'controls': 'initial/trained encoder same-ridge ensembles are diagnostics, not alternative winners',
        'aggregation': 'equal_symbol_within_hour_then_equal_hour', 'automatic_adoption': False, 'formal_claim': False}


def _run_research_worker(data, cutoff, recipe, *, symbols, directory, budget):
    """Only a genuinely registered child can use the future production S worker."""
    from .reserved_work_budget import ReservedChannelBudget
    if (type(budget) is not ReservedChannelBudget or budget.test_only
            or budget.executor_mode != 'child'):
        raise TypeError('Complete S worker requires actual child-only prepaid authority')
    budget.check_time()
    from . import neural_forward
    fit = getattr(neural_forward, '_fit_neural_research_payload', None)
    if fit is None:
        raise NotImplementedError('S payload origin is not implemented; forward admission cannot be borrowed')
    return _evaluate(data, cutoff, recipe, symbols=symbols, directory=directory,
                     budget=budget, fit=fit, artificial=False)


def _evaluate_artificial(data, cutoff, recipe, *, symbols, directory, budget):
    """Private small-fixture seam; never accepts a production wallet as fake proof."""
    if type(budget) is not ChannelBudget:
        raise TypeError('Artificial experiments require explicit local ChannelBudget')
    from .neural_forward import _fit_neural_forward_cold
    return _evaluate(data, cutoff, recipe, symbols=symbols, directory=directory,
                     budget=budget, fit=_fit_neural_forward_cold, artificial=True)


def _evaluate(data, cutoff, recipe, *, symbols, directory, budget, fit, artificial):
    capacity = research_capacity(recipe)
    if any(budget.counts[k] + n > budget.limits[k] for k, n in capacity.items()):
        raise ValueError('Insufficient prepaid capacity for three complete cold folds and K0 controls')
    if (data.spec.digest != recipe.rolling_spec.digest or not isinstance(symbols, (tuple, list))
            or not symbols or len(symbols) > 3 or len(set(symbols)) != len(symbols)
            or any(not isinstance(s, str) or not s for s in symbols)):
        raise ValueError('Research input/protocol/universe differs from its full recipe')
    _, cutoff = recipe.rolling_spec.window(cutoff)
    data = causal_worker_data(data, cutoff)
    directory = project_path(directory)
    if directory.exists():
        raise ValueError('Research attempt directory exists; failed work cannot silently restart')
    directory.mkdir(parents=True)
    manifest = {'version': 'v7-complete-neural-S-research-v1', 'role': 'S',
        'cutoff': cutoff.isoformat(), 'recipe': recipe.to_dict(), 'pipeline_id': recipe.pipeline_id,
        'symbols': list(symbols), 'input_id': input_data_identity(data, cutoff),
        'folds': list(development_folds(cutoff, spec=recipe.rolling_spec)), 'capacity': capacity, 'test_only': artificial,
        'evidence_status': 'exposed_development_only', 'automatic_adoption': False, 'formal_claim': False}
    immutable_record(directory / 'manifest.json', manifest)
    folds = []
    for i, fold in enumerate(manifest['folds']):
        nominal_hours = int((pd.Timestamp(fold['score_end']) - pd.Timestamp(fold['fit_cutoff'])) / HOUR)
        record = {**fold, 'status': 'failed', 'nominal_hours': nominal_hours,
                  'nominal_rows': nominal_hours * len(symbols), 'support': 0, 'coverage': 0., 'hourly': []}
        try:
            budget.check_time()
            cold_data = causal_worker_data(data, fold['fit_cutoff'])
            model = fit(cold_data, fold['fit_cutoff'], recipe, budget)
            path = model.save(directory / f'fold-{i}-model.json')
            restored = NeuralForwardModel.load(path)
            if restored.digest != model.digest:
                raise ValueError('Saved research model did not restore its full identity')
            k0 = fit_pool(cold_data, fold['fit_cutoff'], FrozenPool(), budget)
            k0_path = directory / f'fold-{i}-k0.json'
            immutable_record(k0_path, k0.to_dict())
            import json
            restored_k0 = _joint(json.loads(k0_path.read_text('utf-8')))
            score = score_research_fold(data, restored, restored_k0, start=fold['fit_cutoff'],
                                        end=fold['score_end'], symbols=symbols)
            budget.check_time()
            record.update(score, status='completed', model={'path': str(path), 'sha256': file_hash(path)},
                          k0={'path': str(k0_path), 'sha256': file_hash(k0_path)}, budget_counts=dict(budget.counts))
        except (ValueError, RuntimeError, MemoryError, OSError, np.linalg.LinAlgError) as error:
            record.update(error={'type': type(error).__name__, 'message': str(error)[:1000]},
                          budget_counts=dict(budget.counts))
        immutable_record(directory / f'fold-{i}.json', record)
        folds.append(record)
    hours = [h for f in folds for h in f['hourly']]
    complete = all(f['status'] == 'completed' and f['support'] > 0 for f in folds)
    names = list(hours[0]['losses']) if hours else []
    conditional = {k: float(np.mean([h['losses'][k] for h in hours])) for k in names}
    nominal = sum(f['nominal_rows'] for f in folds)
    result = {**manifest, 'manifest_id': fingerprint(manifest), 'folds': folds,
        'status': 'completed' if complete else 'incomplete', 'nominal_rows': nominal,
        'support': sum(f['support'] for f in folds), 'coverage': sum(f['support'] for f in folds) / nominal,
        'nominal_hours': sum(f['nominal_hours'] for f in folds), 'supported_hours': len(hours),
        'conditional_losses': conditional, 'losses': conditional if complete else None,
        'budget_counts': dict(budget.counts),
        'interpretation': 'Full recipe S numerics only; no F novelty, selection, forward permission or adoption'}
    immutable_record(directory / 'result.json', result)
    return result
