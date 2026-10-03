"""Persist auditable forward-fold forecasts, including rejected candidates."""
from __future__ import annotations

import numpy as np
import pandas as pd

from factorlab.protocol import project_path

from .evidence import file_hash, immutable_json
from .replay import _prediction_inputs
from .robust_selection import macro_metrics
from .scoring import per_sample_loss


def persist_forward_fold(directory, model, candidate, fold, data):
    target = project_path(directory) / 'forward_folds' / candidate['candidate_id'] / f"fold-{fold['fold']:02d}"
    target.mkdir(parents=True, exist_ok=False)
    model.save(target / 'model')
    clean = _prediction_inputs(data)
    models = [('ensemble', model)] + [(f'seed-{member.metadata["settings"]["seed"]}', member)
                                     for member in getattr(model, 'models', [])]
    artifacts = {}
    for name, fitted in models:
        predictions = fitted.predict(clean)
        metrics = macro_metrics(data, predictions)
        rows = data.meta.copy()
        for task in data.tasks:
            mask = data.task_mask(task.name)
            rows[f'{task.name}_target'] = np.where(mask, data.targets[task.name], np.nan)
            rows[f'{task.name}_observed'] = mask
            rows[f'{task.name}_maturity'] = pd.to_datetime(data.maturity[task.name], utc=True)
            if task.name not in predictions:
                continue
            p = np.asarray(predictions[task.name])
            if p.ndim == 1:
                rows[f'{task.name}_prediction'] = p
            else:
                for column in range(p.shape[1]):
                    rows[f'{task.name}_prediction_{column}'] = p[:, column]
            loss = np.full(len(rows), np.nan)
            if mask.any():
                loss[mask] = per_sample_loss(task, data.targets[task.name][mask], p[mask])
            rows[f'{task.name}_loss'] = loss
        path = target / f'{name}-forecasts.parquet'
        rows.to_parquet(path, index=False)
        artifacts[name] = {'path': str(path), 'sha256': file_hash(path),
                           'metrics': metrics, 'model_digest': fitted.digest}
    record = {'candidate_id': candidate['candidate_id'], 'fold': fold['fold'],
              'model_digest': model.digest, 'forecasts': artifacts,
              'split': {k: v for k, v in fold.items() if k not in {'training', 'fit_attempts'}},
              'scope': 'development_only', 'independent_confirmation': False}
    immutable_json(target / 'fold.json', record)
    return {'path': str(target), 'record_sha256': file_hash(target / 'fold.json'),
            'model_digest': model.digest}
