"""Score the complete frozen neural learner, never the mean of seed scores.

No fitting, feedback, selection, admission or claims happen here. The S worker
must supply a cold model from its own fit cutoff and reserve all work first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factorlab.protocol import fingerprint
from factorlab.time_utils import HOUR

from .convex_eval import VOLATILITY_FLOOR, log_mse
from .joint_pool import hour_equal_weights
from .neural_forward import NeuralForwardModel
from .v7_lifecycle import stamp


def score_complete_neural_fold(data, model, *, start, end, as_of, symbols):
    """Same-support per-hour logMSE against this learner's frozen reference.

    Coverage uses the declared UTC-hour grid, including missing source rows.
    Partial symbol availability is reported; it cannot shrink the nominal
    denominator. A caller cannot score an old monthly model as a new cold fit.
    """
    if not isinstance(model, NeuralForwardModel):
        raise TypeError('A complete frozen neural model is required')
    start, end, as_of = map(stamp, (start, end, as_of))
    if (not start < end <= as_of or start != start.floor('h') or end != end.floor('h')
            or stamp(model.receipt['cutoff']) != start or data.spec.digest != model.recipe.rolling_spec.digest
            or not isinstance(symbols, (tuple, list)) or not symbols
            or len(set(symbols)) != len(symbols) or len(symbols) > 3
            or any(not isinstance(s, str) or not s for s in symbols)):
        raise ValueError('Complete ensemble score requires its own cold cutoff and declared window/universe')
    if any(model.receipt['training'].get(key) != value
           for key, value in data.training_identity(start).items()):
        raise ValueError('Complete learner was fitted on a different training source')
    prediction, support = model.predict_history(data.frame)
    reference, reference_support = model.baseline.predict_history(data.frame)
    declared = data.frame.metadata.symbol.isin(symbols).to_numpy()
    mask = data.score_mask(start, end, as_of) & support & reference_support & declared
    nominal_hours = int((end - start) / HOUR)
    nominal_rows = nominal_hours * len(symbols)
    indices = np.flatnonzero(mask)
    metadata = data.frame.metadata.loc[mask]
    # Also checks duplicate symbol/hour origins before statistics are computed.
    weights = hour_equal_weights(metadata) if len(indices) else np.array([])
    if len(indices):
        y, a, b = data.target[mask], reference[mask], prediction[mask]
        reference_loss, loss = log_mse(y, a, weights), log_mse(y, b, weights)
        hour = pd.DatetimeIndex(metadata.available_at).floor('h')
        rows = pd.DataFrame({'hour': hour, 'reference_loss': np.square(np.log(y + VOLATILITY_FLOOR)
                      - np.log(a + VOLATILITY_FLOOR)),
                     'loss': np.square(np.log(y + VOLATILITY_FLOOR) - np.log(b + VOLATILITY_FLOOR))})
        hourly = [{'hour': t.isoformat(), 'reference_loss': float(group.reference_loss.mean()),
                   'loss': float(group.loss.mean()), 'symbols_present': len(group)}
                  for t, group in rows.groupby('hour', sort=True)]
        if len(indices) > nominal_rows:
            raise ValueError('Score support exceeds its frozen nominal denominator')
        # This arithmetic identity also guards accidental row-weight averaging.
        if (not np.isclose(np.mean([h['loss'] for h in hourly]), loss, rtol=1e-12, atol=1e-15)
                or not np.isclose(np.mean([h['reference_loss'] for h in hourly]), reference_loss,
                                  rtol=1e-12, atol=1e-15)):
            raise ValueError('Ensemble hourly scores disagree with the canonical metric')
    else:
        reference_loss = loss = None
        hourly = []
    value = {'version': 'v7-complete-neural-S-fold-v1', 'role': 'S',
             'pipeline_id': model.pipeline_id, 'model_id': model.digest,
             'reference_model_id': model.baseline.digest, 'training': model.receipt['training'],
             'start': start.isoformat(), 'end': end.isoformat(), 'as_of': as_of.isoformat(),
             'aggregation': model.recipe.to_dict()['rules']['aggregation'],
             'metric': 'canonical_logMSE_equal_symbol_then_hour', 'symbols': list(symbols),
             'nominal_rows': nominal_rows, 'support': len(indices),
             'coverage': len(indices) / nominal_rows, 'reference_loss': reference_loss, 'loss': loss,
             'gain': 1 - loss / reference_loss if reference_loss and loss is not None else None,
             'sample_ids': metadata.sample_id.tolist(), 'target': data.target[mask].tolist(),
             'predictions': prediction[mask].tolist(), 'reference_predictions': reference[mask].tolist(),
             'hourly': hourly, 'test_only_model': model.receipt.get('test_only', False),
             'formal_claim': False, 'automatic_adoption': False,
             'interpretation': 'Full learner S score; not seed-score averaging or forward permission'}
    return {**value, 'score_id': fingerprint(value)}
