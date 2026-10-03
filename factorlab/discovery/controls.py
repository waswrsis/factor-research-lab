"""Known-signal and independent-noise checks; these are never market evidence."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .contracts import ExperimentSpec, TaskDataset, TaskSpec, TrainingSpec, partition
from .evidence import immutable_json
from .learning import fit_probe
from .scoring import paired_comparison


def run_controls(output_path=None):
    rng = np.random.default_rng(7231)
    n = 4800
    raw = rng.normal(size=(n, 3))
    for i in range(1, n):
        raw[i] = 0.65 * raw[i - 1] + np.sqrt(1 - 0.65 ** 2) * raw[i]
    dates = pd.date_range('2023-01-01', periods=n, freq='h', tz='UTC')
    spec = ExperimentSpec(tasks=(TaskSpec(),), symbols=('CONTROL',), lookback=1,
        start=dates[0].isoformat(), train_end=dates[2400].isoformat(),
        dev_end=dates[3000].isoformat(), calibration_end=dates[3600].isoformat(),
        end=(dates[-1] + pd.Timedelta(hours=1)).isoformat(),
        training=TrainingSpec(epochs=8), training_seeds=(17,),
        bootstrap_samples=2000, bootstrap_block_hours=24, min_blocks=20,
        minimum_increment=0.02)
    y = np.exp(-3 + 0.7 * raw[:, 0] + 0.25 * raw[:, 2])
    data = TaskDataset(raw[:, None].astype(np.float32), {'volatility': y},
        {'volatility': np.ones(n, dtype=bool)},
        {'volatility': (dates + pd.Timedelta(hours=12)).to_numpy(dtype='datetime64[ns]')},
        pd.DataFrame({'sample_id': [f'C:{i}' for i in range(n)],
                      'symbol': 'CONTROL', 'available_at': dates}), ('signal', 'independent', 'noise'), spec.tasks)
    train, dev, test = (partition(data, spec, name)
                        for name in ('train', 'development', 'confirmation'))
    ref = fit_probe(np.ones((len(train.x), 1)), train, np.ones((len(dev.x), 1)), dev, spec.training)
    reference = ref.predict_encoded(np.ones((len(test.x), 1)), test)
    outputs = {}
    for name, column in [('known_signal', 0), ('independent_noise', 1)]:
        model = fit_probe(train.x[:, 0, column:column+1], train,
                          dev.x[:, 0, column:column+1], dev, spec.training)
        prediction = model.predict_encoded(test.x[:, 0, column:column+1], test)
        outputs[name] = paired_comparison(test, prediction, reference, spec.primary_task, spec, 2)
    # Deterministic equal predictions must never receive an incremental claim.
    outputs['identical_predictions'] = paired_comparison(test, reference, reference,
        spec.primary_task, replace(spec, minimum_increment=0.0), 1)
    success = (outputs['known_signal']['status'] == 'supported'
               and outputs['independent_noise']['status'] != 'supported'
               and outputs['identical_predictions']['status'] != 'supported')
    result = {'passed': success, 'source': 'synthetic_ar_process_seed_7231',
        'hypothesis': 'A fixed linear probe recovers an available known signal but does not promote independent AR noise.',
        'dependence': 'Independent AR(1) channels with persistence 0.65; no shuffled market labels.',
        'scope': 'pipeline and probe check, not market efficacy or universal false-positive control',
        'comparisons': outputs, 'protocol': spec.to_dict()}
    if output_path is not None:
        immutable_json(output_path, result)
    return result
