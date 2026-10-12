"""Whole-procedure paired development comparisons, retaining every forecast hour."""
from __future__ import annotations

import json
import math
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock

from factorlab.protocol import fingerprint, project_path, utc
from factorlab.time_utils import hour_ids

from .contracts import ExperimentSpec, TaskDataset
from .evidence import file_hash, immutable_json
from .evolution import EvolutionStore, inspect_evolution
from .scoring import paired_comparison, per_sample_loss

PAIRWISE_FAMILY_SIZE = 3  # fixed/refit/evolve: three predeclared pairwise comparisons


def _read(directory):
    summary = inspect_evolution(directory)
    store = EvolutionStore(directory)
    try:
        state = store.state()
        manifest = json.loads((directory / 'manifest.json').read_text('utf-8'))
        if fingerprint(manifest['manifest']) != manifest['sha256']:
            raise ValueError('Evolution manifest integrity failure')
        identity = state['identity']
        if any(manifest['manifest'].get(key) != value for key, value in identity.items()):
            raise ValueError('Evolution manifest and committed identity differ')
        columns = {row['name'] for row in store.db.execute('PRAGMA table_info(predictions)')}
        if 'symbol' not in columns:
            raise ValueError('Evolution predictions require persisted symbol metadata for comparison')
        forecasts = [dict(row) for row in store.db.execute(
            "SELECT * FROM predictions WHERE role='incumbent' ORDER BY origin,sample,task")]
        models = {row['id']: store.load_model(row['id'])
                  for row in store.db.execute('SELECT id FROM models').fetchall()}
        return {'summary': summary, 'state': state, 'identity': identity,
                'forecasts': forecasts, 'models': models, 'events': store.events()}
    finally:
        store.close()


def _index(run, spec):
    expected = {}
    for event in run['events']:
        if event['event'] != 'prediction':
            continue
        origin = utc(event['timestamp']).isoformat()
        value = event['value']
        for sample in value['sample_ids']:
            for task in spec.tasks:
                key = (sample, task.name)
                if key in expected:
                    raise ValueError('Duplicate incumbent forecast event coverage')
                expected[key] = (origin, value['incumbent'])
    indexed = {}
    origins = {}
    for row in run['forecasts']:
        key = (row['sample'], row['task'])
        if key in indexed or key not in expected:
            raise ValueError('Incumbent forecast coverage differs from committed events')
        stamp = utc(row['origin'])
        if (stamp.isoformat(), row['model']) != expected[key]:
            raise ValueError('Incumbent forecast identity differs from committed event')
        if not utc(spec.train_end) <= stamp < utc(spec.dev_end):
            raise ValueError('Incumbent forecast origin outside frozen development interval')
        if row['symbol'] not in spec.symbols or row['model'] not in run['models']:
            raise ValueError('Forecast symbol or committed model identity unavailable')
        previous = origins.get(row['sample'])
        identity = (stamp.isoformat(), row['symbol'])
        if previous is not None and previous != identity:
            raise ValueError('Tasks disagree on sample identity')
        origins[row['sample']] = identity
        indexed[key] = row
    if set(indexed) != set(expected) or not origins:
        raise ValueError('Incomplete incumbent forecast coverage; hours cannot be trimmed')
    return indexed, origins


def _validate_prediction(task, row):
    prediction = np.asarray(json.loads(row['value']), dtype=np.float64)
    # A dummy category checks returned probabilities even for pending/unobserved rows.
    per_sample_loss(task, np.asarray([0.]), prediction[None])
    settled = row['settled']
    if settled not in (0, 1):
        raise ValueError('Invalid forecast settlement state')
    if not settled:
        if any(row[name] is not None for name in ('status', 'loss', 'target')):
            raise ValueError('Pending forecast contains settlement values')
    elif row['status'] == 'unobserved':
        if row['loss'] is not None or row['target'] is not None:
            raise ValueError('Unobserved settlement contains a target or loss')
    elif row['status'] == 'scored':
        if row['target'] is None or row['loss'] is None:
            raise ValueError('Scored settlement lacks target/loss')
        loss = float(per_sample_loss(task, np.asarray([row['target']]), prediction[None])[0])
        if not math.isclose(loss, row['loss'], rel_tol=1e-8, abs_tol=1e-10):
            raise ValueError('Recorded settlement loss differs from task-defined loss')
    else:
        raise ValueError('Invalid settled forecast status')
    return prediction


def _resources(run):
    cycles = run['summary']['cycles']
    clocks = []
    missing = []
    parameters = {identifier: model.parameter_count for identifier, model in run['models'].items()}
    for cycle in cycles:
        value = cycle.get('research', {}).get('elapsed_seconds')
        if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
            clocks.append(float(value))
        else:
            missing.append(cycle['id'])
    incumbent_counts = {}
    for row in run['forecasts']:
        incumbent_counts[row['model']] = incumbent_counts.get(row['model'], 0) + 1
    total = sum(incumbent_counts.values())
    return {'cycles_attempted': len(cycles), 'promotions': run['summary']['promotions'],
        'failed_or_interrupted_cycles': sum(cycle['status'] in {
            'failed', 'invalid_candidate', 'interrupted_slot_consumed'} for cycle in cycles),
        'cycle_statuses': {status: sum(c['status'] == status for c in cycles)
                           for status in sorted({c['status'] for c in cycles})},
        'known_search_seconds': sum(clocks), 'unmeasured_cycle_ids': missing,
        'search_seconds_complete': not missing,
        'committed_model_versions': len(parameters), 'parameters_by_model': parameters,
        'maximum_committed_parameters': max(parameters.values()),
        'forecast_weighted_incumbent_parameters': (
            sum(parameters[key] * count for key, count in incumbent_counts.items()) / total),
        'resource_scope': 'recorded research wall clocks and committed model parameters; '
                          'failed/unfinished clocks may be unavailable; no FLOPs/API-cost claim'}


def _descriptive(indices, data, candidate, reference):
    result = {}
    for task in data.tasks:
        selected = indices & data.task_mask(task.name)
        n = int(selected.sum())
        if n:
            cl = per_sample_loss(task, data.targets[task.name][selected], candidate[task.name][selected])
            rl = per_sample_loss(task, data.targets[task.name][selected], reference[task.name][selected])
            rows = data.meta.loc[selected, ['symbol', 'available_at']].copy()
            rows['hour'] = rows.available_at.dt.floor('h')
            rows['candidate'] = cl
            rows['reference'] = rl
            hourly = rows.groupby(['hour', 'symbol'])[['candidate', 'reference']].mean()
            hourly = hourly.groupby(level='hour').mean()
            cm, rm = float(hourly.candidate.mean()), float(hourly.reference.mean())
        else:
            cm, rm = None, None
        result[task.name] = {'support': n, 'total_rows': int(indices.sum()),
            'candidate_loss': cm, 'reference_loss': rm,
            'reference_minus_candidate': rm - cm if n else None,
            'aggregation': 'equal_symbol_within_hour_then_equal_hour',
            'descriptive_only': True}
    return result


def compare_evolutions(candidate_dir: str | Path, reference_dir: str | Path,
                       output_path: str | Path) -> dict:
    """Compare all common incumbent outputs of the complete selection procedures.

    Any difference in sample/task/settlement eligibility rejects comparison;
    training failures stay in the procedure through retained-incumbent forecasts.
    Shadow candidates are never substituted for the operational outputs.
    """
    candidate_dir, reference_dir = project_path(candidate_dir), project_path(reference_dir)
    output_path = project_path(output_path)
    if output_path.exists():
        raise ValueError('Comparison output already exists')
    with ExitStack() as stack:
        for directory in sorted({candidate_dir, reference_dir}, key=str):
            stack.enter_context(FileLock(str(directory / '.evolution.lock'), timeout=0))
        candidate, reference = _read(candidate_dir), _read(reference_dir)
    ci, ri = candidate['identity'], reference['identity']
    for key in ('format', 'protocol', 'schema', 'initial_digest', 'source_files', 'context_policy', 'scope'):
        if ci.get(key) != ri.get(key):
            raise ValueError(f'Evolution procedures have different {key}')
    if ci['scope'] != 'development_sequential_research':
        raise ValueError('Evolution comparisons require declared development-only scope')
    for key in ('last_timestamp', 'visible_hash', 'ticks'):
        if candidate['state'].get(key) != reference['state'].get(key):
            raise ValueError(f'Evolution procedures have different observed {key}')
    if not candidate['state'].get('visible_hash') or not ci.get('initial_digest'):
        raise ValueError('Comparison requires committed observed data and initial model identity')
    scoring_hash = ci['source_files'].get('factorlab/discovery/scoring.py', ci['source_files'].get('scoring.py'))
    if scoring_hash != file_hash(Path(__file__).with_name('scoring.py')):
        raise ValueError('Frozen scoring source differs from current comparison scoring')
    spec = ExperimentSpec.from_dict(ci['protocol'])
    cp, cm = _index(candidate, spec)
    rp, rm = _index(reference, spec)
    if set(cp) != set(rp) or cm != rm:
        raise ValueError('Evolution sample/origin/symbol/task coverage differs; cannot trim hours')
    samples = sorted(cm, key=lambda sample: (*cm[sample], sample))
    n = len(samples)
    targets, observed, maturity, cv, rv = {}, {}, {}, {}, {}
    support = {}
    for task in spec.tasks:
        targets[task.name] = np.full(n, np.nan)
        observed[task.name] = np.zeros(n, dtype=bool)
        # This container never trains/predicts; recorded settlement states determine eligibility.
        maturity[task.name] = np.full(n, np.datetime64('NaT'), dtype='datetime64[ns]')
        c_values, r_values = [], []
        pending, unobserved = 0, 0
        for index, sample in enumerate(samples):
            c, r = cp[(sample, task.name)], rp[(sample, task.name)]
            c_values.append(_validate_prediction(task, c))
            r_values.append(_validate_prediction(task, r))
            if any(c[key] != r[key] for key in ('settled', 'status', 'target')):
                raise ValueError('Evolution settlement/target eligibility differs; cannot trim hours')
            if not c['settled']:
                pending += 1
            elif c['status'] == 'unobserved':
                unobserved += 1
            else:
                targets[task.name][index] = c['target']
                observed[task.name][index] = True
                maturity[task.name][index] = utc(cm[sample][0]).to_datetime64() + np.timedelta64(1, 'ns')
        cv[task.name], rv[task.name] = np.asarray(c_values), np.asarray(r_values)
        support[task.name] = {'pending': pending, 'unobserved': unobserved,
                              'scored': int(observed[task.name].sum()), 'prediction_rows': n}
    meta = pd.DataFrame({'sample_id': samples, 'symbol': [cm[s][1] for s in samples],
        'available_at': pd.to_datetime([cm[s][0] for s in samples], utc=True)})
    data = TaskDataset(np.zeros((n, 1, 1), dtype=np.float32), targets, observed, maturity,
                       meta, ('comparison_container_never_trained',), spec.tasks)
    comparisons = {}
    for task in spec.tasks:
        record = paired_comparison(data, cv, rv, task.name, spec, family_size=PAIRWISE_FAMILY_SIZE)
        record['operational_development_improvement_supported'] = (
            task.name == spec.primary_task and record['predictive_claim_supported'])
        record.update(independent_confirmation=False, predictive_claim_supported=False,
                      evidence_stage='development_only', descriptive_only=task.name != spec.primary_task,
                      **support[task.name])
        comparisons[task.name] = record
    differences = {key: {'candidate': ci['policy'].get(key), 'reference': ri['policy'].get(key)}
                   for key in sorted(set(ci['policy']) | set(ri['policy']))
                   if ci['policy'].get(key) != ri['policy'].get(key)}
    segments = []
    for block, indices in meta.groupby(hour_ids(meta.available_at.dt.floor('h'))
                                       // spec.bootstrap_block_hours).groups.items():
        mask = np.zeros(n, dtype=bool)
        mask[np.asarray(indices)] = True
        segments.append({'calendar_block': int(block), 'start': meta.loc[mask, 'available_at'].min().isoformat(),
                         'end': meta.loc[mask, 'available_at'].max().isoformat(),
                         'tasks': _descriptive(mask, data, cv, rv)})
    result = {'status': 'development_comparison_complete', 'format': 'evolution-comparison-v1',
        'evidence_stage': 'development_only', 'independent_confirmation': False,
        'predictive_claim_supported': False, 'primary_task': spec.primary_task,
        'protocol_hash': fingerprint(ci['protocol']), 'schema': ci['schema'],
        'observed_data_hash': candidate['state']['visible_hash'],
        'information_cutoff': candidate['state']['last_timestamp'],
        'comparison_source_hash': file_hash(__file__), 'family_size': PAIRWISE_FAMILY_SIZE,
        'family': ['fixed_vs_refit', 'fixed_vs_evolve', 'refit_vs_evolve'],
        'primary_comparison': comparisons[spec.primary_task], 'tasks': comparisons,
        'paired_forecast_samples': n, 'full_selection_procedures_compared': True,
        'candidate': {'directory': str(candidate_dir), 'mode': ci['policy']['mode'],
                      'initial_digest': ci['initial_digest'], 'resources': _resources(candidate),
                      'event_head': candidate['summary']['event_head'],
                      'source_evidence': candidate['summary'].get('source_evidence'),
                      'frozen_identity_hash': fingerprint(ci)},
        'reference': {'directory': str(reference_dir), 'mode': ri['policy']['mode'],
                      'initial_digest': ri['initial_digest'], 'resources': _resources(reference),
                      'event_head': reference['summary']['event_head'],
                      'source_evidence': reference['summary'].get('source_evidence'),
                      'frozen_identity_hash': fingerprint(ri)},
        'policy_differences': differences,
        'other_program_differences': {key: {'candidate': ci.get(key), 'reference': ri.get(key)}
            for key in ('candidate_builder', 'memory_path') if ci.get(key) != ri.get(key)},
        'equal_policy_resource_ceilings': not any(key != 'mode' for key in differences),
        'equal_actual_resource_use_claimed': False,
        'per_symbol': {symbol: _descriptive((meta.symbol == symbol).to_numpy(), data, cv, rv)
                       for symbol in sorted(meta.symbol.unique())}, 'calendar_segments': segments,
        'limitations': ['Previously seen development outputs cannot establish independent confirmation.',
            'Every incumbent hour remains; failures are represented by retained-incumbent forecasts.',
            'Primary inference covers three predeclared procedure pairs; other tasks and strata are descriptive.',
            'Pending/censored eligibility must match; no complete-horizon assertion is made for them.',
            'Actual settlement maturity timestamps are not reconstructed from the scoring container.',
            'Resource summaries do not measure FLOPs, prediction/I/O time or independent API costs.']}
    immutable_json(output_path, result)
    return result
