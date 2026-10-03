"""Paired development-only comparisons of complete, verified research replays."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from factorlab.protocol import project_path, utc

from . import scoring
from .contracts import ExperimentSpec, TaskDataset
from .evidence import file_hash, immutable_json
from .learning import FittedResearchModel
from .replay import load_replay
from .scoring import paired_comparison, per_sample_loss


def _initial_model(directory: Path, replay: dict) -> FittedResearchModel:
    versions = replay['summary']['versions']
    if not versions or versions[0]['version'] != 0:
        raise ValueError('Replay has no verifiable initial model')
    model = FittedResearchModel.load(directory / versions[0]['checkpoint'])
    if (model.digest != replay['manifest']['initial_model_digest']
            or model.digest != versions[0]['model_digest']):
        raise ValueError('Replay initial model identity does not match its fitted checkpoint')
    return model


def _check_initial(candidate: FittedResearchModel, reference: FittedResearchModel) -> None:
    # Diagnostic wall time can differ when the CLI rebuilds an identical initial
    # model. Compare actual fitted state and its training contract, not that clock.
    fields = ('format', 'schema', 'mode', 'features', 'lookback', 'active_tasks',
              'target_scales', 'input_mean', 'input_std', 'settings', 'architecture',
              'volatility_floor', 'fit_feature_cutoff', 'fit_label_cutoff',
              'development_feature_cutoff', 'development_label_cutoff',
              'fit_label_cutoffs_by_task', 'development_label_cutoffs_by_task')
    if any(candidate.metadata.get(k) != reference.metadata.get(k) for k in fields):
        raise ValueError('Initial training programs, cutoffs or architectures differ')
    if set(candidate.state) != set(reference.state):
        raise ValueError('Initial fitted states differ')
    for key in candidate.state:
        ca, ra = candidate.state[key], reference.state[key]
        ca = ca.detach().cpu().numpy() if hasattr(ca, 'detach') else np.asarray(ca)
        ra = ra.detach().cpu().numpy() if hasattr(ra, 'detach') else np.asarray(ra)
        if not np.array_equal(ca, ra):
            raise ValueError('Initial fitted states differ; update effects would be confounded')


def _index(replay: dict) -> tuple[dict, dict]:
    predictions, settlements, prediction_sequences = {}, {}, {}
    for record in replay['records']:
        value = record['value']
        if record['event'] == 'prediction':
            sample = value['sample_id']
            if sample in predictions:
                raise ValueError('Duplicate replay prediction sample')
            if (utc(value['available_at']) != utc(record['timestamp'])
                    or not utc(replay['manifest']['start']) <= utc(value['available_at'])
                    < utc(replay['manifest']['end_exclusive'])):
                raise ValueError('Prediction timestamp is outside its declared replay origin')
            predictions[sample] = value
            prediction_sequences[sample] = record['sequence']
        elif record['event'] == 'settlement':
            key = (value['sample_id'], value['task'])
            if key in settlements:
                raise ValueError('Duplicate replay settlement')
            if utc(value['maturity']) > utc(record['timestamp']):
                raise ValueError('Replay settlement precedes label maturity')
            settlements[key] = value
    if any(sample not in predictions for sample, _ in settlements):
        raise ValueError('Settlement has no prediction')
    for key, settled in settlements.items():
        if settled['prediction_sequence'] != prediction_sequences[key[0]]:
            raise ValueError('Settlement refers to the wrong prediction')
    return predictions, settlements


def _cost(replay: dict) -> dict:
    summary = replay['summary']
    updates = summary['updates']
    fit_times = [v['metadata'].get('diagnostics', {}).get('elapsed_seconds')
                 for v in summary['versions']]
    valid_times = all(isinstance(t, (int, float)) and math.isfinite(t) and t >= 0
                      for t in fit_times)
    attempts = sum('attempted_version' in u for u in updates)
    failures = sum(u['status'] == 'failed_old_version_retained' for u in updates)
    return {'scheduled_updates': len(updates), 'update_attempts': attempts,
            'successful_updates': sum(u['status'] == 'refitted' for u in updates),
            'failed_updates': failures, 'fitted_versions': len(summary['versions']),
            'initial_fit_seconds': float(fit_times[0]) if valid_times else None,
            'known_successful_update_fit_seconds': float(sum(fit_times[1:])) if valid_times else None,
            'known_fit_seconds': float(sum(fit_times)) if valid_times else None,
            'failed_update_fit_seconds': None if failures else 0.0,
            'cost_complete': valid_times and not failures,
            'cost_scope': 'fitted-model training clocks; prediction and I/O are not measured'}


def compare_replays(candidate_dir: str | Path, reference_dir: str | Path,
                    output_file: str | Path) -> dict:
    """Compare the entire common replay scope; never promote it to confirmation.

    Unsupported heads are reported unavailable instead of trimming samples.
    Placeholder features below are only a container for the scoring interface;
    they are never passed to a trainer or a prediction model.
    """
    candidate_dir, reference_dir = project_path(candidate_dir), project_path(reference_dir)
    candidate, reference = load_replay(candidate_dir), load_replay(reference_dir)
    cm, rm = candidate['manifest'], reference['manifest']
    for key in ('protocol', 'protocol_sha256', 'schema', 'data_sha256', 'source_files',
                'architecture', 'start', 'end_exclusive', 'update_every_hours',
                'schedule_anchor', 'missing_ticks', 'operation_order',
                'early_stopping_fraction', 'adaptive_rule'):
        if cm.get(key) != rm.get(key):
            raise ValueError(f'Replays have different {key}')
    expected_scope = 'previously_seen_development_interval_not_independent_confirmation'
    if cm['scope'] != expected_scope or rm['scope'] != expected_scope:
        raise ValueError('Replay comparison requires declared development-only scope')
    # A changed scoring implementation must not silently reinterpret old losses.
    scoring_sources = [hash_value for key, hash_value in cm['source_files'].items()
                       if Path(key).name == 'scoring.py']
    if scoring_sources != [file_hash(scoring.__file__)]:
        raise ValueError('Replay scoring version differs from the current scoring implementation')
    spec = ExperimentSpec.from_dict(cm['protocol'])
    _check_initial(_initial_model(candidate_dir, candidate),
                   _initial_model(reference_dir, reference))
    cp, cs = _index(candidate)
    rp, rs = _index(reference)
    if set(cp) != set(rp) or set(cs) != set(rs):
        raise ValueError('Replay sample or settlement coverage differs; samples cannot be trimmed')
    if not cp:
        raise ValueError('No paired replay predictions')
    samples = sorted(cp, key=lambda s: (utc(cp[s]['available_at']), cp[s]['symbol']))
    n = len(samples)
    tasks = spec.tasks
    task_names = {t.name for t in tasks}
    for sample in samples:
        cv, rv = cp[sample], rp[sample]
        if (cv['symbol'] != rv['symbol'] or utc(cv['available_at']) != utc(rv['available_at'])
                or set(cv['predictions']) != task_names or set(rv['predictions']) != task_names):
            raise ValueError('Replay prediction identities, times or tasks differ')
    targets = {t.name: np.full(n, np.nan) for t in tasks}
    observed = {t.name: np.zeros(n, dtype=bool) for t in tasks}
    maturity = {t.name: np.full(n, np.datetime64('NaT'), dtype='datetime64[ns]') for t in tasks}
    c_predictions, r_predictions = {}, {}
    support = {}
    for task in tasks:
        c_values, r_values = [], []
        for i, sample in enumerate(samples):
            c_value, r_value = cp[sample]['predictions'][task.name], rp[sample]['predictions'][task.name]
            c_values.append(c_value)
            r_values.append(r_value)
            for value in (c_value, r_value):
                if value is not None:
                    # Validate every returned prediction, including pending and
                    # unobserved labels. Dummy targets cannot affect scoring.
                    dummy = np.asarray(value, dtype=float)[None]
                    if task.kind == 'first_event':
                        per_sample_loss(task, np.asarray([0.]), dummy)
                    elif (dummy.shape not in {(1,), (1, 1)} or not np.isfinite(dummy).all()
                          or (task.kind == 'volatility' and (dummy < 0).any())):
                        raise ValueError('Invalid replay regression prediction')
            key = (sample, task.name)
            if key not in cs:
                continue  # Both runs have the same pending task, retained in total coverage.
            c_settled, r_settled = cs[key], rs[key]
            if (c_settled['target'] != r_settled['target']
                    or utc(c_settled['maturity']) != utc(r_settled['maturity'])
                    or c_settled['predictive'] != task.predictive
                    or r_settled['predictive'] != task.predictive):
                raise ValueError('Paired task targets or maturity differ')
            statuses = {c_settled['status'], r_settled['status']}
            if not statuses <= {'scored', 'unobserved', 'unavailable_prediction'}:
                raise ValueError('Unknown replay settlement status')
            if 'unobserved' in statuses and statuses != {'unobserved'}:
                raise ValueError('Paired task label observation differs')
            maturity[task.name][i] = utc(c_settled['maturity']).to_datetime64()
            if statuses == {'unobserved'}:
                if c_settled['target'] is not None or c_settled['loss'] is not None or r_settled['loss'] is not None:
                    raise ValueError('Unobserved settlement cannot supply a score or target')
                continue
            target = c_settled['target']
            if target is None or not math.isfinite(target):
                raise ValueError('Observed settlement must have a finite target')
            targets[task.name][i] = target
            observed[task.name][i] = True
            for forecast, settled in ((c_value, c_settled), (r_value, r_settled)):
                if settled['status'] == 'unavailable_prediction':
                    if forecast is not None or settled['loss'] is not None:
                        raise ValueError('Unavailable settlement has a prediction or loss')
                else:
                    if forecast is None:
                        raise ValueError('Scored settlement has no prediction')
                    expected = float(per_sample_loss(task, np.asarray([target]),
                                                    np.asarray(forecast, dtype=float)[None])[0])
                    if (not isinstance(settled['loss'], (int, float)) or not math.isfinite(settled['loss'])
                            or not math.isclose(expected, settled['loss'], rel_tol=1e-8, abs_tol=1e-10)):
                        raise ValueError('Recorded replay loss differs from its frozen task loss')
        c_count, r_count = sum(v is not None for v in c_values), sum(v is not None for v in r_values)
        support[task.name] = {'candidate_prediction_support': c_count,
                              'reference_prediction_support': r_count,
                              'candidate_prediction_coverage': c_count / n,
                              'reference_prediction_coverage': r_count / n}
        if c_count == n and r_count == n:
            c_predictions[task.name] = np.asarray(c_values, dtype=float)
            r_predictions[task.name] = np.asarray(r_values, dtype=float)
    meta = pd.DataFrame({'sample_id': samples, 'symbol': [cp[s]['symbol'] for s in samples],
                         'available_at': pd.to_datetime([cp[s]['available_at'] for s in samples], utc=True)})
    data = TaskDataset(np.zeros((n, 1, 1), dtype=np.float32), targets, observed, maturity,
                       meta, ('scoring_placeholder_never_trained',), tasks)
    comparisons = {}
    for task in tasks:
        if task.name not in c_predictions:
            record = {'status': 'unavailable', 'reason': 'task_prediction_unavailable_on_full_replay_scope',
                      'support': 0, 'eligible_support': int(observed[task.name].sum()),
                      'coverage': 0.0, 'total_rows': n, 'predictive': task.predictive}
        else:
            record = paired_comparison(data, c_predictions, r_predictions, task.name, spec,
                                        family_size=len(tasks))
        record.update(support[task.name], evidence_stage='development_only',
                      independent_confirmation=False, predictive_claim_supported=False)
        comparisons[task.name] = record
    result = {'status': 'development_comparison_complete', 'evidence_stage': 'development_only',
              'independent_confirmation': False, 'protocol_hash': cm['protocol_sha256'],
              'data_hash': cm['data_sha256'], 'source_files': cm['source_files'],
              'comparison_source_hash': file_hash(__file__),
              'start': cm['start'], 'end_exclusive': cm['end_exclusive'],
              'candidate': {'directory': str(candidate_dir), 'update_mode': cm['update_mode'],
                            'initial_model_digest': cm['initial_model_digest'], 'cost': _cost(candidate)},
              'reference': {'directory': str(reference_dir), 'update_mode': rm['update_mode'],
                            'initial_model_digest': rm['initial_model_digest'], 'cost': _cost(reference)},
              'paired_prediction_samples': n, 'paired_settlements': len(cs),
              'comparisons': comparisons,
              'limitations': ['Previously seen development data cannot establish independent confirmation.',
                              'Intervals are conditional on the complete paired replay coverage.',
                              'Training clocks exclude prediction, I/O and failed-fit costs when unrecorded.']}
    immutable_json(output_file, result)
    return result
