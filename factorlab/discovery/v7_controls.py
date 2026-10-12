"""Predeclared equal-ceiling S research controls, never confirmatory evidence.

One immutable run per arm. Interrupted work resumes that same run; a failed arm
is reported rather than replaced by another attempt at obtaining a better score.
GPT is only used when a backend is explicitly supplied by the caller.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
from filelock import FileLock

from factorlab.protocol import fingerprint, project_path
from factorlab.time_utils import calendar_window, utc_index

from .convex_eval import fit_ridge, log_mse
from .evidence import file_hash, snapshot_source
from .joint_pool import FrozenPool, hour_equal_weights
from .known_library import FactorFrame
from .local_mutation import MonthlyBudget
from .rolling import RollingSpec, development_folds
from .v7_run import RunStore, V7Config, _verify_month_artifacts, run_month, source_files
from .v7_storage import immutable_record

ARMS = ('local', 'random', 'gpt')
PASS_RULE = {'relative_S_gain_strictly_above': .005, 'minimum_winning_fold_fraction': 2/3,
             'minimum_each_fold_nominal_pair_coverage': .95, 'scope': 'descriptive_S_only'}


def _passes(result, symbols):
    metric = result['S']['Pv']
    gain = metric['gain']
    coverage = [f['support_rows'] / (len(symbols) *
        (pd.Timestamp(f['score_end']) - pd.Timestamp(f['fit_cutoff'])).total_seconds() / 3600)
        for f in result['folds']]
    return (result['role'] == 'S' and len(coverage) == 3
            and gain is not None and math.isfinite(gain) and gain > PASS_RULE['relative_S_gain_strictly_above']
            and metric['winning_fold_fraction'] >= PASS_RULE['minimum_winning_fold_fraction']
            and all(PASS_RULE['minimum_each_fold_nominal_pair_coverage'] <= c <= 1 for c in coverage))


def _write(path, value):
    immutable_record(path, {'value': value, 'sha256': fingerprint(value)})


def _read(path):
    wrapper = json.loads(path.read_text('utf-8'))
    if wrapper.get('sha256') != fingerprint(wrapper['value']):
        raise ValueError('Control experiment artifact integrity failure')
    return wrapper['value']


def _data_manifest(bars, cutoff, symbols):
    if len(bars) > 100_000:
        raise ValueError('Control input exceeds the bounded 100000-row archive')
    frame = bars.loc[bars.symbol.isin(symbols)].copy()
    frame['available_at'], frame['open_time'] = utc_index(frame.available_at), utc_index(frame.open_time)
    frame = frame.loc[frame.available_at <= cutoff].sort_values(['symbol', 'open_time'])
    if frame.empty or set(frame.symbol) != set(symbols):
        raise ValueError('Control archive does not cover the frozen universe')
    columns = sorted(frame.columns)
    digest = hashlib.sha256()
    # Bounded streaming encoding: no full table -> nested lists -> JSON copy.
    for row in frame[columns].itertuples(index=False, name=None):
        digest.update(json.dumps([str(v) for v in row], separators=(',', ':')).encode())
        digest.update(b'\n')
    return {'rows': len(frame), 'columns': columns, 'sha256': digest.hexdigest(),
            'first_open': frame.open_time.min().isoformat(), 'last_open': frame.open_time.max().isoformat(),
            'cutoff': cutoff.isoformat(), 'symbols': list(symbols),
            'availability_mode': bars.attrs.get('availability_mode'),
            'feed_identity': bars.attrs.get('feed_identity')}


def _check_source(directory, expected):
    if source_files() != expected:
        raise ValueError('Frozen controls source changed; a new experiment identity is required')
    for path, digest in expected.items():
        archived = directory / 'source' / path
        if not archived.is_file() or file_hash(archived) != digest:
            raise ValueError('Controls archived source integrity failure')


def _metered_backend(directory, backend):
    calls = directory / 'provider_calls'
    calls.mkdir(parents=True, exist_ok=True)

    def call(packet, **kwargs):
        sequence = len(list(calls.glob('*-started.json')))
        if sequence >= 6:
            raise ValueError('Controls provider attempt ceiling reached')
        start = {'sequence': sequence, 'request_hash': packet.request_hash,
                 'input_characters': len(packet.text), 'input_utf8_bytes': len(packet.text.encode()),
                 'scope': 'injected_or_explicit_backend_report_not_hidden_token_measurement'}
        _write(calls / f'{sequence}-started.json', start)
        began = time.monotonic()
        result = {**start, 'status': 'failed', 'output_characters': None, 'usage': None}
        try:
            reply = backend(packet, **kwargs)
            result['status'] = 'returned'
            result['output_characters'] = len(reply.payload) if isinstance(reply.payload, str) else None
            usage = reply.usage if isinstance(reply.usage, dict) else {}
            result['usage'] = {key: value if type(value) is int and value >= 0 else None for key, value in
                ((key, usage.get(key)) for key in ('input_tokens', 'cached_input_tokens', 'output_tokens', 'reasoning_tokens'))}
            return reply
        finally:
            result['elapsed_seconds'] = time.monotonic() - began
            _write(calls / f'{sequence}-finished.json', result)
    return call


def _costs(directory, passing):
    starts = sorted((directory / 'provider_calls').glob('*-started.json'))
    calls = []
    for path in starts:
        started = _read(path)
        if path.name != f'{started["sequence"]}-started.json':
            raise ValueError('Provider call filename/index integrity failure')
        end = path.with_name(path.name.replace('-started', '-finished'))
        if end.exists():
            finished = _read(end)
            if any(finished.get(key) != value for key, value in started.items()):
                raise ValueError('Provider result does not match its reserved request')
        else:
            finished = {**started, 'status': 'interrupted', 'usage': None,
                        'output_characters': None, 'elapsed_seconds': None}
        calls.append(finished)
    known = [c['usage']['input_tokens'] for c in calls if c.get('usage') and c['usage']['input_tokens'] is not None]
    full = len(known) == len(calls) and bool(calls)
    total = sum(known) if known else None
    return {'provider_calls': len(calls), 'input_characters': sum(c['input_characters'] for c in calls),
            'output_characters_known_sum': sum(c['output_characters'] for c in calls if c['output_characters'] is not None),
            'known_reported_input_tokens': total, 'calls_with_reported_input_tokens': len(known),
            'all_input_usage_reported': full,
            'known_reported_output_tokens': sum(c['usage']['output_tokens'] for c in calls
                if c.get('usage') and c['usage']['output_tokens'] is not None) if any(
                    c.get('usage') and c['usage']['output_tokens'] is not None for c in calls) else None,
            'passing_per_1000_reported_input_tokens': 1000 * passing / total if full and total else None,
            'usage_scope': 'known_backend_reported_only; no hidden-token or character conversion estimate',
            'calls': calls}


def _duplicates(state):
    statuses = Counter(attempt['status'] for attempt in state['attempts'])
    materializations = Counter(r['status'] for r in state.get('materializations', []))
    seed_duplicates = sum(r['concrete_programs'] - r['unique_programs'] for r in state.get('agent_audit', [])
                          if r.get('status') == 'validated' and 'concrete_programs' in r and 'unique_programs' in r)
    parts = {'validated_reply_seed_duplicates': seed_duplicates,
             'template_duplicates': materializations.get('duplicate', 0),
             'evaluation_duplicate_attempts': statuses.get('duplicate', 0)}
    return {'duplicates': sum(parts.values()), 'duplicate_breakdown': parts,
            'duplicate_scope': 'known_exact_duplicates_only; unscored_ASTs_are_not_all_duplicates',
            'unscored_AST_count': max(0, state['budget']['counts']['ast'] - len(state['pool']['records']))}


def _attempt_counts(attempts):
    statuses = Counter(attempt['status'] for attempt in attempts)
    successful = {'scored', 'duplicate', 'measured_recovered_budget_retained'}
    return {'attempt_statuses': dict(statuses),
            'recovered_measured_results': statuses.get('measured_recovered_budget_retained', 0),
            'failures': sum(v for k, v in statuses.items() if k not in successful)}


def _readout(directory, config, month, *, spec=None):
    if not (directory / 'manifest.json').exists():
        return {'status': 'no_month_checkpoint', 'actual_budget': None, 'descriptive_S_passing': 0,
                'attempt_statuses': {}, 'unique_scored_formulas': 0, 'duplicates': 0, 'failures': 0,
                'costs': _costs(directory, 0)}
    store = RunStore(directory, config, spec or RollingSpec())
    try:
        state = store.get(month)
        if state is None:
            raise ValueError('Arm store exists without its frozen month')
        _verify_month_artifacts(directory, month, state)
        records = list(state['pool']['records'].values())
        passing = 0
        for record in records:
            result = record['result']
            if _passes(result, config.symbols):
                passing += 1
        materializations = Counter(r['status'] for r in state.get('materializations', []))
        return {'status': state['status'], 'actual_budget': state['budget'],
                **_attempt_counts(state['attempts']), 'unique_scored_formulas': len(records),
                **_duplicates(state), 'materializations': dict(materializations),
                'descriptive_S_passing': passing, 'costs': _costs(directory, passing),
                'monthly_seconds': state['spent_seconds'], 'data_id': state['data_id'],
                'pool_reference_hash': fingerprint(state['frozen_reference_pool']),
                'state_hash': fingerprint(state), 'formal_confirmation_enabled': False}
    finally:
        store.close()


def run_controls(directory, bars, cutoff, config=None, *, backend=None, arms=ARMS):
    """Run the predeclared arms once with equal ceilings, retaining every outcome."""
    config = config or V7Config(evaluation_mode='joint', profile_candidates=6)
    if (not isinstance(config, V7Config) or config.evaluation_mode != 'joint'
            or config.profile_candidates != 6 or config.formulas_per_family != 8
            or config.research_seconds != 7200 or config.formal_confirmation_enabled):
        raise ValueError('Controls require joint, six P structures, eight F/family and 7200s per arm')
    arms = tuple(arms)
    if not arms or len(arms) != len(set(arms)) or not set(arms) <= set(ARMS):
        raise ValueError('Freeze unique local/random/gpt arms before the first result')
    if 'gpt' in arms and not callable(backend):
        raise ValueError('GPT controls require an explicit injected or CodexBatchBackend instance')
    _, cutoff = calendar_window(cutoff)
    month = cutoff.strftime('%Y-%m')
    directory = project_path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    backend_type = None if backend is None else (
        f'{getattr(backend, "__module__", type(backend).__module__)}.'
        f'{getattr(backend, "__qualname__", type(backend).__qualname__)}')
    identity = {'version': 'v7-controls-v1', 'arms': list(arms), 'cutoff': cutoff.isoformat(),
                'config': asdict(replace(config, search_mode='local')), 'data': _data_manifest(bars, cutoff, config.symbols),
                'budget_ceiling': MonthlyBudget(month).to_dict()['limits'], 'pass_rule': PASS_RULE,
                'source_files': source_files(), 'backend_type': backend_type,
                'initial_reference_pool_hash': fingerprint(FrozenPool().to_dict()),
                'scope': 'exposed_S_descriptive_control_experiment', 'formal_claims_enabled': False,
                'arm_policy': 'single_immutable_month_per_arm_resume_same_run_only_no_best_restart'}
    with FileLock(str(directory / '.controls.lock'), timeout=0):
        manifest = directory / 'manifest.json'
        if not manifest.exists():
            if any((directory / 'arms').glob('*')) or any(directory.glob('*-outcome.json')):
                raise ValueError('New control experiment cannot inherit preexisting arm runs')
            archived = snapshot_source(directory / 'source')
            if archived['files'] != identity['source_files']:
                raise ValueError('Controls source changed while freezing')
        _write(manifest, identity)
        _check_source(directory, identity['source_files'])
        outcomes = {}
        for arm in arms:
            arm_directory = directory / 'arms' / arm
            arm_directory.mkdir(parents=True, exist_ok=True)
            arm_config = replace(config, search_mode=arm)
            outcome_path = directory / f'{arm}-outcome.json'
            if outcome_path.exists():
                outcome = _read(outcome_path)
                if outcome['readout'] != _readout(arm_directory, arm_config, month):
                    raise ValueError('Completed arm state/costs differ from its frozen outcome')
                outcomes[arm] = outcome
                continue
            _check_source(directory, identity['source_files'])
            invocation = len(list(arm_directory.glob('invocation-*.json')))
            if invocation >= 3:
                raise ValueError('Same-run recovery invocation cap reached; no fresh replacement arm')
            _write(arm_directory / f'invocation-{invocation}.json',
                   {'arm': arm, 'month': month, 'invocation': invocation, 'frozen_identity': fingerprint(identity)})
            began = time.monotonic()
            failure = None
            try:
                run_month(arm_directory, bars, cutoff, config=arm_config,
                          backend=_metered_backend(arm_directory, backend) if arm == 'gpt' else None)
            except Exception as exc:  # noqa: BLE001 - failed arms remain outcomes, never replacements
                failure = {'error_type': type(exc).__name__, 'message': str(exc)[:300]}
            _check_source(directory, identity['source_files'])
            outcome = {'arm': arm, 'failure': failure, 'invocation_elapsed_seconds': time.monotonic() - began,
                       'readout': _readout(arm_directory, arm_config, month)}
            _write(outcome_path, outcome)
            outcomes[arm] = outcome
        reference_hashes = {outcome['readout']['pool_reference_hash'] for outcome in outcomes.values()
                            if 'pool_reference_hash' in outcome['readout']}
        if reference_hashes and reference_hashes != {identity['initial_reference_pool_hash']}:
            raise ValueError('Control arms must share the frozen initial reference pool')
        report = {'identity_hash': fingerprint(identity), 'identity': identity, 'arms': outcomes,
                  'formal_claims_enabled': False, 'V_queries': 0, 'C_queries': 0,
                  'interpretation': 'equal maximum budgets; actual usage differs and all S results are exposed; '
                                    'no GPT superiority, alpha, power, NN permission or profitability claim'}
        _write(directory / 'report.json', report)
        return report


def known_component_recovery_probe(data, cutoff, *, component='rv_24'):
    """Fixed K0 removal/restoration probe. Point recovery is not known population gain."""
    frame = data.frame
    if component not in frame.feature_names:
        raise ValueError('Probe requires a registered K0 component')
    keep = [i for i, name in enumerate(frame.feature_names) if name != component]
    reduced = FactorFrame(frame.values[:, keep], frame.metadata, tuple(frame.feature_names[i] for i in keep),
                          tuple(frame.bypass_mask[i] for i in keep), fingerprint({'base': frame.schema_hash, 'drop': component}))
    folds = []
    for fold in development_folds(cutoff, spec=data.spec):
        train = data.train_mask(fold['fit_cutoff'])
        score = data.score_mask(fold['fit_cutoff'], fold['score_end'], fold['score_end'])
        common = frame.valid_rows & reduced.valid_rows
        train, score = train & common, score & common
        if train.sum() < data.spec.minimum_train_rows or not score.any():
            raise ValueError('Insufficient paired probe support')
        weights = hour_equal_weights(frame.metadata.loc[train])
        score_weights = hour_equal_weights(frame.metadata.loc[score])
        losses = {}
        for name, design in [('removed', reduced), ('restored', frame)]:
            model = fit_ridge(design.take(train), data.target[train], ridge=data.spec.ridge, sample_weight=weights)
            losses[name] = log_mse(data.target[score], model.predict(design.take(score)), sample_weight=score_weights)
        folds.append({**fold, 'losses': losses, 'train_rows': int(train.sum()), 'score_rows': int(score.sum()),
                      'support_hash': fingerprint(np.flatnonzero(score).tolist()), 'solves_per_arm': 1})
    removed = float(np.mean([f['losses']['removed'] for f in folds]))
    restored = float(np.mean([f['losses']['restored'] for f in folds]))
    return {'component': component, 'folds': folds, 'solves': {'removed': 3, 'restored': 3},
            'relative_gain': 1 - restored / removed if removed > 0 else None,
            'aggregation': 'equal_fold_mean_of_hour_weighted_log_MSE',
            'scope': 'fixed_S_known_component_recovery_diagnostic',
            'known_population_effect': False, 'opens_NN_or_confirmation_gate': False}
