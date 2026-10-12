"""Recoverable, bounded OHLC -> fitted S/V/C calibration experiments.

This is a real fitted-pipeline simulator, not a loss-array substitute. Its
proposal protocol is a frozen local list; it does not certify GPT, the six-month
A0 adoption controller, or K0-versus-HAR claims. In particular a DGP amplitude
is NOT the population logMSE effect of a fitted learner. Until that truth bridge
is validated, reports remain diagnostics even with excellent Monte Carlo rates.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sqlite3
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import psutil
from filelock import FileLock
from threadpoolctl import threadpool_limits

from factorlab.protocol import fingerprint, project_path

from .causal_numpy import execute_formula, formula_identity
from .confirmatory import ConfirmationRule, paired_effect_interval, wilson_interval
from .convex_eval import fit_ridge
from .evidence import file_hash
from .exposure import select_on_validation
from .factor_pool import evaluate_formula, fit_formula
from .known_library import FEATURE_NAMES
from .rolling import RollingSpec, build_rolling_data
from .v7_storage import immutable_record

EFFECTS = (0., .003, .005, .01, .02)
SCENARIOS = ('persistent_heavy', 'regime', 'missing_history')
S_CUTOFF = '2021-08-01T00:00:00Z'
V_END = '2021-09-01T00:00:00Z'
C_END = '2021-12-01T00:00:00Z'
SCOPE = 'bounded_local_fitted_factor_S_V_C_v1'


@dataclass(frozen=True)
class CalibrationPlan:
    calibration_repetitions: int = 2
    validation_repetitions: int = 2
    seed: int = 170104
    effects: tuple[float, ...] = EFFECTS
    scenarios: tuple[str, ...] = SCENARIOS
    candidates: int = 4
    profile_candidates: int = 0
    lookback: int = 48
    max_solves_per_trial: int = 600
    trial_seconds: int = 300
    memory_mb: int = 768
    minimum_formal_repetitions: int = 1000
    rule: ConfirmationRule = field(default_factory=ConfirmationRule)

    def __post_init__(self):
        for name, lower, upper in (
            ('calibration_repetitions', 1, 2000), ('validation_repetitions', 1, 2000),
            ('seed', 0, 2**32 - 1), ('candidates', 2, 4), ('profile_candidates', 0, 1),
            ('lookback', 2, 512), ('max_solves_per_trial', 1, 1600),
            ('trial_seconds', 1, 7200), ('memory_mb', 128, 2048),
            ('minimum_formal_repetitions', 1000, 2000),
        ):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f'Invalid bounded calibration {name}')
        if tuple(self.effects) != EFFECTS or tuple(self.scenarios) != SCENARIOS:
            raise ValueError('Freeze all five effect targets and three stress scenarios')
        if not isinstance(self.rule, ConfirmationRule) or self.rule.horizon_hours != 12:
            raise ValueError('Calibration currently isolates volatility12h')
        if self.profile_candidates and self.candidates != 4:
            raise ValueError('The frozen parameterized proposal is candidate four')

    def to_dict(self):
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_dict(cls, value):
        value = copy.deepcopy(value)
        value['effects'], value['scenarios'] = tuple(value['effects']), tuple(value['scenarios'])
        value['rule'] = ConfirmationRule(**value['rule'])
        return cls(**value)


def candidate_programs():
    """Correlated fixed proposals, including exact redundant/null controls."""
    def program(node):
        return {'version': 'causal-operator-program-v1', 'feature_names': list(FEATURE_NAMES),
                'parameters': {}, 'outputs': [node]}
    def feature(name):
        return {'op': 'feature', 'name': name}
    ratio = {'op': 'safe_div', 'left': feature('rv_6'), 'right': feature('rv_24')}
    programs = [program({'op': 'last', 'arg': ratio}),
                program({'op': 'mean', 'arg': copy.deepcopy(ratio)}),
                program({'op': 'last', 'arg': {'op': 'mul', 'left': feature('rv_24'),
                                               'right': feature('rv_6')}}),
                program({'op': 'last', 'arg': {'op': 'ema', 'arg': feature('rv_6'),
                                               'alpha': {'op': 'param', 'name': 'alpha'}}})]
    programs[-1]['parameters'] = {'alpha': {'initial': .3, 'lower': .05, 'upper': .95}}
    nulls = [program({'op': 'const', 'value': value}) for value in (0., 1.)]
    return programs, nulls


def calibration_binding(plan):
    """Bind source, environment lock, proposal families and every budget/rule."""
    root = project_path('.')
    paths = [*root.glob('factorlab/**/*.py'), root / 'pyproject.toml', root / 'uv.lock']
    sources = {str(p.relative_to(root)).replace('\\', '/'): file_hash(p) for p in sorted(paths)}
    programs, nulls = candidate_programs()
    return {'version': 1, 'scope': SCOPE, 'plan': plan.to_dict(), 'sources': sources,
            'test_only': False, 'evaluator_id': 'actual_pipeline_v1',
            'rolling': RollingSpec().digest, 'rule': plan.rule.digest,
            'candidate_family': fingerprint({'programs': programs[:plan.candidates], 'nulls': nulls}),
            'pipeline': 'actual_K0_ridge_fixed_formula_S_three_folds_V_one_C_three_months',
            'covered_claim_types': ['fixed_local_factor_increment'],
            'excluded_claim_types': ['library_skill', 'adaptive_A0_algorithm', 'GPT_search'],
            'claim_stress_test': 'five_correlated_factor_slots_not_the_V7_five_distinct_claims',
            'truth_policy': 'DGP_strength_is_not_learner_effect_no_unverified_truth_certificate'}


def synthetic_bars(*, seed, scenario, effect_target, phase='calibration'):
    """Causal AR volatility, t innovations and overlapping 12h targets downstream.

    Validation uses disjoint random streams AND a held-out nuisance setting.
    The effect target controls signal amplitude only; achieved learner effects
    must be reported separately and cannot be asserted from this argument.
    """
    if scenario not in SCENARIOS or effect_target not in EFFECTS or phase not in {'calibration', 'validation'}:
        raise ValueError('Unknown frozen synthetic scenario')
    rng = np.random.default_rng(seed)
    stamps = pd.date_range('2021-01-01', '2021-12-03', inclusive='left', freq='h', tz='UTC')
    n = len(stamps)
    phi, df = (.94, 5) if phase == 'calibration' else (.975, 4)
    shocks = rng.standard_t(df, n) / math.sqrt(df / (df - 2))
    vol_shocks = rng.normal(0, .05, n)
    log_returns, state, fast, slow = np.zeros(n), 0., 4e-6, 4e-6
    sigma = np.empty(n)
    strength = 5 * math.sqrt(effect_target)
    for i in range(n):
        state = phi * state + vol_shocks[i]
        regime = (0. if scenario == 'persistent_heavy' else
                  (.35 if n // 3 <= i < 2 * n // 3 else -.2))
        # All signal inputs are from strictly earlier returns.
        signal = math.tanh(fast / max(slow, 1e-12) - 1)
        sigma[i] = .002 * math.exp(np.clip(state + regime + strength * signal, -3, 3))
        log_returns[i] = sigma[i] * shocks[i]
        fast = .8 * fast + .2 * log_returns[i] ** 2
        slow = .99 * slow + .01 * log_returns[i] ** 2
    close = np.exp(4 + np.cumsum(log_returns))
    opened = np.r_[math.exp(4), close[:-1]]
    spread = sigma * (1 + np.abs(rng.normal(0, .3, n)))
    bars = pd.DataFrame({'symbol': 'SYNTH', 'open_time': stamps,
        'available_at': stamps + pd.Timedelta(hours=1, seconds=1), 'open': opened, 'close': close,
        'high': np.maximum(opened, close) * np.exp(spread),
        'low': np.minimum(opened, close) * np.exp(-spread)})
    if scenario == 'missing_history':
        # A real gap creates the same 720h warmup loss as production. This tests
        # discovery attrition; do not erase the gap or shorten the denominator.
        first = pd.Timestamp('2021-05-09' if phase == 'calibration' else '2021-05-19', tz='UTC')
        bars = bars.loc[~bars.open_time.between(first, first + pd.Timedelta(hours=2))].reset_index(drop=True)
    bars.attrs['availability_mode'] = 'assumed_historical_delay'
    bars.attrs['feed_identity'] = fingerprint({'seed': seed, 'scenario': scenario,
                                              'phase': phase, 'effect_target': effect_target})
    return bars


class CalibrationBudgetExceeded(RuntimeError):
    pass


class _TrialBudget:
    def __init__(self, plan, persist):
        self.plan, self.persist = plan, persist
        self.solves, self.started, self.peak_rss = 0, time.monotonic(), 0

    def check(self):
        self.peak_rss = max(self.peak_rss, psutil.Process().memory_info().rss)
        if self.peak_rss > self.plan.memory_mb * 1024**2:
            raise CalibrationBudgetExceeded('memory_limit')
        if time.monotonic() - self.started > self.plan.trial_seconds:
            raise CalibrationBudgetExceeded('wall_time_limit')

    def consume(self, kind):
        if kind != 'solve':
            raise ValueError('Calibration evaluator only reserves convex solves')
        self.check()
        if self.solves >= self.plan.max_solves_per_trial:
            raise CalibrationBudgetExceeded('solve_limit')
        self.solves += 1
        self.persist({'event': 'solve_reserved', 'solves': self.solves})


def _loss(y, prediction):
    return np.square(np.log(y + 1e-8) - np.log(prediction + 1e-8))


def _score_month(data, program, start, end, *, plan, budget, profile=False, as_of=None):
    """Identical training and prediction support; cold refits of both pipelines."""
    raw = execute_formula(program, data.frame, lookback=plan.lookback)
    train = data.train_mask(start) & np.isfinite(raw)
    if train.sum() < data.spec.minimum_train_rows:
        raise ValueError('insufficient_common_training_support')
    budget.consume('solve')
    baseline = fit_ridge(data.frame.take(train), data.target[train], ridge=data.spec.ridge)
    bundle, values, support, receipt = fit_formula(data, start, program, channel='S',
        lookback=plan.lookback, profile=profile, budget=budget)
    if not np.array_equal(train, support):
        raise ValueError('common_training_support_mismatch')
    score = data.score_mask(start, end, as_of or end) & np.isfinite(raw) & np.isfinite(values)
    B, y = data.frame.take(score), data.target[score]
    if not len(y):
        raise ValueError('empty_prediction_support')
    origins = data.frame.metadata.available_at[score]
    return origins, _loss(y, baseline.predict(B)), _loss(y, bundle.predict(B, values[score])), {
        'start': str(start), 'end': str(end), 'fit_cutoff': str(start), 'train': int(train.sum()),
        'support': int(score.sum()), 'model': bundle.digest, 'reference_model': baseline.digest,
        'pipeline': bundle.pipeline_id, 'profile': receipt,
        'input_hash': data.training_identity(start)['visible_data_hash']}


def actual_pipeline_trial(plan, job, emit):
    """Production evaluator; every value originates in synthetic OHLC fits."""
    budget = _TrialBudget(plan, emit)
    budget.check()
    bars = synthetic_bars(seed=job['seed'], scenario=job['scenario'],
                          effect_target=job['effect'], phase=job['phase'])
    data = build_rolling_data(bars, RollingSpec())
    emit({'event': 'data_built', 'rows': len(bars), 'feed': bars.attrs['feed_identity'],
          'feature_shape': list(data.frame.values.shape), 'data_hash': fingerprint({
              'bars': bars.astype(str).to_dict('list'), 'feed': bars.attrs['feed_identity']})})
    proposals, nulls = candidate_programs()
    families = {'signal': proposals[:plan.candidates], 'null': nulls}
    caches, claims = {}, []
    # Two exact redundant-factor null slots and three correlated search slots.
    # They share candles and candidates: no independent-claim assumption.
    for family in ('null', 'signal'):
        programs = families[family]
        search, failures = [], []
        for i, program in enumerate(programs):
            identifier = fingerprint(formula_identity(program, plan.lookback))
            profile = family == 'signal' and bool(program['parameters']) and plan.profile_candidates > 0
            try:
                result = evaluate_formula(data, S_CUTOFF, program, lookback=plan.lookback,
                                          budget=budget, profile=profile)
                gain = result['channels']['S']['gain']
                if gain is not None:
                    search.append((gain, identifier, program, profile))
                emit({'event': 'candidate_S', 'family': family, 'candidate': identifier,
                      'result_hash': fingerprint(result), 'folds': result['folds'], 'channels': result['channels']})
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                failures.append({'candidate': identifier, 'stage': 'S', 'error': str(exc)[:300]})
                emit({'event': 'candidate_failed', **failures[-1]})
        short = sorted(search, key=lambda row: (-row[0], row[1]))[:3]
        emit({'event': 'shortlist_frozen', 'family': family, 'candidates': [r[1] for r in short]})
        validation, by_id = [], {}
        for _, identifier, program, profile in short:
            try:
                origin, base, candidate, receipt = _score_month(data, program, S_CUTOFF, V_END,
                    plan=plan, budget=budget, profile=profile)
                eligible = int((pd.Timestamp(V_END) - pd.Timestamp(S_CUTOFF)) / pd.Timedelta(hours=1))
                row = {'role': 'V', 'common_support': True, 'candidate_id': identifier,
                       'coverage': len(origin) / eligible,
                       'gain': float(1 - candidate.mean() / base.mean()) if base.mean() > 0 else None,
                       'complexity': len(json.dumps(program, sort_keys=True))}
                validation.append(row)
                by_id[identifier] = (program, profile)
                emit({'event': 'candidate_V', 'family': family, 'readout': row, 'fit': receipt})
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                failures.append({'candidate': identifier, 'stage': 'V', 'error': str(exc)[:300]})
                emit({'event': 'candidate_failed', **failures[-1]})
        chosen = select_on_validation(validation)
        if chosen is None:
            caches[family] = {'selected': None, 'accepted': False, 'truth': 'exact_null' if family == 'null' else 'unidentified',
                              'failures': failures, 'outcome': 'all_candidates_failed' if not short else 'no_V_candidate_passed',
                              'effect_estimate': None, 'base_sum': None, 'candidate_sum': None}
            continue
        program, profile = by_id[chosen['candidate_id']]
        emit({'event': 'C_frozen', 'family': family, 'candidate': chosen['candidate_id'],
              'formula': fingerprint(program), 'profile': profile, 'start': V_END, 'end': C_END,
              'rule_id': plan.rule.digest})
        origins, bases, candidates, monthly = [], [], [], []
        # Formula, fitting recipe and rule are frozen; coefficients refit monthly.
        # No C outcome is used to alter candidates or the interval endpoint.
        for start in pd.date_range(V_END, C_END, inclusive='left', freq='MS'):
            end = start + pd.DateOffset(months=1)
            as_of = pd.Timestamp(C_END) + pd.Timedelta(hours=plan.rule.horizon_hours + 24)
            origin, base, candidate, receipt = _score_month(data, program, start, end,
                plan=plan, budget=budget, profile=profile, as_of=as_of)
            origins.extend(origin.tolist())
            bases.extend(base.tolist())
            candidates.extend(candidate.tolist())
            monthly.append(receipt)
            emit({'event': 'C_month_fitted', 'family': family, **receipt})
        eligible = int((pd.Timestamp(C_END) - pd.Timestamp(V_END)) / pd.Timedelta(hours=1))
        interval = paired_effect_interval(origins, bases, candidates, rule=plan.rule,
                                          eligible_count=eligible, calibration_passed=False)
        # Equivalent input scaling yields exactly d=Lb-Lc under the same block
        # machinery. Only its d interval is exposed; its rescaled relative gain
        # is intentionally discarded. This is a sensitivity diagnostic, not a
        # second formal claim or a rule change after observing outcomes.
        zero = paired_effect_interval(origins, np.asarray(bases) / (1 - plan.rule.delta0), candidates,
            rule=plan.rule, eligible_count=eligible, calibration_passed=False)
        caches[family] = {'selected': chosen['candidate_id'],
            'accepted': interval.get('diagnostic_status') == 'supported_target',
            'zero_sensitivity_accepted': zero.get('diagnostic_status') == 'supported_target',
            'zero_sensitivity_interval': zero['interval'],
            'truth': 'exact_null' if family == 'null' else 'unidentified',
            'outcome': 'scored', 'effect_estimate': interval['relative_gain'],
            'base_sum': float(np.sum(bases)), 'candidate_sum': float(np.sum(candidates)),
              'interval': interval, 'monthly': monthly, 'failures': failures}
        emit({'event': 'C_fixed_endpoint', 'family': family, 'interval': interval})
    for slot, family in enumerate(('null', 'signal', 'signal', 'null', 'signal')):
        claims.append({'slot': slot, 'family': family, **caches[family]})
    budget.check()
    return {'evaluator': 'actual_pipeline', 'scope': SCOPE, 'claims': claims,
            'solves': budget.solves, 'peak_rss_bytes': budget.peak_rss,
            'elapsed_seconds': time.monotonic() - budget.started,
            'effect_target': job['effect'], 'effect_is_known_truth': False,
            'proposal_protocol': 'fixed_local_list_no_GPT',
            'joint_claim_dependence': 'shared_data_and_identical_family_winners',
            'limitations': ['DGP amplitude is not a known fitted-pipeline logMSE effect',
                           'No actual delayed worker activation is simulated',
                           'Does not simulate six-month A0 adoption or library skill claim']}


def _jobs(plan):
    for phase_index, phase in enumerate(('calibration', 'validation')):
        repetitions = getattr(plan, f'{phase}_repetitions')
        for scenario_index, scenario in enumerate(plan.scenarios):
            for effect_index, effect in enumerate(plan.effects):
                for repetition in range(repetitions):
                    # Domains are injective before SeedSequence; save full key.
                    key = [plan.seed, phase_index, scenario_index, effect_index, repetition]
                    seed = int(np.random.SeedSequence(key).generate_state(1, dtype=np.uint64)[0])
                    yield {'id': f'{phase}:{scenario}:{effect_index}:{repetition}', 'phase': phase,
                           'scenario': scenario, 'effect': effect, 'repetition': repetition,
                           'seed': seed, 'seed_key': key}


def _read_trials(db):
    for identity, text, digest in db.execute('SELECT id,value,hash FROM trials ORDER BY id'):
        value = json.loads(text)
        if value['job']['id'] != identity or fingerprint(value) != digest:
            raise ValueError('Calibration trial integrity mismatch')
        yield value


def _trials_hash(db, phase=None):
    digest = hashlib.sha256(b'[')
    first = True
    for row in _read_trials(db):
        if phase is not None and row['job']['phase'] != phase:
            continue
        if not first:
            digest.update(b',')
        first = False
        digest.update(json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
    digest.update(b']')
    return digest.hexdigest()


def _aggregate(plan, rows, *, test_only):
    """Trial is the Bernoulli unit; never count five correlated slots as n=5."""
    curve = []
    groups = {(p, s, e): {'n': 0, 'finished': 0, 'false': 0, 'unresolved': 0,
                         'possible_false': 0, 'accepts': 0, 'zero_accepts': 0, 'gain_sum': 0., 'gain_n': 0,
                         'slots': [{'slot': i, 'accepts': 0, 'known_null_false_accepts': 0,
                                    'unresolved_accepts': 0, 'no_selection': 0, 'failed_replicates': 0}
                                   for i in range(5)]}
              for p in ('calibration', 'validation') for s in plan.scenarios for e in plan.effects}
    count, reserved = 0, False
    for row in rows:
        count += 1
        reserved |= row['status'] == 'reserved'
        group = groups[row['job']['phase'], row['job']['scenario'], row['job']['effect']]
        group['n'] += 1
        if row['status'] != 'complete':
            group['possible_false'] += 1
            for slot in group['slots']:
                slot['failed_replicates'] += 1
            continue
        claims = row['result']['claims']
        group['finished'] += 1
        group['false'] += int(any(c['accepted'] and c['truth'] == 'exact_null' for c in claims))
        group['unresolved'] += int(any(c['accepted'] and c['truth'] != 'exact_null' for c in claims))
        group['possible_false'] += int(any(c['accepted'] for c in claims))
        group['accepts'] += int(any(c['accepted'] and c['family'] == 'signal' for c in claims))
        group['zero_accepts'] += int(any(c.get('zero_sensitivity_accepted', False) and c['family'] == 'signal' for c in claims))
        for claim, slot in zip(claims, group['slots'], strict=True):
            slot['accepts'] += int(claim['accepted'])
            slot['known_null_false_accepts'] += int(claim['accepted'] and claim['truth'] == 'exact_null')
            slot['unresolved_accepts'] += int(claim['accepted'] and claim['truth'] != 'exact_null')
            slot['no_selection'] += int(claim.get('selected') is None)
        estimate = claims[1]['effect_estimate']
        if estimate is not None:
            group['gain_sum'] += estimate
            group['gain_n'] += 1
    complete = count == sum(1 for _ in _jobs(plan)) and not reserved
    for phase in ('calibration', 'validation'):
        for scenario in plan.scenarios:
            for effect in plan.effects:
                group = groups[phase, scenario, effect]
                false, unresolved = group['false'], group['unresolved']
                # Missing/failed repetitions count against acceptance, and as a
                # possible false rejection for conservative error upper bounds.
                failures = group['n'] - group['finished']
                conservative, any_accept, n = group['possible_false'], group['accepts'], group['n']
                curve.append({'phase': phase, 'scenario': scenario, 'effect_target': effect,
                    'effect_target_role': 'null_or_boundary_target' if effect <= plan.rule.delta0 else 'alternative_target',
                    'repetitions': n, 'completed': group['finished'], 'resource_or_interrupted_failures': failures,
                    'known_null_false_trials': false, 'unresolved_truth_accept_trials': unresolved,
                    'conservative_possible_false_trials': conservative,
                    'known_null_fwer': false / n if n else None,
                    'known_null_fwer_one_sided_95_upper': wilson_interval(false, n, confidence=.90)[1] if n else None,
                    'fwer_one_sided_95_upper': wilson_interval(conservative, n, confidence=.90)[1] if n else None,
                    'acceptance_rate': any_accept / n if n else None,
                    'zero_sensitivity_acceptance_rate': group['zero_accepts'] / n if n else None,
                    'acceptance_one_sided_95_lower': wilson_interval(any_accept, n, confidence=.90)[0] if n else None,
                    'observed_C_gain_mean_selected_only': group['gain_sum'] / group['gain_n'] if group['gain_n'] else None,
                    'claim_slots': [{**slot, 'acceptance_rate': slot['accepts'] / n if n else None,
                        'conservative_false_one_sided_95_upper': wilson_interval(
                            slot['accepts'] + slot['failed_replicates'], n, confidence=.90)[1] if n else None}
                        for slot in group['slots']],
                    'observed_gain_is_truth_estimate': False})
    validation = [c for c in curve if c['phase'] == 'validation']
    enough = complete and all(c['repetitions'] >= plan.minimum_formal_repetitions for c in validation)
    error_ok = enough and all(c['fwer_one_sided_95_upper'] <= .05 for c in validation)
    power_ok = enough and all(c['acceptance_one_sided_95_lower'] >= .8 for c in validation if c['effect_target'] > .005)
    # No numerical rate can repair unknown population truth or expand scope.
    # Fail closed until a separately validated truth bridge is implemented.
    return {'scope': SCOPE, 'complete': complete, 'test_only': test_only, 'curve': curve,
        'sufficient_repetitions': enough, 'conservative_error_bound_passed': error_ok,
        'acceptance_rate_bound_passed': power_ok, 'effect_truth_calibrated': False,
        'formal_gate_eligible': False, 'full_V7_five_claim_pipeline_calibrated': False,
        'covered_claim_types': ['fixed_local_factor_increment'],
        'reasons': ['population_effect_truth_bridge_not_validated',
                    'library_skill_and_A0_claims_not_simulated'] + (['test_evaluator'] if test_only else [])
                    + ([] if enough else ['insufficient_independent_validation_repetitions']),
        'interval_scope': 'per_scenario_pointwise_MC_Wilson_not_simultaneous',
        'power_scope': 'acceptance_diagnostic_only_until_achieved_population_effect_identified',
        'no_C_selection_is_zero_acceptance_not_a_dropped_trial': True}


def run_pipeline_calibration(directory, plan=None, *, evaluator=None, evaluator_id=None, max_new_trials=1):
    """Run at most N new, nonrefundable trials. Reinvoke to continue the same job.

    A custom evaluator is useful for failure tests, but its certificate is
    permanently test-only. Reserved interrupted trials are consumed as failures;
    their seeds are never retried until a favorable outcome appears.
    """
    plan = CalibrationPlan() if plan is None else plan
    if type(max_new_trials) is not int or not 1 <= max_new_trials <= 10000:
        raise ValueError('A finite max_new_trials of 1..10000 is required')
    if evaluator is not None and (not isinstance(evaluator_id, str) or not evaluator_id):
        raise ValueError('A test evaluator requires an explicit frozen identity')
    directory = project_path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    binding = calibration_binding(plan)
    binding.update(test_only=evaluator is not None, evaluator_id=evaluator_id if evaluator else 'actual_pipeline_v1')
    identity = fingerprint(binding)
    with FileLock(str(directory) + '.lock'):
        immutable_record(directory / 'identity.json', {'binding': binding, 'id': identity})
        db = sqlite3.connect(directory / 'calibration.sqlite')
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA synchronous=FULL')
        db.executescript('''CREATE TABLE IF NOT EXISTS trials(id TEXT PRIMARY KEY,value TEXT,hash TEXT);
            CREATE TABLE IF NOT EXISTS receipts(trial TEXT,seq INTEGER,value TEXT,hash TEXT,PRIMARY KEY(trial,seq));
            CREATE TABLE IF NOT EXISTS journal(seq INTEGER PRIMARY KEY,value TEXT,hash TEXT);''')

        def save(value):
            with db:
                head = db.execute('SELECT seq,hash FROM journal ORDER BY seq DESC LIMIT 1').fetchone()
                event = {'sequence': head[0] + 1 if head else 0, 'previous': head[1] if head else '0' * 64,
                         'trial': value['job']['id'], 'state_hash': fingerprint(value)}
                db.execute('INSERT OR REPLACE INTO trials VALUES(?,?,?)',
                    (value['job']['id'], json.dumps(value, sort_keys=True, allow_nan=False), fingerprint(value)))
                db.execute('INSERT INTO journal VALUES(?,?,?)',
                           (event['sequence'], json.dumps(event, sort_keys=True), fingerprint(event)))

        def verify_receipts():
            for trial, seq, text, digest in db.execute('SELECT trial,seq,value,hash FROM receipts ORDER BY trial,seq'):
                value = json.loads(text)
                if value['trial'] != trial or value['sequence'] != seq or fingerprint(value) != digest:
                    raise ValueError('Calibration step receipt integrity mismatch')

        try:
            _verify_journal(db)
            verify_receipts()
            for row in _read_trials(db):
                if row['status'] == 'reserved':
                    row.update(status='interrupted_consumed', error='process_interrupted_no_refund')
                    save(row)
            existing = {r['job']['id'] for r in _read_trials(db)}
            count = 0
            for job in _jobs(plan):
                if job['id'] in existing:
                    continue
                if count == max_new_trials:
                    break
                # The validation manifest is frozen only after calibration has
                # ended, before any validation data are generated or inspected.
                if job['phase'] == 'validation':
                    immutable_record(directory / 'validation_freeze.json', {
                        'binding': identity, 'calibration_trials_hash': _trials_hash(db, 'calibration'),
                        'rule': plan.rule.to_dict(), 'selection': 'unchanged_predeclared_local_protocol',
                        'validation_seed_domain': [plan.seed, 1], 'nuisance_settings': 'held_out_AR_and_t_df'})
                row = {'job': job, 'binding': identity, 'status': 'reserved',
                       'receipt_count': 0, 'receipt_head': '0' * 64}
                save(row)  # Reservation is durable before generating data/fitting.
                sequence = 0

                def emit(payload, job=job, row=row):
                    nonlocal sequence
                    record = {'trial': job['id'], 'sequence': sequence,
                              'previous': row['receipt_head'], 'receipt': payload}
                    with db:
                        db.execute('INSERT INTO receipts VALUES(?,?,?,?)',
                            (job['id'], sequence, json.dumps(record, sort_keys=True, allow_nan=False), fingerprint(record)))
                        row['receipt_count'] = sequence + 1
                        row['receipt_head'] = fingerprint(record)
                        save(row)
                    sequence += 1

                try:
                    with threadpool_limits(limits=1):
                        result = (evaluator or actual_pipeline_trial)(plan, job, emit)
                    if (not isinstance(result, dict) or len(result.get('claims', [])) != 5
                            or any(type(c.get('accepted')) is not bool or c.get('slot') != i
                                   for i, c in enumerate(result['claims']))):
                        raise ValueError('Evaluator must return five uniquely ordered boolean claim outcomes')
                    row.update(status='complete', result=result, receipt_count=sequence)
                except (ValueError, RuntimeError, FloatingPointError, MemoryError, np.linalg.LinAlgError) as exc:
                    row.update(status='failed_consumed', error_type=type(exc).__name__,
                               error=str(exc)[:500], receipt_count=sequence)
                save(row)
                count += 1
            report = _aggregate(plan, _read_trials(db), test_only=evaluator is not None)
            trials = db.execute('SELECT COUNT(*) FROM trials').fetchone()[0]
            report.update(binding_id=identity, trials=trials, planned_trials=sum(1 for _ in _jobs(plan)))
            # Immutable certificates only at the fixed endpoint. Interim output
            # is returned to the caller, not published as a passing certificate.
            if report['complete']:
                certificate = {'binding': binding, 'binding_id': identity, 'trials_hash': _trials_hash(db),
                               'formal_gate_eligible': report['formal_gate_eligible'],
                               'covered_claim_types': report['covered_claim_types'], 'report': report}
                certificate['digest'] = fingerprint(certificate)
                immutable_record(directory / 'certificate.json', certificate)
            return report
        finally:
            db.close()


def verify_calibration_certificate(directory, *, expected_binding=None):
    """Recompute the report; a flipped eligibility bit is never trusted."""
    directory = project_path(directory)
    certificate = json.loads((directory / 'certificate.json').read_text('utf-8'))
    if certificate.get('version') == 'v7-full-original-certificate-reference-v1':
        from .full_trial_certificate import verify_lifecycle_certificate
        return verify_lifecycle_certificate(directory, expected_binding=expected_binding)
    digest = certificate.pop('digest')
    if fingerprint(certificate) != digest:
        raise ValueError('Calibration certificate integrity mismatch')
    binding = certificate['binding']
    plan = CalibrationPlan.from_dict(binding['plan'])
    current = calibration_binding(plan)
    current.update(test_only=binding['test_only'], evaluator_id=binding['evaluator_id'])
    if current != binding or fingerprint(binding) != certificate['binding_id']:
        raise ValueError('Calibration source/protocol binding changed')
    expected_digest = expected_binding if isinstance(expected_binding, str) else fingerprint(expected_binding)
    if expected_binding is not None and expected_digest != certificate['binding_id']:
        raise ValueError('Calibration does not cover requested pipeline/proposal contract')
    identity = json.loads((directory / 'identity.json').read_text('utf-8'))
    if identity != {'binding': binding, 'id': certificate['binding_id']}:
        raise ValueError('Calibration identity manifest mismatch')
    with sqlite3.connect(directory / 'calibration.sqlite') as db:
        _verify_journal(db)
        for row in _read_trials(db):
            receipts = list(db.execute('SELECT seq,value,hash FROM receipts WHERE trial=? ORDER BY seq', (row['job']['id'],)))
            if len(receipts) != row.get('receipt_count', len(receipts)):
                raise ValueError('Calibration step receipt missing')
            for i, (seq, text, saved_hash) in enumerate(receipts):
                value = json.loads(text)
                if seq != i or fingerprint(value) != saved_hash or value['trial'] != row['job']['id']:
                    raise ValueError('Calibration receipt integrity mismatch')
        if _trials_hash(db) != certificate['trials_hash']:
            raise ValueError('Calibration trials differ from certificate')
        frozen = json.loads((directory / 'validation_freeze.json').read_text('utf-8'))
        expected = {'binding': certificate['binding_id'],
                    'calibration_trials_hash': _trials_hash(db, 'calibration'),
                    'rule': plan.rule.to_dict(), 'selection': 'unchanged_predeclared_local_protocol',
                    'validation_seed_domain': [plan.seed, 1], 'nuisance_settings': 'held_out_AR_and_t_df'}
        if frozen != expected:
            raise ValueError('Independent validation freeze integrity mismatch')
        report = _aggregate(plan, _read_trials(db), test_only=binding['test_only'])
        trials = db.execute('SELECT COUNT(*) FROM trials').fetchone()[0]
    report.update(binding_id=certificate['binding_id'], trials=trials, planned_trials=sum(1 for _ in _jobs(plan)))
    if report != certificate['report'] or not report['complete']:
        raise ValueError('Calibration certificate report cannot be reproduced')
    if certificate['formal_gate_eligible'] != report['formal_gate_eligible'] or certificate['covered_claim_types'] != report['covered_claim_types']:
        raise ValueError('Calibration certificate gate/scope mismatch')
    return report


def _verify_journal(db):
    previous, heads, number = '0' * 64, {}, 0
    for sequence, text, digest in db.execute('SELECT seq,value,hash FROM journal ORDER BY seq'):
        event = json.loads(text)
        if sequence != number or event['sequence'] != number or event['previous'] != previous or fingerprint(event) != digest:
            raise ValueError('Calibration journal integrity mismatch')
        previous, number = digest, number + 1
        heads[event['trial']] = event['state_hash']
    actual = {}
    for row in _read_trials(db):
        identity = row['job']['id']
        actual[identity] = fingerprint(row)
        head, count = '0' * 64, 0
        for sequence, text, digest in db.execute('SELECT seq,value,hash FROM receipts WHERE trial=? ORDER BY seq', (identity,)):
            receipt = json.loads(text)
            if (sequence != count or receipt['sequence'] != count or receipt['trial'] != identity
                    or receipt['previous'] != head or fingerprint(receipt) != digest):
                raise ValueError('Calibration receipt chain integrity mismatch')
            head, count = digest, count + 1
        if count != row['receipt_count'] or head != row['receipt_head']:
            raise ValueError('Calibration receipt count/head mismatch')
    if actual != heads:
        raise ValueError('Calibration checkpoint set/hash mismatch')
