"""Five distinct local-policy claims with independent fitted-learner oracles.

This is deliberately NOT a certificate for GPT or the live autonomous service.
Independent oracle paths estimate learner risk, not a known DGP effect. Their
finite Monte Carlo intervals remain approximate; uncertain truth fails closed.
Each experiment and each oracle path is its own bounded, nonrefundable work unit.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import sqlite3
import time
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd
import psutil
from filelock import FileLock
from scipy.stats import t as student_t
from threadpoolctl import threadpool_limits

from factorlab.protocol import fingerprint, project_path

from .causal_numpy import execute_formula
from .confirmatory import ConfirmationRule, paired_effect_interval, wilson_interval
from .convex_eval import fit_ridge
from .evidence import file_hash
from .exposure import select_on_validation
from .joint_pool import FrozenPool, PoolMember, evaluate_candidate, fit_pool, hour_equal_weights
from .known_library import FEATURE_NAMES
from .pipeline_calibration import EFFECTS, candidate_programs
from .rolling import RollingSpec, build_rolling_data
from .v7_storage import immutable_record

SLOTS = ('library_skill', 'factor1', 'factor2', 'factor3', 'algorithm')
SCENARIOS = ('global_null_design', 'mixed_regime_missing')
HAR = ('rv_1', 'rv_24', 'rv_168', 'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos')
SCOPE = 'five_distinct_claims_bounded_local_policy_with_independent_oracle_v1'
COVERED = {'library_skill': 'fixed_K0_vs_HAR_cal_three_future_months',
           'factor1': 'August_local_search_frozen_P0_plus_formula_September_to_November',
           'factor2': 'September_local_search_frozen_P0_plus_formula_October_to_December',
           'factor3': 'October_local_search_frozen_P0_plus_formula_November_to_January',
           'algorithm': 'six_month_bounded_local_shadow_adoption_vs_fixed_K0'}
EXCLUDED = ('GPT_feedback_and_proposal_distribution', 'live_autonomous_worker_and_activation_delays',
            'unbounded_lifetime_adaptivity', 'production_A0_gate_and_controller',
            'exact_conditional_truth_given_selected_training_data', 'multi_asset_universe',
            'OS_enforced_peak_and_deadline_supervision')


@dataclass(frozen=True)
class FullCalibrationPlan:
    calibration_repetitions: int = 1
    validation_repetitions: int = 1
    oracle_paths: int = 2
    minimum_oracle_paths: int = 128
    minimum_validation_repetitions: int = 1000
    seed: int = 771004
    candidates: int = 3
    lookback: int = 48
    max_solves: int = 1600
    unit_seconds: int = 300
    memory_mb: int = 768
    oracle_max_halfwidth: float = .001
    effects: tuple[float, ...] = EFFECTS
    scenarios: tuple[str, ...] = SCENARIOS
    rule: ConfirmationRule = field(default_factory=ConfirmationRule)

    def __post_init__(self):
        for name, lo, hi in [('calibration_repetitions', 1, 2000), ('validation_repetitions', 1, 2000),
                            ('oracle_paths', 2, 2048), ('minimum_oracle_paths', 128, 2048),
                            ('minimum_validation_repetitions', 1000, 2000), ('seed', 0, 2**32 - 1),
                            ('candidates', 2, 3), ('lookback', 2, 512), ('max_solves', 1, 1600),
                            ('unit_seconds', 1, 300), ('memory_mb', 128, 768)]:
            if type(getattr(self, name)) is not int or not lo <= getattr(self, name) <= hi:
                raise ValueError(f'Invalid bounded calibration {name}')
        if (tuple(self.effects) != EFFECTS or tuple(self.scenarios) != SCENARIOS
                or self.oracle_max_halfwidth != .001 or not isinstance(self.rule, ConfirmationRule)
                or self.rule.horizon_hours != 12):
            raise ValueError('Frozen scenarios, targets, oracle precision and volatility12h rule required')

    def to_dict(self):
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_dict(cls, value):
        value = copy.deepcopy(value)
        value['effects'], value['scenarios'] = tuple(value['effects']), tuple(value['scenarios'])
        value['rule'] = ConfirmationRule(**value['rule'])
        return cls(**value)


def binding(plan, *, evaluator_id=None):
    root = project_path('.')
    paths = [*root.glob('factorlab/**/*.py'), root / 'pyproject.toml', root / 'uv.lock']
    sources = {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(paths)}
    return {'scope': SCOPE, 'plan': plan.to_dict(), 'sources': sources,
            'candidate_family': candidate_programs()[0][:plan.candidates],
            'test_only': evaluator_id is not None, 'evaluator_id': evaluator_id or 'actual_five_slot_pipeline_v1',
            'simulated_claim_slots': list(SLOTS), 'covered_scope': COVERED, 'excluded_scope': list(EXCLUDED),
            'pipeline': 'actual_joint_FPS_three_fold_S_one_month_V_three_month_C_cold_refits',
            'algorithm': 'six_month_local_replay_latest_completed_C_pipeline_no_combination',
            'oracle': 'independent_paths_same_calendar_conditional_formula_unconditional_replay_policy',
            'rule': plan.rule.to_dict(), 'rolling': RollingSpec().digest,
            'candidate_budget': {'families': 1, 'AST': plan.candidates, 'F': plan.candidates,
                                 'V': min(3, plan.candidates), 'C': 1, 'parameter_profile': 0},
            'budget_scope': 'per_work_unit_reservations_and_cooperative_300s_768MiB_checks_no_refunds',
            'effect_grid_semantics': 'DGP_amplitude_targets_never_asserted_population_gain'}


def synthetic_history(plan, job):
    """One continuous causal OHLC path; fixed calendar stresses and t innovations."""
    rng = np.random.default_rng(job['seed'])
    dates = pd.date_range('2021-01-01', '2022-04-03', inclusive='left', freq='h', tz='UTC')
    n = len(dates)
    df = 5 if job['phase'] == 'calibration' else 4
    shock = rng.standard_t(df, n) / math.sqrt(df / (df - 2))
    volatility_noise = rng.normal(0, .04, n)
    r, sigma = np.zeros(n), np.zeros(n)
    state, fast, slow = 0., 4e-6, 4e-6
    null = job['scenario'] == 'global_null_design'
    for i in range(n):
        state = (.94 if job['phase'] == 'calibration' else .975) * state + volatility_noise[i]
        regime = .25 * math.sin(2 * math.pi * i / (24 * 150))
        signal = math.tanh(fast / max(slow, 1e-12) - 1)
        # Even iid return innovations do not assert every finite learner claim null.
        log_sigma = 0. if null else state + regime + 5 * math.sqrt(job['effect']) * signal
        sigma[i] = .002 * math.exp(float(np.clip(log_sigma, -3, 3)))
        r[i] = sigma[i] * shock[i]
        fast, slow = .8 * fast + .2 * r[i] ** 2, .99 * slow + .01 * r[i] ** 2
    close = np.exp(4 + np.cumsum(r))
    opened = np.r_[math.exp(4), close[:-1]]
    spread = sigma * (1 + np.abs(rng.normal(0, .3, n)))
    bars = pd.DataFrame({'symbol': 'SYNTH', 'open_time': dates,
                        'available_at': dates + pd.Timedelta(hours=1, seconds=1),
                        'open': opened, 'close': close,
                        'high': np.maximum(opened, close) * np.exp(spread),
                        'low': np.minimum(opened, close) * np.exp(-spread)})
    if not null:
        gap = pd.Timestamp('2021-05-09' if job['phase'] == 'calibration' else '2021-05-19', tz='UTC')
        bars = bars.loc[~bars.open_time.between(gap, gap + pd.Timedelta(hours=2))].reset_index(drop=True)
    bars.attrs['availability_mode'] = 'assumed_historical_delay'
    return bars


class UnitBudget:
    def __init__(self, plan, emit):
        self.plan, self.emit = plan, emit
        self.solves, self.peak_rss, self.started = 0, 0, time.monotonic()

    def check(self):
        self.peak_rss = max(self.peak_rss, psutil.Process().memory_info().rss)
        if self.peak_rss > self.plan.memory_mb * 2**20:
            raise RuntimeError('memory_budget_exhausted')
        if time.monotonic() - self.started >= self.plan.unit_seconds:
            raise RuntimeError('wall_budget_exhausted')

    def consume(self, kind, amount=1, **kwargs):
        if kind != 'solve' or type(amount) is not int or amount < 1:
            raise ValueError('Only actual convex solve reservations are supported')
        self.check()
        if self.solves + amount > self.plan.max_solves:
            raise RuntimeError('solve_budget_exhausted')
        self.solves += amount
        self.emit({'event': 'solve_reserved', 'solves': self.solves})


def _candidate_pool(member):
    reference = FrozenPool()
    return FrozenPool((member,), version='fixed_local_candidate',
                      reference_pipeline_id=reference.pipeline_id, reference_pool=reference)


def _month(data, start, end, budget, emit, *, member=None, library=False, as_of=None):
    """Actual cold predictors, same hour/symbol support and frozen training recipe."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if member is not None:
        raw = execute_formula(member.program, data.frame, lookback=member.lookback, parameters=member.parameters)
        data = replace(data, observed=data.observed & np.isfinite(raw))
    reference = fit_pool(data, start, FrozenPool(), budget=budget)
    reference_prediction, support = reference.predict_history(data.frame)
    train = data.train_mask(start)
    if library:
        idx = [FEATURE_NAMES.index(name) for name in HAR]
        budget.consume('solve')
        har = fit_ridge(data.frame.values[train][:, idx], data.target[train], ridge=data.spec.ridge,
                        sample_weight=hour_equal_weights(data.frame.metadata.loc[train]))
        prediction = reference_prediction.copy()
        reference_prediction = np.full(len(train), np.nan)
        reference_prediction[support] = har.predict(data.frame.values[support][:, idx])
        model_id, reference_id = reference.digest, har.digest
    elif member is not None:
        candidate = fit_pool(data, start, _candidate_pool(member), budget=budget)
        prediction, candidate_support = candidate.predict_history(data.frame)
        # Frozen local ratio/product candidates are finite whenever K0 is valid.
        # Fail instead of letting asymmetric support manufacture a comparison.
        if np.any(train & ~candidate_support):
            raise ValueError('asymmetric_training_support')
        support &= candidate_support
        model_id, reference_id = candidate.digest, reference.digest
    else:
        prediction = reference_prediction.copy()
        model_id = reference_id = reference.digest
    score = data.score_mask(start, end, as_of or end) & support
    if not score.any():
        raise ValueError('empty_common_score_support')
    meta = data.frame.metadata.loc[score]
    target = np.log(data.target[score] + 1e-8)
    base = np.square(target - np.log(reference_prediction[score] + 1e-8))
    candidate_loss = np.square(target - np.log(prediction[score] + 1e-8))
    budget.check()
    emit({'event': 'cold_month', 'start': start.isoformat(), 'end': end.isoformat(),
          'as_of': str(as_of or end), 'member': member.member_id if member else None, 'library': library,
          'model': model_id, 'reference_model': reference_id,
          'train_identity': data.training_identity(start), 'score_rows': int(score.sum()),
          'support_hash': fingerprint(meta.sample_id.tolist()),
          'same_hour_paired': True, 'base_mean': float(base.mean()), 'candidate_mean': float(candidate_loss.mean())})
    return meta.available_at.tolist(), base.tolist(), candidate_loss.tolist()


def _joined(parts):
    return tuple([item for part in parts for item in part[i]] for i in range(3))


def _period(data, start, months, budget, emit, *, member=None, library=False):
    start = pd.Timestamp(start)
    end = start + pd.DateOffset(months=months)
    as_of = end + pd.Timedelta(hours=36)
    return _joined([_month(data, month, month + pd.DateOffset(months=1), budget, emit,
                           member=member, library=library, as_of=as_of)
                    for month in pd.date_range(start, end, inclusive='left', freq='MS')])


def _claim(slot, pair, start, months, plan):
    end = pd.Timestamp(start) + pd.DateOffset(months=months)
    nominal = int((end - pd.Timestamp(start)) / pd.Timedelta(hours=1))
    interval = paired_effect_interval(*pair, rule=plan.rule, eligible_count=nominal,
                                      calibration_passed=False)
    return {'slot': slot, 'accepted': interval.get('diagnostic_status') == 'supported_target',
            'interval': interval, 'base_mean': float(np.mean(pair[1])),
            'candidate_mean': float(np.mean(pair[2])), 'support': len(pair[0]),
            'support_hash': fingerprint([str(t) for t in pair[0]]),
            'start': str(start), 'end': end.isoformat(), 'formal_claim': False}


def _search(data, start, plan, budget, emit, slot):
    candidates, records, by_id, failures = candidate_programs()[0][:plan.candidates], [], {}, []
    for program in candidates:
        member = PoolMember(program, plan.lookback, family=slot)
        emit({'event': 'candidate_reserved', 'slot': slot, 'member': member.member_id})
        try:
            result = evaluate_candidate(data, start, FrozenPool(), member, budget=budget)
            gain = result['S']['Pv']['gain']
            emit({'event': 'S_three_fold', 'slot': slot, 'member': member.member_id,
                  'readouts': {k: result[k] for k in ('F', 'P', 'S')},
                  'folds': [{k: f[k] for k in ('fit_cutoff', 'score_end')} for f in result['folds']]})
            if gain is not None:
                records.append((gain, member.member_id))
                by_id[member.member_id] = member
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            failures.append({'stage': 'S', 'candidate': member.member_id, 'error': str(exc)[:240]})
            emit({'event': 'candidate_failed', 'slot': slot, **failures[-1]})
    shortlist = sorted(records, key=lambda r: (-r[0], r[1]))[:3]
    emit({'event': 'V_shortlist_frozen', 'slot': slot, 'shortlist': [r[1] for r in shortlist]})
    end, validation = start + pd.DateOffset(months=1), []
    for _, identifier in shortlist:
        try:
            pair = _month(data, start, end, budget, emit, member=by_id[identifier])
            base, cand = np.mean(pair[1]), np.mean(pair[2])
            validation.append({'role': 'V', 'common_support': True, 'candidate_id': identifier,
                               'coverage': len(pair[0]) / ((end - start) / pd.Timedelta(hours=1)),
                               'gain': float(1 - cand / base) if base > 0 else None,
                               'complexity': len(json.dumps(by_id[identifier].program, sort_keys=True))})
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            failures.append({'stage': 'V', 'candidate': identifier, 'error': str(exc)[:240]})
            emit({'event': 'candidate_failed', 'slot': slot, **failures[-1]})
    chosen = select_on_validation(validation)
    member = by_id[chosen['candidate_id']] if chosen else None
    emit({'event': 'C_frozen', 'slot': slot, 'member': member.to_dict() if member else None,
          'start': end.isoformat(), 'end': (end + pd.DateOffset(months=3)).isoformat(),
          'rule_id': plan.rule.digest, 'validation': validation, 'failures': failures})
    return member, failures


def _experiment(data, plan, budget, emit):
    library_start = pd.Timestamp('2021-09-01T00:00Z')
    claims = [_claim('library_skill', _period(data, library_start, 3, budget, emit, library=True),
                     library_start, 3, plan)]
    frozen, adoptions = {}, []
    for i, start in enumerate(pd.date_range('2021-08-01', periods=3, freq='MS', tz='UTC'), 1):
        slot, cstart = f'factor{i}', start + pd.DateOffset(months=1)
        member, failures = _search(data, start, plan, budget, emit, slot)
        frozen[slot] = member.to_dict() if member else None
        if member is None:
            claims.append({'slot': slot, 'accepted': False, 'selected': None,
                           'outcome': 'all_candidates_failed_or_no_V_pass', 'failures': failures,
                           'base_mean': None, 'candidate_mean': None, 'formal_claim': False})
            continue
        claim = _claim(slot, _period(data, cstart, 3, budget, emit, member=member), cstart, 3, plan)
        claim.update(selected=member.member_id, failures=failures)
        claims.append(claim)
        # Endpoint + label grace must mature before adoption; coefficients fit at
        # the next UTC month boundary. Adopt the complete pipeline, never a blend.
        ready = cstart + pd.DateOffset(months=3) + pd.Timedelta(hours=36)
        activation = (ready.normalize().replace(day=1) + pd.DateOffset(months=1))
        if claim['accepted']:
            adoptions.append((activation, member))
    parts, trace = [], []
    astart = pd.Timestamp('2021-10-01T00:00Z')
    for month in pd.date_range(astart, periods=6, freq='MS'):
        eligible = [(time_, member) for time_, member in adoptions if time_ <= month]
        member = max(eligible, key=lambda r: r[0])[1] if eligible else None
        parts.append(_month(data, month, month + pd.DateOffset(months=1), budget, emit, member=member,
                            as_of=astart + pd.DateOffset(months=6) + pd.Timedelta(hours=36)))
        trace.append({'month': month.isoformat(), 'pipeline_member': member.member_id if member else None})
    algorithm = _claim('algorithm', _joined(parts), astart, 6, plan)
    algorithm['adoption_trace'] = trace
    claims.append(algorithm)
    return {'claims': claims, 'frozen_factors': frozen, 'algorithm_policy': 'local_shadow_diagnostic_adoption',
            'full_V7': False, 'scope': SCOPE, 'excluded_scope': list(EXCLUDED)}


def actual_work_unit(plan, job, emit, experiment=None):
    budget = UnitBudget(plan, emit)
    budget.check()
    bars = synthetic_history(plan, job)
    data = build_rolling_data(bars, RollingSpec())
    emit({'event': 'OHLC_built', 'rows': len(bars), 'data_hash': fingerprint(bars.astype(str).to_dict('list')),
          'shape': list(data.frame.values.shape), 'seed_key': job['seed_key'], 'unit': job['kind']})
    # Every oracle reruns the actual local adaptive policy on a fresh path. Its
    # library/algorithm risks are unconditional policy risks; factor risks below
    # freeze the ORIGINAL experiment's selected formula, not oracle winners.
    result = _experiment(data, plan, budget, emit)
    if job['kind'] == 'oracle':
        if experiment is None:
            raise ValueError('Oracle requires original frozen experiment')
        original = experiment['frozen_factors']
        risks = {c['slot']: [c['base_mean'], c['candidate_mean']] for c in result['claims']
                 if c['slot'] in ('library_skill', 'algorithm')}
        for i, start in enumerate(pd.date_range('2021-09-01', periods=3, freq='MS', tz='UTC'), 1):
            slot, value = f'factor{i}', original[f'factor{i}']
            if value is None:
                risks[slot] = None
            else:
                pair = _period(data, start, 3, budget, emit, member=PoolMember.from_dict(value))
                risks[slot] = [float(np.mean(pair[1])), float(np.mean(pair[2]))]
        result = {'risks': risks, 'original_frozen_hash': fingerprint(original),
                  'scope': SCOPE, 'full_V7': False}
    budget.check()
    return {**result, 'solves': budget.solves, 'peak_rss_bytes': budget.peak_rss,
            'elapsed_seconds': time.monotonic() - budget.started}


def _jobs(plan):
    for phase_idx, phase in enumerate(('calibration', 'validation')):
        for scenario_idx in range(len(plan.scenarios)):
            for effect_idx in range(len(plan.effects)):
                for repetition in range(getattr(plan, f'{phase}_repetitions')):
                    for unit in range(plan.oracle_paths + 1):
                        yield _job(plan, phase_idx, scenario_idx, effect_idx, repetition, unit)


def _job(plan, phase_idx, scenario_idx, effect_idx, repetition, unit):
    if not 0 <= phase_idx < 2:
        raise ValueError('Work unit phase outside frozen plan')
    phase = ('calibration', 'validation')[phase_idx]
    if (not 0 <= scenario_idx < len(plan.scenarios) or not 0 <= effect_idx < len(plan.effects)
            or not 0 <= repetition < getattr(plan, f'{phase}_repetitions')
            or not 0 <= unit <= plan.oracle_paths):
        raise ValueError('Work unit outside frozen plan')
    experiment_id = f'{phase}:{scenario_idx}:{effect_idx}:{repetition}'
    key = [plan.seed, phase_idx, scenario_idx, effect_idx, repetition, unit]
    return {'id': f'{experiment_id}:{unit}', 'experiment_id': experiment_id,
            'kind': 'experiment' if unit == 0 else 'oracle', 'oracle_index': unit - 1,
            'phase': phase, 'scenario': plan.scenarios[scenario_idx], 'effect': plan.effects[effect_idx],
            'repetition': repetition, 'seed_key': key,
            'seed': int(np.random.SeedSequence(key).generate_state(1, dtype=np.uint64)[0])}


def _expected_job(plan, identifier):
    try:
        phase, scenario, effect, repetition, unit = identifier.split(':')
        return _job(plan, ('calibration', 'validation').index(phase),
                    int(scenario), int(effect), int(repetition), int(unit))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError('Work unit identity is outside frozen plan') from exc


def oracle_truth(plan, risks, *, slot):
    """Independent path means, delta-method ratio CI; NOT exact population truth."""
    if len(risks) < 2:
        return {'slot': slot, 'status': 'uncertain', 'paths': len(risks), 'gain': None, 'interval': None}
    array = np.asarray(risks, dtype=float)
    if array.shape != (len(risks), 2) or not np.isfinite(array).all() or (array < 0).any():
        raise ValueError('Oracle requires finite nonnegative paired risks')
    base, candidate = array.mean(axis=0)
    if base <= 0:
        return {'slot': slot, 'status': 'uncertain', 'paths': len(risks), 'gain': None, 'interval': None}
    gain = float(1 - candidate / base)
    influence = candidate / base**2 * (array[:, 0] - base) - (array[:, 1] - candidate) / base
    # 99% familywise nominal interval across the five oracle slots of this trial.
    critical = student_t.ppf(1 - .01 / (2 * len(SLOTS)), len(risks) - 1)
    half = float(critical * influence.std(ddof=1) / math.sqrt(len(risks)))
    interval, status = [gain - half, gain + half], 'uncertain'
    if len(risks) >= plan.minimum_oracle_paths and half <= plan.oracle_max_halfwidth:
        if interval[1] <= plan.rule.delta0:
            status = 'null'
        elif interval[0] > plan.rule.delta0:
            status = 'alternative'
    return {'slot': slot, 'status': status, 'paths': len(risks), 'gain': gain, 'interval': interval,
            'method': 'independent_path_ratio_delta_t_approximate_Bonferroni5',
            'certified_population_truth': False}


def _read(db):
    for key, payload, digest in db.execute('SELECT id,value,hash FROM units ORDER BY id'):
        row = json.loads(payload)
        if row['job']['id'] != key or fingerprint(row) != digest:
            raise ValueError('Work unit checkpoint integrity mismatch')
        yield row


def _get(db, key):
    found = db.execute('SELECT value,hash FROM units WHERE id=?', (key,)).fetchone()
    if found is None:
        return None
    row = json.loads(found[0])
    if row['job']['id'] != key or fingerprint(row) != found[1]:
        raise ValueError('Work unit checkpoint integrity mismatch')
    return row


def _phase_hash(db, phase):
    digest, first = hashlib.sha256(b'['), True
    for row in _read(db):
        if row['job']['phase'] != phase:
            continue
        if not first:
            digest.update(b',')
        digest.update(json.dumps(fingerprint(row)).encode())
        first = False
    digest.update(b']')
    return digest.hexdigest()


def _verify(db, directory):
    identity = json.loads((directory / 'identity.json').read_text('utf-8'))
    frozen, binding_id = identity['binding'], identity['digest']
    if fingerprint(frozen) != binding_id:
        raise ValueError('Calibration frozen identity digest changed')
    plan = FullCalibrationPlan.from_dict(frozen['plan'])
    previous, sequence = '0' * 64, 0
    db.execute('CREATE TEMP TABLE IF NOT EXISTS verified_heads(id TEXT PRIMARY KEY,hash TEXT)')
    db.execute('DELETE FROM verified_heads')
    for number, payload, digest in db.execute('SELECT seq,value,hash FROM journal ORDER BY seq'):
        event = json.loads(payload)
        if number != sequence or event['seq'] != sequence or event['previous'] != previous or fingerprint(event) != digest:
            raise ValueError('Calibration journal integrity mismatch')
        if event['kind'] == 'state':
            db.execute('INSERT OR REPLACE INTO verified_heads VALUES(?,?)',
                        (event['unit'], event['payload']['state_hash']))
        previous, sequence = digest, sequence + 1
    count = db.execute('SELECT COUNT(*) FROM units').fetchone()[0]
    if count != db.execute('SELECT COUNT(*) FROM verified_heads').fetchone()[0]:
        raise ValueError('Calibration checkpoint set/hash mismatch')
    reservations = 0
    for path in (directory / 'reservations').glob('*.json'):
        job = json.loads(path.read_text('utf-8'))
        if (path.stem != fingerprint(job)
                or db.execute('SELECT 1 FROM units WHERE id=?', (job['id'],)).fetchone() is None):
            raise ValueError('Missing checkpoint or orphan work reservation')
        reservations += 1
    if reservations != count:
        raise ValueError('Missing checkpoint or orphan work reservation')
    for row in _read(db):
        if row['binding'] != binding_id or row['job'] != _expected_job(plan, row['job']['id']):
            raise ValueError('Work unit binding or job differs from frozen plan')
        head = db.execute('SELECT hash FROM verified_heads WHERE id=?', (row['job']['id'],)).fetchone()
        if head is None or head[0] != fingerprint(row):
            raise ValueError('Calibration checkpoint set/hash mismatch')
        path = directory / 'reservations' / f"{fingerprint(row['job'])}.json"
        if json.loads(path.read_text('utf-8')) != row['job']:
            raise ValueError('Work unit reservation changed')
    # Published diagnostics retain a nonrefundable lower bound on consumed work
    # even if both mutable SQLite state and reservation files disappear.
    for path in (directory / 'reports').glob('*.json'):
        report = json.loads(path.read_text('utf-8'))
        if (report['binding_id'] != binding_id or type(report['work_units']) is not int
                or not 0 <= report['work_units'] <= count
                or path.stem != f"{report['work_units']:08d}-{fingerprint(report)}"):
            raise ValueError('Published report differs from consumed checkpoint history')
    _verify_validation_freeze(db, directory, frozen)


def _verify_validation_freeze(db, directory, frozen):
    path = directory / 'validation_freeze.json'
    has_validation = db.execute("SELECT 1 FROM units WHERE id LIKE 'validation:%' LIMIT 1").fetchone() is not None
    if path.exists() or has_validation:
        expected = {'binding': fingerprint(frozen), 'calibration_hash': _phase_hash(db, 'calibration'),
                    'policy': 'unchanged_frozen_policy', 'seed_domain': [frozen['plan']['seed'], 1]}
        if not path.exists() or json.loads(path.read_text('utf-8')) != expected:
            raise ValueError('Independent validation freeze missing or changed')


def _effect_bin(plan, truth):
    if truth['status'] != 'alternative':
        return None
    matches = [target for target in plan.effects if truth['interval'][0] >= target - .001
               and truth['interval'][1] <= target + .001]
    # Disjoint assignment: even at touching bin boundaries a trial/slot counts once.
    return min(matches, key=lambda target: (abs(truth['gain'] - target), target)) if matches else None


def _report(plan, rows, *, test_only):
    total, repetitions, terminal = 0, 0, True
    summaries = {(phase, scenario, effect): {'n': 0, 'false': 0, 'possible': 0,
                 'uncertain_truth_trials': 0, 'all_five_null_trials': 0,
                 'discovery': dict.fromkeys(SLOTS, 0),
                 'power': {(slot, target): [0, 0] for slot in SLOTS for target in plan.effects}}
                 for phase in ('calibration', 'validation') for scenario in plan.scenarios for effect in plan.effects}
    outcomes = []  # Bounded examples only; all units remain in the streaming ledger.
    for name, grouped in itertools.groupby(rows, key=lambda r: r['job']['experiment_id']):
        units = list(grouped)  # At most 2049 small path summaries, never the entire job.
        total += len(units)
        repetitions += 1
        experiment = next((r for r in units if r['job']['kind'] == 'experiment'), None)
        if experiment is None:
            raise ValueError('Oracle lacks its experiment reservation')
        claims = []
        finished = experiment['status'] == 'complete'
        oracle_units = [r for r in units if r['job']['kind'] == 'oracle']
        oracle_complete = (len(oracle_units) == plan.oracle_paths
                           and all(r['status'] == 'complete' for r in oracle_units))
        for slot in SLOTS:
            risks = [r['result']['risks'][slot] for r in units if r['job']['kind'] == 'oracle'
                     and r['status'] == 'complete' and r['result']['risks'].get(slot) is not None]
            truth = oracle_truth(plan, risks, slot=slot)
            if not oracle_complete or len(risks) != plan.oracle_paths:
                truth.update(status='uncertain', incomplete_or_failed_oracle_paths=True)
            claim = next((c for c in experiment.get('result', {}).get('claims', []) if c['slot'] == slot), None)
            if claim is not None and slot.startswith('factor') and claim.get('selected') is None:
                truth.update(status='no_selected_claim')
            claims.append({'slot': slot, 'accepted': bool(claim and claim['accepted']), 'truth': truth})
        false = any(c['accepted'] and c['truth']['status'] == 'null' for c in claims)
        possible = not finished or any(c['accepted'] and c['truth']['status'] != 'alternative' for c in claims)
        outcome = {'id': name, 'phase': experiment['job']['phase'], 'scenario': experiment['job']['scenario'],
                         'effect_amplitude': experiment['job']['effect'], 'claims': claims,
                         'false_trial': false, 'possible_false_trial': possible,
                         'finished': finished, 'all_oracles_terminal': len(units) == plan.oracle_paths + 1
                         and all(r['status'] != 'reserved' for r in units)}
        terminal &= outcome['all_oracles_terminal']
        if len(outcomes) < 16:
            outcomes.append(outcome)
        group = summaries[outcome['phase'], outcome['scenario'], outcome['effect_amplitude']]
        group['n'] += 1
        group['false'] += int(false)
        group['possible'] += int(possible)
        group['uncertain_truth_trials'] += int(any(c['truth']['status'] == 'uncertain' for c in claims))
        group['all_five_null_trials'] += int(all(c['truth']['status'] in {'null', 'no_selected_claim'} for c in claims))
        for claim in claims:
            group['discovery'][claim['slot']] += int(claim['accepted'])
            target = _effect_bin(plan, claim['truth'])
            if target is not None:
                counts = group['power'][claim['slot'], target]
                counts[0] += 1
                counts[1] += int(claim['accepted'])
    curve = []
    cells = 2 * len(plan.scenarios) * len(plan.effects)
    confidence = 1 - 2 * .05 / cells  # One-sided nominal Bonferroni over all MC cells.
    for phase in ('calibration', 'validation'):
        for scenario in plan.scenarios:
            for effect in plan.effects:
                group = summaries[phase, scenario, effect]
                n, false, possible = group['n'], group['false'], group['possible']
                power = []
                for slot in SLOTS:
                    for target in plan.effects:
                        count, accepted = group['power'][slot, target]
                        power.append({'slot': slot, 'learner_effect_target': target, 'n': count,
                                      'accepted': accepted, 'lower': wilson_interval(accepted, count, confidence=confidence)[0] if count else None,
                                      'status': 'available' if count else 'unavailable',
                                      'formal_alternative': target > plan.rule.delta0})
                curve.append({'phase': phase, 'scenario': scenario, 'amplitude': effect, 'n': n,
                              'known_false': false, 'possible_false': possible,
                              'uncertain_truth_trials': group['uncertain_truth_trials'],
                              'all_five_null_trials': group['all_five_null_trials'],
                              'FWER_upper': wilson_interval(possible, n, confidence=confidence)[1] if n else None,
                              'conditional_power': power,
                              'discovery_acceptance': [{'slot': slot, 'n': n, 'accepted': group['discovery'][slot],
                                  'rate': group['discovery'][slot] / n if n else None} for slot in SLOTS],
                              'FWER_truth_scope': 'conditional_on_approximate_oracle_labels_not_a_certified_bound',
                              'MC_interval_scope': 'Bonferroni_cells_only_power_bounds_pointwise'})
    planned = ((plan.calibration_repetitions + plan.validation_repetitions)
               * len(plan.scenarios) * len(plan.effects) * (plan.oracle_paths + 1))
    complete = total == planned and terminal
    validation = [row for row in curve if row['phase'] == 'validation']
    enough = complete and all(row['n'] >= plan.minimum_validation_repetitions for row in validation)
    truth_resolved = complete and all(row['uncertain_truth_trials'] == 0 for row in validation)
    error_ok = enough and all(row['FWER_upper'] <= .05 for row in validation)
    mixture = []
    for slot in SLOTS:
        for target in plan.effects:
            points = [point for row in validation for point in row['conditional_power']
                      if point['slot'] == slot and point['learner_effect_target'] == target]
            n, accepted = sum(p['n'] for p in points), sum(p['accepted'] for p in points)
            mixture.append({'slot': slot, 'learner_effect_target': target, 'n': n, 'accepted': accepted,
                            'rate': accepted / n if n else None,
                            'lower': wilson_interval(accepted, n, confidence=confidence)[0] if n else None,
                            'status': 'available' if n else 'unavailable',
                            'formal_alternative': target > plan.rule.delta0})
    alternatives = [point for point in mixture if point['formal_alternative']]
    scenario_power = []
    for scenario in plan.scenarios:
        for slot in SLOTS:
            for target in plan.effects:
                points = [point for row in validation if row['scenario'] == scenario
                          for point in row['conditional_power']
                          if point['slot'] == slot and point['learner_effect_target'] == target]
                n, accepted = sum(p['n'] for p in points), sum(p['accepted'] for p in points)
                scenario_power.append({'scenario': scenario, 'slot': slot, 'learner_effect_target': target,
                                       'n': n, 'accepted': accepted,
                                       'lower': wilson_interval(accepted, n, confidence=confidence)[0] if n else None,
                                       'status': 'available' if n else 'unavailable'})
    unsupported = [{'slot': p['slot'], 'learner_effect_target': p['learner_effect_target']}
                   for p in alternatives if p['n'] == 0]
    power_ok = enough and all(point['n'] >= plan.minimum_validation_repetitions
                             and point['lower'] >= .8 for point in alternatives)
    return {'scope': SCOPE, 'simulated_claim_slots': list(SLOTS), 'covered_scope': COVERED, 'excluded_scope': list(EXCLUDED),
            'complete': complete, 'work_units': total, 'planned_work_units': planned,
            'experiment_repetitions': repetitions, 'outcomes': outcomes,
            'outcomes_scope': 'first16_examples_all_trials_counted_in_curve', 'curve': curve,
            'formal_gate_eligible': False, 'full_V7': False, 'test_only': test_only,
            'effect_truth_certified': False,
            'sufficient_independent_validation': enough, 'oracle_truth_numerically_resolved': truth_resolved,
            'conservative_FWER_bound_passed': error_ok, 'power_bound_passed': power_ok,
            'validation_mixture_conditional_power': mixture, 'unsupported_effect_bins': unsupported,
            'validation_scenario_conditional_power': scenario_power,
            'power_scope': 'predeclared_validation_scenario_amplitude_mixture_conditioned_on_oracle_effect_band',
            'power_is_end_to_end_discovery_probability': False,
            'reasons': ['limited_local_policy_not_GPT_or_production_A0',
                        'oracle_delta_method_truth_bridge_requires_independent_validation']
                       + (['incomplete_work_units'] if not complete else [])
                       + (['insufficient_oracle_paths'] if plan.oracle_paths < plan.minimum_oracle_paths else [])
                       + (['insufficient_validation_repetitions'] if plan.validation_repetitions < plan.minimum_validation_repetitions else [])
                       + (['unsupported_effect_bins'] if unsupported else [])
                       + (['test_evaluator'] if test_only else []),
            'global_null_policy': 'iid_DGP_is_a_design_label_all_five_learner_nulls_require_oracle_support',
            'power_policy': '0_0.003_0.005_are_null_or_boundary_not_required_80percent_detection',
            'failed_or_no_selection_trials_are_never_dropped': True}


def run_full_pipeline_calibration(directory, plan=None, *, max_new_trials=1, evaluator=None, evaluator_id=None):
    """Execute at most N work units with 300s/768MiB cooperative checks.

    One experiment is followed by separately journaled independent oracle paths.
    This bounds each invocation even when a long validation requests 128+ paths.
    A callback is always test-only. No certificate is emitted by this protocol.
    A hard OS resource ceiling requires an external process supervisor; these
    checks cannot interrupt a single native operation while it is in progress.
    """
    plan = plan or FullCalibrationPlan()
    if type(max_new_trials) is not int or not 1 <= max_new_trials <= 10000:
        raise ValueError('Finite max_new_trials required')
    if evaluator is not None and (not isinstance(evaluator_id, str) or not evaluator_id):
        raise ValueError('Test evaluator needs frozen identity')
    directory = project_path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frozen = binding(plan, evaluator_id=evaluator_id if evaluator else None)
    with FileLock(str(directory / 'calibration.lock'), timeout=0):
        if not (directory / 'identity.json').exists() and any(
                path.name != 'calibration.lock' for path in directory.iterdir()):
            raise ValueError('Missing frozen identity in existing calibration directory')
        immutable_record(directory / 'identity.json', {'binding': frozen, 'digest': fingerprint(frozen)})
        for relative, digest in frozen['sources'].items():
            archive = directory / 'source' / relative
            archive.parent.mkdir(parents=True, exist_ok=True)
            if not archive.exists():
                archive.write_bytes((project_path('.') / relative).read_bytes())
            if file_hash(archive) != digest:
                raise ValueError('Archived source changed')
        db = sqlite3.connect(directory / 'calibration.sqlite')
        db.execute('PRAGMA synchronous=FULL')
        db.executescript('CREATE TABLE IF NOT EXISTS units(id TEXT PRIMARY KEY,value TEXT,hash TEXT);'
                         'CREATE TABLE IF NOT EXISTS journal(seq INTEGER PRIMARY KEY,value TEXT,hash TEXT);')

        def event(kind, unit, payload):
            head = db.execute('SELECT seq,hash FROM journal ORDER BY seq DESC LIMIT 1').fetchone()
            value = {'seq': head[0] + 1 if head else 0, 'previous': head[1] if head else '0' * 64,
                     'kind': kind, 'unit': unit, 'payload': payload}
            db.execute('INSERT INTO journal VALUES(?,?,?)',
                        (value['seq'], json.dumps(value, sort_keys=True, allow_nan=False), fingerprint(value)))

        def save(row):
            with db:
                db.execute('INSERT OR REPLACE INTO units VALUES(?,?,?)',
                            (row['job']['id'], json.dumps(row, sort_keys=True, allow_nan=False), fingerprint(row)))
                event('state', row['job']['id'], {'state_hash': fingerprint(row)})

        try:
            _verify(db, directory)
            for row in _read(db):
                if row['status'] == 'reserved':
                    row.update(status='interrupted_consumed', error='reserved_work_not_refunded',
                               elapsed_seconds_charged=plan.unit_seconds,
                               time_charge_scope='full_reserved_ceiling_after_unobserved_interruption')
                    save(row)
            count = 0
            for job in _jobs(plan):
                if _get(db, job['id']) is not None:
                    continue
                if count >= max_new_trials:
                    break
                if binding(plan, evaluator_id=evaluator_id if evaluator else None) != frozen:
                    raise ValueError('Source changed before next calibration reservation')
                if job['phase'] == 'validation':
                    calibration_hash = _phase_hash(db, 'calibration')
                    immutable_record(directory / 'validation_freeze.json',
                                     {'binding': fingerprint(frozen), 'calibration_hash': calibration_hash,
                                      'policy': 'unchanged_frozen_policy', 'seed_domain': [plan.seed, 1]})
                immutable_record(directory / 'reservations' / f'{fingerprint(job)}.json', job)
                row = {'job': job, 'status': 'reserved', 'binding': fingerprint(frozen), 'solves_reserved': 0,
                       'reserved_at': pd.Timestamp.now(tz='UTC').isoformat(),
                       'time_budget_reserved_seconds': plan.unit_seconds, 'elapsed_seconds_charged': 0.}
                save(row)
                started = time.monotonic()

                def emit(payload, job=job, row=row):
                    with db:
                        event('step', job['id'], payload)
                        if payload.get('event') == 'solve_reserved':
                            row['solves_reserved'] = payload['solves']
                            save(row)

                experiment = _get(db, job['experiment_id'] + ':0') if job['kind'] == 'oracle' else None
                try:
                    if job['kind'] == 'oracle' and (experiment is None or experiment['status'] != 'complete'):
                        raise ValueError('original_experiment_unavailable_or_failed')
                    with threadpool_limits(limits=1):
                        result = (evaluator or actual_work_unit)(plan, job, emit,
                            experiment=experiment['result'] if experiment else None)
                    if job['kind'] == 'experiment':
                        if ([c.get('slot') for c in result.get('claims', [])] != list(SLOTS)
                                or any(type(c.get('accepted')) is not bool for c in result['claims'])):
                            raise ValueError('Five distinct ordered claim slots required')
                    elif set(result.get('risks', {})) != set(SLOTS):
                        raise ValueError('Oracle must report all five risk slots explicitly')
                    if binding(plan, evaluator_id=evaluator_id if evaluator else None) != frozen:
                        raise ValueError('Source changed during calibration work')
                    if time.monotonic() - started >= plan.unit_seconds:
                        raise RuntimeError('wall_budget_exhausted_before_publication')
                    if evaluator is None and psutil.Process().memory_info().rss > plan.memory_mb * 2**20:
                        raise RuntimeError('memory_budget_exhausted_before_publication')
                    row.update(status='complete', result=result)
                except (ValueError, RuntimeError, FloatingPointError, MemoryError, np.linalg.LinAlgError) as exc:
                    row.update(status='failed_consumed', error=str(exc)[:500], error_type=type(exc).__name__)
                row.update(elapsed_seconds_charged=time.monotonic() - started,
                           time_charge_scope='observed_work_and_publication_elapsed')
                save(row)
                count += 1
            _verify(db, directory)
            report = _report(plan, _read(db), test_only=evaluator is not None)
            report['binding_id'] = fingerprint(frozen)
            # Interim and endpoint reports are diagnostics, never passing certificates.
            immutable_record(directory / 'reports' / f"{report['work_units']:08d}-{fingerprint(report)}.json", report)
            return report
        finally:
            db.close()


def inspect_full_pipeline_calibration(directory):
    """Read-only replay of source, reservation, journal and diagnostic accounting."""
    directory = project_path(directory)
    identity = json.loads((directory / 'identity.json').read_text('utf-8'))
    frozen = identity['binding']
    plan = FullCalibrationPlan.from_dict(frozen['plan'])
    current = binding(plan, evaluator_id=frozen['evaluator_id'] if frozen['test_only'] else None)
    if current != frozen or fingerprint(frozen) != identity['digest']:
        raise ValueError('Calibration source or identity changed')
    for relative, digest in frozen['sources'].items():
        if file_hash(directory / 'source' / relative) != digest:
            raise ValueError('Archived source changed')
    uri = (directory / 'calibration.sqlite').resolve().as_uri() + '?mode=ro'
    with sqlite3.connect(uri, uri=True) as db:
        _verify(db, directory)
        report = _report(plan, _read(db), test_only=frozen['test_only'])
    report['binding_id'] = fingerprint(frozen)
    return report
