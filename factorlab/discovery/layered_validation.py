"""Frozen-loss scoring/selection calibration, never a GPT-policy certificate.

The finite loss law has exact binary64 population risks. Each independent
repetition selects within a prospectively fixed candidate list and evaluates
five correlated claims at one fixed endpoint with the production interval.
Whole execution failures remain possible family errors and zero discoveries.
No feature fitting, provider law or arbitrary-market guarantee is covered.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from itertools import product
from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np
import pandas as pd

from factorlab.protocol import fingerprint, project_path
from .calibration_acceptance import exact_binomial_bounds
from .confirmatory import ConfirmationRule, paired_effect_interval
from .v7_run import source_files

VERSION = 'v7-layered-frozen-loss-selection-v1'
SCOPE = 'frozen_candidate_loss_scoring_and_selection'
CLAIMS = ('library_skill', 'factor1', 'factor2', 'factor3', 'algorithm_stream')
SCENARIOS = ('serial', 'regime', 'execution_failures')
EFFECTS = ('0', '0.003', '0.005', '0.01', '0.02', 'mixed')
EXCLUDED = ('GPT_population_distribution', 'feature_and_parameter_fitting',
            'adaptive_architecture_training', 'whole_autonomous_algorithm_power',
            'arbitrary_loss_laws_and_market_drift', 'arbitrary_nominal_failure_laws')
B = 2**20


@dataclass(frozen=True)
class LayeredPlan:
    repetitions: int = 1000
    seed: int = 6101101
    phase: str = 'validation'
    candidates: int = 24
    search_hours: int = 720
    validation_hours: int = 720
    confirmation_hours: int = 2160
    wall_seconds: int = 7200
    noise_multiplier: float = 1.
    confidence_family_size: int = 1

    def __post_init__(self):
        if (type(self.repetitions) is not int or not 20 <= self.repetitions <= 2000
                or type(self.seed) is not int or not 0 <= self.seed < 2**32
                or self.phase not in ('design', 'validation')
                or type(self.candidates) is not int or self.candidates != 24
                or self.search_hours != 720 or self.validation_hours != 720
                or self.confirmation_hours not in (2160, 4320, 4416)
                or type(self.wall_seconds) is not int or not 1 <= self.wall_seconds <= 7200
                or type(self.noise_multiplier) is not float
                or self.noise_multiplier not in (1., 5., 10.)
                or type(self.confidence_family_size) is not int
                or self.confidence_family_size not in (1, 3)):
            raise ValueError('Invalid fixed layered score/selection protocol')

    @property
    def rule(self):
        return ConfirmationRule()  # Original 0.005 / 0.01 / 168h / 10000 bootstrap.

    @property
    def digest(self):
        return fingerprint({'version': VERSION, 'scope': SCOPE, 'plan': asdict(self),
            'rule': self.rule.to_dict(), 'claims': CLAIMS, 'scenarios': SCENARIOS,
            'effects': EFFECTS, 'failure_loss': B,
            'failure_policy': 'possible_family_error_one_discovery_zero_no_retry',
            'excluded': EXCLUDED})


def _centers(effect):
    # Upward rounding ensures the decimal 0.5% boundary is never a tiny
    # accidental binary64 positive. The enumerated law below is the truth.
    value = float(1 - Fraction(effect))
    for _ in range(8):
        value = math.nextafter(value, math.inf)
    return value


def _effect_lists(label):
    if label not in EFFECTS:
        raise ValueError('Unregistered effect cell')
    if label != 'mixed':
        # Include all-null/boundary controls in positive candidate families.
        values = ([label] * 24 if Fraction(label) <= Fraction('0.005')
                  else [label] * 4 + ['0', '0.003', '0.005'] * 6 + ['0', '0.005'])
        return [values[:] for _ in CLAIMS]
    return [([label] * 4 + ['0', '0.003', '0.005'] * 6 + ['0', '0.005'])
            for label in ('0', '0.01', '0.005', '0.02', '0.003')]


def _amplitudes(plan, scenario, block_count):
    if scenario not in SCENARIOS:
        raise ValueError('Unregistered stress scenario')
    # Held-out validation law differs from the design law before any results.
    scale = 1. if plan.phase == 'design' else 1.25
    amplitude = np.full(block_count, .002 * scale * plan.noise_multiplier)
    if scenario == 'regime':
        amplitude[block_count // 3:2 * block_count // 3] *= 1.5
        amplitude[2 * block_count // 3:] *= .75
    return amplitude


def law_risks(plan, scenario, effect_label):
    """Exact risk of the *confirmation* law, conditional on frozen candidates.

    Both noise signs have probability 1/2. Risks enumerate the four binary64
    loss values actually generated, including deterministic regime weights.
    Runtime faults are separate all-trial error/power failures, not deleted
    observations or renamed valid nominal paths.
    """
    n = plan.confirmation_hours
    block = 24
    count = (n + block - 1) // block
    amplitudes = _amplitudes(plan, scenario, count)
    weights = Counter()
    for b, amplitude in enumerate(amplitudes):
        weights[float(amplitude)] += min(block, n - block * b)
    cache = {}
    result = []
    for effects in _effect_lists(effect_label):
        candidates = []
        for effect in effects:
            if effect not in cache:
                center = _centers(effect)
                risk = Fraction()
                for common, weight in weights.items():
                    values = [float(float(center + sign * common) + individual * .001 * plan.noise_multiplier)
                              for sign, individual in product((-1, 1), repeat=2)]
                    risk += weight * sum(map(Fraction.from_float, values), Fraction()) / 4
                cache[effect] = risk / n
            risk = cache[effect]
            candidates.append({'reference': [1, 1],
                'candidate': [risk.numerator, risk.denominator],
                'relative_gain': float(1 - risk),
                'alternative': 1 - risk > Fraction('0.005')})
        result.append(candidates)
    return result


def _stage(rng, plan, scenario, effect_label, hours):
    count = (hours + 23) // 24
    amplitude = _amplitudes(plan, scenario, count)
    common = rng.choice((-1., 1.), size=(count, 1, 1)) * amplitude[:, None, None]
    individual = rng.choice((-1., 1.), size=(count, 5, 24)) * .001 * plan.noise_multiplier
    centers = np.array([[_centers(e) for e in values] for values in _effect_lists(effect_label)])
    # Keep the same ordered operations as the exact finite-law enumerator.
    return np.repeat((centers[None, :, :] + common) + individual, 24, axis=0)[:hours]


def trial(plan, scenario, effect_label, index, *, clock=time.monotonic, deadline=None):
    """One independent synthetic scoring/filter repetition; no model fits."""
    if type(index) is not int or not 0 <= index < plan.repetitions:
        raise ValueError('Trial index is outside the fixed denominator')
    scenario_index = SCENARIOS.index(scenario)
    effect_index = EFFECTS.index(effect_label)
    phase_index = 0 if plan.phase == 'design' else 1
    rng = np.random.default_rng(np.random.SeedSequence(
        [plan.seed, phase_index, scenario_index, effect_index, index]))
    identity = {'scenario': scenario, 'effect': effect_label, 'index': index,
                'plan_digest': plan.digest}
    try:
        if deadline is not None and clock() >= deadline:
            raise TimeoutError('Original scoring validation deadline')
        # A known 1% complete-execution-fault stratum. It tests failure
        # accounting, not a statistical claim about real OS failure rates.
        if scenario == 'execution_failures' and rng.integers(0, 100) == 0:
            raise OSError('predeclared_synthetic_execution_failure')
        search = _stage(rng, plan, scenario, effect_label, plan.search_hours)
        validation = _stage(rng, plan, scenario, effect_label, plan.validation_hours)
        confirmation = _stage(rng, plan, scenario, effect_label, plan.confirmation_hours)
        risks = law_risks(plan, scenario, effect_label)
        origins = pd.date_range('1970-01-01', periods=plan.confirmation_hours, freq='h', tz='UTC')
        base = np.ones(plan.confirmation_hours)
        claims = []
        for slot, name in enumerate(CLAIMS):
            if deadline is not None and clock() >= deadline:
                raise TimeoutError('Original scoring validation deadline')
            short = np.argsort(search[:, slot].mean(0), kind='stable')[:3]
            eligible = [int(k) for k in short if 1 - validation[:, slot, k].mean() > .005]
            selected = min(eligible, key=lambda k: (validation[:, slot, k].mean(), k)) if eligible else None
            interval = None if selected is None else paired_effect_interval(
                origins, base, confirmation[:, slot, selected], rule=plan.rule,
                eligible_count=plan.confirmation_hours, calibration_passed=False)
            # Diagnostic C decision only. No boolean is passed to unlock any
            # lifecycle, promotion or old full-algorithm certificate.
            accepted = bool(interval and interval.get('diagnostic_status') == 'supported_target')
            alternative = False if selected is None else risks[slot][selected]['alternative']
            claims.append({'slot': name, 'shortlist': list(map(int, short)), 'selected': selected,
                'accepted': accepted, 'selected_alternative': alternative,
                'true_discovery': accepted and alternative,
                'false_discovery': accepted and not alternative,
                'known_law_selected_risk': None if selected is None else risks[slot][selected],
                'score_readout': interval})
        return {**identity, 'state': 'complete', 'claims': claims,
                'possible_family_error': any(c['false_discovery'] for c in claims)}
    except (ValueError, OSError, ArithmeticError, RuntimeError, MemoryError) as error:
        return {**identity, 'state': 'failed_consumed', 'claims': [],
                'possible_family_error': True, 'discovery_successes': [False] * 5,
                'reason': type(error).__name__ + ':' + str(error)[:300]}


def summarize(plan, rows):
    """All registered trials or missing failures; simultaneous one-sided CP."""
    expected = {(s, e, i) for s in SCENARIOS for e in EFFECTS for i in range(plan.repetitions)}
    actual = {}
    for row in rows:
        key = row['scenario'], row['effect'], row['index']
        if key not in expected or key in actual or row.get('plan_digest') != plan.digest:
            raise ValueError('Foreign, duplicate or changed layered trial')
        if row.get('state') not in ('complete', 'failed_consumed'):
            raise ValueError('Unknown trial state')
        if row['state'] == 'complete':
            if (len(row.get('claims', [])) != 5
                    or [c['slot'] for c in row['claims']] != list(CLAIMS)
                    or any(type(c.get('accepted')) is not bool for c in row['claims'])):
                raise ValueError('Incomplete five-score trial')
        actual[key] = row
    risks = {(s, e): law_risks(plan, s, e) for s in SCENARIOS for e in EFFECTS}
    alternatives = {key: [any(r['alternative'] for r in slot) for slot in value]
                    for key, value in risks.items()}
    count = sum(1 + sum(flags) for flags in alternatives.values())
    tail_alpha = .025 / (count * plan.confidence_family_size)
    cells = []
    for scenario in SCENARIOS:
        for effect in EFFECTS:
            errors, failures = 0, 0
            successes = [0] * 5
            for index in range(plan.repetitions):
                row = actual.get((scenario, effect, index))
                if row is None or row['state'] != 'complete':
                    errors += 1
                    failures += 1
                    continue
                claims = row['claims']
                # Recompute truth and counts from the frozen law and decision.
                for slot, claim in enumerate(claims):
                    selected = claim['selected']
                    if selected is not None and (type(selected) is not int or not 0 <= selected < 24):
                        raise ValueError('Invalid selected fixed candidate')
                    positive = selected is not None and risks[scenario, effect][slot][selected]['alternative']
                    if claim['accepted'] and selected is None:
                        raise ValueError('Nonselection cannot be a discovery')
                    if selected is not None:
                        score = claim.get('score_readout')
                        if (type(score) is not dict or score.get('rule_id') != plan.rule.digest
                                or score.get('support') != plan.confirmation_hours
                                or score.get('eligible') != plan.confirmation_hours
                                or score.get('calibration_passed') is not False
                                or score.get('method') != 'hour_equal_gap_split_calendar_block_percentile_bootstrap'
                                or claim['known_law_selected_risk'] != risks[scenario, effect][slot][selected]):
                            raise ValueError('Selected score/risk is outside the frozen scoring scope')
                        interval = score.get('interval')
                        diagnostic = score.get('diagnostic_status')
                        decision = bool(interval is not None and len(interval) == 2
                                        and interval[0] > 0 and diagnostic == 'supported_target')
                        if claim['accepted'] != decision:
                            raise ValueError('Decision differs from original score interval')
                    elif claim.get('score_readout') is not None:
                        raise ValueError('Nonselection contains a foreign confirmation score')
                    successes[slot] += int(claim['accepted'] and positive)
                errors += int(any(c['accepted'] and not risks[scenario, effect][slot][c['selected']]['alternative']
                                  for slot, c in enumerate(claims) if c['selected'] is not None))
            upper = exact_binomial_bounds(errors, plan.repetitions, tail_alpha=tail_alpha)[1]
            power = {name: {'successes': successes[slot], 'denominator': plan.repetitions,
                      'lower': exact_binomial_bounds(successes[slot], plan.repetitions,
                                                    tail_alpha=tail_alpha)[0]}
                     for slot, name in enumerate(CLAIMS) if alternatives[scenario, effect][slot]}
            cells.append({'scenario': scenario, 'effect': effect, 'failures_in_denominator': failures,
                'possible_family_errors': errors, 'denominator': plan.repetitions,
                'fwer_upper': upper, 'power': power,
                'criteria_met': upper <= .05 and all(v['lower'] >= .8 for v in power.values())})
    return {'version': VERSION, 'scope': SCOPE, 'plan_digest': plan.digest,
        'phase': plan.phase, 'independent_unit': 'one_complete_fixed_list_selection_repetition',
        'registered_trials': len(expected), 'retained_trials': len(actual),
        'simultaneous_MC_alpha': .025, 'tail_alpha': tail_alpha,
        'confidence_family_size': plan.confidence_family_size,
        'cells': cells, 'criteria_met': plan.phase == 'validation' and plan.repetitions >= 1000
                    and all(c['criteria_met'] for c in cells),
        'production_certificate_issued': False, 'formal_gate_eligible': False,
        'full_V7_five_claim_pipeline_calibrated': False,
        'covered': ['production_paired_interval', 'fixed_candidates_shortlist_and_one_look_selection',
                    'known_finite_loss_law', 'all_trial_failure_accounting'],
        'excluded': list(EXCLUDED), 'failure_loss': B,
        'law_assumptions': {'baseline_loss': 1., 'conditional_losses': 'enumerated_four_sign_binary64_law',
            'within_stage_dependence_hours': 24, 'stages': 'independent_fixed_random_substreams',
            'noise_multiplier': plan.noise_multiplier,
            'real_market_law_match_verified': False, 'arbitrary_missing_hour_patterns': False}}


def binding(plan):
    return {'version': VERSION, 'scope': SCOPE, 'plan': asdict(plan), 'plan_digest': plan.digest,
            'source_files': source_files(),
            'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'excluded': list(EXCLUDED), 'production_certificate_issued': False}


def _write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)


def verify_report(directory, *, expected_binding=None):
    """Recompute all registered denominators and scope; never grants old gates."""
    directory = project_path(directory)
    def read(path):
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError('Duplicate layered evidence JSON key')
                result[key] = value
            return result
        def constant(value):
            raise ValueError('Nonfinite layered evidence: ' + value)
        if not path.is_file() or path.stat().st_size > 4 * 2**20:
            raise ValueError('Missing or oversized layered evidence')
        return json.loads(path.read_text('utf-8'), object_pairs_hook=pairs, parse_constant=constant)
    original = read(directory / 'preregistration.json')
    plan = LayeredPlan(**original['plan'])
    if binding(plan) != original:
        raise ValueError('Layered source/implementation/plan changed')
    if expected_binding is not None and fingerprint(original) != expected_binding:
        raise ValueError('Layered evidence does not match requested binding')
    expected_names = {f'{s}-{e}-{i:04d}.json' for s in SCENARIOS for e in EFFECTS for i in range(plan.repetitions)}
    allowed = expected_names | {'preregistration.json', 'report.json', 'implementation.py'}
    if any(p.name not in allowed or not p.is_file() for p in directory.iterdir()):
        raise ValueError('Foreign layered evidence file')
    if ((directory / 'implementation.py').read_bytes() != Path(__file__).read_bytes()
            or hashlib.sha256((directory / 'implementation.py').read_bytes()).hexdigest()
                != original['implementation_sha256']):
        raise ValueError('Archived layered implementation changed')
    rows = [read(directory / name) for name in sorted(expected_names) if (directory / name).exists()]
    for row in rows:
        if row.get('state') == 'complete':
            reproduced = trial(plan, row['scenario'], row['effect'], row['index'])
            if reproduced != row:
                raise ValueError('Complete trial is not the original deterministic scoring execution')
    report = read(directory / 'report.json')
    recomputed = summarize(plan, rows)
    if any(report.get(key) != value for key, value in recomputed.items()):
        raise ValueError('Layered report cannot be reproduced from original trials')
    if (report.get('binding_hash') != fingerprint(original) or report.get('fits') != 0
            or report.get('GPT_calls') != 0 or report.get('hard_deadline_enforced') is not False):
        raise ValueError('Layered computation/scope metadata changed')
    return report


def run(directory, plan):
    """Once-only finite diagnostic; no retries, unrun coordinates remain failures."""
    directory = project_path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    original = binding(plan)
    _write(directory / 'preregistration.json', original)
    with (directory / 'implementation.py').open('xb') as stream:
        stream.write(Path(__file__).read_bytes())
    deadline = time.monotonic() + plan.wall_seconds
    rows = []
    for scenario in SCENARIOS:
        for effect in EFFECTS:
            for index in range(plan.repetitions):
                if time.monotonic() >= deadline:
                    break  # summarize retains every unrun coordinate as a failure.
                result = trial(plan, scenario, effect, index, deadline=deadline)
                _write(directory / f'{scenario}-{effect}-{index:04d}.json', result)
                rows.append(result)
    if binding(plan) != original:
        raise ValueError('Source changed across layered scoring run')
    report = {**summarize(plan, rows), 'binding_hash': fingerprint(original),
              'completed_at': datetime.now(timezone.utc).isoformat(),
              'fits': 0, 'GPT_calls': 0, 'hard_deadline_enforced': False}
    _write(directory / 'report.json', report)
    return report
