"""Predeclared fixed-endpoint paired effect intervals and control diagnostics.

Calendar block resampling is approximate under dependent, drifting markets.
This module cannot issue a statistical calibration certificate by itself.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from factorlab.protocol import fingerprint
from factorlab.time_utils import hour_ids


@dataclass(frozen=True)
class ConfirmationRule:
    delta0: float = 0.005
    alpha: float = 0.01
    block_hours: int = 168
    minimum_blocks: int = 12
    bootstrap_samples: int = 10000
    seed: int = 1701
    horizon_hours: int = 12
    receipt_grace_hours: int = 24
    coverage: float = 0.95

    def __post_init__(self):
        if self.delta0 != .005 or self.alpha != .01 or self.receipt_grace_hours != 24 or self.coverage != .95:
            raise ValueError('V7 freezes effect, alpha, grace and coverage before data')
        for key in ('block_hours', 'minimum_blocks', 'bootstrap_samples', 'seed', 'horizon_hours'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f'Invalid {key}')
        if self.minimum_blocks < 12 or self.bootstrap_samples < 1000:
            raise ValueError('Insufficient preregistered block/bootstrap budget')

    def to_dict(self):
        return asdict(self)

    @property
    def digest(self):
        return fingerprint({'rule': self.to_dict(), 'aggregation': 'equal_symbol_then_hour',
                            'blocks': 'calendar_and_real_gap_split_v1',
                            'support': 'minimum_full_length_fragments_v1'})


def paired_effect_interval(origins, baseline_loss, candidate_loss, *, rule=None,
                           eligible_count=None, calibration_passed=False,
                           symbols=None, expected_symbols=None):
    """CI for hourly d=(1-delta0)*Lb-Lc, with explicit cross-symbol weighting.

    eligible_count is always HOURS, never rows or symbol-hours. A multisymbol
    call must declare expected_symbols: missing whole symbols must not silently
    shrink coverage's denominator. It reports both hourly and symbol-hour
    coverage and requires both. Each (hour,symbol) occurs at most once; inputs
    without symbols must already contain exactly one observation per hour.

    Actual missing hours split calendar blocks. Partial fragments remain in the
    estimator, but only full block_hours fragments count toward minimum_blocks,
    so many short fragments cannot create apparent long-dependence support.
    The caller enforces the reserved endpoint and calibration certificate.
    """
    rule = ConfirmationRule() if rule is None else rule
    if type(calibration_passed) is not bool:
        raise ValueError('Calibration status must be an explicit boolean')
    base, candidate = np.asarray(baseline_loss, dtype=float), np.asarray(candidate_loss, dtype=float)
    hours = np.asarray(hour_ids(origins))
    if (base.ndim != 1 or base.shape != candidate.shape or len(hours) != len(base)
            or not np.isfinite(base).all() or not np.isfinite(candidate).all()
            or (base < 0).any() or (candidate < 0).any()):
        raise ValueError('Paired losses need identical finite nonnegative support')
    if symbols is None:
        if expected_symbols is not None:
            raise ValueError('Expected symbol universe requires explicit row symbols')
        if len(np.unique(hours)) != len(hours):
            raise ValueError('Single-symbol confirmation requires unique UTC hours')
        universe = ['single_symbol']
    else:
        if (not isinstance(expected_symbols, (tuple, list, set)) or not expected_symbols
                or any(not isinstance(s, str) or not s for s in expected_symbols)
                or len(set(expected_symbols)) != len(expected_symbols)):
            raise ValueError('Multisymbol confirmation requires an explicit unique symbol universe')
        universe = sorted(expected_symbols)
        labels = list(symbols)
        if len(labels) != len(hours) or any(not isinstance(s, str) or s not in universe for s in labels):
            raise ValueError('Row symbols differ from the expected universe')
        if len(set(zip(hours.tolist(), labels, strict=True))) != len(hours):
            raise ValueError('Duplicate UTC hour/symbol pairs must be resolved before confirmation')
    unique_hours, inverse = np.unique(hours, return_inverse=True)
    pair_counts = np.bincount(inverse, minlength=len(unique_hours))
    hourly_base = np.bincount(inverse, weights=base, minlength=len(unique_hours)) / np.maximum(pair_counts, 1)
    hourly_candidate = np.bincount(inverse, weights=candidate, minlength=len(unique_hours)) / np.maximum(pair_counts, 1)
    eligible = len(unique_hours) if eligible_count is None else eligible_count
    if type(eligible) is not int or eligible < len(unique_hours):
        raise ValueError('Eligible hours cannot be smaller than scored hours')
    coverage = len(unique_hours) / eligible if eligible else 0.
    pair_eligible = eligible * len(universe)
    pair_coverage = len(base) / pair_eligible if pair_eligible else 0.
    d = (1 - rule.delta0) * hourly_base - hourly_candidate
    calendar = unique_hours // rule.block_hours
    gap = np.diff(unique_hours) != 1
    cuts = np.r_[True, gap | (np.diff(calendar) != 0)] if len(unique_hours) else np.array([], dtype=bool)
    starts = np.flatnonzero(cuts)
    counts = np.diff(np.r_[starts, len(unique_hours)])
    sums = np.add.reduceat(d, starts) if len(starts) else np.array([], dtype=float)
    full_blocks = int((counts == rule.block_hours).sum())
    details = {'rule_id': rule.digest, 'method': 'hour_equal_gap_split_calendar_block_percentile_bootstrap',
               'approximate': True, 'support_unit': 'UTC_hour',
               'support': len(unique_hours), 'eligible': eligible, 'coverage': coverage,
               'pair_support': len(base), 'pair_eligible': pair_eligible,
               'pair_coverage': pair_coverage, 'expected_symbols': universe,
               'aggregation': 'equal_symbol_within_hour_then_equal_hour',
               'blocks': len(starts), 'full_blocks': full_blocks,
               'partial_blocks': int((counts < rule.block_hours).sum()),
               'minimum_block_support_policy': 'count_full_length_fragments_only',
               'gap_count': int(gap.sum()), 'block_hour_counts': counts.tolist(),
               'calibration_passed': calibration_passed,
               'relative_gain': float(1 - hourly_candidate.sum() / hourly_base.sum())
               if hourly_base.sum() > 0 else None,
               'absolute_delta': float(np.mean(hourly_base - hourly_candidate)) if len(d) else None,
               'd_mean': float(d.mean()) if len(d) else None}
    failures = []
    if full_blocks < rule.minimum_blocks:
        failures.append('insufficient_full_length_blocks')
    if coverage < rule.coverage:
        failures.append('insufficient_hour_coverage')
    if pair_coverage < rule.coverage:
        failures.append('insufficient_symbol_hour_coverage')
    if not len(d):
        failures.append('empty_support')
    if failures:
        return {**details, 'status': 'insufficient_evidence', 'supported': False,
                'interval': None, 'reasons': failures}
    rng = np.random.default_rng(rule.seed)
    estimates = np.empty(rule.bootstrap_samples)
    # Draw indices for fragments, never a bootstrap x full time x symbol tensor.
    # At most ~131k picks per chunk unless a single draw has more blocks.
    chunk = max(1, min(256, 131072 // len(starts)))
    for start in range(0, rule.bootstrap_samples, chunk):
        n = min(chunk, rule.bootstrap_samples - start)
        picks = rng.integers(0, len(starts), size=(n, len(starts)))
        estimates[start:start + n] = sums[picks].sum(1) / counts[picks].sum(1)
    lower, upper = np.quantile(estimates, [rule.alpha / 2, 1 - rule.alpha / 2])
    status = 'supported_target' if lower > 0 else 'target_not_reached' if upper <= 0 else 'indistinguishable'
    return {**details, 'interval': [float(lower), float(upper)], 'diagnostic_status': status,
            'status': status if calibration_passed else 'uncalibrated_diagnostic',
            'supported': bool(calibration_passed and lower > 0)}


def wilson_interval(successes, total, *, confidence=.95):
    if type(successes) is not int or type(total) is not int or not 0 <= successes <= total or total <= 0:
        raise ValueError('Invalid binomial counts')
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError('Confidence must be strictly between zero and one')
    z = norm.ppf((1 + confidence) / 2)
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [float(center - radius), float(center + radius)]


def injected_effect_diagnostics(*, repetitions=100, hours=24 * 120,
                                effects=(0., .003, .005, .01, .02), seed=712):
    """A cheap block-loss smoke diagnostic, NOT full search/selection calibration.

    Outputs explicitly deny formal-gate eligibility. A full mixed true/false
    candidate pipeline with drift/missingness and proposal mechanism is required
    before enabling production confirmation.
    """
    if type(repetitions) is not int or not 20 <= repetitions <= 5000:
        raise ValueError('Bounded diagnostic repetitions must be 20..5000')
    if type(hours) is not int or not 24 <= hours <= 24 * 365:
        raise ValueError('Diagnostic hours must be bounded within one year')
    if not effects or any(not np.isfinite(effect) or not 0 <= effect < 1 for effect in effects):
        raise ValueError('Diagnostic effects must be finite gains in [0,1)')
    rng = np.random.default_rng(seed)
    origins = pd.date_range('2024-01-01', periods=hours, freq='h', tz='UTC')
    rule = ConfirmationRule(bootstrap_samples=1000)
    counts = {str(effect): 0 for effect in effects}
    for _ in range(repetitions):
        shocks = rng.standard_t(5, hours) * .015
        serial = np.empty(hours)
        serial[0] = shocks[0]
        for i in range(1, hours):
            serial[i] = .8 * serial[i - 1] + shocks[i]
        base = np.exp(rng.normal(0, .15, hours))
        for effect in effects:
            candidate = np.maximum(base * (1 - effect) + serial, 1e-12)
            result = paired_effect_interval(origins, base, candidate, rule=rule)
            counts[str(effect)] += int(result.get('diagnostic_status') == 'supported_target')
    return {'scope': 'paired_loss_control_only', 'formal_gate_eligible': False,
            'full_pipeline_calibrated': False, 'repetitions': repetitions, 'seed': seed,
            'effect_threshold': rule.delta0,
            'curve': [{'effect': effect, 'accepts': counts[str(effect)],
                       'rate': counts[str(effect)] / repetitions,
                       'wilson_95': wilson_interval(counts[str(effect)], repetitions)} for effect in effects]}


def fixed_candidate_selection_diagnostics(*, repetitions=20, hours=24 * 120, seed=904,
                                          scenario_effects=None):
    """Mixed-truth five-claim S/V/C CONTROL for fixed candidate loss processes.

    Each synthetic claim has a fixed finite candidate list. S selects at most
    three; independent future V selects at most one above the frozen point
    threshold; C performs one diagnostic interval. All five C streams share a
    serial common shock so their errors are not assumed independent. The exact
    mean effects are known because the generated bounded noise is symmetric.

    This exercises winner selection and mixed false/true claims but DOES NOT
    include data ingestion, feature estimation, GPT, a search policy, drift or
    arbitrary missingness. It is not full research-pipeline calibration and
    cannot unlock promotion, irrespective of its empirical rejection rate.
    """
    if type(repetitions) is not int or not 20 <= repetitions <= 2000:
        raise ValueError('Selection-control repetitions must be 20..2000')
    if type(hours) is not int or not 24 * 100 <= hours <= 24 * 365:
        raise ValueError('Selection-control hours must be 100..365 days')
    if type(seed) is not int or seed < 0:
        raise ValueError('Selection-control seed must be a nonnegative integer')
    if scenario_effects is None:
        scenario_effects = ((0., 0., 0., 0.), (.005, .005, .003, 0.),
                            (0., .005, .01, .02), (.02, .01, .003, 0.),
                            (0., .003, .005, .005))
    effects = np.asarray(scenario_effects, dtype=float)
    if (effects.ndim != 2 or effects.shape[0] != 5 or not 1 <= effects.shape[1] <= 24
            or not np.isfinite(effects).all() or (effects < 0).any() or (effects >= .5).any()):
        raise ValueError('Exactly five bounded fixed candidate effect lists are required')
    rng = np.random.default_rng(seed)
    rule = ConfirmationRule(bootstrap_samples=1000)
    origins = pd.date_range('1970-01-01', periods=hours, freq='h', tz='UTC')
    base = np.ones(hours)
    false_trials, any_trials = 0, 0
    per_claim = [{'selected': [0] * effects.shape[1], 'no_selection': 0,
                  'true_accepts': 0, 'false_accepts': 0} for _ in range(5)]

    def stage(length):
        blocks = (length + 23) // 24
        common = np.repeat(rng.normal(0, .012, size=blocks), 24)[:length, None, None]
        individual = np.repeat(rng.normal(0, .008, size=(blocks, *effects.shape)), 24, axis=0)[:length]
        noise = np.clip(common + individual, -.1, .1)
        return 1 - effects[None, :, :] + noise

    for _ in range(repetitions):
        search, validation, confirmation = stage(24 * 30), stage(24 * 30), stage(hours)
        false_in_trial, any_in_trial = False, False
        for claim in range(5):
            short = np.argsort(search[:, claim, :].mean(0), kind='stable')[:3]
            eligible = [int(index) for index in short
                        if 1 - validation[:, claim, index].mean() > rule.delta0]
            if not eligible:
                per_claim[claim]['no_selection'] += 1
                continue
            selected = min(eligible, key=lambda index: (validation[:, claim, index].mean(), index))
            per_claim[claim]['selected'][selected] += 1
            result = paired_effect_interval(origins, base, confirmation[:, claim, selected], rule=rule)
            accepted = result.get('diagnostic_status') == 'supported_target'
            truth = bool(effects[claim, selected] > rule.delta0)
            if accepted:
                per_claim[claim]['true_accepts' if truth else 'false_accepts'] += 1
                false_in_trial |= not truth
                any_in_trial = True
        false_trials += int(false_in_trial)
        any_trials += int(any_in_trial)
    for i, values in enumerate(per_claim):
        values['effect_list'] = effects[i].tolist()
        values['has_true_candidate'] = bool((effects[i] > rule.delta0).any())
        values['true_pipeline_acceptance_rate'] = values['true_accepts'] / repetitions
        values['true_accepts_wilson_two_sided_95'] = wilson_interval(values['true_accepts'], repetitions)
    # A 90% equal-tail Wilson interval has one-sided 95% endpoints; each Monte
    # Carlo replicate, not each correlated claim, is the FWER Bernoulli unit.
    error_bound = wilson_interval(false_trials, repetitions, confidence=.90)[1]
    return {'scope': 'fixed_candidate_loss_S_V_C_selection_control',
            'formal_gate_eligible': False, 'full_pipeline_calibrated': False,
            'includes_gpt_or_fitted_features': False, 'repetitions': repetitions,
            'hours': hours, 'seed': seed, 'rule_id': rule.digest,
            'scenario_effects': effects.tolist(), 'claims': per_claim,
            'trials_with_any_accept': any_trials, 'trials_with_any_false_accept': false_trials,
            'empirical_fwer': false_trials / repetitions,
            'fwer_wilson_one_sided_95_upper': error_bound,
            'fwer_bound_is_calibration_certificate': False}
