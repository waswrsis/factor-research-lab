"""Conservative fixed-grid Monte Carlo acceptance, never a scope certificate.

Each observation is ONE independent end-to-end repetition, not one hour,
symbol, seed fit or discovered candidate. Missing repetitions and failures
cannot improve error or power estimates. Risk envelopes must come from a
separate, frozen oracle producer; this numeric module does not authenticate
that producer, GPT identity, service execution or resource observations.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from fractions import Fraction

import numpy as np
from scipy.stats import beta

from factorlab.protocol import fingerprint

CLAIMS = ('library_skill', 'factor1', 'factor2', 'factor3', 'algorithm')
TRIAL_STATES = {'complete', 'failed_consumed', 'interrupted_consumed'}
DELTA0 = .005


def _finite(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Finite numeric ' + label + ' required')
    return float(value)


def exact_binomial_bounds(successes, total, *, tail_alpha):
    """Two separately one-sided Clopper--Pearson bounds at alpha each.

    The caller budgets each reported direction, rather than interpreting a
    two-sided 95% interval as a one-sided 95% interval. Boundary cases are
    exact, including zero errors and zero completed repetitions.
    """
    if (type(successes) is not int or type(total) is not int
            or not 0 <= successes <= total or total < 1):
        raise ValueError('Valid integer independent-repetition counts required')
    a = _finite(tail_alpha, 'tail alpha')
    if not 0 < a < .5:
        raise ValueError('Tail alpha must be strictly between zero and .5')
    low = 0. if successes == 0 else float(beta.ppf(a, successes, total - successes + 1))
    high = 1. if successes == total else float(beta.ppf(1 - a, successes + 1, total - successes))
    if not 0 <= low <= high <= 1:
        raise ArithmeticError('Binomial inversion returned invalid bounds')
    return low, high


@dataclass(frozen=True)
class AcceptanceCell:
    """One independently repeated scenario/effect/provider stratum.

    Power targets are registered BEFORE any search. A nonselection contributes
    zero to discovery power. Postselecting only repetitions with an alternative
    learner would instead estimate conditional power and is not allowed here.
    """
    name: str
    scenario_digest: str
    repetitions: int
    power_slots: tuple[str, ...] = ()

    def __post_init__(self):
        if (not isinstance(self.name, str) or not self.name or len(self.name) > 80
                or not isinstance(self.scenario_digest, str) or len(self.scenario_digest) != 64
                or any(c not in '0123456789abcdef' for c in self.scenario_digest)
                or type(self.repetitions) is not int or not 1000 <= self.repetitions <= 2000
                or type(self.power_slots) is not tuple or len(set(self.power_slots)) != len(self.power_slots)
                or not set(self.power_slots) <= set(CLAIMS)):
            raise ValueError('Invalid frozen acceptance cell')


@dataclass(frozen=True)
class AcceptancePlan:
    cells: tuple[AcceptanceCell, ...]
    monte_carlo_alpha: float = .025
    oracle_alpha: float = .025
    family_error_target: float = .05
    discovery_power_target: float = .8
    delta0: float = DELTA0

    def __post_init__(self):
        if (type(self.cells) is not tuple or not 1 <= len(self.cells) <= 64
                or any(type(c) is not AcceptanceCell for c in self.cells)
                or len({c.name for c in self.cells}) != len(self.cells)
                or self.monte_carlo_alpha != .025 or self.oracle_alpha != .025
                or self.family_error_target != .05 or self.discovery_power_target != .8
                or self.delta0 != DELTA0):
            raise ValueError('Frozen .05 confidence budget and .05/.80 targets required')

    @property
    def bound_count(self):
        return sum(1 + len(c.power_slots) for c in self.cells)

    @property
    def tail_alpha(self):
        return self.monte_carlo_alpha / self.bound_count

    @property
    def oracle_risk_alpha(self):
        # Separate lower/upper bounds on TWO risks per registered claim/trial.
        return self.oracle_alpha / (4 * len(CLAIMS) * sum(c.repetitions for c in self.cells))

    def to_dict(self):
        return asdict(self)

    @property
    def digest(self):
        return fingerprint({'version': 'v7-fixed-grid-calibration-acceptance-v1',
                            'plan': self.to_dict(), 'unit': 'independent_full_pipeline_repetition',
                            'error_policy': 'missing_or_failed_is_possible_family_error',
                            'power_policy': 'unconditional_predeclared_slot_discovery'})


@dataclass(frozen=True)
class OracleRiskEnvelope:
    """Numeric conditional-risk enclosure with explicit original oracle scope.

    Merely instantiating this type does NOT certify truth. The production
    verifier must reconstruct it from the original independent oracle artifact,
    including its loss bounds, generator identity, sample count and budget.
    """
    reference: tuple[float, float]
    candidate: tuple[float, float]
    method: str
    oracle_digest: str
    per_risk_tail_alpha: float

    def __post_init__(self):
        if (self.method not in {'finite_exhaustive_risk_enclosure', 'bounded_iid_path_hoeffding'}
                or type(self.reference) is not tuple or type(self.candidate) is not tuple
                or len(self.reference) != 2 or len(self.candidate) != 2
                or not isinstance(self.oracle_digest, str) or len(self.oracle_digest) != 64
                or any(c not in '0123456789abcdef' for c in self.oracle_digest)):
            raise ValueError('Unsupported oracle method or risk enclosure identity')
        for pair in (self.reference, self.candidate):
            lo, hi = (_finite(v, 'oracle risk') for v in pair)
            if not 0 <= lo <= hi:
                raise ValueError('Nonnegative ordered oracle risk bounds required')
        a = _finite(self.per_risk_tail_alpha, 'oracle tail alpha')
        if not 0 < a < .5:
            raise ValueError('Invalid independent oracle confidence budget')

    def effect_interval(self):
        b_lo, b_hi = self.reference
        c_lo, c_hi = self.candidate
        # A zero reference risk cannot identify a relative effect. Even if
        # the candidate is also perfect, do not label that 0/0 an alternative.
        if b_lo <= 0:
            return None
        # Divide/subtract exact binary endpoints as rational numbers, then
        # widen the displayed float interval. Near delta0 a single rounded
        # ratio must never turn a boundary hypothesis into an alternative.
        low = 1 - Fraction.from_float(float(c_hi)) / Fraction.from_float(float(b_lo))
        high = 1 - Fraction.from_float(float(c_lo)) / Fraction.from_float(float(b_hi))
        try:
            values = float(low), float(high)
        except OverflowError:
            return None
        if not all(math.isfinite(v) for v in values):
            return None
        return (math.nextafter(values[0], -math.inf), math.nextafter(values[1], math.inf))

    def truth(self, plan):
        if type(plan) is not AcceptancePlan:
            raise TypeError('Actual fixed acceptance plan required')
        if self.per_risk_tail_alpha > plan.oracle_risk_alpha:
            raise ValueError('Oracle confidence exceeds the full registered-grid budget')
        effect = self.effect_interval()
        if effect is None:
            return 'uncertain'
        # The scientific threshold is exactly 1/200, not a rounded quotient.
        # Also widen oracle endpoints by one binary64 ulp before classifying;
        # upstream finite enumeration must account for its own arithmetic.
        b_lo, b_hi = self.reference
        c_lo, c_hi = self.candidate
        b_lo = math.nextafter(float(b_lo), -math.inf)
        b_hi = math.nextafter(float(b_hi), math.inf)
        c_lo = max(0., math.nextafter(float(c_lo), -math.inf))
        c_hi = math.nextafter(float(c_hi), math.inf)
        if b_lo <= 0 or not math.isfinite(b_hi) or not math.isfinite(c_hi):
            return 'uncertain'
        high = 1 - Fraction.from_float(c_lo) / Fraction.from_float(b_hi)
        low = 1 - Fraction.from_float(c_hi) / Fraction.from_float(b_lo)
        threshold = Fraction(1, 200)
        if high <= threshold:
            return 'null'
        if low > threshold:
            return 'alternative'
        return 'uncertain'


def bounded_path_envelope(reference_loss, candidate_loss, *, loss_upper_bound,
                          expected_paths, oracle_digest, plan):
    """Bound risks using independent WHOLE-path loss averages.

    The bound and sample count must be frozen by the independent producer
    before generating paths. Arrays are one-dimensional path aggregates;
    dependent hourly losses cannot be flattened into extra observations.
    An unbounded/heavy-tail oracle needs another proved method and is refused.
    This computes the inequality; it cannot prove the asserted iid/bound facts.
    """
    if (type(plan) is not AcceptancePlan or type(expected_paths) is not int
            or not 2 <= expected_paths <= 2_000_000):
        raise ValueError('Fixed oracle plan and whole-path sample count required')
    upper = _finite(loss_upper_bound, 'registered loss upper bound')
    if upper <= 0:
        raise ValueError('Strictly positive predeclared loss bound required')
    arrays = [np.asarray(v, dtype=float) for v in (reference_loss, candidate_loss)]
    if any(a.shape != (expected_paths,) or not np.isfinite(a).all()
           or np.any(a < 0) or np.any(a > upper) for a in arrays):
        raise ValueError('Oracle path losses violate frozen sample count or loss bounds')
    alpha = plan.oracle_risk_alpha
    radius = upper * math.sqrt(math.log(1 / alpha) / (2 * expected_paths))
    ranges = [(max(0., float(a.mean()) - radius), min(upper, float(a.mean()) + radius))
              for a in arrays]
    return OracleRiskEnvelope(ranges[0], ranges[1], 'bounded_iid_path_hoeffding',
                              oracle_digest, alpha)


@dataclass(frozen=True)
class ClaimObservation:
    slot: str
    accepted: bool
    oracle: OracleRiskEnvelope | None

    def __post_init__(self):
        if (self.slot not in CLAIMS or type(self.accepted) is not bool
                or self.oracle is not None and type(self.oracle) is not OracleRiskEnvelope):
            raise ValueError('Typed five-claim observation required')


@dataclass(frozen=True)
class TrialObservation:
    cell: str
    index: int
    state: str
    execution_digest: str
    claims: tuple[ClaimObservation, ...]

    def __post_init__(self):
        if (not isinstance(self.cell, str) or type(self.index) is not int or self.index < 0
                or self.state not in TRIAL_STATES or not isinstance(self.execution_digest, str)
                or len(self.execution_digest) != 64
                or any(c not in '0123456789abcdef' for c in self.execution_digest)
                or type(self.claims) is not tuple
                or any(type(c) is not ClaimObservation for c in self.claims)
                or tuple(c.slot for c in self.claims) != CLAIMS):
            raise ValueError('Exactly five ordered claims and original execution identity required')
        if self.state != 'complete' and any(c.accepted for c in self.claims):
            raise ValueError('Failed trial cannot report a successful formal discovery')


def acceptance_report(plan, observations):
    """Calculate conservative error/power bounds without creating permissions.

    Scope authentication and independence are NOT provided by a dataclass or
    these numbers. Even a numeric pass has formal_gate_eligible=False here.
    A full producer/verifier must check actual execution and oracle artifacts.
    """
    if type(plan) is not AcceptancePlan:
        raise TypeError('Frozen acceptance plan required')
    cells = {c.name: c for c in plan.cells}
    seen, execution_ids, rows = set(), set(), {c.name: {} for c in plan.cells}
    for observation in observations:
        if type(observation) is not TrialObservation or observation.cell not in cells:
            raise ValueError('Observation is outside the frozen validation grid')
        key = observation.cell, observation.index
        if observation.index >= cells[observation.cell].repetitions or key in seen:
            raise ValueError('Duplicate or out-of-range validation repetition')
        if observation.execution_digest in execution_ids:
            raise ValueError('The same execution cannot count as independent repetitions')
        seen.add(key)
        execution_ids.add(observation.execution_digest)
        rows[observation.cell][observation.index] = observation
    summaries = []
    for cell in plan.cells:
        completed, missing, failures = 0, 0, 0
        certain_errors, possible_errors = 0, 0
        discoveries = {s: 0 for s in cell.power_slots}
        unknown_truth, alternative_trials, mixed_trials = 0, 0, 0
        for index in range(cell.repetitions):
            row = rows[cell.name].get(index)
            if row is None or row.state != 'complete':
                missing += int(row is None)
                failures += int(row is not None)
                possible_errors += 1
                continue
            completed += 1
            truths = {c.slot: 'uncertain' if c.oracle is None else c.oracle.truth(plan)
                      for c in row.claims}
            certain_errors += int(any(c.accepted and truths[c.slot] == 'null' for c in row.claims))
            possible_errors += int(any(c.accepted and truths[c.slot] != 'alternative' for c in row.claims))
            unknown_truth += int(any(v == 'uncertain' for v in truths.values()))
            alternative_trials += int(any(v == 'alternative' for v in truths.values()))
            mixed_trials += int('null' in truths.values() and 'alternative' in truths.values())
            for c in row.claims:
                if c.slot in discoveries:
                    discoveries[c.slot] += int(c.accepted and truths[c.slot] == 'alternative')
        _, error_upper = exact_binomial_bounds(possible_errors, cell.repetitions,
                                              tail_alpha=plan.tail_alpha)
        power = {}
        for slot, count in discoveries.items():
            power_lower, _ = exact_binomial_bounds(count, cell.repetitions, tail_alpha=plan.tail_alpha)
            power[slot] = {'discoveries': count, 'denominator': cell.repetitions,
                           'lower': power_lower, 'target': plan.discovery_power_target,
                           'target_met': power_lower >= plan.discovery_power_target}
        summaries.append({'cell': cell.name, 'scenario_digest': cell.scenario_digest,
            'registered_repetitions': cell.repetitions, 'complete': completed,
            'missing': missing, 'failed': failures, 'certain_family_errors': certain_errors,
            'possible_family_errors': possible_errors, 'family_error_upper': error_upper,
            'error_target_met': error_upper <= plan.family_error_target,
            'unknown_truth_trials': unknown_truth, 'alternative_trials': alternative_trials,
            'mixed_truth_trials': mixed_trials, 'discovery_power': power})
    # Missing/failed trials already enter the fixed-denominator bounds at the
    # worst case. Completion remains a separate engineering/certificate gate.
    complete = all(s['missing'] == 0 and s['failed'] == 0 for s in summaries)
    return {'version': 'v7-fixed-grid-calibration-acceptance-v1', 'plan_digest': plan.digest,
        'plan': plan.to_dict(), 'cells': summaries, 'all_trials_complete': complete,
        'numeric_targets_met': all(s['error_target_met'] and
            all(p['target_met'] for p in s['discovery_power'].values()) for s in summaries),
        'confidence': {'monte_carlo_alpha': plan.monte_carlo_alpha, 'oracle_alpha': plan.oracle_alpha,
            'joint_failure_budget': plan.monte_carlo_alpha + plan.oracle_alpha,
            'directional_bounds': plan.bound_count, 'per_bound_tail_alpha': plan.tail_alpha,
            'per_oracle_risk_tail_alpha': plan.oracle_risk_alpha,
            'method': 'one_sided_exact_binomial_Bonferroni_plus_independent_oracle_union_budget'},
        'power_is_end_to_end_discovery_probability': True,
        'formal_gate_eligible': False, 'full_V7': False,
        'execution_scope_authenticated': False, 'oracle_scope_authenticated': False,
        'required_producer_checks': ['independent_frozen_validation_generator',
            'all_real_pipeline_executions_and_original_budgets', 'exact_provider_and_source_identity',
            'five_claim_controller_and_resource_scope', 'independent_original_oracle_artifacts'],
        'reason': 'Numeric acceptance does not authenticate the production or oracle scope'}
