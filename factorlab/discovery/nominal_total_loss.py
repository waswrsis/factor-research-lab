"""Unprivileged total nominal loss arithmetic, never execution authority.

Valid losses preserve the original binary64 logMSE operation exactly. The new
user-authorized estimand assigns 2**20 to unavailable or out-of-domain nominal
outputs. It does not turn missing paid work, infrastructure failures, or forged
records into completed nominal computations. Production readers must establish
source, frozen support and algorithm-failure provenance before using this math.
Old partial-domain and joint-service estimands retain their original identities.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from fractions import Fraction
import math

from .finite_market_oracle import LOSS_BOUND, _finite_loss

VERSION = 'nominal-total-loss-v2'
ESTIMAND = 'nominal_full_cold_rolling_total_loss_v2'
USER_CHOICE_SHA256 = '035fb78387f06970c4e4249c3728c2cf865a6857ebfce5fe7cf77be3a0d2b1f0'
PRIOR_CHOICE_SHA256 = '04d04b2571afe6dcf63ac13d88d2c24cc577a4e537bb65e6e9ffe291bf07559e'


def nominal_total_loss(y, p):
    """Return original accepted loss or the newly defined nominal worst loss.

    Strict original Python int/float domain is intentional: arbitrary objects
    and booleans cannot acquire valid prediction status via float conversion.
    Unexpected errors after accepted-domain validation propagate, so a broken
    implementation is not silently described as a proven algorithm outcome.
    """
    if type(y) not in (int, float) or type(p) not in (int, float):
        return float(LOSS_BOUND)
    try:
        y64, p64 = float(y), float(p)
    except OverflowError:
        return float(LOSS_BOUND)
    if not math.isfinite(y64) or not math.isfinite(p64) or min(y64, p64) < 0:
        return float(LOSS_BOUND)
    # Preserve original np.log evaluation and its binary64 rounding unchanged.
    return _finite_loss(y64, p64)


def policy_descriptor():
    return {'version': VERSION, 'estimand': ESTIMAND,
        'user_choice_sha256': USER_CHOICE_SHA256,
        'prior_choice_sha256': PRIOR_CHOICE_SHA256,
        'failure_loss': LOSS_BOUND,
        'valid_domain': 'original_finite_nonnegative_binary64_y_and_p',
        'valid_loss': 'original_finite_market_oracle._finite_loss_unchanged',
        'weights': 'equal_symbol_within_origin_then_equal_origin',
        'rounding': 'exact_rational_mean_of_original_binary64_row_losses',
        'scope': 'arithmetic_only_no_execution_or_truth_certificate',
        'actual_failure': 'error_one_power_zero_full_original_N',
        'formal_acceptance': False}


@dataclass(frozen=True)
class NominalPairRow:
    origin: str
    symbol: str
    target: object
    reference: object
    candidate: object


@dataclass(frozen=True)
class TotalPathRisk:
    reference: Fraction
    candidate: Fraction
    nominal_rows: int
    missing_rows: int
    reference_failure_rows: int
    candidate_failure_rows: int
    origins: int

    def relative_gain(self):
        """Undefined zero-reference ratio stays undefined; no epsilon repair."""
        if self.reference == 0:
            return None
        return 1-self.candidate/self.reference


def _origin(value):
    if type(value) is not str:
        raise TypeError('Frozen UTC origin string required')
    try:
        origin = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError('Invalid frozen UTC origin') from error
    if origin.tzinfo is None or origin.utcoffset() != timedelta(0):
        raise ValueError('UTC origin required')
    return origin


def _failure(y, p):
    if type(y) not in (int, float) or type(p) not in (int, float):
        return True
    try:
        values = float(y), float(p)
    except OverflowError:
        return True
    return any(not math.isfinite(v) or v < 0 for v in values)


def exact_complete_support_risks(support, rows):
    """Exact arithmetic for ALL frozen symbol/origin slots, never a certificate.

    A row missing from the nominal algorithm output gives B to both roles;
    a missing role in an existing row gives B only to that role. Production
    callers must distinguish actual missing evidence from an authenticated
    nominal failure. Dropping support, constructing it from successful rows,
    or declaring IO/budget failures nominal is outside this function's scope.
    """
    if type(support) is not tuple or not support or type(rows) is not tuple:
        raise TypeError('Nonempty immutable original support and row tuple required')
    expected = set()
    counts = {}
    for key in support:
        if (type(key) is not tuple or len(key) != 2 or type(key[1]) is not str
                or not key[1]):
            raise ValueError('Duplicate or invalid frozen support slot')
        when = _origin(key[0])
        identity = (when, key[1])
        if identity in expected:
            raise ValueError('Duplicate UTC slot under a different timestamp spelling')
        expected.add(identity)
        counts[when] = counts.get(when, 0)+1
    observed = {}
    for row in rows:
        if type(row) is not NominalPairRow or type(row.symbol) is not str:
            raise TypeError('Original nominal pair row required')
        key = (_origin(row.origin), row.symbol)
        if key not in expected or key in observed:
            raise ValueError('Foreign or duplicate nominal row')
        observed[key] = row
    reference = candidate = Fraction()
    missing = ref_failed = cand_failed = 0
    for key in expected:
        weight = Fraction(1, len(counts)*counts[key[0]])
        row = observed.get(key)
        if row is None:
            a = b = float(LOSS_BOUND)
            missing += 1
            ref_failed += 1
            cand_failed += 1
        else:
            a = nominal_total_loss(row.target, row.reference)
            b = nominal_total_loss(row.target, row.candidate)
            ref_failed += int(_failure(row.target, row.reference))
            cand_failed += int(_failure(row.target, row.candidate))
        reference += weight*Fraction.from_float(a)
        candidate += weight*Fraction.from_float(b)
    return TotalPathRisk(reference, candidate, len(expected), missing,
                         ref_failed, cand_failed, len(counts))


def exact_population_risks(paths):
    """Uniform mean of a complete caller-authenticated WHOLE-path census.

    Path count and finite-law completeness must be authenticated by the
    production registry; a tuple here cannot prove either or provider IID.
    """
    if type(paths) is not tuple or not paths or any(type(p) is not TotalPathRisk for p in paths):
        raise TypeError('Complete immutable whole-path risk tuple required')
    bound = Fraction(LOSS_BOUND)
    if any(type(v) is not Fraction or not 0 <= v <= bound
           for p in paths for v in (p.reference, p.candidate)):
        raise ValueError('Bounded exact whole-path risks required')
    return tuple(sum((getattr(p, role) for p in paths), Fraction())/len(paths)
                 for role in ('reference', 'candidate'))
