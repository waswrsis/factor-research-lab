"""Exact prospective two-group mixture; arithmetic, never cold-fit authority.

Population risks are the original binary64 full-path risk values interpreted
as rationals. The group assignment is fixed before census observation. The
public production adapter must obtain these values from all original paid
cold records; this pure function cannot certify those inputs or any claim.
"""
from fractions import Fraction
from math import isfinite, lcm

SUPPORT = 128
GROUP_SIZE = 64
LOSS_BOUND = 1 << 20
DELTA = Fraction(1, 200)


def _risk(value):
    if type(value) is not float or not isfinite(value) or not 0 <= value <= LOSS_BOUND:
        raise ValueError('Original finite nonnegative binary64 path risk required')
    return Fraction.from_float(value)


def derive_boundary(risks):
    """Return exact 128-atom weights, or refuse an unavailable strict mixture.

    No inputs are clipped, truncated, dropped, reordered, resampled or searched.
    In particular groups 0..63 and 64..127 remain fixed. This may legitimately
    fail; creating a boundary is not guaranteed by the existence of a learner.
    """
    if type(risks) is not tuple or len(risks) != SUPPORT:
        raise ValueError('The entire fixed ordered 128-atom census is required')
    if any(type(pair) is not tuple or len(pair) != 2 for pair in risks):
        raise ValueError('Each atom requires its original reference/candidate pair')
    exact = tuple((_risk(r), _risk(c)) for r, c in risks)
    groups = []
    for begin in (0, GROUP_SIZE):
        rows = exact[begin:begin+GROUP_SIZE]
        reference = sum((r for r, _ in rows), Fraction())/GROUP_SIZE
        candidate = sum((c for _, c in rows), Fraction())/GROUP_SIZE
        groups.append((reference, candidate, 199*reference-200*candidate))
    left, right = groups
    if left[2]*right[2] >= 0:
        raise ValueError('Fixed cold groups do not strictly straddle the boundary')
    mass = right[2]/(right[2]-left[2])
    weights = (mass/GROUP_SIZE,)*GROUP_SIZE+((1-mass)/GROUP_SIZE,)*GROUP_SIZE
    reference = sum((p*r for p, (r, _) in zip(weights, exact)), Fraction())
    candidate = sum((p*c for p, (_, c) in zip(weights, exact)), Fraction())
    if (not 0 < mass < 1 or any(p <= 0 for p in weights)
            or sum(weights, Fraction()) != 1 or reference <= 0
            or 199*reference != 200*candidate):
        raise ArithmeticError('Exact full-law boundary invariant failed')
    denominator = lcm(*(p.denominator for p in weights))
    return {'weights': weights, 'reference_risk': reference,
            'candidate_risk': candidate, 'gain': 1-candidate/reference,
            'group_mass': mass, 'sampling_denominator': denominator,
            'minimum_block_bits': (denominator-1).bit_length(),
            'conditional_success_law_only': True, 'formal_acceptance': False}


def exact_payload(value):
    """Exact JSON representation, without approximate float equality claims."""
    return {'weights': [[v.numerator, v.denominator] for v in value['weights']],
        **{key: [value[key].numerator, value[key].denominator]
           for key in ('reference_risk', 'candidate_risk', 'gain', 'group_mass')},
        'sampling_denominator': value['sampling_denominator'],
        'minimum_block_bits': value['minimum_block_bits'],
        'conditional_success_law_only': True, 'formal_acceptance': False}
