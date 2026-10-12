"""Fixed library effect grid arithmetic, not authenticated cold-risk evidence.

All risk pairs must eventually be reconstructed from the same entire paid cold
census. These helpers do not run fits, choose candidates, classify four other
claims, grant entropy/verification quotas or authorize formal acceptance.
"""
from fractions import Fraction
from math import isfinite, lcm

TARGETS = (Fraction(0), Fraction(3, 1000), Fraction(1, 200),
           Fraction(1, 100), Fraction(1, 50))
POWER_TARGETS = (Fraction(1, 100), Fraction(1, 50))
SUPPORT = 128
GROUP_SIZE = 64
LOSS_BOUND = 1 << 20


def _risk(value):
    if type(value) is not float or not isfinite(value) or not 0 <= value <= LOSS_BOUND:
        raise ValueError('Original finite nonnegative binary64 path risk required')
    return Fraction.from_float(value)


def _input(risks):
    if type(risks) is not tuple or len(risks) != SUPPORT:
        raise ValueError('All fixed ordered 128 nominal risk pairs required')
    if any(type(pair) is not tuple or len(pair) != 2 for pair in risks):
        raise ValueError('Each atom requires its original two risks')
    return tuple((_risk(r), _risk(c)) for r, c in risks)


def _at(exact, target):
    groups = []
    for start in (0, GROUP_SIZE):
        rows = exact[start:start+GROUP_SIZE]
        r = sum((r for r, _ in rows), Fraction())/GROUP_SIZE
        c = sum((c for _, c in rows), Fraction())/GROUP_SIZE
        groups.append((1-target)*r-c)
    left, right = groups
    if left*right >= 0:
        return {'target': target, 'status': 'unavailable_fixed_groups_do_not_straddle',
                'formal_acceptance': False}
    mass = right/(right-left)
    weights = (mass/GROUP_SIZE,)*GROUP_SIZE+((1-mass)/GROUP_SIZE,)*GROUP_SIZE
    r = sum((p*r for p, (r, _) in zip(weights, exact)), Fraction())
    c = sum((p*c for p, (_, c) in zip(weights, exact)), Fraction())
    if (not 0 < mass < 1 or any(p <= 0 for p in weights)
            or sum(weights, Fraction()) != 1 or r <= 0
            or c != (1-target)*r):
        raise ArithmeticError('Exact full-support effect invariant failed')
    denominator = lcm(*(p.denominator for p in weights))
    return {'target': target, 'status': 'available_numeric_only',
            'weights': weights, 'reference_risk': r, 'candidate_risk': c,
            'gain': 1-c/r, 'group_mass': mass,
            'sampling_denominator': denominator,
            'minimum_block_bits': (denominator-1).bit_length(),
            'truth': 'null' if target <= Fraction(1, 200) else 'alternative',
            'power_target': target in POWER_TARGETS,
            'conditional_pure_sampler_law_only': True, 'formal_acceptance': False}


def derive_effect_grid(risks):
    """Compute exactly the preregistered 0/.3/.5/1/2% grid, without tuning.

    No caller target, groups, omissions, risk clipping or reordering. A missing
    target stays missing. No success implies other slots or simulated power.
    All five laws use the SAME risk census; no duplicate free fitting occurs.
    """
    exact = _input(risks)
    return tuple(_at(exact, target) for target in TARGETS)


def grid_payload(grid):
    """Exact JSON data only; cannot authenticate caller-created evidence."""
    if type(grid) is not tuple or len(grid) != len(TARGETS):
        raise ValueError('Entire fixed grid required')
    out = []
    for target, value in zip(TARGETS, grid):
        if value['target'] != target:
            raise ValueError('Original fixed target order required')
        row = {k: v for k, v in value.items() if k not in
               ('weights', 'reference_risk', 'candidate_risk', 'gain', 'group_mass', 'target')}
        row['target'] = [target.numerator, target.denominator]
        if value['status'] == 'available_numeric_only':
            row['weights'] = [[p.numerator, p.denominator] for p in value['weights']]
            for key in ('reference_risk', 'candidate_risk', 'gain', 'group_mass'):
                row[key] = [value[key].numerator, value[key].denominator]
        out.append(row)
    return out
