"""Prospective effect scenario IDs, fixed before cold risks are observed.

An ID specifies a design, not an available law, oracle proof or execution grant.
The original census digest binds generator, calendar and complete source. Risk
values, fitted models and derived mixture weights must not enter these IDs.
"""
from factorlab.protocol import fingerprint
from .calibration_acceptance import AcceptancePlan, AcceptanceCell

VERSION = 'nominal-library-effect-scenario-v1'
DESIGNS = (('zero', (0, 1), False), ('subthreshold', (3, 1000), False),
           ('boundary', (1, 200), False), ('power1', (1, 100), True),
           ('power2', (1, 50), True))


def _census_digest(value):
    if (type(value) is not str or len(value) != 64
            or any(c not in '0123456789abcdef' for c in value)):
        raise ValueError('Canonical original census digest required')
    return value


def scenario_design(census_digest, name):
    census_digest = _census_digest(census_digest)
    if type(name) is not str or name not in {d[0] for d in DESIGNS}:
        raise ValueError('One of the fixed preregistered effect names required')
    _, target, power = next(d for d in DESIGNS if d[0] == name)
    return {'version': VERSION, 'census_digest': census_digest, 'target': list(target),
        'support': 128, 'groups': [[0, 64], [64, 128]],
        'estimand': 'nominal_full_cold_rolling', 'claim_slot': 'library_skill',
        'cell_name': 'library_'+name, 'power_slot': 'library_skill' if power else None}


def scenario_design_id(census_digest, name):
    return fingerprint(scenario_design(census_digest, name))


def library_acceptance_plan(census_digest, *, repetitions):
    """Unprivileged plan data; requires a separate typed paid registration.

    Same N for all five fixed designs. The original AcceptancePlan validates N
    and unchanged confidence/threshold targets. No law-result-dependent cell
    omissions are possible through this constructor.
    """
    return AcceptancePlan(tuple(AcceptanceCell('library_'+name,
        scenario_design_id(census_digest, name), repetitions,
        ('library_skill',) if power else ()) for name, _, power in DESIGNS))


def require_full_library_grid(plan, census_digest):
    """Verify five preregistered designs while preserving extra frozen cells."""
    if type(plan) is not AcceptancePlan:
        raise TypeError('Original full acceptance plan required')
    expected = {scenario_design_id(census_digest, name): ('library_'+name, power)
                for name, _, power in DESIGNS}
    found = set()
    for cell in plan.cells:
        if cell.scenario_digest in expected:
            name, power = expected[cell.scenario_digest]
            if (cell.scenario_digest in found or cell.name != name
                    or cell.power_slots != (('library_skill',) if power else ())):
                raise ValueError('Original effect cell name or power role changed')
            found.add(cell.scenario_digest)
        elif cell.name.startswith('library_'):
            raise ValueError('A library cell belongs to another prospective design')
    if found != set(expected):
        raise ValueError('All five prospective effect designs must remain in the grid')
    return plan
