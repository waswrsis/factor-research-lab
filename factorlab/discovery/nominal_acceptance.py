"""Exact nominal truth and conservative full-N diagnostic accounting.

This new estimand never changes the old joint-service oracle. Arithmetic
helpers authenticate nothing. The registry reader reconstructs real original
artifacts, but is NOT a metered five-claim acceptance producer.
"""
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from factorlab.protocol import project_path
from .calibration_acceptance import AcceptancePlan, CLAIMS, exact_binomial_bounds
from .evidence import file_hash
from .monthly_work_authority import source_files

VERSION = 'v7-exact-nominal-acceptance-diagnostic-v1'
RELATIVE = 'factorlab/discovery/nominal_acceptance.py'
DELTA = Fraction(1, 200)
PENDING = CLAIMS[1:]


@dataclass(frozen=True)
class ExactNominalRisk:
    """Unprivileged exact risk arithmetic, not an oracle certificate."""
    reference: Fraction
    candidate: Fraction

    def __post_init__(self):
        if (type(self.reference) is not Fraction or type(self.candidate) is not Fraction
                or self.reference <= 0 or self.candidate < 0):
            raise ValueError('Positive exact reference and nonnegative exact candidate required')

    @property
    def gain(self):
        return 1-self.candidate/self.reference

    def truth(self):
        # No binary64 quotient, endpoint widening or tolerance near the null.
        return 'null' if self.gain <= DELTA else 'alternative'


def library_counts(risk, *, complete, guarded_event):
    """Worst-case accounting; booleans supplied here have no authenticity."""
    if type(risk) is not ExactNominalRisk or type(complete) is not bool:
        raise TypeError('Exact risks and strict completion boolean required')
    if not complete:
        if guarded_event is not None:
            raise ValueError('An incomplete trial cannot supply a discovery event')
        return {'denominator': 1, 'possible_error': 1, 'power_success': 0}
    if type(guarded_event) is not bool:
        raise TypeError('Strict original conformance-guarded event required')
    return {'denominator': 1,
            'possible_error': int(guarded_event and risk.truth() == 'null'),
            'power_success': int(guarded_event and risk.truth() == 'alternative')}


def _installed():
    from . import nominal_actual_trial, nominal_cold_law, nominal_trial_controller
    files = source_files()
    modules = [(nominal_actual_trial, 'nominal_actual_trial'),
               (nominal_cold_law, 'nominal_cold_law'),
               (nominal_trial_controller, 'nominal_trial_controller')]
    modules.append((__import__(__name__, fromlist=['*']), 'nominal_acceptance'))
    for module, name in modules:
        relative = 'factorlab/discovery/'+name+'.py'
        if (Path(module.__file__).resolve() != project_path(relative).resolve()
                or files.get(relative) != file_hash(module.__file__)):
            raise ValueError('Original new canonical nominal source required')
    return files


def _risk_from_original_law(law, value):
    """Called only after original ColdBoundaryLaw.verify in the public reader."""
    if (value != law.value or value['law']['gain'] != [1, 200]
            or law.intent['claim_slot'] != 'library_skill'
            or tuple(law.intent['pending_slots']) != PENDING):
        raise ValueError('Unsupported nominal truth scope')
    def rational(pair):
        if (type(pair) is not list or len(pair) != 2
                or any(type(x) is not int for x in pair) or pair[1] <= 0):
            raise ValueError('Original exact risk representation required')
        value = Fraction(*pair)
        if [value.numerator, value.denominator] != pair:
            raise ValueError('Original canonical reduced fraction required')
        return value
    risk = ExactNominalRisk(rational(value['law']['reference_risk']),
                            rational(value['law']['candidate_risk']))
    if risk.gain != DELTA or risk.truth() != 'null':
        raise ValueError('Original entire-law exact boundary invariant differs')
    return risk


def read_nominal_diagnostic(registry):
    """Read original full N; never accepts caller observations or weights.

    Two complete original paid-law verifications bound the read. Each actual
    coordinate is reconstructed by the original controller's read-only guard.
    Verification has no registered charged proof lease in this version. The
    report explicitly remains an unmetered diagnostic and cannot authorize a
    formal gate even if some numeric library bounds pass.
    """
    from .nominal_actual_trial import NominalActualRegistry, _acceptance
    from .nominal_cold_law import ColdBoundaryLaw
    from .nominal_trial_controller import registered_outcome
    if type(registry) is not NominalActualRegistry:
        raise TypeError('Original typed full-N nominal registry required')
    files = _installed()
    identity = registry.verify()
    if identity['source_files'] != files:
        raise ValueError('Original registry belongs to another source')
    plan = _acceptance(identity['acceptance'])
    if type(plan) is not AcceptancePlan:
        raise TypeError('Original complete frozen acceptance plan required')
    law = ColdBoundaryLaw(identity['law']['directory'])
    original = law.verify()
    if (law.digest != identity['law']['digest'] or original != identity['law']['value']
            or law.intent != identity['law']['intent']):
        raise ValueError('Original registered complete paid law differs')
    risk = _risk_from_original_law(law, original)
    cells = []
    for cell in plan.cells:
        complete = errors = power = unsupported = 0
        for index in range(cell.repetitions):
            outcome = registered_outcome(registry, cell.name, index)
            if (outcome['coordinate'] != registry.coordinate(cell.name, index)
                    or outcome['retain_N'] is not True
                    or outcome['formal_acceptance'] is not False
                    or tuple(outcome['pending_slots']) != PENDING):
                raise ValueError('Original outcome coordinate or scope differs')
            supported = cell.scenario_digest == law.digest
            unsupported += int(not supported)
            done = supported and outcome['complete_library'] is True
            counts = library_counts(risk, complete=done,
                guarded_event=outcome['guarded_numerical_event'] if done else None)
            complete += int(done)
            errors += counts['possible_error']
            power += counts['power_success']
        _, upper = exact_binomial_bounds(errors, cell.repetitions, tail_alpha=plan.tail_alpha)
        # All four unimplemented slots remain possible errors, at every N.
        # Library success cannot discharge any other selected/learned hypothesis.
        cells.append({'cell': cell.name, 'scenario_digest': cell.scenario_digest,
            'registered_repetitions': cell.repetitions, 'complete_library': complete,
            'failed_or_missing_library': cell.repetitions-complete,
            'unsupported_scenario_trials': unsupported,
            'library_possible_errors': errors, 'library_guarded_event_error_upper': upper,
            'library_unconditional_power_successes': power,
            'possible_family_errors': cell.repetitions, 'family_error_upper': 1.,
            'discovery_power': {slot: {'discoveries': 0, 'denominator': cell.repetitions,
                'lower': 0., 'target': plan.discovery_power_target, 'target_met': False}
                for slot in cell.power_slots}})
    if (law.verify() != original or registry.verify() != identity or _installed() != files):
        raise ValueError('Original source/law/registry changed during diagnostic read')
    return {'version': VERSION, 'estimand': 'nominal_full_cold_rolling',
        'plan_digest': plan.digest, 'registry_digest': registry.digest, 'law_digest': law.digest,
        'plan': plan.to_dict(), 'cells': cells, 'exact_library_gain': [1, 200],
        'library_truth': 'null', 'pending_slots': list(PENDING),
        'numeric_targets_met': False, 'all_five_claims_authenticated': False,
        'formal_gate_eligible': False, 'formal_acceptance': False, 'full_V7': False,
        'verification_cost_registered': False, 'diagnostic_only': True,
        'confidence': {'monte_carlo_alpha': plan.monte_carlo_alpha,
            'oracle_alpha': plan.oracle_alpha, 'per_bound_tail_alpha': plan.tail_alpha,
            'meaning': 'nominal independent draw events; failures conservatively dominate; '
                       'does not estimate future nonstationary service reliability'},
        'pending': ['other_four_nominal_truth_producers', 'positive_effect_cells',
                    'registered_proof_verification_budget', 'full_real_pipeline_acceptance']}
