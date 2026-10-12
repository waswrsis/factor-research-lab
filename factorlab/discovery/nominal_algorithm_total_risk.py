"""Prospective complete-policy design and conditional whole-path arithmetic.

No IID certificate is constructed here. Execution delegates to the fixed
native worker with its original permit, clocks and wallets. Registered risk
readback keeps the unconditional [0,B] range for the opaque provider law.
Registration is an immutable design, not a wallet.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt
from pathlib import Path

VERSION = 'nominal-complete-algorithm-total-risk-v4-registered-readback'
ESTIMAND = 'nominal_full_cold_rolling_total_loss_v2'
CHOICE = '035fb78387f06970c4e4249c3728c2cf865a6857ebfce5fe7cf77be3a0d2b1f0'
RELATIVE = 'factorlab/discovery/nominal_algorithm_total_risk.py'
B = 2**20
KERNEL_FILES = (
    'full_trial_controller.py', 'full_calibration_registry.py',
    'v7_autonomous.py', 'v7_service.py', 'v7_run.py', 'scheduled_search.py',
    'prompt_builder.py', 'full_mechanism_contract.py',
    'algorithm_shortlist.py', 'algorithm_validation.py',
    'algorithm_validation_producer.py', 'v7_forward.py', 'v7_lifecycle.py',
    'monthly_work_authority.py', 'monthly_partition_policy.py',
    'nominal_total_loss.py', 'nominal_algorithm_execution.py',
    'nominal_algorithm_producer.py', 'nominal_full_scope_proof.py',
    'nominal_algorithm_risk_reader.py',
)


@dataclass(frozen=True)
class ConditionalPathBounds:
    reference: tuple[Fraction, Fraction]
    candidate: tuple[Fraction, Fraction]
    delta_gap: tuple[Fraction, Fraction]
    radius: Fraction
    paths: int
    unavailable_paths: int
    tail_alpha: Fraction


def _alpha(value):
    if type(value) is not Fraction or not 0 < value < Fraction(1, 2):
        raise ValueError('Strict exact positive tail probability required')
    return value


def oracle_tail_alpha(repetitions):
    """Original .025 oracle allocation, 4 directions x 5 slots x whole N.

    This is pure arithmetic, not permission to spend the original alpha twice.
    Production registration derives the counts from its exact AcceptancePlan.
    """
    if (type(repetitions) is not tuple or not 1 <= len(repetitions) <= 64
            or any(type(n) is not int or not 1000 <= n <= 2000 for n in repetitions)):
        raise ValueError('Original full acceptance repetition counts required')
    return Fraction(1, 40*4*5*sum(repetitions))


def conditional_whole_path_bounds(paths, *, tail_alpha):
    """Hoeffding enclosure CONDITIONAL on independent complete-policy paths.

    Each tuple contains two exact whole-path means in [0,B]; None means absent
    evidence/infra failure, never an observed nominal worst-loss computation.
    All original N paths remain. No independence within a path is assumed.
    This public numeric helper intentionally has no authenticity/truth flag.
    """
    a = _alpha(tail_alpha)
    if type(paths) is not tuple or not 1 <= len(paths) <= 2_000_000:
        raise ValueError('Bounded complete ordered path tuple required')
    sums = [Fraction(), Fraction()]
    missing = 0
    for path in paths:
        if path is None:
            missing += 1
            continue
        if (type(path) is not tuple or len(path) != 2
                or any(type(x) is not Fraction or not 0 <= x <= B for x in path)):
            raise ValueError('Exact bounded whole-path pair or None required')
        sums[0] += path[0]
        sums[1] += path[1]
    # Integer k satisfies exp(-k) <= 2**(-k) <= a. This avoids claiming
    # rounded libm log/sqrt values are conservative mathematical bounds.
    k = 0
    while a.numerator*(1 << k) < a.denominator:
        k += 1
    scale = 1 << 64
    square = Fraction(k*scale*scale, 2*len(paths))
    root = isqrt(square.numerator // square.denominator)
    if root*root*square.denominator < square.numerator:
        root += 1
    radius = Fraction(B*root, scale)
    intervals = tuple((max(Fraction(), total/len(paths)-radius),
                       min(Fraction(B), (total+missing*B)/len(paths)+radius))
                      for total in sums)
    r, c = intervals
    # Exact .005 boundary: gap=199R-200C; no ratio at R=0 and no float equality.
    gap = (199*r[0]-200*c[1], 199*r[1]-200*c[0])
    return ConditionalPathBounds(r, c, gap, radius, len(paths), missing, a)


def _installed():
    from .monthly_work_authority import source_files
    from .evidence import file_hash
    from factorlab.protocol import project_path
    from . import nominal_total_loss as loss
    files = source_files()
    loss_relative = 'factorlab/discovery/nominal_total_loss.py'
    if (Path(__file__).resolve() != project_path(RELATIVE).resolve()
            or files.get(RELATIVE) != file_hash(__file__)
            or Path(loss.__file__).resolve() != project_path(loss_relative).resolve()
            or files.get(loss_relative) != file_hash(loss.__file__)
            or loss.ESTIMAND != ESTIMAND or loss.USER_CHOICE_SHA256 != CHOICE
            or loss.LOSS_BOUND != B):
        raise ValueError('New canonical source and exact total-loss v2 required')
    return files


def _kernel(files):
    keys = ['factorlab/discovery/'+name for name in KERNEL_FILES]
    if any(key not in files for key in keys):
        raise ValueError('Complete original adaptive kernel dependencies missing')
    return {key: files[key] for key in keys}


def _fresh_parent(parent):
    # A pre-existing original plan can supply configuration, never old results
    # or permit rights. Any trial entry, including a partial failure, is exposed.
    if (parent.directory/'trials').exists():
        raise ValueError('Old enrolled or failed full runs cannot be reinterpreted')
    for name in ('oracle_estimands', 'controller_plans'):
        if (parent.directory/name).exists():
            raise ValueError('Original full execution already began')


class AlgorithmTotalRiskRegistry:
    """One new design per unused original full-plan namespace; no new credit.

    The source-bound parent remains a configuration ancestry only. This type
    deliberately is not QuarantinedTrialPermit or NominalActualRegistry.
    """
    @classmethod
    def freeze(cls, parent, *, schedule):
        from .full_calibration_registry import FullTrialRegistry
        from .full_trial_controller import FullControllerSchedule
        from .nominal_cold_conformance import _write
        from filelock import FileLock
        from factorlab.protocol import fingerprint
        if type(parent) is not FullTrialRegistry or type(schedule) is not FullControllerSchedule:
            raise TypeError('Original complete plan and fixed adaptive schedule required')
        files = _installed()
        parent.verify()
        _fresh_parent(parent)
        if parent.plan.runtime_scope['source_files'] != files:
            raise ValueError('Original plan belongs to another source')
        directory = parent.directory/'nominal_algorithm_total_v2'
        # Exclusive directory itself consumes a partial administrative freeze;
        # no exception cleanup or idempotent overwrite can create a second try.
        with FileLock(str(parent.directory/'.nominal-algorithm-design.lock'), timeout=0):
            _fresh_parent(parent)
            directory.mkdir(exist_ok=False)
            value = cls._expected(parent, schedule, files)
            _write(directory/'identity.json', value)
            _write(parent.directory/'nominal_algorithm_total_v2_anchor.json', value)
        result = cls.__new__(cls)
        result.parent, result.directory, result.identity = parent, directory, value
        result.digest = fingerprint(value)
        result.verify()
        return result

    @staticmethod
    def _expected(parent, schedule, files):
        from .nominal_total_loss import policy_descriptor
        reps = tuple(c.repetitions for c in parent.plan.acceptance.cells)
        a = oracle_tail_alpha(reps)
        return {'version': VERSION, 'estimand': ESTIMAND, 'choice': CHOICE,
            'parent_directory': str(parent.directory), 'parent_digest': parent.digest,
            'source_files': files, 'kernel_files': _kernel(files),
            'full_plan': parent.plan.to_dict(), 'full_schedule': schedule.to_dict(),
            'loss': policy_descriptor(), 'oracle_tail_alpha': [a.numerator, a.denominator],
            'whole_path_count': parent.plan.oracle_paths,
            'whole_validation_N': sum(reps), 'claim': 'algorithm',
            'required_kernel': 'complete_cold_adaptive_search_selection_refit_adoption_forecast_settlement',
            'provider_scope': 'entire_history_conditioned_provider_kernel_not_final_winner',
            'provider_independence': 'unestablished_no_certificate_from_fresh_process_or_trace_hash',
            'physical_work_budget': parent.plan.runtime_scope['work_budget'],
            'work_clock': 'original_physical_UTC_and_original_nonrenewable_leases',
            'actual_infrastructure_failure': {'possible_error': 1, 'power': 0},
            'execution_issuer': 'fixed_full_scope_native_worker_original_controller_and_wallets',
            'risk_readback': 'original_validation_proof300_remaining_unconditional_0_B',
            'registration_scope': 'prospective_configuration_only_not_execution_or_alpha_credit',
            'formal_acceptance': False}

    @classmethod
    def inspect(cls, parent):
        from .full_calibration_registry import FullTrialRegistry
        from .nominal_cold_conformance import _read
        from factorlab.protocol import fingerprint
        if type(parent) is not FullTrialRegistry:
            raise TypeError('Exact original full-plan parent required')
        obj = cls.__new__(cls)
        obj.parent = parent
        obj.directory = parent.directory/'nominal_algorithm_total_v2'
        obj.identity = _read(obj.directory/'identity.json')
        obj.digest = fingerprint(obj.identity)
        obj.verify()
        return obj

    def verify(self):
        from .full_trial_controller import FullControllerSchedule
        from .nominal_cold_conformance import _read
        from factorlab.protocol import fingerprint
        from .calibration_trial_context import _no_redirect
        files = _installed()
        self.parent.verify()
        _no_redirect(self.directory, self.parent.directory)
        expected_directory = self.parent.directory/'nominal_algorithm_total_v2'
        schedule = FullControllerSchedule.from_dict(self.identity['full_schedule'])
        expected = self._expected(self.parent, schedule, files)
        if (self.directory != expected_directory or self.identity != expected
                or self.digest != fingerprint(expected)
                or _read(self.directory/'identity.json') != expected
                or _read(self.parent.directory/'nominal_algorithm_total_v2_anchor.json') != expected):
            raise ValueError('New algorithm registration source/anchors/whole scope changed')
        return expected

    def execution_request(self, cell, index, *, oracle_index):
        """Return fixed design coordinates, NOT a permit or claimed paid work."""
        from factorlab.protocol import fingerprint
        p = self.verify()
        coordinate = self.parent.identity_for(cell, index, oracle_index=oracle_index)
        if type(oracle_index) is not int:
            raise ValueError('A whole oracle path coordinate is required')
        return {'version': VERSION, 'design_digest': self.digest,
            'coordinate': coordinate, 'request_digest': fingerprint([self.digest, coordinate]),
            'estimand': ESTIMAND, 'kernel_files': p['kernel_files'],
            'original_budget': p['physical_work_budget'],
            'execution_authorized': False, 'formal_acceptance': False,
            'route': 'source_fixed_original_full_scope_worker_not_a_launch_permit',
            'missing': ['independent_complete_provider_and_market_kernel_evidence']}

    def produce_path(self, cell, index, *, oracle_index):
        self.execution_request(cell, index, oracle_index=oracle_index)
        from .nominal_full_scope_proof import run_coordinate
        return run_coordinate(self.parent.directory, cell, index, oracle_index)

    def certified_risk_interval(self, cell, index):
        """No caller rows or IID switch can pass as production whole-path truth."""
        self.verify()
        self.parent.identity_for(cell, index)
        from .nominal_algorithm_risk_reader import registered_risk_interval
        return registered_risk_interval(self, cell, index)
