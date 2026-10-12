"""Metered full-N library diagnostics; four unknown claims remain worst case.

Canonical task handlers read original records only. Numeric helpers authenticate
nothing. Native history support is a required separate source-bound producer.
"""
from fractions import Fraction
from pathlib import Path
import importlib

from factorlab.protocol import fingerprint, project_path
from .evidence import file_hash
from .monthly_work_authority import source_files
from .calibration_acceptance import CLAIMS, AcceptancePlan, exact_binomial_bounds
from .nominal_acceptance import ExactNominalRisk, library_counts

VERSION = 'nominal-metered-full-N-library-report-v1'
RELATIVE = 'factorlab/discovery/nominal_full_N_report.py'
PENDING = list(CLAIMS[1:])
KINDS = ('actual_coordinate_readback', 'acceptance_cell_aggregation', 'final_full_N_report')


def _fraction(value):
    if (type(value) is not list or len(value) != 2
            or any(type(x) is not int for x in value) or value[1] <= 0):
        raise ValueError('Original exact reduced rational required')
    f = Fraction(*value)
    if [f.numerator, f.denominator] != value:
        raise ValueError('Noncanonical exact rational')
    return f


def _risk(row):
    """Private numeric consistency only; caller must verify original paid grid."""
    from .nominal_effect_grid import TARGETS
    if (type(row) is not dict or row.get('status') != 'available_numeric_only'
            or row.get('formal_acceptance') is not False):
        raise ValueError('Unavailable grid cell retains all N')
    target = _fraction(row['target'])
    r = ExactNominalRisk(_fraction(row['reference_risk']), _fraction(row['candidate_risk']))
    if target not in TARGETS or r.gain != target or _fraction(row['gain']) != target:
        raise ValueError('Original fixed effect and exact risk relation differ')
    return r


def _risk_payload(risk):
    return None if risk is None else {k: [v.numerator, v.denominator] for k, v in
        (('reference', risk.reference), ('candidate', risk.candidate), ('gain', risk.gain))}


def _numeric_coordinate(coordinate, risk, outcome=None, *, reason=None):
    """Unprivileged arithmetic fixture boundary, never a public evidence input."""
    complete = False
    event = None
    if risk is not None and outcome is not None and reason is None:
        if (outcome.get('coordinate') != coordinate or outcome.get('retain_N') is not True
                or outcome.get('formal_acceptance') is not False or outcome.get('pending_slots') != PENDING
                or outcome.get('complete_library') is not True
                or type(outcome.get('guarded_numerical_event')) is not bool):
            reason = 'original_outcome_missing_failed_or_scope_mismatch'
        else:
            complete = True
            event = outcome['guarded_numerical_event']
    if complete:
        counts = library_counts(risk, complete=True, guarded_event=event)
    else:
        counts = {'denominator': 1, 'possible_error': 1, 'power_success': 0}
    return {'version': VERSION, 'coordinate': coordinate, 'retain_N': True,
        'complete_library': complete, 'guarded_numerical_event': event,
        'exact_library_risk': _risk_payload(risk), 'library_truth': risk.truth() if risk else 'unknown',
        'library_counts': counts, 'family_counts': {'denominator': 1, 'possible_error': 1},
        'pending_slots': PENDING, 'formal_acceptance': False,
        'failure': None if complete else (reason or 'missing_original_evidence')[:1000],
        'outcome_hash': fingerprint(outcome) if complete else None}


def _cell_numbers(plan, cell, rows):
    """All original repetitions are mandatory, including absent readback records."""
    if type(plan) is not AcceptancePlan or cell not in plan.cells or len(rows) != cell.repetitions:
        raise ValueError('Exact whole registered cell denominator required')
    errors = power = complete = 0
    for row in rows:
        if row is None:
            errors += 1
            continue
        counts = row['library_counts']
        if (counts['denominator'] != 1 or type(counts['denominator']) is not int
                or type(counts['possible_error']) is not int or counts['possible_error'] not in (0, 1)
                or type(counts['power_success']) is not int or counts['power_success'] not in (0, 1)
                or type(row['complete_library']) is not bool):
            raise ValueError('One fixed coordinate must contribute exactly one denominator')
        errors += counts['possible_error']; power += counts['power_success']; complete += row['complete_library']
    _, upper = exact_binomial_bounds(errors, cell.repetitions, tail_alpha=plan.tail_alpha)
    lower, _ = exact_binomial_bounds(power, cell.repetitions, tail_alpha=plan.tail_alpha)
    return {'cell': cell.name, 'scenario_digest': cell.scenario_digest,
        'registered_repetitions': cell.repetitions, 'complete_library': complete,
        'failed_or_missing_library': cell.repetitions-complete,
        'library_possible_errors': errors, 'library_error_upper': upper,
        'library_error_target_met': upper <= plan.family_error_target,
        'library_unconditional_power_successes': power,
        'library_power_lower': lower if 'library_skill' in cell.power_slots else None,
        'library_power_target_met': lower >= plan.discovery_power_target if 'library_skill' in cell.power_slots else None,
        'possible_family_errors': cell.repetitions, 'family_error_upper': 1.,
        'claims': {slot: {'status': 'library_only' if slot == 'library_skill' else 'unknown',
            'denominator': cell.repetitions,
            'possible_errors': errors if slot == 'library_skill' else cell.repetitions,
            'power_successes': power if slot == 'library_skill' else 0,
            'power_lower': (lower if slot == 'library_skill' else 0.) if slot in cell.power_slots else None}
            for slot in CLAIMS}, 'pending_slots': PENDING, 'formal_acceptance': False}


def _installed():
    names = ('nominal_full_N_report', 'nominal_acceptance', 'nominal_actual_trial',
             'nominal_trial_controller', 'nominal_cold_effect_grid', 'nominal_verification_budget',
             'nominal_effect_grid', 'nominal_effect_design')
    files = source_files()
    for name in names:
        module = importlib.import_module('factorlab.discovery.'+name)
        relative = 'factorlab/discovery/'+name+'.py'
        if (Path(module.__file__).resolve() != project_path(relative).resolve()
                or files.get(relative) != file_hash(module.__file__)):
            raise ValueError('Original canonical full-N consumer source required')
    return files


class NominalFullNReport:
    def __init__(self, actual_registry, grid):
        from .nominal_actual_trial import NominalActualRegistry
        from .nominal_cold_effect_grid import ColdEffectGrid
        if type(actual_registry) is not NominalActualRegistry or type(grid) is not ColdEffectGrid:
            raise TypeError('Original actual registry and cold grid required')
        # Construction performs no heavy evidence read. The already acquired
        # original task pays all validation inside execute_allocated.
        self.actual = actual_registry
        self.grid = grid

    def _native(self):
        """Required future canonical implementation; absence never means success.

        There is intentionally no stub producer or callback. This interface must
        authenticate natural owned-subtree closure and actual immutable records.
        Its future implementation is separately reviewed and source-integrated.
        """
        module = importlib.import_module('factorlab.discovery.nominal_native_completion')
        rel = 'factorlab/discovery/nominal_native_completion.py'
        files = source_files()
        if (Path(module.__file__).resolve() != project_path(rel).resolve()
                or files.get(rel) != file_hash(module.__file__)
                or files != self.grid.registry.plan.binding['source_files']):
            raise ValueError('Canonical native completion producer missing or foreign')
        reader = module.NativeCompletionReader(self.grid.registry)
        if type(reader) is not module.NativeCompletionReader:
            raise TypeError('Original typed native history reader required')
        return reader

    def _native_task(self, index, target, payload):
        reader = self._native()
        receipt = reader.verify_task(index, target=str(target), semantic_hash=fingerprint(payload))
        expected = {'verification_digest': self.grid.registry.digest, 'task_index': index,
            'target': str(target), 'semantic_hash': fingerprint(payload), 'native_completed': True}
        if type(receipt) is not dict or receipt != expected:
            raise ValueError('Native original task completion receipt differs')

    def _proof(self, index, service=None):
        manifest = self.grid.verified_manifest(index, service=service)
        self._native_task(index, self.grid.directory/f'verified-{index:06d}.json', manifest['record'])
        return manifest

    def _coordinate(self, cell, index):
        ordinal = next(i for i, c in enumerate(self.grid.registry.plan.acceptance.cells) if c.name == cell.name)
        return {'registry': self.actual.digest, 'cell': cell.name, 'cell_ordinal': ordinal,
            'index': index, 'scenario_digest': cell.scenario_digest}

    def _binding(self):
        from .nominal_actual_trial import _read
        identity = _read(self.actual.directory/'identity.json')
        if (identity != self.actual.identity or fingerprint(identity) != self.actual.digest
                or identity != _read(self.grid.directory/'actual_registry.json')
                or identity != _read(self.grid.registry.directory/'actual_registry.json')
                or identity['directory'] != str(self.actual.directory)
                or identity['grid']['directory'] != str(self.grid.directory)
                or identity['grid']['verification_registry_digest'] != self.grid.registry.digest
                or identity['census']['digest'] != self.grid.census.digest
                or identity['acceptance_digest'] != self.grid.registry.plan.acceptance.digest
                or identity['registered_N'] != self.grid.registry.plan.total_N
                or identity['source_files'] != self.grid.registry.plan.binding['source_files']):
            raise ValueError('Original full-N/grid/census/source anchor binding differs')

    def _risk(self, cell):
        from .nominal_actual_trial import _scenario_row
        from .nominal_effect_design import DESIGNS
        rows = [_scenario_row(self.grid, *design) for design in DESIGNS]
        matches = [r for r in rows if r['design_id'] == cell.scenario_digest
                   and r['numeric']['cell_name'] == cell.name]
        if len(matches) != 1:
            raise ValueError('Unsupported original scenario remains in N')
        return _risk(matches[0]['numeric'])

    def _original_outcome(self, cell, index, coordinate):
        from .nominal_trial_controller import registered_outcome
        identity = self.actual.verify()
        if (identity['grid']['directory'] != str(self.grid.directory)
                or identity['grid']['verification_registry_digest'] != self.grid.registry.digest
                or identity['acceptance_digest'] != self.grid.registry.plan.acceptance.digest
                or identity['registered_N'] != self.grid.registry.plan.total_N
                or identity['source_files'] != self.grid.identity['source_files']
                or self.actual.coordinate(cell.name, index) != coordinate):
            raise ValueError('Original complete actual registry changed')
        outcome = registered_outcome(self.actual, cell.name, index)
        if outcome.get('complete_library') is True:
            receipt = self._native().verify_trial(self.actual, cell.name, index,
                                                  outcome_hash=fingerprint(outcome))
            if receipt != {'coordinate': coordinate, 'outcome_hash': fingerprint(outcome), 'native_completed': True}:
                raise ValueError('Original complete actual trial native history missing')
        return outcome

    def _readback(self, task):
        plan = self.grid.registry.plan.acceptance
        unit = task.task['unit']
        cell = next(c for c in plan.cells if c.name == unit['cell'])
        coordinate = self._coordinate(cell, unit['index'])
        risk = None
        try:
            outcome = self._original_outcome(cell, unit['index'], coordinate)
            self._proof(130)
            risk = self._risk(cell)
            payload = _numeric_coordinate(coordinate, risk, outcome)
        except Exception as error:
            payload = _numeric_coordinate(coordinate, risk, reason=type(error).__name__+':'+str(error))
        task.check()  # Expired resource tasks cannot publish even conservative data.
        return payload

    def _read_coordinate(self, cell, index, task_index, risk):
        path = self.grid.directory/'full-N'/f'coordinate-{task_index:06d}.json'
        try:
            value = self.grid._semantic(task_index, path)
            self._native_task(task_index, path, value)
            coordinate = self._coordinate(cell, index)
            event = value['guarded_numerical_event']
            outcome = {'coordinate': coordinate, 'retain_N': True, 'formal_acceptance': False,
                'pending_slots': PENDING, 'complete_library': True, 'guarded_numerical_event': event}
            expected = _numeric_coordinate(coordinate, risk, outcome if value['complete_library'] else None,
                                          reason=value['failure'])
            # Hash binds original complete outcome; it is not reconstructed from
            # this shorter arithmetic view. Original producer validated that view.
            expected['outcome_hash'] = value['outcome_hash']
            if value != expected:
                raise ValueError('Original registered coordinate numeric fields changed')
            if value['complete_library'] and (type(value['outcome_hash']) is not str or len(value['outcome_hash']) != 64):
                raise ValueError('Complete coordinate lacks original outcome digest')
            return value
        except Exception:
            return None  # Always contributes denominator1,error1,power0.

    def _aggregate(self, task, cell):
        plan = self.grid.registry.plan.acceptance
        try:
            self._proof(130)
            risk = self._risk(cell)
        except Exception:
            risk = None
        base = 131+sum(c.repetitions for c in plan.cells[:list(plan.cells).index(cell)])
        rows = []
        with self.grid.registry.witness_read_scope():
            for index in range(cell.repetitions):
                task.check()
                rows.append(self._read_coordinate(cell, index, base+index, risk))
        return _cell_numbers(plan, cell, rows)

    def _final(self, task):
        plan = self.grid.registry.plan.acceptance
        # These are original paid manifests plus native completion evidence;
        # never invoke free grid.verify or a caller's truth/completion boolean.
        proof_indices = (130, 131+self.grid.registry.plan.total_N)
        before = self.grid.verified_manifest(proof_indices[0])
        after = self.grid.verified_manifest(proof_indices[1], service=task.service)
        proof_native = True
        proof_failure = None
        try:
            for index, manifest in zip(proof_indices, (before, after)):
                self._native_task(index, self.grid.directory/f'verified-{index:06d}.json', manifest['record'])
        except Exception as error:
            proof_native = False
            proof_failure = type(error).__name__+':'+str(error)[:1000]
        cells = []
        start = 132+self.grid.registry.plan.total_N
        for ordinal, cell in enumerate(plan.cells):
            task.check()
            path = self.grid.directory/'full-N'/f'cell-{ordinal:03d}.json'
            try:
                if not proof_native:
                    raise ValueError('Paid manifest lacks original native completion history')
                value = self.grid._semantic(start+ordinal, path)
                self._native_task(start+ordinal, path, value)
                if (value['cell'] != cell.name or value['scenario_digest'] != cell.scenario_digest
                        or value['registered_repetitions'] != cell.repetitions
                        or set(value['claims']) != set(CLAIMS) or value['pending_slots'] != PENDING
                        or value['formal_acceptance'] is not False):
                    raise ValueError('Original cell report changed')
            except Exception:
                value = _cell_numbers(plan, cell, [None]*cell.repetitions)
            cells.append(value)
        return {'version': VERSION, 'estimand': 'nominal_full_cold_rolling',
            'registry_digest': self.actual.digest, 'verification_digest': self.grid.registry.digest,
            'plan_digest': plan.digest, 'plan': plan.to_dict(), 'registered_N': self.grid.registry.plan.total_N,
            'cells': cells, 'claims': list(CLAIMS), 'pending_slots': PENDING,
            'paid_full_grid_proofs': [fingerprint(before), fingerprint(after)],
            'paid_full_grid_native_history_complete': proof_native, 'native_proof_failure': proof_failure,
            'numeric_targets_met': False, 'all_five_claims_authenticated': False,
            'formal_gate_eligible': False, 'formal_acceptance': False, 'full_V7': False,
            'verification_cost_registered': True, 'native_report_completion': 'pending_external_original_reader',
            'confidence': {'monte_carlo_alpha': plan.monte_carlo_alpha, 'oracle_alpha': plan.oracle_alpha,
                'per_bound_tail_alpha': plan.tail_alpha, 'library_error_uses_original_cell_error_bound': True,
                'family_upper_one_is_deterministic': True},
            'pending': ['other_four_nominal_truth_producers', 'original_native_report_completion',
                        'full_real_pipeline_acceptance']}

    def execute_allocated(self, task):
        from .nominal_verification_budget import VerificationTask
        if type(task) is not VerificationTask or task.task['kind'] not in KINDS:
            raise TypeError('Original acquired full-N verification task required')
        index = task.task['index']
        try:
            self.grid._task(task, index)
            if _installed() != self.grid.registry.plan.binding['source_files']:
                raise ValueError('Full-N source differs from original registration')
            self._binding()
            directory = self.grid.directory/'full-N'
            directory.mkdir(exist_ok=True)
            if task.task['kind'] == 'actual_coordinate_readback':
                payload = self._readback(task)
                target = directory/f'coordinate-{index:06d}.json'
            elif task.task['kind'] == 'acceptance_cell_aggregation':
                plan = self.grid.registry.plan.acceptance
                cell = next(c for c in plan.cells if c.name == task.task['unit']['cell'])
                payload = self._aggregate(task, cell)
                target = directory/f'cell-{list(plan.cells).index(cell):03d}.json'
            else:
                payload = self._final(task)
                target = directory/'report.json'
            task.check()
            return self.grid._publish(task, payload, target)
        except BaseException as error:
            if not task.closed:
                try:task.fail(type(error).__name__+':'+str(error)[:1000])
                except Exception as failure_error:error.add_note('Original failure recording also failed: '+str(failure_error)[:1000])
            raise
