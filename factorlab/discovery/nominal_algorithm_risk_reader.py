"""Source-bound whole-algorithm path readback in the original proof allocation.

Opaque provider dependence is not certified by successful process completion.
The unconditional population range remains [0,B]; the narrower sampling bound
is labelled conditional. No caller loss rows or IID switch are accepted.
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import importlib
import os
import threading

VERSION = 'nominal-algorithm-registered-risk-readback-v1'
RELATIVE = 'factorlab/discovery/nominal_algorithm_risk_reader.py'
B = 2**20


def _rational(value):
    if (type(value) is not list or len(value) != 2
            or any(type(v) is not int for v in value) or value[1] <= 0):
        raise ValueError('Exact reduced original rational required')
    result = Fraction(*value)
    if [result.numerator, result.denominator] != value or not 0 <= result <= B:
        raise ValueError('Noncanonical or out-of-domain nominal risk')
    return result


def _fraction(value):
    return [value.numerator, value.denominator]


def _installed(design):
    from factorlab.protocol import project_path
    from .monthly_work_authority import source_files
    from .evidence import file_hash
    files = source_files()
    if files != design.identity['source_files']:
        raise ValueError('Registered complete source changed')
    for name in ('nominal_algorithm_risk_reader', 'nominal_algorithm_total_risk',
                 'nominal_algorithm_producer', 'nominal_full_scope_proof',
                 'nominal_total_loss', 'work_lease'):
        module = importlib.import_module('factorlab.discovery.' + name)
        relative = 'factorlab/discovery/' + name + '.py'
        if (Path(module.__file__).resolve() != project_path(relative).resolve()
                or files.get(relative) != file_hash(module.__file__)):
            raise ValueError('Only exact canonical risk/proof kernels may read evidence')
    return files


def _active_original_task(design, cell, index):
    """Exact private native task, with ownership from its real producer permit."""
    from . import nominal_full_scope_proof as native
    task = native._PROOF.get()
    if (type(task) is not native._ProofTask or task.key is not native._KEY
            or task.owner != (os.getpid(), threading.get_ident()) or task.closed
            or task.producer.permit.registry.digest != design.parent.digest
            or task.producer.transport.identity['design_digest'] != design.digest
            or task.identity['coordinate'] != design.parent.identity_for(cell, index)):
        raise ValueError('Original active registered validation proof allocation required')
    task.check()
    return task


def _path_payload(publication, coordinate, design):
    """Validate one original complete path; never treat a point as population truth."""
    from factorlab.protocol import fingerprint
    from .nominal_algorithm_policy import NominalAlgorithmPolicy
    from .nominal_full_scope_proof import CLAIMS
    identity = publication['identity']
    proof = publication['proof']
    claims = proof['claims']
    if (identity['trial'] != coordinate
            or identity['policy_digest'] != NominalAlgorithmPolicy.inspect(design).digest
            or identity['source_files'] != design.identity['source_files']
            or proof['source_files'] != design.identity['source_files']
            or proof['conditional_unit'] != 'one_complete_original_paid_adaptive_policy_path'
            or proof['provider_IID_certified'] is not False
            or proof['population_risk_certified'] is not False
            or proof['formal_acceptance'] is not False
            or len(claims) != len(CLAIMS)
            or {row['slot'] for row in claims} != set(CLAIMS)):
        raise ValueError('Original complete five-slot path scope changed')
    selected = next(row for row in claims if row['slot'] == 'algorithm')
    if selected.get('truth') == 'not_selected':
        raise ValueError('The complete algorithm endpoint cannot become a fixed nonselection')
    point = (_rational(selected['reference']), _rational(selected['candidate']))
    return point, fingerprint(selected['estimand']), fingerprint(publication)


def registered_risk_interval(design, cell, index):
    """Read all original whole oracle coordinates inside one existing proof300.

    The native worker invokes this before its original quiescence/exit. Each
    original oracle path must already have its own real natural completion.
    Missing/late/mismatched paths stay in the registered denominator as None.
    No second task or service clock, fit, provider call or oracle draw is issued.
    """
    from .nominal_algorithm_total_risk import (AlgorithmTotalRiskRegistry,
                                               conditional_whole_path_bounds)
    from .nominal_full_scope_proof import (FullScopeProofPlan, read_retained,
                                          _producer_publication)
    from factorlab.protocol import fingerprint
    if type(design) is not AlgorithmTotalRiskRegistry:
        raise TypeError('Exact prospective algorithm design required')
    task = _active_original_task(design, cell, index)
    files = _installed(design)
    initial = design.verify()
    plan = FullScopeProofPlan.inspect(design.parent)
    if plan.identity.get('algorithm_risk_readback') != {
            'version': VERSION, 'allocation': 'validation_original_single_proof300_remaining',
            'oracle_paths': design.parent.plan.oracle_paths,
            'extra_fit_GPT_or_task_credit': False, 'provider_IID_granted': False}:
        raise ValueError('Risk readback was not prospectively registered in this proof')
    paths = []
    rows = []
    expected_estimand = None
    for oracle_index in range(design.parent.plan.oracle_paths):
        task.check()
        coordinate = plan.coordinate(cell, index, oracle_index)
        try:
            observation = read_retained(plan, coordinate)
            observation.assert_current()
            publication = _producer_publication(plan, coordinate)
            point, estimand, publication_hash = _path_payload(publication, coordinate, design)
            if expected_estimand is not None and estimand != expected_estimand:
                raise ValueError('Whole algorithm endpoint changed between original paths')
            native_identity = observation.identity
            if (native_identity['coordinate'] != coordinate
                    or native_identity['native_completed'] is not True
                    or native_identity['actual_proof']['producer_publication_hash'] != publication_hash):
                raise ValueError('Point lacks its original source-bound native completion')
            observation.assert_current()
            task.check()
            # All provenance succeeds before the numeric point enters the sum.
            if expected_estimand is None:
                expected_estimand = estimand
            paths.append(point)
            rows.append({'oracle_index': oracle_index, 'state': 'complete_original_path',
                         'native_hash': fingerprint(native_identity),
                         'publication_hash': publication_hash,
                         'observed_nominal_pair': [_fraction(v) for v in point]})
        except (ValueError, OSError, KeyError, TypeError, ArithmeticError) as error:
            task.check()  # Deadline/owner failure must abort, never publish a cheap fallback.
            paths.append(None)
            rows.append({'oracle_index': oracle_index, 'state': 'unknown_retained',
                         'reason': type(error).__name__ + ':' + str(error)[:500]})
    alpha = Fraction(*initial['oracle_tail_alpha'])
    conditional = conditional_whole_path_bounds(tuple(paths), tail_alpha=alpha)
    complete = len(paths) - conditional.unavailable_paths
    sample_mean = None
    if complete == len(paths):
        sample_mean = {role: _fraction(sum((p[j] for p in paths), Fraction()) / len(paths))
                       for j, role in enumerate(('reference', 'candidate'))}
    # [0,B] is the unconditional source-bounded population enclosure. Native
    # completion and fresh contexts cannot narrow an opaque provider's law.
    result = {'version': VERSION, 'design_digest': design.digest,
        'validation_coordinate': design.parent.identity_for(cell, index),
        'estimand': initial['estimand'], 'registered_paths': len(paths),
        'complete_original_paths': complete, 'unavailable_paths': conditional.unavailable_paths,
        'paths': rows, 'sample_mean_numeric_only': sample_mean,
        'unconditional_population_range': {'reference': [[0, 1], [B, 1]],
                                           'candidate': [[0, 1], [B, 1]]},
        'truth': 'unknown', 'provider_IID_certified': False,
        'conditional_iid_bounds': {role: [_fraction(v) for v in pair]
                                  for role, pair in (('reference', conditional.reference),
                                                     ('candidate', conditional.candidate),
                                                     ('delta_gap', conditional.delta_gap))},
        'conditional_tail_alpha': _fraction(alpha),
        'allocation': {'native_proof_hash': fingerprint(task.identity),
                       'lease': task.lease.to_dict(), 'new_task_credit': False},
        'formal_acceptance': False}
    task.check()
    if design.verify() != initial or _installed(design) != files:
        raise ValueError('Registered original source/design changed across risk readback')
    task.check()
    return result


def validate_retained_readback(value, design, coordinate, *, binding_hash, lease):
    """Check the recorded worker readback without granting provider independence."""
    if (type(value) is not dict or value.get('version') != VERSION
            or value.get('design_digest') != design.digest
            or value.get('validation_coordinate') != coordinate
            or coordinate.get('oracle_index') is not None
            or value.get('estimand') != design.identity['estimand']
            or value.get('registered_paths') != design.parent.plan.oracle_paths
            or value.get('truth') != 'unknown'
            or value.get('provider_IID_certified') is not False
            or value.get('formal_acceptance') is not False
            or value.get('unconditional_population_range') != {
                'reference': [[0, 1], [B, 1]], 'candidate': [[0, 1], [B, 1]]}
            or value.get('allocation') != {'native_proof_hash': binding_hash,
                                          'lease': lease, 'new_task_credit': False}):
        raise ValueError('Original worker conservative risk readback changed')
    rows = value.get('paths')
    n = design.parent.plan.oracle_paths
    if type(rows) is not list or len(rows) != n:
        raise ValueError('Every original oracle coordinate must remain')
    complete = 0
    paths = []
    for index, row in enumerate(rows):
        if type(row) is not dict or row.get('oracle_index') != index:
            raise ValueError('Whole path order or denominator changed')
        if row.get('state') == 'complete_original_path':
            if (set(row) != {'oracle_index', 'state', 'native_hash', 'publication_hash',
                            'observed_nominal_pair'}
                    or any(type(row[k]) is not str or len(row[k]) != 64
                           or any(c not in '0123456789abcdef' for c in row[k])
                           for k in ('native_hash', 'publication_hash'))):
                raise ValueError('Retained native path binding missing')
            pair = row['observed_nominal_pair']
            if type(pair) is not list or len(pair) != 2:
                raise ValueError('Exact complete path pair required')
            paths.append(tuple(_rational(v) for v in pair))
            complete += 1
        elif row.get('state') == 'unknown_retained':
            if set(row) != {'oracle_index', 'state', 'reason'} or type(row['reason']) is not str:
                raise ValueError('Unknown path must retain its source read failure')
            paths.append(None)
        else:
            raise ValueError('Unknown whole path state')
    if (any(type(value.get(k)) is not int for k in
            ('registered_paths', 'complete_original_paths', 'unavailable_paths'))
            or value.get('complete_original_paths') != complete
            or value.get('unavailable_paths') != n - complete
            or value.get('conditional_tail_alpha') != design.identity['oracle_tail_alpha']
            or (complete != n and value.get('sample_mean_numeric_only') is not None)):
        raise ValueError('Retained population enclosure or counts changed')
    from .nominal_algorithm_total_risk import conditional_whole_path_bounds
    expected = conditional_whole_path_bounds(tuple(paths),
                                             tail_alpha=Fraction(*design.identity['oracle_tail_alpha']))
    bounds = {role: [_fraction(v) for v in pair]
              for role, pair in (('reference', expected.reference),
                                 ('candidate', expected.candidate),
                                 ('delta_gap', expected.delta_gap))}
    mean = None if complete != n else {
        role: _fraction(sum((p[j] for p in paths), Fraction()) / n)
        for j, role in enumerate(('reference', 'candidate'))}
    if value.get('conditional_iid_bounds') != bounds or value.get('sample_mean_numeric_only') != mean:
        raise ValueError('Recorded conditional arithmetic changed; it is not provider certification')
    return value
