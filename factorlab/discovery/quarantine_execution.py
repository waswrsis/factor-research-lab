"""Reconstruct original quarantine scope; reuse production decision kernels.

Nothing in this module is a certificate, a caller switch or a new work grant.
The ordinary calibration gate remains independent of this isolated execution.
"""
from __future__ import annotations

from factorlab.protocol import fingerprint


def _verified_scope(lifecycle):
    from .calibration_runtime_contract import _original_operations
    from .full_calibration_registry import EVENT, FullTrialRegistry, QuarantinedTrialPermit

    enrolled = [r['detail'] for r in _original_operations(lifecycle) if r['kind'] == EVENT]
    if not enrolled:
        return None, None
    if len(enrolled) != 1:
        raise ValueError('Quarantine execution requires one original enrollment')
    detail = enrolled[0]
    # Fixed layout only; a caller cannot supply another registry or grant object.
    registry = FullTrialRegistry.inspect(lifecycle.directory.parent.parent.parent)
    trial = registry.identity_for(detail['cell'], detail['index'], oracle_index=detail['oracle_index'],
                                  estimand_slot=detail.get('estimand_slot'))
    permit = QuarantinedTrialPermit(registry, trial)
    return permit, permit.verify(lifecycle)


def permit_for(lifecycle):
    return _verified_scope(lifecycle)[0]


def decision_for(lifecycle, result):
    permit, bound = _verified_scope(lifecycle)
    if permit is None:
        return None
    if (result.get('supported') is not False or result.get('calibration_passed') is not False
            or result.get('formal_eligible') is not False):
        raise ValueError('Quarantine cannot authorize a formal result')
    return {'version': 'v7-quarantined-rule-decision-v1',
            'registry_digest': bound['registry_digest'], 'trial_id': bound['trial_id'],
            'runtime_scope_hash': bound['runtime_scope_hash'],
            'rule_id': result['rule_id'],
            'accepted': result.get('diagnostic_status') == 'supported_target',
            'execution_scope': 'quarantined_simulation_only'}


def verify_decision(lifecycle, result):
    expected = decision_for(lifecycle, result)
    if result.get('quarantined_decision') != expected:
        raise ValueError('Quarantine result is outside its original trial or rule')
    return expected


def six_month_claim_scope(protocol):
    from .calibration_runtime_contract import runtime_calibration_contract

    terms = {}
    for slot, endpoint in protocol.endpoints.items():
        value = endpoint.to_dict()
        terms[slot] = {k: v for k, v in value.items() if k not in {'start', 'end'}}
        terms[slot]['nominal_UTC_calendar_months'] = 6
    return {'version': 'v7-six-month-claim-calibration-scope-v1',
            'runtime_scope_hash': fingerprint(runtime_calibration_contract(protocol.lifecycle)),
            'endpoints': terms, 'aggregation': 'equal_symbol_then_hour',
            'support': 'nominal_window_missing_pairs_do_not_shrink_denominator',
            'single_look': True}


def claim_formal_ready(protocol):
    from .v7_lifecycle import Lifecycle
    lc = protocol.lifecycle
    if permit_for(lc) is not None:
        return False
    if (not lc.plan.calibration_directory or not lc.plan.calibration_binding
            or not Lifecycle.calibration_ready(lc)):
        return False
    from .pipeline_calibration import verify_calibration_certificate
    result = verify_calibration_certificate(lc.plan.calibration_directory,
                                           expected_binding=lc.plan.calibration_binding)
    return result.get('six_month_claim_scope_hash') == fingerprint(six_month_claim_scope(protocol))


def verify_claim_result(protocol, score):
    result = score['result']
    formal = score.get('formal_eligible')
    if (type(formal) is not bool or result.get('calibration_passed') is not formal
            or result.get('formal_eligible', formal) is not formal):
        raise ValueError('Claim result gate identity changed or self-authorized')
    if formal:
        if (not claim_formal_ready(protocol)
                or score.get('six_month_claim_scope_hash') != fingerprint(six_month_claim_scope(protocol))
                or result.get('quarantined_decision') is not None):
            raise ValueError('Claim formal certificate/scope no longer verifies')
        accepted = result.get('diagnostic_status') == 'supported_target'
        if result.get('supported') is not accepted:
            raise ValueError('Formal endpoint changed the numerical rule decision')
    else:
        if result.get('supported') is not False:
            raise ValueError('Diagnostic claim cannot self-authorize support')
        verify_decision(protocol.lifecycle, result)
