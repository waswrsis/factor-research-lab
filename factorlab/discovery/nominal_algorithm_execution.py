"""New prospective actual execution transport for a complete adaptive policy.

Reuses original paid kernels, not their old scientific estimand. Returned raw
actual traces ARE NOT nominal cold risks. Complete nominal readiness/conformance
is still needed. There is no injected command, clock, provider or success flag.
"""
from __future__ import annotations
from pathlib import Path
from contextvars import ContextVar
import os
import threading

VERSION = 'nominal-complete-algorithm-original-transport-v1'
RELATIVE = 'factorlab/discovery/nominal_algorithm_execution.py'
EVENT = 'nominal_complete_algorithm_transport_bound'
_BIND = ContextVar('nominal_algorithm_bind', default=None)
_RUN = ContextVar('nominal_algorithm_run', default=None)


def _owner(permit):
    return (permit, os.getpid(), threading.get_ident())


def require_execution_scope(permit):
    parent = permit.registry.directory
    if not ((parent/'nominal_algorithm_total_v2').exists()
            or (parent/'nominal_algorithm_total_v2_anchor.json').exists()):
        return
    if _BIND.get() != _owner(permit) and _RUN.get() != _owner(permit):
        raise ValueError('New mode requires its original private owner invocation')
    require_registered_mode(permit)


def _installed():
    from .monthly_work_authority import source_files
    from .evidence import file_hash
    from factorlab.protocol import project_path
    files = source_files()
    if (Path(__file__).resolve() != project_path(RELATIVE).resolve()
            or files.get(RELATIVE) != file_hash(__file__)):
        raise ValueError('New integrated canonical execution source required')
    from importlib import import_module
    for name in ('full_trial_controller', 'full_calibration_registry', 'monthly_work_authority',
                 'finite_market_oracle', 'claim_protocol', 'quarantined_observation',
                 'nominal_algorithm_total_risk', 'nominal_total_loss'):
        module = import_module('factorlab.discovery.'+name)
        relative = 'factorlab/discovery/'+name+'.py'
        if (Path(module.__file__).resolve() != project_path(relative).resolve()
                or files.get(relative) != file_hash(module.__file__)):
            raise ValueError('Loaded exact kernel dependency differs from canonical source: '+name)
    return files


def _paths(permit):
    from .calibration_trial_context import _no_redirect
    roots = (permit.paths['root']/'nominal_algorithm_execution',
             permit.registry.directory/'nominal_algorithm_executions'/permit.trial['trial_id'])
    for p in roots:
        _no_redirect(p, permit.registry.directory)
    return roots


def _pair(roots, name):
    from .nominal_cold_conformance import _read
    a, b = (_read(p/name) for p in roots)
    if a != b:
        raise ValueError('New algorithm transport mirror differs')
    return a


def _write_pair(roots, name, value):
    from .nominal_cold_conformance import _write
    for p in roots:
        _write(p/name, value)


def _exclusive(roots):
    if any(p.exists() for p in roots):
        raise ValueError('Prior or partial transport allocation consumed; no retry')
    for p in roots:
        p.mkdir(parents=True, exist_ok=False)


def require_registered_mode(permit):
    """Narrow original-controller hook. No effect on parents without new design.

    No FullTrialController call inside this guard: its _permit calls this guard.
    Missing either new anchor cannot silently restore the old controller route.
    """
    from .full_calibration_registry import QuarantinedTrialPermit
    if type(permit) is not QuarantinedTrialPermit:
        raise TypeError('Exact original paid transport permit required')
    parent = permit.registry.directory
    marked = ((parent/'nominal_algorithm_total_v2').exists()
              or (parent/'nominal_algorithm_total_v2_anchor.json').exists())
    if not marked:
        return None
    _installed()
    from .nominal_algorithm_total_risk import AlgorithmTotalRiskRegistry
    from .calibration_runtime_contract import _original_operations
    from factorlab.protocol import fingerprint
    design = AlgorithmTotalRiskRegistry.inspect(permit.registry)
    roots = _paths(permit)
    value = _pair(roots, 'identity.json')
    if (value['version'] != VERSION or value['design_digest'] != design.digest
            or value['trial'] != permit.trial
            or value['source_files'] != design.identity['source_files']
            or value['full_schedule'] != design.identity['full_schedule']
            or value['trace_scope'] != 'actual_complete_policy_transport_not_nominal_truth'
            or value['nominal_risk_certified'] is not False):
        raise ValueError('New transport identity/source/scope changed')
    expected_intent = {k: value[k] for k in ('version', 'design_digest', 'trial', 'source_files',
                                            'endpoints', 'original_service_clock')}
    if _pair(roots, 'intent.json') != expected_intent:
        raise ValueError('Original prospective new-mode intent changed')
    if any((p/'run-attempt'/'failure.json').exists() for p in roots):
        raise ValueError('Any original transport failure remains consumed')
    if not all((p/'controller.json').is_file() for p in roots) and _BIND.get() != _owner(permit):
        raise ValueError('Partial new controller initialization cannot be resumed')
    # Original permit retains wallet/source/LC checks. Event reads are original.
    with permit.open_lifecycle() as lc:
        ops = _original_operations(lc)
        events = [o for o in ops if o['kind'] == EVENT]
        bound = _pair(roots, 'bound.json')
        if (len(events) != 1 or bound != events[0]
                or events[0]['detail'] != {'transport_hash': fingerprint(value)}):
            raise ValueError('Original lifecycle does not bind the new transport')
        from .claim_protocol import ClaimProtocol
        claims = ClaimProtocol(lc)
        if (claims.digest != value['claim_protocol_digest']
                or claims.endpoints['algorithm'].to_dict() != value['algorithm_endpoint']
                or [claims.endpoints[e['slot']].to_dict() for e in value['endpoints']] != value['endpoints']):
            raise ValueError('Original full policy endpoint changed')
    return value


class AlgorithmExecutionTransport:
    """One original full controller invocation, with no additional quota."""
    @classmethod
    def bind(cls, design, permit, *, runtime, authority, generator, endpoints):
        from .nominal_algorithm_total_risk import AlgorithmTotalRiskRegistry
        from .full_calibration_registry import QuarantinedTrialPermit
        from .v7_forward import ForwardRuntime
        from .monthly_work_authority import MonthlyWorkAuthority
        from .finite_market_oracle import FiniteGenerator
        from .claim_protocol import FrozenClaimEndpoint, ClaimProtocol
        from .full_trial_controller import FullControllerSchedule, FullTrialController, ControllerStepLease
        from .quarantined_observation import QuarantinedObservation
        from .calibration_runtime_contract import _original_operations
        from .v7_lifecycle import stamp
        from factorlab.protocol import fingerprint
        from filelock import FileLock
        if (type(design) is not AlgorithmTotalRiskRegistry or type(permit) is not QuarantinedTrialPermit
                or type(runtime) is not ForwardRuntime or type(authority) is not MonthlyWorkAuthority
                or type(generator) is not FiniteGenerator or type(endpoints) is not tuple
                or len(endpoints) != 2 or any(type(e) is not FrozenClaimEndpoint for e in endpoints)):
            raise TypeError('Exact registered original runtime/wallet/generator/whole endpoints required')
        schedule = FullControllerSchedule.from_dict(design.identity['full_schedule'])
        initial = ControllerStepLease.issue(schedule)
        initial.assert_parent_alive()
        files = _installed()
        design.verify(); permit.verify(runtime.lifecycle); generator.verify_registry(permit.registry)
        from .nominal_algorithm_policy import require_endpoint_policy
        require_endpoint_policy(design, endpoints)
        initial.assert_parent_alive()
        if (design.parent.digest != permit.registry.digest or permit.trial.get('estimand_slot') is not None
                or permit.trial['role'] not in ('oracle', 'validation') or authority.directory != permit.paths['authority']
                or runtime.directory != permit.paths['runtime'] or runtime.feed.directory != permit.paths['feed']
                or authority.manifest['source_files'] != files):
            raise ValueError('Transport cannot borrow a selected learner, foreign wallet or old source')
        if schedule.observation_mode != 'bound_logical_observation':
            raise ValueError('This new simulation transport requires registered original observation')
        algorithm = next((e for e in endpoints if e.slot == 'algorithm'), None)
        if (algorithm is None or stamp(schedule.due_at[-1]) < stamp(algorithm.scoring_cutoff)
                or stamp(schedule.due_at[0]) > stamp(algorithm.start)
                or runtime.identity['rolling']['months'] != 3):
            raise ValueError('Whole six-month endpoint and original three-month rolling required')
        roots = _paths(permit)
        with FileLock(str(permit.paths['root']/'.nominal-algorithm-bind.lock'), timeout=0):
            if (authority.verify()['reservations']
                    or any(runtime.lifecycle.db.execute('SELECT COUNT(*) FROM '+t).fetchone()[0]
                           for t in ('jobs', 'active', 'batches', 'settlement'))
                    or runtime.lifecycle.forecasts.db.execute('SELECT COUNT(*) FROM forecasts').fetchone()[0]
                    or runtime.feed.db.execute('SELECT COUNT(*) FROM batches').fetchone()[0]):
                raise ValueError('New mode cannot reinterpret prior actual work')
            _exclusive(roots)
            _write_pair(roots, 'intent.json', {'version': VERSION, 'design_digest': design.digest,
                'trial': permit.trial, 'source_files': files,
                'endpoints': [e.to_dict() for e in endpoints], 'original_service_clock': initial.to_dict()})
            # Preserve original required order: bind observer before claim events.
            observer = QuarantinedObservation.bind(runtime, generator, permit)
            from .finite_market_oracle import publish_due
            initial.assert_parent_alive()
            publish_due(generator, permit, max_new_receipts=1)
            initial.assert_parent_alive()
            observer.advance(runtime)
            initial.assert_parent_alive()
            claims = ClaimProtocol.freeze(runtime.lifecycle, endpoints)
            value = {'version': VERSION, 'design_digest': design.digest, 'trial': permit.trial,
                'source_files': files, 'full_schedule': schedule.to_dict(),
                'authority_manifest_hash': authority.manifest_hash,
                'observation_identity_hash': fingerprint(observer._identity()),
                'claim_protocol_digest': claims.digest, 'algorithm_endpoint': algorithm.to_dict(),
                'endpoints': [e.to_dict() for e in endpoints], 'original_service_clock': initial.to_dict(),
                'trace_scope': 'actual_complete_policy_transport_not_nominal_truth',
                'nominal_risk_certified': False}
            _write_pair(roots, 'identity.json', value)
            with runtime.lifecycle.db:
                runtime.lifecycle._event(EVENT, {'transport_hash': fingerprint(value)})
            events = [o for o in _original_operations(runtime.lifecycle) if o['kind'] == EVENT]
            if len(events) != 1:
                raise ValueError('Ambiguous new mode event')
            _write_pair(roots, 'bound.json', events[0])
            token = _BIND.set(_owner(permit))
            try:
                controller = FullTrialController.freeze(permit, runtime, authority, schedule)
                initial.assert_parent_alive()
                if controller._start_execution(initial).to_dict() != initial.to_dict():
                    raise ValueError('New transport cannot adopt an existing service clock')
            finally:
                _BIND.reset(token)
            _write_pair(roots, 'controller.json', {'identity_hash': fingerprint(controller.identity)})
        return cls.inspect(permit)

    @classmethod
    def inspect(cls, permit):
        value = require_registered_mode(permit)
        if value is None:
            raise ValueError('Old controller has no new algorithm transport registration')
        from .full_trial_controller import FullTrialController
        from factorlab.protocol import fingerprint
        obj = cls.__new__(cls)
        obj.permit, obj.identity, obj.roots = permit, value, _paths(permit)
        obj.controller = FullTrialController.inspect(permit)
        if _pair(obj.roots, 'controller.json') != {'identity_hash': fingerprint(obj.controller.identity)}:
            raise ValueError('Original controller replaced')
        return obj

    def run_once(self):
        """Call the unchanged original complete adaptive policy, once.

        Administrative bind does not execute work. Here the original absolute
        service clock is sampled before marker IO/full revalidation and is used
        by all original controller calls, including trace readback. No new
        fitting wallet, timeout or renewal is introduced.
        """
        from .full_trial_controller import ControllerStepLease
        from factorlab.protocol import fingerprint
        from filelock import FileLock
        initial = ControllerStepLease.from_dict(self.identity['original_service_clock'], self.controller.schedule)
        initial.assert_parent_alive()
        if _paths(self.permit) != self.roots:
            raise ValueError('Transport paths differ from original registered coordinate')
        with FileLock(str(self.roots[0]/'.run.lock'), timeout=0):
            attempts = tuple(p/'run-attempt' for p in self.roots)
            _exclusive(attempts)
            _write_pair(attempts, 'intent.json', {'transport_hash': fingerprint(self.identity),
                'original_service_clock': initial.to_dict()})
            token = _RUN.set(_owner(self.permit))
            try:
                lease = self.controller._start_execution(initial)
                if lease.to_dict() != initial.to_dict():
                    raise ValueError('Original clock already consumed; no transport retry')
                lease.assert_parent_alive()
                current = type(self).inspect(self.permit)
                if current.identity != self.identity:
                    raise ValueError('Transport identity changed')
                result = self.controller.run(max_pumps=7200)
                lease.assert_parent_alive()
                rows = self.controller.verify()
                if (len(rows) != len(self.controller.schedule.due_at)
                        or not all(self.controller.step_completed(row) for row in rows)):
                    raise ValueError('Incomplete or failed original full schedule remains consumed')
                from .nominal_algorithm_producer import finish_original_transport
                trace = finish_original_transport(self, lease)
                lease.assert_parent_alive()
                value = {'version': VERSION, 'transport_hash': fingerprint(self.identity),
                    'original_result': result, 'actual_trace': trace,
                    'original_clock_hash': fingerprint(self.controller._execution_clock()),
                    'physical_finish': lease.observe(), 'nominal_truth': 'unestablished',
                    'formal_acceptance': False, 'publication_state': 'provisional_requires_native_completion'}
                _write_pair(attempts, 'result.json', value)
                lease.assert_parent_alive()
                return value
            except BaseException as error:
                failure = {'version': VERSION, 'error_type': type(error).__name__,
                    'state': 'failed_or_unknown_consumed', 'possible_error': 1, 'power': 0,
                    'transport_hash': fingerprint(self.identity)}
                for p in attempts:
                    try: _write_pair((p,), 'failure.json', failure)
                    except BaseException: pass
                raise
            finally:
                _RUN.reset(token)

    def _actual_trace(self):
        from .calibration_oracle_evidence import OracleClaimSpec, reconstruct_claim, verify_raw_claim
        from factorlab.protocol import fingerprint
        e = self.identity['algorithm_endpoint']
        spec = OracleClaimSpec('algorithm', e['start'], e['end'], e['reference_stream'],
                               e['candidate_stream'], e['reference'], e['candidate'])
        result = reconstruct_claim(self.permit.paths['runtime'], spec)
        raw = verify_raw_claim(self.permit.paths['runtime'], result)
        pairs = result.pop('pairs')
        return {'kind': 'original_actual_trajectory_only', 'original_reconstruction': result,
            'raw_verification': raw, 'pair_digest': fingerprint(pairs),
            'nominal_total_loss_risk': None,
            'pending': ['complete_nominal_algorithm_ready_calendar',
                        'independent_nominal_full_policy_predictions_and_failure_witnesses',
                        'actual_to_nominal_full_path_conformance', 'complete_provider_kernel_independence']}
