"""Prospective simulation-only full-policy order/readiness and conformance.

This new algorithm version settles/selects before predicting. It is not source
equivalent to the previous tick-before-select algorithm or real deployment.
Original fits, provider requests, model admission and all physical caps remain.
"""
from __future__ import annotations
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import os
import threading
import math

VERSION = 'nominal-complete-policy-settle-select-tick-v1'
ELAPSED_VERSION = 'nominal-complete-policy-settle-select-tick-elapsed90-v2'
RELATIVE = 'factorlab/discovery/nominal_algorithm_policy.py'
PHASES = ('original_source_release', 'original_integrity_and_physical_guard',
          'settle_original_mature_labels', 'freeze_due_original_batches',
          'original_V_selection_C_reservation_and_single_look_endpoints',
          'tick_existing_admitted_models_and_forecast',
          'original_monthly_selection_and_plan', 'original_typed_resource_and_fit_priority',
          'original_remaining_one_fit_opportunity', 'original_S_scheduler',
          'original_research_drain', 'original_successor_witness')
_ENDPOINT = ContextVar('nominal_algorithm_endpoint_scope', default=None)
_KEY = object()


def _utc(value):
    if type(value) is not str:
        raise TypeError('Exact UTC timestamp string required')
    d = datetime.fromisoformat(value)
    if d.tzinfo is None or d.utcoffset() != timedelta(0):
        raise ValueError('UTC timestamp required')
    return d


def _month(value, change=0):
    d = _utc(value)
    n = d.year*12+d.month-1+change
    return datetime(n//12, n % 12+1, 1, tzinfo=timezone.utc)


def cold_window(cutoff, *, cadence_anchor=None):
    c = _utc(cutoff)
    if cadence_anchor is not None:
        from .rolling_cadence import RollingCadence
        RollingCadence().slot_identity(_utc(cadence_anchor), c)
        return ((c-timedelta(days=90)).isoformat(), c.isoformat())
    if c != _month(cutoff):
        raise ValueError('Exact original UTC monthly cold cutoff required')
    return (_month(cutoff, -3).isoformat(), c.isoformat())


def full_origin_support(start, end, symbols, delay):
    """No observed-row selection; entire fixed half-open endpoint support."""
    begin, finish = _utc(start), _utc(end)
    if (begin != _month(start) or finish != _month(start, 6)
            or type(delay) is not int or not 1 <= delay < 3600
            or type(symbols) is not tuple or not symbols or len(set(symbols)) != len(symbols)
            or any(type(s) is not str or not s for s in symbols)):
        raise ValueError('Original six-month calendar/universe and in-hour release required')
    for h in range(int((finish-begin).total_seconds())//3600):
        origin = (begin+timedelta(hours=h, seconds=delay)).isoformat()
        for symbol in symbols:
            yield origin, symbol


def c_first_origin(v_end, selected_at, delay, *, policy):
    """Pure eligibility calculation, not selection/model/availability authority.

    Old policy selected after its tick, then rounded activation up an hour.
    New simulation policy selects before tick, from labels available at V_end.
    No model is created by either calculation. Physical late selection is not B.
    """
    end, selected = _utc(v_end), _utc(selected_at)
    if (end != _month(v_end) or type(delay) is not int or not 1 <= delay < 3600
            or selected < end or selected > end+timedelta(hours=24)
            or selected.minute != 0 or selected.second != delay or selected.microsecond
            or policy not in ('original_tick_before_selection', VERSION, ELAPSED_VERSION)):
        raise ValueError('Original C selection opportunity/24h window required')
    if policy in (VERSION, ELAPSED_VERSION):
        return selected.isoformat()
    return (selected.replace(minute=0, second=0, microsecond=0)
            +timedelta(hours=1, seconds=delay)).isoformat()


def model_visible_at(*, completion_prefix, origin, cutoff, same_recipe, cadence_anchor=None):
    """Unprivileged exact nominal readiness predicate; no artifact admission."""
    if type(same_recipe) is not bool:
        raise TypeError('Exact recipe comparison result required')
    cold_window(cutoff, cadence_anchor=cadence_anchor)
    return (same_recipe and _utc(completion_prefix) < _utc(origin)
            and _utc(cutoff) <= _utc(origin))


def _installed():
    from .monthly_work_authority import source_files
    from .evidence import file_hash
    from factorlab.protocol import project_path
    from importlib import import_module
    files = source_files()
    for name in ('nominal_algorithm_policy', 'nominal_algorithm_execution',
                 'nominal_algorithm_total_risk', 'nominal_total_loss', 'v7_service',
                 'v7_lifecycle', 'v7_forward', 'claim_protocol', 'typed_policy_confirmation',
                 'quarantined_observation', 'full_trial_controller'):
        module = import_module('factorlab.discovery.'+name)
        relative = 'factorlab/discovery/'+name+'.py'
        if (Path(module.__file__).resolve() != project_path(relative).resolve()
                or files.get(relative) != file_hash(module.__file__)):
            raise ValueError('New policy and all loaded kernels require exact canonical integration')
    return files


class NominalAlgorithmPolicy:
    @classmethod
    def freeze(cls, design, *, endpoints):
        from .nominal_algorithm_total_risk import AlgorithmTotalRiskRegistry, _fresh_parent
        from .claim_protocol import FrozenClaimEndpoint
        from .nominal_cold_conformance import _write
        from filelock import FileLock
        if type(design) is not AlgorithmTotalRiskRegistry or type(endpoints) is not tuple:
            raise TypeError('Exact original new design and immutable endpoints required')
        if len(endpoints) != 2 or any(type(e) is not FrozenClaimEndpoint for e in endpoints):
            raise TypeError('Both typed original six-month endpoints required')
        _installed(); design.verify(); _fresh_parent(design.parent)
        directory = design.directory/'policy'
        with FileLock(str(design.directory/'.policy-creation.lock'), timeout=0):
            _fresh_parent(design.parent)
            if directory.exists() or (design.parent.directory/'nominal_algorithm_policy.json').exists():
                raise ValueError('Prior/partial policy freeze cannot be replaced')
            directory.mkdir()
            value = cls._expected(design, [e.to_dict() for e in endpoints])
            _write(directory/'identity.json', value)
            _write(design.parent.directory/'nominal_algorithm_policy.json', value)
        return cls.inspect(design)

    @staticmethod
    def _expected(design, endpoints):
        from .claim_protocol import FrozenClaimEndpoint
        schedule = design.identity['full_schedule']
        typed = [FrozenClaimEndpoint.from_dict(e) for e in endpoints]
        if ({e.slot for e in typed} != {'library_skill', 'algorithm'}
                or len({e.start for e in typed}) != 1
                or schedule['observation_mode'] != 'bound_logical_observation'):
            raise ValueError('This policy is registered synthetic simulation only')
        lib = next(e for e in typed if e.slot == 'library_skill')
        algo = next(e for e in typed if e.slot == 'algorithm')
        if lib.candidate != algo.reference or lib.candidate_stream != algo.reference_stream:
            raise ValueError('Original common K0 reference cannot change')
        due = tuple(_utc(s) for s in schedule['due_at'])
        delay = due[0].second
        if (not 1 <= delay < 60 or any(d.minute or d.second != delay or d.microsecond for d in due)
                or any(b-a != timedelta(hours=1) for a, b in zip(due, due[1:]))
                or due[0] > _utc(algo.start)+timedelta(seconds=delay)
                or due[-1] < _utc(algo.scoring_cutoff)):
            raise ValueError('Every original hourly opportunity and complete scoring horizon required')
        scope = design.parent.plan.runtime_scope
        if scope['rolling']['months'] != 3 or scope['lifecycle_policy']['algorithm_months'] != 6:
            raise ValueError('Original cold-three/full-six scope required')
        window_days = scope['rolling'].get('window_days')
        elapsed = window_days == 90
        if ((window_days is not None and (type(window_days) is not int or window_days != 90))
                or scope['forward'].get('training_window_days') != window_days
                or scope['lifecycle_policy'].get('training_window_days') != window_days
                or scope['forward'].get('refit_interval_days') != (15 if elapsed else None)
                or scope['lifecycle_policy'].get('refit_interval_days') != (15 if elapsed else None)):
            raise ValueError('Complete nominal policy changes the original rolling cadence')
        symbols = tuple(scope['work_budget']['universe'])
        support = sum(1 for _ in full_origin_support(algo.start, algo.end, symbols, delay))
        phases = list(PHASES)
        if elapsed:
            phases[6] = 'original_15_day_selection_and_plan_calendar_wallet_unchanged'
        value = {'version': ELAPSED_VERSION if elapsed else VERSION,
            'design_digest': design.digest, 'source_files': _installed(),
            'endpoints': endpoints, 'phases': phases, 'delay_seconds': delay,
            'symbols': list(symbols), 'nominal_symbol_hours': support,
            'fit_priority': 'original_typed_resource_first_then_cutoff_DESC_stream_ASC',
            'fit_opportunities': 'unchanged_original_max_one_per_step_no_prefit_or_retry',
            'ready': 'original_admitted_model_published_before_current_release_same_recipe',
            'cold_window': ('continuous_90_days_at_original_15_day_cutoff' if elapsed else
                            'previous_three_calendar_months_at_each_original_cutoff'),
            'C_start': 'original_V_end_full_support_eligibility_at_current_registered_prefix',
            'C_priming': 'no_new_fit_existing_V_model_only_if_original_admission_and_recipe_match',
            'numeric_fit_policy': __import__('factorlab.discovery.nominal_algorithm_fit',fromlist=['descriptor']).descriptor(),
            'nominal_missing': 'B_only_source_authenticated_structural_or_typed_kernel_inability',
            'physical_failure': 'error_one_power_zero_no_nominal_completion',
            'nominal_truth_certified': False, 'provider_IID_certified': False,
            'deployment_scope': 'registered_simulation_version_not_ordinary_physical_serving'}
        if elapsed:
            from .rolling_cadence import RollingCadence
            value['fit_schedule'] = {'cadence_hash': RollingCadence().digest,
                'training_days': 90, 'refit_days': 15, 'refresh_days': 1,
                'anchor': 'first_original_data_ready_prefix_in_each_complete_path',
                'wallet': 'unchanged_actual_UTC_calendar_month',
                'missed_slots': 'no_backfill_or_refund'}
        return value

    @classmethod
    def inspect(cls, design):
        from .nominal_algorithm_total_risk import AlgorithmTotalRiskRegistry
        from .nominal_cold_conformance import _read
        from factorlab.protocol import fingerprint
        if type(design) is not AlgorithmTotalRiskRegistry:
            raise TypeError('Exact new design required')
        design.verify()
        obj = cls.__new__(cls); obj.design = design
        obj.identity = _read(design.directory/'policy/identity.json')
        expected = cls._expected(design, obj.identity['endpoints'])
        if (obj.identity != expected
                or _read(design.parent.directory/'nominal_algorithm_policy.json') != expected):
            raise ValueError('Policy/source/complete support changed')
        obj.digest = fingerprint(expected)
        return obj


def require_endpoint_policy(design, endpoints):
    policy = NominalAlgorithmPolicy.inspect(design)
    if [e.to_dict() for e in endpoints] != policy.identity['endpoints']:
        raise ValueError('Actual transport differs from prospective complete policy endpoints')
    return policy


def _bound(lifecycle):
    from .quarantine_execution import permit_for
    permit = permit_for(lifecycle)
    if permit is None:
        return None
    parent = permit.registry.directory
    marker = parent/'nominal_algorithm_policy.json'
    if not marker.exists():
        # A partial policy directory cannot silently choose the old order.
        if (parent/'nominal_algorithm_total_v2/policy').exists():
            raise ValueError('Partial new policy binding')
        return None
    from .nominal_algorithm_total_risk import AlgorithmTotalRiskRegistry
    from .nominal_algorithm_execution import require_registered_mode
    design = AlgorithmTotalRiskRegistry.inspect(permit.registry)
    policy = NominalAlgorithmPolicy.inspect(design)
    mode = require_registered_mode(permit)
    if mode is None or mode['endpoints'] != policy.identity['endpoints']:
        raise ValueError('New event order requires its original registered transport')
    return policy, permit


def _live_prefix(lifecycle):
    from .quarantined_observation import current_binding, _physical, _check_physical
    from .v7_lifecycle import stamp
    prefix = current_binding(lifecycle)
    if prefix is None:
        raise ValueError('Physical service cannot borrow synthetic prefix permission')
    fresh = _physical(); previous = prefix['physical_observation']
    _check_physical(fresh, previous)
    if (fresh['boot'] != previous['boot'] or fresh['pid'] != previous['pid']
            or fresh['process_created'] != previous['process_created']
            or max((stamp(fresh['wall'])-stamp(previous['wall'])).total_seconds(),
                   fresh['monotonic']-previous['monotonic']) > lifecycle.plan.prediction_max_lag_seconds):
        raise ValueError('Original actual prediction maxlag/process exceeded; not nominal B')
    return prefix


class _BeforeTick:
    def __init__(self, key, lifecycle, policy, prefix, roots):
        if key is not _KEY:
            raise TypeError('Only the original endpoint phase creates this handle')
        self.lifecycle, self.policy, self.prefix = lifecycle, policy, prefix
        self.roots = roots
        self.owner = (os.getpid(), threading.get_ident()); self.used = False


def prepare_tick(service, actions):
    """Narrow pre-tick branch; calls only original kernels, zero extra fits."""
    from .v7_service import ForwardService
    found = _bound(service.lifecycle)
    if found is None:
        return None
    if type(service) is not ForwardService:
        raise TypeError('Exact original service required')
    policy, permit = found
    from .forward_service_scope import check_service_work
    from .nominal_algorithm_execution import _write_pair
    from factorlab.protocol import fingerprint
    lc = service.lifecycle
    with lc.controller_lock():
        check_service_work(service.runtime)
        prefix = _live_prefix(lc)
        index = service.state['steps']-1
        due = policy.design.identity['full_schedule']['due_at']
        if (type(index) is not int or not 0 <= index < len(due)
                or prefix['logical_at'] != due[index]):
            raise ValueError('Actual tick skipped or relabelled an original nominal opportunity')
        roots = (permit.paths['root']/'algorithm_policy_ticks',
                 permit.registry.directory/'algorithm_policy_ticks'/permit.trial['trial_id'])
        name = f"{prefix['hour']:05d}-before.json"
        if any((p/name).exists() for p in roots):
            raise ValueError('Original tick phase already attempted; no endpoint replay')
        _write_pair(roots, name, {'policy_digest': policy.digest, 'prefix': prefix,
                                 'phase': 'settle_then_original_endpoints'})
        token = _ENDPOINT.set((lc, os.getpid(), threading.get_ident(), prefix))
        try:
            bars = service.runtime._snapshot(lc.now())
            service.runtime._settle_available(bars)
            check_service_work(service.runtime)
            service._freeze_due(lc.now(), actions)
            service._endpoints(lc.now(), actions)
            if _live_prefix(lc) != prefix:
                raise ValueError('Original prefix changed during endpoint phase')
        finally:
            _ENDPOINT.reset(token)
        return _BeforeTick(_KEY, lc, policy, prefix, roots)


def after_tick(handle, service):
    if handle is None:
        return
    if (type(handle) is not _BeforeTick or handle.used or handle.lifecycle is not service.lifecycle
            or handle.owner != (os.getpid(), threading.get_ident())):
        raise ValueError('Foreign or reused new-policy tick handle')
    handle.used = True
    if _live_prefix(service.lifecycle) != handle.prefix:
        raise ValueError('Prefix or physical deadline changed during forecast')
    from .nominal_algorithm_execution import _write_pair
    _write_pair(handle.roots, f"{handle.prefix['hour']:05d}-after.json",
                {'policy_digest': handle.policy.digest, 'prefix': handle.prefix,
                 'phase': 'original_tick_completed_provisional_until_original_native_terminal'})
    if _live_prefix(service.lifecycle) != handle.prefix:
        raise ValueError('Physical prediction bound expired while writing tick result')


def confirmation_start(lifecycle, v_end, legacy_start):
    found = _bound(lifecycle)
    if found is None:
        return legacy_start
    active = _ENDPOINT.get()
    if (active is None or active[:3] != (lifecycle, os.getpid(), threading.get_ident())
            or _live_prefix(lifecycle) != active[3]):
        raise ValueError('New C reservation must occur in original pre-tick endpoint phase')
    from .v7_lifecycle import stamp
    start = stamp(active[3]['logical_at'])
    if not stamp(v_end) <= start <= stamp(v_end)+timedelta(hours=24):
        raise ValueError('Original C activation horizon exceeded')
    return start  # Current exact original origin, NEVER an earlier physical timestamp.


def guard_forecast(lifecycle):
    if _bound(lifecycle) is not None:
        _live_prefix(lifecycle)


@dataclass(frozen=True)
class OriginComparison:
    origin: str
    symbol: str
    nominal_model: str | None
    actual_model: str | None
    nominal_recipe: str | None
    actual_recipe: str | None
    nominal_cutoff: str | None
    actual_cutoff: str | None
    nominal_prediction: float | None
    actual_prediction: float | None
    nominal_revision: str | None
    actual_revision: str | None
    nominal_unavailable_reason: str | None
    nominal_target: float | None
    actual_target: float | None
    nominal_window: str | None
    actual_window: str | None
    nominal_input_digest: str | None
    actual_input_digest: str | None
    nominal_rng_digest: str | None
    actual_rng_digest: str | None
    nominal_statistic_digest: str | None
    actual_statistic_digest: str | None


def compare_complete_support(support, rows):
    """Pure conformance arithmetic, never authenticated cold-path evidence.

    Only structural absence codes can match None here. Caller rows cannot grant
    their own production status; paid nominal cold replay/model/numeric-failure
    readers still must authenticate them before this helper is used for truth.
    """
    if type(support) is not tuple or not support or type(rows) is not tuple:
        raise TypeError('Entire immutable support and comparison tuple required')
    if len(set(support)) != len(support):
        raise ValueError('Duplicate support')
    expected = set(support); found = {}
    for row in rows:
        if type(row) is not OriginComparison or (row.origin, row.symbol) not in expected:
            raise ValueError('Foreign comparison row')
        key = row.origin, row.symbol
        if key in found:
            raise ValueError('Duplicate comparison row')
        found[key] = row
    failures = len(expected-set(found))
    for row in found.values():
        if row.nominal_unavailable_reason is not None:
            matched = (row.nominal_unavailable_reason in {'stream_not_yet_eligible', 'no_prior_fit_opportunity'}
                       and all(getattr(row, f'{role}_{field}') is None
                               for role in ('nominal', 'actual')
                               for field in ('model', 'recipe', 'cutoff', 'prediction', 'revision', 'target',
                                             'window', 'input_digest', 'rng_digest', 'statistic_digest')))
        else:
            matched = all(getattr(row, 'nominal_'+field) == getattr(row, 'actual_'+field)
                          and getattr(row, 'nominal_'+field) is not None
                          for field in ('model', 'recipe', 'cutoff', 'prediction', 'revision', 'target',
                                        'window', 'input_digest', 'rng_digest', 'statistic_digest'))
            matched = matched and all(type(v) in (int, float) and math.isfinite(v) and v >= 0
                                      for v in (row.nominal_prediction, row.actual_prediction,
                                                row.nominal_target, row.actual_target))
        failures += not matched
    return {'expected': len(support), 'compared': len(found), 'mismatches_or_missing': failures,
            'possible_error': int(failures != 0), 'power_credit': 0,
            'scope': 'unprivileged_numeric_comparison_not_path_authentication',
            'formal_acceptance': False}
