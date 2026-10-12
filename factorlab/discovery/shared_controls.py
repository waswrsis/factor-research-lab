"""Three S controls within one prepaid, nontransferable monthly envelope.

The original v7_controls experiment retains its three full-budget semantics.
This protocol splits the total before outcomes: 32 AST/8 F/2 P/533 solves per
arm. One leftover solve is retired. Cooperative deadlines are not OS containment.
"""
from __future__ import annotations

import math
import time
import uuid
from dataclasses import asdict, replace

import pandas as pd
from filelock import FileLock

from factorlab.protocol import fingerprint, project_path

from .evidence import snapshot_source
from .joint_pool import FrozenPool
from .local_mutation import MonthlyBudget
from .rolling import RollingSpec
from .v7_controls import (
    ARMS,
    PASS_RULE,
    _check_source,
    _data_manifest,
    _metered_backend,
    _read,
    _readout,
    _write,
)
from .v7_run import V7Config, monthly_budget, run_month, source_files
from .work_lease import _boot_id


def _allocation_boot_valid(value):
    if not isinstance(value, str):
        return False
    if value.startswith('linux:'):
        try:
            return value == 'linux:' + str(uuid.UUID(value[6:]))
        except ValueError:
            return False
    prefix = 'windows:kuser-boot-v1:'
    if value.startswith(prefix):
        token = value[len(prefix):]
        return (token.isascii() and token.isdigit() and str(int(token)) == token
                and 0 < int(token) <= 0xffffffff)
    return False


def control_config(arm='local', symbols=('BTCUSDT',)):
    if arm not in ARMS:
        raise ValueError('Unknown shared control arm')
    return V7Config(symbols=tuple(symbols), evaluation_mode='joint', search_mode=arm,
                    profile_candidates=2, research_seconds=2400, budget_policy='controls_third')


def _now():
    return pd.Timestamp.now(tz='UTC')


def _spec(value=None):
    if isinstance(value, dict):
        value = {**value, 'event_bins': tuple(value['event_bins'])}
        value = RollingSpec(**value)
    value = value or RollingSpec()
    if not isinstance(value, RollingSpec) or value != replace(
            RollingSpec(), availability_delay_seconds=value.availability_delay_seconds,
            window_days=value.window_days):
        raise ValueError('Shared controls freeze every fitting rule except explicit source availability delay')
    return value


def _arm_readout(directory, config, month, spec):
    return (_readout(directory, config, month) if spec == RollingSpec()
            else _readout(directory, config, month, spec=spec))


def _totals(outcomes, month):
    total = {key: 0 for key in MonthlyBudget(month).counts}
    for arm, outcome in outcomes.items():
        budget = outcome['readout'].get('actual_budget')
        if budget is None:
            continue
        value = MonthlyBudget.from_dict(budget)
        expected = monthly_budget(control_config(arm), month)
        if value.month_id != month or value.limits != expected.limits:
            raise ValueError('Shared control arm used a different budget envelope')
        for key, count in value.counts.items():
            total[key] += count
    limits = MonthlyBudget(month).limits
    if any(count > limits[key] for key, count in total.items()):
        raise ValueError('Shared monthly total exceeds its frozen limit')
    return total


def _report_value(identity, allocation, outcomes):
    month = identity['month']
    result = {'version': 'v7-shared-controls-report-v1',
              'identity_hash': fingerprint(identity), 'allocation_hash': fingerprint(allocation),
              'arms': outcomes, 'actual_total_counts': _totals(outcomes, month),
              'reserved_total_counts': {'gpt_attempt': 6, 'family': 3, 'ast': 96, 'f': 24, 'solve': 1599},
              'retired_solve_credit': 1, 'V_queries': 0, 'C_queries': 0,
              'formal_claims_enabled': False, 'wall_containment': 'cooperative_python_checks_only',
              'completed_budget_comparison': all(o['failure'] is None and o['readout']['status'] == 'complete'
                                                 for o in outcomes.values()),
              'scope': 'shared_monthly_total_nontransferable_exposed_S_controls',
              'interpretation': 'Actual counts differ; a passing S formula is not independent alpha or GPT superiority.'}
    return result


def _report(directory, identity, allocation, outcomes):
    result = _report_value(identity, allocation, outcomes)
    _write(directory / 'report.json', result)
    return result


def inspect_shared_controls(directory):
    """Validate saved source, allocation and every completed arm without new work."""
    directory = project_path(directory)
    identity = _read(directory / 'manifest.json')
    allocation = _read(directory / 'allocation.json')
    if identity['version'] != 'v7-shared-controls-v2':
        raise ValueError('Unknown shared control protocol')
    _check_source(directory, identity['source_files'])
    month = identity['month']
    config = control_config(symbols=identity['symbols'])
    spec = _spec(identity['rolling_spec'])
    if (identity['arms'] != list(ARMS) or identity['config'] != json_config(config)
            or identity['pass_rule'] != PASS_RULE
            or identity['credit_policy'] != 'three_prepaid_nontransferable_thirds_one_solve_retired'):
        raise ValueError('Shared control identity changed the fixed comparison protocol')
    expected = {'identity_hash': fingerprint(identity), 'envelopes': {
        arm: {'budget': monthly_budget(config, month).to_dict(), 'profile_structures': 2,
              'research_seconds': 2400} for arm in ARMS}}
    if (set(allocation) != {'version', 'identity_hash', 'envelopes', 'started_at',
                           'started_monotonic', 'deadline_monotonic', 'host_boot_id', 'deadline'}
            or allocation.get('version') != 'v7-shared-controls-allocation-v2'
            or not _allocation_boot_valid(allocation.get('host_boot_id'))
            or any(allocation.get(key) != value for key, value in expected.items())):
        raise ValueError('Shared control prepaid allocation differs from protocol')
    started, end = pd.Timestamp(allocation['started_at']), pd.Timestamp(allocation['deadline'])
    mono_start, mono_end = allocation['started_monotonic'], allocation['deadline_monotonic']
    if (started.tzinfo is None or end.tzinfo is None
            or end - started != pd.Timedelta(seconds=7200)
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                   for v in (mono_start, mono_end))
            or mono_end - mono_start != 7200):
        raise ValueError('Shared controls cannot extend their original wall budget')
    outcomes = {}
    for arm in ARMS:
        path = directory / f'{arm}-outcome.json'
        if path.exists():
            saved = _read(path)
            readout = _arm_readout(directory / 'arms' / arm, control_config(arm, identity['symbols']), month, spec)
            if saved['readout'] != readout or saved['arm'] != arm:
                raise ValueError('Completed shared arm differs from its frozen outcome')
            completed = pd.Timestamp(saved['completed_at'])
            completed_mono, effective_mono = saved['completed_monotonic'], saved['effective_deadline_monotonic']
            if (completed.tzinfo is None or completed < started
                    or any(type(v) not in (int, float) or not math.isfinite(v)
                           for v in (completed_mono, effective_mono))
                    or not mono_start <= effective_mono <= mono_end
                    or completed_mono < mono_start
                    or (saved['failure'] is None and (completed > end or completed_mono >= effective_mono))):
                raise ValueError('Shared arm completion time differs from original wall envelope')
            outcomes[arm] = saved
    report_path = directory / 'report.json'
    report = _read(report_path) if report_path.exists() else None
    if report and (len(outcomes) != 3 or report != _report_value(identity, allocation, outcomes)):
        raise ValueError('Shared control report does not bind its actual arm ledgers')
    return {'identity': identity, 'allocation': allocation, 'outcomes': outcomes,
            'actual_total_counts': _totals(outcomes, month), 'report': report,
            'outcomes_closed': len(outcomes) == 3,
            'completed_budget_comparison': bool(report and report['completed_budget_comparison']),
            'formal_claims_enabled': False}


def json_config(config):
    # Receipts canonicalize tuples as JSON arrays.
    import json
    return json.loads(json.dumps(asdict(config)))


def run_shared_controls(directory, bars, cutoff, *, backend, symbols=('BTCUSDT',), spec=None):
    """One fixed local/random/GPT comparison; failures and interruptions cost credit."""
    if not callable(backend):
        raise TypeError('Shared GPT controls require an explicit backend')
    spec = _spec(spec)
    _, cutoff = spec.window(cutoff)
    month = cutoff.strftime('%Y-%m')
    config = control_config(symbols=symbols)
    spec = _spec(spec)
    directory = project_path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    identity = {'version': 'v7-shared-controls-v2', 'month': month, 'cutoff': cutoff.isoformat(),
                'arms': list(ARMS), 'symbols': list(symbols), 'config': asdict(config),
                'rolling_spec': asdict(spec),
                'data': _data_manifest(bars, cutoff, symbols), 'source_files': source_files(),
                'pass_rule': PASS_RULE, 'backend_type': f'{type(backend).__module__}.{type(backend).__qualname__}',
                'initial_reference_pool_hash': fingerprint(FrozenPool().to_dict()),
                'credit_policy': 'three_prepaid_nontransferable_thirds_one_solve_retired'}
    with FileLock(str(directory / '.shared-controls.lock'), timeout=0):
        manifest = directory / 'manifest.json'
        fresh = not manifest.exists()
        if not manifest.exists():
            if any(directory.iterdir()) and any(p.name != '.shared-controls.lock' for p in directory.iterdir()):
                raise ValueError('New shared controls cannot inherit a prior experiment')
            archived = snapshot_source(directory / 'source')
            if archived['files'] != identity['source_files']:
                raise ValueError('Source changed while freezing shared controls')
        _write(manifest, identity)
        allocation_path = directory / 'allocation.json'
        if not allocation_path.exists() and not fresh:
            raise ValueError('Shared allocation receipt missing; original wall budget cannot reset')
        if not allocation_path.exists():
            started = _now()
            started_monotonic = time.monotonic()
            _write(allocation_path, {'version': 'v7-shared-controls-allocation-v2',
                'identity_hash': fingerprint(identity),
                'envelopes': {arm: {'budget': monthly_budget(config, month).to_dict(),
                                    'profile_structures': 2, 'research_seconds': 2400} for arm in ARMS},
                'started_at': started.isoformat(),
                'started_monotonic': started_monotonic,
                'deadline_monotonic': started_monotonic + 7200,
                'host_boot_id': _boot_id(),
                'deadline': (started + pd.Timedelta(seconds=7200)).isoformat()})
        inspected = inspect_shared_controls(directory)
        if inspected['report'] is not None:
            return inspected['report']
        allocation, outcomes = inspected['allocation'], inspected['outcomes']
        observed_now = _now()
        if observed_now < pd.Timestamp(allocation['started_at']):
            raise ValueError('Shared controls wall clock moved backward; fail closed')
        if (_boot_id() != allocation['host_boot_id']
                or time.monotonic() < allocation['started_monotonic']):
            raise ValueError('Shared controls host clock identity changed; no fresh lease')
        deadline = min(allocation['deadline_monotonic'], time.monotonic() + max(
            0., (pd.Timestamp(allocation['deadline']) - observed_now).total_seconds()))
        for arm in ARMS:
            if arm in outcomes:
                continue
            arm_directory = directory / 'arms' / arm
            arm_directory.mkdir(parents=True, exist_ok=True)
            invocation = len(list(arm_directory.glob('invocation-*.json')))
            failure = None
            if invocation >= 3:
                failure = {'error_type': 'RecoveryCeiling', 'message': 'Same-run recovery exhausted'}
            elif time.monotonic() >= deadline:
                failure = {'error_type': 'SharedDeadline', 'message': 'Shared 7200-second envelope consumed'}
            else:
                _write(arm_directory / f'invocation-{invocation}.json',
                       {'arm': arm, 'invocation': invocation, 'allocation_hash': fingerprint(allocation)})
                try:
                    run_month(arm_directory, bars, cutoff, config=control_config(arm, symbols),
                              backend=_metered_backend(arm_directory, backend) if arm == 'gpt' else None,
                              external_deadline=deadline, **({'spec': spec} if spec != RollingSpec() else {}))
                except Exception as exc:  # noqa: BLE001 - retain failed arm, never replace it
                    failure = {'error_type': type(exc).__name__, 'message': str(exc)[:300]}
            _check_source(directory, identity['source_files'])
            readout = _arm_readout(arm_directory, control_config(arm, symbols), month, spec)
            if readout.get('pool_reference_hash', identity['initial_reference_pool_hash']) != identity['initial_reference_pool_hash']:
                raise ValueError('Shared arms must start from the same frozen pool')
            outcome = {'arm': arm, 'failure': failure, 'readout': readout,
                       'completed_at': _now().isoformat(), 'completed_monotonic': time.monotonic(),
                       'effective_deadline_monotonic': max(allocation['started_monotonic'], deadline)}
            if pd.Timestamp(outcome['completed_at']) < pd.Timestamp(allocation['started_at']):
                raise ValueError('Shared controls wall clock rolled back during research')
            if (pd.Timestamp(outcome['completed_at']) > pd.Timestamp(allocation['deadline'])
                    or outcome['completed_monotonic'] >= deadline):
                outcome['failure'] = {'error_type': 'SharedDeadline', 'message': 'Late result retained; not a completed-budget comparison'}
            _write(directory / f'{arm}-outcome.json', outcome)
            outcomes[arm] = outcome
        return _report(directory, identity, allocation, outcomes)
