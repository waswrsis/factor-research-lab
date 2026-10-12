"""Bounded production-path coverage evidence, NEVER a statistical certificate.

This adapter executes the real monthly controller on caller-supplied histories.
It records which paths ran without substituting a local simulator for adaptive
GPT, an OS supervisor, a forward clock, or the five-claim adoption controller.
Imported diagnostics retain their original scope. Missing evidence stays missing.
"""
from __future__ import annotations

import copy
import json
import os
import re
import sqlite3
from dataclasses import asdict, dataclass

import pandas as pd
from filelock import FileLock

from factorlab.protocol import fingerprint, project_path

from .evidence import file_hash, snapshot_source
from .exposure import ExposureLedger
from .local_mutation import MonthlyBudget
from .rolling import RollingSpec
from .v7_run import RunStore, V7Config, _verify_month_artifacts, run_month, source_files
from .v7_storage import immutable_record

OBLIGATIONS = {
    'monthly_controller': 'actual run_month, source/RunStore/S exposure and final artifacts verified',
    'adaptive_GPT_distribution': 'actual provider plus measured feedback adaptation across predeclared repetitions',
    'shared_monthly_controls': 'all arms jointly bounded by 96 AST / 24 F / 6 GPT / 1600 solves / 7200 s',
    'service_clock_and_coverage': 'actual service receipt clock, activation delays, gaps and nominal coverage',
    'five_claim_A0_controller': 'library, three factor claims and algorithm through actual frozen A0 controller',
    'independent_truth_validation': 'independent selected-learner null/boundary/alternative truth bridge',
    'joint_error_and_power': 'predeclared mixed-truth FWER upper and alternative-power lower confidence bounds',
    'shared_Linux_resources': 'actual shared cgroup CPU25/high640/max768/swap0/pids64 and deadlines',
}
CLAIMS = ('library_skill', 'factor1', 'factor2', 'factor3', 'algorithm')


def _read(path):
    value = json.loads(path.read_text('utf-8'))
    if set(value) != {'value', 'sha256'} or fingerprint(value['value']) != value['sha256']:
        raise ValueError('Coverage evidence integrity mismatch')
    return value['value']


def _write(path, value):
    immutable_record(path, {'value': value, 'sha256': fingerprint(value)})


def _actual_month_accounting(directory, identity, state, month):
    """Read original wallet/bridge/schedule evidence, without grants or repair.

    Do not reconstruct a controller with its checkpoint callback here: that
    constructor can repair publications. Inspection never creates clocks,
    reopens work, refreshes leases or copies absent mirror files.
    """
    from .monthly_work_authority import DEFAULT_LIMITS
    from .neural_forward_ticket import open_existing_authority
    from .s_preparation import SPreparationEnvelope
    from .scheduled_search import FrozenSWorkSchedule, ScheduledSearch
    from .work_authority_bridge import WorkAuthorityBridge
    from .work_authority_bridge import _read as bridge_read
    contract = identity['work_contract']
    anchor = bridge_read(directory / 'work_authority' / month / 'identity.json')
    authority = open_existing_authority(anchor['authority_directory'])
    schedule = FrozenSWorkSchedule.from_dict(contract['schedule'])
    preparation = SPreparationEnvelope.load(authority, directory, month)
    bridge = WorkAuthorityBridge.__new__(WorkAuthorityBridge)
    bridge.authority, bridge.month, bridge.state = authority, month, copy.deepcopy(state)
    bridge.identifier = fingerprint({'directory': str(directory), 'replay_month': month})
    bridge.directory = directory / 'work_authority' / month
    bridge.anchor = authority.directory / 'search_bridges' / bridge.identifier
    bridge.preparation, bridge.identity = preparation, anchor
    expected = {'bridge_id': bridge.identifier, 'run_directory': str(directory),
                'run_identity_hash': fingerprint(identity), 'replay_month': month,
                'data_id': state['data_id'], 'search_exposure': state['search_exposure'],
                'authority_directory': str(authority.directory),
                'authority_manifest_hash': authority.manifest_hash,
                'source_hash': fingerprint(identity['source_files']),
                'preparation_hash': preparation.digest}
    if (contract['authority_manifest_hash'] != authority.manifest_hash
            or bridge_read(bridge.anchor / 'identity.json') != anchor
            or any(anchor.get(k) != v for k, v in expected.items())):
        raise ValueError('Completed S accounting differs from its original authority')
    store = RunStore.__new__(RunStore)
    store.directory, store.identity = directory, identity
    envelope = preparation.verified_bridge_envelope(store, month, state)
    if anchor['execution_envelope'] != envelope or preparation.identity['work_schedule_hash'] != schedule.digest:
        raise ValueError('Completed S changed its original clock or work schedule')
    bridge._reload(repair=False)
    if (bridge._budget.to_dict() != state['budget']
            or bridge._sequence != state['authority_bridge']['steps']
            or bridge._head != state['authority_bridge']['head']
            or bridge.actual_counts != state['authority_bridge']['actual_counts']):
        raise ValueError('Completed S counters differ from actual prepaid usage')
    reader = ScheduledSearch.__new__(ScheduledSearch)
    reader.schedule = schedule
    reader._root, reader._mirror = bridge.anchor / 'schedule', bridge.directory / 'schedule'
    reader._identity = {'schedule': schedule.to_dict(), 'schedule_hash': schedule.digest,
                        'bridge_identity_hash': fingerprint(anchor)}
    closed = reader._read_cursor()
    if closed != len(schedule.to_dict()['slots']):
        raise ValueError('Completed S must retain all original fixed-slot outcomes')
    reservations = bridge._reservations()
    referenced = set()
    for index, slot in enumerate(schedule.to_dict()['slots']):
        wrapper = json.loads((reader._root / 'results' / f'{index:03d}.json').read_text('utf-8'))
        result = wrapper['record']['result']
        if result['status'] in {'completed', 'quota_consumed'}:
            record = reservations.get(result['reservation_id'])
            expected_status = 'completed' if result['status'] == 'completed' else 'failed_consumed'
            expected_terminal = 'complete' if result['status'] == 'completed' else 'failed_consumed'
            counts = {k: slot['counts'].get(k, 0) for k in DEFAULT_LIMITS}
            if (record is None or record['status'] != expected_status
                    or record['counts'] != counts or record['lease']['lease_seconds'] != slot['seconds']
                    or bridge._works[record['reservation_id']]['terminal'] != expected_terminal
                    or record['reservation_id'] in referenced):
                raise ValueError('Fixed-slot result does not match its original prepaid work')
            if slot['stage'] != 'baseline' and any(record['binding']['work'].get(k) != v for k, v in {
                    'schedule_hash': schedule.digest, 'slot': index, 'stage': slot['stage']}.items()):
                raise ValueError('Completed S charged another fixed work slot')
            referenced.add(record['reservation_id'])
        elif result['status'] not in {'unfunded_fixed_slot', 'unused_nontransferable'}:
            raise ValueError('Unknown or incomplete fixed-slot result')
    if referenced != set(reservations):
        raise ValueError('Completed S has unreported original prepaid work')
    return {'authority_directory': str(authority.directory), 'authority_manifest_hash': authority.manifest_hash,
            'preparation_digest': preparation.digest, 'schedule_digest': schedule.digest,
            'bridge_identity_hash': fingerprint(anchor), 'actual_counts': bridge.actual_counts,
            'reservation_ids': sorted(reservations), 'closed_slots': closed,
            'readback_only': True, 'hard_deadline_enforced': False}


@dataclass(frozen=True)
class CoveragePlan:
    """Explicit finite grid; additional trials require a new design, not retries."""
    cutoffs: tuple[str, ...]
    trials: tuple[str, ...] = ('trial-0',)
    input_scope: str = 'synthetic_engineering'
    provider_context: str = 'explicit_uncertified'

    def __post_init__(self):
        if (not 1 <= len(self.cutoffs) <= 12 or not 1 <= len(self.trials) <= 16
                or len(self.cutoffs) * len(self.trials) > 128
                or len(set(self.trials)) != len(self.trials)
                or any(not re.fullmatch(r'[a-zA-Z0-9_-]{1,40}', t) for t in self.trials)
                or self.input_scope not in {'synthetic_engineering', 'exposed_history', 'live_receipts'}
                or self.provider_context not in {'explicit_uncertified', 'fresh_codex_per_trial_v1'}):
            raise ValueError('Invalid bounded calibration coverage grid')
        dates = [pd.Timestamp(t) for t in self.cutoffs]
        if (any(pd.isna(t) or t.tzinfo is None or t.utcoffset().total_seconds() != 0
                or t.day != 1 or t != t.normalize() for t in dates)
                or dates != sorted(set(dates))):
            raise ValueError('Coverage cutoffs must be unique ordered UTC month boundaries')


def _monthly_evidence(directory, month):
    """Read-only actual RunStore and shared ledger verification; no JSON credit."""
    manifest = json.loads((directory / 'manifest.json').read_text('utf-8'))
    identity = manifest['identity']
    if manifest['hash'] != fingerprint(identity) or identity['source_files'] != source_files():
        raise ValueError('Production monthly source binding changed')
    store = RunStore.__new__(RunStore)
    store.directory, store.identity = directory, identity
    store.db = sqlite3.connect((directory / 'state.sqlite').as_uri() + '?mode=ro', uri=True)
    try:
        store.verify()
        state = store.get(month)
        if not state or state['status'] != 'complete':
            raise ValueError('Production month lacks a committed completion')
        _verify_month_artifacts(directory, month, state)
        summary = json.loads((directory / 'months' / f'{month}.json').read_text('utf-8'))
        if summary != state['summary'] or summary['budget'] != state['budget']:
            raise ValueError('Published summary differs from authoritative final budget')
        MonthlyBudget.from_dict(state['budget'])
        context = state['search_exposure']
        ledger = ExposureLedger.__new__(ExposureLedger)
        ledger.path = project_path(identity['exposure_path'])
        ledger.clock = lambda: pd.Timestamp.now(tz='UTC')
        ledger.db = sqlite3.connect(ledger.path.as_uri() + '?mode=ro', uri=True)
        try:
            ledger.assert_prompt_search(**{k: v for k, v in context.items() if k != 'data_id'})
        finally:
            ledger.close()
        accounting = None
        if identity.get('mode') == 'authority_scheduled_research_v1':
            accounting = _actual_month_accounting(directory, identity, state, month)
        return {'manifest_hash': file_hash(directory / 'manifest.json'),
                'state_hash': fingerprint(state), 'summary_hash': fingerprint(summary),
                'data_id': state['data_id'], 'search_exposure': context,
                'budget': state['budget'], 'cutoff': state['cutoff'],
                'A0': summary['A0'], 'formal_confirmation_enabled': summary['formal_confirmation_enabled'],
                'agent_audit_hash': fingerprint(state['agent_audit']),
                'provider_attempt_records': copy.deepcopy(state['agent_audit']),
                'agent_attempts': len(state['agent_audit']), 'search_mode': identity['config']['search_mode'],
                'shared_accounting': accounting}
    finally:
        store.db.close()


class CoverageHarness:
    """Reservation precedes work; interrupted units cannot be silently rerun.

    A failed unit is evidence of incomplete execution. This adapter deliberately
    does not mint a new run after failure. Its filesystem receipt plus independent
    append-only journal detect missing-unit deletions; torn writes fail closed.
    """
    def __init__(self, directory, plan, *, config=None, spec=None):
        self.directory = project_path(directory)
        self.plan = CoveragePlan(tuple(plan.cutoffs), tuple(plan.trials), plan.input_scope, plan.provider_context)
        self.config = config or V7Config(evaluation_mode='joint', formulas_per_family=1)
        self.spec = spec or RollingSpec()
        if self.config.evaluation_mode != 'joint' or self.spec.task != 'volatility':
            raise ValueError('Coverage adapter requires the actual joint volatility controller')
        if (self.plan.provider_context == 'fresh_codex_per_trial_v1'
                and (self.config.search_mode != 'gpt' or self.plan.input_scope != 'live_receipts')):
            raise ValueError('Isolated Codex trials require GPT and the actual shared receipt controller')
        self.identity = json.loads(json.dumps({'version': 'v7-calibration-coverage-v3',
            'plan': asdict(self.plan), 'config': asdict(self.config), 'rolling': asdict(self.spec),
            'sources': source_files(), 'obligations': OBLIGATIONS, 'claims': CLAIMS,
            'adoption_policy': 'diagnostics_only_never_authorizes_formal_gate',
            'budget_scope': 'explicit_shared_wallet_if_receipts_else_legacy_offline_diagnostic'}))
        self.directory.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.directory / '.lock')):
            manifest = self.directory / 'identity.json'
            if not manifest.exists():
                if any(p.name != '.lock' for p in self.directory.iterdir()):
                    raise ValueError('Missing coverage identity; cannot reset consumed work')
                archived = snapshot_source(self.directory / 'source')
                if archived['files'] != self.identity['sources']:
                    raise ValueError('Source changed during coverage freeze')
            _write(manifest, self.identity)
            self._verify()

    def _verify(self):
        if _read(self.directory / 'identity.json') != self.identity or source_files() != self.identity['sources']:
            raise ValueError('Frozen coverage contract/source changed')
        for name, digest in self.identity['sources'].items():
            if file_hash(self.directory / 'source' / name) != digest:
                raise ValueError('Coverage source archive changed')
        records, previous = {}, '0' * 64
        path = self.directory / 'reservations.jsonl'
        if path.exists():
            for sequence, line in enumerate(path.read_text('utf-8').splitlines()):
                wrapper = json.loads(line)
                row = wrapper['value']
                if (wrapper['sha256'] != fingerprint(row) or row['previous'] != previous
                        or row['sequence'] != sequence or row['key'] in records):
                    raise ValueError('Coverage reservation journal changed')
                records[row['key']] = row['reservation']
                previous = wrapper['sha256']
        disk = {p.name for p in (self.directory / 'units').glob('*') if p.is_dir()}
        reserved_disk = {p.parent.name for p in (self.directory / 'units').glob('*/reservation.json')}
        trials_disk = {p.name for p in (self.directory / 'trials').glob('*') if p.is_dir()}
        if self.plan.provider_context == 'fresh_codex_per_trial_v1':
            providers_disk = {p.name for p in (self.directory / 'providers').glob('*') if p.is_dir()}
            if not providers_disk <= {r['trial'] for r in records.values()}:
                raise ValueError('Provider context exists outside its original trial reservation')
        if (disk != set(records) or reserved_disk != set(records)
                or not trials_disk <= {r['trial'] for r in records.values()}):
            raise ValueError('Coverage journal/unit mismatch; consumed reservations cannot reset')
        if len(records) > len(self.plan.trials) * len(self.plan.cutoffs):
            raise ValueError('Coverage reservation count exceeds its frozen grid')
        for key, value in records.items():
            if (value['trial'] not in self.plan.trials
                    or pd.Timestamp(value['cutoff']) not in map(pd.Timestamp, self.plan.cutoffs)
                    or key != fingerprint({'trial': value['trial'], 'cutoff': value['cutoff']})
                    or value['contract_hash'] != fingerprint(self.identity)
                    or value['run_directory'] != str(self.directory / 'trials' / value['trial'] / 'research')):
                raise ValueError('Coverage reservation lies outside its frozen grid/path/contract')
            if _read(self.directory / 'units' / key / 'reservation.json') != value:
                raise ValueError('Coverage reservation differs from its journal')
        return records, previous

    def _month_evidence(self, run_directory, month, trial):
        evidence = _monthly_evidence(run_directory, month)
        if self.plan.provider_context == 'fresh_codex_per_trial_v1':
            from .calibration_trial_context import TrialScopedCodexBackend
            evidence['trial_context'] = TrialScopedCodexBackend.inspect_existing(
                self.directory, trial, fingerprint(self.identity), through_month=month)
        return evidence

    def run_month(self, trial, bars, cutoff, *, backend=None, exposure_path=None, feed_directory=None,
                  authority=None, schedule=None, preparation=None):
        cutoff = pd.Timestamp(cutoff).isoformat()
        if trial not in self.plan.trials or pd.Timestamp(cutoff) not in map(pd.Timestamp, self.plan.cutoffs):
            raise ValueError('Unplanned calibration trial/month')
        if self.plan.input_scope == 'live_receipts' and (not feed_directory or not exposure_path):
            raise ValueError('Live calibration needs its actual archive/shared exposure paths')
        isolated = self.plan.provider_context == 'fresh_codex_per_trial_v1'
        if isolated and backend is not None:
            raise ValueError('Isolated Codex trials do not accept a reusable injected backend')
        if self.config.search_mode == 'gpt' and backend is None and not isolated:
            raise ValueError('GPT calibration requires an explicitly supplied backend')
        shared = None
        if authority is not None or schedule is not None or preparation is not None:
            from .monthly_work_authority import MonthlyWorkAuthority
            from .scheduled_search import FrozenSWorkSchedule
            if type(authority) is not MonthlyWorkAuthority or type(schedule) is not FrozenSWorkSchedule:
                raise TypeError('Actual shared authority and fixed work schedule required')
            authority.verify()
            if not feed_directory or not exposure_path:
                raise ValueError('Shared calibration needs the actual feed and exposure')
            shared = {'directory': str(authority.directory), 'manifest_hash': authority.manifest_hash,
                      'schedule': schedule.to_dict()}
        if self.plan.input_scope == 'live_receipts' and shared is None:
            raise ValueError('Live receipt calibration requires the shared authority and schedule')
        if isolated:
            from .calibration_trial_context import trial_paths, verify_memory_namespace
            paths = trial_paths(self.directory, trial)
            if (project_path(feed_directory) != paths['feed']
                    or project_path(exposure_path) != paths['exposure']
                    or authority.directory != paths['authority']):
                raise ValueError('Independent trial must own its feed, exposure and original wallet namespace')
            verify_memory_namespace(paths['research'])
        from .v7_controls import _data_manifest
        data = _data_manifest(bars, pd.Timestamp(cutoff), self.config.symbols)
        key = fingerprint({'trial': trial, 'cutoff': cutoff})
        run_directory = self.directory / 'trials' / trial / 'research'
        provider_kind = ('trial_scoped_codex_uncertified' if isolated else
                         ('none' if backend is None else 'explicit_backend_uncertified'))
        reservation = {'trial': trial, 'cutoff': cutoff, 'data': data,
            'exposure_path': str(project_path(exposure_path)) if exposure_path else None,
            'feed_directory': str(project_path(feed_directory)) if feed_directory else None,
            'provider_kind': provider_kind, 'run_directory': str(run_directory),
            'contract_hash': fingerprint(self.identity), 'shared_work_binding': shared}
        with FileLock(str(self.directory / '.lock'), timeout=0):
            records, head = self._verify()
            unit = self.directory / 'units' / key
            control_path = self.directory / 'controls' / f'{key}.json'
            if control_path.exists():
                linked = _read(control_path)
                actual = inspect_shared_controls_coverage(linked['directory'])
                if linked['evidence'] != actual or actual['data'] != data:
                    raise ValueError('Controls use different data from the calibration month')
            if key in records:
                if records[key] != reservation:
                    raise ValueError('Frozen calibration input/run binding changed')
                if not (unit / 'outcome.json').exists():
                    # A committed real month can be recovered without another
                    # invocation, fit, proposal or charge. Incomplete work stays
                    # consumed; never rerun the controller to fill this gap.
                    proof = unit / 'verified_month.json'
                    if proof.exists():
                        evidence = self._month_evidence(run_directory, cutoff[:7], trial)
                        if _read(proof) != {'reservation_hash': fingerprint(reservation), 'evidence': evidence}:
                            raise ValueError('Coverage recovery differs from its verified call evidence')
                        outcome = {'status': 'complete', 'key': key, 'formal_gate_eligible': False,
                                   'evidence': evidence, 'recovery': 'verified_existing_completion_no_work'}
                        _write(unit / 'outcome.json', outcome)
                        return outcome
                    return {'status': 'interrupted_consumed', 'key': key, 'formal_gate_eligible': False}
                outcome = _read(unit / 'outcome.json')
                if outcome['status'] == 'complete' and self._month_evidence(run_directory, cutoff[:7], trial) != outcome['evidence']:
                    raise ValueError('Coverage monthly evidence changed after completion')
                return outcome
            earlier = [r['cutoff'] for r in records.values() if r['trial'] == trial]
            if earlier and cutoff <= max(earlier):
                raise ValueError('Trial months must execute chronologically without replacement')
            event = {'sequence': len(records), 'previous': head, 'key': key, 'reservation': reservation}
            with (self.directory / 'reservations.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'value': event, 'sha256': fingerprint(event)}, sort_keys=True) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            _write(unit / 'reservation.json', reservation)
            outcome = {'status': 'failed_consumed', 'key': key, 'formal_gate_eligible': False}
            try:
                if isolated:
                    from .calibration_trial_context import TrialScopedCodexBackend
                    backend = TrialScopedCodexBackend(self.directory, trial, fingerprint(self.identity))
                shared_kwargs = {} if shared is None else {
                    'authority': authority, 'schedule': schedule, 'preparation': preparation}
                run_month(run_directory, bars, cutoff, config=self.config, spec=self.spec,
                    backend=backend, exposure_path=exposure_path, feed_directory=feed_directory,
                    **shared_kwargs)
                self._verify()
                evidence = self._month_evidence(run_directory, cutoff[:7], trial)
                _write(unit / 'verified_month.json',
                       {'reservation_hash': fingerprint(reservation), 'evidence': evidence})
                outcome.update(status='complete', evidence=evidence)
            except BaseException as exc:
                outcome.update(error_type=type(exc).__name__, error=str(exc)[:300])
                _write(unit / 'outcome.json', outcome)
                raise
            _write(unit / 'outcome.json', outcome)
            return outcome

    def attach_shared_controls(self, trial, directory):
        """Bind the new prepaid-total wrapper, never the old three-wallet study."""
        with FileLock(str(self.directory / '.lock'), timeout=0):
            records, _ = self._verify()
            evidence = inspect_shared_controls_coverage(directory)
            cutoff = evidence['cutoff']
            if trial not in self.plan.trials or pd.Timestamp(cutoff) not in map(pd.Timestamp, self.plan.cutoffs):
                raise ValueError('Controls are outside the predeclared coverage trial/month')
            key = fingerprint({'trial': trial, 'cutoff': cutoff})
            if key in records and records[key]['data'] != evidence['data']:
                raise ValueError('Controls use different data from the calibration month')
            value = {'trial': trial, 'directory': str(project_path(directory)), 'evidence': evidence}
            _write(self.directory / 'controls' / f'{key}.json', value)
            return evidence

    def report(self):
        with FileLock(str(self.directory / '.lock'), timeout=0):
            records, _ = self._verify()
            units = []
            for key, reservation in records.items():
                path = self.directory / 'units' / key / 'outcome.json'
                row = _read(path) if path.exists() else {'status': 'interrupted_consumed', 'key': key}
                if row['status'] == 'complete':
                    actual = self._month_evidence(project_path(reservation['run_directory']), reservation['cutoff'][:7], reservation['trial'])
                    if actual != row['evidence']:
                        raise ValueError('Completed coverage evidence changed')
                units.append(row)
            exercised = ['monthly_controller'] if any(u['status'] == 'complete' for u in units) else []
            controls = []
            for path in sorted((self.directory / 'controls').glob('*.json')):
                value = _read(path)
                actual = inspect_shared_controls_coverage(value['directory'])
                key = fingerprint({'trial': value['trial'], 'cutoff': actual['cutoff']})
                if (path.stem != key or value['evidence'] != actual or value['trial'] not in self.plan.trials
                        or pd.Timestamp(actual['cutoff']) not in map(pd.Timestamp, self.plan.cutoffs)
                        or (key in records and records[key]['data'] != actual['data'])):
                    raise ValueError('Bound shared controls evidence changed')
                controls.append(actual)
            if any(c['completed_budget_comparison'] for c in controls):
                exercised.append('shared_monthly_controls')
            return {'contract_hash': fingerprint(self.identity), 'input_scope': self.plan.input_scope,
                'planned_units': len(self.plan.trials) * len(self.plan.cutoffs), 'reserved_units': len(units),
                'units': units, 'controls': controls, 'exercised_paths': exercised,
                'missing_requirements': {k: v for k, v in OBLIGATIONS.items() if k not in exercised},
                'claims': list(CLAIMS), 'formal_gate_eligible': False, 'full_V7': False,
                'statistical_certificate': None,
                'reason': 'Path execution alone does not establish calibrated error, power, or production scope'}


def inspect_shared_controls_coverage(directory):
    from .shared_controls import inspect_shared_controls
    verified = inspect_shared_controls(directory)
    if verified['report'] is None or not verified['outcomes_closed']:
        raise ValueError('Controls need all original arm outcomes, including failures')
    identity, allocation = verified['identity'], verified['allocation']
    return {'kind': 'prepaid_shared_monthly_controls', 'cutoff': identity['cutoff'],
            'data': copy.deepcopy(identity['data']), 'identity_hash': fingerprint(identity),
            'allocation_hash': fingerprint(allocation), 'report_hash': fingerprint(verified['report']),
            'envelopes': copy.deepcopy(allocation['envelopes']),
            'actual_total_counts': copy.deepcopy(verified['actual_total_counts']),
            'completed_budget_comparison': verified['completed_budget_comparison'],
            'formal_gate_eligible': False, 'full_V7': False,
            'provider_authenticity_certified': False,
            'scope': 'execution_and_shared_budget_only_not_GPT_distribution_or_statistical_power'}


def inspect_local_diagnostic(directory):
    """Verify the existing local five-claim diagnostic and preserve its exclusions."""
    from .full_pipeline_calibration import inspect_full_pipeline_calibration
    report = inspect_full_pipeline_calibration(directory)
    if report.get('full_V7') is not False or report.get('formal_gate_eligible') is not False:
        raise ValueError('Local diagnostic cannot claim production calibration')
    return {'kind': 'local_policy_diagnostic', 'report_hash': fingerprint(report),
            'scope': report['scope'], 'excluded_scope': copy.deepcopy(report['excluded_scope']),
            'formal_gate_eligible': False, 'full_V7': False, 'production_coverage_credit': []}
