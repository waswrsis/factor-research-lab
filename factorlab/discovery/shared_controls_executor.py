"""New actual-wallet control protocol; legacy three-free-wallet runs stay separate."""
import json
from dataclasses import asdict

import pandas as pd
from filelock import FileLock

from factorlab.protocol import fingerprint, project_path

from .monthly_work_authority import DEFAULT_LIMITS, MonthlyWorkAuthority
from .neural_forward_ticket import (
    _LifecycleReader,
    _verify_authority_binding,
    open_existing_authority,
)
from .s_outer_context import SOuterContext, _read, check_clock, original_clock
from .s_preparation import SPreparationEnvelope
from .scheduled_search import FrozenSWorkSchedule
from .v7_run import V7Config, monthly_budget, run_month
from .v7_storage import immutable_record
from .work_lease import FrozenWorkLease

ARMS = ('local', 'random', 'gpt')


def _write(path, value):
    immutable_record(path, {'value': value, 'sha256': fingerprint(value)})


def inspect_shared_authority_controls(directory):
    """Readback only; no fresh outer clock, wallet, reservation or model fit."""
    from .v7_run import inspect_run
    directory = project_path(directory)
    identity = _read(directory / 'shared_controls_identity.json')
    root = project_path(identity['root'])
    authority = open_existing_authority(identity['authority']['directory'])
    if (identity != _read(root / 'identity.json') or str(directory) != identity['directory']
            or authority.manifest_hash != identity['authority']['manifest_hash']):
        raise ValueError('Shared controls identity differs from its original wallet')
    outcomes = {}
    source = identity['source_reference']
    for index, arm in enumerate(ARMS):
        path = root / f'{index}-outcome.json'
        mirror = directory / f'{arm}-outcome.json'
        if path.exists() != mirror.exists():
            raise ValueError('Incomplete control outcome publication')
        if not path.exists():
            continue
        result = _read(path)
        if result != _read(mirror) or result['identity_hash'] != fingerprint(identity) or result['arm'] != arm:
            raise ValueError('Control outcome differs from frozen allocation')
        if result['status'] == 'complete':
            body = identity['schedules'][arm]
            arm_directory = directory / 'arms' / arm
            manifest = json.loads((arm_directory / 'manifest.json').read_text('utf-8'))['identity']
            expected = {'config': body['config'], 'rolling_spec': identity['spec'],
                        'exposure_path': source['exposure_path'], 'feed_identity': source['feed_identity'],
                        'mode': 'authority_scheduled_research_v1', 'work_contract': {
                            'schedule': body, 'authority_manifest_hash': authority.manifest_hash}}
            if any(fingerprint(manifest[k]) != fingerprint(v) for k, v in expected.items()):
                raise ValueError('Control monthly manifest differs from original allocation/source')
            saved = inspect_run(arm_directory)
            if (result['summary'] not in saved['summaries']
                    or fingerprint(result['summary']) != result['summary_hash']):
                raise ValueError('Control completion lacks its actual monthly result')
        outcomes[arm] = result
    return {'identity': identity, 'outcomes': outcomes, 'formal_claims_enabled': False,
            'serving_eligible': False, 'readback_only': True}


class SharedControlsExecutor:
    """No implicit allocation, quota borrowing, replacement wallet or arm retries."""
    def __init__(self, directory, *, authority, schedules, feed_directory, exposure_path, cutoff, spec):
        outer = original_clock(7200)
        if type(authority) is not MonthlyWorkAuthority or set(schedules) != set(ARMS):
            raise TypeError('Three explicit schedules and actual shared authority required')
        from .scheduled_search import validate_schedule_authority
        for schedule in schedules.values():
            validate_schedule_authority(authority, schedule)
        directory = project_path(directory)
        feed = project_path(feed_directory)
        exposure = project_path(exposure_path)
        feed_identity = json.loads((feed / 'identity.json').read_text())['source_id']
        cutoff = pd.Timestamp(cutoff)
        _,cutoff = spec.window(cutoff)
        source = {'feed_identity': feed_identity, 'feed_directory': str(feed),
                  'exposure_path': str(exposure), 'cutoff': cutoff.tz_convert('UTC').isoformat()}
        reader = _LifecycleReader(exposure.parent)
        try:
            _verify_authority_binding(reader, authority, reader.verify()['operations'])
            if reader.plan.source_id != feed_identity:
                raise ValueError('Controls source differs from actual lifecycle')
            if reader.plan.training_window_days != spec.window_days:
                raise ValueError('Controls changed the original training-window rule')
        finally:
            reader.close()
        bodies = {arm: FrozenSWorkSchedule.from_dict(schedules[arm].to_dict()).to_dict() for arm in ARMS}
        totals = dict.fromkeys(DEFAULT_LIMITS, 0)
        seconds = 0
        canonical = None
        holds = None
        for arm, body in bodies.items():
            config = V7Config(**{**body['config'], 'symbols': tuple(body['config']['symbols'])})
            if (config.budget_policy != 'controls_third' or config.search_mode != arm
                    or body['limits'] != monthly_budget(config, cutoff.strftime('%Y-%m')).limits):
                raise ValueError('Each control must retain its original one-third scientific rules')
            common = {**body['config'], 'search_mode': 'local'}
            if canonical is not None and fingerprint(common) != fingerprint(canonical):
                raise ValueError('Controls require identical fitting/search rules except proposal source')
            canonical = common
            allocation = body['allocation']
            proposed = {k: allocation[k] for k in ('forward_solve_holdback', 'nonlinear_solve_holdback',
                       'forward_seconds_holdback', 'nonlinear_seconds_holdback')}
            if (holds is not None and proposed != holds or allocation.get('unused_credit_transfer') is not False
                    or allocation.get('S_solve') != body['allocated_counts']['solve']
                    or any(type(v) is not int or v < (1 if k.startswith('forward') else 0)
                           for k, v in proposed.items())):
                raise ValueError('Controls require one common explicit holdback and no transfer')
            holds = proposed
            seconds += body['reserved_seconds']
            for k in totals:
                totals[k] += body['allocated_counts'][k]
        if (any(totals[k] > authority.limits[k] for k in totals)
                or totals['solve'] + holds['forward_solve_holdback'] + holds['nonlinear_solve_holdback'] > authority.limits['solve']
                or seconds + holds['forward_seconds_holdback'] + holds['nonlinear_seconds_holdback'] > authority.manifest['monthly_seconds']):
            raise ValueError('Combined S arms and forward/nonlinear holdbacks exceed the single wallet')
        key = fingerprint({'directory': str(directory), 'cutoff': source['cutoff']})
        root = authority.directory / 'shared_S_controls' / key
        self.authority, self.directory, self.root = authority, directory, root
        self.identity = {'version': 'v7-shared-authority-controls-v2', 'directory': str(directory),
                         'root': str(root), 'authority': {'directory': str(authority.directory),
                         'manifest_hash': authority.manifest_hash}, 'source_reference': source,
                         'spec': asdict(spec), 'schedules': bodies, 'clock': outer,
                         'allocated_counts': totals, 'reserved_seconds': seconds, 'holdbacks': holds,
                         'unused_credit_transfer': False, 'arm_order': list(ARMS)}
        with FileLock(str(root) + '.lock', timeout=0):
            if root.exists() or (directory / 'shared_controls_identity.json').exists():
                raise ValueError('Shared controls are readback only; never reset an outer clock')
            for path in (root / 'identity.json', directory / 'shared_controls_identity.json'):
                _write(path, self.identity)
        self._next = 0

    @staticmethod
    def verify_active(directory, arm):
        directory = project_path(directory)
        identity = _read(directory / 'shared_controls_identity.json')
        root = project_path(identity['root'])
        authority = open_existing_authority(identity['authority']['directory'])
        if (identity != _read(root / 'identity.json') or identity['directory'] != str(directory)
                or authority.manifest_hash != identity['authority']['manifest_hash'] or arm not in ARMS):
            raise ValueError('Shared controls identity changed')
        index = ARMS.index(arm)
        claim = _read(root / f'{index}-claim.json')
        if (claim != _read(directory / f'{arm}-claim.json')
                or claim != {'identity_hash': fingerprint(identity), 'arm': arm, 'index': index}
                or (root / f'{index}-outcome.json').exists()):
            raise ValueError('Control arm is not its original active claim')
        for previous in range(index):
            result = _read(root / f'{previous}-outcome.json')
            if result != _read(directory / f'{ARMS[previous]}-outcome.json') or result['status'] != 'complete':
                raise ValueError('Control cannot skip or retry an earlier failed arm')
        check_clock(identity['clock'])
        schedule = identity['schedules'][arm]
        return {'clock': identity['clock'], 'authority': identity['authority'],
                'directory': str(directory / 'arms' / arm), 'schedule_hash': fingerprint(schedule),
                'config': schedule['config'], 'spec': identity['spec'],
                'source_reference': identity['source_reference'], 'scope_hash': fingerprint(identity)}

    def run_file(self, path, *, backend):
        """Read the original file only after that arm's original baseline prepay."""
        outcomes = {}
        from .rolling import RollingSpec
        for index, arm in enumerate(ARMS):
            check_clock(self.identity['clock'])
            if index != self._next:
                raise ValueError('No shared control arm replay')
            claim = {'identity_hash': fingerprint(self.identity), 'arm': arm, 'index': index}
            for target in (self.root / f'{index}-claim.json', self.directory / f'{arm}-claim.json'):
                _write(target, claim)
            self._next += 1
            body = self.identity['schedules'][arm]
            config = V7Config(**{**body['config'], 'symbols': tuple(body['config']['symbols'])})
            spec = RollingSpec(**self.identity['spec'])
            source = self.identity['source_reference']
            preparation = None
            try:
                preparation = SPreparationEnvelope.prepare(self.authority, self.directory / 'arms' / arm,
                    source['cutoff'][:7], config=asdict(config), spec=asdict(spec), source_reference=source,
                    schedule_hash=fingerprint(body), clock_scope='before_local_source_load',
                    lease=FrozenWorkLease.issue(300), outer_context=SOuterContext.controls(self.directory, arm))
                bars = pd.read_parquet(project_path(path))
                summary = run_month(self.directory / 'arms' / arm, bars, source['cutoff'], config=config,
                    spec=spec, backend=backend if arm == 'gpt' else None,
                    exposure_path=source['exposure_path'], feed_directory=source['feed_directory'],
                    authority=self.authority, schedule=FrozenSWorkSchedule.from_dict(body), preparation=preparation)
                check_clock(self.identity['clock'])
                result = {'status': 'complete', 'arm': arm, 'identity_hash': fingerprint(self.identity),
                          'summary_hash': fingerprint(summary), 'summary': summary}
            except BaseException as error:
                if preparation is not None:
                    try:
                        record = preparation.baseline_record()
                        if record['status'] in {'reserved', 'running'}:
                            self.authority.fail_consumed(record['reservation_id'], 'control_failed:' + type(error).__name__)
                    except BaseException as accounting_error:  # noqa: BLE001 -- preserve original failure.
                        error.add_note('Original failure accounting: ' + type(accounting_error).__name__)
                result = {'status': 'failed_consumed', 'arm': arm, 'identity_hash': fingerprint(self.identity),
                          'error_type': type(error).__name__}
                for target in (self.root / f'{index}-outcome.json', self.directory / f'{arm}-outcome.json'):
                    _write(target, result)
                raise
            for target in (self.root / f'{index}-outcome.json', self.directory / f'{arm}-outcome.json'):
                _write(target, result)
            outcomes[arm] = result
        return {'version': 'v7-shared-authority-controls-result-v2', 'arms': outcomes,
                'identity_hash': fingerprint(self.identity), 'formal_claims_enabled': False,
                'serving_eligible': False, 'hard_deadline_enforced': False}
