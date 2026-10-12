"""Staged complete-algorithm shadow validation, never factor or formal admission.

Actual S provenance, prospective permissions and prepaid cold ensemble work use
their own producer, shadow admission and single-look endpoints. Qualifier work
is limited to the sole V winner; neither stage grants serving or an alpha slot.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sqlite3
import sys
from contextlib import closing
from dataclasses import asdict, dataclass

import pandas as pd
from filelock import FileLock

from factorlab.protocol import fingerprint, project_path

from .algorithm_shortlist import AlgorithmCatalogue, _copy, _read, _write
from .evidence import file_hash
from .forward_fit_snapshot import prepare_forward_snapshot
from .joint_pool import FrozenPool
from .monthly_work_authority import DEFAULT_LIMITS, MonthlyWorkAuthority, source_files
from .neural_forward import FrozenNeuralRecipe
from .neural_forward_ticket import (
    _bind_authority,
    _LifecycleReader,
    _snapshot_from_actual_source,
    _verify_authority_binding,
    runtime_identity,
)
from .v7_forward import ForwardRuntime
from .v7_lifecycle import boundary, stamp
from .work_lease import FrozenWorkLease

VERSION = 'v7-typed-algorithm-validation-v2'
TICKET_VERSION = 'v7-algorithm-validation-fit-v2'


def _now():
    return pd.Timestamp.now(tz='UTC')


def _elapsed_schedule(lifecycle):
    if lifecycle.plan.training_window_days is None:
        return None
    from .rolling_cadence import RollingCadence
    return {'cadence_hash': RollingCadence().digest,
            'anchor': lifecycle._cadence_anchor().isoformat(),
            'training_days': 90, 'refit_days': 15}


def _elapsed_prediction_interval(lifecycle, cutoff, start, end):
    """Intersect an original fit cycle with the unchanged protected endpoint."""
    cutoff = lifecycle.fit_cutoff(cutoff)
    start, end = stamp(start), stamp(end)
    begin = max(start, cutoff)
    finish = min(end, cutoff + pd.Timedelta(days=15))
    if begin >= finish:
        raise ValueError('Fit differs from the frozen full algorithm V schedule')
    return begin.isoformat(), finish.isoformat()


def observation_now(runtime):
    from .quarantined_observation import logical_now
    logical = logical_now(runtime.lifecycle)
    return _now() if logical is None else logical


def _implementation():
    from pathlib import Path
    paths = (Path(__file__).resolve(), Path(__file__).with_name('algorithm_validation_producer.py').resolve(),
        Path(__file__).with_name('v7_service.py').resolve())
    return {str(p): file_hash(p) for p in paths}


def _producer():
    name = 'factorlab.discovery.algorithm_validation_producer'
    if name not in sys.modules:
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name('algorithm_validation_producer.py'))
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[name] = loaded
        spec.loader.exec_module(loaded)
    return sys.modules[name]


def _paired_selection(rows, candidate_ids, *, start, end, symbols, rule):
    """Private endpoint arithmetic, NOT an evidence acceptance API.

    The dedicated producer obtains observations from its immutable actual-origin
    forecast/settlement ledger after durably consuming its sole endpoint. There
    is no public API for importing this helper's result as evidence.
    """
    start, end = stamp(start), stamp(end)
    if (set(rows) != {*candidate_ids, 'K0', 'Pv'} or not candidate_ids
            or len(set(candidate_ids)) != len(candidate_ids) or not start < end
            or start != start.floor('h') or end != end.floor('h')
            or not symbols or len(set(symbols)) != len(symbols)):
        raise ValueError('Incomplete fixed algorithm endpoint/universe')
    tables = {}
    for name, values in rows.items():
        table = {}
        for row in values:
            if set(row) != {'hour', 'symbol', 'loss', 'label_identity'}:
                raise ValueError('Unknown paired algorithm observation')
            hour = stamp(row['hour'])
            key = (hour.isoformat(), row['symbol'])
            if (key in table or hour != hour.floor('h') or not start <= hour < end
                    or row['symbol'] not in symbols or type(row['loss']) not in (int, float)
                    or not math.isfinite(row['loss']) or row['loss'] < 0
                    or not isinstance(row['label_identity'], str) or not row['label_identity']):
                raise ValueError('Invalid paired algorithm observation')
            table[key] = row
        tables[name] = table
    common = set.intersection(*(set(t) for t in tables.values()))
    for key in common:
        if len({table[key]['label_identity'] for table in tables.values()}) != 1:
            raise ValueError('Algorithm streams use different actual label revisions')
    nominal_hours = int((end-start)/pd.Timedelta(hours=1))
    hours = sorted({hour for hour, _ in common})
    losses = {}
    for name, table in tables.items():
        by_hour = {hour: [] for hour in hours}
        for key in sorted(common):
            by_hour[key[0]].append(table[key]['loss'])
        losses[name] = (sum(sum(v)/len(v) for v in by_hour.values())/len(hours)) if hours else None
    coverage, hour_coverage = len(common)/(nominal_hours*len(symbols)), len(hours)/nominal_hours
    scores = []
    for key in candidate_ids:
        gains = {ref: 1-losses[key]/losses[ref] if losses.get(ref) and losses[key] is not None else None
                 for ref in ('K0', 'Pv')}
        eligible = (min(coverage, hour_coverage) >= rule.minimum_coverage
                    and all(g is not None and g > rule.minimum_gain for g in gains.values()))
        scores.append({'pipeline_id': key, 'gains': gains, 'loss': losses[key], 'eligible': eligible})
    winners = sorted((s for s in scores if s['eligible']),
        key=lambda s: (-min(s['gains'].values()), s['loss'], s['pipeline_id']))
    return {'records': scores, 'selected': winners[0]['pipeline_id'] if winners else None,
        'coverage': coverage, 'hour_coverage': hour_coverage, 'nominal_hours': nominal_hours,
        'nominal_symbol_hours': nominal_hours*len(symbols), 'common_pairs': len(common),
        'common_hash': fingerprint(sorted(common)), 'formal_claim': False, 'serving_eligible': False}


@dataclass(frozen=True)
class AlgorithmValidationRule:
    minimum_coverage: float = .95
    minimum_gain: float = .005

    def __post_init__(self):
        if (type(self.minimum_coverage) not in (float, int)
                or not 0 < self.minimum_coverage <= 1
                or type(self.minimum_gain) not in (float, int)
                or not 0 <= self.minimum_gain < 1):
            raise ValueError('Invalid algorithm validation engineering rule')

    def to_dict(self):
        return {**asdict(self), 'V_months': 1, 'qualifier_months': 3,
            'qualifier_starts': 'month_after_V_end', 'scoring_grace_hours': 36,
            'selection': 'one_fixed_endpoint_actual_ensemble_common_hours_vs_both_K0_Pv',
            'coverage': 'original_nominal_hours_and_symbol_hours_both_required',
            'tie': ['descending_min_K0_Pv_gain', 'ascending_ensemble_loss', 'pipeline_id'],
            'maximum_selected': 1, 'feedback_to_originating_search': 'forbidden',
            'qualifier': 'descriptive_internal_validation_not_confirmation',
            'formal_claim': False, 'automatic_adoption': False, 'serving_eligible': False}


def _actual_shortlist(catalogue):
    if type(catalogue) is not AlgorithmCatalogue:
        raise TypeError('Actual completed algorithm catalogue required')
    # Do not silently create a shortlist while freezing future permission.
    from .structural_queue_review import checked_shortlist
    scoped, cached = checked_shortlist(catalogue)
    if scoped and cached is not None:
        return cached
    old = _read(catalogue.directory / 'shortlist.json')
    current = catalogue.seal_shortlist()
    if old != current or not current['algorithm_shortlist'] or current['status'] != 'sealed':
        raise ValueError('An already sealed nonempty S shortlist is required')
    return current


def _common_work_counts(recipes):
    """All V algorithms receive the same finite counter/time ceiling."""
    budgets = []
    for value in recipes.values():
        recipe = FrozenNeuralRecipe.from_dict(value)
        counts = {k: recipe.budget_limits.get(k, 0) for k in DEFAULT_LIMITS}
        counts['solve'] += 1
        budgets.append(counts)
    if not budgets or any(b != budgets[0] for b in budgets):
        raise ValueError('Algorithm V candidates require identical prepaid fit/epoch/solve ceilings')
    return budgets[0]


def _runtime(runtime, authority):
    if type(runtime) is not ForwardRuntime or type(authority) is not MonthlyWorkAuthority:
        raise TypeError('Actual forward runtime and unique monthly authority required')
    authority.verify()
    runtime.lifecycle.verify()
    runtime.feed.verify()
    files = source_files()
    if (files != runtime.identity['source_files'] or files != authority.manifest['source_files']
            or runtime.feed.identity != runtime.lifecycle.plan.source_id
            or sorted(runtime.config.symbols) != authority.manifest['scope']['universe']
            or authority.manifest['scope']['task'] != 'volatility'):
        raise ValueError('Algorithm validation source, feed or authority scope changed')
    saved = json.loads((runtime.directory / 'forward.json').read_text('utf-8'))
    if saved != {'identity': _copy(runtime.identity), 'sha256': fingerprint(runtime.identity)}:
        raise ValueError('Actual forward runtime manifest changed')
    for name, expected in files.items():
        if file_hash(runtime.directory / 'source' / name) != expected:
            raise ValueError('Actual forward source archive changed')
    operations = [json.loads(t) for (t,) in runtime.lifecycle.db.execute('SELECT record FROM operations ORDER BY seq')]
    _verify_authority_binding(runtime.lifecycle, authority, operations)
    return operations


class AlgorithmValidation:
    """One non-resettable typed validation contract for one completed S catalogue.

    File hashes/local SQL are trusted-producer accounting, not administrator
    authentication. Public clocks or evidence callbacks are intentionally absent.
    A prepared fit is prepaid work, not an admitted model certificate.
    """
    @classmethod
    def freeze(cls, runtime, authority, catalogue, *, start, rule=None):
        if type(rule or AlgorithmValidationRule()) is not AlgorithmValidationRule:
            raise TypeError('Typed frozen validation rule required')
        rule = rule or AlgorithmValidationRule()
        shortlist = _actual_shortlist(catalogue)
        if authority.directory != project_path(catalogue.identity['authority_directory']):
            raise ValueError('Complete algorithm cannot borrow a second wallet')
        # Historical S is allowed, but it must use the same actual exposure ledger.
        from .neural_research_ticket import _reader
        reader = _reader(catalogue.identity['run_directory'])
        try:
            if reader.exposure.path != runtime.lifecycle.exposure.path:
                raise ValueError('S and future validation require the same exposure authority')
        finally:
            reader.close()
        _bind_authority(runtime, authority)
        operations = _runtime(runtime, authority)
        start = boundary(start)
        end = start + pd.DateOffset(months=1)
        q_start, q_end = end + pd.DateOffset(months=1), end + pd.DateOffset(months=4)
        recipes = dict(sorted((v['pipeline_id'], v['recipe']) for v in shortlist['algorithm_shortlist']))
        if any(FrozenNeuralRecipe.from_dict(r).rolling_spec.digest != runtime.spec.digest for r in recipes.values()):
            raise ValueError('Future runtime changes the full recipe rolling contract')
        frozen_recipe = FrozenNeuralRecipe.from_dict(next(iter(recipes.values())))
        work_counts = _common_work_counts(recipes)
        references = {'K0': runtime.recipe_for(FrozenPool()), 'Pv': runtime.recipe_for(frozen_recipe.pool)}
        directory = runtime.directory / 'algorithm_validation' / catalogue.digest
        anchor = authority.directory / 'algorithm_validations' / (catalogue.digest + '.json')
        static = {'version': VERSION, 'catalogue_directory': str(catalogue.directory),
            'catalogue_id': catalogue.digest, 'shortlist_hash': fingerprint(shortlist),
            'runtime_directory': str(runtime.directory), 'runtime_identity_hash': fingerprint(runtime.identity),
            'plan_id': runtime.lifecycle.plan.digest, 'source_hash': fingerprint(source_files()),
            'numeric_runtime': runtime_identity(),
            'implementation': _implementation(),
            'authority_directory': str(authority.directory), 'authority_manifest_hash': authority.manifest_hash,
            'source_id': runtime.feed.identity, 'symbols': list(runtime.config.symbols),
            'exposure_path': str(runtime.lifecycle.exposure.path), 'recipes': recipes,
            'work_counts': work_counts,
            'references': references,
            'start': start.isoformat(), 'V_end': end.isoformat(),
            'V_scoring_cutoff': (end + pd.Timedelta(hours=36)).isoformat(),
            'qualifier_start': q_start.isoformat(), 'qualifier_end': q_end.isoformat(),
            'qualifier_scoring_cutoff': (q_end + pd.Timedelta(hours=36)).isoformat(),
            'rule': rule.to_dict(), 'originating_S_batch': catalogue.identity['exposure']['batch'],
            'batch': 'algorithm-validation:' + catalogue.digest, 'anchor': str(anchor),
            'directory': str(directory), 'formal_claim': False, 'serving_eligible': False,
            'confirmation_slot': None, 'ordinary_factor_route': False,
            'producer': 'dedicated_typed_algorithm_shadow_v2'}
        schedule = _elapsed_schedule(runtime.lifecycle)
        if schedule is not None:
            static['fit_schedule'] = schedule
        with FileLock(str(directory) + '.lock', timeout=0):
            if directory.exists() or anchor.exists():
                old = cls(runtime, authority, catalogue)
                if any(old.identity[k] != v for k, v in static.items()):
                    raise ValueError('Frozen algorithm validation cannot change or restart')
                return old
            if any(o['kind'] == 'algorithm_validation_bound' and o['detail']['catalogue_id'] == catalogue.digest
                   for o in operations):
                raise ValueError('Bound algorithm validation was lost; cannot reset it')
            now = observation_now(runtime)
            from .quarantined_observation import current_binding
            prefix = current_binding(runtime.lifecycle)
            physical = now
            if prefix is not None:
                from .nonlinear_runtime import physical_bound_for_prefix
                physical = physical_bound_for_prefix(runtime.lifecycle, prefix)
            if (start < now or stamp(catalogue.identity['exposure']['end']) > now
                    or stamp(catalogue.identity['frozen_at']) > physical):
                raise ValueError('Algorithm validation must freeze prospectively after S')
            identity = {**static, 'frozen_at': now.isoformat()}
            _write(anchor, identity)
            _write(directory / 'identity.json', identity)
            # Intent precedes the two-ledger publication. Partial initialization
            # stays closed; it may not silently obtain new times or new work.
            with runtime.lifecycle.db:
                runtime.lifecycle._event('algorithm_validation_bound', {
                    'catalogue_id': catalogue.digest, 'directory': str(directory), 'identity_hash': fingerprint(identity)})
            with closing(sqlite3.connect(directory / 'validation.sqlite')) as db, db:
                db.execute('PRAGMA synchronous=FULL')
                db.execute('CREATE TABLE state (id INTEGER PRIMARY KEY CHECK(id=1), state TEXT, identity_hash TEXT)')
                db.execute('INSERT INTO state VALUES (1,?,?)', ('frozen', fingerprint(identity)))
                db.execute('CREATE TABLE events (seq INTEGER PRIMARY KEY, body TEXT, hash TEXT)')
            # Protect both future stages against original-batch GPT feedback.
            # Neither reservation touches confirmations or any of the five alpha slots.
            reference_id = references['Pv']['pipeline_id']
            for batch in (identity['batch'], identity['originating_S_batch']):
                runtime.lifecycle.exposure.reserve_validation(batch=batch, source=identity['source_id'],
                    symbols=identity['symbols'], start=identity['start'], end=identity['qualifier_end'],
                    candidate_ids=list(recipes), reference_id=reference_id, frozen_at=identity['frozen_at'])
        return cls(runtime, authority, catalogue)

    def __init__(self, runtime, authority, catalogue):
        self.runtime, self.authority, self.catalogue = runtime, authority, catalogue
        self.directory = runtime.directory / 'algorithm_validation' / catalogue.digest
        self.identity = _read(self.directory / 'identity.json')
        self.digest = fingerprint(self.identity)
        self.verify()

    def verify(self):
        value = self.identity
        operations = _runtime(self.runtime, self.authority)
        if (_read(self.directory / 'identity.json') != value or fingerprint(value) != self.digest
                or _read(project_path(value['anchor'])) != value
                or value['version'] != VERSION or project_path(value['directory']) != self.directory
                or value['authority_manifest_hash'] != self.authority.manifest_hash
                or value['source_hash'] != fingerprint(source_files())
                or value['numeric_runtime'] != runtime_identity()
                or value['implementation'] != _implementation()
                or value['runtime_identity_hash'] != fingerprint(self.runtime.identity)
                or value.get('fit_schedule') != _elapsed_schedule(self.runtime.lifecycle)
                or value['shortlist_hash'] != fingerprint(_actual_shortlist(self.catalogue))):
            raise ValueError('Typed algorithm validation identity changed')
        bindings = [o['detail'] for o in operations if o['kind'] == 'algorithm_validation_bound'
                    and o['detail']['catalogue_id'] == self.catalogue.digest]
        if bindings != [{'catalogue_id': self.catalogue.digest, 'directory': str(self.directory), 'identity_hash': self.digest}]:
            raise ValueError('Typed validation lost its immutable lifecycle binding')
        with closing(sqlite3.connect((self.directory / 'validation.sqlite').as_uri() + '?mode=ro', uri=True)) as db:
            if db.execute('SELECT * FROM state').fetchall() != [(1, 'frozen', self.digest)]:
                raise ValueError('Typed validation state cannot reset or self-certify completion')
        for batch in (value['batch'], value['originating_S_batch']):
            row = self.runtime.lifecycle.exposure.db.execute('SELECT identity FROM validations WHERE batch=?', (batch,)).fetchone()
            expected = {'batch': batch, 'source': value['source_id'], 'symbols': sorted(value['symbols']),
                'start': value['start'], 'end': value['qualifier_end'], 'candidate_ids': list(value['recipes']),
                'reference_id': value['references']['Pv']['pipeline_id'],
                'frozen_at': value['frozen_at']}
            if not row or json.loads(row[0]) != expected:
                raise ValueError('Typed algorithm protected exposure was lost or changed')
        return {'identity_hash': self.digest, 'status': 'frozen',
                'formal_claim': False, 'serving_eligible': False}

    def fit_plan(self, pipeline_id, cutoff, *, stage='V'):
        from .structural_queue_review import require_queue_completion
        require_queue_completion(self)
        self.verify()
        value, cutoff = self.identity, self.runtime.lifecycle.fit_cutoff(cutoff)
        elapsed = self.runtime.lifecycle.plan.training_window_days is not None
        if stage == 'V':
            allowed = elapsed or cutoff.isoformat() == value['start']
            prediction_start, prediction_end = value['start'], value['V_end']
        elif stage == 'qualifier':
            result = _producer().endpoint_result(self, 'V')
            allowed = (result is not None and result['score']['selected'] == pipeline_id
                and stamp(result['recorded_at']) < stamp(value['qualifier_start'])
                and (elapsed or stamp(value['qualifier_start']) <= cutoff < stamp(value['qualifier_end'])))
            prediction_start, prediction_end = cutoff.isoformat(), (cutoff+pd.DateOffset(months=1)).isoformat()
        else:
            raise ValueError('Unknown typed algorithm stage')
        if pipeline_id not in value['recipes'] or not allowed:
            raise ValueError('Fit differs from the frozen full algorithm V schedule')
        if elapsed:
            start, end = ((value['start'], value['V_end']) if stage == 'V' else
                          (value['qualifier_start'], value['qualifier_end']))
            prediction_start, prediction_end = _elapsed_prediction_interval(
                self.runtime.lifecycle, cutoff, start, end)
        recipe = FrozenNeuralRecipe.from_dict(value['recipes'][pipeline_id])
        if recipe.rolling_spec.digest != self.runtime.spec.digest:
            raise ValueError('Algorithm fit changed its original rolling training contract')
        counts = {k: recipe.budget_limits.get(k, 0) for k in DEFAULT_LIMITS}
        counts['solve'] += 1  # Independent same-cutoff K0, alongside model's Pv baseline.
        if counts != value['work_counts']:
            raise ValueError('Algorithm work changed its common frozen budget')
        return {'version': TICKET_VERSION, 'validation_id': self.digest, 'stage': stage,
            'cutoff': cutoff.isoformat(), 'train_start': recipe.rolling_spec.window(cutoff)[0].isoformat(),
            'prediction_start': prediction_start, 'prediction_end': prediction_end,
            'pipeline_id': pipeline_id, 'recipe': recipe.to_dict(), 'counts': counts,
            'lease_seconds': 300, 'source_id': value['source_id'], 'plan_id': value['plan_id'],
            'stream': value['batch'] + ':' + stage + ':' + pipeline_id, 'formal_claim': False,
            'serving_eligible': False, 'factor_claim_eligible': False,
            'producer': value['producer']}

    def prepare_fit(self, pipeline_id, cutoff, *, gates, stage='V'):
        from .structural_queue_review import require_queue_completion
        require_queue_completion(self)
        """Prepay and seal actual cold inputs for the dedicated shadow launcher.

        All candidates use the same fixed cutoff/limits. Each stable work key
        may execute once under its original lease. Partial preparation consumes
        its original prepaid work and cannot obtain a replacement lease.
        """
        plan = self.fit_plan(pipeline_id, cutoff, stage=stage)
        now = observation_now(self.runtime)
        if not stamp(plan['prediction_start']) <= now < stamp(plan['prediction_end']):
            raise ValueError('Algorithm V preparation is outside its real execution window')
        key = 'algorithm_fit:' + fingerprint({'validation': self.digest, 'pipeline': pipeline_id, 'cutoff': plan['cutoff'], 'stage': stage})
        # The complete stable work key remains in the authority; a collision in
        # this bounded path can only refuse preparation, never change its grant.
        directory = self.directory / 'fits' / fingerprint(key)[:16]
        with FileLock(str(directory) + '.lock', timeout=0):
            with self.authority._transaction() as db:
                prior = [r for r in self.authority._verify(db)['reservations'].values() if r['work_key'] == key]
            if prior or directory.exists():
                raise ValueError('Algorithm work already prepaid; only readback, never retry execution')
            from .quarantined_observation import current_binding
            allocation_prefix = current_binding(self.runtime.lifecycle)
            from .forward_service_scope import check_service_work
            check_service_work(self.runtime, new_work=True)
            lease = FrozenWorkLease.issue(300)
            check_service_work(self.runtime, lease=lease)
            if not stamp(plan['prediction_start']) <= observation_now(self.runtime) < stamp(plan['prediction_end']):
                raise ValueError('Original actual lease differs from the validation window')
            binding = {'kind': TICKET_VERSION, 'validation_id': self.digest,
                       'fit_plan_hash': fingerprint(plan), 'pipeline_id': pipeline_id, 'gates_hash': fingerprint(gates)}
            if self.authority.manifest.get('partition_policy') is not None:
                allocation = {'version': 'algorithm-work-allocation-v1', 'plan': _copy(plan),
                    'validation_directory': str(self.directory), 'binding': binding,
                    'work_key': key, 'lease': lease.to_dict(), 'observation_prefix': allocation_prefix}
                _write(directory/'allocation.json', allocation)
                _write(self.authority.directory/'algorithm_work_intents'/(fingerprint(key)+'.json'), allocation)
            record = self.authority._reserve_with_lease(key, counts=plan['counts'], binding=binding, lease=lease)
            try:
                _write(directory / 'preparation.json', {'reservation_id': record['reservation_id'], 'lease': record['lease']})
                recipe = FrozenNeuralRecipe.from_dict(self.identity['recipes'][pipeline_id])
                job = self.runtime.lifecycle.plan_fit(plan['cutoff'], plan['stream'], self.runtime.recipe_for(recipe))
                snapshot = prepare_forward_snapshot(self.runtime, job)
                _producer().verify_gates(gates, snapshot.data, recipe, plan,
                    forward_job={'lifecycle_directory': str(self.runtime.directory), 'job': job})
                from .nonlinear_launcher import save_worker_data
                payload = save_worker_data(directory / 'payload', snapshot.data, cutoff=plan['cutoff'])
                lease.assert_parent_alive()
                value = {'version': TICKET_VERSION, 'plan': plan, 'job': job,
                    'authority_directory': str(self.authority.directory),
                    'authority_manifest_hash': self.authority.manifest_hash,
                    'reservation_id': record['reservation_id'], 'terms_hash': record['terms_hash'],
                    'lease': record['lease'], 'lease_hash': record['lease_hash'],
                    'validation_directory': str(self.directory), 'gates': _copy(gates), 'data': payload,
                    'snapshot_ref': {'path': str(self.runtime.directory / 'snapshots' / (job + '.json')),
                                     'job': job, 'sha256': file_hash(self.runtime.directory / 'snapshots' / (job + '.json')),
                                     'record_hash': snapshot.verify()},
                    'shadow_only': True, 'serving_eligible': False, 'formal_claim': False}
                _write(directory / 'prepared.json', value)
                lease.assert_parent_alive()
                return _copy(value)
            except BaseException:
                self.authority.fail_consumed(record['reservation_id'], 'typed_algorithm_preparation_failed')
                raise

    def read_prepared(self, pipeline_id, cutoff=None, *, stage='V', statuses=('reserved',)):
        from .structural_queue_review import require_queue_completion
        require_queue_completion(self)
        """Historical verification only, without a new clock or retry permission."""
        if cutoff is None:
            cutoff = (self.runtime.lifecycle.fit_cycle_start(self.identity['start'])
                      if self.runtime.lifecycle.plan.training_window_days is not None else self.identity['start'])
        plan = self.fit_plan(pipeline_id, cutoff, stage=stage)
        key = 'algorithm_fit:' + fingerprint({'validation': self.digest, 'pipeline': pipeline_id, 'cutoff': plan['cutoff'], 'stage': stage})
        directory = self.directory / 'fits' / fingerprint(key)[:16]
        value = _read(directory / 'prepared.json')
        record = self.authority.get(value['reservation_id'])
        expected_binding = {'kind': TICKET_VERSION, 'validation_id': self.digest,
                            'fit_plan_hash': fingerprint(plan), 'pipeline_id': pipeline_id,
                            'gates_hash': fingerprint(value['gates'])}
        if (record['work_key'] != key or record['binding'] != expected_binding
                or record['counts'] != plan['counts'] or value['plan'] != _copy(plan)
                or value['version'] != TICKET_VERSION or value['shadow_only'] is not True
                or value['serving_eligible'] is not False or value['formal_claim'] is not False
                or any(value[k] != record[k] for k in ('lease', 'lease_hash', 'terms_hash',
                    'authority_directory', 'authority_manifest_hash'))
                or record['status'] not in statuses
                or _read(directory / 'preparation.json') != {'reservation_id': record['reservation_id'], 'lease': record['lease']}):
            raise ValueError('Prepared algorithm work differs from original canonical reservation')
        reader = _LifecycleReader(self.runtime.directory)
        try:
            snapshot, _, _ = _snapshot_from_actual_source(reader, value['snapshot_ref'],
                FrozenNeuralRecipe.from_dict(self.identity['recipes'][pipeline_id]))
            if fingerprint(snapshot) != value['snapshot_ref']['record_hash'] or snapshot['job'] != value['job']:
                raise ValueError('Prepared algorithm snapshot binding changed')
        finally:
            reader.close()
        return value

    def launch_fit(self, prepared):
        from .structural_queue_review import require_queue_completion
        require_queue_completion(self)
        return _producer().launch(self, prepared)

    def tick(self):
        from .structural_queue_review import require_queue_completion
        require_queue_completion(self)
        return _producer().tick(self)

    def select_validation(self):
        from .structural_queue_review import require_queue_completion
        require_queue_completion(self)
        return _producer().conclude(self, 'V')

    def conclude_qualifier(self):
        from .structural_queue_review import require_queue_completion
        require_queue_completion(self)
        return _producer().conclude(self, 'qualifier')

    @classmethod
    def discover(cls, runtime, authority):
        """Read the actual lifecycle anchors; directory scans cannot mint a job."""
        operations = [json.loads(t) for (t,) in runtime.lifecycle.db.execute('SELECT record FROM operations ORDER BY seq')]
        bindings = [o['detail'] for o in operations if o['kind'] == 'algorithm_validation_bound']
        if bindings and type(authority) is not MonthlyWorkAuthority:
            raise ValueError('Typed algorithm controller requires its canonical shared wallet')
        protocols = []
        for binding in bindings:
            identity = _read(project_path(binding['directory'])/'identity.json')
            catalogue = AlgorithmCatalogue(identity['catalogue_directory'])
            protocol = cls(runtime, authority, catalogue)
            if (str(protocol.directory) != binding['directory'] or protocol.digest != binding['identity_hash']
                    or protocol.catalogue.digest != binding['catalogue_id']):
                raise ValueError('Algorithm controller lost its original namespace binding')
            from .structural_queue_review import require_queue_completion
            require_queue_completion(protocol)
            protocols.append(protocol)
        if len({p.digest for p in protocols}) != len(protocols):
            raise ValueError('Duplicate algorithm controller binding')
        return protocols

    def owns_job(self, job):
        from .structural_queue_review import require_queue_completion
        require_queue_completion(self)
        """Exact readonly namespace, SQL, recipe and prepaid authority proof.

        A failed snapshot can still have a genuinely owned SQL job; requiring a
        completed ticket here would accidentally send it to the factor handler.
        A prefix without its canonical reservation is rejected, never skipped.
        """
        self.verify()
        row = self.runtime.lifecycle.db.execute('SELECT cutoff,stream,pipeline FROM jobs WHERE id=?', (job,)).fetchone()
        if not row:
            return None
        cutoff, stream, pipeline = row
        namespace = self.identity['batch']+':'
        if not stream.startswith(namespace):
            return None
        candidates = [(stage, pid) for stage in ('V', 'qualifier') for pid in self.identity['recipes']
            if stream == namespace+stage+':'+pid]
        if len(candidates) != 1:
            raise ValueError('Unregistered algorithm namespace job')
        stage, pid = candidates[0]
        plan = self.fit_plan(pid, cutoff, stage=stage)
        expected_job = fingerprint({'run': self.identity['plan_id'], 'cutoff': cutoff, 'stream': stream})
        recipe = FrozenNeuralRecipe.from_dict(plan['recipe'])
        if expected_job != job or json.loads(pipeline) != _copy(self.runtime.recipe_for(recipe)):
            raise ValueError('Algorithm job changed its exact SQL pipeline')
        directory = _producer()._fit_path(self, plan).parent
        preparation = _read(directory/'preparation.json')
        record = self.authority.get(preparation['reservation_id'])
        binding = record['binding']
        if (record['work_key'] != _producer()._key(plan) or record['counts'] != plan['counts']
                or preparation['lease'] != record['lease']
                or binding.get('kind') != TICKET_VERSION or binding.get('validation_id') != self.digest
                or binding.get('fit_plan_hash') != fingerprint(plan) or binding.get('pipeline_id') != pid
                or set(binding) != {'kind', 'validation_id', 'fit_plan_hash', 'pipeline_id', 'gates_hash'}):
            raise ValueError('Algorithm job lacks its exact canonical prepaid owner')
        return {'validation_id': self.digest, 'job': job, 'stage': stage,
            'pipeline_id': pid, 'reservation_id': record['reservation_id'], 'status': record['status']}

    def register_fit_gates(self, pipeline_id, cutoff, *, gates, stage='V'):
        from .structural_queue_review import require_queue_completion
        require_queue_completion(self)
        """Register real producer references, never a boolean/evidence callback.

        Each work's refs are immutable once registered; their actual contents
        are verified against the sealed cutoff data during prepaid preparation.
        """
        plan = self.fit_plan(pipeline_id, cutoff, stage=stage)
        legacy = (type(gates) is dict and set(gates) == {'directory', 'artifacts'}
                  and isinstance(gates['directory'], str) and isinstance(gates['artifacts'], dict))
        shared = (type(gates) is dict and set(gates) == {'shared_S'}
                  and type(gates['shared_S']) is dict and set(gates['shared_S']) == {'epoch', 'probes'})
        if not (legacy or shared):
            raise ValueError('Typed gate producer references required')
        producer = _producer()
        key = producer._key(plan)
        with FileLock(str(self.directory/'events.lock'), timeout=0):
            old = [e['payload'] for e in producer._events(self) if e['kind'] == 'fit_gates' and e['payload']['work_key'] == key]
            value = {'work_key': key, 'plan_hash': fingerprint(plan), 'gates': _copy(gates)}
            if old:
                if old != [value]:
                    raise ValueError('Algorithm gate references cannot be replaced')
                return value
            producer._append(self, 'fit_gates', value)
            return value

    def step(self, *, deadline_monotonic, max_fits=1):
        from .structural_queue_review import require_queue_completion
        require_queue_completion(self)
        return _producer().controller_step(self, deadline_monotonic=deadline_monotonic, max_fits=max_fits)

    def resolve_fit_gates(self):
        """Register exact selected S producer references for presently due work."""
        self.verify()
        producer = _producer()
        now = observation_now(self.runtime)
        from .quarantined_observation import current_binding
        prefix = current_binding(self.runtime.lifecycle)
        physical = now.normalize().replace(day=1)
        if prefix is not None:
            from .nonlinear_runtime import physical_bound_for_prefix
            physical = physical_bound_for_prefix(self.runtime.lifecycle, prefix)
        selected = {v['pipeline_id']: v for v in _actual_shortlist(self.catalogue)['algorithm_shortlist']}
        from .neural_research_launcher import verify_completed_research
        for stage in ('V', 'qualifier'):
            if stage == 'qualifier':
                winner = producer.endpoint_result(self, 'V')
                if not winner or not winner['score']['selected']:
                    continue
            start, end, candidates = producer._stage(self, stage)
            if not start <= now < end:
                continue
            for pid in candidates:
                plan = self.fit_plan(pid, self.runtime.lifecycle.fit_cycle_start(now), stage=stage)
                key = producer._key(plan)
                if any(e['kind'] == 'fit_gates' and e['payload']['work_key'] == key for e in producer._events(self)):
                    continue
                ref = selected[pid]['S_evidence']
                report = verify_completed_research(ref['path'])
                ticket = report['ticket']
                if (ticket.digest != ref['digest'] or ticket.value['recipe'] != self.identity['recipes'][pid]
                        or ticket.value['authority_manifest_hash'] != self.authority.manifest_hash
                        or ticket.value['context']['source_id'] != self.identity['source_id']):
                    raise ValueError('Selected typed learner lost its original S gate producer')
                if stamp(report['publication']['observation']['wall']) > physical:
                    continue
                original = ticket.value['gates']
                if set(original) != {'epoch', 'probes', 'resource'}:
                    raise ValueError('Unknown selected S gate schema')
                self.register_fit_gates(pid, plan['cutoff'], stage=stage,
                    gates={'shared_S': {k: original[k] for k in ('epoch', 'probes')}})

    def pending_work(self):
        from .structural_queue_review import require_queue_completion
        require_queue_completion(self)
        """Readonly exact due gate/work inventory for service attempt prepayment."""
        self.verify()
        producer = _producer()
        now, pending = observation_now(self.runtime), []
        with self.authority._transaction() as db:
            reserved = {r['work_key'] for r in self.authority._verify(db)['reservations'].values()}
        gates = {e['payload']['work_key'] for e in producer._events(self) if e['kind'] == 'fit_gates'}
        for stage in ('V', 'qualifier'):
            if stage == 'qualifier':
                result = producer.endpoint_result(self, 'V')
                if not result or not result['score']['selected'] or stamp(result['recorded_at']) >= stamp(self.identity['qualifier_start']):
                    continue
            start, end, candidates = producer._stage(self, stage)
            if start <= now < end:
                for pipeline in candidates:
                    plan = self.fit_plan(pipeline, self.runtime.lifecycle.fit_cycle_start(now), stage=stage)
                    key = producer._key(plan)
                    if key in gates and key not in reserved:
                        pending.append({'work_key': key, 'plan': plan})
        return pending
