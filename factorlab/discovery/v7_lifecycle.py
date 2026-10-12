"""Persistent V7 clocks: prospective S/V/C, immutable forecasts and activation.

Engineering operation is allowed without a statistical certificate. Formal C
and A0 adoption are fail-closed and require independently verified calibration.
This module never trains or scores an active protected period for GPT feedback.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from filelock import FileLock

from factorlab.protocol import fingerprint, project_path
from factorlab.time_utils import month_start

from .confirmatory import ConfirmationRule, paired_effect_interval
from .exposure import ExposureLedger, select_on_validation
from .model_registry import ForecastLedger, ModelRegistry
from .v7_storage import immutable_record


class ForecastExpired(ValueError):
    """A valid new forecast missed its unchanged prospective issue window."""


def stamp(value):
    result = pd.Timestamp(value)
    if result.tzinfo is None or pd.isna(result):
        raise ValueError('Lifecycle requires an aware UTC timestamp')
    return result.tz_convert('UTC')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def boundary(value):
    value = stamp(value)
    if value != month_start(value):
        raise ValueError('Calendar cutoff must be exactly the UTC month boundary')
    return value


@dataclass(frozen=True)
class LifecyclePlan:
    source_id: str
    symbols: tuple[str, ...] = ('BTCUSDT',)
    confirmation_months: int = 3
    algorithm_months: int = 6
    activation_delay_hours: int = 24
    prediction_max_lag_seconds: int = 300
    calibration_directory: str | None = None
    calibration_binding: str | None = None
    minimum_coverage: float = .95
    training_window_days: int | None = None
    refit_interval_days: int | None = None

    def __post_init__(self):
        periods = (self.training_window_days, self.refit_interval_days)
        if periods != (None, None) and (periods != (90, 15) or any(type(v) is not int for v in periods)):
            raise ValueError('Elapsed mode requires exactly 90-day training and 15-day refit')
        if (not isinstance(self.source_id, str) or not self.source_id
                or not self.symbols or len(set(self.symbols)) != len(self.symbols)
                or len(self.symbols) > 3 or any(not isinstance(s, str) or not s for s in self.symbols)
                or self.confirmation_months not in (3, 6, 9, 12)
                or self.algorithm_months not in (6, 9, 12)
                or self.activation_delay_hours != 24 or self.minimum_coverage != .95
                or type(self.prediction_max_lag_seconds) is not int
                or not 1 <= self.prediction_max_lag_seconds <= 300
                or bool(self.calibration_directory) != bool(self.calibration_binding)):
            raise ValueError('Invalid frozen lifecycle protocol')

    @property
    def digest(self):
        fields = asdict(self)
        if self.training_window_days is None:
            fields.pop('training_window_days'); fields.pop('refit_interval_days')
        version = 'v7-lifecycle-v1' if self.training_window_days is None else 'v7-lifecycle-elapsed90-refit15-v2'
        return fingerprint({'version': version, **fields})


class Lifecycle:
    """One serialized controller; transactional CAS activation and one score look.

    ``clock`` exists for tests. Production callers leave it unset. All output
    paths remain inside the configured project root on either operating system.
    Source bytes are frozen by the parent runtime before opening this store.
    """

    def __init__(self, directory, plan: LifecyclePlan, *, clock=None):
        self.directory = project_path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.plan = plan
        self.clock = clock or (lambda: pd.Timestamp.now(tz='UTC'))
        immutable_record(self.directory / 'lifecycle.json',
                         {'plan': asdict(plan), 'sha256': plan.digest})
        self.db = sqlite3.connect(self.directory / 'lifecycle.sqlite', timeout=30)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, cutoff TEXT, stream TEXT,
                pipeline TEXT, status TEXT, model TEXT, completed TEXT, activation TEXT, failure TEXT);
            CREATE TABLE IF NOT EXISTS active (stream TEXT PRIMARY KEY, job TEXT, generation INTEGER);
            CREATE TABLE IF NOT EXISTS batches (id TEXT PRIMARY KEY, ordinal INTEGER UNIQUE,
                frozen TEXT, identity TEXT, hash TEXT, status TEXT, selected TEXT, result TEXT);
            CREATE TABLE IF NOT EXISTS settlement (run TEXT, stream TEXT, origin TEXT, symbol TEXT,
                task TEXT, record TEXT, hash TEXT, PRIMARY KEY(run,stream,origin,symbol,task));
            CREATE TABLE IF NOT EXISTS operations (seq INTEGER PRIMARY KEY, record TEXT, hash TEXT);
        ''')
        self.forecasts = ForecastLedger(self.directory / 'forecasts.sqlite')
        self.registry = ModelRegistry(self.directory / 'models')
        self.exposure = ExposureLedger(self.directory / 'exposure.sqlite', clock=self.now)

    def close(self):
        self.forecasts.close()
        self.exposure.close()
        self.db.close()

    def now(self):
        from .quarantined_observation import logical_now
        logical = logical_now(self)
        return stamp(self.clock()) if logical is None else logical

    def _event(self, kind, detail):
        previous = self.db.execute('SELECT hash FROM operations ORDER BY seq DESC LIMIT 1').fetchone()
        sequence = self.db.execute('SELECT COALESCE(MAX(seq),0)+1 FROM operations').fetchone()[0]
        value = {'seq': sequence, 'previous': previous[0] if previous else '0' * 64,
                 'kind': kind, 'at': self.now().isoformat(), 'detail': detail}
        self.db.execute('INSERT INTO operations VALUES (?,?,?)',
                        (sequence, canonical(value), fingerprint(value)))

    def _cadence_anchor(self, *, create=False):
        from .rolling_cadence import RollingCadence
        if self.plan.training_window_days is None:
            raise ValueError('Calendar lifecycle has no elapsed cadence anchor')
        rule = RollingCadence()
        def original():
            return [json.loads(r[0]) for r in self.db.execute(
                'SELECT record FROM operations ORDER BY seq')
                if json.loads(r[0])['kind'] == 'fit_cadence_anchored']
        records = original()
        if not records and create:
            if self.db.in_transaction:
                raise ValueError('Cadence anchoring requires its own serialized transaction')
            with self.db:
                self.db.execute('BEGIN IMMEDIATE')
                records = original()
                if not records:
                    self._event('fit_cadence_anchored', {'version': 'v7-fit-anchor-v1',
                        'plan_id': self.plan.digest, 'cadence_hash': rule.digest,
                        'anchor': self.now().isoformat()})
                    records = original()
        if len(records) != 1:
            raise ValueError('Exactly one original elapsed cadence anchor required')
        record = records[0];detail = record['detail'];anchor = stamp(detail['anchor'])
        if (detail != {'version': 'v7-fit-anchor-v1', 'plan_id': self.plan.digest,
                       'cadence_hash': rule.digest, 'anchor': anchor.isoformat()}
                or not anchor <= stamp(record['at']) <= self.now()):
            raise ValueError('Original elapsed cadence anchor changed')
        prior_jobs = [json.loads(r[0]) for r in self.db.execute('SELECT record FROM operations ORDER BY seq')
                      if json.loads(r[0])['kind'] == 'fit_planned'
                      and json.loads(r[0])['seq'] <= record['seq']]
        if prior_jobs:
            raise ValueError('Elapsed fit was planned before its original cadence anchor')
        return anchor

    def due_fit_cutoff(self):
        if self.plan.training_window_days is None:
            return month_start(self.now())
        from .rolling_cadence import RollingCadence
        anchor = self._cadence_anchor(create=True)
        return RollingCadence().latest_fit_cutoff(anchor, self.now())

    def fit_cutoff(self, value):
        if self.plan.training_window_days is None:
            return boundary(value)
        from .rolling_cadence import RollingCadence
        value = stamp(value)
        RollingCadence().slot_identity(self._cadence_anchor(), value)
        return value

    def fit_cycle_start(self, value):
        if self.plan.training_window_days is None:
            return month_start(stamp(value))
        from .rolling_cadence import RollingCadence
        anchor = self._cadence_anchor()
        latest = RollingCadence().latest_fit_cutoff(anchor, value)
        return anchor if latest is None else latest

    def verify_training_view(self, detail):
        anchor = self._cadence_anchor()
        events = [json.loads(r[0]) for r in self.db.execute('SELECT record FROM operations ORDER BY seq')
                  if json.loads(r[0])['kind'] == 'training_view_refreshed'
                  and json.loads(r[0])['detail'].get('refresh_slot') == detail['refresh_slot']]
        if (len(events) != 1 or events[0]['detail'] != detail
                or stamp(events[0]['at']) < stamp(detail['observed_at'])):
            raise ValueError('Daily training view lost its unique original refresh event')
        return self._validate_training_view(detail, anchor)

    def _validate_training_view(self, detail, anchor):
        from .rolling_cadence import RollingCadence
        from .rolling import RollingSpec
        if self.plan.training_window_days is None:
            raise ValueError('Elapsed training view cannot enter a calendar lifecycle')
        fields = {'version','plan_id','refresh_slot','observed_at','source_id',
                  'raw_asof_hash','training','fit_permission','search_exposure'}
        observed = stamp(detail['observed_at'])
        slot = stamp(detail['refresh_slot'])
        # The very first read precedes the newly committed anchor. Later reads
        # belong to the original day grid; their actual receipt time is retained.
        first = slot == anchor and observed <= anchor
        if (set(detail) != fields or detail['version'] != 'v7-elapsed-training-view-v1'
                or detail['plan_id'] != self.plan.digest or detail['source_id'] != self.plan.source_id
                or detail['fit_permission'] is not False or detail['search_exposure'] is not False
                or observed > self.now()
                or (not first and RollingCadence().latest_refresh_cutoff(anchor, observed) != slot)
                or not isinstance(detail['raw_asof_hash'],str) or len(detail['raw_asof_hash']) != 64):
            raise ValueError('Original daily training view cadence or provenance changed')
        start,end = RollingCadence().window(observed)
        training = detail['training']
        if (training.get('train_start') != start.isoformat() or training.get('cutoff') != end.isoformat()
                or training.get('fit') != 'cold' or type(training.get('rows')) is not int
                or training['rows'] < RollingSpec().minimum_train_rows):
            raise ValueError('Daily training view changed its original 90-day support')
        path = self.directory / 'training_views' / (fingerprint(slot.isoformat())+'.json')
        if json.loads(path.read_text('utf-8')) != detail:
            raise ValueError('Daily training view artifact differs from its original event')
        return detail

    def _time_event_verifier(self):
        """One invocation-local time basis; the caller still verifies every hash.

        No persistent cache or caller bypass flag: both full/readonly verifiers
        reconstruct the basis and read each original daily artifact on each call.
        """
        from .rolling_cadence import RollingCadence
        found = self.db.execute("SELECT 1 FROM operations WHERE json_extract(record,'$.kind')=? LIMIT 1",
                                ('fit_cadence_anchored',)).fetchone()
        anchor = self._cadence_anchor() if found else None
        views = set()
        def check(record):
            kind,detail = record['kind'],record['detail']
            if kind == 'fit_cadence_anchored' and self.plan.training_window_days is None:
                raise ValueError('Elapsed anchor cannot enter calendar lifecycle')
            if kind == 'training_view_refreshed':
                slot = detail['refresh_slot']
                if anchor is None or slot in views or stamp(record['at']) < stamp(detail['observed_at']):
                    raise ValueError('Daily training view lost its unique original refresh event')
                views.add(slot)
                self._validate_training_view(detail,anchor)
            if kind == 'fit_planned':
                if self.plan.training_window_days is None:
                    boundary(detail['cutoff'])
                elif anchor is None:
                    raise ValueError('Exactly one original elapsed cadence anchor required')
                else:
                    RollingCadence().slot_identity(anchor,detail['cutoff'])
                if stamp(detail['cutoff']) > stamp(record['at']):
                    raise ValueError('Fit plan preceded its own causal cutoff')
        return check

    def plan_fit(self, cutoff, stream, pipeline):
        from .quarantined_observation import current_binding, job_binding
        cutoff = self.fit_cutoff(cutoff)
        if cutoff > self.now() or not isinstance(pipeline, dict) or not pipeline or not stream:
            raise ValueError('Fit needs a closed monthly snapshot and pipeline identity')
        identifier = fingerprint({'run': self.plan.digest, 'cutoff': cutoff.isoformat(),
                                  'stream': stream})
        value = canonical(pipeline)
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            old = self.db.execute('SELECT pipeline FROM jobs WHERE id=?', (identifier,)).fetchone()
            if old:
                if old[0] != value:
                    raise ValueError('Monthly fit identity cannot change on retry')
                job_binding(self, identifier)
                return identifier
            self.db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?)',
                            (identifier, cutoff.isoformat(), stream, value, 'planned',
                             None, None, None, None))
            prefix = current_binding(self)
            detail = {'job': identifier, 'stream': stream, 'cutoff': cutoff.isoformat(), 'pipeline': pipeline}
            if prefix is not None:
                detail['observation_prefix'] = prefix
            self._event('fit_planned', detail)
        return identifier

    def advance_fit(self, job, status, *, output=None):
        stages = ('planned', 'snapshot_sealed', 'features_ready', 'fitted', 'verified')
        if status not in stages[1:]:
            raise ValueError('Invalid monthly fit transition')
        if status == 'verified':
            raise ValueError('Verified state requires seal_model with a persisted artifact')
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            row = self.db.execute('SELECT status FROM jobs WHERE id=?', (job,)).fetchone()
            if row is None:
                raise ValueError('Unknown job')
            if row[0] == status:
                return False
            if row[0] not in stages or stages.index(status) != stages.index(row[0]) + 1:
                raise ValueError('Monthly fit states cannot skip steps')
            self.db.execute('UPDATE jobs SET status=? WHERE id=?', (status, job))
            self._event(status, {'job': job, 'output': output})
        return True

    def seal_model(self, job, model, *, certificate=None):
        from .quarantined_observation import seal_publication, read_publication
        if (self.directory / 'model_isolations' / f'{job}.json').exists():
            raise ValueError('An isolated candidate cannot be repaired or credited in place')
        row = self.db.execute('SELECT pipeline,status FROM jobs WHERE id=?', (job,)).fetchone()
        if row is None or row[1] not in ('features_ready', 'fitted', 'verified', 'activated'):
            raise ValueError('Model fit requires ready snapshot/features')
        pipeline = json.loads(row[0])
        if self._work_authority_bound() and 'pool' in pipeline and model.kind != 'v7_joint_pool':
            raise ValueError('A bound frozen pool job requires its complete admitted joint model')
        if model.kind in {'v7_neural_forward', 'v7_known_linear'} or (
                model.kind == 'v7_joint_pool' and (self._work_authority_bound() or certificate is not None)):
            identifier = self.registry.save(model, pipeline, certificate=certificate, expected_job=job)
        else:
            if certificate is not None:
                raise ValueError('Neural certificate cannot seal another model kind')
            identifier = self.registry.save(model, pipeline)
        restored, _ = self._load_job_model(job, identifier, pipeline)
        if restored.digest != model.digest:
            raise ValueError('Model verification failed')
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            old = self.db.execute('SELECT model,status FROM jobs WHERE id=?', (job,)).fetchone()
            if old[0] and old[0] != identifier:
                raise ValueError('Verified monthly model is immutable')
            failure = self.db.execute('SELECT failure FROM jobs WHERE id=?', (job,)).fetchone()[0]
            if old[1] == 'activated' or (old[1] == 'verified' and not failure):
                read_publication(self, 'model', job)
                return identifier
            completed = self.now().isoformat()
            self.db.execute('UPDATE jobs SET status=?,model=?,completed=?,failure=NULL WHERE id=?',
                            ('verified', identifier, completed, job))
            self._event('model_verified', {'job': job, 'model': identifier,
                                         'completed': completed})
        seal_publication(self, 'model', job)
        return identifier

    def fail_fit(self, job, code):
        if code not in {'resource_incomplete', 'data_incomplete', 'numerical_failure',
                        'integrity_failure', 'interrupted'}:
            raise ValueError('Typed fit failure required')
        with self.db:
            row = self.db.execute('SELECT status FROM jobs WHERE id=?', (job,)).fetchone()
            if row is None or row[0] == 'activated':
                raise ValueError('Cannot invalidate an activation')
            self.db.execute('UPDATE jobs SET failure=? WHERE id=?', (code, job))
            self._event('fit_failed_old_model_retained', {'job': job, 'code': code})

    def activate(self, job, legal_origin, *, expected_generation=0):
        from .quarantined_observation import publication_visible
        if (self.directory / 'model_isolations' / f'{job}.json').exists():
            raise ValueError('An isolated candidate cannot activate')
        origin = stamp(legal_origin)
        if origin > self.now():
            raise ValueError('Activation cannot precede a real available origin')
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            row = self.db.execute('SELECT stream,status,completed,cutoff,activation,model,failure FROM jobs '
                                  'WHERE id=?', (job,)).fetchone()
            if row is None or row[1] not in ('verified', 'activated'):
                raise ValueError('Only a fully verified model can activate')
            stream, _, completed, cutoff, existing, model, failure = row
            if failure:
                raise ValueError('Failed model must be reverified before activation')
            pipeline = json.loads(self.db.execute('SELECT pipeline FROM jobs WHERE id=?', (job,)).fetchone()[0])
            self._load_job_model(job, model, pipeline)
            if origin < stamp(completed) or origin < stamp(cutoff):
                raise ValueError('Activation cannot precede actual fit completion')
            if publication_visible(self, 'model', job, origin) is False:
                raise ValueError('Activation precedes original physical model publication')
            active = self.db.execute('SELECT job,generation FROM active WHERE stream=?', (stream,)).fetchone()
            if existing:
                if active and active[0] == job:
                    return active[1]
                raise ValueError('Old activated job cannot be reactivated')
            generation = active[1] if active else 0
            if generation != expected_generation:
                raise ValueError('Activation generation changed; retry from current model')
            if active:
                prior = self.db.execute('SELECT cutoff FROM jobs WHERE id=?', (active[0],)).fetchone()[0]
                if stamp(cutoff) <= stamp(prior):
                    raise ValueError('Older monthly model cannot displace current model')
            self.db.execute('INSERT INTO active VALUES (?,?,?) ON CONFLICT(stream) '
                            'DO UPDATE SET job=excluded.job,generation=excluded.generation',
                            (stream, job, generation + 1))
            self.db.execute('UPDATE jobs SET status=?,activation=? WHERE id=?',
                            ('activated', origin.isoformat(), job))
            self._event('activated', {'job': job, 'generation': generation + 1,
                                     'origin': origin.isoformat(),
                                     'activation_delay_seconds': (origin - stamp(cutoff)).total_seconds(),
                                     'within_24h': origin - stamp(cutoff) <= pd.Timedelta(hours=24)})
        return generation + 1

    def active_model(self, stream):
        row = self.db.execute('SELECT j.model,j.pipeline,j.activation,a.generation,j.id FROM active a '
                              'JOIN jobs j ON a.job=j.id WHERE a.stream=?', (stream,)).fetchone()
        if row is None:
            return None
        model, _ = self._load_job_model(row[4], row[0], json.loads(row[1]))
        return {'model': model, 'model_id': row[0], 'pipeline': json.loads(row[1]),
                'activation': row[2], 'generation': row[3]}

    def _load_job_model(self, job, identifier, pipeline):
        model, identity = self.registry.load(identifier, expected_pipeline=pipeline)
        if self._work_authority_bound() and 'pool' in pipeline and model.kind != 'v7_joint_pool':
            raise ValueError('A bound frozen pool job cannot load another model kind')
        if model.kind in {'v7_neural_forward', 'v7_known_linear'} or (
                model.kind == 'v7_joint_pool' and (self._work_authority_bound() or identity.get('version') == 4)):
            version = {'v7_neural_forward': 2, 'v7_known_linear': 3, 'v7_joint_pool': 4}[model.kind]
            if identity.get('version') != version or identity.get('job') != job:
                raise ValueError('Serving model lacks the admission for its actual lifecycle job')
        return model, identity

    def _work_authority_bound(self):
        """Actual forward work always needs admission, including before wallet bind.

        Standalone offline Lifecycle fixtures have no runtime manifest. A real
        runtime cannot gain serving permission by postponing wallet creation.
        The operation anchor also survives loss of the external wallet pointer.
        """
        return (self.directory / 'forward.json').exists() or any(
                   json.loads(row[0])['kind'] in {'monthly_work_authority_bound', 'forward_runtime_bound'}
                   for row in self.db.execute('SELECT record FROM operations ORDER BY seq'))

    def record_forecast(self, *, stream, origin, symbol, task, prediction, received_at,
                        feature_open_time, forecast_window_start, model_id, pipeline_id):
        from .nominal_algorithm_policy import guard_forecast
        guard_forecast(self)
        origin, receipt, opened, window = map(stamp,
            (origin, received_at, feature_open_time, forecast_window_start))
        now = self.now()
        if (symbol not in self.plan.symbols or receipt > origin or origin > now
                or receipt < opened + pd.Timedelta(hours=1)
                or window != origin.ceil('h')):
            raise ValueError('Forecast must precede its future window with a timely real receipt')
        active = self.active_model(stream)
        if (not active or active['model_id'] != model_id
                or active['pipeline'].get('pipeline_id') != pipeline_id
                or origin < stamp(active['activation'])):
            raise ValueError('Forecast must bind its actually active model and pipeline')
        now = self.now()
        if origin > now:
            raise ValueError('Forecast must precede its future window with a timely real receipt')
        if ((now - origin).total_seconds() > self.plan.prediction_max_lag_seconds
                or window <= now):
            raise ForecastExpired('Forecast must precede its future window with a timely real receipt')
        payload = {'value': prediction, 'received_at': receipt.isoformat(),
                   'forecast_window_start': window.isoformat(), 'issued_at': now.isoformat(),
                   'model_id': model_id, 'pipeline_id': pipeline_id, 'availability': 'actual_receipt'}
        from .quarantined_observation import current_binding, snapshot_for_prefix
        prefix = current_binding(self)
        if prefix is not None:
            bars = snapshot_for_prefix(self, prefix, origin)
            chosen = bars[(bars.symbol == symbol) & (bars.open_time == opened)]
            if (len(chosen) != 1 or stamp(chosen.iloc[0].available_at) != receipt
                    or origin != stamp(prefix['logical_at'])):
                raise ValueError('Forecast lacks original released synthetic source')
            payload.update(availability='quarantined_finite_observation', observation_prefix=prefix,
                           physical_received_at=chosen.iloc[0].received_at.isoformat())
        # issued_at is fixed on retry, rather than becoming a prediction revision.
        key = (self.plan.digest, stream, origin.isoformat(), symbol, task)
        old = self.forecasts.db.execute('SELECT prediction FROM forecasts WHERE run=? AND stream=? '
                                        'AND origin=? AND symbol=? AND task=?', key).fetchone()
        if old:
            previous = json.loads(old[0])
            payload['issued_at'] = previous['issued_at']
        return self.forecasts.put(run=self.plan.digest, stream=stream, origin=origin,
            symbol=symbol, task=task, model_id=model_id, pipeline_id=pipeline_id, prediction=payload)

    def settle(self, *, stream, origin, symbol, task, label, maturity, received_at,
               label_revision='first_received_v1'):
        origin, maturity, receipt = map(stamp, (origin, maturity, received_at))
        if receipt > self.now() or maturity > receipt or origin >= maturity or not label_revision:
            raise ValueError('Settlement requires a fully matured and received label')
        key = (self.plan.digest, stream, origin.isoformat(), symbol, task)
        forecast = self.forecasts.db.execute('SELECT prediction FROM forecasts WHERE run=? AND '
            'stream=? AND origin=? AND symbol=? AND task=?', key).fetchone()
        if not forecast:
            raise ValueError('No original forecast exists for settlement')
        original = json.loads(forecast[0])
        horizon = 12 if task == 'volatility' else 20 if task == 'first_event' else None
        if horizon is None or maturity < stamp(original['forecast_window_start']) + pd.Timedelta(hours=horizon):
            raise ValueError('Settlement cannot mature before the full task horizon')
        value = {'label': label, 'maturity': maturity.isoformat(), 'received_at': receipt.isoformat(),
                 'revision': label_revision, 'forecast_hash': fingerprint(json.loads(forecast[0]))}
        from .quarantined_observation import current_binding
        prefix = current_binding(self)
        if prefix is not None:
            value['observation_prefix'] = prefix
            value['availability'] = 'quarantined_finite_observation'
        text, digest = canonical(value), fingerprint(value)
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            old = self.db.execute('SELECT record,hash FROM settlement WHERE run=? AND stream=? '
                                 'AND origin=? AND symbol=? AND task=?', key).fetchone()
            if old:
                if old != (text, digest):
                    raise ValueError('First settlement cannot be overwritten by a revision')
                return False
            self.db.execute('INSERT INTO settlement VALUES (?,?,?,?,?,?,?)', (*key, text, digest))
        return True

    def freeze_batch(self, *, batch, ordinal, start, candidates, reference, evidence_sources):
        start = boundary(start)
        now = self.now()
        existing = self.db.execute('SELECT identity,hash FROM batches WHERE id=?', (batch,)).fetchone()
        if existing:
            old = json.loads(existing[0])
            if (fingerprint(old) != existing[1] or old['start'] != start.isoformat()
                    or old['candidates'] != candidates or old['reference'] != reference
                    or old['evidence_sources'] != evidence_sources or old['ordinal'] != ordinal):
                raise ValueError('Frozen batch cannot change on retry')
            return existing[1]
        if (start < now or type(ordinal) is not int or not 1 <= ordinal <= 3
                or not 1 <= len(candidates) <= 3 or len({c['pipeline_id'] for c in candidates}) != len(candidates)
                or not reference.get('pipeline_id') or not evidence_sources):
            raise ValueError('A batch freezes 1..3 pipelines prospectively in one of three slots')
        for evidence in evidence_sources:
            self.exposure.assert_prompt_search(source=self.plan.source_id, symbols=self.plan.symbols,
                start=evidence['start'], end=evidence['end'], batch=batch)
        identity = {'start': start.isoformat(), 'v_end': (start + pd.DateOffset(months=1)).isoformat(),
                    'c_end': (start + pd.DateOffset(months=1 + self.plan.confirmation_months)).isoformat(),
                    'candidates': candidates, 'reference': reference, 'evidence_sources': evidence_sources,
                    'selection': {'minimum_gain': .005, 'coverage': .95, 'max_candidates': 3},
                    'ordinal': ordinal, 'frozen': now.isoformat(), 'plan': self.plan.digest}
        text, digest = canonical(identity), fingerprint(identity)
        self.exposure.reserve_validation(batch=batch, source=self.plan.source_id,
            symbols=self.plan.symbols, start=identity['start'], end=identity['v_end'],
            candidate_ids=[c['pipeline_id'] for c in candidates],
            reference_id=reference['pipeline_id'], frozen_at=now)
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            prior = self.db.execute('SELECT identity FROM batches WHERE id=?', (batch,)).fetchone()
            if prior:
                if prior[0] != text:
                    raise ValueError('Frozen batch cannot change')
                return digest
            self.db.execute('INSERT INTO batches VALUES (?,?,?,?,?,?,?,?)',
                            (batch, ordinal, now.isoformat(), text, digest, 'frozen_V', None, None))
            self._event('batch_frozen', {'batch': batch, 'identity': digest})
        return digest

    def _batch(self, batch):
        row = self.db.execute('SELECT identity,hash,status,selected,result FROM batches WHERE id=?',
                              (batch,)).fetchone()
        if row is None or fingerprint(json.loads(row[0])) != row[1]:
            raise ValueError('Batch integrity failure')
        return json.loads(row[0]), row[2], json.loads(row[3]) if row[3] else None

    def _losses(self, stream, start, end, as_of, *, expected_pipeline=None):
        rows = self.forecasts.db.execute('SELECT origin,symbol,prediction FROM forecasts WHERE run=? '
            'AND stream=? AND task=? AND origin>=? AND origin<? ORDER BY origin,symbol',
            (self.plan.digest, stream, 'volatility', start, end))
        output = {}
        for origin, symbol, text in rows:
            settled = self.db.execute('SELECT record,hash FROM settlement WHERE run=? AND stream=? '
                'AND origin=? AND symbol=? AND task=?',
                (self.plan.digest, stream, origin, symbol, 'volatility')).fetchone()
            if not settled:
                continue
            label = json.loads(settled[0])
            forecast = json.loads(text)
            if expected_pipeline is not None and forecast['pipeline_id'] != expected_pipeline:
                raise ValueError('Forecast stream does not match its frozen pipeline')
            if fingerprint(label) != settled[1] or label['forecast_hash'] != fingerprint(forecast):
                raise ValueError('Settlement/forecast integrity mismatch')
            if stamp(label['received_at']) > stamp(as_of) or stamp(label['maturity']) > stamp(as_of):
                continue
            y, p = float(label['label']), float(forecast['value'])
            if not np.isfinite([y, p]).all() or min(y, p) < 0:
                raise ValueError('Invalid volatility value cannot be removed from scoring')
            key = (stamp(origin).floor('h').isoformat(), symbol)
            if key in output:
                raise ValueError('Multiple origins in the same symbol-hour')
            output[key] = (float((np.log(y + 1e-8) - np.log(p + 1e-8)) ** 2),
                           label['label'], label['revision'])
        return output

    def select_validation(self, batch):
        self.verify()
        identity, status, _ = self._batch(batch)
        end = stamp(identity['v_end'])
        if status != 'frozen_V' or self.now() < end:
            raise ValueError('V has one fixed monthly endpoint')
        # Reserve the only read before outcomes; interruption is visible, never a new look.
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            count = self.db.execute('UPDATE batches SET status=? WHERE id=? AND status=?',
                ('reading_V', batch, 'frozen_V')).rowcount
            if count != 1:
                raise ValueError('V endpoint already consumed')
            self._event('validation_read_reserved', {'batch': batch})
        reference = identity['reference']['pipeline_id']
        base = self._losses(f'{batch}:reference', identity['start'], identity['v_end'], end,
                            expected_pipeline=reference)
        candidate_losses = {}
        for candidate in identity['candidates']:
            candidate_losses[candidate['pipeline_id']] = self._losses(
                f'{batch}:{candidate["pipeline_id"]}', identity['start'], identity['v_end'], end,
                expected_pipeline=candidate['pipeline_id'])
        common = set(base)
        for losses in candidate_losses.values():
            common &= set(losses)
        # Label revision/value must be identical on paired support.
        for key in common:
            if any(value[key][1:] != base[key][1:] for value in candidate_losses.values()):
                raise ValueError('Paired streams disagree on the label revision')
        hours = int((end - stamp(identity['start'])) / pd.Timedelta(hours=1))
        coverage = len(common) / (hours * len(self.plan.symbols))
        records = []
        for candidate in identity['candidates']:
            value = candidate_losses[candidate['pipeline_id']]
            def hourly_mean(losses):
                grouped = {}
                for hour, symbol in common:
                    grouped.setdefault(hour, []).append(losses[hour, symbol][0])
                return float(np.mean([np.mean(items) for items in grouped.values()])) if grouped else 0.
            baseline = hourly_mean(base)
            gain = float(1 - hourly_mean(value) / baseline) if baseline else None
            records.append({'candidate_id': candidate['pipeline_id'], 'role': 'V',
                'common_support': True, 'coverage': coverage, 'gain': gain,
                'complexity': candidate['complexity'], 'reference_pipeline_id': reference})
        selected = select_on_validation(records)
        for symbol in self.plan.symbols:
            self.exposure.record(source=self.plan.source_id, symbols=[symbol], start=identity['start'],
                end=identity['v_end'], role='V', batch=batch, detail={'one_endpoint': True})
        value = {'records': records, 'selected': selected, 'as_of': end.isoformat(),
                 'nominal_hours': hours, 'common_pairs': len(common)}
        with self.db:
            self.db.execute('UPDATE batches SET result=? WHERE id=? AND status=?',
                            (canonical(value), batch, 'reading_V'))
            self._event('validation_score_committed', {'batch': batch, 'result_hash': fingerprint(value)})
        immutable_record(self.directory / 'validation' / f'{fingerprint(batch)}.json', value)
        with self.db:
            self.db.execute('UPDATE batches SET status=?,selected=? WHERE id=?',
                ('selected_C' if selected else 'concluded_no_candidate',
                 canonical(selected) if selected else None, batch))
            self._event('validation_concluded', {'batch': batch, 'selected': selected})
        return value

    def recover_validation(self, batch):
        self.verify()
        identity, status, _ = self._batch(batch)
        if status != 'reading_V':
            raise ValueError('Only a consumed V read may recover its committed result')
        row = self.db.execute('SELECT result FROM batches WHERE id=?', (batch,)).fetchone()
        if not row or not row[0]:
            return None  # No committed score: never perform another outcome look.
        value = json.loads(row[0])
        commits = [json.loads(text)['detail']['result_hash'] for (text,) in self.db.execute(
            'SELECT record FROM operations') if json.loads(text)['kind'] == 'validation_score_committed'
            and json.loads(text)['detail']['batch'] == batch]
        if commits != [fingerprint(value)] or value['as_of'] != identity['v_end']:
            raise ValueError('Committed V result integrity failure')
        if {r['candidate_id'] for r in value['records']} != {c['pipeline_id'] for c in identity['candidates']}:
            raise ValueError('Committed V result differs from frozen candidate set')
        selected = value['selected']
        immutable_record(self.directory / 'validation' / f'{fingerprint(batch)}.json', value)
        with self.db:
            self.db.execute('UPDATE batches SET status=?,selected=? WHERE id=? AND status=?',
                ('selected_C' if selected else 'concluded_no_candidate',
                 canonical(selected) if selected else None, batch, 'reading_V'))
            self._event('validation_concluded', {'batch': batch, 'selected': selected})
        return value

    def calibration_ready(self):
        if not self.plan.calibration_directory:
            return False
        from .calibration_runtime_contract import (
            RuntimeScopeUnavailable,
            runtime_calibration_contract,
        )
        from .pipeline_calibration import verify_calibration_certificate
        result = verify_calibration_certificate(self.plan.calibration_directory,
                                                expected_binding=self.plan.calibration_binding)
        required = {'library_skill', 'factor1', 'factor2', 'factor3', 'algorithm'}
        covered = result.get('covered_claim_types')
        if (result.get('formal_gate_eligible') is not True
                or result.get('full_V7_five_claim_pipeline_calibrated') is not True
                or type(covered) is not list or len(covered) != len(required)
                or any(type(slot) is not str for slot in covered) or set(covered) != required
                or result.get('test_only') is not False):
            return False
        try:
            actual = runtime_calibration_contract(self)
        except RuntimeScopeUnavailable:
            return False
        # A runtime component is necessary, never sufficient: the verifier above
        # must independently reconstruct the full provider/trial/oracle scope.
        # Existing local certificates never produce that full scope or this term.
        return (result.get('runtime_scope_hash') == fingerprint(actual)
                and result.get('rolling_hash') == actual['rolling_hash']
                and result.get('rule_hash') == actual['factor_confirmation_rule_hash'])

    def confirmation_execution_ready(self):
        from .quarantine_execution import permit_for
        return self.calibration_ready() or permit_for(self) is not None

    def reserve_selected_confirmation(self, batch):
        self.verify()
        identity, status, selected = self._batch(batch)
        if status != 'selected_C' or not selected or not self.confirmation_execution_ready():
            raise ValueError('Formal C requires selected frozen pipeline and verified calibration')
        prior = self._batch_operation(batch, 'confirmation_reserved')
        if prior:
            arguments = prior['arguments']
        else:
            actual_start = max(stamp(identity['v_end']), self.now().ceil('h'))
            from .nominal_algorithm_policy import confirmation_start
            actual_start = confirmation_start(self, identity['v_end'], actual_start)
            if actual_start > stamp(identity['v_end']) + pd.Timedelta(hours=24):
                raise ValueError('Confirmation activation window already missed')
            arguments = {'batch': batch, 'source': self.plan.source_id, 'symbols': list(self.plan.symbols),
                'start': identity['v_end'], 'end': identity['c_end'], 'candidate_id': selected['candidate_id'],
                'reference_id': identity['reference']['pipeline_id'], 'rule': asdict(ConfirmationRule()),
                'frozen_at': self.now().isoformat(), 'cohort': 1, 'claim_slot': f'factor{identity["ordinal"]}',
                'actual_start': actual_start.isoformat()}
            # Persist the exact reservation intent before the second database.
            # A retry uses this same clock/identity, never a later re-freeze.
            with self.db:
                self._event('confirmation_reserved', {'batch': batch, 'arguments': arguments})
        return self.exposure.reserve_confirmation(**arguments)

    def conclude_confirmation(self, batch):
        self.verify()
        if not self.confirmation_execution_ready():
            raise ValueError('Confirmation requires a formal certificate or original quarantine enrollment')
        from .quarantine_execution import decision_for
        formal = self.calibration_ready()
        committed = self._batch_operation(batch, 'confirmation_score_committed')
        prior = self.exposure.db.execute('SELECT identity,consumed,result FROM confirmations WHERE batch=?',
                                         (batch,)).fetchone()
        if committed:
            return self._publish_confirmation(batch, committed['result'])
        if prior and prior[2]:
            raise ValueError('Confirmation ledger result has no verified score commit')
        identity = self.exposure.consume_confirmation(batch, as_of=self.now())
        cutoff = identity['scoring_cutoff']
        start = identity.get('actual_start', identity['start'])
        base = self._losses(f'{batch}:reference', start, identity['end'], cutoff,
                            expected_pipeline=identity['reference_id'])
        candidate = self._losses(f'{batch}:{identity["candidate_id"]}',
                                 start, identity['end'], cutoff, expected_pipeline=identity['candidate_id'])
        common = sorted(set(base) & set(candidate))
        for key in common:
            if base[key][1:] != candidate[key][1:]:
                raise ValueError('Paired confirmation streams have inconsistent labels')
        origins = pd.DatetimeIndex([k[0] for k in common], tz='UTC')
        result = paired_effect_interval(origins, [base[k][0] for k in common],
            [candidate[k][0] for k in common], symbols=[k[1] for k in common],
            expected_symbols=self.plan.symbols,
            eligible_count=int((stamp(identity['end']) - stamp(identity['start'])) / pd.Timedelta(hours=1)),
            rule=ConfirmationRule(**identity['rule']), calibration_passed=formal)
        result['formal_eligible'] = formal
        isolated = decision_for(self, result)
        if isolated is not None:
            result['quarantined_decision'] = isolated
        result |= {'pipeline_id': identity['candidate_id'], 'reference_pipeline_id': identity['reference_id'],
                   'completed_at': self.now().isoformat(), 'scoring_cutoff': cutoff}
        with self.db:
            self._event('confirmation_score_committed', {'batch': batch, 'result': result,
                         'reservation_hash': fingerprint(identity)})
        return self._publish_confirmation(batch, result)

    def recover_confirmation(self, batch):
        """Publish a committed C score, or return None without reading outcomes."""
        self.verify()
        if self._batch_operation(batch, 'confirmation_score_committed') is None:
            return None
        return self.conclude_confirmation(batch)

    def _publish_confirmation(self, batch, result):
        from .quarantined_observation import current_binding, seal_publication
        prior = self.exposure.db.execute('SELECT consumed,result FROM confirmations WHERE batch=?',
                                         (batch,)).fetchone()
        if not prior or not prior[0]:
            raise ValueError('Confirmation publication lacks its consumed exposure')
        if prior[1] is None:
            self.exposure.finish_confirmation(batch, result)
        elif json.loads(prior[1]) != result:
            raise ValueError('Confirmation exposure result differs from committed score')
        immutable_record(self.directory / 'claims' / f'{fingerprint(batch)}.json', result)
        with self.db:
            status = self.db.execute('SELECT status FROM batches WHERE id=?', (batch,)).fetchone()[0]
            if status != 'concluded':
                self.db.execute('UPDATE batches SET status=?,result=? WHERE id=?',
                                ('concluded', canonical(result), batch))
                self._event('confirmation_concluded', {'batch': batch, 'result': result})
        # Sample publication only after the primary SQL conclusion committed.
        # Recovery republishes the same result, but cannot backdate visibility.
        path = self.directory / 'claim_publications' / f'{fingerprint(batch)}.json'
        if not path.exists():
            row = self.db.execute('SELECT record,hash FROM operations WHERE '
                "json_extract(record,'$.kind')='confirmation_concluded' AND "
                "json_extract(record,'$.detail.batch')=?", (batch,)).fetchone()
            record = json.loads(row[0])
            value = {'version': 'v7-confirmation-publication-v1', 'plan_id': self.plan.digest,
                     'batch': batch, 'result_hash': fingerprint(result),
                     'conclusion_hash': row[1], 'concluded_at': record['at'],
                     'published_at': self.now().isoformat()}
            prefix = current_binding(self)
            if prefix is not None:
                value['observation_prefix'] = prefix
            immutable_record(path, {'record': value, 'hash': fingerprint(value)})
            seal_publication(self, 'confirmation', batch)
        self.confirmation_publication(batch)
        return result

    def confirmation_publication(self, batch):
        """Read original post-commit visibility without repairing a lost anchor."""
        path = self.directory / 'claim_publications' / f'{fingerprint(batch)}.json'
        if not path.is_file():
            raise ValueError('Confirmation publication anchor is missing')
        wrapper = json.loads(path.read_text('utf-8'))
        value = wrapper['record']
        fields = {'version', 'plan_id', 'batch', 'result_hash', 'conclusion_hash', 'concluded_at', 'published_at'}
        from .quarantined_observation import read_publication
        observed = read_publication(self, 'confirmation', batch)
        if observed is not None:
            fields.add('observation_prefix')
            if value.get('observation_prefix') != observed['observation_prefix']:
                raise ValueError('Confirmation publication lost original observation prefix')
        found = self.db.execute('SELECT record,hash FROM operations WHERE '
            "json_extract(record,'$.kind')='confirmation_concluded' AND "
            "json_extract(record,'$.detail.batch')=?", (batch,)).fetchall()
        row = self.db.execute('SELECT status,result FROM batches WHERE id=?', (batch,)).fetchone()
        if len(found) != 1 or not row or row[0] != 'concluded':
            raise ValueError('Publication lacks its original committed conclusion')
        record, digest = json.loads(found[0][0]), found[0][1]
        if (set(wrapper) != {'record', 'hash'} or wrapper['hash'] != fingerprint(value)
                or set(value) != fields or value['version'] != 'v7-confirmation-publication-v1'
                or value['plan_id'] != self.plan.digest or value['batch'] != batch
                or value['conclusion_hash'] != digest or fingerprint(record) != digest
                or value['result_hash'] != fingerprint(json.loads(row[1]))
                or value['concluded_at'] != record['at']
                or not stamp(value['concluded_at']) <= stamp(value['published_at']) <= self.now()):
            raise ValueError('Confirmation publication integrity failure')
        return value

    def adoption_pipeline(self, cutoff, k0):
        self.verify()
        cutoff = self.fit_cutoff(cutoff)
        if cutoff > self.now() or not k0.get('pipeline_id'):
            raise ValueError('A0 adoption needs actual cutoff and K0 identity')
        fallback = k0
        rows = self.db.execute('SELECT id,identity,result FROM batches WHERE status=? '
                               'ORDER BY ordinal DESC', ('concluded',))
        for batch, text, result_text in rows:
            result = json.loads(result_text)
            from .quarantine_execution import verify_decision
            isolated = verify_decision(self, result)
            accepted = result.get('supported') is True or (isolated is not None and isolated['accepted'])
            if accepted and stamp(result['completed_at']) <= cutoff:
                publication = self.confirmation_publication(batch)
                path = self.directory / 'claim_publications' / f'{fingerprint(batch)}.json'
                from .quarantined_observation import publication_visible
                visible = publication_visible(self, 'confirmation', batch, cutoff)
                if (stamp(publication['published_at']) > cutoff
                        or (path.stat().st_mtime_ns > cutoff.value if visible is None else not visible)):
                    continue
                if isolated is None and not self.calibration_ready():
                    raise ValueError('Supported adoption certificate no longer verifies')
                identity = json.loads(text)
                fallback = next(c for c in identity['candidates'] if c['pipeline_id'] == result['pipeline_id'])
                break
        # The whole typed policy has a separate completed algorithm endpoint.
        # Its verifier may retain this actual factor/K0 choice, never recurse
        # into adoption or borrow the factor's own confirmation as policy proof.
        from .typed_policy_confirmation import adopted_pipeline
        return adopted_pipeline(self, cutoff, fallback)

    def _batch_operation(self, batch, kind):
        matches = [record['detail'] for (text,) in self.db.execute('SELECT record FROM operations')
                   if (record := json.loads(text))['kind'] == kind and record['detail']['batch'] == batch]
        if len(matches) > 1:
            raise ValueError('A protected batch operation cannot be repeated')
        return matches[0] if matches else None

    def _verify_batches(self, operations):
        """Rebuild mutable V/C indexes from the verified immutable operation chain.

        Score values are compared, never re-estimated. The two databases may be
        between publication stages, but cannot supply conflicting finished facts.
        """
        stages = {}
        for operation in operations:
            kind, detail = operation['kind'], operation['detail']
            if kind not in {'batch_frozen', 'validation_read_reserved', 'validation_score_committed',
                            'validation_concluded', 'confirmation_reserved',
                            'confirmation_score_committed', 'confirmation_concluded'}:
                continue
            batch = detail['batch']
            if kind == 'batch_frozen':
                if batch in stages:
                    raise ValueError('Duplicate frozen batch operation')
                stages[batch] = {'state': 'frozen_V', 'batch_frozen': detail}
                continue
            if batch not in stages or kind in stages[batch]:
                raise ValueError('Missing or duplicate protected batch operation')
            stage = stages[batch]
            if kind == 'validation_read_reserved':
                if stage['state'] != 'frozen_V':
                    raise ValueError('Invalid V read transition')
                stage['state'] = 'reading_V'
            elif kind == 'validation_score_committed':
                if stage['state'] != 'reading_V':
                    raise ValueError('V score lacks its one reserved read')
            elif kind == 'validation_concluded':
                if stage['state'] != 'reading_V' or 'validation_score_committed' not in stage:
                    raise ValueError('V conclusion lacks its committed score')
                stage['state'] = 'selected_C' if detail['selected'] else 'concluded_no_candidate'
            elif kind in {'confirmation_reserved', 'confirmation_score_committed'}:
                if (stage['state'] != 'selected_C' or
                        (kind == 'confirmation_score_committed' and 'confirmation_reserved' not in stage)):
                    raise ValueError('C operation lacks its frozen selection/reservation')
            else:
                if stage['state'] != 'selected_C' or 'confirmation_score_committed' not in stage:
                    raise ValueError('C conclusion lacks its committed score')
                stage['state'] = 'concluded'
            stage[kind] = detail
        rows = {row[0]: row[1:] for row in self.db.execute('SELECT * FROM batches')}
        if set(rows) != set(stages):
            raise ValueError('Batch index differs from verified operations chain')
        for directory, kind in [('validation', 'validation_score_committed'),
                                ('claims', 'confirmation_score_committed')]:
            permitted = {f'{fingerprint(batch)}.json' for batch, stage in stages.items() if kind in stage}
            if {p.name for p in (self.directory / directory).glob('*.json')} - permitted:
                raise ValueError('Protected result file lacks a committed batch score')
        for batch, row in rows.items():
            ordinal, frozen, text, digest, status, selected_text, result_text = row
            identity = json.loads(text)
            stage = stages[batch]
            if (fingerprint(identity) != digest or stage['batch_frozen']['identity'] != digest
                    or identity['ordinal'] != ordinal or identity['frozen'] != frozen
                    or identity['plan'] != self.plan.digest or status != stage['state']):
                raise ValueError('Batch identity/status differs from operations chain')
            expected_v = {'batch': batch, 'source': self.plan.source_id, 'symbols': sorted(self.plan.symbols),
                'start': identity['start'], 'end': identity['v_end'],
                'candidate_ids': [c['pipeline_id'] for c in identity['candidates']],
                'reference_id': identity['reference']['pipeline_id'], 'frozen_at': frozen}
            validation = self.exposure.db.execute('SELECT identity FROM validations WHERE batch=?', (batch,)).fetchone()
            if validation is None or json.loads(validation[0]) != expected_v:
                raise ValueError('Frozen V differs from its exposure reservation')
            selected = stage.get('validation_concluded', {}).get('selected')
            if (json.loads(selected_text) if selected_text else None) != selected:
                raise ValueError('Batch selection differs from committed V decision')
            result = json.loads(result_text) if result_text else None
            v_file = self.directory / 'validation' / f'{fingerprint(batch)}.json'
            if 'validation_score_committed' in stage:
                value = json.loads(v_file.read_text('utf-8')) if v_file.exists() else result
                if (value is None or fingerprint(value) != stage['validation_score_committed']['result_hash']
                        or value['as_of'] != identity['v_end']
                        or len(value['records']) != len(identity['candidates'])
                        or {r['candidate_id'] for r in value['records']} != set(expected_v['candidate_ids'])
                        or any(r['reference_pipeline_id'] != expected_v['reference_id'] for r in value['records'])
                        or value['selected'] != select_on_validation(value['records'])):
                    raise ValueError('Committed V content differs from frozen identities or score hash')
                if 'validation_concluded' in stage and (not v_file.exists() or value['selected'] != selected):
                    raise ValueError('Final V decision lacks its exact published result')
                if status != 'concluded' and value != result:
                    raise ValueError('Mutable V result differs from its committed content')
                self._verify_exposed_period(batch, 'V', identity['start'], identity['v_end'])
            elif result is not None or v_file.exists():
                raise ValueError('V result exists without a committed score')
            reservation = self.exposure.db.execute(
                'SELECT identity,consumed,result,cohort,claim_slot,reserved_at FROM confirmations WHERE batch=?',
                (batch,)).fetchone()
            c_intent = stage.get('confirmation_reserved')
            if not c_intent:
                if reservation is not None:
                    raise ValueError('C ledger has no operation-bound reservation')
                continue
            arguments = c_intent['arguments']
            expected_args = {'batch': batch, 'source': self.plan.source_id, 'symbols': list(self.plan.symbols),
                'start': identity['v_end'], 'end': identity['c_end'], 'candidate_id': selected['candidate_id'],
                'reference_id': expected_v['reference_id'], 'rule': asdict(ConfirmationRule()),
                'cohort': 1, 'claim_slot': f'factor{ordinal}',
                'frozen_at': arguments['frozen_at'], 'actual_start': arguments['actual_start']}
            if (arguments != expected_args or not stamp(identity['v_end']) <= stamp(arguments['frozen_at'])
                    <= stamp(arguments['actual_start']) <= stamp(identity['v_end']) + pd.Timedelta(hours=24)):
                raise ValueError('C intent differs from frozen factor claim or activation contract')
            expected_c = {**arguments, 'symbols': sorted(self.plan.symbols), 'alpha': .01,
                'scoring_cutoff': (stamp(identity['c_end']) + pd.Timedelta(hours=36)).isoformat(),
                'nominal_start': identity['v_end'],
                'activation_delay_seconds': (stamp(arguments['actual_start']) - stamp(identity['v_end'])).total_seconds()}
            score = stage.get('confirmation_score_committed')
            if reservation is None:
                if score:
                    raise ValueError('Committed C score lost its exposure reservation')
                continue  # Intent durable, exposure publication not completed yet.
            c_identity, consumed, c_text, cohort, claim_slot, reserved_at = reservation
            if (json.loads(c_identity) != expected_c or cohort != 1 or claim_slot != f'factor{ordinal}'
                    or not stamp(arguments['frozen_at']) <= stamp(reserved_at) <= stamp(arguments['actual_start'])
                    or consumed not in (0, 1)):
                raise ValueError('C ledger identity/index differs from committed reservation')
            if consumed:
                self._verify_exposed_period(batch, 'C', identity['v_end'], identity['c_end'], expected_c)
            if score:
                c_result = score['result']
                from .quarantine_execution import verify_decision
                verify_decision(self, c_result)
                if (not consumed or score['reservation_hash'] != fingerprint(expected_c)
                        or c_result['pipeline_id'] != expected_c['candidate_id']
                        or c_result['reference_pipeline_id'] != expected_c['reference_id']
                        or c_result['scoring_cutoff'] != expected_c['scoring_cutoff']
                        or stamp(c_result['completed_at']) < stamp(expected_c['scoring_cutoff'])
                        or (c_text is not None and json.loads(c_text) != c_result)):
                    raise ValueError('C score differs from committed full content or reservation')
                c_file = self.directory / 'claims' / f'{fingerprint(batch)}.json'
                if c_file.exists() and json.loads(c_file.read_text('utf-8')) != c_result:
                    raise ValueError('Published C claim differs from committed result')
                if status == 'concluded' and (not c_file.exists() or c_text is None or result != c_result
                        or stage['confirmation_concluded']['result'] != c_result):
                    raise ValueError('Final C index/claim/exposure differs from committed score')
            elif c_text is not None:
                raise ValueError('Exposure C result lacks a committed score')

    def _verify_exposed_period(self, batch, role, start, end, confirmation=None):
        for symbol in self.plan.symbols:
            rows = self.exposure.db.execute('SELECT id,detail,observed_at,recorded_at FROM evidence '
                'WHERE source=? AND symbol=? AND start=? AND end=? AND role=? AND batch=?',
                (self.plan.source_id, symbol, start, end, role, batch)).fetchall()
            if not rows:
                raise ValueError('Committed protected read lost its exposure receipt')
            for digest, detail_text, observed, recorded in rows:
                detail = json.loads(detail_text)
                record = ({**confirmation, 'symbol': symbol, 'role': role} if confirmation else
                          {'source': self.plan.source_id, 'symbol': symbol, 'start': start,
                           'end': end, 'role': role, 'batch': batch, 'detail': detail})
                record.update(observed_at=observed, recorded_at=recorded)
                if (fingerprint(record) != digest or not stamp(end) <= stamp(observed) <= stamp(recorded) <= self.now()
                        or (confirmation is not None and detail != confirmation)):
                    raise ValueError('Protected exposure receipt integrity failure')

    def verify(self):
        return self._verify()

    def verify_serving(self):
        """Serving availability only; never statistical/admission/queue approval."""
        return self._verify(serving=True)

    def _verify(self, *, serving=False):
        self.forecasts.recover_receipts()
        prior = '0' * 64
        jobs, active = {}, {}
        operations = []
        check_time = self._time_event_verifier()
        for sequence, text, digest in self.db.execute('SELECT * FROM operations ORDER BY seq'):
            record = json.loads(text)
            if record['seq'] != sequence or record['previous'] != prior or fingerprint(record) != digest:
                raise ValueError('Lifecycle operations chain mismatch')
            prior = digest
            operations.append(record)
            detail, kind = record['detail'], record['kind']
            check_time(record)
            if kind == 'fit_planned':
                jobs[detail['job']] = [detail['cutoff'], detail['stream'], canonical(detail['pipeline']),
                                       'planned', None, None, None, None]
            elif kind in ('snapshot_sealed', 'features_ready', 'fitted'):
                jobs[detail['job']][3] = kind
            elif kind == 'model_verified':
                job = jobs[detail['job']]
                job[3:6] = ['verified', detail['model'], detail['completed']]
                job[7] = None
            elif kind == 'fit_failed_old_model_retained':
                jobs[detail['job']][7] = detail['code']
            elif kind == 'activated':
                job = jobs[detail['job']]
                job[3], job[6] = 'activated', detail['origin']
                active[job[1]] = (detail['job'], detail['generation'])
        observed = {row[0]: list(row[1:]) for row in self.db.execute('SELECT * FROM jobs')}
        if observed != jobs:
            raise ValueError('Monthly job index differs from verified operations chain')
        if {r[0]: tuple(r[1:]) for r in self.db.execute('SELECT * FROM active')} != active:
            raise ValueError('Active model index differs from activation chain')
        from .model_copy_isolation import read_anchors, candidate, record
        isolated = read_anchors(self, jobs, operations)
        if isolated and not serving:
            raise ValueError('Isolated model copy: full integrity/statistical approval remains unavailable')
        pending = {}
        for identifier, job in jobs.items():
            if job[4] and identifier not in isolated:
                try:
                    self._load_job_model(identifier, job[4], json.loads(job[2]))
                except (ValueError, FileNotFoundError):
                    if not serving:
                        raise
                    # This is not an exception downgrade: candidate rechecks
                    # the exact artifact fault, original full paid launch and
                    # complete operational A0 model. Every other fault fails.
                    pending[identifier] = candidate(self, identifier, job)
        self.forecasts.verify()
        for row in self.db.execute('SELECT * FROM settlement'):
            value = json.loads(row[5])
            forecast = self.forecasts.db.execute('SELECT prediction FROM forecasts WHERE '
                'run=? AND stream=? AND origin=? AND symbol=? AND task=?', row[:5]).fetchone()
            if (fingerprint(value) != row[6] or not forecast
                    or value['forecast_hash'] != fingerprint(json.loads(forecast[0]))):
                raise ValueError('Settlement/original forecast integrity failure')
        self._verify_batches(operations)
        faults = {job: value['evidence'] for job, value in isolated.items()} | pending
        self._verify_quarantined_ledger(jobs, isolated_copies=faults)
        if pending:
            for evidence in pending.values():
                record(self, evidence)
            return self._verify(serving=True)  # Original complete readback, no permission cache.
        return {'event_head': prior, 'jobs': self.db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0],
                'forecasts': self.forecasts.db.execute('SELECT COUNT(*) FROM forecasts').fetchone()[0],
                'settlements': self.db.execute('SELECT COUNT(*) FROM settlement').fetchone()[0],
                'claim_budget': self.exposure.budget_status(),
                'formal_gate_ready': False if faults else self.calibration_ready(),
                **({'serving_only_isolated_jobs': sorted(faults)} if faults else {})}

    def _verify_quarantined_ledger(self, jobs, *, isolated_copies=None):
        """Stored logical records retain original physical/source provenance."""
        from .quarantined_observation import (
            QuarantinedObservation, job_binding, read_publication, snapshot_for_prefix,
        )
        observed = QuarantinedObservation.from_lifecycle(self)
        if observed is None:
            return
        for identifier, job in jobs.items():
            job_binding(self, identifier)
            if job[4]:
                if identifier in (isolated_copies or {}):
                    from .quarantined_observation import _read_publication
                    _read_publication(self, 'model', identifier,
                                      isolated_copy=isolated_copies[identifier])
                else:
                    read_publication(self, 'model', identifier)
        for row in self.forecasts.iter_records():
            value = json.loads(row['prediction'])
            prefix = value.get('observation_prefix')
            observed.verify_binding(prefix)
            origin = stamp(row['origin'])
            bars = snapshot_for_prefix(self, prefix, origin)
            source = bars[(bars.symbol == row['symbol']) &
                          (bars.open_time == origin.floor('h')-pd.Timedelta(hours=1))]
            if (value.get('availability') != 'quarantined_finite_observation'
                    or origin != stamp(prefix['logical_at']) or len(source) != 1
                    or stamp(value['received_at']) != origin
                    or stamp(value['issued_at']) != origin
                    or value.get('physical_received_at') != source.iloc[0].received_at.isoformat()):
                raise ValueError('Quarantined forecast lost original logical/physical provenance')
        for row in self.db.execute('SELECT * FROM settlement'):
            value = json.loads(row[5])
            prefix = value.get('observation_prefix')
            observed.verify_binding(prefix)
            if (value.get('availability') != 'quarantined_finite_observation'
                    or stamp(value['received_at']) > stamp(prefix['logical_at'])):
                raise ValueError('Quarantined settlement lost original observation prefix')
            # The actual finite forward runtime has one volatility task. This
            # recomputes its original full 12-hour raw-law answer, not a score.
            if row[4] != 'volatility':
                raise ValueError('Unregistered quarantine settlement task')
            raw = self.forecasts.db.execute('SELECT prediction FROM forecasts WHERE '
                'run=? AND stream=? AND origin=? AND symbol=? AND task=?', row[:5]).fetchone()
            prediction = json.loads(raw[0])
            start = stamp(prediction['forecast_window_start'])
            end = start+pd.Timedelta(hours=12)
            bars = snapshot_for_prefix(self, prefix, prefix['logical_at'])
            segment = bars[(bars.symbol == row[3]) & (bars.open_time >= start-pd.Timedelta(hours=1))
                           & (bars.open_time < end)].sort_values('open_time')
            expected = pd.date_range(start-pd.Timedelta(hours=1), end, freq='h', inclusive='left')
            if len(segment) != 13 or not pd.DatetimeIndex(segment.open_time).equals(expected):
                raise ValueError('Quarantined label lacks its full original source horizon')
            target = float(np.sqrt(np.square(np.diff(np.log(segment.close.to_numpy()))).sum()))
            maturity = max(end+pd.Timedelta(seconds=5), segment.available_at.max())
            if (value['label'] != target or stamp(value['maturity']) != maturity
                    or stamp(value['received_at']) != maturity
                    or value['revision'] != fingerprint(segment.revision.tolist())):
                raise ValueError('Quarantined label differs from its original source answer')

    def controller_lock(self):
        # One handle per lifecycle instance makes nested controller/tick/service
        # sections reentrant in this thread. Other instances/processes still
        # contend on the same OS file lock. Observation writers take this lock
        # before their journal lock, so a composite step can freeze its prefix
        # without borrowing the paid ticket's source-audit reuse context.
        path = str(self.directory / '.controller.lock')
        lock = getattr(self, '_controller_lock_handle', None)
        if lock is None:
            lock = FileLock(path, timeout=0)
            self._controller_lock_handle = lock
        elif lock.lock_file != path:
            raise ValueError('Original lifecycle lock directory changed')
        return lock
