"""Bounded rolling forward controller. Polling, fitting and research are explicit.

The public CLI performs one tick; a resource-limited supervisor may repeat it.
Research runs outside the prediction tick and never changes protected recipes.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from factorlab.protocol import fingerprint, project_path
from factorlab.time_utils import HOUR, month_start

from .evidence import snapshot_source
from .joint_pool import FrozenPool
from .known_library import build_feature_history
from .rolling import RollingSpec
from .v7_feed import ReceiptArchive, SnapshotUnavailable
from .v7_lifecycle import ForecastExpired, Lifecycle, LifecyclePlan, stamp
from .v7_run import source_files
from .v7_storage import immutable_record


@dataclass(frozen=True)
class ForwardConfig:
    symbols: tuple[str, ...] = ('BTCUSDT',)
    archive_delay_seconds: int = 5
    context_hours: int = 1234  # 720 + max DSL512 + difference close + margin
    polling_pages: int = 1
    live_trading: bool = False
    training_window_days: int | None = None
    refit_interval_days: int | None = None

    def __post_init__(self):
        periods = (self.training_window_days, self.refit_interval_days)
        if periods != (None, None) and (periods != (90, 15) or any(type(v) is not int for v in periods)):
            raise ValueError('Elapsed mode requires exactly 90-day training and 15-day refit')
        if (not 1 <= len(self.symbols) <= 3 or len(set(self.symbols)) != len(self.symbols)
                or type(self.archive_delay_seconds) is not int or not 0 <= self.archive_delay_seconds <= 3600
                or self.context_hours != 1234 or self.polling_pages not in (1, 2, 3, 4)
                or self.live_trading is not False):
            raise ValueError('Invalid frozen forward runtime')


class ForwardRuntime:
    def __init__(self, directory, feed_directory, *, config=None, lifecycle_plan=None, clock=None, research_spec=None):
        self.directory = project_path(directory)
        self.config = config or ForwardConfig()
        self.clock = clock or (lambda: pd.Timestamp.now(tz='UTC'))
        self.feed = ReceiptArchive(feed_directory, self.config.symbols, clock=self.clock)
        self.spec = RollingSpec(availability_delay_seconds=self.config.archive_delay_seconds,
                                window_days=self.config.training_window_days)
        plan = lifecycle_plan or LifecyclePlan(self.feed.identity, self.config.symbols,
            training_window_days=self.config.training_window_days,
            refit_interval_days=self.config.refit_interval_days)
        if (plan.source_id != self.feed.identity or tuple(plan.symbols) != self.config.symbols
                or plan.training_window_days != self.config.training_window_days
                or plan.refit_interval_days != self.config.refit_interval_days):
            self.feed.close()
            raise ValueError('Forward lifecycle must bind the actual feed/universe')
        if research_spec is not None:
            from .layered_research import LayeredResearchSpec
            if (not isinstance(research_spec, LayeredResearchSpec)
                    or (plan.training_window_days, plan.refit_interval_days) != (90, 15)
                    or plan.calibration_directory is not None or plan.calibration_binding is not None
                    or plan.confirmation_months != 3 or plan.algorithm_months != 6
                    or research_spec.source_hash != fingerprint(source_files())):
                self.feed.close()
                raise ValueError('Explicit immutable layered research spec and elapsed plan required')
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identity = {'version': 'v7-forward-v1', 'config': asdict(self.config),
                         'lifecycle': asdict(plan), 'rolling': asdict(self.spec),
                         'source_files': source_files(), 'feed_directory': str(self.feed.directory)}
        if research_spec is not None:
            self.identity['research_spec'] = asdict(research_spec)
            self.identity['research_spec_hash'] = research_spec.digest
            self.identity['statistical_mode'] = 'prospective_uncertified_research'
            # Bind and preserve the code actually loaded. This also makes
            # isolated staging attributable before it is installed in core.
            implementations = {'forward': Path(__file__),
                               'research_policy': Path(inspect.getfile(LayeredResearchSpec))}
            self.identity['research_implementations'] = {
                name: hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in implementations.items()}
        manifest = self.directory / 'forward.json'
        if research_spec is not None:
            for name, path in implementations.items():
                target = self.directory / 'runtime_implementations' / f'{name}.json'
                content = {'sha256': self.identity['research_implementations'][name],
                           'source_utf8': path.read_bytes().decode('utf-8')}
                if manifest.exists() and (not target.is_file()
                        or json.loads(target.read_text('utf-8')) != content):
                    self.feed.close()
                    raise ValueError('Immutable layered implementation snapshot missing or changed')
                if not manifest.exists():
                    immutable_record(target, content)
        if not manifest.exists():
            archived = snapshot_source(self.directory / 'source')
            if archived['files'] != self.identity['source_files']:
                self.feed.close()
                raise ValueError('Forward source changed during snapshot')
        immutable_record(manifest, {'identity': self.identity, 'sha256': fingerprint(self.identity)})
        from .evidence import file_hash
        for relative, expected in self.identity['source_files'].items():
            if file_hash(self.directory / 'source' / relative) != expected:
                self.feed.close()
                raise ValueError('Saved forward source archive integrity failure')
        if research_spec is None:
            self.lifecycle = Lifecycle(self.directory, plan, clock=self.clock)
        else:
            from .layered_research import LayeredResearchLifecycle
            try:
                self.lifecycle = LayeredResearchLifecycle(self.directory, plan,
                    research_spec=research_spec, clock=self.clock)
            except BaseException:
                self.feed.close()
                raise
        runtime_binding = {'runtime_identity_hash': fingerprint(self.identity)}
        bindings = [json.loads(row[0])['detail'] for row in self.lifecycle.db.execute(
            'SELECT record FROM operations ORDER BY seq')
            if json.loads(row[0])['kind'] == 'forward_runtime_bound']
        if bindings and bindings != [runtime_binding]:
            self.close()
            raise ValueError('Forward lifecycle changed its original runtime binding')
        if not bindings:
            with self.lifecycle.db:
                self.lifecycle._event('forward_runtime_bound', runtime_binding)

    def close(self):
        self.lifecycle.close()
        self.feed.close()

    def _snapshot(self, as_of, *, months=3):
        from .quarantined_observation import current_binding, snapshot_for_prefix
        prefix = current_binding(self.lifecycle)
        if prefix is not None:
            return snapshot_for_prefix(self.lifecycle, prefix, as_of, months=months,
                                       context_hours=self.config.context_hours)
        start = ((stamp(as_of) - pd.Timedelta(days=30 * months)) if self.spec.window_days is not None
                 else month_start(stamp(as_of)) - pd.DateOffset(months=months)) - self.config.context_hours * HOUR
        return self.feed.snapshot(as_of, start=start,
                                  archive_delay_seconds=self.config.archive_delay_seconds)

    def refresh_training_view(self):
        """Daily immutable training-view receipt; never fits or grants S exposure.

        The actual first successful read precedes anchoring. A failed import
        cannot create an earlier grid, and skipped refresh days are not backfilled.
        Hourly forecasts still use their own actual as-of snapshots.
        """
        if self.spec.window_days is None:
            return None
        from factorlab.protocol import fingerprint

        from .frame_identity import frame_identity
        from .rolling import build_rolling_data
        from .rolling_cadence import RollingCadence
        now = self.lifecycle.now()
        records = [json.loads(r[0]) for r in self.lifecycle.db.execute(
            'SELECT record FROM operations ORDER BY seq')]
        anchored = any(r['kind'] == 'fit_cadence_anchored' for r in records)
        if anchored:
            anchor = self.lifecycle._cadence_anchor()
            due = RollingCadence().latest_refresh_cutoff(anchor, now)
            found = [r['detail'] for r in records if r['kind'] == 'training_view_refreshed'
                     and r['detail']['refresh_slot'] == due.isoformat()]
            if found:
                if len(found) != 1:
                    raise ValueError('Daily training view was recorded more than once')
                self.lifecycle.verify_training_view(found[0])
                return found[0]
        bars = self._snapshot(now)
        data = build_rolling_data(bars, self.spec)
        if data.train_mask(now).sum() < self.spec.minimum_train_rows:
            raise SnapshotUnavailable('Daily view lacks mature training support; no first anchor created')
        training = data.training_identity(now)
        if not anchored:
            anchor = self.lifecycle._cadence_anchor(create=True)
            due = anchor
        detail = {'version': 'v7-elapsed-training-view-v1', 'plan_id': self.lifecycle.plan.digest,
                  'refresh_slot': due.isoformat(), 'observed_at': now.isoformat(),
                  'source_id': self.feed.identity, 'raw_asof_hash': frame_identity(bars),
                  'training': training, 'fit_permission': False, 'search_exposure': False}
        path = self.lifecycle.directory / 'training_views' / (fingerprint(due.isoformat())+'.json')
        immutable_record(path, detail)
        with self.lifecycle.db:
            self.lifecycle._event('training_view_refreshed', detail)
        self.lifecycle.verify_training_view(detail)
        return detail

    def recipe_for(self, pool=None):
        # Pure complete-recipe identity also serves historical ticket readers;
        # it does not grant work or require an initialized runtime directory.
        from .known_linear_reference import FrozenLinearRecipe
        from .neural_forward import FrozenNeuralRecipe
        if isinstance(pool, FrozenLinearRecipe):
            if pool.rolling_spec.digest != self.spec.digest:
                raise ValueError('Fixed linear reference changed the runtime rolling rule')
            return {'kind': 'v7_known_linear', 'pipeline_id': pool.pipeline_id,
                    'recipe': pool.to_dict(), 'task': 'volatility', 'rolling_hash': self.spec.digest}
        if isinstance(pool, FrozenNeuralRecipe):
            from .neural_forward_contract import neural_pipeline
            return neural_pipeline(pool, self.spec)
        pool = pool or FrozenPool(role='reference', version='K0_v1')
        # Template includes every monthly fitting rule before looking at labels.
        pipeline_id = fingerprint({'pool_template': pool.pipeline_id, 'rolling_hash': self.spec.digest,
                                   'ridge': self.spec.ridge, 'weighting': 'equal_symbol_then_hour'})
        result = {'pipeline_id': pipeline_id, 'pool': pool.to_dict(), 'task': 'volatility',
                'rolling_hash': self.spec.digest}
        if self.spec.window_days is not None:
            from .typed_policy_confirmation import ELAPSED_SCHEDULE
            result['fit_schedule'] = dict(ELAPSED_SCHEDULE)
        return result

    def fit_month(self, cutoff, *, stream='A0', pool=None, authority=None, gates=None):
        from .forward_service_scope import check_service_work
        check_service_work(self)
        from .known_linear_reference import FrozenLinearRecipe
        from .neural_forward import FrozenNeuralRecipe
        if isinstance(pool, FrozenLinearRecipe):
            if gates is not None:
                raise ValueError('Neural gates cannot authorize a fixed linear reference')
            return self._fit_known_month(cutoff, stream=stream, recipe=pool, authority=authority)
        if isinstance(pool, FrozenNeuralRecipe):
            return self._fit_neural_month(cutoff, stream=stream, recipe=pool, authority=authority, gates=gates)
        if gates is not None:
            raise ValueError('Neural gates cannot authorize a convex pool')
        if authority is not None:
            return self._fit_pool_month(cutoff, stream=stream,
                pool=pool if pool is not None else FrozenPool(role='reference', version='K0_v1'),
                authority=authority)
        raise ValueError('Actual shared work authority required; no unmetered legacy pool fit')

    def _fit_pool_month(self, cutoff, *, stream, pool, authority):
        from .monthly_work_authority import MonthlyWorkAuthority
        from .pool_forward import fit_pool_forward
        if type(authority) is not MonthlyWorkAuthority or type(pool) is not FrozenPool:
            raise ValueError('Frozen pool fitting requires its actual shared authority and full pool')
        _, end = self.spec.window(cutoff)
        pipeline = self.recipe_for(pool)
        job = self.lifecycle.plan_fit(end, stream, pipeline)
        from .nominal_algorithm_fit import AlgorithmNumericalFitFailure, record_failed_job
        try:
            self.lifecycle.verify()
            row = self.lifecycle.db.execute('SELECT status,model,failure FROM jobs WHERE id=?', (job,)).fetchone()
            if row[0] in {'verified', 'activated'}:
                model, identity = self.lifecycle.registry.load(row[1], expected_pipeline=pipeline)
                if identity.get('version') != 4 or identity.get('job') != job:
                    raise ValueError('Pool monthly cache lacks its original authority admission')
                if row[2]:
                    from .forward_service_scope import check_service_work
                    check_service_work(self)
                    self.lifecycle.seal_model(job, model, certificate=identity['admission'])
                    check_service_work(self)
                return {'job': job, 'model_id': row[1], 'cached': True}
            result = fit_pool_forward(self, authority, job, pool)
            from .forward_service_scope import check_service_work
            check_service_work(self)
            identifier = self.lifecycle.seal_model(job, result['model'], certificate=result['certificate'])
            check_service_work(self)
            return {'job': job, 'model_id': identifier, 'cached': result['cached'],
                    'train': result['train'], 'ticket_digest': result['ticket_digest']}
        except AlgorithmNumericalFitFailure as error:
            record_failed_job(self,job,error)
            raise
        except MemoryError:
            self.lifecycle.fail_fit(job, 'resource_incomplete')
            raise
        except SnapshotUnavailable:
            self.lifecycle.fail_fit(job, 'data_incomplete')
            raise
        except (ValueError, OSError, RuntimeError, np.linalg.LinAlgError):
            self.lifecycle.fail_fit(job, 'integrity_failure')
            raise

    def _fit_known_month(self, cutoff, *, stream, recipe, authority):
        from .known_linear_forward import fit_known_forward
        from .monthly_work_authority import MonthlyWorkAuthority
        if type(authority) is not MonthlyWorkAuthority:
            raise ValueError('Fixed monthly reference requires the actual shared work authority')
        _, end = self.spec.window(cutoff)
        pipeline = self.recipe_for(recipe)
        job = self.lifecycle.plan_fit(end, stream, pipeline)
        from .nominal_algorithm_fit import AlgorithmNumericalFitFailure, record_failed_job
        try:
            self.lifecycle.verify()
            row = self.lifecycle.db.execute('SELECT status,model,failure FROM jobs WHERE id=?', (job,)).fetchone()
            if row[0] in {'verified', 'activated'}:
                model, identity = self.lifecycle.registry.load(row[1], expected_pipeline=pipeline)
                if identity.get('version') != 3 or identity.get('job') != job:
                    raise ValueError('Fixed monthly cache lacks its original authority admission')
                if row[2]:
                    from .forward_service_scope import check_service_work
                    check_service_work(self)
                    self.lifecycle.seal_model(job, model, certificate=identity['admission'])
                    check_service_work(self)
                return {'job': job, 'model_id': row[1], 'cached': True}
            result = fit_known_forward(self, authority, job, recipe)
            from .forward_service_scope import check_service_work
            check_service_work(self)
            identifier = self.lifecycle.seal_model(job, result['model'], certificate=result['certificate'])
            check_service_work(self)
            return {'job': job, 'model_id': identifier, 'cached': result['cached'],
                    'train': result['train'], 'ticket_digest': result['ticket_digest']}
        except AlgorithmNumericalFitFailure as error:
            record_failed_job(self,job,error)
            raise
        except MemoryError:
            self.lifecycle.fail_fit(job, 'resource_incomplete')
            raise
        except SnapshotUnavailable:
            self.lifecycle.fail_fit(job, 'data_incomplete')
            raise
        except (ValueError, OSError, RuntimeError, np.linalg.LinAlgError):
            self.lifecycle.fail_fit(job, 'integrity_failure')
            raise

    def _fit_neural_month(self, cutoff, *, stream, recipe, authority, gates):
        """Full learner dispatch. Resource trials are never sealed for serving."""
        from .monthly_work_authority import MonthlyWorkAuthority
        from .neural_forward import fit_neural_forward
        from .neural_forward_ticket import ForwardFitTicket, issue_forward_ticket
        if type(authority) is not MonthlyWorkAuthority or not isinstance(gates, dict):
            raise ValueError('Full neural monthly fitting requires the actual authority and registered gates')
        _, end = self.spec.window(cutoff)
        pipeline = self.recipe_for(recipe)
        job = self.lifecycle.plan_fit(end, stream, pipeline)
        self.lifecycle.verify()
        row = self.lifecycle.db.execute('SELECT status,model,failure FROM jobs WHERE id=?', (job,)).fetchone()
        if row[0] in ('verified', 'activated'):
            restored, identity = self.lifecycle.registry.load(row[1], expected_pipeline=pipeline)
            if row[2]:
                from .forward_service_scope import check_service_work
                check_service_work(self)
                self.lifecycle.seal_model(job, restored, certificate=identity['admission'])
                check_service_work(self)
            return {'job': job, 'model_id': row[1], 'cached': True}
        if stream == 'A0':
            purpose = 'A0'
        elif stream == 'A0_policy_trial':
            purpose = 'policy_trial'
        else:
            batch = stream.rpartition(':')[0]
            actual = self.lifecycle.db.execute('SELECT status FROM batches WHERE id=?', (batch,)).fetchone()
            if not actual or actual[0] not in {'frozen_V', 'selected_C'}:
                raise ValueError('Neural monthly fit has no actual protected V/C role')
            purpose = 'V' if actual[0] == 'frozen_V' else 'C'
        try:
            # Read existing canonical work before issuing. A finished worker may
            # be restored after a crash; reserved/running/failed work never gets
            # another deadline, debit or numerical fit on this path.
            matches = []
            root = authority.directory / 'forward_tickets'
            if root.is_dir():
                for path in root.glob('*/ticket.json'):
                    ticket = ForwardFitTicket.load(path)
                    if (ticket.value['context']['plan_id'] == self.lifecycle.plan.digest
                            and ticket.value['context']['job'] == job):
                        matches.append(ticket)
            if len(matches) > 1:
                raise ValueError('Forward job has multiple canonical tickets')
            ticket = matches[0] if matches else issue_forward_ticket(
                self, authority, job, recipe, purpose=purpose, gates=gates)
            canonical_gates = json.loads(json.dumps(gates))
            if isinstance(canonical_gates.get('artifacts', {}).get('probes'), list):
                canonical_gates['artifacts']['probes'] = sorted(canonical_gates['artifacts']['probes'])
            if (ticket.value['recipe'] != json.loads(json.dumps(recipe.to_dict()))
                    or ticket.value['context']['purpose'] != purpose
                    or ticket.value['gates'] != canonical_gates):
                raise ValueError('Neural monthly retry changed its frozen recipe or gate references')
            result = fit_neural_forward(ticket_path=ticket.path)
            if not result['serving_eligible'] or result['context']['job'] != job:
                raise ValueError('Resource or unrelated learner cannot be registered for this job')
            from .forward_service_scope import check_service_work
            check_service_work(self)
            identifier = self.lifecycle.seal_model(job, result['model'], certificate=result['certificate'])
            check_service_work(self)
            return {'job': job, 'model_id': identifier, 'cached': result['cached'],
                    'train': result['snapshot']['training'], 'ticket_digest': ticket.digest}
        except MemoryError:
            self.lifecycle.fail_fit(job, 'resource_incomplete')
            raise
        except (ValueError, OSError, RuntimeError, np.linalg.LinAlgError):
            self.lifecycle.fail_fit(job, 'integrity_failure')
            raise

    def has_registered_neural_completion(self, job, recipe):
        """Positive, independently verified cache probe; never launches work."""
        pipeline = self.recipe_for(recipe)
        for path in self.lifecycle.registry.directory.glob('*/manifest.json'):
            try:
                record = json.loads(path.read_text('utf-8'))
                identity = record['identity']
                if (identity.get('kind') != 'v7_neural_forward' or identity.get('job') != job
                        or identity.get('pipeline') != pipeline):
                    continue
                model, restored = self.lifecycle.registry.load(path.parent.name, expected_pipeline=pipeline)
                if (restored['job'] == job and model.kind == 'v7_neural_forward'
                        and model.pipeline_id == recipe.pipeline_id):
                    return True
            except (ValueError, OSError, KeyError, TypeError):
                # An incomplete/malformed publication is never positive cache
                # evidence and cannot bypass the original attempt ceiling.
                continue
        return False

    def prepare_fit_snapshot(self, job):
        from .forward_fit_snapshot import prepare_forward_snapshot
        return prepare_forward_snapshot(self, job)

    def claim_protocol(self):
        """Read a genuinely bound book; losing its directory never disables it."""
        from .claim_protocol import ClaimProtocol
        bindings = ClaimProtocol._bindings(self.lifecycle)
        if bindings or (self.lifecycle.directory / 'claim_protocol').exists():
            return ClaimProtocol(self.lifecycle)
        return None

    def claim_streams(self, origin, now):
        protocol = self.claim_protocol()
        streams = {}
        if protocol is not None:
            for e in protocol.endpoints.values():
                if stamp(e.start) <= stamp(origin) <= stamp(now) < stamp(e.end):
                    pairs = [(e.reference_stream, e.reference)]
                    if e.slot == 'library_skill':
                        pairs.append((e.candidate_stream, e.candidate))
                    elif e.slot == 'algorithm':
                        from .typed_policy_confirmation import candidate_pipeline
                        pairs.append((e.candidate_stream, candidate_pipeline(
                            self.lifecycle, self.lifecycle.fit_cycle_start(origin), e.reference)))
                    for stream, recipe in pairs:
                        if stream in streams and streams[stream] != recipe:
                            raise ValueError('Claim streams changed their fixed learner')
                        streams[stream] = recipe
        return streams

    def _serving_recipes(self, origin, now):
        """Serving permission from verified frozen windows, never active state.

        Called under the controller lock after Lifecycle.verify(). Windows are
        forecast-origin windows; their existing label/scoring horizons are not
        shortened here. Expired forecasts can still settle independently.
        """
        from .fixed_recipe_oracle import serving_recipes
        fixed = serving_recipes(self.lifecycle, origin, now)
        if fixed is not None:
            return fixed
        allowed = {'A0': None}
        allowed.update(self.claim_streams(origin, now))
        for batch, text, status, selected in self.lifecycle.db.execute(
                'SELECT id,identity,status,selected FROM batches'):
            identity = json.loads(text)
            candidates = []
            if (status == 'frozen_V' and stamp(identity['start']) <= origin <= now < stamp(identity['v_end'])):
                candidates = identity['candidates']
            elif status == 'selected_C' and selected:
                row = self.lifecycle.exposure.db.execute(
                    'SELECT identity,consumed,result FROM confirmations WHERE batch=?', (batch,)).fetchone()
                if row and not row[1] and row[2] is None:
                    confirmation = json.loads(row[0])
                    if (stamp(confirmation['actual_start']) <= origin <= now < stamp(confirmation['end'])):
                        chosen = json.loads(selected)['candidate_id']
                        candidates = [c for c in identity['candidates'] if c['pipeline_id'] == chosen]
            if candidates:
                allowed[f'{batch}:reference'] = identity['reference']
                allowed.update({f'{batch}:{c["pipeline_id"]}': c for c in candidates})
        return allowed

    @staticmethod
    def _assert_serving_recipe(actual, expected):
        # Complexity selects a candidate; it is not an executable fitting rule.
        if expected is not None and actual != {k: v for k, v in expected.items() if k != 'complexity'}:
            raise ValueError('Serving model differs from its frozen pipeline recipe')

    def _activate_ready(self, stream, origin, *, expected_pipeline=None):
        from .nominal_algorithm_fit import numeric_unavailability
        if numeric_unavailability(self,stream,origin,expected_pipeline) is not None:
            return None
        active_cutoff = self.lifecycle.db.execute('SELECT j.cutoff FROM active a JOIN jobs j '
            'ON a.job=j.id WHERE a.stream=?', (stream,)).fetchone()
        row = self.lifecycle.db.execute('SELECT id,pipeline,cutoff FROM jobs WHERE stream=? AND status=? '
            'AND failure IS NULL AND completed<=? AND cutoff<=? AND cutoff>? ORDER BY cutoff DESC LIMIT 1',
            (stream, 'verified', origin.isoformat(), origin.isoformat(),
             active_cutoff[0] if active_cutoff else '')).fetchone()
        active = self.lifecycle.active_model(stream)
        expected = ({k: v for k, v in expected_pipeline.items() if k != 'complexity'}
                    if expected_pipeline is not None else None)
        if row:
            from .quarantined_observation import publication_visible
            old_policy_recipe = (stream == 'A0_policy_trial' and expected is not None
                and json.loads(row[1]) != expected and stamp(row[2]) < self.lifecycle.fit_cycle_start(origin))
            if not old_policy_recipe and publication_visible(self.lifecycle, 'model', row[0], origin) is not False:
                self._assert_serving_recipe(json.loads(row[1]), expected_pipeline)
                from .forward_service_scope import check_service_work
                check_service_work(self)
                self.lifecycle.activate(row[0], origin, expected_generation=active['generation'] if active else 0)
                active = self.lifecycle.active_model(stream)
        if active:
            if (stream == 'A0_policy_trial' and expected is not None
                    and active['pipeline'] != expected and active_cutoff is not None
                    and stamp(active_cutoff[0]) < self.lifecycle.fit_cycle_start(origin)):
                # active_model already verifies the old model against its own
                # original job/admission. It cannot forecast as the newly frozen
                # policy recipe. A missing forecast lets service plan/refit the
                # new month without changing the old activation or generation.
                return None
            self._assert_serving_recipe(active['pipeline'], expected_pipeline)
        return active

    def _settle_available(self, bars):
        """Only pending immutable forecasts; labels use their original future window."""
        now = self.lifecycle.now()
        settled = 0
        # A database cursor does not materialize the entire historical ledger.
        for row in self.lifecycle.forecasts.iter_records():
            key = (row['run'], row['stream'], row['origin'], row['symbol'], row['task'])
            if self.lifecycle.db.execute('SELECT 1 FROM settlement WHERE run=? AND stream=? AND origin=? '
                                        'AND symbol=? AND task=?', key).fetchone():
                continue
            prediction = json.loads(row['prediction'])
            start = stamp(prediction['forecast_window_start'])
            end = start + 12 * HOUR
            if end > now:
                continue
            segment = bars.loc[(bars.symbol == row['symbol']) & (bars.open_time >= start - HOUR)
                               & (bars.open_time < end)].sort_values('open_time')
            expected = pd.date_range(start - HOUR, end, freq='h', inclusive='left')
            if len(segment) != 13 or not pd.DatetimeIndex(segment.open_time).equals(expected):
                continue
            receipt = max(end + pd.Timedelta(seconds=self.spec.availability_delay_seconds),
                          segment.available_at.max())
            from .quarantined_observation import current_binding
            prefix = current_binding(self.lifecycle)
            basis = 'actual_receipt' if prefix is None else 'quarantined_finite_observation'
            if receipt > now or (segment.availability_basis != basis).any():
                continue
            target = float(np.sqrt(np.square(np.diff(np.log(segment.close.to_numpy()))).sum()))
            revision = fingerprint(segment.revision.tolist())
            from .forward_service_scope import check_service_work
            check_service_work(self)
            settled += self.lifecycle.settle(stream=row['stream'], origin=row['origin'], symbol=row['symbol'],
                task='volatility', label=target, maturity=receipt, received_at=receipt,
                label_revision=revision)
        return settled

    def tick(self, *, poll=False, session=None):
        from .forward_service_scope import check_service_work
        check_service_work(self)
        from .operation_inputs import OperationInputs
        # Each tick discards numerical inputs; authorization is always re-read.
        with self.lifecycle.controller_lock(), OperationInputs():
            integrity = self.lifecycle.verify_serving()
            isolated = integrity.get('serving_only_isolated_jobs', [])
            from .quarantined_observation import current_binding
            prefix = current_binding(self.lifecycle)
            if prefix is not None and poll:
                raise ValueError('Enrolled finite observation cannot poll unregistered market rows')
            poll_results = self.feed.poll(session=session, pages=self.config.polling_pages) if poll else []
            now = self.lifecycle.now()
            bars = self._snapshot(now)
            settled = self._settle_available(bars)
            # Entire current source is available; labels are deliberately absent.
            history = build_feature_history(bars)
            from .nominal_algorithm_producer import capture_tick, finish_tick, skip_nonfinite_output
            nominal_capture = capture_tick(self, bars, history, isolated=bool(isolated))
            new = skipped = 0
            for symbol, group in bars.groupby('symbol', sort=False):
                latest = group.iloc[-1]
                basis = 'actual_receipt' if prefix is None else 'quarantined_finite_observation'
                if latest.availability_basis != basis or not latest.prediction_origin:
                    continue
                origin = stamp(latest.available_at)
                if ((now - origin).total_seconds() > self.lifecycle.plan.prediction_max_lag_seconds
                        or stamp(latest.open_time) != origin.floor('h') - HOUR
                        or origin.ceil('h') <= now):
                    skipped += 1
                    continue  # No historical reconstruction of missing live forecasts.
                # A damaged candidate cannot authorize a new activation, claim
                # forecast, fit, endpoint or research attempt. Keep only the
                # already admitted operational A0; missing formal streams keep
                # their original nominal denominators and receive no credit.
                recipes = {'A0': None} if isolated else self._serving_recipes(origin, now)
                for stream, recipe in recipes.items():
                    # verify_serving already recovered and verified immutable receipts.
                    # An ordinary repeated poll must not infer the same forecast again.
                    # Nominal capture keeps its full numerical comparison path.
                    if prefix is None and nominal_capture is None and self.lifecycle.forecasts.db.execute(
                            'SELECT 1 FROM forecasts WHERE run=? AND stream=? AND origin=? AND symbol=? AND task=?',
                            (self.lifecycle.plan.digest, stream, origin.isoformat(), symbol, 'volatility')).fetchone():
                        skipped += 1
                        continue
                    model = (self.lifecycle.active_model(stream) if isolated else
                             self._activate_ready(stream, origin, expected_pipeline=recipe))
                    if not model:
                        skipped += 1
                        continue
                    predictions, support = model['model'].predict_history(history)
                    mask = ((history.metadata.symbol == symbol)
                            & (history.metadata.feature_open_time == latest.open_time)).to_numpy()
                    positions = np.flatnonzero(mask)
                    if len(positions) != 1 or not support[positions[0]]:
                        skipped += 1
                        continue
                    if skip_nonfinite_output(nominal_capture, stream, symbol, predictions[positions[0]]):
                        skipped += 1
                        continue
                    check_service_work(self)
                    try:
                        new += self.lifecycle.record_forecast(stream=stream, origin=origin,
                            symbol=symbol, task='volatility', prediction=float(predictions[positions[0]]),
                            received_at=latest.received_at if prefix is None else latest.available_at,
                            feature_open_time=latest.open_time,
                            forecast_window_start=origin.ceil('h'), model_id=model['model_id'],
                            pipeline_id=model['pipeline']['pipeline_id'])
                    except ForecastExpired:
                        if prefix is not None or nominal_capture is not None:
                            raise  # Missing nominal/quarantined output remains a failure.
                        skipped += 1  # No late forecast, backfill or replacement issuance.
            finish_tick(nominal_capture)
            return {'at': now.isoformat(), 'poll': poll_results, 'new_predictions': new,
                    'new_settlements': settled, 'skipped': skipped,
                    'formal_gate_ready': False if isolated else self.lifecycle.calibration_ready(),
                    **({'serving_only_isolated_jobs': isolated} if isolated else {}),
                    'live_trading': False}

    def status(self):
        return {'source': self.feed.verify(), 'lifecycle': self.lifecycle.verify(),
                'scope': 'actual_forward_prediction_and_factor_research'}
