"""Bounded forward supervisor; no GPT, promotion authority or process killing.

The runtime owns receipts, snapshots, cold fitting and activation. This layer
only schedules those operations and frozen V/C endpoints. Resource observations
are preflight checks, not proof that OS CPU/RAM limits have been installed.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from dataclasses import asdict, dataclass

import pandas as pd
from filelock import FileLock

from factorlab.protocol import fingerprint, project_path
from factorlab.time_utils import month_start

from .joint_pool import FrozenPool
from .v7_lifecycle import canonical, stamp
from .v7_storage import immutable_record


def parse_frozen_recipe(runtime, recipe):
    """Complete executable recipe parsing; this grants no fitting permission."""
    if isinstance(recipe, dict) and recipe.get('kind') in {'v7_known_linear', 'v7_neural_forward'}:
        if recipe['kind'] == 'v7_known_linear':
            from .known_linear_reference import FrozenLinearRecipe
            learner = FrozenLinearRecipe.from_dict(recipe.get('recipe', {}))
        else:
            from .neural_forward import FrozenNeuralRecipe
            learner = FrozenNeuralRecipe.from_dict(recipe.get('recipe', {}))
        if {k: v for k, v in recipe.items() if k != 'complexity'} != runtime.recipe_for(learner):
            raise ValueError('Frozen recipe differs from its runtime fitting policy')
        return learner
    if not isinstance(recipe, dict) or 'pool' not in recipe:
        raise ValueError('Service requires a complete frozen pool recipe')
    pool = FrozenPool.from_dict(recipe['pool'])
    if {k: v for k, v in recipe.items() if k != 'complexity'} != runtime.recipe_for(pool):
        raise ValueError('Frozen recipe differs from runtime task/fit policy')
    return pool


@dataclass(frozen=True)
class ServiceConfig:
    session_id: str = 'initial'
    wall_seconds: int = 3600
    max_steps: int = 120
    tick_seconds: int = 60
    max_fits_per_step: int = 1
    max_fit_attempts: int = 3
    retry_seconds: int = 60
    poll: bool = True

    def __post_init__(self):
        if (not isinstance(self.session_id, str) or not self.session_id
                or len(self.session_id) > 64 or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in self.session_id)
                or type(self.poll) is not bool):
            raise ValueError('Invalid supervisor identity/poll policy')
        for name, bounds in {'wall_seconds': (1, 7200), 'max_steps': (1, 7200),
                             'tick_seconds': (1, 300), 'max_fits_per_step': (1, 4),
                             'max_fit_attempts': (1, 3), 'retry_seconds': (1, 3600)}.items():
            value = getattr(self, name)
            if type(value) is not int or not bounds[0] <= value <= bounds[1]:
                raise ValueError(f'Invalid bounded supervisor {name}')


class HostGuard:
    """Read-only, injectable host observations; never stop another service."""
    def __init__(self, *, max_rss_mib=768, minimum_available_mib=768, inspect=None):
        if min(max_rss_mib, minimum_available_mib) <= 0:
            raise ValueError('Positive resource limits required')
        self.max_rss_mib, self.minimum_available_mib = max_rss_mib, minimum_available_mib
        self.inspect = inspect or self._inspect

    @staticmethod
    def _inspect():
        import psutil
        processes = []
        for process in psutil.process_iter(['pid', 'cmdline', 'name', 'status']):
            if process.info['pid'] == os.getpid():
                continue
            command = process.info['cmdline']
            if not command and 'python' in (process.info['name'] or '').lower():
                # A just-exited owned child can remain in the process table
                # until its parent reaps it. Only confirmed termination is
                # harmless; an unreadable living service still blocks work.
                try:
                    status = process.status()
                except psutil.NoSuchProcess:
                    continue
                if status == psutil.STATUS_ZOMBIE:
                    continue
                error = OSError('Cannot inspect another living Python service')
                error.guard_process = {'pid': process.info['pid'], 'status': status}
                raise error
            if command:
                processes.append({'pid': process.info['pid'], 'command': ' '.join(command)})
        return {'rss_mib': psutil.Process().memory_info().rss / 2**20,
                'available_mib': psutil.virtual_memory().available / 2**20,
                'processes': processes}

    def __call__(self):
        try:
            observation = self.inspect()
            if any(not math.isfinite(observation[key]) or observation[key] < 0
                   for key in ('rss_mib', 'available_mib')):
                raise ValueError('Invalid host resource observation')
            blocked = [p['pid'] for p in observation['processes']
                       if any(marker in p['command'].lower() for marker in
                              ('research-long80', 'server_long80', 'server-long80', '/long80/', '\\long80\\'))]
            reasons = (['legacy_long80_active'] if blocked else [])
            if observation['rss_mib'] >= self.max_rss_mib:
                reasons.append('rss_limit')
            if observation['available_mib'] < self.minimum_available_mib:
                reasons.append('insufficient_available_memory')
            return {'allowed': not reasons, 'reasons': reasons, 'blocking_pids': blocked,
                    'rss_mib': observation['rss_mib'], 'available_mib': observation['available_mib'],
                    'hard_limits_verified': False}
        except Exception as exc:  # noqa: BLE001 - an unreadable injected host check must fail closed
            return {'allowed': False, 'reasons': ['host_inspection_failed'],
                    'error_type': type(exc).__name__, 'hard_limits_verified': False,
                    'unreadable_process': getattr(exc, 'guard_process', None)}


class ForwardService:
    """One independently constructed ForwardRuntime per service instance.

    A session has a durable lifetime budget. Reopening the same session keeps
    attempts and charges unobserved time after an interrupted step. Creating a
    different session is an explicit operator action, never an automatic retry.
    """
    def __init__(self, runtime, *, config=None, guard=None, stop_requested=None,
                 monotonic=None, sleeper=None, neural_authority=None, neural_gates=None, work_authority=None):
        from .work_lease import FrozenWorkLease
        self._native_construction_origin = FrozenWorkLease.issue(300)
        self.runtime, self.lifecycle = runtime, runtime.lifecycle
        self.config = config or ServiceConfig()
        self.guard = guard or HostGuard()
        self.stop_requested = stop_requested or (lambda: False)
        self.monotonic, self.sleeper = monotonic or time.monotonic, sleeper or time.sleep
        self.directory = project_path(runtime.directory / 'service' / self.config.session_id)
        prior_identity=self.directory/'identity.json'
        if prior_identity.exists():
            saved_origin=json.loads(prior_identity.read_text('utf-8'))['native_execution_origin']
            self._native_construction_origin=FrozenWorkLease.from_dict(saved_origin)
        identity = {'config': asdict(self.config), 'lifecycle': self.lifecycle.plan.digest,
                    'native_clock_protocol':'v7-continuous-native-service-ceiling-v1',
                    'native_execution_origin':self._native_construction_origin.to_dict()}
        protocol = runtime.claim_protocol() if hasattr(runtime, 'claim_protocol') else None
        self._claim_protocol_digest = protocol.digest if protocol is not None else None
        identity['claim_protocol_digest'] = self._claim_protocol_digest
        self.work_authority = work_authority if work_authority is not None else neural_authority
        self._work_binding = None
        if self.work_authority is not None:
            from .monthly_work_authority import MonthlyWorkAuthority
            from .neural_forward_ticket import _bind_authority
            if type(self.work_authority) is not MonthlyWorkAuthority:
                raise TypeError('Service requires its actual shared monthly work authority')
            if neural_authority is not None and (neural_authority.directory != self.work_authority.directory
                    or neural_authority.manifest_hash != self.work_authority.manifest_hash):
                raise ValueError('Service channels cannot use separate monthly wallets')
            _bind_authority(runtime, self.work_authority)
            self._work_binding = {'directory': str(self.work_authority.directory),
                                  'manifest_hash': self.work_authority.manifest_hash}
            identity['work_authority'] = self._work_binding
        self._work_binding_digest = fingerprint(self._work_binding)
        from .algorithm_validation import AlgorithmValidation
        from .v7_lifecycle import Lifecycle
        from .model_copy_isolation import needs_serving_check
        serving_integrity = {}
        if isinstance(self.lifecycle, Lifecycle) and needs_serving_check(self.lifecycle):
            with self.lifecycle.controller_lock():
                serving_integrity = self.lifecycle.verify_serving()
        self.algorithm_protocols = ([] if serving_integrity.get('serving_only_isolated_jobs') else
                                    AlgorithmValidation.discover(runtime, self.work_authority))
        # Freeze the trusted dispatch rule, not the set of future S discoveries.
        # Actual new prospective contracts may join this same running session;
        # each original contract remains independently immutable and prepaid.
        identity['algorithm_validation_dispatch'] = 'v7-prospective-shadow-dispatch-v1'
        self.neural_authority = self.work_authority if neural_gates is not None else neural_authority
        self.neural_gates = None
        self._neural_binding = None
        if neural_authority is not None or neural_gates is not None:
            from .monthly_work_authority import MonthlyWorkAuthority
            from .neural_forward_ticket import _bind_authority
            if type(self.neural_authority) is not MonthlyWorkAuthority or not isinstance(neural_gates, dict):
                raise ValueError('Neural service requires its actual authority and fixed gate map')
            if any(not isinstance(k, str) or len(k) != 64 or any(c not in '0123456789abcdef' for c in k)
                   or not isinstance(v, dict) for k, v in neural_gates.items()):
                raise ValueError('Neural gate map must bind complete pipeline digests')
            _bind_authority(runtime, self.neural_authority)
            self.neural_gates = json.loads(canonical(neural_gates))
            self._neural_binding = {'authority_directory': str(self.neural_authority.directory),
                'authority_manifest_hash': self.neural_authority.manifest_hash, 'gates': self.neural_gates}
            identity['neural'] = self._neural_binding
        self._neural_identity_digest = fingerprint(self._neural_binding)
        self._owner_lock = FileLock(str(runtime.directory / '.supervisor-owner.lock'), timeout=0)
        self._owner_lock.acquire()
        try:
            identity_path = self.directory / 'identity.json'
            state_path = self.directory / 'state.sqlite'
            initialize = not self.directory.exists()
            if not initialize and not identity_path.is_file():
                raise ValueError('Supervisor identity missing from existing namespace; cannot reset')
            if not initialize and not state_path.is_file():
                raise ValueError('Supervisor state database missing; cannot reset original budget')
            # The identity is the durable initialization intent. A crash after
            # this write may fail closed, but can never reopen an empty budget.
            # Acquire ownership first so a competing unopened session leaves no
            # false initialization intent behind.
            if initialize:
                self.directory.mkdir(parents=True, exist_ok=False)
            immutable_record(identity_path, identity)
            self._open_state(initialize=initialize)
            self._sync_algorithm_protocols()
        except BaseException:
            if hasattr(self, 'db'):
                self.db.close()
            self._owner_lock.release()
            raise

    def _open_state(self, *, initialize=False):
        path = self.directory / 'state.sqlite'
        self.db = (sqlite3.connect(path) if initialize else
                   sqlite3.connect(path.as_uri() + '?mode=rw', uri=True))
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        if initialize:
            self.db.executescript('''CREATE TABLE checkpoint
                (id INTEGER PRIMARY KEY, value TEXT, hash TEXT);
                CREATE TABLE events (seq INTEGER PRIMARY KEY, value TEXT, hash TEXT);''')
        row = self.db.execute('SELECT value,hash FROM checkpoint WHERE id=1').fetchone()
        if row:
            self.state = json.loads(row[0])
            if fingerprint(self.state) != row[1]:
                raise ValueError('Supervisor checkpoint integrity failure')
            self._verify(row[1])
        else:
            if not initialize or self.db.execute('SELECT COUNT(*) FROM events').fetchone()[0]:
                raise ValueError('Supervisor checkpoint missing from existing event journal')
            self.state = {'steps': 0, 'spent_seconds': 0., 'active_since': None,
                          'fit_attempts': {}, 'freeze_requests': {}, 'last_result': None}
            self._save('created')
        if self.state['active_since'] is not None:
            elapsed = (self.lifecycle.now() - stamp(self.state['active_since'])).total_seconds()
            if elapsed < 0:
                raise ValueError('Supervisor clock moved backward during interruption')
            self.state['spent_seconds'] += elapsed
            self.state['active_since'] = None
            self._save('interrupted_time_charged')

    def close(self, *, close_runtime=True):
        self.db.close()
        self._owner_lock.release()
        if close_runtime:
            self.runtime.close()

    def _verify(self, state_hash):
        previous, last = '0' * 64, None
        for sequence, text, digest in self.db.execute('SELECT seq,value,hash FROM events ORDER BY seq'):
            record = json.loads(text)
            if (record['sequence'] != sequence or record['previous'] != previous
                    or fingerprint(record) != digest):
                raise ValueError('Supervisor event chain integrity failure')
            previous, last = digest, record['state_hash']
        if last != state_hash:
            raise ValueError('Supervisor checkpoint differs from event head')

    def _save(self, event):
        digest = fingerprint(self.state)
        with self.db:
            row = self.db.execute('SELECT seq,hash FROM events ORDER BY seq DESC LIMIT 1').fetchone()
            record = {'sequence': row[0] + 1 if row else 0,
                      'previous': row[1] if row else '0' * 64, 'state_hash': digest,
                      'event': event, 'at': self.lifecycle.now().isoformat()}
            self.db.execute('INSERT INTO events VALUES (?,?,?)',
                            (record['sequence'], canonical(record), fingerprint(record)))
            self.db.execute('INSERT OR REPLACE INTO checkpoint VALUES (1,?,?)',
                            (canonical(self.state), digest))

    def _pool(self, recipe):
        # Keep the full encoder/head/seeds/readout recipe across scheduling;
        # extracting just its pool would silently schedule another learner.
        return parse_frozen_recipe(self.runtime, recipe)

    def queue_batch(self, *, freeze_at=None, **request):
        """Queue already reviewed, complete recipes; no GPT/search is called here."""
        request = json.loads(canonical(request))
        key = request['batch']
        existing = self.state['freeze_requests'].get(key)
        if existing:
            if (existing['request'] != request or (freeze_at is not None and
                    existing['freeze_at'] != stamp(freeze_at).isoformat())):
                raise ValueError('Queued freeze identity cannot change')
            return False
        now = self.lifecycle.now()
        when = now if freeze_at is None else stamp(freeze_at)
        if when < now or stamp(request['start']) < when:
            raise ValueError('Batch freeze must be prospectively scheduled')
        for recipe in [request['reference'], *request['candidates']]:
            self._pool(recipe)
        record = {'freeze_at': when.isoformat(), 'request': request, 'status': 'pending'}
        self.state['freeze_requests'][key] = record
        self._save('batch_queued')
        return True

    def _freeze_due(self, now, actions):
        for item in self.state['freeze_requests'].values():
            if item['status'] != 'pending' or stamp(item['freeze_at']) > now:
                continue
            if stamp(item['request']['start']) < now:
                item['status'] = 'missed_prospective_start'
            else:
                self.lifecycle.freeze_batch(**item['request'])
                item['status'] = 'frozen'
            actions.append({'kind': 'freeze', 'batch': item['request']['batch'], 'status': item['status']})
            self._save('batch_freeze_handled')

    def _endpoints(self, now, actions):
        self._assert_claim_protocol()
        for batch, text, status in list(self.lifecycle.db.execute('SELECT id,identity,status FROM batches')):
            identity = json.loads(text)
            if status == 'frozen_V' and now >= stamp(identity['v_end']):
                result = self.lifecycle.select_validation(batch)
                actions.append({'kind': 'validation', 'batch': batch, 'selected': result['selected']})
                status = 'selected_C' if result['selected'] else 'concluded_no_candidate'
            if status == 'reading_V':
                result = self.lifecycle.recover_validation(batch)
                if result is None:
                    actions.append({'kind': 'validation', 'batch': batch, 'status': 'interrupted_without_committed_result'})
                else:
                    status = 'selected_C' if result['selected'] else 'concluded_no_candidate'
                    actions.append({'kind': 'validation', 'batch': batch, 'status': 'recovered_committed_result'})
            if status != 'selected_C':
                continue
            if not self.lifecycle.confirmation_execution_ready():
                actions.append({'kind': 'confirmation', 'batch': batch, 'status': 'waiting_for_calibration'})
                continue
            row = self.lifecycle.exposure.db.execute(
                'SELECT identity,consumed,result FROM confirmations WHERE batch=?', (batch,)).fetchone()
            if row is None:
                if now > stamp(identity['v_end']) + pd.Timedelta(hours=24):
                    actions.append({'kind': 'confirmation', 'batch': batch, 'status': 'missed_activation_window'})
                    continue
                self.lifecycle.reserve_selected_confirmation(batch)
                row = self.lifecycle.exposure.db.execute(
                    'SELECT identity,consumed,result FROM confirmations WHERE batch=?', (batch,)).fetchone()
            confirmation = json.loads(row[0])
            if now >= stamp(confirmation['scoring_cutoff']):
                if row[1] and not row[2]:
                    result = self.lifecycle.recover_confirmation(batch)
                    actions.append({'kind': 'confirmation', 'batch': batch,
                        'status': 'recovered_committed_result' if result is not None else 'consumed_without_result'})
                else:
                    result = self.lifecycle.conclude_confirmation(batch)
                    actions.append({'kind': 'confirmation', 'batch': batch, 'result': result})

        protocol = self.runtime.claim_protocol() if hasattr(self.runtime, 'claim_protocol') else None
        if protocol is not None:
            for slot in protocol.endpoints:
                result = protocol.finalize(slot)
                actions.append({'kind': 'claim_endpoint', 'slot': slot, 'result': result})

    def _plan_month(self, now):
        self._assert_claim_protocol()
        from .nominal_algorithm_warmboot import plan_original_jobs
        plan_original_jobs(self, now)
        if self.lifecycle.plan.training_window_days is not None:
            from .v7_feed import SnapshotUnavailable
            try:
                self.runtime.refresh_training_view()
            except SnapshotUnavailable:
                return  # No first cutoff or fit permission before actual data readiness.
        cutoff = self.lifecycle.due_fit_cutoff()
        k0 = self.runtime.recipe_for()
        recipe = self.lifecycle.adoption_pipeline(cutoff, k0)
        # Selection metadata (for example complexity) is not part of the
        # runtime's executable fit recipe. Validate then use its exact contract.
        self.lifecycle.plan_fit(cutoff, 'A0', self.runtime.recipe_for(self._pool(recipe)))
        if hasattr(self.runtime, 'claim_streams'):
            for stream, frozen in self.runtime.claim_streams(now, now).items():
                self.lifecycle.plan_fit(cutoff, stream, self.runtime.recipe_for(self._pool(frozen)))
        for batch, text, status, selected in self.lifecycle.db.execute(
                'SELECT id,identity,status,selected FROM batches'):
            identity = json.loads(text)
            candidates = []
            if status == 'frozen_V' and stamp(identity['start']) <= now < stamp(identity['v_end']):
                candidates = identity['candidates']
            elif status == 'selected_C' and selected and now < stamp(identity['c_end']):
                reservation = self.lifecycle.exposure.db.execute(
                    'SELECT identity FROM confirmations WHERE batch=?', (batch,)).fetchone()
                if reservation and now >= stamp(json.loads(reservation[0]).get('actual_start', identity['v_end'])):
                    chosen = json.loads(selected)['candidate_id']
                    candidates = [r for r in identity['candidates'] if r['pipeline_id'] == chosen]
            if candidates:
                for stream, frozen in [(f'{batch}:reference', identity['reference']),
                                      *[(f'{batch}:{c["pipeline_id"]}', c) for c in candidates]]:
                    self.lifecycle.plan_fit(cutoff, stream, self.runtime.recipe_for(self._pool(frozen)))

    def _fit_pending(self, now, actions, deadline, *, fit_allowance=None):
        self._assert_claim_protocol()
        self._sync_algorithm_protocols()
        rows = self.lifecycle.db.execute('SELECT id,cutoff,stream,pipeline,status,failure FROM jobs '
            'WHERE status NOT IN (?,?) OR (status=? AND failure IS NOT NULL) ORDER BY cutoff DESC,stream',
            ('verified', 'activated', 'verified')).fetchall()
        fitted = 0
        allowance = self.config.max_fits_per_step if fit_allowance is None else fit_allowance
        for job, cutoff, stream, text, job_status, failure in rows:
            if stream.startswith('neural-resource:'):
                from .forward_resource_work import verify_resource_plan
                verify_resource_plan(self.lifecycle, job, self._pool(json.loads(text)))
                continue
            owners = [p.owns_job(job) for p in self.algorithm_protocols]
            owners = [owner for owner in owners if owner is not None]
            if len(owners) > 1:
                raise ValueError('Algorithm job has multiple typed owners')
            if owners:
                actions.append({'kind': 'algorithm_owned_job', **owners[0]})
                continue
            if stream.startswith('algorithm-validation:'):
                raise ValueError('Unregistered algorithm namespace cannot enter ordinary fit handler')
            if self.monotonic() >= deadline:
                break
            if fitted >= allowance or stamp(cutoff) > now:
                continue
            prior = self.state['fit_attempts'].get(job, {'count': 0, 'last_at': None})
            pool = self._pool(json.loads(text))
            from .known_linear_reference import FrozenLinearRecipe
            from .neural_forward import FrozenNeuralRecipe
            neural = isinstance(pool, FrozenNeuralRecipe)
            known = isinstance(pool, FrozenLinearRecipe)
            retry_blocked = (prior['count'] >= self.config.max_fit_attempts
                or (prior['last_at'] is not None and
                    (now - stamp(prior['last_at'])).total_seconds() < self.config.retry_seconds))
            if failure == 'integrity_failure':
                continue
            recovery = (job_status in {'verified', 'activated'} or
                neural and self.runtime.has_registered_neural_completion(job, pool))
            if retry_blocked and not recovery:
                continue
            # Reserve the complete original 300s work envelope plus the same
            # fixed publication margin used by the typed algorithm branch.
            # Positive completed-artifact recovery cannot issue new work.
            if not recovery and self.monotonic() + 302 > deadline:
                actions.append({'kind': 'fit', 'stream': stream, 'status': 'insufficient_service_headroom'})
                continue
            kwargs = {}
            if known or (isinstance(pool, FrozenPool) and self.work_authority is not None):
                current = ({'directory': str(self.work_authority.directory),
                    'manifest_hash': self.work_authority.manifest_hash} if self.work_authority is not None else None)
                if current is None or fingerprint(current) != self._work_binding_digest:
                    raise ValueError('Convex forward work requires the unchanged shared monthly authority')
                kwargs = {'authority': self.work_authority}
            if neural:
                current = ({'authority_directory': str(self.neural_authority.directory),
                    'authority_manifest_hash': self.neural_authority.manifest_hash,
                    'gates': self.neural_gates} if self.neural_authority is not None else None)
                if fingerprint(current) != self._neural_identity_digest:
                    raise ValueError('Neural service static authority/gate identity changed')
                gates = self.neural_gates.get(pool.pipeline_id) if self.neural_gates is not None else None
                neural_authority = self.neural_authority
                if gates is None:
                    from .forward_gate_resolver import ForwardGatesPending, resolve_forward_gates
                    try:
                        resolved = resolve_forward_gates(self.runtime, self.work_authority, job, pool)
                    except ForwardGatesPending as exc:
                        if str(exc) == 'waiting_original_same_recipe_full_forward_resource_workload':
                            from .forward_resource_work import produce_resource_work
                            resource_key = 'resource:'+job
                            if resource_key not in self.state['fit_attempts']:
                                if self.monotonic() + 302 > deadline:
                                    actions.append({'kind': 'full_forward_resource', 'stream': stream,
                                                    'status': 'insufficient_service_headroom'})
                                    continue
                                self.state['fit_attempts'][resource_key] = {'count': 1, 'last_at': now.isoformat()}
                                self._save('full_forward_resource_attempt_reserved')
                                # The durable intent itself may be slow. Keep
                                # its consumed attempt, but never start a fit
                                # whose whole unchanged lease no longer fits.
                                if self.monotonic() + 302 > deadline:
                                    actions.append({'kind': 'full_forward_resource', 'stream': stream,
                                                    'status': 'insufficient_service_headroom'})
                                    continue
                                result = produce_resource_work(self.runtime, self.work_authority, job, pool)
                                from .forward_service_scope import check_service_work
                                check_service_work(self.runtime)
                                fitted += 1
                                actions.append({'kind': 'full_forward_resource', 'stream': stream, **result})
                        actions.append({'kind': 'fit', 'stream': stream, 'status': str(exc),
                                        'pipeline_id': pool.pipeline_id})
                        continue
                    anchor = self.directory/'dynamic_neural_gates'/(job+'.json')
                    if anchor.exists():
                        original = json.loads(anchor.read_text('utf-8'))
                        if any(original.get(k) != resolved.get(k) for k in (
                                'version', 'job', 'pipeline_id', 'source_hash', 'authority_manifest_hash', 'gates')):
                            raise ValueError('Original dynamic neural gate references changed')
                    else:
                        immutable_record(anchor, resolved)
                    gates, neural_authority = resolved['gates'], self.work_authority
                # Runtime restores only original completed work. Re-entry here
                # never authorizes a replacement ticket or a renewed lease.
                kwargs = {'authority': neural_authority, 'gates': gates}
            if not recovery and self.monotonic() + 302 > deadline:
                actions.append({'kind': 'fit', 'stream': stream, 'status': 'insufficient_service_headroom'})
                continue
            if not recovery:
                self.state['fit_attempts'][job] = {'count': prior['count'] + 1, 'last_at': now.isoformat()}
            self._save('fit_recovery_started' if recovery else 'fit_attempt_reserved')
            if not recovery and self.monotonic() + 302 > deadline:
                actions.append({'kind': 'fit', 'stream': stream, 'status': 'insufficient_service_headroom'})
                continue
            fitted += 1
            try:
                result = self.runtime.fit_month(cutoff, stream=stream, pool=pool, **kwargs)
                from .forward_service_scope import check_service_work
                check_service_work(self.runtime)
                actions.append({'kind': 'fit', 'stream': stream, 'status': 'verified_waiting_for_receipt', **result})
            except (MemoryError, ValueError, OSError, RuntimeError) as exc:
                if isinstance(exc, RuntimeError) and not neural:
                    raise
                row = self.lifecycle.db.execute('SELECT failure FROM jobs WHERE id=?', (job,)).fetchone()
                code = row[0] if row and row[0] else ('resource_incomplete' if isinstance(exc, MemoryError) else 'data_incomplete')
                if not row or not row[0]:
                    self.lifecycle.fail_fit(job, code)
                actions.append({'kind': 'fit', 'stream': stream, 'status': code, 'error_type': type(exc).__name__})

    def _sync_algorithm_protocols(self):
        from .algorithm_validation import AlgorithmValidation
        from .model_copy_isolation import has_isolations
        if has_isolations(self.lifecycle):
            self.lifecycle.verify_serving()
            # Existing queue anchors/checkpoints are retained unchanged. No
            # discovery, queue action or confirmation permission is returned.
            self.algorithm_protocols = []
            return []
        current = AlgorithmValidation.discover(self.runtime, self.work_authority)
        refs = {}
        now = self.lifecycle.now()
        for protocol in current:
            value = protocol.identity
            if (stamp(value['frozen_at']) > now or stamp(value['frozen_at']) > stamp(value['start'])
                    or value['source_id'] != self.lifecycle.plan.source_id
                    or value['authority_manifest_hash'] != self.work_authority.manifest_hash):
                raise ValueError('New algorithm contract is future-dated, retrospective or outside authority scope')
            refs[protocol.digest] = {'directory': str(protocol.directory), 'identity_hash': protocol.digest,
                'recipes_hash': fingerprint(value['recipes']), 'source_id': value['source_id'],
                'authority_manifest_hash': value['authority_manifest_hash'],
                'frozen_at': value['frozen_at'], 'start': value['start'], 'V_end': value['V_end'],
                'qualifier_start': value['qualifier_start'], 'qualifier_end': value['qualifier_end']}
        seen = self.state.get('algorithm_protocols', {})
        if type(seen) is not dict or any(refs.get(key) != old for key, old in seen.items()):
            raise ValueError('Previously observed algorithm ID/recipe/window was removed or replaced')
        anchors = self.directory / 'algorithm_protocols'
        if {p.stem for p in anchors.glob('*.json')} != set(seen):
            raise ValueError('Seen algorithm anchors/checkpoint differ; cannot forget or replay registration')
        for key, old in seen.items():
            if json.loads((anchors/(key+'.json')).read_text('utf-8')) != old:
                raise ValueError('Seen algorithm immutable anchor changed')
        self.algorithm_protocols = current
        if refs != seen or 'algorithm_protocols' not in self.state:
            for key in sorted(set(refs)-set(seen)):
                immutable_record(anchors/(key+'.json'), refs[key])
            self.state['algorithm_protocols'] = refs
            self._save('algorithm_protocols_appended')
        return current

    def _algorithm_step(self, actions, deadline, *, defer_fits=False):
        current = self._sync_algorithm_protocols()
        used = 0
        for protocol in current:
            if not defer_fits and used < self.config.max_fits_per_step and self.monotonic()+302 <= deadline:
                from .prospective_resource_work import pending_prospective_resource, run_prospective_resource
                for plan in pending_prospective_resource(protocol):
                    if used >= self.config.max_fits_per_step or self.monotonic()+302 > deadline:
                        break
                    key = 'prospective_resource:'+plan['job']
                    if key in self.state['fit_attempts']:
                        continue
                    self.state['fit_attempts'][key] = {'count': 1, 'last_at': self.lifecycle.now().isoformat()}
                    self._save('prospective_full_forward_resource_reserved')
                    used += 1
                    result = run_prospective_resource(protocol, plan['job'])
                    actions.append({'kind': 'prospective_full_forward_resource', 'job': plan['job'], **result})
            protocol.resolve_fit_gates()
            allowance = 0 if defer_fits else min(3, max(0, self.config.max_fits_per_step-used))
            pending = protocol.pending_work()[:allowance] if self.monotonic()+302 <= deadline else []
            # Service/campaign attempt projections are reserved before producer
            # preparation. Crashes never erase the canonical wallet reservation.
            for plan in pending:
                key = 'algorithm:' + plan['work_key']
                if key not in self.state['fit_attempts']:
                    self.state['fit_attempts'][key] = {'count': 1, 'last_at': self.lifecycle.now().isoformat()}
                    self._save('algorithm_fit_attempt_reserved')
                else:
                    # A service attempt with no canonical work means it crashed
                    # before authority reserve. It still cannot regain execution.
                    allowance = 0
            used += len(pending)
            # Only this call's durably projected attempts may execute. A clock
            # boundary or newly registered gates must not mint work between
            # pending_work and the controller's own current-state inspection.
            allowance = min(allowance, len(pending))
            result = protocol.step(deadline_monotonic=deadline, max_fits=allowance)
            actions.append({'kind': 'algorithm_controller', 'validation_id': protocol.digest, **result})
        return used

    def _assert_claim_protocol(self):
        protocol = self.runtime.claim_protocol() if hasattr(self.runtime, 'claim_protocol') else None
        if (protocol.digest if protocol is not None else None) != self._claim_protocol_digest:
            raise ValueError('Service cannot add or replace its frozen six-month claim protocol')

    def step(self, *, session=None, defer_fits=False):
        from .forward_service_scope import _service_step_scope
        from .work_lease import FrozenWorkLease
        # Native sample precedes lock, STOP, guard and all this-step setup IO.
        # It grants no work and is discarded for a pure STOP/read-only boundary.
        original = self._native_construction_origin
        with FileLock(str(self.directory / '.lock'), timeout=0):
            if self.stop_requested() or (self.directory / 'STOP').exists():
                return {'status': 'stopped', 'steps': self.state['steps']}
            if self.state['steps'] >= self.config.max_steps or self.state['spent_seconds'] >= self.config.wall_seconds:
                return {'status': 'budget_exhausted', 'steps': self.state['steps']}
            with _service_step_scope(self, origin=original):
                return self._step_owned(session=session, defer_fits=defer_fits)

    def _step_owned(self, *, session=None, defer_fits=False):
        from .forward_service_scope import check_service_work
        guard = self.guard()
        check_service_work(self.runtime)
        if guard.get('allowed') is not True:
            return {'status': 'blocked_by_host_guard', 'guard': guard}
        started = self.monotonic()
        self.state['active_since'] = self.lifecycle.now().isoformat()
        self.state['steps'] += 1
        self._save('step_started')
        actions = []
        try:
            check_service_work(self.runtime)
            from .nominal_algorithm_policy import prepare_tick, after_tick
            nominal_tick = prepare_tick(self, actions)
            tick = self.runtime.tick(poll=self.config.poll, session=session)
            after_tick(nominal_tick, self)
            check_service_work(self.runtime)
            if tick.get('serving_only_isolated_jobs'):
                result = {'status': 'stepped', 'at': tick['at'], 'tick': tick,
                          'actions': [{'status': 'model_copy_isolated_new_work_and_claims_blocked'}],
                          'guard': guard, 'formal_claim_authority': 'unavailable_integrity_fault'}
                self.state['last_result'] = result
                return result
            with self.lifecycle.controller_lock():
                now = self.lifecycle.now()
                if nominal_tick is None:
                    self._freeze_due(now, actions)
                    self._endpoints(now, actions)
                self._plan_month(now)
                deadline = started + self.config.wall_seconds - self.state['spent_seconds']
                used = self._algorithm_step(actions, deadline, defer_fits=defer_fits)
                if not defer_fits and self.state['spent_seconds'] + self.monotonic() - started < self.config.wall_seconds:
                    self._fit_pending(self.lifecycle.now(), actions, deadline, fit_allowance=max(0, self.config.max_fits_per_step-used))
            result = {'status': 'stepped', 'at': now.isoformat(), 'tick': tick, 'actions': actions,
                      'guard': guard, 'formal_claim_authority': 'lifecycle_certificate_only'}
            self.state['last_result'] = result
            return result
        finally:
            self.state['spent_seconds'] += max(0., self.monotonic() - started)
            self.state['active_since'] = None
            self._save('step_finished')

    def run(self, *, session=None):
        """Return on stop/guard/budget; sleep between every pair of ticks."""
        started = self.monotonic()
        last = None
        while self.monotonic() - started < self.config.wall_seconds:
            last = self.step(session=session)
            if last['status'] != 'stepped':
                return last
            remaining = min(self.config.wall_seconds - (self.monotonic() - started),
                            self.config.wall_seconds - self.state['spent_seconds'])
            if remaining <= 0:
                break
            slept = self.monotonic()
            self.state['active_since'] = self.lifecycle.now().isoformat()
            self._save('sleep_started')
            try:
                self.sleeper(min(self.config.tick_seconds, remaining))
            finally:
                self.state['spent_seconds'] += max(0., self.monotonic() - slept)
                self.state['active_since'] = None
                self._save('sleep_finished')
        return {'status': 'invocation_wall_limit', 'last_step': last}


# Public CLI name; ForwardService remains a descriptive library alias.
RollingService = ForwardService
