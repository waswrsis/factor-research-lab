"""Original paid, multi-month fixed-estimand replicas on independent feeds.

These replicas do not run provider search and never replace the whole-policy
oracle. Every fit still uses the actual known/pool forward ticket, monthly
wallet and physical lease. Their availability is measured, not asserted equal
in distribution to a loaded adaptive service.
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from contextlib import closing
from pathlib import Path

import pandas as pd
from filelock import FileLock

from factorlab.protocol import fingerprint
from .calibration_runtime_contract import _original_operations
from .calibration_trial_context import _no_redirect
from .full_calibration_registry import QuarantinedTrialPermit, _read
from .v7_storage import immutable_record
from .v7_lifecycle import stamp

VERSION = 'v7-original-fixed-recipe-oracle-v2'
BOUND = 'fixed_oracle_controller_bound'
SLOTS = ('library_skill', 'factor1', 'factor2', 'factor3')
STREAMS = ('oracle_truth:reference', 'oracle_truth:candidate')


def _anchor(permit, kind, detail):
    # Existing SQLite only. Physical controller events do not borrow the
    # logical observation clock (which has no first prefix at enrollment).
    from types import SimpleNamespace
    from .quarantined_observation import _event, _physical
    path=permit.paths['runtime']/'lifecycle.sqlite'
    with closing(sqlite3.connect(path.as_uri()+'?mode=rw',uri=True)) as db, db:
        db.execute('PRAGMA synchronous=FULL'); db.execute('BEGIN IMMEDIATE')
        _event(SimpleNamespace(db=db),kind,detail,_physical())


def _mirror(permit):
    return _no_redirect(permit.registry.directory/'fixed_oracle_steps'/permit.trial['trial_id'],
                        permit.registry.directory)


def _write_step(permit, index, kind, value):
    name=f'{index:05d}-{kind}.json'
    for directory in (permit.paths['root']/'fixed_oracle/steps',_mirror(permit)):
        immutable_record(_no_redirect(directory/name,permit.registry.directory),value)


def fixed_enrollment_preflight(permit):
    """Read original selection BEFORE any replica namespace/wallet is created."""
    if type(permit) is not QuarantinedTrialPermit or permit.trial.get('estimand_slot') not in SLOTS:
        raise TypeError('Original registered nonadaptive oracle replica required')
    permit.registry.verify()
    from .calibration_oracle_evidence import _frozen_estimands
    validation = QuarantinedTrialPermit(permit.registry, permit.registry.identity_for(
        permit.trial['cell'], permit.trial['index']))
    frozen, specs = _frozen_estimands(validation)
    spec = next(s for s in specs if s.slot == permit.trial['estimand_slot'])
    if spec.candidate is None:
        raise ValueError('Original nonselection has no fixed oracle learner')
    for pipeline in (spec.reference, spec.candidate):
        if pipeline.get('kind') != 'v7_known_linear' and 'pool' not in pipeline:
            raise ValueError('Fixed five-claim replica supports original known/pool recipes only')
    return {'validation': validation.trial, 'frozen_estimands_hash': fingerprint(frozen),
            'original_estimand': spec.to_dict(), 'original_estimand_hash': spec.digest}


def _selected(permit):
    from .calibration_oracle_evidence import OracleClaimSpec
    binding = fixed_enrollment_preflight(permit)
    spec = OracleClaimSpec(**binding['original_estimand'])
    return binding, spec


def _identity(permit, generator, selection, controller):
    return {'version':VERSION, 'trial':permit.trial, 'selection':selection,
        'generator_directory':str(generator.directory), 'generator_digest':generator.digest,
        'source_hours':generator.law.hours,
        'source_files':permit.registry.identity['source_files'],
        'support':list(range(generator.law.support_size)), 'streams':list(STREAMS),
        'original_controller_hash':fingerprint(controller.identity),
        'service_seconds':controller.schedule.service['wall_seconds'],
        'service_clock':'continuous_original_dual_clock_including_idle_no_restart',
        'completion_policy':'subsequent_locked_observation_of_predecessor_no_self_completion',
        'fit_policy':'one_attempt_per_stream_month_original_paid_cold_forward',
        'availability':'original_released_source_and_actual_physical_publication',
        'resource_distribution_equivalence':'not_established_by_independent_replica',
        'formal_gate_eligible':False}


def _bound_identity(lifecycle, permit=None):
    from .quarantined_observation import _active_read_scope
    scope=_active_read_scope()
    scoped=scope is not None and scope['directory']==str(lifecycle.directory)
    if scoped:
        if scope['object'].permit.trial.get('estimand_slot') is None:
            return None
        if 'fixed_oracle_identity' in scope:
            return scope['fixed_oracle_identity']
    if permit is None:
        from .quarantine_execution import permit_for
        permit = permit_for(lifecycle)
    if permit is None or permit.trial.get('estimand_slot') is None:
        return None
    permit.verify(lifecycle)
    identity = _read(_no_redirect(permit.paths['root']/'fixed_oracle/identity.json',permit.registry.directory))
    binding, spec = _selected(permit)
    operations = _original_operations(lifecycle)
    anchors = [o['detail'] for o in operations if o['kind'] == BOUND]
    if (identity['version'] != VERSION or identity['trial'] != permit.trial
            or identity['selection'] != binding
            or identity['source_files'] != permit.registry.identity['source_files']
            or identity['support'] != list(range(permit.registry.plan.oracle_paths))
            or identity['streams'] != list(STREAMS)
            or identity['formal_gate_eligible'] is not False
            or anchors != [{'identity_hash': fingerprint(identity)}]
            or _read(permit.registry.directory/'fixed_oracle_plans'/(permit.trial['trial_id']+'.json')) != identity):
        raise ValueError('Original fixed oracle identity, selection or lifecycle anchor changed')
    from .full_trial_controller import FullTrialController
    original = QuarantinedTrialPermit(permit.registry, binding['validation'])
    controller = FullTrialController.inspect(original)
    from .full_mechanism_contract import _generator
    generator=_generator(identity['generator_directory'],permit.registry.plan.acceptance)
    generator.verify_registry(permit.registry)
    if identity!=_identity(permit,generator,binding,controller):
        raise ValueError('Fixed oracle changed original source, full window or service resource ceiling')
    if scoped:
        scope['fixed_oracle_identity']=identity
    return identity


def serving_recipes(lifecycle, origin, now):
    identity = _bound_identity(lifecycle)
    if identity is None:
        return None
    spec = identity['selection']['original_estimand']
    if not stamp(spec['start']) <= stamp(origin) <= stamp(now) < stamp(spec['end']):
        return {}
    return {stream: {k:v for k,v in spec[role].items() if k != 'complexity'}
            for stream, role in zip(STREAMS, ('reference', 'candidate'))}


def job_context(lifecycle, job, pipeline):
    identity = _bound_identity(lifecycle)
    if identity is None:
        return None
    row = lifecycle.db.execute('SELECT cutoff,stream,pipeline FROM jobs WHERE id=?',(job,)).fetchone()
    if row is None or row[1] not in STREAMS or json.loads(row[2]) != pipeline:
        raise ValueError('Fixed replica cannot authorize another stream or caller recipe')
    role = ('reference','candidate')[STREAMS.index(row[1])]
    spec = identity['selection']['original_estimand']
    selected = {k:v for k,v in spec[role].items() if k != 'complexity'}
    from .quarantined_observation import job_binding
    prefix = job_binding(lifecycle, job)
    cutoff = stamp(row[0])
    warmup = stamp(spec['start']).normalize().replace(day=1)-pd.DateOffset(months=1)
    if (prefix is None or selected != pipeline or not warmup <= cutoff < stamp(spec['end'])
            or cutoff != cutoff.normalize().replace(day=1)
            or not cutoff <= stamp(prefix['logical_at']) < stamp(spec['end'])):
        raise ValueError('Fixed replica job lost original source/window/recipe binding')
    operations=_original_operations(lifecycle)
    planned=next(o for o in operations if o['kind']=='fit_planned' and o['detail']['job']==job)
    intents=[o for o in operations if o['kind']=='fixed_oracle_attempt' and o['seq']<planned['seq']]
    if not intents:
        raise ValueError('Fixed recipe work lacks an original prepaid controller opportunity')
    anchor=intents[-1]['detail']; index=anchor['index']
    from .quarantine_execution import permit_for
    permit=permit_for(lifecycle)
    attempt=_read(permit.paths['root']/'fixed_oracle/steps'/f'{index:05d}-attempt.json')
    if (fingerprint(attempt)!=anchor['attempt_hash']
            or _read(_mirror(permit)/f'{index:05d}-attempt.json')!=attempt
            or attempt['identity_hash']!=fingerprint(identity) or index!=prefix['hour']
            or any(o['kind']=='fixed_oracle_step' and o['seq']<planned['seq']
                   and o['detail']['attempt_hash']==anchor['attempt_hash'] for o in operations)):
        raise ValueError('Fixed recipe fit was not planned inside its original controller opportunity')
    return {'role':'original_fixed_oracle', 'fixed_oracle_hash':fingerprint(identity),
            'original_estimand_hash':identity['selection']['original_estimand_hash'],
            'estimand_slot':identity['trial']['estimand_slot'], 'oracle_role':role}


def _open_runtime(permit):
    from .v7_cli import _original_trial_runtime
    return _original_trial_runtime(permit)


def _supervisor(attempt):
    """Existing service ceiling, never a spendable model work lease."""
    from datetime import timedelta
    from types import SimpleNamespace
    from .work_lease import FrozenWorkLease, _utc
    original=FrozenWorkLease.from_dict(attempt['physical_lease'])
    seconds=attempt['remaining_service_seconds']
    if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<=0:
        raise ValueError('Original remaining service ceiling required')
    fields=original.to_dict()
    fields['wall_deadline']=(_utc(original.wall_started_at)+timedelta(seconds=seconds)).isoformat()
    fields['monotonic_deadline']=original.monotonic_started+seconds
    return SimpleNamespace(**fields)


def _fit_headroom(attempt):
    """Refuse to open a complete new child lease beyond the service ceiling."""
    from .work_lease import _observe, _evaluate, _utc
    supervisor=_supervisor(attempt)
    observed=_observe(supervisor)
    if not _evaluate(supervisor,observed).allowed:
        raise ValueError('Original service budget closed before paid fit')
    remaining=min(_utc(supervisor.wall_deadline).timestamp()-observed['wall'],
                  supervisor.monotonic_deadline-observed['monotonic'])
    if remaining<301.:
        raise ValueError('Original service budget cannot fund complete 300s fit plus dispatch headroom')


def _charged_seconds(permit, index, attempt, completion):
    """An unconfirmed opportunity never earns refundable service time.

    Diagnostic only: dispatch uses the first lease's absolute deadline, never
    a sum of these records or terminal file timestamps.
    """
    name=f'{index:05d}-witness.json'
    local=permit.paths['root']/'fixed_oracle/steps'/name
    mirror=_mirror(permit)/name
    if not local.is_file() or not mirror.is_file():
        return attempt['remaining_service_seconds']
    witness=_read(local)
    if witness!=_read(mirror) or not _confirmed(attempt,completion,witness):
        return attempt['remaining_service_seconds']
    from .work_lease import _utc
    return max(witness['observed']['wall']-_utc(_supervisor(attempt).wall_started_at).timestamp(),
               witness['observed']['monotonic']-_supervisor(attempt).monotonic_started)


def _confirmed(attempt, completion, witness):
    """A later locked invocation observes a predecessor that already ended."""
    from .work_lease import _evaluate
    return (completion is not None and completion['state']=='awaiting_successor_witness'
            and witness is not None and witness.get('state')=='confirmed'
            and witness.get('attempt_hash')==fingerprint(attempt)
            and witness.get('completion_hash')==fingerprint(completion)
            and _evaluate(_supervisor(attempt),witness['observed'],completion['physical_finish']).allowed)


def _witness_predecessor(permit, index, row):
    """Called only after acquiring the canonical pump lock in a NEW invocation.

    This records the already ended predecessor, not its own publication tail.
    A late/lost-parent observation is immutable negative evidence, never a reset.
    """
    from .work_lease import _observe, _evaluate
    attempt,completion=row['attempt'],row['completion']
    observed=_observe(_supervisor(attempt))
    allowed=(completion is not None and completion['state']=='awaiting_successor_witness'
             and _evaluate(_supervisor(attempt),observed,completion['physical_finish']).allowed)
    if allowed:
        name=f'{index:05d}-commit.json'
        allowed=all(max(p.stat().st_mtime_ns,p.stat().st_ctime_ns)<=observed['wall']*1e9
                    for p in (permit.paths['root']/'fixed_oracle/steps'/name,_mirror(permit)/name))
    witness={'attempt_hash':fingerprint(attempt),'completion_hash':fingerprint(completion),
             'observed':observed,'state':'confirmed' if allowed else 'failed_or_unknown_consumed'}
    _write_step(permit,index,'witness',witness)
    _anchor(permit,'fixed_oracle_witness',{'index':index,'witness_hash':fingerprint(witness)})
    return witness


class FixedRecipeOracle:
    @classmethod
    def freeze(cls, permit, runtime, generator):
        from .v7_forward import ForwardRuntime
        from .finite_market_oracle import FiniteGenerator, bind_draw
        from .full_trial_controller import FullTrialController
        from .neural_forward_ticket import open_existing_authority
        from .quarantined_observation import QuarantinedObservation
        if type(runtime) is not ForwardRuntime or type(generator) is not FiniteGenerator:
            raise TypeError('Actual runtime and original finite generator required')
        selection, spec = _selected(permit)
        permit.verify(runtime.lifecycle); generator.verify_registry(permit.registry)
        if (runtime.directory != permit.paths['runtime'] or generator.law.digest != permit.trial['scenario_digest']
                or generator.law.support_size != permit.registry.plan.oracle_paths):
            raise ValueError('Fixed replica cannot borrow another runtime or population')
        original = QuarantinedTrialPermit(permit.registry, selection['validation'])
        controller = FullTrialController.inspect(original)
        identity = _identity(permit,generator,selection,controller)
        directory = permit.paths['root']/'fixed_oracle'
        with FileLock(str(directory)+'.lock', timeout=0):
            if directory.exists():
                existing = cls.inspect(permit)
                if existing.identity != identity:
                    raise ValueError('Fixed oracle cannot replace an original recipe/window')
                return existing
            ops = _original_operations(runtime.lifecycle)
            if any(o['kind'] not in {'forward_runtime_bound','monthly_work_authority_bound',
                                     'full_calibration_quarantine_enrolled'} for o in ops):
                raise ValueError('Fixed controller must freeze before source exposure or work')
            if open_existing_authority(permit.paths['authority']).verify()['reservations']:
                raise ValueError('Fixed replica cannot borrow consumed work')
            bind_draw(generator, permit)
            immutable_record(directory/'identity.json',identity)
            immutable_record(permit.registry.directory/'fixed_oracle_plans'/(permit.trial['trial_id']+'.json'),identity)
            with runtime.lifecycle.db:
                runtime.lifecycle._event(BOUND, {'identity_hash':fingerprint(identity)})
            QuarantinedObservation.bind(runtime,generator,permit)
        return cls.inspect(permit)

    @classmethod
    def inspect(cls, permit):
        if type(permit) is not QuarantinedTrialPermit or permit.trial.get('estimand_slot') not in SLOTS:
            raise TypeError('Original fixed replica permit required')
        obj=cls.__new__(cls); obj.permit=permit
        obj.directory=permit.paths['root']/'fixed_oracle'
        from .calibration_oracle_evidence import _lifecycle
        with _lifecycle(permit.paths['runtime']) as lc:
            obj.identity=_bound_identity(lc,permit)
            from .quarantined_observation import QuarantinedObservation
            observer=QuarantinedObservation.from_lifecycle(lc)
            if observer is None or observer.generator.digest != obj.identity['generator_digest']:
                raise ValueError('Original fixed replica observation is missing or changed')
        obj.verify()
        return obj

    def verify(self):
        from .calibration_oracle_evidence import _lifecycle
        from .quarantined_observation import QuarantinedObservation
        from .work_lease import _evaluate, FrozenWorkLease
        rows=[]; expected=set(); previous='0'*64
        with _lifecycle(self.permit.paths['runtime']) as lc:
            if _bound_identity(lc,self.permit) != self.identity:
                raise ValueError('Original fixed controller changed')
            observer=QuarantinedObservation.from_lifecycle(lc)
            operations=_original_operations(lc)
            anchors=[o['detail'] for o in operations if o['kind']=='fixed_oracle_step']
            intents=[o['detail'] for o in operations if o['kind']=='fixed_oracle_attempt']
            witnesses=[o['detail'] for o in operations if o['kind']=='fixed_oracle_witness']
            files={p.name for p in (self.directory/'steps').glob('*.json')}
            mirrors={p.name for p in _mirror(self.permit).glob('*.json')}
            if files!=mirrors:
                raise ValueError('Fixed oracle original dual step anchor is missing')
            for name in files:
                local=_no_redirect(self.directory/'steps'/name,self.permit.registry.directory)
                mirror=_no_redirect(_mirror(self.permit)/name,self.permit.registry.directory)
                if _read(local)!=_read(mirror):
                    raise ValueError('Fixed oracle original dual step bytes changed')
            i=0
            while (self.directory/'steps'/f'{i:05d}-attempt.json').is_file():
                path=self.directory/'steps'/f'{i:05d}-attempt.json'; expected.add(path.name)
                attempt=_read(path)
                if (attempt['index']!=i or attempt['previous']!=previous
                        or attempt['identity_hash']!=fingerprint(self.identity)
                        or attempt['next_hour']!=i):
                    raise ValueError('Fixed oracle original step order or identity changed')
                if (attempt['remaining_service_seconds']!=self.identity['service_seconds']
                        or (rows and attempt['physical_lease']!=rows[0]['attempt']['physical_lease'])):
                    raise ValueError('Fixed oracle renewed its original absolute service ceiling')
                lease=_supervisor(attempt)
                if not _evaluate(lease,attempt['opportunity_started'],
                        rows[-1]['witness']['observed'] if rows else None).allowed:
                    raise ValueError('Fixed oracle opened work outside its original service ceiling')
                finish=self.directory/'steps'/f'{i:05d}-result.json'
                result=_read(finish) if finish.is_file() else None
                commit_path=self.directory/'steps'/f'{i:05d}-commit.json'
                completion=_read(commit_path) if commit_path.is_file() else None
                if completion is not None:
                    expected.add(commit_path.name)
                    if (result is None or completion['result_hash']!=fingerprint(result)
                            or completion['attempt_hash']!=fingerprint(attempt)
                            or completion['elapsed_seconds']<result['elapsed_seconds']):
                        raise ValueError('Fixed oracle post-IO witness changed its original completion')
                if result is not None:
                    expected.add(finish.name)
                    if (result['attempt_hash']!=fingerprint(attempt)
                            or not math.isfinite(result['elapsed_seconds']) or result['elapsed_seconds']<0):
                        raise ValueError('Fixed oracle original completion changed')
                    # The lease is a physical supervisor witness only; fitting
                    # receives its independent original <=300s paid ticket.
                    if completion is not None and completion['state']=='awaiting_successor_witness' and not _evaluate(
                            lease,completion['physical_finish'],result['physical_finish']).allowed:
                        raise ValueError('Fixed oracle step completed outside its physical ceiling')
                    prefix=result.get('observation_prefix')
                    if prefix is not None:
                        observer.verify_binding(prefix)
                        if prefix['hour']!=i:
                            raise ValueError('Fixed oracle reordered its original observation')
                    elif result['state']=='complete':
                        raise ValueError('Completed fixed oracle step lost its source prefix')
                    deadline=stamp(lease.wall_deadline).value
                    if completion is not None and completion['state']=='awaiting_successor_witness' and any(
                            max(p.stat().st_mtime_ns,p.stat().st_ctime_ns)>=deadline
                            for p in (path,finish,commit_path,*(_mirror(self.permit)/p.name for p in (path,finish,commit_path)))):
                        raise ValueError('Fixed controller final publication exceeded physical budget')
                witness_path=self.directory/'steps'/f'{i:05d}-witness.json'
                witness=_read(witness_path) if witness_path.is_file() else None
                if witness is not None:
                    expected.add(witness_path.name)
                    if (witness.get('attempt_hash')!=fingerprint(attempt)
                            or witness.get('completion_hash')!=fingerprint(completion)):
                        raise ValueError('Fixed oracle successor witness changed predecessor')
                    if witness['state']=='confirmed':
                        if not _confirmed(attempt,completion,witness):
                            raise ValueError('Fixed oracle successor witness is late or invalid')
                        if any(max(p.stat().st_mtime_ns,p.stat().st_ctime_ns)>witness['observed']['wall']*1e9
                               for p in (commit_path,_mirror(self.permit)/commit_path.name)):
                            raise ValueError('Fixed oracle successor observation preceded original publication')
                        if any(max(p.stat().st_mtime_ns,p.stat().st_ctime_ns)>=stamp(lease.wall_deadline).value
                               for p in (witness_path,_mirror(self.permit)/witness_path.name)):
                            raise ValueError('Fixed oracle successor witness published after original ceiling')
                rows.append({'attempt':attempt,'result':result,'completion':completion,'witness':witness})
                previous=fingerprint(rows[-1]); i+=1
                if not _confirmed(attempt,completion,witness): break
            if files!=expected or anchors!=[{'attempt_hash':fingerprint(r['attempt']),
                    'result_hash':fingerprint(r['result']) if r['result'] is not None else None}
                    for r in rows if r['result'] is not None]:
                raise ValueError('Original fixed step files differ from lifecycle anchors')
            if intents!=[{'index':i,'attempt_hash':fingerprint(r['attempt'])} for i,r in enumerate(rows)]:
                raise ValueError('Fixed oracle lost its original durable attempt intent; no reset')
            if witnesses!=[{'index':i,'witness_hash':fingerprint(r['witness'])}
                           for i,r in enumerate(rows) if r['witness'] is not None]:
                raise ValueError('Fixed oracle lost its original successor witness anchor; no reset')
            actual_order=[(o['kind'],o['detail']) for o in operations if o['kind'] in
                          {'fixed_oracle_attempt','fixed_oracle_step','fixed_oracle_witness'}]
            expected_order=[]
            for i,row in enumerate(rows):
                expected_order.append(('fixed_oracle_attempt',{'index':i,'attempt_hash':fingerprint(row['attempt'])}))
                if row['result'] is not None:
                    expected_order.append(('fixed_oracle_step',{'attempt_hash':fingerprint(row['attempt']),
                                                               'result_hash':fingerprint(row['result'])}))
                if row['witness'] is not None:
                    expected_order.append(('fixed_oracle_witness',{'index':i,'witness_hash':fingerprint(row['witness'])}))
            if actual_order!=expected_order:
                raise ValueError('Fixed oracle successor witness preceded original completion')
            committed=sum(r['result'] is not None and r['result'].get('observation_prefix') is not None for r in rows)
            count=observer.inspect(lc)['events']
            if count not in ({committed,committed+1} if rows and rows[-1]['result'] is None else {committed}):
                raise ValueError('Fixed oracle skipped or orphaned an original observation opportunity')
        return rows

    def pump(self):
        """One original release, at most one cold fit; no caller time/recipe."""
        from .quarantined_observation import QuarantinedObservation, current_binding
        from .finite_market_oracle import publish_due
        from .neural_forward_ticket import open_existing_authority
        from .v7_service import parse_frozen_recipe
        from .work_lease import FrozenWorkLease, _observe, _evaluate
        with FileLock(str(self.directory)+'.pump.lock',timeout=0):
            started=time.monotonic()
            first_path=self.directory/'steps/00000-attempt.json'
            # Read-before-verify is not permission. It avoids issuing even an
            # unused replacement start; verify authenticates the original below.
            initial_lease=(FrozenWorkLease.from_dict(_read(first_path)['physical_lease'])
                           if first_path.is_file() else FrozenWorkLease.issue(300))
            rows=self.verify()
            if any(r['result'] is None or r['completion'] is None or
                   r['completion']['state']!='awaiting_successor_witness' for r in rows):
                return {'state':'interrupted_consumed','retry_allowed':False,'formal_gate_eligible':False}
            if rows and rows[-1]['witness'] is None:
                # A separate locked invocation observes the predecessor AFTER
                # its return/lock-release tail. This invocation only confirms:
                # no data release, new fit, or new service origin is allowed.
                witness=_witness_predecessor(self.permit,len(rows)-1,rows[-1])
                return {'state':'predecessor_confirmed' if witness['state']=='confirmed'
                        else 'failed_or_unknown_consumed','retry_allowed':False,
                        'formal_gate_eligible':False}
            if any(not _confirmed(r['attempt'],r['completion'],r['witness']) for r in rows):
                return {'state':'interrupted_consumed','retry_allowed':False,'formal_gate_eligible':False}
            lease=(FrozenWorkLease.from_dict(rows[0]['attempt']['physical_lease']) if rows else initial_lease)
            supervisor=_supervisor({'physical_lease':lease.to_dict(),
                                   'remaining_service_seconds':self.identity['service_seconds']})
            observation=_observe(supervisor)
            if not _evaluate(supervisor,observation,rows[-1]['witness']['observed'] if rows else None).allowed:
                return {'state':'service_budget_exhausted','retry_allowed':False,'formal_gate_eligible':False}
            spec=self.identity['selection']['original_estimand']
            if rows and stamp(rows[-1]['result']['observation_prefix']['logical_at']) >= stamp(spec['end'])+pd.Timedelta(hours=36):
                return {'state':'original_window_exhausted','formal_gate_eligible':False}
            if len(rows)>=self.identity['source_hours']:
                return {'state':'original_source_exhausted','formal_gate_eligible':False}
            attempt={'index':len(rows),'previous':fingerprint(rows[-1]) if rows else '0'*64,
                'identity_hash':fingerprint(self.identity),'next_hour':len(rows),
                'physical_lease':lease.to_dict(),
                'opportunity_started':observation,
                # Same first lease + same full ceiling, not a new remainder.
                'remaining_service_seconds':self.identity['service_seconds']}
            # Prepay the entire service opportunity before opening/publishing
            # source data. An orphan consumes it and can never be retried.
            _write_step(self.permit,len(rows),'attempt',attempt)
            _anchor(self.permit,'fixed_oracle_attempt',{'index':len(rows),'attempt_hash':fingerprint(attempt)})
            runtime=None; prefix=None; result=None
            try:
                runtime=_open_runtime(self.permit)
                observer=QuarantinedObservation.from_lifecycle(runtime.lifecycle)
                state=observer.inspect(runtime.lifecycle)
                spec=self.identity['selection']['original_estimand']
                cutoff_end=stamp(spec['end'])+pd.Timedelta(hours=36)
                if state['logical_at'] is not None and stamp(state['logical_at'])>=cutoff_end:
                    result={'state':'window_exhausted'}
                    return result
                if state['events']>=observer.generator.law.hours:
                    result={'state':'source_exhausted'}
                    return result
                publish_due(observer.generator,self.permit,max_new_receipts=1)
                observer.advance(runtime)
                prefix=current_binding(runtime.lifecycle)
                # One bounded step can fit at most one stream. Both streams
                # receive actual activation delay; no first-hour backfill.
                from .quarantined_observation import _original_observation_read_scope
                # Tick-only reuse. Paid fitting below opens and closes its OWN
                # scope inside its original 300s ticket; no authorization check
                # is deferred past that work lease's final publication.
                with _original_observation_read_scope(runtime.lifecycle):
                    tick=runtime.tick()
                now=runtime.lifecycle.now(); cutoff=now.normalize().replace(day=1)
                first=stamp(spec['start']).normalize().replace(day=1)-pd.DateOffset(months=1)
                fit=None
                if first<=cutoff<stamp(spec['end']):
                    for stream,role in zip(STREAMS,('reference','candidate')):
                        pipeline={k:v for k,v in spec[role].items() if k!='complexity'}
                        job=runtime.lifecycle.plan_fit(cutoff,stream,pipeline)
                        row=runtime.lifecycle.db.execute('SELECT status,failure FROM jobs WHERE id=?',(job,)).fetchone()
                        if row[0]=='planned' and not row[1]:
                            _fit_headroom(attempt)
                            fit=runtime.fit_month(cutoff,stream=stream,
                                pool=parse_frozen_recipe(runtime,pipeline),
                                authority=open_existing_authority(self.permit.paths['authority']))
                            break
                from .work_lease import _evaluate
                if not _evaluate(_supervisor(attempt),_observe(lease)).allowed:
                    raise ValueError('Original fixed controller service ceiling expired')
                result={'state':'complete','tick':tick,'fit':fit}
                return result
            except BaseException as exc:
                result={'state':'failed_consumed','error_type':type(exc).__name__}
                return result
            finally:
                result=result or {'state':'interrupted_consumed'}
                result.update(attempt_hash=fingerprint(attempt),elapsed_seconds=time.monotonic()-started,
                              physical_finish=_observe(lease),observation_prefix=prefix)
                if not _evaluate(_supervisor(attempt),result['physical_finish'],observation).allowed:
                    result['state']='service_budget_exhausted'
                _write_step(self.permit,len(rows),'result',result)
                if runtime is not None:
                    runtime.close()
                _anchor(self.permit,'fixed_oracle_step',{'attempt_hash':fingerprint(attempt),
                                                        'result_hash':fingerprint(result)})
                from .work_lease import _evaluate
                final_observation=_observe(lease)
                complete=result['state']=='complete' and _evaluate(
                    _supervisor(attempt),final_observation,result['physical_finish']).allowed
                completion={'attempt_hash':fingerprint(attempt),'result_hash':fingerprint(result),
                    'physical_finish':final_observation,'elapsed_seconds':time.monotonic()-started,
                    'state':'awaiting_successor_witness' if complete else 'failed_or_late_consumed'}
                _write_step(self.permit,len(rows),'commit',completion)
                # Terminal witness IO itself cannot produce a timely return
                # after expiry. Preserve both original commits and all charges.
                after_commit=_observe(lease)
                deadline=stamp(_supervisor(attempt).wall_deadline).value
                commits=(self.directory/'steps'/f'{len(rows):05d}-commit.json',
                         _mirror(self.permit)/f'{len(rows):05d}-commit.json')
                complete=complete and _evaluate(_supervisor(attempt),after_commit,final_observation).allowed
                complete=complete and all(max(p.stat().st_mtime_ns,p.stat().st_ctime_ns)<deadline for p in commits)
                # The immutable body remains unchanged; callers see the final
                # post-body, post-ledger-commit, post-close physical outcome.
                if not complete:
                    result['state']='failed_or_late_consumed'
                else:
                    result['state']='awaiting_successor_witness'

    def run(self):
        """Bounded same-issuer execution, including separate successor pumps.

        A one-shot CLI cannot keep the original parent alive between commands.
        This loop preserves it without replacing the first physical deadline,
        retrying a failed opportunity, or allocating additional source/work.
        """
        limit=2*self.identity['source_hours']+1
        last=None
        for count in range(1,limit+1):
            last=self.pump()
            if last['state'] not in {'awaiting_successor_witness','predecessor_confirmed'}:
                return {'state':last['state'],'pump_calls':count,'last':last,
                        'formal_gate_eligible':False}
        return {'state':'invocation_bound_reached','pump_calls':limit,'last':last,
                'formal_gate_eligible':False}

    def report(self):
        from .calibration_oracle_evidence import OracleClaimSpec, reconstruct_claim, verify_raw_claim
        from .neural_forward_ticket import open_existing_authority
        rows=self.verify(); original=self.identity['selection']['original_estimand']
        spec=OracleClaimSpec(**{**original,'reference_stream':STREAMS[0],'candidate_stream':STREAMS[1]})
        wallet=open_existing_authority(self.permit.paths['authority']).verify()
        try:
            result=reconstruct_claim(self.permit.paths['runtime'],spec)
            raw=verify_raw_claim(self.permit.paths['runtime'],result)
        except ValueError as exc:
            if 'not matured' not in str(exc): raise
            result=None; raw=None
        complete=(bool(rows) and all(_confirmed(r['attempt'],r['completion'],r['witness']) for r in rows)
                  and result is not None and result['coverage']==1. and result['pair_coverage']==1.
                  and result['path_risks'] is not None)
        return {'version':VERSION,'trial':self.permit.trial,'identity_hash':fingerprint(self.identity),
            'original_estimand_hash':self.identity['selection']['original_estimand_hash'],
            'original_steps_digest':fingerprint(rows),'original_steps':len(rows),
            'state':'complete' if complete else 'incomplete_consumed_or_pending',
            'claim':result,'raw':raw,'wallet':wallet,'formal_gate_eligible':False,
            'service_clock':'continuous_original_dual_clock_including_idle_no_restart',
            'unconfirmed_opportunities':sum(not _confirmed(r['attempt'],r['completion'],r['witness']) for r in rows),
            'availability_equivalence':'separate_measured_replica_not_proved_same_loaded_service_distribution'}
