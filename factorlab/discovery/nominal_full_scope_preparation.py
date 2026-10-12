"""Once-only metadata enrollment for the registered original full-policy trial.

No feed data, fit, lease, provider, unit launch or calibration credit is created.
All paths/configuration derive from the exact preregistered original full plan.
"""
from __future__ import annotations
from contextlib import closing
from dataclasses import asdict, fields
from pathlib import Path
import importlib
import json
import sqlite3
from factorlab.protocol import fingerprint, project_path

VERSION='nominal-original-fullscope-preparation-v1'
RELATIVE='factorlab/discovery/nominal_full_scope_preparation.py'


def _plain(value):return json.loads(json.dumps(value,sort_keys=True,allow_nan=False))


def _installed(plan):
    from .nominal_full_scope_proof import FullScopeProofPlan
    from .full_calibration_registry import FullTrialRegistry
    from .monthly_work_authority import source_files
    from .evidence import file_hash
    if type(plan)is not FullScopeProofPlan or type(plan.parent)is not FullTrialRegistry:
        raise TypeError('Exact original full proof plan and registry required')
    plan.verify();files=source_files()
    if files!=plan.identity['source_files']or files!=plan.parent.plan.runtime_scope['source_files']:
        raise ValueError('Original registered whole source changed')
    for name in ('nominal_full_scope_preparation','nominal_full_scope_proof',
            'nominal_capacity_preflight','nominal_algorithm_total_risk','nominal_algorithm_policy',
            'full_calibration_registry','v7_forward','v7_lifecycle','v7_feed',
            'monthly_work_authority','calibration_runtime_contract','neural_forward_ticket'):
        module=importlib.import_module('factorlab.discovery.'+name);relative='factorlab/discovery/'+name+'.py'
        if Path(module.__file__).resolve()!=project_path(relative).resolve()or files.get(relative)!=file_hash(module.__file__):
            raise ValueError('Canonical preparation dependency changed: '+name)
    return files


def _capacity(plan):
    from .nominal_algorithm_total_risk import AlgorithmTotalRiskRegistry
    from .nominal_algorithm_policy import NominalAlgorithmPolicy
    from .nominal_capacity_preflight import NominalCapacityPreflight
    policy=NominalAlgorithmPolicy.inspect(AlgorithmTotalRiskRegistry.inspect(plan.parent))
    observation=NominalCapacityPreflight.analyze(policy)
    observation.assert_current();observation.require_not_proven_impossible()
    return observation.report


def _configuration(scope):
    from .v7_forward import ForwardConfig
    from .v7_lifecycle import LifecyclePlan
    from .rolling import RollingSpec
    from .monthly_work_authority import VERSION as AUTHORITY_VERSION
    f=_plain(scope['forward']);l=_plain(scope['lifecycle_policy']);w=_plain(scope['work_budget'])
    if set(f)!={x.name for x in fields(ForwardConfig)}or set(l)!={x.name for x in fields(LifecyclePlan)}-{'source_id','calibration_directory','calibration_binding'}:
        raise ValueError('Full original forward/lifecycle fields required; no defaults or certificate')
    config=ForwardConfig(**{**f,'symbols':tuple(f['symbols'])})
    lifecycle=LifecyclePlan(source_id='validation_only_not_used_for_feed',**{**l,'symbols':tuple(l['symbols'])})
    rolling=RollingSpec(availability_delay_seconds=config.archive_delay_seconds,
                        window_days=config.training_window_days)
    if (_plain(asdict(config))!=f or _plain(asdict(rolling))!=scope['rolling']or rolling.digest!=scope['rolling_hash']
        or l['symbols']!=f['symbols']
        or lifecycle.training_window_days!=config.training_window_days
        or lifecycle.refit_interval_days!=config.refit_interval_days
        or set(w)!={'authority_version','task','universe','limits','monthly_seconds','partition_policy'}
        or w['authority_version']!=AUTHORITY_VERSION or w['universe']!=sorted(f['symbols'])
        or type(w['partition_policy'])is not dict or w['partition_policy'].get('mode')!='standard'):
        raise ValueError('Exact original rolling/universe/standard wallet contract required')
    return config,l,w


def _paths(plan,coordinate):
    from .calibration_trial_context import _no_redirect
    paths=plan.parent.paths(coordinate['cell'],coordinate['index'],oracle_index=coordinate['oracle_index'])
    if coordinate!=plan.parent.identity_for(coordinate['cell'],coordinate['index'],oracle_index=coordinate['oracle_index'])or coordinate.get('estimand_slot')is not None:
        raise ValueError('Only original whole-policy coordinate may prepare')
    roots=(paths['root']/'native-preparation',plan.directory/'preparations'/coordinate['trial_id'])
    for path in (*paths.values(),*roots):_no_redirect(path,plan.parent.directory)
    return paths,roots


def _empty(runtime,authority,*,enrolled):
    from .calibration_runtime_contract import _original_operations
    from .full_calibration_registry import EVENT
    runtime.feed.verify();runtime.lifecycle.verify()
    for table in ('bars','batches','publications'):
        if runtime.feed.db.execute('SELECT COUNT(*) FROM '+table).fetchone()[0]:raise ValueError('Prepared feed is not empty')
    for table in ('jobs','active','batches','settlement'):
        if runtime.lifecycle.db.execute('SELECT COUNT(*) FROM '+table).fetchone()[0]:raise ValueError('Prepared lifecycle has observed work')
    if runtime.lifecycle.forecasts.db.execute('SELECT COUNT(*) FROM forecasts').fetchone()[0]:raise ValueError('Prepared forecast ledger is not empty')
    for table in ('evidence','confirmations','validations'):
        if runtime.lifecycle.exposure.db.execute('SELECT COUNT(*) FROM '+table).fetchone()[0]:raise ValueError('Prepared exposure ledger is not empty')
    allowed={'forward_runtime_bound','monthly_work_authority_bound'}|({EVENT}if enrolled else set())
    if any(o['kind']not in allowed for o in _original_operations(runtime.lifecycle)):
        raise ValueError('Preparation cannot repair prior observations')
    verified=authority.verify()
    if verified['reservations']or verified['events']:raise ValueError('Original wallet has consumed work')
    with closing(sqlite3.connect(authority.database.as_uri()+'?mode=ro',uri=True))as db:
        for table in ('wallets','reservations','partition_wallets','monthly_owners','events'):
            if db.execute('SELECT COUNT(*) FROM '+table).fetchone()[0]:raise ValueError('Original wallet is not empty')
    return verified


def _close_handles(runtime,feed,error=None):
    """Close known DB handles even when an original constructor raises midway."""
    from .v7_forward import ForwardRuntime
    from .v7_feed import ReceiptArchive
    from .v7_lifecycle import Lifecycle
    from .model_registry import ForecastLedger
    from .exposure import ExposureLedger
    classes=(ForwardRuntime,ReceiptArchive,Lifecycle,ForecastLedger,ExposureLedger)
    objects=[v for v in (runtime,feed)if v is not None]
    tb=None if error is None else error.__traceback__
    while tb is not None:
        obj=tb.tb_frame.f_locals.get('self')
        if type(obj)in classes and tb.tb_frame.f_code is type(obj).__init__.__code__:objects.append(obj)
        tb=tb.tb_next
    seen=set();failures=[]
    while objects:
        obj=objects.pop()
        if id(obj)in seen:continue
        seen.add(id(obj))
        for attr in ('feed','lifecycle','forecasts','exposure'):
            child=getattr(obj,attr,None)
            if type(child)in classes:objects.append(child)
        db=getattr(obj,'db',None)
        if type(db)is sqlite3.Connection:
            try:db.close()
            except BaseException as exc:failures.append(exc)
    if failures:
        if error is not None:
            for exc in failures:error.add_note('Preparation handle close failed: '+repr(exc))
        else:raise failures[0]


def prepare_enrollment(plan,cell,index,*,oracle_index=None):
    """Create only fresh empty stores and call the original exact enroll once."""
    from .nominal_full_scope_proof import FullScopeProofPlan,_write_pair,_write
    from .full_calibration_registry import FullTrialRegistry
    from .v7_feed import ReceiptArchive
    from .v7_forward import ForwardRuntime
    from .v7_lifecycle import LifecyclePlan
    from .monthly_work_authority import MonthlyWorkAuthority
    from .calibration_trial_context import _no_redirect
    from filelock import FileLock
    if type(plan)is not FullScopeProofPlan or type(plan.parent)is not FullTrialRegistry:
        raise TypeError('Exact original full proof plan and registry required')
    files=_installed(plan);coordinate=plan.coordinate(cell,index,oracle_index)
    config,lifecycle_fields,work=_configuration(plan.parent.plan.runtime_scope)
    if files!=plan.identity['source_files']or files!=plan.parent.plan.runtime_scope['source_files']:
        raise ValueError('Registered preparation source differs')
    capacity=_capacity(plan);paths,roots=_paths(plan,coordinate)
    run_id=fingerprint({'registry':plan.parent.digest,'trial':coordinate['trial_id']})
    wallet_scope={'run_id':run_id,'task':work['task'],'universe':work['universe']}
    index_path=project_path('work/monthly_work_authority_index')/(fingerprint(wallet_scope)+'.json')
    _no_redirect(index_path,project_path('.'))
    lock=plan.directory/('.preparation-'+coordinate['trial_id']+'.lock')
    with FileLock(str(lock),timeout=0):
        if paths['root'].exists()or any(p.exists()for p in roots)or index_path.exists():
            raise ValueError('Original/partial preparation or wallet already exists; no repair/retry')
        intent={'version':VERSION,'plan_digest':plan.digest,'parent_digest':plan.parent.digest,
            'coordinate':coordinate,'source_files':files,'paths':{k:str(v)for k,v in paths.items()},
            'runtime_scope_hash':fingerprint(plan.parent.plan.runtime_scope),'run_id':run_id,
            'capacity_necessary_only':capacity,'fit_or_lease_credit':False,'metadata_only':True,'retry_allowed':False}
        runtime=feed=None
        try:
            paths['root'].mkdir(parents=True,exist_ok=False)
            for root in roots:root.mkdir(parents=True,exist_ok=False)
            _write_pair(roots,'intent.json',intent)
            feed=ReceiptArchive.__new__(ReceiptArchive)
            ReceiptArchive.__init__(feed,paths['feed'],config.symbols)
            actual_source_id=feed.identity;feed.close()
            lifecycle=LifecyclePlan(source_id=actual_source_id,**{**lifecycle_fields,'symbols':tuple(lifecycle_fields['symbols'])})
            runtime=ForwardRuntime.__new__(ForwardRuntime)
            ForwardRuntime.__init__(runtime,paths['runtime'],paths['feed'],config=config,lifecycle_plan=lifecycle)
            authority=MonthlyWorkAuthority(paths['authority'],run_id=run_id,task=work['task'],universe=work['universe'],
                limits=work['limits'],monthly_seconds=work['monthly_seconds'],partition_policy=work['partition_policy'])
            before=_empty(runtime,authority,enrolled=False)
            permit=plan.parent.enroll(runtime,authority,cell,index,oracle_index=oracle_index)
            verified=permit.verify(runtime.lifecycle)
            after=_empty(runtime,authority,enrolled=True)
            if before!=after:raise ValueError('Metadata enrollment unexpectedly changed original wallet')
            _close_handles(runtime,feed)
            if _installed(plan)!=files:raise ValueError('Source changed during metadata preparation')
            completed={'version':VERSION,'intent_hash':fingerprint(intent),'permit':verified,
                'source_id':actual_source_id,'authority_manifest_hash':authority.manifest_hash,
                'wallet_before':before,'wallet_after':after,'handles_closed':True,
                'metadata_only':True,'fit_reservations':0,'native_completed':False,'formal_acceptance':False}
            _write_pair(roots,'completed.json',completed)
            return permit
        except BaseException as error:
            _close_handles(runtime,feed,error)
            failure={'version':VERSION,'intent_hash':fingerprint(intent),'status':'PREPARATION_FAILED_NO_RETRY',
                'reason':type(error).__name__+':'+str(error)[:1000],'retain_N':True,'possible_error':1,'power':0}
            for root in roots:
                try:
                    if root.is_dir()and not(root/'failure.json').exists():_write(root/'failure.json',failure)
                except BaseException as secondary:error.add_note('Preparation failure marker unavailable: '+repr(secondary))
            raise
