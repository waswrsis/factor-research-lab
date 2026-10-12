"""New prospective three-job warmboot; no forecasts before original endpoints.

Only an existing registered simulation schedule can supply the opportunities.
No original budget, clock, fit kernel, retry rule or support is enlarged.
"""
from pathlib import Path
import json
VERSION='nominal-original-opportunities-warmboot-v1'
ELAPSED_VERSION='nominal-original-opportunities-elapsed-warmboot-v2'
RELATIVE='factorlab/discovery/nominal_algorithm_warmboot.py'

def _installed():
    from importlib import import_module
    from factorlab.protocol import project_path
    from .monthly_work_authority import source_files
    from .evidence import file_hash
    files=source_files()
    for name in ('nominal_algorithm_warmboot','nominal_algorithm_policy',
      'nominal_algorithm_producer','nominal_short_fit_policy','known_linear_forward',
      'pool_forward','typed_policy_confirmation','v7_service','v7_forward'):
        module=import_module('factorlab.discovery.'+name);rel='factorlab/discovery/'+name+'.py'
        if Path(module.__file__).resolve()!=project_path(rel).resolve() or files.get(rel)!=file_hash(module.__file__):
            raise ValueError('Warmboot requires exact canonical source dependencies: '+name)
    return files

def _contract(policy):
    from .nominal_algorithm_policy import _utc,_month
    from datetime import timedelta
    endpoints=policy.identity['endpoints'];by={e['slot']:e for e in endpoints}
    lib,algo=by['library_skill'],by['algorithm']
    start=_utc(algo['start']);cutoff=_month(algo['start'],-1)
    elapsed=policy.identity.get('fit_schedule') is not None
    if lib['start']!=algo['start'] or lib['candidate']!=algo['reference'] or lib['candidate_stream']!=algo['reference_stream']:
        raise ValueError('Original shared K0 endpoint changed')
    har=lib['reference'];k0=algo['reference']
    if (har.get('kind')!='v7_known_linear' or har.get('recipe',{}).get('name')!='HAR_cal'
        or k0.get('pool',{}).get('members')!=[]
        or algo['candidate']['initial_pipeline']!=k0 or algo['candidate_stream']!='A0_policy_trial'):
        raise ValueError('Exact frozen initial HAR, empty-pool K0 and policy K0 required')
    streams={lib['reference_stream']:har,lib['candidate_stream']:k0,algo['candidate_stream']:k0}
    if len(streams)!=3:raise ValueError('Three distinct original jobs/admissions required')
    due=tuple(_utc(s) for s in policy.design.identity['full_schedule']['due_at'])
    opportunities=tuple(d.isoformat() for d in due if (elapsed or cutoff<=d) and d<start)
    # Retain original ordinary A0 planning/priority as well: three new protected
    # jobs need at least three distinct original fit slots, not one multi-fit step.
    if len(opportunities)<3 or due[0]>cutoff+timedelta(seconds=policy.identity['delay_seconds']):
        raise ValueError('Original registered schedule has no complete preendpoint month/opportunities')
    value={'version':VERSION,'policy_digest':policy.digest,'cutoff':cutoff.isoformat(),
      'endpoint_start':start.isoformat(),'cold_start':_month(cutoff.isoformat(),-3).isoformat(),
      'streams':streams,'opportunities':list(opportunities),'source_files':_installed(),
      'additional_real_jobs':3,'old_monthly_real_jobs':18,'same_original_max_one_fit_per_step':True,
      'initial_recipe_only':True,'claim_forecasts_before_start':False,'refund':False,
      'retry_failed_work':False,'wall_month':'actual_original_UTC','warmboot_complete':False,
      'performance_verified':False,'formal_acceptance':False}
    if elapsed:
        value.update(version=ELAPSED_VERSION,cutoff=None,cold_start=None,
          additional_real_jobs='three_per_original_preendpoint_15day_cutoff',
          fit_schedule=policy.identity['fit_schedule'],
          cold_window='continuous_90_days_at_original_data_ready_anchored_cutoff')
    return value

def _preendpoint_cutoffs(lifecycle,value):
    if value['version']==VERSION:return (value['cutoff'],)
    if value['version']!=ELAPSED_VERSION or lifecycle.plan.training_window_days!=90:
        raise ValueError('Warmboot changes its registered original elapsed cadence')
    from .rolling_cadence import RollingCadence
    from .v7_lifecycle import stamp
    anchor=lifecycle._cadence_anchor()
    if anchor>=stamp(value['endpoint_start']):return ()
    return tuple(t.isoformat() for t in RollingCadence().fit_cutoffs(
        anchor,anchor,value['endpoint_start'],include_current_cycle=False))

def _allowed_cutoff(lifecycle,value,cutoff):
    from .v7_lifecycle import stamp
    if value['version']==VERSION:return stamp(cutoff)==stamp(value['cutoff'])
    if value['version']!=ELAPSED_VERSION or lifecycle.plan.training_window_days!=90:
        raise ValueError('Unknown original warmboot cadence')
    if stamp(cutoff)>=stamp(value['endpoint_start']):return False
    lifecycle.fit_cutoff(cutoff)
    return True

class NominalAlgorithmWarmbootPolicy:
    @classmethod
    def freeze(cls,policy):
        from .nominal_algorithm_policy import NominalAlgorithmPolicy
        from .nominal_algorithm_total_risk import _fresh_parent
        from .nominal_cold_conformance import _write
        from filelock import FileLock
        if type(policy) is not NominalAlgorithmPolicy:raise TypeError('Exact newly registered policy required')
        policy=NominalAlgorithmPolicy.inspect(policy.design);_fresh_parent(policy.design.parent)
        value=_contract(policy);paths=_paths(policy)
        with FileLock(str(policy.design.directory/'.warmboot-creation.lock'),timeout=0):
            _fresh_parent(policy.design.parent)
            if paths[0].parent.exists() or any(p.exists() for p in paths):
                raise ValueError('Prior/partial warmboot cannot be replaced')
            paths[0].parent.mkdir()
            for path in paths:_write(path,value)
        return cls.inspect(policy)
    @classmethod
    def inspect(cls,policy):
        from .nominal_algorithm_policy import NominalAlgorithmPolicy
        from .nominal_cold_conformance import _read
        from factorlab.protocol import fingerprint
        if type(policy) is not NominalAlgorithmPolicy:raise TypeError('Exact registered policy required')
        policy=NominalAlgorithmPolicy.inspect(policy.design);value=_contract(policy)
        if any(_read(p)!=value for p in _paths(policy)):raise ValueError('Warmboot source/recipe/schedule changed')
        obj=cls.__new__(cls);obj.policy=policy;obj.identity=value;obj.digest=fingerprint(value);return obj
    def binding(self):
        return {'version':self.identity['version'],'digest':self.digest,'directory':str(_paths(self.policy)[0].parent)}

def _paths(policy):
    return (policy.design.directory/'warmboot/identity.json',policy.design.parent.directory/'nominal_algorithm_warmboot.json')

def policy_binding(policy):
    paths=_paths(policy)
    if not paths[0].parent.exists() and not any(p.exists() for p in paths):return None
    return NominalAlgorithmWarmbootPolicy.inspect(policy).binding()

def _bound(lifecycle):
    from .nominal_algorithm_policy import _bound as original_bound
    found=original_bound(lifecycle)
    if found is None:return None
    policy,permit=found
    if policy_binding(policy) is None:return None
    return NominalAlgorithmWarmbootPolicy.inspect(policy),permit

def plan_original_jobs(service,now):
    found=_bound(service.lifecycle)
    if found is None:return
    from .v7_service import ForwardService
    if type(service) is not ForwardService or service.lifecycle is not service.runtime.lifecycle:
        raise TypeError('Exact original service/runtime/lifecycle required')
    obj,permit=found;v=obj.identity
    from .v7_lifecycle import stamp
    if stamp(now)>=stamp(v['endpoint_start']):return
    if v['version']==VERSION and stamp(now)<stamp(v['cutoff']):return
    from .nominal_algorithm_producer import _active
    from .nominal_algorithm_policy import _live_prefix
    producer=_active(service.runtime)
    if producer is None or producer.identity['warmboot_policy']!=obj.binding():
        raise ValueError('Warmboot planning requires its original private paid producer')
    prefix=_live_prefix(service.lifecycle)
    if prefix['logical_at'] not in v['opportunities'] or stamp(now)!=stamp(prefix['logical_at']):
        raise ValueError('No original prospective warmboot opportunity')
    producer.check()
    if v['version']==ELAPSED_VERSION:
        from .v7_feed import SnapshotUnavailable
        try:service.runtime.refresh_training_view()
        except SnapshotUnavailable:return
        cutoff=service.lifecycle.due_fit_cutoff().isoformat()
        if not _allowed_cutoff(service.lifecycle,v,cutoff):return
    else:cutoff=v['cutoff']
    for stream,pipeline in v['streams'].items():
        # Existing original plan_fit records a separate job and original prefix;
        # original _fit_pending alone dispatches at most one fit per step.
        service.lifecycle.plan_fit(cutoff,stream,pipeline)
    producer.check()

def job_context(lifecycle,job,pipeline):
    found=_bound(lifecycle)
    if found is None:return None
    obj,permit=found;v=obj.identity
    row=lifecycle.db.execute('SELECT cutoff,stream,pipeline FROM jobs WHERE id=?',(job,)).fetchone()
    if row is None:return None
    if row[1] not in v['streams']:return None
    if not _allowed_cutoff(lifecycle,v,row[0]):return None
    if pipeline!=v['streams'][row[1]] or json.loads(row[2])!=pipeline:
        raise ValueError('Warmboot job recipe differs from original frozen initial learner')
    from .quarantined_observation import job_binding
    prefix=job_binding(lifecycle,job)
    if prefix is None or prefix['logical_at'] not in v['opportunities']:
        raise ValueError('Warmboot planning lacks its original preendpoint receipt prefix')
    if v['version']==ELAPSED_VERSION and row[0]!=lifecycle.fit_cycle_start(prefix['logical_at']).isoformat():
        raise ValueError('Warmboot job changed its original observation fit cycle')
    return {'role':'new_nominal_initial_warmboot','warmboot_digest':obj.digest,
      'protocol_policy':v['policy_digest'],'trial':permit.trial,'initial_stream':row[1]}

def initial_pipeline(lifecycle,cutoff,k0):
    found=_bound(lifecycle)
    if found is None:return None
    obj,_=found;v=obj.identity
    from .v7_lifecycle import stamp
    if not _allowed_cutoff(lifecycle,v,cutoff):return None
    if k0!=v['streams']['A0_policy_trial']:raise ValueError('Initial policy K0 changed')
    return k0

def typed_context(lifecycle,job,pipeline):
    value=job_context(lifecycle,job,pipeline)
    if value is None:return None
    # Warmboot initial policy is exact empty-pool K0, never neural. This check
    # prevents the neural typed-policy issuer from borrowing preendpoint scope.
    raise ValueError('Initial warmboot policy is original empty-pool K0, not a neural policy trial')

def complete_jobs(lifecycle):
    found=_bound(lifecycle)
    if found is None:return []
    obj,_=found;v=obj.identity
    from .quarantined_observation import publication_visible,read_publication
    from .v7_lifecycle import stamp
    jobs=[]
    cutoffs=_preendpoint_cutoffs(lifecycle,v)
    if not cutoffs:raise ValueError('Warmboot lacks an original data-ready preendpoint cycle')
    for cutoff,stream,pipeline in ((c,s,p) for c in cutoffs for s,p in v['streams'].items()):
        rows=lifecycle.db.execute('SELECT id,status,failure,model,pipeline FROM jobs WHERE cutoff=? AND stream=?',(cutoff,stream)).fetchall()
        if len(rows)!=1:raise ValueError('Missing original warmboot job')
        job,status,failure,model_id,raw=rows[0]
        if status not in ('verified','activated') or failure is not None or not model_id or json.loads(raw)!=pipeline:
            raise ValueError('Warmboot missing/failed/unaffordable; cannot substitute a model')
        job_context(lifecycle,job,pipeline)
        model,identity=lifecycle.registry.load(model_id,expected_pipeline=pipeline)
        if identity.get('job')!=job or 'admission' not in identity:
            raise ValueError('Warmboot lacks separate original paid model admission')
        publication=read_publication(lifecycle,'model',job)
        if (publication is None or stamp(publication['observation_prefix']['logical_at'])>=stamp(v['endpoint_start'])
            or publication_visible(lifecycle,'model',job,v['endpoint_start']) is not True):
            raise ValueError('Warmboot was not genuinely published before first original endpoint')
        jobs.append(job)
    if len(set(jobs))!=3*len(cutoffs):raise ValueError('Warmboot admissions cannot be shared')
    return jobs

def require_fit_opportunity(runtime,job):
    found=_bound(runtime.lifecycle)
    if found is None:return
    obj,_=found;v=obj.identity
    row=runtime.lifecycle.db.execute('SELECT cutoff,stream,pipeline FROM jobs WHERE id=?',(job,)).fetchone()
    if row is None or row[1] not in v['streams']:return
    if not _allowed_cutoff(runtime.lifecycle,v,row[0]):return
    from .nominal_algorithm_producer import _active
    from .nominal_algorithm_policy import _live_prefix
    producer=_active(runtime)
    if producer is None or producer.identity['warmboot_policy']!=obj.binding():
        raise ValueError('Warmboot issuance lacks its private original producer')
    prefix=_live_prefix(runtime.lifecycle)
    if prefix['logical_at'] not in v['opportunities']:
        raise ValueError('Late warmboot cannot manufacture a preendpoint admission')
    job_context(runtime.lifecycle,job,json.loads(row[2]));producer.check()

