"""New nominal library and three fixed-factor census on original replica wallets.

No library risk reuse, new wallet, provider substitution, or joint-record alias.
The selected recipe is derived from the original actual validation ledger. The
post-selection nominal census freezes before its oracle raw/model exposure; dates in
its synthetic C calendar are not past actual receipt/publication assertions.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json
import math
import pandas as pd
from filelock import FileLock
from factorlab.protocol import project_path,fingerprint
from .evidence import file_hash
from .monthly_work_authority import MonthlyWorkAuthority,source_files,DEFAULT_LIMITS
from .rolling import RollingSpec,build_rolling_data
from .known_linear_reference import FrozenLinearRecipe,KnownLinearPredictor,fit_known_linear
from .joint_pool import FrozenPool,JointPredictor,fit_pool
from .finite_market_oracle import FiniteGenerator,_finite_loss
from .nominal_cold_conformance import _plain,_write,_read,numerical_model,selected_cutoff,_score
from .v7_lifecycle import stamp,boundary

VERSION='nominal-selected-factor-cold-census-v2'
ESTIMAND='nominal_full_cold_rolling_total_loss_v2'
CHOICE_SHA256='035fb78387f06970c4e4249c3728c2cf865a6857ebfce5fe7cf77be3a0d2b1f0'
RELATIVE='factorlab/discovery/nominal_factor_census.py'
SLOTS=('library_skill','factor1','factor2','factor3')
ROLES=('reference','candidate')
BOUND='nominal_selected_factor_protocol_bound'


def _rolling(value):
    value=dict(value);value['event_bins']=tuple(value['event_bins'])
    return RollingSpec(**value)


def _loss_module():
    from . import nominal_total_loss as loss
    files=_installed();relative='factorlab/discovery/nominal_total_loss.py'
    if (Path(loss.__file__).resolve()!=project_path(relative).resolve()
            or files.get(relative)!=file_hash(loss.__file__)
            or loss.VERSION!='nominal-total-loss-v2' or loss.ESTIMAND!=ESTIMAND
            or loss.USER_CHOICE_SHA256!=CHOICE_SHA256):
        raise ValueError('Original new nominal loss source/user choice required')
    return loss


def _number(value):
    value=float(value)
    return value if math.isfinite(value) else None


def _risk_fraction(value):
    from fractions import Fraction
    if (type(value) is not list or len(value)!=2
            or any(type(v) is not int for v in value) or value[1]<=0):
        raise ValueError('Original exact path-risk rational required')
    return Fraction(*value)


def _atom_evidence(value):
    """Hash original paid public evidence; tuple-index streams are redundant.

    Streams index the same predictions already retained in artifacts and pairs.
    Hashing their tuple keys with JSON would reject every authentic atom.
    """
    return fingerprint({k:value[k] for k in ('protocol','atom','artifacts','pairs',
        'exact_path_risks','path_risks','statistic','numerical_event','formal_acceptance')})


def _rows(data,model,protocol,plan):
    """Actual completed model inference, preserving every scheduled endpoint.

    Invalid model mask or nonfinite output becomes the authorized numeric B.
    Exceptions, failed fitting, missing disk/OS evidence do not become completion.
    """
    loss=_loss_module();p=protocol.identity;schedule=p['schedule']
    predictions,valid=model.predict_history(data.frame)
    if len(predictions)!=len(data.frame.metadata) or len(valid)!=len(predictions):
        raise ValueError('Original kernel inference shape changed')
    rows=[]
    for i,meta in data.frame.metadata.iterrows():
        origin=stamp(meta.available_at)
        if not stamp(schedule['start'])<=origin<stamp(schedule['end']):continue
        if selected_cutoff(schedule,plan['role'],origin)!=plan['cutoff']:continue
        pred=_number(predictions[i]) if valid[i] else None
        label=_number(data.target[i]) if data.observed[i] else None
        rows.append({'origin':origin.isoformat(),'hour':origin.floor('h').isoformat(),
            'symbol':meta.symbol,'feature_open_time':stamp(meta.feature_open_time).isoformat(),
            'window':stamp(meta.forecast_window_start).isoformat(),'cutoff':plan['cutoff'],
            'train_start':plan['train_start'],'prediction':pred,'label':label,
            'maturity':stamp(data.maturity[i]).isoformat(),'loss':loss.nominal_total_loss(label,pred)})
    expected=[(t.isoformat(),s) for t in pd.date_range(schedule['start'],schedule['end'],freq='h',inclusive='left')
        +pd.Timedelta(seconds=schedule['delay']) for s in p['law']['symbols']
        if selected_cutoff(schedule,plan['role'],t)==plan['cutoff']]
    actual={(r['origin'],r['symbol']):r for r in rows}
    expected_set=set(expected)
    if len(actual)!=len(rows) or any(k not in expected_set for k in actual):
        raise ValueError('Duplicate or foreign nominal algorithm output')
    # Support comes from the frozen calendar/law, never successful frame rows.
    # This branch follows completed original build_rolling_data AND inference;
    # absent input artifacts and interrupted computations never enter it.
    for origin,symbol in expected:
        if (origin,symbol) in actual:continue
        t=stamp(origin);window=t.ceil('h')
        actual[(origin,symbol)]={'origin':origin,'hour':t.floor('h').isoformat(),'symbol':symbol,
            'feature_open_time':(t.floor('h')-pd.Timedelta(hours=1)).isoformat(),
            'window':window.isoformat(),'cutoff':plan['cutoff'],'train_start':plan['train_start'],
            'prediction':None,'label':None,'maturity':(window+pd.Timedelta(hours=12)).isoformat(),
            'loss':loss.nominal_total_loss(None,None)}
    return [actual[k] for k in sorted(expected)]


def _installed():
    files=source_files()
    if (Path(__file__).resolve()!=project_path(RELATIVE).resolve()
            or files.get(RELATIVE)!=file_hash(__file__)):
        raise ValueError('Canonical reviewed new source required; staged producer refused')
    return files


def _schedule(spec,*,rolling_spec=None,cadence_anchor=None):
    """Original C endpoints, including pre-C warmboot, not six-month relabeling."""
    start,end=boundary(spec.start),boundary(spec.end)
    if not start<end or end>start+pd.DateOffset(months=6):
        raise ValueError('Original bounded whole-month factor C window required')
    elapsed=rolling_spec is not None and rolling_spec.window_days==90
    if elapsed:
        from .rolling_cadence import RollingCadence
        if cadence_anchor is None:raise ValueError('Original selected-trial cadence anchor required')
        cutoffs=RollingCadence().fit_cutoffs(cadence_anchor,start-pd.Timedelta(days=15),end)
        if not cutoffs:raise ValueError('Original nominal endpoint has no data-ready cold support')
        train_start=rolling_spec.window(cutoffs[0])[0]
    else:
        if cadence_anchor is not None:raise ValueError('Calendar nominal scope cannot borrow an elapsed anchor')
        cutoffs=[start-pd.DateOffset(months=1),*pd.date_range(start,end,freq='MS',inclusive='left')]
        train_start=cutoffs[0]-pd.DateOffset(months=3)
    value={'start':start.isoformat(),'end':end.isoformat(),'delay':5,
        'cutoffs':[c.isoformat() for c in cutoffs],
        'raw_start':(train_start-pd.Timedelta(hours=1234)).isoformat(),
        'raw_end':(end+pd.Timedelta(hours=36)).isoformat(),
        'hours':int((end-start)/pd.Timedelta(hours=1)),
        'ready':{role:{c.isoformat():(c+pd.Timedelta(hours=i+1,seconds=5)).isoformat() for c in cutoffs} for i,role in enumerate(ROLES)},
        'rule':'tick_before_fit_reference_then_candidate_previous_month_until_fixed_ready'}
    if elapsed:
        value['rule']='tick_before_fit_reference_then_candidate_original_15day_cycle_until_fixed_ready'
        value['fit_schedule']={'cadence_hash':RollingCadence().digest,
            'anchor':stamp(cadence_anchor).isoformat(),'training_days':90,'refit_days':15,
            'origin':'verified_original_selected_trial_lifecycle_anchor'}
    return value


def _recipe(pipeline,spec):
    """Derive executable original recipe and prepaid bound, never a caller lane."""
    from .v7_forward import ForwardRuntime
    from .pool_forward import PoolForwardRecipe
    stripped={k:v for k,v in pipeline.items() if k!='complexity'}
    runtime=ForwardRuntime.__new__(ForwardRuntime);runtime.spec=spec
    if stripped.get('kind')=='v7_known_linear':
        recipe=FrozenLinearRecipe.from_dict(stripped['recipe']);limit=1
    elif 'pool' in stripped:
        recipe=FrozenPool.from_dict(stripped['pool']);limit=PoolForwardRecipe(recipe,spec).solve_limit
    else:raise ValueError('Actual selected recipe lacks original known/pool nominal kernel')
    if _plain(runtime.recipe_for(recipe))!=stripped:
        raise ValueError('Original full selected pipeline/rolling identity differs')
    return recipe,limit


def _counts_for(recipes,spec):
    limit=sum(_recipe(recipes[role],spec)[1] for role in ROLES)
    return {k:limit if k=='solve' else 0 for k in DEFAULT_LIMITS}


def _fit_recipe(data,cutoff,pipeline,budget):
    recipe,_=_recipe(pipeline,data.spec)
    return (fit_known_linear(data,cutoff,recipe,budget=budget) if type(recipe) is FrozenLinearRecipe
            else fit_pool(data,cutoff,recipe,budget=budget))


def _model_class(pipeline):
    return KnownLinearPredictor if pipeline.get('kind')=='v7_known_linear' else JointPredictor


def _original(registry,cell,index):
    from .full_calibration_registry import FullTrialRegistry,QuarantinedTrialPermit
    from .calibration_oracle_evidence import _frozen_estimands
    from .full_mechanism_contract import verify_mechanism_contract
    if type(registry) is not FullTrialRegistry:raise TypeError('Exact original full registry required')
    registry.verify();files=_installed()
    if registry.identity['source_files']!=files:raise ValueError('Original whole source differs')
    permit=QuarantinedTrialPermit(registry,registry.identity_for(cell,index));permit.verify()
    frozen,specs=_frozen_estimands(permit)
    generator=verify_mechanism_contract(registry.plan.mechanism_contract,registry.plan.acceptance)[permit.trial['scenario_digest']]
    generator.verify_registry(registry)
    if registry.plan.oracle_paths!=128 or generator.law.support_size!=128 or generator.law.gap_probability!=0:
        raise ValueError('Original complete128 fixed market support required')
    return permit,frozen,{s.slot:s for s in specs if s.slot in SLOTS},generator


def _census_value(registry,cell,index):
    permit,frozen,specs,generator=_original(registry,cell,index)
    if set(specs)!=set(SLOTS):raise ValueError('All four original fixed slots required')
    spec=_rolling(registry.plan.runtime_scope['rolling'])
    if spec!=RollingSpec(availability_delay_seconds=5,window_days=spec.window_days):
        raise ValueError('Original default volatility12h cold recipe required')
    cadence_anchor=None
    if spec.window_days is not None:
        from .calibration_oracle_evidence import _lifecycle
        with _lifecycle(permit.paths['runtime']) as lifecycle:
            permit.verify(lifecycle);lifecycle.verify()
            if lifecycle.plan.training_window_days!=90 or lifecycle.plan.refit_interval_days!=15:
                raise ValueError('Selected original trial changes the nominal census cadence')
            cadence_anchor=lifecycle._cadence_anchor()
    values={}
    for slot,selected in specs.items():
        schedule=_schedule(selected,rolling_spec=spec,cadence_anchor=cadence_anchor)
        recipes={'reference':selected.reference,'candidate':selected.candidate}
        if slot=='library_skill':
            from .v7_forward import ForwardRuntime
            runtime=ForwardRuntime.__new__(ForwardRuntime);runtime.spec=spec
            expected={'reference':_plain(runtime.recipe_for(FrozenLinearRecipe('HAR_cal',spec))),
                      'candidate':_plain(runtime.recipe_for(FrozenPool(role='reference',version='K0_v1')))}
            if (recipes!=expected or stamp(schedule['end'])!=stamp(schedule['start'])+pd.DateOffset(months=6)):
                raise ValueError('Original full six-month HAR versus K0 library endpoint required')
        counts=None if selected.candidate is None else _counts_for(recipes,spec)
        if (stamp(generator.law.start)>stamp(schedule['raw_start'])
                or stamp(generator.law.start)+pd.Timedelta(hours=generator.law.hours)<stamp(schedule['raw_end'])):
            raise ValueError('Original generator lacks complete C/warmboot/three-month/1234 raw support')
        values[slot]={'estimand':selected.to_dict(),'estimand_hash':selected.digest,'schedule':schedule,
            'recipes':recipes,'counts_per_pair':counts,'state':'not_selected' if counts is None else 'registered_selected_recipe'}
    directory=registry.directory/'nominal_factor_censuses'/permit.trial['population_trial_id']
    _loss_module()
    return {'version':VERSION,'estimand':ESTIMAND,'claim_kind':'four_original_fixed_slots',
        'user_choice_sha256':CHOICE_SHA256,'loss_version':'nominal-total-loss-v2',
        'directory':str(directory),'parent_registry':str(registry.directory),'parent_digest':registry.digest,
        'validation':permit.trial,'frozen_selected_estimands_hash':fingerprint(frozen),'slots':values,
        'source_files':_installed(),'generator_directory':str(generator.directory),'generator_digest':generator.digest,
        'law':generator.law.to_dict(),'spec':_plain(asdict(spec)),'atoms':list(range(128)),
        'replica_budget':'original_prelisted_full_registry_fixed_estimand_replica_only',
        'scope_extension':'new_nominal_protocol_before_any_replica_execution_not_joint_alias',
        'truth_method':'exact_complete_same_finite_population_census_not_iid_confirmation',
        'physical_clock':'original_UTC_wallet_no_reset_or_transfer','seconds_per_pair':300,
        'raw_availability':'synthetic_nominal_close_plus5_not_historical_receipt',
        'prospective_rule':'selected_specs_frozen_before_oracle_raw_or_fit_exposure_not_before_synthetic_C_calendar',
        'formal_acceptance':False,'pending':['actual_fixed_slot_conformance','whole_algorithm_nominal_producer','metered_truth_readback','typed_nominal_fit_failure']}


class NominalFactorCensus:
    @classmethod
    def freeze(cls,registry,cell,index):
        value=_census_value(registry,cell,index);directory=project_path(value['directory'])
        mirror=registry.directory/'nominal_factor_bindings'/(value['validation']['population_trial_id']+'.json')
        with FileLock(str(directory)+'.registration.lock',timeout=0):
            if directory.exists() or mirror.exists():raise ValueError('Nominal factor census already attempted; no replacement')
            if (project_path(value['generator_directory'])/'nominal_census.json').exists():
                raise ValueError('Old partial-domain library census cannot become a new total-loss run')
            # Registration owns all prelisted factor coordinates, not aliases.
            for slot in SLOTS:
                for atom in range(128):
                    paths=registry.paths(cell,index,oracle_index=atom,estimand_slot=slot)
                    trial=registry.identity_for(cell,index,oracle_index=atom,estimand_slot=slot)
                    draw=project_path(value['generator_directory'])/'draws'/(trial['trial_id']+'.json')
                    if draw.exists() or any((paths['root']/name).exists() for name in ('generator_draw.json','fixed_oracle','controller','generator_publications','nominal_factor_protocol')):
                        raise ValueError('Independent replica already exposed or executed under another mode')
                    if (paths['authority']/'manifest.json').is_file():
                        from .neural_forward_ticket import open_existing_authority
                        if open_existing_authority(paths['authority']).verify()['reservations']:
                            raise ValueError('Nominal census cannot borrow charged replica work')
            directory.mkdir(parents=True)
            value={**value,'frozen_at':pd.Timestamp.now(tz='UTC').isoformat()}
            _write(directory/'identity.json',value);_write(mirror,value)
        return cls(directory)
    def __init__(self,directory):
        from .full_calibration_registry import FullTrialRegistry
        self.directory=project_path(directory);self.identity=_read(self.directory/'identity.json')
        self.registry=FullTrialRegistry.inspect(self.identity['parent_registry'])
        self.digest=fingerprint(self.identity);self.verify()
    def verify(self):
        p=self.identity;v=p['validation'];expected=_census_value(self.registry,v['cell'],v['index'])
        mirror=self.registry.directory/'nominal_factor_bindings'/(v['population_trial_id']+'.json')
        if (p!=_read(self.directory/'identity.json') or p!=_read(mirror)
                or (project_path(p['generator_directory'])/'nominal_census.json').exists()
                or p!={**expected,'frozen_at':p['frozen_at']} or fingerprint(p)!=self.digest):
            raise ValueError('New nominal selected census/parent/source/selection changed')
        return p
    def bind_replica(self,permit,runtime):
        return NominalFactorProtocol.freeze(self,permit,runtime)
    def assemble(self,slot,atom):
        from .full_calibration_registry import QuarantinedTrialPermit
        from .neural_forward_ticket import open_existing_authority
        p=self.verify();v=p['validation']
        trial=self.registry.identity_for(v['cell'],v['index'],oracle_index=atom,estimand_slot=slot)
        permit=QuarantinedTrialPermit(self.registry,trial)
        protocol=NominalFactorProtocol(permit.paths['root']/'nominal_factor_protocol')
        return assemble_atom(protocol,authority=open_existing_authority(permit.paths['authority']),atom=atom)
    def read_census_risk(self,slot):
        """Complete source-bound finite census; no caller risks and no IID credit.

        This reader still needs a future registered proof-task/native wrapper;
        unmetered use cannot mint formal acceptance or actual conformance.
        """
        from fractions import Fraction
        p=self.verify()
        if slot not in SLOTS:raise ValueError('Original factor slot required')
        rows=[];points=[]
        for atom in range(128):
            try:
                if p['slots'][slot]['state']=='not_selected':raise ValueError('Original factor not selected')
                value=self.assemble(slot,atom)
                point=tuple(_risk_fraction(value['exact_path_risks'][r]) for r in ROLES)
                from .finite_market_oracle import LOSS_BOUND
                if any(not 0<=v<=LOSS_BOUND for v in point):
                    raise ValueError('Incomplete finite accepted-loss domain')
                evidence_digest=_atom_evidence(value)
                points.append(tuple(Fraction(v) for v in point))
                rows.append({'atom':atom,'state':'complete','evidence_digest':evidence_digest})
            except (ValueError,OSError,KeyError,TypeError,ArithmeticError) as error:
                rows.append({'atom':atom,'state':'failed_or_unknown_retained','reason':type(error).__name__+':'+str(error)[:400]})
        risk=None;truth='unknown'
        if len(points)==128:
            risks=[sum((v[j] for v in points),Fraction())/128 for j in range(2)]
            risk={r:[v.numerator,v.denominator] for r,v in zip(ROLES,risks)}
            if risks[0]>0:
                gain=1-risks[1]/risks[0];risk['gain']=[gain.numerator,gain.denominator]
                truth='null' if gain<=Fraction(1,200) else 'alternative'
        if self.verify()!=p:raise ValueError('Nominal factor source/selection changed during census')
        return {'version':VERSION,'census_digest':self.digest,'slot':slot,'estimand_hash':p['slots'][slot]['estimand_hash'],
            'registered_atoms':128,'complete_atoms':len(points),'rows':rows,'exact_finite_risk':risk,'truth':truth,
            'method':p['truth_method'],'iid_oracle_credit':False,'actual_conformance':False,
            'verification_cost_registered':False,'native_completion_verified':False,'formal_acceptance':False}


class NominalFactorProtocol:
    @classmethod
    def freeze(cls,census,permit,runtime):
        from .full_calibration_registry import QuarantinedTrialPermit
        from .v7_forward import ForwardRuntime
        from .neural_forward_ticket import open_existing_authority
        from .calibration_runtime_contract import _original_operations
        from .monthly_partition_policy import _runtime
        if type(census) is not NominalFactorCensus or type(permit) is not QuarantinedTrialPermit or type(runtime) is not ForwardRuntime:
            raise TypeError('Exact original census/fixed permit/runtime required')
        p=census.verify();v=p['validation'];slot=permit.trial.get('estimand_slot');atom=permit.trial['oracle_index']
        if slot not in SLOTS or type(atom) is not int or not 0<=atom<128:raise ValueError('Original factor replica coordinate required')
        expected=census.registry.identity_for(v['cell'],v['index'],oracle_index=atom,estimand_slot=slot)
        if permit.registry.digest!=census.registry.digest or permit.trial!=expected or runtime.directory!=permit.paths['runtime']:
            raise ValueError('Replica cannot borrow another parent/coordinate/runtime')
        selected=p['slots'][slot]
        if selected['state']!='registered_selected_recipe':raise ValueError('Original nonselection has no substitute factor')
        directory=permit.paths['root']/'nominal_factor_protocol'
        # Same lock as original joint fixed controller: the two modes cannot win
        # concurrently. The new LC event makes the old mode's freeze refuse.
        with FileLock(str(permit.paths['root']/'fixed_oracle')+'.lock',timeout=0):
            permit.verify(runtime.lifecycle)
            authority=open_existing_authority(permit.paths['authority'])
            if (directory.exists() or (permit.paths['root']/'fixed_oracle').exists()
                    or (permit.paths['root']/'controller').exists()
                    or (permit.paths['runtime']/'quarantined_observation').exists()
                    or (permit.paths['root']/'generator_publications').exists()
                    or (permit.paths['root']/'generator_draw.json').exists()
                    or (project_path(p['generator_directory'])/'draws'/(permit.trial['trial_id']+'.json')).exists()
                    or authority.verify()['reservations']):
                raise ValueError('Original replica already attempted; no joint migration or wallet reset')
            operations=_original_operations(runtime.lifecycle)
            if any(o['kind'] not in {'forward_runtime_bound','monthly_work_authority_bound','full_calibration_quarantine_enrolled'} for o in operations):
                raise ValueError('Freeze new nominal protocol before independent source/work exposure')
            if runtime.spec!=_rolling(p['spec']):raise ValueError('Original rolling scope differs')
            owner=_runtime(authority,runtime.directory)
            mirror=authority.directory/'nominal_factor_protocol'/'identity.json'
            if mirror.exists():raise ValueError('Original wallet already bound another nominal protocol')
            value={'version':VERSION+'-replica','directory':str(directory),'mirror':str(mirror),
                'census_directory':str(census.directory),'census_digest':census.digest,'trial':permit.trial,
                'estimand_version':ESTIMAND,'user_choice_sha256':CHOICE_SHA256,'loss_version':'nominal-total-loss-v2',
                'slot':slot,'atom':atom,'estimand':selected['estimand'],'estimand_hash':selected['estimand_hash'],
                'recipes':selected['recipes'],'counts_per_pair':selected['counts_per_pair'],
                'schedule':selected['schedule'],'spec':p['spec'],'law':p['law'],'population':'evaluation',
                'generator_directory':p['generator_directory'],'generator_digest':p['generator_digest'],
                'runtime_directory':str(runtime.directory),'authority_directory':str(authority.directory),
                'authority_manifest_hash':authority.manifest_hash,'owner':owner,'source_files':p['source_files'],
                'formal_acceptance':False,'lease_seconds_per_pair':300,'retry':False,
                'numeric_native':__import__('factorlab.discovery.nominal_numeric_history',fromlist=['descriptor']).descriptor()}
            directory.mkdir(parents=True);_write(directory/'identity.json',value);_write(mirror,value)
            with runtime.lifecycle.db:runtime.lifecycle._event(BOUND,{'identity_hash':fingerprint(value)})
        return cls(directory)
    def __init__(self,directory):
        self.directory=project_path(directory);self.identity=_read(self.directory/'identity.json')
        self.digest=fingerprint(self.identity);self.verify()
    def verify(self):
        from .full_calibration_registry import QuarantinedTrialPermit
        from .calibration_oracle_evidence import _lifecycle
        from .calibration_runtime_contract import _original_operations
        p=self.identity;c=NominalFactorCensus(p['census_directory']);c.verify()
        from .nominal_numeric_history import descriptor
        if p.get('numeric_native')!=descriptor():raise ValueError('Original prospective numeric native plan required')
        s=c.identity['slots'][p['slot']];permit=QuarantinedTrialPermit(c.registry,p['trial'])
        if (any((permit.paths['root']/name).exists() for name in ('fixed_oracle','controller','generator_draw.json','generator_publications'))
                or (permit.paths['runtime']/'quarantined_observation').exists()
                or (project_path(p['generator_directory'])/'draws'/(permit.trial['trial_id']+'.json')).exists()):
            raise ValueError('Nominal fixed replica acquired incompatible joint/observation use')
        v=c.identity['validation']
        expected=c.registry.identity_for(v['cell'],v['index'],oracle_index=p['atom'],estimand_slot=p['slot'])
        with _lifecycle(permit.paths['runtime']) as lc:
            permit.verify(lc)
            anchors=[o['detail'] for o in _original_operations(lc) if o['kind']==BOUND]
        if (p!=_read(self.directory/'identity.json') or p!=_read(project_path(p['mirror']))
                or p['version']!=VERSION+'-replica' or p['source_files']!=_installed()
                or p['census_digest']!=c.digest or p['atom']!=permit.trial['oracle_index']
                or p['slot']!=permit.trial['estimand_slot'] or p['formal_acceptance'] is not False
                or p['trial']!=expected or s['state']!='registered_selected_recipe'
                or p['population']!='evaluation' or p['lease_seconds_per_pair']!=300 or p['retry'] is not False
                or p['directory']!=str(self.directory) or fingerprint(p)!=self.digest
                or p['estimand_version']!=ESTIMAND or p['user_choice_sha256']!=CHOICE_SHA256
                or p['loss_version']!='nominal-total-loss-v2'
                or str(self.directory)!=str(permit.paths['root']/'nominal_factor_protocol')
                or p['runtime_directory']!=str(permit.paths['runtime'])
                or p['authority_directory']!=str(permit.paths['authority'])
                or p['mirror']!=str(permit.paths['authority']/'nominal_factor_protocol'/'identity.json')
                or any(p[k]!=s[k] for k in ('estimand','estimand_hash','recipes','schedule','counts_per_pair'))
                or any(p[k]!=c.identity[k] for k in ('spec','law','source_files','generator_directory','generator_digest'))
                or anchors!=[{'identity_hash':self.digest}]):
            raise ValueError('Original new nominal factor protocol/source/selection/LC anchor differs')
        return p
    def plan(self,atom,cutoff,role):
        p=self.verify()
        if atom!=p['atom'] or type(atom) is not int or role not in ROLES or cutoff not in p['schedule']['cutoffs']:
            raise ValueError('Outside original selected factor nominal support')
        return {'protocol':self.digest,'atom':atom,'cutoff':cutoff,'role':role,
            'train_start':_rolling(p['spec']).window(cutoff)[0].isoformat(),
            'ready':p['schedule']['ready'][role][cutoff],'counts':p['counts_per_pair'],'lease_seconds':300}
    def pair_plan(self,atom,cutoff):
        roles=[self.plan(atom,cutoff,r) for r in ROLES]
        return {'protocol':self.digest,'atom':atom,'cutoff':cutoff,'roles':roles,
            'counts':self.identity['counts_per_pair'],'lease_seconds':300}


def validate_nominal_factor_route(authority,work_key,binding,lease,debit):
    from .monthly_partition_policy import _runtime,frozen_policy
    intent=_read(authority.directory/'nominal_factor_intents'/fingerprint(work_key)/'intent.json')
    protocol=NominalFactorProtocol(intent['protocol_directory']);p=protocol.verify()
    plan=protocol.pair_plan(p['atom'],intent['plan']['cutoff'])
    key='nominal_factor_cold:'+fingerprint(plan)
    expected={'kind':VERSION,'protocol':protocol.digest,'plan_hash':fingerprint(plan)}
    if (intent!=_read(protocol.directory/'fits'/fingerprint(plan)/'intent.json')
            or intent!={'protocol_directory':str(protocol.directory),'plan':plan,'work_key':key,'binding':expected,'lease':lease.to_dict()}
            or work_key!=key or binding!=expected or debit!=p['counts_per_pair'] or lease.lease_seconds!=300
            or str(authority.directory)!=p['authority_directory'] or authority.manifest_hash!=p['authority_manifest_hash']
            or authority.manifest['source_files']!=p['source_files'] or authority.manifest['limits']!=DEFAULT_LIMITS
            or authority.manifest['monthly_seconds']!=7200 or authority.manifest['partition_policy']!=frozen_policy('standard')):
        raise ValueError('Original factor replica prepayment/lease/source/scope differs')
    c=NominalFactorCensus(p['census_directory'])
    expected_run=fingerprint({'registry':c.registry.digest,'trial':p['trial']['trial_id']})
    owner=_runtime(authority,p['runtime_directory'])
    if authority.manifest['scope']['run_id']!=expected_run or owner!=p['owner']:
        raise ValueError('Only the already registered fixed-estimand replica wallet may pay')
    return {'lane':'forward','owner':owner}


def _verify_original_rows(protocol,folder,role,model,output):
    from .full_mechanism_contract import _generator
    c=NominalFactorCensus(protocol.identity['census_directory']);p=protocol.identity
    generator=_generator(p['generator_directory'],c.registry.plan.acceptance)
    raw=_read(folder/'raw.json')
    if raw!=generator.reconstruct_atom(p['atom'],population='evaluation'):
        raise ValueError('Nominal raw differs from all original deterministic atom rows')
    bars=pd.DataFrame(raw['rows']);bars['open_time']=pd.to_datetime(bars.open_time,utc=True)
    bars['available_at']=bars.open_time+pd.Timedelta(hours=1,seconds=5)
    bars.attrs['availability_mode']='assumed_historical_delay'
    data=build_rolling_data(bars,_rolling(p['spec']))
    if output['rows']!=_rows(data,model,protocol,output['plan']):
        raise ValueError('Original model/raw/calendar nominal predictions or losses differ')


def produce_selected_month(protocol,*,runtime,cutoff):
    """Fixed public dispatch: no caller generator/authority/recipe/role facts."""
    from .neural_forward_ticket import open_existing_authority
    from .full_mechanism_contract import _generator
    if type(protocol) is not NominalFactorProtocol:raise TypeError('Exact nominal factor protocol required')
    p=protocol.verify();c=NominalFactorCensus(p['census_directory'])
    authority=open_existing_authority(p['authority_directory'])
    generator=_generator(p['generator_directory'],c.registry.plan.acceptance)
    return _produce_month(protocol,runtime=runtime,authority=authority,generator=generator,atom=p['atom'],cutoff=cutoff)


def _output_counts(outputs):
    solves = [outputs[r]['solve_calls'] for r in ROLES]
    if any(type(v) is not int or v < 1 for v in solves):
        raise ValueError('Actual original cold solve counts required')
    return {k: sum(solves) if k == 'solve' else 0 for k in DEFAULT_LIMITS}


def _produce_month(protocol, *, runtime, authority, generator, atom, cutoff):
    """One original prepaid selected-recipe pair; exact recursive solve bound and 300-second cap. Never runs staged code, caches fits or retries.

    Original kernel membership/memory/CPU checks and the same 300-second lease
    surround raw reconstruction, feature preparation, fit, predictions and I/O.
    """
    from .v7_forward import ForwardRuntime
    from .reserved_work_budget import ReservedChannelBudget
    from .work_lease import FrozenWorkLease
    from .nonlinear_launcher import _kernel_snapshot, _validate_kernel
    from .neural_forward_launcher import _identity, _member, _observation
    if (type(protocol) is not NominalFactorProtocol or type(runtime) is not ForwardRuntime
            or type(authority) is not MonthlyWorkAuthority or type(generator) is not FiniteGenerator):
        raise TypeError('Actual original objects required, no callback/role credential')
    from .nominal_numeric_history import require_original_pair,bind_original_lease
    require_original_pair(protocol,cutoff)
    p = protocol.verify(); plan = protocol.pair_plan(atom, cutoff)
    COUNTS = p['counts_per_pair']
    if (str(runtime.directory) != p['runtime_directory'] or authority.manifest_hash != p['authority_manifest_hash']
            or generator.digest != p['generator_digest'] or str(generator.directory) != p['generator_directory']):
        raise ValueError('Actual input binding differs')
    key = 'nominal_factor_cold:'+fingerprint(plan); folder = protocol.directory/'fits'/fingerprint(plan)
    mirror = authority.directory/'nominal_factor_intents'/fingerprint(key)
    with FileLock(str(mirror)+'.lock', timeout=0):
        if folder.exists() or mirror.exists():
            raise ValueError('Attempt already allocated: failures/interruption cannot retry')
        lease = FrozenWorkLease.issue(300)
        bind_original_lease(protocol,cutoff,lease)
        intent = {'protocol_directory': str(protocol.directory), 'plan': plan, 'work_key': key,
            'binding': {'kind': VERSION, 'protocol': protocol.digest, 'plan_hash': fingerprint(plan)},
            'lease': lease.to_dict()}
        folder.mkdir(parents=True); mirror.mkdir(parents=True)
        _write(folder/'intent.json', intent); _write(mirror/'intent.json', intent)
        record = None
        try:
            record = authority._reserve_with_lease(key, counts=COUNTS, binding=intent['binding'], lease=lease)
            if record['status'] != 'reserved':
                raise ValueError('Fresh original reservation required')
            first = _kernel_snapshot(); _validate_kernel(first, first); _member(first, _identity())
            record = authority.start(record['reservation_id'], _identity())
            budget = ReservedChannelBudget.from_issuer_reservation(authority, record['reservation_id'])
            raw = generator.reconstruct_atom(atom, population=p['population'])
            if any(s['gap'] for s in raw['schedule']):
                raise ValueError('Registered complete nominal raw path unexpectedly has gaps')
            _write(folder/'raw.json', raw)
            from .nominal_fit_outcome import (prepare_nominal_fit, NominalNumericalFailure,
                                              _finish_failed_pair)
            preparation = prepare_nominal_fit(protocol, authority=authority, budget=budget,
                                              atom=atom, cutoff=cutoff)
            data = preparation.data
            outputs = {}
            for role in ROLES:
                model = preparation.fit_role(role)
                if type(model) is NominalNumericalFailure:
                    return _finish_failed_pair(preparation, model, authority, outputs, first)
                if model.pipeline_id != p['recipes'][role]['pipeline_id']:
                    raise ValueError('Original selected pipeline changed at cold fit')
                role_plan = protocol.plan(atom, cutoff, role)
                rows = _rows(data, model, protocol, role_plan)
                model.save(folder/(role+'-model.json'))
                restored = _model_class(p['recipes'][role]).load(folder/(role+'-model.json'))
                if restored.digest != model.digest:
                    raise ValueError('Original model save/reload differs')
                outputs[role] = {'version': VERSION, 'plan': role_plan, 'raw_hash': fingerprint(raw),
                    'model_digest': model.digest, 'numerical_model': numerical_model(model),
                    'training': model.receipt['training'], 'rows': rows,
                    'solve_calls': 1 if type(model) is KnownLinearPredictor else model.receipt['solve_calls'], 'formal_acceptance': False}
            output = {'plan': plan, 'roles': outputs, 'raw_hash': fingerprint(raw)}
            _write(folder/'output.json', output)
            budget.finalize()
            actual = ReservedChannelBudget.verify_counts(authority, record['reservation_id'])
            if actual['test_only'] or not actual['finalized'] or actual['actual_counts'] != _output_counts(outputs) or any(actual['actual_counts'][k] > COUNTS[k] for k in COUNTS):
                raise ValueError('Original complete actual cold solve accounting required')
            last = _kernel_snapshot(); _validate_kernel(first, last); _member(last, _identity())
            observation = _observation(lease)
            completed = authority.complete(record['reservation_id'], result={
                'nominal_output': fingerprint(output), 'raw_hash': fingerprint(raw),
                'models_hash': fingerprint({r: outputs[r]['model_digest'] for r in ROLES})}, actual_counts=actual['actual_counts'])
            _write(folder/'terminal.json', {'status': 'completed', 'reservation': completed,
                'kernel_before': first, 'kernel_after': last, 'observation': observation,
                'output_hash': fingerprint(output), 'formal_acceptance': False})
            # Missing successor is failure even if the debit completed. No readback
            # accepts terminal alone. Final I/O and lease observation are checked.
            final = _kernel_snapshot(); _validate_kernel(last, final); _member(final, _identity())
            _write(folder/'publication.json', {'terminal_hash': file_hash(folder/'terminal.json'),
                'kernel': final, 'observation': _observation(lease)})
            lease.assert_parent_alive()
            result = {r: read_month(protocol, authority=authority, atom=atom, cutoff=cutoff, role=r) for r in ROLES}
            lease.assert_parent_alive()
            return result
        except BaseException as error:
            try:
                _write(folder/'failure.json', {'reason': type(error).__name__+':'+str(error)[:2000],
                    'fit_slot_consumed': True, 'retry': False, 'formal_acceptance': False})
            finally:
                if record is not None and authority.get(record['reservation_id'])['status'] in ('reserved', 'running'):
                    authority.fail_consumed(record['reservation_id'], str(error)[:1000])
            raise

def read_month(protocol, *, authority, atom, cutoff, role):
    """Original paid record, full kernel witnesses and immutable raw artifacts."""
    from .reserved_work_budget import ReservedChannelBudget
    from .work_lease import FrozenWorkLease
    from .neural_forward_admission import _kernel, _observation, _file_deadline
    if type(protocol) is not NominalFactorProtocol or type(authority) is not MonthlyWorkAuthority:
        raise TypeError('Exact original fixed protocol and wallet required')
    p = protocol.verify(); COUNTS = p['counts_per_pair']
    role_plan = protocol.plan(atom, cutoff, role)
    plan = protocol.pair_plan(atom, cutoff)
    folder = protocol.directory/'fits'/fingerprint(plan)
    if (folder/'failure.json').exists():
        raise ValueError('Failed nominal slot remains in full denominator')
    if (folder/'numerical'/'numerical-terminal.json').exists():
        from .nominal_numeric_history import read_retained_numeric
        return read_retained_numeric(protocol,cutoff).role(role)
    intent, output, terminal, published = (_read(folder/f'{name}.json') for name in
                                           ('intent', 'output', 'terminal', 'publication'))
    authority.verify()
    record = authority.get(terminal['reservation']['reservation_id'])
    lease = FrozenWorkLease.from_dict(intent['lease'])
    validate_nominal_factor_route(authority, intent['work_key'], intent['binding'], lease, COUNTS)
    actual = ReservedChannelBudget.verify_counts(authority, record['reservation_id'])
    raw = _read(folder/'raw.json')
    combined = output
    output = combined['roles'][role]
    model = _model_class(p['recipes'][role]).load(folder/(role+'-model.json'))
    if (terminal['status'] != 'completed' or record != terminal['reservation'] or record['status'] != 'completed'
            or record['work_key'] != intent['work_key'] or output['plan'] != role_plan or combined['plan'] != plan
            or actual['actual_counts'] != _output_counts(combined['roles']) or any(actual['actual_counts'][k] > COUNTS[k] for k in COUNTS) or actual['test_only'] or not actual['finalized']
            or output['raw_hash'] != fingerprint(raw) or output['model_digest'] != model.digest
            or output['numerical_model'] != numerical_model(model)
            or model.pipeline_id != p['recipes'][role]['pipeline_id']
            or output['solve_calls'] != (1 if type(model) is KnownLinearPredictor else model.receipt['solve_calls'])
            or output['training'] != model.receipt['training']
            or output['training']['cutoff'] != cutoff or output['training']['train_start'] != role_plan['train_start']
            or terminal['output_hash'] != fingerprint(combined)
            or record['result'] != {'nominal_output': fingerprint(combined), 'raw_hash': fingerprint(raw),
                'models_hash': fingerprint({r: combined['roles'][r]['model_digest'] for r in ROLES})}
            or published['terminal_hash'] != file_hash(folder/'terminal.json')):
        raise ValueError('Nominal original artifact/accounting chain differs')
    parent = record['process_identity']
    _kernel(terminal['kernel_before'], terminal['kernel_before'], terminal['kernel_after'], lease, parent, parent)
    _kernel(terminal['kernel_before'], terminal['kernel_after'], published['kernel'], lease, parent, parent)
    _observation(terminal['observation'], lease)
    _observation(published['observation'], lease, previous=terminal['observation'])
    _file_deadline([folder/f'{n}.json' for n in ('raw', 'reference-model', 'candidate-model', 'output', 'terminal', 'publication')], lease)
    _verify_original_rows(protocol, folder, role, model, output)
    return {**output, 'paid_pair_evidence': {'directory': str(folder), 'reservation_id': record['reservation_id'],
        'record_hash': fingerprint(record), 'output_sha256': file_hash(folder/'output.json'),
        'publication_sha256': file_hash(folder/'publication.json'), 'raw_sha256': file_hash(folder/'raw.json')}}

def assemble_atom(protocol, *, authority, atom):
    """All original C-window cold fits; complete fixed support and real paid pairs."""
    if type(protocol) is not NominalFactorProtocol or type(authority) is not MonthlyWorkAuthority:
        raise TypeError('Exact original fixed protocol and wallet required')
    p = protocol.verify(); streams = {}; artifacts = {}; loss=_loss_module()
    for role in ROLES:
        records = [read_month(protocol, authority=authority, atom=atom, cutoff=c, role=role)
                   for c in p['schedule']['cutoffs']]
        rows = [r for record in records for r in record['rows']]
        if len({(r['origin'], r['symbol']) for r in rows}) != len(rows):
            raise ValueError('Duplicate nominal forecast')
        streams[role] = {(r['origin'], r['symbol']): r for r in rows}
        artifacts[role] = records
    if set(streams['reference']) != set(streams['candidate']):
        raise ValueError('Nominal paired support differs')
    pairs = []
    for key in sorted(streams['reference']):
        ref, cand = (streams[role][key] for role in ROLES)
        if any(ref[k] != cand[k] for k in ('label', 'window', 'maturity', 'feature_open_time')):
            raise ValueError('Nominal labels/windows differ')
        pairs.append({'hour': ref['hour'], 'symbol': ref['symbol'], 'label': ref['label'],
            'reference_loss': loss.nominal_total_loss(ref['label'], ref['prediction']),
            'candidate_loss': loss.nominal_total_loss(cand['label'], cand['prediction'])})
    support=tuple((t.isoformat(),s) for t in pd.date_range(p['schedule']['start'],p['schedule']['end'],freq='h',inclusive='left')
                  +pd.Timedelta(seconds=p['schedule']['delay']) for s in p['law']['symbols'])
    rows=tuple(loss.NominalPairRow(k[0],k[1],streams['reference'][k]['label'],
        streams['reference'][k]['prediction'],streams['candidate'][k]['prediction']) for k in sorted(streams['reference']))
    exact=loss.exact_complete_support_risks(support,rows)
    return {'protocol': protocol.digest, 'atom': atom, 'streams': streams, 'artifacts': artifacts,
            'exact_path_risks':{r:[v.numerator,v.denominator] for r,v in zip(ROLES,(exact.reference,exact.candidate))},
            'pairs': pairs, **_score(pairs, p['schedule'], p['law']['symbols']), 'formal_acceptance': False}
