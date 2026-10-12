"""Five-slot/full-N readback inside an original full-scope proof allocation.

No new work lease, wallet, native completion or provider IID is created here.
Current live validation remains provisional until its real retained unit closes.
"""
from pathlib import Path
import importlib
import json
from fractions import Fraction
from factorlab.protocol import fingerprint,project_path

VERSION='nominal-original-full-five-actual-conformance-consumer-v3'
RELATIVE='factorlab/discovery/nominal_full_five_consumer.py'
CLAIMS=('library_skill','factor1','factor2','factor3','algorithm')
FIXED=CLAIMS[:4]

def _installed(design):
    from .monthly_work_authority import source_files
    from .evidence import file_hash
    files=source_files()
    if files!=design.identity['source_files']:raise ValueError('Whole registered consumer source changed')
    for name in ('nominal_full_five_consumer','nominal_fixed_conformance','nominal_factor_census',
       'nominal_algorithm_risk_reader','nominal_algorithm_total_risk','nominal_full_scope_proof',
       'calibration_acceptance','nominal_total_loss'):
        module=importlib.import_module('factorlab.discovery.'+name);rel='factorlab/discovery/'+name+'.py'
        if Path(module.__file__).resolve()!=project_path(rel).resolve() or files.get(rel)!=file_hash(module.__file__):
            raise ValueError('Exact canonical full-five dependency required: '+name)
    return files

def _paths(design):
    return (design.parent.directory/'nominal_full_five_consumer/identity.json',
            design.directory/'full_five_consumer.json')

def _expected(design):
    a=design.parent.plan.acceptance
    return {'version':VERSION,'parent_digest':design.parent.digest,'design_digest':design.digest,
      'source_files':_installed(design),'acceptance_digest':a.digest,'acceptance':json.loads(json.dumps(a.to_dict())),
      'claims':list(CLAIMS),'original_N':sum(c.repetitions for c in a.cells),
      'fixed_atoms_per_selected_slot':128,'oracle_paths':design.parent.plan.oracle_paths,
      'allocation':'existing_validation_single_proof300_remaining_within_original7200',
      'execution_order':'original_algorithm_readback_then_fixed_four_then_native_closure',
      'new_seconds':0,'new_fit_or_provider_credit':0,'extra_wallet':False,
      'failed_missing_unmetered':'error1_power0_fullN','provider_IID_granted':False,
      'global_reader':'fixed_last_original_whole_oracle_single_proof300_remaining',
      'final_coordinate':final_coordinate(design),
      'final_requires_all_validation_natural_completion':True,
      'heldback_oracle_in_validation_risk':'original_unknown_retained_no_rewrite',
      'global_current_trial':'whole_oracle_not_a_validation_repetition',
      'algorithm_unknown_truth_counts':'complete_source_conformance_error_equals_original_accepted_otherwise_error1_power0',
      'algorithm_witness':'private_same_invocation_producer_complete_raw_model_forecast_statistic',
      'formal_acceptance':False}

class FullFiveConsumerPlan:
    @classmethod
    def freeze(cls,design):
        from .nominal_algorithm_total_risk import AlgorithmTotalRiskRegistry,_fresh_parent
        from .nominal_cold_conformance import _write
        from filelock import FileLock
        if type(design)is not AlgorithmTotalRiskRegistry:raise TypeError('Exact new complete design required')
        design.verify();_fresh_parent(design.parent);value=_expected(design);paths=_paths(design)
        with FileLock(str(design.directory/'.full-five-consumer-creation.lock'),timeout=0):
            _fresh_parent(design.parent)
            if paths[0].parent.exists() or any(p.exists()for p in paths):raise ValueError('Existing/partial consumer cannot be replaced')
            paths[0].parent.mkdir()
            for path in paths:_write(path,value)
        return cls.inspect(design)
    @classmethod
    def inspect(cls,design):
        from .nominal_algorithm_total_risk import AlgorithmTotalRiskRegistry
        from .nominal_cold_conformance import _read
        if type(design)is not AlgorithmTotalRiskRegistry:raise TypeError('Exact original complete design required')
        design.verify();value=_expected(design)
        if any(_read(path)!=value for path in _paths(design)):raise ValueError('Consumer/source/whole acceptance anchors changed')
        obj=cls.__new__(cls);obj.design=design;obj.identity=value;obj.digest=fingerprint(value);return obj

def descriptor(design):
    plan=FullFiveConsumerPlan.inspect(design)
    return {'version':VERSION,'digest':plan.digest,'allocation':plan.identity['allocation'],
            'claims':list(CLAIMS),'original_N':plan.identity['original_N'],'new_seconds':0,
            'final_coordinate':plan.identity['final_coordinate']}

def _task(design,cell=None,index=None):
    from . import nominal_full_scope_proof as native
    from .nominal_algorithm_risk_reader import _active_original_task
    task=native._PROOF.get()
    if type(task)is not native._ProofTask:raise TypeError('Original active paid proof task required')
    coordinate=task.identity['coordinate']
    if cell is not None and (cell!=coordinate['cell'] or index!=coordinate['index']):
        raise ValueError('A validation consumer cannot borrow another proof allocation')
    return _active_original_task(design,coordinate['cell'],coordinate['index'])

def _unknown(slot,reason):
    return {'slot':slot,'truth':'unknown','conformant':False,'event':False,
      'conditional_error':1,'conditional_power':0,'evidence_hash':None,'reason':reason[:800]}

def _fixed_numbers(slot,outcome):
    """Private arithmetic only; callers must use original typed census reader."""
    if (slot not in FIXED or type(outcome)is not dict or outcome.get('slot')!=slot
        or outcome.get('conformant')is not True or outcome.get('trial_denominator')!=1
        or type(outcome.get('trial_denominator'))is not int or outcome.get('formal_acceptance')is not False):
        return _unknown(slot,'original_fixed_conformance_missing')
    truth=outcome.get('truth');event=outcome.get('nominal_event')
    if type(event)is not bool or truth not in ('null','alternative','not_selected'):
        return _unknown(slot,'original_fixed_truth_unknown')
    if truth=='not_selected' and event:raise ValueError('A nonselection cannot be a discovery')
    expected={'error':int(event and truth=='null'),'power':int(event and truth=='alternative')}
    if outcome.get('conditional_counts_after_required_native_registration')!=expected:
        raise ValueError('Original fixed conditional counts differ')
    return {'slot':slot,'truth':truth,'conformant':True,'event':event,
      'conditional_error':expected['error'],'conditional_power':expected['power'],
      'evidence_hash':fingerprint(outcome),'reason':None}

def consume_original_validation(design,cell,index):
    """Fixed worker hook; original proof300 is already alive, never issued here."""
    task=_task(design,cell,index)
    from .nominal_full_scope_proof import _write_pair
    roots=tuple(root/'five-consumer'for root in task.roots)
    if any(p.exists()for p in roots):raise ValueError('Consumer attempt already exists; no replay')
    for root in roots:root.mkdir()
    coordinate=task.identity['coordinate']
    intent={'version':VERSION,'coordinate':coordinate,'native_binding_hash':fingerprint(task.identity),
            'lease':task.lease.to_dict(),'retry':False}
    _write_pair(roots,'intent.json',intent);task.check()
    plan=FullFiveConsumerPlan.inspect(design);files=_installed(design)
    from .nominal_factor_census import NominalFactorCensus
    from .nominal_fixed_conformance import observe_registered_fixed
    census_directory=design.parent.directory/'nominal_factor_censuses'/coordinate['population_trial_id']
    rows=[]
    for slot in FIXED:
        task.check()
        try:
            census=NominalFactorCensus(census_directory)
            if census.registry.digest!=design.parent.digest or census.identity['validation']!=coordinate:
                raise ValueError('Fixed census belongs to another original validation')
            outcome=observe_registered_fixed(census,slot)
            row=_fixed_numbers(slot,outcome)
            if row['conformant'] and (outcome.get('trial')!=coordinate or outcome.get('census_digest')!=census.digest):
                raise ValueError('Fixed conformance coordinate/census changed')
            task.check();census.verify();task.check()
            rows.append(row)
        except (ValueError,OSError,KeyError,TypeError,ArithmeticError) as error:
            task.check() # Real deadline/owner failure aborts the worker, never a free fallback.
            rows.append(_unknown(slot,type(error).__name__+':'+str(error)))
    rows.append(_algorithm_row(task,design))
    result={'version':VERSION,'plan_digest':plan.digest,'coordinate':coordinate,
      'lease':task.lease.to_dict(),'native_binding_hash':fingerprint(task.identity),
      'claims':rows,'registered_denominator':1,'allocation':plan.identity['allocation'],
      'retained_native_required':True,'native_completed':False,'formal_acceptance':False,
      'provider_IID_certified':False,'new_task_credit':False}
    task.check()
    if FullFiveConsumerPlan.inspect(design).digest!=plan.digest or _installed(design)!=files:
        raise ValueError('Original all-five registered source changed during readback')
    _write_pair(roots,'result.json',result);task.check()
    return result

def validate_retained_consumer(value,design,coordinate,*,binding_hash,lease):
    """Schema plus dual original metered records; caller still needs real OS reader."""
    from .nominal_full_scope_proof import FullScopeProofPlan,_pair
    plan=FullFiveConsumerPlan.inspect(design);native=FullScopeProofPlan.inspect(design.parent)
    roots=tuple(p/'five-consumer'for p in native.roots(coordinate))
    intent={'version':VERSION,'coordinate':coordinate,'native_binding_hash':binding_hash,'lease':lease,'retry':False}
    if _pair(roots,'intent.json')!=intent or _pair(roots,'result.json')!=value:
        raise ValueError('Missing/partial actual paid consumer records')
    expected_keys={'version','plan_digest','coordinate','lease','native_binding_hash','claims',
      'registered_denominator','allocation','retained_native_required','native_completed',
      'formal_acceptance','provider_IID_certified','new_task_credit'}
    if (type(value)is not dict or set(value)!=expected_keys or value['version']!=VERSION
       or value['plan_digest']!=plan.digest or value['coordinate']!=coordinate
       or value['lease']!=lease or value['native_binding_hash']!=binding_hash
       or type(value['registered_denominator'])is not int or value['registered_denominator']!=1
       or value['allocation']!=plan.identity['allocation'] or value['retained_native_required']is not True
       or any(value[k]is not False for k in ('native_completed','formal_acceptance','provider_IID_certified','new_task_credit'))):
        raise ValueError('Original five-slot consumer allocation differs')
    _validate_rows(value['claims'])
    algorithm=value['claims'][-1]
    if algorithm['conformant']:
        from .nominal_full_scope_proof import _producer_publication
        publication=_producer_publication(native,coordinate)
        proof=publication['proof'];conformance=proof['algorithm_actual_conformance']
        claim=proof['actual_execution']['claims']['algorithm']
        risk=next(r for r in proof['claims']if r['slot']=='algorithm')
        if (conformance is None or conformance['complete_actual_support']is not True
            or proof['numeric_fit_events'] or claim['state']!='complete'
            or type(claim['accepted'])is not bool or claim['accepted']is not algorithm['event']
            or conformance['accepted']is not claim['accepted']
            or conformance['score_hash']!=claim['score_digest']
            or any(risk[k]!=0 for k in ('missing_rows','reference_failure_rows','candidate_failure_rows'))):
            raise ValueError('Historical algorithm error reduction lacks original full actual conformance')
        record={'accepted':claim['accepted'],'proof_hash':fingerprint(proof),
            'source_hash':fingerprint(publication['identity']['source_files']),'coordinate':coordinate,
            'claim':claim,'conformance':conformance,'provider_IID_certified':False,'population_truth':'unknown'}
        if algorithm['evidence_hash']!=fingerprint(record):
            raise ValueError('Algorithm private observation differs from original paid producer publication')
    return value

def _validate_rows(rows):
    if type(rows)is not list or len(rows)!=5 or tuple(r.get('slot')for r in rows)!=CLAIMS:
        raise ValueError('Exactly five original ordered slots required')
    for row in rows:
        if set(row)!={'slot','truth','conformant','event','conditional_error','conditional_power','evidence_hash','reason'}:
            raise ValueError('Unknown conditional row schema')
        if (type(row['conformant'])is not bool or type(row['event'])is not bool
            or any(type(row[k])is not int or row[k]not in (0,1)for k in ('conditional_error','conditional_power'))):
            raise ValueError('Strict complete one-repetition counts required')
        if not row['conformant']:
            if row['truth']!='unknown' or row['event'] or row['conditional_error']!=1 or row['conditional_power']!=0:
                raise ValueError('Unknown evidence must retain worst-case counts')
        else:
            truth=row['truth'];event=row['event']
            if row['slot']=='algorithm':
                if (truth!='unknown' or row['conditional_error']!=int(event) or row['conditional_power']!=0
                    or type(row['evidence_hash'])is not str or len(row['evidence_hash'])!=64):
                    raise ValueError('Unknown algorithm truth cannot grant power or waive actual failure')
                continue
            if (truth not in ('null','alternative','not_selected')
                or truth=='not_selected' and event or row['conditional_error']!=int(event and truth=='null')
                or row['conditional_power']!=int(event and truth=='alternative')
                or type(row['evidence_hash'])is not str or len(row['evidence_hash'])!=64):
                raise ValueError('Fixed original conditional counts changed')
    return rows

def _count_rows(rows):
    """Pure numbers, no public evidence input. Missing typed native is all worst."""
    if rows is None:rows=[_unknown(s,'missing_original_native_or_paid_readback')for s in CLAIMS]
    _validate_rows(rows)
    return {'denominator':1,'family_error':int(any(r['conditional_error']for r in rows)),
      'claims':{r['slot']:{'possible_error':r['conditional_error'],'power_success':r['conditional_power']}for r in rows}}

def read_retained_validation(design,cell,index):
    """Must be called inside another genuine existing proof; no free heavy reads."""
    task=_reading_task(design);task.check()
    from .nominal_full_scope_proof import FullScopeProofPlan,read_retained,FullScopeNativeObservation
    plan=FullScopeProofPlan.inspect(design.parent);coordinate=plan.coordinate(cell,index)
    if coordinate==task.identity['coordinate']:
        return {'coordinate':coordinate,'native_completed':False,'counts':_count_rows(None),
                'reason':'current_unit_is_not_naturally_complete'}
    try:
        observation=read_retained(plan,coordinate)
        if type(observation)is not FullScopeNativeObservation:raise TypeError('Exact retained original native observation required')
        observation.assert_current();identity=observation.identity
        if task.identity['coordinate']['oracle_index'] is not None:
            _require_before_final_start(task,observation)
        value=identity['actual_proof']['full_five_readback']
        from .nominal_full_scope_proof import _pair
        proof=_pair(plan.roots(coordinate),'proof.json')
        validate_retained_consumer(value,design,coordinate,binding_hash=identity['native_binding_hash'],lease=proof['lease'])
        observation.assert_current();task.check()
        return {'coordinate':coordinate,'native_completed':True,'counts':_count_rows(value['claims']),
                'native_hash':fingerprint(identity),'consumer_hash':fingerprint(value),'reason':None}
    except (ValueError,OSError,KeyError,TypeError,ArithmeticError) as error:
        task.check()
        return {'coordinate':coordinate,'native_completed':False,'counts':_count_rows(None),
                'reason':type(error).__name__+':'+str(error)[:700]}

def aggregate_original_full_N(design):
    """Actual metered read entry; does not start a new task or fake its own exit.

    Only the fixed last original whole-oracle worker invokes this after its
    original paid producer, within its sole proof300. No new task is created.
    All validation native completions must already exist; no records are retried.
    """
    task=_final_task(design);plan=FullFiveConsumerPlan.inspect(design)
    from .calibration_acceptance import exact_binomial_bounds
    from .nominal_full_scope_proof import _write_pair
    roots=tuple(p/'full-N-consumer'for p in task.roots)
    if any(p.exists()for p in roots):raise ValueError('Original full-N read already attempted')
    for root in roots:root.mkdir()
    _write_pair(roots,'intent.json',{'version':VERSION,'plan_digest':plan.digest,
        'lease':task.lease.to_dict(),'native_binding_hash':fingerprint(task.identity),'retry':False});task.check()
    cells=[];acceptance=design.parent.plan.acceptance
    for cell in acceptance.cells:
        rows=[];errors=0;slots={s:{'possible_errors':0,'power_successes':0}for s in CLAIMS}
        for index in range(cell.repetitions):
            task.check();value=read_retained_validation(design,cell.name,index);task.check()
            counts=value['counts'];errors+=counts['family_error']
            for s in CLAIMS:
                slots[s]['possible_errors']+=counts['claims'][s]['possible_error']
                slots[s]['power_successes']+=counts['claims'][s]['power_success']
            rows.append(value)
        _,upper=exact_binomial_bounds(errors,cell.repetitions,tail_alpha=acceptance.tail_alpha)
        for s in CLAIMS:
            slots[s]['denominator']=cell.repetitions
            slots[s]['power_lower']=exact_binomial_bounds(slots[s]['power_successes'],cell.repetitions,tail_alpha=acceptance.tail_alpha)[0] if s in cell.power_slots else None
        cells.append({'cell':cell.name,'registered_N':cell.repetitions,'possible_family_errors':errors,
          'family_error_upper':upper,'claims':slots,'native_complete':sum(r['native_completed']for r in rows),
          'rows_hash':fingerprint(rows)})
    # Scan every original repetition while the same original clock permits.
    # Any missing native result stays in N, but prevents final certification.
    if any(c['native_complete']!=c['registered_N']for c in cells):
        _write_pair(roots,'incomplete.json',{'version':VERSION,'cells':cells,
          'original_N':plan.identity['original_N'],'formal_acceptance':False,
          'reason':'all_original_validation_natural_completions_required'})
        task.check()
        raise ValueError('Final original oracle cannot certify incomplete validation native history')
    result={'version':VERSION,'plan_digest':plan.digest,'original_N':plan.identity['original_N'],
      'cells':cells,'monte_carlo_alpha':acceptance.monte_carlo_alpha,'oracle_alpha':acceptance.oracle_alpha,
      'tail_alpha':acceptance.tail_alpha,'formal_acceptance':False,'provider_IID_certified':False,
      'current_unit_is_oracle_not_validation':True,'publication':'provisional_requires_original_natural_exit',
      'final_coordinate':final_coordinate(design),'all_validation_native_complete':True,
      'allocation_lease':task.lease.to_dict(),'native_binding_hash':fingerprint(task.identity)}
    task.check();_installed(design);task.check();_write_pair(roots,'result.json',result);task.check()
    return result



def final_coordinate(design):
    cell=design.parent.plan.acceptance.cells[-1]
    return design.parent.identity_for(cell.name,cell.repetitions-1,
                                     oracle_index=design.parent.plan.oracle_paths-1)

def _final_task(design):
    import os,threading
    from . import nominal_full_scope_proof as native
    task=native._PROOF.get()
    if (type(task)is not native._ProofTask or task.key is not native._KEY
        or task.owner!=(os.getpid(),threading.get_ident()) or task.closed
        or task.producer.permit.registry.digest!=design.parent.digest
        or task.producer.transport.identity['design_digest']!=design.digest
        or task.identity['coordinate']!=final_coordinate(design)):
        raise ValueError('Only fixed original final whole-oracle proof may aggregate')
    task.check();return task

def _reading_task(design):
    from . import nominal_full_scope_proof as native
    task=native._PROOF.get()
    if type(task)is not native._ProofTask:raise TypeError('Original paid read scope required')
    if task.identity['coordinate'].get('oracle_index')is None:return _task(design)
    return _final_task(design)

def final_original_hook(design):
    from . import nominal_full_scope_proof as native
    task=native._PROOF.get()
    if type(task)is not native._ProofTask:raise TypeError('Original fixed worker proof required')
    if task.identity['coordinate']!=final_coordinate(design):return None
    return aggregate_original_full_N(design)

def validate_retained_full_N(value,design,coordinate,*,binding_hash,lease):
    from .nominal_full_scope_proof import FullScopeProofPlan,_pair
    plan=FullFiveConsumerPlan.inspect(design);native=FullScopeProofPlan.inspect(design.parent)
    if coordinate!=final_coordinate(design):
        if value is not None:raise ValueError('Other original task cannot publish final full-N')
        return None
    roots=tuple(p/'full-N-consumer'for p in native.roots(coordinate))
    if any((p/'incomplete.json').exists()for p in roots):raise ValueError('Incomplete final scan remains failed')
    expected={'version':VERSION,'plan_digest':plan.digest,'lease':lease,
              'native_binding_hash':binding_hash,'retry':False}
    if _pair(roots,'intent.json')!=expected or _pair(roots,'result.json')!=value:
        raise ValueError('Final original full-N allocation/results missing or changed')
    acceptance=design.parent.plan.acceptance
    if (type(value)is not dict or value.get('version')!=VERSION or value.get('plan_digest')!=plan.digest
        or value.get('original_N')!=plan.identity['original_N']
        or value.get('final_coordinate')!=coordinate or value.get('all_validation_native_complete')is not True
        or value.get('allocation_lease')!=lease or value.get('native_binding_hash')!=binding_hash
        or value.get('formal_acceptance')is not False or value.get('provider_IID_certified')is not False
        or value.get('monte_carlo_alpha')!=acceptance.monte_carlo_alpha
        or value.get('oracle_alpha')!=acceptance.oracle_alpha or value.get('tail_alpha')!=acceptance.tail_alpha
        or value.get('current_unit_is_oracle_not_validation')is not True
        or value.get('publication')!='provisional_requires_original_natural_exit'
        or type(value.get('cells'))is not list or len(value['cells'])!=len(acceptance.cells)):
        raise ValueError('Final original full-N scope or cost differs')
    from .calibration_acceptance import exact_binomial_bounds
    for saved,cell in zip(value['cells'],acceptance.cells):
        if (saved.get('cell')!=cell.name or type(saved.get('registered_N'))is not int
            or saved['registered_N']!=cell.repetitions or saved.get('native_complete')!=cell.repetitions
            or type(saved.get('possible_family_errors'))is not int
            or not 0<=saved['possible_family_errors']<=cell.repetitions or set(saved.get('claims',{}))!=set(CLAIMS)
            or saved.get('family_error_upper')!=exact_binomial_bounds(saved['possible_family_errors'],cell.repetitions,tail_alpha=acceptance.tail_alpha)[1] or type(saved.get('rows_hash'))is not str or len(saved['rows_hash'])!=64):
            raise ValueError('Original unknown algorithm/full denominator changed')
        for slot in CLAIMS:
            row=saved['claims'][slot]
            if (set(row)!={'possible_errors','power_successes','denominator','power_lower'}
                or any(type(row[k])is not int or not 0<=row[k]<=cell.repetitions for k in ('possible_errors','power_successes'))
                or type(row['denominator'])is not int or row['denominator']!=cell.repetitions):
                raise ValueError('Original per-slot denominator/counts changed')
            lower=exact_binomial_bounds(row['power_successes'],cell.repetitions,tail_alpha=acceptance.tail_alpha)[0] if slot in cell.power_slots else None
            if row['power_lower']!=lower or (slot=='algorithm'and row['power_successes']!=0):
                raise ValueError('Original registered confidence/unknown provider changed')
    return value

def _require_before_final_start(task,observation):
    from .nominal_full_scope_proof import FullScopeNativeObservation
    if type(observation)is not FullScopeNativeObservation:
        raise TypeError('Exact retained original validation observation required')
    observation.assert_current()
    # Read actual original D-Bus timestamps from the private typed observation.
    # All validation processes must have naturally exited BEFORE this final
    # original oracle unit began, not merely while its aggregate is scanning.
    raw=observation._raw
    for finish,start in (('ExecMainExitTimestampMonotonic','unit_start_monotonic_us'),
                         ('ExecMainExitTimestamp','unit_start_realtime_us')):
        if type(raw.get(finish))is not int or type(task.identity.get(start))is not int or not 0<raw[finish]<task.identity[start]:
            raise ValueError('Validation must naturally close before the fixed final oracle starts')


def _algorithm_row(task,design):
    """Only the real current producer's private observation can lower an error."""
    task.check();_installed(design)
    from .nominal_algorithm_producer import _AlgorithmConformance
    witness=getattr(task.producer,'_algorithm_conformance',None)
    if type(witness)is not _AlgorithmConformance:
        return _unknown('algorithm','actual_nominal_algorithm_conformance_not_complete')
    try:
        record=witness.for_original_task(task)
        task.check()
        if type(record['accepted'])is not bool:raise ValueError('Original accepted must be strict bool')
        return {'slot':'algorithm','truth':'unknown','conformant':True,'event':record['accepted'],
          'conditional_error':int(record['accepted']),'conditional_power':0,
          'evidence_hash':fingerprint(record),'reason':'opaque_provider_truth_unknown_actual_path_conformant'}
    except (ValueError,OSError,KeyError,TypeError,ArithmeticError)as error:
        task.check()
        return _unknown('algorithm',type(error).__name__+':'+str(error))
