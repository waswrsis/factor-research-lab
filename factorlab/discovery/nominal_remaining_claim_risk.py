"""Remaining-claim conditional oracle arithmetic and original-source read bridge.

No new trials, wallets, fits or provider calls. Existing joint registry results
keep their joint identity: they are NEVER reinterpreted as nominal conformance.
Finite-sample bounds need declared independent whole paths and a bounded latent
loss domain. Local resolved-model logs cannot establish either assumption.
"""
from fractions import Fraction
from pathlib import Path
import math

from factorlab.protocol import fingerprint, project_path

VERSION = 'remaining-claim-original-oracle-bridge-v1'
RELATIVE = 'factorlab/discovery/nominal_remaining_claim_risk.py'
SLOTS = ('factor1', 'factor2', 'factor3', 'algorithm')
DELTA = Fraction(1, 200)


def _pair(q):
    return [q.numerator, q.denominator]


def _ln_upper(q):
    """Exact rational upper bound on ln(q), q>=1; no rounded logarithm."""
    if type(q) is not Fraction or q < 1:
        raise ValueError('Exact rational log argument >=1 required')
    # q = 2**k * m, 1<=m<2; atanh series has x<=1/3.
    k = max(0, q.numerator.bit_length()-q.denominator.bit_length())
    if q < 2**k:k -= 1
    m = q/Fraction(2**k)
    def series(value):
        x = (value-1)/(value+1)
        terms = 32
        lower = 2*sum((x**(2*j+1)/Fraction(2*j+1) for j in range(terms)), Fraction())
        remainder = 2*x**(2*terms+1)/(Fraction(2*terms+1)*(1-x*x))
        return lower+remainder
    return k*series(Fraction(2))+series(m)


def _sqrt_upper(q):
    if type(q) is not Fraction or q < 0:
        raise ValueError('Exact nonnegative squared radius required')
    denominator = 2**64
    n = (q.numerator*denominator**2+q.denominator-1)//q.denominator
    root = math.isqrt(n)
    if root*root < n:root += 1
    result = Fraction(root, denominator)
    if result*result < q:raise AssertionError('Outward radius arithmetic')
    return result


def _bounded_missing_enclosure(points, *, weights, upper, tail):
    """Private unprivileged arithmetic; callers cannot mint a truth certificate.

    Each point is one independently generated WHOLE algorithm execution (or
    one fixed-recipe whole-path replicate). Missingness may depend arbitrarily
    on its values. A missing latent bounded pair is enclosed by [0, upper].
    This DOES NOT prove latent loss is defined/bounded on failed executions.
    """
    if (type(points) is not tuple or type(weights) is not tuple
            or not 1 <= len(points) == len(weights) <= 2_000_000
            or any(type(w) is not Fraction or w <= 0 for w in weights)
            or sum(weights) != 1 or type(upper) is not Fraction or upper <= 0
            or type(tail) is not Fraction or not 0 < tail < Fraction(1, 2)):
        raise ValueError('Complete fixed support, positive exact weights and original bounds required')
    sums = [Fraction(), Fraction()];missing = Fraction();completed = 0
    for w, point in zip(weights, points):
        if point is None:
            missing += w;continue
        if (type(point) is not tuple or len(point) != 2
                or any(type(v) is not Fraction or not 0 <= v <= upper for v in point)):
            raise ValueError('Original bounded exact pair or retained missing slot required')
        for role in range(2):sums[role] += w*point[role]
        completed += 1
    squared_weights = sum(w*w for w in weights)
    radius = _sqrt_upper(upper*upper*squared_weights*_ln_upper(1/tail)/2)
    intervals = tuple((max(Fraction(), s-radius), min(upper, s+missing*upper+radius)) for s in sums)
    return {'reference': intervals[0], 'candidate': intervals[1], 'radius': radius,
        'registered_paths': len(points), 'completed_paths': completed,
        'missing_paths': len(points)-completed, 'missing_weight': missing,
        'squared_weights': squared_weights, 'per_risk_tail_alpha': tail}


def _truth(interval):
    rlo, rhi = interval['reference'];clo, chi = interval['candidate']
    if rlo <= 0:return 'uncertain'
    if 1-clo/rhi <= DELTA:return 'null'
    if 1-chi/rlo > DELTA:return 'alternative'
    return 'uncertain'


def _counts(truth, complete, event):
    """Private event arithmetic. No authentication and no conditional denominator."""
    if truth not in {'null', 'alternative', 'uncertain', 'not_selected'} or type(complete) is not bool:
        raise ValueError('Original truth state and strict completion required')
    if not complete:
        if event is not None:raise ValueError('Failed trial cannot supply a discovery event')
        return {'denominator': 1, 'possible_error': 1, 'power_success': 0}
    if type(event) is not bool:raise TypeError('Original guarded event required')
    if truth == 'not_selected' and event:raise ValueError('Unselected factor cannot be accepted')
    return {'denominator': 1, 'possible_error': int(event and truth in {'null', 'uncertain'}),
            'power_success': int(event and truth == 'alternative')}


def _point(result, upper):
    """Original binary64 PATH risks, not hours masquerading as replicates."""
    if (result is None or result['coverage'] != 1. or result['pair_coverage'] != 1.
            or result['path_risks'] is None):return None
    if (len(result['pairs']) != result['nominal_symbol_hours']
            or len({(r['hour'],r['symbol']) for r in result['pairs']}) != len(result['pairs'])):
        raise ValueError('Whole nominal hour-symbol support required')
    for row in result['pairs']:
        if any(type(row[k]) not in (int,float) or not math.isfinite(row[k])
               or not 0 <= row[k] <= upper for k in ('reference_loss','candidate_loss')):
            raise ValueError('Accepted loss violates original source-bound global proof')
    values = tuple(result['path_risks'][k] for k in ('reference','candidate'))
    if any(type(v) not in (int,float) or not math.isfinite(v) or not 0 <= v <= upper for v in values):
        raise ValueError('Original path risk violates global bound')
    return tuple(Fraction(v) for v in values)


def _installed(registry):
    from .full_calibration_registry import FullTrialRegistry
    from .monthly_work_authority import source_files
    from .evidence import file_hash
    if type(registry) is not FullTrialRegistry:
        raise TypeError('Exact existing full registry required; nominal registry is not an alias')
    files = source_files()
    if (Path(__file__).resolve() != project_path(RELATIVE).resolve()
            or files.get(RELATIVE) != file_hash(__file__)
            or registry.identity['source_files'] != files):
        raise ValueError('New reviewed source integration and fresh original registration required')
    registry.verify()
    return files


def _fixed(registry, cell, index, atom, spec, generator, upper):
    from .full_calibration_registry import QuarantinedTrialPermit
    from .fixed_recipe_oracle import FixedRecipeOracle
    from .finite_market_oracle import _read_draw,verify_publications
    trial = registry.identity_for(cell,index,oracle_index=atom,estimand_slot=spec.slot)
    permit = QuarantinedTrialPermit(registry,trial)
    if not (permit.paths['root']/'enrollment.json').is_file():return None, {'state':'missing_original_enrollment'}
    permit.verify();draw = _read_draw(generator,permit)
    if draw['atom'] != atom:raise ValueError('Original fixed atom order changed')
    if not (permit.paths['root']/'fixed_oracle/identity.json').is_file():return None, {'state':'missing_original_controller'}
    report = FixedRecipeOracle.inspect(permit).report()
    if report['original_estimand_hash'] != spec.digest:
        raise ValueError('Replica replaced the validation-selected full recipe')
    publications=verify_publications(generator,permit)
    point = (_point(report['claim'],upper) if report['state']=='complete'
             and publications['all_registered_receipts_complete'] is True else None)
    return point, {'state':'complete_original_path' if point is not None else 'incomplete_consumed',
                   'execution_digest':fingerprint({'report':report,'publications':publications})}


def _algorithm(registry,cell,index,atom,spec,generator,upper,frozen):
    from .full_calibration_registry import QuarantinedTrialPermit, EVENT
    from .full_trial_controller import FullTrialController
    from .full_trial_certificate import _claims,_search
    from .calibration_oracle_evidence import _lifecycle,_original_specs,reconstruct_claim,verify_raw_claim
    from .calibration_runtime_contract import _original_operations
    from .finite_market_oracle import _read_draw,verify_publications
    from .neural_forward_ticket import open_existing_authority
    from .v7_lifecycle import stamp
    trial = registry.identity_for(cell,index,oracle_index=atom)
    permit = QuarantinedTrialPermit(registry,trial)
    if not (permit.paths['root']/'enrollment.json').is_file():return None, {'state':'missing_original_enrollment'}
    permit.verify();draw = _read_draw(generator,permit)
    if draw['atom'] != atom:raise ValueError('Whole-policy original atom order changed')
    with _lifecycle(permit.paths['runtime']) as lc:
        enrollment=[v for v in _original_operations(lc) if v['kind']==EVENT]
        if len(enrollment)!=1 or stamp(enrollment[0]['at']) < stamp(frozen['frozen_at']):
            raise ValueError('Independent algorithm enrolled before selected estimand freeze')
    if not (permit.paths['root']/'controller/identity.json').is_file():return None, {'state':'missing_original_controller'}
    controller=FullTrialController.inspect(permit);steps=controller.verify()
    original=QuarantinedTrialPermit(registry,registry.identity_for(cell,index))
    if controller.schedule.to_dict()!=FullTrialController.inspect(original).schedule.to_dict():
        raise ValueError('Independent whole algorithm changed original schedule or budget')
    claims,search=_claims(permit),_search(permit)
    wallet=open_existing_authority(permit.paths['authority']).verify()
    evidence={'controller':controller.identity,'steps':steps,'claims':claims,'search':search,'wallet':wallet,'draw':draw}
    done=(len(steps)==len(controller.schedule.due_at) and all(controller.step_completed(r) for r in steps)
          and all(c['state']=='complete' for c in claims.values()) and bool(search['months'])
          and all(v['state']=='complete' for v in search['months']) and search['provider_mechanism_verified'])
    if not done:return None, {'state':'incomplete_original_policy','execution_digest':fingerprint(evidence)}
    actual=next(s for s in _original_specs(permit) if s.slot=='algorithm')
    left,right=spec.to_dict(),actual.to_dict()
    for value in (left,right):value.pop('reference_stream');value.pop('candidate_stream')
    if left!=right:raise ValueError('Original whole adaptive policy/window changed')
    result=reconstruct_claim(permit.paths['runtime'],actual)
    raw=verify_raw_claim(permit.paths['runtime'],result)
    receipts=verify_publications(generator,permit)
    point=_point(result,upper)
    if receipts['all_registered_receipts_complete'] is not True:point=None
    evidence.update(claim=result,raw=raw,receipts=receipts)
    return point, {'state':'complete_original_policy' if point is not None else 'incomplete_support',
                   'execution_digest':fingerprint(evidence)}


def read_registered_remaining_claims(registry,cell,index):
    """Read original registered replicas, never execute or enroll any replica.

    This is an unmetered source-bound bridge diagnostic, NOT a public nominal
    certificate. New nominal registration/conformance and paid verification
    must be wired explicitly; this API cannot silently migrate old records.
    """
    from .full_calibration_registry import QuarantinedTrialPermit
    from .calibration_oracle_evidence import _frozen_estimands
    from .full_mechanism_contract import verify_mechanism_contract,ASSUMPTIONS
    from .finite_market_oracle import proof_terms,LOSS_BOUND
    files=_installed(registry);plan=registry.plan.acceptance
    trial=registry.identity_for(cell,index);validation=QuarantinedTrialPermit(registry,trial)
    validation.verify();frozen,specs=_frozen_estimands(validation)
    contract=registry.plan.mechanism_contract
    generators=verify_mechanism_contract(contract,plan);generator=generators[trial['scenario_digest']]
    generator.verify_registry(registry)
    proof=proof_terms()
    if (generator.identity['proof']!=proof or proof['loss_upper_bound']!=LOSS_BOUND
            or proof['accepted_domain']!='finite_nonnegative_binary64_y_and_p'):
        raise ValueError('Original frozen DGP/accepted-loss bound changed')
    n=registry.plan.oracle_paths
    if n!=generator.law.support_size:raise ValueError('Complete original registered support required')
    total=sum(c.repetitions for c in plan.cells)
    tail=min(Fraction(1,40)/(4*5*total),Fraction(plan.oracle_risk_alpha))
    upper=Fraction(LOSS_BOUND);weights=(Fraction(1,n),)*n
    output={};census=[]
    for spec in specs:
        if spec.slot not in SLOTS:continue
        if spec.candidate is None:
            output[spec.slot]={'state':'original_nonselection','estimand':spec.to_dict(),
                'oracle_interval':None,'conditional_truth':'not_selected','power_success':0,
                'registered_paths':n,'complete_paths':0,'known_boundary':False}
            census.extend({'slot':spec.slot,'atom':atom,'state':'original_nonselection_no_replica_credit'} for atom in range(n))
            continue
        points=[]
        for atom in range(n):
            try:
                point,row=(_algorithm(registry,cell,index,atom,spec,generator,upper,frozen) if spec.slot=='algorithm'
                           else _fixed(registry,cell,index,atom,spec,generator,upper))
            except (ValueError,OSError,KeyError,TypeError) as error:
                point=None;row={'state':'failed_or_unknown_consumed','reason':type(error).__name__+':'+str(error)[:500]}
            points.append(point);census.append({'slot':spec.slot,'atom':atom,**row})
        interval=_bounded_missing_enclosure(tuple(points),weights=weights,upper=upper,tail=tail)
        output[spec.slot]={'state':'conditional_bounded_risk_diagnostic','estimand':spec.to_dict(),
            'oracle_interval':{k:[_pair(v) for v in interval[k]] for k in ('reference','candidate')},
            'conditional_truth':_truth(interval),'known_boundary':False,
            'registered_paths':n,'complete_paths':interval['completed_paths'],'missing_paths':interval['missing_paths'],
            'missing_weight':_pair(interval['missing_weight']),'radius':_pair(interval['radius'])}
    if tuple(output)!=SLOTS:raise ValueError('Original remaining four slots must be retained')
    if _installed(registry)!=files or _frozen_estimands(validation)[0]!=frozen:
        raise ValueError('Original source or selection changed during reconstruction')
    return {'version':VERSION,'registry_digest':registry.digest,'coordinate':trial,
        'registered_estimand':'original_joint_selected_recipes_and_whole_adaptive_policy',
        'nominal_conformance_transfer':False,'claims':output,'census':census,
        'five_slots':['library_skill',*SLOTS],'library_exact_boundary_not_replaced':True,
        'source_files':files,'frozen_estimands_hash':fingerprint(frozen),'loss_proof':proof,
        'confidence':{'per_risk_tail_alpha':_pair(tail),'oracle_alpha':[1,40],
            'monte_carlo_alpha':[1,40],'registered_validation_N':total,'tails':4*5*total},
        'assumptions':list(ASSUMPTIONS),'assumption_status':'declared_not_proved_by_logs',
        'unknown_provider_kernel_certified':False,'latent_failure_loss_domain_proved':False,
        'conditional_interval_requires':'independent whole paths with defined bounded potential losses; missing data do not prove this',
        'verification_cost_registered':False,'formal_gate_eligible':False,'formal_acceptance':False,
        'pending':['new_nominal_four_slot_prospective_registration_and_original_conformance',
                   'whole_policy_bounded_potential_loss_domain_on_unobserved_failures',
                   'provider_kernel_stationarity_and_independence_assumptions_not_OS_facts',
                   'metered_native_readback_and_full_N_consumer_integration']}
