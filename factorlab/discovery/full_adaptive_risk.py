"""Original full-controller path risk enclosures under a prospective mechanism.

No callback, success bit, caller losses, fixed-model replacement or conditional
cold-fit report is accepted. Every registered path executes the actual original
controller with its own original shared UTC wallet. Missing/failed paths remain
in the support census and prevent an enclosure for the affected estimand.
"""
from dataclasses import asdict
import math
from factorlab.protocol import fingerprint
from factorlab.discovery.calibration_acceptance import CLAIMS, bounded_path_envelope
from factorlab.discovery.full_calibration_registry import FullTrialRegistry, QuarantinedTrialPermit
from factorlab.discovery.v7_storage import immutable_record

VERSION = 'v7-original-full-controller-risk-v1'


def prepare_original_oracle(validation_permit):
    from .calibration_oracle_evidence import freeze_oracle_estimands
    # This existing producer rejects any oracle enrollment preceding the
    # original selected-estimand freeze. It never enrolls or spends work.
    return freeze_oracle_estimands(validation_permit)


def reconstruct_full_risk(registry, cell, index):
    if type(registry) is not FullTrialRegistry:
        raise TypeError('Original full registry required')
    registry.verify()
    from .full_mechanism_contract import verify_mechanism_contract
    from .full_trial_controller import FullTrialController
    from .full_trial_certificate import _search, _claims
    from .calibration_oracle_evidence import (_frozen_estimands, _original_specs,
        reconstruct_claim, verify_raw_claim)
    from .finite_market_oracle import _read_draw, LOSS_BOUND, proof_terms
    from .neural_forward_ticket import open_existing_authority
    validation = QuarantinedTrialPermit(registry, registry.identity_for(cell,index))
    points = {s: [] for s in CLAIMS}; census = []
    if registry.plan.mechanism_contract is None:
        return {'version': VERSION, 'expected_paths': registry.plan.oracle_paths,
            'claims': {s: None for s in CLAIMS}, 'census': [],
            'pending': ['prospective_joint_mechanism_contract'], 'scope': 'unknown'}
    generators = verify_mechanism_contract(registry.plan.mechanism_contract, registry.plan.acceptance)
    generator = generators[validation.trial['scenario_digest']]
    generator.verify_registry(registry)
    _read_draw(generator, validation)
    if not (validation.paths['root']/'oracle_estimands.json').is_file():
        return {'version': VERSION, 'expected_paths': registry.plan.oracle_paths,
            'claims': {s: None for s in CLAIMS}, 'census': [],
            'pending': ['original_selected_estimands_frozen_before_oracle_enrollment'], 'scope': 'unknown'}
    frozen, specs = _frozen_estimands(validation)
    original_controller = FullTrialController.inspect(validation)
    _, endpoint_scope = _claims(validation, include_scope=True)
    nonselected = {s.slot for s in specs if s.candidate is None}
    for oracle_index in range(registry.plan.oracle_paths):
        trial = registry.identity_for(cell,index,oracle_index=oracle_index)
        permit = QuarantinedTrialPermit(registry,trial)
        if not (permit.paths['root']/'enrollment.json').is_file():
            census.append({'oracle_index':oracle_index,'state':'missing_or_orphaned'}); continue
        permit.verify()
        from .calibration_oracle_evidence import _lifecycle
        from .calibration_runtime_contract import _original_operations
        from .full_calibration_registry import EVENT
        from .v7_lifecycle import stamp
        with _lifecycle(permit.paths['runtime']) as lc:
            enrollment=[o for o in _original_operations(lc) if o['kind']==EVENT]
            if len(enrollment)!=1 or stamp(enrollment[0]['at'])<stamp(frozen['frozen_at']):
                raise ValueError('Original oracle enrollment preceded selected estimand freeze')
        draw = _read_draw(generator,permit)
        if draw['atom'] != oracle_index:
            raise ValueError('Original complete finite support was resampled')
        if not (permit.paths['root']/'controller/identity.json').is_file():
            census.append({'oracle_index':oracle_index,'state':'missing_controller'}); continue
        controller=FullTrialController.inspect(permit)
        if controller.schedule.to_dict()!=original_controller.schedule.to_dict():
            raise ValueError('Oracle changed original full controller schedule or resource limits')
        steps=controller.verify()
        claims, oracle_scope = _claims(permit, include_scope=True)
        if oracle_scope != endpoint_scope:
            raise ValueError('Oracle changed original frozen six-month endpoint/rule/recipe scope')
        search=_search(permit)
        wallet=open_existing_authority(permit.paths['authority']).verify()
        complete=(len(steps)==len(controller.schedule.due_at)
            and all(controller.step_completed(r) for r in steps)
            and all(c['state']=='complete' for c in claims.values())
            and bool(search['months']) and all(m['state']=='complete' for m in search['months'])
            and search['provider_mechanism_verified'])
        evidence={'controller':controller.identity,'steps':steps,'claims':claims,'search':search,'wallet':wallet,'draw':draw,
                  'six_month_claim_scope':oracle_scope}
        row={'oracle_index':oracle_index,'state':'complete' if complete else 'incomplete_consumed',
             'execution_digest':fingerprint(evidence),'claims':{}}
        if complete:
            # Exact full learner/policy equality, not a claim-slot alias. Oracle
            # selection differs => unknown fixed-learner truth, never substitute
            # that oracle's winner. Algorithm policy varies its selections as
            # specified by its original fixed entire-policy descriptor.
            actual = {s.slot:s for s in _original_specs(permit)}
            for spec in specs:
                if spec.slot != 'algorithm':
                    continue  # Fixed selected learners have dedicated paid replicas below.
                other=actual[spec.slot]
                left=spec.to_dict(); right=other.to_dict()
                for v in (left,right):
                    v.pop('reference_stream'); v.pop('candidate_stream')
                if left!=right:
                    row['claims'][spec.slot]={'state':'original_estimand_not_executed'}; continue
                if spec.candidate is None:
                    row['claims'][spec.slot]={'state':'original_nonselection'}; continue
                result=reconstruct_claim(permit.paths['runtime'],other)
                raw=verify_raw_claim(permit.paths['runtime'],result)
                # Complete nominal support avoids conditioning risk on observed
                # hours or successful symbols. Undefined original path is unknown.
                valid=(result['coverage']==1. and result['pair_coverage']==1. and result['path_risks'] is not None)
                if valid:
                    for pair in result['pairs']:
                        if any(not math.isfinite(pair[k]) or not 0<=pair[k]<=LOSS_BOUND for k in ('reference_loss','candidate_loss')):
                            raise ValueError('Original accepted binary64 loss violates global proof')
                    points[spec.slot].append(result['path_risks'])
                row['claims'][spec.slot]={'state':'complete' if valid else 'incomplete_support',
                    'original_claim_digest':fingerprint(result),'raw_source_digest':fingerprint(raw)}
        census.append(row)
    # Never borrow this path's winner for the validation-selected fixed recipe.
    # Each independently registered replica has its own original wallet and
    # actual source/fit/publication history; the adaptive oracle above is intact.
    from .fixed_recipe_oracle import FixedRecipeOracle, SLOTS
    fixed_census=[]
    for spec in specs:
        if spec.slot not in SLOTS or spec.candidate is None:
            continue
        for atom in range(registry.plan.oracle_paths):
            trial=registry.identity_for(cell,index,oracle_index=atom,estimand_slot=spec.slot)
            fixed=QuarantinedTrialPermit(registry,trial)
            if not (fixed.paths['root']/'enrollment.json').is_file():
                fixed_census.append({'slot':spec.slot,'atom':atom,'state':'missing_or_orphaned'}); continue
            fixed.verify()
            if not (fixed.paths['root']/'fixed_oracle/identity.json').is_file():
                fixed_census.append({'slot':spec.slot,'atom':atom,'state':'missing_original_controller'}); continue
            execution=FixedRecipeOracle.inspect(fixed).report()
            if execution['original_estimand_hash']!=spec.digest:
                raise ValueError('Fixed replica replaced original validation-selected estimand')
            if execution['state']=='complete':
                points[spec.slot].append(execution['claim']['path_risks'])
            fixed_census.append({'slot':spec.slot,'atom':atom,'state':execution['state'],
                'execution_digest':fingerprint(execution),'wallet':execution['wallet']})
    digest=fingerprint({'registry':registry.digest,'cell':cell,'index':index,'frozen':frozen,
        'census':census,'fixed_census':fixed_census,
        'mechanism':registry.plan.mechanism_contract,'proof':proof_terms()})
    envelopes={}; pending=[]
    if any(r['state']!='complete' for r in census):
        pending.append('all_registered_original_full_controller_oracle_paths')
    if any(r['state']!='complete' for r in fixed_census):
        pending.append('all_registered_original_fixed_recipe_replicas')
    from .full_mechanism_contract import FIXED_REPLICA_TRANSFER
    transfer_declared = FIXED_REPLICA_TRANSFER in registry.plan.mechanism_contract['assumptions']
    if fixed_census and not transfer_declared:
        # A measured independent two-stream replica is not proof of the loaded
        # adaptive service's resource/availability distribution. Keep G11 red
        # until that scientific transfer is independently justified.
        pending.append('fixed_recipe_replica_to_original_service_availability_equivalence_unproved')
    for slot in CLAIMS:
        if slot in nonselected:
            envelopes[slot]=None
            continue
        rows=points[slot]
        if len(rows)!=registry.plan.oracle_paths:
            envelopes[slot]=None; pending.append(slot+':incomplete_original_full_path_support'); continue
        envelope=bounded_path_envelope([r['reference'] for r in rows],[r['candidate'] for r in rows],
            loss_upper_bound=LOSS_BOUND,expected_paths=registry.plan.oracle_paths,
            oracle_digest=digest,plan=registry.plan.acceptance)
        # Outward exact averaging and conservative floating transcendental
        # headroom; binary64 numerical error cannot narrow the claimed band.
        from .finite_market_oracle import _exact_interval
        radius=math.nextafter(LOSS_BOUND*math.sqrt(math.log(1/registry.plan.acceptance.oracle_risk_alpha)
            /(2*registry.plan.oracle_paths))*(1+2**-40), math.inf)
        ranges={}
        for role in ('reference','candidate'):
            lo,hi=_exact_interval([r[role] for r in rows])
            ranges[role]=[max(0.,math.nextafter(lo-radius,-math.inf)),
                min(float(LOSS_BOUND),math.nextafter(hi+radius,math.inf))]
        envelopes[slot]={**asdict(envelope),**ranges}
    return {'version':VERSION,'registry_digest':registry.digest,'expected_paths':registry.plan.oracle_paths,
        'claims':envelopes,'nonselected_slots':sorted(nonselected),'census':census,
        'fixed_recipe_census':fixed_census,
        'fixed_replica_transfer':{'prospectively_declared':transfer_declared,
            'condition':FIXED_REPLICA_TRANSFER,
            'empirically_proved':False},
        'calibration_cost':{'registered_full_policy_oracles':registry.plan.oracle_paths,
            'registered_fixed_replicas':len(fixed_census),
            'wallet_policy':'each_original_replica_same_standard_limits_no_production_credit'},
        'pending':pending,'evidence_digest':digest,
        'scope':'original_selected_full_recipes_and_whole_adaptive_policy_under_declared_joint_mechanism',
        'inequality':'independent_bounded_nonidentical_stratified_path_Hoeffding_equal_finite_atom_weights',
        'independence':'prospectively_declared_modelling_assumption_not_proved_server_fact'}


def produce_full_risk(registry,cell,index):
    value=reconstruct_full_risk(registry,cell,index)
    digest=fingerprint(value)
    immutable_record(registry.directory/'full_risks'/(digest+'.json'),value)
    return {'artifact_digest':digest,'pending':value['pending']}


def verify_full_risk(registry,cell,index,artifact_digest):
    from .full_calibration_registry import _read
    value=reconstruct_full_risk(registry,cell,index)
    if fingerprint(value)!=artifact_digest or _read(registry.directory/'full_risks'/(artifact_digest+'.json'))!=value:
        raise ValueError('Full original oracle artifact changed')
    return value
