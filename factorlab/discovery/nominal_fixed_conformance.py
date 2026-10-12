"""Necessary original-ledger guard for the NEW four fixed nominal slots.

No caller directory/atom/recipe/risk/clock is accepted. This is not a native
completion reader, a metered task, or permission to count a calibration success.
"""
from pathlib import Path
import importlib
import json

from factorlab.protocol import project_path,fingerprint
from .evidence import file_hash
from .monthly_work_authority import source_files
from .calibration_oracle_evidence import (OracleClaimSpec,_lifecycle,_read as read_plain,
    reconstruct_claim,verify_raw_claim)
from .nominal_cold_conformance import numerical_model,compare_forecast,_raw_conformance,_score
from .v7_lifecycle import stamp

VERSION='nominal-fixed-total-loss-ledger-conformance-v2'
RELATIVE='factorlab/discovery/nominal_fixed_conformance.py'
FIXED=('library_skill','factor1','factor2','factor3')


def _module(name,files):
    module=importlib.import_module('factorlab.discovery.'+name)
    relative='factorlab/discovery/'+name+'.py'
    if (Path(module.__file__).resolve()!=project_path(relative).resolve()
            or files.get(relative)!=file_hash(module.__file__)):
        raise ValueError('Exact canonical new conformance dependency required')
    return module


def _installed():
    files=source_files()
    if Path(__file__).resolve()!=project_path(RELATIVE).resolve() or files.get(RELATIVE)!=file_hash(__file__):
        raise ValueError('Only new canonical source can read registered conformance')
    return files


def _original_execution(permit):
    """Real source/claim/search/controller ledger completeness, not OS authority."""
    from .full_trial_controller import FullTrialController,ControllerStepLease
    from .full_trial_certificate import _claims,_search
    from .neural_forward_ticket import open_existing_authority
    controller=FullTrialController.inspect(permit);steps=controller.verify()
    claims=_claims(permit);search=_search(permit)
    wallet=open_existing_authority(permit.paths['authority']).verify()
    terminal=controller._terminal()
    clock=controller._execution_clock()
    if (len(steps)!=len(controller.schedule.due_at) or not steps
            or not all(controller.step_completed(s) for s in steps)
            or not all(c['state']=='complete' for c in claims.values())
            or not search['months'] or not all(m['state']=='complete' for m in search['months'])
            or not search['provider_attempt_coverage']['complete']
            or not search['provider_mechanism_verified'] or terminal is None or clock is None):
        raise ValueError('Original full trial incomplete; nonselection cannot erase infrastructure failure')
    lease=ControllerStepLease.from_dict(clock['lease'],controller.schedule)
    if not lease.evaluate(terminal['observed'],steps[-1]['witness']['observed']).allowed:
        raise ValueError('Original full controller closed outside its immutable clock')
    return {'trial':permit.trial,'controller':fingerprint(controller.identity),'steps':steps,
        'claims':claims,'search':search,'wallet':wallet,'terminal':terminal}


def _compare_pairs(actual,nominal):
    expected={(r['hour'],r['symbol']):r for r in nominal['pairs']}
    if len(expected)!=len(nominal['pairs']) or len(actual['pairs'])!=len(expected):
        raise ValueError('Actual full frozen C support missing or duplicated; no denominator trimming')
    seen=set()
    for row in actual['pairs']:
        key=(row['hour'],row['symbol']);want=expected.get(key)
        if key in seen or want is None or any(row.get(k)!=v for k,v in want.items()):
            raise ValueError('Actual raw label/loss differs from nominal complete support')
        seen.add(key)


def _selected_score(lc,spec,nominal,execution):
    """Read already committed original statistic; never score/publish a claim."""
    from .claim_protocol import ClaimProtocol
    from .quarantine_execution import verify_decision
    if spec.slot=='library_skill':
        protocol=ClaimProtocol(lc);protocol.verify()
        endpoint=protocol.endpoints['library_skill'].to_dict()
        if any(endpoint[k]!=v for k,v in spec.to_dict().items()):raise ValueError('Original library endpoint differs')
        row=protocol._row('library_skill')
        if row is None or row[1] is None:raise ValueError('Original library statistic uncommitted')
        saved=json.loads(row[1]);result=saved['result']
        if fingerprint(saved)!=row[2]:raise ValueError('Original library score hash differs')
    else:
        ordinal=int(spec.slot[-1]);rows=lc.db.execute('SELECT id,result,status FROM batches WHERE ordinal=?',(ordinal,)).fetchall()
        if len(rows)!=1 or rows[0][2]!='concluded' or rows[0][1] is None:
            raise ValueError('Original selected factor C has not concluded')
        batch,text,_=rows[0];saved=json.loads(text);result=saved
        if read_plain(lc.directory/'claims'/(fingerprint(batch)+'.json'))!=saved:
            raise ValueError('Original factor score publication differs')
        lc.confirmation_publication(batch)
    decision=verify_decision(lc,result)
    expected=nominal['statistic']
    if any(result.get(k)!=v for k,v in expected.items()):
        raise ValueError('Original frozen statistic/bootstrap result differs')
    if bool(decision['accepted'])!=bool(nominal['numerical_event']):
        raise ValueError('Original numerical decision differs from nominal event')
    if execution['claims'][spec.slot]['score_digest']!=fingerprint(saved):
        raise ValueError('Original full-trial claim changed during read')
    return {'score_digest':fingerprint(saved),'numerical_event':bool(decision['accepted'])}


def _models_and_forecasts(lc,spec,nominal,selected):
    operations={fingerprint(json.loads(t)):json.loads(t) for t, in lc.db.execute('SELECT record FROM operations')}
    models={};schedule=selected['schedule'];proof=[]
    recipes={r:{k:v for k,v in getattr(spec,r).items() if k!='complexity'} for r in ('reference','candidate')}
    for pair in nominal['actual_pairs']:
        for role in ('reference','candidate'):
            evidence=pair[role+'_evidence'];stream=getattr(spec,role+'_stream')
            row=lc.forecasts.db.execute('SELECT prediction FROM forecasts WHERE run=? AND stream=? AND origin=? AND symbol=? AND task=?',
                (lc.plan.digest,stream,evidence['origin'],pair['symbol'],'volatility')).fetchone()
            if row is None:raise ValueError('Original actual forecast missing')
            forecast=json.loads(row[0])
            if fingerprint(forecast)!=evidence['forecast_hash']:raise ValueError('Original forecast bytes changed')
            wanted=nominal['streams'][role][(evidence['origin'],pair['symbol'])]
            activation=operations[evidence['activation_hash']]
            if activation['kind']!='activated':raise ValueError('Original activation operation missing')
            job=lc.db.execute('SELECT cutoff,model,pipeline FROM jobs WHERE id=?',(activation['detail']['job'],)).fetchone()
            if job is None or job[1]!=forecast['model_id'] or json.loads(job[2])!=recipes[role]:
                raise ValueError('Actual original selected job/recipe differs')
            identifier=forecast['model_id']
            if identifier not in models:
                model,identity=lc.registry.load(identifier,expected_pipeline=recipes[role])
                expected_version=3 if recipes[role].get('kind')=='v7_known_linear' else 4
                if identity['version']!=expected_version:raise ValueError('Original paid known/pool model admission required')
                models[identifier]=model
            model=models[identifier]
            matches=[r for r in nominal['artifacts'][role] if r['plan']['cutoff']==job[0]]
            if len(matches)!=1:raise ValueError('Original monthly cold model has no exact nominal counterpart')
            ref=matches[0];training=model.receipt['training'];registered=ref['training']
            keys=('cutoff','train_start','rolling_hash','rows','common_train_rows','common_support_hash')
            if (numerical_model(model)!=ref['numerical_model']
                    or any(training.get(k)!=registered.get(k) for k in keys)
                    or training['cutoff']!=wanted['cutoff'] or training['train_start']!=wanted['train_start']):
                raise ValueError('Original full cold numerical state, training support or readiness differs')
            compare_forecast(wanted,forecast,origin=evidence['origin'],cutoff=job[0],
                activation=activation['detail']['origin'],max_lag=lc.plan.prediction_max_lag_seconds)
            proof.append({'forecast_hash':evidence['forecast_hash'],'activation_hash':evidence['activation_hash'],
                'model_digest':model.digest,'role':role,'paid_nominal_pair':ref['paid_pair_evidence']})
    return fingerprint(proof)


def observe_registered_fixed(census,slot):
    """Read necessary actual conformance; atoms and paths come only from parent.

    Full-N consumers must independently obtain original typed native completion
    and new prospective nominal registration before using the conditional counts.
    Missing native evidence is NOT converted to completion by this reader.
    """
    outcome={'version':VERSION,'slot':slot,'trial_denominator':1,'state':'failed_or_unknown_retained',
        'truth':'unknown','conformant':False,'nominal_event':False,'guarded_event':False,
        'conservative_error_count':1,'power_success_count':0,'formal_acceptance':False,
        'native_completion_verified':False,'verification_cost_registered':False,
        'pending':['new_four_slot_actual_registration','original_full_trial_native_completion','metered_conformance_read']}
    try:
        files=_installed();producer=_module('nominal_factor_census',files)
        if type(census) is not producer.NominalFactorCensus or slot not in FIXED:
            raise TypeError('Exact original new four-slot census and fixed slot required')
        p=census.verify();v=p['validation'];selected=p['slots'][slot]
        from .full_calibration_registry import QuarantinedTrialPermit
        from .finite_market_oracle import _read_draw
        permit=QuarantinedTrialPermit(census.registry,v);permit.verify()
        _,_,specs,generator=producer._original(census.registry,v['cell'],v['index'])
        spec=specs[slot]
        execution=_original_execution(permit);execution_hash=fingerprint(execution)
        draw=_read_draw(generator,permit);atom=draw['atom'];directory=permit.paths['runtime']
        outcome.update(census_digest=census.digest,trial=v,actual_atom=atom,execution_digest=execution_hash)
        if spec.candidate is None:
            if (selected['state']!='not_selected' or execution['claims'][slot].get('status')!='concluded_no_candidate'
                    or execution['claims'][slot]['accepted'] is not False):
                raise ValueError('Nonselection lacks the original complete selection outcome')
            nominal=None
            result={'state':'not_selected','truth':'not_selected','conformant':True,
                'exact_finite_risk':None,'nominal_event':False,
                'registered_atoms':128,'atom_state':'no_selected_recipe_no_work',
                'conditional_counts_after_required_native_registration':{'error':0,'power':0}}
        else:
            nominal=census.assemble(slot,atom)
            manifest=read_plain(directory/'forward.json')
            if (fingerprint(manifest['identity'])!=manifest['sha256']
                    or manifest['identity']['source_files']!=files or manifest['identity']['rolling']!=p['spec']):
                raise ValueError('Original actual source/rolling scope changed')
            actual=reconstruct_claim(directory,spec);verify_raw_claim(directory,actual)
            _compare_pairs(actual,nominal)
            with _lifecycle(directory) as lc:
                permit.verify(lc);lc.verify()
                model_digest=_models_and_forecasts(lc,spec,{**nominal,'actual_pairs':actual['pairs']},selected)
                score=_selected_score(lc,spec,nominal,execution)
            raw=generator.reconstruct_atom(atom,population='evaluation')
            provenance=_raw_conformance(directory,raw,selected['schedule']['raw_start'],selected['schedule']['raw_end'])
            truth=census.read_census_risk(slot)
            if truth['complete_atoms']!=128 or truth['truth'] not in ('null','alternative'):
                raise ValueError('Full original four-slot nominal risk not available')
            event=score['numerical_event'];null=truth['truth']=='null'
            result={'state':'ledger_conformant_native_pending','truth':truth['truth'],'conformant':True,
                'nominal_event':event,'nominal_risk_digest':fingerprint(truth),
                'actual_model_forecast_digest':model_digest,'actual_raw_provenance':provenance,
                'conditional_counts_after_required_native_registration':{'error':int(event and null),'power':int(event and not null)}}
        # A typed nominal agreement is necessary; no earlier cached true escapes
        # a changed source, actual trial or original draw at read completion.
        if (census.verify()!=p or fingerprint(_original_execution(permit))!=execution_hash
                or _read_draw(generator,permit)!=draw or _installed()!=files):
            raise ValueError('Original source/trial/draw changed across actual conformance read')
        return {**outcome,**result}
    except Exception as error:
        return {**outcome,'failure':type(error).__name__+':'+str(error)[:1800]}
