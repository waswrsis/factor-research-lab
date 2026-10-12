"""Prospective paid full-policy nominal execution and bounded numerical readback.

One original adaptive execution computes the nominal policy. No transcript is
substituted for the provider and no fitted model is re-fitted by this reader.
Returned files remain provisional until independently authenticated native close.
"""
from __future__ import annotations
from contextvars import ContextVar
from pathlib import Path
import hashlib
import json
import math
import os
import threading

VERSION = 'nominal-complete-paid-policy-producer-v1'
RELATIVE = 'factorlab/discovery/nominal_algorithm_producer.py'
_ACTIVE = ContextVar('original_nominal_algorithm_producer', default=None)
_KEY = object()


def _installed():
    from importlib import import_module
    from factorlab.protocol import project_path
    from .monthly_work_authority import source_files
    from .evidence import file_hash
    from .nominal_algorithm_policy import _installed as policy_installed
    files = policy_installed()
    for name in ('nominal_algorithm_producer', 'nominal_algorithm_execution',
                 'nominal_total_loss', 'v7_forward', 'known_library', 'model_registry',
                 'calibration_oracle_evidence', 'full_trial_controller', 'full_trial_certificate',
                 'quarantined_observation', 'work_lease', 'nominal_algorithm_fit',
                 'claim_protocol','confirmatory','nominal_full_five_consumer'):
        module = import_module('factorlab.discovery.'+name)
        relative = 'factorlab/discovery/'+name+'.py'
        if (Path(module.__file__).resolve() != project_path(relative).resolve()
                or files.get(relative) != file_hash(module.__file__)):
            raise ValueError('Canonical complete producer dependency changed: '+name)
    if source_files() != files:
        raise ValueError('Source changed during producer admission')
    return files


def _roots(permit):
    return (permit.paths['root']/'nominal_algorithm_producer',
            permit.registry.directory/'nominal_algorithm_producers'/permit.trial['trial_id'])


def _plain(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _frame_digest(frame):
    import pandas as pd
    from factorlab.protocol import fingerprint
    return fingerprint({'columns': list(frame.columns),
        'dtypes': [str(x) for x in frame.dtypes],
        'rows': hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest(),
        'attrs': _plain(frame.attrs)})


def _number(value):
    """Represent a kernel's nonfinite scalar without nonfinite JSON."""
    value = float(value)
    if math.isnan(value): return {'nonfinite': 'nan'}
    if math.isinf(value): return {'nonfinite': 'positive_inf' if value > 0 else 'negative_inf'}
    return value


def _decoded(value):
    if type(value) is dict:
        if value not in ({'nonfinite':'nan'}, {'nonfinite':'positive_inf'}, {'nonfinite':'negative_inf'}):
            raise ValueError('Unknown nonfinite kernel scalar')
        return None
    if type(value) is not float or not math.isfinite(value):
        raise ValueError('Original finite binary64 kernel output required')
    return value


def _identity(transport, policy, files):
    from .nominal_algorithm_warmboot import policy_binding as warmboot_binding
    from factorlab.protocol import fingerprint
    from .nominal_short_fit_policy import policy_binding_for_design
    return {'version':VERSION,'trial':transport.permit.trial,'source_files':files,
        'transport_hash':fingerprint(transport.identity),'policy_digest':policy.digest,
        'short_fit_policy':policy_binding_for_design(policy.design),
        'warmboot_policy':warmboot_binding(policy),
        'proof_seconds':300,'proof_charge':'one_service_overhead_within_original_7200_no_fit_credit',
        'original_service_clock':transport.identity['original_service_clock'],
        'readiness':'same_prefix_original_admission_then_next_prefix_after_original_work_drain',
        'numeric_fit_policy':__import__('factorlab.discovery.nominal_algorithm_fit',fromlist=['descriptor']).descriptor(),
        'nominal_policy':'original_paid_adaptive_kernels_with_registered_pre_tick_endpoint_order',
        'provider_IID_certified':False,'formal_acceptance':False}


class AlgorithmNominalProducer:
    @classmethod
    def bind(cls, design, permit, *, runtime, authority, generator, endpoints):
        from .nominal_algorithm_execution import AlgorithmExecutionTransport, _exclusive, _write_pair
        # Transport samples the ONLY original service clock before expensive IO.
        transport = AlgorithmExecutionTransport.bind(design, permit, runtime=runtime,
            authority=authority, generator=generator, endpoints=endpoints)
        from .full_trial_controller import ControllerStepLease
        lease = ControllerStepLease.from_dict(transport.identity['original_service_clock'], transport.controller.schedule)
        lease.assert_parent_alive()
        files = _installed()
        from .nominal_algorithm_policy import require_endpoint_policy
        policy = require_endpoint_policy(design, endpoints)
        roots = _roots(permit)
        _exclusive(roots)
        from factorlab.protocol import fingerprint
        identity = _identity(transport,policy,files)
        _write_pair(roots,'identity.json',identity)
        lease.assert_parent_alive()
        return cls.inspect(permit)

    @classmethod
    def inspect(cls, permit):
        from .nominal_algorithm_execution import AlgorithmExecutionTransport, _pair
        from factorlab.protocol import fingerprint
        transport = AlgorithmExecutionTransport.inspect(permit)
        identity = _pair(_roots(permit),'identity.json')
        from .nominal_algorithm_total_risk import AlgorithmTotalRiskRegistry
        from .nominal_algorithm_policy import NominalAlgorithmPolicy
        policy=NominalAlgorithmPolicy.inspect(AlgorithmTotalRiskRegistry.inspect(permit.registry))
        if identity != _identity(transport,policy,_installed()):
            raise ValueError('Producer identity/source/transport changed')
        obj=cls.__new__(cls)
        obj.permit, obj.transport, obj.identity, obj.roots = permit,transport,identity,_roots(permit)
        obj.owner=(os.getpid(),threading.get_ident())
        obj.proof=None
        return obj

    def run_once(self):
        from .nominal_algorithm_execution import _exclusive, _write_pair
        if type(self) is not AlgorithmNominalProducer or _ACTIVE.get() is not None:
            raise ValueError('Exact nonnested original producer required')
        from .full_trial_controller import ControllerStepLease
        self.clock=ControllerStepLease.from_dict(self.identity['original_service_clock'],self.transport.controller.schedule)
        self.clock.assert_parent_alive()
        attempts=tuple(p/'attempt' for p in self.roots)
        _exclusive(attempts)
        _write_pair(attempts,'intent.json',{'identity':self.identity,'clock':self.clock.to_dict()})
        token=_ACTIVE.set(self)
        try:
            self.check()
            result=self.transport.run_once()
            self.check()
            if self.proof is None:
                raise ValueError('Original producer proof hook did not execute')
            from factorlab.protocol import fingerprint
            value={'version':VERSION,'trial':self.permit.trial,
                'identity_hash':fingerprint(self.identity),'transport_result_hash':fingerprint(result),
                'proof':self.proof,'proof_lease':self.proof_lease.to_dict(),
                'controller_terminal_hash':fingerprint(self.transport.controller._terminal()),
                'publication_state':'provisional_requires_original_native_completion',
                'actual_calibration_outcome':{'possible_error':1,'power':0,
                    'reason':'no_new_actual_acceptance_gate_issued_by_nominal_path_producer'},
                'provider_IID_certified':False,'formal_acceptance':False}
            _write_pair(attempts,'result.json',value)
            self.check()
            return value
        except BaseException as error:
            for p in attempts:
                try: _write_pair((p,),'failure.json',{'state':'failed_or_unknown_consumed',
                    'possible_error':1,'power':0,'error_type':type(error).__name__})
                except BaseException: pass
            raise
        finally:
            _ACTIVE.reset(token)

    def check(self):
        if (_ACTIVE.get() is not self or self.owner!=(os.getpid(),threading.get_ident())
                or any((p/'attempt/failure.json').exists() for p in self.roots)):
            raise ValueError('Original producer invocation missing, foreign or failed')
        self.clock.assert_parent_alive()
        if getattr(self,'proof_lease',None) is not None:
            self.proof_lease.assert_parent_alive()


def _active(runtime):
    producer=_ACTIVE.get()
    if producer is None:
        from .nominal_algorithm_policy import _bound
        if _bound(runtime.lifecycle) is not None:
            raise ValueError('Registered new policy cannot bypass its nominal producer')
        return None
    from .v7_forward import ForwardRuntime
    if (type(producer) is not AlgorithmNominalProducer or type(runtime) is not ForwardRuntime
            or runtime.directory != producer.permit.paths['runtime']):
        raise ValueError('Foreign runtime cannot borrow nominal producer')
    producer.check()
    return producer


class _Tick:
    def __init__(self,key,producer,runtime,record):
        if key is not _KEY: raise TypeError('Private original tick capture only')
        self.producer,self.runtime,self.record=producer,runtime,record
        self.used=False


def capture_tick(runtime, bars, history, *, isolated=False):
    """Evaluate the SAME admitted cold kernels before their original forecast IO.

    This is a new registered source policy, not a caller-supplied nominal trace.
    Model activation is the original kernel; its repeated call in tick is an
    idempotent lookup after that original activation. No fit or work is added.
    """
    producer=_active(runtime)
    if producer is None:return None
    if type(isolated) is not bool or isolated:
        raise ValueError('Original serving isolation cannot become nominal successful computation')
    from .nominal_algorithm_policy import _live_prefix
    from .nominal_algorithm_execution import _write_pair
    from .v7_lifecycle import stamp
    import numpy as np
    prefix=_live_prefix(runtime.lifecycle)
    origin=stamp(prefix['logical_at'])
    name=f"{prefix['hour']:05d}.json"
    if any((p/'ticks'/name).exists() for p in producer.roots):
        raise ValueError('Nominal tick was already computed; no retry')
    _write_pair(tuple(p/'tick-intents' for p in producer.roots),name,{'prefix':prefix})
    recipes=runtime._serving_recipes(origin,runtime.lifecycle.now())
    rows=[]
    current_symbols=set()
    for symbol in runtime.config.symbols:
        group=bars[bars.symbol==symbol]
        if not group.empty:
            latest=group.iloc[-1]
            if (latest.prediction_origin and stamp(latest.available_at)==origin
                    and stamp(latest.open_time)==origin.floor('h')-__import__('pandas').Timedelta(hours=1)):
                current_symbols.add(symbol)
    from .nominal_algorithm_fit import numeric_unavailability
    for stream,recipe in recipes.items():
        numeric=numeric_unavailability(runtime,stream,origin,recipe)
        model=(runtime._activate_ready(stream,origin,expected_pipeline=recipe)
               if current_symbols else None)
        if model is not None:
            predictions,support=model['model'].predict_history(history)
        for symbol in runtime.config.symbols:
            group=bars[bars.symbol==symbol]
            reason=None; prediction=None; model_id=None
            latest=None if group.empty else group.iloc[-1]
            if symbol not in current_symbols:
                reason='original_source_no_current_origin'
            elif numeric is not None:
                reason='original_numeric_cold_fit_unavailable'
            elif model is None:
                reason='original_no_admitted_matching_model'
            else:
                positions=np.flatnonzero(((history.metadata.symbol==symbol)&
                    (history.metadata.feature_open_time==latest.open_time)).to_numpy())
                model_id=model['model_id']
                if len(positions)!=1 or not support[positions[0]]:
                    reason='original_kernel_no_prediction_support'
                else:
                    prediction=_number(predictions[positions[0]])
                    if type(prediction) is dict:
                        reason='original_kernel_nonfinite_prediction'
            rows.append({'stream':stream,'symbol':symbol,'reason':reason,
                'model_id':model_id,'prediction':prediction,
                'pipeline':None if model is None else model['pipeline'],'numeric_failure':numeric})
        producer.check()
    record={'prefix':prefix,'frame_digest':_frame_digest(bars),'recipes':recipes,'rows':rows}
    _write_pair(tuple(p/'ticks' for p in producer.roots),name,record)
    producer.check()
    return _Tick(_KEY,producer,runtime,record)


def skip_nonfinite_output(ticket, stream, symbol, prediction):
    if ticket is None:return False
    if (type(ticket) is not _Tick or ticket.used
            or _active(ticket.runtime) is not ticket.producer):
        raise ValueError('Missing original nominal scalar context')
    found=[r for r in ticket.record['rows'] if r['stream']==stream and r['symbol']==symbol]
    if len(found)!=1:raise ValueError('Actual kernel output lacks nominal coordinate')
    value=_number(prediction)
    if value!=found[0]['prediction']:
        raise ValueError('Actual numerical output differs from nominal kernel')
    if type(value) is dict:
        if found[0]['reason']!='original_kernel_nonfinite_prediction':
            raise ValueError('Nonfinite output did not originate in the admitted model')
        return True  # The actual policy records absence, never a successful NaN forecast.
    return False


def finish_tick(ticket):
    if ticket is None:return
    if type(ticket) is not _Tick or ticket.used or _active(ticket.runtime) is not ticket.producer:
        raise ValueError('Foreign or reused original nominal tick')
    ticket.used=True
    from .nominal_algorithm_policy import _live_prefix
    from .nominal_algorithm_execution import _write_pair
    from factorlab.protocol import fingerprint
    if _live_prefix(ticket.runtime.lifecycle)!=ticket.record['prefix']:
        raise ValueError('Nominal and actual computations crossed observation prefix')
    lc=ticket.runtime.lifecycle
    for row in ticket.record['rows']:
        found=lc.forecasts.db.execute('SELECT prediction FROM forecasts WHERE run=? AND stream=? '
            'AND origin=? AND symbol=? AND task=?',(lc.plan.digest,row['stream'],
             ticket.record['prefix']['logical_at'],row['symbol'],'volatility')).fetchone()
        if row['reason'] is not None:
            if found is not None:raise ValueError('Actual forecast contradicts original nominal inability')
        else:
            if found is None:raise ValueError('Missing actual prediction is infrastructure failure, not nominal B')
            actual=json.loads(found[0])
            if (actual['value'] != row['prediction'] or actual['model_id'] != row['model_id']
                    or actual['observation_prefix'] != ticket.record['prefix']):
                raise ValueError('Actual prediction differs from complete nominal kernel')
    _write_pair(tuple(p/'tick-completions' for p in ticket.producer.roots),
        f"{ticket.record['prefix']['hour']:05d}.json",{'record_hash':fingerprint(ticket.record)})
    ticket.producer.check()


def finish_original_transport(transport, clock):
    """Fixed original transport hook, not an injected proof callback."""
    producer=_ACTIVE.get()
    if type(producer) is not AlgorithmNominalProducer or transport is not producer.transport:
        raise ValueError('Complete new transport must run through its exact nominal producer')
    producer.check()
    from .nominal_algorithm_execution import _exclusive, _write_pair
    from .work_lease import FrozenWorkLease
    from factorlab.protocol import fingerprint
    roots=tuple(p/'proof' for p in producer.roots)
    _exclusive(roots)
    producer.proof_lease=FrozenWorkLease.issue(300)
    _write_pair(roots,'intent.json',{'identity_hash':fingerprint(producer.identity),
        'lease':producer.proof_lease.to_dict(),'original_clock':clock.to_dict()})
    producer.check()
    proof=_recompute(producer)
    from .full_trial_certificate import _claims, _search
    from .neural_forward_ticket import open_existing_authority
    claims,scope=_claims(producer.permit,include_scope=True)
    if len(claims)!=5 or any(row['state']!='complete' for row in claims.values()):
        raise ValueError('Original actual five-slot endpoint work incomplete')
    producer.check()
    search=_search(producer.permit)
    if (not search['provider_attempt_coverage']['complete']
            or not search['provider_mechanism_verified']
            or any(m['state']!='complete' for m in search['months'])
            or any(a['record']['status']!='completed' for a in search['attempts'])):
        raise ValueError('Provider infrastructure failure or incomplete original search remains unknown')
    producer.check()
    steps=transport.controller.verify()
    if (len(steps)!=len(transport.controller.schedule.due_at)
            or not all(transport.controller.step_completed(row) for row in steps)):
        raise ValueError('Original complete controller step witness missing')
    authority=open_existing_authority(producer.permit.paths['authority'])
    wallet=authority.verify()  # Original authority opens/closes its own SQL transaction.
    proof['actual_execution']={'claims':claims,'scope':scope,
        'controller_hash':fingerprint(transport.controller.identity),
        'steps_hash':fingerprint(steps),'search_hash':fingerprint(search),
        'wallet_hash':fingerprint(wallet),'producer_hash':fingerprint(producer.identity)}
    producer.check()
    terminal=transport.controller._close_execution(clock,'nominal_complete_path_proof')
    proof['controller_terminal_hash']=fingerprint(terminal)
    from .work_lease import _observe, _evaluate
    observed=_observe(producer.proof_lease)
    if not _evaluate(producer.proof_lease,observed).allowed:
        raise ValueError('Original proof lease expired before completion observation')
    proof['proof_completed_observation']=observed
    _write_pair(roots,'result.json',proof)
    producer.check()
    producer.proof=proof
    producer._algorithm_conformance=None
    if proof['algorithm_actual_conformance'] is not None:
        outcome=proof['algorithm_actual_conformance']
        if outcome['accepted'] is not claims['algorithm']['accepted'] or outcome['score_hash']!=claims['algorithm']['score_digest']:
            raise ValueError('Original algorithm claim changed after full nominal comparison')
        witness=object.__new__(_AlgorithmConformance)
        witness._key=_CONFORMANCE_KEY;witness._producer=producer;witness._owner=producer.owner
        witness._proof_hash=fingerprint(proof);witness._accepted=outcome['accepted']
        producer._algorithm_conformance=witness
    producer.check()
    return proof


def _complete_cold_jobs(lc, specs, numerical=()):
    """Never substitute an old model for missing original scheduled cold work."""
    import pandas as pd
    from .v7_lifecycle import stamp
    required=set()
    for spec in specs:
        if spec.slot not in ('library_skill','algorithm'):continue
        if lc.plan.training_window_days is None:
            cutoffs=pd.date_range(spec.start,spec.end,freq='MS',inclusive='left')
        else:
            from .rolling_cadence import RollingCadence
            cutoffs=RollingCadence().fit_cutoffs(lc._cadence_anchor(),spec.start,spec.end,
                                               include_current_cycle=False)
        for cutoff in cutoffs:
            lc.fit_cutoff(cutoff)
            for stream in (spec.reference_stream,spec.candidate_stream):
                required.add((cutoff.isoformat(),stream))
    if lc.plan.training_window_days is None and len(required)!=18:
        raise ValueError('Original shared K0 plus distinct HAR/policy six-month cold scope changed')
    if lc.plan.training_window_days is not None and not required:
        raise ValueError('Complete elapsed cold scope has no original data-ready fit cycle')
    jobs=[]
    for cutoff,stream in sorted(required):
        rows=lc.db.execute('SELECT id,status,failure,model FROM jobs WHERE cutoff=? AND stream=?',
                          (cutoff,stream)).fetchall()
        numeric={r['job'] for r in numerical}
        if len(rows)==1 and rows[0][0] in numeric:
            if rows[0][1]!='planned' or rows[0][2]!='numerical_failure' or rows[0][3] is not None:
                raise ValueError('Numeric cold job cannot claim an admitted model')
            jobs.append({'job':rows[0][0],'outcome':'authenticated_numeric_failure'})
            continue
        if (len(rows)!=1 or rows[0][1] not in ('verified','activated')
                or rows[0][2] is not None or not rows[0][3]):
            raise ValueError('Complete scheduled cold work missing or unaffordable; no stale-model substitute')
        jobs.append(rows[0][0])
    from .nominal_algorithm_warmboot import complete_jobs
    jobs.extend(complete_jobs(lc))
    return jobs


def _recompute(producer):
    """Read all fixed slots; recompute original cold-model predictions and labels."""
    import numpy as np
    import pandas as pd
    from factorlab.protocol import fingerprint
    from .nominal_algorithm_execution import _pair
    from .calibration_oracle_evidence import _lifecycle, _original_specs, _stream
    from .quarantined_observation import QuarantinedObservation, snapshot_for_prefix
    from .known_library import build_feature_history
    from .nominal_total_loss import NominalPairRow, exact_complete_support_risks
    from .v7_lifecycle import stamp
    _installed(); producer.check()
    specs=_original_specs(producer.permit)
    if len(specs)!=5:raise ValueError('All original five endpoint slots required')
    original_steps=producer.transport.controller.verify()
    expected={}
    for step in original_steps:
        prefix=step['attempt']['observation_prefix']
        name=f"{prefix['hour']:05d}.json"
        if name in expected:raise ValueError('Original schedule repeated a nominal observation')
        expected[name]=prefix
    for root in producer.roots:
        for folder in ('tick-intents','ticks','tick-completions'):
            if {p.name for p in (root/folder).glob('*.json')}!=set(expected):
                raise ValueError('Orphan, partial or missing original nominal tick')
    with _lifecycle(producer.permit.paths['runtime']) as lc:
        producer.permit.verify(lc); lc.verify()
        from .nominal_algorithm_fit import verify_failed_jobs,verify_numeric_row
        numerical=verify_failed_jobs(producer,lc)
        cold_jobs=_complete_cold_jobs(lc,specs,numerical)
        observer=QuarantinedObservation.from_lifecycle(lc)
        if observer is None:raise ValueError('Original nominal observation disappeared')
        state=observer.inspect(lc)
        ticks={}; publications=[]
        for h in range(state['events']):
            producer.check()
            name=f'{h:05d}.json'
            paths=tuple(p/'ticks' for p in producer.roots)
            if not any((p/name).exists() for p in paths):continue
            record=_pair(paths,name)
            if record['prefix']!=expected.get(name):
                raise ValueError('Nominal computation differs from original paid controller opportunity')
            if _pair(tuple(p/'tick-intents' for p in producer.roots),name)!={'prefix':record['prefix']}:
                raise ValueError('Nominal computation lacks its original intent')
            if _pair(tuple(p/'tick-completions' for p in producer.roots),name)!={'record_hash':fingerprint(record)}:
                raise ValueError('Nominal computation lacks actual conformance completion')
            observer.verify_binding(record['prefix'])
            bars=snapshot_for_prefix(lc,record['prefix'],record['prefix']['logical_at'])
            if _frame_digest(bars)!=record['frame_digest']:
                raise ValueError('Nominal model input source changed')
            history=build_feature_history(bars)
            for row in record['rows']:
                verify_numeric_row(producer,lc,record,row)
                if row['model_id'] is None:continue
                model,identity=lc.registry.load(row['model_id'],expected_pipeline=row['pipeline'])
                if 'admission' not in identity or 'job' not in identity:
                    raise ValueError('Nominal model must retain its original paid cold admission')
                predictions,support=model.predict_history(history)
                opened=stamp(record['prefix']['logical_at']).floor('h')-pd.Timedelta(hours=1)
                positions=np.flatnonzero(((history.metadata.symbol==row['symbol'])&
                    (history.metadata.feature_open_time==opened)).to_numpy())
                if row['reason']=='original_kernel_no_prediction_support':
                    if len(positions)==1 and support[positions[0]]:
                        raise ValueError('Nominal unsupported row became supported')
                elif row['reason'] in (None,'original_kernel_nonfinite_prediction'):
                    if len(positions)!=1 or not support[positions[0]] or _number(predictions[positions[0]])!=row['prediction']:
                        raise ValueError('Original cold-model prediction fails nominal recomputation')
                producer.check()
            ticks[record['prefix']['logical_at']]=record
            publications.append(fingerprint(record))
        claims=[];algorithm_conformance=None
        delay=producer.transport.identity['full_schedule']['due_at'][0]
        delay=stamp(delay).second
        for spec in specs:
            if spec.candidate is None:
                claims.append({'slot':spec.slot,'estimand':spec.to_dict(),
                    'truth':'not_selected','risk':None,'accepted':False})
                continue
            # Original historical activation + fixed recipe / adaptive adoption
            # checks; they do not turn the original partial losses into truth.
            _stream(lc,spec,spec.reference_stream,spec.reference)
            _stream(lc,spec,spec.candidate_stream,None if spec.slot=='algorithm' else spec.candidate)
            producer.check()
            support=[]; pairs=[];stat_rows=[]
            for hour in pd.date_range(spec.start,spec.end,freq='h',inclusive='left'):
                origin=(hour+pd.Timedelta(seconds=delay)).isoformat()
                if origin not in ticks:
                    raise ValueError('Missing original nominal computation cannot be imputed B')
                record=ticks[origin]
                label_at=hour+pd.Timedelta(hours=13,seconds=delay)
                target_h=int((label_at-stamp(observer.generator.law.start))/pd.Timedelta(hours=1))-1
                prefix=observer.binding(target_h)
                bars=snapshot_for_prefix(lc,prefix,label_at)
                for symbol in lc.plan.symbols:
                    support.append((origin,symbol))
                    segment=bars[(bars.symbol==symbol)&(bars.open_time>=hour)&
                                 (bars.open_time<hour+pd.Timedelta(hours=13))].sort_values('open_time')
                    expected=pd.date_range(hour,hour+pd.Timedelta(hours=13),freq='h',inclusive='left')
                    # Registered source absence is a nominal unavailable target;
                    # actual corruption was rejected by original prefix/feed audit.
                    target=None
                    if len(segment)==13 and pd.DatetimeIndex(segment.open_time).equals(expected):
                        target=float(np.sqrt(np.square(np.diff(np.log(segment.close.to_numpy()))).sum()))
                    values=[]
                    for stream in (spec.reference_stream,spec.candidate_stream):
                        found=[r for r in record['rows'] if r['stream']==stream and r['symbol']==symbol]
                        if len(found)>1:raise ValueError('Duplicate original nominal row')
                        if stream in record['recipes'] and len(found)!=1:
                            raise ValueError('Missing nominal eligible stream is unknown failure')
                        value=None if not found or found[0]['reason'] is not None else _decoded(found[0]['prediction'])
                        values.append(value)
                        if found and found[0]['reason'] is None:
                            forecast=lc.forecasts.db.execute('SELECT prediction FROM forecasts WHERE run=? AND stream=? '
                                'AND origin=? AND symbol=? AND task=?',(lc.plan.digest,stream,origin,symbol,'volatility')).fetchone()
                            if forecast is None:raise ValueError('Actual forecast missing during nominal recomputation')
                            forecast=json.loads(forecast[0])
                            if (forecast['value']!=found[0]['prediction'] or forecast['model_id']!=found[0]['model_id']
                                or forecast['observation_prefix']!=record['prefix']):
                                raise ValueError('Actual forecast changed from recomputed nominal kernel')
                            actual=lc.db.execute('SELECT record FROM settlement WHERE run=? AND stream=? '
                                'AND origin=? AND symbol=? AND task=?',(lc.plan.digest,stream,origin,symbol,'volatility')).fetchone()
                            if target is not None:
                                if actual is None:raise ValueError('Missing actual settlement is infrastructure failure')
                                actual=json.loads(actual[0])
                                if (actual['label']!=target or actual['revision']!=fingerprint(segment.revision.tolist())):
                                    raise ValueError('Actual target/revision differs from original nominal raw calculation')
                    pairs.append(NominalPairRow(origin,symbol,target,*values))
                    from .nominal_total_loss import nominal_total_loss
                    stat_rows.append({'hour':hour.isoformat(),'symbol':symbol,'label':target,
                        'revision':fingerprint(segment.revision.tolist()),
                        'reference_loss':nominal_total_loss(target,values[0]),
                        'candidate_loss':nominal_total_loss(target,values[1])})
                producer.check()
            risk=exact_complete_support_risks(tuple(support),tuple(pairs))
            if spec.slot=='algorithm' and not numerical and not any((risk.missing_rows,risk.reference_failure_rows,risk.candidate_failure_rows)):
                algorithm_conformance=_verify_algorithm_statistic(lc,spec,stat_rows,producer)
            claims.append({'slot':spec.slot,'estimand':spec.to_dict(),
                'reference':[risk.reference.numerator,risk.reference.denominator],
                'candidate':[risk.candidate.numerator,risk.candidate.denominator],
                'nominal_rows':risk.nominal_rows,'missing_rows':risk.missing_rows,
                'reference_failure_rows':risk.reference_failure_rows,
                'candidate_failure_rows':risk.candidate_failure_rows,'origins':risk.origins})
        lc.verify(); producer.permit.verify(lc)
    _installed(); producer.check()
    return {'version':VERSION,'claims':claims,'tick_records_hash':fingerprint(publications),
        'ticks':len(ticks),'complete_cold_jobs_hash':fingerprint(cold_jobs),
        'algorithm_actual_conformance':algorithm_conformance,
        'numeric_fit_events':numerical,'numeric_fit_policy':producer.identity['numeric_fit_policy'],
        'source_files':producer.identity['source_files'],
        'conditional_unit':'one_complete_original_paid_adaptive_policy_path',
        'provider_IID_certified':False,'population_risk_certified':False,
        'actual_calibration_outcome':{'possible_error':1,'power':0,
            'reason':'nominal_B_is_not_a_successful_actual_forecast_or_coverage_waiver'},
        'formal_acceptance':False,'publication_state':'provisional_requires_original_native_completion'}


_CONFORMANCE_KEY=object()

class _AlgorithmConformance:
    """Private same-process observation, minted only after the complete paid proof."""
    def __init__(self,*a,**k):raise TypeError('No caller-created algorithm witness')
    def __reduce__(self):raise TypeError('Algorithm witness cannot be serialized')
    def for_original_task(self,task):
        from . import nominal_full_scope_proof as native
        from factorlab.protocol import fingerprint
        if (type(self)is not _AlgorithmConformance or self._key is not _CONFORMANCE_KEY
            or self._owner!=(os.getpid(),threading.get_ident())
            or type(task)is not native._ProofTask or native._PROOF.get()is not task
            or task.key is not native._KEY or task.producer is not self._producer
            or type(task.producer)is not AlgorithmNominalProducer
            or task.identity['coordinate']!=self._producer.permit.trial):
            raise ValueError('Foreign original algorithm conformance observation')
        task.check();files=_installed()
        plan=native.FullScopeProofPlan.inspect(task.producer.permit.registry)
        publication=native._producer_publication(plan,task.identity['coordinate'])
        if (publication['identity']!=self._producer.identity
            or publication['identity']['source_files']!=files
            or publication['result']!=task.result
            or fingerprint(publication['proof'])!=self._proof_hash
            or publication['intent']['lease']!=task.lease.to_dict()
            or publication['proof']['algorithm_actual_conformance']['accepted']is not self._accepted):
            raise ValueError('Original algorithm proof/source/publication changed')
        task.check()
        return {'accepted':self._accepted,'proof_hash':self._proof_hash,
            'source_hash':fingerprint(files),'coordinate':task.identity['coordinate'],
            'claim':publication['proof']['actual_execution']['claims']['algorithm'],
            'conformance':publication['proof']['algorithm_actual_conformance'],
            'provider_IID_certified':False,'population_truth':'unknown'}


def _verify_algorithm_statistic(lc,spec,rows,producer):
    """Recompute the unchanged original single-look rule/RNG from nominal pairs."""
    import pandas as pd
    from .claim_protocol import ClaimProtocol
    from .confirmatory import paired_effect_interval
    from .quarantine_execution import verify_decision
    from .calibration_oracle_evidence import _read
    from factorlab.protocol import fingerprint
    protocol=ClaimProtocol(lc);protocol.verify();endpoint=protocol.endpoints['algorithm']
    if any(endpoint.to_dict()[k]!=v for k,v in spec.to_dict().items()):
        raise ValueError('Original algorithm endpoint differs from nominal support')
    expected={(t.isoformat(),s)for t in pd.date_range(spec.start,spec.end,freq='h',inclusive='left')for s in lc.plan.symbols}
    if len(rows)!=len(expected) or {(r['hour'],r['symbol'])for r in rows}!=expected:
        raise ValueError('Algorithm actual conformance requires every original hour/symbol')
    rows=sorted(rows,key=lambda r:(r['hour'],r['symbol']))
    stored=protocol._row('algorithm')
    if stored is None or stored[1]is None:raise ValueError('Original algorithm decision missing')
    saved=json.loads(stored[1])
    if fingerprint(saved)!=stored[2]:raise ValueError('Original algorithm decision hash changed')
    inputs=_read(protocol.directory/'algorithm/inputs.json')
    if inputs['pairs']!=rows or inputs['endpoint']!=endpoint.to_dict():
        raise ValueError('Original scoring inputs differ from recomputed nominal raw/model pairs')
    hours=int((pd.Timestamp(spec.end)-pd.Timestamp(spec.start))/pd.Timedelta(hours=1))
    producer.check()
    score=paired_effect_interval(pd.DatetimeIndex([r['hour']for r in rows]),
        [r['reference_loss']for r in rows],[r['candidate_loss']for r in rows],
        symbols=[r['symbol']for r in rows],expected_symbols=lc.plan.symbols,
        eligible_count=hours,rule=endpoint.rule,calibration_passed=False)
    if any(saved['result'].get(k)!=v for k,v in score.items()):
        raise ValueError('Original algorithm statistic/RNG differs from nominal full-path computation')
    decision=verify_decision(lc,saved['result'])
    if type(decision['accepted'])is not bool:raise ValueError('Original algorithm accepted is not boolean')
    producer.check()
    return {'accepted':decision['accepted'],'score_hash':fingerprint(saved),
        'statistic_hash':fingerprint(score),'pairs_hash':fingerprint(rows),
        'nominal_hours':hours,'nominal_symbol_hours':len(expected),
        'complete_actual_support':True,'provider_IID_certified':False}
