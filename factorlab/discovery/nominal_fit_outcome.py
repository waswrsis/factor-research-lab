"""Narrow live numerical failure provenance for new nominal fixed recipes.

Only an exact LinAlgError escaping an actual, finite, source-bound lstsq call is
classified. Persisted records are provisional: they are not model admissions or
historical loss certificates and cannot restore these live objects.
"""
from __future__ import annotations

from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import weakref

import numpy as np
import pandas as pd

from factorlab.protocol import fingerprint, project_path
from .monthly_work_authority import MonthlyWorkAuthority, source_files
from .reserved_work_budget import ReservedChannelBudget
from .rolling import build_rolling_data
from .known_linear_reference import FrozenLinearRecipe, fit_known_linear
from .joint_pool import fit_pool
from .nominal_cold_conformance import _read, _write
from .evidence import file_hash

RELATIVE='factorlab/discovery/nominal_fit_outcome.py'
VERSION='nominal-live-numerical-fit-outcome-v1'
MAX_ROWS=8192
MAX_COLS=64
MAX_RAW_ROWS=32768
MAX_KERNEL_CALLS=16384
_KEY=object()
_ACTIVE=ContextVar('original_nominal_numerical_fit',default=None)
_LIVE=weakref.WeakSet()
_LSTSQ=np.linalg.lstsq
_LINALG_ERROR=np.linalg.LinAlgError


class NumericalEvidenceUnavailable(RuntimeError):
    """Infrastructure evidence failed; never a nominal numeric failure."""


def _owner():return os.getpid(),threading.get_ident()


def _installed():
    from . import convex_eval, nominal_factor_census
    files=source_files()
    for module,relative in ((sys.modules[__name__],RELATIVE),
            (convex_eval,'factorlab/discovery/convex_eval.py'),
            (nominal_factor_census,'factorlab/discovery/nominal_factor_census.py')):
        if Path(module.__file__).resolve()!=project_path(relative).resolve() or files.get(relative)!=file_hash(module.__file__):
            raise NumericalEvidenceUnavailable('Canonical numerical kernel/source binding differs')
    if np.linalg.lstsq is not _LSTSQ or np.linalg.LinAlgError is not _LINALG_ERROR:
        raise NumericalEvidenceUnavailable('Original NumPy function/type identity changed')
    return files


def _array(value,dimension):
    if (type(value) is not np.ndarray or value.dtype!=np.dtype('float64')
            or value.ndim!=dimension or not 1<=len(value)<=MAX_ROWS
            or (dimension==2 and not 1<=value.shape[1]<=MAX_COLS)
            or not np.isfinite(value).all()):
        raise NumericalEvidenceUnavailable('Finite actual binary64 kernel input required')
    return {'shape':list(value.shape),'dtype':'binary64',
            'sha256':hashlib.sha256(value.tobytes(order='C')).hexdigest()}


def _kernel(site,a,b,rcond):
    scope=_ACTIVE.get()
    if scope is None:
        from .nominal_algorithm_fit import kernel_scope
        scope=kernel_scope()
    if scope is None:
        # Preserve the original callable, result and exception outside this new
        # nominal scope; no reinterpretation of old callers or profile search.
        return np.linalg.lstsq(a,b,rcond=rcond)
    from . import convex_eval
    expected=(convex_eval.fit_ridge.__code__ if site=='ridge_augmented'
              else convex_eval.ResidualProjector.fit.__func__.__code__)
    if sys._getframe(2).f_code is not expected:
        raise NumericalEvidenceUnavailable('Only original fixed kernel call site may emit an event')
    scope.check()
    if np.linalg.lstsq is not _LSTSQ or np.linalg.LinAlgError is not _LINALG_ERROR:
        raise NumericalEvidenceUnavailable('Original NumPy callable changed')
    inputs={'matrix':_array(a,2),'response':_array(b,1),
            'rcond':None if rcond is None else float(rcond).hex()}
    if len(a)!=len(b):raise NumericalEvidenceUnavailable('Kernel matrix/response shape mismatch')
    scope.calls+=1
    if scope.calls>MAX_KERNEL_CALLS:raise NumericalEvidenceUnavailable('Fixed kernel event bound exceeded')
    try:
        return _LSTSQ(a,b,rcond=rcond)
    except _LINALG_ERROR as error:
        if type(error) is not _LINALG_ERROR:
            raise
        try:
            scope.check()
            if inputs!={'matrix':_array(a,2),'response':_array(b,1),
                        'rcond':None if rcond is None else float(rcond).hex()}:
                raise NumericalEvidenceUnavailable('Failed kernel input changed during call')
            scope.capture(error,site,inputs)
        except Exception as evidence_error:
            raise NumericalEvidenceUnavailable('Numerical failure evidence could not be retained') from evidence_error
        # Original profile_parameters may catch this exact exception and recover.
        # Never replace it with an early-abort exception inside the algorithm.
        raise


def ridge_lstsq(a,b):return _kernel('ridge_augmented',a,b,None)
def residual_lstsq(a,b,rcond):return _kernel('residual_projection',a,b,rcond)


class NominalNumericalFailure:
    def __init__(self,key,preparation,role,event,error):
        if key is not _KEY or type(self) is not NominalNumericalFailure:raise TypeError('Original kernel emission only')
        self._preparation=preparation;self._role=role;self._event=event;self._error=error
        self._digest=fingerprint(event);self._owner=_owner();_LIVE.add(self)
    def __reduce__(self):raise TypeError('Live numerical outcomes cannot serialize/restore')
    def evidence(self):
        if self not in _LIVE or self._owner!=_owner() or fingerprint(self._event)!=self._digest:
            raise NumericalEvidenceUnavailable('Original live numerical outcome identity lost')
        self._preparation.check()
        for root in self._preparation.roots:
            if _read(root/self._role/'numeric-failure.json')!=self._event:
                raise NumericalEvidenceUnavailable('Original numerical outcome mirrors differ')
        return json.loads(json.dumps(self._event))


class _KernelScope:
    def __init__(self,key,preparation,role):
        if key is not _KEY:raise TypeError('Private original fit scope only')
        self.preparation=preparation;self.role=role;self.owner=_owner()
        self.calls=0;self.last=None;self.active=True
    def check(self):
        if not self.active or _ACTIVE.get() is not self or self.owner!=_owner():
            raise NumericalEvidenceUnavailable('Original active kernel scope lost')
        self.preparation.check(full_source=False)
    def capture(self,error,site,inputs):
        p=self.preparation
        event={'version':VERSION,'status':'executed_linalg_error_provisional',
            'preparation_hash':p.digest,'role':self.role,'kernel_site':site,
            'kernel_ordinal':self.calls,'inputs':inputs,'exception_type':'numpy.linalg.LinAlgError',
            'exception_message':str(error)[:1024],'budget':p.budget.to_dict(),
            'source_files':p.files,'training':p.data.training_identity(p.cutoff),
            'original_lease':p.budget._lease.to_dict(),'raw_hash':p.raw_hash,
            'numpy_version':np.__version__,'historical_truth_certificate':False}
        # One failed invocation need not mean failed algorithm: profile may handle
        # it and succeed. Keep each event, classify only the escaping same object.
        for root in p.roots:_write(root/self.role/f'kernel-{self.calls:06d}.json',event)
        p.budget.check_time()
        self.last=(error,event)


class PreparedNominalFit:
    def __init__(self,*args,**kwargs):raise TypeError('Use original protocol preparation')
    def __reduce__(self):raise TypeError('Prepared paid execution cannot serialize/restore')
    def check(self,full_source=True):
        if self._owner!=_owner() or fingerprint(self._binding)!=self.digest:
            raise NumericalEvidenceUnavailable('Original fit preparation owner/binding changed')
        self.budget.check_time()
        if any((root.parent/'failure.json').exists() for root in self.roots):
            raise NumericalEvidenceUnavailable('Infrastructure failure dominates numerical evidence')
        if full_source and _installed()!=self.files:raise NumericalEvidenceUnavailable('Original fit source changed')
        if (self.budget._binding_hash!=self._binding['budget_binding_hash']
                or self.budget._lease.digest!=self._binding['lease_hash']
                or _read(self.raw_path)!=self._raw):
            raise NumericalEvidenceUnavailable('Original budget/raw input changed')
        for root in self.roots:
            if _read(root/'preparation.json')!=self._binding:
                raise NumericalEvidenceUnavailable('Original preparation anchor changed')
    def fit_role(self,role):
        from .nominal_factor_census import _recipe
        if type(role) is not str or role not in ('reference','candidate'):
            raise ValueError('Original fixed role required')
        if _ACTIVE.get() is not None:raise NumericalEvidenceUnavailable('Nested numerical scopes forbidden')
        self.check()
        recipe,_=_recipe(self.protocol_identity['recipes'][role],self.data.spec)
        # Do not permit a changed mutable dataframe to acquire original raw lineage.
        if self.data.training_identity(self.cutoff)!=self.training:
            raise NumericalEvidenceUnavailable('Original prepared training data changed')
        for root in self.roots:
            (root/role).mkdir(exist_ok=False)
            _write(root/role/'intent.json',{'preparation_hash':self.digest,'role':role,
                'recipe':recipe.to_dict(),'budget_before':self.budget.to_dict()})
        scope=_KernelScope(_KEY,self,role);token=_ACTIVE.set(scope)
        try:
            try:
                model=(fit_known_linear(self.data,self.cutoff,recipe,budget=self.budget)
                       if type(recipe) is FrozenLinearRecipe else fit_pool(self.data,self.cutoff,recipe,budget=self.budget))
            except _LINALG_ERROR as error:
                if type(error) is not _LINALG_ERROR or scope.last is None or scope.last[0] is not error:
                    raise
                event=scope.last[1]
                self.check()
                for root in self.roots:_write(root/role/'numeric-failure.json',event)
                self.budget.check_time()
                return NominalNumericalFailure(_KEY,self,role,event,error)
            self.check()
            return model
        finally:
            scope.active=False;_ACTIVE.reset(token)


def prepare_nominal_fit(protocol,*,authority,budget,atom,cutoff):
    from .nominal_factor_census import NominalFactorProtocol,validate_nominal_factor_route,_rolling
    if (type(protocol) is not NominalFactorProtocol or type(authority) is not MonthlyWorkAuthority
            or type(budget) is not ReservedChannelBudget or budget.test_only or budget.executor_mode!='issuer'):
        raise TypeError('Exact original fixed nominal protocol and issuer budget required')
    budget.check_time();files=_installed();p=protocol.verify();plan=protocol.pair_plan(atom,cutoff)
    folder=protocol.directory/'fits'/fingerprint(plan)
    intent=_read(folder/'intent.json');mirror=authority.directory/'nominal_factor_intents'/fingerprint(intent['work_key'])
    record=authority.get(budget._binding['reservation_id'])
    if (record['status']!='running' or record['binding']!=intent['binding'] or record['lease']!=intent['lease']
            or _read(mirror/'intent.json')!=intent or p['source_files']!=files
            or authority.manifest['source_files']!=files or p['authority_manifest_hash']!=authority.manifest_hash
            or record['lease_hash']!=budget._lease.digest or budget._lease.lease_seconds!=300
            or record['process_identity']['pid']!=os.getpid() or record['counts']!=p['counts_per_pair']):
        raise NumericalEvidenceUnavailable('Original source/pair/paid attempt identity differs')
    validate_nominal_factor_route(authority,intent['work_key'],intent['binding'],budget._lease,record['counts'])
    roots=(folder/'numerical',mirror/'numerical')
    for root in roots:root.mkdir(exist_ok=False)
    raw_path=folder/'raw.json';raw=_read(raw_path)
    if type(raw.get('rows')) is not list or not 1<=len(raw['rows'])<=MAX_RAW_ROWS:
        raise NumericalEvidenceUnavailable('Original raw input bound exceeded')
    binding={'version':VERSION,'protocol_digest':protocol.digest,'plan':plan,'source_files':files,
        'raw_hash':fingerprint(raw),'reservation_id':record['reservation_id'],
        'budget_binding_hash':budget._binding_hash,'lease_hash':budget._lease.digest,
        'authority_manifest_hash':authority.manifest_hash,'historical_truth_certificate':False}
    for root in roots:_write(root/'preparation.json',binding)
    bars=pd.DataFrame(raw['rows']);bars['open_time']=pd.to_datetime(bars.open_time,utc=True)
    bars['available_at']=bars.open_time+pd.Timedelta(hours=1,seconds=5)
    bars.attrs['availability_mode']='assumed_historical_delay'
    budget.check_time();data=build_rolling_data(bars,_rolling(p['spec']));budget.check_time()
    result=object.__new__(PreparedNominalFit)
    result._owner=_owner();result.files=files;result.protocol_identity=p;result.cutoff=cutoff
    result.roots=roots;result.raw_path=raw_path;result._raw=raw;result.raw_hash=fingerprint(raw)
    result._binding=binding;result.digest=fingerprint(binding);result.budget=budget
    result.data=data;result.training=data.training_identity(cutoff)
    result.check();return result


def _numeric_rows(preparation,role):
    """Actual already-built data and fixed calendar, no model or solver."""
    from .nominal_factor_census import selected_cutoff,stamp,_loss_module
    data=preparation.data;p=preparation.protocol_identity;cutoff=preparation.cutoff
    schedule=p['schedule'];loss=_loss_module();metadata={}
    for i,meta in data.frame.metadata.iterrows():
        origin=stamp(meta.available_at)
        if not stamp(schedule['start'])<=origin<stamp(schedule['end']):continue
        if selected_cutoff(schedule,role,origin)!=cutoff:continue
        key=(origin.isoformat(),meta.symbol)
        if key in metadata:raise NumericalEvidenceUnavailable('Duplicate original nominal row')
        label=float(data.target[i]) if data.observed[i] else None
        if label is not None and not __import__('math').isfinite(label):label=None
        metadata[key]={'origin':origin.isoformat(),'hour':origin.floor('h').isoformat(),'symbol':meta.symbol,
            'feature_open_time':stamp(meta.feature_open_time).isoformat(),'window':stamp(meta.forecast_window_start).isoformat(),
            'cutoff':cutoff,'train_start':data.spec.window(cutoff)[0].isoformat(),
            'prediction':None,'label':label,'maturity':stamp(data.maturity[i]).isoformat(),'loss':loss.nominal_total_loss(label,None)}
    expected=[(t.isoformat(),symbol)for t in pd.date_range(schedule['start'],schedule['end'],freq='h',inclusive='left')
        +pd.Timedelta(seconds=schedule['delay'])for symbol in p['law']['symbols']if selected_cutoff(schedule,role,t)==cutoff]
    if not set(metadata)<=set(expected):raise NumericalEvidenceUnavailable('Foreign nominal failed-fit row')
    for origin,symbol in expected:
        if (origin,symbol)in metadata:continue
        t=stamp(origin);window=t.ceil('h')
        metadata[(origin,symbol)]={'origin':origin,'hour':t.floor('h').isoformat(),'symbol':symbol,
            'feature_open_time':(t.floor('h')-pd.Timedelta(hours=1)).isoformat(),'window':window.isoformat(),
            'cutoff':cutoff,'train_start':data.spec.window(cutoff)[0].isoformat(),
            'prediction':None,'label':None,'maturity':(window+pd.Timedelta(hours=12)).isoformat(),'loss':loss.nominal_total_loss(None,None)}
    return [metadata[k]for k in sorted(expected)]


def _finish_failed_pair(preparation,outcome,authority,outputs,kernel_before):
    """Fixed producer tail; failed accounting is never serving completion.

    This keeps the original abort ordering. Later roles are not evaluated and
    receive no model or invented fit receipt. The returned live object remains
    provisional until sanctioned historical proof/recomputation is integrated.
    """
    from . import nominal_factor_census as producer
    from .nonlinear_launcher import _kernel_snapshot,_validate_kernel
    from .neural_forward_launcher import _identity,_member,_observation
    if (sys._getframe(1).f_code is not producer._produce_month.__code__
            or type(preparation) is not PreparedNominalFit or type(outcome) is not NominalNumericalFailure
            or outcome._preparation is not preparation or type(authority) is not MonthlyWorkAuthority):
        raise NumericalEvidenceUnavailable('Only the original paid producer may finalize a numerical pair')
    event=outcome.evidence();preparation.check()
    budget=preparation.budget;lease=budget._lease
    reservation=authority.get(budget._binding['reservation_id'])
    if reservation['status']!='running' or authority.manifest_hash!=preparation._binding['authority_manifest_hash']:
        raise NumericalEvidenceUnavailable('Original running reservation required')
    budget.finalize();actual=ReservedChannelBudget.verify_counts(authority,reservation['reservation_id'])
    if (actual['test_only'] or not actual['finalized'] or actual['actual_counts']!=budget.actual_counts
            or any(actual['actual_counts'][k]>reservation['counts'][k] for k in reservation['counts'])):
        raise NumericalEvidenceUnavailable('Original numerical failure actual counters differ')
    last=_kernel_snapshot();_validate_kernel(kernel_before,last);_member(last,_identity())
    failed=authority.fail_consumed(reservation['reservation_id'],'nominal_numeric_linalg:'+fingerprint(event))
    if failed['status']!='failed_consumed':
        raise NumericalEvidenceUnavailable('Late/interrupted attempt is not a numerical outcome')
    nominal_roles=dict(outputs)
    for role in ('reference','candidate'):
        if role in outputs:continue
        nominal_roles[role]={'version':VERSION,'plan':producer.NominalFactorProtocol(preparation.protocol_identity['directory']).plan(
            preparation.protocol_identity['atom'],preparation.cutoff,role),'raw_hash':preparation.raw_hash,
            'model_digest':None,'numerical_model':None,'training':preparation.training,
            'rows':_numeric_rows(preparation,role),'solve_calls':None,
            'numeric_status':'executed_linalg_error'if role==outcome._role else'not_computed_after_original_abort',
            'formal_acceptance':False}
    lease.assert_parent_alive()
    terminal={'version':VERSION,'status':'failed_consumed_numeric_provisional',
        'reservation':failed,'actual_accounting':actual,'preparation_hash':preparation.digest,
        'numeric_event_hash':fingerprint(event),'failed_role':outcome._role,
        'nominal_roles':nominal_roles,'completed_roles':outputs,'later_roles':'not_computed_after_original_abort',
        'kernel_before':kernel_before,'kernel_after':last,'observation':_observation(lease),
        'raw_hash':preparation.raw_hash,'historical_truth_certificate':False,
        'serving_completion':False,'formal_acceptance':False}
    for root in preparation.roots:_write(root/'numerical-terminal.json',terminal)
    final=_kernel_snapshot();_validate_kernel(last,final);_member(final,_identity())
    publication={'terminal_hash':fingerprint(terminal),'kernel':final,'observation':_observation(lease),
                 'historical_truth_certificate':False,'serving_completion':False}
    for root in preparation.roots:_write(root/'numerical-publication.json',publication)
    for root in preparation.roots:
        if _read(root/'numerical-terminal.json')!=terminal or _read(root/'numerical-publication.json')!=publication:
            raise NumericalEvidenceUnavailable('Numerical failure publication mirrors differ')
    lease.assert_parent_alive()
    return outcome
