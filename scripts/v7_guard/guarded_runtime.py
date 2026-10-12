"""STAGE fixed producer adapters; callable model/provider results are never proofs.

Only use inside multiwork_enforcer's dedicated original issuer. Patching is
process-local, temporary and limited to the actual fixed scheduler producer.
Core integration should replace this patch with an explicit trusted hook.
"""
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from factorlab import protocol
from factorlab.discovery.rolling import RollingSpec
from factorlab.discovery.s_outer_context import SOuterContext
from factorlab.discovery.s_preparation import SPreparationEnvelope
from factorlab.discovery.scheduled_search import FrozenSWorkSchedule, ScheduledSearch
from factorlab.discovery.v7_run import V7Config, run_month


@contextmanager
def _scheduled_publication(scope, authority):
    original=ScheduledSearch.run
    def run(self,*args,**kwargs):
        if self.bridge.authority is not authority:
            raise ValueError('Scheduler cannot switch the original shared wallet')
        result=original(self,*args,**kwargs)
        rid=result.get('reservation_id')
        if rid is not None:
            scope.finish_work(rid,quota_consumed=result['status']=='quota_consumed')
        elif scope.current is not None:
            raise ValueError('Unfunded schedule result cannot hide active prepaid work')
        return result
    ScheduledSearch.run=run
    try:yield
    finally:ScheduledSearch.run=original


def execute_s_file(scope, path, *, authority, schedule, backend=None):
    """Actual original preparation→serialization/read→fixed monthly stages."""
    if scope.scope['role']!='S' or type(schedule) is not FrozenSWorkSchedule:
        raise ValueError('Actual S outer scope and fixed schedule required')
    bound=scope.scope['reference']['bound']
    value=SOuterContext.verify_bound(bound)
    outer=object.__new__(SOuterContext);outer.reference=bound['reference']
    config=V7Config(**{**value['config'],'symbols':tuple(value['config']['symbols'])})
    spec=RollingSpec(**{**value['spec'],'event_bins':tuple(value['spec']['event_bins'])})
    if schedule.digest!=value['schedule_hash']:
        raise ValueError('Cannot replace original S schedule')
    source=value['source_reference'];cutoff=source['cutoff'];directory=value['directory']
    with scope.installed(authority),_scheduled_publication(scope,authority):
        prep=SPreparationEnvelope.prepare(authority,directory,cutoff[:7],config=asdict(config),spec=asdict(spec),
            source_reference=source,schedule_hash=schedule.digest,outer_context=outer)
        bars=pd.read_parquet(protocol.project_path(path))
        summary=run_month(directory,bars,cutoff,config=config,spec=spec,backend=backend,
            feed_directory=source['feed_directory'],exposure_path=source['exposure_path'],
            authority=authority,schedule=schedule,preparation=prep)
        if scope.current is not None:
            raise ValueError('Monthly producer left original work unfinished')
    return summary


def execute_forward_month(scope,runtime,cutoff,*,authority,stream='A0',recipe=None):
    """Actual pool/known/neural dispatch; completed cache does not reserve again."""
    if scope.scope['role']!='forward' or Path(scope.scope['directory'])!=runtime.directory:
        raise ValueError('Only this actual Campaign runtime can dispatch forward work')
    with scope.installed(authority):
        result=runtime.fit_month(cutoff,stream=stream,pool=recipe,authority=authority)
        if scope.current is not None:
            scope.finish_work(scope.current['reservation_id'])
    return result

@contextmanager
def forward_step(scope, service):
    """Wrap real monthly producers; polling-only ticks create no work grant."""
    from factorlab.discovery import algorithm_validation_producer as algorithm
    from factorlab.discovery.v7_feed import SnapshotUnavailable
    authority=service.work_authority
    if scope.scope['role']!='forward' or authority is None or service.runtime.directory!=Path(scope.scope['directory']):
        raise ValueError('Actual Campaign service/shared wallet required')
    original_fit=service.runtime.fit_month
    original_algorithm=algorithm.launch
    from factorlab.discovery import structural_queue_review
    original_queue=structural_queue_review.produce_typed_queue_review
    def queue(actual_runtime,actual_authority,*args,**kwargs):
        if actual_runtime is not service.runtime or actual_authority is not authority:
            raise ValueError('Typed queue changed actual runtime or wallet')
        result=original_queue(actual_runtime,actual_authority,*args,**kwargs)
        if scope.current is not None:
            rid=result.get('queue_review_reservation')
            if rid!=scope.current['reservation_id']:
                raise ValueError('Queue publication lost original guarded reservation')
            scope.finish_work(rid)
        return result
    structural_queue_review.produce_typed_queue_review=queue
    def fitted(*args,**kwargs):
        if kwargs.get('authority') is not authority:raise ValueError('Forward dispatch changed actual wallet')
        try:result=original_fit(*args,**kwargs)
        except SnapshotUnavailable:
            if scope.current is not None:scope.finish_work(scope.current['reservation_id'],closure='unavailable_snapshot')
            raise
        if scope.current is not None:scope.finish_work(scope.current['reservation_id'])
        return result
    def launched(actual_protocol,*args,**kwargs):
        if actual_protocol.authority is not authority:raise ValueError('Typed validation changed actual wallet')
        result=original_algorithm(actual_protocol,*args,**kwargs)
        if scope.current is not None:scope.finish_work(scope.current['reservation_id'])
        return result
    service.runtime.fit_month=fitted;algorithm.launch=launched
    try:
        with scope.installed(authority):yield
    finally:
        service.runtime.fit_month=original_fit;algorithm.launch=original_algorithm
        structural_queue_review.produce_typed_queue_review=original_queue


@contextmanager
def nonlinear_publication(scope, authority):
    """Fixed typed producer hooks, including consumed expected gate failures.

    Numerical launcher return alone does not release original supervision. Each
    work's actual catalogue/epoch registration must return first. The final
    month-summary write remains inside the same immutable original outer clock.
    """
    from factorlab.discovery import s_nonlinear_controller as controller
    from factorlab.discovery.algorithm_shortlist import AlgorithmCatalogue
    from factorlab.discovery.s_nonlinear_gates import NonlinearGateClosed
    originals={name:getattr(controller,name) for name in ('_register','issue_nonlinear_ticket','issue_research_ticket')}
    original_seal=AlgorithmCatalogue.seal_shortlist
    from factorlab.discovery import structural_evolution
    original_review=structural_evolution.produce_structure_review
    def reviewed(bridge,*args,**kwargs):
        if bridge.authority is not authority:
            raise ValueError('Structure review cannot switch the original shared wallet')
        result=original_review(bridge,*args,**kwargs)
        rid=result['body']['review_reservation']
        if rid is not None and scope.current is not None:
            if scope.current['reservation_id']!=rid:
                raise ValueError('Structure review lost original guarded work')
            scope.finish_work(rid)
        return result
    structural_evolution.produce_structure_review=reviewed
    def released():
        if scope.current is not None:scope.finish_work(scope.current['reservation_id'])
    def registered(*args,**kwargs):
        result=originals['_register'](*args,**kwargs);released();return result
    def sealed(self,*args,**kwargs):
        result=original_seal(self,*args,**kwargs)
        if scope.current is not None:
            current=authority.get(scope.current['reservation_id'])
            if current['status']=='completed':released()
        return result
    def issue(name):
        def call(*args,**kwargs):
            try:return originals[name](*args,**kwargs)
            except NonlinearGateClosed as error:
                rid=getattr(error,'reservation_id',None)
                if scope.current is None or rid!=scope.current['reservation_id']:
                    raise ValueError('Gate failure lost its original consumed work') from error
                scope.finish_work(rid,closure='closed_nonlinear_gate')
                raise
        return call
    controller._register=registered
    controller.issue_nonlinear_ticket=issue('issue_nonlinear_ticket')
    controller.issue_research_ticket=issue('issue_research_ticket')
    AlgorithmCatalogue.seal_shortlist=sealed
    try:yield
    finally:
        for name,value in originals.items():setattr(controller,name,value)
        AlgorithmCatalogue.seal_shortlist=original_seal
        structural_evolution.produce_structure_review=original_review
