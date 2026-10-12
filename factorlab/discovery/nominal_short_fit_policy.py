"""Prospective per-class shorter convex fit ceilings; never extra work credit.

Only the new complete nominal producer can issue these leases. Old routes keep
their exact 300-second path. Neural issuers, model recipes and fit opportunities
are untouched. Small ceilings are assumptions to be tested, not measured WCET.
"""
from __future__ import annotations
from pathlib import Path
import json
import os
import threading

VERSION='nominal-prospective-convex-fit-ceilings-v1'
RELATIVE='factorlab/discovery/nominal_short_fit_policy.py'


def _seconds(value):
    if type(value) is not int or not 1<=value<=300:
        raise ValueError('Prospective exact integer ceiling in 1..300 required')
    return value


def _ceilings(known,pool):
    return {'known':_seconds(known),'pool':_seconds(pool),'neural':300}


def _installed():
    from importlib import import_module
    from factorlab.protocol import project_path
    from .monthly_work_authority import source_files
    from .evidence import file_hash
    files=source_files()
    for name in ('nominal_short_fit_policy','nominal_algorithm_producer',
                 'nominal_algorithm_policy','nominal_algorithm_total_risk',
                 'known_linear_forward','known_linear_reference','pool_forward','forward_preparation',
                 'nominal_cold_conformance',
                 'monthly_partition_policy','monthly_work_authority','work_lease'):
        module=import_module('factorlab.discovery.'+name)
        relative='factorlab/discovery/'+name+'.py'
        if (Path(module.__file__).resolve()!=project_path(relative).resolve()
                or files.get(relative)!=file_hash(module.__file__)):
            raise ValueError('Shorter lease source dependency is not canonical: '+name)
    return files


def _policy_paths(design):
    return (design.directory/'short_fit_policy/identity.json',
            design.parent.directory/'nominal_short_fit_policy.json')


def _expected(design,ceilings,files):
    from .monthly_partition_policy import frozen_policy
    if ceilings!=_ceilings(ceilings['known'],ceilings['pool']):
        raise ValueError('Neural ceiling or convex class policy changed')
    if design.parent.plan.runtime_scope['work_budget']['partition_policy']!=frozen_policy('standard'):
        raise ValueError('This prospective policy requires the exact original standard partitions')
    return {'version':VERSION,'design_digest':design.digest,'source_files':files,
        'ceilings':ceilings,'scope':'exact_new_nominal_complete_producer_only',
        'partition_policy':frozen_policy('standard'),
        'preparation_fit_save_reload_publication':'same_single_original_lease',
        'parameters_seed_solve_limits':'unchanged_original_complete_recipe',
        'opportunities':'unchanged_original_max_one_model_per_step',
        'refund':False,'retry_failed_work':False,'neural_changed':False,
        'warmboot_authorized':False,'performance_verified':False,'formal_acceptance':False}


class NominalShortFitPolicy:
    @classmethod
    def freeze(cls,design,*,known_seconds,pool_seconds):
        from .nominal_algorithm_total_risk import AlgorithmTotalRiskRegistry,_fresh_parent
        from .nominal_cold_conformance import _write
        from filelock import FileLock
        if type(design) is not AlgorithmTotalRiskRegistry:
            raise TypeError('Exact prospectively unused complete algorithm design required')
        ceilings=_ceilings(known_seconds,pool_seconds)
        design.verify();_fresh_parent(design.parent);files=_installed()
        paths=_policy_paths(design)
        with FileLock(str(design.directory/'.short-fit-policy-creation.lock'),timeout=0):
            _fresh_parent(design.parent)
            if any(p.exists() for p in paths) or paths[0].parent.exists():
                raise ValueError('Previous or partial policy freeze cannot be replaced')
            paths[0].parent.mkdir()
            value=_expected(design,ceilings,files)
            for p in paths:_write(p,value)
        return cls.inspect(design)

    @classmethod
    def inspect(cls,design):
        from .nominal_algorithm_total_risk import AlgorithmTotalRiskRegistry
        from .nominal_cold_conformance import _read
        from factorlab.protocol import fingerprint
        if type(design) is not AlgorithmTotalRiskRegistry:
            raise TypeError('Exact complete algorithm design required')
        design.verify();files=_installed()
        paths=_policy_paths(design);value=_read(paths[0])
        if value!=_expected(design,value['ceilings'],files) or _read(paths[1])!=value:
            raise ValueError('Prospective ceiling, source or policy mirror changed')
        obj=cls.__new__(cls);obj.design=design;obj.identity=value;obj.digest=fingerprint(value)
        return obj

    def binding(self):
        return {'directory':str(_policy_paths(self.design)[0].parent),'digest':self.digest,
                'ceilings':self.identity['ceilings'],'version':VERSION}


def policy_binding_for_design(design):
    paths=_policy_paths(design)
    if not any(p.exists() for p in paths) and not paths[0].parent.exists():return None
    return NominalShortFitPolicy.inspect(design).binding()


def _class(recipe):
    from .known_linear_reference import FrozenLinearRecipe
    from .pool_forward import PoolForwardRecipe
    if type(recipe) is FrozenLinearRecipe:return 'known'
    if type(recipe) is PoolForwardRecipe:return 'pool'
    raise TypeError('Only exact original complete known/pool recipes can use shorter leases')


def _allocation_paths(runtime_directory,job):
    from .calibration_trial_context import _no_redirect
    from factorlab.protocol import project_path
    root=project_path(runtime_directory)
    if type(job) is not str or len(job)!=64 or any(c not in '0123456789abcdef' for c in job):
        raise ValueError('Exact original job hash required')
    # Exact existing whole-trial layout; original permit/runtime verification
    # remains mandatory before admission. No arbitrary registry path is accepted.
    trial=root.parent;registry=trial.parent.parent
    paths=(trial/'nominal_short_fits'/job,
           registry/'nominal_short_fits'/trial.name/job)
    for p in paths:_no_redirect(p,registry)
    return paths


def _header_binding(value):
    if (type(value) is not dict or set(value)!={'directory','digest','ceilings','version'}
            or value['version']!=VERSION or type(value['digest']) is not str
            or len(value['digest'])!=64 or any(c not in '0123456789abcdef' for c in value['digest'])
            or type(value['directory']) is not str
            or value['ceilings']!=_ceilings(value['ceilings']['known'],value['ceilings']['pool'])):
        raise ValueError('Invalid already-frozen private producer policy header')
    return value


def _stored_producer(runtime_directory):
    path=Path(runtime_directory).parent/'nominal_algorithm_producer/identity.json'
    if not path.exists():
        root=Path(runtime_directory)
        if (root.parent.parent.parent/'nominal_short_fit_policy.json').exists():
            # A new whole-path mode cannot erase its producer and silently use
            # 300 seconds. Existing fixed-estimand replicas retain their route.
            enrollment=json.loads((root.parent/'enrollment.json').read_text('utf-8'))
            if enrollment['intent'].get('estimand_slot') is None:
                raise ValueError('New short-policy whole trial lacks original producer identity')
        return None
    if not path.is_file() or path.stat().st_size>1024*1024:
        raise ValueError('Invalid original producer identity file')
    # Original _write wraps records, and this bounded read only chooses refusal
    # or the unchanged 300 route; it never grants a shortened lease.
    from .nominal_cold_conformance import _read
    value=_read(path)
    return value.get('short_fit_policy')


def issue_registered_lease(runtime,job,recipe):
    """Called only by the unchanged original known/pool ticket issuers."""
    from .nominal_algorithm_producer import AlgorithmNominalProducer,_ACTIVE
    from .v7_forward import ForwardRuntime
    producer=_ACTIVE.get()
    if producer is None:
        if _stored_producer(runtime.directory) is not None:
            raise ValueError('Short-policy original runtime lacks its private paid producer invocation')
        return None
    if (type(producer) is not AlgorithmNominalProducer or type(runtime) is not ForwardRuntime
            or producer.owner!=(os.getpid(),threading.get_ident())
            or producer.permit.paths['runtime']!=runtime.directory):
        raise ValueError('Foreign producer/runtime cannot select a fit ceiling')
    producer.check()
    binding=producer.identity['short_fit_policy']
    if binding is None:return None
    _header_binding(binding)
    kind=_class(recipe);seconds=binding['ceilings'][kind]
    paths=_allocation_paths(runtime.directory,job)
    if any(p.exists() for p in paths):
        raise ValueError('Original shorter fit already attempted; no replacement lease')
    from .work_lease import FrozenWorkLease
    lease=FrozenWorkLease.issue(seconds)  # Heavy source/registry/recipe checks below are timed.
    from .nominal_algorithm_execution import _exclusive,_write_pair
    _exclusive(paths)
    intent={'version':VERSION,'policy':binding,'job':job,'kind':kind,
            'runtime_directory':str(runtime.directory),'recipe':recipe.to_dict(),
            'lease':lease.to_dict(),'trial':producer.permit.trial}
    _write_pair(paths,'intent.json',intent)
    try:
        lease.assert_parent_alive();producer.check();_installed()
        from .nominal_algorithm_total_risk import AlgorithmTotalRiskRegistry
        design=AlgorithmTotalRiskRegistry.inspect(producer.permit.registry)
        if policy_binding_for_design(design)!=binding:
            raise ValueError('Private producer ceiling differs from original policy')
        # Existing issuer _context, preparation and authority reservation still
        # perform their original checks and use this exact already-running lease.
        lease.assert_parent_alive();producer.check()
        return lease
    except BaseException as error:
        for p in paths:
            try:_write_pair((p,),'failure.json',{'state':'failed_or_unknown_consumed','error_type':type(error).__name__})
            except BaseException:pass
        raise


def verify_registered_lease(ticket,recipe,record):
    """Additional historical check; no lease issuance, wallet reservation or fit."""
    from .nominal_algorithm_execution import _pair
    from .work_lease import FrozenWorkLease
    v=ticket.value;runtime=Path(v['context']['lifecycle_directory']);job=v['context']['job']
    stored=_stored_producer(runtime)
    paths=_allocation_paths(runtime,job)
    if stored is None:
        if any(p.exists() for p in paths):
            raise ValueError('Orphan short-fit allocation cannot use the ordinary route')
        return None  # No old mode's existing 300 verification is relaxed.
    binding=_header_binding(stored)
    _installed()
    from .full_calibration_registry import FullTrialRegistry
    from .nominal_algorithm_total_risk import AlgorithmTotalRiskRegistry
    registry=FullTrialRegistry.inspect(runtime.parent.parent.parent)
    design=AlgorithmTotalRiskRegistry.inspect(registry)
    if policy_binding_for_design(design)!=binding:
        raise ValueError('Original historical short-fit policy changed')
    from .nominal_cold_conformance import _read
    producer_roots=(runtime.parent/'nominal_algorithm_producer',
        registry.directory/'nominal_algorithm_producers')
    left=_read(producer_roots[0]/'identity.json')
    trial=left['trial'];expected_trial=registry.identity_for(trial['cell'],trial['index'],
        oracle_index=trial['oracle_index'],estimand_slot=trial.get('estimand_slot'))
    expected_paths=registry.paths(trial['cell'],trial['index'],oracle_index=trial['oracle_index'],
                                 estimand_slot=trial.get('estimand_slot'))
    if (trial!=expected_trial or expected_paths['runtime']!=runtime
            or trial.get('estimand_slot') is not None
            or _read(producer_roots[1]/trial['trial_id']/'identity.json')!=left
            or left['short_fit_policy']!=binding or left['source_files']!=_installed()):
        raise ValueError('Original producer/trial/runtime identity changed')
    if any((p/'failure.json').exists() for p in paths):
        raise ValueError('Failed original shortened allocation remains consumed')
    lease=FrozenWorkLease.from_dict(record['lease'])
    kind=_class(recipe)
    intent={'version':VERSION,'policy':binding,'job':job,'kind':kind,
            'runtime_directory':str(runtime),'recipe':recipe.to_dict(),
            'lease':lease.to_dict(),'trial':trial}
    if (lease.lease_seconds!=binding['ceilings'][kind]
            or record['lease']!=v['lease'] or _pair(paths,'intent.json')!=intent):
        raise ValueError('Historical model did not use its prospectively frozen original ceiling')
    return binding
