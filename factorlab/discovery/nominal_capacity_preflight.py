"""Readonly necessary capacity analysis, never a wallet or execution permit.

Required cold jobs and ordinary planned jobs are deliberately separate. A planned
job is not evidence of an attempted reservation. Unknown adaptive work stays
unknown. The public API accepts only the original source-bound frozen policy.
"""
from __future__ import annotations
import json
from pathlib import Path

VERSION = 'nominal-prospective-capacity-necessary-v1'
ELAPSED_VERSION = 'nominal-prospective-elapsed-capacity-necessary-v2'
_KEY = object()


def _plain(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _installed():
    from importlib import import_module
    from factorlab.protocol import project_path
    from .monthly_work_authority import source_files
    from .evidence import file_hash
    files = source_files()
    for name in ('nominal_capacity_preflight', 'nominal_algorithm_policy',
                 'nominal_algorithm_warmboot', 'nominal_short_fit_policy',
                 'nominal_algorithm_total_risk', 'nominal_algorithm_producer',
                 'full_trial_controller', 'v7_service', 'v7_forward', 'v7_lifecycle',
                 'v7_autonomous', 'scheduled_search', 's_preparation',
                 'monthly_partition_policy', 'monthly_work_authority',
                 'known_linear_forward', 'pool_forward', 'neural_forward'):
        mod = import_module('factorlab.discovery.' + name)
        relative = 'factorlab/discovery/' + name + '.py'
        if (Path(mod.__file__).resolve() != project_path(relative).resolve()
                or files.get(relative) != file_hash(mod.__file__)):
            raise ValueError('Capacity analysis requires canonical source: ' + name)
    return files


def _recipe_cost(pipeline, rolling, ceilings):
    """Use the original complete parser and recursive prepaid solve formula."""
    from .v7_forward import ForwardRuntime
    from .v7_service import parse_frozen_recipe
    from .known_linear_reference import FrozenLinearRecipe
    from .joint_pool import FrozenPool
    from .pool_forward import PoolForwardRecipe
    from .neural_forward import FrozenNeuralRecipe
    from .monthly_partition_policy import counts
    runtime = ForwardRuntime.__new__(ForwardRuntime)
    runtime.spec = rolling
    recipe = parse_frozen_recipe(runtime, pipeline)
    if type(recipe) is FrozenLinearRecipe:
        return {'class': 'known', 'seconds': ceilings['known'], 'counts': counts(solve=1)}
    if type(recipe) is FrozenPool:
        return {'class': 'pool', 'seconds': ceilings['pool'],
                'counts': counts(solve=PoolForwardRecipe(recipe, rolling).solve_limit)}
    if type(recipe) is FrozenNeuralRecipe:
        v = recipe.budget_limits
        return {'class': 'neural', 'seconds': 300,
                'counts': counts(**{k: v[k] for k in ('solve', 'neural_fit', 'epoch')})}
    raise TypeError('Unknown original executable learner')


def _derive(endpoints, due_at, warm, rolling, ceilings, search, autonomous, service):
    """Unprivileged arithmetic over already authenticated inputs; no authority."""
    from .nominal_algorithm_policy import _utc, _month
    from .monthly_partition_policy import counts, frozen_policy
    from factorlab.protocol import fingerprint
    dates = tuple(_utc(x) for x in due_at)
    if not dates or dates != tuple(sorted(set(dates))):
        raise ValueError('Original ordered opportunities required')
    if not 0 < service['wall_seconds'] <= 7200 or service['max_fits_per_step'] != 1:
        raise ValueError('Original service limits required')
    if rolling.window_days == 90:
        return _derive_elapsed(endpoints, dates, warm, rolling, ceilings, search, autonomous, service)
    months = sorted({_month(d.isoformat()).isoformat() for d in dates})
    jobs = {}
    def add(cutoff, stream, role, pipeline=None):
        key = (cutoff, stream)
        if key in jobs:
            old = jobs[key]
            if old['pipeline'] != pipeline:
                raise ValueError('Same original job key has conflicting recipes')
            if role not in old['roles']: old['roles'].append(role)
            return
        cost = (_recipe_cost(pipeline, rolling, ceilings) if pipeline is not None else
                {'class': 'adaptive_unknown', 'seconds': min(ceilings.values()),
                 'counts': counts(solve=1)})
        jobs[key] = {'cutoff': cutoff, 'stream': stream, 'roles': [role],
                     'pipeline': pipeline, 'recipe_exact': pipeline is not None,
                     'prepaid_lower_bound': cost}
    for stream, pipeline in warm['streams'].items():
        add(warm['cutoff'], stream, 'required_warmboot', pipeline)
    for e in endpoints:
        for offset in range(6):
            cutoff = _month(e['start'], offset).isoformat()
            if cutoff not in months: raise ValueError('Missing endpoint month opportunities')
            add(cutoff, e['reference_stream'], 'required_endpoint', e['reference'])
            add(cutoff, e['candidate_stream'], 'required_endpoint',
                None if e['slot'] == 'algorithm' else e['candidate'])
    # Real _plan_month always plans A0, including warmboot/scoring months. It
    # need not actually attempt/finish every such job before service termination.
    for cutoff in months: add(cutoff, 'A0', 'ordinary_A0_planned')
    rows = [jobs[k] for k in sorted(jobs)]
    def total(selected):
        return {'jobs': len(selected),
                'seconds': sum(r['prepaid_lower_bound']['seconds'] for r in selected),
                'counts': {k: sum(r['prepaid_lower_bound']['counts'][k] for r in selected)
                           for k in counts()}}
    required = [r for r in rows if any(x.startswith('required_') for x in r['roles'])]
    cold, planned = total(required), total(rows)
    lane = frozen_policy('standard')['lanes']['forward']
    # Even the most favourable <=2h physical run can intersect at most two UTC
    # calendar months. This gives only a permissive aggregate necessary bound;
    # it does not authorize splitting/moving any charge or promise a boundary.
    physical_months_max = 2
    violations = []
    if cold['seconds'] > physical_months_max * lane['seconds']:
        violations.append('required_cold_prepaid_seconds_exceed_even_two_physical_months')
    for k, n in cold['counts'].items():
        if n > physical_months_max * lane['counts'][k]:
            violations.append('required_cold_' + k + '_exceeds_even_two_physical_months')
    if len(required) > len(dates): violations.append('required_jobs_exceed_original_fit_opportunities')
    if len(warm['streams']) > len(warm['opportunities']):
        violations.append('warmboot_jobs_exceed_original_fit_opportunities')
    not_before = autonomous.get('not_before_month')
    s_months = [x[:7] for x in months if not_before is None or x[:7] >= not_before]
    # Scheduler may stop/fail/skip later slots. These are full planned-work
    # requirements, not a claim every slot is necessarily reserved at runtime.
    s = {'logical_months': s_months, 'planned_monthly_jobs': len(s_months),
         'schedule_hash': fingerprint(search), 'slots_per_month': search['slots'],
         'per_month_reserved_seconds': search['reserved_seconds'],
         'per_month_allocated_counts': search['allocated_counts'],
         'all_planned_reserved_seconds': len(s_months) * search['reserved_seconds'],
         'all_planned_counts': {k: len(s_months) * n for k,n in search['allocated_counts'].items()},
         'owner_key': 'actual_UTC_month|S',
         'owner_identity': 'bridge_id=fingerprint(run_directory,replay_month)',
         'full_planned_search_owner_conflict_even_favourable_two_months': len(s_months) > 2,
         'full_planned_search_same_physical_month_conflict': len(s_months) > 1,
         'actually_consumed': 'unknown_not_prepaid_by_this_analysis'}
    planned_violations = list(violations)
    if planned['seconds'] > 2 * lane['seconds']:
        planned_violations.append('all_planned_model_prepaid_seconds_exceed_even_two_physical_months')
    if len(s_months) > 2:
        planned_violations.append('all_planned_S_months_cannot_each_acquire_distinct_owner_in_at_most_two_physical_months')
    return {'version': VERSION, 'status': 'rejected_necessary_capacity' if violations else 'necessary_only_unknown',
        'proven_required_violations': violations, 'jobs': rows, 'logical_months': months,
        'required_cold_lower_bound': cold, 'all_planned_model_lower_bound': planned,
        'ordinary_A0_planned_jobs': sum('ordinary_A0_planned' in r['roles'] for r in rows),
        'all_planned_completion_conditional_violations': planned_violations,
        'planned_is_not_required_completion': 'Original scheduler can leave planned jobs unattempted or failed; this is not a successful all-planned-work completion.',
        'same_physical_month_forward_limit': lane,
        'physical_start': 'not_assumed_or_backdated', 'max_possible_physical_months': physical_months_max,
        'same_physical_month_required_exceeds': cold['seconds'] > lane['seconds'] or any(cold['counts'][k] > lane['counts'][k] for k in cold['counts']),
        'same_physical_month_all_planned_exceeds': planned['seconds'] > lane['seconds'] or any(planned['counts'][k] > lane['counts'][k] for k in planned['counts']),
        'uniform_convex_ceiling_necessary_max_all_planned_same_month': lane['seconds']//len(rows),
        'uniform_ceiling_condition': 'only if all these jobs are convex at that common ceiling; solves and other jobs still constrain',
        'search': s, 'partition_policy': frozen_policy('standard'),
        'proof': {'seconds_ceiling': 300, 'inside_original_service': True, 'extra_wallet': False},
        'original_fit_admission_headroom_seconds': 302,
        'unknown_costs': ['adaptive A0/algorithm future exact recipes and recursive solves',
            'V/C candidate/reference jobs, algorithm owned jobs and neural resource work',
            'actual S slot execution/early-stop pattern and physical UTC charge placement',
            'IO, source/OS verification, GPT wall time, proof and native completion elapsed time'],
        'prepayment_is_elapsed_lower_bound': False, 'runtime_performance_verified': False,
        'full_cost_enumerated': False, 'execution_authorized': False, 'formal_acceptance': False}


def _derive_elapsed(endpoints, dates, warm, rolling, ceilings, search, autonomous, service):
    """Phase-independent necessary lower bound; never invent data readiness.

    A successful complete path needs a preendpoint job per initial stream and
    at least floor(endpoint_duration / 15 days) distinct subsequent cold jobs.
    Shared streams are counted once. These are aggregate lower bounds, NOT
    exact job keys, a constructed anchor, or proof of successful scheduling.
    """
    from datetime import timedelta
    from .nominal_algorithm_policy import _utc, _month
    from .monthly_partition_policy import counts, frozen_policy
    from .rolling_cadence import RollingCadence
    from .nominal_algorithm_warmboot import ELAPSED_VERSION as WARM_VERSION
    from factorlab.protocol import fingerprint
    cadence=warm.get('fit_schedule')
    if (warm.get('version')!=WARM_VERSION or type(cadence)is not dict
        or cadence.get('cadence_hash')!=RollingCadence().digest
        or cadence.get('training_days')!=90 or cadence.get('refit_days')!=15
        or cadence.get('refresh_days')!=1 or warm.get('cutoff')is not None):
        raise ValueError('Exact original elapsed warmboot schedule required')
    start=_utc(warm['endpoint_start'])
    opportunities=tuple(_utc(t) for t in warm['opportunities'])
    if (len(opportunities)<3 or opportunities!=tuple(sorted(set(opportunities)))
        or any(t not in dates or t>=start for t in opportunities)):
        raise ValueError('Original preendpoint fit opportunities required')
    groups={}
    for stream,pipeline in warm['streams'].items():
        groups[stream]={'stream':stream,'warmboot_min_jobs':1,
            'warmboot_cost':_recipe_cost(pipeline,rolling,ceilings),
            'endpoint_min_jobs':0,'endpoint_cost':None}
    for endpoint in endpoints:
        begin,end=map(_utc,(endpoint['start'],endpoint['end']))
        if begin!=start or end<=begin:
            raise ValueError('Original shared nonempty endpoint window required')
        minimum=(end-begin)//timedelta(days=15)
        for role in ('reference','candidate'):
            stream=endpoint[role+'_stream']
            if stream not in groups:
                raise ValueError('Endpoint changed its original warmboot streams')
            pipeline=None if endpoint['slot']=='algorithm' and role=='candidate' else endpoint[role]
            cost=(_recipe_cost(pipeline,rolling,ceilings) if pipeline is not None else
                  {'class':'adaptive_unknown','seconds':min(ceilings.values()),'counts':counts(solve=1)})
            group=groups[stream]
            if group['endpoint_cost']is not None and group['endpoint_cost']!=cost:
                raise ValueError('Shared original endpoint cost identity differs')
            group['endpoint_min_jobs']=max(group['endpoint_min_jobs'],minimum)
            group['endpoint_cost']=cost
    rows=[groups[k] for k in sorted(groups)]
    cold={'jobs':sum(r['warmboot_min_jobs']+r['endpoint_min_jobs'] for r in rows),
          'seconds':sum(r['warmboot_cost']['seconds']+r['endpoint_min_jobs']*
                        (r['endpoint_cost']['seconds'] if r['endpoint_cost']is not None else 0) for r in rows),
          'counts':{k:sum(r['warmboot_cost']['counts'][k]+r['endpoint_min_jobs']*
                    (r['endpoint_cost']['counts'][k] if r['endpoint_cost']is not None else 0) for r in rows) for k in counts()}}
    lane=frozen_policy('standard')['lanes']['forward'];violations=[]
    if cold['seconds']>2*lane['seconds']:
        violations.append('required_cold_prepaid_seconds_exceed_even_two_physical_months')
    for k,n in cold['counts'].items():
        if n>2*lane['counts'][k]:
            violations.append('required_cold_'+k+'_exceeds_even_two_physical_months')
    if cold['jobs']>len(dates):
        violations.append('required_jobs_exceed_original_fit_opportunities')
    months=sorted({_month(t.isoformat()).isoformat() for t in dates})
    not_before=autonomous.get('not_before_month')
    s_months=[x[:7] for x in months if not_before is None or x[:7]>=not_before]
    return {'version':ELAPSED_VERSION,
        'status':'rejected_necessary_capacity' if violations else 'necessary_only_unknown',
        'proven_required_violations':violations,'required_cold_lower_bound':cold,
        'job_groups_lower_bound':rows,'jobs':None,'cadence_hash':RollingCadence().digest,
        'anchor':'unknown_until_original_actual_data_ready_prefix',
        'logical_months':months,'all_planned_model_lower_bound':None,
        'ordinary_A0_planned_jobs':None,'full_cost_enumerated':False,
        'same_physical_month_forward_limit':lane,'partition_policy':frozen_policy('standard'),
        'max_possible_physical_months':2,'physical_start':'not_assumed_or_backdated',
        'same_physical_month_required_exceeds':cold['seconds']>lane['seconds'] or
            any(cold['counts'][k]>lane['counts'][k] for k in cold['counts']),
        'search':{'logical_months':s_months,'schedule_hash':fingerprint(search),
            'planned_monthly_jobs':len(s_months),'owner_key':'actual_UTC_month|S',
            'actually_consumed':'unknown_not_prepaid_by_this_analysis'},
        'bound_scope':'successful_full_path_phase_independent_minimum_only',
        'unknown_costs':['original actual data-ready anchor and exact cold job keys',
            'adaptive recipes and ordinary A0 execution',
            'actual S execution and physical UTC charge placement',
            'IO, source/OS verification, GPT and proof/native completion'],
        'original_fit_admission_headroom_seconds':302,
        'prepayment_is_elapsed_lower_bound':False,'runtime_performance_verified':False,
        'execution_authorized':False,'formal_acceptance':False}


def _analyze(policy):
    from .nominal_algorithm_policy import NominalAlgorithmPolicy
    from .nominal_algorithm_warmboot import NominalAlgorithmWarmbootPolicy
    from .nominal_short_fit_policy import policy_binding_for_design, NominalShortFitPolicy
    from .full_trial_controller import FullControllerSchedule
    from .monthly_partition_policy import frozen_policy
    from .rolling import RollingSpec
    if type(policy) is not NominalAlgorithmPolicy:
        raise TypeError('Exact frozen nominal algorithm policy required')
    files = _installed()
    policy = NominalAlgorithmPolicy.inspect(policy.design)
    warm = NominalAlgorithmWarmbootPolicy.inspect(policy)
    short = policy_binding_for_design(policy.design)
    ceilings = {'known':300, 'pool':300, 'neural':300}
    if short is not None: ceilings = NominalShortFitPolicy.inspect(policy.design).identity['ceilings']
    scope = policy.design.parent.plan.runtime_scope
    if scope['work_budget']['partition_policy'] != frozen_policy('standard'):
        raise ValueError('Original standard partitions required; no substituted capacity')
    raw = dict(scope['rolling']); raw['event_bins'] = tuple(raw['event_bins'])
    rolling = RollingSpec(**raw)
    schedule = FullControllerSchedule.from_dict(policy.design.identity['full_schedule'])
    value = _derive(policy.identity['endpoints'], schedule.due_at, warm.identity, rolling,
                    ceilings, schedule.search_schedule, schedule.autonomous, schedule.service)
    value.update(source_files=files, policy_digest=policy.digest, design_digest=policy.design.digest,
                 schedule_digest=schedule.digest, warmboot_digest=warm.digest,
                 short_fit_policy=short, ceilings=ceilings)
    if _installed() != files: raise ValueError('Source changed during capacity analysis')
    return _plain(value)


class NominalCapacityPreflight:
    def __init__(self, key, policy, report):
        if key is not _KEY: raise TypeError('Use exact frozen policy analysis')
        self._policy = policy
        self._json = json.dumps(report, sort_keys=True, allow_nan=False)

    @classmethod
    def analyze(cls, policy):
        return cls(_KEY, policy, _analyze(policy))

    @property
    def report(self):
        return json.loads(self._json)

    def assert_current(self):
        if _analyze(self._policy) != self.report:
            raise ValueError('Frozen policy/source/capacity identity changed')

    def require_not_proven_impossible(self):
        """Cold-required-scope rejection only; not a full planned-work grant."""
        self.assert_current()
        if self.report['proven_required_violations']:
            raise ValueError('Known impossible original prepaid cold scope: ' +
                             ','.join(self.report['proven_required_violations']))
        # Deliberately no True/permit: this remains unknown, not an admission.
        return None
