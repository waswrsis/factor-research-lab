"""New nominal-law actual trials; original paid warmboot, no old oracle permit.

Entropy allocations belong to the preregistered full-N experiment, never to a
second fitting wallet. Actual feed receipts and the original forward issuers
retain their physical clocks. This library-only bridge grants no five-slot
calibration or provider certificate.
"""
from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import pandas as pd
from filelock import FileLock

from factorlab.protocol import fingerprint, project_path
from .calibration_acceptance import AcceptanceCell, AcceptancePlan
from .calibration_runtime_contract import _original_operations
from .evidence import file_hash
from .monthly_work_authority import DEFAULT_LIMITS, MonthlyWorkAuthority, source_files
from .v7_lifecycle import stamp
from .v7_storage import immutable_record

VERSION = 'v7-nominal-actual-trial-v5-native-draw'
RELATIVE = 'factorlab/discovery/nominal_actual_trial.py'
EVENT = 'nominal_actual_trial_registered'
SAMPLE_SECONDS = 10
ENTROPY_BITS = 4096
ENTROPY_ATTEMPTS = 1024
STREAMS = {'reference': 'benchmark', 'candidate': 'K0'}
_ACTUAL_READ = ContextVar('original_nominal_actual_registry_read', default=None)
_READ_KEY = object()


class _ActualReadScope:
    """One invocation's checked immutable identity, never a new work lease."""
    def __init__(self, key, registry, lease, identity):
        if key is not _READ_KEY:
            raise TypeError('Only the original private read scope may construct this state')
        self._key = key
        self.registry = registry
        self.lease = lease
        self.owner = (os.getpid(), threading.get_ident())
        self.directory = str(registry.directory)
        self.digest = registry.digest
        self.identity = json.loads(json.dumps(identity, allow_nan=False))
        self.active = True
    def __reduce__(self):
        raise TypeError('Original read scope cannot serialize or restore')
    def read(self, registry):
        if (self._key is not _READ_KEY or not self.active or registry is not self.registry
                or self.owner != (os.getpid(), threading.get_ident())
                or str(registry.directory) != self.directory or registry.digest != self.digest
                or registry.identity != self.identity or fingerprint(registry.identity) != self.digest):
            raise ValueError('Original active same-owner read scope identity differs')
        self.lease.assert_parent_alive()
        # These exact enrollment anchors remain reread on every use. Cached
        # identity is not a caller assertion, persistent certificate or permission
        # to observe newly changed parents. The original source/paid native law
        # are reauthenticated before entry and before successful scope return.
        roots = (registry.directory,
                 project_path(self.identity['grid']['directory']),
                 project_path(self.identity['grid']['verification_registry_directory']))
        names = ('identity.json', 'actual_registry.json', 'actual_registry.json')
        if any(_read(root/name) != self.identity for root, name in zip(roots, names)):
            raise ValueError('Original registry enrollment changed inside read scope')
        self.lease.assert_parent_alive()
        return json.loads(json.dumps(self.identity, allow_nan=False))


def _now():
    return pd.Timestamp.now(tz='UTC')


def _installed():
    files = source_files()
    if (Path(__file__).resolve() != project_path(RELATIVE).resolve()
            or files.get(RELATIVE) != file_hash(__file__)):
        raise ValueError('Canonical new source required; staged component is not production')
    for name in ('nominal_cold_effect_grid', 'nominal_effect_design', 'nominal_effect_grid', 'nominal_verification_budget', 'nominal_cold_conformance', 'rational_sampling', 'nominal_draw_measurement', 'nominal_draw_diagnostic', 'nominal_draw_native', 'nominal_draw_worker'):
        module = __import__(__package__+'.'+name, fromlist=['*'])
        relative = 'factorlab/discovery/'+name+'.py'
        if Path(module.__file__).resolve() != project_path(relative).resolve() or files.get(relative) != file_hash(module.__file__):
            raise ValueError('Canonical original nominal dependencies required')
    return files


def _write(path, value):
    immutable_record(path, {'record': value, 'hash': fingerprint(value)})


def _read(path):
    from .calibration_trial_context import _no_redirect
    original=Path(path)
    if not original.is_absolute():original=project_path('.')/original
    _no_redirect(original,project_path('.'))
    path = project_path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16*2**20:
        raise ValueError('Missing or oversized original actual-trial record')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate actual-trial JSON key')
            result[key] = value
        return result
    value = json.loads(path.read_text('utf8'), object_pairs_hook=unique,
        parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    if set(value) != {'record', 'hash'} or fingerprint(value['record']) != value['hash']:
        raise ValueError('Original actual-trial record integrity changed')
    return value['record']


def _acceptance(value):
    fields = dict(value)
    fields['cells'] = tuple(AcceptanceCell(**{**c, 'power_slots': tuple(c['power_slots'])}) for c in fields['cells'])
    result = AcceptancePlan(**fields)
    if json.loads(json.dumps(result.to_dict())) != value:
        raise ValueError('Original complete acceptance plan changed')
    return result


def _ratio(value):
    if (type(value) is not list or len(value)!=2 or any(type(v) is not int for v in value)
            or value[1]<=0):
        raise ValueError('Canonical exact rational field required')
    f=Fraction(*value)
    if value!=[f.numerator,f.denominator]:
        raise ValueError('Rational field must retain its exact reduced denominator')
    return f


def _scenario_row(grid, name, target, power, *, header):
    from math import lcm
    from .nominal_effect_design import scenario_design_id, scenario_design
    identifier=scenario_design_id(grid.census.digest,name)
    # One freshly authenticated header belongs to this entire full binding.
    # Re-reading the same native task for each of five fixed rows adds neither
    # a new observation nor information. Each exact rational row is still
    # validated separately against its original prospective design below.
    matches = [row for row in header['laws'] if row['scenario_design_id'] == identifier]
    if len(matches) != 1:
        raise ValueError('Unsupported scenario stays unavailable in original N')
    row = matches[0]
    if (type(row) is not dict or row.get('target')!=list(target) or row.get('formal_acceptance') is not False
            or row.get('scenario_design_id')!=identifier or row.get('cell_name')!='library_'+name):
        raise ValueError('Original fixed target grid row required')
    if row.get('status')=='unavailable_fixed_groups_do_not_straddle':
        if set(row)!={'target','status','formal_acceptance','scenario_design_id','cell_name'}:
            raise ValueError('Unavailable target cannot supply alternate weights/proof')
    elif row.get('status')=='available_numeric_only':
        expected={'target','status','formal_acceptance','weights','reference_risk','candidate_risk',
            'gain','group_mass','sampling_denominator','minimum_block_bits','truth','power_target',
            'conditional_pure_sampler_law_only','scenario_design_id','cell_name'}
        if set(row)!=expected or type(row['weights']) is not list or len(row['weights'])!=128:
            raise ValueError('Original complete128 exact effect row schema required')
        weights=tuple(_ratio(w) for w in row['weights']);r=_ratio(row['reference_risk']);c=_ratio(row['candidate_risk'])
        gain=_ratio(row['gain']);mass=_ratio(row['group_mass']);target_f=Fraction(*target)
        denominator=lcm(*(w.denominator for w in weights))
        if (any(w<=0 for w in weights) or sum(weights,Fraction())!=1 or r<=0 or c<0
                or gain!=target_f or c!=(1-target_f)*r or not 0<mass<1
                or weights!=(mass/64,)*64+((1-mass)/64,)*64
                or type(row['sampling_denominator']) is not int or row['sampling_denominator']!=denominator
                or type(row['minimum_block_bits']) is not int or row['minimum_block_bits']!=(denominator-1).bit_length()
                or row['truth']!=('alternative' if power else 'null') or row['power_target'] is not power
                or row['conditional_pure_sampler_law_only'] is not True):
            raise ValueError('Frozen five-effect rational or .5-percent null boundary changed')
    else:
        raise ValueError('Unknown derived-grid availability state')
    return {'design':scenario_design(grid.census.digest,name),'design_id':identifier,
            'numeric':row,'numeric_hash':fingerprint(row)}


def _grid_binding(grid, acceptance):
    from .nominal_cold_effect_grid import ColdEffectGrid
    from .nominal_verification_budget import VerificationRegistry
    from .nominal_cold_conformance import NominalCensus
    from .nominal_effect_design import require_full_library_grid,DESIGNS
    if type(grid) is not ColdEffectGrid or type(grid.registry) is not VerificationRegistry or type(grid.census) is not NominalCensus:
        raise TypeError('Canonical typed full grid with its original verification registry/census required')
    files=_installed();grid.registry.verify();p=grid.census.verify()
    require_full_library_grid(acceptance,grid.census.digest)
    header=grid.header()
    grid_digest=fingerprint(header)
    # This reader verifies the completed canonical semantic artifact and its
    # paid task130/service witnesses. It never invokes full grid.verify here.
    manifest=grid.verified_manifest(130)
    if (set(manifest)!={'grid_digest','registry_digest','census_digest','source_files','acceptance_digest','record','final_service_still_active'}
            or manifest['final_service_still_active'] is not False
            or manifest['grid_digest']!=grid_digest or manifest['registry_digest']!=grid.registry.digest
            or manifest['census_digest']!=grid.census.digest or manifest['source_files']!=files
            or manifest['acceptance_digest']!=acceptance.digest or type(manifest['record']) is not dict
            or not manifest['record'] or grid.registry.plan.acceptance.to_dict()!=acceptance.to_dict()
            or grid.registry.plan.binding['target_directory']!=str(grid.directory)
            or p['source_files']!=files or grid.identity['source_files']!=files):
        raise ValueError('Original paid pre-registration full-grid manifest differs')
    census={'directory':str(grid.census.directory),'digest':grid.census.digest,
        'source_files':p['source_files'],'generator_digest':p['generator_digest'],'law':p['law'],
        'schedule':p['schedule'],'profile_scope':p['profile_scope']}
    binding={'directory':str(grid.directory),'digest':grid_digest,'identity':grid.identity,'header':header,
        'verification_registry_directory':str(grid.registry.directory),'verification_registry_digest':grid.registry.digest,
        'paid_pre_registration_manifest':manifest,
        'scenarios':[_scenario_row(grid,*d,header=header) for d in DESIGNS]}
    return binding,census


def _verify_grid_binding(binding, census, acceptance):
    from .nominal_cold_effect_grid import ColdEffectGrid
    grid=ColdEffectGrid(binding['directory'])
    if _grid_binding(grid,acceptance)!=(binding,census):
        raise ValueError('Original paid complete grid or prospective design mapping changed')
    return grid


def _scenario_binding(binding, design_id):
    matches=[v for v in binding['scenarios'] if v['design_id']==design_id]
    if len(matches)!=1 or matches[0]['numeric']['status']!='available_numeric_only':
        raise ValueError('Unavailable or unsupported cell retains its entire failed/unknown N')
    row=matches[0]
    return {'grid_digest':binding['digest'],'design_id':design_id,'numeric_hash':row['numeric_hash']}


def _entropy(total):
    return {'bits_per_block':ENTROPY_BITS,'attempts_per_trial':ENTROPY_ATTEMPTS,
            'maximum_total_bits':total*ENTROPY_BITS*ENTROPY_ATTEMPTS,
            'seconds_per_allocation':SAMPLE_SECONDS,'maximum_total_seconds':total*SAMPLE_SECONDS,
            'scope':'original_full_N_entropy_quota_not_fit_credit',
            'entropy_model':'independent_uniform_blocks_assumed_not_certified'}


def _work_batches(schedule,frozen_at,mode):
    keys=[{'cutoff':c,'role':r} for c in schedule['cutoffs'] for r in STREAMS]
    if mode=='registered_nominal_simulation':
        first=stamp(frozen_at).normalize().replace(day=1)
        groups=[(first,keys[:9]),(first+pd.DateOffset(months=1),keys[9:])]
    elif mode=='physical_receipts':
        groups=[(stamp(c),[k for k in keys if k['cutoff']==c]) for c in schedule['cutoffs']]
    else:
        raise ValueError('Unknown preregistered actual observation mode')
    return [{'index':i,'actual_UTC_month':m.strftime('%Y-%m'),'model_keys':group,
             'prepaid_fit_seconds':300*len(group),'prepaid_solve':len(group),
             'service_seconds_ceiling':7200,'retry_failed_work':False}
            for i,(m,group) in enumerate(groups)]


def _verify_system_receipt(directory,plan):
    from .rational_sampling import sample_frozen
    directory=project_path(directory)
    def raw(name):
        p=directory/name
        if p.is_symlink() or not p.is_file() or p.stat().st_size>16*2**20:
            raise ValueError('Original bounded entropy file required')
        return p.read_text('utf8')
    allocation=json.loads(raw('allocation.json'))
    terminal=json.loads(raw('terminal.json'))
    events=[json.loads(line) for line in raw('events.jsonl').splitlines()]
    if allocation!={'plan':plan.to_dict(),'reserved_attempts':plan.max_attempts,
                    'directory':str(directory),'retry_allowed':False,'formal_acceptance':False}:
        raise ValueError('Entropy allocation differs from original registry')
    if len(events)>2*ENTROPY_ATTEMPTS or any(e.get('event') not in {'block_request_charged','block_returned'} for e in events):
        raise ValueError('Failed or oversized original entropy journal')
    blocks=tuple(e['block'] for e in events if e['event']=='block_returned')
    replay=sample_frozen(plan,blocks)
    replay['origin']='system_secrets_fresh_blocks_model_assumption'
    if replay!=terminal or terminal['events']!=events or terminal['status']!='sampled':
        raise ValueError('Original entropy journal/terminal differs from exact replay')
    return terminal


def _sampling_plan(binding, design_id):
    from .rational_sampling import SamplingPlan
    _scenario_binding(binding,design_id)
    row=next(v['numeric'] for v in binding['scenarios'] if v['design_id']==design_id)
    return SamplingPlan(tuple(_ratio(w) for w in row['weights']), ENTROPY_BITS, ENTROPY_ATTEMPTS)


class NominalActualRegistry:
    """One whole prospective grid/N; paid-grid anchors forbid per-law experiments."""
    @classmethod
    def freeze(cls, directory, *, grid, acceptance, observation_mode='registered_nominal_simulation'):
        if type(acceptance) is not AcceptancePlan:
            raise TypeError('Original whole-grid AcceptancePlan required')
        binding,census = _grid_binding(grid,acceptance)
        schedule = census['schedule']
        if _now() >= stamp(schedule['cutoffs'][0]):
            raise ValueError('Full actual grid must freeze before warmboot')
        directory = project_path(directory)
        anchor = grid.directory/'actual_registry.json'
        quota_anchor=grid.registry.directory/'actual_registry.json'
        with FileLock(str(anchor)+'.registry-creation.lock', timeout=0):
            if directory.exists() or anchor.exists() or quota_anchor.exists():
                raise ValueError('Original whole grid/quota cannot register another experiment or reset N')
            total = sum(c.repetitions for c in acceptance.cells)
            value = {'version': VERSION, 'directory': str(directory), 'grid': binding, 'census': census,
                'acceptance': json.loads(json.dumps(acceptance.to_dict())),
                'acceptance_digest': acceptance.digest, 'registered_N': total,
                'source_files': _installed(), 'frozen_at': _now().isoformat(),
                'entropy': _entropy(total),
                'actual_observation': observation_mode,
                'fit_budget': 'original_standard_shared_monthly_authority_no_limit_changes',
                'failed_missing_late_mismatch': {'family_error': 1, 'power': 0, 'retain_N': True},
                'library_truth': 'prospective_five_effect_grid_requires_separate_actual_conformance',
                'scenario_availability': {c.name: next((v['numeric']['status'] for v in binding['scenarios'] if v['design_id']==c.scenario_digest),'unsupported_retained_full_N') for c in acceptance.cells},
                'pending_slots': ['factor1', 'factor2', 'factor3', 'algorithm'],
                'formal_acceptance': False}
            value['work_batches']=_work_batches(schedule,value['frozen_at'],observation_mode)
            directory.mkdir(parents=True, exist_ok=False)
            _write(directory/'identity.json', value); _write(anchor, value); _write(quota_anchor,value)
        return cls(directory)

    def __init__(self, directory):
        self.directory = project_path(directory)
        self.identity = _read(self.directory/'identity.json')
        self.digest = fingerprint(self.identity)
        self.verify()

    def verify(self):
        scope = _ACTUAL_READ.get()
        if scope is not None:
            if type(scope) is not _ActualReadScope:
                raise TypeError('Only original private nominal read scope is admissible')
            if scope.registry is self:
                return scope.read(self)
        return self._verify_uncached()

    @contextmanager
    def _read_scope(self, lease):
        from .work_lease import FrozenWorkLease
        if type(self) is not NominalActualRegistry or type(lease) is not FrozenWorkLease:
            raise TypeError('Exact original registry and already-issued sampling lease required')
        if _ACTUAL_READ.get() is not None:
            raise ValueError('Original registry read scope cannot nest or renew')
        lease.assert_parent_alive()
        identity = self._verify_uncached()
        lease.assert_parent_alive()
        scope = _ActualReadScope(_READ_KEY, self, lease, identity)
        token = _ACTUAL_READ.set(scope)
        try:
            yield
            # Successful output is not returned until original source, complete
            # paid law and actual native completion are freshly authenticated.
            # This remains inside the original ten seconds; expiry consumes N.
            lease.assert_parent_alive()
            if self._verify_uncached() != identity:
                raise ValueError('Original paid law changed across bounded read scope')
            scope.read(self)
            lease.assert_parent_alive()
        finally:
            scope.active = False
            _ACTUAL_READ.reset(token)

    def _verify_uncached(self):
        p = self.identity
        plan = _acceptance(p['acceptance'])
        grid = _verify_grid_binding(p['grid'],p['census'],plan)
        if (p != _read(self.directory/'identity.json') or p != _read(grid.directory/'actual_registry.json')
                or p != _read(grid.registry.directory/'actual_registry.json')
                or p['version'] != VERSION or p['directory'] != str(self.directory)
                or p['source_files'] != _installed() or p['entropy']!=_entropy(p['registered_N'])
                or p['scenario_availability']!={c.name: next((v['numeric']['status'] for v in p['grid']['scenarios'] if v['design_id']==c.scenario_digest),'unsupported_retained_full_N') for c in plan.cells}
                or p['work_batches']!=_work_batches(p['census']['schedule'],p['frozen_at'],p['actual_observation'])
                or p['acceptance_digest'] != plan.digest or p['registered_N'] != sum(c.repetitions for c in plan.cells)
                or p['formal_acceptance'] is not False):
            raise ValueError('Original complete nominal actual registry changed')
        return p

    def coordinate(self, cell, index):
        p = self.verify(); plan = _acceptance(p['acceptance'])
        cells = list(plan.cells)
        ordinal = next((i for i,c in enumerate(cells) if c.name == cell), None)
        if ordinal is None or type(index) is not int or not 0 <= index < cells[ordinal].repetitions:
            raise ValueError('Trial is outside original full N')
        return {'registry': self.digest, 'cell': cell, 'cell_ordinal': ordinal, 'index': index,
                'scenario_digest': cells[ordinal].scenario_digest}

    def paths(self, cell, index):
        from .calibration_trial_context import _no_redirect
        c = self.coordinate(cell, index)
        root = self.directory/'trials'/f"v{c['cell_ordinal']:02d}-{index:04d}"
        paths={k: root/v for k,v in {'root': '.', 'runtime': 'forward', 'feed': 'feed',
                'authority': 'wallet', 'entropy': 'entropy'}.items()}
        for path in paths.values():_no_redirect(path,self.directory)
        return paths

    def _allocation_header(self, cell, index):
        """Only this original immutable identity selects an allocation slot.

        Heavy parent/source verification follows durable intent. This bounded
        identity reader grants neither a draw nor a verified semantic witness.
        """
        from .calibration_trial_context import _no_redirect
        p=_read(self.directory/'identity.json')
        if (p!=self.identity or fingerprint(p)!=self.digest or p['version']!=VERSION
                or p['directory']!=str(self.directory)):
            raise ValueError('Original actual allocation header changed')
        plan=_acceptance(p['acceptance'])
        if (plan.digest!=p['acceptance_digest']
                or sum(c.repetitions for c in plan.cells)!=p['registered_N']
                or p['entropy']!=_entropy(p['registered_N'])):
            raise ValueError('Original whole-N allocation header changed')
        ordinal=next((i for i,c in enumerate(plan.cells) if c.name==cell),None)
        if (type(cell) is not str or ordinal is None or type(index) is not int
                or not 0<=index<plan.cells[ordinal].repetitions):
            raise ValueError('Trial is outside original full N')
        c={'registry':self.digest,'cell':cell,'cell_ordinal':ordinal,'index':index,
           'scenario_digest':plan.cells[ordinal].scenario_digest}
        root=self.directory/'trials'/f"v{ordinal:02d}-{index:04d}"
        paths={k:root/v for k,v in {'root':'.','runtime':'forward','feed':'feed',
                'authority':'wallet','entropy':'entropy'}.items()}
        for path in paths.values():_no_redirect(path,self.directory)
        _no_redirect(self.directory/'draw_allocations'/fingerprint(c),self.directory)
        return c,paths

    def draw_trial(self, cell, index):
        """Original finite entropy allocation; no atom/seed/callback argument."""
        from .rational_sampling import sample_system_once
        from .work_lease import FrozenWorkLease, _observe
        c,paths = self._allocation_header(cell,index)
        lease = FrozenWorkLease.issue(SAMPLE_SECONDS)
        mirror=self.directory/'draw_allocations'/fingerprint(c)
        with FileLock(str(paths['root'])+'.lock', timeout=0):
            if paths['root'].exists() or mirror.exists():
                raise ValueError('Trial allocation already consumed; no restart or redraw')
            paths['root'].mkdir(parents=True, exist_ok=False)
            intent = {'version': VERSION, 'coordinate': c, 'registry_directory': str(self.directory),
                'paths': {k: str(v) for k,v in paths.items()}, 'lease': lease.to_dict(),
                'entropy_quota': self.identity['entropy'], 'denominator_contribution': 1,
                'source_files': self.identity['source_files'], 'retry_allowed': False}
            try:
                _write(paths['root']/'draw-intent.json', intent)
                _write(mirror/'intent.json', intent)
                from .nominal_draw_native import native_draw_scope
                with native_draw_scope(self, intent, paths, mirror, lease):
                    from .nominal_draw_diagnostic import draw_scope
                    with draw_scope(self, intent, paths, mirror, lease):
                        with self._read_scope(lease):
                            if self.coordinate(cell,index)!=c or self.paths(cell,index)!=paths:
                                raise ValueError('Original verified coordinate differs from durable allocation')
                            lease.assert_parent_alive()
                            scenario = _scenario_binding(self.identity['grid'],c['scenario_digest'])
                            plan = _sampling_plan(self.identity['grid'],c['scenario_digest'])
                            receipt = sample_system_once(paths['entropy'], plan)
                            checked = _verify_system_receipt(paths['entropy'],plan)
                            if checked != receipt or receipt['status'] != 'sampled':
                                raise ValueError('Original entropy allocation failed or changed')
                            lease.assert_parent_alive()
                            result = {'intent_hash': fingerprint(intent), 'receipt': receipt,
                                'receipt_hash': fingerprint(receipt), 'atom': receipt['atom'], 'scenario_binding':scenario,
                                'physical_finish': _observe(lease), 'status': 'sampled_not_statistically_accepted'}
                            _write(paths['root']/'draw-result.json', result)
                            _write(self.directory/'draw_allocations'/fingerprint(c)/'result.json', result)
                            lease.assert_parent_alive()
                            seal={'result_hash':fingerprint(result),'physical_finish':_observe(lease)}
                            _write(paths['root']/'draw-seal.json',seal)
                            _write(self.directory/'draw_allocations'/fingerprint(c)/'seal.json',seal)
                            lease.assert_parent_alive()
                            trial=NominalActualTrial(self,cell,index)
                            lease.assert_parent_alive()
                    lease.assert_parent_alive()
                lease.assert_parent_alive()
            except BaseException as error:
                failure = {'intent_hash': fingerprint(intent), 'consumed': True,
                    'reason': type(error).__name__, 'family_error': 1, 'power': 0, 'retain_N': True}
                for failure_path in (paths['root']/'failure.json',mirror/'failure.json'):
                    try:
                        if not failure_path.exists():_write(failure_path, failure)
                    except BaseException as recording_error:
                        error.add_note('Original draw failure marker could not be published: '+repr(recording_error))
                raise
        return trial


class NominalActualTrial:
    def __init__(self, registry, cell, index):
        if type(registry) is not NominalActualRegistry:
            raise TypeError('Original typed actual registry required')
        self.registry, self.cell, self.index = registry, cell, index
        self.coordinate = registry.coordinate(cell,index)
        self.paths = registry.paths(cell,index)
        self.digest = fingerprint(self.coordinate)
        self.verify_draw()

    def verify_draw(self):
        from .work_lease import FrozenWorkLease, _evaluate
        self.registry.verify()
        mirror = self.registry.directory/'draw_allocations'/self.digest
        if any(p.exists() for p in (self.paths['root']/'failure.json',mirror/'failure.json')):
            raise ValueError('Original failed trial remains consumed')
        intent = _read(self.paths['root']/'draw-intent.json')
        result = _read(self.paths['root']/'draw-result.json')
        seal=_read(self.paths['root']/'draw-seal.json')
        scenario=_scenario_binding(self.registry.identity['grid'],self.coordinate['scenario_digest'])
        plan=_sampling_plan(self.registry.identity['grid'],self.coordinate['scenario_digest'])
        receipt = _verify_system_receipt(self.paths['entropy'],plan)
        lease = FrozenWorkLease.from_dict(intent['lease'])
        if (intent != _read(mirror/'intent.json') or result != _read(mirror/'result.json')
                or intent['coordinate'] != self.coordinate or intent['paths'] != {k:str(v) for k,v in self.paths.items()}
                or intent['registry_directory'] != str(self.registry.directory)
                or intent['source_files'] != self.registry.identity['source_files']
                or intent['entropy_quota'] != self.registry.identity['entropy']
                or result['scenario_binding']!=scenario
                or lease.lease_seconds != SAMPLE_SECONDS
                or result['intent_hash'] != fingerprint(intent) or result['receipt'] != receipt
                or receipt['plan'] != plan.to_dict()
                or receipt['status'] != 'sampled' or result['atom'] != receipt['atom']
                or result['receipt_hash'] != fingerprint(receipt)
                or seal!=_read(mirror/'seal.json') or seal['result_hash']!=fingerprint(result)
                or not _evaluate(lease,seal['physical_finish'],result['physical_finish']).allowed
                or not _evaluate(lease, result['physical_finish']).allowed):
            raise ValueError('Original registered draw lost its identity, quota or deadline')
        deadline=stamp(lease.wall_deadline).value
        for path in (self.paths['root']/'draw-result.json',self.paths['root']/'draw-seal.json',
                     mirror/'result.json',mirror/'seal.json',self.paths['entropy']/'terminal.json'):
            if path.stat().st_mtime_ns>deadline:
                raise ValueError('Original entropy publication occurred after its physical deadline')
        from .nominal_draw_native import require_native_draw
        require_native_draw(self)
        return result

    def initialize(self):
        """Initialize only this derived actual feed/runtime/standard wallet."""
        from .v7_forward import ForwardRuntime, ForwardConfig
        from .v7_lifecycle import LifecyclePlan
        from .neural_forward_ticket import _bind_authority, open_existing_authority
        from .claim_protocol import ClaimProtocol, FrozenClaimEndpoint
        from .known_linear_reference import FrozenLinearRecipe
        from .joint_pool import FrozenPool
        draw = self.verify_draw(); p=self.registry.identity
        census = p['census']; schedule=census['schedule']; scope=census['profile_scope']
        if _now() >= stamp(schedule['cutoffs'][0]):
            raise ValueError('Actual runtime must initialize before warmboot')
        paths=self.paths
        with FileLock(str(paths['root'])+'.initialize.lock',timeout=0):
            if (paths['root']/'enrollment-intent.json').exists():
                raise ValueError('Enrollment already attempted; incomplete work cannot reset')
            intent={'version': VERSION,'registry_directory':str(self.registry.directory),
                'coordinate':self.coordinate,'draw_hash':fingerprint(draw),'atom':draw['atom'],
                'paths':{k:str(v) for k,v in paths.items()}, 'source_files':p['source_files'],
                'schedule':schedule,'scope':scope,'pending_slots':p['pending_slots'],
                'actual_observation':p['actual_observation'],'work_batches':p['work_batches']}
            _write(paths['root']/'enrollment-intent.json',intent)
            config=dict(scope['forward']); config['symbols']=tuple(config['symbols'])
            runtime=ForwardRuntime(paths['runtime'],paths['feed'],config=ForwardConfig(**config))
            try:
                wanted={**scope['lifecycle_policy'],'source_id':runtime.feed.identity,
                        'calibration_directory':None,'calibration_binding':None}
                actual=json.loads(json.dumps(asdict(runtime.lifecycle.plan)))
                if actual != json.loads(json.dumps(wanted)):
                    raise ValueError('Original actual lifecycle profile requires explicit supported plan')
                wallet=scope['work_budget']
                authority=MonthlyWorkAuthority(paths['authority'],run_id=fingerprint(self.coordinate),
                    task=wallet['task'],universe=wallet['universe'],limits=wallet['limits'],
                    monthly_seconds=wallet['monthly_seconds'],partition_policy=wallet['partition_policy'])
                _bind_authority(runtime,authority)
                har=FrozenLinearRecipe('HAR_cal',runtime.spec); k0=FrozenPool(role='reference',version='K0_v1')
                claims=ClaimProtocol.freeze(runtime.lifecycle,[
                    FrozenClaimEndpoint.library(schedule['start'],runtime.recipe_for(k0),runtime.recipe_for(har)),
                    FrozenClaimEndpoint.algorithm(schedule['start'],runtime.recipe_for(k0))])
                enrolled={**intent,'runtime_identity':runtime.identity,'plan_id':runtime.lifecycle.plan.digest,
                    'authority_manifest_hash':authority.manifest_hash,
                    'claim_protocol_digest':claims.digest,
                    'pipelines':{'reference':runtime.recipe_for(har),'candidate':runtime.recipe_for(k0)}}
                _write(paths['root']/'enrollment.json',enrolled)
                _write(paths['authority']/'nominal_actual_trial.json',enrolled)
                with runtime.lifecycle.db:
                    runtime.lifecycle._event(EVENT,{'registry_directory':str(self.registry.directory),
                        'coordinate':self.coordinate,'enrollment_hash':fingerprint(enrolled)})
                return enrolled
            except BaseException as error:
                _write(paths['root']/'failure.json',{'consumed':True,'reason':type(error).__name__,
                    'family_error':1,'power':0,'retain_N':True})
                raise
            finally:
                runtime.close()

    def fit_month(self, cutoff, role):
        """Actual original paid fit API, including prospective warmboot.

        Uses only the canonical actual receipt feed. No injected clock, bars,
        callback, atom or model is accepted. Failed work retains the full N.
        """
        from .v7_forward import ForwardRuntime, ForwardConfig
        from .v7_lifecycle import LifecyclePlan
        from .neural_forward_ticket import open_existing_authority
        from .known_linear_reference import FrozenLinearRecipe
        from .joint_pool import FrozenPool
        self.verify_draw()
        enrolled=_read(self.paths['root']/'enrollment.json')
        if role not in STREAMS or cutoff not in enrolled['schedule']['cutoffs']:
            raise ValueError('Only the original fourteen actual model work keys are allowed')
        now=_now()
        self.check_work_batch(cutoff,role)
        if enrolled['actual_observation']!='physical_receipts':
            raise ValueError('Simulation fits require the original new nominal controller observation')
        if not stamp(cutoff)<=now<stamp(enrolled['schedule']['end']):
            raise ValueError('Actual cold fit is not physically due/open')
        cfg=dict(enrolled['runtime_identity']['config']);cfg['symbols']=tuple(cfg['symbols'])
        plan=dict(enrolled['runtime_identity']['lifecycle']);plan['symbols']=tuple(plan['symbols'])
        runtime=ForwardRuntime(self.paths['runtime'],self.paths['feed'],
            config=ForwardConfig(**cfg),lifecycle_plan=LifecyclePlan(**plan))
        authority=open_existing_authority(self.paths['authority'])
        try:
            recipe=(FrozenLinearRecipe('HAR_cal',runtime.spec) if role=='reference'
                    else FrozenPool(role='reference',version='K0_v1'))
            return runtime.fit_month(cutoff,stream=STREAMS[role],pool=recipe,authority=authority)
        except BaseException as error:
            _write(self.paths['root']/'failure.json',{'consumed':True,'reason':type(error).__name__,
                'family_error':1,'power':0,'retain_N':True})
            raise
        finally:
            runtime.close()

    def work_allocation(self,cutoff,role):
        self.verify_draw()
        key={'cutoff':cutoff,'role':role}
        matches=[b for b in self.registry.identity['work_batches'] if key in b['model_keys']]
        if len(matches)!=1 or type(role) is not str:
            raise ValueError('Work key lacks one original registered physical batch')
        return json.loads(json.dumps(matches[0]))

    def check_work_batch(self,cutoff,role):
        """Fresh work check from system UTC; caller cannot provide a month."""
        batch=self.work_allocation(cutoff,role)
        if _now().strftime('%Y-%m')!=batch['actual_UTC_month']:
            raise ValueError('Unexecuted original work must await its registered actual UTC batch')
        return batch


def _bound(lifecycle):
    operations = _original_operations(lifecycle)
    bindings = [o for o in operations if o['kind']==EVENT]
    if not bindings:
        return None
    if len(bindings)!=1:
        raise ValueError('Actual trial enrollment is ambiguous')
    b=bindings[0]['detail']; c=b['coordinate']
    registry=NominalActualRegistry(b['registry_directory'])
    trial=NominalActualTrial(registry,c['cell'],c['index'])
    value=_read(trial.paths['root']/'enrollment.json')
    intent=_read(trial.paths['root']/'enrollment-intent.json')
    from .neural_forward_ticket import open_existing_authority, _verify_authority_binding
    authority=open_existing_authority(trial.paths['authority'])
    _verify_authority_binding(lifecycle,authority,operations)
    if (c!=trial.coordinate or value!=_read(authority.directory/'nominal_actual_trial.json')
            or fingerprint(value)!=b['enrollment_hash'] or any(value[k]!=v for k,v in intent.items())
            or lifecycle.directory!=trial.paths['runtime'] or lifecycle.plan.digest!=value['plan_id']
            or value['authority_manifest_hash']!=authority.manifest_hash
            or authority.manifest['scope']['run_id']!=fingerprint(c)
            or value['source_files']!=registry.identity['source_files']
            or authority.manifest['source_files']!=value['source_files']
            or value['draw_hash']!=fingerprint(trial.verify_draw())):
        raise ValueError('Actual trial original runtime/wallet/draw anchors changed')
    from .neural_forward_ticket import _read as original_read
    if original_read(lifecycle.directory/'forward.json')!={'identity':value['runtime_identity'],
            'sha256':fingerprint(value['runtime_identity'])}:
        raise ValueError('Original registered actual forward source identity changed')
    from .claim_protocol import ClaimProtocol
    claims=ClaimProtocol(lifecycle)
    if (claims.digest!=value['claim_protocol_digest']
            or claims.endpoints['library_skill'].start!=value['schedule']['start']
            or claims.endpoints['library_skill'].end!=value['schedule']['end']
            or claims.endpoints['library_skill'].reference!=value['pipelines']['reference']
            or claims.endpoints['library_skill'].candidate!=value['pipelines']['candidate']):
        raise ValueError('Original actual six-month claim endpoints changed')
    return value,bindings[0],operations


def job_context(lifecycle, job, pipeline):
    """Read-only original identity, including warmboot before claim.start."""
    bound=_bound(lifecycle)
    if bound is None:
        return None
    value,event,operations=bound
    row=lifecycle.db.execute('SELECT cutoff,stream,pipeline FROM jobs WHERE id=?',(job,)).fetchone()
    role=next((k for k,v in STREAMS.items() if row is not None and v==row[1]),None)
    if role is None or json.loads(row[2])!=pipeline or pipeline!=value['pipelines'][role]:
        raise ValueError('Actual nominal trial cannot authorize another stream/recipe')
    if row[0] not in value['schedule']['cutoffs']:
        raise ValueError('Actual trial cutoff is outside its seven registered cold fits')
    planned=[o for o in operations if o['kind']=='fit_planned' and o['detail']['job']==job]
    if len(planned)!=1 or planned[0]['seq']<=event['seq']:
        raise ValueError('Original actual work must be planned after its enrollment')
    physical=stamp(planned[0]['at']); observed=physical
    if value['actual_observation']=='registered_nominal_simulation':
        from .nominal_trial_controller import job_binding
        prefix=job_binding(lifecycle,job)
        if (prefix is None or prefix.get('version')!='nominal-observation-v1'
                or prefix.get('trial_digest')!=fingerprint(value['coordinate'])
                or planned[0]['detail'].get('observation_prefix')!=prefix):
            raise ValueError('Actual simulation lost its original new registered observation')
        observed=stamp(prefix['logical_at']);physical=stamp(prefix['physical_observation']['wall'])
    elif value['actual_observation']!='physical_receipts' or 'observation_prefix' in planned[0]['detail']:
        raise ValueError('Physical receipt trial cannot borrow any logical observation')
    batch=next(b for b in value['work_batches'] if {'cutoff':row[0],'role':role} in b['model_keys'])
    if (observed<stamp(row[0]) or observed>=stamp(value['schedule']['end'])
            or physical.strftime('%Y-%m')!=batch['actual_UTC_month']):
        raise ValueError('Original actual planning time/prefix cannot be substituted')
    return {'role':'original_nominal_actual_trial','nominal_trial':value['coordinate'],
            'nominal_enrollment_hash':fingerprint(value),'nominal_role':role,
            'nominal_work_batch':batch['index']}


def validate_preparation(runtime, job, pipeline):
    """Retain the ordinary actual-receipt snapshot; never supply alternate bars."""
    context=job_context(runtime.lifecycle,job,pipeline)
    if context is not None:
        value,_,_=_bound(runtime.lifecycle)
        batch=next(b for b in value['work_batches'] if b['index']==context['nominal_work_batch'])
        if _now().strftime('%Y-%m')!=batch['actual_UTC_month']:
            raise ValueError('Original fit cannot charge another actual work month')
        from .quarantined_observation import current_binding, job_binding
        current=current_binding(runtime.lifecycle); prefix=job_binding(runtime.lifecycle,job)
        if value['actual_observation']=='physical_receipts':
            if current is not None or prefix is not None:
                raise ValueError('Actual physical warmboot cannot borrow logical observation')
        elif current!=prefix or prefix is None or prefix.get('version')!='nominal-observation-v1':
            raise ValueError('New registered simulation requires its current original prefix')
        else:
            from .nominal_trial_controller import require_batch_work
            c=value['coordinate']
            registry=NominalActualRegistry(value['registry_directory'])
            trial=NominalActualTrial(registry,c['cell'],c['index'])
            require_batch_work(trial,runtime.lifecycle.db.execute('SELECT cutoff FROM jobs WHERE id=?',(job,)).fetchone()[0],
                               context['nominal_role'])
    return context
