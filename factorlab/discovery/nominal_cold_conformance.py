"""New scientific identity: paid full cold nominal HAR / empty-pool K0.

This is not the old joint-service estimand, a serving admission, a sampler,
or a complete five-slot calibrator. Local immutable records are not signatures
against an administrator. Missing work always leaves the whole census undefined.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from filelock import FileLock

from factorlab.protocol import fingerprint, project_path
from .evidence import file_hash
from .v7_storage import immutable_record
from .v7_lifecycle import stamp, boundary
from .monthly_work_authority import MonthlyWorkAuthority, source_files, DEFAULT_LIMITS
from .rolling import RollingSpec, build_rolling_data
from .finite_market_oracle import FiniteGenerator, _finite_loss
from .known_linear_reference import FrozenLinearRecipe, KnownLinearPredictor, fit_known_linear
from .joint_pool import FrozenPool, JointPredictor, fit_pool
from .confirmatory import ConfirmationRule, paired_effect_interval

VERSION = 'v7-nominal-full-cold-rolling-library-v1'
ESTIMAND = 'nominal_full_cold_rolling'
RELATIVE = 'factorlab/discovery/nominal_cold_conformance.py'
ROLES = ('reference', 'candidate')
PENDING = ['factor1', 'factor2', 'factor3', 'algorithm']
COUNTS = {k: 2*int(k == 'solve') for k in DEFAULT_LIMITS}


def _plain(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _write(path, value):
    immutable_record(path, {'record': _plain(value), 'hash': fingerprint(_plain(value))})


def _read(path):
    path = project_path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64*2**20:
        raise ValueError('Missing/oversized/symlink nominal evidence')
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError('Duplicate nominal JSON key')
            out[k] = v
        return out
    obj = json.loads(path.read_text('utf-8'), object_pairs_hook=pairs,
                     parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    if set(obj) != {'record', 'hash'} or fingerprint(obj['record']) != obj['hash']:
        raise ValueError('Nominal artifact hash/schema changed')
    return obj['record']


def _installed():
    files = source_files()
    if (Path(__file__).resolve() != project_path(RELATIVE).resolve()
            or files.get(RELATIVE) != file_hash(__file__)):
        raise ValueError('Reviewed integration and fresh canonical source required; staged execution forbidden')
    return files


def calendar(start, delay=5):
    """Tick before fit; HAR ready at next tick, K0 one tick after HAR.

    Warmboot at start minus one month supplies both previous models to the
    first one/two endpoint origins. No instantaneous month-boundary model.
    """
    start = boundary(start)
    if type(delay) is not int or not 1 <= delay < 3600:
        raise ValueError('Strictly inside-hour original prediction delay required')
    end = start + pd.DateOffset(months=6)
    cutoffs = [start+pd.DateOffset(months=i) for i in range(-1, 6)]
    return {'start': start.isoformat(), 'end': end.isoformat(), 'delay': delay,
        'cutoffs': [c.isoformat() for c in cutoffs],
        'raw_start': (cutoffs[0]-pd.DateOffset(months=3)-pd.Timedelta(hours=1234)).isoformat(),
        'raw_end': (end+pd.Timedelta(hours=36)).isoformat(),
        'hours': int((end-start)/pd.Timedelta(hours=1)),
        'ready': {role: {c.isoformat(): (c+pd.Timedelta(hours=i+1, seconds=delay)).isoformat()
                        for c in cutoffs} for i, role in enumerate(ROLES)},
        'rule': 'tick_before_fit_reference_then_candidate_previous_month_until_fixed_ready'}


def selected_cutoff(schedule, role, origin):
    if role not in ROLES:
        raise ValueError('Unknown fixed nominal role')
    origin = stamp(origin)
    if not stamp(schedule['start']) <= origin < stamp(schedule['end']):
        raise ValueError('Origin outside complete six-month endpoint')
    if origin != origin.floor('h')+pd.Timedelta(seconds=schedule['delay']):
        raise ValueError('Origin not on preregistered fixed tick')
    legal = [c for c in schedule['cutoffs'] if stamp(schedule['ready'][role][c]) <= origin]
    if not legal:
        raise ValueError('Warmboot model absent')
    return legal[-1]


def numerical_model(model):
    """Compare all numerical state, retaining recipe/columns/scaler/parameters.

    Only physical provenance and training receipt are separate: their source,
    raw inputs, calendar and paid admission are independently authenticated.
    """
    value = _plain(model.to_dict())
    value.pop('receipt')
    value['predictor'].pop('provenance')
    return value


class NominalCensus:
    """New preregistered 128-replica scope, never an old joint permit.

    Inherits the original per-replica standard wallet ceilings. Each atom has
    exactly seven paired reservations, fourteen cold solves and no retry. One
    raw generator can enroll only once; aliases cannot mint another census.
    """
    @classmethod
    def freeze(cls, directory, *, profile_runtime, profile_authority, generator, start):
        from .v7_forward import ForwardRuntime
        from .calibration_runtime_contract import runtime_calibration_contract
        from .monthly_partition_policy import frozen_policy
        if (type(profile_runtime) is not ForwardRuntime or type(profile_authority) is not MonthlyWorkAuthority
                or type(generator) is not FiniteGenerator):
            raise TypeError('Actual original source/scope and fresh raw generator required')
        files = _installed(); generator.verify(); profile_authority.verify()
        scope = runtime_calibration_contract(profile_runtime.lifecycle)
        budget = scope['work_budget']
        from .monthly_partition_policy import _runtime
        _runtime(profile_authority, profile_runtime.directory)
        if (scope['source_files'] != files or generator.identity['source_files'] != files
                or budget['limits'] != DEFAULT_LIMITS or budget['monthly_seconds'] != 7200
                or budget['partition_policy'] != frozen_policy('standard')
                or budget['universe'] != list(generator.law.symbols)
                or generator.law.support_size != 128 or generator.law.gap_probability != 0):
            raise ValueError('Exact original standard per-replica budget and complete128 law required')
        schedule = calendar(start)
        if (pd.Timestamp.now(tz='UTC') >= stamp(schedule['cutoffs'][0])
                or stamp(generator.law.start) > stamp(schedule['raw_start'])
                or stamp(generator.law.start)+pd.Timedelta(hours=generator.law.hours) < stamp(schedule['raw_end'])):
            raise ValueError('Prospective entire six-month law plus warmboot and full1234 warmup required')
        directory = project_path(directory)
        anchor = generator.directory/'nominal_census.json'
        with FileLock(str(anchor)+'.census-creation.lock', timeout=0):
            if directory.exists() or anchor.exists() or any(
                    p.name not in {'identity.json', 'identity.json.lock', 'nominal_census.json.lock', 'nominal_census.json.census-creation.lock'}
                    for p in generator.directory.iterdir()):
                raise ValueError('Fresh independently registered nominal generator required; no scope reset')
            value = {'version': VERSION+'-census', 'estimand': ESTIMAND,
                'directory': str(directory), 'generator_directory': str(generator.directory),
                'generator_digest': generator.digest, 'law': generator.law.to_dict(),
                'source_files': files, 'profile_scope': scope, 'schedule': schedule,
                'atoms': list(range(128)), 'pair_slots_per_atom': 7, 'cold_fits_per_atom': 14,
                'counts_per_pair': COUNTS, 'seconds_per_pair': 300,
                'seconds_per_atom': 2100, 'whole_census_seconds': 268800,
                'authorization_scope': 'new_nominal_calibration_replicas_not_production_credit',
                'ancestry': 'original_fixed_estimand_replicas_per_atom_standard_wallet',
                'retry': False, 'formal_acceptance': False, 'pending_slots': PENDING,
                'frozen_at': pd.Timestamp.now(tz='UTC').isoformat()}
            directory.mkdir(parents=True)
            _write(directory/'identity.json', value); _write(anchor, value)
        return cls(directory)

    def __init__(self, directory):
        self.directory = project_path(directory)
        self.identity = _read(self.directory/'identity.json')
        self.digest = fingerprint(self.identity)
        self.verify()

    def verify(self):
        p = self.identity
        from .monthly_partition_policy import frozen_policy
        from .finite_market_oracle import _read as read_generator
        generator_identity = read_generator(project_path(p['generator_directory'])/'identity.json')
        if (p != _read(self.directory/'identity.json')
                or p != _read(project_path(p['generator_directory'])/'nominal_census.json')
                or p['directory'] != str(self.directory) or p['source_files'] != _installed()
                or p['version'] != VERSION+'-census' or p['estimand'] != ESTIMAND
                or p['atoms'] != list(range(128)) or p['schedule'] != calendar(p['schedule']['start'])
                or p['counts_per_pair'] != COUNTS or p['seconds_per_pair'] != 300
                or fingerprint(generator_identity) != p['generator_digest']
                or generator_identity['law'] != p['law']
                or generator_identity['source_files'] != p['source_files']
                or p['profile_scope']['work_budget']['limits'] != DEFAULT_LIMITS
                or p['profile_scope']['work_budget']['monthly_seconds'] != 7200
                or p['profile_scope']['work_budget']['partition_policy'] != frozen_policy('standard')):
            raise ValueError('Complete original census scope/source/replica ceilings changed')
        return p

    def paths(self, atom):
        self.verify()
        if type(atom) is not int or not 0 <= atom < 128:
            raise ValueError('Original atom outside complete128 census')
        root = self.directory/'atoms'/f'{atom:04d}'
        return {name: root/name for name in ('runtime', 'feed', 'authority', 'protocol')}

    def run_id(self, atom):
        self.paths(atom)
        return fingerprint({'nominal_census': self.digest, 'atom': atom})

    def initialize_atom(self, atom, *, generator):
        """Future initialization only; no fitting, old budget transfer or clock override.

        A partial initialization remains blocked for inspection. No reset/resume
        factory silently creates a new wallet after an interrupted allocation.
        """
        from .v7_forward import ForwardRuntime, ForwardConfig
        from .v7_lifecycle import LifecyclePlan
        from .v7_feed import ReceiptArchive
        from .neural_forward_ticket import _bind_authority
        from .claim_protocol import ClaimProtocol, FrozenClaimEndpoint
        p = self.verify(); paths = self.paths(atom)
        if type(generator) is not FiniteGenerator or generator.digest != p['generator_digest']:
            raise TypeError('Original registered raw generator required')
        generator.verify()
        root = paths['runtime'].parent
        with FileLock(str(root)+'.lock', timeout=0):
            if root.exists():
                raise ValueError('Replica already initialized/attempted; inspect, never reset')
            root.mkdir(parents=True)
            _write(root/'initialization.json', {'census': self.digest, 'atom': atom, 'run_id': self.run_id(atom)})
            scope = p['profile_scope']; config = dict(scope['forward']); config['symbols'] = tuple(config['symbols'])
            feed = ReceiptArchive(paths['feed'], config['symbols'])
            source = feed.identity; feed.close()
            plan = dict(scope['lifecycle_policy']); plan['symbols'] = tuple(plan['symbols'])
            runtime = ForwardRuntime(paths['runtime'], paths['feed'], config=ForwardConfig(**config),
                                     lifecycle_plan=LifecyclePlan(source_id=source, **plan))
            authority = MonthlyWorkAuthority(paths['authority'], run_id=self.run_id(atom),
                task=scope['work_budget']['task'], universe=config['symbols'], limits=DEFAULT_LIMITS,
                monthly_seconds=7200, partition_policy='standard')
            try:
                _bind_authority(runtime, authority)
                k0 = runtime.recipe_for(FrozenPool(role='reference', version='K0_v1'))
                har = runtime.recipe_for(FrozenLinearRecipe('HAR_cal', runtime.spec))
                ClaimProtocol.freeze(runtime.lifecycle, [FrozenClaimEndpoint.library(p['schedule']['start'], k0, har),
                                                        FrozenClaimEndpoint.algorithm(p['schedule']['start'], k0)])
                return NominalColdProtocol.freeze(paths['protocol'], census=self, atom=atom, runtime=runtime,
                    authority=authority, generator=generator, start=p['schedule']['start'])
            finally:
                runtime.close()

    def assemble_atom(self, atom):
        """Public streaming census API: authority is derived, never caller supplied."""
        from .neural_forward_ticket import open_existing_authority
        paths = self.paths(atom)
        authority = open_existing_authority(paths['authority'])
        if authority.manifest['scope']['run_id'] != self.run_id(atom):
            raise ValueError('Original registered replica wallet differs')
        protocol = NominalColdProtocol(paths['protocol'])
        return assemble_atom(protocol, authority=authority, atom=atom)


class NominalColdProtocol:
    @classmethod
    def freeze(cls, directory, *, census, atom, runtime, authority, generator, start):
        from .v7_forward import ForwardRuntime
        from .monthly_partition_policy import _runtime
        from .neural_forward_ticket import runtime_identity
        if (type(runtime) is not ForwardRuntime or type(authority) is not MonthlyWorkAuthority
                or type(generator) is not FiniteGenerator):
            raise TypeError('Original runtime, shared authority and finite raw generator required')
        if type(census) is not NominalCensus:
            raise TypeError('Original new nominal census registration required')
        census.verify()
        paths = census.paths(atom)
        if (project_path(directory) != paths['protocol'] or runtime.directory != paths['runtime']
                or authority.directory != paths['authority']
                or authority.manifest['scope']['run_id'] != census.run_id(atom)
                or generator.digest != census.identity['generator_digest']):
            raise ValueError('Replica cannot borrow/create a caller-selected wallet')
        files = _installed()
        authority.verify(); runtime.lifecycle.verify(); generator.verify()
        owner = _runtime(authority, runtime.directory)
        from .calibration_runtime_contract import runtime_calibration_contract
        if runtime_calibration_contract(runtime.lifecycle) != census.identity['profile_scope']:
            raise ValueError('Replica changed the original full runtime/budget scope')
        if (authority.manifest['source_files'] != files or runtime.identity['source_files'] != files
                or generator.identity['source_files'] != files
                or authority.manifest['partition_policy'] is None):
            raise ValueError('Fresh same-source identities and original partitioned wallet required')
        # Fresh raw generator identity, never reinterpret old joint/oracle records.
        if any(p.name not in {'identity.json', 'identity.json.lock', 'nominal_census.json', 'nominal_census.json.lock', 'nominal_census.json.census-creation.lock'} for p in generator.directory.iterdir()):
            raise ValueError('Nominal registration requires a fresh unenrolled raw generator')
        if any((runtime.directory/name).exists() for name in ('quarantined_observation', 'fixed_recipe_oracle')):
            raise ValueError('Old joint execution cannot be reinterpreted as nominal')
        spec = runtime.spec
        if spec != RollingSpec(availability_delay_seconds=5):
            raise ValueError('First nominal version freezes original default volatility12h recipe')
        schedule = calendar(start, spec.availability_delay_seconds)
        law = generator.law
        if (law.support_size != 128 or law.gap_probability != 0
                or tuple(law.symbols) != tuple(runtime.config.symbols)
                or stamp(law.start) > stamp(schedule['raw_start'])
                or stamp(law.start)+pd.Timedelta(hours=law.hours) < stamp(schedule['raw_end'])):
            raise ValueError('Full128 complete raw warmup/end support required; gap-law extension pending')
        endpoint = runtime.claim_protocol()
        if endpoint is None:
            raise ValueError('Original prospective six-month claims must already be frozen')
        endpoint.verify()
        library = endpoint.endpoints['library_skill']
        ref = runtime.recipe_for(FrozenLinearRecipe('HAR_cal', spec))
        candidate = runtime.recipe_for(FrozenPool(role='reference', version='K0_v1'))
        if (library.start != schedule['start'] or library.end != schedule['end']
                or library.rule != ConfirmationRule() or library.reference != _plain(ref)
                or library.candidate != _plain(candidate)
                or runtime.lifecycle.now() >= stamp(schedule['cutoffs'][0])
                or pd.Timestamp.now(tz='UTC') >= stamp(schedule['cutoffs'][0])):
            raise ValueError('Preregister unchanged original endpoint before warmboot observation')
        directory = project_path(directory)
        # Single nominal registration for this actual runtime; not aliases per atom.
        mirror = authority.directory/'nominal_cold_protocols'/fingerprint(owner)
        identity = {'version': VERSION, 'estimand': ESTIMAND, 'directory': str(directory),
            'mirror': str(mirror), 'runtime_directory': str(runtime.directory),
            'census_directory': str(census.directory), 'census_digest': census.digest, 'atom': atom,
            'owner': owner, 'authority_directory': str(authority.directory),
            'authority_manifest_hash': authority.manifest_hash, 'source_files': files,
            'generator_directory': str(generator.directory), 'generator_digest': generator.digest,
            'law': law.to_dict(), 'runtime_identity': runtime_identity(), 'schedule': schedule,
            'spec': asdict(spec), 'endpoint': library.to_dict(), 'claim_protocol_digest': endpoint.digest,
            'population': 'evaluation', 'roles': list(ROLES), 'counts_per_pair': COUNTS,
            'lease_seconds_per_pair': 300, 'cold_fit_slots': 7*2,
            'worst_prepaid_seconds': 7*300,
            'model_cache_as_cold': False, 'retry': False, 'formal_acceptance': False,
            'five_slots': ['library_skill', *PENDING], 'pending_slots': PENDING,
            'mc_alpha': .025, 'oracle_alpha': .025,
            'raw_availability': 'synthetic_nominal_close_plus_5s_not_actual_receipt',
            'frozen_at': pd.Timestamp.now(tz='UTC').isoformat()}
        with FileLock(str(mirror)+'.lock', timeout=0):
            if directory.exists() or mirror.exists():
                raise ValueError('Fresh nominal registry required; no resume or migration')
            directory.mkdir(parents=True); mirror.mkdir(parents=True)
            _write(directory/'identity.json', identity); _write(mirror/'identity.json', identity)
        return cls(directory)

    def __init__(self, directory):
        self.directory = project_path(directory)
        self.identity = _read(self.directory/'identity.json')
        self.digest = fingerprint(self.identity)
        self.verify()

    def verify(self):
        p = self.identity
        if (p != _read(self.directory/'identity.json')
                or p != _read(project_path(p['mirror'])/'identity.json')
                or p['version'] != VERSION or p['estimand'] != ESTIMAND
                or p['directory'] != str(self.directory) or p['source_files'] != _installed()
                or p['schedule'] != calendar(p['schedule']['start'])
                or p['counts_per_pair'] != COUNTS or p['lease_seconds_per_pair'] != 300
                or p['formal_acceptance'] is not False):
            raise ValueError('Frozen nominal scientific/source identity changed')
        census = NominalCensus(p['census_directory'])
        paths = census.paths(p['atom'])
        if (census.digest != p['census_digest'] or paths['protocol'] != self.directory
                or paths['authority'] != project_path(p['authority_directory'])
                or paths['runtime'] != project_path(p['runtime_directory'])):
            raise ValueError('Nominal replica mapping differs')
        return p

    def plan(self, atom, cutoff, role):
        p = self.verify()
        if type(atom) is not int or atom != p['atom'] or role not in ROLES or cutoff not in p['schedule']['cutoffs']:
            raise ValueError('Work outside fixed full cold census')
        return {'protocol': self.digest, 'atom': atom, 'cutoff': cutoff, 'role': role,
            'train_start': (stamp(cutoff)-pd.DateOffset(months=3)).isoformat(),
            'ready': p['schedule']['ready'][role][cutoff], 'counts': COUNTS, 'lease_seconds': 300}

    def pair_plan(self, atom, cutoff):
        roles = [self.plan(atom, cutoff, role) for role in ROLES]
        return {'protocol': self.digest, 'atom': atom, 'cutoff': cutoff, 'roles': roles,
                'counts': COUNTS, 'lease_seconds': 300}


def validate_nominal_route(authority, work_key, binding, lease, debit):
    """Called by the one original wallet, including its historical row audit."""
    from .monthly_partition_policy import _runtime
    root = authority.directory/'nominal_cold_intents'/fingerprint(work_key)
    intent = _read(root/'intent.json')
    protocol = NominalColdProtocol(intent['protocol_directory'])
    p = protocol.identity
    plan = protocol.pair_plan(intent['plan']['atom'], intent['plan']['cutoff'])
    expected_key = 'nominal_cold:'+fingerprint(plan)
    expected_binding = {'kind': VERSION, 'protocol': protocol.digest, 'plan_hash': fingerprint(plan)}
    target = protocol.directory/'fits'/fingerprint(plan)
    if (intent != _read(target/'intent.json') or intent != {
            'protocol_directory': str(protocol.directory), 'plan': plan, 'work_key': expected_key,
            'binding': expected_binding, 'lease': lease.to_dict()}
            or expected_key != work_key or binding != expected_binding or debit != COUNTS
            or lease.lease_seconds != 300 or p['authority_manifest_hash'] != authority.manifest_hash
            or p['authority_directory'] != str(authority.directory)
            or p['source_files'] != authority.manifest['source_files']):
        raise ValueError('Nominal work changed fixed role/source/original paid lease')
    owner = _runtime(authority, p['runtime_directory'])
    census = NominalCensus(p['census_directory'])
    budget = census.identity['profile_scope']['work_budget']
    if (owner != p['owner'] or authority.manifest['scope']['run_id'] != census.run_id(p['atom'])
            or authority.manifest['limits'] != budget['limits']
            or authority.manifest['monthly_seconds'] != budget['monthly_seconds']
            or authority.manifest['partition_policy'] != budget['partition_policy']):
        raise ValueError('Nominal cannot acquire another monthly owner')
    return {'lane': 'forward', 'owner': owner}


def _rows(data, model, protocol, plan):
    p = protocol.identity; schedule = p['schedule']
    predictions, valid = model.predict_history(data.frame)
    rows = []
    for i, meta in data.frame.metadata.iterrows():
        origin = stamp(meta.available_at)
        if not stamp(schedule['start']) <= origin < stamp(schedule['end']):
            continue
        if selected_cutoff(schedule, plan['role'], origin) != plan['cutoff']:
            continue
        if not valid[i] or not data.observed[i]:
            raise ValueError('Undefined nominal support cannot be dropped')
        pred, label = float(predictions[i]), float(data.target[i])
        rows.append({'origin': origin.isoformat(), 'hour': origin.floor('h').isoformat(),
            'symbol': meta.symbol, 'feature_open_time': stamp(meta.feature_open_time).isoformat(),
            'window': stamp(meta.forecast_window_start).isoformat(), 'cutoff': plan['cutoff'],
            'train_start': plan['train_start'], 'prediction': pred, 'label': label,
            'maturity': stamp(data.maturity[i]).isoformat(), 'loss': _finite_loss(label, pred)})
    expected = [(t.isoformat(), s) for t in pd.date_range(schedule['start'], schedule['end'], freq='h', inclusive='left')
                +pd.Timedelta(seconds=schedule['delay']) for s in p['law']['symbols']
                if selected_cutoff(schedule, plan['role'], t) == plan['cutoff']]
    if sorted((r['origin'], r['symbol']) for r in rows) != sorted(expected):
        raise ValueError('Incomplete original nominal symbol/hour denominator')
    return rows


def produce_month(protocol, *, runtime, authority, generator, atom, cutoff):
    """One prepaid paired cold fit (two solves, original 300-second cap). Never runs staged code, caches fits or retries.

    Original kernel membership/memory/CPU checks and the same 300-second lease
    surround raw reconstruction, feature preparation, fit, predictions and I/O.
    """
    from .v7_forward import ForwardRuntime
    from .reserved_work_budget import ReservedChannelBudget
    from .work_lease import FrozenWorkLease
    from .nonlinear_launcher import _kernel_snapshot, _validate_kernel
    from .neural_forward_launcher import _identity, _member, _observation
    if (type(protocol) is not NominalColdProtocol or type(runtime) is not ForwardRuntime
            or type(authority) is not MonthlyWorkAuthority or type(generator) is not FiniteGenerator):
        raise TypeError('Actual original objects required, no callback/role credential')
    p = protocol.verify(); plan = protocol.pair_plan(atom, cutoff)
    if (str(runtime.directory) != p['runtime_directory'] or authority.manifest_hash != p['authority_manifest_hash']
            or generator.digest != p['generator_digest'] or str(generator.directory) != p['generator_directory']):
        raise ValueError('Actual input binding differs')
    key = 'nominal_cold:'+fingerprint(plan); folder = protocol.directory/'fits'/fingerprint(plan)
    mirror = authority.directory/'nominal_cold_intents'/fingerprint(key)
    with FileLock(str(mirror)+'.lock', timeout=0):
        if folder.exists() or mirror.exists():
            raise ValueError('Attempt already allocated: failures/interruption cannot retry')
        lease = FrozenWorkLease.issue(300)
        intent = {'protocol_directory': str(protocol.directory), 'plan': plan, 'work_key': key,
            'binding': {'kind': VERSION, 'protocol': protocol.digest, 'plan_hash': fingerprint(plan)},
            'lease': lease.to_dict()}
        folder.mkdir(parents=True); mirror.mkdir(parents=True)
        _write(folder/'intent.json', intent); _write(mirror/'intent.json', intent)
        record = None
        try:
            record = authority._reserve_with_lease(key, counts=COUNTS, binding=intent['binding'], lease=lease)
            if record['status'] != 'reserved':
                raise ValueError('Fresh original reservation required')
            first = _kernel_snapshot(); _validate_kernel(first, first); _member(first, _identity())
            record = authority.start(record['reservation_id'], _identity())
            budget = ReservedChannelBudget.from_issuer_reservation(authority, record['reservation_id'])
            raw = generator.reconstruct_atom(atom, population=p['population'])
            if any(s['gap'] for s in raw['schedule']):
                raise ValueError('Registered complete nominal raw path unexpectedly has gaps')
            _write(folder/'raw.json', raw)
            bars = pd.DataFrame(raw['rows'])
            bars['open_time'] = pd.to_datetime(bars.open_time, utc=True)
            bars['available_at'] = bars.open_time+pd.Timedelta(hours=1, seconds=5)
            bars.attrs['availability_mode'] = 'assumed_historical_delay'
            budget.check_time()
            data = build_rolling_data(bars, RollingSpec(**p['spec']))
            budget.check_time()
            outputs = {}
            for role in ROLES:
                model = (fit_known_linear(data, cutoff, FrozenLinearRecipe('HAR_cal', data.spec), budget=budget)
                         if role == 'reference' else fit_pool(data, cutoff,
                             FrozenPool(role='reference', version='K0_v1'), budget=budget))
                role_plan = protocol.plan(atom, cutoff, role)
                rows = _rows(data, model, protocol, role_plan)
                model.save(folder/(role+'-model.json'))
                restored = (KnownLinearPredictor if role == 'reference' else JointPredictor).load(folder/(role+'-model.json'))
                if restored.digest != model.digest:
                    raise ValueError('Original model save/reload differs')
                outputs[role] = {'version': VERSION, 'plan': role_plan, 'raw_hash': fingerprint(raw),
                    'model_digest': model.digest, 'numerical_model': numerical_model(model),
                    'training': model.receipt['training'], 'rows': rows, 'formal_acceptance': False}
            output = {'plan': plan, 'roles': outputs, 'raw_hash': fingerprint(raw)}
            _write(folder/'output.json', output)
            budget.finalize()
            actual = ReservedChannelBudget.verify_counts(authority, record['reservation_id'])
            if actual['test_only'] or not actual['finalized'] or actual['actual_counts'] != COUNTS:
                raise ValueError('Original complete paid two-solve accounting required')
            last = _kernel_snapshot(); _validate_kernel(first, last); _member(last, _identity())
            observation = _observation(lease)
            completed = authority.complete(record['reservation_id'], result={
                'nominal_output': fingerprint(output), 'raw_hash': fingerprint(raw),
                'models_hash': fingerprint({r: outputs[r]['model_digest'] for r in ROLES})}, actual_counts=actual['actual_counts'])
            _write(folder/'terminal.json', {'status': 'completed', 'reservation': completed,
                'kernel_before': first, 'kernel_after': last, 'observation': observation,
                'output_hash': fingerprint(output), 'formal_acceptance': False})
            # Missing successor is failure even if the debit completed. No readback
            # accepts terminal alone. Final I/O and lease observation are checked.
            final = _kernel_snapshot(); _validate_kernel(last, final); _member(final, _identity())
            _write(folder/'publication.json', {'terminal_hash': file_hash(folder/'terminal.json'),
                'kernel': final, 'observation': _observation(lease)})
            lease.assert_parent_alive()
            return {r: read_month(protocol, authority=authority, atom=atom, cutoff=cutoff, role=r) for r in ROLES}
        except BaseException as error:
            try:
                _write(folder/'failure.json', {'reason': type(error).__name__+':'+str(error)[:2000],
                    'fit_slot_consumed': True, 'retry': False, 'formal_acceptance': False})
            finally:
                if record is not None and authority.get(record['reservation_id'])['status'] in ('reserved', 'running'):
                    authority.fail_consumed(record['reservation_id'], str(error)[:1000])
            raise


def read_month(protocol, *, authority, atom, cutoff, role):
    """Original paid record, full kernel witnesses and immutable raw artifacts."""
    from .reserved_work_budget import ReservedChannelBudget
    from .work_lease import FrozenWorkLease
    from .neural_forward_admission import _kernel, _observation, _file_deadline
    role_plan = protocol.plan(atom, cutoff, role)
    plan = protocol.pair_plan(atom, cutoff)
    folder = protocol.directory/'fits'/fingerprint(plan)
    if (folder/'failure.json').exists():
        raise ValueError('Failed nominal slot remains in full denominator')
    intent, output, terminal, published = (_read(folder/f'{name}.json') for name in
                                           ('intent', 'output', 'terminal', 'publication'))
    authority.verify()
    record = authority.get(terminal['reservation']['reservation_id'])
    lease = FrozenWorkLease.from_dict(intent['lease'])
    validate_nominal_route(authority, intent['work_key'], intent['binding'], lease, COUNTS)
    actual = ReservedChannelBudget.verify_counts(authority, record['reservation_id'])
    raw = _read(folder/'raw.json')
    combined = output
    output = combined['roles'][role]
    model = (KnownLinearPredictor if role == 'reference' else JointPredictor).load(folder/(role+'-model.json'))
    if (terminal['status'] != 'completed' or record != terminal['reservation'] or record['status'] != 'completed'
            or record['work_key'] != intent['work_key'] or output['plan'] != role_plan or combined['plan'] != plan
            or actual['actual_counts'] != COUNTS or actual['test_only'] or not actual['finalized']
            or output['raw_hash'] != fingerprint(raw) or output['model_digest'] != model.digest
            or output['numerical_model'] != numerical_model(model)
            or output['training'] != model.receipt['training']
            or output['training']['cutoff'] != cutoff or output['training']['train_start'] != role_plan['train_start']
            or terminal['output_hash'] != fingerprint(combined)
            or record['result'] != {'nominal_output': fingerprint(combined), 'raw_hash': fingerprint(raw),
                'models_hash': fingerprint({r: combined['roles'][r]['model_digest'] for r in ROLES})}
            or published['terminal_hash'] != file_hash(folder/'terminal.json')):
        raise ValueError('Nominal original artifact/accounting chain differs')
    parent = record['process_identity']
    _kernel(terminal['kernel_before'], terminal['kernel_before'], terminal['kernel_after'], lease, parent, parent)
    _kernel(terminal['kernel_before'], terminal['kernel_after'], published['kernel'], lease, parent, parent)
    _observation(terminal['observation'], lease)
    _observation(published['observation'], lease, previous=terminal['observation'])
    _file_deadline([folder/f'{n}.json' for n in ('raw', 'reference-model', 'candidate-model', 'output', 'terminal', 'publication')], lease)
    return {**output, 'paid_pair_evidence': {'directory': str(folder), 'reservation_id': record['reservation_id'],
        'record_hash': fingerprint(record), 'output_sha256': file_hash(folder/'output.json'),
        'publication_sha256': file_hash(folder/'publication.json'), 'raw_sha256': file_hash(folder/'raw.json')}}


def _score(rows, schedule, symbols):
    expected = [(t.isoformat(), s) for t in pd.date_range(schedule['start'], schedule['end'], freq='h', inclusive='left')
                for s in symbols]
    if sorted((r['hour'], r['symbol']) for r in rows) != sorted(expected):
        raise ValueError('Full nominal hour/symbol denominator missing or duplicated')
    base = [r['reference_loss'] for r in rows]; cand = [r['candidate_loss'] for r in rows]
    hourly = {}
    for r in rows:
        hourly.setdefault(r['hour'], []).append((r['reference_loss'], r['candidate_loss']))
    means = [(math.fsum(a for a, _ in v)/len(v), math.fsum(b for _, b in v)/len(v)) for v in hourly.values()]
    risks = {'reference': math.fsum(a for a, _ in means)/len(means),
             'candidate': math.fsum(b for _, b in means)/len(means)}
    score = paired_effect_interval([r['hour'] for r in rows], base, cand,
        symbols=[r['symbol'] for r in rows], expected_symbols=symbols,
        eligible_count=schedule['hours'], rule=ConfirmationRule(), calibration_passed=False)
    return {'path_risks': risks, 'statistic': score,
            'numerical_event': bool(score.get('interval') and score['interval'][0] > 0)}


def assemble_atom(protocol, *, authority, atom):
    """Fourteen fresh cold fits, seven original paid paired reservations."""
    p = protocol.verify(); streams = {}; artifacts = {}
    for role in ROLES:
        records = [read_month(protocol, authority=authority, atom=atom, cutoff=c, role=role)
                   for c in p['schedule']['cutoffs']]
        rows = [r for record in records for r in record['rows']]
        if len({(r['origin'], r['symbol']) for r in rows}) != len(rows):
            raise ValueError('Duplicate nominal forecast')
        streams[role] = {(r['origin'], r['symbol']): r for r in rows}
        artifacts[role] = records
    if set(streams['reference']) != set(streams['candidate']):
        raise ValueError('Nominal paired support differs')
    pairs = []
    for key in sorted(streams['reference']):
        ref, cand = (streams[role][key] for role in ROLES)
        if any(ref[k] != cand[k] for k in ('label', 'window', 'maturity', 'feature_open_time')):
            raise ValueError('Nominal labels/windows differ')
        pairs.append({'hour': ref['hour'], 'symbol': ref['symbol'], 'label': ref['label'],
            'reference_loss': _finite_loss(ref['label'], ref['prediction']),
            'candidate_loss': _finite_loss(cand['label'], cand['prediction'])})
    return {'protocol': protocol.digest, 'atom': atom, 'streams': streams, 'artifacts': artifacts,
            'pairs': pairs, **_score(pairs, p['schedule'], p['law']['symbols']), 'formal_acceptance': False}


def compare_forecast(nominal, forecast, *, origin, cutoff, activation, max_lag):
    """Pure exact guard component; never authenticates caller-created records."""
    origin, issued = stamp(origin), stamp(forecast['issued_at'])
    if (origin.isoformat() != nominal['origin'] or cutoff != nominal['cutoff']
            or stamp(activation) > origin or not origin <= issued < stamp(nominal['window'])
            or (issued-origin).total_seconds() > max_lag
            or stamp(forecast['received_at']) > origin
            or forecast['forecast_window_start'] != nominal['window']
            or type(forecast['value']) not in (int, float)
            or not math.isfinite(forecast['value']) or forecast['value'] != nominal['prediction']):
        raise ValueError('Actual prediction, selected cold model, window or timeliness differs')


def _raw_conformance(directory, nominal_raw, start, end):
    """Every available revision in the input span must retain identical OHLCV.

    Authentic receipt/import/publication dates are read, never replaced by the
    nominal availability clock. Original feed verification checks object hashes.
    """
    import sqlite3
    from contextlib import closing
    from .v7_feed import ReceiptArchive
    from .calibration_oracle_evidence import _read as read_plain
    manifest = read_plain(directory/'forward.json')['identity']
    feed_directory = project_path(manifest['feed_directory'])
    identity = read_plain(feed_directory/'identity.json')
    if identity != {'source_id': manifest['lifecycle']['source_id'], 'symbols': manifest['config']['symbols']}:
        raise ValueError('Actual raw feed identity differs')
    feed = ReceiptArchive.__new__(ReceiptArchive)
    feed.directory, feed.identity, feed.symbols = feed_directory, identity['source_id'], tuple(identity['symbols'])
    feed.clock = lambda: pd.Timestamp.now(tz='UTC')
    expected = {(r['symbol'], r['open_time']): r for r in nominal_raw['rows']
                if stamp(start) <= stamp(r['open_time']) < stamp(end)}
    found = set(); versions = []; cache = {}
    with closing(sqlite3.connect((feed_directory/'receipts.sqlite').as_uri()+'?mode=ro', uri=True)) as db:
        feed.db = db; db.execute('PRAGMA query_only=ON'); db.execute('BEGIN'); feed.verify()
        for text, version, batch, symbol, opened in db.execute(
                'SELECT body,version,batch,symbol,opened FROM bars WHERE opened>=? AND opened<? ORDER BY opened,symbol,rowid',
                (start, end)):
            body = json.loads(text)
            if fingerprint(body) != version or body['ohlcv'] != expected.get((symbol, opened)):
                raise ValueError('Actual original raw revision differs from nominal input')
            published = feed._verify_bar_source(batch, body, cache=cache)
            if published is None:
                continue
            # This only checks physical provenance. Forecast and settlement
            # readers separately enforce each original as-of visibility.
            found.add((symbol, opened)); versions.append({'batch': batch, 'version': version, 'published': str(published)})
    if found != set(expected):
        raise ValueError('Actual warmup/train/forecast raw span is incomplete')
    return {'raw_ohlcv_hash': fingerprint([expected[k] for k in sorted(expected)]),
            'actual_provenance_hash': fingerprint(versions), 'revisions': len(versions)}


def observe_actual(protocol, *, authority, atom, actual_directory):
    """Read-only conformance to full original actual forecast/settlement paths.

    A matching result is a necessary library guard, not permission to register a
    draw or accept any of five claims. A future nominal trial registry must bind
    the actual directory to the independently sampled atom BEFORE observation.
    No old joint permit, nominal result or caller bool supplies that binding.
    """
    from contextlib import closing
    import sqlite3
    from .calibration_oracle_evidence import (OracleClaimSpec, reconstruct_claim,
        verify_raw_claim, _lifecycle, _read as read_plain)
    from .claim_protocol import ClaimProtocol
    outcome = {'version': VERSION, 'estimand': ESTIMAND, 'atom': atom,
        'trial_denominator': 1, 'formal_acceptance': False, 'pending_slots': PENDING,
        'outer_draw_registration': 'required_separate_new_nominal_registry',
        'accepted_event': False, 'conformant': False,
        'conservative_error_count': 1, 'power_success_count': 0}
    try:
        if type(protocol) is not NominalColdProtocol or type(authority) is not MonthlyWorkAuthority:
            raise TypeError('Original nominal protocol and paid authority required')
        p = protocol.verify(); outcome['protocol'] = protocol.digest
        nominal = assemble_atom(protocol, authority=authority, atom=atom)
        directory = project_path(actual_directory)
        manifest = read_plain(directory/'forward.json')
        if (set(manifest) != {'identity', 'sha256'} or fingerprint(manifest['identity']) != manifest['sha256']
                or manifest['identity']['source_files'] != p['source_files']
                or manifest['identity']['rolling'] != p['spec']):
            raise ValueError('Actual source/rolling identity differs from new nominal version')
        endpoint = p['endpoint']
        spec = OracleClaimSpec(**{k: endpoint[k] for k in
            ('slot', 'start', 'end', 'reference_stream', 'candidate_stream', 'reference', 'candidate')})
        actual = reconstruct_claim(directory, spec)
        verify_raw_claim(directory, actual)
        with _lifecycle(directory) as lc:
            lc.verify()
            claims = ClaimProtocol(lc)
            if (claims.endpoints['library_skill'].to_dict() != endpoint
                    or claims.manifest['source_files'] != p['source_files']
                    or stamp(claims.manifest['frozen_at']) >= stamp(p['schedule']['cutoffs'][0])):
                raise ValueError('Actual endpoint/source/ready rule not prospectively fixed')
            # The old quarantine estimand is expressly not this new contract.
            from .quarantined_observation import QuarantinedObservation
            observed = QuarantinedObservation.from_lifecycle(lc)
            if observed is not None:
                from .nominal_trial_controller import NominalObservation
                if type(observed) is not NominalObservation:
                    raise ValueError('Old joint quarantine cannot authorize new nominal conformance')
                observed.verify_actual_target(protocol, authority, atom, directory)
                observed.inspect(lc)
            models = {}; operations = {fingerprint(json.loads(t)): json.loads(t)
                for t, in lc.db.execute('SELECT record FROM operations')}
            expected_pairs = {(r['hour'], r['symbol']): r for r in nominal['pairs']}
            if len(actual['pairs']) != len(expected_pairs):
                raise ValueError('All nominal symbol-hours required; missing paths are failures')
            seen = set()
            for pair in actual['pairs']:
                key = (pair['hour'], pair['symbol'])
                expected = expected_pairs.get(key)
                if key in seen or expected is None or any(pair[k] != expected[k] for k in expected):
                    raise ValueError('Actual label/loss/support differs from full nominal tensor')
                seen.add(key)
                for role in ROLES:
                    evidence = pair[role+'_evidence']; stream = endpoint[role+'_stream']
                    row = lc.forecasts.db.execute('SELECT prediction FROM forecasts WHERE run=? AND stream=? '
                        'AND origin=? AND symbol=? AND task=?',
                        (lc.plan.digest, stream, evidence['origin'], pair['symbol'], 'volatility')).fetchone()
                    if row is None:
                        raise ValueError('Actual forecast missing')
                    forecast = json.loads(row[0])
                    if fingerprint(forecast) != evidence['forecast_hash']:
                        raise ValueError('Actual forecast hash changed')
                    nominal_row = nominal['streams'][role][(evidence['origin'], pair['symbol'])]
                    activation = operations[evidence['activation_hash']]
                    if activation['kind'] != 'activated':
                        raise ValueError('Original activation operation required')
                    job = lc.db.execute('SELECT cutoff,model,pipeline FROM jobs WHERE id=?',
                                        (activation['detail']['job'],)).fetchone()
                    if job is None or job[1] != forecast['model_id'] or json.loads(job[2]) != endpoint[role]:
                        raise ValueError('Actual selected job/pipeline changed')
                    if forecast['model_id'] not in models:
                        model, identity = lc.registry.load(forecast['model_id'], expected_pipeline=endpoint[role])
                        if identity['version'] != (3 if role == 'reference' else 4):
                            raise ValueError('Original source-bound paid model admission required')
                        models[forecast['model_id']] = model
                    model = models[forecast['model_id']]
                    reference = next(r for r in nominal['artifacts'][role] if r['plan']['cutoff'] == job[0])
                    if (numerical_model(model) != reference['numerical_model']
                            or model.receipt['training']['cutoff'] != nominal_row['cutoff']
                            or model.receipt['training']['train_start'] != nominal_row['train_start']
                            or model.receipt['training']['rows'] != reference['training']['rows']
                            or activation['detail']['origin'] != p['schedule']['ready'][role][job[0]]):
                        raise ValueError('Cold numerical model/train support/fixed activation schedule differs')
                    compare_forecast(nominal_row, forecast, origin=evidence['origin'], cutoff=job[0],
                        activation=activation['detail']['origin'], max_lag=lc.plan.prediction_max_lag_seconds)
            score = _score(actual['pairs'], p['schedule'], p['law']['symbols'])
            if score != {k: nominal[k] for k in score}:
                raise ValueError('Original statistics/support/RNG output differ')
            # Authenticate committed original endpoint inputs and statistic.
            with closing(sqlite3.connect((claims.directory/'claims.sqlite').as_uri()+'?mode=ro', uri=True)) as db:
                db.execute('PRAGMA query_only=ON')
                stored = db.execute('SELECT score,hash FROM claims WHERE slot=?', ('library_skill',)).fetchone()
            if stored is None or stored[0] is None:
                raise ValueError('Original single-look statistic not committed')
            saved = json.loads(stored[0]); inputs_path = claims.directory/'library_skill'/'inputs.json'
            inputs = read_plain(inputs_path)
            wanted_pairs = [{k: r[k] for k in ('hour', 'symbol', 'reference_loss', 'candidate_loss', 'label', 'revision')}
                            for r in actual['pairs']]
            if (fingerprint(saved) != stored[1] or saved['inputs_sha256'] != file_hash(inputs_path)
                    or inputs['pairs'] != wanted_pairs or inputs['endpoint'] != endpoint
                    or inputs['nominal_hours'] != p['schedule']['hours']
                    or inputs['nominal_symbol_hours'] != len(expected_pairs)):
                raise ValueError('Committed original statistical inputs differ')
            expected_statistic = paired_effect_interval([r['hour'] for r in actual['pairs']],
                [r['reference_loss'] for r in actual['pairs']], [r['candidate_loss'] for r in actual['pairs']],
                symbols=[r['symbol'] for r in actual['pairs']], expected_symbols=p['law']['symbols'],
                eligible_count=p['schedule']['hours'], rule=ConfirmationRule(), calibration_passed=False)
            # This candidate admits only diagnostic original claims. Future full
            # nominal acceptance needs its own audited outer registry integration.
            if saved['result'] != {**expected_statistic, 'formal_eligible': False} or saved['formal_eligible']:
                raise ValueError('Original statistic/seed or unsupported formal identity differs')
        first_plan = protocol.pair_plan(atom, p['schedule']['cutoffs'][0])
        raw = _read(protocol.directory/'fits'/fingerprint(first_plan)/'raw.json')
        provenance = _raw_conformance(directory, raw, p['schedule']['raw_start'],
                                      (stamp(p['schedule']['end'])+pd.Timedelta(hours=13)).isoformat())
        event = nominal['numerical_event']
        return {**outcome, 'conformant': True, 'nominal_numerical_event': event,
            'guarded_numerical_event': event, 'actual_raw_provenance': provenance,
            'conservative_error_count': None, 'power_success_count': None,
            'truth_classification': 'requires_complete_cold_census_and_registered_law',
            'tensor_hash': fingerprint(nominal['pairs']), 'statistic': nominal['statistic']}
    except Exception as error:
        return {**outcome, 'failure': type(error).__name__+':'+str(error)[:2000],
                'guarded_numerical_event': False}

