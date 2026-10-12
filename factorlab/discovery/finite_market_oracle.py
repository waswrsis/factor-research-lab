"""Frozen finite-support market law and exact census risk reconstruction.

The law is deliberately a registered synthetic population, not a market claim.
Validation draws and oracle census use the SAME finite population; design uses
another population. No clipping of predictions, targets or logMSE is introduced.
Opaque adaptive provider randomness is outside this finite census scope.
"""
from __future__ import annotations

import json
import math
from contextlib import closing
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from factorlab.discovery.calibration_acceptance import AcceptancePlan, OracleRiskEnvelope
from factorlab.discovery.evidence import file_hash
from factorlab.discovery.monthly_work_authority import source_files
from factorlab.discovery.neural_forward_ticket import runtime_identity
from factorlab.discovery.v7_storage import immutable_record
from factorlab.protocol import fingerprint, project_path

VERSION = 'v7-finite-market-oracle-v1'
LOSS_BOUND = 2**20
PROOF_FILES = ('factorlab/discovery/convex_eval.py', 'factorlab/discovery/known_linear_reference.py',
               'factorlab/discovery/joint_pool.py', 'factorlab/discovery/neural_forward.py',
               'factorlab/discovery/v7_forward.py', 'factorlab/discovery/v7_lifecycle.py')


def _plain(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate finite generator JSON key')
            result[key] = value
        return result
    def invalid(_):
        raise ValueError('Nonfinite generator JSON')
    if not path.is_file() or path.stat().st_size > 64*2**20:
        raise ValueError('Missing or oversized original generator artifact')
    value = json.loads(path.read_text('utf-8'), object_pairs_hook=pairs, parse_constant=invalid)
    if set(value) != {'record', 'hash'} or fingerprint(value['record']) != value['hash']:
        raise ValueError('Finite generator artifact integrity changed')
    return value['record']


def _write(path, value):
    immutable_record(path, {'record': value, 'hash': fingerprint(value)})


@dataclass(frozen=True)
class FiniteMarketLaw:
    start: str
    hours: int = 72
    support_size: int = 128
    symbols: tuple[str, ...] = ('BTCUSDT',)
    seed: int = 638921
    persistence: float = .97
    regime_switch_probability: float = .04
    gap_probability: float = .002
    common_weight: float = .8
    signal_amplitude: float = .01

    def __post_init__(self):
        start = pd.Timestamp(self.start)
        if (start.tzinfo is None or start.utcoffset().total_seconds() != 0 or start != start.floor('h')
                or type(self.hours) is not int or not 24 <= self.hours <= 24000
                or type(self.support_size) is not int or not 2 <= self.support_size <= 2048
                or self.support_size & (self.support_size-1)
                or type(self.seed) is not int or not 0 <= self.seed < 2**32
                or type(self.symbols) is not tuple or not 1 <= len(self.symbols) <= 3
                or len(set(self.symbols)) != len(self.symbols)
                or any(type(s) is not str or not s.isalnum() or not s.isupper() for s in self.symbols)):
            raise ValueError('Bounded UTC finite-support generator law required')
        for field, low, high in (('persistence', 0., .999), ('regime_switch_probability', 0., .5),
                                 ('gap_probability', 0., .1), ('common_weight', 0., 1.),
                                 ('signal_amplitude', 0., .05)):
            value = getattr(self, field)
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError('Invalid registered finite DGP parameter')
        object.__setattr__(self, 'start', start.isoformat())

    def to_dict(self):
        return _plain(asdict(self))

    @property
    def digest(self):
        return fingerprint({'version': VERSION, 'law': self.to_dict(),
                            'distribution': 'uniform_over_complete_registered_path_atoms',
                            'scope': 'finite_synthetic_DGP_only_not_unbounded_financial_tails'})

    @property
    def domains(self):
        return {role+'_digest': fingerprint({'law': self.digest, 'domain': role})
                for role in ('design', 'validation', 'oracle')}


def proof_terms():
    """Conservative mathematical bound for accepted binary64 original loss.

    Finite x>=0 gives x < 2**1024; epsilon=1e-8 > 2**-27.
    Hence -19 < ln(x+epsilon) < 718 since ln(2)<.7 (including
    a factor-two conservative headroom at the upper endpoint). The squared
    difference is below 737**2 < 2**20. Runtime verifies these ranges as well.
    This does not define a loss for overflow, missing predictions or failures.
    """
    return {'method': 'finite_exhaustive_risk_enclosure', 'target': 'volatility12h',
        'loss': '(log(y+1e-8)-log(p+1e-8))**2', 'loss_upper_bound': LOSS_BOUND,
        'accepted_domain': 'finite_nonnegative_binary64_y_and_p', 'log_interval': [-19, 718],
        'decoder_audit': 'known_and_pool_use_ridge_exp;NN_exp;lower_zero_only_no_upper_clip',
        'finite_DGP_close_bounds': [float(2**-16), float(2**16)],
        'failure_policy': 'any_undefined_registered_atom_prevents_risk_enclosure',
        'weights': 'uniform_whole_path_census;equal_symbol_within_hour_then_equal_hour',
        'rounding': 'exact_rational_average_of_original_binary64_path_risks_outward_float',
        'unsupported': ['unbounded_financial_DGP', 'opaque_adaptive_GPT_distribution',
                        'unregistered_learner_or_source', 'dropped_failed_atoms'],
        'source_files': {p: file_hash(project_path(p)) for p in PROOF_FILES}}


class FiniteGenerator:
    def __init__(self, directory, law, acceptance):
        if type(law) is not FiniteMarketLaw or type(acceptance) is not AcceptancePlan:
            raise TypeError('Original law and complete acceptance grid required')
        self.directory = project_path(directory)
        self.law, self.acceptance = law, acceptance
        self.identity = {'version': VERSION, 'directory': str(self.directory), 'law': law.to_dict(),
            'law_digest': law.digest, 'domains': law.domains, 'acceptance_digest': acceptance.digest,
            'acceptance': _plain(acceptance.to_dict()), 'proof': proof_terms(),
            'source_files': source_files(), 'runtime_identity': runtime_identity(),
            'implementation_hash': file_hash(Path(__file__)),
            'test_grid_only': law.support_size < 128, 'formal_gate_eligible': False}
        path = self.directory/'identity.json'
        if not path.exists() and self.directory.exists() and any(self.directory.iterdir()):
            raise ValueError('Original generator identity missing; cannot reset')
        self.directory.mkdir(parents=True, exist_ok=True)
        _write(path, self.identity)
        self.verify()

    @property
    def digest(self):
        return fingerprint(self.identity)

    def verify(self):
        if (_read(self.directory/'identity.json') != self.identity
                or self.identity['source_files'] != source_files()
                or self.identity['implementation_hash'] != file_hash(Path(__file__))
                or self.identity['runtime_identity'] != runtime_identity()
                or self.identity['proof'] != proof_terms()
                or self.identity['law'] != self.law.to_dict()
                or self.identity['acceptance_digest'] != self.acceptance.digest):
            raise ValueError('Frozen generator/source/proof/grid changed')
        return {'generator_digest': self.digest, 'support_size': self.law.support_size,
                'formal_gate_eligible': False}

    def verify_registry(self, registry):
        from factorlab.discovery.full_calibration_registry import FullTrialRegistry
        if type(registry) is not FullTrialRegistry:
            raise TypeError('Actual full registry required')
        self.verify(); registry.verify()
        domains = self.law.domains
        mechanism = getattr(registry.plan, 'mechanism_contract', None)
        if mechanism is not None:
            from .full_mechanism_contract import mechanism_domains
            if mechanism['generators'].get(self.law.digest) != {
                    'directory': str(self.directory), 'digest': self.digest}:
                raise ValueError('Finite law is not an original registered joint-mechanism source')
            domains = mechanism_domains(mechanism)
        if (registry.plan.acceptance.digest != self.acceptance.digest
                or registry.plan.generator_design != domains
                or registry.plan.oracle_paths != self.law.support_size):
            raise ValueError('Generator law differs from whole original trial grid')

    def atom_for(self, domain, draw_identity):
        if domain not in ('design', 'validation', 'oracle'):
            raise ValueError('Unknown original generator domain')
        # Mask is unbiased for a uniformly distributed domain-separated hash;
        # cryptographic pseudorandomness is NOT certified provider independence.
        return int(fingerprint({'domain': self.law.domains[domain+'_digest'],
                                'draw': draw_identity}), 16) & (self.law.support_size-1)

    def reconstruct_atom(self, atom, *, population='evaluation'):
        self.verify()
        if type(atom) is not int or not 0 <= atom < self.law.support_size or population not in ('design', 'evaluation'):
            raise ValueError('Atom outside registered finite support')
        # Validation and oracle must share one distribution, not disjoint laws.
        seed = int(fingerprint({'law': self.law.digest, 'population': population, 'atom': atom}), 16)
        rng = np.random.Generator(np.random.PCG64(seed))
        states = np.zeros(len(self.law.symbols)); prior = np.full(len(states), math.exp(4.))
        regime, rows, schedule = 0, [], []
        for hour in range(self.law.hours):
            if rng.random() < self.law.regime_switch_probability:
                regime = 1-regime
            common = int(rng.integers(-2, 3))
            gap = bool(rng.random() < self.law.gap_probability)
            opened = pd.Timestamp(self.law.start)+pd.Timedelta(hours=hour)
            for i, symbol in enumerate(self.law.symbols):
                innovation = self.law.common_weight*common+(1-self.law.common_weight)*int(rng.integers(-2, 3))
                scale = .003 if regime == 0 else .012
                signal = self.law.signal_amplitude*math.sin(2*math.pi*hour/24)
                states[i] = max(-1., min(1., self.law.persistence*states[i]+scale*innovation+signal))
                price = math.exp(4.+states[i]); wick = .001*(1+int(rng.integers(0, 4)))
                row = {'symbol': symbol, 'open_time': opened.isoformat(), 'open': float(prior[i]),
                    'high': max(float(prior[i]), price)*math.exp(wick),
                    'low': min(float(prior[i]), price)*math.exp(-wick),
                    'close': price, 'volume': float(100+10*regime+abs(common))}
                if not 2**-16 <= row['close'] <= 2**16:
                    raise ArithmeticError('Generated close escaped registered proof support')
                rows.append(row); prior[i] = price
            schedule.append({'hour': hour, 'release_not_before': (opened+pd.Timedelta(hours=1, seconds=5)).isoformat(),
                             'gap': gap, 'regime': regime})
        return {'version': VERSION, 'generator_digest': self.digest, 'population': population,
                'atom': atom, 'law_digest': self.law.digest, 'rows': rows, 'schedule': schedule,
                'scope': 'synthetic_local_receipts_not_real_exchange_response', 'formal_gate_eligible': False}

    def produce_atom(self, atom, *, population='evaluation'):
        value = self.reconstruct_atom(atom, population=population)
        _write(self.directory/'atoms'/population/f'{atom:04d}.json', value)
        return value

    def read_atom(self, atom, *, population='evaluation'):
        value = self.reconstruct_atom(atom, population=population)
        if _read(self.directory/'atoms'/population/f'{atom:04d}.json') != value:
            raise ValueError('Original finite population atom changed')
        return value


def _finite_loss(y, p):
    if type(y) not in (int, float) or type(p) not in (int, float):
        raise ValueError('Original numeric target/prediction required')
    y, p = float(y), float(p)
    if not math.isfinite(y) or not math.isfinite(p) or min(y, p) < 0:
        raise ValueError('Undefined predictions cannot become bounded losses')
    a, b = float(np.log(y+1e-8)), float(np.log(p+1e-8))
    if not -19 < a < 718 or not -19 < b < 718:
        raise ValueError('Original log values outside audited machine-domain proof')
    value = float((np.log(y+1e-8)-np.log(p+1e-8))**2)
    if not 0 <= value <= LOSS_BOUND:
        raise ArithmeticError('Original logMSE escaped proof')
    return value


def _exact_interval(values):
    if not values or any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= LOSS_BOUND for v in values):
        raise ValueError('All original finite census risks required')
    mean = sum((Fraction.from_float(float(v)) for v in values), Fraction())/len(values)
    nearest = float(mean)
    return (max(0., math.nextafter(nearest, -math.inf)), math.nextafter(nearest, math.inf))


def _enclose_census(rows, *, generator, oracle_digest):
    """Private numerical kernel; public producer authenticates every original row."""
    generator.verify()
    n = generator.law.support_size
    if n < 128:
        return None
    if (len(rows) != n or sorted(r['atom'] for r in rows) != list(range(n))
            or any(set(r) != {'atom', 'reference', 'candidate', 'defined'} for r in rows)):
        raise ValueError('Exact original census required; no duplicate/dropped atoms')
    if any(r['defined'] is not True for r in rows):
        return None
    return OracleRiskEnvelope(_exact_interval([r['reference'] for r in rows]),
        _exact_interval([r['candidate'] for r in rows]), 'finite_exhaustive_risk_enclosure',
        oracle_digest, generator.acceptance.oracle_risk_alpha)


def _now():
    return pd.Timestamp.now(tz='UTC')


def _permit(generator, permit):
    from factorlab.discovery.full_calibration_registry import QuarantinedTrialPermit
    if type(generator) is not FiniteGenerator or type(permit) is not QuarantinedTrialPermit:
        raise TypeError('Actual frozen generator and original typed permit required')
    generator.verify_registry(permit.registry)
    permit.verify()
    if permit.trial['scenario_digest'] != generator.law.digest:
        raise ValueError('Finite generator cannot replace another registered scenario')
    return permit


def _draw_identity(generator, permit):
    role = permit.trial['role']
    if role == 'oracle':
        atom = permit.trial['oracle_index']  # complete fixed census, no resampling
    elif role == 'validation':
        atom = generator.atom_for('validation', permit.trial)
    else:
        raise ValueError('Unknown original trial draw role')
    return {'version': VERSION, 'generator_digest': generator.digest, 'trial': permit.trial,
            'registry_digest': permit.registry.digest, 'domain': role, 'population': 'evaluation',
            'atom': atom, 'feed_directory': str(permit.paths['feed']),
            'distribution': 'same_uniform_finite_population_for_validation_and_oracle',
            'scope': 'synthetic_local_receipts_not_exchange_observations'}


def _draw_paths(generator, permit):
    return (permit.paths['root']/'generator_draw.json',
            generator.directory/'draws'/(permit.trial['trial_id']+'.json'))


def bind_draw(generator, permit):
    from filelock import FileLock
    permit = _permit(generator, permit)
    paths = _draw_paths(generator, permit)
    expected = _draw_identity(generator, permit)
    with FileLock(str(paths[0])+'.binding.lock', timeout=0):
        if any(p.exists() for p in paths):
            if any(_read(p) != expected for p in paths):
                raise ValueError('Original finite draw binding missing or changed')
            return expected
        import sqlite3
        dbpath = permit.paths['feed']/'receipts.sqlite'
        with closing(sqlite3.connect(dbpath.as_uri()+'?mode=ro', uri=True)) as db, db:
            if db.execute('SELECT COUNT(*) FROM batches').fetchone()[0]:
                raise ValueError('Finite generator cannot enroll an already observed input path')
        for p in paths:
            _write(p, expected)
        generator.produce_atom(expected['atom'])
        return expected


def _read_draw_sources(generator, permit):
    """Capture freshly verified draw/atom; never stored as a later permission."""
    permit = _permit(generator, permit)
    paths = _draw_paths(generator, permit)
    value = _draw_identity(generator, permit)
    if any(_read(p) != value for p in paths):
        raise ValueError('Original generated path binding changed')
    atom = generator.read_atom(value['atom'])
    return value, atom


def _read_draw(generator, permit):
    value, _ = _read_draw_sources(generator, permit)
    return value


def _open_feed(permit):
    from factorlab.discovery.v7_feed import ReceiptArchive
    path = permit.paths['feed']
    if not (path/'identity.json').is_file() or not (path/'receipts.sqlite').is_file():
        raise ValueError('Original enrolled feed missing; no new source is allowed')
    identity = json.loads((path/'identity.json').read_text('utf-8'))
    # Private system-clock seam only for engineering tests; no public clock API.
    return ReceiptArchive(path, symbols=tuple(identity['symbols']), clock=_now)


def publish_due(generator, permit, *, max_new_receipts=1):
    """Publish actual local receipts now; never synthesize historical availability.

    An attempt marker is durable before feed mutation. Interrupted publication
    cannot retry under a fresh timestamp. It remains incomplete for the oracle.
    This generates no fitting grant and does not initialize a work wallet.
    """
    import time

    from filelock import FileLock
    if type(max_new_receipts) is not int or not 1 <= max_new_receipts <= 96:
        raise ValueError('At most 96 receipts per bounded generator invocation')
    permit = _permit(generator, permit)
    bind_draw(generator, permit)
    root = permit.paths['root']/'generator_publications'
    started, wall = time.monotonic(), _now()
    issued = []
    with FileLock(str(root)+'.lock', timeout=0):
        before, draw, atom = _verified_publication_sources(generator, permit)
        completed = set(before['completed_hours'])
        attempts = set(before['attempted_hours'])
        feed = _open_feed(permit)
        try:
            for item in atom['schedule']:
                if len(issued) >= max_new_receipts or time.monotonic()-started >= 25.:
                    break
                hour = item['hour']
                if item['gap'] or hour in completed:
                    continue
                if hour in attempts:
                    raise ValueError('Original generator attempt was interrupted; cannot renew it')
                now = _now()
                if now < wall:
                    raise ValueError('Generator wall clock moved backward')
                if now < pd.Timestamp(item['release_not_before']):
                    break
                rows = atom['rows'][hour*len(generator.law.symbols):(hour+1)*len(generator.law.symbols)]
                response_hash = fingerprint({'draw': draw, 'hour': hour, 'rows': rows})
                attempt = {'draw_hash': fingerprint(draw), 'hour': hour, 'response_hash': response_hash,
                           'received_at': now.isoformat(), 'release_not_before': item['release_not_before']}
                _write(root/f'{hour:05d}-attempt.json', attempt)
                _write(generator.directory/'publications'/permit.trial['trial_id']/f'{hour:05d}-attempt.json', attempt)
                frame = pd.DataFrame(rows)
                frame['open_time'] = pd.to_datetime(frame.open_time, utc=True)
                result = feed.receive(frame, received_at=now, response_hash=response_hash)
                value = {**attempt, 'receipt': result}
                _write(root/f'{hour:05d}-complete.json', value)
                _write(generator.directory/'publications'/permit.trial['trial_id']/f'{hour:05d}-complete.json', value)
                issued.append(value)
        finally:
            feed.close()
    return {'issued': issued, 'actual_received_at_not_backdated': True,
            'formal_gate_eligible': False, 'budget_granted': False}


def verify_publications(generator, permit):
    """Match original generator atoms to full raw batches and dual attempt anchors."""
    result, _, _ = _verified_publication_sources(generator, permit)
    return result


def _verified_publication_sources(generator, permit):
    """One invocation's full audit and captured source values, never a cache.

    The only entry always checks the actual permit, source, dual draw and atom.
    Callers cannot provide a purported prior verification or source snapshot.
    """
    import sqlite3

    from factorlab.discovery.v7_feed import ReceiptArchive
    draw, atom = _read_draw_sources(generator, permit)
    root = permit.paths['root']/'generator_publications'
    mirror = generator.directory/'publications'/permit.trial['trial_id']
    names = {p.name for p in root.glob('*.json')}
    if names != {p.name for p in mirror.glob('*.json')}:
        raise ValueError('Generator publication anchors missing or changed')
    attempts, completed, batch_ids = {}, {}, set()
    for name in names:
        value = _read(root/name)
        if value != _read(mirror/name):
            raise ValueError('Original generator publication differs from its mirror')
        hour = value['hour']
        if type(hour) is not int or not 0 <= hour < generator.law.hours or atom['schedule'][hour]['gap']:
            raise ValueError('Unregistered generator receipt hour')
        if name == f'{hour:05d}-attempt.json':
            attempts[hour] = value
        elif name == f'{hour:05d}-complete.json':
            completed[hour] = value
        else:
            raise ValueError('Invalid original receipt marker name')
    feed = ReceiptArchive.__new__(ReceiptArchive)
    feed.directory, feed.symbols = permit.paths['feed'], generator.law.symbols
    identity = json.loads((feed.directory/'identity.json').read_text('utf-8'))
    feed.identity, feed.clock = identity['source_id'], _now
    with closing(sqlite3.connect((feed.directory/'receipts.sqlite').as_uri()+'?mode=ro', uri=True)) as db, db:
        feed.db = db; db.execute('PRAGMA query_only=ON'); db.execute('BEGIN')
        feed.verify()
        # Operation-local SQL index: read each original batch once inside this
        # same read transaction. This replaces repeated full-table JSON scans,
        # not any raw-byte, publication, source, draw or whole-set validation.
        # Preserve lists so duplicate responses remain a hard failure.
        batches_by_response = {}
        original_batch_ids = set()
        for identifier, text in db.execute('SELECT id,record FROM batches'):
            batch = json.loads(text)
            original_batch_ids.add(identifier)
            reference = batch['source_ref']
            if type(reference) is dict and set(reference) == {'response_sha256'}:
                digest = reference['response_sha256']
                if type(digest) is str:
                    batches_by_response.setdefault(digest, []).append((identifier, batch))
        draw_hash = fingerprint(draw)
        for hour, attempt in attempts.items():
            rows = atom['rows'][hour*len(generator.law.symbols):(hour+1)*len(generator.law.symbols)]
            response_hash = fingerprint({'draw': draw, 'hour': hour, 'rows': rows})
            if (set(attempt) != {'draw_hash', 'hour', 'response_hash', 'received_at', 'release_not_before'}
                    or attempt['draw_hash'] != draw_hash or attempt['response_hash'] != response_hash
                    or attempt['release_not_before'] != atom['schedule'][hour]['release_not_before']
                    or pd.Timestamp(attempt['received_at']) < pd.Timestamp(attempt['release_not_before'])):
                raise ValueError('Generated receipt changed its original draw or time')
            matches = batches_by_response.get(response_hash, [])
            if len(matches) > 1:
                raise ValueError('A generated receipt attempt was replayed')
            if hour not in completed:
                # If feed mutation committed before publication mirror failed,
                # retain its identity but never pretend the whole attempt finished.
                batch_ids.update(identifier for identifier, _ in matches)
                continue
            value = completed[hour]
            if (set(value) != {*attempt, 'receipt'} or {k: value[k] for k in attempt} != attempt
                    or len(matches) != 1 or matches[0][0] != value['receipt']['batch']):
                raise ValueError('Completed generator publication lacks its original raw batch')
            identifier, batch = matches[0]
            body = json.loads((feed.directory/batch['bars_ref']['path']).read_text('utf-8'))
            if (body != rows or batch['actual'] is not True or batch['received_at'] != attempt['received_at']
                    or feed._publication(identifier, batch) is None):
                raise ValueError('Original raw bars differ from the registered market atom')
            batch_ids.add(identifier)
        if set(completed)-set(attempts):
            raise ValueError('Completion lost its prepaid publication attempt')
        if original_batch_ids != batch_ids:
            raise ValueError('Unregistered source rows or deleted publication anchors')
    required = {s['hour'] for s in atom['schedule'] if not s['gap']}
    return {'atom': draw['atom'], 'attempted_hours': sorted(attempts), 'completed_hours': sorted(completed),
            'all_registered_receipts_complete': set(completed) == required,
            'defined_population_scope': 'finite_synthetic_only', 'formal_gate_eligible': False}, draw, atom


def _reconstruct_finite_envelopes(generator, registry, cell, index):
    """Reconstruct actual whole-path census before issuing scoped risk intervals.

    It does NOT certify provider distribution, full acceptance, or market risk.
    All original cold-refit slots remain refused without a proof of the
    complete stochastic execution/receipt mapping. Even 128 observed completed
    traces do not establish that mapping. The separate fixed-model census has
    an intentionally narrower estimand and cannot supply these slots.
    """
    from factorlab.discovery.calibration_oracle_evidence import read_oracle_path
    from factorlab.discovery.full_calibration_registry import QuarantinedTrialPermit
    generator.verify_registry(registry)
    cell_plan = next((c for c in registry.plan.acceptance.cells if c.name == cell), None)
    if cell_plan is None or cell_plan.scenario_digest != generator.law.digest:
        raise ValueError('Finite generator cannot replace another registered scenario')
    slots = ('library_skill', 'factor1', 'factor2', 'factor3')
    rows, hashes = {s: [] for s in slots}, []
    missing, incomplete = [], []
    for atom in range(generator.law.support_size):
        permit = QuarantinedTrialPermit(registry, registry.identity_for(cell, index, oracle_index=atom))
        if not (permit.paths['root']/'oracle_path.json').is_file():
            missing.append(atom); continue
        receipts = verify_publications(generator, permit)
        value = read_oracle_path(permit)
        if receipts['atom'] != atom:
            raise ValueError('Oracle census changed its original atom order')
        hashes.append(fingerprint(value))
        if not receipts['all_registered_receipts_complete']:
            incomplete.append(atom)
        for claim in value['claims']:
            slot = claim['estimand']['slot']
            if slot not in slots:
                continue
            recipes = (claim['estimand']['reference'], claim['estimand']['candidate'])
            fixed = all(type(r) is dict and ('pool' in r or r.get('kind') == 'v7_known_linear') for r in recipes)
            risk = claim['path_risks']
            defined = bool(fixed and receipts['all_registered_receipts_complete']
                           and claim['status'] == 'observed' and risk is not None)
            rows[slot].append({'atom': atom, 'reference': risk['reference'] if risk else None,
                              'candidate': risk['candidate'] if risk else None, 'defined': defined})
    binding = {'generator': generator.digest, 'registry': registry.digest, 'cell': cell, 'index': index,
               'oracle_path_hashes': hashes, 'proof': generator.identity['proof']}
    # A completed observed trace is not proof of the execution mapping over
    # all possible scheduler/receipt/provider randomness. Do not promote these
    # descriptive means into the cold-refit learner's population truth.
    intervals = {slot: None for slot in slots}
    value = {'version': VERSION, 'binding': binding, 'risk_envelopes': intervals,
             'algorithm': {'risk_envelope': None, 'reason': 'opaque_adaptive_provider_randomness_not_enumerated'},
             'missing_atoms': missing, 'incomplete_atoms': incomplete,
             'formal_gate_eligible': False, 'full_V7': False,
             'scope': 'requested_original_cold_refit_scope_not_yet_proved',
             'refusal': 'complete_clock_receipt_and_stochastic_execution_mapping_not_proved',
             'required_additional_evidence': ['original_complete_cold_refit_transition_mapping',
                 'receipt_and_activation_support_mapping_for_every_atom',
                 'all_training_and_provider_randomness_measure_or_exhaustive_support',
                 'original_failures_and_costs_in_total_population_denominator'],
             'unsupported_original_scenarios_retained': ['unbounded_financial_tails', 'full_adaptive_GPT_A0'],
             'actual_oracle_paths': len(hashes), 'registered_oracle_paths': generator.law.support_size}
    return _plain(value)


def produce_finite_envelopes(generator, registry, cell, index):
    value = _reconstruct_finite_envelopes(generator, registry, cell, index)
    digest = fingerprint(value)
    _write(generator.directory/'envelopes'/f'{digest}.json', value)
    return {'digest': digest, 'record': value}


def verify_finite_envelopes(generator, registry, cell, index, *, artifact_digest):
    value = _reconstruct_finite_envelopes(generator, registry, cell, index)
    if (type(artifact_digest) is not str or fingerprint(value) != artifact_digest
            or _read(generator.directory/'envelopes'/f'{artifact_digest}.json') != value):
        raise ValueError('Finite risk proof differs from complete original census reconstruction')
    return value
