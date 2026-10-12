"""Library boundary derived from all original paid full-cold nominal atoms.

This is a new nominal law, not the old joint-service truth. It does not issue
old trial permits, spend production money, redraw failed observations or grant
formal acceptance. The sampling registry/actual five-slot bridge is separate.
"""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

from factorlab.protocol import fingerprint, project_path
from .evidence import file_hash
from .monthly_work_authority import source_files
from .nominal_boundary_math import derive_boundary, exact_payload
from .v7_storage import immutable_record

VERSION = 'v7-paid-nominal-library-boundary-law-v1'
RELATIVE = 'factorlab/discovery/nominal_cold_law.py'
ESTIMAND = 'nominal_full_cold_rolling'
GROUPS = [list(range(64)), list(range(64, 128))]


def _installed():
    from . import nominal_cold_conformance as producer
    files = source_files()
    for module, relative in ((producer, 'factorlab/discovery/nominal_cold_conformance.py'),
                             (__import__(__package__+'.nominal_boundary_math', fromlist=['*']),
                              'factorlab/discovery/nominal_boundary_math.py')):
        if (Path(module.__file__).resolve() != project_path(relative).resolve()
                or files.get(relative) != file_hash(module.__file__)):
            raise ValueError('Reviewed canonical nominal producer/arithmetic required')
    if Path(__file__).resolve() != project_path(RELATIVE).resolve() or files.get(RELATIVE) != file_hash(__file__):
        raise ValueError('New canonical integrated source required; staged law cannot execute')
    return files


def _write(path, value):
    immutable_record(path, {'record': value, 'hash': fingerprint(value)})


def _read(path):
    path = project_path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2**20:
        raise ValueError('Original bounded immutable law evidence required')
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError('Duplicate boundary JSON key')
            out[key] = value
        return out
    value = json.loads(path.read_text('utf8'), object_pairs_hook=unique,
                       parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    if set(value) != {'record', 'hash'} or fingerprint(value['record']) != value['hash']:
        raise ValueError('Original law hash/schema differs')
    return value['record']


def _original_atom(census, atom):
    """Fixed actual producer call; never an injected evaluator or risk vector."""
    result = census.assemble_atom(atom)
    risks = result['path_risks']
    if set(risks) != {'reference', 'candidate'}:
        raise ValueError('Original two-stream risk schema differs')
    rows = result['pairs']
    if type(rows) is not list or not rows:
        raise ValueError('No original complete nominal losses')
    evidence = {}
    for role in ('reference', 'candidate'):
        outputs = result['artifacts'][role]
        if type(outputs) is not list or len(outputs) != 7:
            raise ValueError('Every stream needs all seven original monthly cold models')
        evidence[role] = []
        for output in outputs:
            paid = output['paid_pair_evidence']
            if type(paid) is not dict or not paid:
                raise ValueError('Original paid pair provenance required')
            evidence[role].append({'plan_hash': fingerprint(output['plan']),
                'model_digest': output['model_digest'], 'raw_hash': output['raw_hash'],
                'paid_pair_evidence': paid, 'original_role_output_hash': fingerprint(output)})
    paired = [evidence['reference'][i]['paid_pair_evidence'] for i in range(7)]
    if (any(evidence['candidate'][i]['paid_pair_evidence'] != paired[i] for i in range(7))
            or len({p['reservation_id'] for p in paired}) != 7
            or len({p['directory'] for p in paired}) != 7):
        raise ValueError('Seven distinct original paid pairs must cover both complete streams')
    # No tuple-key stream dictionary is serialized. Only the public JSON rows
    # and original paid outputs are hashed. Release bulky atom records promptly.
    return {'atom': atom, 'path_risks': risks, 'original_pairs_hash': fingerprint(rows),
            'nominal_pairs_count': len(rows), 'paid_outputs': evidence}


def _census_binding(census):
    from .nominal_cold_conformance import NominalCensus
    if type(census) is not NominalCensus:
        raise TypeError('The original new-version typed census registry is required')
    p = census.verify()
    if p['source_files'] != _installed():
        raise ValueError('Original nominal census uses another frozen source')
    return {'directory': str(census.directory), 'digest': census.digest,
            'source_files': p['source_files'], 'generator_digest': p['generator_digest'],
            'law': p['law'], 'schedule': p['schedule'], 'profile_scope': p['profile_scope']}


class ColdBoundaryLaw:
    @classmethod
    def create(cls, directory, *, census):
        """One all-or-fail census aggregation, retaining failed allocations.

        The fixed arithmetic/group rule is part of canonical source before
        reading any atom. No caller weights, delta, group search or fitted
        model shortcuts are accepted. An unavailable boundary stays unavailable.
        """
        binding = _census_binding(census)
        directory = project_path(directory)
        if directory.is_symlink() or directory.exists() or not directory.parent.is_dir():
            raise ValueError('One fresh canonical allocation with existing parent required')
        directory.mkdir(exist_ok=False)
        intent = {'version': VERSION, 'estimand': ESTIMAND, 'directory': str(directory),
            'census': binding, 'groups': GROUPS, 'delta': [1, 200], 'support_size': 128,
            'claim_slot': 'library_skill', 'pending_slots': ['factor1', 'factor2', 'factor3', 'algorithm'],
            'retry_allowed': False, 'formal_acceptance': False}
        _write(directory/'intent.json', intent)
        risks = []
        records = []
        try:
            for atom in range(128):
                record = _original_atom(census, atom)
                target = directory/f'atom-{atom:03d}.json'
                _write(target, record)
                risks.append((record['path_risks']['reference'], record['path_risks']['candidate']))
                records.append({'atom': atom, 'sha256': file_hash(target)})
                del record
            law = exact_payload(derive_boundary(tuple(risks)))
            if _census_binding(census) != binding:
                raise ValueError('Original census identity changed while reading')
            _write(directory/'law.json', {'intent_hash': fingerprint(intent), 'records': records, 'law': law,
                'status': 'EXACT_LIBRARY_NOMINAL_BOUNDARY_NOT_FORMAL_CALIBRATION',
                'conditional_sampling_law': 'successful_exact_sampler_draws_only',
                'finite_sampler_failures': 'must_remain_failed_in_full_registered_N',
                'actual_conformance': 'required_separate_original_trace_check',
                'formal_acceptance': False})
        except BaseException as error:
            _write(directory/'failure.json', {'reason': type(error).__name__+':'+str(error)[:2000],
                'aggregation_consumed': True, 'retry_allowed': False, 'formal_acceptance': False})
            raise
        return cls(directory)

    def __init__(self, directory):
        self.directory = project_path(directory)
        self.intent = _read(self.directory/'intent.json')
        self.value = _read(self.directory/'law.json')
        self.digest = fingerprint(self.value)

    def sampling_plan(self):
        """Fixed original sampler quota, after complete original law verification.

        This constructs no draw and reserves no trial. The new original trial
        registry must durably consume its own preregistered (cell,index) before
        invoking the one-shot entropy implementation with this plan.
        """
        from . import rational_sampling
        relative = 'factorlab/discovery/rational_sampling.py'
        files = _installed()
        if (Path(rational_sampling.__file__).resolve() != project_path(relative).resolve()
                or files.get(relative) != file_hash(rational_sampling.__file__)):
            raise ValueError('Original canonical exact sampler component required')
        weights = tuple(Fraction(n, d) for n, d in self.verify()['law']['weights'])
        return rational_sampling.SamplingPlan(weights, block_bits=4096, max_attempts=1024)

    def verify(self):
        """Reopen every original paid atom and independently rederive the law."""
        from .nominal_cold_conformance import NominalCensus
        if (self.directory/'failure.json').exists():
            raise ValueError('Failed aggregation is consumed, not replaceable')
        intent = _read(self.directory/'intent.json')
        value = _read(self.directory/'law.json')
        if (intent != self.intent or value != self.value or fingerprint(value) != self.digest
                or intent['version'] != VERSION or intent['estimand'] != ESTIMAND
                or intent['directory'] != str(self.directory) or intent['groups'] != GROUPS
                or intent['delta'] != [1, 200] or intent['support_size'] != 128
                or intent['claim_slot'] != 'library_skill' or intent['formal_acceptance'] is not False
                or intent['pending_slots'] != ['factor1', 'factor2', 'factor3', 'algorithm']
                or intent['retry_allowed'] is not False
                or value['intent_hash'] != fingerprint(intent) or value['formal_acceptance'] is not False):
            raise ValueError('Original boundary identity was changed')
        if (value['status'] != 'EXACT_LIBRARY_NOMINAL_BOUNDARY_NOT_FORMAL_CALIBRATION'
                or value['conditional_sampling_law'] != 'successful_exact_sampler_draws_only'
                or value['finite_sampler_failures'] != 'must_remain_failed_in_full_registered_N'
                or value['actual_conformance'] != 'required_separate_original_trace_check'):
            raise ValueError('Boundary evidence scope/failure semantics were changed')
        census = NominalCensus(intent['census']['directory'])
        if _census_binding(census) != intent['census']:
            raise ValueError('Original source/paid census was changed')
        risks, records = [], []
        for atom in range(128):
            target = self.directory/f'atom-{atom:03d}.json'
            record = _read(target)
            if record != _original_atom(census, atom):
                raise ValueError('Boundary atom does not match original paid full-cold artifacts')
            risks.append((record['path_risks']['reference'], record['path_risks']['candidate']))
            records.append({'atom': atom, 'sha256': file_hash(target)})
            del record
        if records != value['records'] or exact_payload(derive_boundary(tuple(risks))) != value['law']:
            raise ValueError('Exact boundary law or entire ordered atom record differs')
        if _census_binding(census) != intent['census']:
            raise ValueError('Source/census changed during verification')
        return value
