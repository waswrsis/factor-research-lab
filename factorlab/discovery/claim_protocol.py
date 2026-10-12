"""Frozen six-calendar-month library/algorithm endpoints, not calibration.

The existing ExposureLedger owns the same five slots as factor confirmation.
This module implements the other two real-stream, single-look endpoints. Until
the full G09--G11 producer exists, every numerical result remains diagnostic;
there is deliberately no caller-supplied formal/calibration switch.
"""
from __future__ import annotations

import copy
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field

import pandas as pd
from filelock import FileLock

from factorlab.discovery.confirmatory import ConfirmationRule, paired_effect_interval
from factorlab.discovery.evidence import file_hash, snapshot_source
from factorlab.discovery.monthly_work_authority import source_files
from factorlab.discovery.v7_lifecycle import Lifecycle, boundary, canonical, stamp
from factorlab.discovery.v7_storage import immutable_record
from factorlab.protocol import fingerprint

from .typed_policy_confirmation import POLICY as POLICY
from .typed_policy_confirmation import TRIAL_STREAM, policy_definition

SLOTS = ('library_skill', 'algorithm')


def _pipeline(value):
    result = json.loads(canonical(value))
    identifier = result.get('pipeline_id')
    if (not isinstance(identifier, str) or len(identifier) != 64
            or any(c not in '0123456789abcdef' for c in identifier) or len(result) < 2):
        raise ValueError('Claim needs a complete frozen pipeline, not a display name')
    return result


@dataclass(frozen=True)
class FrozenClaimEndpoint:
    slot: str
    start: str
    candidate: dict
    reference: dict
    candidate_stream: str
    reference_stream: str
    rule: ConfirmationRule = field(default_factory=ConfirmationRule)
    research_spec_hash: str | None = None
    _digest: str = field(init=False, repr=False)

    def __post_init__(self):
        if (self.slot not in SLOTS or not isinstance(self.rule, ConfirmationRule)
                or self.rule.horizon_hours != 12 or self.rule.bootstrap_samples > 10000):
            raise ValueError('Unknown typed claim or unsupported frozen rule')
        object.__setattr__(self, 'start', boundary(self.start).isoformat())
        object.__setattr__(self, 'reference', _pipeline(self.reference))
        candidate = json.loads(canonical(self.candidate))
        policy = policy_definition(self.reference, research_spec_hash=self.research_spec_hash)
        if self.slot == 'library_skill':
            if self.research_spec_hash is not None:
                raise ValueError('Only algorithm endpoints carry a research policy identity')
            candidate = _pipeline(candidate)
            if 'pool' not in candidate or candidate['pool'].get('members') != []:
                raise ValueError('Library skill is the frozen empty-pool K0 learner')
        elif (set(candidate) != {*policy, 'initial_pipeline'}
              or {k: candidate[k] for k in policy} != policy
              or _pipeline(candidate['initial_pipeline']) != self.reference
              or self.candidate_stream != TRIAL_STREAM):
            raise ValueError('Algorithm claim requires the complete registered A0 policy and K0 reference')
        object.__setattr__(self, 'candidate', candidate)
        if (any(not isinstance(s, str) or not s or len(s) > 128 for s in
                (self.candidate_stream, self.reference_stream))
                or self.candidate_stream == self.reference_stream):
            raise ValueError('Claims require distinct fixed real forecast streams')
        object.__setattr__(self, '_digest', fingerprint(self.to_dict()))

    @classmethod
    def library(cls, start, k0, reference, *, candidate_stream='K0', reference_stream='benchmark', rule=None):
        return cls('library_skill', start, k0, reference, candidate_stream, reference_stream,
                   rule or ConfirmationRule())

    @classmethod
    def algorithm(cls, start, k0, *, reference_stream='K0', rule=None, research_spec_hash=None):
        return cls('algorithm', start, {**policy_definition(k0, research_spec_hash=research_spec_hash), 'initial_pipeline': copy.deepcopy(k0)}, k0,
                   TRIAL_STREAM, reference_stream, rule or ConfirmationRule(), research_spec_hash)

    @property
    def end(self):
        return (stamp(self.start) + pd.DateOffset(months=6)).isoformat()

    @property
    def scoring_cutoff(self):
        return (stamp(self.end) + pd.Timedelta(hours=36)).isoformat()

    @property
    def candidate_id(self):
        return self.candidate['pipeline_id'] if self.slot == 'library_skill' else fingerprint(self.candidate)

    def to_dict(self):
        value = {'slot': self.slot, 'start': self.start, 'end': self.end, 'candidate': self.candidate,
                'reference': self.reference, 'candidate_stream': self.candidate_stream,
                'reference_stream': self.reference_stream, 'rule': self.rule.to_dict()}
        if self.research_spec_hash is not None:
            value['research_spec_hash'] = self.research_spec_hash
        return value

    @classmethod
    def from_dict(cls, value):
        value = copy.deepcopy(value)
        end = value.pop('end')
        value['rule'] = ConfirmationRule(**value['rule'])
        result = cls(**value)
        if result.end != end:
            raise ValueError('Claim nominal six-month endpoint changed')
        return result

    @property
    def digest(self):
        if fingerprint(self.to_dict()) != self._digest:
            raise ValueError('Frozen endpoint was mutated')
        return self._digest


def _read(path):
    return json.loads(path.read_text('utf-8'))


class ClaimProtocol:
    """Actual ledger execution, with conservative consumed-but-unscored recovery.

    Fixed endpoint scores are committed in SQLite before external publication.
    A committed score is republished, never recomputed. Interrupted reads without
    a committed score remain consumed. This does not manufacture a formal gate.
    """
    def __init__(self, lifecycle):
        self._load(lifecycle)
        self.verify()

    def _load(self, lifecycle):
        if not isinstance(lifecycle, Lifecycle):
            raise TypeError('Claims require the actual Lifecycle ledger')
        self.lifecycle = lifecycle
        self.directory = lifecycle.directory / 'claim_protocol'
        self.path = self.directory / 'claims.sqlite'
        self.manifest = _read(self.directory / 'protocol.json')
        self.digest = fingerprint(self.manifest)
        if not self.path.is_file():
            raise ValueError('Claim index missing; cannot reset consumed endpoints')
        self.endpoints = {v['slot']: FrozenClaimEndpoint.from_dict(v) for v in self.manifest['endpoints']}

    @staticmethod
    def _bindings(lifecycle):
        previous, bindings = '0' * 64, []
        for index, (sequence, text, digest) in enumerate(lifecycle.db.execute(
                'SELECT seq,record,hash FROM operations ORDER BY seq'), 1):
            event = json.loads(text)
            if (sequence != index or event['seq'] != index or event['previous'] != previous
                    or fingerprint(event) != digest):
                raise ValueError('Lifecycle claim binding chain changed')
            if event['kind'] == 'six_month_claim_protocol_bound':
                bindings.append(event['detail'])
            previous = digest
        return bindings

    def _bind(self):
        expected = {'protocol_digest': self.digest, 'directory': str(self.directory)}
        existing = self._bindings(self.lifecycle)
        if existing == [expected]:
            return
        if existing:
            raise ValueError('Lifecycle already bound another six-month claim protocol')
        with self.lifecycle.db:
            self.lifecycle.db.execute('BEGIN IMMEDIATE')
            if self._bindings(self.lifecycle):
                raise ValueError('Concurrent claim binding changed')
            self.lifecycle._event('six_month_claim_protocol_bound', expected)

    @classmethod
    def freeze(cls, lifecycle, endpoints):
        if not isinstance(lifecycle, Lifecycle):
            raise TypeError('Claims require an actual lifecycle')
        endpoints = tuple(endpoints)
        if (len(endpoints) != 2 or {e.slot for e in endpoints} != set(SLOTS)
                or len({e.start for e in endpoints}) != 1 or lifecycle.plan.algorithm_months != 6):
            raise ValueError('Freeze exactly the library and algorithm six-month endpoints together')
        for e in endpoints:
            _ = e.digest
        library = next(e for e in endpoints if e.slot == 'library_skill')
        algorithm = next(e for e in endpoints if e.slot == 'algorithm')
        from .layered_research import verified_research_spec
        research = verified_research_spec(lifecycle)
        if algorithm.research_spec_hash != (None if research is None else research.digest):
            raise ValueError('Claim freeze must match the original research enrollment')
        if library.candidate != algorithm.reference or library.candidate_stream != algorithm.reference_stream:
            raise ValueError('Algorithm reference must be the very same frozen K0 stream')
        directory = lifecycle.directory / 'claim_protocol'
        with FileLock(str(directory) + '.lock', timeout=0):
            if cls._bindings(lifecycle) and not directory.exists():
                raise ValueError('Bound claim directory missing; cannot reset or disable claims')
            if directory.exists():
                old = object.__new__(cls)
                old._load(lifecycle)
                old._verify(allow_unreserved=True)
                if [old.endpoints[s].to_dict() for s in SLOTS] != [next(e for e in endpoints if e.slot == s).to_dict() for s in SLOTS]:
                    raise ValueError('An endpoint cannot change or restart under another name')
                old._bind()
                old._reserve()
                old.verify()
                return old
            lifecycle.verify()
            if lifecycle.now() > stamp(library.start):
                raise ValueError('Six-month endpoints must freeze before any observation period')
            directory.mkdir()
            source = snapshot_source(directory / 'source')['files']
            value = {'version': 'v7-six-month-claims-v2', 'plan_id': lifecycle.plan.digest,
                     'lifecycle_directory': str(lifecycle.directory), 'exposure_path': str(lifecycle.exposure.path),
                     'source_id': lifecycle.plan.source_id, 'symbols': list(lifecycle.plan.symbols),
                     'frozen_at': lifecycle.now().isoformat(), 'source_files': source,
                     'endpoints': [next(e for e in endpoints if e.slot == s).to_dict() for s in SLOTS]}
            immutable_record(directory / 'protocol.json', value)
            with closing(sqlite3.connect(directory / 'claims.sqlite')) as db:
                db.execute('PRAGMA journal_mode=WAL'); db.execute('PRAGMA synchronous=FULL')
                db.execute('CREATE TABLE claims (slot TEXT PRIMARY KEY, protocol TEXT, score TEXT, hash TEXT)')
                db.executemany('INSERT INTO claims VALUES (?,?,NULL,NULL)',
                               [(s, fingerprint(value)) for s in SLOTS])
                db.commit()
            result = object.__new__(cls)
            result.lifecycle, result.directory = lifecycle, directory
            result.path, result.manifest, result.digest = directory / 'claims.sqlite', value, fingerprint(value)
            result.endpoints = {e.slot: e for e in endpoints}
            result._bind()
            result._reserve()
            result.verify()
            return result

    def _reservation_args(self, endpoint):
        return {'batch': 'typed:' + endpoint.slot + ':' + self.digest[:32],
                'source': self.manifest['source_id'], 'symbols': self.manifest['symbols'],
                'start': endpoint.start, 'end': endpoint.end, 'candidate_id': endpoint.candidate_id,
                'reference_id': endpoint.reference['pipeline_id'], 'rule': endpoint.rule.to_dict(),
                'frozen_at': self.manifest['frozen_at'], 'cohort': 1, 'claim_slot': endpoint.slot}

    def _reserve(self):
        # Existing exact reservations are idempotent; a missing second one can
        # only recover while still prospective. No new frozen_at or window.
        for slot in SLOTS:
            self.lifecycle.exposure.reserve_confirmation(**self._reservation_args(self.endpoints[slot]))

    def _row(self, slot):
        with closing(sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True)) as db:
            return db.execute('SELECT protocol,score,hash FROM claims WHERE slot=?', (slot,)).fetchone()

    def verify(self):
        return self._verify()

    def _verify(self, *, allow_unreserved=False):
        m, lc = self.manifest, self.lifecycle
        if (_read(self.directory / 'protocol.json') != m or fingerprint(m) != self.digest
                or m['version'] != 'v7-six-month-claims-v2' or m['plan_id'] != lc.plan.digest
                or m['lifecycle_directory'] != str(lc.directory) or m['exposure_path'] != str(lc.exposure.path)
                or m['source_id'] != lc.plan.source_id or m['symbols'] != list(lc.plan.symbols)
                or source_files() != m['source_files']):
            raise ValueError('Frozen claim source, lifecycle or protocol changed')
        if any(file_hash(self.directory / 'source' / p) != h for p, h in m['source_files'].items()):
            raise ValueError('Original claim source archive changed')
        from .layered_research import verified_research_spec
        research = verified_research_spec(lc)
        if self.endpoints['algorithm'].research_spec_hash != (None if research is None else research.digest):
            raise ValueError('Claim endpoint cannot reinterpret the actual research enrollment')
        expected_binding = {'protocol_digest': self.digest, 'directory': str(self.directory)}
        bindings = self._bindings(lc)
        if bindings != [expected_binding] and not (allow_unreserved and not bindings):
            raise ValueError('Original lifecycle claim binding is missing or changed')
        for slot in SLOTS:
            e = self.endpoints[slot]
            _ = e.digest
            row = self._row(slot)
            if row is None or row[0] != self.digest:
                raise ValueError('Claim index differs from its frozen protocol')
            args = self._reservation_args(e)
            found = lc.exposure.db.execute('SELECT identity,consumed,result FROM confirmations WHERE batch=?',
                                          (args['batch'],)).fetchone()
            expected = {**args, 'alpha': .01, 'scoring_cutoff': e.scoring_cutoff}
            expected['symbols'] = sorted(expected['symbols'])
            if found is None and allow_unreserved:
                if row[1:] != (None, None) or (self.directory / slot / 'read_started.json').exists():
                    raise ValueError('Consumed claim cannot recreate its missing reservation')
                continue
            if not found or json.loads(found[0]) != expected:
                raise ValueError('Actual five-slot reservation is missing or changed')
            if row[1] is not None:
                score = json.loads(row[1])
                if (fingerprint(score) != row[2] or not found[1]
                        or score['protocol'] != self.digest or score['slot'] != slot
                        or type(score.get('formal_eligible')) is not bool):
                    raise ValueError('Committed claim result changed or self-authorized')
                from .quarantine_execution import verify_claim_result
                verify_claim_result(self, score)
                if file_hash(self.directory / slot / 'inputs.json') != score['inputs_sha256']:
                    raise ValueError('Original paired-loss evidence changed')
                mirror = self.directory / slot / 'score.json'
                if mirror.exists() and _read(mirror) != score:
                    raise ValueError('Claim score differs from its committed mirror')
                if found[2] is not None and json.loads(found[2]) != score:
                    raise ValueError('Exposure result differs from its committed endpoint')
            elif row[2] is not None or found[2] is not None:
                raise ValueError('Published claim lacks its original score commit')
        from .quarantine_execution import claim_formal_ready
        return {'protocol': self.digest, 'formal_eligible': claim_formal_ready(self), 'slots': list(SLOTS)}

    def recover_reservations(self):
        self._reserve()
        return self.verify()

    def _losses(self, e, stream, pipeline):
        lc = self.lifecycle
        losses = lc._losses(stream, e.start, e.end, e.scoring_cutoff,
                            expected_pipeline=pipeline['pipeline_id'] if pipeline is not None else None)
        # Verify actual activation lineage for every scored forecast, rather than
        # accepting caller-supplied loss arrays or nominal stream names.
        activations = []
        for text, in lc.db.execute('SELECT record FROM operations ORDER BY seq'):
            operation = json.loads(text)
            if operation['kind'] != 'activated':
                continue
            detail = operation['detail']
            job = lc.db.execute('SELECT cutoff,stream,pipeline,model FROM jobs WHERE id=?', (detail['job'],)).fetchone()
            if job[1] == stream:
                recipe = json.loads(job[2])
                if pipeline is not None and recipe != pipeline:
                    raise ValueError('Fixed claim stream changed its complete pipeline')
                if pipeline is None:
                    from .typed_policy_confirmation import candidate_pipeline
                    expected = candidate_pipeline(lc, job[0], e.reference)
                    # Selection complexity is not part of the deployed learner.
                    # Every other recipe field remains an exact identity check.
                    expected = {key: value for key, value in expected.items() if key != 'complexity'}
                    if recipe != expected:
                        raise ValueError('Candidate policy changed outside its frozen causal selection rule')
                activations.append((stamp(detail['origin']), stamp(operation['at']), job[3], recipe['pipeline_id']))
        for origin, symbol, text in lc.forecasts.db.execute(
                'SELECT origin,symbol,prediction FROM forecasts WHERE run=? AND stream=? AND task=? AND origin>=? AND origin<?',
                (lc.plan.digest, stream, 'volatility', e.start, e.end)):
            if (stamp(origin).floor('h').isoformat(), symbol) not in losses:
                continue
            forecast = json.loads(text)
            legal = [a for a in activations if a[0] <= stamp(origin) and a[1] <= stamp(forecast['issued_at'])]
            if not legal or legal[-1][2:] != (forecast['model_id'], forecast['pipeline_id']):
                raise ValueError('Claim forecast lacks its actual original activation')
        if any(symbol not in self.manifest['symbols'] for _, symbol in losses):
            raise ValueError('Claim support differs from the frozen universe')
        return losses

    def _publish(self, slot, score):
        e = self.endpoints[slot]
        batch = self._reservation_args(e)['batch']
        found = self.lifecycle.exposure.db.execute('SELECT result FROM confirmations WHERE batch=?', (batch,)).fetchone()
        if found[0] is None:
            self.lifecycle.exposure.finish_confirmation(batch, score)
        elif json.loads(found[0]) != score:
            raise ValueError('Claim publication cannot replace an earlier result')
        immutable_record(self.directory / slot / 'score.json', score)
        return score

    def finalize(self, slot):
        if slot not in SLOTS:
            raise ValueError('This protocol cannot consume factor slots')
        with FileLock(str(self.directory) + '.lock', timeout=0):
            self.verify()
            row = self._row(slot)
            if row[1] is not None:
                return self._publish(slot, json.loads(row[1]))
            e, lc = self.endpoints[slot], self.lifecycle
            if lc.now() < stamp(e.scoring_cutoff):
                return {'status': 'pending', 'slot': slot, 'scoring_cutoff': e.scoring_cutoff,
                        'formal_eligible': False}
            args = self._reservation_args(e)
            consumed = lc.exposure.db.execute('SELECT consumed FROM confirmations WHERE batch=?', (args['batch'],)).fetchone()[0]
            started = self.directory / slot / 'read_started.json'
            if consumed or started.exists():
                return {'status': 'interrupted_consumed', 'slot': slot, 'formal_eligible': False}
            lc.verify()
            from .quarantine_execution import (
                claim_formal_ready,
                decision_for,
                six_month_claim_scope,
            )
            formal = claim_formal_ready(self)
            immutable_record(started, {'protocol': self.digest, 'slot': slot, 'at': lc.now().isoformat()})
            lc.exposure.consume_confirmation(args['batch'], as_of=lc.now())
            base = self._losses(e, e.reference_stream, e.reference)
            candidate = self._losses(e, e.candidate_stream, e.candidate if slot == 'library_skill' else None)
            common = sorted(set(base) & set(candidate))
            if any(base[k][1:] != candidate[k][1:] for k in common):
                raise ValueError('Paired claim streams have different actual labels/revisions')
            pairs = [{'hour': k[0], 'symbol': k[1], 'reference_loss': base[k][0],
                      'candidate_loss': candidate[k][0], 'label': base[k][1], 'revision': base[k][2]} for k in common]
            hours = int((stamp(e.end) - stamp(e.start)) / pd.Timedelta(hours=1))
            inputs = {'protocol': self.digest, 'slot': slot, 'endpoint': e.to_dict(), 'pairs': pairs,
                      'nominal_hours': hours, 'nominal_symbol_hours': hours * len(lc.plan.symbols),
                      'reference_pairs': len(base), 'candidate_pairs': len(candidate),
                      'scoring_cutoff': e.scoring_cutoff}
            input_path = self.directory / slot / 'inputs.json'
            immutable_record(input_path, inputs)
            result = paired_effect_interval(pd.DatetimeIndex([k[0] for k in common], tz='UTC'), [base[k][0] for k in common],
                [candidate[k][0] for k in common], symbols=[k[1] for k in common],
                expected_symbols=lc.plan.symbols, eligible_count=hours, rule=e.rule, calibration_passed=formal)
            result['formal_eligible'] = formal
            isolated = decision_for(lc, result)
            if isolated is not None:
                result['quarantined_decision'] = isolated
            score = {'version': 'v7-six-month-claim-result-v2', 'protocol': self.digest, 'slot': slot,
                     'endpoint_id': e.digest, 'candidate_id': e.candidate_id,
                     'reference_id': e.reference['pipeline_id'], 'completed_at': lc.now().isoformat(),
                     'scoring_cutoff': e.scoring_cutoff, 'inputs_sha256': file_hash(input_path),
                     'result': result, 'formal_eligible': formal,
                     'six_month_claim_scope_hash': fingerprint(six_month_claim_scope(self)) if formal else None,
                     'missing_formal_scope': [] if formal else ['full_five_claim_pipeline_calibration_G09_G11']}
            with closing(sqlite3.connect(self.path)) as db, db:
                db.execute('PRAGMA synchronous=FULL')
                db.execute('BEGIN IMMEDIATE')
                if db.execute('UPDATE claims SET score=?,hash=? WHERE slot=? AND score IS NULL',
                              (canonical(score), fingerprint(score), slot)).rowcount != 1:
                    raise ValueError('Claim endpoint already committed')
            return self._publish(slot, score)
