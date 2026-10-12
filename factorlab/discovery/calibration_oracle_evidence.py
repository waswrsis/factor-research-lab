"""Original-ledger oracle loss reconstruction; not a truth certificate.

This module never trains, grants budget, repairs a ledger or truncates logMSE.
A whole generated path is one observation. Independence, generator support and
adaptive policy coverage require the separately frozen full trial registry.
Local hashes protect accidental replacement, not a hostile administrator.
"""
from __future__ import annotations

import json
import math
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass

import pandas as pd

from factorlab.discovery.calibration_acceptance import CLAIMS
from factorlab.discovery.claim_protocol import POLICY
from factorlab.discovery.exposure import ExposureLedger
from factorlab.discovery.model_registry import ForecastLedger, ModelRegistry
from factorlab.discovery.v7_lifecycle import Lifecycle, LifecyclePlan, stamp
from factorlab.protocol import fingerprint, project_path

VERSION = 'v7-independent-oracle-original-loss-v1'


def _plain(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate oracle evidence JSON key')
            result[key] = value
        return result

    def invalid(_):
        raise ValueError('Nonfinite oracle evidence JSON')
    if not path.is_file() or path.stat().st_size > 64 * 2**20:
        raise ValueError('Missing or oversized original oracle evidence')
    return json.loads(path.read_text('utf-8'), object_pairs_hook=pairs, parse_constant=invalid)


def _pipeline(value):
    value = _plain(value)
    identifier = value.get('pipeline_id')
    if (not isinstance(identifier, str) or len(identifier) != 64
            or any(c not in '0123456789abcdef' for c in identifier) or len(value) < 2):
        raise ValueError('Oracle needs the complete original frozen learner')
    return value


@dataclass(frozen=True)
class OracleClaimSpec:
    """An estimand copied from the original trial, never an oracle-selected winner.

    ``candidate=None`` is explicit nonselection. Algorithm means the original
    adaptive policy, not a single final selected model evaluated on more data.
    The full registry, not this value object, authenticates that provenance.
    """
    slot: str
    start: str
    end: str
    reference_stream: str
    candidate_stream: str
    reference: dict
    candidate: dict | None

    def __post_init__(self):
        start, end = stamp(self.start), stamp(self.end)
        if (self.slot not in CLAIMS or start != start.floor('h') or end != end.floor('h')
                or not start < end or end - start > pd.Timedelta(days=366)
                or any(type(s) is not str or not s or len(s) > 128
                       for s in (self.reference_stream, self.candidate_stream))
                or self.reference_stream == self.candidate_stream):
            raise ValueError('Invalid original oracle claim/window')
        reference = _pipeline(self.reference)
        candidate = None if self.candidate is None else _plain(self.candidate)
        if self.slot == 'algorithm':
            if (candidate != {**POLICY, 'initial_pipeline': reference}
                    or self.candidate_stream != POLICY.get('evaluation_stream', 'A0')):
                raise ValueError('Algorithm oracle requires the entire original A0 policy')
        elif candidate is not None:
            candidate = _pipeline(candidate)
        object.__setattr__(self, 'start', start.isoformat())
        object.__setattr__(self, 'end', end.isoformat())
        object.__setattr__(self, 'reference', reference)
        object.__setattr__(self, 'candidate', candidate)
        object.__setattr__(self, '_seal', fingerprint(self.to_dict()))

    def to_dict(self):
        return _plain(asdict(self))

    @property
    def digest(self):
        if fingerprint(self.to_dict()) != self._seal:
            raise ValueError('Original oracle estimand mutated')
        return self._seal

    @property
    def scoring_cutoff(self):
        return (stamp(self.end) + pd.Timedelta(hours=36)).isoformat()


class _ReadOnlyForecasts(ForecastLedger):
    def recover_receipts(self):
        # Lifecycle.verify normally repairs missing SQL from existing receipts.
        # An oracle reader must instead reject that torn state without writes.
        return self.verify()


@contextmanager
def _lifecycle(directory):
    directory = project_path(directory)
    manifest = _read(directory / 'lifecycle.json')
    value = dict(manifest['plan'])
    value['symbols'] = tuple(value['symbols'])
    plan = LifecyclePlan(**value)
    if manifest != {'plan': _plain(asdict(plan)), 'sha256': plan.digest}:
        raise ValueError('Original lifecycle identity changed')
    lc = Lifecycle.__new__(Lifecycle)
    lc.directory, lc.plan = directory, plan
    lc.clock = lambda: pd.Timestamp.now(tz='UTC')
    lc.registry = ModelRegistry.__new__(ModelRegistry)
    lc.registry.directory = directory / 'models'
    lc.forecasts = _ReadOnlyForecasts.__new__(_ReadOnlyForecasts)
    lc.forecasts.records_directory = directory / 'forecasts.records'
    lc.exposure = ExposureLedger.__new__(ExposureLedger)
    lc.exposure.path = directory / 'exposure.sqlite'
    lc.exposure.clock = lc.clock
    connections = []
    try:
        for owner, path in ((lc, directory / 'lifecycle.sqlite'),
                            (lc.forecasts, directory / 'forecasts.sqlite'),
                            (lc.exposure, lc.exposure.path)):
            if not path.is_file():
                raise ValueError('Missing original oracle ledger; cannot initialize')
            owner.db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
            owner.db.execute('PRAGMA query_only=ON')
            owner.db.execute('BEGIN')
            connections.append(owner.db)
        yield lc
    finally:
        for db in connections:
            db.close()


def _stream(lc, spec, stream, pipeline):
    losses = lc._losses(stream, spec.start, spec.end, spec.scoring_cutoff,
                         expected_pipeline=pipeline['pipeline_id'] if pipeline else None)
    activations = []
    for text, in lc.db.execute('SELECT record FROM operations ORDER BY seq'):
        op = json.loads(text)
        if op['kind'] != 'activated':
            continue
        detail = op['detail']
        job = lc.db.execute('SELECT cutoff,stream,pipeline,model FROM jobs WHERE id=?',
                            (detail['job'],)).fetchone()
        if job[1] != stream:
            continue
        recipe = json.loads(job[2])
        if pipeline is not None and recipe != {k: v for k, v in pipeline.items() if k != 'complexity'}:
            raise ValueError('Oracle fixed learner changed during its original path')
        if pipeline is None:
            if stream == 'A0_policy_trial':
                from .typed_policy_confirmation import candidate_pipeline
                expected = candidate_pipeline(lc, job[0], spec.reference)
            else:
                expected = lc.adoption_pipeline(job[0], spec.reference)
            expected = {k: v for k, v in expected.items() if k != 'complexity'}
            if recipe != expected:
                raise ValueError('Oracle algorithm does not follow original adaptive adoption policy')
        activations.append((stamp(detail['origin']), stamp(op['at']), job[3], recipe['pipeline_id'],
                            fingerprint(op)))
    records = {}
    for origin, symbol, text in lc.forecasts.db.execute(
            'SELECT origin,symbol,prediction FROM forecasts WHERE run=? AND stream=? AND task=? '
            'AND origin>=? AND origin<? ORDER BY origin,symbol',
            (lc.plan.digest, stream, 'volatility', spec.start, spec.end)):
        key = (stamp(origin).floor('h').isoformat(), symbol)
        if key not in losses:
            continue
        forecast = json.loads(text)
        legal = [a for a in activations if a[0] <= stamp(origin) and a[1] <= stamp(forecast['issued_at'])]
        if not legal or legal[-1][2:4] != (forecast['model_id'], forecast['pipeline_id']):
            raise ValueError('Oracle forecast lacks its original actual activation')
        window = stamp(forecast['forecast_window_start'])
        settled = lc.db.execute('SELECT record,hash FROM settlement WHERE run=? AND stream=? '
            'AND origin=? AND symbol=? AND task=?',
            (lc.plan.digest, stream, origin, symbol, 'volatility')).fetchone()
        label = json.loads(settled[0])
        if (symbol not in lc.plan.symbols or forecast['availability'] not in {'actual_receipt', 'quarantined_finite_observation'}
                or stamp(forecast['received_at']) > stamp(origin)
                or not stamp(origin) <= stamp(forecast['issued_at']) < window
                or window != stamp(origin).ceil('h')
                or stamp(label['maturity']) < window + pd.Timedelta(hours=12)
                or stamp(label['received_at']) < stamp(label['maturity'])):
            raise ValueError('Oracle receipt or horizon chronology changed')
        records[key] = {'origin': origin, 'forecast_hash': fingerprint(forecast),
                        'settlement_hash': settled[1], 'activation_hash': legal[-1][4],
                        'window': window.isoformat()}
    return losses, records


def reconstruct_claim(directory, spec):
    """Recompute original logMSE and equal-symbol/hour risk, with no repair.

    This authenticates ledger consistency and activation, not raw market label
    correctness or generator independence. Those are full-producer obligations.
    Missing support is retained in nominal denominators; no loss imputation.
    """
    if type(spec) is not OracleClaimSpec:
        raise TypeError('Frozen typed original oracle estimand required')
    _ = spec.digest
    with _lifecycle(directory) as lc:
        lc.verify()
        if lc.now() < stamp(spec.scoring_cutoff):
            raise ValueError('Original oracle fixed scoring cutoff has not matured')
        result = {'version': VERSION, 'lifecycle_directory': str(lc.directory),
                  'lifecycle_digest': lc.plan.digest, 'source_id': lc.plan.source_id,
                  'symbols': list(lc.plan.symbols), 'estimand': spec.to_dict(),
                  'estimand_hash': spec.digest, 'scoring_cutoff': spec.scoring_cutoff,
                  'formal_gate_eligible': False, 'truth': 'uncertain'}
        hours = int((stamp(spec.end) - stamp(spec.start)) / pd.Timedelta(hours=1))
        result.update(nominal_hours=hours, nominal_symbol_hours=hours * len(lc.plan.symbols))
        if spec.candidate is None:
            return {**result, 'status': 'not_selected', 'pairs': [], 'path_risks': None,
                    'hour_support': 0, 'pair_support': 0, 'coverage': 0., 'pair_coverage': 0.}
        base, base_records = _stream(lc, spec, spec.reference_stream, spec.reference)
        candidate, candidate_records = _stream(lc, spec, spec.candidate_stream,
                                               None if spec.slot == 'algorithm' else spec.candidate)
        common = sorted(set(base) & set(candidate))
        pairs, hourly = [], {}
        for key in common:
            if (base[key][1:] != candidate[key][1:]
                    or base_records[key]['window'] != candidate_records[key]['window']):
                raise ValueError('Oracle pairs use different actual labels, revisions or forecast windows')
            row = {'hour': key[0], 'symbol': key[1], 'reference_loss': base[key][0],
                   'candidate_loss': candidate[key][0], 'label': base[key][1],
                   'revision': base[key][2], 'reference_evidence': base_records[key],
                   'candidate_evidence': candidate_records[key]}
            pairs.append(row)
            hourly.setdefault(key[0], []).append((base[key][0], candidate[key][0]))
        means = [(math.fsum(a for a, _ in values) / len(values),
                  math.fsum(b for _, b in values) / len(values)) for values in hourly.values()]
        risk = None if not means else {'reference': math.fsum(a for a, _ in means) / len(means),
                                      'candidate': math.fsum(b for _, b in means) / len(means)}
        coverage, pair_coverage = len(means) / hours, len(pairs) / (hours * len(lc.plan.symbols))
        return {**result, 'status': 'observed' if min(coverage, pair_coverage) >= .95 else 'incomplete_support',
                'pairs': pairs, 'path_risks': risk, 'hour_support': len(means),
                'pair_support': len(pairs), 'coverage': coverage, 'pair_coverage': pair_coverage,
                'reference_support': len(base), 'candidate_support': len(candidate),
                'weighting': 'equal_symbol_within_hour_then_equal_hour',
                'missingness_policy': 'no_imputation_missing_support_prevents_oracle_truth',
                'unresolved': ['independent_generator_original_draw_proof',
                               'unbounded_logMSE_risk_enclosure', 'full_adaptive_execution_coverage',
                               'raw_receipt_label_reconstruction']}


def _raw_labels(feed_directory, rows, *, source_id, symbols, delay):
    """Reconstruct the production thirteen-close target at original receipt time.

    SQL reads only the thirteen required candle positions. Per-call raw-source
    verification cache is bounded to one window, never shared across inspections.
    No reinterpretation of assumed seed import times as historical origins.
    """
    import numpy as np

    from factorlab.discovery.v7_feed import ReceiptArchive
    directory = project_path(feed_directory)
    identity = _read(directory / 'identity.json')
    if identity != {'source_id': source_id, 'symbols': list(symbols)}:
        raise ValueError('Oracle raw feed differs from original runtime source')
    feed = ReceiptArchive.__new__(ReceiptArchive)
    feed.directory, feed.symbols, feed.identity = directory, tuple(symbols), source_id
    feed.clock = lambda: pd.Timestamp.now(tz='UTC')
    path = directory / 'receipts.sqlite'
    if not path.is_file():
        raise ValueError('Missing original raw feed index')
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as db, db:
        feed.db = db
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        feed.verify()
        for row in rows:
            start, received = stamp(row['window']), stamp(row['received_at'])
            end = start + pd.Timedelta(hours=12)
            found, cache = {}, {}
            records = db.execute('SELECT b.body,b.origin,b.version,b.receipt,b.batch,b.symbol,b.opened '
                'FROM bars b LEFT JOIN publications p ON p.batch=b.batch '
                'WHERE (p.batch IS NULL OR p.published_at IS NOT NULL) '
                'AND COALESCE(p.published_at,b.receipt)<=? AND b.symbol=? AND b.opened>=? AND b.opened<? '
                'ORDER BY b.opened,COALESCE(p.published_at,b.receipt),(p.batch IS NOT NULL),p.sequence,b.rowid',
                (received.isoformat(), row['symbol'], (start-pd.Timedelta(hours=1)).isoformat(), end.isoformat()))
            for text, origin, version, receipt, batch, symbol, opened in records:
                body = json.loads(text)
                if (fingerprint(body) != version or body.get('prepared_at', body.get('imported_at')) != receipt
                        or body['prediction_origin'] != bool(origin) or body['ohlcv']['symbol'] != symbol
                        or body['ohlcv']['open_time'] != opened):
                    raise ValueError('Oracle raw candle revision/index changed')
                published = feed._verify_bar_source(batch, body, cache=cache)
                if published is not None:
                    found[opened] = (body, version, stamp(published))
            expected = pd.date_range(start-pd.Timedelta(hours=1), end, freq='h', inclusive='left')
            if len(found) != 13 or list(found) != [t.isoformat() for t in expected]:
                raise ValueError('Oracle label lacks thirteen original as-of raw candles')
            values = list(found.values())
            if any(body['basis'] != 'actual_receipt' for body, _, _ in values):
                raise ValueError('Assumed imported data cannot authenticate an actual future label')
            receipt = max(end+pd.Timedelta(seconds=delay), *(published for _, _, published in values))
            closes = np.asarray([body['ohlcv']['close'] for body, _, _ in values], dtype=float)
            if not np.isfinite(closes).all() or np.any(closes <= 0):
                raise ValueError('Invalid original target candle')
            target = float(np.sqrt(np.square(np.diff(np.log(closes))).sum()))
            revision = fingerprint([version for _, version, _ in values])
            if (row['label'] != target or row['revision'] != revision
                    or stamp(row['maturity']) != receipt or received != receipt):
                raise ValueError('Settled oracle target differs from original production raw calculation')


def verify_raw_claim(directory, result):
    """Read actual forward identity; authenticate every included settled label."""
    directory = project_path(directory)
    manifest = _read(directory / 'forward.json')
    if set(manifest) != {'identity', 'sha256'} or fingerprint(manifest['identity']) != manifest['sha256']:
        raise ValueError('Original forward runtime identity changed')
    identity = manifest['identity']
    rows, seen = [], set()
    with _lifecycle(directory) as lc:
        if lc.plan.digest != result['lifecycle_digest']:
            raise ValueError('Oracle result moved to another lifecycle')
        for pair in result['pairs']:
            for role, stream in (('reference', result['estimand']['reference_stream']),
                                 ('candidate', result['estimand']['candidate_stream'])):
                evidence = pair[role+'_evidence']
                token = (evidence['window'], pair['symbol'], evidence['settlement_hash'])
                if token in seen:
                    continue
                seen.add(token)
                found = lc.db.execute('SELECT record,hash FROM settlement WHERE run=? AND stream=? '
                    'AND origin=? AND symbol=? AND task=?',
                    (lc.plan.digest, stream, evidence['origin'], pair['symbol'], 'volatility')).fetchone()
                if found is None or found[1] != evidence['settlement_hash']:
                    raise ValueError('Original oracle settlement missing or changed')
                record = json.loads(found[0])
                if fingerprint(record) != found[1]:
                    raise ValueError('Original oracle settlement hash changed')
                rows.append({**record, 'symbol': pair['symbol'], 'window': evidence['window']})
    from .quarantined_observation import QuarantinedObservation
    with _lifecycle(directory) as lc:
        if QuarantinedObservation.from_lifecycle(lc) is not None:
            # Canonical LC audit reconstructs every original raw 13-close label,
            # exact released prefix, revision and physical source publication.
            lc.verify()
            if any(r.get('availability') != 'quarantined_finite_observation' for r in rows):
                raise ValueError('Bound raw oracle label lost original observation provenance')
            return {'verified_settlements': len(rows),
                'raw_label_policy': 'original_prefix_13_close_volatility12h', 'formal_gate_eligible': False}
    _raw_labels(identity['feed_directory'], rows, source_id=result['source_id'], symbols=result['symbols'],
                delay=identity['rolling']['availability_delay_seconds'])
    return {'verified_settlements': len(rows), 'raw_label_policy': 'original_13_close_volatility12h',
            'formal_gate_eligible': False}


def _permit(value, role):
    # No duck typing, callback validator, public clock or caller test flag.
    from factorlab.discovery.full_calibration_registry import QuarantinedTrialPermit
    if type(value) is not QuarantinedTrialPermit:
        raise TypeError('Actual preregistered quarantined trial permit required')
    value.verify()
    if value.trial['role'] != role:
        raise ValueError('Oracle operation has the wrong registered trial domain')
    return value


def _original_specs(permit):
    from factorlab.discovery.claim_protocol import ClaimProtocol
    with _lifecycle(permit.paths['runtime']) as lc:
        permit.verify(lc)
        lc.verify()
        protocol = ClaimProtocol(lc)
        specs = []
        for slot in ('library_skill', 'algorithm'):
            e = protocol.endpoints[slot]
            specs.append(OracleClaimSpec(slot, e.start, e.end, e.reference_stream,
                                        e.candidate_stream, e.reference, e.candidate))
        batches = lc.db.execute('SELECT id,ordinal FROM batches ORDER BY ordinal').fetchall()
        if [ordinal for _, ordinal in batches] != [1, 2, 3]:
            raise ValueError('Original trial must retain all three factor selection outcomes')
        for batch, ordinal in batches:
            identity, status, selected = lc._batch(batch)
            if status not in {'selected_C', 'concluded', 'concluded_no_candidate'}:
                raise ValueError('Original factor selection has not reached its fixed endpoint')
            candidate = None
            if selected is not None:
                candidate = next((p for p in identity['candidates']
                                  if p['pipeline_id'] == selected['candidate_id']), None)
                if candidate is None:
                    raise ValueError('Original selected learner missing from frozen candidate list')
            stream = batch + ':' + (candidate['pipeline_id'] if candidate else 'not_selected')
            specs.append(OracleClaimSpec('factor'+str(ordinal), identity['v_end'], identity['c_end'],
                                        batch+':reference', stream, identity['reference'], candidate))
        ordered = sorted(specs, key=lambda s: CLAIMS.index(s.slot))
        if any(lc.now() < stamp(s.scoring_cutoff) for s in ordered):
            raise ValueError('Original trial full observation endpoints have not matured')
        return ordered


def _estimand_paths(permit):
    parent = permit.trial['population_trial_id']
    validation = permit.registry.paths(permit.trial['cell'], permit.trial['index'])
    return (validation['root'] / 'oracle_estimands.json',
            permit.registry.directory / 'oracle_estimands' / (parent+'.json'))


def freeze_oracle_estimands(validation_permit):
    """Copy original selected learners BEFORE independent oracle enrollment.

    Artifacts alone cannot prove all forecast/provider paths ran. This producer
    consequently never emits a truth envelope or a formal certificate.
    """
    from filelock import FileLock

    from factorlab.discovery.evidence import file_hash
    from factorlab.discovery.v7_storage import immutable_record
    permit = _permit(validation_permit, 'validation')
    paths = _estimand_paths(permit)
    with FileLock(str(paths[0])+'.producer.lock', timeout=0):
        specs = _original_specs(permit)
        base = {'version': VERSION, 'registry_digest': permit.registry.digest,
                'validation_trial': permit.trial, 'estimands': [s.to_dict() for s in specs],
                'implementation_hash': file_hash(__file__), 'formal_gate_eligible': False,
                'truth_scope': 'original_selected_factor_learners_and_original_adaptive_A0_policy'}
        if any(p.exists() for p in paths):
            old = _read(paths[0])
            if (_read(paths[1]) != old or {k: old[k] for k in base} != base):
                raise ValueError('Original oracle estimands missing or changed; cannot replace')
            return old
        # Detect already enrolled oracle executions without enumerating the
        # (possibly million-entry) unexecuted fixed path grid.
        for path in (permit.registry.directory/'trials').glob('*/enrollment.json'):
            trial = _read(path)['intent']
            if (trial.get('role') == 'oracle'
                    and trial.get('population_trial_id') == permit.trial['population_trial_id']):
                raise ValueError('Cannot choose an oracle estimand after oracle execution enrollment')
        value = {**base, 'frozen_at': pd.Timestamp.now(tz='UTC').isoformat()}
        for path in paths:
            immutable_record(path, value)
        return value


def _frozen_estimands(permit):
    from factorlab.discovery.evidence import file_hash
    first, second = _estimand_paths(permit)
    value = _read(first)
    if (_read(second) != value or value['version'] != VERSION
            or value['registry_digest'] != permit.registry.digest
            or value['implementation_hash'] != file_hash(__file__)
            or value['validation_trial'] != permit.registry.identity_for(permit.trial['cell'], permit.trial['index'])
            or value['formal_gate_eligible'] is not False):
        raise ValueError('Frozen oracle estimand binding changed')
    specs = [OracleClaimSpec(**s) for s in value['estimands']]
    if tuple(s.slot for s in specs) != CLAIMS:
        raise ValueError('Exactly five ordered original oracle estimands required')
    # Reconstruct original selection, so a second locally rehashed oracle recipe
    # cannot replace the original selected learner while keeping the same label.
    from factorlab.discovery.full_calibration_registry import QuarantinedTrialPermit
    original = QuarantinedTrialPermit(permit.registry, value['validation_trial'])
    if [s.to_dict() for s in _original_specs(original)] != value['estimands']:
        raise ValueError('Frozen oracle learner differs from original trial selection')
    return value, specs


def _reconstruct(oracle_permit):
    permit = _permit(oracle_permit, 'oracle')
    estimands, specs = _frozen_estimands(permit)
    from factorlab.discovery.calibration_runtime_contract import _original_operations
    from factorlab.discovery.full_calibration_registry import EVENT
    with _lifecycle(permit.paths['runtime']) as lc:
        permit.verify(lc)
        enrollment = [o for o in _original_operations(lc) if o['kind'] == EVENT]
        if len(enrollment) != 1 or stamp(enrollment[0]['at']) < stamp(estimands['frozen_at']):
            raise ValueError('Oracle execution preceded original learner freeze')
    claims = []
    for spec in specs:
        result = reconstruct_claim(permit.paths['runtime'], spec)
        raw = verify_raw_claim(permit.paths['runtime'], result)
        result['raw_label_verification'] = raw
        result['unresolved'] = [s for s in result.get('unresolved', [])
                                if s != 'raw_receipt_label_reconstruction']
        # Retain only a digest of the reconstructable per-hour evidence in the
        # endpoint artifact; never hold five full path tables simultaneously.
        result['paired_evidence_hash'] = fingerprint(result.pop('pairs'))
        claims.append(result)
    permit.verify()
    return {'version': VERSION, 'registry_digest': permit.registry.digest, 'oracle_trial': permit.trial,
            'estimand_hash': fingerprint(estimands), 'claims': claims,
            'unit': 'one_original_whole_oracle_path', 'truth': 'uncertain',
            'formal_gate_eligible': False, 'independence_certified': False,
            'full_pipeline_execution_verified': False, 'risk_envelope': None,
            'reason': 'Original path risks are not an unbounded-loss expectation enclosure'}


def produce_oracle_path(oracle_permit):
    from filelock import FileLock

    from factorlab.discovery.v7_storage import immutable_record
    permit = _permit(oracle_permit, 'oracle')
    path = permit.paths['root'] / 'oracle_path.json'
    with FileLock(str(path)+'.producer.lock', timeout=0):
        value = _reconstruct(permit)
        wrapper = {'record': value, 'hash': fingerprint(value)}
        immutable_record(path, wrapper)
        return value


def read_oracle_path(oracle_permit):
    permit = _permit(oracle_permit, 'oracle')
    wrapper = _read(permit.paths['root'] / 'oracle_path.json')
    value = _reconstruct(permit)
    if wrapper != {'record': value, 'hash': fingerprint(value)}:
        raise ValueError('Original oracle artifact differs from reconstructed path evidence')
    return value


def aggregate_oracle_paths(registry, cell, index):
    from factorlab.discovery.full_calibration_registry import (
        FullTrialRegistry,
        QuarantinedTrialPermit,
    )
    if type(registry) is not FullTrialRegistry:
        raise TypeError('Actual frozen full trial registry required')
    registry.verify()
    points = {slot: [] for slot in CLAIMS}
    missing = incomplete = 0
    for oracle_index in range(registry.plan.oracle_paths):
        trial = registry.identity_for(cell, index, oracle_index=oracle_index)
        permit = QuarantinedTrialPermit(registry, trial)
        path = permit.paths['root'] / 'oracle_path.json'
        if not path.exists():
            missing += 1
            continue
        value = read_oracle_path(permit)  # corrupt evidence never becomes a dropped path
        path_complete = True
        for claim in value['claims']:
            if claim['status'] != 'observed' or claim['path_risks'] is None:
                path_complete = False
            else:
                points[claim['estimand']['slot']].append(claim['path_risks'])
        incomplete += int(not path_complete)
    return {'version': VERSION, 'registry_digest': registry.digest,
            'expected_paths': registry.plan.oracle_paths, 'missing_paths': missing,
            'incomplete_paths': incomplete,
            'claims': {slot: {'observed_paths': len(rows),
                'diagnostic_reference_risk': math.fsum(r['reference'] for r in rows)/len(rows) if rows else None,
                'diagnostic_candidate_risk': math.fsum(r['candidate'] for r in rows)/len(rows) if rows else None,
                'truth': 'uncertain', 'risk_envelope': None} for slot, rows in points.items()},
            'weights': 'equal_complete_whole_path_only_descriptive_not_selection_corrected_truth',
            'formal_gate_eligible': False, 'oracle_scope_authenticated': False,
            'unresolved': ['unbounded_logMSE_risk_enclosure', 'independent_generator_draw_proof',
                           'full_adaptive_execution_coverage']}
