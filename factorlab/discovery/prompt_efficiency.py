"""Prospective Q0/Q1/Q2 efficiency pilot on one original standard S wallet.

This is a new algorithm version, not a reconstruction of hidden V6 provider
context. It never produces a confirmation, serving gate or noninferiority claim.
Only the fixed CLI worker executes work; records cannot authorize arbitrary code.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from filelock import FileLock

from factorlab.protocol import fingerprint, project_path
from .monthly_partition_policy import counts, frozen_policy
from .prompt_builder import PromptIdentity, PromptPacket, build_formula_prompt
from .v7_storage import immutable_record
from .work_lease import FrozenWorkLease

VERSION = 'v7-prompt-efficiency-prospective-v2-archive-engineering'
MODES = ('Q0', 'Q1', 'Q2')
FAMILY_FIELDS = ('family_id', 'hypothesis', 'mechanism', 'competing_explanation',
                 'falsification', 'parent_ids', 'template')


def _write(path, value):
    immutable_record(path, {'value': value, 'sha256': fingerprint(value)})


def _read(path):
    from .program_syntax import strict_json
    path = project_path(path)
    if not path.is_file() or path.stat().st_size > 4 * 2**20:
        raise ValueError('Missing or oversized original efficiency evidence')
    row = strict_json(path.read_text('utf-8'), max_bytes=4 * 2**20)
    if set(row) != {'value', 'sha256'} or fingerprint(row['value']) != row['sha256']:
        raise ValueError('Original efficiency evidence digest changed')
    return row['value']


def _pair(identity, name, value=None):
    paths = [project_path(identity[k]) / name for k in ('root', 'directory')]
    if value is not None:
        for path in paths:
            _write(path, value)
    left, right = map(_read, paths)
    if left != right:
        raise ValueError('Original efficiency mirrors differ')
    return left


def schedule():
    # Two actual six-month feature/readout preparations: 2*(3 folds*2 + 1) solves.
    rows = [{'key': 'prepare', 'seconds': 300, 'counts': counts(solve=14)}]
    for mode in MODES:
        rows.append({'key': mode, 'seconds': 300,
                     'counts': counts(family=1, gpt_attempt=2, ast=32)})
        rows.extend({'key': f'{mode}-F-{i}', 'seconds': 90,
                     'counts': counts(f=1, solve=198 if i == 0 else 15)} for i in range(8))
    return rows


def protocol():
    rows = schedule()
    return {'version': VERSION, 'model': 'gpt-6-astra', 'reasoning': 'high',
        'mode_order': list(MODES), 'contexts': 2, 'calls_per_mode': 2,
        'family_rule': 'one prospectively supplied common semantic/template contract in ALL modes and both contexts; no response feedback',
        'family_blocks': 'three nontransferable paid blocks; repeated semantic families are not three novel families',
        'Q0': 'prospective full-single v1; explicit complete bounded action/history JSON; pretty canonical AST',
        'Q1': 'same Q0 semantic content and single output; compact JSON and exact compact AST roundtrip',
        'Q2': 'same context; 1..8 seeds and fixed finite template materialization',
        'legacy_hidden_provider_context': None, 'legacy_prompt_reconstructed': False,
        'AST_rule': '32 per mode including invalid/duplicate seeds and materializations; 16 per context',
        'F_rule': '4 per context, 8 per mode; no transfer; every absent/failed slot retained',
        'P_rule': 'only first F slot per mode funds full 198 solve bound; no clipping or shortened profiling',
        'cold_calendar_months': 3, 'outer_seconds': 7200, 'schedule': rows,
        'S_elapsed_seconds_ceiling': sum(x['seconds'] for x in rows),
        'reserved_seconds': sum(x['seconds'] for x in rows),
        'reserved_counts': {k: sum(x['counts'][k] for x in rows) for k in counts()},
        'V_eligible': False, 'C_eligible': False, 'serving_eligible': False,
        'noninferiority_margin': None, 'required_future_repetitions': None}


def _check_experiment_clock(identity):
    """Charge setup, parent verification and closing to the same original S time."""
    from .s_outer_context import check_clock
    check_clock(identity['clock'])
    origin = identity['clock']['issuer']
    seconds = min(identity['clock']['seconds'], protocol()['S_elapsed_seconds_ceiling'])
    remaining = min(origin['monotonic_started'] + seconds - time.monotonic(),
        datetime.fromisoformat(origin['wall_started_at']).timestamp() + seconds - datetime.now(UTC).timestamp())
    if remaining <= 0:
        raise TimeoutError('Original efficiency allocated S elapsed time consumed')
    return remaining


@dataclass(frozen=True)
class EfficiencyPacket(PromptPacket):
    mode: str = 'Q0'
    semantic_hash: str = ''

    def telemetry(self):
        return {**super().telemetry(), 'template_version': VERSION + ':' + self.mode,
                'request_kind': 'efficiency_single' if self.mode != 'Q2' else 'efficiency_family',
                'mode': self.mode, 'semantic_hash': self.semantic_hash}


def packet_for(mode, base, *, family=None):
    """Lossless full-vs-compact serialization. No historical context is inferred."""
    from .program_syntax import CompactProgram
    if mode not in MODES or type(base) is not PromptPacket:
        raise TypeError('A canonical current S prompt and an original mode are required')
    body = json.loads(base.body)
    # Freeze the entire supplied canonical context. Both encodings carry it all.
    semantic = {'prefix': base.prefix, 'body': body,
        'actions': [{'name': 'propose_single', 'maximum_outputs': 1}],
        'history': {'prior_actions': [], 'scope': 'new experiment; no imported old-provider context'},
        'required_family': family, 'single_candidate': mode != 'Q2'}
    common = {**body, 'efficiency': semantic}
    if mode == 'Q0':
        expanded = [{'parent_id': p['parent_id'],
            'canonical_AST': CompactProgram.from_dict(p['program']).to_program().to_dict()}
            for p in body['parents']]
        common['lossless_expanded_parents'] = expanded
        text = json.dumps(common, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)
    else:
        text = json.dumps(common, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    rule = ('Return exactly ONE concrete program in the usual batch schema. ' if mode != 'Q2' else
            'Return 1..8 concrete programs in ONE finite family. ')
    rule += ('If required_family is present, copy every semantic field and its exact canonical template; '
             'only choose concrete members of that same family. No tools, hidden actions or V/C evidence.')
    packet = EfficiencyPacket(base.prefix + '\n' + rule, text, base.identity,
                              mode=mode, semantic_hash=fingerprint(semantic))
    if len(packet.text.encode('utf-8')) > 50_000:
        raise ValueError('Complete efficiency prompt exceeds original 50KB cap; no silent truncation')
    return packet


def decode_response(payload, packet, *, family=None):
    from .program_syntax import strict_json
    from .v7_agent import ProposalBatch
    from .known_library import FEATURE_NAMES
    body = json.loads(packet.body)
    batch = ProposalBatch.from_dict(strict_json(payload), expected_identity=packet.identity,
        feature_names=FEATURE_NAMES, allowed_parent_ids={p['parent_id'] for p in body['parents']})
    value = batch.to_dict()
    if packet.mode != 'Q2' and len(batch.programs) != 1:
        raise ValueError('Single-candidate control cannot return a family batch')
    semantic = {k: value[k] for k in FAMILY_FIELDS}
    if family is not None and semantic != family:
        raise ValueError('Second context changed original family semantics or canonical AST template')
    return batch, semantic


def materialize(batch, mode, seen, *, reserve):
    """Canonical finite order; exact duplicates remain charged and visible."""
    rows, debit = [], len(batch.programs)
    for program in batch.programs:
        duplicate = program.digest in seen
        seen.add(program.digest)
        rows.append({'status': 'duplicate' if duplicate else 'valid', 'digest': program.digest,
                     'program': program.to_dict(), 'origin': 'provider_seed'})
    if mode == 'Q2':
        choices = batch.template.to_dict()['slots']
        for indices in itertools.product(*(range(len(x['values'])) for x in choices)):
            if debit >= 16:
                break
            reserve(); debit += 1
            try:
                program = batch.template.instantiate(indices)
                duplicate = program.digest in seen
                seen.add(program.digest)
                rows.append({'status': 'duplicate' if duplicate else 'valid', 'digest': program.digest,
                             'program': program.to_dict(), 'origin': 'finite_local', 'choices': list(indices)})
            except ValueError:
                rows.append({'status': 'invalid', 'program': None, 'digest': None,
                             'origin': 'finite_local', 'choices': list(indices)})
    return rows


def _selected(rows):
    result = []
    for row in rows:
        if row['program'] is not None and row['digest'] not in {x['digest'] for x in result}:
            result.append(row)
    return result[:4]


def _source(authority, runtime_directory, *, prefix=None, current=False):
    from .neural_forward_ticket import _LifecycleReader, _verify_authority_binding, _read as raw
    from .quarantined_observation import current_binding, QuarantinedObservation
    from .v7_run import source_files
    lc = _LifecycleReader(runtime_directory)
    try:
        _verify_authority_binding(lc, authority, lc.verify()['operations'])
        forward = raw(lc.directory / 'forward.json')['identity']
        if (forward['source_files'] != authority.manifest['source_files']
                or authority.manifest['source_files'] != source_files()):
            raise ValueError('Efficiency work requires its complete newly frozen executable source')
        if prefix is not None:
            journal = QuarantinedObservation.from_lifecycle(lc)
            if journal is None:
                raise ValueError('Efficiency source lost original observation journal')
            journal.verify_binding(prefix)
            if current and current_binding(lc) != prefix:
                raise ValueError('Efficiency experiment cannot change its frozen original S prefix')
        elif QuarantinedObservation.from_lifecycle(lc) is not None:
            raise ValueError('Quarantined efficiency input requires its original prefix')
        return {'runtime_directory': str(lc.directory), 'feed_directory': forward['feed_directory'],
            'source_id': lc.plan.source_id, 'symbols': list(lc.plan.symbols),
            'exposure_path': str(lc.exposure.path), 'observation_prefix': prefix,
            'rolling': forward['rolling'], 'forward_identity': fingerprint(forward)}
    finally:
        lc.close()


def _exposure(source, cutoffs, *, record=False, experiment=None):
    """Inspect the actual bound ledger, including all protected overlapping tasks."""
    from .exposure import ExposureLedger
    path = project_path(source['exposure_path'])
    if not path.is_file():
        raise ValueError('Missing original bound S exposure ledger')
    exposure = ExposureLedger.__new__(ExposureLedger)
    exposure.path = path
    exposure.clock = lambda: pd.Timestamp.now(tz='UTC')
    exposure.db = sqlite3.connect(path.as_uri() + ('?mode=rw' if record else '?mode=ro'), uri=True)
    try:
        for cutoff in cutoffs:
            end = pd.Timestamp(cutoff)
            start = end - pd.DateOffset(months=6) - pd.Timedelta(hours=1234)
            for symbol in source['symbols']:
                if exposure.db.execute('SELECT 1 FROM evidence WHERE source=? AND symbol=? '
                    "AND start<? AND end>? AND role!='S' LIMIT 1",
                    (source['source_id'], symbol, end.isoformat(), start.isoformat())).fetchone():
                    raise ValueError('Efficiency context intersects non-S original exposure')
            for table in ('validations', 'confirmations'):
                for raw, in exposure.db.execute(f'SELECT identity FROM {table}'):
                    value = json.loads(raw)
                    if (value['source'] == source['source_id']
                            and set(value['symbols']) & set(source['symbols'])
                            and pd.Timestamp(value['start']) < end and pd.Timestamp(value['end']) > start):
                        raise ValueError('Efficiency context intersects a protected original V/C window')
            if record:
                exposure.record(source=source['source_id'], symbols=source['symbols'], start=start, end=end,
                    role='S', batch=experiment, detail={'protocol': VERSION, 'context_cutoff': cutoff})
            elif experiment is not None:
                for symbol in source['symbols']:
                    if not exposure.db.execute('SELECT 1 FROM evidence WHERE source=? AND symbol=? '
                        "AND start=? AND end=? AND role='S' AND batch=?",
                        (source['source_id'], symbol, start.isoformat(), end.isoformat(), experiment)).fetchone():
                        raise ValueError('Efficiency input lost its original actual S exposure')
    finally:
        exposure.close()


def _identity(directory, authority=None):
    from .neural_forward_ticket import open_existing_authority
    identity = _read(project_path(directory) / 'identity.json')
    authority = authority or open_existing_authority(identity['authority_directory'])
    expected_root = authority.directory / 'prompt_efficiency' / identity['actual_month']
    if (identity['version'] != VERSION or identity['protocol'] != protocol()
            or identity['authority_hash'] != authority.manifest_hash
            or project_path(identity['directory']) != project_path(directory)
            or project_path(identity['root']) != expected_root
            or _pair(identity, 'identity.json') != identity
            or authority.manifest['partition_policy'] != frozen_policy('standard')
            or identity['actual_month'] != identity['clock']['issuer']['wall_started_at'][:7]):
        raise ValueError('Efficiency original source/allocation identity changed')
    source = _source(authority, identity['source']['runtime_directory'],
                     prefix=identity['source']['observation_prefix'])
    if source != identity['source']:
        raise ValueError('Efficiency original source reference differs')
    if identity.get('engineering_archive') is not None:
        from .efficiency_archive_input import verify
        value = verify(identity['engineering_archive'], source, authority)
        if value['cutoffs'] != identity['cutoffs']:
            raise ValueError('Original archive cutoffs changed')
    _exposure(source, identity['cutoffs'], experiment=fingerprint(identity))
    if identity['backend']['kind'] == 'artificial':
        from .evidence import file_hash
        fixture = project_path(identity['backend']['path'])
        if not fixture.is_file() or file_hash(fixture) != identity['backend']['sha256']:
            raise ValueError('Original artificial corpus changed; no substituted response evidence')
    elif identity['backend'] != {'kind':'codex'}:
        raise ValueError('Unknown efficiency backend identity')
    return identity, authority


def validate_allocation(authority, work_key, binding, lease, debit, *, current, verification_cache=None):
    """Original wallet calls this both on reservation and historical replay."""
    if set(binding) != {'directory', 'identity_hash', 'slot', 'intent_hash'}:
        raise ValueError('Efficiency allocation requires its exact original intent')
    cache_key = ('prompt_efficiency_identity', str(project_path(binding['directory'])), authority.manifest_hash)
    if verification_cache is not None and cache_key in verification_cache and not current:
        identity = verification_cache[cache_key]
    else:
        identity, _ = _identity(binding['directory'], authority)
        if verification_cache is not None and len(verification_cache) < 32 and not current:
            # Only this original locked wallet audit owns this cache. Original
            # per-row intent/lease/binding/slot comparisons below always run.
            verification_cache[cache_key] = identity
    index = binding['slot']
    if type(index) is not int or not 0 <= index < len(schedule()):
        raise ValueError('Unknown efficiency work slot')
    slot = schedule()[index]
    intent = _pair(identity, f'intents/{index:02d}.json')
    if (binding['identity_hash'] != fingerprint(identity) or fingerprint(intent) != binding['intent_hash']
            or intent != {'identity_hash': fingerprint(identity), 'slot': index, 'work': slot, 'lease': lease.to_dict()}
            or work_key != f'prompt_efficiency:{fingerprint(identity)}:{index}'
            or debit != slot['counts'] or lease.lease_seconds != slot['seconds']
            or lease.wall_started_at[:7] != identity['actual_month']):
        raise ValueError('Efficiency reservation changed its immutable source, counts or physical month')
    if current:
        _check_experiment_clock(identity)
        if lease.parent_pid != identity['clock']['issuer']['parent_pid']:
            raise ValueError('A new process cannot renew this efficiency pilot')
        if _pair_exists(identity, f'results/{index:02d}.json'):
            raise ValueError('Original efficiency slot already consumed')
        for previous in range(index):
            _pair(identity, f'results/{previous:02d}.json')
            _pair(identity, f'confirmations/{previous:02d}.json')
        _source(authority, identity['source']['runtime_directory'],
                prefix=identity['source']['observation_prefix'], current=True)
    return {'lane': 'S', 'owner': {'protocol': VERSION, 'identity_hash': fingerprint(identity),
                                  'directory': identity['directory']}}


def _pair_exists(identity, name):
    exists = [(project_path(identity[k]) / name).is_file() for k in ('root', 'directory')]
    if exists[0] != exists[1]:
        raise ValueError('Incomplete original efficiency publication; no replay')
    return exists[0]


class _Counter:
    """A local consumed counter bounded by the already prepaid original slot."""
    def __init__(self, record):
        self.record = record
        self.used = counts()
        self.lease = FrozenWorkLease.from_dict(record['lease'])

    def check_time(self):
        self.lease.assert_parent_alive()

    def consume(self, kind, amount=1, **unused):
        self.check_time()
        if type(amount) is not int or amount < 1 or self.used[kind] + amount > self.record['counts'][kind]:
            raise ValueError('Original prepaid efficiency slot quota exhausted')
        self.used[kind] += amount


def _snapshot(identity, cutoff):
    if identity.get('engineering_archive') is not None:
        from .efficiency_archive_input import snapshot
        from .neural_forward_ticket import open_existing_authority
        return snapshot(identity['engineering_archive'], identity['source'],
                        open_existing_authority(identity['authority_directory']), cutoff)
    from .v7_feed import ReceiptArchive
    from .neural_forward_ticket import _LifecycleReader
    from .quarantined_observation import snapshot_for_prefix
    source = identity['source']
    if source['observation_prefix'] is not None:
        lc = _LifecycleReader(source['runtime_directory'])
        try:
            bars = snapshot_for_prefix(lc, source['observation_prefix'], cutoff, months=6, context_hours=1234)
        finally:
            lc.close()
    else:
        feed_root = project_path(source['feed_directory'])
        if not (feed_root/'identity.json').is_file() or not (feed_root/'receipts.sqlite').is_file():
            raise ValueError('Missing original efficiency receipt archive; never recreate')
        feed = ReceiptArchive(feed_root, tuple(source['symbols']))
        try:
            bars = feed.snapshot(cutoff,
                start=pd.Timestamp(cutoff) - pd.DateOffset(months=6) - pd.Timedelta(hours=1234),
                archive_delay_seconds=identity['source']['rolling']['availability_delay_seconds'])
        finally:
            feed.close()
    from .v7_run import _verified_S_receipt_source, V7Config
    from .rolling import RollingSpec
    selected, feed_id = _verified_S_receipt_source(bars, pd.Timestamp(cutoff),
        V7Config(symbols=tuple(source['symbols']), evaluation_mode='joint'),
        RollingSpec(**source['rolling']), source['exposure_path'], source['feed_directory'])
    if feed_id != source['source_id']:
        raise ValueError('Efficiency snapshot changed actual source identity')
    return selected


def _base_packet(identity, context):
    from .known_library import FEATURE_NAMES, library_schema
    from .local_mutation import MonthlyBudget
    from .joint_pool import FrozenPool
    from .program_contract import Program
    from .program_syntax import CompactProgram
    from .rolling import RollingSpec
    from .v7_run import literature_seeds
    spec = RollingSpec(**identity['source']['rolling'])
    prompt_id = PromptIdentity('volatility12h', spec.digest, FrozenPool().digest, fingerprint(library_schema()))
    readout = _pair(identity, 'prepared.json')['contexts'][context]
    parents = [{'parent_id': f'literature{i}', 'program': CompactProgram.from_program(Program.from_dict(p)).to_dict(),
        'evidence': {'evidence_id': f'literature{i}', 'role': 'S', 'task_id': prompt_id.task_id,
            'rolling_id': prompt_id.rolling_id, 'pool_id': prompt_id.pool_id,
            'summary': 'Cited literature seed, not a measured candidate or novelty claim.', 'metrics': {}}}
        for i, p in enumerate(literature_seeds())]
    diagnostics = [{'evidence_id': fingerprint({'context': readout['data_id'], 'row': row}), 'role': 'S',
        'task_id': prompt_id.task_id, 'rolling_id': prompt_id.rolling_id, 'pool_id': prompt_id.pool_id,
        'summary': row['summary'], 'metrics': {'loss': row['loss'], 'support': row['support']}}
        for row in readout['baseline'].get('residual_diagnostics', [])]
    return build_formula_prompt(identity=prompt_id,
        task_semantics='Predict 12h realized volatility; development-only logMSE above frozen K0 pool.',
        feature_schema=[{'name': name, 'unit': 'squared log return' if i < 8 else 'dimensionless',
            'preprocessing': 'raw K0; training-only downstream scale'} for i, name in enumerate(FEATURE_NAMES)],
        input_semantics='Full past 48 hour K0 windows, raw units; gaps restart.', parents=parents,
        budget=MonthlyBudget(identity['actual_month']), diagnostics=diagnostics,
        failure_counts={})


def _prepare(identity, budget):
    from .rolling import RollingSpec, build_rolling_data
    from .v7_run import _baseline_readout, typed_search_data_id
    from .joint_pool import hour_equal_weights
    from .convex_eval import fit_ridge
    contexts = []
    spec = RollingSpec(**identity['source']['rolling'])
    for cutoff in identity['cutoffs']:
        budget.check_time()
        bars = _snapshot(identity, cutoff)
        data = build_rolling_data(bars, spec)
        baseline = _baseline_readout(data, pd.Timestamp(cutoff), budget)
        train = data.train_mask(pd.Timestamp(cutoff))
        budget.consume('solve')
        model = fit_ridge(data.frame.take(train), data.target[train], ridge=spec.ridge,
            sample_weight=hour_equal_weights(data.frame.metadata.loc[train]),
            provenance=data.training_identity(pd.Timestamp(cutoff)))
        contexts.append({'cutoff': cutoff, 'data_id': typed_search_data_id(data, bars, pd.Timestamp(cutoff)),
            'baseline': baseline, 'cold_model': model.to_dict(), 'training': data.training_identity(pd.Timestamp(cutoff))})
    value = {'identity_hash': fingerprint(identity), 'contexts': contexts}
    _pair(identity, 'prepared.json', value)
    return {'prepared_hash': fingerprint(value)}


def _call(identity, mode, context, packet, budget):
    from .v7_agent import CodexBatchBackend, BackendReply
    from .evidence import file_hash
    path = project_path(identity['directory']) / 'requests' / f'{mode}-{context}'
    if _pair_exists(identity, f'requests/{mode}-{context}/request.json'):
        raise ValueError('Original request attempt already consumed; never replay')
    # The irreversible attempt marker precedes even the artificial backend read.
    budget.consume('gpt_attempt')
    began = time.monotonic()
    request = {'identity_hash': fingerprint(identity), 'mode': mode, 'context': context,
        **packet.telemetry(), 'started_at': datetime.now(UTC).isoformat(), 'text': packet.text,
        'semantic_hash': packet.semantic_hash, 'attempt_number': context + 1}
    _pair(identity, f'requests/{mode}-{context}/request.json', request)
    record = {'request_hash': packet.request_hash, 'status': 'failed_consumed',
        'input_tokens': None, 'cached_input_tokens': None, 'output_tokens': None,
        'reasoning_output_tokens': None, 'billed_amount': None,
        'provider_context': None, 'spawn_to_first_response_seconds': None,
        'model_resolved': None, 'usage_raw': None}
    try:
        backend = identity['backend']
        if backend['kind'] == 'codex':
            adapter = CodexBatchBackend(path / 'provider', timeout_seconds=budget.lease.check().remaining_seconds)
            reply = adapter(packet, model='gpt-6-astra', reasoning='high')
        else:
            fixture = project_path(backend['path'])
            if file_hash(fixture) != backend['sha256']:
                raise ValueError('Original artificial response corpus changed')
            row = json.loads(fixture.read_text('utf-8'))[f'{mode}-{context}']
            if row.get('failure'):
                raise ValueError('ExplicitArtificialBackendFailure')
            reply = BackendReply(json.dumps(row['payload']), None, 'artificial-no-provider', VERSION)
        budget.check_time()
        if type(reply) is not BackendReply:
            raise TypeError('Only original typed provider response accepted')
        _pair(identity, f'requests/{mode}-{context}/response.json', {'payload': reply.payload})
        record.update(status='response_received', model_resolved=reply.model_resolved, usage_raw=reply.usage)
        for key in ('input_tokens', 'cached_input_tokens', 'output_tokens', 'reasoning_output_tokens'):
            value = (reply.usage or {}).get(key)
            record[key] = value if type(value) is int and value >= 0 else None
        return reply.payload
    except Exception as exc:
        record['error_type'] = type(exc).__name__
        raise
    finally:
        originals = {}
        if identity['backend']['kind'] == 'codex':
            attempts = sorted((path/'provider').glob('*/attempt.json'))
            if len(attempts) > 1:
                raise ValueError('An original efficiency request cannot contain backend retries')
            for attempt in attempts:
                raw_attempt = json.loads(attempt.read_text('utf-8'))
                if raw_attempt['request_hash'] != packet.request_hash:
                    raise ValueError('Original backend attempt changed its request identity')
                # Invalid provider replies can still report real token usage.
                # Preserve it even when normalization raised before BackendReply.
                for key in ('input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens',
                            'model_resolved','usage_raw','spawn_to_first_response_seconds','backend_version'):
                    record[key] = raw_attempt.get(key)
                record['backend_elapsed_seconds'] = raw_attempt.get('elapsed_seconds')
                allowed = ('stdin.txt','response-schema.json','attempt.json','events.jsonl',
                           'stderr.log','response-wire.json','response.json')
                for name in allowed:
                    artifact = attempt.parent/name
                    if artifact.is_file():
                        if artifact.stat().st_size > 2**20:
                            raise ValueError('Original backend evidence exceeds its bounded wire allowance')
                        originals[str(artifact.relative_to(path))] = {'sha256': file_hash(artifact),
                                                                    'bytes': artifact.stat().st_size}
        record['original_backend_artifacts'] = originals
        record.update(elapsed_seconds=time.monotonic() - began, finished_at=datetime.now(UTC).isoformat())
        _pair(identity, f'requests/{mode}-{context}/telemetry.json', record)


def _family(identity, mode, budget):
    from .program_syntax import strict_json
    budget.consume('family')
    family, outputs, seen = identity['family_contract'], [], set()
    for context in range(2):
        packet = packet_for(mode, _base_packet(identity, context), family=family)
        rows, error = [], None
        try:
            payload = _call(identity, mode, context, packet, budget)
            raw = strict_json(payload)
            concrete = raw.get('programs') if isinstance(raw, dict) else None
            # Invalid replies consume at least one potential AST, and never >16.
            debit = max(1, min(16, len(concrete) if isinstance(concrete, list) else 1))
            budget.consume('ast', debit)
            batch, semantic = decode_response(payload, packet, family=family)
            if semantic != family:
                raise ValueError('Reply changed the prospectively shared semantic contract')
            rows = materialize(batch, mode, seen, reserve=lambda: budget.consume('ast'))
        except Exception as exc:
            budget.check_time()
            error = type(exc).__name__ + ':' + str(exc)[:180]
        # Exact formulas repeated in the second S context are measured there,
        # but never counted again as a unique formula/family numerator.
        outputs.append({'context': context, 'materializations': rows, 'selected': _selected(rows), 'error': error})
    value = {'identity_hash': fingerprint(identity), 'mode': mode, 'family': family, 'contexts': outputs}
    _pair(identity, f'families/{mode}.json', value)
    return {'family_hash': fingerprint(value)}


def _candidate(identity, mode, position, budget):
    from .rolling import RollingSpec, build_rolling_data
    from .joint_pool import FrozenPool, PoolMember, evaluate_candidate, DevelopmentPoolArchive, stability_select
    from .v7_s_controller import candidate_solve_bound
    from .v7_run import typed_search_data_id
    family = _pair(identity, f'families/{mode}.json')
    context, within = divmod(position, 4)
    candidates = family['contexts'][context]['selected']
    budget.consume('f')
    if within >= len(candidates):
        return {'status': 'unused_nontransferable', 'reason': 'no_original_candidate'}
    row = candidates[within]
    pool = FrozenPool()
    required = candidate_solve_bound(pool, parameterized=bool(row['program'].get('parameters')))
    if required > budget.record['counts']['solve']:
        return {'status': 'unfunded_fixed_slot', 'reason': 'unchanged_complete_profile_exceeds_original_slot'}
    cutoff = pd.Timestamp(identity['cutoffs'][context])
    bars = _snapshot(identity, cutoff)
    data = build_rolling_data(bars, RollingSpec(**identity['source']['rolling']))
    data_id = typed_search_data_id(data, bars, cutoff)
    if data_id != _pair(identity, 'prepared.json')['contexts'][context]['data_id']:
        raise ValueError('Original efficiency S input changed before F')
    member = PoolMember(row['program'], lookback=48,
        parameter_mode='profile' if row['program'].get('parameters') else 'fixed', family=family['family']['family_id'])
    ctx = DevelopmentPoolArchive.context(pool, cutoff=cutoff.isoformat(), data_hash=data_id,
        protocol_hash=fingerprint(protocol()), source_hash=identity['source_hash'])
    result = evaluate_candidate(data, cutoff, pool, member, budget=budget, context=ctx)
    _pair(identity, f'numeric/{mode}-{position}.json', result)
    return {'status': 'scored', 'formula_id': row['digest'], 'context': context,
            'result_hash': fingerprint(result),
            'family_semantic_hash': fingerprint({k:v for k,v in family['family'].items() if k != 'family_id'}),
            'F_gain': result['F']['Pv']['gain'], 'P_gain': result['P']['gain'], 'S_gain': result['S']['Pv']['gain'],
            'descriptive_F_eligible': bool(stability_select([result], pool)['selected_ids']),
            'V_status': 'not_executed_no_additional_allocation'}


def _worker(directory, slot_index, reservation_id):
    from .neural_forward_ticket import open_existing_authority
    identity = _read(project_path(directory) / 'identity.json')
    authority = open_existing_authority(identity['authority_directory'])
    record = authority.get(reservation_id)
    lease = FrozenWorkLease.from_dict(record['lease'])
    with lease.as_worker():
        identity, _ = _identity(directory, authority)
        validate_allocation(authority, record['work_key'], record['binding'], lease, record['counts'], current=False)
        if record['binding']['slot'] != slot_index or record['status'] != 'running':
            raise ValueError('Fixed efficiency child lost original prepaid work')
        budget = _Counter(record)
        key = schedule()[slot_index]['key']
        if key == 'prepare':
            result = _prepare(identity, budget)
        elif key in MODES:
            result = _family(identity, key, budget)
        else:
            mode, _, position = key.split('-')
            result = _candidate(identity, mode, int(position), budget)
        # Complete original source/exposure validation belongs to this paid lease.
        _identity(directory, authority)
        budget.check_time()
        _pair(identity, f'worker/{slot_index:02d}.json', {'reservation_id': reservation_id,
            'lease_hash': lease.digest, 'result': result, 'actual_counts': budget.used})
        budget.check_time()


class PromptEfficiencyExperiment:
    @classmethod
    def freeze(cls, directory, *, authority, runtime_directory, cutoffs, family_contract, artificial_responses=None,
               engineering_archive=None):
        """Freeze and claim the actual month before reading/fitting either context."""
        from .s_outer_context import original_clock
        clock = original_clock(7200)  # First operation before wallet/contract/source IO; never reset.
        from .monthly_work_authority import MonthlyWorkAuthority
        from .neural_forward_ticket import _LifecycleReader
        from .quarantined_observation import current_binding
        from .evidence import file_hash
        if isinstance(authority, (str, Path)):
            from .neural_forward_ticket import open_existing_authority
            authority = open_existing_authority(authority)
            _check_experiment_clock({'clock': clock})
        if type(authority) is not MonthlyWorkAuthority or authority.manifest['partition_policy'] != frozen_policy('standard'):
            raise TypeError('Q modes require the single actual standard partitioned wallet')
        authority.verify()
        _check_experiment_clock({'clock': clock})
        if isinstance(family_contract, (str, Path)):
            from .program_syntax import strict_json
            contract_path = project_path(family_contract)
            if not contract_path.is_file() or contract_path.stat().st_size > 100_000:
                raise ValueError('Bounded original family contract JSON required')
            family_contract = strict_json(contract_path.read_text('utf-8'), max_bytes=100_000)
            _check_experiment_clock({'clock': clock})
        values = [pd.Timestamp(x) for x in cutoffs]
        if (len(values) != 2 or len(set(values)) != 2 or values != sorted(values)
                or any(x.tzinfo is None or x.utcoffset().total_seconds() != 0
                       or x != x.normalize().replace(day=1) for x in values)):
            raise ValueError('Preregister exactly two increasing distinct UTC monthly S cutoffs')
        lc = _LifecycleReader(runtime_directory)
        try:
            prefix = current_binding(lc)
            if any(x > lc.now() for x in values):
                raise ValueError('Future S context is not available')
        finally:
            lc.close()
        source = _source(authority, runtime_directory, prefix=prefix, current=True)
        if source['rolling']['task'] != 'volatility' or source['rolling']['horizon_hours'] != 12:
            raise ValueError('Efficiency protocol isolates the actual volatility12h task')
        cutoffs = [x.isoformat() for x in values]
        archive_reference = None
        if engineering_archive is not None:
            from .efficiency_archive_input import verify
            archive_path = project_path(engineering_archive)
            archive_reference = {'path': str(archive_path), 'sha256': file_hash(archive_path)}
            archive_value = verify(archive_reference, source, authority)
            if archive_value['cutoffs'] != cutoffs:
                raise ValueError('Archive experiment changed original two S cutoffs')
        _exposure(source, cutoffs)
        from .local_mutation import ProgramTemplate
        from .known_library import FEATURE_NAMES
        from .prompt_builder import _text
        from .local_mutation import _id
        if type(family_contract) is not dict or set(family_contract) != set(FAMILY_FIELDS):
            raise ValueError('Prospectively supply one complete common semantic family contract')
        family_contract = json.loads(json.dumps(family_contract, allow_nan=False))
        _id(family_contract['family_id'])
        for name in ('hypothesis','mechanism','competing_explanation','falsification'):
            _text(family_contract[name], name, 1200)
        if family_contract['parent_ids'] not in (['literature0'], ['literature1'], ['literature0','literature1']):
            raise ValueError('Common contract parents must be the exact cited canonical literature seeds')
        template = ProgramTemplate.from_dict(family_contract['template'])
        from .program_syntax import CompactProgram
        if CompactProgram.from_dict(template.to_dict()['base']).to_program().feature_names != tuple(FEATURE_NAMES):
            raise ValueError('Common family template changed original K0 feature identity')
        family_contract['template'] = template.to_dict()
        backend = {'kind': 'codex'}
        if artificial_responses is not None:
            path = project_path(artificial_responses)
            if not path.is_file() or path.stat().st_size > 2**20:
                raise ValueError('Bounded explicit artificial response corpus required')
            backend = {'kind': 'artificial', 'path': str(path), 'sha256': file_hash(path)}
        directory = project_path(directory)
        month = clock['issuer']['wall_started_at'][:7]
        root = authority.directory / 'prompt_efficiency' / month
        identity = {'version': VERSION, 'protocol': protocol(), 'directory': str(directory), 'root': str(root),
            'authority_directory': str(authority.directory), 'authority_hash': authority.manifest_hash,
            'actual_month': month, 'clock': clock, 'cutoffs': cutoffs, 'source': source,
            'source_hash': fingerprint(authority.manifest['source_files']), 'backend': backend,
            'family_contract': family_contract, 'engineering_archive': archive_reference}
        with FileLock(str(root) + '.lock', timeout=0):
            if root.exists() or directory.exists():
                raise ValueError('Original efficiency opportunity exists; inspect only, no second directory or resume')
            # A cheap precheck does not replace the authoritative same-transaction
            # partition charge below; a concurrent ordinary S owner still wins.
            if authority.status(month)['owners'].get('S') is not None:
                raise ValueError('Actual UTC month already owns its S opportunity')
            _pair(identity, 'identity.json', identity)
            _exposure(source, cutoffs, record=True, experiment=fingerprint(identity))
            _check_experiment_clock(identity)
        obj = cls.__new__(cls)
        obj.identity, obj.authority = identity, authority
        return obj

    def _run_slot(self, index, boundary):
        from .process_policy import isolated_process_kwargs, restricted_environment
        identity, authority = _identity(self.identity['directory'], self.authority)
        remaining = _check_experiment_clock(identity)
        slot = schedule()[index]
        if (_pair_exists(identity, f'intents/{index:02d}.json')
                or _pair_exists(identity, f'results/{index:02d}.json')):
            raise ValueError('Original slot attempt is consumed; never replay even an orphan')
        if remaining < slot['seconds']:
            raise TimeoutError('Original efficiency remaining time cannot fund a complete next slot')
        lease = FrozenWorkLease.issue(slot['seconds'])
        intent = {'identity_hash': fingerprint(identity), 'slot': index, 'work': slot, 'lease': lease.to_dict()}
        _pair(identity, f'intents/{index:02d}.json', intent)
        binding = {'directory': identity['directory'], 'identity_hash': fingerprint(identity),
                   'slot': index, 'intent_hash': fingerprint(intent)}
        record = authority._reserve_with_lease(f'prompt_efficiency:{fingerprint(identity)}:{index}',
            counts=slot['counts'], binding=binding, lease=lease)
        process = None
        result = {'status': 'failed_consumed', 'slot': index, 'reservation_id': record['reservation_id']}
        try:
            kwargs = isolated_process_kwargs()
            if os.name == 'nt':
                kwargs['creationflags'] |= 0x4  # Assign original Job before any child can fork.
            env = restricted_environment()
            env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            # Avoid the Windows venv redirector becoming an unregistered parent.
            # This branch is artificial engineering only; real runs require Linux.
            executable = sys._base_executable if os.name == 'nt' else sys.executable
            if os.name == 'nt':
                sites = [str(Path(x).resolve()) for x in sys.path if x and Path(x).is_dir()
                         and Path(x).name == 'site-packages']
                env['PYTHONPATH'] = os.pathsep.join([str(project_path('.')), *sites])
            with (project_path(identity['directory']) / f'worker-{index:02d}.log').open('xb') as log:
                command = [executable, '-B', '-m', 'factorlab.discovery.prompt_efficiency', '_worker',
                    '--directory', identity['directory'], '--slot', str(index), '--reservation', record['reservation_id']]
                # Fixed worker pauses until canonical running reservation exists.
                authority.start(record['reservation_id'], {'launcher_pid': os.getpid(), 'fixed_worker': VERSION})
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                    cwd=project_path('.'), env=env, **kwargs)
                boundary.attach(process)
                while process.poll() is None:
                    lease.assert_parent_alive(); _check_experiment_clock(identity)
                    time.sleep(.02)
                lease.assert_parent_alive(); _check_experiment_clock(identity)
            if process.returncode != 0:
                raise ValueError('Fixed efficiency worker failed')
            expected_members = {os.getpid()} if sys.platform == 'linux' else set()
            if set(boundary.members()) != expected_members:
                raise ValueError('Original efficiency container retains a descendant')
            value = _pair(identity, f'worker/{index:02d}.json')
            if value['reservation_id'] != record['reservation_id'] or value['lease_hash'] != lease.digest:
                raise ValueError('Efficiency completion changed original reservation')
            authority.complete(record['reservation_id'], result={'artifact_hash': fingerprint(value)},
                               actual_counts=value['actual_counts'])
            lease.assert_parent_alive(); _check_experiment_clock(identity)
            result.update(status='awaiting_witness', output=value['result'])
        except BaseException as exc:
            if process is not None and (process.poll() is None or
                    set(boundary.members()) != ({os.getpid()} if sys.platform == 'linux' else set())):
                boundary.stop()
            actual = authority.get(record['reservation_id'])
            if actual['status'] in {'reserved', 'running'}:
                authority.fail_consumed(record['reservation_id'], type(exc).__name__)
            result['error_type'] = type(exc).__name__
        _pair(identity, f'results/{index:02d}.json', result)
        # The witness samples only a commit which is already complete. Late
        # result publication never upgrades a completed wallet row into evidence.
        if result['status'] == 'awaiting_witness':
            lease.assert_parent_alive(); _check_experiment_clock(identity)
            _pair(identity, f'witnesses/{index:02d}.json', {'result_hash': fingerprint(result),
                'observed_at': datetime.now(UTC).isoformat(), 'observed_monotonic': time.monotonic(),
                'lease_hash': lease.digest, 'issuer': identity['clock']['issuer']})
        return result

    def _confirm_slot(self, index):
        """Confirmation only: read completed witness, then check original clock.

        This cannot fit, call a provider, reserve or renew a lease. Its own
        publication is not claimed to be the paid artifact being witnessed.
        """
        identity = self.identity
        if _pair_exists(identity, f'confirmations/{index:02d}.json'):
            raise ValueError('Original efficiency confirmation already consumed')
        result = _pair(identity, f'results/{index:02d}.json')
        record = self.authority.get(result['reservation_id'])
        lease = FrozenWorkLease.from_dict(record['lease'])
        confirmation = {'slot': index, 'status': 'failed_consumed', 'result_hash': fingerprint(result)}
        try:
            witness = _pair(identity, f'witnesses/{index:02d}.json')
            if (result['status'] != 'awaiting_witness' or witness['result_hash'] != fingerprint(result)
                    or record['status'] != 'completed'
                    or record['binding']['identity_hash'] != fingerprint(identity)
                    or os.getpid() != identity['clock']['issuer']['parent_pid']):
                raise ValueError('Efficiency original pending result/witness differs')
            lease.assert_parent_alive(); _check_experiment_clock(identity)
            confirmation.update(status='completed', witness_hash=fingerprint(witness),
                observed_at=datetime.now(UTC).isoformat(), observed_monotonic=time.monotonic(),
                issuer=identity['clock']['issuer'])
        except (ValueError, RuntimeError, TimeoutError) as exc:
            confirmation['reason'] = type(exc).__name__
        _pair(identity, f'confirmations/{index:02d}.json', confirmation)
        return confirmation

    def run(self):
        """One original issuer, bounded fixed workers, no restart/extra attempts."""
        from scripts.v7_guard.lease_enforcer import _LinuxService, _WindowsJob
        identity = self.identity
        _check_experiment_clock(identity)
        if identity['clock']['issuer']['parent_pid'] != os.getpid():
            raise ValueError('Original efficiency issuer cannot be replaced')
        if sys.platform != 'linux' and identity['backend']['kind'] != 'artificial':
            raise ValueError('Real Codex experiment requires the original dedicated guarded Linux service')
        boundary = _LinuxService() if sys.platform == 'linux' else _WindowsJob()
        stopped = threading.Event()
        def original_outer_watch():
            while not stopped.wait(.05):
                try:
                    _check_experiment_clock(identity)
                except BaseException:
                    boundary.stop()
                    return
        watcher = threading.Thread(target=original_outer_watch, daemon=True, name='original-efficiency-outer-clock')
        watcher.start()
        try:
            with FileLock(str(project_path(identity['root']) / 'run.lock'), timeout=0):
                if _pair_exists(identity, 'run_attempt.json'):
                    raise ValueError('Original efficiency run already consumed; read-only report required')
                _pair(identity, 'run_attempt.json', {'identity_hash': fingerprint(identity), 'pid': os.getpid()})
                _pair(identity, 'container.json', {'identity_hash': fingerprint(identity),
                    'platform': sys.platform, 'issuer': identity['clock']['issuer'],
                    'kernel': ({'cgroup':str(boundary.path), 'inode':boundary.inode, 'boot':boundary.boot,
                                'cpu_fraction':.25,'memory_max_bytes':768*2**20,'swap_bytes':0,'pids':64}
                               if sys.platform == 'linux' else {'kind':'Windows-Job-engineering-only'}),
                    'genuine_Linux_acceptance_claimed':False})
                for index, slot in enumerate(schedule()):
                    self._run_slot(index, boundary)
                    confirmed = self._confirm_slot(index)
                    if index == 0 and confirmed['status'] != 'completed':
                        break
                    if slot['key'] in MODES and confirmed['status'] != 'completed':
                        # Missing family closes its eight fixed slots; they stay
                        # in the complete denominator and cannot become new calls.
                        break
                return inspect(identity['directory'])
        finally:
            stopped.set(); watcher.join(timeout=1)
            if set(boundary.members()) != ({os.getpid()} if sys.platform == 'linux' else set()):
                boundary.stop()
            boundary.close()


def _verify_worker_artifact(identity, key, worker):
    """Reconstruct payload/materialization identity, not caller success booleans."""
    if key == 'prepare':
        value = _pair(identity, 'prepared.json')
        if (worker['result'] != {'prepared_hash': fingerprint(value)}
                or value['identity_hash'] != fingerprint(identity)
                or [x['cutoff'] for x in value['contexts']] != identity['cutoffs']
                or worker['actual_counts']['solve'] != 14
                or any(x['training']['fit'] != 'cold' for x in value['contexts'])):
            raise ValueError('Efficiency preparation lost its two original cold contexts')
        return
    if key in MODES:
        family, seen = identity['family_contract'], set()
        value = _pair(identity, f'families/{key}.json')
        if (worker['result'] != {'family_hash': fingerprint(value)}
                or value['identity_hash'] != fingerprint(identity) or value['mode'] != key
                or len(value['contexts']) != 2):
            raise ValueError('Efficiency family artifact differs from original completed work')
        for i, context in enumerate(value['contexts']):
            packet = packet_for(key, _base_packet(identity, i), family=family)
            request = _pair(identity, f'requests/{key}-{i}/request.json')
            if (context['context'] != i or request['text'] != packet.text
                    or request['request_hash'] != packet.request_hash
                    or request['semantic_hash'] != packet.semantic_hash):
                raise ValueError('Efficiency prompt no longer matches its original S context and family')
            _verify_telemetry(identity, key, i, packet)
            expected = []
            if _pair_exists(identity, f'requests/{key}-{i}/response.json'):
                response = _pair(identity, f'requests/{key}-{i}/response.json')['payload']
                try:
                    batch, semantic = decode_response(response, packet, family=family)
                except ValueError:
                    if context['error'] is None:
                        raise ValueError('Invalid original efficiency reply was marked valid') from None
                else:
                    if semantic != family:
                        raise ValueError('Reply differs from prospectively shared semantic contract')
                    expected = materialize(batch, key, seen, reserve=lambda: None)
                    if context['error'] is not None:
                        raise ValueError('Canonical family result contradicts its original valid reply')
            elif context['error'] is None:
                raise ValueError('Efficiency family invents a missing provider response')
            if context['materializations'] != expected or context['selected'] != _selected(expected):
                raise ValueError('Efficiency finite AST census or F selection changed')
        if value['family'] != family or worker['actual_counts']['gpt_attempt'] != 2:
            raise ValueError('Efficiency family identity or two-attempt census changed')
        return
    if worker['result'].get('status') == 'scored':
        from .joint_pool import FrozenPool, PoolMember, DevelopmentPoolArchive
        mode, _, text = key.split('-'); position = int(text)
        context, within = divmod(position, 4)
        row = _pair(identity, f'families/{mode}.json')['contexts'][context]['selected'][within]
        result = _pair(identity, f'numeric/{mode}-{position}.json')
        pool = FrozenPool()
        expected_context = DevelopmentPoolArchive.context(pool, cutoff=identity['cutoffs'][context],
            data_hash=_pair(identity, 'prepared.json')['contexts'][context]['data_id'],
            protocol_hash=fingerprint(protocol()), source_hash=identity['source_hash'])
        member = PoolMember(row['program'], lookback=48,
            parameter_mode='profile' if row['program'].get('parameters') else 'fixed',
            family=_pair(identity, f'families/{mode}.json')['family']['family_id'])
        from .joint_pool import stability_select
        if (worker['result']['result_hash'] != fingerprint(result) or result['context'] != expected_context
                or worker['result']['formula_id'] != row['digest'] or worker['result']['context'] != context
                or worker['result']['family_semantic_hash'] != fingerprint({k:v for k,v in
                    _pair(identity, f'families/{mode}.json')['family'].items() if k != 'family_id'})
                or result['role'] != 'S' or len(result['folds']) != 3
                or result['candidate_id'] != member.member_id or worker['actual_counts']['f'] != 1
                or worker['result']['F_gain'] != result['F']['Pv']['gain']
                or worker['result']['P_gain'] != result['P']['gain']
                or worker['result']['S_gain'] != result['S']['Pv']['gain']
                or worker['result']['descriptive_F_eligible'] != bool(stability_select([result], pool)['selected_ids'])):
            raise ValueError('Efficiency numerical evidence changed selected learner, context or full folds')


def _verify_telemetry(identity, mode, context, packet=None):
    from .evidence import file_hash
    from .v7_agent import normalize_provider_batch
    row = _pair(identity, f'requests/{mode}-{context}/telemetry.json')
    request = _pair(identity, f'requests/{mode}-{context}/request.json')
    if row['request_hash'] != request['request_hash']:
        raise ValueError('Original request/telemetry hash differs')
    path = project_path(identity['directory'])/'requests'/f'{mode}-{context}'
    originals = row['original_backend_artifacts']
    if identity['backend']['kind'] == 'artificial':
        if originals or any(row[k] is not None for k in (
                'input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens')):
            raise ValueError('Artificial response cannot invent provider token evidence')
        return row
    for relative, expected in originals.items():
        artifact = project_path(path/relative)
        if (not artifact.is_relative_to(path/'provider') or not artifact.is_file()
                or artifact.stat().st_size != expected['bytes'] or file_hash(artifact) != expected['sha256']):
            raise ValueError('Original efficiency backend raw evidence changed')
    attempts = sorted((path/'provider').glob('*/attempt.json'))
    if len(attempts) > 1 or (originals and len(attempts) != 1):
        raise ValueError('Efficiency backend attempt census differs')
    if attempts:
        raw = json.loads(attempts[0].read_text('utf-8'))
        if (raw['request_hash'] != request['request_hash']
                or (attempts[0].parent/'stdin.txt').read_text('utf-8') != request['text']
                or raw['model_requested'] != 'gpt-6-astra' or raw['reasoning'] != 'high'):
            raise ValueError('Efficiency backend changed exact application prompt/model')
        if row['status'] == 'response_received':
            response = _pair(identity, f'requests/{mode}-{context}/response.json')['payload']
            if (raw['status'] != 'completed'
                    or normalize_provider_batch((attempts[0].parent/'response-wire.json').read_text('utf-8')) != response
                    or raw['usage_raw'] != row['usage_raw'] or raw['model_resolved'] != row['model_resolved']):
                raise ValueError('Efficiency result is not its original provider wire response')
        for key in ('input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens',
                    'model_resolved','usage_raw','spawn_to_first_response_seconds','backend_version'):
            if raw.get(key) != row.get(key):
                raise ValueError('Efficiency token report differs from original backend telemetry')
    elif row['status'] == 'response_received':
        raise ValueError('Efficiency response lacks original provider backend identity')
    return row


def _quality_metrics(rows, mode, token_sum):
    """Descriptive measured S gains, separate from syntactic materializations.

    All records have passed original numerical/witness verification in inspect.
    This adds reporting fields only; selection gates and allocations are intact.
    """
    import math
    formulas, families, stable = set(), set(), set()
    outcomes = {}
    for row in rows:
        if not row['work']['key'].startswith(mode+'-F-'):
            continue
        result = row['result']
        output = result.get('output', {}) if result['status'] == 'completed' else {}
        status = output.get('status', result['status'])
        outcomes[status] = outcomes.get(status, 0)+1
        gain = output.get('S_gain')
        if (output.get('status') == 'scored' and type(gain) in (int, float)
                and math.isfinite(gain) and gain > 0):
            formulas.add(output['formula_id'])
            families.add(output['family_semantic_hash'])
            if output.get('descriptive_F_eligible') is True:
                stable.add(output['formula_id'])
    denominator = token_sum if type(token_sum) is int and token_sum > 0 else None
    return {'S_increment_criterion': 'verified full evaluate_candidate S.Pv.gain > 0; descriptive only',
            'unique_positive_S_increment_formulas': len(formulas),
            'unique_positive_S_increment_families': len(families),
            'unique_positive_S_and_original_stability_formulas': len(stable),
            'positive_S_formulas_per_1000_reported_input_tokens': len(formulas)*1000/denominator if denominator else None,
            'positive_S_families_per_1000_reported_input_tokens': len(families)*1000/denominator if denominator else None,
            'complete_F_outcomes': outcomes, 'complete_F_denominator': 8,
            'qualified_V_count': None, 'noninferiority': 'unknown'}


def inspect(directory):
    """Read originals and recompute all denominators; never create missing work."""
    identity, authority = _identity(directory)
    rows = []
    for index, slot in enumerate(schedule()):
        result = {'status': 'not_attempted', 'slot': index}
        if _pair_exists(identity, f'results/{index:02d}.json'):
            result = _pair(identity, f'results/{index:02d}.json')
            record = authority.get(result['reservation_id'])
            lease = FrozenWorkLease.from_dict(record['lease'])
            validate_allocation(authority, record['work_key'], record['binding'], lease, record['counts'], current=False)
            if result['status'] == 'awaiting_witness':
                result = dict(result, status='unwitnessed_consumed')
                if _pair_exists(identity, f'witnesses/{index:02d}.json'):
                    witness = _pair(identity, f'witnesses/{index:02d}.json')
                    original = _pair(identity, f'results/{index:02d}.json')
                    if (witness['result_hash'] != fingerprint(original) or witness['lease_hash'] != lease.digest
                            or witness['issuer'] != identity['clock']['issuer']
                            or not lease.monotonic_started <= witness['observed_monotonic'] < lease.monotonic_deadline
                            or not pd.Timestamp(lease.wall_started_at) <= pd.Timestamp(witness['observed_at']) < pd.Timestamp(lease.wall_deadline)
                            or record['status'] != 'completed'):
                        raise ValueError('Efficiency witness does not attest its original completed work')
                    worker = _pair(identity, f'worker/{index:02d}.json')
                    if (record['result'] != {'artifact_hash': fingerprint(worker)}
                            or original['output'] != worker['result']):
                        raise ValueError('Original numerical/provider work differs from its result')
                    _verify_worker_artifact(identity, slot['key'], worker)
                    if _pair_exists(identity, f'confirmations/{index:02d}.json'):
                        confirmation = _pair(identity, f'confirmations/{index:02d}.json')
                        if confirmation['result_hash'] != fingerprint(original) or confirmation['slot'] != index:
                            raise ValueError('Efficiency successor confirmation differs from original result')
                        if confirmation['status'] == 'completed':
                            if (confirmation['witness_hash'] != fingerprint(witness)
                                    or confirmation['issuer'] != identity['clock']['issuer']
                                    or not lease.monotonic_started <= confirmation['observed_monotonic'] < min(
                                        lease.monotonic_deadline, identity['clock']['monotonic_deadline'],
                                        identity['clock']['issuer']['monotonic_started']+protocol()['S_elapsed_seconds_ceiling'])
                                    or not pd.Timestamp(lease.wall_started_at) <= pd.Timestamp(confirmation['observed_at']) < min(
                                        pd.Timestamp(lease.wall_deadline), pd.Timestamp(identity['clock']['wall_deadline']),
                                        pd.Timestamp(identity['clock']['issuer']['wall_started_at'])+
                                        pd.Timedelta(seconds=protocol()['S_elapsed_seconds_ceiling']))):
                                raise ValueError('Efficiency witness completed after the original paid deadline')
                            result['status'] = 'completed'
                        elif confirmation['status'] == 'failed_consumed':
                            result['status'] = 'late_or_invalid_consumed'
                        else:
                            raise ValueError('Unknown efficiency confirmation status')
        elif _pair_exists(identity, f'intents/{index:02d}.json'):
            result = {'status': 'orphan_consumed', 'slot': index}
        rows.append({'work': slot, 'result': result})
    modes = {}
    for mode in MODES:
        calls, families, formulas, input_tokens, elapsed = [], set(), set(), [], 0.
        family_index = next(i for i, x in enumerate(schedule()) if x['key'] == mode)
        for context in range(2):
            stem = f'requests/{mode}-{context}'
            if _pair_exists(identity, stem + '/request.json'):
                request = _pair(identity, stem + '/request.json')
                intent = _pair(identity, f'intents/{family_index:02d}.json')
                key = f'prompt_efficiency:{fingerprint(identity)}:{family_index}'
                reservation = authority.get(fingerprint({'authority':authority.manifest_hash,'work_key':key}))
                if (request['identity_hash'] != fingerprint(identity) or request['mode'] != mode
                        or request['context'] != context or reservation['lease'] != intent['lease']
                        or request['request_hash'] != hashlib.sha256(json.dumps({'text':request['text'],
                            'model':'gpt-6-astra','reasoning':'high'}, sort_keys=True, separators=(',',':'),
                            ensure_ascii=False).encode()).hexdigest()):
                    raise ValueError('Efficiency request lacks its actual original paid family dispatch')
                row = {'status': 'orphan_consumed', 'input_tokens': None}
                if _pair_exists(identity, stem + '/telemetry.json'):
                    row = _verify_telemetry(identity, mode, context)
                    if row['request_hash'] != request['request_hash']:
                        raise ValueError('Efficiency request/telemetry mismatch')
                calls.append({**row, 'stdin_utf8_bytes': len(request['text'].encode()),
                    'request_file_bytes': (project_path(identity['directory'])/stem/'request.json').stat().st_size,
                    'template_version':request['template_version'], 'prefix_hash':request['prefix_hash'],
                    'semantic_hash':request['semantic_hash']})
                input_tokens.append(row['input_tokens'])
                elapsed += row.get('elapsed_seconds', 0.)
        family_row = rows[family_index]
        if family_row['result']['status'] == 'completed':
            value = _pair(identity, f'families/{mode}.json')
            if any(c['selected'] for c in value['contexts']):
                families.add(fingerprint({k: v for k, v in value['family'].items() if k != 'family_id'}))
            for context in value['contexts']:
                formulas.update(x['digest'] for x in context['materializations'] if x['program'] is not None)
        token_sum = sum(input_tokens) if len(input_tokens) == 2 and all(type(x) is int for x in input_tokens) else None
        spans = []
        for index, work in enumerate(rows):
            if work['work']['key'] == mode or work['work']['key'].startswith(mode+'-F-'):
                if work['result']['status'] != 'completed':
                    spans.append(None)
                else:
                    intent = _pair(identity, f'intents/{index:02d}.json')
                    confirmation = _pair(identity, f'confirmations/{index:02d}.json')
                    spans.append(confirmation['observed_monotonic'] - intent['lease']['monotonic_started'])
        mode_seconds = sum(spans) if all(x is not None for x in spans) else None
        modes[mode] = {'planned_calls': 2, 'actual_attempts': len(calls), 'calls': calls,
            'unique_families': len(families), 'unique_formulas': len(formulas), 'planned_F': 8,
            'unique_families_and_formulas_basis': 'syntactically valid only; not measured S support',
            'F': [x['result'] for x in rows if x['work']['key'].startswith(mode + '-F-')],
            'reported_input_tokens': token_sum,
            'families_per_1000_reported_input_tokens': len(families)*1000/token_sum if token_sum else None,
            'backend_attempt_elapsed_seconds': elapsed if len(calls) == 2 else None,
            'families_per_backend_attempt_second': len(families)/elapsed if elapsed > 0 and len(calls) == 2 else None,
            'completed_original_slot_span_seconds': mode_seconds,
            'families_per_original_slot_span_second': len(families)/mode_seconds if mode_seconds else None}
        modes[mode].update(_quality_metrics(rows, mode, token_sum))
    ratios = []
    for i in range(2):
        if all(len(modes[m]['calls']) > i for m in ('Q0','Q1')):
            denominator = modes['Q0']['calls'][i]['stdin_utf8_bytes']
            ratios.append(modes['Q1']['calls'][i]['stdin_utf8_bytes'] / denominator if denominator else None)
        else:
            ratios.append(None)
    return {'version': VERSION, 'identity_hash': fingerprint(identity), 'source_hash': identity['source_hash'],
        'actual_month': identity['actual_month'], 'wallet': authority.status(identity['actual_month']),
        'backend': identity['backend']['kind'],
        'input_kind': 'historical_archive_engineering_only' if identity.get('engineering_archive') else 'original_receipt_or_quarantined_source',
        'engineering_archive': identity.get('engineering_archive'),
        'paid_dispatch_attempts': sum(x['actual_attempts'] for x in modes.values()),
        'real_provider_calls_executed': None if identity['backend']['kind'] == 'codex' else 0,
        'provider_network_requests': None if identity['backend']['kind'] == 'codex' else 0,
        'work': rows, 'modes': modes, 'complete_F_denominator': 24, 'complete_call_denominator': 6,
        'Q1_to_Q0_observed_stdin_byte_ratios': ratios,
        'token_ratio_basis': 'provider-reported input tokens; output already includes any reported reasoning, never summed twice',
        'actual_V_count': 0, 'qualified_V_count': None,
        'serving_eligible': False, 'formal_claims_enabled': False, 'noninferiority': 'not_established',
        'legacy_hidden_provider_context': None, 'readback_only': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('--directory', required=True)
    run.add_argument('--authority', required=True)
    run.add_argument('--runtime', required=True)
    run.add_argument('--cutoff', action='append', required=True)
    run.add_argument('--artificial-responses')
    run.add_argument('--engineering-archive', help='Original explicit historical archive S-only binding; never formal or live receipt evidence')
    run.add_argument('--family-contract', required=True,
                     help='Explicit preregistered common semantic/template JSON; same in every mode/context')
    report = sub.add_parser('inspect'); report.add_argument('--directory', required=True)
    worker = sub.add_parser('_worker'); worker.add_argument('--directory', required=True)
    worker.add_argument('--slot', type=int, required=True); worker.add_argument('--reservation', required=True)
    args = parser.parse_args(argv)
    if args.command == '_worker':
        _worker(args.directory, args.slot, args.reservation)
        return
    if args.command == 'inspect':
        value = inspect(args.directory)
    else:
        value = PromptEfficiencyExperiment.freeze(args.directory,
            authority=args.authority, runtime_directory=args.runtime,
            cutoffs=args.cutoff, family_contract=args.family_contract,
            artificial_responses=args.artificial_responses, engineering_archive=args.engineering_archive).run()
    print(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == '__main__':
    main()
