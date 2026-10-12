"""Bounded, restartable V7 development construction controller.

This release runs calendar K0 fits, formula search, F/S readouts and frozen
shortlists. It cannot promote an independently confirmed model until the
full-pipeline calibration/forward phase has passed. Nothing trades.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from filelock import FileLock

from factorlab.protocol import fingerprint, project_path
from factorlab.time_utils import calendar_window, utc_index, month_start

from .convex_eval import fit_ridge
from .evidence import file_hash, snapshot_source, source_paths
from .exposure import ExposureLedger
from .factor_pool import FactorPool, evaluate_formula, formula_identity, relative_gain
from .known_library import FEATURE_NAMES, FactorFrame, library_schema
from .local_mutation import BudgetExhausted, MonthlyBudget
from .model_registry import ForecastLedger, ModelRegistry
from .rolling import RollingSpec, build_rolling_data
from .s_data_identity import raw_snapshot_identity, typed_search_data_id
from .v7_storage import immutable_record as immutable_json


@dataclass(frozen=True)
class V7Config:
    symbols: tuple[str, ...] = ('BTCUSDT',)
    search_mode: str = 'local'
    formulas_per_family: int = 8
    lookback: int = 48
    profile_candidates: int = 0
    research_seconds: int = 7200
    model: str = 'gpt-6-astra'
    reasoning: str = 'high'
    formal_confirmation_enabled: bool = False
    evaluation_mode: str = 'single_compat'
    nonlinear_mode: str = 'disabled'
    budget_policy: str = 'standard'

    def __post_init__(self):
        if self.budget_policy not in {'standard', 'controls_third'}:
            raise ValueError('Unknown frozen monthly budget policy')
        if self.budget_policy == 'controls_third' and (
                self.evaluation_mode != 'joint' or self.profile_candidates != 2
                or self.formulas_per_family != 8 or self.research_seconds != 2400
                or self.nonlinear_mode != 'disabled'):
            raise ValueError('Shared controls require one nontransferable third-month envelope')
        if (not 1 <= len(self.symbols) <= (3 if self.evaluation_mode == 'joint' else 1)
                or len(set(self.symbols)) != len(self.symbols)
                or any(not isinstance(s, str) or not s for s in self.symbols)):
            raise ValueError('Initial V7 research pilot freezes exactly one symbol; multi-symbol search is gated')
        if self.evaluation_mode not in {'single_compat', 'joint'}:
            raise ValueError('Unknown evaluation mode')
        if (self.nonlinear_mode not in {'disabled', 'probes', 'isolated'}
                or (self.nonlinear_mode != 'disabled' and self.evaluation_mode != 'joint')):
            raise ValueError('Nonlinear S research requires an explicitly frozen joint configuration')
        if self.search_mode not in {'local', 'gpt', 'random'}:
            raise ValueError('Unknown V7 search mode')
        if type(self.formulas_per_family) is not int or not 1 <= self.formulas_per_family <= 8:
            raise ValueError('Initial pilot admits 1..8 F candidates per family')
        if type(self.lookback) is not int or not 2 <= self.lookback <= 512:
            raise ValueError('Invalid formula lookback')
        if type(self.profile_candidates) is not int or not 0 <= self.profile_candidates <= 6:
            raise ValueError('At most six structures may receive training-only P')
        if type(self.research_seconds) is not int or not 30 <= self.research_seconds <= 7200:
            raise ValueError('Bounded research time required')
        if self.model != 'gpt-6-astra' or self.reasoning != 'high':
            raise ValueError('Changing the proposal model needs a new approved research contract')
        if self.formal_confirmation_enabled:
            raise ValueError('Full-pipeline calibration and Linux/forward gates are not yet accepted')


def monthly_budget(config, month):
    """Immutable nontransferable envelopes, with no unused-credit borrowing."""
    if config.budget_policy == 'standard':
        return MonthlyBudget(month)
    return MonthlyBudget(month, limits={
        'gpt_attempt': 2, 'family': 1, 'ast': 32, 'f': 8, 'solve': 533,
        'family_ast': 32, 'family_f': 8})


def literature_seeds():
    """Two deliberately known mechanisms, never counted as novel discoveries."""
    def program(node):
        return {'version': 'causal-operator-program-v1', 'feature_names': list(FEATURE_NAMES),
                'parameters': {}, 'outputs': [node]}
    feature = lambda name: {'op': 'feature', 'name': name}
    last = lambda arg: {'op': 'last', 'arg': arg}
    return [program(last({'op': 'safe_div', 'left': feature('rv_6'), 'right': feature('rv_24')})),
            program(last({'op': 'sub', 'left': feature('rv_24'), 'right': feature('parkinson_24')}))]


def source_files():
    from .monthly_work_authority import _charge_current_source
    cached = _charge_current_source()
    if cached is not None:
        return cached
    root = project_path('.')
    paths = source_paths(root)
    return {str(path.relative_to(root)).replace('\\', '/'): file_hash(path) for path in sorted(paths)}


def _nonlinear_month(store, month, state, budget, data, cutoff, mode, remaining_seconds, checkpoint):
    """Optional S-only channel; no neural recipe is registered for adoption."""
    from .nonlinear_orchestration import (
        RunStoreBudgetBridge,
        causal_worker_data,
        freeze_epoch,
        input_data_identity,
        prior_gate_artifacts,
        produce_quarterly_artifacts,
        run_isolated_research,
        structure_proposals,
    )
    if state.get('nonlinear_report') is not None:
        return state['nonlinear_report']
    if remaining_seconds < 3600:
        return {'status': 'deferred_monthly_wall_budget', 'automatic_adoption': False}
    state['nonlinear_input_data_id'] = input_data_identity(data, cutoff)
    checkpoint('nonlinear_actual_input_sealed')
    bridge = RunStoreBudgetBridge(store, month, state, budget)
    directory = store.directory / 'nonlinear_epoch'
    epoch = freeze_epoch(directory, bridge, data, cutoff)
    producers = produce_quarterly_artifacts(directory, bridge, data, cutoff, epoch)
    gate = prior_gate_artifacts(directory, bridge, data, cutoff, epoch)
    result = {'status': 'S_channel_complete', 'epoch_id': epoch.digest,
              'producers': producers, 'gate': gate, 'structure_proposals': structure_proposals(epoch),
              'automatic_adoption': False, 'formal_claim': False,
              'recipe_status': 'neural_forward_registry_not_registered'}
    if mode == 'isolated' and gate['status'] == 'research_evidence_verified':
        from .nonlinear_orchestration import epoch_pool
        from .nonlinear_runtime import (
            _capacity,
            _common,
            _context,
            _read,
            _resource,
            _Store,
            verify_gate_artifacts,
        )
        work_data = causal_worker_data(data, cutoff)
        context = _context(work_data, cutoff, epoch_pool(), bridge.source_id)
        resources = _Store(directory)
        eligible, rejected_capacity = [], []
        try:
            for path in sorted((directory / 'receipts').glob('*.json')):
                row = json.loads(path.read_text('utf-8'))
                if row.get('kind') != 'resource' or pd.Timestamp(row['available_at']) > cutoff:
                    continue
                row = _read(resources, path.stem, context, cutoff)
                if row['test_only']:
                    continue
                _resource(row['result'], {'context': _common(context), 'config_id': epoch.config.digest})
                try:
                    _capacity(row['result'], work_data, cutoff)
                except ValueError as exc:
                    rejected_capacity.append({'resource': path.stem, 'reason': str(exc)[:200]})
                    continue
                # Corrupt producer/kernel evidence closes the channel, rather
                # than being silently treated as a small historical workload.
                verify_gate_artifacts(directory, work_data, cutoff, epoch_pool(),
                    {**gate['artifacts'], 'resource': path.stem}, source_id=bridge.source_id, config=epoch.config)
                eligible.append(path.stem)
        finally:
            resources.close()
        result['resource_capacity_rejections'] = rejected_capacity
        result['isolated'] = run_isolated_research(directory, bridge, data, cutoff, epoch,
            resource_receipt_id=eligible[-1] if eligible else None, resource_trial=not eligible)
    state['nonlinear_report'] = result
    checkpoint('nonlinear_S_report_committed')
    return result


def _verify_month_artifacts(directory, month, state):
    if state.get('baseline_model') is not None:
        ModelRegistry(directory / 'models').load(state['baseline_model'])
    for identifier, record in state['pool']['records'].items():
        path = directory / 'results' / f'{month}-{identifier}.json'
        if not path.is_file() or json.loads(path.read_text('utf-8')) != record['result']:
            raise ValueError('V7 candidate result artifact integrity mismatch')
    if state.get('joint_archive') is not None and state.get('status') == 'complete':
        from .joint_pool import DevelopmentPoolArchive
        archive = DevelopmentPoolArchive.load(directory / 'archives' / f'{month}.json')
        if archive.to_dict() != state['joint_archive']:
            raise ValueError('Completed joint archive differs from checkpoint')
        archive.verify_history()
    if state.get('research_memory') is not None and state.get('status') == 'complete':
        saved = json.loads((directory / 'memories' / f'{month}.json').read_text('utf-8'))
        if saved != state['research_memory']:
            raise ValueError('Completed research memory archive differs from checkpoint')
        from .v7_memory import ResearchMemory
        ResearchMemory.from_state(saved).book.verify_history()


def _publish_finalization(directory, month, state, store):
    if state.get('joint_archive'):
        from .joint_pool import DevelopmentPoolArchive
        from .v7_memory import ResearchMemory
        DevelopmentPoolArchive.from_dict(state['joint_archive']).save(directory / 'archives' / f'{month}.json')
        ResearchMemory.from_state(state['research_memory']).save(directory / 'memories' / f'{month}.json')
    immutable_json(directory / 'months' / f'{month}.json', state['summary'])
    state['status'] = 'complete'
    store.save(month, state, 'month_complete')
    return state['summary']


def _prescreen(data, cutoff, programs, lookback, limit, reference_pool=None):
    """Training-feature redundancy hint plus deterministic interaction blind draws.

    It uses no target and cannot establish novelty or statistical significance.
    Near-redundant programs remain eligible for the two blind slots.
    """
    from .causal_numpy import execute_formula
    from .convex_eval import ResidualProjector
    unique = {fingerprint(formula_identity(p, lookback)): p for p in programs}
    if reference_pool is not None:
        from .joint_pool import PoolMember, prescreen_candidates
        members = [PoolMember(p, lookback=lookback) for p in unique.values()]
        result = prescreen_candidates(data, cutoff, reference_pool, members, limit=limit,
                                      blind_spot_slots=min(2, limit))
        selected_ids = set(result['selected_ids'])
        return [m.program for m in members if m.member_id in selected_ids], result['entries']
    if len(unique) <= limit:
        return list(unique.values()), []
    train = data.train_mask(cutoff)
    records = []
    for identifier, program in unique.items():
        try:
            values = execute_formula(program, data.frame, lookback=lookback)
            valid = train & np.isfinite(values)
            if valid.sum() < data.spec.minimum_train_rows:
                raise ValueError('Insufficient prescreen support')
            projection = ResidualProjector.fit(data.frame.take(valid), values[valid])
            records.append({'formula_id': identifier, 'residual_ratio': projection.residual_scale_ratio,
                            'near_zero': projection.near_zero, 'status': 'valid'})
        except ValueError as exc:
            records.append({'formula_id': identifier, 'residual_ratio': None,
                            'status': 'invalid', 'reason': str(exc)[:200]})
    valid = [r for r in records if r['status'] == 'valid']
    ranked = sorted(valid, key=lambda r: (-r['residual_ratio'], r['formula_id']))
    top = ranked[:max(1, limit - 2)]
    remaining = [r for r in valid if r not in top]
    top += sorted(remaining, key=lambda r: r['formula_id'])[:limit - len(top)]
    for record in records:
        record['selected'] = record in top
    return [unique[r['formula_id']] for r in top], records


class RunStore:
    def __init__(self, directory, config: V7Config, spec: RollingSpec, *, mode='research',
                 exposure_path=None, feed_identity=None, work_contract=None):
        self.directory = project_path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.directory / 'state.sqlite', timeout=30)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS months (month TEXT PRIMARY KEY, value TEXT NOT NULL, hash TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, value TEXT NOT NULL, hash TEXT NOT NULL);
        ''')
        if feed_identity and exposure_path is None:
            self.db.close()
            raise ValueError('Live research requires its explicitly shared exposure ledger')
        exposure_path = project_path(exposure_path) if exposure_path is not None else self.directory / 'exposure.sqlite'
        self.exposure = ExposureLedger(exposure_path)
        self.identity = {'version': 'v7-development-controller-v1', 'mode': mode, 'config': asdict(config),
                         'rolling_spec': asdict(spec), 'rolling_hash': spec.digest,
                         'source_files': source_files(), 'formal_claims_enabled': False,
                         'exposure_path': str(exposure_path), 'feed_identity': feed_identity,
                         'scope': 'prediction_and_factor_research_only'}
        if work_contract is not None:
            self.identity['work_contract'] = json.loads(json.dumps(work_contract))
        manifest = self.directory / 'manifest.json'
        if manifest.exists():
            old = json.loads(manifest.read_text('utf-8'))
            if old['hash'] != fingerprint(old['identity']) or fingerprint(old['identity']) != fingerprint(self.identity):
                self.close()
                raise ValueError('Frozen V7 source/config identity changed; use a new run')
        else:
            archived = snapshot_source(self.directory / 'source')
            if archived['files'] != self.identity['source_files']:
                raise ValueError('Source changed while publishing the V7 snapshot')
            immutable_json(manifest, {'identity': self.identity, 'hash': fingerprint(self.identity)})
        self.verify()

    @property
    def spec(self):
        value = dict(self.identity['rolling_spec'])
        value['event_bins'] = tuple(value['event_bins'])
        spec = RollingSpec(**value)
        if spec.digest != self.identity['rolling_hash']:
            raise ValueError('RunStore changed its original rolling specification')
        return spec

    def close(self):
        if hasattr(self, 'exposure'):
            self.exposure.close()
        self.db.close()

    def get(self, month):
        row = self.db.execute('SELECT value,hash FROM months WHERE month=?', (month,)).fetchone()
        if row is None:
            return None
        value = json.loads(row[0])
        if fingerprint(value) != row[1]:
            raise ValueError('V7 monthly state integrity mismatch')
        return value

    def save(self, month, state, event):
        payload_text = json.dumps(state, sort_keys=True, allow_nan=False)
        with self.db:
            row = self.db.execute('SELECT seq,hash FROM events ORDER BY seq DESC LIMIT 1').fetchone()
            record = {'sequence': row[0] + 1 if row else 0, 'previous': row[1] if row else '0' * 64,
                      'month': month, 'event': event, 'state_hash': fingerprint(state)}
            self.db.execute('INSERT OR REPLACE INTO months VALUES (?,?,?)', (month, payload_text, fingerprint(state)))
            self.db.execute('INSERT INTO events VALUES (?,?,?)',
                            (record['sequence'], json.dumps(record, sort_keys=True), fingerprint(record)))

    def verify(self):
        archive = self.directory / 'source'
        archived_manifest = json.loads((archive / 'manifest.json').read_text('utf-8'))
        if archived_manifest != self.identity['source_files']:
            raise ValueError('V7 saved source manifest integrity mismatch')
        for relative, digest in self.identity['source_files'].items():
            path = project_path(archive / relative)
            if not path.is_relative_to(archive.resolve()) or not path.is_file() or file_hash(path) != digest:
                raise ValueError('V7 saved source archive integrity mismatch')
        previous, heads = '0' * 64, {}
        for sequence, text, digest in self.db.execute('SELECT seq,value,hash FROM events ORDER BY seq'):
            event = json.loads(text)
            if event['sequence'] != sequence or event['previous'] != previous or fingerprint(event) != digest:
                raise ValueError('V7 event chain integrity mismatch')
            heads[event['month']], previous = event['state_hash'], digest
        actual_months = set()
        for month, text, digest in self.db.execute('SELECT month,value,hash FROM months'):
            actual_months.add(month)
            if fingerprint(json.loads(text)) != digest or heads.get(month) != digest:
                raise ValueError('V7 state/checkpoint integrity mismatch')
        if actual_months != set(heads):
            raise ValueError('V7 checkpoint month set integrity mismatch')
        return {'event_head': previous, 'months': len(heads)}


class DurableBudget:
    """Persist nonrefundable reservations BEFORE any candidate/provider work."""
    def __init__(self, budget, persist):
        self.budget, self.persist = budget, persist

    def __getattr__(self, name):
        return getattr(self.budget, name)

    def consume(self, *args, **kwargs):
        self.budget.consume(*args, **kwargs)
        self.persist(self.budget.to_dict())

    def register_family(self, *args, **kwargs):
        self.budget.register_family(*args, **kwargs)
        self.persist(self.budget.to_dict())

    def attribute_ast(self, *args, **kwargs):
        self.budget.attribute_ast(*args, **kwargs)
        self.persist(self.budget.to_dict())


def _baseline_readout(data, cutoff, budget):
    from .convex_eval import log_mse
    from .joint_pool import hour_equal_weights
    from .rolling import development_folds
    levels = {'HAR_cal': ('rv_1', 'rv_24', 'rv_168', 'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos'),
              'K0': FEATURE_NAMES}
    records, residual_groups = [], {}
    for fold in development_folds(cutoff, spec=data.spec):
        train = data.train_mask(fold['fit_cutoff'])
        score = data.score_mask(fold['fit_cutoff'], fold['score_end'], fold['score_end'])
        if train.sum() < data.spec.minimum_train_rows or not score.any():
            raise ValueError('Three-fold baseline requires six months plus input warmup')
        train_weights = hour_equal_weights(data.frame.metadata.loc[train])
        score_weights = hour_equal_weights(data.frame.metadata.loc[score])
        hour_count = data.frame.metadata.loc[score, 'available_at'].dt.floor('h').nunique()
        record = {**fold, 'support': int(score.sum()), 'hours': int(hour_count), 'levels': {}}
        y = data.target[train]
        constant = float(np.exp(np.average(np.log(y + 1e-8), weights=train_weights)) - 1e-8)
        record['levels']['log_constant'] = log_mse(data.target[score], np.full(score.sum(), max(0., constant)), score_weights)
        # Fixed half-life 24h, per-symbol past-only recursion, reset on gaps.
        rv = data.frame.values[:, 0]
        ewma = np.full(len(rv), np.nan)
        alpha = 1 - np.exp(np.log(.5) / 24)
        for positions in data.frame.metadata.groupby('symbol', sort=False).groups.values():
            previous, last_time = None, None
            for i in positions:
                stamp = data.frame.metadata.feature_open_time.iloc[i]
                if previous is None or not np.isfinite(rv[i]) or last_time is None or stamp - last_time != pd.Timedelta(hours=1):
                    previous = rv[i] if np.isfinite(rv[i]) else None
                else:
                    previous = alpha * rv[i] + (1 - alpha) * previous
                if previous is not None:
                    ewma[i] = np.sqrt(previous * data.spec.horizon_hours)
                last_time = stamp
        if not np.isfinite(ewma[score]).all():
            raise ValueError('EWMA support differs from frozen common support')
        record['levels']['rolling_24h'] = log_mse(data.target[score],
            np.sqrt(data.frame.values[score, 2] * data.spec.horizon_hours / 24), score_weights)
        record['levels']['EWMA_24h'] = log_mse(data.target[score], ewma[score], score_weights)
        for name, columns in levels.items():
            indices = [data.frame.feature_names.index(column) for column in columns]
            policy = fingerprint({'library': data.frame.schema_hash, 'columns': columns})
            frame = FactorFrame(data.frame.values[:, indices], data.frame.metadata, columns,
                                tuple(data.frame.bypass_mask[i] for i in indices), policy)
            budget.consume('solve')
            model = fit_ridge(frame.take(train), y, ridge=data.spec.ridge, sample_weight=train_weights)
            predicted = model.predict(frame.take(score))
            record['levels'][name] = log_mse(data.target[score], predicted, score_weights)
            if name == 'K0':
                metadata = data.frame.metadata.loc[score].reset_index(drop=True)
                errors = (np.log(data.target[score] + 1e-8) - np.log(predicted + 1e-8)) ** 2
                groups = {f'UTC hours {lo:02d}-{lo+5:02d}': metadata.available_at.dt.hour.between(lo, lo+5).to_numpy()
                          for lo in (0, 6, 12, 18)}
                groups.update({f'Symbol {symbol}': metadata.symbol.eq(symbol).to_numpy()
                               for symbol in sorted(metadata.symbol.unique())})
                threshold = float(np.quantile(data.frame.values[train, 2], 2/3))
                groups['RV24 above training two-thirds quantile'] = data.frame.values[score, 2] >= threshold
                for label, mask in groups.items():
                    if mask.any():
                        # Per-fold weights preserve equal symbols within each hour.
                        weight = float(score_weights[mask].sum())
                        part = residual_groups.setdefault(label, {'weighted_loss': 0., 'weight': 0., 'rows': 0})
                        part['weighted_loss'] += float(np.average(errors[mask], weights=score_weights[mask])) * weight
                        part['weight'] += weight
                        part['rows'] += int(mask.sum())
        records.append(record)
    mean = {level: float(np.average([r['levels'][level] for r in records], weights=[r['hours'] for r in records]))
            for level in records[0]['levels']}
    return {'role': 'S', 'aggregation': 'equal_symbol_within_hour_then_equal_hour', 'folds': records, 'loss': mean,
            'K0_gain_vs_HAR_cal': relative_gain(mean['HAR_cal'], mean['K0']),
            'evidence_status': 'exposed_development_only', 'HAR_cal_definition': list(levels['HAR_cal']),
            'residual_diagnostics': [{'summary': f'K0 squared log residual: {label}. Three exposed S folds; RV threshold fitted per training fold.',
                'loss': part['weighted_loss']/part['weight'], 'support': part['rows']}
                for label, part in sorted(residual_groups.items())[:8]]}



def _verified_S_receipt_source(bars, cutoff, config, spec, exposure_path, feed_directory):
    selected = bars.loc[bars.symbol.isin(config.symbols)].copy()
    selected.attrs = bars.attrs.copy()
    selected['available_at'] = utc_index(selected.available_at)
    selected['open_time'] = utc_index(selected.open_time)
    # The monthly researcher never loads future candles into its feature/label
    # builder. Future malformed prices therefore cannot affect a past readout.
    selected = selected.loc[selected.available_at <= cutoff].copy()
    selected.attrs = bars.attrs.copy()
    if set(selected.symbol) != set(config.symbols):
        raise ValueError('Dataset does not contain the frozen universe')
    feed_identity = selected.attrs.get('feed_identity')
    if bars.attrs.get('availability_mode') == 'quarantined_finite_observation':
        from .quarantined_observation import verify_frame
        from .s_observation_source import validate_reference
        if (not exposure_path or not feed_directory or config.evaluation_mode != 'joint'
                or bars.attrs.get('snapshot_asof') != pd.Timestamp(cutoff).isoformat()
                or bars.attrs.get('months') != 6 or bars.attrs.get('context_hours') != 1234
                or set(bars.symbol) != set(config.symbols)):
            raise ValueError('Logical S requires its exact original six-month source window')
        reference = {'feed_identity': feed_identity, 'feed_directory': str(project_path(feed_directory)),
                     'exposure_path': str(project_path(exposure_path)), 'cutoff': pd.Timestamp(cutoff).isoformat(),
                     'observation_prefix': bars.attrs.get('observation_prefix')}
        prefix = validate_reference(reference)
        if prefix is None or verify_frame(bars) != prefix:
            raise ValueError('Logical S frame differs from its original source prefix')
        return selected, feed_identity
    if feed_identity:
        if not exposure_path or not feed_directory or config.evaluation_mode != 'joint':
            raise ValueError('Live S requires its actual receipt archive and shared lifecycle exposure ledger')
        from .v7_feed import ReceiptArchive
        feed_path = project_path(feed_directory)
        if not (feed_path / 'identity.json').is_file() or not (feed_path / 'receipts.sqlite').is_file():
            raise ValueError('Actual receipt archive is missing; never create a replacement')
        receipt = ReceiptArchive(feed_directory, config.symbols)
        try:
            proof = receipt.snapshot(cutoff, start=cutoff-(pd.DateOffset(months=6) if spec.window_days is None
                else pd.Timedelta(days=180))-pd.Timedelta(hours=1234),
                                     archive_delay_seconds=spec.availability_delay_seconds)
            columns = sorted(proof.columns)
            if (receipt.identity != feed_identity or set(columns) != set(selected.columns)
                    or raw_snapshot_identity(proof, cutoff=cutoff)
                    != raw_snapshot_identity(selected, cutoff=cutoff)):
                raise ValueError('Live S data differs from the actual immutable as-of receipt snapshot')
            shared_path = project_path(exposure_path)
            if shared_path.name != 'exposure.sqlite':
                raise ValueError('Live research must use the frozen lifecycle exposure ledger, not a sibling ledger')
            frozen_lifecycle = json.loads((shared_path.parent / 'lifecycle.json').read_text('utf-8'))
            if frozen_lifecycle['plan']['source_id'] != receipt.identity:
                raise ValueError('Shared exposure ledger belongs to another live source')
        finally:
            receipt.close()
    return selected, feed_identity

def run_month(directory, bars, cutoff, *, config=None, spec=None, backend=None,
              exposure_path=None, feed_directory=None, external_deadline=None,
              authority=None, schedule=None, preparation=None):
    """Offline legacy diagnostic, or explicitly prepaid actual-receipt S engine.

    This integrated engine enforces cooperative original leases. A fixed external
    native/GPT watchdog is still required before unattended service admission.
    """
    if external_deadline is not None and (
            type(external_deadline) not in (int, float) or not np.isfinite(external_deadline)
            or external_deadline <= time.monotonic()):
        raise BudgetExhausted('External controls deadline already consumed')
    if authority is None:
        if schedule is not None or preparation is not None or bars.attrs.get('feed_identity'):
            raise ValueError('Actual receipt S requires the shared authority and fixed work schedule')
        return _run_month_offline(directory, bars, cutoff, config=config, spec=spec, backend=backend,
            exposure_path=exposure_path, feed_directory=feed_directory, external_deadline=external_deadline)
    from .monthly_work_authority import MonthlyWorkAuthority
    from .s_preparation import SPreparationEnvelope
    from .scheduled_search import FrozenSWorkSchedule
    from .work_lease import FrozenWorkLease
    config, spec = config or V7Config(), spec or RollingSpec()
    original = FrozenWorkLease.issue(300) if preparation is None else None
    _, end = spec.window(cutoff)
    if (type(authority) is not MonthlyWorkAuthority or type(schedule) is not FrozenSWorkSchedule
            or config.evaluation_mode != 'joint'
            or config.formulas_per_family != 8 or config.budget_policy not in {'standard', 'controls_third'}
            or not exposure_path or not feed_directory or not bars.attrs.get('feed_identity')
            or external_deadline is not None):
        raise ValueError('Shared S requires actual receipts, standard frozen allocation and isolated joint task')
    from .scheduled_search import validate_schedule_authority
    body = validate_schedule_authority(authority, schedule)
    if fingerprint(body['config']) != fingerprint(asdict(config)):
        raise ValueError('S schedule config differs from RunStore contract')
    limits = monthly_budget(config, end.strftime('%Y-%m')).limits
    if body['limits'] != limits:
        raise ValueError('S schedule counter contract changed')
    slots = body['slots']
    family_count = 1 if config.budget_policy == 'controls_third' else 3
    if family_count == 1:
        from .s_outer_context import SOuterContext
        if preparation is None or not preparation.identity.get('outer_scope'):
            raise ValueError('Shared controls require their actual frozen allocation and active claim')
        scope = preparation.identity['outer_scope']
        if scope['reference']['kind'] != 'controls':
            raise ValueError('Control arm cannot borrow an unrelated outer scope')
        SOuterContext.verify_bound(scope)
    expected = [('baseline','baseline'), *[(key,stage) for i in range(family_count)
        for key,stage in [(f'family:{i}','family'), *[(f'candidate:{i}:{j}','candidate') for j in range(8)]]],
        ('pruning','pruning')]
    if [(slot['key'],slot['stage']) for slot in slots] != expected:
        raise ValueError('S requires the complete original baseline/three-family/24-candidate/pruning schedule')
    if slots[0]['counts'] != {'solve':7} or slots[0]['seconds'] != 300:
        raise ValueError('Cold three-fold baseline requires seven prepaid solves and original 300s')
    for slot in slots:
        if slot['stage'] == 'family' and (slot['counts'] != {'family':1,'gpt_attempt':2,'ast':32} or slot['seconds'] != 300):
            raise ValueError('Each family freezes two attempts and 32 AST under one 300s work lease')
        if slot['stage'] == 'candidate' and (set(slot['counts']) != {'f','solve'} or slot['counts']['f'] != 1):
            raise ValueError('Each candidate has its own fixed F/solve allowance')
        if slot['stage'] == 'pruning' and set(slot['counts']) != {'solve'}:
            raise ValueError('Pruning must have its own frozen solve allowance')
    allocation = body['allocation']
    from .monthly_partition_policy import canonical_policy
    policy = canonical_policy(authority.manifest.get('partition_policy'))
    lane = 'S:' + config.search_mode if family_count == 1 else 'S'
    remaining_S = policy['lanes'][lane]
    if (not isinstance(allocation, dict) or allocation.get('unused_credit_transfer') is not False
            or allocation.get('S_solve') != body['allocated_counts']['solve']
            or any(type(allocation.get(k)) is not int or allocation[k] < low for k,low in
                   [('forward_solve_holdback',1),('nonlinear_solve_holdback',0),
                    ('forward_seconds_holdback',1),('nonlinear_seconds_holdback',0)])
            or remaining_S['counts']['solve'] + allocation['forward_solve_holdback'] + allocation['nonlinear_solve_holdback'] > authority.limits['solve']
            or remaining_S['seconds'] + allocation['forward_seconds_holdback'] + allocation['nonlinear_seconds_holdback'] > authority.manifest['monthly_seconds']):
        raise ValueError('Missing or inconsistent frozen forward/nonlinear counter and time holdbacks')
    source_reference = {'feed_identity': bars.attrs['feed_identity'],
                        'feed_directory': str(project_path(feed_directory)),
                        'exposure_path': str(project_path(exposure_path)), 'cutoff': end.isoformat()}
    if bars.attrs.get('availability_mode') == 'quarantined_finite_observation':
        source_reference['observation_prefix'] = bars.attrs.get('observation_prefix')
    if preparation is None:
        preparation = SPreparationEnvelope.prepare(authority,directory,end.strftime('%Y-%m'),
            config=asdict(config),spec=asdict(spec),source_reference=source_reference,
            schedule_hash=schedule.digest,lease=original)
    if (type(preparation) is not SPreparationEnvelope or preparation.authority is not authority
            or preparation.identity['source_reference'] != source_reference
            or preparation.identity['work_schedule_hash'] != schedule.digest
            or fingerprint(preparation.identity['config']) != fingerprint(asdict(config))
            or fingerprint(preparation.identity['spec']) != fingerprint(asdict(spec))
            or preparation.identity['directory'] != str(project_path(directory))
            or preparation.month != end.strftime('%Y-%m')):
        raise ValueError('S preparation differs from the actual invocation')
    if not preparation._executable:
        # Verification grants no new execution and publishes no S exposure.
        from .neural_forward_ticket import _LifecycleReader, _verify_authority_binding
        from .v7_memory import search_exposure_context
        reader = _LifecycleReader(project_path(exposure_path).parent)
        try:
            _verify_authority_binding(reader, authority, reader.verify()['operations'])
            _, verified_feed = _verified_S_receipt_source(bars, end, config, spec, exposure_path, feed_directory)
            context = search_exposure_context(preparation.digest, end, config.symbols, verified_feed, spec=spec)
            reader.exposure.assert_prompt_search(**{k:v for k,v in context.items() if k != 'data_id'})
        finally:
            reader.close()
        raise ValueError('Recovered S preparation is readback only; no execution grant')
    try:
        preparation.start_preprocessing()
        from .neural_forward_ticket import _LifecycleReader, _verify_authority_binding
        reader = _LifecycleReader(project_path(exposure_path).parent)
        try:
            result = reader.verify()
            _verify_authority_binding(reader, authority, result['operations'])
            if reader.plan.source_id != bars.attrs['feed_identity']:
                raise ValueError('S source differs from authoritative lifecycle exposure')
        finally:
            reader.close()
        return _run_month_scheduled_impl(directory,bars,cutoff,config=config,spec=spec,backend=backend,
            exposure_path=exposure_path,feed_directory=feed_directory,authority=authority,
            schedule=schedule,preparation=preparation)
    except BaseException as error:
        # Persist the original cause BEFORE failure accounting can close the native tree.
        from .v7_storage import immutable_record
        cause = {'preparation_hash': preparation.digest, 'error_type': type(error).__name__,
                 'message': str(error)[:2000], 'stage': 'scheduled_S_preparation'}
        for parent in (preparation.root, preparation.mirror):
            try: immutable_record(parent / 'failure-cause.json', {'value': cause, 'sha256': fingerprint(cause)})
            except BaseException as secondary: error.add_note('Failure diagnostic: ' + type(secondary).__name__)
        try:
            record = preparation.baseline_record()
            if record['status'] not in {'completed','failed_consumed','interrupted_consumed'}:
                authority.fail_consumed(record['reservation_id'], 'S_preparation:' + type(error).__name__)
        except BaseException as secondary:  # noqa: BLE001 -- original error remains authoritative.
            error.add_note('S failure accounting also failed: ' + type(secondary).__name__)
        raise


def _run_month_offline(directory, bars, cutoff, *, config=None, spec=None, backend=None,
              exposure_path=None, feed_directory=None, external_deadline=None):
    config, spec = config or V7Config(), spec or RollingSpec()
    if external_deadline is not None and (
            type(external_deadline) not in (int, float) or not np.isfinite(external_deadline)
            or external_deadline <= time.monotonic()):
        raise BudgetExhausted('External controls deadline already consumed')
    if spec.task != 'volatility' or spec.horizon_hours != 12:
        raise ValueError('Development search controller currently runs the isolated volatility12h task')
    _, cutoff = spec.window(cutoff)
    month = cutoff.strftime('%Y-%m')
    selected = bars.loc[bars.symbol.isin(config.symbols)].copy()
    selected.attrs = bars.attrs.copy()
    selected['available_at'] = utc_index(selected.available_at)
    selected['open_time'] = utc_index(selected.open_time)
    # The monthly researcher never loads future candles into its feature/label
    # builder. Future malformed prices therefore cannot affect a past readout.
    selected = selected.loc[selected.available_at <= cutoff].copy()
    selected.attrs = bars.attrs.copy()
    if set(selected.symbol) != set(config.symbols):
        raise ValueError('Dataset does not contain the frozen universe')
    feed_identity = selected.attrs.get('feed_identity')
    if feed_identity:
        if not exposure_path or not feed_directory or config.evaluation_mode != 'joint':
            raise ValueError('Live S requires its actual receipt archive and shared lifecycle exposure ledger')
        from .v7_feed import ReceiptArchive
        receipt = ReceiptArchive(feed_directory, config.symbols)
        try:
            proof = receipt.snapshot(cutoff, start=cutoff-(pd.DateOffset(months=6) if spec.window_days is None
                else pd.Timedelta(days=180))-pd.Timedelta(hours=1234),
                                     archive_delay_seconds=spec.availability_delay_seconds)
            columns = sorted(proof.columns)
            if (receipt.identity != feed_identity or set(columns) != set(selected.columns)
                    or raw_snapshot_identity(proof, cutoff=cutoff)
                    != raw_snapshot_identity(selected, cutoff=cutoff)):
                raise ValueError('Live S data differs from the actual immutable as-of receipt snapshot')
            shared_path = project_path(exposure_path)
            if shared_path.name != 'exposure.sqlite':
                raise ValueError('Live research must use the frozen lifecycle exposure ledger, not a sibling ledger')
            frozen_lifecycle = json.loads((shared_path.parent / 'lifecycle.json').read_text('utf-8'))
            if frozen_lifecycle['plan']['source_id'] != receipt.identity:
                raise ValueError('Shared exposure ledger belongs to another live source')
        finally:
            receipt.close()
    data = build_rolling_data(selected, spec)
    data_id = typed_search_data_id(data, selected, cutoff)
    directory = project_path(directory)
    with FileLock(str(directory) + '.lock'):
        store = RunStore(directory, config, spec, exposure_path=exposure_path, feed_identity=feed_identity)
        try:
            state = store.get(month)
            if state and state['data_id'] != data_id:
                raise ValueError('Frozen monthly as-of dataset identity changed')
            if state:
                _verify_month_artifacts(directory, month, state)
            if state and state['status'] == 'complete':
                immutable_json(directory / 'months' / f'{month}.json', state['summary'])
                return state['summary']
            if state and state['status'] == 'finalizing':
                return _publish_finalization(directory, month, state, store)
            if state and state.get('search_exposure'):
                exposed = state['search_exposure']
                store.exposure.assert_prompt_search(**{k: v for k, v in exposed.items() if k != 'data_id'})
            if state is None:
                state = {'data_id': data_id, 'cutoff': cutoff.isoformat(), 'status': 'running',
                         'budget': monthly_budget(config, month).to_dict(), 'pool': FactorPool().to_dict(),
                         'attempts': [], 'next_family': 0, 'baseline': None, 'baseline_model': None,
                         'active_candidate': None, 'agent_audit': [], 'spent_seconds': 0.,
                         'wall_checkpoint': None, 'pending_family': None, 'materializations': [], 'prescreen': []}
                if config.evaluation_mode == 'joint':
                    from .joint_pool import DevelopmentPoolArchive, FrozenPool
                    from .v7_memory import ResearchMemory, search_exposure_context
                    previous = store.db.execute("SELECT month,value FROM months WHERE month<? AND json_extract(value,'$.status')='complete' ORDER BY month DESC LIMIT 1",
                                                (month,)).fetchone()
                    prior = json.loads(previous[1]) if previous else {}
                    if previous:
                        _verify_month_artifacts(directory, previous[0], prior)
                    reference = FrozenPool.from_dict(prior['development_pool']) if prior.get('development_pool') else FrozenPool()
                    state['frozen_reference_pool'] = reference.to_dict()
                    state['development_pool'] = reference.to_dict()
                    if prior.get('joint_archive'):
                        old_archive = DevelopmentPoolArchive.from_dict(prior['joint_archive'])
                        state['joint_archive'] = old_archive.rollover(month_start(cutoff),
                            directory / 'archives' / f'{previous[0]}.json').to_dict()
                    else:
                        state['joint_archive'] = DevelopmentPoolArchive(month_start(cutoff)).to_dict()
                    state['profile_attempted_ids'] = []
                    state['search_exposure'] = search_exposure_context(data_id, cutoff, config.symbols, feed_identity, spec=spec)
                    exposed = state['search_exposure']
                    store.exposure.record(source=exposed['source'], batch=exposed['batch'],
                        symbols=exposed['symbols'], start=exposed['start'], end=exposed['end'], role='S',
                        detail={'data_id': data_id, 'scope': 'exposed_development_only'})
                    store.exposure.assert_prompt_search(source=exposed['source'], batch=exposed['batch'],
                        symbols=exposed['symbols'], start=exposed['start'], end=exposed['end'])
                    if prior.get('research_memory'):
                        previous_memory = ResearchMemory.from_state(prior['research_memory'])
                        memory = previous_memory.rollover(month_start(cutoff),
                            directory / 'memories' / f'{previous[0]}.json')
                    else:
                        memory = ResearchMemory.initial(month_start(cutoff), [
                            {'program': seed, 'citation': 'Registered V7 known multiscale volatility seed; no novelty claimed.'}
                            for seed in literature_seeds()])
                    state['research_memory'] = memory.to_state()
                store.save(month, state, 'month_started')

            invocation_started = time.monotonic()
            prior_spent = state['spent_seconds']
            if state['wall_checkpoint'] is not None:
                unobserved = time.time() - state['wall_checkpoint']
                if unobserved < 0:
                    raise ValueError('Research wall clock moved backward; cannot reset its budget')
                # Conservative crash accounting includes downtime since last
                # durable checkpoint, never silently resets a monthly deadline.
                prior_spent += unobserved

            def checkpoint(event):
                state['spent_seconds'] = prior_spent + time.monotonic() - invocation_started
                state['wall_checkpoint'] = time.time()
                store.save(month, state, event)
                if external_deadline is not None and time.monotonic() >= external_deadline:
                    raise BudgetExhausted('External shared controls wall-time limit')

            def persist(value):
                state['budget'] = value
                checkpoint('budget_reserved')
                if state['spent_seconds'] > config.research_seconds:
                    raise BudgetExhausted('Monthly wall-time limit, including conservative crash accounting')

            budget = DurableBudget(MonthlyBudget.from_dict(state['budget']), persist)
            pool = FactorPool.from_dict(state['pool'])
            reference_pool = archive = context = memory = None
            if config.evaluation_mode == 'joint':
                from .joint_pool import DevelopmentPoolArchive, FrozenPool
                reference_pool = FrozenPool.from_dict(state['frozen_reference_pool'])
                archive = DevelopmentPoolArchive.from_dict(state['joint_archive'])
                context = DevelopmentPoolArchive.context(reference_pool, cutoff=cutoff,
                    data_hash=data_id, protocol_hash=spec.digest, source_hash=fingerprint(store.identity['source_files']))
                from .v7_memory import ResearchMemory
                memory = ResearchMemory.from_state(state['research_memory'])

            def persist_memory(value):
                state['research_memory'] = value
                checkpoint('research_memory_updated')
            if state['active_candidate']:
                pending = state['active_candidate']
                if memory and pending['formula_id'] in pool.records:
                    evidence_id = memory.bind_result(store, store.exposure, month, pending['formula_id'])
                    memory.register_measured_parent(family_id=pending['family'], evidence_id=evidence_id,
                                                    store=store, exposure=store.exposure)
                    persist_memory(memory.to_state())
                    state['attempts'].append({**pending, 'status': 'measured_recovered_budget_retained'})
                else:
                    state['attempts'].append({**pending, 'status': 'interrupted_consumed'})
                state['active_candidate'] = None
                checkpoint('interrupted_candidate_retained')
            if state['baseline'] is None:
                from .joint_pool import hour_equal_weights
                readout = _baseline_readout(data, cutoff, budget)
                train = data.train_mask(cutoff)
                budget.consume('solve')
                model = fit_ridge(data.frame.take(train), data.target[train], ridge=spec.ridge,
                                  sample_weight=hour_equal_weights(data.frame.metadata.loc[train]),
                                  provenance=data.training_identity(cutoff))
                state['baseline_model'] = ModelRegistry(directory / 'models').save(model, {
                    'rolling_hash': spec.digest, 'library': library_schema(), 'train': data.training_identity(cutoff)})
                state['baseline'] = readout
                checkpoint('cold_baseline_fitted')
            started = time.monotonic()
            seeds = literature_seeds()
            stop = state['spent_seconds'] >= config.research_seconds
            while state['next_family'] < budget.limits['family'] and not stop:
                index = state['next_family']
                family_id = f'{month}:local:{index}'
                budget_family_id = family_id
                island = 'refinement' if index % 2 == 0 else 'exploration'
                if config.search_mode == 'gpt':
                    if budget.counts['gpt_attempt'] >= budget.limits['gpt_attempt']:
                        stop = True
                        break
                    from .program_contract import Program
                    from .program_syntax import CompactProgram
                    from .prompt_builder import PromptIdentity, build_formula_prompt
                    from .v7_agent import CodexBatchBackend, propose_batch
                    identity = PromptIdentity('volatility12h', spec.digest, reference_pool.digest if reference_pool else pool.digest,
                                              fingerprint(library_schema()))
                    parents = []
                    parent_seeds = [(f'literature{i}', program) for i, program in enumerate(seeds)]
                    if memory:
                        resolved = memory.prompt_parents(store, store.exposure, island=island,
                                                         current_context=context)
                        state['parent_provenance'] = resolved['receipt']
                        parent_seeds = [(p['parent_id'], p['program']) for p in resolved['parents']]
                    for parent_id, program in parent_seeds:
                        parents.append({'parent_id': parent_id,
                            'program': CompactProgram.from_program(Program.from_dict(program)).to_dict(),
                            'evidence': {'evidence_id': parent_id, 'role': 'S', 'task_id': identity.task_id,
                                         'rolling_id': identity.rolling_id, 'pool_id': identity.pool_id,
                                         'summary': 'Verified S ancestor or cited literature seed; prior-context formula only, no current gain claimed.',
                                         'metrics': {}}})
                        if memory:
                            parent_record = next(p for p in resolved['parents'] if p['parent_id'] == parent_id)
                            if parent_record['kind'] == 'research':
                                prior_readouts = []
                                for evidence_id in parent_record['evidence_sources'][:2]:
                                    verified = memory.resolver(store, store.exposure)(evidence_id)
                                    origin_state = store.get(verified['month'])
                                    origin_result = origin_state['pool']['records'][verified['formula_id']]['result']
                                    prior_readouts.append({'cutoff': origin_state['cutoff'],
                                        'data_id': verified['data_id'], 'pool_id': origin_result['pool_hash'],
                                        'result_hash': verified['result_hash'],
                                        'gain_F': verified['prior_gain_F'], 'gain_S': verified['prior_gain_S']})
                                parents[-1]['evidence']['prior_readouts'] = prior_readouts
                                parents[-1]['evidence']['summary'] = 'Verified ancestor S readout relative to its original frozen Pv, equal-hour aggregation; prior context only; current gain unavailable.'
                            else:
                                parents[-1]['evidence']['summary'] = 'Cited known literature seed; no measured candidate gain or novelty claimed.'
                    for key, record in ([] if memory else pool.shortlist()[:2]):
                        metrics = record['result']['channels']['S']
                        clean = {k: metrics[k] for k in ('loss', 'reference_loss', 'gain') if metrics[k] is not None}
                        parents.append({'parent_id': key, 'program': CompactProgram.from_program(
                            Program.from_dict(record['program'])).to_dict(), 'evidence': {
                            'evidence_id': key, 'role': 'S', 'task_id': identity.task_id,
                            'rolling_id': identity.rolling_id, 'pool_id': identity.pool_id,
                            'summary': ('Measured equal-hour development readout relative to frozen Pv; exposed evidence.'
                                        if reference_pool else 'Measured equal-fold development readout relative to K0; exposed evidence.'),
                            'metrics': clean}})
                    packet = build_formula_prompt(identity=identity,
                        task_semantics=('Predict 12h realized volatility; development-only logMSE above frozen Pv.'
                                        if reference_pool else 'Predict 12h realized volatility; development-only logMSE above K0.'),
                        feature_schema=[{'name': name, 'unit': 'squared log return' if i < 8 else 'dimensionless',
                                         'preprocessing': 'raw K0; training-only downstream scale'}
                                        for i, name in enumerate(FEATURE_NAMES)],
                        input_semantics=f'Full past {config.lookback} hour K0 windows, raw units; gaps restart.',
                        parents=parents, budget=budget, diagnostics=[{
                            'evidence_id': fingerprint({'data_id': state['data_id'], 'diagnostic': row}),
                            'role': 'S', 'task_id': identity.task_id, 'rolling_id': identity.rolling_id,
                            'pool_id': identity.pool_id, 'summary': row['summary'],
                            'metrics': {'loss': row['loss'], 'support': row['support']}}
                            for row in state['baseline'].get('residual_diagnostics', [])],
                        failure_counts={'failed': sum(a['status'] != 'scored' for a in state['attempts'])})
                    try:
                        batch = propose_batch(packet, backend=backend or CodexBatchBackend(directory / 'agent'),
                            budget=budget, feature_names=FEATURE_NAMES,
                            allowed_parent_ids={p['parent_id'] for p in parents}, audit=state['agent_audit'],
                            persist_budget=lambda *_: persist(budget.to_dict()))
                    except (ValueError, TimeoutError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
                        state['attempts'].append({'family': family_id, 'status': 'proposal_failed', 'error': str(exc)[:500]})
                        checkpoint('proposal_failed')
                        continue
                    budget_family_id = batch.family_id
                    family_id = f'{month}:gpt:{fingerprint(batch.family_id)[:32]}' if memory else batch.family_id
                    if memory:
                        proposal = batch.to_dict() | {'family_id': family_id}
                        memory.register_proposal(proposal, island=island,
                            next_experiment='Test conditional F/P/S on the frozen three development folds.',
                            minimum_progress=.005, remaining_budget=24,
                            stop_conditions='Dormant after two adjacent monthly checks without progress; no gate changes.')
                        persist_memory(memory.to_state())
                    programs = [p.to_dict() for p in batch.programs]
                    if batch.template.to_dict()['slots']:
                        from .local_mutation import expand_template
                        seen = {p.digest for p in batch.programs}
                        try:
                            for variant in expand_template(batch.template, family_id=budget_family_id,
                                                           budget=budget, seen=seen):
                                state['materializations'].append({k: v for k, v in variant.items() if k != 'program'})
                                if variant['status'] == 'valid':
                                    programs.append(variant['program'])
                                checkpoint('template_materialized')
                        except BudgetExhausted:
                            checkpoint('template_budget_reached')
                    if reference_pool is None:
                        programs, prescreen = _prescreen(data, cutoff, programs, config.lookback, config.formulas_per_family)
                    else:
                        programs, prescreen = _prescreen(data, cutoff, programs, config.lookback, config.formulas_per_family,
                                                        reference_pool=reference_pool)
                    state['prescreen'] += prescreen
                else:
                    state['pending_family'] = family_id
                    checkpoint('family_planned')
                    if family_id not in budget.families:
                        budget.register_family(family_id)
                    local_parents = None
                    if memory:
                        local_parents = memory.prompt_parents(store, store.exposure, island=island,
                                                             current_context=context)['parents']
                    if memory and family_id not in memory.proposals:
                        memory.register_proposal({'family_id': family_id, 'parent_ids': [local_parents[-1]['parent_id']],
                            'hypothesis': 'Known volatility mechanism with bounded local changes.',
                            'mechanism': 'Past multiscale volatility and range discrepancy.',
                            'competing_explanation': 'The existing K0/Pv already spans this information.',
                            'falsification': 'No common-support conditional marginal gain.'}, island=island,
                            next_experiment='Same-access F/P/S conditional comparison.', minimum_progress=.005,
                            remaining_budget=24, stop_conditions='Two adjacent cycles without progress.')
                        persist_memory(memory.to_state())
                    programs = []
                    rng = np.random.default_rng(index + 702)
                    if memory and config.search_mode == 'local':
                        from .v7_search_ops import local_variations
                        parent = local_parents[-1]
                        mutations = local_variations(parent['program'], local_parents[-2]['program'],
                            config.formulas_per_family, feature_names=FEATURE_NAMES, seed=index + 702)
                        records = memory.materialize(parent['parent_id'], mutations, family_id=family_id,
                            budget=budget, store=store, exposure=store.exposure,
                            seen={fingerprint(record['program']) for record in pool.records.values()})
                        state['materializations'].extend({k: v for k, v in r.items() if k != 'program'} for r in records)
                        programs = [r['program'] for r in records if r['status'] == 'valid']
                    for i in range(0 if memory and config.search_mode == 'local' else config.formulas_per_family):
                        pair = (i + index) % 2
                        program = json.loads(json.dumps(seeds[pair]))
                        if index == 0 and i < 2:
                            pass
                        elif config.search_mode == 'random':
                            program['outputs'][0]['arg']['right']['name'] = FEATURE_NAMES[int(rng.integers(0, 8))]
                        else:
                            program['outputs'][0] = {'op': 'std' if index == 1 else 'mean',
                                                     'arg': program['outputs'][0]['arg']}
                            program['outputs'][0]['arg']['right']['name'] = FEATURE_NAMES[(i + index) % 8]
                        budget.consume('ast', family_id=family_id)
                        programs.append(program)
                state['next_family'] += 1
                state['pending_family'] = None
                checkpoint('family_materialized')
                if memory:
                    programs = list({fingerprint(formula_identity(p, config.lookback)): p for p in programs}.values())
                    scheduled = memory.reserve_schedule([
                        {'candidate_id': fingerprint(formula_identity(program, config.lookback)),
                         'family_id': budget_family_id,
                         'lane': 'blind' if position >= max(1, len(programs) - 2)
                                 else 'refine' if island == 'refinement' else 'new'}
                        for position, program in enumerate(programs)
                        if fingerprint(formula_identity(program, config.lookback)) not in memory.scheduled],
                        budget, persist=persist_memory)
                    admitted_ids = {r['candidate_id'] for r in scheduled['scheduled']}
                    programs = [p for p in programs if fingerprint(formula_identity(p, config.lookback)) in admitted_ids]
                for program in programs[:config.formulas_per_family]:
                    if prior_spent + time.monotonic() - invocation_started >= config.research_seconds:
                        stop = True
                        break
                    identifier = fingerprint(formula_identity(program, config.lookback))
                    if identifier in pool.records:
                        state['attempts'].append({'formula_id': identifier, 'family': family_id, 'status': 'duplicate'})
                        checkpoint('exact_duplicate')
                        continue
                    try:
                        budget.consume('f', family_id=budget_family_id)
                        state['active_candidate'] = {'formula_id': identifier, 'family': family_id}
                        checkpoint('candidate_started')
                        if reference_pool is None:
                            result = evaluate_formula(data, cutoff, program, lookback=config.lookback,
                                                      budget=budget, profile=len(pool.records) < config.profile_candidates)
                        else:
                            from .joint_pool import PoolMember, evaluate_candidate
                            has_parameters = bool(program.get('parameters'))
                            if has_parameters and len(state['profile_attempted_ids']) >= config.profile_candidates:
                                raise BudgetExhausted('Parameterized structure quota reached; no uncounted P fitting')
                            if has_parameters:
                                state['profile_attempted_ids'].append(identifier)
                                checkpoint('parameter_structure_reserved')
                            member = PoolMember(program, lookback=config.lookback,
                                parameter_mode='profile' if has_parameters else 'fixed', family=family_id)
                            result = evaluate_candidate(data, cutoff, reference_pool, member, budget=budget, context=context)
                            result['formula_id'] = identifier
                            result['channels'] = {name: {'loss': result[name]['Pv']['candidate_loss'],
                                'reference_loss': result[name]['Pv']['reference_loss'], 'gain': result[name]['Pv']['gain']}
                                for name in ('F', 'S')}
                            archive.record(member, result, context)
                            state['joint_archive'] = archive.to_dict()
                        path = directory / 'results' / f'{month}-{identifier}.json'
                        immutable_json(path, result)
                        pool.add(program, result, family=family_id, lookback=config.lookback)
                        state['pool'] = pool.to_dict()
                        checkpoint('measured_result_persisted')
                        if memory:
                            evidence_id = memory.bind_result(store, store.exposure, month, identifier)
                            memory.register_measured_parent(family_id=family_id, evidence_id=evidence_id,
                                                            store=store, exposure=store.exposure)
                            persist_memory(memory.to_state())
                        state['attempts'].append({'formula_id': identifier, 'family': family_id, 'status': 'scored',
                                                  'gain_F': result['channels']['F']['gain'], 'gain_S': result['channels']['S']['gain']})
                    except BudgetExhausted as exc:
                        state['attempts'].append({'formula_id': identifier, 'family': family_id,
                                                  'status': 'budget_exhausted', 'error': str(exc)})
                        stop = True
                    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                        state['attempts'].append({'formula_id': identifier, 'family': family_id,
                                                  'status': 'invalid', 'error': str(exc)[:500]})
                    state['active_candidate'] = None
                    checkpoint('candidate_finished')
                    if stop:
                        break
            shortlist = [{'formula_id': k, 'program': v['program'], 'lookback': v['lookback'],
                          'gain_F': v['result']['channels']['F']['gain'],
                          'gain_S': v['result']['channels']['S']['gain']} for k, v in pool.shortlist()]
            if reference_pool is not None:
                from .joint_pool import PoolMember, stability_select
                observations = {}
                for family_id, proposal in memory.proposals.items():
                    observations.setdefault(proposal['version'], []).extend(
                        (identifier, record) for identifier, record in pool.records.items() if record['family'] == family_id)
                # Only a genuinely re-evaluated original parent can advance an
                # older hypothesis clock; new descendants do not impersonate it.
                for parent in memory.book.parents_by_id.values():
                    version = parent['hypothesis_version']
                    if version:
                        observations.setdefault(version, []).extend(
                            (identifier, record) for identifier, record in pool.records.items()
                            if record['program'] == parent['program'])
                for version, group in observations.items():
                    hypothesis = memory.book.versions.get(version)
                    if not group or not hypothesis or hypothesis['state'] != 'active':
                        continue
                    if hypothesis['history'] and hypothesis['history'][-1]['month'] == cutoff.isoformat():
                        continue
                    group = list({identifier: record for identifier, record in group}.items())
                    best_id, best = max(group, key=lambda pair: pair[1]['result']['channels']['F']['gain']
                                       if pair[1]['result']['channels']['F']['gain'] is not None else -1e100)
                    evidence_id = memory.bind_result(store, store.exposure, month, best_id)
                    memory.observe(version, store=store, exposure=store.exposure, month=month_start(cutoff),
                        progress=best['result']['channels']['F']['gain'] or 0.,
                        evidence_sources=[evidence_id], spent=min(len(group), hypothesis['remaining_budget']),
                        failure='No meaningful conditional increment' if (best['result']['channels']['F']['gain'] or 0.) < .005 else None)
                persist_memory(memory.to_state())
                results = [r['result'] for r in pool.records.values()]
                selection = stability_select(results, reference_pool)
                for record in pool.records.values():
                    member = PoolMember(record['program'], lookback=record['lookback'], family=record['family'],
                        parameter_mode='profile' if record['program'].get('parameters') else 'fixed')
                    if member.member_id in selection['selected_ids']:
                        if len(reference_pool.members) < 8:
                            reference_pool, admission = archive.admit(reference_pool, member, month=month_start(cutoff), selection=selection)
                            state['pool_admission'] = admission
                        else:
                            state['pool_admission'] = {'status': 'capacity_deferred', 'affects_A0': False}
                        break
                if reference_pool.members and not state.get('prune_attempted'):
                    from .joint_pool import backward_prune
                    state['prune_attempted'] = True
                    checkpoint('pruning_budget_reserved')
                    try:
                        pruned, pruning = backward_prune(data, cutoff, reference_pool, budget=budget, max_remove=1)
                        state['pruning'] = pruning
                        for step in pruning['steps']:
                            for trial in step['trials']:
                                member_id = trial['removed']
                                if member_id in archive.lifecycle:
                                    archive.monitor_decay(member_id, month=month_start(cutoff), k0_gain=None,
                                        pool_gain=1-step['baseline_loss']/trial['loss'] if trial['loss'] > 0 else None)
                        for member_id in pruning['removed_ids']:
                            if member_id in archive.lifecycle:
                                archive.lifecycle[member_id]['state'] = 'retired'
                        reference_pool = pruned
                    except (BudgetExhausted, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                        state['pruning'] = {'status': 'consumed_incomplete', 'error': str(exc)[:500], 'affects_A0': False}
                elif state.get('prune_attempted') and not state.get('pruning'):
                    state['pruning'] = {'status': 'interrupted_consumed', 'affects_A0': False}
                state['development_pool'] = reference_pool.to_dict()
                state['joint_archive'] = archive.to_dict()
            nonlinear_report = None
            if config.nonlinear_mode != 'disabled':
                checkpoint('nonlinear_controller_handoff')
                try:
                    nonlinear_report = _nonlinear_month(store, month, state, budget, data, cutoff,
                        config.nonlinear_mode,
                        config.research_seconds - prior_spent - (time.monotonic() - invocation_started), checkpoint)
                except (BudgetExhausted, ValueError, MemoryError, OSError) as exc:
                    nonlinear_report = {'status': 'closed_consumed_or_invalid', 'error_type': type(exc).__name__,
                                        'error': str(exc)[:400], 'automatic_adoption': False, 'formal_claim': False}
                    state['nonlinear_report'] = nonlinear_report
                    checkpoint('nonlinear_channel_closed')
            summary = {'version': 'V7 construction pilot', 'month': month, 'cutoff': cutoff.isoformat(),
                       'scope': 'exposed_development_only', 'search_mode': config.search_mode,
                       'execution_scope': 'legacy_offline_diagnostic', 'serving_eligible': False,
                       'baseline': state['baseline'], 'baseline_model': state['baseline_model'],
                       'budget': budget.to_dict(), 'attempts': state['attempts'], 'shortlist': shortlist,
                       'formal_confirmation_enabled': False, 'A0': 'K0',
                       'evaluation_mode': config.evaluation_mode,
                       'development_pool': state.get('development_pool'),
                       'nonlinear': nonlinear_report,
                       'search_seconds_this_invocation': time.monotonic() - started,
                       'monthly_seconds': prior_spent + time.monotonic() - invocation_started,
                       'limitations': ['Full-pipeline calibration not accepted', 'Linux resource envelope not tested',
                                       'No independent forward confirmation',
                                       'Legacy single reference is K0' if reference_pool is None else 'Joint pool updates are development evidence only']}
            state['summary'], state['status'] = summary, 'finalizing'
            state['wall_checkpoint'] = None
            state['spent_seconds'] = summary['monthly_seconds']
            store.save(month, state, 'month_finalization_committed')
            return _publish_finalization(directory, month, state, store)
        finally:
            store.close()


def _run_month_scheduled_impl(directory, bars, cutoff, *, config=None, spec=None, backend=None,
              exposure_path=None, feed_directory=None, external_deadline=None,
              authority=None, schedule=None, preparation=None):
    from .scheduled_search import ScheduledSearch
    from .work_authority_bridge import LocalQuotaExhausted, WorkAuthorityBridge
    config, spec = config or V7Config(), spec or RollingSpec()
    if external_deadline is not None and (
            type(external_deadline) not in (int, float) or not np.isfinite(external_deadline)
            or external_deadline <= time.monotonic()):
        raise BudgetExhausted('External controls deadline already consumed')
    if spec.task != 'volatility' or spec.horizon_hours != 12:
        raise ValueError('Development search controller currently runs the isolated volatility12h task')
    _, cutoff = spec.window(cutoff)
    month = cutoff.strftime('%Y-%m')
    selected, feed_identity = _verified_S_receipt_source(bars, cutoff, config, spec, exposure_path, feed_directory)
    from .v7_memory import search_exposure_context
    provisional = search_exposure_context(preparation.digest, cutoff, config.symbols, feed_identity, spec=spec)
    evidence = ExposureLedger(project_path(exposure_path))
    try:
        interval = {k: v for k, v in provisional.items() if k != 'data_id'}
        evidence.record(**interval, role='S', detail={'preparation': preparation.digest, 'scope': 'input_preparation'})
        evidence.assert_prompt_search(**interval)
    finally:
        evidence.close()
    data = build_rolling_data(selected, spec)
    data_id = typed_search_data_id(data, selected, cutoff)
    directory = project_path(directory)
    with FileLock(str(directory) + '.lock'):
        store = RunStore(directory, config, spec, mode='authority_scheduled_research_v1',
            exposure_path=exposure_path, feed_identity=feed_identity,
            work_contract={'schedule': schedule.to_dict(), 'authority_manifest_hash': authority.manifest_hash})
        try:
            state = store.get(month)
            if state and state['data_id'] != data_id:
                raise ValueError('Frozen monthly as-of dataset identity changed')
            if state:
                _verify_month_artifacts(directory, month, state)
            if state and state['status'] == 'complete':
                immutable_json(directory / 'months' / f'{month}.json', state['summary'])
                return state['summary']
            if state and state['status'] == 'finalizing':
                return _publish_finalization(directory, month, state, store)
            if state and state.get('search_exposure'):
                exposed = state['search_exposure']
                store.exposure.assert_prompt_search(**{k: v for k, v in exposed.items() if k != 'data_id'})
            if state is None:
                state = {'data_id': data_id, 'cutoff': cutoff.isoformat(), 'status': 'running',
                         'budget': monthly_budget(config, month).to_dict(), 'pool': FactorPool().to_dict(),
                         'attempts': [], 'next_family': 0, 'baseline': None, 'baseline_model': None,
                         'active_candidate': None, 'agent_audit': [], 'spent_seconds': 0.,
                         'wall_checkpoint': None, 'pending_family': None, 'materializations': [], 'prescreen': []}
                if config.evaluation_mode == 'joint':
                    from .joint_pool import DevelopmentPoolArchive, FrozenPool
                    from .v7_memory import ResearchMemory, search_exposure_context
                    previous = store.db.execute("SELECT month,value FROM months WHERE month<? AND json_extract(value,'$.status')='complete' ORDER BY month DESC LIMIT 1",
                                                (month,)).fetchone()
                    prior = json.loads(previous[1]) if previous else {}
                    if previous:
                        _verify_month_artifacts(directory, previous[0], prior)
                    reference = FrozenPool.from_dict(prior['development_pool']) if prior.get('development_pool') else FrozenPool()
                    state['frozen_reference_pool'] = reference.to_dict()
                    state['development_pool'] = reference.to_dict()
                    if prior.get('joint_archive'):
                        old_archive = DevelopmentPoolArchive.from_dict(prior['joint_archive'])
                        state['joint_archive'] = old_archive.rollover(month_start(cutoff),
                            directory / 'archives' / f'{previous[0]}.json').to_dict()
                    else:
                        state['joint_archive'] = DevelopmentPoolArchive(month_start(cutoff)).to_dict()
                    state['profile_attempted_ids'] = []
                    state['search_exposure'] = search_exposure_context(data_id, cutoff, config.symbols, feed_identity, spec=spec)
                    exposed = state['search_exposure']
                    store.exposure.record(source=exposed['source'], batch=exposed['batch'],
                        symbols=exposed['symbols'], start=exposed['start'], end=exposed['end'], role='S',
                        detail={'data_id': data_id, 'scope': 'exposed_development_only'})
                    store.exposure.assert_prompt_search(source=exposed['source'], batch=exposed['batch'],
                        symbols=exposed['symbols'], start=exposed['start'], end=exposed['end'])
                    if prior.get('research_memory'):
                        previous_memory = ResearchMemory.from_state(prior['research_memory'])
                        memory = previous_memory.rollover(month_start(cutoff),
                            directory / 'memories' / f'{previous[0]}.json')
                    else:
                        memory = ResearchMemory.initial(month_start(cutoff), [
                            {'program': seed, 'citation': 'Registered V7 known multiscale volatility seed; no novelty claimed.'}
                            for seed in literature_seeds()])
                    state['research_memory'] = memory.to_state()
                store.save(month, state, 'month_started')

            invocation_started = preparation.identity['execution_envelope']['issuer']['monotonic_started']
            prior_spent = 0.
            if state['wall_checkpoint'] is not None:
                unobserved = time.time() - state['wall_checkpoint']
                if unobserved < 0:
                    raise ValueError('Research wall clock moved backward; cannot reset its budget')
                # Conservative crash accounting includes downtime since last
                # durable checkpoint, never silently resets a monthly deadline.
                prior_spent += unobserved

            def checkpoint(event):
                preparation.check()
                state['spent_seconds'] = prior_spent + time.monotonic() - invocation_started
                state['wall_checkpoint'] = time.time()
                store.save(month, state, event)
                if external_deadline is not None and time.monotonic() >= external_deadline:
                    raise BudgetExhausted('External shared controls wall-time limit')

            def persist(value):
                state['budget'] = value
                checkpoint('budget_reserved')
                if state['spent_seconds'] > config.research_seconds:
                    raise BudgetExhausted('Monthly wall-time limit, including conservative crash accounting')

            preparation.seal_input(data_id=data_id, search_exposure=state['search_exposure'])
            bridge = WorkAuthorityBridge(store, month, state, authority, persist, preparation=preparation)
            scheduler = ScheduledSearch(bridge, schedule)
            pool = FactorPool.from_dict(state['pool'])
            reference_pool = archive = context = memory = None
            if config.evaluation_mode == 'joint':
                from .joint_pool import DevelopmentPoolArchive, FrozenPool
                reference_pool = FrozenPool.from_dict(state['frozen_reference_pool'])
                archive = DevelopmentPoolArchive.from_dict(state['joint_archive'])
                context = DevelopmentPoolArchive.context(reference_pool, cutoff=cutoff,
                    data_hash=data_id, protocol_hash=spec.digest, source_hash=fingerprint(store.identity['source_files']))
                from .v7_memory import ResearchMemory
                memory = ResearchMemory.from_state(state['research_memory'])

            def persist_memory(value):
                state['research_memory'] = value
                checkpoint('research_memory_updated')
            if state['active_candidate']:
                pending = state['active_candidate']
                if memory and pending['formula_id'] in pool.records:
                    evidence_id = memory.bind_result(store, store.exposure, month, pending['formula_id'])
                    memory.register_measured_parent(family_id=pending['family'], evidence_id=evidence_id,
                                                    store=store, exposure=store.exposure)
                    persist_memory(memory.to_state())
                    state['attempts'].append({**pending, 'status': 'measured_recovered_budget_retained'})
                else:
                    state['attempts'].append({**pending, 'status': 'interrupted_consumed'})
                state['active_candidate'] = None
                checkpoint('interrupted_candidate_retained')
            def baseline_work(budget):
                from .joint_pool import hour_equal_weights
                readout = _baseline_readout(data, cutoff, budget)
                train = data.train_mask(cutoff)
                budget.consume('solve')
                model = fit_ridge(data.frame.take(train), data.target[train], ridge=spec.ridge,
                                  sample_weight=hour_equal_weights(data.frame.metadata.loc[train]),
                                  provenance=data.training_identity(cutoff))
                state['baseline_model'] = ModelRegistry(directory / 'models').save(model, {
                    'rolling_hash': spec.digest, 'library': library_schema(), 'train': data.training_identity(cutoff)})
                state['baseline'] = readout
                checkpoint('cold_baseline_fitted')
                return {'baseline': state['baseline'], 'model_id': state['baseline_model']}
            outcome = scheduler.run('baseline', baseline_work, input_binding={'data_id': data_id})
            if outcome['status'] != 'completed':
                raise ValueError('Required baseline work did not complete')
            started = time.monotonic()
            seeds = literature_seeds()
            stop = state['spent_seconds'] >= config.research_seconds
            while state['next_family'] < (1 if config.budget_policy == 'controls_third' else 3) and not stop:
                index = state['next_family']
                family_id = f'{month}:local:{index}'
                budget_family_id = family_id
                island = 'refinement' if index % 2 == 0 else 'exploration'
                def family_work(budget, index=index, island=island):
                    family_id = f'{month}:local:{index}'
                    budget_family_id = family_id
                    if memory:
                        budget.check_time()
                        memory.automatic_revive(store, store.exposure, island=island, persist=persist_memory)
                        budget.check_time()
                    if config.search_mode == 'gpt':
                        from .program_contract import Program
                        from .program_syntax import CompactProgram
                        from .prompt_builder import PromptIdentity, build_formula_prompt
                        from .v7_agent import CodexBatchBackend, propose_batch
                        identity = PromptIdentity('volatility12h', spec.digest, reference_pool.digest if reference_pool else pool.digest,
                                                  fingerprint(library_schema()))
                        parents = []
                        parent_seeds = [(f'literature{i}', program) for i, program in enumerate(seeds)]
                        if memory:
                            resolved = memory.prompt_parents(store, store.exposure, island=island,
                                                             current_context=context)
                            state['parent_provenance'] = resolved['receipt']
                            parent_seeds = [(p['parent_id'], p['program']) for p in resolved['parents']]
                        for parent_id, program in parent_seeds:
                            parents.append({'parent_id': parent_id,
                                'program': CompactProgram.from_program(Program.from_dict(program)).to_dict(),
                                'evidence': {'evidence_id': parent_id, 'role': 'S', 'task_id': identity.task_id,
                                             'rolling_id': identity.rolling_id, 'pool_id': identity.pool_id,
                                             'summary': 'Verified S ancestor or cited literature seed; prior-context formula only, no current gain claimed.',
                                             'metrics': {}}})
                            if memory:
                                parent_record = next(p for p in resolved['parents'] if p['parent_id'] == parent_id)
                                if parent_record['kind'] == 'research':
                                    prior_readouts = []
                                    for evidence_id in parent_record['evidence_sources'][:2]:
                                        verified = memory.resolver(store, store.exposure)(evidence_id)
                                        origin_state = store.get(verified['month'])
                                        origin_result = origin_state['pool']['records'][verified['formula_id']]['result']
                                        prior_readouts.append({'cutoff': origin_state['cutoff'],
                                            'data_id': verified['data_id'], 'pool_id': origin_result['pool_hash'],
                                            'result_hash': verified['result_hash'],
                                            'gain_F': verified['prior_gain_F'], 'gain_S': verified['prior_gain_S']})
                                    parents[-1]['evidence']['prior_readouts'] = prior_readouts
                                    parents[-1]['evidence']['summary'] = 'Verified ancestor S readout relative to its original frozen Pv, equal-hour aggregation; prior context only; current gain unavailable.'
                                else:
                                    parents[-1]['evidence']['summary'] = 'Cited known literature seed; no measured candidate gain or novelty claimed.'
                        for key, record in ([] if memory else pool.shortlist()[:2]):
                            metrics = record['result']['channels']['S']
                            clean = {k: metrics[k] for k in ('loss', 'reference_loss', 'gain') if metrics[k] is not None}
                            parents.append({'parent_id': key, 'program': CompactProgram.from_program(
                                Program.from_dict(record['program'])).to_dict(), 'evidence': {
                                'evidence_id': key, 'role': 'S', 'task_id': identity.task_id,
                                'rolling_id': identity.rolling_id, 'pool_id': identity.pool_id,
                                'summary': ('Measured equal-hour development readout relative to frozen Pv; exposed evidence.'
                                            if reference_pool else 'Measured equal-fold development readout relative to K0; exposed evidence.'),
                                'metrics': clean}})
                        packet = build_formula_prompt(identity=identity,
                            task_semantics=('Predict 12h realized volatility; development-only logMSE above frozen Pv.'
                                            if reference_pool else 'Predict 12h realized volatility; development-only logMSE above K0.'),
                            feature_schema=[{'name': name, 'unit': 'squared log return' if i < 8 else 'dimensionless',
                                             'preprocessing': 'raw K0; training-only downstream scale'}
                                            for i, name in enumerate(FEATURE_NAMES)],
                            input_semantics=f'Full past {config.lookback} hour K0 windows, raw units; gaps restart.',
                            parents=parents, budget=budget, diagnostics=[{
                                'evidence_id': fingerprint({'data_id': state['data_id'], 'diagnostic': row}),
                                'role': 'S', 'task_id': identity.task_id, 'rolling_id': identity.rolling_id,
                                'pool_id': identity.pool_id, 'summary': row['summary'],
                                'metrics': {'loss': row['loss'], 'support': row['support']}}
                                for row in state['baseline'].get('residual_diagnostics', [])],
                            failure_counts={'failed': sum(a['status'] != 'scored' for a in state['attempts'])})
                        checkpoint('prompt_context_persisted')
                        from .prompt_request_kinds import (enabled as kinds_enabled, semantic_packet,
                            structural_packet, request_structure)
                        specialized = kinds_enabled(store.identity['config'])
                        original_packet, last_reply = packet, None
                        batch = None
                        for attempt_in_family in range(2):
                            if specialized and attempt_in_family == 1:
                                if index == 2:
                                    break  # This slot belongs only to the structural request.
                                if index == 0:
                                    packet = semantic_packet(original_packet, last_reply, budget)
                                    if packet is None:
                                        break
                            previous_audit = list(state['agent_audit'])
                            attempt_audit = []
                            def persist_attempt(value, previous_audit=previous_audit, attempt_audit=attempt_audit):
                                state['agent_audit'] = previous_audit + json.loads(json.dumps(attempt_audit))
                                checkpoint('proposal_audit_updated')
                                budget.persist(value)
                            try:
                                def bounded_backend(packet, *, model, reasoning):
                                    nonlocal last_reply
                                    budget.check_time()
                                    remaining = budget.lease.assert_parent_alive().remaining_seconds
                                    chosen = backend or CodexBatchBackend(directory / 'agent', timeout_seconds=remaining)
                                    from .calibration_trial_context import TrialScopedCodexBackend
                                    from .full_trial_controller import RegisteredTrialCodexBackend
                                    reply = (chosen.call_reserved(packet, model=model, reasoning=reasoning, work_budget=budget)
                                             if type(chosen) in (TrialScopedCodexBackend, RegisteredTrialCodexBackend) else
                                             chosen(packet, model=model, reasoning=reasoning))
                                    budget.check_time()
                                    last_reply = reply
                                    return reply
                                batch = propose_batch(packet, backend=bounded_backend,
                                    budget=budget, feature_names=FEATURE_NAMES,
                                    allowed_parent_ids={p['parent_id'] for p in parents}, audit=attempt_audit,
                                    persist_budget=persist_attempt)
                                break
                            except LocalQuotaExhausted:
                                raise
                            except ValueError as exc:
                                # Only a reply validation failure may use the second
                                # originally prepaid attempt; verify all trusted state first.
                                if (budget._closed or not state['agent_audit']
                                        or state['agent_audit'][-1]['status'] != 'failed'
                                        or 'backend_version' not in state['agent_audit'][-1]):
                                    raise
                                budget.check_time()
                                state['attempts'].append({'family': family_id, 'status': 'proposal_failed',
                                                          'error': str(exc)[:500]})
                                checkpoint('proposal_failed')
                        if batch is None:
                            return {'programs': [], 'family_id': family_id, 'budget_family_id': budget_family_id}
                        if specialized and index == 2 and config.nonlinear_mode == 'isolated':
                            structure = structural_packet(original_packet, budget)
                            previous_audit = list(state['agent_audit'])
                            attempt_audit = []
                            def persist_structure(value):
                                state['agent_audit'] = previous_audit + json.loads(json.dumps(attempt_audit))
                                checkpoint('structural_request_audit_updated')
                                budget.persist(value)
                            try:
                                request_structure(structure, budget=budget, backend=bounded_backend,
                                    audit=attempt_audit, persist_budget=persist_structure)
                            except LocalQuotaExhausted:
                                raise
                            except ValueError:
                                if (budget._closed or not attempt_audit
                                        or attempt_audit[-1]['status'] != 'failed'
                                        or 'backend_version' not in attempt_audit[-1]):
                                    raise
                                budget.check_time()  # Invalid replies remain charged and unused.
                        budget_family_id = batch.family_id
                        family_id = f'{month}:gpt:{fingerprint(batch.family_id)[:32]}' if memory else batch.family_id
                        if memory:
                            proposal = batch.to_dict() | {'family_id': family_id}
                            memory.register_proposal(proposal, island=island,
                                next_experiment='Test conditional F/P/S on the frozen three development folds.',
                                minimum_progress=.005, remaining_budget=24,
                                stop_conditions='Dormant after two adjacent monthly checks without progress; no gate changes.')
                            persist_memory(memory.to_state())
                        programs = [p.to_dict() for p in batch.programs]
                        if batch.template.to_dict()['slots']:
                            from .local_mutation import expand_template
                            seen = {p.digest for p in batch.programs}
                            try:
                                for variant in expand_template(batch.template, family_id=budget_family_id,
                                                               budget=budget, seen=seen):
                                    state['materializations'].append({k: v for k, v in variant.items() if k != 'program'})
                                    if variant['status'] == 'valid':
                                        programs.append(variant['program'])
                                    checkpoint('template_materialized')
                            except LocalQuotaExhausted:
                                checkpoint('template_budget_reached')
                                raise
                        if reference_pool is None:
                            programs, prescreen = _prescreen(data, cutoff, programs, config.lookback, config.formulas_per_family)
                        else:
                            programs, prescreen = _prescreen(data, cutoff, programs, config.lookback, config.formulas_per_family,
                                                            reference_pool=reference_pool)
                        state['prescreen'] += prescreen
                    else:
                        state['pending_family'] = family_id
                        checkpoint('family_planned')
                        if family_id not in budget.families:
                            budget.register_family(family_id)
                        local_parents = None
                        if memory:
                            local_parents = memory.prompt_parents(store, store.exposure, island=island,
                                                                 current_context=context)['parents']
                        if memory and family_id not in memory.proposals:
                            memory.register_proposal({'family_id': family_id, 'parent_ids': [local_parents[-1]['parent_id']],
                                'hypothesis': 'Known volatility mechanism with bounded local changes.',
                                'mechanism': 'Past multiscale volatility and range discrepancy.',
                                'competing_explanation': 'The existing K0/Pv already spans this information.',
                                'falsification': 'No common-support conditional marginal gain.'}, island=island,
                                next_experiment='Same-access F/P/S conditional comparison.', minimum_progress=.005,
                                remaining_budget=24, stop_conditions='Two adjacent cycles without progress.')
                            persist_memory(memory.to_state())
                        programs = []
                        rng = np.random.default_rng(index + 702)
                        if memory and config.search_mode == 'local':
                            from .v7_search_ops import local_variations
                            parent = local_parents[-1]
                            mutations = local_variations(parent['program'], local_parents[-2]['program'],
                                config.formulas_per_family, feature_names=FEATURE_NAMES, seed=index + 702)
                            records = memory.materialize(parent['parent_id'], mutations, family_id=family_id,
                                budget=budget, store=store, exposure=store.exposure,
                                seen={fingerprint(record['program']) for record in pool.records.values()})
                            state['materializations'].extend({k: v for k, v in r.items() if k != 'program'} for r in records)
                            programs = [r['program'] for r in records if r['status'] == 'valid']
                        for i in range(0 if memory and config.search_mode == 'local' else config.formulas_per_family):
                            pair = (i + index) % 2
                            program = json.loads(json.dumps(seeds[pair]))
                            if index == 0 and i < 2:
                                pass
                            elif config.search_mode == 'random':
                                program['outputs'][0]['arg']['right']['name'] = FEATURE_NAMES[int(rng.integers(0, 8))]
                            else:
                                program['outputs'][0] = {'op': 'std' if index == 1 else 'mean',
                                                         'arg': program['outputs'][0]['arg']}
                                program['outputs'][0]['arg']['right']['name'] = FEATURE_NAMES[(i + index) % 8]
                            budget.consume('ast', family_id=family_id)
                            programs.append(program)
                    state['pending_family'] = None
                    checkpoint('family_materialized')
                    if memory:
                        programs = list({fingerprint(formula_identity(p, config.lookback)): p for p in programs}.values())
                        scheduled = memory.reserve_schedule([
                            {'candidate_id': fingerprint(formula_identity(program, config.lookback)),
                             'family_id': budget_family_id,
                             'lane': 'blind' if position >= max(1, len(programs) - 2)
                                     else 'refine' if island == 'refinement' else 'new'}
                            for position, program in enumerate(programs)
                            if fingerprint(formula_identity(program, config.lookback)) not in memory.scheduled],
                            budget, persist=persist_memory)
                        admitted_ids = {r['candidate_id'] for r in scheduled['scheduled']}
                        programs = [p for p in programs if fingerprint(formula_identity(p, config.lookback)) in admitted_ids]
                    result = {'programs': programs, 'family_id': family_id, 'budget_family_id': budget_family_id}
                    structural_audits = [a for a in state['agent_audit'] if a['request_kind'] == 'structural_probe']
                    if index == 2 and structural_audits:
                        result['structural_audit_hash'] = fingerprint(structural_audits[-1])
                    return result
                outcome = scheduler.run(f'family:{index}', family_work,
                    input_binding={'pool': reference_pool.digest, 'island': island, 'index': index})
                state['next_family'] += 1
                if outcome['status'] == 'completed':
                    programs = outcome['value']['programs']
                    family_id = outcome['value']['family_id']
                    budget_family_id = outcome['value']['budget_family_id']
                else:
                    programs = []
                    state['attempts'].append({'family': family_id, 'status': outcome['status']})
                checkpoint('fixed_family_closed')
                for position in range(8):
                    slot_key = f'candidate:{index}:{position}'
                    if position >= len(programs):
                        scheduler.skip(slot_key, reason='no_candidate')
                        continue
                    program = programs[position]
                    if prior_spent + time.monotonic() - invocation_started >= config.research_seconds:
                        raise TimeoutError('Original S deadline reached before remaining fixed slots')
                    identifier = fingerprint(formula_identity(program, config.lookback))
                    if identifier in pool.records:
                        state['attempts'].append({'formula_id': identifier, 'family': family_id, 'status': 'duplicate'})
                        checkpoint('exact_duplicate')
                        scheduler.skip(slot_key, reason='no_candidate')
                        continue
                    def candidate_work(budget, program=program, identifier=identifier,
                                       family_id=family_id, budget_family_id=budget_family_id):
                        from .v7_s_controller import candidate_solve_bound
                        required_solves = candidate_solve_bound(reference_pool, parameterized=bool(program.get('parameters')))
                        if budget.record['counts']['solve'] < required_solves:
                            raise LocalQuotaExhausted('Fixed candidate slot cannot fund its unchanged cold-fold solve bound')
                        budget.consume('f', family_id=budget_family_id)
                        state['active_candidate'] = {'formula_id': identifier, 'family': family_id}
                        checkpoint('candidate_started')
                        if reference_pool is None:
                            result = evaluate_formula(data, cutoff, program, lookback=config.lookback,
                                                      budget=budget, profile=len(pool.records) < config.profile_candidates)
                        else:
                            from .joint_pool import PoolMember, evaluate_candidate
                            has_parameters = bool(program.get('parameters'))
                            if has_parameters and len(state['profile_attempted_ids']) >= config.profile_candidates:
                                raise LocalQuotaExhausted('Parameterized structure quota reached; no uncounted P fitting')
                            if has_parameters:
                                state['profile_attempted_ids'].append(identifier)
                                checkpoint('parameter_structure_reserved')
                            member = PoolMember(program, lookback=config.lookback,
                                parameter_mode='profile' if has_parameters else 'fixed', family=family_id)
                            result = evaluate_candidate(data, cutoff, reference_pool, member, budget=budget, context=context)
                            result['formula_id'] = identifier
                            result['channels'] = {name: {'loss': result[name]['Pv']['candidate_loss'],
                                'reference_loss': result[name]['Pv']['reference_loss'], 'gain': result[name]['Pv']['gain']}
                                for name in ('F', 'S')}
                            archive.record(member, result, context)
                            state['joint_archive'] = archive.to_dict()
                        path = directory / 'results' / f'{month}-{identifier}.json'
                        immutable_json(path, result)
                        pool.add(program, result, family=family_id, lookback=config.lookback)
                        state['pool'] = pool.to_dict()
                        checkpoint('measured_result_persisted')
                        if memory:
                            evidence_id = memory.bind_result(store, store.exposure, month, identifier)
                            memory.register_measured_parent(family_id=family_id, evidence_id=evidence_id,
                                                            store=store, exposure=store.exposure)
                            persist_memory(memory.to_state())
                        state['attempts'].append({'formula_id': identifier, 'family': family_id, 'status': 'scored',
                                                  'gain_F': result['channels']['F']['gain'], 'gain_S': result['channels']['S']['gain']})
                        checkpoint('candidate_result_checkpointed')
                        return {'formula_id': identifier, 'result_hash': fingerprint(result)}
                    outcome = scheduler.run(slot_key, candidate_work,
                        input_binding={'formula': program, 'lookback': config.lookback, 'context': context})
                    if outcome['status'] != 'completed':
                        state['attempts'].append({'formula_id': identifier, 'family': family_id,
                                                  'status': outcome['status']})
                    state['active_candidate'] = None
                    checkpoint('candidate_finished')
                    if stop:
                        break
            shortlist = [{'formula_id': k, 'program': v['program'], 'lookback': v['lookback'],
                          'gain_F': v['result']['channels']['F']['gain'],
                          'gain_S': v['result']['channels']['S']['gain']} for k, v in pool.shortlist()]
            if reference_pool is not None:
                from .joint_pool import PoolMember, stability_select
                observations = {}
                for family_id, proposal in memory.proposals.items():
                    observations.setdefault(proposal['version'], []).extend(
                        (identifier, record) for identifier, record in pool.records.items() if record['family'] == family_id)
                # Only a genuinely re-evaluated original parent can advance an
                # older hypothesis clock; new descendants do not impersonate it.
                for parent in memory.book.parents_by_id.values():
                    version = parent['hypothesis_version']
                    if version:
                        observations.setdefault(version, []).extend(
                            (identifier, record) for identifier, record in pool.records.items()
                            if record['program'] == parent['program'])
                for version, group in observations.items():
                    hypothesis = memory.book.versions.get(version)
                    if not group or not hypothesis or hypothesis['state'] != 'active':
                        continue
                    if hypothesis['history'] and hypothesis['history'][-1]['month'] == cutoff.isoformat():
                        continue
                    group = list({identifier: record for identifier, record in group}.items())
                    best_id, best = max(group, key=lambda pair: pair[1]['result']['channels']['F']['gain']
                                       if pair[1]['result']['channels']['F']['gain'] is not None else -1e100)
                    evidence_id = memory.bind_result(store, store.exposure, month, best_id)
                    memory.observe(version, store=store, exposure=store.exposure, month=month_start(cutoff),
                        progress=best['result']['channels']['F']['gain'] or 0.,
                        evidence_sources=[evidence_id], spent=min(len(group), hypothesis['remaining_budget']),
                        failure='No meaningful conditional increment' if (best['result']['channels']['F']['gain'] or 0.) < .005 else None)
                persist_memory(memory.to_state())
                results = [r['result'] for r in pool.records.values()]
                selection = stability_select(results, reference_pool)
                for record in pool.records.values():
                    member = PoolMember(record['program'], lookback=record['lookback'], family=record['family'],
                        parameter_mode='profile' if record['program'].get('parameters') else 'fixed')
                    if member.member_id in selection['selected_ids']:
                        if len(reference_pool.members) < 8:
                            reference_pool, admission = archive.admit(reference_pool, member, month=month_start(cutoff), selection=selection)
                            state['pool_admission'] = admission
                        else:
                            state['pool_admission'] = {'status': 'capacity_deferred', 'affects_A0': False}
                        break
                if reference_pool.members and not state.get('prune_attempted'):
                    from .joint_pool import backward_prune
                    state['prune_attempted'] = True
                    checkpoint('pruning_budget_reserved')
                    def pruning_work(budget):
                        nonlocal reference_pool
                        pruned, pruning = backward_prune(data, cutoff, reference_pool, budget=budget, max_remove=1)
                        state['pruning'] = pruning
                        for step in pruning['steps']:
                            for trial in step['trials']:
                                member_id = trial['removed']
                                if member_id in archive.lifecycle:
                                    archive.monitor_decay(member_id, month=month_start(cutoff), k0_gain=None,
                                        pool_gain=1-step['baseline_loss']/trial['loss'] if trial['loss'] > 0 else None)
                        for member_id in pruning['removed_ids']:
                            if member_id in archive.lifecycle:
                                archive.lifecycle[member_id]['state'] = 'retired'
                        reference_pool = pruned
                        checkpoint('pruning_result_checkpointed')
                        return state['pruning']
                    outcome = scheduler.run('pruning', pruning_work, input_binding={'pool': reference_pool.digest})
                    if outcome['status'] != 'completed':
                        state['pruning'] = {'status': outcome['status'], 'affects_A0': False}
                elif state.get('prune_attempted') and not state.get('pruning'):
                    state['pruning'] = {'status': 'interrupted_consumed', 'affects_A0': False}
                if not state.get('prune_attempted'):
                    scheduler.skip('pruning', reason='no_pool_to_prune')
                state['development_pool'] = reference_pool.to_dict()
                state['joint_archive'] = archive.to_dict()
            nonlinear_report = None
            if config.nonlinear_mode != 'disabled':
                from .s_nonlinear_controller import run_shared_nonlinear
                checkpoint('shared_nonlinear_controller_handoff')
                nonlinear_report = run_shared_nonlinear(bridge, data, cutoff,
                    mode=config.nonlinear_mode, feed_directory=feed_directory)
                checkpoint('shared_nonlinear_controller_closed')
            summary = {'version': 'V7 construction pilot', 'month': month, 'cutoff': cutoff.isoformat(),
                       'scope': 'exposed_development_only', 'accounting': 'shared_authority_fixed_schedule_v1',
                       'serving_eligible': False, 'hard_deadline_enforced': False,
                       'schedule_hash': schedule.digest, 'authority_manifest_hash': authority.manifest_hash,
                       'search_mode': config.search_mode,
                       'baseline': state['baseline'], 'baseline_model': state['baseline_model'],
                       'budget': state['budget'], 'attempts': state['attempts'], 'shortlist': shortlist,
                       'formal_confirmation_enabled': False, 'A0': 'unchanged_by_S',
                       'evaluation_mode': config.evaluation_mode,
                       'development_pool': state.get('development_pool'),
                       'nonlinear': nonlinear_report,
                       'search_seconds_this_invocation': time.monotonic() - started,
                       'monthly_seconds': prior_spent + time.monotonic() - invocation_started,
                       'limitations': ['Full-pipeline calibration not accepted', 'Linux resource envelope not tested',
                                       'No independent forward confirmation',
                                       'Legacy single reference is K0' if reference_pool is None else 'Joint pool updates are development evidence only']}
            state['summary'], state['status'] = summary, 'finalizing'
            state['wall_checkpoint'] = None
            state['spent_seconds'] = summary['monthly_seconds']
            store.save(month, state, 'month_finalization_committed')
            return _publish_finalization(directory, month, state, store)
        finally:
            store.close()


def replay_baseline(directory, bars, cutoffs, *, spec=None):
    """Three-or-more cold monthly K0 refits with logical forecast recovery.

    Historical replay is development engineering evidence. Receipt provenance
    remains in model artifacts and cannot be relabeled as independent forward.
    """
    spec = spec or RollingSpec()
    data = build_rolling_data(bars, spec)
    symbols = tuple(sorted(bars.symbol.unique()))
    if len(symbols) != 1:
        raise ValueError('Initial source-frozen V7 replay requires one symbol')
    directory = project_path(directory)
    with FileLock(str(directory) + '.lock'):
        if (directory / 'forecasts.sqlite').exists() and not (directory / 'manifest.json').exists():
            raise ValueError('Existing unfrozen replay ledger cannot be retroactively registered; use a new run')
        store = RunStore(directory, V7Config(symbols=symbols, formulas_per_family=1), spec, mode='baseline_replay')
        try:
            return _replay_baseline_locked(directory, data, cutoffs, spec, store)
        finally:
            store.close()


def _replay_baseline_locked(directory, data, cutoffs, spec, store):
    registry = ModelRegistry(project_path(directory) / 'models')
    ledger = ForecastLedger(project_path(directory) / 'forecasts.sqlite')
    results = []
    try:
        for cutoff in cutoffs:
            _, end = spec.window(cutoff)
            next_month = end + (pd.DateOffset(months=1) if spec.window_days is None else pd.Timedelta(days=15))
            train = data.train_mask(end)
            provenance = data.training_identity(end)
            month = end.strftime('%Y-%m') if spec.window_days is None else end.isoformat()
            previous = store.get(month)
            if previous and previous['train'] != provenance:
                raise ValueError('Frozen replay training snapshot changed')
            if previous:
                registry.load(previous['model'])
            if spec.task == 'first_event':
                from .event_eval import build_event_library, fit_event
                frame = build_event_library(data.frame)
                model = fit_event(frame.take(train), data.target[train], provenance=provenance, ridge=spec.ridge)
            else:
                frame = data.frame
                model = fit_ridge(frame.take(train), data.target[train], provenance=provenance, ridge=spec.ridge)
            pipeline = {'rolling_hash': spec.digest, 'train': provenance, 'input_schema': frame.schema_hash}
            identifier = registry.save(model, pipeline)
            restored, _ = registry.load(identifier, expected_pipeline=pipeline)
            selected = ((frame.metadata.available_at >= end) & (frame.metadata.available_at < next_month)).to_numpy() & frame.valid_rows
            prediction = restored.predict(frame.take(selected))
            meta = frame.metadata.loc[selected]
            inserted = 0
            for row, value in zip(meta.itertuples(), prediction, strict=True):
                inserted += ledger.put(run='v7-baseline', stream='A0', origin=row.available_at,
                    symbol=row.symbol, task=spec.task, model_id=identifier,
                    pipeline_id=spec.digest, prediction=value.tolist())
            results.append({'cutoff': end.isoformat(), 'model': identifier, 'train': provenance,
                            'origins': int(selected.sum()), 'new_predictions': inserted,
                            'availability_mode': data.availability_mode, 'formal_claim': False})
            store.save(month, {'train': provenance, 'model': identifier}, 'replay_model_frozen')
    finally:
        ledger.close()
    return results


def inspect_run(directory):
    """Historical read-only inspection verifies saved bytes, not current code."""
    directory = project_path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text('utf-8'))
    if manifest['hash'] != fingerprint(manifest['identity']):
        raise ValueError('V7 manifest integrity mismatch')
    store = RunStore.__new__(RunStore)
    store.directory, store.identity = directory, manifest['identity']
    store.db = sqlite3.connect(f'file:{(directory / "state.sqlite").as_posix()}?mode=ro', uri=True)
    try:
        result = store.verify()
        summaries = []
        for month, text in store.db.execute('SELECT month,value FROM months ORDER BY month'):
            state = json.loads(text)
            if 'pool' in state:
                _verify_month_artifacts(directory, month, state)
            elif state.get('model'):
                ModelRegistry(directory / 'models').load(state['model'])
            if state.get('status') == 'complete':
                path = directory / 'months' / f'{month}.json'
                if not path.is_file() or json.loads(path.read_text('utf-8')) != state['summary']:
                    raise ValueError('V7 completed summary artifact integrity mismatch')
                summaries.append(state['summary'])
        return {**result, 'summaries': summaries, 'mode': 'historical_read_only',
                'source_matches_current': manifest['identity']['source_files'] == source_files()}
    finally:
        store.close()


