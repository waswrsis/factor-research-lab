"""Frozen complete-learner catalogue and descriptive S-only algorithm shortlist.

This is not factor novelty, a statistical claim, or a V/C/adoption ticket. The
ordinary Lifecycle factor1..3 confirmation path MUST NOT consume these entries.
A separate typed algorithm-validation contract remains necessary.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime

import pandas as pd
from filelock import FileLock

from factorlab.protocol import fingerprint, project_path

from .neural_forward import FrozenNeuralRecipe
from .neural_forward_ticket import open_existing_authority
from .neural_research_launcher import verify_completed_research
from .neural_research_ticket import ResearchFitTicket, _binding, _reader
from .nonlinear_channel import StructureSpec
from .rolling import development_folds
from .v7_storage import immutable_record
from .work_authority_bridge import WorkAuthorityBridge

VERSION = 'v7-complete-algorithm-catalogue-v1'
PROPOSAL_VERSION = 'v7-registered-algorithm-proposal-v1'
STRUCTURES = tuple((w, d) for w in (4, 8, 16, 32) for d in (2, 4, 8) if d <= w)


def _copy(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _read(path):
    if not path.is_file() or path.stat().st_size > 2**20:
        raise ValueError('Missing/oversized algorithm catalogue evidence')
    value = json.loads(path.read_text('utf-8'))
    if set(value) != {'value', 'hash'} or fingerprint(value['value']) != value['hash']:
        raise ValueError('Algorithm catalogue evidence checksum changed')
    return value['value']


def _write(path, value):
    if len(json.dumps(value, sort_keys=True, allow_nan=False).encode()) > 2**20 - 256:
        raise ValueError('Algorithm catalogue evidence exceeds 1 MiB bound')
    immutable_record(path, {'value': value, 'hash': fingerprint(value)})


@dataclass(frozen=True)
class AlgorithmShortlistRule:
    minimum_hour_coverage: float = .8
    minimum_symbol_hour_coverage: float = .8
    minimum_gain: float = 0.
    max_shortlist: int = 2
    version: str = 'v7-algorithm-S-filter-v1'

    def __post_init__(self):
        if (self.version != 'v7-algorithm-S-filter-v1'
                or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= 1
                       for v in (self.minimum_hour_coverage, self.minimum_symbol_hour_coverage))
                or type(self.minimum_gain) not in (int, float) or not math.isfinite(self.minimum_gain)
                or not 0 <= self.minimum_gain < 1
                or type(self.max_shortlist) is not int or not 1 <= self.max_shortlist <= 3):
            raise ValueError('Invalid frozen descriptive algorithm S filter')

    def to_dict(self):
        return {**asdict(self), 'selection': 'ensemble_common_hour_gain_over_both_K0_and_Pv_strictly_above_minimum',
                'coverage': 'both_thresholds_in_every_one_of_three_fixed_folds',
                'epoch0_only': 'retain_record_but_do_not_select',
                'tiebreak': ['descending_min_K0_Pv_gain', 'ascending_ensemble_loss', 'ascending_pipeline_id'],
                'cross_candidate_support': 'identical_sample_ids_required_or_entire_shortlist_closed',
                'diagnostic_controls': 'retained_never_used_as_winners', 'formal_claim': False}

    @classmethod
    def from_dict(cls, value):
        result = cls(**{k: value[k] for k in cls.__dataclass_fields__})
        if result.to_dict() != value:
            raise ValueError('Frozen algorithm selection rule changed')
        return result


def registered_recipes(base, *, structures=((8, 2), (16, 4)), outputs=('residual', 'encoder_ridge')):
    """Finite explicit architecture/output grid; this does not spend or fit anything."""
    if type(base) is not FrozenNeuralRecipe or not structures or not outputs:
        raise TypeError('A complete base recipe and nonempty registered grid are required')
    recipes = []
    for structure in structures:
        if type(structure) is not tuple or structure not in STRUCTURES:
            raise ValueError('Only registered single-expert structures are supported')
        for output in outputs:
            if output not in {'residual', 'encoder_ridge'}:
                raise ValueError('Unknown complete learner output')
            recipes.append(FrozenNeuralRecipe(base.pool, base.rolling_spec,
                replace(base.config, structures=(StructureSpec(*structure, experts=1),)), output))
    if len(recipes) > 24 or len({r.pipeline_id for r in recipes}) != len(recipes):
        raise ValueError('Duplicate or excessive registered algorithms')
    return tuple(recipes)


def _score(result, recipe, rule):
    """Pure private arithmetic; callers cannot submit this output as evidence."""
    if (result.get('role') != 'S' or result.get('status') != 'completed'
            or result.get('test_only') is not False or result.get('formal_claim') is not False
            or result.get('pipeline_id') != recipe.pipeline_id or len(result.get('folds', [])) != 3):
        raise ValueError('Only verified complete-recipe three-fold S evidence is scoreable')
    symbols = result['symbols']
    folds, all_hours, reasons, trained = [], [], [], False
    for index, (fold, expected) in enumerate(zip(result['folds'], development_folds(result['cutoff'], spec=recipe.rolling_spec), strict=True)):
        hours = int((pd.Timestamp(expected['score_end']) - pd.Timestamp(expected['fit_cutoff'])) / pd.Timedelta(hours=1))
        nominal = hours * len(symbols)
        if (fold['status'] != 'completed' or fold.get('role') != 'S'
                or any(fold[k] != expected[k] for k in ('fit_cutoff', 'score_end'))
                or fold['nominal_rows'] != nominal or fold['nominal_hours'] != hours
                or fold['support'] != len(fold['common_sample_ids'])
                or len(set(fold['common_sample_ids'])) != fold['support']):
            raise ValueError('Algorithm fold/support differs from the frozen calendar')
        hourly = fold['hourly']
        seen, support = set(), 0
        for row in hourly:
            stamp = pd.Timestamp(row['hour'])
            if (stamp.tzinfo is None or stamp != stamp.floor('h') or row['hour'] in seen
                    or not pd.Timestamp(expected['fit_cutoff']) <= stamp < pd.Timestamp(expected['score_end'])
                    or type(row['symbols_present']) is not int or not 1 <= row['symbols_present'] <= len(symbols)
                    or not {'ensemble', 'K0', 'Pv'} <= row['losses'].keys()
                    or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in row['losses'].values())):
                raise ValueError('Invalid actual common-hour losses')
            seen.add(row['hour'])
            support += row['symbols_present']
        if support != fold['support'] or fold['coverage'] != support / nominal:
            raise ValueError('Algorithm pair coverage differs from nominal support')
        hour_coverage, pair_coverage = len(hourly) / hours, support / nominal
        if hour_coverage < rule.minimum_hour_coverage or pair_coverage < rule.minimum_symbol_hour_coverage:
            reasons.append(f'fold_{index}_coverage')
        checkpoints = fold['seed_checkpoints']
        if ([s['seed'] for s in checkpoints] != list(recipe.config.seeds)
                or any(type(s['epoch']) is not int or not 0 <= s['epoch'] <= recipe.config.epochs for s in checkpoints)):
            raise ValueError('Seed checkpoint record differs from frozen complete recipe')
        trained |= any(s['epoch'] > 0 for s in checkpoints)
        losses = {k: sum(h['losses'][k] for h in hourly) / len(hourly) for k in ('ensemble', 'K0', 'Pv')} if hourly else {}
        folds.append({'index': index, 'nominal_hours': hours, 'nominal_symbol_hours': nominal,
            'support': support, 'hours': len(hourly), 'hour_coverage': hour_coverage,
            'symbol_hour_coverage': pair_coverage, 'losses': losses,
            'sample_ids_hash': fingerprint(fold['common_sample_ids']), 'seed_checkpoints': checkpoints,
            'diagnostic_losses': {k: v for k, v in fold['losses'].items() if k not in {'ensemble', 'K0', 'Pv'}},
            'mean_seed_loss_diagnostic': fold['mean_seed_loss_diagnostic']})
        all_hours.extend(hourly)
    if not trained:
        reasons.append('epoch0_only')
    losses = {k: sum(h['losses'][k] for h in all_hours) / len(all_hours)
              for k in ('ensemble', 'K0', 'Pv')} if all_hours else {}
    gains = {k: (1 - losses['ensemble'] / losses[k] if losses.get(k, 0) > 0 else None) for k in ('K0', 'Pv')}
    if any(g is None or g <= rule.minimum_gain for g in gains.values()):
        reasons.append('no_strict_ensemble_gain_over_both_references')
    return {'eligible': not reasons, 'reasons': reasons, 'folds': folds, 'losses': losses, 'gains': gains,
            'rank_score': min(gains.values()) if all(g is not None for g in gains.values()) else None,
            'epoch0_only': not trained, 'role': 'S', 'formal_claim': False}


class AlgorithmCatalogue:
    """Frozen directory of full learners; no formula or factor-claim conversion API."""
    def __init__(self, directory):
        self.directory = project_path(directory)
        self.identity = _read(self.directory / 'identity.json')
        self.digest = fingerprint(self.identity)
        self.rule = AlgorithmShortlistRule.from_dict(self.identity['rule'])
        self.recipes = {key: FrozenNeuralRecipe.from_dict(value) for key, value in self.identity['recipes'].items()}
        self._context()

    @classmethod
    def freeze(cls, bridge, recipes, *, rule=None):
        if type(bridge) is not WorkAuthorityBridge:
            raise TypeError('Actual S WorkAuthorityBridge required')
        bridge._guard()
        rule = rule or AlgorithmShortlistRule()
        if type(rule) is not AlgorithmShortlistRule:
            raise TypeError('Typed preregistered algorithm S rule required')
        recipes = tuple(recipes)
        if not 1 <= len(recipes) <= 24 or any(type(r) is not FrozenNeuralRecipe for r in recipes):
            raise ValueError('A bounded complete recipe catalogue is required')
        entries = {}
        for recipe in recipes:
            structure = recipe.config.structures[0]
            if (structure.experts != 1 or (structure.width, structure.representation_dim) not in STRUCTURES
                    or _copy(recipe.pool.to_dict()) != _copy(bridge.state['frozen_reference_pool'])
                    or recipe.rolling_spec.digest != bridge.store.identity['rolling_hash']):
                raise ValueError('Catalogue learner differs from actual frozen Pv/protocol or finite grammar')
            entries[recipe.pipeline_id] = _copy(recipe.to_dict())
        if len(entries) != len(recipes):
            raise ValueError('Duplicate full learner ID')
        directory = bridge.store.directory / 'algorithm_catalogue' / bridge.month
        anchor = bridge.authority.directory / 'algorithm_catalogues' / (bridge.identifier + '.json')
        static = {'version': VERSION, 'run_directory': str(bridge.store.directory), 'replay_month': bridge.month,
            'run_identity_hash': fingerprint(bridge.store.identity), 'bridge_id': bridge.identifier,
            'bridge_identity_hash': fingerprint(bridge.identity), 'controller_data_id': bridge.state['data_id'],
            'input_id': bridge.state['nonlinear_input_data_id'], 'exposure': bridge.state['search_exposure'],
            'authority_directory': str(bridge.authority.directory), 'authority_manifest_hash': bridge.authority.manifest_hash,
            'rule': rule.to_dict(), 'recipes': dict(sorted(entries.items())), 'anchor': str(anchor),
            'factor_claim_eligible': False, 'serving_eligible': False, 'formal_claim': False}
        with FileLock(str(directory) + '.lock', timeout=0):
            if directory.exists() or anchor.exists():
                old = cls(directory)
                if any(old.identity.get(k) != v for k, v in static.items()):
                    raise ValueError('Registered recipes/protocol cannot change after freeze')
                return old
            identity = {**static, 'frozen_at': datetime.now(UTC).isoformat()}
            _write(anchor, identity)
            _write(directory / 'identity.json', identity)
        return cls(directory)

    def _context(self, *, prompt=False):
        identity = self.identity
        if (identity.get('version') != VERSION or _read(self.directory / 'identity.json') != identity
                or fingerprint(identity) != self.digest or _read(project_path(identity['anchor'])) != identity
                or self.directory != project_path(identity['run_directory']) / 'algorithm_catalogue' / identity['replay_month']
                or self.rule.to_dict() != identity['rule']
                or {key: _copy(r.to_dict()) for key, r in self.recipes.items()} != identity['recipes']
                or any(r.pipeline_id != key for key, r in self.recipes.items())):
            raise ValueError('Frozen algorithm catalogue identity changed')
        authority = open_existing_authority(identity['authority_directory'])
        if authority.manifest_hash != identity['authority_manifest_hash']:
            raise ValueError('Algorithm catalogue changed its unique authority')
        reader = _reader(identity['run_directory'])
        try:
            state = reader.get(identity['replay_month'])
            if (fingerprint(reader.identity) != identity['run_identity_hash']
                    or state['data_id'] != identity['controller_data_id']
                    or state['nonlinear_input_data_id'] != identity['input_id']
                    or state['search_exposure'] != identity['exposure']
                    or state.get('authority_bridge', {}).get('identity_hash') != identity['bridge_identity_hash']
                    or (prompt and state['status'] != 'running')):
                raise ValueError('Catalogue no longer has its actual original S context')
            reader.exposure.assert_prompt_search(**{k: v for k, v in identity['exposure'].items() if k != 'data_id'})
        finally:
            reader.close()
        return authority

    def record_ticket(self, path):
        authority = self._context()
        ticket = ResearchFitTicket.load(path)
        value, context = ticket.value, ticket.value['context']
        recipe = FrozenNeuralRecipe.from_dict(value['recipe'])
        if (recipe.pipeline_id not in self.recipes or value['recipe'] != self.identity['recipes'][recipe.pipeline_id]
                or context['bridge_identity_hash'] != self.identity['bridge_identity_hash']
                or context['input_id'] != self.identity['input_id']
                or context['controller_data_id'] != self.identity['controller_data_id']
                or value['authority_manifest_hash'] != authority.manifest_hash
                or pd.Timestamp(value['lease']['wall_started_at']) < pd.Timestamp(self.identity['frozen_at'])):
            raise ValueError('Algorithm evidence was not an original registered S learner')
        record = authority.get(value['reservation_id'])
        base = {k: value[k] for k in value if k not in {
            'data', 'authority_manifest_hash', 'reservation_id', 'terms_hash', 'lease', 'lease_hash'}}
        if (record['binding'] != _binding(base, recipe)
                or any(record[k] != value[k] for k in ('terms_hash', 'lease', 'lease_hash'))):
            raise ValueError('Algorithm evidence differs from canonical prepaid S work')
        reference = {'path': str(ticket.path), 'digest': ticket.digest}
        saved = {'pipeline_id': recipe.pipeline_id, 'catalogue_id': self.digest, 'ticket': reference,
                 'reservation_id': record['reservation_id'], 'reservation_hash': fingerprint(record),
                 'prepaid_counts': record['counts'], 'actual_counts': record.get('actual_counts'),
                 'status': record['status'], 'role': 'S', 'formal_claim': False}
        if record['status'] == 'completed':
            verified = verify_completed_research(ticket.path)
            result = verified['result']
            saved.update(summary=_score(result, recipe, self.rule),
                research=verified['receipt']['research'], publication_hash=fingerprint(verified['publication']))
        elif record['status'] in {'failed_consumed', 'interrupted_consumed'}:
            saved.update(summary=None, failure_reason=record['reason'],
                         evidence_status='canonical_failure_cost_only_no_performance_claim')
        else:
            raise ValueError('Running/reserved work is not a terminal algorithm observation')
        with FileLock(str(self.directory) + '.lock', timeout=0):
            path = self.directory / 'records' / (recipe.pipeline_id + '.json')
            if (self.directory / 'shortlist.json').exists() and not path.exists():
                raise ValueError('Sealed shortlist cannot add later favorable evidence')
            _write(path, saved)
        return saved

    def _records(self):
        records = {}
        for path in sorted((self.directory / 'records').glob('*.json')):
            saved = _read(path)
            key = saved['pipeline_id']
            if path.stem != key or key not in self.recipes or saved['catalogue_id'] != self.digest:
                raise ValueError('Algorithm result is outside its registered catalogue')
            # Revalidate the real evidence rather than trusting cached scores.
            current = self.record_ticket(saved['ticket']['path'])
            if saved != current:
                raise ValueError('Algorithm S evidence changed after recording')
            records[key] = saved
        # The caller cannot turn an omitted/deleted losing record into an
        # unattempted candidate. Recover its actual ticket, or retain canonical
        # prepaid preparation/failure evidence even if no ticket was completed.
        authority = self._context()
        with authority._transaction() as db:
            canonical = _copy(list(authority._verify(db)['reservations'].values()))
        matched = set()
        for record in canonical:
            binding = record['binding']
            key = binding.get('pipeline_id')
            if (binding.get('research_bridge_id') != self.identity['bridge_id']
                    or binding.get('input_id') != self.identity['input_id'] or key not in self.recipes
                    or pd.Timestamp(record['lease']['wall_started_at']) < pd.Timestamp(self.identity['frozen_at'])):
                continue
            if key in matched:
                raise ValueError('More than one prepaid attempt for the same registered learner')
            matched.add(key)
            if key in records:
                if records[key]['reservation_id'] != record['reservation_id']:
                    raise ValueError('Algorithm record omitted a different actual attempt')
                continue
            work_key = record['work_key']
            if not work_key.startswith('neural_S:'):
                raise ValueError('Unexpected canonical S work key')
            suffix = work_key.removeprefix('neural_S:')
            if len(suffix) != 64 or any(c not in '0123456789abcdef' for c in suffix):
                raise ValueError('Noncanonical complete S work directory')
            ticket_path = authority.directory / 'research_tickets' / suffix / 'ticket.json'
            if ticket_path.is_file() and record['status'] in {'completed', 'failed_consumed', 'interrupted_consumed'}:
                records[key] = self.record_ticket(ticket_path)
            else:
                if record['status'] == 'completed':
                    raise ValueError('Completed algorithm ticket missing; cannot erase observed evidence')
                records[key] = {'pipeline_id': key, 'catalogue_id': self.digest,
                    'reservation_id': record['reservation_id'], 'reservation_hash': fingerprint(record),
                    'status': record['status'], 'prepaid_counts': record['counts'], 'actual_counts': record.get('actual_counts'),
                    'summary': None, 'role': 'S', 'formal_claim': False,
                    'evidence_status': 'canonical_prepaid_intent_only_not_performance',
                    'failure_reason': record.get('reason')}
        return records

    def prompt_packet(self):
        self._context(prompt=True)
        records = self._records()
        return {'version': PROPOSAL_VERSION, 'catalogue_id': self.digest, 'role': 'S',
            'context': {'controller_data_id': self.identity['controller_data_id'], 'input_id': self.identity['input_id'],
                        'exposure_hash': fingerprint(self.identity['exposure'])},
            'registered_candidates': [{'pipeline_id': key, 'recipe': self.identity['recipes'][key]} for key in sorted(self.recipes)],
            'S_readouts': [{'pipeline_id': key, 'status': row['status'], 'summary': row['summary']} for key, row in sorted(records.items())],
            'response_schema': {'version': PROPOSAL_VERSION, 'catalogue_id': self.digest, 'candidate_ids': ['registered IDs only']},
            'meaning': 'Choose registered complete learners for bounded S work; no new code, MoE, factor claims or V/C feedback.'}

    def parse_proposal(self, reply):
        self._context(prompt=True)
        if (type(reply) is not dict or set(reply) != {'version', 'catalogue_id', 'candidate_ids'}
                or reply['version'] != PROPOSAL_VERSION or reply['catalogue_id'] != self.digest
                or type(reply['candidate_ids']) is not list or not 1 <= len(reply['candidate_ids']) <= 3
                or any(type(key) is not str or key not in self.recipes for key in reply['candidate_ids'])
                or len(set(reply['candidate_ids'])) != len(reply['candidate_ids'])):
            raise ValueError('GPT may only choose unique registered complete learner IDs in this S catalogue')
        return tuple(FrozenNeuralRecipe.from_dict(self.identity['recipes'][key]) for key in reply['candidate_ids'])

    def seal_shortlist(self):
        self._context()
        records = self._records()
        if any(row['status'] in {'reserved', 'running'} for row in records.values()):
            raise ValueError('Outstanding prepaid algorithm attempts must close before selection')
        complete = [row for row in records.values() if row['status'] == 'completed']
        supports = {tuple(f['sample_ids_hash'] for f in row['summary']['folds']) for row in complete}
        same_support = len(supports) <= 1
        ranked = [row for row in complete if row['summary']['eligible']] if same_support else []
        ranked.sort(key=lambda row: (-row['summary']['rank_score'], row['summary']['losses']['ensemble'], row['pipeline_id']))
        selected = ranked[:self.rule.max_shortlist]
        value = {'version': VERSION, 'catalogue_id': self.digest, 'role': 'S',
            'status': 'sealed' if same_support else 'closed_different_candidate_common_support',
            'algorithm_shortlist': [{'candidate_type': 'complete_neural_algorithm', 'pipeline_id': row['pipeline_id'],
                'recipe': self.identity['recipes'][row['pipeline_id']], 'S_evidence': row['ticket'],
                'S_summary': row['summary'], 'factor_claim_eligible': False, 'serving_eligible': False,
                'next_stage': 'requires_separate_typed_algorithm_validation_contract'} for row in selected],
            'all_candidates': [{'pipeline_id': key, 'status': records[key]['status'] if key in records else 'not_attempted',
                'evidence_hash': fingerprint(records[key]) if key in records else None} for key in sorted(self.recipes)],
            'formal_claim': False, 'serving_eligible': False, 'factor_claim_eligible': False,
            'automatic_adoption': False,
            'prohibited_route': 'Lifecycle.freeze_batch/factor1/factor2/factor3',
            'interpretation': 'Predeclared descriptive S filter only; no independent algorithm claim or V/C permission'}
        with FileLock(str(self.directory) + '.lock', timeout=0):
            _write(self.directory / 'shortlist.json', value)
        return value
