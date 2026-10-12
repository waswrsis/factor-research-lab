"""Bounded development hypothesis memory and explicit local research scheduling.

Evidence resolvers are trusted controller callbacks: they must check the actual
exposure ledger, source coverage and batch. This module rejects non-S provenance,
including any ancestor's provenance, before returning a prompt parent. Research
state never constitutes confirmation or changes a serving model.
"""
from __future__ import annotations

import copy
import json
import math
from itertools import chain

import pandas as pd

from factorlab.protocol import fingerprint, project_path

from .local_mutation import _get, _id, _put
from .v7_storage import immutable_record

ISLANDS = ('refinement', 'exploration')
QUOTAS = {'refine': 12, 'new': 6, 'blind': 6}
MAX_RECORDS = 96
MAX_SHARDS = 120
LIVE_PARENTS_PER_ISLAND = 8
LIVE_RECENT_VERSIONS = 16
AUTOMATIC_REVIVAL = {'version': 'original-S-new-data-v1', 'trigger': 'new_data',
    'minimum_new_months': 1, 'per_island_per_month': 1, 'version_candidate_ceiling': 8,
    'selection': 'latest_retained_dormant_parent', 'adds_work_credit': False}


def _receipt(stage, **facts):
    value = {'stage': stage, 'version': 1, **facts}
    return {**value, 'receipt_id': fingerprint(value)}


def _text(value, maximum=1200):
    if type(value) is not str or not value.strip() or len(value) > maximum:
        raise ValueError('Expected bounded nonempty hypothesis text')
    return value


def _month(value):
    stamp = pd.Timestamp(value)
    if (stamp.tzinfo is None or pd.isna(stamp) or stamp.tz_convert('UTC').day != 1
            or stamp.tz_convert('UTC') != stamp.tz_convert('UTC').normalize()):
        raise ValueError('Hypothesis cycle requires a UTC calendar-month boundary')
    return stamp.tz_convert('UTC').isoformat()


def resolve_sources(sources, resolver):
    """Return controller-verified S receipts; metadata claims alone are insufficient."""
    if (not callable(resolver) or type(sources) not in (list, tuple)
            or not 1 <= len(sources) <= 32 or len(set(sources)) != len(sources)):
        raise ValueError('Evidence needs one to 32 unique source IDs and a trusted resolver')
    result = []
    for source in sources:
        _id(source)
        record = resolver(source)
        if (type(record) is not dict or record.get('evidence_id') != source
                or record.get('role') != 'S'):
            raise ValueError('Every evidence source and ancestor must resolve to S')
        fingerprint(record)
        result.append(copy.deepcopy(record))
    return result


class HypothesisBook:
    """Two logical islands and an immutable-version archive capped at 96 versions.

    A full shard must be persisted before rollover. Old evidence is referenced,
    never overwritten or turned into a result on a new dataset.
    """
    def __init__(self, epoch, *, predecessor=None):
        self.epoch = _month(epoch)
        self.predecessor = copy.deepcopy(predecessor)
        self.versions = {}
        self.parents_by_id = {}

    def _history(self, *, max_shards=MAX_SHARDS):
        if type(max_shards) is not int or not 1 <= max_shards <= MAX_SHARDS:
            raise ValueError('History verification is bounded to 120 shards')
        record, seen, newer_epoch = self.predecessor, set(), self.epoch
        while record is not None:
            path = str(project_path(record['path']))
            if path in seen or len(seen) >= max_shards:
                raise ValueError('Cyclic or oversized hypothesis predecessor history')
            seen.add(path)
            previous = type(self).load(path)
            if (previous.digest != record['digest']
                    or pd.Timestamp(previous.epoch) >= pd.Timestamp(newer_epoch)):
                raise ValueError('Hypothesis predecessor digest/chronology mismatch')
            yield previous
            record, newer_epoch = previous.predecessor, previous.epoch

    def _lookup(self, collection, identifier):
        local = getattr(self, collection)
        if identifier in local:
            return copy.deepcopy(local[identifier])
        for previous in self._history():
            values = getattr(previous, collection)
            if identifier in values:
                return copy.deepcopy(values[identifier])
        raise ValueError('Unresolved hypothesis parent/version ancestry')

    def get_parent(self, identifier, *, resolver=None):
        value = self._lookup('parents_by_id', identifier)
        if resolver is not None:
            self.verify_parents([identifier], resolver=resolver)
        return value

    def get_version(self, identifier):
        return self._lookup('versions', identifier)

    def _mutable_version(self, identifier):
        if identifier not in self.versions:
            if len(self.versions) >= MAX_RECORDS:
                raise ValueError('Hypothesis shard full; persist and roll over')
            record = self.get_version(identifier)
            record['history'] = record['history'][-2:]
            record['previous_failures'] = record['previous_failures'][-8:]
            self.versions[identifier] = record
        return self.versions[identifier]

    def add(self, spec):
        expected = {'hypothesis_id', 'island', 'mechanism', 'competing_explanation',
                    'falsification', 'next_experiment', 'minimum_progress', 'remaining_budget',
                    'stop_conditions'}
        if type(spec) is not dict or set(spec) != expected:
            raise ValueError('Hypothesis requires explicit mechanism, alternatives and stopping contract')
        _id(spec['hypothesis_id'])
        if spec['island'] not in ISLANDS:
            raise ValueError('Exactly two logical research islands are supported')
        for key in ('mechanism', 'competing_explanation', 'falsification', 'next_experiment', 'stop_conditions'):
            _text(spec[key])
        if (type(spec['minimum_progress']) not in (int, float)
                or not math.isfinite(spec['minimum_progress']) or spec['minimum_progress'] <= 0
                or type(spec['remaining_budget']) is not int or not 0 <= spec['remaining_budget'] <= 24):
            raise ValueError('Minimum progress and remaining budget must be declared before research')
        if any(v['spec']['hypothesis_id'] == spec['hypothesis_id']
               for book in chain((self,), self._history()) for v in book.versions.values()):
            raise ValueError('Existing hypotheses require explicit versioned revival')
        return self._insert(spec, parent=None, trigger=None)

    def _insert(self, spec, *, parent, trigger, created_epoch=None):
        if len(self.versions) >= MAX_RECORDS:
            raise ValueError('Hypothesis shard full; persist and roll over')
        record = {'spec': copy.deepcopy(spec), 'created_epoch': created_epoch or self.epoch, 'parent_version': parent,
                  'revival_trigger': copy.deepcopy(trigger), 'state': 'active', 'history': [],
                  'previous_failures': [], 'remaining_budget': spec['remaining_budget']}
        version_id = fingerprint(record)
        if version_id in self.versions:
            raise ValueError('Repeated identical hypothesis version')
        self.versions[version_id] = record
        return version_id

    def observe(self, version_id, *, month, progress, new_mechanism=False,
                evidence_sources, resolver, spent=0, failure=None, next_experiment=None):
        record = self._mutable_version(version_id)
        month = _month(month)
        if (record['state'] != 'active' or type(progress) not in (int, float)
                or not math.isfinite(progress) or type(new_mechanism) is not bool
                or type(spent) is not int or not 0 <= spent <= record['remaining_budget']):
            raise ValueError('Invalid active development observation or budget')
        if pd.Timestamp(month) < pd.Timestamp(record['created_epoch']):
            raise ValueError('Observation predates the hypothesis version')
        if record['history'] and pd.Timestamp(month) <= pd.Timestamp(record['history'][-1]['month']):
            raise ValueError('Development cycles must increase without rewriting history')
        if len(record['history']) >= 96:
            raise ValueError('Observation shard full; persist and roll over before continuing')
        evidence = resolve_sources(evidence_sources, resolver)
        if failure is not None:
            _text(failure)
        if next_experiment is not None:
            _text(next_experiment)
        improved = progress >= record['spec']['minimum_progress'] or new_mechanism
        observation = _receipt('hypothesis_development_cycle', month=month, progress=progress,
                               new_mechanism=new_mechanism, improved=improved,
                               evidence=evidence, spent=spent, failure=failure,
                               next_experiment=next_experiment or record['spec']['next_experiment'])
        record['history'].append(observation)
        record['remaining_budget'] -= spent
        if failure:
            record['previous_failures'].append({'month': month, 'failure': failure})
        # Consecutive means adjacent PREDECLARED observed monthly cycles. A gap
        # cannot masquerade as two consecutive checks.
        last = record['history'][-2:]
        stalled = (len(last) == 2 and not any(r['improved'] for r in last)
                   and pd.Timestamp(last[0]['month']) + pd.DateOffset(months=1) == pd.Timestamp(last[1]['month']))
        if stalled or record['remaining_budget'] == 0:
            record['state'] = 'dormant'
        return _receipt('hypothesis_transition', version_id=version_id, state=record['state'],
                        reason='two_stalled_cycles' if stalled else 'budget_exhausted' if record['remaining_budget'] == 0 else 'continue',
                        observation=observation, affects_A0=False)

    def revive(self, version_id, *, trigger, resolver, remaining_budget):
        old = self.get_version(version_id)
        if (old['state'] != 'dormant' or type(trigger) is not dict
                or set(trigger) != {'kind', 'evidence_id', 'description'}
                or trigger['kind'] not in {'new_data', 'new_mechanism', 'new_input'}
                or type(remaining_budget) is not int or not 1 <= remaining_budget <= 24):
            raise ValueError('Dormant hypotheses need an explicit qualifying revival trigger')
        _text(trigger['description'])
        evidence = resolve_sources([trigger['evidence_id']], resolver)
        if any(v['parent_version'] == version_id
               for book in chain((self,), self._history()) for v in book.versions.values()):
            raise ValueError('This dormant version already has a successor')
        spec = {**old['spec'], 'remaining_budget': remaining_budget}
        created = max([self.epoch, *[r['month'] for r in old['history']]], key=pd.Timestamp)
        new_id = self._insert(spec, parent=version_id, trigger={**trigger, 'verified_evidence': evidence},
                              created_epoch=created)
        self.versions[new_id]['previous_failures'] = copy.deepcopy(old['previous_failures'])
        return new_id

    def register_parent(self, record, *, _restore=False):
        expected = {'parent_id', 'island', 'program', 'ancestor_ids', 'evidence_sources', 'kind', 'citation',
                    'hypothesis_version'}
        if type(record) is not dict or set(record) != expected:
            raise ValueError('Parent requires complete ancestry and evidence source IDs')
        _id(record['parent_id'])
        if (record['island'] not in ISLANDS or record['kind'] not in {'research', 'literature_seed'}
                or type(record['ancestor_ids']) is not list or len(record['ancestor_ids']) > 8
                or len(set(record['ancestor_ids'])) != len(record['ancestor_ids'])
                or record['parent_id'] in record['ancestor_ids']):
            raise ValueError('Invalid parent island or unresolved ancestry')
        for ancestor in record['ancestor_ids']:
            self.get_parent(ancestor)
        if record['parent_id'] in self.parents_by_id or len(self.parents_by_id) >= MAX_RECORDS:
            raise ValueError('Parent identities are immutable and bounded')
        for previous in self._history():
            existing = previous.parents_by_id.get(record['parent_id'])
            if existing is not None:
                if not _restore or existing != record:
                    raise ValueError('Parent identities are immutable across shards')
                break
        if record['kind'] == 'literature_seed':
            _text(record['citation'], 500)
            if record['evidence_sources'] or record['ancestor_ids'] or record['hypothesis_version'] is not None:
                raise ValueError('Literature seeds cannot disguise measured evidence or ancestry')
        elif not record['evidence_sources']:
            raise ValueError('Research parents need verifiable S evidence')
        elif self.get_version(record['hypothesis_version'])['spec']['island'] != record['island']:
            raise ValueError('Research parents must bind a hypothesis version on the same island')
        _canonical(record['program'])
        self.parents_by_id[record['parent_id']] = copy.deepcopy(record)

    def parents(self, island, *, resolver, count=4):
        if island not in ISLANDS or type(count) is not int or not 2 <= count <= 4:
            raise ValueError('Each of two islands supplies two to four parents')
        eligible = [p for p in self.parents_by_id.values() if p['island'] == island
                    and (p['kind'] == 'literature_seed'
                         or self.get_version(p['hypothesis_version'])['state'] == 'active')]
        if len(eligible) < 2:
            raise ValueError('Provide explicit literature seeds when fewer than two parents exist')
        # Stable insertion order is a declared scheduling rule, not a score sort.
        selected = eligible[-count:]
        checked = self.verify_parents([p['parent_id'] for p in selected], resolver=resolver)
        return {'parents': copy.deepcopy(selected), 'ancestry_evidence': list(checked.values()),
                'receipt': _receipt('resolved_island_parents', island=island,
                                    parent_ids=[p['parent_id'] for p in selected],
                                    ancestry_hash=fingerprint(checked))}

    def verify_parents(self, identifiers, *, resolver):
        checked, visiting = {}, set()
        pending = [(identifier, False) for identifier in identifiers]
        while pending:
            identifier, exiting = pending.pop()
            if exiting:
                visiting.remove(identifier)
                continue
            if identifier in visiting:
                raise ValueError('Cyclic parent ancestry')
            if identifier in checked:
                continue
            if len(checked) >= MAX_RECORDS * (MAX_SHARDS + 1):
                raise ValueError('Oversized parent ancestry')
            parent = self.get_parent(identifier)
            evidence = resolve_sources(parent['evidence_sources'], resolver) if parent['kind'] == 'research' else []
            if parent['kind'] == 'research':
                version = self.get_version(parent['hypothesis_version'])
                trigger = version.get('revival_trigger')
                if trigger is not None:
                    actual = resolve_sources([trigger['evidence_id']], resolver)
                    if actual != trigger['verified_evidence']:
                        raise ValueError('Original revival trigger evidence changed')
            checked[identifier] = {'parent_id': identifier, 'evidence': evidence,
                                   'kind': parent['kind'], 'citation': parent['citation']}
            visiting.add(identifier)
            pending.append((identifier, True))
            pending.extend((ancestor, False) for ancestor in parent['ancestor_ids'])
        return checked

    def to_dict(self):
        return {'version': 1, 'epoch': self.epoch, 'predecessor': copy.deepcopy(self.predecessor),
                'versions': copy.deepcopy(self.versions), 'parents': copy.deepcopy(self.parents_by_id)}

    @property
    def digest(self):
        return fingerprint(self.to_dict())

    def save(self, path):
        return immutable_record(path, {'book': self.to_dict(), 'digest': self.digest})

    @classmethod
    def load(cls, path):
        wrapper = json.loads(project_path(path).read_text('utf-8'))
        value = wrapper['book']
        if (wrapper.get('digest') != fingerprint(value) or value.get('version') != 1
                or len(value['versions']) > MAX_RECORDS or len(value['parents']) > MAX_RECORDS):
            raise ValueError('Hypothesis archive integrity/version/bounds failure')
        result = cls(value['epoch'], predecessor=value['predecessor'])
        result.versions, result.parents_by_id = value['versions'], value['parents']
        return result

    def rollover(self, epoch, saved_path):
        if self.load(saved_path).digest != self.digest or pd.Timestamp(_month(epoch)) <= pd.Timestamp(self.epoch):
            raise ValueError('Persist the exact old book before forward rollover')
        if self.verify_history()['shard_count'] >= MAX_SHARDS:
            raise ValueError('Hypothesis predecessor limit of 120 shards reached')
        result = type(self)(epoch, predecessor={'path': str(project_path(saved_path)), 'digest': self.digest})
        keep = set()
        for island in ISLANDS:
            research = [k for k, p in self.parents_by_id.items()
                        if p['island'] == island and p['kind'] == 'research']
            keep.update(research[-LIVE_PARENTS_PER_ISLAND:])
            seeds = [k for k, p in self.parents_by_id.items()
                     if p['island'] == island and p['kind'] == 'literature_seed']
            keep.update(seeds[-4:])
        result.parents_by_id = {k: copy.deepcopy(p) for k, p in self.parents_by_id.items() if k in keep}
        needed = {p['hypothesis_version'] for p in result.parents_by_id.values() if p['kind'] == 'research'}
        ordered = sorted(self.versions, key=lambda k: (self.versions[k]['created_epoch'], k))
        needed.update(ordered[-LIVE_RECENT_VERSIONS:])
        result.versions = {k: self.get_version(k) for k in sorted(needed)}
        for record in result.versions.values():
            record['history'] = record['history'][-2:]
            record['previous_failures'] = record['previous_failures'][-8:]
        return result

    def verify_history(self, *, max_shards=120):
        """Explicitly validate the entire predecessor chain; load checks one shard."""
        digests = [previous.digest for previous in self._history(max_shards=max_shards)]
        return _receipt('verified_hypothesis_history', shard_count=len(digests), digests=digests)


def schedule_candidates(candidates, budget, *, used_by_lane=None):
    """Deterministic 12/6/6 ceilings, never extra families or forced quota fill.

    Input order must be frozen before label scoring. This function reserves no
    F work; the controller reserves durably just before each actual evaluation.
    Repeated scheduling calls must supply controller-persisted used_by_lane.
    """
    if type(candidates) not in (list, tuple) or len(candidates) > 96:
        raise ValueError('At most 96 materialized candidates may be scheduled')
    ledger = budget.to_dict()
    left = min(24, ledger['limits']['f'] - ledger['counts']['f'])
    family_left = {f: ledger['limits']['family_f'] - v['f'] for f, v in ledger['families'].items()}
    used_by_lane = dict.fromkeys(QUOTAS, 0) if used_by_lane is None else dict(used_by_lane)
    if set(used_by_lane) != set(QUOTAS) or any(type(v) is not int or not 0 <= v <= QUOTAS[k]
                                             for k, v in used_by_lane.items()):
        raise ValueError('Invalid persisted scheduling-lane budget')
    lane_left = {k: QUOTAS[k] - used_by_lane[k] for k in QUOTAS}
    seen, chosen, deferred = set(), [], []
    for candidate in candidates:
        if type(candidate) is not dict or set(candidate) != {'candidate_id', 'family_id', 'lane'}:
            raise ValueError('Scheduler requires candidate, registered family and declared lane')
        identifier, family, lane = candidate['candidate_id'], candidate['family_id'], candidate['lane']
        _id(identifier)
        if identifier in seen or family not in family_left or lane not in QUOTAS:
            raise ValueError('Duplicate candidate, unregistered family or unknown scheduling lane')
        seen.add(identifier)
        if left and family_left[family] and lane_left[lane]:
            chosen.append(copy.deepcopy(candidate))
            left -= 1
            family_left[family] -= 1
            lane_left[lane] -= 1
        else:
            deferred.append(copy.deepcopy(candidate))
    return {'scheduled': chosen, 'deferred': deferred,
            'receipt': _receipt('bounded_research_schedule', quotas=QUOTAS,
                                actual={k: QUOTAS[k] - used_by_lane[k] - lane_left[k] for k in QUOTAS},
                                cumulative_reserved={k: QUOTAS[k] - lane_left[k] for k in QUOTAS},
                                budget_before=fingerprint(ledger), reserves_f=False,
                                order='predeclared_input_order', unused_slots=24-len(chosen))}


def _canonical(value):
    from .program_contract import Program
    program = Program.from_dict(value)
    if len(program.outputs) != 1:
        raise ValueError('V7 mutation requires exactly one scalar output')
    return program


def _kind(node):
    op = node['op']
    if op == 'feature':
        return 'series'
    if op in {'const', 'param', 'last', 'mean', 'std', 'min', 'max'}:
        return 'scalar'
    if op in {'add', 'sub', 'mul', 'safe_div'}:
        return 'series' if 'series' in (_kind(node['left']), _kind(node['right'])) else 'scalar'
    return _kind(node['arg'])


def _used_parameters(node):
    result = set()
    if isinstance(node, dict):
        if node.get('op') == 'param':
            result.add(node['name'])
        for child in node.values():
            result |= _used_parameters(child)
    elif isinstance(node, list):
        for child in node:
            result |= _used_parameters(child)
    return result


def materialize_mutations(parent_program, attempts, *, family_id, budget, seen=None):
    """Finite exact replacement/crossover. Reserve AST BEFORE each attempted creation.

    No float-to-int window coercion, parameter clipping, domain widening or
    repair. Invalid/duplicate outputs stay in the returned cost receipts.
    """
    if type(attempts) not in (list, tuple) or not 1 <= len(attempts) <= 32:
        raise ValueError('Local mutation requires one to 32 explicit attempts')
    seen = set() if seen is None else seen
    records = []
    for index, attempt in enumerate(attempts):
        budget.consume('ast', family_id=family_id)
        try:
            base = _canonical(parent_program).to_dict()
            path = attempt['path']
            if type(path) is not list or not 2 <= len(path) <= 16 or path[0] not in {'outputs', 'parameters'}:
                raise ValueError('Mutation path must name an existing output or parameter initial')
            original = _get(base, path)
            if attempt['kind'] == 'replace':
                if set(attempt) != {'kind', 'path', 'value'}:
                    raise ValueError('Unknown replacement fields')
                replacement = copy.deepcopy(attempt['value'])
                if (type(original) is not type(replacement)
                        or (path[0] == 'parameters' and (len(path) != 3 or path[-1] != 'initial'))
                        or path[-1] == 'op'):
                    raise ValueError('Replacement must preserve exact field types and parameter domains')
                if isinstance(original, dict) and _kind(original) != _kind(replacement):
                    raise ValueError('Replacement changes scalar/series type')
            elif attempt['kind'] == 'crossover':
                if set(attempt) != {'kind', 'path', 'donor', 'donor_path'} or path[0] != 'outputs':
                    raise ValueError('Crossover requires two declared output subtrees')
                donor = _canonical(attempt['donor']).to_dict()
                donor_path = attempt['donor_path']
                if type(donor_path) is not list or len(donor_path) < 2 or donor_path[0] != 'outputs':
                    raise ValueError('Donor path must refer to an output subtree')
                replacement = copy.deepcopy(_get(donor, donor_path))
                if (donor['feature_names'] != base['feature_names'] or not isinstance(original, dict)
                        or not isinstance(replacement, dict) or _kind(original) != _kind(replacement)):
                    raise ValueError('Crossover requires matching input contract and scalar/series type')
                for name in _used_parameters(replacement):
                    spec = donor['parameters'][name]
                    if name in base['parameters'] and base['parameters'][name] != spec:
                        raise ValueError('Crossover cannot reinterpret an existing parameter domain')
                    base['parameters'][name] = copy.deepcopy(spec)
            else:
                raise ValueError('Only declared replacement and crossover are supported')
            _put(base, path, replacement)
            # Removing declarations rendered unused by a subtree replacement is
            # explicit bookkeeping; no surviving parameter domain is altered.
            used = _used_parameters(base['outputs'])
            removed = sorted(set(base['parameters']) - used)
            base['parameters'] = {k: v for k, v in base['parameters'].items() if k in used}
            program = _canonical(base)
            duplicate = program.digest in seen
            seen.add(program.digest)
            records.append(_receipt('local_AST_materialization', index=index, family_id=family_id,
                                    status='duplicate' if duplicate else 'valid', digest=program.digest,
                                    program=program.to_dict(), attempt=copy.deepcopy(attempt),
                                    removed_unused_declarations=removed))
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            records.append(_receipt('local_AST_materialization', index=index, family_id=family_id,
                                    status='invalid', digest=None, program=None,
                                    error_type=type(exc).__name__, reason=str(exc)))
    return records
