"""Controller-owned S-only memory. Archived readouts are priors, never new evidence.

The resolver reads a real RunStore checkpoint/result and the exposure ledger on
EVERY prompt construction, including every ancestor. This is bounded storage,
not a proof that unregistered human/provider observations never happened.
"""
from __future__ import annotations

import copy
import json

import pandas as pd

from factorlab.protocol import fingerprint, project_path

from .factor_pool import FactorPool
from .hypotheses import (
    AUTOMATIC_REVIVAL,
    ISLANDS,
    MAX_RECORDS,
    MAX_SHARDS,
    QUOTAS,
    HypothesisBook,
    _month,
    _text,
    materialize_mutations,
    schedule_candidates,
)
from .local_mutation import _id
from .rolling import development_folds, RollingSpec
from .v7_storage import immutable_record


def search_exposure_context(data_id, cutoff, symbols, feed_identity=None, *, spec=None):
    """Frozen dataset binding, with one persistent source for a live receipt feed."""
    spec = spec or RollingSpec()
    _,cutoff = spec.window(cutoff)
    start = ((cutoff - pd.Timedelta(days=180)) if spec.window_days is not None else
             cutoff - pd.DateOffset(months=6)) - pd.Timedelta(hours=1234)
    return {'source': feed_identity or fingerprint({'data_id': data_id, 'kind': 'v7_exposed_archive'}),
            'batch': cutoff.strftime('%Y-%m'), 'symbols': list(symbols),
            'start': start.isoformat(),
            'end': cutoff.isoformat(), 'data_id': data_id}


class ResearchMemory:
    def __init__(self, book, *, evidence=None, proposals=None, lanes=None, scheduled=None, predecessor=None):
        self.book = book
        self.predecessor = copy.deepcopy(predecessor)
        self.evidence = copy.deepcopy(evidence or {})
        self.proposals = copy.deepcopy(proposals or {})
        self.lanes = dict.fromkeys(QUOTAS, 0) if lanes is None else copy.deepcopy(lanes)
        self.scheduled = list(scheduled or [])
        if (len(self.evidence) > MAX_RECORDS or len(self.proposals) > MAX_RECORDS
                or len(self.scheduled) > 24 or len(set(self.scheduled)) != len(self.scheduled)
                or set(self.lanes) != set(QUOTAS)
                or any(type(v) is not int or not 0 <= v <= QUOTAS[k] for k, v in self.lanes.items())
                or sum(self.lanes.values()) != len(self.scheduled)):
            raise ValueError('Research memory capacity/lane integrity failure')
        if any(not isinstance(p, dict) or not {'version', 'parent_ids', 'hypothesis', 'proposal_hash'} <= set(p)
               for p in self.proposals.values()):
            raise ValueError('Research memory capacity exhausted by invalid proposal records')

    def _history(self):
        link, seen, newer_epoch = self.predecessor, set(), self.book.epoch
        expected_book = self.book.predecessor
        while link is not None:
            path = str(project_path(link['path']))
            if path in seen or len(seen) >= MAX_SHARDS:
                raise ValueError('Cyclic or oversized research memory predecessor history')
            seen.add(path)
            wrapper = json.loads(project_path(path).read_text('utf-8'))
            value = wrapper['memory']
            if (wrapper.get('digest') != fingerprint(value) or wrapper['digest'] != link['digest']
                    or value.get('version') != 1
                    or pd.Timestamp(_month(value['book']['epoch'])) >= pd.Timestamp(newer_epoch)):
                raise ValueError('Research memory predecessor digest/chronology mismatch')
            if not expected_book or fingerprint(value['book']) != expected_book['digest']:
                raise ValueError('Book and memory predecessor identities differ')
            yield value
            link, newer_epoch = value.get('predecessor'), value['book']['epoch']
            expected_book = value['book']['predecessor']

    def verify_history(self):
        self.book.verify_history()
        digests = [fingerprint(previous) for previous in self._history()]
        return {'shard_count': len(digests), 'digests': digests}

    def _get(self, name, identifier):
        local = getattr(self, name)
        if identifier in local:
            return copy.deepcopy(local[identifier])
        for previous in self._history():
            if identifier in previous[name]:
                return copy.deepcopy(previous[name][identifier])
        raise ValueError('Unresolved archived research evidence/proposal')

    def get_evidence(self, identifier):
        value = self._get('evidence', identifier)
        if fingerprint(value) != identifier:
            raise ValueError('Evidence reference identity mismatch')
        return value

    def get_proposal(self, identifier):
        return self._get('proposals', identifier)

    def get_parent(self, identifier, *, store, exposure):
        self.verify_history()
        return self.book.get_parent(identifier, resolver=self.resolver(store, exposure))

    @classmethod
    def initial(cls, epoch, seeds):
        """Two to four explicit cited program seeds for EACH island."""
        if type(seeds) is not list or not 2 <= len(seeds) <= 4:
            raise ValueError('Two to four cited seeds are required')
        book = HypothesisBook(epoch)
        for island in ISLANDS:
            for index, seed in enumerate(seeds):
                if type(seed) is not dict or set(seed) != {'program', 'citation'}:
                    raise ValueError('Seeds require program and explicit citation')
                book.register_parent({'parent_id': f'{island}:seed:{index}', 'island': island,
                    'program': seed['program'], 'ancestor_ids': [], 'evidence_sources': [],
                    'kind': 'literature_seed', 'citation': seed['citation'], 'hypothesis_version': None})
        return cls(book)

    def to_state(self):
        value = {'version': 1, 'book': self.book.to_dict(), 'evidence': copy.deepcopy(self.evidence),
                 'proposals': copy.deepcopy(self.proposals), 'lanes': dict(self.lanes),
                 'scheduled': list(self.scheduled), 'parent_order': list(self.book.parents_by_id),
                 'predecessor': copy.deepcopy(self.predecessor)}
        return {'memory': value, 'digest': fingerprint(value)}

    @classmethod
    def from_state(cls, wrapper):
        value = wrapper['memory']
        if wrapper.get('digest') != fingerprint(value) or value.get('version') != 1:
            raise ValueError('Research memory state integrity failure')
        raw = value['book']
        if raw.get('version') != 1 or len(raw['versions']) > MAX_RECORDS or len(raw['parents']) > MAX_RECORDS:
            raise ValueError('Hypothesis book capacity/version failure')
        book = HypothesisBook(raw['epoch'], predecessor=raw['predecessor'])
        book.versions = copy.deepcopy(raw['versions'])
        # Reapply parent schema/ancestry validation in original insertion order.
        order = value['parent_order']
        if len(order) != len(set(order)) or set(order) != set(raw['parents']):
            raise ValueError('Parent scheduling order integrity failure')
        pending = {identifier: copy.deepcopy(raw['parents'][identifier]) for identifier in order}
        while pending:
            ready = []
            for key, record in pending.items():
                if any(p in pending for p in record['ancestor_ids']):
                    continue
                for ancestor in record['ancestor_ids']:
                    book.get_parent(ancestor)  # A hashbound predecessor may own it.
                ready.append(key)
            if not ready:
                raise ValueError('Unresolved or cyclic memory ancestry')
            for identifier in ready:
                record = pending.pop(identifier)
                if record['parent_id'] != identifier:
                    raise ValueError('Parent index mismatch')
                book.register_parent(record, _restore=True)
        book.parents_by_id = {identifier: book.parents_by_id[identifier] for identifier in order}
        result = cls(book, evidence=value['evidence'], proposals=value['proposals'],
                     lanes=value['lanes'], scheduled=value['scheduled'], predecessor=value.get('predecessor'))
        for identifier, ref in result.evidence.items():
            if identifier != fingerprint(ref):
                raise ValueError('Evidence reference identity mismatch')
        result.verify_history()
        return result

    def save(self, path):
        return immutable_record(path, self.to_state())

    def rollover(self, epoch, saved_path):
        """Persist old evidence before resetting ONLY monthly scheduling counters.

        Only recent parent/version/evidence indices remain live. Complete earlier
        records stay in immutable, hashbound predecessors and resolve lazily.
        """
        saved_wrapper = json.loads(project_path(saved_path).read_text('utf-8'))
        saved = type(self).from_state(saved_wrapper)
        if saved.to_state() != self.to_state() or pd.Timestamp(_month(epoch)) <= pd.Timestamp(self.book.epoch):
            raise ValueError('Persist exact memory before forward rollover')
        book_path = project_path(saved_path).with_suffix('.book.json')
        self.book.save(book_path)
        book = self.book.rollover(epoch, book_path)
        needed_evidence = {source for parent in book.parents_by_id.values() for source in parent['evidence_sources']}
        evidence = {key: self.get_evidence(key) for key in sorted(needed_evidence)}
        proposals = {key: copy.deepcopy(value) for key, value in self.proposals.items()
                     if value['version'] in book.versions}
        return type(self)(book, evidence=evidence, proposals=proposals,
            predecessor={'path': str(project_path(saved_path)), 'digest': saved_wrapper['digest']})

    def _read_result(self, store, exposure, month, formula_id):
        from .v7_run import _verify_month_artifacts
        if project_path(exposure.path) != project_path(store.identity['exposure_path']):
            raise ValueError('Research memory requires the frozen shared exposure ledger')
        store.verify()
        state = store.get(month)
        if state is None or (pd.Timestamp(state['cutoff']) > pd.Timestamp(self.book.epoch) if store.spec.window_days is None
                else pd.Timestamp(state['cutoff']) >= pd.Timestamp(self.book.epoch)+pd.DateOffset(months=1)):
            raise ValueError('Missing or future research checkpoint')
        if pd.Timestamp(state['cutoff']).strftime('%Y-%m') != month:
            raise ValueError('Research month/cutoff mismatch')
        _verify_month_artifacts(store.directory, month, state)
        pool = FactorPool.from_dict(state['pool'])
        if formula_id not in pool.records:
            raise ValueError('No measured candidate in the actual monthly pool')
        context = search_exposure_context(state['data_id'], state['cutoff'], store.identity['config']['symbols'],
                                          store.identity.get('feed_identity'), spec=store.spec)
        if state.get('search_exposure') != context:
            raise ValueError('Missing exact frozen search exposure/data identity')
        columns = ('source', 'symbol', 'start', 'end', 'role', 'batch', 'detail', 'observed_at', 'recorded_at')
        for row in exposure.db.execute('SELECT * FROM evidence WHERE source=? AND batch=?',
                                       (context['source'], context['batch'])):
            evidence = dict(zip(columns, row[1:], strict=True))
            evidence['detail'] = json.loads(evidence['detail'])
            if fingerprint(evidence) != row[0]:
                raise ValueError('Exposure ledger content/index integrity failure')
        exposure.assert_prompt_search(**{k: v for k, v in context.items() if k != 'data_id'})
        record = pool.records[formula_id]
        result = record['result']
        if result.get('role') != 'S' or result.get('formula_id') != formula_id:
            raise ValueError('Only actual S readouts can enter research memory')
        actual = [(f['fit_cutoff'], f['score_end']) for f in result['folds']]
        expected = [(f['fit_cutoff'], f['score_end']) for f in development_folds(state['cutoff'], spec=store.spec)]
        if actual != expected:
            raise ValueError('S readout does not cover the frozen three development folds')
        ref = {'month': month, 'formula_id': formula_id, 'data_id': state['data_id'],
               'result_hash': fingerprint(result), 'run_identity': fingerprint(store.identity),
               'search_exposure': context}
        return ref, record

    def bind_result(self, store, exposure, month, formula_id):
        ref, _ = self._read_result(store, exposure, month, formula_id)
        identifier = fingerprint(ref)
        if identifier not in self.evidence and len(self.evidence) >= MAX_RECORDS:
            raise ValueError('Evidence reference capacity exhausted')
        self.evidence[identifier] = ref
        return identifier

    def resolver(self, store, exposure):
        def resolve(identifier):
            ref = self.get_evidence(identifier)
            if ref.get('kind') == 'original_S_data_context':
                actual = self._data_context(store, exposure, ref['month'])
                if actual != ref or fingerprint(actual) != identifier:
                    raise ValueError('Original S revival context changed')
                return {'evidence_id': identifier, 'role': 'S', **actual,
                        'scope': 'data_trigger_only_not_a_score', 'current_gain': None}
            actual, record = self._read_result(store, exposure, ref['month'], ref['formula_id'])
            if actual != ref or fingerprint(actual) != identifier:
                raise ValueError('Archived research evidence changed')
            return {'evidence_id': identifier, 'role': 'S', **actual,
                    'prior_gain_F': record['result']['channels']['F']['gain'],
                    'prior_gain_S': record['result']['channels']['S']['gain'],
                    'scope': 'prior_context_only', 'current_gain': None}
        return resolve

    def _data_context(self, store, exposure, month):
        """Reconstruct sealed paid S input and its actual original source.

        This is a data-arrival witness, never a numerical result or caller's
        new_data flag. Historical revalidation retains the original prefix.
        """
        from .v7_run import RunStore, _verify_month_artifacts
        from .work_authority_bridge import _read
        from .neural_forward_ticket import open_existing_authority
        from .s_preparation import SPreparationEnvelope
        from .s_observation_source import validate_reference
        if type(store) is not RunStore or project_path(exposure.path) != project_path(store.identity['exposure_path']):
            raise TypeError('Original RunStore and shared S exposure required')
        store.verify()
        state = store.get(month)
        if state is None or (pd.Timestamp(state['cutoff']) > pd.Timestamp(self.book.epoch) if store.spec.window_days is None
                else pd.Timestamp(state['cutoff']) >= pd.Timestamp(self.book.epoch)+pd.DateOffset(months=1)):
            raise ValueError('Missing or future original S data context')
        _verify_month_artifacts(store.directory, month, state)
        context = search_exposure_context(state['data_id'], state['cutoff'],
            store.identity['config']['symbols'], store.identity.get('feed_identity'), spec=store.spec)
        if state.get('search_exposure') != context or context['batch'] != month:
            raise ValueError('Revival requires exact original S data/exposure context')
        # Check row content as well as interval coverage; changing a role cannot
        # turn a V/C row into an apparently valid S witness.
        columns = ('source', 'symbol', 'start', 'end', 'role', 'batch', 'detail', 'observed_at', 'recorded_at')
        for row in exposure.db.execute('SELECT * FROM evidence WHERE source=? AND batch=?',
                                      (context['source'], context['batch'])):
            evidence = dict(zip(columns, row[1:], strict=True))
            evidence['detail'] = json.loads(evidence['detail'])
            if fingerprint(evidence) != row[0]:
                raise ValueError('Revival exposure ledger integrity failure')
        exposure.assert_prompt_search(**{k: v for k, v in context.items() if k != 'data_id'})
        bridge = _read(store.directory/'work_authority'/month/'identity.json')
        authority = open_existing_authority(bridge['authority_directory'])
        if (bridge != _read(authority.directory/'search_bridges'/bridge['bridge_id']/'identity.json')
                or state.get('authority_bridge', {}).get('identity_hash') != fingerprint(bridge)
                or bridge['run_identity_hash'] != fingerprint(store.identity)
                or bridge['data_id'] != state['data_id'] or bridge['search_exposure'] != context):
            raise ValueError('Revival lost its original paid S bridge')
        prep = SPreparationEnvelope.load(authority, store.directory, month)
        prep.verified_bridge_envelope(store, month, state)
        baseline = prep.baseline_record()
        reference = prep.identity['source_reference']
        if (bridge.get('preparation_hash') != prep.digest
                or baseline['status'] not in {'running', 'completed'}
                or reference.get('feed_identity') != context['source']
                or reference.get('cutoff') != state['cutoff']
                or project_path(reference['exposure_path']) != project_path(exposure.path)):
            raise ValueError('Revival requires original live S preparation/source')
        validate_reference(reference)
        return {'kind': 'original_S_data_context', 'month': month, 'data_id': state['data_id'],
                'run_identity': fingerprint(store.identity), 'search_exposure': context,
                'preparation_hash': prep.digest, 'bridge_identity_hash': fingerprint(bridge),
                'source_reference': copy.deepcopy(reference), 'revival_policy': fingerprint(AUTOMATIC_REVIVAL)}

    def automatic_revive(self, store, exposure, *, island, persist):
        """At most one new-data successor per island/month, inside paid S work.

        New versions supply prompt parents only. Existing family/AST/F/solve
        reservations still govern every actual proposal and evaluation.
        """
        if island not in ISLANDS or not callable(persist):
            raise ValueError('Automatic revival needs an island and durable checkpoint')
        self.verify_history()
        policy = AUTOMATIC_REVIVAL['version']
        versions = list(self.book.versions.values())
        if any(v['created_epoch'] == self.book.epoch and v['spec']['island'] == island
               and (v.get('revival_trigger') or {}).get('description') == policy for v in versions):
            return {'status': 'already_considered', 'adds_work_credit': False}
        # Do not revive arbitrary archived programs or exceed the retained
        # parent/version capacities. Unselected archives remain dormant.
        candidates = [p for p in self.book.parents_by_id.values()
                      if p['kind'] == 'research' and p['island'] == island
                      and self.book.get_version(p['hypothesis_version'])['state'] == 'dormant']
        if not candidates or len(self.book.versions) >= MAX_RECORDS or len(self.book.parents_by_id) >= MAX_RECORDS:
            return {'status': 'no_eligible_retained_parent', 'adds_work_credit': False}
        month = pd.Timestamp(self.book.epoch).strftime('%Y-%m')
        if (store.get(month) or {}).get('status') != 'running':
            raise ValueError('Automatic revival requires the active original S month')
        current = self._data_context(store, exposure, month)
        resolver = self.resolver(store, exposure)
        for parent in reversed(candidates):
            old_id = parent['hypothesis_version']
            old = self.book.get_version(old_id)
            if (not old['history'] or old['spec']['remaining_budget'] <= 0
                    or any(v['parent_version'] == old_id
                    for book in (self.book, *self.book._history()) for v in book.versions.values())):
                continue
            self.book.verify_parents([parent['parent_id']], resolver=resolver)
            last = old['history'][-1]
            actual = [resolver(e['evidence_id']) for e in last['evidence']]
            if actual != last['evidence']:
                raise ValueError('Dormant original observation evidence changed')
            if (pd.Timestamp(self.book.epoch) < pd.Timestamp(last['month']) + pd.DateOffset(months=1)
                    or any(e['month'] != pd.Timestamp(last['month']).strftime('%Y-%m') for e in actual)
                    or any(e['data_id'] == current['data_id'] for e in actual)):
                continue
            # No old gain is attached to the new dataset. The original scores
            # remain accessible only through the old ancestry evidence.
            evidence_id = fingerprint(current)
            if evidence_id not in self.evidence and len(self.evidence) >= MAX_RECORDS:
                return {'status': 'evidence_capacity', 'adds_work_credit': False}
            self.evidence[evidence_id] = current
            new_id = self.book.revive(old_id,
                trigger={'kind': 'new_data', 'evidence_id': evidence_id, 'description': policy},
                resolver=resolver, remaining_budget=min(old['spec']['remaining_budget'],
                    AUTOMATIC_REVIVAL['version_candidate_ceiling']))
            parent_id = fingerprint({'revival_version': new_id, 'original_parent': parent['parent_id']})
            self.book.register_parent({**copy.deepcopy(parent), 'parent_id': parent_id,
                'ancestor_ids': [parent['parent_id']], 'hypothesis_version': new_id})
            persist(self.to_state())
            return {'status': 'revived', 'version': new_id, 'parent_id': parent_id,
                    'original_version': old_id, 'evidence_id': evidence_id,
                    'current_gain': None, 'adds_work_credit': False}
        return {'status': 'new_data_trigger_not_met', 'adds_work_credit': False}

    def register_proposal(self, proposal, *, island, next_experiment, minimum_progress,
                          remaining_budget, stop_conditions):
        if island not in ISLANDS or type(proposal) is not dict:
            raise ValueError('Invalid proposal island/schema')
        parents = proposal['parent_ids']
        if type(parents) is not list or not 1 <= len(parents) <= 4 or len(set(parents)) != len(parents):
            raise ValueError('Proposal requires one to four existing same-island parents')
        try:
            records = [self.book.get_parent(p) for p in parents]
        except ValueError as exc:
            raise ValueError('Proposal requires existing same-island parents') from exc
        if any(p['island'] != island for p in records):
            raise ValueError('Proposal requires existing same-island parents')
        if any(p['kind'] == 'research' and self.book.get_version(p['hypothesis_version'])['state'] != 'active'
               for p in records):
            raise ValueError('Dormant hypotheses require explicit revival before selection')
        _text(proposal['hypothesis'])
        identifier = proposal['family_id']
        _id(identifier)
        if (identifier in self.proposals or any(identifier in value['proposals'] for value in self._history())
                or len(self.proposals) >= MAX_RECORDS):
            raise ValueError('Proposal family immutable/capacity exhausted')
        version = self.book.add({'hypothesis_id': identifier, 'island': island,
            'mechanism': proposal['mechanism'], 'competing_explanation': proposal['competing_explanation'],
            'falsification': proposal['falsification'], 'next_experiment': next_experiment,
            'minimum_progress': minimum_progress, 'remaining_budget': remaining_budget,
            'stop_conditions': stop_conditions})
        self.proposals[identifier] = {'version': version, 'parent_ids': list(parents),
            'hypothesis': proposal['hypothesis'], 'proposal_hash': fingerprint(proposal)}
        return version

    def register_measured_parent(self, *, family_id, evidence_id, store, exposure):
        proposal = self.get_proposal(family_id)
        self.resolver(store, exposure)(evidence_id)
        ref = self.get_evidence(evidence_id)
        _, record = self._read_result(store, exposure, ref['month'], ref['formula_id'])
        if record['family'] != family_id:
            raise ValueError('Measured formula belongs to another proposal family')
        version = proposal['version']
        island = self.book.get_version(version)['spec']['island']
        identifier = fingerprint({'evidence': evidence_id, 'version': version})
        if identifier in self.book.parents_by_id:
            return identifier
        for previous in self.book._history():
            if identifier in previous.parents_by_id:
                return identifier
        self.book.register_parent({'parent_id': identifier, 'island': island, 'program': record['program'],
            'ancestor_ids': proposal['parent_ids'], 'evidence_sources': [evidence_id],
            'kind': 'research', 'citation': None, 'hypothesis_version': version})
        return identifier

    def prompt_parents(self, store, exposure, *, island, current_context, count=4):
        self.verify_history()
        result = self.book.parents(island, resolver=self.resolver(store, exposure), count=count)
        for parent in result['parents']:
            parent['evidence_role'] = 'literature' if parent['kind'] == 'literature_seed' else 'prior_context_only'
            parent['current_gain'] = None
        result['current_context'] = copy.deepcopy(current_context)
        result['scope'] = 'prior_context_only_not_current_ranking_or_confirmation'
        return result

    def observe(self, version, *, store, exposure, **kwargs):
        return self.book.observe(version, resolver=self.resolver(store, exposure), **kwargs)

    def revive(self, version, *, store, exposure, **kwargs):
        return self.book.revive(version, resolver=self.resolver(store, exposure), **kwargs)

    def reserve_schedule(self, candidates, budget, *, persist):
        if not callable(persist):
            raise TypeError('Scheduling requires durable memory persistence before work')
        fresh = [c for c in candidates if c['candidate_id'] not in self.scheduled]
        result = schedule_candidates(fresh, budget, used_by_lane=self.lanes)
        self.lanes = result['receipt']['cumulative_reserved']
        self.scheduled += [c['candidate_id'] for c in result['scheduled']]
        persist(self.to_state())
        return result

    def materialize(self, parent_id, attempts, *, family_id, budget, store, exposure, seen=None):
        # Resolve the full ancestor graph before any local use, not only GPT prompts.
        if not callable(getattr(budget, 'persist', None)):
            raise TypeError('Local materialization requires a durable reservation budget')
        self.verify_history()
        parent = self.book.get_parent(parent_id)
        if (parent['kind'] == 'research'
                and self.book.get_version(parent['hypothesis_version'])['state'] != 'active'):
            raise ValueError('Dormant hypotheses require explicit revival before selection')
        resolver = self.resolver(store, exposure)
        self.book.verify_parents([parent_id], resolver=resolver)
        return materialize_mutations(parent['program'], attempts, family_id=family_id, budget=budget, seen=seen)
