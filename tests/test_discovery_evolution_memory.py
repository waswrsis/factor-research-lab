"""Persistent research context follows causal input semantics across rolling rounds."""
from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest
from test_discovery_evolution import fixture

from factorlab.discovery import evolution, population
from factorlab.discovery.contracts import ModuleSpec, RepresentationSpec
from factorlab.discovery.learning import fit
from factorlab.discovery.research_memory import ResearchMemory
from factorlab.discovery.search import ResearchProposal


def test_builtin_research_excludes_other_feature_schema_but_reuses_rolling_facts(tmp_path, monkeypatch):
    data, spec, _ = fixture()
    architecture = RepresentationSpec((ModuleSpec(dim=4, width=8, scale=2),))
    spec = replace(spec, candidates=1)
    original_search = population.search_population
    histories = []

    def search(*args, **kwargs):
        histories.append(deepcopy(kwargs['history']))
        return original_search(*args, **kwargs)

    def proposal(parent, current_spec, history, iteration, mode, exchange, **kwargs):
        return ResearchProposal(current_spec.digest, parent.digest, current_spec.dev_end,
            'learning_rate', 'Try a bounded learning rate.', 'Check development loss.',
            'test_rule', {'value': .003})

    monkeypatch.setattr(population, 'search_population', search)
    monkeypatch.setattr(population, 'propose', proposal)
    memory_path = tmp_path / 'memory.sqlite'

    def research(source, name, cutoff):
        train = source.subset(np.arange(35))
        dev = source.subset(np.arange(40, 60))
        model = fit(train, dev, architecture, spec.training)
        return evolution._research_candidate(train, dev, model, spec, evolution.EvolutionPolicy(),
            tmp_path / name, pd.Timestamp(cutoff), 'cpu', memory_path, None)

    research(data, 'original', data.meta.available_at.iloc[64])
    changed = replace(data, features=('a', 'funding'))
    research(changed, 'different-schema', data.meta.available_at.iloc[66])
    research(data, 'rolling-original', data.meta.available_at.iloc[68])
    assert histories[0] == []
    assert histories[1] == [], 'Same primary task cannot import another input schema'
    assert len(histories[2]) == 1
    assert histories[2][0]['source_protocol_hash'] != histories[2][0]['protocol_hash']
    memory = ResearchMemory(memory_path)
    task = asdict(spec.tasks[0])
    original = memory.retrieve(task, spec.dev_end, context_hash=evolution._memory_context_hash(data, spec))
    other = memory.retrieve(task, spec.dev_end, context_hash=evolution._memory_context_hash(changed, spec))
    assert len(original) == 2 and len(other) == 1
    assert all(row['structure']['modules'][0]['features'] == [] for row in original + other)


@pytest.mark.parametrize('change', ['symbols', 'lookback', 'context_rules', 'availability', 'feed', 'delay'])
def test_memory_identity_changes_with_input_or_knowability_semantics(change):
    data, spec, _ = fixture()
    before = evolution._memory_context_hash(data, spec)
    data.meta.attrs.clear()
    if change == 'symbols':
        spec = replace(spec, symbols=('OTHER',))
    elif change == 'lookback':
        spec = replace(spec, lookback=3)
    elif change == 'context_rules':
        data.meta.attrs['context_policy'] = {'max_age_hours': 8, 'sources': {'funding': {}}}
    elif change == 'availability':
        data.meta.attrs['availability_mode'] = 'actual_rest_receipt'
    elif change == 'delay':
        spec = replace(spec, availability_delay_seconds=60)
    else:
        data.meta.attrs['feed_identity'] = 'different_feed_seed'
    assert evolution._memory_context_hash(data, spec) != before


def test_memory_identity_ignores_rolling_cutoffs_and_context_source_append_hashes():
    data, spec, _ = fixture()
    data.meta.attrs['context_policy'] = {'max_age_hours': 24, 'cross_assets': True,
        'sources': {'funding': {'rows': 1, 'sha256': 'first', 'provenance': {'path': 'old'}}}}
    before = evolution._memory_context_hash(data, spec)
    data.meta.attrs['context_policy']['sources']['funding'] = {
        'rows': 2, 'sha256': 'appended', 'provenance': {'path': 'new'}}
    rolling = replace(spec, train_end=data.meta.available_at.iloc[65].isoformat(),
                      dev_end=data.meta.available_at.iloc[146].isoformat())
    assert evolution._memory_context_hash(data, rolling) == before
    data.meta.attrs['context_policy']['sources']['open_interest'] = {}
    assert evolution._memory_context_hash(data, rolling) != before
