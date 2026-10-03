"""Frozen diagnostics on tiny synthetic models; no market-efficacy assertions."""
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_discovery_learning import dataset, settings

from factorlab.discovery.contracts import ModuleSpec, RepresentationSpec
from factorlab.discovery.diagnostics import representation_diagnostics
from factorlab.discovery.experiment import ModelGroup
from factorlab.discovery.learning import fit


@pytest.mark.parametrize('combination', ['concat', 'gated'])
def test_frozen_ablation_group_states_and_integrity(combination):
    train, dev = dataset(32), dataset(20, seed=2)
    architecture = RepresentationSpec((ModuleSpec('left', 'mlp', 2, 4, 3),
                                        ModuleSpec('right', 'temporal', 2, 4, 3)), combination)
    config = settings(epochs=1, batch_size=16)
    model = fit(train, dev, architecture, config)
    group = ModelGroup([model, fit(train, dev, architecture, replace(config, seed=2))])
    digest, before = group.digest, group.predict(dev)
    result = representation_diagnostics(group, dev, state_feature='a', state_threshold=.2)
    assert result['status'] == 'ok'
    assert {module['name'] for module in result['modules']} == {'left', 'right'}
    assert len(result['redundancy']) == 1
    assert len(result['redundancy'][0]['per_seed']) == 2
    assert sum(state['rows'] for state in result['states'].values()) == len(dev.x)
    assert all(np.isfinite(module['tasks']['ret']['loss_increase']) for module in result['modules'])
    assert group.digest == digest
    for name, values in before.items():
        np.testing.assert_array_equal(values, group.predict(dev)[name])


def test_unused_module_has_zero_frozen_head_contribution():
    train, dev = dataset(24), dataset(12, seed=2)
    architecture = RepresentationSpec((ModuleSpec('used', 'mlp', 2, 4, 3),
                                        ModuleSpec('unused', 'mlp', 2, 4, 3)))
    model = fit(train, dev, architecture, settings(epochs=1))
    # Make a deliberately unused module, retaining checkpoint/network consistency.
    with torch.no_grad():
        for i, head in enumerate(model.network.heads):
            head.weight[:, 2:] = 0
            model.state[f'heads.{i}.weight'] = head.weight.detach().clone()
    result = representation_diagnostics(model, dev, state_feature='a')
    unused = next(module for module in result['modules'] if module['name'] == 'unused')
    for task in unused['tasks'].values():
        assert task['loss_increase'] == pytest.approx(0, abs=1e-10)


def test_cutoff_hides_unmatured_labels_and_later_origins():
    train, dev = dataset(24), dataset(12, seed=2)
    model = fit(train, dev, None, settings(epochs=1))
    cutoff = dev.meta.available_at.iloc[7]
    before = representation_diagnostics(model, dev, cutoff=cutoff, state_feature='a')
    changed = dev.subset(np.ones(len(dev.meta), dtype=bool))
    for task in dev.tasks:
        future = changed.maturity[task.name] > cutoff.to_datetime64()
        changed.targets[task.name][future] = 0 if task.kind == 'first_event' else 900
    after = representation_diagnostics(model, changed, cutoff=cutoff, state_feature='a')
    assert before == after
    assert before['rows'] == 8
    assert before['metrics']['ret']['support'] == 6
    assert before['modules'] == []


def test_future_append_cannot_change_diagnostics_at_fixed_cutoff():
    train, dev = dataset(24), dataset(12, seed=2)
    model = fit(train, dev, RepresentationSpec(), settings(epochs=1))
    cutoff = dev.meta.available_at.iloc[7]
    prefix = dev.subset(np.arange(8))
    assert representation_diagnostics(model, prefix, cutoff=cutoff, state_feature='a') == (
        representation_diagnostics(model, dev, cutoff=cutoff, state_feature='a'))
