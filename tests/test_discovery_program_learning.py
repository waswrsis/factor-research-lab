"""Actual tiny program fits exercise mechanics; they are not market evidence."""
from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest
import torch

from factorlab.discovery.contracts import (
    ModuleSpec,
    RepresentationSpec,
    TaskDataset,
    TaskSpec,
    TrainingSpec,
)
from factorlab.discovery.learning import (
    FittedResearchModel,
    _inherit_weights,
    _Module,
    _Network,
    fit,
)
from factorlab.discovery.operator_program import VERSION
from factorlab.protocol import fingerprint


def program():
    return {'version': VERSION, 'feature_names': ['a'],
            'parameters': {'alpha': {'initial': .15, 'lower': .02, 'upper': .98}},
            'outputs': [{'op': 'last', 'arg': {
                'op': 'ema', 'arg': {'op': 'feature', 'name': 'a'},
                'alpha': {'op': 'param', 'name': 'alpha'}}}]}


def fixture():
    task = TaskSpec('ret', 'return', 1)
    settings = TrainingSpec(epochs=50, batch_size=64, patience=15, learning_rate=.035,
                            l2=.00001, max_seconds=20)

    def dataset(seed):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(96, 6, 2)).astype(np.float32)
        target = x[:, 0, 0].copy()
        for i in range(1, x.shape[1]):
            target = .8 * x[:, i, 0] + .2 * target
        dates = pd.date_range('2024-01-01' if seed == 1 else '2024-02-01',
                              periods=len(x), freq='h', tz='UTC')
        return TaskDataset(x, {'ret': target}, {'ret': np.ones(len(x), dtype=bool)},
            {'ret': (dates + pd.Timedelta(hours=1)).to_numpy(dtype='datetime64[ns]')},
            pd.DataFrame({'sample_id': [f'{seed}-{i}' for i in range(len(x))],
                          'symbol': 'TEST', 'available_at': dates}), ('a', 'b'), (task,))

    architecture = RepresentationSpec((ModuleSpec('formula', 'program', 3, 8, 6,
                                                 ('a',), program=program()),))
    return dataset(1), dataset(2), architecture, settings


def test_program_contract_validation_defensive_copy_and_legacy_digest():
    source = program()
    module = ModuleSpec('formula', 'program', features=('a',), program=source)
    source['parameters']['alpha']['initial'] = .6
    assert module.program['parameters']['alpha']['initial'] == .15
    with pytest.raises(ValueError):
        ModuleSpec(kind='program')
    with pytest.raises(ValueError, match='Only program'):
        ModuleSpec(program=program())
    with pytest.raises(ValueError):
        ModuleSpec(kind='program', features=('b',), program=program())
    invalid = program()
    invalid['python'] = 'import os'
    with pytest.raises(ValueError):
        ModuleSpec(kind='program', program=invalid)
    root = RepresentationSpec((ModuleSpec('old', 'summary', 3, 8, 6),))
    old = asdict(root)
    for record in old['modules']:
        record.pop('program')
        record.pop('depth')
    assert root.digest == fingerprint(old) == RepresentationSpec.from_dict(old).digest
    assert RepresentationSpec.from_dict(asdict(RepresentationSpec((module,)))).digest == (
        RepresentationSpec((module,)).digest)


def test_program_roots_are_projected_and_selected_features_and_scale_are_enforced():
    source = program()
    source['outputs'].append({'op': 'mean', 'arg': {'op': 'feature', 'name': 'a'}})
    spec = ModuleSpec('formula', 'program', 3, 8, 3, ('a',), program=source)
    module = _Module(spec, ('a', 'b'), 6)
    inputs = torch.randn(4, 6, 2)
    assert module.network[0].in_features == 2
    assert module(inputs).shape == (4, 3)
    altered = inputs.clone()
    altered[:, :3, 0] += 10
    altered[:, :, 1] += 100
    torch.testing.assert_close(module(inputs), module(altered), rtol=0, atol=0)
    too_long = program()
    too_long['outputs'][0]['arg']['arg'] = {
        'op': 'rolling_mean', 'window': 4, 'arg': {'op': 'feature', 'name': 'a'}}
    with pytest.raises(ValueError, match='lookback|history|window'):
        _Module(ModuleSpec('bad', 'program', scale=3, features=('a',), program=too_long),
                ('a', 'b'), 6)


def test_real_fit_updates_internal_indicator_parameter_and_roundtrip(tmp_path):
    train, dev, architecture, settings = fixture()
    model = fit(train, dev, architecture, settings)
    effective = model.program_parameters['formula']['alpha']
    assert .02 <= effective <= .98 and abs(effective - .15) > .001
    assert model.diagnostics['programs']['formula']['trainable_parameters'] == 1
    assert model.diagnostics['programs']['formula']['selected_raw_update_norm'] > .001
    assert model.diagnostics['programs']['formula']['selected_parameter_values']['alpha'] == effective
    updates = [row['program_parameter_updates']['formula'] for row in model.diagnostics['history']]
    assert max(row['preclip_gradient_norm'] for row in updates) > 0
    assert max(row['actual_update_norm'] for row in updates) > 0
    assert model.parameter_count == sum(p.numel() for p in model.network.parameters())
    assert model.encode(dev).shape == (len(dev.x), 3)
    prediction = model.predict(dev)['ret']
    assert np.mean((prediction - dev.targets['ret']) ** 2) < np.var(dev.targets['ret'])
    loaded = FittedResearchModel.load(model.save(tmp_path / 'program.pt'))
    assert loaded.digest == model.digest
    assert loaded.program_parameters == model.program_parameters
    np.testing.assert_array_equal(loaded.predict(dev)['ret'], prediction)
    np.testing.assert_array_equal(loaded.encode(dev), model.encode(dev))


def test_only_exact_program_semantics_inherit_internal_and_head_weights():
    train, dev, architecture, settings = fixture()
    parent = fit(train, dev, architecture, replace(settings, epochs=2))
    matching = _Network(architecture, train.features, train.x.shape[1], train.tasks)
    report = _inherit_weights(matching, parent.metadata, parent)
    assert report['coverage'] == 1
    for name, parameter in matching.named_parameters():
        torch.testing.assert_close(parameter, parent.state[name], rtol=0, atol=0)
    changed = deepcopy(program())
    changed['outputs'][0] = {'op': 'tanh', 'arg': changed['outputs'][0]}
    child_arch = RepresentationSpec((replace(architecture.modules[0], program=changed),))
    assert child_arch.digest != architecture.digest
    child = fit(train, dev, child_arch, replace(settings, epochs=2), initial_model=parent)
    transfer = child.diagnostics['weight_transfer']
    assert transfer['coverage'] == 0
    assert transfer['skipped']['formula'] == 'new_module_or_module_semantics_changed'
    assert transfer['skipped']['heads.0.'] == 'latent_basis_or_target_semantics_changed'
    cold = fit(train, dev, child_arch, replace(settings, epochs=2))
    np.testing.assert_array_equal(child.predict(dev)['ret'], cold.predict(dev)['ret'])


def test_historical_fitted_model_without_program_or_depth_loads(tmp_path):
    train, dev, _architecture, settings = fixture()
    architecture = RepresentationSpec((ModuleSpec('old', 'summary', 3, 8, 6),))
    fitted = fit(train, dev, architecture, replace(settings, epochs=2))
    metadata = deepcopy(fitted.metadata)
    for record in metadata['architecture']['modules']:
        record.pop('program')
        record.pop('depth')
    old = FittedResearchModel(metadata, fitted.state)
    loaded = FittedResearchModel.load(old.save(tmp_path / 'historical.pt'))
    assert loaded.digest == old.digest
    np.testing.assert_array_equal(loaded.predict(dev)['ret'], fitted.predict(dev)['ret'])
