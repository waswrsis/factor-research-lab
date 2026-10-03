"""Tiny real fits verify mechanics only; no market efficacy is implied."""
from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest
import torch

from factorlab.discovery import population
from factorlab.discovery.contracts import (
    ExperimentSpec,
    ModuleSpec,
    RepresentationSpec,
    TaskDataset,
    TaskSpec,
    TrainingSpec,
)
from factorlab.discovery.learning import FittedResearchModel, _inherit_weights, _Network, fit
from factorlab.discovery.search import ResearchProposal, _clean_history, action_menu, propose
from factorlab.protocol import fingerprint


def fixture():
    task = TaskSpec('ret', 'return', 1)
    settings = TrainingSpec(epochs=1, batch_size=32, patience=1, max_seconds=15)
    spec = ExperimentSpec(tasks=(task,), primary_task='ret', lookback=4, symbols=('TEST',),
        start='2024-01-01', train_end='2024-01-04', dev_end='2024-01-07',
        calibration_end='2024-01-09', end='2024-01-11', training=settings,
        training_seeds=(17,), candidates=4, search_seconds=60)

    def data(start, seed):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(48, 4, 2)).astype(np.float32)
        dates = pd.date_range(start, periods=48, freq='h', tz='UTC')
        return TaskDataset(x, {'ret': 2 * x[:, -1, 0] - x[:, -1, 1]},
            {'ret': np.ones(48, dtype=bool)},
            {'ret': (dates + pd.Timedelta(hours=1)).to_numpy(dtype='datetime64[ns]')},
            pd.DataFrame({'sample_id': [f'{seed}-{i}' for i in range(48)],
                          'symbol': 'TEST', 'available_at': dates}), ('a', 'b'), (task,))
    architecture = RepresentationSpec((ModuleSpec('one', 'temporal', 3, 5, 3),
                                       ModuleSpec('two', 'mlp', 3, 5, 3)))
    return data('2024-01-01', 1), data('2024-01-04', 2), architecture, spec


def test_real_population_retains_roots_and_multiple_structures(monkeypatch):
    train, dev, root, spec = fixture()
    other = RepresentationSpec((ModuleSpec('other', 'summary', 3, 5, 3),))
    seen_parents = []

    def structural(parent, protocol, history, iteration, *args, **kwargs):
        seen_parents.append(parent.digest)
        module = parent.modules[0]
        return ResearchProposal(protocol.digest, parent.digest, protocol.dev_end,
            'width', 'Change bounded capacity', 'Reject without development benefit', 'test',
            {'name': module.name, 'width': 8})

    monkeypatch.setattr(population, 'propose', structural)
    best, result = population.search_population(train, dev, [root, other], spec,
                                                inherit_weights=True, population_size=4)
    assert best is not None
    assert seen_parents == [root.digest, other.digest]
    assert result['newly_fitted'] == spec.candidates
    assert len(result['root_ids']) == 2
    assert set(result['root_ids']) <= set(result['population_ids'])
    assert len(result['archive_ids']) == result['distinct_architectures'] == 4
    assert best.diagnostics['optimizer_steps'] > 0 if hasattr(best, 'diagnostics') else all(
        m.diagnostics['optimizer_steps'] > 0 for m in best.models)
    assert result['best_loss'] <= min(row['primary_loss'] for row in result['records'][:2])
    assert result['monotonic_improvement_claimed'] is False
    child = result['records'][2]
    assert child['parent_id'] == result['root_ids'][0] and child['generation'] == 1
    assert child['training'][0]['weight_transfer']['coverage'] > 0


def test_reused_original_is_evaluated_and_no_fit_budget_is_spent(monkeypatch):
    train, dev, root, spec = fixture()
    fitted = fit(train, dev, root, spec.training)
    monkeypatch.setattr(population, 'propose', lambda *args, **kwargs: (_ for _ in ()).throw(
        ValueError('declared test rejection')))
    best, result = population.search_population(train, dev, root, replace(spec, candidates=1),
                                                initial_models={root.digest: fitted})
    assert best is fitted and result['newly_fitted'] == 0
    assert result['records'][0]['reused'] is True
    assert result['records'][-1]['error'] == 'ValueError: declared test rejection'
    assert result['proposal_attempts'] == 16


def test_initial_model_trained_inside_current_development_is_replaced_by_cold_root():
    train, dev, root, spec = fixture()
    prefix = dev.subset(np.arange(8))
    contaminated = TaskDataset(np.concatenate((train.x, prefix.x)),
        {name: np.concatenate((train.targets[name], prefix.targets[name])) for name in train.targets},
        {name: np.concatenate((train.observed[name], prefix.observed[name])) for name in train.observed},
        {name: np.concatenate((train.maturity[name], prefix.maturity[name])) for name in train.maturity},
        pd.concat((train.meta, prefix.meta), ignore_index=True), train.features, train.tasks)
    parent = fit(contaminated, dev, root, spec.training)
    best, result = population.search_population(train, dev, root, replace(spec, candidates=1),
                                                initial_models={root.digest: parent})
    assert best is not None and best.digest != parent.digest
    assert result['newly_fitted'] == 1
    assert result['records'][0]['reused'] is False
    assert result['records'][0]['rejected_initial_reason'] == 'Initial candidate contains future evidence'
    assert best.models[0].diagnostics['weight_transfer']['requested'] is False


def test_inheritance_copies_named_semantic_modules_and_skips_changed_basis():
    train, dev, root, spec = fixture()
    parent = fit(train, dev, root, spec.training)
    child_arch = RepresentationSpec((root.modules[1], replace(root.modules[0], width=8)))
    child = fit(train, dev, child_arch, spec.training, initial_model=parent)
    transfer = child.diagnostics['weight_transfer']
    assert 0 < transfer['coverage'] < 1
    assert transfer['source_digest'] == parent.digest
    assert transfer['skipped']['one'] == 'new_module_or_module_semantics_changed'
    assert transfer['skipped']['heads.0.'] == 'latent_basis_or_target_semantics_changed'
    initialized = _Network(child_arch, train.features, spec.lookback, train.tasks)
    report = _inherit_weights(initialized, child.metadata, parent)
    assert report['coverage'] == transfer['coverage']
    for name, value in initialized.named_parameters():
        if name.startswith('modules_by_name.0.'):
            torch.testing.assert_close(value, parent.state[name.replace('modules_by_name.0.',
                                                                        'modules_by_name.1.')])
    same = fit(train, dev, root, spec.training, initial_model=parent)
    assert same.diagnostics['weight_transfer']['coverage'] == 1
    assert same.diagnostics['optimizer_steps'] > 0
    changed_train = replace(train, x=train.x + 1)
    skipped = fit(changed_train, dev, root, spec.training, initial_model=parent)
    assert skipped.diagnostics['weight_transfer']['coverage'] == 0
    assert skipped.diagnostics['weight_transfer']['skipped']['all'] == 'input_standardization_changed'
    cold = fit(changed_train, dev, root, spec.training)
    np.testing.assert_array_equal(cold.predict(dev)['ret'], skipped.predict(dev)['ret'])
    retained = fit(changed_train, dev, root, spec.training, initial_model=parent,
                   retain_initial_scalers=True)
    np.testing.assert_array_equal(retained.metadata['input_mean'], parent.metadata['input_mean'])
    assert retained.metadata['target_scales'] == parent.metadata['target_scales']
    assert retained.diagnostics['weight_transfer']['coverage'] == 1
    assert retained.diagnostics['weight_transfer']['scaler_source_digest'] == parent.digest
    assert retained.diagnostics['optimizer_steps'] > 0
    assert np.isfinite(retained.predict(dev)['ret']).all()


def test_inheritance_rejects_future_parent_and_population_rejects_future_inputs():
    train, dev, root, spec = fixture()
    parent = fit(train, dev, root, spec.training)
    earlier_dev = dev.subset(np.arange(20))
    with pytest.raises(ValueError, match='later'):
        fit(train, earlier_dev, root, spec.training, initial_model=parent)
    with pytest.raises(ValueError, match='partition'):
        population.search_population(train, replace(dev, meta=dev.meta.assign(
            available_at=dev.meta.available_at + pd.Timedelta(days=5)), maturity={
                'ret': dev.maturity['ret'] + np.timedelta64(5, 'D')}), root, spec)


def test_depth_feature_scale_and_pair_merge_actions_are_executable():
    train, dev, root, spec = fixture()
    from factorlab.discovery.search import mutate
    parent = RepresentationSpec((*root.modules, ModuleSpec('three', 'summary', 3, 5, 3)))
    actions = action_menu(parent, spec, features=train.features)
    assert {'features', 'scale', 'width', 'depth', 'dimension'} <= {a['operation'] for a in actions}
    merges = [a for a in actions if a['operation'] == 'merge']
    assert len(merges) == 3 and all(len(a['arguments']['names']) == 2 for a in merges)
    for operation in ('features', 'scale', 'width', 'depth', 'merge'):
        chosen = next(a for a in actions if a['operation'] == operation)
        proposal = ResearchProposal(spec.digest, parent.digest, spec.dev_end, operation,
                                    'bounded edit', 'reject if worse', 'test', chosen['arguments'])
        child, _ = mutate(parent, spec.training, proposal, spec)
        assert child != parent
        model = fit(train, dev, child, spec.training)
        assert np.isfinite(model.predict(dev)['ret']).all()


def test_action_memory_survives_roundtrip_and_external_packet_excludes_extra_data(tmp_path):
    _train, _dev, root, spec = fixture()
    chosen = next(a for a in action_menu(root, spec) if a['operation'] == 'width')
    proposal = ResearchProposal(spec.digest, root.digest, spec.dev_end, chosen['operation'],
                                'bounded edit', 'reject if worse', 'test', chosen['arguments'])
    record = {'partition': 'development', 'evidence_cutoff': spec.dev_end, 'status': 'failed',
              'architecture': asdict(root), 'parent_architecture': asdict(root),
              'parent_settings': asdict(spec.training), 'proposal': asdict(proposal),
              'error': 'Parameter budget exceeded', 'metrics': {'ret': {'loss': .2,
              'confirmation_loss': 99}}, 'training': [{'test': 123}], 'iteration': 0}
    clean = _clean_history([record], spec)
    assert _clean_history(clean, spec) == clean
    assert clean[0]['proposal'] == asdict(proposal) and clean[0]['error'] == record['error']
    assert 'training' not in clean[0] and clean[0]['metrics'] == {'ret': {'loss': .2}}
    from factorlab.discovery.evidence import immutable_json
    immutable_json(tmp_path / 'proposal-1.json', asdict(proposal))
    propose(root, spec, clean, 1, 'external', tmp_path)
    import json
    packet = json.loads((tmp_path / 'request-1.json').read_text())
    assert packet['history'] == json.loads(json.dumps(clean))
    bad = deepcopy(record)
    bad['proposal']['arguments']['arbitrary_python'] = 'execute()'
    with pytest.raises(ValueError, match='arguments'):
        _clean_history([bad], spec)
    with pytest.raises(ValueError, match='Confirmation'):
        _clean_history([{**record, 'confirmation_loss': 99}], spec)


def test_old_default_depth_architecture_identity_and_fitted_load(tmp_path):
    train, dev, root, spec = fixture()
    old = asdict(root)
    for module in old['modules']:
        module.pop('depth')
        module.pop('program', None)
    assert RepresentationSpec.from_dict(old).digest == fingerprint(old) == root.digest
    fitted = fit(train, dev, root, spec.training)
    metadata = deepcopy(fitted.metadata)
    metadata['architecture'] = old
    legacy = FittedResearchModel(metadata, fitted.state)
    loaded = FittedResearchModel.load(legacy.save(tmp_path / 'legacy.pt'))
    assert loaded.digest == legacy.digest
    np.testing.assert_array_equal(loaded.predict(dev)['ret'], fitted.predict(dev)['ret'])
