from dataclasses import asdict, replace

import pytest

from factorlab.discovery.contracts import ExperimentSpec, ModuleSpec, RepresentationSpec
from factorlab.discovery.search import ResearchProposal, action_menu, mutate, propose


def proposal(parent, spec, operation, arguments):
    return ResearchProposal(spec.digest, parent.digest, spec.dev_end, operation,
                            'Useful hypothesis', 'Reject if not supported', 'test', arguments)


def test_add_merge_drop_and_dimension_change_are_real():
    spec = ExperimentSpec()
    parent = RepresentationSpec()
    added, _ = mutate(parent, spec.training, proposal(parent, spec, 'add', {
        'module': asdict(ModuleSpec('extra', 'mlp'))}), spec)
    assert len(added.modules) == 2
    merged, _ = mutate(added, spec.training, proposal(added, spec, 'merge', {
        'names': ['temporal', 'extra'], 'name': 'both'}), spec)
    assert len(merged.modules) == 1 and merged.modules[0].kind == 'mlp'
    dropped, _ = mutate(added, spec.training, proposal(added, spec, 'drop', {'name': 'extra'}), spec)
    assert dropped == parent
    changed, _ = mutate(parent, spec.training, proposal(parent, spec, 'dimension', {
        'name': 'temporal', 'dim': 16}), spec)
    assert changed.modules[0].dim == 16


def test_proposal_cannot_change_task_budget_or_consume_confirmation():
    spec, parent = ExperimentSpec(), RepresentationSpec()
    p = proposal(parent, spec, 'dimension', {'name': 'temporal', 'dim': 16, 'epochs': 100})
    with pytest.raises(ValueError):
        mutate(parent, spec.training, p, spec)
    p = replace(p, evidence_cutoff='2026-01-01')
    with pytest.raises(ValueError, match='future'):
        p.validate(spec, parent)
    with pytest.raises(ValueError, match='memory'):
        propose(parent, spec, [{'partition': 'confirmation', 'evidence_cutoff': spec.end}], 0)


def test_random_and_external_have_one_shared_menu(tmp_path):
    from factorlab.discovery.evidence import immutable_json
    spec, parent = ExperimentSpec(), RepresentationSpec()
    for i in range(15):
        p = propose(parent, spec, [], i)
        assert {'operation': p.operation, 'arguments': p.arguments} in action_menu(parent, spec)
    chosen = action_menu(parent, spec)[0]
    p = proposal(parent, spec, **chosen)
    immutable_json(tmp_path / 'proposal-0.json', asdict(p))
    assert propose(parent, spec, [], 0, 'external', tmp_path) == p
    assert (tmp_path / 'request-0.json').exists()


def test_last_module_and_noop_cannot_be_deleted():
    spec, parent = ExperimentSpec(), RepresentationSpec()
    with pytest.raises(ValueError):
        mutate(parent, spec.training, proposal(parent, spec, 'drop', {'name': 'temporal'}), spec)
    with pytest.raises(ValueError, match='No-op'):
        mutate(parent, spec.training, proposal(parent, spec, 'learning_rate', {
            'value': spec.training.learning_rate}), spec)
