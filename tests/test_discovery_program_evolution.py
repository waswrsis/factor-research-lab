"""Tiny autonomous formula evolution; artificial labels are not market evidence."""
import json
import sqlite3
from dataclasses import replace

import pytest
from test_discovery_evolution import fixture

from factorlab.discovery.evolution import (
    EvolutionPolicy,
    EvolutionStore,
    inspect_evolution,
    run_evolution,
)
from factorlab.discovery.learning import FittedResearchModel
from factorlab.discovery.operator_program import Program
from factorlab.discovery.operator_search import clean_diagnostics
from factorlab.protocol import fingerprint


def test_builtin_program_search_two_cycles_recovers_memory_and_ancestors(tmp_path):
    data, spec, incumbent = fixture()
    spec = replace(spec, candidates=3, search_seconds=45)
    root = tmp_path / 'operator-evolution'
    memory = tmp_path / 'memory.sqlite'
    policy = EvolutionPolicy(max_cycles=2, shadow_hours=16, update_every_hours=24,
                             program_search=True, revived_parents=1, population_size=3)
    result = run_evolution(data, incumbent, spec, root, policy=policy,
                           memory_path=memory, registry_path=tmp_path / 'evidence.sqlite',
                           evidence_source='synthetic')
    assert len(result['cycles']) == 2
    assert all(cycle['status'] in {'promoted', 'retained', 'unchanged_retained'}
               for cycle in result['cycles']), result['cycles']
    first, second = [cycle['research'] for cycle in result['cycles']]
    assert first['program_search'] and second['program_search']
    assert first['revived_architectures'] == []
    assert len(second['revived_architectures']) == 1
    assert all(round_['newly_fitted'] <= 3 for round_ in (first, second))
    generated = [row for row in first['records']
                 if row.get('proposal', {}).get('operation') == 'write_program'
                 and row['status'] == 'ok']
    assert generated, first['records']
    assert any(module['kind'] == 'program' for row in generated
               for module in row['architecture']['modules'])
    # Current research receives durable typed outcomes from the previous cycle.
    prior = [row for row in second['development_memory']
             if row.get('proposal', {}).get('source') == 'persistent_development_action_memory']
    assert any(row['proposal']['operation'] == 'write_program' for row in prior)
    assert all(row['partition'] == 'development' for row in prior)
    assert all(row['source_protocol_hash'] != row['protocol_hash'] for row in prior)
    # Revived structures are new roots and are re-scored/refitted for this split.
    old_architectures = {row['architecture_hash'] for row in first['records']
                         if row['status'] == 'ok'}
    new_roots = [row for row in second['records'] if row.get('origin') == 'root'
                 and row['status'] == 'ok']
    assert len(new_roots) >= 2
    assert any(row['architecture_hash'] in old_architectures for row in new_roots)
    revived = [row for row in new_roots
               if row['architecture_hash'] in second['revived_architectures']]
    assert len(revived) == 1 and revived[0]['reused'] is False
    assert revived[0]['architecture_hash'] in old_architectures
    with sqlite3.connect(memory) as connection:
        facts = [json.loads(row[0]) for row in connection.execute(
            'SELECT payload FROM action_results ORDER BY cutoff_ns')]
    successful = [fact for fact in facts if fact['status'] == 'ok'
                  and fact['action']['operation'] == 'write_program']
    assert successful
    for fact in successful:
        diagnostics = fact['diagnostics']
        assert clean_diagnostics(diagnostics) == diagnostics
        assert diagnostics['optimizer_steps'] > 0
        assert diagnostics['max_gradient_norm'] > 0
        assert diagnostics['max_update_norm'] > 0
        assert diagnostics['program_parameter_update_norm'] > 0
    assert any(member['programs'] for row in generated for member in row['training'])
    # Both cycles save readable definitions and fitted seed values, while the
    # shared local library keeps one immutable definition per semantic digest.
    exported = list(root.glob('research/*/candidate-*/operators.json'))
    assert exported
    cycle_exports = {path.parents[1].name for path in exported}
    assert len(cycle_exports) == 2
    library = list((root / 'operator-library').glob('*.json'))
    assert library
    for path in library:
        definition = json.loads(path.read_text('utf-8'))
        compiled = Program.from_dict(definition['program'])
        assert path.stem == definition['digest'] == compiled.digest
        assert definition['source_listing'] == compiled.source
    for path in exported:
        definitions = json.loads(path.read_text('utf-8'))
        for definition in definitions.values():
            compiled = Program.from_dict(definition['program'])
            assert definition['digest'] == compiled.digest
            assert definition['source_listing'] == compiled.source
            assert len(definition['seed_parameters']) == len(spec.training_seeds)
            assert set(definition['seed_parameters'][0]) == set(compiled.parameters)
    archived = json.loads((root / 'source' / 'manifest.json').read_text('utf-8'))
    assert {'factorlab/discovery/operator_program.py',
            'factorlab/discovery/operator_search.py'} <= set(archived)
    assert result['source_evidence']['archive_verified'] is True
    inspected = inspect_evolution(root)
    assert inspected['event_head'] == result['event_head']
    assert inspected['cycles'] == result['cycles']
    # Reopening completed state does not generate another cycle or modify history.
    resumed = run_evolution(data, None, spec, root, policy=policy, resume=True,
                            memory_path=memory, registry_path=tmp_path / 'evidence.sqlite',
                            evidence_source='synthetic')
    assert resumed == result


def test_preprogram_fixed_policy_checkpoint_remains_historically_inspectable(tmp_path):
    """Test-only legacy-shaped ledger; no existing historical run is relabeled."""
    data, spec, incumbent = fixture()
    root = tmp_path / 'legacy-fixed'
    policy = EvolutionPolicy(mode='fixed')
    run_evolution(data, incumbent, spec, root, policy=policy, max_ticks=3,
                  memory_path=tmp_path / 'memory.sqlite',
                  registry_path=tmp_path / 'evidence.sqlite', evidence_source='synthetic')
    checkpoint = next((root / 'models').rglob('model.pt'))
    loaded = FittedResearchModel.load(checkpoint)
    assert loaded.digest == incumbent.digest
    assert loaded.program_parameters == {}
    store = EvolutionStore(root)
    try:
        state = store.state()
        for key in ('program_search', 'revived_parents'):
            state['identity']['policy'].pop(key)
        manifest = json.loads((root / 'manifest.json').read_text('utf-8'))
        manifest['manifest']['policy'] = state['identity']['policy']
        manifest['sha256'] = fingerprint(manifest['manifest'])
        (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        with store.db:
            store.save_state(state)
    finally:
        store.close()
    historical = inspect_evolution(root)
    assert historical['mode'] == 'fixed'
    assert historical['incumbent'] == incumbent.digest
    assert historical['ticks'] == 3
    assert historical['source_evidence']['archive_verified'] is True
    # Legacy policy/source identities stay historical; a new run owns new behavior.
    with pytest.raises(ValueError, match='identity changed'):
        run_evolution(data, None, spec, root, policy=policy, resume=True,
                      memory_path=tmp_path / 'memory.sqlite',
                      registry_path=tmp_path / 'evidence.sqlite', evidence_source='synthetic')
