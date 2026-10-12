"""Descriptive S selection checks; no statistical or Linux acceptance claim."""
import copy
import multiprocessing
import os
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from test_v7_neural_research_ticket import (
    _actual_child,
    _synthetic_kernel,
    launcher,
)
from test_v7_neural_research_ticket import (
    case as case,  # noqa: PLC0414
)

from factorlab.discovery import algorithm_shortlist as module
from factorlab.discovery.neural_research_ticket import issue_research_ticket
from factorlab.discovery.nonlinear_channel import StructureSpec
from factorlab.discovery.rolling import development_folds


def catalogue(case, recipes=None, rule=None):
    return module.AlgorithmCatalogue.freeze(case[0], recipes or (case[2],), rule=rule)


def arithmetic_result(recipe, *, ensemble=.8, k0=1., pv=1., epoch=1, fraction=1., seed_loss=0.):
    """Private arithmetic fixture; never passed to record_ticket as actual evidence."""
    folds = []
    for i, fold in enumerate(development_folds('2025-04-01T00:00:00+00:00')):
        hours = pd.date_range(fold['fit_cutoff'], fold['score_end'], freq='h', inclusive='left')
        observed = hours[:int(len(hours) * fraction)]
        losses = {'ensemble': ensemble, 'K0': k0, 'Pv': pv, 'seed_17': seed_loss,
                  'initial_encoder_ridge': .000001, 'trained_encoder_ridge': .0000001}
        folds.append({**fold, 'role': 'S', 'status': 'completed', 'nominal_hours': len(hours),
            'nominal_rows': len(hours), 'support': len(observed), 'coverage': len(observed) / len(hours),
            'common_sample_ids': [f'{i}:{j}' for j in range(len(observed))],
            'hourly': [{'hour': t.isoformat(), 'symbols_present': 1, 'losses': losses} for t in observed],
            'seed_checkpoints': [{'seed': seed, 'epoch': epoch} for seed in recipe.config.seeds],
            'losses': losses, 'mean_seed_loss_diagnostic': seed_loss})
    return {'role': 'S', 'status': 'completed', 'test_only': False, 'formal_claim': False,
            'pipeline_id': recipe.pipeline_id, 'symbols': ['X'], 'cutoff': '2025-04-01T00:00:00+00:00', 'folds': folds}


def test_registry_is_full_recipe_and_every_config_change_has_new_id(case):
    base = case[2]
    variants = module.registered_recipes(base)
    cat = catalogue(case, variants)
    assert len(cat.recipes) == 4
    changes = [replace(base, output='encoder_ridge'),
        replace(base, config=replace(base.config, seeds=(19, 31, 47))),
        replace(base, config=replace(base.config, epochs=9)),
        replace(base, config=replace(base.config, structures=(StructureSpec(16, 4),)))]
    assert len({base.pipeline_id, *(r.pipeline_id for r in changes)}) == 5
    for key, recipe in cat.recipes.items():
        assert recipe.pipeline_id == key
        assert recipe.to_dict()['pool'] == base.pool.to_dict()
        assert recipe.to_dict()['rules'] == base.to_dict()['rules']


@pytest.mark.parametrize('extra', [{'code': 'import os'}, {'experts': 2}, {'role': 'V'}, {'C_gain': .9}])
def test_GPT_accepts_only_registered_IDs_and_no_feedback_fields(case, extra):
    cat = catalogue(case)
    reply = {'version': module.PROPOSAL_VERSION, 'catalogue_id': cat.digest, 'candidate_ids': [case[2].pipeline_id]}
    assert cat.parse_proposal(reply)[0].to_dict() == case[2].to_dict()
    with pytest.raises(ValueError, match='only choose'):
        cat.parse_proposal({**reply, **extra})
    with pytest.raises(ValueError):
        cat.parse_proposal({**reply, 'candidate_ids': ['unregistered']})


def test_protocol_and_registry_cannot_change_after_freeze(case):
    cat = catalogue(case)
    with pytest.raises(ValueError, match='cannot change'):
        catalogue(case, rule=module.AlgorithmShortlistRule(minimum_hour_coverage=.1))
    with pytest.raises(ValueError, match='cannot change'):
        catalogue(case, recipes=module.registered_recipes(case[2]))
    with pytest.raises(ValueError):
        module.registered_recipes(case[2], structures=((16, 4, 2),))
    cat.rule = module.AlgorithmShortlistRule(minimum_hour_coverage=.1)
    with pytest.raises(ValueError, match='identity changed'):
        cat.prompt_packet()


@pytest.mark.parametrize('missing', ['identity', 'anchor'])
def test_missing_catalogue_identity_or_external_anchor_cannot_reset(case, missing):
    cat = catalogue(case)
    path = cat.directory / 'identity.json' if missing == 'identity' else Path(cat.identity['anchor'])
    path.unlink()
    with pytest.raises(ValueError, match='Missing/oversized'):
        catalogue(case)


def test_actual_ensemble_not_best_seed_or_encoder_control_selects(case):
    recipe = case[2]
    result = arithmetic_result(recipe, ensemble=1.1, seed_loss=0.)
    score = module._score(result, recipe, module.AlgorithmShortlistRule())
    assert not score['eligible']
    assert score['gains']['Pv'] < 0
    assert score['folds'][0]['mean_seed_loss_diagnostic'] == 0
    assert score['folds'][0]['diagnostic_losses']['trained_encoder_ridge'] < .00001


def test_both_references_required_and_epoch0_kept_not_selected(case):
    recipe, rule = case[2], module.AlgorithmShortlistRule()
    assert not module._score(arithmetic_result(recipe, ensemble=.8, pv=.7), recipe, rule)['eligible']
    result = module._score(arithmetic_result(recipe, epoch=0), recipe, rule)
    assert not result['eligible'] and result['epoch0_only']
    assert 'epoch0_only' in result['reasons']
    assert result['folds'][0]['seed_checkpoints'][0]['epoch'] == 0


def test_every_fold_keeps_original_nominal_coverage(case):
    recipe, rule = case[2], module.AlgorithmShortlistRule()
    result = arithmetic_result(recipe, fraction=.5)
    score = module._score(result, recipe, rule)
    assert not score['eligible']
    assert score['folds'][0]['hour_coverage'] == .5
    assert score['folds'][0]['nominal_hours'] == 31 * 24
    result['folds'][0]['nominal_rows'] //= 2
    with pytest.raises(ValueError, match='frozen calendar'):
        module._score(result, recipe, rule)


def test_tiebreak_actual_recipe_shortlist_and_incompatible_support_closes(case, monkeypatch):
    variants = module.registered_recipes(case[2], structures=((8, 2),))
    cat = catalogue(case, variants)
    # Private arithmetic isolation only. Real receipt acceptance is exercised by
    # the actual OS-child integration below, not by these fabricated score rows.
    records = {r.pipeline_id: {'pipeline_id': r.pipeline_id, 'status': 'completed',
        'ticket': {'private_arithmetic_fixture': True},
        'summary': module._score(arithmetic_result(r), r, cat.rule)} for r in variants}
    monkeypatch.setattr(cat, '_records', lambda: copy.deepcopy(records))
    selected = cat.seal_shortlist()
    assert [r['pipeline_id'] for r in selected['algorithm_shortlist']] == sorted(records)
    assert selected['prohibited_route'] == 'Lifecycle.freeze_batch/factor1/factor2/factor3'
    assert all(r['recipe']['rules'] and r['candidate_type'] == 'complete_neural_algorithm'
               and r['factor_claim_eligible'] is False for r in selected['algorithm_shortlist'])
    key = min(records)
    records[key]['summary']['folds'][0]['sample_ids_hash'] = 'different'
    with pytest.raises(ValueError):
        cat.seal_shortlist()  # Already-sealed output cannot be silently replaced.


def test_current_V_contamination_closes_GPT_packet(case):
    cat = catalogue(case)
    exposed = case[0].state['search_exposure']
    case[0].store.exposure.record(**{k: v for k, v in exposed.items() if k != 'data_id'}, role='V')
    with pytest.raises(ValueError):
        cat.prompt_packet()


def test_failed_ticket_retains_prepaid_cost_but_no_performance(case):
    cat = catalogue(case)
    ticket = issue_research_ticket(case[0], case[1], case[2], gates=case[3])
    case[0].authority.fail_consumed(ticket.value['reservation_id'], 'synthetic resource failure')
    record = cat.record_ticket(ticket.path)
    assert record['status'] == 'failed_consumed' and record['prepaid_counts']['neural_fit'] == 9
    assert record['summary'] is None
    assert cat.seal_shortlist()['algorithm_shortlist'] == []


def test_unrecorded_pending_attempt_cannot_be_omitted_from_selection(case):
    cat = catalogue(case)
    issue_research_ticket(case[0], case[1], case[2], gates=case[3])
    with pytest.raises(ValueError, match='Outstanding prepaid'):
        cat.seal_shortlist()
    assert cat.prompt_packet()['S_readouts'][0]['status'] == 'reserved'


def test_unrecorded_failed_attempt_recovered_from_authority(case):
    cat = catalogue(case)
    ticket = issue_research_ticket(case[0], case[1], case[2], gates=case[3])
    case[0].authority.fail_consumed(ticket.value['reservation_id'], 'failed before catalogue callback')
    result = cat.seal_shortlist()
    assert result['all_candidates'][0]['status'] == 'failed_consumed'
    assert result['algorithm_shortlist'] == []
    assert cat._records()[case[2].pipeline_id]['prepaid_counts']['neural_fit'] == 9


def test_preparation_crash_is_retained_without_performance_or_retry(case, monkeypatch):
    cat = catalogue(case)
    monkeypatch.setattr('factorlab.discovery.neural_research_ticket.save_worker_data',
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError('artificial write failure')))
    with pytest.raises(OSError, match='artificial write failure'):
        issue_research_ticket(case[0], case[1], case[2], gates=case[3])
    records = cat._records()
    record = records[case[2].pipeline_id]
    assert record['status'] == 'failed_consumed' and record['summary'] is None
    assert record['prepaid_counts']['epoch'] == 3 * len(case[2].config.seeds) * case[2].config.epochs
    original = case[0].authority.verify()
    with pytest.raises(ValueError, match='Partial S preparation'):
        issue_research_ticket(case[0], case[1], case[2], gates=case[3])
    assert case[0].authority.verify() == original
    result = cat.seal_shortlist()
    assert result['all_candidates'][0]['status'] == 'failed_consumed'
    assert case[0].authority.verify()['reservations'] == 2  # Original baseline plus failed neural work.


def test_different_actual_common_support_closes_entire_shortlist(case, monkeypatch):
    variants = module.registered_recipes(case[2], structures=((8, 2),))
    cat = catalogue(case, variants)
    records = {r.pipeline_id: {'pipeline_id': r.pipeline_id, 'status': 'completed',
        'ticket': {'private_arithmetic_fixture': True},
        'summary': module._score(arithmetic_result(r), r, cat.rule)} for r in variants}
    records[min(records)]['summary']['folds'][0]['sample_ids_hash'] = 'different'
    monkeypatch.setattr(cat, '_records', lambda: copy.deepcopy(records))
    result = cat.seal_shortlist()
    assert result['status'] == 'closed_different_candidate_common_support'
    assert result['algorithm_shortlist'] == []


def test_real_three_fold_ticket_result_schema_and_complete_shortlist_artifact(case, monkeypatch):
    bridge, data, recipe, gates = case
    # Execute the already-frozen complete recipe, including its epochs.
    # Changing epochs here would violate the original nonlinear work plan.
    cat = catalogue(case, (recipe,), module.AlgorithmShortlistRule(
        minimum_hour_coverage=.001, minimum_symbol_hour_coverage=.001))
    ticket = issue_research_ticket(bridge, data, recipe, gates=gates)
    children = []

    class ActualChild:
        def __init__(self, command, **kwargs):
            assert command[3] == 'factorlab.discovery.neural_research_launcher'
            self.process = multiprocessing.get_context('spawn').Process(target=_actual_child,
                args=(str(bridge.store.directory.parent), str(ticket.path)))
            self.process.start()
            self.pid = self.process.pid
            children.append(self)

        @property
        def returncode(self):
            return self.process.exitcode

        def poll(self):
            return self.process.exitcode

    monkeypatch.setattr(launcher.subprocess, 'Popen', ActualChild)
    monkeypatch.setattr(launcher, '_kernel_snapshot', lambda: _synthetic_kernel(
        [os.getpid()] + [p.pid for p in children if p.poll() is None]))
    monkeypatch.setattr(launcher, '_strict_kernel', lambda *args: None)
    launcher.launch_research(ticket.path)
    recorded = cat.record_ticket(ticket.path)
    assert recorded['status'] == 'completed'
    assert len(recorded['summary']['folds']) == 3
    assert recorded['actual_counts']['neural_fit'] == 9
    assert recorded['research']['path'].endswith('research\\result.json') or recorded['research']['path'].endswith('research/result.json')
    output = cat.seal_shortlist()
    assert output['all_candidates'][0]['status'] == 'completed'
    assert output['formal_claim'] is False and output['factor_claim_eligible'] is False
    assert len(output['algorithm_shortlist']) == int(recorded['summary']['eligible'])
    if output['algorithm_shortlist']:
        assert output['algorithm_shortlist'][0]['recipe'] == module._copy(recipe.to_dict())
    # Numerical artifacts remain referenced once, rather than copying large raw
    # models/predictions into each catalogue receipt and GPT prompt.
    assert 'predictions' not in str(recorded.keys())
    ticket.path.unlink()
    with pytest.raises(ValueError):
        cat.seal_shortlist()  # A cached summary cannot replace lost original evidence.
