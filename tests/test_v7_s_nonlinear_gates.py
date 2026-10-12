"""Gate chronology/identity with private producer-read seams, never Linux proof."""
import copy
from types import SimpleNamespace

import pytest
from test_v7_neural_research_ticket import case as original_case  # noqa: F401
from test_v7_s_nonlinear_ticket import case as case  # noqa: PLC0414

from factorlab.discovery import s_nonlinear_gates as gates
from factorlab.discovery import s_nonlinear_launcher as launcher
from factorlab.discovery.neural_research_ticket import _context
from factorlab.protocol import fingerprint


def setup(case, monkeypatch):
    bridge, data, recipe, _ = case
    epoch = gates.freeze_shared_epoch(bridge, data, recipe)
    context = _context(bridge.store, bridge.month, bridge.state, bridge.identity, recipe)
    source = fingerprint(bridge.authority.manifest['source_files'])
    # Only the read seam is synthetic; epoch/authority/double anchors are actual.
    context = {**context, 'cutoff': '2027-07-01T00:00:00+00:00'}
    reports = {}
    for index, quarter in enumerate(('2027-01-01T00:00:00+00:00', '2027-04-01T00:00:00+00:00')):
        value = {'kind': 'quarterly_probe', 'context': {**context, 'cutoff': quarter},
                 'recipe': gates._canonical(recipe.to_dict()), 'source_hash': source,
                 'authority_directory': str(bridge.authority.directory),
                 'lease': {'wall_started_at': '2027-05-01T00:00:00+00:00'}}
        reports[str(index)] = {'ticket': SimpleNamespace(value=value),
            'publication': {'observation': {'wall': '2027-06-01T00:00:00+00:00'}},
            'result': {'probe': {'folds': [{'gain': .1} for _ in range(3)]},
                       'positive_control': {'relative_gain': .01}}}
    monkeypatch.setattr(launcher, 'verify_completed_nonlinear', lambda path: reports[path])
    return {'epoch': epoch, 'probes': ['0', '1']}, context, source, reports


def test_two_actual_prior_quarters_enable_only_resource_opportunity(case, monkeypatch):
    refs, context, source, _ = setup(case, monkeypatch)
    value = gates.verify_shared_gates(refs, case[1], case[2], context, source_hash=source)
    assert value['permission'] == 'controlled_resource_trial_only'
    assert value['research_enabled'] is False and value['serving_eligible'] is False
    with pytest.raises(ValueError, match='producer tickets'):
        gates.verify_shared_gates(refs, case[1], case[2], context, source_hash=source, require_resource=True)


@pytest.mark.parametrize('mutation', ['duplicate', 'future', 'pool', 'source', 'direction', 'control'])
def test_metadata_cannot_open_missing_or_cross_epoch_gates(case, monkeypatch, mutation):
    refs, context, source, reports = setup(case, monkeypatch)
    if mutation == 'duplicate':
        refs['probes'] = ['0', '0']
    elif mutation == 'future':
        reports['1']['publication']['observation']['wall'] = '2027-07-01T00:00:01+00:00'
    elif mutation == 'pool':
        reports['1']['ticket'].value['recipe']['pool']['version'] = 'other'
    elif mutation == 'source':
        reports['1']['ticket'].value['source_hash'] = 'a'*64
    elif mutation == 'direction':
        reports['1']['result']['probe']['folds'][1]['gain'] = -.01
    else:
        reports['1']['result']['positive_control']['relative_gain'] = 0
    with pytest.raises(ValueError):
        gates.verify_shared_gates(refs, case[1], case[2], context, source_hash=source)


def test_epoch_cannot_silently_recreate_missing_anchor(case, monkeypatch):
    bridge, data, recipe, _ = case
    reference = gates.freeze_shared_epoch(bridge, data, recipe)
    from pathlib import Path
    path = Path(reference['path'])
    path.rename(path.with_suffix('.retained'))
    with pytest.raises(ValueError, match='no regeneration'):
        gates.freeze_shared_epoch(bridge, data, recipe)


def test_changed_recipe_epoch_does_not_borrow_old_effects(case, monkeypatch):
    refs, context, source, _ = setup(case, monkeypatch)
    changed = copy.deepcopy(context)
    changed['source_id'] = 'changed-actual-source'
    with pytest.raises(ValueError, match='epoch differs'):
        gates.verify_shared_gates(refs, case[1], case[2], changed, source_hash=source)
