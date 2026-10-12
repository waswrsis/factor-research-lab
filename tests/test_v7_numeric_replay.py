"""Float64 replay is not a scientific selection tolerance."""
import copy
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from factorlab.discovery import neural_research_launcher as launcher
from factorlab.discovery.algorithm_shortlist import AlgorithmShortlistRule, _score
from factorlab.discovery.evidence import file_hash
from factorlab.discovery.neural_forward import FrozenNeuralRecipe
from factorlab.discovery.numeric_replay import RULE, assert_replayed
from factorlab.discovery.rolling import development_folds
from factorlab.protocol import fingerprint


def test_float64_one_ULP_is_roundoff_but_not_exact_identity():
    value = {'losses': [1., math.nextafter(1., 2.)], 'epoch': 1, 'serving': False}
    expected = {'losses': [1., 1.], 'epoch': 1, 'serving': False}
    assert_replayed(value, expected, roundoff=True)
    with pytest.raises(ValueError):
        assert_replayed(value, expected)
    assert RULE['epsilon_multiplier'] == 64


@pytest.mark.parametrize(('actual', 'expected'), [
    (float('nan'), float('nan')), (float('inf'), float('inf')),
    (-float('inf'), -float('inf')), (1.+1e-10, 1.),
    (True, 1), (False, 0.), (1., 1), (2, 1),
    ([1.], [1., 1.]), ((1.,), [1.]), ({'x': 1.}, {'y': 1.}),
    ({'job': 'new'}, {'job': 'old'}), (np.float32(1.), 1.),
    ({'epoch': 2}, {'epoch': 1}), ({'support': 1.}, {'support': 1}),
    ({'sample_ids': ['a', 'b']}, {'sample_ids': ['b', 'a']}),
])
def test_roundoff_never_accepts_changed_shape_dtype_identity_nonfinite(actual, expected):
    with pytest.raises(ValueError):
        assert_replayed(actual, expected, roundoff=True)


def test_parent_scientific_scores_cannot_borrow_child_roundoff_gain(tmp_path, monkeypatch):
    """Private numerical reconstruction fixture, not a production completion.

    The actual child/weights integration is independently exercised by the S
    launcher and >=240-row typed-validation tests. Here the authoritative parent
    scorer is fixed exactly at zero gain while child bytes show +one ULP gain.
    """
    recipe = FrozenNeuralRecipe()
    cutoff = '2025-04-01T00:00:00+00:00'
    expected = list(development_folds(cutoff))
    directory = tmp_path / 'launch/research'
    directory.mkdir(parents=True)
    ticket = SimpleNamespace(path=tmp_path/'ticket.json', value={'context': {
        'input_id': 'artificial-input', 'cutoff': cutoff, 'symbols': ['X']}})
    manifest = {'version': 'v7-complete-neural-S-research-v1', 'role': 'S', 'cutoff': cutoff,
        'recipe': json.loads(json.dumps(recipe.to_dict())), 'pipeline_id': recipe.pipeline_id,
        'symbols': ['X'], 'input_id': 'artificial-input', 'folds': expected,
        'test_only': False, 'automatic_adoption': False, 'formal_claim': False}
    authoritative, folds = {}, []
    for index, fold in enumerate(expected):
        import pandas as pd
        hours = int((pd.Timestamp(fold['score_end'])-pd.Timestamp(fold['fit_cutoff']))/pd.Timedelta(hours=1))
        losses = {'ensemble': 1., 'K0': 1., 'Pv': 1.}
        score = {'role': 'S', 'nominal_hours': hours, 'nominal_rows': hours,
            'support': 1, 'coverage': 1/hours, 'common_sample_ids': [f's{index}'],
            'seed_checkpoints': [{'seed': seed, 'epoch': 1} for seed in recipe.config.seeds],
            'hourly': [{'hour': fold['fit_cutoff'], 'symbols_present': 1, 'losses': losses}],
            'predictions': {'ensemble': [1.], 'K0': [1.], 'Pv': [1.]}, 'targets': [1.],
            'losses': losses, 'mean_seed_loss_diagnostic': 1.}
        authoritative[fold['fit_cutoff']] = copy.deepcopy(score)
        raw = copy.deepcopy(score)
        raw['losses']['ensemble'] = math.nextafter(1., 0.)
        raw['hourly'][0]['losses']['ensemble'] = math.nextafter(1., 0.)
        model_path, k0_path = directory/f'fold-{index}-model.json', directory/f'fold-{index}-k0.json'
        model_path.write_text('{}')
        k0_path.write_text('{}')
        folds.append({**fold, **raw, 'status': 'completed', 'budget_counts': {},
            'model': {'path': str(model_path), 'sha256': file_hash(model_path)},
            'k0': {'path': str(k0_path), 'sha256': file_hash(k0_path)}})
    nominal = sum(f['nominal_hours'] for f in folds)
    raw_losses = {'ensemble': math.nextafter(1., 0.), 'K0': 1., 'Pv': 1.}
    raw = {**manifest, 'manifest_id': fingerprint(manifest), 'folds': folds, 'status': 'completed',
        'support': 3, 'nominal_rows': nominal, 'nominal_hours': nominal, 'supported_hours': 3,
        'coverage': 3/nominal, 'losses': raw_losses, 'conditional_losses': raw_losses}
    (directory/'manifest.json').write_text(json.dumps(manifest))
    result_path = directory/'result.json'
    result_path.write_text(json.dumps(raw))
    original_bytes = result_path.read_bytes()
    monkeypatch.setattr(launcher.NeuralForwardModel, 'load', lambda *_: SimpleNamespace(
        recipe=recipe, receipt={'payload_origin': 'isolated_research_worker'}))
    monkeypatch.setattr(launcher, '_joint', lambda *_: None)
    monkeypatch.setattr(launcher, 'score_research_fold', lambda *a, start, **k: copy.deepcopy(authoritative[start]))
    returned, reference = launcher._research_result(ticket, None, recipe)
    rule = AlgorithmShortlistRule(minimum_hour_coverage=.001, minimum_symbol_hour_coverage=.001)
    assert _score(raw, recipe, rule)['eligible']
    assert not _score(returned, recipe, rule)['eligible']
    assert returned['losses']['ensemble'] == returned['losses']['Pv'] == 1.
    assert reference['digest'] == fingerprint(raw)
    assert result_path.read_bytes() == original_bytes
    assert returned['numeric_replay'] == RULE
    raw['losses']['ensemble'] -= 1e-6
    result_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='roundoff'):
        launcher._research_result(ticket, None, recipe)
