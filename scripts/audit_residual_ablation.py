"""Post-hoc diagnostic only: matched fitted baseline versus residual ensemble."""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from factorlab.dataset import make_dataset
from factorlab.evaluation import evaluate_predictions
from factorlab.experiment import load_market
from factorlab.protocol import Protocol, project_path, save_json
from factorlab.splits import outer_folds, segment_masks
from factorlab.statistics import paired_block_bootstrap, risk_metrics
from factorlab.training import FittedModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='runs/policy-replay-v1')
    args = parser.parse_args()
    run = project_path(args.run)
    protocol = Protocol.read(run / 'protocol.json')
    market, _ = load_market(protocol)
    dataset = make_dataset(market['klines'], protocol)
    baseline_parts, candidate_parts, folds = [], [], []
    for fold in outer_folds(protocol):
        test = dataset.subset(segment_masks(dataset.meta, fold)['test'])
        models = [FittedModel.load(run / fold.name / 'B_fixed' / 'selected' / f'model-{i}')
                  for i in range(len(protocol.seeds))]
        predictions = np.mean([replace(m, alpha=0).predict(test) for m in models], axis=0)
        result = evaluate_predictions(test, predictions, market['klines'], market['mark'],
                                      market['funding'], fold.test_start, fold.test_end, protocol)
        result.equity.to_parquet(run / fold.name / 'B_fixed' / 'matched_baseline_equity.parquet')
        actual = pd.read_parquet(run / fold.name / 'B_fixed' / 'test_equity.parquet')
        baseline_parts.append(result.equity['return'])
        candidate_parts.append(actual['return'])
        folds.append({'fold': fold.name, 'alphas': [m.alpha for m in models],
                      'matched_baseline': result.metrics})
    paired = paired_block_bootstrap(pd.concat(candidate_parts), pd.concat(baseline_parts),
                                    block_length=protocol.bootstrap_block_hours,
                                    n_bootstrap=protocol.bootstrap_samples,
                                    min_blocks=protocol.promotion_min_blocks,
                                    risk_aversion=protocol.risk_aversion)
    save_json('outputs/phase_audits/residual_ablation.json', {
        'status': 'post_hoc_diagnostic_not_used_for_selection', 'protocol_hash': protocol.digest,
        'question': 'Does the learned residual improve its own fitted baseline component?',
        'reason': 'A_ridge uses all training samples; expert models reserve 20% for chronological calibration.',
        'fitting_performed': False, 'promotion_decisions_changed': False,
        'folds': folds, 'baseline_component_risk': risk_metrics(pd.concat(baseline_parts)),
        'paired_fixed_vs_matched_baseline': paired,
        'limitation': 'Exploratory paired comparison after results; not preregistered alpha proof.'})


if __name__ == '__main__':
    main()
