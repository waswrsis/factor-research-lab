"""Bounded policy replay: every search consumes only its earlier time block."""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .dataset import Dataset, make_dataset
from .evaluation import confirmation_gate, evaluate_predictions
from .graph import ExpertSpec, GraphSpec, mutate, parameter_count, to_dict
from .protocol import Protocol, fingerprint, project_path, save_json
from .research_agent import diagnostic_proposal, read_proposal
from .splits import outer_folds, segment_masks
from .statistics import paired_block_bootstrap, risk_metrics
from .training import fit_model


def load_market(protocol: Protocol):
    directory = project_path('data/prepared') / protocol.digest[:16]
    manifest = json.loads((directory / 'manifest.json').read_text('utf-8'))
    if manifest['protocol_hash'] != protocol.digest:
        raise ValueError('Data preparation protocol differs')
    frames = {}
    for kind in ('klines', 'mark', 'funding'):
        path = project_path(manifest['outputs'][kind]['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['outputs'][kind]['sha256']:
            raise ValueError('Prepared market data changed after audit')
        frames[kind] = pd.read_parquet(path)
    return frames, manifest


def starter_graph(data: Dataset, protocol: Protocol) -> GraphSpec:
    scales = tuple(sorted({x for x in (4, 12, protocol.lookback) if x <= protocol.lookback}))
    return GraphSpec(data.features,
                     (ExpertSpec('temporal', scales=scales),
                      ExpertSpec('interaction', kind='mlp', scales=scales)),
                     lookback=protocol.lookback, gating='dynamic')


def graph_id(spec: GraphSpec) -> str:
    return fingerprint(to_dict(spec))[:16]


def code_manifest() -> dict:
    return {str(path.relative_to(project_path('.'))): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(project_path('factorlab').glob('*.py'))}


def run_experiment(protocol: Protocol, output: str | Path, *, phases: str = 'all',
                   fold_limit: int | None = None, external_proposals: str | Path | None = None,
                   require_external: bool = False) -> dict:
    directory = project_path(output)
    if (directory / 'started.json').exists():
        raise ValueError('Experiment directory already used; preserve evidence with a new run name')
    protocol.freeze(directory)
    if require_external and not external_proposals:
        raise ValueError('Required external agent needs an explicit proposal directory')
    if fold_limit is not None and not 1 <= fold_limit <= len(protocol.test_starts):
        raise ValueError('Invalid fold limit')
    (directory / 'source').mkdir(parents=True, exist_ok=True)
    for source in project_path('factorlab').glob('*.py'):
        shutil.copy2(source, directory / 'source' / source.name)
    save_json(directory / 'started.json', {'protocol': protocol.digest, 'code': code_manifest(),
                                          'mode': phases, 'fold_limit': fold_limit,
                                          'started_at': pd.Timestamp.now(tz='UTC').isoformat()})
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    market, manifest = load_market(protocol)
    data = make_dataset(market['klines'], protocol)
    save_json(directory / 'dataset.json', {'shape': list(data.x.shape),
                                          'features': list(data.features), 'device': device,
                                          'source_manifest': manifest})
    base_spec = starter_graph(data, protocol)
    outcomes = []
    arms = ('A_ridge', 'B_fixed') if phases == 'p1' else (
        'A_ridge', 'B_fixed', 'C_random_search', 'D_diagnostic_agent')
    folds = outer_folds(protocol)
    if fold_limit is not None:
        folds = folds[:fold_limit]
    all_returns = {arm: [] for arm in arms}
    deployed_returns = {arm: [] for arm in arms}
    for fold in folds:
        print(f'{fold.name}: training cutoff {fold.validation_start}, test {fold.test_start}', flush=True)
        fd = directory / fold.name
        masks = segment_masks(data.meta, fold)
        segments = {name: data.subset(mask) for name, mask in masks.items()}
        boundaries = {'search': (fold.validation_start, fold.confirmation_start),
                      'confirmation': (fold.confirmation_start, fold.test_start),
                      'test': (fold.test_start, fold.test_end)}
        cutoff = (fold.confirmation_start-pd.Timedelta(nanoseconds=1)).isoformat()

        def evaluate(models, partition, multiplier=1.0, segments=segments, boundaries=boundaries):
            ds = segments[partition]
            prediction = np.mean([m.predict(ds, device) for m in models], axis=0)
            result = evaluate_predictions(ds, prediction, market['klines'], market['mark'],
                                          market['funding'], *boundaries[partition], protocol,
                                          cost_multiplier=multiplier)
            return result, prediction

        def train_candidate(spec, path, masks=masks, evaluate=evaluate):
            if parameter_count(spec) > protocol.max_parameters:
                raise ValueError('Candidate exceeds frozen parameter budget')
            models = [fit_model(data, masks['train'], protocol, spec, seed, device)
                      for seed in protocol.seeds]
            for seed, model in zip(protocol.seeds, models, strict=True):
                model.save(path / f'seed-{seed}')
            result, _ = evaluate(models, 'search')
            return models, result

        baseline = [fit_model(data, masks['train'], protocol, device=device)]
        baseline[0].save(fd / 'A_ridge' / 'model')
        fixed, fixed_search = train_candidate(base_spec, fd / 'B_fixed' / 'model')
        selected = {'A_ridge': (baseline, None), 'B_fixed': (fixed, base_spec)}
        records_by_arm = {}
        if phases != 'p1':
            for arm in ('C_random_search', 'D_diagnostic_agent'):
                population = [(fixed_search.metrics['utility_hourly'], base_spec, fixed)]
                records, seen = [], {graph_id(base_spec)}
                rng = random.Random(731)
                for trial in range(protocol.candidates_per_search):
                    started = time.monotonic()
                    parent = population[0][1] if arm.startswith('D_') else rng.choice(population)[1]
                    parent_id = graph_id(parent)
                    record = {'trial': trial, 'parent_id': parent_id, 'partition': 'search',
                              'evidence_cutoff': cutoff, 'status': 'failed'}
                    try:
                        if arm.startswith('D_'):
                            request = {'parent_id': parent_id, 'protocol_hash': protocol.digest,
                                       'evidence_cutoff': cutoff, 'parent': to_dict(parent),
                                       'history': records, 'trial': trial,
                                       'allowed_operations': ['add_expert','drop_expert','width','depth',
                                                              'scales','gate','loss','optimizer',
                                                              'learning_rate','drop_feature',
                                                              'add_interaction','drop_interaction']}
                            save_json(fd / arm / f'request-{trial}.json', request)
                            proposal_file = (project_path(external_proposals) / fold.name /
                                             f'proposal-{trial}.json') if external_proposals else None
                            if require_external:
                                print(f'AGENT_REQUEST {fd / arm / f"request-{trial}.json"}', flush=True)
                                deadline = time.monotonic() + 300
                                while not proposal_file.is_file() and time.monotonic() < deadline:
                                    time.sleep(1)
                                if not proposal_file.is_file():
                                    raise ValueError('External agent proposal timed out; no hidden fallback')
                            if proposal_file and proposal_file.is_file():
                                record['transport'] = {'kind': 'external_file',
                                    'path': str(proposal_file),
                                    'sha256': hashlib.sha256(proposal_file.read_bytes()).hexdigest()}
                                proposal = read_proposal(proposal_file, protocol, cutoff, parent_id)
                            else:
                                record['transport'] = {'kind': 'local_diagnostic_rules'}
                                proposal = diagnostic_proposal(parent_id, protocol, cutoff, records, trial)
                            proposal.validate(protocol, cutoff, parent_id)
                            record['proposal'] = asdict(proposal)
                            child = mutate(parent, {'operation': proposal.operation, **proposal.arguments}, proposal.seed,
                                           feature_pool=data.features)
                        else:
                            operations = ['width', 'depth', 'scales', 'gate', 'loss', 'optimizer',
                                          'learning_rate', 'add_expert', 'add_interaction', 'drop_feature']
                            if len(parent.experts) > 1:
                                operations.append('drop_expert')
                            if parent.feature_ops:
                                operations.append('drop_interaction')
                            operation = rng.choice(operations)
                            record['operation'] = operation
                            record['mutation_seed'] = rng.randrange(2**31)
                            child = mutate(parent, operation, record['mutation_seed'], feature_pool=data.features)
                        candidate_id = graph_id(child)
                        if candidate_id in seen:
                            raise ValueError('Duplicate graph proposal consumes attempt but is not retrained')
                        seen.add(candidate_id)
                        candidate_models, score = train_candidate(child, fd / arm / f'trial-{trial}')
                        value = score.metrics['utility_hourly']
                        record.update(status='evaluated', candidate_id=candidate_id, graph=to_dict(child),
                                      parameters=parameter_count(child), utility_hourly=value,
                                      metrics=score.metrics,
                                      alpha=float(np.mean([m.alpha for m in candidate_models])))
                        population.append((value, child, candidate_models))
                        population.sort(key=lambda x: x[0], reverse=True)
                        population = population[:3]
                    except (ValueError, RuntimeError, TypeError, KeyError) as error:
                        record['failure'] = f'{type(error).__name__}: {error}'
                    record['elapsed_seconds'] = time.monotonic()-started
                    records.append(record)
                    save_json(fd / arm / 'lineage.json', records)
                    print(f'{fold.name} {arm} trial {trial}: {record["status"]}', flush=True)
                selected[arm] = (population[0][2], population[0][1])
                records_by_arm[arm] = records
        reference_confirmation, _ = evaluate(baseline, 'confirmation')
        reference_test, _ = evaluate(baseline, 'test')
        fold_results = {'fold': fold.name, 'partitions': {k: int(v.sum()) for k, v in masks.items()},
                        'training_cutoff': fold.validation_start.isoformat(),
                        'test_start': fold.test_start.isoformat(), 'arms': {}}
        for arm, (models, spec) in selected.items():
            confirmation, _ = evaluate(models, 'confirmation')
            gate = confirmation_gate(confirmation, reference_confirmation, protocol)
            # Unpromoted candidate tests remain visible as research-policy evidence;
            # deployment retains baseline and is reported separately.
            test, predictions = evaluate(models, 'test')
            arm_dir = fd / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            test.equity.to_parquet(arm_dir / 'test_equity.parquet')
            test.trades.to_parquet(arm_dir / 'test_trades.parquet', index=False)
            deployed = test if gate['passed'] or arm == 'A_ridge' else reference_test
            deployed.equity.to_parquet(arm_dir / 'deployed_policy_equity.parquet')
            deployed_returns[arm].append(deployed.equity['return'])
            saved_predictions = segments['test'].meta.copy()
            saved_predictions['prediction'] = predictions
            saved_predictions.to_parquet(arm_dir / 'test_predictions.parquet', index=False)
            for i, fitted in enumerate(models):
                fitted.save(arm_dir / 'selected' / f'model-{i}')
            stresses = {str(multiplier): evaluate(models, 'test', multiplier)[0].metrics
                        for multiplier in protocol.stress_cost_multipliers if multiplier != 1}
            fold_results['arms'][arm] = {'graph': to_dict(spec) if spec else None,
                                        'selected_graph_id': graph_id(spec) if spec else 'ridge',
                                        'model_diagnostics': [m.diagnostics for m in models],
                                        'confirmation': gate, 'test': test.metrics,
                                        'deployed_policy': deployed.metrics,
                                        'cost_stresses': stresses, 'search_records': records_by_arm.get(arm, [])}
            all_returns[arm].append(test.equity['return'])
            print(f'{fold.name} {arm}: completed, confirmation={gate["decision"]}', flush=True)
        outcomes.append(fold_results)
        save_json(fd / 'results.json', fold_results)
    aggregate = {}
    reference = pd.concat(all_returns['A_ridge'])
    for arm, parts in all_returns.items():
        returns = pd.concat(parts)
        aggregate[arm] = {'risk': risk_metrics(returns),
                          'deployed_policy_risk': risk_metrics(pd.concat(deployed_returns[arm])),
                          'paired_vs_ridge': paired_block_bootstrap(returns, reference,
                              block_length=protocol.bootstrap_block_hours,
                              n_bootstrap=protocol.bootstrap_samples,
                              min_blocks=protocol.promotion_min_blocks,
                              risk_aversion=protocol.risk_aversion, seed=17)}
    result = {'status': 'completed', 'protocol_hash': protocol.digest, 'run_directory': str(directory),
              'folds': outcomes, 'aggregate': aggregate, 'device': device,
              'agent_sources': sorted({r.get('proposal', {}).get('source', 'unknown')
                  for f in outcomes for r in f['arms'].get('D_diagnostic_agent', {}).get('search_records', [])
                  if 'proposal' in r}),
              'limitations': ['Fixed three surviving assets; not a point-in-time market universe.',
                              '6-month engineering replay; no prospective performance evidence.',
                              'Candidate attempts, epochs and seeds matched; GPU seconds/parameters vary.',
                              'Seven-day confirmation has fewer than required 20 time blocks; promotion must fail.',
                              'Aggregate pairwise intervals are exploratory, not family-wise significance.',
                              'Rejected candidates are evaluated as research evidence only; no trading deployment.',
                              'LLM pretraining contamination cannot be ruled out for external historical proposals.']}
    save_json(directory / 'results.json', result)
    return result
