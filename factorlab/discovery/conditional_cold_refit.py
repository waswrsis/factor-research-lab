"""Paid finite-census cold learner truth, conditional on original selection.

No provider/selection/activation is replayed. The estimand is the first original
selected confirmation month under deterministic computational availability.
It is NEVER the full adaptive A0 policy or an original five-claim certificate.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock
from threadpoolctl import threadpool_limits

from factorlab.protocol import fingerprint, project_path
from .evidence import file_hash
from .finite_market_oracle import FiniteGenerator, FiniteMarketLaw, _enclose_census, _finite_loss, _plain, _read, _write
from .full_calibration_registry import FullTrialRegistry, QuarantinedTrialPermit
from .monthly_partition_policy import counts
from .monthly_work_authority import source_files
from .neural_forward_ticket import open_existing_authority
from .reserved_work_budget import ReservedChannelBudget
from .work_lease import FrozenWorkLease, _evaluate, _observe

VERSION = 'v7-conditional-selected-cold-refit-census-v1'
SCOPE = 'conditional_original_selected_learner_first_confirmation_month_finite_population'


def _selected(permit, slot):
    """No caller recipes, losses, booleans, replacement winner or arbitrary window."""
    from .calibration_oracle_evidence import _original_specs
    permit.verify()
    if isinstance(slot, str) and slot.startswith('typed_neural_'):
        from .typed_cold_selection import select_typed_neural
        return select_typed_neural(permit, slot)
    if slot not in ('library_skill', 'factor1', 'factor2', 'factor3'):
        raise ValueError('Adaptive A0 is not a fixed cold learner; choose an original nonadaptive claim')
    spec = next(s for s in _original_specs(permit) if s.slot == slot)
    if spec.candidate is None:
        raise ValueError('Original nonselection cannot be replaced by a convenient learner')
    start = pd.Timestamp(spec.start)
    if start != start.normalize().replace(day=1):
        raise ValueError('Original learner confirmation must start at its cold UTC month cutoff')
    # An explicit conditional sub-estimand, not silently relabelled full C risk.
    end = min(pd.Timestamp(spec.end), start + pd.DateOffset(months=1))
    return {'original_estimand': spec.to_dict(), 'original_estimand_digest': spec.digest,
            'start': start.isoformat(), 'end': end.isoformat(),
            'reference': spec.reference, 'candidate': spec.candidate}


def _spec(identity):
    from .rolling import RollingSpec
    value = dict(identity['rolling']); value['event_bins'] = tuple(value['event_bins'])
    return RollingSpec(**value)


def _recipe(pipeline, spec):
    from .joint_pool import FrozenPool
    from .known_linear_reference import FrozenLinearRecipe
    from .neural_forward import FrozenNeuralRecipe
    from .v7_forward import ForwardRuntime
    if pipeline.get('kind') == 'v7_neural_forward':
        recipe = FrozenNeuralRecipe.from_dict(pipeline['recipe'])
    elif pipeline.get('kind') == 'v7_known_linear':
        recipe = FrozenLinearRecipe.from_dict(pipeline['recipe'])
    elif 'pool' in pipeline:
        recipe = FrozenPool.from_dict(pipeline['pool'])
    else:
        raise ValueError('Original selected learner lacks a supported full cold recipe')
    descriptor = ForwardRuntime.__new__(ForwardRuntime); descriptor.spec = spec
    if _plain(descriptor.recipe_for(recipe)) != pipeline:
        raise ValueError('Original selected learner differs from fixed production cold kernel')
    return recipe


def _limits(identity):
    from .joint_pool import FrozenPool
    from .known_linear_reference import FrozenLinearRecipe
    from .pool_forward import PoolForwardRecipe
    result = counts()
    for key in ('reference', 'candidate'):
        recipe = _recipe(identity['selection'][key], _spec(identity))
        if isinstance(recipe, FrozenPool):
            debit = {'solve': PoolForwardRecipe(recipe, _spec(identity)).solve_limit}
        elif isinstance(recipe, FrozenLinearRecipe):
            debit = {'solve': 1}
        else:
            debit = recipe.budget_limits
        for kind in result:
            result[kind] += debit.get(kind, 0)
    return result


def _open_generator(identity, registry):
    path = project_path(identity['generator_directory'])
    saved = _read(path/'identity.json')
    law = dict(saved['law']); law['symbols'] = tuple(law['symbols'])
    generator = FiniteGenerator(path, FiniteMarketLaw(**law), registry.plan.acceptance)
    if generator.digest != identity['generator_digest']:
        raise ValueError('Cold oracle original generator identity changed')
    generator.verify_registry(registry)
    if sorted(generator.law.symbols) != sorted(registry.plan.runtime_scope['forward']['symbols']):
        raise ValueError('Cold oracle generator changed the original runtime universe')
    return generator


def _load(directory):
    directory = project_path(directory)
    identity = _read(directory/'identity.json')
    registry = FullTrialRegistry.inspect(identity['registry_directory'])
    trial = registry.identity_for(identity['validation_trial']['cell'], identity['validation_trial']['index'])
    permit = QuarantinedTrialPermit(registry, trial)
    if (identity['version'] != VERSION or identity['directory'] != str(directory)
            or identity['registry_digest'] != registry.digest or identity['validation_trial'] != trial
            or identity['source_files'] != source_files() or identity['implementation_hash'] != file_hash(__file__)
            or identity['selection'] != _selected(permit, identity['slot'])
            or identity['rolling'] != registry.plan.runtime_scope['rolling']
            or _read(permit.paths['root']/'cold_refit'/identity['slot']/'identity.json') != identity):
        raise ValueError('Cold oracle original registry/selection/source changed')
    generator = _open_generator(identity, registry)
    if (identity['support'] != list(range(generator.law.support_size))
            or generator.law.support_size != registry.plan.oracle_paths
            or generator.law.support_size < 128 or identity['counts'] != _limits(identity)):
        raise ValueError('Cold oracle original complete support or exact fitting costs changed')
    return identity, registry, generator


def freeze_cold_refit(validation_permit, generator, slot):
    if type(validation_permit) is not QuarantinedTrialPermit or type(generator) is not FiniteGenerator:
        raise TypeError('Actual original trial and registered finite generator required')
    permit = validation_permit; permit.verify(); generator.verify_registry(permit.registry)
    if (permit.trial['role'] != 'validation' or permit.trial['scenario_digest'] != generator.law.digest
            or generator.law.support_size != permit.registry.plan.oracle_paths or generator.law.support_size < 128
            or sorted(generator.law.symbols) != sorted(permit.registry.plan.runtime_scope['forward']['symbols'])):
        raise ValueError('Cold oracle requires all original 128 or more path atoms')
    selection = _selected(permit, slot)
    directory = permit.registry.directory/'conditional_cold_refits'/permit.trial['trial_id']/slot
    local = permit.paths['root']/'cold_refit'/slot/'identity.json'
    identity = {'version': VERSION, 'directory': str(directory), 'registry_directory': str(permit.registry.directory),
        'registry_digest': permit.registry.digest, 'validation_trial': permit.trial,
        'generator_directory': str(generator.directory), 'generator_digest': generator.digest,
        'slot': slot, 'selection': selection, 'rolling': permit.registry.plan.runtime_scope['rolling'],
        'source_files': source_files(), 'implementation_hash': file_hash(__file__),
        'support': list(range(generator.law.support_size)), 'scope': SCOPE,
        'availability': 'assumed_original_runtime_close_plus_delay_computational_only',
        'activation': 'counterfactual_cold_model_available_at_logical_cutoff_no_service_delay_or_A0_adoption',
        'risk': 'original_unclipped_logMSE_equal_symbol_within_hour_then_equal_hour_uniform_paths',
        'risk_hour': 'floor_of_assumed_available_at_matching_original_forecast_origin_hour',
        'path_missing_policy': 'any_missing_failed_or_undefined_atom_prevents_envelope',
        'selection_feedback': 'none_oracle_only_never_GPT_or_model_selection',
        'full_adaptive_A0_truth': False, 'original_claim_scope_eligible': False,
        'formal_gate_eligible': False, 'lease_seconds': 300}
    identity['counts'] = _limits(identity)
    if (pd.Timestamp(selection['start']) < pd.Timestamp(generator.law.start)+pd.Timedelta(hours=1234)
            or pd.Timestamp(selection['end'])+pd.Timedelta(hours=14) >
                pd.Timestamp(generator.law.start)+pd.Timedelta(hours=generator.law.hours)):
        raise ValueError('Generator does not cover original cold training warmup and entire scoring month')
    with FileLock(str(directory)+'.freeze.lock', timeout=0):
        if directory.exists() or local.exists():
            old, _, _ = _load(directory)
            if old != identity:
                raise ValueError('Original cold oracle cannot replace its selection or denominator')
            return {'directory': str(directory), 'digest': fingerprint(old), 'scope': SCOPE}
        for path in permit.registry.directory.glob('trials/*/enrollment.json'):
            value = json.loads(path.read_text('utf-8'))['intent']
            if value['role'] == 'oracle' and value['population_trial_id'] == permit.trial['population_trial_id']:
                raise ValueError('Cold selection must freeze before original oracle path enrollment')
        _write(directory/'identity.json', identity)
        _write(local, identity)
    return {'directory': str(directory), 'digest': fingerprint(identity), 'scope': SCOPE}


def validate_allocation(authority, key, binding, lease, debit):
    """Canonical monthly route; no current-wallet recursion or numerical replay."""
    from .monthly_partition_policy import _runtime
    intent = _read(authority.directory/'cold_refit_intents'/(fingerprint(key)+'.json'))
    identity, registry, _ = _load(intent['oracle_directory'])
    trial = registry.identity_for(identity['validation_trial']['cell'], identity['validation_trial']['index'],
                                  oracle_index=intent['atom'])
    paths = registry.paths(trial['cell'], trial['index'], oracle_index=trial['oracle_index'])
    unit = project_path(identity['directory'])/'paths'/str(intent['atom'])
    enrollment = json.loads((paths['root']/'enrollment.json').read_text('utf-8'))
    original = json.loads((paths['root']/'enrollment_intent.json').read_text('utf-8'))
    if (enrollment['intent'] != original or any(original[k] != v for k, v in trial.items())
            or intent != _read(unit/'intent.json') or intent['identity_hash'] != fingerprint(identity)
            or intent['oracle_trial'] != trial or intent['authority_manifest_hash'] != authority.manifest_hash
            or authority.directory != paths['authority'] or authority.manifest['partition_policy']['mode'] != 'standard'
            or binding != {'role': 'conditional_cold_refit', 'oracle_directory': identity['directory'],
                           'atom': intent['atom'], 'intent_hash': fingerprint(intent)}
            or key != 'conditional_cold_refit:'+fingerprint({'oracle': fingerprint(identity), 'atom': intent['atom']})
            or intent['lease'] != lease.to_dict() or lease.lease_seconds != 300 or debit != identity['counts']):
        raise ValueError('Cold oracle changed original trial/shared wallet/allocation')
    owner = _runtime(authority, paths['runtime'])
    from .calibration_runtime_contract import _original_operations
    from .full_calibration_registry import EVENT
    from .neural_forward_ticket import _LifecycleReader
    reader = _LifecycleReader(paths['runtime'])
    try:
        events = [e for e in _original_operations(reader) if e['kind'] == EVENT]
    finally:
        reader.close()
    if (len(events) != 1 or events[0]['detail'] != original
            or enrollment['original_event_hash'] != fingerprint(events[0])):
        raise ValueError('Cold oracle lacks its original actual trial enrollment event')
    return {'lane': 'forward', 'owner': owner}


def _data(generator, atom, spec):
    from .rolling import build_rolling_data
    value = generator.read_atom(atom)
    gaps = {s['hour'] for s in value['schedule'] if s['gap']}
    n = len(generator.law.symbols)
    frame = pd.DataFrame([r for i, r in enumerate(value['rows']) if i//n not in gaps])
    frame['open_time'] = pd.to_datetime(frame.open_time, utc=True)
    frame['available_at'] = frame.open_time+pd.Timedelta(hours=1, seconds=spec.availability_delay_seconds)
    return build_rolling_data(frame, spec), fingerprint(value)


def _fit(data, cutoff, recipe, budget):
    from .joint_pool import FrozenPool, fit_pool
    from .known_linear_reference import FrozenLinearRecipe, fit_known_linear
    from .neural_forward import _fit_neural_forward_payload
    if isinstance(recipe, FrozenPool):
        return fit_pool(data, cutoff, recipe, budget)
    if isinstance(recipe, FrozenLinearRecipe):
        return fit_known_linear(data, cutoff, recipe, budget=budget)
    return _fit_neural_forward_payload(data, cutoff, recipe, budget)


def _model(path, pipeline):
    from .joint_pool import JointPredictor
    from .known_linear_reference import KnownLinearPredictor
    from .neural_forward import NeuralForwardModel
    cls = NeuralForwardModel if pipeline.get('kind') == 'v7_neural_forward' else (
        KnownLinearPredictor if pipeline.get('kind') == 'v7_known_linear' else JointPredictor)
    model = cls.load(path)
    if model.pipeline_id != pipeline['pipeline_id']:
        raise ValueError('Cold model differs from original selected complete recipe')
    return model


def _risk(identity, generator, atom, data, models):
    import math
    start, end = pd.Timestamp(identity['selection']['start']), pd.Timestamp(identity['selection']['end'])
    # Original lifecycle losses index the prediction origin hour, not the last
    # candle's open hour (which is one hour earlier under close-plus-delay).
    origin = pd.to_datetime(data.frame.metadata.available_at, utc=True).dt.floor('h')
    expected = {(t.isoformat(), s) for t in pd.date_range(start, end, freq='h', inclusive='left')
                for s in generator.law.symbols}
    predictions = [m.predict_history(data.frame) for m in models]
    mask = ((origin >= start) & (origin < end)).to_numpy() & data.observed & predictions[0][1] & predictions[1][1]
    actual = {(origin.iloc[i].isoformat(), data.frame.metadata.symbol.iloc[i]) for i in np.flatnonzero(mask)}
    result = {'atom': atom, 'defined': actual == expected, 'expected_pairs': len(expected),
              'actual_pairs': len(actual), 'reference': None, 'candidate': None}
    if actual != expected:
        return {**result, 'reason': 'missing_original_common_support'}
    hourly = {}
    for i in np.flatnonzero(mask):
        hourly.setdefault(origin.iloc[i].isoformat(), []).append([
            _finite_loss(float(data.target[i]), float(p[0][i])) for p in predictions])
    risks = [math.fsum(math.fsum(p[k] for p in rows)/len(rows) for rows in hourly.values())/len(hourly)
             for k in range(2)]
    return {**result, 'reference': risks[0], 'candidate': risks[1], 'support_hash': fingerprint(sorted(actual))}


def _unit(directory, atom):
    identity, registry, generator = _load(directory)
    if type(atom) is not int or atom not in identity['support']:
        raise ValueError('Oracle atom outside complete original support')
    trial = registry.identity_for(identity['validation_trial']['cell'], identity['validation_trial']['index'], oracle_index=atom)
    permit = QuarantinedTrialPermit(registry, trial)
    return identity, registry, generator, permit, project_path(directory)/'paths'/str(atom)


def worker(directory, atom):
    from .neural_research_launcher import _identity, _kernel_snapshot
    identity, _, generator, permit, unit = _unit(directory, atom)
    intent = _read(unit/'intent.json'); lease = FrozenWorkLease.from_dict(intent['lease'])
    with lease.as_worker():
        while not (unit/'start.json').exists():
            lease.assert_parent_alive(as_worker=True); time.sleep(.01)
        permit.verify()
        authority = open_existing_authority(permit.paths['authority'])
        rid = fingerprint({'authority': authority.manifest_hash, 'work_key': intent['work_key']})
        if _read(unit/'start.json') != {'process': _identity(), 'reservation': rid, 'lease_hash': lease.digest}:
            raise ValueError('Cold child handshake changed')
        budget = ReservedChannelBudget.from_reservation(authority, rid)
        _write(unit/'membership.json', {'process': _identity(), 'kernel': _kernel_snapshot()})
        generator.produce_atom(atom)
        data, atom_hash = _data(generator, atom, _spec(identity))
        models = []
        with threadpool_limits(limits=1):
            for name in ('reference', 'candidate'):
                recipe = _recipe(identity['selection'][name], _spec(identity))
                model = _fit(data, identity['selection']['start'], recipe, budget)
                model.save(unit/(name+'.json'))
                models.append(_model(unit/(name+'.json'), identity['selection'][name]))
            result = _risk(identity, generator, atom, data, models)
        budget.check_time(); budget.finalize()
        _write(unit/'worker_result.json', {'risk': result, 'market_atom_hash': atom_hash,
            'counts': budget.actual_counts, 'kernel': _kernel_snapshot(), 'process': _identity()})
        lease.assert_parent_alive(as_worker=True)


def _verify_final(identity, authority, row, unit):
    final = _read(unit/'final_io.json')
    lease = FrozenWorkLease.from_dict(row['lease'])
    result = _read(unit/'result.json')
    if (row['status'] != 'completed' or row['result'] != {'cold_refit_result_hash': fingerprint(result)}
            or final['reservation_hash'] != fingerprint(row) or final['result_hash'] != fingerprint(result)
            or final['identity_hash'] != fingerprint(identity) or final['lease_hash'] != lease.digest
            or not _evaluate(lease, final['observation']).allowed):
        raise ValueError('Original cold-refit completion/publication changed or expired')
    actual = ReservedChannelBudget.verify_counts(authority, row['reservation_id'])
    if (not actual['finalized'] or actual['test_only'] or actual['actual_counts'] != row['actual_counts']
            or actual['actual_counts'] != result['actual_counts']):
        raise ValueError('Original cold-refit actual prepaid counts changed')
    for name, digest in result['artifacts'].items():
        if file_hash(unit/name) != digest:
            raise ValueError('Original cold-refit numeric artifact changed')
    deadline = pd.Timestamp(lease.wall_deadline).value
    for path in [unit/'final_io.json', unit/'result.json', *(unit/name for name in result['artifacts'])]:
        if path.is_symlink() or max(path.stat().st_mtime_ns, path.stat().st_ctime_ns) >= deadline:
            raise ValueError('Original cold-refit final IO passed original deadline')
    return result


def run_cold_refit(directory, atom):
    """One existing original oracle path, once, through its real shared wallet."""
    from .neural_research_launcher import (
        _environment, _identity, _kernel_snapshot, _member, _strict_kernel, _validate_kernel,
        isolated_process_kwargs, terminate_process_tree,
    )
    identity, _, generator, permit, unit = _unit(directory, atom)
    permit.verify()
    authority = open_existing_authority(permit.paths['authority'])
    key = 'conditional_cold_refit:'+fingerprint({'oracle': fingerprint(identity), 'atom': atom})
    rid = fingerprint({'authority': authority.manifest_hash, 'work_key': key})
    anchor = authority.directory/'cold_refit_intents'/(fingerprint(key)+'.json')
    with FileLock(str(unit)+'.lock', timeout=0):
        if unit.exists() or anchor.exists():
            intent = _read(unit/'intent.json'); row = authority.get(rid)
            if intent != _read(anchor):
                raise ValueError('Cold refit original paid intent changed')
            if row['status'] != 'completed' or (unit/'failure.json').exists():
                raise ValueError('Cold oracle original path already consumed; no retry or refund')
            return _verify_final(identity, authority, row, unit)
        lease = FrozenWorkLease.issue(300)
        intent = {'version': VERSION, 'oracle_directory': identity['directory'], 'atom': atom,
            'identity_hash': fingerprint(identity), 'oracle_trial': permit.trial, 'work_key': key,
            'authority_manifest_hash': authority.manifest_hash, 'lease': lease.to_dict()}
        _write(unit/'intent.json', intent); _write(anchor, intent)
        binding = {'role': 'conditional_cold_refit', 'oracle_directory': identity['directory'],
                   'atom': atom, 'intent_hash': fingerprint(intent)}
        try:
            row = authority._reserve_with_lease(key, counts=identity['counts'], binding=binding, lease=lease)
        except BaseException as error:
            _write(unit/'reservation_failure.json', {'work_key': key, 'reason': type(error).__name__})
            raise
        process = None
        try:
            first = _kernel_snapshot(); _validate_kernel(first, first); _member(first, _identity())
            command = [sys.executable, '-B', '-m', 'factorlab.discovery.conditional_cold_refit',
                       '--worker', identity['directory'], str(atom)]
            with (unit/'worker.log').open('xb') as log:
                process = subprocess.Popen(command, cwd=str(Path(__file__).resolve().parents[2]),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    env=_environment(), **isolated_process_kwargs())
                child = _identity(process.pid)
                row = authority.start(rid, child)
                _write(unit/'start.json', {'process': child, 'reservation': rid, 'lease_hash': lease.digest})
                while process.poll() is None:
                    lease.assert_parent_alive()
                    if log.tell() > 2**20:
                        raise ValueError('Cold oracle worker output exceeded bound')
                    time.sleep(.05)
                if process.returncode != 0:
                    raise RuntimeError('Cold oracle original worker failed: '+str(process.returncode))
            last = _kernel_snapshot(); membership = _read(unit/'membership.json')
            output = _read(unit/'worker_result.json')
            parent = {'pid': lease.parent_pid, 'created': lease.parent_created}
            for sample in (membership, output):
                if sample['process'] != child:
                    raise ValueError('Cold oracle replaced its original worker')
                _member(sample['kernel'], parent, child)
                _validate_kernel(first, sample['kernel']); _validate_kernel(sample['kernel'], last)
                _strict_kernel(first, sample['kernel'], last, lease, parent, child)
            data, atom_hash = _data(generator, atom, _spec(identity))
            with threadpool_limits(limits=1):
                models = [_model(unit/(n+'.json'), identity['selection'][n]) for n in ('reference', 'candidate')]
                risk = _risk(identity, generator, atom, data, models)
            if output['risk'] != risk or output['market_atom_hash'] != atom_hash:
                raise ValueError('Cold child losses differ from paid original numerical replay')
            final_kernel = _kernel_snapshot()
            _validate_kernel(last, final_kernel); _member(final_kernel, parent)
            for sample in (membership, output):
                _strict_kernel(first, sample['kernel'], final_kernel, lease, parent, child)
            actual = ReservedChannelBudget.verify_counts(authority, rid)
            if actual['test_only'] or not actual['finalized'] or actual['actual_counts'] != output['counts']:
                raise ValueError('Cold oracle incomplete actual paid worker')
            result = {'identity_hash': fingerprint(identity), 'atom': atom, 'risk': risk,
                'market_atom_hash': atom_hash, 'actual_counts': actual['actual_counts'],
                'kernel_before': first, 'kernel_after': final_kernel,
                'artifacts': {n: file_hash(unit/n) for n in ('reference.json', 'candidate.json', 'membership.json',
                    'worker_result.json', 'start.json', 'intent.json')}, 'scope': SCOPE,
                'original_claim_scope_eligible': False, 'formal_gate_eligible': False}
            lease.assert_parent_alive(); _write(unit/'result.json', result)
            row = authority.complete(rid, result={'cold_refit_result_hash': fingerprint(result)}, actual_counts=actual['actual_counts'])
            lease.assert_parent_alive()
            _write(unit/'final_io.json', {'identity_hash': fingerprint(identity), 'reservation_hash': fingerprint(row),
                'result_hash': fingerprint(result), 'lease_hash': lease.digest, 'observation': _observe(lease)})
            verified = _verify_final(identity, authority, row, unit)
            lease.assert_parent_alive()
            return verified
        except BaseException as error:
            try:
                _write(unit/'failure.json', {'reservation': rid, 'reason': type(error).__name__})
                if authority.get(rid)['status'] in {'reserved', 'running'}:
                    authority.fail_consumed(rid, 'conditional_cold_refit:'+type(error).__name__)
            except BaseException as persistence_error:
                error.add_note('Original consumed closure unavailable: '+repr(persistence_error))
            raise
        finally:
            if process is not None and process.poll() is None:
                terminate_process_tree(process)


def report_cold_refit(directory):
    identity, registry, generator = _load(directory)
    results, missing, failed = [], [], []
    for atom in identity['support']:
        unit = project_path(directory)/'paths'/str(atom)
        if not unit.exists():
            missing.append(atom); continue
        trial = registry.identity_for(identity['validation_trial']['cell'], identity['validation_trial']['index'], oracle_index=atom)
        paths = registry.paths(trial['cell'], trial['index'], oracle_index=atom)
        authority = open_existing_authority(paths['authority'])
        key = 'conditional_cold_refit:'+fingerprint({'oracle': fingerprint(identity), 'atom': atom})
        rid = fingerprint({'authority': authority.manifest_hash, 'work_key': key})
        with authority._transaction() as db:
            row = authority._verify(db)['reservations'].get(rid)
        if row is None:
            failure = _read(unit/'reservation_failure.json')
            if failure['work_key'] != key:
                raise ValueError('Original cold path reservation failure changed')
            failed.append(atom); continue
        if row['status'] != 'completed' or (unit/'failure.json').exists():
            failed.append(atom); continue
        result = _verify_final(identity, authority, row, unit)
        results.append(result)
        if not result['risk']['defined']:
            failed.append(atom)
    digest = fingerprint({'identity': identity, 'results': results, 'missing': missing, 'failed': failed})
    envelope = None
    if not missing and not failed:
        envelope = _enclose_census([{k: r['risk'][k] for k in ('atom', 'reference', 'candidate', 'defined')}
                                  for r in results], generator=generator, oracle_digest=digest)
    return {'version': VERSION, 'identity_hash': fingerprint(identity), 'evidence_digest': digest,
        'scope': SCOPE, 'registered_paths': len(identity['support']), 'completed_paths': len(results),
        'selection': identity['selection'], 'risk': identity['risk'], 'risk_hour': identity['risk_hour'],
        'availability': identity['availability'], 'activation': identity['activation'],
        'missing_paths': missing, 'failed_or_undefined_paths': failed,
        'conditional_risk_envelope': asdict(envelope) if envelope else None,
        'conditional_truth_verified': envelope is not None, 'original_claim_scope_eligible': False,
        'full_adaptive_A0_truth': False, 'formal_gate_eligible': False}


def read_conditional_cold_refit(registry, cell, index):
    if type(registry) is not FullTrialRegistry:
        raise TypeError('Actual original full registry required')
    registry.verify(); trial = registry.identity_for(cell, index)
    root = registry.directory/'conditional_cold_refits'/trial['trial_id']
    return [report_cold_refit(path) for path in sorted(root.iterdir()) if path.is_dir()] if root.exists() else []


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] != '--worker':
        raise SystemExit('Only the fixed original cold oracle worker entry point is supported')
    worker(sys.argv[2], int(sys.argv[3]))
