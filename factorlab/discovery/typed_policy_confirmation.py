"""One whole-policy confirmation, never a sixth or per-neural claim.

The candidate policy is measured prospectively on its own forecast stream. Its
descriptive internal selectors do not authorize A0. Only the original six-month
algorithm result can authorize this policy at a later monthly cutoff.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager

from filelock import FileLock

from factorlab.protocol import fingerprint, project_path

from .v7_lifecycle import stamp
from .v7_storage import immutable_record

TRIAL_STREAM = 'A0_policy_trial'
POLICY = {'version': 'v7-adaptive-algorithm-v2',
    'selection': 'latest_visible_factor_C_or_typed_V_qualifier_then_identity_tiebreak',
    'refit': 'cold_three_calendar_months', 'no_selection': 'initial_K0',
    'integrity_failure': 'fallback_K0', 'development_retirement': 'does_not_change_A0',
    'evaluation_stream': TRIAL_STREAM,
    'adoption': 'same_algorithm_slot_six_month_supported_then_next_month_policy',
    'typed_qualifier': 'descriptive_selection_only_no_extra_claim_or_alpha'}

ELAPSED_SCHEDULE = {'training_days': 90, 'refit_days': 15,
                    'anchor': 'original_actual_data_ready_time'}


def policy_definition(reference, *, research_spec_hash=None):
    """Source-frozen policy version from the complete benchmark recipe."""
    if research_spec_hash is not None:
        if (type(research_spec_hash) is not str or len(research_spec_hash) != 64
                or any(c not in '0123456789abcdef' for c in research_spec_hash)
                or reference.get('fit_schedule') != ELAPSED_SCHEDULE):
            raise ValueError('Layered policy needs its immutable elapsed research enrollment')
        return {**policy_definition(reference), 'version': 'v7-layered-elapsed-research-policy-v4',
                'research_spec_hash': research_spec_hash,
                'adoption': 'prospective_unconfirmed_formula_or_typed_selector_retention',
                'statistical_status': 'descriptive_one_endpoint_no_population_power_certificate'}
    if 'fit_schedule' not in reference:
        return dict(POLICY)
    if reference['fit_schedule'] != ELAPSED_SCHEDULE:
        raise ValueError('Unregistered algorithm fit schedule')
    return {**POLICY, 'version': 'v7-adaptive-elapsed90-refit15-v3',
            'refit': 'cold_continuous_90_days_on_original_15_day_grid',
            'adoption': 'same_algorithm_slot_six_month_supported_then_next_original_fit_cutoff'}


@contextmanager
def _original_reader(lifecycle):
    from .neural_forward_ticket import _LifecycleReader
    from .v7_lifecycle import Lifecycle
    if not isinstance(lifecycle, Lifecycle):
        raise TypeError('Actual original lifecycle required')
    if type(lifecycle) is _LifecycleReader:
        lifecycle.verify()
        yield lifecycle
        return
    reader = _LifecycleReader(lifecycle.directory)
    reader.clock = lifecycle.clock
    try:
        if reader.plan != lifecycle.plan:
            raise ValueError('Policy lifecycle differs from its original plan')
        reader.verify()  # Original chain/index/exposure, without model re-admission.
        yield reader
        reader.verify()
    finally:
        reader.close()


def _protocol(lifecycle):
    from .claim_protocol import ClaimProtocol
    protocol = ClaimProtocol(lifecycle)
    endpoint = protocol.endpoints['algorithm']
    from .layered_research import verified_research_spec
    research = verified_research_spec(lifecycle)
    if endpoint.research_spec_hash != (None if research is None else research.digest):
        raise ValueError('Whole-policy endpoint differs from original research enrollment')
    policy = policy_definition(endpoint.reference, research_spec_hash=endpoint.research_spec_hash)
    if ({k: endpoint.candidate.get(k) for k in policy} != policy
            or endpoint.candidate_stream != TRIAL_STREAM):
        raise ValueError('Typed policy requires its newly frozen v2 whole-policy endpoint')
    elapsed = lifecycle.plan.training_window_days is not None
    if elapsed != ('fit_schedule' in endpoint.reference):
        raise ValueError('Claim endpoint changed the original lifecycle fit cadence')
    return protocol


def _physical_cutoff(lifecycle, cutoff):
    from .quarantined_observation import QuarantinedObservation, current_binding
    current = current_binding(lifecycle)
    if current is None:
        return cutoff
    observer = QuarantinedObservation.from_lifecycle(lifecycle)
    from .calibration_runtime_contract import _original_operations
    from .quarantined_observation import ADVANCED
    advances = [o for o in _original_operations(lifecycle) if o['kind'] == ADVANCED]
    # The FIRST original prefix crossing the cutoff; the current prefix cannot
    # make a late publication appear to have existed at the old month boundary.
    for hour in range(current['hour'] + 1):
        event = json.loads((observer.directory/'events'/f'{hour:05d}.json').read_text('utf-8'))
        if stamp(event['source']['logical_at']) >= cutoff:
            return stamp(advances[hour]['detail']['physical_observation']['wall'])
    raise ValueError('Original observation has not reached this policy cutoff')


def _visible(path, physical):
    stat = path.stat()
    return max(stat.st_mtime_ns, stat.st_ctime_ns) <= physical.value


@contextmanager
def _runtime(lifecycle):
    """Actual existing runtime, without constructor initialization or repair."""
    from .neural_forward_ticket import open_existing_authority
    from .rolling import RollingSpec
    from .v7_feed import ReceiptArchive
    from .v7_forward import ForwardConfig, ForwardRuntime
    wrapper = json.loads((lifecycle.directory/'forward.json').read_text('utf-8'))
    identity = wrapper['identity']
    if wrapper != {'identity': identity, 'sha256': fingerprint(identity)}:
        raise ValueError('Original policy runtime identity changed')
    runtime = ForwardRuntime.__new__(ForwardRuntime)
    runtime.directory = lifecycle.directory; runtime.identity = identity
    runtime.lifecycle = lifecycle; runtime.clock = lifecycle.now
    runtime.config = ForwardConfig(**{**identity['config'], 'symbols': tuple(identity['config']['symbols'])})
    runtime.spec = RollingSpec(availability_delay_seconds=runtime.config.archive_delay_seconds,
                               window_days=runtime.config.training_window_days)
    from dataclasses import asdict
    if (json.loads(json.dumps(asdict(runtime.spec))) != identity['rolling']
            or runtime.config.training_window_days != lifecycle.plan.training_window_days
            or runtime.config.refit_interval_days != lifecycle.plan.refit_interval_days):
        raise ValueError('Original policy runtime rolling cadence changed')
    feed = ReceiptArchive.__new__(ReceiptArchive)
    feed.directory = project_path(identity['feed_directory']); feed.clock = lifecycle.now
    feed.symbols = runtime.config.symbols
    stored = json.loads((feed.directory/'identity.json').read_text('utf-8'))
    feed.identity = stored['source_id']
    if stored != {'source_id': feed.identity, 'symbols': list(feed.symbols)}:
        raise ValueError('Original policy feed identity changed')
    feed.db = sqlite3.connect((feed.directory/'receipts.sqlite').as_uri()+'?mode=ro', uri=True)
    feed.db.execute('PRAGMA query_only=ON'); runtime.feed = feed
    try:
        binding = json.loads((lifecycle.directory/'work_authority.json').read_text('utf-8'))
        authority = open_existing_authority(binding['authority_directory'])
        yield runtime, authority
    finally:
        feed.close()


def _selection(lifecycle, cutoff, k0):
    lifecycle.verify(); cutoff = lifecycle.fit_cutoff(cutoff)
    if cutoff > lifecycle.now():
        raise ValueError('Policy cannot inspect a future monthly cutoff')
    protocol = _protocol(lifecycle)
    if k0 != protocol.endpoints['algorithm'].reference:
        raise ValueError('Candidate policy changed its original K0 reference')
    physical = _physical_cutoff(lifecycle, cutoff)
    choices = []
    # Factor C remains its existing independent claim, without reassignment.
    from .quarantine_execution import verify_decision
    from .quarantined_observation import publication_visible
    for batch, raw, result_raw in lifecycle.db.execute(
            "SELECT id,identity,result FROM batches WHERE status='concluded'"):
        result, identity = json.loads(result_raw), json.loads(raw)
        decision = verify_decision(lifecycle, result)
        from .layered_research import research_retention_decision
        research = research_retention_decision(lifecycle, result)
        if not (result.get('supported') is True or decision is not None and decision['accepted']
                or research is not None and research['status'] == 'eligible_for_unconfirmed_research_retention'):
            continue
        if stamp(result['completed_at']) >= cutoff:
            continue
        publication = lifecycle.confirmation_publication(batch)
        path = lifecycle.directory/'claim_publications'/(fingerprint(batch)+'.json')
        observed = publication_visible(lifecycle, 'confirmation', batch, cutoff)
        if stamp(publication['published_at']) >= cutoff or (not _visible(path, physical) if observed is None else not observed):
            continue
        if decision is None and research is None and not lifecycle.calibration_ready():
            raise ValueError('Factor policy baseline lost original calibration')
        recipe = next(c for c in identity['candidates'] if c['pipeline_id'] == result['pipeline_id'])
        recipe = {k: v for k, v in recipe.items() if k != 'complexity'}
        choices.append((publication['published_at'], 'factor:'+batch, recipe,
            {'kind': 'factor_C' if research is None else 'unconfirmed_research_factor_C', 'batch': batch, 'result_hash': fingerprint(result),
             'publication_hash': fingerprint(publication), 'files': _file_bindings([path])}))
    from .algorithm_validation import AlgorithmValidation
    from .algorithm_validation_producer import _events
    from .neural_forward import FrozenNeuralRecipe
    from .structural_queue_review import checked_shortlist
    with _runtime(lifecycle) as (runtime, authority):
        for typed in AlgorithmValidation.discover(runtime, authority):
            scoped, _ = checked_shortlist(typed.catalogue, require_completed=True)
            if not scoped:
                raise ValueError('Policy cannot promote an unscoped legacy typed catalogue')
            eligible = {}
            for event in _events(typed):
                if event['kind'] != 'endpoint_result' or stamp(event['recorded_at']) >= cutoff:
                    continue
                anchor = authority.directory/'algorithm_validation_events'/typed.digest/(str(event['seq'])+'.json')
                if _visible(anchor, physical):
                    eligible[event['payload']['stage']] = event
            if not {'V', 'qualifier'} <= set(eligible):
                continue
            v, q = eligible['V'], eligible['qualifier']
            winner = v['payload']['score']['selected']
            if (winner is None or q['payload']['score']['selected'] != winner
                    or stamp(v['recorded_at']) >= stamp(typed.identity['qualifier_start'])):
                continue
            recipe = runtime.recipe_for(FrozenNeuralRecipe.from_dict(typed.identity['recipes'][winner]))
            choices.append((q['recorded_at'], 'typed:'+typed.digest, recipe,
                {'kind': 'typed_descriptive_selector', 'validation_id': typed.digest,
                 'catalogue_id': typed.catalogue.digest, 'V_hash': fingerprint(v),
                 'qualifier_hash': fingerprint(q), 'no_individual_confirmation': True,
                 'files': _file_bindings([typed.directory/'identity.json',
                    typed.catalogue.directory/'shortlist.json',
                    *(authority.directory/'algorithm_validation_events'/typed.digest/(str(e['seq'])+'.json')
                      for e in (v, q))])}))
    if not choices:
        return k0, {'kind': 'initial_K0'}
    _, _, recipe, evidence = max(choices, key=lambda value: (stamp(value[0]), value[1]))
    return recipe, evidence


def candidate_pipeline(lifecycle, cutoff, k0):
    from .nominal_algorithm_warmboot import initial_pipeline
    initial = initial_pipeline(lifecycle, cutoff, k0)
    if initial is not None:
        return initial
    cutoff = lifecycle.fit_cutoff(cutoff)
    if not _selection_paths(lifecycle, cutoff)[0].exists():
        _freeze_selection(lifecycle, cutoff, k0)
    with _original_reader(lifecycle) as reader:
        return _read_selection(reader, cutoff, k0)['pipeline']


def _selection_paths(lifecycle, cutoff):
    name = ((cutoff.strftime('%Y-%m')) if lifecycle.plan.training_window_days is None
            else fingerprint(cutoff.isoformat()))+'.json'
    return (lifecycle.directory/'policy_selections'/name,
            lifecycle.directory/'claim_protocol'/'policy_selections'/name)


def _file_bindings(paths):
    from .evidence import file_hash
    return {str(p): file_hash(p) for p in paths}


def _read_selection(lifecycle, cutoff, k0):
    protocol = _protocol(lifecycle)
    paths = _selection_paths(lifecycle, cutoff)
    value = json.loads(paths[0].read_text('utf-8'))
    from .calibration_runtime_contract import _original_operations
    operations = _original_operations(lifecycle)
    original = [o for o in operations if o['kind'] == 'whole_policy_selection_bound'
                and o['detail']['cutoff'] == cutoff.isoformat()]
    bindings = [o['detail'] for o in original]
    if (json.loads(paths[1].read_text('utf-8')) != value or bindings != [value]
            or value['version'] != policy_definition(protocol.endpoints['algorithm'].reference, research_spec_hash=protocol.endpoints['algorithm'].research_spec_hash)['version'] or value['policy_hash'] != fingerprint(policy_definition(protocol.endpoints['algorithm'].reference, research_spec_hash=protocol.endpoints['algorithm'].research_spec_hash))
            or value['claim_protocol'] != protocol.digest or value['plan_id'] != lifecycle.plan.digest
            or value['cutoff'] != cutoff.isoformat() or value['reference'] != k0
            or k0 != protocol.endpoints['algorithm'].reference):
        raise ValueError('Original monthly whole-policy selection changed')
    from .evidence import file_hash
    if any(file_hash(project_path(p)) != digest for p, digest in value['selection'].get('files', {}).items()):
        raise ValueError('Original policy selector evidence changed')
    if any(o['kind'] == 'fit_planned' and o['detail']['cutoff'] == cutoff.isoformat()
           and o['detail']['stream'] == TRIAL_STREAM and o['seq'] <= original[0]['seq'] for o in operations):
        raise ValueError('Policy trial was planned before its original selection binding')
    return value


def _freeze_selection(lifecycle, cutoff, k0):
    """Bind once before the first monthly trial/A0 job; historical reads cannot repair."""
    from .neural_forward_ticket import _LifecycleReader
    if type(lifecycle) is _LifecycleReader or cutoff != lifecycle.fit_cycle_start(lifecycle.now()):
        raise ValueError('Missing original policy selection cannot be frozen retrospectively')
    paths = _selection_paths(lifecycle, cutoff)
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(paths[0])+'.selection.lock', timeout=0):
        from .calibration_runtime_contract import _original_operations
        found = [o for o in _original_operations(lifecycle) if o['kind'] == 'whole_policy_selection_bound'
                 and o['detail']['cutoff'] == cutoff.isoformat()]
        if found or any(p.exists() for p in paths):
            return _read_selection(lifecycle, cutoff, k0)
        if lifecycle.db.execute('SELECT 1 FROM jobs WHERE cutoff=? AND stream=?',
                (cutoff.isoformat(), TRIAL_STREAM)).fetchone():
            raise ValueError('Policy selection must precede its original trial job')
        with _original_reader(lifecycle) as reader:
            pipeline, selection = _selection(reader, cutoff, k0)
            protocol = _protocol(reader)
        value = {'version': policy_definition(protocol.endpoints['algorithm'].reference, research_spec_hash=protocol.endpoints['algorithm'].research_spec_hash)['version'], 'policy_hash': fingerprint(policy_definition(protocol.endpoints['algorithm'].reference, research_spec_hash=protocol.endpoints['algorithm'].research_spec_hash)),
            'plan_id': lifecycle.plan.digest, 'claim_protocol': protocol.digest,
            'cutoff': cutoff.isoformat(), 'reference': k0, 'pipeline': pipeline,
            'selection': selection, 'frozen_at': lifecycle.now().isoformat()}
        for path in paths:
            immutable_record(path, value)
        with lifecycle.db:
            lifecycle._event('whole_policy_selection_bound', value)
        return _read_selection(lifecycle, cutoff, k0)


def canonical_adoption(lifecycle, cutoff, pipeline=None, *, expected=None, historical=False):
    """Verify the one original whole-policy claim, never a qualifier as C."""
    with _original_reader(lifecycle) as reader:
        return _canonical_adoption(reader, cutoff, pipeline, expected=expected, historical=historical)


def _canonical_adoption(lifecycle, cutoff, pipeline=None, *, expected=None, historical=False):
    cutoff = lifecycle.fit_cutoff(cutoff)
    if cutoff > lifecycle.now() or historical and not isinstance(expected, dict):
        raise ValueError('Actual completed policy adoption cutoff/binding required')
    if isinstance(expected, dict) and expected.get('version') in (
            'v7-confirmed-neural-adoption-v1', 'v7-unconfirmed-research-adoption-v1'):
        return None  # The caller must still run the original exact factor-C verifier.
    from .claim_protocol import ClaimProtocol
    if (not ClaimProtocol._bindings(lifecycle)
            and not (lifecycle.directory/'claim_protocol'/'protocol.json').exists()):
        if expected is not None:
            raise ValueError('Original whole-policy adoption authority missing')
        return None
    protocol = _protocol(lifecycle)
    from .layered_research import verified_research_spec
    research = verified_research_spec(lifecycle)
    if research is not None:
        selected = _read_selection(lifecycle, cutoff, protocol.endpoints['algorithm'].reference)
        chosen = selected['pipeline']
        if pipeline is not None and chosen != pipeline:
            if expected is not None:
                raise ValueError('Original research selected pipeline changed')
            return None
        if chosen == protocol.endpoints['algorithm'].reference:
            return None
        value = {'version': 'v7-unconfirmed-layered-policy-adoption-v1',
            'claim_protocol': protocol.digest, 'research_spec_hash': research.digest,
            'policy_hash': fingerprint(policy_definition(protocol.endpoints['algorithm'].reference,
                research_spec_hash=research.digest)), 'selection_hash': fingerprint(selected),
            'cutoff': cutoff.isoformat(), 'pipeline_id': chosen['pipeline_id'],
            'pipeline': chosen, 'selection': selected['selection'],
            'formal_gate_eligible': False, 'whole_agent_power_certified': False,
            'no_individual_neural_claim': True}
        if expected is not None and expected != value:
            raise ValueError('Original research adoption binding changed')
        return value
    row = protocol._row('algorithm')
    if row[1] is None:
        return None
    score = json.loads(row[1])
    from .quarantine_execution import verify_claim_result, verify_decision
    verify_claim_result(protocol, score)
    decision = verify_decision(lifecycle, score['result'])
    if not (score['result'].get('supported') is True or decision is not None and decision['accepted']):
        return None
    if stamp(score['completed_at']) >= cutoff:
        return None
    path = protocol.directory/'algorithm'/'score.json'
    if not path.is_file() or json.loads(path.read_text('utf-8')) != score:
        raise ValueError('Original policy confirmation publication missing or changed')
    if not _visible(path, _physical_cutoff(lifecycle, cutoff)):
        return None
    selected = _read_selection(lifecycle, cutoff, protocol.endpoints['algorithm'].reference)
    chosen, selector = selected['pipeline'], selected['selection']
    if pipeline is not None and chosen != pipeline:
        if expected is not None:
            raise ValueError('Original whole-policy selected pipeline changed')
        return None
    value = {'version': 'v7-confirmed-whole-policy-adoption-v2',
        'claim_protocol': protocol.digest, 'endpoint_id': score['endpoint_id'],
        'claim_slot': 'algorithm', 'alpha': .01, 'policy_hash': fingerprint(policy_definition(protocol.endpoints['algorithm'].reference, research_spec_hash=protocol.endpoints['algorithm'].research_spec_hash)),
        'result_hash': fingerprint(score), 'completed_at': score['completed_at'],
        'cutoff': cutoff.isoformat(), 'pipeline_id': chosen['pipeline_id'],
        'pipeline': chosen, 'selection': selector, 'no_individual_neural_claim': True}
    if expected is not None and value != expected:
        raise ValueError('Original whole-policy adoption authority changed')
    return value


def adopted_pipeline(lifecycle, cutoff, default):
    # A prospective scheduler may publish the new month's selector after the
    # original whole-policy result is supported. Historical ticket verification
    # only reads this original record and never re-enters wallet verification.
    from .claim_protocol import ClaimProtocol
    if ClaimProtocol._bindings(lifecycle):
        protocol = _protocol(lifecycle)
        row = protocol._row('algorithm')
        if row[1] is not None and stamp(json.loads(row[1])['completed_at']) < lifecycle.fit_cutoff(cutoff):
            from .quarantine_execution import verify_decision
            result = json.loads(row[1])['result']; isolated = verify_decision(lifecycle, result)
            if result.get('supported') is True or isolated is not None and isolated['accepted']:
                candidate_pipeline(lifecycle, cutoff, protocol.endpoints['algorithm'].reference)
    bound = canonical_adoption(lifecycle, cutoff)
    return default if bound is None else bound['pipeline']


def _nominal_pool_source(policy):
    """Source registration, not a caller role or relaxed old policy issuer."""
    from importlib import import_module
    from pathlib import Path

    from .evidence import file_hash
    from .monthly_work_authority import source_files
    files = source_files()
    for name in ('typed_policy_confirmation', 'pool_forward', 'nominal_algorithm_policy',
                 'nominal_algorithm_warmboot', 'nominal_algorithm_producer'):
        module = import_module('factorlab.discovery.' + name)
        relative = 'factorlab/discovery/' + name + '.py'
        if (Path(module.__file__).resolve() != project_path(relative).resolve()
                or files.get(relative) != file_hash(module.__file__)):
            raise ValueError('Canonical nominal pool policy source changed')
    if files != policy.identity['source_files']:
        raise ValueError('Nominal policy belongs to another complete source')
    return fingerprint(files)


def nominal_pool_context(lifecycle, job, recipe, *, authorized_at, active):
    """New nominal-only pool trial; original neural policy path is unchanged.

    Read the pre-job original monthly selection. The original reader cannot
    freeze a missing selection retrospectively. This issues no work or credit.
    """
    from .pool_forward import PoolForwardRecipe, pipeline_for
    if type(recipe) is not PoolForwardRecipe or type(active) is not bool:
        raise TypeError('Exact original pool recipe and activity state required')
    with _original_reader(lifecycle) as reader:
        from .nominal_algorithm_policy import _bound
        found = _bound(reader)
        if found is None:
            raise ValueError('Old ordinary policy cannot borrow new nominal pool route')
        policy, permit = found
        source_hash = _nominal_pool_source(policy)
        protocol = _protocol(reader); endpoint = protocol.endpoints['algorithm']
        if [e.to_dict() for e in protocol.endpoints.values()] != policy.identity['endpoints']:
            raise ValueError('Nominal pool route changed the complete registered endpoints')
        row = reader.db.execute('SELECT cutoff,stream,pipeline FROM jobs WHERE id=?', (job,)).fetchone()
        pipeline = pipeline_for(recipe)
        if row is None or row[1] != TRIAL_STREAM or json.loads(row[2]) != pipeline:
            raise ValueError('Nominal pool job differs from its original recipe/stream')
        cutoff, observed = reader.fit_cutoff(row[0]), stamp(authorized_at)
        if (not stamp(endpoint.start) <= cutoff <= observed < stamp(endpoint.end)
                or active and not cutoff <= reader.now() < stamp(endpoint.end)):
            raise ValueError('Nominal pool trial is outside its original open six-month endpoint')
        chosen = candidate_pipeline(reader, cutoff, endpoint.reference)
        selected = _read_selection(reader, cutoff, endpoint.reference)
        if chosen != pipeline or selected['pipeline'] != pipeline:
            raise ValueError('Nominal pool job changed its original causal monthly selection')
        return {'role': 'new_nominal_whole_policy_pool_trial',
            'version': 'nominal-whole-policy-pool-context-v1',
            'nominal_policy_digest': policy.digest, 'trial': permit.trial,
            'source_hash': source_hash, 'claim_protocol': protocol.digest,
            'endpoint_id': endpoint.digest, 'claim_slot': 'algorithm', 'alpha': .01,
            'policy_hash': fingerprint(policy_definition(protocol.endpoints['algorithm'].reference, research_spec_hash=protocol.endpoints['algorithm'].research_spec_hash)), 'selection_hash': fingerprint(selected),
            'selection': selected['selection'], 'A0_adoption_authorized': False}


def policy_context(lifecycle, job, recipe):
    with _original_reader(lifecycle) as reader:
        return _policy_context(reader, job, recipe)


def _policy_context(lifecycle, job, recipe):
    from .neural_forward_contract import neural_pipeline
    protocol = _protocol(lifecycle); endpoint = protocol.endpoints['algorithm']
    row = lifecycle.db.execute('SELECT cutoff,stream,pipeline FROM jobs WHERE id=?', (job,)).fetchone()
    pipeline = neural_pipeline(recipe, recipe.rolling_spec)
    from .nominal_algorithm_warmboot import typed_context
    typed_context(lifecycle, job, pipeline)
    if (not row or row[1] != TRIAL_STREAM or json.loads(row[2]) != pipeline
            or not stamp(endpoint.start) <= stamp(row[0]) < stamp(endpoint.end)):
        raise ValueError('Policy trial job differs from the original six-month claim')
    selected = _read_selection(lifecycle, stamp(row[0]), endpoint.reference)
    chosen, selector = selected['pipeline'], selected['selection']
    if chosen != pipeline:
        raise ValueError('Policy trial changed its causal complete learner selection')
    return {'version': 'v7-whole-policy-trial-context-v2', 'claim_protocol': protocol.digest,
        'endpoint_id': endpoint.digest, 'claim_slot': 'algorithm', 'alpha': .01,
        'policy_hash': fingerprint(policy_definition(protocol.endpoints['algorithm'].reference, research_spec_hash=protocol.endpoints['algorithm'].research_spec_hash)), 'job': job, 'cutoff': row[0],
        'stream': TRIAL_STREAM, 'pipeline_id': pipeline['pipeline_id'],
        'selection': selector, 'A0_adoption_authorized': False}
