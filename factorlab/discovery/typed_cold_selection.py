"""Read the actual paid typed-V winner as a conditional cold learner recipe.

The winner is a complete neural learner, never the adaptive A0 policy. All
original protocol/paid catalogue/event bindings are checked without constructing
or repairing runtime databases and without executing a second validation look.
"""
from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager

import pandas as pd

from factorlab.protocol import fingerprint, project_path
from .calibration_oracle_evidence import _lifecycle
from .full_calibration_registry import QuarantinedTrialPermit


@contextmanager
def _runtime_reader(permit):
    from .v7_forward import ForwardRuntime, ForwardConfig
    from .v7_feed import ReceiptArchive
    from .rolling import RollingSpec
    directory = permit.paths['runtime']
    wrapper = json.loads((directory/'forward.json').read_text('utf-8'))
    if set(wrapper) != {'identity', 'sha256'} or fingerprint(wrapper['identity']) != wrapper['sha256']:
        raise ValueError('Original typed winner runtime identity changed')
    identity = wrapper['identity']
    fields = dict(identity['config']); fields['symbols'] = tuple(fields['symbols'])
    runtime = ForwardRuntime.__new__(ForwardRuntime)
    runtime.directory = directory; runtime.identity = identity; runtime.config = ForwardConfig(**fields)
    runtime.spec = RollingSpec(availability_delay_seconds=runtime.config.archive_delay_seconds,
                               window_days=runtime.config.training_window_days)
    from dataclasses import asdict
    if json.loads(json.dumps(asdict(runtime.spec))) != identity['rolling']:
        raise ValueError('Original typed winner rolling cadence changed')
    runtime.clock = lambda: pd.Timestamp.now(tz='UTC')
    feed = ReceiptArchive.__new__(ReceiptArchive)
    feed.directory = project_path(identity['feed_directory']); feed.clock = runtime.clock
    feed.symbols = runtime.config.symbols
    stored = json.loads((feed.directory/'identity.json').read_text('utf-8'))
    feed.identity = stored['source_id']
    if stored != {'source_id': feed.identity, 'symbols': list(feed.symbols)} or feed.directory != permit.paths['feed']:
        raise ValueError('Typed selection changed its original registered feed')
    feed.db = sqlite3.connect((feed.directory/'receipts.sqlite').as_uri()+'?mode=ro', uri=True)
    feed.db.execute('PRAGMA query_only=ON')
    runtime.feed = feed
    try:
        with _lifecycle(directory) as lifecycle:
            permit.verify(lifecycle)
            if (runtime.config.training_window_days != lifecycle.plan.training_window_days
                    or runtime.config.refit_interval_days != lifecycle.plan.refit_interval_days):
                raise ValueError('Original typed winner lifecycle cadence changed')
            runtime.lifecycle = lifecycle
            yield runtime
    finally:
        feed.close()


def _selected_protocol(protocol):
    from .algorithm_validation import AlgorithmValidation
    from .algorithm_validation_producer import endpoint_result
    from .structural_queue_review import require_queue_completion
    from .neural_forward import FrozenNeuralRecipe
    if type(protocol) is not AlgorithmValidation:
        raise TypeError('Actual canonical typed AlgorithmValidation required')
    require_queue_completion(protocol)
    protocol.verify()
    selected = endpoint_result(protocol, 'V')
    value = protocol.identity
    if (selected is None or selected['score']['selected'] is None
            or pd.Timestamp(selected['recorded_at']) >= pd.Timestamp(value['qualifier_start'])):
        raise ValueError('Cold neural oracle requires the original timely actual typed-V winner')
    pipeline_id = selected['score']['selected']
    if pipeline_id not in value['recipes']:
        raise ValueError('Typed winner is outside its original frozen recipe catalogue')
    recipe = FrozenNeuralRecipe.from_dict(value['recipes'][pipeline_id])
    if recipe.pipeline_id != pipeline_id:
        raise ValueError('Typed winner complete recipe identity changed')
    candidate = protocol.runtime.recipe_for(recipe)
    reference = protocol.runtime.recipe_for(recipe.pool)
    if reference != value['references']['Pv']:
        raise ValueError('Typed winner changed its original same-pool reference')
    original = {'kind': 'original_typed_V_winner_first_qualifier_month',
        'validation_id': protocol.digest, 'catalogue_id': protocol.catalogue.digest,
        'endpoint_result_digest': fingerprint(selected), 'winner': pipeline_id,
        'full_recipe': recipe.to_dict(), 'reference': reference,
        'qualifier_start': value['qualifier_start'], 'qualifier_end': value['qualifier_end'],
        'full_adaptive_A0_truth': False, 'ordinary_factor_claim': False}
    start = pd.Timestamp(value['qualifier_start'])
    end = min(pd.Timestamp(value['qualifier_end']), start+pd.DateOffset(months=1))
    return json.loads(json.dumps({'original_estimand': original,
        'original_estimand_digest': fingerprint(original), 'start': start.isoformat(), 'end': end.isoformat(),
        'reference': reference, 'candidate': candidate}, sort_keys=True, allow_nan=False))


def select_typed_neural(permit, slot):
    from .algorithm_shortlist import AlgorithmCatalogue, _read
    from .algorithm_validation import AlgorithmValidation
    from .neural_forward_ticket import open_existing_authority
    from .structural_queue_review import checked_shortlist
    if type(permit) is not QuarantinedTrialPermit or permit.trial['role'] != 'validation':
        raise TypeError('Actual original validation trial permit required')
    permit.verify()
    if type(slot) is not str or not re.fullmatch(r'typed_neural_[0-9a-f]{64}', slot):
        raise ValueError('Typed neural selector must name its actual original catalogue')
    catalogue_id = slot.removeprefix('typed_neural_')
    directory = permit.paths['runtime']/'algorithm_validation'/catalogue_id
    identity = _read(directory/'identity.json')
    catalogue = AlgorithmCatalogue(identity['catalogue_directory'])
    if catalogue.digest != catalogue_id:
        raise ValueError('Typed selector cannot switch original catalogue namespace')
    # Enforce the completed paid G03 closure BEFORE AlgorithmValidation opens;
    # an unscoped legacy catalogue cannot trigger unpaid numerical replay.
    scoped, shortlist = checked_shortlist(catalogue, require_completed=True)
    if not scoped or shortlist is None:
        raise ValueError('Typed cold selection requires the original paid complete queue closure')
    authority = open_existing_authority(permit.paths['authority'])
    with _runtime_reader(permit) as runtime:
        protocol = AlgorithmValidation(runtime, authority, catalogue)
        return _selected_protocol(protocol)
