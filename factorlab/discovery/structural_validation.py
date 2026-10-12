"""Publish closed structural S catalogues only to the distinct typed V route."""
from __future__ import annotations

import json

import pandas as pd

from factorlab.protocol import fingerprint, project_path

from .algorithm_shortlist import AlgorithmCatalogue
from .algorithm_validation import AlgorithmValidation
from .neural_research_ticket import _reader


def _queue_numerical(runtime, authority, run_directory, month):
    """Read real closed S state; no numerical fit or ordinary factor claim.

    The start is fixed by the original S cutoff. A missed window is never
    silently moved forward. Existing exact protocol publication is read back.
    """
    store = _reader(run_directory)
    try:
        state = store.get(month)
        if state is None or state['status'] != 'complete':
            raise ValueError('Only a genuinely closed S month may queue algorithm validation')
        exposure = state['search_exposure']
        if (store.exposure.path != runtime.lifecycle.exposure.path
                or store.identity.get('feed_identity') != runtime.feed.identity
                or store.identity.get('mode') != 'authority_scheduled_research_v1'
                or exposure['source'] != runtime.feed.identity
                or sorted(exposure['symbols']) != sorted(runtime.config.symbols)
                or state['authority_bridge']['identity_hash'] is None):
            raise ValueError('Algorithm queue source/exposure does not match the actual runtime')
        store.exposure.assert_prompt_search(**{k: v for k, v in exposure.items() if k != 'data_id'})
        nonlinear = state.get('nonlinear_report') or {}
        if 'catalogue' not in nonlinear:
            return {'status': 'no_closed_full_learner', 'A0_changed': False, 'formal_claim': False}
        selection = state.get('structural_plan')
        if selection is None:
            raise ValueError('Typed structural queue requires completed paid structure review')
        from .structural_review import verify_review_completion
        verify_review_completion(authority, selection)
        catalogue = AlgorithmCatalogue(store.directory / 'algorithm_catalogue' / month)
        if (catalogue.identity['authority_directory'] != str(authority.directory)
                or catalogue.identity['authority_manifest_hash'] != authority.manifest_hash
                or catalogue.identity['controller_data_id'] != state['data_id']
                or catalogue.identity['bridge_identity_hash'] != state['authority_bridge']['identity_hash']):
            raise ValueError('Algorithm queue changed its original shared work authority')
        if set(catalogue.identity['recipes']) != {selection['body']['pipeline_id']}:
            raise ValueError('Catalogue differs from the completely reviewed structure')
        path = catalogue.directory / 'shortlist.json'
        # Missing saved shortlist is not repaired into a favorable result.
        value = json.loads(path.read_text('utf-8'))
        if (set(value) != {'value', 'hash'} or fingerprint(value['value']) != value['hash']
                or value['value'] != nonlinear['catalogue']):
            raise ValueError('Algorithm queue differs from closed S shortlist')
        current = catalogue.seal_shortlist()
        if current != value['value']:
            raise ValueError('Algorithm queue original S evidence changed')
        if not current['algorithm_shortlist']:
            return {'status': 'no_eligible_complete_learner', 'catalogue_id': catalogue.digest,
                    'A0_changed': False, 'formal_claim': False}
        from factorlab.time_utils import month_start
        start = month_start(pd.Timestamp(state['cutoff'])) + pd.DateOffset(months=1)
        existing = runtime.directory / 'algorithm_validation' / catalogue.digest
        anchor = authority.directory / 'algorithm_validations' / (catalogue.digest + '.json')
        if existing.exists() or anchor.exists():
            protocol = AlgorithmValidation(runtime, authority, catalogue)
            if pd.Timestamp(protocol.identity['start']) != start:
                raise ValueError('Algorithm queue cannot move its original prospective window')
        else:
            # The lifecycle derives logical time only from its original bound
            # observation journal; AlgorithmValidation retains physical IO checks.
            if start <= runtime.lifecycle.now():
                return {'status': 'missed_original_next_month', 'catalogue_id': catalogue.digest,
                        'start': start.isoformat(), 'A0_changed': False, 'formal_claim': False}
            protocol = AlgorithmValidation.freeze(runtime, authority, catalogue, start=start)
        if project_path(protocol.identity['catalogue_directory']) != catalogue.directory:
            raise ValueError('Published typed protocol belongs to another catalogue')
        return {'status': 'queued_typed_algorithm_V_only', 'catalogue_id': catalogue.digest,
                'directory': str(protocol.directory), 'identity_hash': protocol.digest,
                'start': start.isoformat(), 'A0_changed': False, 'formal_claim': False,
                'ordinary_factor_route': False}
    finally:
        store.close()


def queue_structural_validation(runtime, authority, run_directory, month):
    from .structural_queue_review import produce_typed_queue_review
    return produce_typed_queue_review(runtime, authority, run_directory, month)
