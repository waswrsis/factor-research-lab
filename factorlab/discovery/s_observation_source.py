"""Original prepaid S observation provenance; physical deadlines stay physical."""
from pathlib import Path

import pandas as pd

from factorlab.protocol import project_path


def validate_reference(reference):
    from .neural_forward_ticket import _LifecycleReader, _read
    from .quarantined_observation import QuarantinedObservation
    required = {'feed_identity', 'feed_directory', 'exposure_path', 'cutoff'}
    if set(reference) not in (required, required | {'observation_prefix'}):
        raise ValueError('S source reference schema changed')
    exposure = project_path(reference['exposure_path'])
    if exposure.name != 'exposure.sqlite':
        raise ValueError('S observation requires the exact lifecycle exposure ledger')
    reader = _LifecycleReader(exposure.parent)
    try:
        reader.verify()
        identity = _read(reader.directory/'forward.json')['identity']
        if (reader.plan.source_id != reference['feed_identity']
                or project_path(identity['feed_directory']) != project_path(reference['feed_directory'])):
            raise ValueError('S observation changed its original source or lifecycle')
        obj = QuarantinedObservation._open_original(reader)
        prefix = reference.get('observation_prefix')
        if obj is None:
            if 'observation_prefix' in reference:
                raise ValueError('Unbound S source cannot borrow an observation prefix')
        else:
            obj.verify_binding(prefix)
            if (obj.permit.paths['feed'] != project_path(reference['feed_directory'])
                    or pd.Timestamp(reference['cutoff']) > pd.Timestamp(prefix['logical_at'])):
                raise ValueError('S source cutoff exceeds its original observation')
        return prefix
    finally:
        reader.close()


def preparation_context(store, month, bridge):
    """Replay the actual dual S preparation already bound by the bridge."""
    if bridge.get('preparation_hash') is None:
        return {}
    from .neural_forward_ticket import open_existing_authority
    from .s_preparation import SPreparationEnvelope
    authority = open_existing_authority(bridge['authority_directory'])
    prep = SPreparationEnvelope.load(authority, store.directory, month)
    prep.baseline_record()  # Original paid reservation, not merely matching JSON hashes.
    reference = prep.identity['source_reference']
    if (prep.digest != bridge['preparation_hash']
            or authority.manifest_hash != bridge['authority_manifest_hash']
            or project_path(reference['exposure_path']) != store.exposure.path
            or reference['feed_identity'] != store.identity['feed_identity']
            or reference['cutoff'][:7] != month):
        raise ValueError('S context differs from its original paid source preparation')
    prefix = validate_reference(reference)
    if prefix is None:
        return {}
    return {'observation_prefix': prefix,
            'lifecycle_directory': str(project_path(reference['exposure_path']).parent),
            's_preparation': {'authority_directory': str(authority.directory),
                'run_directory': str(store.directory), 'month': month, 'digest': prep.digest}}


def reference_from_context(context):
    from .neural_forward_ticket import open_existing_authority
    from .s_preparation import SPreparationEnvelope
    binding = context.get('s_preparation')
    if type(binding) is not dict or set(binding) != {'authority_directory', 'run_directory', 'month', 'digest'}:
        raise ValueError('Logical S context lacks original paid preparation')
    authority = open_existing_authority(binding['authority_directory'])
    prep = SPreparationEnvelope.load(authority, binding['run_directory'], binding['month'])
    prep.baseline_record()  # Original paid reservation, not merely matching JSON hashes.
    reference = prep.identity['source_reference']
    if (prep.digest != binding['digest']
            or project_path(binding['run_directory']) != project_path(context['run_directory'])
            or binding['month'] != context['replay_month']
            or reference['cutoff'] != context['cutoff']
            or reference['feed_identity'] != context['source_id']
            or str(project_path(reference['exposure_path']).parent) != context['lifecycle_directory']
            or validate_reference(reference) != context['observation_prefix']):
        raise ValueError('Logical S context changed its original source prefix')
    return reference


def physical_cutoff(context):
    if context.get('observation_prefix') is None:
        return pd.Timestamp(context['cutoff'])
    reference_from_context(context)
    from .neural_forward_ticket import _LifecycleReader
    from .nonlinear_runtime import physical_bound_for_prefix
    reader = _LifecycleReader(context['lifecycle_directory'])
    try:
        return physical_bound_for_prefix(reader, context['observation_prefix'])
    finally:
        reader.close()


def snapshot_from_context(context, feed_directory):
    reference = reference_from_context(context)
    if project_path(feed_directory) != project_path(reference['feed_directory']):
        raise ValueError('S snapshot changed its original feed directory')
    from .neural_forward_ticket import _LifecycleReader
    from .quarantined_observation import snapshot_for_prefix
    reader = _LifecycleReader(context['lifecycle_directory'])
    try:
        return snapshot_for_prefix(reader, context['observation_prefix'], context['cutoff'],
                                   months=6, context_hours=1234)
    finally:
        reader.close()
