"""New-run-only S data identity using exact typed values and mature labels."""
import numpy as np
import pandas as pd

from factorlab.discovery.frame_identity import frame_identity
from factorlab.protocol import fingerprint


def raw_snapshot_identity(frame, *, cutoff):
    columns = tuple(sorted(frame.columns))
    visible = frame.loc[frame.available_at <= cutoff, list(columns)]
    ordered = visible.sort_values(['symbol', 'open_time'], kind='stable')
    return {'version': 'v7-S-raw-typed-v2', 'columns': list(columns),
            'order': ['symbol', 'open_time'], 'frame': frame_identity(ordered)}


def typed_search_data_id(data, selected, cutoff):
    visible = (data.frame.metadata.available_at < cutoff).to_numpy() & data.frame.valid_rows
    observed = visible & data.observed & np.asarray(data.maturity <= cutoff)
    metadata = data.frame.metadata.loc[visible].drop(columns=['label_status'], errors='ignore')
    features = pd.DataFrame(data.frame.values[visible], columns=list(data.frame.feature_names))
    labels = pd.DataFrame({'sample_id': data.frame.metadata.loc[observed, 'sample_id'].to_numpy(),
                          'target': data.target[observed], 'maturity': data.maturity[observed]})
    return fingerprint({'version': 'v7-S-visible-data-typed-v2',
        'sample_order': 'trusted_builder_frame_positions', 'feature_schema': data.frame.schema_hash,
        'feature_columns': list(data.frame.feature_names), 'X': frame_identity(features),
        'meta': frame_identity(metadata), 'labels': frame_identity(labels),
        'rolling': data.spec.digest, 'availability_mode': data.availability_mode,
        'feed_identity': selected.attrs.get('feed_identity'),
        'raw_asof_source': raw_snapshot_identity(selected, cutoff=cutoff)})
