"""One typed numerical readback envelope in the existing nonlinear partition."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing

from factorlab.protocol import fingerprint, project_path

from .monthly_partition_policy import counts
from .v7_storage import immutable_record

VERSION = 'v7-structure-review-intent-v2'


def _read(path):
    path = project_path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2**20:
        raise ValueError('Missing/bounded original structure review intent')
    value = json.loads(path.read_text('utf-8'))
    if set(value) != {'body', 'hash'} or fingerprint(value['body']) != value['hash']:
        raise ValueError('Structure review intent checksum differs')
    return value


def _history_refs(store, month):
    refs = []
    rows = store.db.execute('SELECT month FROM months WHERE month<? ORDER BY month LIMIT 121', (month,)).fetchall()
    if len(rows) > 120:
        raise ValueError('Bounded structure review history exhausted')
    for (old_month,) in rows:
        value = store.get(old_month).get('structural_plan')
        if value is not None:
            refs.append({'month': old_month, 'plan_hash': value['hash']})
    return refs


def freeze_review_intent(bridge, base):
    from .structural_evolution import POLICY
    bridge._guard()
    if bridge.authority.manifest.get('partition_policy') is None:
        raise ValueError('Structural review requires actual nontransferable monthly partitions')
    if (int(bridge.month[5:7])-1) % 3 == 0:
        raise ValueError('Quarterly probe obligation defers structural review')
    if bridge.store.identity['config']['budget_policy'] == 'controls_third' and bridge.store.identity['config']['search_mode'] != 'gpt':
        raise ValueError('Only the original gpt controls owner has nonlinear credit')
    schedule = json.loads((bridge.directory/'schedule/identity.json').read_text('utf-8'))
    body = {'version': VERSION, 'bridge_identity_hash': fingerprint(bridge.identity),
        'authority_manifest_hash': bridge.authority.manifest_hash, 'data_id': bridge.state['data_id'],
        'input_id': bridge.state['nonlinear_input_data_id'], 'cutoff': bridge.state['cutoff'],
        'base_recipe': json.loads(json.dumps(base.to_dict())), 'policy': POLICY,
        'history': _history_refs(bridge.store, bridge.month), 'schedule_hash': schedule['schedule_hash'],
        'counts': counts(solve=1), 'seconds': 300, 'no_training': True}
    value = {'body': body, 'hash': fingerprint(body)}
    for root in (bridge.directory, bridge.anchor):
        immutable_record(root/'structure_review.json', value)
    return value


def validate_review_route(authority, key, binding, bridge, lease, debit):
    """Read-only route; no numerical model replay and no recursive wallet open."""
    from .neural_forward import FrozenNeuralRecipe
    from .neural_research_ticket import _reader
    from .structural_evolution import POLICY
    local = project_path(bridge['run_directory'])/'work_authority'/bridge['replay_month']
    anchor = authority.directory/'search_bridges'/bridge['bridge_id']
    value = _read(local/'structure_review.json')
    if _read(anchor/'structure_review.json') != value:
        raise ValueError('Structure review dual intent differs')
    body = value['body']
    store = _reader(bridge['run_directory'])
    try:
        state = store.get(bridge['replay_month'])
        base = FrozenNeuralRecipe.from_dict(body['base_recipe'])
        history = _history_refs(store, bridge['replay_month'])
        config = store.identity['config']
        if (base.pool.to_dict() != state['frozen_reference_pool'] or base.rolling_spec.digest != store.identity['rolling_hash']
                or config['budget_policy'] == 'controls_third' and config['search_mode'] != 'gpt'):
            raise ValueError('Review lost its original pool/spec/owner')
    finally:
        store.close()
    schedule = json.loads((local/'schedule/identity.json').read_text('utf-8'))
    expected = {'version': VERSION, 'bridge_identity_hash': fingerprint(bridge),
        'authority_manifest_hash': authority.manifest_hash, 'data_id': state['data_id'],
        'input_id': state['nonlinear_input_data_id'], 'cutoff': state['cutoff'],
        'base_recipe': json.loads(json.dumps(base.to_dict())), 'policy': POLICY,
        'history': history, 'schedule_hash': schedule['schedule_hash'],
        'counts': counts(solve=1), 'seconds': 300, 'no_training': True}
    if (body != expected or binding['work'] != {'kind': 'structure_review', 'intent_hash': value['hash']}
            or key != 'work-authority-bridge-v1:'+bridge['bridge_id']+':structure_review'
            or lease.lease_seconds != 300 or debit != counts(solve=1)
            or (int(bridge['replay_month'][5:7])-1) % 3 == 0):
        raise ValueError('Review allocation changed original typed work')
    return value


def verify_review_completion(authority, selection):
    """Local completed-row check within the enclosing canonical wallet verifier.

    Using an already verified authority permits a full get; historical producer
    route checks use the fixed SQL row to avoid recursive wallet verification.
    """
    from .monthly_work_authority import MonthlyWorkAuthority
    body = selection['body']; rid = body['review_reservation']
    if rid is None:
        raise ValueError('Full neural S requires a completed structure review')
    if type(authority) is MonthlyWorkAuthority:
        row = authority.get(rid)
    else:
        root = project_path(authority)
        with closing(sqlite3.connect((root/'authority.sqlite').as_uri()+'?mode=ro', uri=True)) as db:
            raw = db.execute('SELECT body FROM reservations WHERE id=?', (rid,)).fetchone()
        if raw is None:
            raise ValueError('Original structure review reservation missing')
        row = json.loads(raw[0])
    if (row['status'] != 'completed' or row['binding']['bridge_id'] != body['bridge_id']
            or row['binding']['work']['kind'] != 'structure_review' or row['counts'] != counts(solve=1)
            or row['actual_counts'] != counts(solve=1) or row['lease']['lease_seconds'] != 300
            or row['result']['result'] != {'kind': 'structure_review', 'structural_plan_hash': selection['hash'],
                 'intent_hash': row['binding']['work']['intent_hash'], 'training_performed': False}):
        raise ValueError('Original structure review is not fully completed')
    _verify_review_publication(row, selection)
    return row


def _publication_paths(row, selection):
    from .work_authority_bridge import _read as read_bridge
    authority = project_path(row['authority_directory'])
    bid = selection['body']['bridge_id']
    anchor = authority/'search_bridges'/bid
    identity = read_bridge(anchor/'identity.json')
    local = project_path(identity['run_directory'])/'work_authority'/identity['replay_month']
    if (read_bridge(local/'identity.json') != identity
            or fingerprint(identity) != selection['body']['bridge_identity_hash']):
        raise ValueError('Review publication lost its original bridge identity')
    return local, anchor


def publish_review_completion(bridge, selection, row):
    """Publish only after the original bridge's final completion IO returns."""
    from .work_lease import FrozenWorkLease, _observe
    lease = FrozenWorkLease.from_dict(row['lease'])
    lease.assert_parent_alive()
    body = {'version': 'v7-structure-review-publication-v1',
            'reservation_id': row['reservation_id'], 'reservation_hash': fingerprint(row),
            'selection_hash': selection['hash'], 'lease_hash': lease.digest,
            'terminal_sequence': bridge._sequence, 'terminal_hash': bridge._head,
            'observation': _observe(lease)}
    value = {'body': body, 'hash': fingerprint(body)}
    for root in _publication_paths(row, selection):
        immutable_record(root/'structure_review_final_io.json', value)
    _verify_review_publication(row, selection)
    lease.assert_parent_alive()


def _verify_review_publication(row, selection):
    import pandas as pd
    from .work_authority_bridge import _read as read_bridge
    from .work_lease import FrozenWorkLease, _evaluate
    roots = _publication_paths(row, selection)
    paths = [root/'structure_review_final_io.json' for root in roots]
    final = _read(paths[0])
    if _read(paths[1]) != final:
        raise ValueError('Original review final publication mirrors differ')
    body = final['body']
    lease = FrozenWorkLease.from_dict(row['lease'])
    if (set(body) != {'version', 'reservation_id', 'reservation_hash', 'selection_hash',
                     'lease_hash', 'terminal_sequence', 'terminal_hash', 'observation'}
            or body['version'] != 'v7-structure-review-publication-v1'
            or body['reservation_id'] != row['reservation_id']
            or body['reservation_hash'] != fingerprint(row)
            or body['selection_hash'] != selection['hash'] or body['lease_hash'] != lease.digest
            or type(body['terminal_sequence']) is not int or body['terminal_sequence'] <= 0
            or not _evaluate(lease, body['observation']).allowed):
        raise ValueError('Original review final publication exceeded or changed its lease')
    terminal_paths = [root/'steps'/f"{body['terminal_sequence']:06d}.json" for root in roots]
    terminal = read_bridge(terminal_paths[0])
    if (read_bridge(terminal_paths[1]) != terminal or fingerprint(terminal) != body['terminal_hash']
            or terminal['operation'] != 'complete' or terminal['reservation_id'] != row['reservation_id']
            or terminal['terms_hash'] != row['terms_hash']
            or terminal['previous'] != row['result']['bridge_head']):
        raise ValueError('Original review final bridge completion changed')
    deadline = pd.Timestamp(lease.wall_deadline).value
    for path in paths + terminal_paths:
        stat = path.stat()
        if path.is_symlink() or max(stat.st_mtime_ns, stat.st_ctime_ns) >= deadline:
            raise ValueError('Original review final publication IO passed original deadline')


def combined_capacity(recipe, kinds, allocation, *, review):
    """No fourth300s grant and no transfer from S/forward unused credit."""
    from .s_nonlinear_ticket import capacity
    used = counts(solve=1 if review else 0)
    seconds = 300 if review else 0
    for kind in kinds:
        seconds += 300
        for key, amount in capacity(recipe, kind).items():
            used[key] += amount
    limits = allocation['nonlinear_max_counts']
    if seconds > allocation['nonlinear_seconds_holdback'] or any(used[k] > limits.get(k, 0) for k in used):
        raise ValueError('Combined review and producers exceed original nonlinear capacity')
    return {'counts': used, 'seconds': seconds}
