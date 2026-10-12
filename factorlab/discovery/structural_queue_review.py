"""Prepaid, fixed-catalogue numeric review and prospective typed-V publication."""
from __future__ import annotations

import json
import os

import pandas as pd
import psutil
from factorlab.time_utils import month_start

from factorlab.protocol import fingerprint, project_path

from .monthly_partition_policy import counts
from .v7_storage import immutable_record
from .work_lease import FrozenWorkLease

VERSION = 'v7-typed-queue-review-v2'


def _read(path):
    path = project_path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2**20:
        raise ValueError('Missing/bounded original typed queue evidence')
    value = json.loads(path.read_text('utf-8'))
    if set(value) != {'body', 'hash'} or fingerprint(value['body']) != value['hash']:
        raise ValueError('Typed queue evidence checksum differs')
    return value


def _key(runtime_directory, run_directory, month):
    return fingerprint({'runtime': str(project_path(runtime_directory)),
                        'run': str(project_path(run_directory)), 'month': month})


def validate_queue_route(authority, key, binding, lease, debit):
    from .monthly_partition_policy import _runtime, _search
    from .neural_forward import FrozenNeuralRecipe
    from .neural_research_ticket import _reader
    from .s_nonlinear_ticket import _plan
    value = _read(authority.directory/'typed_queue_reviews'/key.split(':', 1)[1]/'intent.json')
    body = value['body']; bridge = body['bridge']
    if set(body) != {'version', 'runtime_directory', 'bridge', 'authority_manifest_hash', 'source_id',
                     'data_id', 'selection_hash', 'plan_hash', 'catalogue_hash', 'start', 'lease'}:
        raise ValueError('Unknown typed queue intent schema')
    store = _reader(bridge['run_directory'])
    try:
        state = store.get(bridge['replay_month'])
        recipe = FrozenNeuralRecipe.from_dict(state['structural_plan']['body']['recipe'])
        plan = _plan(bridge, state, recipe, 'typed_queue_review', store)
        expected_binding = {'role': 'S_typed_queue', 'queue_bridge_id': bridge['bridge_id'],
            'runtime_directory': body['runtime_directory'], 'intent_hash': value['hash']}
        source = store.identity.get('feed_identity')
        catalogue = _read_catalogue_identity(store.directory, bridge['replay_month'])
        catalogue_identity = catalogue['value']
        if (json.loads(project_path(catalogue_identity['anchor']).read_text('utf-8')) != catalogue
                or catalogue_identity['bridge_identity_hash'] != fingerprint(bridge)
                or catalogue_identity['controller_data_id'] != state['data_id']
                or catalogue_identity['authority_manifest_hash'] != authority.manifest_hash
                or catalogue_identity['recipes'] != {recipe.pipeline_id: json.loads(json.dumps(recipe.to_dict()))}):
            raise ValueError('Queue needs the actual original full recipe catalogue')
        if (state['status'] != 'complete' or body['version'] != VERSION or binding != expected_binding
                or key != 'typed_queue_review:'+_key(body['runtime_directory'], store.directory, bridge['replay_month'])
                or value != _read(store.directory/'work_authority'/bridge['replay_month']/'typed_queue_review.json')
                or body['lease'] != lease.to_dict() or lease.lease_seconds != 300 or debit != counts(solve=1)
                or body['plan_hash'] != plan['hash'] or body['selection_hash'] != state['structural_plan']['hash']
                or body['authority_manifest_hash'] != authority.manifest_hash
                or body['data_id'] != state['data_id'] or body['source_id'] != source
                or body['catalogue_hash'] != _read_catalogue_identity(store.directory, bridge['replay_month'])['hash']
                or body['start'] != (month_start(pd.Timestamp(state['cutoff']))+pd.DateOffset(months=1)).isoformat()):
            raise ValueError('Typed queue lost its original catalogue/runtime/lease plan')
        from .neural_forward_ticket import _LifecycleReader
        reader = _LifecycleReader(body['runtime_directory'])
        try:
            if source != reader.plan.source_id or store.exposure.path != reader.exposure.path:
                raise ValueError('Queue original S and runtime exposure differ')
        finally:
            reader.close()
    finally:
        store.close()
    _runtime(authority, body['runtime_directory'])
    origin = FrozenWorkLease.from_dict(bridge['execution_envelope']['issuer'])
    if lease.wall_started_at[:7] != origin.wall_started_at[:7]:
        raise ValueError('Queue cannot borrow another actual month S owner')
    owner = _search(authority, bridge['bridge_id'], current=False, lease=origin)
    return {**owner, 'S_lane': owner['lane'], 'lane': 'nonlinear'}


def _read_catalogue_identity(directory, month):
    path = project_path(directory)/'algorithm_catalogue'/month/'identity.json'
    value = json.loads(path.read_text('utf-8'))
    if set(value) != {'value', 'hash'} or fingerprint(value['value']) != value['hash']:
        raise ValueError('Original catalogue identity checksum differs')
    return value


def produce_typed_queue_review(runtime, authority, run_directory, month):
    from .algorithm_validation import _runtime
    from .neural_research_ticket import _reader
    from .reserved_work_budget import ReservedChannelBudget
    from .structural_validation import _queue_numerical
    if authority.manifest.get('partition_policy') is None:
        raise ValueError('Typed queue requires actual nontransferable monthly partitions')
    root = project_path(run_directory)
    key = 'typed_queue_review:'+_key(runtime.directory, root, month)
    directory = authority.directory/'typed_queue_reviews'/key.split(':', 1)[1]
    if directory.exists():
        intent = _read(directory/'intent.json')
        rid = fingerprint({'authority': authority.manifest_hash, 'work_key': key})
        row = authority.get(rid)
        validate_queue_route(authority, key, row['binding'], FrozenWorkLease.from_dict(row['lease']), row['counts'])
        saved = _read(directory/'result.json')
        if (row['status'] != 'completed' or row['result'] != {'typed_queue_result_hash': saved['hash'], 'intent_hash': intent['hash'],
                    'numeric_closure_hash': _read(directory/'numeric_closure.json')['hash']}
                or row['actual_counts'] != counts(solve=1)):
            raise ValueError('Partial/failed typed queue review is consumed and cannot execute again')
        ReservedChannelBudget.verify_counts(authority, rid)
        verify_queue_closure(authority, directory, require_completed=True)
        return saved['body']
    # This original clock precedes source/plan validation and every numerical read.
    lease = FrozenWorkLease.issue(300)
    _runtime(runtime, authority)
    store = _reader(root)
    try:
        state = store.get(month)
        if state is None or state['status'] != 'complete':
            raise ValueError('Typed queue needs its actual completed S month')
        if 'catalogue' not in (state.get('nonlinear_report') or {}):
            return {'status': 'no_closed_full_learner', 'A0_changed': False, 'formal_claim': False}
        start = month_start(pd.Timestamp(state['cutoff']))+pd.DateOffset(months=1)
        if start <= runtime.lifecycle.now():
            return {'status': 'missed_original_next_month', 'start': start.isoformat(),
                    'A0_changed': False, 'formal_claim': False}
        bridge = json.loads((root/'work_authority'/month/'identity.json').read_text('utf-8'))['record']
        selection = state.get('structural_plan')
        if selection is None:
            raise ValueError('Typed structural queue requires completed paid structure review')
        body = {'version': VERSION, 'runtime_directory': str(runtime.directory), 'bridge': bridge,
            'authority_manifest_hash': authority.manifest_hash, 'source_id': store.identity['feed_identity'],
            'data_id': state['data_id'], 'selection_hash': selection['hash'],
            'plan_hash': state['nonlinear_work_plan']['hash'],
            'catalogue_hash': _read_catalogue_identity(root, month)['hash'],
            'start': start.isoformat(), 'lease': lease.to_dict()}
        intent = {'body': body, 'hash': fingerprint(body)}
        immutable_record(root/'work_authority'/month/'typed_queue_review.json', intent)
        immutable_record(directory/'intent.json', intent)
    finally:
        store.close()
    binding = {'role': 'S_typed_queue', 'queue_bridge_id': bridge['bridge_id'],
        'runtime_directory': str(runtime.directory), 'intent_hash': intent['hash']}
    record = authority._reserve_with_lease(key, counts={'solve': 1}, binding=binding, lease=lease)
    rid = record['reservation_id']
    try:
        authority.start(rid, {'pid': os.getpid(), 'created': psutil.Process().create_time(), 'mode': 'typed_queue_issuer'})
        _prepare_publication(directory, rid, lease)
        budget = ReservedChannelBudget.from_issuer_reservation(authority, rid)
        budget.consume('solve')
        result = {**_queue_numerical(runtime, authority, root, month), 'queue_review_reservation': rid}
        budget.check_time()
        from .algorithm_shortlist import AlgorithmCatalogue
        closure = freeze_numeric_closure(AlgorithmCatalogue(root/'algorithm_catalogue'/month), result)
        closure_value = {'body': closure, 'hash': fingerprint(closure)}
        immutable_record(directory/'numeric_closure.json', closure_value)
        immutable_record(root/'work_authority'/month/'numeric_closure.json', closure_value)
        budget.check_time()
        saved = {'body': result, 'hash': fingerprint(result)}
        immutable_record(directory/'result.json', saved)
        budget.finalize()
        lease.assert_parent_alive()
        closed = authority.complete(rid, result={'typed_queue_result_hash': saved['hash'], 'intent_hash': intent['hash'], 'numeric_closure_hash': closure_value['hash']},
            actual_counts=budget.actual_counts)
        if closed['status'] != 'completed':
            raise ValueError('Typed queue original work did not complete')
        _complete_publication(directory, closed, lease)
        return result
    except BaseException as error:
        try:
            if authority.get(rid)['status'] in {'reserved', 'running'}:
                authority.fail_consumed(rid, 'typed_queue_review:'+type(error).__name__)
        except BaseException as persistence_error:  # noqa: BLE001 - preserve original failure
            error.add_note('Typed queue closure persistence failed: '+repr(persistence_error))
        raise


def _sql_snapshot(path):
    import sqlite3
    from contextlib import closing
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro', uri=True)) as db:
        tables = db.execute("SELECT name,sql FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
        result = []
        for name, schema in tables:
            if not name.replace('_', '').isalnum():
                raise ValueError('Unknown original publication table')
            rows = db.execute('SELECT * FROM "'+name+'" LIMIT 257').fetchall()
            if len(rows) > 256:
                raise ValueError('Original publication row bound exceeded')
            result.append({'table': name, 'schema': schema, 'rows': rows})
    return json.loads(json.dumps(result, allow_nan=False))


def _artifact_snapshot(catalogue):
    """Exact immutable artifact graph, not the growing authority's whole SQL DB."""
    from .evidence import file_hash
    from .neural_research_ticket import ResearchFitTicket
    roots = {catalogue.directory}
    related = []
    authority = catalogue._context()
    with authority._transaction() as db:
        for row in authority._verify(db)['reservations'].values():
            if row['binding'].get('research_bridge_id') == catalogue.identity['bridge_id']:
                related.append({'id': row['reservation_id'], 'hash': fingerprint(row)})
    for path in sorted((catalogue.directory/'records').glob('*.json')):
        value = json.loads(path.read_text('utf-8'))['value']
        if value.get('ticket') is not None:
            ticket = ResearchFitTicket.load(value['ticket']['path'])
            roots.add(ticket.path.parent)
            roots.add(authority.directory/'child_actual'/ticket.value['reservation_id'])
    files, sql, root_files = {}, {}, {}
    pending = [authority.directory/'child_actual_claims'/(r['id']+'.json') for r in related
               if (authority.directory/'child_actual_claims'/(r['id']+'.json')).is_file()]
    for root in sorted(roots):
        if root.is_symlink() or not root.is_dir():
            raise ValueError('Original numeric artifact directory disappeared')
        children = sorted(p for p in root.rglob('*') if p.is_file())
        if len(children) > 4096:
            raise ValueError('Original numeric artifact bound exceeded')
        # SQLite transient locks are not immutable artifacts; table rows are.
        children = [p for p in children if not p.name.endswith(('-wal', '-shm', '.lock'))]
        root_files[str(root)] = [str(p.relative_to(root)) for p in children]
        pending.extend(children)
    seen, total = set(), 0
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError('Original numerical artifact missing or linked')
        seen.add(path); total += path.stat().st_size
        if len(seen) > 4096 or total > 256*2**20:
            raise ValueError('Numeric review exact artifact graph exceeds fixed bound')
        if path.suffix in {'.sqlite', '.db'}:
            sql[str(path)] = _sql_snapshot(path)
        else:
            files[str(path)] = file_hash(path)
        if path.name == 'ticket.json' and str(path.parent) not in root_files:
            extra = sorted(p for p in path.parent.rglob('*') if p.is_file() and not p.name.endswith(('-wal', '-shm', '.lock')))
            if len(extra) > 4096:
                raise ValueError('Gate artifact directory exceeds bound')
            root_files[str(path.parent)] = [str(p.relative_to(path.parent)) for p in extra]
            pending.extend(extra)
        if path.suffix == '.json':
            value = json.loads(path.read_text('utf-8'))
            stack = [value]
            while stack:
                item = stack.pop()
                if isinstance(item, dict):
                    stack.extend(item.values())
                elif isinstance(item, list):
                    stack.extend(item)
                elif isinstance(item, str) and len(item) < 1024:
                    from pathlib import Path
                    ref = Path(item)
                    if ref.is_absolute() and ref.suffix in {'.json', '.npy', '.npz', '.parquet'} and ref.is_file():
                        pending.append(ref)
    return {'files': dict(sorted(files.items())), 'publication_rows': sql,
            'roots': root_files, 'research_reservations': sorted(related, key=lambda r: r['id'])}


def freeze_numeric_closure(catalogue, result):
    # A final independent real numerical replay inside the already prepaid work.
    shortlist = catalogue.seal_shortlist()
    return {'version': VERSION, 'catalogue_id': catalogue.digest,
        'shortlist': shortlist, 'queue_result_hash': fingerprint(result),
        'artifacts': _artifact_snapshot(catalogue)}


def _queue_for_catalogue(catalogue):
    from .neural_research_ticket import _reader
    store = _reader(catalogue.identity['run_directory'])
    try:
        state = store.get(catalogue.identity['replay_month'])
        if state.get('structural_plan') is None:
            return None
        path = store.directory/'work_authority'/catalogue.identity['replay_month']/'typed_queue_review.json'
        intent = _read(path)
        key = _key(intent['body']['runtime_directory'], store.directory, catalogue.identity['replay_month'])
        return project_path(catalogue.identity['authority_directory'])/'typed_queue_reviews'/key
    finally:
        store.close()


def verify_queue_closure(authority, directory, *, require_completed):
    from .algorithm_shortlist import AlgorithmCatalogue
    from .reserved_work_budget import ReservedChannelBudget
    intent = _read(directory/'intent.json'); body = intent['body']
    key = 'typed_queue_review:'+directory.name
    rid = fingerprint({'authority': authority.manifest_hash, 'work_key': key})
    row = authority.get(rid)
    validate_queue_route(authority, key, row['binding'], FrozenWorkLease.from_dict(row['lease']), row['counts'])
    if row['status'] == 'running' and not require_completed:
        lease = FrozenWorkLease.from_dict(row['lease'])
        if row['process_identity']['pid'] != os.getpid() or lease.parent_pid != os.getpid():
            raise ValueError('Only actual current queue issuer may construct unpublished protocol')
        lease.assert_parent_alive()
        return None
    if row['status'] != 'completed':
        raise ValueError('Typed protocol lacks completed original paid queue closure')
    _verify_publication(directory, row)
    saved = _read(directory/'result.json')
    closure = _read(directory/'numeric_closure.json')
    local = project_path(body['bridge']['run_directory'])/'work_authority'/body['bridge']['replay_month']/'numeric_closure.json'
    if (_read(local) != closure or row['result'] != {'typed_queue_result_hash': saved['hash'],
            'intent_hash': intent['hash'], 'numeric_closure_hash': closure['hash']}
            or closure['body']['queue_result_hash'] != saved['hash'] or row['actual_counts'] != counts(solve=1)):
        raise ValueError('Original queue completion/result/closure mirrors changed')
    actual = ReservedChannelBudget.verify_counts(authority, rid)
    if not actual['finalized'] or actual['test_only']:
        raise ValueError('Queue lacks finalized original issuer execution')
    from .evidence import file_hash
    for name, digest in closure['body']['artifacts']['files'].items():
        path = project_path(name)
        if path.is_symlink() or not path.is_file() or file_hash(path) != digest:
            raise ValueError('Original independently reviewed artifact bytes changed')
    for name, expected in closure['body']['artifacts']['roots'].items():
        root = project_path(name)
        if root.is_symlink() or not root.is_dir():
            raise ValueError('Original artifact set directory missing')
        actual_names = sorted(str(p.relative_to(root)) for p in root.rglob('*')
            if p.is_file() and not p.name.endswith(('-wal', '-shm', '.lock')))
        if actual_names != expected:
            raise ValueError('Original independently reviewed artifact set changed')
    catalogue = AlgorithmCatalogue(project_path(body['bridge']['run_directory'])/'algorithm_catalogue'/body['bridge']['replay_month'])
    if (closure['body']['catalogue_id'] != catalogue.digest
            or closure['body']['artifacts'] != _artifact_snapshot(catalogue)
            or fingerprint(closure['body']['shortlist']) != json.loads((catalogue.directory/'shortlist.json').read_text('utf-8'))['hash']):
        raise ValueError('Original independently reviewed artifacts/rows/set changed')
    if saved['body']['status'] == 'queued_typed_algorithm_V_only':
        path = project_path(saved['body']['directory'])/'identity.json'
        value = json.loads(path.read_text('utf-8'))
        if value['hash'] != saved['body']['identity_hash'] or fingerprint(value['value']) != value['hash']:
            raise ValueError('Original queued protocol identity changed')
    return closure['body']['shortlist']


def checked_shortlist(catalogue, *, require_completed=False):
    """None means legacy; pending is a true same-issuer lease, never caller bool."""
    directory = _queue_for_catalogue(catalogue)
    if directory is None:
        return False, None
    authority = catalogue._context()
    value = verify_queue_closure(authority, directory, require_completed=require_completed)
    return True, value


def require_queue_completion(protocol):
    checked_shortlist(protocol.catalogue, require_completed=True)


def _prepare_publication(directory, rid, lease):
    import sqlite3
    from contextlib import closing
    path = directory/'publication.sqlite'
    if path.exists():
        raise ValueError('Queue publication is already prepared; no rerun')
    with closing(sqlite3.connect(path)) as db:
        db.execute('PRAGMA synchronous=FULL')
        db.execute('PRAGMA journal_mode=DELETE')
        db.execute('CREATE TABLE publication(id INTEGER PRIMARY KEY, status TEXT, reservation TEXT, lease_hash TEXT, result_hash TEXT, observation TEXT)')
        db.execute('INSERT INTO publication VALUES(1,?,?,?,?,?)', ('pending', rid, lease.digest, None, None))
        db.commit()


def _complete_publication(directory, row, lease):
    import sqlite3
    from contextlib import closing

    from .work_lease import _observe
    with closing(sqlite3.connect((directory/'publication.sqlite').as_uri()+'?mode=rw', uri=True)) as db:
        db.execute('PRAGMA synchronous=FULL')
        db.execute('BEGIN IMMEDIATE')
        if db.execute('SELECT status,reservation,lease_hash FROM publication WHERE id=1').fetchone() != ('pending', row['reservation_id'], lease.digest):
            raise ValueError('Original queue publication pending row changed')
        observed = _observe(lease)
        db.execute('UPDATE publication SET status=?,result_hash=?,observation=? WHERE id=1',
            ('completed', fingerprint(row['result']), json.dumps(observed, sort_keys=True, allow_nan=False)))
        # Final pre-COMMIT check plus independent post-COMMIT IO evidence.
        # Historical readers require both; a completed row alone is insufficient.
        lease.assert_parent_alive()
        _commit_publication(db)
    # Independent post-COMMIT observation is required by historical readers.
    # A slow commit never inherits the earlier check as a completion timestamp.
    observed = _observe(lease)
    lease.assert_parent_alive()
    body = {'reservation_id': row['reservation_id'], 'lease_hash': lease.digest,
            'result_hash': fingerprint(row['result']), 'observation': observed}
    immutable_record(directory/'final_io.json', {'body': body, 'hash': fingerprint(body)})
    # Publication is only successful after the final write has completed under
    # the same physical lease. Use the historical reader's canonical checks so
    # the producer cannot report a result its next reader would reject.
    _verify_publication(directory, row)
    lease.assert_parent_alive()


def _commit_publication(db):
    db.commit()


def _verify_publication(directory, row):
    import sqlite3
    from contextlib import closing
    lease = FrozenWorkLease.from_dict(row['lease'])
    publication = directory/'publication.sqlite'
    if publication.is_symlink() or not publication.is_file():
        raise ValueError('Missing original queue publication database')
    with closing(sqlite3.connect(publication.as_uri()+'?mode=ro', uri=True)) as db:
        if db.execute('SELECT COUNT(*) FROM publication').fetchone() != (1,):
            raise ValueError('Queue publication row set changed')
        actual = db.execute('SELECT status,reservation,lease_hash,result_hash,observation FROM publication WHERE id=1').fetchone()
    if (actual is None or actual[:4] != ('completed', row['reservation_id'], lease.digest, fingerprint(row['result']))):
        raise ValueError('Typed queue lacks canonical completed publication')
    observation = json.loads(actual[4])
    from .work_lease import _evaluate
    if not _evaluate(lease, observation).allowed:
        raise ValueError('Typed queue publication observation exceeded original lease')
    final = _read(directory/'final_io.json')['body']
    if (set(final) != {'reservation_id', 'lease_hash', 'result_hash', 'observation'}
            or final['reservation_id'] != row['reservation_id'] or final['lease_hash'] != lease.digest
            or final['result_hash'] != fingerprint(row['result'])
            or not _evaluate(lease, final['observation'], observation).allowed):
        raise ValueError('Typed queue final IO observation exceeded original lease')
    deadline = pd.Timestamp(lease.wall_deadline).value
    for path in (directory, directory/'publication.sqlite', directory/'final_io.json',
                 directory/'intent.json', directory/'result.json', directory/'numeric_closure.json'):
        stat = path.stat()
        if path.is_symlink() or max(stat.st_mtime_ns, stat.st_ctime_ns) >= deadline:
            raise ValueError('Typed queue final publication IO passed original deadline')
    if (directory/'publication.sqlite-journal').exists():
        raise ValueError('Typed queue publication has an unresolved DELETE journal')
