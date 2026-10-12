"""Registered nominal simulation observation and bounded library controller.

Logical DGP releases are not market receipts or UTC work months. Every import,
publication, work lease and independent physical batch retains native time.
No old joint permit is accepted. This is not a five-claim/GPT controller.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from filelock import FileLock

from factorlab.protocol import fingerprint, project_path
from .evidence import file_hash
from .monthly_work_authority import source_files
from .v7_storage import immutable_record
from .v7_lifecycle import stamp
from .calibration_runtime_contract import _original_operations

VERSION = 'nominal-observation-v1'
RELATIVE = 'factorlab/discovery/nominal_trial_controller.py'
BOUND = 'nominal_observation_bound'
ADVANCED = 'nominal_observation_advanced'
_BATCH = ContextVar('nominal_original_physical_batch', default=None)
_READ = ContextVar('nominal_original_prefix_read', default=None)


def _installed():
    files = source_files()
    if Path(__file__).resolve() != project_path(RELATIVE).resolve() or files.get(RELATIVE) != file_hash(__file__):
        raise ValueError('Canonical reviewed new controller source required')
    return files


def _write(path, record):
    immutable_record(path, {'record': record, 'hash': fingerprint(record)})


def _read(path):
    from .nominal_actual_trial import _read as read
    return read(path)


def _physical():
    from .quarantined_observation import _physical as observe
    return observe()


def _event(lifecycle, kind, detail, observation):
    from .quarantined_observation import _event as append
    with lifecycle.db:
        lifecycle.db.execute('BEGIN IMMEDIATE')
        return append(lifecycle, kind, detail, observation)


def _check(value, prior=None):
    from .quarantined_observation import _check_physical
    return _check_physical(value, prior)


def _original_trial(value):
    from .nominal_actual_trial import NominalActualTrial
    if type(value) is not NominalActualTrial:
        raise TypeError('Original independently drawn NominalActualTrial required')
    value.verify_draw()
    if value.registry.identity['actual_observation'] != 'registered_nominal_simulation':
        raise ValueError('Nominal observation is not a physical forward-time shortcut')
    if value.registry.identity['source_files'] != _installed():
        raise ValueError('Original nominal simulation source changed')
    return value


def _raw_source(trial):
    """Read already paid cold raw atom; do not regenerate/prefit for free."""
    from .nominal_cold_conformance import NominalCensus, NominalColdProtocol, read_month
    from .neural_forward_ticket import open_existing_authority
    census = NominalCensus(trial.registry.identity['census']['directory'])
    atom = trial.verify_draw()['atom']; paths = census.paths(atom)
    protocol = NominalColdProtocol(paths['protocol'])
    cutoff = census.identity['schedule']['cutoffs'][0]
    output = read_month(protocol, authority=open_existing_authority(paths['authority']),
                        atom=atom, cutoff=cutoff, role='reference')
    path = project_path(output['paid_pair_evidence']['directory'])/'raw.json'
    from .finite_market_oracle import _read as read_atom
    raw = read_atom(path)
    if (fingerprint(raw) != output['raw_hash'] or raw['atom'] != atom
            or raw['generator_digest'] != census.identity['generator_digest']
            or any(r['gap'] for r in raw['schedule'])):
        raise ValueError('Original paid nominal atom differs from registered draw')
    return census, path, raw


class NominalObservation:
    """Source-bound append-only logical prefixes with separate physical receipts."""
    def __init__(self, trial):
        self.trial = _original_trial(trial)
        self.directory = trial.paths['root']/'nominal_observation'
        self.mirror = trial.registry.directory/'nominal_observations'/trial.digest
        self.identity = _read(self.directory/'identity.json')
        if self.identity != _read(self.mirror/'identity.json') or self.identity['source_files'] != _installed():
            raise ValueError('Original observation identity mirrors/source differ')
        self.digest = fingerprint(self.identity)
        # Compatibility data for the existing snapshot reader; this object is
        # a NEW typed trial, never a QuarantinedTrialPermit or old enrollment.
        self.permit = trial
        from .finite_market_oracle import FiniteGenerator, FiniteMarketLaw, _read as read_generator
        from .nominal_actual_trial import _acceptance
        generator = FiniteGenerator.__new__(FiniteGenerator)
        generator.directory = project_path(self.identity['generator_directory'])
        generator.identity = read_generator(generator.directory/'identity.json')
        fields = dict(generator.identity['law']); fields['symbols'] = tuple(fields['symbols'])
        generator.law = FiniteMarketLaw(**fields); generator.acceptance = _acceptance(generator.identity['acceptance'])
        generator.verify(); self.generator = generator
        if (generator.digest != self.identity['generator_digest']
                or self.identity['trial_digest'] != trial.digest
                or self.identity['coordinate'] != trial.coordinate
                or self.identity['runtime_directory'] != str(trial.paths['runtime'])):
            raise ValueError('Original observation trial/generator mapping changed')

    @classmethod
    def bind(cls, trial, runtime):
        from .v7_forward import ForwardRuntime
        trial = _original_trial(trial)
        if type(runtime) is not ForwardRuntime or runtime.directory != trial.paths['runtime']:
            raise TypeError('Actual original trial ForwardRuntime required')
        from .nominal_actual_trial import _bound
        bound = _bound(runtime.lifecycle)
        if bound is None or bound[0]['coordinate'] != trial.coordinate:
            raise ValueError('Actual trial enrollment missing')
        census, raw_path, raw = _raw_source(trial)
        directory = trial.paths['root']/'nominal_observation'
        mirror = trial.registry.directory/'nominal_observations'/trial.digest
        with runtime.lifecycle.controller_lock(), FileLock(str(directory)+'.creation.lock', timeout=0):
            if directory.exists() or mirror.exists():
                raise ValueError('Observation binding already attempted; never reset')
            if (runtime.feed.db.execute('SELECT COUNT(*) FROM batches').fetchone()[0]
                    or runtime.lifecycle.db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]):
                raise ValueError('Observation must precede all source exposure and work')
            value = {'version': VERSION, 'trial_digest': trial.digest, 'coordinate': trial.coordinate,
                'registry_directory': str(trial.registry.directory), 'runtime_directory': str(runtime.directory),
                'source_files': _installed(), 'generator_directory': census.identity['generator_directory'],
                'generator_digest': census.identity['generator_digest'], 'raw_path': str(raw_path),
                'raw_hash': fingerprint(raw), 'raw_sha256': file_hash(raw_path), 'atom': raw['atom'],
                'schedule': census.identity['schedule'], 'enrollment_hash': fingerprint(bound[0]),
                'clock_contract': 'registered_synthetic_logical_release_and_native_physical_IO_and_work',
                'formal_acceptance': False}
            for path in (directory, mirror):
                path.mkdir(parents=True); _write(path/'identity.json', value)
            observation = _physical(); _check(observation)
            op = _event(runtime.lifecycle, BOUND, {'identity_hash': fingerprint(value),
                'directory': str(directory), 'physical_observation': observation}, observation)
            for path in (directory, mirror): _write(path/'bound.json', op)
        return cls(trial)

    def verify_actual_target(self, protocol, authority, atom, directory):
        from .nominal_cold_conformance import NominalCensus
        census = NominalCensus(self.trial.registry.identity['census']['directory'])
        drawn = self.trial.verify_draw()['atom']; paths = census.paths(drawn)
        if (atom != drawn or Path(directory).resolve() != self.trial.paths['runtime']
                or protocol.directory != paths['protocol'] or authority.directory != paths['authority']
                or self.identity['schedule'] != census.identity['schedule']):
            raise ValueError('Actual conformance must use its original drawn atom and canonical runtime')
        return self.trial.coordinate

    def raw(self):
        from .finite_market_oracle import _read as read_generator
        path = project_path(self.identity['raw_path'])
        raw = read_generator(path)
        if file_hash(path) != self.identity['raw_sha256'] or fingerprint(raw) != self.identity['raw_hash']:
            raise ValueError('Original paid raw source changed')
        return raw

    def _prefix(self, hour, operation, source):
        return {'version': VERSION, 'trial_digest': self.trial.digest,
            'registry_digest': self.trial.registry.digest, 'identity_hash': self.digest,
            'hour': hour, 'logical_at': source['logical_at'], 'event_hash': fingerprint(source),
            'operation_hash': fingerprint(operation), 'physical_observation': operation['detail']['physical_observation'],
            'availability': 'registered_nominal_synthetic_prefix_not_market_receipt',
            'formal_gate_eligible': False, 'work_budget_granted': False}

    def inspect(self, lifecycle):
        scope = _READ.get()
        if scope is not None and scope['directory'] == str(lifecycle.directory) and scope.get('state') is not None:
            return scope['state']
        if lifecycle.directory != self.trial.paths['runtime']:
            raise ValueError('Observation moved to another lifecycle')
        operations = _original_operations(lifecycle)
        bound = [o for o in operations if o['kind'] == BOUND]
        if len(bound) != 1 or any(_read(p/'bound.json') != bound[0] for p in (self.directory, self.mirror)):
            raise ValueError('Original nominal observation binding changed')
        if bound[0]['detail']['identity_hash'] != self.digest:
            raise ValueError('Original nominal observation identity changed')
        raw = self.raw(); prior = bound[0]['detail']['physical_observation']; _check(prior)
        advanced = [o for o in operations if o['kind'] == ADVANCED]
        head = fingerprint(bound[0]); prefix = None; batch_ids = []
        for hour, op in enumerate(advanced):
            name = f'{hour:05d}.json'
            entry = _read(self.directory/'events'/name)
            source = entry['source']; pub = self._publication(hour, raw)
            if (entry != _read(self.mirror/'events'/name)
                    or op != _read(self.directory/'commits'/name) or op != _read(self.mirror/'commits'/name)
                    or entry['previous'] != head
                    or source != {'hour': hour, 'logical_at': raw['schedule'][hour]['release_not_before'],
                        'scheduled_gap': False, 'publication_hash': fingerprint(pub)}
                    or op['detail']['event_hash'] != fingerprint(entry)
                    or op['detail']['identity_hash'] != self.digest):
                raise ValueError('Original nominal observation event chain differs')
            physical = op['detail']['physical_observation']; wall = _check(physical, prior)
            if stamp(pub['published_at']) > wall or stamp(pub['received_at']) > wall or op['at'] != physical['wall']:
                raise ValueError('Original physical publication/observation order differs')
            prefix = self._prefix(hour, op, source); head = fingerprint(op); prior = physical
            batch_ids.append(pub['receipt']['batch'])
        names = {f'{h:05d}.json' for h in range(len(advanced))}
        for path in (self.directory, self.mirror):
            for folder in ('events', 'commits'):
                if {p.name for p in (path/folder).glob('*.json')} != names:
                    raise ValueError('Partial or extra prefix cannot recover as completed')
        import sqlite3
        from contextlib import closing
        with closing(sqlite3.connect((self.trial.paths['feed']/'receipts.sqlite').as_uri()+'?mode=ro', uri=True)) as db:
            if {r[0] for r in db.execute('SELECT id FROM batches')} != set(batch_ids):
                raise ValueError('Feed contains an unregistered or incomplete source publication')
        state = {'events': len(advanced), 'head': head, 'identity_hash': self.digest,
            'latest_binding': prefix, 'logical_at': None if prefix is None else prefix['logical_at'],
            'physical_observation': prior, 'batch_ids': batch_ids}
        if scope is not None and scope['directory'] == str(lifecycle.directory): scope['state'] = state
        return state

    def _publication(self, hour, raw=None):
        raw = self.raw() if raw is None else raw
        left = self.trial.paths['root']/'generator_publications'
        value = _read(left/f'{hour:05d}-complete.json')
        if value != _read(self.mirror/'receipts'/f'{hour:05d}-complete.json'):
            raise ValueError('Nominal physical publication mirrors differ')
        n = len(self.generator.law.symbols); rows = raw['rows'][hour*n:(hour+1)*n]
        intent = _read(left/f'{hour:05d}-intent.json')
        if (intent != _read(self.mirror/'receipts'/f'{hour:05d}-intent.json')
                or intent['trial_digest'] != self.trial.digest or intent['hour'] != hour
                or intent['rows_hash'] != fingerprint(rows) or value['intent_hash'] != fingerprint(intent)
                or value['received_at'] != intent['received_at']):
            raise ValueError('Original nominal source rows/draw/physical import changed')
        import sqlite3
        from contextlib import closing
        from .v7_feed import ReceiptArchive
        feed = ReceiptArchive.__new__(ReceiptArchive); feed.directory = self.trial.paths['feed']
        feed.identity = json.loads((feed.directory/'identity.json').read_text('utf-8'))['source_id']
        with closing(sqlite3.connect((feed.directory/'receipts.sqlite').as_uri()+'?mode=ro', uri=True)) as db:
            feed.db = db; db.execute('PRAGMA query_only=ON')
            row = db.execute('SELECT record,hash FROM batches WHERE id=?', (value['receipt']['batch'],)).fetchone()
            if row is None: raise ValueError('Original physical source batch missing')
            batch = json.loads(row[0])
            raw_batch = json.loads((feed.directory/'raw'/(row[1]+'.json')).read_text('utf-8'))
            if raw_batch != batch:
                raise ValueError('Original source raw receipt differs from database')
            if (fingerprint(batch) != row[1] or row[1] != value['receipt']['batch']
                    or feed._batch_bars(batch) != rows or batch['source_ref'] != {'nominal_publication': fingerprint(intent)}
                    or batch['received_at'] != value['received_at'] or batch['actual'] is not True
                    or feed._publication(row[1], batch) != value['published_at']):
                raise ValueError('Original synthetic receipt/CAS/publication differs')
        return value

    def source_event(self, hour):
        value = _read(self.directory/'events'/f'{hour:05d}.json')
        if value != _read(self.mirror/'events'/f'{hour:05d}.json'):
            raise ValueError('Original source event mirrors differ')
        return value['source']

    def binding(self, hour, lifecycle=None):
        if lifecycle is None:
            from .neural_forward_ticket import _LifecycleReader
            reader = _LifecycleReader(self.trial.paths['runtime'])
            try: return self.binding(hour, reader)
            finally: reader.close()
        state = self.inspect(lifecycle)
        if type(hour) is not int or not 0 <= hour < state['events']:
            raise ValueError('Original committed nominal prefix required')
        op = _read(self.directory/'commits'/f'{hour:05d}.json')
        return self._prefix(hour, op, _read(self.directory/'events'/f'{hour:05d}.json')['source'])

    def verify_binding(self, prefix):
        if type(prefix) is not dict or prefix != self.binding(prefix.get('hour')):
            raise ValueError('Nominal prefix changed or belongs to another trial')
        return prefix

    def publish_advance(self, runtime):
        batch = _BATCH.get()
        if batch is None or batch.trial.digest != self.trial.digest:
            raise ValueError('Source publication requires original active physical batch')
        batch.check()
        with runtime.lifecycle.controller_lock(), FileLock(str(self.directory)+'.lock', timeout=0):
            state = self.inspect(runtime.lifecycle); hour = state['events']; raw = self.raw()
            if hour >= len(raw['schedule']): raise ValueError('Registered source exhausted')
            n = len(self.generator.law.symbols); rows = raw['rows'][hour*n:(hour+1)*n]
            roots = (self.trial.paths['root']/'generator_publications', self.mirror/'receipts')
            if any((r/f'{hour:05d}-intent.json').exists() for r in roots):
                raise ValueError('Original publication already attempted; no retry')
            physical = _physical(); _check(physical, state['physical_observation'])
            intent = {'version': VERSION, 'trial_digest': self.trial.digest, 'hour': hour,
                'rows_hash': fingerprint(rows), 'received_at': physical['wall'], 'physical_observation': physical, 'physical_batch': batch.binding,
                'logical_release': raw['schedule'][hour]['release_not_before'], 'market_data': False}
            for r in roots: _write(r/f'{hour:05d}-intent.json', intent)
            permit = NominalPublication(self, intent)
            frame = pd.DataFrame(rows); frame['open_time'] = pd.to_datetime(frame.open_time, utc=True)
            receipt = runtime.feed._commit(frame, received_at=physical['wall'], actual=True,
                source_ref={'nominal_publication': fingerprint(intent)}, nominal_source=permit)
            batch.check()
            raw_batch = json.loads((runtime.feed.directory/'raw'/(receipt['batch']+'.json')).read_text('utf-8'))
            value = {'intent_hash': fingerprint(intent), 'receipt': receipt, 'received_at': physical['wall'],
                'published_at': runtime.feed._publication(receipt['batch'], raw_batch)}
            for r in roots: _write(r/f'{hour:05d}-complete.json', value)
            entry = {'previous': state['head'], 'source': {'hour': hour,
                'logical_at': intent['logical_release'], 'scheduled_gap': False, 'publication_hash': fingerprint(value)}}
            for r in (self.directory, self.mirror): _write(r/'events'/f'{hour:05d}.json', entry)
            physical = _physical(); _check(physical, state['physical_observation'])
            op = _event(runtime.lifecycle, ADVANCED, {'identity_hash': self.digest,
                'event_hash': fingerprint(entry), 'physical_observation': physical}, physical)
            for r in (self.directory, self.mirror): _write(r/'commits'/f'{hour:05d}.json', op)
            batch.check()
            return self.binding(hour, runtime.lifecycle)


class NominalPublication:
    """Only a real active registered source allocation can use private feed I/O."""
    def __init__(self, observation, intent):
        if type(observation) is not NominalObservation:
            raise TypeError('Actual source-bound observation required')
        self.observation, self.intent = observation, intent

    def verify(self, feed, frame, received_at, actual, source_ref):
        batch = _BATCH.get(); o = self.observation; i = self.intent
        if batch is None or batch.trial.digest != o.trial.digest:
            raise ValueError('No original nominal batch allocation')
        batch.check()
        rows = o.raw()['rows']; n = len(o.generator.law.symbols); expected = rows[i['hour']*n:(i['hour']+1)*n]
        actual_rows = [{**r, 'open_time': stamp(r['open_time']).isoformat()} for r in frame.to_dict('records')]
        if (feed.directory != o.trial.paths['feed'] or actual is not True
                or actual_rows != expected or i['rows_hash'] != fingerprint(expected)
                or stamp(received_at).isoformat() != i['received_at']
                or source_ref != {'nominal_publication': fingerprint(i)} or i['physical_batch'] != batch.binding
                or i != _read(o.trial.paths['root']/'generator_publications'/f"{i['hour']:05d}-intent.json")):
            raise ValueError('Synthetic publication changed its actual original allocation/input')
        if i != _read(o.mirror/'receipts'/f"{i['hour']:05d}-intent.json"):
            raise ValueError('Publication allocation dual anchor differs')
        for text, in feed.db.execute('SELECT record FROM batches'):
            if json.loads(text).get('source_ref') == source_ref:
                raise ValueError('Nominal publication already consumed; cannot retry')
        if i['physical_observation']['wall'] != i['received_at']:
            raise ValueError('Original physical import observation changed')
        _check(i['physical_observation'])
        return True


def from_lifecycle(lifecycle):
    operations = _original_operations(lifecycle)
    records = [o for o in operations if o['kind'] == BOUND]
    if not records: return None
    if len(records) != 1 or any(o['kind'] == 'quarantined_observation_bound' for o in operations):
        raise ValueError('Old and nominal observation cannot share a lifecycle')
    value = _read(project_path(records[0]['detail']['directory'])/'identity.json')
    from .nominal_actual_trial import NominalActualRegistry, NominalActualTrial, EVENT
    registry = NominalActualRegistry(value['registry_directory']); c = value['coordinate']
    trial = NominalActualTrial(registry, c['cell'], c['index'])
    enrollment = _read(trial.paths['root']/'enrollment.json')
    anchors = [o for o in operations if o['kind'] == EVENT]
    if (len(anchors) != 1 or enrollment != _read(trial.paths['authority']/'nominal_actual_trial.json')
            or fingerprint(enrollment) != value['enrollment_hash']
            or anchors[0]['detail']['enrollment_hash'] != value['enrollment_hash']
            or trial.paths['runtime'] != lifecycle.directory or fingerprint(value) != records[0]['detail']['identity_hash']):
        raise ValueError('Original nominal observation enrollment/draw differs')
    return NominalObservation(trial)


def current_binding(lifecycle):
    obj = from_lifecycle(lifecycle)
    return None if obj is None else obj.inspect(lifecycle)['latest_binding']


def job_binding(lifecycle, job):
    obj = from_lifecycle(lifecycle)
    if obj is None: return None
    records = [o for o in _original_operations(lifecycle) if o['kind'] == 'fit_planned' and o['detail']['job'] == job]
    if len(records) != 1: raise ValueError('Original nominal planning operation missing')
    prefix = records[0]['detail'].get('observation_prefix'); obj.verify_binding(prefix)
    if records[0]['at'] != prefix['logical_at'] or stamp(records[0]['detail']['cutoff']) > stamp(prefix['logical_at']):
        raise ValueError('Actual model planning changed original logical prefix')
    return prefix


@contextmanager
def observation_read_scope(lifecycle):
    obj = from_lifecycle(lifecycle)
    if obj is None:
        yield; return
    parent = _READ.get()
    if parent is not None:
        if parent['directory'] != str(lifecycle.directory): raise ValueError('Another trial cannot borrow read scope')
        yield; return
    with FileLock(str(obj.directory)+'.lock', timeout=0):
        scope = {'directory': str(lifecycle.directory), 'state': None}
        token = _READ.set(scope)
        try: yield
        finally:
            _READ.reset(token)
            if scope['state'] is not None and obj.inspect(lifecycle) != scope['state']:
                raise ValueError('Nominal source/prefix changed inside original paid invocation')


def publication_visible(lifecycle, kind, key, logical_cutoff):
    from .quarantined_observation import read_publication
    obj = from_lifecycle(lifecycle)
    if obj is None: return None
    value = read_publication(lifecycle, kind, key)
    if value is None: return None
    state = obj.inspect(lifecycle)
    for hour in range(state['events']):
        prefix = obj.binding(hour, lifecycle)
        if stamp(prefix['logical_at']) >= stamp(logical_cutoff):
            return (stamp(value['observation_prefix']['logical_at']) <= stamp(logical_cutoff)
                and stamp(value['physical_observation']['wall']) <= stamp(prefix['physical_observation']['wall']))
    return False


def require_batch_work(trial, cutoff, role):
    batch = _BATCH.get()
    if batch is None or batch.trial.digest != trial.digest:
        raise ValueError('Simulation fit requires its original running registered physical batch')
    batch.check(new_fit=True)
    if trial.check_work_batch(cutoff, role) != batch.plan:
        raise ValueError('Original trial issuer physical allocation differs')
    if {'cutoff': cutoff, 'role': role} not in batch.plan['model_keys']:
        raise ValueError('Fit cannot move into a different physical month/batch')
    return batch.binding


class _PhysicalBatch:
    def __init__(self, trial, index):
        from .full_trial_controller import ControllerStepLease
        self.trial = _original_trial(trial)
        if type(index) is not int or not 0 <= index < len(trial.registry.identity['work_batches']):
            raise ValueError('Original registered physical batch index required')
        self.plan = trial.registry.identity['work_batches'][index]
        self.directory = trial.paths['root']/'controller_batches'/f'{index:02d}'
        self.mirror = trial.registry.directory/'controller_batches'/trial.digest/f'{index:02d}'
        if pd.Timestamp.now(tz='UTC').strftime('%Y-%m') != self.plan['actual_UTC_month']:
            raise ValueError('Physical UTC batch is not due; no work or clock allocation')
        if self.directory.exists() or self.mirror.exists():
            raise ValueError('Physical batch already attempted; never resume/retry its old lease')
        self.clock = ControllerStepLease.issue(SimpleNamespace(service={'wall_seconds': self.plan['service_seconds_ceiling']}))
        self.native = None
        self.binding = {'trial_digest': trial.digest, 'batch': self.plan, 'clock': self.clock.to_dict()}
        for p in (self.directory, self.mirror): p.mkdir(parents=True); _write(p/'intent.json', self.binding)
        self.check()
        from .nominal_trial_native_completion import bind_current_batch
        self.native = bind_current_batch(self)

    def check(self, new_fit=False):
        check = self.clock.assert_parent_alive()
        if pd.Timestamp.now(tz='UTC').strftime('%Y-%m') != self.plan['actual_UTC_month']:
            raise ValueError('Work cannot cross physical UTC wallet month')
        if new_fit and check.remaining_seconds <= 302:
            raise ValueError('Full original fit lease no longer fits physical batch')
        if any(_read(p/'intent.json') != self.binding for p in (self.directory, self.mirror)):
            raise ValueError('Original physical batch identity changed')
        if new_fit and self.native is not None:
            self.native.check_live()
        return check


def verify_completed_batch(trial, index):
    """Reconstruct native deadline, immutable mirrors, model keys and prefix."""
    from .full_trial_controller import ControllerStepLease
    from .nonlinear_launcher import _validate_kernel
    trial = _original_trial(trial)
    if type(index) is not int or not 0 <= index < len(trial.registry.identity['work_batches']):
        raise ValueError('Registered batch index required')
    plan = trial.registry.identity['work_batches'][index]
    roots = (trial.paths['root']/'controller_batches'/f'{index:02d}',
             trial.registry.directory/'controller_batches'/trial.digest/f'{index:02d}')
    if any((p/name).exists() for p in (trial.paths['root'], *roots)
           for name in ('failure.json', 'prefix-session-failure.json')):
        raise ValueError('Any original root or batch failure consumes the trial')
    values = {}
    for name in ('intent.json', 'terminal.json', 'successor.json'):
        values[name] = _read(roots[0]/name)
        if values[name] != _read(roots[1]/name): raise ValueError('Physical batch mirrors differ')
    binding, terminal, successor = [values[n] for n in ('intent.json','terminal.json','successor.json')]
    if (binding['trial_digest'] != trial.digest or binding['batch'] != plan
            or terminal['binding'] != binding or terminal['completed_keys'] != plan['model_keys']
            or terminal['state'] != 'completed_independent_batch'
            or successor['terminal_hash'] != fingerprint(terminal)):
        raise ValueError('Original completed physical work allocation differs')
    clock = ControllerStepLease.from_dict(binding['clock'], SimpleNamespace(service={'wall_seconds':plan['service_seconds_ceiling']}))
    if not clock.evaluate(terminal['physical_finish']).allowed or not clock.evaluate(successor['observed'],terminal['physical_finish']).allowed:
        raise ValueError('Original physical batch did not finish within its deadline')
    _validate_kernel(terminal['kernel_before'], terminal['kernel_after'])
    observation = NominalObservation(trial)
    observation.verify_binding(terminal['last_prefix'])
    for i, key in enumerate(plan['model_keys']):
        record = _read(roots[0]/f'model-{i:02d}.json')
        if record != _read(roots[1]/f'model-{i:02d}.json') or record['key'] != key:
            raise ValueError('Actual completed model receipt differs')
        observation.verify_binding(record['prefix']); _check(record['physical'])
        if stamp(record['physical']['wall']).strftime('%Y-%m') != plan['actual_UTC_month']:
            raise ValueError('Model moved physical work month')
    from .neural_forward_ticket import _LifecycleReader
    lc = _LifecycleReader(trial.paths['runtime'])
    try:
        lc.verify()
        from .nominal_actual_trial import STREAMS
        for key in plan['model_keys']:
            rows = lc.db.execute('SELECT status,failure,id,model,pipeline FROM jobs WHERE cutoff=? AND stream=?',
                                (key['cutoff'], STREAMS[key['role']])).fetchall()
            if len(rows) != 1 or rows[0][0] not in ('verified', 'activated') or rows[0][1] is not None:
                raise ValueError('Actual admitted model missing or ambiguous')
            record = _read(roots[0]/f"model-{plan['model_keys'].index(key):02d}.json")
            if record['result']['job'] != rows[0][2] or record['result']['model_id'] != rows[0][3]:
                raise ValueError('Recorded model result does not name the original job')
            _, identity = lc.registry.load(rows[0][3], expected_pipeline=json.loads(rows[0][4]))
            if identity['job'] != rows[0][2] or identity['version'] != (3 if key['role']=='reference' else 4):
                raise ValueError('Original paid model admission differs')
    finally: lc.close()
    return terminal


def run_batch(trial, index):
    """Future executable library simulation; separate preregistered work batches.

    No polling/network, injected clock, speculative model or old-service resume.
    A failed/partial batch consumes the trial; only completed earlier batches
    allow the next explicitly registered physical batch to start.
    """
    from .nominal_actual_trial import _read as read_trial
    from .v7_forward import ForwardRuntime, ForwardConfig
    from .v7_lifecycle import LifecyclePlan
    from .nonlinear_launcher import _kernel_snapshot, _validate_kernel
    from .neural_forward_launcher import _identity, _member
    trial = _original_trial(trial)
    if type(index) is not int:
        raise TypeError('Registered integer batch index required')
    if not 0 <= index < len(trial.registry.identity['work_batches']):
        raise ValueError('Original registered physical batch index required')
    if (trial.paths['root']/'failure.json').exists():
        raise ValueError('Failed trial remains consumed; no continuation')
    for attempted_index in range(index+1):
        anchors = (trial.paths['root']/'controller_batches'/f'{attempted_index:02d}',
                   trial.registry.directory/'controller_batches'/trial.digest/f'{attempted_index:02d}')
        if any((p/name).exists() for p in anchors
               for name in ('failure.json', 'prefix-session-failure.json')):
            raise ValueError('Any original batch failure consumes the trial')
    for previous_index in range(index):
        verify_completed_batch(trial, previous_index)
        from .nominal_trial_native_completion import read_retained_batch
        read_retained_batch(trial.registry, trial.cell, trial.index, previous_index).assert_current()
    enrolled = read_trial(trial.paths['root']/'enrollment.json')
    cfg = dict(enrolled['runtime_identity']['config']); cfg['symbols'] = tuple(cfg['symbols'])
    lp = dict(enrolled['runtime_identity']['lifecycle']); lp['symbols'] = tuple(lp['symbols'])
    batch = _PhysicalBatch(trial, index); token = _BATCH.set(batch)
    runtime = None
    try:
        batch.native.check_live()
        first = _kernel_snapshot(); _validate_kernel(first, first); _member(first, _identity())
        runtime = ForwardRuntime(trial.paths['runtime'], trial.paths['feed'], config=ForwardConfig(**cfg),
                                 lifecycle_plan=LifecyclePlan(**lp))
        observer = from_lifecycle(runtime.lifecycle)
        if observer is None: observer = NominalObservation.bind(trial, runtime)
        schedule = observer.identity['schedule']; keys = batch.plan['model_keys']; completed = []
        last_batch = index == len(trial.registry.identity['work_batches'])-1
        from .nominal_prefix_session import NominalPrefixSession
        with NominalPrefixSession.open_original(trial, runtime, batch) as session:
            while True:
                batch.check()
                with session.append():
                    prefix = observer.publish_advance(runtime)
                now = stamp(prefix['logical_at'])
                # Warmup rows are actual published synthetic source; no model can
                # exist before the first registered cold cutoff.
                if now < stamp(schedule['cutoffs'][0]): continue
                with session.read_epoch(), observation_read_scope(runtime.lifecycle):
                    if now < stamp(schedule['start']):
                        # Warmboot activation is registered even before serving
                        # begins; the full six-month guard checks these exact times.
                        from .nominal_actual_trial import STREAMS
                        for role in ('reference', 'candidate'):
                            runtime._activate_ready(STREAMS[role], now, expected_pipeline=enrolled['pipelines'][role])
                    runtime.tick(poll=False)
                    remaining = [k for k in keys if k not in completed and stamp(k['cutoff']) <= now]
                    if remaining:
                        key = remaining[0]; require_batch_work(trial, key['cutoff'], key['role'])
                        from .known_linear_reference import FrozenLinearRecipe
                        from .joint_pool import FrozenPool
                        from .neural_forward_ticket import open_existing_authority
                        from .nominal_actual_trial import STREAMS
                        recipe = (FrozenLinearRecipe('HAR_cal', runtime.spec) if key['role'] == 'reference'
                                  else FrozenPool(role='reference', version='K0_v1'))
                        result = runtime.fit_month(key['cutoff'], stream=STREAMS[key['role']], pool=recipe,
                            authority=open_existing_authority(trial.paths['authority']))
                        completed.append(key)
                        model_record = {'key': key, 'prefix': prefix, 'result': result, 'physical': _physical()}
                        for p in (batch.directory, batch.mirror):
                            _write(p/f'model-{len(completed)-1:02d}.json', model_record)
                if completed == keys and not last_batch: break
                if now >= stamp(schedule['end'])+pd.Timedelta(hours=36):
                    if completed != keys: raise ValueError('Required actual cold work absent at fixed endpoint')
                    break
            batch.check()
            outcome = None
            if last_batch:
                claims = runtime.claim_protocol(); claims.finalize('library_skill')
                from .nominal_cold_conformance import NominalCensus, NominalColdProtocol, observe_actual
                from .neural_forward_ticket import open_existing_authority
                census = NominalCensus(trial.registry.identity['census']['directory'])
                atom = trial.verify_draw()['atom']; paths = census.paths(atom)
                outcome = observe_actual(NominalColdProtocol(paths['protocol']),
                    authority=open_existing_authority(paths['authority']), atom=atom, actual_directory=runtime.directory)
                if not outcome['conformant']: raise ValueError('Actual full nominal conformance failed: '+outcome.get('failure',''))
        last = _kernel_snapshot(); _validate_kernel(first, last); _member(last, _identity()); batch.check()
        terminal = {'state': 'completed_independent_batch', 'binding': batch.binding, 'completed_keys': completed,
            'last_prefix': current_binding(runtime.lifecycle), 'kernel_before': first, 'kernel_after': last,
            'outcome': outcome, 'physical_finish': batch.clock.observe(), 'formal_acceptance': False}
        for p in (batch.directory, batch.mirror): _write(p/'terminal.json', terminal)
        batch.check()
        successor = {'terminal_hash': fingerprint(terminal), 'observed': batch.clock.observe()}
        for p in (batch.directory, batch.mirror): _write(p/'successor.json', successor)
        batch.check()
        batch.native.publish_completed(terminal, successor)
        return terminal
    except BaseException as error:
        failure = {'consumed': True, 'reason': type(error).__name__+':'+str(error)[:2000],
                   'family_error': 1, 'power': 0, 'retain_N': True}
        for p in (batch.directory, batch.mirror): _write(p/'failure.json', failure)
        if not (trial.paths['root']/'failure.json').exists():
            _write(trial.paths['root']/'failure.json', failure)
        raise
    finally:
        _BATCH.reset(token)
        if runtime is not None: runtime.close()


def registered_outcome(registry, cell, index):
    """One original N coordinate. Missing work never disappears from N.

    Only library boundary truth is implemented. The four unimplemented slots
    retain conservative family_error=1; no five-claim acceptance is inferred.
    """
    from .nominal_actual_trial import NominalActualRegistry, NominalActualTrial
    if type(registry) is not NominalActualRegistry:
        raise TypeError('Original typed full-N registry required')
    coordinate = registry.coordinate(cell, index)
    base = {'coordinate':coordinate, 'retain_N':True, 'family_error':1,
            'power':0, 'library_error':1, 'complete_library':False,
            'formal_acceptance':False, 'pending_slots':registry.identity['pending_slots']}
    try:
        trial = NominalActualTrial(registry,cell,index)
        if (trial.paths['root']/'failure.json').exists():
            raise ValueError('Original trial failed and remains consumed')
        for batch in registry.identity['work_batches']:
            terminal = verify_completed_batch(trial,batch['index'])
        from .nominal_cold_conformance import NominalCensus, NominalColdProtocol, observe_actual
        from .neural_forward_ticket import open_existing_authority
        census = NominalCensus(registry.identity['census']['directory'])
        atom = trial.verify_draw()['atom']; paths = census.paths(atom)
        fresh = observe_actual(NominalColdProtocol(paths['protocol']),
            authority=open_existing_authority(paths['authority']),atom=atom,actual_directory=trial.paths['runtime'])
        if fresh != terminal['outcome'] or not fresh['conformant']:
            raise ValueError('Original final conformance no longer verifies')
        return {**base, 'complete_library':True, 'library_error':None,
                'guarded_numerical_event':fresh['guarded_numerical_event'],
                'truth_classification':'pending_typed_exact_nominal_law_adapter',
                'tensor_hash':fresh['tensor_hash'], 'statistic':fresh['statistic']}
    except Exception as error:
        return {**base,'failure':type(error).__name__+':'+str(error)[:2000]}
