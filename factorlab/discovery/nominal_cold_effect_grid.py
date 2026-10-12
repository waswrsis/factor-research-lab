"""Metered complete nominal cold census -> fixed prospective library grid.

Source-bound semantic consumers of the new verification quota. Header reads
are not full truth verification. Nothing here grants fits, draws or acceptance.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, UTC

from factorlab.protocol import project_path, fingerprint
from .evidence import file_hash
from .monthly_work_authority import source_files
from .nominal_verification_budget import (
    VerificationRegistry, VerificationService, VerificationTask, _path, _read, _write)
from .nominal_cold_law import _original_atom
from .nominal_effect_grid import derive_effect_grid, grid_payload
from .nominal_effect_design import DESIGNS, scenario_design, scenario_design_id
from .work_lease import FrozenWorkLease, _observe, _evaluate

VERSION = 'metered-nominal-cold-library-effect-grid-allocated-v3'
RELATIVE = 'factorlab/discovery/nominal_cold_effect_grid.py'
PENDING = ['factor1', 'factor2', 'factor3', 'algorithm']
ACK_VERSION = 'nominal-semantic-publication-ack-v1'


def _deadline_ns(lease):
    delta = datetime.fromisoformat(lease.wall_deadline)-datetime(1970, 1, 1, tzinfo=UTC)
    return (delta.days*86400+delta.seconds)*10**9+delta.microseconds*1000


def _ack_paths(registry, index, target):
    return (target.with_name(target.name+'.semantic-ack.json'),
            *(p/'tasks'/f'{index:06d}'/'semantic-ack.json' for p in
              (registry.directory, registry.anchor)))


def _installed():
    from . import nominal_effect_grid, nominal_effect_design, nominal_cold_law
    from . import nominal_verification_budget, nominal_cold_conformance
    files = source_files()
    for module in (nominal_effect_grid, nominal_effect_design, nominal_cold_law,
                   nominal_verification_budget, nominal_cold_conformance):
        relative = 'factorlab/discovery/' + module.__name__.rsplit('.', 1)[1] + '.py'
        if (Path(module.__file__).resolve() != project_path(relative).resolve()
                or files.get(relative) != file_hash(module.__file__)):
            raise ValueError('Canonical integrated effect-grid dependencies required')
    if (Path(__file__).resolve() != project_path(RELATIVE).resolve()
            or files.get(RELATIVE) != file_hash(__file__)):
        raise ValueError('Canonical source required; staged consumer cannot execute')
    return files


def _failure(path, error):
    # Preserve the original failed attempt; never replace its evidence.
    if not path.exists():
        _write(path, {'consumed': True, 'retain_all_N': True, 'retry_allowed': False,
            'formal_acceptance': False,
            'reason': type(error).__name__ + ':' + str(error)[:1000]})


class ColdEffectGrid:
    @classmethod
    def begin(cls, registry):
        """Reserve only an empty semantic output namespace, no paid risk read.

        Source/header checks have already been allocated to registration task0.
        Each actual heavy consumer below requires its own original task lease.
        """
        if type(registry) is not VerificationRegistry:
            raise TypeError('Original typed verification registry required')
        registry.verify()
        binding = registry.plan.binding
        files = _installed()
        if files != binding['source_files']:
            raise ValueError('Grid source differs from preregistered verification source')
        directory = _path(binding['target_directory'])
        if directory.exists():
            raise ValueError('Semantic grid already attempted; no replacement')
        directory.mkdir(parents=True, exist_ok=False)
        intent = {'version': VERSION, 'directory': str(directory),
            'verification_directory': str(registry.directory),
            'verification_digest': registry.digest, 'census_directory': binding['census_directory'],
            'census_digest': binding['census_digest'], 'source_files': files,
            'acceptance_digest': binding['acceptance_digest'],
            'designs': binding['library_designs'], 'estimand': 'nominal_full_cold_rolling',
            'support': 128, 'groups': [[0, 64], [64, 128]],
            'pending_slots': PENDING, 'formal_acceptance': False, 'retry_allowed': False}
        _write(directory/'intent.json', intent)
        return cls(directory)

    def __init__(self, directory):
        from .nominal_cold_conformance import NominalCensus
        self.directory = _path(directory)
        self.identity = _read(self.directory/'intent.json')
        self.registry = VerificationRegistry(self.identity['verification_directory'])
        self.census = NominalCensus(self.identity['census_directory'])
        self._identity()

    def _identity(self):
        self.registry.verify()
        binding = self.registry.plan.binding
        expected = {'version': VERSION, 'directory': str(self.directory),
            'verification_directory': str(self.registry.directory),
            'verification_digest': self.registry.digest, 'census_directory': binding['census_directory'],
            'census_digest': binding['census_digest'], 'source_files': _installed(),
            'acceptance_digest': binding['acceptance_digest'],
            'designs': [scenario_design(self.census.digest, n) for n, _, _ in DESIGNS],
            'estimand': 'nominal_full_cold_rolling', 'support': 128,
            'groups': [[0, 64], [64, 128]], 'pending_slots': PENDING,
            'formal_acceptance': False, 'retry_allowed': False}
        if (self.identity != expected or _read(self.directory/'intent.json') != expected
                or str(self.directory) != binding['target_directory']
                or self.census.digest != binding['census_digest']
                or self.census.verify()['source_files'] != binding['source_files']):
            raise ValueError('Original whole-grid identity changed')
        if (self.directory/'failure.json').exists():
            raise ValueError('Failed grid is consumed; all original cells remain')
        return expected

    def _task(self, task, index):
        if type(task) is not VerificationTask:
            raise TypeError('Original active paid verification task required')
        task.check()
        if (task.service.registry.digest != self.registry.digest
                or task.service.registry.directory != self.registry.directory
                or task.task != self.registry.plan.task(index)
                or task.intent['registry_digest'] != self.registry.digest):
            raise ValueError('Task cannot borrow another grid/census/coordinate')
        self._identity()
        task.check()

    def _publish(self, task, payload, target):
        """Write actual semantic payload BEFORE the original resource terminal.

        Three copies bind the exact task intent/lease. Resource-only finish is
        not treated as proof unless these canonical semantic records also pass.
        """
        task.check()
        observed = _observe(task.lease)
        if not _evaluate(task.lease, observed).allowed:
            raise ValueError('Semantic publication outside original task lease')
        record = {'version': VERSION, 'grid_intent_hash': fingerprint(self.identity),
            'registry_digest': self.registry.digest, 'task': task.task,
            'task_intent_hash': fingerprint(task.intent), 'lease': task.lease.to_dict(),
            'observed': observed, 'payload': payload, 'formal_acceptance': False}
        for path in (target, *(p/'semantic-output.json' for p in task.roots)):
            _write(path, record)
        task.check()
        terminal = task.finish()
        # Readback performed by finish also belongs to the SAME original lease.
        # A late return poisons the grid rather than turning an early terminal
        # timestamp into semantic permission.
        task.lease.assert_parent_alive()
        # This final semantic ACK is the commit point, after paid computation,
        # resource completion and the last original lease check. No subsequent
        # diagnostic return is defined as another semantic success operation.
        successor = _read(task.roots[0]/'successor.json')
        if successor != _read(task.roots[1]/'successor.json'):
            raise ValueError('Original resource successor differs before semantic commit')
        observed = _observe(task.lease)
        if not _evaluate(task.lease, observed, successor['observed']).allowed:
            raise ValueError('Semantic ACK observation outside original task lease')
        ack = {'version': ACK_VERSION, 'grid_intent_hash': fingerprint(self.identity),
            'registry_digest': self.registry.digest, 'task': task.task,
            'task_intent_hash': fingerprint(task.intent), 'lease': task.lease.to_dict(),
            'target': str(target), 'semantic_record_hash': fingerprint(record),
            'semantic_file_sha256': file_hash(target),
            'resource_terminal_hash': fingerprint(terminal),
            'resource_successor_hash': fingerprint(successor),
            'observed': observed, 'formal_acceptance': False}
        for path in _ack_paths(self.registry, task.task['index'], target):
            _write(path, ack)
        # A reader MUST authenticate every ACK copy and actual publication
        # deadline before using the payload. A partial/late ACK is unavailable
        # even if failure storage is broken and resource witnesses completed.
        return payload

    def _native_semantic(self, index, target, payload):
        """Stored resource/ACK records require original live or retained OS proof."""
        from .nominal_native_completion import NativeCompletionReader
        reader = NativeCompletionReader(self.registry)
        expected = {'verification_digest': self.registry.digest, 'task_index': index,
                    'target': str(target), 'semantic_hash': fingerprint(payload),
                    'native_completed': True}
        actual = reader.verify_task(index, target, expected['semantic_hash'])
        if (type(actual) is not dict or actual != expected
                or any(type(actual.get(key)) is not type(value) for key, value in expected.items())):
            raise ValueError('Original source-bound native semantic completion differs')

    def _semantic(self, index, target):
        """Authenticate original semantic bytes and actual native completion."""
        witness = self.registry.task_witness(index)
        roots = [p/'tasks'/f'{index:06d}' for p in
                 (self.registry.directory, self.registry.anchor)]
        intent = _read(roots[0]/'intent.json')
        record = _read(target)
        if any(_read(p/'semantic-output.json') != record for p in roots):
            raise ValueError('Semantic output mirrors changed')
        lease = FrozenWorkLease.from_dict(intent['lease'])
        if (record['version'] != VERSION or record['grid_intent_hash'] != fingerprint(self.identity)
                or record['registry_digest'] != self.registry.digest
                or record['task'] != self.registry.plan.task(index)
                or record['task_intent_hash'] != fingerprint(intent)
                or record['lease'] != lease.to_dict() or record['formal_acceptance'] is not False
                or not _evaluate(lease, record['observed']).allowed
                or not _evaluate(lease, witness['observed'], record['observed']).allowed):
            raise ValueError('Semantic provenance/deadline does not match original task')
        successor = _read(roots[0]/'successor.json')
        paths = _ack_paths(self.registry, index, target)
        ack = _read(paths[0])
        if any(_read(p) != ack for p in paths[1:]):
            raise ValueError('Final semantic ACK copies differ or are incomplete')
        expected = {'version': ACK_VERSION, 'grid_intent_hash': fingerprint(self.identity),
            'registry_digest': self.registry.digest, 'task': self.registry.plan.task(index),
            'task_intent_hash': fingerprint(intent), 'lease': lease.to_dict(),
            'target': str(target), 'semantic_record_hash': fingerprint(record),
            'semantic_file_sha256': file_hash(target),
            'resource_terminal_hash': fingerprint(witness),
            'resource_successor_hash': fingerprint(successor),
            'observed': ack['observed'], 'formal_acceptance': False}
        if (ack != expected or not _evaluate(lease, ack['observed'], successor['observed']).allowed
                or any(p.stat().st_mtime_ns > _deadline_ns(lease) for p in paths)):
            raise ValueError('Original semantic ACK identity or physical publication deadline differs')
        # File mtime and JSON timestamps are consistency checks, not sufficient
        # authority. No output is consumed without its exact original typed
        # live batch or actual retained original systemd invocation.
        self._native_semantic(index, target, record['payload'])
        return record['payload']

    def _atom_snapshot(self, atom):
        target = self.directory/f'atom-{atom:03d}.json'
        return {'atom': atom, 'sha256': file_hash(target),
            'semantic_ack_sha256': [file_hash(p) for p in
                _ack_paths(self.registry, atom+1, target)]}

    def collect_atom(self, service, atom):
        """Read all original paid cold evidence for exactly this allocated atom."""
        if type(service) is not VerificationService or type(atom) is not int or not 0 <= atom < 128:
            raise TypeError('Typed original service and fixed integer atom required')
        if service.registry.digest != self.registry.digest:
            raise ValueError('Another service cannot consume this grid')
        task = service.acquire(atom+1)
        return self._collect_allocated(task, atom)

    def _collect_allocated(self, task, atom):
        try:
            self._task(task, atom+1)
            record = _original_atom(self.census, atom)
            task.check()
            return self._publish(task, record, self.directory/f'atom-{atom:03d}.json')
        except BaseException as error:
            if not task.closed:
                task.fail(type(error).__name__+':'+str(error))
            _failure(self.directory/'failure.json', error)
            raise

    def _records(self):
        risks, records = [], []
        with self.registry.witness_read_scope():
            for atom in range(128):
                path = self.directory/f'atom-{atom:03d}.json'
                record = self._semantic(atom+1, path)
                if record['atom'] != atom:
                    raise ValueError('Atom order/coordinate changed')
                risks.append((record['path_risks']['reference'], record['path_risks']['candidate']))
                records.append(self._atom_snapshot(atom))
        return tuple(risks), records

    def reduce(self, service):
        if type(service) is not VerificationService or service.registry.digest != self.registry.digest:
            raise TypeError('Original same-grid service required')
        task = service.acquire(129)
        return self._reduce_allocated(task)

    def _reduce_allocated(self, task):
        try:
            self._task(task, 129)
            risks, records = self._records()
            laws = grid_payload(derive_effect_grid(risks))
            for (name, _, _), law in zip(DESIGNS, laws):
                law['scenario_design_id'] = scenario_design_id(self.census.digest, name)
                law['cell_name'] = 'library_'+name
            value = {'intent_hash': fingerprint(self.identity), 'records': records, 'laws': laws,
                'source_files': self.identity['source_files'],
                'acceptance_digest': self.identity['acceptance_digest'],
                'status': 'FIXED_GRID_NUMERIC_NOT_FORMAL_CALIBRATION',
                'unavailable_cells': 'retain_full_N_conservative_failure',
                'actual_conformance': 'required_separate_original_trace_check',
                'pending_slots': PENDING, 'formal_acceptance': False}
            return self._publish(task, value, self.directory/'grid.json')
        except BaseException as error:
            if not task.closed:
                task.fail(type(error).__name__+':'+str(error))
            _failure(self.directory/'failure.json', error)
            raise

    def header(self):
        self._identity()
        value = self._semantic(129, self.directory/'grid.json')
        if (value['intent_hash'] != fingerprint(self.identity)
                or value['source_files'] != self.identity['source_files']
                or value['acceptance_digest'] != self.identity['acceptance_digest']
                or value['status'] != 'FIXED_GRID_NUMERIC_NOT_FORMAL_CALIBRATION'
                or value['unavailable_cells'] != 'retain_full_N_conservative_failure'
                or value['actual_conformance'] != 'required_separate_original_trace_check'
                or value['pending_slots'] != PENDING or value['formal_acceptance'] is not False
                or len(value['laws']) != 5
                or [x['atom'] for x in value['records']] != list(range(128))):
            raise ValueError('Whole-grid scope/order/semantics changed')
        for (name, target, _), row in zip(DESIGNS, value['laws']):
            if (row['target'] != list(target) or row['cell_name'] != 'library_'+name
                    or row['scenario_design_id'] != scenario_design_id(self.census.digest, name)
                    or row['status'] not in ('available_numeric_only', 'unavailable_fixed_groups_do_not_straddle')
                    or row['formal_acceptance'] is not False):
                raise ValueError('Prospective scenario mapped to another derived row')
        return value

    @property
    def digest(self):
        return fingerprint(self.header())

    def scenario(self, design_id):
        matches = [r for r in self.header()['laws'] if r['scenario_design_id'] == design_id]
        if len(matches) != 1:
            raise ValueError('Unsupported scenario stays unavailable in original N')
        return matches[0]

    def verify(self, *, task):
        """Original wrapper retains the original already-allocated task."""
        return self._verify_allocated(task)

    def _verify_allocated(self, task):
        """Full original paid re-read in an allocated, nonrenewable 300s task."""
        indices = (130, 131+self.registry.plan.total_N)
        if type(task) is not VerificationTask or task.task['index'] not in indices:
            raise TypeError('Original pre-registration or final full-verification task required')
        index = task.task['index']
        try:
            self._task(task, index)
            value = self.header()
            risks, records = [], []
            with self.registry.witness_read_scope():
                for atom in range(128):
                    task.check()
                    path = self.directory/f'atom-{atom:03d}.json'
                    stored = self._semantic(atom+1, path)
                    actual = _original_atom(self.census, atom)
                    if stored != actual:
                        raise ValueError('Grid atom differs from complete original paid cold evidence')
                    risks.append((actual['path_risks']['reference'], actual['path_risks']['candidate']))
                    records.append(self._atom_snapshot(atom))
                    del actual, stored
            laws = grid_payload(derive_effect_grid(tuple(risks)))
            for (name, _, _), row in zip(DESIGNS, laws):
                row['scenario_design_id'] = scenario_design_id(self.census.digest, name)
                row['cell_name'] = 'library_'+name
            if records != value['records'] or laws != value['laws'] or self.header() != value:
                raise ValueError('Complete cold evidence or fixed effect grid changed')
            self._identity()
            payload = {'grid_digest': fingerprint(value), 'grid_file_sha256': file_hash(self.directory/'grid.json'),
                'grid_intent_hash': fingerprint(self.identity), 'records': records,
                'census_digest': self.census.digest, 'registry_digest': self.registry.digest,
                'source_files': self.identity['source_files'],
                'acceptance_digest': self.identity['acceptance_digest'],
                'task_index': index, 'full_original_paid_census_reconstructed': True,
                'pending_slots': PENDING, 'formal_acceptance': False}
            return self._publish(task, payload, self.directory/f'verified-{index:06d}.json')
        except BaseException as error:
            if type(task) is VerificationTask and not task.closed:
                task.fail(type(error).__name__+':'+str(error))
            _failure(self.directory/'failure.json', error)
            raise

    def execute_allocated(self, task):
        """Execute one exact original parent-owned task, without allocation.

        Outputs remain provisional: this entry does not solve the historical
        ACK/native-publication proof. _publish closes the original resource;
        a parent RPC must not call resource_finish a second time.
        """
        if type(task) is not VerificationTask:
            raise TypeError('Exact already-acquired original verification task required')
        try:
            index = task.task['index']
            if type(index) is not int:
                raise TypeError('Exact original integer task coordinate required')
            self._task(task, index)
            kind = task.task['kind']
            if 1 <= index <= 128 and kind == 'cold_atom_evidence_validation':
                if task.task['unit'] != {'atom': index-1}:
                    raise ValueError('Original atom coordinate differs')
                payload = self._collect_allocated(task, index-1)
            elif index == 129 and kind == 'law_reduction':
                payload = self._reduce_allocated(task)
            elif ((index == 130 and kind == 'full_law_verify_before_actual_registry')
                  or (index == 131+self.registry.plan.total_N
                      and kind == 'full_law_verify_before_final_report')):
                payload = self._verify_allocated(task)
            else:
                raise NotImplementedError('Canonical allocated task handler not implemented')
            if task.closed is not True:
                raise ValueError('Original resource completion required')
            return {'task_index': index, 'kind': kind,
                'task_intent_hash': fingerprint(task.intent),
                'payload_hash': fingerprint(payload), 'resource_closed': True,
                'semantic_completion': 'provisional', 'formal_acceptance': False}
        except BaseException as error:
            if not task.closed:
                task.fail(type(error).__name__+':'+str(error))
            _failure(self.directory/'failure.json', error)
            raise

    def verified_manifest(self, task_index, *, service=None):
        """Authenticate completed paid reconstruction, without repeating it free.

        Cold services must already have closed before actual registration. The
        final report consumer may use its exact still-live final service; it
        cannot require that service to close before its own remaining tasks.
        This is a snapshot, not OS immutability or a model/entropy grant.
        """
        if type(task_index) is not int or task_index not in (130, 131+self.registry.plan.total_N):
            raise ValueError('Fixed original full-verification coordinate required')
        live_service = self.registry.plan.task(task_index)['service']
        if service is not None:
            if (task_index == 130 or type(service) is not VerificationService
                    or service.registry.digest != self.registry.digest
                    or service.registry.directory != self.registry.directory
                    or service.plan != self.registry.plan.service(live_service)):
                raise TypeError('Only the original still-live final report service can borrow its completed proof')
            service.check()
        value = self.header()
        manifest = self._semantic(task_index, self.directory/f'verified-{task_index:06d}.json')
        if (manifest['grid_digest'] != fingerprint(value)
                or manifest['grid_file_sha256'] != file_hash(self.directory/'grid.json')
                or manifest['grid_intent_hash'] != fingerprint(self.identity)
                or manifest['census_digest'] != self.census.digest
                or manifest['registry_digest'] != self.registry.digest
                or manifest['source_files'] != self.identity['source_files']
                or manifest['acceptance_digest'] != self.identity['acceptance_digest']
                or manifest['task_index'] != task_index
                or manifest['full_original_paid_census_reconstructed'] is not True
                or manifest['records'] != value['records'] or manifest['pending_slots'] != PENDING
                or manifest['formal_acceptance'] is not False):
            raise ValueError('Paid reconstruction manifest changed')
        for record in manifest['records']:
            if self._atom_snapshot(record['atom']) != record:
                raise ValueError('Verified cold snapshot changed')
        # Six original cold-phase services, plus the final service if requested.
        services = {self.registry.plan.task(i)['service'] for i in range(1, 131)}
        services.add(self.registry.plan.task(task_index)['service'])
        with self.registry.witness_read_scope():
            for index in sorted(services):
                if service is not None and index == live_service:
                    service.check()
                else:
                    self.registry.service_witness(index)
        return {'grid_digest': fingerprint(value), 'registry_digest': self.registry.digest,
            'census_digest': self.census.digest, 'source_files': self.identity['source_files'],
            'acceptance_digest': self.identity['acceptance_digest'], 'record': manifest,
            'final_service_still_active': service is not None}

