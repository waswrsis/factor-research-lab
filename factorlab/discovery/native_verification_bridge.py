"""One original verification task's parent-owned IPC bridge.

Staged construction, not a launcher or semantic completion certificate.
All worker output remains provisional until a separate live kernel terminal
reader authenticates natural exit. No registry/service is reopened in a child.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import threading

VERSION = 'native-verification-ipc-v1'
TICKET_VERSION = 'native-verification-task-ticket-v1'
RELATIVE = 'factorlab/discovery/native_verification_bridge.py'
WORKER = 'factorlab.discovery.native_verification_worker'
MAX_PACKET = 65536
OPS = frozenset({'check', 'consume', 'resource_finish', 'fail', 'publication_ready'})


def encode_packet(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'),
                     allow_nan=False).encode('utf8')
    if not 0 < len(raw) <= MAX_PACKET:
        raise ValueError('Bounded IPC packet required')
    return raw


def decode_packet(raw):
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_PACKET:
        raise ValueError('Bounded complete IPC packet required')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate IPC field')
            result[key] = value
        return result
    def finite_float(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError('Nonfinite IPC number')
        return value
    value = json.loads(raw.decode('utf8'), object_pairs_hook=unique, parse_float=finite_float,
                      parse_constant=lambda _: (_ for _ in ()).throw(
                          ValueError('Nonfinite IPC value')))
    if type(value) is not dict:
        raise ValueError('IPC object required')
    return value


def validate_request(value, digest, sequence):
    if (set(value) != {'version', 'ticket_digest', 'sequence', 'op', 'args'}
            or value['version'] != VERSION or value['ticket_digest'] != digest
            or type(value['sequence']) is not int or value['sequence'] != sequence
            or type(value['op']) is not str or value['op'] not in OPS
            or type(value['args']) is not dict):
        raise ValueError('Foreign, replayed or unsupported original task request')
    args = value['args']
    if value['op'] == 'fail':
        if (set(args) != {'reason'} or type(args['reason']) is not str
                or not 0 < len(args['reason']) <= 1000):
            raise ValueError('Bounded failure reason required')
    elif args:
        raise ValueError('Caller-selected task, output, lease or command forbidden')
    return value['op'], args


class _RequestState:
    """Pure protocol state; does not authenticate a task or invoke authority."""
    def __init__(self, digest):
        self.digest = digest
        self.sequence = 0
        self.resource_finished = False
        self.consume_started = False
        self.publication_ready = False
        self.poisoned = False

    def admit(self, value):
        if self.poisoned or self.publication_ready:
            raise ValueError('Terminal bridge cannot be reused')
        op, args = validate_request(value, self.digest, self.sequence)
        if self.resource_finished and op == 'resource_finish':
            raise ValueError('Original resource completion cannot be repeated')
        if op == 'consume' and (self.consume_started or self.resource_finished):
            raise ValueError('Original semantic task can execute only once')
        if op == 'publication_ready' and not self.resource_finished:
            raise ValueError('Resource completion must precede final publication')
        # Consumption precedes the authority operation and response write.
        self.sequence += 1
        if op == 'consume':
            self.consume_started = True
        return op, args


def _pid_start(pid):
    text = Path(f'/proc/{pid}/stat').read_text('ascii')
    end = text.rfind(')')
    fields = text[end+2:].split()
    if end < 0 or len(fields) < 20:
        raise ValueError('Original Linux process identity unavailable')
    return int(fields[1]), int(fields[19])  # ppid and kernel start ticks


def _claim_attempt(task):
    """Exclusive, irreversible local attempt anchors before any worker ticket.

    immutable_record intentionally accepts identical bytes on readback, so a
    ticket alone is not an exclusive attempt marker. Half-created directories
    also consume the attempt. Failure recording is best effort, never a refund.
    """
    markers = [root/'native-attempt' for root in task.roots]
    if any(marker.exists() or (root/'native-ticket.json').exists()
           for root, marker in zip(task.roots, markers)):
        raise ValueError('Original native task attempt already consumed')
    try:
        for marker in markers:
            marker.mkdir(exist_ok=False)
    except BaseException as error:
        try:
            task.fail(type(error).__name__ + ':' + str(error)[:900])
        except BaseException as recording_error:
            error.add_note('Failure persistence unavailable: ' + type(recording_error).__name__)
        raise


class NativeVerificationAttempt:
    """Same-thread original parent, one consumed task, one fixed direct child.

    prepare() publishes the original ticket BEFORE launch. bind_child() neither
    starts a process nor grants time. serve_one() executes original task methods
    in their issuer, authenticating SCM_CREDENTIALS on every packet. A successful
    IPC reply is provisional; this class never claims semantic completion.
    """
    @classmethod
    def prepare(cls, task, *, worker_cgroup):
        from factorlab.protocol import project_path, fingerprint
        from .evidence import file_hash
        from .monthly_work_authority import source_files
        from .nominal_verification_budget import VerificationTask, _write
        if sys.platform != 'linux':
            raise OSError('Native task bridge requires Linux')
        if type(task) is not VerificationTask:
            raise TypeError('Already acquired exact original task required')
        task.check()
        _claim_attempt(task)
        files = source_files()
        if (Path(__file__).resolve() != project_path(RELATIVE).resolve()
                or files.get(RELATIVE) != file_hash(__file__)):
            raise ValueError('Integrated canonical source required')
        worker_file = 'factorlab/discovery/native_verification_worker.py'
        if files.get(worker_file) != file_hash(project_path(worker_file)):
            raise ValueError('Fixed worker absent from original source inventory')
        if files != task.service.registry.plan.binding['source_files']:
            raise ValueError('Original registered source differs')
        cgroup = Path(worker_cgroup)
        if (not cgroup.is_absolute() or cgroup.is_symlink()
                or not str(cgroup).startswith('/sys/fs/cgroup/')
                or cgroup.resolve() != cgroup):
            raise ValueError('Original exclusive worker cgroup required')
        # Root directories derive solely from the original allocated task.
        ticket_path = task.roots[0]/'native-ticket.json'
        paths = [str(p/'semantic-output.json') for p in task.roots]
        ack_paths = [str(p/'semantic-ack.json') for p in task.roots]
        identity = {
            'version': TICKET_VERSION, 'protocol': VERSION, 'worker_module': WORKER,
            'registry_digest': task.service.registry.digest,
            'task': task.task, 'task_intent_hash': fingerprint(task.intent),
            'service_binding_hash': fingerprint(task.service.binding),
            'lease': task.lease.to_dict(),
            'service_clock': task.service.clock.to_dict(), 'source_files': files,
            'task_roots': [str(p) for p in task.roots],
            'semantic_paths': paths, 'semantic_ack_paths': ack_paths,
            'worker_cgroup': str(cgroup), 'retry_allowed': False,
            'grants_fit_or_GPT_credit': False, 'semantic_completion': 'provisional',
        }
        # Each mirror is exclusive. Partial publication consumes the attempt.
        for root in task.roots:
            _write(root/'native-ticket.json', identity)
        task.check()
        self = cls.__new__(cls)
        self.task, self.identity, self.ticket_path = task, identity, ticket_path
        self.digest = fingerprint(identity)
        self.worker_cgroup = cgroup
        self.owner = (os.getpid(), threading.get_ident())
        self.state = _RequestState(self.digest)
        self.process = self.channel = self.child_fd = self.child_start = None
        self.publication_snapshot = None
        self._gate_sent = False
        self._terminal_handle = None
        self._completion_observed = False
        return self

    def _original(self):
        from .nominal_verification_budget import _read
        if self.owner != (os.getpid(), threading.get_ident()):
            raise ValueError('Exact original issuer thread required')
        if self.state.poisoned:
            raise ValueError('Failed attempt remains consumed')
        for root in self.task.roots:
            if (_read(root/'native-ticket.json') != self.identity
                    or (root/'failure.json').exists()):
                raise ValueError('Original ticket changed or task failed')
        self.task.service.check()
        self.task.lease.assert_parent_alive()

    def expected_argv(self, child_fd):
        if type(child_fd) is not int or child_fd < 0:
            raise ValueError('Original child descriptor required')
        return [sys.executable, '-B', '-m', WORKER, '--ticket',
                str(self.ticket_path), '--fd', str(child_fd)]

    def bind_child(self, process, channel, *, child_fd):
        self._original()
        if self.process is not None:
            raise ValueError('One original child; no replacement')
        if (type(process) is not subprocess.Popen
                or process.args != self.expected_argv(child_fd)
                or type(channel) is not socket.socket
                or channel.family != socket.AF_UNIX
                or channel.type & 0xf != socket.SOCK_SEQPACKET):
            raise ValueError('Exact fixed worker and seqpacket channel required')
        ppid, start = _pid_start(process.pid)
        if ppid != self.owner[0] or process.returncode is not None:
            raise ValueError('Live original direct child required')
        # socketpair SO_PEERCRED identifies its creator, not the sender after
        # fork. Per-message SCM_CREDENTIALS is therefore mandatory.
        channel.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
        self.process, self.channel = process, channel
        self.child_fd, self.child_start = child_fd, start

    def bind_terminal_handle(self, handle):
        from .native_verification_terminal import LiveTerminalHandle
        self._original()
        if (self._terminal_handle is not None or type(handle) is not LiveTerminalHandle
                or handle.task is not self.task or handle.process is not self.process
                or handle.worker_cgroup != self.worker_cgroup):
            raise ValueError('One exact original child terminal handle required')
        self._terminal_handle = handle

    def release_start_gate(self):
        """Release the fixed child only after its original containment is bound."""
        self._original()
        if self._gate_sent or self.process is None or self._terminal_handle is None:
            raise ValueError('One contained original child start gate required')
        # Consume before send. Uncertain/partial gate never permits a retry.
        self._gate_sent = True
        try:
            packet = encode_packet({'version': VERSION, 'ticket_digest': self.digest,
                'sequence': -1, 'ok': True, 'value': {'start_original_task': True}})
            task_check = self.task.lease.assert_parent_alive()
            service_check = self.task.service.check()
            self.channel.settimeout(min(task_check.remaining_seconds, service_check.remaining_seconds))
            if self.channel.send(packet) != len(packet):
                raise OSError('Incomplete original child gate')
            self._original()
        except BaseException as error:
            self.state.poisoned = True
            self._terminal_handle.invalidate(type(error).__name__)
            try:
                self.task.fail(type(error).__name__ + ':' + str(error)[:900])
            except BaseException as recording_error:
                error.add_note('Failure persistence unavailable: ' + type(recording_error).__name__)
            raise

    def _receive(self):
        self._original()
        if self.process is None:
            raise ValueError('Original worker not attached')
        if not self._gate_sent:
            raise ValueError('Original contained child gate not released')
        remaining = min(self.task.lease.check().remaining_seconds,
                        self.task.service.check().remaining_seconds)
        if remaining <= 0:
            raise ValueError('Original task or service deadline reached')
        self.channel.settimeout(remaining)
        raw, ancillary, flags, _ = self.channel.recvmsg(
            MAX_PACKET, socket.CMSG_SPACE(struct.calcsize('3i')))
        if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
            raise ValueError('Truncated native IPC packet or credentials')
        credentials = [struct.unpack('3i', data) for level, kind, data in ancillary
                       if level == socket.SOL_SOCKET and kind == socket.SCM_CREDENTIALS
                       and len(data) == struct.calcsize('3i')]
        if (len(credentials) != 1 or credentials[0] !=
                (self.process.pid, os.getuid(), os.getgid())
                or _pid_start(self.process.pid) != (self.owner[0], self.child_start)):
            raise ValueError('Foreign or reused original worker identity')
        self._original()  # Waiting and parsing never extend either deadline.
        return decode_packet(raw)

    def _publications(self):
        from .nominal_verification_budget import _read
        from factorlab.protocol import fingerprint
        paths = [Path(p) for p in self.identity['semantic_paths']]
        acks = [Path(p) for p in self.identity['semantic_ack_paths']]
        semantic = [_read(p) for p in paths]
        ack = [_read(p) for p in acks]
        if semantic[0] != semantic[1] or ack[0] != ack[1]:
            raise ValueError('Original semantic mirrors differ')
        if (semantic[0]['task_intent_hash'] != self.identity['task_intent_hash']
                or ack[0]['task_intent_hash'] != self.identity['task_intent_hash']
                or ack[0]['semantic_record_hash'] != fingerprint(semantic[0])):
            raise ValueError('Foreign semantic publication')
        # Target comes from the frozen task-specific handler's ACK, and must be
        # in the registered grid; it is not accepted as an IPC argument.
        from .nominal_verification_budget import _path
        target = _path(ack[0]['target'])
        grid = _path(self.task.service.registry.plan.binding['target_directory'])
        kind = self.task.task['kind']
        if kind == 'cold_atom_evidence_validation':
            expected = grid/f"atom-{self.task.task['unit']['atom']:03d}.json"
        elif kind == 'law_reduction':
            expected = grid/'grid.json'
        elif kind in {'full_law_verify_before_actual_registry',
                      'full_law_verify_before_final_report'}:
            expected = grid/f"verified-{self.task.task['index']:06d}.json"
        elif kind == 'actual_coordinate_readback':
            expected = grid/'full-N'/f"coordinate-{self.task.task['index']:06d}.json"
        elif kind == 'acceptance_cell_aggregation':
            cells = self.task.service.registry.plan.acceptance.cells
            ordinal = next(i for i, cell in enumerate(cells)
                           if cell.name == self.task.task['unit']['cell'])
            expected = grid/'full-N'/f'cell-{ordinal:03d}.json'
        elif kind == 'final_full_N_report':
            expected = grid/'full-N'/'report.json'
        else:
            # Other consumers require their own canonical source-bound output
            # map. They must not borrow this grid handler's success path.
            raise ValueError('Canonical task publication mapping not implemented')
        if target != expected or _read(target) != semantic[0]:
            raise ValueError('Original canonical target differs')
        target_ack = target.with_name(target.name+'.semantic-ack.json')
        if _read(target_ack) != ack[0]:
            raise ValueError('All three original ACK publications required')
        paths.extend(acks + [target, target_ack])
        paths.extend(root/name for root in self.task.roots
                     for name in ('intent.json', 'terminal.json', 'successor.json',
                                  'native-ticket.json'))
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

    def _consume_original(self):
        """Fixed full-grid or full-N consumer; never an IPC selected callable."""
        from .nominal_cold_effect_grid import ColdEffectGrid
        from .monthly_work_authority import source_files
        from .evidence import file_hash
        from factorlab.protocol import project_path
        grid = ColdEffectGrid(self.task.service.registry.plan.binding['target_directory'])
        consumer = grid
        if self.task.task['kind'] in {'actual_coordinate_readback',
                                    'acceptance_cell_aggregation', 'final_full_N_report'}:
            from .nominal_full_N_report import NominalFullNReport
            from .nominal_actual_trial import NominalActualRegistry
            from .nominal_verification_budget import _read
            # This original immutable enrollment selects the actual registry.
            # No caller path, alternate plan or observations are admitted.
            identity = _read(grid.directory/'actual_registry.json')
            actual = NominalActualRegistry(identity['directory'])
            consumer = NominalFullNReport(actual, grid)
        cls = type(consumer)
        module = sys.modules[cls.__module__]
        relative = 'factorlab/discovery/' + cls.__module__.rsplit('.', 1)[1] + '.py'
        files = source_files()
        if (Path(module.__file__).resolve() != project_path(relative).resolve()
                or files.get(relative) != file_hash(module.__file__)
                or files != self.task.service.registry.plan.binding['source_files']):
            raise ValueError('Canonical original allocated semantic consumer required')
        result = consumer.execute_allocated(self.task)
        if consumer is grid:
            summary = result
        else:
            from .nominal_verification_budget import _read
            from factorlab.protocol import fingerprint
            # Full-N's public API preserves its original payload return, unlike
            # the grid's small execution summary. Normalize only after checking
            # the actual original semantic output, never guess either shape.
            if _read(self.task.roots[0]/'semantic-output.json')['payload'] != result:
                raise ValueError('Original report payload differs from publication')
            summary = {'task_index': self.task.task['index'], 'kind': self.task.task['kind'],
                'task_intent_hash': fingerprint(self.task.intent),
                'payload_hash': fingerprint(result), 'resource_closed': self.task.closed,
                'semantic_completion': 'provisional', 'formal_acceptance': False}
        if (not self.task.closed or summary.get('resource_closed') is not True
                or summary.get('task_index') != self.task.task['index']
                or summary.get('kind') != self.task.task['kind']
                or summary.get('formal_acceptance') is not False):
            raise ValueError('Original semantic consumer did not complete resource work')
        return summary

    def serve_one(self):
        """Execute exactly one authenticated request in the original issuer."""
        try:
            request = self._receive()
            op, args = self.state.admit(request)
            if op == 'check':
                self._original()
                value = {'remaining_seconds': self.task.lease.check().remaining_seconds}
            elif op == 'consume':
                # The worker chooses no handler, frame, target or work key.
                # Execution stays in the original issuer which owns the task.
                self._consume_original()
                self._original()
                self.state.resource_finished = True
                value = {'resource_finished': True, 'semantic_provisional': True}
            elif op == 'resource_finish':
                value = self.task.finish()
                self.state.resource_finished = True
            elif op == 'publication_ready':
                self._original()
                self.publication_snapshot = self._publications()
                self._original()
                value = {'provisional': True, 'publication_sha256': self.publication_snapshot}
                self.state.publication_ready = True
            else:
                self.state.poisoned = True
                self.task.fail(args['reason'])
                value = {'failed_consumed': True, 'retain_all_N': True}
            reply = {'version': VERSION, 'ticket_digest': self.digest,
                     'sequence': request['sequence'], 'ok': True, 'value': value}
            packet = encode_packet(reply)
            check = self.task.lease.assert_parent_alive()
            service_check = self.task.service.check()
            self.channel.settimeout(min(check.remaining_seconds, service_check.remaining_seconds))
            if self.channel.send(packet) != len(packet):
                raise OSError('Incomplete provisional IPC response')
            if not self.state.poisoned:
                self._original()
            return op
        except BaseException as error:
            self.state.poisoned = True
            if self._terminal_handle is not None:
                self._terminal_handle.invalidate(type(error).__name__)
            # A persisted success-looking resource receipt does not override
            # this in-memory poison or the absent live terminal authority.
            try:
                self.task.fail(type(error).__name__ + ':' + str(error)[:900])
            except BaseException as recording_error:
                error.add_note('Failure persistence unavailable: ' + type(recording_error).__name__)
            raise

    def recheck_publications(self):
        """Still provisional; invoke while the original task is live."""
        self._original()
        if not self.state.publication_ready or self.publication_snapshot is None:
            raise ValueError('No completed original publication handshake')
        if self._publications() != self.publication_snapshot:
            raise ValueError('Publication changed after original ready handshake')
        self._original()
        return dict(self.publication_snapshot)

    def observe_completion(self):
        """Bind provisional publication to a still-live kernel exit authority.

        Not durable or whole-grid acceptance. The handle cannot be serialized;
        retaining thousands of zombies is not the project's final architecture.
        No additional evidence write is claimed as part of completion here.
        """
        if self._terminal_handle is None or self._completion_observed:
            raise ValueError('One original native completion observation required')
        try:
            self.recheck_publications()
            observation = self._terminal_handle.observe()
            if (observation.get('origin') != 'live_kernel_reader'
                    or observation.get('status') != 'native_observed_natural_exit'):
                raise ValueError('Original timely natural exit not observed')
            self._completion_observed = True
            return {'status': 'single_task_publication_bound_to_live_exit',
                    'native_diagnostic': observation, 'durable_authority': False,
                    'formal_acceptance': False}
        except BaseException as error:
            self.state.poisoned = True
            self._terminal_handle.invalidate(type(error).__name__)
            try:
                self.task.fail(type(error).__name__ + ':' + str(error)[:900])
            except BaseException as recording_error:
                error.add_note('Failure persistence unavailable: ' + type(recording_error).__name__)
            raise

    def read_completion(self):
        """Historical reread requires the original unreaped live native handle.

        This read grants no new work time. A copied result, process disappearance,
        completed resource receipt or ACK mtime cannot reconstitute authority.
        """
        from .nominal_verification_budget import _read
        if (self.owner != (os.getpid(), threading.get_ident()) or self.state.poisoned
                or not self._completion_observed or self._terminal_handle is None):
            raise ValueError('Original live completion authority unavailable')
        for root in self.task.roots:
            if ((root/'failure.json').exists()
                    or _read(root/'native-ticket.json') != self.identity):
                raise ValueError('Failed or replaced original task')
        if self._publications() != self.publication_snapshot:
            raise ValueError('Original accepted publication changed')
        observed = self._terminal_handle.read_live()
        if (observed.get('origin') != 'live_kernel_reader'
                or observed.get('status') != 'native_observed_natural_exit'):
            raise ValueError('Original kernel completion authority lost')
        return {'status': 'single_task_publication_bound_to_live_exit',
                'native_diagnostic': observed, 'durable_authority': False,
                'formal_acceptance': False}
