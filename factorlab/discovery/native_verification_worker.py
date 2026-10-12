"""Fixed Linux verification-worker IPC admission and parent-owned task dispatch.

The client never creates or renews a lease and never opens issuer authority.
Replies are provisional resource messages, not native or scientific acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import struct
import sys
import threading
from types import SimpleNamespace

VERSION = 'native-verification-ipc-v1'
TICKET_VERSION = 'native-verification-task-ticket-v1'
WORKER = 'factorlab.discovery.native_verification_worker'
RELATIVE = 'factorlab/discovery/native_verification_worker.py'
MAX_PACKET = 65536
MAX_RECORD = 2**20
OPS = frozenset({'check', 'consume', 'resource_finish', 'fail', 'publication_ready'})
KINDS = frozenset({'cold_atom_evidence_validation', 'law_reduction',
    'full_law_verify_before_actual_registry', 'actual_coordinate_readback',
    'full_law_verify_before_final_report', 'acceptance_cell_aggregation',
    'final_full_N_report'})
TICKET_FIELDS = frozenset({'version','protocol','worker_module','registry_digest',
    'task','task_intent_hash','service_binding_hash','lease','service_clock',
    'source_files','task_roots','semantic_paths','semantic_ack_paths','worker_cgroup',
    'retry_allowed','grants_fit_or_GPT_credit','semantic_completion'})
_KEY = object()


class ProtocolError(ValueError):
    pass


def _float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError('Nonfinite JSON number')
    return result


def _decode(raw, limit):
    if type(raw) is not bytes or not 0 < len(raw) <= limit:
        raise ProtocolError('Bounded complete JSON bytes required')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ProtocolError('Duplicate JSON field')
            result[key] = value
        return result
    try:
        value = json.loads(raw.decode('utf8'), object_pairs_hook=unique,
            parse_float=_float, parse_constant=lambda _: (_ for _ in ()).throw(
                ProtocolError('Nonfinite JSON value')))
    except (UnicodeError, ValueError, RecursionError) as error:
        raise ProtocolError('Invalid bounded JSON') from error
    if type(value) is not dict:
        raise ProtocolError('JSON object required')
    return value


def encode_packet(value):
    if type(value) is not dict:
        raise ProtocolError('IPC object required')
    try:
        raw = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         allow_nan=False).encode('utf8')
    except (TypeError, ValueError, RecursionError) as error:
        raise ProtocolError('Invalid JSON IPC value') from error
    if not 0 < len(raw) <= MAX_PACKET:
        raise ProtocolError('Bounded IPC packet required')
    return raw


def decode_packet(raw):
    return _decode(raw, MAX_PACKET)


def _digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _hash(value):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ProtocolError('Canonical digest required')
    return value


def _absolute(value):
    if (type(value) is not str or not value.startswith('/') or value == '/'
            or '\\' in value or '\x00' in value
            or any(part in ('', '.', '..') for part in value.split('/')[1:])):
        raise ProtocolError('Canonical absolute Linux path required')
    return value


def parse_ticket(raw):
    """Pure schema check; does not authenticate source, parent or permissions."""
    from .work_lease import FrozenWorkLease, _utc
    wrapper = decode_packet(raw)
    if set(wrapper) != {'record','hash'} or type(wrapper['record']) is not dict:
        raise ProtocolError('Original wrapped ticket required')
    ticket = wrapper['record']
    if _hash(wrapper['hash']) != _digest(ticket) or set(ticket) != TICKET_FIELDS:
        raise ProtocolError('Ticket digest/schema changed')
    if (ticket['version'] != TICKET_VERSION or ticket['protocol'] != VERSION
            or ticket['worker_module'] != WORKER or ticket['retry_allowed'] is not False
            or ticket['grants_fit_or_GPT_credit'] is not False
            or ticket['semantic_completion'] != 'provisional'):
        raise ProtocolError('Unsupported fixed worker protocol')
    for key in ('registry_digest','task_intent_hash','service_binding_hash'):
        _hash(ticket[key])
    files = ticket['source_files']
    if type(files) is not dict or not files or RELATIVE not in files:
        raise ProtocolError('Complete original worker source inventory required')
    for name, digest in files.items():
        if (type(name) is not str or name.startswith('/') or '\\' in name
                or any(p in ('', '.', '..') for p in name.split('/'))):
            raise ProtocolError('Relative original source paths required')
        _hash(digest)
    for key in ('task_roots','semantic_paths','semantic_ack_paths'):
        paths = ticket[key]
        if type(paths) is not list or len(paths) != 2 or paths[0] == paths[1]:
            raise ProtocolError('Two distinct original task paths required')
        for path in paths:
            _absolute(path)
    for root, semantic, ack in zip(ticket['task_roots'],ticket['semantic_paths'],ticket['semantic_ack_paths']):
        if semantic != root+'/semantic-output.json' or ack != root+'/semantic-ack.json':
            raise ProtocolError('Original fixed task output paths required')
    if not _absolute(ticket['worker_cgroup']).startswith('/sys/fs/cgroup/'):
        raise ProtocolError('Original Linux worker cgroup required')
    task = ticket['task']
    if (type(task) is not dict or set(task) != {'index','kind','unit','service','seconds','retry_allowed','grants_fit_or_GPT_credit'}
            or type(task['index']) is not int or task['index'] < 1
            or type(task['service']) is not int or task['service'] < 1
            or type(task['kind']) is not str or task['kind'] not in KINDS
            or type(task['seconds']) is not int or task['seconds'] != 300
            or task['retry_allowed'] is not False or task['grants_fit_or_GPT_credit'] is not False):
        raise ProtocolError('Fixed original verification task required')
    for root in ticket['task_roots']:
        if not root.endswith('/tasks/'+f"{task['index']:06d}"):
            raise ProtocolError('Task path/index mismatch')
    unit = task['unit']; kind = task['kind']
    if kind == 'cold_atom_evidence_validation':
        if (type(unit) is not dict or set(unit) != {'atom'} or type(unit['atom']) is not int
                or not 0 <= unit['atom'] < 128 or task['index'] != unit['atom']+1):
            raise ProtocolError('Original atom coordinate required')
    elif kind == 'actual_coordinate_readback':
        if (type(unit) is not dict or set(unit) != {'cell','index'} or type(unit['index']) is not int
                or not 0 <= unit['index'] < 2000 or type(unit['cell']) is not str or not 0 < len(unit['cell']) <= 80):
            raise ProtocolError('Original actual coordinate required')
    elif kind == 'acceptance_cell_aggregation':
        if type(unit) is not dict or set(unit) != {'cell'} or type(unit['cell']) is not str or not 0 < len(unit['cell']) <= 80:
            raise ProtocolError('Original aggregation cell required')
    elif unit is not None:
        raise ProtocolError('No caller-selected task arguments allowed')
    if kind == 'law_reduction' and task['index'] != 129 or kind == 'full_law_verify_before_actual_registry' and task['index'] != 130:
        raise ProtocolError('Fixed cold proof task index changed')
    lease = FrozenWorkLease.from_dict(ticket['lease'])
    clock = ticket['service_clock']
    if (type(clock) is not dict or set(clock) != {'issuer','service_seconds'}
            or type(clock['service_seconds']) not in (int,float) or clock['service_seconds'] != 7200):
        raise ProtocolError('Original7200-second service clock required')
    issuer = FrozenWorkLease.from_dict(clock['issuer'])
    if (lease.lease_seconds != 300 or issuer.lease_seconds != 300
            or not lease.boot_id.startswith('linux:')
            or any(getattr(lease,k) != getattr(issuer,k) for k in ('parent_pid','parent_created','boot_id'))
            or lease.monotonic_started < issuer.monotonic_started
            or lease.monotonic_deadline > issuer.monotonic_started+7200
            or _utc(lease.wall_started_at) < _utc(issuer.wall_started_at)
            or _utc(lease.wall_deadline).timestamp() > _utc(issuer.wall_started_at).timestamp()+7200):
        raise ProtocolError('Task lease outside original parent service')
    return ticket, wrapper['hash']


def validate_reply(raw, digest, sequence):
    value = decode_packet(raw)
    if (value.get('version') != VERSION or value.get('ticket_digest') != digest
            or type(value.get('sequence')) is not int or value['sequence'] != sequence
            or type(value.get('ok')) is not bool):
        raise ProtocolError('Foreign or replayed IPC response')
    if value['ok']:
        if set(value) != {'version','ticket_digest','sequence','ok','value'}:
            raise ProtocolError('Unknown success response fields')
        return value['value']
    if (set(value) != {'version','ticket_digest','sequence','ok','reason'}
            or type(value['reason']) is not str or not 0 < len(value['reason']) <= 1000):
        raise ProtocolError('Bounded original failure response required')
    raise ProtocolError('Parent refused original task: '+value['reason'])


def _path(value):
    from factorlab.protocol import project_path
    from .calibration_trial_context import _no_redirect
    path = Path(value)
    _no_redirect(path, project_path('.'))
    return project_path(path)


def _record(path):
    path = _path(path)
    with path.open('rb') as handle:
        raw = handle.read(MAX_RECORD+1)
    value = _decode(raw,MAX_RECORD)
    if set(value) != {'record','hash'} or _hash(value['hash']) != _digest(value['record']):
        raise ProtocolError('Original bounded record hash changed')
    return value['record']


class NativeVerificationClient:
    def __init__(self,key):
        if key is not _KEY:
            raise TypeError('Use original inherited-ticket admission')

    @classmethod
    def open(cls,ticket_path,fd):
        if sys.platform != 'linux' or type(fd) is not int or fd < 3:
            raise ProtocolError('Linux original inherited nonstdio descriptor required')
        channel = None
        try:
            channel = socket.socket(fileno=fd)
            if channel.family != socket.AF_UNIX or channel.getsockopt(socket.SOL_SOCKET,socket.SO_TYPE) != socket.SOCK_SEQPACKET:
                raise ProtocolError('Original AF_UNIX SOCK_SEQPACKET required')
            peer = struct.unpack('3i',channel.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,struct.calcsize('3i')))
            if peer != (os.getppid(),os.getuid(),os.getgid()):
                raise ProtocolError('Original direct parent credentials required')
            path = _path(ticket_path)
            with path.open('rb') as handle:
                raw = handle.read(MAX_PACKET+1)
            ticket,digest = parse_ticket(raw)
            if peer[0] != ticket['lease']['parent_pid'] or str(path) != ticket['task_roots'][0]+'/native-ticket.json':
                raise ProtocolError('Ticket does not belong to original parent/task')
            self = cls(_KEY)
            self._channel=channel;self._ticket=ticket;self._digest=digest;self._peer=peer
            self._owner=(os.getpid(),threading.get_ident());self._sequence=0
            self._closed=False;self._finished=False;self._terminal=False;self._previous=None
            self._gate_ready=False
            from .work_lease import FrozenWorkLease,_utc
            from datetime import timedelta
            self._lease=FrozenWorkLease.from_dict(ticket['lease'])
            fields=dict(ticket['service_clock']['issuer'])
            fields['wall_deadline']=(_utc(fields['wall_started_at'])+timedelta(seconds=7200)).isoformat()
            fields['monotonic_deadline']=fields['monotonic_started']+7200
            self._supervisor=SimpleNamespace(**fields)
            channel.set_inheritable(False)
            channel.setsockopt(socket.SOL_SOCKET,socket.SO_PASSCRED,1)
            self._live();self._await_start();self._admit_records()
            self._live()
            return self
        except BaseException:
            if channel is not None:channel.close()
            raise

    def _admit_records(self):
        from factorlab.protocol import project_path
        from .monthly_work_authority import source_files
        from .nominal_verification_budget import FrozenVerificationPlan
        files=source_files()
        if (Path(__file__).resolve()!=project_path(RELATIVE).resolve()
                or files!=self._ticket['source_files']
                or files.get(RELATIVE)!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
            raise ProtocolError('Original canonical worker source changed')
        roots=[_path(p) for p in self._ticket['task_roots']]
        registry=_record(roots[0].parent.parent/'identity.json')
        binding=registry['plan']['binding'];plan=FrozenVerificationPlan(binding)
        if (_digest(registry)!=self._ticket['registry_digest']
                or registry['directory']!=str(roots[0].parent.parent)
                or registry['anchor']!=str(roots[1].parent.parent)
                or registry['plan']!=plan.to_dict() or binding['source_files']!=files
                or plan.task(self._ticket['task']['index'])!=self._ticket['task']):
            raise ProtocolError('Original whole registry/task/source differs')
        for root in roots:
            if (_record(root.parent.parent/'identity.json')!=registry
                    or _record(root/'native-ticket.json')!=self._ticket
                    or (root/'failure.json').exists()):
                raise ProtocolError('Original ticket/registry mirrors failed or changed')
            intent=_record(root/'intent.json')
            service=_record(root.parent.parent/'services'/f"{self._ticket['task']['service']:06d}"/'intent.json')
            if (_digest(intent)!=self._ticket['task_intent_hash']
                    or intent['registry_digest']!=self._ticket['registry_digest']
                    or intent['task']!=self._ticket['task'] or intent['lease']!=self._ticket['lease']
                    or intent['service_binding']!=self._ticket['service_binding_hash']
                    or _digest(service)!=self._ticket['service_binding_hash']
                    or service['registry_digest']!=self._ticket['registry_digest']
                    or service['clock']!=self._ticket['service_clock']
                    or service['plan']!=plan.service(self._ticket['task']['service'])
                    or (root.parent.parent/'services'/f"{self._ticket['task']['service']:06d}"/'failure.json').exists()
                    or (root.parent.parent/'registration-failure.json').exists()):
                raise ProtocolError('Original allocated task/service/lease differs')
        cgroup=Path(self._ticket['worker_cgroup'])
        if cgroup.is_symlink() or cgroup.resolve()!=cgroup or not cgroup.is_dir():
            raise ProtocolError('Original worker cgroup path unavailable')
        lines=Path('/proc/self/cgroup').read_text('ascii').splitlines()
        expected=str(cgroup)[len('/sys/fs/cgroup'):]
        if lines!=['0::'+expected]:
            raise ProtocolError('Worker not in its original dedicated cgroupv2')

    def _live(self):
        from .work_lease import _observe,_evaluate
        if self._closed or self._owner!=(os.getpid(),threading.get_ident()):
            raise ProtocolError('Closed or moved original client')
        observation=_observe(self._lease)
        if observation.get('pid')==self._lease.parent_pid:
            raise ProtocolError('Issuer cannot impersonate its child')
        checks=[_evaluate(lease,observation,self._previous) for lease in (self._lease,self._supervisor)]
        if not all(c.allowed for c in checks):
            raise ProtocolError('Original task/service lease closed')
        self._previous=observation
        return min(c.remaining_seconds for c in checks)

    def _receive(self,sequence):
        self._channel.settimeout(self._live())
        raw,ancillary,flags,_=self._channel.recvmsg(MAX_PACKET,socket.CMSG_SPACE(struct.calcsize('3i')))
        if flags & (socket.MSG_TRUNC|socket.MSG_CTRUNC):raise ProtocolError('Truncated response/credentials')
        expected=(socket.SOL_SOCKET,socket.SCM_CREDENTIALS)
        if (len(ancillary)!=1 or ancillary[0][:2]!=expected or len(ancillary[0][2])!=struct.calcsize('3i')
                or struct.unpack('3i',ancillary[0][2])!=self._peer):
            raise ProtocolError('Original parent response credentials changed')
        self._live()
        return validate_reply(raw,self._digest,sequence)

    def _await_start(self):
        try:
            if self._gate_ready or self._sequence!=0:
                raise ProtocolError('Start gate cannot repeat')
            value=self._receive(-1)
            if (type(value) is not dict or set(value)!={'start_original_task'}
                    or value['start_original_task'] is not True):
                raise ProtocolError('Original fixed parent start gate required')
            self._gate_ready=True
        except BaseException:
            self.close();raise

    def _request(self,op,args):
        if not self._gate_ready or self._terminal or op not in OPS:
            raise ProtocolError('Terminal or unsupported original request')
        if op in ('consume','resource_finish') and self._finished or op=='publication_ready' and not self._finished:
            raise ProtocolError('Original completion order changed')
        if op=='fail':
            if type(args) is not dict or set(args)!={'reason'} or type(args['reason']) is not str or not 0<len(args['reason'])<=1000:
                raise ProtocolError('Bounded failure reason required')
        elif args!={}:raise ProtocolError('No caller-selected operation arguments')
        try:
            remaining=self._live();sequence=self._sequence
            packet=encode_packet({'version':VERSION,'ticket_digest':self._digest,'sequence':sequence,'op':op,'args':args})
            self._sequence+=1  # Never retry an uncertain send/response.
            self._channel.settimeout(remaining)
            if self._channel.send(packet)!=len(packet):raise ProtocolError('Partial seqpacket send')
            value=self._receive(sequence)
            if op=='consume':
                if (type(value) is not dict or set(value)!={'resource_finished','semantic_provisional'}
                        or value['resource_finished'] is not True or value['semantic_provisional'] is not True):
                    raise ProtocolError('Fixed parent task completion summary required')
            if op in ('consume','resource_finish'):self._finished=True
            if op in ('fail','publication_ready'):self._terminal=True
            return value
        except BaseException:
            self.close();raise

    def check(self):return self._request('check',{})
    def consume(self):return self._request('consume',{})
    def resource_finish(self):return self._request('resource_finish',{})
    def publication_ready(self):return self._request('publication_ready',{})
    def fail(self,reason):return self._request('fail',{'reason':reason})
    def close(self):
        if not self._closed:self._closed=True;self._channel.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def main(argv=None):
    parser=argparse.ArgumentParser(allow_abbrev=False,description=__doc__)
    parser.add_argument('--ticket',required=True);parser.add_argument('--fd',required=True,type=int)
    values=list(sys.argv[1:] if argv is None else argv)
    if values!=['--help'] and (len(values)!=4 or values.count('--ticket')!=1 or values.count('--fd')!=1):
        parser.error('Only --ticket PATH --fd INT is supported')
    args=parser.parse_args(values)
    try:
        with NativeVerificationClient.open(args.ticket,args.fd) as client:
            try:
                client.check()
                client.consume()
                client.publication_ready()
            except Exception as error:
                try:client.fail(('Original parent task failed: '+type(error).__name__+':'+str(error))[:1000])
                except Exception:pass  # Closed/uncertain channels cannot retry or replace the attempt.
                raise
        return 0
    except Exception as error:
        print('Native verification worker refused: '+type(error).__name__+':'+str(error)[:1000],file=sys.stderr)
        return 125


if __name__=='__main__':
    raise SystemExit(main())
