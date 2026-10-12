"""Source-bound native history reader; never reconstruct authority from JSON.

Stored snapshots are provisional. Public success also needs the current original
typed live batch, or a freshly queried retained systemd unit with natural exit.
Actual trial batches require their own OS producer, never verification units.
"""
from pathlib import Path
import hashlib
import importlib
import os
import sys
import threading

from factorlab.protocol import fingerprint, project_path
from .evidence import file_hash
from .monthly_work_authority import source_files
from .nominal_verification_budget import VerificationRegistry, _path, _read

VERSION = 'original-nominal-native-completion-reader-v1'
RELATIVE = 'factorlab/discovery/nominal_native_completion.py'
MAX_RECORD = 2**20


def _digest(value):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Exact lowercase digest required')
    return value


def _module(name, files):
    module = importlib.import_module('factorlab.discovery.'+name)
    relative = 'factorlab/discovery/'+name+'.py'
    if (Path(module.__file__).resolve() != project_path(relative).resolve()
            or files.get(relative) != file_hash(module.__file__)):
        raise ValueError('Original integrated native reader/producer source required')
    return module


def _sha(path):
    path = _path(path)
    with path.open('rb') as handle:
        raw = handle.read(MAX_RECORD+1)
    if len(raw) > MAX_RECORD:
        raise ValueError('Original native artifact exceeds bounded record size')
    return hashlib.sha256(raw).hexdigest()


def _mirrors(roots, name):
    first = _read(roots[0]/name)
    if any(_read(root/name) != first for root in roots[1:]):
        raise ValueError('Original native record mirrors differ')
    return first


def _target(quota, task):
    grid = _path(quota.plan.binding['target_directory'])
    kind = task['kind']
    if kind == 'cold_atom_evidence_validation':
        return grid/f"atom-{task['unit']['atom']:03d}.json"
    if kind == 'law_reduction':return grid/'grid.json'
    if kind in ('full_law_verify_before_actual_registry','full_law_verify_before_final_report'):
        return grid/f"verified-{task['index']:06d}.json"
    if kind == 'actual_coordinate_readback':
        return grid/'full-N'/f"coordinate-{task['index']:06d}.json"
    if kind == 'acceptance_cell_aggregation':
        ordinal = next(i for i,c in enumerate(quota.plan.acceptance.cells) if c.name == task['unit']['cell'])
        return grid/'full-N'/f'cell-{ordinal:03d}.json'
    if kind == 'final_full_N_report':return grid/'full-N'/'report.json'
    raise ValueError('No original native task mapping')


def _publication_paths(roots, target):
    return [*(r/'semantic-output.json' for r in roots),
            *(r/'semantic-ack.json' for r in roots), target,
            target.with_name(target.name+'.semantic-ack.json'),
            *(r/name for r in roots for name in
              ('intent.json','terminal.json','successor.json','native-ticket.json'))]


def _snapshot_check(snapshot, data, batch, files, terminal_api):
    """Pure stored-data consistency; NEVER sufficient for public success."""
    fields={'version','registry_digest','service_index','task_index','batch_intent_hash',
        'task_intent','task_intent_hash','ticket_digest','ticket','lease','source_files',
        'unit_binding_digest','target','semantic_payload_hash','publication','terminal_binding',
        'native_observation','native_recheck','native_observation_digest','provisional_only'}
    task=data['task'];intent=data['intent'];ticket=data['ticket']
    if (type(snapshot) is not dict or set(snapshot)!=fields
            or snapshot['version']!='native-verification-task-provisional-v1'
            or snapshot['registry_digest']!=intent['registry_digest']
            or snapshot['service_index']!=task['service'] or snapshot['task_index']!=task['index']
            or snapshot['batch_intent_hash']!=fingerprint(batch)
            or snapshot['task_intent']!=intent or snapshot['task_intent_hash']!=fingerprint(intent)
            or snapshot['ticket_digest']!=fingerprint(ticket) or snapshot['ticket']!=ticket
            or snapshot['lease']!=intent['lease'] or snapshot['source_files']!=files
            or snapshot['unit_binding_digest']!=batch['unit_binding_digest']
            or snapshot['target']!=data['ack']['target']
            or snapshot['semantic_payload_hash']!=fingerprint(data['record']['payload'])
            or snapshot['publication']!=data['publication']
            or snapshot['provisional_only'] is not True
            or snapshot['native_observation_digest']!=fingerprint(snapshot['native_observation'])):
        raise ValueError('Original provisional snapshot differs from whole task/publication')
    binding=snapshot['terminal_binding']
    if (binding['registry_digest']!=intent['registry_digest'] or binding['task_index']!=task['index']
            or binding['source_files']!=files or binding['lease']!=intent['lease']
            or binding['service_clock']!=data['service']['clock']
            or binding['worker']['path']!=ticket['worker_cgroup']
            or binding['records']!={'task_intent':fingerprint(intent),'service_binding':fingerprint(data['service'])}):
        raise ValueError('Stored original native task binding differs')
    command=binding['command']
    unit_command=batch['unit_binding']['command']
    if (type(command) is not list or len(command)!=8 or command[0]!=unit_command[0]
            or command[1:5]!=['-B','-m','factorlab.discovery.native_verification_worker','--ticket']
            or command[5]!=str(data['roots'][0]/'native-ticket.json') or command[6]!='--fd'
            or type(command[7]) is not str or not command[7].isascii() or not command[7].isdigit()
            or str(int(command[7]))!=command[7] or int(command[7])<3):
        raise ValueError('Stored fixed original worker command differs')
    first=None
    from .work_lease import FrozenWorkLease,_evaluate
    lease=FrozenWorkLease.from_dict(intent['lease'])
    for value in (snapshot['native_observation'],snapshot['native_recheck']):
        if (value.get('status')!='single_task_publication_bound_to_live_exit'
                or value.get('durable_authority') is not False or value.get('formal_acceptance') is not False):
            raise ValueError('Stored native observations cannot declare durable authority')
        diagnostic=value['native_diagnostic']
        if first is None:first=diagnostic['historical_first_observation']
        if (diagnostic['origin']!='live_kernel_reader'
                or diagnostic['status']!='native_observed_natural_exit'
                or diagnostic['binding_digest']!=fingerprint(binding)
                or diagnostic['historical_first_observation']!=first
                or diagnostic['resource_records']!={'task_intent':fingerprint(intent),
                    'service_binding':fingerprint(data['service']),
                    'resource_terminal':fingerprint(data['terminal'])}
                or diagnostic['durable_authority'] is not False
                or not _evaluate(lease,diagnostic['sample']['ended'],first['ended']).allowed
                or terminal_api._predicate(binding,diagnostic['sample'],historic=first)[0]!='native_observed_natural_exit'):
            raise ValueError('Stored native exit/deadline data inconsistent')
    return first['wait']


class NativeCompletionReader:
    def __init__(self, quota):
        if type(quota) is not VerificationRegistry:
            raise TypeError('Exact original whole verification registry required')
        self.quota = quota
        self._owner = (os.getpid(), threading.get_ident())

    def _original(self):
        if (type(self.quota) is not VerificationRegistry
                or self._owner != (os.getpid(), threading.get_ident())):
            raise ValueError('Native reader cannot change process/thread')
        files = source_files()
        if (Path(__file__).resolve() != project_path(RELATIVE).resolve()
                or files.get(RELATIVE) != file_hash(__file__)
                or files != self.quota.plan.binding['source_files']):
            raise ValueError('Original whole quota/source changed')
        self.quota.verify()
        for name in ('nominal_verification_budget','nominal_cold_effect_grid',
                     'native_verification_bridge','native_verification_terminal',
                     'native_verification_worker','native_verification_launcher'):
            _module(name,files)
        return files

    def _records(self, index, target, semantic_hash):
        """Data consistency only; caller still must have original OS authority."""
        task = self.quota.plan.task(index)
        if type(index) is not int or index <= 0 or target != _target(self.quota,task):
            raise ValueError('Original task-specific target required')
        roots = tuple(p/'tasks'/f'{index:06d}' for p in (self.quota.directory,self.quota.anchor))
        service_roots = tuple(p/'services'/f"{task['service']:06d}" for p in (self.quota.directory,self.quota.anchor))
        for root in (*roots,*service_roots,self.quota.directory,self.quota.anchor):
            for name in ('failure.json','registration-failure.json','native-failure.json'):
                if (root/name).exists():raise ValueError('Original failure dominates native snapshots')
        for root in service_roots:
            if (root/'native-batch'/'failure.json').exists():
                raise ValueError('Failed original native batch')
        intent = _mirrors(roots,'intent.json')
        service = _mirrors(service_roots,'intent.json')
        ticket = _mirrors(roots,'native-ticket.json')
        from .native_verification_worker import parse_ticket,encode_packet
        parsed,ticket_digest=parse_ticket(encode_packet({'record':ticket,'hash':fingerprint(ticket)}))
        if parsed!=ticket or ticket_digest!=fingerprint(ticket):
            raise ValueError('Original fixed native ticket schema differs')
        if (intent['registry_digest'] != self.quota.digest or intent['task'] != task
                or intent['service_binding'] != fingerprint(service)
                or service['registry_digest'] != self.quota.digest
                or service['plan'] != self.quota.plan.service(task['service'])
                or ticket['registry_digest'] != self.quota.digest or ticket['task'] != task
                or ticket['task_intent_hash'] != fingerprint(intent)
                or ticket['service_binding_hash'] != fingerprint(service)
                or ticket['lease'] != intent['lease'] or ticket['service_clock'] != service['clock']
                or ticket['source_files'] != self.quota.plan.binding['source_files']
                or ticket['task_roots'] != [str(p) for p in roots]):
            raise ValueError('Original task/service/ticket binding differs')
        terminal = self.quota.task_witness(index)
        successor = _mirrors(roots,'successor.json')
        record = _mirrors(roots,'semantic-output.json')
        ack = _mirrors(roots,'semantic-ack.json')
        from .nominal_cold_effect_grid import VERSION as GRID_VERSION, ACK_VERSION
        grid_intent = _read(_path(self.quota.plan.binding['target_directory'])/'intent.json')
        if (_read(target) != record or _read(target.with_name(target.name+'.semantic-ack.json')) != ack
                or grid_intent['version']!=GRID_VERSION or record['version']!=GRID_VERSION
                or ack['version']!=ACK_VERSION
                or grid_intent['verification_digest'] != self.quota.digest
                or grid_intent['source_files'] != self.quota.plan.binding['source_files']
                or grid_intent['census_digest'] != self.quota.plan.binding['census_digest']
                or record['grid_intent_hash'] != fingerprint(grid_intent)
                or ack['grid_intent_hash'] != fingerprint(grid_intent)
                or fingerprint(record['payload']) != semantic_hash
                or record['registry_digest'] != self.quota.digest or record['task'] != task
                or record['task_intent_hash'] != fingerprint(intent) or record['lease'] != intent['lease']
                or record['formal_acceptance'] is not False
                or ack['target'] != str(target) or ack['registry_digest'] != self.quota.digest
                or ack['task'] != task or ack['task_intent_hash'] != fingerprint(intent)
                or ack['lease'] != intent['lease'] or ack['semantic_record_hash'] != fingerprint(record)
                or ack['semantic_file_sha256'] != _sha(target)
                or ack['resource_terminal_hash'] != fingerprint(terminal)
                or ack['resource_successor_hash'] != fingerprint(successor)
                or ack['formal_acceptance'] is not False):
            raise ValueError('Original semantic resource/ACK/payload differs')
        from .work_lease import FrozenWorkLease, _evaluate
        lease=FrozenWorkLease.from_dict(intent['lease'])
        if (not _evaluate(lease,record['observed']).allowed
                or not _evaluate(lease,terminal['observed'],record['observed']).allowed
                or not _evaluate(lease,ack['observed'],successor['observed']).allowed):
            raise ValueError('Original semantic observation outside original task deadline')
        # No mtime statement is authority. The live snapshot must commit these
        # very bytes before the original native observation and bounded exit.
        publications = {str(p):_sha(p) for p in _publication_paths(roots,target)}
        return {'task':task,'roots':roots,'service_roots':service_roots,
            'intent':intent,'service':service,'ticket':ticket,'terminal':terminal,
            'record':record,'ack':ack,'publication':publications}

    def _batch_records(self, data, files):
        roots=tuple(p/'native-batch' for p in data['service_roots'])
        batch=_mirrors(roots,'intent.json')
        unit=_mirrors(data['service_roots'],'native-unit.json')
        service=data['task']['service'];plan=self.quota.plan.service(service)
        indices=list(range(plan['task_start'],plan['task_end_exclusive']))
        expected={'version':'native-verification-batch-v1','registry_digest':self.quota.digest,
            'service_index':service,'service_binding':data['service'],'task_indices':indices,
            'source_files':files,'unit_binding':unit,'unit_binding_digest':fingerprint(unit),
            'max_retained_children':23,'provisional_only':True}
        if batch!=expected or not 0<len(indices)<=23:
            raise ValueError('Original complete service batch/OS unit binding differs')
        return roots,batch,unit

    def _history(self, data, batch_roots, batch, unit, files, units, terminal):
        service=data['task']['service']
        authority=units.read_retained_service(self.quota,service)
        if type(authority) is not units.RetainedServiceObservation:
            raise TypeError('Actual retained systemd observation required, not stored facts')
        if authority.identity!=unit:
            raise ValueError('Original retained InvocationID/source/service differs')
        inventory=_mirrors(batch_roots,'inventory.json')
        reaped=_mirrors(batch_roots,'reaped.json')
        witness=self.quota.service_witness(service)
        rows=[];events=[]
        with self.quota.witness_read_scope():
            for index in batch['task_indices']:
                task=self.quota.plan.task(index);target=_target(self.quota,task)
                target_record=_read(target)
                original=self._records(index,target,fingerprint(target_record['payload']))
                snapshot=_mirrors(original['roots'],'native-provisional.json')
                wait=_snapshot_check(snapshot,original,batch,files,terminal)
                rows.append({'task_index':index,'snapshot_hash':fingerprint(snapshot)})
                events.append({'task_index':index,'wait':wait,'natural_reap':True,'cleanup_signal':False})
        expected={'version':'native-verification-batch-v1','registry_digest':self.quota.digest,
            'service_index':service,'batch_intent_hash':fingerprint(batch),
            'unit_binding_digest':fingerprint(unit),'source_files':files,'records':rows,
            'service_witness_hash':fingerprint(witness),'provisional_only':True,'reap_pending':True}
        if (inventory!=expected or reaped!={'version':'native-verification-batch-v1',
                'batch_intent_hash':fingerprint(batch),'inventory_hash':fingerprint(inventory),
                'reaped':events,'provisional_only':True,'cleanup_signals':False}):
            raise ValueError('Original whole-service inventory or natural zero reap differs')
        if authority.identity!=unit:raise ValueError('Retained original unit replaced during history read')
        return authority

    def verify_task(self, index, target, semantic_hash):
        files=self._original();_digest(semantic_hash)
        target=_path(target)
        data=self._records(index,target,semantic_hash)
        batch_roots,binding,unit=self._batch_records(data,files)
        batches=_module('native_verification_batch',files)
        terminal=_module('native_verification_terminal',files)
        active=batches.current_batch(self.quota,data['task']['service'])
        snapshot=_mirrors(data['roots'],'native-provisional.json')
        _snapshot_check(snapshot,data,binding,files,terminal)
        if active is not None:
            if type(active) is not batches.VerificationBatch:
                raise TypeError('Only original in-memory service batch may authorize live evidence')
            observation=active.read_task(index)
            if type(observation) is not batches.BatchTaskObservation or observation.snapshot!=snapshot:
                raise TypeError('Exact freshly observed original native snapshot required')
            fresh=observation.assert_current()
            if type(fresh) is not batches.BatchTaskObservation or fresh.snapshot!=snapshot:
                raise ValueError('Original live snapshot changed during read')
            authority=observation
        else:
            units=_module('nominal_systemd_unit',files)
            authority=self._history(data,batch_roots,binding,unit,files,units,terminal)
        # Recheck bytes and source after the potentially expensive native/history
        # read. Final authority assertion is a real OS query, not dict replay.
        if (self._records(index,target,semantic_hash)!=data or self._original()!=files
                or _mirrors(data['roots'],'native-provisional.json')!=snapshot
                or self._batch_records(data,files)!=(batch_roots,binding,unit)):
            raise ValueError('Original data/source changed across native completion read')
        current=authority.assert_current()
        if active is not None and (type(current) is not batches.BatchTaskObservation or current.snapshot!=snapshot):
            raise ValueError('Original live authority expired during read')
        return {'verification_digest':self.quota.digest,'task_index':index,
            'target':str(target),'semantic_hash':semantic_hash,'native_completed':True}

    def verify_trial(self, actualregistry, cell, index, outcome_hash):
        """Actual trial OS evidence is never supplied by verification workers.

        The physical-batch producer below is intentionally not implemented here.
        Its absence refuses; a directory, boolean or copied record cannot supply
        it. Future integration requires the exact producer API and source identity.
        All original real month/model allocations are preserved.
        """
        files = self._original()
        actual_module = _module('nominal_actual_trial',files)
        controller = _module('nominal_trial_controller',files)
        if type(actualregistry) is not actual_module.NominalActualRegistry:
            raise TypeError('Exact original actual trial registry required')
        _digest(outcome_hash)
        identity = actualregistry.verify()
        coordinate = actualregistry.coordinate(cell,index)
        if (identity['source_files'] != files
                or identity['grid']['verification_registry_digest'] != self.quota.digest
                or identity['acceptance_digest'] != self.quota.plan.acceptance.digest
                or identity['registered_N'] != self.quota.plan.total_N):
            raise ValueError('Actual trial belongs to another whole quota/source')
        producer = _module('nominal_trial_native_completion',files)
        # No verification-batch authority can satisfy this distinct original
        # physical-batch type. In the current source composition it is absent.
        trial = actual_module.NominalActualTrial(actualregistry,cell,index)
        batches = identity['work_batches']
        if not batches or [b['index'] for b in batches] != list(range(len(batches))):
            raise ValueError('Whole original physical batch allocation required')
        observations = []
        for batch in batches:
            native = producer.read_retained_batch(actualregistry,cell,index,batch['index'])
            if type(native) is not producer.RetainedTrialBatchObservation:
                raise TypeError('Exact original retained physical batch OS observation required')
            native.assert_current()
            roots = (trial.paths['root']/'controller_batches'/f"{batch['index']:02d}",
                actualregistry.directory/'controller_batches'/trial.digest/f"{batch['index']:02d}")
            terminal = controller.verify_completed_batch(trial,batch['index'])
            successor = _mirrors(roots,'successor.json')
            expected = {'coordinate':coordinate,'batch_index':batch['index'],'batch':batch,
                'controller_roots':[str(p) for p in roots],
                'controller_terminal_hash':fingerprint(terminal),
                'controller_successor_hash':fingerprint(successor),'source_files':files}
            native_identity = native.identity
            if any(native_identity.get(k) != v for k,v in expected.items()):
                raise ValueError('Original physical-batch OS identity differs')
            native.assert_current()
            observations.append((native,native_identity))
        outcome = controller.registered_outcome(actualregistry,cell,index)
        if (outcome['coordinate'] != coordinate or outcome['complete_library'] is not True
                or outcome['retain_N'] is not True or fingerprint(outcome) != outcome_hash):
            raise ValueError('Original actual conformance outcome differs')
        if actualregistry.verify() != identity or self._original() != files:
            raise ValueError('Original actual source/registry changed during native read')
        for native,expected in observations:
            native.assert_current()
            if native.identity != expected:raise ValueError('Physical unit replaced during trial read')
        return {'coordinate':coordinate,'outcome_hash':outcome_hash,'native_completed':True}

