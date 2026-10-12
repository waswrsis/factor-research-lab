"""Live Linux child-exit facts for ONE original pure verification task.

No launch, signal, reap, quota, semantic publication or durable certificate.
Only the original parent/thread can retain this authority. Fixture verdicts
never enter the native API. Kernel state must remain available on every read.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import threading
from fractions import Fraction

import psutil
from factorlab.protocol import fingerprint,project_path
from .work_lease import FrozenWorkLease,_observe,_evaluate
from .full_trial_controller import ControllerStepLease

RELATIVE='factorlab/discovery/native_verification_terminal.py'
VERSION='live-native-verification-terminal-v1'
WORKER='factorlab.discovery.native_verification_worker'
MAX_FILE_BYTES=65536
MAX_SUBGROUPS=128
MAX_MEMBERS=64
_TOKEN=object()


def _plain(value):
    return json.loads(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False))


def _bounded(path):
    path=Path(path)
    if path.is_symlink():raise ValueError('Native file alias refused')
    with path.open('rb') as stream:raw=stream.read(MAX_FILE_BYTES+1)
    if len(raw)>MAX_FILE_BYTES:raise ValueError('Native file exceeds fixed read bound')
    return raw.decode('ascii')


def _proc(pid):
    if type(pid) is not int or pid<=0:raise ValueError('Exact native PID required')
    raw=_bounded(Path('/proc')/str(pid)/'stat')
    tail=raw.rsplit(')',1)[1].split()
    if int(raw.split(' ',1)[0])!=pid or len(tail)<20:raise ValueError('Incomplete original proc stat')
    value={'pid':pid,'state':tail[0],'ppid':int(tail[1]),'start_ticks':int(tail[19]),'raw':raw}
    if value['ppid']<=0 or value['start_ticks']<=0:raise ValueError('Original proc identity invalid')
    return value


def _group(pid):
    rows=[s[3:] for s in _bounded(Path('/proc')/str(pid)/'cgroup').splitlines() if s.startswith('0::')]
    if len(rows)!=1 or not rows[0].startswith('/') or '..' in Path(rows[0]).parts:
        raise ValueError('Original unified cgroup membership unavailable')
    return _group_path(Path('/sys/fs/cgroup')/rows[0].lstrip('/'))


def _group_path(value):
    path=Path(value);root=Path('/sys/fs/cgroup')
    if (not path.is_absolute() or not path.is_relative_to(root) or '..' in path.parts
            or path.resolve()!=path or not path.is_dir()):
        raise ValueError('Original cgroup path missing or redirected')
    for p in (path,*path.parents):
        if p==root.parent:break
        if p.is_symlink():raise ValueError('Native cgroup alias refused')
    return path


def _values(path):
    names=('cpu.max','memory.high','memory.max','memory.swap.max','pids.max','memory.events')
    return {name:_bounded(path/name).strip() for name in names}


def _limits(raw):
    cpu=raw['cpu.max'].split()
    if len(cpu)!=2 or any(not x.isascii() or not x.isdigit() for x in cpu):return False
    if int(cpu[1])<=0 or Fraction(int(cpu[0]),int(cpu[1]))!=Fraction(1,1):return False
    if any(raw[k]!=str(v) for k,v in {'memory.high':640*2**20,'memory.max':768*2**20,
            'memory.swap.max':0,'pids.max':64}.items()):return False
    pairs=[s.split() for s in raw['memory.events'].splitlines()]
    if any(len(p)!=2 for p in pairs) or len({p[0] for p in pairs})!=len(pairs):return False
    events=dict(pairs)
    return all(events.get(k)=='0' for k in ('oom','oom_kill'))


def _shared(worker,parent_pid):
    parent=_group(parent_pid)
    if parent==worker or parent.is_relative_to(worker):raise ValueError('Worker cgroup is not exclusive of parent')
    root=Path('/sys/fs/cgroup')
    for p in (parent,*parent.parents):
        if not p.is_relative_to(root):break
        if worker!=p and worker.is_relative_to(p):
            try:raw=_values(p)
            except OSError:continue
            if _limits(raw):return {'path':str(p),'inode':p.stat().st_ino,'raw':raw}
    raise ValueError('Actual shared CPU100/640soft768hard/swap0/tasks64 boundary missing')


def _subtree(worker):
    worker=_group_path(worker);pending=[worker];rows=[];members=set()
    while pending:
        if len(rows)+len(pending)>MAX_SUBGROUPS:raise ValueError('Native subtree exceeds registered read bound')
        p=_group_path(pending.pop())
        if not p.is_relative_to(worker):raise ValueError('Foreign descendant cgroup')
        events=_bounded(p/'cgroup.events')
        pairs=[s.split() for s in events.splitlines()]
        if any(len(v)!=2 for v in pairs) or len(pairs)!=len({v[0] for v in pairs}):raise ValueError('Ambiguous cgroup events')
        values=dict(pairs)
        if values.get('populated') not in ('0','1'):raise ValueError('Native population unavailable')
        raw=_bounded(p/'cgroup.procs');pids=[int(v) for v in raw.split()]
        if any(v<=0 for v in pids) or len(pids)!=len(set(pids)):raise ValueError('Native members invalid')
        members.update(pids)
        if len(members)>MAX_MEMBERS:raise ValueError('Native subtree exceeds original task bound')
        rows.append({'path':str(p),'inode':p.stat().st_ino,'events':events,'members_raw':raw,
                     'populated':values['populated']=='1','members':pids})
        children=[]
        for v in p.iterdir():
            if v.is_dir():
                children.append(v)
                if len(rows)+len(pending)+len(children)>MAX_SUBGROUPS:
                    raise ValueError('Native subtree exceeds registered read bound')
        pending.extend(sorted(children))
    return {'nodes':rows,'members':sorted(members),'populated':any(v['populated'] for v in rows)}


def _installed(task):
    from .nominal_verification_budget import VerificationTask
    from .monthly_work_authority import source_files
    from .evidence import file_hash
    if type(task) is not VerificationTask:raise TypeError('Exact original verification task required')
    files=source_files()
    if (Path(__file__).resolve()!=project_path(RELATIVE).resolve() or files.get(RELATIVE)!=file_hash(__file__)
            or files!=task.service.registry.plan.binding['source_files']):
        raise ValueError('Canonical terminal reader and original task source required')
    task.service.registry.verify()
    return files


def _task_records(task,*,completed):
    from .nominal_verification_budget import _read
    if any((p/'failure.json').exists() for p in (*task.roots,*task.service.roots)):
        raise ValueError('Original task/service failure dominates terminal facts')
    if any(_read(p/'intent.json')!=task.intent for p in task.roots):raise ValueError('Original task intents changed')
    if any(_read(p/'intent.json')!=task.service.binding for p in task.service.roots):raise ValueError('Original service intents changed')
    value={'task_intent':fingerprint(task.intent),'service_binding':fingerprint(task.service.binding)}
    if completed:value['resource_terminal']=fingerprint(task.service.registry.task_witness(task.task['index']))
    return value


def _command(process):
    args=process.args
    if (type(args) is not list or len(args)!=8 or args[:5]!=[sys.executable,'-B','-m',WORKER,'--ticket']
            or args[6]!='--fd' or type(args[7]) is not str or not args[7].isascii()
            or not args[7].isdigit() or str(int(args[7]))!=args[7] or int(args[7])<3
            or type(args[5]) is not str or not Path(args[5]).is_absolute()
            or project_path(args[5])!=Path(args[5])):
        raise ValueError('Only fixed original native verification worker argv accepted')
    from .calibration_trial_context import _no_redirect
    _no_redirect(Path(args[5]),project_path('.'))
    return list(args)


def _wait(fd,pid):
    kind=getattr(os,'P_PIDFD',None)
    if kind is None:kind,identifier=os.P_PID,pid
    else:identifier=fd
    value=os.waitid(kind,identifier,os.WEXITED|os.WNOHANG|os.WNOWAIT)
    if value is None or value.si_pid==0:return None
    return {k:int(getattr(value,k)) for k in ('si_pid','si_uid','si_signo','si_status','si_code')}


def _predicate(binding,sample,*,historic=None):
    """Internal deterministic facts; it cannot manufacture a live handle."""
    lease=FrozenWorkLease.from_dict(binding['lease'])
    service=ControllerStepLease.from_dict(binding['service_clock'],type('Schedule',(),{'service':{'wall_seconds':7200}})())
    if lease.lease_seconds!=300 or any(getattr(lease,k)!=service.issuer[k] for k in ('parent_pid','parent_created','boot_id')):
        return 'unknown','original_lease_service_identity'
    first=sample if historic is None else historic
    if (not _evaluate(lease,first['started']).allowed or not _evaluate(lease,first['ended'],first['started']).allowed
            or not service.evaluate(first['ended'],first['started']).allowed):
        return 'failed','original_absolute_deadline_or_identity'
    current=sample['ended']
    if (current['parent_pid']!=lease.parent_pid or current['pid']!=lease.parent_pid
            or current['parent_created']!=lease.parent_created or current['boot_id']!=lease.boot_id
            or current['parent_alive'] is not True or current['monotonic']<sample['started']['monotonic']
            or current['wall']<sample['started']['wall'] or current['monotonic']<first['ended']['monotonic']
            or current['wall']<first['ended']['wall']):return 'unknown','original_parent_or_clock_lost'
    child=sample['child'];old=binding['child']
    if any(type(child[k]) is not int or child[k]<=0 for k in ('pid','ppid','start_ticks')):
        return 'unknown','invalid_native_child_identity'
    if any(child[k]!=old[k] for k in ('pid','ppid','start_ticks')):return 'unknown','original_child_identity_lost'
    if sample['worker']!=binding['worker'] or sample['shared']['path']!=binding['shared']['path'] or sample['shared']['inode']!=binding['shared']['inode']:
        return 'unknown','original_cgroup_identity_lost'
    if not _limits(sample['shared']['raw']):return 'failed','shared_resource_boundary_changed_or_oom'
    event=sample['wait']
    if event is None:return 'running','no_live_exit_event'
    if set(event)!={'si_pid','si_uid','si_signo','si_status','si_code'} or any(type(v) is not int for v in event.values()):
        return 'unknown','invalid_native_wait_event'
    if (event['si_pid']!=old['pid'] or event['si_uid']!=binding['uid']
            or event['si_signo']!=int(getattr(signal,'SIGCHLD',17))):return 'unknown','foreign_native_wait_event'
    if event['si_code']!=1 or event['si_status']!=0:return 'failed','not_natural_zero_exit'
    if child['state']!='Z':return 'unknown','original_waitable_zombie_unavailable'
    if sample['subtree']['members'] or sample['subtree']['populated']:return 'failed','owned_worker_descendants_not_empty'
    if historic is not None and event!=historic['wait']:return 'unknown','original_native_event_changed'
    return 'native_observed_natural_exit','live_nonreaped_kernel_event_only'


def evaluate_fixture(binding,sample,*,historic=None):
    """Artificial pure checks. Output is never accepted by native entry points."""
    try:status,reason=_predicate(_plain(binding),_plain(sample),historic=None if historic is None else _plain(historic))
    except (ValueError,TypeError,KeyError,OSError,OverflowError):status,reason='unknown','invalid_fixture'
    return {'origin':'artificial_fixture_not_native','fixture_predicate':status,'reason':reason,
            'native_authority':False,'formal_acceptance':False}


class LiveTerminalHandle:
    """Nonserializable original-parent authority; all dictionaries are diagnostics."""
    def __init__(self,*args,**kwargs):raise TypeError('Use attach_original_child')
    def __reduce__(self):raise TypeError('Live kernel authority cannot serialize or resume')
    @property
    def task(self):return self._task
    @property
    def process(self):return self._process
    @property
    def worker_cgroup(self):return Path(self._binding['worker']['path'])
    def _owner(self):
        if (self._token is not _TOKEN or self._closed or sys.platform!='linux'
                or self._invalidated is not None or self._owner_id!=(os.getpid(),threading.get_ident())):
            raise ValueError('Original live Linux parent/thread handle unavailable')
        stat=os.fstat(self._fd)
        if (stat.st_dev,stat.st_ino)!=self._fd_identity:raise ValueError('Original pidfd replaced')
        if self._process.returncode is not None:raise ValueError('Popen child already reaped')
        if _command(self._process)!=self._binding['command']:raise ValueError('Original fixed worker command changed')
    def _sample(self):
        self._owner()
        begin=_observe(self._task.lease)
        files=_installed(self._task)
        if files!=self._source:raise ValueError('Original source changed')
        records=_task_records(self._task,completed=True)
        if self._accepted_records is not None and records!=self._accepted_records:raise ValueError('Original completed resource records changed')
        child=_proc(self._process.pid)
        group=_group(self._process.pid)
        worker={'path':str(group),'inode':group.stat().st_ino}
        shared=_shared(group,self._owner_id[0]);subtree=_subtree(group)
        event=_wait(self._fd,self._process.pid)
        # Re-read proc after wait to detect reap/identity loss during collection.
        after=_proc(self._process.pid)
        if any(after[k]!=child[k] for k in ('pid','ppid','start_ticks')):raise ValueError('Child changed across native wait')
        child=after
        self._owner()
        sample={'started':begin,'ended':_observe(self._task.lease),'child':child,
                'worker':worker,'shared':shared,'subtree':subtree,'wait':event}
        return sample,records
    def observe(self):
        """First success must be observed under the original300/7200 clocks."""
        if self._first is not None:return self.read_live()
        return self._inspect(first=True)
    def read_live(self):
        """Re-read the same actual kernel event. No old dict can resume this."""
        if self._first is None:return {'origin':'live_kernel_reader','status':'unknown','reason':'no_original_timely_observation','formal_acceptance':False}
        return self._inspect(first=False)
    def _inspect(self,*,first):
        try:
            sample,records=self._sample()
            status,reason=_predicate(self._binding,sample,historic=None if first else self._first)
            if status=='native_observed_natural_exit' and first:
                self._first=_plain(sample);self._accepted_records=records
            return {'origin':'live_kernel_reader','status':status,'reason':reason,
                    'binding_digest':fingerprint(self._binding),'sample':sample,
                    'historical_first_observation':self._first,'resource_records':records,
                    'semantic_publication_certified':False,'durable_authority':False,'formal_acceptance':False}
        except (OSError,ValueError,TypeError,KeyError,RuntimeError,psutil.Error) as error:
            return {'origin':'live_kernel_reader','status':'unknown','reason':type(error).__name__+':'+str(error)[:300],
                    'semantic_publication_certified':False,'durable_authority':False,'formal_acceptance':False}
    def close(self):
        """Close only pidfd; never wait/reap/signal/terminate or confer success."""
        if self._closed:return
        if self._token is not _TOKEN or self._owner_id!=(os.getpid(),threading.get_ident()):
            raise ValueError('Only original handle owner can close pidfd')
        os.close(self._fd);self._closed=True

    def invalidate(self,reason):
        """Irreversible refusal before root's any abort/cleanup; cannot grant."""
        if self._token is not _TOKEN or self._owner_id!=(os.getpid(),threading.get_ident()):
            raise ValueError('Only original handle owner can invalidate')
        if self._invalidated is None:self._invalidated=str(reason)[:300]


def attach_original_child(task,process,*,worker_cgroup):
    """Capture facts for root's fixed typed-attempt protocol; does not authorize it.

    Call after the original direct child joins its pre-registered exclusive
    cgroup, before releasing its IPC start gate. Never poll/wait/communicate it.
    """
    if sys.platform!='linux':raise OSError('Linux live pidfd/waitid required; Windows unsupported')
    if type(process) is not subprocess.Popen or process.returncode is not None:
        raise TypeError('Original unreaped fixed Popen direct child required')
    if not hasattr(os,'pidfd_open') or not hasattr(os,'waitid'):raise OSError('Native pidfd/waitid unavailable')
    files=_installed(task);task.check()
    command=_command(process);pid=process.pid;child=_proc(pid)
    if child['ppid']!=os.getpid() or child['state']=='Z':raise ValueError('Live original direct child required at attach')
    group=_group(pid)
    if group!=_group_path(worker_cgroup):raise ValueError('Child did not join its exact pre-registered worker cgroup')
    shared=_shared(group,os.getpid());records=_task_records(task,completed=False)
    fd=os.pidfd_open(pid,0)
    try:
        after=_proc(pid)
        if any(child[k]!=after[k] for k in ('pid','ppid','start_ticks')):raise ValueError('Child identity changed across pidfd acquisition')
        task.check();clock=task.service.clock
        binding={'version':VERSION,'registry_digest':task.service.registry.digest,'task_index':task.task['index'],
                 'records':records,'source_files':files,'lease':task.lease.to_dict(),'service_clock':clock.to_dict(),
                 'child':child,'uid':os.getuid(),'worker':{'path':str(group),'inode':group.stat().st_ino},
                 'shared':shared,'command':command}
        obj=object.__new__(LiveTerminalHandle)
        obj._token=_TOKEN;obj._closed=False;obj._invalidated=None;obj._task=task;obj._process=process;obj._fd=fd
        info=os.fstat(fd);obj._fd_identity=(info.st_dev,info.st_ino)
        obj._owner_id=(os.getpid(),threading.get_ident());obj._source=files;obj._binding=_plain(binding)
        obj._first=None;obj._accepted_records=None
        return obj
    except BaseException:
        os.close(fd);raise
