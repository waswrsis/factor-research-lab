"""STAGE whole-owned-tree guard. Never a new spend/lease or model admission.

Linux entry must itself be the already capped systemd MainPID. Every registered
actor is a descendant of the one frozen controller, in that exact owned leaf.
The original service session/step and S outer clocks remain independently live.
No process-group fallback and no production Windows Campaign clock substitute.
"""
from __future__ import annotations

import json
import os
import runpy
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil
from filelock import FileLock

sys.path.insert(0,str(Path(__file__).resolve().parent))
import lease_enforcer as base
import multiwork_enforcer as multi

from factorlab import protocol
from scripts import v7_long_running as campaign


def _sources():
    paths=[Path(__file__),Path(multi.__file__),Path(base.__file__),
           Path(__file__).with_name('guarded_runtime.py'),
           Path(__file__).with_name('campaign_guard_entry.py')]
    return {str(p):base._hash(p) for p in paths}


class CampaignObservation:
    """Read-only original journal with incremental verification, no new session.

    One guard-start wall/mono anchor is used for all session deadlines; never
    translate a session deadline from its first observation or a refreshed now.
    Source bytes are frozen at entry and checked again at every actor admission.
    Old journal rows are fully verified on process restart, as by CampaignStore.
    """
    def __init__(self,directory):
        self.directory=protocol.project_path(directory)
        campaign.inspect_campaign(self.directory)
        self.manifest=(self.directory/'campaign.json').read_bytes()
        self.plan=campaign.read_plan(self.directory)
        self.store=campaign.CampaignStore.__new__(campaign.CampaignStore)
        self.store._verified_head=None;self.store.last_verification_rows=0
        self.store.db=sqlite3.connect((self.directory/'campaign.sqlite').as_uri()+'?mode=ro',uri=True)
        self.origin_wall=datetime.now(UTC).timestamp();self.origin_mono=time.monotonic()
        self.boot=multi._boot_id();self.previous=(self.origin_wall,self.origin_mono)
        self.original_end=datetime.fromisoformat(self.plan.end_at).timestamp()
        self.clock={'wall_started':datetime.fromtimestamp(self.origin_wall,UTC).isoformat(),
            'wall_deadline':self.plan.end_at,'monotonic_started':self.origin_mono,
            'monotonic_deadline':self.origin_mono+self.original_end-self.origin_wall,
            'boot_id':self.boot,'parent':base._identity(psutil.Process())}

    def read(self):
        if (self.directory/'campaign.json').read_bytes()!=self.manifest:
            raise ValueError('Original campaign manifest changed')
        if not (self.directory/'campaign.sqlite').is_file():
            raise ValueError('Campaign journal disappeared')
        state=self.store.load()
        self.previous=multi._clock(self.clock,self.previous)
        if (self.directory/'STOP').exists() or state['terminal']:
            raise RuntimeError('Original campaign is stopped or terminal')
        now_wall,now_mono=self.previous
        active=state['active_step']
        if active is not None and campaign.active_step_expired(state,active['launch_token'],{}):
            raise TimeoutError('Original active step deadline consumed')
        if state['sessions']:
            session=state['sessions'][-1]
            wall_started=datetime.fromisoformat(session['reserved_at']).timestamp()
            deadline=min(self.original_end,wall_started+session['wall_seconds'])
            # First observation/restart never grants another full duration.
            monotonic_deadline=self.origin_mono+deadline-self.origin_wall
            if now_wall<wall_started or now_wall>=deadline or now_mono>=monotonic_deadline:
                raise TimeoutError('Original prepaid session deadline consumed')
        return state

    def verify_actor(self,reference,issuer):
        state=self.read()
        if reference['kind']=='campaign':
            if protocol.project_path(reference['directory'])!=self.directory:
                raise ValueError('Actor belongs to a different campaign')
        elif reference['kind']=='S':
            ref=reference['bound']['reference']
            expected=protocol.project_path(self.plan.forward_directory)/'autonomous/jobs'
            if ref['kind']!='autonomous' or protocol.project_path(ref['path']).parent.parent!=expected:
                raise ValueError('Campaign accepts only its actual autonomous S job')
            path=protocol.project_path(ref['path'])
            with sqlite3.connect((path.parents[2]/'scheduler.sqlite').as_uri()+'?mode=ro',uri=True) as db:
                row=db.execute('SELECT value,hash FROM jobs WHERE month=?',(path.parent.name,)).fetchone()
            if row is None:raise ValueError('Missing actual S job')
            value=json.loads(row[0])
            if protocol.fingerprint(value)!=row[1] or not value['attempts']:
                raise ValueError('Corrupt actual S attempt')
            process=value['attempts'][-1].get('process')
            if process is None or {k:process[k] for k in ('pid','created')}!=issuer:
                raise ValueError('S actor is not the actual reserved attempt process')
        else:raise ValueError('Only original service or autonomous S actors')
        if not state['launches']:raise ValueError('No original campaign launch')
        return multi._scope(reference,issuer)

    def state_clock(self,state):
        result=dict(self.clock)
        if state['sessions']:
            session=state['sessions'][-1]
            end=min(self.original_end,datetime.fromisoformat(session['reserved_at']).timestamp()+session['wall_seconds'])
            result['wall_deadline']=datetime.fromtimestamp(end,UTC).isoformat()
            result['monotonic_deadline']=self.origin_mono+end-self.origin_wall
        active=state['active_step']
        if active is not None:
            if datetime.fromisoformat(active['deadline'])<datetime.fromisoformat(result['wall_deadline']):
                result['wall_deadline']=active['deadline']
            result['monotonic_deadline']=min(result['monotonic_deadline'],active['monotonic_deadline'])
        return result

    def committed_rollover(self, expired_clock):
        """Resolve only a pre-deadline committed successor, never grace time.

        The main observer may be between iterations when the old slot ends.
        Read the real SQL chain using the last verified immutable cursor; a new
        slot must have been paid BEFORE the expired boundary to replace it.
        """
        if (self.directory/'campaign.json').read_bytes()!=self.manifest or (self.directory/'STOP').exists():
            raise ValueError('Campaign original source or stop changed')
        reader=campaign.CampaignStore.__new__(campaign.CampaignStore)
        reader._verified_head=self.store._verified_head;reader.last_verification_rows=0
        reader.db=sqlite3.connect((self.directory/'campaign.sqlite').as_uri()+'?mode=ro',uri=True)
        try:state=reader.load()
        finally:reader.close()
        if state['terminal'] or not state['sessions']:raise TimeoutError('No prospective paid rollover')
        session=state['sessions'][-1]
        began=datetime.fromisoformat(session['reserved_at']).timestamp()
        old=datetime.fromisoformat(expired_clock['wall_deadline']).timestamp()
        end=min(self.original_end,began+session['wall_seconds'])
        if not began<=old<end:raise TimeoutError('No pre-boundary committed successor session')
        result=self.state_clock(state)
        multi._clock(result,parent_required=False)
        active=state['active_step']
        if active is not None and campaign.active_step_expired(state,active['launch_token'],{}):
            raise TimeoutError('Original active step cannot borrow next session')
        return result

    def close(self):self.store.close()


class TreeIssuerScope(multi.IssuerScope):
    """Register this actual PID once; original scope is reread by the root guard."""
    def __init__(self,reference):
        tree=Path(os.environ['V7_TREE_GUARD_DIRECTORY'])
        marker=base._read(tree/'controller.json')
        issuer=base._identity(psutil.Process())
        if issuer==marker['controller']:
            raise ValueError('Controller is not a service/research numeric issuer')
        ancestry=base._ancestry(psutil.Process(),marker['controller'])
        if not ancestry:raise ValueError('Actor is not a descendant of actual controller')
        key=protocol.fingerprint(issuer)
        directory=tree/'actors'/key
        request={'issuer':issuer,'reference':reference,'controller':marker['controller'],
                 'sources':_sources()}
        if directory.exists():
            original=base._read(directory/'registration.json')
            if original['issuer']!=issuer or original['controller']!=marker['controller'] or original['sources']!=_sources():
                raise ValueError('Actor registration identity changed')
        else:multi._write(directory/'registration.json',request)
        while not (directory/'bootstrap.json').exists():
            multi._clock(multi._scope(reference,issuer)['clock'])
            owner=psutil.Process(marker['supervisor']['pid'])
            if base._identity(owner)!=marker['supervisor']:os._exit(125)
            time.sleep(.01)
        os.environ['V7_MULTI_GUARD_DIRECTORY']=str(directory)
        super().__init__(reference)


def _bind_once(observer,root,request):
    """Original campaign journal is the no-reset anchor, not a new directory.

    This startup reservation creates no numerical budget. Partial publication or
    guard loss consumes this launch capability and requires an explicit audited
    successor; changing guard_directory cannot silently reset it.
    """
    value={'guard_directory':str(root),'request_hash':protocol.fingerprint(request),
           'issuer':base._identity(psutil.Process()),'boot_id':multi._boot_id()}
    anchor=observer.directory/'whole_tree_guard.json'
    with FileLock(str(observer.directory/'.whole_tree_guard.lock'),timeout=0):
        store=campaign.CampaignStore(observer.directory,observer.plan)
        try:
            if store.load().get('whole_tree_guard') is not None or anchor.exists():
                raise ValueError('Original whole-tree guard attempt already consumed; no reset')
            store.update('whole_tree_guard_reserved_before_launch',lambda state:state.update(whole_tree_guard=value))
            multi._write(anchor,value)
        finally:store.close()
    return value


def _bind_boundary(observer, root, boundary, request):
    value = {'version': 'whole-tree-kernel-boundary-v1',
             'request_hash': protocol.fingerprint(request),
             'guardian': base._identity(psutil.Process()), 'boot_id': multi._boot_id(),
             'platform': os.name}
    if os.name != 'nt':
        value.update(cgroup=str(boundary.path), relative=boundary.relative,
                     inode=boundary.inode, unit=boundary.path.name)
    multi._write(root/'boundary.json', value)
    store = campaign.CampaignStore(observer.directory, observer.plan)
    try:
        def bind(state):
            if state.get('whole_tree_boundary') is not None:
                raise ValueError('Original boundary cannot be replaced')
            state['whole_tree_boundary'] = value
        store.update('whole_tree_kernel_bound_before_spawn', bind)
    finally:
        store.close()


def _original_dead(identity):
    try:
        actual = psutil.Process(identity['pid'])
        return (actual.create_time() != identity['created']
                or actual.status() in {psutil.STATUS_DEAD, psutil.STATUS_ZOMBIE})
    except psutil.NoSuchProcess:
        return True


def _postmortem_boundary(value):
    """No new grant or signal: inspect original boundary after it has stopped.

    Linux ExecStopPost runs alone in the service or its direct .control child. An
    absent/recreated cgroup or reboot requires separate operator diagnosis;
    this interface never invents a replacement kernel ownership proof.
    """
    if value['platform'] != os.name or value['boot_id'] != multi._boot_id():
        raise ValueError('Original postmortem platform/boot differs')
    if not _original_dead(value['guardian']):
        raise ValueError('Original guardian is still alive')
    if os.name == 'nt':
        return  # Frozen producer used kill-on-close Job; no Windows resource claim.
    leaf = Path(value['cgroup'])
    if (leaf != Path('/sys/fs/cgroup') / value['relative'].lstrip('/')
            or leaf.resolve() != leaf or leaf.name != value['unit']
            or not leaf.name.endswith('.service') or leaf.stat().st_ino != value['inode']):
        raise ValueError('Postmortem requires the original exclusive service leaf')

    def census():
        membership = Path('/proc/self/cgroup').read_text().strip()
        if membership not in {'0::'+value['relative'], '0::'+value['relative']+'/.control'}:
            raise ValueError('Postmortem left the original service or direct control subgroup')
        process_leaf = leaf if membership == '0::'+value['relative'] else leaf/'.control'
        pending = [(leaf, 0)]
        rows = []
        while pending:
            group, depth = pending.pop()
            if depth > 8 or len(rows) >= 64 or group.is_symlink() or group.resolve() != group:
                raise ValueError('Postmortem cgroup hierarchy is unknown or exceeds its read bound')
            inode = group.stat().st_ino
            def bounded_read(name):
                path = group/name
                if path.is_symlink() or path.resolve() != path:
                    raise ValueError('Linked postmortem cgroup evidence')
                with path.open() as stream:
                    text = stream.read(4097)
                if len(text) > 4096:
                    raise ValueError('Postmortem cgroup evidence exceeds its read bound')
                return text
            members = bounded_read('cgroup.procs').split()
            if members != ([str(os.getpid())] if group == process_leaf else []):
                raise ValueError('Postmortem service still contains another process')
            events = [row.split() for row in bounded_read('cgroup.events').splitlines()]
            if (any(len(row) != 2 for row in events)
                    or len({row[0] for row in events}) != len(events)):
                raise ValueError('Unknown postmortem cgroup population evidence')
            populated = dict(events).get('populated')
            if populated != ('1' if group == leaf or group == process_leaf else '0'):
                raise ValueError('Postmortem cgroup population is not exclusively its own process')
            children = []
            with os.scandir(group) as entries:
                for index, entry in enumerate(entries):
                    if index >= 256 or entry.is_symlink():
                        raise ValueError('Postmortem cgroup entries are unknown or exceed their read bound')
                    if entry.is_dir(follow_symlinks=False):
                        children.append(entry.name)
                        if len(rows)+len(pending)+len(children) >= 64:
                            raise ValueError('Postmortem cgroup hierarchy exceeds its read bound')
            if group.stat().st_ino != inode or group.resolve() != group:
                raise ValueError('Original postmortem cgroup changed during inspection')
            rows.append((str(group), inode, tuple(sorted(children))))
            pending.extend((group/name, depth+1) for name in children)
        if (leaf.stat().st_ino != value['inode'] or leaf.resolve() != leaf
                or str(process_leaf) not in {row[0] for row in rows}
                or Path('/proc/self/cgroup').read_text().strip() != membership):
            raise ValueError('Original postmortem service or process membership changed')
        return membership, sorted(rows)

    before = census()
    observed = subprocess.check_output(['systemctl', '--user', 'show', value['unit'],
        '--property=MainPID,KillMode,Restart,ControlGroup'], text=True, timeout=2)
    props = dict(row.split('=', 1) for row in observed.splitlines() if '=' in row)
    if props != {'MainPID': '0', 'KillMode': 'control-group', 'Restart': 'no',
                 'ControlGroup': value['relative']}:
        raise ValueError('Original unit has not completed its main-process stop')
    if census() != before or value['boot_id'] != multi._boot_id():
        raise ValueError('Original postmortem cgroup changed during unit inspection')


def postmortem(directory, guard_directory):
    """Terminalize a stopped original attempt without returning any allowance.

    Source/campaign/boundary/issuer evidence is re-read each invocation. The
    frozen intent permits finishing partial accounting after I/O failure;
    canonical wallet recover is idempotent. Historical scheduler attempts stay
    intact; their original running flag is diagnostic, never new permission.
    """
    directory = protocol.project_path(directory)
    root = protocol.project_path(guard_directory)
    with FileLock(str(directory/'.whole_tree_postmortem.lock'), timeout=0):
        state = campaign.inspect_campaign(directory)
        request = base._read(root/'request.json')
        original = base._read(directory/'whole_tree_guard.json')
        boundary = base._read(root/'boundary.json')
        if (state.get('whole_tree_guard') != original
                or original['guard_directory'] != str(root)
                or original['request_hash'] != protocol.fingerprint(request)
                or request['campaign'] != str(directory)
                or request['source_root'] != str(protocol.ROOT)
                or request['sources'] != _sources()
                or request['script_hash'] != base._hash(request['controller_script'])
                or request['campaign_manifest_hash'] != protocol.fingerprint(
                    json.loads((directory/'campaign.json').read_text()))
                or boundary != state.get('whole_tree_boundary')
                or boundary['request_hash'] != original['request_hash']
                or boundary['guardian'] != original['issuer']
                or boundary['boot_id'] != original['boot_id']):
            raise ValueError('Original guard/campaign/boundary identity changed')
        _postmortem_boundary(boundary)
        identities = [original['issuer']]
        marker = root/'controller.json'
        if marker.exists():
            controller = base._read(marker)
            if controller['supervisor'] != original['issuer']:
                raise ValueError('Original controller owner changed')
            identities.append(controller['controller'])
        else:
            controller = None
        for path in (root/'actors').glob('*'):
            row = base._read(path/'registration.json')
            if (controller is None or row['controller'] != controller['controller']
                    or row['sources'] != request['sources']
                    or path.name != protocol.fingerprint(row['issuer'])):
                raise ValueError('Original actor evidence changed')
            identities.append(row['issuer'])
        identities.extend({k: item['process'][k] for k in ('pid', 'created')}
                          for item in state['launches'] if item.get('process'))
        if any(not _original_dead(identity) for identity in identities):
            raise ValueError('Original owned process still alive; no accounting recovery')
        plan = campaign.read_plan(directory)
        authority, _ = campaign.actual_work_contract(plan)
        issuers = {(item['pid'], item['created']) for item in identities}
        with authority._transaction() as db:
            records = authority._verify(db)['reservations']
        owned = {rid: row for rid, row in records.items()
                 if (row['lease']['parent_pid'], row['lease']['parent_created']) in issuers}
        preserved = {rid: {'lease': row['lease'], 'counts': row['counts'],
                          'binding': row['binding']} for rid, row in owned.items()}
        intent_path = root/'postmortem_intent.json'
        fixed = {'request_hash': original['request_hash'],
                 'boundary_hash': protocol.fingerprint(boundary),
                 'authority_manifest_hash': authority.manifest_hash,
                 'preserved_work_hash': protocol.fingerprint(preserved),
                 'owned_work_count': len(preserved),
                 'campaign_charges_hash': protocol.fingerprint(
                     {k: state[k] for k in ('sessions', 'launches', 'fit_attempts')})}
        if intent_path.exists():
            intent = base._read(intent_path)
            if any(intent[k] != v for k, v in fixed.items()):
                raise ValueError('Postmortem original charged evidence changed')
        else:
            intent = {**fixed, 'original_active_step': state['active_step']}
            multi._write(intent_path, intent)
        for rid, row in owned.items():
            if row['status'] in {'reserved', 'running'}:
                after = authority.recover(rid)
                if after['status'] != 'interrupted_consumed':
                    raise ValueError('Original work did not close as consumed')
        intent_hash = protocol.fingerprint(intent)
        store = campaign.CampaignStore(directory, plan)
        try:
            def finish(current):
                prior = current.get('whole_tree_postmortem')
                if prior is not None:
                    if prior != {'intent_hash': intent_hash}:
                        raise ValueError('Postmortem identity changed')
                    return
                if current['active_step'] != intent['original_active_step']:
                    raise ValueError('Original active step changed during recovery')
                current['interrupted_active_step'] = current['active_step']
                current['active_step'] = None
                if current['terminal'] is None:
                    current['terminal'] = {'reason': 'whole_tree_guard_consumed',
                                           'at': datetime.now(UTC).isoformat()}
                current['whole_tree_postmortem'] = {'intent_hash': intent_hash}
            if store.load().get('whole_tree_postmortem') is None:
                store.update('whole_tree_postmortem_consumed_no_refund', finish)
            else:
                finish(store.load())
        finally:
            store.close()
        with authority._transaction() as db:
            final_work = authority._verify(db)['reservations']
        result = {'version': 'whole-tree-postmortem-v1', 'intent_hash': intent_hash,
                  'terminal': campaign.inspect_campaign(directory)['terminal'],
                  'work_status_hash': protocol.fingerprint(
                      {rid: final_work[rid]['status'] for rid in owned}),
                  'owned_work_count': len(owned),
                  'new_work_permission': False, 'refunds': False,
                  'scheduler_attempts_preserved': True}
        output = root/'postmortem.json'
        if output.exists():
            if base._read(output) != result:
                raise ValueError('Original postmortem result changed')
        else:
            multi._write(output, result)
        return result


class _Ownership:
    def __init__(self,boundary):self.boundary=boundary;self.lock=threading.Lock();self.stopped=False
    def members(self):
        with self.lock:return [] if self.stopped else self.boundary.members()
    def stop(self):
        with self.lock:
            if not self.stopped:
                self.boundary.stop()
                self.stopped=True


def supervise_campaign(directory,*,guard_directory,controller_script,arguments=()):
    """Fixed trusted controller entry only; script is source-frozen, not a plugin.

    STAGE caller supplies the fixed trusted entry to permit engineering fixtures.
    Production adapter must hardcode its path; this is not an arbitrary-code API.
    """
    root=protocol.project_path(guard_directory);script=Path(controller_script).resolve()
    if root.exists():raise ValueError('Guard attempt is consumed and cannot reset')
    root.mkdir(parents=True)
    observer=CampaignObservation(directory)
    request={'campaign':str(observer.directory),'source_root':str(protocol.ROOT),
        'controller_script':str(script),'arguments':list(arguments),
        'script_hash':base._hash(script),'sources':_sources(),
        'campaign_manifest_hash':protocol.fingerprint(json.loads(observer.manifest))}
    multi._write(root/'request.json',request)
    try:
        original_guard=_bind_once(observer,root,request)
        boundary=base._WindowsJob() if os.name=='nt' else base._LinuxService()
        _bind_boundary(observer,root,boundary,request)
    except BaseException:
        observer.close()
        raise
    ownership=_Ownership(boundary);threads={};results={};seen={};pending_paths={};controller=None
    stop=threading.Event();failures=[];previous_clock=None;deadline=None;lock=threading.Lock()
    def watch():
        nonlocal previous_clock,deadline
        try:
            while not stop.wait(.01):
                with lock:clock=deadline
                if clock is not None:
                    try:previous_clock=multi._clock(clock,previous_clock,parent_required=False)
                    except TimeoutError:
                        replacement=observer.committed_rollover(clock)
                        with lock:deadline=replacement
                        previous_clock=multi._clock(replacement,previous_clock,parent_required=False)
        except BaseException as exc:  # noqa: BLE001 -- kill before journal/logging
            failures.append(type(exc).__name__+':'+str(exc));ownership.stop()
    watcher=threading.Thread(target=watch,daemon=True)
    def actor_main(path,identity):
        try:
            results[path.name]=multi.supervise(script,[],directory=path,source_root=protocol.ROOT,
                _ownership=ownership,_actor=identity)
        except BaseException as exc:  # noqa: BLE001 -- shared failure cannot leave siblings running
            failures.append(type(exc).__name__+':'+str(exc));ownership.stop()
    status='failed';reason='startup';initial=time.monotonic()
    try:
        with lock:deadline=observer.clock
        watcher.start()
        with (root/'controller.log').open('xb') as log:
            env=dict(os.environ,V7_TREE_GUARD_DIRECTORY=str(root))
            controller=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'controller',str(root)],
                cwd=base.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)|(4 if os.name=='nt' else 0))
            boundary.attach(controller)
            while not (root/'controller_hello.json').exists():
                if controller.poll() is not None or time.monotonic()-initial>60:raise RuntimeError('Controller bootstrap failed')
                time.sleep(.01)
            identity=base._read(root/'controller_hello.json');actual=psutil.Process(identity['pid'])
            if identity!=base._identity(actual) or actual.pid not in boundary.members():raise ValueError('Unowned controller')
            base._ancestry(actual,base._identity(psutil.Process()))
            multi._write(root/'controller.json',{'controller':identity,'supervisor':base._identity(psutil.Process())})
            while controller.poll() is None and not failures:
                state=observer.read()
                if (state.get('whole_tree_guard')!=original_guard
                        or base._read(observer.directory/'whole_tree_guard.json')!=original_guard):
                    raise ValueError('Original whole-tree guard journal/anchor changed')
                with lock:deadline=observer.state_clock(state)
                if os.name!='nt':boundary.check()
                if any(r['status']!='completed' for r in results.values()):raise RuntimeError('An owned actor failed')
                paths={p.name:p for p in (root/'actors').glob('*')}
                if not set(seen)<=set(paths):raise ValueError('Registered actor identity disappeared')
                for name,path in paths.items():
                    registration=path/'registration.json'
                    if name in seen:
                        if base._read(registration)!=seen[name]:raise ValueError('Original actor registration changed')
                        continue
                    if not registration.exists():
                        pending_paths.setdefault(name,time.monotonic())
                        if time.monotonic()-pending_paths[name]>60:raise TimeoutError('Unpublished actor bootstrap expired')
                        continue
                    value=base._read(registration);actor=value['issuer'];proc=psutil.Process(actor['pid'])
                    if (path.name!=protocol.fingerprint(actor) or base._identity(proc)!=actor
                            or proc.pid not in boundary.members() or value['controller']!=identity
                            or value['sources']!=_sources() or request['sources']!=_sources()):raise ValueError('Actor ownership/source changed')
                    base._ancestry(proc,identity)
                    observer.verify_actor(value['reference'],actor)
                    seen[path.name]=value
                    if len(seen)>8192:raise ValueError('Bounded actor ledger exhausted')
                    thread=threading.Thread(target=actor_main,args=(path,actor),daemon=True)
                    threads[path.name]=thread;thread.start()
                time.sleep(.05)
            if failures:raise RuntimeError(failures[0])
            if controller.returncode or any(t.is_alive() for t in threads.values()) or any(r['status']!='completed' for r in results.values()):
                raise RuntimeError('Controller exited with unfinished actors')
            status,reason='completed','original_controller_and_actors_exited'
    except BaseException as exc:reason=type(exc).__name__+':'+str(exc)  # noqa: BLE001
    finally:
        # Close whole owned tree before persistence, even if errors prevent receipts.
        stop.set();watcher.join(timeout=1)
        try:
            remaining=set(boundary.members())-({os.getpid()} if os.name!='nt' else set())
            if status!='completed' or remaining:ownership.stop()
        finally:
            # Borrowed actor watchers may be simultaneously reacting to the
            # same expired outer. Keep the kernel handle alive until they exit;
            # idempotent shared stop prevents use-after-close panic.
            for thread in threads.values():thread.join(timeout=2)
            boundary.close();observer.close()
        result={'status':status,'reason':reason,'actors':results,'admission_authority':False,
                'finished_at':datetime.now(UTC).isoformat()}
        multi._write(root/'result.json',result)
    return result


def _controller(root):
    root=Path(root);request=base._read(root/'request.json')
    if request['sources']!=_sources() or base._hash(request['controller_script'])!=request['script_hash']:
        raise ValueError('Frozen controller source changed')
    protocol.ROOT=Path(request['source_root'])
    multi._write(root/'controller_hello.json',base._identity(psutil.Process()))
    while not (root/'controller.json').exists():time.sleep(.01)
    sys.argv=[request['controller_script'],*request['arguments']]
    runpy.run_path(request['controller_script'],run_name='__main__')


if __name__=='__main__':
    if sys.argv[1]=='controller':_controller(sys.argv[2])
    else:raise ValueError('Fixed controller bootstrap only')
