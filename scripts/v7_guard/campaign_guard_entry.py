"""STAGE fixed Campaign/controller/research entry under whole-tree supervision.

Requires the separately reviewed G01 scheduler/authority core integration.
No arbitrary backend/module/callback is accepted here. This is not installed,
not a service unit, and has no deployment/GPT/network acceptance evidence.
"""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path

import psutil

sys.path.insert(0,str(Path(__file__).resolve().parent))
import lease_enforcer as base
import multiwork_enforcer as multi
import tree_work_enforcer as tree
from guarded_runtime import _scheduled_publication, forward_step, nonlinear_publication

from factorlab import protocol
from factorlab.discovery import neural_forward_ticket, v7_autonomous
from factorlab.discovery.process_policy import isolated_process_kwargs, restricted_environment
from scripts import v7_long_running as campaign


def _tree():
    directory=protocol.project_path(os.environ['V7_TREE_GUARD_DIRECTORY'])
    marker=base._read(directory/'controller.json');request=base._read(directory/'request.json')
    if request['sources']!=tree._sources():raise ValueError('Frozen dedicated entry source changed')
    owner=psutil.Process(marker['supervisor']['pid'])
    if base._identity(owner)!=marker['supervisor']:raise ValueError('Original whole-tree guardian lost')
    current=psutil.Process()
    if base._identity(current)!=marker['controller']:base._ancestry(current,marker['controller'])
    return directory


def _command(*args):
    executable = sys._base_executable if os.name == 'nt' else sys.executable
    return [executable, '-B', str(Path(__file__).resolve()), *map(str, args)]


def _spawn_files(directory, token):
    if not token or any(c not in '0123456789abcdef' for c in token):
        raise ValueError('Actual reserved random launch token required')
    return {key: Path(directory)/f'spawn-{token}-{key}.json'
            for key in ('intent', 'popen', 'native', 'ack')}


def _validate_spawn(intent, owner, native):
    if (intent['sources'] != tree._sources() or native['nonce'] != intent['nonce']
            or native['intent_hash'] != protocol.fingerprint(intent)):
        raise ValueError('Fixed spawn source/nonce changed')
    parent = psutil.Process(intent['parent']['pid'])
    process = psutil.Process(native['identity']['pid'])
    launcher = psutil.Process(owner['identity']['pid'])
    if (base._identity(parent) != intent['parent'] or base._identity(process) != native['identity']
            or base._identity(launcher) != owner['identity']
            or process.cmdline() != intent['command'] or native['command'] != intent['command']
            or launcher.cmdline() != owner['command']):
        raise ValueError('Actual original parent/launcher/native identity changed')
    if os.name == 'nt':
        if (Path(process.exe()).resolve() != Path(intent['native_interpreter']).resolve()
                or Path(launcher.exe()).resolve() != Path(intent['venv_interpreter']).resolve()
                or process.ppid() != launcher.pid or launcher.ppid() != parent.pid
                or owner['command'] != [intent['venv_interpreter'], *intent['command'][1:]]):
            raise ValueError('Not the exact venv redirector/native interpreter pair')
    elif (process.pid != launcher.pid or process.ppid() != parent.pid
            or owner['command'] != intent['command']):
        raise ValueError('Linux issuer must be the original Popen process')


def _spawn_hello(directory, token):
    _tree()
    v7_autonomous.inherited_limits()
    files = _spawn_files(directory, token)
    began = time.monotonic()
    while not files['popen'].exists():
        if time.monotonic()-began > 10:raise TimeoutError('Original launch handshake missing')
        time.sleep(.005)
    intent, owner = base._read(files['intent']), base._read(files['popen'])
    me = psutil.Process()
    native = {'identity': base._identity(me), 'command': me.cmdline(),
              'nonce': intent['nonce'], 'intent_hash': protocol.fingerprint(intent)}
    _validate_spawn(intent, owner, native)
    base._write(files['native'], native)
    while not files['ack'].exists():
        _validate_spawn(intent, owner, native)
        if time.monotonic()-began > 10:raise TimeoutError('Original parent did not acknowledge native issuer')
        time.sleep(.005)
    if base._read(files['ack']) != {'native_hash': protocol.fingerprint(native), 'nonce': intent['nonce']}:
        raise ValueError('Wrong native launch acknowledgment')


def _spawn(command,temporary,log_path,children):
    # The existing service cgroup is mandatory. This does not create another
    # service, broaden limits, or bypass the canonical reserved spawn record.
    v7_autonomous.inherited_limits()
    root=_tree();temporary.mkdir(exist_ok=True)
    files = _spawn_files(temporary.parent, command[-1])
    intent = {'nonce': uuid.uuid4().hex, 'parent': base._identity(psutil.Process()),
              'command': command, 'sources': tree._sources(),
              'venv_interpreter': sys.executable, 'native_interpreter': sys._base_executable}
    base._write(files['intent'], intent)
    environment=restricted_environment()
    environment.update(V7_TREE_GUARD_DIRECTORY=str(root),TMP=str(temporary),TEMP=str(temporary),
        TMPDIR=str(temporary),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with log_path.open('ab') as log:
        launched = [sys.executable, *command[1:]] if os.name == 'nt' else command
        child=subprocess.Popen(launched,cwd=protocol.project_path('.'),env=environment,shell=False,
            stdin=subprocess.DEVNULL,stdout=log,stderr=log,**isolated_process_kwargs())
    owner = {'identity': base._identity(psutil.Process(child.pid)), 'command': launched}
    base._write(files['popen'], owner)
    began = time.monotonic()
    while not files['native'].exists():
        if child.poll() is not None or time.monotonic()-began > 10:
            raise RuntimeError('Original owned native launch did not complete its handshake')
        time.sleep(.005)
    native = base._read(files['native'])
    _validate_spawn(intent, owner, native)
    base._write(files['ack'], {'native_hash': protocol.fingerprint(native), 'nonce': intent['nonce']})
    children[native['identity']['pid']] = child
    return {**native['identity'], 'command': command}


class _CampaignRunner(campaign.CampaignRunner):
    @staticmethod
    def command(directory,launch):
        return _command('worker',directory,launch['token'])
    def start(self,directory,launch):
        path=protocol.project_path(directory)
        return _spawn(self.command(directory,launch),path/'tmp',path/f'worker-{launch["token"]}.log',self.children)


class _ResearchRunner(v7_autonomous.SubprocessRunner):
    @staticmethod
    def command(job_path,attempt):
        return _command('research',job_path,attempt['number'],attempt['token'])
    def start(self,job_path,attempt):
        path=protocol.project_path(job_path)
        return _spawn(self.command(path,attempt),path.parent/'tmp',path.parent/f'worker-{attempt["number"]}.log',self.children)


def _worker(directory,token):
    _tree()
    _spawn_hello(directory,token)
    began=time.monotonic()
    while True:
        state=campaign.inspect_campaign(directory)
        launch=state['launches'][-1]
        if launch['token']!=token:raise ValueError('Original campaign launch changed')
        if launch['process'] is not None:
            if {k:launch['process'][k] for k in ('pid','created')}!=base._identity(psutil.Process()):
                raise ValueError('Actual campaign native issuer differs from reservation')
            break
        if time.monotonic()-began>10:raise TimeoutError('Original native issuer was not durably registered')
        time.sleep(.005)
    # Exact actual core remains the controller: source/host guards, shared
    # counts, step reservation, transport policy and session rollover unchanged.
    original_scheduler=campaign.CampaignScheduler
    original_step=campaign.campaign_step
    class GuardedScheduler(original_scheduler):
        def __init__(self,*args,**kwargs):
            if 'runner' in kwargs:raise ValueError('Fixed research runner cannot be replaced')
            super().__init__(*args,runner=_ResearchRunner(),**kwargs)
    def step(store,autonomous,session_id):
        actual=autonomous.step
        def work():
            active=store.load()['active_step']
            if active is None:raise ValueError('Original Campaign step was not prepaid')
            scope=tree.TreeIssuerScope({'kind':'campaign','directory':str(store.directory),'step_id':active['id']})
            with forward_step(scope,autonomous.service):result=actual()
            scope.close()
            return result
        autonomous.step=work
        try:return original_step(store,autonomous,session_id)
        finally:autonomous.step=actual
    campaign.CampaignScheduler=GuardedScheduler;campaign.campaign_step=step
    try:
        campaign.run_worker(directory,token)
        if os.environ.get('V7_MULTI_GUARD_DIRECTORY'):multi.finish()
    finally:
        campaign.CampaignScheduler=original_scheduler;campaign.campaign_step=original_step


def _research(job_path,attempt,token):
    _tree()
    _spawn_hello(Path(job_path).parent,token)
    from factorlab.discovery.s_outer_context import SOuterContext
    path=protocol.project_path(job_path)
    context=SOuterContext.autonomous(path,attempt,token)
    value=context._verify()
    # Spawn reservation precedes registration of its actual process. Waiting
    # here spends the ORIGINAL job clock, never a new startup lease.
    while True:
        context._verify()
        with sqlite3.connect((path.parents[2]/'scheduler.sqlite').as_uri()+'?mode=ro',uri=True) as db:
            import json
            state=json.loads(db.execute('SELECT value FROM jobs WHERE month=?',(path.parent.name,)).fetchone()[0])
        identity=state['attempts'][-1].get('process')
        if identity is not None:
            if {k:identity[k] for k in ('pid','created')}!=base._identity(psutil.Process()):
                raise ValueError('Actual research attempt names another process')
            break
        time.sleep(.01)
    scope=tree.TreeIssuerScope({'kind':'S','bound':{'reference':context.reference,
        'scope_hash':value['scope_hash'],'clock':value['clock']}})
    original_open=neural_forward_ticket.open_existing_authority
    authority=original_open(value['authority']['directory'])
    def same_wallet(directory):
        if protocol.project_path(directory)!=authority.directory:raise ValueError('Research cannot switch the original wallet')
        authority.verify()
        return authority
    # Only the fixed original worker's same-wallet open is adapted. Numerical
    # source, actual feed, role, ticket/gates and canonical budget checks remain.
    neural_forward_ticket.open_existing_authority=same_wallet
    try:
        with scope.installed(authority),_scheduled_publication(scope,authority),nonlinear_publication(scope,authority):
            v7_autonomous.run_worker(path,attempt,token)
        scope.close();multi.finish()
    finally:neural_forward_ticket.open_existing_authority=original_open


def main():
    mode,*args=sys.argv[1:]
    if mode=='guard':
        if len(args)!=2:raise ValueError('Exact existing campaign and fresh guard directory required')
        result=tree.supervise_campaign(args[0],guard_directory=args[1],controller_script=Path(__file__),
            arguments=['controller',args[0]])
        sys.exit(0 if result['status']=='completed' else 124)
    elif mode=='controller':
        _tree();directory=protocol.project_path(args[0])
        controller=campaign.LongController(directory,campaign.read_plan(directory),runner=_CampaignRunner(),
                                            research_runner=_ResearchRunner())
        try:controller.run()
        finally:controller.close()
    elif mode=='worker':_worker(args[0],args[1])
    elif mode=='research':_research(args[0],int(args[1]),args[2])
    elif mode=='postmortem':
        if len(args)!=2:raise ValueError('Original campaign and guard directory required')
        print(tree.postmortem(args[0],args[1]))
    else:raise ValueError('Fixed trusted production entry required')


if __name__=='__main__':main()
