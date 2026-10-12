from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import linux_production_path as driver


def test_cutoffs_preserve_two_quarters_and_future_months():
    assert driver.validate_cutoffs(['2025-04-01Z'.replace('Z', 'T00:00:00+00:00'),
        '2025-07-01T00:00:00+00:00', '2025-08-01T00:00:00+00:00',
        '2025-09-01T00:00:00+00:00'])[-1] == '2025-09-01T00:00:00+00:00'
    for dates in [
        ['2025-04-01', '2025-07-01', '2025-08-01', '2025-09-01'],
        ['2025-04-01T00:00Z', '2025-04-01T00:00Z', '2025-05-01T00:00Z', '2025-06-01T00:00Z'],
        ['2025-04-01T00:00Z', '2025-07-01T00:00Z', '2025-07-01T00:00Z', '2025-09-01T00:00Z'],
    ]:
        with pytest.raises(ValueError):
            driver.validate_cutoffs(dates)


@pytest.mark.parametrize('action', ['status', 'reload_original'])
def test_missing_status_or_reload_never_initializes(tmp_path, action):
    root = tmp_path / 'absent'
    with pytest.raises(ValueError, match='Missing original evidence'):
        getattr(driver, action)(root)
    assert not root.exists()


@pytest.mark.parametrize('count', [0, 97, True])
def test_invalid_batch_rejected_before_any_original_open(tmp_path, count):
    with pytest.raises(ValueError, match='batch bound'):
        driver.execute(tmp_path/'absent', 'advance', max_events=count)
    assert list(tmp_path.iterdir()) == []


def test_original_month_owner_cannot_shift_with_logical_month(tmp_path):
    from factorlab.discovery.monthly_partition_policy import frozen_policy, charge, counts
    directory = str(tmp_path / 'original_S')
    owner = {'bridge_id': driver.digest({'directory': directory, 'replay_month': '2025-04'}),
             'run_directory': directory}
    status = {'owners': {'S': owner}}
    assert not driver.S_owner_wait(status, directory, '2025-04')
    assert driver.S_owner_wait(status, directory, '2025-07')
    partitions, owners = {}, {}
    charge(frozen_policy('standard'), partitions, owners, '2026-10',
           {'lane': 'S', 'owner': owner}, counts(solve=7), 300)
    changed = {'bridge_id': driver.digest({'directory': directory, 'replay_month': '2025-07'}),
               'run_directory': directory}
    with pytest.raises(ValueError, match='Another actual owner'):
        charge(frozen_policy('standard'), partitions, owners, '2026-10',
               {'lane': 'S', 'owner': changed}, counts(solve=7), 300)


def test_two_probe_cost_exceeds_original_tree_lane():
    from factorlab.discovery.monthly_partition_policy import frozen_policy, counts, charge
    from factorlab.discovery.monthly_work_authority import BudgetExhausted
    policy = frozen_policy('standard'); owner = {'bridge_id': 'original', 'run_directory': 'original'}
    partitions, owners = {}, {}
    charge(policy, partitions, owners, '2026-10', {'lane': 'S', 'owner': owner}, counts(solve=7), 300)
    charge(policy, partitions, owners, '2026-10',
        {'lane': 'nonlinear', 'S_lane': 'S', 'owner': owner}, counts(tree_fit=3), 300)
    with pytest.raises(BudgetExhausted, match='nonlinear'):
        charge(policy, partitions, owners, '2026-10',
            {'lane': 'nonlinear', 'S_lane': 'S', 'owner': owner}, counts(tree_fit=3), 300)


def test_harness_calls_real_workers_without_injected_gates():
    import ast
    tree = ast.parse(Path(driver.__file__).read_text('utf-8'))
    banned = {'_kernel', '_fit_neural_forward_cold', '_fit_neural_forward_payload',
              '_fit_neural_research_payload', 'setattr', 'monkeypatch', 'sleep_clock'}
    names = {node.func.id if isinstance(node.func, ast.Name) else node.func.attr
             for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, (ast.Name, ast.Attribute))}
    assert not names & banned
    assert {'ResearchScheduler', 'run_prospective_resource', 'verify_completed_artifact'} <= names


def test_changed_harness_identity_rejected_before_original_stores(tmp_path):
    import json
    from factorlab.discovery.v7_run import source_files
    body = dict(version=driver.VERSION, root=str(tmp_path.resolve()), harness_sha256='0'*64,
                source_files=source_files(), S_cutoffs=['2025-04-01T00:00Z', '2025-07-01T00:00Z',
                    '2025-08-01T00:00Z', '2025-09-01T00:00Z'])
    (tmp_path/'identity.json').write_text(json.dumps({'body': body, 'hash': driver.digest(body)}))
    with pytest.raises(ValueError, match='never migrate'):
        driver.load_contract(tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == ['identity.json']


def test_zero_wait_never_opens_or_creates_originals(tmp_path):
    with pytest.raises(ValueError,match='positive same-issuer'):
        driver.execute(tmp_path/'absent','research',wait_seconds=0)
    assert list(tmp_path.iterdir())==[]


@pytest.fixture
def lifetime(tmp_path):
    """Thin immutable job fixture; real clocks/SQLite/wallet/process teardown.

    The idle child is diagnostic, not an S producer or provider result.
    """
    import json,sqlite3,psutil,os
    if os.name != 'nt':
        pytest.skip('Actual Windows Job containment component; Linux uses the separate dedicated-unit drill')
    from scripts.v7_guard.lease_enforcer import _WindowsJob
    class NativeJobBoundary:
        # Production gets _LinuxService only. This test adapter adds the test
        # issuer to real Windows kernel membership; no fake descendant census.
        def __init__(self):
            self.job = _WindowsJob()
            self.boot='Windows-component-only'; self.path=tmp_path/'native-job'; self.inode=0
        def check(self): assert self.job.handle
        def members(self): return [os.getpid(), *self.job.members()]
        def stop(self): self.job.stop()
        def close(self): self.job.close()
    boundary = NativeJobBoundary()
    from factorlab.discovery.v7_autonomous import ResearchScheduler,SubprocessRunner,source_files
    from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
    from factorlab.discovery.s_outer_context import original_clock
    from factorlab.protocol import fingerprint
    scheduler=object.__new__(ResearchScheduler)
    scheduler._diagnostic_boundary=boundary
    scheduler.directory=tmp_path/'scheduler';folder=scheduler.directory/'jobs'/'2026-01'
    folder.mkdir(parents=True)
    scheduler.db=sqlite3.connect(scheduler.directory/'scheduler.sqlite')
    scheduler.db.execute('CREATE TABLE jobs (month TEXT PRIMARY KEY,value TEXT,hash TEXT)')
    scheduler.authority=MonthlyWorkAuthority(tmp_path/'wallet',run_id='idle-'+fingerprint(str(tmp_path)),task='volatility',universe=['X'])
    charged=scheduler.authority.reserve('idle-boundary-diagnostic',counts={'solve':1},
        binding={'purpose':'engineering_only','serving_eligible':False})
    scheduler.identity={'source_files':source_files(),'work_authority':{
        'directory':str(scheduler.authority.directory),'manifest_hash':scheduler.authority.manifest_hash}}
    outer=original_clock(30)
    job={'version':2,'month':'2026-01','source_files':scheduler.identity['source_files'],
         'outer_clock':outer,'deadline':outer['wall_deadline']}
    path=folder/'job.json';path.write_text(json.dumps({'value':job,'sha256':fingerprint(job)}))
    state={'month':'2026-01','job_path':str(path),'job_hash':fingerprint(job),'status':'planned','attempts':[]}
    scheduler._save(state);scheduler.runner=SubprocessRunner();calls=[]
    def launch(*,complete=False,raise_after=False):
        import subprocess
        token='fixed-test-token';outcome={'job_hash':fingerprint(job),'token':token,'attempt':1,'status':'complete'}
        code=('import time,json,hashlib;from datetime import datetime,timezone;from pathlib import Path;'
              'time.sleep(.1);'+f'v={outcome!r};v["finished_at"]=datetime.now(timezone.utc).isoformat();'
              'h=hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest();'
              f'Path({str(folder/"outcome-1.json")!r}).write_text(json.dumps({{"value":v,"sha256":h}}))') if complete else 'import time;time.sleep(60)'
        command=[sys._base_executable,'-B','-c',code]
        process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)|4)
        boundary.job.attach(process)  # Resume only after native containment.
        actual=psutil.Process(process.pid)
        identity={'pid':process.pid,'created':actual.create_time(),'command':actual.cmdline()}
        scheduler.runner.children[process.pid]=process
        state.update(status='running',attempts=[{'number':1,'token':token,'process':identity}])
        scheduler._save(state);calls.append(process)
        if raise_after:raise RuntimeError('diagnostic after durable spawn')
        return {'status':'stepped'}
    scheduler.step=launch
    yield scheduler,launch,calls,charged
    for process in calls:
        if process.poll() is None:process.kill()
        process.wait(timeout=5)
    scheduler.close()
    boundary.close()


def test_same_issuer_terminal_once_and_original_charge_preserved(lifetime):
    import time
    scheduler,launch,calls,charged=lifetime
    scheduler.step=lambda:launch(complete=True)
    result=driver._research_until_terminal(scheduler,'2026-01',time.monotonic()+10,boundary=scheduler._diagnostic_boundary)
    assert result['job_status']=='complete' and len(calls)==1 and calls[0].poll()==0
    assert scheduler.authority.get(charged['reservation_id'])==charged
    assert len(scheduler._load('2026-01')['attempts'])==1


@pytest.mark.parametrize('exception',[False,True])
def test_wait_or_exception_reaps_real_child_without_refund(lifetime,exception):
    import time
    scheduler,launch,calls,charged=lifetime
    scheduler.step=lambda:launch(raise_after=exception)
    try:
        result=driver._research_until_terminal(scheduler,'2026-01',
            time.monotonic()+(10 if exception else .25),boundary=scheduler._diagnostic_boundary)
    except ValueError as exc:
        # A direct child wait does not substitute for the kernel Job census.
        # Conservative rejection must kill the original container and prove
        # it empty; it must never be treated as successful research completion.
        assert str(exc)=='Original research container still contains descendants'
        if exception:
            assert isinstance(exc.__context__,RuntimeError)
            assert str(exc.__context__)=='diagnostic after durable spawn'
        assert scheduler._diagnostic_boundary.job.members()==[]
    except RuntimeError as exc:
        assert exception and str(exc)=='diagnostic after durable spawn'
    else:
        assert not exception
        assert result['status']=='research_wait_consumed' and result['retry_or_refund'] is False
    state=scheduler._load('2026-01')
    assert state['status']=='stopped' and len(state['attempts'])==1
    assert len(calls)==1 and calls[0].poll() is not None
    assert scheduler._diagnostic_boundary.job.members()==[]
    assert scheduler.authority.get(charged['reservation_id'])==charged


def test_foreign_original_parent_is_not_signalled_or_relaunched(lifetime,monkeypatch):
    import json,subprocess,time,site
    from factorlab.protocol import fingerprint
    scheduler,launch,calls,_=lifetime
    code=('import site;[site.addsitedir(p) for p in '+repr(site.getsitepackages())+'];'
          'import json,time;from factorlab.discovery.s_outer_context import original_clock;'
          'print(json.dumps(original_clock(30)),flush=True);time.sleep(60)')
    process=subprocess.Popen([sys._base_executable,'-B','-c',code],stdout=subprocess.PIPE,text=True,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    try:
        outer=json.loads(process.stdout.readline());state=scheduler._load('2026-01')
        path=Path(state['job_path']);job=json.loads(path.read_text())['value'];job['outer_clock']=outer;job['deadline']=outer['wall_deadline']
        path.write_text(json.dumps({'value':job,'sha256':fingerprint(job)}));state['job_hash']=fingerprint(job);scheduler._save(state)
        monkeypatch.setattr(scheduler,'step',lambda:pytest.fail('foreign issuer must not launch'))
        monkeypatch.setattr(scheduler,'stop_and_reap',lambda *a:pytest.fail('foreign issuer must not be signalled'))
        with pytest.raises(ValueError,match='issuer is different'):
            driver._research_until_terminal(scheduler,'2026-01',time.monotonic()+1,boundary=scheduler._diagnostic_boundary)
        assert process.poll() is None and not calls
        process.kill();process.wait(timeout=5)
        with pytest.raises(ValueError,match='issuer is different'):
            driver._research_until_terminal(scheduler,'2026-01',time.monotonic()+1,boundary=scheduler._diagnostic_boundary)
        assert not calls and scheduler._load('2026-01')['attempts']==[]
    finally:
        if process.poll() is None:process.kill()
        process.wait(timeout=5)


def test_cleanup_failure_cannot_return_terminal_success(lifetime,monkeypatch):
    import time
    scheduler,launch,_,_=lifetime
    scheduler.step=lambda:launch(complete=True)
    def failed(*a):raise RuntimeError('actual teardown unresolved')
    monkeypatch.setattr(scheduler,'stop_and_reap',failed)
    with pytest.raises(RuntimeError,match='teardown unresolved'):
        driver._research_until_terminal(scheduler,'2026-01',time.monotonic()+10,boundary=scheduler._diagnostic_boundary)


def test_exited_parent_live_grandchild_cannot_return_terminal(lifetime, tmp_path):
    """Actual native Job membership survives loss of the original process tree."""
    import subprocess,time,psutil,json
    scheduler,_,calls,charged=lifetime
    pid_file=tmp_path/'grandchild.json'; observed={}
    def orphan():
        code=('import subprocess,sys,time,json;from pathlib import Path;'
              'p=subprocess.Popen([sys.executable,"-B","-c","import time;time.sleep(60)"],creationflags=0x08000000);'
              f'Path({str(pid_file)!r}).write_text(json.dumps({{"pid":p.pid}}));time.sleep(.25)')
        command=[sys._base_executable,'-B','-c',code]
        process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,creationflags=0x08000000|4)
        scheduler._diagnostic_boundary.job.attach(process)
        actual=psutil.Process(process.pid)
        identity={'pid':process.pid,'created':actual.create_time(),'command':actual.cmdline()}
        scheduler.runner.children[process.pid]=process;calls.append(process)
        state=scheduler._load('2026-01')
        state.update(status='running',attempts=[{'number':1,'token':'orphan-diagnostic','process':identity}])
        scheduler._save(state)
        end=time.monotonic()+3
        while not pid_file.exists() and time.monotonic()<end:time.sleep(.01)
        grand=psutil.Process(json.loads(pid_file.read_text())['pid'])
        observed.update(pid=grand.pid,created=grand.create_time())
        process.wait(timeout=3)
        assert grand.is_running() and grand.pid in scheduler._diagnostic_boundary.job.members()
        return {'status':'diagnostic_parent_exited'}
    scheduler.step=orphan
    with pytest.raises(ValueError,match='still contains descendants'):
        driver._research_until_terminal(scheduler,'2026-01',time.monotonic()+10,
            boundary=scheduler._diagnostic_boundary)
    assert scheduler._load('2026-01')['status']=='interrupted_consumed'
    assert scheduler._diagnostic_boundary.job.members()==[]
    if psutil.pid_exists(observed['pid']):
        actual=psutil.Process(observed['pid'])
        if actual.create_time()==observed['created']:
            assert actual.wait(timeout=5) is not None  # Windows may retain a terminated PID handle.
    assert scheduler.authority.get(charged['reservation_id'])==charged
    assert len(calls)==1 and len(scheduler._load('2026-01')['attempts'])==1


def test_scope_checks_again_after_queue_and_on_error(lifetime,monkeypatch,tmp_path):
    from types import SimpleNamespace
    from scripts.v7_guard import lease_enforcer
    scheduler,launch,calls,charged=lifetime
    boundary=scheduler._diagnostic_boundary
    # Production factory is fixed _LinuxService; only this Windows component
    # substitutes its actual kernel Job, never a success flag or fake members.
    monkeypatch.setattr(driver,'sys',SimpleNamespace(platform='linux'))
    monkeypatch.setattr(lease_enforcer,'_LinuxService',lambda:boundary)
    with pytest.raises(ValueError,match='still contains descendants'):
        with driver._research_container(tmp_path):
            launch()  # Simulates an unexpected child left by post-drain queue.
    assert calls[0].wait(timeout=5) is not None and boundary.job.handle is None
    assert scheduler.authority.get(charged['reservation_id'])==charged


def test_already_expired_wait_closes_existing_owned_child_without_new_step(lifetime,monkeypatch):
    import time
    scheduler,launch,calls,charged=lifetime
    launch()
    monkeypatch.setattr(scheduler,'step',lambda:pytest.fail('expired wait cannot dispatch'))
    try:
        result=driver._research_until_terminal(scheduler,'2026-01',time.monotonic()-1,boundary=scheduler._diagnostic_boundary)
    except ValueError as exc:
        # A native membership read may still see a terminating member after
        # direct-PID wait. Conservative container kill/rejection is required,
        # never a false terminal. The exact kernel membership must then be empty.
        assert 'still contains descendants' in str(exc)
        assert scheduler._diagnostic_boundary.job.members()==[]
    else:
        assert result['status']=='research_wait_consumed'
    assert calls[0].wait(timeout=5) is not None
    assert len(calls)==1 and scheduler.authority.get(charged['reservation_id'])==charged


def test_complete_result_still_routes_original_typed_queue(monkeypatch,tmp_path):
    # Dispatch-only seam: registration/kernel/worker completion are stubbed;
    # actual native process ownership is covered by the separate tests above.
    from types import SimpleNamespace
    from contextlib import contextmanager
    from dataclasses import asdict
    import pandas as pd
    from factorlab.discovery import v7_autonomous as auto,quarantined_observation as obs,structural_validation as typed
    runtime=SimpleNamespace(lifecycle=object());wallet=object()
    permit=SimpleNamespace(paths={'research':tmp_path/'original-S'})
    body={'S_cutoffs':['2026-01-01T00:00:00+00:00'],'autonomous':asdict(auto.AutonomousConfig())}
    @contextmanager
    def opened(_):yield runtime
    @contextmanager
    def scope(root):yield object()
    calls=[]
    monkeypatch.setattr(driver,'sys',SimpleNamespace(platform='linux'))
    monkeypatch.setattr(driver,'load_contract',lambda root:body)
    monkeypatch.setattr(driver,'originals',lambda body:(permit,None,wallet))
    monkeypatch.setattr(driver,'runtime_for',opened)
    monkeypatch.setattr(driver,'_research_container',scope)
    monkeypatch.setattr(driver,'scheduler_rows',lambda permit:[{'month':'2026-01','status':'complete'}])
    monkeypatch.setattr(auto,'inherited_limits',lambda:None)
    monkeypatch.setattr(obs,'logical_now',lambda lc:pd.Timestamp('2026-01-01T00:00:05Z'))
    monkeypatch.setattr(auto,'ResearchScheduler',lambda *a,**kw:SimpleNamespace(close=lambda:calls.append('closed')))
    monkeypatch.setattr(driver,'_research_until_terminal',lambda *a,**kw:{'status':'research_terminal','job_status':'complete'})
    def queued(r,w,d,m):
        assert (r,w,d,m)==(runtime,wallet,permit.paths['research'],'2026-01')
        calls.append('typed_queue');return {'status':'original_queue_route'}
    monkeypatch.setattr(typed,'queue_structural_validation',queued)
    result=driver.execute(tmp_path,'research',wait_seconds=10)
    assert result['typed_queue']=={'status':'original_queue_route'} and calls==['typed_queue','closed']
