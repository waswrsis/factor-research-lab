"""Real 30s original S clock and real exited child; synthetic no-work outcome.

Both production receiver methods are exercised. Authority enrollment and queue
production are isolated boundaries; no GPT, model fit or market proof is made.
"""
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import psutil
import pytest

from factorlab.protocol import fingerprint
from factorlab.discovery import v7_autonomous as auto
from factorlab.discovery.s_outer_context import original_clock


def scheduler(root, clock):
    path=root/'jobs/2020-01/job.json';path.parent.mkdir(parents=True)
    sources=auto.source_files()
    job={'version':2,'month':'2020-01','source_files':sources,'outer_clock':clock,
         'deadline':clock['wall_deadline']}
    auto._write(path,job)
    obj=auto.ResearchScheduler.__new__(auto.ResearchScheduler)
    obj.directory=root;obj.runner=auto.SubprocessRunner()
    obj.authority=SimpleNamespace(directory=root/'wallet',manifest_hash='original',verify=lambda:None)
    obj.identity={'source_files':sources,'work_authority':{'directory':str(root/'wallet'),'manifest_hash':'original'}}
    obj.runtime=SimpleNamespace(lifecycle=SimpleNamespace(now=lambda:pd.Timestamp.now(tz='UTC')))
    obj.config=SimpleNamespace(not_before_month='9999-01')
    obj.db=sqlite3.connect(root/'scheduler.sqlite')
    obj.db.execute('CREATE TABLE jobs(month TEXT PRIMARY KEY,value TEXT,hash TEXT)')
    obj._queue=lambda *a:pytest.fail('late outcome cannot enter queue')
    obj._publish_queue=obj._queue;obj._queue_algorithm=obj._queue
    obj._save({'month':'2020-01','status':'running','job_path':str(path),'job_hash':fingerprint(job),
        'validation':None,'attempts':[{'number':0,'token':'original-test-token','process':None}]})
    return obj,path,job


def attach_child(obj,path,job,delay_until):
    outcome={'job_hash':fingerprint(job),'token':'original-test-token','attempt':0,'status':'complete',
             'finished_at':job['outer_clock']['issuer']['wall_started_at']}
    code=('import json,time;from pathlib import Path;'
          'from factorlab.discovery.v7_autonomous import _write;'
          f'time.sleep(max(0.,{delay_until!r}-time.monotonic()));'
          f'_write(Path({str(path.parent/"outcome-0.json")!r}),json.loads({json.dumps(outcome)!r}))')
    command=[sys.executable,'-B','-c',code]
    child=subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,
                           creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),
                           cwd=Path(auto.__file__).resolve().parents[2])
    obj.runner.children[child.pid]=child
    state=obj._load('2020-01')
    state['attempts'][0]['process']={'pid':child.pid,'created':psutil.Process(child.pid).create_time(),'command':command}
    obj._save(state)
    return child


def test_both_exited_receivers_reject_real_late_publication(tmp_path):
    clock=original_clock(30)
    setups=[scheduler(tmp_path/name,clock) for name in ('settle','step')]
    children=[attach_child(*s,clock['monotonic_deadline']+.05) for s in setups]
    try:
        for child in children:
            _,err=child.communicate(timeout=40)
            assert child.returncode==0,err
        for (obj,path,job),method in zip(setups,('settle_running','step')):
            assert (path.parent/'outcome-0.json').stat().st_mtime > pd.Timestamp(job['deadline']).timestamp()
            getattr(obj,method)()
            state=obj._load('2020-01')
            assert state['status']=='deadline_consumed'
            assert state['outcome']['status']=='complete'  # original late data retained
            assert state['stop_reason']=='original_outer_clock_lost_at_readback'
            assert len(state['attempts'])==1
    finally:
        for child in children:
            if child.poll() is None:child.kill();child.wait(timeout=5)
        for obj,_,_ in setups:obj.close()


def test_original_issuer_timely_readback_can_settle(tmp_path):
    clock=original_clock(30)
    obj,path,job=scheduler(tmp_path/'timely',clock)
    child=attach_child(obj,path,job,time.monotonic()+.05)
    try:
        _,err=child.communicate(timeout=10)
        assert child.returncode==0,err
        obj.settle_running()
        assert obj._load('2020-01')['status']=='complete'
        assert clock['issuer']['parent_pid']==os.getpid()
    finally:
        if child.poll() is None:child.kill();child.wait(timeout=5)
        obj.close()
