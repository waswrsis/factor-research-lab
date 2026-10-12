"""Real bounded CPU children and SQL registrations; not Linux guard acceptance."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import psutil
import pytest

REPO = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path[:0] = [str(REPO), str(REPO/'tests')]
from test_v7_run_factor_pool import source_root

from factorlab.discovery.v7_autonomous import SubprocessRunner
from factorlab.protocol import fingerprint


@pytest.fixture
def case(tmp_path, monkeypatch, request):
    source_root(tmp_path, monkeypatch)
    staged = Path(__file__).parent.parent/'guard_stop_delta/v7_long_running.py'
    source = staged if staged.is_file() else REPO/'scripts/v7_long_running.py'
    spec = importlib.util.spec_from_file_location('stop_audit_campaign', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    plan = module.CampaignPlan(end_at=(module.now()+module.pd.Timedelta(hours=1)).isoformat(),
        max_sessions=2, max_reserved_seconds=240, max_reserved_steps=4, max_fit_attempts=2,
        forward_directory=str(tmp_path/'forward'), feed_directory=str(tmp_path/'feed'),
        research={'backend': 'disabled', 'research': {'evaluation_mode': 'joint'}},
        lifecycle={'source_id': 'artificial-stop-only'},
        session_seconds=120, session_steps=2)
    script = tmp_path/'bounded_fixed_research.py'
    script.write_text('import time\ntime.sleep(20)\n')
    class FixedArtificialRunner(SubprocessRunner):
        @staticmethod
        def command(job_path, attempt):
            return [sys.executable, '-B', str(script), 'research', str(job_path),
                    str(attempt['number']), attempt['token']]
    runner = FixedArtificialRunner()
    controller = module.LongController(tmp_path/'campaign', plan,
        research_runner=None if getattr(request, 'param', None) == 'legacy' else runner)
    job = tmp_path/'forward/autonomous/jobs/2026-10/job.json'
    job.parent.mkdir(parents=True)
    attempt = {'number': 0, 'token': 'bounded-artificial-original-token'}
    child = subprocess.Popen(runner.command(job, attempt), stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    attempt['process'] = {'pid': child.pid, 'created': psutil.Process(child.pid).create_time(),
                          'command': runner.command(job, attempt)}
    journal = tmp_path/'forward/autonomous/scheduler.sqlite'
    import sqlite3
    with sqlite3.connect(journal) as db:
        db.execute('CREATE TABLE jobs(month TEXT,value TEXT,hash TEXT)')
        state = {'month': '2026-10', 'job_path': str(job), 'attempts': [attempt]}
        db.execute('INSERT INTO jobs VALUES(?,?,?)', ('2026-10', json.dumps(state), fingerprint(state)))
    try:
        yield module, plan, runner, child, journal, state, controller
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        controller.close()


def test_terminal_uses_the_exact_registered_fixed_runner_and_stops_its_cpu_child(case, tmp_path):
    _, _, _, child, _, _, controller = case
    assert controller._stop('explicit-stop-test')['status'] == 'terminal'
    child.wait(timeout=5)
    assert child.poll() is not None


@pytest.mark.parametrize('case', ['legacy'], indirect=True)
def test_legacy_runner_does_not_guess_or_stop_a_different_registered_command(case, tmp_path):
    _, _, _, child, _, _, controller = case
    with pytest.raises(ValueError, match='command differs'):
        controller._stop_registered_research()
    assert child.poll() is None


def test_recorded_pid_birth_mismatch_never_stops_the_other_process(case, tmp_path):
    import sqlite3
    _, _, _, child, journal, state, controller = case
    state['attempts'][0]['process']['created'] += 10
    with sqlite3.connect(journal) as db:
        db.execute('UPDATE jobs SET value=?,hash=?', (json.dumps(state), fingerprint(state)))
    controller._stop_registered_research()
    assert child.poll() is None


def test_arbitrary_callback_is_not_a_research_stop_capability(case, tmp_path):
    module, plan, _, _, _, _, _ = case
    with pytest.raises(TypeError, match='SubprocessRunner'):
        module.LongController(tmp_path/'other-campaign', plan, research_runner=lambda *a: None)
    assert not (tmp_path/'other-campaign').exists()
