"""Independent ownership and durable-job fault injection; no real child processes."""
import pytest
from test_v7_autonomous import setup as autonomous_fixture

from factorlab.discovery import v7_autonomous as autonomous


@pytest.fixture
def setup(tmp_path, monkeypatch):
    yield from autonomous_fixture.__wrapped__(tmp_path, monkeypatch)


def test_owned_stop_keeps_the_same_verified_psutil_handle(monkeypatch):
    constructed, signalled = [], []
    identity = {'pid': 123, 'created': 100., 'command': ['owned', 'token']}

    class Process:
        def __init__(self, pid):
            self.pid = pid
            self.foreign = bool(constructed)
            constructed.append(self)

        def create_time(self):
            return 200. if self.foreign else 100.

        def cmdline(self):
            return ['unrelated'] if self.foreign else ['owned', 'token']

        def status(self):
            return 'running'

        def children(self, recursive):
            return []

        def kill(self):
            signalled.append(self.foreign)

    monkeypatch.setattr(autonomous.psutil, 'Process', Process)
    autonomous.SubprocessRunner().stop(identity)
    assert len(constructed) == 1
    assert signalled == [False]  # Reconstructing a handle can target the reused PID.


def test_missing_scheduler_checkpoint_cannot_reset_reserved_attempts(setup):
    _, _, scheduler, runner, _ = setup
    scheduler.step()
    assert len(runner.starts) == 1
    scheduler.db.execute('DELETE FROM jobs')
    scheduler.db.commit()
    with pytest.raises(ValueError, match='checkpoint|manifest|reservation|orphan|existing|missing'):
        scheduler.step()
    assert len(runner.starts) == 1
