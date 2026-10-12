"""Independent daily deadline audit; fake child only, never public HTTP."""
import pandas as pd
from test_v7_daily import setup as daily_setup

from factorlab.discovery import v7_daily as daily


def test_wall_clock_rollback_cannot_expand_owned_worker_lease(tmp_path, monkeypatch):
    directory, _session, clock = daily_setup.__wrapped__(tmp_path, monkeypatch)
    request = directory / 'request.json'
    daily._record(request, {'reserved_at': (clock[0] + pd.Timedelta(hours=1)).isoformat()})
    launches = []

    class Child:
        returncode = 0

        def wait(self, timeout):
            launches.append({'wait_seconds': timeout})

        def poll(self):
            return 0

    def launch(*a, **kw):
        launches.append('started')
        return Child()

    monkeypatch.setattr(daily.subprocess, 'Popen', launch)
    try:
        outcome = daily._download(request, None)
    except (ValueError, RuntimeError):
        outcome = {'status': 'failed'}
    assert outcome['status'] == 'failed'
    assert not launches, 'A future reservation must fail closed, not get an hour-long worker lease'
