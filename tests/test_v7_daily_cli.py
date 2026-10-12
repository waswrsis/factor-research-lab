import json

from factorlab.discovery import v7_cli, v7_daily, v7_feed


def test_daily_cli_only_performs_explicit_bounded_preparation(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(v7_daily, 'prepare_daily_bridge',
                        lambda *args, **kwargs: calls.append((args, kwargs)) or {'data_ready': False})
    assert v7_cli.main(['prepare-daily-bridge', '--run', 'work/daily-cli', '--symbol', 'BTCUSDT',
                       '--start', '2026-10-01T00:00Z']) == 0
    assert calls == [(('work/daily-cli',), {'symbols': ('BTCUSDT',), 'start': '2026-10-01T00:00Z',
                                          'end': None, 'as_of': None, 'max_new_archives': 1})]
    assert json.loads(capsys.readouterr().out) == {'data_ready': False}


def test_daily_import_uses_actual_archive_and_always_closes_it(monkeypatch, capsys):
    calls = []
    class Feed:
        def __init__(self, *args):
            calls.append(('open', args))
        def close(self):
            calls.append(('close',))
    monkeypatch.setattr(v7_feed, 'ReceiptArchive', Feed)
    monkeypatch.setattr(v7_daily, 'import_daily_bridge',
                        lambda directory, feed: calls.append(('import', directory, type(feed).__name__))
                        or {'prediction_origins': 0})
    assert v7_cli.main(['import-daily-bridge', '--run', 'work/daily-cli', '--feed', 'work/feed-cli',
                       '--symbol', 'BTCUSDT']) == 0
    assert calls == [('open', ('work/feed-cli', ('BTCUSDT',))),
                     ('import', 'work/daily-cli', 'Feed'), ('close',)]
    assert json.loads(capsys.readouterr().out)['prediction_origins'] == 0
