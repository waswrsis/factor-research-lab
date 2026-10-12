"""Production-import, no-fit components for historical efficiency inputs.

All files and state are artificial and fixture-owned. Blockers are reversible;
there are no import-time audit hooks, module substitutions or staged imports.
"""
import hashlib
import io
import json
from pathlib import Path
import socket
import subprocess
import zipfile

import pandas as pd
import pytest

from factorlab.discovery import efficiency_archive_input as archive
from factorlab.discovery import prompt_efficiency as pe


@pytest.fixture(autouse=True)
def no_expensive_work(monkeypatch):
    from factorlab.discovery import convex_eval, joint_pool, v7_agent, v7_run
    import requests

    def denied(*args, **kwargs):
        raise AssertionError('No fitting, provider, network or child processes in these tests')

    monkeypatch.setattr(convex_eval, 'fit_ridge', denied)
    monkeypatch.setattr(joint_pool, 'evaluate_candidate', denied)
    monkeypatch.setattr(v7_run, '_baseline_readout', denied)
    monkeypatch.setattr(v7_agent.CodexBatchBackend, '__call__', denied)
    monkeypatch.setattr(subprocess, 'Popen', denied)
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)
    monkeypatch.setattr(requests.sessions.Session, 'request', denied)


@pytest.fixture
def family():
    from factorlab.discovery.known_library import FEATURE_NAMES
    return {
        'family_id': 'efficiency-volatility-timescale-ratio-v1',
        'hypothesis': 'Short-to-long realized variance ratios may describe volatility persistence beyond the frozen K0 pool.',
        'mechanism': 'Compare causal past variance at different horizons using the cited literature0 ratio seed.',
        'competing_explanation': 'The ratio duplicates information already represented by K0, or positive development gains are sampling noise.',
        'falsification': 'Use the unchanged full three cold-fold conditional evaluate_candidate pipeline; nonpositive S conditional gain or failure of the original stability rule is descriptive lack of support. No V or C claim.',
        'parent_ids': ['literature0'],
        'template': {
            'base': {'expressions': ['last(safe_div(f("rv_6"),f("rv_24")))'],
                     'feature_names': list(FEATURE_NAMES), 'parameters': {},
                     'syntax_version': 'compact-causal-v1'},
            'slots': [
                {'path': ['outputs', 0, 'arg', 'left', 'name'],
                 'values': ['rv_1', 'rv_6', 'rv_24', 'rv_48']},
                {'path': ['outputs', 0, 'arg', 'right', 'name'],
                 'values': ['rv_24', 'rv_168']}],
        },
    }


@pytest.fixture
def bound_archive(tmp_path, monkeypatch, no_expensive_work):
    from factorlab import protocol
    from factorlab.discovery.evidence import source_paths
    from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
    from factorlab.discovery.neural_forward_ticket import _bind_authority
    from factorlab.discovery.v7_forward import ForwardRuntime

    original_root = protocol.ROOT
    root = tmp_path / 'artificial-archive'
    for src in source_paths(original_root):
        dest = root / src.relative_to(original_root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())
    monkeypatch.setattr(protocol, 'ROOT', root)
    runtime = ForwardRuntime(root / 'runtime', root / 'feed')
    try:
        wallet = MonthlyWorkAuthority(root / 'wallet', run_id='artificial-archive-no-fit',
            task='volatility', universe=['BTCUSDT'], partition_policy='standard')
        _bind_authority(runtime, wallet)
        originals = []
        for month in range(1, 10):
            start = pd.Timestamp(f'2026-{month:02d}-01', tz='UTC')
            stamps = pd.date_range(start, start + pd.DateOffset(months=1), freq='h', inclusive='left')
            lines = ['open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore']
            lines += [f'{t.value//10**6},100,101,99,100,1,{(t+pd.Timedelta(hours=1)).value//10**6-1},100,1,0.5,50,0' for t in stamps]
            filename = f'BTCUSDT-1h-2026-{month:02d}.zip'
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as container:
                container.writestr(filename.replace('.zip', '.csv'), '\n'.join(lines))
            payload = buffer.getvalue()
            digest = hashlib.sha256(payload).hexdigest()
            path = root / 'raw-fixture' / f'{filename[:-4]}.{digest}.zip'
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(payload)
            path.with_suffix('.zip.CHECKSUM').write_text(f'{digest}  {filename}\n')
            path.with_suffix('.zip.json').write_text(json.dumps({
                'symbol': 'BTCUSDT', 'month': f'2026-{month:02d}', 'kind': 'klines',
                'sha256': digest, 'bytes': len(payload), 'source_filename': filename,
                'source_url': f'https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1h/{filename}',
                'ingested_at': pd.Timestamp.now(tz='UTC').isoformat(),
                'actual_feed_imported_at': None, 'artificial_test_fixture': True}))
            originals.append(path)
        cutoffs = ['2026-09-01T00:00:00+00:00', '2026-10-01T00:00:00+00:00']
        reference = archive.bind(runtime, wallet, originals, cutoffs)
        source = pe._source(wallet, runtime.directory)
        yield root, runtime, wallet, reference, source, cutoffs
    finally:
        runtime.close()


def test_real_publication_is_not_historical_availability(bound_archive):
    _, runtime, wallet, reference, source, cutoffs = bound_archive
    from factorlab.discovery.v7_feed import SnapshotUnavailable
    value = archive.verify(reference, source, wallet)
    assert all(pd.Timestamp(x['actually_published_at']) > pd.Timestamp(cutoffs[-1])
               for x in value['archives'])
    with pytest.raises(SnapshotUnavailable):
        runtime.feed.snapshot(cutoffs[0])
    identity = {'engineering_archive': reference, 'source': source,
                'authority_directory': str(wallet.directory)}
    for cutoff in cutoffs:
        frame = pe._snapshot(identity, cutoff)
        end = pd.Timestamp(cutoff)
        start = end - pd.DateOffset(months=6) - pd.Timedelta(hours=1234)
        expected = pd.date_range(start, end - pd.Timedelta(hours=1), freq='h', inclusive='left')
        assert pd.DatetimeIndex(frame.open_time).equals(expected)
        assert (frame.available_at == frame.open_time + pd.Timedelta(hours=1, seconds=5)).all()
        assert frame.available_at.max() == end - pd.Timedelta(hours=1) + pd.Timedelta(seconds=5)
        assert frame.attrs['availability_mode'] == 'assumed_historical_delay'
        assert frame.attrs['formal_eligible'] is False
        assert frame.attrs['historical_received_at'] is None and 'feed_identity' not in frame.attrs
    assert wallet.status(pd.Timestamp.now(tz='UTC').strftime('%Y-%m'))['counts']['solve'] == 0


def test_unregistered_cutoff_rejected(bound_archive):
    _, _, wallet, reference, source, _ = bound_archive
    with pytest.raises(ValueError, match='Unregistered'):
        archive.snapshot(reference, source, wallet, '2026-08-01T00:00:00+00:00')


def test_one_shot_binding_and_reference_hash(bound_archive):
    _, runtime, wallet, reference, source, cutoffs = bound_archive
    paths = [x['path'] for x in archive.verify(reference, source, wallet)['archives']]
    with pytest.raises(ValueError, match='one-shot'):
        archive.bind(runtime, wallet, paths, cutoffs)
    with pytest.raises(ValueError, match='binding changed'):
        archive.verify(dict(reference, sha256='0'*64), source, wallet)


@pytest.mark.parametrize('kind', ['zip', 'checksum', 'metadata', 'publication', 'import'])
def test_changed_original_input_rejected(bound_archive, kind):
    _, _, wallet, reference, source, _ = bound_archive
    original = archive.verify(reference, source, wallet)['archives'][0]
    paths = {'zip': Path(original['path']),
             'checksum': Path(original['path']).with_suffix('.zip.CHECKSUM'),
             'metadata': Path(original['path']).with_suffix('.zip.json'),
             'publication': Path(source['feed_directory'])/'publications'/f"{original['batch']}.json",
             'import': Path(original['import_file'])}
    target = paths[kind]
    target.write_bytes(target.read_bytes() + b'\nchanged')
    with pytest.raises((ValueError, json.JSONDecodeError)):
        archive.verify(reference, source, wallet)


def test_quarantine_prefix_rejected(bound_archive):
    _, _, wallet, reference, source, _ = bound_archive
    with pytest.raises(ValueError, match='quarantine'):
        archive.verify(reference, dict(source, observation_prefix={'fake': True}), wallet)


def test_actual_database_publication_required(bound_archive):
    _, runtime, wallet, reference, source, _ = bound_archive
    record = archive.verify(reference, source, wallet)['archives'][0]
    runtime.feed.db.execute('UPDATE publications SET published_at=? WHERE batch=?',
                           ('2026-01-01T00:00:00+00:00', record['batch']))
    runtime.feed.db.commit()
    with pytest.raises(ValueError, match='actual database'):
        archive.verify(reference, source, wallet)


def test_freeze_records_exposure_and_retains_all_unattempted_denominators(bound_archive, family):
    root, runtime, wallet, reference, _, cutoffs = bound_archive
    obj = pe.PromptEfficiencyExperiment.freeze(root/'pilot', authority=wallet,
        runtime_directory=runtime.directory, cutoffs=cutoffs, family_contract=family,
        engineering_archive=reference['path'])
    identity, authority = pe._identity(obj.identity['directory'], wallet)
    assert authority is wallet and identity['engineering_archive'] == reference
    assert identity['backend'] == {'kind': 'codex'}
    assert wallet.status(identity['actual_month'])['counts']['solve'] == 0
    assert runtime.lifecycle.exposure.db.execute("SELECT COUNT(*) FROM evidence WHERE role='S'").fetchone()[0] == 2
    report = pe.inspect(obj.identity['directory'])
    assert report['complete_F_denominator'] == 24 and report['complete_call_denominator'] == 6
    assert report['paid_dispatch_attempts'] == 0 and report['actual_V_count'] == 0
    assert report['input_kind'] == 'historical_archive_engineering_only'
    assert report['formal_claims_enabled'] is False and report['serving_eligible'] is False
    for mode in report['modes'].values():
        assert mode['complete_F_outcomes'] == {'not_attempted': 8}
        assert mode['unique_positive_S_increment_formulas'] == 0
        assert mode['positive_S_formulas_per_1000_reported_input_tokens'] is None
        assert mode['qualified_V_count'] is None


def test_unchanged_budget_and_canonical_family(family):
    from factorlab.discovery.local_mutation import ProgramTemplate
    value = pe.protocol()
    assert value['reserved_counts']['solve'] == 923
    assert value['reserved_seconds'] == 3360 and value['outer_seconds'] == 7200
    assert len(pe.schedule()) == 28
    assert pe.schedule()[0]['seconds'] == 300
    assert {row['seconds'] for row in pe.schedule() if '-F-' in row['key']} == {90}
    assert [value['reserved_counts'][k] for k in ('gpt_attempt', 'f', 'ast')] == [6, 24, 96]
    template = ProgramTemplate.from_dict(family['template'])
    assert template.size == 8
    assert len({template.instantiate((a, b)).digest for a in range(4) for b in range(2)}) == 8


def scored(index, gain, status='completed', formula=None, stable=False):
    return {'work': {'key': f'Q0-F-{index}'}, 'result': {'status': status,
        'output': {'status': 'scored', 'S_gain': gain, 'formula_id': formula or str(index),
                   'family_semantic_hash': 'one-family', 'descriptive_F_eligible': stable}}}


@pytest.mark.parametrize('gain', [False, True, float('nan'), float('inf'), float('-inf'), 0, -.1, '.1'])
def test_nonpositive_nonfinite_or_untyped_gain_not_positive(gain):
    value = pe._quality_metrics([scored(i, gain) for i in range(8)], 'Q0', 1000)
    assert value['unique_positive_S_increment_formulas'] == 0
    assert value['unique_positive_S_increment_families'] == 0
    assert sum(value['complete_F_outcomes'].values()) == 8


@pytest.mark.parametrize('tokens', [None, 0, 1000])
def test_failed_slots_dedup_stability_and_unknown_tokens(tokens):
    states = ['failed_consumed', 'not_attempted', 'orphan_consumed',
              'unwitnessed_consumed', 'late_or_invalid_consumed']
    rows = [scored(i, 10, status) for i, status in enumerate(states)]
    rows += [scored(5, .1, formula='same'), scored(6, .2, formula='same', stable=True),
             scored(7, -.1, formula='negative', stable=True)]
    value = pe._quality_metrics(rows, 'Q0', tokens)
    assert value['unique_positive_S_increment_formulas'] == 1
    assert value['unique_positive_S_increment_families'] == 1
    assert value['unique_positive_S_and_original_stability_formulas'] == 1
    expected = 1 if tokens == 1000 else None
    assert value['positive_S_formulas_per_1000_reported_input_tokens'] == expected
    assert value['positive_S_families_per_1000_reported_input_tokens'] == expected
    assert sum(value['complete_F_outcomes'].values()) == value['complete_F_denominator'] == 8
