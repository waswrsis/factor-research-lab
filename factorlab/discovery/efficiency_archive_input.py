"""Explicit historical archive input ONLY for the bounded efficiency S pilot.

Physical feed import/publication timestamps stay real. Historical availability
is an assumption, never an observed receipt, quarantine permit or formal gate.
No feature construction, fitting, provider execution or network in this module.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from factorlab.data import _source, _verify, load_archive
from factorlab.protocol import fingerprint, project_path
from .evidence import file_hash
from .v7_storage import immutable_record

VERSION = 'efficiency-historical-archive-S-only-v1'
NAME = 'efficiency_archive_input.json'
POLICY = {'availability_mode': 'assumed_historical_delay', 'engineering_only': True,
          'V_eligible': False, 'C_eligible': False, 'serving_eligible': False,
          'formal_eligible': False, 'historical_received_at': None}


def _raw(path):
    path = project_path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 4 * 2**20:
        raise ValueError('Missing bounded original archive evidence')
    return json.loads(path.read_text('utf-8'))


def _archive(path):
    path = project_path(path)
    metadata = _verify(path)
    url, filename = _source(metadata['symbol'], metadata['month'], 'klines')
    if (metadata['kind'] != 'klines' or metadata['source_url'] != url
            or metadata['source_filename'] != filename
            or metadata.get('actual_feed_imported_at') is not None):
        raise ValueError('Original public monthly archive provenance required')
    now = pd.Timestamp.now(tz='UTC')
    downloaded = pd.Timestamp(metadata['ingested_at'])
    if downloaded.tzinfo is None or downloaded > now:
        raise ValueError('Archive actual download time cannot be future or naive')
    files = {str(p): file_hash(p) for p in
             (path, path.with_suffix('.zip.json'), path.with_suffix('.zip.CHECKSUM'))}
    return {'path': str(path), 'symbol': metadata['symbol'], 'month': metadata['month'],
            'downloaded_at': downloaded.isoformat(), 'files': files}


def bind(runtime, authority, archives, cutoffs):
    """One explicit ingestion/binding, using original wall time and shared wallet.

    The initializer must first clear reviewed regression/Linux/capacity gates.
    This does not create a wallet, renew a clock, claim a work slot or run GPT.
    """
    from .neural_forward_ticket import _verify_authority_binding, _LifecycleReader
    from .monthly_partition_policy import frozen_policy
    reader = _LifecycleReader(runtime.directory)
    try:
        _verify_authority_binding(reader, authority, reader.verify()['operations'])
    finally:
        reader.close()
    authority.verify()
    if authority.manifest['partition_policy'] != frozen_policy('standard'):
        raise ValueError('Original shared standard wallet required')
    if authority.status(pd.Timestamp.now(tz='UTC').strftime('%Y-%m'))['owners'].get('S') is not None:
        raise ValueError('Actual month already owns S; never create replacement wallet')
    if ((runtime.directory / NAME).exists()
            or (runtime.directory/'efficiency_archive_initialization_intent.json').exists()
            or runtime.feed.db.execute('SELECT COUNT(*) FROM batches').fetchone()[0]):
        raise ValueError('Archive pilot initialization is one-shot; existing input is read-only')
    if len(archives) != 9 or tuple(runtime.config.symbols) != ('BTCUSDT',):
        raise ValueError('This preregistered engineering input requires exactly nine BTC monthly originals')
    records = [_archive(p) for p in archives]
    expected = [('BTCUSDT', f'2026-{m:02d}') for m in range(1, 10)]
    if [(r['symbol'], r['month']) for r in records] != expected:
        raise ValueError('All January through September originals required in canonical order')
    required_cutoffs = ['2026-09-01T00:00:00+00:00', '2026-10-01T00:00:00+00:00']
    if list(cutoffs) != required_cutoffs or any(pd.Timestamp(x) > pd.Timestamp.now(tz='UTC') for x in cutoffs):
        raise ValueError('Original two historical monthly cutoffs required')
    intent = {'version': VERSION, 'archives': records, 'cutoffs': required_cutoffs,
              'policy': POLICY, 'authority_hash': authority.manifest_hash}
    immutable_record(runtime.directory/'efficiency_archive_initialization_intent.json', intent)
    for number, record in enumerate(records):
        frame = load_archive(record['path'], record['symbol'], availability_delay_seconds=runtime.config.archive_delay_seconds)
        start = pd.Timestamp(record['month']+'-01', tz='UTC')
        if not pd.DatetimeIndex(frame.open_time).equals(pd.date_range(start, start+pd.DateOffset(months=1), freq='h', inclusive='left')):
            raise ValueError('Original archive month incomplete; no interpolation or row deletion')
        imported = runtime.directory/'efficiency_archive_inputs'/f'{number:02d}.parquet'
        imported.parent.mkdir(exist_ok=True)
        if imported.exists():
            raise ValueError('Interrupted original ingestion cannot be replayed')
        frame.to_parquet(imported, index=False)
        result = runtime.feed.import_seed(imported)  # Actual wall clock, actual=False.
        batch_path = runtime.feed.directory/'raw'/f"{result['batch']}.json"
        publication_path = runtime.feed.directory/'publications'/f"{result['batch']}.json"
        batch, publication = _raw(batch_path), _raw(publication_path)
        record.update({'import_file': str(imported), 'import_sha256': file_hash(imported),
                       'batch': result['batch'], 'batch_sha256': file_hash(batch_path),
                       'publication_sha256': file_hash(publication_path),
                       'actually_imported_at': batch['prepared_at'],
                       'actually_published_at': publication['published_at']})
    value = {**intent, 'archives': records, 'runtime_directory': str(runtime.directory),
             'feed_directory': str(runtime.feed.directory), 'source_id': runtime.feed.identity,
             'delay_seconds': runtime.config.archive_delay_seconds,
             'source_files': authority.manifest['source_files']}
    for path in (runtime.directory/NAME, authority.directory/'efficiency_archive_inputs'/fingerprint(str(runtime.directory))/NAME):
        immutable_record(path, {'value': value, 'sha256': fingerprint(value)})
    return {'path': str(runtime.directory/NAME), 'sha256': file_hash(runtime.directory/NAME)}


def verify(reference, source, authority):
    from .neural_forward_ticket import _LifecycleReader, _verify_authority_binding
    path = project_path(reference['path'])
    if (set(reference) != {'path', 'sha256'} or path != project_path(source['runtime_directory'])/NAME
            or file_hash(path) != reference['sha256']):
        raise ValueError('Original historical archive binding changed')
    wrapped = _raw(path); value = wrapped['value']
    mirror = authority.directory/'efficiency_archive_inputs'/fingerprint(source['runtime_directory'])/NAME
    if wrapped != _raw(mirror) or wrapped['sha256'] != fingerprint(value):
        raise ValueError('Original archive binding mirrors differ')
    if (value['version'] != VERSION or value['policy'] != POLICY
            or value['source_files'] != authority.manifest['source_files']
            or value['authority_hash'] != authority.manifest_hash
            or value['runtime_directory'] != source['runtime_directory']
            or value['feed_directory'] != source['feed_directory']
            or value['source_id'] != source['source_id'] or source['observation_prefix'] is not None
            or value['delay_seconds'] != source['rolling']['availability_delay_seconds']):
        raise ValueError('Archive input cannot borrow a runtime, wallet, source or quarantine')
    reader = _LifecycleReader(source['runtime_directory'])
    try:
        _verify_authority_binding(reader, authority, reader.verify()['operations'])
    finally:
        reader.close()
    for record in value['archives']:
        original = _archive(record['path'])
        if any(record[k] != v for k, v in original.items()) or file_hash(project_path(record['import_file'])) != record['import_sha256']:
            raise ValueError('Original archive/import bytes changed')
        root = project_path(source['feed_directory'])
        batch_path = root/'raw'/f"{record['batch']}.json"
        pub_path = root/'publications'/f"{record['batch']}.json"
        batch, publication = _raw(batch_path), _raw(pub_path)
        if (file_hash(batch_path) != record['batch_sha256'] or file_hash(pub_path) != record['publication_sha256']
                or fingerprint(batch) != record['batch'] or batch['actual'] is not False
                or batch['source'] != source['source_id'] or publication['source'] != source['source_id']
                or publication['batch'] != record['batch']
                or batch['source_ref'] != {'path': record['import_file'], 'sha256': record['import_sha256']}
                or batch['prepared_at'] != record['actually_imported_at']
                or publication['published_at'] != record['actually_published_at']
                or pd.Timestamp(record['downloaded_at']) > pd.Timestamp(batch['prepared_at'])
                or pd.Timestamp(batch['prepared_at']) > pd.Timestamp(publication['published_at'])
                or pd.Timestamp(publication['published_at']) > pd.Timestamp.now(tz='UTC')):
            raise ValueError('Real archive import/publication lineage changed')
        database = root/'receipts.sqlite'
        with sqlite3.connect(database.as_uri()+'?mode=ro', uri=True) as db:
            original_batch = db.execute('SELECT record,hash FROM batches WHERE id=?', (record['batch'],)).fetchone()
            original_publication = db.execute('SELECT record,hash,published_at FROM publications WHERE batch=?', (record['batch'],)).fetchone()
        if (original_batch is None or original_publication is None
                or json.loads(original_batch[0]) != batch or original_batch[1] != fingerprint(batch)
                or json.loads(original_publication[0]) != publication
                or original_publication[1] != fingerprint(publication)
                or original_publication[2] != publication['published_at']):
            raise ValueError('Original physical feed publication missing from actual database')
    return value


def snapshot(reference, source, authority, cutoff):
    """Called only inside the original paid preparation/F worker lease."""
    value = verify(reference, source, authority)
    from .v7_feed import ReceiptArchive
    feed = ReceiptArchive(source['feed_directory'], tuple(source['symbols']))
    try:
        # Verifies original bar objects, publication and revision chains at actual
        # present time. It does not claim availability at the historical cutoff.
        actual = feed.snapshot(pd.Timestamp.now(tz='UTC'), archive_delay_seconds=value['delay_seconds'])
    finally:
        feed.close()
    if (actual.received_at.notna().any() or actual.prediction_origin.any()
            or set(actual.availability_basis) != {'assumed_archive'}
            or set(actual.batch_id) != {r['batch'] for r in value['archives']}):
        raise ValueError('Historical engineering input cannot mix actual receipts or later imports')
    end = pd.Timestamp(cutoff)
    if end.isoformat() not in value['cutoffs'] or end > pd.Timestamp.now(tz='UTC'):
        raise ValueError('Unregistered or future historical cutoff')
    frames = []
    for record in value['archives']:
        frame = load_archive(record['path'], record['symbol'], availability_delay_seconds=value['delay_seconds'])
        imported = pd.read_parquet(record['import_file'])
        pd.testing.assert_frame_equal(frame, imported, check_exact=True)
        frames.append(frame)
    full = pd.concat(frames, ignore_index=True).sort_values(['symbol','open_time']).reset_index(drop=True)
    columns = ['symbol','open_time','open','high','low','close','volume']
    observed = actual.loc[:, columns].sort_values(['symbol','open_time']).reset_index(drop=True)
    expected = full.loc[:, columns].copy()
    for column in columns[2:]:
        observed[column] = observed[column].astype(float)
        expected[column] = expected[column].astype(float)
    pd.testing.assert_frame_equal(expected, observed, check_exact=True)
    start = end-pd.DateOffset(months=6)-pd.Timedelta(hours=1234)
    selected = full.loc[(full.open_time >= start) & (full.open_time < end)].copy()
    for symbol in source['symbols']:
        stamps = pd.DatetimeIndex(selected.loc[selected.symbol == symbol, 'open_time'])
        if not stamps.equals(pd.date_range(start, end, freq='h', inclusive='left')):
            raise ValueError('Complete six calendar months plus 1234 warmup hours required')
    selected = selected.loc[selected.available_at <= end].copy()
    # No feed_identity: this frame must never masquerade as an as-of live receipt.
    selected.attrs = {'availability_mode': 'assumed_historical_delay',
                      'efficiency_archive_binding': reference, 'engineering_only': True,
                      'historical_received_at': None, 'formal_eligible': False}
    return selected
