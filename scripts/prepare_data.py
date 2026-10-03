"""Explicit bounded ingestion for the frozen protocol, never import side effects."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from factorlab.data import audit_dataset, download_month, load_archive
from factorlab.protocol import Protocol, project_path, save_json, utc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', default='config/protocol.json')
    args = parser.parse_args()
    protocol = Protocol.read(args.protocol)
    out = project_path('data/prepared') / protocol.digest[:16]
    protocol.freeze(out)
    months = pd.date_range(utc(protocol.start), utc(protocol.end), freq='MS', inclusive='left')
    provenance, outputs = [], {}
    for kind in ('klines', 'mark', 'funding'):
        parts = []
        for symbol in protocol.symbols:
            for month in months:
                path = download_month(symbol, month.strftime('%Y-%m'), kind=kind)
                frame = load_archive(path, symbol, kind, protocol.availability_delay_seconds)
                provenance.append({'path': str(path), **frame.attrs['provenance']})
                frame.attrs = {}
                parts.append(frame)
                print(f'{kind} {symbol} {month:%Y-%m}: {len(frame)} rows', flush=True)
        joined = pd.concat(parts, ignore_index=True)
        key = 'funding_time' if kind == 'funding' else 'open_time'
        joined = joined.sort_values(['symbol', key]).reset_index(drop=True)
        if joined.duplicated(['symbol', key]).any():
            raise ValueError('Duplicate observations across archives')
        if kind != 'funding':
            audit = audit_dataset(joined)
            expected = pd.date_range(utc(protocol.start), utc(protocol.end),
                                     freq='h', inclusive='left')
            for symbol in protocol.symbols:
                actual = pd.DatetimeIndex(joined.loc[joined.symbol == symbol, key])
                if not actual.equals(expected):
                    raise ValueError(f'Incomplete requested calendar {kind}/{symbol}')
        else:
            audit = {'rows': len(joined), 'event_count_by_symbol':
                     joined.groupby('symbol').size().to_dict(),
                     'intervals_are_observed': True}
        file = out / f'{kind}.parquet'
        joined.to_parquet(file, index=False)
        outputs[kind] = {'path': str(file), 'sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
                         'audit': audit}
    result = {'protocol_hash': protocol.digest, 'outputs': outputs, 'archives': provenance,
              'limitations': ['Fixed surviving asset selection, not point-in-time universe.',
                              'Archive revisions and historical 5-second feed delay are assumptions.',
                              'Hour-open fills and mark prices are proxies, not executable tick quotes.']}
    save_json(out / 'manifest.json', result)
    save_json('outputs/phase_audits/data_manifest.json', result)
    print(json.dumps({'prepared_directory': str(out), 'protocol_hash': protocol.digest}))


if __name__ == '__main__':
    main()
