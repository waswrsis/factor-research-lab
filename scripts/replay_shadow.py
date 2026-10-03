"""Verify frozen model observation semantics on labelled historical replay."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from factorlab.dataset import make_dataset
from factorlab.experiment import load_market
from factorlab.protocol import Protocol, fingerprint, project_path, save_json
from factorlab.shadow import ShadowLedger, freeze_bundle, verify_bundle
from factorlab.training import FittedModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', default='config/protocol.json')
    parser.add_argument('--model', default='runs/p1-market-v1/fold-0/A_ridge/model')
    parser.add_argument('--bundle', default='models/frozen/p1-baseline-v1')
    parser.add_argument('--ledger', default='runs/shadow-replay-v1')
    args = parser.parse_args()
    protocol = Protocol.read(args.protocol)
    model_metadata = json.loads((project_path(args.model)/'model.json').read_text('utf-8'))
    manifest = freeze_bundle(args.model, args.bundle, {
        'protocol_hash': protocol.digest, 'model_directory': '.',
        'role': 'engineering_baseline_not_confirmed_alpha',
        'fit_label_cutoff': model_metadata['diagnostics']['fit_label_cutoff']})
    verified = verify_bundle(args.bundle)
    if verified['bundle_hash'] != manifest['bundle_hash']:
        raise ValueError('Trusted bundle hash mismatch')
    model = FittedModel.load(args.bundle)
    market, _ = load_market(protocol)
    data = make_dataset(market['klines'], protocol)
    start = pd.Timestamp(protocol.test_starts[0], tz='UTC')
    subset = data.subset(((data.meta.available_at >= start) &
                          (data.meta.available_at < start+pd.Timedelta(hours=72))).to_numpy())
    prediction = model.predict(subset)
    ledger = ShadowLedger(args.ledger, protocol.digest, manifest['bundle_hash'],
                          observation_mode='historical_replay',
                          availability_delay_seconds=protocol.availability_delay_seconds)
    events = []
    for i, row in subset.meta.iterrows():
        identity = fingerprint({'symbol': row.symbol, 'entry_time': row.entry_time.isoformat(),
                                'model': manifest['bundle_hash']})
        events.append((row.available_at, 'prediction', identity, i))
        events.append((row.label_available_at, 'outcome', identity, i))
    for now, kind, identity, i in sorted(events):
        row = subset.meta.iloc[i]
        if kind == 'prediction':
            ledger.append_prediction(identity, row.symbol, row.available_at, row.entry_time,
                                      row.label_end, float(prediction[i]), now)
        else:
            ledger.append_outcome(identity, float(subset.y[i]), row.label_available_at, now)
    summary = ledger.summary(max(x[0] for x in events))
    result = {'bundle': str(project_path(args.bundle)), 'trusted_bundle_hash': manifest['bundle_hash'],
              'replay': summary, 'actual_forward_observations': 0,
              'claim': 'Historical replay verifies software mechanics, not future performance.'}
    save_json('outputs/phase_audits/shadow_replay.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
