"""Explicit one-shot public-data observation; no service or background task."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factorlab.forward import observe_once
from factorlab.protocol import Protocol, project_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--expected-hash', required=True)
    parser.add_argument('--ledger', required=True)
    parser.add_argument('--protocol', default='config/protocol.json')
    parser.add_argument('--once', action='store_true', help='One-shot is always the default')
    parser.add_argument('--settle', action='store_true', help='Settle at most 20 mature prior forecasts')
    args = parser.parse_args()
    result = observe_once(args.bundle, args.expected_hash, args.ledger,
                          Protocol.read(project_path(args.protocol)), settle=args.settle)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
