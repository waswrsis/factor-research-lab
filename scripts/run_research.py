import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse

from factorlab.experiment import run_experiment
from factorlab.protocol import Protocol


def main():
    parser = argparse.ArgumentParser(description='D-drive offline research, no trading orders')
    parser.add_argument('--protocol', default='config/protocol.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--phases', choices=['p1', 'all'], default='all')
    parser.add_argument('--fold-limit', type=int)
    parser.add_argument('--external-proposals')
    parser.add_argument('--require-external', action='store_true')
    args = parser.parse_args()
    run_experiment(Protocol.read(args.protocol), args.output, phases=args.phases,
                   fold_limit=args.fold_limit, external_proposals=args.external_proposals,
                   require_external=args.require_external)


if __name__ == '__main__':
    main()
