"""Explicit finite score/selection diagnostics; never launches research/trading."""
import argparse
import json
from dataclasses import asdict

from factorlab.discovery.layered_validation import LayeredPlan, binding, run, verify_report

from factorlab.protocol import project_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('plan', 'run'):
        action = commands.add_parser(command)
        action.add_argument('--plan', required=True, help='Explicit frozen LayeredPlan JSON')
        action.add_argument('--directory', required=True)
    review = commands.add_parser('verify')
    review.add_argument('--directory', required=True)
    review.add_argument('--binding', required=True)
    args = parser.parse_args(argv)
    if args.command == 'verify':
        value = verify_report(args.directory, expected_binding=args.binding)
    else:
        fields = json.loads(project_path(args.plan).read_text('utf-8'))
        if not isinstance(fields, dict):
            raise ValueError('Explicit layered plan object required')
        plan = LayeredPlan(**fields)
        if args.command == 'plan':
            value = {'status': 'PLANNING_ONLY', 'binding': binding(plan),
                     'plan': asdict(plan), 'directory': str(project_path(args.directory)),
                     'fits': 0, 'GPT_calls': 0, 'work_started': False}
        else:
            value = run(args.directory, plan)
    print(json.dumps(value, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
