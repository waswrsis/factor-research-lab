"""Finite local variation plans; no data, labels, fitting or executable code."""
from __future__ import annotations

import copy

from .program_contract import Program


def local_variations(program, donor, count, *, feature_names, seed=0):
    if type(count) is not int or not 1 <= count <= 8:
        raise ValueError('One to eight bounded local variations required')
    base, donor = Program.from_dict(program).to_dict(), Program.from_dict(donor).to_dict()
    paths = []

    def walk(node, path):
        if not isinstance(node, dict):
            return
        if node.get('op') == 'feature':
            paths.append((path + ['name'], 'feature'))
        if node.get('op') in {'rolling_mean', 'rolling_std'}:
            paths.append((path + ['window'], 'window'))
        for key in ('arg', 'left', 'right', 'alpha'):
            if key in node:
                walk(node[key], path + [key])

    walk(base['outputs'][0], ['outputs', 0])
    attempts = []
    for index in range(count):
        if index == 0 or not paths:
            # Original parent is a charged control; exact duplicates stay visible.
            attempts.append({'kind': 'replace', 'path': ['outputs', 0],
                             'value': copy.deepcopy(base['outputs'][0])})
        elif index == count - 1 and not donor['parameters'] and donor['feature_names'] == base['feature_names']:
            attempts.append({'kind': 'replace', 'path': ['outputs', 0],
                             'value': copy.deepcopy(donor['outputs'][0])})
        else:
            path, kind = paths[(seed + index) % len(paths)]
            replacement = feature_names[(seed + index) % min(8, len(feature_names))] if kind == 'feature' else (2, 4, 8, 12, 24, 48)[(seed + index) % 6]
            attempts.append({'kind': 'replace', 'path': path, 'value': replacement})
    return attempts
