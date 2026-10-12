"""Read-only scientific scope of an actual forward runtime and its wallet.

This is one component of the full calibration binding, never a certificate.
The original runtime journal authenticates configuration; hashing a replacement
forward.json cannot change its rolling rule. No store is initialized or repaired.
Provider, independent trials, truth and statistical acceptance need their own
verified full-pipeline artifacts before a formal gate can open.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import asdict

from factorlab.protocol import fingerprint, project_path

from .confirmatory import ConfirmationRule
from .evidence import file_hash
from .monthly_work_authority import source_files
from .rolling import RollingSpec

VERSION = 'v7-actual-runtime-calibration-component-v1'


class RuntimeScopeUnavailable(ValueError):
    """An offline lifecycle or not-yet-bound wallet has no production scope."""


def _read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate calibration runtime JSON key')
            result[key] = value
        return result

    def invalid(_):
        raise ValueError('Nonfinite calibration runtime JSON')

    if not path.is_file() or path.stat().st_size > 16 * 2**20:
        raise ValueError('Missing or oversized calibration runtime evidence')
    return json.loads(path.read_text('utf-8'), object_pairs_hook=unique, parse_constant=invalid)


def _plain(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _original_operations(lifecycle):
    """Read the actual immutable chain, without recursively calling LC.verify."""
    path = lifecycle.directory / 'lifecycle.sqlite'
    if not path.is_file():
        raise ValueError('Original lifecycle journal missing')
    previous, operations = '0' * 64, []
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as db:
        for sequence, text, digest in db.execute('SELECT * FROM operations ORDER BY seq'):
            operation = json.loads(text)
            if (sequence != len(operations) + 1 or operation['seq'] != sequence
                    or operation['previous'] != previous or fingerprint(operation) != digest):
                raise ValueError('Calibration runtime journal chain changed')
            previous = digest
            operations.append(operation)
    return operations


def runtime_calibration_contract(lifecycle):
    """Derive actual rolling/availability/source/budget scope, never defaults.

    Paths, run IDs and the requested certificate's own path/digest are excluded
    from the scientific component: including its own digest creates a circular
    binding. Their original on-disk identities are nevertheless verified here.
    No result from this helper establishes full execution or oracle scope.
    """
    from .neural_forward_ticket import _verify_authority_binding, open_existing_authority
    from .v7_forward import ForwardConfig
    from .v7_lifecycle import Lifecycle

    if not isinstance(lifecycle, Lifecycle):
        raise TypeError('Actual Lifecycle required for calibration scope')
    root = lifecycle.directory
    if not (root / 'forward.json').is_file():
        raise RuntimeScopeUnavailable('Offline lifecycle has no actual forward calibration scope')
    wrapper = _read(root / 'forward.json')
    if (set(wrapper) != {'identity', 'sha256'}
            or fingerprint(wrapper['identity']) != wrapper['sha256']):
        raise ValueError('Forward calibration manifest integrity changed')
    identity = wrapper['identity']
    if (set(identity) != {'version', 'config', 'lifecycle', 'rolling', 'source_files', 'feed_directory'}
            or identity['version'] != 'v7-forward-v1'):
        raise ValueError('Unsupported actual forward calibration identity')
    fields = dict(identity['config'])
    fields['symbols'] = tuple(fields['symbols'])
    config = ForwardConfig(**fields)
    rolling_fields = dict(identity['rolling'])
    rolling_fields['event_bins'] = tuple(rolling_fields['event_bins'])
    spec = RollingSpec(**rolling_fields)
    # The real ForwardRuntime has precisely this rule; caller-added fields or
    # a separately supplied RollingSpec are not another calibrated variant.
    expected_spec = RollingSpec(availability_delay_seconds=config.archive_delay_seconds,
                                window_days=config.training_window_days)
    if (_plain(asdict(config)) != identity['config'] or _plain(asdict(spec)) != identity['rolling']
            or spec.digest != expected_spec.digest):
        raise ValueError('Actual forward rolling/availability contract differs')
    plan = _plain(asdict(lifecycle.plan))
    if (identity['lifecycle'] != plan or tuple(config.symbols) != tuple(lifecycle.plan.symbols)
            or config.training_window_days != lifecycle.plan.training_window_days
            or config.refit_interval_days != lifecycle.plan.refit_interval_days
            or _read(root / 'lifecycle.json') != {'plan': plan, 'sha256': lifecycle.plan.digest}):
        raise ValueError('Calibration runtime lifecycle/universe differs')
    operations = _original_operations(lifecycle)
    bindings = [o['detail'] for o in operations if o['kind'] == 'forward_runtime_bound']
    if bindings != [{'runtime_identity_hash': wrapper['sha256']}]:
        raise ValueError('Original runtime binding missing or replaced')
    if identity['source_files'] != source_files():
        raise ValueError('Calibration runtime source differs from current frozen source')
    if _read(root / 'source/manifest.json') != identity['source_files']:
        raise ValueError('Original runtime source manifest changed')
    for name, digest in identity['source_files'].items():
        if file_hash(root / 'source' / name) != digest:
            raise ValueError('Original runtime source archive bytes changed')
    feed = project_path(identity['feed_directory'])
    if _read(feed / 'identity.json') != {'source_id': lifecycle.plan.source_id,
                                       'symbols': list(config.symbols)}:
        raise ValueError('Calibration runtime actual feed identity changed')
    wallet_bindings = [o['detail'] for o in operations if o['kind'] == 'monthly_work_authority_bound']
    if not wallet_bindings:
        raise RuntimeScopeUnavailable('Actual runtime has no bound shared monthly authority')
    if len(wallet_bindings) != 1:
        raise ValueError('Calibration runtime has ambiguous wallet bindings')
    authority = open_existing_authority(wallet_bindings[0]['authority_directory'])
    _verify_authority_binding(lifecycle, authority, operations)
    if authority.manifest['source_files'] != identity['source_files']:
        raise ValueError('Calibration wallet and actual runtime source differ')
    lifecycle_policy = {k: v for k, v in plan.items()
                        if k not in {'source_id', 'calibration_directory', 'calibration_binding'}}
    # Factor C uses this frozen rule in reserve_selected_confirmation and
    # _verify_batches. Other endpoint rules remain separate full-scope terms.
    rule = ConfirmationRule()
    value = {
        'version': (VERSION if spec.window_days is None else
                    'v7-actual-elapsed-runtime-calibration-component-v2'),
        'scope': 'actual_runtime_component_only',
        'forward': _plain(asdict(config)), 'rolling': _plain(asdict(spec)),
        'rolling_hash': spec.digest, 'receipt_source_id': lifecycle.plan.source_id,
        'lifecycle_policy': lifecycle_policy, 'factor_confirmation_rule': rule.to_dict(),
        'factor_confirmation_rule_hash': rule.digest, 'source_files': identity['source_files'],
        'work_budget': {'authority_version': authority.manifest['version'],
                       'task': authority.manifest['scope']['task'],
                       'universe': authority.manifest['scope']['universe'],
                       'limits': authority.manifest['limits'],
                       'monthly_seconds': authority.manifest['monthly_seconds'],
                       'partition_policy': authority.manifest.get('partition_policy')},
    }
    return _plain(value)


def runtime_calibration_digest(lifecycle):
    return fingerprint(runtime_calibration_contract(lifecycle))
