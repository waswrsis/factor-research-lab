"""Original preregistration and quarantine binding for full pipeline trials.

This is execution-scope infrastructure, not a calibration certificate. It
does not authorize production claims, assert generator independence, identify
a resolved provider model, mint another budget, or simulate a UTC work month.
The future full controller must reuse the production kernels under this
verified namespace; a mere enrollment does not prove those kernels ran.
"""
from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from filelock import FileLock

from factorlab.discovery.calibration_acceptance import AcceptanceCell, AcceptancePlan
from factorlab.discovery.calibration_runtime_contract import (
    _original_operations,
    runtime_calibration_contract,
)
from factorlab.discovery.calibration_trial_context import _no_redirect
from factorlab.discovery.evidence import file_hash, snapshot_source
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority, source_files
from factorlab.discovery.neural_forward_ticket import _bind_authority, _LifecycleReader
from factorlab.discovery.v7_forward import ForwardRuntime
from factorlab.discovery.v7_lifecycle import Lifecycle, canonical
from factorlab.discovery.v7_storage import immutable_record
from factorlab.protocol import fingerprint, project_path

VERSION = 'v7-full-pipeline-quarantine-registry-v1'
EVENT = 'full_calibration_quarantine_enrolled'


def _plain(value):
    return json.loads(canonical(value))


def _directory(value):
    path = Path(value)
    if not path.is_absolute():
        path = project_path('.') / path
    _no_redirect(path, project_path('.'))
    return project_path(path)


def _identity(directory, plan):
    return {'version': VERSION, 'plan': plan.to_dict(),
        'plan_digest': plan.digest, 'directory': str(directory),
        'source_files': source_files(), 'registry_implementation_hash': file_hash(Path(__file__)),
        'claims': ['library_skill', 'factor1', 'factor2', 'factor3', 'algorithm'],
        'fixed_estimand_replicas': {'slots': ['library_skill', 'factor1', 'factor2', 'factor3'],
            'support': 'one_original_replica_per_registered_atom_and_selected_slot',
            'budget': 'separate_original_standard_wallet_per_replica_same_limits_not_production_credit'},
        'execution_scope': 'quarantined_simulation_only',
        'work_clock': 'original_physical_UTC_and_monotonic_never_virtual_reset',
        'generator_independence_certified': False, 'provider_authenticity_certified': False,
        'formal_gate_eligible': False, 'full_pipeline_execution_verified': False}


def _digest(value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Original digest required')
    return value


def _read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate quarantine evidence key')
            result[key] = value
        return result

    def invalid(_):
        raise ValueError('Nonfinite quarantine evidence')

    if not path.is_file() or path.stat().st_size > 16 * 2**20:
        raise ValueError('Original quarantine evidence missing or oversized')
    return json.loads(path.read_text('utf-8'), object_pairs_hook=unique, parse_constant=invalid)


@dataclass(frozen=True)
class FullTrialPlan:
    acceptance: AcceptancePlan
    runtime_scope: dict
    provider_policy: dict
    generator_design: dict
    seed: int = 918273
    oracle_paths: int = 128
    mechanism_contract: dict | None = None

    def __post_init__(self):
        if (type(self.acceptance) is not AcceptancePlan or type(self.seed) is not int
                or not 0 <= self.seed < 2**32 or type(self.oracle_paths) is not int
                or not 128 <= self.oracle_paths <= 2_000_000):
            raise ValueError('Full fixed-grid acceptance and original seed/path plan required')
        for field in ('runtime_scope', 'provider_policy', 'generator_design'):
            value = getattr(self, field)
            if type(value) is not dict or not value or len(canonical(value).encode()) > 65536:
                raise ValueError('Bounded explicit full trial scope required')
            object.__setattr__(self, field, _plain(value))
        if self.runtime_scope.get('scope') != 'actual_runtime_component_only':
            raise ValueError('Actual runtime component required, not default rolling hashes')
        if self.runtime_scope.get('source_files') != source_files():
            raise ValueError('Original runtime source differs from current source')
        policy = self.runtime_scope['work_budget'].get('partition_policy')
        if type(policy) is not dict or policy.get('mode') != 'standard':
            raise ValueError('Full adaptive trials require the original standard shared wallet')
        if set(self.provider_policy) != {'backend', 'model_requested', 'reasoning_requested', 'context_policy'}:
            raise ValueError('Exact requested provider policy required')
        if (self.provider_policy['backend'] != 'codex'
                or self.provider_policy['context_policy'] != 'fresh_codex_per_trial_v1'
                or any(type(self.provider_policy[k]) is not str or not self.provider_policy[k]
                       for k in ('model_requested', 'reasoning_requested'))):
            raise ValueError('Actual isolated Codex policy required; no callback or cached provider')
        from .prompt_builder import DEFAULT_MODEL, DEFAULT_REASONING
        if (self.provider_policy['model_requested'], self.provider_policy['reasoning_requested']) != (
                DEFAULT_MODEL, DEFAULT_REASONING):
            raise ValueError('Full trial provider must match the actual restricted production worker')
        if (set(self.generator_design) != {'design_digest', 'validation_digest', 'oracle_digest'}
                or len(set(self.generator_design.values())) != 3):
            raise ValueError('Separate original design, validation and oracle generator domains required')
        for value in self.generator_design.values():
            _digest(value)
        if self.mechanism_contract is not None:
            from .full_mechanism_contract import verify_mechanism_contract, mechanism_domains
            original = _plain(self.mechanism_contract)
            generators = verify_mechanism_contract(original, self.acceptance)
            if (self.generator_design != mechanism_domains(original)
                    or any(g.law.support_size != self.oracle_paths for g in generators.values())):
                raise ValueError('Prospective mechanism differs from original draw domains or full support')
            object.__setattr__(self, 'mechanism_contract', original)

    def to_dict(self):
        value = _plain(asdict(self))
        if self.mechanism_contract is None:
            value.pop("mechanism_contract")
        return value

    @classmethod
    def from_dict(cls, value):
        value = _plain(value)
        fields = value['acceptance']
        fields['cells'] = tuple(AcceptanceCell(**{**cell, 'power_slots': tuple(cell['power_slots'])})
                                for cell in fields['cells'])
        value['acceptance'] = AcceptancePlan(**fields)
        return cls(**value)

    @property
    def digest(self):
        return fingerprint(self.to_dict())


class FullTrialRegistry:
    """Freeze the whole grid before enrollment; inspection never initializes it."""
    def __init__(self, directory, plan):
        if type(plan) is not FullTrialPlan:
            raise TypeError('Frozen full trial plan required')
        self.directory = _directory(directory)
        self.plan = FullTrialPlan.from_dict(plan.to_dict())
        self.identity = _identity(self.directory, self.plan)
        self.digest = fingerprint(self.identity)
        self.directory.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.directory / '.lock'), timeout=0):
            path = self.directory / 'identity.json'
            if not path.exists():
                if any(p.name != '.lock' for p in self.directory.iterdir()):
                    raise ValueError('Existing original registry cannot be reset')
                archived = snapshot_source(self.directory / 'source')
                if archived['files'] != self.identity['source_files']:
                    raise ValueError('Source changed during preregistration')
            immutable_record(path, self.identity)
            self.verify()

    @classmethod
    def inspect(cls, directory):
        result = cls.__new__(cls)
        result.directory = _directory(directory)
        result.identity = _read(result.directory / 'identity.json')
        result.plan = FullTrialPlan.from_dict(result.identity['plan'])
        result.digest = fingerprint(result.identity)
        result.verify()
        return result

    def verify(self):
        _no_redirect(self.directory, project_path('.'))
        if (_read(self.directory / 'identity.json') != self.identity
                or self.identity != _identity(self.directory, self.plan)
                or self.identity['version'] != VERSION
                or self.identity['directory'] != str(self.directory)
                or self.identity['plan_digest'] != self.plan.digest
                or self.identity['plan'] != self.plan.to_dict()
                or self.identity['source_files'] != source_files()
                or self.identity['registry_implementation_hash'] != file_hash(Path(__file__))
                or self.digest != fingerprint(self.identity)
                or self.identity['formal_gate_eligible'] is not False
                or self.identity['full_pipeline_execution_verified'] is not False):
            raise ValueError('Original full trial registry changed or self-authorized')
        for name, digest in self.identity['source_files'].items():
            if file_hash(_no_redirect(self.directory / 'source' / name, self.directory)) != digest:
                raise ValueError('Original registry source bytes changed')
        return {'registry_digest': self.digest, 'formal_gate_eligible': False,
                'planned_validation_repetitions': sum(c.repetitions for c in self.plan.acceptance.cells)}

    def identity_for(self, cell, index, *, oracle_index=None, estimand_slot=None):
        found = next((c for c in self.plan.acceptance.cells if c.name == cell), None)
        if (found is None or type(index) is not int or not 0 <= index < found.repetitions
                or oracle_index is not None and (type(oracle_index) is not int
                    or not 0 <= oracle_index < self.plan.oracle_paths)):
            raise ValueError('Trial/path is outside the whole preregistered grid')
        if estimand_slot is not None and (oracle_index is None or estimand_slot not in
                ('library_skill', 'factor1', 'factor2', 'factor3')):
            raise ValueError('Fixed oracle replica requires an original nonadaptive claim and atom')
        role = 'validation' if oracle_index is None else 'oracle'
        base = {'registry_digest': self.digest, 'cell': cell, 'index': index}
        parent = fingerprint(base)
        binding = {**base, 'role': role, 'oracle_index': oracle_index}
        if estimand_slot is not None:
            binding['estimand_slot'] = estimand_slot
        trial_id = fingerprint(binding)
        # Domain-separated original seed material, not a claim of random independence.
        seed_material = fingerprint({'seed': self.plan.seed, 'trial': binding,
            'generator': self.plan.generator_design[role + '_digest']})
        return {**binding, 'trial_id': trial_id, 'population_trial_id': parent,
                'scenario_digest': found.scenario_digest, 'market_seed_material': seed_material}

    def paths(self, cell, index, *, oracle_index=None, estimand_slot=None):
        self.identity_for(cell, index, oracle_index=oracle_index, estimand_slot=estimand_slot)
        # Canonical grid coordinates avoid nesting two 64-character hashes in
        # existing wallet paths on Windows. This is injective for the entire
        # frozen grid; the complete 256-bit trial identity remains in every
        # original enrollment, so the short spelling cannot relabel a trial.
        ordinal = next(i for i, c in enumerate(self.plan.acceptance.cells) if c.name == cell)
        name = (f'v{ordinal:02d}-{index:04d}' if oracle_index is None
                else f'o{ordinal:02d}-{index:04d}-{oracle_index:07d}')
        if estimand_slot is not None:
            slot = ('library_skill', 'factor1', 'factor2', 'factor3').index(estimand_slot)
            name = f'f{ordinal:02d}-{index:04d}-{slot}-{oracle_index:07d}'
        root = self.directory / 'trials' / name
        paths = {'root': root, 'runtime': root / 'forward', 'feed': root / 'feed',
                 'authority': root / 'wallet', 'research': root / 'forward' / 'research',
                 'provider': root / 'provider'}
        for path in paths.values():
            _no_redirect(path, self.directory)
        return paths

    def enroll(self, runtime, authority, cell, index, *, oracle_index=None, estimand_slot=None):
        """Bind before the first controller job; recover only the exact same intent.

        Enrollment alone grants no confirmation or adoption permission yet.
        The production kernels must explicitly recognize the verified typed
        permit in their quarantine branch; they currently do not do so.
        """
        if type(runtime) is not ForwardRuntime or type(authority) is not MonthlyWorkAuthority:
            raise TypeError('Actual production runtime and original shared authority required')
        self.verify()
        trial = self.identity_for(cell, index, oracle_index=oracle_index, estimand_slot=estimand_slot)
        paths = self.paths(cell, index, oracle_index=oracle_index, estimand_slot=estimand_slot)
        if estimand_slot is not None:
            from .calibration_oracle_evidence import _frozen_estimands
            validation = QuarantinedTrialPermit(self, self.identity_for(cell, index))
            _, specs = _frozen_estimands(validation)
            selected = next(s for s in specs if s.slot == estimand_slot)
            if selected.candidate is None:
                raise ValueError('Original nonselection cannot initialize a replacement oracle learner')
        if (runtime.directory != paths['runtime'] or runtime.feed.directory != paths['feed']
                or authority.directory != paths['authority']):
            raise ValueError('Runtime/feed/wallet must belong to this exact quarantined trial')
        if runtime.lifecycle.plan.calibration_directory is not None:
            raise ValueError('A quarantined trial must not borrow a production certificate')
        expected_run = fingerprint({'registry': self.digest, 'trial': trial['trial_id']})
        if authority.manifest['scope']['run_id'] != expected_run:
            raise ValueError('Another run wallet cannot enter this trial')
        with FileLock(str(paths['root'] / '.enrollment.lock'), timeout=0):
            _bind_authority(runtime, authority)
            lc = runtime.lifecycle
            lc.verify()
            actual = runtime_calibration_contract(lc)
            # receipt source is trial-specific; everything scientific remains exact.
            scoped = {k: v for k, v in actual.items() if k != 'receipt_source_id'}
            expected = {k: v for k, v in self.plan.runtime_scope.items() if k != 'receipt_source_id'}
            if scoped != expected:
                raise ValueError('Trial changes the preregistered production runtime scope')
            intent = {**trial, 'paths': {k: str(v) for k, v in paths.items()},
                      'runtime_scope': actual, 'runtime_scope_hash': fingerprint(actual),
                      'lifecycle_plan_digest': lc.plan.digest,
                      'authority_manifest_hash': authority.manifest_hash,
                      'execution_scope': 'quarantined_simulation_only'}
            operations = _original_operations(lc)
            enrolled = [o for o in operations if o['kind'] == EVENT]
            target = paths['root'] / 'enrollment.json'
            original_intent = paths['root'] / 'enrollment_intent.json'
            if enrolled:
                if len(enrolled) != 1 or enrolled[0]['detail'] != intent or _read(original_intent) != intent:
                    raise ValueError('Original quarantine enrollment changed')
                # If final mirror was lost after a commit, never recreate it here.
                if not target.exists():
                    raise ValueError('Original enrolled marker missing; inspect without resetting')
            else:
                if target.exists() or original_intent.exists():
                    raise ValueError('Ambiguous enrollment intent; no new permission')
                allowed = {'forward_runtime_bound', 'monthly_work_authority_bound'}
                if any(o['kind'] not in allowed for o in operations):
                    raise ValueError('Quarantine must enroll before any production job or outcome')
                if any(lc.db.execute('SELECT COUNT(*) FROM ' + table).fetchone()[0]
                       for table in ('jobs', 'active', 'batches', 'settlement')):
                    raise ValueError('Quarantine cannot retrospectively enroll observed work')
                if lc.forecasts.db.execute('SELECT COUNT(*) FROM forecasts').fetchone()[0]:
                    raise ValueError('Quarantine cannot enroll after forecasts have been observed')
                if authority.verify()['reservations']:
                    raise ValueError('Quarantine cannot borrow already charged research history')
                immutable_record(original_intent, intent)
                with lc.db:
                    lc._event(EVENT, intent)
                event = next(o for o in _original_operations(lc) if o['kind'] == EVENT)
                immutable_record(target, {'intent': intent, 'original_event_hash': fingerprint(event)})
            permit = QuarantinedTrialPermit(self, trial)
            permit.verify(lc)
            return permit


class QuarantinedTrialPermit:
    """Original scope handle; public verification never means formal acceptance."""
    def __init__(self, registry, trial):
        if type(registry) is not FullTrialRegistry:
            raise TypeError('Original full registry required')
        self.registry = registry
        expected = registry.identity_for(trial['cell'], trial['index'], oracle_index=trial['oracle_index'],
                                         estimand_slot=trial.get('estimand_slot'))
        if trial != expected:
            raise ValueError('Unplanned quarantine trial')
        self.trial = _plain(trial)
        self.paths = registry.paths(trial['cell'], trial['index'], oracle_index=trial['oracle_index'],
                                    estimand_slot=trial.get('estimand_slot'))

    @contextmanager
    def open_lifecycle(self):
        """Only existing mode=ro stores; never a constructor repairing evidence."""
        reader = _LifecycleReader(self.paths['runtime'])
        forecasts = None
        try:
            self.verify(reader)
            from factorlab.discovery.model_registry import ForecastLedger
            reader.forecasts = ForecastLedger.__new__(ForecastLedger)
            reader.forecasts.path = self.paths['runtime'] / 'forecasts.sqlite'
            reader.forecasts.records_directory = reader.forecasts.path.with_suffix('.records')
            _no_redirect(reader.forecasts.path, self.registry.directory)
            _no_redirect(reader.forecasts.records_directory, self.registry.directory)
            if (not reader.forecasts.path.is_file()
                    or not reader.forecasts.records_directory.is_dir()):
                raise ValueError('Original forecast ledger or receipt directory missing')
            forecasts = sqlite3.connect(reader.forecasts.path.as_uri() + '?mode=ro', uri=True)
            reader.forecasts.db = forecasts
            reader.forecasts.verify()
            yield reader
        finally:
            if forecasts is not None:
                forecasts.close()
            reader.close()

    def verify(self, lifecycle=None):
        self.registry.verify()
        for path in self.paths.values():
            _no_redirect(path, self.registry.directory)
        target = self.paths['root'] / 'enrollment.json'
        value = _read(_no_redirect(target, self.registry.directory))
        intent = _read(_no_redirect(self.paths['root'] / 'enrollment_intent.json', self.registry.directory))
        if value.get('intent') != intent:
            raise ValueError('Quarantine enrollment differs from its original intent')
        if lifecycle is None:
            reader = _LifecycleReader(self.paths['runtime'])
            try:
                return self.verify(reader)
            finally:
                reader.close()
        if not isinstance(lifecycle, Lifecycle) or lifecycle.directory != self.paths['runtime']:
            raise ValueError('Quarantine permission cannot authorize a foreign lifecycle')
        operations = _original_operations(lifecycle)
        events = [o for o in operations if o['kind'] == EVENT]
        actual = runtime_calibration_contract(lifecycle)
        if (len(events) != 1 or events[0]['detail'] != intent
                or value != {'intent': intent, 'original_event_hash': fingerprint(events[0])}
                or any(intent.get(k) != v for k, v in self.trial.items())
                or intent['paths'] != {k: str(v) for k, v in self.paths.items()}
                or intent['runtime_scope'] != actual
                or intent['runtime_scope_hash'] != fingerprint(actual)
                or intent['lifecycle_plan_digest'] != lifecycle.plan.digest):
            raise ValueError('Original quarantine runtime/chain/scope changed')
        return {**self.trial, 'paths': intent['paths'], 'runtime_scope_hash': fingerprint(actual),
                'execution_scope': 'quarantined_simulation_only', 'formal_gate_eligible': False,
                'provider_authenticity_certified': False, 'generator_independence_certified': False,
                'full_pipeline_execution_verified': False}
