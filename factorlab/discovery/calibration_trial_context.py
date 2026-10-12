"""Application-context isolation for predeclared calibration repetitions.

Fresh CLI turns and distinct research/feed/exposure/wallet paths eliminate
software-carried history between repetitions. They do NOT certify independent
provider randomness, model identity, a market generator, OS containment or the
complete five-claim procedure. Those remain separate calibration obligations.
"""
from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path

from factorlab.protocol import fingerprint, project_path

from .hypotheses import MAX_SHARDS
from .process_policy import DISABLED_FEATURES, RESTRICTED_OVERRIDES
from .prompt_builder import DEFAULT_MODEL, DEFAULT_REASONING
from .v7_agent import CodexBatchBackend
from .v7_storage import immutable_record

POLICY = 'fresh_codex_per_trial_v1'


class TrialIsolationError(RuntimeError):
    """Trusted-context failure, never a reply-repair request or fresh retry."""


def _read(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise TrialIsolationError('Duplicate trial context JSON key')
            value[key] = item
        return value

    def invalid(_):
        raise TrialIsolationError('Nonfinite trial context JSON')

    try:
        if not path.is_file() or path.stat().st_size > 16 * 2**20:
            raise TrialIsolationError('Missing or oversized original trial context evidence')
        return json.loads(path.read_text('utf-8'), object_pairs_hook=unique, parse_constant=invalid)
    except (OSError, ValueError) as exc:
        raise TrialIsolationError('Damaged original trial context evidence') from exc


def _no_redirect(path, root):
    """Reject symlinks/junctions before reading potentially foreign memory."""
    path, root = Path(path), Path(root)
    if not path.is_absolute() or not path.is_relative_to(root):
        raise TrialIsolationError('Trial context path is outside its own namespace')
    for node in [path, *path.parents]:
        if node.is_symlink() or (hasattr(node, 'is_junction') and node.is_junction()):
            raise TrialIsolationError('Redirected trial context namespace')
        if node == root:
            break
    if path.resolve() != path:
        raise TrialIsolationError('Noncanonical trial context namespace')
    return path


def trial_paths(directory, trial):
    if type(trial) is not str or not re.fullmatch(r'[a-zA-Z0-9_-]{1,40}', trial):
        raise TrialIsolationError('Invalid predeclared trial name')
    root = project_path(directory)
    inputs = root/'isolated_inputs'/trial
    paths = {'research': root/'trials'/trial/'research', 'provider': root/'providers'/trial,
             'feed': inputs/'feed', 'exposure': inputs/'forward'/'exposure.sqlite',
             'authority': inputs/'wallet'}
    for path in paths.values():
        _no_redirect(path, root)
    return paths


def _verify_memory_namespace(research, *, through_month=None):
    """Check each link BEFORE loading it, then retain normal memory validation.

    Genuine predecessor chains inside one repetition stay intact across months.
    An external predecessor is refused even if its JSON and digest are valid.
    ResearchMemory's own resolver still verifies actual S evidence when used.
    """
    research = Path(research)
    if not research.is_absolute():
        research = project_path('.')/research
    _no_redirect(research, project_path('.'))
    database = research/'state.sqlite'
    if not database.exists():
        if research.exists() and any(research.iterdir()):
            raise TrialIsolationError('Research namespace exists without its original journal')
        return {'months': 0, 'memory_shards': 0}
    _no_redirect(database, research)
    manifest = _read(_no_redirect(research/'manifest.json', research))
    if set(manifest) != {'identity', 'hash'} or fingerprint(manifest['identity']) != manifest['hash']:
        raise TrialIsolationError('Original trial RunStore identity changed')
    expected_run = fingerprint(manifest['identity'])
    seen = set()

    def check_value(value, kind):
        if value.get('version') != 1:
            raise TrialIsolationError('Unknown research memory context version')
        if kind == 'memory':
            for key, ref in value['evidence'].items():
                if fingerprint(ref) != key or ref['run_identity'] != expected_run:
                    raise TrialIsolationError('Research evidence belongs to another trial RunStore')
            check_chain(value['book'].get('predecessor'), 'book')
        check_chain(value.get('predecessor'), kind)

    def check_chain(link, kind):
        local_seen = set()
        while link is not None:
            if type(link) is not dict or set(link) != {'path', 'digest'}:
                raise TrialIsolationError('Invalid original memory predecessor binding')
            # project_path resolves redirects, so validate the original absolute
            # spelling before using any read or the normal memory loader.
            path = _no_redirect(Path(link['path']), research)
            if path in local_seen or len(local_seen) >= MAX_SHARDS:
                raise TrialIsolationError('Cyclic or oversized trial memory chain')
            local_seen.add(path)
            token = (str(path), kind)
            wrapper = _read(path)
            value = wrapper[kind]
            if fingerprint(value) != wrapper.get('digest') or wrapper['digest'] != link['digest']:
                raise TrialIsolationError('Trial memory predecessor digest changed')
            if token not in seen:
                seen.add(token)
                # The iterative walk owns this chain; only the OTHER book chain
                # is traversed here, so a malicious self-link cannot recurse.
                if kind == 'memory':
                    for key, ref in value['evidence'].items():
                        if fingerprint(ref) != key or ref['run_identity'] != expected_run:
                            raise TrialIsolationError('Archived evidence belongs to another trial')
                    check_chain(value['book'].get('predecessor'), 'book')
            link = value.get('predecessor')

    if through_month is not None and (type(through_month) is not str
            or not re.fullmatch(r'[0-9]{4}-(0[1-9]|1[0-2])', through_month)):
        raise TrialIsolationError('Invalid original context readback month')
    with closing(sqlite3.connect(database.as_uri()+'?mode=ro', uri=True)) as db:
        rows = (db.execute('SELECT month,value,hash FROM months ORDER BY month').fetchall()
                if through_month is None else db.execute(
                    'SELECT month,value,hash FROM months WHERE month<=? ORDER BY month',
                    (through_month,)).fetchall())
    for _, text, digest in rows:
        state = json.loads(text)
        if fingerprint(state) != digest:
            raise TrialIsolationError('Original trial month index changed')
        memory = state.get('research_memory')
        if memory is not None:
            if fingerprint(memory['memory']) != memory.get('digest'):
                raise TrialIsolationError('Original trial memory digest changed')
            check_value(memory['memory'], 'memory')
            # This is deliberately after every predecessor path was checked.
            from .v7_memory import ResearchMemory
            ResearchMemory.from_state(memory)
    return {'months': len(rows), 'memory_shards': len(seen)}


def verify_memory_namespace(research, *, through_month=None):
    try:
        return _verify_memory_namespace(research, through_month=through_month)
    except TrialIsolationError:
        raise
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
        raise TrialIsolationError('Invalid original trial research context') from exc


class TrialScopedCodexBackend:
    """One trial-owned wrapper around the actual restricted Codex adapter.

    No callbacks/backends are accepted; the normal adapter sends only the
    controller-built packet in an ephemeral, tool-restricted CLI turn. Usage and
    provider resolution remain the original measured values, including unknown.
    """
    def __init__(self, directory, trial, contract_hash):
        self._set_identity(directory, trial, contract_hash)
        if not self.path.exists() and self.paths['provider'].exists() and any(self.paths['provider'].iterdir()):
            raise TrialIsolationError('Provider context missing; cannot reuse unbound attempt history')
        _no_redirect(self.path, self.paths['provider'])
        immutable_record(self.path, {'value': self.identity, 'sha256': fingerprint(self.identity)})
        self.verify()

    def _set_identity(self, directory, trial, contract_hash):
        if type(contract_hash) is not str or not re.fullmatch('[0-9a-f]{64}', contract_hash):
            raise TrialIsolationError('Invalid original coverage contract digest')
        self.paths = trial_paths(directory, trial)
        self.identity = {
            'version': POLICY, 'trial': trial, 'coverage_contract_hash': contract_hash,
            'paths': {k: str(v) for k, v in self.paths.items()},
            'model_requested': DEFAULT_MODEL, 'reasoning_requested': DEFAULT_REASONING,
            'session_policy': 'ephemeral_no_resume_no_user_rules_no_host_memory',
            'disabled_features': list(DISABLED_FEATURES), 'overrides': list(RESTRICTED_OVERRIDES),
            'provider_authenticity_certified': False, 'statistical_independence_certified': False,
        }
        self.path = self.paths['provider']/'context.json'

    @classmethod
    def inspect_existing(cls, directory, trial, contract_hash, *, through_month=None):
        # Do not call the constructor: missing metadata must not be repaired by
        # historical inspection or completion recovery.
        result = cls.__new__(cls)
        result._set_identity(directory, trial, contract_hash)
        memory = result.verify(through_month=through_month)
        return {'policy': POLICY, 'context_hash': fingerprint(result.identity),
                'memory': memory, 'provider_authenticity_certified': False,
                'statistical_independence_certified': False}

    def verify(self, *, through_month=None):
        for path in self.paths.values():
            _no_redirect(path, project_path(self.paths['provider'].parents[1]))
        _no_redirect(self.path, self.paths['provider'])
        _no_redirect(self.paths['provider']/'attempts', self.paths['provider'])
        if _read(self.path) != {'value': self.identity, 'sha256': fingerprint(self.identity)}:
            raise TrialIsolationError('Original trial provider context changed')
        return verify_memory_namespace(self.paths['research'], through_month=through_month)

    def __call__(self, packet, *, model, reasoning):
        """Diagnostic adapter only; actual S uses call_reserved's original work."""
        if (model, reasoning) != (DEFAULT_MODEL, DEFAULT_REASONING):
            raise TrialIsolationError('Trial provider identity changed')
        self.verify()
        backend = CodexBatchBackend(self.paths['provider']/'attempts')
        reply = backend(packet, model=model, reasoning=reasoning)
        self.verify()
        return reply

    def call_reserved(self, packet, *, model, reasoning, work_budget):
        """Never restart time: consume only the handed actual S family lease."""
        from .work_authority_bridge import AuthorityWorkBudget
        if (type(work_budget) is not AuthorityWorkBudget
                or work_budget.bridge._active is not work_budget
                or work_budget.bridge.store.directory != self.paths['research']
                or work_budget.bridge.authority.directory != self.paths['authority']
                or work_budget.record['binding']['work']['stage'] != 'family'):
            raise TrialIsolationError('Trial provider requires its original active shared S family work')
        if (model, reasoning) != (DEFAULT_MODEL, DEFAULT_REASONING):
            raise TrialIsolationError('Trial provider identity changed')
        work_budget.check_time()
        self.verify()

        class ReservedBackend(CodexBatchBackend):
            def _run(self, command, text, runtime):
                # Packet/schema setup and prior failed attempts already consumed
                # this same original lease. Recompute immediately before child
                # execution, never from a new construction/start timestamp.
                work_budget.check_time()
                self.timeout_seconds = min(180., work_budget.lease.assert_parent_alive().remaining_seconds)
                return super()._run(command, text, runtime)

        remaining = work_budget.lease.assert_parent_alive().remaining_seconds
        backend = ReservedBackend(self.paths['provider']/'attempts', timeout_seconds=min(180., remaining))
        reply = backend(packet, model=model, reasoning=reasoning)
        work_budget.check_time()
        self.verify()
        work_budget.check_time()
        return reply
