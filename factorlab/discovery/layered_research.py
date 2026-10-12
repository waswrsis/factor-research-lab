"""Explicit research-only lifecycle adapter, never an old calibration certificate.

The immutable new enrollment permits prospective C *measurement*. Numerical
success may retain a formula for subsequent research fits; it cannot certify
market alpha, whole-agent power, or trading eligibility. Legacy runs keep their
original calibration gate and are never silently converted to this mode.
"""
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from factorlab.protocol import fingerprint

from .confirmatory import ConfirmationRule
from .v7_lifecycle import Lifecycle, stamp
from .v7_run import source_files
from .v7_storage import immutable_record

VERSION = 'v7-layered-research-measurement-v1'


def research_retention_decision(lifecycle, result):
    """Typed adapter result, never an authority supplied by a caller boolean."""
    spec = verified_research_spec(lifecycle)
    if spec is None:
        return None
    return _assessment(spec, result)


def _identity(lifecycle, spec):
    return {'version': VERSION, 'plan_id': lifecycle.plan.digest,
            'spec': asdict(spec), 'spec_hash': spec.digest,
            'source_hash': fingerprint(source_files()),
            'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'formal_gate_eligible': False, 'whole_agent_power_certified': False,
            'purpose': 'measurement_and_unconfirmed_research_retention_only'}


def verified_research_spec(lifecycle):
    if isinstance(lifecycle, LayeredResearchLifecycle):
        lifecycle.verify_research_enrollment()
        lifecycle._verify_research_manifest()
        return lifecycle.research_spec
    from .neural_forward_ticket import _LifecycleReader
    if type(lifecycle) is not _LifecycleReader:
        return None
    from .calibration_runtime_contract import _original_operations
    operations = _original_operations(lifecycle)
    events = [o for o in operations if o['kind'] == 'layered_research_enrolled']
    path = lifecycle.directory / 'layered-research.json'
    if not events and not path.exists():
        return None
    if len(events) != 1 or events[0]['seq'] != 1:
        raise ValueError('Readonly research scope lost its original enrollment')
    spec = LayeredResearchSpec(**events[0]['detail']['spec'])
    if (events[0]['detail'] != _identity(lifecycle, spec)
            or not path.is_file() or json.loads(path.read_text('utf-8')) != events[0]['detail']
            or stamp(events[0]['at']) > lifecycle.now()
            or (lifecycle.plan.training_window_days, lifecycle.plan.refit_interval_days) != (90, 15)
            or lifecycle.plan.calibration_directory is not None
            or lifecycle.plan.calibration_binding is not None):
        raise ValueError('Readonly research scope changed or mixed legacy certification')
    return spec


def _assessment(spec, result):
    if (result.get('formal_eligible') is not False
            or result.get('calibration_passed') is not False
            or result.get('supported') is not False
            or result.get('rule_id') != ConfirmationRule().digest
            or result.get('method') != 'hour_equal_gap_split_calendar_block_percentile_bootstrap'
            or result.get('quarantined_decision') is not None):
        raise ValueError('Layered diagnostics cannot carry formal or quarantine authority')
    interval = result.get('interval')
    if result.get('diagnostic_status') == 'supported_target' and (
            type(interval) is not list or len(interval) != 2
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in interval)
            or not 0 < interval[0] <= interval[1]):
        raise ValueError('Layered retention needs the original positive numerical interval')
    return {'version': VERSION, 'research_spec_hash': spec.digest,
            'status': ('eligible_for_unconfirmed_research_retention'
                       if result.get('diagnostic_status') == 'supported_target'
                       else 'not_supported_by_this_endpoint'),
            'numerical_diagnostic': result.get('diagnostic_status'),
            'formal_gate_eligible': False, 'market_alpha_certified': False,
            'whole_agent_power_certified': False}


@dataclass(frozen=True)
class LayeredResearchSpec:
    source_hash: str
    mode: str = 'prospective_research_diagnostics'
    statistical_claim: str = 'no_whole_agent_population_power_certificate'
    minimum_gain: float = .005
    confirmation_months: int = 3
    algorithm_months: int = 6
    training_window_days: int = 90
    refit_interval_days: int = 15

    def __post_init__(self):
        if (type(self.source_hash) is not str or len(self.source_hash) != 64
                or any(c not in '0123456789abcdef' for c in self.source_hash)
                or self.mode != 'prospective_research_diagnostics'
                or self.statistical_claim != 'no_whole_agent_population_power_certificate'
                or type(self.minimum_gain) is not float or self.minimum_gain != .005
                or self.confirmation_months != 3 or self.algorithm_months != 6
                or self.training_window_days != 90 or self.refit_interval_days != 15
                or any(type(v) is not int for v in (self.confirmation_months, self.algorithm_months,
                                                    self.training_window_days, self.refit_interval_days))):
            raise ValueError('Invalid layered research protocol; old gates cannot be relaxed')

    @property
    def digest(self):
        return fingerprint({'version': VERSION, 'spec': asdict(self),
                            'rule': ConfirmationRule().to_dict()})


class LayeredResearchLifecycle(Lifecycle):
    """A separate, prospectively enrolled research lifecycle, not production trading."""

    def __init__(self, directory, plan, *, research_spec, clock=None):
        if (not isinstance(research_spec, LayeredResearchSpec)
                or plan.calibration_directory is not None or plan.calibration_binding is not None
                or (plan.training_window_days, plan.refit_interval_days) != (90, 15)
                or plan.confirmation_months != 3 or plan.algorithm_months != 6
                or fingerprint(source_files()) != research_spec.source_hash):
            raise ValueError('Layered enrollment needs its own source, elapsed plan and no legacy certificate')
        self.research_spec = research_spec
        super().__init__(directory, plan, clock=clock)
        try:
            enrolled = self._research_events()
            identity = self._research_identity()
            if not enrolled:
                # Never retrofit exposed jobs or an old lifecycle, even if no C
                # result has yet been published. Feed binding may be done later.
                count = sum(self.db.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
                            for table in ('jobs', 'batches', 'settlement', 'operations'))
                if count or (self.directory / 'layered-research.json').exists():
                    raise ValueError('Layered research requires a fresh lifecycle')
                with self.db:
                    self._event('layered_research_enrolled', identity)
            else:
                # Reopening is verification, never repair. A lost manifest
                # must not be silently reconstructed from the journal.
                self._verify_research_manifest()
            self.verify_research_enrollment()
            immutable_record(self.directory / 'layered-research.json', identity)
        except BaseException:
            self.close()
            raise

    def _research_identity(self):
        return _identity(self, self.research_spec)

    def _research_events(self):
        return [json.loads(row[0]) for row in self.db.execute('SELECT record FROM operations ORDER BY seq')
                if json.loads(row[0])['kind'] == 'layered_research_enrolled']

    def verify_research_enrollment(self):
        events = self._research_events()
        if (len(events) != 1 or events[0]['seq'] != 1
                or events[0]['detail'] != self._research_identity()
                or stamp(events[0]['at']) > self.now()):
            raise ValueError('Layered research lost its original first enrollment')
        return events[0]['detail']

    def calibration_ready(self):
        self.verify_research_enrollment()
        return False  # This new mode never impersonates the legacy full certificate.

    def verify(self):
        result = super().verify()
        self.verify_research_enrollment()
        self._verify_research_manifest()
        return result

    def verify_serving(self):
        result = super().verify_serving()
        self.verify_research_enrollment()
        self._verify_research_manifest()
        return result

    def _verify_research_manifest(self):
        path = self.directory / 'layered-research.json'
        if not path.is_file() or json.loads(path.read_text('utf-8')) != self._research_identity():
            raise ValueError('Layered research manifest missing or changed')

    def confirmation_execution_ready(self):
        self.verify_research_enrollment()
        return True  # Allows the original one-look C measurement, not support.

    def research_assessment(self, result):
        self.verify_research_enrollment()
        return _assessment(self.research_spec, result)

    def adoption_pipeline(self, cutoff, k0):
        """Retain a measured formula prospectively, with no algorithm certificate."""
        self.verify()
        self.verify_research_enrollment()
        cutoff = self.fit_cutoff(cutoff)
        if cutoff > self.now() or not k0.get('pipeline_id'):
            raise ValueError('Research retention requires an actual cutoff and K0')
        from .claim_protocol import ClaimProtocol
        if ClaimProtocol._bindings(self) or (self.directory / 'claim_protocol/protocol.json').exists():
            from .typed_policy_confirmation import candidate_pipeline
            return candidate_pipeline(self, cutoff, k0)
        for batch, text, raw in self.db.execute(
                'SELECT id,identity,result FROM batches WHERE status=? ORDER BY ordinal DESC', ('concluded',)):
            result = json.loads(raw)
            assessment = self.research_assessment(result)
            if (assessment['status'] != 'eligible_for_unconfirmed_research_retention'
                    or stamp(result['completed_at']) > cutoff):
                continue
            publication = self.confirmation_publication(batch)
            path = self.directory / 'claim_publications' / f'{fingerprint(batch)}.json'
            if stamp(publication['published_at']) > cutoff or path.stat().st_mtime_ns > cutoff.value:
                continue
            identity = json.loads(text)
            return next(c for c in identity['candidates'] if c['pipeline_id'] == result['pipeline_id'])
        return k0
