"""Auditable proposals. Agents may edit model choices, never the evaluation protocol."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .protocol import Protocol, fingerprint, utc

OPERATIONS = ('add_expert', 'drop_expert', 'width', 'depth', 'scales', 'gate',
              'loss', 'optimizer', 'learning_rate', 'add_feature', 'drop_feature',
              'add_interaction', 'drop_interaction')


@dataclass(frozen=True)
class Proposal:
    parent_id: str
    protocol_hash: str
    evidence_cutoff: str
    operation: str
    seed: int
    hypothesis: str
    falsification: str
    source: str
    expected_unit: str = 'simple_return'
    horizon_hours: int = 12
    arguments: dict = field(default_factory=dict)

    def validate(self, protocol: Protocol, cutoff: str, parent_id: str) -> None:
        if not all(isinstance(x, str) for x in (self.parent_id, self.protocol_hash,
                  self.evidence_cutoff, self.operation, self.hypothesis, self.falsification,
                  self.source, self.expected_unit)):
            raise ValueError('Proposal text fields must be strings')
        if not isinstance(self.arguments, dict) or set(self.arguments) - {
            'expert', 'value', 'name', 'kind', 'inputs', 'scales'
        }:
            raise ValueError('Only bounded structural arguments may be supplied')
        try:
            json.dumps(self.arguments, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError('Structural arguments must be finite JSON values') from exc
        if self.protocol_hash != protocol.digest or self.parent_id != parent_id:
            raise ValueError('Proposal protocol/parent mismatch')
        if utc(self.evidence_cutoff) > utc(cutoff):
            raise ValueError('Proposal uses evidence after the research cutoff')
        if self.operation not in OPERATIONS:
            raise ValueError('Unregistered structure operation')
        if self.horizon_hours != protocol.horizon_hours or self.expected_unit != 'simple_return':
            raise ValueError('Proposal cannot alter target contract')
        if not self.hypothesis.strip() or not self.falsification.strip():
            raise ValueError('A testable hypothesis and falsification rule are required')
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError('Invalid proposal seed')


def diagnostic_proposal(parent_id: str, protocol: Protocol, cutoff: str,
                        history: list[dict], iteration: int) -> Proposal:
    """A transparent local rule agent; never presented as an LLM run.

    Failure and mature validation evidence choose a hypothesis. Confirmation
    and outer-test scores must never appear in this history.
    """
    for item in history:
        if not item.get('evidence_cutoff') or item.get('partition') != 'search':
            raise ValueError('Every consumed memory item requires search partition and cutoff')
        if utc(item['evidence_cutoff']) > utc(cutoff):
            raise ValueError('Future research memory is not permitted')
    if history and history[-1].get('status') == 'failed':
        operation, hypothesis = 'loss', 'Robust residual loss may reduce numerical instability.'
    elif history and history[-1].get('alpha', 1) < 0.05:
        operation, hypothesis = 'drop_expert', 'A rejected residual may benefit from less complexity.'
    elif iteration % 4 == 0:
        operation, hypothesis = 'add_interaction', 'A bounded nonlinear interaction may explain residuals.'
    elif iteration % 4 == 1:
        operation, hypothesis = 'scales', 'Changing temporal scales may improve mature search errors.'
    elif iteration % 4 == 2:
        operation, hypothesis = 'gate', 'The alternative gate may reduce state-dependent error.'
    else:
        operation, hypothesis = 'add_expert', 'An additional specialist may provide complementary signal.'
    return Proposal(parent_id, protocol.digest, cutoff, operation, 1000+iteration,
                    hypothesis, 'Reject if net search utility does not improve after costs.',
                    'local_diagnostic_rules')


def read_proposal(path: str | Path, protocol: Protocol, cutoff: str,
                  parent_id: str) -> Proposal:
    # Constructor rejects unknown fields: no code, URLs, protocol overrides or
    # instructions are executed by importing a proposal.
    try:
        value = json.loads(Path(path).read_text('utf-8-sig'))
        proposal = Proposal(**value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError('Malformed proposal JSON/schema') from exc
    proposal.validate(protocol, cutoff, parent_id)
    return proposal


def proposal_id(proposal: Proposal) -> str:
    return fingerprint(asdict(proposal))[:16]
