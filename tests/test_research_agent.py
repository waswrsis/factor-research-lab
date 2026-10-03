from dataclasses import replace

import pytest

from factorlab.protocol import Protocol
from factorlab.research_agent import diagnostic_proposal


def test_agent_cannot_change_protocol_or_see_future():
    p = Protocol()
    proposal = diagnostic_proposal('parent', p, '2025-03-01', [], 0)
    proposal.validate(p, '2025-03-01', 'parent')
    for bad in (replace(proposal, horizon_hours=24),
                replace(proposal, operation='execute_python'),
                replace(proposal, evidence_cutoff='2025-04-01'),
                replace(proposal, protocol_hash='changed')):
        with pytest.raises(ValueError):
            bad.validate(p, '2025-03-01', 'parent')
    with pytest.raises(ValueError):
        diagnostic_proposal('parent', p, '2025-03-01',
                            [{'partition': 'test', 'utility': 99}], 0)
