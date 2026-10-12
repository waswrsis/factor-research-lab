"""Real paid cold child; synthetic law and explicit private selection/kernel seams."""
import json
import multiprocessing
import os
from pathlib import Path

import pandas as pd
import pytest

from factorlab.protocol import fingerprint
from factorlab.discovery import conditional_cold_refit as m
from factorlab.discovery import neural_research_launcher as launcher
from factorlab.discovery.calibration_acceptance import AcceptanceCell, AcceptancePlan
from factorlab.discovery.calibration_runtime_contract import runtime_calibration_contract
from factorlab.discovery.finite_market_oracle import FiniteGenerator, FiniteMarketLaw, PROOF_FILES, _read
from factorlab.discovery.full_calibration_registry import FullTrialPlan, FullTrialRegistry
from factorlab.discovery.joint_pool import FrozenPool
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
from factorlab.discovery.neural_forward import FrozenNeuralRecipe
from factorlab.discovery.neural_forward_ticket import _bind_authority
from factorlab.discovery.v7_forward import ForwardConfig, ForwardRuntime
from test_v7_neural_research_ticket import _synthetic_kernel


def private_selection(permit, slot):
    # Explicit engineering seam; no actual selected-C or neural V claim.
    return json.loads((permit.registry.directory.parent/'selection.json').read_text())


def child(root, directory, atom):
    import factorlab.protocol
    factorlab.protocol.ROOT = Path(root)
    m._selected = private_selection
    launcher._kernel_snapshot = lambda: _synthetic_kernel([os.getppid(), os.getpid()])
    m.worker(directory, atom)


@pytest.fixture(scope='module')
def case(tmp_path_factory):
    root = tmp_path_factory.mktemp('cr')
    patch = pytest.MonkeyPatch()
    import factorlab.protocol
    original_root = factorlab.protocol.ROOT
    (root/'factorlab').mkdir()
    # Actual proof source files, tiny scope explicitly limited to engineering.
    for name in PROOF_FILES:
        target = root/name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((original_root/name).read_bytes())
    (root/'pyproject.toml').write_text('[project]')
    (root/'uv.lock').write_text('version=1')
    patch.setattr(factorlab.protocol, 'ROOT', root)
    patch.setattr(m, '_selected', private_selection)
    law = FiniteMarketLaw('2025-12-01T00:00:00Z', hours=2180, support_size=128, gap_probability=0.)
    acceptance = AcceptancePlan((AcceptanceCell('cold', law.digest, 1000),))
    generator = FiniteGenerator(root/'g', law, acceptance)
    profile = ForwardRuntime(root/'p', root/'f', config=ForwardConfig())
    wallet = MonthlyWorkAuthority(root/'w', run_id='profile', task='volatility', universe=['BTCUSDT'], partition_policy='standard')
    _bind_authority(profile, wallet)
    scope = runtime_calibration_contract(profile.lifecycle)
    registry = FullTrialRegistry(root/'r', FullTrialPlan(acceptance, scope,
        {'backend':'codex','model_requested':'gpt-6-astra','reasoning_requested':'high','context_policy':'fresh_codex_per_trial_v1'}, law.domains))
    opened = [profile]
    def enroll(atom=None):
        paths = registry.paths('cold', 0, oracle_index=atom)
        runtime = ForwardRuntime(paths['runtime'], paths['feed']); opened.append(runtime)
        trial = registry.identity_for('cold', 0, oracle_index=atom)
        authority = MonthlyWorkAuthority(paths['authority'], run_id=fingerprint({'registry':registry.digest,'trial':trial['trial_id']}),
            task='volatility', universe=['BTCUSDT'], partition_policy='standard')
        return registry.enroll(runtime, authority, 'cold', 0, oracle_index=atom), runtime, authority
    validation, runtime, _ = enroll()
    selection = {'original_estimand': {'private':'artificial original-selection seam'},
        'original_estimand_digest': fingerprint('engineering'),
        'start':'2026-02-01T00:00:00+00:00','end':'2026-03-01T00:00:00+00:00',
        'reference':runtime.recipe_for(FrozenPool()),
        'candidate':runtime.recipe_for(FrozenNeuralRecipe(rolling_spec=runtime.spec))}
    (root/'selection.json').write_text(json.dumps(selection))
    frozen = m.freeze_cold_refit(validation, generator, 'factor1')
    children = []
    class ActualChild:
        def __init__(self, command, **kwargs):
            process = multiprocessing.get_context('spawn').Process(target=child, args=(str(root), command[-2], int(command[-1])))
            process.start(); self.process=process; self.pid=process.pid; children.append(self)
        @property
        def returncode(self):
            return self.process.exitcode
        def poll(self):
            return self.process.exitcode
    patch.setattr(m.subprocess, 'Popen', ActualChild)
    patch.setattr(launcher, '_kernel_snapshot', lambda: _synthetic_kernel([os.getpid()]+[p.pid for p in children if p.poll() is None]))
    patch.setattr(launcher, '_strict_kernel', lambda *args: None)
    yield root, registry, generator, validation, enroll, frozen
    for runtime in opened:
        runtime.close()
    patch.undo()


def test_real_cold_neural_and_pool_same_paid_path_full_128_denominator(case):
    root, registry, _, _, enroll, frozen = case
    _, _, authority = enroll(0)
    result = m.run_cold_refit(frozen['directory'], 0)
    assert result['risk']['defined'] and result['risk']['expected_pairs'] == 672
    assert result['actual_counts']['neural_fit'] == 3 and result['actual_counts']['epoch'] == 12
    assert result['actual_counts']['solve'] >= 5
    report = m.report_cold_refit(frozen['directory'])
    assert report['registered_paths'] == 128 and report['completed_paths'] == 1
    assert report['missing_paths'] == list(range(1,128)) and report['conditional_risk_envelope'] is None
    before = authority.verify()
    assert m.run_cold_refit(frozen['directory'], 0) == result
    assert authority.verify() == before
    assert not result['formal_gate_eligible'] and not report['original_claim_scope_eligible']
    unit = Path(frozen['directory'])/'paths/0'
    # Tamper with a real model, even while preserving JSON validity.
    path = unit/'candidate.json'; raw=path.read_bytes(); path.write_bytes(raw+b' ')
    with pytest.raises(ValueError, match='numeric artifact changed'):
        m.report_cold_refit(frozen['directory'])
    path.write_bytes(raw)
    # Missing final IO does not recreate success or allow a fresh work clock.
    final = unit/'final_io.json'; raw=final.read_bytes(); final.unlink()
    with pytest.raises(ValueError, match='Missing or oversized'):
        m.run_cold_refit(frozen['directory'], 0)
    final.write_bytes(raw)
    assert authority.verify() == before
    (root/'numeric-evidence.json').write_text(json.dumps({'result':result,'report':report},indent=2))


def test_late_actual_final_file_cannot_publish_or_retry(case, monkeypatch):
    _, _, _, _, enroll, frozen = case
    _, _, authority = enroll(1)
    original = m._write
    def late(path, value):
        original(path, value)
        if path.name == 'final_io.json':
            intent = _read(path.parent/'intent.json')
            deadline = pd.Timestamp(intent['lease']['wall_deadline']).value+1_000_000_000
            os.utime(path, ns=(deadline,deadline))
    monkeypatch.setattr(m, '_write', late)
    with pytest.raises(ValueError, match='final IO passed'):
        m.run_cold_refit(frozen['directory'], 1)
    before = authority.verify()
    with pytest.raises(ValueError, match='already consumed'):
        m.run_cold_refit(frozen['directory'], 1)
    report = m.report_cold_refit(frozen['directory'])
    assert 1 in report['failed_or_undefined_paths'] and report['conditional_risk_envelope'] is None
    assert report['registered_paths'] == 128 and authority.verify() == before


def test_changed_original_selection_and_shrunken_support_rejected(case):
    root, _, _, _, _, frozen = case
    selection = root/'selection.json'; raw=selection.read_bytes()
    changed=json.loads(raw); changed['candidate']=changed['reference']; selection.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='registry/selection/source changed'):
        m.report_cold_refit(frozen['directory'])
    selection.write_bytes(raw)
    path=Path(frozen['directory'])/'identity.json'; raw=path.read_bytes()
    identity=_read(path); identity['support']=[0]
    path.write_text(json.dumps({'record':identity,'hash':fingerprint(identity)}))
    with pytest.raises(ValueError):
        m.report_cold_refit(frozen['directory'])
    path.write_bytes(raw)
