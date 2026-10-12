"""Real runtime/wallet scope checks on empty artificial archives, no GPT.

Report injection in two tests is a gate plumbing seam, NOT a valid full-scope
certificate producer or evidence that any statistical target has passed.
"""
import json
from dataclasses import replace

import pytest
from test_v7_run_factor_pool import source_root

from factorlab.discovery.calibration_runtime_contract import (
    RuntimeScopeUnavailable,
    runtime_calibration_contract,
    runtime_calibration_digest,
)
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
from factorlab.discovery.neural_forward_ticket import _bind_authority
from factorlab.discovery.rolling import RollingSpec
from factorlab.discovery.v7_feed import ReceiptArchive
from factorlab.discovery.v7_forward import ForwardConfig, ForwardRuntime
from factorlab.discovery.v7_lifecycle import Lifecycle, LifecyclePlan
from factorlab.protocol import fingerprint


@pytest.fixture
def bound(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    runtime = ForwardRuntime(tmp_path / 'runtime', tmp_path / 'feed')
    wallet = MonthlyWorkAuthority(tmp_path / 'wallet', run_id='runtime-scope-test',
                                 task='volatility', universe=['BTCUSDT'])
    _bind_authority(runtime, wallet)
    try:
        yield runtime, wallet
    finally:
        runtime.close()


def overwrite(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding='utf-8')


def test_actual_availability_delay_and_wallet_contract_readback_no_debits(bound):
    runtime, wallet = bound
    before = wallet.verify()
    actual = runtime_calibration_contract(runtime.lifecycle)
    assert actual['rolling_hash'] == runtime.spec.digest
    assert actual['rolling_hash'] != RollingSpec().digest
    assert actual['rolling']['availability_delay_seconds'] == 5
    assert actual['forward']['archive_delay_seconds'] == 5
    assert actual['work_budget']['limits'] == wallet.limits
    assert actual['work_budget']['monthly_seconds'] == 7200
    assert actual['work_budget']['partition_policy'] is None
    assert actual['scope'] == 'actual_runtime_component_only'
    assert 'formal_gate_eligible' not in actual
    assert runtime_calibration_digest(runtime.lifecycle) == fingerprint(actual)
    assert runtime_calibration_contract(runtime.lifecycle) == actual
    assert wallet.verify() == before
    assert before['reservations'] == 0


def test_two_actual_delays_require_different_scopes(bound):
    runtime, wallet = bound
    other = ForwardRuntime(runtime.directory.parent / 'other-runtime', runtime.feed.directory,
                           config=ForwardConfig(archive_delay_seconds=19))
    try:
        _bind_authority(other, wallet)
        first, second = (runtime_calibration_contract(v.lifecycle) for v in (runtime, other))
        assert first['rolling_hash'] != second['rolling_hash']
        assert fingerprint(first) != fingerprint(second)
        assert wallet.verify()['reservations'] == 0
    finally:
        other.close()


def test_operational_certificate_names_do_not_create_self_referential_scope(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    first = ForwardRuntime(tmp_path/'initial', tmp_path/'feed')
    source = first.feed.identity
    first.close()
    wallet = MonthlyWorkAuthority(tmp_path/'wallet', run_id='scope-circularity',
                                 task='volatility', universe=['BTCUSDT'])
    runtimes = []
    try:
        for name in ('a', 'b'):
            runtime = ForwardRuntime(tmp_path/name, tmp_path/'feed',
                lifecycle_plan=LifecyclePlan(source, calibration_directory=str(tmp_path/name/'calibration'),
                                              calibration_binding=fingerprint(name)))
            _bind_authority(runtime, wallet)
            runtimes.append(runtime)
        assert runtime_calibration_contract(runtimes[0].lifecycle) == runtime_calibration_contract(runtimes[1].lifecycle)
    finally:
        for runtime in runtimes:
            runtime.close()


@pytest.mark.parametrize('kind', ['rehashed_forward', 'source_archive', 'source_manifest',
                                 'feed_identity', 'wallet_mirror', 'wallet_manifest', 'lifecycle_plan',
                                 'operation_hash', 'missing_runtime_binding'])
def test_original_binding_and_actual_stores_cannot_be_replaced(bound, kind):
    runtime, wallet = bound
    root = runtime.directory
    if kind == 'rehashed_forward':
        path = root/'forward.json'
        wrapper = json.loads(path.read_text())
        wrapper['identity']['config']['archive_delay_seconds'] = 1
        wrapper['identity']['rolling']['availability_delay_seconds'] = 1
        wrapper['sha256'] = fingerprint(wrapper['identity'])
        overwrite(path, wrapper)
    elif kind == 'source_archive':
        (root/'source/factorlab/v7.py').write_text('version=8', encoding='utf-8')
    elif kind == 'source_manifest':
        overwrite(root/'source/manifest.json', {})
    elif kind == 'feed_identity':
        overwrite(runtime.feed.directory/'identity.json', {'source_id': 'replacement', 'symbols': ['BTCUSDT']})
    elif kind == 'wallet_mirror':
        (root/'work_authority.json').unlink()
    elif kind == 'wallet_manifest':
        path = wallet.directory/'manifest.json'
        wrapper = json.loads(path.read_text())
        wrapper['record']['monthly_seconds'] = 7199
        wrapper['hash'] = fingerprint(wrapper['record'])
        overwrite(path, wrapper)
    elif kind == 'lifecycle_plan':
        plan = replace(runtime.lifecycle.plan, prediction_max_lag_seconds=299)
        overwrite(root/'lifecycle.json', {'plan': vars(plan), 'sha256': plan.digest})
    else:
        with runtime.lifecycle.db:
            if kind == 'operation_hash':
                runtime.lifecycle.db.execute("UPDATE operations SET hash=? WHERE seq=1", ('f'*64,))
            else:
                runtime.lifecycle.db.execute('DELETE FROM operations WHERE seq=1')
    with pytest.raises((ValueError, OSError)):
        runtime_calibration_contract(runtime.lifecycle)
    if kind == 'wallet_mirror':
        assert not (root/'work_authority.json').exists()


def test_offline_and_unbound_runtime_do_not_receive_default_production_scope(tmp_path, monkeypatch):
    source_root(tmp_path, monkeypatch)
    offline = Lifecycle(tmp_path/'offline', LifecyclePlan('artificial-source'))
    runtime = ForwardRuntime(tmp_path/'runtime', tmp_path/'feed')
    try:
        with pytest.raises(RuntimeScopeUnavailable, match='Offline lifecycle'):
            runtime_calibration_contract(offline)
        with pytest.raises(RuntimeScopeUnavailable, match='bound shared'):
            runtime_calibration_contract(runtime.lifecycle)
        assert not (runtime.directory/'work_authority.json').exists()
        assert not (tmp_path/'wallet').exists()
    finally:
        offline.close()
        runtime.close()


def report_seam(runtime, monkeypatch):
    actual = runtime_calibration_contract(runtime.lifecycle)
    report = {'formal_gate_eligible': True, 'full_V7_five_claim_pipeline_calibrated': True,
              'covered_claim_types': ['library_skill', 'factor1', 'factor2', 'factor3', 'algorithm'],
              'runtime_scope_hash': fingerprint(actual), 'rolling_hash': actual['rolling_hash'],
              'rule_hash': actual['factor_confirmation_rule_hash'], 'test_only': False}
    # Only the verifier return is faked here. The real runtime, source, binding
    # chain and wallet are still read. This is not certificate authenticity.
    monkeypatch.setattr('factorlab.discovery.pipeline_calibration.verify_calibration_certificate',
                        lambda *a, **k: report)
    runtime.lifecycle.plan = replace(runtime.lifecycle.plan, calibration_directory='seam', calibration_binding='seam')
    # Restore the actual plan/manifest consistency for this explicit gate seam.
    for path, key in [(runtime.directory/'lifecycle.json', 'plan'), (runtime.directory/'forward.json', None)]:
        value = json.loads(path.read_text())
        if key:
            value[key].update(calibration_directory='seam', calibration_binding='seam')
            value['sha256'] = runtime.lifecycle.plan.digest
        else:
            value['identity']['lifecycle'].update(calibration_directory='seam', calibration_binding='seam')
            value['sha256'] = fingerprint(value['identity'])
        overwrite(path, value)
    # Deliberately do not re-anchor replaced identities. A flipped report must
    # be rejected even when every advertised eligibility flag is true.
    return report


def test_full_scope_flags_cannot_bypass_original_runtime_binding(bound, monkeypatch):
    runtime, _ = bound
    report_seam(runtime, monkeypatch)
    with pytest.raises(ValueError, match='Original runtime binding'):
        runtime.lifecycle.calibration_ready()


def test_local_scope_is_refused_before_any_production_scope_read(bound, monkeypatch):
    runtime, _ = bound
    runtime.lifecycle.plan = replace(runtime.lifecycle.plan, calibration_directory='seam', calibration_binding='seam')
    monkeypatch.setattr('factorlab.discovery.pipeline_calibration.verify_calibration_certificate',
                        lambda *a, **k: {'formal_gate_eligible': False})
    monkeypatch.setattr('factorlab.discovery.calibration_runtime_contract.runtime_calibration_contract',
                        lambda *a: pytest.fail('Local report must not become full scope'))
    assert runtime.lifecycle.calibration_ready() is False


@pytest.mark.parametrize('change', ['none', 'default_delay', 'different_scope', 'missing_scope',
                                   'duplicate_slot', 'missing_slot', 'wrong_rule', 'test_only'])
def test_actual_nondefault_scope_gate_plumbing_with_explicit_verifier_seam(tmp_path, monkeypatch, change):
    source_root(tmp_path, monkeypatch)
    feed = ReceiptArchive(tmp_path/'feed')
    source = feed.identity
    feed.close()
    runtime = ForwardRuntime(tmp_path/'runtime', tmp_path/'feed',
        config=ForwardConfig(archive_delay_seconds=19),
        lifecycle_plan=LifecyclePlan(source, calibration_directory='explicit-test-seam',
                                      calibration_binding='explicit-test-seam'))
    wallet = MonthlyWorkAuthority(tmp_path/'wallet', run_id='explicit-gate-seam',
                                 task='volatility', universe=['BTCUSDT'])
    try:
        _bind_authority(runtime, wallet)
        actual = runtime_calibration_contract(runtime.lifecycle)
        report = {'formal_gate_eligible': True, 'full_V7_five_claim_pipeline_calibrated': True,
                  'covered_claim_types': ['library_skill', 'factor1', 'factor2', 'factor3', 'algorithm'],
                  'runtime_scope_hash': fingerprint(actual), 'rolling_hash': actual['rolling_hash'],
                  'rule_hash': actual['factor_confirmation_rule_hash'], 'test_only': False}
        if change == 'default_delay':
            report['rolling_hash'] = RollingSpec().digest
        elif change == 'different_scope':
            changed = json.loads(json.dumps(actual))
            changed['work_budget']['monthly_seconds'] = 7199
            report['runtime_scope_hash'] = fingerprint(changed)
        elif change == 'missing_scope':
            del report['runtime_scope_hash']
        elif change == 'duplicate_slot':
            report['covered_claim_types'].append('algorithm')
        elif change == 'missing_slot':
            report['covered_claim_types'].remove('factor3')
        elif change == 'wrong_rule':
            report['rule_hash'] = 'f'*64
        elif change == 'test_only':
            report['test_only'] = True
        # There is deliberately no certificate artifact in this test. Only this
        # fake verifier return permits the positive scope-predicate case. It is
        # evidence for configuration wiring, never full statistical acceptance.
        monkeypatch.setattr('factorlab.discovery.pipeline_calibration.verify_calibration_certificate',
                            lambda *a, **k: report)
        assert runtime.lifecycle.calibration_ready() is (change == 'none')
        assert wallet.verify()['reservations'] == 0
        assert not (tmp_path/'explicit-test-seam').exists()
    finally:
        runtime.close()
