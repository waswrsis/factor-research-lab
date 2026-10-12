"""No-fit components for staged code. Never paid forward/Linux acceptance."""
from __future__ import annotations
import json
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from factorlab.protocol import fingerprint
from factorlab.discovery.v7_lifecycle import Lifecycle, LifecyclePlan, canonical
from factorlab.discovery.convex_eval import RidgePredictor, TrainScaler
from factorlab.discovery.v7_forward import ForwardRuntime
from factorlab.discovery.v7_service import ForwardService
from factorlab.discovery.v7_autonomous import AutonomousService

OUTPUT = None


@pytest.fixture(autouse=True)
def _pytest_component_namespace(tmp_path, monkeypatch):
    # The standalone unittest runner sets OUTPUT itself. Under pytest each
    # test gets an independent ordinary project-root fixture, never a clock.
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    monkeypatch.setitem(globals(), 'OUTPUT', tmp_path / 'component')


class IsolationComponents(unittest.TestCase):
    def setUp(self):
        self.directory = OUTPUT / fingerprint(self._testMethodName)[:12]
        self.lc = Lifecycle(self.directory, LifecyclePlan(source_id='unpaid_component_only'))
        self.addCleanup(self.lc.close)

    def model(self, number):
        return RidgePredictor(TrainScaler(np.zeros(1), np.ones(1), (False,)),
                              np.array([float(number)]), 0., .1, ('component_x',),
                              provenance={'unpaid_not_fit': number})

    def job(self, month, *, stream='A0', activate=False, seal=True):
        pipeline = {'kind': 'v7_log_ridge', 'pipeline_id': f'component-{month}-{stream}'}
        job = self.lc.plan_fit(f'2026-{month:02d}-01T00:00:00+00:00', stream, pipeline)
        self.lc.advance_fit(job, 'snapshot_sealed')
        self.lc.advance_fit(job, 'features_ready')
        model = self.model(month)
        identifier = self.lc.seal_model(job, model) if seal else None
        if activate:
            row = self.lc.db.execute('SELECT generation FROM active WHERE stream=?', (stream,)).fetchone()
            self.lc.activate(job, self.lc.now(), expected_generation=row[0] if row else 0)
        return job, identifier, pipeline, model

    def damage(self, identifier):
        path = self.lc.registry.directory / identifier / 'model.json'
        body = path.read_bytes()
        (path.parent / 'original-component-artifact').write_bytes(body)
        path.write_bytes(b'{}\n')
        return path, body

    def pair(self):
        old = self.job(1, activate=True)
        new = self.job(2)
        self.lc.verify()
        damaged = self.damage(new[1])
        return old, new, damaged

    def no_anchors(self):
        self.assertFalse(list((self.directory / 'model_isolations').glob('*.json')))
        self.assertFalse(list(self.lc.registry.directory.glob('*/serving_isolation.json')))

    def test_old_active_survives_strict_audit_stays_red(self):
        old, new, _ = self.pair()
        result = self.lc.verify_serving()
        self.assertEqual(result['serving_only_isolated_jobs'], [new[0]])
        self.assertIs(result['formal_gate_ready'], False)
        self.assertEqual(self.lc.active_model('A0')['model_id'], old[1])
        with self.assertRaisesRegex(ValueError, 'full integrity'):
            self.lc.verify()
        runtime = ForwardRuntime.__new__(ForwardRuntime)
        runtime.lifecycle = self.lc
        self.assertEqual(runtime._activate_ready('A0', self.lc.now())['model_id'], old[1])
        self.assertEqual(self.lc.db.execute('SELECT failure FROM jobs WHERE id=?', (new[0],)).fetchone(),
                         ('integrity_failure',))

    def test_readback_is_stable_and_no_second_failure_credit(self):
        _, new, _ = self.pair()
        first = self.lc.verify_serving()
        count = self.lc.db.execute('SELECT COUNT(*) FROM operations').fetchone()[0]
        reloaded = Lifecycle(self.directory, self.lc.plan)
        try:
            self.assertEqual(reloaded.verify_serving(), first)
            self.assertEqual(reloaded.db.execute('SELECT COUNT(*) FROM operations').fetchone()[0], count)
            with self.assertRaises(ValueError):
                reloaded.activate(new[0], reloaded.now(), expected_generation=1)
        finally:
            reloaded.close()

    def test_partial_new_save_failure_needs_no_isolation(self):
        old = self.job(1, activate=True)
        new = self.job(2, seal=False)
        identifier = self.lc.registry.save(new[3], new[2])
        self.damage(identifier)
        with self.assertRaises((ValueError, KeyError)):
            self.lc.seal_model(new[0], new[3])
        self.lc.fail_fit(new[0], 'integrity_failure')
        self.assertNotIn('serving_only_isolated_jobs', self.lc.verify_serving())
        self.lc.verify()
        self.assertEqual(self.lc.active_model('A0')['model_id'], old[1])
        self.no_anchors()

    def test_sole_active_damage_rejected(self):
        old = self.job(1, activate=True)
        self.damage(old[1])
        with self.assertRaises(ValueError):
            self.lc.verify_serving()
        self.no_anchors()

    def test_no_active_rejected(self):
        new = self.job(2)
        self.damage(new[1])
        with self.assertRaises(ValueError):
            self.lc.verify_serving()
        self.no_anchors()

    def test_other_stream_active_cannot_authorize(self):
        self.job(1, stream='different', activate=True)
        new = self.job(2)
        self.damage(new[1])
        with self.assertRaises(ValueError):
            self.lc.verify_serving()
        self.no_anchors()

    def first_stream(self, stream, *, active_month=1, candidate_month=2):
        old = self.job(active_month, activate=True)
        new = self.job(candidate_month, stream=stream)
        self.lc.verify()
        self.damage(new[1])
        first = self.lc.verify_serving()
        self.assertEqual(first['serving_only_isolated_jobs'], [new[0]])
        self.assertIs(first['formal_gate_ready'], False)
        self.assertIsNone(self.lc.active_model(stream))
        self.assertEqual(self.lc.active_model('A0')['model_id'], old[1])
        anchor = json.loads((self.directory / 'model_isolations' / f'{new[0]}.json').read_text())
        self.assertEqual(anchor['evidence']['retained_active'], {
            'stream': 'A0', 'job': old[0], 'model': old[1], 'generation': 1,
            'activation': self.lc.active_model('A0')['activation']})
        self.assertEqual(self.lc.verify_serving(), first)
        with self.assertRaisesRegex(ValueError, 'full integrity'):
            self.lc.verify()

    def test_first_v_stream_uses_operational_A0(self):
        self.first_stream('V:first')

    def test_first_c_stream_uses_operational_A0(self):
        self.first_stream('C:first')

    def test_first_policy_stream_uses_operational_A0(self):
        self.first_stream('policy:first')

    def test_same_month_first_stream_uses_operational_A0(self):
        self.first_stream('V:first', active_month=2, candidate_month=2)

    def test_other_stream_cutoff_does_not_order_operational_A0(self):
        self.first_stream('C:first', active_month=3, candidate_month=2)

    def test_same_stream_active_without_A0_cannot_authorize(self):
        self.job(1, stream='V:first', activate=True)
        new = self.job(2, stream='V:first')
        self.damage(new[1])
        with self.assertRaisesRegex(ValueError, 'operational A0'):
            self.lc.verify_serving()
        self.no_anchors()

    def test_first_stream_with_damaged_A0_cannot_authorize(self):
        old = self.job(1, activate=True)
        new = self.job(2, stream='policy:first')
        self.damage(old[1])
        self.damage(new[1])
        with self.assertRaises(ValueError):
            self.lc.verify_serving()
        self.no_anchors()

    def test_activated_other_stream_damage_cannot_isolate(self):
        self.job(1, activate=True)
        other = self.job(2, stream='V:first', activate=True)
        self.damage(other[1])
        with self.assertRaisesRegex(ValueError, 'unactivated'):
            self.lc.verify_serving()
        self.no_anchors()

    def test_original_A0_binding_cannot_change_on_readback(self):
        self.first_stream('V:first')
        self.job(3, activate=True)
        with self.assertRaisesRegex(ValueError, 'retained active changed'):
            self.lc.verify_serving()

    def test_invalid_original_launch_payload_cannot_isolate(self):
        # Rejection-only envelope component, deliberately NOT a paid admission.
        # The real known-linear loader must reject the corrupt original before
        # any downstream paid ticket/receipt validation could grant permission.
        from factorlab.discovery.model_copy_isolation import candidate
        self.job(1, activate=True)
        new = self.job(2, stream='V:first')
        ticket = self.directory / 'invalid-paid-envelope' / 'ticket.json'
        original = ticket.parent / 'launch' / 'model.json'
        original.parent.mkdir(parents=True)
        original.write_text('{}')
        identity = {'version': 3, 'kind': 'v7_known_linear', 'model_digest': '0' * 64,
                    'schema': {}, 'task_support': ['volatility'], 'pipeline': new[2],
                    'admission': {'ticket': {'path': str(ticket)}}, 'job': new[0]}
        identifier = fingerprint(identity)
        folder = self.lc.registry.directory / identifier
        folder.mkdir()
        (folder / 'manifest.json').write_text(json.dumps({
            'identity': identity, 'id': identifier, 'artifact_sha256': '1' * 64}))
        (folder / 'model.json').write_text('{}')
        row = list(self.lc.db.execute('SELECT * FROM jobs WHERE id=?', (new[0],)).fetchone()[1:])
        row[4] = identifier
        with self.assertRaisesRegex(ValueError, 'Linear model integrity'):
            candidate(self.lc, new[0], row)
        self.no_anchors()

    def test_historical_superseded_damage_stays_strict(self):
        old = self.job(1, activate=True)
        self.job(2, activate=True)
        self.damage(old[1])
        with self.assertRaises(ValueError):
            self.lc.verify_serving()
        self.no_anchors()

    def test_active_and_candidate_both_bad_rejected(self):
        old, _, _ = self.pair()
        self.damage(old[1])
        with self.assertRaises(ValueError):
            self.lc.verify_serving()
        self.no_anchors()

    def test_wrong_manifest_rejected_without_isolation(self):
        _, new, _ = self.pair()
        path = self.lc.registry.directory / new[1] / 'manifest.json'
        value = json.loads(path.read_text())
        value['identity']['pipeline']['pipeline_id'] = 'borrowed'
        path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            self.lc.verify_serving()
        self.no_anchors()

    def test_operation_hash_damage_rejected_before_fault_write(self):
        self.pair()
        with self.lc.db:
            self.lc.db.execute('UPDATE operations SET hash=? WHERE seq=1', ('1' * 64,))
        with self.assertRaisesRegex(ValueError, 'chain mismatch'):
            self.lc.verify_serving()
        self.no_anchors()

    def test_original_forecast_receipt_damage_rejected(self):
        self.pair()
        ledger = self.lc.forecasts
        ledger.put(run='unpaid-component', stream='A0', origin='2026-01-02T00:00:05+00:00',
                   symbol='BTCUSDT', task='volatility', model_id='component',
                   pipeline_id='component', prediction={'value': 1})
        receipt = next(ledger.records_directory.glob('*.json'))
        receipt.write_bytes(b'{}\n')
        with self.assertRaises(ValueError):
            self.lc.verify_serving()
        self.no_anchors()

    def test_copy_restoration_does_not_credit_or_activate(self):
        _, new, (path, body) = self.pair()
        self.lc.verify_serving()
        path.write_bytes(body)
        for action in (self.lc.verify_serving, self.lc.verify,
                       lambda: self.lc.activate(new[0], self.lc.now(), expected_generation=1),
                       lambda: self.lc.seal_model(new[0], new[3]),
                       lambda: self.lc.registry.load(new[1]),
                       lambda: self.lc.registry.save(new[3], new[2])):
            with self.assertRaises(ValueError):
                action()

    def test_missing_one_mirror_cannot_reset(self):
        _, new, _ = self.pair()
        self.lc.verify_serving()
        mirror = self.lc.registry.directory / new[1] / 'serving_isolation.json'
        mirror.rename(mirror.with_name('retained-removed-isolation.json'))
        with self.assertRaises((FileNotFoundError, ValueError)):
            self.lc.verify_serving()

    def test_orphan_anchor_cannot_be_consumed(self):
        _, new, _ = self.pair()
        folder = self.directory / 'model_isolations'
        folder.mkdir()
        (folder / f'{new[0]}.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Orphan'):
            self.lc.verify_serving()

    def test_isolation_anchor_tamper_rejected(self):
        _, new, _ = self.pair()
        self.lc.verify_serving()
        path = self.directory / 'model_isolations' / f'{new[0]}.json'
        value = json.loads(path.read_text())
        value['evidence']['retained_active']['model'] = '0' * 64
        path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            self.lc.verify_serving()

    def test_bound_forward_cannot_use_unpaid_legacy_branch(self):
        self.pair()
        with self.lc.db:
            self.lc._event('forward_runtime_bound', {'runtime_identity_hash': 'component-invalid'})
        with self.assertRaisesRegex(ValueError, 'Original paid registry admission'):
            self.lc.verify_serving()
        self.no_anchors()

    def test_publication_copy_exception_keeps_other_bytes_and_prefix_strict(self):
        # Only the publication-file comparison is isolated here. The object
        # opening/finite-law identity is a declared component adapter, not a
        # registered generator or genuine quarantine admission.
        from factorlab.discovery import quarantined_observation as q
        from factorlab.discovery.model_copy_isolation import candidate
        from factorlab.discovery.v7_storage import immutable_record
        self.job(1, activate=True)
        new = self.job(2)
        folder = self.lc.registry.directory / new[1]
        (folder / 'registration_completed.json').write_text('{"component_only":true}')
        prefix = {'physical_observation': q._physical(), 'component': 'no_finite_law_admission'}
        files = [q._stat(path) for path in q._publication_paths(self.lc, 'model', new[0])]
        physical = q._physical()
        detail = {'kind': 'model', 'key': new[0], 'observation_prefix': prefix,
                  'files': files, 'physical_observation': physical}
        with self.lc.db:
            event = q._event(self.lc, 'quarantined_publication', detail, physical)
        root, mirror = self.directory / 'component-q', self.directory / 'component-q-mirror'
        name = fingerprint({'kind': 'model', 'key': new[0]}) + '.json'
        for base in (root, mirror):
            immutable_record(base / 'publications' / name, event)
        calls = []
        def binding(value):
            self.assertEqual(value, prefix)
            calls.append(value)
        adapter = SimpleNamespace(directory=root, mirror=mirror, verify_binding=binding)
        with patch.object(q.QuarantinedObservation, '_open_original', return_value=adapter):
            self.assertEqual(q.read_publication(self.lc, 'model', new[0]), detail)
            self.damage(new[1])
            row = list(self.lc.db.execute('SELECT * FROM jobs WHERE id=?', (new[0],)).fetchone()[1:])
            evidence = candidate(self.lc, new[0], row)
            with self.assertRaises(ValueError):
                q.read_publication(self.lc, 'model', new[0])
            self.assertEqual(q._read_publication(self.lc, 'model', new[0], isolated_copy=evidence), detail)
            forged = {**evidence, 'expected_artifact_sha256': '0' * 64}
            with self.assertRaises(ValueError):
                q._read_publication(self.lc, 'model', new[0], isolated_copy=forged)
            (folder / 'registration_completed.json').write_text('{"changed":true}')
            with self.assertRaises(ValueError):
                q._read_publication(self.lc, 'model', new[0], isolated_copy=evidence)
        self.assertGreaterEqual(len(calls), 4)

    def test_native_service_body_blocks_all_new_work_on_degraded_tick(self):
        # Control-flow component only: no _Scope or paid runtime is claimed.
        self.pair()
        integrity = self.lc.verify_serving()
        tick = {'at': self.lc.now().isoformat(), 'serving_only_isolated_jobs':
                integrity['serving_only_isolated_jobs']}
        service = ForwardService.__new__(ForwardService)
        service.lifecycle = self.lc
        service.runtime = SimpleNamespace(tick=lambda **_: tick)
        service.guard = lambda: {'allowed': True}
        service.monotonic = time.monotonic
        service.config = SimpleNamespace(poll=False)
        service.state = {'spent_seconds': 0., 'steps': 0}
        saves = []
        service._save = saves.append
        def forbidden(*args, **kwargs):
            raise AssertionError('Degraded service must not dispatch new work')
        service._freeze_due = service._endpoints = service._plan_month = forbidden
        service._algorithm_step = service._fit_pending = forbidden
        result = service._step_owned()
        self.assertEqual(result['status'], 'stepped')
        self.assertEqual(result['formal_claim_authority'], 'unavailable_integrity_fault')
        self.assertEqual(saves, ['step_started', 'step_finished'])
        self.assertGreaterEqual(service.state['spent_seconds'], 0.)
        service.step = lambda **_: result
        stops = []
        scheduler = SimpleNamespace(runtime=service.runtime, worker_active=lambda: False,
                                    step=forbidden, stop=lambda reason: stops.append(reason))
        AutonomousService(service, scheduler).step()
        self.assertEqual(stops, ['supervisor_stopped'])
