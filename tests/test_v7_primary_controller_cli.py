"""Canonical primary driver CLI boundaries; no provider/statistical evidence."""
import pytest

from factorlab.discovery.v7_cli import main


@pytest.mark.parametrize('command', ('trial-controller-pump', 'trial-controller-run', 'trial-controller-status'))
def test_primary_driver_missing_registry_never_initializes(tmp_path, monkeypatch, command):
    from factorlab import protocol
    monkeypatch.setattr(protocol, 'ROOT', tmp_path)
    missing = tmp_path / 'absent-registry'
    with pytest.raises((ValueError, FileNotFoundError)):
        main([command, '--registry', str(missing), '--cell', 'null', '--repetition', '0'])
    assert not missing.exists()


@pytest.mark.parametrize('value', ('0', '-1', '7201', '1.5'))
def test_primary_driver_invalid_limit_rejected_before_inspection(tmp_path, value):
    missing = tmp_path / 'absent-registry'
    with pytest.raises(SystemExit) as exc:
        main(['trial-controller-run', '--registry', str(missing), '--cell', 'null',
              '--repetition', '0', '--max-pumps', value])
    assert exc.value.code == 2
    assert not missing.exists()


def test_primary_driver_uses_original_inspected_controller_run(tmp_path, monkeypatch, capsys):
    from factorlab.discovery import full_calibration_registry as registry_module
    from factorlab.discovery import full_trial_controller as controller_module
    calls = []

    class Registry:
        def identity_for(self, cell, repetition, **coordinates):
            calls.append(('identity', cell, repetition, coordinates))
            return {'actual_test_identity': True}

    registry = Registry()
    monkeypatch.setattr(registry_module.FullTrialRegistry, 'inspect',
                        classmethod(lambda cls, path: registry))
    monkeypatch.setattr(registry_module, 'QuarantinedTrialPermit',
                        lambda inspected, identity: (inspected, identity))

    class Controller:
        def run(self, *, max_pumps):
            calls.append(('run', max_pumps))
            return {'state': 'test_only', 'formal_gate_eligible': False}

    def inspect(cls, permit):
        assert permit == (registry, {'actual_test_identity': True})
        calls.append(('inspect_existing',))
        return Controller()

    monkeypatch.setattr(controller_module.FullTrialController, 'inspect', classmethod(inspect))
    main(['trial-controller-run', '--registry', str(tmp_path / 'r'), '--cell', 'null',
          '--repetition', '0', '--max-pumps', '7'])
    assert calls == [('identity', 'null', 0, {'oracle_index': None}),
                     ('inspect_existing',), ('run', 7)]
    assert 'test_only' in capsys.readouterr().out



def test_primary_status_counts_only_canonical_successor_witness(tmp_path, monkeypatch, capsys):
    import json
    from types import SimpleNamespace
    from factorlab.discovery import full_calibration_registry as registry_module
    from factorlab.discovery import full_trial_controller as controller_module
    rows = [
        {'result': {'state': 'awaiting_successor_witness'}, 'observed_original': True},
        {'result': {'state': 'awaiting_successor_witness'}, 'observed_original': False},
        {'result': {'state': 'completed_step'}, 'observed_original': False},
    ]
    evaluated = []

    class Registry:
        def identity_for(self, *args, **kwargs): return {'fixture': True}

    class Controller:
        identity = {'fixture': True}
        schedule = SimpleNamespace(due_at=(1, 2, 3))
        def verify(self): return rows
        def pending_dependencies(self): return ()
        def step_completed(self, row):
            evaluated.append(row)
            return row['observed_original']
        def pump(self): raise AssertionError('Read-only status executed work')
        def run(self, **kwargs): raise AssertionError('Read-only status executed work')

    monkeypatch.setattr(registry_module.FullTrialRegistry, 'inspect',
                        classmethod(lambda cls, path: Registry()))
    monkeypatch.setattr(registry_module, 'QuarantinedTrialPermit', lambda *args: object())
    monkeypatch.setattr(controller_module.FullTrialController, 'inspect',
                        classmethod(lambda cls, permit: Controller()))
    main(['trial-controller-status', '--registry', str(tmp_path / 'r'),
          '--cell', 'null', '--repetition', '0'])
    value = json.loads(capsys.readouterr().out)
    assert evaluated == rows
    assert value['completed_opportunities'] == 1
    assert value['attempted_opportunities'] == 3
    assert value['read_only']
    assert not value['formal_gate_eligible']


@pytest.mark.parametrize('command', ('trial-controller-run', 'policy-execution-run'))
def test_full_run_default_retains_issuer_through_original_schedule(tmp_path, monkeypatch, capsys, command):
    import json
    from factorlab.discovery import full_calibration_registry as registry_module
    from factorlab.discovery import full_trial_controller as controller_module
    from factorlab.discovery import adaptive_policy_execution as execution_module
    calls = []

    class Registry:
        def identity_for(self, *args, **kwargs): return {'fixture': True}

    class Controller:
        def run(self, *, max_pumps):
            calls.append(max_pumps)
            return {'state': 'fixture', 'formal_gate_eligible': False}

    monkeypatch.setattr(registry_module.FullTrialRegistry, 'inspect',
                        classmethod(lambda cls, path: Registry()))
    monkeypatch.setattr(registry_module, 'QuarantinedTrialPermit', lambda *args: object())
    monkeypatch.setattr(controller_module.FullTrialController, 'inspect',
                        classmethod(lambda cls, permit: Controller()))
    monkeypatch.setattr(execution_module, 'advance_original_policy',
                        lambda permit, *, max_pumps: Controller().run(max_pumps=max_pumps))
    args = [command, '--registry', str(tmp_path / 'r'), '--cell', 'null', '--repetition', '0']
    if command == 'policy-execution-run':
        args += ['--oracle-index', '0']
    main(args)
    assert calls == [7200]
    assert not json.loads(capsys.readouterr().out)['formal_gate_eligible']
