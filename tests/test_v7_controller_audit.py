"""Independent controller audit with artificial candles and a legal fake proposer.

No GPT/network calls. The main integration test uses actual data construction,
three-fold ridge/F/S evaluation, source manifests and SQLite persistence.
"""
import copy
import json
import sqlite3
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from factorlab.discovery import v7_cli, v7_run
from factorlab.discovery.operator_program import Program
from factorlab.discovery.program_syntax import CompactProgram, parse_expression
from factorlab.discovery.rolling import RollingSpec
from factorlab.discovery.v7_agent import BackendReply
from factorlab.discovery.v7_run import RunStore, V7Config, run_month

CUTOFF = '2025-04-01T00:00:00Z'


@pytest.fixture
def artificial_bars():
    times = pd.date_range('2024-09-01', '2025-04-03', freq='h', tz='UTC', inclusive='left')
    price = np.exp(4 + np.cumsum(.0001 + .002 * np.sin(np.arange(len(times)) / 5)))
    return pd.DataFrame({'symbol': 'BTCUSDT', 'open_time': times,
                         'available_at': times + pd.Timedelta(hours=1, seconds=1),
                         'open': price, 'close': price,
                         'high': price * 1.003, 'low': price / 1.003})


@pytest.fixture
def audit_root(tmp_path, monkeypatch):
    monkeypatch.setattr('factorlab.protocol.ROOT', tmp_path)
    (tmp_path / 'factorlab' / 'deep').mkdir(parents=True)
    (tmp_path / 'factorlab' / 'deep' / 'implementation.py').write_text('version=7\n', encoding='utf-8')
    (tmp_path / 'factorlab' / 'other.py').write_text('other=True\n', encoding='utf-8')
    (tmp_path / 'pyproject.toml').write_text('[project]\n', encoding='utf-8')
    (tmp_path / 'uv.lock').write_text('version=1\n', encoding='utf-8')
    return tmp_path


class LegalBatchBackend:
    def __init__(self):
        self.calls = []

    def __call__(self, packet, *, model, reasoning):
        body = json.loads(packet.body)
        self.calls.append({'packet': packet, 'model': model, 'reasoning': reasoning})
        index = len(self.calls)
        original = CompactProgram.from_dict(body['parents'][0]['program']).to_program().to_dict()
        original['outputs'] = [parse_expression('last(ema(f("rv_6"),p("alpha")))')]
        original['parameters'] = {'alpha': {'initial': .1 + index * .1, 'lower': .01, 'upper': .99}}
        compact = CompactProgram.from_program(Program.from_dict(original)).to_dict()
        value = {'version': 'v7-proposal-batch-v1', 'identity': body['identity'],
                 'family_id': f'audit-family-{index}', 'hypothesis': 'Artificial persistence hypothesis.',
                 'mechanism': 'A recent volatility statistic may have conditional predictive information.',
                 'competing_explanation': 'K0 already represents the relevant information.',
                 'falsification': 'Reject if preregistered S conditional evaluation gives no support.',
                 'parent_ids': [body['parents'][0]['parent_id']],
                 'template': {'base': copy.deepcopy(compact), 'slots': []},
                 'programs': [compact]}
        return BackendReply(json.dumps(value), {'input_tokens': 17, 'output_tokens': 9},
                            model_resolved='gpt-6-astra-fixture')


def read_month(directory):
    with sqlite3.connect(directory / 'state.sqlite') as connection:
        row = connection.execute('SELECT value FROM months WHERE month=?', ('2025-04',)).fetchone()
    return json.loads(row[0])


def cheap_candidate(data, cutoff, program, **kwargs):
    """Use only for restart/failure mechanics, never as numeric efficacy evidence."""
    from factorlab.discovery.causal_numpy import formula_identity
    from factorlab.protocol import fingerprint
    kwargs['budget'].consume('solve')
    return {'role': 'S', 'formula_id': fingerprint(formula_identity(program, kwargs['lookback'])),
            'channels': {channel: {'loss': 1., 'reference_loss': 1., 'gain': 0.}
                         for channel in ('F', 'S')}, 'evidence_status': 'synthetic_test_stub'}


def test_gpt_month_real_end_to_end_budget_parameters_feedback_and_complete_restart(audit_root, artificial_bars):
    backend = LegalBatchBackend()
    config = V7Config(search_mode='gpt', formulas_per_family=1, lookback=4)
    directory = audit_root / 'gpt-run'
    summary = run_month(directory, artificial_bars, CUTOFF, config=config, backend=backend)
    assert len(backend.calls) == 3
    assert summary['budget']['counts'] == {'gpt_attempt': 3, 'family': 3, 'ast': 3, 'f': 3, 'solve': 34}
    assert [row['status'] for row in summary['attempts']] == ['scored'] * 3
    assert summary['A0'] == 'K0' and summary['formal_confirmation_enabled'] is False
    state = read_month(directory)
    assert len(state['agent_audit']) == 3
    assert all(item['status'] == 'validated' for item in state['agent_audit'])
    assert all(call['model'] == 'gpt-6-astra' and call['reasoning'] == 'high' for call in backend.calls)
    for call in backend.calls:
        body = json.loads(call['packet'].body)
        assert body['identity']['train_calendar_months'] == 3
        assert body['identity']['fit_initialization'] == 'cold'
        assert all(parent['evidence']['role'] == 'S' for parent in body['parents'])
        assert 'actions' not in body
    assert len(json.loads(backend.calls[-1]['packet'].body)['parents']) == 4
    assert all(p['parameters']['alpha']['lower'] == .01 for p in
               (r['program'] for r in summary['shortlist']))
    repeated = run_month(directory, artificial_bars, CUTOFF, config=config, backend=backend)
    assert repeated == summary and len(backend.calls) == 3
    # New future rows/labels cannot invalidate or influence this completed as-of month.
    polluted = artificial_bars.copy()
    future = polluted.open_time >= pd.Timestamp(CUTOFF)
    polluted.loc[future, ['open', 'high', 'low', 'close']] *= 7
    assert run_month(directory, polluted, CUTOFF, config=config, backend=backend) == summary
    assert len(backend.calls) == 3
    polluted.loc[future, ['open', 'high', 'low', 'close']] = np.nan
    assert run_month(directory, polluted, CUTOFF, config=config, backend=backend) == summary
    # Mature historical inputs do bind the frozen month.
    changed = artificial_bars.copy()
    changed.loc[1000, 'high'] *= 1.1
    with pytest.raises(ValueError, match='dataset'):
        run_month(directory, changed, CUTOFF, config=config, backend=backend)


def test_unseen_future_cannot_change_new_run_prompt_feedback(audit_root, artificial_bars, monkeypatch):
    monkeypatch.setattr(v7_run, 'evaluate_formula', cheap_candidate)
    config = V7Config(search_mode='gpt', formulas_per_family=1, lookback=4)
    first, second = LegalBatchBackend(), LegalBatchBackend()
    run_month(audit_root / 'first', artificial_bars, CUTOFF, config=config, backend=first)
    polluted = artificial_bars.copy()
    polluted.loc[polluted.open_time >= pd.Timestamp(CUTOFF), ['open', 'high', 'low', 'close']] *= 9
    run_month(audit_root / 'second', polluted, CUTOFF, config=config, backend=second)
    assert [c['packet'].text for c in first.calls] == [c['packet'].text for c in second.calls]


def test_failed_transport_is_charged_and_durably_audited(audit_root, artificial_bars):
    directory = audit_root / 'failure'
    calls = []

    def timeout(packet, **kwargs):
        calls.append(packet.request_hash)
        raise TimeoutError('Artificial backend timeout; no network attempted')

    try:
        run_month(directory, artificial_bars, CUTOFF,
                  config=V7Config(search_mode='gpt', formulas_per_family=1), backend=timeout)
    except TimeoutError:
        pass  # Either explicit stop or bounded retry is acceptable, but persistence is not optional.
    state = read_month(directory)
    assert state['budget']['counts']['gpt_attempt'] == len(calls) > 0
    assert len(state['agent_audit']) == len(calls)
    assert all(a['status'] == 'failed' and a['error_type'] == 'TimeoutError' for a in state['agent_audit'])


def test_provider_start_is_durably_recorded_before_interruptible_external_work(audit_root, artificial_bars, monkeypatch):
    directory = audit_root / 'provider-interruption'
    captured = []

    class ProviderInterrupted(BaseException):
        pass

    def interrupted(packet, **kwargs):
        # Read through another connection while the controller is alive. This
        # distinguishes durable publication from an in-memory list entry.
        captured.append(read_month(directory))
        raise ProviderInterrupted()

    with pytest.raises(ProviderInterrupted):
        run_month(directory, artificial_bars, CUTOFF,
                  config=V7Config(search_mode='gpt', formulas_per_family=1), backend=interrupted)
    assert captured[0]['budget']['counts']['gpt_attempt'] == 1
    assert len(captured[0]['agent_audit']) == 1
    assert captured[0]['agent_audit'][0]['status'] == 'started'
    assert captured[0]['agent_audit'][0]['request_hash']
    state = read_month(directory)
    assert state['agent_audit'][0]['status'] == 'interrupted'
    assert state['agent_audit'][0]['finished_at_unix'] is not None
    monkeypatch.setattr(v7_run, 'evaluate_formula', cheap_candidate)
    backend = LegalBatchBackend()
    summary = run_month(directory, artificial_bars, CUTOFF,
                        config=V7Config(search_mode='gpt', formulas_per_family=1), backend=backend)
    assert summary['budget']['counts']['gpt_attempt'] == 4
    assert summary['budget']['counts']['family'] == 3
    assert len(read_month(directory)['agent_audit']) == 4


def test_active_candidate_restart_preserves_reservation_without_reproposal(audit_root, artificial_bars, monkeypatch):
    class ArtificialInterruption(BaseException):
        pass

    def interrupted(*args, **kwargs):
        raise ArtificialInterruption()

    config = V7Config(search_mode='gpt', formulas_per_family=1, lookback=4)
    backend = LegalBatchBackend()
    directory = audit_root / 'interruption'
    monkeypatch.setattr(v7_run, 'evaluate_formula', interrupted)
    with pytest.raises(ArtificialInterruption):
        run_month(directory, artificial_bars, CUTOFF, config=config, backend=backend)
    before = read_month(directory)
    assert before['active_candidate'] is not None
    assert before['budget']['counts']['f'] == 1
    assert before['budget']['counts']['gpt_attempt'] == 1
    old_id = before['active_candidate']['formula_id']
    monkeypatch.setattr(v7_run, 'evaluate_formula', cheap_candidate)
    summary = run_month(directory, artificial_bars, CUTOFF, config=config, backend=backend)
    assert len(backend.calls) == 3
    assert summary['budget']['counts']['f'] == 3
    assert [a['formula_id'] for a in summary['attempts'] if a['status'] == 'interrupted_consumed'] == [old_id]
    assert read_month(directory)['active_candidate'] is None


def test_month_identity_binds_availability_provenance(audit_root, artificial_bars, monkeypatch):
    monkeypatch.setattr(v7_run, 'evaluate_formula', cheap_candidate)
    config = V7Config(search_mode='gpt', formulas_per_family=1, lookback=4)
    directory = audit_root / 'provenance'
    run_month(directory, artificial_bars, CUTOFF, config=config, backend=LegalBatchBackend())
    relabeled = artificial_bars.copy()
    relabeled.attrs['availability_mode'] = 'observed_rest_feed'
    relabeled['received_at'] = relabeled.available_at
    relabeled['prediction_origin'] = True
    with pytest.raises(ValueError, match='dataset|provenance|identity'):
        run_month(directory, relabeled, CUTOFF, config=config, backend=LegalBatchBackend())


@pytest.mark.parametrize('relative', ['factorlab/deep/implementation.py', 'factorlab/other.py',
                                     'pyproject.toml', 'uv.lock'])
def test_entire_declared_source_and_dependency_identity_is_frozen(audit_root, relative):
    directory = audit_root / 'source-binding'
    store = RunStore(directory, V7Config(), RollingSpec())
    store.close()
    source = audit_root / relative
    source.write_text(source.read_text('utf-8') + '# modified\n', encoding='utf-8')
    with pytest.raises(ValueError, match='source/config'):
        RunStore(directory, V7Config(), RollingSpec())


def test_archived_source_tampering_is_detected_on_resume(audit_root):
    directory = audit_root / 'archive-binding'
    store = RunStore(directory, V7Config(), RollingSpec())
    store.close()
    archived = directory / 'source' / 'factorlab' / 'deep' / 'implementation.py'
    archived.write_text('tampered_archive=True\n', encoding='utf-8')
    with pytest.raises(ValueError, match='source|snapshot|archive|integrity'):
        RunStore(directory, V7Config(), RollingSpec())


def test_non12h_task_cannot_reach_hardcoded_12h_prompt(audit_root, artificial_bars, monkeypatch):
    def unexpected_build(*args, **kwargs):
        raise RuntimeError('Non12h task reached dataset construction despite fixed 12h prompt')

    monkeypatch.setattr(v7_run, 'build_rolling_data', unexpected_build)
    with pytest.raises(ValueError, match='12'):
        run_month(audit_root / 'wrong-horizon', artificial_bars, CUTOFF,
                  config=V7Config(search_mode='gpt'), spec=RollingSpec(horizon_hours=24),
                  backend=LegalBatchBackend())


def test_cli_passes_explicit_config_without_hidden_model_switch(audit_root, artificial_bars, monkeypatch, capsys):
    config = V7Config(search_mode='gpt', formulas_per_family=1)
    path = audit_root / 'config.json'
    path.write_text(json.dumps(asdict(config)), encoding='utf-8')
    calls = []

    def observed_call(directory, bars, cutoff, **kwargs):
        calls.append((directory, bars, cutoff, kwargs))
        return {'scope': 'synthetic_cli_audit'}

    monkeypatch.setattr(pd, 'read_parquet', lambda _: artificial_bars)
    monkeypatch.setattr(v7_run, 'run_month', observed_call)
    assert v7_cli.main(['research', '--offline-diagnostic', '--bars', str(audit_root / 'bars.parquet'),
                        '--run', str(audit_root / 'cli'), '--cutoff', CUTOFF,
                        '--config', str(path), '--availability-delay-seconds', '5']) == 0
    assert len(calls) == 1
    passed = calls[0][-1]['config']
    assert passed.search_mode == 'gpt' and passed.model == 'gpt-6-astra' and passed.reasoning == 'high'
    assert calls[0][-1]['spec'].availability_delay_seconds == 5
    assert calls[0][-1]['spec'].months == 3
    assert json.loads(capsys.readouterr().out)['scope'] == 'synthetic_cli_audit'


@pytest.mark.parametrize(('task', 'horizon'), [('volatility', 12), ('first_event', 20)])
def test_replay_cli_passes_explicit_five_second_availability_policy(
    audit_root, artificial_bars, monkeypatch, capsys, task, horizon,
):
    calls = []

    def replay(directory, bars, cutoffs, *, spec):
        calls.append((directory, bars, cutoffs, spec))
        return [{'cutoff': cutoffs[0], 'new_predictions': 0, 'formal_claim': False}]

    monkeypatch.setattr(pd, 'read_parquet', lambda _: artificial_bars)
    monkeypatch.setattr(v7_run, 'replay_baseline', replay)
    assert v7_cli.main(['replay', '--bars', str(audit_root / 'bars.parquet'),
                        '--run', str(audit_root / f'replay-{task}'), '--cutoffs', CUTOFF,
                        '--symbol', 'BTCUSDT', '--task', task,
                        '--availability-delay-seconds', '5']) == 0
    assert len(calls) == 1
    spec = calls[0][-1]
    assert spec.availability_delay_seconds == 5
    assert spec.months == 3 and spec.task == task and spec.horizon_hours == horizon
    assert json.loads(capsys.readouterr().out)[0]['formal_claim'] is False


def test_historical_inspection_accepts_changed_current_source_without_resuming_or_rewriting(
    audit_root, capsys,
):
    directory = audit_root / 'historical-inspect'
    config, spec = V7Config(), RollingSpec(availability_delay_seconds=5)
    store = RunStore(directory, config, spec)
    store.save('2025-04', {'status': 'running', 'audit_fixture': True}, 'test_fixture')
    store.close()
    archived = directory / 'source' / 'factorlab' / 'deep' / 'implementation.py'
    watched = [directory / 'state.sqlite', directory / 'manifest.json', archived]
    before = {path: path.read_bytes() for path in watched}
    live_source = audit_root / 'factorlab' / 'deep' / 'implementation.py'
    live_source.write_text('version=8\n', encoding='utf-8')

    assert v7_cli.main(['inspect', '--run', str(directory)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['mode'] == 'historical_read_only'
    assert result['source_matches_current'] is False
    assert result['months'] == 1 and result['summaries'] == []
    assert {path: path.read_bytes() for path in watched} == before
    # Read-only inspection does not grant the changed source permission to resume.
    with pytest.raises(ValueError, match='source/config'):
        RunStore(directory, config, spec)
    # Historical inspection still verifies the archive itself, even though it
    # deliberately does not require its bytes to match today's live source.
    archived.write_text('corrupted_history=True\n', encoding='utf-8')
    with pytest.raises(ValueError, match='archive integrity'):
        v7_run.inspect_run(directory)


def test_nonempty_parameter_template_counts_duplicates_and_prescreens_without_targets(
    audit_root, artificial_bars, monkeypatch,
):
    """One seed plus four materializations includes a duplicate, all charged.

    Only downstream F/S is stubbed; parsing, expansion, strict batch membership,
    raw causal features, residual prescreen and durable ledger remain real.
    """
    class TemplateBackend(LegalBatchBackend):
        def __call__(self, packet, **kwargs):
            reply = super().__call__(packet, **kwargs)
            value = json.loads(reply.payload)
            initial = value['template']['base']['parameters']['alpha']['initial']
            value['template']['slots'] = [{
                'path': ['parameters', 'alpha', 'initial'],
                'values': [initial + offset for offset in (0., .03, .06, .09)],
            }]
            return BackendReply(json.dumps(value), reply.usage, reply.model_resolved)

    class NoTargetView:
        def __init__(self, data):
            self.frame, self.spec = data.frame, data.spec
            self.train_mask = data.train_mask

        @property
        def target(self):
            raise AssertionError('Residual prescreen may not read target values')

    prescreen = v7_run._prescreen
    observed_screens = []

    def guarded_prescreen(data, cutoff, programs, lookback, limit):
        selected, records = prescreen(NoTargetView(data), cutoff, programs, lookback, limit)
        observed_screens.append((len(programs), len(selected), records))
        return selected, records

    monkeypatch.setattr(v7_run, '_prescreen', guarded_prescreen)
    monkeypatch.setattr(v7_run, 'evaluate_formula', cheap_candidate)
    backend = TemplateBackend()
    directory = audit_root / 'nonempty-template'
    summary = run_month(directory, artificial_bars, CUTOFF,
                        config=V7Config(search_mode='gpt', formulas_per_family=2, lookback=4),
                        backend=backend)
    state = read_month(directory)
    assert len(backend.calls) == 3
    assert summary['budget']['counts']['gpt_attempt'] == 3
    assert summary['budget']['counts']['ast'] == 15  # Three seeds + twelve template visits.
    assert summary['budget']['counts']['f'] == 6
    assert all(counts['ast'] == 5 and counts['f'] == 2
               for counts in summary['budget']['families'].values())
    assert all(counts['ast'] <= summary['budget']['limits']['family_ast']
               and counts['f'] <= summary['budget']['limits']['family_f']
               for counts in summary['budget']['families'].values())
    materialized = state['materializations']
    assert len(materialized) == 12
    assert sum(row['status'] == 'duplicate' for row in materialized) == 3
    assert sum(row['status'] == 'valid' for row in materialized) == 9
    assert len(observed_screens) == 3
    assert all(total == 4 and selected == 2 for total, selected, _ in observed_screens)
    assert all(len(records) == 4 and sum(r['selected'] for r in records) == 2
               for _, _, records in observed_screens)
    assert all(a['status'] == 'validated' for a in state['agent_audit'])
