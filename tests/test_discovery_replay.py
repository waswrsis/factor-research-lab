"""Sequential information-flow checks, using dummy checkpoints and fit spies."""
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from factorlab.discovery import replay
from factorlab.discovery.contracts import ExperimentSpec, TaskDataset, TaskSpec
from factorlab.discovery.evidence import EvidenceRegistry
from factorlab.protocol import fingerprint


@pytest.fixture(autouse=True)
def isolated_evidence_registry(monkeypatch, tmp_path):
    # Every test, including older replay cases, must stay out of the actual ledger.
    path = tmp_path / 'test_replay_evidence.sqlite'
    monkeypatch.setattr(replay, 'DEFAULT_REGISTRY_PATH', path)
    monkeypatch.setattr(replay, 'DEFAULT_EVIDENCE_SOURCE', 'synthetic_test_replay')
    return path


def setup_data(n=30):
    config = ExperimentSpec(
        tasks=(TaskSpec('vol', 'volatility', 2),
               TaskSpec('event', 'first_event', 4, (2, 4)), TaskSpec('state', 'state', 1)),
        primary_task='vol', lookback=2, symbols=('BTCUSDT',),
        start='2025-01-01', train_end='2025-01-01T12:00Z', dev_end='2025-01-02T06:00Z',
        calibration_end='2025-01-02T12:00Z', end='2025-01-03',
    )
    times = pd.date_range('2025-01-01', periods=n, freq='h', tz='UTC')
    x = np.zeros((n, 2, 2), dtype=np.float32)
    x[:, :, 0] = np.arange(n)[:, None]
    x[:, :, 1] = 1.
    meta = pd.DataFrame({'sample_id': [f'BTCUSDT:{i}' for i in range(n)],
                         'symbol': 'BTCUSDT', 'available_at': times,
                         'feature_open_time': times - pd.Timedelta(hours=1),
                         'label_diagnostic_future': np.arange(n) + 100})
    maturity = {t.name: (times + pd.Timedelta(hours=t.horizon_hours if t.predictive else 0)).to_numpy(dtype='datetime64[ns]')
                for t in config.tasks}
    observed = {t.name: maturity[t.name] <= times[-1].to_datetime64() for t in config.tasks}
    targets = {'vol': .02 + np.arange(n) * .001, 'event': (np.arange(n) % 5).astype(float),
               'state': np.arange(n) * .001}
    for name, target in targets.items():
        target[~observed[name]] = np.nan
    data = TaskDataset(x, targets, observed, maturity, meta, ('age', 'constant'), config.tasks)
    return data, config


class DummyModel:
    def __init__(self, schema, cutoff='2025-01-01T08:00Z', level=0., calls=None):
        self.schema, self.level = schema, level
        self.calls = calls if calls is not None else []
        self.diagnostics = {}
        self.metadata = {'mode': 'linear', 'architecture': None, 'level': level,
                         **{key: cutoff for key in replay.CUTOFF_FIELDS}}

    @property
    def digest(self):
        return fingerprint(self.metadata)

    def predict(self, data):
        assert all(not mask.any() for mask in data.observed.values())
        assert all(np.isnan(y).all() for y in data.targets.values())
        assert all(np.isnat(ready).all() for ready in data.maturity.values())
        assert 'label_diagnostic_future' not in data.meta
        self.calls.append(('predict', data.meta.available_at.iloc[0], self.level))
        n = len(data.x)
        return {'vol': np.full(n, .025 + self.level),
                'event': np.tile([.4, .2, .1, .2, .1], (n, 1)),
                'state': data.x[:, -1, 0].astype(float) * .001 + self.level}

    def encode(self, data):
        return data.x[:, -1]

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'metadata': self.metadata, 'digest': self.digest}), encoding='utf-8')
        return path


def install_spy(monkeypatch, calls, *, fail_first=False):
    fitted = []

    def fit(train, early, architecture, settings, device='cpu'):
        timestamp = [value[1] for value in calls if value[0] == 'predict'][-1]
        calls.append(('fit', timestamp, None))
        for part in (train, early):
            assert (part.meta.available_at < timestamp).all()
            for name, ready in part.maturity.items():
                assert (ready <= timestamp.to_datetime64()).all()
                assert np.isnan(part.targets[name][~part.observed[name]]).all()
        assert train.meta.available_at.max() < early.meta.available_at.min()
        assert all((ready <= early.meta.available_at.min().to_datetime64()).all()
                   for ready in train.maturity.values())
        fitted.append((train, early, timestamp))
        if fail_first and len(fitted) == 1:
            raise TimeoutError('synthetic bounded update failure')
        level = float(train.targets['vol'][train.observed['vol']].mean())
        return DummyModel(train.schema, timestamp.isoformat(), level, calls)

    monkeypatch.setattr(replay, 'fit', fit)
    return fitted


def prediction_values(loaded):
    return [record['value'] for record in loaded['records'] if record['event'] == 'prediction']


def test_fixed_replay_settles_each_mature_task_and_keeps_pending(monkeypatch, tmp_path):
    data, config = setup_data()
    monkeypatch.setattr(replay, 'fit', lambda *_args, **_kwargs: pytest.fail('fixed must not fit'))
    result = replay.run_replay(data, DummyModel(data.schema), config, tmp_path / 'fixed',
                               update_every_hours=3)
    loaded = replay.load_replay(tmp_path / 'fixed')
    assert result['predictions'] == 18
    assert result['pending_tasks'] == 4
    assert result['scored_tasks'] == 48
    assert result['censored_tasks'] == 2  # Dataset ended before two nominally mature outcomes.
    assert result['tasks']['state']['support'] == 18
    assert result['tasks']['vol']['support'] == 16
    assert result['tasks']['event']['support'] == 14
    assert result['scope'] == 'previously_seen_development_interval_not_independent_confirmation'
    assert all(update['status'] == 'fixed_model_retained' for update in result['updates'])
    records = loaded['records']
    forecasts = {r['sequence']: r for r in records if r['event'] == 'prediction'}
    for record in records:
        if record['event'] != 'settlement':
            continue
        value = record['value']
        forecast = forecasts[value['prediction_sequence']]
        assert forecast['sequence'] < record['sequence']
        assert pd.Timestamp(value['maturity']) <= pd.Timestamp(record['timestamp'])
        assert record['timestamp'] > forecast['timestamp']
    first = prediction_values(loaded)[0]
    assert first['latent_groups']['fixed_summary'] == [12., 1.]
    assert first['adaptive_horizons']['event']['evaluation_target'] == 'complete_joint_first_event_distribution'
    assert not first['adaptive_horizons']['event']['concentration_is_confidence']


def test_refit_label_cutoff_chronological_purge_and_prediction_before_update(monkeypatch, tmp_path):
    data, config = setup_data()
    calls = []
    fitted = install_spy(monkeypatch, calls)
    result = replay.run_replay(data, DummyModel(data.schema, calls=calls), config,
                               tmp_path / 'refit', update_every_hours=3, update_mode='refit')
    assert len(fitted) == 5
    assert all(update['status'] == 'refitted' for update in result['updates'])
    for i, call in enumerate(calls):
        if call[0] == 'fit':
            assert calls[i - 1][0] == 'predict' and calls[i - 1][1] == call[1]
    loaded = replay.load_replay(tmp_path / 'refit')
    predictions = prediction_values(loaded)
    by_time = {pd.Timestamp(value['available_at']): value for value in predictions}
    for update in result['updates']:
        timestamp = pd.Timestamp(update['information_cutoff'])
        assert by_time[timestamp]['version'] == update['previous_version']
        assert by_time[timestamp + pd.Timedelta(hours=1)]['version'] == update['version']
    assert len(result['versions']) == 6
    assert all(version['training'] is not None for version in result['versions'][1:])


def test_future_target_perturbation_cannot_change_earlier_scores(monkeypatch, tmp_path):
    data, config = setup_data()
    original_calls = []
    install_spy(monkeypatch, original_calls)
    replay.run_replay(data, DummyModel(data.schema, calls=original_calls), config,
                      tmp_path / 'original', update_every_hours=3, update_mode='refit')
    changed = data.subset(np.arange(len(data.meta)))
    # Earliest perturbed label becomes known at hour 20; inputs stay identical.
    changed.targets['vol'][20:28] += 100
    changed.targets['event'][20:26] = 4
    changed.targets['state'][20:] += 100
    changed.x[21:, :, 0] += 100  # Future input perturbations cannot reach earlier fit/predict.
    changed_calls = []
    install_spy(monkeypatch, changed_calls)
    replay.run_replay(changed, DummyModel(changed.schema, calls=changed_calls), config,
                      tmp_path / 'changed', update_every_hours=3, update_mode='refit')
    cutoff = pd.Timestamp('2025-01-01T20:00Z')
    original = [v for v in prediction_values(replay.load_replay(tmp_path / 'original'))
                if pd.Timestamp(v['available_at']) <= cutoff]
    altered = [v for v in prediction_values(replay.load_replay(tmp_path / 'changed'))
               if pd.Timestamp(v['available_at']) <= cutoff]
    assert original == altered


def test_failed_update_retains_previous_model_and_later_updates_continue(monkeypatch, tmp_path):
    data, config = setup_data()
    calls = []
    install_spy(monkeypatch, calls, fail_first=True)
    result = replay.run_replay(data, DummyModel(data.schema, calls=calls), config,
                               tmp_path / 'fail', update_every_hours=3, update_mode='refit')
    assert result['updates'][0]['status'] == 'failed_old_version_retained'
    assert result['updates'][0]['error_type'] == 'TimeoutError'
    assert result['updates'][1]['status'] == 'refitted'
    forecasts = prediction_values(replay.load_replay(tmp_path / 'fail'))
    assert all(v['version'] == 0 for v in forecasts[:7])  # Through hour 18 prediction.
    assert forecasts[7]['version'] == 2  # Failed attempts also have distinct version identities.


def test_ambiguous_and_missing_targets_never_score(monkeypatch, tmp_path):
    data, config = setup_data()
    data.observed['event'][12] = False
    data.targets['event'][12] = np.nan
    data.observed['vol'][13] = False
    data.targets['vol'][13] = np.nan
    result = replay.run_replay(data, DummyModel(data.schema), config, tmp_path / 'missing')
    settlements = [r['value'] for r in replay.load_replay(tmp_path / 'missing')['records']
                   if r['event'] == 'settlement']
    assert result['censored_tasks'] == 4
    unobserved = [value for value in settlements if value['status'] == 'unobserved']
    assert len(unobserved) == 4
    assert any(value['sample_id'] == 'BTCUSDT:12' and value['task'] == 'event' for value in unobserved)
    assert any(value['sample_id'] == 'BTCUSDT:13' and value['task'] == 'vol' for value in unobserved)
    assert all(value['target'] is None and value['loss'] is None for value in unobserved)
    assert result['pending_tasks'] == 4


@pytest.mark.parametrize('problem', ['missing', 'future_fit', 'future_development'])
def test_initial_checkpoint_requires_past_training_provenance(problem, tmp_path):
    data, config = setup_data()
    model = DummyModel(data.schema)
    if problem == 'missing':
        del model.metadata['fit_label_cutoff']
    else:
        key = 'fit_label_cutoff' if problem == 'future_fit' else 'development_feature_cutoff'
        model.metadata[key] = '2025-01-01T13:00Z'
    with pytest.raises(ValueError, match='provenance|information cutoff'):
        replay.run_replay(data, model, config, tmp_path / problem)
    assert not (tmp_path / problem).exists()


def test_future_trained_refit_checkpoint_is_rejected_and_previous_version_kept(monkeypatch, tmp_path):
    data, config = setup_data()
    monkeypatch.setattr(replay, 'fit', lambda *_args, **_kwargs:
                        DummyModel(data.schema, '2025-01-04'))
    result = replay.run_replay(data, DummyModel(data.schema), config, tmp_path / 'bad-update',
                               update_every_hours=3, update_mode='refit')
    assert all(v['status'] == 'failed_old_version_retained' for v in result['updates'])
    assert len(result['versions']) == 1


def test_integer_event_targets_with_unobserved_placeholders_are_redacted_before_refit(monkeypatch, tmp_path):
    data, config = setup_data()
    data.targets['event'] = np.where(data.observed['event'], data.targets['event'], -1).astype(np.int64)
    data.observed['event'][1] = False
    data.targets['event'][1] = -1
    calls = []
    install_spy(monkeypatch, calls)
    result = replay.run_replay(data, DummyModel(data.schema, calls=calls), config,
                               tmp_path / 'integer', update_every_hours=3, update_mode='refit')
    assert all(update['status'] == 'refitted' for update in result['updates'])


@pytest.mark.parametrize('target', ['journal_change', 'journal_truncate', 'version', 'source', 'summary'])
def test_reload_detects_history_and_bundle_tampering(target, tmp_path):
    data, config = setup_data()
    output = tmp_path / target
    replay.run_replay(data, DummyModel(data.schema), config, output)
    if target.startswith('journal'):
        path = output / 'records.jsonl'
        lines = path.read_text('utf-8').splitlines()
        if target == 'journal_change':
            record = json.loads(lines[1])
            record['value']['predictions']['vol'] = 999.
            lines[1] = json.dumps(record)
        else:
            lines = lines[:-1]
        path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    elif target == 'summary':
        path = output / 'summary.json'
        record = json.loads(path.read_text())
        record['summary']['predictions'] = 999
        path.write_text(json.dumps(record), encoding='utf-8')
    else:
        path = output / ('versions/0000/model.pt' if target == 'version' else 'source/replay.py')
        path.write_bytes(path.read_bytes() + b'tamper')
    with pytest.raises(ValueError, match='integrity'):
        replay.load_replay(output)


def test_completed_history_cannot_be_overwritten_or_retroactively_updated(tmp_path):
    data, config = setup_data()
    output = tmp_path / 'immutable'
    replay.run_replay(data, DummyModel(data.schema), config, output)
    before = (output / 'records.jsonl').read_bytes()
    with pytest.raises(ValueError, match='history already exists'):
        replay.run_replay(data, DummyModel(data.schema), config, output, update_mode='refit')
    assert (output / 'records.jsonl').read_bytes() == before


def test_unavailable_prediction_is_not_scored(tmp_path):
    data, config = setup_data()

    class MissingHead(DummyModel):
        def predict(self, inputs):
            predictions = super().predict(inputs)
            del predictions['event']
            return predictions

    result = replay.run_replay(data, MissingHead(data.schema), config, tmp_path / 'no-head')
    assert result['tasks']['event']['support'] == 0
    settlements = [r['value'] for r in replay.load_replay(tmp_path / 'no-head')['records']
                   if r['event'] == 'settlement' and r['value']['task'] == 'event']
    assert all(v['status'] in {'unavailable_prediction', 'unobserved'} and v['loss'] is None
               for v in settlements)


def test_schedule_is_coalesced_across_input_gap_and_keeps_fixed_time_anchor(monkeypatch, tmp_path):
    data, config = setup_data()
    data = data.subset(~np.isin(np.arange(len(data.meta)), [15, 16, 17, 18, 19]))
    result = replay.run_replay(data, DummyModel(data.schema), config, tmp_path / 'gap',
                               update_every_hours=3)
    assert result['updates'][0]['scheduled_at'] == ['2025-01-01T15:00:00+00:00',
                                                   '2025-01-01T18:00:00+00:00']
    assert result['updates'][0]['information_cutoff'] == '2025-01-01T20:00:00+00:00'
    assert result['updates'][1]['scheduled_at'] == ['2025-01-01T21:00:00+00:00']


def test_invalid_schedule_and_protocol_are_rejected(tmp_path):
    data, config = setup_data()
    with pytest.raises(ValueError, match='schedule'):
        replay.run_replay(data, DummyModel(data.schema), config, tmp_path / 'schedule', update_every_hours=0)
    with pytest.raises(ValueError, match='lookback'):
        replay.run_replay(data, DummyModel(data.schema), replace(config, lookback=3), tmp_path / 'schema')


def test_replay_registers_actual_warmup_before_predict_fit_and_blocks_later_confirmation(monkeypatch, tmp_path):
    data, config = setup_data()
    registry_path = tmp_path / 'injected.sqlite'
    source = 'isolated_market_source'
    registry = EvidenceRegistry(registry_path)
    calls = []
    original_fit = install_spy(monkeypatch, calls)
    fit_spy = replay.fit

    class RegisteredModel(DummyModel):
        def predict(self, inputs):
            assert len(registry.records()) == 1
            return super().predict(inputs)

    def checked_fit(*args, **kwargs):
        assert len(registry.records()) == 1
        return fit_spy(*args, **kwargs)

    monkeypatch.setattr(replay, 'fit', checked_fit)
    replay.run_replay(data, RegisteredModel(data.schema, calls=calls), config,
                      tmp_path / 'registered', update_every_hours=3, update_mode='refit',
                      registry_path=registry_path, evidence_source=source)
    assert original_fit
    rows = registry.records()
    expected_start = data.meta.feature_open_time.min() - pd.Timedelta(hours=config.lookback - 1 + 48)
    assert rows[0]['start'] == expected_start.isoformat()
    assert rows[0]['end'] == pd.Timestamp(config.dev_end).isoformat()
    assert rows[0]['role'] == 'development' and rows[0]['source'] == source
    exposure = replay.load_replay(tmp_path / 'registered')['manifest']['evidence_exposure']
    assert exposure['receipts'] == rows
    assert exposure['registered_before'] == 'dataset_hash_prediction_fit_and_user_results'
    with pytest.raises(ValueError, match='Previously exposed'):
        registry.consume(source=source, symbols=('BTCUSDT',), start=config.train_end,
                         end=config.dev_end, run='later_confirmation', claim='fresh_claim', role='confirmation')


def test_replay_confirmation_suffix_is_never_hashed_fitted_or_registered(monkeypatch, tmp_path):
    data, config = setup_data(45)
    registry_path = tmp_path / 'suffix.sqlite'
    source = 'suffix_market_source'
    baseline = data.subset(np.arange(len(data.meta)))
    # Full supplied data includes later unused material, but replay reads only through dev_end.
    end = pd.Timestamp(config.dev_end)
    future = data.meta.available_at >= end
    data.x[future] *= 100
    for name, values in data.targets.items():
        hidden = data.maturity[name] > end.to_datetime64()
        values[hidden] = ((values[hidden] + 1) % 5 if name == 'event' else values[hidden] + 100)
    replay.run_replay(baseline, DummyModel(data.schema), config, tmp_path / 'baseline-suffix',
                      registry_path=registry_path, evidence_source=source)
    replay.run_replay(data, DummyModel(data.schema), config, tmp_path / 'changed-suffix',
                      registry_path=registry_path, evidence_source=source)
    first = replay.load_replay(tmp_path / 'baseline-suffix')
    second = replay.load_replay(tmp_path / 'changed-suffix')
    assert first['manifest']['data_sha256'] == second['manifest']['data_sha256']
    assert prediction_values(first) == prediction_values(second)
    rows = EvidenceRegistry(registry_path).records()
    assert all(row['end'] == end.isoformat() for row in rows)
    # Half-open exposure does not consume the unused post-replay confirmation suffix.
    EvidenceRegistry(registry_path).consume(source=source, symbols=('BTCUSDT',),
        start=config.calibration_end, end=config.end, run='later', claim='unused_suffix', role='confirmation')


def test_registry_failure_prevents_any_prediction_or_fit(monkeypatch, tmp_path):
    data, config = setup_data()
    calls = []
    monkeypatch.setattr(EvidenceRegistry, 'consume', lambda *_args, **_kwargs:
                        (_ for _ in ()).throw(OSError('synthetic ledger failure')))
    with pytest.raises(OSError, match='ledger failure'):
        replay.run_replay(data, DummyModel(data.schema, calls=calls), config, tmp_path / 'failed-ledger')
    assert not calls
    assert not (tmp_path / 'failed-ledger' / 'records.jsonl').exists()
