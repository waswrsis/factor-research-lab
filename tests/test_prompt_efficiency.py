"""Artificial inputs only. No real GPT, market inference or serving evidence."""
import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from factorlab.discovery import prompt_efficiency as pe
from factorlab.protocol import fingerprint


def base_packet():
    from factorlab.discovery.prompt_builder import PromptIdentity, build_formula_prompt
    from factorlab.discovery.local_mutation import MonthlyBudget
    from factorlab.discovery.known_library import FEATURE_NAMES
    from factorlab.discovery.program_contract import Program
    from factorlab.discovery.program_syntax import CompactProgram
    from factorlab.discovery.v7_run import literature_seeds
    identity = PromptIdentity('volatility12h', 'rolling', 'pool', 'features')
    parents = [{'parent_id': f'literature{i}', 'program': CompactProgram.from_program(Program.from_dict(p)).to_dict(),
        'evidence': {'evidence_id': f'lit{i}', 'role': 'S', 'task_id': identity.task_id,
            'rolling_id': identity.rolling_id, 'pool_id': identity.pool_id,
            'summary': 'Artificial fixture literature seed, no empirical result.', 'metrics': {}}}
        for i, p in enumerate(literature_seeds())]
    return build_formula_prompt(identity=identity, task_semantics='Predict 12h volatility.',
        feature_schema=[{'name': x, 'unit': 'raw', 'preprocessing': 'training-only'} for x in FEATURE_NAMES],
        input_semantics='Past 48 causal hours.', parents=parents, budget=MonthlyBudget('2026-10'))


def reply(packet, family='fixture'):
    body = json.loads(packet.body)
    compact = body['parents'][0]['program']
    return {'version': 'v7-proposal-batch-v1', 'identity': body['identity'], 'family_id': family,
        'hypothesis': 'A fixture hypothesis.', 'mechanism': 'Past volatility persistence.',
        'competing_explanation': 'No information beyond K0.', 'falsification': 'Original S readout only.',
        'parent_ids': [body['parents'][0]['parent_id']], 'template': {'base': compact, 'slots': []},
        'programs': [compact]}


def test_precise_fixed_budget_and_two_context_no_new_family():
    p = pe.protocol()
    assert p['reserved_seconds'] == 300 + 3*300 + 24*90 == 3360
    assert p['reserved_counts']['solve'] == 2*(3*2+1) + 3*(198+7*15) == 923
    assert {k: p['reserved_counts'][k] for k in ('ast', 'f', 'family', 'gpt_attempt')} == {
        'ast': 96, 'f': 24, 'family': 3, 'gpt_attempt': 6}
    assert len(pe.schedule()) == 28
    assert p['legacy_hidden_provider_context'] is None
    assert p['required_future_repetitions'] is None and not p['V_eligible']


def test_single_controls_lossless_semantic_hash_and_measured_payload_ratio():
    base = base_packet()
    q0, q1, q2 = [pe.packet_for(mode, base) for mode in pe.MODES]
    assert q0.semantic_hash == q1.semantic_hash != q2.semantic_hash
    first, second = json.loads(q0.body), json.loads(q1.body)
    expanded = first.pop('lossless_expanded_parents')
    assert first == second
    from factorlab.discovery.program_syntax import CompactProgram
    assert expanded[0]['canonical_AST'] == CompactProgram.from_dict(second['parents'][0]['program']).to_program().to_dict()
    ratio = len(q1.text.encode())/len(q0.text.encode())
    assert 0 < ratio < .95
    for packet in (q0, q1):
        value = reply(packet)
        pe.decode_response(json.dumps(value), packet)
        value['programs'] *= 2
        with pytest.raises(ValueError, match='Single-candidate'):
            pe.decode_response(json.dumps(value), packet)


@pytest.mark.parametrize('field', pe.FAMILY_FIELDS)
def test_second_context_rejects_same_name_changed_semantics(field):
    packet = pe.packet_for('Q2', base_packet())
    value = reply(packet)
    _, family = pe.decode_response(json.dumps(value), packet)
    changed = copy.deepcopy(value)
    if field == 'template':
        changed[field]['base']['expressions'] = ['last(f("rv_24"))']
        changed['programs'] = [changed[field]['base']]
    elif field == 'parent_ids':
        changed[field] = ['literature1']
    else:
        changed[field] += 'changed'
    with pytest.raises(ValueError, match='semantics|template'):
        pe.decode_response(json.dumps(changed), packet, family=family)


@pytest.fixture
def actual(tmp_path, monkeypatch):
    """Real immutable runtime, ledger and wallet; empty artificial receipt source."""
    from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
    from factorlab.discovery.v7_forward import ForwardRuntime
    from factorlab.discovery.neural_forward_ticket import _bind_authority
    from factorlab.discovery.evidence import source_paths
    from factorlab import protocol
    original = protocol.ROOT
    for source in source_paths(original):
        dest = tmp_path/source.relative_to(original)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source.read_bytes())
    monkeypatch.setattr(protocol, 'ROOT', tmp_path)
    runtime = ForwardRuntime(tmp_path/'runtime', tmp_path/'feed')
    authority = MonthlyWorkAuthority(tmp_path/'wallet', run_id='efficiency-fixture', task='volatility',
        universe=['BTCUSDT'], partition_policy='standard')
    _bind_authority(runtime, authority)
    fixture = tmp_path/'artificial.json'; fixture.write_text('{}')
    yield tmp_path, runtime, authority, fixture
    runtime.close()


def freeze(actual, name='experiment'):
    root, runtime, authority, fixture = actual
    return pe.PromptEfficiencyExperiment.freeze(root/name, authority=authority,
        runtime_directory=runtime.directory, cutoffs=['2026-08-01T00:00:00+00:00','2026-09-01T00:00:00+00:00'],
        family_contract={k:v for k,v in reply(pe.packet_for('Q0',base_packet())).items() if k in pe.FAMILY_FIELDS},
        artificial_responses=fixture)


def reserve_first(obj):
    lease = pe.FrozenWorkLease.issue(300)
    identity = obj.identity
    intent = {'identity_hash': fingerprint(identity), 'slot': 0, 'work': pe.schedule()[0], 'lease': lease.to_dict()}
    pe._pair(identity, 'intents/00.json', intent)
    return obj.authority._reserve_with_lease(f'prompt_efficiency:{fingerprint(identity)}:0',
        counts=pe.schedule()[0]['counts'], lease=lease,
        binding={'directory': identity['directory'], 'identity_hash': fingerprint(identity),
            'slot': 0, 'intent_hash': fingerprint(intent)})


def test_real_authority_router_claims_original_month_and_second_directory_cannot_reset(actual):
    obj = freeze(actual)
    row = reserve_first(obj)
    assert row['actual_month'] == datetime.now(UTC).strftime('%Y-%m')
    assert row['allocation']['lane'] == 'S'
    obj.authority.fail_consumed(row['reservation_id'], 'fixture no fit')
    before = obj.authority.status(row['actual_month'])
    assert before['counts']['solve'] == 14 and before['reserved_seconds'] == 300
    with pytest.raises(ValueError, match='opportunity exists'):
        freeze(actual, 'second-directory')
    report = pe.inspect(obj.identity['directory'])
    assert report['work'][0]['result']['status'] == 'orphan_consumed'
    assert report['complete_F_denominator'] == 24 and report['complete_call_denominator'] == 6
    assert report['real_provider_calls_executed'] == 0
    assert obj.authority.status(row['actual_month']) == before
    with pytest.raises(ValueError, match='consumed'):
        obj._run_slot(0, None)


def ordinary_search(root, authority):
    from factorlab.discovery.v7_run import RunStore, V7Config, monthly_budget
    from factorlab.discovery.rolling import RollingSpec
    from factorlab.discovery.v7_memory import search_exposure_context
    from factorlab.discovery.work_authority_bridge import WorkAuthorityBridge
    store = RunStore(root/'ordinary', V7Config(evaluation_mode='joint'), RollingSpec())
    exposed = search_exposure_context('input', '2025-04-01T00:00:00+00:00', ['BTCUSDT'])
    store.exposure.record(**{k:v for k,v in exposed.items() if k!='data_id'}, role='S')
    state = {'status':'running','data_id':exposed['data_id'],'search_exposure':exposed,
        'budget':monthly_budget(V7Config(evaluation_mode='joint'),'2025-04').to_dict(),
        'cutoff':'2025-04-01T00:00:00+00:00'}
    store.save('2025-04',state,'original')
    def persist(value):
        state['budget']=value; store.save('2025-04',state,'budget')
    return WorkAuthorityBridge(store,'2025-04',state,authority,persist)


@pytest.mark.parametrize('efficiency_first', [True, False])
def test_real_normal_S_owner_conflicts_in_both_directions(actual, efficiency_first):
    root, _, authority, _ = actual
    bridge = ordinary_search(root, authority)
    try:
        if efficiency_first:
            row = reserve_first(freeze(actual))
            authority.fail_consumed(row['reservation_id'], 'no fit fixture')
            with pytest.raises(ValueError, match='Another actual owner'):
                bridge.begin_work('candidate', counts={'solve':1}, lease_seconds=30, binding={'stage':'candidate'})
        else:
            bridge.begin_work('candidate', counts={'solve':1}, lease_seconds=30,
                binding={'stage':'candidate'}).complete({'no_fit':True})
            with pytest.raises(ValueError, match='already owns'):
                freeze(actual)
        assert authority.status(datetime.now(UTC).strftime('%Y-%m'))['counts']['solve'] == (14 if efficiency_first else 1)
    finally:
        bridge.store.close()


def test_missing_exposure_or_changed_pair_fails_readonly_without_repair(actual):
    obj = freeze(actual)
    identity = obj.identity
    path = Path(identity['directory'])/'identity.json'
    path.rename(path.with_suffix('.retained'))
    with pytest.raises(ValueError, match='Missing'):
        pe.inspect(identity['directory'])
    assert not path.exists()


def test_V_and_event_context_rejected_before_original_debit(actual):
    _, runtime, authority, _ = actual
    runtime.lifecycle.exposure.record(source=runtime.feed.identity, symbols=['BTCUSDT'],
        start='2026-07-01T00:00:00+00:00', end='2026-08-01T00:00:00+00:00',
        role='V', batch='different-task-protected')
    with pytest.raises(ValueError, match='non-S'):
        freeze(actual)
    assert authority.verify()['reservations'] == 0


def test_controls_wallet_cannot_relabel_Q_modes(actual):
    root, runtime, _, fixture = actual
    from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
    other = MonthlyWorkAuthority(root/'controls-wallet', run_id='different-controls-fixture',
        task='volatility', universe=['BTCUSDT'], partition_policy='controls')
    with pytest.raises(TypeError, match='standard'):
        pe.PromptEfficiencyExperiment.freeze(root/'wrong', authority=other, runtime_directory=runtime.directory,
            cutoffs=['2026-08-01T00:00Z','2026-09-01T00:00Z'], family_contract={}, artificial_responses=fixture)
    assert other.verify()['reservations'] == 0


def test_failed_artificial_call_is_once_only_and_usage_unknown(actual, monkeypatch):
    obj = freeze(actual)
    slot = pe.schedule()[1]
    record = {'counts':slot['counts'], 'lease':pe.FrozenWorkLease.issue(300).to_dict()}
    budget = pe._Counter(record)
    packet = pe.packet_for('Q0', base_packet())
    # Empty corpus is an explicit artificial read failure, never a provider call.
    with pytest.raises(KeyError):
        pe._call(obj.identity,'Q0',0,packet,budget)
    assert budget.used['gpt_attempt'] == 1
    telemetry = pe._pair(obj.identity,'requests/Q0-0/telemetry.json')
    assert telemetry['input_tokens'] is None and telemetry['status'] == 'failed_consumed'
    with pytest.raises(ValueError, match='already consumed'):
        pe._call(obj.identity,'Q0',0,packet,budget)
    assert budget.used['gpt_attempt'] == 1
    assert pe._pair(obj.identity,'requests/Q0-0/telemetry.json') == telemetry


def test_changed_recipe_or_counter_cannot_route_to_original_S(actual):
    obj = freeze(actual); row = reserve_first(obj)
    bad = dict(row['binding'], slot=1)
    with pytest.raises(ValueError):
        pe.validate_allocation(obj.authority,row['work_key'],bad,
            pe.FrozenWorkLease.from_dict(row['lease']),row['counts'],current=False)
    with pytest.raises(ValueError, match='reservation changed'):
        pe.validate_allocation(obj.authority,row['work_key'],row['binding'],
            pe.FrozenWorkLease.from_dict(row['lease']),pe.counts(solve=13),current=False)


def test_six_artificial_responses_one_family_each_failure_and_duplicate_census(actual, monkeypatch):
    root, _, _, fixture = actual
    base = base_packet()
    values = {}
    for mode in pe.MODES:
        for i in range(2):
            values[f'{mode}-{i}'] = {'payload': reply(pe.packet_for(mode, base), family='fixture')}
    values['Q1-1'] = {'failure': 'explicit artificial transport failure'}
    fixture.write_text(json.dumps(values))
    obj = freeze(actual)
    monkeypatch.setattr(pe, '_base_packet', lambda *_: base)
    usages = []
    for mode in pe.MODES:
        budget = pe._Counter({'counts': pe.counts(family=1,gpt_attempt=2,ast=32),
                             'lease': pe.FrozenWorkLease.issue(300).to_dict()})
        pe._family(obj.identity, mode, budget)
        usages.append(budget.used)
        row = pe._pair(obj.identity, f'families/{mode}.json')
        assert row['family']['family_id'] == 'fixture'
        assert len({x['digest'] for c in row['contexts'] for x in c['materializations'] if x['program']}) == 1
        if mode == 'Q1':
            assert row['contexts'][1]['error'] and row['contexts'][1]['selected'] == []
        else:
            assert row['contexts'][1]['materializations'][0]['status'] == 'duplicate'
    assert sum(x['gpt_attempt'] for x in usages) == 6
    assert sum(x['family'] for x in usages) == 3
    assert sum(x['ast'] for x in usages) <= 96
    # These are private materializer tests, not fabricated completed wallet rows.
    assert obj.authority.verify()['reservations'] == 0
    for context in range(2):
        q0 = pe._pair(obj.identity, f'requests/Q0-{context}/request.json')
        q1 = pe._pair(obj.identity, f'requests/Q1-{context}/request.json')
        assert q0['semantic_hash'] == q1['semantic_hash']


def test_finite_Q2_invalid_duplicate_choices_are_charged_before_materializing():
    from factorlab.discovery.program_syntax import CompactProgram, parse_expression
    from factorlab.discovery.program_contract import Program
    packet = pe.packet_for('Q2', base_packet())
    value = reply(packet)
    program = CompactProgram.from_dict(value['programs'][0]).to_program().to_dict()
    program['outputs'] = [parse_expression('last(rolling_mean(f("rv_6"),2))')]
    compact = CompactProgram.from_program(Program.from_dict(program)).to_dict()
    value.update(template={'base': compact, 'slots': [{'path': ['outputs',0,'arg','window'], 'values':[2,2,0,3]}]}, programs=[compact])
    batch, _ = pe.decode_response(json.dumps(value), packet)
    charges = []
    rows = pe.materialize(batch, 'Q2', set(), reserve=lambda: charges.append('paid'))
    assert len(charges) == 4
    assert [r['status'] for r in rows] == ['valid','duplicate','duplicate','invalid','valid']
    assert len(pe._selected(rows)) == 2


def test_actual_fixed_worker_complete_artificial_Q012_original_wallet(tmp_path):
    """Full fixed child dispatch; artificial archive chronology, no provider.

    Run this in an isolated complete source checkout: child imports the same
    original sources. The archive clock is a disclosed historical-input fixture.
    Original work leases and UTC monthly wallet use the real operating-system clocks.
    """
    from factorlab import protocol
    if not tmp_path.is_relative_to(protocol.ROOT):
        pytest.skip('Complete same-source child check requires temporary root inside the checkout')
    from factorlab.discovery.v7_forward import ForwardRuntime
    from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
    from factorlab.discovery.neural_forward_ticket import _bind_authority
    from factorlab.discovery.prompt_builder import PromptIdentity
    from factorlab.discovery.joint_pool import FrozenPool
    from factorlab.discovery.known_library import library_schema
    from test_v7_rolling_registry import candles
    now = [pd.Timestamp('2026-07-31T23:30:00Z')]
    runtime = ForwardRuntime(tmp_path/'r', tmp_path/'feed', clock=lambda: now[0])
    try:
        path = tmp_path/'artificial.parquet'
        all_bars = candles('2025-12-01T00:00:00Z','2026-08-31T23:00:00Z')
        first_mask = all_bars.open_time < pd.Timestamp('2026-07-31T23:00:00Z')
        all_bars.loc[first_mask].to_parquet(path)
        runtime.feed.import_seed(path)
        now[0] = pd.Timestamp('2026-08-31T23:30:00Z')
        second = tmp_path/'artificial-august.parquet'
        all_bars.loc[~first_mask].to_parquet(second)
        runtime.feed.import_seed(second)
        now[0] = pd.Timestamp.now(tz='UTC')
        authority = MonthlyWorkAuthority(tmp_path/'wallet', run_id=fingerprint(str(tmp_path))+'-full-Q-fixture',
            task='volatility', universe=['BTCUSDT'], partition_policy='standard')
        _bind_authority(runtime, authority)
        prompt_id = PromptIdentity('volatility12h',runtime.spec.digest,FrozenPool().digest,fingerprint(library_schema()))
        base = base_packet()
        corpus = {}
        for mode in pe.MODES:
            for i in range(2):
                payload = reply(pe.packet_for(mode,base), family='same-mechanism')
                payload['identity'] = prompt_id.to_dict()
                payload['parent_ids'] = ['literature0']
                corpus[f'{mode}-{i}'] = {'payload': payload}
        fixture = tmp_path/'artificial-responses.json'; fixture.write_text(json.dumps(corpus))
        obj = pe.PromptEfficiencyExperiment.freeze(tmp_path/'pilot', authority=authority,
            runtime_directory=runtime.directory,
            cutoffs=['2026-08-01T00:00:00+00:00','2026-09-01T00:00:00+00:00'],
            family_contract={k:v for k,v in corpus['Q0-0']['payload'].items() if k in pe.FAMILY_FIELDS},
            artificial_responses=fixture)
        result = obj.run()
        (tmp_path/'report.json').write_text(json.dumps(result,indent=2))
        assert len(result['work']) == 28
        assert all(row['result']['status'] == 'completed' for row in result['work']), result
        assert result['wallet']['counts']['solve'] == 923
        assert result['wallet']['reserved_seconds'] == 3360
        assert result['wallet']['counts']['gpt_attempt'] == 6
        assert result['wallet']['counts']['family'] == 3
        assert result['wallet']['counts']['f'] == 24
        assert result['real_provider_calls_executed'] == 0
        assert all(v['unique_families'] == 1 and v['unique_formulas'] == 1 for v in result['modes'].values())
        assert all(v['reported_input_tokens'] is None for v in result['modes'].values())
        assert result['noninferiority'] == 'not_established'
        before = authority.status(obj.identity['actual_month'])
        with pytest.raises(ValueError, match='already consumed'):
            obj.run()
        assert authority.status(obj.identity['actual_month']) == before
    finally:
        runtime.close()


@pytest.mark.parametrize('late', [False, True])
def test_successor_reads_completed_witness_before_original_real_deadline(tmp_path, monkeypatch, late):
    """Real 1s clock and fsynced pair; synthetic reservation lookup, not a 300s fit."""
    from types import SimpleNamespace
    from factorlab.discovery.s_outer_context import original_clock
    from factorlab import protocol
    import time
    monkeypatch.setattr(protocol, 'ROOT', tmp_path)
    lease = pe.FrozenWorkLease.issue(1)
    identity = {'root':str(tmp_path/'original'), 'directory':str(tmp_path/'mirror'), 'clock':original_clock(30)}
    result = {'status':'awaiting_witness','slot':0,'reservation_id':'synthetic-lookup'}
    pe._pair(identity,'results/00.json',result)
    witness = {'result_hash':fingerprint(result)}
    if late:
        original_write = pe._write
        def slow_final_write(path, value):
            original_write(path, value)
            if path.parent.name == 'witnesses' and 'mirror' in path.parts:
                time.sleep(1.05)
        monkeypatch.setattr(pe, '_write', slow_final_write)
    # Real immutable dual-file publication with an artificial slow final tail.
    pe._pair(identity,'witnesses/00.json',witness)
    obj = pe.PromptEfficiencyExperiment.__new__(pe.PromptEfficiencyExperiment)
    obj.identity = identity
    obj.authority = SimpleNamespace(get=lambda _: {'lease':lease.to_dict(), 'status':'completed',
        'binding':{'identity_hash':fingerprint(identity)}})
    confirmation = obj._confirm_slot(0)
    assert confirmation['status'] == ('failed_consumed' if late else 'completed')
    assert pe._pair(identity,'confirmations/00.json') == confirmation
    with pytest.raises(ValueError, match='already consumed'):
        obj._confirm_slot(0)


def test_cli_defers_wallet_and_contract_IO_to_original_freeze(monkeypatch, capsys):
    from types import SimpleNamespace
    seen = []
    def freeze(cls, directory, **kwargs):
        seen.append((directory, kwargs))
        assert kwargs['authority'] == 'not-opened-wallet'
        assert kwargs['family_contract'] == 'not-read-contract'
        return SimpleNamespace(run=lambda: {'component': 'no work dispatched'})
    monkeypatch.setattr(pe.PromptEfficiencyExperiment, 'freeze', classmethod(freeze))
    monkeypatch.setattr(pe, 'project_path', lambda *a: pytest.fail('CLI performed original IO before freeze'))
    pe.main(['run', '--directory', 'new', '--authority', 'not-opened-wallet',
        '--runtime', 'runtime', '--cutoff', '2026-08-01T00:00:00Z',
        '--cutoff', '2026-09-01T00:00:00Z', '--family-contract', 'not-read-contract'])
    assert len(seen) == 1
    assert json.loads(capsys.readouterr().out) == {'component': 'no work dispatched'}


@pytest.mark.parametrize('path_inputs', [False, True])
def test_freeze_first_real_clock_precedes_wallet_contract_and_debits_setup(tmp_path, monkeypatch, path_inputs):
    """Component: real original7200 clock; boundary wallet verify is instrumented."""
    import time
    from factorlab.discovery import s_outer_context as outer, neural_forward_ticket as ticket
    from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
    events, clocks, remaining = [], [], []
    original_clock, check = outer.original_clock, pe._check_experiment_clock
    def clock(seconds):
        assert seconds == 7200
        value = original_clock(seconds); clocks.append(value); events.append('clock'); return value
    monkeypatch.setattr(outer, 'original_clock', clock)
    wallet = MonthlyWorkAuthority.__new__(MonthlyWorkAuthority)
    wallet.manifest = {'partition_policy': pe.frozen_policy('standard')}
    def verify(self):
        assert clocks and self is wallet; events.append('verify'); time.sleep(.035)
    monkeypatch.setattr(MonthlyWorkAuthority, 'verify', verify)
    def opened(path):
        assert clocks; events.append('open'); wallet.verify(); return wallet
    monkeypatch.setattr(ticket, 'open_existing_authority', opened)
    contract = tmp_path/'contract.json'; contract.write_text('{}')
    original_read = Path.read_text
    def read(path, *args, **kwargs):
        if path == contract:
            assert clocks; events.append('contract'); time.sleep(.035)
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read)
    def checked(identity):
        assert identity['clock'] is clocks[0]
        value = check(identity); remaining.append(value); return value
    monkeypatch.setattr(pe, '_check_experiment_clock', checked)
    class ReachedSource(Exception): pass
    def source(*args):
        events.append('source'); raise ReachedSource
    monkeypatch.setattr(ticket, '_LifecycleReader', source)
    with pytest.raises(ReachedSource):
        pe.PromptEfficiencyExperiment.freeze(tmp_path/'new', authority='wallet' if path_inputs else wallet,
            runtime_directory='runtime', cutoffs=['2026-08-01T00:00:00Z','2026-09-01T00:00:00Z'],
            family_contract=str(contract) if path_inputs else {})
    assert len(clocks) == 1 and events[0] == 'clock' and events[-1] == 'source'
    assert remaining[-1] <= 3360-.03 and remaining == sorted(remaining, reverse=True)
    if path_inputs:
        assert events.index('clock') < events.index('open') < events.index('contract')
        assert remaining[-1] <= 3360-.09


@pytest.mark.parametrize('slow_boundary', ['wallet', 'contract'])
def test_freeze_expensive_setup_rejected_by_same_original_clock(tmp_path, monkeypatch, slow_boundary):
    """Real30-second original clock component, scaled from production7200; no work/wallet debit."""
    import time
    from factorlab.discovery import s_outer_context as outer, neural_forward_ticket as ticket
    from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
    original_clock = outer.original_clock
    clocks, events = [], []
    def clock(seconds):
        assert seconds == 7200
        value = original_clock(30); clocks.append(value); events.append('clock'); return value
    monkeypatch.setattr(outer, 'original_clock', clock)
    wallet = MonthlyWorkAuthority.__new__(MonthlyWorkAuthority)
    wallet.manifest = {'partition_policy': pe.frozen_policy('standard')}
    monkeypatch.setattr(MonthlyWorkAuthority, 'verify', lambda self: None)
    def opened(path):
        assert len(clocks) == 1; events.append('wallet')
        if slow_boundary == 'wallet': time.sleep(30.05)
        return wallet
    monkeypatch.setattr(ticket, 'open_existing_authority', opened)
    contract = tmp_path/'contract.json'; contract.write_text('{}')
    original_read = Path.read_text
    def read(path, *args, **kwargs):
        if path == contract:
            assert len(clocks) == 1; events.append('contract')
            if slow_boundary == 'contract': time.sleep(30.05)
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read)
    monkeypatch.setattr(ticket, '_LifecycleReader', lambda *a: pytest.fail('Expired setup reached source or paid work'))
    with pytest.raises(TimeoutError):
        pe.PromptEfficiencyExperiment.freeze(tmp_path/'new', authority='wallet', runtime_directory='runtime',
            cutoffs=['2026-08-01T00:00:00Z','2026-09-01T00:00:00Z'], family_contract=str(contract))
    assert len(clocks) == 1 and events[0] == 'clock' and not (tmp_path/'new').exists()
    if slow_boundary == 'wallet': assert events == ['clock', 'wallet']
