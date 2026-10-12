"""No clock, kernel, resource or formal-permission mock at ticket boundary.

Only prior observation clocks and the artificial prerecorded loss population
belong to the quarantine fixture. Work clocks, archive receipt and filesystem
publication clocks remain physical. This intentionally proves closed gates.
"""
import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from test_v7_quarantine_ticket_fixture import case, recorded_pairs  # noqa: F401

from factorlab.discovery.joint_pool import FrozenPool, PoolMember
from factorlab.discovery.known_library import library_schema
from factorlab.discovery.neural_forward import FrozenNeuralRecipe
from factorlab.discovery.neural_forward_ticket import issue_forward_ticket
from factorlab.discovery.nonlinear_channel import ResidualConfig
from factorlab.discovery.pool_forward import issue_pool_ticket
from factorlab.protocol import fingerprint


def test_real_pool_and_neural_ticket_boundaries(case):  # noqa: F811
    registry, runtime, wallet, _permit, clock = case
    lc = runtime.lifecycle
    now = pd.Timestamp.now(tz='UTC')
    cutoff = now.normalize().replace(day=1)
    start = cutoff-pd.DateOffset(months=5)
    k0 = FrozenPool(role='reference', version='K0_v1')
    program = {'version': 'causal-operator-program-v1', 'feature_names': library_schema()['feature_names'],
               'parameters': {}, 'outputs': [{'op': 'last', 'arg': {'op': 'feature', 'name': 'rv_24'}}]}
    pool = FrozenPool((PoolMember(program, 2, 'fixed'),), reference_pool=k0,
                      reference_pipeline_id=k0.pipeline_id)
    candidate = runtime.recipe_for(pool)
    reference = runtime.recipe_for(k0)
    neural = FrozenNeuralRecipe(pool, runtime.spec, ResidualConfig(enabled=True))
    assert len(neural.config.seeds) == 3 and neural.config.epochs == 4
    lc.exposure.record(source=lc.plan.source_id, symbols=lc.plan.symbols,
        start=clock[0]-pd.DateOffset(months=3), end=clock[0], role='S', batch='A')
    lc.freeze_batch(batch='A', ordinal=1, start=start, candidates=[{**candidate, 'complexity': 1}],
                    reference=reference, evidence_sources=[{'start': (clock[0]-pd.DateOffset(months=3)).isoformat(),
                                                         'end': clock[0].isoformat()}])
    cstart = start+pd.DateOffset(months=1)
    clock[0] = cstart+pd.Timedelta(seconds=5)
    recorded_pairs(lc, start, cstart-pd.Timedelta(hours=13))
    assert lc.select_validation('A')['selected']['candidate_id'] == candidate['pipeline_id']
    lc.reserve_selected_confirmation('A')
    cend = cstart+pd.DateOffset(months=3)
    clock[0] = cend+pd.Timedelta(days=2)
    recorded_pairs(lc, cstart+pd.Timedelta(hours=1), cend)
    result = lc.conclude_confirmation('A')
    assert result['quarantined_decision']['accepted'] and not result['supported']
    clock[0] = pd.Timestamp.now(tz='UTC')
    publication = lc.confirmation_publication('A')
    path = lc.directory/'claim_publications'/f'{fingerprint("A")}.json'
    assert pd.Timestamp(publication['published_at']) < cutoff
    assert path.stat().st_mtime_ns > cutoff.value
    assert lc.adoption_pipeline(cutoff, reference) == reference
    before = wallet.verify()
    job = lc.plan_fit(cutoff, 'A0', candidate)
    with pytest.raises(ValueError, match='month-start adoption policy'):
        issue_pool_ticket(runtime, wallet, job, pool)
    assert wallet.verify() == before
    # A complete but unselected neural learner cannot borrow the pool's C.
    neural_job = lc.plan_fit(cutoff, 'A0_neural_invalid', runtime.recipe_for(neural))
    with pytest.raises(ValueError):
        issue_forward_ticket(runtime, wallet, neural_job, neural, purpose='A0', gates={})
    assert wallet.verify() == before
    # Real archive import now cannot create historical as-of availability.
    runtime.feed.clock = lambda: pd.Timestamp.now(tz='UTC')
    times = pd.date_range(cutoff-pd.DateOffset(months=6), cutoff, freq='h', inclusive='left')
    price = np.exp(4+np.sin(np.arange(len(times))/20)*.01)
    bars = pd.DataFrame({'symbol': 'BTCUSDT', 'open_time': times,
        'available_at': times+pd.Timedelta(hours=1, seconds=1), 'open': price, 'close': price,
        'high': price*1.003, 'low': price/1.003})
    seed = registry.directory.parent/'artificial-bars.parquet'
    bars.to_parquet(seed)
    runtime.feed.import_seed(seed)
    # A future LC observation can select original C, but cannot advance archive
    # or lease clocks. The original prepaid snapshot fails, retaining quota.
    future = cutoff+pd.DateOffset(months=1)
    clock[0] = future
    assert lc.adoption_pipeline(future, reference)['pipeline_id'] == candidate['pipeline_id']
    future_job = lc.plan_fit(future, 'A0', candidate)
    with pytest.raises(ValueError, match='future receipt boundary'):
        issue_pool_ticket(runtime, wallet, future_job, pool)
    after = wallet.verify()
    assert after != before
    with wallet._transaction() as db:
        rows = db.execute('SELECT id FROM reservations').fetchall()
    assert len(rows) == 1
    record = wallet.get(rows[0][0])
    assert record['status'] == 'failed_consumed' and record['counts']['solve'] > 0
    assert pd.Timestamp(record['lease']['wall_started_at']).month == now.month
    with pytest.raises(ValueError, match='Partial pool forward ticket'):
        issue_pool_ticket(runtime, wallet, future_job, pool)
    assert wallet.verify() == after
    evidence = {'observed_physical_at': datetime.now(UTC).isoformat(),
        'actual_cutoff': cutoff.isoformat(), 'publication': publication,
        'publication_mtime_ns': path.stat().st_mtime_ns, 'pool': candidate,
        'neural_pipeline': runtime.recipe_for(neural), 'wallet_before': before, 'wallet_after': after,
        'consumed_record': record, 'genuine_resource_proof': False, 'ticket_issued': False,
        'model_fitted': False, 'scope': 'real permissions and wallet; artificial recorded predictions'}
    (registry.directory.parent/'ticket-boundaries.json').write_text(json.dumps(evidence, indent=2))
