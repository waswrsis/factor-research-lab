"""CLI initialization is actual bookkeeping, not implicit training/provider work."""
import json

import pytest

from factorlab.discovery import v7_cli
from factorlab.discovery.claim_protocol import ClaimProtocol
from factorlab.discovery.known_linear_reference import FrozenLinearRecipe
from factorlab.discovery.monthly_work_authority import MonthlyWorkAuthority
from factorlab.discovery.neural_forward_ticket import open_existing_authority
from factorlab.discovery.v7_forward import ForwardRuntime


@pytest.fixture
def paths(tmp_path, monkeypatch):
    from test_v7_run_factor_pool import source_root
    source_root(tmp_path, monkeypatch)
    return tmp_path / 'run', tmp_path / 'feed', tmp_path / 'authority'


def options(paths):
    run, feed, _ = paths
    return ['--run', str(run), '--feed', str(feed), '--symbol', 'BTCUSDT']


def test_initialize_one_actual_wallet_and_freeze_claims_without_training(paths, capsys):
    args = options(paths)
    assert v7_cli.main(['work-authority-init', *args, '--work-authority', str(paths[2]), '--partition-policy', 'standard']) == 0
    wallet = json.loads(capsys.readouterr().out)
    assert wallet['work_started'] is False
    from factorlab.discovery.monthly_partition_policy import canonical_policy
    assert wallet['partition_policy'] == canonical_policy('standard')
    authority = open_existing_authority(paths[2])
    assert type(authority) is MonthlyWorkAuthority and authority.verify()['reservations'] == 0
    assert authority.manifest['partition_policy'] == canonical_policy('standard')
    assert v7_cli.main(['claims-freeze', *args, '--start', '2030-01-01T00:00:00Z']) == 0
    frozen = json.loads(capsys.readouterr().out)
    assert frozen['training_started'] is False and frozen['formal_claims_enabled'] is False
    runtime = ForwardRuntime(paths[0], paths[1])
    try:
        protocol = ClaimProtocol(runtime.lifecycle)
        assert protocol.digest == frozen['protocol_digest']
        assert protocol.endpoints['library_skill'].reference == runtime.recipe_for(FrozenLinearRecipe('HAR_cal', runtime.spec))
        assert runtime.lifecycle.db.execute('SELECT count(*) FROM jobs').fetchone()[0] == 0
    finally:
        runtime.close()
    assert authority.verify()['reservations'] == 0


def test_typed_fit_requires_shared_authority_no_legacy_fallback(paths, tmp_path, capsys):
    runtime = ForwardRuntime(paths[0], paths[1])
    recipe = runtime.recipe_for(FrozenLinearRecipe('K0', runtime.spec))
    runtime.close()
    file = tmp_path / 'recipe.json'
    file.write_text(json.dumps(recipe), encoding='utf-8')
    with pytest.raises(ValueError, match='actual shared'):
        v7_cli.main(['fit', *options(paths), '--recipe', str(file), '--cutoff', '2026-10-01T00:00:00Z'])
    assert v7_cli.main(['work-authority-init', *options(paths), '--work-authority', str(paths[2]), '--partition-policy', 'standard']) == 0
    capsys.readouterr()
    with pytest.raises(ValueError, match='unmetered legacy'):
        v7_cli.main(['fit', *options(paths), '--cutoff', '2026-10-01T00:00:00Z'])
    assert open_existing_authority(paths[2]).verify()['reservations'] == 0


def test_service_rejects_adding_claim_protocol_after_session_freeze(paths):
    from factorlab.discovery.claim_protocol import FrozenClaimEndpoint
    from factorlab.discovery.v7_service import ForwardService, ServiceConfig
    runtime = ForwardRuntime(paths[0], paths[1])
    service = ForwardService(runtime, config=ServiceConfig(poll=False))
    try:
        k0 = runtime.recipe_for()
        har = runtime.recipe_for(FrozenLinearRecipe('HAR_cal', runtime.spec))
        ClaimProtocol.freeze(runtime.lifecycle, [FrozenClaimEndpoint.library('2030-01-01T00:00:00Z', k0, har),
                                                 FrozenClaimEndpoint.algorithm('2030-01-01T00:00:00Z', k0)])
        with pytest.raises(ValueError, match='frozen six-month'):
            service._plan_month(runtime.lifecycle.now())
        assert runtime.lifecycle.db.execute('SELECT count(*) FROM jobs').fetchone()[0] == 0
    finally:
        service.close()
