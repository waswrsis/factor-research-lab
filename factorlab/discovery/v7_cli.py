"""D-contained construction CLI. No network/data download is implicit."""
from __future__ import annotations

import argparse
import json

import pandas as pd

from factorlab.protocol import fingerprint, project_path


def _bounded_pump_count(value):
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError('Pump count must be an integer from 1 to 7200') from error
    if not 1 <= result <= 7200:
        raise argparse.ArgumentTypeError('Pump count must be an integer from 1 to 7200')
    return result


def _original_trial_runtime(permit):
    """Execution commands open verified originals; inspect never calls this."""
    from .full_calibration_registry import _read
    from .v7_forward import ForwardConfig, ForwardRuntime
    from .v7_lifecycle import LifecyclePlan
    permit.verify()
    identity = _read(permit.paths['runtime'] / 'forward.json')['identity']
    config = dict(identity['config'])
    config['symbols'] = tuple(config['symbols'])
    plan = dict(identity['lifecycle'])
    plan['symbols'] = tuple(plan['symbols'])
    return ForwardRuntime(permit.paths['runtime'], permit.paths['feed'],
        config=ForwardConfig(**config), lifecycle_plan=LifecyclePlan(**plan))


def _existing_trial_generator(registry, directory):
    """Open an existing generator identity; no missing identity is recreated."""
    from .finite_market_oracle import FiniteGenerator, FiniteMarketLaw
    from .full_calibration_registry import _read
    directory = project_path(directory)
    original = _read(directory / 'identity.json')
    law = dict(original['law'])
    law['symbols'] = tuple(law['symbols'])
    generator = FiniteGenerator(directory, FiniteMarketLaw(**law), registry.plan.acceptance)
    if generator.identity != original:
        raise ValueError('Existing generator changed during open')
    generator.verify_registry(registry)
    return generator


def _enroll_original_trial(registry, cell, repetition, oracle_index, *, estimand_slot=None):
    """Explicitly initialize one fresh grid namespace from its frozen scope.

    An interrupted partial namespace is preserved and rejected. Opening an
    already enrolled trial verifies original evidence without constructors.
    """
    from .full_calibration_registry import QuarantinedTrialPermit
    from .monthly_work_authority import MonthlyWorkAuthority
    from .v7_feed import ReceiptArchive
    from .v7_forward import ForwardConfig, ForwardRuntime
    from .v7_lifecycle import LifecyclePlan
    coordinates = {'oracle_index': oracle_index}
    if estimand_slot is not None:
        coordinates['estimand_slot'] = estimand_slot
    trial = registry.identity_for(cell, repetition, **coordinates)
    paths = registry.paths(cell, repetition, **coordinates)
    permit = QuarantinedTrialPermit(registry, trial)
    if estimand_slot is not None:
        from .fixed_recipe_oracle import fixed_enrollment_preflight
        fixed_enrollment_preflight(permit)
    if (paths['root'] / 'enrollment.json').exists():
        return permit.verify()
    if paths['root'].exists() and any(paths['root'].iterdir()):
        raise ValueError('Partial original trial cannot be reset by enrollment')
    scope = registry.plan.runtime_scope
    config = dict(scope['forward'])
    config['symbols'] = tuple(config['symbols'])
    config = ForwardConfig(**config)
    feed = ReceiptArchive(paths['feed'], config.symbols)
    try:
        plan = dict(scope['lifecycle_policy'])
        plan['symbols'] = tuple(plan['symbols'])
        plan = LifecyclePlan(source_id=feed.identity, **plan)
    finally:
        feed.close()
    runtime = ForwardRuntime(paths['runtime'], paths['feed'], config=config, lifecycle_plan=plan)
    try:
        budget = scope['work_budget']
        wallet = MonthlyWorkAuthority(paths['authority'],
            run_id=fingerprint({'registry': registry.digest, 'trial': trial['trial_id']}),
            task=budget['task'], universe=budget['universe'], limits=budget['limits'],
            monthly_seconds=budget['monthly_seconds'], partition_policy=budget['partition_policy'])
        return registry.enroll(runtime, wallet, cell, repetition, **coordinates).verify()
    finally:
        runtime.close()


def _research_from_file(args):
    """Prepay the original feature/baseline lease before the first parquet read."""
    from dataclasses import asdict

    from .rolling import RollingSpec
    from .v7_run import V7Config, run_month
    fields = json.loads(project_path(args.config).read_text('utf-8'))
    fields['symbols'] = tuple(fields['symbols'])
    config = V7Config(**fields)
    spec = RollingSpec(availability_delay_seconds=args.availability_delay_seconds,
                       window_days=args.window_days)
    if args.offline_diagnostic:
        if args.work_authority or args.work_schedule or args.feed_directory or args.exposure_path:
            raise ValueError('Offline diagnostic cannot use a live feed, shared ledger or authority flags')
        return run_month(args.run,pd.read_parquet(project_path(args.bars)),args.cutoff,config=config,spec=spec)
    if not all((args.work_authority,args.work_schedule,args.feed_directory,args.exposure_path)):
        raise ValueError('Research requires its actual work authority, work schedule, feed and shared ledger; offline requires --offline-diagnostic')
    from .monthly_schedule import load_schedule
    from .neural_forward_ticket import (
        _LifecycleReader,
        _read,
        _verify_authority_binding,
        open_existing_authority,
    )
    from .s_preparation import SPreparationEnvelope
    from .work_lease import FrozenWorkLease
    origin = FrozenWorkLease.issue(300)
    authority = open_existing_authority(args.work_authority)
    schedule = load_schedule(args.work_schedule, config=config, month=str(args.cutoff)[:7])
    from .scheduled_search import validate_schedule_authority
    validate_schedule_authority(authority, schedule)
    if fingerprint(schedule.to_dict()['config']) != fingerprint(asdict(config)):
        raise ValueError('CLI work schedule differs from the exact research config')
    _, cutoff = spec.window(args.cutoff)
    feed = project_path(args.feed_directory)
    identity = _read(feed / 'identity.json')
    if (set(identity) != {'source_id','symbols'} or identity['symbols'] != list(config.symbols)
            or not (feed / 'receipts.sqlite').is_file()):
        raise ValueError('CLI requires its actual existing receipt archive')
    exposure = project_path(args.exposure_path)
    if exposure.name != 'exposure.sqlite' or not exposure.is_file():
        raise ValueError('CLI requires the exact existing lifecycle exposure ledger')
    reader = _LifecycleReader(exposure.parent)
    observation_prefix = None
    try:
        result = reader.verify()
        _verify_authority_binding(reader,authority,result['operations'])
        if reader.plan.source_id != identity['source_id']:
            raise ValueError('CLI feed differs from actual lifecycle source')
        if reader.plan.training_window_days != spec.window_days:
            raise ValueError('Research CLI changed the original training-window rule')
        from .quarantined_observation import current_binding
        observation_prefix = current_binding(reader)
    finally:
        reader.close()
    source = {'feed_identity':identity['source_id'],'feed_directory':str(feed),
              'exposure_path':str(exposure),'cutoff':cutoff.isoformat()}
    if observation_prefix is not None:
        source['observation_prefix'] = observation_prefix
    prep = SPreparationEnvelope.prepare(authority,args.run,cutoff.strftime('%Y-%m'),
        config=asdict(config),spec=asdict(spec),source_reference=source,
        schedule_hash=schedule.digest,clock_scope='before_local_source_load',lease=origin)
    if not prep._executable:
        raise ValueError('CLI preparation is readback only; local source cannot reload as new work')
    try:
        bars = pd.read_parquet(project_path(args.bars))
        return run_month(args.run,bars,cutoff,config=config,spec=spec,exposure_path=exposure,
            feed_directory=feed,authority=authority,schedule=schedule,preparation=prep)
    except BaseException as error:
        try:
            record = prep.baseline_record()
            if record['status'] not in {'completed','failed_consumed','interrupted_consumed'}:
                authority.fail_consumed(record['reservation_id'],'S_CLI_source:' + type(error).__name__)
        except BaseException as failure:  # noqa: BLE001 -- keep the initiating error and spent credit.
            error.add_note('CLI source failure accounting also failed: ' + type(failure).__name__)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description='V7 frozen rolling factor research construction')
    sub = parser.add_subparsers(dest='command', required=True)
    research = sub.add_parser('research')
    research.add_argument('--bars', required=True)
    research.add_argument('--run', required=True)
    research.add_argument('--cutoff', required=True)
    research.add_argument('--config', required=True)
    research.add_argument('--exposure-path')
    research.add_argument('--feed-directory')
    research.add_argument('--work-authority')
    research.add_argument('--work-schedule', help='Exact FrozenSWorkSchedule JSON, including explicit holdbacks')
    research.add_argument('--offline-diagnostic', action='store_true', help='Explicit legacy offline research only')
    research.add_argument('--availability-delay-seconds', type=int, default=1)
    research.add_argument('--window-days', type=int, choices=(90,))
    replay = sub.add_parser('replay')
    replay.add_argument('--bars', required=True)
    replay.add_argument('--run', required=True)
    replay.add_argument('--cutoffs', nargs='+', required=True)
    replay.add_argument('--task', choices=['volatility', 'first_event'], default='volatility')
    replay.add_argument('--symbol', action='append', required=True)
    replay.add_argument('--availability-delay-seconds', type=int, default=1)
    replay.add_argument('--window-days', type=int, choices=(90,))
    inspect = sub.add_parser('inspect')
    inspect.add_argument('--run', required=True)
    seed = sub.add_parser('seed')
    seed.add_argument('--feed', required=True)
    seed.add_argument('--bars', required=True)
    seed.add_argument('--symbol', action='append', required=True)
    prepare_seed = sub.add_parser('prepare-seed')
    prepare_seed.add_argument('--run', required=True, help='Immutable bounded public archive download job')
    prepare_seed.add_argument('--symbol', action='append', required=True)
    prepare_seed.add_argument('--as-of')
    prepare_seed.add_argument('--months', type=int, default=8)
    prepare_seed.add_argument('--max-new-archives', type=int, default=1)
    prepare_seed.add_argument('--allow-daily-bridge', action='store_true',
                              help='Freeze explicit daily-archive recovery policy; never backfill forecasts')
    daily = sub.add_parser('prepare-daily-bridge')
    daily.add_argument('--run', required=True)
    daily.add_argument('--symbol', action='append', required=True)
    daily.add_argument('--start', required=True)
    daily.add_argument('--end')
    daily.add_argument('--as-of')
    daily.add_argument('--max-new-archives', type=int, default=1)
    daily_import = sub.add_parser('import-daily-bridge')
    daily_import.add_argument('--run', required=True)
    daily_import.add_argument('--feed', required=True)
    daily_import.add_argument('--symbol', action='append', required=True)
    for name in ('fit', 'tick', 'forward-status', 'service', 'autonomous', 'work-authority-init', 'claims-freeze'):
        command = sub.add_parser(name)
        command.add_argument('--run', required=True)
        command.add_argument('--feed', required=True)
        command.add_argument('--symbol', action='append', required=True)
        command.add_argument('--plan', help='Frozen LifecyclePlan JSON; source_id must match feed')
        command.add_argument('--research-spec', help='Explicit LayeredResearchSpec JSON for a fresh uncertified run')
        command.add_argument('--polling-pages', type=int, choices=(1, 2, 3, 4), default=1)
        command.add_argument('--window-days', type=int, choices=(90,),
                             help='Continuous 90-day window and original 15-day fit cadence')
        if name in ('fit', 'service', 'autonomous', 'work-authority-init'):
            command.add_argument('--work-authority', required=name == 'work-authority-init',
                                 help='One existing shared monthly wallet; only init creates it')
        if name in ('fit', 'service', 'autonomous'):
            command.add_argument('--neural-gates', help='Fixed complete-pipeline gate map JSON')
        if name == 'work-authority-init':
            command.add_argument('--partition-policy', required=True,
                                 choices=('standard', 'controls', 'diagnostic'),
                                 help='Immutable actual monthly policy; diagnostic has no protected holdbacks')
        if name == 'fit':
            command.add_argument('--cutoff', required=True)
            command.add_argument('--stream', default='A0')
            recipe_options = command.add_mutually_exclusive_group()
            recipe_options.add_argument('--pool', help='Legacy FrozenPool recipe JSON')
            recipe_options.add_argument('--recipe', help='Complete typed pipeline JSON')
        elif name == 'claims-freeze':
            command.add_argument('--start', required=True, help='Prospective UTC month start for both six-month claims')
        elif name == 'tick':
            command.add_argument('--poll', action='store_true')
        elif name in ('service', 'autonomous'):
            command.add_argument('--config', required=True)
            if name == 'autonomous':
                command.add_argument('--research-config', required=True)
                command.add_argument('--research-work-schedule', required=True,
                                     help='Frozen actual shared-wallet monthly S schedule')
    calibrate = sub.add_parser('calibrate')
    calibrate.add_argument('--run', required=True)
    calibrate.add_argument('--plan')
    calibrate.add_argument('--max-new-trials', type=int, default=1)
    full_calibrate = sub.add_parser('full-calibrate')
    full_calibrate.add_argument('--run', required=True)
    full_calibrate.add_argument('--plan')
    full_calibrate.add_argument('--max-new-trials', type=int, default=1,
                                help='Bounded work units: experiment or independent oracle path')
    calibration_status = sub.add_parser('full-calibration-status')
    calibration_status.add_argument('--run', required=True)
    for name in ('trial-registry-freeze', 'trial-registry-status'):
        registry_command = sub.add_parser(name)
        registry_command.add_argument('--registry', required=True)
        if name == 'trial-registry-freeze':
            registry_command.add_argument('--plan', required=True)
    for name in ('trial-enroll', 'trial-controller-freeze', 'trial-controller-status', 'trial-controller-pump', 'trial-controller-run',
                 'trial-observation-bind', 'trial-observation-status', 'trial-observation-advance'):
        controller = sub.add_parser(name)
        controller.add_argument('--registry', required=True)
        controller.add_argument('--cell', required=True)
        controller.add_argument('--repetition', type=int, required=True)
        controller.add_argument('--oracle-index', type=int)
        if name == 'trial-enroll':
            controller.add_argument('--estimand-slot', choices=('library_skill', 'factor1', 'factor2', 'factor3'),
                                    help='Separate registered fixed-recipe oracle replica; requires --oracle-index')
        if name == 'trial-controller-freeze':
            controller.add_argument('--schedule', required=True)
        elif name == 'trial-controller-run':
            controller.add_argument('--max-pumps', type=_bounded_pump_count, default=7200)
        elif name == 'trial-observation-bind':
            controller.add_argument('--generator', required=True)
        elif name == 'trial-observation-advance':
            controller.add_argument('--max-events', type=int, default=96, choices=range(1, 97))
    certificate_produce = sub.add_parser('trial-certificate-produce')
    certificate_produce.add_argument('--registry', required=True)
    certificate_status = sub.add_parser('trial-certificate-status')
    certificate_status.add_argument('--registry', required=True)
    certificate_status.add_argument('--artifact-digest', required=True)
    certificate_publish = sub.add_parser('trial-certificate-publish')
    certificate_publish.add_argument('--registry', required=True)
    certificate_publish.add_argument('--directory', required=True)
    certificate_publish.add_argument('--artifact-digest', required=True)
    for name in ('trial-risk-prepare', 'trial-risk-produce', 'trial-risk-status'):
        risk = sub.add_parser(name)
        risk.add_argument('--registry', required=True)
        risk.add_argument('--cell', required=True)
        risk.add_argument('--repetition', type=int, required=True)
        if name == 'trial-risk-status':
            risk.add_argument('--artifact-digest', required=True)
    for name in ('cold-refit-status', 'cold-refit-run'):
        cold = sub.add_parser(name)
        cold.add_argument('--directory', required=True)
        if name == 'cold-refit-run':
            cold.add_argument('--atom', type=int, required=True)
    cold_freeze = sub.add_parser('cold-refit-freeze')
    cold_freeze.add_argument('--registry', required=True)
    cold_freeze.add_argument('--cell', required=True)
    cold_freeze.add_argument('--repetition', type=int, required=True)
    cold_freeze.add_argument('--generator', required=True)
    cold_freeze.add_argument('--slot', required=True,
                             help='Original selected slot, or typed_neural_<catalogue digest>')
    for name in ('policy-execution-status', 'policy-execution-run'):
        policy_execution = sub.add_parser(name)
        policy_execution.add_argument('--registry', required=True)
        policy_execution.add_argument('--cell', required=True)
        policy_execution.add_argument('--repetition', type=int, required=True)
        if name == 'policy-execution-run':
            policy_execution.add_argument('--oracle-index', type=int, required=True)
            policy_execution.add_argument('--max-pumps', type=_bounded_pump_count, default=7200)
    for name in ('fixed-oracle-freeze', 'fixed-oracle-status', 'fixed-oracle-pump', 'fixed-oracle-run', 'fixed-oracle-report'):
        fixed_oracle = sub.add_parser(name)
        fixed_oracle.add_argument('--registry', required=True)
        fixed_oracle.add_argument('--cell', required=True)
        fixed_oracle.add_argument('--repetition', type=int, required=True)
        fixed_oracle.add_argument('--oracle-index', type=int, required=True)
        fixed_oracle.add_argument('--slot', required=True,
                                  choices=('library_skill', 'factor1', 'factor2', 'factor3'))
        if name == 'fixed-oracle-freeze':
            fixed_oracle.add_argument('--generator', required=True)
    probe = sub.add_parser('probe')
    probe.add_argument('--bars', required=True)
    probe.add_argument('--run', required=True)
    probe.add_argument('--cutoff', required=True)
    probe.add_argument('--pool')
    controls = sub.add_parser('controls')
    controls.add_argument('--bars', required=True)
    controls.add_argument('--run', required=True)
    controls.add_argument('--cutoff', required=True)
    controls.add_argument('--config', required=True)
    controls.add_argument('--gpt', action='store_true', help='Explicitly include the real Codex proposal arm')
    shared_controls = sub.add_parser('shared-controls')
    shared_controls.add_argument('--bars', required=True)
    shared_controls.add_argument('--run', required=True)
    shared_controls.add_argument('--cutoff', required=True)
    shared_controls.add_argument('--symbol', action='append', required=True)
    shared_controls.add_argument('--availability-delay-seconds', type=int, default=1)
    shared_controls.add_argument('--window-days', type=int, choices=(90,))
    shared_controls.add_argument('--work-authority', required=True)
    shared_controls.add_argument('--work-schedules', required=True,
                                 help='JSON mapping local/random/gpt to their fixed work schedules')
    shared_controls.add_argument('--feed-directory', required=True)
    shared_controls.add_argument('--exposure-path', required=True)
    shared_controls.add_argument('--gpt', action='store_true', required=True,
                                 help='Explicit real GPT arm within one shared monthly budget')
    shared_status = sub.add_parser('shared-controls-status')
    shared_status.add_argument('--run', required=True)
    target = sub.add_parser('variance-target')
    target.add_argument('--bars', required=True)
    target.add_argument('--origins', required=True)
    target.add_argument('--run', required=True)
    target.add_argument('--as-of', required=True)
    args = parser.parse_args(argv)
    from .rolling import RollingSpec
    from .v7_run import V7Config, inspect_run, replay_baseline
    from .v7_storage import immutable_record as immutable_json
    if args.command == 'prepare-seed':
        from .v7_seed import prepare_seed as download_seed
        bridge_options = {'allow_daily_bridge': True} if args.allow_daily_bridge else {}
        value = download_seed(args.run, symbols=tuple(args.symbol), as_of=args.as_of,
                              months=args.months, max_new_archives=args.max_new_archives, **bridge_options)
    elif args.command == 'prepare-daily-bridge':
        from .v7_daily import prepare_daily_bridge
        value = prepare_daily_bridge(args.run, symbols=tuple(args.symbol), start=args.start,
            end=args.end, as_of=args.as_of, max_new_archives=args.max_new_archives)
    elif args.command == 'import-daily-bridge':
        from .v7_daily import import_daily_bridge
        from .v7_feed import ReceiptArchive
        feed = ReceiptArchive(args.feed, tuple(args.symbol))
        try:
            value = import_daily_bridge(args.run, feed)
        finally:
            feed.close()
    elif args.command == 'seed':
        from .v7_feed import ReceiptArchive
        feed = ReceiptArchive(args.feed, tuple(args.symbol))
        try:
            value = feed.import_seed(args.bars)
        finally:
            feed.close()
    elif args.command in ('fit', 'tick', 'forward-status', 'service', 'autonomous', 'work-authority-init', 'claims-freeze'):
        from .joint_pool import FrozenPool
        from .v7_forward import ForwardConfig, ForwardRuntime
        from .v7_lifecycle import LifecyclePlan
        plan = None
        if args.plan:
            fields = json.loads(project_path(args.plan).read_text('utf-8'))
            fields['symbols'] = tuple(fields['symbols'])
            plan = LifecyclePlan(**fields)
        window_days = args.window_days if args.window_days is not None else (
            plan.training_window_days if plan is not None else None)
        research_kwargs = {}
        if args.research_spec:
            from .layered_research import LayeredResearchSpec
            research_kwargs['research_spec'] = LayeredResearchSpec(**json.loads(
                project_path(args.research_spec).read_text('utf-8')))
        runtime = ForwardRuntime(args.run, args.feed, config=ForwardConfig(symbols=tuple(args.symbol),
                                 polling_pages=args.polling_pages, training_window_days=window_days,
                                 refit_interval_days=15 if window_days is not None else None),
                                 lifecycle_plan=plan, **research_kwargs)
        try:
            authority, gates = None, None
            if getattr(args, 'work_authority', None) and args.command != 'work-authority-init':
                from .neural_forward_ticket import open_existing_authority
                authority = open_existing_authority(args.work_authority)
            if getattr(args, 'neural_gates', None):
                gates = json.loads(project_path(args.neural_gates).read_text('utf-8'))
                if not isinstance(gates, dict):
                    raise ValueError('Neural gates require a complete pipeline map')
            if args.command == 'work-authority-init':
                from .monthly_work_authority import MonthlyWorkAuthority
                from .neural_forward_ticket import _bind_authority
                authority = MonthlyWorkAuthority(args.work_authority,
                    run_id='v7-forward:' + fingerprint(runtime.identity), task='volatility', universe=args.symbol,
                    partition_policy=None if args.partition_policy == 'diagnostic' else args.partition_policy)
                _bind_authority(runtime, authority)
                value = {'status': 'initialized', 'directory': str(authority.directory),
                         'manifest_hash': authority.manifest_hash, 'work_started': False,
                         'partition_policy': authority.manifest['partition_policy']}
            elif args.command == 'claims-freeze':
                from .claim_protocol import ClaimProtocol, FrozenClaimEndpoint
                from .known_linear_reference import FrozenLinearRecipe
                k0 = runtime.recipe_for()
                reference = runtime.recipe_for(FrozenLinearRecipe('HAR_cal', runtime.spec))
                protocol = ClaimProtocol.freeze(runtime.lifecycle, [
                    FrozenClaimEndpoint.library(args.start, k0, reference),
                    FrozenClaimEndpoint.algorithm(args.start, k0,
                        research_spec_hash=(None if not args.research_spec
                                            else research_kwargs['research_spec'].digest))])
                value = {'status': 'frozen', 'protocol_digest': protocol.digest,
                         'endpoints': [e.to_dict() for e in protocol.endpoints.values()],
                         'formal_claims_enabled': False, 'training_started': False}
            elif args.command == 'fit':
                if args.recipe:
                    from .v7_service import parse_frozen_recipe
                    full = json.loads(project_path(args.recipe).read_text('utf-8'))
                    pool = parse_frozen_recipe(runtime, full)
                    if runtime.recipe_for(pool) != full:
                        raise ValueError('Typed CLI recipe differs from the complete runtime pipeline')
                else:
                    pool = FrozenPool.from_dict(json.loads(project_path(args.pool).read_text('utf-8'))) if args.pool else None
                kwargs = {'authority': authority} if authority is not None else {}
                if gates is not None:
                    key = runtime.recipe_for(pool)['pipeline_id']
                    if key not in gates:
                        raise ValueError('Neural CLI fit lacks its exact registered gate entry')
                    kwargs['gates'] = gates[key]
                value = runtime.fit_month(args.cutoff, stream=args.stream, pool=pool, **kwargs)
            elif args.command == 'tick':
                value = runtime.tick(poll=args.poll)
            elif args.command == 'forward-status':
                value = runtime.status()
            else:
                from .v7_service import RollingService, ServiceConfig
                config = ServiceConfig(**json.loads(project_path(args.config).read_text('utf-8')))
                kwargs = {'work_authority': authority} if authority is not None else {}
                if gates is not None:
                    kwargs['neural_gates'] = gates
                runner = RollingService(runtime, config=config, **kwargs)
                scheduler = None
                try:
                    if args.command == 'autonomous':
                        from .v7_autonomous import (
                            AutonomousConfig,
                            AutonomousService,
                            ResearchScheduler,
                        )
                        fields = json.loads(project_path(args.research_config).read_text('utf-8'))
                        research_fields = fields.pop('research', {})
                        research_fields.setdefault('symbols', tuple(args.symbol))
                        research_fields['symbols'] = tuple(research_fields['symbols'])
                        fields['research'] = V7Config(**research_fields)
                        from .monthly_schedule import load_schedule
                        schedule = load_schedule(args.research_work_schedule, config=fields['research'],
                                                 month=pd.Timestamp.now(tz='UTC').strftime('%Y-%m'))
                        scheduler = ResearchScheduler(runtime, config=AutonomousConfig(**fields),
                                                      authority=authority, schedule=schedule)
                        value = AutonomousService(runner, scheduler).run()
                    else:
                        value = runner.run()
                finally:
                    if scheduler:
                        scheduler.close()
                    runner.close(close_runtime=False)
            if research_kwargs and isinstance(value, dict):
                value['research_spec_hash'] = research_kwargs['research_spec'].digest
                value['statistical_mode'] = 'prospective_uncertified_research'
                value['whole_agent_power_certified'] = False
        finally:
            runtime.close()
    elif args.command in ('trial-registry-freeze', 'trial-registry-status'):
        from .full_calibration_registry import FullTrialPlan, FullTrialRegistry, _read
        if args.command == 'trial-registry-freeze':
            registry = FullTrialRegistry(args.registry, FullTrialPlan.from_dict(_read(project_path(args.plan))))
        else:
            registry = FullTrialRegistry.inspect(args.registry)
        value = {**registry.verify(), 'read_only': args.command == 'trial-registry-status',
                 'work_started': False}
    elif args.command in ('trial-enroll', 'trial-controller-freeze', 'trial-controller-status', 'trial-controller-pump', 'trial-controller-run',
                          'trial-observation-bind', 'trial-observation-status', 'trial-observation-advance'):
        from dataclasses import asdict

        from .full_calibration_registry import FullTrialRegistry, QuarantinedTrialPermit
        from .full_trial_controller import FullControllerSchedule, FullTrialController
        registry = FullTrialRegistry.inspect(args.registry)
        coordinates = {'oracle_index': args.oracle_index}
        if args.command == 'trial-enroll' and args.estimand_slot is not None:
            coordinates['estimand_slot'] = args.estimand_slot
        permit = QuarantinedTrialPermit(registry, registry.identity_for(args.cell, args.repetition,
                                                                       **coordinates))
        if args.command == 'trial-enroll':
            value = {**_enroll_original_trial(registry, args.cell, args.repetition, args.oracle_index,
                                             estimand_slot=args.estimand_slot),
                     'work_started': False}
        elif args.command in ('trial-observation-bind', 'trial-observation-status', 'trial-observation-advance'):
            from .quarantined_observation import QuarantinedObservation
            if args.command == 'trial-observation-status':
                with permit.open_lifecycle() as lifecycle:
                    observer = QuarantinedObservation.from_lifecycle(lifecycle)
                    if observer is None:
                        raise ValueError('Original observation is not bound; inspect cannot initialize it')
                    value = {**observer.inspect(lifecycle), 'read_only': True}
            else:
                runtime = _original_trial_runtime(permit)
                try:
                    if args.command == 'trial-observation-bind':
                        generator = _existing_trial_generator(registry, args.generator)
                        observer = QuarantinedObservation.bind(runtime, generator, permit)
                        value = {**observer.inspect(runtime.lifecycle), 'work_started': False}
                    else:
                        observer = QuarantinedObservation.from_lifecycle(runtime.lifecycle)
                        if observer is None:
                            raise ValueError('Original observation is not bound')
                        value = observer.advance_batch(runtime, max_events=args.max_events)
                finally:
                    runtime.close()
        elif args.command == 'trial-controller-freeze':
            from .full_calibration_registry import _read
            from .neural_forward_ticket import open_existing_authority
            runtime = _original_trial_runtime(permit)
            try:
                controller = FullTrialController.freeze(permit, runtime,
                    open_existing_authority(permit.paths['authority']),
                    FullControllerSchedule.from_dict(_read(project_path(args.schedule))))
                value = {'status': 'frozen', 'identity_digest': fingerprint(controller.identity),
                         'work_started': False, 'formal_gate_eligible': False}
            finally:
                runtime.close()
        elif args.command == 'trial-controller-run':
            controller = FullTrialController.inspect(permit)
            # Keep the original issuer alive through bounded canonical pumps;
            # each dispatch drains only its already-paid worker before exit.
            value = controller.run(max_pumps=args.max_pumps)
        elif args.command == 'trial-controller-pump':
            controller = FullTrialController.inspect(permit)
            # One original registered opportunity; no schedule or wallet is created.
            # Its explicit execution may call the original registered provider.
            value = controller.pump()
        else:
            controller = FullTrialController.inspect(permit)
            rows = controller.verify()
            value = {'controller_digest': fingerprint(controller.identity),
                'original_opportunities': len(controller.schedule.due_at),
                'attempted_opportunities': len(rows),
                'completed_opportunities': sum(controller.step_completed(r) for r in rows),
                'pending_dependencies': [asdict(d) for d in controller.pending_dependencies()],
                'steps': rows, 'read_only': True, 'formal_gate_eligible': False}
    elif args.command in ('fixed-oracle-freeze', 'fixed-oracle-status', 'fixed-oracle-pump', 'fixed-oracle-run', 'fixed-oracle-report'):
        from .fixed_recipe_oracle import FixedRecipeOracle
        from .full_calibration_registry import FullTrialRegistry, QuarantinedTrialPermit
        registry = FullTrialRegistry.inspect(args.registry)
        permit = QuarantinedTrialPermit(registry, registry.identity_for(
            args.cell, args.repetition, oracle_index=args.oracle_index, estimand_slot=args.slot))
        if args.command == 'fixed-oracle-freeze':
            runtime = _original_trial_runtime(permit)
            try:
                generator = _existing_trial_generator(registry, args.generator)
                controller = FixedRecipeOracle.freeze(permit, runtime, generator)
                value = {'controller_digest': fingerprint(controller.identity), 'work_started': False,
                         'formal_gate_eligible': False}
            finally:
                runtime.close()
        else:
            controller = FixedRecipeOracle.inspect(permit)
            if args.command == 'fixed-oracle-run':
                value = controller.run()
            elif args.command == 'fixed-oracle-pump':
                value = controller.pump()
            else:
                value = {**controller.report(), 'read_only': True}
    elif args.command == 'cold-refit-freeze':
        from .conditional_cold_refit import freeze_cold_refit
        from .full_calibration_registry import FullTrialRegistry, QuarantinedTrialPermit
        registry = FullTrialRegistry.inspect(args.registry)
        permit = QuarantinedTrialPermit(registry, registry.identity_for(args.cell, args.repetition))
        generator = _existing_trial_generator(registry, args.generator)
        value = freeze_cold_refit(permit, generator, args.slot)
    elif args.command in ('policy-execution-status', 'policy-execution-run'):
        from .adaptive_policy_execution import advance_original_policy, read_policy_execution
        from .full_calibration_registry import FullTrialRegistry, QuarantinedTrialPermit
        registry = FullTrialRegistry.inspect(args.registry)
        if args.command == 'policy-execution-run':
            permit = QuarantinedTrialPermit(registry, registry.identity_for(args.cell, args.repetition,
                                                                          oracle_index=args.oracle_index))
            value = advance_original_policy(permit, max_pumps=args.max_pumps)
        else:
            value = {**read_policy_execution(registry, args.cell, args.repetition), 'read_only': True}
    elif args.command in ('cold-refit-status', 'cold-refit-run'):
        from .conditional_cold_refit import report_cold_refit, run_cold_refit
        if args.command == 'cold-refit-run':
            # One originally registered path, once only; this cannot initialize
            # an oracle, alter its support, select a learner or grant more work.
            value = run_cold_refit(args.directory, args.atom)
        else:
            value = {**report_cold_refit(args.directory), 'read_only': True}
    elif args.command in ('trial-risk-prepare', 'trial-risk-produce', 'trial-risk-status'):
        from .full_adaptive_risk import prepare_original_oracle, produce_full_risk, verify_full_risk
        from .full_calibration_registry import FullTrialRegistry, QuarantinedTrialPermit
        registry = FullTrialRegistry.inspect(args.registry)
        if args.command == 'trial-risk-prepare':
            permit = QuarantinedTrialPermit(registry, registry.identity_for(args.cell, args.repetition))
            value = prepare_original_oracle(permit)
        elif args.command == 'trial-risk-produce':
            value = produce_full_risk(registry, args.cell, args.repetition)
        else:
            value = {**verify_full_risk(registry, args.cell, args.repetition, args.artifact_digest),
                     'read_only': True}
    elif args.command in ('trial-certificate-produce', 'trial-certificate-status', 'trial-certificate-publish'):
        from .full_calibration_registry import FullTrialRegistry
        from .full_trial_certificate import produce_full_certificate, verify_full_certificate
        registry = FullTrialRegistry.inspect(args.registry)
        if args.command == 'trial-certificate-produce':
            value = produce_full_certificate(registry)
        elif args.command == 'trial-certificate-publish':
            from .full_trial_certificate import publish_lifecycle_certificate
            value = publish_lifecycle_certificate(registry, args.directory, artifact_digest=args.artifact_digest)
        else:
            value = verify_full_certificate(registry, artifact_digest=args.artifact_digest,
                expected_runtime_scope_hash=fingerprint(registry.plan.runtime_scope), require_formal=False)
    elif args.command in ('full-calibrate', 'full-calibration-status'):
        from .full_pipeline_calibration import (
            FullCalibrationPlan,
            inspect_full_pipeline_calibration,
            run_full_pipeline_calibration,
        )
        if args.command == 'full-calibration-status':
            value = inspect_full_pipeline_calibration(args.run)
        else:
            fields = FullCalibrationPlan().to_dict()
            if args.plan:
                fields.update(json.loads(project_path(args.plan).read_text('utf-8')))
            value = run_full_pipeline_calibration(args.run, plan=FullCalibrationPlan.from_dict(fields),
                                                  max_new_trials=args.max_new_trials)
    elif args.command == 'calibrate':
        from .pipeline_calibration import CalibrationPlan, run_pipeline_calibration
        fields = json.loads(project_path(args.plan).read_text('utf-8')) if args.plan else {}
        value = run_pipeline_calibration(args.run, plan=CalibrationPlan(**fields), max_new_trials=args.max_new_trials)
    elif args.command == 'probe':
        from .joint_pool import FrozenPool
        from .nonlinear_channel import run_probe
        from .rolling import build_rolling_data
        bars = pd.read_parquet(project_path(args.bars))
        data = build_rolling_data(bars, RollingSpec())
        pool = FrozenPool.from_dict(json.loads(project_path(args.pool).read_text('utf-8'))) if args.pool else None
        value = run_probe(data, args.cutoff, pool=pool)
        immutable_json(project_path(args.run) / f'probe-{fingerprint(args.cutoff)[:16]}.json', value)
    elif args.command in ('shared-controls', 'shared-controls-status'):
        if args.command == 'shared-controls-status':
            if (project_path(args.run) / 'shared_controls_identity.json').exists():
                from .shared_controls_executor import inspect_shared_authority_controls
                value = inspect_shared_authority_controls(args.run)
            else:
                from .shared_controls import inspect_shared_controls
                value = inspect_shared_controls(args.run)
        else:
            from .neural_forward_ticket import open_existing_authority
            from .shared_controls_executor import SharedControlsExecutor
            from .v7_agent import CodexBatchBackend
            authority = open_existing_authority(args.work_authority)
            from .monthly_schedule import load_control_schedules
            schedules = load_control_schedules(args.work_schedules, symbols=args.symbol, month=str(args.cutoff)[:7])
            if any(tuple(s.to_dict()['config']['symbols']) != tuple(args.symbol) for s in schedules.values()):
                raise ValueError('Control schedule universe differs from explicit CLI universe')
            executor = SharedControlsExecutor(args.run, authority=authority, schedules=schedules,
                feed_directory=args.feed_directory, exposure_path=args.exposure_path, cutoff=args.cutoff,
                spec=RollingSpec(availability_delay_seconds=args.availability_delay_seconds,
                                 window_days=args.window_days))
            value = executor.run_file(args.bars, backend=CodexBatchBackend(project_path(args.run) / 'agent'))
    elif args.command == 'controls':
        from .v7_agent import CodexBatchBackend
        from .v7_controls import run_controls
        fields = json.loads(project_path(args.config).read_text('utf-8'))
        fields['symbols'] = tuple(fields['symbols'])
        backend = CodexBatchBackend(project_path(args.run) / 'agent') if args.gpt else None
        value = run_controls(args.run, pd.read_parquet(project_path(args.bars)), args.cutoff,
            config=V7Config(**fields), backend=backend, arms=('local', 'random', 'gpt') if args.gpt else ('local', 'random'))
    elif args.command == 'variance-target':
        from .evidence import file_hash
        from .target_experiments import build_variance_targets
        from .v7_run import source_files
        bars_path, origins_path = project_path(args.bars), project_path(args.origins)
        targets = build_variance_targets(pd.read_parquet(bars_path), pd.read_parquet(origins_path), as_of=args.as_of)
        value = {'source_files': source_files(), 'bars_hash': file_hash(bars_path),
                 'origins_hash': file_hash(origins_path), 'as_of': args.as_of,
                 'target_protocol': targets.attrs, 'targets': targets.astype(str).to_dict('list'),
                 'formal_claims_enabled': False}
        immutable_json(project_path(args.run) / 'variance-target.json', value)
    elif args.command == 'inspect':
        value = inspect_run(args.run)
    elif args.command == 'research':
        value = _research_from_file(args)
    else:
        bars = pd.read_parquet(project_path(args.bars))
        if args.command == 'replay':
            bars = bars.loc[bars.symbol.isin(args.symbol)].copy()
            spec = RollingSpec(task=args.task, horizon_hours=20 if args.task == 'first_event' else 12,
                               availability_delay_seconds=args.availability_delay_seconds,
                               window_days=args.window_days)
            value = replay_baseline(args.run, bars, args.cutoffs, spec=spec)
            # Replay ledger is restartable; per-invocation new counts differ.
            total = [{k: v for k, v in item.items() if k != 'new_predictions'} for item in value]
            immutable_json(project_path(args.run) / f'replay-{fingerprint(args.cutoffs)[:16]}.json', total)
    print(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
