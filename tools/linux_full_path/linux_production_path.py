"""Bounded engineering driver of original V7 producers; no acceptance bypasses.

Run from the frozen source checkout, with that checkout on PYTHONPATH. Enrollment
and observation binding are explicit production CLI operations preceding bind.
This local-search engineering run is NOT a full-provider algorithm calibration.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from uuid import uuid4

VERSION = 'v7-linux-original-producers-engineering-v3'


def read(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError('Missing original evidence: ' + str(path))
    return json.loads(path.read_text('utf-8'))


def digest(value):
    from factorlab.protocol import fingerprint
    return fingerprint(value)


def own_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def validate_cutoffs(values):
    import pandas as pd
    dates = [pd.Timestamp(v) for v in values]
    if (len(dates) != 4 or any(d.tzinfo is None or d.utcoffset().total_seconds()
            or d.day != 1 or d != d.normalize() for d in dates)
            or dates != sorted(set(dates))
            or any((d.month - 1) % 3 for d in dates[:2])
            or dates[2] != dates[1] + pd.DateOffset(months=1)
            or dates[3] != dates[2] + pd.DateOffset(months=1)):
        raise ValueError('Freeze two distinct quarter starts, then the next two month starts')
    return [d.isoformat() for d in dates]


def load_contract(root):
    from factorlab.discovery.v7_run import source_files
    value = read(Path(root) / 'identity.json')
    body = value['body']
    if (set(value) != {'body', 'hash'} or digest(body) != value['hash']
            or body['version'] != VERSION or body['harness_sha256'] != own_hash()
            or body['source_files'] != source_files()
            or body['root'] != str(Path(root).resolve())):
        raise ValueError('Original harness/source changed; never migrate a frozen run')
    validate_cutoffs(body['S_cutoffs'])
    return body


def originals(body):
    from factorlab.discovery.full_calibration_registry import FullTrialRegistry, QuarantinedTrialPermit
    from factorlab.discovery.finite_market_oracle import FiniteGenerator, FiniteMarketLaw
    from factorlab.discovery.neural_forward_ticket import open_existing_authority
    registry = FullTrialRegistry.inspect(body['registry'])
    permit = QuarantinedTrialPermit(registry, registry.identity_for(
        body['cell'], body['index'], oracle_index=body['oracle_index']))
    permit.verify()
    # Read-only reconstruction: never call a creating generator constructor here.
    generator = object.__new__(FiniteGenerator)
    generator.directory = Path(body['generator'])
    generator.identity = read(generator.directory / 'identity.json')
    law = dict(generator.identity['law']); law['symbols'] = tuple(law['symbols'])
    generator.law = FiniteMarketLaw(**law)
    generator.acceptance = registry.plan.acceptance
    generator.verify_registry(registry)
    wallet = open_existing_authority(permit.paths['authority'])
    if body.get('trial_id', permit.trial['trial_id']) != permit.trial['trial_id']:
        raise ValueError('Original trial changed')
    return permit, generator, wallet


def bind(root, registry, generator, cell, index, oracle_index, cutoffs):
    from factorlab.discovery.v7_run import source_files, V7Config
    from factorlab.discovery.v7_autonomous import AutonomousConfig
    from factorlab.discovery.v7_service import ServiceConfig
    from factorlab.discovery.quarantined_observation import QuarantinedObservation
    from factorlab.discovery.v7_storage import immutable_record
    root = Path(root).resolve()
    body = dict(version=VERSION, root=str(root), registry=str(Path(registry).resolve()),
        generator=str(Path(generator).resolve()), cell=cell, index=index, oracle_index=oracle_index,
        S_cutoffs=validate_cutoffs(cutoffs), harness_sha256=own_hash(), source_files=source_files(),
        evidence_scope='artificial_engineering_local_search_no_provider_or_market_claim')
    if (root / 'identity.json').exists():
        old = load_contract(root)
        if any(old[k] != v for k, v in body.items()):
            raise ValueError('Frozen acceptance identity cannot change')
        return status(root)
    if root.exists() and any(root.iterdir()):
        raise ValueError('Partial acceptance namespace cannot be reset')
    permit, gen, wallet = originals(body)
    import pandas as pd
    if (gen.law.gap_probability != 0 or gen.law.support_size < 128
            or pd.Timestamp(gen.law.start) + pd.Timedelta(hours=gen.law.hours)
                <= pd.Timestamp(body['S_cutoffs'][-1])):
        raise ValueError('Frozen no-gap 128-path law must cover all four S opportunities')
    with permit.open_lifecycle() as lc:
        observer = QuarantinedObservation(gen, permit)
        observed = observer.inspect(lc)
        if observed['events']:
            raise ValueError('Harness must freeze before first observation; use a fresh namespace')
    from factorlab.discovery.monthly_partition_policy import frozen_policy
    if wallet.manifest['partition_policy'] != frozen_policy('standard'):
        raise ValueError('Original standard nontransferable wallet required')
    research = V7Config(symbols=gen.law.symbols, search_mode='local', evaluation_mode='joint',
                        nonlinear_mode='isolated')
    body.update(trial_id=permit.trial['trial_id'], authority_manifest_hash=wallet.manifest_hash,
        service=asdict(ServiceConfig(session_id='linux-original-neural', poll=False,
            wall_seconds=7200, max_steps=7200, max_fits_per_step=1, max_fit_attempts=1)),
        autonomous=asdict(AutonomousConfig(backend='disabled', max_attempts=1, research=research,
                                           not_before_month=body['S_cutoffs'][0][:7])))
    immutable_record(root / 'identity.json', {'body': body, 'hash': digest(body)})
    return status(root)


def scheduler_rows(permit):
    path = permit.paths['runtime'] / 'autonomous' / 'scheduler.sqlite'
    if not path.is_file():
        return []
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as db:
        rows = db.execute('SELECT month,value,hash FROM jobs ORDER BY month').fetchall()
    result = []
    for month, raw, sha in rows:
        value = json.loads(raw)
        if digest(value) != sha or value['month'] != month:
            raise ValueError('Original scheduler state changed')
        result.append(value)
    return result


def status(root):
    body = load_contract(root)
    from factorlab.discovery.quarantined_observation import QuarantinedObservation
    permit, generator, wallet = originals(body)
    with permit.open_lifecycle() as lc:
        observer = QuarantinedObservation(generator, permit).inspect(lc)
    return dict(version=VERSION, scope=body['evidence_scope'], observation=observer,
        wallet=wallet.status(datetime.now(UTC).strftime('%Y-%m')), jobs=scheduler_rows(permit),
        S_cutoffs=body['S_cutoffs'], genuine_Linux_admission='not_inferred_from_driver_status')


def S_owner_wait(wallet_status, run_directory, month):
    expected = {'bridge_id': digest({'directory': str(run_directory), 'replay_month': month}),
                'run_directory': str(run_directory)}
    owner = wallet_status['owners'].get('S')
    return owner is not None and owner != expected


@contextmanager
def runtime_for(permit):
    from factorlab.discovery.v7_cli import _original_trial_runtime
    runtime = _original_trial_runtime(permit)
    try:
        permit.verify(runtime.lifecycle)
        yield runtime
    finally:
        runtime.close()


def _owned_research(scheduler):
    """Never signal or take over an active job issued by a different process."""
    import psutil
    from factorlab.discovery.v7_autonomous import ResearchScheduler, _read as read_job
    from factorlab.discovery.work_lease import _boot_id
    if type(scheduler) is not ResearchScheduler:
        raise TypeError('Actual original research scheduler required')
    me = psutil.Process()
    rows = []
    for month, in scheduler.db.execute('SELECT month FROM jobs ORDER BY month'):
        state = scheduler._load(month)
        if state['status'] not in {'planned', 'running', 'interrupted'}:
            continue
        job = read_job(state['job_path'])
        if digest(job) != state['job_hash'] or job['month'] != month:
            raise ValueError('Original research job differs from its durable reservation')
        issuer = job['outer_clock']['issuer']
        if (issuer['parent_pid'] != me.pid or issuer['parent_created'] != me.create_time()
                or issuer['boot_id'] != _boot_id()):
            raise ValueError('Original research issuer is different; use read-only status, never takeover')
        rows.append(state)
    return rows


def _assert_research_empty(boundary):
    """The originally owned kernel container, including reparented descendants."""
    try:
        boundary.check()
        if set(boundary.members()) != {os.getpid()}:
            raise ValueError('Original research container still contains descendants')
    except BaseException:
        # The original Linux cgroup.kill descriptor includes this issuer. Never
        # select an ancestor/new container or return a false terminal result.
        boundary.stop()
        raise


@contextmanager
def _research_container(root):
    if sys.platform != 'linux':
        raise ValueError('Original dedicated Linux research service required')
    from scripts.v7_guard.lease_enforcer import _LinuxService
    from factorlab.discovery.v7_storage import immutable_record
    import psutil
    # Requires actual MainPID, initially alone, Restart=no, control-group kill,
    # original inode/boot and original resource ceilings BEFORE any spawn.
    boundary = _LinuxService()
    try:
        identity = {'version':'original-research-container-v1', 'pid':os.getpid(),
            'created':psutil.Process().create_time(), 'boot':boundary.boot,
            'cgroup':str(boundary.path), 'inode':boundary.inode, 'harness_sha256':own_hash()}
        path = Path(root)/'research_containers'/digest(identity)
        immutable_record(path/'identity.json', identity)
        yield boundary
    finally:
        try:
            _assert_research_empty(boundary)
            # This records a completed kernel read, not a future service exit
            # or a numerical completion. An orphan kill cannot publish it.
            if 'path' in locals():
                immutable_record(path/'empty.json', {'identity_hash':digest(identity),
                    'only_original_issuer_remaining':identity['pid']})
        finally:
            boundary.close()


def _research_until_terminal(scheduler, month, deadline, *, boundary):
    """One original dispatch, then observation-only drain in the same issuer.

    The caller's wait ceiling only restricts the unchanged original job clock.
    No repeated step can plan a replacement opportunity during this drain.
    """
    owned = _owned_research(scheduler)  # Foreign jobs are untouched before cleanup.
    if time.monotonic() >= deadline:
        if owned:
            scheduler.stop_and_reap('supervisor_budget')
            _assert_research_empty(boundary)
            state = scheduler._load(month)
            return {'status': 'research_wait_consumed', 'job_status': state['status'] if state else None,
                    'retry_or_refund': False}
        _assert_research_empty(boundary)
        return {'status': 'wait_budget_exhausted_no_work', 'job_status': None}
    reason = 'supervisor_stopped'
    try:
        result = scheduler.step()
        while True:
            _owned_research(scheduler)
            state = scheduler._load(month)
            if state is None or state['status'] not in {'planned', 'running', 'interrupted'}:
                return {'status': 'research_terminal', 'research': result,
                        'job_status': state['status'] if state else None}
            if time.monotonic() >= deadline:
                reason = 'supervisor_budget'
                scheduler.stop_and_reap(reason)
                settled = scheduler.settle_running()
                state = scheduler._load(month)
                if state is not None and state['status'] in {'planned', 'running', 'interrupted'}:
                    raise ValueError('Original research teardown remains unresolved')
                return {'status': 'research_wait_consumed', 'job_status': state['status'] if state else None,
                        'settlement': settled, 'retry_or_refund': False}
            result = scheduler.settle_running()
            if result['status'] == 'pending':
                time.sleep(min(.2, max(0., deadline-time.monotonic())))
    finally:
        # Revalidate exact original issuer and process identities before any
        # signal. Cleanup failure propagates; it cannot report a settled result.
        _owned_research(scheduler)
        try:
            scheduler.stop_and_reap(reason)
        finally:
            _assert_research_empty(boundary)


def execute(root, action, *, max_events=96, wait_seconds=0):
    entered = time.monotonic()
    if type(max_events) is not int or not 1 <= max_events <= 96:
        raise ValueError('Original receipt batch bound is 1..96')
    if type(wait_seconds) is not int or not 0 <= wait_seconds <= 7200:
        raise ValueError('Bounded original worker wait required')
    if action == 'research' and wait_seconds == 0:
        raise ValueError('Research requires positive same-issuer wait; use status for read-only monitoring')
    body = load_contract(root)
    permit, generator, wallet = originals(body)
    if action == 'advance':
        from factorlab.discovery.finite_market_oracle import publish_due, verify_publications
        from factorlab.discovery.quarantined_observation import QuarantinedObservation
        with runtime_for(permit) as runtime:
            observer = QuarantinedObservation(generator, permit)
            observed = observer.inspect(runtime.lifecycle)
            # Do not pass an unprocessed S opportunity in the frozen logical schedule.
            import pandas as pd
            terminal = {r['month'] for r in scheduler_rows(permit)
                        if r['status'] not in {'planned', 'running', 'interrupted'}}
            boundary = next((pd.Timestamp(c) for c in body['S_cutoffs'] if c[:7] not in terminal), None)
            if boundary is not None:
                # The original source event is candle close + the fixed 5-second delay.
                needed = int((boundary - pd.Timestamp(generator.law.start)).total_seconds() // 3600)
                remaining = needed - observed['events']
                if remaining <= 0:
                    return {'status': 'waiting_original_S_opportunity', 'cutoff': boundary.isoformat()}
                max_events = min(max_events, remaining)
            published = publish_due(generator, permit, max_new_receipts=max_events)
            complete = set(verify_publications(generator, permit)['completed_hours'])
            count = 0
            while count < max_events and observed['events'] + count in complete:
                count += 1
            advanced = (observer.advance_batch(runtime, max_events=count) if count else
                        {'status': 'waiting_original_receipt', 'events_committed': 0})
            return {'publication': published, 'observation': advanced}
    if sys.platform != 'linux':
        raise ValueError('Production workers require real constrained Linux; no platform substitute')
    from factorlab.discovery.v7_autonomous import inherited_limits
    inherited_limits()
    with runtime_for(permit) as runtime:
        if action == 'policy-service':
            if runtime.claim_protocol() is None:
                return {'status': 'waiting_original_frozen_five_claim_protocol', 'executed': False}
            from factorlab.discovery.v7_service import ForwardService, ServiceConfig
            service = ForwardService(runtime, config=ServiceConfig(**body['service']), work_authority=wallet)
            try:
                return service.step()
            finally:
                service.close(close_runtime=False)
        if action == 'research':
            from factorlab.discovery.quarantined_observation import logical_now
            from factorlab.discovery.v7_autonomous import ResearchScheduler, AutonomousConfig
            from factorlab.discovery.v7_run import V7Config
            from factorlab.discovery.monthly_schedule import default_schedule
            month = logical_now(runtime.lifecycle).strftime('%Y-%m')
            if month not in {c[:7] for c in body['S_cutoffs']}:
                return {'status': 'waiting_frozen_S_cutoff', 'logical_month': month}
            rows = scheduler_rows(permit)
            old = next((r for r in rows if r['month'] == month), None)
            if old is None and S_owner_wait(wallet.status(datetime.now(UTC).strftime('%Y-%m')),
                                            permit.paths['research'], month):
                return {'status': 'waiting_actual_UTC_month', 'logical_month': month,
                        'reason': 'original nontransferable S owner already charged', 'retry_or_refund': False}
            fields = dict(body['autonomous']); config = dict(fields['research'])
            config['symbols'] = tuple(config['symbols']); fields['research'] = V7Config(**config)
            with _research_container(root) as boundary:
                scheduler = ResearchScheduler(runtime, config=AutonomousConfig(**fields), authority=wallet,
                    schedule=default_schedule(fields['research'], month=month))
                try:
                    result = _research_until_terminal(scheduler, month, entered + wait_seconds, boundary=boundary)
                    if result['job_status'] == 'complete':
                        from factorlab.discovery.structural_validation import queue_structural_validation
                        result['typed_queue'] = queue_structural_validation(runtime, wallet,
                            permit.paths['research'], month)
                    return result
                finally:
                    scheduler.close()
        from factorlab.discovery.algorithm_validation import AlgorithmValidation
        protocols = AlgorithmValidation.discover(runtime, wallet)
        if not protocols:
            return {'status': 'waiting_original_paid_S_and_typed_queue', 'executed': False}
        if action == 'resource':
            from factorlab.discovery.prospective_resource_work import pending_prospective_resource, run_prospective_resource
            for protocol in protocols:
                plans = pending_prospective_resource(protocol)
                if plans:
                    return run_prospective_resource(protocol, plans[0]['job'])
            return {'status': 'waiting_original_prospective_resource_plan', 'executed': False}
        if action == 'typed-forward':
            for protocol in protocols:
                protocol.resolve_fit_gates()
                if protocol.pending_work():
                    # Original frozen outer supervision supplies preparation/readback
                    # headroom. The producer still issues its own <=300s paid lease.
                    return protocol.step(deadline_monotonic=time.monotonic()+body['service']['wall_seconds'],
                                         max_fits=1)
                protocol.tick()
            return {'status': 'waiting_original_typed_forward_work', 'executed': False}
        raise ValueError('Unknown producer action')


def reload_original(root):
    """Fresh CPU process only verifies original completed artifacts/registrations."""
    body = load_contract(root); permit, _, _ = originals(body)
    from factorlab.discovery.neural_forward_admission import verify_completed_artifact
    from factorlab.discovery.neural_forward import NeuralForwardModel
    from factorlab.discovery.model_registry import ModelRegistry
    resources, models = [], []
    for path in sorted((permit.paths['runtime'] / 'resource_work').glob('*/result.json')):
        value = read(path)
        if value['status'] == 'completed':
            result = verify_completed_artifact(value['certificate'], allow_resource_trial=True)
            model_path = Path(value['certificate']['ticket']['path']).parent / 'launch' / 'model.json'
            model = NeuralForwardModel.load(model_path)
            if model.digest != result['model'].digest:
                raise ValueError('Fresh resource model digest changed')
            resources.append({'model_digest': model.digest, 'serving_eligible': False})
    registry = object.__new__(ModelRegistry)
    registry.directory = permit.paths['runtime'] / 'models'
    for path in sorted(registry.directory.glob('*/manifest.json')):
        value = read(path)
        if value['identity']['kind'] == 'v7_neural_forward':
            model, _ = registry.load(value['id'], expected_pipeline=value['identity']['pipeline'])
            models.append({'id': value['id'], 'model_digest': model.digest})
    return {'status': 'verified_originals' if resources or models else 'pending_original_completed_artifacts',
            'resource_models': resources, 'registered_neural_models': models,
            'scope': 'resource evidence and serving admission are separate', 'cpu_only': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    sub = parser.add_subparsers(dest='action', required=True)
    freeze = sub.add_parser('bind')
    for key in ('registry', 'generator', 'cell'):
        freeze.add_argument('--'+key, required=True)
    freeze.add_argument('--index', type=int, default=0)
    freeze.add_argument('--oracle-index', type=int)
    freeze.add_argument('--cutoffs', nargs=4, required=True)
    sub.add_parser('status'); sub.add_parser('reload'); sub.add_parser('_reload-child')
    for name in ('advance', 'research', 'resource', 'typed-forward', 'policy-service'):
        p = sub.add_parser(name)
        p.add_argument('--max-events', type=int, default=96)
        p.add_argument('--wait-seconds', type=int, default=7200 if name == 'research' else 0)
    args = parser.parse_args(argv)
    if args.action == 'bind':
        result = bind(args.root, args.registry, args.generator, args.cell, args.index, args.oracle_index, args.cutoffs)
    elif args.action == 'status':
        result = status(args.root)
    elif args.action == 'reload':
        load_contract(args.root)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        child = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--root', args.root,
            '_reload-child'], env=env, capture_output=True, text=True, timeout=300, check=True)
        result = json.loads(child.stdout)
    elif args.action == '_reload-child':
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise ValueError('Reload requires a fresh CPU-only child')
        result = reload_original(args.root)
    else:
        from factorlab.discovery.v7_storage import immutable_record
        from filelock import FileLock
        load_contract(args.root)
        with FileLock(str(Path(args.root)/'driver.lock'), timeout=0):
            attempt = Path(args.root)/'attempts'/uuid4().hex
            immutable_record(attempt/'started.json', {'action': args.action, 'physical_UTC': datetime.now(UTC).isoformat()})
            try:
                result = execute(args.root, args.action, max_events=args.max_events, wait_seconds=args.wait_seconds)
            except BaseException as exc:
                immutable_record(attempt/'failed.json', {'type': type(exc).__name__, 'message': str(exc),
                    'physical_UTC': datetime.now(UTC).isoformat(), 'original_evidence_retained': True})
                raise
            immutable_record(attempt/'result.json', result)
    print(json.dumps(result, sort_keys=True, default=str))


if __name__ == '__main__':
    main()
