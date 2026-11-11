#!/usr/bin/env python3
"""Run the fixed development-cache -> weak-label -> 200-step smoke sequence.

Commands: prepare --spec SPEC --job-dir DIR; launch --job-dir DIR;
run --job-dir DIR; status --job-dir DIR. A pipeline never starts formal training.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import research10_background as bg

SCRIPT = Path(__file__).resolve()
SCHEMA = 'research10-pipeline.v1'


def _resolved(value):
    return Path(value).expanduser().resolve()


def _unit(job):
    return 'research10-' + hashlib.sha256(str(job).encode()).hexdigest()[:16] + '.service'


def _active(job):
    result = subprocess.run(['systemctl', '--user', 'is-active', '--quiet', _unit(job)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def _identity(spec, job):
    required = ('frontend_job', 'dataset', 'house_protocol', 'output_dir', 'python')
    if any(not isinstance(spec.get(key), str) for key in required):
        raise ValueError('spec requires frontend_job, dataset, house_protocol, output_dir, python')
    values = {key: str(_resolved(spec[key])) for key in required}
    frontend, dataset, protocol, output = map(Path, (values['frontend_job'], values['dataset'],
                                                      values['house_protocol'], values['output_dir']))
    if not dataset.is_dir() or not protocol.is_file() or not Path(values['python']).is_file():
        raise ValueError('dataset, house protocol or interpreter missing')
    if output == job or output in job.parents or job in output.parents:
        raise ValueError('pipeline state and result directories must be separate')
    front = bg.load_json(frontend / 'contract.json')
    if front.get('kind') != 'frontend-cache' or front.get('cache_scope') != 'development':
        raise ValueError('frontend job must be the development cache')
    cohort = protocol.parent / 'cohort.json'
    data = bg.load_json(cohort); roles = bg.load_json(protocol)
    if roles.get('schema_version') != 'research10.house-protocol.v1' or data.get('schema_version') != 'research10.cohort.v1':
        raise ValueError('invalid cohort or house protocol schema')
    groups = []
    for role in ('fit', 'selection', 'calibration', 'audit'):
        group = roles.get(role)
        if not isinstance(group, list) or not group or len(group) != len(set(group)) or group != data.get('roles', {}).get(role):
            raise ValueError('protocol and cohort roles differ: ' + role)
        groups.append(set(group))
    if any(a & b for i, a in enumerate(groups) for b in groups[i+1:]):
        raise ValueError('protocol house roles overlap')
    if bg.file_hash(cohort) != roles.get('source_sha256'):
        raise ValueError('protocol cohort fingerprint mismatch')
    claimed = roles.get('fingerprint')
    raw = {k: v for k, v in roles.items() if k != 'fingerprint'}
    calculated = hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if claimed != calculated:
        raise ValueError('protocol fingerprint mismatch')
    if _resolved(data['dataset_root']) != dataset:
        raise ValueError('protocol dataset differs')
    official = bg.load_json(dataset/'splits/official_house_split.json')
    if not set.union(*groups) <= set(official['train']):
        raise ValueError('protocol includes a non-training house')
    argv = front['argv']
    if (bg._path(bg._option(argv, '--cohort'), Path(front['cwd'])) != cohort or
            bg._path(bg._option(argv, '--dataset'), Path(front['cwd'])) != dataset):
        raise ValueError('frontend contract does not match cohort or dataset')
    if any(p == output or p in output.parents or p == job or p in job.parents
           for p in (dataset, frontend)):
        raise ValueError('output overlaps source dataset or frontend job')
    immutable = [dataset, protocol, cohort, frontend/'contract.json', SCRIPT, bg.SCRIPT]
    source_hashes = {str(path): bg.tree_hash(path) for path in immutable}
    identity = dict(values, schema=SCHEMA, source_hashes=source_hashes,
                    frontend_contract_sha256=front['contract_sha256'])
    return dict(identity, fingerprint=hashlib.sha256(bg.canonical(identity)).hexdigest())


def _check(identity):
    data = dict(identity); expected = data.pop('fingerprint')
    if data.get('schema') != SCHEMA or hashlib.sha256(bg.canonical(data)).hexdigest() != expected:
        raise ValueError('pipeline identity changed')
    for path, value in identity['source_hashes'].items():
        if bg.tree_hash(path) != value:
            raise ValueError('pipeline input changed: ' + path)


def prepare(spec_path, job):
    job = _resolved(job)
    if job.exists() and any(job.iterdir()): raise ValueError('pipeline job directory must be new/empty')
    identity = _identity(bg.load_json(spec_path), job)
    output = Path(identity['output_dir'])
    if output.exists() and any(output.iterdir()): raise ValueError('pipeline output directory must be new/empty')
    job.mkdir(parents=True, exist_ok=True)
    bg.atomic_json(job/'contract.json', identity)
    bg.atomic_json(job/'status.json', {'stage': 'waiting_cache', 'status': 'prepared',
                                     'contract_sha256': identity['fingerprint'], 'updated_at': bg.now()})
    return identity


def _front(identity):
    job = Path(identity['frontend_job'])
    contract, state = bg.load_json(job/'contract.json'), bg.load_json(job/'status.json')
    if contract.get('contract_sha256') != identity['frontend_contract_sha256'] or state.get('contract_sha256') != contract.get('contract_sha256'):
        raise ValueError('frontend contract/status mismatch')
    if state.get('status') != 'completed':
        raise ValueError('frontend cache has not completed')
    bg.check_contract(contract)
    report = bg.load_json(Path(contract['output_dir'])/'run_report.json')
    cohort = bg.load_json(Path(identity['house_protocol']).parent/'cohort.json')
    expected = {r['floor_uid'] for r in cohort['cohort'] if r['role'] in ('fit', 'selection', 'calibration')}
    floors = report.get('floors', [])
    if (report.get('status') != 'complete' or report.get('scope') != 'development' or
            report.get('ground_truth_read') is not False or report.get('input_errors') != [] or
            report.get('floor_samples') != len(expected) or len(floors) != len(expected) or
            {r['floor_uid'] for r in floors} != expected):
        raise ValueError('frontend report does not exactly cover development cohort')
    return contract


def _weak(identity, path):
    data = bg.load_json(path)
    protocol = bg.load_json(identity['house_protocol'])
    if data.get('schema_version') != 'opening-pair-supervision.v1' or data.get('usage') != 'training_calibration' or data.get('house_protocol') != protocol:
        raise ValueError('weak supervision protocol or usage differs')
    raw = {k: v for k, v in data.items() if k != 'fingerprint'}
    digest = hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if data.get('fingerprint') != digest:
        raise ValueError('weak supervision fingerprint mismatch')
    cohort = bg.load_json(Path(identity['house_protocol']).parent/'cohort.json')
    allowed = {r['floor_uid']: r['house_id'] for r in cohort['cohort'] if r['role'] in ('fit', 'selection', 'calibration')}
    for row in data.get('pairs', []):
        if row.get('floor_uid') not in allowed or allowed[row['floor_uid']] != row.get('house_id'):
            raise ValueError('weak supervision includes audit, heldout or foreign house')
    if not data.get('pairs'):
        raise ValueError('weak supervision has no pairs')
    return data


def _training_spec(identity, cache, weak, diagnostic=False):
    output = Path(identity['output_dir'])/('diagnostic_pairs' if diagnostic else 'smoke_pairs')
    dataset, protocol = identity['dataset'], identity['house_protocol']
    argv = [identity['python'], str(ROOT/'tools/run_floor_layout.py'), 'train', '--stage', 'pairs',
            '--dataset', dataset, '--house-protocol', protocol, '--prediction-cache', str(cache),
            '--supervision', str(weak), '--output', str(output), '--device', 'cuda', '--mode', 'full' if diagnostic else 'smoke',
            '--seed', '123', '--protocol-seed', '123', '--pair-steps', '5000' if diagnostic else '200', '--pair-batch-size', '2',
            '--validation-interval', '500' if diagnostic else '100', '--patience', '5', '--pair-objective', 'known_cells',
            '--selection-metric', 'cross_room_ap']
    return {'kind': 'floor-train', 'purpose': 'diagnostic', 'cwd': str(ROOT), 'argv': argv,
            'output_dir': str(output), 'max_retries': 2,
            'fingerprint_paths': [dataset, protocol, str(Path(protocol).parent/'cohort.json'), str(cache), str(weak), str(SCRIPT)]}


def _weak_command(identity, cache, output):
    return [identity['python'], str(ROOT/'tools/run_floor_layout.py'), 'prepare', '--stage', 'opening',
            '--prediction-cache', str(cache), '--dataset', identity['dataset'], '--house-protocol',
            identity['house_protocol'], '--output', str(output), '--device', 'cuda']


def _check_smoke(identity):
    report = bg.load_json(Path(identity['output_dir'])/'smoke_pairs'/'training_report.json')
    gradients = report.get('gradient_norms_first_step', {})
    if (report.get('training_status') != 'smoke_trained' or report.get('steps') != 200 or
            report.get('parameters_changed') is not True or not isinstance(report.get('skipped_batches'), int) or
            not 0 <= report['skipped_batches'] < report['steps'] or any(
                not isinstance(gradients.get(key), (int, float)) or not math.isfinite(gradients[key]) or gradients[key] <= 0
                for key in ('encoder', 'matcher'))):
        raise ValueError('smoke training did not demonstrate 200 steps and finite positive gradients')
    return report


def _check_diagnostic(identity):
    report = bg.load_json(Path(identity['output_dir'])/'diagnostic_pairs'/'training_report.json')
    gradients = report.get('gradient_norms_first_step', {})
    if (report.get('training_status') != 'trained' or not isinstance(report.get('steps'), int) or
            not 1 <= report['steps'] <= 5000 or report.get('parameters_changed') is not True or
            not isinstance(report.get('skipped_batches'), int) or not 0 <= report['skipped_batches'] < report['steps'] or any(
                not isinstance(gradients.get(key), (int, float)) or not math.isfinite(gradients[key]) or gradients[key] <= 0
                for key in ('encoder', 'matcher'))):
        raise ValueError('diagnostic run did not demonstrate finite updates without skipped batches')
    return report


def _write_state(job, state, **updates):
    state.update(updates, updated_at=bg.now())
    bg.atomic_json(job/'status.json', state)


def run(job, poll_seconds=10):
    job = _resolved(job)
    with (job/'runner.lock').open('a+') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError('another pipeline coordinator is running')
        identity, state = bg.load_json(job/'contract.json'), bg.load_json(job/'status.json')
        if state.get('contract_sha256') != identity.get('fingerprint'):
            raise ValueError('pipeline contract/status mismatch')
        if state.get('status') in ('completed', 'blocked'):
            raise ValueError('pipeline is already ' + state['status'])
        output, frontjob = Path(identity['output_dir']), Path(identity['frontend_job'])
        cache = Path(bg.load_json(frontjob/'contract.json')['output_dir'])
        weakdir, smokejob, diagnosticjob = output/'weak_supervision', job/'smoke_job', job/'diagnostic_job'
        weak = weakdir/'opening_weak.json'
        try:
            _check(identity)
            _write_state(job, state, status='running')
            while True:
                stage = state['stage']
                if stage == 'waiting_cache':
                    frontstate = bg.load_json(frontjob/'status.json')
                    if frontstate.get('contract_sha256') != identity['frontend_contract_sha256']:
                        raise ValueError('frontend state contract changed')
                    if frontstate.get('status') == 'completed':
                        _front(identity); _check(identity)
                        _write_state(job, state, stage='prepare_pending')
                    elif frontstate.get('status') in ('blocked', 'interrupted', 'failed'):
                        raise ValueError('frontend cache is ' + frontstate['status'])
                    elif frontstate.get('status') == 'prepared' and not _active(frontjob):
                        if state.get('cache_launch_issued'):
                            raise ValueError('frontend service exited before starting the cache job')
                        bg.launch(frontjob)
                        _write_state(job, state, cache_launch_issued=True)
                    elif frontstate.get('status') not in ('prepared', 'running', 'retrying'):
                        raise ValueError('unexpected frontend job status: ' + str(frontstate.get('status')))
                    elif frontstate.get('status') in ('running', 'retrying') and not _active(frontjob):
                        raise ValueError('frontend service inactive while job state is ' + frontstate['status'])
                    time.sleep(poll_seconds)
                elif stage == 'prepare_pending':
                    _front(identity); _check(identity)
                    if weakdir.exists() and any(weakdir.iterdir()):
                        raise ValueError('weak output already exists before preparation')
                    output.mkdir(parents=True, exist_ok=True)
                    _write_state(job, state, stage='preparing')
                    cmd = _weak_command(identity, cache, weakdir)
                    with (job/'prepare.log').open('ab') as log:
                        result = subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                            env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
                    if result.returncode != 0: raise ValueError('weak-supervision preparation failed, see prepare.log')
                    if not weak.is_file(): raise ValueError('prepare exited successfully without opening_weak.json')
                    _weak(identity, weak)
                    bg.atomic_json(job/'weak_prepare_completion.json', {
                        'status': 'completed', 'exit_code': result.returncode,
                        'weak_sha256': bg.file_hash(weak), 'contract_sha256': identity['fingerprint'],
                        'argv_sha256': hashlib.sha256(bg.canonical(cmd)).hexdigest(), 'completed_at': bg.now()})
                elif stage == 'preparing':
                    marker_path = job/'weak_prepare_completion.json'
                    if not weak.is_file() or not marker_path.is_file():
                        raise ValueError('weak preparation completion marker missing; output retained')
                    marker = bg.load_json(marker_path)
                    if (marker.get('status') != 'completed' or marker.get('exit_code') != 0 or
                            marker.get('contract_sha256') != identity['fingerprint'] or
                            marker.get('argv_sha256') != hashlib.sha256(bg.canonical(_weak_command(identity, cache, weakdir))).hexdigest() or
                            marker.get('weak_sha256') != bg.file_hash(weak)):
                        raise ValueError('weak preparation completion marker mismatch')
                    _check(identity); _front(identity); _weak(identity, weak)
                    _write_state(job, state, stage='smoke_prepare_pending', weak_sha256=bg.file_hash(weak))
                elif stage == 'smoke_prepare_pending':
                    _check(identity); _front(identity); _weak(identity, weak)
                    if bg.file_hash(weak) != state.get('weak_sha256'): raise ValueError('weak supervision changed')
                    spec = _training_spec(identity, cache, weak)
                    specpath = job/'smoke_spec.json'
                    if specpath.exists() and bg.load_json(specpath) != spec: raise ValueError('smoke specification changed')
                    if not specpath.exists(): bg.atomic_json(specpath, spec)
                    if (smokejob/'contract.json').exists():
                        if bg.load_json(smokejob/'contract.json') != bg.build_contract(spec, smokejob):
                            raise ValueError('existing smoke job has a different contract')
                    else:
                        bg.prepare(specpath, smokejob)
                    _write_state(job, state, stage='smoke_launch_pending')
                elif stage == 'smoke_launch_pending':
                    smoke = bg.load_json(smokejob/'status.json')
                    if smoke['status'] == 'prepared' and not _active(smokejob): bg.launch(smokejob)
                    elif smoke['status'] not in ('prepared', 'running', 'retrying', 'completed'):
                        raise ValueError('smoke job is ' + smoke['status'])
                    _write_state(job, state, stage='waiting_smoke', smoke_launched_at=time.time())
                elif stage == 'waiting_smoke':
                    smoke = bg.load_json(smokejob/'status.json')
                    if smoke['status'] == 'completed':
                        contract = bg.load_json(smokejob/'contract.json')
                        if smoke.get('contract_sha256') != contract.get('contract_sha256'):
                            raise ValueError('smoke contract/status mismatch')
                        bg.check_contract(contract); report = _check_smoke(identity)
                        _write_state(job, state, stage='diagnostic_prepare_pending',
                            smoke_report_sha256=bg.file_hash(output/'smoke_pairs'/'training_report.json'),
                            smoke_report_status=report.get('status'),
                            smoke_skipped_batches=report['skipped_batches'],
                            note='200-step engineering smoke passed; matching quality not yet established')
                        continue
                    if smoke['status'] in ('blocked', 'failed', 'interrupted'):
                        raise ValueError('smoke job is ' + smoke['status'] + ': ' + str(smoke.get('reason')))
                    if smoke['status'] not in ('prepared', 'running', 'retrying'):
                        raise ValueError('unexpected smoke status: ' + str(smoke['status']))
                    if time.time() - state.get('smoke_launched_at', time.time()) > 30 and not _active(smokejob):
                        raise ValueError('smoke service inactive before a terminal status was recorded')
                    time.sleep(poll_seconds)
                elif stage == 'diagnostic_prepare_pending':
                    _check(identity); _front(identity); _weak(identity, weak); _check_smoke(identity)
                    if bg.file_hash(weak) != state.get('weak_sha256'): raise ValueError('weak supervision changed')
                    spec = _training_spec(identity, cache, weak, diagnostic=True)
                    specpath = job/'diagnostic_spec.json'
                    if specpath.exists() and bg.load_json(specpath) != spec: raise ValueError('diagnostic specification changed')
                    if not specpath.exists(): bg.atomic_json(specpath, spec)
                    if (diagnosticjob/'contract.json').exists():
                        if bg.load_json(diagnosticjob/'contract.json') != bg.build_contract(spec, diagnosticjob):
                            raise ValueError('existing diagnostic job has a different contract')
                    else:
                        bg.prepare(specpath, diagnosticjob)
                    _write_state(job, state, stage='diagnostic_launch_pending')
                elif stage == 'diagnostic_launch_pending':
                    diagnostic = bg.load_json(diagnosticjob/'status.json')
                    if diagnostic['status'] == 'prepared' and not _active(diagnosticjob): bg.launch(diagnosticjob)
                    elif diagnostic['status'] not in ('prepared', 'running', 'retrying', 'completed'):
                        raise ValueError('diagnostic job is ' + diagnostic['status'])
                    _write_state(job, state, stage='waiting_diagnostic', diagnostic_launched_at=time.time())
                elif stage == 'waiting_diagnostic':
                    diagnostic = bg.load_json(diagnosticjob/'status.json')
                    if diagnostic['status'] == 'completed':
                        contract = bg.load_json(diagnosticjob/'contract.json')
                        if diagnostic.get('contract_sha256') != contract.get('contract_sha256'):
                            raise ValueError('diagnostic contract/status mismatch')
                        bg.check_contract(contract); report = _check_diagnostic(identity)
                        _write_state(job, state, stage='done', status='completed',
                            diagnostic_report_sha256=bg.file_hash(output/'diagnostic_pairs'/'training_report.json'),
                            diagnostic_report_status=report.get('status'), diagnostic_steps=report['steps'],
                            diagnostic_skipped_batches=report['skipped_batches'],
                            note='one seed123 known_cells diagnostic only; R04 ablation and formal-training gates not established')
                        return 0
                    if diagnostic['status'] in ('blocked', 'failed', 'interrupted'):
                        raise ValueError('diagnostic job is ' + diagnostic['status'] + ': ' + str(diagnostic.get('reason')))
                    if diagnostic['status'] not in ('prepared', 'running', 'retrying'):
                        raise ValueError('unexpected diagnostic status: ' + str(diagnostic['status']))
                    if time.time() - state.get('diagnostic_launched_at', time.time()) > 30 and not _active(diagnosticjob):
                        raise ValueError('diagnostic service inactive before a terminal status was recorded')
                    time.sleep(poll_seconds)
                elif stage == 'done': return 0
                else: raise ValueError('unknown pipeline stage: ' + str(stage))
        except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
            _write_state(job, state, status='blocked', reason=str(exc))
            return 2


def launch(job):
    job = _resolved(job)
    unit = 'research10-pipeline-' + hashlib.sha256(str(job).encode()).hexdigest()[:16]
    cmd = ['systemd-run', '--user', '--unit=' + unit, '--collect', '--property=Restart=no',
           '--property=KillMode=control-group', '--property=TimeoutStopSec=30',
           '--working-directory=' + str(ROOT), sys.executable, str(SCRIPT), 'run', '--job-dir', str(job)]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode: raise ValueError('systemd launch failed: ' + result.stderr.strip())
    return {'unit': unit + '.service', 'job_dir': str(job)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'run', 'launch', 'status'):
        p = sub.add_parser(name); p.add_argument('--job-dir', required=True)
        if name == 'prepare': p.add_argument('--spec', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare':
            identity = prepare(args.spec, args.job_dir)
            print(json.dumps({'status': 'prepared', 'contract_sha256': identity['fingerprint']}))
        elif args.command == 'launch': print(json.dumps(launch(args.job_dir)))
        elif args.command == 'status': print(json.dumps(bg.load_json(Path(args.job_dir)/'status.json'), ensure_ascii=False, indent=2))
        else: return run(args.job_dir)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        print('research10-pipeline: ' + str(exc), file=sys.stderr); return 2
    return 0


if __name__ == '__main__': raise SystemExit(main())
