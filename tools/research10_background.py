#!/usr/bin/env python3
"""Prepare and run a contract-bound, resumable job under a user systemd service.

This runner does not create a training job or decide whether a research gate has
passed. A prepared spec must supply an explicit command, a separate training
output directory, and immutable input paths. The command must write its own
atomic checkpoint; on a retry the runner appends ``--resume <checkpoint>``.

Commands: prepare --spec SPEC --job-dir DIR; launch --job-dir DIR;
status --job-dir DIR. ``run`` is the service entry point (also useful in tests).
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
SCHEMA = 'research10-background.v1'
IGNORE = {'__pycache__', '.pytest_cache', '.git'}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=str(path.parent),
                                     prefix=path.name + '.', delete=False) as stream:
        tmp = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    try:
        os.replace(str(tmp), str(path))
    finally:
        tmp.unlink(missing_ok=True)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def tree_hash(path):
    """Hash paths and bytes, including file and directory symlinks.

    Directory symlinks are rejected to prevent a mutable, untracked external
    tree from being omitted. Python bytecode and test caches are not inputs.
    """
    path = Path(path)
    if path.is_file():
        return {'kind': 'file', 'sha256': file_hash(path)}
    if not path.is_dir():
        raise ValueError('missing fingerprint path: ' + str(path))
    h = hashlib.sha256()
    count = 0
    for base, dirs, files in os.walk(str(path), followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in IGNORE)
        for name in dirs:
            if (Path(base) / name).is_symlink():
                raise ValueError('directory symlink inside fingerprint path: ' + str(Path(base) / name))
        for name in sorted(files):
            if name.endswith(('.pyc', '.pyo')):
                continue
            file = Path(base) / name
            if not file.is_file():
                raise ValueError('non-file input in fingerprint tree: ' + str(file))
            rel = str(file.relative_to(path))
            entry = [rel, os.readlink(str(file)) if file.is_symlink() else None, file_hash(file)]
            h.update(canonical(entry) + b'\n')
            count += 1
    return {'kind': 'tree', 'sha256': h.hexdigest(), 'files': count}


def _path(value, cwd):
    value = Path(value).expanduser()
    return (cwd / value).resolve() if not value.is_absolute() else value.resolve()


def _option(argv, key):
    if argv.count(key) > 1:
        raise ValueError('duplicate argument: ' + key)
    if key not in argv:
        return None
    i = argv.index(key)
    if i + 1 == len(argv) or argv[i + 1].startswith('--'):
        raise ValueError('missing value for ' + key)
    return argv[i + 1]


def _inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError('input escapes bound root: ' + str(relative))
    return path


def _frontend_paths(argv, cwd):
    required = ('--cohort', '--scope', '--dataset', '--raw-root', '--config', '--device', '--batch-size')
    for key in required:
        if _option(argv, key) is None:
            raise ValueError('frontend cache requires explicit ' + key)
    scope = _option(argv, '--scope')
    if scope not in ('development', 'audit'):
        raise ValueError('scope must be development or audit')
    if not _option(argv, '--batch-size').isdigit() or int(_option(argv, '--batch-size')) < 1:
        raise ValueError('frontend batch size must be positive')
    cohort_path = _path(_option(argv, '--cohort'), cwd)
    dataset = _path(_option(argv, '--dataset'), cwd)
    raw_root = _path(_option(argv, '--raw-root'), cwd)
    config_path = _path(_option(argv, '--config'), cwd)
    cohort = load_json(cohort_path)
    if cohort.get('schema_version') != 'research10.cohort.v1':
        raise ValueError('unsupported cohort schema')
    if dataset != Path(cohort['dataset_root']).resolve() or raw_root != Path(cohort['raw_root']).resolve():
        raise ValueError('dataset/raw-root differ from frozen cohort')
    rows, manifest_hashes = cohort['cohort'], cohort['input_manifest_sha256']
    if not rows or len({row['floor_uid'] for row in rows}) != len(rows):
        raise ValueError('cohort floors missing or duplicated')
    if set(manifest_hashes) != {row['floor_uid'] for row in rows}:
        raise ValueError('cohort manifest fingerprints incomplete')
    config = load_json(config_path)
    paths = {cohort_path, config_path}
    for name in ('bi_layout_config', 'bi_layout_checkpoint', 'opening_checkpoint'):
        paths.add(_path(config[name], ROOT))
    selected_count = 0
    allowed = {'fit', 'selection', 'calibration'} if scope == 'development' else {'audit'}
    for row in rows:
        if row['role'] not in ('fit', 'selection', 'calibration', 'audit'):
            raise ValueError('unknown cohort role')
        manifest_path = _inside(dataset, row['input_manifest'])
        if file_hash(manifest_path) != manifest_hashes[row['floor_uid']]:
            raise ValueError('cohort input manifest hash mismatch: ' + row['floor_uid'])
        manifest = load_json(manifest_path)
        if manifest['floor_uid'] != row['floor_uid']:
            raise ValueError('cohort input manifest floor differs')
        paths.add(manifest_path)
        if row['role'] in allowed:
            selected_count += 1
            for obs in manifest['observations']:
                if obs['rgb']['root_key'] != 'zind_raw':
                    raise ValueError('unsupported RGB root key')
                paths.add(_inside(raw_root, obs['rgb']['relative_path']))
    if not selected_count:
        raise ValueError('cohort has no floors in requested scope')
    return paths, scope, selected_count


def build_contract(spec, job_dir):
    kind = spec.get('kind')
    if kind not in ('floor-train', 'simulation', 'frontend-cache', 'simulation-cache'):
        raise ValueError('unsupported job kind')
    is_cache = kind in ('frontend-cache', 'simulation-cache')
    cwd = Path(spec.get('cwd', str(ROOT))).expanduser().resolve()
    if not cwd.is_dir():
        raise ValueError('cwd does not exist')
    argv = spec.get('argv')
    if not isinstance(argv, list) or not argv or not all(isinstance(x, str) and x for x in argv):
        raise ValueError('argv must be a non-empty string array')
    argv = list(argv)
    executable = shutil.which(argv[0])
    if not executable:
        raise ValueError('command executable not found: ' + argv[0])
    argv[0] = str(Path(executable).resolve())
    if kind == 'floor-train':
        direct = len(argv) >= 3 and _path(argv[1], cwd) == ROOT / 'tools' / 'run_floor_layout.py' and argv[2] == 'train'
        routed = len(argv) >= 6 and argv[1:6] == ['-m', 'portallayout', 'floor', 'run', 'train']
        if not (direct or routed):
            raise ValueError('floor-train command must invoke the PortalLayout floor train entry point')
    if kind == 'frontend-cache' and not (len(argv) >= 2 and
            _path(argv[1], cwd) == ROOT / 'tools' / 'research10_frontend_cache.py'):
        raise ValueError('frontend-cache command must invoke the dedicated frontend cache entry point')
    if '--resume' in argv or any(x.startswith('--resume=') for x in argv):
        raise ValueError('runner supplies --resume itself')
    if any(x.startswith('--stage=') or x.startswith('--output=') for x in argv):
        raise ValueError('use separated --stage and --output arguments')
    stage = 'frontend-cache' if is_cache else _option(argv, '--stage')
    if not is_cache and stage not in ('pairs', 'auxiliary'):
        raise ValueError('floor training requires explicit stage pairs or auxiliary')
    output = _path(spec['output_dir'], cwd)
    if _option(argv, '--output') is None or _path(_option(argv, '--output'), cwd) != output:
        raise ValueError('command --output must match output_dir')
    if output == job_dir or output in job_dir.parents or job_dir in output.parents:
        raise ValueError('job and trainer output directories must be separate')
    checkpoint_name = 'cache_contract.json' if is_cache else 'last.pt'
    checkpoint = _path(spec.get('checkpoint', str(output / checkpoint_name)), cwd)
    if checkpoint != output / checkpoint_name:
        raise ValueError('checkpoint must be output/' + checkpoint_name)
    retries = spec.get('max_retries', 2)
    if not isinstance(retries, int) or isinstance(retries, bool) or not 0 <= retries <= 5:
        raise ValueError('max_retries must be an integer between 0 and 5')
    resources = spec.get('fingerprint_paths', [])
    if not isinstance(resources, list) or (not resources and not is_cache) or not all(isinstance(x, str) for x in resources):
        raise ValueError('fingerprint_paths must list source data and model inputs')
    paths = {_path(p, cwd) for p in resources}
    paths.update((ROOT / 'portallayout', SCRIPT, Path(argv[0])))
    if not is_cache:
        paths.add(ROOT / 'tools' / 'run_floor_layout.py')
    cache_scope, cache_floors = None, None
    if is_cache:
        frontend_paths, cache_scope, cache_floors = _frontend_paths(argv, cwd)
        paths.update(frontend_paths)
    # Script entry points are part of the identity, including test harnesses.
    for arg in argv[1:]:
        candidate = _path(arg, cwd)
        if arg.endswith('.py') and candidate.is_file():
            paths.add(candidate)
    for key in (() if is_cache else ('--dataset', '--prediction-cache', '--supervision', '--pair-checkpoint', '--house-protocol',
                '--auxiliary-initialization')):
        value = _option(argv, key)
        if value is None:
            continue
        used = _path(value, cwd)
        if not used.exists():
            raise ValueError('missing command input ' + key + ': ' + str(used))
        if not any(used == p or p in used.parents for p in paths):
            raise ValueError('input must be covered by fingerprint_paths: ' + key)
    for p in paths:
        if output == p or p in output.parents or job_dir == p or p in job_dir.parents:
            raise ValueError('job/output cannot be inside an immutable input: ' + str(p))
    if stage == 'pairs' and any(_option(argv, key) is None for key in ('--dataset', '--prediction-cache', '--supervision')):
        raise ValueError('pairs require explicit --dataset, --prediction-cache and --supervision')
    if stage == 'auxiliary' and any(_option(argv, key) is None for key in ('--supervision', '--pair-checkpoint')):
        raise ValueError('auxiliary requires explicit --supervision and --pair-checkpoint')
    purpose = spec.get('purpose', 'diagnostic')
    if purpose not in ('diagnostic', 'formal'):
        raise ValueError('purpose must be diagnostic or formal')
    if kind in ('simulation', 'simulation-cache') and purpose == 'formal':
        raise ValueError('simulation cannot be formal training')
    if kind == 'frontend-cache' and purpose == 'formal':
        raise ValueError('frontend cache cannot be formal training')
    if not is_cache and any(_option(argv, key) is None for key in ('--seed', '--mode', '--device')):
        raise ValueError('training requires explicit --seed, --mode and --device')
    if _option(argv, '--device') == 'auto':
        raise ValueError('training device must be explicit, not auto')
    if purpose == 'diagnostic' and not is_cache:
        step_key = '--pair-steps' if stage == 'pairs' else '--auxiliary-steps'
        steps = _option(argv, step_key)
        if steps is None or not steps.isdigit() or not 1 <= int(steps) <= 5000:
            raise ValueError('diagnostic job requires explicit 1..5000 steps')
    elif purpose == 'formal':
        gate = spec.get('gate_report')
        if not isinstance(gate, str):
            raise ValueError('formal job requires gate_report')
        gate_path = _path(gate, cwd)
        if output == gate_path or output in gate_path.parents or job_dir == gate_path or job_dir in gate_path.parents:
            raise ValueError('gate report must be outside job and training outputs')
        report = load_json(gate_path)
        if report.get('status') != 'ready' or any(
                report.get('rounds', {}).get('R{:02d}'.format(i)) != 'passed' for i in range(1, 11)):
            raise ValueError('formal training gate report is not ready across all ten rounds')
        paths.add(gate_path)
    fingerprint = {str(p): tree_hash(p) for p in sorted(paths)}
    cuda_devices = spec.get('cuda_visible_devices', os.environ.get('CUDA_VISIBLE_DEVICES'))
    if cuda_devices is not None and not isinstance(cuda_devices, str):
        raise ValueError('cuda_visible_devices must be a string or null')
    environment = {'CUDA_VISIBLE_DEVICES': cuda_devices, 'PYTHONHASHSEED': '0',
                   'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                   'PYTHONUNBUFFERED': '1'}
    identity = {'schema': SCHEMA, 'kind': spec['kind'], 'purpose': purpose, 'cwd': str(cwd), 'argv': argv,
                'output_dir': str(output), 'checkpoint': str(checkpoint), 'stage': stage,
                'max_retries': retries, 'fingerprint_paths': fingerprint, 'environment': environment,
                'cache_scope': cache_scope, 'cache_floor_count': cache_floors}
    return dict(identity, contract_sha256=hashlib.sha256(canonical(identity)).hexdigest())


def check_contract(contract):
    value = dict(contract)
    expected = value.pop('contract_sha256', None)
    if value.get('schema') != SCHEMA or hashlib.sha256(canonical(value)).hexdigest() != expected:
        raise ValueError('job contract was modified or invalid')
    for path, expected_hash in value['fingerprint_paths'].items():
        if tree_hash(path) != expected_hash:
            raise ValueError('contract source or data changed: ' + path)


def load_json(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def prepare(spec_file, job_dir):
    job_dir = Path(job_dir).expanduser().resolve()
    if job_dir.exists() and any(job_dir.iterdir()):
        raise ValueError('job directory must be new/empty')
    contract = build_contract(load_json(spec_file), job_dir)
    output = Path(contract['output_dir'])
    if output.exists() and any(output.iterdir()):
        raise ValueError('training output must be new/empty at prepare')
    job_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(job_dir / 'contract.json', contract)
    atomic_json(job_dir / 'status.json', {'status': 'prepared', 'updated_at': now(),
        'contract_sha256': contract['contract_sha256'], 'attempts': 0, 'failures': 0})
    return contract


def _resource(pid):
    rss = None
    try:
        for row in Path('/proc/{}/status'.format(pid)).read_text().splitlines():
            if row.startswith('VmRSS:'):
                rss = int(row.split()[1])
                break
    except (OSError, ValueError):
        pass
    gpu = None
    try:
        q = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                            '--format=csv,noheader,nounits'], text=True, capture_output=True, timeout=3)
        if q.returncode == 0:
            matches = [int(x.split(',')[1].strip()) for x in q.stdout.splitlines()
                       if len(x.split(',')) == 2 and x.split(',')[0].strip() == str(pid)]
            gpu = sum(matches) if matches else 0
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return {'rss_kib': rss, 'gpu_memory_mib': gpu}


def _fatal_log(path):
    try:
        with Path(path).open('rb') as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 65536))
            tail = stream.read().decode('utf-8', errors='replace').lower()
    except OSError:
        return None
    for token in ('nonfinite', 'non-finite', 'out of memory', 'contract differs',
                  'contract changed', 'source code changed', 'source cache changed'):
        if token in tail:
            return token
    return None


def run(job_dir):
    job_dir = Path(job_dir).expanduser().resolve()
    contract = load_json(job_dir / 'contract.json')
    status_path, log_path = job_dir / 'status.json', job_dir / 'job.log'
    state = load_json(status_path)
    with (job_dir / 'runner.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('another runner owns this job')
        state = load_json(status_path)
        if state.get('contract_sha256') != contract.get('contract_sha256'):
            raise ValueError('status and contract fingerprints differ')
        if state.get('status') not in ('prepared', 'interrupted', 'running'):
            raise ValueError('job is not resumable from state: ' + str(state.get('status')))
        interrupted = [False]
        process = [None]
        def stop(signum, _frame):
            interrupted[0] = True
            if process[0] is not None and process[0].poll() is None:
                try:
                    process[0].terminate()
                except ProcessLookupError:
                    pass
        old_handlers = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
        def update(**values):
            state.update(values, updated_at=now())
            atomic_json(status_path, state)
        try:
            while True:
                try:
                    check_contract(contract)
                except (ValueError, OSError) as exc:
                    update(status='blocked', reason=str(exc))
                    return 2
                if interrupted[0]:
                    update(status='interrupted', reason='service stopped before child start', pid=None)
                    return 130
                attempts = state.get('attempts', 0)
                checkpoint = Path(contract['checkpoint'])
                output = Path(contract['output_dir'])
                is_cache = contract['kind'] in ('frontend-cache', 'simulation-cache')
                has_checkpoint = checkpoint.is_file() and checkpoint.stat().st_size > 0
                retry_fresh = bool(is_cache and attempts and not has_checkpoint and
                                   (not output.exists() or not any(output.iterdir())))
                if attempts and not has_checkpoint and not retry_fresh:
                    update(status='blocked', reason='no valid checkpoint after interrupted or failed attempt')
                    return 2
                if not attempts and output.exists() and any(output.iterdir()):
                    update(status='blocked', reason='training output was not empty before first run')
                    return 2
                if state.get('failures', 0) > contract['max_retries']:
                    update(status='blocked', reason='retry limit exhausted')
                    return 2
                argv = list(contract['argv'])
                if attempts and not retry_fresh:
                    argv += ['--resume'] if is_cache else ['--resume', str(checkpoint)]
                update(status='running', attempts=attempts + 1, started_at=now(),
                       resumed=bool(attempts and not retry_fresh), pid=None, reason=None)
                env = os.environ.copy()
                for key, value in contract['environment'].items():
                    if value is None:
                        env.pop(key, None)
                    else:
                        env[key] = value
                with log_path.open('ab', buffering=0) as log:
                    log.write(('\n[{}] attempt {} resume={}\n'.format(now(), attempts + 1, bool(attempts and not retry_fresh))).encode())
                    if interrupted[0]:
                        update(status='interrupted', attempts=attempts,
                               reason='service stopped before child start', pid=None)
                        return 130
                    try:
                        process[0] = subprocess.Popen(argv, cwd=contract['cwd'], env=env,
                                                      stdout=log, stderr=subprocess.STDOUT)
                    except OSError as exc:
                        update(status='blocked', reason='cannot start child: ' + str(exc))
                        return 2
                    if interrupted[0]:
                        process[0].terminate()
                    update(pid=process[0].pid)
                    while process[0].poll() is None:
                        try:
                            rc = process[0].wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            update(resources=_resource(process[0].pid))
                    rc = process[0].returncode
                if interrupted[0]:
                    update(status='interrupted', reason='service stopped; resume requires checkpoint',
                           exit_code=rc, pid=None)
                    return 130
                if rc == 0:
                    try:
                        check_contract(contract)
                    except (ValueError, OSError) as exc:
                        update(status='blocked', reason='inputs changed during training: ' + str(exc), pid=None)
                        return 2
                    required_files = ('cache_contract.json', 'run_report.json') if is_cache else (
                        'last.pt', 'checkpoint.pt', 'training_report.json')
                    if not all((output / name).is_file() and (output / name).stat().st_size
                               for name in required_files):
                        update(status='blocked', reason='successful exit without required artifacts', pid=None)
                        return 2
                    try:
                        report = load_json(output / ('run_report.json' if is_cache else 'training_report.json'))
                    except (OSError, json.JSONDecodeError) as exc:
                        update(status='blocked', reason='invalid process report: ' + str(exc), pid=None)
                        return 2
                    if is_cache:
                        if report.get('status') != 'complete' or report.get('input_errors') != [] or report.get('ground_truth_read') is not False:
                            update(status='blocked', reason='cache report is incomplete or not RGB-only', pid=None)
                            return 2
                        if report.get('floor_samples') != contract['cache_floor_count']:
                            update(status='blocked', reason='cache report floor count differs from cohort scope', pid=None)
                            return 2
                        update(status='completed', exit_code=0, finished_at=now(), pid=None,
                               note='cache completed; no experiment quality gates were assessed')
                    else:
                        if report.get('training_status') not in ('trained', 'smoke_trained'):
                            update(status='blocked', reason='training report is incomplete', pid=None)
                            return 2
                        update(status='completed', exit_code=0, finished_at=now(), pid=None,
                               note='process completed; experiment quality gates were not assessed',
                               training_status=report['training_status'])
                    return 0
                fatal = _fatal_log(log_path)
                if fatal:
                    update(status='blocked', reason='non-retryable failure: ' + fatal, exit_code=rc, pid=None)
                    return 2
                update(status='retrying', exit_code=rc, pid=None, failures=state.get('failures', 0) + 1)
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)


def launch(job_dir):
    job_dir = Path(job_dir).expanduser().resolve()
    contract, state = load_json(job_dir / 'contract.json'), load_json(job_dir / 'status.json')
    if state.get('contract_sha256') != contract.get('contract_sha256'):
        raise ValueError('status and contract fingerprints differ')
    if state.get('status') not in ('prepared', 'interrupted', 'running'):
        raise ValueError('job is not launchable from state: ' + str(state.get('status')))
    check_contract(contract)
    unit = 'research10-' + hashlib.sha256(str(job_dir).encode()).hexdigest()[:16]
    cmd = ['systemd-run', '--user', '--unit=' + unit, '--collect',
           '--property=Restart=no', '--property=KillMode=control-group',
           '--property=TimeoutStopSec=30', '--working-directory=' + str(ROOT),
           sys.executable, str(SCRIPT), 'run', '--job-dir', str(job_dir)]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode:
        raise ValueError('systemd launch failed: ' + result.stderr.strip())
    return {'unit': unit + '.service', 'job_dir': str(job_dir), 'message': result.stderr.strip()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'launch', 'run', 'status'):
        item = subs.add_parser(name)
        item.add_argument('--job-dir', required=True)
        if name == 'prepare':
            item.add_argument('--spec', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare':
            result = prepare(args.spec, args.job_dir)
            print(json.dumps({'status': 'prepared', 'contract_sha256': result['contract_sha256']}, ensure_ascii=False))
        elif args.command == 'launch':
            print(json.dumps(launch(args.job_dir), ensure_ascii=False))
        elif args.command == 'run':
            return run(args.job_dir)
        else:
            print(json.dumps(load_json(Path(args.job_dir) / 'status.json'), indent=2, ensure_ascii=False))
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        print('research10-background: ' + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
