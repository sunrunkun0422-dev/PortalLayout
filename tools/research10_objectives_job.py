#!/usr/bin/env python3
"""Two bounded seed-123 objective diagnostics after R05 has no workpoint.

Each objective has an independent 200-step smoke and, only when the smoke has
finite gradients and updates, an independent <=5000-step diagnostic run. This
does not train a formal model, change the fixed protocol or tune on calibration.
"""
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
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools import research10_background as bg

BASE = ROOT/'experiments/portallayout_design10_20261105'
JOB = BASE/'jobs/objectives_seed123'
OUTPUT = BASE/'R04/objectives_seed123'
PYTHON = Path('/home/feixia/anaconda3/envs/bi_layout/bin/python')


def unit(job):
    return 'research10-' + hashlib.sha256(str(job.resolve()).encode()).hexdigest()[:16] + '.service'


def active(job):
    return subprocess.run(['systemctl', '--user', 'is-active', '--quiet', unit(job)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def training_spec(objective, stage, cache, dataset, weak, protocol, cohort):
    if objective not in ('assignment', 'positive_margin') or stage not in ('smoke', 'diagnostic'):
        raise ValueError('unsupported objective or stage')
    output = OUTPUT/objective/stage
    argv = [str(PYTHON), str(ROOT/'tools/run_floor_layout.py'), 'train', '--stage', 'pairs',
            '--dataset', str(dataset), '--house-protocol', str(protocol), '--prediction-cache', str(cache),
            '--supervision', str(weak), '--output', str(output), '--device', 'cuda',
            '--mode', 'smoke' if stage == 'smoke' else 'full', '--seed', '123', '--protocol-seed', '123',
            '--pair-steps', '200' if stage == 'smoke' else '5000', '--pair-batch-size', '2',
            '--validation-interval', '100' if stage == 'smoke' else '500', '--patience', '5',
            '--selection-metric', 'cross_room_ap', '--pair-objective', objective]
    return {'kind': 'floor-train', 'purpose': 'diagnostic', 'cwd': str(ROOT), 'argv': argv,
            'output_dir': str(output), 'max_retries': 2,
            'fingerprint_paths': [str(dataset), str(cache), str(weak), str(protocol), str(cohort), str(Path(__file__).resolve())]}


def verify_report(report, stage):
    gradients = report.get('gradient_norms_first_step', {})
    steps = report.get('steps')
    expected = 'smoke_trained' if stage == 'smoke' else 'trained'
    if (report.get('training_status') != expected or not isinstance(steps, int) or
            (steps != 200 if stage == 'smoke' else not 1 <= steps <= 5000) or
            report.get('parameters_changed') is not True or
            not isinstance(report.get('skipped_batches'), int) or not 0 <= report['skipped_batches'] < steps or
            any(not isinstance(gradients.get(k), (int, float)) or not math.isfinite(gradients[k]) or gradients[k] <= 0
                for k in ('encoder', 'matcher'))):
        raise ValueError('training report does not demonstrate finite updates')


def main():
    JOB.mkdir(parents=True, exist_ok=True); OUTPUT.mkdir(parents=True, exist_ok=True)
    with (JOB/'coordinator.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cache = BASE/'R01/frontend_cache_development'; dataset = ROOT/'data/ZInD-FloorSet-v1/train_20261021'
        weak = BASE/'R04/pipeline_seed123/weak_supervision/opening_weak.json'
        protocol = BASE/'R01/house_protocol.json'; cohort = BASE/'R01/cohort.json'
        prior = BASE/'jobs/pipeline_seed123_v2/status.json'; r05 = BASE/'R05/opening_workpoint_seed123/matching_report.json'
        sources = [Path(__file__).resolve(), bg.SCRIPT, ROOT/'tools/run_floor_layout.py',
                   cache/'cache_contract.json', cache/'run_report.json', weak, protocol, cohort, prior, r05]
        payload = {'schema': 'research10.objective-diagnostics.v1',
                   'sources': {str(p): bg.file_hash(p) for p in sources},
                   'core_source_tree': bg.tree_hash(ROOT/'portallayout'),
                   'objectives': ['assignment', 'positive_margin'], 'seed': 123}
        payload['fingerprint'] = hashlib.sha256(bg.canonical(payload)).hexdigest()
        marker = JOB/'contract.json'
        if marker.exists():
            if bg.load_json(marker) != payload: raise ValueError('objective coordinator inputs changed')
        else: bg.atomic_json(marker, payload)
        prior_state, calibration = bg.load_json(prior), bg.load_json(r05)
        if prior_state.get('status') != 'completed' or calibration.get('calibration', {}).get('status') != 'calibration_infeasible':
            raise ValueError('requires completed diagnostic and infeasible R05 workpoint')
        def state(status, stage, **kwargs):
            bg.atomic_json(JOB/'status.json', dict(status=status, stage=stage, updated_at=bg.now(),
                           contract_sha256=payload['fingerprint'], **kwargs))
        try:
            for objective in payload['objectives']:
                for stage in ('smoke', 'diagnostic'):
                    subjob = JOB/objective/stage
                    spec = training_spec(objective, stage, cache, dataset, weak, protocol, cohort)
                    specfile = JOB/(objective+'_'+stage+'_spec.json')
                    if specfile.exists():
                        if bg.load_json(specfile) != spec: raise ValueError('training specification changed')
                    else: bg.atomic_json(specfile, spec)
                    if not (subjob/'contract.json').exists():
                        state('running', objective+'_'+stage+'_preparing')
                        bg.prepare(specfile, subjob)
                    job_state = bg.load_json(subjob/'status.json')
                    if job_state['status'] == 'prepared' and not active(subjob): bg.launch(subjob)
                    state('running', objective+'_'+stage+'_running', job_dir=str(subjob))
                    launched_at = time.monotonic()
                    while True:
                        job_state = bg.load_json(subjob/'status.json')
                        if job_state['status'] == 'completed': break
                        if job_state['status'] in ('blocked', 'interrupted', 'failed'):
                            raise ValueError('training blocked: '+str(job_state.get('reason')))
                        if job_state['status'] not in ('prepared', 'running', 'retrying'):
                            raise ValueError('unexpected child state: '+str(job_state['status']))
                        if time.monotonic()-launched_at > 60 and not active(subjob):
                            raise ValueError('training service exited without recording a terminal status')
                        time.sleep(10)
                    contract, report_path = bg.load_json(subjob/'contract.json'), OUTPUT/objective/stage/'training_report.json'
                    if job_state.get('contract_sha256') != contract.get('contract_sha256'):
                        raise ValueError('child training contract mismatch')
                    bg.check_contract(contract)
                    report = bg.load_json(report_path); verify_report(report, stage)
                    if report.get('objective') != objective: raise ValueError('training objective mismatch')
            for path, digest in payload['sources'].items():
                if bg.file_hash(path) != digest: raise ValueError('source changed: '+path)
            if bg.tree_hash(ROOT/'portallayout') != payload['core_source_tree']: raise ValueError('core source changed')
            state('completed', 'finished', note='two additional single-seed objective diagnostics only; no formal training')
        except BaseException as exc:
            state('blocked', 'stopped', reason=str(exc), error_type=type(exc).__name__)
            raise


if __name__ == '__main__': main()
