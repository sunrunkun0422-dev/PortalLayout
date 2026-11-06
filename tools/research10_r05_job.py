#!/usr/bin/env python3
"""Run a frozen-checkpoint opening calibration diagnostic on development data."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from portallayout.utils.room_feature_cache import content_hash, fingerprint, write_json_atomic
from tools.research10_background import tree_hash


def main():
    base = ROOT/'experiments/portallayout_design10_20261105'
    output = base/'R05/opening_workpoint_seed123'
    if output.exists() and any(output.iterdir()): raise ValueError('use a new R05 output directory')
    output.mkdir(parents=True, exist_ok=True)
    state_file = output/'status.json'
    def state(stage, status, **extra):
        write_json_atomic(state_file, dict(stage=stage, status=status, **extra))
    try:
        pipeline_path = base/'jobs/pipeline_seed123_v2/status.json'
        pipeline = json.loads(pipeline_path.read_text())
        if pipeline.get('status') != 'completed' or pipeline.get('stage') != 'done':
            raise ValueError('diagnostic pipeline is not complete')
        diagnostic = base/'R04/pipeline_seed123/diagnostic_pairs'
        training_report, checkpoint = diagnostic/'training_report.json', diagnostic/'checkpoint.pt'
        if content_hash(training_report) != pipeline.get('diagnostic_report_sha256'):
            raise ValueError('diagnostic report changed')
        cache = base/'R01/frontend_cache_development'
        dataset = ROOT/'data/ZInD-FloorSet-v1/train_20261021'
        weak = base/'R04/pipeline_seed123/weak_supervision/opening_weak.json'
        protocol = base/'R01/house_protocol.json'
        report = output/'matching_report.json'
        sources = [Path(__file__).resolve(), ROOT/'tools/report_opening_matches.py',
                   pipeline_path, training_report, checkpoint, weak, protocol,
                   protocol.parent/'cohort.json', cache/'cache_contract.json', cache/'run_report.json']
        contract = {'schema': 'research10.r05-workpoint-job.v1',
                    'sources': {str(path): content_hash(path) for path in sources},
                    'core_source_tree': tree_hash(ROOT/'portallayout'), 'roles': ['selection', 'calibration'],
                    'scope': 'development_weak_label_calibration_only', 'uses_audit_for_threshold': False}
        contract['fingerprint'] = fingerprint(contract)
        write_json_atomic(output/'contract.json', contract)
        state('scoring', 'running')
        cmd = [sys.executable, str(ROOT/'tools/report_opening_matches.py'),
               '--weak-supervision', str(weak), '--prediction-cache', str(cache),
               '--pair-checkpoint', str(checkpoint), '--dataset', str(dataset),
               '--house-protocol', str(protocol), '--roles', 'selection', 'calibration',
               '--calibrate', '--device', 'cuda', '--output', str(report)]
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)
        result = json.loads(report.read_text())
        if fingerprint({k:v for k,v in result.items() if k != 'fingerprint'}) != result.get('fingerprint'):
            raise ValueError('report fingerprint mismatch')
        if result['pair_checkpoint_sha256'] != content_hash(checkpoint) or 'audit' in result['roles']:
            raise ValueError('report model or role mismatch')
        for path, expected in contract['sources'].items():
            if content_hash(path) != expected: raise ValueError('source changed: '+path)
        if tree_hash(ROOT/'portallayout') != contract['core_source_tree']:
            raise ValueError('core source changed during evaluation')
        calibration = result['calibration']
        state('finished', 'completed_diagnostic_only', workpoint_status=calibration['status'],
              report_sha256=content_hash(report))
        print(json.dumps({'status': 'completed_diagnostic_only', 'workpoint_status': calibration['status']}))
    except BaseException as exc:
        state('stopped', 'blocked', error_type=type(exc).__name__, reason=str(exc))
        raise


if __name__ == '__main__': main()
