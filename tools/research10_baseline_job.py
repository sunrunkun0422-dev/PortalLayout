#!/usr/bin/env python3
"""Run the fixed selection baseline plan and scores without reading GT.

The job is restartable and fails closed if its immutable inputs change. Its
outputs are uncalibrated candidate scores, not a matching accuracy assessment.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from portallayout.utils.room_feature_cache import content_hash, fingerprint, write_json_atomic
from tools.research10_background import tree_hash


def run(output):
    base = ROOT/'experiments/portallayout_design10_20261105'
    output = Path(output).resolve()
    if output == base or base not in output.parents:
        raise ValueError('output must be a separate directory under the experiment')
    output.mkdir(parents=True, exist_ok=True)
    with (output/'job.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pipeline = json.loads((base/'jobs/pipeline_seed123_v2/status.json').read_text())
        if pipeline.get('status') != 'completed' or pipeline.get('stage') != 'done':
            raise ValueError('diagnostic pipeline is not complete')
        report = base/'R04/pipeline_seed123/diagnostic_pairs/training_report.json'
        checkpoint = base/'R04/pipeline_seed123/diagnostic_pairs/checkpoint.pt'
        if content_hash(report) != pipeline.get('diagnostic_report_sha256'):
            raise ValueError('diagnostic report does not match completed pipeline')
        cohort = base/'R01/cohort.json'; protocol = base/'R01/house_protocol.json'
        cache = base/'R01/frontend_cache_development'
        dataset = ROOT/'data/ZInD-FloorSet-v1/train_20261021'
        sources = [Path(__file__).resolve(), ROOT/'tools/research10_baseline_export.py',
                   ROOT/'tools/research10_matching_baselines.py', cohort, protocol,
                   cache/'cache_contract.json', cache/'run_report.json', checkpoint, report]
        contract = {'schema': 'research10.baseline-job.v1', 'sources': {str(p): content_hash(p) for p in sources},
                    'core_source_tree': tree_hash(ROOT/'portallayout'),
                    'scope': 'selection_all_pairs', 'geometry_scale': 'shared_camera_height_assumption',
                    'device': 'cpu', 'ground_truth_read': False}
        contract['fingerprint'] = fingerprint(contract)
        marker = output/'contract.json'
        if marker.exists():
            if json.loads(marker.read_text()) != contract:
                raise ValueError('baseline job contract changed')
        else:
            write_json_atomic(marker, contract)
        def status(stage, state, reason=None):
            write_json_atomic(output/'status.json', {'stage': stage, 'status': state, 'reason': reason,
                                                      'contract_fingerprint': contract['fingerprint']})
        def verify_sources():
            for path, digest in contract['sources'].items():
                if content_hash(path) != digest: raise ValueError('source changed: '+path)
            if tree_hash(ROOT/'portallayout') != contract['core_source_tree']:
                raise ValueError('core source tree changed')
        common = ['--cache', str(cache), '--dataset', str(dataset), '--house-protocol', str(protocol)]
        plan = output/'selection_fixed_pairs.json'; scores = output/'selection_baseline_scores.json'
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        try:
            if not plan.exists():
                status('plan', 'running')
                subprocess.run([sys.executable, str(ROOT/'tools/research10_baseline_export.py'), 'plan',
                                *common, '--roles', 'selection', '--max-pairs', '0', '--seed', '123',
                                '--output', str(plan)], cwd=str(ROOT), env=env, check=True)
            document = json.loads(plan.read_text())
            if (document.get('roles') != ['selection'] or document.get('selected_pairs') != document.get('population_pairs')
                    or not document.get('selected_pairs')):
                raise ValueError('plan does not cover every selection pair')
            verify_sources()
            if not scores.exists():
                status('score', 'running')
                subprocess.run([sys.executable, str(ROOT/'tools/research10_baseline_export.py'), 'score',
                                *common, '--plan', str(plan), '--checkpoint', str(checkpoint), '--device', 'cpu',
                                '--shared-camera-height-assumption', '--output', str(scores)],
                               cwd=str(ROOT), env=env, check=True)
            result = json.loads(scores.read_text())
            if (result.get('fixed_plan_sha256') != content_hash(plan) or
                    result.get('checkpoint_sha256') != content_hash(checkpoint) or
                    result.get('observation_pair_count') != document['selected_pairs'] or
                    result.get('ground_truth_read') is not False or result.get('scoring_only') is not True):
                raise ValueError('baseline score artifact is inconsistent')
            verify_sources()
            status('finished', 'completed')
        except (ValueError, OSError, subprocess.CalledProcessError) as exc:
            status('stopped', 'blocked', str(exc))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
