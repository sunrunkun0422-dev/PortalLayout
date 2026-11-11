#!/usr/bin/env python3
"""Offline metric pose diagnostics for frozen, uncalibrated pair predictions."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from portallayout.evaluation.floor_metrics import load_floor_ground_truth
from portallayout.geometry.floor_geometry import parameters, similarity
from portallayout.utils.room_feature_cache import content_hash, fingerprint, write_json_atomic


def metric_pose_error(predicted, ga, gb, metric_scale):
    """Compare B->A translation in meters using GT anchor scale for evaluation."""
    if metric_scale is None or not np.isfinite(metric_scale) or metric_scale <= 0:
        return None
    a, b, pred = similarity(ga), similarity(gb), similarity(predicted)
    truth = np.linalg.inv(a) @ b
    meters_per_anchor_unit = float(metric_scale)*math.sqrt(float(np.linalg.det(a[:2, :2])))
    distance = float(np.linalg.norm(pred[:2, 2]-truth[:2, 2]))*meters_per_anchor_unit
    delta = parameters(pred)[2]-parameters(truth)[2]
    angle = abs(math.degrees(math.atan2(math.sin(delta), math.cos(delta))))
    scale_error = abs(math.exp(parameters(pred)[3]-parameters(truth)[3])-1.)
    return {'translation_m': distance, 'rotation_deg': angle, 'relative_scale_error': scale_error,
            'correct_50cm_10deg': bool(distance <= .5 and angle <= 10.)}


def checked(path):
    data = json.loads(Path(path).read_text()); claimed = data.get('fingerprint')
    if not claimed or fingerprint({k:v for k,v in data.items() if k != 'fingerprint'}) != claimed:
        raise ValueError('invalid artifact fingerprint: '+str(path))
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--followup', required=True); p.add_argument('--cohort', required=True); p.add_argument('--output', required=True)
    args = p.parse_args(); followup = Path(args.followup).resolve(); cohort_path = Path(args.cohort).resolve()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()): raise ValueError('output must be new')
    contract = checked(followup/'run_contract.json'); barrier = checked(followup/'prediction_barrier.json')
    summary = checked(followup/'summary.json'); cohort = json.loads(cohort_path.read_text())
    if (summary['status'] != 'diagnostic_only' or content_hash(cohort_path) != contract['cohort_sha256'] or
            not barrier['all_predictions_committed_before_labels']):
        raise ValueError('followup is not a frozen prediction with matching cohort')
    selected = [r for r in cohort['cohort'] if r['role'] == 'selection']
    rows, sources = [], {}
    for floor in selected:
        uid = floor['floor_uid']; pred_path = followup/'predictions'/(uid+'.json'); audit_path = followup/'audits'/(uid+'.json')
        if content_hash(pred_path) != barrier['prediction_hashes'].get(uid): raise ValueError('prediction hash mismatch')
        pred, audit = checked(pred_path), checked(audit_path)
        if pred['status'] != 'predicted' or audit['status'] != 'audited': raise ValueError('floor prediction/audit unavailable')
        gt = load_floor_ground_truth(cohort['dataset_root'], uid)
        if gt['house_id'] != floor['house_id']: raise ValueError('GT house differs from cohort')
        for name in gt['source_files']: sources[name] = content_hash(name)
        sources[str(pred_path)] = content_hash(pred_path); sources[str(audit_path)] = content_hash(audit_path)
        scopes = {r['pair_id']: r['relationship'] for r in audit['geometry']['inventory']}
        for pair in pred['selected_pairs']:
            uid_a, uid_b = pair['observation_a'], pair['observation_b']
            ga, gb = gt['observations'][uid_a], gt['observations'][uid_b]
            scope = scopes[pair['pair_id']]
            valid = ga['pose_valid'] and gb['pose_valid']
            metric_available = gt['metric_scale'] is not None
            transform = pair.get('T_B_to_A')
            error, reason = None, None
            if transform is None: reason = 'no_pose'
            elif not valid: reason = 'pose_gt_unavailable'
            elif not metric_available: reason = 'metric_scale_unavailable'
            else:
                try: error = metric_pose_error(transform, ga['pose'], gb['pose'], gt['metric_scale'])
                except (ValueError, TypeError, np.linalg.LinAlgError): reason = 'invalid_prediction'
            rows.append({'floor_uid': uid, 'house_id': floor['house_id'], 'pair_id': pair['pair_id'], 'scope': scope,
                         'pose_gt_valid': bool(valid), 'metric_scale_available': metric_available,
                         'pose_predicted': transform is not None, 'reason': reason, 'error': error})
    if any(content_hash(name) != digest for name,digest in sources.items()): raise ValueError('source changed')
    summaries = {}
    for scope in ('cross_room', 'same_room', 'unknown_room'):
        subset = [r for r in rows if r['scope'] == scope]
        evaluated = [r for r in subset if r['error'] is not None]
        valid_metric = [r for r in subset if r['pose_gt_valid'] and r['metric_scale_available']]
        reasons = Counter(r['reason'] for r in subset if r['reason'])
        by_house = defaultdict(Counter)
        for r in subset:
            by_house[r['house_id']]['pairs'] += 1
            if r['error'] is not None:
                by_house[r['house_id']]['evaluated_poses'] += 1
                by_house[r['house_id']]['correct'] += int(r['error']['correct_50cm_10deg'])
        summaries[scope] = {'pairs': len(subset), 'gt_metric_pairs': len(valid_metric),
            'predicted_poses': sum(r['pose_predicted'] for r in subset), 'evaluated_poses': len(evaluated),
            'correct_50cm_10deg': sum(r['error']['correct_50cm_10deg'] for r in evaluated),
            'reasons': dict(reasons), 'houses': {h: dict(v) for h,v in sorted(by_house.items())}}
    output.mkdir(parents=True, exist_ok=True)
    with (output/'pairs.jsonl').open('x') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    result = {'status': 'diagnostic_only', 'unit': 'meters', 'translation_threshold_m': .5,
              'rotation_threshold_deg': 10., 'selection_pair_count': len(rows), 'scopes': summaries,
              'source_sha256': sources, 'pairs_sha256': content_hash(output/'pairs.jsonl'),
              'limitations': ['uncalibrated exploratory poses, not accepted connection edges',
                              'GT anchor scale is used only to express evaluation translation in meters',
                              'metric pair accuracy does not establish physical adjacency',
                              'data and training weights are not independent of the opening frontend']}
    write_json_atomic(output/'report.json', result)
    print(json.dumps({'status': result['status'], 'selection_pairs': len(rows), 'cross_room': summaries['cross_room']}))


if __name__ == '__main__': main()
