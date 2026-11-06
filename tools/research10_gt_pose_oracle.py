#!/usr/bin/env python3
"""Offline GT-opening pose upper bound on R01 fit/selection houses only.

Portal links are annotation-derived geometric proxies, not human confirmation.
Each GT transform is confined to evaluation. This is not model inference.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from portallayout.datasets.floor_auxiliary_supervision import pose_target
from portallayout.datasets.zind_floor_dataset import read_jsonl
from portallayout.evaluation.floor_metrics import load_floor_ground_truth
from portallayout.evaluation.pose_candidate_audit import normalized_endpoints
from portallayout.geometry.floor_geometry import points_transform
from portallayout.geometry.pose_geometry import pose_hypotheses
from portallayout.utils.room_feature_cache import content_hash, write_json_atomic


def candidate_status(labels):
    """GT-positive wins; gray labels remain unknown; no proposal is failure."""
    return 'success' if 1 in labels else 'unknown' if -1 in labels else 'failure'


def run_floor(dataset, cohort_row):
    uid = cohort_row['floor_uid']
    base = Path(dataset)/'labels'/uid
    paths = [base/n for n in ('floor_gt.json', 'observations_gt.jsonl', 'portal_links.jsonl')]
    source_hashes = {str(p): content_hash(p) for p in paths}
    gt = load_floor_ground_truth(dataset, uid)
    annotations = {a['observation_id']: a for a in read_jsonl(base/'observations_gt.jsonl')}
    interfaces = {}
    projection_errors = []
    for obs, ann in annotations.items():
        for item in ann['interfaces_by_layout']['layout_raw']:
            interfaces[item['interface_observation_id']] = (obs, item)
            if item.get('floor_endpoints') is None or not gt['observations'][obs]['pose_valid']:
                continue
            try:
                local = normalized_endpoints(ann, item)
                ground = np.asarray(item['floor_endpoints'])*[-1., 1.]
                projection_errors.append(float(np.max(np.abs(points_transform(local, gt['observations'][obs]['pose'])-ground))))
            except (TypeError, ValueError, KeyError):
                pass
    rows = []
    total_links, ambiguous_links = 0, 0
    for link in read_jsonl(base/'portal_links.jsonl'):
        total_links += 1
        if link.get('ambiguous'):
            ambiguous_links += 1
            continue
        ids = [link['source_interface_a'], link['source_interface_b']]
        row = dict(floor_uid=uid, house_id=cohort_row['house_id'], role=cohort_row['role'],
                   interface_ids=ids, link_type=link['type'], label_source=link.get('label_source'),
                   human_verified=bool(link.get('human_verified')), status='unknown', relationship='unknown_room',
                   candidate_count=0, labels=[], rotation_errors_degrees=[])
        try:
            aid, a = interfaces[ids[0]]; bid, b = interfaces[ids[1]]
            ga, gb = gt['observations'][aid], gt['observations'][bid]
            row['observation_ids'] = [aid, bid]
            row['relationship'] = ('same_room' if ga['room_id'] == gb['room_id'] else 'cross_room') if (
                ga['membership_confirmed'] and gb['membership_confirmed']) else 'unknown_room'
            if not ga['pose_valid'] or not gb['pose_valid']:
                row['reason'] = 'pose_gt_unavailable'
            else:
                truth = np.linalg.inv(ga['pose']) @ np.asarray(gb['pose'])
                ea = normalized_endpoints(annotations[aid], a)
                eb = normalized_endpoints(annotations[bid], b)
                candidates = pose_hypotheses(ea, eb, alpha=1.)
                evaluated = [pose_target(h['T_B_to_A'], truth) for h in candidates]
                labels = [x[0] for x in evaluated]
                # A gray-zone pose is not a negative label. Preserve it even
                # when a different hypothesis is known to be incorrect.
                status = candidate_status(labels)
                row.update(candidate_count=len(candidates), labels=labels,
                           rotation_errors_degrees=[x[1].get('rotation_degrees') for x in evaluated],
                           status=status, reason=('no_valid_pose_candidate' if not candidates else
                                                  'pose_label_in_gray_zone' if status == 'unknown' else None))
        except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as e:
            row['reason'] = type(e).__name__+': '+str(e)
        rows.append(row)
    if any(content_hash(p) != h for p, h in source_hashes.items()):
        raise ValueError('GT source changed during audit')
    return rows, {'checks': len(projection_errors), 'max_error': max(projection_errors, default=None),
                  'all_links': total_links, 'ambiguous_excluded': ambiguous_links}, source_hashes


def summarize(rows):
    result = {}
    for role in ('fit', 'selection'):
        result[role] = {}
        for scope in ('cross_room', 'same_room', 'unknown_room'):
            subset = [r for r in rows if r['role'] == role and r.get('relationship') == scope]
            counts = Counter(r['status'] for r in subset)
            by_house = defaultdict(Counter)
            for row in subset:
                by_house[row['house_id']][row['status']] += 1
            result[role][scope] = {'links': len(subset), 'success': counts['success'],
                'failure': counts['failure'], 'unknown': counts['unknown'],
                'success_including_unknown': counts['success']/len(subset) if subset else None,
                'houses': len(by_house), 'by_house': {h: dict(c) for h, c in sorted(by_house.items())}}
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', required=True); p.add_argument('--output', required=True)
    args = p.parse_args(argv)
    cohort_path = Path(args.cohort).resolve(); cohort = json.loads(cohort_path.read_text())
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()): raise ValueError('output must be new')
    output.mkdir(parents=True, exist_ok=True)
    rows, floors, hashes = [], [], {}
    for item in cohort['cohort']:
        if item['role'] not in ('fit', 'selection'): continue
        result, audit, source = run_floor(cohort['dataset_root'], item)
        rows.extend(result); hashes.update(source)
        floors.append(dict(floor_uid=item['floor_uid'], house_id=item['house_id'], role=item['role'],
                           links=len(result), coordinate=audit))
        print('floor={} role={} links={}'.format(item['floor_uid'], item['role'], len(result)), flush=True)
    if any(content_hash(p) != h for p, h in hashes.items()): raise ValueError('source changed before commit')
    with (output/'links.jsonl').open('x') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    report = {'status': 'completed_gt_oracle_only', 'cohort_sha256': content_hash(cohort_path),
              'cohort_roles': ['fit', 'selection'], 'official_val_or_test_used': False,
              'human_reviewed_correspondences': sum(r['human_verified'] for r in rows),
              'label_semantics': 'raw annotation geometry proxy, not visually verified',
              'denominator_semantics': 'non-ambiguous interface links, not distinct panorama pairs; unknown included',
              'projection_audit_limit': 'projection uses related annotations; not an independent sensor calibration',
              'pose_success_semantics': 'pose_target normalized forward and inverse <=0.25, angle<=10deg, relative scale<=0.1',
              'floors': floors, 'summary': summarize(rows), 'links_file_sha256': content_hash(output/'links.jsonl'),
              'sources_sha256': hashes}
    write_json_atomic(output/'report.json', report)
    print(json.dumps({'status': report['status'], 'floors': len(floors),
                      'summary': report['summary']['selection']}, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__': raise SystemExit(main())
