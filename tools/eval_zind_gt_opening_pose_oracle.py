"""Two-edge train-only GT-opening oracle using the unchanged production solver.

GT endpoints/correspondence replace proposal and Matcher. Frozen predicted
pre-Manhattan boundaries remain the solver's orientation-disambiguation input.
GT transforms, global endpoints and metric scales never enter solve_pose().
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from portallayout.datasets.zind_pair_mining import _interfaces, relative_zind_transform
from tools.evaluate_zind_room_group import _bi_points_to_zind, _physical_candidate_pairs, _pose
from tools.evaluate_zind_portal_geometry_val import evaluate_geometry_mode
from portallayout.geometry.joint_layout import build_joint_layout_from_endpoints

FROZEN = ROOT / 'outputs/zind_room_group_debug/t3_best_0665_floor03_candidate_local_m2'
OUTPUT = ROOT / 'outputs/zind_pose_oracle_gt_opening_0665_floor03'
EDGES = [('04-01', '0665_floor_03_pano_34_pano_37_3'),
         ('01-05', '0665_floor_03_pano_34_pano_30_4')]


def solve_pose(endpoints_a, endpoints_b, boundary_a, boundary_b):
    """Minimal adapter: explicit ZInD xy boundaries avoid layout_xz conversion.

Empty layout dictionaries only omit display heights. Production's endpoint
fitting, scale calibration and centroid-side selection are called unchanged.
No GT pose, score, candidate index or ground-truth polygon is accepted here.
    """
    joint = build_joint_layout_from_endpoints(
        {}, {}, endpoints_a, endpoints_b, boundary_a=boundary_a,
        boundary_b=boundary_b, calibrate_scale=True,
        coordinate_system='ZInD panorama-A local xy',
    )
    return _pose(np.asarray(joint['alignment']['roomBToWorld'])), joint['alignment']


def assess(pose, target, factor):
    # Existing evaluator; omit BEV because this experiment evaluates pose only.
    return evaluate_geometry_mode(
        {'pose_valid': True, 'relative_pose': pose},
        {'is_match': True, 'relative_pose': target, 'metric_factor_A': factor,
         'joint_bev_target': {'instances': []}},
    )


def opening_description(interface):
    points = np.asarray(interface['local'])
    delta = points[1] - points[0]
    width = float(np.linalg.norm(delta))
    return {'type': interface['type'], 'annotation_index': interface['index'],
            'local_endpoints_xy': points.tolist(),
            'global_endpoints_xy': np.asarray(interface['global']).tolist(),
            'width_local': width, 'tangent_xy': (delta / width).tolist(),
            'left_normal_xy': [-float(delta[1] / width), float(delta[0] / width)],
            'normal_usage': 'diagnostic only; solver uses centroid-side rule',
            'endpoint_order': 'original annotation order, never chosen using GT pose'}


def run(output):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {output}')
    inputs = {}

    def read(path):
        path = Path(path).resolve()
        blob = path.read_bytes()
        inputs[str(path)] = hashlib.sha256(blob).hexdigest()
        return json.loads(blob)

    manifest = read(ROOT / 'data/ZInD_matching/train.json')
    assert manifest['split'] == 'train'
    group = read(ROOT / 'data/ZInD_matching_groups/debug_train_0665_floor_03.json')
    annotation_path = Path(manifest['dataRoot']) / '0665/zind_data.json'
    annotation = read(annotation_path)
    floor_scale = float(annotation['scale_meters_per_coordinate']['floor_03'])
    records = {p['id']: p for p in manifest['pairs'] if p['id'] in dict(EDGES).values()}
    assert len(records) == 2
    rows = []
    for label, pair_id in EDGES:
        record = records[pair_id]
        assert record['scene_id'] == '0665' and record['floor_id'] == 'floor_03'
        assert record['supervision']['is_match'] is True
        boundaries, panos, interfaces = [], [], []
        for side in ('a', 'b'):
            room = record['room_' + side]
            mapping = group['room_to_pano'][room]
            pano = annotation['merger']['floor_03'][room][mapping['partial_room_id']][mapping['pano_id']]
            panos.append(pano)
            # Dataset's existing annotation adapter uses layout_complete, raw fallback.
            interfaces.append(_interfaces(pano))
            frozen_room = read(FROZEN / 'rooms' / (room + '.json'))
            assert frozen_room['pano_id'] == mapping['pano_id']
            boundaries.append(_bi_points_to_zind(frozen_room['single_room_layout'][
                'layout_polygon_pre_manhattan_local_bi_xz']))
        ia = int(record['supervision']['target_candidate_a'])
        ib = int(record['supervision']['target_candidate_b'])
        selected = [interfaces[0][ia], interfaces[1][ib]]
        for side, interface in zip(('a', 'b'), selected):
            candidate = record['candidates_' + side][ia if side == 'a' else ib]
            np.testing.assert_allclose(interface['local'], candidate['localEndpoints'], atol=1e-12, rtol=0)
            np.testing.assert_allclose(interface['global'], candidate['globalEndpoints'], atol=1e-12, rtol=0)
        physical = _physical_candidate_pairs(record, 0.06)
        assert any(a == ia and b == ib for a, b, _ in physical)
        # Prediction is complete before any GT relative transform is constructed.
        pose, alignment = solve_pose(selected[0]['local'], selected[1]['local'], *boundaries)
        reversal_deltas = []
        for reverse_a, reverse_b in ((True, False), (False, True), (True, True)):
            ea = selected[0]['local'][::-1] if reverse_a else selected[0]['local']
            eb = selected[1]['local'][::-1] if reverse_b else selected[1]['local']
            alternate, _ = solve_pose(ea, eb, *boundaries)
            reversal_deltas.append(float(np.max(np.abs(np.asarray(alternate['T_B_to_A']) - pose['T_B_to_A']))))
        target_matrix = relative_zind_transform(panos[0]['floor_plan_transformation'], panos[1]['floor_plan_transformation'])
        np.testing.assert_allclose(target_matrix, record['supervision']['relative_transform_b_to_a'], atol=1e-12, rtol=0)
        target = _pose(target_matrix)
        factor = float(panos[0]['floor_plan_transformation']['scale']) * floor_scale
        metrics = assess(pose, target, factor)
        # Read only the two positive frozen pair files, exclusively for comparison.
        baseline = read(FROZEN / 'pairs' / (pair_id + '.json'))
        assert baseline['prediction']['room_a'] == record['room_a']
        assert baseline['prediction']['room_b'] == record['room_b']
        baseline_metrics = assess(baseline['prediction']['relative_pose'], target, factor)
        keys = ['yaw_error_degrees', 'translation_error_meters', 'absolute_log_scale_error']
        for key in keys:
            np.testing.assert_allclose(baseline_metrics[key], baseline['portallayout.evaluation'][key], atol=1e-10, rtol=0)
        rows.append({
            'pair': label, 'pair_id': pair_id, 'house_id': '0665', 'floor_id': 'floor_03',
            'split': 'train', 'room_a': record['room_a'], 'room_b': record['room_b'],
            'pano_a': record['pano_a'], 'pano_b': record['pano_b'],
            'transform_direction': 'B_to_A', 'coordinate_system': 'ZInD panorama-A local xy',
            'opening_source': 'gt', 'correspondence_source': 'gt',
            'boundary_source': 'frozen predicted pre-Manhattan polygons; used for centroid disambiguation',
            'gt_opening_a': opening_description(selected[0]), 'gt_opening_b': opening_description(selected[1]),
            'gt_correspondence_indices': [ia, ib], 'physical_compatible_correspondences': physical,
            'pred_pose': pose, 'gt_pose': target, 'alignment': alignment,
            'pred_relative_yaw_deg': pose['relative_yaw_degrees'],
            'gt_relative_yaw_deg': target['relative_yaw_degrees'],
            'pred_translation_xy_m': (np.asarray(pose['translation']) * factor).tolist(),
            'gt_translation_xy_m': (np.asarray(target['translation']) * factor).tolist(),
            'pred_translation_magnitude_m': float(np.linalg.norm(pose['translation']) * factor),
            'gt_translation_magnitude_m': float(np.linalg.norm(target['translation']) * factor),
            'pred_scale': pose['relative_scale'], 'gt_scale': target['relative_scale'],
            'pred_log_scale': math.log(pose['relative_scale']),
            'gt_log_scale': math.log(target['relative_scale']),
            'absolute_scale_error': abs(pose['relative_scale'] - target['relative_scale']),
            'metric_factor_A': factor, 'metric_factor_usage': 'reporting/evaluation only, never solver input',
            'metrics': metrics, 'baseline_frozen_metrics': baseline_metrics,
            'improvement_absolute': {k: baseline_metrics[k] - metrics[k] for k in keys},
            'improvement_percent': {k: 100 * (1 - metrics[k] / baseline_metrics[k]) for k in keys},
            'endpoint_reversal_transform_max_abs_deltas': reversal_deltas,
            'prediction_has_gt_pose_access': False,
        })
    for path, digest in inputs.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    summary = {
        'purpose': 'two-edge train-only qualitative pose oracle; no generalization claim',
        'pair_count': 2, 'split': 'train', 'test_data_accessed': False,
        'training_started': False, 'gpu_forward_executed': False, 'model_loaded': False,
        'matcher_executed': False, 'proposal_executed': False,
        'checkpoint_modified': False, 'gt_pose_used_for_prediction': False,
        'gt_room_polygon_used_for_prediction': False,
        'solver': 'portallayout.geometry.joint_layout.build_joint_layout_from_endpoints',
        'evaluator': 'tools.evaluate_zind_portal_geometry_val.evaluate_geometry_mode',
        'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'input_sha256': inputs,
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [Path(__file__).resolve(), ROOT/'portallayout/geometry/joint_layout.py',
                                    ROOT/'tools/evaluate_zind_portal_geometry_val.py']},
        'pairs': [{k: row[k] for k in ['pair', 'room_a', 'room_b', 'metrics',
                  'baseline_frozen_metrics', 'improvement_absolute', 'improvement_percent',
                  'endpoint_reversal_transform_max_abs_deltas']} for row in rows],
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, value in [('per_pair.json', rows), ('summary.json', summary)]:
        (output/name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(summary['pairs'], indent=2, allow_nan=False))
    return rows


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output_dir.resolve())
