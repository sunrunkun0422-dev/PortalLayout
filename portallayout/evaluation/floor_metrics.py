"""Independent whole-floor evaluator. No GT-assisted assembly or room grouping."""
from dataclasses import dataclass, asdict
import csv
import json
import math
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import GeometryCollection
from shapely.ops import unary_union
from portallayout.geometry.floor_geometry import (polygon, similarity, parameters, transform_shape, geojson,
    partition_graph, point_segment_distances, sampled_segments)


@dataclass(frozen=True)
class FloorMetricProtocol:
    version: str = 'floor-all-panoramas.partition.v2'
    scale_track: str = 'S'
    translation_threshold_m: float = .50
    rotation_threshold_degrees: float = 10.
    junction_threshold_m: float = .10
    wall_threshold_m: float = .20
    wall_angle_degrees: float = 5.
    wall_overlap: float = .5
    graph_snap_m: float = .05
    collinear_degrees: float = 3.
    wall_sample_m: float = .02
    wall_distance_cap_m: float = 2.
    room_iou_threshold: float = .5


def load_floor_ground_truth(dataset_root, floor_uid):
    """Explicit offline boundary. Reflect *both* local and global ZInD axes.

    C G C is orientation preserving; G C alone is a reflection and is never
    silently interpreted as a camera yaw. Units remain native until evaluation.
    """
    from portallayout.datasets.zind_floor_dataset import safe_path
    base = safe_path(Path(dataset_root), 'labels/'+floor_uid)
    floor = json.loads((base/'floor_gt.json').read_text())
    raw = [json.loads(line) for line in (base/'observations_gt.jsonl').read_text().splitlines()]
    c = np.diag([-1., 1., 1.])
    observations, invalid = {}, []
    for row in raw:
        value = None
        if row.get('pose_gt_available'):
            try:
                height = 1. if row.get('coordinate_spec', {}).get('camera_height_normalized_to_one_by_source') else row['camera_height_annotation']
                if height is None or height <= 0: raise ValueError('source camera normalization unavailable')
                value = similarity(c @ np.asarray(row['G_local_to_floor_gt']) @ np.diag([height, height, 1.]) @ c).tolist()
            except (ValueError, KeyError, TypeError): pass
        observations[row['observation_id']] = {'pose': value, 'pose_valid': value is not None,
            'membership_confirmed': row.get('membership_status') == 'confirmed',
            'room_id': row['complete_room_uid']}
    rooms = {}
    for row in floor['complete_rooms']:
        p = polygon(row.get('polygon_floor_native'))
        if not row.get('geometry_available') or not row.get('geometry_consistent') or p is None:
            invalid.append(row['room_uid']); continue
        # GT room contours have no holes in the current source schema.
        points = np.asarray(p.exterior.coords)[:-1]*np.array([-1., 1.])
        rooms[row['room_uid']] = geojson(polygon(points))
    metric = floor.get('metric_scale')
    metric_valid = bool(floor.get('metric_valid') and metric is not None and np.isfinite(metric) and metric > 0)
    return {'floor_uid': floor_uid, 'house_id': floor['house_id'], 'observations': observations,
            'rooms': rooms, 'geometry_valid': bool(rooms) and not invalid, 'invalid_rooms': invalid,
            'metric_scale': float(metric) if metric_valid else None,
            'coordinate_frame': 'bi_floor_xz', 'offline_only': True,
            'source_files': [str(base/'floor_gt.json'), str(base/'observations_gt.jsonl')]}


def maximum_cardinality_matching(feasible, costs):
    """Lexicographic objective: cardinality first, cost second, with dummies."""
    feasible, costs = np.asarray(feasible, bool), np.asarray(costs, float)
    n, m = feasible.shape
    if not n or not m: return []
    if costs.shape != (n, m) or not np.isfinite(costs[feasible]).all():
        raise ValueError('invalid feasible matching costs')
    normalized = costs/max(float(np.max(np.abs(costs[feasible]))) if feasible.any() else 0., 1.)
    reward = min(n, m)+1.
    padded = np.zeros((n+m, n+m))
    padded[:n, :m] = np.where(feasible, normalized-reward, reward*(n+m+1))
    rows, cols = linear_sum_assignment(padded)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if i < n and j < m and feasible[i, j]]


def counts(tp, n, m):
    return {'tp': tp, 'fp': n-tp, 'fn': m-tp,
            'precision': tp/n if n else (0. if m else None),
            'recall': tp/m if m else None,
            'f1': 2.*tp/(n+m) if n+m else None}


def junction_metrics(pred, gt, tolerance):
    a = np.array([j['point'] for j in pred['junctions']]).reshape(-1, 2)
    b = np.array([j['point'] for j in gt['junctions']]).reshape(-1, 2)
    distance = np.linalg.norm(a[:, None]-b[None], axis=-1)
    match = maximum_cardinality_matching(distance <= tolerance+1e-10, distance)
    return dict(counts(len(match), len(a), len(b)), matches=match)


def wall_metrics(pred, gt, protocol):
    a = np.array([w['endpoints'] for w in pred['walls']]).reshape(-1, 2, 2)
    b = np.array([w['endpoints'] for w in gt['walls']]).reshape(-1, 2, 2)
    costs, feasible = np.zeros((len(a), len(b))), np.zeros((len(a), len(b)), bool)
    for i, x in enumerate(a):
        direction = x[1]-x[0]; length = np.linalg.norm(direction); u = direction/length
        for j, y in enumerate(b):
            delta = min(np.linalg.norm(x-y, axis=1).max(), np.linalg.norm(x-y[::-1], axis=1).max())
            v = y[1]-y[0]; other_length = np.linalg.norm(v)
            angle = math.degrees(math.acos(float(np.clip(abs(u@v/other_length), 0, 1))))
            xproj, yproj = sorted(x@u), sorted(y@u)
            overlap = max(0., min(xproj[1], yproj[1])-max(xproj[0], yproj[0]))/max(length, other_length)
            costs[i, j] = delta
            feasible[i, j] = (delta <= protocol.wall_threshold_m+1e-10 and angle <= protocol.wall_angle_degrees+1e-8
                              and overlap >= protocol.wall_overlap-1e-10)
    match = maximum_cardinality_matching(feasible, costs)
    return dict(counts(len(match), len(a), len(b)), matches=match)


def wall_distances(pred, gt, protocol):
    a = np.array([w['endpoints'] for w in pred['walls']]).reshape(-1, 2, 2)
    b = np.array([w['endpoints'] for w in gt['walls']]).reshape(-1, 2, 2)
    if not len(a) or not len(b):
        return {'distance_cm': 100*protocol.wall_distance_cap_m if len(a) or len(b) else None,
                'unclipped_cm': None, 'pred_to_gt_cm': None, 'gt_to_pred_cm': None,
                'missing_gt_wall_length_fraction': 1. if len(b) else None}
    def directional(source, target):
        points, weights = sampled_segments(source, protocol.wall_sample_m)
        distance = point_segment_distances(points, target)
        return (float(np.average(np.minimum(distance, protocol.wall_distance_cap_m), weights=weights)),
                float(np.average(distance, weights=weights)),
                float(weights[distance > protocol.wall_threshold_m].sum()/weights.sum()))
    ab, ba = directional(a, b), directional(b, a)
    return {'distance_cm': 50*(ab[0]+ba[0]), 'unclipped_cm': 50*(ab[1]+ba[1]),
            'pred_to_gt_cm': 100*ab[0], 'gt_to_pred_cm': 100*ba[0],
            'missing_gt_wall_length_fraction': ba[2]}


def statistics(values):
    values = np.asarray(values, float)
    return {'count': len(values), 'median': float(np.median(values)) if len(values) else None,
            'mean': float(np.mean(values)) if len(values) else None,
            'p90': float(np.percentile(values, 90)) if len(values) else None}


def evaluate_floor(prediction, ground_truth, protocol=None):
    protocol = protocol or FloorMetricProtocol()
    if protocol.scale_track not in ('S', 'M'): raise ValueError('unknown scale track')
    if prediction['floor_uid'] != ground_truth['floor_uid']: raise ValueError('floor identity mismatch')
    inputs = sorted(prediction['observation_order'])
    if len(inputs) != len(set(inputs)): raise ValueError('duplicate input observations')
    gt = ground_truth; metric = gt.get('metric_scale') is not None
    conversion = np.diag([gt['metric_scale'] if metric else 1.]*2+[1.])
    gt_poses = {uid: conversion @ similarity(row['pose']) for uid, row in gt['observations'].items()
                if row['pose_valid']}
    gt_rooms = {uid: transform_shape(polygon(value), conversion) for uid, value in gt['rooms'].items()}
    comps = sorted(prediction['components'], key=lambda c: (-len(c['observation_ids']), -c['edge_quality_sum'], c['anchor_id']))
    component = comps[0] if comps else None
    anchor = component['anchor_id'] if component else None
    members = set(component['observation_ids']) if component else set()
    observations = {o['observation_id']: o for o in prediction['observations']}
    if set(observations) != set(inputs): raise ValueError('all input observations must remain in output')
    valid_targets = sorted(set(inputs)&set(gt_poses)-({anchor} if anchor else set()))
    reasons, alignment = [], None
    if component and anchor in gt_poses:
        pa = similarity(observations[anchor]['T_local_to_component'])
        ga = gt_poses[anchor].copy()
        if protocol.scale_track == 'M':
            if not metric or prediction['coordinate_spec']['units'] != 'meters':
                raise ValueError('M track requires metric predictions and metric GT')
            ga[:2, :2] /= math.sqrt(np.linalg.det(ga[:2, :2]))
            if abs(parameters(pa)[3]) > 1e-6: raise ValueError('M anchor must have metric unit scale')
        alignment = ga @ np.linalg.inv(pa)
    elif component:
        reasons.append('prediction_anchor_has_no_gt_pose')
    if not metric: reasons.append('metric_scale_unavailable')
    if not gt['geometry_valid']: reasons.append('target_geometry_invalid_or_missing')
    moved_rooms, translations, rotations, normalized, scales = {}, [], [], [], []
    errors = []
    if alignment is not None:
        for room in prediction['rooms']:
            if room['component_id'] == component['component_id']:
                p = polygon(room.get('geometry'))
                if p is not None: moved_rooms[room['room_id']] = transform_shape(p, alignment)
        for uid in valid_targets:
            row = observations[uid]
            if uid not in members or row.get('T_local_to_component') is None:
                errors.append({'observation_id': uid, 'translation_cm': None, 'rotation_degrees': None,
                               'correct': False, 'reason': 'not_in_main_component'}); continue
            try: estimate = alignment @ similarity(row['T_local_to_component'])
            except ValueError:
                errors.append({'observation_id': uid, 'correct': False, 'reason': 'invalid_pose'}); continue
            truth = gt_poses[uid]
            distance = float(np.linalg.norm(estimate[:2, 2]-truth[:2, 2]))
            yaw = parameters(estimate)[2]-parameters(truth)[2]
            angle = abs(math.degrees(math.atan2(math.sin(yaw), math.cos(yaw))))
            scale_error = abs(math.exp(parameters(estimate)[3]-parameters(truth)[3])-1.) if protocol.scale_track == 'S' else 0.
            norm_error = distance/math.sqrt(np.linalg.det(gt_poses[anchor][:2, :2]))
            if metric: translations.append(100*distance)
            rotations.append(angle); normalized.append(norm_error); scales.append(scale_error)
            errors.append({'observation_id': uid, 'translation_cm': 100*distance if metric else None,
                'rotation_degrees': angle, 'translation_anchor_height_units': norm_error, 'relative_scale_error': scale_error,
                'correct': bool(metric and distance <= protocol.translation_threshold_m and angle <= protocol.rotation_threshold_degrees)})
    target_union = unary_union(list(gt_rooms.values())) if gt_rooms else GeometryCollection()
    pred_union = unary_union(list(moved_rooms.values())) if moved_rooms else GeometryCollection()
    alignment_available = alignment is not None or not component
    geometry_available = gt['geometry_valid'] and alignment_available
    iou = float(pred_union.intersection(target_union).area/pred_union.union(target_union).area) if geometry_available else None
    coverage = (sum(e.get('correct', False) for e in errors)/len(valid_targets)
                if metric and valid_targets and alignment_available and len(inputs) > 1 else None)
    wall = junction = distance = None
    if geometry_available and metric:
        pg = partition_graph(moved_rooms, protocol.graph_snap_m, protocol.collinear_degrees)
        gg = partition_graph(gt_rooms, protocol.graph_snap_m, protocol.collinear_degrees)
        wall = wall_metrics(pg, gg, protocol); junction = junction_metrics(pg, gg, protocol.junction_threshold_m)
        distance = wall_distances(pg, gg, protocol)
    room_result = None
    if geometry_available:
        a, b = list(moved_rooms.values()), list(gt_rooms.values())
        overlaps = np.array([[p.intersection(q).area/p.union(q).area for q in b] for p in a]).reshape(len(a), len(b))
        matches = maximum_cardinality_matching(overlaps >= protocol.room_iou_threshold, 1-overlaps)
        duplicate = sum(bool(np.any(row >= protocol.room_iou_threshold)) for row in overlaps)-len(matches)
        room_result = dict(counts(len(matches), len(a), len(b)), matched_ious=[float(overlaps[i, j]) for i, j in matches],
                           duplicate_rooms=duplicate, matching_threshold=protocol.room_iou_threshold)
    metrics = {'overall_iou': iou, 'localization_coverage': coverage,
        'translation_error_cm': statistics(translations)['median'], 'rotation_error_degrees': statistics(rotations)['median'],
        'wall_f1': wall['f1'] if wall else None, 'junction_f1': junction['f1'] if junction else None,
        'wall_distance_cm': distance['distance_cm'] if distance else None}
    return {'schema_version': 'floor-metrics.v2', 'floor_uid': gt['floor_uid'], 'house_id': gt['house_id'],
        'protocol': asdict(protocol), 'metrics': metrics, 'metric_scale_available': metric,
        'anchor_id': anchor, 'alignment': alignment.tolist() if alignment is not None else None,
        'assembly_coverage': len(members)/len(inputs) if inputs else 0.,
        'input_observations': len(inputs), 'evaluable_nonanchor_observations': len(valid_targets),
        'valid_localized_observations': len(rotations), 'evaluation_mask_coverage': len(set(inputs)&set(gt_poses))/len(inputs) if inputs else 0.,
        'translation_statistics_cm': statistics(translations), 'rotation_statistics_degrees': statistics(rotations),
        'normalized_translation_statistics': statistics(normalized), 'pose_errors': errors,
        'wall': wall, 'junction': junction, 'wall_distance': distance, 'room_recovery': room_result,
        'unavailable_reasons': reasons, 'runtime_failure': prediction.get('status') == 'runtime_failure',
        'prediction_status': prediction.get('status'), 'target_scope': 'all_annotated_complete_rooms',
        'counts': {'predicted_rooms_main': len(moved_rooms), 'target_rooms': len(gt_rooms), 'components': len(comps)}}


def aggregate_metrics(rows):
    if not rows: return {'floors': 0, 'metrics': {}}
    summary, evaluated = {}, {}
    for key in rows[0]['metrics']:
        values = [r['metrics'][key] for r in rows if r['metrics'][key] is not None]
        evaluated[key] = len(values)
        if key in ('translation_error_cm', 'rotation_error_degrees'):
            name = 'translation_cm' if key == 'translation_error_cm' else 'rotation_degrees'
            values = [e[name] for r in rows for e in r['pose_errors'] if e.get(name) is not None]
            summary[key] = float(np.median(values)) if values else None
        else: summary[key] = float(np.mean(values)) if values else None
    return {'floors': len(rows), 'houses': len({r['house_id'] for r in rows}), 'metrics': summary,
            'evaluable_floors': evaluated, 'runtime_failures': sum(r['runtime_failure'] for r in rows),
            'aggregation': 'floor macro except pooled observation median translation/rotation'}


def write_metrics(rows, output):
    from portallayout.utils.room_feature_cache import write_json_atomic
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    summary = aggregate_metrics(rows)
    write_json_atomic(output/'metrics.json', {'summary': summary, 'floors': rows})
    keys = list(rows[0]['metrics']) if rows else []
    with (output/'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, ['floor_uid', 'house_id']+keys); writer.writeheader()
        for row in rows: writer.writerow(dict(floor_uid=row['floor_uid'], house_id=row['house_id'], **row['metrics']))
    return summary
