"""Prediction-only, explicitly tentative geometry baseline for whole floors."""
import copy
import itertools
import math
import numpy as np
import torch
from shapely.geometry import Point
from portallayout.models.opening_pair_matcher import mutual_matches
from portallayout.geometry.floor_auxiliary_features import _nonseed_segments
from portallayout.geometry.floor_geometry import (observation_polygon, local_scale, points_transform,
    transform_shape, sampled_segments, point_segment_distances, parameters, segments)
from portallayout.geometry.floor_pose_graph import forest, path_transform, edge_residual
from portallayout.geometry.joint_layout import generate_opening_pose_hypotheses
from portallayout.utils.room_feature_cache import fingerprint


VERSION = 'exploratory-opening-geometry.v1'


def _boundary_mean(source, target, scale):
    if not source or not target:
        return None
    points, weights = sampled_segments(source, max(scale/8., .01))
    return float(np.average(point_segment_distances(points, target), weights=weights)/scale)


def _directed_cost(a, b, transform, sa, sb):
    pa, pb = observation_polygon(a), observation_polygon(b)
    if pa is None or pb is None:
        return None
    moved = transform_shape(pb, transform)
    intersection = pa.intersection(moved).area
    iou = intersection/pa.union(moved).area
    overlap = intersection/min(pa.area, moved.area)
    inside = (int(moved.covers(Point(0., 0.))) + int(pa.covers(Point(transform[:2, 2]))))/2.
    na, nb = _nonseed_segments(a, sa), _nonseed_segments(b, sb)
    boundary = _boundary_mean(na, [points_transform(s, transform) for s in nb], local_scale(a))
    same = None if boundary is None else 1.-iou+min(boundary, 1.)+1.-inside
    p, q = np.asarray(sa['endpoints_local'], float)
    cross_a = np.cross(q-p, -p); cross_b = np.cross(q-p, transform[:2, 2]-p)
    adjacent = overlap+inside+float(cross_a*cross_b > 1e-10)
    return dict(same_space_cost=same, adjacent_space_cost=float(adjacent),
                nonseed_boundary_distance=boundary, iou=float(iou), overlap_min=float(overlap))


def geometry_cost(a, b, transform, sa, sb):
    directions = [_directed_cost(a, b, transform, sa, sb),
                  _directed_cost(b, a, np.linalg.inv(transform), sb, sa)]
    if any(d is None for d in directions):
        return {'cost': None, 'valid': False, 'reason': 'invalid_local_region'}
    same = (float(np.mean([d['same_space_cost'] for d in directions]))
            if all(d['same_space_cost'] is not None for d in directions) else None)
    adjacent = float(np.mean([d['adjacent_space_cost'] for d in directions]))
    return dict(cost=min(same, adjacent) if same is not None else adjacent, valid=True,
                same_space_cost=same, adjacent_space_cost=adjacent, directions=directions,
                evidence_semantics='heuristic_not_pose_verification')


def _same_segment(a, b):
    return np.allclose(a, b, atol=1e-7, rtol=0) or np.allclose(a, b[::-1], atol=1e-7, rtol=0)


def _heldout_evidence(mapped, seed, transform):
    result, used = [], []
    sa, sb, _ = seed
    for a, b, confidence in mapped:
        if (a['candidate_id'] == sa['candidate_id'] or b['candidate_id'] == sb['candidate_id'] or
                a.get('wall_index') == sa.get('wall_index') or b.get('wall_index') == sb.get('wall_index')):
            continue
        x, y = np.asarray(a['endpoints_local']), np.asarray(b['endpoints_local'])
        if (_same_segment(x, np.asarray(sa['endpoints_local'])) or
                _same_segment(y, np.asarray(sb['endpoints_local'])) or
                any(_same_segment(x, u) or _same_segment(y, v) for u, v in used)):
            continue
        used.append((x, y))
        def error(target, source, matrix):
            moved = points_transform(source, matrix)
            return min(np.linalg.norm(target-moved, axis=1).max(),
                       np.linalg.norm(target-moved[::-1], axis=1).max())/np.linalg.norm(target[1]-target[0])
        residual = max(error(x, y, transform), error(y, x, np.linalg.inv(transform)))
        result.append(dict(candidate_ids=[a['candidate_id'], b['candidate_id']],
                           confidence=confidence, relative_endpoint_error=float(residual), supports=bool(residual <= .15)))
    return result


def predict_exploratory_pair(a, b, va, vb, bundle):
    """Raw zero-threshold mutual retrieval. No supervision or auxiliary head."""
    aid, bid = a['observation_id'], b['observation_id']
    row = dict(pair_id=fingerprint([aid, bid]), observation_a=aid, observation_b=bid,
               match_status='diagnostic_only', numerical_status='missing_prediction', pose_status='unverified',
               calibrated=False, matches=[], pose_hypotheses=[], T_B_to_A=None, pose_quality=None,
               assembly_status='not_used', reason_codes=[], confidence_source=VERSION)
    if va is None or vb is None:
        row['reason_codes'].append('prediction_unavailable'); return row
    with torch.no_grad(): output = bundle.matcher(va, vb)
    row.update(numerical_status=output['status'], candidate_ids_a=list(va.candidate_ids),
               candidate_ids_b=list(vb.candidate_ids), assignment_method=output.get('assignment_method', 'log_sinkhorn'),
               sinkhorn={k: output[k] for k in ('iterations', 'marginal_error', 'converged')})
    for key in ('scores', 'conditional_ab', 'conditional_ba'):
        value = output.get(key); row[key] = value.detach().cpu().tolist() if torch.is_tensor(value) else value
    if output['status'] != 'assigned':
        row['reason_codes'].append(output['status']); return row
    mapped = []
    for m in mutual_matches(output, 0.):
        oa, ob = a['prediction']['openings'][m['index_a']], b['prediction']['openings'][m['index_b']]
        m = dict(m, candidate_id_a=oa['candidate_id'], candidate_id_b=ob['candidate_id'])
        row['matches'].append(m)
        if oa.get('geometry_status') == ob.get('geometry_status') == 'mapped_dominant_wall':
            try:
                generate_opening_pose_hypotheses(oa['endpoints_local'], ob['endpoints_local'])
                mapped.append((oa, ob, float(m['confidence'])))
            except (ValueError, KeyError):
                row['reason_codes'].append('degenerate_opening_endpoints')
    for sa, sb, confidence in mapped:
        proposal = generate_opening_pose_hypotheses(sa['endpoints_local'], sb['endpoints_local'])
        for h in proposal['hypotheses']:
            h = dict(h, hypothesis_id=fingerprint([row['pair_id'], sa['candidate_id'], sb['candidate_id'], h['endpoint_mapping']]),
                     seed_candidate_ids=[sa['candidate_id'], sb['candidate_id']], match_confidence=confidence,
                     usable=False, status='unverified', independent_evidence=[])
            scale = h['scale_B_to_A']
            if not np.isfinite(scale) or not .5 <= scale <= 2.:
                h['reason'] = 'scale_outside_exploratory_range'
            else:
                transform = np.asarray(h['T_B_to_A'])
                h['geometry'] = geometry_cost(a, b, transform, sa, sb)
                h['usable'] = h['geometry']['valid']
                h['independent_evidence'] = _heldout_evidence(mapped, (sa, sb, confidence), transform)
            row['pose_hypotheses'].append(h)
    return row


def _alternatives(row, threshold):
    values = []
    for h in row['pose_hypotheses']:
        if not h.get('usable') or h['match_confidence'] < threshold:
            continue
        h['independent_support_count'] = sum(e['supports'] and e['confidence'] >= threshold for e in h['independent_evidence'])
        values.append(h)
    values.sort(key=lambda h: (-h['independent_support_count'], h['geometry']['cost'],
                               -h['match_confidence'], h['hypothesis_id']))
    unique = []
    for h in values:
        if not any(np.allclose(h['T_B_to_A'], v['T_B_to_A'], atol=1e-5, rtol=1e-5) for v in unique): unique.append(h)
    return unique


def _choose(row, h, reason):
    row.update(pose_status='tentative', calibrated=False, selected_hypothesis_id=h['hypothesis_id'],
               T_B_to_A=h['T_B_to_A'], pose_quality=max(1e-6, h['match_confidence']/(1.+h['geometry']['cost'])),
               selection_reason=reason, assembly_status='tentative_eligible')


def select_exploratory_poses(raw_pairs, observations, threshold, graph_config):
    from portallayout.geometry.floor_assembly import pair_edges
    if not math.isfinite(threshold) or not 0 <= threshold <= 1: raise ValueError('invalid exploratory threshold')
    pairs = copy.deepcopy(raw_pairs); nodes = sorted(o['observation_id'] for o in observations)
    scales = {o['observation_id']: local_scale(o) for o in observations}
    for row in pairs:
        row['matches'] = [m for m in row['matches'] if m['confidence'] >= threshold]
        row['match_status'] = ('tentative' if row['matches'] else 'tentative_no_match') if row['numerical_status'] == 'assigned' else row['numerical_status']
        candidates = _alternatives(row, threshold)
        if not candidates:
            row['reason_codes'].append('no_usable_exploratory_hypothesis'); continue
        first = candidates[0]
        row['pose_ambiguous'] = (first['independent_support_count'] == 0 or (len(candidates) > 1 and
            candidates[1]['independent_support_count'] == first['independent_support_count']))
        _choose(row, first, 'heldout_openings_then_predicted_geometry')
    for iteration in range(3):
        changed = False
        for row in sorted(pairs, key=lambda r: r['pair_id']):
            if row['pose_status'] != 'tentative': continue
            edges = [e for e in pair_edges(pairs, 'exploratory') if e['edge_id'] != row['pair_id']]
            tree = forest(nodes, edges)
            transform, path = path_transform(tree, row['observation_a'], row['observation_b'])
            if transform is None or not path: continue
            weights = {e['edge_id']: e['weight'] for e in edges}
            if min(weights[x] for x in path)+1e-12 < row['pose_quality']: continue
            compatible = []
            for h in _alternatives(row, threshold):
                e = dict(a=row['observation_a'], b=row['observation_b'], T_B_to_A=h['T_B_to_A'])
                residual = float(np.linalg.norm(edge_residual(e, {e['a']: np.eye(3), e['b']: transform}, scales, graph_config)))
                if residual <= graph_config.conflict_sigma: compatible.append(h)
            if len(compatible) == 1:
                h = compatible[0]; changed |= row['selected_hypothesis_id'] != h['hypothesis_id']
                _choose(row, h, 'tentative_independent_path_consistency')
                row.update(pose_ambiguous=False, independent_path_edge_ids=path, hypothesis_adjustment_round=iteration+1)
        if not changed: break
    return pairs


def geometric_room_relationships(observations, graph):
    lookup = {o['observation_id']: o for o in observations}; result = []
    for component in graph['components']:
        for aid, bid in itertools.combinations(sorted(component['observation_ids']), 2):
            a, b = lookup[aid], lookup[bid]; pa, pb = observation_polygon(a), observation_polygon(b)
            row = dict(a=aid, b=bid, valid=False, rule_passed=False, source='geometry_complete_link', calibrated=False)
            if pa is not None and pb is not None:
                transform = np.linalg.inv(np.asarray(component['poses'][aid])) @ np.asarray(component['poses'][bid])
                moved, inverse = transform_shape(pb, transform), np.linalg.inv(transform)
                iou = pa.intersection(moved).area/pa.union(moved).area
                da = _boundary_mean(segments(pa), [points_transform(s, transform) for s in segments(pb)], local_scale(a))
                db = _boundary_mean(segments(pb), [points_transform(s, inverse) for s in segments(pa)], local_scale(b))
                inside = (pa.buffer(.05*local_scale(a)).covers(Point(transform[:2, 2])) and
                          pb.buffer(.05*local_scale(b)).covers(Point(inverse[:2, 2])))
                row.update(valid=True, rule_passed=bool(iou >= .6 and inside and da <= .15 and db <= .15),
                           iou=float(iou), boundary_a_to_b=da, boundary_b_to_a=db, cameras_mutually_inside=bool(inside))
            result.append(row)
    return result
