"""Pose-conditioned auxiliary features. No dataset/GT access is permitted here."""
import math
import numpy as np
import torch
from PIL import Image, ImageDraw
from shapely.geometry import Point
from portallayout.models.floor_auxiliary import FEATURE_VERSION, GEOMETRY_NAMES
from portallayout.geometry.floor_geometry import (observation_polygon, local_scale, transform_shape,
    points_transform, parameters, segments, point_segment_distances, sampled_segments)


def _opening(obs, candidate_id):
    return next((o for o in obs.get('prediction', obs).get('openings', []) if o['candidate_id'] == candidate_id), None)


def _nonseed_segments(obs, seed):
    p = observation_polygon(obs)
    if p is None: return []
    ss = segments(p)
    if seed is None: return ss
    if 'support_provenance_valid' in seed and not seed['support_provenance_valid']: return []
    # Entire support wall is excluded, also collinear subdivisions of that wall.
    endpoints = np.asarray(seed.get('endpoints_local'), float)
    if endpoints.shape != (2, 2): return []
    tangent = endpoints[1]-endpoints[0]; length = np.linalg.norm(tangent)
    if length < 1e-8: return []
    result = []
    for i, line in enumerate(ss):
        distance_to_support = np.abs(np.cross(tangent, line-endpoints[0]))/length
        support = seed.get('support_wall_ids', [seed.get('wall_index')])
        if i in support or distance_to_support.max() < 1e-5: continue
        result.append(line)
    return result


def _raster(a, b, boundary_a, boundary_b, seed, scale, size=128):
    """Seven geometry channels; no fabricated portal when no seed exists."""
    center, rotation = np.zeros(2), np.eye(2)
    if seed is not None:
        points = np.asarray(seed['endpoints_local'], float)
        center = points.mean(axis=0); d = points[1]-points[0]
        scale = float(np.linalg.norm(d)); theta = -math.atan2(d[1], d[0])
        rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    scale = max(scale, 1e-4); extent = 6.
    def pixels(points):
        q = (np.asarray(points)-center) @ rotation.T/scale
        q = (q+extent)/(2*extent)*(size-1); q[:, 1] = size-1-q[:, 1]
        return [tuple(v) for v in np.rint(np.clip(q, -1e6, 1e6)).astype(int)]
    def fill(p):
        im = Image.new('L', (size, size)); draw = ImageDraw.Draw(im)
        draw.polygon(pixels(p.exterior.coords), fill=1)
        for interior in p.interiors: draw.polygon(pixels(interior.coords), fill=0)
        return np.array(im, dtype=np.float32)
    def lines(ss):
        im = Image.new('L', (size, size)); draw = ImageDraw.Draw(im)
        for line in ss: draw.line(pixels(line), fill=1, width=1)
        return np.array(im, dtype=np.float32)
    aa, bb = fill(a), fill(b)
    portal = lines([np.asarray(seed['endpoints_local'])]) if seed is not None else np.zeros_like(aa)
    return np.stack([aa, bb, lines(boundary_a), lines(boundary_b), np.minimum(aa, bb), portal, np.ones_like(aa)])


def _pooled(view):
    return view.memory.detach().float().mean(dim=(-2, -1)).cpu().numpy()


def _descriptor(view, candidate):
    if candidate in view.candidate_ids:
        return view.descriptors[view.candidate_ids.index(candidate)].detach().float().cpu().numpy()
    return np.zeros(256, np.float32)


def _directed(a, b, va, vb, transform, seed_a, seed_b, matches, score, margin, task, exclude_duplicates=False):
    pa, pb = observation_polygon(a), observation_polygon(b)
    values, masks = np.zeros(len(GEOMETRY_NAMES), np.float32), np.zeros(len(GEOMETRY_NAMES), np.float32)
    visual = np.r_[_pooled(va), _pooled(vb)]
    if task == 'pose': visual = np.r_[visual, _descriptor(va, seed_a), _descriptor(vb, seed_b)]
    if pa is None or pb is None:
        return np.zeros((7, 128, 128), np.float32), visual, np.r_[values, masks], False
    moved = transform_shape(pb, transform); scale = local_scale(a)
    sa, sb = _opening(a, seed_a), _opening(b, seed_b)
    # Pose seeds must exist with valid endpoints; the room head has no portal seed.
    valid_statuses = ('mapped_dominant_wall', 'predicted_geometry_overlay')
    if task == 'pose' and (sa is None or sb is None or sa.get('geometry_status') not in valid_statuses or
                           sb.get('geometry_status') not in valid_statuses):
        return np.zeros((7, 128, 128), np.float32), visual, np.r_[values, masks], False
    na = _nonseed_segments(a, sa if task == 'pose' else None)
    nb = [points_transform(line, transform) for line in _nonseed_segments(b, sb if task == 'pose' else None)]
    def put(name, value):
        if value is not None and np.isfinite(value):
            i = GEOMETRY_NAMES.index(name); values[i] = np.clip(value, -20, 20); masks[i] = 1.
    intersection = pa.intersection(moved).area
    put('iou', intersection/pa.union(moved).area); put('overlap_min', intersection/min(pa.area, moved.area))
    put('log_area_ratio', math.log(pa.area/moved.area))
    put('camera_a_inside_b', float(moved.covers(Point(0, 0))))
    put('camera_b_inside_a', float(pa.covers(Point(transform[:2, 2]))))
    def boundary_stats(x, y):
        if not x or not y: return None, None
        samples, weights = sampled_segments(x, max(scale/8., .01))
        distance = point_segment_distances(samples, y)/scale
        return float(np.average(distance, weights=weights)), float(np.percentile(distance, 90))
    # Exclude the fitted support walls from both the raster boundary channels
    # and residual features. Their zero fitting error cannot certify a pose.
    mean, p90 = boundary_stats(na, nb)
    put('boundary_mean', mean); put('boundary_p90', p90)
    put('nonseed_boundary_mean', mean); put('nonseed_boundary_p90', p90)
    other = []; used = []
    for pair in matches:
        ca, cb = pair['candidate_id_a'], pair['candidate_id_b']
        if ca == seed_a or cb == seed_b: continue
        oa, ob = _opening(a, ca), _opening(b, cb)
        if oa is None or ob is None or oa.get('endpoints_local') is None or ob.get('endpoints_local') is None: continue
        x, raw_y = np.asarray(oa['endpoints_local']), np.asarray(ob['endpoints_local'])
        y = points_transform(raw_y, transform)
        if sa is not None:
            def shared_support(x, y):
                if 'support_provenance_valid' in x or 'support_provenance_valid' in y:
                    if not x.get('support_provenance_valid') or not y.get('support_provenance_valid'): return True
                    return bool(set(x['support_wall_ids']) & set(y['support_wall_ids']))
                return x.get('wall_index') == y.get('wall_index')
            if shared_support(oa, sa) or shared_support(ob, sb): continue
        if exclude_duplicates:
            def same(u, v):
                return np.allclose(u, v, atol=1e-7, rtol=0) or np.allclose(u, v[::-1], atol=1e-7, rtol=0)
            if (same(x, np.asarray(sa['endpoints_local'])) or same(raw_y, np.asarray(sb['endpoints_local'])) or
                    any(same(x, u) or same(raw_y, v) for u, v in used)):
                continue
            used.append((x, raw_y))
        other.append(min(np.linalg.norm(x-y, axis=1).max(), np.linalg.norm(x-y[::-1], axis=1).max())/scale)
    if other:
        put('other_opening_mean', np.mean(other)); put('other_opening_max', max(other))
    put('match_score', score); put('dustbin_margin', margin)
    put('absolute_log_scale', abs(parameters(transform)[3]))
    put('independent_wall_fraction', min(len(na)/max(len(segments(pa)), 1), len(nb)/max(len(segments(pb)), 1)))
    put('geometry_valid', 1.)
    independent = bool(other) or (len(na) >= 2 and len(nb) >= 2)
    raster = _raster(pa, moved, na, nb, sa if task == 'pose' else None, scale)
    return raster, visual.astype(np.float32), np.r_[values, masks], independent


def auxiliary_features(a, b, va, vb, transform, *, seed_ids=(None, None), matches=(), score=0., margin=0., task='pose', exclude_duplicates=False):
    if task not in ('pose', 'room'): raise ValueError('unknown auxiliary task')
    transform = np.asarray(transform, float)
    reverse_matches = [dict(candidate_id_a=x['candidate_id_b'], candidate_id_b=x['candidate_id_a']) for x in matches]
    forward = _directed(a, b, va, vb, transform, seed_ids[0], seed_ids[1], matches, score, margin, task, exclude_duplicates)
    reverse = _directed(b, a, vb, va, np.linalg.inv(transform), seed_ids[1], seed_ids[0], reverse_matches, score, margin, task, exclude_duplicates)
    return {'raster': np.stack([forward[0], reverse[0]]), 'visual': np.stack([forward[1], reverse[1]]),
            'geometry': np.stack([forward[2], reverse[2]]), 'independent_evidence': bool(forward[3] and reverse[3]),
            'feature_version': FEATURE_VERSION}


def feature_tensors(features, device='cpu'):
    return [torch.as_tensor(np.stack([x[key] for x in features]), dtype=torch.float32, device=device)
            for key in ('raster', 'visual', 'geometry')]
