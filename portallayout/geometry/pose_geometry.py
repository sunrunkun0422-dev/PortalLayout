"""Prediction-only geometry overlays; never re-encode the frozen matcher views."""
import copy
import math
import numpy as np

from portallayout.geometry.candidate_local_portal_geometry import estimate_candidate_local_portal_geometry, panorama_ray
from portallayout.engine.cross_scene_estimator import wall_token_assignment
from portallayout.geometry.joint_layout import generate_opening_pose_hypotheses
from portallayout.utils.room_feature_cache import fingerprint

VERSION = 'pose-geometry-overlay.v1'
FEATURE_VERSION = 'floor-auxiliary.pose-geometry.v4'
METHODS = ('dominant_wall', 'raw_depth', 'local_m2')


def geometry_config(config=None):
    config = config or {}
    result = dict(pose_geometry=config.get('pose_geometry', 'dominant_wall'),
                  scale_shrinkage=float(config.get('scale_shrinkage', 1.)))
    if result['pose_geometry'] not in METHODS or result['scale_shrinkage'] not in (1., .5, 0.):
        raise ValueError('unsupported pose geometry configuration')
    return result


def variant_id(config):
    c = geometry_config(config)
    return c['pose_geometry'] + '_s' + str(c['scale_shrinkage']).replace('.', 'p')


def opening_overlay(record, arrays, method):
    """Return a copy, retaining local polygons and stable candidate identities."""
    if method not in METHODS: raise ValueError('unknown geometry method')
    result = copy.deepcopy(record)
    depth = np.maximum(np.abs(np.asarray(arrays['depth_enclosed']).reshape(-1)), .001)
    probability = np.asarray(arrays['opening_probability']).reshape(-1)
    n = len(depth)
    if len(probability) != n: raise ValueError('depth/probability token mismatch')
    contour = np.stack([panorama_ray(i, n) for i in range(n)]) * depth[:, None]
    polygon = record['prediction'].get('geometry', {}).get('enclosed', {}).get('polygon_local') or []
    wall_ids = None
    try:
        layout = {'layoutPoints': {'points': [{'xyz': [x, 1., z]} for x, z in polygon]}}
        wall_ids, _ = wall_token_assignment(layout, sample_count=n)
    except (ValueError, IndexError, RuntimeError, ZeroDivisionError): pass
    for opening in result['prediction']['openings']:
        start, end = opening['token_interval_inclusive']
        support = [start, end]; endpoints = None; failure = None
        if method == 'dominant_wall':
            endpoints = opening.get('endpoints_local')
            if opening.get('geometry_status') != 'mapped_dominant_wall': endpoints = None
            support = ((np.arange((end-start) % n+1)+start) % n).tolist()
        elif method == 'raw_depth': endpoints = contour[[start, end]].tolist()
        else:
            fit = estimate_candidate_local_portal_geometry(contour, [start, end], n, probability)
            if fit.primary is not None:
                support = list(fit.primary.support_indices)
                if fit.geometry_valid: endpoints = fit.primary.endpoints_local_bi_xz
            opening['local_fit'] = fit.to_json()
            failure = fit.failure_reason
        e = np.asarray(endpoints, dtype=float)
        valid = e.shape == (2, 2) and np.isfinite(e).all() and np.linalg.norm(e[1]-e[0]) > 1e-8
        walls = sorted(set(int(wall_ids[t]) for t in support)) if wall_ids is not None else []
        if method == 'dominant_wall' and opening.get('wall_index') is not None:
            walls = [int(opening['wall_index'])]
        opening.update(geometry_source=method, geometry_valid=bool(valid),
            geometry_status='predicted_geometry_overlay' if valid else 'unavailable',
            endpoints_local=e.tolist() if valid else None, support_tokens=support,
            support_wall_ids=walls, support_provenance_valid=wall_ids is not None,
            geometry_failure=None if valid else failure or 'invalid_or_degenerate_endpoints')
    result['pose_geometry_overlay'] = dict(version=VERSION, method=method,
        source_cache_key=record.get('cache_key'), source_prediction_fingerprint=fingerprint(record['prediction']))
    return result


def pose_hypotheses(a, b, alpha=1.):
    """Filter the original width ratio before shrinkage; retain both mappings."""
    if alpha not in (1., .5, 0.): raise ValueError('unsupported scale shrinkage')
    a, b = np.asarray(a, float), np.asarray(b, float)
    proposal = generate_opening_pose_hypotheses(a, b)
    out = []
    for h in proposal['hypotheses']:
        raw_scale = h['scale_B_to_A']
        if not np.isfinite(raw_scale) or not .5 <= raw_scale <= 2.: continue
        t = np.asarray(h['T_B_to_A'], float)
        rotation = t[:2, :2]/raw_scale
        scale = math.exp(float(alpha)*math.log(raw_scale))
        t[:2, :2] = scale*rotation
        t[:2, 2] = a.mean(0)-t[:2, :2]@b.mean(0)
        out.append(dict(h, T_B_to_A=t.tolist(), scale_B_to_A=scale,
                        raw_scale_B_to_A=raw_scale, scale_shrinkage=float(alpha)))
    return out


def regenerate_pair(row, a, b, config):
    """Keep scores, dustbin decisions and all matches exactly as cached."""
    config = geometry_config(config)
    result = copy.deepcopy(row)
    result.update(pose_hypotheses=[], pose_geometry=config, pose_geometry_version=VERSION,
                  T_B_to_A=None, pose_status='unverified', calibrated=False,
                  assembly_status='not_used', pose_quality=None, geometry_failures=[])
    lookup_a = {o['candidate_id']: o for o in a['prediction'].get('openings', [])}
    lookup_b = {o['candidate_id']: o for o in b['prediction'].get('openings', [])}
    for m in row['matches']:
        ids = [m['candidate_id_a'], m['candidate_id_b']]
        sa, sb = lookup_a.get(ids[0]), lookup_b.get(ids[1])
        try:
            if not sa or not sb or not sa.get('geometry_valid') or not sb.get('geometry_valid'):
                raise ValueError('unavailable_endpoint_geometry')
            hs = pose_hypotheses(sa['endpoints_local'], sb['endpoints_local'], config['scale_shrinkage'])
            if not hs: raise ValueError('raw_scale_outside_range')
            for h in hs:
                result['pose_hypotheses'].append(dict(h,
                    hypothesis_id=fingerprint([row['pair_id'], ids, h['endpoint_mapping'], config]),
                    seed_candidate_ids=ids, match_confidence=m['confidence'], usable=True,
                    status='unverified', independent_evidence=[]))
        except (ValueError, TypeError) as error:
            result['geometry_failures'].append(dict(candidate_ids=ids, reason=str(error)))
    return result


def overlay_floor(observations, cache, method):
    result = []
    for obs in observations:
        if obs['observation_id'] in cache.records:
            record, arrays = cache.load(obs['observation_id'])
            overlay = opening_overlay(record, arrays, method)
            result.append(dict(obs, prediction=overlay['prediction'], pose_geometry_overlay=overlay['pose_geometry_overlay']))
        else: result.append(copy.deepcopy(obs))
    return result
