"""Prediction-only local room-boundary refinement from accepted observations.

This experimental primitive does not extract image evidence or decide whether
an opening is visible.  Callers must provide those predictions explicitly.
Coordinates and the displacement thresholds use the target polygon's units.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import least_squares
from shapely.geometry import Polygon

from portallayout.geometry.floor_geometry import points_transform, similarity


@dataclass(frozen=True)
class RefinementConfig:
    min_confidence: float = .8
    max_association_distance: float = .5
    max_angle_degrees: float = 30.
    max_displacement: float = .25
    prior_weight: float = 1.
    length_weight: float = .25
    robust_scale: float = .1

    def __post_init__(self):
        vals = (self.min_confidence, self.max_association_distance,
                self.max_angle_degrees, self.max_displacement,
                self.prior_weight, self.length_weight, self.robust_scale)
        if (not np.isfinite(vals).all() or not 0 <= self.min_confidence <= 1
                or not 0 < self.max_angle_degrees < 90
                or min(self.max_association_distance, self.max_displacement,
                       self.prior_weight, self.robust_scale) <= 0
                or self.length_weight < 0):
            raise ValueError('invalid refinement configuration')


def _valid_polygon(points):
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        return False
    poly = Polygon(points)
    return bool(poly.is_valid and not poly.is_empty and poly.area > 1e-10
                and np.all(np.linalg.norm(np.roll(points, -1, axis=0)-points, axis=1) > 1e-8))


def _associate(points, vertices, config):
    """Use the observation's own line and finite wall segments for gating."""
    centered = points - points.mean(axis=0)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    if not len(singular) or singular[0] < 1e-6:
        return None, 'degenerate_boundary'
    direction = vh[0]
    best = None
    for i, start in enumerate(vertices):
        end = vertices[(i+1) % len(vertices)]
        wall = end-start
        length = float(np.linalg.norm(wall))
        unit = wall/length
        if abs(float(unit @ direction)) < math.cos(math.radians(config.max_angle_degrees)):
            continue
        fraction = np.clip((points-start) @ wall/(length*length), 0., 1.)
        distances = np.linalg.norm(points-(start+fraction[:, None]*wall), axis=1)
        if float(np.max(distances)) > config.max_association_distance:
            continue
        candidate = (float(np.mean(distances)), i)
        if best is None or candidate < best:
            best = candidate
    return (best[1], None) if best else (None, 'no_compatible_wall')


def refine_local_layout(polygon, observations, transforms, config=None):
    """Refine a single polygon using explicitly accepted source observations.

    ``observations`` are dicts containing source_observation_id,
    target_observation_id, source_opening_id, target_opening_id, frame
    (``source`` or ``target``), boundary_points (at least two 2D points),
    confidence, visibility (``visible``), status and pose_status (both
    ``accepted``). If frame
    is ``source``, transforms[source_observation_id] is T_source_to_target.
    Images, GT and annotation labels are not inputs to this geometry primitive.
    It preserves unsupported vertices, and returns the input if optimization
    fails or produces an invalid polygon or an overlarge displacement.
    """
    config = config or RefinementConfig()
    raw = np.asarray(polygon, dtype=float)
    closed = bool(raw.ndim == 2 and len(raw) > 1 and np.array_equal(raw[0], raw[-1]))
    initial = raw[:-1].copy() if closed else raw.copy()
    if not _valid_polygon(initial):
        raise ValueError('initial layout must be a finite simple polygon')
    original = raw.tolist()
    diagnostics = []
    accepted = []
    target_id = None
    for idx, item in enumerate(observations):
        reason = None
        try:
            if not all(item.get(k) for k in ('source_observation_id', 'target_observation_id',
                                              'source_opening_id', 'target_opening_id')):
                reason = 'missing_identity'
            elif (item.get('status') != 'accepted' or item.get('pose_status') != 'accepted'
                  or item.get('visibility') != 'visible'):
                reason = 'unverified_or_not_visible'
            elif target_id is not None and item['target_observation_id'] != target_id:
                reason = 'different_target_observation'
            else:
                confidence = float(item.get('confidence', float('nan')))
                if not math.isfinite(confidence) or not config.min_confidence <= confidence <= 1:
                    reason = 'invalid_or_low_confidence'
                else:
                    points = np.asarray(item.get('boundary_points'), dtype=float)
                    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
                        reason = 'invalid_boundary_points'
                    elif item.get('frame') == 'source':
                        transform = similarity(transforms[item['source_observation_id']])
                        points = points_transform(points, transform)
                    elif item.get('frame') != 'target':
                        reason = 'invalid_coordinate_frame'
                    if reason is None:
                        edge, reason = _associate(points, initial, config)
                        if reason is None:
                            target_id = item['target_observation_id']
                            accepted.append((edge, points, confidence))
        except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
            reason = 'invalid_observation_or_transform'
        diagnostics.append({'index': idx, 'accepted': reason is None, 'reason': reason})

    def unchanged(reason):
        return {'status': 'unchanged', 'reason': reason, 'polygon': original,
                'changed_vertex_indices': [], 'observations': diagnostics}

    if not accepted:
        return unchanged('no_usable_observation')
    indices = sorted({i for edge, _, _ in accepted for i in (edge, (edge+1) % len(initial))})
    initial_lengths = np.linalg.norm(np.roll(initial, -1, axis=0)-initial, axis=1)

    def residual(delta):
        vertices = initial.copy()
        vertices[indices] += delta.reshape(-1, 2)
        pieces = []
        for edge, points, confidence in accepted:
            start = vertices[edge]
            wall = vertices[(edge+1) % len(vertices)]-start
            length = max(float(np.linalg.norm(wall)), 1e-8)
            normal = np.array([-wall[1], wall[0]])/length
            pieces.extend((math.sqrt(confidence/len(points)) * ((points-start) @ normal)).tolist())
        pieces.extend((math.sqrt(config.prior_weight) * delta).tolist())
        if config.length_weight:
            lengths = np.linalg.norm(np.roll(vertices, -1, axis=0)-vertices, axis=1)
            pieces.extend((math.sqrt(config.length_weight)*(lengths-initial_lengths)).tolist())
        return np.asarray(pieces)

    try:
        solution = least_squares(residual, np.zeros(2*len(indices)),
                                 bounds=(-config.max_displacement, config.max_displacement),
                                 loss='soft_l1', f_scale=config.robust_scale, max_nfev=200)
        if not solution.success or not np.isfinite(solution.x).all():
            return unchanged('optimization_failed')
        delta = solution.x.reshape(-1, 2)
        if np.any(np.linalg.norm(delta, axis=1) > config.max_displacement+1e-9):
            return unchanged('displacement_exceeded')
        result = initial.copy()
        result[indices] += delta
        if not _valid_polygon(result):
            return unchanged('invalid_refined_polygon')
        changed = [int(i) for i, d in zip(indices, delta) if np.linalg.norm(d) > 1e-8]
        if not changed:
            return unchanged('no_material_change')
        returned = np.vstack((result, result[0])) if closed else result
        return {'status': 'refined', 'reason': None, 'polygon': returned.tolist(),
                'changed_vertex_indices': changed, 'observations': diagnostics,
                'initial_cost': float(np.sum(residual(np.zeros_like(solution.x))**2)),
                'final_cost': float(np.sum(residual(solution.x)**2)),
                'max_displacement': float(np.linalg.norm(delta, axis=1).max())}
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return unchanged('optimization_exception')
