"""Project explicitly supplied ERP floor-boundary candidates to local geometry.

This module performs only ray/ground-plane geometry. It does not extract image
features, read labels, decide visibility or accept matching/pose hypotheses.
Its output is in the source panorama's Bi-Layout local XZ frame, so any
cross-view transformation and all verification remain the caller's duties.
"""
import math

import numpy as np


def project_floor_boundary(uv_points, camera_height, *, valid, confidence,
                           height_units, horizon_epsilon=1e-6):
    """Project normalized ERP points onto the plane below the source camera.

    Args:
        uv_points: ordered ``(N, 2)`` normalized image coordinates. ``u`` is
            horizontal in [0, 1], with 0 and 1 the same panorama seam. ``v``
            is vertical in [0, 1], increasing downwards; the horizon is .5.
            These are continuous image coordinates, with no pixel-center offset.
        camera_height: positive distance from camera to the common ground plane,
            or None if unknown. This assumes a gravity-aligned panorama.
        valid: explicit boolean mask supplied by the caller, shape ``(N,)``.
        confidence: explicit per-point confidence supplied by the caller,
            shape ``(N,)``, each between 0 and 1. The values are passed through;
            they are not calibrated or inferred here.
        height_units: ``'meters'`` for a metric height, or
            ``'camera_height_normalized'`` with height exactly 1. Never infer
            metric units from an unknown or normalized camera height.
        horizon_epsilon: reject downward ray vertical components <= this value;
            this prevents a finite-looking division by a near-horizontal ray.

    Returns:
        A JSON-friendly dict. ``boundary_points`` holds only projectable 2D
        ``[x, z]`` points in input order, ``source_indices`` maps them back to
        the input, and ``point_confidences`` preserves the corresponding scores.
        ``point_reasons`` has one entry per input point; None means projected.
        ``valid_runs`` stores indices into ``boundary_points`` for contiguous
        stretches of projectable input; callers must not bridge invalid points.
        No visibility, correspondence, pose, or acceptance state is produced.

    Raises:
        ValueError for a malformed array, unsupported units, or invalid epsilon.
        Missing or nonpositive/nonfinite camera heights are reported in the
        result; they do not trigger an implicit scale or identity fallback.
    """
    try:
        uv = np.asarray(uv_points, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('uv_points must be an N by 2 numeric array') from exc
    if uv.size == 0 and uv.ndim == 1:
        uv = np.empty((0, 2), dtype=float)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError('uv_points must be an N by 2 numeric array')
    count = len(uv)
    mask = np.asarray(valid)
    if count == 0 and mask.shape == (0,):
        mask = mask.astype(bool)
    if mask.shape != (count,) or mask.dtype.kind != 'b':
        raise ValueError('valid must be an explicit boolean N-vector')
    try:
        scores = np.asarray(confidence, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('confidence must be a numeric N-vector') from exc
    if scores.shape != (count,):
        raise ValueError('confidence must be a numeric N-vector')
    if height_units not in ('meters', 'camera_height_normalized'):
        raise ValueError('height_units must be meters or camera_height_normalized')
    try:
        epsilon = float(horizon_epsilon)
    except (TypeError, ValueError) as exc:
        raise ValueError('horizon_epsilon must be finite and between 0 and 1') from exc
    if not math.isfinite(epsilon) or not 0 < epsilon < 1:
        raise ValueError('horizon_epsilon must be finite and between 0 and 1')

    height_ok = False
    height = None
    if camera_height is not None and not isinstance(camera_height, (bool, np.bool_)):
        try:
            height = float(camera_height)
            height_ok = math.isfinite(height) and height > 0
        except (TypeError, ValueError):
            pass
    # In a height-normalized frame the camera-to-floor distance is one.
    # Prevent accidentally treating a metric height as normalized scale.
    if height_ok and height_units == 'camera_height_normalized' and not math.isclose(height, 1., rel_tol=0., abs_tol=1e-12):
        raise ValueError('camera_height_normalized requires camera_height=1')

    points, indices, point_scores, reasons = [], [], [], []
    for i, ((u, v), allowed, score) in enumerate(zip(uv, mask, scores)):
        reason = None
        if not allowed:
            reason = 'caller_invalid'
        elif not np.isfinite((u, v)).all():
            reason = 'nonfinite_uv'
        elif not (0. <= u <= 1. and 0. <= v <= 1.):
            reason = 'uv_out_of_range'
        elif not math.isfinite(score) or not 0. <= score <= 1.:
            reason = 'invalid_confidence'
        elif not height_ok:
            reason = 'camera_height_unavailable'
        else:
            lon = ((u % 1.) - .5) * (2. * math.pi)
            lat = (v - .5) * math.pi
            vertical = math.sin(lat)
            if vertical <= epsilon:
                reason = 'at_or_above_horizon'
            else:
                # The Bi-Layout spherical ray is (cos(lat) sin(lon),
                # sin(lat), cos(lat) cos(lon)), with downwards positive Y.
                radius = height * (math.cos(lat) / vertical)
                x, z = radius * math.sin(lon), radius * math.cos(lon)
                if not math.isfinite(x) or not math.isfinite(z):
                    reason = 'nonfinite_projection'
                else:
                    points.append([x, z])
                    indices.append(i)
                    point_scores.append(float(score))
        reasons.append(reason)

    runs = []
    for output_index, source_index in enumerate(indices):
        if output_index == 0 or source_index != indices[output_index-1] + 1:
            runs.append([])
        runs[-1].append(output_index)
    projection_status = ('no_valid_points' if not points else
                         'projected' if len(points) == count else 'partially_projected')
    return {'projection_status': projection_status, 'boundary_points': points,
            'source_indices': indices, 'point_confidences': point_scores,
            'point_reasons': reasons, 'valid_runs': runs,
            'coordinate_spec': {'frame': 'bi_panorama_local_xz', 'units': height_units,
                                'ground_plane': 'y=+camera_height',
                                'vertical_axis': 'down', 'input_uv': 'normalized_erp'}}
