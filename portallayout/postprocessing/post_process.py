""" 
@Date: 2021/10/08
@description:
"""
import numpy as np
import cv2

from portallayout.postprocessing.dula.layout import fit_layout
from portallayout.postprocessing.dula.layout_old import fit_layout_old
from portallayout.geometry.conversion import depth2xyz, xyz2depth


def _rotation_xz(angle):
    cosine = np.cos(float(angle))
    sine = np.sin(float(angle))
    return np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)


def rotate_floor_xz(points, angle):
    """Rotate row-major ``[x, z]`` points using a CCW column-vector matrix."""
    value = np.asarray(points, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError("floor_xz must have shape [N, 2]")
    if not np.isfinite(value).all():
        raise ValueError("floor_xz must contain finite values")
    return value @ _rotation_xz(angle).T


def _manhattan_axis_residual(angles, heading):
    """Return unoriented residuals to a Manhattan family in ``[0, pi/4]``."""
    return np.abs((angles - float(heading) + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0)


def estimate_manhattan_heading(floor_xz):
    """Estimate a panorama-local Manhattan wall heading from predicted geometry.

    The fourth circular moment makes directions separated by 90 degrees
    equivalent. Edge lengths supply geometric support, while percentile
    clipping prevents a single corner/occlusion jump from dominating. No GT or
    image annotation is consumed.
    """
    points = np.asarray(floor_xz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4:
        raise ValueError("floor_xz must have shape [N, 2] with N >= 4")
    if not np.isfinite(points).all():
        raise ValueError("floor_xz must contain finite values")
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    positive = lengths > max(1e-10, float(np.max(lengths)) * 1e-8)
    if int(np.count_nonzero(positive)) < 2:
        raise ValueError("pre-Manhattan geometry has too few non-degenerate edges")
    edges = edges[positive]
    lengths = lengths[positive]
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    upper = float(np.quantile(lengths, 0.95))
    weights = np.minimum(lengths, max(upper, 1e-10))

    moment = np.sum(weights * np.exp(4j * angles))
    total_weight = float(np.sum(weights))
    if total_weight <= 0 or abs(moment) <= total_weight * 1e-8:
        raise ValueError("predicted geometry has no stable Manhattan orientation")
    heading = float(np.angle(moment) / 4.0)

    # One robust refinement suppresses edge directions farther than 22.5
    # degrees from the initial Manhattan family. Clean Manhattan polygons keep
    # every wall because all four directed axes are equivalent under 4*theta.
    residual = _manhattan_axis_residual(angles, heading)
    robust_weights = weights * np.clip(
        1.0 - residual / (np.pi / 8.0), 0.0, 1.0
    )
    if float(np.sum(robust_weights)) > total_weight * 0.1:
        refined = np.sum(robust_weights * np.exp(4j * angles))
        if abs(refined) > 0:
            heading = float(np.angle(refined) / 4.0)
            weights_for_confidence = robust_weights
            moment_for_confidence = refined
        else:  # pragma: no cover - guarded by the nonzero robust weight
            weights_for_confidence = weights
            moment_for_confidence = moment
    else:
        weights_for_confidence = weights
        moment_for_confidence = moment

    heading = float((heading + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0)
    final_residual = _manhattan_axis_residual(angles, heading)
    confidence = float(
        abs(moment_for_confidence) / max(float(np.sum(weights_for_confidence)), 1e-12)
    )
    return heading, {
        "headingSource": "predicted_pre_manhattan_edges_mod90_length_weighted",
        "edgeCount": int(len(edges)),
        "confidence": confidence,
        "meanAxisResidualDegrees": float(
            np.degrees(np.average(final_residual, weights=weights))
        ),
        "maxAxisResidualDegrees": float(np.degrees(np.max(final_residual))),
        "upperEdgeWeightClip": upper,
    }


def frame_preserving_manhattan(b_depth, need_cube=False):
    """Fit Manhattan polygons while preserving each predicted panorama heading.

    Returns a list of fitted ``[x, y, z]`` polygons in panorama-local geometry
    and a parallel list of explicit frame reports. ``C`` maps local to
    canonical; the final polygon is transformed by ``C^-1`` before serialization
    or interval-to-wall mapping.
    """
    depth = np.asarray(b_depth, dtype=np.float64)
    if depth.ndim == 1:
        depth = depth[None, :]
    if depth.ndim != 2 or depth.shape[1] < 4:
        raise ValueError("b_depth must have shape [B, N] with N >= 4")
    if not np.isfinite(depth).all():
        raise ValueError("b_depth must contain finite values")
    xyz_batch = depth2xyz(depth, plan_y=1)
    outputs = []
    reports = []
    for xyz in xyz_batch:
        pre_local = np.asarray(xyz[..., ::2], dtype=np.float64)
        heading, heading_report = estimate_manhattan_heading(pre_local)
        local_to_canonical = _rotation_xz(-heading)
        canonical_to_local = _rotation_xz(heading)
        pre_canonical = pre_local @ local_to_canonical.T
        fitted_canonical = np.asarray(
            fit_layout(
                floor_xz=pre_canonical,
                need_cube=need_cube,
                show=False,
            ),
            dtype=np.float64,
        )
        fitted_local = fitted_canonical @ canonical_to_local.T
        roundtrip = (pre_local @ local_to_canonical.T) @ canonical_to_local.T
        canonical_area = abs(
            0.5
            * np.sum(
                fitted_canonical[:, 0] * np.roll(fitted_canonical[:, 1], -1)
                - np.roll(fitted_canonical[:, 0], -1) * fitted_canonical[:, 1]
            )
        )
        local_area = abs(
            0.5
            * np.sum(
                fitted_local[:, 0] * np.roll(fitted_local[:, 1], -1)
                - np.roll(fitted_local[:, 0], -1) * fitted_local[:, 1]
            )
        )
        outputs.append(np.insert(fitted_local, 1, 1.0, axis=1))
        reports.append(
            {
                **heading_report,
                "manhattanHeadingRadians": heading,
                "manhattanHeadingDegrees": float(np.degrees(heading)),
                "headingDefinition": "canonical x-axis wall family in panorama-local Bi xz, modulo 90 degrees",
                "localToCanonical": local_to_canonical.tolist(),
                "canonicalToLocal": canonical_to_local.tolist(),
                "roundtripMaxAbsError": float(np.max(np.abs(roundtrip - pre_local))),
                "canonicalLocalAreaAbsDifference": float(abs(canonical_area - local_area)),
                "layoutPolygonPreManhattanLocalBiXZ": pre_local.tolist(),
                "layoutPolygonManhattanCanonicalBiXZ": fitted_canonical.tolist(),
                "layoutPolygonPanoramaLocalBiXZ": fitted_local.tolist(),
            }
        )
    return outputs, reports


def post_process(b_depth, type_name='manhattan', need_cube=False):
    plan_y = 1
    b_xyz = depth2xyz(b_depth, plan_y)

    b_processed_xyz = []
    for xyz in b_xyz:
        if type_name == 'manhattan':
            processed_xz = fit_layout(floor_xz=xyz[..., ::2], need_cube=need_cube, show=False)
        elif type_name == 'manhattan_old':
            processed_xz = fit_layout_old(floor_xz=xyz[..., ::2], need_cube=need_cube, show=False)
        elif type_name == 'atalanta':
            processed_xz = cv2.approxPolyDP(xyz[..., ::2].astype(np.float32), 0.1, False)[:, 0, :]
        else:
            raise NotImplementedError("Unknown post-processing type")

        if need_cube:
            assert len(processed_xz) == 4

        processed_xyz = np.insert(processed_xz, 1, plan_y, axis=1)
        b_processed_xyz.append(processed_xyz)

    return np.array(b_processed_xyz)
