"""
Shared-opening alignment for two independently predicted room layouts.

The layout network currently predicts one polygon per panorama. This module
keeps that model unchanged and aligns two polygons through a passable wall
opening observed from both rooms.
"""
from dataclasses import asdict, dataclass
from html import escape
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class DoorSpec:
    """Legacy name for a shared opening/interface segment on a layout wall."""

    wall_index: int
    start_ratio: float
    end_ratio: float

    def __post_init__(self):
        if self.wall_index < 0:
            raise ValueError("wall_index must be non-negative")
        if not 0 <= self.start_ratio < self.end_ratio <= 1:
            raise ValueError("opening ratios must satisfy 0 <= start < end <= 1")

    @classmethod
    def parse(cls, value: str) -> "DoorSpec":
        try:
            wall_index, start_ratio, end_ratio = value.split(":")
            return cls(int(wall_index), float(start_ratio), float(end_ratio))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "opening spec must use wall_index:start_ratio:end_ratio, for example 1:0.25:0.55"
            ) from exc


def _as_xz(point: Dict) -> np.ndarray:
    xyz = np.asarray(point["xyz"], dtype=np.float64)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError("layout point xyz must contain three finite values")
    return xyz[[0, 2]]


def layout_xz(layout: Dict) -> np.ndarray:
    points = layout.get("layoutPoints", {}).get("points", [])
    if len(points) < 3:
        raise ValueError("layout must contain at least three layout points")
    return np.asarray([_as_xz(point) for point in points], dtype=np.float64)


def _wall_indices(layout: Dict, wall_index: int) -> Tuple[int, int]:
    points = layout["layoutPoints"]["points"]
    walls = layout.get("layoutWalls", {}).get("walls", [])
    if walls:
        if wall_index >= len(walls):
            raise ValueError(f"wall_index {wall_index} is outside the {len(walls)} layout walls")
        indices = walls[wall_index]["pointsIdx"]
        if len(indices) != 2:
            raise ValueError("layout wall pointsIdx must contain exactly two point indices")
        start_index, end_index = indices
    else:
        if wall_index >= len(points):
            raise ValueError(f"wall_index {wall_index} is outside the {len(points)} layout walls")
        start_index, end_index = wall_index, (wall_index + 1) % len(points)

    if not 0 <= start_index < len(points) or not 0 <= end_index < len(points):
        raise ValueError("layout wall references an invalid layout point")
    return start_index, end_index


def door_endpoints(layout: Dict, spec: DoorSpec) -> np.ndarray:
    start_index, end_index = _wall_indices(layout, spec.wall_index)
    points = layout_xz(layout)
    wall_start = points[start_index]
    wall_vector = points[end_index] - wall_start
    if np.linalg.norm(wall_vector) < 1e-8:
        raise ValueError("opening wall must have non-zero length")
    return np.asarray([
        wall_start + spec.start_ratio * wall_vector,
        wall_start + spec.end_ratio * wall_vector,
    ])


def _cross_2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    return (matrix @ homogeneous.T).T[:, :2]


def _fit_similarity(source: np.ndarray, target: np.ndarray, calibrate_scale: bool) -> Tuple[np.ndarray, float]:
    source_vector = source[1] - source[0]
    target_vector = target[1] - target[0]
    source_width = np.linalg.norm(source_vector)
    target_width = np.linalg.norm(target_vector)
    if source_width < 1e-8 or target_width < 1e-8:
        raise ValueError("shared opening must have non-zero width in both rooms")

    scale = target_width / source_width if calibrate_scale else 1.0
    source_angle = np.arctan2(source_vector[1], source_vector[0])
    target_angle = np.arctan2(target_vector[1], target_vector[0])
    angle = target_angle - source_angle
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotation = np.asarray([
        [cos_angle, -sin_angle],
        [sin_angle, cos_angle],
    ])

    source_midpoint = source.mean(axis=0)
    target_midpoint = target.mean(axis=0)
    translation = target_midpoint - scale * rotation @ source_midpoint
    matrix = np.asarray([
        [scale * cos_angle, -scale * sin_angle, translation[0]],
        [scale * sin_angle, scale * cos_angle, translation[1]],
        [0.0, 0.0, 1.0],
    ])
    return matrix, float(scale)


def generate_opening_pose_hypotheses(endpoints_a, endpoints_b, *, independent_matches=(),
                                     seed_ids=None, calibrated=False, tolerance=0.15):
    """Both endpoint mappings in canonical Bi local xz / camera-height units.

    Unlike the legacy joint-layout visualization below, no centroid-side rule
    selects a transform. Other openings, never seed endpoints or their support
    direction, provide held-out evidence. A uniquely supported hypothesis can
    become verified only when the caller supplies calibrated correspondences.
    """
    def segment(value):
        x = np.asarray(value, dtype=np.float64)
        if x.shape != (2, 2) or not np.isfinite(x).all() or np.linalg.norm(x[1]-x[0]) < 1e-8:
            raise ValueError('opening endpoints must be a finite nondegenerate segment')
        return x

    if tolerance <= 0:
        raise ValueError('positive verification tolerance required')
    a, b = segment(endpoints_a), segment(endpoints_b)
    heldout = []
    for match in independent_matches:
        x, y = segment(match['endpoints_a']), segment(match['endpoints_b'])
        ids = (match.get('candidate_id_a'), match.get('candidate_id_b'))
        repeated_id = seed_ids is not None and any(s is not None and s == t for s, t in zip(seed_ids, ids))
        repeated_geometry = any(np.allclose(x, a[::order]) or np.allclose(y, b[::order]) for order in (1, -1))
        if repeated_id or repeated_geometry:
            continue
        heldout.append((x, y, ids))
    hypotheses = []
    for mapping, target in (('same', a), ('reversed', a[::-1])):
        transform, scale = _fit_similarity(b, target, True)
        evidence = []
        for x, y, ids in heldout:
            moved = _transform_points(y, transform)
            residual = min(np.linalg.norm(moved-x, axis=1).max(),
                           np.linalg.norm(moved-x[::-1], axis=1).max()) / np.linalg.norm(x[1]-x[0])
            evidence.append({'candidate_ids': ids, 'relative_endpoint_error': float(residual),
                             'supports': bool(residual <= tolerance)})
        compatible = bool(evidence) and all(e['supports'] for e in evidence)
        hypotheses.append({'endpoint_mapping': mapping, 'T_B_to_A': transform.tolist(),
            'scale_B_to_A': scale, 'independent_evidence': evidence,
            'supported': compatible, 'status': 'unverified' if not evidence else
            ('ambiguous' if compatible else 'rejected')})
    supported = [h for h in hypotheses if h['supported']]
    if len(supported) == 1:
        supported[0]['status'] = 'verified' if calibrated else 'unverified'
    status = ('verified' if len(supported) == 1 and calibrated else
              'ambiguous' if len(supported) > 1 else
              'rejected' if heldout and not supported else 'unverified')
    return {'status': status, 'hypotheses': hypotheses, 'transform_direction': 'B_to_A',
            'coordinate_spec': {'frame': 'bi_panorama_local_xz', 'units': 'camera_height_normalized'},
            'selection': 'held_out_opening_evidence', 'calibrated_correspondences': bool(calibrated)}


def _boundary_lines(points: np.ndarray) -> List[Dict]:
    return [
        {
            "wallIndex": index,
            "start": points[index].tolist(),
            "end": points[(index + 1) % len(points)].tolist(),
        }
        for index in range(len(points))
    ]


def _room_output(room_id: str, layout: Dict, points: np.ndarray, camera_center: np.ndarray) -> Dict:
    return {
        "id": room_id,
        "layoutHeight": layout.get("layoutHeight"),
        "cameraHeight": layout.get("cameraHeight"),
        "cameraCenter": camera_center.tolist(),
        "boundary": points.tolist(),
        "boundaryLines": _boundary_lines(points),
    }


def build_joint_layout(layout_a: Dict, layout_b: Dict, door_a: DoorSpec, door_b: DoorSpec,
                       calibrate_scale: bool = True) -> Dict:
    points_a = layout_xz(layout_a)
    points_b = layout_xz(layout_b)
    endpoints_a = door_endpoints(layout_a, door_a)
    endpoints_b = door_endpoints(layout_b, door_b)

    centroid_a = points_a.mean(axis=0)
    door_vector_a = endpoints_a[1] - endpoints_a[0]
    side_a = _cross_2d(door_vector_a, centroid_a - endpoints_a.mean(axis=0))

    candidates = []
    for mapping, target in (("same", endpoints_a), ("reversed", endpoints_a[::-1])):
        matrix, scale = _fit_similarity(endpoints_b, target, calibrate_scale)
        transformed_points_b = _transform_points(points_b, matrix)
        transformed_endpoints_b = _transform_points(endpoints_b, matrix)
        centroid_b = transformed_points_b.mean(axis=0)
        side_b = _cross_2d(door_vector_a, centroid_b - endpoints_a.mean(axis=0))
        candidates.append({
            "mapping": mapping,
            "matrix": matrix,
            "scale": scale,
            "points_b": transformed_points_b,
            "endpoints_b": transformed_endpoints_b,
            "side_product": side_a * side_b,
        })

    # The shared opening is an interior interface, so the room centroids should
    # be on opposite sides of it after alignment.
    best = min(candidates, key=lambda candidate: candidate["side_product"])
    camera_center_a = np.zeros(2, dtype=np.float64)
    camera_center_b = _transform_points(np.zeros((1, 2), dtype=np.float64), best["matrix"])[0]
    shared_opening = endpoints_a
    shared_opening_output = {
        "roomA": {
            "spec": asdict(door_a),
            "localEndpoints": endpoints_a.tolist(),
            "worldEndpoints": endpoints_a.tolist(),
        },
        "roomB": {
            "spec": asdict(door_b),
            "localEndpoints": endpoints_b.tolist(),
            "worldEndpoints": best["endpoints_b"].tolist(),
        },
        "worldEndpoints": shared_opening.tolist(),
        "width": float(np.linalg.norm(shared_opening[1] - shared_opening[0])),
    }

    return {
        "formatVersion": 1,
        "coordinateSystem": "floorplan-xz-meters",
        "rooms": [
            _room_output("A", layout_a, points_a, camera_center_a),
            _room_output("B", layout_b, best["points_b"], camera_center_b),
        ],
        "sharedOpening": shared_opening_output,
        "sharedDoor": shared_opening_output,
        "alignment": {
            "roomBToWorld": best["matrix"].tolist(),
            "roomBScale": best["scale"],
            "endpointMapping": best["mapping"],
            "centroidSideProduct": float(best["side_product"]),
            "scaleCalibratedFromSharedOpening": calibrate_scale,
            "scaleCalibratedFromSharedDoor": calibrate_scale,
        },
    }


def build_joint_layout_from_endpoints(
    layout_a: Dict,
    layout_b: Dict,
    endpoints_a: Sequence[Sequence[float]],
    endpoints_b: Sequence[Sequence[float]],
    boundary_a: Optional[Sequence[Sequence[float]]] = None,
    boundary_b: Optional[Sequence[Sequence[float]]] = None,
    calibrate_scale: bool = True,
    room_ids: Tuple[str, str] = ("A", "B"),
    coordinate_system: str = "ZInD panorama-A local xy",
    opening_metadata_a: Optional[Mapping[str, Any]] = None,
    opening_metadata_b: Optional[Mapping[str, Any]] = None,
) -> Dict:
    """Build a joint layout from explicit panorama-local portal endpoints.

    This is the endpoint-based counterpart of :func:`build_joint_layout`.  It
    intentionally reuses the same similarity fit, width-ratio scale, and
    opposite-centroid-side selection.  The endpoint order is the predicted
    interval start ray followed by its end ray; both same and reversed B-to-A
    mappings are evaluated without ground truth.
    """

    endpoints_a_value = np.asarray(endpoints_a, dtype=np.float64)
    endpoints_b_value = np.asarray(endpoints_b, dtype=np.float64)
    if endpoints_a_value.shape != (2, 2) or endpoints_b_value.shape != (2, 2):
        raise ValueError("portal endpoints must each have shape [2, 2]")
    if not np.isfinite(endpoints_a_value).all() or not np.isfinite(endpoints_b_value).all():
        raise ValueError("portal endpoints must be finite")
    points_a = (
        layout_xz(layout_a)
        if boundary_a is None
        else np.asarray(boundary_a, dtype=np.float64)
    )
    points_b = (
        layout_xz(layout_b)
        if boundary_b is None
        else np.asarray(boundary_b, dtype=np.float64)
    )
    for side, points in (("A", points_a), ("B", points_b)):
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            raise ValueError(f"room {side} boundary must contain at least three [x, y] points")
        if not np.isfinite(points).all():
            raise ValueError(f"room {side} boundary must be finite")

    centroid_a = points_a.mean(axis=0)
    portal_vector_a = endpoints_a_value[1] - endpoints_a_value[0]
    portal_midpoint_a = endpoints_a_value.mean(axis=0)
    side_a = _cross_2d(portal_vector_a, centroid_a - portal_midpoint_a)
    candidates = []
    for mapping, target in (
        ("same", endpoints_a_value),
        ("reversed", endpoints_a_value[::-1]),
    ):
        matrix, scale = _fit_similarity(
            endpoints_b_value, target, calibrate_scale
        )
        transformed_points_b = _transform_points(points_b, matrix)
        transformed_endpoints_b = _transform_points(endpoints_b_value, matrix)
        side_b = _cross_2d(
            portal_vector_a,
            transformed_points_b.mean(axis=0) - portal_midpoint_a,
        )
        candidates.append({
            "mapping": mapping,
            "matrix": matrix,
            "scale": scale,
            "points_b": transformed_points_b,
            "endpoints_b": transformed_endpoints_b,
            "side_product": side_a * side_b,
        })
    best = min(
        candidates,
        key=lambda candidate: (candidate["side_product"], candidate["mapping"]),
    )
    camera_center_a = np.zeros(2, dtype=np.float64)
    camera_center_b = _transform_points(
        np.zeros((1, 2), dtype=np.float64), best["matrix"]
    )[0]
    shared_opening = {
        "roomA": {
            "spec": dict(opening_metadata_a or {}),
            "localEndpoints": endpoints_a_value.tolist(),
            "worldEndpoints": endpoints_a_value.tolist(),
        },
        "roomB": {
            "spec": dict(opening_metadata_b or {}),
            "localEndpoints": endpoints_b_value.tolist(),
            "worldEndpoints": best["endpoints_b"].tolist(),
        },
        "worldEndpoints": endpoints_a_value.tolist(),
        "width": float(np.linalg.norm(portal_vector_a)),
    }
    return {
        "formatVersion": 1,
        "coordinateSystem": coordinate_system,
        "rooms": [
            _room_output(str(room_ids[0]), layout_a, points_a, camera_center_a),
            _room_output(
                str(room_ids[1]), layout_b, best["points_b"], camera_center_b
            ),
        ],
        "sharedOpening": shared_opening,
        "sharedDoor": dict(shared_opening),
        "alignment": {
            "roomBToWorld": best["matrix"].tolist(),
            "roomBScale": best["scale"],
            "endpointMapping": best["mapping"],
            "centroidSideProduct": float(best["side_product"]),
            "scaleCalibratedFromSharedOpening": calibrate_scale,
            "scaleCalibratedFromSharedDoor": calibrate_scale,
            "transformDirection": "B_to_A",
            "selection": "predicted polygon centroid opposite-side rule",
        },
    }


def _joint_boundary_projector(joint_layout: Dict, side_length: int, padding: int):
    rooms = joint_layout["rooms"]
    room_points = [np.asarray(room["boundary"], dtype=np.float64) for room in rooms]
    shared = joint_layout.get("sharedOpening", joint_layout["sharedDoor"])
    door = np.asarray(shared["worldEndpoints"], dtype=np.float64)
    all_points = np.concatenate(room_points + [door], axis=0)

    minimum = all_points.min(axis=0)
    maximum = all_points.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-6)
    scale = (side_length - 2 * padding) / span.max()
    center = (minimum + maximum) / 2

    def to_pixel(points: np.ndarray) -> np.ndarray:
        pixels = (points - center) * scale
        pixels[:, 1] *= -1
        pixels += side_length / 2
        return np.rint(pixels).astype(np.int32)

    return rooms, room_points, door, to_pixel


def render_joint_boundary_svg(
    joint_layout: Dict,
    save_path: str,
    side_length: int = 900,
    padding: int = 70,
    show_wall_indices: bool = True,
) -> None:
    rooms, room_points, door, to_pixel = _joint_boundary_projector(joint_layout, side_length, padding)
    colors = ["#1e6ecd", "#289128"]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{side_length}" height="{side_length}" '
        f'viewBox="0 0 {side_length} {side_length}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for room, points, color in zip(rooms, room_points, colors):
        pixels = to_pixel(points)
        polygon = " ".join(f"{x},{y}" for x, y in pixels)
        svg.append(f'<polygon points="{polygon}" fill="none" stroke="{color}" stroke-width="4"/>')
        room_center = to_pixel(np.asarray([points.mean(axis=0)]))[0]
        svg.append(
            f'<text x="{room_center[0]}" y="{room_center[1]}" fill="{color}" '
            f'font-size="24">Room {escape(str(room["id"]))}</text>'
        )
        if show_wall_indices:
            for wall_index, wall_start in enumerate(pixels):
                wall_end = pixels[(wall_index + 1) % len(pixels)]
                wall_midpoint = (wall_start + wall_end) // 2
                svg.append(
                    f'<text x="{wall_midpoint[0]}" y="{wall_midpoint[1]}" fill="{color}" '
                    f'font-size="18">{escape(str(room["id"]))}:{wall_index}</text>'
                )
        camera = to_pixel(np.asarray([room["cameraCenter"]], dtype=np.float64))[0]
        svg.append(f'<circle cx="{camera[0]}" cy="{camera[1]}" r="7" fill="{color}"/>')

    door_pixels = to_pixel(door)
    svg.append(
        f'<line x1="{door_pixels[0, 0]}" y1="{door_pixels[0, 1]}" '
        f'x2="{door_pixels[1, 0]}" y2="{door_pixels[1, 1]}" stroke="#dc1e1e" stroke-width="8"/>'
    )
    door_midpoint = door_pixels.mean(axis=0).astype(np.int32)
    svg.append(
        f'<text x="{door_midpoint[0]}" y="{door_midpoint[1]}" fill="#dc1e1e" '
        f'font-size="20">shared opening</text>'
    )
    svg.append("</svg>")
    with open(save_path, "w") as file:
        file.write("\n".join(svg) + "\n")


def render_joint_boundary(
    joint_layout: Dict,
    save_path: str,
    side_length: int = 900,
    padding: int = 70,
    show_wall_indices: bool = True,
) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to render the joint boundary PNG image") from exc

    rooms, room_points, door, to_pixel = _joint_boundary_projector(joint_layout, side_length, padding)
    canvas = np.full((side_length, side_length, 3), 255, dtype=np.uint8)
    colors = [(205, 110, 30), (40, 145, 40)]
    for room, points, color in zip(rooms, room_points, colors):
        pixels = to_pixel(points)
        cv2.polylines(canvas, [pixels], isClosed=True, color=color, thickness=4, lineType=cv2.LINE_AA)
        label_at = tuple(to_pixel(np.asarray([points.mean(axis=0)]))[0])
        cv2.putText(canvas, f"Room {room['id']}", label_at, cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2, cv2.LINE_AA)
        if show_wall_indices:
            for wall_index, wall_start in enumerate(pixels):
                wall_end = pixels[(wall_index + 1) % len(pixels)]
                wall_midpoint = tuple(((wall_start + wall_end) // 2).tolist())
                cv2.putText(canvas, f"{room['id']}:{wall_index}", wall_midpoint,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        camera = to_pixel(np.asarray([room["cameraCenter"]], dtype=np.float64))[0]
        cv2.drawMarker(canvas, tuple(camera), color, cv2.MARKER_CROSS, 18, 3)

    door_pixels = to_pixel(door)
    cv2.line(canvas, tuple(door_pixels[0]), tuple(door_pixels[1]), (30, 30, 220), 8, cv2.LINE_AA)
    cv2.putText(canvas, "shared opening", tuple(door_pixels.mean(axis=0).astype(np.int32)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 220), 2, cv2.LINE_AA)
    cv2.imwrite(save_path, canvas)
