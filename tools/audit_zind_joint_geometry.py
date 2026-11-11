"""CPU-only geometry oracle audit for the fixed ZInD three-room T3 output.

This tool never calls a model.  It freezes the candidate pairs already chosen
by the saved T3 inference and attaches train ground truth only after those
predictions have been loaded.  Oracle hypotheses are diagnostic artifacts and
must not be routed back into inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from portallayout.datasets.zind_pair_mining import (
    relative_zind_transform,
    transform_points,
    zind_transform_matrix,
)
from portallayout.engine.cross_scene_estimator import (
    opening_candidates_from_intervals,
    polygon_overlap_ratio,
    polygon_validity,
)
from portallayout.geometry.joint_layout import door_endpoints


DEFAULT_GROUP = REPO_ROOT / "data/ZInD_matching_groups/debug_train_0665_floor_03.json"
DEFAULT_T3 = REPO_ROOT / "outputs/zind_room_group_debug/t3_best_0665_floor03"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/zind_room_group_debug/t3_geometry_oracle_0665_floor03"
DEFAULT_TRAIN = REPO_ROOT / "data/ZInD_matching/train.json"
DEFAULT_VAL = REPO_ROOT / "data/ZInD_matching/val.json"

TOKEN_COUNT = 256
BI_TO_ZIND_2D = np.diag([-1.0, 1.0])
CASES = ("C0", "C1", "C2", "C3", "C4")
FRAME_MODES = ("current", "frame_restored")
SCALE_MODES = ("width_ratio", "unit", "gt_scale")
ENDPOINT_MAPPINGS = ("same", "reversed")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group_manifest", default=str(DEFAULT_GROUP))
    parser.add_argument("--rooms_json", default=str(DEFAULT_T3 / "rooms.json"))
    parser.add_argument("--pairs_json", default=str(DEFAULT_T3 / "pairs.json"))
    parser.add_argument("--group_summary", default=str(DEFAULT_T3 / "group_summary.json"))
    parser.add_argument("--train_manifest", default=str(DEFAULT_TRAIN))
    parser.add_argument("--val_manifest", default=str(DEFAULT_VAL))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args(argv)


def reject_test_data_path(path: Path, label: str) -> None:
    """Reject a data input whose exact path component/stem is ``test``."""
    resolved = path.expanduser().resolve()
    if resolved.stem.lower() == "test" or any(part.lower() == "test" for part in resolved.parts):
        raise ValueError(f"{label} points to forbidden test data: {resolved}")


def resolved_input(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    reject_test_data_path(path, label)
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def dump_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")


def json_pointer(payload: Mapping[str, Any], pointer: str) -> Any:
    value: Any = payload
    if not pointer.startswith("/"):
        raise ValueError(f"JSON pointer must start with '/': {pointer}")
    for raw in pointer[1:].split("/"):
        key = raw.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, Mapping) or key not in value:
            raise KeyError(f"JSON pointer does not resolve: {pointer}")
        value = value[key]
    return value


def rotation_matrix(angle_radians: float) -> np.ndarray:
    cosine, sine = math.cos(angle_radians), math.sin(angle_radians)
    return np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)


def homogeneous_rotation(angle_radians: float) -> np.ndarray:
    output = np.eye(3, dtype=np.float64)
    output[:2, :2] = rotation_matrix(angle_radians)
    return output


def apply_transform(points: Sequence[Sequence[float]], matrix: np.ndarray) -> np.ndarray:
    return transform_points(points, np.asarray(matrix, dtype=np.float64))


def transform_scale(matrix: np.ndarray) -> float:
    return float(math.hypot(float(matrix[0, 0]), float(matrix[1, 0])))


def transform_yaw(matrix: np.ndarray) -> float:
    return float(math.atan2(float(matrix[1, 0]), float(matrix[0, 0])))


def wrap_angle_radians(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def circular_yaw_error(first: float, second: float) -> float:
    return abs(wrap_angle_radians(float(first) - float(second)))


def tokens_in_interval(start: int, end: int, token_count: int = TOKEN_COUNT) -> np.ndarray:
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    start, end = int(start) % token_count, int(end) % token_count
    if start <= end:
        return np.arange(start, end + 1, dtype=np.int64)
    return np.concatenate(
        (np.arange(start, token_count, dtype=np.int64), np.arange(0, end + 1, dtype=np.int64))
    )


def compose_canonical_to_local(
    transform_b_to_a_canonical: np.ndarray,
    canonical_to_local_a: np.ndarray,
    canonical_to_local_b: np.ndarray,
) -> np.ndarray:
    """Convert B->A from canonical coordinates when C maps canonical->local."""
    return (
        np.asarray(canonical_to_local_a, dtype=np.float64)
        @ np.asarray(transform_b_to_a_canonical, dtype=np.float64)
        @ np.linalg.inv(np.asarray(canonical_to_local_b, dtype=np.float64))
    )


def fit_similarity_from_endpoints(
    source_b: Sequence[Sequence[float]],
    target_a: Sequence[Sequence[float]],
    scale: float,
) -> np.ndarray:
    """Return p_A = s R p_B + t for an explicitly supplied scale."""
    source = np.asarray(source_b, dtype=np.float64)
    target = np.asarray(target_a, dtype=np.float64)
    if source.shape != (2, 2) or target.shape != (2, 2):
        raise ValueError("endpoint arrays must have shape [2, 2]")
    if not math.isfinite(float(scale)) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    vector_b = source[1] - source[0]
    vector_a = target[1] - target[0]
    if min(np.linalg.norm(vector_a), np.linalg.norm(vector_b)) < 1e-10:
        raise ValueError("opening endpoints must have non-zero width")
    angle = math.atan2(vector_a[1], vector_a[0]) - math.atan2(vector_b[1], vector_b[0])
    rotation = rotation_matrix(angle)
    translation = target.mean(axis=0) - float(scale) * rotation @ source.mean(axis=0)
    output = np.eye(3, dtype=np.float64)
    output[:2, :2] = float(scale) * rotation
    output[:2, 2] = translation
    return output


def signed_polygon_area(points: Sequence[Sequence[float]]) -> float:
    value = np.asarray(points, dtype=np.float64)
    return float(
        0.5
        * np.sum(
            value[:, 0] * np.roll(value[:, 1], -1)
            - np.roll(value[:, 0], -1) * value[:, 1]
        )
    )


def angle_degrees(vector: np.ndarray) -> float:
    return math.degrees(math.atan2(float(vector[1]), float(vector[0])))


def axis_angle_error(first: float, second: float) -> float:
    """Unoriented line-angle error in radians, in [0, pi/2]."""
    return abs((float(first) - float(second) + math.pi / 2.0) % math.pi - math.pi / 2.0)


def polygon_to_bi_layout(points_zind: Sequence[Sequence[float]]) -> Dict[str, Any]:
    points = np.asarray(points_zind, dtype=np.float64) @ BI_TO_ZIND_2D.T
    return {
        "layoutPoints": {
            "num": int(len(points)),
            "points": [
                {"xyz": [float(point[0]), 1.0, float(point[1])], "id": index}
                for index, point in enumerate(points)
            ],
        },
        "layoutWalls": {"num": 0, "walls": []},
    }


def map_interval_to_polygon(
    polygon_zind: Sequence[Sequence[float]],
    interval: Sequence[int],
    probabilities: Sequence[float],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    layout = polygon_to_bi_layout(polygon_zind)
    candidates = opening_candidates_from_intervals(layout, [interval], probabilities)
    if len(candidates) != 1:
        raise RuntimeError("one interval must produce exactly one geometry candidate")
    candidate = candidates[0]
    endpoints_bi = door_endpoints(layout, candidate.spec)
    endpoints_zind = endpoints_bi @ BI_TO_ZIND_2D.T
    wall_index = int(candidate.wall_index)
    polygon = np.asarray(polygon_zind, dtype=np.float64)
    wall = np.asarray(
        [polygon[wall_index], polygon[(wall_index + 1) % len(polygon)]], dtype=np.float64
    )
    return endpoints_zind, {
        "candidate": candidate.to_json(),
        "wall_endpoints": wall.tolist(),
    }


def portal_diagnostics(
    polygon: Sequence[Sequence[float]],
    endpoints: Sequence[Sequence[float]],
    wall_endpoints: Optional[Sequence[Sequence[float]]] = None,
) -> Dict[str, Any]:
    poly = np.asarray(polygon, dtype=np.float64)
    portal = np.asarray(endpoints, dtype=np.float64)
    vector = portal[1] - portal[0]
    width = float(np.linalg.norm(vector))
    if width < 1e-10:
        raise ValueError("portal width is zero")
    tangent = vector / width
    left_normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
    centroid = poly.mean(axis=0)
    midpoint = portal.mean(axis=0)
    if float(np.dot(left_normal, centroid - midpoint)) < 0:
        inward = -left_normal
    else:
        inward = left_normal
    side = float(np.cross(vector, centroid - midpoint))
    wall = portal if wall_endpoints is None else np.asarray(wall_endpoints, dtype=np.float64)
    wall_vector = wall[1] - wall[0]
    return {
        "opening_endpoints": portal.tolist(),
        "opening_midpoint": midpoint.tolist(),
        "opening_width": width,
        "opening_tangent_degrees": angle_degrees(vector),
        "wall_endpoints": wall.tolist(),
        "wall_tangent_degrees": angle_degrees(wall_vector),
        "polygon_orientation": "ccw" if signed_polygon_area(poly) > 0 else "clockwise",
        "polygon_centroid": centroid.tolist(),
        "centroid_portal_cross": side,
        "centroid_side": "left" if side > 0 else "right" if side < 0 else "on",
        "inward_normal": inward.tolist(),
        "outward_normal": (-inward).tolist(),
    }


def derive_oracle_frame_heading(
    predicted_polygon: Sequence[Sequence[float]],
    predicted_endpoints: Sequence[Sequence[float]],
    gt_local_polygon: Sequence[Sequence[float]],
    gt_global_endpoints: Sequence[Sequence[float]],
    floor_plan_transformation: Mapping[str, Any],
) -> Dict[str, Any]:
    """Recover a diagnostic canonical->pano heading without error search.

    The quadrant is fixed by two physical constraints: the predicted portal
    axis must agree with the labeled global portal axis, and the predicted room
    centroid must remain on the labeled room side of that portal.  GT makes
    this oracle-only; no pose-error minimization is used.
    """
    pred_poly = np.asarray(predicted_polygon, dtype=np.float64)
    pred_portal = np.asarray(predicted_endpoints, dtype=np.float64)
    gt_poly = np.asarray(gt_local_polygon, dtype=np.float64)
    gt_portal = np.asarray(gt_global_endpoints, dtype=np.float64)
    local_to_global = zind_transform_matrix(floor_plan_transformation)
    gt_poly_global = apply_transform(gt_poly, local_to_global)
    global_vector = gt_portal[1] - gt_portal[0]
    global_midpoint = gt_portal.mean(axis=0)
    gt_side = float(np.cross(global_vector, gt_poly_global.mean(axis=0) - global_midpoint))
    pred_axis = math.atan2(
        float(pred_portal[1, 1] - pred_portal[0, 1]),
        float(pred_portal[1, 0] - pred_portal[0, 0]),
    )
    global_axis = math.atan2(float(global_vector[1]), float(global_vector[0]))
    options = []
    for quadrant in range(4):
        canonical_to_global = quadrant * math.pi / 2.0
        rotation = rotation_matrix(canonical_to_global)
        predicted_centroid_global = global_midpoint + rotation @ (
            pred_poly.mean(axis=0) - pred_portal.mean(axis=0)
        )
        predicted_side = float(
            np.cross(global_vector, predicted_centroid_global - global_midpoint)
        )
        options.append(
            {
                "quadrant": quadrant,
                "canonical_to_global_degrees": math.degrees(canonical_to_global),
                "axis_error_degrees": math.degrees(
                    axis_angle_error(pred_axis + canonical_to_global, global_axis)
                ),
                "same_room_side": bool(gt_side * predicted_side > 0),
                "gt_centroid_side_cross": gt_side,
                "predicted_centroid_side_cross": predicted_side,
            }
        )
    selected = min(
        options,
        key=lambda value: (
            not value["same_room_side"],
            value["axis_error_degrees"],
            value["quadrant"],
        ),
    )
    heading = math.radians(selected["canonical_to_global_degrees"]) - math.radians(
        float(floor_plan_transformation["rotation"])
    )
    matrix = homogeneous_rotation(heading)
    sample = np.concatenate((pred_poly[: min(4, len(pred_poly))], pred_portal), axis=0)
    roundtrip = apply_transform(apply_transform(sample, matrix), np.linalg.inv(matrix))
    return {
        "definition": "canonical_to_pano_local",
        "oracle_source": "GT global portal axis and GT room side; no pose-error search",
        "heading_radians": wrap_angle_radians(heading),
        "heading_degrees": math.degrees(wrap_angle_radians(heading)),
        "matrix": matrix.tolist(),
        "inverse": np.linalg.inv(matrix).tolist(),
        "roundtrip_max_abs_error": float(np.max(np.abs(roundtrip - sample))),
        "selected_quadrant": selected,
        "all_quadrants": options,
    }


def formal_prediction_payload(pair: Mapping[str, Any], rooms: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the saved-prediction-only fields used to construct C0."""
    room_a, room_b = str(pair["room_a"]), str(pair["room_b"])
    index_a, index_b = (int(value) for value in pair["best_candidate_pair"])
    return {
        "pair_id": str(pair["pair_id"]),
        "room_a": room_a,
        "room_b": room_b,
        "best_candidate_pair": [index_a, index_b],
        "polygon_a": rooms[room_a]["single_room_layout"]["predicted_polygon_zind_local_xy"],
        "polygon_b": rooms[room_b]["single_room_layout"]["predicted_polygon_zind_local_xy"],
        "interval_a": rooms[room_a]["opening_prediction"]["intervals"][index_a],
        "interval_b": rooms[room_b]["opening_prediction"]["intervals"][index_b],
        "probability_a": rooms[room_a]["opening_prediction"]["probability"],
        "probability_b": rooms[room_b]["opening_prediction"]["probability"],
    }


def audit_negative_prediction(pair: Mapping[str, Any]) -> Dict[str, Any]:
    if bool(pair.get("is_match")):
        raise ValueError("negative audit received a matched prediction")
    if pair.get("best_candidate_pair") != [-1, -1]:
        raise ValueError("negative prediction must use the dustbin pair [-1, -1]")
    if pair.get("relative_pose") is not None or pair.get("joint_bev") is not None:
        raise ValueError("negative prediction must not contain pose or joint BEV")
    return {
        "pair_id": pair["pair_id"],
        "is_match": False,
        "best_candidate_pair": [-1, -1],
        "pose_generated": False,
        "joint_bev_generated": False,
        "oracle_pose_generated": False,
    }


@dataclass
class EndpointCase:
    name: str
    polygon_a: np.ndarray
    polygon_b: np.ndarray
    endpoints_a: np.ndarray
    endpoints_b: np.ndarray
    source_a: Dict[str, Any]
    source_b: Dict[str, Any]
    oracle_only: bool


def build_endpoint_cases(
    pair: Mapping[str, Any],
    record: Mapping[str, Any],
    rooms: Mapping[str, Any],
) -> Dict[str, EndpointCase]:
    frozen = formal_prediction_payload(pair, rooms)
    room_a, room_b = frozen["room_a"], frozen["room_b"]
    target_a = int(record["supervision"]["target_candidate_a"])
    target_b = int(record["supervision"]["target_candidate_b"])
    gt_candidate_a = record["candidates_a"][target_a]
    gt_candidate_b = record["candidates_b"][target_b]
    predicted_a = np.asarray(frozen["polygon_a"], dtype=np.float64)
    predicted_b = np.asarray(frozen["polygon_b"], dtype=np.float64)
    gt_a = np.asarray(rooms[room_a]["ground_truth"]["layouts"]["complete"]["vertices"], dtype=np.float64)
    gt_b = np.asarray(rooms[room_b]["ground_truth"]["layouts"]["complete"]["vertices"], dtype=np.float64)

    definitions = {
        "C0": (predicted_a, predicted_b, frozen["interval_a"], frozen["interval_b"]),
        "C1": (
            predicted_a,
            predicted_b,
            gt_candidate_a["tokenInterval"],
            gt_candidate_b["tokenInterval"],
        ),
        "C2": (gt_a, gt_b, frozen["interval_a"], frozen["interval_b"]),
        "C3": (
            gt_a,
            gt_b,
            gt_candidate_a["tokenInterval"],
            gt_candidate_b["tokenInterval"],
        ),
    }
    output: Dict[str, EndpointCase] = {}
    for name, (polygon_a, polygon_b, interval_a, interval_b) in definitions.items():
        endpoints_a, mapped_a = map_interval_to_polygon(
            polygon_a, interval_a, frozen["probability_a"]
        )
        endpoints_b, mapped_b = map_interval_to_polygon(
            polygon_b, interval_b, frozen["probability_b"]
        )
        mapped_a.update(
            interval=[int(value) for value in interval_a],
            interval_source="predicted" if name in ("C0", "C2") else "GT manifest",
            endpoint_source="current interval-to-wall mapper",
        )
        mapped_b.update(
            interval=[int(value) for value in interval_b],
            interval_source="predicted" if name in ("C0", "C2") else "GT manifest",
            endpoint_source="current interval-to-wall mapper",
        )
        output[name] = EndpointCase(
            name=name,
            polygon_a=np.asarray(polygon_a, dtype=np.float64),
            polygon_b=np.asarray(polygon_b, dtype=np.float64),
            endpoints_a=endpoints_a,
            endpoints_b=endpoints_b,
            source_a=mapped_a,
            source_b=mapped_b,
            oracle_only=name != "C0",
        )
    output["C4"] = EndpointCase(
        name="C4",
        polygon_a=gt_a,
        polygon_b=gt_b,
        endpoints_a=np.asarray(gt_candidate_a["localEndpoints"], dtype=np.float64),
        endpoints_b=np.asarray(gt_candidate_b["localEndpoints"], dtype=np.float64),
        source_a={
            "interval": gt_candidate_a["tokenInterval"],
            "interval_source": "GT manifest (not used for endpoints)",
            "endpoint_source": "GT localEndpoints direct",
            "candidate": None,
            "wall_endpoints": None,
        },
        source_b={
            "interval": gt_candidate_b["tokenInterval"],
            "interval_source": "GT manifest (not used for endpoints)",
            "endpoint_source": "GT localEndpoints direct",
            "candidate": None,
            "wall_endpoints": None,
        },
        oracle_only=True,
    )
    return output


def transform_case_frame(
    case: EndpointCase,
    mode: str,
    pair: Mapping[str, Any],
    record: Mapping[str, Any],
    rooms: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    if mode not in FRAME_MODES:
        raise ValueError(f"unknown frame mode: {mode}")
    if mode == "current":
        identity = np.eye(3, dtype=np.float64).tolist()
        return case.polygon_a, case.polygon_b, case.endpoints_a, case.endpoints_b, {
            "mode": mode,
            "room_a": {"matrix": identity, "reason": "current saved coordinate interpretation"},
            "room_b": {"matrix": identity, "reason": "current saved coordinate interpretation"},
        }
    if case.name in ("C2", "C3", "C4"):
        identity = np.eye(3, dtype=np.float64).tolist()
        return case.polygon_a, case.polygon_b, case.endpoints_a, case.endpoints_b, {
            "mode": mode,
            "room_a": {
                "matrix": identity,
                "heading_degrees": 0.0,
                "roundtrip_max_abs_error": 0.0,
                "reason": "GT polygon is already panorama-local; restoration is identity",
            },
            "room_b": {
                "matrix": identity,
                "heading_degrees": 0.0,
                "roundtrip_max_abs_error": 0.0,
                "reason": "GT polygon is already panorama-local; restoration is identity",
            },
        }

    room_a, room_b = str(pair["room_a"]), str(pair["room_b"])
    target_a = int(record["supervision"]["target_candidate_a"])
    target_b = int(record["supervision"]["target_candidate_b"])
    frame_a = derive_oracle_frame_heading(
        case.polygon_a,
        case.endpoints_a,
        rooms[room_a]["ground_truth"]["layouts"]["complete"]["vertices"],
        record["candidates_a"][target_a]["globalEndpoints"],
        rooms[room_a]["ground_truth"]["floor_plan_transformation"],
    )
    frame_b = derive_oracle_frame_heading(
        case.polygon_b,
        case.endpoints_b,
        rooms[room_b]["ground_truth"]["layouts"]["complete"]["vertices"],
        record["candidates_b"][target_b]["globalEndpoints"],
        rooms[room_b]["ground_truth"]["floor_plan_transformation"],
    )
    matrix_a = np.asarray(frame_a["matrix"], dtype=np.float64)
    matrix_b = np.asarray(frame_b["matrix"], dtype=np.float64)
    return (
        apply_transform(case.polygon_a, matrix_a),
        apply_transform(case.polygon_b, matrix_b),
        apply_transform(case.endpoints_a, matrix_a),
        apply_transform(case.endpoints_b, matrix_b),
        {"mode": mode, "room_a": frame_a, "room_b": frame_b},
    )


def evaluate_hypothesis(
    pair_id: str,
    room_a: str,
    room_b: str,
    case: EndpointCase,
    frame_mode: str,
    scale_mode: str,
    endpoint_mapping: str,
    polygon_a: np.ndarray,
    polygon_b: np.ndarray,
    endpoints_a: np.ndarray,
    endpoints_b: np.ndarray,
    gt_transform: np.ndarray,
    metric_factor: float,
    target_instances: Sequence[Mapping[str, Any]],
    frame_report: Mapping[str, Any],
) -> Dict[str, Any]:
    if endpoint_mapping not in ENDPOINT_MAPPINGS:
        raise ValueError(endpoint_mapping)
    target = endpoints_a if endpoint_mapping == "same" else endpoints_a[::-1]
    width_a = float(np.linalg.norm(target[1] - target[0]))
    width_b = float(np.linalg.norm(endpoints_b[1] - endpoints_b[0]))
    gt_scale = transform_scale(gt_transform)
    if scale_mode == "width_ratio":
        scale = width_a / width_b
    elif scale_mode == "unit":
        scale = 1.0
    elif scale_mode == "gt_scale":
        scale = gt_scale
    else:
        raise ValueError(scale_mode)
    transform = fit_similarity_from_endpoints(endpoints_b, target, scale)
    transformed_b = apply_transform(polygon_b, transform)
    transformed_endpoints_b = apply_transform(endpoints_b, transform)
    portal_vector = endpoints_a[1] - endpoints_a[0]
    portal_midpoint = endpoints_a.mean(axis=0)
    side_a = float(np.cross(portal_vector, polygon_a.mean(axis=0) - portal_midpoint))
    side_b = float(np.cross(portal_vector, transformed_b.mean(axis=0) - portal_midpoint))
    side_product = side_a * side_b
    endpoint_delta = transformed_endpoints_b - target
    endpoint_rmse = float(math.sqrt(float(np.mean(np.square(endpoint_delta)))))
    midpoint_distance = float(
        np.linalg.norm(transformed_endpoints_b.mean(axis=0) - target.mean(axis=0))
    )
    tangent_error = circular_yaw_error(
        math.atan2(
            transformed_endpoints_b[1, 1] - transformed_endpoints_b[0, 1],
            transformed_endpoints_b[1, 0] - transformed_endpoints_b[0, 0],
        ),
        math.atan2(target[1, 1] - target[0, 1], target[1, 0] - target[0, 0]),
    )
    predicted_yaw = transform_yaw(transform)
    gt_yaw = transform_yaw(gt_transform)
    translation_error = float(np.linalg.norm(transform[:2, 2] - gt_transform[:2, 2]))
    residual = np.linalg.inv(gt_transform) @ transform
    validity_a = polygon_validity(polygon_a)
    validity_b = polygon_validity(transformed_b)
    target_a = np.asarray(target_instances[0]["polygon"], dtype=np.float64)
    target_b = np.asarray(target_instances[1]["polygon"], dtype=np.float64)
    row: Dict[str, Any] = {
        "pair_id": pair_id,
        "room_a": room_a,
        "room_b": room_b,
        "case": case.name,
        "frame_mode": frame_mode,
        "scale_mode": scale_mode,
        "endpoint_mapping": endpoint_mapping,
        "selected_by_centroid_side": False,
        "oracle_only": True,
        "valid_for_inference": False,
        "predicted_yaw_degrees": math.degrees(predicted_yaw),
        "gt_yaw_degrees": math.degrees(gt_yaw),
        "yaw_error_degrees": math.degrees(circular_yaw_error(predicted_yaw, gt_yaw)),
        "scale": scale,
        "gt_scale": gt_scale,
        "absolute_log_scale_error": abs(math.log(scale / gt_scale)),
        "translation": transform[:2, 2].tolist(),
        "gt_translation": gt_transform[:2, 2].tolist(),
        "translation_error": translation_error,
        "translation_meter_error": translation_error * float(metric_factor),
        "T_B_to_A": transform.tolist(),
        "gt_T_B_to_A": gt_transform.tolist(),
        "residual_inv_gt_times_pred": residual.tolist(),
        "endpoint_rmse": endpoint_rmse,
        "opening_midpoint_distance": midpoint_distance,
        "opening_tangent_error_degrees": math.degrees(tangent_error),
        "opening_width_a": width_a,
        "opening_width_b": width_b,
        "width_ratio": width_a / width_b,
        "centroid_side_a": side_a,
        "centroid_side_b": side_b,
        "centroid_side_product": side_product,
        "rooms_on_opposite_portal_sides": bool(side_product < 0),
        "polygon_a_valid": bool(validity_a["valid"]),
        "polygon_b_valid": bool(validity_b["valid"]),
        "polygon_a_orientation": "ccw" if signed_polygon_area(polygon_a) > 0 else "clockwise",
        "polygon_b_orientation": "ccw" if signed_polygon_area(transformed_b) > 0 else "clockwise",
        "joint_bev_instance_overlap_A": polygon_overlap_ratio(polygon_a, target_a),
        "joint_bev_instance_overlap_B": polygon_overlap_ratio(transformed_b, target_b),
        "frame_heading_a_degrees": frame_report["room_a"].get("heading_degrees", 0.0),
        "frame_heading_b_degrees": frame_report["room_b"].get("heading_degrees", 0.0),
    }
    return row


def select_centroid_hypotheses(rows: List[Dict[str, Any]]) -> None:
    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (row["pair_id"], row["case"], row["frame_mode"], row["scale_mode"])
        grouped.setdefault(key, []).append(row)
    for values in grouped.values():
        if {value["endpoint_mapping"] for value in values} != set(ENDPOINT_MAPPINGS):
            raise RuntimeError("every hypothesis group must enumerate same and reversed")
        selected = min(values, key=lambda value: (value["centroid_side_product"], value["endpoint_mapping"]))
        selected["selected_by_centroid_side"] = True
        if (
            selected["case"] == "C0"
            and selected["frame_mode"] == "current"
            and selected["scale_mode"] == "width_ratio"
        ):
            selected["oracle_only"] = False
            selected["valid_for_inference"] = True


def selected_row(
    rows: Sequence[Mapping[str, Any]],
    pair_id: str,
    case: str,
    frame_mode: str = "current",
    scale_mode: str = "width_ratio",
) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if row["pair_id"] == pair_id
        and row["case"] == case
        and row["frame_mode"] == frame_mode
        and row["scale_mode"] == scale_mode
        and row["selected_by_centroid_side"]
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one selected row, found {len(matches)}")
    return matches[0]


def finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    return True


def descriptive(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        raise ValueError("cannot summarize an empty array")
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def scale_label_statistics(manifest_paths: Sequence[Path]) -> Dict[str, Any]:
    scales: List[float] = []
    log_scales: List[float] = []
    yaw_modulo: List[float] = []
    split_counts: Dict[str, int] = {}
    split_non_unit_counts: Dict[str, int] = {}
    for path in manifest_paths:
        reject_test_data_path(path, "scale-stat manifest")
        manifest = load_json(path)
        split = str(manifest.get("split"))
        if split not in ("train", "val"):
            raise ValueError(f"scale-stat manifest must be train or val, got {split}")
        split_count = 0
        split_scales: List[float] = []
        for record in manifest.get("pairs", []):
            supervision = record.get("supervision", {})
            if supervision.get("is_match") is not True or supervision.get("pose_valid") is not True:
                continue
            matrix = np.asarray(supervision["relative_transform_b_to_a"], dtype=np.float64)
            scale = transform_scale(matrix)
            yaw_degrees = math.degrees(transform_yaw(matrix))
            residual = abs((yaw_degrees + 45.0) % 90.0 - 45.0)
            scales.append(scale)
            split_scales.append(scale)
            log_scales.append(abs(math.log(scale)))
            yaw_modulo.append(residual)
            split_count += 1
        split_counts[split] = split_count
        split_non_unit_counts[split] = sum(
            abs(value - 1.0) > 1e-9 for value in split_scales
        )
    array = np.asarray(scales, dtype=np.float64)
    non_unit = array[np.abs(array - 1.0) > 1e-9]
    return {
        "source": "derived from supervision.relative_transform_b_to_a for pose_valid positive pairs",
        "splits_read": sorted(split_counts),
        "split_counts": split_counts,
        "split_non_unit_counts": split_non_unit_counts,
        "relative_scale": descriptive(scales),
        "relative_scale_exactly_equal_one_ratio": float(np.mean(array == 1.0)),
        "relative_scale_within_1e-9_of_one_ratio": float(np.mean(np.abs(array - 1.0) <= 1e-9)),
        "relative_scale_outside_1e-9_of_one_count": int(len(non_unit)),
        "relative_scale_outside_1e-9_of_one_ratio": float(len(non_unit) / len(array)),
        "non_unit_relative_scale": descriptive(non_unit.tolist()),
        "absolute_log_relative_scale": descriptive(log_scales),
        "absolute_gt_yaw_residual_to_nearest_90_degrees": descriptive(yaw_modulo),
    }


def load_inputs(args: argparse.Namespace) -> Dict[str, Any]:
    paths = {
        "group": resolved_input(args.group_manifest, "group manifest"),
        "rooms": resolved_input(args.rooms_json, "T3 rooms output"),
        "pairs": resolved_input(args.pairs_json, "T3 pairs output"),
        "summary": resolved_input(args.group_summary, "T3 group summary"),
        "train": resolved_input(args.train_manifest, "train manifest"),
        "val": resolved_input(args.val_manifest, "val manifest"),
    }
    group = load_json(paths["group"])
    rooms = load_json(paths["rooms"])
    pair_output = load_json(paths["pairs"])
    group_summary = load_json(paths["summary"])
    train = load_json(paths["train"])
    val = load_json(paths["val"])
    if group.get("split") != "train" or train.get("split") != "train" or val.get("split") != "val":
        raise ValueError("fixed group/train/val split contract changed")
    if group_summary.get("split") != "train" or group_summary.get("test_data_accessed") is not False:
        raise ValueError("saved T3 output does not certify train-only access")
    if group.get("house_id") != "0665" or group.get("floor_id") != "floor_03":
        raise ValueError("this audit is fixed to house 0665 floor_03")
    expected_rooms = ["complete_room_04", "complete_room_01", "complete_room_05"]
    if group.get("room_ids") != expected_rooms or set(rooms) != set(expected_rooms):
        raise ValueError("fixed room selection changed")
    pair_predictions = {
        value["prediction"]["pair_id"]: value["prediction"]
        for value in pair_output.get("pairs", [])
    }
    pair_ground_truth = {
        value["prediction"]["pair_id"]: value["ground_truth"]
        for value in pair_output.get("pairs", [])
    }
    wanted = set(group["pair_ids"].values())
    records = {record["id"]: record for record in train.get("pairs", []) if record.get("id") in wanted}
    if set(records) != wanted or set(pair_predictions) != wanted:
        raise ValueError("fixed pair records are incomplete")
    data_root = Path(str(train["dataRoot"])).expanduser().resolve()
    reject_test_data_path(data_root, "ZInD data root")
    annotations = {}
    annotation_payloads: Dict[Path, Mapping[str, Any]] = {}
    for room in expected_rooms:
        reference = group["instance_layout_refs"][room]
        path = (data_root / reference["annotation_path"]).resolve()
        reject_test_data_path(path, "train raw annotation")
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = annotation_payloads.setdefault(path, load_json(path))
        panorama = json_pointer(payload, reference["json_pointer"])
        selection = group["room_to_pano"][room]
        expected_image = (
            f"panos/{group['floor_id']}_{selection['partial_room_id']}_"
            f"{selection['pano_id']}.jpg"
        )
        if panorama["image_path"] != expected_image:
            raise ValueError(f"raw panorama identity changed for {room}")
        annotations[room] = {
            "path": str(path),
            "json_pointer": reference["json_pointer"],
            "panorama": panorama,
        }
        saved = rooms[room]["ground_truth"]
        for layout_name in ("raw", "visible", "complete"):
            raw_name = "layout_" + layout_name
            for field in ("vertices", "doors", "openings", "windows"):
                if saved["layouts"][layout_name].get(field, []) != panorama[raw_name].get(field, []):
                    raise ValueError(
                        f"saved T3 GT differs from raw train annotation: "
                        f"{room}.{raw_name}.{field}"
                    )
        if saved["floor_plan_transformation"] != panorama["floor_plan_transformation"]:
            raise ValueError(f"saved transformation differs from raw train annotation: {room}")
    floor_scale = float(group["scale_meters_per_coordinate"])
    raw_floor_scale = next(iter(annotation_payloads.values()))["scale_meters_per_coordinate"][group["floor_id"]]
    if not math.isclose(floor_scale, float(raw_floor_scale), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("group and raw train floor scales disagree")
    return {
        "paths": paths,
        "group": group,
        "rooms": rooms,
        "pair_predictions": pair_predictions,
        "pair_ground_truth": pair_ground_truth,
        "records": records,
        "annotations": annotations,
        "floor_scale": floor_scale,
    }


def coordinate_roundtrips(audit: Mapping[str, Any]) -> Dict[str, Any]:
    rooms = audit["rooms"]
    output: Dict[str, Any] = {}
    for room, value in rooms.items():
        polygon = np.asarray(value["ground_truth"]["layouts"]["complete"]["vertices"], dtype=np.float64)
        local_to_floor = zind_transform_matrix(value["ground_truth"]["floor_plan_transformation"])
        floor_to_local = np.linalg.inv(local_to_floor)
        restored = apply_transform(apply_transform(polygon, local_to_floor), floor_to_local)
        bi = polygon @ BI_TO_ZIND_2D.T
        zind = bi @ BI_TO_ZIND_2D.T
        output[room] = {
            "local_to_floor": local_to_floor.tolist(),
            "floor_to_local": floor_to_local.tolist(),
            "local_floor_local_max_abs_error": float(np.max(np.abs(restored - polygon))),
            "zind_bi_zind_max_abs_error": float(np.max(np.abs(zind - polygon))),
        }
    pair_checks = {}
    for pair_id, record in audit["records"].items():
        if record["supervision"]["pose_valid"] is not True:
            continue
        room_a, room_b = record["room_a"], record["room_b"]
        expected = relative_zind_transform(
            rooms[room_a]["ground_truth"]["floor_plan_transformation"],
            rooms[room_b]["ground_truth"]["floor_plan_transformation"],
        )
        saved = np.asarray(record["supervision"]["relative_transform_b_to_a"], dtype=np.float64)
        target_a = int(record["supervision"]["target_candidate_a"])
        target_b = int(record["supervision"]["target_candidate_b"])
        endpoints_a = np.asarray(record["candidates_a"][target_a]["localEndpoints"], dtype=np.float64)
        endpoints_b = np.asarray(record["candidates_b"][target_b]["localEndpoints"], dtype=np.float64)
        transformed_b = apply_transform(endpoints_b, saved)
        direct = np.sqrt(np.mean(np.square(transformed_b - endpoints_a)))
        reversed_error = np.sqrt(np.mean(np.square(transformed_b - endpoints_a[::-1])))
        sample = np.concatenate((endpoints_a, endpoints_b), axis=0)
        inverse_roundtrip = apply_transform(apply_transform(sample, saved), np.linalg.inv(saved))
        pair_checks[pair_id] = {
            "direction": "B_to_A",
            "matrix_formula": "inverse(local_to_floor_A) @ local_to_floor_B",
            "saved_vs_recomputed_max_abs_error": float(np.max(np.abs(saved - expected))),
            "T_inverse_T_max_abs_error": float(np.max(np.abs(inverse_roundtrip - sample))),
            "GT_B_endpoint_to_A_RMSE_best_order": float(min(direct, reversed_error)),
        }
    output["pair_B_to_A_checks"] = pair_checks
    return output


def scalar_for_csv(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False)
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return ""
    return value


def write_hypotheses_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: scalar_for_csv(row.get(key)) for key in fields})


def svg_points(points: np.ndarray, projector) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in (projector(point) for point in points))


def render_visualization(
    path: Path,
    title: str,
    row: Mapping[str, Any],
    polygon_a: np.ndarray,
    polygon_b: np.ndarray,
    endpoints_a: np.ndarray,
    target_instances: Sequence[Mapping[str, Any]],
) -> None:
    transform = np.asarray(row["T_B_to_A"], dtype=np.float64)
    predicted_b = apply_transform(polygon_b, transform)
    gt_a = np.asarray(target_instances[0]["polygon"], dtype=np.float64)
    gt_b = np.asarray(target_instances[1]["polygon"], dtype=np.float64)
    arrays = [polygon_a, predicted_b, endpoints_a, gt_a, gt_b]
    all_points = np.concatenate(arrays, axis=0)
    minimum, maximum = all_points.min(axis=0), all_points.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-6)
    scale = 650.0 / float(max(span))
    center = (minimum + maximum) / 2.0

    def project(point: np.ndarray) -> Tuple[float, float]:
        value = (np.asarray(point) - center) * scale
        return float(value[0] + 400.0), float(380.0 - value[1])

    text = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="760" viewBox="0 0 800 760">',
        '<rect width="800" height="760" fill="white"/>',
        f'<text x="24" y="30" font-size="18" font-family="monospace">{title}</text>',
        f'<text x="24" y="55" font-size="14" font-family="monospace">yaw={row["predicted_yaw_degrees"]:.3f} err={row["yaw_error_degrees"]:.3f} scale={row["scale"]:.4f} t_err={row["translation_error"]:.4f}</text>',
        f'<polygon points="{svg_points(gt_a, project)}" fill="none" stroke="#2ca02c" stroke-width="2" stroke-dasharray="7,5"/>',
        f'<polygon points="{svg_points(gt_b, project)}" fill="none" stroke="#2ca02c" stroke-width="2" stroke-dasharray="7,5"/>',
        f'<polygon points="{svg_points(polygon_a, project)}" fill="#1f77b4" fill-opacity="0.18" stroke="#1f77b4" stroke-width="3"/>',
        f'<polygon points="{svg_points(predicted_b, project)}" fill="#ff7f0e" fill-opacity="0.18" stroke="#ff7f0e" stroke-width="3"/>',
        f'<polyline points="{svg_points(endpoints_a, project)}" fill="none" stroke="#d62728" stroke-width="7" stroke-linecap="round"/>',
        '<text x="24" y="725" font-size="13" font-family="monospace" fill="#1f77b4">A predicted/source</text>',
        '<text x="220" y="725" font-size="13" font-family="monospace" fill="#ff7f0e">B transformed</text>',
        '<text x="400" y="725" font-size="13" font-family="monospace" fill="#2ca02c">GT dashed</text>',
        '<text x="560" y="725" font-size="13" font-family="monospace" fill="#d62728">portal</text>',
        '</svg>',
    ]
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def coordinate_contract_markdown(audit: Mapping[str, Any], roundtrips: Mapping[str, Any]) -> str:
    paths = audit["paths"]
    lines = [
        "# ZInD geometry oracle coordinate contract",
        "",
        "This document describes the saved T3 prediction and train-only GT used by the CPU audit. Oracle-only fields are never valid inference inputs.",
        "",
        "| Field | Source | Coordinate system / units | Axes and positive angle | Manhattan canonicalization | Direction / inference use |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| Predicted layout polygon | `{paths['rooms']} -> room.single_room_layout.predicted_layout_bi_xz` and mirrored `predicted_polygon_zind_local_xy` | Bi horizontal x/z model-layout units; exported value is labeled ZInD panorama-local x/y | Bi longitude uses `atan2(x,z)`; export mirrors x and uses CCW `atan2(y,x)` | **Yes.** `prediction_to_layout` calls fixed-axis `post_process(..., manhattan)` | Saved prediction; allowed for C0 formal path |",
        f"| Predicted opening interval | `{paths['rooms']} -> room.opening_prediction.intervals/candidates.tokenStart,tokenEnd` | Inclusive circular panorama token indices, N=256 | Panorama u increases right; longitude `(u-0.5)*2pi` | No, but its later wall projection assumes the polygon retains panorama heading | Saved prediction; allowed for C0 |",
        "| Interval-mapped endpoints | `opening_candidates_from_intervals` then `door_endpoints` | Same units/frame as selected polygon wall | Directed wall ratio start->end | Inherits polygon frame | C0 is inference-valid; C1-C3 are oracle substitutions |",
        f"| GT layout_raw/visible/complete | raw train `zind_data.json` via group JSON pointer; mirrored copy at `{paths['rooms']} -> ground_truth.layouts` | ZInD panorama-local x/y, annotation coordinate units | Standard XY, positive rotation CCW | No audit canonicalization; walls retain arbitrary pano heading | Post-inference evaluation/oracle only |",
        f"| GT local opening endpoints | `{paths['train']} -> candidates_a/b.localEndpoints` | ZInD panorama-local x/y, annotation coordinate units | Endpoint order is annotation order | No | Oracle only in C4 |",
        f"| GT global opening endpoints | `{paths['train']} -> candidates_a/b.globalEndpoints` | Floor/global x/y, floor-coordinate units | Standard XY, CCW | No | Generated by local-to-floor; oracle/evaluation only |",
        "| floor_plan_transformation | raw train annotation | `p_floor = scale * R(rotation) * p_local + translation` | rotation is degrees CCW; translation is floor/global x/y | No | local->floor; GT only |",
        "| Manhattan canonical frame | `portallayout/postprocessing/dula/layout.py::fit_layout` | Axis-aligned Bi x/z model-layout units | walls are forced to 0/90-degree axes | This is the canonicalization under audit; no heading is saved | Not a declared ZInD annotation frame |",
        "| Panorama-local XY | ZInD layout vertices and localEndpoints | camera-centered annotation units | positive yaw CCW | No | Per-panorama frame |",
        "| Floor/global XY | ZInD transformed vertices/endpoints | floor-coordinate units | positive yaw CCW | No | Common floor frame |",
        f"| T_B_to_A | `{paths['train']} -> supervision.relative_transform_b_to_a` | ZInD panorama-A local x/y | `p_A = T_B_to_A p_B`, yaw CCW | No | Always B -> A; GT oracle/evaluation only |",
        "| relative_scale | Derived as `hypot(T[0,0], T[1,0])` from the manifest transform | Dimensionless B-local to A-local similarity scale | Positive scalar | No | Label follows panorama-transform scale ratios; it is not stored as a separate field |",
        "| translation_meters | Derived during evaluation | meters | same A-local axes as translation | No | `translation * pano_A.floor_plan_transformation.scale * scale_meters_per_coordinate` |",
        f"| scale_meters_per_coordinate | group/raw floor label, value `{audit['floor_scale']}` | meters per floor coordinate | scalar | No | Metric conversion only |",
        "",
        "## Observed fixed-frame transformations",
        "",
    ]
    for room, report in roundtrips.items():
        if room == "pair_B_to_A_checks":
            continue
        transform = audit["rooms"][room]["ground_truth"]["floor_plan_transformation"]
        lines.append(
            f"- `{room}` local->floor: rotation={transform['rotation']:.9f} deg, scale={transform['scale']:.12f}, translation={transform['translation']}; local/floor/local error={report['local_floor_local_max_abs_error']:.3e}."
        )
    lines.extend(
        [
            "",
            "The saved predicted polygon is exported under a panorama-local name, but its only retained geometry has already been snapped to fixed Manhattan axes. No pre-Manhattan polygon, raw predicted depth, or canonical-to-panorama heading is present in the saved T3 JSON. Therefore frame restoration from these files is necessarily marked oracle-only.",
            "",
            "For a matrix `C_i` defined as canonical -> panorama-local, the checked composition is `T_local_B_to_A = C_A @ T_canonical_B_to_A @ inverse(C_B)`. This direction is covered by the CPU test and all matrix round-trips are reported in `audit_summary.json`.",
        ]
    )
    return "\n".join(lines) + "\n"


def diagnosis_markdown(summary: Mapping[str, Any]) -> str:
    key = summary["key_results"]
    stats = summary["scale_label_statistics"]
    pair_ids = list(key)
    first, second = key[pair_ids[0]], key[pair_ids[1]]

    def metric(result: Mapping[str, Any], name: str) -> str:
        return f"{float(result[name]):.6f}"

    lines = [
        "# T3 three-room geometry oracle diagnosis",
        "",
        "## Executive conclusion",
        "",
        "The frozen matcher/opening selection is not the failure source. Both edges recover the exact GT yaw when the same predicted interval is projected onto the GT panorama-local polygon (C2). The first irreversible error is the fixed-axis Manhattan polygon being treated as if it still retained each panorama's local heading; interval-to-wall projection then attaches physical doors to incorrect canonical walls. Door-width ratio has a valid Sim(2) interpretation for ideal geometry, but the raw predicted widths are unstable and amplify translation errors; rotation/frame loss must be repaired first.",
        "",
        "## Key numeric chain",
        "",
        "| Pair | C0 yaw err | C1 yaw err | C2 yaw err | C3 yaw err | C4 yaw err | C0 scale | C4 width scale |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pair_id, values in key.items():
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                pair_id,
                metric(values["C0_current_width_ratio"], "yaw_error_degrees"),
                metric(values["C1_current_width_ratio"], "yaw_error_degrees"),
                metric(values["C2_current_width_ratio"], "yaw_error_degrees"),
                metric(values["C3_current_width_ratio"], "yaw_error_degrees"),
                metric(values["C4_current_width_ratio"], "yaw_error_degrees"),
                metric(values["C0_current_width_ratio"], "scale"),
                metric(values["C4_current_width_ratio"], "scale"),
            )
        )
    lines.extend(
        [
            "",
            "## Required questions",
            "",
            "1. **Can GT polygon + GT endpoints recover GT pose with the current formula?** It recovers GT yaw to numerical precision and translation within a small annotation-level residual. Width-ratio scale is not exact because the two independently annotated door widths differ; C4 scales are {:.6f} and {:.6f} although GT scale is 1.".format(
                first["C4_current_width_ratio"]["scale"], second["C4_current_width_ratio"]["scale"]
            ),
            "2. **Where does yaw first become a multiple of 90 degrees?** In `prediction_to_layout -> post_process(type_name='manhattan') -> fit_layout`, before candidate geometry is built. The mapper and joint layout preserve/consume those axis-aligned wall tangents; they do not create the lost arbitrary heading.",
            "3. **Was per-panorama canonical-to-local heading lost?** Yes in the retained geometry contract. T3 JSON contains only the post-Manhattan layout and no pre-Manhattan polygon/raw depth heading. The train GT rotations are arbitrary and differ per panorama.",
            "4. **Where is the loss?** Primarily Manhattan postprocessing/export. It becomes observable in candidate mapping because the mapper assumes that polygon rays are panorama-local. Joint layout then deterministically fits the already-wrong endpoints.",
            "5. **Does GT interval through the current mapper recover the opening?** On the GT polygon (C3), yes: both yaw errors are numerical zero and translation residuals are small. On the predicted polygon (C1), no; one small wrap-interval change even switches the dominant wall.",
            "6. **Does predicted interval remain wrong on the GT polygon?** No. C2 recovers exact GT yaw on both edges, so the frozen predicted intervals are sufficient for these portals.",
            "7. **Does predicted polygon remain wrong with GT interval?** Yes. C1 remains wrong, demonstrating that interval supervision alone cannot repair the canonicalized polygon/wall association.",
            "8. **Is unit scale better than width ratio?** For this fixed group it always eliminates scale-label error because GT scale is 1. For C0 it strongly improves translation on the 01-05 edge but cannot repair rotation/frame errors; for C2-C4 width fitting can slightly reduce translation by absorbing endpoint-annotation mismatch. Across the full manifest, width ratio is theoretically consistent with the rare non-unit Sim(2) labels, so unit scale is an oracle for this group rather than a universal replacement.",
            "9. **Is train/val relative scale essentially one?** Statistically yes but contractually not always. {}/{} pose-valid positives are within 1e-9 of 1 (ratio {:.6f}); {} are non-unit, with range {:.6f}--{:.6f} and |log(scale)| max {:.3e}. The manifest is therefore Sim(2), overwhelmingly concentrated on an SE(2) mode.".format(
                int(stats["relative_scale"]["count"] * stats["relative_scale_within_1e-9_of_one_ratio"]),
                stats["relative_scale"]["count"],
                stats["relative_scale_within_1e-9_of_one_ratio"],
                stats["relative_scale_outside_1e-9_of_one_count"],
                stats["non_unit_relative_scale"]["min"],
                stats["non_unit_relative_scale"]["max"],
                stats["absolute_log_relative_scale"]["max"],
            ),
            "10. **What dominates translation error?** C0 translation inherits both wrong rotation/wall placement and extreme width scale. C2 reduces translation to a small residual without changing the predicted interval, proving polygon frame/wall geometry is primary; unit scale is the second correction. Opening midpoint fitting itself is exact by construction and is not the primary source.",
            "11. **Where should the minimal formal repair live?** Preserve or restore panorama heading around `tools/debug_cross_scene_flow.py::prediction_to_layout` (or its `post_process/fit_layout` helper) so `opening_candidates_from_intervals` receives a true panorama-local polygon. For this fixed unit-scale group, `calibrate_scale=False` is a valid diagnostic baseline. A global change in `portallayout/geometry/joint_layout.py::_fit_similarity` must instead retain the rare non-unit Sim(2) labels through a robust/clamped estimator or an explicitly supervised scale source. Do not patch candidate ranks or add a verifier.",
            "12. **Does this require retraining?** The demonstrated frame/wall failure is deterministic parsing geometry and does not require retraining. A heading-preserving postprocess and fixed-group unit-scale validation should come first. Only a learned replacement for the full-dataset scale estimator would require additional training.",
            "",
            "## Important wall-association evidence",
            "",
            "For room01, the two labeled shared doors lie on the same GT complete-layout wall family. In C0 they map to predicted wall 2 and predicted wall 8, which are opposite parallel walls. A single global heading cannot repair that wall-side swap; the panorama-local polygon/ray contract must be corrected before fusion.",
            "",
            "All oracle rows have `oracle_only=true` and `valid_for_inference=false`. Only the exact C0/current/width-ratio row selected by the existing centroid-side rule is marked as the frozen formal path.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_audit(args: argparse.Namespace) -> Dict[str, Any]:
    audit = load_inputs(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        marker = output_dir / "audit_summary.json"
        existing = load_json(marker) if marker.is_file() else {}
        if existing.get("schema_version") != "zind-joint-geometry-oracle-audit-v1":
            raise FileExistsError(f"refusing to overwrite foreign audit output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = output_dir / "visualizations"
    visual_dir.mkdir(exist_ok=True)

    rows: List[Dict[str, Any]] = []
    source_reports: Dict[str, Any] = {}
    pair_case_frames: Dict[Tuple[str, str, str], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    negative_checks = []
    positive_ids = []
    for pair_id in audit["group"]["pair_ids"].values():
        prediction = audit["pair_predictions"][pair_id]
        record = audit["records"][pair_id]
        if record["supervision"]["is_match"] is not True:
            negative_checks.append(audit_negative_prediction(prediction))
            continue
        if prediction.get("is_match") is not True or prediction.get("pose_valid") is not True:
            raise ValueError(f"saved positive edge is not accepted: {pair_id}")
        positive_ids.append(pair_id)
        room_a, room_b = str(record["room_a"]), str(record["room_b"])
        if prediction["room_a"] != room_a or prediction["room_b"] != room_b:
            raise ValueError(f"saved A/B direction changed: {pair_id}")
        cases = build_endpoint_cases(prediction, record, audit["rooms"])
        gt_transform = np.asarray(record["supervision"]["relative_transform_b_to_a"], dtype=np.float64)
        gt_public = audit["pair_ground_truth"][pair_id]
        target_instances = gt_public["joint_bev_target"]["instances"]
        pano_a_transform = audit["rooms"][room_a]["ground_truth"]["floor_plan_transformation"]
        metric_factor = float(pano_a_transform["scale"]) * audit["floor_scale"]
        source_reports[pair_id] = {}
        for case_name in CASES:
            case = cases[case_name]
            source_reports[pair_id][case_name] = {
                "oracle_only": case.oracle_only,
                "valid_for_inference": not case.oracle_only,
                "side_a": {
                    **case.source_a,
                    "diagnostics": portal_diagnostics(
                        case.polygon_a, case.endpoints_a, case.source_a.get("wall_endpoints")
                    ),
                },
                "side_b": {
                    **case.source_b,
                    "diagnostics": portal_diagnostics(
                        case.polygon_b, case.endpoints_b, case.source_b.get("wall_endpoints")
                    ),
                },
            }
            for frame_mode in FRAME_MODES:
                polygon_a, polygon_b, endpoints_a, endpoints_b, frame_report = transform_case_frame(
                    case, frame_mode, prediction, record, audit["rooms"]
                )
                source_reports[pair_id][case_name].setdefault("frames", {})[frame_mode] = frame_report
                pair_case_frames[(pair_id, case_name, frame_mode)] = (
                    polygon_a,
                    polygon_b,
                    endpoints_a,
                    endpoints_b,
                )
                for scale_mode in SCALE_MODES:
                    for mapping in ENDPOINT_MAPPINGS:
                        rows.append(
                            evaluate_hypothesis(
                                pair_id,
                                room_a,
                                room_b,
                                case,
                                frame_mode,
                                scale_mode,
                                mapping,
                                polygon_a,
                                polygon_b,
                                endpoints_a,
                                endpoints_b,
                                gt_transform,
                                metric_factor,
                                target_instances,
                                frame_report,
                            )
                        )
    if len(positive_ids) != 2 or len(negative_checks) != 1:
        raise ValueError("fixed audit requires two positives and one negative")
    select_centroid_hypotheses(rows)

    # Confirm C0 exactly reproduces the frozen formal transform rather than an oracle variant.
    formal_reproduction = {}
    for pair_id in positive_ids:
        row = selected_row(rows, pair_id, "C0")
        saved = np.asarray(audit["pair_predictions"][pair_id]["relative_pose"]["T_B_to_A"])
        reproduced = np.asarray(row["T_B_to_A"])
        difference = float(np.max(np.abs(saved - reproduced)))
        if difference > 1e-8:
            raise RuntimeError(f"C0 failed to reproduce saved T3 transform: {pair_id} ({difference})")
        formal_reproduction[pair_id] = {"max_abs_matrix_error": difference}

    stats = scale_label_statistics([audit["paths"]["train"], audit["paths"]["val"]])
    roundtrips = coordinate_roundtrips(audit)
    key_results: Dict[str, Dict[str, Any]] = {}
    for pair_id in positive_ids:
        key_results[pair_id] = {}
        for case in CASES:
            key_results[pair_id][f"{case}_current_width_ratio"] = dict(
                selected_row(rows, pair_id, case, "current", "width_ratio")
            )
            key_results[pair_id][f"{case}_current_unit"] = dict(
                selected_row(rows, pair_id, case, "current", "unit")
            )
        key_results[pair_id]["C0_frame_restored_width_ratio"] = dict(
            selected_row(rows, pair_id, "C0", "frame_restored", "width_ratio")
        )
        key_results[pair_id]["C0_frame_restored_unit"] = dict(
            selected_row(rows, pair_id, "C0", "frame_restored", "unit")
        )
        key_results[pair_id]["C4_frame_restored_width_ratio"] = dict(
            selected_row(rows, pair_id, "C4", "frame_restored", "width_ratio")
        )
        key_results[pair_id]["C4_frame_restored_unit"] = dict(
            selected_row(rows, pair_id, "C4", "frame_restored", "unit")
        )

    summary: Dict[str, Any] = {
        "schema_version": "zind-joint-geometry-oracle-audit-v1",
        "purpose": "debug/oracle diagnosis only, not inference or scientific evaluation",
        "oracle_only": True,
        "valid_for_inference": False,
        "gpu_used": False,
        "model_forward_called": False,
        "test_data_accessed": False,
        "split": "train",
        "house_id": audit["group"]["house_id"],
        "floor_id": audit["group"]["floor_id"],
        "room_ids": audit["group"]["room_ids"],
        "positive_pair_ids": positive_ids,
        "negative_checks": negative_checks,
        "hypothesis_count": len(rows),
        "formal_C0_reproduction": formal_reproduction,
        "coordinate_roundtrips": roundtrips,
        "endpoint_sources": source_reports,
        "scale_label_statistics": stats,
        "key_results": key_results,
        "diagnosis": {
            "primary": "fixed-axis Manhattan predicted polygon is consumed as panorama-local geometry",
            "secondary": "raw door-width ratio is Sim(2)-consistent in theory but unstable under predicted wall geometry",
            "candidate_selection_failure": False,
            "requires_retraining": False,
            "recommended_next_step": "repair panorama-local heading first; use unit scale only for this fixed group, then design a robust full-dataset Sim(2) scale estimator",
        },
    }
    if not finite_tree(summary) or not all(finite_tree(row) for row in rows):
        raise RuntimeError("audit generated a non-finite value")

    write_hypotheses_csv(output_dir / "audit_hypotheses.csv", rows)
    dump_json(output_dir / "audit_summary.json", summary)
    (output_dir / "coordinate_contract.md").write_text(
        coordinate_contract_markdown(audit, roundtrips), encoding="utf-8"
    )
    (output_dir / "diagnosis.md").write_text(diagnosis_markdown(summary), encoding="utf-8")

    visual_specs = [
        ("C0", "current", "width_ratio", "C0_current"),
        ("C0", "frame_restored", "width_ratio", "C0_frame_restored"),
        ("C1", "current", "width_ratio", "C1_current"),
        ("C2", "current", "width_ratio", "C2_current"),
        ("C3", "current", "width_ratio", "C3_current"),
        ("C4", "current", "width_ratio", "C4_current_formula"),
        ("C4", "frame_restored", "width_ratio", "C4_frame_restored"),
        ("C4", "frame_restored", "unit", "C4_frame_restored_unit"),
    ]
    for pair_id in positive_ids:
        target_instances = audit["pair_ground_truth"][pair_id]["joint_bev_target"]["instances"]
        for case, frame_mode, scale_mode, suffix in visual_specs:
            row = selected_row(rows, pair_id, case, frame_mode, scale_mode)
            polygon_a, polygon_b, endpoints_a, _ = pair_case_frames[(pair_id, case, frame_mode)]
            render_visualization(
                visual_dir / f"{pair_id}_{suffix}.svg",
                f"{pair_id} {case}/{frame_mode}/{scale_mode}",
                row,
                polygon_a,
                polygon_b,
                endpoints_a,
                target_instances,
            )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = run_audit(args)
    print(
        json.dumps(
            {
                "schema_version": summary["schema_version"],
                "output_dir": str(Path(args.output_dir).expanduser().resolve()),
                "hypothesis_count": summary["hypothesis_count"],
                "gpu_used": summary["gpu_used"],
                "model_forward_called": summary["model_forward_called"],
                "test_data_accessed": summary["test_data_accessed"],
                "diagnosis": summary["diagnosis"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
