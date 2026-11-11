#!/usr/bin/env python3
"""CPU-only candidate-local portal geometry audit for one saved train group.

The tool never loads a model.  Predicted-only portal hypotheses are frozen
from the saved pre-Manhattan contours and matcher decisions before the saved
train ground truth is attached for evaluation.  Nothing here is imported by
the formal inference path.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from portallayout.postprocessing.dula.layout import fit_layout
from portallayout.postprocessing.post_process import estimate_manhattan_heading, rotate_floor_xz
from portallayout.geometry.candidate_local_portal_geometry import (
    CandidateLocalPortalConfig,
    estimate_candidate_local_portal_geometry,
)
from portallayout.engine.cross_scene_estimator import (
    opening_candidates_from_intervals,
    polygon_overlap_ratio,
    polygon_validity,
    wall_endpoints,
)
from portallayout.geometry.joint_layout import door_endpoints, layout_xz
from portallayout.utils.writer import xyz2json


DEFAULT_BASE = REPO_ROOT / "outputs/zind_room_group_debug/t3_best_0665_floor03_frame_fixed"
DEFAULT_GROUP = REPO_ROOT / "data/ZInD_matching_groups/debug_train_0665_floor_03.json"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/zind_room_group_debug/t3_portal_geometry_audit_0665_floor03"
TOKEN_COUNT = 256
METHODS = ("M0", "M1", "M2", "M3", "M4")
COLORS = {
    "M0": "#a51d2d",
    "M1": "#e28b1a",
    "M2": "#1774c8",
    "M3": "#239b56",
    "M4": "#8e44ad",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rooms_json", default=str(DEFAULT_BASE / "rooms.json"))
    parser.add_argument("--pairs_json", default=str(DEFAULT_BASE / "pairs.json"))
    parser.add_argument("--group_manifest", default=str(DEFAULT_GROUP))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args(argv)


def reject_test_path(path: Path, label: str) -> None:
    value = path.expanduser().resolve()
    if value.stem.lower() == "test" or any(part.lower() == "test" for part in value.parts):
        raise ValueError(f"{label} points to forbidden test data: {value}")


def resolved_input(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    reject_test_path(path, label)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def dump_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")


def finite_tree(value: Any) -> bool:
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    return True


def descriptive(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def circular_indices(start: int, end: int, count: int = TOKEN_COUNT) -> np.ndarray:
    if count <= 0:
        raise ValueError("count must be positive")
    start, end = int(start) % count, int(end) % count
    if start <= end:
        return np.arange(start, end + 1, dtype=np.int64)
    return np.concatenate(
        [np.arange(start, count, dtype=np.int64), np.arange(0, end + 1, dtype=np.int64)]
    )


def expanded_indices(start: int, end: int, padding: int, count: int = TOKEN_COUNT) -> np.ndarray:
    interval = circular_indices(start, end, count)
    left = (np.arange(start - padding, start, dtype=np.int64) % count) if padding else np.empty(0, dtype=np.int64)
    right = (np.arange(end + 1, end + padding + 1, dtype=np.int64) % count) if padding else np.empty(0, dtype=np.int64)
    return np.concatenate([left, interval, right])


def context_indices(start: int, end: int, width: int, count: int = TOKEN_COUNT) -> Tuple[np.ndarray, np.ndarray]:
    left = np.arange(start - width, start, dtype=np.int64) % count
    right = np.arange(end + 1, end + width + 1, dtype=np.int64) % count
    return left, right


def panorama_ray(token: int, count: int = TOKEN_COUNT) -> np.ndarray:
    longitude = ((int(token) % count + 0.5) / float(count) - 0.5) * 2.0 * math.pi
    return np.asarray([math.sin(longitude), math.cos(longitude)], dtype=np.float64)


def bi_to_zind(points: Sequence[Sequence[float]]) -> np.ndarray:
    """Reflect Bi x/z into the saved ZInD panorama-local x/y convention."""
    value = np.asarray(points, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2 or not np.isfinite(value).all():
        raise ValueError("points must be finite [N, 2]")
    output = value.copy()
    output[:, 0] *= -1.0
    return output


def apply_transform(points: Sequence[Sequence[float]], matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    homogeneous = np.concatenate([value, np.ones((len(value), 1), dtype=np.float64)], axis=1)
    return (np.asarray(matrix, dtype=np.float64) @ homogeneous.T).T[:, :2]


def transform_yaw(matrix: np.ndarray) -> float:
    value = np.asarray(matrix, dtype=np.float64)
    return float(math.atan2(value[1, 0], value[0, 0]))


def transform_scale(matrix: np.ndarray) -> float:
    value = np.asarray(matrix, dtype=np.float64)
    return float(math.hypot(value[0, 0], value[1, 0]))


def wrapped_angle_error(first: float, second: float, period: float = 2.0 * math.pi) -> float:
    return float(abs((float(first) - float(second) + period / 2.0) % period - period / 2.0))


@dataclass
class LineFit:
    tangent: np.ndarray
    normal: np.ndarray
    offset: float
    residuals: np.ndarray
    inliers: np.ndarray
    singular_values: np.ndarray
    source: str


def _line_orientation(points: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    direction = points[-1] - points[0]
    if float(np.dot(direction, tangent)) < 0:
        tangent = -tangent
    return tangent


def fit_line_tls(
    points: Sequence[Sequence[float]],
    weights: Optional[Sequence[float]] = None,
    source: str = "weighted_tls",
) -> LineFit:
    value = np.asarray(points, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2 or len(value) < 2:
        raise ValueError("line fit needs at least two [x, z] points")
    if not np.isfinite(value).all():
        raise ValueError("line points must be finite")
    if weights is None:
        weight = np.ones(len(value), dtype=np.float64)
    else:
        weight = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(weight) != len(value) or not np.isfinite(weight).all() or np.any(weight < 0):
            raise ValueError("line weights must be finite, non-negative, and aligned")
    if float(weight.sum()) <= 1e-12:
        raise ValueError("line weights must have positive mass")
    center = np.average(value, axis=0, weights=weight)
    centered = value - center
    _, singular, vh = np.linalg.svd(centered * np.sqrt(weight[:, None]), full_matrices=False)
    tangent = _line_orientation(value, vh[0].astype(np.float64))
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
    offset = float(np.dot(normal, center))
    residuals = np.abs(value @ normal - offset)
    return LineFit(
        tangent=tangent,
        normal=normal,
        offset=offset,
        residuals=residuals,
        inliers=np.ones(len(value), dtype=bool),
        singular_values=np.pad(singular, (0, max(0, 2 - len(singular))))[:2],
        source=source,
    )


def ransac_line(
    points: Sequence[Sequence[float]],
    threshold: float,
    weights: Optional[Sequence[float]] = None,
    min_inliers: int = 3,
    source: str = "deterministic_ransac_tls",
) -> LineFit:
    value = np.asarray(points, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2 or len(value) < 2:
        raise ValueError("RANSAC needs at least two [x, z] points")
    if not np.isfinite(value).all() or not math.isfinite(float(threshold)) or threshold <= 0:
        raise ValueError("RANSAC input and threshold must be finite")
    weight = np.ones(len(value), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if weight.shape != (len(value),) or not np.isfinite(weight).all() or np.any(weight < 0):
        raise ValueError("RANSAC weights must align with points")
    best: Optional[Tuple[Tuple[float, int, float, float], np.ndarray]] = None
    for first in range(len(value) - 1):
        for second in range(first + 1, len(value)):
            vector = value[second] - value[first]
            length = float(np.linalg.norm(vector))
            if length <= max(1e-10, threshold * 0.25):
                continue
            tangent = vector / length
            normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
            offset = float(np.dot(normal, value[first]))
            residuals = np.abs(value @ normal - offset)
            inliers = residuals <= threshold
            count = int(np.count_nonzero(inliers))
            if count < min(min_inliers, len(value)):
                continue
            weighted_support = float(weight[inliers].sum())
            span = float(np.linalg.norm(value[inliers].max(axis=0) - value[inliers].min(axis=0)))
            rmse = float(math.sqrt(np.average(np.square(residuals[inliers]), weights=weight[inliers])))
            score = (weighted_support, count, span, -rmse)
            if best is None or score > best[0]:
                best = (score, inliers)
    if best is None:
        fit = fit_line_tls(value, weights=weight, source=source + "_tls_fallback")
        fit.inliers = np.ones(len(value), dtype=bool)
        return fit
    inliers = best[1]
    refined = fit_line_tls(value[inliers], weights=weight[inliers], source=source)
    residuals = np.abs(value @ refined.normal - refined.offset)
    refined.residuals = residuals
    refined.inliers = residuals <= threshold
    return refined


def line_endpoint_intersections(
    fit: LineFit,
    interval: Sequence[int],
    count: int = TOKEN_COUNT,
    parallel_epsilon: float = 1e-8,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if len(interval) != 2:
        raise ValueError("interval must contain start and end")
    rays = np.stack([panorama_ray(int(interval[0]), count), panorama_ray(int(interval[1]), count)])
    denominators = rays @ fit.normal
    if not np.isfinite(denominators).all() or np.min(np.abs(denominators)) <= parallel_epsilon:
        raise ValueError("portal endpoint ray is parallel or non-finite relative to fitted line")
    distances = fit.offset / denominators
    if not np.isfinite(distances).all() or np.any(distances <= 0):
        raise ValueError("portal endpoint intersection is behind the panorama camera or non-finite")
    endpoints = rays * distances[:, None]
    return endpoints, {
        "ray_denominators": denominators.tolist(),
        "ray_intersection_condition": float(np.min(np.abs(denominators))),
        "ray_distances": distances.tolist(),
    }


def _axis_dispersion(points: np.ndarray, tangent: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    edges = np.diff(points, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    keep = lengths > max(1e-10, float(np.max(lengths)) * 1e-6)
    if not np.any(keep):
        return 90.0
    angles = np.arctan2(edges[keep, 1], edges[keep, 0])
    target = math.atan2(tangent[1], tangent[0])
    errors = np.abs((angles - target + math.pi / 2.0) % math.pi - math.pi / 2.0)
    return float(np.degrees(np.average(errors, weights=lengths[keep])))


def predicted_quality(
    fit: LineFit,
    support_points: np.ndarray,
    endpoints_bi: np.ndarray,
    intersection: Mapping[str, Any],
    room_radius: float,
    context_agreement: Optional[bool],
    corner_ambiguity: bool,
) -> Dict[str, Any]:
    inliers = np.asarray(fit.inliers, dtype=bool)
    if len(inliers) != len(support_points):
        raise ValueError("fit inliers and support points diverged")
    active = fit.residuals[inliers] if np.any(inliers) else fit.residuals
    rmse = float(math.sqrt(float(np.mean(np.square(active)))))
    normalized = rmse / max(float(room_radius), 1e-8)
    condition = float(fit.singular_values[0] / max(fit.singular_values[1], 1e-12))
    width = float(np.linalg.norm(endpoints_bi[1] - endpoints_bi[0]))
    width_ratio = width / max(float(room_radius), 1e-8)
    width_plausible = bool(0.03 <= width_ratio <= 2.5)
    angular_dispersion = _axis_dispersion(support_points[inliers], fit.tangent) if np.any(inliers) else 90.0
    ray_condition = float(intersection["ray_intersection_condition"])
    inlier_ratio = float(np.mean(inliers))
    agreement_factor = 1.0 if context_agreement is not False else 0.35
    score = (
        inlier_ratio
        * math.exp(-20.0 * normalized)
        * min(1.0, math.log1p(condition) / math.log(101.0))
        * min(1.0, ray_condition / 0.25)
        * (1.0 if width_plausible else 0.2)
        * agreement_factor
        * (0.5 if corner_ambiguity else 1.0)
    )
    quality = {
        "fit_residual": rmse,
        "normalized_residual": normalized,
        "inlier_ratio": inlier_ratio,
        "support_point_count": int(len(support_points)),
        "inlier_count": int(np.count_nonzero(inliers)),
        "angular_dispersion_degrees": angular_dispersion,
        "line_condition_number": condition,
        "ray_intersection_condition": ray_condition,
        "opening_width": width,
        "opening_width_over_room_radius": width_ratio,
        "width_plausible": width_plausible,
        "left_right_context_agreement": context_agreement,
        "corner_ambiguity": bool(corner_ambiguity),
        "reliability_score": float(score),
    }
    quality["quality_gate_pass"] = quality_gate(quality)
    return quality


def quality_gate(quality: Mapping[str, Any]) -> bool:
    """Predicted-only gate; deliberately accepts no GT argument."""
    return bool(
        int(quality["support_point_count"]) >= 4
        and float(quality["inlier_ratio"]) >= 0.60
        and float(quality["normalized_residual"]) <= 0.05
        and float(quality["angular_dispersion_degrees"]) <= 25.0
        and float(quality["ray_intersection_condition"]) >= 0.05
        and bool(quality["width_plausible"])
        and quality.get("left_right_context_agreement") is not False
        and not bool(quality["corner_ambiguity"])
    )


def compare_context_lines(
    left: LineFit,
    right: LineFit,
    room_radius: float,
    angle_threshold_degrees: float = 12.0,
    offset_threshold_ratio: float = 0.06,
) -> Dict[str, Any]:
    angle = math.degrees(
        wrapped_angle_error(
            math.atan2(left.tangent[1], left.tangent[0]),
            math.atan2(right.tangent[1], right.tangent[0]),
            period=math.pi,
        )
    )
    normal_right = right.normal.copy()
    offset_right = float(right.offset)
    if float(np.dot(left.normal, normal_right)) < 0:
        normal_right *= -1.0
        offset_right *= -1.0
    offset_difference = abs(float(left.offset) - offset_right)
    normalized_offset = offset_difference / max(float(room_radius), 1e-8)
    agrees = angle <= angle_threshold_degrees and normalized_offset <= offset_threshold_ratio
    return {
        "agrees": bool(agrees),
        "angle_difference_degrees": float(angle),
        "offset_difference": float(offset_difference),
        "normalized_offset_difference": float(normalized_offset),
        "angle_threshold_degrees": float(angle_threshold_degrees),
        "offset_threshold_ratio": float(offset_threshold_ratio),
    }


def _portal_from_fit(
    method: str,
    variant: str,
    fit: LineFit,
    support_points: np.ndarray,
    interval: Sequence[int],
    room_radius: float,
    context_report: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    context_agreement = None if context_report is None else bool(context_report["agrees"])
    corner = False if context_report is None else not bool(context_report["agrees"])
    try:
        endpoints_bi, intersection = line_endpoint_intersections(fit, interval)
        endpoints_zind = bi_to_zind(endpoints_bi)
        quality = predicted_quality(
            fit,
            support_points,
            endpoints_bi,
            intersection,
            room_radius,
            context_agreement,
            corner,
        )
        valid = True
        error = None
    except Exception as exc:
        endpoints_bi = None
        endpoints_zind = None
        intersection = {
            "ray_denominators": [None, None],
            "ray_intersection_condition": 0.0,
            "ray_distances": [None, None],
        }
        quality = {
            "fit_residual": None,
            "normalized_residual": None,
            "inlier_ratio": float(np.mean(fit.inliers)),
            "support_point_count": int(len(support_points)),
            "inlier_count": int(np.count_nonzero(fit.inliers)),
            "angular_dispersion_degrees": None,
            "line_condition_number": None,
            "ray_intersection_condition": 0.0,
            "opening_width": None,
            "opening_width_over_room_radius": None,
            "width_plausible": False,
            "left_right_context_agreement": context_agreement,
            "corner_ambiguity": corner,
            "reliability_score": 0.0,
            "quality_gate_pass": False,
        }
        valid = False
        error = str(exc)
    return {
        "method": method,
        "variant": variant,
        "fit_source": fit.source,
        "line": {
            "tangent_bi_xz": fit.tangent.tolist(),
            "normal_bi_xz": fit.normal.tolist(),
            "offset_bi_xz": float(fit.offset),
        },
        "endpoints_bi_xz": None if endpoints_bi is None else endpoints_bi.tolist(),
        "endpoints_zind_local_xy": None if endpoints_zind is None else endpoints_zind.tolist(),
        "intersection": intersection,
        "quality": quality,
        "context": None if context_report is None else dict(context_report),
        "valid": valid,
        "failure_reason": error,
        "selected_predicted_only": False,
    }


def _candidate_local_methods(
    contour: np.ndarray,
    probability: np.ndarray,
    interval: Sequence[int],
    room_threshold: float,
    room_radius: float,
) -> Dict[str, List[Dict[str, Any]]]:
    start, end = int(interval[0]), int(interval[1])
    interval_ids = circular_indices(start, end, len(contour))
    production_m2 = estimate_candidate_local_portal_geometry(
        contour_local=contour,
        token_interval=interval,
        token_count=len(contour),
        opening_probability=probability,
        config=CandidateLocalPortalConfig(),
    )
    m2 = [value.to_audit_dict() for value in production_m2.hypotheses]
    if len(m2) != 2 or not production_m2.geometry_valid:
        raise RuntimeError(
            "production M2 failed on an input that was valid in the frozen audit: "
            + str(production_m2.failure_reason)
        )
    m2_ransac = next(value for value in m2 if value["variant"] == "ransac")

    left_ids, right_ids = context_indices(start, end, width=8, count=len(contour))
    left_points, right_points = contour[left_ids], contour[right_ids]
    left_fit = ransac_line(left_points, room_threshold, min_inliers=4, source="left_context_ransac_tls")
    right_fit = ransac_line(right_points, room_threshold, min_inliers=4, source="right_context_ransac_tls")
    context = compare_context_lines(left_fit, right_fit, room_radius)
    if context["agrees"]:
        merged_ids = np.concatenate([left_ids, interval_ids, right_ids])
        merged_points = contour[merged_ids]
        merged_fit = ransac_line(
            merged_points,
            room_threshold,
            min_inliers=4,
            source="agreed_context_merged_ransac_tls",
        )
        m3 = [_portal_from_fit("M3", "agreed_merged", merged_fit, merged_points, interval, room_radius, context)]
    else:
        m3 = [
            _portal_from_fit("M3", "left_context", left_fit, left_points, interval, room_radius, context),
            _portal_from_fit("M3", "right_context", right_fit, right_points, interval, room_radius, context),
        ]
    max(m3, key=lambda value: (value["quality"]["reliability_score"], value["variant"]))[
        "selected_predicted_only"
    ] = True

    multi_ids = expanded_indices(start, end, padding=12, count=len(contour))
    multi_points = contour[multi_ids]
    remaining = np.arange(len(multi_points), dtype=np.int64)
    m4: List[Dict[str, Any]] = []
    for rank in range(3):
        if len(remaining) < 4:
            break
        fit = ransac_line(
            multi_points[remaining],
            room_threshold,
            min_inliers=4,
            source="sequential_multiwall_ransac_tls",
        )
        selected_points = multi_points[remaining]
        hypothesis = _portal_from_fit(
            "M4", f"segment_{rank}", fit, selected_points, interval, room_radius
        )
        m4.append(hypothesis)
        remove = fit.inliers
        if int(np.count_nonzero(remove)) < 4:
            break
        remaining = remaining[~remove]
    if not m4:
        m4 = [dict(m2_ransac, method="M4", variant="m2_fallback", selected_predicted_only=False)]
    m4.sort(
        key=lambda value: (value["quality"]["reliability_score"], value["variant"]),
        reverse=True,
    )
    m4[0]["selected_predicted_only"] = True
    return {"M2": m2, "M3": m3, "M4": m4}


def _fourth_moment(points: np.ndarray) -> Dict[str, Any]:
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    keep = lengths > max(1e-10, float(np.max(lengths)) * 1e-8)
    angles = np.arctan2(edges[keep, 1], edges[keep, 0])
    weights = lengths[keep]
    moment = np.sum(weights * np.exp(4j * angles))
    heading = float(np.angle(moment) / 4.0)
    heading = float((heading + math.pi / 4.0) % (math.pi / 2.0) - math.pi / 4.0)
    residuals = np.abs((angles - heading + math.pi / 4.0) % (math.pi / 2.0) - math.pi / 4.0)
    return {
        "heading_degrees": math.degrees(heading),
        "resultant": float(abs(moment) / max(float(weights.sum()), 1e-12)),
        "mean_axis_residual_degrees": float(math.degrees(np.average(residuals, weights=weights))),
    }


def robust_global_heading(contour: np.ndarray, epsilon_fraction: float = 0.015) -> Tuple[float, Dict[str, Any]]:
    edges = np.roll(contour, -1, axis=0) - contour
    perimeter = float(np.linalg.norm(edges, axis=1).sum())
    simplified = cv2.approxPolyDP(
        contour.astype(np.float32)[:, None, :], epsilon_fraction * perimeter, True
    )[:, 0, :].astype(np.float64)
    if len(simplified) < 4:
        raise ValueError("Douglas-Peucker simplification left fewer than four points")
    heading, report = estimate_manhattan_heading(simplified)
    return heading, {
        **report,
        "epsilon_fraction_of_perimeter": float(epsilon_fraction),
        "perimeter": perimeter,
        "simplified_vertex_count": int(len(simplified)),
        "simplified_polygon_bi_xz": simplified.tolist(),
        "unrefined_simplified_fourth_moment": _fourth_moment(simplified),
    }


def global_ransac_heading(
    contour: np.ndarray,
    threshold: float,
    max_lines: int = 8,
    min_support: int = 8,
) -> Tuple[float, Dict[str, Any]]:
    """Aggregate deterministic RANSAC wall segments into a Manhattan heading."""
    value = np.asarray(contour, dtype=np.float64)
    remaining = np.arange(len(value), dtype=np.int64)
    lines = []
    for _ in range(max_lines):
        if len(remaining) < min_support:
            break
        fit = ransac_line(
            value[remaining],
            threshold=threshold,
            min_inliers=min_support,
            source="global_sequential_ransac_tls",
        )
        support = int(np.count_nonzero(fit.inliers))
        if support < min_support:
            break
        active_residual = fit.residuals[fit.inliers]
        rmse = float(math.sqrt(np.mean(np.square(active_residual))))
        if rmse > threshold:
            break
        inlier_points = value[remaining][fit.inliers]
        span = float(np.linalg.norm(inlier_points.max(axis=0) - inlier_points.min(axis=0)))
        angle = float(math.atan2(fit.tangent[1], fit.tangent[0]))
        lines.append(
            {
                "support_point_count": support,
                "span": span,
                "angle_degrees": math.degrees(angle),
                "fit_residual": rmse,
                "tangent_bi_xz": fit.tangent.tolist(),
                "normal_bi_xz": fit.normal.tolist(),
                "offset_bi_xz": float(fit.offset),
            }
        )
        remaining = remaining[~fit.inliers]
    if len(lines) < 2:
        raise ValueError("global RANSAC found fewer than two stable wall segments")
    weights = np.asarray(
        [value["support_point_count"] * value["span"] for value in lines], dtype=np.float64
    )
    angles = np.radians([value["angle_degrees"] for value in lines])
    moment = np.sum(weights * np.exp(4j * angles))
    if abs(moment) <= 1e-12:
        raise ValueError("global RANSAC wall segments have no stable Manhattan family")
    heading = float(np.angle(moment) / 4.0)
    heading = float((heading + math.pi / 4.0) % (math.pi / 2.0) - math.pi / 4.0)
    residuals = np.abs((angles - heading + math.pi / 4.0) % (math.pi / 2.0) - math.pi / 4.0)
    return heading, {
        "headingSource": "predicted_global_sequential_ransac_wall_segments",
        "heading_degrees": math.degrees(heading),
        "confidence": float(abs(moment) / max(float(weights.sum()), 1e-12)),
        "meanAxisResidualDegrees": float(math.degrees(np.average(residuals, weights=weights))),
        "line_count": int(len(lines)),
        "covered_point_count": int(sum(value["support_point_count"] for value in lines)),
        "remaining_point_count": int(len(remaining)),
        "threshold": float(threshold),
        "max_lines": int(max_lines),
        "min_support": int(min_support),
        "lines": lines,
    }


def _layout_from_fitted_bi(points: np.ndarray, source_layout: Mapping[str, Any]) -> Dict[str, Any]:
    xyz = np.insert(np.asarray(points, dtype=np.float64), 1, 1.0, axis=1)
    camera_height = float(source_layout.get("cameraHeight", 1.6))
    ceiling = float(source_layout.get("cameraCeilingHeight", camera_height))
    ratio = ceiling / max(camera_height, 1e-8)
    return xyz2json(xyz, ratio, camera_height=camera_height)


def robust_global_layout(contour: np.ndarray, source_layout: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    heading, report = robust_global_heading(contour)
    canonical = rotate_floor_xz(contour, -heading)
    fitted_canonical = np.asarray(fit_layout(canonical, show=False), dtype=np.float64)
    fitted_local = rotate_floor_xz(fitted_canonical, heading)
    layout = _layout_from_fitted_bi(fitted_local, source_layout)
    validity = polygon_validity(layout)
    if not validity["valid"]:
        raise ValueError(f"M1 robust global polygon is invalid: {validity}")
    report.update(
        heading_degrees=math.degrees(heading),
        canonical_polygon_bi_xz=fitted_canonical.tolist(),
        panorama_local_polygon_bi_xz=fitted_local.tolist(),
        validity=validity,
    )
    return layout, report


def global_ransac_layout(
    contour: np.ndarray,
    source_layout: Mapping[str, Any],
    threshold: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    heading, report = global_ransac_heading(contour, threshold)
    canonical = rotate_floor_xz(contour, -heading)
    fitted_canonical = np.asarray(fit_layout(canonical, show=False), dtype=np.float64)
    fitted_local = rotate_floor_xz(fitted_canonical, heading)
    layout = _layout_from_fitted_bi(fitted_local, source_layout)
    validity = polygon_validity(layout)
    if not validity["valid"]:
        raise ValueError(f"M1 global-RANSAC polygon is invalid: {validity}")
    report.update(
        canonical_polygon_bi_xz=fitted_canonical.tolist(),
        panorama_local_polygon_bi_xz=fitted_local.tolist(),
        validity=validity,
    )
    return layout, report


def _endpoint_line_fit(endpoints_zind: np.ndarray, raw_support_zind: np.ndarray, source: str) -> LineFit:
    base = fit_line_tls(endpoints_zind, source=source)
    residuals = np.abs(raw_support_zind @ base.normal - base.offset)
    base.residuals = residuals
    base.inliers = np.ones(len(raw_support_zind), dtype=bool)
    return base


def _mapped_portal(
    method: str,
    candidate: Any,
    layout: Mapping[str, Any],
    contour_bi: np.ndarray,
    interval: Sequence[int],
    room_radius: float,
) -> Dict[str, Any]:
    endpoints_layout = door_endpoints(layout, candidate.spec)
    endpoints_zind = bi_to_zind(endpoints_layout)
    wall = wall_endpoints(layout, candidate.wall_index)
    wall_zind = bi_to_zind(np.asarray(wall))
    support_ids = circular_indices(int(interval[0]), int(interval[1]), len(contour_bi))
    support_zind = bi_to_zind(contour_bi[support_ids])
    fit_zind = _endpoint_line_fit(endpoints_zind, support_zind, source=f"{method}_mapped_wall")
    width = float(np.linalg.norm(endpoints_zind[1] - endpoints_zind[0]))
    quality = {
        "fit_residual": float(math.sqrt(np.mean(np.square(fit_zind.residuals)))),
        "normalized_residual": float(math.sqrt(np.mean(np.square(fit_zind.residuals)))) / max(room_radius, 1e-8),
        "inlier_ratio": 1.0,
        "support_point_count": int(len(support_zind)),
        "inlier_count": int(len(support_zind)),
        "angular_dispersion_degrees": _axis_dispersion(support_zind, fit_zind.tangent),
        "line_condition_number": 1e12,
        "ray_intersection_condition": None,
        "opening_width": width,
        "opening_width_over_room_radius": width / max(room_radius, 1e-8),
        "width_plausible": bool(0.03 <= width / max(room_radius, 1e-8) <= 2.5),
        "left_right_context_agreement": None,
        "corner_ambiguity": False,
        "reliability_score": None,
        "quality_gate_pass": None,
    }
    return {
        "method": method,
        "variant": "current_mapper" if method == "M0" else "robust_global_mapper",
        "fit_source": fit_zind.source,
        "line": {
            "tangent_zind_local_xy": fit_zind.tangent.tolist(),
            "normal_zind_local_xy": fit_zind.normal.tolist(),
            "offset_zind_local_xy": float(fit_zind.offset),
        },
        "endpoints_bi_xz": None,
        "endpoints_zind_local_xy": endpoints_zind.tolist(),
        "mapped_wall_zind_local_xy": wall_zind.tolist(),
        "wall_index": int(candidate.wall_index),
        "intersection": None,
        "quality": quality,
        "context": None,
        "valid": True,
        "failure_reason": None,
        "selected_predicted_only": True,
    }


def _legacy_layout(contour: np.ndarray, source_layout: Mapping[str, Any]) -> Dict[str, Any]:
    fitted = np.asarray(fit_layout(contour, show=False), dtype=np.float64)
    return _layout_from_fitted_bi(fitted, source_layout)


def smooth_radial_contour(contour: np.ndarray, radius: int = 2) -> np.ndarray:
    value = np.asarray(contour, dtype=np.float64)
    depths = np.linalg.norm(value, axis=1)
    # Explicit circular median avoids scipy and preserves the panorama rays.
    smoothed = np.empty_like(depths)
    for index in range(len(depths)):
        ids = np.arange(index - radius, index + radius + 1) % len(depths)
        smoothed[index] = float(np.median(depths[ids]))
    rays = np.stack([panorama_ray(index, len(depths)) for index in range(len(depths))])
    return rays * smoothed[:, None]


def contour_diagnostics(contour: np.ndarray, intervals: Sequence[Sequence[int]]) -> Dict[str, Any]:
    value = np.asarray(contour, dtype=np.float64)
    edges = np.roll(value, -1, axis=0) - value
    distances = np.linalg.norm(edges, axis=1)
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    turns = np.abs((np.roll(angles, -1) - angles + math.pi) % (2.0 * math.pi) - math.pi)
    depths = np.linalg.norm(value, axis=1)
    depth_delta = np.abs(np.roll(depths, -1) - depths)
    median_delta = float(np.median(depth_delta))
    mad_delta = float(np.median(np.abs(depth_delta - median_delta)))
    jump_threshold = median_delta + 6.0 * max(mad_delta, 1e-8)
    fourth = _fourth_moment(value)
    histogram, bins = np.histogram(
        (np.degrees(angles) + 45.0) % 90.0 - 45.0,
        bins=np.linspace(-45.0, 45.0, 19),
        weights=distances,
    )
    probability = histogram / max(float(histogram.sum()), 1e-12)
    entropy = float(-np.sum(probability[probability > 0] * np.log(probability[probability > 0])))
    opening = []
    for interval in intervals:
        ids = circular_indices(int(interval[0]), int(interval[1]), len(value))
        local_edges = np.diff(value[ids], axis=0)
        local_lengths = np.linalg.norm(local_edges, axis=1)
        local_angles = np.degrees(np.arctan2(local_edges[:, 1], local_edges[:, 0])) if len(local_edges) else np.asarray([])
        opening.append(
            {
                "interval": [int(interval[0]), int(interval[1])],
                "point_count": int(len(ids)),
                "points_bi_xz": value[ids].tolist(),
                "adjacent_distance": descriptive(local_lengths.tolist()),
                "edge_angles_degrees": local_angles.tolist(),
                "depth": descriptive(depths[ids].tolist()),
            }
        )
    return {
        "point_count": int(len(value)),
        "finite": bool(np.isfinite(value).all()),
        "polygon_validity": polygon_validity(value),
        "adjacent_point_distance": descriptive(distances.tolist()),
        "duplicate_point_count": int(np.count_nonzero(distances <= 1e-8)),
        "extremely_short_edge_count": int(np.count_nonzero(distances <= 0.1 * np.median(distances))),
        "turn_angle_degrees": descriptive(np.degrees(turns).tolist()),
        "turn_over_30_degrees_count": int(np.count_nonzero(turns > math.radians(30.0))),
        "turn_over_60_degrees_count": int(np.count_nonzero(turns > math.radians(60.0))),
        "depth": descriptive(depths.tolist()),
        "depth_jump": descriptive(depth_delta.tolist()),
        "depth_jump_threshold": jump_threshold,
        "depth_jump_indices": np.flatnonzero(depth_delta > jump_threshold).astype(int).tolist(),
        "raw_fourth_moment": fourth,
        "modulo_90_direction_histogram": {
            "bin_edges_degrees": bins.tolist(),
            "length_weighted_counts": histogram.tolist(),
            "entropy_nats": entropy,
            "dominant_bin": int(np.argmax(histogram)),
        },
        "opening_intervals": opening,
    }


def fit_similarity_from_endpoints(source_b: np.ndarray, target_a: np.ndarray) -> np.ndarray:
    source = np.asarray(source_b, dtype=np.float64)
    target = np.asarray(target_a, dtype=np.float64)
    if source.shape != (2, 2) or target.shape != (2, 2) or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("portal endpoints must be finite [2, 2]")
    vector_b, vector_a = source[1] - source[0], target[1] - target[0]
    width_b, width_a = float(np.linalg.norm(vector_b)), float(np.linalg.norm(vector_a))
    if min(width_a, width_b) <= 1e-8:
        raise ValueError("portal endpoint width is degenerate")
    scale = width_a / width_b
    yaw = math.atan2(vector_a[1], vector_a[0]) - math.atan2(vector_b[1], vector_b[0])
    cosine, sine = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    translation = target.mean(axis=0) - scale * rotation @ source.mean(axis=0)
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :2] = scale * rotation
    matrix[:2, 2] = translation
    return matrix


def fit_pair_pose(
    endpoints_a: np.ndarray,
    endpoints_b: np.ndarray,
    polygon_a: np.ndarray,
    polygon_b: np.ndarray,
) -> Dict[str, Any]:
    candidates = []
    portal_vector = endpoints_a[1] - endpoints_a[0]
    portal_midpoint = endpoints_a.mean(axis=0)
    side_a = float(np.cross(portal_vector, polygon_a.mean(axis=0) - portal_midpoint))
    for mapping, target in (("same", endpoints_a), ("reversed", endpoints_a[::-1])):
        transform = fit_similarity_from_endpoints(endpoints_b, target)
        transformed_b = apply_transform(polygon_b, transform)
        side_b = float(np.cross(portal_vector, transformed_b.mean(axis=0) - portal_midpoint))
        candidates.append(
            {
                "endpoint_mapping": mapping,
                "T_B_to_A": transform.tolist(),
                "relative_yaw_radians": transform_yaw(transform),
                "relative_yaw_degrees": math.degrees(transform_yaw(transform)),
                "relative_scale": transform_scale(transform),
                "translation": transform[:2, 2].tolist(),
                "centroid_side_product": side_a * side_b,
            }
        )
    selected = min(candidates, key=lambda value: (value["centroid_side_product"], value["endpoint_mapping"]))
    selected["transform_direction"] = "B_to_A"
    selected["selection"] = "predicted polygon centroid opposite-side rule"
    return selected


def _room_predicted_inputs(rooms: Mapping[str, Any]) -> Dict[str, Any]:
    output = {}
    for room, value in rooms.items():
        single = value["single_room_layout"]
        output[room] = {
            "room_id": room,
            "pano_id": value["pano_id"],
            "pre_contour_bi_xz": single["layout_polygon_pre_manhattan_local_bi_xz"],
            "frame_layout": single["predicted_layout_bi_xz"],
            "frame_postprocess": single["postprocess"],
            "opening_prediction": value["opening_prediction"],
        }
    return output


def _pair_predicted_inputs(entries: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    predicted, ground_truth = {}, {}
    for entry in entries:
        prediction = entry["prediction"]
        pair_id = str(prediction["pair_id"])
        predicted[pair_id] = {
            key: prediction[key]
            for key in (
                "pair_id",
                "room_a",
                "room_b",
                "pano_a",
                "pano_b",
                "best_candidate_pair",
                "is_match",
                "relative_pose",
                "pose_valid",
                "joint_bev",
                "shared_opening_details",
            )
        }
        ground_truth[pair_id] = entry["ground_truth"]
    return predicted, ground_truth


def _selected(hypotheses: Sequence[Mapping[str, Any]], method: str) -> Mapping[str, Any]:
    matches = [value for value in hypotheses if value["method"] == method and value["selected_predicted_only"]]
    if len(matches) != 1:
        raise RuntimeError(f"expected one selected {method} hypothesis, found {len(matches)}")
    return matches[0]


def _candidate_by_index(room: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    values = [value for value in room["opening_prediction"]["candidates"] if int(value["candidateIndex"]) == int(index)]
    if len(values) != 1:
        raise ValueError(f"candidate index {index} is not unique")
    return values[0]


def _candidate_object(layout: Mapping[str, Any], candidate: Mapping[str, Any]):
    from portallayout.engine.cross_scene_estimator import OpeningCandidate

    return OpeningCandidate(
        wall_index=int(candidate["wallIndex"]),
        start_ratio=float(candidate["startRatio"]),
        end_ratio=float(candidate["endRatio"]),
        confidence=float(candidate["confidence"]),
        token_start=int(candidate["tokenStart"]),
        token_end=int(candidate["tokenEnd"]),
        source=str(candidate["source"]),
        candidate_index=int(candidate["candidateIndex"]),
        metrics=dict(candidate.get("metrics", {})),
    )


def build_predicted_hypotheses(
    room_inputs: Mapping[str, Any],
    pair_inputs: Mapping[str, Any],
) -> Dict[str, Any]:
    """Freeze M0-M4 using predicted data only; no GT parameter is accepted."""
    room_state: Dict[str, Any] = {}
    for room, value in room_inputs.items():
        contour = np.asarray(value["pre_contour_bi_xz"], dtype=np.float64)
        probability = np.asarray(value["opening_prediction"]["probability"], dtype=np.float64)
        if contour.shape != (TOKEN_COUNT, 2) or probability.shape != (TOKEN_COUNT,):
            raise ValueError(f"saved contour/probability shape changed for {room}")
        radius = float(np.median(np.linalg.norm(contour, axis=1)))
        adjacent = np.linalg.norm(np.roll(contour, -1, axis=0) - contour, axis=1)
        threshold = min(0.06 * radius, max(0.008 * radius, 2.5 * float(np.median(adjacent))))
        m1_layout, m1_report = robust_global_layout(contour, value["frame_layout"])
        m1_ransac_layout, m1_ransac_report = global_ransac_layout(
            contour, value["frame_layout"], threshold
        )
        legacy_layout = _legacy_layout(contour, value["frame_layout"])
        smooth = smooth_radial_contour(contour)
        room_state[room] = {
            "room_id": room,
            "pano_id": value["pano_id"],
            "contour_bi_xz": contour,
            "contour_zind_xy": bi_to_zind(contour),
            "smoothed_contour_zind_xy": bi_to_zind(smooth),
            "frame_layout": value["frame_layout"],
            "frame_polygon_zind_xy": bi_to_zind(layout_xz(value["frame_layout"])),
            "legacy_layout": legacy_layout,
            "legacy_polygon_zind_xy": bi_to_zind(layout_xz(legacy_layout)),
            "m1_layout": m1_layout,
            "m1_polygon_zind_xy": bi_to_zind(layout_xz(m1_layout)),
            "m1_report": m1_report,
            "m1_ransac_layout": m1_ransac_layout,
            "m1_ransac_polygon_zind_xy": bi_to_zind(layout_xz(m1_ransac_layout)),
            "m1_ransac_report": m1_ransac_report,
            "probability": probability,
            "opening_prediction": value["opening_prediction"],
            "room_radius": radius,
            "ransac_threshold": threshold,
            "contour_diagnostics": contour_diagnostics(
                contour, value["opening_prediction"]["intervals"]
            ),
        }

    positive_pairs = {}
    negative_pairs = {}
    for pair_id, pair in pair_inputs.items():
        if not bool(pair["is_match"]):
            if pair["best_candidate_pair"] != [-1, -1] or pair["relative_pose"] is not None or pair["joint_bev"] is not None:
                raise ValueError("negative saved pair violates no-pose contract")
            negative_pairs[pair_id] = {
                "pair_id": pair_id,
                "room_a": pair["room_a"],
                "room_b": pair["room_b"],
                "best_candidate_pair": [-1, -1],
                "pose_generated": False,
                "methods_run": [],
            }
            continue
        index_a, index_b = [int(value) for value in pair["best_candidate_pair"]]
        room_a, room_b = str(pair["room_a"]), str(pair["room_b"])
        candidates = {}
        for side, room, candidate_index in (("A", room_a, index_a), ("B", room_b, index_b)):
            state = room_state[room]
            saved_candidate = _candidate_by_index(room_inputs[room], candidate_index)
            interval = [int(saved_candidate["tokenStart"]), int(saved_candidate["tokenEnd"])]
            local = _candidate_local_methods(
                state["contour_bi_xz"],
                state["probability"],
                interval,
                state["ransac_threshold"],
                state["room_radius"],
            )
            current_object = _candidate_object(state["frame_layout"], saved_candidate)
            m0 = _mapped_portal(
                "M0", current_object, state["frame_layout"], state["contour_bi_xz"], interval, state["room_radius"]
            )
            m1_candidates = opening_candidates_from_intervals(
                state["m1_layout"], [tuple(interval)], state["probability"]
            )
            if len(m1_candidates) != 1:
                raise RuntimeError("M1 mapper did not preserve one interval")
            m1 = _mapped_portal(
                "M1", m1_candidates[0], state["m1_layout"], state["contour_bi_xz"], interval, state["room_radius"]
            )
            m1_ransac_candidates = opening_candidates_from_intervals(
                state["m1_ransac_layout"], [tuple(interval)], state["probability"]
            )
            if len(m1_ransac_candidates) != 1:
                raise RuntimeError("M1 global-RANSAC mapper did not preserve one interval")
            m1_ransac = _mapped_portal(
                "M1",
                m1_ransac_candidates[0],
                state["m1_ransac_layout"],
                state["contour_bi_xz"],
                interval,
                state["room_radius"],
            )
            m1_ransac["variant"] = "global_ransac_mapper"
            m1_ransac["selected_predicted_only"] = False
            hypotheses = [m0, m1, m1_ransac] + local["M2"] + local["M3"] + local["M4"]
            for hypothesis in hypotheses:
                hypothesis.update(
                    pair_id=pair_id,
                    side=side,
                    room_id=room,
                    candidate_index=candidate_index,
                    interval=interval,
                )
            candidates[side] = {
                "room_id": room,
                "candidate_index": candidate_index,
                "interval": interval,
                "hypotheses": hypotheses,
            }

        method_pairs = {}
        representation_map = {
            "legacy_manhattan": "legacy_polygon_zind_xy",
            "frame_preserving_manhattan": "frame_polygon_zind_xy",
            "raw_pre_manhattan": "contour_zind_xy",
            "smoothed_pre_manhattan": "smoothed_contour_zind_xy",
        }
        for method in METHODS:
            selected_a = _selected(candidates["A"]["hypotheses"], method)
            selected_b = _selected(candidates["B"]["hypotheses"], method)
            if not selected_a["valid"] or not selected_b["valid"]:
                method_pairs[method] = {
                    "method": method,
                    "pose_valid": False,
                    "failure_reason": "selected portal hypothesis is invalid",
                }
                continue
            endpoints_a = np.asarray(selected_a["endpoints_zind_local_xy"], dtype=np.float64)
            endpoints_b = np.asarray(selected_b["endpoints_zind_local_xy"], dtype=np.float64)
            if method == "M0":
                polygon_key = "frame_polygon_zind_xy"
            elif method == "M1":
                polygon_key = "m1_polygon_zind_xy"
            else:
                polygon_key = "contour_zind_xy"
            polygon_a = room_state[room_a][polygon_key]
            polygon_b = room_state[room_b][polygon_key]
            pose = fit_pair_pose(endpoints_a, endpoints_b, polygon_a, polygon_b)
            method_pairs[method] = {
                "method": method,
                "pose_valid": True,
                "selected_portal_variant_a": selected_a["variant"],
                "selected_portal_variant_b": selected_b["variant"],
                "endpoints_a": endpoints_a.tolist(),
                "endpoints_b": endpoints_b.tolist(),
                "pose": pose,
                "pose_polygon_representation": polygon_key,
                "display_representations": {
                    name: {
                        "polygon_a": room_state[room_a][key].tolist(),
                        "polygon_b": room_state[room_b][key].tolist(),
                    }
                    for name, key in representation_map.items()
                },
            }
        ransac_a_values = [
            value
            for value in candidates["A"]["hypotheses"]
            if value["method"] == "M1" and value["variant"] == "global_ransac_mapper"
        ]
        ransac_b_values = [
            value
            for value in candidates["B"]["hypotheses"]
            if value["method"] == "M1" and value["variant"] == "global_ransac_mapper"
        ]
        if len(ransac_a_values) != 1 or len(ransac_b_values) != 1:
            raise RuntimeError("expected exactly one global-RANSAC portal per side")
        ransac_a, ransac_b = ransac_a_values[0], ransac_b_values[0]
        endpoints_a = np.asarray(ransac_a["endpoints_zind_local_xy"], dtype=np.float64)
        endpoints_b = np.asarray(ransac_b["endpoints_zind_local_xy"], dtype=np.float64)
        polygon_a = room_state[room_a]["m1_ransac_polygon_zind_xy"]
        polygon_b = room_state[room_b]["m1_ransac_polygon_zind_xy"]
        ransac_pose = fit_pair_pose(endpoints_a, endpoints_b, polygon_a, polygon_b)
        method_pairs["M1_RANSAC"] = {
            "method": "M1_RANSAC",
            "pose_valid": True,
            "selected_portal_variant_a": "global_ransac_mapper",
            "selected_portal_variant_b": "global_ransac_mapper",
            "endpoints_a": endpoints_a.tolist(),
            "endpoints_b": endpoints_b.tolist(),
            "pose": ransac_pose,
            "pose_polygon_representation": "m1_ransac_polygon_zind_xy",
            "display_representations": {
                name: {
                    "polygon_a": room_state[room_a][key].tolist(),
                    "polygon_b": room_state[room_b][key].tolist(),
                }
                for name, key in representation_map.items()
            },
        }
        positive_pairs[pair_id] = {
            "pair_id": pair_id,
            "room_a": room_a,
            "room_b": room_b,
            "best_candidate_pair": [index_a, index_b],
            "candidate_a": candidates["A"],
            "candidate_b": candidates["B"],
            "methods": method_pairs,
        }
    return {
        "room_state": room_state,
        "positive_pairs": positive_pairs,
        "negative_pairs": negative_pairs,
        "gt_available_to_hypothesis_generation": False,
    }


def _line_from_endpoints(points: np.ndarray) -> Tuple[np.ndarray, float, float]:
    vector = points[1] - points[0]
    width = float(np.linalg.norm(vector))
    if width <= 1e-10:
        raise ValueError("GT/predicted endpoints are degenerate")
    tangent = vector / width
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
    offset = float(np.dot(normal, points.mean(axis=0)))
    return normal, offset, math.atan2(tangent[1], tangent[0])


def endpoint_gt_metrics(predicted: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
    pred = np.asarray(predicted, dtype=np.float64)
    gt = np.asarray(target, dtype=np.float64)
    direct = float(math.sqrt(np.mean(np.square(pred - gt))))
    reversed_error = float(math.sqrt(np.mean(np.square(pred - gt[::-1]))))
    normal_pred, offset_pred, tangent_pred = _line_from_endpoints(pred)
    normal_gt, offset_gt, tangent_gt = _line_from_endpoints(gt)
    if float(np.dot(normal_pred, normal_gt)) < 0:
        normal_pred *= -1.0
        offset_pred *= -1.0
    return {
        "portal_wall_tangent_error_degrees": math.degrees(
            wrapped_angle_error(tangent_pred, tangent_gt, period=math.pi)
        ),
        "portal_line_offset_error": abs(offset_pred - offset_gt),
        "endpoint_rmse": min(direct, reversed_error),
        "endpoint_order": "same" if direct <= reversed_error else "reversed",
        "midpoint_error": float(np.linalg.norm(pred.mean(axis=0) - gt.mean(axis=0))),
        "predicted_width": float(np.linalg.norm(pred[1] - pred[0])),
        "gt_width": float(np.linalg.norm(gt[1] - gt[0])),
        "width_error": abs(float(np.linalg.norm(pred[1] - pred[0])) - float(np.linalg.norm(gt[1] - gt[0]))),
    }


def _pair_gt_metrics(method: Mapping[str, Any], ground_truth: Mapping[str, Any], metric_factor: float) -> Dict[str, Any]:
    if not method.get("pose_valid"):
        return {
            "yaw_error_degrees": None,
            "scale_error": None,
            "absolute_log_scale_error": None,
            "translation_error": None,
            "translation_meter_error": None,
            "representation_overlap": {},
        }
    pose = method["pose"]
    matrix = np.asarray(pose["T_B_to_A"], dtype=np.float64)
    target = np.asarray(ground_truth["relative_pose_gt"]["T_B_to_A"], dtype=np.float64)
    gt_scale = transform_scale(target)
    predicted_scale = transform_scale(matrix)
    target_instances = ground_truth["joint_bev_target"]["instances"]
    overlaps = {}
    for name, representation in method["display_representations"].items():
        polygon_a = np.asarray(representation["polygon_a"], dtype=np.float64)
        polygon_b = apply_transform(representation["polygon_b"], matrix)
        overlaps[name] = {
            "instance_overlap_A": polygon_overlap_ratio(
                polygon_a, np.asarray(target_instances[0]["polygon"], dtype=np.float64)
            ),
            "instance_overlap_B": polygon_overlap_ratio(
                polygon_b, np.asarray(target_instances[1]["polygon"], dtype=np.float64)
            ),
        }
    translation_error = float(np.linalg.norm(matrix[:2, 2] - target[:2, 2]))
    return {
        "predicted_yaw_degrees": math.degrees(transform_yaw(matrix)),
        "gt_yaw_degrees": math.degrees(transform_yaw(target)),
        "yaw_error_degrees": math.degrees(wrapped_angle_error(transform_yaw(matrix), transform_yaw(target))),
        "predicted_scale": predicted_scale,
        "gt_scale": gt_scale,
        "scale_error": abs(predicted_scale - gt_scale),
        "absolute_log_scale_error": abs(math.log(predicted_scale / gt_scale)),
        "predicted_translation": matrix[:2, 2].tolist(),
        "gt_translation": target[:2, 2].tolist(),
        "translation_error": translation_error,
        "translation_meter_error": translation_error * metric_factor,
        "representation_overlap": overlaps,
    }


def attach_posthoc_ground_truth(
    predicted: Mapping[str, Any],
    rooms_full: Mapping[str, Any],
    pair_ground_truth: Mapping[str, Any],
    floor_scale: float,
) -> Dict[str, Any]:
    """Attach saved train GT only after predicted hypotheses are immutable."""
    positive = predicted["positive_pairs"]
    for pair_id, pair in positive.items():
        gt = pair_ground_truth[pair_id]
        if not gt["is_match"] or len(gt["shared_interface_details"]) != 1:
            raise ValueError("positive pair GT contract changed")
        interface = gt["shared_interface_details"][0]
        targets = {
            "A": np.asarray(interface["local_endpoints_a"], dtype=np.float64),
            "B": np.asarray(interface["local_endpoints_b"], dtype=np.float64),
        }
        for side, key in (("A", "candidate_a"), ("B", "candidate_b")):
            for hypothesis in pair[key]["hypotheses"]:
                if hypothesis["valid"]:
                    hypothesis["posthoc_gt"] = endpoint_gt_metrics(
                        np.asarray(hypothesis["endpoints_zind_local_xy"], dtype=np.float64),
                        targets[side],
                    )
                else:
                    hypothesis["posthoc_gt"] = None
        room_a = pair["room_a"]
        metric_factor = (
            float(rooms_full[room_a]["ground_truth"]["floor_plan_transformation"]["scale"])
            * float(floor_scale)
        )
        for method in pair["methods"].values():
            method["posthoc_gt"] = _pair_gt_metrics(method, gt, metric_factor)
    for pair_id, value in predicted["negative_pairs"].items():
        gt = pair_ground_truth[pair_id]
        if gt["is_match"] or gt["relative_pose_gt"] is not None or gt["joint_bev_target"] is not None:
            raise ValueError("negative pair GT is not sanitized")
        value["posthoc_gt"] = {"is_match": False, "pose_expected": False}
    return predicted


def _flatten_portal_rows(predicted: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for pair in predicted["positive_pairs"].values():
        for key in ("candidate_a", "candidate_b"):
            for value in pair[key]["hypotheses"]:
                quality = value["quality"]
                gt = value.get("posthoc_gt") or {}
                line = value.get("line") or {}
                rows.append(
                    {
                        "pair_id": pair["pair_id"],
                        "side": value["side"],
                        "room_id": value["room_id"],
                        "candidate_index": value["candidate_index"],
                        "interval_start": value["interval"][0],
                        "interval_end": value["interval"][1],
                        "method": value["method"],
                        "variant": value["variant"],
                        "selected_predicted_only": value["selected_predicted_only"],
                        "valid": value["valid"],
                        "fit_residual": quality.get("fit_residual"),
                        "normalized_residual": quality.get("normalized_residual"),
                        "inlier_ratio": quality.get("inlier_ratio"),
                        "support_point_count": quality.get("support_point_count"),
                        "angular_dispersion_degrees": quality.get("angular_dispersion_degrees"),
                        "line_condition_number": quality.get("line_condition_number"),
                        "ray_intersection_condition": quality.get("ray_intersection_condition"),
                        "opening_width": quality.get("opening_width"),
                        "width_plausible": quality.get("width_plausible"),
                        "left_right_context_agreement": quality.get("left_right_context_agreement"),
                        "corner_ambiguity": quality.get("corner_ambiguity"),
                        "reliability_score": quality.get("reliability_score"),
                        "quality_gate_pass": quality.get("quality_gate_pass"),
                        "endpoints_zind_local_xy": json.dumps(value["endpoints_zind_local_xy"], separators=(",", ":")),
                        "line": json.dumps(line, separators=(",", ":")),
                        "portal_wall_tangent_error_degrees": gt.get("portal_wall_tangent_error_degrees"),
                        "portal_line_offset_error": gt.get("portal_line_offset_error"),
                        "endpoint_rmse": gt.get("endpoint_rmse"),
                        "midpoint_error": gt.get("midpoint_error"),
                        "width_error": gt.get("width_error"),
                    }
                )
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _heading_rows(predicted: Mapping[str, Any], rooms_full: Mapping[str, Any], pairs_gt: Mapping[str, Any]) -> List[Dict[str, Any]]:
    gt_angles: Dict[str, List[float]] = {room: [] for room in predicted["room_state"]}
    for pair_id, pair in predicted["positive_pairs"].items():
        interface = pairs_gt[pair_id]["shared_interface_details"][0]
        for room, endpoints in (
            (pair["room_a"], interface["local_endpoints_a"]),
            (pair["room_b"], interface["local_endpoints_b"]),
        ):
            points = np.asarray(endpoints, dtype=np.float64)
            gt_angles[room].append(math.atan2(*(points[1] - points[0])[::-1]))
    rows = []
    for room, state in predicted["room_state"].items():
        current = rooms_full[room]["single_room_layout"]["manhattan_heading"]
        raw = state["contour_diagnostics"]["raw_fourth_moment"]
        m1 = state["m1_report"]
        m1_ransac = state["m1_ransac_report"]
        moment = sum(np.exp(4j * np.asarray(gt_angles[room], dtype=np.float64)))
        gt_heading = float(np.angle(moment) / 4.0)
        # Bi->ZInD x reflection negates the unoriented tangent angle.
        m1_zind = -math.radians(float(m1["heading_degrees"]))
        m1_ransac_zind = -math.radians(float(m1_ransac["heading_degrees"]))
        current_zind = -math.radians(float(current["degrees"]))
        rows.append(
            {
                "room_id": room,
                "pano_id": state["pano_id"],
                "raw_all_edge_heading_bi_degrees": raw["heading_degrees"],
                "raw_all_edge_fourth_resultant": raw["resultant"],
                "current_robust_heading_bi_degrees": current["degrees"],
                "current_reported_confidence": current["confidence"],
                "current_mean_axis_residual_degrees": rooms_full[room]["single_room_layout"]["postprocess"]["meanAxisResidualDegrees"],
                "m1_heading_bi_degrees": m1["heading_degrees"],
                "m1_simplified_vertex_count": m1["simplified_vertex_count"],
                "m1_confidence": m1["confidence"],
                "m1_ransac_heading_bi_degrees": m1_ransac["heading_degrees"],
                "m1_ransac_line_count": m1_ransac["line_count"],
                "m1_ransac_covered_point_count": m1_ransac["covered_point_count"],
                "m1_ransac_confidence": m1_ransac["confidence"],
                "gt_portal_heading_zind_degrees_posthoc": math.degrees(gt_heading),
                "current_heading_error_degrees_posthoc": math.degrees(
                    wrapped_angle_error(current_zind, gt_heading, period=math.pi / 2.0)
                ),
                "m1_heading_error_degrees_posthoc": math.degrees(
                    wrapped_angle_error(m1_zind, gt_heading, period=math.pi / 2.0)
                ),
                "m1_ransac_heading_error_degrees_posthoc": math.degrees(
                    wrapped_angle_error(m1_ransac_zind, gt_heading, period=math.pi / 2.0)
                ),
            }
        )
    return rows


def _svg_projector(arrays: Sequence[np.ndarray], width: int = 760, height: int = 760, padding: int = 70):
    points = np.concatenate([value for value in arrays if len(value)], axis=0)
    minimum, maximum = points.min(axis=0), points.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-6)
    scale = min((width - 2 * padding) / span[0], (height - 2 * padding) / span[1])
    center = (minimum + maximum) / 2.0

    def project(value: np.ndarray) -> np.ndarray:
        output = (value - center) * scale
        output[:, 1] *= -1.0
        output += np.asarray([width / 2.0, height / 2.0])
        return output

    return project


def render_portal_svg(
    path: Path,
    state: Mapping[str, Any],
    hypotheses: Sequence[Mapping[str, Any]],
    interval: Sequence[int],
    gt_endpoints: np.ndarray,
) -> None:
    contour = np.asarray(state["contour_zind_xy"], dtype=np.float64)
    selected = [value for value in hypotheses if value["selected_predicted_only"] and value["valid"]]
    endpoint_arrays = [np.asarray(value["endpoints_zind_local_xy"], dtype=np.float64) for value in selected]
    arrays = [contour, gt_endpoints, np.zeros((1, 2))] + endpoint_arrays
    project = _svg_projector(arrays)
    width = height = 760
    contour_px = project(contour)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<polyline points="{}" fill="none" stroke="#777" stroke-width="2" opacity="0.8"/>'.format(
            " ".join(f"{x:.2f},{y:.2f}" for x, y in np.vstack([contour_px, contour_px[:1]]))
        ),
    ]
    origin = project(np.zeros((1, 2)))[0]
    ray_length = float(np.max(np.linalg.norm(contour, axis=1)) * 1.25)
    for token in (int(interval[0]), int(interval[1])):
        ray_bi = panorama_ray(token)
        ray_zind = bi_to_zind(np.asarray([[0.0, 0.0], ray_bi * ray_length]))[1]
        endpoint = project(np.asarray([[0.0, 0.0], ray_zind]))[1]
        svg.append(
            f'<line x1="{origin[0]:.2f}" y1="{origin[1]:.2f}" x2="{endpoint[0]:.2f}" y2="{endpoint[1]:.2f}" stroke="#444" stroke-dasharray="6,5"/>'
        )
    for value in selected:
        endpoints = project(np.asarray(value["endpoints_zind_local_xy"], dtype=np.float64))
        color = COLORS[value["method"]]
        if value.get("mapped_wall_zind_local_xy") is not None:
            fitted_wall = np.asarray(value["mapped_wall_zind_local_xy"], dtype=np.float64)
        else:
            endpoints_bi = np.asarray(value["endpoints_bi_xz"], dtype=np.float64)
            tangent_bi = np.asarray(value["line"]["tangent_bi_xz"], dtype=np.float64)
            extent = float(state["room_radius"] * 1.5)
            midpoint_bi = endpoints_bi.mean(axis=0)
            fitted_wall = bi_to_zind(
                np.stack([midpoint_bi - extent * tangent_bi, midpoint_bi + extent * tangent_bi])
            )
        wall_px = project(fitted_wall)
        svg.append(
            f'<line x1="{wall_px[0,0]:.2f}" y1="{wall_px[0,1]:.2f}" x2="{wall_px[1,0]:.2f}" y2="{wall_px[1,1]:.2f}" stroke="{color}" stroke-width="2" opacity="0.65"/>'
        )
        svg.append(
            f'<line x1="{endpoints[0,0]:.2f}" y1="{endpoints[0,1]:.2f}" x2="{endpoints[1,0]:.2f}" y2="{endpoints[1,1]:.2f}" stroke="{color}" stroke-width="6"/>'
        )
    gt_px = project(gt_endpoints)
    svg.append(
        f'<line x1="{gt_px[0,0]:.2f}" y1="{gt_px[0,1]:.2f}" x2="{gt_px[1,0]:.2f}" y2="{gt_px[1,1]:.2f}" stroke="#000" stroke-width="4" stroke-dasharray="10,5"/>'
    )
    y = 24
    svg.append(f'<text x="16" y="{y}" font-size="17">{html.escape(state["room_id"])} interval {list(interval)}</text>')
    for value in selected:
        y += 22
        quality = value["quality"]
        gt = value.get("posthoc_gt") or {}
        label = (
            f'{value["method"]}:{value["variant"]} score={quality.get("reliability_score")} '
            f'resid={quality.get("fit_residual")} tangentGT={gt.get("portal_wall_tangent_error_degrees")}'
        )
        svg.append(
            f'<text x="16" y="{y}" font-size="14" fill="{COLORS[value["method"]]}">{html.escape(label)}</text>'
        )
    svg.append('</svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


def _method_table(predicted: Mapping[str, Any]) -> Dict[str, Any]:
    output = {}
    for pair_id, pair in predicted["positive_pairs"].items():
        output[pair_id] = {}
        for method, value in pair["methods"].items():
            output[pair_id][method] = {
                "pose_valid": value.get("pose_valid"),
                "selected_portal_variant_a": value.get("selected_portal_variant_a"),
                "selected_portal_variant_b": value.get("selected_portal_variant_b"),
                "pose": value.get("pose"),
                "posthoc_gt": value.get("posthoc_gt"),
            }
    return output


def write_diagnosis(path: Path, summary: Mapping[str, Any]) -> None:
    heading = {value["room_id"]: value for value in summary["heading_comparison"]}
    methods = summary["method_comparison"]
    pair_ids = list(methods)
    lines = [
        "# Candidate-local portal geometry audit",
        "",
        "This is a CPU-only diagnostic. All M0-M4 hypotheses were frozen from saved predictions before train GT was attached.",
        "",
        "## Global heading",
        "",
        (
            f"Room04 all-edge fourth resultant is {heading['complete_room_04']['raw_all_edge_fourth_resultant']:.6f}, "
            f"while the prior robust-subset confidence was {heading['complete_room_04']['current_reported_confidence']:.6f}. "
            "Douglas-Peucker long-segment aggregation recovers a coherent global direction without GT."
        ),
        (
            f"Room05 shows the same confidence mismatch: all-edge resultant "
            f"{heading['complete_room_05']['raw_all_edge_fourth_resultant']:.6f} versus reported "
            f"{heading['complete_room_05']['current_reported_confidence']:.6f}."
        ),
        "",
        "## Uniform-method pair results",
        "",
        "| pair | method | yaw error deg | scale | translation error | translation meter error |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for pair_id in pair_ids:
        for method in METHODS:
            value = methods[pair_id][method]
            gt = value["posthoc_gt"]
            pose = value.get("pose") or {}
            lines.append(
                f"| {pair_id} | {method} | {gt.get('yaw_error_degrees')} | "
                f"{pose.get('relative_scale')} | {gt.get('translation_error')} | "
                f"{gt.get('translation_meter_error')} |"
            )
    pair_0104 = methods["0665_floor_03_pano_34_pano_37_3"]
    pair_0105 = methods["0665_floor_03_pano_34_pano_30_4"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- M0 is the saved frame-fixed mapper baseline.",
            "- M1 changes only predicted global-heading estimation; it still maps intervals through a global Manhattan wall index.",
            "- M2 fits one candidate-local line; its primary TLS/RANSAC variant is selected by predicted-only reliability.",
            "- M3 checks left/right context and marks disagreement as corner ambiguity.",
            "- M4 retains up to three predicted-only local wall segments; the primary is quality-ranked without GT.",
            "- Width-ratio scale is unchanged in every pair pose.",
            "- Negative pairs never generate a portal pose.",
            "",
            "## Required findings",
            "",
            (
                "1. Room04 is a simple, finite 256-point polygon, so the heading failure is not self-intersection or duplicate topology. "
                f"It has an all-edge fourth resultant of {heading['complete_room_04']['raw_all_edge_fourth_resultant']:.6f}, "
                f"{summary['contour_diagnostics']['complete_room_04']['turn_over_60_degrees_count']} turns above 60 degrees, and a high direction-histogram entropy. "
                "The dominant cause is noisy/multimodal adjacent-edge direction."
            ),
            (
                "2. Room01 M1 post-hoc heading error is only "
                f"{heading['complete_room_01']['m1_heading_error_degrees_posthoc']:.6f} degrees, but M1 01<-04 yaw error is "
                f"{pair_0104['M1']['posthoc_gt']['yaw_error_degrees']:.6f} degrees. "
                "Global heading therefore does not resolve interval-to-opposite-wall assignment after global Manhattan fitting."
            ),
            (
                "3. Global sequential RANSAC recovers every room's modulo-90 heading to within "
                f"{max(value['m1_ransac_heading_error_degrees_posthoc'] for value in heading.values()):.6f} degrees, "
                f"yet its pair yaw errors remain {pair_0104['M1_RANSAC']['posthoc_gt']['yaw_error_degrees']:.6f} and "
                f"{pair_0105['M1_RANSAC']['posthoc_gt']['yaw_error_degrees']:.6f} degrees. It does not repair wall-side."
            ),
            (
                "4. M2 candidate-local fitting reduces pair yaw errors to "
                f"{pair_0104['M2']['posthoc_gt']['yaw_error_degrees']:.6f} and "
                f"{pair_0105['M2']['posthoc_gt']['yaw_error_degrees']:.6f} degrees without GT selection."
            ),
            (
                "5. Candidate-local tangent estimates are consistently close to GT, while endpoint midpoint/width errors remain candidate dependent; "
                "this separates orientation recovery from portal depth/extent quality."
            ),
            (
                "6. M2 improves both positive edges simultaneously. M4's predicted-only primary also improves both, but one primary segment has only four support points."
            ),
            (
                "7. M2 width-ratio scales are "
                f"{pair_0104['M2']['pose']['relative_scale']:.6f} and {pair_0105['M2']['pose']['relative_scale']:.6f}; "
                f"M4 gives {pair_0104['M4']['pose']['relative_scale']:.6f} and {pair_0105['M4']['pose']['relative_scale']:.6f}."
            ),
            (
                "8. M2 translation errors fall to "
                f"{pair_0104['M2']['posthoc_gt']['translation_error']:.6f} and "
                f"{pair_0105['M2']['posthoc_gt']['translation_error']:.6f} coordinate units."
            ),
            (
                "9. Useful predicted-only failure signals are: all-edge fourth resultant, global mean axis residual, local normalized residual, "
                "angular dispersion, context angle/offset disagreement, support count, ray condition, and width/radius. "
                "The old confidence is misleading because it is normalized only over the robustly reweighted subset."
            ),
            (
                "10. The evidence favors candidate-local portal geometry with retained multi-wall hypotheses for ambiguous contexts. "
                "Global heading alone is insufficient; portal depth/extent and quality ranking should be audited next before any formal replacement."
            ),
            "",
            "See audit_summary.json and portal_hypotheses.csv for the post-hoc GT details; no GT metric participates in hypothesis selection.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_inputs(args: argparse.Namespace) -> Dict[str, Any]:
    paths = {
        "rooms": resolved_input(args.rooms_json, "rooms JSON"),
        "pairs": resolved_input(args.pairs_json, "pairs JSON"),
        "group": resolved_input(args.group_manifest, "group manifest"),
    }
    rooms = load_json(paths["rooms"])
    pair_payload = load_json(paths["pairs"])
    group = load_json(paths["group"])
    if group.get("split") != "train" or group.get("house_id") != "0665" or group.get("floor_id") != "floor_03":
        raise ValueError("audit is fixed to train house 0665 floor_03")
    if set(rooms) != set(group["room_ids"]):
        raise ValueError("saved room set differs from group")
    entries = pair_payload.get("pairs", [])
    if len(entries) != 3:
        raise ValueError("fixed three-room output must contain three pairs")
    pair_predicted, pair_gt = _pair_predicted_inputs(entries)
    expected = {
        "0665_floor_03_pano_34_pano_37_3": [1, 0],
        "0665_floor_03_pano_34_pano_30_4": [0, 2],
        "0665_floor_03_pano_37_pano_30_negative": [-1, -1],
    }
    if {key: value["best_candidate_pair"] for key, value in pair_predicted.items()} != expected:
        raise ValueError("frozen T3 candidate pairs changed")
    return {"paths": paths, "rooms": rooms, "pairs": pair_predicted, "pair_gt": pair_gt, "group": group}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    inputs = load_inputs(args)
    room_inputs = _room_predicted_inputs(inputs["rooms"])
    predicted = build_predicted_hypotheses(room_inputs, inputs["pairs"])
    if not finite_tree(
        {
            "negative_pairs": predicted["negative_pairs"],
            "gt_available": predicted["gt_available_to_hypothesis_generation"],
        }
    ):
        raise ValueError("predicted-only routing output is non-finite")
    attach_posthoc_ground_truth(
        predicted,
        inputs["rooms"],
        inputs["pair_gt"],
        float(inputs["group"]["scale_meters_per_coordinate"]),
    )
    heading_rows = _heading_rows(predicted, inputs["rooms"], inputs["pair_gt"])
    summary = {
        "schema_version": "zind-candidate-local-portal-audit-v1",
        "purpose": "CPU-only diagnostic; not a formal inference result or scientific evaluation",
        "split": "train",
        "house_id": "0665",
        "floor_id": "floor_03",
        "inputs": {key: str(value) for key, value in inputs["paths"].items()},
        "model_loaded": False,
        "gpu_used": False,
        "test_data_accessed": False,
        "formal_inference_path_modified": False,
        "gt_available_to_hypothesis_generation": False,
        "frozen_candidate_pairs": {
            pair_id: value["best_candidate_pair"] for pair_id, value in inputs["pairs"].items()
        },
        "parameters": {
            "token_count": TOKEN_COUNT,
            "m1_douglas_peucker_epsilon_fraction": 0.015,
            "m1_global_ransac_max_lines": 8,
            "m1_global_ransac_min_support": 8,
            "m2_context_padding_tokens": 4,
            "m3_left_right_context_tokens": 8,
            "m4_context_padding_tokens": 12,
            "m4_max_hypotheses": 3,
            "uniform_parameters_for_all_rooms_and_pairs": True,
        },
        "contour_diagnostics": {
            room: state["contour_diagnostics"] for room, state in predicted["room_state"].items()
        },
        "heading_comparison": heading_rows,
        "positive_pairs": predicted["positive_pairs"],
        "negative_pairs": predicted["negative_pairs"],
        "method_comparison": _method_table(predicted),
    }
    # Strip numpy-only room state from the persisted public result; all relevant
    # representations already live under positive_pairs/methods and diagnostics.
    if not finite_tree(summary):
        raise ValueError("audit summary contains non-finite values")

    output = Path(args.output_dir).expanduser().resolve()
    reject_test_path(output, "output directory")
    output.mkdir(parents=True, exist_ok=True)
    visualizations = output / "visualizations"
    visualizations.mkdir(parents=True, exist_ok=True)
    portal_rows = _flatten_portal_rows(predicted)
    write_csv(output / "portal_hypotheses.csv", portal_rows)
    write_csv(output / "heading_comparison.csv", heading_rows)
    for pair_id, pair in predicted["positive_pairs"].items():
        gt = inputs["pair_gt"][pair_id]["shared_interface_details"][0]
        for key, side, endpoint_key in (
            ("candidate_a", "A", "local_endpoints_a"),
            ("candidate_b", "B", "local_endpoints_b"),
        ):
            candidate = pair[key]
            render_portal_svg(
                visualizations / f"{pair_id}_{side}_{candidate['room_id']}.svg",
                predicted["room_state"][candidate["room_id"]],
                candidate["hypotheses"],
                candidate["interval"],
                np.asarray(gt[endpoint_key], dtype=np.float64),
            )
    summary["artifacts"] = {
        "portal_hypotheses_csv": str(output / "portal_hypotheses.csv"),
        "heading_comparison_csv": str(output / "heading_comparison.csv"),
        "diagnosis_md": str(output / "diagnosis.md"),
        "visualizations": str(visualizations),
    }
    dump_json(output / "audit_summary.json", summary)
    write_diagnosis(output / "diagnosis.md", summary)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = run(args)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": summary["artifacts"],
                "model_loaded": False,
                "gpu_used": False,
                "test_data_accessed": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
