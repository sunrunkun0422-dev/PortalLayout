"""Predicted-only helpers for candidate-conditioned portal verification.

The functions in this module never inspect a ZInD label.  Ground-truth target
mapping is kept in the preflight orchestration tool so it cannot accidentally
enter a verifier feature vector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import LineString, Polygon

from portallayout.models.cross_scene_matcher import circular_interval_iou


FORBIDDEN_FEATURE_TOKENS = (
    "gt", "ground_truth", "target_candidate", "target_index", "label",
    "adjacent", "is_match", "pose_error", "global_endpoint",
)


@dataclass(frozen=True)
class RankedCandidate:
    candidate_a: int
    candidate_b: int
    score: float
    rank: int


def stable_topn(score_matrix: Sequence[Sequence[float]], count: int) -> List[RankedCandidate]:
    """Rank real-real candidates by score with a fixed row/column tie break."""
    if count <= 0:
        raise ValueError("Top-N count must be positive")
    value = np.asarray(score_matrix, dtype=np.float64)
    if value.ndim != 2 or value.size == 0 or not np.isfinite(value).all():
        raise ValueError("candidate score matrix must be non-empty and finite")
    ranked = sorted(
        ((float(value[a, b]), int(a), int(b)) for a in range(value.shape[0]) for b in range(value.shape[1])),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    return [
        RankedCandidate(candidate_a=a, candidate_b=b, score=score, rank=index + 1)
        for index, (score, a, b) in enumerate(ranked[:count])
    ]


def pad_candidate_rows(
    rows: Sequence[Sequence[float]], maximum: int, feature_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pad one pair's candidates while preserving an explicit validity mask."""
    if maximum <= 0 or feature_dim <= 0:
        raise ValueError("padding dimensions must be positive")
    if len(rows) > maximum:
        raise ValueError("candidate rows exceed the fixed maximum")
    output = np.zeros((maximum, feature_dim), dtype=np.float32)
    mask = np.zeros(maximum, dtype=bool)
    for index, row in enumerate(rows):
        value = np.asarray(row, dtype=np.float32)
        if value.shape != (feature_dim,) or not np.isfinite(value).all():
            raise ValueError("candidate feature row is malformed or non-finite")
        output[index] = value
        mask[index] = True
    return output, mask


def manifest_target_pair(
    target_interval_a: Sequence[int],
    target_interval_b: Sequence[int],
    predicted_intervals_a: Sequence[Sequence[int]],
    predicted_intervals_b: Sequence[Sequence[int]],
    token_count: int = 256,
    minimum_iou: float = 0.30,
) -> Tuple[Optional[Tuple[int, int]], Dict[str, Any]]:
    """Map the manifest's single supervised portal into predicted candidates.

    Multiple predicted intervals can overlap one target.  Exactly one label is
    produced by maximizing the two-sided IoU product, then using stable indices.
    """
    matches_a = [
        (index, circular_interval_iou(tuple(interval), tuple(target_interval_a), token_count))
        for index, interval in enumerate(predicted_intervals_a)
    ]
    matches_b = [
        (index, circular_interval_iou(tuple(interval), tuple(target_interval_b), token_count))
        for index, interval in enumerate(predicted_intervals_b)
    ]
    matches_a = [(index, iou) for index, iou in matches_a if iou >= minimum_iou]
    matches_b = [(index, iou) for index, iou in matches_b if iou >= minimum_iou]
    combinations = [
        (iou_a * iou_b, index_a, index_b, iou_a, iou_b)
        for index_a, iou_a in matches_a for index_b, iou_b in matches_b
    ]
    if not combinations:
        return None, {
            "mapped": False, "candidate_matches_A": len(matches_a),
            "candidate_matches_B": len(matches_b), "mapping_count": 0,
        }
    best = min(combinations, key=lambda item: (-item[0], item[1], item[2]))
    return (best[1], best[2]), {
        "mapped": True, "candidate_matches_A": len(matches_a),
        "candidate_matches_B": len(matches_b), "mapping_count": len(combinations),
        "selected_iou_A": float(best[3]), "selected_iou_B": float(best[4]),
        "selection": "max two-sided IoU product; stable A/B index tie-break",
    }


def circular_interval_width(interval: Sequence[int], token_count: int = 256) -> int:
    if len(interval) != 2 or token_count <= 0:
        raise ValueError("interval must contain two endpoints")
    start, end = int(interval[0]) % token_count, int(interval[1]) % token_count
    return end - start + 1 if start <= end else token_count - start + end + 1


def interval_mean(values: Sequence[float], interval: Sequence[int]) -> float:
    probability = np.asarray(values, dtype=np.float64).reshape(-1)
    width = circular_interval_width(interval, len(probability))
    start = int(interval[0]) % len(probability)
    indices = (start + np.arange(width, dtype=np.int64)) % len(probability)
    result = float(np.mean(probability[indices]))
    if not math.isfinite(result):
        raise ValueError("interval confidence is non-finite")
    return result


def matcher_cell_features(
    assignment_logits: Sequence[Sequence[float]],
    count_a: int,
    count_b: int,
    candidate_a: int,
    candidate_b: int,
    dustbin_logit: float,
) -> Dict[str, float]:
    """Reconstruct bidirectional affinity from real logits and frozen dustbin.

    Some historical batched artifacts placed the padded dustbin at the batch
    maximum rather than the sample-local index.  The real-real block is exact;
    the frozen scalar dustbin parameter makes reconstruction unambiguous.
    """
    matrix = np.asarray(assignment_logits, dtype=np.float64)
    real = matrix[:count_a, :count_b]
    if real.shape != (count_a, count_b) or not np.isfinite(real).all():
        raise ValueError("real-real assignment logits are unavailable or non-finite")
    if not (0 <= candidate_a < count_a and 0 <= candidate_b < count_b):
        raise ValueError("candidate index is outside real-real logits")
    if not math.isfinite(float(dustbin_logit)):
        raise ValueError("dustbin logit must be finite")

    def probability(logits: np.ndarray, selected: int) -> float:
        values = np.concatenate((logits, np.asarray([dustbin_logit], dtype=np.float64)))
        values -= float(values.max())
        weights = np.exp(values)
        return float(weights[selected] / weights.sum())

    affinity_ab = probability(real[candidate_a], candidate_b)
    affinity_ba = probability(real[:, candidate_b], candidate_a)
    return {
        "mutual_affinity": math.sqrt(max(0.0, affinity_ab * affinity_ba)),
        "assignment_probability_A_to_B": affinity_ab,
        "assignment_probability_B_to_A": affinity_ba,
        "row_candidate_to_dustbin_margin": float(real[candidate_a, candidate_b] - dustbin_logit),
        "column_candidate_to_dustbin_margin": float(real[candidate_a, candidate_b] - dustbin_logit),
        "real_real_logit": float(real[candidate_a, candidate_b]),
    }


def transform_points(points: Sequence[Sequence[float]], matrix: Sequence[Sequence[float]]) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    transform = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2 or transform.shape != (3, 3):
        raise ValueError("points must be [N,2] and transform must be [3,3]")
    homogeneous = np.concatenate((value, np.ones((len(value), 1), dtype=np.float64)), axis=1)
    output = homogeneous @ transform.T
    if not np.isfinite(output).all() or np.any(np.abs(output[:, 2]) < 1e-12):
        raise ValueError("B_to_A transform produced invalid homogeneous points")
    return output[:, :2] / output[:, 2:3]


def yaw_scale_translation(matrix: Sequence[Sequence[float]]) -> Dict[str, float]:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise ValueError("relative transform must be finite [3,3]")
    scale = float(math.sqrt(max(0.0, value[0, 0] ** 2 + value[1, 0] ** 2)))
    if scale <= 1e-12:
        raise ValueError("relative scale is degenerate")
    yaw = math.atan2(float(value[1, 0] / scale), float(value[0, 0] / scale))
    return {
        "relative_yaw_sin": math.sin(yaw),
        "relative_yaw_cos": math.cos(yaw),
        "relative_scale": scale,
        "absolute_log_scale": abs(math.log(scale)),
        "translation_norm": float(np.linalg.norm(value[:2, 2])),
    }


def endpoint_consistency(
    endpoints_a: Sequence[Sequence[float]],
    endpoints_b: Sequence[Sequence[float]],
    matrix_b_to_a: Sequence[Sequence[float]],
    mapping: str,
) -> Dict[str, float]:
    a = np.asarray(endpoints_a, dtype=np.float64)
    b = transform_points(endpoints_b, matrix_b_to_a)
    if a.shape != (2, 2) or b.shape != (2, 2):
        raise ValueError("portal endpoints must be [2,2]")
    target = a if mapping == "same" else a[::-1]
    delta = b - target
    tangent_a = a[1] - a[0]
    tangent_b = b[1] - b[0]
    norm_a, norm_b = np.linalg.norm(tangent_a), np.linalg.norm(tangent_b)
    if min(norm_a, norm_b) <= 1e-12:
        raise ValueError("portal tangent is degenerate")
    cosine = float(np.clip(np.dot(tangent_a, tangent_b) / (norm_a * norm_b), -1.0, 1.0))
    return {
        "endpoint_alignment_residual": float(math.sqrt(np.mean(np.square(delta)))),
        "midpoint_alignment_residual": float(np.linalg.norm(b.mean(axis=0) - a.mean(axis=0))),
        "portal_tangent_angle_degrees": float(math.degrees(math.acos(cosine))),
    }


def predicted_polygon_features(
    polygon_a: Sequence[Sequence[float]],
    polygon_b_in_a: Sequence[Sequence[float]],
    portal_endpoints_a: Sequence[Sequence[float]],
) -> Tuple[Dict[str, float], Optional[str]]:
    """Extract fixed scalar BEV features without repairing invalid polygons."""
    points_a = np.asarray(polygon_a, dtype=np.float64)
    points_b = np.asarray(polygon_b_in_a, dtype=np.float64)
    portal = np.asarray(portal_endpoints_a, dtype=np.float64)
    if points_a.ndim != 2 or points_b.ndim != 2 or points_a.shape[1:] != (2,) or points_b.shape[1:] != (2,):
        return {}, "polygon_shape_invalid"
    if len(points_a) < 3 or len(points_b) < 3 or portal.shape != (2, 2):
        return {}, "polygon_or_portal_too_short"
    if not np.isfinite(points_a).all() or not np.isfinite(points_b).all() or not np.isfinite(portal).all():
        return {}, "polygon_or_portal_nonfinite"
    shape_a, shape_b = Polygon(points_a), Polygon(points_b)
    if not shape_a.is_valid or not shape_b.is_valid or shape_a.area <= 1e-12 or shape_b.area <= 1e-12:
        return {}, "predicted_polygon_invalid"
    intersection = shape_a.intersection(shape_b).area
    union = shape_a.union(shape_b).area
    centroid_distance = shape_a.centroid.distance(shape_b.centroid)
    portal_width = float(np.linalg.norm(portal[1] - portal[0]))
    local_radius = max(1e-6, 0.25 * portal_width)
    portal_region = LineString(portal).buffer(local_radius, cap_style=2)
    local_overlap = shape_a.intersection(shape_b).intersection(portal_region).area
    min_x, min_y, max_x, max_y = shape_a.union(shape_b).bounds
    box_width, box_height = max_x - min_x, max_y - min_y
    tangent = portal[1] - portal[0]
    midpoint = portal.mean(axis=0)
    side_a = float(np.cross(tangent, np.asarray(shape_a.centroid.coords[0]) - midpoint))
    side_b = float(np.cross(tangent, np.asarray(shape_b.centroid.coords[0]) - midpoint))
    area_a, area_b = float(shape_a.area), float(shape_b.area)
    normalizer = math.sqrt(max(union, 1e-12))
    result = {
        "polygon_A_valid": 1.0,
        "polygon_B_transformed_valid": 1.0,
        "intersection_area": float(intersection),
        "intersection_over_A": float(intersection / area_a),
        "intersection_over_B": float(intersection / area_b),
        "union_area": float(union),
        "polygon_iou": float(intersection / max(union, 1e-12)),
        "centroid_distance": float(centroid_distance),
        "normalized_centroid_distance": float(centroid_distance / normalizer),
        "containment_ratio": float(intersection / min(area_a, area_b)),
        "boundary_minimum_distance": float(shape_a.boundary.distance(shape_b.boundary)),
        "portal_local_overlap": float(local_overlap / max(portal_region.area, 1e-12)),
        "portal_centroid_side_product": float(side_a * side_b / max(portal_width ** 2 * union, 1e-12)),
        "portal_centroids_opposite_sides": float(side_a * side_b < 0.0),
        "scaled_area_ratio": float(max(area_a, area_b) / min(area_a, area_b)),
        "combined_bbox_aspect": float(max(box_width, box_height) / max(min(box_width, box_height), 1e-12)),
        "abnormal_penetration": float(intersection / min(area_a, area_b) > 0.50),
        "complete_separation": float(shape_a.boundary.distance(shape_b.boundary) > max(0.25 * portal_width, 1e-6)),
    }
    if not all(math.isfinite(value) for value in result.values()):
        return {}, "polygon_feature_nonfinite"
    return result, None


def feature_names_are_predicted_only(names: Iterable[str]) -> bool:
    for name in names:
        lowered = str(name).lower()
        if any(token in lowered for token in FORBIDDEN_FEATURE_TOKENS):
            return False
    return True
