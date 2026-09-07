"""Predicted-only candidate-local portal geometry.

This module turns one predicted opening interval and a panorama-local raw
depth contour into explicit portal endpoints.  It deliberately has no ground
truth inputs and does not decide whether two candidates match.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class CandidateLocalPortalConfig:
    """Frozen M2 parameters from the CPU portal geometry audit."""

    context_padding_tokens: int = 4
    ransac_min_inliers: int = 4
    probability_weight_floor: float = 0.25
    probability_weight_span: float = 0.75
    ransac_radius_max_fraction: float = 0.06
    ransac_radius_min_fraction: float = 0.008
    ransac_median_edge_multiplier: float = 2.5
    ray_parallel_epsilon: float = 1e-8
    width_ratio_min: float = 0.03
    width_ratio_max: float = 2.5
    quality_min_support: int = 4
    quality_min_inlier_ratio: float = 0.60
    quality_max_normalized_residual: float = 0.05
    quality_max_angular_dispersion_degrees: float = 25.0
    quality_min_ray_condition: float = 0.05

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LineFit:
    tangent: np.ndarray
    normal: np.ndarray
    offset: float
    residuals: np.ndarray
    inliers: np.ndarray
    singular_values: np.ndarray
    source: str


@dataclass(frozen=True)
class PortalGeometryHypothesis:
    method: str
    variant: str
    interval: Tuple[int, int]
    support_indices: Tuple[int, ...]
    support_points_local_bi_xz: Optional[List[List[float]]]
    line: Optional[Dict[str, Any]]
    endpoints_local_bi_xz: Optional[List[List[float]]]
    endpoints_local_zind_xy: Optional[List[List[float]]]
    intersection: Dict[str, Any]
    quality: Dict[str, Any]
    geometry_valid: bool
    failure_reason: Optional[str]
    selected_predicted_only: bool

    def to_audit_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "variant": self.variant,
            "fit_source": None if self.line is None else self.line["fit_source"],
            "line": None if self.line is None else {
                "tangent_bi_xz": self.line["tangent_bi_xz"],
                "normal_bi_xz": self.line["normal_bi_xz"],
                "offset_bi_xz": self.line["offset_bi_xz"],
            },
            "endpoints_bi_xz": self.endpoints_local_bi_xz,
            "endpoints_zind_local_xy": self.endpoints_local_zind_xy,
            "intersection": self.intersection,
            "quality": self.quality,
            "context": None,
            "valid": self.geometry_valid,
            "failure_reason": self.failure_reason,
            "selected_predicted_only": self.selected_predicted_only,
        }

    def to_json(self) -> Dict[str, Any]:
        value = self.to_audit_dict()
        value.update({
            "frame": "panorama_local",
            "input_coordinate_system": "Bi panorama-local x/z",
            "output_coordinate_system": "ZInD panorama-local x/y",
            "token_interval": list(self.interval),
            "support_indices": list(self.support_indices),
            "local_fitting_points_bi_xz": self.support_points_local_bi_xz,
            "geometry_valid": self.geometry_valid,
        })
        return value


@dataclass(frozen=True)
class PortalGeometryResult:
    method: str
    frame: str
    input_coordinate_system: str
    output_coordinate_system: str
    token_count: int
    token_interval: Tuple[int, int]
    config: CandidateLocalPortalConfig
    hypotheses: Tuple[PortalGeometryHypothesis, ...]
    primary_variant: Optional[str]
    geometry_valid: bool
    failure_reason: Optional[str]

    @property
    def primary(self) -> Optional[PortalGeometryHypothesis]:
        selected = [value for value in self.hypotheses if value.selected_predicted_only]
        if len(selected) > 1:
            raise RuntimeError("multiple M2 hypotheses are marked primary")
        return selected[0] if selected else None

    def to_json(self) -> Dict[str, Any]:
        primary = self.primary
        quality = {} if primary is None else dict(primary.quality)
        line = None if primary is None else primary.line
        return {
            "method": self.method,
            "frame": self.frame,
            "input_coordinate_system": self.input_coordinate_system,
            "output_coordinate_system": self.output_coordinate_system,
            "token_count": self.token_count,
            "token_interval": list(self.token_interval),
            "config": self.config.to_json(),
            "primary_variant": self.primary_variant,
            "geometry_valid": self.geometry_valid,
            "failure_reason": self.failure_reason,
            "endpoints_local": None if primary is None else primary.endpoints_local_zind_xy,
            "endpoints_local_bi_xz": None if primary is None else primary.endpoints_local_bi_xz,
            "wall_tangent": None if line is None else line["tangent_zind_xy"],
            "wall_normal": None if line is None else line["normal_zind_xy"],
            "wall_line": line,
            "opening_width": quality.get("opening_width"),
            "fit_residual": quality.get("fit_residual"),
            "normalized_residual": quality.get("normalized_residual"),
            "inlier_ratio": quality.get("inlier_ratio"),
            "support_count": quality.get("support_point_count"),
            "angular_dispersion": quality.get("angular_dispersion_degrees"),
            "ray_condition": quality.get("ray_intersection_condition"),
            "width_plausibility": quality.get("width_plausible"),
            "predicted_only_quality": quality,
            "local_fitting_points": (
                None if primary is None else primary.support_points_local_bi_xz
            ),
            "hypotheses": [value.to_json() for value in self.hypotheses],
            "gt_available_to_geometry": False,
        }


def circular_indices(start: int, end: int, count: int = 256) -> np.ndarray:
    if count <= 0:
        raise ValueError("token_count must be positive")
    start, end = int(start) % count, int(end) % count
    if start <= end:
        return np.arange(start, end + 1, dtype=np.int64)
    return np.concatenate(
        [np.arange(start, count, dtype=np.int64), np.arange(0, end + 1, dtype=np.int64)]
    )


def expanded_indices(start: int, end: int, padding: int, count: int = 256) -> np.ndarray:
    interval = circular_indices(start, end, count)
    left = (
        np.arange(start - padding, start, dtype=np.int64) % count
        if padding
        else np.empty(0, dtype=np.int64)
    )
    right = (
        np.arange(end + 1, end + padding + 1, dtype=np.int64) % count
        if padding
        else np.empty(0, dtype=np.int64)
    )
    return np.concatenate([left, interval, right])


def panorama_ray(token: int, count: int = 256) -> np.ndarray:
    longitude = ((int(token) % count + 0.5) / float(count) - 0.5) * 2.0 * math.pi
    return np.asarray([math.sin(longitude), math.cos(longitude)], dtype=np.float64)


def bi_to_zind(points: Sequence[Sequence[float]]) -> np.ndarray:
    """Reflect Bi x/z into the ZInD panorama-local x/y convention."""
    value = np.asarray(points, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2 or not np.isfinite(value).all():
        raise ValueError("points must be finite [N, 2]")
    output = value.copy()
    output[:, 0] *= -1.0
    return output


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
    _, singular, vh = np.linalg.svd(
        centered * np.sqrt(weight[:, None]), full_matrices=False
    )
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
    weight = (
        np.ones(len(value), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if weight.shape != (len(value),) or not np.isfinite(weight).all() or np.any(weight < 0):
        raise ValueError("RANSAC weights must align with points")
    best = None
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
            span = float(
                np.linalg.norm(value[inliers].max(axis=0) - value[inliers].min(axis=0))
            )
            rmse = float(
                math.sqrt(np.average(np.square(residuals[inliers]), weights=weight[inliers]))
            )
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
    count: int = 256,
    parallel_epsilon: float = 1e-8,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if len(interval) != 2:
        raise ValueError("interval must contain start and end")
    rays = np.stack(
        [panorama_ray(int(interval[0]), count), panorama_ray(int(interval[1]), count)]
    )
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


def quality_gate(
    quality: Mapping[str, Any],
    config: CandidateLocalPortalConfig = CandidateLocalPortalConfig(),
) -> bool:
    """Soft predicted-only quality gate; it is not a production rejection gate."""
    return bool(
        int(quality["support_point_count"]) >= config.quality_min_support
        and float(quality["inlier_ratio"]) >= config.quality_min_inlier_ratio
        and float(quality["normalized_residual"])
        <= config.quality_max_normalized_residual
        and float(quality["angular_dispersion_degrees"])
        <= config.quality_max_angular_dispersion_degrees
        and float(quality["ray_intersection_condition"])
        >= config.quality_min_ray_condition
        and bool(quality["width_plausible"])
        and quality.get("left_right_context_agreement") is not False
        and not bool(quality["corner_ambiguity"])
    )


def predicted_quality(
    fit: LineFit,
    support_points: np.ndarray,
    endpoints_bi: np.ndarray,
    intersection: Mapping[str, Any],
    room_radius: float,
    context_agreement: Optional[bool],
    corner_ambiguity: bool,
    config: CandidateLocalPortalConfig = CandidateLocalPortalConfig(),
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
    width_plausible = bool(
        config.width_ratio_min <= width_ratio <= config.width_ratio_max
    )
    angular_dispersion = (
        _axis_dispersion(support_points[inliers], fit.tangent)
        if np.any(inliers)
        else 90.0
    )
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
    quality["quality_gate_pass"] = quality_gate(quality, config)
    return quality


def _line_json(fit: LineFit) -> Dict[str, Any]:
    tangent_zind = np.asarray([-fit.tangent[0], fit.tangent[1]], dtype=np.float64)
    normal_zind = np.asarray([-fit.normal[0], fit.normal[1]], dtype=np.float64)
    return {
        "fit_source": fit.source,
        "tangent_bi_xz": fit.tangent.tolist(),
        "normal_bi_xz": fit.normal.tolist(),
        "offset_bi_xz": float(fit.offset),
        "tangent_zind_xy": tangent_zind.tolist(),
        "normal_zind_xy": normal_zind.tolist(),
        "offset_zind_xy": float(fit.offset),
    }


def _failed_hypothesis(
    variant: str,
    interval: Tuple[int, int],
    support_indices: np.ndarray,
    support_points: Optional[np.ndarray],
    reason: str,
) -> PortalGeometryHypothesis:
    return PortalGeometryHypothesis(
        method="M2",
        variant=variant,
        interval=interval,
        support_indices=tuple(int(value) for value in support_indices),
        support_points_local_bi_xz=(
            None if support_points is None else support_points.tolist()
        ),
        line=None,
        endpoints_local_bi_xz=None,
        endpoints_local_zind_xy=None,
        intersection={
            "ray_denominators": [None, None],
            "ray_intersection_condition": 0.0,
            "ray_distances": [None, None],
        },
        quality={
            "fit_residual": None,
            "normalized_residual": None,
            "inlier_ratio": 0.0,
            "support_point_count": int(len(support_indices)),
            "inlier_count": 0,
            "angular_dispersion_degrees": None,
            "line_condition_number": None,
            "ray_intersection_condition": 0.0,
            "opening_width": None,
            "opening_width_over_room_radius": None,
            "width_plausible": False,
            "left_right_context_agreement": None,
            "corner_ambiguity": False,
            "reliability_score": 0.0,
            "quality_gate_pass": False,
        },
        geometry_valid=False,
        failure_reason=reason,
        selected_predicted_only=False,
    )


def _hypothesis_from_fit(
    variant: str,
    fit: LineFit,
    interval: Tuple[int, int],
    support_indices: np.ndarray,
    support_points: np.ndarray,
    room_radius: float,
    token_count: int,
    config: CandidateLocalPortalConfig,
) -> PortalGeometryHypothesis:
    try:
        endpoints_bi, intersection = line_endpoint_intersections(
            fit,
            interval,
            count=token_count,
            parallel_epsilon=config.ray_parallel_epsilon,
        )
        endpoints_zind = bi_to_zind(endpoints_bi)
        quality = predicted_quality(
            fit,
            support_points,
            endpoints_bi,
            intersection,
            room_radius,
            context_agreement=None,
            corner_ambiguity=False,
            config=config,
        )
        return PortalGeometryHypothesis(
            method="M2",
            variant=variant,
            interval=interval,
            support_indices=tuple(int(value) for value in support_indices),
            support_points_local_bi_xz=support_points.tolist(),
            line=_line_json(fit),
            endpoints_local_bi_xz=endpoints_bi.tolist(),
            endpoints_local_zind_xy=endpoints_zind.tolist(),
            intersection=dict(intersection),
            quality=quality,
            geometry_valid=True,
            failure_reason=None,
            selected_predicted_only=False,
        )
    except Exception as exc:
        failed = _failed_hypothesis(
            variant, interval, support_indices, support_points, str(exc)
        )
        return PortalGeometryHypothesis(
            **{
                **failed.__dict__,
                "line": _line_json(fit),
                "quality": {
                    **failed.quality,
                    "inlier_ratio": float(np.mean(fit.inliers)),
                    "inlier_count": int(np.count_nonzero(fit.inliers)),
                },
            }
        )


def _select_primary(
    hypotheses: Sequence[PortalGeometryHypothesis],
) -> Tuple[PortalGeometryHypothesis, ...]:
    if not hypotheses:
        return tuple()
    selected = max(
        hypotheses,
        key=lambda value: (value.quality["reliability_score"], value.variant),
    )
    return tuple(
        PortalGeometryHypothesis(
            **{**value.__dict__, "selected_predicted_only": value is selected}
        )
        for value in hypotheses
    )


def estimate_candidate_local_portal_geometry(
    contour_local: Sequence[Sequence[float]],
    token_interval: Sequence[int],
    token_count: int,
    opening_probability: Sequence[float],
    config: CandidateLocalPortalConfig = CandidateLocalPortalConfig(),
) -> PortalGeometryResult:
    """Estimate the frozen M2 primary without consulting GT or matcher scores."""

    interval = (-1, -1)
    output_token_count = -1
    hypotheses: List[PortalGeometryHypothesis] = []
    try:
        output_token_count = int(token_count)
        if token_interval is None:
            raise ValueError("token_interval must contain start and end")
        if len(token_interval) != 2:
            raise ValueError("token_interval must contain start and end")
        interval = (int(token_interval[0]), int(token_interval[1]))
        if output_token_count <= 2:
            raise ValueError("token_count must be greater than two")
        contour = np.asarray(contour_local, dtype=np.float64)
        probability = np.asarray(opening_probability, dtype=np.float64).reshape(-1)
        if contour.shape != (output_token_count, 2):
            raise ValueError("contour_local must have shape [token_count, 2]")
        if not np.isfinite(contour).all():
            raise ValueError("contour_local contains non-finite values")
        if probability.shape != (output_token_count,) or not np.isfinite(probability).all():
            raise ValueError("opening_probability must be finite [token_count]")
        if np.any(probability < 0.0) or np.any(probability > 1.0):
            raise ValueError("opening_probability must be in [0, 1]")
        room_radius = float(np.median(np.linalg.norm(contour, axis=1)))
        if not math.isfinite(room_radius) or room_radius <= 1e-10:
            raise ValueError("contour_local has invalid panorama radius")
        adjacent = np.linalg.norm(np.roll(contour, -1, axis=0) - contour, axis=1)
        median_adjacent = float(np.median(adjacent))
        threshold = min(
            config.ransac_radius_max_fraction * room_radius,
            max(
                config.ransac_radius_min_fraction * room_radius,
                config.ransac_median_edge_multiplier * median_adjacent,
            ),
        )
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("derived RANSAC threshold is invalid")

        start, end = interval
        interval_ids = circular_indices(start, end, output_token_count)
        interval_points = contour[interval_ids]
        interval_weights = (
            config.probability_weight_floor
            + config.probability_weight_span * probability[interval_ids]
        )
        try:
            tls = fit_line_tls(
                interval_points, interval_weights, source="opening_weighted_tls"
            )
            hypotheses.append(
                _hypothesis_from_fit(
                    "weighted_tls",
                    tls,
                    interval,
                    interval_ids,
                    interval_points,
                    room_radius,
                    output_token_count,
                    config,
                )
            )
        except Exception as exc:
            hypotheses.append(
                _failed_hypothesis(
                    "weighted_tls", interval, interval_ids, interval_points, str(exc)
                )
            )

        expanded_ids = expanded_indices(
            start, end, config.context_padding_tokens, output_token_count
        )
        expanded_points = contour[expanded_ids]
        expanded_weights = (
            config.probability_weight_floor
            + config.probability_weight_span * probability[expanded_ids]
        )
        try:
            ransac = ransac_line(
                expanded_points,
                threshold=threshold,
                weights=expanded_weights,
                min_inliers=config.ransac_min_inliers,
                source="opening_context_ransac_tls",
            )
            hypotheses.append(
                _hypothesis_from_fit(
                    "ransac",
                    ransac,
                    interval,
                    expanded_ids,
                    expanded_points,
                    room_radius,
                    output_token_count,
                    config,
                )
            )
        except Exception as exc:
            hypotheses.append(
                _failed_hypothesis(
                    "ransac", interval, expanded_ids, expanded_points, str(exc)
                )
            )
        selected = _select_primary(hypotheses)
        primary = next(value for value in selected if value.selected_predicted_only)
        valid = bool(primary.geometry_valid)
        failure = None if valid else primary.failure_reason
    except Exception as exc:
        selected = tuple(hypotheses)
        primary = None
        valid = False
        failure = str(exc)

    return PortalGeometryResult(
        method="candidate_local_m2",
        frame="panorama_local",
        input_coordinate_system="Bi panorama-local x/z",
        output_coordinate_system="ZInD panorama-local x/y",
        token_count=output_token_count,
        token_interval=interval,
        config=config,
        hypotheses=selected,
        primary_variant=None if primary is None else primary.variant,
        geometry_valid=valid,
        failure_reason=failure,
    )
